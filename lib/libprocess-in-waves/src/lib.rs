pub mod context;
pub mod db;
pub mod scan;
pub mod embed;
pub mod similarity;
pub mod league;
pub mod compare;
pub mod error;

pub use context::{ProcessContext, ProcessConfig, JobConfig};
pub use db::DbHandle;
pub use db::open_connection_at;
pub use league::MaxPriorityQueueByScore;
pub use error::ProcessError;
pub use similarity::Target;

use crate::compare::ComparisonsCache;

pub fn process_in_waves(ctx: &mut context::ProcessContext, config: context::ProcessConfig) -> Result<(), error::ProcessError> {
    let _ = ctx.db.with_writer(|c| {
        c.execute(
            "INSERT OR IGNORE INTO jobs (id, title, job_ad_text, import_directory_path) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![ctx.job.id, ctx.job.title, ctx.job.job_ad_text, ctx.job.import_directory_path.as_ref().map(|p| p.to_string_lossy().to_string())],
        )?;
        Ok(())
    });

    let quick_model = ctx.job.quick_model.clone();
    let deep_model = ctx.job.deep_model.clone();
    let jobad_hash = if ctx.job.job_ad_hash.is_empty() { hash_text(&ctx.job.job_ad_text) } else { ctx.job.job_ad_hash.clone() };
    let targets: Vec<similarity::Target> = {
        let mut t = vec![similarity::Target { hash: jobad_hash.clone(), text: ctx.job.job_ad_text.clone() }];
        for c in &ctx.job.criteria {
            t.push(similarity::Target { hash: hash_text(c), text: c.clone() });
        }
        t
    };

    let mut league = league::MaxPriorityQueueByScore::new();
    let mut comparisons = ComparisonsCache::new();

    let mut wave: usize = 0;
    loop {
        if let Some(max) = config.max_waves {
            if wave >= max { break; }
        }
        let running: i64 = ctx.db.with_reader(|c| {
            Ok(c.query_row("SELECT running FROM waves_rank_state WHERE job_id = ?1", [ctx.job.id.as_str()], |r| r.get::<_, i64>(0)).unwrap_or(1))
        }).unwrap_or(1);
        if running == 0 { break; }

        scan::scan_watch_dir(ctx)?;

        let cv_hashes = ctx.db.get_all_cv_hashes()?;

        embed::ensure_quick_embeddings(ctx, &cv_hashes)?;

        similarity::compute_batch_similarity(ctx, &cv_hashes, &targets, &quick_model, ctx.job.similarity_method)?;

        let deep_queue = ctx.db.get_deep_queue(&quick_model, &deep_model, config.wave_size_deep)?;
        let slice: Vec<String> = deep_queue.into_iter().take(config.wave_size_deep).collect();
        embed::ensure_deep_embeddings(ctx, &slice, config.wave_size_deep)?;

        let cv_hashes_all = ctx.db.get_all_cv_hashes()?;
        similarity::compute_batch_similarity(ctx, &cv_hashes_all, &targets, &deep_model, ctx.job.similarity_method)?;

        let to_compare = ctx.db.get_cvs_for_comparison(&deep_model, &jobad_hash)?;

        if league.is_empty() {
            for h in &cv_hashes_all {
                if ctx.db.has_embedding(h, &deep_model)? {
                    let s = ctx.db.get_cv_score(h)?;
                    league.push(h.clone(), s);
                }
            }
        }

        let league_hashes: std::collections::HashSet<String> = league.iter().cloned().collect();
        let to_compare_filtered: Vec<String> = to_compare.into_iter().filter(|h| !league_hashes.contains(h)).collect();

        let artifact = ctx.comparator_artifact.clone();
        if let Some(artifact) = artifact {
            compare::run_comparison_wave(
                ctx,
                &to_compare_filtered,
                &mut league,
                config.wave_size_comparison,
                &jobad_hash,
                &deep_model,
                &artifact,
                &mut comparisons,
                config.wave_size_comparison,
            )?;
        } else {
            log::info!("No comparator artifact available, skipping comparison wave");
        }

        let candidates = cv_hashes_all.len() as i64;
        let embedded_done = cv_hashes_all.iter()
            .filter(|h| ctx.db.has_embedding(h, &deep_model).unwrap_or(false)).count() as i64;
        ctx.db.update_waves_rank_state(&ctx.job.id, wave as i64, candidates, embedded_done, to_compare_filtered.len() as i64, candidates, 0, None, true)?;

        wave += 1;
    }
    Ok(())
}

fn hash_text(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}
