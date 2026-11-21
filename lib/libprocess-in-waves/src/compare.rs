use crate::db::WavesDb;
use crate::context::ProcessContext;
use crate::error::ProcessError;
use crate::league::MaxPriorityQueueByScore;

pub struct ComparisonsCache {
    pub pairs: Vec<(String, String, f64)>,
}

impl ComparisonsCache {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }
}

#[tracing::instrument(
    name = "run_comparison_wave",
    skip(
        ctx,
        to_compare,
        league,
        wave_size,
        _jobad_hash,
        deep_model_id,
        comparator,
        comparisons_cache,
        _wave_size_comparison
    )
)]
pub fn run_comparison_wave(
    ctx: &mut ProcessContext,
    to_compare: &[String],
    league: &mut MaxPriorityQueueByScore,
    wave_size: usize,
    _jobad_hash: &str,
    deep_model_id: &str,
    comparator: &libcomparator::ComparatorArtifact,
    comparisons_cache: &mut ComparisonsCache,
    _wave_size_comparison: usize,
) -> Result<(), ProcessError> {
    let model = libcomparator::ComparatorModel::from_artifact(comparator)
        .map_err(|e| ProcessError::Comparator(e))?;
    let spec = comparator.feature_spec.clone();
    let n = wave_size.min(to_compare.len());
    for i in 0..n {
        let cv_new = to_compare[i].clone();
        if league.contains(&cv_new) {
            continue;
        }
        let mut new_league = MaxPriorityQueueByScore::new();
        while !league.is_empty() {
            let cv_old = match league.pop() {
                Some(x) => x,
                None => break,
            };
            // load vectors
            let old_bytes = ctx
                .db
                .get_embedding(&cv_old, deep_model_id)?
                .unwrap_or_default();
            let new_bytes = ctx
                .db
                .get_embedding(&cv_new, deep_model_id)?
                .unwrap_or_default();
            let job_text = ctx.job.job_ad_text.clone();
            // job embedding not stored; embed jobad
            ctx.runtime.ensure_started();
            let _ = ctx
                .runtime
                .wait_until_ready(std::time::Duration::from_secs(120));
            let embedder = ctx
                .runtime
                .embedder_for(deep_model_id)
                .map_err(|e| ProcessError::Runtime(e.to_string()))?;
            let job_vecs = embedder
                .embed(&[job_text.as_str()])
                .map_err(|e| ProcessError::Runtime(e.to_string()))?;
            let job_vec = job_vecs.into_iter().next().unwrap_or_default();
            let old_vec =
                libtriqui_core::embed::bytes_to_vector(&old_bytes, None).unwrap_or_default();
            let new_vec =
                libtriqui_core::embed::bytes_to_vector(&new_bytes, None).unwrap_or_default();
            // convert to f64 chunks
            let job_chunks: Vec<Vec<f64>> = vec![job_vec.iter().map(|x| *x as f64).collect()];
            let old_chunks: Vec<Vec<f64>> = vec![old_vec.iter().map(|x| *x as f64).collect()];
            let new_chunks: Vec<Vec<f64>> = vec![new_vec.iter().map(|x| *x as f64).collect()];
            let job_df = libcomparator::features::doc_features(&job_chunks, &job_chunks)
                .map_err(|e| ProcessError::Comparator(e))?;
            let old_df = libcomparator::features::doc_features(&job_chunks, &old_chunks)
                .map_err(|e| ProcessError::Comparator(e))?;
            let new_df = libcomparator::features::doc_features(&job_chunks, &new_chunks)
                .map_err(|e| ProcessError::Comparator(e))?;
            let x = libcomparator::features::assemble_pair(&spec, &job_df, &old_df, &new_df);
            let prob = model.pair_prob(x);
            comparisons_cache
                .pairs
                .push((cv_old.clone(), cv_new.clone(), prob));
            ctx.db
                .insert_comparison(&ctx.job.id, &cv_old, &cv_new, prob)?;
            // rescore
            ctx.db.rescore_cv(&cv_old)?;
            ctx.db.rescore_cv(&cv_new)?;
            let old_score = ctx.db.get_cv_score(&cv_old)?;
            let new_score = ctx.db.get_cv_score(&cv_new)?;
            new_league.push(cv_old, old_score);
            new_league.push(cv_new.clone(), new_score);
        }
        *league = new_league;
    }
    Ok(())
}
