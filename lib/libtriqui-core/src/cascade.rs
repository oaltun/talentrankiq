//! The cascade: quick, then waves of embed + comparator, projected by rescore.
//!
//! The whole orchestration, in one loop, with no budgets inside the scoring path
//! (`gui/design/app1/cascade-simple.md`):
//!
//! ```text
//! quick.run()                      # whole corpus, once — every candidate gets a total
//! ensure_vectors(deep)            # the rubric in the deep space
//! pool = freeze_pool(top P)        # frozen: comparator win-rates are set-relative
//! while not deep.has_all(job) or not comparator.has_all(job):
//!     slice = deep.next_slice(k1) # best-current-order first
//!     deep.embed(slice)           # → stage_scores(deep)
//!     comparator.run(pool)              # → candidate_scores[ad_comparator], memoized per pair
//!     rescore(slice)               # → points + rank + display rows
//! ```
//!
//! # The guarantees
//!
//! - **Convergence.** The loop ends when every candidate holds deep-space chunks
//!   and every pool member holds a comparator grade. At that point the sheet is
//!   exactly what a full blocking sort would produce, for the pool.
//! - **Monotone improvement.** A wave only ever replaces a candidate's grades
//!   with better ones (quick space → deep space; ad cosine → win-rate), and the
//!   pool is frozen, so the top of the sheet stabilises early and never
//!   regresses for reasons unrelated to the candidate.
//! - **A run never survives its inputs changing in a way that matters.** Embedding
//!   is ad-independent, so an input change does not stop a run — it re-projects.
//!   See [`crate::rescore`]; the caller decides when to re-project.

use std::sync::Arc;

use crate::comparator;
use crate::rescore;
use crate::rubric::{self, Rubric};
use crate::runtime::EngineRuntime;
use crate::quick;
use crate::types::DomainError;

/// Budget for one cascade wave: `wave1_budget` candidates the deep embedder
/// embeds this wave, `pool_size` candidates the comparator compares. Both come from
/// `cascade:` in `triq.yaml`.
#[derive(Clone, Copy, Debug)]
pub struct CascadeBudget {
    pub wave1_budget: usize,
    pub pool_size: usize,
}

impl Default for CascadeBudget {
    fn default() -> Self {
        CascadeBudget {
            wave1_budget: libconfigreader::DEFAULT_CASCADE_K1,
            pool_size: libconfigreader::DEFAULT_CASCADE_POOL,
        }
    }
}

/// What a wave did — the reported behaviour the gui and tests assert on (never
/// instrumentation: this is the contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveOutcome {
    /// 1-based wave number.
    pub wave: u32,
    /// Candidates the deep embedder embedded this wave.
    pub embedded: i64,
    /// Candidates that failed to embed this wave.
    pub embed_failed: i64,
    /// Candidates still awaiting the deep embedder after this wave.
    pub remaining_embed: i64,
    /// Whether the comparator ran (an artifact is configured and the ad and CVs clear
    /// its length floors) and had pool members to compare.
    pub comparator_ran: bool,
    /// Fresh comparator verdicts computed this wave.
    pub comparator_pairs_fresh: u32,
    /// Comparator verdicts served from the memoization cache.
    pub comparator_pairs_cached: u32,
    /// Pool members still without a comparator grade after this wave.
    pub remaining_comparator: i64,
    /// True when nothing is left to embed and the pool is fully comparatord.
    pub converged: bool,
}

/// Run stage 0.5: the quick pass over every candidate. Returns `Ok(None)` when no
/// quick is configured, or when the quick *is* the current embedder — a second
/// pass in the same space would be pure waste.
pub fn run_scout_stage(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    progress: crate::embeddings::EmbedProgress<'_>,
) -> Result<Option<quick::ScoutOutcome>, DomainError> {
    if quick::scout_model_for(&rt.model()).is_none() {
        return Ok(None);
    }
    Ok(Some(quick::run_scout(rt, job_id, progress)?))
}

/// The configured quick slug, or `None` when the stage is off.
pub fn scout_model(rt: &Arc<EngineRuntime>) -> Option<String> {
    quick::scout_model_for(&rt.model())
}

/// How many candidates still need the deep embedder.
pub fn remaining_embed_count(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
) -> Result<i64, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    rescore::count_needing_embed(&conn, job_id, &rt.model())
}

/// Ensure the rubric has vectors in the deep space, so the waves' scoring is
/// never waiting on an embed. Cheap and cache-first: after the first pass it is a
/// few indexed reads.
pub fn prepare_heavy_rubric(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<(), DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    let rubric = Rubric::load(&conn, job_id)?;
    rubric::ensure_vectors(rt, &conn, &rubric, &rt.model(), None)
        .map_err(|e| DomainError::DbError { message: e.to_string() })?;
    Ok(())
}

/// Freeze the comparator pool: the top `pool_size` candidates by current total.
///
/// Called once per run, after the quick pass has graded the corpus and the first
/// rescore has ranked it. Re-freezing is the *only* thing that changes the pool,
/// which is what keeps a win-rate meaning the same thing at two moments. An ad
/// change invalidates every memoized verdict, so `inputs_changed` clears the pool
/// and the next run re-freezes it.
pub fn freeze_pool(rt: &Arc<EngineRuntime>, job_id: &str, pool_size: usize) -> Result<usize, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    crate::db::freeze_comparator_pool(&conn, job_id, pool_size)
        .map_err(|e| DomainError::DbError { message: e.to_string() })
}

/// How many pool members are still uncomparatord.
pub fn remaining_comparator_count(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
) -> Result<i64, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    let ids = crate::db::comparator_pool_ids(&conn, job_id).map_err(|e| DomainError::DbError { message: e.to_string() })?;
    if ids.is_empty() {
        return Ok(0);
    }
    let holes = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT COUNT(*) FROM comparator_pool jp
         WHERE jp.job_id = ?1 AND NOT EXISTS (
             SELECT 1 FROM candidate_chunks cc
             WHERE cc.candidate_id = jp.candidate_id AND cc.vector_model = ?2)
             AND NOT EXISTS (
             SELECT 1 FROM candidate_scores cs
             WHERE cs.candidate_id = jp.candidate_id AND cs.question_id = ?3)
             AND jp.candidate_id IN ({holes})"
    );
    let model = rt.model();
    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&job_id, &model, &rescore::JUDGE_Q];
    params.extend(ids.iter().map(|i| i as &dyn rusqlite::ToSql));
    conn.query_row(
        &sql,
        rusqlite::params_from_iter(params),
        |r| r.get(0),
    )
    .map_err(|e| DomainError::DbError { message: e.to_string() })
}

/// Run one wave: embed the next slice in best-current-order, comparator whoever in the
/// pool is ready, and re-project.
///
/// The comparator runs *after* the embed, on whatever pool members the deep embedder
/// has reached — it cannot compare a CV it has no vectors for. Returns whether
/// anything remains, so the caller can loop.
pub fn run_wave(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    wave: u32,
    budget: CascadeBudget,
    progress: crate::embeddings::EmbedProgress<'_>,
) -> Result<WaveOutcome, DomainError> {
    let deep = rt.model();
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;

    // 1. This wave's slice, best-current-order first. A zero budget means "embed
    //    nothing this wave" (a comparator-only pass over an already-embedded pool).
    let slice = rescore::candidates_needing_embed(&conn, job_id, &deep, budget.wave1_budget)?;

    // 2. Make sure the rubric has vectors in the deep space *before* scoring in it.
    //    The ad can land after the run started (or be edited mid-run), and without
    //    this the deep space has no ad question at all — so the wave would "grade"
    //    nothing and the run would converge with quick-only rows. Cache-first, and by
    //    now the engine is up anyway.
    {
        let rubric = Rubric::load(&conn, job_id)?;
        rubric::ensure_vectors(rt, &conn, &rubric, &deep, None)
            .map_err(|e| DomainError::DbError { message: e.to_string() })?;
    }

    // 3. Embed the slice, then grade it in the deep space. `gate_wait` inside the
    //    embed loop yields to an open CV Inspector between candidates.
    let (embedded, embed_failed) = if slice.is_empty() {
        (0, 0)
    } else {
        let r = crate::embeddings::embed_selected_candidates(rt, &conn, job_id, &slice, progress)?;
        let rubric = Rubric::with_vectors(&conn, job_id, &deep)?;
        rubric::score_candidates_in_space(&conn, job_id, &deep, &rubric, Some(&slice))?;
        r
    };

    // 4. Comparator whoever in the frozen pool is ready.
    let stats = comparator::run_pool(rt, job_id).map_err(|e| DomainError::DbError { message: e })?;
    drop(conn);

    // 5. Project. Only the slice's rows change, but the rank is global.
    let _report = rescore::rescore_ids(rt, job_id, Some(&slice))?;
    let remaining_embed = remaining_embed_count(rt, job_id)?;
    let remaining_comparator = remaining_comparator_count(rt, job_id)?;
    let converged = remaining_embed == 0 && remaining_comparator == 0;

    {
        let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
        rescore::write_report(
            &conn,
            job_id,
            stats.as_ref(),
            "waves",
            wave,
            converged,
        )?;
    }

    Ok(WaveOutcome {
        wave,
        embedded,
        embed_failed,
        remaining_embed,
        comparator_ran: stats.is_some(),
        comparator_pairs_fresh: stats.as_ref().map(|s| s.fresh).unwrap_or(0),
        comparator_pairs_cached: stats.as_ref().map(|s| s.cached).unwrap_or(0),
        remaining_comparator,
        converged,
    })
}

/// Run the whole pipeline to convergence in one blocking pass: embed anything
/// missing, comparator the pool, project. This is the small-job path (where the wave
/// loop has nothing to iterate) and the path a test or a CLI wants.
pub fn run_to_convergence(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    budget: CascadeBudget,
    progress: crate::embeddings::EmbedProgress<'_>,
) -> Result<WaveOutcome, DomainError> {
    prepare_heavy_rubric(rt, job_id)?;
    run_wave(rt, job_id, 1, budget, progress)
}

/// Re-project a job without doing any engine work: the cheap half of a rank, and
/// the *whole* of an input change. See [`crate::rescore`].
pub fn project(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<rescore::RescoreReport, DomainError> {
    rescore::rescore(rt, job_id)
}

/// Whether a comparator artifact is usable for the current embedder.
pub fn comparator_available(rt: &Arc<EngineRuntime>) -> bool {
    comparator::stage_for(rt).is_some()
}