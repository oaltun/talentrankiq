//! The cascade: scout, then waves of embed + judge, projected by rescore.
//!
//! The whole orchestration, in one loop, with no budgets inside the scoring path
//! (`gui/design/app1/cascade-simple.md`):
//!
//! ```text
//! scout.run()                      # whole corpus, once — every candidate gets a total
//! ensure_vectors(heavy)            # the rubric in the heavy space
//! pool = freeze_pool(top P)        # frozen: judge win-rates are set-relative
//! while not heavy.has_all(job) or not judge.has_all(job):
//!     slice = heavy.next_slice(k1) # best-current-order first
//!     heavy.embed(slice)           # → stage_scores(heavy)
//!     judge.run(pool)              # → candidate_scores[ad_judge], memoized per pair
//!     rescore(slice)               # → points + rank + display rows
//! ```
//!
//! # The guarantees
//!
//! - **Convergence.** The loop ends when every candidate holds heavy-space chunks
//!   and every pool member holds a judge grade. At that point the sheet is
//!   exactly what a full blocking sort would produce, for the pool.
//! - **Monotone improvement.** A wave only ever replaces a candidate's grades
//!   with better ones (scout space → heavy space; ad cosine → win-rate), and the
//!   pool is frozen, so the top of the sheet stabilises early and never
//!   regresses for reasons unrelated to the candidate.
//! - **A run never survives its inputs changing in a way that matters.** Embedding
//!   is ad-independent, so an input change does not stop a run — it re-projects.
//!   See [`crate::rescore`]; the caller decides when to re-project.

use std::sync::Arc;

use crate::judge;
use crate::rescore;
use crate::rubric::{self, Rubric};
use crate::runtime::EngineRuntime;
use crate::scout;
use crate::types::DomainError;

/// Budget for one cascade wave: `wave1_budget` candidates the heavy embedder
/// embeds this wave, `pool_size` candidates the judge compares. Both come from
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
    /// Candidates the heavy embedder embedded this wave.
    pub embedded: i64,
    /// Candidates that failed to embed this wave.
    pub embed_failed: i64,
    /// Candidates still awaiting the heavy embedder after this wave.
    pub remaining_embed: i64,
    /// Whether the judge ran (an artifact is configured and the ad and CVs clear
    /// its length floors) and had pool members to compare.
    pub judge_ran: bool,
    /// Fresh judge verdicts computed this wave.
    pub judge_pairs_fresh: u32,
    /// Judge verdicts served from the memoization cache.
    pub judge_pairs_cached: u32,
    /// Pool members still without a judge grade after this wave.
    pub remaining_judge: i64,
    /// True when nothing is left to embed and the pool is fully judged.
    pub converged: bool,
}

/// Run stage 0.5: the scout pass over every candidate. Returns `Ok(None)` when no
/// scout is configured, or when the scout *is* the current embedder — a second
/// pass in the same space would be pure waste.
pub fn run_scout_stage(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    progress: crate::embeddings::EmbedProgress<'_>,
) -> Result<Option<scout::ScoutOutcome>, DomainError> {
    if scout::scout_model_for(&rt.model()).is_none() {
        return Ok(None);
    }
    Ok(Some(scout::run_scout(rt, job_id, progress)?))
}

/// The configured scout slug, or `None` when the stage is off.
pub fn scout_model(rt: &Arc<EngineRuntime>) -> Option<String> {
    scout::scout_model_for(&rt.model())
}

/// How many candidates still need the heavy embedder.
pub fn remaining_embed_count(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
) -> Result<i64, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    rescore::count_needing_embed(&conn, job_id, &rt.model())
}

/// Ensure the rubric has vectors in the heavy space, so the waves' scoring is
/// never waiting on an embed. Cheap and cache-first: after the first pass it is a
/// few indexed reads.
pub fn prepare_heavy_rubric(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<(), DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    let rubric = Rubric::load(&conn, job_id)?;
    rubric::ensure_vectors(rt, &conn, &rubric, &rt.model(), None)
        .map_err(|e| DomainError::DbError { message: e.to_string() })?;
    Ok(())
}

/// Freeze the judge pool: the top `pool_size` candidates by current total.
///
/// Called once per run, after the scout pass has graded the corpus and the first
/// rescore has ranked it. Re-freezing is the *only* thing that changes the pool,
/// which is what keeps a win-rate meaning the same thing at two moments. An ad
/// change invalidates every memoized verdict, so `inputs_changed` clears the pool
/// and the next run re-freezes it.
pub fn freeze_pool(rt: &Arc<EngineRuntime>, job_id: &str, pool_size: usize) -> Result<usize, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    crate::db::freeze_judge_pool(&conn, job_id, pool_size)
        .map_err(|e| DomainError::DbError { message: e.to_string() })
}

/// How many pool members are still unjudged.
pub fn remaining_judge_count(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
) -> Result<i64, DomainError> {
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;
    let ids = crate::db::judge_pool_ids(&conn, job_id).map_err(|e| DomainError::DbError { message: e.to_string() })?;
    if ids.is_empty() {
        return Ok(0);
    }
    let holes = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT COUNT(*) FROM judge_pool jp
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

/// Run one wave: embed the next slice in best-current-order, judge whoever in the
/// pool is ready, and re-project.
///
/// The judge runs *after* the embed, on whatever pool members the heavy embedder
/// has reached — it cannot compare a CV it has no vectors for. Returns whether
/// anything remains, so the caller can loop.
pub fn run_wave(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    wave: u32,
    budget: CascadeBudget,
    progress: crate::embeddings::EmbedProgress<'_>,
) -> Result<WaveOutcome, DomainError> {
    let heavy = rt.model();
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e })?;

    // 1. This wave's slice, best-current-order first. A zero budget means "embed
    //    nothing this wave" (a judge-only pass over an already-embedded pool).
    let slice = rescore::candidates_needing_embed(&conn, job_id, &heavy, budget.wave1_budget)?;

    // 2. Make sure the rubric has vectors in the heavy space *before* scoring in it.
    //    The ad can land after the run started (or be edited mid-run), and without
    //    this the heavy space has no ad question at all — so the wave would "grade"
    //    nothing and the run would converge with scout-only rows. Cache-first, and by
    //    now the engine is up anyway.
    {
        let rubric = Rubric::load(&conn, job_id)?;
        rubric::ensure_vectors(rt, &conn, &rubric, &heavy, None)
            .map_err(|e| DomainError::DbError { message: e.to_string() })?;
    }

    // 3. Embed the slice, then grade it in the heavy space. `gate_wait` inside the
    //    embed loop yields to an open CV Inspector between candidates.
    let (embedded, embed_failed) = if slice.is_empty() {
        (0, 0)
    } else {
        let r = crate::embeddings::embed_selected_candidates(rt, &conn, job_id, &slice, progress)?;
        let rubric = Rubric::with_vectors(&conn, job_id, &heavy)?;
        rubric::score_candidates_in_space(&conn, job_id, &heavy, &rubric, Some(&slice))?;
        r
    };

    // 4. Judge whoever in the frozen pool is ready.
    let stats = judge::run_pool(rt, job_id).map_err(|e| DomainError::DbError { message: e })?;
    drop(conn);

    // 5. Project. Only the slice's rows change, but the rank is global.
    let _report = rescore::rescore_ids(rt, job_id, Some(&slice))?;
    let remaining_embed = remaining_embed_count(rt, job_id)?;
    let remaining_judge = remaining_judge_count(rt, job_id)?;
    let converged = remaining_embed == 0 && remaining_judge == 0;

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
        judge_ran: stats.is_some(),
        judge_pairs_fresh: stats.as_ref().map(|s| s.fresh).unwrap_or(0),
        judge_pairs_cached: stats.as_ref().map(|s| s.cached).unwrap_or(0),
        remaining_judge,
        converged,
    })
}

/// Run the whole pipeline to convergence in one blocking pass: embed anything
/// missing, judge the pool, project. This is the small-job path (where the wave
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

/// Whether a judge artifact is usable for the current embedder.
pub fn judge_available(rt: &Arc<EngineRuntime>) -> bool {
    judge::stage_for(rt).is_some()
}