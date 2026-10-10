//! Rank: the entry points, kept for the callers that still ask for "rank this
//! job" rather than "run the cascade".
//!
//! Ranking is no longer a pipeline of its own. It is three things that already
//! live elsewhere:
//!
//! - **stages** — `scout::run_scout`, `cascade::run_wave`, `judge::run_pool` —
//!   which own all the engine work and all the budgets;
//! - **one projection** — [`crate::rescore`], which turns stage grades into
//!   `points` and `rank` and makes no model calls;
//! - **one loop** — [`crate::cascade`] — which alternates the two until nothing is
//!   left.
//!
//! So this module is a compatibility shim: `sort_job_blocking` is "make sure the
//! stages have run, then project", which is exactly the small-job path and
//! exactly what a CLI or a test wants. The old `SortOptions` (six fields of
//! budget/skip flags threaded through the scoring code) is gone, along with the
//! counted/display-only cut it needed.

use std::sync::Arc;

use crate::cascade::{self, CascadeBudget};
use crate::embeddings::EmbedProgress;
use crate::runtime::EngineRuntime;
use crate::types::{DomainError, TriqiItem};

/// Label of the heavy embedder's ad column in the results table.
pub const AD_LABEL: &str = "Job fit (embed)";

/// Question id of the heavy embedder's ad grade. It lives in [`crate::rubric`]
/// now (the ad is one of the rubric's questions); re-exported here because the
/// CSV and bundle exporters speak in these ids.
pub use crate::rescore::JUDGE_Q as JUDGE_QUESTION_ID;
pub use crate::rubric::AD_Q as AD_QUESTION_ID;

/// Run the whole pipeline to convergence and return the triqui.
///
/// Blocking: it embeds anything missing, judges the pool, and projects. Callers
/// that want it in the background spawn the cascade instead
/// ([`crate::cascade::run_wave`] in a loop); this is for the paths that genuinely
/// want to wait — a CLI rank, a bundle import, a test.
pub fn sort_job_blocking(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    progress: EmbedProgress<'_>,
) -> Result<Vec<TriqiItem>, DomainError> {
    // Deliberately **no** `wait_until_ready` up front. The stages that do need the
    // engine (`embed_missing_candidates`, `rubric::ensure_vectors`) each start it
    // themselves, and only after their cache-only pass has shown there is real work.
    // Waiting here would reload a model for a rank the cache can answer in
    // milliseconds — which is the whole difference between "switching back to the
    // previous embedder costs nothing" and it costing a model load.
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e.to_string() })?;
    let heavy = rt.model();
    let missing = crate::rescore::count_needing_embed(&conn, job_id, &heavy)?;
    if missing > 0 {
        crate::embeddings::embed_missing_candidates(rt, &conn, job_id, progress)?;
    }
    // Score unconditionally, not only when something was embedded: after an embedder
    // switch every candidate has chunks again but no *grades*, and a grade is what
    // the projection reads. This is a cosine pass over stored vectors — cheap, and
    // the only way a "rank this job" request can be trusted.
    let rubric = crate::rubric::Rubric::with_vectors(&conn, job_id, &heavy)?;
    crate::rubric::score_candidates_in_space(&conn, job_id, &heavy, &rubric, None)?;
    drop(conn);
    // A blocking sort judges *everyone*, which is what "sorted" has always meant
    // to a caller that waits: pool size = the candidate count.
    let count = {
        let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e.to_string() })?;
        crate::rescore::capacity(&conn, job_id).ok();
        conn.query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
            rusqlite::params![job_id],
            |r| r.get::<_, i64>(0),
        )
        .map_err(|e| DomainError::DbError { message: e.to_string() })? as usize
    };
    cascade::prepare_heavy_rubric(rt, job_id)?;
    cascade::freeze_pool(rt, job_id, count)?;
    // Report what the judge actually did — "cached" when every verdict came from the
    // memoized cache is the whole point of the verdict cache, and the gui and the
    // tests read it. Passing `None` here would report `skipped` for a run that
    // judged, which is a lie in the direction of "nothing happened".
    let stats = crate::judge::run_pool(rt, job_id)
        .map_err(|e| DomainError::DbError { message: e })?;
    cascade::project(rt, job_id)?;

    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e.to_string() })?;
    crate::rescore::write_report(&conn, job_id, stats.as_ref(), "full", 0, true)?;
    let (triqui, _) = crate::rescore::read_triqui(&conn, job_id)?;
    Ok(triqui)
}

/// The current ranked triqui and the last report, straight from the DB.
pub fn converged_report(
    conn: &rusqlite::Connection,
    job_id: &str,
) -> Result<(Vec<TriqiItem>, crate::types::SortReport), DomainError> {
    crate::rescore::read_triqui(conn, job_id)
}

/// Rank a job in one pass under the cascade's own budgets — the shape the gui
/// uses for a small job, where the wave loop has nothing to iterate.
pub fn sort_small_job(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    progress: EmbedProgress<'_>,
) -> Result<Vec<TriqiItem>, DomainError> {
    let budget = CascadeBudget {
        wave1_budget: usize::MAX,
        pool_size: usize::MAX,
    };
    cascade::run_to_convergence(rt, job_id, budget, progress)?;
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e.to_string() })?;
    let (triqui, _) = crate::rescore::read_triqui(&conn, job_id)?;
    Ok(triqui)
}
