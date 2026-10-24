//! The scout: one cheap multilingual pass that leaves every candidate with a
//! *complete* scorecard.
//!
//! The scout is a second embedder over the shared engine client. It stores its own
//! chunk vectors under its own `vector_model` in `candidate_chunks` and its own
//! grades in `stage_scores`, leaving the heavy embedder's `candidates.vector` and
//! its grades untouched. It is deliberately *transient*: it loads, embeds the
//! corpus, and unloads again before the heavy embedder loads, so only one model is
//! ever resident.
//!
//! # Why it scores the whole rubric
//!
//! Scoring only the job ad — what this did until scoring v3 — leaves every
//! candidate the heavy embedder has not reached with an ad similarity and nothing
//! else. That single limitation is what forced the old design to invent pending
//! rows, a counted/display-only cut, and a basis tier to keep the sheet
//! meaningful. Scoring the whole rubric costs one static embed per criterion
//! (microseconds) plus `N · questions` cosines over vectors the scout already
//! stored (milliseconds at 2000 CVs), and it removes all three.
//!
//! The consequence is the property the whole simplification rests on: **from the
//! first pass, every candidate has a total**, and the background work only ever
//! *upgrades its precision*.
//!
//! Which model it is comes from `cascade.scout_model` in `triq.yaml` — swapping
//! the scout is a one-line config change (the default is a Model2Vec-style static
//! embedder chosen on measured speed, see `bench/results/scout/`).

use std::sync::Arc;

use crate::embeddings::{self, EmbedProgress};
use crate::rubric::{self, Rubric, AD_Q};
use crate::runtime::EngineRuntime;
use crate::types::DomainError;

/// Question id under which the scout's ad grade appears in `candidate_scores`.
pub const SCOUT_QUESTION_ID: &str = "scout";
/// `kind` of the scout's row in `candidate_scores`.
pub const SCOUT_KIND: &str = "Scout";
/// Column label of the scout stage in the results table.
pub const SCOUT_LABEL: &str = "Scout (fast embed)";

/// The configured scout model slug, or `None` when the stage is off
/// (`cascade.scout_enabled: false`, no `scout_model`, or a slug equal to the
/// full embedder — which would be a pointless second pass in the same space).
pub fn scout_model_for(full_model: &str) -> Option<String> {
    let cfg = crate::app_config::cascade_config();
    if !cfg.scout_enabled() {
        return None;
    }
    let slug = cfg.scout_model.trim();
    if slug.is_empty() || slug == full_model {
        return None;
    }
    Some(slug.to_string())
}

/// Chunk size for the scout. The scout is not a `gui_embedders` entry (it is not
/// user-selectable), so its chunking comes from `cascade.scout_chunk_size`.
pub fn scout_chunk_size() -> usize {
    crate::app_config::cascade_config().scout_chunk_size()
}

/// Outcome of one scout pass over a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoutOutcome {
    /// Candidates embedded (newly or from cache) by the scout.
    pub embedded: i64,
    /// Candidates whose scout embedding failed validation.
    pub failed: i64,
    /// Candidates that ended up with a stored scout grade.
    pub graded: i64,
    /// Rubric questions the scout embedded into its own space.
    pub questions_embedded: usize,
    /// Whether scoring was deferred because the job has no ad yet.
    pub deferred: bool,
}

/// Run the scout stage for `job_id`: embed the job's rubric and every extracted
/// candidate in the scout's own space, then grade every candidate against every
/// question it has a vector for.
///
/// Purely additive — it writes only its own space's chunks and grades, and the
/// whole pass is cache-keyed by `(model, content_hash)`, so a re-run after an ad or
/// import change only re-embeds what actually changed. `progress(done, total,
/// failed, active)` reports candidate embed progress.
///
/// A job with **no ad and no criteria** still gets its candidates embedded (that
/// work is ad-independent) but no grades, since there is nothing to measure
/// against. The caller re-runs this pass when a query lands; every candidate's
/// text is already cached, so it costs a few scout embeds and no corpus pass.
pub fn run_scout(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    progress: EmbedProgress<'_>,
) -> Result<ScoutOutcome, DomainError> {
    let full_model = rt.model();
    let Some(model) = scout_model_for(&full_model) else {
        return Ok(ScoutOutcome {
            embedded: 0,
            failed: 0,
            graded: 0,
            questions_embedded: 0,
            deferred: false,
        });
    };
    let conn = rt.open_conn().map_err(|e| DomainError::DbError { message: e.to_string() })?;
    // Arm freshness for the scout slug (predictably `ort` — it declares no mlx
    // variant — so this records once and never clears scout rows).
    rt.ensure_backend_fresh(&conn, &model);
    let _activity = embeddings::ActivityGuard { rt };
    let job_title: String = conn
        .query_row(
            "SELECT COALESCE(NULLIF(title, ''), id) FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| DomainError::DbError { message: e.to_string() })?;

    // The scout's own session, over the shared engine client. It is loaded now and
    // released at the end of the pass, before the heavy embedder loads.
    rt.preload(&model)?;
    let embedder = rt.embedder_for(&model)?;

    // The corpus. This is the expensive half and it is ad-independent: it can run
    // while the user is still writing the job ad, and it never has to run again
    // when the ad changes.
    rt.gate_wait();
    rt.set_activity("scout", &job_title, Some("corpus".into()), Some(&model));
    let (embedded, failed) =
        embeddings::embed_candidates_with(rt, &conn, job_id, &model, embedder.clone(), None, progress)
            .map_err(|e| DomainError::DbError { message: e.to_string() })?;

    // The rubric, in the scout's space. A handful of short texts through a static
    // bag-of-words model — the ad, every criterion, and whatever feedback vectors
    // have been learned in this space.
    let questions_embedded =
        rubric::ensure_vectors(rt, &conn, &Rubric::load(&conn, job_id)?, &model, Some(&embedder))
            .map_err(|e| DomainError::DbError { message: e.to_string() })?;
    // Re-read with the vectors this pass just stored, so the grading below sees
    // them (an ad that landed minutes ago has no vector in this space yet).
    let rubric = Rubric::with_vectors(&conn, job_id, &model)
        .map_err(|e| DomainError::DbError { message: e.to_string() })?;

    // Grade the whole corpus against every question this space can measure.
    let has_ad = rubric.vector(AD_Q, &model).is_some();
    let scoreable = rubric.scored_questions(&model).len();
    if !has_ad && scoreable <= 1 {
        // No query at all: the corpus is embedded, the grades wait for one.
        rt.unload_secondary(&model);
        log::info!(
            "scout: job {job_id} has no rubric yet — corpus embedded ({embedded}), grading deferred"
        );
        return Ok(ScoutOutcome {
            embedded,
            failed,
            graded: 0,
            questions_embedded,
            deferred: true,
        });
    }

    let graded = rubric::score_candidates_in_space(&conn, job_id, &model, &rubric, None)?;
    drop(conn);

    // Free the scout's session before the heavy embedder loads (one resident model
    // at a time).
    rt.unload_secondary(&model);

    log::info!(
        "scout: job {job_id} model {model}: {embedded} embedded, {graded} graded \
         ({} questions), {failed} failed",
        scoreable
    );
    Ok(ScoutOutcome {
        embedded,
        failed,
        graded: graded as i64,
        questions_embedded,
        deferred: false,
    })
}

/// Count of the candidates that still have no scout-space grade.
pub fn count_needing_grading(
    conn: &rusqlite::Connection,
    job_id: &str,
    model: &str,
) -> Result<i64, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM candidates c
         WHERE c.job_id = ?1 AND c.extraction_status = 'Extracted' AND c.extracted_text IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM candidate_chunks cc
                           WHERE cc.candidate_id = c.id AND cc.vector_model = ?2)",
        rusqlite::params![job_id, model],
        |r| r.get(0),
    )
    .map_err(|e| e.to_string())
}