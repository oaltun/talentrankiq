//! Embedding engine handlers (was gui `commands/embed.rs`): start, status
//! snapshot, and background candidate backfill. The engine lifecycle lives in
//! `libtriqui_core::runtime::EngineRuntime` inside the serve process.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::params::*;
use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::DomainError;
use libtriqui_core::chunk_match;
use libtriqui_core::db;
use libtriqui_core::embeddings;

use crate::serve::handlers::import::embedding_progress_emitter;
use crate::serve::handlers::{domain, ok_json};
use crate::serve::ServeState;

/// Kick off the embedding engine (connect-or-spawn) if it hasn't already started.
/// Called by the frontend once React has loaded, so app launch isn't blocked waiting
/// for model loading. Embedding/Sort lazily wait for it too, as a safety net.
pub fn start_runtime(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    state.runtime.ensure_started();
    ok_json(serde_json::Value::Null)
}

/// Snapshot of the embedding engine state for the UI.
pub fn runtime_status(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    ok_json(state.runtime.status_info())
}

/// Backfill embeddings for a job (e.g. after the engine becomes ready). Runs
/// on its own thread + connection; progress arrives via the
/// `job-embedding-progress` event. Responds immediately (start/track shape).
pub fn embed_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let st = Arc::clone(state);
    let jid = job_id.to_string();
    std::thread::spawn(move || {
        let emit = embedding_progress_emitter(&st, &jid);
        match db::open_connection_at(&st.db_path) {
            Ok(conn) => {
                if let Err(e) =
                    embeddings::embed_missing_candidates(&st.runtime, &conn, &jid, &emit)
                {
                    log::warn!("embed_job {jid}: {e}");
                    return;
                }
                // New vectors mean new grades, so grade and re-project: a backfill
                // that left the sheet unranked would make the button look broken
                // (scoring v3 — the projection reads `stage_scores`, which a
                // backfill alone does not touch).
                let heavy = st.runtime.model();
                match libtriqui_core::rubric::Rubric::with_vectors(&conn, &jid, &heavy)
                    .and_then(|rubric| {
                        libtriqui_core::rubric::score_candidates_in_space(
                            &conn, &jid, &heavy, &rubric, None,
                        )
                    }) {
                    Ok(scored) => {
                        log::info!("embed_job {jid}: {scored} candidates graded under {heavy}");
                    }
                    Err(e) => log::warn!("embed_job {jid}: grading failed: {e}"),
                }
                drop(conn);
                match libtriqui_core::rescore::rescore(&st.runtime, &jid) {
                    Ok(r) => log::info!(
                        "embed_job {jid}: re-projected ({} scored, {} ungraded)",
                        r.scored,
                        r.ungraded
                    ),
                    Err(e) => log::warn!("embed_job {jid}: re-projection failed: {e}"),
                }
            }
            Err(e) => log::error!("embed_job {jid}: failed to open db: {e}"),
        }
    });
    ok_json(serde_json::Value::Null)
}

/// Match Lens (CV Inspector): chunk a candidate's CV at a user-set token
/// size, embed every chunk, and cosine-score each against a target (job ad or
/// one criterion). Blocking model work, so it runs on its OWN
/// connection (not the shared DB lock) — the dispatcher already put this
/// request on a dedicated thread, so other rpcs stay responsive while it runs.
pub fn chunk_match(state: &Arc<ServeState>, p: &ChunkMatchParams) -> Result<Value, ProtocolError> {
    let conn = db::open_connection_at(&state.db_path).map_err(|e| {
        domain(&DomainError::DbError {
            message: format!("failed to open db: {e}"),
        })
    })?;
    chunk_match::chunk_match(
        &state.runtime,
        &conn,
        &p.candidate_id,
        &p.target,
        p.chunk_size,
    )
    .map_err(|e| domain(&e))
    .and_then(ok_json)
}
