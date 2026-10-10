//! Feedback-learning handlers (was gui `commands/feedback.rs`), with progress
//! forwarded through the serve event sink instead of Tauri events.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_core::feedback;

use crate::serve::handlers::import::embedding_progress_emitter;
use crate::serve::handlers::{domain, lock_err, ok_json};
use crate::serve::ServeState;

/// Apply staged feedback (a reordered triqui + a desired liked-id set) to the learned
/// vectors, persist liked flags, then re-rank and return the new triqui.
/// The request stays open until the re-rank completes (long operation; the
/// request loop is never blocked — only this handler's thread is).
pub fn apply_feedback(
    state: &Arc<ServeState>,
    job_id: &str,
    ordered_candidate_ids: Vec<String>,
    liked_candidate_ids: Vec<String>,
) -> Result<Value, ProtocolError> {
    let emit = embedding_progress_emitter(state, job_id);
    let triqui = feedback::apply_feedback_blocking(
        &state.runtime,
        job_id,
        ordered_candidate_ids,
        liked_candidate_ids,
        &emit,
    )
    .map_err(|e| domain(&e))?;
    ok_json(triqui)
}

/// Zero the learned feedback vectors + clear liked flags, then re-rank.
pub fn reset_feedback(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let emit = embedding_progress_emitter(state, job_id);
    let triqui =
        feedback::reset_feedback_blocking(&state.runtime, job_id, &emit).map_err(|e| domain(&e))?;
    ok_json(triqui)
}

/// All liked candidate ids for a job (seeds the frontend staging store).
pub fn list_liked_candidate_ids(
    state: &Arc<ServeState>,
    job_id: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let ids = feedback::list_liked_candidate_ids(&conn, job_id).map_err(|e| domain(&e))?;
    ok_json(ids)
}
