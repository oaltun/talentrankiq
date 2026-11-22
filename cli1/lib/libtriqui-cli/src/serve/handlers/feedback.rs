//! Feedback handlers (likes / ordered candidate ids).

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

/// Re-project a job whose feedback just changed.
///
/// Opens its own connection, so it must be called with `state.db` **not** held —
/// `inputs_changed` writes to the same database from a second connection and
/// would deadlock against the request's own writer lock.
fn reproject(state: &Arc<ServeState>, job_id: &str) -> Result<(), ProtocolError> {
    match libtriqui_core::rescore::inputs_changed(&state.runtime, job_id, false) {
        Ok(inv) => log::info!(
            "feedback: job {job_id} re-projected — {} candidates, {} rubric vector(s) embedded",
            inv.rescored,
            inv.embedded,
        ),
        Err(e) => {
            log::warn!("feedback: re-projection failed for {job_id}: {e}");
            return Err(string_err(format!("re-rank failed for {job_id}: {e}")));
        }
    }
    Ok(())
}

pub fn apply_feedback(
    state: &Arc<ServeState>,
    job_id: &str,
    _ordered_candidate_ids: Vec<String>,
    liked_candidate_ids: Vec<String>,
) -> Result<Value, ProtocolError> {
    {
        let mut conn = state.db.writer();
        conn.execute("UPDATE candidates SET liked = 0 WHERE job_id = ?1", rusqlite::params![job_id])
            .map_err(|e| string_err(e.to_string()))?;
        for cid in &liked_candidate_ids {
            conn.execute(
                "UPDATE candidates SET liked = 1 WHERE id = ?1",
                rusqlite::params![cid],
            )
            .map_err(|e| string_err(e.to_string()))?;
        }
        // Bump inputs_version so the next rank is stale (feedback is a rubric input).
        conn.execute(
            "UPDATE jobs SET inputs_updated_at = datetime('now'), inputs_version = inputs_version + 1 WHERE id = ?1",
            rusqlite::params![job_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
    }
    // The writer lock is released above; now learn the feedback vector and
    // re-project every candidate through the same path criteria use.
    reproject(state, job_id)?;
    // Request a rank after feedback change.
    state.request_rank(job_id);
    ok_json(liked_candidate_ids.len() as i32)
}

pub fn reset_feedback(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let updated = {
        let mut conn = state.db.writer();
        let updated = conn
            .execute("UPDATE candidates SET liked = 0 WHERE job_id = ?1", rusqlite::params![job_id])
            .map_err(|e| string_err(e.to_string()))?;
        conn.execute(
            "UPDATE jobs SET inputs_updated_at = datetime('now'), inputs_version = inputs_version + 1 WHERE id = ?1",
            rusqlite::params![job_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
        updated
    };
    reproject(state, job_id)?;
    // Request a rank after feedback reset.
    state.request_rank(job_id);
    ok_json(updated as i32)
}

pub fn list_liked_candidate_ids(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let mut stmt = conn
        .prepare("SELECT id FROM candidates WHERE job_id = ?1 AND liked = 1")
        .map_err(|e| string_err(e.to_string()))?;
    let ids = stmt
        .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
        .map_err(|e| string_err(e.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| string_err(e.to_string()))?;
    ok_json(ids)
}