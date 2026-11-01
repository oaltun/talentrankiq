//! Candidate detail + browser handlers (thin serve wrappers over
//! `libtriqui_core::candidates`; was gui `commands/candidates.rs`).

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::DomainError;
use libtriqui_core::candidates;

use crate::serve::handlers::{domain, lock_err, ok_json, string_err};
use crate::serve::ServeState;

pub fn soft_delete_cv(state: &Arc<ServeState>, cv_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    libtriqui_core::candidates::soft_delete_candidate(&conn, cv_id).map_err(string_err)?;
    drop(conn);
    state.shadow_dispatch(libcv_state::Event::CvSoftDeleted { cv_id: cv_id.to_string() });
    Ok(serde_json::Value::Null)
}

pub fn forget_cv(state: &Arc<ServeState>, cv_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    libtriqui_core::candidates::forget_candidate(&conn, cv_id).map_err(string_err)?;
    drop(conn);
    state.shadow_dispatch(libcv_state::Event::CvForgotten { cv_id: cv_id.to_string() });
    Ok(serde_json::Value::Null)
}

fn as_domain(e: DomainError) -> ProtocolError {
    domain(&e)
}

pub fn get_candidate(state: &Arc<ServeState>, candidate_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    candidates::get_candidate(&conn, candidate_id)
        .map_err(as_domain)
        .and_then(ok_json)
}

/// List all of a job's candidates as a page, sorted by total rubric points (default)
/// or by a single column (`sort_by`: `None`/`"__total"` = total points, `"__name"` =
/// file name, `"__rank"` = stored rank, else a rubric-question id). `sort_dir` = `true`
/// sorts ascending; `false`/None sorts descending. `query` (when set) filters by the
/// imported CV **content** (substring, ASCII case-insensitive — SQLite `lower()` is
/// ASCII-only). Each row carries its per-question `scores` for the spreadsheet columns.
pub fn list_candidates(
    state: &Arc<ServeState>,
    job_id: &str,
    offset: i32,
    limit: i32,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    candidates::list_candidates(&conn, job_id, offset, limit, query, sort_by, sort_dir)
        .map_err(as_domain)
        .and_then(ok_json)
}

/// Prev/next candidate ids in the job's browser ordering (same sort as `list_candidates`,
/// defaulting to total points) so the overlay's `<-`/`->` buttons walk every CV in the
/// current sort. `query` (when set) restricts the walk to the filtered set.
#[allow(clippy::too_many_arguments)]
pub fn candidate_neighbors(
    state: &Arc<ServeState>,
    job_id: &str,
    candidate_id: &str,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    candidates::candidate_neighbors(&conn, job_id, candidate_id, query, sort_by, sort_dir)
        .map_err(as_domain)
        .and_then(ok_json)
}
