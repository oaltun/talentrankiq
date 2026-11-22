//! Job ad file handlers.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{DomainError, JobAdFile};

use crate::serve::handlers::{domain, ok_json, string_err};
use crate::serve::ServeState;

fn resolve_ad_path(state: &Arc<ServeState>, job_id: &str) -> Result<(String, bool), ProtocolError> {
    let conn = state.db.read_conn();
    let existing: Option<String> = conn
        .query_row(
            "SELECT job_ad_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| string_err(e.to_string()))?;
    if let Some(p) = existing {
        if !p.is_empty() {
            return Ok((p, false));
        }
    }
    // Default: <app_dir>/<job_id>/job-ad.txt
    let path = state.app_dir.join(job_id).join("job-ad.txt");
    let p = path.to_string_lossy().to_string();
    drop(conn);
    let mut wconn = state.db.writer();
    wconn.execute(
        "UPDATE jobs SET job_ad_path = ?1, updated_at = datetime('now') WHERE id = ?2",
        rusqlite::params![&p, job_id],
    )
    .map_err(|e| string_err(e.to_string()))?;
    Ok((p, true))
}

pub fn get_job_ad(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let exists: bool = conn
        .query_row("SELECT 1 FROM jobs WHERE id = ?1", rusqlite::params![job_id], |_| Ok(true))
        .unwrap_or(false);
    if !exists {
        return Err(domain(&DomainError::NotFound { entity: "Job".into(), id: job_id.to_string() }));
    }
    drop(conn);
    let (path, created) = resolve_ad_path(state, job_id)?;
    libtriqui_core::job_ad::ensure_file(std::path::Path::new(&path)).map_err(string_err)?;
    let text = libtriqui_core::job_ad::read_ad(std::path::Path::new(&path));
    ok_json(JobAdFile { path, text, created })
}

pub fn set_job_ad_path(state: &Arc<ServeState>, job_id: &str, path: &str) -> Result<Value, ProtocolError> {
    let mut conn = state.db.writer();
    let updated = conn
        .execute(
            "UPDATE jobs SET job_ad_path = ?1, updated_at = datetime('now') WHERE id = ?2",
            rusqlite::params![path, job_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound { entity: "Job".into(), id: job_id.to_string() }));
    }
    // Load the ad text into jobs.job_ad_text so the ad is available for
    // embedding and appears in the job rubric.
    libtriqui_core::job_ad::ensure_file(std::path::Path::new(path)).map_err(string_err)?;
    let text = libtriqui_core::job_ad::read_ad(std::path::Path::new(path));
    conn.execute(
        "UPDATE jobs SET job_ad_text = ?1, inputs_updated_at = datetime('now'), inputs_version = inputs_version + 1, updated_at = datetime('now') WHERE id = ?2",
        rusqlite::params![&text, job_id],
    )
    .map_err(|e| string_err(e.to_string()))?;
    drop(conn);
    // Embed the ad (and any missing criteria / candidate centroids) now, so the
    // ad vector exists before a later ranking trigger.
    if let Err(e) = libtriqui_core::ensure::ensure_inputs(&state.runtime, job_id) {
        log::warn!("job_ad: ensure_inputs after set_job_ad_path failed: {e}");
    }
    // Request a rank now that inputs are ready.
    state.request_rank(job_id);
    ok_json(true)
}

pub fn reload_job_ad(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    reload_and_rerank(state, job_id)
}

pub fn reload_and_rerank(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let (path, created) = resolve_ad_path(state, job_id)?;
    libtriqui_core::job_ad::ensure_file(std::path::Path::new(&path)).map_err(string_err)?;
    let text = libtriqui_core::job_ad::read_ad(std::path::Path::new(&path));
    {
        let mut conn = state.db.writer();
        conn.execute(
            "UPDATE jobs SET job_ad_text = ?1, inputs_updated_at = datetime('now'), inputs_version = inputs_version + 1, updated_at = datetime('now') WHERE id = ?2",
            rusqlite::params![&text, job_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
    }
    // Re-embed the ad from the reloaded file (and fill any other missing
    // scoring inputs) so the ad vector matches the new contents.
    if let Err(e) = libtriqui_core::ensure::ensure_inputs(&state.runtime, job_id) {
        log::warn!("job_ad: ensure_inputs after reload_and_rerank failed: {e}");
    }
    // Request a rank now that inputs are ready.
    state.request_rank(job_id);
    ok_json(JobAdFile { path, text, created })
}