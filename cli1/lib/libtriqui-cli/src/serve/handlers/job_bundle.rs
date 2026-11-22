//! Job bundle export/import handlers.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::BundleExportOptions;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

pub fn export_job_bundle(
    state: &Arc<ServeState>,
    job_id: &str,
    _options: &BundleExportOptions,
    out_path: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| string_err(e.to_string()))?;
    std::fs::write(out_path, format!("{{\"job_id\":\"{job_id}\"}}"))
        .map_err(|e| string_err(e.to_string()))?;
    ok_json(libtriqui_client::types::BundleExportSummary {
        candidates_total: count,
        files_missing: 0,
        criteria_included: 0,
    })
}

pub fn import_job_bundle(state: &Arc<ServeState>, path: &str) -> Result<Value, ProtocolError> {
    let _ = state;
    let contents = std::fs::read_to_string(path).map_err(|e| string_err(e.to_string()))?;
    ok_json(contents)
}