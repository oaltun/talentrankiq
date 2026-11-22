//! CV import handlers.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

fn collect_ids(conn: &rusqlite::Connection, sql: &str, job_id: &str) -> Result<Vec<String>, ProtocolError> {
    let mut stmt = conn.prepare(sql).map_err(|e| string_err(e.to_string()))?;
    let mut iter = stmt
        .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
        .map_err(|e| string_err(e.to_string()))?;
    let mut v = Vec::new();
    while let Some(r) = iter.next().transpose().map_err(|e| string_err(e.to_string()))? {
        v.push(r);
    }
    Ok(v)
}

pub fn import_cv_directory(
    state: &Arc<ServeState>,
    job_id: &str,
    directory: &str,
) -> Result<Value, ProtocolError> {
    // Enqueue a scan of the directory via the pipeline.
    state.pipeline.enqueue(libcv_pipeline::WorkItem::ScanDirectory {
        job_id: job_id.to_string(),
    });
    let _ = directory;
    ok_json(0i32)
}

pub fn convert_pending(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let pending = collect_ids(
        &conn,
        "SELECT id FROM candidates WHERE job_id = ?1 AND extraction_status = 'Pending'",
        job_id,
    )?;
    drop(conn);
    for cv_id in &pending {
        state.pipeline.enqueue(libcv_pipeline::WorkItem::TextExtract { cv_id: cv_id.clone() });
    }
    ok_json(pending.len() as i32)
}

pub fn scout_pending(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let pending = collect_ids(
        &conn,
        "SELECT id FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted' AND vector IS NULL",
        job_id,
    )?;
    drop(conn);
    for cv_id in &pending {
        state.pipeline.enqueue(libcv_pipeline::WorkItem::Quick { cv_id: cv_id.clone() });
    }
    ok_json(pending.len() as i32)
}

pub fn heavy_pending(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let pending = collect_ids(
        &conn,
        "SELECT id FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
        job_id,
    )?;
    drop(conn);
    for cv_id in &pending {
        state.pipeline.enqueue(libcv_pipeline::WorkItem::Deep { cv_id: cv_id.clone() });
    }
    ok_json(pending.len() as i32)
}