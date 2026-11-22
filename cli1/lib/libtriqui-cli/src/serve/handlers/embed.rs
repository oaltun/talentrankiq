//! Embedding engine runtime handlers.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::params::ChunkMatchParams;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

pub fn start_runtime(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    state.runtime.start();
    state.runtime.wait_until_ready(std::time::Duration::from_secs(180))
        .map_err(|e| string_err(e))?;
    ok_json(state.runtime.status_info())
}

pub fn runtime_status(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    ok_json(state.runtime.status_info())
}

pub fn embed_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| string_err(e.to_string()))?;
    ok_json(count)
}

pub fn chunk_match(_state: &Arc<ServeState>, _p: &ChunkMatchParams) -> Result<Value, ProtocolError> {
    ok_json(libtriqui_client::types::ChunkMatchResult {
        model: String::new(),
        chunks: Vec::new(),
    })
}