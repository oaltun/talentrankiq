//! Sort/ranking handlers (blocking sort_job + the new waves pipeline).

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

pub fn sort_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let outcome = libtriqui_core::sort::sort_job(state.runtime.as_ref(), job_id).map_err(string_err)?;
    ok_json(outcome.ranked as i32)
}