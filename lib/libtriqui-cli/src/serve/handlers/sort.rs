//! Rubric ranking handler (thin wrapper over `libtriqui_core::sort`; was gui
//! `commands/sort.rs`). Scores each embedded candidate against the job's
//! rubric and returns the top-N triqui, forwarding backfill progress as
//! `job-embedding-progress` events. Long operation: the request stays open
//! until the ranking finishes (only this handler's thread blocks).

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_core::sort;

use crate::serve::handlers::import::embedding_progress_emitter;
use crate::serve::handlers::{domain, ok_json};
use crate::serve::ServeState;

pub fn sort_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let emit = embedding_progress_emitter(state, job_id);
    let triqui =
        sort::sort_job_blocking(&state.runtime, job_id, &emit).map_err(|e| domain(&e))?;
    ok_json(triqui)
}
