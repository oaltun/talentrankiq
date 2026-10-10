//! Results CSV export handler (was gui `commands/results_csv.rs`): writes one
//! row per ranked candidate to a caller-chosen path.

use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_core::results_csv;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

pub fn export_results_csv(
    state: &Arc<ServeState>,
    job_id: &str,
    out_path: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().expect("db lock poisoned");
    let summary =
        results_csv::export_results_csv(&conn, job_id, Path::new(out_path)).map_err(string_err)?;
    ok_json(summary)
}
