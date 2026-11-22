//! Results CSV export handler.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

pub fn export_results_csv(
    state: &Arc<ServeState>,
    job_id: &str,
    out_path: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let mut stmt = conn
        .prepare(
            "SELECT file_name, rank, points, points_max FROM candidates
             WHERE job_id = ?1 AND rank IS NOT NULL ORDER BY rank",
        )
        .map_err(|e| string_err(e.to_string()))?;
    let mut rows = Vec::new();
    let iter = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok(format!(
                "{},{},{},{}",
                r.get::<_, String>(0)?,
                r.get::<_, i32>(1)?,
                r.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                r.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
            ))
        })
        .map_err(|e| string_err(e.to_string()))?;
    for r in iter {
        rows.push(r.map_err(|e| string_err(e.to_string()))?);
    }
    let mut csv = String::from("file_name,rank,points,points_max\n");
    csv.push_str(&rows.join("\n"));
    std::fs::write(out_path, csv).map_err(|e| string_err(e.to_string()))?;
    let count = rows.len() as i64;
    let questions = 1usize;
    ok_json(libtriqui_client::types::ResultsCsvSummary { rows: count, questions })
}