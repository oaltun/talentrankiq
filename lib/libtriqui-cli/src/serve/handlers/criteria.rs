//! Criterion CRUD handlers (was gui `commands/criteria.rs`).

use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{Criterion, DomainError};
use libtriqui_core::app_config;

use crate::serve::handlers::{domain, lock_err, ok_json, string_err};
use crate::serve::ServeState;

pub const CRIT_COLS: &str = "id, job_id, text, weight, created_at";

pub fn criterion_from_row(row: &rusqlite::Row) -> rusqlite::Result<Criterion> {
    Ok(Criterion {
        id: row.get(0)?,
        job_id: row.get(1)?,
        text: row.get(2)?,
        weight: row.get(3)?,
        created_at: row.get(4)?,
    })
}


/// Resolve the owning job id for a criterion (used to bump `inputs_updated_at`).
fn job_id_for_criterion(conn: &rusqlite::Connection, criterion_id: &str) -> Option<String> {
    conn.query_row(
        "SELECT job_id FROM criteria WHERE id = ?1",
        rusqlite::params![criterion_id],
        |r| r.get::<_, String>(0),
    )
    .ok()
}

fn validate(text: &str, weight: f64) -> Result<(), DomainError> {
    if text.trim().is_empty() {
        return Err(DomainError::ValidationError {
            message: "Criterion text cannot be empty".into(),
        });
    }
    if weight < 0.0 {
        return Err(DomainError::ValidationError {
            message: "Weight must be ≥ 0".into(),
        });
    }
    Ok(())
}

pub fn add_criterion(
    state: &Arc<ServeState>,
    job_id: &str,
    text: &str,
    weight: Option<f64>,
) -> Result<Value, ProtocolError> {
    // `None` uses the criterion default from `ikapp.yaml` (`rubric_defaults:`).
    let weight = weight.unwrap_or_else(|| app_config::rubric_defaults().criterion_points);
    validate(text, weight).map_err(|e| domain(&e))?;
    let id = format!("crit_{}", Uuid::new_v4());
    let conn = state.db.0.lock().map_err(lock_err)?;
    conn.execute(
        "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, ?2, ?3, 'Positive', ?4)",
        rusqlite::params![&id, job_id, text, weight],
    )
    .map_err(libtriqui_core::db::db_err)?;
    let criterion = conn
        .query_row(
            &format!("SELECT {CRIT_COLS} FROM criteria WHERE id = ?1"),
            rusqlite::params![&id],
            criterion_from_row,
        )
        .map_err(libtriqui_core::db::db_err)?;
    drop(conn);
    // One invalidation path for every ranking input (`rescore::inputs_changed`):
    // embed the one criterion text that changed and re-project. No cancel, and no
    // "stale until you press Re-rank" — the rubric is the only state a rank reads,
    // and a criterion edit touches nothing a running cascade is computing (the judge
    // never read a criterion).
    reproject(state, job_id)?;
    ok_json(criterion)
}

/// Re-project a job whose rubric just changed.
///
/// Opens its own connection, so it must be called with `state.db` **not** held —
/// `inputs_changed` writes to the same database from a second connection and would
/// deadlock against the request's own lock.
fn reproject(state: &Arc<ServeState>, job_id: &str) -> Result<(), ProtocolError> {
    match libtriqui_core::rescore::inputs_changed(&state.runtime, job_id, false) {
        Ok(inv) => log::info!(
            "criteria: job {job_id} re-projected — {} candidates, {} rubric vector(s) embedded",
            inv.rescored,
            inv.embedded,
        ),
        Err(e) => {
            log::warn!("criteria: re-projection failed for {job_id}: {e}");
            return Err(string_err(format!("re-rank failed for {job_id}: {e}")));
        }
    }
    Ok(())
}

pub fn update_criterion(
    state: &Arc<ServeState>,
    id: &str,
    text: &str,
    weight: f64,
) -> Result<Value, ProtocolError> {
    validate(text, weight).map_err(|e| domain(&e))?;
    let conn = state.db.0.lock().map_err(lock_err)?;
    let updated = conn
        .execute(
            "UPDATE criteria SET text = ?1, weight = ?2 WHERE id = ?3",
            rusqlite::params![text, weight, id],
        )
        .map_err(libtriqui_core::db::db_err)?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Criterion".into(),
            id: id.to_string(),
        }));
    }
    let jid = job_id_for_criterion(&conn, id);
    let criterion = conn
        .query_row(
            &format!("SELECT {CRIT_COLS} FROM criteria WHERE id = ?1"),
            rusqlite::params![id],
            criterion_from_row,
        )
        .map_err(libtriqui_core::db::db_err)?;
    drop(conn);
    if let Some(jid) = jid {
        reproject(state, &jid)?;
    }
    ok_json(criterion)
}

pub fn delete_criterion(state: &Arc<ServeState>, id: &str) -> Result<(), ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let jid = job_id_for_criterion(&conn, id);
    conn.execute("DELETE FROM criteria WHERE id = ?1", rusqlite::params![id])
        .map_err(libtriqui_core::db::db_err)?;
    drop(conn);
    if let Some(jid) = jid {
        reproject(state, &jid)?;
    }
    Ok(())
}
