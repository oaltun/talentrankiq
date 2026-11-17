use serde::Serialize;
use rusqlite::Connection;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Progress {
    pub total: i64,
    pub done: i64,
    pub failed: i64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RankProgress {
    pub job_id: String,
    pub running: bool,
    pub wave: i64,
    pub embedded_total: i64,
    pub embedded_done: i64,
    pub compared: i64,
    pub candidates: i64,
    pub converged: bool,
    pub reason: Option<String>,
}

pub fn get_conversion_progress(conn: &Connection, _job_id: &str) -> Result<Progress, rusqlite::Error> {
    // total = cv count, done = cv count (all extracted), failed = 0 (no extraction status in this schema)
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM cv", [], |r| r.get(0)).unwrap_or(0);
    Ok(Progress { total, done: total, failed: 0 })
}

pub fn get_embedding_progress(conn: &Connection, _job_id: &str, model_id: &str) -> Result<Progress, rusqlite::Error> {
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM cv", [], |r| r.get(0)).unwrap_or(0);
    let done: i64 = if model_id.is_empty() {
        conn.query_row("SELECT COUNT(*) FROM cv_embedding", [], |r| r.get(0)).unwrap_or(0)
    } else {
        conn.query_row("SELECT COUNT(*) FROM cv_embedding WHERE model_id = ?1", [model_id], |r| r.get(0)).unwrap_or(0)
    };
    Ok(Progress { total, done, failed: 0 })
}

pub fn get_rank_progress(conn: &Connection, job_id: &str) -> Result<RankProgress, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT job_id, running, wave, embedded_total, embedded_done, compared, candidates, converged, reason FROM rank_state WHERE job_id = ?1")?;
    let mut rows = stmt.query([job_id])?;
    if let Some(row) = rows.next()? {
        Ok(RankProgress {
            job_id: row.get(0)?,
            running: row.get::<_, i64>(1)? != 0,
            wave: row.get(2)?,
            embedded_total: row.get(3)?,
            embedded_done: row.get(4)?,
            compared: row.get(5)?,
            candidates: row.get(6)?,
            converged: row.get::<_, i64>(7)? != 0,
            reason: row.get(8).ok(),
        })
    } else {
        Ok(RankProgress { job_id: job_id.to_string(), ..Default::default() })
    }
}
