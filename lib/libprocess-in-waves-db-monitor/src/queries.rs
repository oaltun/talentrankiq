use serde::Serialize;
use rusqlite::Connection;

#[derive(Clone, Debug, Default, Serialize)]
pub struct EmbeddingProgress {
    pub job_id: String,
    pub done: i64,
    pub total: i64,
    pub failed: i64,
    pub active: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RankProgress {
    pub job_id: String,
    pub stage: String,
    pub wave: i64,
    pub embedded_done: i64,
    pub embedded_total: i64,
    pub compared: i64,
    pub candidates: i64,
    pub running: bool,
    pub reason: Option<String>,
    pub converged: bool,
}

pub fn get_conversion_progress(conn: &Connection, job_id: &str) -> Result<EmbeddingProgress, rusqlite::Error> {
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM cv", [], |r| r.get(0)).unwrap_or(0);
    Ok(EmbeddingProgress { job_id: job_id.to_string(), done: total, total, failed: 0, active: false })
}

pub fn get_embedding_progress(conn: &Connection, job_id: &str, model_id: &str) -> Result<EmbeddingProgress, rusqlite::Error> {
    let total: i64 = conn.query_row("SELECT COUNT(*) FROM cv", [], |r| r.get(0)).unwrap_or(0);
    let done: i64 = if model_id.is_empty() {
        conn.query_row("SELECT COUNT(*) FROM cv_embedding", [], |r| r.get(0)).unwrap_or(0)
    } else {
        conn.query_row("SELECT COUNT(*) FROM cv_embedding WHERE model_id = ?1", [model_id], |r| r.get(0)).unwrap_or(0)
    };
    let running: i64 = conn.query_row("SELECT COALESCE((SELECT running FROM waves_rank_state WHERE job_id=?1),0)", [job_id], |r| r.get(0)).unwrap_or(0);
    Ok(EmbeddingProgress { job_id: job_id.to_string(), done, total, failed: 0, active: running != 0 })
}

pub fn get_rank_progress(conn: &Connection, job_id: &str) -> Result<RankProgress, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT job_id, running, wave, embedded_total, embedded_done, compared, candidates, converged, reason FROM waves_rank_state WHERE job_id = ?1")?;
    let mut rows = stmt.query([job_id])?;
    if let Some(row) = rows.next()? {
        let running: i64 = row.get(1)?;
        Ok(RankProgress {
            job_id: row.get(0)?,
            stage: "waves".to_string(),
            wave: row.get(2)?,
            embedded_total: row.get(3)?,
            embedded_done: row.get(4)?,
            compared: row.get(5)?,
            candidates: row.get(6)?,
            running: running != 0,
            reason: row.get(8).ok(),
            converged: row.get::<_, i64>(7)? != 0,
        })
    } else {
        Ok(RankProgress { job_id: job_id.to_string(), stage: "waves".to_string(), ..Default::default() })
    }
}