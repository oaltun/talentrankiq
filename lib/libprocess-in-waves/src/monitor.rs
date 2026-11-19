use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use libtriqui_client::types::{EmbeddingProgress, RankProgress};
use rusqlite::Connection;

pub fn monitor_loop(
    db_path: PathBuf,
    config: MonitorConfig,
    events: EventSink,
    shutdown_rx: mpsc::Receiver<()>,
) {
    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("monitor: open db failed: {e}");
            return;
        }
    };
    let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
    loop {
        let _ = poll_once(&conn, &config, &events);
        match shutdown_rx.recv_timeout(config.poll_interval) {
            Ok(_) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        }
    }
    let _ = (db_path, thread::current().id());
}

fn poll_once(
    conn: &Connection,
    config: &MonitorConfig,
    events: &EventSink,
) -> Result<(), rusqlite::Error> {
    let progress = queries::get_rank_progress(conn, &config.job_id)?;
    events.emit("rank-progress", &progress);
    let quick = queries::get_embedding_progress(conn, &config.job_id, &config.quick_model)?;
    events.emit("job-quick-progress", &quick);
    let deep = queries::get_embedding_progress(conn, &config.job_id, &config.deep_model)?;
    events.emit("job-deep-progress", &deep);
    Ok(())
}

#[derive(Clone)]
pub struct EventSink(mpsc::Sender<String>);

impl EventSink {
    pub fn new(tx: mpsc::Sender<String>) -> Self {
        EventSink(tx)
    }
    pub fn emit<D: serde::Serialize>(&self, name: &str, data: &D) {
        let event = serde_json::json!({"event": name, "data": data});
        if let Ok(line) = serde_json::to_string(&event) {
            let _ = self.0.send(line);
        }
    }
    pub fn clone_sender(&self) -> mpsc::Sender<String> {
        self.0.clone()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorError {
    #[error("db error: {0}")]
    Db(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone, Debug)]
pub struct MonitorConfig {
    pub poll_interval: std::time::Duration,
    pub job_id: String,
    pub quick_model: String,
    pub deep_model: String,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: std::time::Duration::from_secs(2),
            job_id: String::new(),
            quick_model: String::new(),
            deep_model: String::new(),
        }
    }
}

pub struct MonitorHandle {
    shutdown_tx: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MonitorHandle {
    pub fn shutdown(mut self) -> thread::Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            t.join()
        } else {
            Ok(())
        }
    }
}

pub fn spawn_monitor(
    db_path: PathBuf,
    config: MonitorConfig,
    events: EventSink,
) -> Result<MonitorHandle, MonitorError> {
    let (tx, rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || {
        monitor_loop(db_path, config, events, rx);
    });
    Ok(MonitorHandle {
        shutdown_tx: Some(tx),
        thread: Some(handle),
    })
}

pub mod queries {
    use super::*;

    pub fn get_rank_progress(
        conn: &Connection,
        job_id: &str,
    ) -> Result<RankProgress, rusqlite::Error> {
        let wave: i64 = conn
            .query_row(
                "SELECT wave FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let candidates: i64 = conn
            .query_row(
                "SELECT candidates FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let embedded_done: i64 = conn
            .query_row(
                "SELECT embedded_done FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let compared: i64 = conn
            .query_row(
                "SELECT compared FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let running: i64 = conn
            .query_row(
                "SELECT running FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(RankProgress {
            job_id: job_id.to_string(),
            stage: if running != 0 {
                "embed".into()
            } else {
                "done".into()
            },
            wave: wave as u32,
            embedded_done: embedded_done as u32,
            embedded_total: candidates as u32,
            judged: compared as u32,
            compared: compared as u32,
            candidates: candidates as u32,
            running: running != 0,
            reason: None,
        })
    }

    pub fn get_embedding_progress(
        conn: &Connection,
        job_id: &str,
        model_id: &str,
    ) -> Result<EmbeddingProgress, rusqlite::Error> {
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM cv WHERE job_id = ?1", [job_id], |r| {
                r.get(0)
            })
            .unwrap_or(0);
        let done: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM cv_embedding WHERE job_id = ?1 AND model_id = ?2",
                [job_id, model_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let failed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM cv_embedding WHERE job_id = ?1 AND model_id = ?2 AND error IS NOT NULL",
            [job_id, model_id],
            |r| r.get(0),
        ).unwrap_or(0);
        let running: i64 = conn
            .query_row(
                "SELECT running FROM waves_rank_state WHERE job_id = ?1",
                [job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(EmbeddingProgress {
            job_id: job_id.to_string(),
            done,
            total,
            failed,
            active: running != 0,
        })
    }
}
