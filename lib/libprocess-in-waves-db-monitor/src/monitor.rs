use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::thread;

use libtriqui_cli::serve::EventSink;
use rusqlite::Connection;

use crate::queries;
use crate::MonitorConfig;

pub fn monitor_loop(
    db_path: PathBuf,
    config: MonitorConfig,
    events: EventSink,
    shutdown_rx: Receiver<()>,
) {
    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => { log::warn!("monitor: open db failed: {e}"); return; }
    };
    let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;");
    loop {
        // poll once
        let _ = poll_once(&conn, &config.job_id, &events);
        // wait for interval or shutdown
        match shutdown_rx.recv_timeout(config.poll_interval) {
            Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
        }
    }
    let _ = (db_path, thread::current().id());
}

fn poll_once(conn: &Connection, job_id: &str, events: &EventSink) -> Result<(), rusqlite::Error> {
    let progress = queries::get_rank_progress(conn, job_id).unwrap_or_default();
    events.emit("rank-progress", &progress);
    let quick = queries::get_embedding_progress(conn, job_id, "").unwrap_or_default();
    events.emit("job-quick-progress", &quick);
    let deep = queries::get_embedding_progress(conn, job_id, "").unwrap_or_default();
    events.emit("job-deep-progress", &deep);
    Ok(())
}
