use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;


pub mod monitor;
pub mod queries;

#[derive(Clone)]
pub struct EventSink(std::sync::mpsc::Sender<String>);

impl EventSink {
    pub fn new(tx: std::sync::mpsc::Sender<String>) -> Self { EventSink(tx) }
    pub fn emit<D: serde::Serialize>(&self, name: &str, data: &D) {
        let event = serde_json::json!({"event": name, "data": data});
        if let Ok(line) = serde_json::to_string(&event) {
            let _ = self.0.send(line);
        }
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
    pub poll_interval: Duration,
    pub job_id: String,
    pub quick_model: String,
    pub deep_model: String,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self { poll_interval: Duration::from_secs(2), job_id: String::new(), quick_model: String::new(), deep_model: String::new() }
    }
}

pub struct MonitorHandle {
    shutdown_tx: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MonitorHandle {
    pub fn shutdown(mut self) -> thread::Result<()> {
        if let Some(tx) = self.shutdown_tx.take() { let _ = tx.send(()); }
        if let Some(t) = self.thread.take() { t.join() } else { Ok(()) }
    }
}

pub fn spawn_monitor(
    db_path: PathBuf,
    config: MonitorConfig,
    events: EventSink,
) -> Result<MonitorHandle, MonitorError> {
    let (tx, rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || {
        monitor::monitor_loop(db_path, config, events, rx);
    });
    Ok(MonitorHandle { shutdown_tx: Some(tx), thread: Some(handle) })
}
