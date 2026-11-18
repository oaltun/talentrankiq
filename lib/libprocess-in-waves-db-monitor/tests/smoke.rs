use std::sync::mpsc;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn monitor_spawns_and_shuts_down() {
    let dir = tempdir().unwrap();
    let dbpath = dir.path().join("t.db");
    let db = libprocess_in_waves::DbHandle::open_at(&dbpath).unwrap();
    db.update_waves_rank_state("job1", 0, 0, 0, 0, 0, 0, None, true).unwrap();
    let (tx, _rx) = mpsc::channel();
    let events = libprocess_in_waves_db_monitor::EventSink::new(tx);
    let cfg = libprocess_in_waves_db_monitor::MonitorConfig {
        poll_interval: Duration::from_millis(100),
        job_id: "job1".to_string(),
    };
    let handle = libprocess_in_waves_db_monitor::spawn_monitor(dbpath, cfg, events).unwrap();
    std::thread::sleep(Duration::from_millis(250));
    handle.shutdown().unwrap();
}
