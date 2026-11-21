use std::path::PathBuf;
use std::sync::{mpsc, Arc};

use libdb_manager::Db;
use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::RankState;
use libtriqui_core::runtime::EngineRuntime;
use libtriqui_core::scout::scout_model_for;

use crate::db::WavesDb;
use crate::{
    hash_text,
    monitor::{spawn_monitor, EventSink, MonitorConfig},
    JobConfig, ProcessConfig, ProcessContext,
};

#[tracing::instrument(
    name = "run_process_in_waves",
    skip(db, app_dir, runtime, events, job_id, config, request_log_file)
)]
pub fn run_process_in_waves(
    db: Arc<Db>,
    app_dir: PathBuf,
    runtime: Arc<EngineRuntime>,
    events: mpsc::Sender<String>,
    job_id: &str,
    config: ProcessConfig,
    request_log_file: Option<Arc<std::sync::Mutex<std::fs::File>>>,
) -> Result<RankState, ProtocolError> {
    let waves_db: Arc<Db> = Arc::clone(&db);
    let _ = waves_db.write(|c| crate::db::init_schema(c));
    let (title, job_ad_text, import_dir): (String, Option<String>, Option<String>) = {
        waves_db.read(|c| {
            c.query_row(
                "SELECT title, job_ad_text, import_directory_path FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                ProtocolError::internal(format!("job {job_id} not found"))
            }
            e => ProtocolError::internal(format!("db: {e}")),
        })?
    };

    let watch_dir = import_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| app_dir.clone());
    let job_ad_text = job_ad_text.unwrap_or_default();

    let full_model = runtime.model();
    let quick_model = scout_model_for(&full_model).unwrap_or_else(|| full_model.clone());
    let deep_model = full_model;

    let already_running: bool = {
        waves_db.read(|c| {
            c.query_row(
                "SELECT running FROM waves_rank_state WHERE job_id = ?1",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            )
            .map(|v| v != 0)
            .unwrap_or(false)
        })
    };
    if already_running {
        return Ok(RankState {
            running: true,
            ranking: true,
            stage: Some("waves".into()),
            wave: None,
            embedded_total: 0,
            embedded_done: 0,
            judged: 0,
            candidates: 0,
            converged: false,
            reason: None,
        });
    }

    let jobad_hash = hash_text(&job_ad_text);
    let criteria: Vec<String> = {
        waves_db.read(|c| {
            let mut stmt = c
                .prepare("SELECT text FROM criteria WHERE job_id = ?1")
                .unwrap();
            let rows = stmt.query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0)).unwrap();
            let mut out = Vec::new();
            for r in rows {
                out.push(r.unwrap());
            }
            out
        })
    };
    let _ = waves_db.update_waves_rank_state(job_id, 0, 0, 0, 0, 0, 0, None, true);

    let ctx = ProcessContext {
        db: Arc::clone(&waves_db),
        watch_dir: watch_dir.clone(),
        job: JobConfig {
            id: job_id.to_string(),
            title: title.clone(),
            job_ad_text: job_ad_text.clone(),
            job_ad_hash: jobad_hash.clone(),
            criteria,
            import_directory_path: Some(watch_dir.clone()),
            quick_model: quick_model.clone(),
            deep_model: deep_model.clone(),
            comparator_model_path: None,
            similarity_method: libembedrank::RetrievalStrategy::MaxSim,
        },
        runtime: Arc::clone(&runtime),
        comparator_artifact: None,
    };

    let mut ctx = ctx;

    let monitor_events = EventSink::new(events.clone());
    let monitor_cfg = MonitorConfig {
        poll_interval: std::time::Duration::from_millis(config.poll_interval_ms),
        job_id: job_id.to_string(),
        quick_model: quick_model.clone(),
        deep_model: deep_model.clone(),
    };
    // Keep the handle alive for the whole run: dropping it closes the shutdown
    // channel, and `monitor_loop` treats a disconnected channel as a shutdown
    // signal — so a dropped handle kills the monitor after its very first poll
    // (before any embeddings exist), leaving the gui's progress stuck at 0.
    let monitor = spawn_monitor(app_dir.join("triqui.db"), monitor_cfg, monitor_events)
        .map_err(|e| ProtocolError::internal(format!("spawn monitor: {e}")))?;

    let job_id_owned = job_id.to_string();
    let had_request_log = request_log_file.is_some();
    std::thread::spawn(move || {
        if let Some(file) = request_log_file {
            libtriqui_core::logging::set_current_request_log(file);
        }
        if let Err(e) = crate::process_in_waves(&mut ctx, config) {
            tracing::error!("process_in_waves failed for {job_id_owned}: {e}");
        }
        let _ = ctx
            .db
            .update_waves_rank_state(&job_id_owned, 0, 0, 0, 0, 0, 0, None, false);
        if had_request_log {
            libtriqui_core::logging::clear_current_request_log();
        }
        // Emit one final progress snapshot so the gui reflects the terminal
        // counts even if the monitor's poll interval outlived the last wave.
        monitor.shutdown();
    });

    Ok(RankState {
        running: true,
        ranking: true,
        stage: Some("waves".into()),
        wave: None,
        embedded_total: 0,
        embedded_done: 0,
        judged: 0,
        candidates: 0,
        converged: false,
        reason: None,
    })
}
