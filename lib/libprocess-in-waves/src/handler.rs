use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::RankState;
use libtriqui_core::runtime::EngineRuntime;
use libtriqui_core::scout::scout_model_for;

use crate::{ProcessConfig, ProcessContext, JobConfig, DbHandle as WavesDbHandle, hash_text, monitor::{spawn_monitor, MonitorConfig, EventSink}};

pub fn run_process_in_waves(
    db_path: PathBuf,
    app_dir: PathBuf,
    runtime: Arc<EngineRuntime>,
    events: mpsc::Sender<String>,
    job_id: &str,
    config: ProcessConfig,
) -> Result<RankState, ProtocolError> {
    let (title, job_ad_text, import_dir): (String, Option<String>, Option<String>) = {
        let conn = WavesDbHandle::open_at(&db_path)
            .map_err(|e| ProtocolError::internal(format!("waves db open: {e}")))?;
        conn.with_reader(|c| {
            c.query_row(
                "SELECT title, job_ad_text, import_directory_path FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
        })
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => ProtocolError::internal(format!("job {job_id} not found")),
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

    let waves_db = WavesDbHandle::open_at(&db_path)
        .map_err(|e| ProtocolError::internal(format!("waves db open: {e}")))?;
    let already_running: bool = {
        let conn = WavesDbHandle::open_at(&db_path)
            .map_err(|e| ProtocolError::internal(format!("waves db open: {e}")))?;
        conn.with_reader(|c| {
            Ok(c.query_row(
                "SELECT running FROM waves_rank_state WHERE job_id = ?1",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            ).map(|v| v != 0).unwrap_or(false))
        }).unwrap_or(false)
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
        let conn = WavesDbHandle::open_at(&db_path)
            .map_err(|e| ProtocolError::internal(format!("waves db open: {e}")))?;
        conn.with_reader(|c| {
            let mut stmt = c.prepare("SELECT text FROM criteria WHERE job_id = ?1")
                .map_err(|e| rusqlite::Error::ExecuteReturnedResults)?; // placeholder, we'll handle differently
            let rows = stmt.query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
        .map_err(|e| ProtocolError::internal(e.to_string()))?
    };
    let _ = waves_db.update_waves_rank_state(job_id, 0, 0, 0, 0, 0, 0, None, true);

    let ctx = ProcessContext {
        db: waves_db.clone(),
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
    let _ = spawn_monitor(db_path.clone(), monitor_cfg, monitor_events);

    let job_id_owned = job_id.to_string();
    std::thread::spawn(move || {
        if let Err(e) = crate::process_in_waves(&mut ctx, config) {
            log::error!("process_in_waves failed for {job_id_owned}: {e}");
        }
        let _ = ctx.db.update_waves_rank_state(&job_id_owned, 0, 0, 0, 0, 0, 0, None, false);
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