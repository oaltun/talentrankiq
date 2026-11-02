//! CV import handlers (was gui `commands/import.rs`): validate + collect the
//! file list synchronously, mark the job as importing, then run the
//! extract → embed → rank pipeline on a background thread, forwarding
//! progress as `job-import-progress` / `job-embedding-progress` events.
//! Shared with the CV-directory watcher (`crate::serve::watch`).

use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{DomainError, EmbeddingProgress, ImportProgress};
use libtriqui_core::import;

use crate::serve::handlers::ok_json;
use crate::serve::{EventSink, ProgressThrottle, ServeState};

/// Progress emitter with per-source throttling (~10 Hz) plus pass-through for
/// phase changes and the final event, so bursts from huge imports never flood
/// the pipe while the UI still gets every phase edge and completion.
pub struct ImportProgressEmitter {
    job_id: String,
    sink: EventSink,
    throttle: ProgressThrottle,
    last_phase: std::sync::Mutex<String>,
}

/// Build an `job-embedding-progress` emitter for `(done, total, failed, active)`
/// callbacks — the shape `libtriqui_core::embeddings` reports through.
pub fn embedding_progress_emitter(
    state: &Arc<ServeState>,
    job_id: &str,
) -> impl Fn(i64, i64, i64, bool) + use<> {
    let emitter = ImportProgressEmitter::new(state.events.clone(), job_id);
    move |done, total, failed, active| emitter.embed(done, total, failed, active)
}

impl ImportProgressEmitter {
    pub fn new(sink: EventSink, job_id: &str) -> Self {
        Self {
            job_id: job_id.to_string(),
            sink,
            throttle: ProgressThrottle::default(),
            last_phase: std::sync::Mutex::new(String::new()),
        }
    }

    /// Import-pipeline progress: `(done, total, failed, active, phase)`.
    pub fn import(&self, done: i64, total: i64, failed: i64, active: bool, phase: &str) {
        let phase_changed = {
            let mut last = self.last_phase.lock().expect("phase lock");
            if *last != phase {
                *last = phase.to_string();
                true
            } else {
                false
            }
        };
        // The completion event (active=false) and every phase edge always pass.
        if !active || phase_changed || self.throttle.allow() {
            self.sink.emit(
                "job-import-progress",
                &ImportProgress {
                    job_id: self.job_id.clone(),
                    done,
                    total,
                    failed,
                    active,
                    phase: phase.to_string(),
                },
            );
        }
    }

    /// Embedding progress: `(done, total, failed, active)`.
    pub fn embed(&self, done: i64, total: i64, failed: i64, active: bool) {
        if !active || self.throttle.allow() {
            self.sink.emit(
                "job-embedding-progress",
                &EmbeddingProgress {
                    job_id: self.job_id.clone(),
                    done,
                    total,
                    failed,
                    active,
                },
            );
        }
    }
}

/// Shared import spawn: mark the job active and run the extract → embed → rank
/// pipeline on a background thread, forwarding progress events. Used by the
/// manual import handler and by the CV-directory watcher.
pub fn spawn_dir_import(
    state: &Arc<ServeState>,
    job_id: &str,
    directory: &str,
    files: Vec<std::path::PathBuf>,
) -> Result<(), DomainError> {
    // Mark this job as actively importing (the pipeline removes it when done).
    state
        .import_state
        .0
        .lock()
        .expect("import state lock poisoned")
        .insert(job_id.to_string());

    let st = Arc::clone(state);
    let jid = job_id.to_string();
    let dirstr = directory.to_string();
    std::thread::spawn(move || {
        let emitter = ImportProgressEmitter::new(st.events.clone(), &jid);
        let emit = |done: i64, total: i64, failed: i64, active: bool, phase: &str| {
            emitter.import(done, total, failed, active, phase);
        };
        import::run_import(
            &st.runtime,
            &jid,
            &dirstr,
            files,
            Some(libtriqui_core::db::ImportState(Arc::clone(
                &st.import_state.0,
            ))),
            &emit,
        );
        // Import is extract-only: it left a stage-0 order on disk, and the
        // embed→judge cascade starts automatically from here, converging in the
        // background (design §9). Import-done has already fired, so the gui
        // shows the stage-0 sheet immediately and follows `rank-progress` until
        // convergence.
        auto_start_cascade(&st, &jid);
        // Tell the shadow reducer about every CV that landed in the database so
        // cv_state mirrors the candidates table. Idempotent: re-dispatching for a
        // CV already tracked is a no-op, so re-imports and watcher catch-up
        // scans never duplicate state.
        dispatch_cv_added(&st, &jid);
    });

    Ok(())
}

/// Start the convergence cascade after an import, unless the job has nothing
/// to rank or a cascade is already in flight.
///
/// This runs even when the job has **no job ad**: the embed stages are
/// ad-independent, so the run extracts scout vectors (stage 0.5, whole corpus)
/// and heavy chunk vectors (the waves) while the user writes the ad. Scoring
/// needs a query and stays empty until one exists — the waves re-derive stage
/// 0 and re-sort every pass, so grades land as soon as the ad (or positive
/// criteria) does; if the run already converged, setting the ad spawns a
/// scoring-only cascade (`jobs::update_job`).
fn auto_start_cascade(state: &Arc<ServeState>, job_id: &str) {
    let extracted: i64 = {
    let conn = match state.db.0.lock() {
            Ok(c) => c,
            Err(e) => {
                log::error!("import: db lock failed, cascade not started: {e}");
                return;
            }
        };
        conn.query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .unwrap_or(0)
    };
    if extracted == 0 {
        return;
    }
    match crate::serve::handlers::rank::spawn_cascade(state, job_id) {
        Ok(true) => log::info!("import: cascade started for job {job_id} ({extracted} CVs)"),
        Ok(false) => log::info!("import: cascade already running for job {job_id}"),
        Err(e) => log::warn!("import: could not start cascade for job {job_id}: {e}"),
    }
}

/// Tell the shadow reducer about every successfully imported CV of `job_id`.
///
/// Runs after the import pipeline finishes, on the background thread. The
/// `CvAdded` reducer inserts a `CvRecord` keyed by `cv_id`, so re-dispatching
/// for a CV already tracked is idempotent — re-imports and watcher catch-up
/// scans never duplicate shadow state.
fn dispatch_cv_added(state: &Arc<ServeState>, job_id: &str) {
    let mut conn = match state.db.0.lock() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("import: db lock failed, not dispatching CvAdded for {job_id}: {e}");
            return;
        }
    };
    let rows = query_extracted_candidates(&mut *conn, job_id);
    // `conn` (the DB lock) is released here, before the reducer dispatch below.
    drop(conn);
    let rows = match rows {
        Ok(r) => r,
        Err(e) => {
            log::warn!("import: {e}");
            return;
        }
    };
    for (cv_id, file_path, file_name, file_type) in rows {
        state.pipeline.dispatch_event(libcv_state::Event::CvAdded {
            cv_id,
            job_id: job_id.to_string(),
            file_path,
            file_name,
            file_type,
        });
    }
}

/// Query the successfully imported CVs of `job_id` from the database.
fn query_extracted_candidates(
    conn: &mut rusqlite::Connection,
    job_id: &str,
) -> Result<Vec<(String, String, String, String)>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, source_file_path, file_name, file_type FROM candidates \
             WHERE job_id = ?1 AND extraction_status = 'Extracted'",
        )
        .map_err(|e| format!("CvAdded prepare failed for {job_id}: {e}"))?;
    let mapped = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| format!("CvAdded query failed for {job_id}: {e}"))?;
    mapped
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| format!("CvAdded collect failed for {job_id}: {e}"))
}

pub fn import_cv_directory(
    state: &Arc<ServeState>,
    job_id: &str,
    directory: &str,
) -> Result<serde_json::Value, ProtocolError> {
    // Validate + collect up front so bad input fails synchronously.
    let files = import::collect_supported_files(directory)
        .map_err(|e| crate::serve::handlers::domain(&e))?;
    spawn_dir_import(state, job_id, directory, files)
        .map_err(|e| crate::serve::handlers::domain(&e))?;
    ok_json(serde_json::Value::Null)
}
