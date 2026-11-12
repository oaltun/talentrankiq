//! CV import handlers (was gui `commands/import.rs`): validate + collect the
//! file list synchronously, mark the job as importing, then run the
//! extract → embed → rank pipeline on a background thread, forwarding
//! progress as `job-import-progress` / `job-embedding-progress` events.
//! Shared with the CV-directory watcher (`crate::serve::watch`).

use std::sync::Arc;

use chrono;
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
            // Mirror conversion progress for the CV-watch three-bar UI.
            self.sink.emit(
                "job-conversion-progress",
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
    // Mark this job as actively importing (cleared when done).
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
        use libtriqui_core::db::open_connection_at;
        use std::collections::HashSet;

        let emitter = ImportProgressEmitter::new(st.events.clone(), &jid);

        let conn = match open_connection_at(&st.db_path) {
            Ok(c) => c,
            Err(e) => {
                log::error!("import: failed to open db: {e}");
                let mut active = st.import_state.0.lock().expect("import state lock poisoned");
                active.remove(&jid);
                return;
            }
        };

        let _ = conn.execute(
            "UPDATE jobs SET import_directory_path = ?1, updated_at = datetime('now') WHERE id = ?2",
            rusqlite::params![dirstr, jid],
        );

        let existing: HashSet<String> = {
            let mut stmt = match conn.prepare(
                "SELECT source_file_path FROM candidates WHERE job_id = ?1",
            ) {
                Ok(s) => s,
                Err(_) => {
                    let mut active = st.import_state.0.lock().expect("import state lock poisoned");
                    active.remove(&jid);
                    return;
                }
            };
            stmt.query_map(rusqlite::params![jid], |r| r.get(0))
                .map(|rows| rows.filter_map(Result::ok).collect())
                .unwrap_or_default()
        };

        let total = files.len() as i64;
        let mut done = 0i64;
        let mut failed = 0i64;
        emitter.import(0, total, 0, true, "importing");

        for path in &files {
            let path_str = path.to_string_lossy().to_string();

            if existing.contains(&path_str) {
                done += 1;
                emitter.import(done, total, failed, true, "importing");
                continue;
            }

            let _ = conn.execute(
                "DELETE FROM candidates WHERE job_id = ?1 AND source_file_path = ?2",
                rusqlite::params![jid, path_str],
            );

            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            let file_type = match path.extension().and_then(|e| e.to_str()) {
                Some("pdf") => "Pdf",
                Some("docx") => "Docx",
                Some("doc") => "Doc",
                Some("txt") => "Txt",
                _ => "Other",
            }
            .to_string();

            let cv_id = format!("cv_{}", uuid::Uuid::new_v4());
            let _ = conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, imported_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'Pending', datetime('now'))",
                rusqlite::params![cv_id, jid, path_str, file_name, file_type],
            );

            done += 1;
            emitter.import(done, total, failed, true, "importing");
        }

        emitter.import(done, total, failed, false, "done");

        let mut active = st.import_state.0.lock().expect("import state lock poisoned");
        active.remove(&jid);
    });

    Ok(())
}

/// Start the convergence cascade after an import, unless the job has nothing
/// to rank or a cascade is already in flight.
///
/// This runs even when the job has **no job ad**: the embed stages are
/// ad-independent, so the run extracts quick vectors (stage 0.5, whole corpus)
/// and deep chunk vectors (the waves) while the user writes the ad. Scoring
/// needs a query and stays empty until one exists — the waves re-derive stage
/// 0 and re-sort every pass, so grades land as soon as the ad (or positive
/// criteria) does; if the run already converged, setting the ad spawns a
/// scoring-only cascade (`jobs::update_job`).
fn auto_start_cascade(state: &Arc<ServeState>, job_id: &str) {
    let extracted: i64 = {
        let conn = match state.db.read() {
            Ok(c) => c,
            Err(e) => {
                log::error!("import: db read failed, cascade not started: {e}");
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
    let mut conn = match state.db.read() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("import: db read failed, not dispatching CvAdded for {job_id}: {e}");
            return;
        }
    };
    let rows = query_extracted_candidates(&mut conn, job_id);
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
            imported_at: chrono::Utc::now().to_rfc3339(),
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


pub fn convert_pending(
    state: &Arc<ServeState>,
    job_id: &str,
) -> Result<serde_json::Value, ProtocolError> {
    use libtriqui_core::embed;
    use rusqlite::params;

    let mut conn = state.db.writer();
    let mut stmt = conn
        .prepare(
            "SELECT id, source_file_path FROM candidates \
             WHERE job_id = ?1 AND (extraction_status IN ('Pending','Failed') OR extracted_text IS NULL)",
        )
        .map_err(|e| crate::serve::handlers::domain(&libtriqui_core::types::DomainError::DbError {
            message: e.to_string(),
        }))?;
    let rows: Vec<(String, String)> = stmt
        .query_map(params![job_id], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(|e| crate::serve::handlers::domain(&libtriqui_core::types::DomainError::DbError {
            message: e.to_string(),
        }))?
        .filter_map(Result::ok)
        .collect();
    drop(stmt);

    let total = rows.len() as i64;
    let mut done = 0i64;
    let mut failed = 0i64;

    for (cid, path) in rows {
        let _ = conn.execute(
            "UPDATE candidates SET extraction_status = 'Extracting' WHERE id = ?1",
            params![cid],
        );
        match libtriqui_core::import::extract_text_for(std::path::Path::new(&path)) {
            Ok(text) => {
                let hash = embed::content_hash(&text);
                let dup: bool = conn
                    .query_row(
                        "SELECT 1 FROM candidates WHERE job_id = ?1 AND content_hash = ?2 AND id <> ?3",
                        params![job_id, hash, cid],
                        |_| Ok(true),
                    )
                    .unwrap_or(false);
                if dup {
                    let _ = conn.execute("DELETE FROM candidates WHERE id = ?1", params![cid]);
                } else {
                    let _ = conn.execute(
                        "UPDATE candidates SET extraction_status = 'Extracted', extracted_text = ?1, content_hash = ?2 WHERE id = ?3",
                        params![text.clone(), hash, cid],
                    );
                }
                if state
                    .cv_state
                    .lock()
                    .map(|s| s.cvs.contains_key(&cid))
                    .unwrap_or(false)
                {
                    state.dispatch_event(libcv_state::Event::TextExtracted {
                        cv_id: cid.clone(),
                        text,
                        status: libcv_state::ExtractionStatus::Extracted,
                    });
                }
            }
            Err(_) => {
                failed += 1;
                let _ = conn.execute(
                    "UPDATE candidates SET extraction_status = 'Failed' WHERE id = ?1",
                    params![cid],
                );
                if state
                    .cv_state
                    .lock()
                    .map(|s| s.cvs.contains_key(&cid))
                    .unwrap_or(false)
                {
                    state.dispatch_event(libcv_state::Event::TextExtracted {
                        cv_id: cid.clone(),
                        text: String::new(),
                        status: libcv_state::ExtractionStatus::Failed,
                    });
                }
            }
        }
        done += 1;
        state.events.emit(
            "job-conversion-progress",
            &libtriqui_client::types::EmbeddingProgress {
                job_id: job_id.to_string(),
                done,
                total,
                failed,
                active: done < total,
            },
        );
    }

    if total == 0 {
        let extracted: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
                params![job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let failed_total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Failed'",
                params![job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let total_all = extracted + failed_total;
        state.events.emit(
            "job-conversion-progress",
            &libtriqui_client::types::EmbeddingProgress {
                job_id: job_id.to_string(),
                done: total_all,
                total: total_all,
                failed: failed_total,
                active: false,
            },
        );
    }

    crate::serve::handlers::ok_json(serde_json::Value::Null)
}

pub fn scout_pending(
    state: &Arc<ServeState>,
    job_id: &str,
) -> Result<serde_json::Value, ProtocolError> {
    state.runtime.ensure_started();
    let scout_model = libtriqui_core::cascade::scout_model(&state.runtime);
    if scout_model.is_none() {
        let conn = state.db.read().map_err(|e| crate::serve::handlers::lock_err(e))?;
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
                rusqlite::params![job_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        state.events.emit(
            "job-quick-progress",
            &libtriqui_client::types::EmbeddingProgress {
                job_id: job_id.to_string(),
                done: total,
                total,
                failed: 0,
                active: false,
            },
        );
        return crate::serve::handlers::ok_json(serde_json::Value::Null);
    }
    let st = Arc::clone(state);
    let jid = job_id.to_string();
    let _ = libtriqui_core::quick::run_scout(&state.runtime, job_id, &|d, t, f, a| {
        st.events.emit(
            "job-quick-progress",
            &libtriqui_client::types::EmbeddingProgress {
                job_id: jid.clone(),
                done: d,
                total: t,
                failed: f,
                active: a,
            },
        );
    });
    crate::serve::handlers::ok_json(serde_json::Value::Null)
}

pub fn heavy_pending(
    state: &Arc<ServeState>,
    job_id: &str,
) -> Result<serde_json::Value, ProtocolError> {
    state.runtime.ensure_started();
    let conn = state.db.writer();
    let st = Arc::clone(state);
    let jid = job_id.to_string();
    let _ = libtriqui_core::embeddings::embed_missing_candidates(
        &state.runtime,
        &conn,
        job_id,
        &|d, t, f, a| {
            st.events.emit(
                "job-deep-progress",
                &libtriqui_client::types::EmbeddingProgress {
                    job_id: jid.clone(),
                    done: d,
                    total: t,
                    failed: f,
                    active: a,
                },
            );
        },
    );
    drop(conn);
    crate::serve::handlers::ok_json(serde_json::Value::Null)
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
