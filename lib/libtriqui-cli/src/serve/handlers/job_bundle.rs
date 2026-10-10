//! Job bundle export/import handlers (was gui `commands/job_bundle.rs`).
//! Export writes a `.triquijob` zip to a caller-chosen path; import
//! creates a NEW job, stages the bundle's CV files into a fresh directory
//! under the default CV root (which becomes the job's watched CV dir), runs
//! the normal extract→embed→rank pipeline, then applies the results snapshot
//! so the imported triqui matches the sender's view.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::BundleExportOptions;
use libtriqui_core::job_bundle::{
    apply_bundle_results, create_job_from_bundle, export_job, extract_bundle_candidates,
    read_bundle,
};

use crate::serve::handlers::{lock_err, ok_json, string_err};
use crate::serve::watch;
use crate::serve::ServeState;

/// File-name-safe slug of the job title for the staging directory.
fn slug(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    let collapsed = cleaned
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let truncated: String = collapsed.chars().take(40).collect();
    if truncated.is_empty() {
        "job".into()
    } else {
        truncated
    }
}

pub fn export_job_bundle(
    state: &Arc<ServeState>,
    job_id: &str,
    options: &BundleExportOptions,
    out_path: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let summary = export_job(
        &conn,
        job_id,
        options,
        Path::new(out_path),
        &state.app_version,
    )
    .map_err(string_err)?;
    ok_json(summary)
}

pub fn import_job_bundle(state: &Arc<ServeState>, path: &str) -> Result<Value, ProtocolError> {
    let bundle_path = PathBuf::from(path);
    let meta = read_bundle(&bundle_path).map_err(string_err)?;

    // Stage the CV files into a fresh dir under the default CV root; it becomes
    // the job's real, watched CV directory (visible in SetupPane).
    let stage = watch::cv_root(state).join("imported").join(format!(
        "{}-{}",
        slug(&meta.job.title),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    ));
    extract_bundle_candidates(&bundle_path, &stage).map_err(string_err)?;
    let stage_str = stage.to_string_lossy().to_string();

    // Create the job + criteria rows (new ids, weights defaulted where absent).
    let created = {
        let conn = state.db.0.lock().map_err(lock_err)?;
        create_job_from_bundle(&conn, &meta, &stage_str).map_err(string_err)?
    };
    let job_id = created.job_id.clone();

    // Mark the job active so the UI shows the import pipeline and the CV-dir
    // watcher's scan (fired by staging noise) stays out until we're done.
    state
        .import_state
        .0
        .lock()
        .expect("import state lock poisoned")
        .insert(job_id.clone());

    let st = Arc::clone(state);
    let jid = job_id.clone();
    let results = meta.results.clone();
    let criterion_map = created.criterion_map.clone();
    let include_likes = true; // rows carry `liked: false` when likes were excluded
    std::thread::spawn(move || {
        let files = libtriqui_core::import::collect_supported_files(&stage_str).unwrap_or_default();
        {
            let emitter =
                crate::serve::handlers::import::ImportProgressEmitter::new(st.events.clone(), &jid);
            let emit = |done: i64, total: i64, failed: i64, active: bool, phase: &str| {
                emitter.import(done, total, failed, active, phase);
            };
            libtriqui_core::import::run_import(
                &st.runtime,
                &jid,
                &stage_str,
                files,
                Some(libtriqui_core::db::ImportState(std::sync::Arc::clone(
                    &st.import_state.0,
                ))),
                &emit,
            );
        }

        // The pipeline's auto-rank ran on the receiving machine; overlay the
        // sender's snapshot so the imported triqui matches what was sent.
        match libtriqui_core::db::open_connection_at(&st.db_path) {
            Ok(conn) => {
                if let Some(results) = &results {
                    match apply_bundle_results(&conn, &jid, results, &criterion_map, include_likes)
                    {
                        Ok(stats) => log::info!(
                            "bundle import {jid}: snapshot applied ({} matched, {} unmatched, {} liked)",
                            stats.rows_matched,
                            stats.rows_unmatched,
                            stats.liked_applied
                        ),
                        Err(e) => log::warn!("bundle import {jid}: snapshot apply failed: {e}"),
                    }
                }
            }
            Err(e) => log::warn!("bundle import {jid}: no db for snapshot: {e}"),
        }

        // The staged dir is the job's CV directory. Under the single-watcher
        // policy we deliberately do NOT install a watcher here (it would
        // outlive the single-watcher slot): the frontend navigates to the
        // fresh job, whose `watch_job` rpc installs the watcher; its
        // catch-up scan skips the files just imported via the DB filter.
    });

    // Read the fresh job back for the frontend to navigate to.
    let conn = state.db.0.lock().map_err(lock_err)?;
    let job = conn
        .query_row(
            &format!(
                "SELECT {} FROM jobs WHERE id = ?1",
                crate::serve::handlers::jobs::JOB_COLS
            ),
            rusqlite::params![&job_id],
            crate::serve::handlers::jobs::job_from_row,
        )
        .map_err(crate::serve::handlers::jobs::db_err)?;
    ok_json(job)
}
