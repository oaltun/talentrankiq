//! Job-ad-as-a-file handlers: resolve the path, seed the file, read it, and
//! react when the user edits it on disk.
//!
//! There is no ad editor in the app any more — the ad is a text file the user
//! owns. This module is the whole of the app's involvement: make sure the file
//! exists, hand its path and contents to the gui, and keep `job_ad_text` in
//! sync when the file changes underneath us.
//!
//! Two entry points matter:
//! - [`get_job_ad`] — read-only, for the gui (also seeds the default file the
//!   first time, and arms the watcher).
//! - [`set_job_ad_path`] — the "Change…" button: point the job at another file.
//! - [`reload_and_rerank`] — called from the watcher thread when the file
//!   changes on disk. Reloads the ad and, if it really changed, re-ranks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::JobAdFile;
use libtriqui_core::job_ad;
use libtriqui_core::types::DomainError;

use crate::serve::handlers::domain;
use crate::serve::handlers::jobs::{job_from_row, JOB_COLS};
use crate::serve::handlers::{lock_err, string_err};
use crate::serve::watch;
use crate::serve::ServeState;

/// File-name-safe slug for a job's folder, matching the CV directory's so both
/// live side by side (`<root>/job-N/cv` and `<root>/job-N/job-ad.txt`).
pub fn job_slug(title: &str) -> String {
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

/// The job's ad path, resolving (and persisting, and creating) the default the
/// first time. Returns `(path, created_the_file_now)`.
fn resolve_path(state: &Arc<ServeState>, job_id: &str) -> Result<(PathBuf, bool), DomainError> {
    let existing: Option<String> = {
        let conn = state.db.0.lock().map_err(lock_err)?;
        conn.query_row(
            "SELECT job_ad_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .ok()
        .flatten()
    };
    if let Some(p) = existing.filter(|p| !p.trim().is_empty()) {
        let path = PathBuf::from(p);
        // The user may have deleted it; recreate rather than showing a dead path.
        let existed = path.exists();
        job_ad::ensure_file(&path).map_err(|e| DomainError::ValidationError { message: e })?;
        return Ok((path, !existed));
    }

    let title: String = {
        let conn = state.db.0.lock().map_err(lock_err)?;
        conn.query_row(
            "SELECT title FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|_| DomainError::NotFound {
            entity: "Job".into(),
            id: job_id.to_string(),
        })?
    };
    let root = watch::cv_root(state);
    let path = job_ad::default_path(&root, &job_slug(&title));
    if path.as_os_str().is_empty() {
        return Err(DomainError::ValidationError {
            message: "Cannot resolve a job-ad location: no CV root is configured".into(),
        });
    }
    let existed = path.exists();
    job_ad::ensure_file(&path).map_err(|e| DomainError::ValidationError { message: e })?;
    let path_str = path.to_string_lossy().to_string();
    {
        let conn = state.db.0.lock().map_err(lock_err)?;
        conn.execute(
            "UPDATE jobs SET job_ad_path = ?1 WHERE id = ?2",
            rusqlite::params![&path_str, job_id],
        )
        .map_err(libtriqui_core::db::db_err)?;
    }
    Ok((path, !existed))
}

/// Read the job's ad file. Seeds the default (empty) file on first call and
/// keeps the watcher armed on it.
pub fn get_job_ad(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let (path, created) = resolve_path(state, job_id).map_err(|e| domain(&e))?;
    let text = job_ad::read_text(&path).map_err(|e| DomainError::ValidationError { message: e })?;
    watch::watch_job_ad(state, job_id, &path);
    Ok(serde_json::to_value(JobAdFile {
        path: path.to_string_lossy().to_string(),
        text,
        created,
    })
    .map_err(|e| string_err(e.to_string()))?)
}

/// "Change…": point the job at a different ad file, then adopt its contents.
pub fn set_job_ad_path(
    state: &Arc<ServeState>,
    job_id: &str,
    path: &str,
) -> Result<Value, ProtocolError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(domain(&DomainError::ValidationError {
            message: "Job-ad path cannot be empty".into(),
        }));
    }
    let new_path = PathBuf::from(trimmed);
    if !new_path.is_file() {
        return Err(domain(&DomainError::ValidationError {
            message: format!("Not a file: {trimmed}"),
        }));
    }
    {
        let conn = state.db.0.lock().map_err(lock_err)?;
        conn.execute(
            "UPDATE jobs SET job_ad_path = ?1 WHERE id = ?2",
            rusqlite::params![trimmed, job_id],
        )
        .map_err(libtriqui_core::db::db_err)?;
    }
    watch::watch_job_ad(state, job_id, &new_path);
    // Adopt the new file's text and, since that is a ranking input, kick the same
    // full re-rank a disk edit would. Without this the job would keep scoring
    // against the *previous* file's ad until the user noticed and re-ranked.
    reload_and_rerank(state, job_id);
    get_job_ad(state, job_id)
}

/// Adopt the file's contents into `job_ad_text` when they differ. Bumps
/// `inputs_version` so the job reads as dirty and a rank is warranted.
/// Returns whether the ad changed. Shared by the watcher and the "Change" path.
fn adopt_text(state: &Arc<ServeState>, job_id: &str, path: &Path) -> Result<bool, String> {
    let text = job_ad::read_text(path)?;
    let stored: Option<String> = {
        let conn = state.db.0.lock().map_err(|e| e.to_string())?;
        conn.query_row(
            "SELECT job_ad_text FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?
    };
    if stored == job_ad::text_to_db(&text) {
        return Ok(false);
    }
    let conn = state.db.0.lock().map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE jobs SET job_ad_text = ?1, inputs_version = inputs_version + 1, \
         inputs_updated_at = datetime('now'), updated_at = datetime('now') WHERE id = ?2",
        rusqlite::params![job_ad::text_to_db(&text), job_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(true)
}

/// The ad file changed on disk: reload it and, if the text really changed, run a
/// **full** re-rank — the ad is the heaviest ranking input, so a partial stage
/// would leave the results stale. Called from the watcher thread.
pub fn reload_and_rerank(state: &Arc<ServeState>, job_id: &str) {
    let path: Option<String> = {
        let conn = match state.db.0.lock() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("ad-watch {job_id}: db lock failed: {e}");
                return;
            }
        };
        conn.query_row(
            "SELECT job_ad_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .ok()
        .flatten()
    };
    let Some(path) = path.filter(|p| !p.is_empty()) else {
        return;
    };
    let path = PathBuf::from(path);

    let changed = match adopt_text(state, job_id, &path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("ad-watch {job_id}: cannot read {}: {e}", path.display());
            return;
        }
    };
    if !changed {
        // Touch/rename noise, or an editor rewriting identical bytes.
        return;
    }
    log::info!(
        "ad-watch {job_id}: ad changed in {}, re-ranking",
        path.display()
    );

    // Tell the gui the file it is previewing moved under it. `rank-progress`
    // only fires once a cascade actually starts, which never happens on a job
    // with no extracted CVs — so the preview would otherwise sit stale there.
    state
        .events
        .emit("job-ad-reloaded", &serde_json::json!({ "job_id": job_id }));

    let extracted: i64 = {
        let conn = match state.db.0.lock() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("ad-watch {job_id}: db lock failed: {e}");
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
        // Nothing to rank yet; the ad is stored and will be used by the first
        // rank (import auto-starts the cascade).
        return;
    }
    match crate::serve::handlers::rank::spawn_cascade(state, job_id) {
        Ok(true) => log::info!("ad-watch {job_id}: full re-rank started"),
        Ok(false) => log::info!("ad-watch {job_id}: a rank was already running"),
        Err(e) => log::warn!("ad-watch {job_id}: could not re-rank: {e}"),
    }
}

/// Re-read the file into the DB without re-ranking. Used by the gui after the
/// user edits the file in an external editor and comes back to the window.
pub fn reload_job_ad(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let (path, _) = resolve_path(state, job_id).map_err(|e| domain(&e))?;
    adopt_text(state, job_id, &path).map_err(string_err)?;
    let conn = state.db.0.lock().map_err(lock_err)?;
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(libtriqui_core::db::db_err)?;
    Ok(serde_json::to_value(job).map_err(|e| string_err(e.to_string()))?)
}
