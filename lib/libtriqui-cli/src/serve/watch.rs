//! Watched CV directories: auto-import new CVs as they appear (serve side).
//!
//! Port of the pre-migration gui `watch.rs` — identical behaviour, but events
//! go through the import pipeline's [`crate::serve::EventSink`] instead of
//! Tauri and the DB is the serve process's shared connection. Only ONE job is
//! watched at a time (the one currently open in the gui): whenever the viewed
//! job changes, the gui sends the `watch_job` rpc, which drops every other
//! watcher, watches that job's directory, and fires a catch-up scan — so
//! files added while the job was not viewed (or the app was closed) are
//! imported too, and "Import CVs" stays fast. The scan is cheap when nothing
//! is new: supported files already present in the `candidates` table are
//! filtered out before the import pipeline (and thus the embedding engine) is
//! ever touched. Files added while another job is viewed are picked up by the
//! catch-up scan on the next switch (or the manual "Import CVs" button).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use notify::RecursiveMode;
use notify::Watcher;

use crate::serve::handlers::import::spawn_dir_import;
use crate::serve::handlers::prefs;
use crate::serve::ServeState;
use crate::serve::WATCH_DEBOUNCE;

/// Root folder for new jobs' default CV directories (`<root>/job-N/cv`), first
/// match wins: the saved gui preference (Settings → General → CV directory,
/// stored in `preferences.json`), `gui_cv_dir_root` from `triq.yaml`, else
/// the platform Documents dir + `jobs` (`~/Documents/jobs`, Windows Known
/// Folder equivalent, XDG on Linux). Empty when not even a home dir can be
/// found.
pub fn cv_root(state: &ServeState) -> PathBuf {
    resolve_cv_root(prefs::load_app_preferences(state).cv_dir_root)
}

/// Pure resolution behind [`cv_root`]: a non-empty preference string wins,
/// then `triq.yaml`'s `gui_cv_dir_root`, then the platform default.
fn resolve_cv_root(pref: Option<String>) -> PathBuf {
    pref.map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(libtriqui_core::app_config::cv_dir_root)
        .or_else(|| dirs::document_dir().map(|d| d.join("jobs")))
        .unwrap_or_default()
}

/// Default CV directory for a new job titled `title`: `<root>/title/cv` — the
/// job's own folder under the root with CVs inside its `cv/` subfolder, so a
/// job folder can hold non-CV material next to the watched directory. Empty
/// string when no root could be resolved (job then has no CV dir).
pub fn default_job_dir(root: &Path, title: &str) -> String {
    if root.as_os_str().is_empty() {
        return String::new();
    }
    root.join(title).join("cv").to_string_lossy().to_string()
}

/// Per-serve watch state: one OS watcher per job + the debounce channel
/// shared by all watchers' event handlers.
pub struct WatchHandle {
    watchers: Mutex<HashMap<String, notify::RecommendedWatcher>>,
    jobs_tx: mpsc::Sender<String>,
    /// Job-ad file watchers, kept apart from the CV watchers on purpose: a save
    /// in the user's editor must reload the ad and re-rank, never kick off a CV
    /// import scan (and vice versa). Separate map because `watch_dir` drops the
    /// previous watcher stored under the job id.
    ad_watchers: Mutex<HashMap<String, notify::RecommendedWatcher>>,
    ad_tx: mpsc::Sender<String>,
}

impl WatchHandle {
    pub fn new(jobs_tx: mpsc::Sender<String>, ad_tx: mpsc::Sender<String>) -> Self {
        Self {
            watchers: Mutex::new(HashMap::new()),
            jobs_tx,
            ad_watchers: Mutex::new(HashMap::new()),
            ad_tx,
        }
    }
}

/// Watch the job-ad **file** for `job_id`, replacing any previous ad watcher.
/// The file is created empty when missing, so the user always has something to
/// open. Failures are logged, never propagated: an unwritable ad file must not
/// stop the app from watching CVs.
pub fn watch_job_ad(state: &Arc<ServeState>, job_id: &str, path: &Path) {
    let mut watchers = state
        .watch
        .ad_watchers
        .lock()
        .expect("ad watch lock poisoned");
    let _ = watchers.remove(job_id);
    if path.as_os_str().is_empty() {
        return;
    }
    if let Err(e) = libtriqui_core::job_ad::ensure_file(path) {
        log::warn!("ad-watch {job_id}: cannot create {}: {e}", path.display());
        return;
    }
    let tx = state.watch.ad_tx.clone();
    let jid = job_id.to_string();
    let watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                if matches!(event.kind, notify::event::EventKind::Access(_)) {
                    return;
                }
                let _ = tx.send(jid.clone());
            }
        });
    match watcher {
        Ok(mut w) => match w.watch(path, RecursiveMode::NonRecursive) {
            Ok(()) => {
                watchers.insert(job_id.to_string(), w);
                log::info!("ad-watch: watching {} for {job_id}", path.display());
            }
            Err(e) => log::warn!("ad-watch {job_id}: cannot watch {}: {e}", path.display()),
        },
        Err(e) => log::warn!("ad-watch {job_id}: cannot create watcher: {e}"),
    }
}

/// Drop the job-ad watcher for `job_id` (job deleted, or the ad path cleared).
pub fn stop_watching_ad(state: &Arc<ServeState>, job_id: &str) {
    if state
        .watch
        .ad_watchers
        .lock()
        .expect("ad watch lock poisoned")
        .remove(job_id)
        .is_some()
    {
        log::info!("ad-watch: stopped watching for {job_id}");
    }
}

/// Debounce loop for job-ad edits: collect job ids until the stream is quiet for
/// [`WATCH_DEBOUNCE`] (an editor save can emit several events), then reload
/// each job's ad once.
pub fn coordinate_ad(state: Arc<ServeState>, rx: mpsc::Receiver<String>) {
    while let Ok(first) = rx.recv() {
        let mut pending: HashSet<String> = HashSet::from([first]);
        while let Ok(job_id) = rx.recv_timeout(WATCH_DEBOUNCE) {
            pending.insert(job_id);
        }
        for job_id in pending {
            crate::serve::handlers::job_ad::reload_and_rerank(&state, &job_id);
        }
    }
}

/// Single debounce loop: collect job ids until the event stream is quiet for
/// [`WATCH_DEBOUNCE`], then run one scan per distinct job.
pub fn coordinate(state: Arc<ServeState>, rx: mpsc::Receiver<String>) {
    while let Ok(first) = rx.recv() {
        let mut pending: HashSet<String> = HashSet::from([first]);
        while let Ok(job_id) = rx.recv_timeout(WATCH_DEBOUNCE) {
            pending.insert(job_id);
        }
        for job_id in pending {
            scan_and_import(&state, &job_id);
        }
    }
}

/// Debounced event handler for one job: import not-yet-imported supported
/// files in the job's watched directory. No-op when the dir is unset/missing,
/// the job is already importing, or nothing new is on disk.
fn scan_and_import(state: &Arc<ServeState>, job_id: &str) {
    let dir: Option<String> = {
        let conn = state.db.0.lock().expect("db lock poisoned");
        conn.query_row(
            "SELECT import_directory_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .ok()
    };
    let Some(dir) = dir.filter(|d| !d.is_empty()) else {
        return;
    };

    // Skip when this job is already importing; the next event re-triggers.
    if state
        .import_state
        .0
        .lock()
        .expect("import state lock poisoned")
        .contains(job_id)
    {
        log::info!("cv-watch {job_id}: import already active, skipping scan");
        return;
    }

    let files = match libtriqui_core::import::collect_supported_files(&dir) {
        Ok(files) => files,
        Err(e) => {
            log::warn!("cv-watch {job_id}: scan of {dir} failed: {e}");
            return;
        }
    };
    if files.is_empty() {
        return;
    }

    // Cheap pre-check: drop files already successfully imported (failed ones are
    // left through so they retry) so unrelated filesystem noise (touch, editor
    // temp files, attribute changes) never starts the embedding engine.
    let new_files: Vec<PathBuf> = {
        let conn = state.db.0.lock().expect("db lock poisoned");
        files
            .into_iter()
            .filter(|p| {
                let path_str = p.to_string_lossy().to_string();
                !conn
                    .query_row(
                        "SELECT 1 FROM candidates WHERE job_id = ?1 AND source_file_path = ?2 AND extraction_status = 'Extracted'",
                        rusqlite::params![job_id, &path_str],
                        |_| Ok(true),
                    )
                    .unwrap_or(false)
            })
            .collect()
    };
    if new_files.is_empty() {
        return;
    }

    log::info!(
        "cv-watch {job_id}: {} new file(s) in {dir}",
        new_files.len()
    );
    if let Err(e) = spawn_dir_import(state, job_id, &dir, new_files) {
        log::warn!("cv-watch {job_id}: auto-import failed: {e}");
    }
}

/// Watch `dir` for `job_id`, replacing any previous watcher for the job.
/// Creates the directory when missing, so typing a fresh path just works.
/// An empty `dir` stops watching. Called with `directory` already trimmed.
pub fn watch_dir(
    state: &Arc<ServeState>,
    job_id: &str,
    dir: &str,
) -> Result<(), libtriqui_core::types::DomainError> {
    let mut watchers = state.watch.watchers.lock().expect("cv watch lock poisoned");
    // Drop the old watcher first: dropping removes the OS watch.
    let _ = watchers.remove(job_id);
    if dir.is_empty() {
        return Ok(());
    }
    let path = PathBuf::from(dir);
    if let Err(e) = std::fs::create_dir_all(&path) {
        return Err(libtriqui_core::types::DomainError::ValidationError {
            message: format!("Cannot create directory {dir}: {e}"),
        });
    }
    let tx = state.watch.jobs_tx.clone();
    let jid = job_id.to_string();
    let mut watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                // Reads alone (access events) never mean new content.
                if matches!(event.kind, notify::event::EventKind::Access(_)) {
                    return;
                }
                let _ = tx.send(jid.clone());
            }
        })
        .map_err(|e| libtriqui_core::types::DomainError::ValidationError {
            message: format!("Cannot watch {dir}: {e}"),
        })?;
    watcher
        .watch(&path, RecursiveMode::NonRecursive)
        .map_err(|e| libtriqui_core::types::DomainError::ValidationError {
            message: format!("Cannot watch {dir}: {e}"),
        })?;
    watchers.insert(job_id.to_string(), watcher);
    log::info!("cv-watch: watching {dir} for {job_id}");
    Ok(())
}

/// Stop watching `job_id`'s directory (job deleted or dir cleared).
pub fn stop_watching(state: &Arc<ServeState>, job_id: &str) {
    if state
        .watch
        .watchers
        .lock()
        .expect("cv watch lock poisoned")
        .remove(job_id)
        .is_some()
    {
        log::info!("cv-watch: stopped watching for {job_id}");
    }
}

/// Single-watcher policy: make `job_id` the ONLY watched job. Drops every
/// watcher, watches the job's CV directory (creating it when missing), and
/// fires one debounced catch-up scan so files dropped while the job was not
/// viewed are imported too. No-op logging when the job has no CV dir set.
/// Called by the `watch_job` rpc (gui switches jobs), `create_job` (new jobs
/// start fresh + viewed), `set_job_cv_dir` (Browse… selection), and bundle
/// import.
pub fn watch_job(state: &Arc<ServeState>, job_id: &str) {
    state
        .watch
        .watchers
        .lock()
        .expect("cv watch lock poisoned")
        .clear();
    // The ad file travels with the job view: same single-watcher policy, so the
    // ad of the job being looked at is the one being watched.
    state
        .watch
        .ad_watchers
        .lock()
        .expect("ad watch lock poisoned")
        .clear();
    let dir: Option<String> = {
        let conn = state.db.0.lock().expect("db lock poisoned");
        conn.query_row(
            "SELECT import_directory_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .ok()
    };
    let ad_path: Option<String> = {
        let conn = state.db.0.lock().expect("db lock poisoned");
        conn.query_row(
            "SELECT job_ad_path FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .ok()
        .flatten()
    };
    if let Some(ad) = ad_path.filter(|p| !p.is_empty()) {
        watch_job_ad(state, job_id, Path::new(&ad));
        // Catch-up read, mirroring the CV side: the user may have edited (and
        // saved) the ad while another job was being viewed, and no filesystem
        // event is coming now that the file is no longer changing.
        let _ = state.watch.ad_tx.send(job_id.to_string());
    }
    let Some(dir) = dir.filter(|d| !d.is_empty()) else {
        log::info!("cv-watch {job_id}: no CV directory set, nothing to watch");
        return;
    };
    if let Err(e) = watch_dir(state, job_id, &dir) {
        log::warn!("cv-watch {job_id}: cannot watch {dir}: {e}");
        return;
    }
    // Catch-up scan (debounced like watcher events): imports files already in
    // the dir before any new filesystem event arrives.
    let _ = state.watch.jobs_tx.send(job_id.to_string());
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{default_job_dir, resolve_cv_root};

    #[test]
    fn non_empty_preference_wins() {
        assert_eq!(
            resolve_cv_root(Some("/tmp/cv-root".into())),
            PathBuf::from("/tmp/cv-root")
        );
        // Whitespace-only values are treated as unset, not as a relative path.
        assert_eq!(resolve_cv_root(Some("   ".into())), resolve_cv_root(None));
    }

    #[test]
    fn unset_preference_falls_through_to_a_resolvable_default() {
        // The repo config ships with `gui_cv_dir_root` commented out, so this
        // exercises the platform fallback (Documents + jobs); either way the
        // root must never be empty on a machine with a home dir.
        assert!(!resolve_cv_root(None).as_os_str().is_empty());
    }

    #[test]
    fn default_job_dir_joins_root_title_and_cv() {
        assert_eq!(
            default_job_dir(Path::new("/tmp/cv-root"), "job-7"),
            "/tmp/cv-root/job-7/cv"
        );
        // No resolvable root → no default dir (job keeps an empty CV dir).
        assert_eq!(default_job_dir(Path::new(""), "job-7"), "");
    }
}
