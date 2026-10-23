//! `serve` — the long-lived request loop behind the gui (thin gui pattern,
//! `gui/design/app1/thin-gui.md`).
//!
//! One JSON object per line on the request stream (a [`Request`]) and the
//! response stream (a [`Response`] or an [`Event`]). Two transports:
//! - stdio (the `triqui-cli serve` subcommand: tests, humans, scripts);
//! - in-process duplex pipes inside the gui binary via
//!   [`crate::inprocess::InProcessServe`] — the gui's production path, no
//!   child process.
//!
//! Concurrency rules:
//!
//! - the request loop never runs command bodies inline — each request is
//!   handled on its own thread, so one long command (a rank, a download) never
//!   delays another (the "edit a job ad while an import runs" case);
//! - long operations either respond immediately with a handle
//!   (`import_cv_directory`, `embed_job`) or keep their request open until the
//!   work finishes — progress always travels as events, never in the response;
//! - a dedicated writer thread owns the response stream so response/event
//!   lines from concurrent threads never interleave;
//! - all logs go through the `log` crate (stderr for the stdio shell, the
//!   gui's Tauri backend in-process); the response stream is protocol only.

pub mod handlers;
pub mod watch;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use libtriqui_client::protocol::{Request, MAX_LINE_BYTES};
use libtriqui_client::types::RuntimeStatusInfo;
use serde::Serialize;
use libtriqui_core::db::{self, DbConnection, ImportState};
use libtriqui_core::runtime::{EngineRuntime, RuntimeConfig};

use crate::serve::watch::WatchHandle;

/// Quiet period after the last filesystem event before a CV-dir scan runs.
pub const WATCH_DEBOUNCE: Duration = Duration::from_millis(1000);

/// Progress events are throttled to at most this many emissions per second
/// per source — the pipe stays quiet even for huge imports.
pub const PROGRESS_MIN_INTERVAL: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------
// State + event sink
// ---------------------------------------------------------------------------

/// Cloneable handle that serializes and queues one event line for the writer
/// thread. Event names and payloads are identical to the pre-migration Tauri
/// events, so frontend listeners are unchanged.
#[derive(Clone)]
pub struct EventSink(std::sync::mpsc::Sender<String>);

impl EventSink {
    pub fn emit<D: Serialize>(&self, name: &str, data: &D) {
        let event = libtriqui_client::Event {
            event: name.to_string(),
            data: serde_json::to_value(data).unwrap_or(serde_json::Value::Null),
        };
        if let Ok(line) = serde_json::to_string(&event) {
            let _ = self.0.send(line);
        }
    }

    /// Raw line (already serialized) — used by the writer thread itself.
    pub fn send_line(&self, line: String) {
        let _ = self.0.send(line);
    }
}

/// Shared serve state. Immutable after construction; all fields are handles
/// with interior mutability.
pub struct ServeState {
    /// App-data dir: `triqui.db`, `preferences.json`, `recovery/`,
    /// extracted `benchmark-cvs/`.
    pub app_dir: PathBuf,
    /// Host app version (the gui's), reported in exported job bundles.
    pub app_version: String,
    pub db: DbConnection,
    pub db_path: PathBuf,
    pub import_state: ImportState,
    pub runtime: Arc<EngineRuntime>,
    pub events: EventSink,
    pub watch: WatchHandle,
    /// Cascade config for progressive ranking (from `cascade:` in `triq.yaml`).
    pub config: libtriqui_core::config::CascadeSection,
    /// Live cascade runners per job (see [`handlers::rank::CascadeGens`]).
    pub cascade_gens: Arc<crate::serve::handlers::rank::CascadeGens>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the stdio serve loop until stdin EOF (the gui closing stdin is the
/// shutdown signal), then tear the engine down and return. Thin shell for the
/// `triqui-cli serve` subcommand: only logger setup, then [`run_over`].
pub fn run(app_dir: Option<PathBuf>, app_version: String) -> Result<(), String> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init()
        .ok();
    run_over(
        app_dir,
        app_version,
        std::io::stdin().lock(),
        std::io::stdout(),
    )
}

/// The serve loop over arbitrary byte streams — the library core the gui runs
/// in-process (one binary, no child; `gui/design/app1/thin-gui.md`). `input`/`output`
/// are the request/response channels; EOF on `input` shuts the engine down.
/// Logging is the caller's business: the stdio shell inits `env_logger`, the
/// in-process gui keeps its own Tauri log backend.
pub fn run_over<R: BufRead, W: Write + Send + 'static>(
    app_dir: Option<PathBuf>,
    app_version: String,
    input: R,
    output: W,
) -> Result<(), String> {
    let app_dir = match app_dir {
        Some(dir) => dir,
        None => default_app_dir(),
    };
    std::fs::create_dir_all(&app_dir).map_err(|e| format!("cannot create app dir: {e}"))?;
    let db_path = app_dir.join("triqui.db");
    let conn = db::open_connection_at(&db_path)?;

    // No cascade thread exists yet, so a rank_state row still claiming a run in
    // flight is a leftover from a previous process (quit/crash mid-cascade).
    // Recover it before anything reads the state, or the job reports running
    // forever and refuses every new cascade (the wedged-job bug).
    match db::recover_stale_running_cascades(&conn) {
        Ok(0) => {}
        Ok(n) => log::info!("serve: recovered {n} stale running cascade(s) from a previous run"),
        Err(e) => log::warn!("serve: could not recover stale cascade states: {e}"),
    }

    // Per-thread config override (see `InProcessServe::start_with_config`):
    // this loop thread has it; workers spawned below don't inherit
    // thread-locals, so each re-applies the captured value.
    let cfg_override = libtriqui_core::app_config::config_path_override();
    let user_overrides = user_overrides_from_prefs(&app_dir);
    let user_ovr_opt = if user_overrides.is_default() {
        None
    } else {
        Some(user_overrides)
    };
    libtriqui_core::app_config::set_user_overrides(user_ovr_opt.clone());

    // Model resolution mirrors the pre-migration gui: env var, then the saved
    // preference, then the embedded default (inside `RuntimeConfig::discover`).
    let model = std::env::var("TRIQI_EMBED_MODEL")
        .ok()
        .or_else(|| prefs_default_model(&app_dir));
    let runtime = EngineRuntime::new(RuntimeConfig::discover(model), db_path.clone());

    // Outbound queue + dedicated writer thread (invariant: only it touches
    // the output stream, so lines never interleave and a blocked write never
    // stalls the request loop).
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::Builder::new()
        .name("serve-writer".into())
        .spawn(move || {
            let mut out = output;
            for line in rx {
                if writeln!(out, "{line}").and_then(|_| out.flush()).is_err() {
                    break; // gui is gone
                }
            }
        })
        .map_err(|e| format!("spawn writer: {e}"))?;
    let events = EventSink(tx);

    // Forward engine status changes exactly like the pre-migration gui hook.
    {
        let events = events.clone();
        runtime.set_status_hook(Arc::new(move |info: &RuntimeStatusInfo| {
            events.emit("runtime-status", info);
        }));
    }
    // Watch for engine inactivity and free the model memory when idle.
    runtime.spawn_idle_monitor();

    // CV-dir watching is demand-driven (single-watcher policy): the gui sends
    // the `watch_job` rpc whenever the viewed job changes, so only the job on
    // screen is watched and auto-imported. The coordinator gets the Arc only
    // after the state is fully built (see below).
    let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<String>();
    let (ad_tx, ad_rx) = std::sync::mpsc::channel::<String>();

    let state = Arc::new(ServeState {
        app_dir,
        app_version,
        db: DbConnection(Mutex::new(conn)),
        db_path,
        import_state: ImportState::new(),
        runtime,
        events: events.clone(),
        watch: WatchHandle::new(jobs_tx, ad_tx),
        config: libtriqui_core::app_config::cascade_config(),
        cascade_gens: Arc::new(crate::serve::handlers::rank::CascadeGens::new()),
    });

    {
        let state = Arc::clone(&state);
        let cfg_override = cfg_override.clone();
        let user_ovr = user_ovr_opt.clone();
        std::thread::Builder::new()
            .name("cv-watch-coordinator".into())
            .spawn(move || {
                libtriqui_core::app_config::set_config_path_override(cfg_override);
                libtriqui_core::app_config::set_user_overrides(user_ovr);
                watch::coordinate(state, jobs_rx)
            })
            .map_err(|e| format!("spawn watch coordinator: {e}"))?;
    }

    {
        let state = Arc::clone(&state);
        let cfg_override = cfg_override.clone();
        let user_ovr = user_ovr_opt.clone();
        std::thread::Builder::new()
            .name("ad-watch-coordinator".into())
            .spawn(move || {
                libtriqui_core::app_config::set_config_path_override(cfg_override);
                libtriqui_core::app_config::set_user_overrides(user_ovr);
                watch::coordinate_ad(state, ad_rx)
            })
            .map_err(|e| format!("spawn ad watch coordinator: {e}"))?;
    }

    log::info!("serve: ready");

    // Request loop: read, spawn a handler thread, never block on a command.
    // Handler panics are caught and answered with an error envelope — in the
    // in-process gui a panic must degrade to a failed command, never take the
    // app down (note: this needs the host binary to unwind, i.e. not
    // `panic = "abort"`).
    let mut lines = input.lines();
    while let Some(line) = lines.next() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.len() as u64 > MAX_LINE_BYTES {
            log::error!("serve: over-long request line dropped");
            continue;
        }
        let request = match serde_json::from_str::<Request>(trimmed) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("serve: unparsable request line: {e}");
                continue;
            }
        };
        let state = Arc::clone(&state);
        let events = events.clone();
        let cfg_override = cfg_override.clone();
        let user_ovr = user_ovr_opt.clone();
        std::thread::Builder::new()
            .name(format!("cmd-{}", request.cmd))
            .spawn(move || {
                libtriqui_core::app_config::set_config_path_override(cfg_override);
                libtriqui_core::app_config::set_user_overrides(user_ovr);
                let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || handlers::handle(&state, &request),
                )) {
                    Ok(response) => response,
                    Err(panic) => {
                        let msg = panic_message(&panic);
                        log::error!("serve: handler for '{}' panicked: {msg}", request.cmd);
                        libtriqui_client::protocol::Response::err(
                            request.id,
                            libtriqui_client::protocol::ProtocolError::internal(format!(
                                "internal error in {}: {msg}",
                                request.cmd
                            )),
                        )
                    }
                };
                events.send_line(serde_json::to_string(&response).expect("response serializes"));
            })
            .ok();
    }

    log::info!("serve: stdin closed — shutting down");
    state.runtime.shutdown();
    Ok(())
}

/// Default app dir, mirroring the Tauri app-data layout:
/// `<data_root>/com.oaltun.triqui` (see `db::default_db_path`).
fn default_app_dir() -> PathBuf {
    db::default_db_path()
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Best-effort panic payload → message (mirrors std's panic reporting).
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// The embedding-model default from the saved preferences (env var is checked
/// by the caller first, `triq.yaml` inside `RuntimeConfig::discover` last).
fn prefs_default_model(app_dir: &std::path::Path) -> Option<String> {
    let prefs = load_app_preferences(app_dir);
    prefs.embedding_model
}

/// Convert saved AppPreferences engine overrides into the libconfigreader
/// `UserOverrides` shape used by `Config::with_user_overrides()`.
fn user_overrides_from_prefs(app_dir: &std::path::Path) -> libconfigreader::config::UserOverrides {
    let prefs = load_app_preferences(app_dir);
    libconfigreader::config::UserOverrides {
        engine_backend: prefs.engine_backend,
        engine_exec_provider: prefs.engine_exec_provider,
        engine_embed_batch: prefs.engine_embed_batch.map(|v| v as usize),
        engine_intra_threads: prefs.engine_intra_threads.map(|v| v as usize),
        engine_inter_threads: prefs.engine_inter_threads.map(|v| v as usize),
        engine_embed_sessions: None,
        engine_provider_logging: None,
        engine_max_cores: prefs.engine_max_cores.map(|v| v as usize),
        engine_no_cache: prefs.engine_no_cache,
    }
}

/// Load AppPreferences from the preferences.json in app_dir.
/// Returns defaults if the file is missing or unparseable.
fn load_app_preferences(app_dir: &std::path::Path) -> libtriqui_client::types::AppPreferences {
    let path = app_dir.join("preferences.json");
    let contents = std::fs::read_to_string(&path).ok();
    match contents {
        Some(c) => serde_json::from_str(&c).unwrap_or_default(),
        None => libtriqui_client::types::AppPreferences::default(),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used by handlers + watcher
// ---------------------------------------------------------------------------

/// Throttling state for one progress source (e.g. one import).
#[derive(Default)]
pub struct ProgressThrottle {
    last: Mutex<Option<std::time::Instant>>,
}

impl ProgressThrottle {
    /// True when enough time has passed since the last allowed emission.
    pub fn allow(&self) -> bool {
        let mut last = self.last.lock().expect("throttle lock");
        let now = std::time::Instant::now();
        match *last {
            Some(t) if now.duration_since(t) < PROGRESS_MIN_INTERVAL => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }
}

/// Parse one `Inbound` line (used by tests).
#[cfg(test)]
fn parse_line(line: &str) -> Result<libtriqui_client::protocol::Inbound, serde_json::Error> {
    serde_json::from_str(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_response_and_event_lines() {
        assert!(parse_line(r#"{"id":1,"ok":{}}"#).is_ok());
        assert!(parse_line(r#"{"event":"e","data":{}}"#).is_ok());
        assert!(parse_line("not json").is_err());
    }

    #[test]
    fn progress_throttle_passes_first_then_limits_rate() {
        // The backpressure the gui relies on (progressive-ranking §10: ≤10 Hz):
        // the first emission of a burst passes, immediate follow-ups are
        // dropped, and the next emission lands once the min interval elapsed.
        // An unthrottled per-candidate source froze the gui mid-embed.
        let throttle = ProgressThrottle::default();
        assert!(throttle.allow(), "first emission of a burst passes");
        assert!(!throttle.allow(), "immediate follow-up is throttled");
        assert!(!throttle.allow(), "still within the interval");
        std::thread::sleep(PROGRESS_MIN_INTERVAL + std::time::Duration::from_millis(5));
        assert!(throttle.allow(), "next emission after the interval passes");
    }
}
