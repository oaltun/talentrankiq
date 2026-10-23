//! The gui's in-process serve connection: one binary, no child process.
//!
//! [`InProcessServe::start`] runs the [`serve::run_over`] loop on a dedicated
//! thread, wired to a [`TriqiClient`] through the in-memory duplex
//! transport (`libtriqui_client::transport`). Protocol, DTOs, event names,
//! and teardown semantics are byte-for-byte the stdio child's: dropping the
//! client's write end is EOF for the loop, which shuts the engine down —
//! exactly what "the gui closes the child's stdin" used to do.
//!
//! This is used by **tests** (e.g. gui/src-tauri/tests/e2e.rs) to run the
//! serve loop in-process with pinned config. The production GUI now uses
//! the subprocess bridge (`gui/src-tauri/src/cli_subprocess.rs`).

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use libtriqui_client::transport;
use libtriqui_client::TriqiClient;

use crate::serve;

/// An in-process serve loop plus its client half. The loop tears the engine
/// down (modelcaller off, ORT wrapper thread ended, DB closed) when dropped
/// client-side EOF reaches it — [`InProcessServe::shutdown`] waits for that
/// teardown so app exit doesn't strand the engine.
pub struct InProcessServe {
    client: TriqiClient,
    handle: Option<thread::JoinHandle<()>>,
}

impl InProcessServe {
    /// Start the serve loop on a thread and return the connected client.
    /// `app_dir` holds the DB/prefs/recovery/benchmark-cvs (the Tauri
    /// app-data dir); `app_version` rides along in exported job bundles.
    ///
    /// Config resolution is the usual discovery (env var, then repo root).
    /// Use [`InProcessServe::start_with_config`] to pin the serve thread to a
    /// specific `triq.yaml` without process-global state.
    pub fn start(app_dir: PathBuf, app_version: String) -> Result<Self, String> {
        Self::start_inner(app_dir, app_version, None)
    }

    /// Like [`InProcessServe::start`], but every config lookup made from the
    /// serve thread (handlers + engine runtime) resolves `config_path` —
    /// scoped to that thread, so several serves with different configs can
    /// coexist in one process (contract tests, tools).
    pub fn start_with_config(
        app_dir: PathBuf,
        app_version: String,
        config_path: PathBuf,
    ) -> Result<Self, String> {
        Self::start_inner(app_dir, app_version, Some(config_path))
    }

    fn start_inner(
        app_dir: PathBuf,
        app_version: String,
        config_override: Option<PathBuf>,
    ) -> Result<Self, String> {
        // Side a = the host (client), side b = the serve loop: a_write →
        // b_read carries requests, b_write → a_read carries responses/events.
        let (a_write, a_read, b_write, b_read) = transport::duplex();
        let handle = thread::Builder::new()
            .name("triqui-serve".into())
            .spawn(move || {
                if let Some(p) = config_override {
                    libtriqui_core::app_config::set_config_path_override(Some(p));
                }
                if let Err(e) = serve::run_over(Some(app_dir), app_version, b_read, b_write) {
                    log::error!("in-process serve loop failed: {e}");
                }
                libtriqui_core::app_config::set_config_path_override(None);
            })
            .map_err(|e| format!("spawn serve thread: {e}"))?;
        let client = TriqiClient::over(Box::new(a_read), Box::new(a_write));
        Ok(Self {
            client,
            handle: Some(handle),
        })
    }

    /// The connected client (cloneable; shares ids, pipe, event channel).
    pub fn client(&self) -> TriqiClient {
        self.client.clone()
    }

    /// Signal EOF to the serve loop and wait (bounded) for its teardown —
    /// engine shutdown includes ending the in-process `onnxwrapperrs`
    /// wrapper thread, so app exit must not skip it. Safe to call at most
    /// once; the final drop of the client covers any earlier miss.
    pub fn shutdown(mut self) {
        self.client.shutdown();
        if let Some(handle) = self.handle.take() {
            let (tx, rx) = mpsc::channel();
            thread::Builder::new()
                .name("serve-join".into())
                .spawn(move || {
                    let _ = handle.join();
                    let _ = tx.send(());
                })
                .ok();
            // The teardown chain is bounded but not instant: serve loop end →
            // runtime shutdown → modelcaller drain → wrapper thread finishing
            // any in-flight session load (seconds warm, tens cold — see the
            // 120 s wrapper bound in modelcaller). Exiting before it completes
            // is exactly the exit-time ORT SIGSEGV, so wait generously; a
            // wedged handler still cannot hang app exit forever.
            let _ = rx.recv_timeout(Duration::from_secs(150));
        }
    }
}