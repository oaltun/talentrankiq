//! Subprocess bridge: owns the `triq serve` child process and the JSONL
//! protocol client wired to its stdin/stdout.
//!
//! This is the same transport the Vite dev proxy uses for the browser e2e
//! path (spawn `triq serve`, one JSON object per line), so the Tauri app and
//! the Playwright tests speak an identical protocol to an identical process
//! with an identical environment.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use libtriqui_client::TriqiClient;
use tauri::Manager;

/// Managed state: the `triq serve` child plus its protocol client.
///
/// `spawn` populates both; a failed spawn leaves them empty so commands
/// report "bridge not running" instead of panicking on missing state.
pub struct SubprocessBridge {
    client: Mutex<Option<TriqiClient>>,
    child: Mutex<Option<Child>>,
}

impl SubprocessBridge {
    /// Spawn `triq serve` and connect the JSONL client to its stdio.
    pub fn spawn(app: &tauri::AppHandle) -> Result<Self, String> {
        let triq_bin = triq_binary_path(app)?;
        // Same layout as the dev proxy: <app_data_dir>/<identifier>/triq.
        let app_dir = app
            .path()
            .app_data_dir()
            .map_err(|e| format!("app data dir: {e}"))?
            .join("triq");
        Self::spawn_at(&triq_bin, &app_dir)
    }

    /// Spawn `triq serve` from an explicit binary and app dir.
    fn spawn_at(triq_bin: &Path, app_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(app_dir)
            .map_err(|e| format!("cannot create app dir {}: {e}", app_dir.display()))?;

        log::info!(
            "spawning triq serve: {} (TRIQ_APP_DIR={})",
            triq_bin.display(),
            app_dir.display()
        );

        let mut child = Command::new(&triq_bin)
            .arg("serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("TRIQ_APP_DIR", &app_dir)
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", triq_bin.display()))?;

        let stdin = child.stdin.take().ok_or("child stdin not captured")?;
        let stdout = child.stdout.take().ok_or("child stdout not captured")?;

        // TriqiClient owns both pipes: writes are requests, reads demux
        // responses by id and push events to the receiver taken by the host.
        let client = TriqiClient::over(Box::new(std::io::BufReader::new(stdout)), Box::new(stdin));

        Ok(Self {
            client: Mutex::new(Some(client)),
            child: Mutex::new(Some(child)),
        })
    }

    /// The connected client, or an error when the child is not running.
    pub fn client(&self) -> Result<TriqiClient, String> {
        self.client
            .lock()
            .expect("client lock poisoned")
            .clone()
            .ok_or_else(|| "triq serve bridge is not running".to_string())
    }

    /// Graceful shutdown: drop the client (EOF tells serve to tear the engine
    /// down), then wait briefly for the child before killing it.
    pub fn shutdown(&self) {
        if let Some(client) = self.client.lock().expect("client lock poisoned").take() {
            client.shutdown();
        }

        let mut guard = self.child.lock().expect("child lock poisoned");
        if let Some(child) = guard.take() {
            let child = std::sync::Arc::new(Mutex::new(child));
            let (tx, rx) = std::sync::mpsc::channel();
            let waiter = std::sync::Arc::clone(&child);
            let _ = std::thread::Builder::new()
                .name("triq-wait".into())
                .spawn(move || {
                    let _ = waiter.lock().expect("child lock poisoned").wait();
                    let _ = tx.send(());
                });
            if rx.recv_timeout(Duration::from_secs(10)).is_err() {
                // Serve did not exit on EOF; do not strand it.
                let mut child = child.lock().expect("child lock poisoned");
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

/// Resolve the `triq` binary: explicit override, cargo-provided path,
/// workspace checkout (debug then release), bundle resource dir, then PATH.
fn triq_binary_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let bin_name = if cfg!(windows) { "triq.exe" } else { "triq" };

    if let Ok(path) = std::env::var("TRIQ_BIN") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(path) = std::env::var("CARGO_BIN_EXE_triq") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }

    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        // <repo>/talentrankiq_gui/src-tauri -> <repo>
        roots.push(PathBuf::from(manifest_dir).join("..").join(".."));
    }
    if let Ok(current_exe) = std::env::current_exe() {
        let mut base = current_exe.clone();
        for _ in 0..5 {
            match base.parent() {
                Some(parent) => base = parent.to_path_buf(),
                None => break,
            }
            roots.push(base.clone());
        }
        if let Some(parent) = current_exe.parent() {
            roots.push(parent.to_path_buf());
        }
    }

    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    for root in &roots {
        for profile in [profile, if profile == "debug" { "release" } else { "debug" }] {
            let candidate = root
                .join("talentrankiq_cli")
                .join("target")
                .join(profile)
                .join(bin_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        let sidecar = root.join(bin_name);
        if sidecar.exists() {
            return Ok(sidecar);
        }
    }

    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled = resource_dir.join(bin_name);
        if bundled.exists() {
            return Ok(bundled);
        }
    }

    Ok(PathBuf::from(bin_name))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::commands::{client_error_to_string, to_snake_case_params};

    fn triq_bin() -> PathBuf {
        let bin = if cfg!(windows) { "triq.exe" } else { "triq" };
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("talentrankiq_cli")
            .join("target")
            .join("debug")
            .join(bin)
    }

    /// End-to-end over the real transport: a real `triq serve` child, the real
    /// dispatcher helpers, and a real app dir. Nothing is mocked.
    #[test]
    fn dispatch_round_trips_real_serve_process() {
        let bin = triq_bin();
        assert!(
            bin.exists(),
            "triq binary missing at {} (build it: cd ../talentrankiq_cli && cargo build)",
            bin.display()
        );

        let app_dir = std::env::temp_dir().join(format!("triq-bridge-test-{}", std::process::id()));
        let bridge = SubprocessBridge::spawn_at(&bin, &app_dir).expect("spawn triq serve");
        let client = bridge.client().expect("client");

        // The serve loop answers a real command.
        let handshake = client.call("handshake", &json!({})).expect("handshake");
        assert_eq!(handshake["server_version"], "1.2.0");

        // Webview params (camelCase) become wire keys (snake_case) exactly like
        // the dev proxy, and the real server acts on them.
        let params = to_snake_case_params(json!({
            "jobId": "missing-job",
            "jobAdText": null,
            "triquiSize": 3,
        }));
        assert_eq!(
            params,
            json!({ "job_id": "missing-job", "job_ad_text": null, "triqui_size": 3 })
        );

        let error = client
            .call(
                "get_job_detail",
                &to_snake_case_params(json!({ "jobId": "missing-job" })),
            )
            .expect_err("unknown job must fail");
        let message = client_error_to_string(error);
        assert!(
            message.contains("missing-job"),
            "server error should name the job, got: {message}"
        );

        // Unknown commands surface as errors instead of hanging the webview.
        let error = client
            .call("list_candidates", &json!({ "jobId": "x" }))
            .expect_err("unknown command must fail");
        assert!(
            client_error_to_string(error).contains("unknown command"),
            "unexpected error"
        );

        bridge.shutdown();
        let _ = std::fs::remove_dir_all(&app_dir);
    }
}
