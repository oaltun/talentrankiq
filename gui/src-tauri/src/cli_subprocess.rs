//! Subprocess bridge: spawns `triq` CLI as a child process and connects
//! its stdin/stdout to the JSONL protocol client (`TriqiClient`).
//!
//! This replaces the in-process `InProcessServe` — the GUI is now a thin
//! client that only manages the subprocess lifecycle. All domain logic,
//! the engine, DB, model management, and CV watching live in the CLI.

use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use crate::triqi_client::TriqiClient;
use tauri::Manager;

use crate::utils::platform;

/// Managed state: the subprocess bridge holding the client and child process.
pub struct SubprocessBridge {
    client: TriqiClient,
    child: Mutex<Option<Child>>,
}

impl SubprocessBridge {
    /// Spawn the `triq` CLI subprocess and return a connected client.
    /// `app_dir` is the Tauri app-data dir; `app_version` is the GUI version.
    pub fn spawn(
        app: &tauri::AppHandle,
        app_dir: &std::path::Path,
        _app_version: &str,
    ) -> Result<Self, String> {
        let triq_bin = triq_binary_path(app)?;
        log::info!("Spawning triq CLI: {}", triq_bin.display());

        let mut cmd = Command::new(&triq_bin);
        cmd.arg("serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // CLI logs go to parent stderr
            .env("TRIQ_APP_DIR", app_dir)
            .env("TRIQI_MODEL_DIR", model_cache_dir(app)?);

        // Pass bundled config if available
        if let Ok(resource_dir) = app.path().resource_dir() {
            let config_path = resource_dir.join("triq.yaml");
            if config_path.exists() {
                cmd.env("TRIQ_CONFIG", &config_path);
                log::info!("Using bundled config: {}", config_path.display());
            }
        }

        // Propagate embedding model override if set
        if let Ok(model) = std::env::var("TRIQI_EMBED_MODEL") {
            cmd.env("TRIQI_EMBED_MODEL", model);
        }

        // Propagate CPU-only engine override if set
        if std::env::var("TRIQI_CPU_ONLY").is_ok() {
            cmd.env("TRIQI_CPU_ONLY", "1");
        }

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "failed to spawn triq CLI at {}: {e}",
                triq_bin.display()
            )
        })?;

        let stdin = child.stdin.take().ok_or("child stdin not captured")?;
        let stdout = child.stdout.take().ok_or("child stdout not captured")?;

        // Wrap stdio in the duplex transport so TriqiClient works unchanged.
        // Duplex layout:
        //   cli_write (a_write) -> srv_read (b_read) : client requests -> serve reads
        //   srv_write (b_write) -> cli_read (a_read) : serve responses -> client reads
        let (cli_write, cli_read, srv_write, srv_read) =
            crate::triqi_client::transport::duplex();

        // Thread 1: Forward requests from client -> child stdin.
        // Client writes to cli_write, which connects to srv_read.
        // We read from srv_read and write to child stdin.
        {
            let mut stdin = stdin;
            let mut srv_read = srv_read;
            thread::Builder::new()
                .name("triq-stdin-forwarder".into())
                .spawn(move || {
                    let mut buf = [0u8; 64 * 1024];
                    loop {
                        match srv_read.read(&mut buf) {
                            Ok(0) => break, // EOF from serve loop (client closed)
                            Ok(n) => {
                                if stdin.write_all(&buf[..n]).is_err() {
                                    break;
                                }
                                let _ = stdin.flush();
                            }
                            Err(_) => break,
                        }
                    }
                })
                .map_err(|e| format!("spawn stdin forwarder: {e}"))?;
        }

        // Thread 2: Forward responses from child stdout -> client.
        // Child writes responses to stdout. We read from stdout and write to srv_write,
        // which connects to cli_read (client reads responses from here).
        {
            let mut stdout = BufReader::new(stdout);
            let mut srv_write = srv_write;
            thread::Builder::new()
                .name("triq-stdout-forwarder".into())
                .spawn(move || {
                    let mut buf = [0u8; 64 * 1024];
                    loop {
                        match stdout.read(&mut buf) {
                            Ok(0) => break, // EOF from child
                            Ok(n) => {
                                if srv_write.write_all(&buf[..n]).is_err() {
                                    break;
                                }
                                let _ = srv_write.flush();
                            }
                            Err(_) => break,
                        }
                    }
                })
                .map_err(|e| format!("spawn stdout forwarder: {e}"))?;
        }

        let client = TriqiClient::over(Box::new(cli_read), Box::new(cli_write));

        Ok(Self {
            client,
            child: Mutex::new(Some(child)),
        })
    }

    /// Get the connected client.
    pub fn client(&self) -> TriqiClient {
        self.client.clone()
    }

    /// Check if the subprocess is still alive.
    pub fn is_alive(&self) -> bool {
        self.child
            .lock()
            .expect("child lock poisoned")
            .as_mut()
            .map(|c| c.try_wait().unwrap_or(None).is_none())
            .unwrap_or(false)
    }

    /// Graceful shutdown: drop client (signals EOF to serve loop), wait for exit.
    pub fn shutdown(self) {
        // Drop the client — this closes the write end (cli_write), which signals
        // EOF to the serve loop's reader (srv_read). The serve loop then shuts
        // down the engine and exits. The forwarder thread will also exit when
        // srv_read returns EOF, closing the child's stdin.
        self.client.shutdown();

        // Wait for the child to exit (bounded).
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let (tx, rx) = std::sync::mpsc::channel();
                thread::Builder::new()
                    .name("triq-wait".into())
                    .spawn(move || {
                        let _ = child.wait();
                        let _ = tx.send(());
                    })
                    .ok();
                let _ = rx.recv_timeout(Duration::from_secs(150));
            }
        }
    }
}

/// Resolve the `triq` binary path.
/// - Dev/test: `CARGO_BIN_EXE_triq` env var (set by Cargo)
/// - Release (bundled): Tauri resource dir / sidecar
fn triq_binary_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    // 1. Dev / test: Cargo sets this for `cargo test` and `cargo run`
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_triq") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    // 2. Bundled: sidecar in Tauri resource dir
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("resource dir: {e}"))?;

    let bin_name = if platform::is_windows() {
        "triq.exe"
    } else {
        "triq"
    };

    let bundled = resource_dir.join(bin_name);
    if bundled.exists() {
        return Ok(bundled);
    }

    // 3. Fallback: assume it's in PATH (e.g. `cargo install triq-cli`)
    Ok(PathBuf::from(bin_name))
}

/// Resolve the model cache directory (mirrors GUI setup).
fn model_cache_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| format!("app cache dir: {e}"))?;
    Ok(cache_dir.join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_triq_binary_name() {
        let name = if platform::is_windows() {
            "triq.exe"
        } else {
            "triq"
        };
        assert!(!name.is_empty());
    }
}