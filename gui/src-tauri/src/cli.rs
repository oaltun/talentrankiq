//! The gui's bridge to the TalentRankIQ backend (thin gui pattern,
//! `gui/design/app1/thin-gui.md`). The `triqui-cli serve` loop runs **in-process**
//! on a dedicated thread (`libtriqui_cli::InProcessServe`, in-memory duplex
//! transport) — one binary, no child process at all: the engine runs
//! modelcaller as a library, and the ORT wrapper (`onnxwrapperrs`) runs
//! in-process on its own thread. All command shells in `commands/` route
//! through [`rpc`] / [`rpc_blocking`]; serve events are forwarded as Tauri
//! events with the same names/payloads, so the frontend is unchanged.

use std::path::Path;
use std::sync::Mutex;

use serde::de::DeserializeOwned;
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};

use include_dir::include_dir;

use libtriqui_client::client::{ClientError, TriqiClient};
use libtriqui_client::types::{DomainError, RecoveryError};
use libtriqui_cli::InProcessServe;

/// Managed state: the (lazily started) in-process serve connection.
pub struct Bridge(Mutex<Option<InProcessServe>>);

impl Bridge {
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }
}

/// Start the in-process serve loop, install the event forwarder, and stash
/// the connection in managed state. `app_dir` is the Tauri app-data dir (the
/// serve loop keeps its DB/prefs/recovery there); `app_version` rides along
/// so exported job bundle metadata keeps describing the gui.
pub fn spawn_bridge(app: &AppHandle, app_dir: &Path, app_version: &str) -> Result<(), String> {
    let serve = InProcessServe::start(app_dir.to_path_buf(), app_version.to_string())
        .map_err(|e| format!("failed to start in-process serve: {e}"))?;

    // Forward serve events to the frontend under the same event names the gui
    // has always emitted (job-import-progress, runtime-status, ...).
    if let Some(rx) = serve.client().take_event_receiver() {
        let forward = app.clone();
        std::thread::Builder::new()
            .name("serve-event-forwarder".into())
            .spawn(move || {
                for event in rx {
                    let _ = forward.emit(&event.event, event.data);
                }
                log::warn!("serve event stream ended");
            })
            .map_err(|e| format!("failed to spawn event forwarder: {e}"))?;
    }

    let bridge = app.state::<Bridge>();
    *bridge.0.lock().expect("bridge lock poisoned") = Some(serve);
    log::info!(
        "in-process serve loop started (app dir {})",
        app_dir.display()
    );
    Ok(())
}

/// The live client, starting the serve loop on first use if the setup-time
/// start failed or was never attempted. One retry when closed: if the loop
/// died mid-session (a bug), the next command restarts it transparently.
fn get_or_spawn(app: &AppHandle) -> Result<TriqiClient, ClientError> {
    let live = {
        let bridge = app.state::<Bridge>();
        let guard = bridge.0.lock().expect("bridge lock poisoned");
        guard
            .as_ref()
            .filter(|s| !s.client().is_closed())
            .map(|s| s.client())
    };
    if let Some(client) = live {
        return Ok(client);
    }
    // (Re)start. Kept outside the lock: spawn installs the forwarder, which
    // touches the app handle but not the Bridge state.
    let app_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| ClientError::Protocol(format!("app data dir: {e}")))?;
    let version = app.package_info().version.to_string();
    spawn_bridge(app, &app_dir, &version).map_err(ClientError::Protocol)?;
    let client = {
        let bridge = app.state::<Bridge>();
        let guard = bridge.0.lock().expect("bridge lock poisoned");
        guard.as_ref().map(|s| s.client())
    };
    client.ok_or_else(|| ClientError::Protocol("bridge start produced no client".into()))
}

/// Send one command and wait for its response. Blocking; safe to call
/// concurrently (the reader demuxes by id). Use [`rpc_blocking`] from async
/// commands so the async runtime is never blocked.
pub fn rpc(
    app: &AppHandle,
    cmd: &str,
    params: &impl serde::Serialize,
) -> Result<Value, ClientError> {
    let client = get_or_spawn(app)?;
    client.call(cmd, params)
}

/// [`rpc`] for long-running commands invoked from `async fn` shells: runs on
/// the blocking thread pool, mirroring the pre-migration `spawn_blocking`
/// pattern.
pub async fn rpc_blocking(
    app: AppHandle,
    cmd: &'static str,
    params: impl serde::Serialize + Send + 'static,
) -> Result<Value, ClientError> {
    tauri::async_runtime::spawn_blocking(move || rpc(&app, cmd, &params))
        .await
        .map_err(|e| ClientError::Protocol(format!("rpc task failed: {e}")))?
}

/// [`rpc_blocking`] + decode the `ok` payload into a typed DTO.
pub async fn rpc_blocking_typed<T: DeserializeOwned + Send + 'static>(
    app: AppHandle,
    cmd: &'static str,
    params: impl serde::Serialize + Send + 'static,
) -> Result<T, ClientError> {
    let value = rpc_blocking(app, cmd, params).await?;
    serde_json::from_value(value).map_err(|e| ClientError::Json(e.to_string()))
}

// ---------------------------------------------------------------------------
// Error rehydration — every command keeps its pre-migration error type, so
// the generated TypeScript bindings (and frontend matching) stay unchanged.
// ---------------------------------------------------------------------------

/// `Result<_, String>` commands: the envelope's message is the string.
pub fn str_err(e: ClientError) -> String {
    e.to_string()
}

/// `Result<_, DomainError>` commands: rebuild the exact variant the serve
/// process raised from the serialized `domain` payload.
pub fn domain_err(e: ClientError) -> DomainError {
    if let ClientError::Domain(pe) = e {
        if let Some(domain) = pe.domain {
            if let Ok(de) = serde_json::from_value::<DomainError>(domain) {
                return de;
            }
        }
        return DomainError::ValidationError {
            message: pe.message,
        };
    }
    DomainError::ValidationError {
        message: e.to_string(),
    }
}

/// `Result<_, RecoveryError>` commands: same rehydration for the recovery
/// error enum.
pub fn recovery_err(e: ClientError) -> RecoveryError {
    if let ClientError::Domain(pe) = e {
        if let Some(domain) = pe.domain {
            if let Ok(re) = serde_json::from_value::<RecoveryError>(domain) {
                return re;
            }
        }
        return RecoveryError::IoError {
            message: pe.message,
        };
    }
    RecoveryError::IoError {
        message: e.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Setup helpers (app-data contract between the gui and the serve process)
// ---------------------------------------------------------------------------

/// Extract the embedded benchmark CVs into `<app_dir>/benchmark-cvs` (skip
/// existing files). `try_model` on the serve side reads them from there.
pub fn extract_benchmark_cvs(app_dir: &Path) {
    static BENCHMARK_CVS: include_dir::Dir = include_dir!("$CARGO_MANIFEST_DIR/benchmark-cvs");
    let dest = app_dir.join("benchmark-cvs");
    let mut extracted = 0usize;
    for file in BENCHMARK_CVS.files() {
        let Some(name) = file.path().file_name() else {
            continue;
        };
        let target = dest.join(name);
        if target.exists() {
            continue;
        }
        if std::fs::write(&target, file.contents()).is_ok() {
            extracted += 1;
        } else {
            log::warn!("could not write benchmark cv {}", target.display());
        }
    }
    if extracted > 0 {
        log::info!("extracted {extracted} benchmark cvs to {}", dest.display());
    }
}

/// Shut the backend down (app exit): signals EOF to the serve loop and waits
/// (bounded) for its teardown — engine shutdown ends the in-process ORT
/// wrapper thread, so this must run before the process exits.
pub fn shutdown_bridge(app: &AppHandle) {
    let bridge = app.state::<Bridge>();
    let serve = bridge.0.lock().expect("bridge lock poisoned").take();
    if let Some(serve) = serve {
        serve.shutdown();
    }
}
