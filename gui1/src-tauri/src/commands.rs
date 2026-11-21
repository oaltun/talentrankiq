//! Tauri commands: a single dispatcher that forwards to the `triq serve`
//! child, plus the bridge lifecycle (spawn, event forwarding, shutdown).
//!
//! Keeping one command instead of one per protocol command means the GUI
//! surface is the protocol itself: a new serve command needs no Rust change,
//! and the Tauri path cannot drift from the browser (Vite proxy) path.

use serde_json::{Map, Value};
use tauri::{AppHandle, Emitter, Manager};

use libtriqui_client::ClientError;

use crate::cli_subprocess::SubprocessBridge;

/// Start the `triq serve` child and forward its events to the webview.
///
/// Best-effort: a spawn failure is logged and commands report the bridge as
/// unavailable instead of the app failing to start.
pub fn init(app: &AppHandle) {
    match SubprocessBridge::spawn(app) {
        Ok(bridge) => {
            let client = bridge.client().ok();
            app.manage(bridge);
            if let Some(client) = client {
                start_event_listener(AppHandle::clone(app), client);
            }
        }
        Err(error) => log::error!("failed to start triq serve bridge: {error}"),
    }
}

/// Drain the client's event channel and re-emit every protocol event on the
/// webview event bus, so `listenEvent` works identically in Tauri and browser.
fn start_event_listener(app: AppHandle, client: libtriqui_client::TriqiClient) {
    let Some(events) = client.take_event_receiver() else {
        log::warn!("triq serve event receiver already taken; events will not be forwarded");
        return;
    };

    let _ = std::thread::Builder::new()
        .name("triq-event-forwarder".into())
        .spawn(move || {
            for event in events {
                if let Err(error) = app.emit(&event.event, &event.data) {
                    log::warn!("failed to emit triq event {}: {error}", event.event);
                }
            }
            log::info!("triq serve event stream closed");
        });
}

/// Forward one protocol command to `triq serve` and return its `ok` payload.
///
/// `params` are the command params as the frontend sent them (camelCase or
/// snake_case); keys are normalized to the wire form (snake_case) exactly like
/// the dev proxy does, so both transports reject the same requests.
#[tauri::command]
pub async fn triq_dispatch(
    app: AppHandle,
    cmd: String,
    params: Option<Value>,
) -> Result<Value, String> {
    let bridge = app
        .try_state::<SubprocessBridge>()
        .ok_or_else(|| "triq serve bridge is not running".to_string())?;
    let client = bridge.client()?;
    let params = to_snake_case_params(params.unwrap_or_else(|| Value::Object(Map::new())));

    tauri::async_runtime::spawn_blocking(move || client.call(&cmd, &params))
        .await
        .map_err(|error| format!("triq dispatch join failed: {error}"))?
        .map_err(client_error_to_string)
}

/// Normalize top-level param names to the protocol's snake_case keys.
///
/// Mirrors `toSnakeCaseParams` in `src/lib/invoke.ts` (browser transport).
/// Nested values are left untouched, so their keys must already be wire keys.
pub(crate) fn to_snake_case_params(params: Value) -> Value {
    match params {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (to_snake_case(&key), value))
                .collect::<Map<String, Value>>(),
        ),
        other => other,
    }
}

pub(crate) fn to_snake_case(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for ch in key.chars() {
        if ch.is_ascii_uppercase() {
            out.push('_');
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Protocol errors carry the server's display message; the frontend matches on
/// that text, so unwrap the domain error instead of prefixing it.
pub(crate) fn client_error_to_string(error: ClientError) -> String {
    match error {
        ClientError::Write(message) => message,
        ClientError::Json(message) => message,
        ClientError::Closed => "triq serve has exited".to_string(),
        ClientError::Protocol(message) => message,
        ClientError::Domain(error) => error.message,
    }
}