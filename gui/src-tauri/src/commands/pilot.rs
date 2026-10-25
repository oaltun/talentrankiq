// Pilot bridge commands - replaces broken tauri-pilot JS bridge injection
// The frontend loads a JS shim that listens for Tauri events and executes scripts.
// This module handles the Rust side: sends script to frontend, waits for response.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Listener, Manager};
use tokio::sync::oneshot;
use tokio::time::timeout;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::OnceLock;
use std::time::Duration;
use crate::types::DomainError;

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum PilotRequest {
    Query { selector: String },
    Click { selector: String },
    Eval { js: String },
    Snapshot,
    Ready,
}

#[derive(Debug, Deserialize, Serialize)]
struct PilotMessage {
    id: u64,
    request: PilotRequest,
}

#[derive(Debug, Serialize)]
struct PilotResponse {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// Global registry for pending pilot requests
static PILOT_REGISTRY: OnceLock<Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>>> = 
    OnceLock::new();

fn get_registry() -> &'static Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>> {
    PILOT_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

fn register_pending(id: u64) -> oneshot::Receiver<Result<Value, String>> {
    let (tx, rx) = oneshot::channel();
    get_registry().lock().unwrap().insert(id, tx);
    rx
}

fn complete_pending(id: u64, result: Result<Value, String>) {
    if let Some(tx) = get_registry().lock().unwrap().remove(&id) {
        let _ = tx.send(result);
    }
}

/// Generate script for query selector
fn make_query_script(selector: &str) -> String {
    format!(
        r#"
        (function() {{
            const elements = document.querySelectorAll({});
            const results = [];
            for (const el of elements) {{
                const tag = el.tagName.toLowerCase();
                const id = el.id || null;
                const classes = Array.from(el.classList);
                const text = el.textContent?.slice(0, 500) || '';
                const rect = el.getBoundingClientRect();
                const rectData = rect.width > 0 && rect.height > 0 ? {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height }} : null;
                const attrs = {{}};
                for (const attr of el.attributes) {{
                    attrs[attr.name] = attr.value;
                }}
                results.push({{ tag, id, classes, text, rect: rectData, attributes: attrs, children: [] }});
            }}
            return results;
        }})()
        "#,
        serde_json::to_string(selector).unwrap()
    )
}

/// Generate script for click
fn make_click_script(selector: &str) -> String {
    format!(
        r#"
        (function() {{
            const el = document.querySelector({});
            if (!el) return {{ success: false, error: "Element not found" }};
            el.click();
            return {{ success: true }};
        }})()
        "#,
        serde_json::to_string(selector).unwrap()
    )
}

/// Generate script for snapshot
const SNAPSHOT_SCRIPT: &str = r#"
    (function() {
        const elements = [];
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
        while (walker.nextNode()) {
            const el = walker.currentNode;
            const tag = el.tagName.toLowerCase();
            const id = el.id || null;
            const classes = Array.from(el.classList);
            const text = el.textContent?.slice(0, 500) || '';
            const rect = el.getBoundingClientRect();
            const rectData = rect.width > 0 && rect.height > 0 ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null;
            const attrs = {};
            for (const attr of el.attributes) {
                attrs[attr.name] = attr.value;
            }
            elements.push({ tag, id, classes, text, rect: rectData, attributes: attrs, children: [] });
        }
        return {
            url: window.location.href,
            title: document.title,
            elements: elements
        };
    })()
"#;

/// Main pilot bridge command - receives messages from external tools/tests
#[tauri::command]
#[specta::specta]
pub async fn pilot_bridge(
    app: AppHandle,
    message: String,
) -> Result<String, DomainError> {
    let msg: PilotMessage = serde_json::from_str(&message)
        .map_err(|e| DomainError::ValidationError { message: format!("Invalid pilot message: {}", e) })?;
    
    let window = app.get_webview_window("main")
        .ok_or_else(|| DomainError::ValidationError { message: "Main window not found".into() })?;
    
    // For Ready, just acknowledge
    if matches!(msg.request, PilotRequest::Ready) {
        let response = PilotResponse {
            id: msg.id,
            result: Some(serde_json::json!({ "status": "ready" })),
            error: None,
        };
        return serde_json::to_string(&response)
            .map_err(|e| DomainError::ValidationError { message: format!("Response serialization failed: {}", e) });
    }
    
    // Generate script based on request type
    let script = match &msg.request {
        PilotRequest::Query { selector } => make_query_script(selector),
        PilotRequest::Click { selector } => make_click_script(selector),
        PilotRequest::Eval { js } => js.clone(),
        PilotRequest::Snapshot => SNAPSHOT_SCRIPT.to_string(),
        PilotRequest::Ready => unreachable!(),
    };
    
    // Register pending request
    let rx = register_pending(msg.id);
    
    // Emit event to frontend to execute script
    let event_name = "pilot:execute";
    let payload = serde_json::json!({
        "id": msg.id,
        "script": script,
    });
    
    window.emit(event_name, &payload)
        .map_err(|e| DomainError::ValidationError { message: format!("Failed to emit pilot event: {}", e) })?;
    
    // Wait for response with timeout
    let result = match timeout(Duration::from_secs(10), rx).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err("Pilot response channel closed".to_string()),
        Err(_) => {
            // Clean up pending request on timeout
            get_registry().lock().unwrap().remove(&msg.id);
            Err("Pilot request timed out".to_string())
        }
    };
    
    let response = PilotResponse {
        id: msg.id,
        result: result.as_ref().ok().cloned(),
        error: result.as_ref().err().cloned(),
    };
    
    serde_json::to_string(&response)
        .map_err(|e| DomainError::ValidationError { message: format!("Response serialization failed: {}", e) })
}

/// Callback command - called by frontend JS shim to return script result
#[tauri::command]
#[specta::specta]
pub fn pilot_callback(
    _app: AppHandle,
    id: u64,
    result: Value,
    error: Option<String>,
) -> Result<(), DomainError> {
    let result = if let Some(err) = error {
        Err(err)
    } else {
        Ok(result)
    };
    complete_pending(id, result);
    Ok(())
}

static EVENT_LISTENER_REGISTRY: OnceLock<Arc<Mutex<HashMap<u64, tauri::EventId>>>> = OnceLock::new();

fn get_event_registry() -> &'static Arc<Mutex<HashMap<u64, tauri::EventId>>> {
    EVENT_LISTENER_REGISTRY.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

/// Tauri command to listen for events from the serve process
#[tauri::command]
#[specta::specta]
pub fn event_listen(
    app: AppHandle,
    event: String,
    id: u64,
) -> Result<(), DomainError> {
    let app_handle = app.clone();
    let event_name = event.clone();
    let listener_id = app.listen(event_name, move |e: tauri::Event| {
        let _ = app_handle.emit(&format!("pilot:event:{}", event), e.payload());
    });
    
    get_event_registry().lock().unwrap().insert(id, listener_id);
    Ok(())
}

/// Tauri command to unlisten for events
#[tauri::command]
#[specta::specta]
pub fn event_unlisten(
    app: AppHandle,
    id: u64,
) -> Result<(), DomainError> {
    if let Some(listener_id) = get_event_registry().lock().unwrap().remove(&id) {
        app.unlisten(listener_id);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pilot_message_serialization() {
        let msg = PilotMessage {
            id: 1,
            request: PilotRequest::Query { selector: "#test".into() },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("query"));
        assert!(json.contains("#test"));
    }
}