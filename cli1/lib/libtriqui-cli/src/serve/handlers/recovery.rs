//! Emergency recovery data handlers.

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

fn recovery_dir(state: &Arc<ServeState>) -> std::path::PathBuf {
    state.app_dir.join("recovery")
}

pub fn save_emergency_data(
    state: &Arc<ServeState>,
    filename: &str,
    data: &serde_json::Value,
) -> Result<Value, ProtocolError> {
    let dir = recovery_dir(state);
    std::fs::create_dir_all(&dir).map_err(|e| string_err(e.to_string()))?;
    let path = dir.join(filename);
    let bytes = serde_json::to_vec(data).map_err(|e| string_err(e.to_string()))?;
    std::fs::write(&path, bytes).map_err(|e| string_err(e.to_string()))?;
    ok_json(true)
}

pub fn load_emergency_data(state: &Arc<ServeState>, filename: &str) -> Result<Value, ProtocolError> {
    let path = recovery_dir(state).join(filename);
    let bytes = std::fs::read(&path).map_err(|e| string_err(e.to_string()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| string_err(e.to_string()))?;
    ok_json(value)
}

pub fn cleanup_old_recovery_files(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    let dir = recovery_dir(state);
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    ok_json(true)
}