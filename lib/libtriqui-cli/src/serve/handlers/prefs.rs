//! Preferences load/save (was gui `commands/preferences.rs`), resolved
//! against the serve app dir instead of the Tauri app-data handle.

use serde_json::Value;
use std::path::PathBuf;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{validate_color_theme, validate_theme, AppPreferences};

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::watch::cv_root;
use crate::serve::ServeState;

/// The preferences file path under the serve app dir.
pub fn preferences_path(state: &ServeState) -> PathBuf {
    state.app_dir.join("preferences.json")
}

/// Load the full preferences struct from disk.
/// Returns `AppPreferences::default()` if the file is missing or unparseable.
pub fn load_app_preferences(state: &ServeState) -> AppPreferences {
    let path = preferences_path(state);
    if !path.exists() {
        return AppPreferences::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => AppPreferences::default(),
    }
}

pub fn load_preferences(state: &ServeState) -> Result<Value, ProtocolError> {
    log::debug!("loading preferences from disk");
    let path = preferences_path(state);
    if !path.exists() {
        log::info!("preferences file not found, using defaults");
        return ok_json(AppPreferences::default());
    }
    let contents = std::fs::read_to_string(&path).map_err(|e| {
        log::error!("failed to read preferences file: {e}");
        string_err(format!("Failed to read preferences file: {e}"))
    })?;
    let preferences: AppPreferences = serde_json::from_str(&contents).map_err(|e| {
        log::error!("failed to parse preferences JSON: {e}");
        ProtocolError::string(format!("Failed to parse preferences: {e}"))
    })?;
    ok_json(preferences)
}

/// Save preferences with theme validation and an atomic temp+rename write.
pub fn save_preferences(
    state: &ServeState,
    preferences: AppPreferences,
) -> Result<(), ProtocolError> {
    validate_theme(&preferences.theme).map_err(string_err)?;
    validate_color_theme(&preferences.color_theme).map_err(string_err)?;

    log::debug!("saving preferences to disk: {preferences:?}");
    let prefs_path = preferences_path(state);

    let json_content = serde_json::to_string_pretty(&preferences)
        .map_err(|e| string_err(format!("Failed to serialize preferences: {e}")))?;

    // Write to a temporary file first, then rename (atomic operation)
    let temp_path = prefs_path.with_extension("tmp");

    std::fs::write(&temp_path, json_content)
        .map_err(|e| string_err(format!("Failed to write preferences file: {e}")))?;

    if let Err(rename_err) = std::fs::rename(&temp_path, &prefs_path) {
        log::error!("failed to finalize preferences file: {rename_err}");
        // Clean up the temp file to avoid leaving orphaned files on disk
        if let Err(remove_err) = std::fs::remove_file(&temp_path) {
            log::warn!("failed to remove temp file after rename failure: {remove_err}");
        }
        return Err(string_err(format!(
            "Failed to finalize preferences file: {rename_err}"
        )));
    }

    log::info!("saved preferences to {prefs_path:?}");
    Ok(())
}

/// Internal convenience for handlers that modify + persist prefs (models).
pub fn save_app_preferences(state: &ServeState, prefs: AppPreferences) -> Result<(), String> {
    save_preferences(state, prefs).map_err(|e| e.message)
}

/// The effective root folder new jobs' default CV directories live under
/// (saved preference when set, else `triq.yaml`, else the platform default).
/// The Settings dialog shows it as the fallback while no custom root is set.
pub fn get_cv_dir_root(state: &ServeState) -> Result<Value, ProtocolError> {
    ok_json(cv_root(state).to_string_lossy().to_string())
}
