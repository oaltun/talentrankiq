//! Preferences: `greet` stays local demo plumbing; load/save are thin shells
//! over `triqui-cli serve` rpcs (the file lives in the serve app dir). The
//! startup shortcut lookup still reads `preferences.json` directly — the
//! path+layout are part of the gui↔serve app-dir contract, and the bridge is
//! not up yet during setup.

use libtriqui_client::params::*;
use libtriqui_client::types::{
    validate_color_theme, validate_string_input, validate_theme, AppPreferences,
};

use crate::cli::{rpc_blocking, rpc_blocking_typed, str_err};

use tauri::AppHandle;

#[cfg(desktop)]
use std::path::PathBuf;
#[cfg(desktop)]
use tauri::Manager;

#[cfg(desktop)]
/// Gets the path to the preferences file (app-data contract with the serve
/// process: `<app_data_dir>/preferences.json`).
fn get_preferences_path(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to get app data directory: {e}"))?;

    // Ensure the directory exists
    std::fs::create_dir_all(&app_data_dir)
        .map_err(|e| format!("Failed to create app data directory: {e}"))?;

    Ok(app_data_dir.join("preferences.json"))
}

#[cfg(desktop)]
/// Load the saved quick pane shortcut from preferences, returning None on any failure.
/// Used at startup before the full preferences system is available.
pub fn load_quick_pane_shortcut(app: &AppHandle) -> Option<String> {
    let path = get_preferences_path(app).ok()?;
    if !path.exists() {
        return None;
    }
    let contents = std::fs::read_to_string(&path)
        .inspect_err(|e| log::warn!("Failed to read preferences: {e}"))
        .ok()?;
    let prefs: AppPreferences = serde_json::from_str(&contents)
        .inspect_err(|e| log::warn!("Failed to parse preferences: {e}"))
        .ok()?;
    prefs.quick_pane_shortcut
}

/// Simple greeting command for demonstration purposes.
#[tauri::command]
#[specta::specta]
pub fn greet(name: &str) -> Result<String, String> {
    // Input validation
    validate_string_input(name, 100, "Name").map_err(|e| {
        log::warn!("Invalid greet input: {e}");
        e
    })?;

    log::info!("Greeting user: {name}");
    Ok(format!("Hello, {name}! You've been greeted from Rust!"))
}

/// Loads user preferences from disk (inside the serve process).
/// Returns default preferences if the file doesn't exist.
#[tauri::command]
#[specta::specta]
pub async fn load_preferences(app: AppHandle) -> Result<AppPreferences, String> {
    log::debug!("Loading preferences from disk");
    let preferences = rpc_blocking_typed::<AppPreferences>(app, "load_preferences", EmptyParams {})
        .await
        .map_err(str_err)?;
    log::info!("Successfully loaded preferences");
    Ok(preferences)
}

/// Saves user preferences to disk (inside the serve process; atomic write).
#[tauri::command]
#[specta::specta]
pub async fn save_preferences(app: AppHandle, preferences: AppPreferences) -> Result<(), String> {
    // Validate theme value
    validate_theme(&preferences.theme)?;
    // Validate color-scheme (palette) value
    validate_color_theme(&preferences.color_theme)?;

    log::debug!("Saving preferences to disk: {preferences:?}");
    rpc_blocking(
        app,
        "save_preferences",
        SavePreferencesParams { preferences },
    )
    .await
    .map(|_| ())
    .map_err(str_err)?;
    log::info!("Successfully saved preferences");
    Ok(())
}

/// Effective root folder for new jobs' default CV directories, resolved
/// serve-side: the saved gui preference (Settings → General → CV directory)
/// when set, then `gui_cv_dir_root` from `triq.yaml`, then the platform
/// Documents dir + `jobs`. The Settings dialog shows it as the fallback while
/// no custom root is set.
#[tauri::command]
#[specta::specta]
pub async fn get_cv_dir_root(app: AppHandle) -> Result<String, String> {
    rpc_blocking_typed::<String>(app, "get_cv_dir_root", EmptyParams {})
        .await
        .map_err(str_err)
}
