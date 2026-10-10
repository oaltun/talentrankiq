//! Debug-only commands used to surface frontend events in the terminal stdout.

use tauri::AppHandle;

/// Print a message from the frontend to the terminal stdout so frontend markers
/// (e.g. splash-screen lifecycle) are visible alongside Rust logs during dev.
#[tauri::command]
#[specta::specta]
pub fn log_stdout(_app: AppHandle, msg: String) {
    println!("[frontend] {msg}");
}
