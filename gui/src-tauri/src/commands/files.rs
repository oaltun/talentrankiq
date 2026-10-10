//! Local file-access commands (no `triqui-cli` involvement). Reads
//! user-picked text files (e.g. a job ad chosen through the native dialog)
//! straight from disk — the gui must not touch the fs plugin directly, so the
//! capability surface stays minimal and size/IO guards live here.

/// Upper bound for a "text file" read through this command (10 MiB) — job ads
/// and criteria are a few KiB; anything larger is almost certainly the wrong
/// file and would bloat the frontend.
const MAX_TEXT_FILE_BYTES: u64 = 10 * 1024 * 1024;

#[tauri::command]
#[specta::specta]
pub async fn read_text_file(path: String) -> Result<String, String> {
    log::info!("Reading text file: {path}");
    tauri::async_runtime::spawn_blocking(move || read_text_file_sync(&path))
        .await
        .map_err(|e| format!("File read task failed: {e}"))?
}

fn read_text_file_sync(path: &str) -> Result<String, String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("Could not stat file: {e}"))?;
    if !metadata.is_file() {
        return Err(format!("Not a regular file: {path}"));
    }
    if metadata.len() > MAX_TEXT_FILE_BYTES {
        return Err(format!(
            "File too large ({} bytes, limit {MAX_TEXT_FILE_BYTES})",
            metadata.len()
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("Could not read file: {e}"))
}
