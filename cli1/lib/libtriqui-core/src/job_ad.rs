use std::path::Path;

/// Create the job-ad file at `path` (and its parent dir) if it does not exist.
pub fn ensure_file(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create ad dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path, "").map_err(|e| format!("cannot create ad file {}: {e}", path.display()))?;
    Ok(())
}

/// Read the ad text at `path`; missing file reads as empty string.
pub fn read_ad(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}
