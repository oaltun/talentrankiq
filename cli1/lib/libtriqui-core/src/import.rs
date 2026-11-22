//! CV import helpers: filesystem enumeration shared by the scan worker.

use std::path::{Path, PathBuf};

/// File extensions the importer recognizes.
pub fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
        Some(ext) if matches!(ext.as_str(), "pdf" | "docx" | "doc" | "txt")
    )
}

/// Recursively collect all supported CV files under `dir`, sorted for
/// deterministic ordering. Pure filesystem I/O — holds no DB lock.
pub fn collect_supported_files(dir: &str) -> Result<Vec<PathBuf>, String> {
    let root = Path::new(dir);
    let mut out = Vec::new();
    collect_into(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_into(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip hidden / system files (e.g. .DS_Store).
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_into(&path, out)?;
        } else if path.is_file() && is_supported(&path) {
            out.push(path);
        }
    }
    Ok(())
}