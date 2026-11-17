use std::path::Path;

use crate::context::ProcessContext;
use crate::error::ProcessError;
use libcvparser;

pub fn list_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, ProcessError> {
    let mut out = Vec::new();
    if !dir.exists() { return Ok(out); }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

pub fn scan_watch_dir(ctx: &mut ProcessContext) -> Result<usize, ProcessError> {
    // 1. collect file list without DB lock
    let files = list_files(&ctx.watch_dir)?;
    let mut new_count = 0usize;
    for path in files {
        let path_str = path.to_string_lossy().to_string();
        // extract text
        let text = if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            std::fs::read_to_string(&path).map_err(ProcessError::Io)?
        } else {
            libcvparser::extract_text(&path).map_err(ProcessError::CvParse)?
        };
        let file_hash = hash_text(&text);
        // per file: acquire writer
        if ctx.db.file_state_has(&path_str)? {
            continue;
        }
        ctx.db.upsert_file_state(&path_str, &file_hash)?;
        if ctx.db.cv_has(&file_hash)? {
            continue;
        }
        ctx.db.insert_cv(&file_hash, &text)?;
        new_count += 1;
    }
    Ok(new_count)
}

fn hash_text(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}
