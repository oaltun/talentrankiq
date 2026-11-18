use std::path::Path;

use crate::context::ProcessContext;
use crate::error::ProcessError;

pub fn scan_watch_dir(ctx: &mut ProcessContext) -> Result<usize, ProcessError> {
    // Collect the same supported CV files the import watcher uses, recursively.
    // This keeps stray files (e.g. a macOS `.DS_Store`) out of the scan instead
    // of letting one unsupported file abort the whole run.
    let dir = ctx.watch_dir.to_string_lossy().to_string();
    let files = libtriqui_core::import::collect_supported_files(&dir)?;
    // Reconcile this job's membership to exactly the files in its watch dir: drop
    // rows for paths no longer present (e.g. after a directory change) and re-add
    // the current ones. Content-hash dedupe stays global across jobs.
    ctx.db.clear_job_cv(&ctx.job.id)?;
    let mut new_count = 0usize;
    for path in files {
        let path_str = path.to_string_lossy().to_string();
        // A single unreadable/odd file must not abort the wave; log and move on.
        let text = match read_text(&path) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("process_in_waves: skipping {path_str}: {e}");
                continue;
            }
        };
        let file_hash = hash_text(&text);
        ctx.db.upsert_file_state(&path_str, &file_hash)?;
        ctx.db.insert_job_cv(&ctx.job.id, &file_hash)?;
        if !ctx.db.cv_has(&file_hash)? {
            ctx.db.insert_cv(&file_hash, &text)?;
            new_count += 1;
        }
    }
    Ok(new_count)
}

fn read_text(path: &Path) -> Result<String, ProcessError> {
    if path.extension().and_then(|e| e.to_str()) == Some("txt") {
        Ok(std::fs::read_to_string(path)?)
    } else {
        Ok(libcvparser::extract_text(path)?)
    }
}

fn hash_text(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}
