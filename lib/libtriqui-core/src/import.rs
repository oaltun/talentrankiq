//! CV import operation (no Tauri): walk a directory's supported files, parse each
//! to text, sequentially (parallelism 1), reporting progress through callbacks.
//! After extraction it runs a stage-0 lexical (BM25) sort immediately so the
//! triqui is readable within ~1 s of a 2000-CV import. The embed→cosine→judge
//! cascade to full convergence runs later via `start_rank` (gui) or `sort_job_blocking` (cli).
//! Runs on a background thread with its own SQLite connection (WAL allows
//! concurrent access); the engine connection is opened via [`EngineRuntime::open_conn`].

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::db::ImportState;
use crate::embed::content_hash;
use crate::runtime::EngineRuntime;
use crate::types::{DomainError, FileType};

/// Import progress callback: `(done, total, failed, active, phase)`. Phase is
/// `"extracting" | "embedding" | "ranking" | "engine-unavailable" | "done"`.
pub type ImportProgress<'a> = &'a dyn Fn(i64, i64, i64, bool, &str);

static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("bad regex"));

pub fn file_type_from_ext(path: &Path) -> FileType {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());
    match ext.as_deref() {
        Some("txt") => FileType::Txt,
        Some("docx") => FileType::Docx,
        Some("pdf") => FileType::Pdf,
        Some("doc") => FileType::Doc,
        _ => FileType::Other,
    }
}

fn is_supported(path: &Path) -> bool {
    !matches!(file_type_from_ext(path), FileType::Other)
}

/// Extract plain text from a supported file. Unknown types error.
/// `.doc` (legacy Word 97-2003 binary) goes through libcvparser/anydoc.
fn extract_text(path: &Path, ft: &FileType) -> Result<String, String> {
    match ft {
        FileType::Txt => std::fs::read_to_string(path).map_err(|e| e.to_string()),
        FileType::Docx => extract_docx(path),
        FileType::Pdf => pdf_extract::extract_text(path).map_err(|e| e.to_string()),
        FileType::Doc => libcvparser::extract_text(path).map_err(|e| e.to_string()),
        FileType::Other => Err("unsupported file type".into()),
    }
}

/// A .docx is a zip; pull `word/document.xml` and strip tags.
fn extract_docx(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut xml = String::new();
    for i in 0..archive.len() {
        let mut zf = archive.by_index(i).map_err(|e| e.to_string())?;
        if zf.name() == "word/document.xml" {
            zf.read_to_string(&mut xml).map_err(|e| e.to_string())?;
            break;
        }
    }
    if xml.is_empty() {
        return Err("docx has no word/document.xml".into());
    }
    Ok(strip_xml(&xml))
}

fn strip_xml(xml: &str) -> String {
    let with_breaks = xml
        .replace("</w:p>", "\n")
        .replace("<w:br/>", "\n")
        .replace("<w:br />", "\n");
    let no_tags = TAG_RE.replace_all(&with_breaks, "");
    no_tags
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .trim()
        .to_string()
}

/// Validate `directory` and collect its supported files (recursing into
/// subdirectories), sorted for stable progress. Called before spawning the
/// background import so bad input fails synchronously.
pub fn collect_supported_files(directory: &str) -> Result<Vec<PathBuf>, DomainError> {
    let dir = PathBuf::from(directory);
    // Missing vs unreadable are different failures and the gui treats them
    // differently: a missing folder is app-managed (the watcher re-creates it),
    // so it can offer a one-click repair instead of a dead-end error. Typed,
    // not string-matched, so the branch survives a message rewording.
    if !dir.exists() {
        return Err(DomainError::NotFound {
            entity: "CV directory".into(),
            id: directory.into(),
        });
    }
    if !dir.is_dir() {
        return Err(DomainError::ValidationError {
            message: format!("Not a directory: {directory}"),
        });
    }
    let mut files = Vec::new();
    walk_supported_files(&dir, &mut files);
    files.sort();
    Ok(files)
}

fn walk_supported_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("import: cannot read {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries.filter_map(Result::ok) {
        let p = entry.path();
        if p.is_dir() {
            walk_supported_files(&p, out);
        } else if p.is_file() && is_supported(&p) {
            out.push(p);
        }
    }
}

/// Run the full import pipeline for `job_id` over `files`. Blocking — call from
/// a background thread. `import_state` (when given) has the job removed from the
/// active set on completion; `progress`/`embed_progress` report phases and
/// candidate embedding progress respectively.
pub fn run_import(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    directory: &str,
    files: Vec<PathBuf>,
    import_state: Option<ImportState>,
    progress: ImportProgress<'_>,
) {
    let conn = match rt.open_conn() {
        Ok(c) => c,
        Err(e) => {
            log::error!("import: failed to open db: {e}");
            finish_import(import_state, job_id, 0, files.len() as i64, 0, progress);
            return;
        }
    };

    // Remember the import directory on the job.
    let _ = conn.execute(
        "UPDATE jobs SET import_directory_path = ?1, updated_at = datetime('now') WHERE id = ?2",
        rusqlite::params![directory, job_id],
    );

    let total = files.len() as i64;
    let mut done = 0i64;
    let mut failed = 0i64;

    for path in &files {
        let path_str = path.to_string_lossy().to_string();

        // Dedup by (job, path): skip if already successfully extracted. Rows from
        // failed/interrupted attempts are removed so the file is retried.
        let extracted: bool = conn
            .query_row(
                "SELECT 1 FROM candidates WHERE job_id = ?1 AND source_file_path = ?2 AND extraction_status = 'Extracted'",
                rusqlite::params![job_id, &path_str],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if extracted {
            done += 1;
            progress(done, total, failed, true, "extracting");
            continue;
        }
        let _ = conn.execute(
            "DELETE FROM candidates WHERE job_id = ?1 AND source_file_path = ?2 AND extraction_status <> 'Extracted'",
            rusqlite::params![job_id, &path_str],
        );

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let ft = file_type_from_ext(path);
        let cid = format!("cand_{}", uuid::Uuid::new_v4());

        let _ = conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES (?1, ?2, ?3, ?4, ?5, 'Extracting')",
            rusqlite::params![&cid, job_id, &path_str, &file_name, ft.as_str()],
        );

        match extract_text(path, &ft) {
            Ok(text) => {
                // Stable content id (for caching + dedup). Skip if the same content is
                // already in this job under another row (e.g. a renamed CV).
                let hash = content_hash(&text);
                let dup: bool = conn
                    .query_row(
                        "SELECT 1 FROM candidates WHERE job_id = ?1 AND content_hash = ?2",
                        rusqlite::params![job_id, &hash],
                        |_| Ok(true),
                    )
                    .unwrap_or(false);
                if dup {
                    let _ = conn.execute(
                        "DELETE FROM candidates WHERE id = ?1",
                        rusqlite::params![&cid],
                    );
                    log::info!("import: skipping duplicate content {}", path.display());
                } else {
                    let _ = conn.execute(
                        "UPDATE candidates SET extraction_status = 'Extracted', extracted_text = ?1, content_hash = ?2 WHERE id = ?3",
                        rusqlite::params![&text, &hash, &cid],
                    );
                }
            }
            Err(e) => {
                log::warn!("import: failed to parse {}: {e}", path.display());
                failed += 1;
                let _ = conn.execute(
                    "UPDATE candidates SET extraction_status = 'Failed' WHERE id = ?1",
                    rusqlite::params![&cid],
                );
            }
        }

        done += 1;
        progress(done, total, failed, true, "extracting");
    }

    // Extraction phase complete. The scout cascade (first ordering + the
    // embed→cosine→judge waves) runs later via `start_rank` (gui) or
    // `sort_job_blocking` (cli); import no longer derives a lexical stage-0
    // order (v2, progressive-ranking design §21).
    finish_import(import_state, job_id, done, total, failed, progress);
}

fn finish_import(
    import_state: Option<ImportState>,
    job_id: &str,
    done: i64,
    total: i64,
    failed: i64,
    progress: ImportProgress<'_>,
) {
    progress(done, total, failed, false, "done");
    if let Some(state) = import_state {
        let mut active = state.0.lock().expect("import state lock poisoned");
        active.remove(job_id);
    }
    log::info!("import finished for job {job_id}: {done}/{total} done, {failed} failed");
}

#[cfg(test)]
mod tests {
    use super::collect_supported_files;
    use crate::types::DomainError;

    /// A missing CV folder and an unreadable one are different failures: the
    /// gui repairs the first in one click (the folder is app-managed) and only
    /// reports the second. Both must stay typed so the branch is not a string
    /// match on the message.
    #[test]
    fn missing_folder_is_typed_not_found_and_a_file_is_not_a_directory() {
        let dir = std::env::temp_dir().join(format!("triqui-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let missing = dir.join("gone");
        match collect_supported_files(missing.to_str().unwrap()) {
            Err(DomainError::NotFound { entity, id }) => {
                assert_eq!(entity, "CV directory");
                assert_eq!(id, missing.to_string_lossy());
            }
            other => panic!("missing dir must be NotFound, got {other:?}"),
        }

        // Exists but is a file: a different error, so the gui does not offer to
        // "create" something that is already there.
        let as_file = dir.join("not-a-dir");
        std::fs::write(&as_file, b"x").unwrap();
        match collect_supported_files(as_file.to_str().unwrap()) {
            Err(DomainError::ValidationError { message }) => {
                assert!(message.starts_with("Not a directory:"), "{message}");
            }
            other => panic!("a file in the dir's place must be a validation error, got {other:?}"),
        }

        // The happy path: an existing empty folder collects cleanly.
        assert!(collect_supported_files(dir.to_str().unwrap()).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
