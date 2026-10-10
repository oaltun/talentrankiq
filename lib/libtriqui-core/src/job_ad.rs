//! The job ad as a **file on disk**, not app state.
//!
//! The ad is a long block of prose the user pastes or writes, so it belongs in
//! a plain text file they can open in their own editor, diff, back up and keep
//! next to the job — the same "a path you own, an app that watches it" shape as
//! the CV directory. The app's job is to seed the file, read it, notice when it
//! changes, and re-rank.
//!
//! Default location is a sibling of the CV directory:
//!
//! ```text
//! <cv-root>/<job-slug>/cv/          # watched CV folder
//! <cv-root>/<job-slug>/job-ad.txt   # this
//! ```
//!
//! The file is created **empty** when missing — an ad the user has not written
//! yet is a normal state, not an error, and the app must not invent content.

use std::path::{Path, PathBuf};

/// File name of the job ad inside a job's folder.
pub const JOB_AD_FILE: &str = "job-ad.txt";

/// Largest ad we will read, mirroring the gui command's own limit. A job ad is
/// prose; anything past this is a wrong-file-selected mistake, not an ad.
pub const MAX_JOB_AD_BYTES: u64 = 5 * 1024 * 1024;

/// Default ad path for a job: `<root>/<job-slug>/job-ad.txt`.
///
/// An empty/unresolvable `root` yields an empty path, mirroring
/// `crate::app_config::cv_dir_root` being optional — the caller then leaves the
/// job's ad path unset rather than inventing one.
pub fn default_path(cv_root: &Path, job_slug: &str) -> PathBuf {
    if cv_root.as_os_str().is_empty() || job_slug.is_empty() {
        return PathBuf::new();
    }
    cv_root.join(job_slug).join(JOB_AD_FILE)
}

/// Create the ad file (and its parent folder) when missing, leaving existing
/// content untouched. Returns the path so callers can report it.
///
/// Parent folders are created too: the job folder normally already exists
/// because the CV directory lives beside the ad, but a job whose CV folder was
/// pointed elsewhere has no folder of its own yet.
pub fn ensure_file(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err("No job-ad path resolved".into());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {e}", parent.display()))?;
    }
    if path.exists() {
        return Ok(());
    }
    std::fs::write(path, "")
        .map_err(|e| format!("Cannot create {}: {e}", path.display()))
}

/// Read the ad text. A missing file reads as empty (the user deleted it, or the
/// job has no ad yet) rather than erroring — the ad is optional.
pub fn read_text(path: &Path) -> Result<String, String> {
    if path.as_os_str().is_empty() {
        return Ok(String::new());
    }
    match std::fs::metadata(path) {
        Err(_) => return Ok(String::new()),
        Ok(m) if !m.is_file() => {
            return Err(format!("Not a regular file: {}", path.display()))
        }
        Ok(m) if m.len() > MAX_JOB_AD_BYTES => {
            return Err(format!(
                "Job ad file is too large ({} bytes, limit {MAX_JOB_AD_BYTES})",
                m.len()
            ))
        }
        Ok(_) => {}
    }
    let raw = std::fs::read(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// Normalise ad text for storage: a blank ad is stored as NULL so the rest of
/// the app can keep using "no ad" as a single state.
pub fn text_to_db(text: &str) -> Option<String> {
    let t = text.trim();
    if t.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_path_sits_beside_the_cv_folder() {
        assert_eq!(
            default_path(Path::new("/root/jobs"), "job-7"),
            PathBuf::from("/root/jobs/job-7/job-ad.txt")
        );
        // An unresolvable root yields no path — the caller leaves it unset.
        assert_eq!(default_path(Path::new(""), "job-7"), PathBuf::new());
        assert_eq!(default_path(Path::new("/root"), ""), PathBuf::new());
    }

    #[test]
    fn ensure_creates_an_empty_file_and_never_clobbers_content() {
        let dir = std::env::temp_dir().join(format!("jobad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = default_path(&dir, "job-1");

        ensure_file(&path).expect("first ensure creates the file");
        assert_eq!(read_text(&path).unwrap(), "", "seeds empty, not invented");
        // The parent folder came with it (no CV dir here).
        assert!(path.parent().unwrap().is_dir());

        std::fs::write(&path, "Hiring a diesel mechanic.").unwrap();
        ensure_file(&path).expect("second ensure is a no-op");
        assert_eq!(read_text(&path).unwrap(), "Hiring a diesel mechanic.");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_reads_as_empty_not_an_error() {
        let dir = std::env::temp_dir().join(format!("jobad-gone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(read_text(&dir.join("nope.txt")).unwrap(), "");
        assert_eq!(read_text(Path::new("")).unwrap(), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_blank_ad_is_stored_as_absent() {
        assert_eq!(text_to_db(""), None);
        assert_eq!(text_to_db("   \n\t "), None);
        assert_eq!(text_to_db(" real ad "), Some(" real ad ".to_string()));
    }
}