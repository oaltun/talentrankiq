use std::path::{Path, PathBuf};

pub const DEFAULT_REVISION: &str = "main";

pub const DEFAULT_FILES: &[&str] = &[
    "model.onnx",
    "model.onnx.gz",
    "tokenizer.json",
    "config.json",
];

pub const REPO_DIR: &str = "repo";
pub const META_DIR: &str = "meta";
pub const META_FILE: &str = "source.json";

pub fn default_model_dir() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".cache").join("modelcaller"))
        .unwrap_or_else(|_| PathBuf::from(".cache").join("modelcaller"))
}

pub fn resolve_model_dir(cli_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_dir {
        if !p.as_os_str().is_empty() {
            return p.to_path_buf();
        }
    }
    if let Some(p) = std::env::var_os("MODELCALLER_MODEL_DIR") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Some(p) = std::env::var_os("MODELDOWNLOADER_MODEL_DIR") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    default_model_dir()
}

pub fn slug_dir(model_dir: &Path, slug: &str) -> PathBuf {
    model_dir.join(slug)
}

pub fn repo_dir(model_dir: &Path, slug: &str) -> PathBuf {
    slug_dir(model_dir, slug).join(REPO_DIR)
}

pub fn legacy_dir(model_dir: &Path, slug: &str, revision: &str) -> PathBuf {
    slug_dir(model_dir, slug).join(revision)
}

pub fn meta_dir(model_dir: &Path, slug: &str) -> PathBuf {
    slug_dir(model_dir, slug).join(META_DIR)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    Repo(PathBuf),
    Legacy(PathBuf),
}

impl Resolution {
    pub fn path(&self) -> &Path {
        match self {
            Resolution::Repo(p) | Resolution::Legacy(p) => p,
        }
    }
}

#[derive(Debug)]
pub enum ResolveError {
    NotFound,
    Ambiguous(String),
    Io(String),
}

pub fn resolve_onnx(model_dir: &Path, slug: &str, revision: &str) -> Result<Resolution, ResolveError> {
    let rd = repo_dir(model_dir, slug);
    if rd.is_dir() {
        match scan_single_onnx(&rd) {
            Ok(p) => return Ok(Resolution::Repo(p)),
            Err(ResolveError::Ambiguous(_)) => return Err(ResolveError::Ambiguous(rd.to_string_lossy().to_string())),
            Err(ResolveError::NotFound) => {}
            Err(e) => return Err(e),
        }
    }

    let ld = legacy_dir(model_dir, slug, revision);
    if ld.is_dir() {
        match scan_single_onnx(&ld) {
            Ok(p) => return Ok(Resolution::Legacy(p)),
            Err(ResolveError::Ambiguous(_)) => return Err(ResolveError::Ambiguous(ld.to_string_lossy().to_string())),
            Err(ResolveError::NotFound) => {}
            Err(e) => return Err(e),
        }
    }

    Err(ResolveError::NotFound)
}

pub fn onnx_exists(model_dir: &Path, slug: &str, revision: &str) -> bool {
    resolve_onnx(model_dir, slug, revision).is_ok()
}

pub fn scan_single_onnx(dir: &Path) -> Result<PathBuf, ResolveError> {
    let mut found: Vec<PathBuf> = Vec::new();
    collect_onnx(dir, &mut found)?;
    match found.len() {
        0 => Err(ResolveError::NotFound),
        1 => Ok(found.into_iter().next().unwrap()),
        n => Err(ResolveError::Ambiguous(format!(
            "{} .onnx files found under '{}' (expected 1)",
            n,
            dir.display()
        ))),
    }
}

pub fn collect_onnx(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), ResolveError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| ResolveError::Io(format!("failed to read {:?}: {}", dir, e)))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| ResolveError::Io(format!("read_dir error: {}", e)))?;
        let path = entry.path();
        if path.is_dir() {
            collect_onnx(&path, out)?;
        } else if is_onnx_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

pub fn is_onnx_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        == Some("onnx")
        && !path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .contains("_data")
}

pub fn local_size(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_and_repo_dir() {
        let base = Path::new("/cache");
        assert_eq!(slug_dir(base, "m1"), PathBuf::from("/cache/m1"));
        assert_eq!(repo_dir(base, "m1"), PathBuf::from("/cache/m1/repo"));
        assert_eq!(legacy_dir(base, "m1", "main"), PathBuf::from("/cache/m1/main"));
        assert_eq!(meta_dir(base, "m1"), PathBuf::from("/cache/m1/meta"));
    }

    #[test]
    fn is_onnx_detection() {
        assert!(is_onnx_file(Path::new("/x/model.onnx")));
        assert!(is_onnx_file(Path::new("/x/model_fp16.onnx")));
        assert!(!is_onnx_file(Path::new("/x/model.onnx_data")));
        assert!(!is_onnx_file(Path::new("/x/model.onnx_data_1")));
        assert!(!is_onnx_file(Path::new("/x/config.json")));
        assert!(!is_onnx_file(Path::new("/x/model.onnx.gz")));
    }

    #[test]
    fn scan_single_onnx_exact_one() {
        let tmp = std::env::temp_dir().join("ms_test_single_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("onnx")).unwrap();
        std::fs::write(tmp.join("onnx/model.onnx"), b"x").unwrap();
        let r = scan_single_onnx(&tmp).unwrap();
        assert_eq!(r, tmp.join("onnx/model.onnx"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_single_onnx_ignores_data_files() {
        let tmp = std::env::temp_dir().join("ms_test_data_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("onnx")).unwrap();
        std::fs::write(tmp.join("onnx/model.onnx"), b"x").unwrap();
        std::fs::write(tmp.join("onnx/model.onnx_data"), b"y").unwrap();
        std::fs::write(tmp.join("onnx/model.onnx_data_1"), b"z").unwrap();
        let r = scan_single_onnx(&tmp).unwrap();
        assert_eq!(r, tmp.join("onnx/model.onnx"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_single_onnx_ambiguous() {
        let tmp = std::env::temp_dir().join("ms_test_ambig_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("a.onnx"), b"x").unwrap();
        std::fs::write(tmp.join("b.onnx"), b"y").unwrap();
        matches!(scan_single_onnx(&tmp), Err(ResolveError::Ambiguous(_)));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn scan_single_onnx_empty() {
        let tmp = std::env::temp_dir().join("ms_test_empty_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        matches!(scan_single_onnx(&tmp), Err(ResolveError::NotFound));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_prefers_repo_over_legacy() {
        let tmp = std::env::temp_dir().join("ms_test_resolve_");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = &tmp;
        std::fs::create_dir_all(repo_dir(base, "m1").join("onnx")).unwrap();
        std::fs::write(repo_dir(base, "m1").join("onnx/model.onnx"), b"repo").unwrap();
        std::fs::create_dir_all(legacy_dir(base, "m1", "main")).unwrap();
        std::fs::write(legacy_dir(base, "m1", "main").join("model.onnx"), b"legacy").unwrap();
        match resolve_onnx(base, "m1", "main") {
            Ok(Resolution::Repo(p)) => assert_eq!(p.file_name().unwrap().to_str(), Some("model.onnx")),
            other => panic!("expected Repo resolution, got {:?}", other),
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_falls_through_to_legacy() {
        let tmp = std::env::temp_dir().join("ms_test_legacy_");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = &tmp;
        std::fs::create_dir_all(legacy_dir(base, "m1", "main")).unwrap();
        std::fs::write(legacy_dir(base, "m1", "main").join("model.onnx"), b"legacy").unwrap();
        match resolve_onnx(base, "m1", "main") {
            Ok(Resolution::Legacy(p)) => assert_eq!(p.file_name().unwrap().to_str(), Some("model.onnx")),
            other => panic!("expected Legacy resolution, got {:?}", other),
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn local_size_missing() {
        assert_eq!(local_size(Path::new("/no/such/file_xyz_123")), None);
    }
}
