use std::fs;
use std::path::{Path, PathBuf};

use libmodelstorage::{default_model_dir, repo_dir, slug_dir};

pub use libmodelstorage::{local_size, DEFAULT_FILES};

#[derive(Debug, Clone, Copy)]
pub enum Exit {
    Network = 1,
    BadArgs = 2,
    SourceNotFound = 3,
    DiskWrite = 4,
    NoOnnx = 5,
    /// `--artifact mlx` finished without any `*.safetensors` landing. Kept
    /// distinct from NoOnnx so the message can say which artifact was
    /// expected: a download that quietly produces neither is a bug in the
    /// planner, not a missing file.
    NoWeights = 6,
}

impl std::fmt::Display for Exit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exit::Network => write!(f, "network error"),
            Exit::BadArgs => write!(f, "bad arguments"),
            Exit::SourceNotFound => write!(f, "source not found"),
            Exit::DiskWrite => write!(f, "disk write error"),
            Exit::NoOnnx => write!(f, "no ONNX model file found after download"),
            Exit::NoWeights => write!(f, "no safetensors weight file found after download"),
        }
    }
}

impl std::error::Error for Exit {}

impl Exit {
    pub fn code(self) -> u8 {
        self as u8
    }
}

pub const ONNX_FILES: &[&str] = &["model.onnx", "model.onnx.gz"];

pub fn verify_onnx_present(target: &Path) -> bool {
    ONNX_FILES.iter().any(|f| target.join(f).is_file())
}

/// The weight files an MLX checkpoint is made of: safetensors shards (or a
/// single `model.safetensors`) plus the index that ties shards to names.
///
/// Sharded repos (`model-00001-of-00003.safetensors`) are why this is a
/// predicate over the tree rather than a fixed-name check.
pub fn verify_safetensors_present(target: &Path) -> bool {
    let mut stack = vec![target.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".safetensors"))
            {
                return true;
            }
        }
    }
    false
}

/// Files an MLX checkpoint needs beyond the weights themselves.
///
/// Small, and all of them load-bearing: `config.json` carries the shapes,
/// quantization and the semantics this repo reads instead of guessing
/// (`use_bidirectional_attention`, pooling, rope base); `tokenizer.json` is
/// the tokenizer; the rest are what the `tokenizers` crate consults for
/// post-processor and added-token rules when it loads that file.
pub const MLX_METADATA_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
];

/// Plan an MLX artifact from a HuggingFace repo listing: every safetensors
/// shard, its index, and the metadata above — anything present, so a repo
/// that ships only some of them still downloads.
///
/// Missing metadata is deliberately *not* an error here (unlike
/// `--extra-file`): a repo without `vocab.json` is normal, and
/// `libmlxembed` fails with a precise message at load time if a file it
/// needs is genuinely absent.
pub fn plan_mlx_downloads(tree: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for path in tree {
        let name = path.rsplit('/').next().unwrap_or(path);
        let wanted = name.ends_with(".safetensors")
            || name.ends_with(".safetensors.index.json")
            || MLX_METADATA_FILES.contains(&name);
        if wanted && !out.contains(path) {
            out.push(path.clone());
        }
    }
    // Weights first: a partial download then still has the big file present,
    // and the failure a user sees is "config.json missing", not "weights
    // missing" after a long wait.
    out.sort_by_key(|p| {
        let name = p.rsplit('/').next().unwrap_or(p);
        if name.ends_with(".safetensors") {
            0
        } else if name == "config.json" {
            1
        } else {
            2
        }
    });
    out
}

pub fn resolve_model_dir(cli_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_dir {
        return p.to_path_buf();
    }
    if let Ok(p) = std::env::var("MODELDOWNLOADER_MODEL_DIR") {
        return PathBuf::from(p);
    }
    default_model_dir()
}

pub fn target_dir(model_dir: &Path, id: &str, revision: &str) -> PathBuf {
    slug_dir(model_dir, id).join(revision)
}

pub fn url_target_dir(model_dir: &Path, slug: &str) -> PathBuf {
    repo_dir(model_dir, slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_dir_layout() {
        let d = PathBuf::from("/tmp/cache");
        let p = target_dir(&d, "nomic-embed-text-v1", "main");
        assert_eq!(p, PathBuf::from("/tmp/cache/nomic-embed-text-v1/main"));
    }

    #[test]
    fn url_target_dir_layout() {
        let d = PathBuf::from("/tmp/cache");
        let p = url_target_dir(&d, "my-slug");
        assert_eq!(p, PathBuf::from("/tmp/cache/my-slug/repo"));
    }

    #[test]
    fn mlx_plan_takes_weights_index_and_metadata_from_the_tree() {
        let tree: Vec<String> = [
            ".gitattributes",
            "README.md",
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "onnx/model_q4.onnx",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let plan = plan_mlx_downloads(&tree);
        // The ONNX graph in an MLX repo is not fetched: it is a different
        // backend's artifact and would double the download for nothing.
        assert!(!plan.iter().any(|f| f.ends_with(".onnx")), "{plan:?}");
        assert!(plan.iter().any(|f| f == "model.safetensors"));
        assert!(plan.iter().any(|f| f == "config.json"));
        assert!(plan.iter().any(|f| f == "tokenizer.json"));
        // Weights are planned first so an interrupted download fails on the
        // small file, not after a long wait for the large one.
        assert_eq!(plan[0], "model.safetensors", "{plan:?}");
        assert_eq!(plan[1], "config.json", "{plan:?}");
    }

    #[test]
    fn mlx_plan_handles_sharded_weights() {
        let tree: Vec<String> = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let plan = plan_mlx_downloads(&tree);
        let shards = plan
            .iter()
            .filter(|f| f.ends_with(".safetensors") && !f.ends_with(".index.json"))
            .count();
        assert_eq!(shards, 3, "every shard, not just the first: {plan:?}");
    }

    #[test]
    fn mlx_plan_is_empty_for_a_repo_with_no_weights() {
        // So the caller can report "no weights here" instead of downloading
        // three JSON files and then failing verification.
        let tree: Vec<String> = ["README.md", "config.json", "tokenizer.json"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let plan = plan_mlx_downloads(&tree);
        assert!(plan.iter().all(|f| !f.ends_with(".safetensors")), "{plan:?}");
    }

    #[test]
    fn verify_onnx_present_no_files() {
        let tmp = std::env::temp_dir().join("md_verify_none_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!verify_onnx_present(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn verify_onnx_present_with_model() {
        let tmp = std::env::temp_dir().join("md_verify_some_");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("model.onnx"), b"x").unwrap();
        assert!(verify_onnx_present(&tmp));
        std::fs::write(tmp.join("config.json"), b"{}").unwrap();
        assert!(verify_onnx_present(&tmp));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_model_dir_env() {
        std::env::set_var("MODELDOWNLOADER_MODEL_DIR", "/tmp/fromenv/mdcache");
        let d = resolve_model_dir(None);
        assert_eq!(d, PathBuf::from("/tmp/fromenv/mdcache"));
        std::env::remove_var("MODELDOWNLOADER_MODEL_DIR");
    }

    #[test]
    fn local_size_missing() {
        assert_eq!(
            local_size(std::path::Path::new("/no/such/file_xyz_123")),
            None
        );
    }
}
