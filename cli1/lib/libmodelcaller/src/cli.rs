//! Engine options for [`crate::run_stdio_over`] — the loop's constructor
//! struct. There is no `modelcaller` binary anymore, so these are plain
//! library parameters (flags the host omits default to the same values the
//! old argv parser used).

use std::path::PathBuf;

/// A model's Metal artifact: a different file set of the same weights, in
/// MLX's safetensors layout. Kept next to the args because it travels with a
/// resolved model through the loader.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct MlxArtifact {
    pub slug: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct StdioArgs {
    /// Cache root for ONNX models (falls back to the libmodelstorage default
    /// when `None`).
    pub model_dir: Option<PathBuf>,

    /// HF revision (tag/branch/commit) to load from the cache.
    pub revision: String,

    /// Default execution provider ("auto" or explicit).
    pub exec_provider: String,
    /// Which runtime serves embeds (design 15 §15.4): `auto` takes the MLX arm
    /// where it exists, `ort` pins the CPU floor (bench), `mlx` demands Metal.
    pub backend: String,

    /// Default compute precision.
    pub precision: String,

    /// Automatically download missing models via modeldownloader (in-process).
    pub auto_download_models: bool,
}

impl Default for StdioArgs {
    fn default() -> Self {
        Self {
            model_dir: None,
            revision: "main".to_string(),
            exec_provider: "auto".to_string(),
            backend: "auto".to_string(),
            precision: "fp32".to_string(),
            auto_download_models: true,
        }
    }
}
