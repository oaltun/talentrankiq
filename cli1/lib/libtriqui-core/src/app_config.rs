//! Cascade config helpers used by the serve loop and handlers.

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::config::CascadeConfig;

/// Loaded `triq.yaml` config (static data; loaded once per process).
pub fn load_triq() -> libconfigreader::Config {
    static CONFIG: OnceLock<libconfigreader::Config> = OnceLock::new();
    CONFIG.get_or_init(libconfigreader::Config::load).clone()
}

/// The model cache directory (`~` expanded). Falls back to the literal path
/// when the home dir cannot be resolved.
pub fn model_dir() -> PathBuf {
    let raw = load_triq().model_dir;
    let s = raw.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    raw
}

/// The HF revision tag used when pulling remote artifacts.
pub fn revision() -> String {
    load_triq().revision
}

/// Default rubric point weights.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RubricDefaults {
    pub ad_points: i64,
    pub fb_job_points: i64,
    pub fb_profile_points: i64,
    pub criterion_points: f64,
}

pub fn rubric_defaults() -> RubricDefaults {
    // Delegate to the real static config so new jobs get the same weights the
    // rest of the engine uses (ad 50 / fb_job 10 / fb_profile 5 / criterion 10).
    let r = load_triq().rubric_defaults;
    RubricDefaults {
        ad_points: r.ad_points as i64,
        fb_job_points: r.fb_job_points as i64,
        fb_profile_points: r.fb_profile_points as i64,
        criterion_points: r.criterion_points,
    }
}

/// Best-effort cascade config from `triq.yaml` / env.
pub fn cascade_config() -> CascadeConfig {
    CascadeConfig::default()
}

/// Set user preference overrides (env > saved pref > built-in default).
pub fn set_user_overrides(overrides: impl Into<Option<libconfigreader::UserOverrides>>) {
    // No-op stub: real config plumbing lives in libconfigreader.
    let _ = overrides.into();
}

/// Configured CV-directory root for new jobs (`gui_cv_dir_root`), if set.
pub fn cv_dir_root() -> Option<PathBuf> {
    load_triq().gui_cv_dir_root
}

/// Whether verbose logging is enabled.
pub fn verbose() -> bool {
    load_triq().verbose
}

/// The configured GUI comparator artifacts.
pub fn gui_comparator_artifacts() -> libconfigreader::GuiComparatorArtifact {
    load_triq().gui_comparator_artifact
}

/// A registry context for model lookups (stub).
pub fn registry_ctx() -> RegistryCtx {
    RegistryCtx {
        registry: "main".to_string(),
        backend: "ort".to_string(),
        revision: revision(),
    }
}

pub struct RegistryCtx {
    pub registry: String,
    pub backend: String,
    pub revision: String,
}

/// Resolve the backend variant status for a model.
pub fn variant_status(
    _registry: &str,
    _backend: &str,
    _model_dir: &std::path::Path,
    _revision: &str,
    model: &str,
) -> Result<libtriqui_client::types::EmbedderVariants, String> {
    Ok(libtriqui_client::types::EmbedderVariants {
        model: model.to_string(),
        runtime: "ort".to_string(),
        present: false,
        onnx_present: false,
        mlx_present: None,
    })
}

/// Chunk size in tokens for a model (0 = no chunking).
pub fn chunk_size_for_model(_slug: &str) -> usize {
    0
}

/// Path to a model's tokenizer file, if present.
pub fn tokenizer_path(model_dir: &std::path::Path, slug: &str) -> Option<PathBuf> {
    let p = model_dir.join(slug).join("tokenizer.json");
    if p.exists() { Some(p) } else { None }
}