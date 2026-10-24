//! Shared application configuration.
//!
//! The configuration is now embedded as pure Rust data in libconfigreader.
//! There is no YAML file loading anymore — Config::load() returns the static config.

use std::path::{Path, PathBuf};

use libconfigreader::{Config, EngineConfig, UserOverrides};

/// Load the embedded configuration.
/// User preference overrides (from `AppPreferences`) are applied if set.
pub fn load_triq() -> Config {
    let cfg = Config::load();
    // Apply user overrides if set
    if let Some(overrides) = user_overrides() {
        if !overrides.is_default() {
            return cfg.with_user_overrides(&overrides);
        }
    }
    cfg
}

thread_local! {
    /// Per-thread user overrides (from AppPreferences), applied on top of the
    /// embedded config. Set once at serve startup.
    static USER_OVERRIDES: std::cell::RefCell<Option<UserOverrides>> =
        std::cell::RefCell::new(None);
}

/// Set user preference overrides for the calling thread. Pass `None` to clear.
pub fn set_user_overrides(o: Option<UserOverrides>) {
    USER_OVERRIDES.with(|cell| *cell.borrow_mut() = o);
}

/// The calling thread's user overrides (see [`set_user_overrides`]).
pub fn user_overrides() -> Option<UserOverrides> {
    USER_OVERRIDES.with(|cell| cell.borrow().clone())
}

/// Expand a leading `~` in a path to the user's home directory.
fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    if path == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

/// Model cache root: env `TRIQI_MODEL_DIR` > config `model_dir` > `~/.cache/talentrankiq/models`.
pub fn model_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TRIQI_MODEL_DIR") {
        if !p.is_empty() {
            return expand_home(&p);
        }
    }
    let cfg = load_triq();
    if !cfg.model_dir.as_os_str().is_empty() {
        return cfg.model_dir;
    }
    expand_home("~/.cache/talentrankiq/models")
}

/// ONNX model revision (HF tag/branch/commit) used for cache resolution.
pub fn revision() -> String {
    let cfg = load_triq();
    cfg.revision
}

/// Whether verbose engine logging is requested (env or config).
pub fn verbose() -> bool {
    std::env::var("TRIQI_MODELCALLER_VERBOSE")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
        || load_triq().verbose
}

/// Idle-unload timeout for the embedding engine (default 300 s = 5 minutes, `0` disables idle unloading).
pub fn gui_idle_unload_secs() -> u64 {
    load_triq().gui_idle_unload_secs
}

/// Texts per embeddings request. `engine.embed_batch` is an optional ceiling;
/// unset, the batch derives from the libsyscap budget.
pub fn engine_embed_batch() -> usize {
    let cfg = load_triq();
    let b = crate::syscap::budget_now(
        &cfg.syscap,
        &cfg.engine,
    );
    crate::syscap::effective_embed_batch(&b)
}

/// `true` when `engine.no_cache` is set: never serve a
/// vector-embedding or judge verdict from cache — recompute everything for
/// honest cold timings. Writes still happen, so a benchmark run leaves the
/// warm caches intact.
pub fn engine_no_cache() -> bool {
    load_triq().engine.no_cache.unwrap_or(false)
}

/// Judge stage artifacts for the gui rubric.
pub fn gui_judge_artifacts() -> libconfigreader::GuiJudgeArtifact {
    load_triq().gui_judge_artifact
}

/// Default rubric weights for new jobs and new criteria.
pub fn rubric_defaults() -> libconfigreader::RubricDefaults {
    load_triq().rubric_defaults
}

/// Root folder for new jobs' default CV directories.
/// Fallback only: a saved gui preference (Settings → General → CV directory,
/// `AppPreferences::cv_dir_root`) wins over this; when both are unset the
/// caller uses the platform Documents dir + `jobs`.
pub fn cv_dir_root() -> Option<PathBuf> {
    load_triq().gui_cv_dir_root
}

/// Cascade config for progressive ranking.
pub fn cascade_config() -> libconfigreader::CascadeConfig {
    load_triq().cascade
}

/// Best-effort: convert a `PathBuf` to a `&str` for passing to the client.
pub fn as_str(p: &Path) -> &str {
    p.to_str().unwrap_or("")
}

/// Chunk size for a given model slug from `gui_embedders`, and
/// from `cascade.scout_chunk_size` for the cascade's scout (which is not a
/// `gui_embedders` entry — it is a stage, not a user-selectable embedder).
/// Returns 0 (no chunking) if the model is in neither list.
pub fn chunk_size_for_model(slug: &str) -> usize {
    let cfg = load_triq();
    if let Some(ge) = cfg.gui_embedders.iter().find(|ge| ge.slug == slug) {
        return ge.chunk_size as usize;
    }
    if cfg.cascade.scout() == Some(slug) {
        return cfg.cascade.scout_chunk_size();
    }
    0
}

/// Path to the tokenizer.json for a given model slug.
pub fn tokenizer_path(model_dir: &Path, slug: &str) -> Option<std::path::PathBuf> {
    let revision = revision();
    let res = match libmodelstorage::resolve_onnx(model_dir, slug, &revision) {
        Ok(r) => r,
        Err(_) => return None,
    };
    let path = res.path().parent()?.join("tokenizer.json");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

static ENGINE_OVERRIDE: std::sync::OnceLock<
    std::sync::RwLock<Option<EngineConfig>>,
> = std::sync::OnceLock::new();

/// Replace the engine knobs every `EngineRuntime` starts with, regardless of
/// what the embedded config says — the `--cpu-only` gui flag sets this before the
/// serve (and therefore the engine) comes up: ONNX on the CPU EP with
/// one-thread sessions and small batches, i.e. a deliberately slow floor for
/// testing progressive-ranking behaviour on weak machines. Fields the
/// override leaves `None` fall through to the config.
pub fn set_engine_override(o: Option<EngineConfig>) {
    let lock = ENGINE_OVERRIDE.get_or_init(|| std::sync::RwLock::new(None));
    *lock.write().expect("engine override lock poisoned") = o;
}

/// The process-wide engine override (see [`set_engine_override`]).
pub fn engine_override() -> Option<EngineConfig> {
    ENGINE_OVERRIDE
        .get_or_init(|| std::sync::RwLock::new(None))
        .read()
        .expect("engine override lock poisoned")
        .clone()
}

/// A slow machine in software: the `--cpu-only` engine override — ONNX arm
/// pinned to the CPU execution provider, single-threaded sessions, small
/// embed batches. Used by the gui's `--cpu-only` flag to rehearse
/// low-end-machine behaviour on fast hardware.
pub fn cpu_only_engine() -> EngineConfig {
    EngineConfig {
        backend: libconfigreader::Backend::Ort,
        exec_provider: Some(libconfigreader::ExecutionProvider::Cpu),
        intra_threads: Some(1),
        inter_threads: Some(1),
        embed_batch: Some(4),
        ..Default::default()
    }
}

/// Registry + backend + revision, memoized: the inputs modelcaller needs to
/// answer engine-free questions (`variant_status` — which arm serves a slug,
/// are its weights on disk). Config is process-immutable, so one load serves
/// every embed wave and serve request.
/// Re-exported so serve handlers can ask registry-side variant questions
/// (onnx/mlx presence, resolved arm) without linking `modelcaller` directly.
pub use libmodelcaller::{variant_status, VariantStatus};

pub fn registry_ctx() -> std::sync::Arc<RegistryCtx> {
    static CTX: std::sync::OnceLock<std::sync::Arc<RegistryCtx>> =
        std::sync::OnceLock::new();
    CTX.get_or_init(|| {
        let cfg = load_triq();
        let backend = cfg.engine.backend;
        let revision = cfg.revision.clone();
        std::sync::Arc::new(RegistryCtx {
            registry: libmodelcaller::Registry::from_config(&cfg),
            backend,
            revision,
        })
    })
    .clone()
}

pub struct RegistryCtx {
    pub registry: libmodelcaller::Registry,
    pub backend: libconfigreader::Backend,
    pub revision: String,
}