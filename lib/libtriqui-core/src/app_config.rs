//! Shared discovery of the config file (`triq.yaml` preferred, `ikapp.yaml`
//! fallback) and the engine's paths.
//!
//! Both the engine runtime and the model-management commands need the same
//! paths (config, model dir), so the lookup logic lives here. Everything
//! falls back gracefully: an explicit env var wins, then the config file
//! (`triq.yaml` or `ikapp.yaml`), then a filesystem walk from the repo root
//! (found by walking up from the running exe).
//!
//! `modelcaller`, `modeldownloader`, and `onnxwrapperrs` are all in-process
//! now — the gui runs the whole engine inside its own binary and needs no
//! binary paths at all (thin gui pattern, `gui/design/app1/thin-gui.md`).

use std::path::{Path, PathBuf};

use libconfigreader::Config;

/// Walk up from the running executable to find the repo root (a dir containing
/// `triq.yaml` or `ikapp.yaml`). Used to locate config + binaries when no
/// env/config is set.
pub fn repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut cur = exe.parent();
    while let Some(p) = cur {
        if p.join("triq.yaml").is_file() || p.join("ikapp.yaml").is_file() {
            return Some(p.to_path_buf());
        }
        cur = p.parent();
    }
    None
}

thread_local! {
    /// Per-thread config override (`triq.yaml`/`ikapp.yaml` path). Set on the serve thread by
    /// `InProcessServe::start_with_config` (tests, tools) so one process can
    /// host several serves, each with its own config, without process-global
    /// state like `TRIQI_MODELCALLER_CONFIG` — concurrent serves in test
    /// binaries would otherwise read each other's isolated configs.
    static CONFIG_PATH_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Override config file discovery **for the calling thread only**; pass
/// `None` to clear. All config lookups made from this thread — serve
/// handlers and the engine runtime it starts — resolve through the override.
pub fn set_config_path_override(path: Option<PathBuf>) {
    CONFIG_PATH_OVERRIDE.with(|o| *o.borrow_mut() = path);
}

/// The calling thread's current override (see
/// [`set_config_path_override`]). Serve spawns worker threads per request;
/// thread-locals don't inherit, so the serve loop captures this and the
/// workers re-apply it.
pub fn config_path_override() -> Option<PathBuf> {
    CONFIG_PATH_OVERRIDE.with(|o| o.borrow().clone())
}

/// Path to the config file (`triq.yaml` preferred, `ikapp.yaml` fallback),
/// from the per-thread override, `TRIQI_MODELCALLER_CONFIG`, or repo-root
/// discovery. An explicit env var only wins when it names an existing file;
/// a stale or bogus `TRIQI_MODELCALLER_CONFIG` falls through to repo-root
/// discovery instead of silently making config undiscoverable.
pub fn ikapp_config_path() -> Option<PathBuf> {
    if let Some(p) = CONFIG_PATH_OVERRIDE.with(|o| o.borrow().clone()) {
        return Some(p);
    }
    if let Ok(p) = std::env::var("TRIQI_MODELCALLER_CONFIG") {
        if !p.is_empty() {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
            // env points to a missing file; fall through to repo_root()
        }
    }
    let root = repo_root()?;
    if root.join("triq.yaml").is_file() {
        Some(root.join("triq.yaml"))
    } else {
        Some(root.join("ikapp.yaml"))
    }
}

/// Load the config file (`triq.yaml` or `ikapp.yaml`) (best-effort). Returns `None` if it can't be read/parsed.
pub fn load_ikapp() -> Option<Config> {
    let path = ikapp_config_path()?;
    libconfigreader::load(&path, &libconfigreader::CliOverrides::new()).ok()
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
    if let Some(cfg) = load_ikapp() {
        if !cfg.model_dir.as_os_str().is_empty() {
            return cfg.model_dir;
        }
    }
    expand_home("~/.cache/talentrankiq/models")
}

/// ONNX model revision (HF tag/branch/commit) used for cache resolution.
pub fn revision() -> String {
    load_ikapp()
        .map(|c| c.revision)
        .unwrap_or_else(|| "main".to_string())
}

/// Whether verbose engine logging is requested (env or config).
pub fn verbose() -> bool {
    std::env::var("TRIQI_MODELCALLER_VERBOSE")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
        || load_ikapp().map(|c| c.verbose).unwrap_or(false)
}

/// Idle-unload timeout for the embedding engine, from `gui_idle_unload_secs` in
/// `ikapp.yaml` (default 300 s = 5 minutes, `0` disables idle unloading).
pub fn gui_idle_unload_secs() -> u64 {
    load_ikapp()
        .map(|c| c.gui_idle_unload_secs)
        .unwrap_or(libconfigreader::DEFAULT_GUI_IDLE_UNLOAD_SECS)
}

/// Texts per embeddings request. `engine.embed_batch` in `ikapp.yaml` is an
/// optional ceiling; unset, the batch derives from the libsyscap budget:
/// the budget's batch when it has an opinion, else the thread cap
/// (capacity when idle — 10 on this box — halved under Polite, 1 under
/// Backoff/Hold). Sampled once per process (the value is read into a
/// `LazyLock` by the embedding path; libsyscap design §7.3).
pub fn engine_embed_batch() -> usize {
    let cfg = load_ikapp();
    let b = crate::syscap::budget_now(
        &cfg.as_ref().map(|c| c.syscap.clone()).unwrap_or_default(),
        &cfg.as_ref().map(|c| c.engine.clone()).unwrap_or_default(),
    );
    crate::syscap::effective_embed_batch(&b)
}

/// `true` when `engine.no_cache` is set in `ikapp.yaml`: never serve a
/// vector-embedding or judge verdict from cache — recompute everything for
/// honest cold timings. Writes still happen, so a benchmark run leaves the
/// warm caches intact.
pub fn engine_no_cache() -> bool {
    load_ikapp()
        .and_then(|c| c.engine.no_cache)
        .unwrap_or(false)
}

/// Judge stage artifacts for the gui rubric, from `gui_judge_artifact` in
/// `ikapp.yaml` (empty = judge stage disabled).
pub fn gui_judge_artifacts() -> libconfigreader::GuiJudgeArtifacts {
    load_ikapp()
        .and_then(|c| c.gui_judge_artifact)
        .unwrap_or_default()
}

/// Default rubric weights for new jobs and new criteria, from `rubric_defaults`
/// in `ikapp.yaml` (ad 50, fb-job 10, fb-profile 5, criterion 10 when unset).
pub fn rubric_defaults() -> libconfigreader::RubricDefaults {
    load_ikapp().map(|c| c.rubric_defaults).unwrap_or_default()
}

/// Root folder for new jobs' default CV directories, from `gui_cv_dir_root` in
/// `ikapp.yaml` (`~` and relative paths are resolved by `libconfigreader`).
/// Fallback only: a saved gui preference (Settings → General → CV directory,
/// `AppPreferences::cv_dir_root`) wins over this; when both are unset the
/// caller uses the platform Documents dir + `jobs`.
pub fn cv_dir_root() -> Option<PathBuf> {
    load_ikapp().and_then(|c| c.gui_cv_dir_root)
}

/// Cascade config for progressive ranking, from `cascade:` in `ikapp.yaml`.
pub fn cascade_config() -> libconfigreader::CascadeSection {
    load_ikapp().map(|c| c.cascade).unwrap_or_default()
}

/// Best-effort: convert a `PathBuf` to a `&str` for passing to the client.
pub fn as_str(p: &Path) -> &str {
    p.to_str().unwrap_or("")
}

/// Chunk size for a given model slug from `gui_embedders` in `ikapp.yaml`, and
/// from `cascade.scout_chunk_size` for the cascade's scout (which is not a
/// `gui_embedders` entry — it is a stage, not a user-selectable embedder).
/// Returns 0 (no chunking) if the model is in neither list.
pub fn chunk_size_for_model(slug: &str) -> usize {
    if let Some(cfg) = load_ikapp() {
        if let Some(ge) = cfg.gui_embedders.iter().find(|ge| ge.slug == slug) {
            return ge.chunk_size as usize;
        }
        if cfg.cascade.scout() == Some(slug) {
            return cfg.cascade.scout_chunk_size();
        }
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
    std::sync::RwLock<Option<libconfigreader::EngineSection>>,
> = std::sync::OnceLock::new();

/// Replace the engine knobs every `EngineRuntime` starts with, regardless of
/// what `ikapp.yaml` says — the `--cpu-only` gui flag sets this before the
/// serve (and therefore the engine) comes up: ONNX on the CPU EP with
/// one-thread sessions and small batches, i.e. a deliberately slow floor for
/// testing progressive-ranking behaviour on weak machines. Fields the
/// override leaves `None` fall through to the config.
pub fn set_engine_override(o: Option<libconfigreader::EngineSection>) {
    let lock = ENGINE_OVERRIDE.get_or_init(|| std::sync::RwLock::new(None));
    *lock.write().expect("engine override lock poisoned") = o;
}

/// The process-wide engine override (see [`set_engine_override`]).
pub fn engine_override() -> Option<libconfigreader::EngineSection> {
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
pub fn cpu_only_engine() -> libconfigreader::EngineSection {
    libconfigreader::EngineSection {
        backend: Some(libconfigreader::Backend::Ort),
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

pub fn registry_ctx() -> Option<std::sync::Arc<RegistryCtx>> {
    static CTX: std::sync::OnceLock<Option<std::sync::Arc<RegistryCtx>>> =
        std::sync::OnceLock::new();
    CTX.get_or_init(|| {
        let cfg = load_ikapp()?;
        let backend = cfg.engine.backend.unwrap_or_default();
        let revision = cfg.revision.clone();
        Some(std::sync::Arc::new(RegistryCtx {
            registry: libmodelcaller::Registry::from_config(&cfg),
            backend,
            revision,
        }))
    })
    .clone()
}

pub struct RegistryCtx {
    pub registry: libmodelcaller::Registry,
    pub backend: libconfigreader::Backend,
    pub revision: String,
}
