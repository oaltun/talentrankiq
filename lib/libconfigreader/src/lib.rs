use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::de::{Deserializer, Error as DeError, Visitor};
use serde::{Deserialize, Serialize};

/// Returns true when `sha256` is `None` or empty — the skip-serializing guard
/// so the field is omitted from YAML when no checksum is configured.
fn is_default_sha256(sha: &Option<String>) -> bool {
    sha.as_deref().map_or(true, |s| s.is_empty())
}

pub mod config;

/// Default `gui_idle_unload_secs` when `ikapp.yaml` does not set it: unload the
/// model after 5 minutes without a request. `0` disables idle unloading.
pub const DEFAULT_GUI_IDLE_UNLOAD_SECS: u64 = 300;

/// Default rubric weights when `ikapp.yaml` sets no `rubric_defaults`:
/// job ad 50, like-feedback (this job) 10, like-feedback (all jobs) 5,
/// each new criterion 10.
pub const DEFAULT_RUBRIC_AD_POINTS: f64 = 50.0;
pub const DEFAULT_RUBRIC_FB_JOB_POINTS: f64 = 10.0;
pub const DEFAULT_RUBRIC_FB_PROFILE_POINTS: f64 = 5.0;
pub const DEFAULT_RUBRIC_CRITERION_POINTS: f64 = 10.0;

/// Weights assigned to a new job's rubric rows and to each newly added
/// criterion. Existing jobs and criteria keep their stored weights.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct RubricDefaults {
    #[serde(default = "default_ad_points")]
    pub ad_points: f64,
    #[serde(default = "default_fb_job_points")]
    pub fb_job_points: f64,
    #[serde(default = "default_fb_profile_points")]
    pub fb_profile_points: f64,
    #[serde(default = "default_criterion_points")]
    pub criterion_points: f64,
}

fn default_ad_points() -> f64 {
    DEFAULT_RUBRIC_AD_POINTS
}
fn default_fb_job_points() -> f64 {
    DEFAULT_RUBRIC_FB_JOB_POINTS
}
fn default_fb_profile_points() -> f64 {
    DEFAULT_RUBRIC_FB_PROFILE_POINTS
}
fn default_criterion_points() -> f64 {
    DEFAULT_RUBRIC_CRITERION_POINTS
}

impl Default for RubricDefaults {
    fn default() -> Self {
        RubricDefaults {
            ad_points: DEFAULT_RUBRIC_AD_POINTS,
            fb_job_points: DEFAULT_RUBRIC_FB_JOB_POINTS,
            fb_profile_points: DEFAULT_RUBRIC_FB_PROFILE_POINTS,
            criterion_points: DEFAULT_RUBRIC_CRITERION_POINTS,
        }
    }
}

/// `cascade:` section: progressive-ranking cascade knobs (stage-0 instant
/// sort + embed/judge waves; see `gui/design/app1/progressive-ranking.md`).
/// Every field is optional; absent fields and an absent section fall back to
/// the defaults.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CascadeSection {
    /// Candidates embedded per wave, best-current-order first. Default 100.
    pub k1_embed_per_wave: Option<usize>,
    /// DEPRECATED (scoring v3): the judge pool is frozen once per run, not
    /// grown per wave. Ignored; use `judge_pool_size`.
    pub k2_judge_per_wave: Option<usize>,
    /// How many candidates the judge compares against each other. The pool is
    /// frozen when a run opens it and never re-picked, because judge win-rates
    /// are set-relative. `0` turns the judge off. Default 50.
    pub judge_pool_size: Option<usize>,
    /// Jobs at or below this candidate count rank in one full pass instead
    /// of running waves (the cascade degenerates to it anyway). Default 300.
    pub small_job_threshold: Option<usize>,
    /// Scout embedder slug (cascade stage 0.5, the fair multilingual gate) —
    /// a `models:` slug. Unset = the scout stage does not run.
    pub scout_model: Option<String>,
    /// Whether the scout stage runs. Default: on whenever `scout_model` is set,
    /// so enabling the scout is a one-line change.
    pub scout_enabled: Option<bool>,
    /// Chunk size used to split CVs for the scout. The scout is not a
    /// `gui_embedders` entry (it is not user-selectable), so its chunking comes
    /// from here. Default [`DEFAULT_SCOUT_CHUNK_SIZE`].
    pub scout_chunk_size: Option<usize>,
    /// Static (Model2Vec-style) stage-0 semantic prior slug, pure Rust
    /// token-lookup + mean, no ONNX. Reserved — not yet wired.
    pub stage0_static_prior: Option<String>,
}

impl Default for CascadeSection {
    fn default() -> Self {
        CascadeSection {
            k1_embed_per_wave: None,
            k2_judge_per_wave: None,
            judge_pool_size: None,
            small_job_threshold: None,
            scout_model: None,
            scout_enabled: None,
            scout_chunk_size: None,
            stage0_static_prior: None,
        }
    }
}

impl CascadeSection {
    pub fn k1(&self) -> usize {
        self.k1_embed_per_wave.unwrap_or(DEFAULT_CASCADE_K1)
    }
    /// The frozen judge pool size. `k2_judge_per_wave` is the superseded
    /// per-wave knob and is read only when `judge_pool_size` is unset, so an
    /// existing config keeps its number.
    pub fn pool_size(&self) -> usize {
        self.judge_pool_size
            .or(self.k2_judge_per_wave)
            .unwrap_or(DEFAULT_CASCADE_POOL)
    }
    pub fn small_job_threshold(&self) -> usize {
        self.small_job_threshold
            .unwrap_or(DEFAULT_CASCADE_SMALL_JOB)
    }
    /// Chunk size for the scout's chunking.
    pub fn scout_chunk_size(&self) -> usize {
        self.scout_chunk_size.unwrap_or(DEFAULT_SCOUT_CHUNK_SIZE)
    }
    /// Whether the scout stage runs: explicit `scout_enabled` wins, otherwise
    /// it follows the presence of a `scout_model`.
    pub fn scout_enabled(&self) -> Option<bool> {
        self.scout_enabled
            .or_else(|| self.scout_model.as_ref().map(|_| true))
    }
    /// The scout slug, only when the stage is actually on.
    pub fn scout(&self) -> Option<&str> {
        if self.scout_enabled()? {
            self.scout_model.as_deref().filter(|s| !s.trim().is_empty())
        } else {
            None
        }
    }
}

pub const DEFAULT_CASCADE_K1: usize = 100;
pub const DEFAULT_CASCADE_POOL: usize = 50;
pub const DEFAULT_CASCADE_SMALL_JOB: usize = 300;
/// The scout is a cheap model, so it chunks generously; the whole point is one
/// fast pass over everything, and its context window is 512 tokens.
pub const DEFAULT_SCOUT_CHUNK_SIZE: usize = 512;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FamilyDef {
    pub family: String,
    pub usage: String,
}

/// Which ONNX execution provider the wrapper registers for a session
/// (design 15 §15.5). `Auto` (default) registers none, which is ORT's plain
/// CPU EP — today's behaviour. The others name a provider; whether it is
/// actually compiled in is the build's business (onnxwrapperrs compiles
/// CoreML on macOS), and a provider that fails to register is a load error,
/// not a silent CPU fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    #[default]
    Auto,
    Cpu,
    CoreMl,
    XnnPack,
    Cuda,
    TensorRt,
    DirectMl,
    OpenVino,
    Rocm,
}

impl ExecutionProvider {
    /// The wire name the wrapper understands.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionProvider::Auto => "auto",
            ExecutionProvider::Cpu => "cpu",
            ExecutionProvider::CoreMl => "coreml",
            ExecutionProvider::XnnPack => "xnnpack",
            ExecutionProvider::Cuda => "cuda",
            ExecutionProvider::TensorRt => "tensorrt",
            ExecutionProvider::DirectMl => "directml",
            ExecutionProvider::OpenVino => "openvino",
            ExecutionProvider::Rocm => "rocm",
        }
    }
}

/// Which runtime serves an `embed` request (design 15 §15.4/§15.5.3).
///
/// `Auto` (default) takes the MLX arm when this build has it **and** the model
/// declares an `mlx:` variant, and ONNX otherwise — so macOS gets Metal and
/// every other platform (and every build without `--features mlx`) keeps the
/// CPU floor without config changes. `Ort` pins ONNX, which is what bench does:
/// its recorded results are only comparable if the floor never moves under
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Auto,
    Ort,
    Mlx,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Ort => "ort",
            Backend::Mlx => "mlx",
        }
    }

    /// Does this setting *permit* the MLX arm? `Auto` permits it, `Mlx` demands
    /// it (a load error rather than a quiet fall back — the same rule as a
    /// provider that fails to register), `Ort` forbids it.
    pub fn allows_mlx(&self) -> bool {
        matches!(self, Backend::Auto | Backend::Mlx)
    }

    /// Does this setting *demand* MLX, so a missing arm must fail loudly?
    pub fn requires_mlx(&self) -> bool {
        matches!(self, Backend::Mlx)
    }
}

impl std::str::FromStr for ExecutionProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(ExecutionProvider::Auto),
            "cpu" => Ok(ExecutionProvider::Cpu),
            "coreml" => Ok(ExecutionProvider::CoreMl),
            "xnnpack" => Ok(ExecutionProvider::XnnPack),
            "cuda" => Ok(ExecutionProvider::Cuda),
            "tensorrt" => Ok(ExecutionProvider::TensorRt),
            "directml" => Ok(ExecutionProvider::DirectMl),
            "openvino" => Ok(ExecutionProvider::OpenVino),
            "rocm" => Ok(ExecutionProvider::Rocm),
            _ => Err(format!("unknown execution provider: {s}")),
        }
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Backend::Auto),
            "ort" => Ok(Backend::Ort),
            "mlx" => Ok(Backend::Mlx),
            _ => Err(format!("unknown backend: {s}")),
        }
    }
}

/// `engine:` section: resource caps for the ONNX inference engine
/// (modelcaller + onnxwrapperrs). Every field is optional and every field is
/// a CEILING, never a target: when unset, the libsyscap budget derives the
/// effective value from live machine capacity (`intra_threads` → capacity at
/// Full, halved under Polite, floored under Backoff/Hold; `embed_batch` →
/// the same budget). Consumers resolve via
/// `libsyscap::effective_intra_threads`/`effective_inter_threads` — unset
/// caps never fall through to the ONNX Runtime default (one worker per
/// physical core). `intra_threads`/`inter_threads` are forwarded to the
/// wrapper's session builder; `embed_batch` is consumed by embedders that
/// batch texts (triqui-core gui/bench paths keep their own knobs).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct EngineSection {
    /// Intra-op thread pool size (per session). `None` = derived from the
    /// libsyscap budget (capacity when idle).
    pub intra_threads: Option<usize>,
    /// Inter-op thread pool size (parallel graph nodes). `None` = disabled
    /// (single-threaded dispatch), which is right for small models.
    pub inter_threads: Option<usize>,
    /// Texts per embeddings request. Lower = flatter memory + finer progress,
    /// slower; higher = faster, spikier RAM. Unset = derived from the
    /// libsyscap budget (the thread cap), which measurement puts at the good
    /// end of the curve for the big embedders — bigger batches are slower.
    pub embed_batch: Option<usize>,
    /// How many sessions a batched embed is spread over (design 15 §15.7).
    /// `auto` (default) picks by the model's weight footprint; a number pins
    /// the pool. The thread budget is DIVIDED across the pool, never
    /// multiplied.
    pub embed_sessions: Option<EmbedSessions>,
    /// Execution provider for the ONNX sessions (design 15 §15.5). Unset or
    /// `auto` = ORT's CPU EP (today's behaviour). `coreml` puts batched
    /// embeds on the Apple GPU/ANE; other names are per-arch.
    pub exec_provider: Option<ExecutionProvider>,
    /// Raise ORT's own logging (node placement per provider) on load. This
    /// is the gate for any accelerated rung: a provider that silently falls
    /// back to CPU looks identical to a session that never tried.
    pub provider_logging: Option<bool>,
    /// Which runtime serves embeds (design 15 §15.4). Unset or `auto` = the MLX
    /// arm where it exists and the model declares one, ONNX everywhere else.
    /// `ort` pins the CPU floor, which is what keeps bench results comparable;
    /// `mlx` demands the Metal arm and fails loudly without it.
    pub backend: Option<Backend>,
    /// Hard cap on the modelcaller process tree: when set (> 0), modelcaller
    /// pins itself and every child it spawns (onnxwrapperrs, modeldownloader)
    /// to the first `max_cores` CPUs via CPU affinity at startup. Unlike the
    /// thread knobs this bounds TOTAL tree CPU usage, not per-session pool
    /// sizes. `None`/`0` = no pinning.
    pub max_cores: Option<usize>,
    /// Benchmark switch: when `true`, no vector-embedding or judge-verdict
    /// cache is ever consulted — every embed and every pairwise verdict is
    /// computed fresh, so timings show the true cold cost of the process.
    /// Writes are unaffected: a no-cache run leaves the warm caches intact
    /// (recomputed values overwrite their rows in place). Scheduler state
    /// (which candidate has which model's chunks) is not a cache and still
    /// applies — measure a fully cold pipeline on a freshly imported job.
    /// Default: `false` (caches on).
    pub no_cache: Option<bool>,
}

/// `syscap.build_jobs:` — cargo `-j` for the build tools (§12 of the
/// libsyscap design). `auto` derives it from the live budget (engine thread
/// budget, 1 under load); a number is the human override. Anything else is a
/// config error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuildJobs {
    Auto,
    Fixed(u32),
}

pub const DEFAULT_SYSCAP_BUILD_JOBS: BuildJobs = BuildJobs::Auto;

impl Serialize for BuildJobs {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            BuildJobs::Auto => s.serialize_str("auto"),
            BuildJobs::Fixed(n) => s.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for BuildJobs {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = BuildJobs;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("\"auto\" or a job count")
            }
            fn visit_u64<E: DeError>(self, v: u64) -> Result<BuildJobs, E> {
                u32::try_from(v)
                    .map(BuildJobs::Fixed)
                    .map_err(|_| E::custom(format!("build_jobs {v} out of range")))
            }
            fn visit_str<E: DeError>(self, v: &str) -> Result<BuildJobs, E> {
                if v.eq_ignore_ascii_case("auto") {
                    Ok(BuildJobs::Auto)
                } else {
                    Err(E::custom(format!(
                        "unknown build_jobs value '{v}', expected \"auto\" or a job count"
                    )))
                }
            }
        }
        d.deserialize_any(V)
    }
}

/// `syscap:` section: system capability + live-load budget knobs
/// (`lib/libsyscap`). The budget decides how much CPU/RAM the gui, bench, and
/// tests may use RIGHT NOW — always clamped to the `engine:` caps above (it
/// can only lower the load on a busy machine, never raise it). Every field is
/// optional; an absent section falls back to the `DEFAULT_SYSCAP_*` constants.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SyscapSection {
    /// Master switch. `false` = the budget reports the configured engine caps
    /// untouched and samples no load. Default: `true`.
    pub enabled: Option<bool>,
    /// Whole cores kept for the user (headroom). Default: 1.
    pub reserve_cores: Option<u32>,
    /// ...or this fraction of the logical cores, whichever reserve is larger.
    /// Default: 0.10. Capacity counts every core (P and E alike); the reserve
    /// is what the user keeps whole.
    pub reserve_core_frac: Option<f64>,
    /// Free-RAM fraction below which the budget drops to Backoff. Default 0.20.
    pub min_free_mem_frac: Option<f64>,
    /// Free-RAM fraction below which heavy work pauses (Hold; the gui must
    /// surface this to the user). Default: 0.05.
    pub hold_free_mem_frac: Option<f64>,
    /// 1-minute load average per logical core above which the budget drops to
    /// Polite. Default: 1.0 — i.e. only when the run queue exceeds the machine
    /// (more runnable work than cores). Sub-1.0 load is NOT politeness on a
    /// many-core box: one background compile, or a consumer's own threads,
    /// legitimately load 60 %+ of an 18-core machine while the desktop sits
    /// idle, and yielding there just wastes the idle cores (and makes an
    /// embedder throttle itself mid-wave, since its own threads are in the
    /// load average).
    pub load_backoff: Option<f64>,
    /// 1-minute load average per logical core above which the budget floors
    /// everything (Backoff) — twice oversubscribed, or RAM short. Default: 2.0.
    pub load_stall: Option<f64>,
    /// Cap the budget at Polite while on battery power. Default: `true`.
    pub on_battery_polite: Option<bool>,
    /// Minimum wall time between load samples in ms (sysinfo needs >= 200 ms
    /// between refreshes for a meaningful cpu%). Default: 250.
    pub sample_min_interval_ms: Option<u64>,
    /// Cargo `-j` for the build tools (`ikbuild`, `benchdriverrs`): `auto`
    /// derives it from the live budget, a number overrides. Default: `auto`.
    pub build_jobs: Option<BuildJobs>,
}

/// How many sessions a batched embed is spread over (design 15 §15.7).
/// `Auto` (default) lets the wrapper decide by the model's weight footprint:
/// a cache-resident model is compute-bound and scales across sessions, a
/// big one is bandwidth-bound and does not. A fixed count overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedSessions {
    Auto,
    Fixed(u32),
}

impl Default for EmbedSessions {
    fn default() -> Self {
        EmbedSessions::Auto
    }
}

impl Serialize for EmbedSessions {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            EmbedSessions::Auto => s.serialize_str("auto"),
            EmbedSessions::Fixed(n) => s.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for EmbedSessions {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Text(String),
            Num(u32),
        }
        match Raw::deserialize(d)? {
            Raw::Num(n) => Ok(EmbedSessions::Fixed(n)),
            Raw::Text(t) if t.eq_ignore_ascii_case("auto") => Ok(EmbedSessions::Auto),
            Raw::Text(t) => Err(serde::de::Error::custom(format!(
                "engine.embed_sessions must be \"auto\" or a session count, got {t:?}"
            ))),
        }
    }
}

pub const DEFAULT_SYSCAP_RESERVE_CORES: u32 = 1;
pub const DEFAULT_SYSCAP_RESERVE_CORE_FRAC: f64 = 0.10;
pub const DEFAULT_SYSCAP_MIN_FREE_MEM_FRAC: f64 = 0.20;
pub const DEFAULT_SYSCAP_HOLD_FREE_MEM_FRAC: f64 = 0.05;
pub const DEFAULT_SYSCAP_LOAD_BACKOFF: f64 = 1.0;
pub const DEFAULT_SYSCAP_LOAD_STALL: f64 = 2.0;
pub const DEFAULT_SYSCAP_ON_BATTERY_POLITE: bool = true;
pub const DEFAULT_SYSCAP_SAMPLE_MIN_INTERVAL_MS: u64 = 250;

impl SyscapSection {
    pub fn enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }
    pub fn reserve_cores(&self) -> u32 {
        self.reserve_cores.unwrap_or(DEFAULT_SYSCAP_RESERVE_CORES)
    }
    pub fn reserve_core_frac(&self) -> f64 {
        self.reserve_core_frac.unwrap_or(DEFAULT_SYSCAP_RESERVE_CORE_FRAC)
    }
    pub fn min_free_mem_frac(&self) -> f64 {
        self.min_free_mem_frac.unwrap_or(DEFAULT_SYSCAP_MIN_FREE_MEM_FRAC)
    }
    pub fn hold_free_mem_frac(&self) -> f64 {
        self.hold_free_mem_frac.unwrap_or(DEFAULT_SYSCAP_HOLD_FREE_MEM_FRAC)
    }
    pub fn load_backoff(&self) -> f64 {
        self.load_backoff.unwrap_or(DEFAULT_SYSCAP_LOAD_BACKOFF)
    }
    pub fn load_stall(&self) -> f64 {
        self.load_stall.unwrap_or(DEFAULT_SYSCAP_LOAD_STALL)
    }
    pub fn on_battery_polite(&self) -> bool {
        self.on_battery_polite.unwrap_or(DEFAULT_SYSCAP_ON_BATTERY_POLITE)
    }
    pub fn sample_min_interval_ms(&self) -> u64 {
        self.sample_min_interval_ms
            .unwrap_or(DEFAULT_SYSCAP_SAMPLE_MIN_INTERVAL_MS)
    }
    pub fn build_jobs(&self) -> BuildJobs {
        self.build_jobs.unwrap_or(DEFAULT_SYSCAP_BUILD_JOBS)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GuiEmbedderDef {
    pub slug: String,
    pub name: String,
    pub memory_mb: u32,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub chunk_size: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelDef {
    pub slug: String,
    pub family: String,
    pub kind: String,
    pub dim: u32,
    pub max_context_window: u32,
    /// Model artifacts (variants). At least one required.
    pub artifacts: Vec<ModelArtifact>,
    /// Multi-vector (late interaction / ColBERT) model: `embed` returns one
    /// vector per token (dim = per-token size) instead of one pooled vector
    /// per text. Projection weights from `extra_files` are applied by
    /// onnxwrapperrs. See bench/bench1/design/17-late-interaction.md.
    #[serde(default)]
    pub multi_vector: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArtifact {
    /// Backend this artifact serves: `onnx` or `mlx`.
    #[serde(rename = "backend")]
    pub backend: Backend,
    /// Quantization/format identifier (e.g., `q4`, `fp16`, `q4f16`, `4bit`).
    pub quantization: String,
    /// Where to fetch it (`hf:<org>/<repo>`).
    pub source: String,
    /// Download URL for the artifact.
    pub url: String,
    /// Additional files to download alongside the main artifact.
    #[serde(default)]
    pub extra_files: Vec<String>,
    /// sha256 checksum of the primary artifact (model.onnx or model.safetensors).
    /// When present, the downloader verifies the download matches this hash.
    #[serde(default, skip_serializing_if = "is_default_sha256")]
    pub sha256: Option<String>,
}

/// Executable paths for the binaries that still exist. `modelcaller`,
/// `modeldownloader`, and `onnxwrapperrs` are libraries now — every consumer
/// runs them in-process, so they need no path here.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BinaryPaths {
    pub bench: Option<PathBuf>,
    pub benchleaderboard: Option<PathBuf>,
    pub judgetrainer: Option<PathBuf>,
}

/// `judgetrainer:` section: every tunable for judge training sweeps
/// (consumed by `judgetrainer` and `judge/judgetrainer/train-m0.nu` /
/// `train-sweep.nu`). Paths are absolute or relative to the ikapp.yaml dir.
/// Absent fields fall back to the `judgetrainer train` CLI defaults.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct JudgetrainerSection {
    // ---- paths ----
    /// cnam train.csv (texts for training; test.csv is never touched)
    pub train_csv: Option<String>,
    /// build-dataset output dir; re-applies its recorded split when set
    pub dataset_dir: Option<String>,
    /// Persistent directory for judge artifacts (`judge-<rung>-<tag>-e<N>.json`)
    pub out_dir: Option<String>,
    // ---- sweep ladder (train-sweep.nu) ----
    /// Ladder rungs to sweep: m0 | m1 | m2 | m3
    pub rungs: Option<Vec<String>>,
    /// Real embedder slug (must exist in `models:`)
    pub embedder: Option<String>,
    // ---- data budget ----
    /// Cap train jobs (smoke/first-signal runs only)
    pub max_jobs: Option<usize>,
    /// Validation fraction when no dataset_dir is set
    pub val_frac: Option<f64>,
    /// Max texts per modelcaller embed call (keep small for long texts)
    pub embed_batch: Option<usize>,
    // ---- features ----
    /// none | ctx_size | fixed
    pub chunking: Option<String>,
    /// Fixed chunk size in tokens (only with chunking=fixed)
    pub chunk_size: Option<usize>,
    /// off | static
    pub contextual: Option<String>,
    /// none | lowercase
    pub text_preprocessing: Option<String>,
    // ---- optimization ----
    /// Default epochs per run (`--continue` adds to it)
    pub epochs: Option<u32>,
    pub patience: Option<u32>,
    pub lr: Option<f64>,
    pub wd: Option<f64>,
    pub batch: Option<usize>,
    pub seed: Option<u64>,
    // ---- capacity / overrides ----
    /// Hidden width (m2/m3)
    pub hidden: Option<usize>,
    /// Engineered stats coverage threshold tau
    pub tau: Option<f64>,
    /// Embedding dim override (auto-queried from modelcaller when absent)
    pub dim: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BenchSection {
    pub dataset: Option<String>,
    pub task: Option<String>,
    /// Where bench result yamls + `leaderboard.html` are written. A leading
    /// `~` expands to the home directory; a relative path resolves against
    /// the config file's directory, so the committed result corpus can live
    /// in the repo.
    pub results_dir: Option<String>,
    /// Where the derived caches live (embeddings here; the `../rerank`,
    /// `../late`, `../evidence` and `../schemas` siblings are derived from it).
    /// Same `~`/relative resolution as `results_dir`.
    pub cache_dir: Option<String>,
    pub baselines: Option<Vec<String>>,
    pub embed_models: Option<Vec<String>>,
    pub chunked_variants: Option<ChunkedVariants>,
    pub timezone: Option<String>,
}

impl BenchSection {
    /// Expand `chunked_variants` into the concrete list of combinations,
    /// following GitHub Actions matrix semantics (matrix / include / exclude).
    pub fn chunked_variant_combos(&self) -> Vec<HashMap<String, String>> {
        match &self.chunked_variants {
            Some(spec) => expand_matrix(&spec.matrix, &spec.include, &spec.exclude),
            None => Vec::new(),
        }
    }
}

/// Expand a matrix exactly like GitHub Actions `strategy.matrix`:
/// 1. cartesian product of `matrix` axes (axis order is sorted for determinism),
/// 2. merge each `include` entry into matching combinations (or add as new),
/// 3. drop any combination that contains an `exclude` entry.
pub fn expand_matrix(
    matrix: &HashMap<String, Vec<String>>,
    include: &[HashMap<String, String>],
    exclude: &[HashMap<String, String>],
) -> Vec<HashMap<String, String>> {
    let mut axes: Vec<&String> = matrix.keys().collect();
    axes.sort();
    let mut combos: Vec<HashMap<String, String>> = vec![HashMap::new()];
    for axis in axes {
        let values = &matrix[axis];
        let mut next = Vec::with_capacity(combos.len() * values.len());
        for combo in &combos {
            for value in values {
                let mut extended = combo.clone();
                extended.insert(axis.clone(), value.clone());
                next.push(extended);
            }
        }
        combos = next;
    }

    for entry in include {
        let mut additions: Vec<HashMap<String, String>> = Vec::new();
        let mut merged = false;
        for combo in &combos {
            if is_compatible(entry, combo) {
                let mut clone = combo.clone();
                for (k, v) in entry {
                    clone.insert(k.clone(), v.clone());
                }
                additions.push(clone);
                merged = true;
            }
        }
        if !merged {
            additions.push(entry.clone());
        }
        combos.extend(additions);
    }

    combos.retain(|combo| !exclude.iter().any(|exc| is_subset(exc, combo)));
    combos
}

/// True when every key/value in `small` is present in `big` (used for `exclude`).
fn is_subset(small: &HashMap<String, String>, big: &HashMap<String, String>) -> bool {
    small
        .iter()
        .all(|(k, v)| big.get(k).map_or(false, |bv| bv == v))
}

/// True when `inc` can be merged into `combo` without overwriting a different
/// value (used for `include`): every key in `inc` either is absent from `combo`
/// or already equals `inc`'s value.
fn is_compatible(inc: &HashMap<String, String>, combo: &HashMap<String, String>) -> bool {
    inc.iter()
        .all(|(k, v)| combo.get(k).map_or(true, |bv| bv == v))
}

/// GitHub-Actions-style matrix for `chunked_variants`.
///
/// Expansion semantics follow the GitHub Actions `strategy.matrix` feature
/// (matrix / include / exclude):
/// <https://docs.github.com/actions/using-jobs/using-a-matrix-for-your-jobs>
///
/// - `matrix`: axes mapped to their possible values; the cartesian product
///   forms the base set of combinations.
/// - `include`: each entry is merged into every base combination whose keys it
///   is a subset of (adding/overwriting fields); if it matches no base
///   combination it becomes its own combination.
/// - `exclude`: each entry removes any combination it is a subset of.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChunkedVariants {
    #[serde(default)]
    pub matrix: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub include: Vec<HashMap<String, String>>,
    #[serde(default)]
    pub exclude: Vec<HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RawConfig {
    pub model_dir: Option<PathBuf>,
    pub verbose: Option<bool>,
    pub logfile: Option<PathBuf>,
    pub revision: Option<String>,
    pub binaries: Option<BinaryPaths>,
    pub models: Option<Vec<ModelDef>>,
    pub families: Option<Vec<FamilyDef>>,
    pub gui_embedders: Option<Vec<GuiEmbedderDef>>,
    /// Seconds of engine inactivity after which the gui unloads the model
    /// (shuts modelcaller down; it restarts on next use). `0` disables.
    pub gui_idle_unload_secs: Option<u64>,
    /// Root folder for new jobs' default CV directories (`<root>/job-N`).
    /// `~` expands to the home directory; relative paths resolve against this
    /// file's dir. Unset = the platform Documents dir + `jobs`.
    pub gui_cv_dir_root: Option<PathBuf>,
    /// Judge stage for the gui rubric: a single trained pairwise judge
    /// artifact path, or a map with a `default` path plus per-embedder-slug
    /// overrides. When the resolved artifact's embedder/dim match the selected
    /// gui embedder, the "Job ad" cosine row is replaced by a
    /// "Job fit (judge)" win-rate row. Absent = judge stage disabled.
    pub gui_judge_artifact: Option<GuiJudgeArtifacts>,
    /// Default rubric weights for new jobs and new criteria. Absent fields and
    /// an absent section fall back to the `DEFAULT_RUBRIC_*` constants.
    pub rubric_defaults: Option<RubricDefaults>,
    pub bench: Option<BenchSection>,
    pub judgetrainer: Option<JudgetrainerSection>,
    /// ONNX engine resource caps (`intra_threads`/`inter_threads`/`embed_batch`).
    pub engine: Option<EngineSection>,
    /// Progressive-ranking cascade knobs (`k1`/`k2`/small-job threshold).
    pub cascade: Option<CascadeSection>,
    /// System capability + live-load budget knobs (`lib/libsyscap`).
    pub syscap: Option<SyscapSection>,
}

/// One `gui_judge_artifact` entry: either a single artifact path (string
/// form — applies to every embedder whose spec matches) or a mapping with a
/// `default` path plus per-embedder-slug overrides, e.g.
/// `{default: /path/600m.json, harrier-oss-v1-270m-ONNX_model_q4f16: /path/270m.json}`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GuiJudgeArtifacts {
    /// Used when no per-embedder override matches.
    pub default: Option<PathBuf>,
    /// Embedder slug → artifact path.
    pub by_embedder: std::collections::BTreeMap<String, PathBuf>,
}

impl GuiJudgeArtifacts {
    /// Artifact for embedder `slug`: the per-embedder override, else the
    /// default (`None` when the entry is empty or has no match).
    pub fn for_embedder(&self, slug: &str) -> Option<&Path> {
        self.by_embedder
            .get(slug)
            .map(|p| p.as_path())
            .or(self.default.as_deref())
    }

    pub fn is_empty(&self) -> bool {
        self.default.is_none() && self.by_embedder.is_empty()
    }
}

impl<'de> Deserialize<'de> for GuiJudgeArtifacts {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_yaml::Value::deserialize(d)?;
        match value {
            serde_yaml::Value::String(s) => Ok(Self {
                default: Some(PathBuf::from(s)),
                by_embedder: std::collections::BTreeMap::new(),
            }),
            serde_yaml::Value::Mapping(m) => {
                let mut this = Self::default();
                for (k, v) in m {
                    let path: PathBuf =
                        serde::Deserialize::deserialize(v).map_err(serde::de::Error::custom)?;
                    match k {
                        serde_yaml::Value::String(s) if s == "default" => {
                            this.default = Some(path);
                        }
                        serde_yaml::Value::String(slug) => {
                            this.by_embedder.insert(slug, path);
                        }
                        _ => {
                            return Err(serde::de::Error::custom(
                                "gui_judge_artifact map keys must be strings",
                            ))
                        }
                    }
                }
                if this.default.is_none() && this.by_embedder.is_empty() {
                    return Err(serde::de::Error::custom(
                        "gui_judge_artifact map must set `default` or at least one embedder slug",
                    ));
                }
                Ok(this)
            }
            _ => Err(serde::de::Error::custom(
                "gui_judge_artifact must be an artifact path or a {default, <slug>: path} map",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_dir: PathBuf,
    pub verbose: bool,
    pub logfile: Option<PathBuf>,
    pub revision: String,
    pub binaries: BinaryPaths,
    pub models: Vec<ModelDef>,
    pub families: Vec<FamilyDef>,
    pub gui_embedders: Vec<GuiEmbedderDef>,
    /// Idle-unload timeout for the gui engine; defaults to 300 (5 minutes),
    /// `0` disables idle unloading.
    pub gui_idle_unload_secs: u64,
    /// Root folder for new jobs' default CV directories (see
    /// `RawConfig::gui_cv_dir_root`). `None` = caller applies the platform
    /// default.
    pub gui_cv_dir_root: Option<PathBuf>,
    /// Judge stage for the gui rubric (see `RawConfig::gui_judge_artifact`).
    pub gui_judge_artifact: Option<GuiJudgeArtifacts>,
    /// Default rubric weights for new jobs and new criteria.
    pub rubric_defaults: RubricDefaults,
    pub bench: Option<BenchSection>,
    pub judgetrainer: Option<JudgetrainerSection>,
    /// ONNX engine resource caps; every field is an optional ceiling — unset
    /// fields derive from the libsyscap budget (capacity when idle), never
    /// from an ONNX Runtime default.
    pub engine: EngineSection,
    /// Progressive-ranking cascade knobs (see [`CascadeSection`]); defaults
    /// apply when the section is absent.
    pub cascade: CascadeSection,
    /// System capability + live-load budget knobs (see [`SyscapSection`]);
    /// defaults apply when the section is absent.
    pub syscap: SyscapSection,
    pub config_path: PathBuf,
}

#[derive(Debug, Default)]
pub struct CliOverrides {
    pub model_dir: Option<PathBuf>,
    pub verbose: Option<bool>,
    pub logfile: Option<PathBuf>,
    pub revision: Option<String>,
}

impl CliOverrides {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct ConfigError {
    pub msg: String,
}

impl ConfigError {
    fn new(msg: impl Into<String>) -> Self {
        ConfigError { msg: msg.into() }
    }

    fn conflict(field: &str, cli: &str, cfg: &str) -> Self {
        ConfigError::new(format!(
            "--{} '{}' conflicts with config value '{}'",
            field, cli, cfg
        ))
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[config] {}", self.msg)
    }
}

impl std::error::Error for ConfigError {}

fn merge_path(
    field: &str,
    cli: &Option<PathBuf>,
    cfg: &Option<PathBuf>,
) -> Result<Option<PathBuf>, ConfigError> {
    match (cli, cfg) {
        (Some(c), Some(y)) => {
            let cs = c.to_string_lossy();
            let ys = y.to_string_lossy();
            if cs != ys {
                return Err(ConfigError::conflict(field, &cs, &ys));
            }
            Ok(Some(c.clone()))
        }
        (Some(c), None) => Ok(Some(c.clone())),
        (None, Some(y)) => Ok(Some(y.clone())),
        (None, None) => Ok(None),
    }
}

fn merge_string(
    field: &str,
    cli: &Option<String>,
    cfg: &Option<String>,
) -> Result<Option<String>, ConfigError> {
    match (cli, cfg) {
        (Some(c), Some(y)) => {
            if c != y {
                return Err(ConfigError::conflict(field, c, y));
            }
            Ok(Some(c.clone()))
        }
        (Some(c), None) => Ok(Some(c.clone())),
        (None, Some(y)) => Ok(Some(y.clone())),
        (None, None) => Ok(None),
    }
}

fn merge_bool(
    field: &str,
    cli: &Option<bool>,
    cfg: &Option<bool>,
) -> Result<Option<bool>, ConfigError> {
    match (cli, cfg) {
        (Some(c), Some(y)) => {
            if c != y {
                return Err(ConfigError::conflict(
                    field,
                    if *c { "true" } else { "false" },
                    if *y { "true" } else { "false" },
                ));
            }
            Ok(Some(*c))
        }
        (Some(c), None) => Ok(Some(*c)),
        (None, Some(y)) => Ok(Some(*y)),
        (None, None) => Ok(None),
    }
}

fn resolve_relative(base: &Path, p: &Path) -> PathBuf {
    if p.is_relative() {
        base.parent().unwrap_or(Path::new(".")).join(p)
    } else {
        p.to_path_buf()
    }
}

/// Expand a leading `~`/`~/` to the user's home directory (`HOME`, falling
/// back to `USERPROFILE` on Windows). Anything else is returned untouched.
fn expand_tilde(p: &Path) -> PathBuf {
    let Some(s) = p.to_str() else {
        return p.to_path_buf();
    };
    let rest = if s == "~" {
        Some("")
    } else {
        s.strip_prefix("~/")
    };
    let Some(rest) = rest else {
        return p.to_path_buf();
    };
    let home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()));
    match home {
        Some(home) => {
            let mut out = PathBuf::from(home);
            if !rest.is_empty() {
                out.push(rest);
            }
            out
        }
        None => p.to_path_buf(),
    }
}

/// Parse a `RawConfig` into a `Config`, resolving relative paths against
/// `config_path` and validating. Used by both `load` (file-based) and
/// `config::default_config` (embedded YAML).
pub fn from_raw(raw: RawConfig, config_path: &Path) -> Result<Config, ConfigError> {
    let model_dir = raw
        .model_dir
        .as_ref()
        .map(|p| resolve_relative(config_path, p))
        .unwrap_or_else(libmodelstorage::default_model_dir);

    let verbose = raw.verbose.unwrap_or(false);
    let logfile = raw
        .logfile
        .as_ref()
        .map(|p| resolve_relative(config_path, p));
    let revision = raw.revision.clone().unwrap_or_else(|| "main".to_string());

    let bench = raw.bench.as_ref().map(|b| {
        let mut b = b.clone();
        b.results_dir = b.results_dir.map(|d| {
            resolve_relative(config_path, &expand_tilde(Path::new(&d)))
                .to_string_lossy()
                .into_owned()
        });
        b.cache_dir = b.cache_dir.map(|d| {
            resolve_relative(config_path, &expand_tilde(Path::new(&d)))
                .to_string_lossy()
                .into_owned()
        });
        b
    });

    let config = Config {
        model_dir,
        verbose,
        logfile,
        revision,
        binaries: resolve_binaries_ref(config_path, raw.binaries.as_ref()),
        models: raw.models.unwrap_or_default(),
        families: raw.families.unwrap_or_default(),
        gui_embedders: raw.gui_embedders.unwrap_or_default(),
        gui_idle_unload_secs: raw
            .gui_idle_unload_secs
            .unwrap_or(DEFAULT_GUI_IDLE_UNLOAD_SECS),
        gui_cv_dir_root: raw
            .gui_cv_dir_root
            .as_ref()
            .map(|p| resolve_relative(config_path, &expand_tilde(p))),
        gui_judge_artifact: raw.gui_judge_artifact.as_ref().map(|arts| {
            let mut arts = arts.clone();
            if let Some(p) = arts.default.as_mut() {
                *p = resolve_relative(config_path, p);
            }
            for p in arts.by_embedder.values_mut() {
                *p = resolve_relative(config_path, p);
            }
            arts
        }),
        rubric_defaults: raw.rubric_defaults.unwrap_or_default(),
        bench,
        judgetrainer: raw.judgetrainer.clone(),
        engine: raw.engine.unwrap_or_default(),
        cascade: raw.cascade.unwrap_or_default(),
        syscap: raw.syscap.unwrap_or_default(),
        config_path: config_path.to_path_buf(),
    };

    validate(&config)?;
    Ok(config)
}

pub fn load(path: &Path, overrides: &CliOverrides) -> Result<Config, ConfigError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| ConfigError::new(format!("failed to read '{}': {}", path.display(), e)));
    let content = content?;
    let raw: RawConfig = serde_yaml::from_str(&content)
        .map_err(|e| ConfigError::new(format!("failed to parse '{}': {}", path.display(), e)))?;

    // Merge CLI overrides with the file config, detecting conflicts when
    // both the CLI and the file set the same value differently.
    let model_dir = merge_path("model-dir", &overrides.model_dir, &raw.model_dir)?
        .map(|p| resolve_relative(path, &p))
        .unwrap_or_else(libmodelstorage::default_model_dir);

    let verbose = merge_bool("verbose", &overrides.verbose, &raw.verbose)?.unwrap_or(false);
    let logfile = merge_path("logfile", &overrides.logfile, &raw.logfile)?
        .map(|p| resolve_relative(path, &p));
    let revision =
        merge_string("revision", &overrides.revision, &raw.revision)?.unwrap_or("main".into());

    let bench = raw.bench.clone().map(|mut b| {
        b.results_dir = b.results_dir.map(|d| {
            resolve_relative(path, &expand_tilde(Path::new(&d)))
                .to_string_lossy()
                .into_owned()
        });
        b.cache_dir = b.cache_dir.map(|d| {
            resolve_relative(path, &expand_tilde(Path::new(&d)))
                .to_string_lossy()
                .into_owned()
        });
        b
    });

    let config = Config {
        model_dir,
        verbose,
        logfile,
        revision,
        binaries: resolve_binaries(path, raw.binaries),
        models: raw.models.unwrap_or_default(),
        families: raw.families.unwrap_or_default(),
        gui_embedders: raw.gui_embedders.unwrap_or_default(),
        gui_idle_unload_secs: raw.gui_idle_unload_secs.unwrap_or(DEFAULT_GUI_IDLE_UNLOAD_SECS),
        gui_cv_dir_root: raw
            .gui_cv_dir_root
            .map(|p| resolve_relative(path, &expand_tilde(&p))),
        gui_judge_artifact: raw.gui_judge_artifact.map(|mut arts| {
            if let Some(p) = arts.default.as_mut() {
                *p = resolve_relative(path, p);
            }
            for p in arts.by_embedder.values_mut() {
                *p = resolve_relative(path, p);
            }
            arts
        }),
        rubric_defaults: raw.rubric_defaults.unwrap_or_default(),
        bench,
        judgetrainer: raw.judgetrainer,
        engine: raw.engine.unwrap_or_default(),
        cascade: raw.cascade.unwrap_or_default(),
        syscap: raw.syscap.unwrap_or_default(),
        config_path: path.to_path_buf(),
    };

    validate(&config)?;
    Ok(config)
}

fn resolve_binaries_ref(config_path: &Path, raw: Option<&BinaryPaths>) -> BinaryPaths {
    resolve_binaries(config_path, raw.cloned())
}

fn resolve_binaries(config_path: &Path, raw: Option<BinaryPaths>) -> BinaryPaths {
    let raw = raw.unwrap_or_default();
    BinaryPaths {
        bench: raw.bench.map(|p| resolve_relative(config_path, &p)),
        benchleaderboard: raw
            .benchleaderboard
            .map(|p| resolve_relative(config_path, &p)),
        judgetrainer: raw.judgetrainer.map(|p| resolve_relative(config_path, &p)),
    }
}

fn validate(cfg: &Config) -> Result<(), ConfigError> {
    if let Some(ref bench) = cfg.bench {
        if let Some(ref embed_models) = bench.embed_models {
            for slug in embed_models {
                if !cfg.models.iter().any(|m| m.slug == *slug) {
                    return Err(ConfigError::new(format!(
                        "bench.embed_models slug '{}' not found in models section",
                        slug
                    )));
                }
            }
        }
    }
    if let Some(ref jt) = cfg.judgetrainer {
        if let Some(ref embedder) = jt.embedder {
            if !cfg.models.iter().any(|m| m.slug == *embedder) {
                return Err(ConfigError::new(format!(
                    "judgetrainer.embedder slug '{}' not found in models section",
                    embedder
                )));
            }
        }
        if let Some(ref rungs) = jt.rungs {
            for r in rungs {
                if !["m0", "m1", "m2", "m3"].contains(&r.as_str()) {
                    return Err(ConfigError::new(format!(
                        "judgetrainer.rungs entry '{}' invalid (expected m0|m1|m2|m3)",
                        r
                    )));
                }
            }
        }
        if let Some(ref c) = jt.chunking {
            if !["none", "ctx_size", "fixed"].contains(&c.as_str()) {
                return Err(ConfigError::new(format!(
                    "judgetrainer.chunking '{}' invalid (expected none|ctx_size|fixed)",
                    c
                )));
            }
        }
        if let Some(ref c) = jt.contextual {
            if !["off", "static"].contains(&c.as_str()) {
                return Err(ConfigError::new(format!(
                    "judgetrainer.contextual '{}' invalid (expected off|static)",
                    c
                )));
            }
        }
        if let Some(ref p) = jt.text_preprocessing {
            if !["none", "lowercase"].contains(&p.as_str()) {
                return Err(ConfigError::new(format!(
                    "judgetrainer.text_preprocessing '{}' invalid (expected none|lowercase)",
                    p
                )));
            }
        }
    }
    for ge in &cfg.gui_embedders {
        if !cfg.models.iter().any(|m| m.slug == ge.slug) {
            return Err(ConfigError::new(format!(
                "gui_embedders slug '{}' not found in models section",
                ge.slug
            )));
        }
    }
    // Validate each model has at least one artifact
    for m in &cfg.models {
        if m.artifacts.is_empty() {
            return Err(ConfigError::new(format!(
                "model '{}' must have at least one artifact",
                m.slug
            )));
        }
        for art in &m.artifacts {
            if art.url.is_empty() {
                return Err(ConfigError::new(format!(
                    "model '{}' artifact with backend '{:?}' quantization '{}' has empty url",
                    m.slug, art.backend, art.quantization
                )));
            }
        }
    }
    Ok(())
}

impl Config {
    /// Load the config from the embedded default YAML (replacing triq.yaml).
    /// `config_path` is used for resolving relative paths and is typically the
    /// path to the config file (or a synthetic path for the embedded fallback).
    pub fn default_from_embedded(config_path: PathBuf) -> Self {
        let raw: RawConfig = serde_yaml::from_str(crate::config::DEFAULT_CONFIG_YAML)
            .expect("embedded DEFAULT_CONFIG_YAML must parse");
        crate::from_raw(raw, &config_path).expect("embedded config must validate")
    }

    /// Merge user preference overrides on top of the embedded config.
    /// Only fields set in `overrides` replace the defaults; the rest stay.
    pub fn with_user_overrides(mut self, overrides: &crate::config::UserOverrides) -> Self {
        if let Some(ref b) = overrides.engine_backend {
            if let Ok(backend) = b.parse::<Backend>() {
                self.engine.backend = Some(backend);
            }
        }
        if let Some(ref p) = overrides.engine_exec_provider {
            if let Ok(ep) = p.parse::<ExecutionProvider>() {
                self.engine.exec_provider = Some(ep);
            }
        }
        if let Some(v) = overrides.engine_embed_batch {
            self.engine.embed_batch = Some(v);
        }
        if let Some(v) = overrides.engine_intra_threads {
            self.engine.intra_threads = Some(v);
        }
        if let Some(v) = overrides.engine_inter_threads {
            self.engine.inter_threads = Some(v);
        }
        if let Some(v) = overrides.engine_max_cores {
            self.engine.max_cores = Some(v);
        }
        if let Some(v) = overrides.engine_no_cache {
            self.engine.no_cache = Some(v);
        }
        self
    }

    pub fn bench_bin(&self) -> Option<&Path> {
        self.binaries.bench.as_deref()
    }
    pub fn leaderboard_bin(&self) -> Option<&Path> {
        self.binaries.benchleaderboard.as_deref()
    }
    pub fn judgetrainer_bin(&self) -> Option<&Path> {
        self.binaries.judgetrainer.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use std::sync::atomic::{AtomicUsize, Ordering};
    static CNT: AtomicUsize = AtomicUsize::new(0);

    fn unique() -> usize {
        CNT.fetch_add(1, Ordering::Relaxed)
    }

    fn write_tmp(yaml: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "libconfigreader_test_{}_{}.yaml",
            std::process::id(),
            unique()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        write!(f, "{}", yaml).unwrap();
        p
    }

    #[test]
    fn minimal_config() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert!(config.models.is_empty());
        assert!(config.families.is_empty());
        assert!(!config.verbose);
        assert_eq!(config.revision, "main");
        assert_eq!(config.gui_idle_unload_secs, DEFAULT_GUI_IDLE_UNLOAD_SECS);
        assert_eq!(config.rubric_defaults, RubricDefaults::default());
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn rubric_defaults_override_and_partial_section() {
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\nrubric_defaults:\n  ad_points: 60\n  fb_profile_points: 8\n",
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        let rd = config.rubric_defaults;
        assert_eq!(rd.ad_points, 60.0);
        assert_eq!(rd.fb_profile_points, 8.0);
        // Fields absent from a present section keep their defaults.
        assert_eq!(rd.fb_job_points, DEFAULT_RUBRIC_FB_JOB_POINTS);
        assert_eq!(rd.criterion_points, DEFAULT_RUBRIC_CRITERION_POINTS);
        std::fs::remove_file(&cfg_path).ok();

        let cfg_path = write_tmp(
            "models: []\nfamilies: []\nrubric_defaults:\n  ad_points: 1\n  fb_job_points: 2\n  fb_profile_points: 3\n  criterion_points: 4\n",
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(
            config.rubric_defaults,
            RubricDefaults {
                ad_points: 1.0,
                fb_job_points: 2.0,
                fb_profile_points: 3.0,
                criterion_points: 4.0,
            }
        );
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn gui_judge_artifact_optional_and_relative_resolved() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        assert!(config.gui_judge_artifact.is_none());

        // string form: one default artifact, no per-embedder overrides
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\ngui_judge_artifact: \"/tmp/kilo/judge-m2.json\"\n",
        );
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        let arts = config.gui_judge_artifact.expect("string form parses");
        assert_eq!(arts.default.as_deref(), Some(std::path::Path::new("/tmp/kilo/judge-m2.json")));
        assert!(arts.by_embedder.is_empty());
        assert_eq!(arts.for_embedder("any-embedder"), Some(std::path::Path::new("/tmp/kilo/judge-m2.json")));
    }

    #[test]
    fn gui_judge_artifact_map_form_resolves_per_embedder() {
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\n\
             gui_judge_artifact:\n  \
             default: \"600m/judge-default.json\"\n  \
             harrier-oss-v1-270m: \"270m/judge-270m.json\"\n",
        );
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        let arts = config.gui_judge_artifact.expect("map form parses");
        // relative paths resolve against the config file's directory
        let base = config.config_path.parent().unwrap();
        assert_eq!(
            arts.for_embedder("harrier-oss-v1-270m"),
            Some(base.join("270m/judge-270m.json").as_path())
        );
        assert_eq!(
            arts.for_embedder("harrier-oss-v1-0.6b"),
            Some(base.join("600m/judge-default.json").as_path())
        );
        assert!(!arts.is_empty());

        // empty map is rejected
        let cfg_path = write_tmp("models: []\nfamilies: []\ngui_judge_artifact: {}\n");
        assert!(load(&cfg_path, &CliOverrides::default()).is_err());
    }

    #[test]
    fn gui_cv_dir_root_optional_tilde_and_relative_resolved() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        assert!(config.gui_cv_dir_root.is_none());

        // a leading ~ expands to the home directory
        let cfg_path = write_tmp("models: []\nfamilies: []\ngui_cv_dir_root: \"~/jobs-root\"\n");
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
            .expect("home dir for tilde expansion");
        assert_eq!(
            config.gui_cv_dir_root,
            Some(PathBuf::from(home).join("jobs-root"))
        );

        // relative paths resolve against the config file's directory
        let cfg_path = write_tmp("models: []\nfamilies: []\ngui_cv_dir_root: jobs-root\n");
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        assert_eq!(
            config.gui_cv_dir_root,
            Some(config.config_path.parent().unwrap().join("jobs-root"))
        );
    }

    #[test]
    fn bench_dirs_relative_resolved_absolute_kept() {
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\n\
             bench:\n  \
             results_dir: \"bench/results\"\n  \
             cache_dir: \"~/.cache/ik/embeddings\"\n",
        );
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        let bench = config.bench.as_ref().expect("bench section parses");
        let base = config.config_path.parent().unwrap();

        // relative resolves against the config file's directory, so the
        // result corpus is found no matter what cwd the tool runs from
        assert_eq!(
            bench.results_dir.as_deref(),
            Some(base.join("bench/results").to_string_lossy().as_ref())
        );
        // a leading ~ expands to the home directory
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
            .expect("home dir for tilde expansion");
        assert_eq!(
            bench.cache_dir.as_deref(),
            Some(
                PathBuf::from(home)
                    .join(".cache/ik/embeddings")
                    .to_string_lossy()
                    .as_ref()
            )
        );

        let cfg_path = write_tmp(
            "models: []\nfamilies: []\n\
             bench:\n  \
             results_dir: \"/abs/results\"\n  \
             cache_dir: \"/abs/embeddings\"\n",
        );
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        let bench = config.bench.as_ref().expect("bench section parses");
        assert_eq!(bench.results_dir.as_deref(), Some("/abs/results"));
        assert_eq!(bench.cache_dir.as_deref(), Some("/abs/embeddings"));
    }

    #[test]
    fn bench_dirs_absent_when_no_bench_section() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::default()).unwrap();
        assert!(config.bench.is_none());
    }

    #[test]
    fn gui_idle_unload_secs_override_and_disable() {
        let cfg_path = write_tmp("models: []\nfamilies: []\ngui_idle_unload_secs: 600\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.gui_idle_unload_secs, 600);
        std::fs::remove_file(&cfg_path).ok();

        let cfg_path = write_tmp("models: []\nfamilies: []\ngui_idle_unload_secs: 0\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.gui_idle_unload_secs, 0);
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn engine_section_defaults_when_absent_and_parses_when_present() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert!(config.engine.intra_threads.is_none());
        assert!(config.engine.inter_threads.is_none());
        assert!(config.engine.embed_batch.is_none());
        assert!(config.engine.max_cores.is_none());
        std::fs::remove_file(&cfg_path).ok();

        let cfg_path = write_tmp(
            "models: []\nfamilies: []\nengine:\n  intra_threads: 2\n  inter_threads: 1\n  embed_batch: 4\n  max_cores: 1\n",
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.engine.intra_threads, Some(2));
        assert_eq!(config.engine.inter_threads, Some(1));
        assert_eq!(config.engine.embed_batch, Some(4));
        assert_eq!(config.engine.max_cores, Some(1));
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn syscap_section_defaults_when_absent_and_parses_when_present() {
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert!(config.syscap.enabled());
        assert_eq!(config.syscap.reserve_cores(), DEFAULT_SYSCAP_RESERVE_CORES);
        assert_eq!(
            config.syscap.sample_min_interval_ms(),
            DEFAULT_SYSCAP_SAMPLE_MIN_INTERVAL_MS
        );
        std::fs::remove_file(&cfg_path).ok();

        let cfg_path = write_tmp(
            "models: []\nfamilies: []\nsyscap:\n  enabled: false\n  reserve_cores: 2\n  load_backoff: 0.5\n  load_stall: 3.0\n",
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert!(!config.syscap.enabled());
        assert_eq!(config.syscap.reserve_cores(), 2);
        assert_eq!(config.syscap.load_backoff(), 0.5);
        assert_eq!(config.syscap.load_stall(), 3.0);
        // Unset fields still fall back to their defaults.
        assert_eq!(
            config.syscap.min_free_mem_frac(),
            DEFAULT_SYSCAP_MIN_FREE_MEM_FRAC
        );
        std::fs::remove_file(&cfg_path).ok();

        // engine.embed_sessions: absent -> Auto; a number pins the pool.
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.engine.embed_sessions, None);
        std::fs::remove_file(&cfg_path).ok();
        let cfg_path = write_tmp("models: []\nfamilies: []\nengine:\n  embed_sessions: 4\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.engine.embed_sessions, Some(EmbedSessions::Fixed(4)));
        std::fs::remove_file(&cfg_path).ok();
        let cfg_path =
            write_tmp("models: []\nfamilies: []\nengine:\n  embed_sessions: auto\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(
            config.engine.embed_sessions,
            Some(EmbedSessions::Auto),
            "explicit `auto` parses as Auto"
        );
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn syscap_build_jobs_parses_auto_fixed_and_rejects_garbage() {
        // Absent section -> default Auto.
        let cfg_path = write_tmp("models: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.syscap.build_jobs(), DEFAULT_SYSCAP_BUILD_JOBS);
        std::fs::remove_file(&cfg_path).ok();

        // Explicit auto (any case) and a fixed count.
        for (yaml, want) in [
            ("auto", BuildJobs::Auto),
            ("AUTO", BuildJobs::Auto),
            ("4", BuildJobs::Fixed(4)),
            ("1", BuildJobs::Fixed(1)),
        ] {
            let cfg_path = write_tmp(&format!(
                "models: []\nfamilies: []\nsyscap:\n  build_jobs: {yaml}\n"
            ));
            let config = load(&cfg_path, &CliOverrides::new()).unwrap();
            assert_eq!(config.syscap.build_jobs(), want, "yaml: {yaml}");
            std::fs::remove_file(&cfg_path).ok();
        }

        // Unrecognized values are a config error, not a silent fallback.
        let cfg_path = write_tmp("models: []\nfamilies: []\nsyscap:\n  build_jobs: lots\n");
        assert!(load(&cfg_path, &CliOverrides::new()).is_err());
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn override_model_dir() {
        let cfg_path = write_tmp("model_dir: /cache/models\nmodels: []\nfamilies: []\n");
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.model_dir, PathBuf::from("/cache/models"));
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn conflict_detected() {
        let cfg_path = write_tmp("model_dir: /cache/a\nmodels: []\nfamilies: []\n");
        let mut ovr = CliOverrides::new();
        ovr.model_dir = Some(PathBuf::from("/cache/b"));
        let err = load(&cfg_path, &ovr).unwrap_err().to_string();
        assert!(err.contains("conflicts"));
        assert!(err.contains("model-dir"));
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn bench_embed_model_not_in_models_is_error() {
        let cfg_path = write_tmp(
            "models:\n  - slug: m1\n    family: f1\n    kind: embedder\n    dim: 768\n    max_context_window: 8192\n    artifacts:\n      - backend: ort\n        quantization: \"none\"\n        source: hf:x/y\n        url: http://x\n        sha256: \"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"\nfamilies: []\nbench:\n  embed_models:\n    - m1\n    - missing\n"
        );
        let err = load(&cfg_path, &CliOverrides::new()).unwrap_err().to_string();
        assert!(err.contains("missing"));
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn parse_full_config() {
        let cfg_path = write_tmp(
            "model_dir: /my/cache\nverbose: true\nrevision: v2\nlogfile: /tmp/test.log\nbinaries:\n  modelcaller: /bin/mc\n  bench: /bin/bench1\nmodels:\n  - slug: m1\n    family: f1\n    kind: embedder\n    dim: 768\n    max_context_window: 8192\n    artifacts:\n      - backend: ort\n        quantization: \"none\"\n        source: hf:x/y\n        url: http://x\n        sha256: \"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"\nfamilies:\n  - family: f1\n    usage: plain\nbench:\n  dataset: /d.csv\n  task: t\n  baselines:\n    - bm25\n  embed_models:\n    - m1\n"
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        assert_eq!(config.model_dir, PathBuf::from("/my/cache"));
        assert_eq!(config.verbose, true);
        assert_eq!(config.revision, "v2");
        assert_eq!(config.logfile, Some(PathBuf::from("/tmp/test.log")));
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.families.len(), 1);
        assert_eq!(config.bench_bin().unwrap().to_str(), Some("/bin/bench1"));
        assert!(config.bench.is_some());
        assert_eq!(
            config
                .bench
                .as_ref()
                .unwrap()
                .baselines
                .as_deref()
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn judgetrainer_section_validated() {
        let cfg_path = write_tmp(
            "models:\n  - slug: m1\n    family: f1\n    kind: embedder\n    dim: 768\n    max_context_window: 8192\n    artifacts:\n      - backend: ort\n        quantization: \"none\"\n        source: hf:x/y\n        url: http://x\n        sha256: \"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"\nfamilies: []\nbinaries:\n  judgetrainer: /bin/jt\njudgetrainer:\n  train_csv: /d/train.csv\n  out_dir: /cache/jt\n  rungs: [m0, m1]\n  embedder: m1\n  epochs: 10\n"
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        let jt = config.judgetrainer.clone().expect("section present");
        assert_eq!(jt.train_csv.as_deref(), Some("/d/train.csv"));
        assert_eq!(jt.epochs, Some(10));
        assert_eq!(jt.rungs.as_deref().unwrap(), &["m0".to_string(), "m1".to_string()]);
        assert_eq!(
            config.judgetrainer_bin().unwrap().to_str(),
            Some("/bin/jt")
        );
        std::fs::remove_file(&cfg_path).ok();

        // unknown rung is rejected
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\njudgetrainer:\n  rungs: [m9]\n"
        );
        let err = load(&cfg_path, &CliOverrides::new()).unwrap_err().to_string();
        assert!(err.contains("m9"));
        std::fs::remove_file(&cfg_path).ok();

        // invalid chunking is rejected
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\njudgetrainer:\n  chunking: sideways\n"
        );
        let err = load(&cfg_path, &CliOverrides::new()).unwrap_err().to_string();
        assert!(err.contains("sideways"));
        std::fs::remove_file(&cfg_path).ok();

        // invalid contextual is rejected
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\njudgetrainer:\n  contextual: always\n"
        );
        let err = load(&cfg_path, &CliOverrides::new()).unwrap_err().to_string();
        assert!(err.contains("always"));
        std::fs::remove_file(&cfg_path).ok();

        // full settings parse with types intact (embedder omitted: slug
        // validation is covered by the test above)
        let cfg_path = write_tmp(
            "models: []\nfamilies: []\njudgetrainer:\n  train_csv: /d/train.csv\n  dataset_dir: /d/split\n  out_dir: /cache/jt\n  rungs: [m0]\n  max_jobs: 30\n  val_frac: 0.2\n  embed_batch: 4\n  chunking: fixed\n  chunk_size: 512\n  contextual: static\n  text_preprocessing: lowercase\n  epochs: 10\n  patience: 5\n  lr: 0.001\n  wd: 0.0001\n  batch: 64\n  seed: 42\n  hidden: 256\n  tau: 0.5\n  dim: 1024\n"
        );
        let config = load(&cfg_path, &CliOverrides::new()).unwrap();
        let jt = config.judgetrainer.clone().expect("section present");
        assert_eq!(jt.chunking.as_deref(), Some("fixed"));
        assert_eq!(jt.chunk_size, Some(512));
        assert_eq!(jt.lr, Some(0.001));
        assert_eq!(jt.seed, Some(42));
        assert_eq!(jt.hidden, Some(256));
        assert_eq!(jt.tau, Some(0.5));
        assert_eq!(jt.dim, Some(1024));
        std::fs::remove_file(&cfg_path).ok();
    }

    #[test]
    fn include_clones_base_combos_instead_of_overwriting() {
        // Regression: include entries with the same matching key must each
        // produce a *new* combo, not overwrite the previous one.
        //
        // matrix: {chunking: [fixed, ctx_size], retrieval: [max_sim]}
        // include: [{chunking: fixed, chunk_size: 256},
        //          {chunking: fixed, chunk_size: 512}]
        //
        // Expected: base combos kept, PLUS two new fixed combos with
        // respective chunk_sizes.
        let mut matrix = HashMap::new();
        matrix.insert("chunking".into(), vec!["fixed".into(), "ctx_size".into()]);
        matrix.insert("retrieval".into(), vec!["max_sim".into()]);

        let include = vec![
            HashMap::from([("chunking".into(), "fixed".into()), ("chunk_size".into(), "256".into())]),
            HashMap::from([("chunking".into(), "fixed".into()), ("chunk_size".into(), "512".into())]),
        ];

        let combos = expand_matrix(&matrix, &include, &[]);

        // Base: {fixed, max_sim}, {ctx_size, max_sim}  (2)
        // Include 1 matches {fixed, max_sim} → new {fixed, max_sim, 256}
        // Include 2 matches {fixed, max_sim} → new {fixed, max_sim, 512}
        // Total: 4
        assert_eq!(combos.len(), 4);

        let chunk_sizes: Vec<Option<&str>> = combos
            .iter()
            .map(|c| c.get("chunk_size").map(|s| s.as_str()))
            .collect();
        assert_eq!(chunk_sizes, vec![None, None, Some("256"), Some("512")]);
    }

    #[test]
    fn expand_matrix_follows_github_actions_semantics() {
        // Mirrors the GitHub Actions `strategy.matrix` example:
        // https://docs.github.com/actions/using-jobs/using-a-matrix-for-your-jobs
        //
        // With clone-per-include, each include that matches a base combo
        // creates a new job. The original base combos are preserved.
        let mut matrix = HashMap::new();
        matrix.insert(
            "fruit".to_string(),
            vec!["apple".to_string(), "pear".to_string()],
        );
        matrix.insert(
            "animal".to_string(),
            vec!["cat".to_string(), "dog".to_string()],
        );

        let mut include = Vec::new();
        include.push(HashMap::from([("color".to_string(), "green".to_string())]));
        include.push(HashMap::from([
            ("color".to_string(), "pink".to_string()),
            ("animal".to_string(), "cat".to_string()),
        ]));
        include.push(HashMap::from([
            ("fruit".to_string(), "apple".to_string()),
            ("shape".to_string(), "circle".to_string()),
        ]));
        include.push(HashMap::from([("fruit".to_string(), "banana".to_string())]));
        include.push(HashMap::from([
            ("fruit".to_string(), "banana".to_string()),
            ("animal".to_string(), "cat".to_string()),
        ]));

        let combos = expand_matrix(&matrix, &include, &[]);

        let mut norm: Vec<Vec<(String, String)>> = combos
            .into_iter()
            .map(|m| {
                let mut v: Vec<(String, String)> = m.into_iter().collect();
                v.sort();
                v
            })
            .collect();
        norm.sort();

        // Base combos (4) are always preserved.
        // Include {color:green} matches all 4 → 4 new.
        // Include {color:pink, animal:cat} matches {apple,cat} and {pear,cat}
        //   from base (the green+cat ones have color=green ≠ pink, so no match)
        //   → 2 new.
        // Include {fruit:apple, shape:circle} matches {apple,cat}, {apple,dog}
        //   from base, plus {apple,cat,green}, {apple,dog,green} from step 1,
        //   plus {apple,cat,pink} from step 2 → 5 new.
        // Include {fruit:banana} → no match → standalone.
        // Include {fruit:banana, animal:cat} → no match → standalone.
        // Total: 4 + 4 + 2 + 5 + 1 + 1 = 17
        assert_eq!(norm.len(), 17);

        // Spot-check: original base combos are preserved
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("fruit".into(), "apple".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "dog".into()),
            ("fruit".into(), "apple".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("fruit".into(), "pear".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "dog".into()),
            ("fruit".into(), "pear".into()),
        ]));

        // Spot-check: green combos added for all base
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("color".into(), "green".into()),
            ("fruit".into(), "apple".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "dog".into()),
            ("color".into(), "green".into()),
            ("fruit".into(), "pear".into()),
        ]));

        // Spot-check: pink only on cat combos
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("color".into(), "pink".into()),
            ("fruit".into(), "apple".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("color".into(), "pink".into()),
            ("fruit".into(), "pear".into()),
        ]));

        // Spot-check: standalone banana combos
        assert!(norm.contains(&vec![
            ("fruit".into(), "banana".into()),
        ]));
        assert!(norm.contains(&vec![
            ("animal".into(), "cat".into()),
            ("fruit".into(), "banana".into()),
        ]));
    }

    const TWO_MODELS: &str = r#"
model_dir: /cache/models
models:
  - slug: harrier-oss-v1-0.6b
    family: harrier-oss-v1
    kind: embedder
    dim: 1024
    max_context_window: 32768
    artifacts:
      - backend: ort
        quantization: "q4"
        source: hf:onnx-community/harrier-oss-v1-0.6b-ONNX
        url: https://huggingface.co/x/resolve/main/onnx/model_q4.onnx
        sha256: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
      - backend: mlx
        quantization: "4bit"
        source: hf:majentik/harrier-oss-v1-0.6b-MLX-4bit
        url: https://huggingface.co/majentik/resolve/main/model.safetensors
        sha256: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
  - slug: plain-model
    family: other
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:someone/plain
        url: https://huggingface.co/x/resolve/main/onnx/model.onnx
        sha256: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
families: []
"#;

#[test]
    fn mlx_variant_parses_and_is_optional() {
        let cfg_path = write_tmp(TWO_MODELS);
        let cfg = load(&cfg_path, &CliOverrides::new()).expect("loads");
        let with = cfg
            .models
            .iter()
            .find(|m| m.slug == "harrier-oss-v1-0.6b")
            .expect("model");
        let mlx_art = with.artifacts.iter().find(|a| a.backend == Backend::Mlx).expect("mlx artifact");
        assert_eq!(mlx_art.quantization, "4bit");
        assert_eq!(mlx_art.source, "hf:majentik/harrier-oss-v1-0.6b-MLX-4bit");
        let onnx_art = with.artifacts.iter().find(|a| a.backend == Backend::Ort).expect("onnx artifact");
        assert_eq!(onnx_art.quantization, "q4");
        // Model without MLX artifact
        let without = cfg
            .models
            .iter()
            .find(|m| m.slug == "plain-model")
            .expect("model");
        assert!(without.artifacts.iter().all(|a| a.backend == Backend::Ort));
    }

    #[test]
    fn backend_defaults_to_auto_and_reads_explicit_names() {
        let cfg_path = write_tmp(TWO_MODELS);
        let cfg = load(&cfg_path, &CliOverrides::new()).expect("loads");
        assert_eq!(cfg.engine.backend, None, "unset means auto");
        assert!(Backend::default().allows_mlx());
        assert!(!Backend::default().requires_mlx());
        // `ort` is what bench pins, so it must forbid the arm.
        assert!(!Backend::Ort.allows_mlx());
        // `mlx` demands it: a missing arm is a load error, not a quiet fall
        // back to the CPU — the same rule an unregistered EP follows.
        assert!(Backend::Mlx.requires_mlx());
        assert_eq!(Backend::Mlx.as_str(), "mlx");
        std::fs::remove_file(&cfg_path).ok();

        for (yaml, want) in [
            ("ort", Backend::Ort),
            ("mlx", Backend::Mlx),
            ("auto", Backend::Auto),
        ] {
            let body = format!("engine:\n  backend: {yaml}\n{TWO_MODELS}");
            let p = write_tmp(&body);
            let cfg = load(&p, &CliOverrides::new()).expect("loads");
            assert_eq!(cfg.engine.backend, Some(want), "for {yaml}");
            std::fs::remove_file(&p).ok();
        }
    }

    #[test]
    fn an_unknown_backend_name_is_rejected_rather_than_defaulted() {
        // `backend: metal` (a plausible typo for `mlx`) must not silently
        // become `auto` and quietly change which runtime serves embeds.
        let body = format!("engine:\n  backend: metal\n{TWO_MODELS}");
        let p = write_tmp(&body);
        let err = load(&p, &CliOverrides::new()).unwrap_err().to_string();
        assert!(err.contains("backend"), "error names the field: {err}");
        std::fs::remove_file(&p).ok();
    }
}






