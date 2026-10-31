//! Static configuration for the IK application.
//!
//! Previously stored as a YAML file (`config.yaml`); now expressed as typed
//! Rust data. There is no embedded YAML / JSON / TOML string anywhere in this
//! module — every value lives in a struct or enum at compile time.
//!
//! Call [`Config::load`] to obtain the singleton configuration.
//!
//! # Conventions
//!
//! * All `model_dir` / path fields preserve the literal `~` character.
//!   Expand it at runtime with e.g. `dirs::home_dir()` or `shellexpand`.
//! * `Option<T>` fields map directly to YAML `null` / unset keys.
//! * Closed string sets (`usage`, `kind`, `backend`, `quantization`,
//!   `build_jobs`) are modeled as enums — exhaustive matches are enforced
//!   by the compiler.

use std::path::{Path, PathBuf};

// ============================================================================
// Top-level
// ============================================================================

/// Top-level application configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Directory where downloaded model artifacts are cached.
    /// `~` is literal — expand at runtime.
    pub model_dir: PathBuf,

    pub verbose: bool,

    /// HF revision tag used when pulling remote artifacts.
    pub revision: String,

    /// Seconds the GUI stays idle before unloading model weights.
    pub gui_idle_unload_secs: u64,

    /// Optional override for the CV-directory root used by new jobs.
    /// `None` means "fall back to `<platform Documents>/jobs`".
    pub gui_cv_dir_root: Option<PathBuf>,

    pub engine: EngineConfig,
    pub syscap: SyscapConfig,
    pub rubric_defaults: RubricDefaults,

    pub families: Vec<Family>,
    pub models: Vec<Model>,
    pub gui_embedders: Vec<GuiEmbedder>,
    pub gui_judge_artifact: GuiJudgeArtifact,
    pub cascade: CascadeConfig,
}

impl Config {
    /// Returns the application configuration, fully populated from static data.
    ///
    /// Cheap to call but not free (allocates a few `Vec`s). Cache the result
    /// in a `OnceLock<Config>` if you want a true singleton.
    pub fn load() -> Self {
        Self {
            model_dir: PathBuf::from("~/.cache/ik/models"),
            verbose: false,
            revision: "main".to_string(),
            gui_idle_unload_secs: 300,
            gui_cv_dir_root: None,

            engine: EngineConfig {
                intra_threads: None,
                inter_threads: None,
                embed_batch: None,
                embed_sessions: None,
                exec_provider: None,
                provider_logging: None,
                backend: Backend::Auto,
                max_cores: None,
                no_cache: None,
            },

            syscap: SyscapConfig {
                enabled: true,
                on_battery_polite: true,
                build_jobs: BuildJobs::Auto,
                reserve_cores: 1,
                reserve_core_frac: 0.10,
                min_free_mem_frac: 0.20,
                hold_free_mem_frac: 0.05,
                load_backoff: 1.0,
                load_stall: 2.0,
                sample_min_interval_ms: 250,
            },

            rubric_defaults: RubricDefaults {
                ad_points: 50.0,
                fb_job_points: 10.0,
                fb_profile_points: 5.0,
                criterion_points: 10.0,
            },

            families: vec![
                Family { family: "static-similarity-mrl".into(), usage: Usage::Plain },
                Family { family: "nomic-embed-text-v1".into(), usage: Usage::NomicPrefix },
                Family { family: "bge-small-en-v1.5".into(), usage: Usage::BgePrefix },
                Family { family: "multilingual-e5-large".into(), usage: Usage::E5Prefix },
                Family { family: "harrier-oss-v1".into(), usage: Usage::HarrierInstruct },
                Family { family: "bge-reranker-v2-m3".into(), usage: Usage::PlainPair },
                Family { family: "jina-reranker-v1-tiny-en".into(), usage: Usage::PlainPair },
                Family { family: "granite-embedding-311m-multilingual-r2".into(), usage: Usage::Plain },
                Family { family: "mlateon".into(), usage: Usage::ColbertPrefix },
                Family { family: "qwen-instruct".into(), usage: Usage::ExtractionJson },
            ],

            models: vec![
                // -------- Harrier 600M ONNX --------
                Model {
                    slug: "harrier-oss-v1-0.6b-ONNX-q4".into(),
                    family: "harrier-oss-v1".into(),
                    kind: ModelKind::Embedder,
                    dim: 1024,
                    max_context_window: 32768,
                    source: "hf:onnx-community/harrier-oss-v1-0.6b-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Q4,
                        source: "hf:onnx-community/harrier-oss-v1-0.6b-ONNX".into(),
                        url: "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_q4.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- Harrier 600M MLX --------
                Model {
                    slug: "harrier-oss-v1-0.6b-MLX-4bit".into(),
                    family: "harrier-oss-v1".into(),
                    kind: ModelKind::Embedder,
                    dim: 1024,
                    max_context_window: 32768,
                    source: "hf:majentik/harrier-oss-v1-0.6b-MLX-4bit".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Mlx,
                        quantization: Quantization::FourBit,
                        source: "hf:majentik/harrier-oss-v1-0.6b-MLX-4bit".into(),
                        url: "https://huggingface.co/majentik/harrier-oss-v1-0.6b-MLX-4bit".into(),
                        sha256: None,
                        extra_files: vec!["tokenizer.json".into(), "config.json".into()],
                    }],
                    multi_vector: false,
                },
                // -------- Harrier 270M ONNX (DEFAULT) --------
                Model {
                    slug: "harrier-oss-v1-270m-ONNX-q4f16".into(),
                    family: "harrier-oss-v1".into(),
                    kind: ModelKind::Embedder,
                    dim: 640,
                    max_context_window: 32768,
                    source: "hf:onnx-community/harrier-oss-v1-270m-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Q4F16,
                        source: "hf:onnx-community/harrier-oss-v1-270m-ONNX".into(),
                        url: "https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- Harrier 270M MLX --------
                Model {
                    slug: "harrier-oss-v1-270m-MLX-4bit".into(),
                    family: "harrier-oss-v1".into(),
                    kind: ModelKind::Embedder,
                    dim: 640,
                    max_context_window: 32768,
                    source: "hf:majentik/harrier-oss-v1-270m-MLX-4bit".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Mlx,
                        quantization: Quantization::FourBit,
                        source: "hf:majentik/harrier-oss-v1-270m-MLX-4bit".into(),
                        url: "https://huggingface.co/majentik/harrier-oss-v1-270m-MLX-4bit".into(),
                        sha256: None,
                        extra_files: vec!["tokenizer.json".into(), "config.json".into()],
                    }],
                    multi_vector: false,
                },
                // -------- Qwen3-4B Instruct (LLM evidence extractor) --------
                Model {
                    slug: "qwen3-4b-instruct-2507-onnx-q4".into(),
                    family: "qwen-instruct".into(),
                    kind: ModelKind::Extractor,
                    dim: 0,
                    max_context_window: 4096,
                    source: "hf:onnx-community/Qwen3-4B-Instruct-2507-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Q4,
                        source: "hf:onnx-community/Qwen3-4B-Instruct-2507-ONNX".into(),
                        url: "https://huggingface.co/onnx-community/Qwen3-4B-Instruct-2507-ONNX/resolve/main/onnx/model_q4.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- bge-reranker-v2-m3 --------
                Model {
                    slug: "bge-reranker-v2-m3".into(),
                    family: "bge-reranker-v2-m3".into(),
                    kind: ModelKind::Reranker,
                    dim: 0,
                    max_context_window: 8192,
                    source: "hf:onnx-community/bge-reranker-v2-m3-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Quantized,
                        source: "hf:onnx-community/bge-reranker-v2-m3-ONNX".into(),
                        url: "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/onnx/model_quantized.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- jina-reranker-v1-tiny-en --------
                Model {
                    slug: "jina-reranker-v1-tiny-en".into(),
                    family: "jina-reranker-v1-tiny-en".into(),
                    kind: ModelKind::Reranker,
                    dim: 0,
                    max_context_window: 512,
                    source: "hf:jinaai/jina-reranker-v1-tiny-en".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:jinaai/jina-reranker-v1-tiny-en".into(),
                        url: "https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- granite-embedding-311m-multilingual-r2 --------
                Model {
                    slug: "granite-embedding-311m-multilingual-r2".into(),
                    family: "granite-embedding-311m-multilingual-r2".into(),
                    kind: ModelKind::Embedder,
                    dim: 768,
                    max_context_window: 32768,
                    source: "hf:ibm-granite/granite-embedding-311m-multilingual-r2".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:ibm-granite/granite-embedding-311m-multilingual-r2".into(),
                        url: "https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- mLateOn (multi-vector) --------
                Model {
                    slug: "mlateon".into(),
                    family: "mlateon".into(),
                    kind: ModelKind::Embedder,
                    dim: 128,
                    max_context_window: 8192,
                    source: "hf:lightonai/mLateOn".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:lightonai/mLateOn".into(),
                        url: "https://huggingface.co/lightonai/mLateOn/resolve/main/model.onnx".into(),
                        sha256: None,
                        extra_files: vec!["tokenizer.json".into(), "config.json".into()],
                    }],
                    multi_vector: true,
                },
                // -------- multilingual-e5-small --------
                Model {
                    slug: "multilingual-e5-small".into(),
                    family: "multilingual-e5-small".into(),
                    kind: ModelKind::Embedder,
                    dim: 384,
                    max_context_window: 512,
                    source: "hf:intfloat/multilingual-e5-small".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:intfloat/multilingual-e5-small".into(),
                        url: "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- paraphrase-multilingual-MiniLM-L12-v2 --------
                Model {
                    slug: "paraphrase-multilingual-MiniLM-L12-v2".into(),
                    family: "paraphrase-multilingual-MiniLM-L12-v2".into(),
                    kind: ModelKind::Embedder,
                    dim: 384,
                    max_context_window: 128,
                    source: "hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".into(),
                        url: "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- snowflake-arctic-embed-xs --------
                Model {
                    slug: "snowflake-arctic-embed-xs".into(),
                    family: "snowflake-arctic-embed-xs".into(),
                    kind: ModelKind::Embedder,
                    dim: 384,
                    max_context_window: 512,
                    source: "hf:Snowflake/snowflake-arctic-embed-xs".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Int8,
                        source: "hf:Snowflake/snowflake-arctic-embed-xs".into(),
                        url: "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model_int8.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- snowflake-arctic-embed-s --------
                Model {
                    slug: "snowflake-arctic-embed-s".into(),
                    family: "snowflake-arctic-embed-s".into(),
                    kind: ModelKind::Embedder,
                    dim: 384,
                    max_context_window: 512,
                    source: "hf:Snowflake/snowflake-arctic-embed-s".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:Snowflake/snowflake-arctic-embed-s".into(),
                        url: "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- snowflake-arctic-embed-m-v2.0 --------
                Model {
                    slug: "snowflake-arctic-embed-m-v2.0".into(),
                    family: "snowflake-arctic-embed-m-v2.0".into(),
                    kind: ModelKind::Embedder,
                    dim: 768,
                    max_context_window: 8192,
                    source: "hf:Snowflake/snowflake-arctic-embed-m-v2.0".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:Snowflake/snowflake-arctic-embed-m-v2.0".into(),
                        url: "https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- amaretto-embed-148m-ONNX --------
                Model {
                    slug: "amaretto-embed-148m-ONNX".into(),
                    family: "amaretto-embed-148m".into(),
                    kind: ModelKind::Embedder,
                    dim: 384,
                    max_context_window: 512,
                    source: "hf:AmarettoLabs/amaretto-embed-148m-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:AmarettoLabs/amaretto-embed-148m-ONNX".into(),
                        url: "https://huggingface.co/AmarettoLabs/amaretto-embed-148m-ONNX/resolve/main/model.onnx".into(),
                        sha256: None,
                        extra_files: vec!["tokenizer.json".into(), "config.json".into()],
                    }],
                    multi_vector: false,
                },
                // -------- multilingual-e5-base --------
                Model {
                    slug: "multilingual-e5-base".into(),
                    family: "multilingual-e5-base".into(),
                    kind: ModelKind::Embedder,
                    dim: 768,
                    max_context_window: 512,
                    source: "hf:intfloat/multilingual-e5-base".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:intfloat/multilingual-e5-base".into(),
                        url: "https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/onnx/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- embeddinggemma-300m-ONNX --------
                Model {
                    slug: "embeddinggemma-300m-ONNX".into(),
                    family: "embeddinggemma-300m".into(),
                    kind: ModelKind::Embedder,
                    dim: 768,
                    max_context_window: 2048,
                    source: "hf:thoratsr7/embeddinggemma-300m-ONNX".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::None,
                        source: "hf:thoratsr7/embeddinggemma-300m-ONNX".into(),
                        url: "https://huggingface.co/thoratsr7/embeddinggemma-300m-ONNX/resolve/main/model.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
                // -------- Cascade scout: static-similarity-mrl-multilingual-v1 --------
                Model {
                    slug: "static-similarity-mrl-multilingual-v1".into(),
                    family: "static-similarity-mrl".into(),
                    kind: ModelKind::Embedder,
                    dim: 1024,
                    max_context_window: 512,
                    source: "hf:sentence-transformers/static-similarity-mrl-multilingual-v1".into(),
                    artifacts: vec![Artifact {
                        backend: BackendKind::Ort,
                        quantization: Quantization::Int8,
                        source: "hf:sentence-transformers/static-similarity-mrl-multilingual-v1".into(),
                        url: "https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1/resolve/main/onnx/model_int8.onnx".into(),
                        sha256: None,
                        extra_files: vec![],
                    }],
                    multi_vector: false,
                },
            ],

            gui_embedders: vec![
                GuiEmbedder {
                    slug: "harrier-oss-v1-270m-ONNX-q4f16".into(),
                    name: "Harrier 270M ONNX".into(),
                    memory_mb: 600,
                    default: true,
                    chunk_size: 1024,
                },
                GuiEmbedder {
                    slug: "harrier-oss-v1-270m-MLX-4bit".into(),
                    name: "Harrier 270M MLX (Metal)".into(),
                    memory_mb: 600,
                    default: false,
                    chunk_size: 1024,
                },
                GuiEmbedder {
                    slug: "harrier-oss-v1-0.6b-ONNX-q4".into(),
                    name: "Harrier 600M ONNX".into(),
                    memory_mb: 2200,
                    default: false,
                    chunk_size: 1024,
                },
                GuiEmbedder {
                    slug: "harrier-oss-v1-0.6b-MLX-4bit".into(),
                    name: "Harrier 600M MLX (Metal)".into(),
                    memory_mb: 2200,
                    default: false,
                    chunk_size: 1024,
                },
            ],

            gui_judge_artifact: GuiJudgeArtifact {
                default: Some(PathBuf::from("gui/judges/harrier-600m-v1.json")),
                by_embedder: {
                    let mut map = std::collections::BTreeMap::new();
                    map.insert(
                        "harrier-oss-v1-270m-ONNX-q4f16".into(),
                        PathBuf::from("gui/judges/harrier-270m-v1.json"),
                    );
                    map
                },
            },

            cascade: CascadeConfig {
                scout_model: "static-similarity-mrl-multilingual-v1".into(),
                scout_enabled: true,
                scout_chunk_size: 512,
                k1_embed_per_wave: 100,
                k2_judge_per_wave: 50,
                judge_pool_size: 50,
                small_job_threshold: 300,
                stage0_static_prior: None,
            },
        }
    }

    // ------------------------------------------------------------------
    // Convenience accessors
    // ------------------------------------------------------------------

    /// Looks up a model definition by its slug.
    pub fn find_model(&self, slug: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.slug == slug)
    }

    /// Looks up a family definition by its family name.
    pub fn find_family(&self, family: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.family == family)
    }

    /// Returns the GUI embedder marked `default: true` (if any).
    pub fn default_embedder(&self) -> Option<&GuiEmbedder> {
        self.gui_embedders.iter().find(|e| e.default)
    }

    /// Resolves which judge-artifact path to use for a given embedder slug.
    /// Falls back to `gui_judge_artifact.default` when no per-slug override
    /// exists.
    pub fn judge_artifact_for(&self, embedder_slug: &str) -> &str {
        self.gui_judge_artifact
            .by_embedder
            .get(embedder_slug)
            .map(|p| p.to_str().unwrap_or(""))
            .or(self.gui_judge_artifact.default.as_deref().and_then(|p| p.to_str()))
            .unwrap_or("")
    }

    /// Merge user preference overrides on top of the config.
    /// Only fields set in `overrides` replace the defaults; the rest stay.
    pub fn with_user_overrides(mut self, overrides: &UserOverrides) -> Self {
        if let Some(ref b) = overrides.engine_backend {
            if let Ok(backend) = b.parse::<Backend>() {
                self.engine.backend = backend;
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
        if let Some(v) = overrides.engine_embed_sessions {
            self.engine.embed_sessions = Some(EmbedSessions::Fixed(v as u32));
        }
        if let Some(v) = overrides.engine_provider_logging {
            self.engine.provider_logging = Some(v);
        }
        self
    }
}

// ============================================================================
// Engine / syscap / rubric
// ============================================================================

/// ONNX-runtime resource ceilings. Every `Option` field is `None` here so the
/// runtime budget derived from [`SyscapConfig`] takes effect — set to `Some(_)`
/// only to clamp further.
#[derive(Debug, Clone, Default)]
pub struct EngineConfig {
    /// Intra-op thread count. `None` = derived from the libsyscap budget.
    pub intra_threads: Option<usize>,
    /// Inter-op thread count. `None` = derived (single-threaded dispatch).
    pub inter_threads: Option<usize>,
    /// Embedding batch size. `None` = derived (bigger batches are slower for
    /// bandwidth-bound embedders).
    pub embed_batch: Option<usize>,
    /// How many sessions a batched embed is spread over.
    pub embed_sessions: Option<EmbedSessions>,
    /// Execution provider for the ONNX sessions.
    pub exec_provider: Option<ExecutionProvider>,
    /// Raise ORT's own logging on load.
    pub provider_logging: Option<bool>,
    /// Backend selection policy.
    pub backend: Backend,
    /// Hard cap on the modelcaller process tree CPU affinity.
    pub max_cores: Option<usize>,
    /// Benchmark switch: no cache ever consulted.
    pub no_cache: Option<bool>,
}

impl EngineConfig {
    /// Get intra_threads with default from libsyscap budget.
    pub fn intra_threads(&self) -> Option<usize> {
        self.intra_threads
    }
    /// Get inter_threads with default.
    pub fn inter_threads(&self) -> Option<usize> {
        self.inter_threads
    }
    /// Get embed_batch with default.
    pub fn embed_batch(&self) -> Option<usize> {
        self.embed_batch
    }
    /// Get embed_sessions with default.
    pub fn embed_sessions(&self) -> Option<EmbedSessions> {
        self.embed_sessions
    }
    /// Get exec_provider with default.
    pub fn exec_provider(&self) -> Option<ExecutionProvider> {
        self.exec_provider
    }
    /// Get provider_logging with default.
    pub fn provider_logging(&self) -> Option<bool> {
        self.provider_logging
    }
    /// Get backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }
    /// Get max_cores with default.
    pub fn max_cores(&self) -> Option<usize> {
        self.max_cores
    }
    /// Get no_cache with default.
    pub fn no_cache(&self) -> Option<bool> {
        self.no_cache
    }
}

/// ONNX backend selection policy (for engine config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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

    pub fn allows_mlx(&self) -> bool {
        matches!(self, Backend::Auto | Backend::Mlx)
    }

    pub fn requires_mlx(&self) -> bool {
        matches!(self, Backend::Mlx)
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

/// Which ONNX execution provider the wrapper registers for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// How many sessions a batched embed is spread over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedSessions {
    #[default]
    Auto,
    Fixed(u32),
}

impl EmbedSessions {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            EmbedSessions::Auto => None,
            EmbedSessions::Fixed(n) => Some(*n),
        }
    }
}

impl std::str::FromStr for EmbedSessions {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(EmbedSessions::Auto),
            n => n.parse::<u32>().map(EmbedSessions::Fixed).map_err(|_| format!("invalid embed_sessions: {s}")),
        }
    }
}

/// Live system-capacity budget consulted by the GUI, bench, tests, and build
/// tools to decide how much CPU/RAM they may consume right now.
#[derive(Debug, Clone, Default)]
pub struct SyscapConfig {
    /// Master switch. `false` = the budget reports the configured engine caps
    /// untouched and samples no load.
    pub enabled: bool,
    /// When true, throttle under Polite mode if running on battery.
    pub on_battery_polite: bool,
    /// Parallel build jobs (`auto` derives from `reserve_cores`).
    pub build_jobs: BuildJobs,
    /// Cores reserved for the OS / GUI and never consumed by inference.
    pub reserve_cores: u32,
    /// Fractional core reservation, applied on top of `reserve_cores`.
    pub reserve_core_frac: f32,
    /// Free-RAM fraction below which the budget drops to Backoff.
    pub min_free_mem_frac: f64,
    /// Free-RAM fraction below which heavy work pauses (Hold).
    pub hold_free_mem_frac: f64,
    /// 1-minute load average per logical core above which the budget drops to Polite.
    pub load_backoff: f64,
    /// 1-minute load average per logical core above which the budget floors everything.
    pub load_stall: f64,
    /// Minimum wall time between load samples in ms.
    pub sample_min_interval_ms: u64,
}

impl SyscapConfig {
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn reserve_cores(&self) -> u32 {
        self.reserve_cores
    }
    pub fn reserve_core_frac(&self) -> f64 {
        self.reserve_core_frac as f64
    }
    pub fn min_free_mem_frac(&self) -> f64 {
        self.min_free_mem_frac
    }
    pub fn hold_free_mem_frac(&self) -> f64 {
        self.hold_free_mem_frac
    }
    pub fn load_backoff(&self) -> f64 {
        self.load_backoff
    }
    pub fn load_stall(&self) -> f64 {
        self.load_stall
    }
    pub fn on_battery_polite(&self) -> bool {
        self.on_battery_polite
    }
    pub fn sample_min_interval_ms(&self) -> u64 {
        self.sample_min_interval_ms
    }
    pub fn build_jobs(&self) -> BuildJobs {
        self.build_jobs
    }
}

/// Build-parallelism policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildJobs {
    /// Derive from `reserve_cores` and live capacity.
    #[default]
    Auto,
    /// Hard pin to a fixed count.
    Fixed(usize),
}

/// Default rubric weights (points) applied to new jobs and newly added
/// criteria. Uses f64 for backward compatibility with YAML parsing.
#[derive(Debug, Clone, Copy)]
pub struct RubricDefaults {
    pub ad_points: f64,
    pub fb_job_points: f64,
    pub fb_profile_points: f64,
    pub criterion_points: f64,
}

impl Default for RubricDefaults {
    fn default() -> Self {
        RubricDefaults {
            ad_points: 50.0,
            fb_job_points: 10.0,
            fb_profile_points: 5.0,
            criterion_points: 10.0,
        }
    }
}

// ============================================================================
// Families
// ============================================================================

/// A model family definition: how tokenizer/text-prefix conventions are
/// applied for that family.
#[derive(Debug, Clone)]
pub struct Family {
    /// Family identifier, matched against `Model::family`.
    pub family: String,
    /// Tokenization/prefix convention to apply.
    pub usage: Usage,
}

/// Tokenization / prefix convention used by a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Usage {
    /// Plain pass-through (no prefix).
    Plain,
    /// Prepend the `nomic` query/passage prefix.
    NomicPrefix,
    /// Prepend the BGE `Represent this sentence...` instruction.
    BgePrefix,
    /// Prepend the E5 `query:` / `passage:` prefix.
    E5Prefix,
    /// Harrier-style instruct prompt.
    HarrierInstruct,
    /// Plain pair input for cross-encoder rerankers.
    PlainPair,
    /// ColBERT-style multi-token prefix.
    ColbertPrefix,
    /// JSON-mode extraction prompt (LLM).
    ExtractionJson,
}

// ============================================================================
// Models / artifacts
// ============================================================================

/// A single model entry with all of its downloadable artifacts.
#[derive(Debug, Clone)]
pub struct Model {
    /// Unique human-readable identifier (also used as the on-disk dir name).
    pub slug: String,
    /// Family this model belongs to (matches [`Family::family`]).
    pub family: String,
    /// Functional role of the model.
    pub kind: ModelKind,
    /// Embedding dimensionality; `0` for non-embedding models.
    pub dim: u32,
    /// Maximum input tokens the model accepts.
    pub max_context_window: u32,
    /// Canonical source identifier (`hf:<org>/<repo>`).
    pub source: String,
    /// One or more concrete, downloadable artifacts.
    pub artifacts: Vec<Artifact>,
    /// `true` when the model emits multiple vectors per input (ColBERT-style).
    pub multi_vector: bool,
}

/// Functional role of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Produces dense (or multi-) vector embeddings.
    Embedder,
    /// LLM used for structured evidence extraction.
    Extractor,
    /// Cross-encoder pair reranker.
    Reranker,
}

/// A concrete, downloadable artifact for a model.
#[derive(Debug, Clone)]
pub struct Artifact {
    /// Required inference backend.
    pub backend: BackendKind,
    /// Quantization scheme applied to this artifact.
    pub quantization: Quantization,
    /// Source identifier for this specific artifact.
    pub source: String,
    /// Direct download URL.
    pub url: String,
    /// Optional SHA-256 checksum for download verification.
    pub sha256: Option<String>,
    /// Extra files (tokenizer, config, ...) that must be downloaded alongside
    /// the main model file.
    pub extra_files: Vec<String>,
}

/// Concrete inference backend required by an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// ONNX Runtime.
    Ort,
    /// Apple MLX.
    Mlx,
}

/// Quantization scheme applied to an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// 4-bit weight-only quantization (`q4`).
    Q4,
    /// 4-bit weight + FP16 activations (`q4f16`).
    Q4F16,
    /// MLX 4-bit packing.
    FourBit,
    /// Generic ONNX dynamic quantization (`quantized`).
    Quantized,
    /// 8-bit integer quantization (`int8`).
    Int8,
    /// No quantization — full-precision weights.
    None,
}

// ============================================================================
// GUI embedders
// ============================================================================

/// GUI-facing embedder entry: surfaces a subset of [`Model`]s to the user
/// with per-entry memory / chunking hints.
#[derive(Debug, Clone)]
pub struct GuiEmbedder {
    /// Slug of the underlying [`Model`].
    pub slug: String,
    /// Human-readable display name shown in the UI.
    pub name: String,
    /// Approximate peak resident memory in MiB (used for budgeting).
    pub memory_mb: u32,
    /// `true` selects this embedder as the GUI default.
    pub default: bool,
    /// Token chunk size used when ingesting documents.
    pub chunk_size: u32,
}

// ============================================================================
// Judge artifacts
// ============================================================================

/// Per-embedder judge-artifact path overrides.
/// This mirrors the old GuiJudgeArtifacts structure for backward compatibility.
#[derive(Debug, Clone, Default)]
pub struct GuiJudgeArtifact {
    /// Fallback path used when no per-slug override matches.
    pub default: Option<PathBuf>,
    /// Embedder slug → artifact path.
    pub by_embedder: std::collections::BTreeMap<String, PathBuf>,
}

impl GuiJudgeArtifact {
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

// ============================================================================
// Cascade
// ============================================================================

/// Progressive-ranking cascade configuration.
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Slug of the cheap scout model used at stage 0.5.
    pub scout_model: String,
    /// Master switch for the scout stage.
    pub scout_enabled: bool,
    /// Token chunk size for the scout model.
    pub scout_chunk_size: u32,
    /// Top-K candidates kept after stage 1 (per wave).
    pub k1_embed_per_wave: u32,
    /// Top-K candidates kept after the judge stage (per wave).
    pub k2_judge_per_wave: u32,
    /// The frozen judge pool size.
    pub judge_pool_size: u32,
    /// Below this document count the cascade collapses to a single pass.
    pub small_job_threshold: u32,
    /// Static (Model2Vec-style) stage-0 semantic prior slug.
    pub stage0_static_prior: Option<String>,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        CascadeConfig {
            scout_model: "static-similarity-mrl-multilingual-v1".into(),
            scout_enabled: true,
            scout_chunk_size: 512,
            k1_embed_per_wave: 100,
            k2_judge_per_wave: 50,
            judge_pool_size: 50,
            small_job_threshold: 300,
            stage0_static_prior: None,
        }
    }
}

impl CascadeConfig {
    pub fn k1(&self) -> usize {
        self.k1_embed_per_wave as usize
    }
    pub fn pool_size(&self) -> usize {
        self.judge_pool_size as usize
    }
    pub fn small_job_threshold(&self) -> usize {
        self.small_job_threshold as usize
    }
    pub fn scout_chunk_size(&self) -> usize {
        self.scout_chunk_size as usize
    }
    pub fn scout_enabled(&self) -> bool {
        self.scout_enabled
    }
    pub fn scout(&self) -> Option<&str> {
        if self.scout_enabled() {
            Some(&self.scout_model)
        } else {
            None
        }
    }
}

// ============================================================================
// User Overrides (from AppPreferences)
// ============================================================================

/// User preference overrides for engine settings (from app preferences).
#[derive(Debug, Default, Clone)]
pub struct UserOverrides {
    pub engine_backend: Option<String>,
    pub engine_exec_provider: Option<String>,
    pub engine_embed_batch: Option<usize>,
    pub engine_intra_threads: Option<usize>,
    pub engine_inter_threads: Option<usize>,
    pub engine_embed_sessions: Option<usize>,
    pub engine_provider_logging: Option<bool>,
    pub engine_max_cores: Option<usize>,
    pub engine_no_cache: Option<bool>,
    /// Idle-unload timeout for the embedding engine in seconds (0 = disable idle unload).
    pub gui_idle_unload_secs: Option<u64>,
}

impl UserOverrides {
    /// Returns true if all fields are `None` (i.e., no overrides set).
    pub fn is_default(&self) -> bool {
        self.engine_backend.is_none()
            && self.engine_exec_provider.is_none()
            && self.engine_embed_batch.is_none()
            && self.engine_intra_threads.is_none()
            && self.engine_inter_threads.is_none()
            && self.engine_embed_sessions.is_none()
            && self.engine_provider_logging.is_none()
            && self.engine_max_cores.is_none()
            && self.engine_no_cache.is_none()
    }
}

// ============================================================================
// CLI Overrides (for command-line config overrides)
// ============================================================================

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

// ============================================================================
// Config Error
// ============================================================================

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

// ============================================================================
// Constants
// ============================================================================

pub const DEFAULT_GUI_IDLE_UNLOAD_SECS: u64 = 300;
pub const DEFAULT_CASCADE_K1: usize = 100;
pub const DEFAULT_CASCADE_POOL: usize = 50;
pub const DEFAULT_CASCADE_SMALL_JOB: usize = 300;
pub const DEFAULT_SCOUT_CHUNK_SIZE: usize = 512;

pub const DEFAULT_SYSCAP_RESERVE_CORES: u32 = 1;
pub const DEFAULT_SYSCAP_RESERVE_CORE_FRAC: f64 = 0.10;
pub const DEFAULT_SYSCAP_MIN_FREE_MEM_FRAC: f64 = 0.20;
pub const DEFAULT_SYSCAP_HOLD_FREE_MEM_FRAC: f64 = 0.05;
pub const DEFAULT_SYSCAP_LOAD_BACKOFF: f64 = 1.0;
pub const DEFAULT_SYSCAP_LOAD_STALL: f64 = 2.0;
pub const DEFAULT_SYSCAP_ON_BATTERY_POLITE: bool = true;
pub const DEFAULT_SYSCAP_SAMPLE_MIN_INTERVAL_MS: u64 = 250;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_roundtrips_expected_counts() {
        let cfg = Config::load();
        assert_eq!(cfg.families.len(), 10);
        assert_eq!(cfg.models.len(), 18);
        assert_eq!(cfg.gui_embedders.len(), 4);
    }

    #[test]
    fn default_embedder_is_harrier_270m_onnx() {
        let cfg = Config::load();
        let default = cfg.default_embedder().expect("a default embedder exists");
        assert_eq!(default.slug, "harrier-oss-v1-270m-ONNX-q4f16");
    }

    #[test]
    fn judge_artifact_resolution() {
        let cfg = Config::load();
        assert_eq!(
            cfg.judge_artifact_for("harrier-oss-v1-270m-ONNX-q4f16"),
            "gui/judges/harrier-270m-v1.json"
        );
        // Unknown slug falls back to default.
        assert_eq!(
            cfg.judge_artifact_for("does-not-exist"),
            "gui/judges/harrier-600m-v1.json"
        );
    }

    #[test]
    fn scout_model_is_present() {
        let cfg = Config::load();
        assert!(cfg.find_model(&cfg.cascade.scout_model).is_some());
    }

    #[test]
    fn mlateon_is_multi_vector() {
        let cfg = Config::load();
        let m = cfg.find_model("mlateon").expect("mlateon exists");
        assert!(m.multi_vector);
    }

    #[test]
    fn engine_config_has_all_fields() {
        let cfg = Config::load();
        // Just verify the fields exist and have correct types
        let _ = cfg.engine.intra_threads;
        let _ = cfg.engine.inter_threads;
        let _ = cfg.engine.embed_batch;
        let _ = cfg.engine.embed_sessions;
        let _ = cfg.engine.exec_provider;
        let _ = cfg.engine.provider_logging;
        let _ = cfg.engine.backend;
        let _ = cfg.engine.max_cores;
        let _ = cfg.engine.no_cache;
    }

    #[test]
    fn syscap_config_has_all_fields() {
        let cfg = Config::load();
        let _ = cfg.syscap.enabled;
        let _ = cfg.syscap.on_battery_polite;
        let _ = cfg.syscap.build_jobs;
        let _ = cfg.syscap.reserve_cores;
        let _ = cfg.syscap.reserve_core_frac;
        let _ = cfg.syscap.min_free_mem_frac;
        let _ = cfg.syscap.hold_free_mem_frac;
        let _ = cfg.syscap.load_backoff;
        let _ = cfg.syscap.load_stall;
        let _ = cfg.syscap.sample_min_interval_ms;
    }

    #[test]
    fn cascade_config_has_all_fields() {
        let cfg = Config::load();
        let _ = cfg.cascade.scout_model;
        let _ = cfg.cascade.scout_enabled;
        let _ = cfg.cascade.scout_chunk_size;
        let _ = cfg.cascade.k1_embed_per_wave;
        let _ = cfg.cascade.k2_judge_per_wave;
        let _ = cfg.cascade.judge_pool_size;
        let _ = cfg.cascade.small_job_threshold;
        let _ = cfg.cascade.stage0_static_prior;
    }
}