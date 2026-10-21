//! Embedded default configuration, replacing the former `triq.yaml` file.
//!
//! The model registry, families, and gui_embedders are baked into the binary
//! as a YAML string. At runtime, `Config::default_from_embedded()` parses it
//! once, and `Config::with_user_overrides()` merges in optional user overrides
//! (engine/backend settings from `AppPreferences`).

/// The full default config as YAML, replacing triq.yaml.
/// model_dir is platform-specific, set at runtime via TRIQI_MODEL_DIR.
pub const DEFAULT_CONFIG_YAML: &str = r#"
verbose: false
revision: main
gui_idle_unload_secs: 300

engine:
  backend: auto

syscap:
  on_battery_polite: true
  build_jobs: auto
  reserve_cores: 1
  reserve_core_frac: 0.10

rubric_defaults:
  ad_points: 50
  fb_job_points: 10
  fb_profile_points: 5
  criterion_points: 10

families:
  - family: static-similarity-mrl
    usage: plain
  - family: nomic-embed-text-v1
    usage: nomic-prefix
  - family: bge-small-en-v1.5
    usage: bge-prefix
  - family: multilingual-e5-large
    usage: e5-prefix
  - family: harrier-oss-v1
    usage: harrier-instruct
  - family: bge-reranker-v2-m3
    usage: plain-pair
  - family: jina-reranker-v1-tiny-en
    usage: plain-pair
  - family: granite-embedding-311m-multilingual-r2
    usage: plain
  - family: mlateon
    usage: colbert-prefix
  - family: qwen-instruct
    usage: extraction-json

models:

  - slug: nomic-embed-text-v1.5
    family: nomic-embed-text-v1
    kind: embedder
    dim: 768
    max_context_window: 2048
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:nomic-ai/nomic-embed-text-v1.5
        url: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx

  - slug: bge-small-en-v1.5
    family: bge-small-en-v1.5
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:Xenova/bge-small-en-v1.5
        url: https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx

  - slug: multilingual-e5-large
    family: multilingual-e5-large
    kind: embedder
    dim: 1024
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "qint8_avx512_vnni"
        source: hf:Xenova/multilingual-e5-large
        url: https://huggingface.co/Xenova/multilingual-e5-large/resolve/main/onnx/model_quantized.onnx

  - slug: harrier-oss-v1-0.6b
    family: harrier-oss-v1
    kind: embedder
    dim: 1024
    max_context_window: 32768
    artifacts:
      - backend: ort
        quantization: "fp16"
        source: hf:onnx-community/harrier-oss-v1-0.6b-ONNX
        url: https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_fp16.onnx
        sha256: "460cf6e8d8da587e60976b1e36bb5e433c79e5af9681dbab48abad9353d4957c"
      - backend: ort
        quantization: "q4"
        source: hf:onnx-community/harrier-oss-v1-0.6b-ONNX
        url: https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_q4.onnx
        sha256: "4a41218d088c2e31fd6a3953d838d08536a27e98cf987796104a9511f9d9a671"
      - backend: mlx
        quantization: "4bit"
        source: hf:majentik/harrier-oss-v1-0.6b-MLX-4bit
        url: https://huggingface.co/majentik/harrier-oss-v1-0.6b-MLX-4bit/resolve/main/model.safetensors
        sha256: "362866d8cd7476138d095b1123599d25d5a590d6d9a9130892e0516b8b57481c"

  - slug: harrier-oss-v1-270m
    family: harrier-oss-v1
    kind: embedder
    dim: 640
    max_context_window: 32768
    artifacts:
      - backend: ort
        quantization: "q4f16"
        source: hf:onnx-community/harrier-oss-v1-270m-ONNX
        url: https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx
        sha256: "e844fc75743d24f9bea799c35018f8d47efec3fd34d74f43906ee0a1899bcc83"
      - backend: mlx
        quantization: "4bit"
        source: hf:majentik/harrier-oss-v1-270m-MLX-4bit
        url: https://huggingface.co/majentik/harrier-oss-v1-270m-MLX-4bit/resolve/main/model.safetensors
        sha256: "f8bbe0be03d9e45b3297908218659b5fa8ef290f666bede0060644ecb283a537"

  # LLM evidence extractor
  - slug: qwen3-4b-instruct-2507
    family: qwen-instruct
    kind: extractor
    dim: 0
    max_context_window: 4096
    artifacts:
      - backend: ort
        quantization: "q4"
        source: hf:onnx-community/Qwen3-4B-Instruct-2507-ONNX
        url: https://huggingface.co/onnx-community/Qwen3-4B-Instruct-2507-ONNX/resolve/main/onnx/model_q4.onnx

  - slug: bge-reranker-v2-m3
    family: bge-reranker-v2-m3
    kind: reranker
    dim: 0
    max_context_window: 8192
    artifacts:
      - backend: ort
        quantization: "quantized"
        source: hf:onnx-community/bge-reranker-v2-m3-ONNX
        url: https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/onnx/model_quantized.onnx

  - slug: jina-reranker-v1-tiny-en
    family: jina-reranker-v1-tiny-en
    kind: reranker
    dim: 0
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:jinaai/jina-reranker-v1-tiny-en
        url: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/resolve/main/onnx/model.onnx

  - slug: granite-embedding-311m-multilingual-r2
    family: granite-embedding-311m-multilingual-r2
    kind: embedder
    dim: 768
    max_context_window: 32768
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:ibm-granite/granite-embedding-311m-multilingual-r2
        url: https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/resolve/main/onnx/model.onnx
      - backend: ort
        quantization: "quint8_avx2"
        source: hf:ibm-granite/granite-embedding-311m-multilingual-r2
        url: https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/resolve/main/onnx/model_quint8_avx2.onnx

  - slug: mlateon
    family: mlateon
    kind: embedder
    dim: 128
    max_context_window: 8192
    multi_vector: true
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:lightonai/mLateOn
        url: https://huggingface.co/lightonai/mLateOn/resolve/main/model.onnx
      - backend: ort
        quantization: "int8"
        source: hf:lightonai/mLateOn
        url: https://huggingface.co/lightonai/mLateOn/resolve/main/model_int8.onnx

  # --- Additional fast multilingual embedders ---
  - slug: multilingual-e5-small
    family: multilingual-e5-small
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:intfloat/multilingual-e5-small
        url: https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx

  - slug: paraphrase-multilingual-MiniLM-L12-v2
    family: paraphrase-multilingual-MiniLM-L12-v2
    kind: embedder
    dim: 384
    max_context_window: 128
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        url: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx

  - slug: snowflake-arctic-embed-xs
    family: snowflake-arctic-embed-xs
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "int8"
        source: hf:Snowflake/snowflake-arctic-embed-xs
        url: https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model_int8.onnx

  - slug: snowflake-arctic-embed-s
    family: snowflake-arctic-embed-s
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:Snowflake/snowflake-arctic-embed-s
        url: https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model.onnx

  - slug: snowflake-arctic-embed-m-v2.0
    family: snowflake-arctic-embed-m-v2.0
    kind: embedder
    dim: 768
    max_context_window: 8192
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:Snowflake/snowflake-arctic-embed-m-v2.0
        url: https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/resolve/main/onnx/model.onnx

  - slug: granite-embedding-97M-Multilingual-R2
    family: granite-embeding-97M-Multilingual-R2
    kind: embedder
    dim: 384
    max_context_window: 8192
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:ibm-granite/granite-embedding-97M-Multilingual-R2
        url: https://huggingface.co/ibm-granite/granite-embedding-97M-Multilingual-R2/resolve/main/onnx/model.onnx

  - slug: amaretto-embed-148m
    family: amaretto-embed-148m
    kind: embedder
    dim: 384
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:AmarettoLabs/amaretto-embed-148m-ONNX
        url: https://huggingface.co/AmarettoLabs/amaretto-embed-148m-ONNX/resolve/main/model.onnx

  - slug: multilingual-e5-base
    family: multilingual-e5-base
    kind: embedder
    dim: 768
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:intfloat/multilingual-e5-base
        url: https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/onnx/model.onnx

  - slug: embeddinggemma-300m
    family: embeddinggemma-300m
    kind: embedder
    dim: 768
    max_context_window: 2048
    artifacts:
      - backend: ort
        quantization: "none"
        source: hf:thoratsr7/embeddinggemma-300m-ONNX
        url: https://huggingface.co/thoratsr7/embeddinggemma-300m-ONNX/resolve/main/onnx/model.onnx

  # Cascade scout (stage 0.5): cheap multilingual gate
  - slug: static-similarity-mrl-multilingual-v1
    family: static-similarity-mrl
    kind: embedder
    dim: 1024
    max_context_window: 512
    artifacts:
      - backend: ort
        quantization: "int8"
        source: hf:sentence-transformers/static-similarity-mrl-multilingual-v1
        url: https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1/resolve/main/onnx/model_int8.onnx

gui_embedders:
  - slug: harrier-oss-v1-0.6b
    name: Harrier 600M
    memory_mb: 2200
    default: true
    chunk_size: 1024
  - slug: harrier-oss-v1-270m
    name: Harrier 270M
    memory_mb: 600
    chunk_size: 1024

# gui judge stage artifacts
gui_judge_artifact:
  default: "gui/judges/harrier-600m-v1.json"
  harrier-oss-v1-270m: "gui/judges/harrier-270m-v1.json"

# Progressive-ranking cascade
cascade:
  scout_model: static-similarity-mrl-multilingual-v1
  scout_enabled: true
  scout_chunk_size: 512
  k1_embed_per_wave: 100
  k2_judge_per_wave: 50
  small_job_threshold: 300
"#;

/// Parse the embedded default config YAML into a `Config`.
pub fn default_config(config_path: std::path::PathBuf) -> crate::Config {
    let raw: crate::RawConfig = serde_yaml::from_str(DEFAULT_CONFIG_YAML)
        .expect("embedded DEFAULT_CONFIG_YAML must parse");
    crate::from_raw(raw, &config_path)
        .expect("embedded config must validate")
}

/// User preference overrides that layer on top of the embedded config.
/// Populated from `AppPreferences` in the tauri frontend.
/// Only the fields the user has explicitly set are `Some`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UserOverrides {
    /// Override the engine backend: "auto" | "ort" | "mlx".
    pub engine_backend: Option<String>,
    /// Override the ONNX execution provider.
    pub engine_exec_provider: Option<String>,
    /// Override the embed batch size.
    pub engine_embed_batch: Option<usize>,
    /// Override intra-op thread count.
    pub engine_intra_threads: Option<usize>,
    /// Override inter-op thread count.
    pub engine_inter_threads: Option<usize>,
    /// Override max cores cap.
    pub engine_max_cores: Option<usize>,
    /// Disable the embedding cache (benchmark mode).
    pub engine_no_cache: Option<bool>,
}

impl UserOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when no override field is set.
    pub fn is_default(&self) -> bool {
        self.engine_backend.is_none()
            && self.engine_exec_provider.is_none()
            && self.engine_embed_batch.is_none()
            && self.engine_intra_threads.is_none()
            && self.engine_inter_threads.is_none()
            && self.engine_max_cores.is_none()
            && self.engine_no_cache.is_none()
    }
}
