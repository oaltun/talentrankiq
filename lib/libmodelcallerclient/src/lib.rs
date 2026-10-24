//! Shared stdio-JSONL client to the `modelcaller` inference backend.
//!
//! This crate is the client-side of the modelcaller protocol contract. Both
//! `bench` and the gui use it so the envelope shape, command names
//! (`ping`/`embed`/`rerank`/`discover_schema`/`extract_constrained`/`models`/
//! `load`/`unload`/`shutdown`) and payload schemas
//! cannot drift between the two consumers (contract-lib pattern, `NOAI.md`).
//!
//! The crate is a thin protocol/transport layer with two transports:
//! [`ModelCallerClient::spawn_inprocess`] runs the modelcaller loop on a
//! thread inside the host process (bench, judgetrainer — there is no
//! modelcaller binary anymore), while [`ModelCallerClient::over`] attaches to
//! an already-running loop over arbitrary readers/writers — triqui-core
//! uses it to run modelcaller in-process behind the gui (thin gui pattern).
//! Requests are framed as envelopes and parsed into protocol-native types
//! (`Vec<Vec<f64>>`, `RerankResult`). Each consumer maps [`ModelCallerError`]
//! onto its own error type at the edge.
//!
//! Commands: `ping`/`embed`/`embed multi_vector`/`rerank`/`discover_schema`/
//! `extract_constrained`/`models`/`load`/`unload`/`shutdown`.

use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "inprocess")]
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelCallerError {
    #[error("failed to start modelcaller ({0}): {1}")]
    Spawn(String, String),
    #[error("modelcaller has no stdin")]
    NoStdin,
    #[error("modelcaller has no stdout")]
    NoStdout,
    #[error("failed to write to modelcaller: {0}")]
    Write(String),
    #[error("failed to read from modelcaller: {0}")]
    Read(String),
    #[error("modelcaller closed stdout")]
    Closed,
    #[error("serialization error: {0}")]
    Json(String),
    #[error("missing field in modelcaller response: {0}")]
    MissingField(String),
    #[error("modelcaller error: {0}")]
    Protocol(String),
}

/// JSONL envelope framing every request/response on the modelcaller stdio pipe.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    pub id: Option<String>,
    pub r#type: String,
    pub payload: Value,
}

/// One reranker result row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
}

/// Output of `discover_schema`: the validated `ExtractionSchema` json
/// (libcvevidence shape: `{"base":"v1","fields":[...]}`) and the
/// generation wall time in ms.
#[derive(Debug, Clone)]
pub struct DiscoveryOutput {
    pub schema: Value,
    pub gen_ms: u64,
}

/// Output of `extract_constrained`: the validated `extensions` map, the
/// field names dropped by validation, validation issues, and the
/// generation wall time in ms.
#[derive(Debug, Clone)]
pub struct ExtractOutput {
    pub extensions: serde_json::Map<String, Value>,
    pub dropped: Vec<String>,
    pub issues: Vec<String>,
    pub gen_ms: u64,
}

/// Multi-vector (late interaction) embedding of one text: `counts` tokens,
/// each with a `dim`-sized f32 vector. `data` is all token vectors
/// concatenated, row-major (`row(i)` = `data[i*dim..(i+1)*dim]`). Vectors are
/// L2-normalized by the wrapper. See bench/bench1/design/17-late-interaction.md.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiVecEmbedding {
    pub dim: usize,
    pub counts: Vec<u32>,
    pub data: Vec<f32>,
}

impl MultiVecEmbedding {
    pub fn n_tokens(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.dim)
    }
}

/// Registry metadata for one model (from the `models` command).
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub model: String,
    pub kind: String,
    pub dim: usize,
    pub max_context_window: usize,
    /// Which runtime served the load, as the arm reported it: `"ort"` or
    /// `"mlx"` (design 15 §15.4). `None` when the model was never loaded, so
    /// a caller can tell "not loaded yet" from "loaded on the CPU".
    ///
    /// This is the field that keeps an accelerated claim honest: the value
    /// comes from the arm's own load response, not from what the host asked
    /// for, so a silent fallback cannot read as acceleration.
    pub runtime: Option<String>,
}

pub struct ModelCallerClient {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    /// Request side: the in-memory writer handed to [`ModelCallerClient::over`].
    /// Dropping it is EOF for the loop.
    stdin: Option<Box<dyn std::io::Write + Send>>,
    stdout: Option<BufReader<Box<dyn std::io::Read + Send>>>,
    next_id: u64,
    /// Time of the most recent request (attempted, whether or not it succeeded).
    /// `None` until the first request. Lets a host idle-unload the engine.
    last_used: Option<Instant>,
}

impl Clone for ModelCallerClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Inner {
    /// Ask modelcaller to shut down (which ends its in-process ORT wrapper
    /// thread). The write end dropping is EOF for the JSONL loop, which then
    /// tears the engine down.
    fn terminate(&mut self) {
        if let Some(ref mut stdin) = self.stdin {
            let envelope = Envelope {
                id: Some("driver-shutdown".to_string()),
                r#type: "request".to_string(),
                payload: serde_json::json!({ "cmd": "shutdown" }),
            };
            if let Ok(line) = serde_json::to_string(&envelope) {
                let _ = writeln!(stdin, "{}", line);
                let _ = stdin.flush();
            }
        }

        // Drain any unread response(s) left in the stdout pipe/buffer.
        // terminate writes the shutdown request but never reads the
        // response back, so modelcaller's {"ok":true,"shutdown_ok":true}
        // envelope lingers. If the client is reused (e.g. via a surviving
        // clone or Drop→terminate on a later Arc), a subsequent
        // send_request read_loop would encounter the stale envelope — it
        // gets skipped (id mismatch) but then read_line can return a
        // previously-buffered line that does match, yielding a bogus
        // MissingField error (e.g. "extensions" absent from the shutdown
        // payload that masqueraded as the extract_constrained response).
        if let Some(stdout) = &mut self.stdout {
            let mut buf = String::new();
            while stdout.read_line(&mut buf).map(|n| n > 0).unwrap_or(false) {
                buf.clear();
            }
        }

        // Drop stdin/stdout so any surviving clone gets NoStdin/NoStdout
        // on a subsequent send_request rather than writing to / reading
        // from a dead pipe.
        self.stdin = None;
        self.stdout = None;
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// `(backend, exec_provider)` wire strings for a config plus the effective
/// engine section: the section wins — it is what a clamping consumer passed
/// (bench1 pins `backend: ort` there for result comparability, design 15
/// §15.4) — the config falls through, the wire defaults come last. Reading
/// these from the raw config instead silently discarded the clamp: every
/// "ort-pinned" bench embed of a model with an `mlx:` variant ran on Metal.
#[cfg(any(test, feature = "inprocess"))]
fn effective_engine_io(
    engine: &libconfigreader::EngineConfig,
    cfg: &libconfigreader::Config,
) -> (String, String) {
    let backend = if engine.backend != libconfigreader::Backend::Auto {
        engine.backend.as_str().to_string()
    } else {
        cfg.engine.backend.as_str().to_string()
    };
    let exec_provider = engine
        .exec_provider
        .as_ref()
        .map(|p| p.as_str().to_string())
        .or_else(|| {
            cfg.engine
                .exec_provider
                .as_ref()
                .map(|p| p.as_str().to_string())
        })
        .unwrap_or_else(|| "auto".to_string());
    (backend, exec_provider)
}

impl ModelCallerClient {
    /// Gracefully shut down the underlying modelcaller (which in turn ends
    /// its in-process onnxwrapperrs wrapper thread). Safe to call multiple times.
    pub fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.terminate();
    }

    /// Elapsed time since the most recent request, or `None` if no request was
    /// ever attempted. Clones share the stamp (it lives in the shared `Inner`).
    pub fn idle_for(&self) -> Option<Duration> {
        let inner = self.inner.lock().unwrap();
        inner.last_used.map(|t| t.elapsed())
    }

    /// Run the modelcaller JSONL loop **in this process** on a dedicated
    /// thread, wired through an OS-pipe pair, and return the connected
    /// client. Requires the `inprocess` feature. `config_path` is the shared
    /// `triq.yaml` (registry, model_dir, engine caps); the loop tears itself
    /// down on `shutdown` (which also ends the in-process ORT wrapper).
    #[cfg(feature = "inprocess")]
    pub fn spawn_inprocess(
        config_path: &Path,
        verbose: bool,
        logfile: Option<&str>,
    ) -> Result<Self, ModelCallerError> {
        let cfg = libconfigreader::load(config_path, &libconfigreader::CliOverrides::new())
            .map_err(|e| {
                ModelCallerError::Spawn("in-process".into(), format!("load config: {e}"))
            })?;
        Self::spawn_inprocess_with_engine(config_path, cfg.engine.clone(), verbose, logfile)
    }

    /// [`spawn_inprocess`] with explicit engine caps instead of the config's
    /// `engine:` section (contract-lib pattern: the signature here IS the
    /// contract). Consumers that clamp the caps — e.g. bench1 flooring its
    /// knobs to the `libsyscap` budget — pass the effective section; the
    /// registry/model_dir still resolve from `config_path`.
    #[cfg(feature = "inprocess")]
    pub fn spawn_inprocess_with_engine(
        config_path: &Path,
        engine: libconfigreader::EngineSection,
        verbose: bool,
        logfile: Option<&str>,
    ) -> Result<Self, ModelCallerError> {
        let cfg = libconfigreader::load(config_path, &libconfigreader::CliOverrides::new())
            .map_err(|e| {
                ModelCallerError::Spawn("in-process".into(), format!("load config: {e}"))
            })?;
        let registry = libmodelcaller::Registry::from_config(&cfg);
        let (backend, exec_provider) = effective_engine_io(&engine, &cfg);
         let args = libmodelcaller::StdioArgs {
            model_dir: Some(cfg.model_dir.clone()),
            revision: cfg.revision.clone(),
            backend,
            exec_provider,
            ..Default::default()
        };

        // Side a = host client, side b = the modelcaller loop: a_write →
        // b_read carries requests, b_write → a_read carries responses. The
        // thread ends when the loop does (`shutdown` request or write-end EOF).
        let pipe = |what: &str| {
            std::io::pipe().map_err(|e| {
                ModelCallerError::Spawn("in-process".into(), format!("{what} pipe: {e}"))
            })
        };
        let (b_read, a_write) = pipe("request")?;
        let (a_read, b_write) = pipe("response")?;
        let logfile = logfile.map(String::from);
        std::thread::Builder::new()
            .name("modelcaller".into())
            .spawn(move || {
                 if let Err(e) = libmodelcaller::run_stdio_over(
                    registry,
                    args,
                    engine,
                    cfg.syscap.clone(),
                    verbose,
                    logfile,
                    std::io::BufReader::new(b_read),
                    b_write,
                ) {
                    log::error!("in-process modelcaller exited: {e}");
                }
            })
            .map_err(|e| {
                ModelCallerError::Spawn("in-process".into(), format!("spawn thread: {e}"))
            })?;

        Ok(Self::over(Box::new(a_read), Box::new(a_write)))
    }

    /// Connect to an already-running modelcaller loop over an arbitrary
    /// transport (triqui-core's in-process engine wires
    /// `modelcaller::run_stdio_over` through an in-memory duplex — thin gui
    /// pattern; [`spawn_inprocess`] uses it too). The loop's `shutdown`
    /// command tears it down (which also ends its `onnxwrapperrs` wrapper).
    pub fn over(
        read: Box<dyn std::io::Read + Send>,
        write: Box<dyn std::io::Write + Send>,
    ) -> Self {
        let inner = Inner {
            stdin: Some(write),
            stdout: Some(BufReader::new(read)),
            next_id: 1,
            last_used: None,
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// A disconnected client. Useful for unit tests that only exercise
    /// request framing / parsing and never hit the wire.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn new_test() -> Self {
        let inner = Inner {
            stdin: None,
            stdout: None,
            next_id: 1,
            last_used: None,
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Unload a model from the wrapper session, freeing its RAM.
    pub fn unload(&self, model: &str) -> Result<(), ModelCallerError> {
        let resp = self.send_request(json!({ "cmd": "unload", "model": model }))?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller unload error: {error}"
            )));
        }
        Ok(())
    }

    /// Warm a model into the wrapper session without embedding anything.
    pub fn load(&self, model: &str) -> Result<(), ModelCallerError> {
        let resp = self.send_request(json!({ "cmd": "load", "model": model }))?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!("modelcaller load error: {error}")));
        }
        Ok(())
    }

    /// Max context window for `model` (falls back to 2048 if unknown).
    pub fn model_max_ctx(&self, model: &str) -> Result<usize, ModelCallerError> {
        Ok(self.model_metadata(model)?.max_context_window)
    }

    /// Registry metadata for `model`: slug, kind (embedder/reranker),
    /// embedding dim, max context window. Errors if the model is unknown.
    pub fn model_metadata(&self, model: &str) -> Result<ModelMeta, ModelCallerError> {
        let _ = self.ping();
        let resp = self.send_request(json!({ "cmd": "models", "model": model }))?;
        if let Some(models) = resp.get("models").and_then(|v| v.as_array()) {
            for m in models {
                if m.get("model").and_then(|v| v.as_str()) == Some(model) {
                    return Ok(ModelMeta {
                        model: model.to_string(),
                        kind: m
                            .get("kind")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        dim: m.get("dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        max_context_window: m
                            .get("max_context_window")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(2048) as usize,
                        runtime: m
                            .get("runtime")
                            .and_then(|v| v.as_str())
                            .map(str::to_string),
                    });
                }
            }
        }
        Err(ModelCallerError::Protocol(format!("model not found in registry: {model}")))
    }

    /// Embed `texts`. `texts_role` is optional; when present it must have the
    /// same length as `texts` and is `["query" | "document", ...]`. Tagging the
    /// job ad as `query` and CVs/criteria as `document` is correctness-critical
    /// for nomic/e5/bge models — see the gui backend design doc.
    pub fn embed(
        &self,
        model: &str,
        texts: &[&str],
        texts_role: Option<&[&str]>,
    ) -> Result<Vec<Vec<f64>>, ModelCallerError> {
        let mut payload = json!({
            "cmd": "embed",
            "model": model,
            "texts": texts,
        });
        if let Some(roles) = texts_role {
            if roles.len() != texts.len() {
                return Err(ModelCallerError::Protocol(format!(
                    "texts_role length ({}) != texts length ({})",
                    roles.len(),
                    texts.len()
                )));
            }
            payload["texts_role"] = json!(roles);
        }
        let resp = self.send_request(payload)?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller embed error: {error}"
            )));
        }
        let embeddings = resp
            .get("embeddings")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ModelCallerError::MissingField("embeddings".to_string()))?;
        let mut result = Vec::with_capacity(embeddings.len());
        for emb in embeddings {
            let vec = emb
                .as_array()
                .ok_or_else(|| ModelCallerError::MissingField("embedding row".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect::<Vec<_>>();
            result.push(vec);
        }
        Ok(result)
    }

    /// Multi-vector embed: one L2-normalized token vector set per text. Only
    /// valid for models flagged `multi_vector` in the registry (e.g.
    /// ColBERT-style late-interaction models); the wrapper applies the
    /// projection chain server-side. Role tagging is handled by the model's
    /// usage adapter (`colbert-prefix`).
    pub fn embed_multi(
        &self,
        model: &str,
        texts: &[&str],
    ) -> Result<Vec<MultiVecEmbedding>, ModelCallerError> {
        let resp = self.send_request(json!({
            "cmd": "embed",
            "model": model,
            "texts": texts,
        }))?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller embed error: {error}"
            )));
        }
        parse_multi_vec_response(&resp, texts.len())
    }

    pub fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<RerankResult>, ModelCallerError> {
        let payload = json!({
            "cmd": "rerank",
            "model": model,
            "query": query,
            "docs": documents,
        });
        let resp = self.send_request(payload)?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller rerank error: {error}"
            )));
        }
        let results = resp
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ModelCallerError::MissingField("results".to_string()))?;
        let mut vec = Vec::with_capacity(results.len());
        for r in results {
            let mut index = 0;
            let mut relevance_score = 0.0;
            if let Some(obj) = r.as_object() {
                if let Some(i) = obj.get("index").and_then(|v| v.as_u64()) {
                    index = i as usize;
                }
                if let Some(s) = obj.get("relevance_score").and_then(|v| v.as_f64()) {
                    relevance_score = s;
                }
            }
            vec.push(RerankResult { index, relevance_score });
        }
        Ok(vec)
    }

    pub fn ping(&self) -> Result<bool, ModelCallerError> {
        let resp = self.send_request(json!({ "cmd": "ping" }))?;
        Ok(resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    /// llm schema discovery (design 18 m3a / modelcaller design 14): a job
    /// ad becomes a validated `ExtractionSchema` (libcvevidence shape).
    /// `opts` is forwarded (e.g. `max_new_tokens`).
    pub fn discover_schema(
        &self,
        model: &str,
        ad_text: &str,
        opts: Option<&Value>,
    ) -> Result<DiscoveryOutput, ModelCallerError> {
        let mut payload = json!({
            "cmd": "discover_schema",
            "model": model,
            "ad_text": ad_text,
        });
        if let Some(o) = opts {
            payload["opts"] = o.clone();
        }
        let resp = self.send_request(payload)?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller discover_schema error: {error}"
            )));
        }
        let schema = resp
            .get("schema")
            .cloned()
            .ok_or_else(|| ModelCallerError::MissingField("schema".to_string()))?;
        let gen_ms = resp.get("gen_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        Ok(DiscoveryOutput { schema, gen_ms })
    }

    /// llm constrained extraction (design 18 m3a): a cv + discovered schema
    /// becomes the validated `extensions` map plus per-field drop/issues
    /// diagnostics. The caller merges them onto the stage-a deterministic
    /// `CvEvidence` (base fields are never model-generated).
    pub fn extract_constrained(
        &self,
        model: &str,
        cv_text: &str,
        schema: &Value,
        opts: Option<&Value>,
    ) -> Result<ExtractOutput, ModelCallerError> {
        let mut payload = json!({
            "cmd": "extract_constrained",
            "model": model,
            "cv_text": cv_text,
            "schema": schema,
        });
        if let Some(o) = opts {
            payload["opts"] = o.clone();
        }
        let resp = self.send_request(payload)?;
        if let Some(error) = resp.get("error") {
            return Err(ModelCallerError::Protocol(format!(
                "modelcaller extract_constrained error: {error}"
            )));
        }
        let parse_map = |key: &str| -> Result<serde_json::Map<String, Value>, ModelCallerError> {
            resp.get(key)
                .and_then(|v| v.as_object())
                .cloned()
                .ok_or_else(|| ModelCallerError::MissingField(key.to_string()))
        };
        let extensions = parse_map("extensions")?;
        let dropped: Vec<String> = resp
            .get("dropped")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let issues: Vec<String> = resp
            .get("issues")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let gen_ms = resp.get("gen_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        Ok(ExtractOutput {
            extensions,
            dropped,
            issues,
            gen_ms,
        })
    }

    fn send_request(&self, payload: Value) -> Result<Value, ModelCallerError> {
        // One lock guard across id alloc + write + read: the stdio protocol is
        // lock-step, and releasing the lock between the write and read phases
        // let two concurrent callers interleave and consume each other's
        // response lines — the earlier-id waiter then blocked forever while the
        // busy flag stayed set (imports stuck at "0/N").
        let mut inner = self.inner.lock().unwrap();
        inner.last_used = Some(Instant::now());
        let id = inner.next_id;
        inner.next_id += 1;
        let id_str = id.to_string();

        let envelope = Envelope {
            id: Some(id_str.clone()),
            r#type: "request".to_string(),
            payload,
        };

        let line = serde_json::to_string(&envelope)
            .map_err(|e| ModelCallerError::Json(e.to_string()))?;
        if let Some(stdin) = &mut inner.stdin {
            writeln!(stdin, "{}", line).map_err(|e| ModelCallerError::Write(e.to_string()))?;
            stdin
                .flush()
                .map_err(|e| ModelCallerError::Write(e.to_string()))?;
        } else {
            return Err(ModelCallerError::NoStdin);
        }

        if let Some(stdout) = &mut inner.stdout {
            let mut buf = String::new();
            loop {
                buf.clear();
                let n = stdout
                    .read_line(&mut buf)
                    .map_err(|e| ModelCallerError::Read(e.to_string()))?;
                if n == 0 {
                    return Err(ModelCallerError::Closed);
                }
                if let Ok(envelope) = serde_json::from_str::<Envelope>(buf.trim()) {
                    if envelope.id.as_deref() == Some(&id_str) {
                        // an error envelope carries {code, msg} in its payload;
                        // surface it verbatim instead of silently dropping
                        // type="" and returning a MissingField downstream
                        if envelope.r#type == "error" {
                            let code = envelope
                                .payload
                                .get("code")
                                .and_then(|v| v.as_str())
                                .unwrap_or("error");
                            let msg = envelope
                                .payload
                                .get("msg")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            return Err(ModelCallerError::Protocol(format!(
                                "modelcaller {code}: {msg}"
                            )));
                        }
                        return Ok(envelope.payload);
                    }
                }
            }
        } else {
            Err(ModelCallerError::NoStdout)
        }
    }
}

/// Cosine similarity between two equal-length f64 vectors; 0.0 on length
/// mismatch or zero vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Parse a `multi_vector` embed response: `{"multi_vector": true, "dim": D,
/// "counts": [...], "embeddings_b64": ["<f32 LE base64>", ...]}` — one entry
/// per text. Validates shapes and decodes to f32 rows.
pub fn parse_multi_vec_response(
    resp: &Value,
    expected_texts: usize,
) -> Result<Vec<MultiVecEmbedding>, ModelCallerError> {
    if resp.get("multi_vector").and_then(|v| v.as_bool()) != Some(true) {
        return Err(ModelCallerError::Protocol(
            "expected multi_vector embed response".to_string(),
        ));
    }
    let dim = resp
        .get("dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ModelCallerError::MissingField("dim".to_string()))? as usize;
    if dim == 0 {
        return Err(ModelCallerError::Protocol(
            "multi_vector response has dim 0".to_string(),
        ));
    }
    let counts = resp
        .get("counts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ModelCallerError::MissingField("counts".to_string()))?;
    let blobs = resp
        .get("embeddings_b64")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ModelCallerError::MissingField("embeddings_b64".to_string()))?;
    if counts.len() != expected_texts || blobs.len() != expected_texts {
        return Err(ModelCallerError::Protocol(format!(
            "multi_vector response has {} texts, expected {expected_texts}",
            counts.len()
        )));
    }

    use base64::Engine as _;
    let mut out = Vec::with_capacity(expected_texts);
    for (count, blob) in counts.iter().zip(blobs.iter()) {
        let count = count
            .as_u64()
            .ok_or_else(|| ModelCallerError::Protocol("bad counts entry".to_string()))?
            as usize;
        let b64 = blob
            .as_str()
            .ok_or_else(|| ModelCallerError::Protocol("bad embeddings_b64 entry".to_string()))?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| ModelCallerError::Protocol(format!("bad base64 embedding: {e}")))?;
        if bytes.len() != count * dim * 4 {
            return Err(ModelCallerError::Protocol(format!(
                "embedding blob is {} bytes, expected {} ({} tokens x {dim} dims x 4)",
                bytes.len(),
                count * dim * 4,
                count
            )));
        }
        let data = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.push(MultiVecEmbedding {
            dim,
            counts: vec![count as u32],
            data,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_section(backend: Option<libconfigreader::Backend>) -> libconfigreader::EngineConfig {
        libconfigreader::EngineConfig {
            intra_threads: None,
            inter_threads: None,
            embed_batch: None,
            embed_sessions: None,
            exec_provider: None,
            provider_logging: None,
            backend: backend.unwrap_or(libconfigreader::Backend::Auto),
            max_cores: None,
            no_cache: None,
        }
    }

    fn config_with_engine(yaml_engine: &str) -> libconfigreader::Config {
        let dir = std::env::temp_dir().join(format!("ikmc-io-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("cfg-{}.yaml", yaml_engine.len() as u64 * 31 + yaml_engine.as_bytes().iter().map(|&b| b as u64).sum::<u64>()));
        std::fs::write(
            &path,
            format!("models: []\nfamilies: []\n{yaml_engine}"),
        )
        .unwrap();
        libconfigreader::Config::load()
    }

    #[test]
    fn the_effective_engine_section_wins_over_the_raw_config() {
        // bench1 pins `backend: ort` in the section it passes while the
        // config says `auto` (Metal wherever a model declares an mlx
        // variant). Reading the config instead sent every pinned bench run
        // to the MLX arm — the pin existed precisely to prevent that.
        let cfg = config_with_engine("engine:\n  backend: auto\n");
        let (backend, exec_provider) =
            effective_engine_io(&engine_section(Some(libconfigreader::Backend::Ort)), &cfg);
        assert_eq!(backend, "ort", "the clamped section must win over the config");
        assert_eq!(exec_provider, "auto", "unset falls through to the config/default");

        // No pin in the section: the config's `auto` falls through.
        let (backend, _) = effective_engine_io(&engine_section(None), &cfg);
        assert_eq!(backend, "auto");
    }

    #[test]
    fn a_config_without_engine_defaults_to_the_wire_defaults() {
        let cfg = config_with_engine("");
        let (backend, exec_provider) = effective_engine_io(&engine_section(None), &cfg);
        assert_eq!(backend, "auto");
        assert_eq!(exec_provider, "auto");
    }

    #[test]
    fn over_transport_roundtrip() {
        // Fake modelcaller loop on the far end of an in-memory duplex: read
        // one envelope, echo an ok response. Exercises the `over` transport
        // (the gui's in-process engine) end to end without spawning a process.
        let (cw, cr, sw, sr) = libtriqui_client::transport::duplex();
        let server = std::thread::spawn(move || {
            use std::io::{BufRead, Write};
            let (mut sr, mut sw) = (sr, sw);
            let mut line = String::new();
            sr.read_line(&mut line).unwrap();
            let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
            let resp = serde_json::json!({
                "id": v["id"],
                "type": "response",
                "payload": { "ok": true },
            });
            sw.write_all(format!("{resp}\n").as_bytes()).unwrap();
            // sw dropped here -> the client sees EOF after the response.
        });
        let client = ModelCallerClient::over(Box::new(cr), Box::new(cw));
        assert!(client.ping().unwrap());
        server.join().unwrap();
    }

    #[test]
    fn envelope_round_trip() {
        let env = Envelope {
            id: Some("7".to_string()),
            r#type: "request".to_string(),
            payload: json!({ "cmd": "ping" }),
        };
        let s = serde_json::to_string(&env).unwrap();
        let back: Envelope = serde_json::from_str(&s).unwrap();
        assert_eq!(back.id.as_deref(), Some("7"));
        assert_eq!(back.r#type, "request");
        assert_eq!(back.payload["cmd"], "ping");
    }

    #[test]
    fn texts_role_length_guarded() {
        let client = ModelCallerClient::new_test();
        let err = client
            .embed("m", &["a", "b"], Some(&["query"]))
            .unwrap_err();
        assert!(format!("{err}").contains("texts_role length"));
    }

    #[test]
    fn idle_for_stamps_on_attempted_requests() {
        let client = ModelCallerClient::new_test();
        assert!(client.idle_for().is_none(), "no request yet");
        // A disconnected client still stamps activity on an attempted request
        // (it fails with NoStdin, but the attempt counts as usage).
        let _ = client.ping();
        let idle = client.idle_for().expect("stamped by attempted request");
        assert!(idle < Duration::from_secs(10));
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert_eq!(cosine_similarity(&[1.0], &[2.0, 3.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    fn b64(vals: &[f32]) -> String {
        use base64::Engine as _;
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn parse_multi_vec_response_round_trip() {
        let resp = json!({
            "multi_vector": true,
            "dim": 2,
            "counts": [2, 1, 0],
            "embeddings_b64": [
                b64(&[1.0, 0.0, 0.0, 1.0]),
                b64(&[0.6, 0.8]),
                b64(&[]),
            ],
        });
        let out = parse_multi_vec_response(&resp, 3).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].dim, 2);
        assert_eq!(out[0].n_tokens(), 2);
        assert_eq!(out[0].row(0), &[1.0, 0.0]);
        assert_eq!(out[0].row(1), &[0.0, 1.0]);
        assert_eq!(out[1].row(0), &[0.6, 0.8]);
        assert_eq!(out[2].n_tokens(), 0);
        assert_eq!(out[2].data.len(), 0);
        assert_eq!(out[0].rows().count(), 2);
    }

    #[test]
    fn parse_multi_vec_response_rejects_bad() {
        // not flagged multi_vector
        let resp = json!({ "dim": 2, "counts": [], "embeddings_b64": [] });
        assert!(parse_multi_vec_response(&resp, 0).is_err());
        // wrong text count
        let resp = json!({ "multi_vector": true, "dim": 2, "counts": [1], "embeddings_b64": [b64(&[1.0, 0.0])] });
        assert!(parse_multi_vec_response(&resp, 2).is_err());
        // blob size mismatch
        let resp = json!({ "multi_vector": true, "dim": 2, "counts": [1], "embeddings_b64": [b64(&[1.0])] });
        assert!(parse_multi_vec_response(&resp, 1).is_err());
        // missing fields
        let resp = json!({ "multi_vector": true });
        assert!(parse_multi_vec_response(&resp, 1).is_err());
    }

    #[test]
    fn embed_multi_requires_live_pipe() {
        let client = ModelCallerClient::new_test();
        assert!(client.embed_multi("m", &["a"]).is_err());
    }
}
