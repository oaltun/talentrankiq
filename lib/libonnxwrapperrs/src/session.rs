use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

use libtokenizer::TokenizerHandle;
use ndarray::{Array2, Array3, ArrayD, ArrayView2, ArrayView3, Axis, Ix2, Ix3};
use ort::session::{Session, SessionInputValue, SessionOutputs, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use serde_json::Value;

use crate::projection;

pub struct SessionInfo {
    pub handle: u32,
    pub kind: String,
    pub dim: usize,
    /// Sessions the batched embed path spreads the text list over (design 15
    /// §15.7). 1 = no fan-out. Reported so the host can see the split.
    pub embed_sessions: usize,
    /// Threads each pool member gets (`intra_threads / embed_sessions`).
    pub threads_per_session: Option<usize>,
    /// Execution provider this session was built with ("auto" = ORT CPU EP).
    /// Reported so a host or test can assert which rung actually ran — a
    /// silent CPU fallback is otherwise indistinguishable from success.
    pub exec_provider: String,
}

/// Multi-vector embed result: `counts[i]` tokens for text i, each token a
/// `dim`-sized f32 vector, all rows concatenated little-endian in `blobs[i]`.
pub struct MultiVecEmbeddings {
    pub dim: usize,
    pub counts: Vec<u32>,
    pub blobs: Vec<Vec<u8>>,
}

enum EncoderOutput {
    /// All texts tokenized to zero tokens.
    Empty,
    /// `[batch, seq, dim]` token states (pre-pooling models).
    TokenStates(Array3<f32>),
    /// `[batch, dim]` already-pooled embedding.
    Pooled(Array2<f32>),
}

pub(crate) struct LoadedSession {
    /// Sessions of the SAME model. Index 0 is the primary (the one the
    /// single-sequence paths — rerank, generate — use). A batched embed
    /// spreads the text list across ALL of them in parallel, each with
    /// `intra_threads / pool.len()` threads (design
    /// `lib/modelcaller/design/15-acceleration-backends.md` §15.7).
    ///
    /// Why: on a many-core box a single batched forward pass does not
    /// saturate the cores (measured: 5 -> 16 threads bought only ~1.3x on
    /// the gui suite), so N sessions over disjoint slices of one text list
    /// scale better than one session with N threads. This is the
    /// CPU-native alternative to an accelerator, which wants batches
    /// instead. Pool of 1 = today's behavior, bit for bit.
    pub(crate) pool: Vec<Session>,
    pub(crate) kind: String,
    pub(crate) dim: usize,
    pub(crate) tokenizer: TokenizerHandle,
    pub(crate) max_context_window: usize,
    /// Pad batches up to a fixed length ladder before running (design 15
    /// §15.6). True when an accelerator provider is registered: those
    /// compile the graph per shape, so a per-batch `max_len` is a recompile
    /// per batch — measured at ~100 s per distinct length on the CoreML EP,
    /// which made the accelerated leg 8.5x SLOWER than the CPU it replaced.
    /// The CPU EP does not care (padding is pure waste there), so this is
    /// provider-conditional.
    pub(crate) bucket_shapes: bool,
    /// Directory of the loaded ONNX; relative `projection_files` resolve here.
    pub(crate) model_dir: PathBuf,
    /// Lazily-loaded projection chain for multi-vector (late interaction)
    /// models, cached across embed calls; `projection_key` is the resolved
    /// projection file list it was built from.
    pub(crate) projection: Option<std::sync::Arc<projection::ProjectionChain>>,
    pub(crate) projection_key: Option<Vec<String>>,
    /// Lazily-built vocabulary category sets for `extractor` sessions
    /// (design 04 §4.3), cached across generate calls.
    pub(crate) token_class: Option<std::sync::Arc<crate::generate::TokenClass>>,
    /// `(num_key_value_heads, head_dim)` for decoder kv-cache tensors —
    /// the shape of an empty past-kv feed on prefill
    pub(crate) kv_spec: Option<(usize, usize)>,
}

pub struct SessionManager {
    sessions: HashMap<u32, LoadedSession>,
    next_handle: u32,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_handle: 1,
        }
    }

    /// `fan_out` = sessions to spread a batched embed over (design 15 §15.7).
    /// `0` (or absent in the payload) = auto: pick by the model's weight
    /// footprint via [`auto_fan_out`]. An explicit count wins. When > 1 the
    /// total thread budget is split evenly across the pool (each session gets
    /// `intra_threads / fan_out`, floor 1).
    ///
    /// `verbose_provider` raises ORT's own logging to VERBOSE so node
    /// placement is visible. It exists because an accelerated rung's failure
    /// mode is SILENT: a session whose ops all fall back to CPU loads fine
    /// and looks like a provider that "works". Never on by default — the log
    /// is verbose and per-load.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        &mut self,
        model_path: &str,
        exec_provider: &str,
        _precision: &str,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        fan_out: usize,
        verbose_provider: bool,
    ) -> Result<SessionInfo, (String, String)> {
        log::info!("starting loading model {}", model_path);

        let fan_out = if fan_out == 0 {
            let dir = std::path::Path::new(model_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            auto_fan_out(dir)
        } else {
            fan_out
        };
        let fan_out = fan_out.max(1);
        let per_session_threads = per_session_threads(intra_threads, fan_out);

        let model_dir = std::path::Path::new(model_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = TokenizerHandle::from_file(&tokenizer_path).map_err(|e| {
            (
                "load_err".into(),
                format!("failed to load tokenizer from {tokenizer_path:?}: {e}"),
            )
        })?;

        let config_path = model_dir.join("config.json");
        let model_config = ModelConfig::load(&config_path)?;

        let mut session_builder = Session::builder().map_err(|e| {
            (
                "load_err".into(),
                format!("failed to create session builder: {e}"),
            )
        })?;

        if let Some(threads) = per_session_threads {
            session_builder = session_builder.with_intra_threads(threads).map_err(|e| {
                (
                    "load_err".into(),
                    format!("failed to set intra_threads: {e}"),
                )
            })?;
        }
        if let Some(threads) = inter_threads {
            session_builder = session_builder.with_inter_threads(threads).map_err(|e| {
                (
                    "load_err".into(),
                    format!("failed to set inter_threads: {e}"),
                )
            })?;
        }

        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| {
                (
                    "load_err".into(),
                    format!("failed to set optimization level: {e}"),
                )
            })?;

        // ORT's placement report is the ONLY way to tell a provider that
        // worked from one that silently CPU-fell-back (design 15 §15.4.1), so
        // raise it when asked and forward it into our log.
        if verbose_provider {
            session_builder = session_builder
                .with_log_level(ort::logging::LogLevel::Verbose)
                .map_err(|e| {
                    ("load_err".into(), format!("failed to set log level: {e}"))
                })?;
            session_builder = session_builder
                .with_logger(std::sync::Arc::new(
                    |level: ort::logging::LogLevel,
                     category: &str,
                     _id: &str,
                     _code_location: &str,
                     message: &str| {
                        log::info!("ort[{level:?}/{category}]: {message}");
                    },
                ))
                .map_err(|e| {
                    ("load_err".into(), format!("failed to set logger: {e}"))
                })?;
        }

        if exec_provider != "auto" && exec_provider != "cpu" {
            session_builder = apply_exec_provider(session_builder, exec_provider)?;
        }

        let session = session_builder.commit_from_file(model_path).map_err(|e| {
            (
                "load_err".into(),
                format!("failed to load model {model_path}: {e}"),
            )
        })?;

        log::info!("ended loading model {}", model_path);

        let kind = model_config.kind.unwrap_or_else(|| infer_kind(&session));
        // a decoder's hidden size is meaningless as an embedding dim
        let kv_spec = match (model_config.num_key_value_heads, model_config.num_attention_heads) {
            (Some(kv_heads), Some(heads)) if heads > 0 => {
                model_config.dim.map(|hidden| (kv_heads, hidden / heads))
            }
            _ => None,
        };
        let dim = if kind == "extractor" {
            0
        } else {
            model_config.dim.unwrap_or_else(|| infer_dim(&session))
        };
        let max_context_window = model_config.max_context_window.unwrap_or(512);

        let handle = self.next_handle;
        self.next_handle += 1;

        let mut pool = vec![session];
        if fan_out > 1 {
            pool.extend(self.build_pool(
                model_path,
                exec_provider,
                per_session_threads,
                inter_threads,
                fan_out - 1,
            )?);
        }
        let pool_len = pool.len();
        let embed_sessions = pool_len;
        let threads_per_session = per_session_threads;
        let log_line = format!(
            "loaded model {model_path}: handle={handle}, kind={kind}, dim={dim}, embed_sessions={embed_sessions}, exec_provider={exec_provider}"
        );

        self.sessions.insert(
            handle,
            LoadedSession {
                pool,
                kind: kind.clone(),
                dim,
                tokenizer,
                max_context_window,
                bucket_shapes: exec_provider != "auto" && exec_provider != "cpu",
                model_dir: model_dir.to_path_buf(),
                projection: None,
                projection_key: None,
                token_class: None,
                kv_spec,
            },
        );
        let info = SessionInfo {
            handle,
            kind,
            dim,
            embed_sessions,
            threads_per_session,
            exec_provider: exec_provider.to_string(),
        };

        log::info!("{log_line}");

        Ok(info)
    }

    /// Build `n` additional sessions of the same model. Split out of `load`
    /// so the pool and the primary are built by identical code.
    fn build_pool(
        &self,
        model_path: &str,
        exec_provider: &str,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        n: usize,
    ) -> Result<Vec<Session>, (String, String)> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut builder = Session::builder().map_err(|e| {
                ("load_err".into(), format!("failed to create session builder: {e}"))
            })?;
        if let Some(threads) = intra_threads {
            builder = builder.with_intra_threads(threads).map_err(|e| {
                    ("load_err".into(), format!("failed to set intra_threads: {e}"))
                })?;
            }
            if let Some(threads) = inter_threads {
                builder = builder.with_inter_threads(threads).map_err(|e| {
                    ("load_err".into(), format!("failed to set inter_threads: {e}"))
                })?;
            }
            builder = builder
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| {
                    ("load_err".into(), format!("failed to set optimization level: {e}"))
                })?;
            if exec_provider != "auto" {
                builder = apply_exec_provider(builder, exec_provider)?;
            }
            let s = builder.commit_from_file(model_path).map_err(|e| {
                ("load_err".into(), format!("failed to load model {model_path}: {e}"))
            })?;
            out.push(s);
        }
        Ok(out)
    }

    pub fn embed(
        &mut self,
        handle: u32,
        texts: &[&str],
        opts: Option<&serde_json::Map<String, Value>>,
    ) -> Result<Vec<Vec<f32>>, (String, String)> {
        let normalize = opts
            .and_then(|o| o.get("normalize"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let max_ctx = opts
            .and_then(|o| o.get("max_context_window"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or_else(|| self.session_max_ctx(handle));

        let (dim, tokenized, output) = self.run_encoder(handle, texts, max_ctx)?;

        match output {
            EncoderOutput::Empty => Ok(texts.iter().map(|_| vec![0.0f32; dim]).collect()),
            EncoderOutput::TokenStates(hidden) => {
                let n = texts.len();
                let max_len = tokenized.iter().map(|t| t.len()).max().unwrap_or(0);
                let mut mask = Array2::<i64>::zeros((n, max_len));
                for (i, toks) in tokenized.iter().enumerate() {
                    for j in 0..toks.len() {
                        mask[[i, j]] = 1;
                    }
                }
                Ok(mean_pool_normalize(&hidden, &mask, dim, normalize))
            }
            EncoderOutput::Pooled(pooled) => {
                let mut result = Vec::with_capacity(pooled.nrows());
                for row in pooled.rows() {
                    let mut v: Vec<f32> = row.to_vec();
                    if normalize {
                        projection::l2_normalize(&mut v);
                    }
                    result.push(v);
                }
                Ok(result)
            }
        }
    }

    /// Multi-vector (late interaction) embed: returns one L2-normalized
    /// projection-dim vector per *token*, packed f32 little-endian. When
    /// `opts.projection_files` is given (PyLate Dense safetensors), the chain
    /// is applied to the token states first and cached on the session; when
    /// absent, the model is expected to already emit projected token states
    /// (e.g. mLateOn's fused exports) and they are only normalized. See
    /// bench/bench1/design/17-late-interaction.md.
    pub fn embed_multi(
        &mut self,
        handle: u32,
        texts: &[&str],
        opts: Option<&serde_json::Map<String, Value>>,
    ) -> Result<MultiVecEmbeddings, (String, String)> {
        let proj_files: Vec<PathBuf> = opts
            .and_then(|o| o.get("projection_files"))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(PathBuf::from))
                    .collect()
            })
            .unwrap_or_default();

        {
            let loaded = self
                .sessions
                .get_mut(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            if loaded.kind != "embedder" {
                return Err((
                    "bad_kind".into(),
                    format!("session {handle} is a {}, not an embedder", loaded.kind),
                ));
            }

            // projection_files are repo-relative paths inside the model dir
            // (as downloaded); absolute paths pass through untouched.
            let resolved: Vec<PathBuf> = if proj_files.is_empty() {
                Vec::new()
            } else {
                proj_files
                    .iter()
                    .map(|p| {
                        if p.is_absolute() {
                            p.clone()
                        } else {
                            loaded.model_dir.join(p)
                        }
                    })
                    .collect()
            };

            let key: Vec<String> = resolved
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            if loaded.projection_key.as_ref() != Some(&key) {
                let chain = if resolved.is_empty() {
                    None
                } else {
                    Some(std::sync::Arc::new(projection::ProjectionChain::load(
                        &resolved, loaded.dim,
                    )?))
                };
                loaded.projection = chain;
                loaded.projection_key = Some(key);
                log::info!(
                    "multi-vector: {} projection chain ({})",
                    if resolved.is_empty() {
                        "without"
                    } else {
                        "with"
                    },
                    resolved.len()
                );
            }
        }

        let max_ctx = opts
            .and_then(|o| o.get("max_context_window"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or_else(|| self.session_max_ctx(handle));

        let (_, tokenized, output) = self.run_encoder(handle, texts, max_ctx)?;

        let (dim, counts, blobs) = match output {
            EncoderOutput::Empty => {
                let dim = {
                    let loaded = self.sessions.get(&handle);
                    loaded
                        .and_then(|l| l.projection.as_ref().map(|p| p.out_dim))
                        .unwrap_or_else(|| loaded.map(|l| l.dim).unwrap_or(0))
                };
                (dim, vec![0u32; texts.len()], vec![Vec::new(); texts.len()])
            }
            EncoderOutput::TokenStates(hidden) => {
                let (chain, model_dim) = {
                    let loaded = self.sessions.get(&handle);
                    (
                        loaded.and_then(|l| l.projection.clone()),
                        loaded.map(|l| l.dim).unwrap_or(0),
                    )
                };
                let d = hidden.dim().2;
                let dim = chain.as_ref().map(|c| c.out_dim).unwrap_or(d);
                if let Some(c) = &chain {
                    if d != c.in_dim {
                        return Err((
                            "projection_err".into(),
                            format!(
                                "model emits {d}-dim token states, projection expects {}",
                                c.in_dim
                            ),
                        ));
                    }
                }
                let mut counts = Vec::with_capacity(tokenized.len());
                let mut blobs = Vec::with_capacity(tokenized.len());
                for (i, toks) in tokenized.iter().enumerate() {
                    let mut blob = Vec::with_capacity(toks.len() * dim * 4);
                    for j in 0..toks.len() {
                        let mut v: Vec<f32> = (0..d).map(|k| hidden[[i, j, k]]).collect();
                        match &chain {
                            Some(c) => v = c.project_token(&v),
                            None => projection::l2_normalize(&mut v),
                        }
                        for x in &v {
                            blob.extend_from_slice(&x.to_le_bytes());
                        }
                    }
                    counts.push(toks.len() as u32);
                    blobs.push(blob);
                }
                let _ = model_dim;
                (dim, counts, blobs)
            }
            EncoderOutput::Pooled(_) => {
                return Err((
                    "bad_model".into(),
                    "multi_vector embed requires a token-state ([batch, seq, dim]) model output"
                        .into(),
                ));
            }
        };

        Ok(MultiVecEmbeddings { dim, counts, blobs })
    }

    fn session_max_ctx(&self, handle: u32) -> usize {
        self.sessions
            .get(&handle)
            .map(|l| l.max_context_window)
            .unwrap_or(512)
    }

    /// Tokenize, pad, run the encoder, and return the raw output tensor.
    /// `dim` is the model's declared dim (for the Empty case). Borrow of the
    /// session ends before this returns (output data is owned).
    fn run_encoder(
        &mut self,
        handle: u32,
        texts: &[&str],
        max_ctx: usize,
    ) -> Result<(usize, Vec<Vec<u32>>, EncoderOutput), (String, String)> {
        let (dim, tokenized) = {
            let loaded = self
                .sessions
                .get_mut(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            if loaded.kind != "embedder" {
                return Err((
                    "bad_kind".into(),
                    format!("session {handle} is a {}, not an embedder", loaded.kind),
                ));
            }
            let tokenized: Vec<Vec<u32>> = texts
                .iter()
                .map(|t| {
                    loaded
                        .tokenizer
                        .encode_truncated(t, true, max_ctx)
                        .map(|enc| enc.ids)
                        .unwrap_or_default()
                })
                .collect();
            (loaded.dim, tokenized)
        };

        let max_len = tokenized.iter().map(|t| t.len()).max().unwrap_or(0);
        if max_len == 0 {
            return Ok((dim, tokenized, EncoderOutput::Empty));
        }

        // Shape bucketing (design 15 §15.6): under an accelerator provider,
        // round the padded length up to the fixed ladder so the session
        // compiles a handful of shapes instead of one per distinct request
        // length. CPU-only sessions skip it — padding is pure waste there.
        let max_len = {
            let loaded = self
                .sessions
                .get(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            if loaded.bucket_shapes {
                bucket_len(max_len, loaded.max_context_window)
            } else {
                max_len
            }
        };

        // Fan-out (design 15 §15.7): with more than one session in the pool,
        // slice the token rows across the sessions and run the slices
        // CONCURRENTLY. Each slice pads to the same global `max_len`, so the
        // per-row math is identical — only the batch shape and the number of
        // threads per forward pass change.
        let parts = {
            let loaded = self
                .sessions
                .get(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            loaded.pool.len().min(tokenized.len())
        };
        if parts > 1 {
            return self.run_encoder_pooled(handle, tokenized, max_len, dim, parts);
        }

        let (input_ids_val, attention_mask_val, token_type_ids_val) =
            build_batch_tensors(&tokenized, max_len)?;

        let output = {
            let loaded = self
                .sessions
                .get_mut(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            let inputs = build_text_inputs(
                &loaded.pool[0],
                input_ids_val,
                attention_mask_val,
                token_type_ids_val,
            );
            let outputs = loaded.pool[0]
                .run(inputs)
                .map_err(|e| ("inference_err".into(), format!("inference failed: {e}")))?;
            first_output_f32(&outputs)?
        };

        let output = classify_output(output)?;

        Ok((dim, tokenized, output))
    }

    /// Run one batched embed across `parts` sessions concurrently, then stitch
    /// the slices back in input order. Sessions are moved out of the map for
    /// the duration (each thread needs its own `&mut Session`) and put back
    /// before returning, so the manager keeps owning them.
    fn run_encoder_pooled(
        &mut self,
        handle: u32,
        tokenized: Vec<Vec<u32>>,
        max_len: usize,
        dim: usize,
        parts: usize,
    ) -> Result<(usize, Vec<Vec<u32>>, EncoderOutput), (String, String)> {
        let mut pool = {
            let loaded = self
                .sessions
                .get_mut(&handle)
                .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
            std::mem::take(&mut loaded.pool)
        };
        let n = tokenized.len();
        // Contiguous, balanced slices: row i goes to part (i * parts) / n, so
        // the reconstruction below is a plain concatenation.
        let mut slices: Vec<(usize, usize)> = Vec::with_capacity(parts);
        for p in 0..parts {
            let start = n * p / parts;
            let end = n * (p + 1) / parts;
            slices.push((start, end));
        }

        let results: Vec<Result<ArrayD<f32>, (String, String)>> = std::thread::scope(|s| {
            // `iter_mut` hands out disjoint `&mut Session` borrows, one per
            // worker; each worker owns its session for the run's duration.
            let mut sessions_iter = pool.iter_mut();
            let mut handles = Vec::with_capacity(parts);
            for (start, end) in slices.iter() {
                let rows = &tokenized[*start..*end];
                let session = sessions_iter
                    .next()
                    .expect("one session per slice");
                let max_len = max_len;
                handles.push(s.spawn(move || {
                    let (ids, mask, types) = build_batch_tensors(rows, max_len)?;
                    let inputs = build_text_inputs(session, ids, mask, types);
                    let outputs = session
                        .run(inputs)
                        .map_err(|e| ("inference_err".into(), format!("inference failed: {e}")))?;
                    first_output_f32(&outputs)
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err((
                            "inference_err".into(),
                            "embed pool worker panicked".to_string(),
                        ))
                    })
                })
                .collect()
        });

        // Put the sessions back before anything can fail, so a mid-flight
        // error does not lose the model (the host would otherwise have to
        // reload it).
        if let Some(loaded) = self.sessions.get_mut(&handle) {
            loaded.pool = pool;
        }

        let mut parts_out: Vec<ArrayD<f32>> = Vec::with_capacity(parts);
        for r in results {
            parts_out.push(r?);
        }
        let rank_error = || {
            (
                "inference_err".into(),
                "embed pool mixed tensor ranks across sessions".to_string(),
            )
        };
        let merged: ArrayD<f32> = match parts_out[0].ndim() {
            2 => {
                let views: Vec<ArrayView2<'_, f32>> = parts_out
                    .iter()
                    .map(|a| a.view().into_dimensionality::<Ix2>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| rank_error())?;
                ndarray::concatenate(Axis(0), &views)
                    .map_err(|e| ("inference_err".into(), format!("failed to join pool slices: {e}")))?
                    .into_dimensionality::<ndarray::IxDyn>()
                    .map_err(|e| {
                        ("inference_err".into(), format!("pool join rank error: {e}"))
                    })?
            }
            3 => {
                let views: Vec<ArrayView3<'_, f32>> = parts_out
                    .iter()
                    .map(|a| a.view().into_dimensionality::<Ix3>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| rank_error())?;
                ndarray::concatenate(Axis(0), &views)
                    .map_err(|e| ("inference_err".into(), format!("failed to join pool slices: {e}")))?
                    .into_dimensionality::<ndarray::IxDyn>()
                    .map_err(|e| {
                        ("inference_err".into(), format!("pool join rank error: {e}"))
                    })?
            }
            r => {
                return Err((
                    "inference_err".into(),
                    format!("expected 2D/3D tensor, got rank {r}"),
                ));
            }
        };

        let output = classify_output(merged)?;
        Ok((dim, tokenized, output))
    }

    pub fn rerank(
        &mut self,
        handle: u32,
        query: &str,
        docs: &[&str],
        _opts: Option<&serde_json::Map<String, Value>>,
    ) -> Result<Vec<f32>, (String, String)> {
        let loaded = self
            .sessions
            .get_mut(&handle)
            .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;

        if loaded.kind != "reranker" {
            return Err((
                "bad_kind".into(),
                format!("session {handle} is a {}, not a reranker", loaded.kind),
            ));
        }

        let max_ctx = loaded.max_context_window;
        let mut scores = Vec::with_capacity(docs.len());

        for doc in docs {
            let text = format!("{query} [SEP] {doc}");
            let encoded = loaded
                .tokenizer
                .encode_truncated(&text, true, max_ctx)
                .map_err(|e| ("tokenize_err".into(), format!("tokenization failed: {e}")))?;

            let ids: Vec<i64> = encoded.ids.iter().map(|&t| t as i64).collect();
            let mask: Vec<i64> = encoded.attention_mask.iter().map(|&m| m as i64).collect();
            let tti: Vec<i64> = encoded.type_ids.iter().map(|&t| t as i64).collect();

            let seq_len = ids.len();
            let input_ids = Array2::<i64>::from_shape_vec((1, seq_len), ids)
                .map_err(|e| ("inference_err".into(), format!("shape error: {e}")))?;
            let attention_mask = Array2::<i64>::from_shape_vec((1, seq_len), mask)
                .map_err(|e| ("inference_err".into(), format!("shape error: {e}")))?;
            let token_type_ids = Array2::<i64>::from_shape_vec((1, seq_len), tti)
                .map_err(|e| ("inference_err".into(), format!("shape error: {e}")))?;

            let input_ids_val = Tensor::from_array(input_ids)
                .map_err(|e| ("inference_err".into(), format!("tensor error: {e}")))?;
            let attention_mask_val = Tensor::from_array(attention_mask)
                .map_err(|e| ("inference_err".into(), format!("tensor error: {e}")))?;
            let token_type_ids_val = Tensor::from_array(token_type_ids)
                .map_err(|e| ("inference_err".into(), format!("tensor error: {e}")))?;

            let inputs = build_text_inputs(
                &loaded.pool[0],
                input_ids_val,
                attention_mask_val,
                token_type_ids_val,
            );

            let outputs = loaded
                .pool[0]
                .run(inputs)
                .map_err(|e| ("inference_err".into(), format!("inference failed: {e}")))?;

            let score = extract_rerank_score(&outputs)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Greedy generation over an `extractor` session (design 04 §4.4);
    /// `schema` present switches to constrained filling.
    pub fn generate(
        &mut self,
        handle: u32,
        prompt: &str,
        max_new_tokens: usize,
        stop_ids: &[u32],
        stop_strings: &[String],
        schema: Option<&serde_json::Value>,
    ) -> Result<crate::generate::GenOutput, (String, String)> {
        let loaded = self
            .sessions
            .get_mut(&handle)
            .ok_or_else(|| ("no_session".into(), format!("session {handle} not loaded")))?;
        crate::generate::generate(
            loaded,
            prompt,
            max_new_tokens,
            stop_ids,
            stop_strings,
            schema,
        )
    }

    pub fn unload(&mut self, handle: u32) -> Result<(), (String, String)> {
        if self.sessions.remove(&handle).is_some() {
            log::info!("unloaded session {handle}");
            Ok(())
        } else {
            Err(("no_session".into(), format!("session {handle} not loaded")))
        }
    }
}

/// Round a padded sequence length up to a small fixed ladder (design 15
/// §15.6), so an accelerator compiles a handful of shapes instead of one per
/// request. The ladder is coarse on purpose: a shape is a compiled artifact,
/// so few-and-wide beats fine-and-true. Never exceeds the model's context
/// window (padding past it is not representable).
fn bucket_len(len: usize, max_ctx: usize) -> usize {
    const LADDER: [usize; 5] = [128, 256, 512, 768, 1024];
    if len >= max_ctx {
        return len;
    }
    for b in LADDER {
        if len <= b && b <= max_ctx {
            return b;
        }
    }
    ((len + 127) / 128 * 128).min(max_ctx)
}

/// Build the three text-encoder input tensors for a slice of tokenized rows,
/// right-padded to `max_len` (attention mask 0 on the padding).
fn build_batch_tensors(
    tokenized: &[Vec<u32>],
    max_len: usize,
) -> Result<(Tensor<i64>, Tensor<i64>, Tensor<i64>), (String, String)> {
    let n = tokenized.len();
    let mut input_ids = Array2::<i64>::zeros((n, max_len));
    let mut attention_mask = Array2::<i64>::ones((n, max_len));
    let token_type_ids = Array2::<i64>::zeros((n, max_len));
    for (i, tokens) in tokenized.iter().enumerate() {
        for (j, &tok) in tokens.iter().enumerate() {
            input_ids[[i, j]] = tok as i64;
        }
        for j in tokens.len()..max_len {
            attention_mask[[i, j]] = 0;
        }
    }
    let wrap = |what: &str, r: ort::Result<Tensor<i64>>| {
        r.map_err(|e| {
            (
                "inference_err".into(),
                format!("failed to create {what} tensor: {e}"),
            )
        })
    };
    Ok((
        wrap("input_ids", Tensor::from_array(input_ids))?,
        wrap("attention_mask", Tensor::from_array(attention_mask))?,
        wrap("token_type_ids", Tensor::from_array(token_type_ids))?,
    ))
}

/// First output tensor of a run, as an owned array.
fn first_output_f32(outputs: &SessionOutputs) -> Result<ArrayD<f32>, (String, String)> {
    let arr = outputs
        .iter()
        .next()
        .ok_or_else(|| ("inference_err".into(), "model produced no outputs".into()))?
        .1
        .try_extract_array::<f32>()
        .map_err(|e| {
            (
                "inference_err".into(),
                format!("failed to extract tensor: {e}"),
            )
        })?
        .into_owned();
    Ok(arr)
}

/// Shape-check a run's output into the encoder's two output shapes.
fn classify_output(output: ArrayD<f32>) -> Result<EncoderOutput, (String, String)> {
    match output.shape() {
        [_n, _seq, _d] => {
            let hidden = output.into_dimensionality::<Ix3>().map_err(|e| {
                (
                    "inference_err".into(),
                    format!("unexpected tensor shape: {e}"),
                )
            })?;
            Ok(EncoderOutput::TokenStates(hidden))
        }
        [_n, _d] => {
            let pooled = output.into_dimensionality::<Ix2>().map_err(|e| {
                (
                    "inference_err".into(),
                    format!("unexpected tensor shape: {e}"),
                )
            })?;
            Ok(EncoderOutput::Pooled(pooled))
        }
        shape => Err((
            "inference_err".into(),
            format!("expected 2D/3D tensor, got shape {shape:?}"),
        )),
    }
}

/// The configured thread budget is the TOTAL across the pool: each member gets
/// an equal share (floor 1), so N sessions x (`intra` / N) threads is the same
/// silicon as one session with `intra` threads — fan-out changes the shape, it
/// never multiplies the budget (design 15 §15.7).
fn per_session_threads(intra_threads: Option<usize>, fan_out: usize) -> Option<usize> {
    if fan_out > 1 {
        intra_threads.map(|t| (t / fan_out).max(1))
    } else {
        intra_threads
    }
}

/// Auto fan-out rule: spread a batched embed over several sessions when the
/// model's weights are small enough to stay cache-resident, and keep ONE
/// session when they are not.
///
/// Measured on the 6P+12E box, batch 16 (design 15 §15.7; the A/B harness that
/// produced these numbers has since been deleted — the table there is the
/// record):
/// - `static-similarity-mrl` (108 MB, cache-resident): 1x16 = 827 ms ->
///   8x2 = 609 ms = **1.36x**. Compute-bound, so splitting the batch across
///   cores is exactly the right shape.
/// - `harrier-oss-v1-0.6b-ONNX-q4` (~600 MB, streams from unified memory):
///   390 / 404 / 393 / 389 ms for 1x16 / 2x8 / 4x4 / 8x2 = **1.00x** —
///   bandwidth-bound, and the thread split neither helps nor hurts. Its
///   remaining levers are batch size and silicon, not core count.
///
/// The threshold sits between the two measured models. It is a heuristic
/// with a fixed-count override (`embed_sessions` in the load payload) for
/// when a specific model deserves a different call.
fn auto_fan_out(model_dir: &std::path::Path) -> usize {
    const CACHE_RESIDENT_BYTES: u64 = 384 << 20;
    let bytes: u64 = std::fs::read_dir(model_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    matches!(
                        e.path().extension().and_then(|x| x.to_str()),
                        Some("onnx" | "bin" | "data")
                    )
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(u64::MAX);
    if bytes <= CACHE_RESIDENT_BYTES {
        8
    } else {
        1
    }
}

fn extract_rerank_score(outputs: &SessionOutputs) -> Result<f32, (String, String)> {
    let output = outputs
        .iter()
        .next()
        .ok_or_else(|| ("inference_err".into(), "model produced no outputs".into()))?
        .1;

    let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
        (
            "inference_err".into(),
            format!("failed to extract tensor: {e}"),
        )
    })?;

    match &**shape {
        [1, 1] | [1] | [] => Ok(data[0]),
        [1, _] => Ok(data[0]),
        _ => Err((
            "inference_err".into(),
            format!("unexpected rerank output shape {shape:?}"),
        )),
    }
}

fn mean_pool_normalize(
    hidden: &Array3<f32>,
    mask: &Array2<i64>,
    dim: usize,
    normalize: bool,
) -> Vec<Vec<f32>> {
    let (n, seq, d) = hidden.dim();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut pooled = vec![0.0f32; dim];
        let mut mask_sum = 0i64;

        for j in 0..seq {
            let m = mask[[i, j]] as f32;
            for k in 0..d {
                pooled[k] += hidden[[i, j, k]] * m;
            }
            mask_sum += mask[[i, j]];
        }

        let inv = if mask_sum > 0 {
            1.0 / mask_sum as f32
        } else {
            0.0
        };
        for v in pooled.iter_mut() {
            *v *= inv;
        }

        if normalize {
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }
        }

        result.push(pooled);
    }

    result
}

fn infer_dim(session: &Session) -> usize {
    session
        .outputs()
        .iter()
        .find(|o| o.name() == "last_hidden_state")
        .or_else(|| session.outputs().iter().next())
        .and_then(|o| o.dtype().tensor_shape())
        .and_then(|s| s.last().copied())
        .filter(|&d| d > 0)
        .map(|d| d as usize)
        .unwrap_or(0)
}

fn infer_kind(session: &Session) -> String {
    let input_names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    let output_names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
    if output_names.contains(&"logits") {
        "extractor".into()
    } else if input_names.contains(&"query") || input_names.contains(&"query_input_ids") {
        "reranker".into()
    } else {
        "embedder".into()
    }
}

/// Build the input map for a text encoder, sending only the inputs the model
/// actually declares. Qwen3-style models have no `token_type_ids` input, so we
/// must not unconditionally forward it (ORT rejects unknown input names).
fn build_text_inputs(
    session: &Session,
    input_ids: Tensor<i64>,
    attention_mask: Tensor<i64>,
    token_type_ids: Tensor<i64>,
) -> Vec<(Cow<'static, str>, SessionInputValue<'static>)> {
    let expected: std::collections::HashSet<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = Vec::new();
    inputs.push((Cow::Borrowed("input_ids"), input_ids.into()));
    inputs.push((Cow::Borrowed("attention_mask"), attention_mask.into()));
    if expected.contains("token_type_ids") {
        inputs.push((Cow::Borrowed("token_type_ids"), token_type_ids.into()));
    }
    inputs
}

#[derive(Debug, serde::Deserialize)]
struct ModelConfig {
    kind: Option<String>,
    #[serde(alias = "hidden_size")]
    dim: Option<usize>,
    #[serde(alias = "max_position_embeddings")]
    max_context_window: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_attention_heads: Option<usize>,
}

impl ModelConfig {
    fn load(path: &std::path::Path) -> Result<Self, (String, String)> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            (
                "load_err".into(),
                format!("failed to read config {path:?}: {e}"),
            )
        })?;
        serde_json::from_str(&content).map_err(|e| {
            (
                "load_err".into(),
                format!("failed to parse config {path:?}: {e}"),
            )
        })
    }
}

fn apply_exec_provider(
    builder: ort::session::builder::SessionBuilder,
    ep: &str,
) -> Result<ort::session::builder::SessionBuilder, (String, String)> {
    #[allow(unused_mut)]
    let mut dispatch: Option<ort::ep::ExecutionProviderDispatch> = None;
    #[cfg(feature = "xnnpack")]
    {
        if ep == "xnnpack" {
            dispatch = Some(ort::ep::XNNPACK::default().into());
        }
    }
    #[cfg(feature = "cuda")]
    {
        if ep == "cuda" {
            dispatch = Some(ort::ep::CUDA::default().into());
        }
    }
    #[cfg(feature = "tensorrt")]
    {
        if ep == "tensorrt" {
            dispatch = Some(ort::ep::TensorRT::default().into());
        }
    }
    // CoreML is compiled into ort on macOS by the target-scoped dependency in
    // Cargo.toml, so the arm is a target gate, not a feature gate: the
    // provider is available on every macOS build (design 15 §15.5).
    #[cfg(target_os = "macos")]
    {
        if ep == "coreml" {
            dispatch = Some(ort::ep::CoreML::default().into());
        }
    }
    #[cfg(feature = "directml")]
    {
        if ep == "directml" {
            dispatch = Some(ort::ep::DirectML::default().into());
        }
    }
    #[cfg(feature = "openvino")]
    {
        if ep == "openvino" {
            dispatch = Some(ort::ep::OpenVINO::default().into());
        }
    }
    #[cfg(feature = "rocm")]
    {
        if ep == "rocm" {
            dispatch = Some(ort::ep::ROCm::default().into());
        }
    }

    match dispatch {
        Some(d) => builder
            .with_execution_providers([d])
            .map_err(|e| ("load_err".into(), format!("failed to enable {ep} EP: {e}"))),
        None => {
            if ep == "cpu" || ep == "auto" {
                Ok(builder)
            } else {
                Err((
                    "load_err".into(),
                    format!("unknown or unsupported execution provider: {ep}"),
                ))
            }
        }
    }
}

pub fn available_eps() -> Vec<String> {
    #[allow(unused_mut)]
    let mut eps = vec!["cpu".into()];
    #[cfg(feature = "xnnpack")]
    eps.push("xnnpack".into());
    #[cfg(feature = "cuda")]
    eps.push("cuda".into());
    #[cfg(feature = "tensorrt")]
    eps.push("tensorrt".into());
    #[cfg(target_os = "macos")]
    eps.push("coreml".into());
    #[cfg(feature = "directml")]
    eps.push("directml".into());
    #[cfg(feature = "openvino")]
    eps.push("openvino".into());
    #[cfg(feature = "rocm")]
    eps.push("rocm".into());
    eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_manager_new() {
        let mgr = SessionManager::new();
        assert_eq!(mgr.next_handle, 1);
    }

    #[test]
    fn unload_nonexistent_session() {
        let mut mgr = SessionManager::new();
        let err = mgr.unload(99).unwrap_err();
        assert_eq!(err.0, "no_session");
    }

    #[test]
    fn embed_nonexistent_session() {
        let mut mgr = SessionManager::new();
        let err = mgr.embed(99, &["hello"], None).unwrap_err();
        assert_eq!(err.0, "no_session");
    }

    #[test]
    fn rerank_nonexistent_session() {
        let mut mgr = SessionManager::new();
        let err = mgr.rerank(99, "q", &["d"], None).unwrap_err();
        assert_eq!(err.0, "no_session");
    }

    #[test]
    fn model_config_missing_file() {
        let result = ModelConfig::load(std::path::Path::new("/nonexistent/config.json"));
        assert_eq!(result.unwrap_err().0, "load_err");
    }

    /// The regression gate the deleted fan-out A/B used to provide: the pool
    /// divides the thread budget, it never multiplies it (design 15 §15.7).
    #[test]
    fn fan_out_splits_the_thread_budget() {
        assert_eq!(per_session_threads(Some(16), 1), Some(16));
        assert_eq!(per_session_threads(Some(16), 2), Some(8));
        assert_eq!(per_session_threads(Some(16), 4), Some(4));
        assert_eq!(per_session_threads(Some(16), 8), Some(2));
        // floor 1: a pool wider than the budget still gets one thread each
        assert_eq!(per_session_threads(Some(3), 4), Some(1));
        assert_eq!(per_session_threads(Some(1), 8), Some(1));
        // unset budget stays unset (ORT picks its own default)
        assert_eq!(per_session_threads(None, 8), None);
        for fan_out in [1usize, 2, 4, 8] {
            let per = per_session_threads(Some(16), fan_out).unwrap();
            assert!(
                per * fan_out <= 16,
                "pool of {fan_out} must not exceed the 16-thread budget"
            );
        }
    }

    #[test]
    fn auto_fan_out_follows_the_weight_footprint() {
        let dir = std::env::temp_dir().join(format!(
            "onnxwrapperrs-fanout-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let write = |name: &str, bytes: u64| {
            let path = dir.join(name);
            let f = std::fs::File::create(&path).expect("create");
            f.set_len(bytes).expect("set_len");
        };
        // cache-resident weights (the scout's regime) fan out; a
        // bandwidth-bound footprint keeps a single session.
        write("model.onnx", 108 << 20);
        assert_eq!(auto_fan_out(&dir), 8);
        write("model.onnx", 600 << 20);
        assert_eq!(auto_fan_out(&dir), 1);
        // weights in sibling files count too (external-data .onnx is tiny)
        write("model.onnx", 1 << 20);
        write("weights.data", 500 << 20);
        assert_eq!(auto_fan_out(&dir), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mean_pool_normalize_basic() {
        let dim = 3;
        let hidden = ndarray::arr3(&[[[2.0f32, 4.0, 6.0], [0.0, 0.0, 0.0]]]);
        let mask = ndarray::arr2(&[[1i64, 0]]);
        let result = mean_pool_normalize(&hidden, &mask, dim, false);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn mean_pool_normalize_with_l2() {
        let hidden = ndarray::arr3(&[[[3.0f32, 4.0, 0.0]]]);
        let mask = ndarray::arr2(&[[1i64]]);
        let result = mean_pool_normalize(&hidden, &mask, 3, true);
        let norm: f32 = result[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_normalize_two_texts() {
        let hidden = ndarray::arr3(&[
            [[2.0f32, 0.0, 0.0], [0.0, 4.0, 0.0]],
            [[1.0f32, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]);
        let mask = ndarray::arr2(&[[1i64, 1], [1, 0]]);
        let result = mean_pool_normalize(&hidden, &mask, 3, false);
        assert_eq!(result[0], vec![1.0, 2.0, 0.0]);
        assert_eq!(result[1], vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn available_eps_contains_cpu() {
        let eps = available_eps();
        assert!(eps.contains(&"cpu".to_string()));
    }

    #[test]
    fn integration_load_embed_rerank() {
        let model_dir = match std::env::var("ONNXWRAPPERRS_TEST_MODEL") {
            Ok(d) => d,
            Err(_) => {
                eprintln!(
                    "skipping integration test: set ONNXWRAPPERS_TEST_MODEL to a dir with model.onnx + tokenizer.json + config.json"
                );
                return;
            }
        };
        let model_path = format!("{model_dir}/model.onnx");

        let mut mgr = SessionManager::new();
        let info = mgr
            .load(&model_path, "auto", "fp32", None, None, 1, false)
            .expect("load should succeed");
        assert!(info.dim > 0);
        assert!(matches!(info.kind.as_str(), "embedder" | "reranker"));

        if info.kind == "embedder" {
            let embeddings = mgr
                .embed(info.handle, &["hello world"], None)
                .expect("embed should succeed");
            assert_eq!(embeddings.len(), 1);
            assert_eq!(embeddings[0].len(), info.dim);
            let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embeddings should be L2-normalized"
            );
        } else {
            let scores = mgr
                .rerank(info.handle, "rust developer", &["cv A", "cv B"], None)
                .expect("rerank should succeed");
            assert_eq!(scores.len(), 2);
            assert!(scores.iter().all(|s| s.is_finite()));
        }

        mgr.unload(info.handle).expect("unload should succeed");
    }
}
