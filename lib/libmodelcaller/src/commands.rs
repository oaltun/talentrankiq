use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};

use crate::cli::StdioArgs;
use crate::error::{ModelError, Result};
use crate::model_registry::{
    format_document, format_query, unknown_usage_err, ModelArtifact, Registry, ResolvedModel,
};
use crate::protocol::Envelope;
use crate::wrapper::Wrapper;
use libmodelstorage::{onnx_exists, resolve_onnx, ResolveError};

pub struct State {
    model_dir: PathBuf,
    revision: String,
    exec_provider: String,
    precision: String,
    /// ONNX session thread caps from `ikapp.yaml` `engine:` — optional
    /// ceilings; unset fields resolve from the libsyscap budget.
    engine: libconfigreader::EngineSection,
    /// The `syscap:` section the host runs with (defaults when the host has
    /// none) — inputs to the thread-cap resolution below.
    syscap: libconfigreader::SyscapSection,
    /// Resolved session knobs (threads + embed pool), computed once on first
    /// model load: one fresh load sample, then the libsyscap resolvers.
    threads: std::sync::OnceLock<SessionKnobs>,
    /// Which runtime serves embeds (design 15 §15.4). `auto` takes the MLX arm
    /// for models that declare one; `ort` pins the CPU floor.
    backend: libconfigreader::Backend,
    /// One wrapper per runtime, spawned on first use. Both can be alive at
    /// once because the *models* are not: the cascade loads the scout, embeds,
    /// unloads it, and only then loads the heavy embedder, so at most one arm
    /// ever holds weights.
    wrappers: HashMap<Runtime, Wrapper>,
    loaded: HashMap<String, LoadedModel>,
    /// Global session id -> (arm, that arm's own handle).
    ///
    /// Session handles are only unique *per arm*: each wrapper numbers from 1.
    /// modelcaller hands callers a global id and rewrites it to the arm's own
    /// handle on the way out. Keying a single global `session -> runtime` map
    /// instead is a silent-corruption bug: with the scout on one arm and the
    /// heavy embedder on the other, both can hold handle 2, and the heavy
    /// embedder's embed gets routed to the scout's session — which is how a
    /// 640-dim model came back 1024-dim.
    sessions: HashMap<u64, (Runtime, u64)>,
    /// Next global session id. Distinct from every arm's own counter.
    next_session: u64,
    verbose: bool,
    auto_download: bool,
    registry: Registry,
}

struct LoadedModel {
    session: u64,
    model: ResolvedModel,
    runtime: Runtime,
}

/// Parse the `engine.backend` wire value. An unrecognised name is `auto`,
/// which is the safe direction: the ladder then serves a model on the arm its
/// config actually declares. (`libconfigreader` rejects unknown names when it
/// reads the file, so this only sees values a host passed programmatically.)
fn parse_backend(s: &str) -> libconfigreader::Backend {
    match s {
        "ort" => libconfigreader::Backend::Ort,
        "mlx" => libconfigreader::Backend::Mlx,
        _ => libconfigreader::Backend::Auto,
    }
}

/// Which arm served a model. Ordinal order is the preference order, so
/// `Runtime::default()` is the floor every build falls back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Runtime {
    /// ONNX Runtime — every platform, every build.
    Ort,
    /// MLX on Metal — macOS, and only when `libmlxembed` was built with
    /// `--features mlx`.
    Mlx,
}

impl Runtime {
    fn as_str(&self) -> &'static str {
        match self {
            Runtime::Ort => "ort",
            Runtime::Mlx => "mlx",
        }
    }
}

/// Run the stdio JSONL loop over arbitrary streams — the library core. Hosts
/// pass a pipe/duplex half: triqui-core and libmodelcallerclient run this
/// on a dedicated thread inside their process (thin gui pattern,
/// `gui/design/app1/thin-gui.md`).
///
/// `logfile` is accepted for API compatibility with the old spawned-binary
/// protocol but is ignored here: in-process hosts own logging (there is no
/// logger to redirect — the loop only emits the JSONL stream).
pub fn run_stdio_over(
    registry: Registry,
    args: StdioArgs,
    engine: libconfigreader::EngineSection,
    syscap: libconfigreader::SyscapSection,
    verbose: bool,
    _logfile: Option<String>,
    input: impl BufRead,
    output: impl Write + Send + 'static,
) -> Result<()> {
    let state = State::new(registry, args, engine, syscap, verbose);
    stdio_loop(state, input, output)
}

impl State {
    pub fn new(
        registry: Registry,
        args: StdioArgs,
        engine: libconfigreader::EngineSection,
        syscap: libconfigreader::SyscapSection,
        verbose: bool,
    ) -> Self {
        State {
            model_dir: resolve_model_dir(args.model_dir.as_deref().unwrap_or(Path::new(""))),
            revision: args.revision.clone(),
            // The effective engine section wins over the wire args: hosts
            // pass the section they clamped (bench pins `backend: ort` for
            // result comparability), and sourcing these from the raw args
            // instead let a config's `backend: auto` override the pin —
            // every "ort-pinned" bench embed of a model with an `mlx:`
            // variant ran on Metal. Args stay the fallback for a section
            // that leaves a field unset.
            exec_provider: engine
                .exec_provider
                .as_ref()
                .map(|p| p.as_str().to_string())
                .unwrap_or_else(|| args.exec_provider.clone()),
            precision: args.precision.clone(),
            backend: engine
                .backend
                .unwrap_or_else(|| parse_backend(&args.backend)),
            engine,
            syscap,
            threads: std::sync::OnceLock::new(),
            wrappers: HashMap::new(),
            loaded: HashMap::new(),
            sessions: HashMap::new(),
            next_session: 1,
            verbose,
            auto_download: args.auto_download_models,
            registry,
        }
    }

    /// The ONNX arm. Callers that can name the runtime should use
    /// [`State::wrapper_for`]; this stays for the commands that are ONNX-only
    /// by definition (rerank, schema discovery) or not model-addressed
    /// (probe, shutdown).
    fn ensure_wrapper(&mut self) -> Result<&mut Wrapper> {
        self.wrapper_for(Runtime::Ort)
    }

    fn wrapper_for(&mut self, runtime: Runtime) -> Result<&mut Wrapper> {
        if !self.wrappers.contains_key(&runtime) {
            log::debug!(
                "ensure_wrapper: spawning in-process {} wrapper",
                runtime.as_str()
            );
            let w = Wrapper::spawn(runtime)?;
            log::debug!("ensure_wrapper: {} wrapper spawned", runtime.as_str());
            self.wrappers.insert(runtime, w);
        }
        Ok(self.wrappers.get_mut(&runtime).expect("just inserted"))
    }

    /// Hand the caller a globally unique session id for an arm's own handle.
    fn alloc_session(&mut self, runtime: Runtime, arm_handle: u64) -> u64 {
        let id = self.next_session;
        self.next_session += 1;
        self.sessions.insert(id, (runtime, arm_handle));
        id
    }

    /// The arm and the handle to put on the wire for a global session id.
    fn arm_session(&self, id: u64) -> Result<(Runtime, u64)> {
        self.sessions
            .get(&id)
            .copied()
            .ok_or_else(|| ModelError::bad_req(format!("unknown session {id}")))
    }

    /// Which runtime serves `record`, and why.
    ///
    /// The order of the checks is the contract: a model without an MLX
    /// artifact stays on ONNX even on a Mac (that is the scout, and no port
    /// changes it), a build without the arm stays on ONNX, and `backend: mlx`
    /// refuses to serve anything rather than quietly handing back CPU
    /// vectors labelled as Metal.
    fn resolve_runtime(&self, record: &ResolvedModel) -> Result<Runtime> {
        resolve_runtime_setting(self.backend, record)
    }

    /// Effective session knobs: the configured `engine:` caps, or — for unset
    /// fields — the libsyscap budget's derivation (capacity at Full, halved
    /// under Polite, floored at 1 under Backoff/Hold; inter defaults to 1, a
    /// model-shape choice). Computed once per process on first load; caps are
    /// ALWAYS concrete in the payload so an unset config line can never fall
    /// through to ONNX Runtime's own default (one worker per physical core),
    /// which would bypass the syscap policy and pin the whole machine during
    /// embeds.
    ///
    /// `embed_sessions` is the one knob passed through as a *decision* rather
    /// than a number: 0 means "the wrapper picks by the model's weight
    /// footprint" (design 15 §15.7 — a cache-resident model is compute-bound
    /// and scales across sessions, a bandwidth-bound one does not), a positive
    /// count pins the pool.
    ///
    /// `provider_logging` raises ORT's logging on load so node placement is
    /// visible — the only way to distinguish a provider that worked from one
    /// that silently fell back to CPU (design 15 §15.4.1).
    fn resolved_session_knobs(&self) -> SessionKnobs {
        *self.threads.get_or_init(|| {
            let mut sampler = libsyscap::LoadSampler::new();
            let load = sampler.sample(Duration::from_millis(self.syscap.sample_min_interval_ms()));
            let b = libsyscap::budget(
                libsyscap::Inventory::global(),
                &load,
                &self.syscap,
                &self.engine,
            );
            SessionKnobs {
                intra_threads: libsyscap::effective_intra_threads(&self.engine, &b),
                inter_threads: libsyscap::effective_inter_threads(&self.engine, &b),
                embed_sessions: match self.engine.embed_sessions {
                    Some(libconfigreader::EmbedSessions::Fixed(n)) => n.min(64) as usize,
                    _ => 0,
                },
                provider_logging: self.engine.provider_logging.unwrap_or(false),
            }
        })
    }

    /// Wrapper `load` payload for `onnx`, including the effective thread
    /// caps (see [`State::resolved_session_knobs`]).
    fn wrapper_load_payload(&self, artifact: &Path, runtime: Runtime, max_ctx: usize) -> Value {
        let knobs = self.resolved_session_knobs();
        json!({
            "cmd": "load",
            "model_path": artifact.to_string_lossy(),
            "runtime": runtime.as_str(),
            // Both arms truncate with this. A disagreement is invisible in the
            // vectors and reads as a broken conversion (design 15 §15.5.3.3).
            "max_ctx": max_ctx,
            "exec_provider": self.exec_provider,
            "precision": self.precision,
            "intra_threads": knobs.intra_threads,
            "inter_threads": knobs.inter_threads,
            "embed_sessions": knobs.embed_sessions,
            "provider_logging": knobs.provider_logging,
        })
    }
}

/// Resolved ONNX session knobs, cached once per process (see
/// [`State::resolved_session_knobs`]).
#[derive(Debug, Clone, Copy)]
struct SessionKnobs {
    intra_threads: usize,
    inter_threads: usize,
    /// 0 = let the wrapper decide by model footprint.
    embed_sessions: usize,
    /// Raise ORT logging so node placement is visible on load.
    provider_logging: bool,
}

fn resolve_model_dir(cli_dir: &Path) -> PathBuf {
    libmodelstorage::resolve_model_dir(if cli_dir.as_os_str().is_empty() {
        None
    } else {
        Some(cli_dir)
    })
}

struct ResponseCtx<W: Write> {
    out: W,
}

impl<W: Write> ResponseCtx<W> {
    fn new(out: W) -> Self {
        ResponseCtx { out }
    }
    fn write(&mut self, env: &Envelope) {
        if let Ok(line) = env.to_line() {
            let _ = self.out.write_all(line.as_bytes());
            let _ = self.out.write_all(b"\n");
            let _ = self.out.flush();
        }
    }
}

fn stdio_loop(mut state: State, input: impl BufRead, output: impl Write) -> Result<()> {
    let mut ctx = ResponseCtx::new(output);

    log::info!("modelcaller 0.1.0 starting (stdio mode)");
    log::debug!("model_dir: {}", state.model_dir.display());
    log::debug!("revision: {}", state.revision);
    log::debug!("exec_provider: {}", state.exec_provider);
    log::debug!("precision: {}", state.precision);
    log::debug!("verbose: {}", state.verbose);
    log::debug!("auto_download: {}", state.auto_download);

    let mut lines = input.lines();
    while let Some(Ok(line)) = lines.next() {
        let preview = match line.get(..200) {
            Some(prefix) if line.len() > 200 => format!("{}... ({} bytes)", prefix, line.len()),
            _ => line.clone(),
        };
        log::debug!("stdin_loop: received line: {}", preview);
        log::debug!("stdin_loop: parsing JSON line...");
        let request = match Envelope::parse_line(&line) {
            Ok(e) => e,
            Err(e) => {
                log::debug!("stdin_loop: parse error: {}", e);
                ctx.write(&Envelope::error(None, "bad_req", e));
                log::debug!("stdin_loop: writing error response");
                continue;
            }
        };
        log::debug!(
            "stdin_loop: parsed successfully, id={}, type={}",
            request.id.clone().unwrap_or_default(),
            request.r#type
        );
        let req_id = request.id.clone().unwrap_or_default();
        if request.r#type != "request" {
            log::debug!(
                "stdin_loop: expected type 'request', got '{}', sending error",
                request.r#type
            );
            ctx.write(&Envelope::error(
                Some(&req_id),
                "bad_req",
                format!("expected type 'request', got '{}'", request.r#type),
            ));
            continue;
        }
        log::debug!("stdin_loop: dispatching to handle_request");
        let result = handle_request(&mut state, &request.payload);
        log::debug!("stdin_loop: handle_request returned");
        let (env, is_shutdown) = match result {
            Ok(payload) => {
                let shutdown = payload.get("shutdown_ok").is_some();
                (Envelope::response(&req_id, payload), shutdown)
            }
            Err(e) => {
                log::error!("{}", e);
                (Envelope::error(request.id.as_deref(), e.code, e.msg), false)
            }
        };
        log::debug!("stdin_loop: writing response to stdout");
        ctx.write(&env);
        log::debug!("stdin_loop: response written");
        if is_shutdown {
            break;
        }
    }

    for (_, w) in state.wrappers.iter_mut() {
        log::debug!("stdin_loop: killing wrapper thread");
        w.kill();
        // Drain before returning: the loop may end while the wrapper thread
        // is still inside a request (a session load takes seconds), and a
        // host process that exits then races ONNX Runtime's global teardown
        // (observed as a SIGSEGV inside CreateSession). Bounded so a wedged
        // wrapper delays shutdown without ever hanging it.
        if w.wait_exited(Duration::from_secs(120)) {
            log::debug!("stdin_loop: wrapper thread exited");
        } else {
            log::warn!("stdin_loop: wrapper thread still running after 120s, giving up");
        }
    }
    log::info!("modelcaller shutting down");
    Ok(())
}

fn handle_request(state: &mut State, payload: &Value) -> Result<Value> {
    let cmd = payload
        .get("cmd")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("missing 'cmd' field"))?;

    log::debug!("handle_request: cmd = {}", cmd);

    match cmd {
        "ping" => {
            log::debug!("handle_request: ping -> ok");
            Ok(json!({ "ok": true, "version": "0.1.0" }))
        }
        "shutdown" => {
            log::debug!("handle_request: shutdown -> ensuring wrapper");
            let _ = state.ensure_wrapper().map(|w| {
                log::debug!("handle_request: sending shutdown to wrapper");
                let _ = w.send(json!({ "cmd": "shutdown" }));
            });
            Ok(json!({ "ok": true, "shutdown_ok": true }))
        }
        "probe" => {
            log::debug!("handle_request: probe -> ensure_wrapper");
            let w = state.ensure_wrapper()?;
            log::debug!("handle_request: probe -> sending to wrapper");
            Ok(w.send(json!({ "cmd": "probe" }))?)
        }
        "load" => {
            log::debug!("handle_request: load -> cmd_load");
            cmd_load(state, payload)
        }
        "unload" => {
            log::debug!("handle_request: unload -> cmd_unload");
            cmd_unload(state, payload)
        }
        "embed" => {
            log::debug!("handle_request: embed -> cmd_embed");
            cmd_embed(state, payload)
        }
        "rerank" => {
            log::debug!("handle_request: rerank -> cmd_rerank");
            cmd_rerank(state, payload)
        }
        "discover_schema" => {
            log::debug!("handle_request: discover_schema -> cmd_discover_schema");
            cmd_discover_schema(state, payload)
        }
        "extract_constrained" => {
            log::debug!("handle_request: extract_constrained -> cmd_extract_constrained");
            cmd_extract_constrained(state, payload)
        }
        "models" => {
            log::debug!("handle_request: models -> cmd_models");
            cmd_models(state, payload)
        }
        "list" => {
            log::debug!("handle_request: list -> cmd_list");
            cmd_list(state, payload)
        }
        "recommend" => {
            log::debug!("handle_request: recommend -> cmd_recommend");
            cmd_recommend(state, payload)
        }
        _ => {
            log::debug!(
                "handle_request: unknown cmd '{}', forwarding to wrapper",
                cmd
            );
            let w = state.ensure_wrapper()?;
            Ok(w.send(payload.clone())?)
        }
    }
}

fn resolve_model(state: &mut State, model: &str) -> Result<ResolvedModel> {
    let record = state
        .registry
        .find(model)
        .ok_or_else(|| ModelError::no_model(model))?;

    let known_usages = [
        "plain",
        "nomic-prefix",
        "bge-prefix",
        "e5-prefix",
        "harrier-instruct",
        "colbert-prefix",
        "plain-pair",
        "extraction-json",
    ];
    if !record.usage.is_empty() && !known_usages.contains(&record.usage.as_str()) {
        return Err(ModelError::bad_req(unknown_usage_err(&record.usage)));
    }

    if !model_onnx_path_exists(state, model) {
        if state.auto_download {
            auto_download(state, &record)?;
        } else {
            return Err(ModelError::no_model(model));
        }
    }

    Ok(record)
}

fn model_onnx_path_exists(state: &State, model: &str) -> bool {
    onnx_exists(&state.model_dir, model, &state.revision)
}

/// Where a model's weights live for `runtime`: the ONNX graph under
/// `<slug>/repo/`, or the Metal artifact under `<mlx-slug>/<revision>/`.
fn model_artifact_path(state: &State, record: &ResolvedModel, runtime: Runtime) -> Result<PathBuf> {
    match runtime {
        Runtime::Ort => {
            // Find the ONNX artifact (prefer q4 if multiple)
            let _art = record
                .artifacts
                .iter()
                .find(|a| a.backend == libconfigreader::Backend::Ort)
                .ok_or_else(|| ModelError::load_err(format!("{} has no ONNX artifact", record.slug)))?;
            model_onnx_path(state, &record.slug)
        }
        Runtime::Mlx => {
            let variant = record
                .artifacts
                .iter()
                .find(|a| a.backend == libconfigreader::Backend::Mlx)
                .ok_or_else(|| ModelError::load_err(format!("{} has no mlx variant", record.slug)))?;
            let dir = libmodelstorage::slug_dir(&state.model_dir, &record.slug).join(&state.revision);
            if !libmodelstorage::local_size(&dir.join("model.safetensors")).is_some() {
                if state.auto_download {
                    auto_download_mlx(state, &record.slug, variant)?;
                } else {
                    return Err(ModelError::no_model(&record.slug));
                }
            }
            Ok(dir)
        }
    }
}

/// The arm decision, independent of any live [`State`]: `setting` + the
/// model's registry record + whether this build has the MLX arm. The order of
/// the checks is the contract (see [`State::resolve_runtime`]).
fn resolve_runtime_setting(
    setting: libconfigreader::Backend,
    record: &ResolvedModel,
) -> Result<Runtime> {
    if !setting.allows_mlx() {
        return Ok(Runtime::Ort);
    }
    let has_mlx = record
        .artifacts
        .iter()
        .any(|a| a.backend == libconfigreader::Backend::Mlx);
    if !has_mlx {
        log::debug!(
            "backend: {} has no mlx variant declared; serving it on ort",
            record.slug
        );
        return Ok(Runtime::Ort);
    }
    if !libmlxembed::AVAILABLE {
        if setting.requires_mlx() {
            return Err(ModelError::load_err(format!(
                "engine.backend is `mlx` but this build has no MLX arm \
                 (rebuild with --features mlx on macOS); refusing to serve \
                 '{}' on the CPU and call it accelerated",
                record.slug
            )));
        }
        log::info!(
            "backend: this build has no MLX arm; serving '{}' on ort",
            record.slug
        );
        return Ok(Runtime::Ort);
    }
    Ok(Runtime::Mlx)
}

/// Which arm would serve `slug`, and whether its weights are on disk —
/// WITHOUT loading anything or starting the wrapper.
///
/// The gui consults this before engine use: the effective variant is what a
/// consent-to-download prompt must name, and it is what makes a backend
/// switch observable before any vector is read (`triqui-core` clears the
/// slug's stale vectors when the recorded arm differs — same-machine
/// different-arm vectors are NOT interchangeable: the 270m MLX/ONNX cosine
/// agreement is 0.94, not 1.0).
pub struct VariantStatus {
    /// `"ort"` or `"mlx"` — the arm `embed` would be served by.
    pub runtime: &'static str,
    /// The arm's weights exist under `model_dir` (ONNX graph, or the MLX
    /// variant's safetensors).
    pub present: bool,
}

pub fn variant_status(
    registry: &Registry,
    backend: libconfigreader::Backend,
    model_dir: &Path,
    revision: &str,
    slug: &str,
) -> Result<VariantStatus> {
    let record = registry
        .find(slug)
        .ok_or_else(|| ModelError::no_model(slug))?;
    let runtime = resolve_runtime_setting(backend, &record)?;
    let present = match runtime {
        Runtime::Ort => onnx_exists(model_dir, slug, revision),
        Runtime::Mlx => {
            let _variant = record
                .artifacts
                .iter()
                .find(|a| a.backend == libconfigreader::Backend::Mlx)
                .ok_or_else(|| ModelError::no_model(slug))?;
            let dir = libmodelstorage::slug_dir(model_dir, slug).join(revision);
            libmodelstorage::local_size(&dir.join("model.safetensors")).is_some()
        }
    };
    Ok(VariantStatus {
        runtime: runtime.as_str(),
        present,
    })
}

/// Fetch the Metal artifact. Repo mode with `Artifact::Mlx`, not URL mode:
/// an MLX checkpoint is a safetensors file (or shards) plus a config and a
/// tokenizer, planned from the repo listing rather than named by a url.
fn auto_download_mlx(state: &State, model_slug: &str, variant: &ModelArtifact) -> Result<()> {
    log::info!(
        "auto-downloading MLX artifact '{}' from {}",
        variant.quantization,
        variant.source
    );
    let cli = libmodeldownloader::cli::Cli {
        config: None,
        model: Some(model_slug.to_string()),
        source: Some(variant.source.clone()),
        url: None,
        slug: None,
        revision: state.revision.clone(),
        model_dir: Some(state.model_dir.clone()),
        force: false,
        dry_run: false,
        files: vec![],
        extra_files: variant.extra_files.clone(),
        artifact: libmodeldownloader::cli::Artifact::Mlx,
        verbose: state.verbose,
        logfile: None,
    };
    libmodeldownloader::Downloader::run(cli).map_err(|e| {
        ModelError::load_err(format!(
            "modeldownloader failed for mlx artifact '{}': {e}",
            variant.quantization
        ))
    })?;
    log::info!("auto-download complete for '{}'", model_slug);
    Ok(())
}

fn auto_download(state: &State, record: &ResolvedModel) -> Result<()> {
    log::info!(
        "auto-downloading model '{}' via modeldownloader",
        record.slug
    );
    // URL mode (`download_url`), NOT hf-source mode: hf-source fetches the
    // repo-root DEFAULT_FILES (fp32 graph shell, wrong variant, megabytes of
    // `_data` shards it cannot know about), while URL mode downloads the
    // registry's curated ONNX variant plus its external-weight `_data`
    // siblings into `<slug>/repo/` — the layout `resolve_onnx` prefers.
    // In-process library call: the gui binary needs no modeldownloader
    // executable (thin gui pattern, `gui/design/app1/thin-gui.md`).
    let onnx_art = record
        .artifacts
        .iter()
        .find(|a| a.backend == libconfigreader::Backend::Ort)
        .ok_or_else(|| ModelError::load_err(format!("{} has no ONNX artifact", record.slug)))?;
    let mut req = libmodeldownloader::UrlDownload::new(&onnx_art.url, &record.slug, state.model_dir.clone());
    for extra in &onnx_art.extra_files {
        req.extra_files.push(extra.clone());
    }
    libmodeldownloader::download_url(req, |ev| match ev {
        libmodeldownloader::DownloadEvent::Started { file, total } => {
            log::info!(
                "auto-download: fetching {file} ({} bytes)",
                fmt_total(total)
            );
        }
        libmodeldownloader::DownloadEvent::Finished { file, .. } => {
            log::info!("auto-download: {file} done");
        }
        libmodeldownloader::DownloadEvent::Cached { file, bytes } => {
            log::info!("auto-download: {file} cached ({bytes} bytes)");
        }
        libmodeldownloader::DownloadEvent::Progress { .. } => {}
    })
    .map_err(|e| {
        ModelError::load_err(format!(
            "modeldownloader failed for model '{}': {e}",
            record.slug
        ))
    })?;

    log::info!("auto-download complete for '{}'", record.slug);
    Ok(())
}

fn fmt_total(total: Option<u64>) -> String {
    match total {
        Some(n) => n.to_string(),
        None => "unknown size".to_string(),
    }
}

fn model_onnx_path(state: &State, model: &str) -> Result<PathBuf> {
    log::debug!("model_onnx_path: looking up model '{}'", model);
    match resolve_onnx(&state.model_dir, model, &state.revision) {
        Ok(res) => {
            log::debug!(
                "model_onnx_path: found onnx at {} ({:?})",
                res.path().display(),
                res
            );
            Ok(res.path().to_path_buf())
        }
        Err(ResolveError::NotFound) => {
            log::debug!("model_onnx_path: no model found");
            Err(ModelError::no_model(model))
        }
        Err(ResolveError::Ambiguous(msg)) => {
            log::debug!("model_onnx_path: ambiguous: {}", msg);
            Err(ModelError::ambiguous_onnx(&msg))
        }
        Err(ResolveError::Io(msg)) => {
            log::debug!("model_onnx_path: io error: {}", msg);
            Err(ModelError::load_err(msg))
        }
    }
}

fn load_into(state: &mut State, model: &str) -> Result<u64> {
    if let Some(lm) = state.loaded.get(model) {
        log::debug!(
            "load_into: model '{}' already loaded (session={})",
            model,
            lm.session
        );
        return Ok(lm.session);
    }
    log::debug!("load_into: starting to load model {}", model);
    log::info!("starting loading model {}", model);

    let record = resolve_model(state, model)?;
    let runtime = state.resolve_runtime(&record)?;
    let artifact = model_artifact_path(state, &record, runtime)?;
    log::info!(
        "loading '{}' on the {} arm ({})",
        record.slug,
        runtime.as_str(),
        artifact.display()
    );

    let payload =
        state.wrapper_load_payload(&artifact, runtime, record.max_context_window as usize);
    let resp = {
        let w = state.wrapper_for(runtime)?;
        log::debug!(
            "load_into: sending load request to the {} wrapper",
            runtime.as_str()
        );
        w.send(payload)?
    };

    let session = resp
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ModelError::load_err("load response missing session"))?;
    // The arm that actually served it, from its own report — not from what we
    // asked for. That is the field the gui shows and the gate that stops a
    // silent fallback from reading as acceleration (design 15 §15.4.1).
    let served = resp
        .get("runtime")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "mlx" => Some(Runtime::Mlx),
            _ => Some(Runtime::Ort),
        })
        .unwrap_or(runtime);
    if served != runtime {
        log::warn!(
            "arm mismatch: asked {} for '{}', it reported {}",
            runtime.as_str(),
            record.slug,
            served.as_str()
        );
    }
    log::debug!(
        "load_into: model '{}' loaded with session {}",
        model,
        session
    );
    log::info!("ended loading model {}", model);

    let global = state.alloc_session(served, session);
    state.loaded.insert(
        model.to_string(),
        LoadedModel {
            session: global,
            model: record,
            runtime: served,
        },
    );
    Ok(global)
}

fn cmd_load(state: &mut State, payload: &Value) -> Result<Value> {
    let model = require_model(payload)?;
    let record = resolve_model(state, model)?;
    let runtime = state.resolve_runtime(&record)?;
    let artifact = model_artifact_path(state, &record, runtime)?;
    let load = state.wrapper_load_payload(&artifact, runtime, record.max_context_window as usize);
    let resp = {
        let w = state.wrapper_for(runtime)?;
        w.send(load)?
    };
    let session = resp
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ModelError::load_err("load response missing session"))?;
    let global = state.alloc_session(runtime, session);
    state.loaded.insert(
        model.to_string(),
        LoadedModel {
            session: global,
            model: record,
            runtime,
        },
    );
    Ok(resp)
}

fn cmd_unload(state: &mut State, payload: &Value) -> Result<Value> {
    let model = require_model(payload)?;
    // Unload from the arm that loaded it — the cascade's scout/heavy handoff
    // depends on this: the scout's weights must leave the GPU before the heavy
    // embedder's arrive, or both are resident at once.
    let loaded = state.loaded.get(model).map(|lm| (lm.runtime, lm.session));
    let (runtime, global) = match loaded {
        Some(rt_and_session) => rt_and_session,
        // Unloading something this process never loaded: still ask the ONNX
        // arm (the arm that owns the ONNX cache layout) and let it answer.
        None => (Runtime::Ort, 0),
    };
    let arm_handle = if global == 0 {
        0
    } else {
        state.arm_session(global)?.1
    };
    let resp = {
        let w = state.wrapper_for(runtime)?;
        let mut req = json!({ "cmd": "unload", "model": model });
        if arm_handle != 0 {
            req["session"] = json!(arm_handle);
        }
        w.send(req)?
    };
    if let Some((_, global)) = loaded {
        state.sessions.remove(&global);
    }
    state.loaded.remove(model);
    Ok(resp)
}

fn cmd_embed(state: &mut State, payload: &Value) -> Result<Value> {
    log::debug!("cmd_embed: extracting model and texts from payload");
    let model = payload.get("model").and_then(|v| v.as_str());
    log::debug!("cmd_embed: model specified: {:?}", model);

    let model_str = model.ok_or_else(|| ModelError::bad_req("missing 'model' field"))?;
    let record = resolve_model(state, model_str)?;
    ensure_kind(&record, &["embedder", "reranker"], "embed")?;
    log::debug!(
        "cmd_embed: resolved model='{}', dim={}, usage='{}'",
        record.slug,
        record.dim,
        record.usage
    );
    let session = load_into(state, &record.slug)?;

    let raw_texts: Vec<String> = payload
        .get("texts")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let texts_role: Vec<String> = payload
        .get("texts_role")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let formatted: Vec<String> = if texts_role.len() == raw_texts.len() {
        raw_texts
            .iter()
            .zip(texts_role.iter())
            .map(|(t, role)| match role.as_str() {
                "query" => format_query(t, &record.usage),
                _ => format_document(t, &record.usage),
            })
            .collect()
    } else {
        raw_texts
    };

    log::debug!("cmd_embed: {} texts to embed", formatted.len());
    if formatted.is_empty() {
        let mut empty = json!({ "model": record.slug, "dim": record.dim, "usage": record.usage, "embeddings": [] });
        if record.multi_vector {
            empty["multi_vector"] = json!(true);
            empty["embeddings_b64"] = json!([]);
            empty["counts"] = json!([]);
        }
        return Ok(empty);
    }
    let mut opts = payload.get("opts").cloned().unwrap_or_else(|| json!({}));
    if opts.get("max_context_window").is_none() {
        opts["max_context_window"] = json!(record.max_context_window);
    }
    if record.multi_vector {
        // late-interaction model: wrapper skips pooling, applies the
        // projection chain from the model dir, returns token vectors.
        opts["multi_vector"] = json!(true);
        let onnx_art = record
            .artifacts
            .iter()
            .find(|a| a.backend == libconfigreader::Backend::Ort)
            .ok_or_else(|| ModelError::load_err(format!("{} has no ONNX artifact", record.slug)))?;
        opts["projection_files"] = json!(onnx_art.extra_files);
    }

    let (runtime, arm_handle) = state.arm_session(session)?;
    log::debug!("cmd_embed: sending embed to the {} arm", runtime.as_str());
    let resp = {
        let w = state.wrapper_for(runtime)?;
        log::debug!(
            "cmd_embed: sending embed request to wrapper: session={}, {} texts",
            arm_handle,
            formatted.len()
        );
        w.send(json!({
            "cmd": "embed",
            "session": arm_handle,
            "texts": formatted,
            "opts": opts,
        }))?
    };
    log::debug!("cmd_embed: received embed response from wrapper");
    let mut resp = resp;
    resp["model"] = json!(record.slug);
    resp["dim"] = json!(record.dim);
    resp["usage"] = json!(record.usage);
    Ok(resp)
}

fn cmd_rerank(state: &mut State, payload: &Value) -> Result<Value> {
    let model = payload.get("model").and_then(|v| v.as_str());
    let model_str = model.ok_or_else(|| ModelError::bad_req("rerank requires 'model'"))?;
    let record = resolve_model(state, model_str)?;
    ensure_kind(&record, &["reranker"], "rerank")?;
    let session = load_into(state, &record.slug)?;

    let query = payload
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("rerank requires 'query'"))?;
    let docs: Vec<String> = payload
        .get("docs")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let mut opts = payload.get("opts").cloned().unwrap_or_else(|| json!({}));
    if opts.get("max_context_window").is_none() {
        opts["max_context_window"] = json!(record.max_context_window);
    }
    let resp = {
        let w = state.ensure_wrapper()?;
        w.send(json!({
            "cmd": "rerank",
            "session": session,
            "query": query,
            "docs": docs,
            "opts": opts,
        }))?
    };
    let mut resp = resp;
    adapt_wrapper_rerank_response(&mut resp);
    resp["model"] = json!(record.slug);
    Ok(resp)
}

/// The wrapper's internal rerank protocol returns `{ "scores": [f32] }`.
/// Clients (libmodelcallerclient) own the public rerank contract:
/// `{ "results": [ { "index": usize, "relevance_score": f64 } ] }`.
/// Adapt in place; no-op if the response already carries `results`.
fn adapt_wrapper_rerank_response(resp: &mut Value) {
    if resp.get("results").is_some() {
        return;
    }
    let Some(scores) = resp.get("scores").and_then(|v| v.as_array()) else {
        return;
    };
    let results: Vec<Value> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| {
            json!({
                "index": i,
                "relevance_score": s.as_f64().unwrap_or(0.0),
            })
        })
        .collect();
    resp["results"] = json!(results);
}

// ---------------------------------------------------------------------------
// llm cv-evidence extraction (modelcaller design 14 / design 18 m3a)
// ---------------------------------------------------------------------------

/// qwen2.5 `<|im_end|>` — stops decode on the assistant turn's end marker
const IM_END_ID: u32 = 151645;
// 320: the 1.5b model writes verbose descriptions; a 6-field schema with
// the tightened system prompt fits well under this, overflow is clamped/errored
const DISCOVER_MAX_NEW_TOKENS: usize = 320;
const EXTRACT_MAX_NEW_TOKENS: usize = 400;
/// conservative char budgets: ~3.5 chars/token for fr/en text keeps the
/// rendered prompt inside the 4096-token window with room for the answer
const MAX_AD_CHARS: usize = 8_000;
const MAX_CV_CHARS: usize = 12_000;

fn ensure_kind(record: &ResolvedModel, allowed: &[&str], cmd: &str) -> Result<()> {
    if allowed.contains(&record.kind.as_str()) {
        return Ok(());
    }
    Err(ModelError::bad_req(format!(
        "{} requires kind {} but model '{}' is kind '{}'",
        cmd,
        allowed.join("|"),
        record.slug,
        record.kind
    )))
}

fn qwen_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

const DISCOVER_SYSTEM: &str = "You design json schemas for cv screening. Given a job ad, output the list of cv facts worth extracting as one json object with a single key \"fields\". At most 6 fields; stop after the closing brace. Each field is an object {\"name\": snake_case identifier, \"type\": one of numeric|string|bool|list_string, \"description\": short description under 8 words, \"required\": bool, \"enum\": [only for string fields]}. Output only json, no markdown.";

const DISCOVER_FEWSHOT: &str = r#"Job ad:
We are looking for a senior java developer to lead our payments team. Requirements: 5+ years of professional java experience, spring ecosystem, kafka, experience with microservice architecture at scale, mentoring junior developers.
Schema:
{"fields":[{"name":"years_professional_java","type":"numeric","description":"years building java services professionally","required":true},{"name":"architecture_evidence","type":"list_string","description":"architecture-level work items (migration, scaling, design)","required":false},{"name":"leadership_evidence","type":"list_string","description":"mentoring or leading activities","required":false},{"name":"spring_level","type":"string","description":"depth with the spring ecosystem","required":false,"enum":["basic","advanced","expert"]}]}
Job ad:
Data analyst position. You will build dashboards and run analyses on product data. We require sql fluency, python for analysis, and experience with a BI tool such as tableau or power bi. A degree in a quantitative field is required.
Schema:
{"fields":[{"name":"years_sql","type":"numeric","description":"years of professional sql use","required":true},{"name":"bi_tools_used","type":"list_string","description":"BI or dashboard tools used","required":false},{"name":"python_analysis","type":"bool","description":"uses python for data analysis","required":false},{"name":"quantitative_degree","type":"bool","description":"holds a degree in a quantitative field","required":true}]}
"#;

const EXTRACT_SYSTEM: &str = "You fill json forms from cv text. Output only the json object with the requested fields; use zero / empty / false when the cv gives no evidence. No markdown.";

fn render_schema_form(schema: &libcvevidence::ExtractionSchema) -> String {
    let mut lines = Vec::new();
    for f in &schema.fields {
        let mut line = format!(
            "- {} ({}): {}",
            f.name,
            f.kind.as_str(),
            f.description.as_deref().unwrap_or("-")
        );
        if let Some(vals) = &f.enum_values {
            line.push_str(&format!(" [one of: {}]", vals.join(", ")));
        }
        lines.push(line);
    }
    lines.join("\n")
}

fn cmd_discover_schema(state: &mut State, payload: &Value) -> Result<Value> {
    let model = require_model(payload)?;
    let record = resolve_model(state, model)?;
    ensure_kind(
        &record,
        &["extractor", "schema_discoverer"],
        "discover_schema",
    )?;
    let ad_text = payload
        .get("ad_text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("discover_schema requires 'ad_text'"))?;
    let max_new_tokens = payload
        .get("opts")
        .and_then(|o| o.get("max_new_tokens"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(DISCOVER_MAX_NEW_TOKENS);
    let session = load_into(state, &record.slug)?;

    let user = format!(
        "{DISCOVER_FEWSHOT}\nJob ad:\n{}\n\nOutput the json schema now.",
        truncate_chars(ad_text, MAX_AD_CHARS)
    );
    let prompt = qwen_prompt(DISCOVER_SYSTEM, &user);

    let t0 = std::time::Instant::now();
    let resp = {
        let w = state.ensure_wrapper()?;
        w.send(json!({
            "cmd": "generate",
            "session": session,
            "prompt": prompt,
            "opts": {
                "max_new_tokens": max_new_tokens,
                "stop_ids": [IM_END_ID],
                "stop_strings": ["<|im_end|>"],
            },
        }))?
    };
    let gen_ms = t0.elapsed().as_millis() as u64;
    let raw = resp
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let repaired = repair_json(&raw).ok_or_else(|| {
        ModelError::invalid_output(format!(
            "discover_schema: no json object in model output: {:?}",
            truncate_chars(&raw, 300)
        ))
    })?;
    let schema = clamp_schema(&repaired)?;

    Ok(json!({
        "cmd": "discover_schema",
        "schema": serde_json::to_value(&schema).unwrap_or_default(),
        "raw": raw,
        "gen_ms": gen_ms,
    }))
}

fn cmd_extract_constrained(state: &mut State, payload: &Value) -> Result<Value> {
    let model = require_model(payload)?;
    let record = resolve_model(state, model)?;
    ensure_kind(&record, &["extractor"], "extract_constrained")?;
    let cv_text = payload
        .get("cv_text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("extract_constrained requires 'cv_text'"))?;
    let schema_value = payload
        .get("schema")
        .cloned()
        .ok_or_else(|| ModelError::bad_req("extract_constrained requires 'schema'"))?;
    let schema: libcvevidence::ExtractionSchema = serde_json::from_value(schema_value.clone())
        .map_err(|e| ModelError::bad_req(format!("invalid schema: {e}")))?;
    let max_new_tokens = payload
        .get("opts")
        .and_then(|o| o.get("max_new_tokens"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(EXTRACT_MAX_NEW_TOKENS);
    let session = load_into(state, &record.slug)?;

    let user = format!(
        "Fields to fill:\n{}\n\nCV text:\n{}\n\nOutput the filled json object now.",
        render_schema_form(&schema),
        truncate_chars(cv_text, MAX_CV_CHARS)
    );
    let prompt = qwen_prompt(EXTRACT_SYSTEM, &user);

    let t0 = std::time::Instant::now();
    let resp = {
        let w = state.ensure_wrapper()?;
        w.send(json!({
            "cmd": "generate_constrained",
            "session": session,
            "prompt": prompt,
            "schema": schema_value,
            "opts": {
                "max_new_tokens": max_new_tokens,
                "stop_ids": [IM_END_ID],
                "stop_strings": ["<|im_end|>"],
            },
        }))?
    };
    let gen_ms = t0.elapsed().as_millis() as u64;
    let raw = resp
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    // constrained decode emits the full object; the repair pass stays
    // belt-and-braces (design 14 §14.2)
    let repaired = repair_json(&raw).ok_or_else(|| {
        ModelError::invalid_output(format!(
            "extract_constrained: no json object in model output: {:?}",
            truncate_chars(&raw, 300)
        ))
    })?;
    let obj: serde_json::Map<String, Value> = serde_json::from_str(&repaired)
        .map_err(|e| ModelError::invalid_output(format!("extract_constrained: {e}")))?;
    let (cleaned, dropped, issues) = schema.validate_extensions(&obj);

    Ok(json!({
        "cmd": "extract_constrained",
        "extensions": cleaned,
        "dropped": dropped,
        "issues": issues,
        "gen_ms": gen_ms,
    }))
}

/// strip markdown fences, then take the first balanced `{...}` block
/// (string-aware); if the block is truncated (no closing brace at the
/// top level) fall back to the longest prefix that is still valid json —
/// the "truncate to last balanced brace" repair (design 14 §14.2 / 18.6
/// belt-and-braces). returns the substring, unparseable → None
fn repair_json(raw: &str) -> Option<String> {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("```") {
        s = rest;
    }
    if let Some(stripped) = s.strip_suffix("```") {
        s = stripped;
    }
    let s = s.trim();
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    // truncated under the token cap: discovery emits {"fields":[{...},{...},
    // and the last field + the array/object are cut mid-token. walk back from
    // the end to the last complete field object (`}`) and append `]}` to close
    // the array + object, keeping the longest prefix that parses as json — the
    // half-emitted final field drops off. constrained extraction can't
    // truncate (the FillMachine forces closers) so this only rescues
    // discover_schema.
    let mut i = bytes.len();
    while i > start {
        i -= 1;
        if bytes[i] == b'}' {
            let mut cand = s[start..=i].to_string();
            cand.push_str("]}");
            if serde_json::from_str::<serde_json::Value>(&cand).is_ok() {
                return Some(cand);
            }
        }
    }
    None
}

/// clamp a discovered schema to the contract (design 14 §14.2): max 12
/// fields, snake_case names, known types, enum only on strings; a schema
/// with zero surviving fields is invalid_output
fn clamp_schema(repaired: &str) -> Result<libcvevidence::ExtractionSchema> {
    let value: Value = serde_json::from_str(repaired)
        .map_err(|e| ModelError::invalid_output(format!("discover_schema: json parse: {e}")))?;
    let raw_fields = value
        .get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| ModelError::invalid_output("discover_schema: missing 'fields' array"))?;

    let name_ok = |n: &str| {
        n.len() >= 3 && n.len() <= 40 && n.chars().all(|c| c.is_ascii_lowercase() || c == '_')
    };
    let mut fields = Vec::new();
    for f in raw_fields {
        if fields.len() >= 12 {
            break;
        }
        let name = f.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if !name_ok(name) {
            continue;
        }
        let kind = match f.get("type").and_then(|v| v.as_str()) {
            Some("numeric") => libcvevidence::FieldType::Numeric,
            Some("string") => libcvevidence::FieldType::String,
            Some("bool") => libcvevidence::FieldType::Bool,
            Some("list_string") => libcvevidence::FieldType::ListString,
            _ => continue, // unknown type strings drop the field
        };
        let enum_values = if kind == libcvevidence::FieldType::String {
            f.get("enum").and_then(|v| v.as_array()).and_then(|a| {
                let vals: Vec<String> = a
                    .iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect();
                if vals.is_empty() {
                    None
                } else {
                    Some(vals)
                }
            })
        } else {
            None
        };
        fields.push(libcvevidence::SchemaField {
            name: name.to_string(),
            kind,
            description: f
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from),
            required: f.get("required").and_then(|v| v.as_bool()).unwrap_or(false),
            enum_values,
        });
    }
    if fields.is_empty() {
        return Err(ModelError::invalid_output(
            "discover_schema: zero valid fields after clamping",
        ));
    }
    let schema = libcvevidence::ExtractionSchema::new(fields);
    // round-trip through the strict contract
    let json = schema
        .to_json()
        .map_err(|e| ModelError::invalid_output(format!("discover_schema: {e}")))?;
    libcvevidence::ExtractionSchema::from_json(&json)
        .map_err(|e| ModelError::invalid_output(format!("discover_schema: {e}")))
}

fn cmd_models(state: &State, payload: &Value) -> Result<Value> {
    log::debug!("cmd_models: called");
    let filter = payload.get("model").and_then(|v| v.as_str());
    log::debug!("cmd_models: filter = {:?}", filter);
    let mut out = Vec::new();
    for m in state.registry.all() {
        if let Some(f) = filter {
            if m.slug != f {
                continue;
            }
        }
        let loaded = state.loaded.get(&m.slug).map(|lm| lm.runtime);
        out.push(model_metadata_json(m, loaded));
    }
    if let Some(f) = filter {
        if out.is_empty() {
            log::debug!("cmd_models: model '{}' not found in registry", f);
            return Err(ModelError::no_model(f));
        }
    }
    log::debug!("cmd_models: returning {} models", out.len());
    Ok(json!({ "models": out }))
}

fn cmd_list(state: &State, _payload: &Value) -> Result<Value> {
    let loaded: Vec<Value> = state
        .loaded
        .values()
        .map(|lm| model_metadata_json_from_resolved(&lm.model, Some(lm.runtime)))
        .collect();
    let available: Vec<String> = state
        .registry
        .all()
        .iter()
        .map(|m| m.slug.clone())
        .collect();
    Ok(json!({ "loaded": loaded, "available": available }))
}

fn cmd_recommend(state: &State, payload: &Value) -> Result<Value> {
    let task = payload
        .get("task")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("recommend requires 'task'"))?;
    let best = recommend_for(state, task);
    let best = best
        .as_ref()
        .ok_or_else(|| ModelError::no_model("no model for task"))?;
    Ok(json!({
        "task": task,
        "model": best.slug,
        "reason": "well_known_default"
    }))
}

fn require_model(payload: &Value) -> Result<&str> {
    payload
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ModelError::bad_req("missing 'model' field"))
}

fn recommend_for(state: &State, task: &str) -> Option<ResolvedModel> {
    let candidates: Vec<ResolvedModel> = match task {
        "embed" => state
            .registry
            .all()
            .iter()
            .filter_map(|m| {
                if m.kind != "embedder" {
                    return None;
                }
                state.registry.find(&m.slug)
            })
            .collect(),
        "rerank" => state
            .registry
            .all()
            .iter()
            .filter_map(|m| {
                if m.kind != "reranker" {
                    return None;
                }
                state.registry.find(&m.slug)
            })
            .collect(),
        "discover_schema" | "extract_constrained" => state
            .registry
            .all()
            .iter()
            .filter_map(|m| {
                if m.kind != "extractor" && m.kind != "schema_discoverer" {
                    return None;
                }
                state.registry.find(&m.slug)
            })
            .collect(),
        _ => return None,
    };
    candidates.into_iter().next()
}

/// Model metadata, plus the runtime that served it when the model is loaded.
///
/// `runtime` is `None` for a model that has not been loaded: "not loaded yet"
/// and "loaded, on the CPU" are different facts, and a gui that shows an
/// accelerator badge needs to tell them apart (design 15 §15.4.1).
fn model_metadata_json(m: &crate::model_registry::ModelDef, loaded: Option<Runtime>) -> Value {
    let mut v = json!({
        "model": m.slug,
        "kind": m.kind,
        "dim": m.dim,
        "max_context_window": m.max_context_window,
        "loaded": loaded.is_some(),
    });
    if let Some(rt) = loaded {
        v["runtime"] = json!(rt.as_str());
    }
    v
}

fn model_metadata_json_from_resolved(
    m: &crate::model_registry::ResolvedModel,
    loaded: Option<Runtime>,
) -> Value {
    let mut v = json!({
        "model": m.slug,
        "kind": m.kind,
        "dim": m.dim,
        "max_context_window": m.max_context_window,
        "loaded": loaded.is_some(),
    });
    if let Some(rt) = loaded {
        v["runtime"] = json!(rt.as_str());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::StdioArgs;
    use crate::model_registry;

    fn stdio_args() -> StdioArgs {
        StdioArgs {
            backend: "auto".to_string(),
            model_dir: Some(PathBuf::from("/tmp/cache")),
            revision: "main".to_string(),
            exec_provider: "auto".to_string(),
            precision: "fp32".to_string(),
            auto_download_models: false,
        }
    }

    /// Session ids must be unique ACROSS arms, because each arm numbers its
    /// own handles from 1 and the scout and the heavy embedder routinely land on
    /// the same number. Before this was modelled, the heavy embedder's embed was
    /// routed to the scout's session on the other arm and returned 1024-dim
    /// vectors for a 640-dim model — silent corruption with no error anywhere.
    #[test]
    fn a_session_id_from_one_arm_never_resolves_to_another() {
        let mut state = State::new(
            test_registry(),
            stdio_args(),
            Default::default(),
            Default::default(),
            false,
        );
        // Both arms hand out handle 2: the scout on ort, the heavy model on mlx.
        let scout = state.alloc_session(Runtime::Ort, 2);
        let heavy = state.alloc_session(Runtime::Mlx, 2);
        assert_ne!(
            scout, heavy,
            "global ids must differ even when arm handles match"
        );

        assert_eq!(state.arm_session(scout).expect("scout"), (Runtime::Ort, 2));
        assert_eq!(state.arm_session(heavy).expect("heavy"), (Runtime::Mlx, 2));
    }

    #[test]
    fn an_unknown_session_id_is_an_error_not_a_default_arm() {
        let state = State::new(
            test_registry(),
            stdio_args(),
            Default::default(),
            Default::default(),
            false,
        );
        // Defaulting to the ONNX arm here would send the request to an arm that
        // does not hold the model, which answers with someone else's vectors.
        assert!(state.arm_session(999).is_err());
    }

    fn test_registry() -> Registry {
        let yaml = r#"
families:
  - family: test-fam
    usage: plain
models:
  - slug: test-model
    family: test-fam
    kind: embedder
    dim: 768
    max_context_window: 8192
    source: hf:example/test-model
    onnx_url: https://example.com/model.onnx
"#;
        model_registry::Registry::load_from_str(yaml).unwrap()
    }

    #[test]
    fn ping_response() {
        let mut state = State::new(
            test_registry(),
            stdio_args(),
            libconfigreader::EngineSection::default(),
            libconfigreader::SyscapSection::default(),
            false,
        );
        let resp = handle_request(&mut state, &json!({ "cmd": "ping" })).unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["version"], "0.1.0");
    }

    #[test]
    fn adapt_wrapper_scores_to_results() {
        let mut resp = json!({ "scores": [0.5f32, -1.25f32, 2.0f32] });
        adapt_wrapper_rerank_response(&mut resp);
        let results = resp["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["index"], 0);
        assert_eq!(results[0]["relevance_score"], 0.5);
        assert_eq!(results[2]["relevance_score"], 2.0);
    }

    #[test]
    fn adapt_rerank_noop_when_results_present() {
        let mut resp = json!({
            "scores": [9.0],
            "results": [{ "index": 0, "relevance_score": 1.5 }],
        });
        adapt_wrapper_rerank_response(&mut resp);
        assert_eq!(resp["results"][0]["relevance_score"], 1.5);
    }

    #[test]
    fn adapt_rerank_noop_without_scores() {
        let mut resp = json!({ "error": "boom" });
        adapt_wrapper_rerank_response(&mut resp);
        assert!(resp.get("results").is_none());
    }

    #[test]
    fn recommend_embed() {
        let mut state = State::new(
            test_registry(),
            stdio_args(),
            libconfigreader::EngineSection::default(),
            libconfigreader::SyscapSection::default(),
            false,
        );
        let resp =
            handle_request(&mut state, &json!({ "cmd": "recommend", "task": "embed" })).unwrap();
        assert_eq!(resp["model"], "test-model");
    }

    #[test]
    fn unknown_model_is_error() {
        let mut state = State::new(
            test_registry(),
            stdio_args(),
            libconfigreader::EngineSection::default(),
            libconfigreader::SyscapSection::default(),
            false,
        );
        let r = handle_request(&mut state, &json!({ "cmd": "load", "model": "nope-model" }));
        assert!(matches!(r, Err(e) if e.code == "no_model"));
    }

    #[test]
    fn models_list_all() {
        let state = State::new(
            test_registry(),
            stdio_args(),
            libconfigreader::EngineSection::default(),
            libconfigreader::SyscapSection::default(),
            false,
        );
        let resp = cmd_models(&state, &json!({})).unwrap();
        let arr = resp["models"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn repair_strips_fences_and_takes_balanced_object() {
        let raw = "Sure!\n```json\n{\"fields\": []}\n```\nignore me { too";
        assert_eq!(repair_json(raw).unwrap(), r#"{"fields": []}"#);
        assert!(repair_json("no json here").is_none());
        assert_eq!(
            repair_json("prefix {\"a\": \"}\"} suffix").unwrap(),
            r#"{"a": "}"}"#
        );
    }

    #[test]
    fn repair_recovers_truncated_object() {
        // model cut off at the token cap before the closing brace —
        // the last field is incomplete, the longest valid prefix drops it
        let raw = r#"```json
{"fields": [
  {"name":"years_experience","type":"numeric","description":"years of practice","required":true},
  {"name":"certified","type":"bool","description":"part"#;
        let repaired = repair_json(raw).unwrap();
        assert!(repaired.starts_with(r#"{"fields": ["#));
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        // first valid field survives, the truncated last one does not
        assert_eq!(v["fields"].as_array().unwrap().len(), 1);
        assert_eq!(v["fields"][0]["name"], "years_experience");
        // a genuinely unbalanced blob still fails
        assert!(repair_json(
            r#"```json
{"fields": [{"name":"x","type":"num"#
        )
        .is_none());
    }

    #[test]
    fn clamp_drops_bad_fields_keeps_good() {
        let schema = clamp_schema(
            r#"{"fields":[
                {"name":"years_sql","type":"numeric","description":"yrs","required":true},
                {"name":"Bad Name","type":"numeric"},
                {"name":"valid_name","type":"dictionary"},
                {"name":"level","type":"string","enum":["junior","senior"]},
                {"name":"has_degree","type":"bool","required":false}
            ]}"#,
        )
        .unwrap();
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[0].name, "years_sql");
        assert!(schema.fields[0].required);
        assert_eq!(
            schema.fields[1].enum_values.as_deref(),
            Some(&["junior".to_string(), "senior".to_string()][..])
        );
        assert!(clamp_schema(r#"{"fields":[{"name":"x","type":"numeric"}]}"#).is_err());
        assert!(clamp_schema(r#"{"nope":1}"#).is_err());
    }

    #[test]
    fn clamped_schema_validates_extensions() {
        let schema =
            clamp_schema(r#"{"fields":[{"name":"years_sql","type":"numeric","required":true}]}"#)
                .unwrap();
        let ext = json!({"years_sql": 5, "junk": true})
            .as_object()
            .unwrap()
            .clone();
        let (cleaned, dropped, _) = schema.validate_extensions(&ext);
        assert_eq!(cleaned.get("years_sql"), Some(&json!(5)));
        assert!(dropped.contains(&"junk".to_string()));
    }

    #[test]
    fn qwen_prompt_has_chat_markers() {
        let p = qwen_prompt("sys", "usr");
        assert!(p.starts_with("<|im_start|>system\nsys<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "héllo wörld".repeat(100);
        let t = truncate_chars(&s, 50);
        assert!(t.len() <= 50);
        assert!(t.is_char_boundary(t.len()));
        assert_eq!(truncate_chars(&s, s.len() + 10), s);
    }

    #[test]
    fn schema_form_renders_fields_and_enums() {
        let schema = clamp_schema(
            r#"{"fields":[{"name":"years_sql","type":"numeric","description":"yrs","required":true},{"name":"level","type":"string","enum":["junior"]}]}"#,
        )
        .unwrap();
        let form = render_schema_form(&schema);
        assert!(form.contains("- years_sql (numeric): yrs"));
        assert!(form.contains("- level (string): - [one of: junior]"));
    }
}
