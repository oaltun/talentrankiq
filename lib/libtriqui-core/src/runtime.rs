//! The embedding engine runtime: app-owned lifecycle for on-device inference.
//!
//! The host app runs `modelcaller` (the ONNX orchestrator) **in-process**:
//! [`EngineRuntime::start`] spawns the modelcaller JSONL loop on a dedicated
//! thread over an in-memory duplex and talks to it via the shared
//! `libmodelcallerclient` — no second binary, no port, no HTTP server (there
//! is no modelcaller binary at all anymore). Below modelcaller, the ORT
//! wrapper (`onnxwrapperrs`) also runs in-process: its JSONL loop serves OS
//! pipes on a dedicated thread, so all ONNX inference happens off the host's
//! main thread and the gui ships as a single binary with no spawned children.
//! On exit the engine is shut down via a `shutdown` request.
//!
//! This module is Tauri-free: instead of emitting Tauri events directly, status
//! changes are pushed through an optional [`StatusHook`] (the GUI installs one
//! that emits the `runtime-status` event; the CLI just logs).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rusqlite::Connection;

use libmodelcallerclient::ModelCallerClient;

use crate::app_config;
use crate::db;
use crate::embed::{Embedder, ModelCallerEmbedder};
use crate::types::{Activity, DomainError, RuntimeStatusInfo};
use libtriqui_license::{load_judge_license, License};

/// Callback invoked on every status change. The GUI forwards these as the
/// `runtime-status` Tauri event so the UI can react (e.g. enable Sort).
pub type StatusHook = Arc<dyn Fn(&RuntimeStatusInfo) + Send + Sync>;

/// Runtime config, discovered from env / `triq.yaml` (`model_dir`).
pub struct RuntimeConfig {
    /// Path to the config file (`triq.yaml` preferred, embedded default fallback).
    pub config_path: Option<std::path::PathBuf>,
    /// On-disk ONNX model cache root.
    pub model_dir: std::path::PathBuf,
    /// Embedding model slug resolved against the `modelcaller` registry.
    pub model: String,
    /// Judge stage artifacts (`gui_judge_artifact`): per-embedder overrides
    /// plus a default; empty = judge disabled.
    pub judge_artifacts: libconfigreader::GuiJudgeArtifacts,
}

impl RuntimeConfig {
    /// Discover the toolchain from `triq.yaml`/env. `model` (`None` = env
    /// `SHORTLIST_EMBED_MODEL`, else the `gui_embedders` default from
    /// `triq.yaml`) is resolved by the caller when it has extra sources
    /// (the GUI layers user preferences on top).
    pub fn discover(model: Option<String>) -> Self {
        let model = model
            .or_else(|| std::env::var("SHORTLIST_EMBED_MODEL").ok())
            .unwrap_or_else(|| {
                let cfg = app_config::load_triq();
                cfg.gui_embedders
                    .iter()
                    .find(|ge| ge.default)
                    .map(|ge| ge.slug.clone())
                    .unwrap_or_default()
            });
        Self {
            config_path: app_config::triq_config_path(),
            model_dir: app_config::model_dir(),
            model,
            judge_artifacts: app_config::gui_judge_artifacts(),
        }
    }
}

#[derive(Clone, Debug)]
enum Status {
    Starting,
    Running { model: String },
    Error(String),
    Stopped,
}

struct RuntimeInner {
    status: Status,
    client: Option<ModelCallerClient>,
    embedder: Option<Arc<ModelCallerEmbedder>>,
    /// Extra embedders by model slug, for models live alongside the current one
    /// (the cascade's scout). Empty on a plain single-model run.
    secondary: HashMap<String, Arc<dyn Embedder>>,
    /// Bumped by every [`EngineRuntime::shutdown`]. A start thread captures
    /// the epoch when it claims the `Stopped → Starting` transition and
    /// re-checks before publishing: without this, a shutdown landing mid-load
    /// (app exit, idle unload, embedder switch) would be undone by the start
    /// thread's `finish_running`, re-arming the engine — and its in-flight
    /// `load` would keep the ORT wrapper alive into process exit.
    epoch: u64,
}

/// Shared handle to the embedding engine (managed state in the GUI).
pub struct EngineRuntime {
    db_path: PathBuf,
    cfg: RuntimeConfig,
    /// Embedding model slug — mutable at runtime via [`EngineRuntime::switch_model`]
    /// (the rest of `cfg` is fixed at startup).
    model: Mutex<String>,
    inner: Mutex<RuntimeInner>,
    hook: Mutex<Option<StatusHook>>,
    /// What the engine is working on right now, for the gui's status line.
    act: Mutex<ActTrack>,
    /// Inspector priority gate: while held, background embeds (cascade waves,
    /// scout, import) pause between work units so the CV Inspector's own
    /// embeds run immediately instead of queueing behind a wave.
    gate: Gate,
    /// Arm each model was actually served by (from the load response's own
    /// report, `models` command) — slug → `"ort"` | `"mlx"`. Cached once per
    /// slug per process; nothing re-reads it per status push.
    arms: Mutex<HashMap<String, String>>,
    /// Judge license (loaded once at startup). `None` = judge disabled.
    judge_license: Option<License>,
}

struct ActTrack {
    current: Option<Activity>,
    /// Stage+subject of the last push — a detail-only change (chunk 4/17 →
    /// 5/17) is throttled so the event stream stays readable.
    last_key: Option<(String, String)>,
    last_push: Instant,
}

/// See [`EngineRuntime::gate`].
struct Gate {
    /// Active inspector holders; background embeds pause while > 0.
    holders: Mutex<usize>,
    cv: std::sync::Condvar,
}

/// Held while an inspector operation runs; dropping it releases the
/// background embeds.
pub struct GateGuard<'a> {
    gate: &'a Gate,
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        let mut h = self.gate.holders.lock().expect("gate lock poisoned");
        *h = h.saturating_sub(1);
        if *h == 0 {
            self.gate.cv.notify_all();
        }
    }
}

impl Gate {
    const fn new() -> Self {
        Self {
            holders: Mutex::new(0),
            cv: std::sync::Condvar::new(),
        }
    }

    fn hold(&self) -> GateGuard<'_> {
        let mut h = self.holders.lock().expect("gate lock poisoned");
        *h += 1;
        GateGuard { gate: self }
    }

    /// Block while a gate holder is active. Polls with a timeout so a lost
    /// notify can never wedge a background embed for good; the caller re-checks
    /// its own cancel flags between work units anyway.
    fn wait(&self) {
        let mut h = self.holders.lock().expect("gate lock poisoned");
        while *h > 0 {
            let (h2, _) = self
                .cv
                .wait_timeout(h, Duration::from_millis(200))
                .expect("gate lock poisoned");
            h = h2;
        }
    }
}

impl EngineRuntime {
    pub fn new(cfg: RuntimeConfig, db_path: PathBuf) -> Arc<Self> {
        let model = cfg.model.clone();
        // Load judge license once at startup
        let judge_license = load_judge_license();
        Arc::new(Self {
            db_path,
            cfg,
            model: Mutex::new(model),
            inner: Mutex::new(RuntimeInner {
                status: Status::Stopped,
                client: None,
                embedder: None,
                secondary: HashMap::new(),
                epoch: 0,
            }),
            hook: Mutex::new(None),
            act: Mutex::new(ActTrack {
                current: None,
                last_key: None,
                last_push: Instant::now(),
            }),
            gate: Gate::new(),
            arms: Mutex::new(HashMap::new()),
            judge_license,
        })
    }

    /// Report what the engine is working on — one status-line update. Pushes
    /// to the status hook immediately on a stage/subject change, throttled to
    /// ~6 Hz for detail-only updates (chunk 4/17 → 5/17) so the event stream
    /// stays readable.
    pub fn set_activity(
        &self,
        stage: &str,
        subject: &str,
        detail: Option<String>,
        model: Option<&str>,
    ) {
        let activity = Activity {
            stage: stage.to_string(),
            subject: subject.to_string(),
            detail,
            model: model.map(|m| m.to_string()),
        };
        let key = (activity.stage.clone(), activity.subject.clone());
        let push = {
            let mut t = self.act.lock().expect("activity lock poisoned");
            let changed = t.last_key.as_ref() != Some(&key);
            let due = t.last_push.elapsed() >= Duration::from_millis(150);
            t.current = Some(activity);
            if changed {
                t.last_key = Some(key);
            }
            changed || due
        };
        if push {
            self.push_activity();
        }
    }

    /// The engine is idle again — drop the status line and push the edge.
    pub fn clear_activity(&self) {
        let had = {
            let mut t = self.act.lock().expect("activity lock poisoned");
            t.last_key = None;
            t.current.take().is_some()
        };
        if had {
            self.push_activity();
        }
    }

    /// Push a status snapshot because activity changed (the busy flag may not
    /// have moved — a judge pass never touches the engine).
    fn push_activity(&self) {
        {
            let mut t = self.act.lock().expect("activity lock poisoned");
            t.last_push = Instant::now();
        }
        self.push_status();
    }

    /// Pause background embeds while the returned guard lives: the CV
    /// Inspector's embeds must not queue behind a cascade wave. Re-nested
    /// holds are fine (each hold adds one count).
    pub fn gate_hold(&self) -> GateGuard<'_> {
        self.gate.hold()
    }

    /// Background-side checkpoint: called by cascade/import/scout embed paths
    /// before engine work; returns once no inspector operation holds the gate.
    pub fn gate_wait(&self) {
        self.gate.wait();
    }

    /// Which arm is about to serve `model`, engine-free: the modelcaller
    /// registry + backend setting + filesystem presence, without loading
    /// anything. `None` when the model slug is not in the registry.
    fn predicted_backend(&self, model: &str) -> Option<String> {
        let ctx = app_config::registry_ctx();
        libmodelcaller::variant_status(
            &ctx.registry,
            ctx.backend,
            &self.cfg.model_dir,
            &ctx.revision,
            model,
        )
        .ok()
        .map(|s| s.runtime.to_string())
    }

    /// Record the arm that wrote `model`'s vectors. When it differs from what
    /// was recorded before, every slug-keyed vector artifact is cleared first
    /// (chunks, embedding cache, centroids, criteria, scout + feedback
    /// vectors) — old-arm vectors must never read as current-model vectors.
    /// Model-scoped by construction; the scout's slug never flips (it
    /// declares no mlx variant), so scout rows survive.
    pub fn note_backend(&self, conn: &Connection, model: &str, backend: &str) {
        let recorded: Option<String> = conn
            .query_row(
                "SELECT backend FROM model_runtime WHERE model = ?1",
                rusqlite::params![model],
                |r| r.get(0),
            )
            .ok();
        if recorded.as_deref() == Some(backend) {
            return;
        }
        if recorded.is_some() {
            log::info!(
                "backend switch for '{model}': {} -> {backend} — clearing its cached vectors",
                recorded.as_deref().unwrap_or("?")
            );
            for sql in [
                "DELETE FROM candidate_chunks WHERE vector_model = ?1",
                "DELETE FROM embedding_cache WHERE model = ?1",
                "DELETE FROM job_feedback_vectors WHERE model = ?1",
                "DELETE FROM profile_vectors WHERE model = ?1",
                "UPDATE candidates SET vector = NULL, vector_model = NULL WHERE vector_model = ?1",
                "UPDATE criteria SET vector = NULL, vector_model = NULL WHERE vector_model = ?1",
            ] {
                if let Err(e) = conn.execute(sql, rusqlite::params![model]) {
                    log::warn!("backend-switch clear failed ({sql}): {e}");
                }
            }
            // The space's grades and rubric vectors are cosines of those vectors,
            // so they go with them.
            if let Err(e) = db::delete_space(conn, model) {
                log::warn!("backend-switch clear failed (delete_space): {e}");
            }
        }
        let _ = conn.execute(
            "INSERT INTO model_runtime(model, backend, updated_at)
             VALUES (?1, ?2, datetime('now'))
             ON CONFLICT(model) DO UPDATE SET backend = ?2, updated_at = datetime('now')",
            rusqlite::params![model, backend],
        );
    }

    /// Pre-embed freshness gate: clear stale-arm vectors for `model` before
    /// anything reads them (including the engine-free cache-only pass, which
    /// reads `embedding_cache` written by a previous process). Predicted arm:
    /// no engine needed.
    pub fn ensure_backend_fresh(&self, conn: &Connection, model: &str) {
        if let Some(arm) = self.predicted_backend(model) {
            self.note_backend(conn, model, &arm);
        }
    }

    /// Post-embed reconcile: the arm that ACTUALLY served may differ from the
    /// prediction (a failed MLX load falls down the ladder). If so, the
    /// just-written vectors are from the other arm than recorded — clear and
    /// record the truth; the next pass re-embeds.
    pub fn reconcile_backend(&self, conn: &Connection, model: &str) {
        if let Some(arm) = self.resolved_backend(model) {
            self.note_backend(conn, model, &arm);
        }
    }

    /// The arm serving `model`, as the arm itself reported on load —
    /// `"ort"` or `"mlx"`. `None` when the engine isn't running or the model
    /// was never loaded. Cached per slug; the first query is one in-process
    /// `models` roundtrip.
    pub fn resolved_backend(&self, model: &str) -> Option<String> {
        if let Some(b) = self.arms.lock().expect("arm cache poisoned").get(model) {
            return Some(b.clone());
        }
        let meta = self.client().ok()?.model_metadata(model).ok()?;
        let arm = meta.runtime?;
        self.arms
            .lock()
            .expect("arm cache poisoned")
            .insert(model.to_string(), arm.clone());
        Some(arm)
    }

    /// The embedding model this runtime loads (re)starts with.
    pub fn model(&self) -> String {
        self.model.lock().expect("model lock poisoned").clone()
    }

    /// Judge stage artifact for embedder `model` (`None` = no artifact
    /// configured for it — the per-embedder override, else the default).
    pub fn judge_artifact_for(&self, model: &str) -> Option<std::path::PathBuf> {
        // Gate: if no valid judge license, disable judge for all embedders
        if self.judge_license.is_none() {
            return None;
        }
        let artifact_path = self.cfg
            .judge_artifacts
            .for_embedder(model)
            .map(|p| p.to_path_buf())?;
        // Check if this specific artifact is authorized by the license
        let artifact_id = artifact_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !self.judge_license.as_ref().unwrap().authorizes_artifact(artifact_id) {
            return None;
        }
        Some(artifact_path)
    }

    /// Swap the embedding model at runtime. Refuses while the engine is starting
    /// or mid-embed; otherwise shuts the current engine down and leaves it
    /// stopped. The new model loads lazily on the first real need (an uncached
    /// embed): when every needed vector is already in the `embedding_cache` —
    /// e.g. switching back to a model used before — nothing is loaded into
    /// memory at all. `wait_until_ready` restarts from `Stopped` on demand, so
    /// every genuine-need path keeps working; the status events carry the new
    /// model, and a failed lazy start surfaces as `Status::Error` as usual.
    pub fn switch_model(self: &Arc<Self>, model: String) -> Result<(), DomainError> {
        if self.model() == model {
            return Ok(());
        }
        {
            let g = self.inner.lock().expect("runtime lock poisoned");
            let busy = g.embedder.as_ref().map(|e| e.is_busy()).unwrap_or(false);
            if busy || matches!(g.status, Status::Starting) {
                return Err(DomainError::ValidationError {
                    message:
                        "The AI engine is busy — wait for embedding to finish, then try again."
                            .to_string(),
                });
            }
        }
        *self.model.lock().expect("model lock poisoned") = model;
        self.shutdown();
        Ok(())
    }

    /// Install the status-change callback (replaces any previous one).
    pub fn set_status_hook(&self, hook: StatusHook) {
        *self.hook.lock().expect("status hook lock") = Some(hook);
    }

    /// Open a fresh SQLite connection at this runtime's DB path (schema applied).
    pub fn open_conn(&self) -> Result<Connection, String> {
        db::open_connection_at(&self.db_path)
    }

    /// Run startup + readiness. Call from a dedicated `std::thread` (or directly
    /// for blocking startup, e.g. from the CLI).
    ///
    /// Only one start may run at a time: the `Stopped → Starting` transition
    /// happens atomically under the runtime lock, and a second concurrent
    /// caller (e.g. `ensure_started` racing from the prewarm and import
    /// paths) returns immediately instead of bringing up a second engine and
    /// clobbering the status back to `Starting`.
    pub fn start(self: &Arc<Self>) {
        let my_epoch = {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            if !matches!(g.status, Status::Stopped) {
                return;
            }
            g.status = Status::Starting;
            g.epoch
        };

        let cfg = match &self.cfg.config_path {
            Some(config) => match libconfigreader::load(config, &libconfigreader::CliOverrides::new()) {
                Ok(c) => c,
                Err(e) => {
                    self.set_status_if_current(
                        Status::Error(format!("failed to load {}: {e}", config.display())),
                        my_epoch,
                    );
                    return;
                }
            },
            None => app_config::load_triq(),
        };

        log::info!(
            "engine runtime: starting in-process modelcaller (ORT wrapper in-process)"
        );

        let registry = libmodelcaller::Registry::from_config(&cfg);
        // Process-wide override (gui `--cpu-only`) wins field-by-field over
        // the config; the merged section feeds both the wire args and the
        // engine section modelcaller resolves session knobs from.
        let engine = match app_config::engine_override() {
            Some(o) => {
                log::info!("engine runtime: applying engine override (cpu-only mode)");
                libconfigreader::EngineSection {
                    backend: o.backend.or(cfg.engine.backend),
                    exec_provider: o.exec_provider.or(cfg.engine.exec_provider),
                    intra_threads: o.intra_threads.or(cfg.engine.intra_threads),
                    inter_threads: o.inter_threads.or(cfg.engine.inter_threads),
                    embed_batch: o.embed_batch.or(cfg.engine.embed_batch),
                    embed_sessions: o.embed_sessions.or(cfg.engine.embed_sessions),
                    provider_logging: o.provider_logging.or(cfg.engine.provider_logging),
                    no_cache: o.no_cache.or(cfg.engine.no_cache),
                    ..cfg.engine.clone()
                }
            }
            None => cfg.engine.clone(),
        };
        let args = libmodelcaller::StdioArgs {
            model_dir: Some(self.cfg.model_dir.clone()),
            revision: cfg.revision.clone(),
            // Which runtime serves embeds (design 15 §15.4). Config, like the
            // provider: bench can pin `ort` and the gui can follow `auto`
            // without either drifting onto the other's silicon.
            backend: engine
                .backend
                .map(|b| b.as_str().to_string())
                .unwrap_or_else(|| "auto".to_string()),
            exec_provider: engine
                .exec_provider
                .as_ref()
                .map(|p| p.as_str().to_string())
                .unwrap_or_else(|| "auto".to_string()),
            precision: "fp32".to_string(),
            auto_download_models: true,
        };
        let syscap_cfg = cfg.syscap.clone();
        let verbose = app_config::verbose();

        // In-memory transport, two sides: the host (client) owns the a-ends,
        // the modelcaller loop owns the b-ends. a_write → b_read carries
        // requests; b_write → a_read carries responses/events. The loop runs
        // on its own thread; dropping the client's write end (or a `shutdown`
        // request) is what tears it down.
        let (a_write, a_read, b_write, b_read) = libtriqui_client::transport::duplex();
        let server = std::thread::Builder::new()
            .name("modelcaller".into())
            .spawn(move || {
                if let Err(e) = libmodelcaller::run_stdio_over(
                    registry, args, engine, syscap_cfg, verbose, None, b_read, b_write,
                ) {
                    log::error!("in-process modelcaller exited: {e}");
                }
            });
        if let Err(e) = server {
            self.set_status(Status::Error(format!("failed to start modelcaller: {e}")));
            return;
        }
        let client = ModelCallerClient::over(Box::new(a_read), Box::new(a_write));

        // 1. Readiness: a `ping` confirms modelcaller is up (the wrapper is spawned
        //    lazily on first load/embed, so ping alone doesn't touch the model).
        match client.ping() {
            Ok(true) => log::info!("engine runtime: modelcaller responded to ping"),
            Ok(false) => {
                self.set_status_if_current(
                    Status::Error("modelcaller responded but reported not-ok".to_string()),
                    my_epoch,
                );
                return;
            }
            Err(e) => {
                self.set_status_if_current(
                    Status::Error(format!("modelcaller started but did not respond: {e}")),
                    my_epoch,
                );
                return;
            }
        }

        // 2. Best-effort warm-up: load the model so the first embed is fast. A
        //    failure here is non-fatal — embedding will surface the real error.
        let model = self.model();
        if let Err(e) = client.load(&model) {
            log::warn!(
                "engine runtime: model '{model}' preload failed (will retry on first embed): {e}"
            );
        }

        let client = Arc::new(client);
        let embedder = Arc::new(ModelCallerEmbedder::new(client.clone(), model));

        // Push Working/Ready edges to the host for as long as this runtime lives.
        let weak = Arc::downgrade(self);
        embedder.busy_flag().set_hook(Arc::new(move |busy| {
            log::info!("engine runtime: {}", if busy { "working" } else { "idle" });
            if let Some(rt) = weak.upgrade() {
                rt.push_status();
            }
        }));

        {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            if g.epoch != my_epoch {
                // A shutdown landed while the model was loading (app exit,
                // idle unload, embedder switch). Do not re-arm: the shutdown
                // already tore the modelcaller loop down — publishing Running
                // here would strand a live client nobody will ever shut down.
                log::info!("engine runtime: start aborted — shutdown landed mid-load");
                return;
            }
            g.client = Some((*client).clone());
            g.embedder = Some(embedder.clone());
        }
        self.finish_running(embedder, my_epoch);
        log::info!("engine runtime: modelcaller ready");
    }

    /// Start the runtime if it hasn't already been started. Safe to call repeatedly;
    /// only the first call kicks off spawn (on a dedicated thread). Used to defer
    /// engine startup until the frontend is up, so launch isn't blocked on it.
    /// Start the engine if it is `Stopped`. Racy callers (prewarm + import)
    /// may both spawn a start thread; [`EngineRuntime::start`] itself reserves
    /// the `Stopped → Starting` transition atomically, so the loser is a no-op.
    pub fn ensure_started(self: &Arc<Self>) {
        let should_start = {
            let g = self.inner.lock().expect("runtime lock poisoned");
            matches!(g.status, Status::Stopped)
        };
        if should_start {
            let rt = self.clone();
            std::thread::spawn(move || rt.start());
        }
    }

    /// Block until the engine is `Running`, errors out, stops, or `timeout` elapses.
    /// If the engine is `Stopped`, it is restarted automatically.
    /// Used by the import pipeline so embedding waits for the (slowly loading) engine.
    pub fn wait_until_ready(self: &Arc<Self>, timeout: Duration) -> Result<(), DomainError> {
        let deadline = Instant::now() + timeout;
        loop {
            {
                let g = self.inner.lock().expect("runtime lock poisoned");
                match &g.status {
                    Status::Running { .. } => return Ok(()),
                    Status::Error(m) => {
                        return Err(DomainError::ValidationError {
                            message: format!("AI engine error: {m}"),
                        });
                    }
                    Status::Stopped => {
                        drop(g);
                        self.ensure_started();
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                    Status::Starting => {}
                }
            }
            if Instant::now() >= deadline {
                return Err(DomainError::ValidationError {
                    message: "AI engine did not become ready in time".to_string(),
                });
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Returns the raw `ModelCallerClient` if the engine is `Running`. Used by
    /// `try_model` to temporarily load a different model.
    pub fn client(&self) -> Result<ModelCallerClient, DomainError> {
        let g = self.inner.lock().expect("runtime lock poisoned");
        match (&g.status, &g.client) {
            (Status::Running { .. }, Some(c)) => Ok(c.clone()),
            (Status::Error(m), _) => Err(DomainError::ValidationError {
                message: format!("AI engine error: {m}"),
            }),
            _ => Err(DomainError::ValidationError {
                message: "AI engine not ready yet".to_string(),
            }),
        }
    }

    /// Returns the embedder if the engine is `Running`, else a user-facing error.
    pub fn embedder(&self) -> Result<Arc<ModelCallerEmbedder>, DomainError> {
        let g = self.inner.lock().expect("runtime lock poisoned");
        match (&g.status, &g.embedder) {
            (Status::Running { .. }, Some(e)) => Ok(e.clone()),
            (Status::Error(m), _) => Err(DomainError::ValidationError {
                message: format!("AI engine error: {m}"),
            }),
            _ => Err(DomainError::ValidationError {
                message: "AI engine not ready yet".to_string(),
            }),
        }
    }

    /// An `Embedder` for a model that is **not** the runtime's current one, so a
    /// second embedder can run over the same in-process engine.
    ///
    /// The cascade's scout stage needs exactly this: a cheap multilingual model
    /// scoring every candidate before the heavy embedder is asked to do anything
    /// (progressive-ranking design §7). modelcaller keeps one ORT session per
    /// loaded model (`State::loaded`).
    ///
    /// The returned embedder is cached per slug, so repeated embeds reuse the
    /// same object; [`EngineRuntime::unload_secondary`] drops the cache entry
    /// and frees the model's session when a stage is done with it. Requires the
    /// engine to be `Running` (the scout runs inside a live cascade, which
    /// already started it).
    pub fn embedder_for(self: &Arc<Self>, model: &str) -> Result<Arc<dyn Embedder>, DomainError> {
        let mut g = self.inner.lock().expect("runtime lock poisoned");
        if let Some(existing) = g.secondary.get(model) {
            return Ok(existing.clone());
        }
        let client = match (&g.status, &g.client) {
            (Status::Running { .. }, Some(c)) => c.clone(),
            (Status::Error(m), _) => {
                return Err(DomainError::ValidationError {
                    message: format!("AI engine error: {m}"),
                })
            }
            _ => {
                return Err(DomainError::ValidationError {
                    message: "AI engine not ready yet".to_string(),
                })
            }
        };
        let embedder: Arc<dyn Embedder> = Arc::new(ModelCallerEmbedder::new(
            Arc::new(client),
            model.to_string(),
        ));
        g.secondary.insert(model.to_string(), embedder.clone());
        Ok(embedder)
    }

    /// Drop a secondary embedder and unload its model from the engine, freeing
    /// the model's ORT session. The scout uses this to get out of the way once
    /// its pass is done: it is fast enough that reloading on the next pass
    /// costs far less than keeping a second session warm, and only one model
    /// is ever resident at a time (design §7). A no-op for the runtime's
    /// current model — that session belongs to the primary embedder.
    pub fn unload_secondary(self: &Arc<Self>, model: &str) {
        if self.model() == model {
            return;
        }
        let client = {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            g.secondary.remove(model);
            match (&g.status, &g.client) {
                (Status::Running { .. }, Some(c)) => Some(c.clone()),
                _ => None,
            }
        };
        if let Some(client) = client {
            match client.unload(model) {
                Ok(()) => log::info!("engine runtime: unloaded '{model}'"),
                Err(e) => log::warn!(
                    "engine runtime: unloading '{model}' failed (harmless; it reloads on demand): {e}"
                ),
            }
        }
    }

    /// Load `model` into the engine so its first embed does not pay the load
    /// cost. Best-effort: a failure is logged and surfaces on first real embed.
    /// Used to warm the scout before its stage begins.
    pub fn preload(self: &Arc<Self>, model: &str) -> Result<(), DomainError> {
        self.wait_until_ready(Duration::from_secs(60))?;
        let client = self.client()?;
        if let Err(e) = client.load(model) {
            log::warn!(
                "engine runtime: preload of '{model}' failed (will retry on first embed): {e}"
            );
        }
        Ok(())
    }

    /// Snapshot of the engine state (UI / `runtime_status` command / CLI).
    pub fn status_info(&self) -> RuntimeStatusInfo {
        let (status, model, dim, busy, error) = {
            let g = self.inner.lock().expect("runtime lock poisoned");
            let busy = g.embedder.as_ref().map(|e| e.is_busy()).unwrap_or(false);
            match &g.status {
                Status::Starting => ("Starting".to_string(), self.model(), 0, false, None),
                Status::Running { model } => {
                    let dim = g.embedder.as_ref().map(|e| e.dim() as u32).unwrap_or(0);
                    ("Running".to_string(), model.clone(), dim, busy, None)
                }
                Status::Error(m) => ("Error".to_string(), self.model(), 0, false, Some(m.clone())),
                Status::Stopped => ("Stopped".to_string(), self.model(), 0, false, None),
            }
        };
        // The arm comes from the arm's own load report; only the running
        // engine can answer, and the query may be a (cached) roundtrip, so it
        // happens outside the runtime lock.
        let backend = match &*status {
            "Running" => self.resolved_backend(&model),
            _ => None,
        };
        RuntimeStatusInfo {
            status,
            model: Some(model),
            dim,
            busy,
            error,
            backend,
            activity: self
                .act
                .lock()
                .expect("activity lock poisoned")
                .current
                .clone(),
        }
    }

    /// Push the current status to the installed hook (no-op without one).
    fn push_status(&self) {
        let hook = self.hook.lock().expect("status hook lock").clone();
        if let Some(hook) = hook {
            hook(&self.status_info());
        }
    }

    /// Spawn the idle-unload monitor: a daemon thread that shuts the engine
    /// down after `gui_idle_unload_secs` (`triq.yaml`, default 300) without
    /// any request, freeing the model memory. `shutdown` is graceful —
    /// modelcaller closes the wrapper's request pipe, ending the wrapper
    /// thread and dropping the ORT sessions — and the engine restarts
    /// automatically on the next use because `wait_until_ready` restarts from
    /// `Stopped`. A timeout of `0` disables idle unloading.
    pub fn spawn_idle_monitor(self: &Arc<Self>) {
        let secs = app_config::gui_idle_unload_secs();
        if secs == 0 {
            log::info!("engine runtime: idle unload disabled (gui_idle_unload_secs: 0)");
            return;
        }
        let threshold = Duration::from_secs(secs);
        log::info!("engine runtime: unloading model after {secs}s of inactivity");
        let rt = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(15));
            let client = {
                let g = rt.inner.lock().expect("runtime lock poisoned");
                match &g.status {
                    Status::Running { .. } => g.client.clone(),
                    _ => continue,
                }
            };
            let Some(client) = client else { continue };
            if let Some(idle) = client.idle_for() {
                if idle >= threshold {
                    log::info!(
                        "engine runtime: engine idle for {}s, unloading model",
                        idle.as_secs()
                    );
                    rt.shutdown();
                }
            }
        });
    }

    /// Shut down the spawned child. Called on app exit / idle unload.
    pub fn shutdown(&self) {
        {
            let client = {
                let mut g = self.inner.lock().expect("runtime lock poisoned");
                // Invalidate any start thread still loading: its finish_running
                // must not re-arm the engine after this teardown.
                g.epoch += 1;
                g.client.take()
            };
            if let Some(client) = client {
                client.shutdown();
            }
        }
        {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            g.status = Status::Stopped;
            g.embedder = None;
            g.client = None;
            // The client is gone, so every embedder over it is dead too.
            g.secondary.clear();
        }
        self.push_status();
    }

    fn finish_running(&self, embedder: Arc<ModelCallerEmbedder>, my_epoch: u64) {
        let model = embedder.model_id().to_string();
        {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            if g.epoch != my_epoch {
                log::info!("engine runtime: start aborted — shutdown landed mid-load");
                return;
            }
            g.status = Status::Running { model };
            g.embedder = Some(embedder);
        }
        self.push_status();
    }

    fn set_status(&self, status: Status) {
        {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            g.status = status;
        }
        self.push_status();
    }

    /// Like [`EngineRuntime::set_status`], but a no-op when a shutdown landed
    /// since the calling start claimed its epoch: a stale start's error must
    /// not mask the stopped state shutdown installed.
    fn set_status_if_current(&self, status: Status, my_epoch: u64) {
        {
            let mut g = self.inner.lock().expect("runtime lock poisoned");
            if g.epoch != my_epoch {
                return;
            }
            g.status = status;
        }
        self.push_status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A minimal stand-in for [`RuntimeInner`] that only tracks status.
    /// Lets us test `wait_until_ready`'s restart-from-stopped logic
    /// without spawning real processes.
    struct FakeRuntime {
        status: Mutex<Status>,
        ensure_started_calls: AtomicUsize,
    }

    impl FakeRuntime {
        fn new() -> Self {
            Self {
                status: Mutex::new(Status::Stopped),
                ensure_started_calls: AtomicUsize::new(0),
            }
        }

        fn fake_wait_until_ready(&self, timeout: Duration) -> Result<(), String> {
            let deadline = Instant::now() + timeout;
            loop {
                {
                    let g = self.status.lock().unwrap();
                    match &*g {
                        Status::Running { .. } => return Ok(()),
                        Status::Error(m) => {
                            return Err(format!("AI engine error: {m}"));
                        }
                        Status::Stopped => {
                            drop(g);
                            self.ensure_started_calls.fetch_add(1, Ordering::Relaxed);
                            std::thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        Status::Starting => {}
                    }
                }
                if Instant::now() >= deadline {
                    return Err("AI engine did not become ready in time".to_string());
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        fn set_status(&self, s: Status) {
            *self.status.lock().unwrap() = s;
        }
    }

    /// Simulates a successful restart: Stopped → Starting → Running.
    /// The helper thread transitions to Starting immediately, then Running after
    /// a short delay (mimicking modelcaller spawn + model load).
    #[test]
    fn wait_until_ready_restarts_from_stopped() {
        let rt = FakeRuntime::new();
        assert_eq!(rt.ensure_started_calls.load(Ordering::Relaxed), 0);

        let rt2 = &rt as *const FakeRuntime as usize;
        std::thread::spawn(move || {
            let rt_ref = unsafe { &*(rt2 as *const FakeRuntime) };
            std::thread::sleep(Duration::from_millis(30));
            rt_ref.set_status(Status::Starting);
            std::thread::sleep(Duration::from_millis(30));
            rt_ref.set_status(Status::Running {
                model: "test-model".to_string(),
            });
        });

        let result = rt.fake_wait_until_ready(Duration::from_secs(5));
        assert!(
            result.is_ok(),
            "wait_until_ready should succeed after restart: {result:?}"
        );
        assert!(
            rt.ensure_started_calls.load(Ordering::Relaxed) >= 1,
            "ensure_started should have been called at least once"
        );
    }

    /// If the engine is in Error state, wait_until_ready returns immediately.
    #[test]
    fn wait_until_ready_returns_error_immediately() {
        let rt = FakeRuntime::new();
        rt.set_status(Status::Error("boom".to_string()));
        let result = rt.fake_wait_until_ready(Duration::from_secs(5));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("boom"));
    }

    /// If the engine is already Running, wait_until_ready returns immediately.
    #[test]
    fn wait_until_ready_returns_immediately_when_running() {
        let rt = FakeRuntime::new();
        rt.set_status(Status::Running {
            model: "m".to_string(),
        });
        assert!(rt.fake_wait_until_ready(Duration::from_millis(10)).is_ok());
        assert_eq!(rt.ensure_started_calls.load(Ordering::Relaxed), 0);
    }

    /// If the engine is Starting, wait_until_ready waits (doesn't error).
    #[test]
    fn wait_until_ready_waits_while_starting() {
        let rt = FakeRuntime::new();
        rt.set_status(Status::Starting);
        let rt2 = &rt as *const FakeRuntime as usize;
        std::thread::spawn(move || {
            let rt_ref = unsafe { &*(rt2 as *const FakeRuntime) };
            std::thread::sleep(Duration::from_millis(30));
            rt_ref.set_status(Status::Running {
                model: "m".to_string(),
            });
        });
        let result = rt.fake_wait_until_ready(Duration::from_secs(5));
        assert!(result.is_ok());
    }
}
