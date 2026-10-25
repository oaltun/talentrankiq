//! ONNX embedder management for the Models activity (was gui
//! `commands/models.rs`). The curated list comes from `triq.yaml`
//! `gui_embedders:`; downloads run in-process through the `modeldownloader`
//! library with typed progress events (no helper binary, no stderr parsing —
//! thin gui pattern, `gui/design/app1/thin-gui.md`); on-disk state lives in the
//! `libmodelstorage` cache layout. Progress goes through the serve
//! [`EventSink`] instead of Tauri events; the benchmark CVs for `try_model`
//! are read from `<app_dir>/benchmark-cvs` (extracted there by the gui at
//! setup).

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{
    CosinePair, GuiEmbedder, ModelDownloadProgress, ModelKind, TryModelProgress, TryModelResult,
};
use libtriqui_core::app_config;
use libtriqui_core::embed::Embedder;
use libmodeldownloader::cli::Artifact;
use libmodeldownloader::modelcache::verify_safetensors_present;

use crate::serve::handlers::prefs::{load_app_preferences, save_app_preferences};
use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

/// Bounded wait for cancelled cascade runners to retire during an embedder
/// switch. A cancel lands at the next wave boundary, and the scout stage runs
/// before the first check, so a wave in flight persists its vectors when it
/// lands — the vector nuke must wait for the threads, not just the flag.
const CASCADE_RETIRE_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// List the curated embedders (`gui_embedders`) with download/default state.
pub fn list_embedders(state: &Arc<ServeState>) -> Result<serde_json::Value, ProtocolError> {
    let cfg = app_config::load_triq();
    let model_dir = app_config::model_dir();
    let revision = app_config::revision();
    let prefs = load_app_preferences(state).embedding_model;

    let model_map: std::collections::HashMap<_, _> =
        cfg.models.iter().map(|m| (m.slug.as_str(), m)).collect();

    let mut out = Vec::new();
    // The active model is the saved preference, else the `triq.yaml` default.
    let active_slug = prefs.as_deref().or_else(|| {
        cfg.gui_embedders
            .iter()
            .find(|ge| ge.default)
            .map(|ge| ge.slug.as_str())
    });
    for ge in &cfg.gui_embedders {
        let model_def = model_map
            .get(ge.slug.as_str())
            .ok_or_else(|| string_err(format!("gui_embedders slug '{}' not in models", ge.slug)))?;
        
        // Each model entry has only one artifact (Ort OR Mlx)
        let artifact = model_def.artifacts.first()
            .ok_or_else(|| string_err(format!("model '{}' has no artifact", ge.slug)))?;
        
        let downloaded = match artifact.backend {
            libconfigreader::BackendKind::Ort => {
                libmodelstorage::onnx_exists(&model_dir, &ge.slug, &revision)
            }
            libconfigreader::BackendKind::Mlx => {
                let target = libmodelstorage::slug_dir(&model_dir, &ge.slug).join(&revision);
                verify_safetensors_present(&target)
            }
        };
        
        let size = if downloaded {
            dir_size(&libmodelstorage::slug_dir(&model_dir, &ge.slug))
        } else {
            0
        };
        out.push(GuiEmbedder {
            slug: ge.slug.clone(),
            name: ge.name.clone(),
            memory_mb: ge.memory_mb,
            downloaded,
            is_default: active_slug == Some(ge.slug.as_str()),
            dim: model_def.dim,
            max_context_window: model_def.max_context_window,
            source: model_def
                .artifacts
                .first()
                .map(|a| a.source.clone())
                .unwrap_or_default(),
            size_bytes: size,
        });
    }
    ok_json(out)
}

/// Recursively sum file sizes under a directory.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = fs::metadata(&p) {
                total += m.len();
            }
        }
    }
    total
}

/// Emit a `model-download-progress` event so the UI can show a spinner/bar.
fn emit_progress(state: &ServeState, slug: &str, filename: &str, done: u64, total: Option<u64>) {
    state.events.emit(
        "model-download-progress",
        &ModelDownloadProgress {
            repo_id: slug.to_string(),
            filename: filename.to_string(),
            done,
            total,
        },
    );
}

/// Running aggregate of typed [`libmodeldownloader::DownloadEvent`] progress:
/// per-file known totals and bytes done, keyed by file name. Replaces the old
/// stderr log parsing — events carry exact byte counts.
#[derive(Default)]
struct DownloadProgressAggregate {
    totals: std::collections::HashMap<String, u64>,
    dones: std::collections::HashMap<String, u64>,
    current: Option<String>,
}

impl DownloadProgressAggregate {
    fn done(&self) -> u64 {
        self.dones.values().sum()
    }
    fn total(&self) -> Option<u64> {
        let total: u64 = self.totals.values().sum();
        (total > 0).then_some(total)
    }
    /// Fold one event; true when a progress re-emit is warranted. `Started`
    /// resets the file's done-bytes (a fresh attempt/retry restarts it).
    fn apply(&mut self, ev: libmodeldownloader::DownloadEvent) -> bool {
        match ev {
            libmodeldownloader::DownloadEvent::Started { file, total } => {
                let label = file_label(&file);
                if let Some(t) = total {
                    self.totals.insert(label.clone(), t);
                }
                self.dones.insert(label.clone(), 0);
                self.current = Some(label);
                false
            }
            libmodeldownloader::DownloadEvent::Progress { file, done, total } => {
                let label = file_label(&file);
                if let Some(t) = total {
                    self.totals.insert(label.clone(), t);
                }
                self.dones.insert(label.clone(), done);
                self.current = Some(label);
                true
            }
            libmodeldownloader::DownloadEvent::Finished { file, total } => {
                let label = file_label(&file);
                let done = total
                    .or_else(|| self.totals.get(&label).copied())
                    .or_else(|| self.dones.get(&label).copied())
                    .unwrap_or(0);
                if done > 0 {
                    self.totals.entry(label.clone()).or_insert(done);
                }
                self.dones.insert(label, done);
                self.current = None;
                true
            }
            libmodeldownloader::DownloadEvent::Cached { file, bytes } => {
                let label = file_label(&file);
                self.totals.entry(label.clone()).or_insert(bytes);
                self.dones.insert(label, bytes);
                true
            }
        }
    }
}

/// Last path segment as the progress label (repo paths and dest paths both).
fn file_label(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// Which ONNX/MLX artifacts are present for an embedder slug, and which arm
/// `backend: auto` would resolve to. Engine-free (registry-side resolve +
/// on-disk presence), so the gui can ask before it consents to a download.
pub fn model_variants(
    _state: &Arc<ServeState>,
    params: &libtriqui_client::params::ModelVariantsParams,
) -> Result<Value, ProtocolError> {
    let cfg = app_config::load_triq();
    let model_dir = app_config::model_dir();
    let model_def = cfg
        .models
        .iter()
        .find(|m| m.slug == params.model)
        .ok_or_else(|| string_err(format!("model '{}' not in models registry", params.model)))?;
    let registry = app_config::registry_ctx();
    let status = app_config::variant_status(
        &registry.registry,
        registry.backend,
        &model_dir,
        registry.revision.as_str(),
        &params.model,
    )
    .map_err(|e| string_err(format!("resolve {}: {e}", params.model)))?;
    // ONNX presence = the file `resolve_onnx` prefers (URL mode lands under
    // `<slug>/repo/`) on disk. MLX presence comes from the variant status.
    let model_dir = app_config::model_dir();
    
    // Find the ONNX artifact to get the filename
    let onnx_art = model_def
        .artifacts
        .iter()
        .find(|a| a.backend == libconfigreader::BackendKind::Ort);
    let onnx_file = onnx_art
        .map(|a| a.url.rsplit('/').next().unwrap_or(&a.url).to_string())
        .unwrap_or_default();
    let onnx_present = std::path::Path::new(&model_dir)
        .join(&params.model)
        .join("repo")
        .join(&onnx_file)
        .exists();
    
    let has_mlx = model_def
        .artifacts
        .iter()
        .any(|a| a.backend == libconfigreader::BackendKind::Mlx);
    ok_json(libtriqui_client::types::EmbedderVariants {
        model: params.model.clone(),
        runtime: status.runtime.to_string(),
        present: status.present,
        onnx_present,
        mlx_present: if has_mlx { Some(status.present) } else { None },
    })
}

/// Download an embedder (ONNX or MLX) via `modeldownloader`, emitting progress events.
/// If the engine is running with a different model loaded, that model is
/// unloaded first to minimise RAM pressure during the download.
pub fn download_embedder(state: &Arc<ServeState>, slug: &str) -> Result<(), ProtocolError> {
    log::info!("download_embedder called for slug: {}", slug);
    if let Ok(client) = state.runtime.client() {
        if let Ok(embedder) = state.runtime.embedder() {
            let current = embedder.model_id().to_string();
            if current != slug {
                let _ = client.unload(&current);
            }
        }
    }

    let cfg = app_config::load_triq();
    let model_def = cfg
        .models
        .iter()
        .find(|m| m.slug == slug)
        .ok_or_else(|| string_err(format!("model '{slug}' not in models registry")))?;
    
    // Each model entry now has only one artifact (Ort OR Mlx)
    let artifact = model_def
        .artifacts
        .first()
        .ok_or_else(|| string_err(format!("model '{slug}' has no artifact")))?;
    
    log::info!("download_embedder: artifact backend = {:?}", artifact.backend);
    
    match artifact.backend {
        libconfigreader::BackendKind::Ort => {
            // ONNX uses URL mode
            log::info!("download_embedder: running ONNX download for {}", slug);
            run_embedder_download(
                state, 
                slug, 
                &artifact.url, 
                artifact.sha256.clone(), 
                &artifact.extra_files
            )
        }
        libconfigreader::BackendKind::Mlx => {
            // MLX uses repo mode
            log::info!("download_embedder: running MLX download for {}", slug);
            run_mlx_download(state, slug, artifact)
        }
    }
}

/// The one shared embedder download (the Models activity AND the startup
/// modal both land here). Runs `modeldownloader` **in-process** (library
/// call — thin gui pattern, `gui/design/app1/thin-gui.md`): URL mode pulls the
/// external-weights `_data` siblings of the chosen ONNX variant (hf: source
/// mode would only grab the graph shell `model.onnx`, leaving the model
/// unloadable) and lands everything under `<slug>/repo/`, the layout
/// `resolve_onnx` prefers. Typed progress events are aggregated and
/// re-emitted as `model-download-progress` events so the gui bar moves.
fn run_embedder_download(
    state: &ServeState,
    slug: &str,
    onnx_url: &str,
    sha256: Option<String>,
    extra_files: &[String],
) -> Result<(), ProtocolError> {
    let model_dir = app_config::model_dir();

    // Re-check right before downloading: on a fresh install the engine's
    // modelcaller auto-download may race this call and complete first, and a
    // client retry after a finished download would re-fetch needlessly.
    if libmodelstorage::onnx_exists(&model_dir, slug, &app_config::revision()) {
        emit_progress(state, slug, "", 1, Some(1));
        return Ok(());
    }

    emit_progress(state, slug, "", 0, None);

    let mut req = libmodeldownloader::UrlDownload::new(onnx_url, slug, model_dir, sha256);
    req.extra_files = extra_files.to_vec();
    // The progress callback must be 'static, so hand it an owned clone of the
    // event sink instead of borrowing `state`; the aggregate is shared so the
    // settled totals survive the call.
    let events = state.events.clone();
    let events_complete = events.clone();
    let slug_owned = slug.to_string();
    let agg = Arc::new(Mutex::new(DownloadProgressAggregate::default()));
    let agg_cb = Arc::clone(&agg);
    libmodeldownloader::download_url(req, move |ev| {
        let mut agg = agg_cb.lock().unwrap();
        if agg.apply(ev) {
            let filename = agg.current.clone().unwrap_or_default();
            let done = agg.done();
            let total = agg.total();
            log::info!("Emitting model-download-progress: slug={}, filename={}, done={}, total={:?}", slug_owned, filename, done, total);
            events.emit(
                "model-download-progress",
                &ModelDownloadProgress {
                    repo_id: slug_owned.clone(),
                    filename,
                    done,
                    total,
                },
            );
        }
    })
    .map_err(|e| string_err(format!("modeldownloader failed for '{slug}': {e}")))?;

    // Settle the bar at 100% (per-file bookkeeping may be a few bytes shy).
    let agg = agg.lock().unwrap();
    let done: u64 = agg.done();
    let total = agg.total().unwrap_or(done).max(done);
    emit_progress(state, slug, "", total, Some(total));

    // Emit completion event with the saved path so the GUI can notify the user.
    let model_dir = app_config::model_dir();
    let save_path = libmodelstorage::slug_dir(&model_dir, slug);
    events_complete.emit(
        "model-download-complete",
        &libtriqui_client::types::ModelDownloadComplete {
            slug: slug.to_string(),
            path: save_path.to_string_lossy().to_string(),
        },
    );
    Ok(())
}

/// Remove a downloaded ONNX embedder from the model cache.
pub fn remove_embedder(slug: &str) -> Result<(), ProtocolError> {
    let model_dir = app_config::model_dir();
    let dir = libmodelstorage::slug_dir(&model_dir, slug);
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .map_err(|e| string_err(format!("failed to remove {slug}: {e}")))?;
    }
    Ok(())
}

/// Download an MLX embedder via `modeldownloader` (repo mode), emitting progress events.
/// MLX artifacts are fetched from a HuggingFace repo listing (all .safetensors shards + metadata).
fn run_mlx_download(
    state: &ServeState,
    slug: &str,
    artifact: &libconfigreader::Artifact,
) -> Result<(), ProtocolError> {
    let model_dir = app_config::model_dir();
    let revision = app_config::revision();
    
    // Check if already present
    let target = libmodelstorage::slug_dir(&model_dir, slug).join(&revision);
    if verify_safetensors_present(&target) {
        emit_progress(state, slug, "", 1, Some(1));
        return Ok(());
    }
    
    emit_progress(state, slug, "", 0, None);
    
    // Use modeldownloader in repo mode with Artifact::Mlx
    let cli = libmodeldownloader::cli::Cli {
        config: None,
        model: Some(slug.to_string()),
        source: Some(artifact.source.clone()),
        url: None,
        slug: Some(slug.to_string()),
        revision: revision.clone(),
        model_dir: Some(model_dir.clone()),
        force: false,
        dry_run: false,
        files: vec![],
        extra_files: artifact.extra_files.clone(),
        artifact: Artifact::Mlx,
        sha256: artifact.sha256.clone(),
        verbose: app_config::verbose(),
        logfile: None,
    };
    
    let events = state.events.clone();
    let events_complete = events.clone();
    let slug_owned = slug.to_string();
    let agg = Arc::new(Mutex::new(DownloadProgressAggregate::default()));
    let agg_cb = Arc::clone(&agg);
    
    let progress_sink: libmodeldownloader::ProgressSink = Arc::new(Mutex::new(move |ev| {
        let mut agg = agg_cb.lock().unwrap();
        if agg.apply(ev) {
            let filename = agg.current.clone().unwrap_or_default();
            let done = agg.done();
            let total = agg.total();
            log::info!("Emitting MLX model-download-progress: slug={}, filename={}, done={}, total={:?}", slug_owned, filename, done, total);
            events.emit(
                "model-download-progress",
                &ModelDownloadProgress {
                    repo_id: slug_owned.clone(),
                    filename,
                    done,
                    total,
                },
            );
        }
    }));
    
    libmodeldownloader::Downloader::run_with_progress(cli, Some(progress_sink))
        .map_err(|e| string_err(format!("modeldownloader failed for MLX '{slug}': {e}")))?;
    
    // Settle the bar at 100%
    let agg = agg.lock().unwrap();
    let done: u64 = agg.done();
    let total = agg.total().unwrap_or(done).max(done);
    emit_progress(state, slug, "", total, Some(total));
    
    // Emit completion event
    let save_path = libmodelstorage::slug_dir(&model_dir, slug);
    events_complete.emit(
        "model-download-complete",
        &libtriqui_client::types::ModelDownloadComplete {
            slug: slug.to_string(),
            path: save_path.to_string_lossy().to_string(),
        },
    );
    Ok(())
}

const BENCHMARK_LABELS: &[&str] = &[
    "CV 1 — Geologist",
    "CV 2 — IT Professional",
    "CV 3 — Admin Manager",
    "CV 4 — Teaching Professional",
    "CV 5 — Sales / Accountant",
    "CV 6 — SCM Logistics",
    "CV 7 — Academic Researcher",
    "CV 8 — Professor / Engineering",
    "CV 9 — HVAC Engineer",
    "CV 10 — Mechanical Engineer",
    "CV 11 — Mechanical Engineer [twin]",
];

/// The benchmark CVs for `try_model`: `*.txt` files under
/// `<app_dir>/benchmark-cvs`, name-sorted, labelled with [`BENCHMARK_LABELS`].
/// The gui extracts them there from its embedded copy at setup.
fn benchmark_cvs(state: &ServeState) -> Result<Vec<(String, String)>, String> {
    let dir = state.app_dir.join("benchmark-cvs");
    let entries = fs::read_dir(&dir)
        .map_err(|e| format!("benchmark CVs missing at {}: {e}", dir.display()))?;
    let mut files: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    let pairs: Vec<(String, String)> = files
        .into_iter()
        .map(|p| {
            let label = p
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("CV")
                .to_string();
            let content = fs::read_to_string(&p).unwrap_or_default();
            (label, content)
        })
        .collect();
    if pairs.is_empty() {
        return Err("no benchmark CVs found".into());
    }
    // Label by position when the extraction matches the canonical set; fall
    // back to the file stem otherwise.
    Ok(pairs
        .into_iter()
        .enumerate()
        .map(|(i, (label, content))| {
            (
                BENCHMARK_LABELS
                    .get(i)
                    .map(|l| l.to_string())
                    .unwrap_or(label),
                content,
            )
        })
        .collect())
}

struct TryModelLibEmbedder {
    client: libmodelcallerclient::ModelCallerClient,
    model: String,
}

impl libembedrank::Embedder for TryModelLibEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, libembedrank::EmbedRankError> {
        self.client
            .embed(&self.model, texts, None)
            .map_err(|e| libembedrank::EmbedRankError::Embed(e.to_string()))
    }
}

/// Embed the sample CVs using the given model, compute all pair-wise cosine
/// similarities, measure per-CV timing, and return the results. Blocking (the
/// dispatch layer runs each request on its own thread). Progress travels as
/// `try-model-progress` events.
pub fn try_model(state: &Arc<ServeState>, slug: &str) -> Result<serde_json::Value, ProtocolError> {
    state
        .runtime
        .wait_until_ready(std::time::Duration::from_secs(180))
        .map_err(|e| string_err(e.to_string()))?;
    let embedder = state
        .runtime
        .embedder()
        .map_err(|e| string_err(e.to_string()))?;

    let original_model = embedder.model_id().to_string();
    let client = state
        .runtime
        .client()
        .map_err(|e| string_err(e.to_string()))?;

    if slug != original_model {
        let _ = client.unload(&original_model);
        client
            .load(slug)
            .map_err(|e| string_err(format!("failed to load model '{slug}': {e}")))?;
    }

    let chunk_size = app_config::chunk_size_for_model(slug);
    let tokenizer = if chunk_size > 0 {
        let model_dir = app_config::model_dir();
        app_config::tokenizer_path(&model_dir, slug)
            .and_then(|p| libtokenizer::TokenizerHandle::from_file(&p).ok())
    } else {
        None
    };

    let cvs = benchmark_cvs(state).map_err(string_err)?;
    let labels: Vec<String> = cvs.iter().map(|(l, _)| l.clone()).collect();
    let texts: Vec<String> = cvs.iter().map(|(_, t)| t.clone()).collect();
    let total = texts.len();

    let mut per_cv_ms = Vec::with_capacity(total);
    let mut all_embeddings: Vec<Vec<f64>> = Vec::with_capacity(total);

    for (i, text) in texts.iter().enumerate() {
        state.events.emit(
            "try-model-progress",
            &TryModelProgress {
                cv_index: i,
                cv_total: total,
                cv_label: labels[i].clone(),
            },
        );
        let start = std::time::Instant::now();
        let emb = if let Some(tok) = &tokenizer {
            let config = libembedrank::ChunkingConfig::new(chunk_size);
            let libemb = TryModelLibEmbedder {
                client: client.clone(),
                model: slug.to_string(),
            };
            libembedrank::document_embedding(text, tok, &config, &libemb, "")
                .map_err(|e| string_err(format!("chunked embed failed: {e}")))?
        } else {
            client
                .embed(slug, &[text], None)
                .map_err(|e| string_err(format!("embed failed: {e}")))?
                .into_iter()
                .next()
                .unwrap_or_default()
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        per_cv_ms.push(elapsed_ms);
        all_embeddings.push(emb);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let mut sorted = per_cv_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median_ms = if sorted.len() % 2 == 0 && sorted.len() > 1 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };

    let mut pairs = Vec::new();
    for i in 0..all_embeddings.len() {
        for j in (i + 1)..all_embeddings.len() {
            let sim =
                libmodelcallerclient::cosine_similarity(&all_embeddings[i], &all_embeddings[j]);
            pairs.push(CosinePair {
                i,
                j,
                label_i: labels[i].clone(),
                label_j: labels[j].clone(),
                similarity: sim,
            });
        }
    }

    if slug != original_model {
        let _ = client.unload(slug);
        let _ = client.load(&original_model);
    }

    ok_json(TryModelResult {
        per_cv_ms,
        median_ms,
        pairs,
        labels,
    })
}

/// Set a model as the default for its kind (persisted to preferences).
pub fn set_default_model(
    state: &Arc<ServeState>,
    kind: ModelKind,
    repo_id: &str,
) -> Result<(), ProtocolError> {
    let mut prefs = load_app_preferences(state);
    if kind.is_embedding() {
        prefs.embedding_model = Some(repo_id.to_string());
    } else {
        prefs.inference_model = Some(repo_id.to_string());
    }
    save_app_preferences(state, prefs).map_err(string_err)
}

/// Switch the embedding model: persists the choice, wipes every stored vector
/// that belonged to the previous model (candidate chunk vectors + centroids,
/// job-ad + criteria vectors), and swaps the engine's model over. The engine
/// itself is left unloaded — the new model starts lazily on the first embed
/// that has a cache miss, so switching back to a model whose vectors are all
/// cached never loads anything into memory.
///
/// Candidate ROWS are kept — file paths, extracted text, and the `liked` stars
/// survive a model switch, and no re-import is needed: the wiped vectors are
/// re-embedded from the stored text on the next embed/sort. Learned feedback
/// vectors are per model (`job_feedback_vectors` / `profile_vectors` keyed by
/// model) and are left untouched — each model keeps its own, nothing is lost
/// or mixed. The `embedding_cache` is also kept: it is keyed by
/// (model, content_hash), so switching back to a previous model re-serves the
/// old vectors from cache without a single model call.
pub fn set_embedding_model(state: &Arc<ServeState>, slug: &str) -> Result<(), ProtocolError> {
    let cfg = app_config::load_triq();
    if !cfg.gui_embedders.iter().any(|ge| ge.slug == slug) {
        return Err(string_err(format!("unknown embedding model '{slug}'")));
    }
    
    // Check if model is downloaded (ONNX or MLX)
    let model_def = cfg.models.iter().find(|m| m.slug == slug)
        .ok_or_else(|| string_err(format!("model '{slug}' not in models registry")))?;
    
    let artifact = model_def.artifacts.first()
        .ok_or_else(|| string_err(format!("model '{slug}' has no artifact")))?;
    
    let model_dir = app_config::model_dir();
    let revision = app_config::revision();
    let downloaded = match artifact.backend {
        libconfigreader::BackendKind::Ort => {
            libmodelstorage::onnx_exists(&model_dir, slug, &revision)
        }
        libconfigreader::BackendKind::Mlx => {
            let target = libmodelstorage::slug_dir(&model_dir, slug).join(&revision);
            verify_safetensors_present(&target)
        }
    };
    
    if !downloaded {
        return Err(string_err(
            "Model not downloaded — download it in the Models screen first",
        ));
    }

    // No-op when it is already the saved choice.
    let mut prefs = load_app_preferences(state);
    if prefs.embedding_model.as_deref() == Some(slug) {
        return Ok(());
    }
    prefs.embedding_model = Some(slug.to_string());
    save_app_preferences(state, prefs).map_err(string_err)?;

    // Hot-swap first — it refuses while a job is embedding, so a switch never
    // destroys data mid-import. Status events carry the new model.
    state
        .runtime
        .switch_model(slug.to_string())
        .map_err(|e| string_err(e.to_string()))?;
    let conn = state
        .runtime
        .open_conn()
        .map_err(|e| string_err(format!("failed to open db: {e}")))?;
    let job_ids: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT id FROM jobs")
            .map_err(|e| string_err(format!("failed to list jobs: {e}")))?;
        let ids = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(|e| string_err(format!("failed to list jobs: {e}")))?
            .collect::<Result<Vec<String>, _>>()
            .map_err(|e| string_err(format!("failed to list jobs: {e}")))?;
        ids
    };

    // Switching the embedder invalidates every in-flight ranking (it is scoring
    // in the old space), so stop the cascades BEFORE the nuke below — and wait
    // for the runners to retire, not just for the cancel flag to be set: a
    // wave in flight (or the scout stage, which runs before the first cancel
    // check) persists its chunk vectors when it lands, and that write would
    // re-populate the table we are about to wipe with old-model rows no later
    // wave would ever overwrite.
    for job_id in &job_ids {
        if let Ok(true) = libtriqui_core::db::cancel_running_cascade(&conn, job_id) {
            log::info!("rank: cancelled the running cascade for {job_id} (embedder changed)");
        }
    }
    let deadline = std::time::Instant::now() + CASCADE_RETIRE_WAIT;
    while job_ids.iter().any(|id| state.cascade_gens.is_live(id)) {
        if std::time::Instant::now() >= deadline {
            log::warn!(
                "models: cascade runners still alive {}s after the embedder switch — \
                 wiping vectors anyway (a late wave may leave stale rows behind)",
                CASCADE_RETIRE_WAIT.as_secs()
            );
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Every stored vector belonged to the previous model: drop the chunk
    // vectors + stale rubric scores, reset the shortlists, and clear the ad +
    // criteria vectors for re-embedding. Candidate rows (files, extracted
    // text, liked stars) stay — they are model-independent, and their vectors
    // are rebuilt from the stored text by the next embed/sort.
    conn.execute("DELETE FROM candidate_chunks", [])
        .map_err(|e| string_err(format!("failed to clear chunk vectors: {e}")))?;
    conn.execute("DELETE FROM candidate_scores", [])
        .map_err(|e| string_err(format!("failed to clear rubric scores: {e}")))?;
    // The scout's space is untouched (its slug never changes on a switch), but the
    // heavy space's rubric vectors and every space's grades for this job's
    // candidates go: they are cosines of the vectors we just wiped.
    conn.execute("DELETE FROM rubric_vectors", [])
        .map_err(|e| string_err(format!("failed to clear rubric vectors: {e}")))?;
    conn.execute("DELETE FROM stage_scores", [])
        .map_err(|e| string_err(format!("failed to clear stage scores: {e}")))?;
    conn.execute("DELETE FROM judge_pool", [])
        .map_err(|e| string_err(format!("failed to clear the judge pool: {e}")))?;
    conn.execute(
        "UPDATE candidates SET vector = NULL, vector_model = NULL,
         rank = NULL, ranked_at = NULL, points = NULL, points_max = NULL",
        [],
    )
    .map_err(|e| string_err(format!("failed to reset candidate vectors: {e}")))?;
    conn.execute(
        "UPDATE jobs SET ead = NULL, ead_model = NULL, last_ranked_at = NULL,
         inputs_updated_at = datetime('now'), updated_at = datetime('now'), inputs_version = inputs_version + 1",
        [],
    )
    .map_err(|e| string_err(format!("failed to reset job vectors: {e}")))?;
    conn.execute("UPDATE criteria SET vector = NULL, vector_model = NULL", [])
        .map_err(|e| string_err(format!("failed to reset criterion vectors: {e}")))?;
    // Re-embed every job's ad + criteria with the new model in the background
    // (sequential, on one thread; prewarm waits for the engine).
    libtriqui_core::embeddings::prewarm_all_rubric_inputs(&state.runtime, job_ids);
    Ok(())
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    use libmodeldownloader::DownloadEvent as Ev;

    fn apply_all(agg: &mut DownloadProgressAggregate, evs: &[Ev]) {
        for ev in evs {
            agg.apply(ev.clone());
        }
    }

    #[test]
    fn full_download_sequence_aggregates_bytes() {
        let mut agg = DownloadProgressAggregate::default();
        apply_all(
            &mut agg,
            &[
                Ev::Started {
                    file: "onnx/model_q4.onnx".into(),
                    total: Some(629145600),
                },
                Ev::Progress {
                    file: "onnx/model_q4.onnx".into(),
                    done: 314572800,
                    total: Some(629145600),
                },
                Ev::Finished {
                    file: "onnx/model_q4.onnx".into(),
                    total: Some(629145600),
                },
                Ev::Started {
                    file: "onnx/tokenizer.json".into(),
                    total: Some(17_000_000),
                },
                Ev::Finished {
                    file: "onnx/tokenizer.json".into(),
                    total: Some(17_000_000),
                },
            ],
        );
        assert_eq!(agg.done(), 629145600 + 17_000_000);
        assert_eq!(agg.total(), Some(629145600 + 17_000_000));
    }

    #[test]
    fn cached_files_count_as_done() {
        let mut agg = DownloadProgressAggregate::default();
        apply_all(
            &mut agg,
            &[
                Ev::Cached {
                    file: "onnx/tokenizer.json".into(),
                    bytes: 12345,
                },
                Ev::Started {
                    file: "model_q4.onnx".into(),
                    total: Some(100),
                },
                Ev::Progress {
                    file: "model_q4.onnx".into(),
                    done: 100,
                    total: Some(100),
                },
                Ev::Finished {
                    file: "model_q4.onnx".into(),
                    total: Some(100),
                },
            ],
        );
        assert!(agg.done() >= 12345);
        assert!(agg.total().unwrap() >= 12345);
    }

    #[test]
    fn unknown_total_download_still_tracks_done() {
        // No content-length: events carry total = None.
        let mut agg = DownloadProgressAggregate::default();
        apply_all(
            &mut agg,
            &[
                Ev::Started {
                    file: "config.json".into(),
                    total: None,
                },
                Ev::Finished {
                    file: "config.json".into(),
                    total: None,
                },
            ],
        );
        assert_eq!(agg.done(), 0); // no sizes known anywhere
        assert_eq!(agg.total(), None);
    }

    #[test]
    fn fresh_download_resets_previous_done() {
        let mut agg = DownloadProgressAggregate::default();
        apply_all(
            &mut agg,
            &[
                Ev::Started {
                    file: "m.onnx".into(),
                    total: Some(1000),
                },
                Ev::Progress {
                    file: "m.onnx".into(),
                    done: 900,
                    total: Some(1000),
                },
            ],
        );
        assert!(agg.done() > 0);
        // A retry restarts the file: Started resets its done-bytes.
        agg.apply(Ev::Started {
            file: "m.onnx".into(),
            total: Some(1000),
        });
        assert_eq!(agg.dones.get("m.onnx"), Some(&0));
    }
}
