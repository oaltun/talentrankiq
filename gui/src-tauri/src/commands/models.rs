//! Thin shell over `triqui-cli serve` rpcs (ONNX embedder management) —
//! see `jobs.rs`. Downloads, the benchmark, and model switching all run
//! inside the serve process; progress arrives as `model-download-progress` /
//! `try-model-progress` events. `Result<_, String>` signatures are preserved.

use crate::cli::{rpc_blocking, rpc_blocking_typed, str_err};
use crate::triqi_client::params::*;
use crate::types::{DownloadStatus, EmbedderVariants, GuiEmbedder, ModelKind, TryModelResult};
use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub async fn list_embedders(app: AppHandle) -> Result<Vec<GuiEmbedder>, String> {
    rpc_blocking_typed::<Vec<GuiEmbedder>>(app, "list_embedders", EmptyParams {})
        .await
        .map_err(str_err)
}

/// Which arm (`backend: auto`) would serve an embedder and which artifacts are
/// on disk — engine-free, so the gui can ask before it consents to a download.
#[tauri::command]
#[specta::specta]
pub async fn model_variants(app: AppHandle, model: String) -> Result<EmbedderVariants, String> {
    rpc_blocking_typed::<EmbedderVariants>(app, "model_variants", ModelVariantsParams { model })
        .await
        .map_err(str_err)
}

/// Download an ONNX embedder via `modeldownloader` (serve side), emitting progress events.
/// If the engine is running with a different model loaded, that model is unloaded first
/// to minimise RAM pressure during the download.
#[tauri::command]
#[specta::specta]
pub async fn download_embedder(app: AppHandle, slug: String) -> Result<(), String> {
    rpc_blocking(app, "download_embedder", DownloadEmbedderParams { slug })
        .await
        .map(|_| ())
        .map_err(str_err)
}

/// Remove a downloaded ONNX embedder from the model cache.
#[tauri::command]
#[specta::specta]
pub async fn remove_embedder(app: AppHandle, slug: String) -> Result<(), String> {
    rpc_blocking(app, "remove_embedder", RemoveEmbedderParams { slug })
        .await
        .map(|_| ())
        .map_err(str_err)
}

/// Embed the sample CVs using the given model, compute all pair-wise cosine
/// similarities, measure per-CV timing, and return the results. Runs inside
/// the serve process; per-CV progress arrives as `try-model-progress`.
#[tauri::command]
#[specta::specta]
pub async fn try_model(app: AppHandle, slug: String) -> Result<TryModelResult, String> {
    rpc_blocking_typed::<TryModelResult>(app, "try_model", TryModelParams { slug })
        .await
        .map_err(str_err)
}

/// Set a model as the default for its kind (persisted to preferences).
#[tauri::command]
#[specta::specta]
pub async fn set_default_model(
    app: AppHandle,
    kind: ModelKind,
    repo_id: String,
) -> Result<(), String> {
    rpc_blocking(
        app,
        "set_default_model",
        SetDefaultModelParams { kind, repo_id },
    )
    .await
    .map(|_| ())
    .map_err(str_err)
}

/// Switch the embedding model: persists the choice, then wipes every job's
/// candidates (stored CV vectors only match the model that produced them —
/// re-import is needed), clears the job-ad + criteria vectors so they re-embed
/// with the new model, and hot-swaps the engine over to it. Learned feedback
/// vectors are per model: each model keeps its own, nothing is lost or mixed.
#[tauri::command]
#[specta::specta]
pub async fn set_embedding_model(app: AppHandle, slug: String) -> Result<(), String> {
    rpc_blocking(app, "set_embedding_model", SetEmbeddingModelParams { slug })
        .await
        .map(|_| ())
        .map_err(str_err)
}

/// Poll the current download status for a model slug.
#[tauri::command]
#[specta::specta]
pub async fn download_status(app: AppHandle, slug: String) -> Result<DownloadStatus, String> {
    rpc_blocking_typed::<DownloadStatus>(app, "download_status", DownloadStatusParams { slug })
        .await
        .map_err(str_err)
}

/// Check if download is complete for a model slug (yes/no).
#[tauri::command]
#[specta::specta]
pub async fn download_complete(app: AppHandle, slug: String) -> Result<bool, String> {
    rpc_blocking_typed::<bool>(app, "download_complete", DownloadStatusParams { slug })
        .await
        .map_err(str_err)
}
