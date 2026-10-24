//! Thin shell over `triqui-cli serve` rpcs (embedding engine) — see
//! `jobs.rs`. The engine runs inside the serve process; `start_runtime` just
//! asks it to connect-or-spawn modelcaller.

use crate::triqi_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking, rpc_blocking_typed};
use crate::types::{ChunkMatchResult, ChunkMatchTarget, DomainError, RuntimeStatusInfo};

/// Kick off the embedding engine (connect-or-spawn) if it hasn't already started.
/// Called by the frontend once React has loaded, so launch isn't blocked waiting
/// for model loading. Embedding/Sort lazily wait for it too, as a safety net.
#[tauri::command]
#[specta::specta]
pub fn start_runtime(app: AppHandle) {
    if let Err(e) = crate::cli::rpc(&app, "start_runtime", &EmptyParams {}) {
        log::warn!("start_runtime rpc failed: {e}");
    }
}

/// Snapshot of the embedding engine state for the UI.
#[tauri::command]
#[specta::specta]
pub async fn runtime_status(app: AppHandle) -> Result<RuntimeStatusInfo, DomainError> {
    rpc_blocking_typed::<RuntimeStatusInfo>(app, "runtime_status", EmptyParams {})
        .await
        .map_err(cli::domain_err)
}

/// Backfill embeddings for a job (e.g. after the engine becomes ready). Runs in the
/// background inside the serve process; progress arrives via `job-embedding-progress`.
#[tauri::command]
#[specta::specta]
pub async fn embed_job(app: AppHandle, job_id: String) -> Result<(), DomainError> {
    rpc_blocking(app, "embed_job", EmbedJobParams { job_id })
        .await
        .map(|_| ())
        .map_err(cli::domain_err)
}

/// Match Lens (CV Inspector): chunk a candidate's CV at a user-set token
/// size, embed every chunk, and cosine-score each against `target` (the job
/// ad or one criterion). Returns EVERY chunk with char offsets + similarity;
/// the frontend picks the top-n and ranks.
#[tauri::command]
#[specta::specta]
pub async fn chunk_match(
    app: AppHandle,
    candidate_id: String,
    target: ChunkMatchTarget,
    chunk_size: usize,
) -> Result<ChunkMatchResult, DomainError> {
    rpc_blocking_typed::<ChunkMatchResult>(
        app,
        "chunk_match",
        ChunkMatchParams {
            candidate_id,
            target,
            chunk_size,
        },
    )
    .await
    .map_err(cli::domain_err)
}
