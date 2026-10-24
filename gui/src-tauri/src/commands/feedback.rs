//! Thin shell over `triqui-cli serve` rpcs (feedback) — see `jobs.rs`.

use crate::triqi_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{DomainError, TriqiItem};

/// Apply staged feedback (a reordered triqui + a desired liked-id set) to the learned
/// vectors, persist liked flags, then re-rank and return the new triqui.
#[tauri::command]
#[specta::specta]
pub async fn apply_feedback(
    app: AppHandle,
    job_id: String,
    ordered_candidate_ids: Vec<String>,
    liked_candidate_ids: Vec<String>,
) -> Result<Vec<TriqiItem>, DomainError> {
    rpc_blocking_typed::<Vec<TriqiItem>>(
        app,
        "apply_feedback",
        ApplyFeedbackParams {
            job_id,
            ordered_candidate_ids,
            liked_candidate_ids,
        },
    )
    .await
    .map_err(cli::domain_err)
}

/// Zero the learned feedback vectors + clear liked flags, then re-rank.
#[tauri::command]
#[specta::specta]
pub async fn reset_feedback(
    app: AppHandle,
    job_id: String,
) -> Result<Vec<TriqiItem>, DomainError> {
    rpc_blocking_typed::<Vec<TriqiItem>>(app, "reset_feedback", ResetFeedbackParams { job_id })
        .await
        .map_err(cli::domain_err)
}

/// All liked candidate ids for a job (seeds the frontend staging store).
#[tauri::command]
#[specta::specta]
pub async fn list_liked_candidate_ids(
    app: AppHandle,
    job_id: String,
) -> Result<Vec<String>, DomainError> {
    rpc_blocking_typed::<Vec<String>>(
        app,
        "list_liked_candidate_ids",
        ListLikedCandidateIdsParams { job_id },
    )
    .await
    .map_err(cli::domain_err)
}
