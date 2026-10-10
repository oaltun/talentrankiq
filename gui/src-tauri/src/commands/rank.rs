//! Progressive-ranking cascade Tauri commands (start/cancel/poll).
//!
//! The cascade runs on the serve process's background thread. The gui drives it
//! via `start_rank` (returns immediately with initial RankState), polls
//! `rank-progress` events, and can call `cancel_rank` to stop it.
//! `get_rank_state` is the synchronous getter for the current RankState snapshot.

use libtriqui_client::params::*;
use libtriqui_client::types::{JobUiState, RankState, SystemBudget};
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking, rpc_blocking_typed};
use crate::types::DomainError;

/// Backend-derived enablement for a job's Sort/Re-rank buttons: `sort_allowed`,
/// dirtiness, and a reason code (`no-cvs` | `no-jobad` | `engine-not-ready` |
/// `stale` | `clean`) the gui only i18n-maps. Pulled, never recomputed.
#[tauri::command]
#[specta::specta]
pub async fn job_ui_state(app: AppHandle, job_id: String) -> Result<JobUiState, DomainError> {
    rpc_blocking_typed::<JobUiState>(app, "job_ui_state", GetJobDetailParams { job_id })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn start_rank(app: AppHandle, job_id: String) -> Result<RankState, DomainError> {
    rpc_blocking_typed::<RankState>(app, "start_rank", StartRankParams { job_id })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn cancel_rank(app: AppHandle, job_id: String) -> Result<(), DomainError> {
    rpc_blocking(app, "cancel_rank", CancelRankParams { job_id })
        .await
        .map(|_| ())
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn get_rank_state(app: AppHandle, job_id: String) -> Result<RankState, DomainError> {
    rpc_blocking_typed::<RankState>(app, "get_rank_state", GetJobDetailParams { job_id })
        .await
        .map_err(cli::domain_err)
}

/// The current system-capacity budget (libsyscap): what the machine allows
/// right now. `holding: true` is the only state the gui must surface — a
/// visible notice while the ranking waves are paused.
#[tauri::command]
#[specta::specta]
pub async fn system_budget(app: AppHandle) -> Result<SystemBudget, DomainError> {
    rpc_blocking_typed::<SystemBudget>(app, "system_budget", ())
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn mark_candidate_viewed(
    app: AppHandle,
    candidate_id: String,
) -> Result<(), DomainError> {
    rpc_blocking(
        app,
        "mark_candidate_viewed",
        MarkCandidateViewedParams { candidate_id },
    )
    .await
    .map(|_| ())
    .map_err(cli::domain_err)
}
