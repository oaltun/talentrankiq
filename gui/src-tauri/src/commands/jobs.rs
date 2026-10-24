//! Thin shell over `triqui-cli serve` rpcs: `crate::cli` handles spawn +
//! demux. The domain logic lives behind the cli (thin gui pattern) — each
//! command here only translates its Tauri arguments into the shared
//! `libtriqui_client` params and rehydrates the error type the frontend
//! expects, so the generated bindings are unchanged.

use crate::triqi_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking, rpc_blocking_typed};
use crate::types::{DomainError, Job, JobDetail, JobSummary};

#[tauri::command]
#[specta::specta]
pub async fn create_job(app: AppHandle) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(app, "create_job", EmptyParams {})
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn list_job_summaries(app: AppHandle) -> Result<Vec<JobSummary>, DomainError> {
    rpc_blocking_typed::<Vec<JobSummary>>(app, "list_job_summaries", EmptyParams {})
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn get_job_detail(app: AppHandle, job_id: String) -> Result<JobDetail, DomainError> {
    rpc_blocking_typed::<JobDetail>(app, "get_job_detail", GetJobDetailParams { job_id })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn update_job(
    app: AppHandle,
    job_id: String,
    title: String,
    job_ad_text: Option<String>,
    triqui_size: i32,
) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(
        app,
        "update_job",
        UpdateJobParams {
            job_id,
            title,
            job_ad_text,
            triqui_size,
        },
    )
    .await
    .map_err(cli::domain_err)
}

/// Update the rubric weights (job-ad + the two feedback rows). Weights are ranking
/// inputs, so this bumps `inputs_updated_at` (drives the stale badge).
#[tauri::command]
#[specta::specta]
pub async fn update_job_weights(
    app: AppHandle,
    job_id: String,
    ad_weight: f64,
    fb_job_weight: f64,
    fb_profile_weight: f64,
) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(
        app,
        "update_job_weights",
        UpdateJobWeightsParams {
            job_id,
            ad_weight,
            fb_job_weight,
            fb_profile_weight,
        },
    )
    .await
    .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn archive_job(
    app: AppHandle,
    job_id: String,
    archived: bool,
) -> Result<(), DomainError> {
    rpc_blocking(app, "archive_job", ArchiveJobParams { job_id, archived })
        .await
        .map(|_| ())
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn empty_candidates(app: AppHandle, job_id: String) -> Result<i32, DomainError> {
    rpc_blocking_typed::<i32>(app, "empty_candidates", EmptyCandidatesParams { job_id })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn delete_job(app: AppHandle, job_id: String) -> Result<(), DomainError> {
    rpc_blocking(app, "delete_job", DeleteJobParams { job_id })
        .await
        .map(|_| ())
        .map_err(cli::domain_err)
}

/// Set (or clear, with an empty string) the job's watched CV directory and
/// restart its watcher. Creating the directory is attempted by the watcher so
/// a typed fresh path just works.
#[tauri::command]
#[specta::specta]
pub async fn set_job_cv_dir(
    app: AppHandle,
    job_id: String,
    directory: String,
) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(
        app,
        "set_job_cv_dir",
        SetJobCvDirParams { job_id, directory },
    )
    .await
    .map_err(cli::domain_err)
}

/// Make this the only watched CV directory (single-watcher policy): serve
/// drops every other watcher, creates the job's CV folder when missing, and
/// runs a catch-up import scan. The frontend calls this whenever the viewed
/// job changes, so exactly the job on screen auto-imports new files.
#[tauri::command]
#[specta::specta]
pub async fn watch_job_cv_dir(app: AppHandle, job_id: String) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(app, "watch_job", WatchJobParams { job_id })
        .await
        .map_err(cli::domain_err)
}
