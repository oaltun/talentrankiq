//! Thin shell over `triqui-cli serve` rpcs (job ad) — see `jobs.rs`.
//! The job ad is a **text file on disk**: `get_job_ad` hands back its path and
//! contents (creating the file empty on first use), `set_job_ad_path` is the
//! "Change…" button, and `reload_job_ad` re-reads it after the user edits it in
//! an external editor. There is no ad editor in the gui.

use libtriqui_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{DomainError, Job, JobAdFile};

#[tauri::command]
#[specta::specta]
pub async fn get_job_ad(app: AppHandle, job_id: String) -> Result<JobAdFile, DomainError> {
    rpc_blocking_typed::<JobAdFile>(app, "get_job_ad", GetJobAdParams { job_id })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn set_job_ad_path(
    app: AppHandle,
    job_id: String,
    path: String,
) -> Result<JobAdFile, DomainError> {
    rpc_blocking_typed::<JobAdFile>(app, "set_job_ad_path", SetJobAdPathParams { job_id, path })
        .await
        .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn reload_job_ad(app: AppHandle, job_id: String) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(app, "reload_job_ad", ReloadJobAdParams { job_id })
        .await
        .map_err(cli::domain_err)
}
