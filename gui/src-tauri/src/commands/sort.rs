//! Thin shell over `triqui-cli serve` rpcs (rubric ranking) — see
//! `jobs.rs`. The request stays open until the ranking finishes; progress
//! travels as `job-embedding-progress` events.

use libtriqui_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{DomainError, TriqiItem};

#[tauri::command]
#[specta::specta]
pub async fn sort_job(app: AppHandle, job_id: String) -> Result<Vec<TriqiItem>, DomainError> {
    rpc_blocking_typed::<Vec<TriqiItem>>(app, "sort_job", SortJobParams { job_id })
        .await
        .map_err(cli::domain_err)
}
