//! Thin shell over `triqui-cli serve` rpcs (results CSV export) — see
//! `jobs.rs`.

use crate::triqi_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{DomainError, ResultsCsvSummary};

#[tauri::command]
#[specta::specta]
pub async fn export_results_csv(
    app: AppHandle,
    job_id: String,
    out_path: String,
) -> Result<ResultsCsvSummary, DomainError> {
    rpc_blocking_typed::<ResultsCsvSummary>(
        app,
        "export_results_csv",
        ExportResultsCsvParams { job_id, out_path },
    )
    .await
    .map_err(cli::domain_err)
}
