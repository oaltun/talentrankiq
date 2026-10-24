//! Thin shell over `triqui-cli serve` rpcs (CV import) — see `jobs.rs`.
//! The serve process validates + collects the file list, then runs the
//! extract → embed → rank pipeline in the background; progress travels as
//! `job-import-progress` / `job-embedding-progress` events.

use crate::triqi_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking};
use crate::types::DomainError;

#[tauri::command]
#[specta::specta]
pub async fn import_cv_directory(
    app: AppHandle,
    job_id: String,
    directory: String,
) -> Result<(), DomainError> {
    rpc_blocking(
        app,
        "import_cv_directory",
        ImportCvDirectoryParams { job_id, directory },
    )
    .await
    .map(|_| ())
    .map_err(cli::domain_err)
}
