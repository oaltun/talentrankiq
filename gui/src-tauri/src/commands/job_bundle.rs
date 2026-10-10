//! Thin shell over `triqui-cli serve` rpcs (job bundles) — see `jobs.rs`.
//! Export writes the `.triquijob` zip on the serve side; import creates a
//! NEW job there, stages its CV files, runs the pipeline, and applies the
//! results snapshot, then returns the created job for navigation.

use libtriqui_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{BundleExportOptions, BundleExportSummary, DomainError, Job};

#[tauri::command]
#[specta::specta]
pub async fn export_job_bundle(
    app: AppHandle,
    job_id: String,
    options: BundleExportOptions,
    out_path: String,
) -> Result<BundleExportSummary, DomainError> {
    rpc_blocking_typed::<BundleExportSummary>(
        app,
        "export_job_bundle",
        ExportJobBundleParams {
            job_id,
            options,
            out_path,
        },
    )
    .await
    .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn import_job_bundle(app: AppHandle, path: String) -> Result<Job, DomainError> {
    rpc_blocking_typed::<Job>(app, "import_job_bundle", ImportJobBundleParams { path })
        .await
        .map_err(cli::domain_err)
}
