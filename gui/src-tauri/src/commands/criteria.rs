//! Thin shell over `triqui-cli serve` rpcs (criteria) — see `jobs.rs`.

use libtriqui_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking, rpc_blocking_typed};
use crate::types::{Criterion, DomainError};

#[tauri::command]
#[specta::specta]
pub async fn add_criterion(
    app: AppHandle,
    job_id: String,
    text: String,
    weight: Option<f64>,
) -> Result<Criterion, DomainError> {
    rpc_blocking_typed::<Criterion>(
        app,
        "add_criterion",
        AddCriterionParams {
            job_id,
            text,
            weight,
        },
    )
    .await
    .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn update_criterion(
    app: AppHandle,
    id: String,
    text: String,
    weight: f64,
) -> Result<Criterion, DomainError> {
    rpc_blocking_typed::<Criterion>(
        app,
        "update_criterion",
        UpdateCriterionParams { id, text, weight },
    )
    .await
    .map_err(cli::domain_err)
}

#[tauri::command]
#[specta::specta]
pub async fn delete_criterion(app: AppHandle, id: String) -> Result<(), DomainError> {
    rpc_blocking(app, "delete_criterion", DeleteCriterionParams { id })
        .await
        .map(|_| ())
        .map_err(cli::domain_err)
}
