//! Emergency data recovery commands — thin shells over `triqui-cli serve`
//! rpcs (the recovery files live in the serve app dir). The typed
//! `RecoveryError` is rehydrated from the protocol error envelope so frontend
//! matching is unchanged.

use libtriqui_client::params::*;
use serde_json::Value;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking};
use crate::types::RecoveryError;

#[tauri::command]
#[specta::specta]
pub async fn save_emergency_data(
    app: AppHandle,
    filename: String,
    data: Value,
) -> Result<(), RecoveryError> {
    log::info!("Saving emergency data to file: {filename}");
    rpc_blocking(
        app,
        "save_emergency_data",
        SaveEmergencyDataParams { filename, data },
    )
    .await
    .map(|_| ())
    .map_err(cli::recovery_err)
}

#[tauri::command]
#[specta::specta]
pub async fn load_emergency_data(app: AppHandle, filename: String) -> Result<Value, RecoveryError> {
    log::info!("Loading emergency data from file: {filename}");
    rpc_blocking(
        app,
        "load_emergency_data",
        LoadEmergencyDataParams { filename },
    )
    .await
    .map_err(cli::recovery_err)
}

#[tauri::command]
#[specta::specta]
pub async fn cleanup_old_recovery_files(app: AppHandle) -> Result<u32, RecoveryError> {
    log::info!("Cleaning up old recovery files");
    rpc_blocking_typed_u32(app, "cleanup_old_recovery_files")
        .await
        .map_err(cli::recovery_err)
}

async fn rpc_blocking_typed_u32(
    app: AppHandle,
    cmd: &'static str,
) -> Result<u32, libtriqui_client::client::ClientError> {
    let value = rpc_blocking(app, cmd, EmptyParams {}).await?;
    serde_json::from_value(value)
        .map_err(|e| libtriqui_client::client::ClientError::Json(e.to_string()))
}
