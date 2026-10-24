use crate::triqi_client::params::{LicenseActivateParams, LicenseCheckArtifactParams};
use crate::triqi_client::types::LicenseStatus;
use tauri::AppHandle;

use crate::cli::{rpc_blocking, rpc_blocking_typed, str_err};

#[tauri::command]
#[specta::specta]
pub async fn license_activate(
    app: AppHandle,
    license_key: String,
) -> Result<LicenseStatus, String> {
    rpc_blocking_typed::<LicenseStatus>(
        app,
        "license_activate",
        LicenseActivateParams { license_key },
    )
    .await
    .map_err(str_err)
}

#[tauri::command]
#[specta::specta]
pub async fn license_deactivate(app: AppHandle) -> Result<(), String> {
    rpc_blocking(
        app,
        "license_deactivate",
        crate::triqi_client::EmptyParams {},
    )
    .await
    .map(|_| ())
    .map_err(str_err)
}

#[tauri::command]
#[specta::specta]
pub async fn license_status_full(app: AppHandle) -> Result<LicenseStatus, String> {
    rpc_blocking_typed::<LicenseStatus>(app, "license_status", crate::triqi_client::EmptyParams {})
        .await
        .map_err(str_err)
}

#[tauri::command]
#[specta::specta]
pub async fn license_check_artifact(app: AppHandle, artifact_id: String) -> Result<bool, String> {
    rpc_blocking_typed::<bool>(
        app,
        "license_check_artifact",
        LicenseCheckArtifactParams { artifact_id },
    )
    .await
    .map_err(str_err)
}
