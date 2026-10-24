use crate::triqi_client::types::LicenseStatus;
use tauri::AppHandle;

use crate::cli::{rpc_blocking_typed, str_err};

#[tauri::command]
#[specta::specta]
pub async fn judge_license_status(app: AppHandle) -> Result<LicenseStatus, String> {
    rpc_blocking_typed::<LicenseStatus>(
        app,
        "judge_license_status",
        crate::triqi_client::EmptyParams {},
    )
    .await
    .map_err(str_err)
}