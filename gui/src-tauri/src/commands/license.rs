use libtriqui_license::{activate_license, deactivate_license, load_and_verify_judge_license, load_judge_license, LicenseStatus, set_license_api_url};
use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub async fn license_activate(_app: AppHandle, license_key: String) -> Result<LicenseStatus, String> {
    // Set API URL from config/env
    set_license_api_url("https://talentrankiq.com/api/license".to_string());

    match activate_license(&license_key).await {
        Ok(license) => Ok(license.status()),
        Err(e) => Ok(LicenseStatus::Invalid { reason: e.to_string() }),
    }
}

#[tauri::command]
#[specta::specta]
pub async fn license_deactivate(_app: AppHandle) -> Result<(), String> {
    deactivate_license().map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn license_status_full() -> Result<LicenseStatus, String> {
    set_license_api_url("https://talentrankiq.com/api/license".to_string());

    match load_and_verify_judge_license().await {
        Ok(Some(license)) => Ok(license.status()),
        Ok(None) => Ok(LicenseStatus::NotFound),
        Err(e) => {
            // Check if it's a revocation error
            let err_str = e.to_string();
            if err_str.contains("revoked") || err_str.contains("Revoked") {
                Ok(LicenseStatus::Revoked { reason: err_str })
            } else {
                Ok(LicenseStatus::Invalid { reason: err_str })
            }
        }
    }
}

#[tauri::command]
#[specta::specta]
pub async fn license_check_artifact(artifact_id: String) -> Result<bool, String> {
    if let Some(license) = load_judge_license() {
        Ok(license.authorizes_artifact(&artifact_id))
    } else {
        Ok(false)
    }
}