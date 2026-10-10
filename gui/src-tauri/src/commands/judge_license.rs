use libtriqui_license::{load_judge_license, LicenseStatus};

#[tauri::command]
    #[specta::specta]
    pub async fn judge_license_status() -> Result<LicenseStatus, String> {
    Ok(load_judge_license()
        .map(|l| l.status())
        .unwrap_or(LicenseStatus::NotFound))
}
