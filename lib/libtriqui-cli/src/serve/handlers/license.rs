//! License command handlers (judge license) — thin wrappers over
//! `libtriqui_license` so the GUI can call them via JSONL RPC.

use libtriqui_client::params::{LicenseActivateParams, LicenseCheckArtifactParams};
use libtriqui_client::types::LicenseStatus;
use libtriqui_license::{
    activate_license, deactivate_license, load_and_verify_judge_license, load_judge_license,
    set_license_api_url, LicenseStatus as LicenseLibStatus,
};
use serde_json::Value;
use tokio::runtime::Runtime;

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

fn rt() -> Runtime {
    Runtime::new().expect("tokio runtime")
}

fn convert_status(s: LicenseLibStatus) -> LicenseStatus {
    match s {
        LicenseLibStatus::Licensed { customer, expires_at, artifacts } => {
            LicenseStatus::Licensed { customer, expires_at, artifacts }
        }
        LicenseLibStatus::Invalid { reason } => LicenseStatus::Invalid { reason },
        LicenseLibStatus::NotFound => LicenseStatus::NotFound,
        LicenseLibStatus::Revoked { reason } => LicenseStatus::Revoked { reason },
    }
}

/// Activate a judge license with a license key.
/// Downloads the license from the server, verifies it, and saves it locally.
pub fn license_activate(state: &ServeState, params: LicenseActivateParams) -> Result<Value, String> {
    set_license_api_url("https://talentrankiq.com/api/license".to_string());

    match rt().block_on(activate_license(&params.license_key)) {
        Ok(license) => {
            let status = convert_status(license.status());
            ok_json(status).map_err(|e| e.to_string())
        }
        Err(e) => {
            let status = LicenseStatus::Invalid { reason: e.to_string() };
            ok_json(status).map_err(|e| e.to_string())
        }
    }
}

/// Deactivate (remove) the local judge license file.
pub fn license_deactivate(_state: &ServeState, _params: ()) -> Result<Value, String> {
    deactivate_license().map_err(|e| e.to_string())?;
    ok_json(()).map_err(|e| e.to_string())
}

/// Get the current judge license status (with online revocation check).
pub fn license_status(_state: &ServeState, _params: ()) -> Result<Value, String> {
    set_license_api_url("https://talentrankiq.com/api/license".to_string());

    match rt().block_on(load_and_verify_judge_license()) {
        Ok(Some(license)) => {
            let status = convert_status(license.status());
            ok_json(status).map_err(|e| e.to_string())
        }
        Ok(None) => ok_json(LicenseStatus::NotFound).map_err(|e| e.to_string()),
        Err(e) => {
            let err_str = e.to_string();
            let status = if err_str.contains("revoked") || err_str.contains("Revoked") {
                LicenseStatus::Revoked { reason: err_str }
            } else {
                LicenseStatus::Invalid { reason: err_str }
            };
            ok_json(status).map_err(|e| e.to_string())
        }
    }
}

/// Check if the judge license authorizes a given artifact id.
pub fn license_check_artifact(
    _state: &ServeState,
    params: LicenseCheckArtifactParams,
) -> Result<Value, String> {
    if let Some(license) = load_judge_license() {
        let authorized = license.authorizes_artifact(&params.artifact_id);
        ok_json(authorized).map_err(|e| e.to_string())
    } else {
        ok_json(false).map_err(|e| e.to_string())
    }
}

/// Get the judge license status (without online verification - used by judge_license_status).
pub fn judge_license_status(_state: &ServeState, _params: ()) -> Result<Value, String> {
    let status = load_judge_license()
        .map(|l| convert_status(l.status()))
        .unwrap_or(LicenseStatus::NotFound);
    ok_json(status).map_err(|e| e.to_string())
}