//! License handlers (comparator license).

use serde_json::Value;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::params::{
    LicenseActivateParams, LicenseCheckArtifactParams,
};

use crate::serve::handlers::ok_json;
use crate::serve::ServeState;

pub fn license_activate(
    _state: &Arc<ServeState>,
    _p: LicenseActivateParams,
) -> Result<Value, String> {
    Ok(serde_json::json!({"status": "activated"}))
}

pub fn license_deactivate(_state: &Arc<ServeState>, _p: ()) -> Result<Value, String> {
    Ok(serde_json::json!({"status": "deactivated"}))
}

pub fn license_status(_state: &Arc<ServeState>, _p: ()) -> Result<Value, String> {
    Ok(serde_json::json!({"status": "notfound"}))
}

pub fn license_check_artifact(
    _state: &Arc<ServeState>,
    _p: LicenseCheckArtifactParams,
) -> Result<Value, String> {
    Ok(serde_json::json!({"entitled": false}))
}

pub fn comparator_license_status(_state: &Arc<ServeState>, _p: ()) -> Result<Value, String> {
    Ok(serde_json::json!({"status": "notfound"}))
}