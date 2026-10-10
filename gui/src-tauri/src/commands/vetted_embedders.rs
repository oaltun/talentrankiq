//! Thin shell over `triqui-cli serve` rpcs (vetted embedder cards) — see
//! `jobs.rs`.

use tauri::AppHandle;

use crate::cli::{rpc, str_err};
use crate::types::VettedEmbeddersResponse;

#[tauri::command]
#[specta::specta]
pub fn list_vetted_embedders(app: AppHandle) -> Result<VettedEmbeddersResponse, String> {
    let value = rpc(
        &app,
        "list_vetted_embedders",
        &libtriqui_client::EmptyParams {},
    )
    .map_err(str_err)?;
    serde_json::from_value(value).map_err(|e| format!("bad response: {e}"))
}
