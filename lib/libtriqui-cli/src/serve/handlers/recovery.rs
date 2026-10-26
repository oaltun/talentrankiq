//! Emergency data recovery handlers (was gui `commands/recovery.rs`), rooted
//! at `<app_dir>/recovery` instead of the Tauri app-data handle.

use serde_json::Value;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::{validate_filename, MAX_RECOVERY_DATA_BYTES, RecoveryError};

use crate::serve::handlers::ok_json;
use crate::serve::ServeState;

type RecoveryResult<T> = Result<T, RecoveryError>;

fn recovery_err(e: RecoveryError) -> ProtocolError {
    ProtocolError::domain(&e)
}

/// Gets the path to the recovery directory, creating it if necessary.
fn get_recovery_dir(state: &ServeState) -> RecoveryResult<PathBuf> {
    let recovery_dir = state.app_dir.join("recovery");

    // Ensure the recovery directory exists
    std::fs::create_dir_all(&recovery_dir).map_err(|e| RecoveryError::IoError {
        message: format!("Failed to create recovery directory: {e}"),
    })?;

    Ok(recovery_dir)
}

/// Saves emergency data to a JSON file for later recovery.
/// Validates filename and enforces a 10MB size limit.
pub fn save_emergency_data(
    state: &ServeState,
    filename: &str,
    data: &Value,
) -> Result<Value, ProtocolError> {
    log::info!("saving emergency data to file: {filename}");

    // Validate filename with proper security checks
    validate_filename(filename).map_err(|e| RecoveryError::ValidationError { message: e })?;

    // Serialize to pretty JSON once for both size validation and writing
    let json_content = serde_json::to_string_pretty(data).map_err(|e| {
        log::error!("failed to serialize emergency data: {e}");
        RecoveryError::ParseError {
            message: e.to_string(),
        }
    })?;

    // Validate size (10MB limit) on the actual content that will be written
    if json_content.len() > MAX_RECOVERY_DATA_BYTES as usize {
        return Err(recovery_err(RecoveryError::DataTooLarge {
            max_bytes: MAX_RECOVERY_DATA_BYTES,
        }));
    }

    let recovery_dir = get_recovery_dir(state).map_err(recovery_err)?;
    let file_path = recovery_dir.join(format!("{filename}.json"));

    // Write to a temporary file first, then rename (atomic operation)
    let temp_path = file_path.with_extension("tmp");

    std::fs::write(&temp_path, json_content).map_err(|e| {
        log::error!("failed to write emergency data file: {e}");
        recovery_err(RecoveryError::IoError {
            message: e.to_string(),
        })
    })?;

    if let Err(rename_err) = std::fs::rename(&temp_path, &file_path) {
        log::error!("failed to finalize emergency data file: {rename_err}");
        // Clean up the temp file to avoid leaving orphaned files on disk
        if let Err(remove_err) = std::fs::remove_file(&temp_path) {
            log::warn!("failed to remove temp file after rename failure: {remove_err}");
        }
        return Err(recovery_err(RecoveryError::IoError {
            message: rename_err.to_string(),
        }));
    }

    log::info!("saved emergency data to {file_path:?}");
    ok_json(Value::Null)
}

/// Loads emergency data from a previously saved JSON file.
/// Returns FileNotFound if the file doesn't exist.
pub fn load_emergency_data(state: &ServeState, filename: &str) -> Result<Value, ProtocolError> {
    log::info!("loading emergency data from file: {filename}");

    // Validate filename with proper security checks
    validate_filename(filename).map_err(|e| RecoveryError::ValidationError { message: e })?;

    let recovery_dir = get_recovery_dir(state).map_err(recovery_err)?;
    let file_path = recovery_dir.join(format!("{filename}.json"));

    if !file_path.exists() {
        log::info!("recovery file not found: {file_path:?}");
        return Err(recovery_err(RecoveryError::FileNotFound));
    }

    let contents = std::fs::read_to_string(&file_path).map_err(|e| {
        log::error!("failed to read recovery file: {e}");
        recovery_err(RecoveryError::IoError {
            message: e.to_string(),
        })
    })?;

    let data: Value = serde_json::from_str(&contents).map_err(|e| {
        log::error!("failed to parse recovery JSON: {e}");
        recovery_err(RecoveryError::ParseError {
            message: e.to_string(),
        })
    })?;

    log::info!("loaded emergency data");
    ok_json(data)
}

/// Removes recovery files older than 7 days.
/// Returns the count of removed files.
pub fn cleanup_old_recovery_files(state: &ServeState) -> Result<Value, ProtocolError> {
    log::info!("cleaning up old recovery files");

    let recovery_dir = get_recovery_dir(state).map_err(recovery_err)?;
    let mut removed_count: u32 = 0;

    // Calculate cutoff time (7 days ago)
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| {
            recovery_err(RecoveryError::IoError {
                message: e.to_string(),
            })
        })?
        .as_secs();
    let seven_days_ago = now - (7 * 24 * 60 * 60);

    // Read directory and check each file
    let entries = std::fs::read_dir(&recovery_dir).map_err(|e| {
        log::error!("failed to read recovery directory: {e}");
        recovery_err(RecoveryError::IoError {
            message: e.to_string(),
        })
    })?;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                log::warn!("failed to read directory entry: {e}");
                continue;
            }
        };

        let path = entry.path();

        // Only process JSON files
        if path.extension().is_none_or(|ext| ext != "json") {
            continue;
        }

        // Check file modification time
        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("failed to get file metadata: {e}");
                continue;
            }
        };

        let modified = match metadata.modified() {
            Ok(m) => m,
            Err(e) => {
                log::warn!("failed to get file modification time: {e}");
                continue;
            }
        };

        let modified_secs = match modified.duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs(),
            Err(e) => {
                log::warn!("failed to convert modification time: {e}");
                continue;
            }
        };

        // Remove if older than 7 days
        if modified_secs < seven_days_ago {
            match std::fs::remove_file(&path) {
                Ok(_) => {
                    log::info!("removed old recovery file: {path:?}");
                    removed_count += 1;
                }
                Err(e) => {
                    log::warn!("failed to remove old recovery file: {e}");
                }
            }
        }
    }

    log::info!("cleanup complete. removed {removed_count} old recovery files");
    ok_json(removed_count)
}
