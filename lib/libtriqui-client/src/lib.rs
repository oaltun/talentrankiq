//! `libtriqui_client` — client side of the gui↔serve protocol.
//!
//! This crate is the contract between the Tauri gui and the headless
//! serve loop (thin gui pattern, `NOAI.md` / `gui/design/app1/thin-gui.md`). Both
//! sides share the envelope ([`protocol`]), the command params
//! ([`params`]), and every DTO that crosses the wire ([`types`]), so the
//! contract cannot drift (contract-lib pattern, same role
//! `libmodelcallerclient` plays for `modelcaller`).
//!
//! The envelope runs in-process: serve runs on a thread inside the host
//! (the gui, or the e2e contract tests), connected through
//! [`transport::duplex`] and [`client::TriqiClient::over`] — no child
//! process, no `triqui-cli` binary (thin gui pattern).
//!
//! The host links only this crate plus the `triqui-cli` library that hosts
//! the serve loop; the DB, triq.yaml, engine runtime, model management, and
//! CV watching all live behind it.
//!
//! This crate re-exports all wire types from `libcli-client` (the single
//! source of truth for the JSONL protocol) and adds internal utilities
//! (validation functions, constants) used by the serve loop but not part
//! of the wire contract.

pub use libcli_client::*;

// Internal utilities (not part of wire protocol)
mod validation {
    use regex::Regex;
    use std::sync::LazyLock;

    /// Default shortcut for the quick pane
    pub const DEFAULT_QUICK_PANE_SHORTCUT: &str = "CommandOrControl+Shift+.";

    /// Maximum size for recovery data files (10MB)
    pub const MAX_RECOVERY_DATA_BYTES: u32 = 10_485_760;

    /// Pre-compiled regex pattern for filename validation.
    /// Only allows alphanumeric characters, dashes, underscores, and a single extension.
    pub static FILENAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9]+)?$")
            .expect("Failed to compile filename regex pattern")
    });

    /// Validates a filename for safe file system operations.
    /// Only allows alphanumeric characters, dashes, underscores, and a single extension.
    pub fn validate_filename(filename: &str) -> Result<(), String> {
        if filename.is_empty() {
            return Err("Filename cannot be empty".to_string());
        }

        if filename.chars().count() > 100 {
            return Err("Filename too long (max 100 characters)".to_string());
        }

        if !FILENAME_PATTERN.is_match(filename) {
            return Err(
                "Invalid filename: only alphanumeric characters, dashes, underscores, and dots allowed"
                    .to_string(),
            );
        }

        Ok(())
    }

    /// Validates string input length (by character count, not bytes).
    pub fn validate_string_input(input: &str, max_len: usize, field_name: &str) -> Result<(), String> {
        let char_count = input.chars().count();
        if char_count > max_len {
            return Err(format!("{field_name} too long (max {max_len} characters)"));
        }
        Ok(())
    }

    /// Validates theme value.
    pub fn validate_theme(theme: &str) -> Result<(), String> {
        match theme {
            "light" | "dark" | "system" => Ok(()),
            _ => Err("Invalid theme: must be 'light', 'dark', or 'system'".to_string()),
        }
    }

    /// Validates the color-scheme preference (palette overlay on light/dark).
    pub fn validate_color_theme(color_theme: &str) -> Result<(), String> {
        match color_theme {
            "default" | "slate" | "ocean" | "forest" | "amber" | "contrast" => Ok(()),
            _ => Err(
                "Invalid color theme: must be 'default', 'slate', 'ocean', 'forest', 'amber', or 'contrast'"
                    .to_string(),
            ),
        }
    }
}

pub use validation::*;