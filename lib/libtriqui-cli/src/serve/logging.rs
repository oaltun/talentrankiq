//! Per-request log files for the `triq serve` JSONL loop.
//!
//! Thin serve-side surface over the core logging implementation
//! ([`libtriqui_core::logging`]): per-request file resolution + open + init.

pub use libtriqui_core::logging::*;

/// Install the global tracing subscriber (stderr fallback + per-request files).
pub fn init() {
    libtriqui_core::logging::init()
}

/// Resolve the log directory for a serve app dir (CLI flag > env > prefs > default).
pub fn resolve_log_dir(
    app_dir: Option<&std::path::Path>,
    cli_override: Option<&std::path::Path>,
) -> Option<std::path::PathBuf> {
    libtriqui_core::logging::resolve_log_dir(app_dir, cli_override)
}
