//! `triq` CLI — headless JSONL event loop over stdio.
//!
//! Reads requests from stdin, writes responses/events to stdout.
//! Logs go to stderr. Same protocol as the in-process GUI serve loop.

use std::path::PathBuf;
use std::sync::OnceLock;

use libconfigreader::Config;
use libtriqui_cli::serve;

static CONFIG: OnceLock<Config> = OnceLock::new();

fn main() {
    // Stderr = human-readable logs, stdout = JSONL protocol
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .target(env_logger::Target::Stderr)
        .init();

    // Resolve app dir (same logic as GUI: env var > default)
    let app_dir = resolve_app_dir();

    // Load config from ikapp.yaml (same as GUI)
    let config = load_config();
    CONFIG.set(config).ok();

    let app_version = env!("CARGO_PKG_VERSION").to_string();

    log::info!("triq CLI starting (version {})", app_version);
    log::info!("app_dir: {}", app_dir.display());

    if let Err(e) = serve::run_over(
        Some(app_dir),
        app_version,
        std::io::stdin().lock(),
        std::io::stdout(),
    ) {
        eprintln!("cli error: {e}");
        std::process::exit(1);
    }
}

fn resolve_app_dir() -> PathBuf {
    // TRIQ_APP_DIR env var overrides (for tests/isolation)
    if let Ok(dir) = std::env::var("TRIQ_APP_DIR") {
        return PathBuf::from(dir);
    }
    // Default: same as GUI — ~/.local/share/com.talentrankiq.triqui (Linux)
    // ~/Library/Application Support/com.talentrankiq.triqui (macOS)
    // %APPDATA%\com.talentrankiq.triqui (Windows)
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("com.talentrankiq.triqui")
}

fn load_config() -> Config {
    // Config resolution mirrors the GUI: env var > repo root > embedded default
    let config_path = std::env::var("TRIQ_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // Default to repo root / ikapp.yaml (same as libconfigreader default)
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("ikapp.yaml")
        });
    let config = libconfigreader::load(&config_path, &libconfigreader::CliOverrides::new()).unwrap_or_else(|e| {
        log::warn!("Failed to load config from {}: {e}; using embedded defaults", config_path.display());
        Config::default_from_embedded(config_path)
    });
    // Store in OnceLock for test access
    CONFIG.set(config.clone()).ok();
    config
}

// Allow tests to access the loaded config
pub fn config() -> &'static Config {
    CONFIG.get().expect("config not loaded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_loading_works() {
        // load_config() initializes the OnceLock
        let _ = load_config();
        // config() returns &'static Config (always valid, falls back to embedded defaults)
        let _ = config();
    }
}