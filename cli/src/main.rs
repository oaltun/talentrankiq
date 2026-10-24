//! `triq` CLI — headless JSONL event loop over stdio.
//!
//! Reads requests from stdin, writes responses/events to stdout.
//! Logs go to stderr. Same protocol as the in-process GUI serve loop.

use std::path::PathBuf;
use std::sync::OnceLock;

use clap::{Parser, Subcommand};
use libconfigreader::Config;
use libmodeldownloader::{download_url, UrlDownload};
use libtriqui_cli::serve;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Parser, Debug)]
#[command(name = "triq", version, about = "TalentRankIQ CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the JSONL serve loop over stdio
    Serve {
        /// App directory (overrides TRIQ_APP_DIR)
        #[arg(long)]
        app_dir: Option<PathBuf>,
    },
    /// Download a model using libmodeldownloader
    Download {
        /// Model slug (e.g., harrier-oss-v1-270m-ONNX-q4f16)
        #[arg(long)]
        slug: String,
        /// HuggingFace blob URL to the ONNX file
        #[arg(long)]
        url: String,
        /// Additional repo-relative files to fetch (repeatable)
        #[arg(long = "extra-file")]
        extra_files: Vec<String>,
        /// Model cache directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
        /// SHA256 checksum of the primary artifact
        #[arg(long)]
        sha256: Option<String>,
    },
    /// List available embedders from config
    ListEmbedders,
}

fn main() {
    let cli = Cli::parse();

    // Stderr = human-readable logs, stdout = JSONL protocol
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .target(env_logger::Target::Stderr)
        .init();

    match cli.command {
        Commands::Serve { app_dir } => {
            let app_dir = app_dir.unwrap_or_else(resolve_app_dir);

            // Load embedded config (no more YAML file)
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
        Commands::Download {
            slug,
            url,
            extra_files,
            model_dir,
            force,
            sha256,
        } => {
            let model_dir = model_dir.unwrap_or_else(|| {
                libtriqui_core::app_config::model_dir()
            });

            log::info!("Downloading model: {}", slug);
            log::info!("URL: {}", url);
            log::info!("Model dir: {}", model_dir.display());
            log::info!("Extra files: {:?}", extra_files);
            log::info!("Force: {}", force);

            let mut req = UrlDownload::new(url, slug, model_dir, sha256);
            req.extra_files = extra_files;
            req.force = force;

            match download_url(req, |ev| {
                log::info!("Progress: {:?}", ev);
            }) {
                Ok(path) => {
                    log::info!("Download complete: {}", path.display());
                }
                Err(e) => {
                    eprintln!("Download failed: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::ListEmbedders => {
            let config = load_config();
            CONFIG.set(config.clone()).ok();
            for ge in &config.gui_embedders {
                println!("{} ({}) - {} MB", ge.slug, ge.name, ge.memory_mb);
            }
        }
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
    // Load embedded config directly — no YAML file, no env var override
    let config = libconfigreader::load();
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
        // config() returns &'static Config (always valid, embedded defaults)
        let _ = config();
    }
}