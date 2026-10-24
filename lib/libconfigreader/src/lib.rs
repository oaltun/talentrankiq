//! libconfigreader — shared configuration types for the ik toolchain.
//!
//! This crate provides the pure-Rust `Config` struct with all model, family,
//! embedder, engine, syscap, cascade, and judge artifact definitions.
//! There is no YAML parsing anymore — the configuration is embedded as
//! typed Rust data in `config.rs`.

pub use config::*;

// Backward-compatible re-exports for dependent crates
pub use config::EngineConfig as EngineSection;
pub use config::SyscapConfig as SyscapSection;
pub use config::CascadeConfig as CascadeSection;
pub use config::GuiJudgeArtifact as GuiJudgeArtifacts;

// Types that keep the same name are already exported via `pub use config::*;`

mod config;

/// Embedded default configuration constants (kept for reference).
/// The actual defaults are now in `Config::load()`.
pub const DEFAULT_CONFIG_YAML: &str = r#"
model_dir: "~/.cache/ik/models"
verbose: false
revision: main
gui_idle_unload_secs: 300
engine:
  backend: auto
syscap:
  on_battery_polite: true
  build_jobs: auto
  reserve_cores: 1
  reserve_core_frac: 0.10
rubric_defaults:
  ad_points: 50
  fb_job_points: 10
  fb_profile_points: 5
  criterion_points: 10
families: []
models: []
gui_embedders: []
"#;

/// Load the embedded configuration.
/// This replaces the old YAML-based `load()` function.
/// Returns the static config directly — no file I/O, no errors possible.
pub fn load() -> Config {
    Config::load()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_returns_valid_config() {
        let cfg = load();
        assert_eq!(cfg.families.len(), 10);
        assert_eq!(cfg.models.len(), 18);
        assert_eq!(cfg.gui_embedders.len(), 4);
        assert!(!cfg.verbose);
        assert_eq!(cfg.revision, "main");
        assert_eq!(cfg.gui_idle_unload_secs, 300);
    }
}