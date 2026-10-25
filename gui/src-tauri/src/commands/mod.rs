//! Tauri command handlers organized by domain.
//!
//! Each submodule contains related commands and their helper functions.
//! Import specific commands via their submodule (e.g., `commands::preferences::greet`).

pub mod candidates;
pub mod criteria;
pub mod debug;
pub mod embed;
pub mod feedback;
pub mod files;
pub mod import;
pub mod job_ad;
pub mod job_bundle;
pub mod jobs;
pub mod judge_license;
pub mod license;
pub mod models;
pub mod notifications;
pub mod pilot;
pub mod preferences;
pub mod quick_pane;
pub mod rank;
pub mod recovery;
pub mod results_csv;
pub mod sort;
pub mod vetted_embedders;
