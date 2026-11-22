//! TalentRankIQ domain core.
//!
//! Everything the ranking pipeline needs, free of any Tauri dependency:
//! shared types, SQLite persistence, the embedding backends, rubric scoring,
//! the engine runtime, and the import/embed/sort/feedback operations.
//! The Tauri GUI re-exports these modules and wraps them in thin commands;
//! `triqui-cli` drives them headlessly.

pub mod app_config;
pub mod cascade;
pub mod config;
pub mod db;
pub mod embed;
pub mod embeddings;
pub mod ensure;
pub mod import;
pub mod job_ad;
pub mod rescore;
pub mod runtime;
pub mod sort;
pub mod types;