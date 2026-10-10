//! TalentRankIQ domain core.
//!
//! Everything the ranking pipeline needs, free of any Tauri dependency: shared
//! types, SQLite persistence, the embedding backends, rubric scoring, the
//! engine runtime, and the import/embed/sort/feedback operations. The Tauri GUI
//! re-exports these modules (`pub use libtriqui_core::...`) and wraps them in
//! thin `#[tauri::command]`s; `triqui-cli` drives them headlessly.

pub mod app_config;
pub mod candidates;
pub mod cascade;
pub mod chunk_match;
pub mod config;
pub mod db;
pub mod embed;
pub mod embeddings;
pub mod feedback;
pub mod import;
pub mod job_ad;
pub mod job_bundle;
pub mod judge;
pub mod judge_cache;
pub mod ranker;
pub mod results_csv;
 pub mod rescore;
 /// The rubric: a job's ranking inputs and their vectors, per embedding space.
 pub mod rubric;
 pub mod runtime;
 /// Stage-0.5 scout (cheap multilingual gate) for the progressive-ranking cascade.
 pub mod scout;
 /// System-capacity budget: cascade wave scaling + the `holding` reason code
 /// (libsyscap design §7.3).
 pub mod syscap;
 pub mod sort;
 pub mod types;
