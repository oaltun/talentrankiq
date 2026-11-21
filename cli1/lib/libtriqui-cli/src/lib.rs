//! `triqui-cli` library — headless access to the TalentRankIQ ranking
//! pipeline. The `triqui-cli serve` subcommand runs the serve loop over
//! stdio (tests, humans, scripts). The GUI now runs the CLI as a subprocess
//! instead of in-process.
//!
//! In-process serve ([`InProcessServe`]) is available for tests that need
//! to run the serve loop directly with pinned config.

use std::path::PathBuf;

use libtriqui_core::db;
use libtriqui_core::embeddings;
use libtriqui_core::runtime::{EngineRuntime, RuntimeConfig};
use libtriqui_core::sort;
use libconfigreader::Config;

pub mod inprocess;
pub mod serve;

pub use inprocess::InProcessServe;

/// Parsed top-level command for [`run`] (tools / future thin shells).
pub enum Cmd {
    Status,
    Rank {
        job_id: String,
    },
    Prewarm {
        job_id: String,
    },
    Like {
        candidate_id: String,
    },
    Serve {
        app_dir: Option<PathBuf>,
        app_version: String,
    },
}

/// Dispatch a parsed command. Returns on completion (or exits the process on
/// error, matching the historical CLI behaviour).
pub fn run(db_path: Option<PathBuf>, model: Option<String>, cmd: Cmd) {
    match cmd {
        Cmd::Serve {
            app_dir,
            app_version,
        } => {
            if let Err(e) = serve::run(app_dir, app_version) {
                eprintln!("serve error: {e}");
                std::process::exit(1);
            }
        }
        Cmd::Status => cmd_status(&db_path.unwrap_or_else(db::default_db_path)),
        Cmd::Rank { job_id } => cmd_rank(
            &db_path.unwrap_or_else(db::default_db_path),
            &model,
            &job_id,
        ),
        Cmd::Prewarm { job_id } => cmd_prewarm(
            &db_path.unwrap_or_else(db::default_db_path),
            &model,
            &job_id,
        ),
        Cmd::Like { candidate_id } => {
            cmd_like(&db_path.unwrap_or_else(db::default_db_path), &candidate_id)
        }
    }
}

pub fn open(db_path: &std::path::Path) -> rusqlite::Connection {
    match db::open_connection_at(db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Spawn the engine and block until ready (or exit with the error).
pub fn start_engine(model: &Option<String>, db_path: &std::path::Path) -> std::sync::Arc<EngineRuntime> {
    let app_dir = db_path.parent().unwrap();
    let user_ovr = crate::serve::user_overrides_from_prefs(app_dir);
    let cfg = Config::load().with_user_overrides(&user_ovr);
    let rt = EngineRuntime::new(
        RuntimeConfig::discover(model.clone(), Some(cfg)),
        db_path.to_path_buf(),
    );
    // Surface status changes as info logs instead of UI events.
    rt.set_status_hook(std::sync::Arc::new(|info| {
        log::info!("engine: {} (model {:?})", info.status, info.model);
    }));
    rt.start();
    match rt.wait_until_ready(std::time::Duration::from_secs(300)) {
        Ok(()) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn cmd_status(db_path: &std::path::Path) {
    let conn = open(db_path);
    println!("db: {}", db_path.display());
    println!(
        "embedding_cache rows: {}",
        db::count(&conn, "embedding_cache").unwrap_or(-1)
    );
    println!(
        "candidate_chunks rows: {}",
        db::count(&conn, "candidate_chunks").unwrap_or(-1)
    );
    println!();
    println!(
        "{:<40} {:<5} {:<42} {:>6} {:>7} {:>6}",
        "job id", "arch", "title", "cands", "ranked", "liked"
    );
    for (id, title, archived, cands, ranked) in db::job_overview(&conn).unwrap_or_default() {
        let liked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND liked = 1",
                rusqlite::params![&id],
                |r| r.get(0),
            )
            .unwrap_or(-1);
        println!(
            "{:<40} {:<5} {:<42} {:>6} {:>7} {:>6}",
            id,
            if archived { "yes" } else { "no" },
            truncate(&title, 40),
            cands,
            ranked,
            liked
        );
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut: String = s.chars().take(max_chars).collect();
    format!("{cut}…")
}

fn cmd_rank(db_path: &std::path::Path, model: &Option<String>, job_id: &str) {
    let rt = start_engine(model, db_path);
    let triqui = match sort::sort_job_blocking(&rt, job_id, &|_, _, _, _| {}) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "{:>5}  {:>10}  {:>10}  candidate",
        "rank", "points", "of max"
    );
    for item in &triqui {
        println!(
            "{:>5}  {:>10.2}  {:>10.2}  {} ({})",
            item.rank,
            item.points.unwrap_or(0.0),
            item.points_max.unwrap_or(0.0),
            item.file_name,
            item.id
        );
    }
}

fn cmd_prewarm(db_path: &std::path::Path, model: &Option<String>, job_id: &str) {
    let rt = start_engine(model, db_path);
    if let Err(e) = embeddings::prewarm_job_blocking(&rt, job_id) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    println!("prewarmed rubric vectors for job {job_id}");
}

fn cmd_like(db_path: &std::path::Path, candidate_id: &str) {
    let conn = open(db_path);
    let affected = conn
        .execute(
            "UPDATE candidates SET liked = 1 - liked WHERE id = ?1",
            rusqlite::params![&candidate_id],
        )
        .unwrap_or(0);
    if affected == 0 {
        eprintln!("error: candidate '{candidate_id}' not found");
        std::process::exit(1);
    }
    let liked: bool = conn
        .query_row(
            "SELECT liked FROM candidates WHERE id = ?1",
            rusqlite::params![&candidate_id],
            |r| Ok(r.get::<_, i64>(0)? != 0),
        )
        .unwrap_or(false);
    println!("{candidate_id}: liked = {liked}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_works() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("very long string", 10), "very long …");
    }
}