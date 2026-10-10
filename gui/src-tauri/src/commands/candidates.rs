//! Thin shell over `triqui-cli serve` rpcs (candidates) — see `jobs.rs`.

use libtriqui_client::params::*;
use tauri::AppHandle;

use crate::cli::{self, rpc_blocking_typed};
use crate::types::{CandidateDetail, CandidateNeighbors, CandidatePage, DomainError};

#[tauri::command]
#[specta::specta]
pub async fn get_candidate(
    app: AppHandle,
    candidate_id: String,
) -> Result<CandidateDetail, DomainError> {
    rpc_blocking_typed::<CandidateDetail>(app, "get_candidate", GetCandidateParams { candidate_id })
        .await
        .map_err(cli::domain_err)
}

/// List all of a job's candidates as a page, sorted by total rubric points (default)
/// or by a single column (`sort_by`: `None`/`"__total"` = total points, `"__name"` =
/// file name, `"__rank"` = stored rank, else a rubric-question id). `sort_dir` = `true`
/// sorts ascending; `false`/None sorts descending. `query` (when set) filters by the
/// imported CV **content** (substring, ASCII case-insensitive — SQLite `lower()` is
/// ASCII-only). Each row carries its per-question `scores` for the spreadsheet columns.
#[tauri::command]
#[specta::specta]
pub async fn list_candidates(
    app: AppHandle,
    job_id: String,
    offset: i32,
    limit: i32,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<CandidatePage, DomainError> {
    rpc_blocking_typed::<CandidatePage>(
        app,
        "list_candidates",
        ListCandidatesParams {
            job_id,
            offset,
            limit,
            query,
            sort_by,
            sort_dir,
        },
    )
    .await
    .map_err(cli::domain_err)
}

/// Prev/next candidate ids in the job's browser ordering (same sort as `list_candidates`,
/// defaulting to total points) so the overlay's `<-`/`->` buttons walk every CV in the
/// current sort. `query` (when set) restricts the walk to the filtered set.
#[tauri::command]
#[specta::specta]
pub async fn candidate_neighbors(
    app: AppHandle,
    job_id: String,
    candidate_id: String,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<CandidateNeighbors, DomainError> {
    rpc_blocking_typed::<CandidateNeighbors>(
        app,
        "candidate_neighbors",
        CandidateNeighborsParams {
            job_id,
            candidate_id,
            query,
            sort_by,
            sort_dir,
        },
    )
    .await
    .map_err(cli::domain_err)
}
