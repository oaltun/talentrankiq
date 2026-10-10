//! Job lifecycle handlers (was gui `commands/jobs.rs`), over the serve
//! process's shared DB connection. CV-dir watching moves here too: job
//! create/delete/dir-change manage the serve-side watchers directly.

use serde_json::Value;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{
    Criterion, DomainError, ExtractionStatus, FileType, Job, JobDetail, JobSummary, JobUiState,
    QuestionKind, RankState, RubricQuestion, TriqiItem, SortReport,
};
use libtriqui_core::app_config;

use crate::serve::handlers::criteria::criterion_from_row;
use crate::serve::handlers::{domain, lock_err, ok_json};
use crate::serve::{watch, ServeState};

pub const JOB_COLS: &str = "id, title, job_ad_text, import_directory_path, triqui_size, archived, created_at, updated_at, last_ranked_at, inputs_updated_at, ad_weight, fb_job_weight, fb_profile_weight, job_ad_path";

pub fn db_err(e: rusqlite::Error) -> DomainError {
    libtriqui_core::db::db_err(e)
}

pub fn job_from_row(row: &rusqlite::Row) -> rusqlite::Result<Job> {
    Ok(Job {
        id: row.get(0)?,
        title: row.get(1)?,
        job_ad_text: row.get(2)?,
        import_directory_path: row.get(3)?,
        triqui_size: row.get(4)?,
        archived: row.get::<_, i64>(5)? != 0,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        last_ranked_at: row.get(8)?,
        inputs_updated_at: row.get(9)?,
        ad_weight: row.get(10)?,
        fb_job_weight: row.get(11)?,
        fb_profile_weight: row.get(12)?,
        job_ad_path: row.get(13)?,
    })
}

pub fn triqui_from_row(row: &rusqlite::Row) -> rusqlite::Result<TriqiItem> {
    Ok(TriqiItem {
        id: row.get(0)?,
        file_name: row.get(1)?,
        file_type: FileType::from_str(row.get::<_, String>(2)?.as_str()),
        extraction_status: ExtractionStatus::from_str(row.get::<_, String>(3)?.as_str()),
        rank: row.get(4)?,
        points: row.get(5)?,
        points_max: row.get(6)?,
        liked: row.get::<_, i64>(7)? != 0,
        viewed_at: row.get(8)?,
    })
}

/// Build the results-table column list: the cascade's stage grades first, in the
/// order the pipeline produces them (scout → embed → judge), then the
/// rubric the total is made of (job ad, criteria, the two feedback rows).
///
/// The stage columns are always present, whether or not the stage has produced a
/// grade yet — the sheet shows `–` for a stage a candidate has not reached, and
/// columns that appear mid-cascade would make the row jump under the reader.
/// Pure config: nothing here depends on what has been computed yet.
fn build_rubric(job: &Job, criteria: &[Criterion]) -> Vec<RubricQuestion> {
    let stage = |question_id: &str, label: &str, kind: QuestionKind| RubricQuestion {
        question_id: question_id.into(),
        label: label.into(),
        kind,
        // Every stage is scored in the job ad's currency, so its points read on
        // the same scale as the ad column they sit next to.
        max_points: job.ad_weight as f32,
    };
    let mut out = vec![
        stage(
            libtriqui_core::scout::SCOUT_QUESTION_ID,
            libtriqui_core::scout::SCOUT_LABEL,
            QuestionKind::Scout,
        ),
        RubricQuestion {
            question_id: libtriqui_core::sort::AD_QUESTION_ID.into(),
            label: "Job fit (embed)".into(),
            kind: QuestionKind::Ad,
            max_points: job.ad_weight as f32,
        },
        stage(
            libtriqui_core::sort::JUDGE_QUESTION_ID,
            libtriqui_core::judge::JudgeStage::AD_LABEL,
            QuestionKind::AdJudge,
        ),
    ];
    for c in criteria {
        out.push(RubricQuestion {
            question_id: c.id.clone(),
            label: c.text.clone(),
            kind: QuestionKind::Criterion,
            max_points: c.weight as f32,
        });
    }
    out.push(RubricQuestion {
        question_id: "fb_job".into(),
        label: "Feedback (this job)".into(),
        kind: QuestionKind::FeedbackJob,
        max_points: job.fb_job_weight as f32,
    });
    out.push(RubricQuestion {
        question_id: "fb_profile".into(),
        label: "Feedback (all jobs)".into(),
        kind: QuestionKind::FeedbackProfile,
        max_points: job.fb_profile_weight as f32,
    });
    out
}

/// Σ |max_points| over the questions that count towards `points` — the "Total /N"
/// caption. Stage columns are excluded: they report what a stage thought of a
/// CV, not a slice of the rubric the rank is made of.
fn rubric_capacity(rubric: &[RubricQuestion]) -> f32 {
    rubric
        .iter()
        .filter(|q| !q.kind.is_stage())
        .map(|q| q.max_points.abs())
        .sum()
}

/// Whether a candidate CV directory still holds CVs. Only a *non-empty*
/// folder blocks: an empty leftover is safe to hand to a new job, and reusing
/// it is what keeps the numbering from creeping upward across sessions.
///
/// Fails safe — an unreadable path (or a non-directory in its place) counts as
/// holding data, so a number is never handed to a job while something unknown
/// sits in its folder.
fn folder_still_holds_cvs(dir: &str) -> bool {
    if dir.is_empty() {
        return false;
    }
    match std::fs::read_dir(dir) {
        Ok(mut entries) => entries.next().is_some(),
        Err(_) => std::path::Path::new(dir).exists(),
    }
}

/// Whether a new job's default folder `<root>/job-N` still holds CVs, i.e.
/// whether its number must be skipped when naming a new job. The current
/// layout keeps CVs in `<root>/job-N/cv/`, but folders from before that
/// change (or hand-made ones) may hold them directly in `<root>/job-N` —
/// both layouts block, otherwise a new job would nest its fresh `cv/` beside
/// a legacy folder's files, and the import scan (which recurses into
/// subdirectories) would sweep those into whichever job watches the parent.
/// Only *content* blocks: an empty leftover — with or without its empty `cv/`
/// subfolder — is safe to hand to a new job, and reusing it is what keeps the
/// numbering from creeping upward across sessions.
///
/// Fails safe — an unreadable path (or a non-directory in its place) counts as
/// holding data, so a number is never handed to a job while something unknown
/// sits in its folder.
fn job_folder_holds_cvs(job_dir: &Path) -> bool {
    match std::fs::read_dir(job_dir) {
        Ok(entries) => {
            for entry in entries.flatten() {
                // The current layout's `cv/` subfolder only blocks when it
                // itself holds CVs; every other entry is someone's data.
                if *entry.file_name() == *OsStr::new("cv")
                    && entry.path().is_dir()
                    && !folder_still_holds_cvs(entry.path().to_string_lossy().as_ref())
                {
                    continue;
                }
                return true;
            }
            false
        }
        Err(_) => job_dir.exists(),
    }
}

pub fn create_job(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let id = format!("job_{}", Uuid::new_v4());
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0))
        .map_err(db_err)?;
    let rubric = app_config::rubric_defaults();
    // Fresh, empty per-job CV directory under the configured root
    // (`<root>/job-N/cv`, e.g. `~/Documents/jobs/job-1/cv`). A number is free
    // when no live job holds the title and `<root>/job-N` holds no CVs — so a
    // deleted job never leaks its CVs into a new one, while an *empty* leftover
    // folder is reused. Reusing empty ones matters: folders are never removed
    // on delete, so keying off mere existence ratcheted the counter up by one
    // per deleted job and left an unbounded trail of empty dirs.
    let root = watch::cv_root(state);
    let mut n = count + 1;
    loop {
        let title = format!("job-{n}");
        let title_taken = conn
            .query_row(
                "SELECT 1 FROM jobs WHERE title = ?1",
                rusqlite::params![&title],
                |_| Ok(true),
            )
            .unwrap_or(false);
        // An empty root (no resolvable Documents dir) means no folder to check.
        let folder_free = root.as_os_str().is_empty() || !job_folder_holds_cvs(&root.join(&title));
        if title_taken || !folder_free {
            n += 1;
            continue;
        }
        break;
    }
    let title = format!("job-{n}");
    let cv_dir = watch::default_job_dir(&root, &title);
    conn.execute(
        "INSERT INTO jobs (id, title, ad_weight, fb_job_weight, fb_profile_weight, import_directory_path) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            &id,
            &title,
            rubric.ad_points,
            rubric.fb_job_points,
            rubric.fb_profile_points,
            &cv_dir
        ],
    )
    .map_err(db_err)?;
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![&id],
            job_from_row,
        )
        .map_err(db_err)?;
    drop(conn);
    // The new job is (about to be) the viewed one: point the single watcher
    // at it. This also creates the fresh CV folder on disk.
    watch::watch_job(state, &id);
    log::info!("created job {id} with cv dir {cv_dir}");
    ok_json(job)
}

pub fn list_job_summaries(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let active = state.import_state.0.lock().map_err(lock_err)?;
    let mut stmt = conn
        .prepare(
            "SELECT j.id, j.title, j.triqui_size, j.archived, j.created_at, j.last_ranked_at,
                    COALESCE(SUM(CASE WHEN c.extraction_status = 'Extracted' THEN 1 ELSE 0 END), 0) AS cv_imported,
                    COALESCE(SUM(CASE WHEN c.extraction_status = 'Failed'   THEN 1 ELSE 0 END), 0) AS cv_failed,
                    CASE WHEN j.job_ad_text IS NOT NULL AND j.job_ad_text <> '' THEN 1 ELSE 0 END AS ad_set,
                    (SELECT COUNT(*) FROM criteria cr WHERE cr.job_id = j.id) AS want_count,
                    0 AS avoid_count
             FROM jobs j
             LEFT JOIN candidates c ON c.job_id = j.id
             GROUP BY j.id
             ORDER BY j.created_at DESC",
        )
        .map_err(db_err)?;
    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        Ok(JobSummary {
            id,
            title: row.get(1)?,
            triqui_size: row.get(2)?,
            archived: row.get::<_, i64>(3)? != 0,
            created_at: row.get(4)?,
            last_ranked_at: row.get(5)?,
            cv_imported: row.get(6)?,
            cv_failed: row.get(7)?,
            ad_set: row.get::<_, i64>(8)? != 0,
            want_count: row.get(9)?,
            avoid_count: row.get(10)?,
            importing: false,
        })
    });
    let mut out = Vec::new();
    for r in rows.map_err(db_err)? {
        let mut s = r.map_err(db_err)?;
        s.importing = active.contains(&s.id);
        out.push(s);
    }
    ok_json(out)
}

pub fn get_job_detail(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let job = match conn.query_row(
        &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
        rusqlite::params![job_id],
        job_from_row,
    ) {
        Ok(j) => j,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(domain(&DomainError::NotFound {
                entity: "Job".into(),
                id: job_id.to_string(),
            }));
        }
        Err(e) => return Err(domain(&db_err(e))),
    };
    let n = job.triqui_size;

    let mut stmt = conn
        .prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at
             FROM candidates WHERE job_id = ?1 AND rank IS NOT NULL ORDER BY rank LIMIT ?2",
        )
        .map_err(db_err)?;
    let triqui: Vec<TriqiItem> = stmt
        .query_map(rusqlite::params![job_id, n], triqui_from_row)
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;

    let mut stmt = conn
        .prepare("SELECT id, job_id, text, weight, created_at FROM criteria WHERE job_id = ?1 ORDER BY created_at")
        .map_err(db_err)?;
    let criteria: Vec<Criterion> = stmt
        .query_map(rusqlite::params![job_id], criterion_from_row)
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;

    let (cv_imported, cv_failed) = conn
        .query_row(
            "SELECT COALESCE(SUM(CASE WHEN extraction_status = 'Extracted' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN extraction_status = 'Failed'   THEN 1 ELSE 0 END), 0)
             FROM candidates WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
        )
        .map_err(db_err)?;

    let importing = state
        .import_state
        .0
        .lock()
        .map_err(lock_err)?
        .contains(job_id);

    let rubric = build_rubric(&job, &criteria);
    let capacity = rubric_capacity(&rubric);
    ok_json(JobDetail {
        job,
        criteria,
        rubric,
        rubric_capacity: capacity,
        triqui,
        cv_imported: cv_imported as i32,
        cv_failed: cv_failed as i32,
        importing,
    })
}

pub fn update_job(
    state: &Arc<ServeState>,
    job_id: &str,
    title: &str,
    job_ad_text: Option<String>,
    triqui_size: i32,
) -> Result<Value, ProtocolError> {
    if title.trim().is_empty() {
        return Err(domain(&DomainError::ValidationError {
            message: "Title cannot be empty".into(),
        }));
    }
    if triqui_size < 1 {
        return Err(domain(&DomainError::ValidationError {
            message: "TalentRankIQ size must be a positive integer".into(),
        }));
    }
    let conn = state.db.0.lock().map_err(lock_err)?;

    // Fetch current job to detect ranking-input changes.
    let current: Job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(db_err)?;

    let inputs_changed =
        current.job_ad_text != job_ad_text || current.triqui_size != triqui_size;

    let sql = if inputs_changed {
        "UPDATE jobs SET title = ?1, job_ad_text = ?2, triqui_size = ?3, updated_at = datetime('now'), inputs_updated_at = datetime('now'), inputs_version = inputs_version + 1 WHERE id = ?4"
    } else {
        "UPDATE jobs SET title = ?1, job_ad_text = ?2, triqui_size = ?3, updated_at = datetime('now') WHERE id = ?4"
    };
    let updated = conn
        .execute(
            sql,
            rusqlite::params![title, job_ad_text, triqui_size, job_id],
        )
        .map_err(db_err)?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Job".into(),
            id: job_id.to_string(),
        }));
    }
    // An ad landing on a job that had none is **not** an invalidation: every
    // embed the running cascade has produced is ad-independent, so the run
    // keeps its vectors and its next wave scores from the stored chunks.
    // Cancelling here would only throw away mid-flight embedding. Edits to an
    // ad that already exists are the opposite — scores against the old ad are
    // stale — and still cancel.
    let ad_was_absent = current
        .job_ad_text
        .as_deref()
        .map_or(true, |t| t.trim().is_empty());
    let ad_now_present = job_ad_text
        .as_deref()
        .map_or(false, |t| !t.trim().is_empty());
    let ad_lands_first_time = inputs_changed && ad_was_absent && ad_now_present;

    // One invalidation path for every ranking input. The rubric is the only thing a
    // rank reads, so the response to an edit is: embed the one text that changed,
    // re-project, and let any run in flight carry on. A cascade computes candidate
    // *vectors*, which are text-only and ad-independent, so nothing about this edit
    // invalidates its work — cancelling used to throw away embedding that was still
    // good. A changed *ad* additionally empties the judge pool, because every
    // memoized verdict is keyed on the ad's content hash.
    if inputs_changed {
        let ad_edited = !ad_lands_first_time;
        let invalidation =
            libtriqui_core::rescore::inputs_changed(&state.runtime, job_id, ad_edited)?;
        log::info!(
            "update_job: {job_id} re-projected — {} candidates, {} rubric vector(s) embedded{}",
            invalidation.rescored,
            invalidation.embedded,
            if ad_edited { ", judge pool cleared" } else { "" },
        );
    }
    // Re-judging is the only expensive part an edit can leave behind, and only a run
    // can do it. So: start a cascade unless one is already live (which refuses the
    // spawn itself and will pick the work up at its next wave boundary). An ad that
    // lands on a job that had none used to need this too — its import-time run had
    // already converged with nothing scoreable.
    let start_cascade = inputs_changed
        && {
            conn.query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
                > 0
        };
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(db_err)?;
    drop(conn);
    if start_cascade {
        match crate::serve::handlers::rank::spawn_cascade(state, job_id) {
            Ok(true) => log::info!("rank: inputs changed for {job_id} — cascade started"),
            Ok(false) => log::info!("rank: cascade already running for {job_id}"),
            Err(e) => log::warn!("rank: could not start cascade for {job_id}: {e}"),
        }
    }
    ok_json(job)
}

/// Update the rubric weights (job-ad + the two feedback rows). Weights are ranking
/// inputs, so this bumps `inputs_updated_at` (drives the stale badge).
pub fn update_job_weights(
    state: &Arc<ServeState>,
    job_id: &str,
    ad_weight: f64,
    fb_job_weight: f64,
    fb_profile_weight: f64,
) -> Result<Value, ProtocolError> {
    if ad_weight < 0.0 || fb_job_weight < 0.0 || fb_profile_weight < 0.0 {
        return Err(domain(&DomainError::ValidationError {
            message: "Rubric weights must be non-negative".into(),
        }));
    }
    let conn = state.db.0.lock().map_err(lock_err)?;
    let updated = conn
        .execute(
            "UPDATE jobs SET ad_weight = ?1, fb_job_weight = ?2, fb_profile_weight = ?3,
                             inputs_updated_at = datetime('now'), updated_at = datetime('now'), inputs_version = inputs_version + 1 WHERE id = ?4",
            rusqlite::params![ad_weight, fb_job_weight, fb_profile_weight, job_id],
        )
        .map_err(db_err)?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Job".into(),
            id: job_id.to_string(),
        }));
    }
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(db_err)?;
    drop(conn);
    // Weights are rubric too, so they go down the same path as text: re-project,
    // nothing cancelled. The judge never read a weight — its grade is a win-rate
    // and `ad_weight` only scales it — so every memoized verdict still holds.
    let invalidation = libtriqui_core::rescore::inputs_changed(&state.runtime, job_id, false)?;
    log::info!(
        "update_job_weights: {job_id} re-projected ({} candidates, capacity now {:.1})",
        invalidation.rescored,
        invalidation.rescored as f64,
    );
    ok_json(job)
}

pub fn archive_job(
    state: &Arc<ServeState>,
    job_id: &str,
    archived: bool,
) -> Result<(), ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let updated = conn
        .execute(
            "UPDATE jobs SET archived = ?1, updated_at = datetime('now') WHERE id = ?2",
            rusqlite::params![archived as i64, job_id],
        )
        .map_err(db_err)?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Job".into(),
            id: job_id.to_string(),
        }));
    }
    Ok(())
}

pub fn empty_candidates(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    // Delete all candidates for this job. Criteria are kept. FK ON DELETE CASCADE
    // removes each candidate's criterion_matches + candidate_scores rows.
    let deleted = conn
        .execute(
            "DELETE FROM candidates WHERE job_id = ?1",
            rusqlite::params![job_id],
        )
        .map_err(db_err)?;
    // Also reset last_ranked_at since the triqui is now empty.
    conn.execute(
        "UPDATE jobs SET last_ranked_at = NULL, updated_at = datetime('now') WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(db_err)?;
    log::info!("emptied {deleted} candidates for job {job_id}");
    ok_json(deleted as i32)
}

pub fn delete_job(state: &Arc<ServeState>, job_id: &str) -> Result<(), ProtocolError> {
    {
        let conn = state.db.0.lock().map_err(lock_err)?;
        // criteria/candidates cascade via FK ON DELETE CASCADE (PRAGMA foreign_keys = ON).
        conn.execute("DELETE FROM jobs WHERE id = ?1", rusqlite::params![job_id])
            .map_err(db_err)?;
    }
    watch::stop_watching(state, job_id);
    Ok(())
}

/// Set (or clear, with an empty string) the job's watched CV directory and
/// restart its watcher. The directory is created when missing so a typed
/// fresh path just works. Selecting a directory (Browse…) therefore restarts
/// watching AND auto-import: a catch-up scan imports the files already there.
pub fn set_job_cv_dir(
    state: &Arc<ServeState>,
    job_id: &str,
    directory: &str,
) -> Result<Value, ProtocolError> {
    let dir = directory.trim().to_string();
    let conn = state.db.0.lock().map_err(lock_err)?;
    let updated = conn
        .execute(
            "UPDATE jobs SET import_directory_path = ?1, updated_at = datetime('now') WHERE id = ?2",
            rusqlite::params![&dir, job_id],
        )
        .map_err(db_err)?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Job".into(),
            id: job_id.to_string(),
        }));
    }
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(db_err)?;
    drop(conn);
    watch::watch_job(state, job_id);
    ok_json(job)
}

/// Make `job_id` the only watched CV directory (single-watcher policy). The
/// gui sends this whenever the viewed job changes; serve restarts watching,
/// creates the directory when missing, and runs a catch-up import scan so
/// files added while the job was not viewed are imported too. Returns the job
/// so callers can refresh state.
pub fn watch_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let job = conn
        .query_row(
            &format!("SELECT {JOB_COLS} FROM jobs WHERE id = ?1"),
            rusqlite::params![job_id],
            job_from_row,
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => domain(&DomainError::NotFound {
                entity: "Job".into(),
                id: job_id.to_string(),
            }),
            e => domain(&db_err(e)),
        })?;
    drop(conn);
    watch::watch_job(state, job_id);
    ok_json(job)
}

/// Backend-owned state behind the gui's Re-rank button (state contract: the
/// gui projects this; it computes nothing). `rerank_dirty` is version-based —
/// `inputs_version != ranked_version` — so an edit landing in the same second
/// as a rank still reads as dirty.
pub fn job_ui_state(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let (cv_imported, ad_set, inputs_version, ranked_version): (i64, bool, i64, Option<i64>) = conn
        .query_row(
            "SELECT (SELECT COUNT(*) FROM candidates
                      WHERE job_id = jobs.id AND extraction_status = 'Extracted'),
                    COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0),
                    inputs_version, ranked_version
             FROM jobs WHERE jobs.id = ?1",
            rusqlite::params![job_id],
             |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => domain(&DomainError::NotFound {
                entity: "Job".into(),
                id: job_id.to_string(),
            }),
            e => domain(&db_err(e)),
        })?;
    // The right-moment nudge, backend-decided (the gui only maps the code):
    // an ad-less job whose embed-prefill cascade has converged has finished
    // every piece of ad-independent work — embedding won't get them closer
    // to a ranking, so the job ad is the one thing missing. Not while a run
    // is still going (the StatusBar shows that activity) and not for a
    // cancelled one (the user stopped it deliberately).
    let prefill_converged: bool = conn
        .query_row(
            "SELECT complete <> 0 AND cancelled = 0 FROM rank_state WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| r.get::<_, bool>(0),
        )
        .unwrap_or(false);
    let nudge = if !ad_set && cv_imported >= 1 && prefill_converged {
        Some("add-job-ad".to_string())
    } else {
        None
    };
    drop(conn);
    let engine_ready = state.runtime.status_info().status == "Running";
    // Both blockers are backend-owned: no CVs is meaningless, and no job ad
    // has nothing to score against (the old behaviour ranked anyway and
    // produced all-zero totals — read as "broken" and spun forever).
    let sort_allowed = cv_imported >= 1 && ad_set;
    let rerank_dirty = ranked_version.is_none() || ranked_version != Some(inputs_version);
    let reason = if cv_imported < 1 {
        Some("no-cvs".to_string())
    } else if !ad_set {
        Some("no-jobad".to_string())
    } else if !engine_ready {
        Some("engine-not-ready".to_string())
    } else if rerank_dirty {
        Some("stale".to_string())
    } else {
        Some("clean".to_string())
    };
    ok_json(JobUiState {
        engine_ready,
        sort_allowed,
        rerank_dirty,
        reason,
        nudge,
        rank: RankState {
            running: false,
            ranking: false,
            stage: None,
            wave: None,
            embedded_total: 0,
            embedded_done: 0,
            judged: 0,
            candidates: 0,
            converged: false,
            reason: None,
        },
    })
}

/// The report of the job's last rank (what ran, what was memoized).
pub fn get_sort_report(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(lock_err)?;
    let report = conn
        .query_row(
            "SELECT job_id, judge_mode, judge_pairs_fresh, judge_pairs_cached, mode, waves, complete, cancelled, created_at
             FROM sort_reports WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| {
                Ok(SortReport {
                    job_id: r.get(0)?,
                    judge_mode: r.get(1)?,
                    judge_pairs_fresh: r.get(2)?,
                    judge_pairs_cached: r.get(3)?,
                    mode: r.get(4)?,
                    waves: r.get(5)?,
                    complete: r.get(6)?,
                    cancelled: r.get(7)?,
                    created_at: r.get(8)?,
                })
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => domain(&DomainError::NotFound {
                entity: "Job".into(),
                id: job_id.to_string(),
            }),
            e => domain(&db_err(e)),
        })?;
    ok_json(report)
}

#[cfg(test)]
mod tests {
    use super::{folder_still_holds_cvs, job_folder_holds_cvs};

    /// Only a folder that actually holds CVs burns its `job-N` number. Pin the
    /// branches of both helpers: `folder_still_holds_cvs` is the raw
    /// non-empty-folder check, `job_folder_holds_cvs` the layout-aware policy
    /// over `<root>/job-N` (current `cv/` subfolder + legacy flat layout).
    #[test]
    fn only_folders_holding_cvs_block_their_job_number() {
        let dir = std::env::temp_dir().join(format!("triqui-jobdir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let sub = dir.join("job-1");

        assert!(
            !folder_still_holds_cvs(""),
            "no resolvable CV root means no folder to block on"
        );
        assert!(!job_folder_holds_cvs(&sub), "a missing folder is free");

        std::fs::create_dir_all(&sub).unwrap();
        assert!(
            !job_folder_holds_cvs(&sub),
            "an empty leftover folder is reusable"
        );

        // Current layout: the folder's `cv/` subfolder only blocks when it
        // itself holds files.
        let cv = sub.join("cv");
        std::fs::create_dir_all(&cv).unwrap();
        assert!(
            !job_folder_holds_cvs(&sub),
            "an empty cv/ subfolder is still an empty leftover"
        );
        std::fs::write(cv.join("leftover.pdf"), b"%PDF-1.4").unwrap();
        assert!(
            job_folder_holds_cvs(&sub),
            "a folder still holding a deleted job's CVs is not reusable"
        );

        // Legacy layout: CVs directly in the job folder block too — a new job
        // must not nest its fresh cv/ beside someone's data, which the
        // recursive import scan would sweep into the parent-watching job.
        let legacy = dir.join("job-2");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("old.pdf"), b"%PDF-1.4").unwrap();
        assert!(job_folder_holds_cvs(&legacy), "flat CVs block the number");

        // A *file* named cv is data, not the layout's subfolder.
        let tricky = dir.join("job-3");
        std::fs::create_dir_all(&tricky).unwrap();
        std::fs::write(tricky.join("cv"), b"not a folder").unwrap();
        assert!(job_folder_holds_cvs(&tricky));

        // Fails safe: something unreadable in the way must not hand the number out.
        let file = dir.join("job-4");
        std::fs::write(&file, b"not a directory").unwrap();
        assert!(job_folder_holds_cvs(&file));
        assert!(folder_still_holds_cvs(file.to_str().unwrap()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
