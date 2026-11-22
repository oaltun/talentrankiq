//! Ranking handlers: start/cancel/get state, system budget, viewed tracking,
//! and the `process_in_waves` entry point that drives the progressive cascade.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{DomainError, RankState, SystemBudget, SystemLoad};

use crate::serve::handlers::{domain, lock_err, ok_json, string_err};
use crate::serve::ServeState;

// ---------------------------------------------------------------------------
// Follow-up decision helper (pure, testable)
// ---------------------------------------------------------------------------

/// Determines whether a follow-up cascade should be requested after a run.
///
/// Returns `true` only when the run achieved full coverage (ranked > 0) and
/// the job's inputs changed during the run. Returns `false` for all blocked
/// outcomes, cancelled runs, and clean completions.
pub fn should_request_followup(
    outcome_is_full_coverage: bool,
    inputs_changed: bool,
    cancelled: bool,
) -> bool {
    if cancelled {
        return false;
    }
    // Only re-request if we had a full-coverage run AND inputs changed during it.
    outcome_is_full_coverage && inputs_changed
}

#[cfg(test)]
mod followup_tests {
    use super::should_request_followup;

    #[test]
    fn cancelled_never_reranks() {
        assert!(!should_request_followup(true, true, true));
        assert!(!should_request_followup(false, true, true));
        assert!(!should_request_followup(true, false, true));
    }

    #[test]
    fn blocked_missing_rubric_vectors_never_reranks() {
        assert!(!should_request_followup(false, true, false));
        assert!(!should_request_followup(false, false, false));
    }

    #[test]
    fn blocked_candidate_embedding_failed_never_reranks() {
        assert!(!should_request_followup(false, true, false));
        assert!(!should_request_followup(false, false, false));
    }

    #[test]
    fn blocked_no_scoreable_candidates_never_reranks() {
        assert!(!should_request_followup(false, true, false));
        assert!(!should_request_followup(false, false, false));
    }

    #[test]
    fn cascade_error_never_reranks() {
        assert!(!should_request_followup(false, true, false));
        assert!(!should_request_followup(false, false, false));
    }

    #[test]
    fn full_coverage_with_inputs_changed_reranks() {
        assert!(should_request_followup(true, true, false));
    }

    #[test]
    fn full_coverage_without_inputs_changed_does_not_rerank() {
        assert!(!should_request_followup(true, false, false));
    }
}

// ---------------------------------------------------------------------------
// Generation counters: one per job, used to signal a running cascade to stop.
// ---------------------------------------------------------------------------

/// Per-job generation counters. Bumping a job's counter is how a cancel reaches
/// a background cascade thread: it samples its generation at every wave and
/// stops when it no longer matches.
pub struct CascadeGens {
    gens: Mutex<HashMap<String, u64>>,
}

impl CascadeGens {
    pub fn new() -> Self {
        Self {
            gens: Mutex::new(HashMap::new()),
        }
    }

    /// Current generation for a job (0 when never bumped).
    pub fn current(&self, job_id: &str) -> u64 {
        self.gens
            .lock()
            .map(|m| m.get(job_id).copied().unwrap_or(0))
            .unwrap_or(0)
    }

    /// Increment and return the new generation for a job.
    pub fn bump(&self, job_id: &str) -> u64 {
        if let Ok(mut m) = self.gens.lock() {
            let e = m.entry(job_id.to_string()).or_insert(0);
            *e += 1;
            *e
        } else {
            0
        }
    }
}

impl Default for CascadeGens {
    fn default() -> Self {
        Self::new()
    }
}

/// The pre-cascade blockers, in one place so `get_rank_state` and
/// `job_ui_state` cannot drift apart. Returns the bare code of the highest
/// precedence blocker; callers render it `blocked:<code>`.
///
/// Precedence: no-cvs → no-jobad → no-active-embedder → engine-not-ready.
pub(crate) fn pre_cascade_blocked_reason(
    has_candidates: bool,
    ad_set: bool,
    embedder_ok: bool,
    engine_ready: bool,
) -> Option<&'static str> {
    if !has_candidates {
        Some("no-cvs")
    } else if !ad_set {
        Some("no-jobad")
    } else if !embedder_ok {
        Some("no-active-embedder")
    } else if !engine_ready {
        Some("engine-not-ready")
    } else {
        None
    }
}

fn job_exists(state: &Arc<ServeState>, job_id: &str) -> bool {
    state
        .db
        .read_conn()
        .query_row(
            "SELECT 1 FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |_| Ok(true),
        )
        .unwrap_or(false)
}

fn not_found(job_id: &str) -> ProtocolError {
    domain(&DomainError::NotFound {
        entity: "Job".into(),
        id: job_id.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Public handlers
// ---------------------------------------------------------------------------

/// Start a progressive cascade for a job. A second concurrent run for the same
/// job is a no-op (`Ok(false)` from `spawn_cascade`), not an error.
pub fn start_rank(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    if !job_exists(state, job_id) {
        return Err(not_found(job_id));
    }
    spawn_cascade(state, job_id).map_err(string_err)?;
    ok_json(true)
}

/// `process_in_waves` — the GUI's ranking trigger. Same as `start_rank` but
/// returns the ranking state so callers can immediately poll progress.
pub fn process_in_waves(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    if !job_exists(state, job_id) {
        return Err(not_found(job_id));
    }
    spawn_cascade(state, job_id).map_err(string_err)?;
    ok_json(true)
}

/// Request cancellation of a job's running cascade.
pub fn cancel_rank(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    if !job_exists(state, job_id) {
        return Err(not_found(job_id));
    }
    // Bump the generation so the cascade thread stops at its next wave.
    state.cascade_gens.bump(job_id);
    {
        let conn = state.db.writer();
        libtriqui_core::db::cancel_running_cascade(&conn, job_id)
            .map_err(string_err)?;
    }
    state.dispatch_event(libcv_state::Event::RankCancelled {
        job_id: job_id.to_string(),
    });
    ok_json(true)
}

/// The live cascade state of a job, folded from `rank_state` + candidate counts.
pub fn get_rank_state(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    if !job_exists(state, job_id) {
        return Err(not_found(job_id));
    }
    let conn = state.db.read_conn();

    let (candidates, embedded_total, embedded_done): (u32, u32, u32) = conn
        .query_row(
            "SELECT
                (SELECT COUNT(*) FROM candidates WHERE job_id = jobs.id AND extraction_status = 'Extracted' AND deleted = 0),
                (SELECT COUNT(*) FROM candidates WHERE job_id = jobs.id AND extraction_status = 'Extracted' AND deleted = 0 AND extracted_text IS NOT NULL),
                (SELECT COUNT(*) FROM candidates WHERE job_id = jobs.id AND extraction_status = 'Extracted' AND deleted = 0 AND vector IS NOT NULL)
             FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get::<_, i64>(0)? as u32, r.get::<_, i64>(1)? as u32, r.get::<_, i64>(2)? as u32)),
        )
        .map_err(|e| string_err(e.to_string()))?;

    let (inputs_version, ranked_version): (i64, Option<i64>) = conn
        .query_row(
            "SELECT inputs_version, ranked_version FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| string_err(e.to_string()))?;

    let ad_set: bool = conn
        .query_row(
            "SELECT COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0) FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get::<_, i64>(0).map(|v: i64| v != 0),
        )
        .unwrap_or(false);

    let rank_row = libtriqui_core::db::rank_state_for_job(&conn, job_id).ok();
    let (running, ranking, complete, cancelled, wave, blocked_reason) = match &rank_row {
        Some(r) => {
            let running = rank_row_running(r);
            (running, running, r.complete, r.cancelled, None, r.blocked.clone())
        }
        None => (false, false, false, false, None, None),
    };

    let embedder_ok = state.runtime.embedder().is_ok();
    let engine_ready = state.runtime.status_info().status == "Running";

    let clean = ranked_version.is_some() && ranked_version == Some(inputs_version);
    // converged requires complete, clean, candidates > 0, ranked > 0, and all embedded
    let converged = complete && clean && candidates > 0 && embedded_done == embedded_total && embedded_total > 0;

    // Reason precedence (identical to `job_ui_state`, shared pre-cascade
    // blockers first): running → cancelled → blocked pre-cascade → blocked
    // cascade code → stale → converged → partial. One vocabulary, so the GUI
    // never translates between endpoints.
    let reason = if running {
        Some("running".to_string())
    } else if cancelled {
        Some("cancelled".to_string())
    } else if let Some(code) =
        pre_cascade_blocked_reason(candidates > 0, ad_set, embedder_ok, engine_ready)
    {
        Some(format!("blocked:{code}"))
    } else if let Some(blocked) = blocked_reason {
        Some(format!("blocked:{}", blocked))
    } else if ranked_version.is_none() || !clean {
        Some("stale".to_string())
    } else if converged {
        Some("converged".to_string())
    } else {
        Some("partial".to_string())
    };

    ok_json(RankState {
        running,
        ranking,
        stage: if running { Some("sort".to_string()) } else { None },
        wave,
        embedded_total,
        embedded_done,
        compared: 0,
        candidates,
        converged,
        reason,
    })
}

/// The current system-capacity budget.
pub fn system_budget(_state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    ok_json(SystemBudget {
        verdict: "full".to_string(),
        wave_scale: 1.0,
        thread_cap: std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1),
        embed_batch: None,
        holding: false,
        load: SystemLoad {
            cpu_pct: 0.0,
            loadavg_1: 0.0,
            mem_available_bytes: 0,
            mem_total_bytes: 0,
            on_battery: None,
        },
    })
}

/// Stamp `viewed_at` on a candidate.
pub fn mark_candidate_viewed(
    state: &Arc<ServeState>,
    candidate_id: &str,
) -> Result<Value, ProtocolError> {
    let mut conn = state.db.writer();
    let updated = conn
        .execute(
            "UPDATE candidates SET viewed_at = datetime('now') WHERE id = ?1",
            rusqlite::params![candidate_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Candidate".into(),
            id: candidate_id.to_string(),
        }));
    }
    ok_json(true)
}

// ---------------------------------------------------------------------------
// Cascade runner
// ---------------------------------------------------------------------------

/// Whether a run is still in flight for `job_id` and would hold the
/// `spawn_cascade` gate.
///
/// A blocked run (`sort = 'Blocked'`) is terminal, not in flight: `complete`
/// and `cancelled` are both false, but the run has finished with a blocked
/// reason. Treating it as in-flight is what wedged blocked jobs — a new
/// cascade could never start, and the GUI could not recover. So the gate
/// additionally excludes `sort = 'Blocked'`.
fn rank_in_flight(conn: &rusqlite::Connection, job_id: &str) -> bool {
    conn.query_row(
        "SELECT complete = 0 AND cancelled = 0 AND sort <> 'Blocked' FROM rank_state WHERE job_id = ?1",
        rusqlite::params![job_id],
        |r| r.get(0),
    )
    .unwrap_or(false)
}

/// Whether a stored `rank_state` row represents a run still in flight.
///
/// Shared by `get_rank_state` and `job_ui_state` so both report the same
/// `running`/`ranking`. A blocked run is terminal, not in flight: `blocked`
/// is `Some` exactly when `sort = 'Blocked'`, so a blocked row yields `false`
/// even though `complete` and `cancelled` are both false.
pub(crate) fn rank_row_running(row: &libtriqui_core::db::RankStateRow) -> bool {
    !row.complete && !row.cancelled && row.blocked.is_none()
}

/// Spawn a background cascade for `job_id`. Returns `Ok(true)` when a run was
/// started, `Ok(false)` when one is already live for the job.
pub fn spawn_cascade(state: &Arc<ServeState>, job_id: &str) -> Result<bool, String> {
    // Refuse if a run is already in flight. A blocked run is terminal, not in
    // flight, so a new cascade may start over it.
    {
        let conn = state.db.read_conn();
        if rank_in_flight(&conn, job_id) {
            return Ok(false);
        }
    }

    // Capture the input version the run is about to score, before marking the
    // run in flight. The post-run read compares against this to decide whether
    // an input landed while the run was going.
    let inputs_version_at_start: i64 = {
        let conn = state.db.read_conn();
        conn.query_row(
            "SELECT inputs_version FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?
    };

    // Mark the run as in flight.
    {
        let conn = state.db.writer();
        libtriqui_core::db::set_rank_state(&conn, job_id, "Stage0", "[]", false, false)
            .map_err(|e| e.to_string())?;
    }

    let gen = state.cascade_gens.bump(job_id);
    let state_clone = Arc::clone(state);
    let job = job_id.to_string();

    std::thread::Builder::new()
        .name(format!("cascade-{job}"))
        .spawn(move || {
            let cfg = libtriqui_core::cascade::WaveConfig {
                candidates_per_wave: state_clone.config.candidates_per_wave.max(1),
                max_waves: state_clone.config.waves,
                wave_scale: 1.0,
            };

            let gens = Arc::clone(&state_clone.cascade_gens);
            let jid = job.clone();
            let cancel_state = Arc::clone(&state_clone);
            let is_cancelled = move || {
                if gens.current(&jid) != gen {
                    return true;
                }
                // A DB-side cancel flag also stops the run.
                let conn = cancel_state.db.read_conn();
                conn.query_row(
                    "SELECT cancelled FROM rank_state WHERE job_id = ?1",
                    rusqlite::params![jid],
                    |r| r.get::<_, i64>(0),
                )
                .map(|c| c != 0)
                .unwrap_or(false)
            };

            let progress_state = Arc::clone(&state_clone);
            let progress_job = job.clone();
            let on_wave = move |wave: u32, done: usize, total: usize| {
                progress_state.events.emit(
                    "rank-progress",
                    &libtriqui_client::types::RankProgress {
                        job_id: progress_job.clone(),
                        stage: "sort".to_string(),
                        wave,
                        embedded_done: done as u32,
                        embedded_total: total as u32,
                        compared: 0,
                        candidates: total as u32,
                        running: true,
                        reason: None,
                    },
                );
            };

            let outcome = libtriqui_core::cascade::run(
                &state_clone.runtime,
                &job,
                &cfg,
                &is_cancelled,
                &on_wave,
            );

            // Determine terminal state from CascadeOutcome
            let (sort_marker, sort_reports, complete, cancelled, blocked_reason, is_full_coverage) = match outcome {
                Ok(o) => {
                    if o.cancelled {
                        // User cancelled - mark as cancelled, don't mark complete
                        let report = format!(
                            "[{{\"mode\":\"waves\",\"waves\":{},\"complete\":{},\"cancelled\":{}}}]",
                            o.waves, false, true
                        );
                        ("Stage3", report, false, true, None, false)
                    } else if o.rubric_vectors == 0 {
                        // No rubric vectors - blocked
                        ("Blocked", "{\"blocked\":{\"reason\":\"missing-rubric-vectors\"}}".to_string(), false, false, Some("missing-rubric-vectors".to_string()), false)
                    } else if o.skipped_missing_vector > 0 {
                        // Some candidates missing vectors - blocked
                        ("Blocked", "{\"blocked\":{\"reason\":\"candidate-embedding-failed\"}}".to_string(), false, false, Some("candidate-embedding-failed".to_string()), false)
                    } else if o.ranked == 0 {
                        // No candidates scored - blocked
                        ("Blocked", "{\"blocked\":{\"reason\":\"no-scoreable-candidates\"}}".to_string(), false, false, Some("no-scoreable-candidates".to_string()), false)
                    } else {
                        // Full coverage - complete
                        let report = format!(
                            "[{{\"mode\":\"waves\",\"waves\":{},\"complete\":{},\"cancelled\":{}}}]",
                            o.waves, true, false
                        );
                        ("Stage3", report, true, false, None, true)
                    }
                }
                Err(e) => {
                    log::warn!("cascade {job}: failed: {e}");
                    // On error, mark as blocked with generic reason
                    ("Blocked", "{\"blocked\":{\"reason\":\"candidate-embedding-failed\"}}".to_string(), false, false, Some("candidate-embedding-failed".to_string()), false)
                }
            };

            {
                let conn = state_clone.db.writer();
                let _ = libtriqui_core::db::set_rank_state(
                    &conn, &job, sort_marker, &sort_reports, complete, cancelled,
                );
            }

            if cancelled {
                state_clone.dispatch_event(libcv_state::Event::RankCancelled {
                    job_id: job.clone(),
                });
            }
            state_clone.events.emit(
                "rank-progress",
                &libtriqui_client::types::RankProgress {
                    job_id: job.clone(),
                    stage: "done".to_string(),
                    wave: 0,
                    embedded_done: 0,
                    embedded_total: 0,
                    compared: 0,
                    candidates: 0,
                    running: false,
                    reason: blocked_reason,
                },
            );

            // Compare the start-captured version with the value now. A bump
            // during the run means this run scored stale inputs, so a
            // follow-up is owed. Reading "after" twice (the old code) compared
            // the same value to itself and never detected the change.
            let inputs_version_after: i64 = {
                let conn = state_clone.db.read_conn();
                conn.query_row(
                    "SELECT inputs_version FROM jobs WHERE id = ?1",
                    rusqlite::params![&job],
                    |r| r.get(0),
                )
                .unwrap_or(0)
            };
            let inputs_changed = inputs_version_after != inputs_version_at_start;
            let should_rerank = should_request_followup(is_full_coverage, inputs_changed, cancelled);
            if should_rerank {
                state_clone.request_rank(&job);
            }
        })
        .map_err(|e| e.to_string())?;

    Ok(true)
}

#[cfg(test)]
mod running_state_tests {
    use super::{rank_in_flight, rank_row_running};
    use libtriqui_core::db::{
        open_single_connection_at, set_blocked_rank_state, set_rank_state, RankStateRow,
    };

    fn temp_db() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("triq-rank-inflight-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn conn_with_job() -> (rusqlite::Connection, std::path::PathBuf) {
        let path = temp_db();
        let conn = open_single_connection_at(&path).expect("open db");
        conn.execute("INSERT INTO jobs (id, title) VALUES ('job1', 'Job One')", [])
            .unwrap();
        (conn, path)
    }

    fn row(complete: bool, cancelled: bool, blocked: Option<&str>) -> RankStateRow {
        RankStateRow {
            sort: if blocked.is_some() { "Blocked".to_string() } else { "Stage3".to_string() },
            sort_reports_json: "[]".to_string(),
            complete,
            cancelled,
            blocked: blocked.map(|s| s.to_string()),
        }
    }

    // --- spawn_cascade running check (rank_in_flight) ---

    #[test]
    fn blocked_run_does_not_hold_spawn_gate() {
        let (conn, path) = conn_with_job();
        set_blocked_rank_state(&conn, "job1", "missing-rubric-vectors").unwrap();
        assert!(
            !rank_in_flight(&conn, "job1"),
            "a blocked run is terminal, so a new cascade may start"
        );
        drop(conn);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn live_run_holds_spawn_gate() {
        let (conn, path) = conn_with_job();
        set_rank_state(&conn, "job1", "Stage0", "[]", false, false).unwrap();
        assert!(rank_in_flight(&conn, "job1"), "an in-flight run is not re-spawned");
        drop(conn);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn complete_run_does_not_hold_spawn_gate() {
        let (conn, path) = conn_with_job();
        set_rank_state(&conn, "job1", "Stage3", "[]", true, false).unwrap();
        assert!(!rank_in_flight(&conn, "job1"), "a completed run is not in flight");
        drop(conn);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cancelled_run_does_not_hold_spawn_gate() {
        let (conn, path) = conn_with_job();
        set_rank_state(&conn, "job1", "Stage3", "[]", false, true).unwrap();
        assert!(!rank_in_flight(&conn, "job1"), "a cancelled run is not in flight");
        drop(conn);
        let _ = std::fs::remove_file(&path);
    }

    // --- get_rank_state / job_ui_state running derivation (rank_row_running) ---

    #[test]
    fn blocked_row_reports_not_running() {
        assert!(
            !rank_row_running(&row(false, false, Some("candidate-embedding-failed"))),
            "a blocked row must report running = false"
        );
    }

    #[test]
    fn live_row_reports_running() {
        assert!(rank_row_running(&row(false, false, None)), "an in-flight row is running");
    }

    #[test]
    fn complete_row_reports_not_running() {
        assert!(!rank_row_running(&row(true, false, None)), "a complete row is not running");
    }

    #[test]
    fn cancelled_row_reports_not_running() {
        assert!(!rank_row_running(&row(false, true, None)), "a cancelled row is not running");
    }

    #[test]
    fn blocked_reason_variants_all_report_not_running() {
        for reason in [
            "missing-rubric-vectors",
            "candidate-embedding-failed",
            "no-scoreable-candidates",
        ] {
            assert!(
                !rank_row_running(&row(false, false, Some(reason))),
                "blocked:{reason} must not be running"
            );
        }
    }
}
#[cfg(test)]
mod convergence_tests {
    //! In-process convergence invariant tests. No mocks: a real
    //! `EngineRuntime` with a downloaded embedder, the real scheduler, the
    //! real cascade, and a real SQLite database. They pin the guarantees that
    //! an input change mid-run is never lost and never falsely converged.

    use super::*;
    use crate::serve::watch::WatchHandle;
    use crate::serve::EventSink;
    use libcv_pipeline::PipelineRunner;
    use libcv_state::State as CvState;
    use libtriqui_core::config::CascadeConfig;
    use libtriqui_core::db::{open_connection_at, ImportState};
    use libtriqui_core::runtime::{EngineRuntime, RuntimeConfig};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const MODEL: &str = "harrier-oss-v1-270m-ONNX-q4f16";

    struct Harness {
        state: Arc<ServeState>,
        events: std::sync::mpsc::Receiver<String>,
        dir: std::path::PathBuf,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            self.state.runtime.shutdown();
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn harness(with_scheduler: bool) -> Harness {
        let mut dir = std::env::temp_dir();
        dir.push(format!("triq-convergence-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create app dir");
        let db_path = dir.join("triqui.db");
        let db = open_connection_at(&db_path).expect("open db");
        let runtime = EngineRuntime::new(
            RuntimeConfig::discover(Some(MODEL.to_string()), None),
            db_path.clone(),
        );
        runtime.switch_model(MODEL.to_string()).expect("real embedder available");
        let (events, rx) = EventSink::for_test();
        let (ad_tx, _ad_rx) = std::sync::mpsc::channel();
        let state = Arc::new(ServeState {
            app_dir: dir.clone(),
            app_version: "test".to_string(),
            db: Arc::new(db),
            db_path,
            import_state: ImportState::new(),
            runtime,
            events,
            watch: WatchHandle::new(ad_tx),
            config: CascadeConfig { waves: 0, candidates_per_wave: 1 },
            cascade_gens: Arc::new(CascadeGens::new()),
            rank_scheduler: std::sync::OnceLock::new(),
            download_progress: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cv_state: Arc::new(Mutex::new(CvState::default())),
            pipeline: Arc::new(PipelineRunner::new()),
        });
        if with_scheduler {
            state.init_rank_scheduler(Arc::clone(&state));
        }
        Harness { state, events: rx, dir }
    }

    /// Seed a job with `cv_count` distinct extracted candidates. Their
    /// centroids are absent, so the real embedder fills them during the
    /// cascade's `ensure_inputs` — no hand-made vectors anywhere.
    fn seed_job(state: &Arc<ServeState>, cv_count: usize, ad: &str) -> String {
        let job = "job1".to_string();
        let conn = state.db.writer();
        conn.execute(
            "INSERT INTO jobs (id, title, job_ad_text, ad_weight)
             VALUES (?1, 'Job', ?2, 10.0)",
            rusqlite::params![&job, ad],
        )
        .unwrap();
        for i in 0..cv_count {
            conn.execute(
                "INSERT INTO candidates
                   (id, job_id, source_file_path, file_name, file_type,
                    extraction_status, extracted_text)
                 VALUES (?1, ?2, ?3, ?4, 'txt', 'Extracted', ?5)",
                rusqlite::params![
                    format!("c{i}"),
                    &job,
                    format!("/tmp/c{i}.txt"),
                    format!("c{i}.txt"),
                    candidate_text(i),
                ],
            )
            .unwrap();
        }
        job
    }

    fn candidate_text(i: usize) -> String {
        format!(
            "Candidate {i}. Skills: {} and {}. Experience: {} years.",
            ["Rust", "TypeScript", "React", "Python", "Go"][i % 5],
            ["Kubernetes", "SQL", "AWS", "Docker", "GraphQL"][(i + 2) % 5],
            i + 1
        )
    }

    fn versions(state: &Arc<ServeState>, job: &str) -> (i64, Option<i64>) {
        let conn = state.db.read_conn();
        conn.query_row(
            "SELECT inputs_version, ranked_version FROM jobs WHERE id = ?1",
            rusqlite::params![job],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    }

    fn bump_inputs(state: &Arc<ServeState>, job: &str, new_ad: Option<&str>) {
        let conn = state.db.writer();
        match new_ad {
            Some(ad) => conn
                .execute(
                    "UPDATE jobs SET job_ad_text = ?2, inputs_version = inputs_version + 1 WHERE id = ?1",
                    rusqlite::params![job, ad],
                )
                .unwrap(),
            None => conn
                .execute(
                    "UPDATE jobs SET inputs_version = inputs_version + 1 WHERE id = ?1",
                    rusqlite::params![job],
                )
                .unwrap(),
        };
    }

    fn rank_state_json(state: &Arc<ServeState>, job: &str) -> serde_json::Value {
        get_rank_state(state, job).expect("get_rank_state")
    }

    /// Block until the cascade reports its first scored wave (stage `sort`,
    /// wave >= 1), so a subsequent input change is guaranteed to land after
    /// the run captured its start version and read its rubric.
    fn wait_for_first_wave(rx: &std::sync::mpsc::Receiver<String>, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(line) = rx.recv_timeout(Duration::from_millis(100)) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v["event"] == "rank-progress"
                        && v["data"]["stage"] == "sort"
                        && v["data"]["wave"].as_u64().unwrap_or(0) >= 1
                    {
                        return;
                    }
                }
            }
        }
        panic!("cascade never reported a scored wave within {timeout:?}");
    }

    /// Drain the event stream until `finished` holds, counting `done` events.
    /// Returns the number of done events seen.
    fn wait_until<F: Fn() -> bool>(
        rx: &std::sync::mpsc::Receiver<String>,
        finished: F,
        timeout: Duration,
    ) -> usize {
        let deadline = Instant::now() + timeout;
        let mut done = 0usize;
        while Instant::now() < deadline {
            if finished() {
                return done;
            }
            if let Ok(line) = rx.recv_timeout(Duration::from_millis(50)) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v["event"] == "rank-progress" && v["data"]["stage"] == "done" {
                        done += 1;
                    }
                }
            }
        }
        if finished() {
            done
        } else {
            panic!("condition never became true within {timeout:?}");
        }
    }

    fn drain_done(rx: &std::sync::mpsc::Receiver<String>, window: Duration) -> usize {
        let deadline = Instant::now() + window;
        let mut done = 0usize;
        while Instant::now() < deadline {
            if let Ok(line) = rx.recv_timeout(Duration::from_millis(50)) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if v["event"] == "rank-progress" && v["data"]["stage"] == "done" {
                        done += 1;
                    }
                }
            }
        }
        done
    }

    /// 2.4 / 5.1 — an ad change that lands mid-run triggers exactly one
    /// follow-up cascade, and the job ends converged with points reflecting
    /// the new ad (the candidate whose text *is* the new ad ranks first).
    #[test]
    fn mid_run_ad_change_produces_one_followup_and_converges_on_new_input() {
        let h = harness(true);
        let job = seed_job(
            &h.state,
            6,
            "Senior platform engineer. Required: Kubernetes, Go, AWS, Docker, SQL.",
        );

        process_in_waves(&h.state, &job).expect("start cascade");
        wait_for_first_wave(&h.events, Duration::from_secs(180));

        // The new ad is exactly candidate 2's text.
        bump_inputs(&h.state, &job, Some(&candidate_text(2)));

        let done = wait_until(
            &h.events,
            || {
                let st = rank_state_json(&h.state, &job);
                st["converged"] == true
            },
            Duration::from_secs(240),
        );
        // Catch a late second done event before asserting the count.
        let extra = drain_done(&h.events, Duration::from_secs(2));
        assert_eq!(
            done + extra,
            2,
            "exactly one follow-up run (initial + follow-up)"
        );

        let (inputs_version, ranked_version) = versions(&h.state, &job);
        assert_eq!(inputs_version, 1);
        assert_eq!(ranked_version, Some(1), "follow-up scored the new input");

        let st = rank_state_json(&h.state, &job);
        assert_eq!(st["converged"], true);
        assert_eq!(st["reason"], "converged");

        let conn = h.state.db.read_conn();
        let (top_id, top_points): (String, f64) = conn
            .query_row(
                "SELECT id, points FROM candidates WHERE job_id = ?1 ORDER BY rank ASC LIMIT 1",
                rusqlite::params![&job],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(top_id, "c2", "the candidate matching the new ad ranks first");
        assert!(
            top_points > 9.0,
            "matching the ad should score near the 10-point capacity, got {top_points}"
        );
    }

    /// 3.2 / 5.2 — two rank requests that arrive while a run is in flight are
    /// not dropped and coalesce into a single follow-up run.
    #[test]
    fn two_requests_during_a_run_coalesce_into_one_followup() {
        let h = harness(true);
        let job = seed_job(&h.state, 5, "Senior Rust engineer. Kubernetes, Go, AWS.");

        process_in_waves(&h.state, &job).expect("start cascade");
        wait_for_first_wave(&h.events, Duration::from_secs(180));

        h.state.request_rank(&job);
        h.state.request_rank(&job);

        let done = wait_until(
            &h.events,
            || {
                let st = rank_state_json(&h.state, &job);
                st["converged"] == true
            },
            Duration::from_secs(240),
        );
        let extra = drain_done(&h.events, Duration::from_secs(2));
        assert_eq!(
            done + extra,
            2,
            "two in-flight requests coalesce into one follow-up (initial + one)"
        );

        let (inputs_version, ranked_version) = versions(&h.state, &job);
        assert_eq!(ranked_version, Some(inputs_version));
        assert_eq!(rank_state_json(&h.state, &job)["reason"], "converged");
    }

    /// 5.3 — a run that scored version N is never stamped N+1, even when an
    /// input lands mid-run. With no scheduler the follow-up cannot start, so
    /// the first run's terminal state is observable: it stays stale.
    #[test]
    fn a_run_scored_against_version_n_is_not_stamped_n_plus_one() {
        let h = harness(false);
        let job = seed_job(&h.state, 5, "Senior Rust engineer. Kubernetes, Go, AWS.");
        assert_eq!(versions(&h.state, &job), (0, None));

        process_in_waves(&h.state, &job).expect("start cascade");
        wait_for_first_wave(&h.events, Duration::from_secs(180));

        bump_inputs(&h.state, &job, None);

        // Wait for the run's completion marker to land.
        let _ = wait_until(
            &h.events,
            || versions(&h.state, &job).1.is_some(),
            Duration::from_secs(240),
        );
        let (inputs_version, ranked_version) = versions(&h.state, &job);
        assert_eq!(inputs_version, 1, "the live input version did advance");
        assert_eq!(
            ranked_version,
            Some(0),
            "the completion marker stays at the scored version"
        );

        let st = rank_state_json(&h.state, &job);
        assert_eq!(st["converged"], false, "a stale run must not report converged");
        assert_eq!(st["reason"], "stale");

        // And it stays that way: no follow-up is possible without a scheduler.
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(versions(&h.state, &job).1, Some(0));
    }

    /// 3.2 — a `CvDeleted` event makes `dispatch_event` request a rank, and the
    /// retired candidate leaves the ranked set and loses its score rows. The
    /// scan persists the deletion + version bump before dispatching, so this
    /// mirrors the real directory-scan sequence.
    #[test]
    fn cv_deleted_event_reruns_and_drops_the_candidate_from_scores() {
        let h = harness(true);
        let job = seed_job(&h.state, 4, "Senior Rust engineer. Kubernetes, Go, AWS.");

        process_in_waves(&h.state, &job).expect("start cascade");
        wait_until(
            &h.events,
            || rank_state_json(&h.state, &job)["converged"] == true,
            Duration::from_secs(240),
        );
        assert_eq!(rank_state_json(&h.state, &job)["candidates"], 4);
        let (inputs_before, ranked_before) = versions(&h.state, &job);
        assert_eq!(ranked_before, Some(inputs_before), "starts converged");

        // What the directory scan persists before dispatching CvDeleted.
        {
            let conn = h.state.db.writer();
            conn.execute("UPDATE candidates SET deleted = 1 WHERE id = 'c1'", [])
                .unwrap();
            conn.execute("DELETE FROM candidate_chunks WHERE candidate_id = 'c1'", [])
                .unwrap();
            conn.execute("DELETE FROM candidate_scores WHERE candidate_id = 'c1'", [])
                .unwrap();
            conn.execute(
                "UPDATE jobs SET inputs_version = inputs_version + 1 WHERE id = ?1",
                rusqlite::params![&job],
            )
            .unwrap();
        }

        h.state.dispatch_event(libcv_state::Event::CvDeleted {
            cv_id: "c1".into(),
            job_id: job.clone(),
        });

        // The event must have requested a rank: a new run scores the remaining
        // three candidates and converges.
        wait_until(
            &h.events,
            || {
                let st = rank_state_json(&h.state, &job);
                st["converged"] == true && st["candidates"] == 3
            },
            Duration::from_secs(240),
        );

        let (inputs_after, ranked_after) = versions(&h.state, &job);
        assert_eq!(
            inputs_after,
            inputs_before + 1,
            "the scan's single version bump is what the run scored"
        );
        assert_eq!(ranked_after, Some(inputs_after), "the follow-up converged");

        let conn = h.state.db.read_conn();
        let c1_scores: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidate_scores WHERE candidate_id = 'c1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(c1_scores, 0, "the retired candidate has no score rows");
        // The retired candidate's rank is left untouched by the cascade (it filters deleted=0).
        // What matters is it's excluded from the ranked set and list_candidates.
        let scored: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT candidate_id) FROM candidate_scores cs
                 JOIN candidates c ON c.id = cs.candidate_id
                 WHERE c.job_id = ?1 AND c.deleted = 0",
                rusqlite::params![&job],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(scored, 3, "the remaining candidates were re-scored");

    }
}
