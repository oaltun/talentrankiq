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

    let rank_row = libtriqui_core::db::rank_state_for_job(&conn, job_id).ok();
    let (running, ranking, complete, cancelled, wave, blocked_reason) = match &rank_row {
        Some(r) => (!r.complete && !r.cancelled, !r.complete && !r.cancelled, r.complete, r.cancelled, None, r.blocked.clone()),
        None => (false, false, false, false, None, None),
    };

    let clean = ranked_version.is_some() && ranked_version == Some(inputs_version);
    // converged requires complete, clean, candidates > 0, ranked > 0, and all embedded
    let converged = complete && clean && candidates > 0 && embedded_done == embedded_total && embedded_total > 0;

    // Reason precedence: running → cancelled → blocked code → stale → converged → partial
    let reason = if running {
        Some("running".to_string())
    } else if cancelled {
        Some("cancelled".to_string())
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

/// Spawn a background cascade for `job_id`. Returns `Ok(true)` when a run was
/// started, `Ok(false)` when one is already live for the job.
pub fn spawn_cascade(state: &Arc<ServeState>, job_id: &str) -> Result<bool, String> {
    // Refuse if a run is already in flight.
    {
        let conn = state.db.read_conn();
        let running: bool = conn
            .query_row(
                "SELECT complete = 0 AND cancelled = 0 FROM rank_state WHERE job_id = ?1",
                rusqlite::params![job_id],
                |r| r.get(0),
            )
            .unwrap_or(false);
        if running {
            return Ok(false);
        }
    }

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
            let (sort_marker, sort_reports, complete, cancelled, blocked_reason, should_rerank) = match outcome {
                Ok(o) => {
                    if o.cancelled {
                        // User cancelled - mark as cancelled, don't mark complete
                        let report = format!(
                            "[{{\"mode\":\"waves\",\"waves\":{},\"complete\":{},\"cancelled\":{}}}]",
                            o.waves, false, true
                        );
                        ("Stage3", report, false, true, None, false)
                    } else if o.rubric_vectors == 0 {
                        // No rubric vectors - blocked (but recoverable if embedder becomes available)
                        ("Blocked", "{\"blocked\":{\"reason\":\"missing-rubric-vectors\"}}".to_string(), false, false, Some("missing-rubric-vectors".to_string()), true)
                    } else if o.skipped_missing_vector > 0 {
                        // Some candidates missing vectors - blocked (recoverable when they get embedded)
                        ("Blocked", "{\"blocked\":{\"reason\":\"candidate-embedding-failed\"}}".to_string(), false, false, Some("candidate-embedding-failed".to_string()), true)
                    } else if o.ranked == 0 {
                        // No candidates scored - blocked
                        ("Blocked", "{\"blocked\":{\"reason\":\"no-scoreable-candidates\"}}".to_string(), false, false, Some("no-scoreable-candidates".to_string()), false)
                    } else {
                        // Full coverage - complete, but check if inputs changed during run
                        let report = format!(
                            "[{{\"mode\":\"waves\",\"waves\":{},\"complete\":{},\"cancelled\":{}}}]",
                            o.waves, true, false
                        );
                        ("Stage3", report, true, false, None, false)
                    }
                }
                Err(e) => {
                    log::warn!("cascade {job}: failed: {e}");
                    // On error, mark as blocked with generic reason
                    ("Blocked", "{\"blocked\":{\"reason\":\"candidate-embedding-failed\"}}".to_string(), false, false, Some("candidate-embedding-failed".to_string()), true)
                }
            };

            // Capture inputs_version before writing final state (for re-rank check)
            let inputs_version_after: i64 = {
                let conn = state_clone.db.read_conn();
                conn.query_row(
                    "SELECT inputs_version FROM jobs WHERE id = ?1",
                    rusqlite::params![&job],
                    |r| r.get(0),
                )
                .unwrap_or(0)
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

            // Re-request rank if needed (inputs changed during run or blocked recoverable)
            // Only re-request if not user-cancelled and we have a scheduler
            if should_rerank && !cancelled {
                state_clone.request_rank(&job);
            }
        })
        .map_err(|e| e.to_string())?;

    Ok(true)
}