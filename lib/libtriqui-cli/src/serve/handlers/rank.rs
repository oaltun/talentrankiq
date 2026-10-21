//! Progressive-ranking cascade handlers (start/cancel/poll).
//!
//! The cascade runs on a background thread. The gui drives it via
//! `start_rank` (returns immediately with a run id), polls `rank-progress`
//! events, and can call `cancel_rank` to stop it. `get_rank_state` is the
//! synchronous getter for the current `RankState` snapshot.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::Value;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{RankProgress, RankState, SortReport, SystemBudget, SystemLoad};
use libtriqui_core::cascade::{freeze_pool, project, run_scout_stage, run_wave, CascadeBudget};
use libtriqui_core::config::CascadeSection;
use libtriqui_core::db::{rank_state_for_job, set_rank_state};
use libtriqui_core::runtime::EngineRuntime;
use libtriqui_core::syscap;

use crate::serve::handlers::{domain, ok_json, string_err};
use crate::serve::ServeState;

/// How long a held runner waits between budget re-checks while waves are
/// paused (libsyscap design §4: `hold` = nearly out of RAM; the run stays
/// alive and resumes by itself when the verdict lifts).
const HOLD_POLL: std::time::Duration = std::time::Duration::from_secs(5);

/// In-memory registry of live cascade runner threads, keyed by job id. The
/// `u64` is the runner's generation, bumped on every spawn: a runner that is
/// no longer the newest for its job is *superseded* and must exit without
/// writing anything. This is the guard for the cancel-then-respawn race — an
/// old runner's terminal `finish(false)` landing after a newer run spawned
/// would overwrite the new run's `rank_state` row (the gui would show the new
/// run as cancelled while it works, and further spawns would stack on top).
///
/// Presence in the map means "a runner thread is alive for this job", which
/// [`spawn_cascade`] consults before refusing or superseding; entries are
/// removed when their runner retires.
#[derive(Default)]
pub struct CascadeGens {
    /// The live runner's generation per job. Removed when that runner retires.
    live: Mutex<HashMap<String, u64>>,
    /// Monotonic high-water mark per job — never removed, so a generation is
    /// never handed out twice and a stale runner from an earlier cycle can
    /// never look current again (it must stay superseded even after its
    /// replacement retired and a third run started).
    last: Mutex<HashMap<String, u64>>,
}

impl CascadeGens {
    pub fn new() -> Self {
        Self {
            live: Mutex::new(HashMap::new()),
            last: Mutex::new(HashMap::new()),
        }
    }

    fn has(&self, job_id: &str) -> bool {
        self.live
            .lock()
            .expect("cascade generations poisoned")
            .contains_key(job_id)
    }

    /// Whether a runner thread is still alive for `job_id`. The embedder
    /// switch waits on this before wiping vectors: a cancelled wave (or the
    /// scout stage) persists its vectors when it lands, so the nuke must
    /// follow the *retirement* of the threads, not the cancel flag — a late
    /// write would otherwise re-populate the wiped tables with rows stamped
    /// with the old model, which no later wave overwrites.
    pub fn is_live(&self, job_id: &str) -> bool {
        self.has(job_id)
    }

    /// Register a new runner for `job_id`, superseding any current one, and
    /// return its generation.
    fn bump(&self, job_id: &str) -> u64 {
        let gen = {
            let mut last = self.last.lock().expect("cascade generations poisoned");
            let e = last.entry(job_id.to_string()).or_insert(0);
            *e += 1;
            *e
        };
        self.live
            .lock()
            .expect("cascade generations poisoned")
            .insert(job_id.to_string(), gen);
        gen
    }

    fn is_current(&self, job_id: &str, gen: u64) -> bool {
        self.live
            .lock()
            .expect("cascade generations poisoned")
            .get(job_id)
            == Some(&gen)
    }

    /// Deregister a runner on exit. A superseded runner must not remove the
    /// entry — it belongs to the runner that replaced it.
    fn retire(&self, job_id: &str, gen: u64) {
        let mut live = self.live.lock().expect("cascade generations poisoned");
        if live.get(job_id) == Some(&gen) {
            live.remove(job_id);
        }
    }
}

/// Background state for a running cascade. One per job at a time.
struct CascadeRunner {
    job_id: String,
    runtime: Arc<EngineRuntime>,
    db_path: std::path::PathBuf,
    events: crate::serve::EventSink,
    cascade: CascadeSection,
    /// `syscap:` + `engine:` sections, resolved on the spawning (serve
    /// worker) thread where the config-path override is set — the runner
    /// thread itself never loads config.
    syscap: libconfigreader::SyscapSection,
    engine: libconfigreader::EngineSection,
    gens: Arc<CascadeGens>,
    gen: u64,
    /// Throttles the per-candidate `rank-progress` stream to ~10 Hz: embed
    /// callbacks arrive at engine speed, and an unthrottled event per CV
    /// makes the gui refetch + re-render the whole sheet each time — a big
    /// import freezes the webview until the embed ends (clicks and native
    /// dialogs go dead). Wave edges, hold notices, and terminal events
    /// bypass the throttle (state edges must always land).
    throttle: crate::serve::ProgressThrottle,
}

impl CascadeRunner {
    /// Whether a newer runner took this job over (cancel followed by a new
    /// spawn). A superseded runner owns nothing anymore: it must not write
    /// `rank_state`, must not emit progress, and must exit at the next
    /// boundary instead of spending waves on a run nobody wants.
    fn superseded(&self) -> bool {
        !self.gens.is_current(&self.job_id, self.gen)
    }

    fn run(self) {
        // Cloned ahead of the move: the retire must run after the loop exits,
        // whatever the exit was.
        let gens = Arc::clone(&self.gens);
        let job_id = self.job_id.clone();
        let gen = self.gen;
        self.run_loop();
        gens.retire(&job_id, gen);
    }

    fn run_loop(self) {
        let conn = match libtriqui_core::db::open_connection_at(&self.db_path) {
            Ok(c) => c,
            Err(e) => {
                log::error!("cascade: failed to open db for {}: {}", self.job_id, e);
                return;
            }
        };

        // A runner that was already superseded while waiting to start must not
        // even do the free stages — its writes would land on the newer run.
        if self.superseded() {
            log::info!(
                "cascade: runner for {} superseded before it started — exiting",
                self.job_id
            );
            return;
        }

        let candidate_count = self.count_candidates(&conn).unwrap_or(0);
        let is_small_job = candidate_count <= self.cascade.small_job_threshold() as i64;
        // The judge pool is frozen once per run and never scaled by the machine:
        // a pool that grew and shrank with the load would move every member's
        // set-relative grade, which is the one thing a win-rate must not do.
        let pool_size = if is_small_job {
            candidate_count.max(0) as usize
        } else {
            self.cascade.pool_size()
        };
        // Small jobs embed everything in one wave (the budget's wave scaling has
        // nothing to shrink); big jobs take their slice size per wave from the
        // live system budget below.
        let small_job_budget = CascadeBudget {
            wave1_budget: candidate_count.max(0) as usize,
            pool_size,
        };

        // Stage 0.5 — the scout. One cheap multilingual pass over every candidate
        // before the heavy embedder is asked for anything, so the waves that
        // follow are gated by a language-fair score (design §7/§8).
        self.emit_progress(RankProgress {
            job_id: self.job_id.clone(),
            stage: "scout".into(),
            wave: 0,
            embedded_done: 0,
            embedded_total: candidate_count as u32,
            judged: 0,
            candidates: candidate_count as u32,
            running: true,
            reason: None,
        });
        let scout_model = libtriqui_core::cascade::scout_model(&self.runtime);
        // A scout with no job ad to score against embeds the corpus and
        // defers its grading (scout.rs). The waves below catch it up at the
        // first boundary after an ad lands — one ad embed plus maxsim over
        // the scout chunks already on disk, no further corpus model calls.
        let mut scout_deferred = false;
        match run_scout_stage(&self.runtime, &self.job_id, &|_d, _t, _f, _a| {}) {
            Ok(Some(outcome)) => {
                scout_deferred = outcome.deferred;
                log::info!(
                    "cascade: scout stage done for {} (model {:?}, embedded {}, graded {}{})",
                    self.job_id,
                    scout_model,
                    outcome.embedded,
                    outcome.graded,
                    if outcome.deferred { ", grading deferred" } else { "" },
                );
            }
            Ok(None) => {}
            // A scout failure must not sink the run: the waves still converge, they
            // just have nothing to gate them on until the heavy embedder lands. Log
            // loudly and carry on.
            Err(e) => log::warn!("cascade: scout stage failed for {}: {}", self.job_id, e),
        }

        // The scout has graded the whole corpus, so the sheet can be projected —
        // this is the "ok sorting in front of the user as soon as possible"
        // moment: a complete, if coarse, ranking for every CV.
        match project(&self.runtime, &self.job_id) {
            Ok(report) => log::info!(
                "cascade: scout pass projected for {} ({} graded, {} still scout-only)",
                self.job_id,
                report.scored,
                report.scout_only
            ),
            Err(e) => log::warn!("cascade: scout projection failed for {}: {}", self.job_id, e),
        }

        // Freeze the judge pool now, from the projected order. One pool per run.
        if pool_size > 0 {
            match freeze_pool(&self.runtime, &self.job_id, pool_size) {
                Ok(n) => log::info!("cascade: judge pool frozen for {} with {n} members", self.job_id),
                Err(e) => log::warn!("cascade: could not freeze the pool for {}: {}", self.job_id, e),
            }
        }

        // Waves. Each one embeds its K1 slice, judges the top K2, re-sorts and
        // persists; the loop ends when nothing is left to embed, which makes the
        // last wave a full pass (the convergence guarantee, design §5).
        let mut wave: u32 = 0;
        let mut total_embedded: i64 = 0;
        let mut total_judged: u32 = 0;
        loop {
            // The supersede check comes first: a superseded runner must exit
            // silently even when the DB row (the newer run's) is not
            // cancelled — writing `finish(false)` here is exactly the
            // cancel-then-respawn clobber this guard exists for.
            if self.superseded() {
                log::info!(
                    "cascade: runner for {} superseded — exiting without writing",
                    self.job_id
                );
                return;
            }
            if self.cancelled() {
                log::info!("cascade: cancelled before wave {} for {}", wave + 1, self.job_id);
                self.finish(false);
                return;
            }

            // A scout that deferred its scoring grades the corpus here once an
            // ad exists: this is the "ad lands mid-run" path (design §9 — the
            // landing must not cancel the run, its embeds are ad-independent).
            // Cheap when it fires: the scout chunks are on disk, so the pass
            // costs one ad embed plus maxsim, and the wave's sort picks the
            // grades and the refreshed best-current-order up.
            if scout_deferred && self.job_has_ad(&conn) {
                match run_scout_stage(&self.runtime, &self.job_id, &|_d, _t, _f, _a| {}) {
                    Ok(Some(outcome)) if outcome.graded > 0 => {
                        scout_deferred = false;
                        log::info!(
                            "cascade: scout caught up for {} after a query landed ({} graded)",
                            self.job_id,
                            outcome.graded
                        );
                        let _ = project(&self.runtime, &self.job_id);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        log::warn!("cascade: scout catch-up failed for {}: {}", self.job_id, e)
                    }
                }
            }

            // System-capacity budget (libsyscap design §7.3): sampled fresh
            // every wave, so slices shrink under load AND restore when the
            // machine goes idle; the config k1/k2 stay the ceiling. `hold`
            // (nearly out of RAM) pauses the waves entirely: the run stays
            // alive, reports `holding`, and re-checks until the verdict
            // lifts — no wave, no writes, no busy loop.
            let sys = syscap::budget_now(&self.syscap, &self.engine);
            if !sys.verdict.heavy_ok() {
                log::warn!(
                    "cascade: system budget holds heavy work for {} (free memory low) — pausing waves",
                    self.job_id
                );
                self.events.emit("rank-progress", &RankProgress {
                    job_id: self.job_id.clone(),
                    stage: "embed".into(),
                    wave,
                    embedded_done: total_embedded as u32,
                    embedded_total: candidate_count as u32,
                    judged: total_judged,
                    candidates: candidate_count as u32,
                    running: true,
                    reason: syscap::holding_code(&sys),
                });
                thread::sleep(HOLD_POLL);
                continue;
            }
            let budget = if is_small_job {
                small_job_budget
            } else {
                syscap::wave_budget(self.cascade.k1(), pool_size, &sys)
            };

            wave += 1;
            let progress = &|done: i64, total: i64, _failed: i64, active: bool| {
                // The one throttled spot: per-candidate embed callbacks. The
                // wave's final (active=false) event and the wave-end emit
                // below always pass, so counts stay exact at every edge.
                if !active || self.throttle.allow() {
                    self.events.emit("rank-progress", &RankProgress {
                        job_id: self.job_id.clone(),
                        stage: "embed".into(),
                        wave,
                        embedded_done: (total_embedded + done) as u32,
                        embedded_total: (total_embedded + total) as u32,
                        judged: total_judged,
                        candidates: candidate_count as u32,
                        running: active,
                        reason: None,
                    });
                }
            };
            match run_wave(&self.runtime, &self.job_id, wave, budget, progress) {
                Ok(outcome) => {
                    total_embedded += outcome.embedded;
                    if outcome.judge_ran {
                        total_judged += outcome.judge_pairs_fresh;
                    }
                    self.events.emit("rank-progress", &RankProgress {
                        job_id: self.job_id.clone(),
                        stage: "embed".into(),
                        wave,
                        embedded_done: total_embedded as u32,
                        embedded_total: candidate_count as u32,
                        judged: total_judged,
                        candidates: candidate_count as u32,
                        running: !outcome.converged,
                        reason: None,
                    });
                    if outcome.converged {
                        break;
                    }
                }
                Err(e) => {
                    log::warn!("cascade: wave {} failed for {}: {}", wave, self.job_id, e);
                    self.fail();
                    return;
                }
            }
        }

        // Reaching here means a wave reported `converged`, so the run is
        // complete: every candidate is embedded and the last wave was a full
        // pass. The only other exits were the cancel (handled above) and a
        // wave failure, both of which return early.
        log::info!(
            "cascade: converged for {} after {wave} wave(s) (embedded {total_embedded}, small_job: {is_small_job})",
            self.job_id
        );
        self.finish(true);
    }

    /// Whether a cancel was requested for this job (`rank_state.cancelled`).
    /// Checked between waves, so a cancel lands at a wave boundary rather than
    /// interrupting an embed mid-batch.
    fn cancelled(&self) -> bool {
        match libtriqui_core::db::open_connection_at(&self.db_path) {
            Ok(conn) => libtriqui_core::db::rank_state_for_job(&conn, &self.job_id)
                .map(|rs| rs.cancelled)
                .unwrap_or(false),
            Err(e) => {
                log::warn!("cascade: cannot read rank state for {}: {}", self.job_id, e);
                false
            }
        }
    }

    fn emit_progress(&self, p: RankProgress) {
        self.events.emit("rank-progress", &p);
    }

    /// Terminal: converged (or stopped short) with a readable report.
    fn finish(&self, converged: bool) {
        // A superseded runner's terminal write would clobber the run that
        // replaced it — skip everything, including the progress event.
        if self.superseded() {
            log::info!(
                "cascade: runner for {} superseded — terminal write skipped",
                self.job_id
            );
            return;
        }
        let conn = match libtriqui_core::db::open_connection_at(&self.db_path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("cascade: cannot read back for {}: {}", self.job_id, e);
                return;
            }
        };
        // Read the report the last wave persisted rather than re-running the
        // pipeline: a second pass would find every judge pair memoized and
        // report `fresh = 0`, which is a lie about what happened.
        let reports_json = libtriqui_core::rescore::read_triqui(&conn, &self.job_id)
            .map(|(_, report)| serde_json::to_string(&vec![report]).unwrap_or_default())
            .unwrap_or_else(|e| {
                log::warn!("cascade: convergence read failed for {}: {}", self.job_id, e);
                "[]".to_string()
            });
        // A run that stops short of convergence is *cancelled*, not still
        // running: `get_rank_state` derives `running` from
        // `!complete && !cancelled`, so writing `complete=false,
        // cancelled=false` here would leave the job reporting a run in flight
        // forever — and `spawn_cascade` refuses to start a new one while that
        // holds, wedging the job until the app restarts.
        let _ = libtriqui_core::db::set_rank_state(
            &conn,
            &self.job_id,
            "sort",
            &reports_json,
            converged,
            !converged,
        );
        self.events.emit("rank-progress", &RankProgress {
            job_id: self.job_id.clone(),
            stage: if converged { "done".into() } else { "stopped".into() },
            wave: 0,
            embedded_done: 0,
            embedded_total: 0,
            judged: 0,
            candidates: 0,
            running: false,
            reason: None,
        });
    }

    fn fail(&self) {
        // Same rule as `finish`: an error state written by a superseded
        // runner would mask the run that replaced it.
        if self.superseded() {
            log::info!(
                "cascade: runner for {} superseded — error write skipped",
                self.job_id
            );
            return;
        }
        let conn = match libtriqui_core::db::open_connection_at(&self.db_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let _ = libtriqui_core::db::set_rank_state(&conn, &self.job_id, "error", "[]", true, true);
        self.events.emit("rank-progress", &RankProgress {
            job_id: self.job_id.clone(),
            stage: "error".into(),
            wave: 0,
            embedded_done: 0,
            embedded_total: 0,
            judged: 0,
            candidates: 0,
            running: false,
            reason: None,
        });
    }

    fn count_candidates(&self, conn: &rusqlite::Connection) -> Result<i64, rusqlite::Error> {
        conn.query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
            rusqlite::params![self.job_id],
            |r| r.get(0),
        )
    }

    /// Whether the job has job-ad text to score against (a blank or NULL ad is
    /// no ad — the same rule `job_ui_state`'s `no-jobad` code applies).
    fn job_has_ad(&self, conn: &rusqlite::Connection) -> bool {
        conn.query_row(
            "SELECT COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0)
             FROM jobs WHERE id = ?1",
            rusqlite::params![self.job_id],
            |r| r.get::<_, i64>(0),
        )
        .map(|v| v != 0)
        .unwrap_or(false)
    }
}

/// Spawn the convergence cascade for `job_id` on a background thread (stage 0
/// → waves → final sort, emitting `rank-progress` events). Returns `Ok(false)`
/// when a cascade is already in flight for that job, so callers don't stack a
/// second one.
///
/// Shared by the `start_rank` command and the import auto-start (design
/// §9: import is extract-only, "the cascade run starts automatically" once
/// extraction is done).
pub fn spawn_cascade(state: &Arc<ServeState>, job_id: &str) -> Result<bool, String> {
    // Server-side floor for every entry — command, import auto-start,
    // ad-landing auto-start (the gui's Re-rank button is gated above this by
    // `job_ui_state`'s reason codes). A job with no extracted CVs has nothing
    // to embed and nothing to rank.
    //
    // A job with **no job ad** is NOT refused: the embed stages are
    // ad-independent, so its cascade embeds the corpus (scout stage 0.5 +
    // heavy waves) and converges with scoring deferred — the scout stores no
    // grades without a query and the waves' sorts run an empty rubric. Grades
    // land when the ad does: the same run catches the scout up and re-sorts
    // from the stored chunks at the next wave boundary, or `update_job`
    // spawns a scoring-only run once this one converged.
    {
        let conn = state.db.0.lock().map_err(|e| format!("db lock: {e}"))?;
        let cvs: i64 = conn
            .query_row(
                "SELECT (SELECT COUNT(*) FROM candidates
                          WHERE job_id = ?1 AND extraction_status = 'Extracted')
                 FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| r.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => format!("job {job_id} not found"),
                e => format!("db: {e}"),
            })?;
        if cvs < 1 {
            return Err("cannot rank: no extracted CVs for this job".into());
        }
    }

    // Already running for this job? A live runner whose run was cancelled
    // (DB row says so) is on its way out at the next wave boundary — a spawn
    // now supersedes it via the generation bump instead of refusing or
    // waiting for it.
    if state.cascade_gens.has(job_id) {
        let cancelled = {
            let conn = state.db.0.lock().map_err(|e| format!("db lock: {e}"))?;
            rank_state_for_job(&conn, job_id)
                .map(|rs| rs.cancelled)
                .unwrap_or(false)
        };
        if !cancelled {
            return Ok(false);
        }
    }

    // A running rank_state row with no live runner is a leftover this process
    // did not recover (kept as belt-and-braces behind the startup recovery).
    {
        let conn = state.db.0.lock().map_err(|e| format!("db lock: {e}"))?;
        if let Ok(rs) = rank_state_for_job(&conn, job_id) {
            if !rs.complete && !rs.cancelled {
                return Ok(false);
            }
        }
    }

    // Claim the generation before initializing the row: from here on the new
    // runner owns this job's rank_state writes, and any runner it displaced
    // goes stale.
    let gen = state.cascade_gens.bump(job_id);

    // Initialize rank_state as running
    {
        let mut conn = state.db.0.lock().map_err(|e| format!("db lock: {e}"))?;
        let _ = set_rank_state(&mut conn, job_id, "lex", "[]", false, false);
    }

    // Resolve the syscap sections HERE, on the serve worker thread where the
    // config-path override is set — the runner thread never loads config
    // (thread-locals don't inherit). Missing config = default sections:
    // syscap enabled, engine caps unset (libsyscap design: absent yaml =
    // defaults).
    let (syscap, engine) = {
        let cfg = libtriqui_core::app_config::load_ikapp();
        (cfg.syscap, cfg.engine)
    };

    let runner = CascadeRunner {
        job_id: job_id.to_string(),
        runtime: Arc::clone(&state.runtime),
        db_path: state.db_path.clone(),
        events: state.events.clone(),
        cascade: state.config.clone(),
        syscap,
        engine,
        gens: Arc::clone(&state.cascade_gens),
        gen,
        throttle: crate::serve::ProgressThrottle::default(),
    };

    thread::spawn(move || {
        runner.run();
    });
    Ok(true)
}

/// Start a progressive-ranking cascade for the job. Returns immediately with
/// the initial RankState (running=true). The actual work runs on a background
/// thread and emits `rank-progress` events.
///
/// The explicit rank command requires a rubric query: this is the gui's
/// Re-rank button, and ranking without a job ad produces all-zero totals
/// (`job_ui_state` reports the same `no-jobad` blocker for the button). The
/// auto-starts bypass this check deliberately — [`spawn_cascade`] runs an
/// ad-less job as an embed-only cascade (scout + waves embed the corpus,
/// scoring defers to when the ad lands), which is the import auto-start's
/// contract (design §9).
pub fn start_rank(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    {
        let conn = state.db.0.lock().map_err(|e| string_err(format!("db lock: {e}")))?;
        let (cvs, ad_set): (i64, bool) = conn
            .query_row(
                "SELECT (SELECT COUNT(*) FROM candidates
                          WHERE job_id = ?1 AND extraction_status = 'Extracted'),
                        COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0)
                 FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| Ok((r.get(0)?, r.get::<_, i64>(1)? != 0)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => string_err(format!("job {job_id} not found")),
                e => string_err(format!("db: {e}")),
            })?;
        if cvs < 1 {
            return Err(string_err("cannot rank: no extracted CVs for this job"));
        }
        if !ad_set {
            return Err(string_err("cannot rank: job has no job ad text"));
        }
    }
    match spawn_cascade(state, job_id) {
        Ok(true) => {}
        Ok(false) => return Err(string_err("cascade already running for this job")),
        Err(e) => return Err(string_err(e)),
    }

    // Return initial state — start_rank refused an ad-less job above, so the
    // run it starts is always a real ranking.
    let initial = RankState {
        running: true,
        ranking: true,
        stage: Some("lex".into()),
        wave: None,
        embedded_total: 0,
        embedded_done: 0,
        judged: 0,
        candidates: 0,
        converged: false,
        reason: None,
    };
    ok_json(initial)
}

/// Request cancellation of the job's running cascade.
pub fn cancel_rank(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let mut conn = state.db.0.lock().map_err(|e| string_err(format!("db lock: {}", e)))?;
    let _ = set_rank_state(&mut conn, job_id, "sort", "[]", true, true);
    ok_json(Value::Null)
}

/// The `reason` code for a job's rank state: a cancelled run wins (it is
/// terminal), then a live run paused by the system budget reports
/// `holding` (libsyscap design §4 — the ONLY budget state the gui must
/// surface); quiet adaptation (Polite/Backoff) stays `None`.
fn state_reason(cancelled: bool, budget: &libsyscap::Budget) -> Option<String> {
    if cancelled {
        Some("cancelled".into())
    } else {
        libtriqui_core::syscap::holding_code(budget)
    }
}

/// The system budget as a fresh sample (pull-based — every poll re-derives
/// it, so the answer can never go stale). Missing config = default sections.
fn sample_budget_for_state() -> (libsyscap::Budget, libsyscap::LoadSnapshot) {
    let (syscap, engine) = {
        let cfg = libtriqui_core::app_config::load_ikapp();
        (cfg.syscap, cfg.engine)
    };
    libtriqui_core::syscap::sample_budget(&syscap, &engine)
}

/// Get the current cascade rank state for a job.
///
/// Progress counters are **derived from the data**, not carried in a column that
/// the background thread has to remember to update: `candidates` is the
/// extracted candidate count, `embedded_done` counts candidates that actually
/// hold chunk vectors for the full embedder, and `judged` is the last wave's
/// fresh-verdict count from the sort report. Deriving them means a poll can
/// never disagree with the sheet, and it costs one indexed count each.
///
/// `running` = a cascade is in flight (including the import auto-start's
/// ad-less embed-prefill run); `ranking` = in flight **and** the job has an
/// ad — the flag the gui's rank buttons spin on.
pub fn get_rank_state(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(|e| string_err(format!("db lock: {}", e)))?;
    // A job the cascade has not touched yet has no `rank_state` row. That is a
    // normal state a poll lands in routinely — between an import-done event
    // and the auto-started run's first write, or on an ad-less job whose
    // embed-only run has not been spawned — so report a coherent not-started
    // snapshot instead of erroring.
    let rs = match rank_state_for_job(&conn, job_id) {
        Ok(rs) => Some(rs),
        Err(_) => None,
    };
    let model = state.runtime.model();

    let count = |sql: &str, params: Vec<Box<dyn rusqlite::ToSql>>| -> i64 {
        conn.query_row(
            sql,
            rusqlite::params_from_iter(params.iter().map(|b| b.as_ref())),
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
    };
    let candidates = count(
        "SELECT COUNT(*) FROM candidates
         WHERE job_id = ?1 AND extraction_status = 'Extracted' AND extracted_text IS NOT NULL",
        vec![Box::new(job_id.to_string())],
    );
    let embedded_done = count(
        "SELECT COUNT(*) FROM candidates c
         WHERE c.job_id = ?1 AND c.extraction_status = 'Extracted' AND c.extracted_text IS NOT NULL
           AND EXISTS (SELECT 1 FROM candidate_chunks cc
                       WHERE cc.candidate_id = c.id AND cc.vector_model = ?2)",
        vec![
            Box::new(job_id.to_string()) as Box<dyn rusqlite::ToSql>,
            Box::new(model.clone()),
        ],
    );

    // The latest report is the most recent wave, and it is the one that says
    // whether the run converged.
    let (running, converged, cancelled, stage, reports) = match &rs {
        Some(rs) => {
            let reports: Vec<SortReport> =
                serde_json::from_str(&rs.sort_reports_json).unwrap_or_default();
            (
                !rs.complete && !rs.cancelled,
                rs.complete && !rs.cancelled,
                rs.cancelled,
                Some(rs.sort.clone()),
                reports,
            )
        }
        None => (false, false, false, None, Vec::new()),
    };
    // The import auto-start deliberately runs ad-less jobs as an embed-prefill
    // cascade (spawn_cascade's doc), which is status-bar activity — not a
    // ranking in progress. `ranking` is the button-facing flag: in flight AND
    // scoreable. It re-evaluates every poll, so an ad landing mid-run flips
    // the buttons on without any new state write.
    let ad_set: bool = conn
        .query_row(
            "SELECT COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0)
             FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get::<_, i64>(0),
        )
        .map(|v| v != 0)
        .unwrap_or(false);
    let ranking = running && ad_set;
    let last = reports.last();
    // Sampled fresh per poll: a run that is being held reports `holding`
    // until the machine frees up, then silently goes back to `None`.
    let budget = sample_budget_for_state().0;
    // A job with no row has nothing in flight, so it never reports a budget
    // reason — only a live or finished run can be held.
    let reason = match &rs {
        Some(_) => state_reason(cancelled, &budget),
        None => None,
    };
    let out = RankState {
        running,
        ranking,
        stage,
        wave: last.filter(|r| r.waves > 0).map(|r| r.waves),
        embedded_total: candidates as u32,
        embedded_done: embedded_done as u32,
        judged: last.map(|r| r.judge_pairs_fresh).unwrap_or(0),
        candidates: candidates as u32,
        converged,
        reason,
    };
    ok_json(out)
}

/// The current system-capacity budget (libsyscap), for the gui to surface.
/// The only state the gui must render is `holding` (a visible notice);
/// `polite`/`backoff`/the load numbers are provenance and phase-2 badge
/// groundwork.
pub fn system_budget(_state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    let (b, load) = sample_budget_for_state();
    ok_json(SystemBudget {
        verdict: serde_json::to_value(&b.verdict)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".into()),
        wave_scale: b.wave_scale,
        thread_cap: b.thread_cap as u32,
        embed_batch: b.embed_batch.map(|n| n as u32),
        holding: !b.verdict.heavy_ok(),
        load: SystemLoad {
            cpu_pct: load.cpu_pct as f64,
            loadavg_1: load.loadavg[0],
            mem_available_bytes: load.mem_available_bytes,
            mem_total_bytes: load.mem_total_bytes,
            on_battery: {
                let inv = libsyscap::Inventory::global();
                inv.power.on_battery
            },
        },
    })
}

/// Mark a candidate as viewed (stamps viewed_at).
pub fn mark_candidate_viewed(
    state: &Arc<ServeState>,
    candidate_id: &str,
) -> Result<Value, ProtocolError> {
    let conn = state.db.0.lock().map_err(|e| string_err(format!("db lock: {}", e)))?;
    conn.execute(
        "UPDATE candidates SET viewed_at = datetime('now') WHERE id = ?1",
        rusqlite::params![candidate_id],
    )
    .map_err(|e| domain(&libtriqui_core::types::DomainError::DbError { message: e.to_string() }))?;
    ok_json(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsyscap::Verdict;

    fn budget_of(verdict: Verdict) -> libsyscap::Budget {
        libsyscap::Budget {
            verdict,
            thread_cap: 2,
            embed_batch: None,
            wave_scale: verdict.wave_scale(),
            mem_cap_bytes: 0,
            capacity: 2,
        }
    }

    /// The reported-behavior contract (design §8): a cancelled run reports
    /// `cancelled`; a live run reports `holding` exactly when the budget
    /// says hold — never for the quiet verdicts; a clean state reports no
    /// reason at all.
    #[test]
    fn rank_reason_maps_cancel_and_hold_only() {
        assert_eq!(state_reason(true, &budget_of(Verdict::Hold)).as_deref(), Some("cancelled"),
            "cancel wins even mid-hold");
        assert_eq!(state_reason(false, &budget_of(Verdict::Hold)).as_deref(), Some("holding"));
        assert_eq!(state_reason(false, &budget_of(Verdict::Full)), None);
        assert_eq!(state_reason(false, &budget_of(Verdict::Polite)), None,
            "polite adapts silently");
        assert_eq!(state_reason(false, &budget_of(Verdict::Backoff)), None,
            "backoff adapts silently");
    }

    /// The generation guard's whole contract: a spawn supersedes the current
    /// runner, the superseded runner's exit must not deregister its
    /// replacement, and only the current runner's exit frees the job for a
    /// fresh spawn (the cancel-then-respawn race, guard side).
    #[test]
    fn cascade_generations_supersede_and_retire() {
        let gens = CascadeGens::new();
        assert!(!gens.has("j"), "no runner before the first spawn");

        let a = gens.bump("j");
        assert!(gens.has("j"));
        assert!(gens.is_current("j", a));

        // Runner B spawns over the cancelled A.
        let b = gens.bump("j");
        assert_ne!(a, b);
        assert!(!gens.is_current("j", a), "A is superseded by B");
        assert!(gens.is_current("j", b));

        // A exits: it must not deregister B.
        gens.retire("j", a);
        assert!(gens.has("j"), "B still owns the job");
        assert!(gens.is_current("j", b));

        // B exits: the job is free again.
        gens.retire("j", b);
        assert!(!gens.has("j"));

        // Generations keep counting across retire/re-register cycles (the
        // high-water mark is never reset), so a stale runner from an earlier
        // cycle can never look current again.
        let c = gens.bump("j");
        assert_ne!(c, a);
        assert_ne!(c, b);
        assert!(gens.is_current("j", c));
        assert!(!gens.is_current("j", a));
        assert!(!gens.is_current("j", b));
    }
}