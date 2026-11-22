use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use libtriqui_core::embed::Embedder;
use std::sync::{Arc, Mutex};

use libcv_state::{Event, State};

pub trait Worker {
    type In;
    type Out: Into<Event>;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError>;
}

#[derive(Debug)]
pub enum WorkerError {
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkItem {
    TextExtract { cv_id: String },
    Quick { cv_id: String },
    Deep { cv_id: String },
    Match { cv_id: String, job_id: String },
    Comparator { pair: (String, String) },
    ScanDirectory { job_id: String },
}

pub struct EffectRunner {
    workers: usize,
}

impl EffectRunner {
    pub fn new() -> Self {
        Self {
            workers: env::var("CV_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1),
        }
    }

    pub fn with_workers(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
        }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Scheduling decisions derived from post-reducer state and the triggering event.
    pub fn schedule(&self, state: &State, event: &Event) -> Vec<WorkItem> {
        self.effects(state, event)
    }

    pub fn effects(&self, state: &State, event: &Event) -> Vec<WorkItem> {
        let mut out = Vec::new();
        match event {
            Event::CvAdded { cv_id, .. } => {
                // Auto-chain: a new CV starts with text extraction.
                out.push(WorkItem::TextExtract { cv_id: cv_id.clone() });
            }
            Event::TextExtracted { cv_id, status, .. } => {
                // Auto-chain: extracted text goes straight to the scout (quick) embed.
                if *status == libcv_state::ExtractionStatus::Extracted {
                    out.push(WorkItem::Quick { cv_id: cv_id.clone() });
                }
            }
            Event::ScoutCompleted { cv_id, .. } => {
                // Auto-chain: scout vector done -> run the deep (heavy) embed.
                out.push(WorkItem::Deep { cv_id: cv_id.clone() });
            }
            Event::HeavyCompleted { cv_id, .. } => {
                // Match work is enqueued when a job ad version is known.
                for (job_id, job) in &state.jobs {
                    if let Some(cv) = state.cvs.get(cv_id) {
                        if cv.job_id == *job_id {
                            out.push(WorkItem::Match {
                                cv_id: cv_id.clone(),
                                job_id: job_id.clone(),
                            });
                        }
                    }
                    let _ = job;
                }
            }
            Event::JobAdUpdated { job_id, .. } => {
                for cv in state.cvs.values() {
                    if cv.job_id == *job_id && cv.heavy_vector.is_some() {
                        out.push(WorkItem::Match {
                            cv_id: cv.id.clone(),
                            job_id: job_id.clone(),
                        });
                    }
                }
            }
            Event::HeavyMatched { .. } => {
                // Ask comparator orderer for next pair (handled by caller with state).
            }
            Event::ComparisonCompleted { .. } => {}
            _ => {}
        }
        // Comparator scheduling: after HeavyMatched accumulation, consult next_pair.
        // Delegated to caller; here we leave comparator work to explicit enqueue.
        out
    }

    /// Decide comparator work given current state (call after HeavyMatched/ComparisonCompleted).
    pub fn next_comparator_work(&self, state: &State) -> Option<WorkItem> {
        use libcv_comparator_order::{ComparatorOrderState, next_pair};
        let provisional = provisional_sim_rank(state);
        let comparisons: HashSet<(String, String)> = state
            .comparisons
            .iter()
            .map(|(a, b, _, _, _)| {
                if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                }
            })
            .collect();
        let job_version = state.jobs.values().next().map(|j| j.version).unwrap_or(0);
        let order_state = ComparatorOrderState {
            glicko: state
                .glicko
                .iter()
                .map(|((cv, ver), g)| {
                    (
                        (cv.clone(), *ver),
                        libcv_comparator_order::GlickoState { rating: g.rating, rd: g.rd },
                    )
                })
                .collect(),
            comparisons,
            provisional_sim_rank: provisional,
            job_version,
            entitled: state.license.is_some(),
        };
        next_pair(&order_state).map(|p| WorkItem::Comparator { pair: p })
    }
}

/// Candidates eligible for comparator scheduling, highest similarity first.
///
/// Deleted candidates are excluded: a retired CV must never be scheduled for a
/// head-to-head comparison, or it would re-enter the tournament it just left.
pub fn provisional_sim_rank(state: &State) -> Vec<(String, f64)> {
    let mut provisional: Vec<(String, f64)> = state
        .cvs
        .values()
        .filter(|c| !c.deleted)
        .map(|c| (c.id.clone(), c.heavy_similarity.unwrap_or(0.0)))
        .collect();
    provisional.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    provisional
}

pub struct TextWorker;
impl Worker for TextWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        match libcvparser::extract_text(std::path::Path::new(&item)) {
            Ok(text) => Ok(Event::TextExtracted {
                cv_id: item,
                text,
                status: libcv_state::ExtractionStatus::Extracted,
            }),
            Err(_) => Ok(Event::TextExtracted {
                cv_id: item,
                text: String::new(),
                status: libcv_state::ExtractionStatus::Failed,
            }),
        }
    }
}

pub struct ScoutWorker;
pub struct HeavyWorker;
pub struct MatchWorker;
pub struct ComparatorWorker;

impl TextWorker {
    pub fn new() -> Self { Self }
}
impl Default for TextWorker {
    fn default() -> Self { Self }
}
impl ScoutWorker {
    pub fn new() -> Self { Self }
}
impl Default for ScoutWorker {
    fn default() -> Self { Self }
}
impl HeavyWorker {
    pub fn new() -> Self { Self }
}
impl Default for HeavyWorker {
    fn default() -> Self { Self }
}
impl MatchWorker {
    pub fn new() -> Self { Self }
}
impl Default for MatchWorker {
    fn default() -> Self { Self }
}
impl ComparatorWorker {
    pub fn new() -> Self { Self }
}
impl Default for ComparatorWorker {
    fn default() -> Self { Self }
}

pub struct ScanWorker;

impl ScanWorker {
    pub fn new() -> Self { Self }
}

impl Default for ScanWorker {
    fn default() -> Self { Self }
}

impl TextWorker {
    pub fn extract(&self, cv_id: String, path: &str) -> Event {
        match libcvparser::extract_text(std::path::Path::new(path)) {
            Ok(text) => Event::TextExtracted {
                cv_id,
                text,
                status: libcv_state::ExtractionStatus::Extracted,
            },
            Err(_) => Event::TextExtracted {
                cv_id,
                text: String::new(),
                status: libcv_state::ExtractionStatus::Failed,
            },
        }
    }
}

impl Worker for ScoutWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        self.process_with_context(item, "", &ScoutContext::default(), 0)
    }
}

impl Worker for HeavyWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        self.process_with_context(item, "", &HeavyContext::default(), 0)
    }
}

impl Worker for MatchWorker {
    type In = (String, String);
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        self.process_with_vectors(item.0, 0, None, None)
    }
}

impl Worker for ComparatorWorker {
    type In = (String, String);
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        self.process_with_model(item, 0, 0, None, None)
    }
}

pub struct ScoutContext {
    pub embedder: Option<std::sync::Arc<dyn libtriqui_core::embed::Embedder>>,
}

impl Default for ScoutContext {
    fn default() -> Self {
        Self { embedder: None }
    }
}

pub struct HeavyContext {
    pub embedder: Option<std::sync::Arc<dyn libtriqui_core::embed::Embedder>>,
}

impl Default for HeavyContext {
    fn default() -> Self {
        Self { embedder: None }
    }
}

pub struct MatchContext {
    pub cv_vector: Option<Vec<f32>>,
    pub target_vector: Option<Vec<f32>>,
}

impl Default for MatchContext {
    fn default() -> Self {
        Self {
            cv_vector: None,
            target_vector: None,
        }
    }
}

pub struct ComparatorContext {
    pub model: Option<libcomparator::ComparatorModel>,
}

impl Default for ComparatorContext {
    fn default() -> Self {
        Self { model: None }
    }
}

impl ScoutWorker {
    pub fn embed(&self, ctx: &ScoutContext, text: &str) -> Vec<f32> {
        if let Some(embedder) = &ctx.embedder {
            if let Ok(vec) = embedder.embed(&[text]) {
                return vec;
            }
        }
        Vec::new()
    }

    pub fn process_with_context(
        &mut self,
        cv_id: String,
        text: &str,
        ctx: &ScoutContext,
        model_bundle_version: u32,
    ) -> Result<Event, WorkerError> {
        let vector = self.embed(ctx, text);
        Ok(Event::ScoutCompleted {
            cv_id,
            vector,
            model_bundle_version,
        })
    }
}

impl HeavyWorker {
    pub fn embed(&self, ctx: &HeavyContext, text: &str) -> Vec<f32> {
        if let Some(embedder) = &ctx.embedder {
            if let Ok(vec) = embedder.embed(&[text]) {
                return vec;
            }
        }
        Vec::new()
    }

    pub fn process_with_context(
        &mut self,
        cv_id: String,
        text: &str,
        ctx: &HeavyContext,
        model_bundle_version: u32,
    ) -> Result<Event, WorkerError> {
        let vector = self.embed(ctx, text);
        Ok(Event::HeavyCompleted {
            cv_id,
            vector,
            model_bundle_version,
        })
    }
}

impl MatchWorker {
    pub fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        libcv_match::cosine(a, b)
    }

    pub fn cosine(&self, a: &[f64], b: &[f64]) -> f64 {
        libcv_match::cosine(a, b)
    }

    pub fn process_with_vectors(
        &mut self,
        cv_id: String,
        job_version: u32,
        cv_vector: Option<Vec<f64>>,
        target_vector: Option<Vec<f64>>,
    ) -> Result<Event, WorkerError> {
        let similarity = match (cv_vector, target_vector) {
            (Some(a), Some(b)) => libcv_match::cosine(&a, &b),
            _ => 0.0,
        };
        Ok(Event::HeavyMatched {
            cv_id,
            job_version,
            similarity,
        })
    }
}

impl ComparatorWorker {
    pub fn pair_prob(&self, model: &libcomparator::ComparatorModel, x: Vec<f32>) -> f64 {
        model.pair_prob(x)
    }

    pub fn process_with_model(
        &mut self,
        pair: (String, String),
        job_version: u32,
        model_bundle_version: u32,
        model: Option<&libcomparator::ComparatorModel>,
        features: Option<Vec<f32>>,
    ) -> Result<Event, WorkerError> {
        let result = match (model, features) {
            (Some(m), Some(x)) => m.pair_prob(x),
            _ => 0.5,
        };
        Ok(Event::ComparisonCompleted {
            pair,
            job_version,
            model_bundle_version,
            result,
        })
    }
}

impl ScanWorker {
    /// Reconcile a job's watched CV directory with its candidate set and
    /// import anything new.
    ///
    /// One scan converges additions, edits, and deletions:
    ///   * a candidate whose source file disappeared is marked `deleted = 1`,
    ///     with its chunks and scores dropped, and a `CvDeleted` event is
    ///     returned so the ranking side re-runs without it;
    ///   * a candidate whose file fingerprint changed is reset to `Pending`
    ///     with cleared derived data and re-dispatched as `CvAdded`;
    ///   * a candidate with a NULL fingerprint (pre-upgrade row) is backfilled
    ///     without being reset;
    ///   * a path with no active candidate row is imported.
    ///
    /// Concurrency design (see `openspec/changes/database-concurrency-fix`):
    ///   * The directory path and the candidate rows are read through pooled
    ///     read-only connections, so the scan never blocks the writer or other
    ///     readers.
    ///   * File enumeration and hashing are pure filesystem I/O — no DB lock is
    ///     held.
    ///   * Each candidate mutation (delete/reset/backfill/insert) acquires and
    ///     releases the writer lock independently, matching the existing
    ///     per-file lock granularity.
    pub fn scan(
        &self,
        job_id: String,
        db: &Arc<libtriqui_core::db::DbConnection>,
    ) -> Vec<Event> {
        use libtriqui_core::import::{collect_supported_files, hash_file};
        use rusqlite::params;
        use std::collections::HashMap;

        // Phase 1: read dir via a pooled read-only connection (never blocks
        // the writer, and does not serialize behind other readers).
        let dir: String = db
            .read_conn()
            .query_row(
                "SELECT import_directory_path FROM jobs WHERE id = ?1",
                params![job_id],
                |r| r.get(0),
            )
            .ok()
            .flatten()
            .unwrap_or_default();
        if dir.is_empty() {
            return Vec::new();
        }

        // Phase 2: collect files and fingerprint them (no DB lock at all).
        // An empty directory is NOT an early return: every existing candidate
        // must be retired when the directory is emptied.
        let files = match collect_supported_files(&dir) {
            Ok(files) => files,
            Err(e) => {
                log::warn!("scan {}: scan of {} failed: {}", job_id, dir, e);
                return Vec::new();
            }
        };
        // path -> fingerprint (`None` when the file vanished or is unreadable
        // between enumeration and hashing). Presence in this map is what keeps
        // an unreadable-but-present file from being treated as deleted.
        let mut present: HashMap<String, Option<String>> = HashMap::with_capacity(files.len());
        for file_path in &files {
            let path_str = file_path.to_string_lossy().to_string();
            present.insert(path_str, hash_file(file_path));
        }

        log::info!("scan {}: {} file(s) in {}", job_id, files.len(), dir);

        // Phase 3: read the job's active candidate rows up front, so the set
        // comparison happens from one consistent snapshot taken before writes.
        let rows: Vec<(String, String, Option<String>)> = {
            let conn = db.read_conn();
            let mut stmt = match conn.prepare(
                "SELECT id, source_file_path, content_hash FROM candidates
                 WHERE job_id = ?1 AND deleted = 0",
            ) {
                Ok(stmt) => stmt,
                Err(e) => {
                    log::warn!("scan {}: cannot read candidates: {}", job_id, e);
                    return Vec::new();
                }
            };
            let mapped = stmt.query_map(params![job_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                ))
            });
            let mut out = Vec::new();
            match mapped {
                Ok(iter) => {
                    for row in iter {
                        match row {
                            Ok(r) => out.push(r),
                            Err(e) => log::warn!("scan {}: bad candidate row: {}", job_id, e),
                        }
                    }
                }
                Err(e) => log::warn!("scan {}: cannot query candidates: {}", job_id, e),
            }
            out
        };

        let mut events = Vec::new();
        // A deletion or modification is a ranking-input change: bump once below.
        let mut ranking_inputs_changed = false;

        // Phase 4a: retire candidates whose source file disappeared.
        for (cv_id, path, _hash) in &rows {
            if present.contains_key(path) {
                continue;
            }
            let conn = db.writer();
            let _ = conn.execute(
                "UPDATE candidates SET deleted = 1, rank = NULL, ranked_at = NULL,
                        points = NULL, points_max = NULL
                 WHERE id = ?1",
                params![cv_id],
            );
            let _ = conn.execute(
                "DELETE FROM candidate_chunks WHERE candidate_id = ?1",
                params![cv_id],
            );
            let _ = conn.execute(
                "DELETE FROM candidate_scores WHERE candidate_id = ?1",
                params![cv_id],
            );
            drop(conn);
            events.push(Event::CvDeleted {
                cv_id: cv_id.clone(),
                job_id: job_id.clone(),
            });
            ranking_inputs_changed = true;
        }

        // Phase 4b: backfill unknown fingerprints, and reset changed ones.
        for (cv_id, path, stored_hash) in &rows {
            let new_hash = match present.get(path).and_then(|h| h.as_ref()) {
                Some(h) => h,
                // Deleted above, or unreadable: nothing to compare against.
                None => continue,
            };
            match stored_hash {
                None => {
                    // Pre-upgrade row: record the fingerprint without resetting,
                    // so upgrading never re-extracts every existing CV.
                    let conn = db.writer();
                    let _ = conn.execute(
                        "UPDATE candidates SET content_hash = ?1 WHERE id = ?2",
                        params![new_hash, cv_id],
                    );
                    drop(conn);
                }
                Some(old) if old != new_hash => {
                    // Edited on disk: reset to Pending, clear derived data, keep
                    // the row (and its likes/history) but store the new hash.
                    let conn = db.writer();
                    let _ = conn.execute(
                        "UPDATE candidates SET extraction_status = 'Pending',
                                extracted_text = NULL, vector = NULL, vector_model = NULL,
                                rank = NULL, ranked_at = NULL, points = NULL, points_max = NULL,
                                match_score = NULL, explanation_summary = NULL,
                                content_hash = ?1
                         WHERE id = ?2",
                        params![new_hash, cv_id],
                    );
                    let _ = conn.execute(
                        "DELETE FROM candidate_chunks WHERE candidate_id = ?1",
                        params![cv_id],
                    );
                    let _ = conn.execute(
                        "DELETE FROM candidate_scores WHERE candidate_id = ?1",
                        params![cv_id],
                    );
                    drop(conn);
                    let (file_name, file_type) = file_identity(path);
                    events.push(Event::CvAdded {
                        cv_id: cv_id.clone(),
                        job_id: job_id.clone(),
                        file_path: path.clone(),
                        file_name,
                        file_type,
                        imported_at: chrono::Utc::now().to_rfc3339(),
                    });
                    ranking_inputs_changed = true;
                }
                Some(_) => {}
            }
        }

        // Phase 4c: import every supported file with no active candidate row.
        // A path whose only row is `deleted` is imported as a fresh candidate.
        for (path, hash) in &present {
            let already_active = rows.iter().any(|(_, p, _)| p == path);
            if already_active {
                continue;
            }
            let file_name = file_name_for(path);
            let file_type = file_type_for(path);
            let cv_id = format!("cv_{}", uuid::Uuid::new_v4());
            let conn = db.writer();
            let inserted = conn
                .execute(
                    "INSERT INTO candidates
                        (id, job_id, source_file_path, file_name, file_type,
                         extraction_status, content_hash, imported_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, 'Pending', ?6, datetime('now'))",
                    params![cv_id, job_id, path, file_name, file_type, hash],
                )
                .is_ok();
            drop(conn);
            if inserted {
                events.push(Event::CvAdded {
                    cv_id,
                    job_id: job_id.clone(),
                    file_path: path.clone(),
                    file_name,
                    file_type,
                    imported_at: chrono::Utc::now().to_rfc3339(),
                });
            }
        }

        // Phase 5: one inputs-version bump per scan when anything changed, so
        // the job re-ranks once instead of once per candidate.
        if ranking_inputs_changed {
            let conn = db.writer();
            let _ = conn.execute(
                "UPDATE jobs SET inputs_version = inputs_version + 1,
                        inputs_updated_at = datetime('now'), updated_at = datetime('now')
                 WHERE id = ?1",
                params![job_id],
            );
        }

        events
    }
}

/// File name of `path`, falling back to the whole path when it has none.
fn file_name_for(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

/// File type label for `path`, mirroring the importer's supported extensions.
fn file_type_for(path: &str) -> String {
    match std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("pdf") => "Pdf",
        Some("docx") => "Docx",
        Some("doc") => "Doc",
        Some("txt") => "Txt",
        _ => "Other",
    }
    .to_string()
}

/// `(file_name, file_type)` for a path, as stored on the candidate row.
fn file_identity(path: &str) -> (String, String) {
    (file_name_for(path), file_type_for(path))
}

pub struct Orchestrator {
    workers: usize,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            workers: env::var("CV_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1),
        }
    }

    pub fn with_workers(workers: usize) -> Self {
        Self { workers: workers.max(1) }
    }

    pub fn worker_count(&self) -> usize {
        self.workers
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EffectRunner {
    fn default() -> Self {
        Self::new()
    }
}


pub mod types {
    pub use super::*;
}
pub mod worker {
    pub use super::{Worker, WorkerError};
}
pub mod work_item {
    pub use super::WorkItem;
}
pub mod effect_runner {
    pub use super::EffectRunner;
}
pub mod orchestrator {
    pub use super::Orchestrator;
}
pub mod text_worker {
    pub use super::TextWorker;
}
pub mod scout_worker {
    pub use super::{ScoutWorker, ScoutContext};
}
pub mod heavy_worker {
    pub use super::{HeavyWorker, HeavyContext};
}
pub mod match_worker {
    pub use super::{MatchWorker, MatchContext};
}
pub mod comparator_worker {
    pub use super::{ComparatorWorker, ComparatorContext};
}

#[derive(Clone, Default)]
pub struct RunnerContext {
    pub runtime: Option<Arc<libtriqui_core::runtime::EngineRuntime>>,
    pub scout_model: Option<String>,
    pub db_path: Option<PathBuf>,
    pub db: Option<Arc<libtriqui_core::db::DbConnection>>,
}

pub struct PipelineRunner {
    _workers: usize,
    dispatch: Arc<Mutex<Option<Arc<dyn Fn(Event) + Send + Sync>>>>,
    state: Arc<Mutex<Option<Arc<Mutex<State>>>>>,
    context: Arc<Mutex<RunnerContext>>,
    effect_runner: EffectRunner,
}

impl PipelineRunner {
    pub fn new() -> Self {
        Self::with_workers(1)
    }

    pub fn with_workers(workers: usize) -> Self {
        Self {
            _workers: workers.max(1),
            dispatch: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(None)),
            context: Arc::new(Mutex::new(RunnerContext::default())),
            effect_runner: EffectRunner::with_workers(workers),
        }
    }

    pub fn configure(
        &self,
        dispatch: Arc<dyn Fn(Event) + Send + Sync>,
        state: Option<Arc<Mutex<State>>>,
        context: RunnerContext,
    ) {
        *self.dispatch.lock().expect("pipeline dispatch lock") = Some(dispatch);
        *self.state.lock().expect("pipeline state lock") = state;
        *self.context.lock().expect("pipeline context lock") = context;
    }

    pub fn dispatch_event(&self, event: Event) {
        let mut queue = vec![event];
        while let Some(ev) = queue.pop() {
            if let Some(dispatch) = self.dispatch.lock().expect("pipeline dispatch lock").clone() {
                dispatch(ev.clone());
            }
            let state_clone = self
                .state
                .lock()
                .expect("pipeline state lock")
                .as_ref()
                .and_then(|s| s.lock().ok().map(|g| g.clone()));
            if let Some(st) = state_clone {
                let work = self.effect_runner.effects(&st, &ev);
                for w in work {
                    if let Some(e) = self.process_item(w) {
                        queue.push(e);
                    }
                }
                match &ev {
                    Event::HeavyMatched { .. } | Event::ComparisonCompleted { .. } => {
                        if let Some(j) = self.effect_runner.next_comparator_work(&st) {
                            if let Some(e) = self.process_item(j) {
                                queue.push(e);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn enqueue(&self, item: WorkItem) {
        if let Some(ev) = self.process_item(item) {
            self.dispatch_event(ev);
        }
    }

    fn process_item(&self, item: WorkItem) -> Option<Event> {
        let ctx = self.context.lock().expect("pipeline context lock").clone();
        let state_handle = self.state.lock().expect("pipeline state lock").clone();
        let event = match item {
            WorkItem::TextExtract { cv_id } => {
                let path = state_handle
                    .as_ref()
                    .and_then(|s| s.lock().ok())
                    .and_then(|st| st.cvs.get(&cv_id).map(|c| c.file_path.clone()))
                    .unwrap_or_else(|| cv_id.clone());
                Some(TextWorker::new().extract(cv_id, &path))
            }
            WorkItem::Quick { cv_id } => {
                let text = state_handle
                    .as_ref()
                    .and_then(|s| s.lock().ok())
                    .and_then(|st| st.cvs.get(&cv_id).and_then(|c| c.text.clone()))
                    .unwrap_or_default();
                let embedder = ctx.runtime.as_ref().and_then(|rt| {
                    let model = ctx
                        .scout_model
                        .clone()
                        .or_else(|| Some("static-similarity-mrl-multilingual-v1".to_string()))?;
                    rt.embedder_for(&model).ok()
                });
                let sctx = ScoutContext { embedder };
                ScoutWorker::new()
                    .process_with_context(cv_id, &text, &sctx, 0)
                    .ok()
            }
            WorkItem::Deep { cv_id } => {
                let text = state_handle
                    .as_ref()
                    .and_then(|s| s.lock().ok())
                    .and_then(|st| st.cvs.get(&cv_id).and_then(|c| c.text.clone()))
                    .unwrap_or_default();
                let embedder = ctx
                    .runtime
                    .as_ref()
                    .and_then(|rt| rt.embedder().ok().map(|e| e as Arc<dyn libtriqui_core::embed::Embedder>));
                let hctx = HeavyContext { embedder };
                HeavyWorker::new()
                    .process_with_context(cv_id, &text, &hctx, 0)
                    .ok()
            }
            WorkItem::Match { cv_id, job_id } => {
                let (cv_vec, job_text, job_version) = if let Some(s) = state_handle.as_ref() {
                    if let Ok(st) = s.lock() {
                        let cv_vec = st
                            .cvs
                            .get(&cv_id)
                            .and_then(|c| c.heavy_vector.clone())
                            .map(|v| v.into_iter().map(|x| x as f64).collect());
                        let job_text = st.jobs.get(&job_id).map(|j| j.text.clone()).unwrap_or_default();
                        let job_version = st.jobs.get(&job_id).map(|j| j.version).unwrap_or(0);
                        (cv_vec, job_text, job_version)
                    } else {
                        (None, String::new(), 0)
                    }
                } else {
                    (None, String::new(), 0)
                };
                let target_vec = ctx.runtime.as_ref().and_then(|rt| {
                    let embedder = rt.embedder().ok()?;
                    embedder
                        .embed(&[&job_text])
                        .ok()
                        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
                });
                MatchWorker::new()
                    .process_with_vectors(cv_id, job_version, cv_vec, target_vec)
                    .ok()
            }
            WorkItem::Comparator { pair } => {
                let job_version = state_handle
                    .as_ref()
                    .and_then(|s| s.lock().ok())
                    .and_then(|st| st.jobs.values().next().map(|j| j.version))
                    .unwrap_or(0);
                ComparatorWorker::new()
                    .process_with_model(pair, job_version, 0, None, None)
                    .ok()
            }
            WorkItem::ScanDirectory { job_id } => {
                let db = match ctx.db.clone() {
                    Some(db) => db,
                    None => {
                        log::warn!("scan {}: no db configured", job_id);
                        return None;
                    }
                };
                let events = ScanWorker::new().scan(job_id, &db);
                for ev in events {
                    self.dispatch_event(ev);
                }
                None
            }
        };
        event
    }
}

pub mod pipeline_runner {
    pub use super::{PipelineRunner, RunnerContext};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn effect_runner_default_one_worker() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("CV_PIPELINE_WORKERS");
        let r = EffectRunner::new();
        assert_eq!(r.workers(), 1);
    }

    #[test]
    fn cv_added_schedules_text_extract() {
        let mut s = State::default();
        let e = Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
            imported_at: "2024-01-01T00:00:00Z".into(),
        };
        s = libcv_state::reduce(s, &e);
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert_eq!(work, vec![WorkItem::TextExtract { cv_id: "c1".into() }]);
    }

    #[test]
    fn text_extracted_schedules_scout_only() {
        let e = Event::TextExtracted {
            cv_id: "c1".into(),
            text: "hi".into(),
            status: libcv_state::ExtractionStatus::Extracted,
        };
        let s = State::default();
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert_eq!(work, vec![WorkItem::Quick { cv_id: "c1".into() }]);
    }

    #[test]
    fn scout_completed_schedules_heavy() {
        let e = Event::ScoutCompleted {
            cv_id: "c1".into(),
            vector: vec![0.1, 0.2],
            model_bundle_version: 1,
        };
        let s = State::default();
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert_eq!(work, vec![WorkItem::Deep { cv_id: "c1".into() }]);
    }

    #[test]
    fn orchestrator_reads_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("CV_PIPELINE_WORKERS", "4");
        let o = Orchestrator::new();
        assert_eq!(o.worker_count(), 4);
        std::env::remove_var("CV_PIPELINE_WORKERS");
    }

    // -----------------------------------------------------------------------
    // Directory reconciliation (reconcile-cv-directory-with-candidate-set)
    // -----------------------------------------------------------------------

    use libtriqui_core::chunks::hash_bytes;
    use libtriqui_core::db::{open_connection_at, DbConnection};
    use rusqlite::params as sql_params;

    /// A temporary watched CV directory + database, plus helpers to drive scans
    /// and inspect the candidate rows the scan produces.
    struct ScanFixture {
        dir: std::path::PathBuf,
        db: Option<Arc<DbConnection>>,
        root: std::path::PathBuf,
    }

    impl ScanFixture {
        fn new(files: &[(&str, &str)]) -> Self {
            let root = std::env::temp_dir().join(format!("triqui-scan-{}", uuid::Uuid::new_v4()));
            let dir = root.join("cv");
            std::fs::create_dir_all(&dir).unwrap();
            for (name, body) in files {
                std::fs::write(dir.join(name), body).unwrap();
            }
            let db = Arc::new(open_connection_at(&root.join("triqui.db")).unwrap());
            {
                let conn = db.writer();
                conn.execute(
                    "INSERT INTO jobs (id, title, import_directory_path)
                     VALUES ('job1', 'Job', ?1)",
                    sql_params![dir.to_string_lossy().to_string()],
                )
                .unwrap();
            }
            Self {
                dir,
                db: Some(db),
                root,
            }
        }

        fn db(&self) -> &Arc<DbConnection> {
            self.db.as_ref().expect("fixture db")
        }

        fn write(&self, name: &str, body: &str) {
            std::fs::write(self.dir.join(name), body).unwrap();
        }

        fn remove(&self, name: &str) {
            std::fs::remove_file(self.dir.join(name)).unwrap();
        }

        fn scan(&self) -> Vec<Event> {
            ScanWorker::new().scan("job1".into(), self.db())
        }

        /// `(id, path, extraction_status, content_hash, deleted)` ordered by path.
        fn rows(&self) -> Vec<(String, String, String, Option<String>, bool)> {
            let conn = self.db().read_conn();
            let mut stmt = conn
                .prepare(
                    "SELECT id, source_file_path, extraction_status, content_hash, deleted
                     FROM candidates WHERE job_id = 'job1' ORDER BY source_file_path, id",
                )
                .unwrap();
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, i64>(4)? != 0,
                    ))
                })
                .unwrap();
            rows.map(|r| r.unwrap()).collect()
        }

        fn id_for(&self, suffix: &str) -> String {
            self.rows()
                .into_iter()
                .find(|r| r.1.ends_with(suffix))
                .unwrap_or_else(|| panic!("no candidate for {suffix}"))
                .0
        }

        fn inputs_version(&self) -> i64 {
            let conn = self.db().read_conn();
            conn.query_row(
                "SELECT inputs_version FROM jobs WHERE id = 'job1'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        }

        /// Simulate a candidate that already went through extraction + scoring.
        fn mark_extracted_and_scored(&self, id: &str, text: &str, liked: bool) {
            let conn = self.db().writer();
            conn.execute(
                "UPDATE candidates SET extraction_status = 'Extracted', extracted_text = ?1,
                        vector = X'0102', vector_model = 'model-a', rank = 1,
                        points = 5.0, points_max = 10.0, liked = ?2
                 WHERE id = ?3",
                sql_params![text, liked as i64, id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO candidate_chunks
                    (id, candidate_id, chunk_index, chunk_text, char_start, char_end,
                     content_hash, vector, vector_model)
                 VALUES (?1, ?2, 0, 'text', 0, 4, 'h', X'00', 'model-a')",
                sql_params![format!("cc_{id}"), id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO candidate_scores
                    (id, candidate_id, question_id, kind, sign, label, points, max_points)
                 VALUES (?1, ?2, 'ad', 'Ad', 'Positive', 'Job ad', 3.0, 10.0)",
                sql_params![format!("cs_{id}"), id],
            )
            .unwrap();
        }

        fn scalar_i64(&self, sql: &str, id: &str) -> i64 {
            let conn = self.db().read_conn();
            conn.query_row(sql, sql_params![id], |r| r.get(0)).unwrap()
        }

        fn chunk_count(&self, id: &str) -> i64 {
            self.scalar_i64(
                "SELECT COUNT(*) FROM candidate_chunks WHERE candidate_id = ?1",
                id,
            )
        }

        fn score_count(&self, id: &str) -> i64 {
            self.scalar_i64(
                "SELECT COUNT(*) FROM candidate_scores WHERE candidate_id = ?1",
                id,
            )
        }

        fn vector_is_null(&self, id: &str) -> bool {
            self.scalar_i64(
                "SELECT CASE WHEN vector IS NULL THEN 1 ELSE 0 END FROM candidates WHERE id = ?1",
                id,
            ) == 1
        }

        fn extracted_text(&self, id: &str) -> Option<String> {
            let conn = self.db().read_conn();
            conn.query_row(
                "SELECT extracted_text FROM candidates WHERE id = ?1",
                sql_params![id],
                |r| r.get(0),
            )
            .unwrap()
        }

        fn liked(&self, id: &str) -> bool {
            self.scalar_i64("SELECT liked FROM candidates WHERE id = ?1", id) != 0
        }
    }

    impl Drop for ScanFixture {
        fn drop(&mut self) {
            drop(self.db.take());
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn scan_populates_content_hash_and_second_scan_is_a_noop() {
        let fx = ScanFixture::new(&[("a.txt", "alpha"), ("b.txt", "beta")]);

        let first = fx.scan();
        assert_eq!(first.len(), 2, "both files import: {first:?}");

        let rows = fx.rows();
        assert_eq!(rows.len(), 2);
        for (_, path, _status, hash, deleted) in &rows {
            assert!(hash.is_some(), "content_hash populated for {path}");
            assert!(!deleted, "new candidate is active");
        }

        let version = fx.inputs_version();
        let second = fx.scan();
        assert!(
            second.is_empty(),
            "an unchanged directory must emit no events: {second:?}"
        );
        assert_eq!(fx.rows().len(), 2, "no duplicate candidates");
        assert_eq!(
            fx.inputs_version(),
            version,
            "an unchanged directory must not request a re-rank"
        );
    }

    #[test]
    fn scan_retires_deleted_files_and_drops_derived_rows() {
        let fx = ScanFixture::new(&[("a.txt", "alpha"), ("b.txt", "beta")]);
        fx.scan();
        let a = fx.id_for("a.txt");
        let b = fx.id_for("b.txt");
        fx.mark_extracted_and_scored(&a, "alpha", false);
        fx.mark_extracted_and_scored(&b, "beta", false);
        assert_eq!(fx.inputs_version(), 0);

        fx.remove("a.txt");
        let events = fx.scan();
        assert!(
            matches!(events.as_slice(), [Event::CvDeleted { cv_id, .. }] if cv_id == &a),
            "expected one CvDeleted for the missing file: {events:?}"
        );

        let rows = fx.rows();
        assert!(rows.iter().find(|r| r.0 == a).unwrap().4, "a is deleted");
        assert!(!rows.iter().find(|r| r.0 == b).unwrap().4, "b is untouched");
        assert_eq!(fx.chunk_count(&a), 0, "deleted chunks dropped");
        assert_eq!(fx.score_count(&a), 0, "deleted scores dropped");
        assert_eq!(fx.chunk_count(&b), 1, "other candidates keep their chunks");
        assert_eq!(fx.score_count(&b), 1, "other candidates keep their scores");
        assert_eq!(fx.inputs_version(), 1, "deletion bumps the input version");
    }

    #[test]
    fn scan_resets_modified_candidate_and_keeps_the_row() {
        let fx = ScanFixture::new(&[("a.txt", "alpha")]);
        fx.scan();
        let id = fx.id_for("a.txt");
        fx.mark_extracted_and_scored(&id, "alpha", true);

        fx.write("a.txt", "alpha beta");
        let events = fx.scan();
        assert!(
            matches!(events.as_slice(), [Event::CvAdded { cv_id, .. }] if cv_id == &id),
            "the same candidate is re-dispatched: {events:?}"
        );

        let row = fx.rows().remove(0);
        assert_eq!(row.0, id, "the row is reused, not replaced");
        assert_eq!(row.2, "Pending", "reset to Pending");
        assert_eq!(
            row.3.as_deref(),
            Some(hash_bytes(b"alpha beta").as_str()),
            "the new fingerprint is stored"
        );
        assert!(!row.4, "the candidate is still active");
        assert_eq!(fx.extracted_text(&id), None, "extracted text cleared");
        assert!(fx.vector_is_null(&id), "vector cleared");
        assert_eq!(fx.chunk_count(&id), 0, "chunks cleared");
        assert_eq!(fx.score_count(&id), 0, "scores cleared");
        assert!(fx.liked(&id), "the liked star survives the reset");
        assert_eq!(fx.inputs_version(), 1, "modification bumps the input version");
    }

    #[test]
    fn scan_backfills_null_content_hash_without_resetting() {
        let fx = ScanFixture::new(&[("a.txt", "alpha")]);
        fx.scan();
        let id = fx.id_for("a.txt");
        fx.mark_extracted_and_scored(&id, "alpha", false);
        // Simulate a pre-upgrade row whose fingerprint was never recorded.
        fx.db()
            .writer()
            .execute(
                "UPDATE candidates SET content_hash = NULL WHERE id = ?1",
                sql_params![id],
            )
            .unwrap();

        let events = fx.scan();
        assert!(events.is_empty(), "a NULL hash is backfilled silently: {events:?}");
        let row = fx.rows().remove(0);
        assert_eq!(row.2, "Extracted", "the candidate is NOT reset");
        assert_eq!(
            row.3.as_deref(),
            Some(hash_bytes(b"alpha").as_str()),
            "the fingerprint is backfilled"
        );
        assert_eq!(fx.extracted_text(&id).as_deref(), Some("alpha"));
        assert_eq!(fx.chunk_count(&id), 1);
        assert_eq!(fx.inputs_version(), 0, "a backfill is not a ranking change");
    }

    #[test]
    fn scan_reimports_path_whose_only_row_is_deleted() {
        let fx = ScanFixture::new(&[("a.txt", "alpha")]);
        fx.scan();
        let first_id = fx.id_for("a.txt");

        fx.remove("a.txt");
        fx.scan(); // retires the row
        fx.write("a.txt", "alpha revised");

        let events = fx.scan();
        let new_id = match events.as_slice() {
            [Event::CvAdded { cv_id, .. }] => cv_id.clone(),
            other => panic!("expected exactly one import: {other:?}"),
        };
        assert_ne!(new_id, first_id, "a fresh candidate row is created");

        let rows = fx.rows();
        assert_eq!(rows.len(), 2, "the old row stays deleted alongside the new one");
        let active: Vec<_> = rows.iter().filter(|r| !r.4).collect();
        assert_eq!(active.len(), 1, "exactly one active candidate");
        assert_eq!(active[0].0, new_id);
    }

    #[test]
    fn scan_converges_new_modified_and_deleted_in_one_pass() {
        let fx = ScanFixture::new(&[("keep.txt", "k"), ("edit.txt", "e1"), ("gone.txt", "g")]);
        fx.scan();
        let edit_id = fx.id_for("edit.txt");
        let gone_id = fx.id_for("gone.txt");
        fx.mark_extracted_and_scored(&edit_id, "e1", false);

        fx.write("edit.txt", "e2");
        fx.remove("gone.txt");
        fx.write("new.txt", "n");

        let events = fx.scan();
        assert_eq!(events.len(), 3, "one event per outcome: {events:?}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::CvDeleted { cv_id, .. } if cv_id == &gone_id)),
            "the deleted file retires its candidate: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::CvAdded { cv_id, .. } if cv_id == &edit_id)),
            "the edited file resets its candidate: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::CvAdded { file_name, .. } if file_name == "new.txt")),
            "the new file imports: {events:?}"
        );

        let rows = fx.rows();
        let active: Vec<_> = rows.iter().filter(|r| !r.4).collect();
        let names: Vec<&str> = active.iter().map(|r| r.1.rsplit('/').next().unwrap()).collect();
        assert_eq!(active.len(), 3, "keep + edit + new are active: {names:?}");
        assert!(names.contains(&"keep.txt"));
        assert!(names.contains(&"edit.txt"));
        assert!(names.contains(&"new.txt"));
        assert!(rows.iter().find(|r| r.0 == gone_id).unwrap().4, "gone is retired");
        assert_eq!(
            fx.inputs_version(),
            1,
            "all three outcomes share a single version bump"
        );
    }

    #[test]
    fn cv_deleted_retires_the_record_and_excludes_it_from_comparator_scheduling() {
        let mut s = State::default();
        for id in ["c1", "c2", "c3"] {
            s = libcv_state::reduce(
                s,
                &Event::CvAdded {
                    cv_id: id.into(),
                    job_id: "j1".into(),
                    file_path: format!("/tmp/{id}.txt"),
                    file_name: format!("{id}.txt"),
                    file_type: "Txt".into(),
                    imported_at: "2024-01-01T00:00:00Z".into(),
                },
            );
        }
        assert_eq!(provisional_sim_rank(&s).len(), 3);

        let ev = Event::CvDeleted {
            cv_id: "c2".into(),
            job_id: "j1".into(),
        };
        s = libcv_state::reduce(s, &ev);

        assert!(s.cvs.get("c2").unwrap().deleted, "the reducer marks it deleted");
        let eligible: Vec<String> = provisional_sim_rank(&s)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(eligible.len(), 2, "the retired candidate is excluded");
        assert!(!eligible.iter().any(|id| id == "c2"));
    }
}
