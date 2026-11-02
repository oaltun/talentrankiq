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
    Scout { cv_id: String },
    Heavy { cv_id: String },
    Match { cv_id: String, job_id: String },
    Judge { pair: (String, String) },
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
                out.push(WorkItem::TextExtract { cv_id: cv_id.clone() });
            }
            Event::TextExtracted { cv_id, .. } => {
                out.push(WorkItem::Scout { cv_id: cv_id.clone() });
                out.push(WorkItem::Heavy { cv_id: cv_id.clone() });
            }
            Event::HeavyCompleted { cv_id, .. } => {
                // Match work is enqueued when a job ad version is known.
                for (job_id, job) in &state.jobs {
                    // Only if this CV belongs to the job.
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
                // Ask judge orderer for next pair (handled by caller with state).
            }
            Event::ComparisonCompleted { .. } => {}
            _ => {}
        }
        // Judge scheduling: after HeavyMatched accumulation, consult next_pair.
        // Delegated to caller; here we leave judge work to explicit enqueue.
        out
    }

    /// Decide judge work given current state (call after HeavyMatched/ComparisonCompleted).
    pub fn next_judge_work(&self, state: &State) -> Option<WorkItem> {
        use libcv_judge_order::{JudgeOrderState, next_pair};
        let mut provisional: Vec<(String, f64)> = state
            .cvs
            .values()
            .filter(|c| !c.deleted)
            .map(|c| (c.id.clone(), c.heavy_similarity.unwrap_or(0.0)))
            .collect();
        provisional.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let comparisons: HashSet<(String, String)> = state
            .comparisons
            .iter()
            .map(|(a, b, _, _)| {
                if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                }
            })
            .collect();
        let job_version = state.jobs.values().next().map(|j| j.version).unwrap_or(0);
        let order_state = JudgeOrderState {
            glicko: state
                .glicko
                .iter()
                .map(|((cv, ver), g)| {
                    (
                        (cv.clone(), *ver),
                        libjudge::tournament::GlickoState { rating: g.rating, rd: g.rd },
                    )
                })
                .collect(),
            comparisons,
            provisional_sim_rank: provisional,
            job_version,
            entitled: state.license.is_some(),
        };
        next_pair(&order_state).map(|p| WorkItem::Judge { pair: p })
    }
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
pub struct JudgeWorker;

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
impl JudgeWorker {
    pub fn new() -> Self { Self }
}
impl Default for JudgeWorker {
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

impl Worker for JudgeWorker {
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

pub struct JudgeContext {
    pub model: Option<libjudge::JudgeModel>,
}

impl Default for JudgeContext {
    fn default() -> Self {
        Self { model: None }
    }
}

impl ScoutWorker {
    pub fn embed(&self, ctx: &ScoutContext, text: &str) -> Vec<f32> {
        if let Some(embedder) = &ctx.embedder {
                        if let Ok(mut v) = embedder.embed(&[text]) {
                if let Some(vec) = v.pop() {
                    return vec;
                }
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
                        if let Ok(mut v) = embedder.embed(&[text]) {
                if let Some(vec) = v.pop() {
                    return vec;
                }
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

impl JudgeWorker {
    pub fn pair_prob(&self, model: &libjudge::JudgeModel, x: Vec<f32>) -> f64 {
        model.pair_prob(x)
    }

    pub fn process_with_model(
        &mut self,
        pair: (String, String),
        job_version: u32,
        model_bundle_version: u32,
        model: Option<&libjudge::JudgeModel>,
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
    pub fn scan(
        &self,
        job_id: String,
        db_path: &PathBuf,
        dispatch: &Arc<dyn Fn(Event) + Send + Sync>,
    ) {
        use libtriqui_core::db::open_connection_at;
        use libtriqui_core::import::collect_supported_files;
        use rusqlite::params;

        let conn = match open_connection_at(db_path) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("scan: failed to open db for {}: {}", job_id, e);
                return;
            }
        };

        let dir: Option<String> = conn
            .query_row(
                "SELECT import_directory_path FROM jobs WHERE id = ?1",
                params![job_id],
                |r| r.get(0),
            )
            .ok()
            .flatten();

        let Some(dir) = dir.filter(|d| !d.is_empty()) else {
            return;
        };

        let files = match collect_supported_files(&dir) {
            Ok(files) => files,
            Err(e) => {
                log::warn!("scan {}: scan of {} failed: {}", job_id, dir, e);
                return;
            }
        };
        if files.is_empty() {
            return;
        }

        let new_files: Vec<PathBuf> = {
            files
                .into_iter()
                .filter(|p| {
                    let path_str = p.to_string_lossy().to_string();
                    !conn
                        .query_row(
                            "SELECT 1 FROM candidates WHERE job_id = ?1 AND source_file_path = ?2 AND extraction_status = 'Extracted'",
                            params![job_id, &path_str],
                            |_| Ok(true),
                        )
                        .unwrap_or(false)
                })
                .collect()
        };
        if new_files.is_empty() {
            return;
        }

        log::info!("scan {}: {} new file(s) in {}", job_id, new_files.len(), dir);

        for file_path in new_files {
            let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
            let file_type = match file_path.extension().and_then(|e| e.to_str()) {
                Some("pdf") => "Pdf",
                Some("docx") => "Docx",
                Some("doc") => "Doc",
                Some("txt") => "Txt",
                _ => "Other",
            }.to_string();

            let cv_id = format!("cv_{}", uuid::Uuid::new_v4());

            conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, imported_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'Pending', datetime('now'))",
                params![cv_id, job_id, file_path.to_string_lossy().to_string(), file_name, file_type],
            ).ok();

            dispatch(Event::CvAdded {
                cv_id: cv_id.clone(),
                job_id: job_id.clone(),
                file_path: file_path.to_string_lossy().to_string(),
                file_name,
                file_type,
                imported_at: chrono::Utc::now().to_rfc3339(),
            });
        }
    }
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
pub mod judge_worker {
    pub use super::{JudgeWorker, JudgeContext};
}

#[derive(Clone, Default)]
pub struct RunnerContext {
    pub runtime: Option<Arc<libtriqui_core::runtime::EngineRuntime>>,
    pub scout_model: Option<String>,
    pub db_path: Option<PathBuf>,
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
        if let Some(dispatch) = self.dispatch.lock().expect("pipeline dispatch lock").clone() {
            dispatch(event.clone());
        }
        let state_clone = self
            .state
            .lock()
            .expect("pipeline state lock")
            .as_ref()
            .and_then(|s| s.lock().ok().map(|g| g.clone()));
        if let Some(st) = state_clone {
            let work = self.effect_runner.effects(&st, &event);
            for w in work {
                self.process_item(w);
            }
            match &event {
                Event::HeavyMatched { .. } | Event::ComparisonCompleted { .. } => {
                    if let Some(j) = self.effect_runner.next_judge_work(&st) {
                        self.process_item(j);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn enqueue(&self, item: WorkItem) {
        self.process_item(item);
    }

    fn process_item(&self, item: WorkItem) {
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
            WorkItem::Scout { cv_id } => {
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
            WorkItem::Heavy { cv_id } => {
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
                        .and_then(|mut v| v.pop())
                        .map(|vec| vec.into_iter().map(|x| x as f64).collect())
                });
                MatchWorker::new()
                    .process_with_vectors(cv_id, job_version, cv_vec, target_vec)
                    .ok()
            }
            WorkItem::Judge { pair } => {
                let job_version = state_handle
                    .as_ref()
                    .and_then(|s| s.lock().ok())
                    .and_then(|st| st.jobs.values().next().map(|j| j.version))
                    .unwrap_or(0);
                JudgeWorker::new()
                    .process_with_model(pair, job_version, 0, None, None)
                    .ok()
            }
            WorkItem::ScanDirectory { job_id } => {
                let db_path = match ctx.db_path.clone() {
                    Some(p) => p,
                    None => {
                        log::warn!("scan {}: no db_path configured", job_id);
                        return;
                    }
                };
                let dispatch = match self.dispatch.lock().expect("pipeline dispatch lock").clone() {
                    Some(d) => d,
                    None => {
                        log::warn!("scan {}: no dispatch configured", job_id);
                        return;
                    }
                };
                ScanWorker::new().scan(job_id, &db_path, &dispatch);
                None
            }
        };
        if let Some(ev) = event {
            self.dispatch_event(ev);
        }
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
        assert!(work.iter().any(|w| matches!(w, WorkItem::TextExtract { cv_id } if cv_id == "c1")));
    }

    #[test]
    fn text_extracted_schedules_scout_and_heavy() {
        let e = Event::TextExtracted {
            cv_id: "c1".into(),
            text: "hi".into(),
            status: libcv_state::ExtractionStatus::Extracted,
        };
        let s = State::default();
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert!(work.iter().any(|w| matches!(w, WorkItem::Scout { .. })));
        assert!(work.iter().any(|w| matches!(w, WorkItem::Heavy { .. })));
    }

    #[test]
    fn orchestrator_reads_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("CV_PIPELINE_WORKERS", "4");
        let o = Orchestrator::new();
        assert_eq!(o.worker_count(), 4);
        std::env::remove_var("CV_PIPELINE_WORKERS");
    }
}
