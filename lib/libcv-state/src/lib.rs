use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

pub type CvId = String;
pub type JobId = String;
pub type CriterionId = String;
pub type FavId = String;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExtractionStatus {
    #[default]
    Pending,
    Extracted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    CvAdded { cv_id: CvId, job_id: JobId, file_path: String, file_name: String, file_type: String },
    TextExtracted { cv_id: CvId, text: String, status: ExtractionStatus },
    ScoutCompleted { cv_id: CvId, vector: Vec<f32>, model_bundle_version: u32 },
    HeavyCompleted { cv_id: CvId, vector: Vec<f32>, model_bundle_version: u32 },
    JobAdUpdated { job_id: JobId, version: u32, text: String },
    CriterionAdded { criterion_id: CriterionId, job_id: JobId, text: String, weight: f64 },
    CriterionUpdated { criterion_id: CriterionId, text: String, weight: f64 },
    CriterionRemoved { criterion_id: CriterionId },
    ScoutMatched { cv_id: CvId, job_version: u32, similarity: f64 },
    HeavyMatched { cv_id: CvId, job_version: u32, similarity: f64 },
    ComparisonCompleted { pair: (CvId, CvId), job_version: u32, model_bundle_version: u32, result: f64 },
    GlickoUpdated { cv_id: CvId, job_version: u32, rating: f64, rd: f64 },
    FavToggled { cv_id: CvId, workspace_id: JobId, fav_id: FavId, ts: i64 },
    RubricWeightChanged { job_id: JobId, weights: Vec<f64> },
    LicenseActivated { license: String },
    LicenseExpired { license: String },
    CvSoftDeleted { cv_id: CvId },
    CvForgotten { cv_id: CvId },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CvRecord {
    pub id: CvId,
    pub job_id: JobId,
    pub file_path: String,
    pub file_name: String,
    pub file_type: String,
    pub text: Option<String>,
    pub extraction_status: ExtractionStatus,
    pub scout_vector: Option<Vec<f32>>,
    pub heavy_vector: Option<Vec<f32>>,
    pub scout_similarity: Option<f64>,
    pub heavy_similarity: Option<f64>,
    pub deleted: bool,
    pub model_bundle_version: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: JobId,
    pub text: String,
    pub version: u32,
    pub criteria: Vec<CriterionId>,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlickoState {
    pub rating: f64,
    pub rd: f64,
}

impl Default for GlickoState {
    fn default() -> Self {
        Self { rating: 1000.0, rd: 350.0 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FavRecord {
    pub fav_id: FavId,
    pub cv_id: CvId,
    pub workspace_id: JobId,
    pub ts: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct State {
    pub cvs: HashMap<CvId, CvRecord>,
    pub jobs: HashMap<JobId, JobRecord>,
    pub comparisons: HashSet<(CvId, CvId, u32, u32)>,
    pub glicko: HashMap<(CvId, u32), GlickoState>,
    pub favs: HashMap<FavId, FavRecord>,
    pub license: Option<String>,
}

pub fn reduce(mut state: State, event: &Event) -> State {
    match event {
        Event::CvAdded { cv_id, job_id, file_path, file_name, file_type } => {
            state.cvs.insert(cv_id.clone(), CvRecord {
                id: cv_id.clone(),
                job_id: job_id.clone(),
                file_path: file_path.clone(),
                file_name: file_name.clone(),
                file_type: file_type.clone(),
                ..Default::default()
            });
        }
        Event::TextExtracted { cv_id, text, status } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.text = Some(text.clone());
                cv.extraction_status = *status;
            }
        }
        Event::ScoutCompleted { cv_id, vector, model_bundle_version } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.scout_vector = Some(vector.clone());
                cv.model_bundle_version = *model_bundle_version;
            }
        }
        Event::HeavyCompleted { cv_id, vector, model_bundle_version } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.heavy_vector = Some(vector.clone());
                cv.model_bundle_version = *model_bundle_version;
            }
        }
        Event::JobAdUpdated { job_id, version, text } => {
            let job = state.jobs.entry(job_id.clone()).or_default();
            job.id = job_id.clone();
            job.text = text.clone();
            job.version = *version;
        }
        Event::CriterionAdded { criterion_id, job_id, text: _, weight: _ } => {
            let job = state.jobs.entry(job_id.clone()).or_default();
            if !job.criteria.contains(criterion_id) {
                job.criteria.push(criterion_id.clone());
            }
        }
        Event::CriterionUpdated { criterion_id: _, text: _, weight: _ } => {}
        Event::CriterionRemoved { criterion_id } => {
            for job in state.jobs.values_mut() {
                job.criteria.retain(|c| c != criterion_id);
            }
        }
        Event::ScoutMatched { cv_id, job_version: _, similarity } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.scout_similarity = Some(*similarity);
            }
        }
        Event::HeavyMatched { cv_id, job_version: _, similarity } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.heavy_similarity = Some(*similarity);
            }
        }
        Event::ComparisonCompleted { pair, job_version, model_bundle_version, result: _ } => {
            state.comparisons.insert((pair.0.clone(), pair.1.clone(), *job_version, *model_bundle_version));
        }
        Event::GlickoUpdated { cv_id, job_version, rating, rd } => {
            state.glicko.insert((cv_id.clone(), *job_version), GlickoState { rating: *rating, rd: *rd });
        }
        Event::FavToggled { fav_id, cv_id, workspace_id, ts } => {
            if state.favs.contains_key(fav_id) {
                state.favs.remove(fav_id);
            } else {
                state.favs.insert(fav_id.clone(), FavRecord {
                    fav_id: fav_id.clone(),
                    cv_id: cv_id.clone(),
                    workspace_id: workspace_id.clone(),
                    ts: *ts,
                });
            }
        }
        Event::RubricWeightChanged { job_id, weights } => {
            if let Some(job) = state.jobs.get_mut(job_id) {
                job.weights = weights.clone();
            }
        }
        Event::LicenseActivated { license } => {
            state.license = Some(license.clone());
        }
        Event::LicenseExpired { .. } => {
            state.license = None;
        }
        Event::CvSoftDeleted { cv_id } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.deleted = true;
            }
        }
        Event::CvForgotten { cv_id } => {
            state.cvs.remove(cv_id);
        }
    }
    state
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Entitlements {
    pub head_to_head: bool,
}

pub fn compute_entitlements(state: &State) -> Entitlements {
    Entitlements { head_to_head: state.license.is_some() }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Progress {
    pub total: usize,
    pub done: usize,
}

pub fn compute_progress(state: &State, job_id: &str) -> Progress {
    let total = state.cvs.values().filter(|c| c.job_id == job_id).count();
    let done = state.cvs.values().filter(|c| c.job_id == job_id && c.heavy_vector.is_some()).count();
    Progress { total, done }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub cv_id: CvId,
    pub heavy_similarity: Option<f64>,
    pub glicko_rating: Option<f64>,
}

pub fn compute_ranking(state: &State, job_id: &str) -> Vec<RankedCandidate> {
    let mut out: Vec<RankedCandidate> = state
        .cvs
        .values()
        .filter(|c| c.job_id == job_id && !c.deleted)
        .map(|c| RankedCandidate {
            cv_id: c.id.clone(),
            heavy_similarity: c.heavy_similarity,
            glicko_rating: state.glicko.get(&(c.id.clone(), state.jobs.get(job_id).map(|j| j.version).unwrap_or(0))).map(|g| g.rating),
        })
        .collect();
    out.sort_by(|a, b| b.heavy_similarity.partial_cmp(&a.heavy_similarity).unwrap_or(std::cmp::Ordering::Equal));
    out
}

pub fn append_event(conn: &rusqlite::Connection, event: &Event) -> Result<(), String> {
    let payload = serde_json::to_string(event).map_err(|e| e.to_string())?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS event_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, cv_id TEXT, job_id TEXT, event_type TEXT NOT NULL, payload TEXT NOT NULL, ts INTEGER NOT NULL)",
        [],
    ).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO event_log (cv_id, job_id, event_type, payload, ts) VALUES (NULL, NULL, ?1, ?2, strftime('%s','now'))",
        rusqlite::params!["event", payload],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn replay_from(conn: &rusqlite::Connection, _seq: i64) -> Result<Vec<Event>, String> {
    let _ = conn;
    Ok(Vec::new())
}