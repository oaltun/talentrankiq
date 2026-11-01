use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

pub use libcv_core::{CvId, JobId, CriterionId, FavId, PairKey};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, specta::Type)]
pub enum ExtractionStatus {
    #[default]
    Pending,
    Extracted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, specta::Type)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct JobRecord {
    pub id: JobId,
    pub text: String,
    pub version: u32,
    pub criteria: Vec<CriterionId>,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct GlickoState {
    pub rating: f64,
    pub rd: f64,
}

impl Default for GlickoState {
    fn default() -> Self {
        Self { rating: 1000.0, rd: 350.0 }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct FavRecord {
    pub fav_id: FavId,
    pub cv_id: CvId,
    pub workspace_id: JobId,
    pub ts: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
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
        Event::ComparisonCompleted { pair, job_version, model_bundle_version, result } => {
            let key = (pair.0.clone(), pair.1.clone(), *job_version, *model_bundle_version);
            if state.comparisons.contains(&key) {
                return state;
            }
            state.comparisons.insert(key);
            let score_min = if *result < 0.0 {
                0.0
            } else if *result > 1.0 {
                1.0
            } else {
                *result
            };
            let score_max = 1.0 - score_min;
            let g_a = state
                .glicko
                .get(&(pair.0.clone(), *job_version))
                .copied()
                .unwrap_or_default();
            let g_b = state
                .glicko
                .get(&(pair.1.clone(), *job_version))
                .copied()
                .unwrap_or_default();
            let new_a = libjudge::tournament::glicko_update(
                libjudge::tournament::GlickoState { rating: g_a.rating, rd: g_a.rd },
                &[(libjudge::tournament::GlickoState { rating: g_b.rating, rd: g_b.rd }, score_min)],
            );
            let new_b = libjudge::tournament::glicko_update(
                libjudge::tournament::GlickoState { rating: g_b.rating, rd: g_b.rd },
                &[(libjudge::tournament::GlickoState { rating: g_a.rating, rd: g_a.rd }, score_max)],
            );
            state.glicko.insert(
                (pair.0.clone(), *job_version),
                GlickoState { rating: new_a.rating, rd: new_a.rd },
            );
            state.glicko.insert(
                (pair.1.clone(), *job_version),
                GlickoState { rating: new_b.rating, rd: new_b.rd },
            );
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct Entitlements {
    pub head_to_head: bool,
}

pub fn compute_entitlements(state: &State) -> Entitlements {
    Entitlements { head_to_head: state.license.is_some() }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct Progress {
    pub total: usize,
    pub done: usize,
}

pub fn compute_progress(state: &State, job_id: &str) -> Progress {
    let total = state.cvs.values().filter(|c| c.job_id == job_id).count();
    let done = state.cvs.values().filter(|c| c.job_id == job_id && c.heavy_vector.is_some()).count();
    Progress { total, done }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct RankedCandidate {
    pub cv_id: CvId,
    pub heavy_similarity: Option<f64>,
    pub glicko_rating: Option<f64>,
}

pub fn compute_ranking(state: &State, job_id: &str) -> Vec<RankedCandidate> {
    let job_version = state.jobs.get(job_id).map(|j| j.version).unwrap_or(0);
    let mut out: Vec<RankedCandidate> = state
        .cvs
        .values()
        .filter(|c| c.job_id == job_id && !c.deleted)
        .map(|c| {
            let glicko_rating = state
                .glicko
                .get(&(c.id.clone(), job_version))
                .map(|g| g.rating);
            RankedCandidate {
                cv_id: c.id.clone(),
                heavy_similarity: c.heavy_similarity,
                glicko_rating,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        let ga = a
            .glicko_rating
            .map(|r| {
                libjudge::tournament::glicko_to_score(libjudge::tournament::GlickoState {
                    rating: r,
                    rd: 350.0,
                })
            })
            .unwrap_or(0.5);
        let gb = b
            .glicko_rating
            .map(|r| {
                libjudge::tournament::glicko_to_score(libjudge::tournament::GlickoState {
                    rating: r,
                    rd: 350.0,
                })
            })
            .unwrap_or(0.5);
        match b
            .heavy_similarity
            .partial_cmp(&a.heavy_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        }
    });
    out
}

pub fn append_event(conn: &rusqlite::Connection, event: &Event) -> Result<(), String> {
    let payload = serde_json::to_string(event).map_err(|e| e.to_string())?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS event_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, cv_id TEXT, job_id TEXT, event_type TEXT NOT NULL, payload TEXT NOT NULL, ts INTEGER NOT NULL)",
        [],
    ).map_err(|e| e.to_string())?;
    let event_type = serde_json::to_value(event)
        .ok()
        .and_then(|v| v.get("type").and_then(|t| t.as_str()).map(String::from))
        .unwrap_or_else(|| "event".to_string());
    conn.execute(
        "INSERT INTO event_log (cv_id, job_id, event_type, payload, ts) VALUES (NULL, NULL, ?1, ?2, strftime('%s','now'))",
        rusqlite::params![event_type, payload],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn replay_from(conn: &rusqlite::Connection, seq: i64) -> Result<Vec<Event>, String> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS event_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, cv_id TEXT, job_id TEXT, event_type TEXT NOT NULL, payload TEXT NOT NULL, ts INTEGER NOT NULL)",
        [],
    ).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT payload FROM event_log WHERE seq > ?1 ORDER BY seq")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![seq], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let payload = row.map_err(|e| e.to_string())?;
        let event: Event = serde_json::from_str(&payload).map_err(|e| e.to_string())?;
        out.push(event);
    }
    Ok(out)
}

pub mod event {
    pub use super::ExtractionStatus;
    pub use super::Event;
}
pub mod state {
    pub use super::{State, CvRecord, JobRecord, GlickoState, FavRecord, Entitlements, Progress, RankedCandidate};
}
pub mod reducer {
    pub use super::reduce;
}
pub mod projections {
    pub use super::{compute_entitlements, compute_progress, compute_ranking, Entitlements, Progress, RankedCandidate};
}
pub mod event_log {
    pub use super::{append_event, replay_from};
}


pub mod shadow {
    pub use super::reduce;
    pub use super::State;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducer_purity() {
        let s = State::default();
        let e = Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
        };
        let a = reduce(s.clone(), &e);
        let b = reduce(s, &e);
        assert_eq!(a, b);
    }

    #[test]
    fn reducer_idempotent_cv_added() {
        let s = State::default();
        let e = Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
        };
        let once = reduce(s.clone(), &e);
        let twice = reduce(once.clone(), &e);
        assert_eq!(once, twice);
    }

    #[test]
    fn replay_produces_same_state() {
        let e1 = Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
        };
        let e2 = Event::TextExtracted {
            cv_id: "c1".into(),
            text: "hello".into(),
            status: ExtractionStatus::Extracted,
        };
        let mut s = State::default();
        s = reduce(s, &e1);
        s = reduce(s, &e2);
        // replay via fresh
        let mut s2 = State::default();
        for e in [&e1, &e2] {
            s2 = reduce(s2, e);
        }
        assert_eq!(s, s2);
    }

    #[test]
    fn compute_ranking_filters_deleted() {
        let mut s = State::default();
        s.cvs.insert("c1".into(), CvRecord {
            id: "c1".into(),
            job_id: "j1".into(),
            heavy_similarity: Some(0.9),
            deleted: false,
            ..Default::default()
        });
        s.cvs.insert("c2".into(), CvRecord {
            id: "c2".into(),
            job_id: "j1".into(),
            heavy_similarity: Some(0.5),
            deleted: true,
            ..Default::default()
        });
        let r = compute_ranking(&s, "j1");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].cv_id, "c1");
    }
}
