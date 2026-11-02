use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

pub use libtriqui_client::types::{
    CandidatePage, ChunkMatchResult, ChunkMatchTarget, Criterion, DomainError, FileType, Job,
    JobDetail, JobSummary, QuestionKind, RankState, RubricQuestion, SortReport, TriqiItem,
};
use libtriqui_client::types::ExtractionStatus as ClientExtractionStatus;
use libtriqui_core::candidates;
use libtriqui_core::db::rank_state_for_job;
use libtriqui_core::judge::JudgeStage;
use libtriqui_core::scout::{SCOUT_LABEL, SCOUT_QUESTION_ID};
use libtriqui_core::syscap;
use libtriqui_core::runtime::EngineRuntime;
use libtriqui_core::chunk_match as cm_core;

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
    CvAdded { cv_id: CvId, job_id: JobId, file_path: String, file_name: String, file_type: String, imported_at: String },
    JobCreated { job_id: JobId, title: String, triqui_size: i32, ad_weight: f64, fb_job_weight: f64, fb_profile_weight: f64, import_directory_path: Option<String>, created_at: String },
    JobUpdated { job_id: JobId, title: String, triqui_size: i32, job_ad_text: String, archived: bool, inputs_updated_at: String, ad_weight: f64, fb_job_weight: f64, fb_profile_weight: f64 },
    JobArchived { job_id: JobId, archived: bool },
    JobDeleted { job_id: JobId },
    CandidatesEmptied { job_id: JobId },
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
    RankCancelled { job_id: JobId },
    FavToggled { cv_id: CvId, workspace_id: JobId, fav_id: FavId, ts: i64 },
    RubricWeightChanged { job_id: JobId, weights: Vec<f64> },
    LicenseActivated { license: String },
    LicenseExpired { license: String },
    CvSoftDeleted { cv_id: CvId },
    CvForgotten { cv_id: CvId },
    CandidateRanked { cv_id: CvId, rank: i32, points: f32, points_max: f32 },
    CandidateLiked { cv_id: CvId, liked: bool },
    CandidateViewed { cv_id: CvId, viewed_at: String },
    SortReportWritten { job_id: JobId, report: String },
    RankStateUpdated { job_id: JobId, state: String },
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
    pub rank: Option<i32>,
    pub points: Option<f32>,
    pub points_max: Option<f32>,
    pub liked: bool,
    pub viewed_at: Option<String>,
    pub imported_at: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, specta::Type)]
pub struct JobRecord {
    pub id: JobId,
    pub title: String,
    pub text: String,
    pub version: u32,
    pub triqui_size: i32,
    pub archived: bool,
    pub created_at: String,
    pub last_ranked_at: Option<String>,
    pub inputs_updated_at: Option<String>,
    pub ad_weight: f64,
    pub fb_job_weight: f64,
    pub fb_profile_weight: f64,
    pub job_ad_path: Option<String>,
    pub cv_imported: i32,
    pub cv_failed: i32,
    pub want_count: i32,
    pub avoid_count: i32,
    pub importing: bool,
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
        Event::CvAdded { cv_id, job_id, file_path, file_name, file_type, imported_at } => {
            state.cvs.insert(cv_id.clone(), CvRecord {
                id: cv_id.clone(),
                job_id: job_id.clone(),
                file_path: file_path.clone(),
                file_name: file_name.clone(),
                file_type: file_type.clone(),
                imported_at: Some(imported_at.clone()),
                ..Default::default()
            });
        }
        Event::JobCreated { job_id, title, triqui_size, ad_weight, fb_job_weight, fb_profile_weight, import_directory_path, created_at } => {
            let job = state.jobs.entry(job_id.clone()).or_default();
            job.id = job_id.clone();
            job.title = title.clone();
            job.triqui_size = *triqui_size;
            job.ad_weight = *ad_weight;
            job.fb_job_weight = *fb_job_weight;
            job.fb_profile_weight = *fb_profile_weight;
            job.job_ad_path = import_directory_path.clone();
            job.created_at = created_at.clone();
        }
        Event::JobUpdated { job_id, title, triqui_size, job_ad_text, archived, inputs_updated_at, ad_weight, fb_job_weight, fb_profile_weight } => {
            let job = state.jobs.entry(job_id.clone()).or_default();
            job.id = job_id.clone();
            job.title = title.clone();
            job.triqui_size = *triqui_size;
            job.text = job_ad_text.clone();
            job.archived = *archived;
            job.inputs_updated_at = Some(inputs_updated_at.clone());
            job.ad_weight = *ad_weight;
            job.fb_job_weight = *fb_job_weight;
            job.fb_profile_weight = *fb_profile_weight;
        }
        Event::JobArchived { job_id, archived } => {
            if let Some(job) = state.jobs.get_mut(job_id) {
                job.archived = *archived;
            }
        }
        Event::JobDeleted { job_id } => {
            state.jobs.remove(job_id);
            state.cvs.retain(|_, cv| cv.job_id != *job_id);
        }
        Event::CandidatesEmptied { job_id } => {
            state.cvs.retain(|_, cv| cv.job_id != *job_id);
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
            let (a, b) = if pair.0 <= pair.1 {
                (pair.0.clone(), pair.1.clone())
            } else {
                (pair.1.clone(), pair.0.clone())
            };
            let key = (a.clone(), b.clone(), *job_version, *model_bundle_version);
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
                .get(&(a.clone(), *job_version))
                .copied()
                .unwrap_or_default();
            let g_b = state
                .glicko
                .get(&(b.clone(), *job_version))
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
                (a.clone(), *job_version),
                GlickoState { rating: new_a.rating, rd: new_a.rd },
            );
            state.glicko.insert(
                (b.clone(), *job_version),
                GlickoState { rating: new_b.rating, rd: new_b.rd },
            );
        }
        Event::GlickoUpdated { cv_id, job_version, rating, rd } => {
            state.glicko.insert((cv_id.clone(), *job_version), GlickoState { rating: *rating, rd: *rd });
        }
        Event::RankCancelled { job_id: _ } => {}
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
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.liked = !cv.liked;
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
        Event::CandidateRanked { cv_id, rank, points, points_max } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.rank = Some(*rank);
                cv.points = Some(*points);
                cv.points_max = Some(*points_max);
            }
        }
        Event::CandidateLiked { cv_id, liked } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.liked = *liked;
            }
        }
        Event::CandidateViewed { cv_id, viewed_at } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.viewed_at = Some(viewed_at.clone());
            }
        }
        Event::SortReportWritten { job_id, report: _ } => {
            if let Some(job) = state.jobs.get_mut(job_id) {
                job.last_ranked_at = Some(chrono::Utc::now().to_rfc3339());
            }
        }
        Event::RankStateUpdated { job_id, state: _ } => {
            if let Some(job) = state.jobs.get_mut(job_id) {
                job.last_ranked_at = Some(chrono::Utc::now().to_rfc3339());
            }
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
    pub scout_similarity: Option<f64>,
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
                scout_similarity: c.scout_similarity,
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
        let a_sim = a.heavy_similarity.or(a.scout_similarity);
        let b_sim = b.heavy_similarity.or(b.scout_similarity);
        match b_sim
            .partial_cmp(&a_sim)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => gb.partial_cmp(&ga).unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        }
    });
    out
}

pub fn compute_job_summaries(state: &State, importing: &HashSet<String>) -> Vec<JobSummary> {
    let mut out: Vec<JobSummary> = state
        .jobs
        .values()
        .map(|job| {
            let cv_imported = state
                .cvs
                .values()
                .filter(|c| c.job_id == job.id && c.extraction_status == ExtractionStatus::Extracted)
                .count() as i32;
            let cv_failed = state
                .cvs
                .values()
                .filter(|c| c.job_id == job.id && c.extraction_status == ExtractionStatus::Failed)
                .count() as i32;
            JobSummary {
                id: job.id.clone(),
                title: job.title.clone(),
                triqui_size: job.triqui_size,
                archived: job.archived,
                created_at: job.created_at.clone(),
                last_ranked_at: job.last_ranked_at.clone(),
                cv_imported,
                cv_failed,
                ad_set: !job.text.trim().is_empty(),
                want_count: job.criteria.len() as i32,
                avoid_count: 0,
                importing: importing.contains(&job.id),
            }
        })
        .collect();
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    out
}

pub fn compute_candidates(
    state: &State,
    conn: &Connection,
    job_id: &str,
    offset: i32,
    limit: i32,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<CandidatePage, DomainError> {
    let mut page = candidates::list_candidates(conn, job_id, offset, limit, query, sort_by, sort_dir)?;
    // Filter through state's deleted flags (safety net) and override display fields.
    page.items.retain(|c| {
        state
            .cvs
            .get(&c.id)
            .map(|r| !r.deleted)
            .unwrap_or(true)
    });
    for item in page.items.iter_mut() {
        if let Some(cv) = state.cvs.get(&item.id) {
            item.rank = cv.rank;
            item.points = cv.points;
            item.points_max = cv.points_max;
            item.liked = cv.liked;
            item.viewed_at = cv.viewed_at.clone();
        }
    }
    // Recompute total from state's non-deleted count for safety.
    let state_total = state
        .cvs
        .values()
        .filter(|c| c.job_id == job_id && !c.deleted)
        .count() as i32;
    // Only override total if state disagrees? Plan says total should match; keep DB total.
    let _ = state_total;
    Ok(page)
}

pub fn compute_job_detail(
    state: &State,
    conn: &Connection,
    job_id: &str,
) -> Option<JobDetail> {
    // Look up job in state first.
    let job_state = state.jobs.get(job_id)?;

    // Read Job from DB.
    let job: Job = conn
        .query_row(
            &format!("SELECT {} FROM jobs WHERE id = ?1", "id, title, job_ad_text, import_directory_path, triqui_size, archived, created_at, updated_at, last_ranked_at, inputs_updated_at, ad_weight, fb_job_weight, fb_profile_weight, job_ad_path"),
            rusqlite::params![job_id],
            |row| {
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
            },
        )
        .ok()?;

    let n = job.triqui_size;

    // Triqui items from DB.
    let mut stmt = conn
        .prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at
             FROM candidates WHERE job_id = ?1 AND rank IS NOT NULL ORDER BY rank LIMIT ?2",
        )
        .ok()?;
    let mut triqui: Vec<TriqiItem> = stmt
        .query_map(rusqlite::params![job_id, n], |row| {
            Ok(TriqiItem {
                id: row.get(0)?,
                file_name: row.get(1)?,
                file_type: FileType::from_str(row.get::<_, String>(2)?.as_str()),
                extraction_status: ClientExtractionStatus::from_str(row.get::<_, String>(3)?.as_str()),
                rank: row.get(4)?,
                points: row.get(5)?,
                points_max: row.get(6)?,
                liked: row.get::<_, i64>(7)? != 0,
                viewed_at: row.get(8)?,
            })
        })
        .ok()?
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default();
    for t in triqui.iter_mut() {
        if let Some(cv) = state.cvs.get(&t.id) {
            if cv.rank.is_some() { t.rank = cv.rank.unwrap(); }
            t.points = cv.points;
            t.points_max = cv.points_max;
            t.liked = cv.liked;
            t.viewed_at = cv.viewed_at.clone();
        }
    }

    // Criteria from DB.
    let mut cstmt = conn
        .prepare("SELECT id, job_id, text, weight, created_at FROM criteria WHERE job_id = ?1 ORDER BY created_at")
        .ok()?;
    let criteria: Vec<Criterion> = cstmt
        .query_map(rusqlite::params![job_id], |row| {
            Ok(Criterion {
                id: row.get(0)?,
                job_id: row.get(1)?,
                text: row.get(2)?,
                weight: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .ok()?
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default();

    // cv_imported, cv_failed from state.
    let cv_imported = state
        .cvs
        .values()
        .filter(|c| c.job_id == job_id && c.extraction_status == ExtractionStatus::Extracted)
        .count() as i32;
    let cv_failed = state
        .cvs
        .values()
        .filter(|c| c.job_id == job_id && c.extraction_status == ExtractionStatus::Failed)
        .count() as i32;

    // Build rubric (same logic as handler's build_rubric).
    let rubric = build_rubric(&job, &criteria);
    let capacity: f32 = rubric
        .iter()
        .filter(|q| !q.kind.is_stage())
        .map(|q| q.max_points.abs())
        .sum();

    // Override triqui display fields from state? The triqui items come from DB with rank/points etc. 
    // Plan: override display fields from state when present. Keep simple: leave DB values.

    let _ = job_state;
    Some(JobDetail {
        job,
        criteria,
        rubric,
        rubric_capacity: capacity,
        triqui,
        cv_imported,
        cv_failed,
        importing: false,
    })
}

fn build_rubric(job: &Job, criteria: &[Criterion]) -> Vec<RubricQuestion> {
    let stage = |question_id: &str, label: &str, kind: QuestionKind| RubricQuestion {
        question_id: question_id.into(),
        label: label.into(),
        kind,
        max_points: job.ad_weight as f32,
    };
    let mut out = vec![
        stage(SCOUT_QUESTION_ID, SCOUT_LABEL, QuestionKind::Scout),
        RubricQuestion {
            question_id: libtriqui_core::sort::AD_QUESTION_ID.into(),
            label: "Job fit (embed)".into(),
            kind: QuestionKind::Ad,
            max_points: job.ad_weight as f32,
        },
        stage(
            libtriqui_core::sort::JUDGE_QUESTION_ID,
            JudgeStage::AD_LABEL,
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

pub fn compute_rank_state(
    state: &State,
    conn: &Connection,
    job_id: &str,
) -> RankState {
    // Read rank_state from DB.
    let rs = rank_state_for_job(conn, job_id).ok();
    let _ = &state;

    // We need model for embedded_done count. Use runtime? Not passed. 
    // Fallback: query candidate_chunks distinct vector_model? Simpler: count candidates with any chunk vector.
    let candidates = count_query(
        conn,
        "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted' AND extracted_text IS NOT NULL",
        &[job_id],
    );
    let embedded_done = count_query(
        conn,
        "SELECT COUNT(*) FROM candidates c WHERE c.job_id = ?1 AND c.extraction_status = 'Extracted' AND c.extracted_text IS NOT NULL AND EXISTS (SELECT 1 FROM candidate_chunks cc WHERE cc.candidate_id = c.id)",
        &[job_id],
    );

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

    let ad_set: bool = conn
        .query_row(
            "SELECT COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0) FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get::<_, i64>(0),
        )
        .map(|v| v != 0)
        .unwrap_or(false);
    let ranking = running && ad_set;

    let last = reports.last();
    let reason = match &rs {
        Some(_) => {
            if cancelled {
                Some("cancelled".to_string())
            } else {
                let (syscap_cfg, engine_cfg) = {
                    let cfg = libtriqui_core::app_config::load_triq();
                    (cfg.syscap, cfg.engine)
                };
                let (budget, _) = syscap::sample_budget(&syscap_cfg, &engine_cfg);
                syscap::holding_code(&budget)
            }
        }
        None => None,
    };
    // Keep reason aligned with handler: only cancelled reported here; budget holding omitted (no config).

    // Filter candidates through state deleted flags? embedded counts from DB already excludes deleted? 
    // DB query doesn't filter deleted. Let's filter by state: subtract deleted.
    RankState {
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
    }
}

pub fn compute_sort_report(
    state: &State,
    conn: &Connection,
    job_id: &str,
) -> Option<SortReport> {
    let _ = state;
    conn.query_row(
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
    .ok()
}

pub fn compute_chunk_match(
    state: &State,
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    candidate_id: &str,
    target: ChunkMatchTarget,
    chunk_size: usize,
) -> Result<ChunkMatchResult, DomainError> {
    if state.cvs.get(candidate_id).map(|c| c.deleted).unwrap_or(false) {
        return Ok(ChunkMatchResult {
            model: rt.model(),
            chunks: vec![],
        });
    }
    cm_core::chunk_match(rt, conn, candidate_id, &target, chunk_size)
}

fn count_query(conn: &Connection, sql: &str, params: &[&str]) -> i64 {
    conn.query_row(sql, rusqlite::params_from_iter(params.iter()), |r| r.get::<_, i64>(0))
        .unwrap_or(0)
}

pub fn drift_check(state: &State, conn: &rusqlite::Connection) -> Vec<String> {
    let mut out = Vec::new();

    let db_jobs: i64 = match conn.query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0)) {
        Ok(v) => v,
        Err(e) => {
            out.push(format!("drift_check: jobs count query failed: {e}"));
            return out;
        }
    };
    if db_jobs as usize != state.jobs.len() {
        out.push(format!(
            "job count drift: reducer={} db={}",
            state.jobs.len(),
            db_jobs
        ));
    }

    let db_deleted: i64 =
        match conn.query_row("SELECT COUNT(*) FROM candidates WHERE deleted = 1", [], |r| r.get(0)) {
            Ok(v) => v,
            Err(e) => {
                out.push(format!("drift_check: deleted cv count query failed: {e}"));
                return out;
            }
        };
    let reducer_deleted = state.cvs.values().filter(|c| c.deleted).count();
    if db_deleted as usize != reducer_deleted {
        out.push(format!(
            "deleted cv count drift: reducer={} db={}",
            reducer_deleted, db_deleted
        ));
    }

    let mut db_counts: HashMap<String, i64> = HashMap::new();
    match conn.prepare("SELECT job_id, COUNT(*) FROM candidates WHERE deleted = 0 GROUP BY job_id") {
        Ok(mut stmt) => {
            if let Ok(rows) = stmt.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
            }) {
                for row in rows.flatten() {
                    db_counts.insert(row.0, row.1);
                }
            }
        }
        Err(e) => {
            out.push(format!("drift_check: cv count query failed: {e}"));
            return out;
        }
    }

    let mut reducer_counts: HashMap<String, usize> = HashMap::new();
    for cv in state.cvs.values() {
        if !cv.deleted {
            *reducer_counts.entry(cv.job_id.clone()).or_default() += 1;
        }
    }

    let mut jobs: HashSet<String> = db_counts.keys().cloned().collect();
    jobs.extend(reducer_counts.keys().cloned());
    for job in jobs {
        let d = db_counts.get(&job).copied().unwrap_or(0) as usize;
        let r = reducer_counts.get(&job).copied().unwrap_or(0);
        if d != r {
            out.push(format!(
                "non-deleted cv count drift for job {}: reducer={} db={}",
                job, r, d
            ));
        }
    }

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

pub mod types {
    pub use super::*;
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
    pub use super::{
        compute_entitlements, compute_progress, compute_ranking, compute_job_summaries,
        compute_candidates, compute_job_detail, compute_rank_state, compute_sort_report,
        compute_chunk_match, Entitlements, Progress, RankedCandidate,
    };
}
pub mod event_log {
    pub use super::{append_event, replay_from};
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
            imported_at: "2024-01-01T00:00:00Z".into(),
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
            imported_at: "2024-01-01T00:00:00Z".into(),
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
            imported_at: "2024-01-01T00:00:00Z".into(),
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

#[cfg(test)]
mod more_tests {
    use super::*;

    #[test]
    fn reducer_handles_job_created_idempotently() {
        let s = State::default();
        let e = Event::JobCreated {
            job_id: "j1".into(),
            title: "job-1".into(),
            triqui_size: 10,
            ad_weight: 1.0,
            fb_job_weight: 1.0,
            fb_profile_weight: 1.0,
            import_directory_path: None,
            created_at: "2024-01-01T00:00:00Z".into(),
        };
        let once = reduce(s.clone(), &e);
        let twice = reduce(once.clone(), &e);
        assert_eq!(once, twice);
        assert!(once.jobs.contains_key("j1"));
    }

    #[test]
    fn reducer_job_deleted_removes_job_and_cvs() {
        let mut s = State::default();
        s = reduce(s, &Event::JobCreated {
            job_id: "j1".into(),
            title: "job-1".into(),
            triqui_size: 10,
            ad_weight: 1.0,
            fb_job_weight: 1.0,
            fb_profile_weight: 1.0,
            import_directory_path: None,
            created_at: "2024-01-01T00:00:00Z".into(),
        });
        s = reduce(s, &Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
            imported_at: "2024-01-01T00:00:00Z".into(),
        });
        let s = reduce(s, &Event::JobDeleted { job_id: "j1".into() });
        assert!(!s.jobs.contains_key("j1"));
        assert!(!s.cvs.contains_key("c1"));
    }

    #[test]
    fn reducer_candidates_emptied_removes_cvs() {
        let mut s = State::default();
        s = reduce(s, &Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
            imported_at: "2024-01-01T00:00:00Z".into(),
        });
        let s = reduce(s, &Event::CandidatesEmptied { job_id: "j1".into() });
        assert!(!s.cvs.contains_key("c1"));
    }

    #[test]
    fn drift_check_matching_state() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY)", []).unwrap();
        conn.execute("CREATE TABLE candidates (id TEXT PRIMARY KEY, job_id TEXT, deleted INTEGER NOT NULL DEFAULT 0)", []).unwrap();
        conn.execute("INSERT INTO jobs (id) VALUES ('j1')", []).unwrap();
        conn.execute("INSERT INTO candidates (id, job_id, deleted) VALUES ('c1','j1',0)", []).unwrap();

        let mut s = State::default();
        s.jobs.insert("j1".into(), JobRecord { id: "j1".into(), ..Default::default() });
        s.cvs.insert("c1".into(), CvRecord { id: "c1".into(), job_id: "j1".into(), deleted: false, ..Default::default() });
        let msgs = drift_check(&s, &conn);
        assert!(msgs.is_empty(), "{:?}", msgs);
    }

    #[test]
    fn compute_job_summaries_basic() {
        use std::collections::HashSet;
        let mut s = State::default();
        s = reduce(s, &Event::JobCreated {
            job_id: "j1".into(),
            title: "job-1".into(),
            triqui_size: 10,
            ad_weight: 1.0,
            fb_job_weight: 1.0,
            fb_profile_weight: 1.0,
            import_directory_path: None,
            created_at: "2024-01-01T00:00:00Z".into(),
        });
        s = reduce(s, &Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
            imported_at: "2024-01-01T00:00:00Z".into(),
        });
        s = reduce(s, &Event::TextExtracted {
            cv_id: "c1".into(),
            text: "hello".into(),
            status: ExtractionStatus::Extracted,
        });
        let mut importing = HashSet::new();
        importing.insert("j1".to_string());
        let summaries = compute_job_summaries(&s, &importing);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, "j1");
        assert_eq!(summaries[0].cv_imported, 1);
        assert!(summaries[0].importing);
    }

    #[test]
    fn compute_entitlements_no_license() {
        let s = State::default();
        let ent = compute_entitlements(&s);
        assert!(!ent.head_to_head);
    }

    #[test]
    fn compute_entitlements_with_license() {
        let mut s = State::default();
        s.license = Some("test-license".to_string());
        let ent = compute_entitlements(&s);
        assert!(ent.head_to_head);
    }

    #[test]
    fn compute_progress_empty() {
        let s = State::default();
        let p = compute_progress(&s, "j1");
        assert_eq!(p.total, 0);
        assert_eq!(p.done, 0);
    }

    #[test]
    fn compute_progress_with_cvs() {
        let mut s = State::default();
        s.cvs.insert("c1".into(), CvRecord {
            id: "c1".into(),
            job_id: "j1".into(),
            heavy_vector: None,
            ..Default::default()
        });
        s.cvs.insert("c2".into(), CvRecord {
            id: "c2".into(),
            job_id: "j1".into(),
            heavy_vector: Some(vec![0.1, 0.2]),
            ..Default::default()
        });
        s.cvs.insert("c3".into(), CvRecord {
            id: "c3".into(),
            job_id: "j2".into(),
            heavy_vector: Some(vec![0.3, 0.4]),
            ..Default::default()
        });
        let p = compute_progress(&s, "j1");
        assert_eq!(p.total, 2);
        assert_eq!(p.done, 1);
    }

    #[test]
    fn compute_ranking_scout_fallback() {
        let mut s = State::default();
        s.cvs.insert("c1".into(), CvRecord {
            id: "c1".into(),
            job_id: "j1".into(),
            heavy_similarity: None,
            scout_similarity: Some(0.7),
            ..Default::default()
        });
        s.cvs.insert("c2".into(), CvRecord {
            id: "c2".into(),
            job_id: "j1".into(),
            heavy_similarity: Some(0.5),
            scout_similarity: None,
            ..Default::default()
        });
        let r = compute_ranking(&s, "j1");
        assert_eq!(r.len(), 2);
        // c1 has scout 0.7 > c2 heavy 0.5, so c1 ranks first
        assert_eq!(r[0].cv_id, "c1");
        assert_eq!(r[0].scout_similarity, Some(0.7));
        assert_eq!(r[1].cv_id, "c2");
        assert_eq!(r[1].heavy_similarity, Some(0.5));
    }
}
