use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    CvAdded { cv_id: String, job_id: String, file_path: String, file_name: String, file_type: String, imported_at: String },
    TextExtracted { cv_id: String, text: String, status: ExtractionStatus },
    ScoutCompleted { cv_id: String, vector: Vec<f32>, model_bundle_version: u32 },
    HeavyCompleted { cv_id: String, vector: Vec<f32>, model_bundle_version: u32 },
    HeavyMatched { cv_id: String, job_version: u32, similarity: f64 },
    ComparisonCompleted { pair: (String, String), job_version: u32, model_bundle_version: u32, result: f64 },
    GlickoUpdated { cv_id: String, job_version: u32, rating: f64, rd: f64 },
    LicenseActivated,
    LicenseExpired,
    RankCancelled { job_id: String },
    FavToggled { fav_id: String, cv_id: String, workspace_id: String, ts: i64 },
    JobCreated { job_id: String, title: String, triqui_size: i32, ad_weight: f64, fb_job_weight: f64, fb_profile_weight: f64, import_directory_path: String, created_at: String },
    JobUpdated { job_id: String, title: String, triqui_size: i32, job_ad_text: String, archived: bool, inputs_updated_at: String, ad_weight: f64, fb_job_weight: f64, fb_profile_weight: f64 },
    JobArchived { job_id: String, archived: bool },
    CandidatesEmptied { job_id: String },
    JobDeleted { job_id: String },
    CriterionAdded { criterion_id: String, job_id: String, text: String, weight: f64 },
    CriterionUpdated { criterion_id: String, text: String, weight: f64 },
    CriterionRemoved { criterion_id: String },
    RubricWeightChanged { job_id: String, weights: Vec<f64> },
    JobAdUpdated { job_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionStatus {
    Queued,
    Extracting,
    Extracted,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvRecord {
    pub id: String,
    pub job_id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_type: String,
    pub extraction_status: ExtractionStatus,
    pub extracted_text: Option<String>,
    pub scout_vector: Option<Vec<f32>>,
    pub heavy_vector: Option<Vec<f32>>,
    pub heavy_similarity: Option<f64>,
    pub text: Option<String>,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub title: String,
    pub text: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub cvs: HashMap<String, CvRecord>,
    pub jobs: HashMap<String, Job>,
    pub glicko: HashMap<(String, u32), libcv_comparator_order::GlickoState>,
    pub comparisons: Vec<(String, String, u32, u32, f64)>,
    pub license: Option<bool>,
}

impl State {
    pub fn new() -> Self {
        Self {
            cvs: HashMap::new(),
            jobs: HashMap::new(),
            glicko: HashMap::new(),
            comparisons: Vec::new(),
            license: None,
        }
    }
}

impl Default for State {
    fn default() -> Self { Self::new() }
}

pub fn reduce(mut state: State, event: &Event) -> State {
    match event {
        Event::CvAdded { cv_id, job_id, file_path, file_name, file_type, .. } => {
            state.cvs.insert(cv_id.clone(), CvRecord {
                id: cv_id.clone(),
                job_id: job_id.clone(),
                file_path: file_path.clone(),
                file_name: file_name.clone(),
                file_type: file_type.clone(),
                extraction_status: ExtractionStatus::Extracted,
                extracted_text: None,
                scout_vector: None,
                heavy_vector: None,
                heavy_similarity: None,
                text: None,
                deleted: false,
            });
        }
        Event::TextExtracted { cv_id, text, status } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.extracted_text = Some(text.clone());
                cv.extraction_status = status.clone();
            }
        }
        Event::ScoutCompleted { cv_id, vector, .. } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.scout_vector = Some(vector.clone());
            }
        }
        Event::HeavyCompleted { cv_id, vector, .. } => {
            if let Some(cv) = state.cvs.get_mut(cv_id) {
                cv.heavy_vector = Some(vector.clone());
            }
        }
        _ => {}
    }
    state
}

pub mod projections {
    use super::*;

    pub fn compute_job_summaries(_cv_state: &State, _importing: &std::collections::HashSet<String>) -> Vec<libtriqui_client::types::JobSummary> {
        Vec::new()
    }

    pub fn compute_job_detail(_cv_state: &State, conn: &rusqlite::Connection, job_id: &str) -> Option<libtriqui_client::types::JobDetail> {
        use libtriqui_client::types::{Criterion, FileType, Job, JobDetail, TriqiItem};
        use libtriqui_client::types::ExtractionStatus;

        // Load job row.
        let job: Job = conn
            .query_row(
                "SELECT id, title, job_ad_text, import_directory_path, triqui_size, archived, created_at, updated_at, last_ranked_at, inputs_updated_at, ad_weight, fb_job_weight, fb_profile_weight, job_ad_path FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| {
                    Ok(Job {
                        id: r.get(0)?,
                        title: r.get(1)?,
                        job_ad_text: r.get(2)?,
                        import_directory_path: r.get(3)?,
                        triqui_size: r.get(4)?,
                        archived: r.get::<_, i64>(5)? != 0,
                        created_at: r.get(6)?,
                        updated_at: r.get(7)?,
                        last_ranked_at: r.get(8)?,
                        inputs_updated_at: r.get(9)?,
                        ad_weight: r.get(10)?,
                        fb_job_weight: r.get(11)?,
                        fb_profile_weight: r.get(12)?,
                        job_ad_path: r.get(13)?,
                    })
                },
            )
            .ok()?;

        // Criteria.
        let mut criteria: Vec<Criterion> = Vec::new();
        if let Ok(mut stmt) = conn.prepare("SELECT id, job_id, text, weight, created_at FROM criteria WHERE job_id = ?1 ORDER BY created_at") {
            if let Ok(rows) = stmt.query_map(rusqlite::params![job_id], |r| {
                Ok(Criterion {
                    id: r.get(0)?,
                    job_id: r.get(1)?,
                    text: r.get(2)?,
                    weight: r.get(3)?,
                    created_at: r.get(4)?,
                })
            }) {
                for row in rows.flatten() {
                    criteria.push(row);
                }
            }
        }

        // Rubric (mirrors sort::rubric_for_job).
        let mut rubric: Vec<libtriqui_client::types::RubricQuestion> = Vec::new();
        let ad_weight = job.ad_weight;
        let fb_job_weight = job.fb_job_weight;
        let fb_profile_weight = job.fb_profile_weight;
        let ad_set = job
            .job_ad_text
            .as_deref()
            .map(|t| !t.trim().is_empty())
            .unwrap_or(false);
        if ad_set && ad_weight > 0.0 {
            rubric.push(libtriqui_client::types::RubricQuestion {
                question_id: "ad".to_string(),
                label: "Job ad".to_string(),
                kind: libtriqui_client::types::QuestionKind::Ad,
                max_points: ad_weight as f32,
            });
        }
        if let Ok(mut stmt) = conn.prepare("SELECT id, text, weight FROM criteria WHERE job_id = ?1 ORDER BY created_at") {
            if let Ok(rows) = stmt.query_map(rusqlite::params![job_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, f64>(2)?))
            }) {
                for row in rows.flatten() {
                    let (id, text, weight) = row;
                    if weight > 0.0 {
                        rubric.push(libtriqui_client::types::RubricQuestion {
                            question_id: id,
                            label: text,
                            kind: libtriqui_client::types::QuestionKind::Criterion,
                            max_points: weight as f32,
                        });
                    }
                }
            }
        }
        if fb_job_weight > 0.0 {
            rubric.push(libtriqui_client::types::RubricQuestion {
                question_id: "fb_job".to_string(),
                label: "Job feedback".to_string(),
                kind: libtriqui_client::types::QuestionKind::FeedbackJob,
                max_points: fb_job_weight as f32,
            });
        }
        if fb_profile_weight > 0.0 {
            rubric.push(libtriqui_client::types::RubricQuestion {
                question_id: "fb_profile".to_string(),
                label: "Profile feedback".to_string(),
                kind: libtriqui_client::types::QuestionKind::FeedbackProfile,
                max_points: fb_profile_weight as f32,
            });
        }
        let rubric_capacity: f32 = rubric.iter().map(|q| q.max_points).sum();

        // Triqui: ranked candidates.
        let mut triqui: Vec<TriqiItem> = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at
             FROM candidates WHERE job_id = ?1 AND rank IS NOT NULL ORDER BY rank",
        ) {
            if let Ok(rows) = stmt.query_map(rusqlite::params![job_id], |r| {
                Ok(TriqiItem {
                    id: r.get(0)?,
                    file_name: r.get(1)?,
                    file_type: FileType::from_str(&r.get::<_, String>(2)?),
                    extraction_status: ExtractionStatus::from_str(&r.get::<_, String>(3)?),
                    rank: r.get::<_, i32>(4)?,
                    points: r.get(5)?,
                    points_max: r.get(6)?,
                    liked: r.get::<_, i64>(7)? != 0,
                    viewed_at: r.get(8)?,
                })
            }) {
                for row in rows.flatten() {
                    triqui.push(row);
                }
            }
        }

        let cv_imported: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0) as i32;
        let cv_failed: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Failed'",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0) as i32;

        Some(JobDetail {
            job,
            criteria,
            rubric,
            rubric_capacity,
            triqui,
            cv_imported,
            cv_failed,
            importing: false,
        })
    }

    pub fn compute_sort_report(_cv_state: &State, _conn: &rusqlite::Connection, _job_id: &str) -> Option<libtriqui_client::types::SortReport> {
        None
    }
}

pub fn compute_entitlements(_state: &State) -> libcv_entitlements::Entitlements {
    libcv_entitlements::Entitlements { head_to_head: false }
}
