//! Candidate read/manage handlers.

use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::DomainError;
use libtriqui_client::types::{
    AdSource, CandidateDetail, CandidateNeighbors, CandidatePage, CandidateSummary, ExtractionStatus,
    FileType, GradeSpace, QuestionKind, RubricScore,
};

use crate::serve::handlers::{domain, ok_json, string_err};
use crate::serve::ServeState;

fn load_candidates(
    conn: &rusqlite::Connection,
    job_id: &str,
    query: Option<&str>,
    sort_by: Option<&str>,
    sort_dir: Option<bool>,
) -> Result<Vec<CandidateSummary>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at, extracted_text
             FROM candidates WHERE job_id = ?1 AND deleted = 0",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<i32>>(4)?,
                r.get::<_, Option<f32>>(5)?,
                r.get::<_, Option<f32>>(6)?,
                r.get::<_, i64>(7)? != 0,
                r.get::<_, Option<String>>(8)?,
                r.get::<_, Option<String>>(9)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut raw: Vec<(
        String,
        String,
        String,
        String,
        Option<i32>,
        Option<f32>,
        Option<f32>,
        bool,
        Option<String>,
        Option<String>,
    )> = Vec::new();
    for r in rows {
        raw.push(r.map_err(|e| e.to_string())?);
    }

    let q = query.map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty());

    let mut summaries = Vec::new();
    for (id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at, extracted_text) in raw {
        if let Some(qq) = &q {
            let name_match = file_name.to_lowercase().contains(qq);
            let text_match = extracted_text
                .as_ref()
                .map(|t| t.to_lowercase().contains(qq))
                .unwrap_or(false);
            if !name_match && !text_match {
                continue;
            }
        }

        let mut score_stmt = conn
            .prepare(
                "SELECT question_id, kind, label, points, max_points FROM candidate_scores WHERE candidate_id = ?1",
            )
            .map_err(|e| e.to_string())?;
        let score_rows = score_stmt
            .query_map(rusqlite::params![id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, f32>(3)?,
                    r.get::<_, f32>(4)?,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut scores = Vec::new();
        for sr in score_rows {
            let (qid, kind, label, pts, max_pts) = sr.map_err(|e| e.to_string())?;
            scores.push(RubricScore {
                question_id: qid,
                label,
                kind: QuestionKind::from_str(&kind),
                points: pts,
                max_points: max_pts,
                chunk_explanations: Vec::new(),
            });
        }

        let grade_space = GradeSpace::from_scores(&scores);
        let ad_source = AdSource::from_scores(&scores);

        summaries.push(CandidateSummary {
            id,
            file_name,
            file_type: FileType::from_str(&file_type),
            extraction_status: ExtractionStatus::from_str(&extraction_status),
            rank,
            points,
            points_max,
            scores,
            grade_space,
            ad_source,
            liked,
            viewed_at,
        });
    }

    let asc = sort_dir.unwrap_or(true);
    let key = sort_by.unwrap_or("__rank");

    summaries.sort_by(|a, b| {
        let rank_cmp = |x: &CandidateSummary, y: &CandidateSummary| match (x.rank, y.rank) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        let points_cmp = |x: &CandidateSummary, y: &CandidateSummary| match (x.points, y.points) {
            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        let opt_str_cmp = |a: Option<&str>, b: Option<&str>| match (a, b) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };

        let mut ord = match key {
            "__rank" | "rank" => rank_cmp(a, b),
            "__name" | "name" | "file_name" => a.file_name.cmp(&b.file_name),
            "__total" | "total" | "points" => points_cmp(a, b),
            "__space" | "space" | "grade_space" => opt_str_cmp(
                a.grade_space.map(|g| g.as_str()),
                b.grade_space.map(|g| g.as_str()),
            ),
            "__source" | "source" | "ad_source" => opt_str_cmp(
                a.ad_source.map(|s| s.as_str()),
                b.ad_source.map(|s| s.as_str()),
            ),
            other => {
                let pa = a
                    .scores
                    .iter()
                    .find(|s| s.question_id == other)
                    .map(|s| s.points);
                let pb = b
                    .scores
                    .iter()
                    .find(|s| s.question_id == other)
                    .map(|s| s.points);
                match (pa, pb) {
                    (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                }
            }
        };

        if !asc {
            ord = ord.reverse();
        }

        if ord == std::cmp::Ordering::Equal {
            ord = a.file_name.cmp(&b.file_name);
            if ord == std::cmp::Ordering::Equal {
                ord = a.id.cmp(&b.id);
            }
        }
        ord
    });

    Ok(summaries)
}

pub fn list_candidates(
    state: &Arc<ServeState>,
    job_id: &str,
    offset: i32,
    limit: i32,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let all = load_candidates(&conn, job_id, query.as_deref(), sort_by.as_deref(), sort_dir)
        .map_err(string_err)?;
    let total = all.len() as i32;
    let start = offset.max(0) as usize;
    let lim = limit.max(0) as usize;
    let end = (start + lim).min(all.len());
    let items = if start < all.len() {
        all[start..end].to_vec()
    } else {
        Vec::new()
    };
    ok_json(CandidatePage { items, total })
}

pub fn get_candidate(state: &Arc<ServeState>, candidate_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM candidates WHERE id = ?1",
            rusqlite::params![candidate_id],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if !exists {
        return Err(domain(&DomainError::NotFound {
            entity: "Candidate".into(),
            id: candidate_id.to_string(),
        }));
    }

    let (id, job_id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at, extracted_text, source_file_path): (
        String,
        String,
        String,
        String,
        String,
        Option<i32>,
        Option<f32>,
        Option<f32>,
        bool,
        Option<String>,
        Option<String>,
        String,
    ) = conn
        .query_row(
            "SELECT id, job_id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at, extracted_text, source_file_path
             FROM candidates WHERE id = ?1",
            rusqlite::params![candidate_id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                    r.get::<_, i64>(8)? != 0,
                    r.get(9)?,
                    r.get(10)?,
                    r.get(11)?,
                ))
            },
        )
        .map_err(|e| string_err(e.to_string()))?;

    let mut score_stmt = conn
        .prepare("SELECT question_id, kind, label, points, max_points FROM candidate_scores WHERE candidate_id = ?1")
        .map_err(|e| string_err(e.to_string()))?;
    let score_rows = score_stmt
        .query_map(rusqlite::params![id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, f32>(3)?,
                r.get::<_, f32>(4)?,
            ))
        })
        .map_err(|e| string_err(e.to_string()))?;

    let mut rubric_scores = Vec::new();
    for sr in score_rows {
        let (qid, kind, label, pts, max_pts) = sr.map_err(|e| string_err(e.to_string()))?;
        rubric_scores.push(RubricScore {
            question_id: qid,
            label,
            kind: QuestionKind::from_str(&kind),
            points: pts,
            max_points: max_pts,
            chunk_explanations: Vec::new(),
        });
    }

    let detail = CandidateDetail {
        id,
        job_id,
        file_name,
        file_type: FileType::from_str(&file_type),
        rank,
        extraction_status: ExtractionStatus::from_str(&extraction_status),
        extracted_text,
        source_file_path: source_file_path.clone(),
        original_available: Path::new(&source_file_path).exists(),
        points,
        points_max,
        rubric_scores,
        liked,
        viewed_at,
    };
    ok_json(detail)
}

pub fn soft_delete_cv(state: &Arc<ServeState>, cv_id: &str) -> Result<Value, ProtocolError> {
    let mut conn = state.db.writer();
    let updated = conn
        .execute(
            "UPDATE candidates SET deleted = 1 WHERE id = ?1",
            rusqlite::params![cv_id],
        )
        .map_err(|e| string_err(e.to_string()))?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Cv".into(),
            id: cv_id.to_string(),
        }));
    }
    ok_json(updated as i32)
}

pub fn forget_cv(state: &Arc<ServeState>, cv_id: &str) -> Result<Value, ProtocolError> {
    let mut conn = state.db.writer();
    let updated = conn
        .execute("DELETE FROM candidates WHERE id = ?1", rusqlite::params![cv_id])
        .map_err(|e| string_err(e.to_string()))?;
    if updated == 0 {
        return Err(domain(&DomainError::NotFound {
            entity: "Cv".into(),
            id: cv_id.to_string(),
        }));
    }
    ok_json(true)
}

pub fn candidate_neighbors(
    state: &Arc<ServeState>,
    job_id: &str,
    candidate_id: &str,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let all = load_candidates(&conn, job_id, query.as_deref(), sort_by.as_deref(), sort_dir)
        .map_err(string_err)?;
    let ids: Vec<String> = all.iter().map(|c| c.id.clone()).collect();
    let (prev_id, next_id) = match ids.iter().position(|id| id == candidate_id) {
        Some(i) => {
            let prev = if i > 0 {
                Some(ids[i - 1].clone())
            } else {
                None
            };
            let next = if i + 1 < ids.len() {
                Some(ids[i + 1].clone())
            } else {
                None
            };
            (prev, next)
        }
        None => (None, None),
    };
    ok_json(CandidateNeighbors {
        prev_id,
        next_id,
    })
}
