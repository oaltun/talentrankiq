//! Multi-stage candidate ranking (scout → deep → comparator) and rubric write-back.
//!
//! Stage grades live in `stage_scores` keyed `(candidate, model, question)`; this
//! module turns them into the rubric the GUI reads: `candidate_scores` rows for
//! the job ad, every criterion, and feedback, plus `candidates.points` /
//! `points_max` and the `rank` ordering.
//!
//! The heavy lifting (embedding candidates / questions) is done by the pipeline
//! stages; `sort_job` here is a *blocking* pass that, for each candidate,
//! scores every rubric question against the candidate's best chunk and writes
//! back the full rubric. It is idempotent and safe to run repeatedly.

use rusqlite::Connection;

use crate::runtime::EngineRuntime;

/// One rubric question (the job ad, a criterion, or a feedback signal).
#[derive(Debug, Clone)]
pub struct RubricQuestion {
    pub question_id: String,
    pub label: String,
    /// `Ad` | `Criterion` | `FeedbackJob` | `FeedbackProfile`.
    pub kind: &'static str,
    pub sign: &'static str,
    pub max_points: f64,
}

impl RubricQuestion {
    /// Lower-case storage form of the kind, matching `candidate_scores.kind`.
    pub fn kind_wire(&self) -> &'static str {
        match self.kind {
            "Ad" => "Ad",
            "Criterion" => "Criterion",
            "FeedbackJob" => "FeedbackJob",
            _ => "FeedbackProfile",
        }
    }
}

/// Build the rubric for a job: the job ad plus every criterion plus the two
/// feedback rows. Weights come from the job row and each criterion row.
pub fn rubric_for_job(conn: &Connection, job_id: &str) -> Result<Vec<RubricQuestion>, String> {
    let mut out = Vec::new();

    let (ad_weight, fb_job_weight, fb_profile_weight, ad_set): (f64, f64, f64, bool) = conn
        .query_row(
            "SELECT COALESCE(ad_weight, 10.0), COALESCE(fb_job_weight, 0.0),
                    COALESCE(fb_profile_weight, 0.0),
                    COALESCE(job_ad_text <> '' AND job_ad_text IS NOT NULL, 0)
             FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .map_err(|e| e.to_string())?;

    if ad_set && ad_weight > 0.0 {
        out.push(RubricQuestion {
            question_id: "ad".to_string(),
            label: "Job ad".to_string(),
            kind: "Ad",
            sign: "Positive",
            max_points: ad_weight,
        });
    }

    {
        let mut stmt = conn
            .prepare("SELECT id, text, weight FROM criteria WHERE job_id = ?1 ORDER BY created_at")
            .map_err(|e| e.to_string())?;
        let mut iter = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        while let Some(r) = iter.next().transpose().map_err(|e| e.to_string())? {
            let (id, text, weight) = r;
            if weight > 0.0 {
                out.push(RubricQuestion {
                    question_id: id,
                    label: text,
                    kind: "Criterion",
                    sign: "Positive",
                    max_points: weight,
                });
            }
        }
    }

    if fb_job_weight > 0.0 {
        out.push(RubricQuestion {
            question_id: "fb_job".to_string(),
            label: "Job feedback".to_string(),
            kind: "FeedbackJob",
            sign: "Positive",
            max_points: fb_job_weight,
        });
    }
    if fb_profile_weight > 0.0 {
        out.push(RubricQuestion {
            question_id: "fb_profile".to_string(),
            label: "Profile feedback".to_string(),
            kind: "FeedbackProfile",
            sign: "Positive",
            max_points: fb_profile_weight,
        });
    }

    Ok(out)
}

/// The rubric's total capacity (Σ max_points).
pub fn rubric_capacity(rubric: &[RubricQuestion]) -> f64 {
    rubric.iter().map(|q| q.max_points).sum()
}

/// Resolve the embedding space used for scoring: the runtime embedder's model,
/// else whichever model the stored rubric vectors / candidate centroids use.
fn resolve_model(rt: &EngineRuntime, conn: &Connection, job_id: &str) -> Option<String> {
    if let Ok(e) = rt.embedder() {
        return Some(e.model_id().to_string());
    }
    conn.query_row(
        "SELECT model FROM rubric_vectors WHERE job_id = ?1 LIMIT 1",
        rusqlite::params![job_id],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .or_else(|| {
        conn.query_row(
            "SELECT vector_model FROM candidates WHERE job_id = ?1 AND vector_model IS NOT NULL LIMIT 1",
            rusqlite::params![job_id],
            |r| r.get::<_, String>(0),
        )
        .ok()
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Score a single candidate (centroid vector) against every rubric question
/// vector, returning `(question_id, cosine)` pairs.
pub fn score_candidate(
    rubric_vectors: &[(String, Vec<f32>)],
    candidate_vector: &[f32],
) -> Vec<(String, f64)> {
    rubric_vectors
        .iter()
        .map(|(qid, qvec)| (qid.clone(), cosine(candidate_vector, qvec)))
        .collect()
}

/// Coverage outcome from a write-back operation.
#[derive(Debug, Clone, Default)]
pub struct WriteBackOutcome {
    /// Number of candidates that were actually scored (had centroid + rubric vectors).
    pub ranked: usize,
    /// Total number of extracted candidates in scope.
    pub candidates: usize,
    /// Rubric capacity (sum of max_points).
    pub capacity: f64,
    /// Number of rubric questions.
    pub rubric_questions: usize,
    /// Number of candidates that had no centroid or rubric vectors was empty.
    pub skipped: usize,
}

/// Write the full rubric for a job: `candidate_scores` rows for ad + each
/// criterion + feedback, and `candidates.points` / `points_max` / `rank`.
///
/// `scores` maps `(candidate_id, question_id) -> cosine (0..1 after clamping)`.
/// Returns a `WriteBackOutcome` with coverage info. Only stamps `ranked_version` /
/// `last_ranked_at` / `sort_reports.complete = 1` when `ranked > 0`,
/// `rubric_questions > 0`, and `skipped == 0`.
pub fn write_back(
    conn: &Connection,
    job_id: &str,
    rubric: &[RubricQuestion],
    scores: &std::collections::HashMap<(String, String), f64>,
    chunk_texts: &std::collections::HashMap<(String, String), String>,
) -> Result<WriteBackOutcome, String> {
    let capacity = rubric_capacity(rubric);
    let rubric_questions = rubric.len();

    // Candidates to score (extracted only, not deleted).
    let candidates: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, file_name FROM candidates
                 WHERE job_id = ?1 AND extraction_status = 'Extracted' AND deleted = 0",
            )
            .map_err(|e| e.to_string())?;
        let mut iter = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        let mut v = Vec::new();
        while let Some(r) = iter.next().transpose().map_err(|e| e.to_string())? {
            v.push(r);
        }
        v
    };
    let total_candidates = candidates.len();

    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM candidate_scores WHERE candidate_id IN
            (SELECT id FROM candidates WHERE job_id = ?1)",
        rusqlite::params![job_id],
    )
    .map_err(|e| e.to_string())?;

    // Compute totals, then rank.
    let mut totals: Vec<(String, f64)> = Vec::with_capacity(candidates.len());
    let mut ranked_count = 0;
    {
        let mut insert = tx
            .prepare(
                "INSERT INTO candidate_scores
                   (id, candidate_id, question_id, kind, sign, label, points, max_points, chunk_texts)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(|e| e.to_string())?;
        for (cid, _fname) in &candidates {
            let mut total = 0.0f64;
            let mut candidate_scored = false;
            for q in rubric {
                let cos = scores
                    .get(&(cid.clone(), q.question_id.clone()))
                    .copied()
                    .unwrap_or(0.0);
                let frac = cos.clamp(0.0, 1.0);
                let points = frac * q.max_points;
                total += points;
                if cos != 0.0 {
                    candidate_scored = true;
                }
                let ct = chunk_texts
                    .get(&(cid.clone(), q.question_id.clone()))
                    .cloned();
                insert
                    .execute(rusqlite::params![
                        format!("s_{}_{}", cid, q.question_id),
                        cid,
                        q.question_id,
                        q.kind_wire(),
                        q.sign,
                        q.label,
                        points,
                        q.max_points,
                        ct,
                    ])
                    .map_err(|e| e.to_string())?;
            }
            if candidate_scored {
                ranked_count += 1;
            }
            totals.push((cid.clone(), total));
        }
    }

    // Rank by total DESC, tie-break by file_name then id (deterministic).
    let fname: std::collections::HashMap<&str, &str> =
        candidates.iter().map(|(id, f)| (id.as_str(), f.as_str())).collect();
    totals.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let fa = fname.get(a.0.as_str()).copied().unwrap_or("");
                let fb = fname.get(b.0.as_str()).copied().unwrap_or("");
                fa.cmp(fb).then_with(|| a.0.cmp(&b.0))
            })
    });

    {
        let mut upd = tx
            .prepare("UPDATE candidates SET points = ?1, points_max = ?2, rank = ?3, ranked_at = datetime('now') WHERE id = ?4")
            .map_err(|e| e.to_string())?;
        for (i, (cid, total)) in totals.iter().enumerate() {
            upd.execute(rusqlite::params![total, capacity, (i + 1) as i64, cid])
                .map_err(|e| e.to_string())?;
        }
    }

    let skipped = total_candidates.saturating_sub(ranked_count);
    let full_coverage = ranked_count > 0 && rubric_questions > 0 && skipped == 0;

    if full_coverage {
        tx.execute(
            "UPDATE jobs SET last_ranked_at = datetime('now'), ranked_version = inputs_version, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![job_id],
        )
        .map_err(|e| e.to_string())?;

        tx.execute(
            "INSERT INTO sort_reports (job_id, comparator_mode, comparator_pairs_fresh, comparator_pairs_cached, mode, waves, complete, cancelled)
             VALUES (?1, 'none', 0, 0, 'full', 0, 1, 0)
             ON CONFLICT(job_id) DO UPDATE SET
                comparator_mode = 'none', mode = 'full', complete = 1, cancelled = 0, created_at = datetime('now')",
            rusqlite::params![job_id],
        )
        .map_err(|e| e.to_string())?;
    }

    tx.commit().map_err(|e| e.to_string())?;
    Ok(WriteBackOutcome {
        ranked: ranked_count,
        candidates: total_candidates,
        capacity,
        rubric_questions,
        skipped,
    })
}

/// Full blocking sort: build rubric, resolve vectors, score every candidate,
/// write back the rubric. Returns the write-back outcome with coverage info.
pub fn sort_job(runtime: &EngineRuntime, job_id: &str) -> Result<WriteBackOutcome, String> {
    let conn = runtime.open_conn()?;

    let rubric = rubric_for_job(&conn, job_id)?;
    let model = resolve_model(runtime, &conn, job_id);

    // Load rubric vectors for the resolved space.
    let mut rubric_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    if let Some(model) = &model {
        for (qid, bytes) in crate::db::rubric_vectors(&conn, job_id, model)? {
            rubric_vectors.push((qid, crate::embed::bytes_to_vector(&bytes)));
        }
    }

    // Score candidates against rubric vectors (candidate centroid space).
    let mut scores: std::collections::HashMap<(String, String), f64> =
        std::collections::HashMap::new();
    if let Some(model) = &model {
        for (cid, vec_bytes, _hash) in
            crate::db::candidates_with_centroids(&conn, job_id, model, None)?
        {
            if rubric_vectors.is_empty() {
                break;
            }
            let cvec = crate::embed::bytes_to_vector(&vec_bytes);
            for (qid, cos) in score_candidate(&rubric_vectors, &cvec) {
                scores.insert((cid.clone(), qid), cos);
            }
        }
    }

    write_back(&conn, job_id, &rubric, &scores, &std::collections::HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn rubric_capacity_sums_weights() {
        let rubric = vec![
            RubricQuestion {
                question_id: "ad".into(),
                label: "ad".into(),
                kind: "Ad",
                sign: "Positive",
                max_points: 50.0,
            },
            RubricQuestion {
                question_id: "c1".into(),
                label: "c1".into(),
                kind: "Criterion",
                sign: "Positive",
                max_points: 10.0,
            },
        ];
        assert_eq!(rubric_capacity(&rubric), 60.0);
    }

    #[test]
    fn score_candidate_returns_every_question() {
        let rubric = vec![
            ("ad".to_string(), vec![1.0f32, 0.0]),
            ("c1".to_string(), vec![0.0f32, 1.0]),
        ];
        let cv = vec![1.0f32, 0.0];
        let scores = score_candidate(&rubric, &cv);
        assert_eq!(scores.len(), 2);
        let ad = scores.iter().find(|(q, _)| q == "ad").unwrap().1;
        let c1 = scores.iter().find(|(q, _)| q == "c1").unwrap().1;
        assert!((ad - 1.0).abs() < 1e-9);
        assert!(c1.abs() < 1e-9);
    }
}