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

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
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

/// Score a candidate by maxsim over its chunks: for each rubric question, the
/// maximum cosine across every chunk and the text of the winning chunk.
///
/// Returns `(question_id, cosine, chunk_text)` per question, or an empty vector
/// when the candidate has no chunks (the caller falls back to the centroid).
pub fn score_candidate_chunks(
    rubric_vectors: &[(String, Vec<f32>)],
    chunks: &[(String, Vec<f32>)],
) -> Vec<(String, f64, String)> {
    if chunks.is_empty() {
        return Vec::new();
    }
    rubric_vectors
        .iter()
        .map(|(qid, qvec)| {
            let mut best = 0.0f64;
            let mut best_text = String::new();
            for (text, cvec) in chunks {
                let sim = cosine(cvec, qvec);
                if sim >= best {
                    best = sim;
                    best_text = text.clone();
                }
            }
            (qid.clone(), best, best_text)
        })
        .collect()
}

/// One rubric question's coverage result.
#[derive(Debug, Clone)]
pub struct CoverageScore {
    pub question_id: String,
    /// Mean over the question's rubric chunks of the max cosine to any
    /// candidate chunk.
    pub score: f64,
    /// Candidate chunk text that best matched the most salient rubric chunk.
    pub candidate_chunk_text: String,
    /// The most salient rubric chunk (highest best-cosine), with its range.
    pub rubric_chunk: crate::chunks::ChunkRow,
}

/// Coverage scoring: for each rubric question that has persisted rubric chunks,
/// the mean over those rubric chunks of the maximum cosine to any candidate
/// chunk. Questions without rubric chunks are not included (the caller falls
/// back to whole-question vectors).
///
/// A one-chunk question reduces to exactly today's maxsim score.
pub fn score_candidate_chunks_coverage(
    rubric_chunks_by_question: &std::collections::HashMap<String, Vec<crate::chunks::ChunkRow>>,
    candidate_chunks: &[(String, Vec<f32>)],
) -> Vec<CoverageScore> {
    if candidate_chunks.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (qid, rchunks) in rubric_chunks_by_question {
        if rchunks.is_empty() {
            continue;
        }
        let mut sum = 0.0f64;
        let mut best_salient = f64::NEG_INFINITY;
        let mut best_cv_text = String::new();
        let mut best_rubric: Option<&crate::chunks::ChunkRow> = None;
        for rc in rchunks {
            let mut best = 0.0f64;
            let mut best_text = String::new();
            for (ctext, cvec) in candidate_chunks {
                let sim = cosine(cvec, &rc.vector);
                if sim >= best {
                    best = sim;
                    best_text = ctext.clone();
                }
            }
            sum += best;
            if best > best_salient {
                best_salient = best;
                best_cv_text = best_text;
                best_rubric = Some(rc);
            }
        }
        out.push(CoverageScore {
            question_id: qid.clone(),
            score: sum / rchunks.len() as f64,
            candidate_chunk_text: best_cv_text,
            rubric_chunk: best_rubric.expect("non-empty rubric chunks yield a salient chunk").clone(),
        });
    }
    out
}

/// One candidate's full rubric score, split into the pieces `write_back` needs.
#[derive(Debug, Clone, Default)]
pub struct CandidateRubricScore {
    /// `question_id -> cosine`.
    pub scores: std::collections::HashMap<String, f64>,
    /// `question_id -> winning candidate chunk text`.
    pub chunk_texts: std::collections::HashMap<String, String>,
    /// `question_id -> (matched rubric chunk text, char_start, char_end)`.
    pub rubric_chunks: std::collections::HashMap<String, (String, usize, usize)>,
}

/// Score one candidate against the whole rubric.
///
/// Questions with persisted rubric chunks are scored by coverage (mean over
/// rubric chunks of max cosine to any candidate chunk); questions without are
/// scored by maxsim over the candidate chunks against the whole-question vector
/// (`score_candidate_chunks`). When the candidate has no chunks, the centroid
/// (`centroid`) is used as a single chunk so coverage still works.
pub fn score_candidate_against_rubric(
    rubric_vectors: &[(String, Vec<f32>)],
    rubric_chunks_by_question: &std::collections::HashMap<String, Vec<crate::chunks::ChunkRow>>,
    candidate_chunks: &[(String, Vec<f32>)],
    centroid: Option<&[f32]>,
    whole_text: &str,
) -> CandidateRubricScore {
    let mut out = CandidateRubricScore::default();

    let whole_rubric_vectors: Vec<(String, Vec<f32>)> = rubric_vectors
        .iter()
        .filter(|(qid, _)| !rubric_chunks_by_question.contains_key(qid))
        .cloned()
        .collect();

    // Treat the centroid as a single whole-text chunk when no chunks exist, so
    // the coverage path is shared by both branches.
    let single: Vec<(String, Vec<f32>)>;
    let chunks: &[(String, Vec<f32>)] = if !candidate_chunks.is_empty() {
        candidate_chunks
    } else if let Some(cvec) = centroid {
        single = vec![(whole_text.to_string(), cvec.to_vec())];
        &single
    } else {
        return out;
    };

    for cov in score_candidate_chunks_coverage(rubric_chunks_by_question, chunks) {
        out.scores.insert(cov.question_id.clone(), cov.score);
        out.chunk_texts
            .insert(cov.question_id.clone(), cov.candidate_chunk_text);
        out.rubric_chunks.insert(
            cov.question_id,
            (
                cov.rubric_chunk.chunk_text,
                cov.rubric_chunk.char_start,
                cov.rubric_chunk.char_end,
            ),
        );
    }

    if !whole_rubric_vectors.is_empty() {
        if candidate_chunks.is_empty() {
            // Single-chunk (centroid) path: maxsim over the one chunk.
            for (qid, cos) in score_candidate(&whole_rubric_vectors, chunks[0].1.as_slice()) {
                out.scores.insert(qid.clone(), cos);
                out.chunk_texts.insert(qid, whole_text.to_string());
            }
        } else {
            for (qid, cos, text) in score_candidate_chunks(&whole_rubric_vectors, chunks) {
                out.scores.insert(qid.clone(), cos);
                out.chunk_texts.insert(qid, text);
            }
        }
    }

    out
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
///
/// `scored_inputs_version` is the `inputs_version` the caller actually scored
/// against. It is stamped verbatim as `ranked_version`, so an input bump that
/// lands while a run is finishing leaves the run marked stale instead of
/// falsely converged.
pub fn write_back(
    conn: &Connection,
    job_id: &str,
    rubric: &[RubricQuestion],
    scores: &std::collections::HashMap<(String, String), f64>,
    chunk_texts: &std::collections::HashMap<(String, String), String>,
    rubric_chunks: &std::collections::HashMap<(String, String), (String, usize, usize)>,
    scored_inputs_version: i64,
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
                   (id, candidate_id, question_id, kind, sign, label, points, max_points, chunk_texts,
                    rubric_chunk_text, rubric_char_start, rubric_char_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
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
                let rc = rubric_chunks.get(&(cid.clone(), q.question_id.clone()));
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
                        rc.map(|r| r.0.clone()),
                        rc.map(|r| r.1 as i64),
                        rc.map(|r| r.2 as i64),
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
            "UPDATE jobs SET last_ranked_at = datetime('now'), ranked_version = ?2, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![job_id, scored_inputs_version],
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

    // The blocking sort scores the current inputs, so its scored version is
    // whatever the column reads at entry.
    let scored_inputs_version: i64 = conn
        .query_row(
            "SELECT inputs_version FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;

    let rubric = rubric_for_job(&conn, job_id)?;
    let model = resolve_model(runtime, &conn, job_id);

    // Load rubric vectors for the resolved space.
    let mut rubric_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    if let Some(model) = &model {
        for (qid, bytes) in crate::db::rubric_vectors(&conn, job_id, model)? {
            rubric_vectors.push((qid, crate::embed::bytes_to_vector(&bytes)));
        }
    }

    // Load persisted rubric chunks (coverage scoring) for the resolved space.
    let mut rubric_chunks_by_question: std::collections::HashMap<String, Vec<crate::chunks::ChunkRow>> =
        std::collections::HashMap::new();
    if let Some(model) = &model {
        for q in &rubric {
            let rcs = crate::chunks::rubric_chunks_for(&conn, job_id, &q.question_id, model)
                .unwrap_or_default();
            if !rcs.is_empty() {
                rubric_chunks_by_question.insert(q.question_id.clone(), rcs);
            }
        }
    }

    // Score candidates against rubric vectors. Coverage scoring is used for any
    // question with persisted rubric chunks; the rest use the whole-question
    // vector. Candidates with persisted chunks score by maxsim over those
    // chunks (the winning chunk text becomes the evidence); candidates without
    // chunks fall back to the centroid and use their whole extracted text.
    let mut scores: std::collections::HashMap<(String, String), f64> =
        std::collections::HashMap::new();
    let mut chunk_texts: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    let mut rubric_chunk_map: std::collections::HashMap<
        (String, String),
        (String, usize, usize),
    > = std::collections::HashMap::new();
    if let Some(model) = &model {
        let extracted = extracted_texts(&conn, job_id)?;
        if rubric_vectors.is_empty() && rubric_chunks_by_question.is_empty() {
            // Nothing to score against; still write back the (zeroed) rubric.
        } else {
            for (cid, vec_bytes, _hash) in
                crate::db::candidates_with_centroids(&conn, job_id, model, None)?
            {
                let chunks: Vec<(String, Vec<f32>)> =
                    crate::chunks::candidate_chunks_for(&conn, &cid, model)?
                        .into_iter()
                        .map(|c| (c.chunk_text, c.vector))
                        .collect();
                let centroid = crate::embed::bytes_to_vector(&vec_bytes);
                let whole = extracted.get(&cid).cloned().unwrap_or_default();
                let scored = score_candidate_against_rubric(
                    &rubric_vectors,
                    &rubric_chunks_by_question,
                    &chunks,
                    Some(&centroid),
                    &whole,
                );
                for (qid, s) in scored.scores {
                    scores.insert((cid.clone(), qid), s);
                }
                for (qid, t) in scored.chunk_texts {
                    chunk_texts.insert((cid.clone(), qid), t);
                }
                for (qid, r) in scored.rubric_chunks {
                    rubric_chunk_map.insert((cid.clone(), qid), r);
                }
            }
        }
    }

    write_back(
        &conn,
        job_id,
        &rubric,
        &scores,
        &chunk_texts,
        &rubric_chunk_map,
        scored_inputs_version,
    )
}

/// Extracted text of every extracted candidate of a job, keyed by candidate id.
fn extracted_texts(
    conn: &Connection,
    job_id: &str,
) -> Result<std::collections::HashMap<String, String>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, COALESCE(extracted_text, '') FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted' AND deleted = 0",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows.into_iter().collect())
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

    #[test]
    fn score_candidate_chunks_takes_max_and_winning_text() {
        let rubric = vec![("q1".to_string(), vec![1.0f32, 0.0])];
        let chunks = vec![
            ("a".to_string(), vec![0.0f32, 1.0]),
            ("b".to_string(), vec![0.6f32, 0.8]),
            ("c".to_string(), vec![1.0f32, 0.0]),
        ];
        let scored = score_candidate_chunks(&rubric, &chunks);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].0, "q1");
        assert!((scored[0].1 - 1.0).abs() < 1e-9);
        assert_eq!(scored[0].2, "c");
    }

    #[test]
    fn score_candidate_chunks_empty_is_empty() {
        let rubric = vec![("q1".to_string(), vec![1.0f32, 0.0])];
        assert!(score_candidate_chunks(&rubric, &[]).is_empty());
    }

    #[test]
    fn score_candidate_chunks_coverage_one_chunk_equals_maxsim() {
        let mut rq: std::collections::HashMap<String, Vec<crate::chunks::ChunkRow>> =
            std::collections::HashMap::new();
        rq.insert(
            "q1".to_string(),
            vec![crate::chunks::ChunkRow {
                chunk_index: 0,
                chunk_text: "r".to_string(),
                char_start: 0,
                char_end: 1,
                vector: vec![1.0f32, 0.0],
            }],
        );
        let chunks = vec![
            ("a".to_string(), vec![0.0f32, 1.0]),
            ("b".to_string(), vec![1.0f32, 0.0]),
        ];
        let scored = score_candidate_chunks_coverage(&rq, &chunks);
        assert_eq!(scored.len(), 1);
        assert!((scored[0].score - 1.0).abs() < 1e-9);
        assert_eq!(scored[0].candidate_chunk_text, "b");
        assert_eq!(scored[0].rubric_chunk.chunk_text, "r");
    }

    fn rc(index: i64, text: &str, start: usize, end: usize, vector: Vec<f32>) -> crate::chunks::ChunkRow {
        crate::chunks::ChunkRow {
            chunk_index: index,
            chunk_text: text.to_string(),
            char_start: start,
            char_end: end,
            vector,
        }
    }

    // 3.2 / 3.3 — a question with rubric chunks is scored by coverage (mean
    // over rubric chunks of best candidate cosine) and the matched rubric chunk
    // is reported alongside the winning candidate chunk.
    #[test]
    fn score_candidate_against_rubric_uses_coverage_and_reports_matched_chunk() {
        let rubric_vectors = vec![("ad".to_string(), vec![1.0f32, 0.0])];
        let mut rq = std::collections::HashMap::new();
        rq.insert(
            "ad".to_string(),
            vec![
                rc(0, "req one", 0, 7, vec![1.0f32, 0.0]),
                rc(1, "req two", 8, 15, vec![0.0f32, 1.0]),
            ],
        );
        let candidate_chunks = vec![
            ("cv one".to_string(), vec![1.0f32, 0.0]),
            ("cv two".to_string(), vec![0.0f32, 1.0]),
        ];

        let scored = score_candidate_against_rubric(
            &rubric_vectors,
            &rq,
            &candidate_chunks,
            None,
            "whole cv text",
        );

        // Each rubric chunk has a perfect candidate match → mean is 1.0.
        assert!((scored.scores["ad"] - 1.0).abs() < 1e-9);
        // The most salient rubric chunk is "req one" (first max).
        assert_eq!(scored.rubric_chunks["ad"], ("req one".to_string(), 0, 7));
        assert_eq!(scored.chunk_texts["ad"], "cv one");
    }

    // 3.2 — a question without rubric chunks falls back to maxsim over the
    // whole-question vector.
    #[test]
    fn score_candidate_against_rubric_falls_back_without_rubric_chunks() {
        let rubric_vectors = vec![
            ("ad".to_string(), vec![1.0f32, 0.0]),
            ("c1".to_string(), vec![0.0f32, 1.0]),
        ];
        let candidate_chunks = vec![
            ("a".to_string(), vec![0.0f32, 1.0]),
            ("b".to_string(), vec![1.0f32, 0.0]),
        ];

        let scored = score_candidate_against_rubric(
            &rubric_vectors,
            &std::collections::HashMap::new(),
            &candidate_chunks,
            None,
            "whole cv text",
        );

        assert!((scored.scores["ad"] - 1.0).abs() < 1e-9);
        assert!((scored.scores["c1"] - 1.0).abs() < 1e-9);
        assert_eq!(scored.chunk_texts["ad"], "b");
        assert_eq!(scored.chunk_texts["c1"], "a");
        assert!(scored.rubric_chunks.is_empty());
    }

    /// A run scores version N; an input bump to N+1 lands before write-back.
    /// The completion marker must stay at N — stamping the current column value
    /// would report a version this run never scored and hide the staleness.
    #[test]
    fn write_back_stamps_the_scored_version_not_a_later_bump() {
        use crate::db::open_single_connection_at;

        let mut path = std::env::temp_dir();
        path.push(format!("triq-sort-version-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&path);
        let conn = open_single_connection_at(&path).expect("open db");
        conn.execute(
            "INSERT INTO jobs (id, title, job_ad_text, ad_weight) VALUES ('job1', 'j', 'ad', 10.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates
               (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES ('c1', 'job1', 'p/c1.pdf', 'c1.pdf', 'Pdf', 'Extracted')",
            [],
        )
        .unwrap();

        let rubric = vec![RubricQuestion {
            question_id: "ad".into(),
            label: "ad".into(),
            kind: "Ad",
            sign: "Positive",
            max_points: 10.0,
        }];
        let mut scores = std::collections::HashMap::new();
        scores.insert(("c1".to_string(), "ad".to_string()), 1.0f64);

        // The run captured inputs_version = 0 at start.
        let scored_version: i64 = conn
            .query_row("SELECT inputs_version FROM jobs WHERE id = 'job1'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(scored_version, 0);

        // An input lands mid-run: bump the live column to 1.
        conn.execute("UPDATE jobs SET inputs_version = inputs_version + 1 WHERE id = 'job1'", [])
            .unwrap();

        let outcome = write_back(
            &conn,
            "job1",
            &rubric,
            &scores,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            scored_version,
        )
        .expect("write_back");
        assert_eq!(outcome.ranked, 1);

        let (inputs_version, ranked_version): (i64, Option<i64>) = conn
            .query_row(
                "SELECT inputs_version, ranked_version FROM jobs WHERE id = 'job1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(inputs_version, 1, "the live input version did bump");
        assert_eq!(
            ranked_version,
            Some(0),
            "ranked_version must stay at the version actually scored"
        );
    }

    /// 1.1 — a job with no rubric vectors must leave `ranked_version` NULL while
    /// still writing partial (zeroed) rows, so the run is never reported as
    /// converged and the GUI can show the partial rubric instead of empty rows.
    #[test]
    fn write_back_no_rubric_vectors_leaves_ranked_version_null_but_writes_rows() {
        use crate::db::open_single_connection_at;

        let mut path = std::env::temp_dir();
        path.push(format!("triq-sort-norubric-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&path);
        let conn = open_single_connection_at(&path).expect("open db");
        conn.execute(
            "INSERT INTO jobs (id, title, job_ad_text, ad_weight) VALUES ('job1', 'j', 'ad', 10.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates
                (id, job_id, source_file_path, file_name, file_type, extraction_status)
              VALUES ('c1', 'job1', 'p/c1.pdf', 'c1.pdf', 'Pdf', 'Extracted')",
            [],
        )
        .unwrap();

        let rubric = vec![RubricQuestion {
            question_id: "ad".into(),
            label: "ad".into(),
            kind: "Ad",
            sign: "Positive",
            max_points: 10.0,
        }];
        // No scores at all — simulating a run with no rubric vectors.
        let outcome = write_back(
            &conn,
            "job1",
            &rubric,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            0,
        )
        .expect("write_back");

        assert_eq!(outcome.ranked, 0, "nothing was scored");
        assert_eq!(outcome.candidates, 1);
        assert_eq!(outcome.rubric_questions, 1);
        assert_eq!(outcome.skipped, 1);

        // ranked_version must stay NULL — the run scored nothing.
        let ranked_version: Option<i64> = conn
            .query_row(
                "SELECT ranked_version FROM jobs WHERE id = 'job1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ranked_version, None, "ranked_version must be NULL when nothing was scored");

        // Partial rows are still written (zeroed) so the UI shows the rubric.
        let row_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidate_scores WHERE candidate_id = 'c1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(row_count, 1, "partial (zeroed) rows must still be written");

        // sort_reports must not be marked complete.
        let complete: Option<i64> = conn
            .query_row(
                "SELECT complete FROM sort_reports WHERE job_id = 'job1'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(None);
        assert!(complete != Some(1), "sort_reports must not be marked complete");
    }
}