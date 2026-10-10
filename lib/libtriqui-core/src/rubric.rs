//! The rubric: what a candidate is scored against, and in which spaces.
//!
//! A [`Rubric`] is the whole ranking input of a job in one value: the job ad,
//! the two learned feedback directions, and a dynamic list of criteria — each
//! with its weight. It carries no grades of its own; grades live in
//! `stage_scores`, one per (candidate, space, question).
//!
//! **The rubric is the only thing a rank depends on.** That is the property
//! `gui/design/app1/cascade-simple.md` buys its simplification with: when an
//! input changes (ad text, a criterion, a like) the change is a mutation of this
//! object plus one [`crate::rescore`] call. Nothing else needs invalidating,
//! because nothing else is state a rank reads.
//!
//! # Spaces
//!
//! Every question holds one vector **per space** (a space is a model slug). The
//! scout has its own space, the heavy embedder its own, and the two are never
//! mixed. A candidate's usable scorecard is the one in the best space it has:
//! the scout's until the heavy embedder reaches it, the heavy embedder's after.
//!
//! That is why vectors live in one `rubric_vectors` table instead of per-column
//! slots: the scout can score the *whole* rubric — ad, criteria and both feedback
//! rows — only if it can hold a vector per question in its own space. Scoring the
//! whole rubric is what makes every candidate's total complete from the first
//! minute, instead of only its ad similarity while the heavy embedder catches up.

use std::collections::HashMap;
use std::sync::Arc;

use rusqlite::Connection;

use crate::embed::{bytes_to_vector, Embedder};
use crate::embeddings;
use crate::runtime::EngineRuntime;
use crate::types::{DomainError, QuestionKind};

/// Question id of the job ad — the ad slot of a total.
pub const AD_Q: &str = "ad";
/// Question id of the learned per-job feedback direction (`fb1`).
pub const FB_JOB_Q: &str = "fb_job";
/// Question id of the learned cross-job feedback direction (`fb2`).
pub const FB_PROFILE_Q: &str = "fb_profile";

/// Any DB-layer error (a `rusqlite::Error` or a plain message) into the domain
/// error, so callers do not have to care which a helper returned.
fn dberr(e: impl std::fmt::Display) -> DomainError {
    DomainError::DbError {
        message: e.to_string(),
    }
}

/// One thing a candidate is scored against.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub label: String,
    pub kind: QuestionKind,
    /// Points this question contributes at similarity 1.0.
    pub max_points: f32,
    /// The text to embed, when the question has one. `None` for the two learned
    /// feedback rows, whose vector comes from `apply_feedback` instead.
    pub text: Option<String>,
}

/// A job's ranking inputs, plus the vectors it has in one or more spaces.
#[derive(Clone, Debug, Default)]
pub struct Rubric {
    pub job_id: String,
    pub ad_weight: f32,
    /// The job ad's text, or `None` when the ad has not been written yet.
    pub ad_text: Option<String>,
    /// Feedback rows then criteria, in a stable order.
    pub questions: Vec<Question>,
    /// `(question_id, space)` → vector.
    vectors: HashMap<(String, String), Vec<f64>>,
}

impl Rubric {
    /// Read a job's rubric from the database. Loads **no** vectors — call
    /// [`Rubric::with_vectors`] for each space the caller intends to score in.
    pub fn load(conn: &Connection, job_id: &str) -> Result<Rubric, DomainError> {
        let (ad_weight, fb_job_weight, fb_profile_weight, ad_text): (f64, f64, f64, Option<String>) =
            conn.query_row(
                "SELECT COALESCE(ad_weight, 1.0), COALESCE(fb_job_weight, 0.0),
                        COALESCE(fb_profile_weight, 0.0), job_ad_text
                 FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .map_err(dberr)?;
        let ad_text = ad_text
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());

        let mut stmt = conn
            .prepare("SELECT id, text, weight FROM criteria WHERE job_id = ?1 ORDER BY created_at")
            .map_err(dberr)?;
        let crits: Vec<(String, String, f64)> = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .map_err(dberr)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(dberr)?;
        drop(stmt);

        // A feedback row is a question only while its weight says it counts —
        // the same rule the rubric builder used to decide whether to emit the row.
        let mut questions = Vec::with_capacity(crits.len() + 2);
        if fb_job_weight > 0.0 {
            questions.push(Question {
                id: FB_JOB_Q.into(),
                label: "Feedback (this job)".into(),
                kind: QuestionKind::FeedbackJob,
                max_points: fb_job_weight as f32,
                text: None,
            });
        }
        if fb_profile_weight > 0.0 {
            questions.push(Question {
                id: FB_PROFILE_Q.into(),
                label: "Feedback (all jobs)".into(),
                kind: QuestionKind::FeedbackProfile,
                max_points: fb_profile_weight as f32,
                text: None,
            });
        }
        for (id, text, weight) in crits {
            questions.push(Question {
                label: text.clone(),
                id,
                kind: QuestionKind::Criterion,
                max_points: weight as f32,
                text: Some(text),
            });
        }

        Ok(Rubric {
            job_id: job_id.to_string(),
            ad_weight: ad_weight as f32,
            ad_text,
            questions,
            vectors: HashMap::new(),
        })
    }

    /// [`Rubric::load`] plus every stored vector of one space.
    pub fn with_vectors(
        conn: &Connection,
        job_id: &str,
        space: &str,
    ) -> Result<Rubric, DomainError> {
        let mut rubric = Rubric::load(conn, job_id)?;
        for (qid, bytes) in crate::db::rubric_vectors(conn, job_id, space).map_err(dberr)? {
            if let Ok(v) = bytes_to_vector(&bytes, None) {
                rubric.vectors.insert(
                    (qid, space.to_string()),
                    v.into_iter().map(|x| x as f64).collect(),
                );
            }
        }
        Ok(rubric)
    }

    /// A question's vector in one space, if stored.
    pub fn vector(&self, question_id: &str, space: &str) -> Option<&[f64]> {
        self.vectors
            .get(&(question_id.to_string(), space.to_string()))
            .map(|v| v.as_slice())
    }

    /// The question ids that can be scored in `space`: the ad (when it has a
    /// vector there) plus every feedback row and criterion that has one.
    pub fn scored_questions(&self, space: &str) -> Vec<&str> {
        std::iter::once(AD_Q)
            .chain(self.questions.iter().map(|q| q.id.as_str()))
            .filter(|id| self.vector(id, space).is_some())
            .collect()
    }

    /// The ad as a question, whether or not an ad exists.
    pub fn ad_question(&self) -> Question {
        Question {
            id: AD_Q.into(),
            label: "Job fit".into(),
            kind: QuestionKind::Ad,
            max_points: self.ad_weight,
            text: self.ad_text.clone(),
        }
    }

    /// `Σ max_points` over the questions a total is made of — the "Total /N"
    /// caption. The ad slot is one question whichever stage backs its *value*, so
    /// it is counted exactly once and the judge never changes this number.
    pub fn capacity(&self) -> f32 {
        self.ad_weight + self.questions.iter().map(|q| q.max_points).sum::<f32>()
    }

    /// The ad weight plus the questions `graded` names — the denominator a single
    /// candidate's total is reported against.
    ///
    /// Narrower than [`Rubric::capacity`] on purpose: a feedback row with no learned
    /// vector is not a question the row was measured on, and a criterion the heavy
    /// embedder has not reached is not one either. Counting them would advertise
    /// capacity the ranking never used.
    pub fn capacity_for(&self, graded: &[&str]) -> f32 {
        self.ad_weight
            + self
                .questions
                .iter()
                .filter(|q| graded.contains(&q.id.as_str()))
                .map(|q| q.max_points)
                .sum::<f32>()
    }

    /// A candidate's total from one space's scorecard.
    ///
    /// `fit` is the ad slot's value (a judge's win-rate when the candidate has
    /// one, otherwise that space's ad cosine); `sims` are the feedback/criterion
    /// similarities in the same space. Both are [0,1], which is what lets a
    /// scout-space total and a heavy-space total sit in one ranking.
    pub fn total(&self, fit: f64, sims: &HashMap<String, f64>) -> f64 {
        let mut total = self.ad_weight as f64 * fit;
        for q in &self.questions {
            if let Some(sim) = sims.get(&q.id) {
                total += q.max_points as f64 * sim;
            }
        }
        total
    }
}

/// Embed and store every rubric question that lacks a vector in `space`.
///
/// Cache-first per text, so a re-run after an edit only embeds what actually
/// changed. The two learned feedback rows are not embedded at all: they are means
/// of candidate vectors, already stored per model, so they are copied across.
///
/// `embedder` lets the caller pass the space's own session (the scout's) instead
/// of having the runtime start its own model. Pass `None` for the heavy space.
///
/// This is the **only** function in the scoring path that calls a model, which is
/// what lets [`crate::rescore`] be a pure projection of the database: a criteria
/// edit costs one criteria embed plus a cosine pass, not a sort.
pub fn ensure_vectors(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    rubric: &Rubric,
    space: &str,
    embedder: Option<&Arc<dyn Embedder>>,
) -> Result<usize, DomainError> {
    let mut embedded = 0usize;

    // The two learned rows: already keyed by model, so a space that has learned
    // them has a vector to copy in.
    let fb_job: Option<Vec<u8>> = conn
        .query_row(
            "SELECT ejobfeedback FROM job_feedback_vectors WHERE job_id = ?1 AND model = ?2",
            rusqlite::params![rubric.job_id, space],
            |r| r.get(0),
        )
        .ok();
    if let Some(v) = fb_job
        .as_deref()
        .and_then(|b| bytes_to_vector(b, None).ok())
        .filter(|v| embeddings::vector_is_sane(v))
    {
        crate::db::put_rubric_vector(conn, &rubric.job_id, FB_JOB_Q, space, &v).map_err(dberr)?;
    }
    let fb_profile: Option<Vec<u8>> = conn
        .query_row(
            "SELECT eprofile FROM profile_vectors WHERE profile_id = 'profile_default' AND model = ?1",
            rusqlite::params![space],
            |r| r.get(0),
        )
        .ok();
    if let Some(v) = fb_profile
        .as_deref()
        .and_then(|b| bytes_to_vector(b, None).ok())
        .filter(|v| embeddings::vector_is_sane(v))
    {
        crate::db::put_rubric_vector(conn, &rubric.job_id, FB_PROFILE_Q, space, &v).map_err(dberr)?;
    }

    // The ad and the criteria: text, so embed them.
    let mut texts: Vec<(String, String)> = Vec::new();
    if let Some(ad) = rubric.ad_text.clone() {
        texts.push((AD_Q.to_string(), ad));
    }
    for q in &rubric.questions {
        if let Some(text) = q.text.clone() {
            texts.push((q.id.clone(), text));
        }
    }

    let tok_cfg = embeddings::tokenizer_for(space);
    let (tok, cfg) = match tok_cfg.as_ref() {
        Some((t, c)) => (Some(t.as_ref()), Some(c)),
        None => (None, None),
    };
    let mut pending: Vec<(String, String)> = Vec::new();
    for (qid, text) in &texts {
        let stored: Option<Vec<u8>> = conn
            .query_row(
                "SELECT vector FROM rubric_vectors
                 WHERE job_id = ?1 AND question_id = ?2 AND model = ?3",
                rusqlite::params![rubric.job_id, qid, space],
                |r| r.get(0),
            )
            .ok();
        if stored.is_none() {
            pending.push((qid.clone(), text.clone()));
        }
    }
    if pending.is_empty() {
        return Ok(0);
    }

    // Cache-only pass first: a rubric whose every text is already in the
    // `embedding_cache` (the common case after a prewarm, or a switch back to a
    // previously used model) is filled without ever touching the engine.
    let still_missing: Vec<(String, String)> = pending
        .into_iter()
        .filter(|(qid, text)| {
            let hash = crate::embed::content_hash(text);
            let cached = embeddings::cached_vector(conn, space, &hash).unwrap_or(None);
            match cached {
                Some(bytes) => {
                    if let Ok(v) = bytes_to_vector(&bytes, None) {
                        if embeddings::vector_is_sane(&v) {
                            let _ = crate::db::put_rubric_vector(
                                conn,
                                &rubric.job_id,
                                qid,
                                space,
                                &v,
                            );
                            embedded += 1;
                            return false;
                        }
                    }
                    true
                }
                None => true,
            }
        })
        .collect();
    if still_missing.is_empty() {
        return Ok(embedded);
    }

    // Genuine misses: start or wait for the engine and embed them.
    let owned = match embedder {
        Some(e) => Arc::clone(e),
        None => {
            rt.wait_until_ready(std::time::Duration::from_secs(60))?;
            rt.embedder()?
        }
    };
    for (qid, text) in &still_missing {
        match embeddings::embed_text_cached(conn, &owned, space, tok, cfg, text) {
            Ok(v) if embeddings::vector_is_sane(&v) => {
                crate::db::put_rubric_vector(conn, &rubric.job_id, qid, space, &v).map_err(dberr)?;
                embedded += 1;
            }
            Ok(_) => log::warn!("rubric: question {qid} in {space} failed its sanity check"),
            Err(e) => log::warn!("rubric: cannot embed question {qid} in {space}: {e}"),
        }
    }
    Ok(embedded)
}

/// [`ensure_vectors`] without ever touching the engine: only texts already in the
/// `embedding_cache` for that space are stored, and anything else is left for the
/// stage that owns the space to embed.
///
/// This is what a request thread uses for the *scout's* space. Loading the scout's
/// session from a request would evict the heavy embedder a running cascade is
/// using, and the scout pass will pick the rest up at its next boundary anyway.
pub fn ensure_vectors_cache_only(
    conn: &Connection,
    rubric: &Rubric,
    space: &str,
) -> Result<usize, DomainError> {
    let mut stored = 0usize;
    let mut texts: Vec<(String, String)> = Vec::new();
    if let Some(ad) = rubric.ad_text.clone() {
        texts.push((AD_Q.to_string(), ad));
    }
    for q in &rubric.questions {
        if let Some(text) = q.text.clone() {
            texts.push((q.id.clone(), text));
        }
    }
    for (qid, text) in texts {
        let present: Option<Vec<u8>> = conn
            .query_row(
                "SELECT vector FROM rubric_vectors
                 WHERE job_id = ?1 AND question_id = ?2 AND model = ?3",
                rusqlite::params![rubric.job_id, qid, space],
                |r| r.get(0),
            )
            .ok();
        if present.is_some() {
            continue;
        }
        let hash = crate::embed::content_hash(&text);
        let Some(bytes) = embeddings::cached_vector(conn, space, &hash).unwrap_or(None) else {
            continue;
        };
        let Ok(v) = bytes_to_vector(&bytes, None) else {
            continue;
        };
        if !embeddings::vector_is_sane(&v) {
            continue;
        }
        crate::db::put_rubric_vector(conn, &rubric.job_id, &qid, space, &v).map_err(dberr)?;
        stored += 1;
    }
    Ok(stored)
}

/// Score candidates against the whole rubric in one space, writing one
/// `stage_scores` row per (candidate, question).
///
/// `ids` restricts the pass to a slice (a wave's K1 candidates); `None` scores
/// everything, which is what the scout does once over the whole corpus.
///
/// A candidate with no chunks in `space` is skipped: there is nothing to compare
/// against, and a space it has not reached is a space it has no grades in. That is
/// the whole definition of "not reached" — no bookkeeping, no pending list.
pub fn score_candidates_in_space(
    conn: &Connection,
    job_id: &str,
    space: &str,
    rubric: &Rubric,
    ids: Option<&[String]>,
) -> Result<usize, DomainError> {
    let candidates = candidate_ids(conn, job_id, ids)?;

    // Only questions with a vector in this space can be scored here.
    let queries: Vec<(&str, Vec<f64>)> = rubric
        .scored_questions(space)
        .into_iter()
        .filter_map(|id| rubric.vector(id, space).map(|v| (id, v.to_vec())))
        .collect();
    if queries.is_empty() {
        return Ok(0);
    }

    let mut rows: Vec<(String, (String, String), f64)> = Vec::new();
    let mut scored = 0usize;
    for cid in &candidates {
        let Some((chunked, _texts)) =
            embeddings::load_candidate_chunks(conn, cid, space).map_err(dberr)?
        else {
            continue;
        };
        for (qid, vector) in &queries {
            let (sim, _idx) = chunked.max_sim_top_k(vector, 1);
            rows.push((
                cid.clone(),
                (space.to_string(), (*qid).to_string()),
                sim,
            ));
        }
        scored += 1;
    }
    crate::db::put_stage_scores(conn, &rows).map_err(dberr)?;
    Ok(scored)
}

/// The candidates a scoring pass covers: the requested slice, or every extracted
/// one.
fn candidate_ids(
    conn: &Connection,
    job_id: &str,
    ids: Option<&[String]>,
) -> Result<Vec<String>, DomainError> {
    match ids {
        Some(list) if !list.is_empty() => Ok(list.to_vec()),
        _ => {
            let mut stmt = conn
                .prepare(
                    "SELECT id FROM candidates
                     WHERE job_id = ?1 AND extraction_status = 'Extracted'
                       AND extracted_text IS NOT NULL
                     ORDER BY imported_at, id",
                )
                .map_err(dberr)?;
            let ids = stmt
                .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
                .map_err(dberr)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(dberr)?;
            Ok(ids)
        }
    }
}

/// A candidate's centroid vector in one space, for a model without chunking.
/// Chunked models read `candidate_chunks` instead; this is the fallback so the
/// judge and the fit score still work for a plain whole-text model.
pub fn centroid_in_space(
    conn: &Connection,
    candidate_id: &str,
    space: &str,
) -> Result<Option<Vec<f64>>, DomainError> {
    if let Some((chunked, _texts)) =
        embeddings::load_candidate_chunks(conn, candidate_id, space).map_err(dberr)?
    {
        return Ok(chunked.centroid());
    }
    let blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vector FROM candidates WHERE id = ?1 AND vector IS NOT NULL AND vector_model = ?2",
            rusqlite::params![candidate_id, space],
            |r| r.get(0),
        )
        .ok();
    let Some(bytes) = blob else { return Ok(None) };
    let v = bytes_to_vector(&bytes, None).map_err(dberr)?;
    let mut f: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
    let norm = (f.iter().map(|x| x * x).sum::<f64>()).sqrt();
    if norm > 0.0 {
        for x in &mut f {
            *x /= norm;
        }
    }
    Ok(Some(f))
}