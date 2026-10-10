//! Rescore: the sheet, derived.
//!
//! This is the projection. It reads `stage_scores` (what each stage thought of
//! each candidate, per space) and `rubric_vectors` (what everything is measured
//! against) and writes the two things the gui reads: `candidates.points` and the
//! `candidate_scores` breakdown rows.
//!
//! It makes **no model calls** and takes **no budgets**. That is the point
//! (`gui/design/app1/cascade-simple.md`): budgeting lives in the stages
//! (`scout::run_scout`, `cascade::run_wave`, `judge::run_pool`), projection lives
//! here, and the two never meet. So an input change — a new criterion, an edited
//! ad, a like — is one call to this function and the sheet is correct
//! immediately, whether or not a cascade is running.
//!
//! # What it computes
//!
//! ```text
//! space(c) = heavy if c has heavy-space grades else scout
//! fit(c)   = judge win-rate if judged, else the ad grade in space(c)
//! total(c) = ad_weight · fit(c) + Σ q.weight · sim(c, q, space(c))
//! ```
//!
//! Every candidate has a total from the first scout pass, because the scout
//! scores the whole rubric rather than only the ad. There are no "pending" rows
//! and no counted/display-only cut: `points` is **computed from `stage_scores`**,
//! not summed out of the display rows, so which row happens to carry the ad slot
//! can never change what the sheet ranks by. The rows exist to explain the
//! number, not to produce it.
//!
//! Exactly one row carries the ad slot per candidate, and the row id says which:
//! `ad_judge` when judged, `ad` when the heavy embedder has graded it, `scout`
//! while it has not. `rubric_capacity` counts the ad slot once regardless, since
//! it is `ad_weight` plus the rubric's own questions.

use std::collections::HashMap;
use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension};

use crate::db::db_err;
use crate::rubric::{self, Rubric, AD_Q};
use crate::runtime::EngineRuntime;
use crate::scout;
use crate::types::{
    ChunkExplanation, DomainError, QuestionKind, RubricScore, TriqiItem, SortReport,
};

/// Question id of the judge's win-rate row — the judge's own, so the heavy
/// embedder's ad cosine survives as a separate column.
pub const JUDGE_Q: &str = "ad_judge";

/// What one rescore pass did. The reported-behaviour contract the gui and the
/// tests assert on (never instrumentation).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RescoreReport {
    /// Candidates whose scorecard was recomputed.
    pub scored: i64,
    /// Candidates with no grade at all — no ad yet and no criteria, so there is
    /// nothing to measure them against.
    pub ungraded: i64,
    /// Candidates whose total is backed by a judge's win-rate.
    pub judged: i64,
    /// Candidates still graded only in the scout space.
    pub scout_only: i64,
    /// The job's rubric capacity — the "Total /N" denominator.
    pub capacity: f64,
}

/// Recompute every candidate's scorecard and re-rank the job.
pub fn rescore(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<RescoreReport, DomainError> {
    rescore_ids(rt, job_id, None)
}

/// Recompute the scorecards of `ids`, then re-rank the whole job.
///
/// `None` scores every candidate (an input change). A slice (a wave's K1) touches
/// only those candidates' rows: a stage writes grades only for the candidates it
/// processed, so every other candidate's stored scorecard is still correct.
///
/// Re-ranking is global either way — rank is a property of the order, not of the
/// candidate — and reads the persisted totals, so it costs one indexed scan.
pub fn rescore_ids(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    ids: Option<&[String]>,
) -> Result<RescoreReport, DomainError> {
    let conn = rt.open_conn().map_err(dberr)?;
    let heavy = rt.model();
    rescore_conn(&conn, job_id, &heavy, ids)
}

/// [`rescore_ids`] over a connection the caller already holds, with the embedder slug
/// named explicitly.
///
/// The engine-free core, and the reason this module is testable without a model: a
/// projection needs the database and the model slug, nothing else. A caller that has
/// a connection (a test, an export, a handler holding the lock) uses this; a caller
/// that does not goes through [`rescore_ids`].
pub fn rescore_conn(
    conn: &Connection,
    job_id: &str,
    heavy: &str,
    ids: Option<&[String]>,
) -> Result<RescoreReport, DomainError> {
    let scout_space = scout::scout_model_for(heavy);
    rescore_in_spaces(conn, job_id, heavy, scout_space.as_deref(), ids)
}

/// [`rescore_conn`] with the scout's slug named explicitly instead of read from
/// `ikapp.yaml`, so a test can name a space of its own instead of depending on
/// whatever this box is configured with.
pub fn rescore_in_spaces(
    conn: &Connection,
    job_id: &str,
    heavy: &str,
    scout_space: Option<&str>,
    ids: Option<&[String]>,
) -> Result<RescoreReport, DomainError> {
    let heavy = heavy.to_string();
    let scout_space = scout_space.map(str::to_string);

    let heavy_grades = crate::db::stage_scores_for_job(conn, job_id, &heavy).map_err(dberr)?;
    let scout_grades = match scout_space.as_deref() {
        Some(s) => crate::db::stage_scores_for_job(conn, job_id, s).map_err(dberr)?,
        None => HashMap::new(),
    };
    if heavy_grades.is_empty() && scout_grades.is_empty() {
        // Nothing has ever been graded for this job — no ad and no criteria, so there
        // is nothing to measure against. Leaving the sheet empty is honest: import
        // order is the only order there is until a stage runs, and inventing a zero
        // total would be a grade that means nothing.
        let ungraded = conn
            .query_row(
                "SELECT COUNT(*) FROM candidates
                 WHERE job_id = ?1 AND extraction_status = 'Extracted' AND extracted_text IS NOT NULL",
                rusqlite::params![job_id],
                |r| r.get::<_, i64>(0),
            )
            .map_err(dberr)?;
        return Ok(RescoreReport {
            ungraded,
            ..Default::default()
        });
    }

    let heavy_rubric = Rubric::with_vectors(conn, job_id, &heavy).map_err(dberr)?;
    let scout_rubric = match scout_space.as_deref() {
        Some(s) => Some(Rubric::with_vectors(conn, job_id, s).map_err(dberr)?),
        None => None,
    };
    let ad_weight = heavy_rubric.ad_weight;

    let candidates = target_candidates(conn, job_id, ids)?;
    let judge_grades = judge_grades(conn, &candidates)?;

    let mut report = RescoreReport {
        capacity: heavy_rubric.capacity() as f64,
        ..Default::default()
    };
    let mut pending: Vec<Pending> = Vec::with_capacity(candidates.len());
    for cid in &candidates {
        // The candidate's space: the heavy one when the heavy embedder has graded
        // it, the scout's otherwise. The judge does not decide this — it replaces
        // the ad slot's value, while the criteria and feedback terms still come
        // from the candidate's own space.
        let heavy_fit = heavy_grades.get(&(cid.clone(), AD_Q.to_string())).copied();
        let scout_fit = match scout_space.as_deref() {
            Some(_) => scout_grades.get(&(cid.clone(), AD_Q.to_string())).copied(),
            None => None,
        };
        let (space, rubric, grades) = match (&heavy_fit, &scout_rubric) {
            (Some(_), _) => (heavy.as_str(), &heavy_rubric, &heavy_grades),
            (None, Some(r)) if r.vector(AD_Q, scout_space.as_deref().unwrap_or("")).is_some() => (
                scout_space.as_deref().unwrap_or_default(),
                r,
                &scout_grades,
            ),
            _ => (heavy.as_str(), &heavy_rubric, &heavy_grades),
        };
        let space_fit = grades.get(&(cid.clone(), AD_Q.to_string())).copied();
        let judged = judge_grades.get(cid).copied().filter(|w| (0.0..=1.0).contains(w));
        let Some(fit) = judged.or(space_fit) else {
            report.ungraded += 1;
            continue;
        };
        if space == heavy.as_str() {
            if judged.is_some() {
                report.judged += 1;
            }
        } else {
            report.scout_only += 1;
            if judged.is_some() {
                report.judged += 1;
            }
        }

        // Criteria and feedback similarities, from the candidate's space.
        let mut sims: HashMap<String, f64> = HashMap::new();
        let mut scores: Vec<RubricScore> = Vec::with_capacity(rubric.questions.len() + 3);
        for q in &rubric.questions {
            let Some(sim) = grades.get(&(cid.clone(), q.id.clone())).copied() else {
                continue;
            };
            sims.insert(q.id.clone(), sim);
            let explanations =
                explanations_for(conn, cid, space, &q.id, 2).map_err(dberr)?;
            scores.push(RubricScore {
                question_id: q.id.clone(),
                label: q.label.clone(),
                kind: q.kind.clone(),
                points: q.max_points * sim as f32,
                max_points: q.max_points,
                chunk_explanations: explanations,
            });
        }
        let total = rubric.total(fit, &sims);

        // The ad slot, unless the judge has taken it over. Exactly one row here
        // carries it and its id names the stage; when the judge *has*, its own row
        // (written by the judge stage — the only writer of `ad_judge`) is the one the
        // sheet shows, and this pass only reads its value.
        if judged.is_none() {
            let (slot_id, slot_label, slot_kind, slot_points) = match heavy_fit {
                Some(cos) => (
                    AD_Q.to_string(),
                    crate::sort::AD_LABEL.to_string(),
                    QuestionKind::Ad,
                    ad_weight as f64 * cos,
                ),
                None => (
                    scout::SCOUT_QUESTION_ID.to_string(),
                    scout::SCOUT_LABEL.to_string(),
                    QuestionKind::Scout,
                    ad_weight as f64 * space_fit.unwrap_or(0.0),
                ),
            };
            scores.insert(
                0,
                RubricScore {
                    question_id: slot_id,
                    label: slot_label,
                    kind: slot_kind,
                    points: slot_points as f32,
                    max_points: ad_weight,
                    chunk_explanations: vec![],
                },
            );
        }

        // Stage columns: every stage's own view of this CV's ad fit, so the sheet can
        // show what each step said. Whichever row is already carrying the ad slot is
        // not repeated, and none of them reaches `points` — the total above already
        // decided.
        let emitted: Vec<String> = scores.iter().map(|s| s.question_id.clone()).collect();
        if let Some(cos) = heavy_fit {
            if !emitted.iter().any(|q| q == AD_Q) {
                scores.push(RubricScore {
                    question_id: AD_Q.to_string(),
                    label: crate::sort::AD_LABEL.to_string(),
                    kind: QuestionKind::Ad,
                    points: (ad_weight as f64 * cos) as f32,
                    max_points: ad_weight,
                    chunk_explanations: vec![],
                });
            }
        }
        if let Some(cos) = scout_fit {
            if !emitted.iter().any(|q| q == scout::SCOUT_QUESTION_ID) {
                scores.push(RubricScore {
                    question_id: scout::SCOUT_QUESTION_ID.to_string(),
                    label: scout::SCOUT_LABEL.to_string(),
                    kind: QuestionKind::Scout,
                    points: (ad_weight as f64 * cos) as f32,
                    max_points: ad_weight,
                    chunk_explanations: vec![],
                });
            }
        }

        // The denominator is what this row was *measured* against: the ad slot plus
        // the questions it has a grade for. A feedback row with no learned vector is
        // not capacity the ranking used, and a criterion the heavy embedder has not
        // reached is not one either.
        let graded: Vec<&str> = sims.keys().map(|k| k.as_str()).collect();
        pending.push(Pending {
            id: cid.clone(),
            total,
            scores,
            capacity: heavy_rubric.capacity_for(&graded),
        });
        report.scored += 1;
    }

    persist(conn, &pending)?;
    re_rank(conn, job_id)?;
    stamp_ranked(conn, job_id)?;
    Ok(report)
}


/// Any DB-layer error (a `rusqlite::Error` or a plain message) into the domain
/// error, so callers do not have to care which a helper returned.
fn dberr(e: impl std::fmt::Display) -> DomainError {
    DomainError::DbError {
        message: e.to_string(),
    }
}

/// A candidate's recomputed scorecard, ready to write.
struct Pending {
    id: String,
    total: f64,
    scores: Vec<RubricScore>,
    /// `ad_weight + Σ weights of the questions this row was scored against`.
    capacity: f32,
}

/// The candidates a pass covers: the requested slice, or every extracted one.
fn target_candidates(
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

/// The judge's win-rate for each of `ids`, read from the `candidate_scores` rows
/// the judge stage owns (`points / max_points` **is** the win-rate).
fn judge_grades(
    conn: &Connection,
    ids: &[String],
) -> Result<HashMap<String, f64>, DomainError> {
    let mut out = HashMap::new();
    if ids.is_empty() {
        return Ok(out);
    }
    let holes = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT candidate_id, points / NULLIF(max_points, 0) FROM candidate_scores
         WHERE question_id = '{JUDGE_Q}' AND candidate_id IN ({holes})"
    );
    let mut stmt = conn.prepare(&sql).map_err(dberr)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
    let rows = stmt
        .query_map(params.as_slice(), |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<f64>>(1)?))
        })
        .map_err(dberr)?;
    for row in rows {
        let (cid, win) = row.map_err(dberr)?;
        if let Some(w) = win {
            out.insert(cid, w);
        }
    }
    Ok(out)
}

/// The top-`top_k` matching chunks of one question, as the "why is this here"
/// explanation the results sheet and the rubric dialog show.
///
/// Recomputed here rather than stored by the stage, because it is pure arithmetic
/// over vectors that are already on disk — which is why rescore can afford it on
/// every pass. (The ad slot has no explanation: a judge's win-rate is a
/// comparison, not a chunk, and an ad cosine is the same thing the SCO/EMB
/// columns already show.)
fn explanations_for(
    conn: &Connection,
    candidate_id: &str,
    space: &str,
    question_id: &str,
    top_k: usize,
) -> Result<Vec<ChunkExplanation>, DomainError> {
    let qv: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vector FROM rubric_vectors
             WHERE job_id = (SELECT job_id FROM candidates WHERE id = ?1)
               AND question_id = ?2 AND model = ?3",
            rusqlite::params![candidate_id, question_id, space],
            |r| r.get(0),
        )
        .ok();
    let (Some(qv), Some((chunked, texts))) = (
        qv,
        crate::embeddings::load_candidate_chunks(conn, candidate_id, space).map_err(dberr)?,
    ) else {
        return Ok(vec![]);
    };
    let qv: Vec<f64> = crate::embed::bytes_to_vector(&qv, None)
        .map_err(dberr)?
        .into_iter()
        .map(|x| x as f64)
        .collect();
    let (_score, indices) = chunked.max_sim_top_k(&qv, top_k);
    Ok(indices
        .iter()
        .map(|&i| ChunkExplanation {
            text: texts.get(i).cloned().unwrap_or_default(),
            similarity: libembedrank::cosine_similarity(&qv, &chunked.chunks[i]),
        })
        .collect())
}

/// Replace the computed scorecards' display rows and totals, in one transaction.
///
/// A partially written sheet would be visible to a concurrent reader as a mixture
/// of old and new totals, so the whole pass commits at once.
fn persist(conn: &Connection, pending: &[Pending]) -> Result<(), DomainError> {
    if pending.is_empty() {
        return Ok(());
    }
    let tx = conn.unchecked_transaction().map_err(dberr)?;
    for p in pending {
        // This pass owns every row in the rubric: the ad slot and every feedback
        // and criterion row. Two rows it must not touch: `fb_job`/`fb_profile`
        // belong to the feedback learner when this rubric's own `questions` list
        // does not carry them (weight 0), and the `sort_reports`/judge bookkeeping
        // lives elsewhere entirely. Both are named explicitly so a future rubric
        // row cannot be clobbered by omission.
        tx.execute(
            "DELETE FROM candidate_scores WHERE candidate_id = ?1 AND question_id != ?2",
            rusqlite::params![p.id, JUDGE_Q],
        )
        .map_err(dberr)?;
        {
            let mut ins = tx
                .prepare(
                    "INSERT INTO candidate_scores
                       (id, candidate_id, question_id, kind, label, points, max_points, sign, chunk_texts)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'Positive', ?8)",
                )
                .map_err(dberr)?;
            for s in &p.scores {
                let chunks = if s.chunk_explanations.is_empty() {
                    None
                } else {
                    Some(serde_json::to_string(&s.chunk_explanations).unwrap_or_default())
                };
                ins.execute(rusqlite::params![
                    format!("cs_{}", uuid::Uuid::new_v4()),
                    p.id,
                    s.question_id,
                    s.kind.as_str(),
                    s.label,
                    s.points,
                    s.max_points,
                    chunks
                ])
                .map_err(dberr)?;
            }
        }
        tx.execute(
            "UPDATE candidates SET points = ?1, points_max = ?2 WHERE id = ?3",
            rusqlite::params![p.total, p.capacity, p.id],
        )
        .map_err(dberr)?;
    }
    tx.commit().map_err(dberr)?;
    Ok(())
}

/// Re-rank the job from the persisted totals.
///
/// Ties break on file name so the order is stable across passes: a wave that
/// changes nothing must not shuffle the sheet under the user.
fn re_rank(conn: &Connection, job_id: &str) -> Result<(), DomainError> {
    let mut stmt = conn
        .prepare(
            "SELECT id, points, file_name FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted' AND points IS NOT NULL",
        )
        .map_err(dberr)?;
    let mut rows: Vec<(String, f64, String)> = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(dberr)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(dberr)?;
    drop(stmt);
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    let tx = conn.unchecked_transaction().map_err(dberr)?;
    tx.execute(
        "UPDATE candidates SET rank = NULL WHERE job_id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(dberr)?;
    {
        let mut upd = tx
            .prepare("UPDATE candidates SET rank = ?1 WHERE id = ?2")
            .map_err(dberr)?;
        for (i, (id, _, _)) in rows.iter().enumerate() {
            upd.execute(rusqlite::params![i as i64 + 1, id])
                .map_err(dberr)?;
        }
    }
    tx.commit().map_err(dberr)?;
    Ok(())
}

/// Stamp the job as ranked at its current input version.
///
/// Rescore runs on every input change, so a rescored job is never *stale*: the
/// sheet is always the best answer available right now, and what is left is
/// *missing* grades rather than wrong ones. That is why `rerank_dirty` reads
/// false in practice, and the rank button ends up meaning "do more work" instead
/// of "fix staleness".
fn stamp_ranked(conn: &Connection, job_id: &str) -> Result<(), DomainError> {
    conn.execute(
        "UPDATE jobs SET ranked_version = inputs_version, last_ranked_at = datetime('now'),
                         updated_at = datetime('now')
         WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(dberr)?;
    Ok(())
}

/// Record what the last pass did, for the gui and the tests to assert on.
pub fn write_report(
    conn: &Connection,
    job_id: &str,
    judge_stats: Option<&crate::judge_cache::PairVerdictStats>,
    mode: &str,
    wave: u32,
    complete: bool,
) -> Result<(), DomainError> {
    let fresh = judge_stats.map(|s| s.fresh).unwrap_or(0);
    let cached = judge_stats.map(|s| s.cached).unwrap_or(0);
    // Three states, not two: the judge did not run at all, ran and every verdict came
    // from the memoized cache, or ran and computed something. "Ran" with nothing
    // computed is the ambiguous middle the gui must not show.
    let judge_mode = match judge_stats {
        None => "skipped",
        Some(s) if s.fresh == 0 && s.cached > 0 => "cached",
        Some(s) if s.fresh == 0 => "skipped",
        Some(_) => "ran",
    };
    conn.execute(
        "INSERT OR REPLACE INTO sort_reports
           (job_id, judge_mode, judge_pairs_fresh, judge_pairs_cached, mode, waves, complete, cancelled, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, datetime('now'))",
        rusqlite::params![
            job_id,
            judge_mode,
            fresh as i64,
            cached as i64,
            mode,
            wave as i64,
            complete as i64
        ],
    )
    .map_err(dberr)?;
    Ok(())
}

/// The current ranked triqui plus the last report, read straight from the DB.
///
/// A finished pass already persisted everything, so a caller that needs the report
/// reads it back rather than re-running the pipeline: a second pass would find
/// every judge pair memoized and report `fresh = 0`, a lie about what happened.
pub fn read_triqui(
    conn: &Connection,
    job_id: &str,
) -> Result<(Vec<TriqiItem>, SortReport), DomainError> {
    let size: i64 = conn
        .query_row(
            "SELECT triqui_size FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(dberr)?;
    let mut stmt = conn
        .prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at
             FROM candidates WHERE job_id = ?1 AND rank IS NOT NULL ORDER BY rank LIMIT ?2",
        )
        .map_err(dberr)?;
    let triqui: Vec<TriqiItem> = stmt
        .query_map(rusqlite::params![job_id, size], triqui_item_from_row)
        .map_err(dberr)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(dberr)?;
    // A job that has been projected but never reported (a rescore with no judge in
    // the loop) has no row yet, and "no row" is not an error — it reads as a
    // blocking rank that skipped the judge.
    let report = conn
        .query_row(
            "SELECT judge_mode, judge_pairs_fresh, judge_pairs_cached, mode, waves, created_at
             FROM sort_reports WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| {
                Ok(SortReport {
                    job_id: job_id.to_string(),
                    judge_mode: r.get(0)?,
                    judge_pairs_fresh: r.get(1)?,
                    judge_pairs_cached: r.get(2)?,
                    mode: r.get(3)?,
                    waves: r.get(4)?,
                    complete: true,
                    cancelled: false,
                    created_at: r.get(5)?,
                })
            },
        )
        .optional()
        .map_err(dberr)?
        .unwrap_or(SortReport {
            job_id: job_id.to_string(),
            judge_mode: "skipped".into(),
            judge_pairs_fresh: 0,
            judge_pairs_cached: 0,
            mode: "full".into(),
            waves: 0,
            complete: true,
            cancelled: false,
            created_at: String::new(),
        });
    Ok((triqui, report))
}

fn triqui_item_from_row(row: &rusqlite::Row) -> rusqlite::Result<TriqiItem> {
    use crate::types::{ExtractionStatus, FileType};
    Ok(TriqiItem {
        id: row.get(0)?,
        file_name: row.get(1)?,
        file_type: FileType::from_str(row.get::<_, String>(2)?.as_str()),
        extraction_status: ExtractionStatus::from_str(row.get::<_, String>(3)?.as_str()),
        rank: row.get(4)?,
        points: row.get(5)?,
        points_max: row.get(6)?,
        liked: row.get::<_, i64>(7)? != 0,
        viewed_at: row.get(8)?,
    })
}

/// Every candidate of a job in best-current-order: strongest available score
/// first, import order for the ones nothing has graded yet.
///
/// Ordering by the persisted total — one indexed column — is what makes wave
/// selection O(N) integer compares instead of O(N · questions) float work, and it
/// is the score the user actually sees.
pub fn best_order(
    conn: &Connection,
    job_id: &str,
) -> Result<Vec<String>, DomainError> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted' AND extracted_text IS NOT NULL
             ORDER BY points DESC NULLS LAST, imported_at, id",
        )
        .map_err(dberr)?;
    let ids = stmt
        .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
        .map_err(dberr)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(dberr)?;
    Ok(ids)
}

/// The first `limit` candidates in [`best_order`] with no chunk vectors for
/// `model` yet — the next wave's work.
pub fn candidates_needing_embed(
    conn: &Connection,
    job_id: &str,
    model: &str,
    limit: usize,
) -> Result<Vec<String>, DomainError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let ordered = best_order(conn, job_id)?;
    let mut stmt = conn
        .prepare("SELECT 1 FROM candidate_chunks WHERE candidate_id = ?1 AND vector_model = ?2 LIMIT 1")
        .map_err(dberr)?;
    let mut out = Vec::with_capacity(limit.min(64));
    for cid in ordered {
        let has: Option<i64> = stmt
            .query_row(rusqlite::params![&cid, model], |r| r.get(0))
            .ok();
        if has.is_none() {
            out.push(cid);
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok(out)
}

/// How many candidates still need the heavy embedder.
pub fn count_needing_embed(
    conn: &Connection,
    job_id: &str,
    heavy: &str) -> Result<i64, DomainError> {
    conn.query_row(
        "SELECT COUNT(*) FROM candidates c
         WHERE c.job_id = ?1 AND c.extraction_status = 'Extracted' AND c.extracted_text IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM candidate_chunks cc
                           WHERE cc.candidate_id = c.id AND cc.vector_model = ?2)",
        rusqlite::params![job_id, heavy],
        |r| r.get(0),
    )
    .map_err(db_err)
}

/// A job's rubric capacity — the "Total /N" denominator.
pub fn capacity(conn: &Connection, job_id: &str) -> Result<f64, DomainError> {
    Ok(Rubric::load(conn, job_id).map_err(dberr)?.capacity() as f64)
}

/// Sum of a job's totals — the headline number in the gui.
pub fn total_points(conn: &Connection, job_id: &str) -> Result<f64, DomainError> {
    conn.query_row(
        "SELECT COALESCE(SUM(points), 0.0) FROM candidates
         WHERE job_id = ?1 AND points IS NOT NULL",
        rusqlite::params![job_id],
        |r| r.get(0),
    )
    .map_err(db_err)
}

/// What a ranking input change does to the derived state.
///
/// One call, from every handler that mutates the rubric. This is the whole of
/// `gui/design/app1/cascade-simple.md` §4: because the rubric is the only thing a
/// rank reads, an input change is (1) embed the one text that changed, (2) project,
/// and (3) at most invalidate the judge — never "cancel the run", never "mark it
/// stale until somebody re-ranks".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Invalidation {
    /// Rubric vectors this pass had to embed (usually 0 or 1).
    pub embedded: usize,
    /// Candidates re-projected.
    pub rescored: i64,
    /// Whether the judge pool was cleared (an ad change invalidates every
    /// memoized verdict, because the ad's content hash is part of the key).
    pub pool_cleared: bool,
    /// Whether the sheet was already up to date before this call.
    pub was_current: bool,
}

/// Re-project a job after a ranking input changed.
///
/// `ad_changed` distinguishes the two cases that behave differently:
///
/// - **criteria / weights / feedback.** The judge never read them, so every
///   memoized verdict is still valid and the pool still holds. The sheet is fixed
///   by this function alone — no run needed, no cancel, nothing lost.
/// - **the job ad.** Every judge verdict is keyed on the ad's content hash, so the
///   pool is emptied and the `ad_judge` rows are dropped; the sheet falls back to
///   the embedder's cosine until a run re-judges. Candidates' own vectors are
///   untouched (they are text-only), so no re-embedding is ever needed for an ad
///   change.
///
/// An empty ad is not a ranking input: with nothing to measure against, every
/// total would be 0, so the rubric simply has no ad question this pass.
pub fn inputs_changed(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    ad_changed: bool,
) -> Result<Invalidation, DomainError> {
    let conn = rt.open_conn().map_err(dberr)?;
    conn.execute(
        "UPDATE jobs SET inputs_version = inputs_version + 1, updated_at = datetime('now')
         WHERE id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(dberr)?;

    let was_current = conn
        .query_row(
            "SELECT ranked_version = inputs_version - 1 FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
        != 0;

    // Embed whatever rubric text has no vector yet, in every space the job grades
    // in. One criterion edited is one embed; the rest is a cache hit.
    let heavy = rt.model();
    let spaces: Vec<String> = {
        let mut v = vec![heavy.clone()];
        if let Some(s) = scout::scout_model_for(&heavy) {
            v.push(s);
        }
        v
    };
    let mut embedded = 0usize;
    for space in &spaces {
        let rubric = Rubric::load(&conn, job_id)?;
        // The scout's space is filled from the embedding cache only — loading its
        // session from a request thread would evict the heavy embedder a running
        // cascade is using, and the scout pass picks up anything left at its next
        // boundary. The heavy space is filled properly, because its vectors are
        // what the totals are made of.
        embedded += if space == &heavy {
            rubric::ensure_vectors(rt, &conn, &rubric, space, None).map_err(dberr)?
        } else {
            rubric::ensure_vectors_cache_only(&conn, &rubric, space).map_err(dberr)?
        };
    }
    // A vector alone changes nothing: the grade is what rescore reads. So re-score
    // every live space — a pure cosine pass over stored chunk vectors, no model.
    score_live_spaces(
        &conn,
        job_id,
        &heavy,
        scout::scout_model_for(&heavy).as_deref(),
    )?;

    if ad_changed {
        crate::db::clear_judge_pool(&conn, job_id).map_err(dberr)?;
        // Drop the judge grades too: they were earned against the *old* ad, so
        // leaving them would rank by a verdict about a job description the user
        // no longer has. The pool is refrozen by the next run.
        conn.execute(
            "DELETE FROM candidate_scores WHERE question_id = ?1
             AND candidate_id IN (SELECT id FROM candidates WHERE job_id = ?2)",
            rusqlite::params![JUDGE_Q, job_id],
        )
        .map_err(dberr)?;
        // An empty ad leaves nothing to measure against, so the heavy space has no
        // ad vector and every candidate falls back to its scout grade.
        let has_ad: Option<String> = conn
            .query_row(
                "SELECT job_ad_text FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| r.get(0),
            )
            .ok();
        if has_ad.as_deref().map(|t| t.trim().is_empty()).unwrap_or(true) {
            conn.execute(
                "DELETE FROM rubric_vectors
                 WHERE job_id = ?1 AND question_id = 'ad'",
                rusqlite::params![job_id],
            )
            .map_err(dberr)?;
            conn.execute(
                "DELETE FROM stage_scores WHERE model = ?1 AND question_id = 'ad'
                 AND candidate_id IN (SELECT id FROM candidates WHERE job_id = ?2)",
                rusqlite::params![heavy, job_id],
            )
            .map_err(dberr)?;
        }
    }
    drop(conn);

    let report = rescore(rt, job_id)?;
    Ok(Invalidation {
        embedded,
        rescored: report.scored,
        pool_cleared: ad_changed,
        was_current,
    })
}

/// Re-score every candidate against the rubric in every space the job already
/// grades in.
///
/// Call this whenever a *new* question becomes scoreable — a criterion added, or the
/// two learned feedback rows appearing for the first time. Embedding a question's
/// vector is not enough: the grade is what the projection reads, and a question with
/// a vector but no grade is a column with a header and no numbers.
///
/// Only spaces that already hold chunk vectors for this job's candidates are touched:
/// those are the ones a stage has reached, and they are exactly the spaces a grade
/// can be computed in. (Testing for *grades* instead would skip the heavy space on a
/// job that was imported without an ad — it has chunks from the prefill, no grades,
/// and is the space that most needs one.) It never calls a model: every vector
/// involved is on disk by the time a question can be scored.
pub fn score_live_spaces(
    conn: &Connection,
    job_id: &str,
    heavy: &str,
    scout_space: Option<&str>,
) -> Result<usize, DomainError> {
    let mut spaces = vec![heavy.to_string()];
    if let Some(s) = scout_space {
        spaces.push(s.to_string());
    }
    let mut scored = 0usize;
    for space in &spaces {
        let reached: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM candidate_chunks cc
                 JOIN candidates c ON c.id = cc.candidate_id
                 WHERE c.job_id = ?1 AND cc.vector_model = ?2 LIMIT 1",
                rusqlite::params![job_id, space],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if reached == 0 {
            continue;
        }
        let rubric = Rubric::with_vectors(conn, job_id, space).map_err(dberr)?;
        if rubric.scored_questions(space).is_empty() {
            continue;
        }
        scored += rubric::score_candidates_in_space(conn, job_id, space, &rubric, None)
            .map_err(dberr)?;
    }
    Ok(scored)
}

/// Whether a job still has work a cascade would do: candidates the heavy embedder
/// has not reached, or pool members unjudged.
///
/// This is what the rank button means now. A rescored job is never *stale* — the
/// sheet is always the best answer available right now — but it can be
/// *incomplete*, and this is the honest question to ask the user about it.
pub fn pending_work(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<PendingWork, DomainError> {
    let conn = rt.open_conn().map_err(dberr)?;
    let heavy = rt.model();
    let remaining_embed = count_needing_embed(&conn, job_id, &heavy)?;
    let pool = crate::db::judge_pool_size(&conn, job_id).map_err(dberr)? as usize;
    let pool_remaining = if pool == 0 {
        0
    } else {
        conn.query_row(
            "SELECT COUNT(*) FROM judge_pool jp
             LEFT JOIN candidate_scores cs
               ON cs.candidate_id = jp.candidate_id AND cs.question_id = ?2
             WHERE jp.job_id = ?1 AND cs.id IS NULL",
            rusqlite::params![job_id, JUDGE_Q],
            |r| r.get(0),
        )
        .map_err(dberr)?
    };
    let ungraded: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted' AND points IS NULL",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(dberr)?;
    Ok(PendingWork {
        remaining_embed,
        pool,
        pool_remaining,
        ungraded,
    })
}

/// What a cascade would still do for a job. Distinct from "stale": a rescored job
/// is never stale, only incomplete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingWork {
    /// Candidates the heavy embedder has not reached.
    pub remaining_embed: i64,
    /// The frozen judge pool's size.
    pub pool: usize,
    /// Pool members with no judge grade yet.
    pub pool_remaining: i64,
    /// Candidates with no total at all (no ad and no criteria yet).
    pub ungraded: i64,
}

impl PendingWork {
    /// Whether anything at all is left to do.
    pub fn is_empty(&self) -> bool {
        self.remaining_embed == 0 && self.pool_remaining == 0 && self.ungraded == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{migrate, SCHEMA};
    use crate::rubric::{AD_Q, FB_JOB_Q};

    /// The embedder slug the tests pretend is selected. No model is ever loaded: the
    /// projection reads stored vectors only, which is the point of testing it here.
    const HEAVY: &str = "heavy-slug";
    const SCOUT: &str = "scout-slug";

    /// The projection, with both spaces named — no dependency on this box's
    /// `ikapp.yaml`.
    fn rescore(
        c: &rusqlite::Connection,
        job_id: &str,
        ids: Option<&[String]>,
    ) -> Result<RescoreReport, DomainError> {
        rescore_in_spaces(c, job_id, HEAVY, Some(SCOUT), ids)
    }

    fn conn() -> rusqlite::Connection {
        let c = rusqlite::Connection::open_in_memory().expect("in-memory db");
        c.execute_batch(SCHEMA).expect("schema");
        migrate(&c).expect("migrate");
        c
    }

    fn job(c: &rusqlite::Connection, id: &str) {
        c.execute(
            "INSERT INTO jobs (id, title, ad_weight, fb_job_weight, fb_profile_weight, job_ad_text)
             VALUES (?1, 'j', 50.0, 0.0, 0.0, 'a rust backend engineer with postgres')",
            rusqlite::params![id],
        )
        .unwrap();
    }

    fn candidate(c: &rusqlite::Connection, job_id: &str, id: &str) {
        c.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type,
                                     extraction_status, extracted_text)
             VALUES (?1, ?2, ?3, ?4, 'Txt', 'Extracted', 'some cv text')",
            rusqlite::params![id, job_id, format!("/tmp/{id}"), id],
        )
        .unwrap();
    }

    fn criterion(c: &rusqlite::Connection, job_id: &str, id: &str, weight: f64) {
        c.execute(
            "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, ?2, 'postgres', 'Positive', ?3)",
            rusqlite::params![id, job_id, weight],
        )
        .unwrap();
    }

    /// A unit vector along `axis`, so a grade is exactly `axis` — the tests assert on
    /// arithmetic, not on cosine subtleties.
    fn axis(id: &str, dim: usize) -> Vec<f32> {
        let axis: usize = id.bytes().map(|b| (b - b'0') as usize).sum::<usize>() % dim;
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    fn chunk(c: &rusqlite::Connection, cand: &str, space: &str, vec: &[f32]) {
        c.execute(
            "INSERT INTO candidate_chunks (id, candidate_id, chunk_index, chunk_text,
                                           content_hash, vector, vector_model)
             VALUES (?1, ?2, 0, 't', 'h', ?3, ?4)",
            rusqlite::params![
                format!("cc_{cand}_{space}"),
                cand,
                crate::embed::vector_to_bytes(vec),
                space
            ],
        )
        .unwrap();
    }

    /// The state the scout leaves behind: rubric vectors in the scout space, chunks,
    /// and a grade for every candidate against every question.
    fn scout_pass(c: &rusqlite::Connection, job_id: &str, space: &str, dim: usize) {
        crate::db::put_rubric_vector(c, job_id, AD_Q, space, &axis("1", dim)).unwrap();
        crate::db::put_rubric_vector(c, job_id, "crit_a", space, &axis("2", dim)).unwrap();
        // Only candidates the test created, so a two-candidate job does not trip the
        // chunk table's foreign key.
        let present: Vec<String> = {
            let mut stmt = c
                .prepare("SELECT id FROM candidates WHERE job_id = ?1 ORDER BY id")
                .unwrap();
            stmt.query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
                .unwrap()
                .map_while(Result::ok)
                .collect()
        };
        for (i, cand) in present.iter().enumerate() {
            chunk(c, cand, space, &axis(&(i + 1).to_string(), dim));
        }
        let rubric = Rubric::with_vectors(c, job_id, space).unwrap();
        crate::rubric::score_candidates_in_space(c, job_id, space, &rubric, None).unwrap();
    }

    fn points_of(c: &rusqlite::Connection, cand: &str) -> Option<f64> {
        c.query_row(
            "SELECT points FROM candidates WHERE id = ?1",
            rusqlite::params![cand],
            |r| r.get(0),
        )
        .ok()
    }

    fn question_ids(c: &rusqlite::Connection, cand: &str) -> Vec<String> {
        let mut stmt = c
            .prepare("SELECT question_id FROM candidate_scores WHERE candidate_id = ?1 ORDER BY question_id")
            .unwrap();
        stmt.query_map(rusqlite::params![cand], |r| r.get::<_, String>(0))
            .unwrap()
            .map_while(Result::ok)
            .collect()
    }

    /// The headline promise: after the scout pass every candidate has a total built
    /// from the *whole* rubric, not just the ad. This is the single invariant the
    /// whole scoring-v3 simplification rests on.
    #[test]
    fn every_candidate_is_scored_against_the_whole_rubric_after_one_pass() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        for cand in ["c1", "c2", "c3"] {
            candidate(&c, "j1", cand);
        }
        scout_pass(&c, "j1", "scout-slug", 4);

        let rubric = Rubric::with_vectors(&c, "j1", "scout-slug").unwrap();
        assert_eq!(rubric.scored_questions("scout-slug"), vec![AD_Q, "crit_a"]);

        for cand in ["c1", "c2", "c3"] {
            let graded = stage_question_ids(&c, cand, "scout-slug");
            assert!(
                graded.contains(&"crit_a".to_string()),
                "{cand} must be scored against the criterion, not only the ad: {graded:?}"
            );
            assert!(graded.contains(&AD_Q.to_string()), "{graded:?}");
        }

        // And after the projection, the sheet carries that criterion grade for every
        // candidate — which is the property that used to require a "pending" tier.
        rescore(&c, "j1", None).unwrap();
        for cand in ["c1", "c2", "c3"] {
            assert!(points_of(&c, cand).is_some(), "{cand} has a total after one pass");
            let ids = question_ids(&c, cand);
            assert!(
                ids.contains(&"crit_a".to_string()),
                "{cand}'s sheet row carries its criterion: {ids:?}"
            );
            assert!(
                ids.contains(&scout::SCOUT_QUESTION_ID.to_string()),
                "{cand} carries the scout's ad grade: {ids:?}"
            );
            assert!(
                !ids.contains(&AD_Q.to_string()),
                "an unembedded candidate has no heavy ad row: {ids:?}"
            );
        }
    }

    fn stage_question_ids(c: &rusqlite::Connection, cand: &str, space: &str) -> Vec<String> {
        let mut stmt = c
            .prepare(
                "SELECT question_id FROM stage_scores WHERE candidate_id = ?1 AND model = ?2
                 ORDER BY question_id",
            )
            .unwrap();
        stmt.query_map(rusqlite::params![cand, space], |r| r.get::<_, String>(0))
            .unwrap()
            .map_while(Result::ok)
            .collect()
    }

    /// A scout-only candidate's total is `ad_weight·fit + Σ w·sim` — complete, and
    /// comparable with another scout-only candidate's.
    #[test]
    fn scout_space_totals_include_criteria() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        candidate(&c, "j1", "c1"); // chunk axis 1 → ad sim 0, criterion sim 0
        candidate(&c, "j1", "c2"); // chunk axis 2 → ad sim 1, criterion sim 1
        scout_pass(&c, "j1", "scout-slug", 4);

        let rubric = Rubric::with_vectors(&c, "j1", "scout-slug").unwrap();
        let mut sims = HashMap::new();
        sims.insert("crit_a".to_string(), 0.0);
        assert_eq!(rubric.total(1.0, &sims), 50.0, "ad slot only when the criterion is 0");

        let mut sims = HashMap::new();
        sims.insert("crit_a".to_string(), 1.0);
        assert_eq!(rubric.total(0.0, &sims), 10.0, "the criterion's full weight");
        assert_eq!(rubric.capacity(), 60.0, "ad + one criterion");
    }

    /// `points` comes from `stage_scores`, not from summing the display rows — so a
    /// display-only row can never change what the sheet ranks by.
    #[test]
    fn the_ad_slot_is_one_row_and_the_total_ignores_the_display_rows() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        candidate(&c, "j1", "c1");
        candidate(&c, "j1", "c2");
        scout_pass(&c, "j1", "scout-slug", 4);
        // c1 also reaches the heavy embedder, so it gets an `ad` row as well.
        chunk(&c, "c1", "heavy-slug", &axis("1", 4));
        crate::db::put_rubric_vector(&c, "j1", AD_Q, "heavy-slug", &axis("1", 4)).unwrap();
        crate::db::put_rubric_vector(&c, "j1", "crit_a", "heavy-slug", &axis("2", 4)).unwrap();
        let rubric = Rubric::with_vectors(&c, "j1", "heavy-slug").unwrap();
        crate::rubric::score_candidates_in_space(&c, "j1", "heavy-slug", &rubric, None).unwrap();

        let report = rescore(&c, "j1", None).unwrap();
        assert_eq!(report.scored, 2);

        // The heavy-space candidate carries both stage views plus its criterion.
        let ids = question_ids(&c, "c1");
        assert!(ids.contains(&AD_Q.to_string()));
        assert!(ids.contains(&scout::SCOUT_QUESTION_ID.to_string()));
        assert!(ids.contains(&"crit_a".to_string()));

        // The ad slot: `ad` for the embedded one, `scout` for the other.
        assert_eq!(count_kind(&c, "c1", "Ad"), 1);
        assert_eq!(count_kind(&c, "c2", "Ad"), 0);
        assert_eq!(count_kind(&c, "c2", "Scout"), 1);
    }

    fn count_kind(c: &rusqlite::Connection, cand: &str, kind: &str) -> i64 {
        c.query_row(
            "SELECT COUNT(*) FROM candidate_scores WHERE candidate_id = ?1 AND kind = ?2",
            rusqlite::params![cand, kind],
            |r| r.get(0),
        )
        .unwrap()
    }

    /// The judge's grade replaces the ad slot's *value* and changes no row's
    /// max-points, so the capacity is unchanged and the judge row is counted.
    #[test]
    fn a_judge_grade_replaces_the_ad_value_and_leaves_capacity_alone() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        candidate(&c, "j1", "c1");
        scout_pass(&c, "j1", "scout-slug", 4);
        // A judge grade on its own (the judge stage is the only writer of it).
        c.execute(
            "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, label,
                                           points, max_points, sign)
             VALUES ('j1', 'c1', 'ad_judge', 'AdJudge', 'Judge', 40.0, 50.0, 'Positive')",
            [],
        )
        .unwrap();

        rescore(&c, "j1", None).unwrap();

        // 50 (ad weight) * 0.8 win-rate.
        assert_eq!(points_of(&c, "c1"), Some(40.0));
        assert_eq!(count_kind(&c, "c1", "AdJudge"), 1);
        let capacity: f64 = c
            .query_row("SELECT points_max FROM candidates WHERE id = 'c1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(capacity, 60.0, "the judge scales the ad slot, it does not add to it");
    }

    /// Ranking is global and stable: one candidate's new scorecard reorders the
    /// whole sheet, and a pass that changes nothing does not shuffle it.
    #[test]
    fn reranking_is_global_and_stable() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        for cand in ["c1", "c2", "c3"] {
            candidate(&c, "j1", cand);
        }
        scout_pass(&c, "j1", "scout-slug", 4);
        rescore(&c, "j1", None).unwrap();
        let first = ranks(&c);
        rescore(&c, "j1", None).unwrap();
        assert_eq!(ranks(&c), first, "an unchanged pass must not shuffle");

        // c3 overtakes c1: a one-candidate change re-ranks everyone. Only c3's grades
        // change — the point of the wave path.
        crate::db::put_stage_score(&c, "c3", "scout-slug", AD_Q, 1.0).unwrap();
        crate::db::put_stage_score(&c, "c3", "scout-slug", "crit_a", 1.0).unwrap();
        rescore(&c, "j1", Some(&["c3".to_string()])).unwrap();
        let after = ranks(&c);
        assert_eq!(
            after,
            vec!["c3".to_string(), "c1".to_string(), "c2".to_string()],
            "the improved candidate leads, the others keep their relative order"
        );
    }

    fn ranks(c: &rusqlite::Connection) -> Vec<String> {
        let mut stmt = c
            .prepare("SELECT id FROM candidates WHERE rank IS NOT NULL ORDER BY rank")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map_while(Result::ok)
            .collect()
    }

    /// A frozen pool is a property of the run, not of the wave: the top N by total,
    /// once, and re-picking is the only thing that changes it.
    #[test]
    fn the_judge_pool_is_the_top_n_by_total_and_only_a_refreeze_changes_it() {
        let c = conn();
        job(&c, "j1");
        for cand in ["c1", "c2", "c3"] {
            candidate(&c, "j1", cand);
        }
        scout_pass(&c, "j1", "scout-slug", 4);
        rescore(&c, "j1", None).unwrap();

        let all = crate::db::freeze_judge_pool(&c, "j1", 3).unwrap();
        assert_eq!(all, 3);
        let first = crate::db::judge_pool_ids(&c, "j1").unwrap();
        assert_eq!(first.len(), 3);
        assert_eq!(first[0], ranks(&c)[0], "the pool is in rank order");

        // A later, better grade does not move the pool by itself.
        crate::db::put_stage_score(&c, "c3", "scout-slug", AD_Q, 1.0).unwrap();
        rescore(&c, "j1", None).unwrap();
        assert_eq!(
            crate::db::judge_pool_ids(&c, "j1").unwrap(),
            first,
            "the pool must not re-pick itself"
        );

        // An explicit re-freeze does.
        crate::db::freeze_judge_pool(&c, "j1", 1).unwrap();
        assert_eq!(crate::db::judge_pool_ids(&c, "j1").unwrap().len(), 1);
    }

    /// `pool_size: 0` is a pool of nobody — the old budget read `0` as "no limit",
    /// so switching the judge off started the tournament.
    #[test]
    fn a_pool_of_zero_is_a_pool_of_nobody() {
        let c = conn();
        job(&c, "j1");
        candidate(&c, "j1", "c1");
        candidate(&c, "j1", "c2");
        scout_pass(&c, "j1", "scout-slug", 4);
        assert_eq!(crate::db::freeze_judge_pool(&c, "j1", 0).unwrap(), 0);
        assert!(crate::db::judge_pool_ids(&c, "j1").unwrap().is_empty());
    }

    /// The migration is idempotent and leaves the legacy slots alone (they are no
    /// longer read, but dropping a column in SQLite means a table rebuild).
    #[test]
    fn the_migration_folds_old_slots_in_once() {
        let c = conn();
        job(&c, "j1");
        criterion(&c, "j1", "crit_a", 10.0);
        let v = axis("3", 4);
        c.execute(
            "UPDATE jobs SET ead = ?1, ead_model = 'heavy-slug' WHERE id = 'j1'",
            rusqlite::params![crate::embed::vector_to_bytes(&v)],
        )
        .unwrap();
        c.execute(
            "UPDATE criteria SET vector = ?1, vector_model = 'heavy-slug' WHERE id = 'crit_a'",
            rusqlite::params![crate::embed::vector_to_bytes(&v)],
        )
        .unwrap();
        c.execute(
            "INSERT INTO job_feedback_vectors (job_id, model, ejobfeedback)
             VALUES ('j1', 'heavy-slug', ?1)",
            rusqlite::params![crate::embed::vector_to_bytes(&v)],
        )
        .unwrap();

        migrate(&c).unwrap();
        migrate(&c).unwrap();

        let ids: Vec<String> = crate::db::rubric_vectors(&c, "j1", "heavy-slug")
            .unwrap()
            .into_iter()
            .map(|(q, _)| q)
            .collect();
        assert!(ids.contains(&AD_Q.to_string()), "{ids:?}");
        assert!(ids.contains(&"crit_a".to_string()), "{ids:?}");
        assert!(ids.contains(&FB_JOB_Q.to_string()), "{ids:?}");

        // The columns themselves are still there, just unread.
        let still: Option<Vec<u8>> = c
            .query_row("SELECT ead FROM jobs WHERE id = 'j1'", [], |r| r.get(0))
            .unwrap();
        assert!(still.is_some());
    }

    /// The wave gate is the best available score, so a candidate the heavy embedder
    /// has not reached but the scout loves is embedded before one the scout dislikes
    /// and that is already embedded.
    #[test]
    fn the_wave_slice_is_the_best_scored_candidates_that_still_need_embedding() {
        let c = conn();
        job(&c, "j1");
        for cand in ["c1", "c2", "c3"] {
            candidate(&c, "j1", cand);
        }
        scout_pass(&c, "j1", "scout-slug", 4);
        rescore(&c, "j1", None).unwrap();
        // c2 is embedded already; c1 and c3 are not.
        chunk(&c, "c2", "heavy-slug", &axis("2", 4));

        let slice = candidates_needing_embed(&c, "j1", "heavy-slug", 10).unwrap();
        assert_eq!(slice.len(), 2);
        assert!(!slice.contains(&"c2".to_string()), "already embedded");
        assert_eq!(
            slice[0], ranks(&c)[0],
            "the best-scored candidate needing embedding comes first"
        );
        assert_eq!(count_needing_embed(&c, "j1", "heavy-slug").unwrap(), 2);
    }

    /// A rubric with no ad and no criteria is legal: everything is ungraded, and the
    /// sheet stays in import order rather than showing zeros as if they were grades.
    #[test]
    fn an_empty_rubric_grades_nothing_and_invents_no_totals() {
        let c = conn();
        c.execute(
            "INSERT INTO jobs (id, title, ad_weight, job_ad_text) VALUES ('j1', 'j', 50.0, NULL)",
            [],
        )
        .unwrap();
        candidate(&c, "j1", "c1");
        let report = rescore(&c, "j1", None).unwrap();
        assert_eq!(report.ungraded, 1);
        assert_eq!(points_of(&c, "c1"), None, "no ad and no criteria means no total");
        assert!(ranks(&c).is_empty());
    }

}
