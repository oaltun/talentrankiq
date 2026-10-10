//! Feedback-learning operations (no Tauri).
//!
//! [`apply_feedback_blocking`] consumes a staged reorder + like set: it updates the two
//! learned vectors (`ejobfeedback` per-job, `eprofile` cross-job) per
//! `ranker::compute_feedback_delta`, persists the new liked flags, then re-ranks.
//! [`reset_feedback_blocking`] zeroes the learned state.
//! [`list_liked_candidate_ids`] seeds the frontend staging store.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rusqlite::Connection;

use crate::db::db_err;
use crate::embed::{add_weighted, bytes_to_vector, normalize, vector_to_bytes};
use crate::embeddings::EmbedProgress;
use crate::ranker;
use crate::runtime::EngineRuntime;
use crate::types::{DomainError, TriqiItem};

fn in_clause(ids: &[String]) -> String {
    ids.iter().map(|_| "?").collect::<Vec<_>>().join(",")
}

/// Persist the desired liked set: clear every liked flag for the job, then set
/// the given ids. Likes are plain rows — they must survive a restart even when
/// there is nothing embedded to learn from yet.
fn persist_liked_flags(
    conn: &Connection,
    job_id: &str,
    liked_candidate_ids: &[String],
) -> Result<(), DomainError> {
    conn.execute(
        "UPDATE candidates SET liked = 0 WHERE job_id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(db_err)?;
    if !liked_candidate_ids.is_empty() {
        let placeholders = in_clause(liked_candidate_ids);
        let sql =
            format!("UPDATE candidates SET liked = 1 WHERE job_id = ? AND id IN ({placeholders})");
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&job_id];
        for id in liked_candidate_ids {
            params.push(id);
        }
        conn.execute(&sql, params.as_slice()).map_err(db_err)?;
    }
    Ok(())
}

/// Apply staged feedback (a reordered triqui + a desired liked-id set) to the
/// learned vectors, persist liked flags, and re-project. Blocking — call via a
/// `spawn_blocking` wrapper.
///
/// **Learned in every space, re-projected, never re-sorted.** Two changes from
/// before scoring v3:
///
/// - The vectors are learned in *every* space the job has candidate vectors for —
///   the scout's as well as the heavy embedder's — so a like moves a scout-space
///   candidate's total too. Learning only in the heavy space meant a CV the heavy
///   embedder had not reached was unmoved by feedback.
/// - The tail is `rescore`, not a blocking sort. No candidate vector changed, so
///   nothing needs re-embedding and the judge is untouched (it never read the
///   feedback rows); the whole update is a cosine pass. On a 2000-CV job that is
///   the difference between milliseconds and hours.
///
/// `progress` is unused and kept for signature stability: rescore makes no model
/// calls, so there is nothing to report. A caller that wants a progress bar is
/// reporting on the cascade, not on this.
pub fn apply_feedback_blocking(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    ordered_candidate_ids: Vec<String>,
    liked_candidate_ids: Vec<String>,
    _progress: EmbedProgress<'_>,
) -> Result<Vec<TriqiItem>, DomainError> {
    let conn = rt
        .open_conn()
        .map_err(|e| DomainError::DbError { message: e })?;

    // Every id referenced (triqui reorder + likes).
    let mut id_set: Vec<String> = Vec::new();
    for id in ordered_candidate_ids
        .iter()
        .chain(liked_candidate_ids.iter())
    {
        if !id_set.contains(id) {
            id_set.push(id.clone());
        }
    }

    let rank_of: HashMap<String, i32> = {
        let mut out = HashMap::new();
        if !id_set.is_empty() {
            let placeholders = in_clause(&id_set);
            let sql = format!(
                "SELECT id, rank FROM candidates WHERE id IN ({placeholders}) AND job_id = ?"
            );
            let mut params: Vec<&dyn rusqlite::ToSql> =
                id_set.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            params.push(&job_id);
            let mut stmt = conn.prepare(&sql).map_err(db_err)?;
            let rows = stmt
                .query_map(params.as_slice(), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Option<i32>>(1)?))
                })
                .map_err(db_err)?;
            for r in rows {
                let (id, rank) = r.map_err(db_err)?;
                if let Some(rk) = rank {
                    out.insert(id, rk);
                }
            }
        }
        out
    };

    // Stored liked set for the job (to diff against the desired set).
    let stored_liked: HashSet<String> = {
        let mut stmt = conn
            .prepare("SELECT id FROM candidates WHERE job_id = ?1 AND liked = 1")
            .map_err(db_err)?;
        let rows = stmt
            .query_map(rusqlite::params![&job_id], |r| r.get::<_, String>(0))
            .map_err(db_err)?;
        let mut s = HashSet::new();
        for r in rows {
            s.insert(r.map_err(db_err)?);
        }
        s
    };

    let valid: Vec<String> = ordered_candidate_ids
        .iter()
        .filter(|id| rank_of.contains_key(*id))
        .cloned()
        .collect();
    let mut baseline_order = valid.clone();
    baseline_order.sort_by_key(|id| rank_of.get(id).copied().unwrap_or(i32::MAX));
    let liked_set: HashSet<&String> = liked_candidate_ids.iter().collect();
    let has_signal = valid != baseline_order
        || liked_candidate_ids.iter().any(|id| !stored_liked.contains(id))
        || stored_liked.iter().any(|id| !liked_set.contains(id));
    let has_signal = has_signal && !id_set.is_empty();

    if has_signal {
        enable_feedback_weights(&conn, job_id)?;
    }

    // Learn in every space the job has vectors in. The heavy space and the scout's
    // are independent learnings over independent vectors; neither clobbers the other.
    let spaces = learnable_spaces(&conn, job_id, &rt.model())?;
    let mut learned_any = false;
    for space in &spaces {
        let vectors = candidate_vectors_in_space(&conn, job_id, &id_set, space)?;
        let dim = vectors.values().next().map(|v| v.len()).unwrap_or(0);
        if dim == 0 {
            continue;
        }
        let baseline: Vec<(&str, &[f32])> = baseline_order
            .iter()
            .filter_map(|id| vectors.get(id).map(|v| (id.as_str(), v.as_slice())))
            .collect();
        let staged: Vec<(&str, &[f32])> = valid
            .iter()
            .filter_map(|id| vectors.get(id).map(|v| (id.as_str(), v.as_slice())))
            .collect();
        let added_vecs: Vec<&[f32]> = liked_candidate_ids
            .iter()
            .filter(|id| !stored_liked.contains(*id))
            .filter_map(|id| vectors.get(id).map(|v| v.as_slice()))
            .collect();
        let removed_vecs: Vec<&[f32]> = stored_liked
            .iter()
            .filter(|id| !liked_set.contains(id))
            .filter_map(|id| vectors.get(id).map(|v| v.as_slice()))
            .collect();

        let delta = ranker::compute_feedback_delta(
            &baseline, &staged, &added_vecs, &removed_vecs, dim,
        );
        let mut efb = learned_feedback(&conn, job_id, space, dim, true)
            .unwrap_or_else(|| vec![0.0; dim]);
        let mut eprof = learned_feedback(&conn, job_id, space, dim, false)
            .unwrap_or_else(|| vec![0.0; dim]);
        add_weighted(&mut efb, &delta.ejobfeedback, 1.0);
        add_weighted(&mut eprof, &delta.eprofile, 1.0);
        normalize(&mut efb);
        normalize(&mut eprof);
        persist_feedback(&conn, job_id, space, &efb, &eprof)?;
        learned_any = true;
    }

    // The star must survive a restart even when nothing was embedded enough to
    // learn from, so the flags are persisted unconditionally.
    persist_liked_flags(&conn, job_id, &liked_candidate_ids)?;
    drop(conn);

    if !learned_any {
        log::info!("feedback: job {job_id} has no vectors in any space — flags persisted, nothing learned");
    }
    // The learned rows are new *questions* for the projection: a vector alone does
    // not put a column on the sheet, the grade does. Re-score every space that
    // already grades this job (a cosine pass over stored chunks), then project.
    {
        let conn = rt
            .open_conn()
            .map_err(|e| DomainError::DbError { message: e })?;
        let heavy = rt.model();
        let scout_space = crate::scout::scout_model_for(&heavy);
        crate::rescore::score_live_spaces(&conn, job_id, &heavy, scout_space.as_deref())?;
    }
    let report = crate::rescore::rescore(rt, job_id)?;
    log::info!(
        "feedback: job {job_id} rescored ({} candidates, {} judged) across {} space(s)",
        report.scored,
        report.judged,
        spaces.len()
    );
    let conn = rt
        .open_conn()
        .map_err(|e| DomainError::DbError { message: e })?;
    let (triqui, _) = crate::rescore::read_triqui(&conn, job_id)?;
    Ok(triqui)
}

/// The spaces a job can learn feedback in: the selected embedder plus the scout
/// (when it has graded anything). Both are derived from stored vectors, never from
/// the engine, so this costs two indexed queries.
fn learnable_spaces(
    conn: &Connection,
    job_id: &str,
    heavy: &str,
) -> Result<Vec<String>, DomainError> {
    let mut spaces = vec![heavy.to_string()];
    // A space is learnable when this job has candidate vectors in it. The scout's
    // chunks live under its own slug, so one job-scoped probe finds it.
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT cc.vector_model FROM candidate_chunks cc
             JOIN candidates c ON c.id = cc.candidate_id
             WHERE c.job_id = ?1 AND cc.vector_model != ?2",
        )
        .map_err(db_err)?;
    let others = stmt
        .query_map(rusqlite::params![job_id, heavy], |r| r.get::<_, String>(0))
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;
    for s in others {
        if !spaces.contains(&s) {
            spaces.push(s);
        }
    }
    Ok(spaces)
}

/// The candidates' centroid vectors in one space (the scout has no centroid
/// column, so it is computed from its chunks).
fn candidate_vectors_in_space(
    conn: &Connection,
    job_id: &str,
    ids: &[String],
    space: &str,
) -> Result<HashMap<String, Vec<f32>>, DomainError> {
    let mut out = HashMap::new();
    for id in ids {
        if let Some(v) = crate::rubric::centroid_in_space(conn, id, space)? {
            out.insert(id.clone(), v.into_iter().map(|x| x as f32).collect());
        }
    }
    let _ = job_id;
    Ok(out)
}

/// One learned vector for `(job, space)`, dimension-checked so a stale blob from a
/// different model is treated as absent rather than mixed in.
fn learned_feedback(
    conn: &Connection,
    job_id: &str,
    space: &str,
    dim: usize,
    job_scoped: bool,
) -> Option<Vec<f32>> {
    let blob: Option<Vec<u8>> = if job_scoped {
        conn.query_row(
            "SELECT ejobfeedback FROM job_feedback_vectors WHERE job_id = ?1 AND model = ?2",
            rusqlite::params![job_id, space],
            |r| r.get(0),
        )
        .ok()
    } else {
        conn.query_row(
            "SELECT eprofile FROM profile_vectors WHERE profile_id = 'profile_default' AND model = ?1",
            rusqlite::params![space],
            |r| r.get(0),
        )
        .ok()
    };
    blob.and_then(|b| bytes_to_vector(&b, Some(dim)).ok())
        .filter(|v| v.len() == dim)
}

/// Persist a space's learned vectors, and mirror them into `rubric_vectors` so
/// rescore reads every question's vector from one place.
fn persist_feedback(
    conn: &Connection,
    job_id: &str,
    space: &str,
    efb: &[f32],
    eprof: &[f32],
) -> Result<(), DomainError> {
    let efb_bytes = vector_to_bytes(efb);
    let eprof_bytes = vector_to_bytes(eprof);
    conn.execute(
        "INSERT INTO job_feedback_vectors (job_id, model, ejobfeedback, updated_at)
         VALUES (?1, ?2, ?3, datetime('now'))
         ON CONFLICT(job_id, model) DO UPDATE SET
           ejobfeedback = excluded.ejobfeedback, updated_at = excluded.updated_at",
        rusqlite::params![job_id, space, &efb_bytes],
    )
    .map_err(db_err)?;
    conn.execute(
        "INSERT INTO profile_vectors (profile_id, model, eprofile, updated_at)
         VALUES ('profile_default', ?1, ?2, datetime('now'))
         ON CONFLICT(profile_id, model) DO UPDATE SET
           eprofile = excluded.eprofile, updated_at = excluded.updated_at",
        rusqlite::params![space, &eprof_bytes],
    )
    .map_err(db_err)?;
    let put = |qid: &str, v: &[f32]| {
        crate::db::put_rubric_vector(conn, job_id, qid, space, v)
            .map_err(|e| DomainError::DbError { message: e })
    };
    put(crate::rubric::FB_JOB_Q, efb)?;
    // The profile row is global, but only this job's read path matters right now,
    // so it is copied into this job's rubric rather than into every job's on every
    // like — which would be a surprising write amplification.
    put(crate::rubric::FB_PROFILE_Q, eprof)?;
    Ok(())
}

/// Turn the two feedback rows on at their configured weights the first time the
/// user gives any signal.
///
/// The weights come from `ikapp.yaml rubric_defaults`, like every other default in
/// the system — they used to be hard-coded constants here, which meant a config
/// that said `fb_job_points: 10` still got 40 on first feedback.
fn enable_feedback_weights(conn: &Connection, job_id: &str) -> Result<(), DomainError> {
    let (fb_job_w, fb_prof_w): (f64, f64) = conn
        .query_row(
            "SELECT fb_job_weight, fb_profile_weight FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(db_err)?;
    let defaults = crate::app_config::rubric_defaults();
    let (new_job, new_prof) = (
        if fb_job_w == 0.0 {
            defaults.fb_job_points
        } else {
            fb_job_w
        },
        if fb_prof_w == 0.0 {
            defaults.fb_profile_points
        } else {
            fb_prof_w
        },
    );
    if new_job != fb_job_w || new_prof != fb_prof_w {
        conn.execute(
            "UPDATE jobs SET fb_job_weight = ?1, fb_profile_weight = ?2, updated_at = datetime('now')
             WHERE id = ?3",
            rusqlite::params![new_job, new_prof, job_id],
        )
        .map_err(db_err)?;
    }
    Ok(())
}

/// Zero the current model's learned feedback vectors + clear liked flags, then
/// re-rank. Other models' learned vectors are kept — feedback is per model.
/// Blocking — call via a spawn_blocking wrapper.
pub fn reset_feedback_blocking(
    rt: &Arc<EngineRuntime>,
    job_id: &str,
    _progress: EmbedProgress<'_>,
) -> Result<Vec<TriqiItem>, DomainError> {
    let conn = rt
        .open_conn()
        .map_err(|e| DomainError::DbError { message: e })?;
    // Learnings are per space; the slug alone keys them — no engine needed. Every
    // space the job learned in is cleared, and the copies in `rubric_vectors` with
    // them, so a stale vector can never be read back in.
    let spaces: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT model FROM job_feedback_vectors WHERE job_id = ?1")
            .map_err(db_err)?;
        let v = stmt
            .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
            .map_err(db_err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(db_err)?;
        v
    };
    for space in spaces {
        conn.execute(
            "DELETE FROM job_feedback_vectors WHERE job_id = ?1 AND model = ?2",
            rusqlite::params![job_id, &space],
        )
        .map_err(db_err)?;
        conn.execute(
            "DELETE FROM profile_vectors WHERE profile_id = 'profile_default' AND model = ?1",
            rusqlite::params![&space],
        )
        .map_err(db_err)?;
        conn.execute(
            "DELETE FROM rubric_vectors
             WHERE job_id = ?1 AND question_id IN ('fb_job', 'fb_profile') AND model = ?2",
            rusqlite::params![job_id, &space],
        )
        .map_err(db_err)?;
        conn.execute(
            "DELETE FROM stage_scores
             WHERE question_id IN ('fb_job', 'fb_profile') AND model = ?2
               AND candidate_id IN (SELECT id FROM candidates WHERE job_id = ?1)",
            rusqlite::params![job_id, &space],
        )
        .map_err(db_err)?;
    }
    // Clearing = persisting the empty desired set.
    persist_liked_flags(&conn, job_id, &[])?;
    drop(conn);

    crate::rescore::rescore(rt, job_id)?;
    let conn = rt
        .open_conn()
        .map_err(|e| DomainError::DbError { message: e })?;
    let (triqui, _) = crate::rescore::read_triqui(&conn, job_id)?;
    Ok(triqui)
}

/// All liked candidate ids for a job (seeds the frontend staging store).
pub fn list_liked_candidate_ids(
    conn: &Connection,
    job_id: &str,
) -> Result<Vec<String>, DomainError> {
    let mut stmt = conn
        .prepare("SELECT id FROM candidates WHERE job_id = ?1 AND liked = 1")
        .map_err(db_err)?;
    let ids: Vec<String> = stmt
        .query_map(rusqlite::params![&job_id], |r| r.get::<_, String>(0))
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{migrate, SCHEMA};
    use rusqlite::Connection;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(SCHEMA).expect("schema");
        migrate(&conn).expect("migrate");
        conn
    }

    fn insert_job(conn: &Connection, id: &str) {
        conn.execute(
            "INSERT INTO jobs (id, title) VALUES (?1, ?2)",
            rusqlite::params![id, "test job"],
        )
        .unwrap();
    }

    fn insert_candidate(conn: &Connection, job_id: &str, id: &str, liked: bool) {
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type,
                                     extraction_status, liked)
             VALUES (?1, ?2, ?3, ?4, 'Txt', 'Extracted', ?5)",
            rusqlite::params![id, job_id, format!("/tmp/{id}"), id, liked as i64],
        )
        .unwrap();
    }

    fn liked_ids(conn: &Connection, job_id: &str) -> Vec<String> {
        list_liked_candidate_ids(conn, job_id).unwrap()
    }

    #[test]
    fn persist_liked_flags_replaces_the_whole_set() {
        let conn = test_conn();
        insert_job(&conn, "j1");
        insert_job(&conn, "j2");
        insert_candidate(&conn, "j1", "c1", true);
        insert_candidate(&conn, "j1", "c2", false);
        insert_candidate(&conn, "j1", "c3", false);
        insert_candidate(&conn, "j2", "c_other_job", true); // untouched

        persist_liked_flags(&conn, "j1", &["c3".to_string()]).unwrap();

        assert_eq!(liked_ids(&conn, "j1"), vec!["c3".to_string()]);
        assert_eq!(liked_ids(&conn, "j2"), vec!["c_other_job".to_string()]);
    }

    #[test]
    fn persist_liked_flags_with_empty_set_clears_every_star() {
        let conn = test_conn();
        insert_job(&conn, "j1");
        insert_candidate(&conn, "j1", "c1", true);
        insert_candidate(&conn, "j1", "c2", true);

        persist_liked_flags(&conn, "j1", &[]).unwrap();

        assert!(liked_ids(&conn, "j1").is_empty());
    }
}
