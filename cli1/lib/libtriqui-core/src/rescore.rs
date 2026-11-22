//! Rescore / invalidation on ranking-input changes (ad, criteria, feedback).
//!
//! The handlers already bump `inputs_version` when an input changes; this module
//! does the expensive part: re-embed the rubric's question texts (the ad and each
//! criterion) in the active space and re-project every candidate's total. A
//! changed ad also freezes/clears the comparator pool because every memoized
//! verdict is keyed on the ad's content hash.

use rusqlite::OptionalExtension;

use crate::runtime::EngineRuntime;

#[derive(Debug, Default)]
pub struct InvalidationResult {
    /// Candidates whose rubric was recomputed.
    pub rescored: i64,
    /// Rubric question vectors (re-)embedded.
    pub embedded: i64,
}

/// Re-embed the rubric inputs for `job_id` and recompute candidate totals.
///
/// `ad_edited` additionally clears the comparator pool (its verdict keys embed
/// the ad hash, so an ad edit invalidates every cached win-rate).
pub fn inputs_changed(
    runtime: &EngineRuntime,
    job_id: &str,
    ad_edited: bool,
) -> Result<InvalidationResult, String> {
    let conn = runtime.open_conn()?;

    let model = resolve_model(runtime, &conn, job_id);
    let mut embedded = 0i64;

    // Re-embed the ad + criterion texts in the active space when an embedder is
    // available. Feedback vectors are learned, not embedded from text, so they
    // are left untouched here.
    if let Some(model) = &model {
        if let Ok(embedder) = runtime.embedder() {
            // Job ad.
            if let Ok(Some(ad_text)) = conn
                .query_row(
                    "SELECT job_ad_text FROM jobs WHERE id = ?1",
                    rusqlite::params![job_id],
                    |r| r.get::<_, Option<String>>(0),
                )
                .map_err(|e| e.to_string())
            {
                if !ad_text.trim().is_empty() {
                    if let Ok(vec) = embedder.embed(&[&ad_text]) {
                        crate::db::put_rubric_vector(&conn, job_id, "ad", model, &vec)?;
                        embedded += 1;
                    }
                }
            }
    // Criteria.
    let crits: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, text FROM criteria WHERE job_id = ?1")
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
            for (cid, text) in crits {
                if text.trim().is_empty() {
                    continue;
                }
                if let Ok(vec) = embedder.embed(&[&text]) {
                    crate::db::put_rubric_vector(&conn, job_id, &cid, model, &vec)?;
                    embedded += 1;
                }
            }
        }

        // Feedback rebuild: learn (or clear) the feedback rubric vectors in this
        // space. It reads only already-stored centroids / profile vectors, so it
        // runs whenever a model resolves — inside `if let Some(model)` but
        // outside the `if let Ok(embedder)` block above, because `embedder()` is
        // a stub that always errors and feedback learning must not depend on it.
        // It must run before `sort_job` below so the re-score reads the new rows.
        rebuild_feedback_vectors(&conn, job_id, model)?;
    }

    if ad_edited {
        crate::db::clear_comparator_pool(&conn, job_id)?;
    }

    // Recompute every candidate's total against the current rubric.
    let outcome = crate::sort::sort_job(runtime, job_id)?;
    let rescored = outcome.ranked as i64;

    Ok(InvalidationResult { rescored, embedded })
}

/// (Re)build the feedback rubric vectors for `job_id` in `model`:
///
/// * `fb_job` — the component-wise mean of the liked candidates' centroid
///   vectors in this space (the candidate's own centroid when exactly one is
///   liked). When no liked candidate has a centroid here, the vector is cleared
///   so its contribution becomes zero. Written to both `job_feedback_vectors`
///   (the per-model learning store) and `rubric_vectors` (the single read path
///   `sort_job` scores against).
/// * `fb_profile` — the global profile vector (`profile_vectors` for
///   `profile_default` in this model) when one exists, otherwise cleared. This
///   change mirrors an existing vector; it does not learn `eprofile`.
pub fn rebuild_feedback_vectors(
    conn: &rusqlite::Connection,
    job_id: &str,
    model: &str,
) -> Result<(), String> {
    // fb_job from liked candidates' centroids. Guard the empty case: an empty id
    // slice must mean "no likes" (clear), not "all candidates" — the latter is
    // what `candidates_with_centroids` returns for an empty/`None` filter.
    let liked = crate::db::liked_candidate_ids(conn, job_id)?;
    let mean = if liked.is_empty() {
        None
    } else {
        let centroids = crate::db::candidates_with_centroids(conn, job_id, model, Some(&liked))?;
        let vectors: Vec<Vec<f32>> = centroids
            .iter()
            .map(|(_, bytes, _)| crate::embed::bytes_to_vector(bytes))
            .collect();
        crate::db::mean_vector(&vectors)
    };
    match mean {
        Some(mean) => {
            crate::db::put_job_feedback_vector(conn, job_id, model, &mean)?;
            crate::db::put_rubric_vector(conn, job_id, "fb_job", model, &mean)?;
        }
        None => {
            crate::db::clear_job_feedback_vector(conn, job_id, model)?;
            crate::db::clear_rubric_vector(conn, job_id, "fb_job", model)?;
        }
    }

    // fb_profile mirrors the global profile vector, else cleared.
    let profile: Option<Vec<u8>> = conn
        .query_row(
            "SELECT eprofile FROM profile_vectors
             WHERE profile_id = 'profile_default' AND model = ?1 AND eprofile IS NOT NULL",
            rusqlite::params![model],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    match profile {
        Some(bytes) => {
            let vec = crate::embed::bytes_to_vector(&bytes);
            crate::db::put_rubric_vector(conn, job_id, "fb_profile", model, &vec)?;
        }
        None => {
            crate::db::clear_rubric_vector(conn, job_id, "fb_profile", model)?;
        }
    }

    Ok(())
}

fn resolve_model(rt: &EngineRuntime, conn: &rusqlite::Connection, job_id: &str) -> Option<String> {
    if let Ok(e) = rt.embedder() {
        return Some(e.model_id().to_string());
    }
    // No live embedder: resolve the active space from the stored vectors — the
    // rubric vectors first, then the candidates' centroids (same fallback
    // `sort::resolve_model` uses). Feedback learning needs only stored centroids,
    // so it can run even before any rubric vector exists.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{open_single_connection_at, put_rubric_vector};
    use crate::runtime::{EngineRuntime, RuntimeConfig};
    use rusqlite::Connection;

    fn temp_db() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("triqui-rescore-fb-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn insert_candidate(
        conn: &Connection,
        id: &str,
        job_id: &str,
        liked: bool,
        model: &str,
        vector: &[f32],
    ) {
        conn.execute(
            "INSERT INTO candidates
               (id, job_id, source_file_path, file_name, file_type, extraction_status,
                extracted_text, vector, vector_model, liked)
             VALUES (?1, ?2, ?3, ?4, 'txt', 'Extracted', 'text', ?5, ?6, ?7)",
            rusqlite::params![
                id,
                job_id,
                format!("/tmp/{id}.txt"),
                format!("{id}.txt"),
                crate::embed::vector_to_bytes(vector),
                model,
                liked as i32
            ],
        )
        .unwrap();
    }

    fn rubric_vector_count(conn: &Connection, job_id: &str, question_id: &str) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM rubric_vectors WHERE job_id = ?1 AND question_id = ?2",
            rusqlite::params![job_id, question_id],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn fb_job_points(conn: &Connection, candidate_id: &str) -> f64 {
        conn.query_row(
            "SELECT points FROM candidate_scores WHERE candidate_id = ?1 AND question_id = 'fb_job'",
            rusqlite::params![candidate_id],
            |r| r.get(0),
        )
        .unwrap()
    }

    // 2.1 / 2.2 — a liked candidate makes `inputs_changed` write an fb_job rubric
    // vector and positive fb_job points, even though `EngineRuntime::embedder()`
    // always errors (learning is centroid-based, not embedder-based); unliking all
    // clears the fb_job vector and drops its points to zero.
    #[test]
    fn inputs_changed_learns_fb_job_on_like_and_clears_when_none_liked() {
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute(
                "INSERT INTO jobs (id, title, fb_job_weight) VALUES ('job1', 'Job', 10.0)",
                [],
            )
            .unwrap();
            // Anchor the active space so resolve_model finds it without an embedder.
            put_rubric_vector(&conn, "job1", "ad", "model-a", &[1.0, 0.0]).unwrap();
            insert_candidate(&conn, "c1", "job1", false, "model-a", &[1.0, 0.0]);
            insert_candidate(&conn, "c2", "job1", false, "model-a", &[0.0, 1.0]);
        }

        let rt = EngineRuntime::new(RuntimeConfig::discover(None, None), db_path.clone());

        // Like c1, then re-project.
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute("UPDATE candidates SET liked = 1 WHERE id = 'c1'", [])
                .unwrap();
        }
        inputs_changed(&rt, "job1", false).unwrap();

        {
            let conn = open_single_connection_at(&path).unwrap();
            assert_eq!(rubric_vector_count(&conn, "job1", "fb_job"), 1);
            assert!(fb_job_points(&conn, "c1") > 0.0);
        }

        // Unlike everyone, then re-project: fb_job is removed and points fall to 0.
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute("UPDATE candidates SET liked = 0", []).unwrap();
        }
        inputs_changed(&rt, "job1", false).unwrap();

        {
            let conn = open_single_connection_at(&path).unwrap();
            assert_eq!(rubric_vector_count(&conn, "job1", "fb_job"), 0);
            assert_eq!(fb_job_points(&conn, "c1"), 0.0);
        }

        let _ = std::fs::remove_file(&path);
    }

    // 2.1 — feedback rebuild resolves the space from candidates alone: with no
    // `rubric_vectors` row and an always-erroring embedder, a liked candidate
    // still produces an fb_job rubric vector.
    #[test]
    fn inputs_changed_resolves_model_from_candidates_without_rubric_vectors() {
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute(
                "INSERT INTO jobs (id, title, fb_job_weight) VALUES ('job1', 'Job', 10.0)",
                [],
            )
            .unwrap();
            // No 'ad' rubric vector: the space must resolve from candidates.
            insert_candidate(&conn, "c1", "job1", true, "model-a", &[1.0, 0.0]);
            insert_candidate(&conn, "c2", "job1", false, "model-a", &[0.0, 1.0]);
        }

        let rt = EngineRuntime::new(RuntimeConfig::discover(None, None), db_path.clone());
        inputs_changed(&rt, "job1", false).unwrap();

        {
            let conn = open_single_connection_at(&path).unwrap();
            assert_eq!(rubric_vector_count(&conn, "job1", "fb_job"), 1);
            assert!(fb_job_points(&conn, "c1") > 0.0);
        }

        let _ = std::fs::remove_file(&path);
    }

    // 2.3 — fb_profile mirrors the global profile vector when a profile_vectors
    // row exists for the space, and is cleared when it does not.
    #[test]
    fn inputs_changed_mirrors_and_clears_fb_profile() {
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute(
                "INSERT INTO jobs (id, title, fb_profile_weight) VALUES ('job1', 'Job', 5.0)",
                [],
            )
            .unwrap();
            put_rubric_vector(&conn, "job1", "ad", "model-a", &[1.0, 0.0]).unwrap();
            insert_candidate(&conn, "c1", "job1", false, "model-a", &[1.0, 0.0]);
            conn.execute(
                "INSERT INTO profile_vectors (profile_id, model, eprofile)
                 VALUES ('profile_default', 'model-a', ?1)",
                rusqlite::params![crate::embed::vector_to_bytes(&[0.0, 1.0])],
            )
            .unwrap();
        }

        let rt = EngineRuntime::new(RuntimeConfig::discover(None, None), db_path.clone());

        // Profile vector present → fb_profile rubric vector is written.
        inputs_changed(&rt, "job1", false).unwrap();
        {
            let conn = open_single_connection_at(&path).unwrap();
            assert_eq!(rubric_vector_count(&conn, "job1", "fb_profile"), 1);
        }

        // Remove the profile vector → fb_profile rubric vector is cleared.
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute("DELETE FROM profile_vectors WHERE profile_id = 'profile_default'", [])
                .unwrap();
        }
        inputs_changed(&rt, "job1", false).unwrap();
        {
            let conn = open_single_connection_at(&path).unwrap();
            assert_eq!(rubric_vector_count(&conn, "job1", "fb_profile"), 0);
        }

        let _ = std::fs::remove_file(&path);
    }
}