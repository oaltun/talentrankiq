//! "Ensure inputs" barrier for the scoring join.
//!
//! Ranking joins `candidates.vector` against `rubric_vectors` (ad + criteria +
//! feedback). Both sides must exist in the active model's space before scoring;
//! when either side is missing the run still "succeeds" but every total is zero.
//!
//! `ensure_inputs` supplies the missing side using the existing helpers: it
//! pre-warms the ad + criteria (`embeddings::prewarm_job_blocking`), re-embeds
//! candidates whose centroid is absent or belongs to another space from their
//! durable `extracted_text`, and rebuilds the learned feedback vectors
//! (`rescore::rebuild_feedback_vectors`).
//!
//! It is idempotent: only gaps are filled, so re-running leaves existing
//! vectors untouched.

use std::sync::Arc;

use rusqlite::params;

use crate::embed::Embedder;
use crate::runtime::EngineRuntime;

/// Guarantee the job's scoring inputs exist in the active model's space.
///
/// With no embedder available the embedding part is a no-op (stored vectors are
/// used as-is), but feedback vectors are still rebuilt whenever a model can be
/// resolved from the stored vectors.
pub fn ensure_inputs(runtime: &Arc<EngineRuntime>, job_id: &str) -> Result<(), String> {
    let conn = runtime.open_conn()?;

    let embedder = runtime.embedder().ok();
    let model = match &embedder {
        Some(e) => Some(e.model_id().to_string()),
        None => resolve_model(&conn, job_id),
    };

    if let Some(embedder) = &embedder {
        crate::embeddings::prewarm_job_blocking(runtime, job_id)?;
        reembed_missing_candidates(&conn, job_id, embedder.as_ref(), embedder.model_id())?;
    }

    if let Some(model) = &model {
        crate::rescore::rebuild_feedback_vectors(&conn, job_id, model)?;
    }

    Ok(())
}

/// Re-embed every extracted candidate of `job_id` whose centroid is missing or
/// stored in another space, persisting `vector` / `vector_model`. Candidates
/// that already hold a centroid in `model` are left untouched.
fn reembed_missing_candidates(
    conn: &rusqlite::Connection,
    job_id: &str,
    embedder: &dyn Embedder,
    model: &str,
) -> Result<(), String> {
    let pending: Vec<(String, String)> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, extracted_text FROM candidates
                 WHERE job_id = ?1 AND extraction_status = 'Extracted' AND deleted = 0
                   AND extracted_text IS NOT NULL AND TRIM(extracted_text) <> ''
                   AND (vector IS NULL OR vector_model IS NOT ?2)
                 ORDER BY imported_at, id",
            )
            .map_err(|e| e.to_string())?;
        let mut iter = stmt
            .query_map(params![job_id, model], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        let mut v = Vec::new();
        while let Some(r) = iter.next().transpose().map_err(|e| e.to_string())? {
            v.push(r);
        }
        v
    };

    for (candidate_id, text) in pending {
        let vector = match embedder.embed(&[&text]) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("ensure: failed to embed candidate {candidate_id}: {e}");
                continue;
            }
        };
        conn.execute(
            "UPDATE candidates SET vector = ?1, vector_model = ?2 WHERE id = ?3 AND job_id = ?4",
            params![
                crate::embed::vector_to_bytes(&vector),
                model,
                candidate_id,
                job_id
            ],
        )
        .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Resolve the active space from the stored vectors when no embedder is live:
/// rubric vectors first, then candidate centroids (the same fallback
/// `cascade`/`rescore` use).
fn resolve_model(conn: &rusqlite::Connection, job_id: &str) -> Option<String> {
    conn.query_row(
        "SELECT model FROM rubric_vectors WHERE job_id = ?1 LIMIT 1",
        params![job_id],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .or_else(|| {
        conn.query_row(
            "SELECT vector_model FROM candidates WHERE job_id = ?1 AND vector_model IS NOT NULL LIMIT 1",
            params![job_id],
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
use std::path::PathBuf;
use std::sync::Arc;


    fn temp_db() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("triqui-ensure-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn runtime_for(db_path: PathBuf) -> Arc<EngineRuntime> {
        EngineRuntime::new(RuntimeConfig::discover(None, None), db_path)
    }

    fn job_and_text(conn: &Connection, job_id: &str) {
        conn.execute("INSERT INTO jobs (id, title, ad_weight) VALUES (?1, 'Job', 10.0)", [job_id])
            .unwrap();
        put_rubric_vector(conn, job_id, "ad", "model-a", &[1.0, 0.0]).unwrap();
    }

    fn candidate_null_vector(conn: &Connection, cid: &str, job_id: &str, text: &str) {
        conn.execute(
            "INSERT INTO candidates
               (id, job_id, source_file_path, file_name, file_type, extraction_status,
                extracted_text, vector, vector_model, liked)
             VALUES (?1, ?2, '/tmp/x.txt', 'x.txt', 'txt', 'Extracted', ?3, NULL, NULL, 0)",
            rusqlite::params![cid, job_id, text],
        )
        .unwrap();
    }

    fn candidate_vector(conn: &Connection, cid: &str) -> Option<Vec<u8>> {
        conn.query_row(
            "SELECT vector FROM candidates WHERE id = ?1",
            rusqlite::params![cid],
            |r| r.get::<_, Option<Vec<u8>>>(0),
        )
        .unwrap()
    }

    fn vector_model(conn: &Connection, cid: &str) -> Option<String> {
        conn.query_row(
            "SELECT vector_model FROM candidates WHERE id = ?1",
            rusqlite::params![cid],
            |r| r.get::<_, Option<String>>(0),
        )
        .unwrap()
    }

    fn fb_job_count(conn: &Connection, job_id: &str) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM rubric_vectors WHERE job_id = ?1 AND question_id = 'fb_job'",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .unwrap()
    }

    // 1.2 — without a live embedder the barrier is a no-op for embedding: an
    // extracted candidate with a missing centroid keeps a NULL vector, but a
    // model still resolves from stored rubric vectors so feedback is rebuilt.
    #[test]
    fn ensure_inputs_no_op_embedding_without_embedder() {
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            job_and_text(&conn, "job1");
            candidate_null_vector(&conn, "c1", "job1", "candidate one text");
        }

        let rt = runtime_for(db_path.clone());
        ensure_inputs(&rt, "job1").unwrap();

        {
            let conn = open_single_connection_at(&path).unwrap();
            // No embedder was available, so the missing centroid was not filled.
            assert!(candidate_vector(&conn, "c1").is_none());
            assert_eq!(vector_model(&conn, "c1"), None);
            // A model resolved from stored vectors, so feedback rebuild ran.
            assert_eq!(fb_job_count(&conn, "job1"), 0); // no likes → cleared row
        }

        let _ = std::fs::remove_file(&path);
    }

    // 5.1 — with a live embedder the barrier fills a missing candidate centroid
    // from `extracted_text` and persists `vector` / `vector_model`; re-running
    // leaves the existing centroid untouched.
    #[test]
    fn ensure_inputs_fills_and_preserves_candidate_centroid() {
        let slug = "harrier-oss-v1-270m-ONNX-q4f16";
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            job_and_text(&conn, "job1");
            candidate_null_vector(&conn, "c1", "job1", "platform engineer with kubernetes");
        }

        let rt = EngineRuntime::new(
            RuntimeConfig::discover(Some(slug.to_string()), None),
            db_path.clone(),
        );
        rt.switch_model(slug.to_string()).unwrap();
        assert!(rt.embedder().is_ok());

        ensure_inputs(&rt, "job1").unwrap();
        let first = {
            let conn = open_single_connection_at(&path).unwrap();
            (candidate_vector(&conn, "c1").is_some(), vector_model(&conn, "c1"))
        };
        assert!(first.0);
        assert_eq!(first.1, Some(slug.to_string()));

        let before = {
            let conn = open_single_connection_at(&path).unwrap();
            candidate_vector(&conn, "c1").unwrap()
        };
        ensure_inputs(&rt, "job1").unwrap();
        let after = {
            let conn = open_single_connection_at(&path).unwrap();
            candidate_vector(&conn, "c1").unwrap()
        };
        assert_eq!(before, after);

        let _ = std::fs::remove_file(&path);
    }

    // 2.1 / 5.3 — an ad-only job driven through the real cascade with the real
    // embedder gets a non-zero `ad` row and `points_max` equal to the rubric
    // capacity, proving the barrier supplies the missing rubric side before
    // scoring instead of silently producing zeros.
    #[test]
    fn cascade_ad_only_job_scores_nonzero_ad_points() {
        let slug = "harrier-oss-v1-270m-ONNX-q4f16";
        let path = temp_db();
        let db_path = path.clone();
        {
            let conn = open_single_connection_at(&path).unwrap();
            conn.execute(
                "INSERT INTO jobs (id, title, job_ad_text, ad_weight)
                 VALUES ('job1', 'Job',
                         'Senior Platform Engineer. Required: Kubernetes, Go, AWS, Docker.',
                         50.0)",
                [],
            )
            .unwrap();
            for (cid, text) in [
                ("c1", "Platform engineer with Kubernetes, Go, AWS, Docker experience"),
                ("c2", "Marketing manager with brand strategy and social media"),
            ] {
                conn.execute(
                    "INSERT INTO candidates
                       (id, job_id, source_file_path, file_name, file_type,
                        extraction_status, extracted_text)
                     VALUES (?1, 'job1', ?2, ?3, 'txt', 'Extracted', ?4)",
                    rusqlite::params![cid, format!("/tmp/{cid}.txt"), format!("{cid}.txt"), text],
                )
                .unwrap();
            }
        }

        let rt = EngineRuntime::new(
            RuntimeConfig::discover(Some(slug.to_string()), None),
            db_path.clone(),
        );
        rt.switch_model(slug.to_string()).unwrap();

        let cfg = crate::cascade::WaveConfig::default();
        let outcome = crate::cascade::run(&rt, "job1", &cfg, &|| false, &|_, _, _| {}).unwrap();
        assert!(!outcome.cancelled);

        let conn = open_single_connection_at(&path).unwrap();
        let (ad_points, max_points): (f64, f64) = conn
            .query_row(
                "SELECT points, max_points FROM candidate_scores
                 WHERE question_id = 'ad' AND candidate_id = 'c1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert!(ad_points > 0.0, "ad points should be non-zero, got {ad_points}");
        assert!(
            (max_points - 50.0).abs() < 1e-9,
            "points_max should equal the ad capacity (50.0), got {max_points}"
        );

        let _ = std::fs::remove_file(&path);
    }
}
