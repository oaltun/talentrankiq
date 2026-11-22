//! Embedding pre-warming helpers.

use std::sync::Arc;

use crate::runtime::EngineRuntime;
use rusqlite::params;

/// Pre-warm all rubric inputs for a set of jobs: embed the ad + each criterion
/// into `rubric_vectors` for the active model.
pub fn prewarm_all_rubric_inputs(
    runtime: &Arc<EngineRuntime>,
    job_ids: &[String],
) -> Result<(), String> {
    for job_id in job_ids {
        prewarm_job_blocking(runtime, job_id)?;
    }
    Ok(())
}

/// Pre-warm rubric inputs for a single job (blocking).
/// Embeds the job's ad text (if non-empty and ad weight is positive) and each
/// criterion's text into `rubric_vectors` under the active model.
pub fn prewarm_job_blocking(runtime: &Arc<EngineRuntime>, job_id: &str) -> Result<(), String> {
    let conn = runtime.open_conn()?;

    let embedder = runtime.embedder()?;
    let model = embedder.model_id().to_string();

    // Job ad.
    let ad_text: Option<String> = conn
        .query_row(
            "SELECT job_ad_text FROM jobs WHERE id = ?1",
            params![job_id],
            |r| r.get::<_, Option<String>>(0),
        )
        .ok()
        .flatten();
    if let Some(text) = ad_text {
        if !text.trim().is_empty() {
            if let Ok(vec) = embedder.embed(&[&text]) {
                crate::db::put_rubric_vector(&conn, job_id, "ad", &model, &vec)?;
            }
        }
    }

    // Criteria.
    let crits: Vec<(String, String)> = {
        let mut stmt = conn.prepare("SELECT id, text FROM criteria WHERE job_id = ?1").map_err(|e| e.to_string())?;
        let mut iter = stmt.query_map(params![job_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        }).map_err(|e| e.to_string())?;
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
            crate::db::put_rubric_vector(&conn, job_id, &cid, &model, &vec)?;
        }
    }

    Ok(())
}
