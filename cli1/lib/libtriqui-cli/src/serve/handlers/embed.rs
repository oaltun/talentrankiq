//! Embedding engine runtime handlers.

use std::sync::Arc;

use serde_json::Value;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::params::ChunkMatchParams;
use libtriqui_client::types::{ChunkMatchResult, ChunkMatchTarget, ChunkSim, DomainError};
use libtriqui_core::app_config;
use libtriqui_core::chunks;
use libtokenizer::TokenizerHandle;

use crate::serve::handlers::{domain, ok_json, string_err};
use crate::serve::ServeState;

pub fn start_runtime(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    state.runtime.start();
    state.runtime.wait_until_ready(std::time::Duration::from_secs(180))
        .map_err(|e| string_err(e))?;
    ok_json(state.runtime.status_info())
}

pub fn runtime_status(state: &Arc<ServeState>) -> Result<Value, ProtocolError> {
    ok_json(state.runtime.status_info())
}

pub fn embed_job(state: &Arc<ServeState>, job_id: &str) -> Result<Value, ProtocolError> {
    let conn = state.db.read_conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND extraction_status = 'Extracted'",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| string_err(e.to_string()))?;
    ok_json(count)
}

/// Match every persisted chunk of a candidate against a target (job ad or a
/// criterion), returning the model slug and each chunk's char range and cosine
/// similarity in document order.
pub fn chunk_match(state: &Arc<ServeState>, p: &ChunkMatchParams) -> Result<Value, ProtocolError> {
    let (job_id, extracted_text) = {
        let conn = state.db.read_conn();
        conn.query_row(
            "SELECT job_id, COALESCE(extracted_text, '') FROM candidates WHERE id = ?1",
            rusqlite::params![p.candidate_id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => domain(&DomainError::NotFound {
                entity: "Candidate".into(),
                id: p.candidate_id.clone(),
            }),
            other => string_err(other.to_string()),
        })?
    };

    let model = {
        let conn = state.db.read_conn();
        resolve_model(state, &conn, &job_id, &p.candidate_id)
    };
    let model = match model {
        Some(m) => m,
        None => {
            return ok_json(ChunkMatchResult {
                model: String::new(),
                chunks: Vec::new(),
            })
        }
    };

    // Lazily persist chunks if this candidate has none for the active model.
    let has_chunks = {
        let conn = state.db.read_conn();
        chunks::has_candidate_chunks(&conn, &p.candidate_id, &model).unwrap_or(false)
    };
    if !has_chunks {
        if let Ok(embedder) = state.runtime.embedder() {
            let tokenizer = {
                let model_dir = app_config::model_dir();
                app_config::tokenizer_path(&model_dir, &model)
                    .and_then(|path| TokenizerHandle::from_file(&path).ok())
            };
            let writer = state.db.writer();
            if let Err(e) = chunks::ensure_candidate_chunks(
                &writer,
                &p.candidate_id,
                &extracted_text,
                embedder.as_ref(),
                &model,
                tokenizer.as_ref(),
            ) {
                log::warn!(
                    "chunk_match: failed to persist chunks for {}: {}",
                    p.candidate_id,
                    e
                );
            }
        }
    }

    let rows = {
        let conn = state.db.read_conn();
        chunks::candidate_chunks_for(&conn, &p.candidate_id, &model).unwrap_or_default()
    };
    if rows.is_empty() {
        return ok_json(ChunkMatchResult {
            model,
            chunks: Vec::new(),
        });
    }

    let is_job_ad = matches!(p.target, ChunkMatchTarget::JobAd);

    let target_text: Option<String> = match &p.target {
        ChunkMatchTarget::JobAd => {
            let conn = state.db.read_conn();
            conn.query_row(
                "SELECT job_ad_text FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| r.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
        }
        ChunkMatchTarget::Criterion { id } => {
            let conn = state.db.read_conn();
            conn.query_row(
                "SELECT text FROM criteria WHERE id = ?1",
                rusqlite::params![id],
                |r| r.get::<_, String>(0),
            )
            .ok()
        }
    };

    // For JobAd, score each CV chunk against the persisted ad rubric chunks
    // (best-over-ad-chunks), mirroring coverage scoring. Lazily persist them
    // if missing, mirroring candidate chunks.
    let ad_rubric_chunks: Vec<chunks::ChunkRow> = if is_job_ad {
        if let Some(text) = &target_text {
            if !text.trim().is_empty() {
                let has = {
                    let conn = state.db.read_conn();
                    chunks::has_rubric_chunks(&conn, &job_id, "ad", &model).unwrap_or(false)
                };
                if !has {
                    if let Ok(embedder) = state.runtime.embedder() {
                        let tokenizer = {
                            let model_dir = app_config::model_dir();
                            app_config::tokenizer_path(&model_dir, &model)
                                .and_then(|path| TokenizerHandle::from_file(&path).ok())
                        };
                        let writer = state.db.writer();
                        if let Err(e) = chunks::ensure_rubric_chunks(
                            &writer,
                            &job_id,
                            "ad",
                            text,
                            embedder.as_ref(),
                            &model,
                            tokenizer.as_ref(),
                        ) {
                            log::warn!("chunk_match: failed to persist ad rubric chunks: {e}");
                        }
                    }
                }
            }
        }
        let conn = state.db.read_conn();
        chunks::rubric_chunks_for(&conn, &job_id, "ad", &model).unwrap_or_default()
    } else {
        Vec::new()
    };

    let target_vector = if is_job_ad {
        None
    } else {
        match (state.runtime.embedder(), target_text) {
            (Ok(embedder), Some(text)) if !text.trim().is_empty() => {
                embedder.embed(&[text.as_str()]).ok()
            }
            _ => None,
        }
    };

    let chunks_out: Vec<ChunkSim> = rows
        .iter()
        .map(|c| {
            let mut matched: Option<&chunks::ChunkRow> = None;
            let similarity = if is_job_ad {
                let mut best = 0.0f64;
                for rc in &ad_rubric_chunks {
                    let sim = libtriqui_core::sort::cosine(&c.vector, &rc.vector);
                    if sim >= best {
                        best = sim;
                        matched = Some(rc);
                    }
                }
                best
            } else {
                match &target_vector {
                    Some(t) => libtriqui_core::sort::cosine(&c.vector, t),
                    None => 0.0,
                }
            };
            ChunkSim {
                char_start: c.char_start,
                char_end: c.char_end,
                similarity,
                matched_target_text: matched.map(|m| m.chunk_text.clone()),
                matched_target_char_start: matched.map(|m| m.char_start),
                matched_target_char_end: matched.map(|m| m.char_end),
            }
        })
        .collect();

    ok_json(ChunkMatchResult {
        model,
        chunks: chunks_out,
    })
}

/// Resolve the embedding space for a candidate: the live embedder's model when
/// available, else whichever model the stored rubric vectors / candidate
/// centroid / persisted chunks use.
fn resolve_model(
    state: &Arc<ServeState>,
    conn: &rusqlite::Connection,
    job_id: &str,
    candidate_id: &str,
) -> Option<String> {
    if let Ok(e) = state.runtime.embedder() {
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
    .or_else(|| {
        conn.query_row(
            "SELECT vector_model FROM candidate_chunks WHERE candidate_id = ?1 LIMIT 1",
            rusqlite::params![candidate_id],
            |r| r.get::<_, String>(0),
        )
        .ok()
    })
}