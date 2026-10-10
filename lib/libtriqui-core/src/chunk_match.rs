//! On-demand chunk-vs-target matching for the gui's Match Lens (CV Inspector).
//!
//! Given a candidate, a target (the job ad or one criterion), and a
//! user-chosen token size, chunks the candidate's extracted text, embeds every
//! chunk (cache-aware), and cosine-scores each chunk against the target's
//! reference embedding. The response carries EVERY chunk with char offsets +
//! similarity; the frontend picks the top-n and ranks, so changing n never
//! re-runs the model.
//!
//! Reference rule: a target text up to [`REFERENCE_CHUNK_TOKENS`] tokens is
//! embedded whole; a longer text is chunked at that size and the normalized
//! chunk centroid becomes the reference (no silent model-window truncation).

use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension};

use libtriqui_client::types::{ChunkMatchResult, ChunkMatchTarget, ChunkSim};

use crate::db::db_err;
use crate::embed::{content_hash, vector_to_bytes, Embedder};
use crate::embeddings::{
    cache_vector, cached_vector, centroid, tokenizer_handle_for, vector_is_sane, EMBED_BATCH,
};
use crate::runtime::EngineRuntime;
use crate::types::DomainError;

/// Whole-text reference cap. Harrier-oss has a 32k token window; 30k leaves
/// headroom for special tokens. Longer texts are chunked at this size and the
/// chunk centroid is the reference.
pub const REFERENCE_CHUNK_TOKENS: usize = 30_000;

/// Chunk-size bounds enforced on every request (tokens).
pub const MIN_CHUNK_SIZE: usize = 8;
pub const MAX_CHUNK_SIZE: usize = 2048;

/// Fixed chunk overlap ratio, matching `libembedrank::ChunkingConfig::default`.
const OVERLAP_RATIO: f64 = 0.15;

fn overlap_for(size: usize) -> usize {
    ((size as f64 * OVERLAP_RATIO) as usize).max(1)
}

fn validate_chunk_size(size: usize) -> Result<(), DomainError> {
    if size < MIN_CHUNK_SIZE || size > MAX_CHUNK_SIZE {
        return Err(DomainError::ValidationError {
            message: format!(
                "chunk size must satisfy {MIN_CHUNK_SIZE} <= chunk_size <= {MAX_CHUNK_SIZE} (got {size})"
            ),
        });
    }
    Ok(())
}

/// Byte offset -> char index, clamping to the nearest char boundary so a
/// tokenizer offset that splits a multi-byte char cannot panic the slice.
fn char_boundary_floor(text: &str, mut i: usize) -> usize {
    if i >= text.len() {
        return text.len();
    }
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn char_boundary_ceil(text: &str, mut i: usize) -> usize {
    if i >= text.len() {
        return text.len();
    }
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn char_index(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

/// Embed every text cache-first: hits serve from the chunk-level
/// `embedding_cache`, misses embed in batches through `embedder` and are
/// written back. One vector per input, in order.
pub(crate) fn embed_texts_cached(
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, DomainError> {
    let mut vecs: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut misses: Vec<usize> = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        let hash = content_hash(t);
        if let Some(bytes) = cached_vector(conn, model, &hash)? {
            vecs[i] = Some(crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                DomainError::ParseError {
                    message: format!("bad cached vector: {e}"),
                }
            })?);
        } else {
            misses.push(i);
        }
    }
    for batch in misses.chunks(*EMBED_BATCH) {
        let refs: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
        let embedded = embedder.embed(&refs)?;
        if embedded.len() != refs.len() {
            return Err(DomainError::ParseError {
                message: "embedding count mismatch".to_string(),
            });
        }
        for (&i, v) in batch.iter().zip(embedded) {
            if !vector_is_sane(&v) {
                return Err(DomainError::ParseError {
                    message: "embedding failed sanity check (empty or all-zero)".to_string(),
                });
            }
            cache_vector(conn, model, &content_hash(&texts[i]), &vector_to_bytes(&v))?;
            vecs[i] = Some(v);
        }
    }
    Ok(vecs.into_iter().flatten().collect())
}

/// The target's reference embedding: whole-text when it fits the model window,
/// else the normalized centroid of ~30k-token pieces.
fn reference_vector(
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    tok: &libtokenizer::TokenizerHandle,
    text: &str,
) -> Result<Vec<f32>, DomainError> {
    if tok.count_tokens(text) <= REFERENCE_CHUNK_TOKENS {
        return crate::embeddings::embed_text_cached(conn, embedder, model, None, None, text);
    }
    let pieces = tok
        .chunk_text(text, REFERENCE_CHUNK_TOKENS, 1)
        .map_err(|e| DomainError::ParseError {
            message: format!("reference chunk_text: {e}"),
        })?;
    if pieces.is_empty() {
        return Err(DomainError::ParseError {
            message: "reference text produced no chunks".to_string(),
        });
    }
    let texts: Vec<String> = pieces.into_iter().map(|c| c.text).collect();
    let vecs = embed_texts_cached(conn, embedder, model, &texts)?;
    Ok(centroid(&vecs))
}

/// Chunk `text` at `size` tokens, embed every chunk, and cosine-score each
/// against `reference`. Returned in document order with char offsets.
fn size_chunks(
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    tok: &libtokenizer::TokenizerHandle,
    text: &str,
    size: usize,
    reference: &[f32],
) -> Result<Vec<ChunkSim>, DomainError> {
    let raw =
        tok.chunk_text(text, size, overlap_for(size))
            .map_err(|e| DomainError::ParseError {
                message: format!("chunk_text: {e}"),
            })?;
    if raw.is_empty() {
        return Ok(vec![]);
    }
    let texts: Vec<String> = raw.iter().map(|c| c.text.clone()).collect();
    let vecs = embed_texts_cached(conn, embedder, model, &texts)?;
    let out = raw
        .iter()
        .zip(vecs)
        .map(|(c, v)| ChunkSim {
            char_start: char_index(text, char_boundary_floor(text, c.char_start)),
            char_end: char_index(text, char_boundary_ceil(text, c.char_end)),
            similarity: crate::embed::cosine(&v, reference) as f64,
        })
        .collect();
    Ok(out)
}

/// Run one Match Lens request. Blocking (model calls); the serve dispatcher
/// already runs each request on its own thread, and the connection is the
/// caller's own (not the shared write lock).
pub fn chunk_match(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    candidate_id: &str,
    target: &ChunkMatchTarget,
    chunk_size: usize,
) -> Result<ChunkMatchResult, DomainError> {
    validate_chunk_size(chunk_size)?;

    let (job_id, text) = match conn.query_row(
        "SELECT job_id, extracted_text FROM candidates WHERE id = ?1",
        rusqlite::params![candidate_id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
    ) {
        Ok(v) => v,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(DomainError::NotFound {
                entity: "Candidate".into(),
                id: candidate_id.to_string(),
            });
        }
        Err(e) => return Err(db_err(e)),
    };
    let text = text.unwrap_or_default();
    if text.trim().is_empty() {
        return Err(DomainError::ValidationError {
            message: "candidate has no extracted text".to_string(),
        });
    }

    let reference_text: String = match target {
        ChunkMatchTarget::JobAd => {
            let ad: Option<String> = conn
                .query_row(
                    "SELECT job_ad_text FROM jobs WHERE id = ?1",
                    rusqlite::params![job_id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(db_err)?;
            ad.map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| DomainError::ValidationError {
                    message: "job has no job ad text".to_string(),
                })?
        }
        ChunkMatchTarget::Criterion { id } => {
            let t: Option<String> = conn
                .query_row(
                    "SELECT text FROM criteria WHERE id = ?1 AND job_id = ?2",
                    rusqlite::params![id, job_id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(db_err)?;
            t.map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| DomainError::NotFound {
                    entity: "Criterion".into(),
                    id: id.clone(),
                })?
        }
    };

    let model = rt.model();
    let tok = tokenizer_handle_for(&model).ok_or_else(|| DomainError::ValidationError {
        message: format!("tokenizer for model {model} is unavailable; chunk matching cannot run"),
    })?;

    rt.wait_until_ready(std::time::Duration::from_secs(60))?;
    let embedder: Arc<dyn Embedder> = rt.embedder()?;

    // Inspector priority: hold the gate so background embeds (cascade waves,
    // scout, import) pause between work units, then run this request's embeds
    // immediately. Setting changes re-enter here — a new request, a new hold.
    let _gate = rt.gate_hold();
    let file_name: String = conn
        .query_row(
            "SELECT COALESCE(NULLIF(file_name, ''), id) FROM candidates WHERE id = ?1",
            rusqlite::params![candidate_id],
            |r| r.get(0),
        )
        .unwrap_or_else(|_| candidate_id.to_string());
    let target_label = match target {
        ChunkMatchTarget::JobAd => "job ad".to_string(),
        ChunkMatchTarget::Criterion { .. } => "criterion".to_string(),
    };
    let _activity = crate::embeddings::ActivityGuard { rt: rt.as_ref() };
    rt.set_activity("inspect", &file_name, Some(target_label), Some(&model));

    let reference = reference_vector(conn, &embedder, &model, &tok, &reference_text)?;
    let chunks = size_chunks(conn, &embedder, &model, &tok, &text, chunk_size, &reference)?;

    Ok(ChunkMatchResult { model, chunks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::SCHEMA;
    use crate::embed::MockEmbedder;
    use rusqlite::Connection;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(SCHEMA).expect("schema");
        conn
    }

    fn tok() -> libtokenizer::TokenizerHandle {
        // Minimal WordPiece tokenizer (same fixture style as libtokenizer tests).
        static DIR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "chunk_match_test_{}",
            DIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {
                "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 200,
                "vocab": {
                    "[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5,
                    "zeta": 6, "eta": 7, "theta": 8, "iota": 9, "kappa": 10
                }
            }
        });
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&vocab).unwrap()).unwrap();
        libtokenizer::TokenizerHandle::from_file(&path).unwrap()
    }

    #[test]
    fn chunk_size_must_be_in_bounds() {
        assert!(validate_chunk_size(64).is_ok());
        assert!(validate_chunk_size(MIN_CHUNK_SIZE).is_ok());
        assert!(validate_chunk_size(MAX_CHUNK_SIZE).is_ok());
        assert!(validate_chunk_size(4).is_err());
        assert!(validate_chunk_size(MAX_CHUNK_SIZE + 1).is_err());
    }

    #[test]
    fn overlap_is_ratio_at_least_one() {
        assert_eq!(overlap_for(8), 1);
        assert_eq!(overlap_for(64), 9);
        assert_eq!(overlap_for(256), 38);
    }

    #[test]
    fn char_index_handles_multibyte_and_bad_boundaries() {
        let text = "héllo 日本 wörld";
        // Byte offset of 'é' start is 1; char index 1.
        assert_eq!(char_index(text, 1), 1);
        // Mid-char byte offset floors to a boundary.
        assert_eq!(char_index(text, char_boundary_floor(text, 2)), 1);
        // End of text.
        assert_eq!(char_index(text, text.len()), text.chars().count());
        assert_eq!(
            char_index(text, char_boundary_ceil(text, text.len() + 9)),
            text.chars().count()
        );
    }

    #[test]
    fn size_chunks_returns_document_order_char_offsets() {
        let conn = test_conn();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let t = tok();
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let reference = vec![0.5f32; 8];

        let chunks =
            size_chunks(&conn, &embedder, "mock", &t, text, 4, &reference).expect("size_chunks");
        assert!(chunks.len() > 1, "expected multiple chunks, got {chunks:?}");
        // Document order: starts are non-decreasing, ends within bounds.
        for w in chunks.windows(2) {
            assert!(w[0].char_start <= w[1].char_start);
        }
        for c in &chunks {
            assert!(c.char_start < c.char_end);
            assert!(c.char_end <= text.chars().count());
            assert!((-1.0..=1.0).contains(&c.similarity));
        }
        // Round-trip: slicing the text by char offsets yields the chunk text.
        let chars: Vec<char> = text.chars().collect();
        let first = &chunks[0];
        let sliced: String = chars[first.char_start..first.char_end].iter().collect();
        assert_eq!(sliced.trim(), "alpha beta gamma delta");
    }

    #[test]
    fn embed_texts_cached_serves_second_call_from_cache() {
        let conn = test_conn();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let texts = vec!["alpha beta".to_string(), "gamma delta".to_string()];

        let first = embed_texts_cached(&conn, &embedder, "mock", &texts).expect("first embed");
        assert_eq!(first.len(), 2);

        // Zero-embed check: a fresh call must be served fully from the cache.
        let second = embed_texts_cached(&conn, &embedder, "mock", &texts).expect("second embed");
        assert_eq!(first, second);
    }

    #[test]
    fn reference_vector_long_text_chunks_instead_of_whole_text() {
        let conn = test_conn();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let t = tok();
        // ~35k one-word tokens > the 30k cap: forces the chunked-centroid path.
        let text = "alpha ".repeat(35_000);

        let v = reference_vector(&conn, &embedder, "mock", &t, &text).expect("reference");
        assert_eq!(v.len(), 8);
        assert!(vector_is_sane(&v));

        // The chunked path caches per-piece vectors and never a whole-text one.
        let rows: i64 = conn
            .query_row(
                "SELECT count(*) FROM embedding_cache WHERE model = 'mock'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(rows >= 2, "expected per-piece cache rows, got {rows}");
        let whole_rows: i64 = conn
            .query_row(
                "SELECT count(*) FROM embedding_cache WHERE model = 'mock' AND content_hash = ?1",
                rusqlite::params![content_hash(&text)],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(whole_rows, 0, "whole text must not be embedded directly");
    }
}
