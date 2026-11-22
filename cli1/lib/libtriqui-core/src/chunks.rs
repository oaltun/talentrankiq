//! Per-chunk embedding persistence for maxsim scoring.
//!
//! The `candidate_chunks` table stores one embedding per chunk of a candidate's
//! extracted text, keyed by `(candidate, model, chunk_index)`. Scoring reads
//! these rows and takes the maximum cosine per rubric question (maxsim), while
//! the inspector reads the same rows to show which passage matched.
//!
//! Chunking uses the active model's tokenizer when one is available and falls
//! back to a single whole-text chunk otherwise, so persistence degrades
//! gracefully for models without a tokenizer.

use rusqlite::{params, Connection};

use libtokenizer::TokenizerHandle;

use crate::embed::{vector_to_bytes, Embedder};

/// Canonical chunk size in tokens, matching the GUI default.
pub const CANONICAL_CHUNK_TOKENS: usize = 256;

/// One persisted chunk, with its vector and char range in the extracted text.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub chunk_index: i64,
    pub chunk_text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub vector: Vec<f32>,
}

/// Delete every chunk row for `(candidate_id, model)`.
pub fn delete_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM candidate_chunks WHERE candidate_id = ?1 AND vector_model = ?2",
        params![candidate_id, model],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// True when at least one chunk is persisted for `(candidate_id, model)`.
pub fn has_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
) -> Result<bool, String> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidate_chunks WHERE candidate_id = ?1 AND vector_model = ?2",
            params![candidate_id, model],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    Ok(count > 0)
}

/// Replace the persisted chunks for `(candidate_id, model)` with `chunks`.
///
/// Each tuple is `(chunk_index, chunk_text, char_start, char_end, vector)`.
/// Existing rows for the same `(candidate, model)` are deleted first, so this is
/// idempotent and a re-embedding never leaves stale rows behind.
pub fn write_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
    chunks: &[(usize, String, usize, usize, Vec<f32>)],
) -> Result<(), String> {
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM candidate_chunks WHERE candidate_id = ?1 AND vector_model = ?2",
        params![candidate_id, model],
    )
    .map_err(|e| e.to_string())?;
    {
        let mut insert = tx
            .prepare(
                "INSERT INTO candidate_chunks
                   (id, candidate_id, chunk_index, chunk_text, char_start, char_end,
                    content_hash, vector, vector_model)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(|e| e.to_string())?;
        for (index, text, char_start, char_end, vector) in chunks {
            insert
                .execute(params![
                    format!("cc_{}_{}_{}", candidate_id, model, index),
                    candidate_id,
                    *index as i64,
                    text,
                    *char_start as i64,
                    *char_end as i64,
                    content_hash(text),
                    vector_to_bytes(vector),
                    model,
                ])
                .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())
}

/// Persist chunks for a candidate if none exist yet for `model`.
///
/// The text is split with `tokenizer` at [`CANONICAL_CHUNK_TOKENS`] when one is
/// available, otherwise stored as a single whole-text chunk. Every chunk is
/// embedded through `embedder` and written with its char range. Idempotent: a
/// candidate that already has chunks for this model is left untouched.
pub fn ensure_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    text: &str,
    embedder: &dyn Embedder,
    model: &str,
    tokenizer: Option<&TokenizerHandle>,
) -> Result<(), String> {
    if has_candidate_chunks(conn, candidate_id, model)? {
        return Ok(());
    }

    let chunked: Vec<(String, usize, usize)> = match tokenizer {
        Some(tok) => tok
            .chunk_text(text, CANONICAL_CHUNK_TOKENS, 0)
            .map_err(|e| format!("chunk candidate {candidate_id}: {e}"))?
            .into_iter()
            .map(|c| (c.text, c.char_start, c.char_end))
            .collect(),
        None => vec![(text.to_string(), 0, text.len())],
    };

    let mut rows: Vec<(usize, String, usize, usize, Vec<f32>)> =
        Vec::with_capacity(chunked.len());
    for (index, (chunk_text, char_start, char_end)) in chunked.into_iter().enumerate() {
        let vector = embedder
            .embed(&[chunk_text.as_str()])
            .map_err(|e| format!("embed chunk {index} of candidate {candidate_id}: {e}"))?;
        rows.push((index, chunk_text, char_start, char_end, vector));
    }

    write_candidate_chunks(conn, candidate_id, model, &rows)
}

/// Read a candidate's persisted chunks for `model`, ordered by chunk index.
pub fn candidate_chunks_for(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
) -> Result<Vec<ChunkRow>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT chunk_index, chunk_text, char_start, char_end, vector
             FROM candidate_chunks
             WHERE candidate_id = ?1 AND vector_model = ?2
             ORDER BY chunk_index",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![candidate_id, model], |r| {
            Ok(ChunkRow {
                chunk_index: r.get(0)?,
                chunk_text: r.get(1)?,
                char_start: r.get::<_, i64>(2)? as usize,
                char_end: r.get::<_, i64>(3)? as usize,
                vector: crate::embed::bytes_to_vector(&r.get::<_, Vec<u8>>(4)?),
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

/// FNV-1a 64-bit hash of arbitrary bytes, formatted as 16 lowercase hex chars.
///
/// This is the durable content fingerprint shared by chunk rows and candidate
/// rows (`candidates.content_hash`): stable across processes, models, and
/// platforms.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Stable content hash for a chunk's text (FNV-1a, 64-bit).
fn content_hash(text: &str) -> String {
    hash_bytes(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::open_single_connection_at;
    use rusqlite::Connection;
    use std::path::PathBuf;

    struct MockEmbedder;

    impl Embedder for MockEmbedder {
        fn model_id(&self) -> &str {
            "mock"
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<f32>, String> {
            let first = texts.first().copied().unwrap_or("");
            Ok(vec![first.len() as f32, 1.0])
        }
    }

    fn temp_db() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("triqui-chunks-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn insert_candidate(conn: &Connection, cid: &str) {
        conn.execute(
            "INSERT INTO jobs (id, title) VALUES ('job1', 'Job')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates
               (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES (?1, 'job1', '/tmp/x.txt', 'x.txt', 'txt', 'Extracted')",
            params![cid],
        )
        .unwrap();
    }

    fn make_tokenizer() -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let tid = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("triqui-chunks-tok-{tid}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 200,
                "vocab": {
                    "[UNK]": 0, "the": 1, "quick": 2, "brown": 3,
                    "fox": 4, "jumps": 5, "over": 6, "lazy": 7,
                    "dog": 8, "cat": 9, "sits": 10, "on": 11, "mat": 12
                }
            }
        });
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_string(&vocab).unwrap()).unwrap();
        path
    }

    #[test]
    fn hash_bytes_is_stable() {
        // The fingerprint must be a pure function of the bytes: same input,
        // same output, across calls (and therefore across processes/scans).
        assert_eq!(hash_bytes(b"hello cv"), hash_bytes(b"hello cv"));
        assert_eq!(hash_bytes(b"hello cv"), "a86788c6b8a22940");
        assert_ne!(hash_bytes(b"hello cv"), hash_bytes(b"hello cv!"));
        assert_eq!(hash_bytes(b""), "cbf29ce484222325");
        // Hex, 16 chars, lowercase.
        let h = hash_bytes(b"resume");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn write_and_read_round_trip() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_candidate(&conn, "c1");

        let chunks = vec![
            (
                0usize,
                "first chunk".to_string(),
                0usize,
                11usize,
                vec![1.0f32, 0.0],
            ),
            (
                1usize,
                "second chunk".to_string(),
                12usize,
                24usize,
                vec![0.0f32, 1.0],
            ),
        ];
        write_candidate_chunks(&conn, "c1", "model-a", &chunks).unwrap();

        assert!(has_candidate_chunks(&conn, "c1", "model-a").unwrap());
        assert!(!has_candidate_chunks(&conn, "c1", "model-b").unwrap());

        let read = candidate_chunks_for(&conn, "c1", "model-a").unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].chunk_text, "first chunk");
        assert_eq!(read[0].char_start, 0);
        assert_eq!(read[0].char_end, 11);
        assert_eq!(read[1].chunk_text, "second chunk");
        assert_eq!(read[1].vector, vec![0.0f32, 1.0]);

        // Idempotent rewrite replaces, never duplicates.
        write_candidate_chunks(&conn, "c1", "model-a", &chunks[..1]).unwrap();
        assert_eq!(candidate_chunks_for(&conn, "c1", "model-a").unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ensure_with_tokenizer_splits_long_text() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_candidate(&conn, "c1");
        let tok = TokenizerHandle::from_file(&make_tokenizer()).unwrap();

        let text = "the quick brown fox jumps over the lazy dog cat sits on the mat "
            .repeat(30);
        ensure_candidate_chunks(
            &conn,
            "c1",
            &text,
            &MockEmbedder,
            "model-a",
            Some(&tok),
        )
        .unwrap();

        let read = candidate_chunks_for(&conn, "c1", "model-a").unwrap();
        assert!(
            read.len() > 1,
            "a long text with a tokenizer must split into multiple chunks, got {}",
            read.len()
        );
        // Offsets are monotonic and valid for the source text.
        for w in read.windows(2) {
            assert!(w[0].char_end <= w[1].char_end);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ensure_without_tokenizer_stores_one_chunk() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_candidate(&conn, "c1");

        let text = "a single whole-document chunk";
        ensure_candidate_chunks(&conn, "c1", text, &MockEmbedder, "model-a", None).unwrap();

        let read = candidate_chunks_for(&conn, "c1", "model-a").unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].chunk_text, text);
        assert_eq!(read[0].char_start, 0);
        assert_eq!(read[0].char_end, text.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ensure_is_idempotent() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_candidate(&conn, "c1");

        let text = "idempotent candidate text";
        ensure_candidate_chunks(&conn, "c1", text, &MockEmbedder, "model-a", None).unwrap();
        let first = candidate_chunks_for(&conn, "c1", "model-a").unwrap();
        ensure_candidate_chunks(&conn, "c1", text, &MockEmbedder, "model-a", None).unwrap();
        let second = candidate_chunks_for(&conn, "c1", "model-a").unwrap();

        assert_eq!(first.len(), second.len());
        assert_eq!(first[0].vector, second[0].vector);

        let _ = std::fs::remove_file(&path);
    }

    fn insert_job(conn: &Connection, job_id: &str) {
        conn.execute("INSERT INTO jobs (id, title) VALUES (?1, 'Job')", [job_id])
            .unwrap();
    }

    #[test]
    fn rubric_chunk_write_and_read_round_trip() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_job(&conn, "job1");

        let chunks = vec![
            (
                0usize,
                "requirement one".to_string(),
                0usize,
                15usize,
                vec![1.0f32, 0.0],
            ),
            (
                1usize,
                "requirement two".to_string(),
                16usize,
                31usize,
                vec![0.0f32, 1.0],
            ),
        ];
        write_rubric_chunks(&conn, "job1", "ad", "model-a", "srchash", &chunks).unwrap();

        assert!(has_rubric_chunks(&conn, "job1", "ad", "model-a").unwrap());
        assert!(!has_rubric_chunks(&conn, "job1", "ad", "model-b").unwrap());
        assert!(!has_rubric_chunks(&conn, "job1", "crit1", "model-a").unwrap());

        let read = rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].chunk_text, "requirement one");
        assert_eq!(read[0].char_start, 0);
        assert_eq!(read[1].char_end, 31);
        assert_eq!(read[1].vector, vec![0.0f32, 1.0]);

        // Idempotent rewrite replaces, never duplicates.
        write_rubric_chunks(&conn, "job1", "ad", "model-a", "srchash", &chunks[..1]).unwrap();
        assert_eq!(rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rubric_chunks_current_tracks_source_text() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_job(&conn, "job1");

        let text = "Senior engineer. Required: Rust, Tokio.";
        assert!(!rubric_chunks_current(&conn, "job1", "ad", "model-a", text).unwrap());
        ensure_rubric_chunks(&conn, "job1", "ad", text, &MockEmbedder, "model-a", None).unwrap();
        assert!(rubric_chunks_current(&conn, "job1", "ad", "model-a", text).unwrap());
        // A different text is not current, so the question rebuilds.
        assert!(!rubric_chunks_current(&conn, "job1", "ad", "model-a", "Different").unwrap());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ensure_rubric_chunks_is_idempotent() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_job(&conn, "job1");

        let text = "Senior engineer. Required: Rust, Tokio, async.";
        ensure_rubric_chunks(&conn, "job1", "ad", text, &MockEmbedder, "model-a", None).unwrap();
        let first = rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap();
        ensure_rubric_chunks(&conn, "job1", "ad", text, &MockEmbedder, "model-a", None).unwrap();
        let second = rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap();

        assert_eq!(first.len(), second.len());
        assert_eq!(first[0].chunk_text, second[0].chunk_text);
        assert_eq!(first[0].vector, second[0].vector);

        // New text rebuilds the rows.
        ensure_rubric_chunks(&conn, "job1", "ad", "Different text", &MockEmbedder, "model-a", None)
            .unwrap();
        let third = rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap();
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].chunk_text, "Different text");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ensure_rubric_chunks_splits_long_text() {
        let path = temp_db();
        let conn = open_single_connection_at(&path).unwrap();
        insert_job(&conn, "job1");
        let tok = TokenizerHandle::from_file(&make_tokenizer()).unwrap();

        let text = "the quick brown fox jumps over the lazy dog cat sits on the mat ".repeat(30);
        ensure_rubric_chunks(&conn, "job1", "ad", &text, &MockEmbedder, "model-a", Some(&tok))
            .unwrap();

        let read = rubric_chunks_for(&conn, "job1", "ad", "model-a").unwrap();
        assert!(
            read.len() > 1,
            "long ad text with a tokenizer must split into multiple chunks, got {}",
            read.len()
        );
        for w in read.windows(2) {
            assert!(w[0].char_end <= w[1].char_end);
        }

        let _ = std::fs::remove_file(&path);
    }
}
/// Delete every rubric chunk row for `(job_id, question_id, model)`.
pub fn delete_rubric_chunks(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
) -> Result<(), String> {
    conn.execute(
        "DELETE FROM rubric_chunks WHERE job_id = ?1 AND question_id = ?2 AND vector_model = ?3",
        params![job_id, question_id, model],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// True when at least one rubric chunk is persisted for `(job_id, question_id, model)`.
pub fn has_rubric_chunks(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
) -> Result<bool, String> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM rubric_chunks WHERE job_id = ?1 AND question_id = ?2 AND vector_model = ?3",
            params![job_id, question_id, model],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    Ok(count > 0)
}

/// Replace the persisted rubric chunks for `(job_id, question_id, model)` with `chunks`.
///
/// `source_hash` is the stable hash of the full question text; storing it lets
/// `ensure_rubric_chunks` skip re-tokenizing an unchanged question.
pub fn write_rubric_chunks(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
    source_hash: &str,
    chunks: &[(usize, String, usize, usize, Vec<f32>)],
) -> Result<(), String> {
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM rubric_chunks WHERE job_id = ?1 AND question_id = ?2 AND vector_model = ?3",
        params![job_id, question_id, model],
    )
    .map_err(|e| e.to_string())?;
    {
        let mut insert = tx
            .prepare(
                "INSERT INTO rubric_chunks
                   (id, job_id, question_id, chunk_index, chunk_text, char_start, char_end,
                    content_hash, vector, vector_model, source_hash)
                  VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )
            .map_err(|e| e.to_string())?;
        for (index, text, char_start, char_end, vector) in chunks {
            insert
                .execute(params![
                    format!("rc_{}_{}_{}_{}", job_id, question_id, model, index),
                    job_id,
                    question_id,
                    *index as i64,
                    text,
                    *char_start as i64,
                    *char_end as i64,
                    content_hash(text),
                    vector_to_bytes(vector),
                    model,
                    source_hash,
                ])
                .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())
}

/// True when the persisted rubric chunks for `(job_id, question_id, model)`
/// already reflect `text` (same stable source hash). Lets callers skip
/// re-tokenizing and re-embedding an unchanged question.
pub fn rubric_chunks_current(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
    text: &str,
) -> Result<bool, String> {
    let hash = content_hash(text);
    let existing: Option<String> = conn
        .query_row(
            "SELECT source_hash FROM rubric_chunks
             WHERE job_id = ?1 AND question_id = ?2 AND vector_model = ?3
             LIMIT 1",
            params![job_id, question_id, model],
            |r| r.get(0),
        )
        .ok();
    Ok(existing.as_deref() == Some(hash.as_str()))
}

/// Read a rubric question's persisted chunks for `model`, ordered by chunk index.
pub fn rubric_chunks_for(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
) -> Result<Vec<ChunkRow>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT chunk_index, chunk_text, char_start, char_end, vector
             FROM rubric_chunks
             WHERE job_id = ?1 AND question_id = ?2 AND vector_model = ?3
             ORDER BY chunk_index",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![job_id, question_id, model], |r| {
            Ok(ChunkRow {
                chunk_index: r.get(0)?,
                chunk_text: r.get(1)?,
                char_start: r.get::<_, i64>(2)? as usize,
                char_end: r.get::<_, i64>(3)? as usize,
                vector: crate::embed::bytes_to_vector(&r.get::<_, Vec<u8>>(4)?),
            })
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

/// Persist rubric chunks for a question if missing or stale.
pub fn ensure_rubric_chunks(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    text: &str,
    embedder: &dyn Embedder,
    model: &str,
    tokenizer: Option<&TokenizerHandle>,
) -> Result<(), String> {
    // Idempotent: skip re-tokenizing when the persisted chunks already reflect
    // this exact text.
    if rubric_chunks_current(conn, job_id, question_id, model, text)? {
        return Ok(());
    }
    if has_rubric_chunks(conn, job_id, question_id, model)? {
        delete_rubric_chunks(conn, job_id, question_id, model)?;
    }

    let chunked: Vec<(String, usize, usize)> = match tokenizer {
        Some(tok) => tok
            .chunk_text(text, CANONICAL_CHUNK_TOKENS, 0)
            .map_err(|e| format!("chunk rubric {job_id}/{question_id}: {e}"))?
            .into_iter()
            .map(|c| (c.text, c.char_start, c.char_end))
            .collect(),
        None => vec![(text.to_string(), 0, text.len())],
    };

    let source_hash = content_hash(text);
    let mut rows: Vec<(usize, String, usize, usize, Vec<f32>)> =
        Vec::with_capacity(chunked.len());
    for (index, (chunk_text, char_start, char_end)) in chunked.into_iter().enumerate() {
        let vector = embedder
            .embed(&[chunk_text.as_str()])
            .map_err(|e| format!("embed rubric chunk {index} of {job_id}/{question_id}: {e}"))?;
        rows.push((index, chunk_text, char_start, char_end, vector));
    }

    write_rubric_chunks(conn, job_id, question_id, model, &source_hash, &rows)
}
