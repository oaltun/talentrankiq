//! Candidate/rubric embedding operations (no Tauri).
//!
//! [`embed_missing_candidates`] (re)embeds every Extracted candidate of a job whose
//! stored vector is absent or was produced by a different model, reporting progress
//! through a callback as it goes. It runs on the caller's `Connection`.
//!
//! With chunking active, each candidate text is split into chunks and every chunk
//! vector is stored in `candidate_chunks` (and the global `embedding_cache`), so
//! Re-rank is pure scoring with no model calls. The normalized chunk centroid is
//! stored in `candidates.vector` as a fallback. Every vector passes a sanity check
//! (non-empty, not all zeros) before it is stored.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension};

use crate::app_config;
use crate::db::db_err;
use crate::embed::{content_hash, vector_to_bytes, Embedder};
use crate::runtime::EngineRuntime;
use crate::types::DomainError;

/// A stored chunk vector attached to a candidate: chunk text plus f32 embedding.
pub struct CandidateChunk {
    pub chunk_index: usize,
    pub chunk_text: String,
    pub content_hash: String,
    pub vector: Vec<f32>,
}

/// Sanity check for any embedding before it is persisted or scored: it must be
/// non-empty and not identically zero (a real model never produces an all-zero
/// vector; zeros mean a broken model / truncated download / decode bug).
pub fn vector_is_sane(v: &[f32]) -> bool {
    !v.is_empty() && v.iter().any(|&x| x != 0.0)
}

/// Replace a candidate's stored chunk vectors **for `model` only** with `chunks`
/// (one transaction). Scoped by `vector_model` so a second model (the cascade's
/// scout embedder) keeps its own chunks for the same candidate: two embedders
/// are live at once during a cascade run, and they must not evict each other.
pub fn store_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
    chunks: &[CandidateChunk],
) -> Result<(), DomainError> {
    let tx = conn.unchecked_transaction().map_err(db_err)?;
    tx.execute(
        "DELETE FROM candidate_chunks WHERE candidate_id = ?1 AND vector_model = ?2",
        rusqlite::params![candidate_id, model],
    )
    .map_err(db_err)?;
    {
        let mut ins = tx
            .prepare(
                "INSERT INTO candidate_chunks
                 (id, candidate_id, chunk_index, chunk_text, content_hash, vector, vector_model)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )
            .map_err(db_err)?;
        for c in chunks {
            let id = format!("cc_{}", uuid::Uuid::new_v4());
            ins.execute(rusqlite::params![
                id,
                candidate_id,
                c.chunk_index as i64,
                c.chunk_text,
                c.content_hash,
                vector_to_bytes(&c.vector),
                model,
            ])
            .map_err(db_err)?;
        }
    }
    tx.commit().map_err(db_err)?;
    Ok(())
}

/// Load a candidate's chunk vectors for `model`, ordered by chunk index.
/// Returns `(ChunkedEmbedding, chunk_texts)` for maxsim scoring, or `None` when
/// the candidate has no current-model chunks.
pub fn load_candidate_chunks(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
) -> Result<Option<(libembedrank::ChunkedEmbedding, Vec<String>)>, DomainError> {
    let mut stmt = conn
        .prepare(
            "SELECT chunk_text, vector FROM candidate_chunks
             WHERE candidate_id = ?1 AND vector_model = ?2 ORDER BY chunk_index",
        )
        .map_err(db_err)?;
    let rows: Vec<(String, Vec<u8>)> = stmt
        .query_map(rusqlite::params![candidate_id, model], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;
    drop(stmt);
    if rows.is_empty() {
        return Ok(None);
    }
    let mut texts = Vec::with_capacity(rows.len());
    let mut chunks = Vec::with_capacity(rows.len());
    for (text, bytes) in rows {
        let v =
            crate::embed::bytes_to_vector(&bytes, None).map_err(|e| DomainError::ParseError {
                message: format!("bad stored chunk vector: {e}"),
            })?;
        texts.push(text);
        chunks.push(v.into_iter().map(|x| x as f64).collect::<Vec<f64>>());
    }
    Ok(Some((libembedrank::ChunkedEmbedding::new(chunks), texts)))
}

/// How many texts to send per embeddings request. `engine.embed_batch` in
/// `ikapp.yaml` is an optional ceiling; unset, the batch derives from the
/// libsyscap budget (thread cap — capacity when idle): each batch is one
/// ONNX forward pass whose activation memory scales with
/// `batch x chunk_tokens`, and smaller batches keep peak RAM flat and give
/// finer `x of y` progress.
pub(crate) static EMBED_BATCH: LazyLock<usize> = LazyLock::new(app_config::engine_embed_batch);

/// Look up a cached vector for `(model, content_hash)`, if present.
///
/// When `engine.no_cache` is set (benchmarking) this never hits: the cache
/// stays warm for everyone else, but every caller recomputes.
pub(crate) fn cached_vector(
    conn: &Connection,
    model: &str,
    hash: &str,
) -> Result<Option<Vec<u8>>, DomainError> {
    if app_config::engine_no_cache() {
        return Ok(None);
    }
    conn.query_row(
        "SELECT vector FROM embedding_cache WHERE model = ?1 AND content_hash = ?2",
        rusqlite::params![model, hash],
        |r| r.get::<_, Vec<u8>>(0),
    )
    .optional()
    .map_err(db_err)
}

/// Store a vector in the cache, keyed by `(model, content_hash)`.
pub(crate) fn cache_vector(
    conn: &Connection,
    model: &str,
    hash: &str,
    bytes: &[u8],
) -> Result<(), DomainError> {
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (model, content_hash, vector) VALUES (?1, ?2, ?3)",
        rusqlite::params![model, hash, bytes],
    )
    .map_err(db_err)?;
    Ok(())
}

/// Write a vector + model onto a candidate row.
fn set_candidate_vector(
    conn: &Connection,
    id: &str,
    bytes: &[u8],
    model: &str,
) -> Result<(), DomainError> {
    conn.execute(
        "UPDATE candidates SET vector = ?1, vector_model = ?2 WHERE id = ?3",
        rusqlite::params![bytes, model, id],
    )
    .map_err(db_err)?;
    Ok(())
}

/// Embed a text with the engine started lazily: a fully-cached text (the
/// common case after a prewarm or a switch back to a previously used model)
/// is served from the `embedding_cache` with the engine left untouched;
/// only a genuine miss waits out (and starts) the engine to embed.
pub fn embed_text_lazy(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    model: &str,
    tok: Option<&libtokenizer::TokenizerHandle>,
    config: Option<&libembedrank::ChunkingConfig>,
    text: &str,
) -> Result<Vec<f32>, DomainError> {
    if let Some(v) = text_vector_cached_only(conn, model, tok, config, text)? {
        return Ok(v);
    }
    rt.wait_until_ready(Duration::from_secs(60))?;
    let embedder: Arc<dyn Embedder> = rt.embedder()?;
    embed_text_cached(conn, &embedder, model, tok, config, text)
}

/// Per-process memoized tokenizer handles, keyed by model slug. Tokenizer files
/// are big (the harrier one is ~20 MB of JSON), so parsing one on every sort /
/// prewarm adds noticeable latency; each model's tokenizer is loaded once and
/// shared (`TokenizerHandle` methods take `&self`).
static TOKENIZERS: LazyLock<Mutex<HashMap<String, Arc<libtokenizer::TokenizerHandle>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The memoized tokenizer handle for `model`, loading (and caching) the
/// tokenizer file on first use. `None` when the model's tokenizer file is
/// unavailable — chunk-aware features cannot run without it.
pub fn tokenizer_handle_for(model: &str) -> Option<Arc<libtokenizer::TokenizerHandle>> {
    let mut cache = TOKENIZERS.lock().expect("tokenizer cache lock");
    if let Some(t) = cache.get(model) {
        return Some(t.clone());
    }
    let model_dir = app_config::model_dir();
    let path = app_config::tokenizer_path(&model_dir, model)?;
    match libtokenizer::TokenizerHandle::from_file(&path) {
        Ok(t) => {
            let t = Arc::new(t);
            cache.insert(model.to_string(), t.clone());
            Some(t)
        }
        Err(e) => {
            log::warn!("embed: failed to load tokenizer, falling back to non-chunked: {e}");
            None
        }
    }
}

/// Resolve the tokenizer + chunking config for `model`, or `None` when the model
/// has no configured chunk size or its tokenizer file is unavailable (in which
/// case callers fall back to whole-text embedding). The tokenizer is memoized
/// per model for the lifetime of the process.
pub fn tokenizer_for(
    model: &str,
) -> Option<(
    Arc<libtokenizer::TokenizerHandle>,
    libembedrank::ChunkingConfig,
)> {
    let chunk_size = app_config::chunk_size_for_model(model);
    if chunk_size == 0 {
        return None;
    }
    let t = tokenizer_handle_for(model)?;
    Some((t, libembedrank::ChunkingConfig::new(chunk_size)))
}

/// The normalized centroid of chunk vectors (each chunk vector is already
/// normalized by the model; the centroid is re-normalized after averaging).
pub(crate) fn centroid(vecs: &[Vec<f32>]) -> Vec<f32> {
    let dim = vecs.first().map(|v| v.len()).unwrap_or(0);
    let mut c = vec![0.0f32; dim];
    for v in vecs {
        for (acc, x) in c.iter_mut().zip(v) {
            *acc += x;
        }
    }
    let n = vecs.len().max(1) as f32;
    for x in &mut c {
        *x /= n;
    }
    crate::embed::normalize(&mut c);
    c
}

/// Cache-only vector for a text: chunked models return the normalized chunk
/// centroid when EVERY chunk is cached; non-chunked models the cached
/// whole-text vector. `Ok(None)` means at least one chunk (or the whole text)
/// is uncached — embedding the misses needs a running engine. Never touches
/// the engine, so callers can serve fully-cached work with the model
/// unloaded (e.g. right after a model switch).
pub fn text_vector_cached_only(
    conn: &Connection,
    model: &str,
    tok: Option<&libtokenizer::TokenizerHandle>,
    config: Option<&libembedrank::ChunkingConfig>,
    text: &str,
) -> Result<Option<Vec<f32>>, DomainError> {
    match (tok, config) {
        (Some(tok), Some(config)) => {
            let raw = tok
                .chunk_text(text, config.chunk_size, config.overlap())
                .map_err(|e| DomainError::ParseError {
                    message: format!("chunk_text: {e}"),
                })?;
            if raw.is_empty() {
                return Err(DomainError::ParseError {
                    message: "text produced no chunks".to_string(),
                });
            }
            let mut vecs = Vec::with_capacity(raw.len());
            for c in &raw {
                let hash = content_hash(&c.text);
                match cached_vector(conn, model, &hash)? {
                    Some(bytes) => {
                        vecs.push(crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                            DomainError::ParseError {
                                message: format!("bad cached vector: {e}"),
                            }
                        })?)
                    }
                    None => return Ok(None),
                }
            }
            Ok(Some(centroid(&vecs)))
        }
        _ => {
            let hash = content_hash(text);
            match cached_vector(conn, model, &hash)? {
                Some(bytes) => {
                    let v = crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                        DomainError::ParseError {
                            message: format!("bad cached vector: {e}"),
                        }
                    })?;
                    Ok(Some(v))
                }
                None => Ok(None),
            }
        }
    }
}

/// Embed one rubric text (job ad / criterion) with the chunk-level
/// `embedding_cache` served first, so a previously prewarmed text costs zero model
/// calls. Chunked models return the normalized chunk centroid (per-chunk misses
/// are embedded in batches and sanity-checked); non-chunked models embed the whole
/// text through the cache. New vectors are written back to the cache.
pub fn embed_text_cached(
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    tok: Option<&libtokenizer::TokenizerHandle>,
    config: Option<&libembedrank::ChunkingConfig>,
    text: &str,
) -> Result<Vec<f32>, DomainError> {
    match (tok, config) {
        (Some(tok), Some(config)) => {
            let raw = tok
                .chunk_text(text, config.chunk_size, config.overlap())
                .map_err(|e| DomainError::ParseError {
                    message: format!("chunk_text: {e}"),
                })?;
            if raw.is_empty() {
                return Err(DomainError::ParseError {
                    message: "text produced no chunks".to_string(),
                });
            }

            // 1. Cache hits per chunk; misses collected.
            let mut vecs: Vec<Option<Vec<f32>>> = vec![None; raw.len()];
            let mut misses: Vec<usize> = Vec::new();
            for (i, c) in raw.iter().enumerate() {
                let hash = content_hash(&c.text);
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

            // 2. Embed misses in batches, sanity-checking each vector.
            let t_misses = std::time::Instant::now();
            for batch in misses.chunks(*EMBED_BATCH) {
                let texts: Vec<&str> = batch.iter().map(|&i| raw[i].text.as_str()).collect();
                let embedded = embedder.embed(&texts)?;
                if embedded.len() != texts.len() {
                    return Err(DomainError::ParseError {
                        message: "embedding count mismatch".to_string(),
                    });
                }
                for (&i, v) in batch.iter().zip(embedded) {
                    if !vector_is_sane(&v) {
                        return Err(DomainError::ParseError {
                            message: "embedding failed sanity check (empty or all-zero)"
                                .to_string(),
                        });
                    }
                    let hash = content_hash(&raw[i].text);
                    cache_vector(conn, model, &hash, &vector_to_bytes(&v))?;
                    vecs[i] = Some(v);
                }
            }
            if !misses.is_empty() {
                log::info!(
                    "embed_text: {} chunk(s) embedded in {} ms ({} from cache)",
                    misses.len(),
                    t_misses.elapsed().as_millis(),
                    raw.len() - misses.len(),
                );
            }

            // 3. Normalized chunk centroid.
            let vecs: Vec<Vec<f32>> = vecs.into_iter().flatten().collect();
            Ok(centroid(&vecs))
        }
        _ => {
            let hash = content_hash(text);
            if let Some(bytes) = cached_vector(conn, model, &hash)? {
                return crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                    DomainError::ParseError {
                        message: format!("bad cached vector: {e}"),
                    }
                });
            }
            let t_whole = std::time::Instant::now();
            let mut embedded = embedder.embed(std::slice::from_ref(&text))?;
            if embedded.len() != 1 {
                return Err(DomainError::ParseError {
                    message: "embedding count mismatch".to_string(),
                });
            }
            let v = embedded.remove(0);
            if !vector_is_sane(&v) {
                return Err(DomainError::ParseError {
                    message: "embedding failed sanity check (empty or all-zero)".to_string(),
                });
            }
            cache_vector(conn, model, &hash, &vector_to_bytes(&v))?;
            log::info!(
                "embed_text: whole-text embed in {} ms (cache miss)",
                t_whole.elapsed().as_millis()
            );
            Ok(v)
        }
    }
}

/// Job ids with a rubric prewarm currently in flight, so rapid successive saves
/// don't pile up threads. The running pass re-reads the latest texts from the DB
/// right before embedding, so a save made mid-pass is still picked up.
static PREWARM_IN_FLIGHT: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Fire-and-forget prewarm of a job's rubric vectors (job ad + all criteria).
///
/// Called after the user saves a job ad or adds/changes a criterion. Runs on a
/// background thread with its own DB connection; the caller returns immediately,
/// so the UI never blocks. Once done, the vectors live in
/// `jobs.ead` / `criteria.vector` and the chunk-level `embedding_cache`, making
/// the next Re-rank free of candidate-question model calls. Failures are only
/// logged — Sort embeds on demand as a fallback.
pub fn prewarm_rubric_inputs(rt: &Arc<EngineRuntime>, job_id: &str) {
    if !PREWARM_IN_FLIGHT
        .lock()
        .expect("prewarm lock")
        .insert(job_id.to_string())
    {
        return;
    }
    let rt = rt.clone();
    let jid = job_id.to_string();
    std::thread::spawn(move || {
        let result = prewarm_job_blocking(&rt, &jid);
        PREWARM_IN_FLIGHT.lock().expect("prewarm lock").remove(&jid);
        match result {
            Ok(()) => log::info!("prewarm: job {jid} rubric vectors cached"),
            Err(e) => log::warn!("prewarm: job {jid}: {e}"),
        }
    });
}

/// Fire-and-forget prewarm of several jobs' rubric vectors (job ad + all
/// criteria), run sequentially on ONE background thread. Used after a model
/// switch, where every job's rubric vectors were invalidated: a thread-per-job
/// fan-out would flood the engine with concurrent requests (each fighting the
/// client's lock-step stdio protocol) ahead of user-triggered embeds.
pub fn prewarm_all_rubric_inputs(rt: &Arc<EngineRuntime>, job_ids: Vec<String>) {
    if job_ids.is_empty() {
        return;
    }
    let rt = rt.clone();
    std::thread::spawn(move || {
        {
            let mut in_flight = PREWARM_IN_FLIGHT.lock().expect("prewarm lock");
            for jid in &job_ids {
                in_flight.insert(jid.clone());
            }
        }
        for jid in &job_ids {
            match prewarm_job_blocking(&rt, jid) {
                Ok(()) => log::info!("prewarm: job {jid} rubric vectors cached"),
                Err(e) => log::warn!("prewarm: job {jid}: {e}"),
            }
        }
        PREWARM_IN_FLIGHT
            .lock()
            .expect("prewarm lock")
            .retain(|j| !job_ids.contains(j));
    });
}

/// Fill a job's rubric vectors in every space it grades in, without waiting for the
/// engine unless it has to.
///
/// Scoring v3 stores a question's vector in `rubric_vectors (job, question, model)`,
/// so this is a thin wrapper over [`crate::rubric::ensure_vectors`]:
///
/// - the **heavy** space gets a real fill — cache-only first, and only a genuine
///   cache miss waits out (and starts) the engine;
/// - the **scout's** space is filled from the embedding cache only, because loading
///   the scout's session here would evict the heavy embedder whatever is running,
///   and the scout pass picks up anything left at its next boundary.
///
/// Called after an embedder switch (every job's vectors were just wiped) and
/// wherever a warm cache is worth having before a wave.
pub fn prewarm_job_blocking(rt: &Arc<EngineRuntime>, job_id: &str) -> Result<(), DomainError> {
    let conn = rt
        .open_conn()
        .map_err(|e| DomainError::DbError { message: e })?;
    let rubric = crate::rubric::Rubric::load(&conn, job_id)?;
    let heavy = rt.model();
    crate::rubric::ensure_vectors(rt, &conn, &rubric, &heavy, None)?;
    if let Some(scout) = crate::scout::scout_model_for(&heavy) {
        crate::rubric::ensure_vectors_cache_only(&conn, &rubric, &scout)?;
    }
    Ok(())
}

/// Progress callback for [`embed_missing_candidates`]:
/// `(done, total, failed, active)`.
pub type EmbedProgress<'a> = &'a dyn Fn(i64, i64, i64, bool);

/// (Re)embed a job's candidates that have text but no current-model vector.
///
/// Runs in two passes so the engine is only loaded when it is genuinely needed:
/// pass 1 rebuilds every fully-cached candidate straight from the
/// `embedding_cache` with no engine at all (the common case right after
/// switching back to a previously used model); pass 2 starts/waits for the
/// engine and embeds only the candidates that still have uncached chunks.
///
/// With a tokenizer available (chunked models), each candidate is split into
/// chunks; chunk vectors are served from `embedding_cache` when the chunk text is
/// unchanged, embedded otherwise, sanity-checked, stored in `candidate_chunks`,
/// and summarized as a normalized centroid in `candidates.vector`. Without a
/// tokenizer the whole text is embedded as before. Progress is reported through
/// `progress(done, total, failed, active)` throughout. Returns `(embedded, failed)`.
pub fn embed_missing_candidates(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    job_id: &str,
    progress: EmbedProgress<'_>,
) -> Result<(i64, i64), DomainError> {
    embed_missing_candidates_filtered(
        rt,
        conn,
        job_id,
        &rt.model(),
        None,
        true,
        None,
        "embed",
        progress,
    )
}

/// Like [`embed_missing_candidates`] but restricts the candidate set to the
/// given ids (best-first order is preserved when the caller sorts the rows;
/// the IN list is only a set filter).
pub fn embed_selected_candidates(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    job_id: &str,
    ids: &[String],
    progress: EmbedProgress<'_>,
) -> Result<(i64, i64), DomainError> {
    let model = rt.model();
    embed_missing_candidates_filtered(
        rt,
        conn,
        job_id,
        &model,
        None,
        true,
        Some(ids),
        "embed",
        progress,
    )
}

/// Embed candidates with a model **other than** the runtime's current one, using
/// an explicit embedder — the cascade's scout stage (`cascade.scout_model`).
///
/// The scout scores every candidate in one cheap multilingual space before the
/// heavy embedder touches anything, so it must not wait on the runtime's model
/// (which is the heavy one, and may not even be loaded). `embedder` is the
/// scout's own `Embedder` over the shared engine client; `write_centroid` is
/// false so the scout stores only its own model-scoped chunks and leaves
/// `candidates.vector` (the full embedder's centroid) untouched.
///
/// `ids` restricts the set (a wave embeds a slice); `None` means all candidates.
pub fn embed_candidates_with(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    job_id: &str,
    model: &str,
    embedder: Arc<dyn Embedder>,
    ids: Option<&[String]>,
    progress: EmbedProgress<'_>,
) -> Result<(i64, i64), DomainError> {
    embed_missing_candidates_filtered(
        rt,
        conn,
        job_id,
        model,
        Some(embedder),
        false,
        ids,
        "scout",
        progress,
    )
}

/// Clears the engine's activity line when the embed scope ends (success,
/// failure, early return — all exits).
pub(crate) struct ActivityGuard<'a> {
    pub(crate) rt: &'a EngineRuntime,
}

impl Drop for ActivityGuard<'_> {
    fn drop(&mut self) {
        self.rt.clear_activity();
    }
}

#[allow(clippy::too_many_arguments)]
fn embed_missing_candidates_filtered(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    job_id: &str,
    model: &str,
    embedder: Option<Arc<dyn Embedder>>,
    write_centroid: bool,
    ids: Option<&[String]>,
    stage: &str,
    progress: EmbedProgress<'_>,
) -> Result<(i64, i64), DomainError> {
    // Arm freshness BEFORE anything reads vector artifacts: a backend flip
    // (ort ↔ mlx) makes the slug's stored vectors the other arm's, and the
    // cache-only pass below would serve them as current.
    rt.ensure_backend_fresh(conn, model);
    let _activity = ActivityGuard { rt };
    let tok_and_cfg = tokenizer_for(model);

    // Candidates with text whose vector is missing or from another model.
    let sql = if ids.map_or(false, |i| !i.is_empty()) {
        let n = ids.as_ref().unwrap().len();
        let placeholders = (0..n).map(|_| "?").collect::<Vec<_>>().join(",");
        format!(
            "SELECT id, content_hash, extracted_text, file_name FROM candidates
             WHERE job_id = ?1
               AND extraction_status = 'Extracted'
               AND extracted_text IS NOT NULL
               AND (vector IS NULL OR vector_model <> ?2)
               AND id IN ({placeholders})",
            placeholders = placeholders
        )
    } else {
        "SELECT id, content_hash, extracted_text, file_name FROM candidates
         WHERE job_id = ?1
           AND extraction_status = 'Extracted'
           AND extracted_text IS NOT NULL
           AND (vector IS NULL OR vector_model <> ?2)"
            .to_string()
    };

    type Row = (String, Option<String>, String, String);
    let mut stmt = conn.prepare(&sql).map_err(db_err)?;
    let rows: Vec<Row> = if ids.map_or(false, |i| !i.is_empty()) {
        let idlist = ids.unwrap();
        let mut params: Vec<&dyn rusqlite::ToSql> = vec![&job_id, &model];
        for id in idlist {
            params.push(id);
        }
        stmt.query_map(rusqlite::params_from_iter(params), |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?
    } else {
        stmt.query_map(rusqlite::params![job_id, &model], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?
    };
    drop(stmt);

    let total = rows.len() as i64;
    let mut done = 0i64;
    let mut failed = 0i64;
    let mut from_cache = 0i64;
    progress(done, total, failed, true);

    // Pass 1 — cache-only: no engine start, no model calls.
    let mut deferred: Vec<Row> = Vec::new();
    for (cid, hash, text, _file_name) in &rows {
        let served = match tok_and_cfg.as_ref() {
            Some((tok, cfg)) => store_candidate_from_cache_only(
                conn,
                model,
                tok,
                cfg,
                cid,
                text,
                &mut from_cache,
                write_centroid,
            ),
            None => store_candidate_plain_from_cache_only(conn, model, cid, hash, &mut from_cache),
        };
        match served? {
            Some(true) => done += 1,
            Some(false) => failed += 1,
            None => deferred.push((cid.clone(), hash.clone(), text.clone(), _file_name.clone())),
        }
        progress(done, total, failed, true);
    }

    // Pass 2 — only the uncached leftovers need the engine (this starts it
    // when it was idle-unloaded or never started). A caller-supplied embedder
    // (the scout) is already live, so only resolve the runtime's own when none
    // was passed.
    if !deferred.is_empty() {
        let embedder = match embedder {
            Some(e) => e,
            None => {
                rt.wait_until_ready(Duration::from_secs(60))?;
                rt.embedder()?
            }
        };
        for (cid, hash, text, file_name) in &deferred {
            rt.gate_wait();
            let name: &str = if file_name.is_empty() {
                cid.as_str()
            } else {
                file_name.as_str()
            };
            rt.set_activity(stage, name, None, Some(model));
            let result: Result<bool, DomainError> = match tok_and_cfg.as_ref() {
                Some((tok, cfg)) => embed_candidate_chunked(
                    rt,
                    conn,
                    &embedder,
                    model,
                    stage,
                    name,
                    tok,
                    cfg,
                    cid,
                    hash,
                    text,
                    &mut from_cache,
                    write_centroid,
                ),
                None => embed_candidate_plain(
                    rt,
                    conn,
                    &embedder,
                    model,
                    stage,
                    name,
                    cid,
                    hash,
                    text,
                    &mut from_cache,
                ),
            };
            match result {
                Ok(true) => done += 1,
                Ok(false) => failed += 1,
                Err(e) => {
                    log::warn!("embed: candidate {cid} failed: {e}");
                    failed += 1;
                }
            }
            progress(done, total, failed, true);
        }
        // The arm that actually served may differ from the pre-embed
        // prediction (a failed MLX load falls down the ladder); reconcile so
        // the recorded arm always names the vectors on disk.
        rt.reconcile_backend(conn, model);
    }

    progress(done, total, failed, false);
    log::info!("embed: job {job_id}: {done} embedded ({from_cache} from cache), {failed} failed");
    Ok((done, failed))
}

/// Persist a candidate's complete chunk set: sanity-checked centroid into
/// `candidates.vector` + chunk rows. Returns `Ok(true)` on success,
/// `Ok(false)` when validation failed.
///
/// `write_centroid` is false for a second embedder sharing the store (the
/// cascade's scout): it stores only its own model-scoped chunks and leaves
/// `candidates.vector` alone, because that column is the *full* embedder's
/// centroid and the scout must never clobber it.
fn persist_chunked_candidate(
    conn: &Connection,
    model: &str,
    cid: &str,
    stored: Vec<CandidateChunk>,
    write_centroid: bool,
) -> Result<bool, DomainError> {
    let c = centroid(&stored.iter().map(|c| c.vector.clone()).collect::<Vec<_>>());
    if !vector_is_sane(&c) {
        log::warn!("embed: candidate {cid} centroid failed sanity check");
        return Ok(false);
    }
    store_candidate_chunks(conn, cid, model, &stored)?;
    if write_centroid {
        set_candidate_vector(conn, cid, &vector_to_bytes(&c), model)?;
    }
    Ok(true)
}

/// Cache-only embed of one chunked candidate: when every chunk vector is
/// cached, the candidate is rebuilt and stored without any engine. Returns
/// `Ok(Some(true))` stored, `Ok(Some(false))` failed validation, `Ok(None)`
/// when at least one chunk is uncached (the engine must embed the misses).
fn store_candidate_from_cache_only(
    conn: &Connection,
    model: &str,
    tok: &libtokenizer::TokenizerHandle,
    config: &libembedrank::ChunkingConfig,
    cid: &str,
    text: &str,
    from_cache: &mut i64,
    write_centroid: bool,
) -> Result<Option<bool>, DomainError> {
    let raw_chunks = tok
        .chunk_text(text, config.chunk_size, config.overlap())
        .map_err(|e| DomainError::ParseError {
            message: format!("chunk_text: {e}"),
        })?;
    if raw_chunks.is_empty() {
        return Ok(Some(false));
    }
    let mut stored = Vec::with_capacity(raw_chunks.len());
    for (i, c) in raw_chunks.iter().enumerate() {
        let hash = content_hash(&c.text);
        match cached_vector(conn, model, &hash)? {
            Some(bytes) => {
                let v = crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                    DomainError::ParseError {
                        message: format!("bad cached vector: {e}"),
                    }
                })?;
                *from_cache += 1;
                stored.push(CandidateChunk {
                    chunk_index: i,
                    chunk_text: c.text.clone(),
                    content_hash: hash,
                    vector: v,
                });
            }
            None => return Ok(None),
        }
    }
    persist_chunked_candidate(conn, model, cid, stored, write_centroid).map(Some)
}

/// Cache-only embed of one non-chunked candidate: a cached whole-text vector
/// is written straight onto the candidate row. Returns `Ok(Some(true))`
/// stored, `Ok(None)` when uncached (the engine must embed it).
fn store_candidate_plain_from_cache_only(
    conn: &Connection,
    model: &str,
    cid: &str,
    hash: &Option<String>,
    from_cache: &mut i64,
) -> Result<Option<bool>, DomainError> {
    let key: Option<&str> = match hash.as_deref() {
        Some(h) if !h.is_empty() => Some(h),
        _ => return Ok(None),
    };
    let bytes = match cached_vector(conn, model, key.unwrap())? {
        Some(b) => b,
        None => return Ok(None),
    };
    let v = crate::embed::bytes_to_vector(&bytes, None).map_err(|e| DomainError::ParseError {
        message: format!("bad cached vector: {e}"),
    })?;
    if !vector_is_sane(&v) {
        log::warn!("embed: candidate {cid} failed sanity check (empty or all-zero)");
        return Ok(Some(false));
    }
    *from_cache += 1;
    set_candidate_vector(conn, cid, &vector_to_bytes(&v), model)?;
    Ok(Some(true))
}

/// Chunked embed of one candidate: cache-aware per chunk, sanity-checked, stored
/// in `candidate_chunks` + centroid in `candidates.vector`. Returns `Ok(true)`
/// when the candidate is embedded, `Ok(false)` when it failed validation.
/// Reports chunk-level activity and pauses at the inspector gate per batch.
#[allow(clippy::too_many_arguments)]
fn embed_candidate_chunked(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    stage: &str,
    name: &str,
    tok: &libtokenizer::TokenizerHandle,
    config: &libembedrank::ChunkingConfig,
    cid: &str,
    _hash: &Option<String>,
    text: &str,
    from_cache: &mut i64,
    write_centroid: bool,
) -> Result<bool, DomainError> {
    let raw_chunks = tok
        .chunk_text(text, config.chunk_size, config.overlap())
        .map_err(|e| DomainError::ParseError {
            message: format!("chunk_text: {e}"),
        })?;
    if raw_chunks.is_empty() {
        return Ok(false);
    }

    // 1. Serve cache hits per chunk; collect misses.
    let mut chunks: Vec<Option<CandidateChunk>> = Vec::with_capacity(raw_chunks.len());
    let mut misses: Vec<usize> = Vec::new();
    for (i, c) in raw_chunks.iter().enumerate() {
        let hash = content_hash(&c.text);
        if let Some(bytes) = cached_vector(conn, model, &hash)? {
            let v = crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                DomainError::ParseError {
                    message: format!("bad cached vector: {e}"),
                }
            })?;
            *from_cache += 1;
            chunks.push(Some(CandidateChunk {
                chunk_index: i,
                chunk_text: c.text.clone(),
                content_hash: hash,
                vector: v,
            }));
        } else {
            chunks.push(None);
            misses.push(i);
        }
    }

    // 2. Embed the misses in batches, sanity-checking each vector. The gate
    // wait sits between engine batches so an inspector hold takes effect
    // within one batch, not one candidate.
    let total_chunks = raw_chunks.len();
    for batch in misses.chunks(*EMBED_BATCH) {
        rt.gate_wait();
        let served = chunks.iter().filter(|c| c.is_some()).count();
        rt.set_activity(
            stage,
            name,
            Some(format!(
                "chunk {}/{}",
                served.min(total_chunks),
                total_chunks
            )),
            Some(model),
        );
        let texts: Vec<&str> = batch.iter().map(|&i| raw_chunks[i].text.as_str()).collect();
        let vecs = embedder.embed(&texts)?;
        if vecs.len() != texts.len() {
            return Ok(false);
        }
        for (&i, v) in batch.iter().zip(vecs) {
            if !vector_is_sane(&v) {
                log::warn!(
                    "embed: candidate {cid} chunk {i} failed sanity check (empty or all-zero)"
                );
                return Ok(false);
            }
            let hash = content_hash(&raw_chunks[i].text);
            cache_vector(conn, model, &hash, &vector_to_bytes(&v))?;
            chunks[i] = Some(CandidateChunk {
                chunk_index: i,
                chunk_text: raw_chunks[i].text.clone(),
                content_hash: hash,
                vector: v,
            });
        }
    }

    let stored: Vec<CandidateChunk> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.ok_or_else(|| DomainError::ParseError {
                message: format!("chunk {i} not embedded"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    persist_chunked_candidate(conn, model, cid, stored, write_centroid)
}

/// Non-chunked embed of one candidate (models without a tokenizer): cache-aware,
/// sanity-checked, stored in `candidates.vector`. Returns `Ok(true)` when the
/// candidate is embedded, `Ok(false)` when it failed validation.
#[allow(clippy::too_many_arguments)]
fn embed_candidate_plain(
    rt: &Arc<EngineRuntime>,
    conn: &Connection,
    embedder: &Arc<dyn Embedder>,
    model: &str,
    stage: &str,
    name: &str,
    cid: &str,
    hash: &Option<String>,
    text: &str,
    from_cache: &mut i64,
) -> Result<bool, DomainError> {
    rt.gate_wait();
    rt.set_activity(stage, name, None, Some(model));
    let key: Option<&str> = match hash.as_deref() {
        Some(h) if !h.is_empty() => Some(h),
        _ => None,
    };
    let v: Option<Vec<f32>> = match key {
        Some(h) => match cached_vector(conn, model, h)? {
            Some(bytes) => {
                let v = crate::embed::bytes_to_vector(&bytes, None).map_err(|e| {
                    DomainError::ParseError {
                        message: format!("bad cached vector: {e}"),
                    }
                })?;
                *from_cache += 1;
                Some(v)
            }
            None => None,
        },
        None => None,
    };
    let v = match v {
        Some(v) => v,
        None => {
            let mut vecs = embedder.embed(std::slice::from_ref(&text))?;
            match vecs.len() {
                1 => vecs.remove(0),
                _ => return Ok(false),
            }
        }
    };
    if !vector_is_sane(&v) {
        log::warn!("embed: candidate {cid} failed sanity check (empty or all-zero)");
        return Ok(false);
    }
    if let Some(h) = key {
        cache_vector(conn, model, h, &vector_to_bytes(&v))?;
    }
    set_candidate_vector(conn, cid, &vector_to_bytes(&v), model)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::SCHEMA;
    use crate::embed::MockEmbedder;

    fn test_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(SCHEMA).expect("schema");
        conn
    }

    fn candidate(conn: &Connection, id: &str) {
        conn.execute(
            "INSERT OR IGNORE INTO jobs (id, title) VALUES ('job_t', 't')",
            [],
        )
        .expect("job");
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES (?1, 'job_t', '/tmp/f.pdf', 'f.pdf', 'Pdf', 'Extracted')",
            rusqlite::params![id],
        )
        .expect("candidate");
    }

    #[test]
    fn sanity_rejects_empty_and_zero_vectors() {
        assert!(!vector_is_sane(&[]));
        assert!(!vector_is_sane(&[0.0, 0.0, 0.0]));
        assert!(vector_is_sane(&[0.0, 1.0]));
        assert!(vector_is_sane(&[-0.5]));
    }

    /// The cascade keeps the scout and the full embedder resident at the same
    /// time, so their chunk rows must coexist instead of evicting each other.
    /// `store_candidate_chunks` used to delete by candidate alone, which made the
    /// table single-model and silently lost one side's vectors.
    #[test]
    fn chunks_are_stored_per_model_and_never_evict_each_other() {
        let conn = test_conn();
        candidate(&conn, "c1");
        let chunk = |idx: usize, _v: f32| CandidateChunk {
            chunk_index: idx,
            chunk_text: "t".into(),
            content_hash: format!("h{idx}"),
            vector: vec![0.5, 1.0, 0.0],
        };

        store_candidate_chunks(&conn, "c1", "scout_m", &[chunk(0, 0.1), chunk(1, 0.2)])
            .expect("store scout chunks");
        store_candidate_chunks(&conn, "c1", "full_m", &[chunk(0, 0.9), chunk(1, 0.8)])
            .expect("store full chunks");

        // Both models' rows survive, at the same chunk indices.
        for model in ["scout_m", "full_m"] {
            let indices: Vec<i64> = {
                let mut stmt = conn
                    .prepare(
                        "SELECT chunk_index FROM candidate_chunks
                         WHERE candidate_id = 'c1' AND vector_model = ?1
                         ORDER BY chunk_index",
                    )
                    .unwrap();
                let v = stmt
                    .query_map(rusqlite::params![model], |r| r.get::<_, i64>(0))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                v
            };
            assert_eq!(indices, vec![0, 1], "{model} kept both chunk indices");
        }

        // Re-storing one model replaces only that model's rows.
        store_candidate_chunks(&conn, "c1", "scout_m", &[chunk(0, 0.5)])
            .expect("restage scout chunks");
        let count = |model: &str| -> i64 {
            conn.query_row(
                "SELECT COUNT(*) FROM candidate_chunks WHERE candidate_id = 'c1' AND vector_model = ?1",
                rusqlite::params![model],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(count("scout_m"), 1, "scout's own rows were replaced");
        assert_eq!(count("full_m"), 2, "the full embedder's rows are untouched");
    }

    /// `tokenizer_for` memoizes per model: a second call returns the same handle
    /// instead of re-parsing the (multi-MB) tokenizer file. Degrades to a no-op
    /// when no chunked model is configured in the environment.
    #[test]
    fn tokenizer_for_memoizes_per_model() {
        let model = "harrier-oss-v1-270m-ONNX_model_q4f16";
        if let (Some((a, _)), Some((b, _))) = (tokenizer_for(model), tokenizer_for(model)) {
            assert!(
                Arc::ptr_eq(&a, &b),
                "second tokenizer_for call should return the memoized handle"
            );
        }
    }

    #[test]
    fn embed_text_cached_serves_second_call_from_cache() {
        let conn = test_conn();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new(8));
        let text = "senior rust engineer";

        let first =
            embed_text_cached(&conn, &embedder, "mock", None, None, text).expect("first embed");
        assert_eq!(first.len(), 8);
        // A cache row was written for the text.
        let hash = content_hash(text);
        let cached: Vec<u8> = conn
            .query_row(
                "SELECT vector FROM embedding_cache WHERE model = 'mock' AND content_hash = ?1",
                rusqlite::params![hash],
                |r| r.get(0),
            )
            .expect("cache row");

        // Tamper with the cached vector and drop the embedder: if the second call
        // returns the tampered values, it was served from the cache, not re-embedded.
        let mut tampered = cached.clone();
        tampered[0] = 0x7f;
        tampered[1] = 0xc0;
        conn.execute(
            "UPDATE embedding_cache SET vector = ?1 WHERE model = 'mock' AND content_hash = ?2",
            rusqlite::params![tampered, hash],
        )
        .expect("tamper");
        let second =
            embed_text_cached(&conn, &embedder, "mock", None, None, text).expect("second embed");
        let expected = crate::embed::bytes_to_vector(&tampered, None).expect("decode");
        assert_eq!(second, expected);
    }

    #[test]
    fn chunks_roundtrip_in_order() {
        let conn = test_conn();
        candidate(&conn, "cand_1");
        let chunks = vec![
            CandidateChunk {
                chunk_index: 0,
                chunk_text: "first".into(),
                content_hash: "h0".into(),
                vector: vec![1.0, 0.0],
            },
            CandidateChunk {
                chunk_index: 1,
                chunk_text: "second".into(),
                content_hash: "h1".into(),
                vector: vec![0.0, 2.0],
            },
        ];
        store_candidate_chunks(&conn, "cand_1", "m1", &chunks).expect("store");

        let (chunked, texts) = load_candidate_chunks(&conn, "cand_1", "m1")
            .expect("load")
            .expect("chunks");
        assert_eq!(texts, vec!["first".to_string(), "second".to_string()]);
        assert_eq!(chunked.chunks.len(), 2);
        assert_eq!(chunked.chunks[0], vec![1.0, 0.0]);
        assert_eq!(chunked.chunks[1], vec![0.0, 2.0]);
    }

    #[test]
    fn chunks_are_model_guarded_and_replaced() {
        let conn = test_conn();
        candidate(&conn, "cand_1");
        let v1 = vec![CandidateChunk {
            chunk_index: 0,
            chunk_text: "old".into(),
            content_hash: "h0".into(),
            vector: vec![1.0],
        }];
        store_candidate_chunks(&conn, "cand_1", "m1", &v1).expect("store m1");
        assert!(load_candidate_chunks(&conn, "cand_1", "m2")
            .expect("load")
            .is_none());

        let v2 = vec![CandidateChunk {
            chunk_index: 0,
            chunk_text: "new".into(),
            content_hash: "h1".into(),
            vector: vec![2.0],
        }];
        store_candidate_chunks(&conn, "cand_1", "m1", &v2).expect("replace");
        let (chunked, texts) = load_candidate_chunks(&conn, "cand_1", "m1")
            .expect("load")
            .expect("chunks");
        assert_eq!(texts, vec!["new".to_string()]);
        assert_eq!(chunked.chunks[0], vec![2.0]);
    }
}
