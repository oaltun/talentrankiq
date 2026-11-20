use crate::context::ProcessContext;
use crate::error::ProcessError;

/// Candidates embedded per `embed()` call. Batching keeps memory bounded and,
/// because each sub-batch is upserted before the next one starts, lets readers
/// (the gui's `get_rank_state` poll) observe embedding progress incrementally
/// instead of only after every candidate has been embedded.
const EMBED_CHUNK: usize = 64;

#[tracing::instrument(name = "ensure_quick_embeddings", skip(ctx, cv_hashes))]
pub fn ensure_quick_embeddings(
    ctx: &mut ProcessContext,
    cv_hashes: &[String],
) -> Result<usize, ProcessError> {
    ensure_embeddings(ctx, cv_hashes, &ctx.job.quick_model.clone())
}

#[tracing::instrument(name = "ensure_deep_embeddings", skip(ctx, cv_hashes))]
pub fn ensure_deep_embeddings(
    ctx: &mut ProcessContext,
    cv_hashes: &[String],
    _limit: usize,
) -> Result<usize, ProcessError> {
    ensure_embeddings(ctx, cv_hashes, &ctx.job.deep_model.clone())
}

#[tracing::instrument(name = "ensure_embeddings", skip(ctx, cv_hashes))]
fn ensure_embeddings(
    ctx: &mut ProcessContext,
    cv_hashes: &[String],
    model_id: &str,
) -> Result<usize, ProcessError> {
    // filter missing
    let mut missing: Vec<String> = Vec::new();
    for h in cv_hashes {
        if !ctx.db.has_embedding(h, model_id)? {
            missing.push(h.clone());
        }
    }
    if missing.is_empty() {
        return Ok(0);
    }
    ctx.runtime.ensure_started();
    let _ = ctx
        .runtime
        .wait_until_ready(std::time::Duration::from_secs(120));
    let embedder = match ctx.runtime.embedder_for(model_id) {
        Ok(e) => e,
        Err(_) => return Ok(0),
    };
    let mut done = 0usize;
    for chunk in missing.chunks(EMBED_CHUNK) {
        // load texts for this chunk
        let mut texts: Vec<&str> = Vec::with_capacity(chunk.len());
        let mut owned: Vec<String> = Vec::with_capacity(chunk.len());
        for h in chunk {
            let t = ctx
                .db
                .with_reader(|c| {
                    c.query_row("SELECT text FROM cv WHERE hash = ?1", [h.as_str()], |r| {
                        r.get::<_, String>(0)
                    })
                })
                .unwrap_or_default();
            owned.push(t);
        }
        for t in &owned {
            texts.push(t.as_str());
        }
        let vecs = match embedder.embed(&texts) {
            Ok(v) => v,
            Err(_) => return Ok(done),
        };
        for (i, h) in chunk.iter().enumerate() {
            let v = &vecs[i];
            let bytes = libtriqui_core::embed::vector_to_bytes(v);
            ctx.db.upsert_embedding(h, model_id, &bytes, None)?;
        }
        done += chunk.len();
    }
    Ok(done)
}
