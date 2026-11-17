use crate::context::ProcessContext;
use crate::error::ProcessError;

pub fn ensure_quick_embeddings(ctx: &mut ProcessContext, cv_hashes: &[String]) -> Result<(), ProcessError> {
    ensure_embeddings(ctx, cv_hashes, &ctx.job.quick_model.clone())
}

pub fn ensure_deep_embeddings(ctx: &mut ProcessContext, cv_hashes: &[String], _limit: usize) -> Result<(), ProcessError> {
    ensure_embeddings(ctx, cv_hashes, &ctx.job.deep_model.clone())
}

fn ensure_embeddings(ctx: &mut ProcessContext, cv_hashes: &[String], model_id: &str) -> Result<(), ProcessError> {
    // filter missing
    let mut missing: Vec<String> = Vec::new();
    for h in cv_hashes {
        if !ctx.db.has_embedding(h, model_id)? {
            missing.push(h.clone());
        }
    }
    if missing.is_empty() { return Ok(()); }
    // load texts
    let mut texts: Vec<(String, String)> = Vec::new();
    for h in &missing {
        // fetch text
        let t = ctx.db.with_reader(|c| {
            c.query_row("SELECT text FROM cv WHERE hash = ?1", [h.as_str()], |r| r.get::<_, String>(0))
        }).unwrap_or_default();
        texts.push((h.clone(), t));
    }
    ctx.runtime.ensure_started();
    let _ = ctx.runtime.wait_until_ready(std::time::Duration::from_secs(120));
    let embedder = ctx.runtime.embedder_for(model_id).map_err(|e| ProcessError::Runtime(e.to_string()))?;
    // batch embed
    let refs: Vec<&str> = texts.iter().map(|(_, t)| t.as_str()).collect();
    let vecs = embedder.embed(&refs).map_err(|e| ProcessError::Runtime(e.to_string()))?;
    for (i, (h, _)) in texts.iter().enumerate() {
        let v = &vecs[i];
        let bytes = libtriqui_core::embed::vector_to_bytes(v);
        ctx.db.upsert_embedding(h, model_id, &bytes, None)?;
    }
    Ok(())
}
