use crate::context::ProcessContext;
use crate::error::ProcessError;
use libembedrank::RetrievalStrategy;

#[derive(Clone, Debug)]
pub struct Target {
    pub hash: String,
    pub text: String,
}

#[tracing::instrument(
    name = "compute_batch_similarity",
    skip(ctx, cv_hashes, targets, model_id, _method)
)]
pub fn compute_batch_similarity(
    ctx: &mut ProcessContext,
    cv_hashes: &[String],
    targets: &[Target],
    model_id: &str,
    _method: RetrievalStrategy,
) -> Result<(), ProcessError> {
    if cv_hashes.is_empty() || targets.is_empty() {
        return Ok(());
    }
    // embed targets
    ctx.runtime.ensure_started();
    let _ = ctx
        .runtime
        .wait_until_ready(std::time::Duration::from_secs(120));
    let embedder = ctx
        .runtime
        .embedder_for(model_id)
        .map_err(|e| ProcessError::Runtime(e.to_string()))?;
    let trefs: Vec<&str> = targets.iter().map(|t| t.text.as_str()).collect();
    let tvecs = embedder
        .embed(&trefs)
        .map_err(|e| ProcessError::Runtime(e.to_string()))?;
    // for each cv, load embedding, compute cosine
    for cv_hash in cv_hashes {
        let bytes = match ctx.db.get_embedding(cv_hash, model_id)? {
            Some(b) => b,
            None => continue,
        };
        let cv_vec = libtriqui_core::embed::bytes_to_vector(&bytes, None)
            .map_err(|e| ProcessError::Runtime(e.to_string()))?;
        for (ti, target) in targets.iter().enumerate() {
            let sim = libtriqui_core::embed::cosine(&cv_vec, &tvecs[ti]) as f64;
            ctx.db
                .upsert_similarity(cv_hash, model_id, &target.hash, sim)?;
        }
    }
    Ok(())
}
