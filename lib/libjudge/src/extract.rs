//! Chunk embedding with the step-14 chunk cache: byte-identical text prep
//! and identical cache keys to `bench`'s `EmbedCache`, so already-cached
//! chunk embeddings are reused as-is (design 15.4/15.11). Feature prep is
//! this one rust code path end-to-end for both the trainer and bench (15.10).

use std::collections::HashMap;
use std::path::PathBuf;

use libcvevidence::CvEvidence;
use libtokenizer::TokenizerHandle;
use sha2::{Digest, Sha256};

use crate::artifact::{ChunkingMode, Contextual, FeatureSpec, FeatureVariant};
use crate::evidence::{ad_terms, build_vocabulary, compute_features, today_utc, x_block};
use crate::features::{doc_features, DocFeatures, TextPrep};
use crate::JudgeError;

/// Contextual prefixes — must match bench step 14 byte-for-byte.
pub const RESUME_PREFIX: &str = "This is a chunk from a CV of an applicant applying for a position. ";
pub const JOB_PREFIX: &str = "This is a chunk from a job advertisement describing a position. ";

/// Embedding backend seam. `libmodelcallerclient`'s client adapts into this
/// at the consumer side; tests can plug a fake embedder.
pub trait TextEmbedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, JudgeError>;
}

/// File cache with bench step-14's exact on-disk contract:
/// key = first 16 bytes of sha256(model \0 trimmed-text) as hex, file content
/// = `{"embedding": [...]}`. Bench's `EmbedCache` and this type are
/// interoperable (a parity test in bench asserts key equality).
pub struct ChunkCache {
    dir: PathBuf,
}

impl ChunkCache {
    pub fn new(dir: &str) -> Result<Self, JudgeError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| JudgeError::Other(format!("failed to create cache dir '{dir}': {e}")))?;
        Ok(Self {
            dir: PathBuf::from(dir),
        })
    }

    pub fn disabled() -> Self {
        Self { dir: PathBuf::new() }
    }

    pub fn enabled(&self) -> bool {
        !self.dir.as_os_str().is_empty()
    }

    /// Cache key — keep byte-identical to bench `EmbedCache::key`.
    pub fn key(model: &str, text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b"\0");
        hasher.update(text.trim().as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(16).map(|b| format!("{b:02x}")).collect()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    pub fn get(&self, model: &str, text: &str) -> Option<Vec<f64>> {
        if !self.enabled() {
            return None;
        }
        let data = std::fs::read_to_string(self.path(&Self::key(model, text))).ok()?;
        let val: serde_json::Value = serde_json::from_str(&data).ok()?;
        val.get("embedding")?.as_array()?.iter().map(|v| v.as_f64()).collect()
    }

    pub fn put(&self, model: &str, text: &str, embedding: &[f64]) {
        if !self.enabled() {
            return;
        }
        let val = serde_json::json!({ "embedding": embedding });
        let _ = std::fs::write(self.path(&Self::key(model, text)), val.to_string());
    }
}

/// Chunking parameters — same formula as bench `chunk_params` (step 14).
pub fn chunk_params(mode: ChunkingMode, max_ctx: usize, explicit_size: Option<usize>) -> (usize, usize) {
    match mode {
        ChunkingMode::None => (max_ctx, 0),
        ChunkingMode::CtxSize => {
            let cs = ((max_ctx as f64 * 0.9) as usize).max(1);
            let overlap = ((cs as f64 * 0.15) as usize).max(1);
            (cs, overlap)
        }
        ChunkingMode::Fixed => {
            let cs = explicit_size.unwrap_or(256).max(1);
            let overlap = ((cs as f64 * 0.15) as usize).max(1);
            (cs, overlap)
        }
    }
}

/// Stage-a evidence cache for the `fused` variant (design 18 §18.4/18.6):
/// one `CvEvidence` json per cv under `<dir>/<key>.json`, key =
/// sha256(stage \0 model \0 schema \0 trimmed-text). On-disk contract is
/// byte-identical to bench's `EvidenceCache`, so trainer and bench share
/// the same entries (same interop pattern as `ChunkCache`/`EmbedCache`).
/// `model` is `-` for the deterministic stage-a parser.
#[derive(Clone)]
pub struct EvidenceCache {
    dir: PathBuf,
}

impl EvidenceCache {
    pub fn new(dir: &str) -> Result<Self, JudgeError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| JudgeError::Other(format!("failed to create evidence cache dir '{dir}': {e}")))?;
        Ok(Self { dir: PathBuf::from(dir) })
    }

    pub fn disabled() -> Self {
        Self { dir: PathBuf::new() }
    }

    pub fn enabled(&self) -> bool {
        !self.dir.as_os_str().is_empty()
    }

    /// Cache key — keep byte-identical to bench `EvidenceCache::key`.
    pub fn key(stage: &str, model: &str, schema: &str, text: &str) -> String {
        let mut hasher = Sha256::new();
        for part in [stage, model, schema] {
            hasher.update(part.as_bytes());
            hasher.update(b"\0");
        }
        hasher.update(text.trim().as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(16).map(|b| format!("{b:02x}")).collect()
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    pub fn get(&self, key: &str) -> Option<CvEvidence> {
        if !self.enabled() {
            return None;
        }
        let data = std::fs::read_to_string(self.path(key)).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn put(&self, key: &str, evidence: &CvEvidence) {
        if !self.enabled() {
            return;
        }
        if let Ok(json) = serde_json::to_string(evidence) {
            let _ = std::fs::write(self.path(key), json);
        }
    }
}

/// Turns raw text into `DocFeatures` through chunking + the cache + an
/// embedder, batching all cache misses of one observation into single embed
/// calls. One rust code path for trainer and bench.
pub struct FeatureExtractor {
    pub model: String,
    pub tokenizer: Option<TokenizerHandle>,
    pub chunking: ChunkingMode,
    pub chunk_size: Option<usize>,
    pub contextual: Contextual,
    pub preprocessing: TextPrep,
    pub max_ctx: usize,
    pub cache: ChunkCache,
    pub embedder: Box<dyn TextEmbedder>,
    pub embed_batch: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    /// `true` when the spec's variant is `Fused` — `extract` then parses
    /// each cv (stage-a, evidence-cached) and fills `DocFeatures::x`.
    pub fused: bool,
    /// Evidence cache for the fused variant; disabled by default, set by
    /// the consumer (bench context / judgetrainer) after construction.
    pub evidence_cache: EvidenceCache,
}

/// One observation's texts: the job ad plus its resumes.
pub struct ObservationTexts<'a> {
    pub job_text: &'a str,
    pub resume_texts: Vec<&'a str>,
}

impl FeatureExtractor {
    pub fn from_spec(
        spec: &FeatureSpec,
        max_ctx: usize,
        tokenizer: Option<TokenizerHandle>,
        cache: ChunkCache,
        embedder: Box<dyn TextEmbedder>,
    ) -> Self {
        Self {
            model: spec.embedder.clone(),
            tokenizer,
            chunking: spec.chunking,
            chunk_size: spec.chunk_size,
            contextual: spec.contextual,
            preprocessing: spec.text_preprocessing,
            max_ctx,
            cache,
            embedder,
            embed_batch: 64,
            cache_hits: 0,
            cache_misses: 0,
            fused: spec.variant == FeatureVariant::Fused,
            evidence_cache: EvidenceCache::disabled(),
        }
    }

    fn preprocess(&self, text: &str) -> String {
        match self.preprocessing {
            TextPrep::None => text.to_string(),
            TextPrep::Lowercase => text.to_lowercase(),
        }
    }

    fn prefix(&self, is_job: bool) -> &'static str {
        match self.contextual {
            Contextual::Off => "",
            Contextual::Static => {
                if is_job {
                    JOB_PREFIX
                } else {
                    RESUME_PREFIX
                }
            }
        }
    }

    /// Pre-prefixed chunk texts for one document.
    fn chunk_texts(&self, text: &str, is_job: bool) -> Result<Vec<String>, JudgeError> {
        let preprocessed = self.preprocess(text);
        let prefix = self.prefix(is_job);
        match self.chunking {
            ChunkingMode::None => {
                // truncate to ~90% of the model context, mirroring bench
                let estimated_tokens = (preprocessed.len() as f64 * 0.25).ceil() as usize;
                let truncated = if estimated_tokens > self.max_ctx {
                    let ratio = (self.max_ctx as f64 * 0.9) / estimated_tokens as f64;
                    let keep = ((preprocessed.chars().count() as f64) * ratio) as usize;
                    preprocessed.chars().take(keep).collect()
                } else {
                    preprocessed
                };
                Ok(vec![format!("{prefix}{truncated}")])
            }
            ChunkingMode::CtxSize | ChunkingMode::Fixed => {
                let tokenizer = self
                    .tokenizer
                    .as_ref()
                    .ok_or_else(|| JudgeError::Other("chunking requested but no tokenizer loaded".into()))?;
                let (cs, overlap) = chunk_params(self.chunking, self.max_ctx, self.chunk_size);
                let chunks = tokenizer
                    .chunk_text(&preprocessed, cs, overlap)
                    .map_err(|e| JudgeError::Other(format!("chunking failed: {e}")))?;
                Ok(chunks
                    .into_iter()
                    .map(|c| format!("{prefix}{}", c.text))
                    .collect())
            }
        }
    }

    /// Embed one (pre-prefixed) text with cache read/write. Use only for
    /// one-off texts; prefer [`extract`](Self::extract) for observations.
    pub fn embed_cached(&mut self, text: &str) -> Result<Vec<f64>, JudgeError> {
        if let Some(emb) = self.cache.get(&self.model, text) {
            self.cache_hits += 1;
            return Ok(emb);
        }
        self.cache_misses += 1;
        let embs = self.embedder.embed(&[text.to_string()])?;
        match embs.into_iter().next().filter(|e| !e.iter().all(|&v| v == 0.0)) {
            Some(emb) => {
                self.cache.put(&self.model, text, &emb);
                Ok(emb)
            }
            None => Err(JudgeError::Other("embedding returned empty or zero vector".into())),
        }
    }

    /// Extract features for one observation: each resume's blocks, with `a`
    /// derived from the ad's chunk embeddings. All cache misses are
    /// deduplicated and embedded in batches of `embed_batch` texts per call.
    /// Returns one `DocFeatures` per resume, in input order.
    pub fn extract(
        &mut self,
        texts: ObservationTexts<'_>,
    ) -> Result<Vec<DocFeatures>, JudgeError> {
        let job_chunk_texts = self.chunk_texts(texts.job_text, true)?;
        let resume_chunk_texts: Vec<Vec<String>> = texts
            .resume_texts
            .iter()
            .map(|&t| self.chunk_texts(t, false))
            .collect::<Result<_, _>>()?;

        // resolve cache hits; collect unique misses
        let mut emb_by_text: HashMap<String, Vec<f64>> = HashMap::new();
        let mut misses: Vec<String> = Vec::new();
        {
            let mut lookup = |this: &mut Self, t: &str, misses: &mut Vec<String>| {
                if emb_by_text.contains_key(t) {
                    return;
                }
                if let Some(emb) = this.cache.get(&this.model, t) {
                    this.cache_hits += 1;
                    emb_by_text.insert(t.to_string(), emb);
                } else {
                    misses.push(t.to_string());
                }
            };
            for t in &job_chunk_texts {
                lookup(self, t, &mut misses);
            }
            for group in &resume_chunk_texts {
                for t in group {
                    lookup(self, t, &mut misses);
                }
            }
        }

        // embed misses in batches
        for batch in misses.chunks(self.embed_batch.max(1)) {
            self.cache_misses += batch.len();
            let embs = self.embedder.embed(batch)?;
            if embs.len() != batch.len() {
                return Err(JudgeError::Other(format!(
                    "embedder returned {} vectors for {} texts",
                    embs.len(),
                    batch.len()
                )));
            }
            for (t, emb) in batch.iter().zip(embs.into_iter()) {
                if emb.iter().all(|&v| v == 0.0) {
                    return Err(JudgeError::Other("embedding returned empty or zero vector".into()));
                }
                self.cache.put(&self.model, t, &emb);
                emb_by_text.insert(t.clone(), emb);
            }
        }

        let resolve = |texts: &[String]| -> Result<Vec<Vec<f64>>, JudgeError> {
            texts
                .iter()
                .map(|t| {
                    emb_by_text
                        .get(t)
                        .cloned()
                        .ok_or_else(|| JudgeError::Other("internal: missing embedding".into()))
                })
                .collect()
        };

        let job_chunks = resolve(&job_chunk_texts)?;

        let mut docs = Vec::with_capacity(resume_chunk_texts.len());
        for group in &resume_chunk_texts {
            let cv_chunks = resolve(group)?;
            docs.push(doc_features(&job_chunks, &cv_chunks)?);
        }

        if self.fused {
            fill_evidence_blocks(self, texts, &mut docs)?;
        }
        Ok(docs)
    }
}

/// Fused-variant evidence extraction (design 18 §18.3/§18.5): vocabulary
/// from this observation's resumes (identical derivation to the
/// `evidence-rules` method — deterministic, no leakage), stage-a parse per
/// cv through the evidence cache, then the x block per cv.
fn fill_evidence_blocks(
    extractor: &mut FeatureExtractor,
    texts: ObservationTexts<'_>,
    docs: &mut [DocFeatures],
) -> Result<(), JudgeError> {
    let resume_texts: Vec<String> = texts.resume_texts.iter().map(|t| (*t).to_string()).collect();
    let vocabulary = build_vocabulary(&resume_texts, 1500);
    let terms = ad_terms(texts.job_text, &vocabulary);
    let now = today_utc();
    let opts = libcvparser::EvidenceOptions {
        vocabulary: &vocabulary,
        now,
    };

    if docs.len() != texts.resume_texts.len() {
        return Err(JudgeError::Other("internal: doc/chunk count mismatch".into()));
    }
    for (i, text) in texts.resume_texts.iter().enumerate() {
        let key = EvidenceCache::key("stage-a", "-", "schema-1", text);
        let ev = match extractor.evidence_cache.get(&key) {
            Some(ev) => ev,
            None => {
                let parsed = libcvparser::parse_evidence(text, &opts);
                extractor.evidence_cache.put(&key, &parsed);
                parsed
            }
        };
        let feats = compute_features(texts.job_text, &ev, &terms);
        docs[i].x = x_block(&feats);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ChunkingMode, Contextual};
    use crate::evidence::X_BLOCK_LEN;
    use crate::features::TextPrep;

    struct ConstEmbedder;
    impl TextEmbedder for ConstEmbedder {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, JudgeError> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v: Vec<f64> = t.bytes().map(|b| (b % 7) as f64 + 1.0).collect();
                    v.resize(8, 1.0);
                    v
                })
                .collect())
        }
    }

    fn extractor(spec: &FeatureSpec, evidence_dir: Option<&str>) -> FeatureExtractor {
        let cache = ChunkCache::disabled();
        let mut ex = FeatureExtractor::from_spec(
            spec,
            4096,
            None,
            cache,
            Box::new(ConstEmbedder),
        );
        if let Some(dir) = evidence_dir {
            ex.evidence_cache = EvidenceCache::new(dir).unwrap();
        }
        ex
    }

    fn observation<'a>(job: &'a str, resumes: Vec<&'a str>) -> ObservationTexts<'a> {
        ObservationTexts {
            job_text: job,
            resume_texts: resumes,
        }
    }

    #[test]
    fn raw_variant_leaves_x_empty() {
        let spec = FeatureSpec::raw_pair("m", 8, ChunkingMode::None, None, Contextual::Off, TextPrep::None);
        let mut ex = extractor(&spec, None);
        let docs = ex.extract(observation("java job", vec!["Experience04/2018toCurrentJava DevAcme", "nothing here"])).unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().all(|d| d.x.is_empty()));
    }

    #[test]
    fn fused_variant_fills_x_and_reuses_evidence_cache() {
        let dir = std::env::temp_dir().join("lj_fused_cache_test");
        let _ = std::fs::remove_dir_all(&dir);
        let spec = FeatureSpec::fused_pair("m", 8, ChunkingMode::None, None, Contextual::Off, TextPrep::None);
        let mut ex = extractor(&spec, Some(dir.to_str().unwrap()));
        let rich = "Experience04/2018toCurrentSenior Java DeveloperAcmeCorp-Lyon,FR,Built kafka pipelines with spring boot";
        // filler resumes keep the per-observation vocabulary non-degenerate
        // (df >= 2 across resumes): "java" reaches df 4 of 5
        let fillers = [
            "Skills java testing services five years experience",
            " java backend developer sql apis years",
            "java microservices docker tooling years",
            "completely unrelated warehouse operator",
        ];
        let docs = ex.extract(observation(
            "senior java developer with kafka",
            vec![rich, "no evidence at all", fillers[0], fillers[1], fillers[2], fillers[3]],
        ))
        .unwrap();
        assert_eq!(docs[0].x.len(), X_BLOCK_LEN);
        assert_eq!(docs[1].x.len(), X_BLOCK_LEN);
        // the evidenced cv must show depth + coverage; the bare one must not
        assert!(docs[0].x[2] > 0.0, "skill depth: {}", docs[0].x[2]);
        assert_eq!(docs[0].x[9], 1.0, "coverage_roles");
        assert_eq!(docs[1].x[9], 0.0);

        // second run: same texts -> cache hits, identical blocks
        let hits_before = ex.evidence_cache.get(&EvidenceCache::key("stage-a", "-", "schema-1", rich)).is_some();
        assert!(hits_before);
        let docs2 = ex.extract(observation(
            "senior java developer with kafka",
            vec![rich, "no evidence at all", fillers[0], fillers[1], fillers[2], fillers[3]],
        ))
        .unwrap();
        assert_eq!(docs[0].x, docs2[0].x);
    }

    #[test]
    fn evidence_cache_key_matches_contract() {
        // stage \0 model \0 schema \0 text, first 16 bytes of sha256 as hex
        let k = EvidenceCache::key("stage-a", "-", "schema-1", "  padded  ");
        assert_eq!(k.len(), 32);
        assert_eq!(k, EvidenceCache::key("stage-a", "-", "schema-1", "padded"));
        assert_ne!(k, EvidenceCache::key("stage-b", "-", "schema-1", "padded"));
        assert!(EvidenceCache::disabled().get("x").is_none());
    }
}
