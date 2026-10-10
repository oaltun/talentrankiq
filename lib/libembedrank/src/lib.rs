use thiserror::Error;

use libtokenizer::TokenizerHandle;

#[derive(Debug, Error)]
pub enum EmbedRankError {
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] libtokenizer::TokenizerError),
    #[error("embedding error: {0}")]
    Embed(String),
    #[error("no chunks produced for text ({} chars)", .0.len())]
    NoChunks(String),
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, EmbedRankError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub enum RetrievalStrategy {
    #[default]
    MaxSim,
    Centroid,
    MeanTopK,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub overlap_ratio: f64,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 0,
            overlap_ratio: 0.15,
        }
    }
}

impl ChunkingConfig {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            overlap_ratio: 0.15,
        }
    }

    pub fn overlap(&self) -> usize {
        ((self.chunk_size as f64 * self.overlap_ratio) as usize).max(1)
    }

    pub fn is_active(&self) -> bool {
        self.chunk_size > 0
    }
}

pub struct ChunkedEmbedding {
    pub chunks: Vec<Vec<f64>>,
}

impl ChunkedEmbedding {
    pub fn new(chunks: Vec<Vec<f64>>) -> Self {
        Self { chunks }
    }

    pub fn is_single(&self) -> bool {
        self.chunks.len() <= 1
    }

    pub fn centroid(&self) -> Option<Vec<f64>> {
        if self.chunks.is_empty() {
            return None;
        }
        if self.chunks.len() == 1 {
            return Some(self.chunks[0].clone());
        }
        let dim = self.chunks[0].len();
        if dim == 0 {
            return None;
        }
        let mut centroid = vec![0.0f64; dim];
        for emb in &self.chunks {
            for (i, v) in emb.iter().enumerate() {
                centroid[i] += v;
            }
        }
        let n = self.chunks.len() as f64;
        for v in centroid.iter_mut() {
            *v /= n;
        }
        Some(centroid)
    }

    pub fn max_sim(&self, query: &[f64]) -> f64 {
        self.chunks
            .iter()
            .map(|c| cosine_similarity(query, c))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// MaxSim score and the indices of the top-K most similar chunks (descending).
    pub fn max_sim_top_k(&self, query: &[f64], k: usize) -> (f64, Vec<usize>) {
        let mut scored: Vec<(f64, usize)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (cosine_similarity(query, c), i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let score = scored.first().map(|&(s, _)| s).unwrap_or(0.0);
        let indices: Vec<usize> = scored.into_iter().take(k).map(|(_, i)| i).collect();
        (score, indices)
    }

    pub fn aggregate_score(&self, query: &[f64], strategy: RetrievalStrategy) -> f64 {
        match strategy {
            RetrievalStrategy::MaxSim => self.max_sim(query),
            RetrievalStrategy::Centroid => {
                match self.centroid() {
                    Some(c) => cosine_similarity(query, &c),
                    None => 0.0,
                }
            }
            RetrievalStrategy::MeanTopK => {
                let k = 3.min(self.chunks.len()).max(1);
                let mut sims: Vec<f64> = self
                    .chunks
                    .iter()
                    .map(|c| cosine_similarity(query, c))
                    .collect();
                sims.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                sims[..k].iter().sum::<f64>() / k as f64
            }
        }
    }
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub fn embed_text_chunked(
    text: &str,
    tokenizer: &TokenizerHandle,
    config: &ChunkingConfig,
    embedder: &dyn Embedder,
    prefix: &str,
) -> Result<ChunkedEmbedding, EmbedRankError> {
    let chunks = tokenizer.chunk_text(text, config.chunk_size, config.overlap())?;
    if chunks.is_empty() {
        return Err(EmbedRankError::NoChunks(text.to_string()));
    }
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| {
            if prefix.is_empty() {
                c.text.clone()
            } else {
                format!("{prefix}{}", c.text)
            }
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(&refs)?;
    Ok(ChunkedEmbedding::new(embeddings))
}

pub fn document_embedding(
    text: &str,
    tokenizer: &TokenizerHandle,
    config: &ChunkingConfig,
    embedder: &dyn Embedder,
    prefix: &str,
) -> Result<Vec<f64>, EmbedRankError> {
    let chunked = embed_text_chunked(text, tokenizer, config, embedder, prefix)?;
    chunked
        .centroid()
        .ok_or_else(|| EmbedRankError::NoChunks(text.to_string()))
}

/// Chunk a text, embed all chunks, and return both the `ChunkedEmbedding` and the
/// raw chunk texts (after prefix application). Used by callers that need to map
/// chunk indices back to source text for explanations.
pub fn embed_text_chunked_with_texts(
    text: &str,
    tokenizer: &TokenizerHandle,
    config: &ChunkingConfig,
    embedder: &dyn Embedder,
    prefix: &str,
) -> Result<(ChunkedEmbedding, Vec<String>), EmbedRankError> {
    let chunks = tokenizer.chunk_text(text, config.chunk_size, config.overlap())?;
    if chunks.is_empty() {
        return Err(EmbedRankError::NoChunks(text.to_string()));
    }
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| {
            if prefix.is_empty() {
                c.text.clone()
            } else {
                format!("{prefix}{}", c.text)
            }
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed(&refs)?;
    Ok((ChunkedEmbedding::new(embeddings), texts))
}

pub fn score_document(
    text: &str,
    query: &[f64],
    tokenizer: &TokenizerHandle,
    config: &ChunkingConfig,
    embedder: &dyn Embedder,
    prefix: &str,
    strategy: RetrievalStrategy,
) -> Result<f64, EmbedRankError> {
    let chunked = embed_text_chunked(text, tokenizer, config, embedder, prefix)?;
    Ok(chunked.aggregate_score(query, strategy))
}

pub fn query_embedding(
    text: &str,
    tokenizer: &TokenizerHandle,
    config: &ChunkingConfig,
    embedder: &dyn Embedder,
    prefix: &str,
) -> Result<Vec<f64>, EmbedRankError> {
    let chunked = embed_text_chunked(text, tokenizer, config, embedder, prefix)?;
    if chunked.is_single() {
        chunked
            .chunks
            .into_iter()
            .next()
            .ok_or_else(|| EmbedRankError::NoChunks(text.to_string()))
    } else {
        chunked
            .centroid()
            .ok_or_else(|| EmbedRankError::NoChunks(text.to_string()))
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbedder {
        dim: usize,
    }

    impl MockEmbedder {
        fn new(dim: usize) -> Self {
            Self { dim }
        }

        fn hash_vec(&self, text: &str) -> Vec<f64> {
            let mut v = vec![0.0_f64; self.dim];
            for (i, b) in text.as_bytes().iter().enumerate() {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                v[i % self.dim] += sign * (*b as f64);
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            v
        }
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, EmbedRankError> {
            Ok(texts.iter().map(|t| self.hash_vec(t)).collect())
        }
    }

    fn make_test_dir() -> std::path::PathBuf {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let tid = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("libembedrank_test_{tid}"));
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
        let json = serde_json::to_string(&vocab).unwrap();
        let path = dir.join("tokenizer.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, "{}", json).unwrap();
        dir
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
    }

    #[test]
    fn cosine_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn chunked_embedding_centroid_two_chunks() {
        let dir = make_test_dir();
        let tok = TokenizerHandle::from_file(&dir.join("tokenizer.json")).unwrap();
        let embedder = MockEmbedder::new(8);
        let config = ChunkingConfig::new(4);
        let text = "the quick brown fox jumps over the lazy dog";
        let result = document_embedding(text, &tok, &config, &embedder, "").unwrap();
        assert_eq!(result.len(), 8);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn chunked_embedding_single_chunk_passthrough() {
        let dir = make_test_dir();
        let tok = TokenizerHandle::from_file(&dir.join("tokenizer.json")).unwrap();
        let embedder = MockEmbedder::new(8);
        let config = ChunkingConfig::new(100);
        let text = "the quick brown fox";
        let result = document_embedding(text, &tok, &config, &embedder, "").unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn aggregate_max_sim_picks_best() {
        let query = vec![1.0, 0.0];
        let ce = ChunkedEmbedding::new(vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.8, 0.6],
        ]);
        let score = ce.aggregate_score(&query, RetrievalStrategy::MaxSim);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_centroid() {
        let query = vec![1.0, 0.0];
        let ce = ChunkedEmbedding::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let score = ce.aggregate_score(&query, RetrievalStrategy::Centroid);
        let expected = cosine_similarity(&query, &[0.5, 0.5]);
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn aggregate_mean_top_k() {
        let query = vec![1.0, 0.0];
        let ce = ChunkedEmbedding::new(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.8, 0.6],
        ]);
        let score = ce.aggregate_score(&query, RetrievalStrategy::MeanTopK);
        let sims: Vec<f64> = [1.0, 0.0, 0.8].to_vec();
        let mut sorted = sims.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let k = 3;
        let expected: f64 = sorted[..k].iter().sum::<f64>() / k as f64;
        assert!((score - expected).abs() < 1e-9);
    }

    #[test]
    fn centroid_of_single_chunk_is_the_chunk() {
        let ce = ChunkedEmbedding::new(vec![vec![1.0, 2.0, 3.0]]);
        let c = ce.centroid().unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn centroid_of_empty_is_none() {
        let ce = ChunkedEmbedding::new(vec![]);
        assert!(ce.centroid().is_none());
    }

    #[test]
    fn chunking_config_defaults() {
        let cfg = ChunkingConfig::default();
        assert!(!cfg.is_active());
        assert_eq!(cfg.chunk_size, 0);
    }

    #[test]
    fn chunking_config_overlap() {
        let cfg = ChunkingConfig::new(1024);
        assert!(cfg.is_active());
        assert_eq!(cfg.overlap(), 153);
    }

    #[test]
    fn prefix_prepended_to_chunks() {
        let dir = make_test_dir();
        let tok = TokenizerHandle::from_file(&dir.join("tokenizer.json")).unwrap();
        let embedder = MockEmbedder::new(8);
        let config = ChunkingConfig::new(100);
        let plain = document_embedding("the fox", &tok, &config, &embedder, "").unwrap();
        let prefixed = document_embedding("the fox", &tok, &config, &embedder, "PREFIX: ").unwrap();
        assert_ne!(plain, prefixed);
    }
}
