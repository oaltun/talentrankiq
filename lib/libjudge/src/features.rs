//! Judge input assembly (design 15.4).
//!
//! Per (job, cv) document, from cached chunk embeddings:
//! - `a` = L2-normalized centroid of the job ad's chunk embeddings
//! - `e` = L2-normalized centroid of the cv's chunk embeddings
//! - `c` = the cv chunk most similar to `a` (argmax cosine), L2-normalized;
//!   with a single chunk `c = e`
//!
//! Raw (default) pair input: `x = [a; e1; e2; c1; c2]` — the head learns the
//! interactions. Engineered (ablation-only) pair input:
//! `[e1; e2; e1-e2; e1*e2; a; h1; h2; h1-h2; s]` with `h = a*e` and
//! `s = [max, mean, top3, coverage]` (tau = 0.5), 8*dim + 4 dims.

use serde::{Deserialize, Serialize};

use crate::artifact::{Block, FeatureSpec, FeatureVariant};
use crate::evidence::X_BLOCK_LEN;
use crate::{cosine, l2_normalize, JudgeError};

/// Text preprocessing applied before chunking/embedding (mirrors bench).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextPrep {
    None,
    Lowercase,
}

/// Feature blocks of one document.
#[derive(Clone, Debug)]
pub struct DocFeatures {
    pub a: Vec<f32>,
    pub e: Vec<f32>,
    pub c: Vec<f32>,
    /// Number of chunks the document was split into.
    pub n_chunks: usize,
    /// Cosine of each (raw) chunk against `a`; kept for the engineered
    /// stats block. Empty for raw-only usage.
    pub chunk_sims: Vec<f64>,
    /// Fused-variant evidence block (design 18 §18.5), [`X_BLOCK_LEN`]
    /// scalars. Empty unless the spec's variant is `Fused` and the
    /// extractor computed it (evidence extraction needs the job ad, so it
    /// is filled per observation inside `FeatureExtractor::extract`).
    pub x: Vec<f32>,
}

/// Build the per-document blocks `a`/`e`/`c` from raw chunk embeddings.
/// `job_chunks` are the ad's chunk embeddings (already the preprocessed,
/// prefixed strings' embeddings); `cv_chunks` the cv's.
pub fn doc_features(job_chunks: &[Vec<f64>], cv_chunks: &[Vec<f64>]) -> Result<DocFeatures, JudgeError> {
    let dim = cv_chunks
        .first()
        .map(|c| c.len())
        .ok_or_else(|| JudgeError::Other("cv has no chunks".into()))?;
    if job_chunks.is_empty() {
        return Err(JudgeError::Other("job ad has no chunks".into()));
    }
    if job_chunks.iter().chain(cv_chunks.iter()).any(|c| c.len() != dim) {
        return Err(JudgeError::DimMismatch {
            expected: dim,
            got: 0,
        });
    }

    let centroid = |chunks: &[Vec<f64>]| -> Vec<f64> {
        let mut acc = vec![0.0f64; dim];
        for c in chunks {
            for (acc_x, x) in acc.iter_mut().zip(c.iter()) {
                *acc_x += x;
            }
        }
        let n = chunks.len() as f64;
        for x in acc.iter_mut() {
            *x /= n;
        }
        acc
    };

    let mut a = centroid(job_chunks);
    l2_normalize(&mut a);
    let mut e = centroid(cv_chunks);
    l2_normalize(&mut e);

    let chunk_sims: Vec<f64> = cv_chunks.iter().map(|ch| cosine(ch, &a)).collect();
    let best = chunk_sims
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut c = cv_chunks[best].clone();
    l2_normalize(&mut c);

    Ok(DocFeatures {
        a: a.iter().map(|&x| x as f32).collect(),
        e: e.iter().map(|&x| x as f32).collect(),
        c: c.iter().map(|&x| x as f32).collect(),
        n_chunks: cv_chunks.len(),
        chunk_sims,
        x: Vec::new(),
    })
}

/// Raw-variant input for an ordered pair (cv1 better than cv2):
/// `x = [a; e1; e2; c1; c2]`, `5*dim` dims.
#[allow(unused_variables)]
pub fn assemble_raw_pair(spec: &FeatureSpec, ad: &DocFeatures, cv1: &DocFeatures, cv2: &DocFeatures) -> Vec<f32> {
    debug_assert_eq!(spec.variant, FeatureVariant::Raw);
    assemble_raw_pair_inner(ad, cv1, cv2)
}

/// Engineered-variant input for an ordered pair (ablation only, 15.4):
/// `[e1; e2; e1-e2; e1*e2; a; h1; h2; h1-h2; s1; s2]`, `8*dim + 8` dims,
/// with `h = a*e` and per-cv stats `s = [max, mean, top3, coverage]` at tau.
pub fn assemble_engineered_pair(
    spec: &FeatureSpec,
    ad: &DocFeatures,
    cv1: &DocFeatures,
    cv2: &DocFeatures,
) -> Vec<f32> {
    debug_assert_eq!(spec.variant, FeatureVariant::Engineered);
    let dim = spec.dim;
    let tau = spec.tau.unwrap_or(0.5);

    let h = |cv: &DocFeatures| -> Vec<f32> {
        ad.a.iter().zip(cv.e.iter()).map(|(x, y)| x * y).collect()
    };
    let stats = |cv: &DocFeatures| -> Vec<f32> {
        let sims = &cv.chunk_sims;
        if sims.is_empty() {
            return vec![0.0; 4];
        }
        let max = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = sims.iter().sum::<f64>() / sims.len() as f64;
        let mut sorted = sims.clone();
        sorted.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        let k = 3.min(sorted.len());
        let top3 = sorted[..k].iter().sum::<f64>() / k as f64;
        let coverage = sims.iter().filter(|&&s| s > tau).count() as f64 / sims.len() as f64;
        vec![max, mean, top3, coverage].into_iter().map(|v| v as f32).collect()
    };

    let mut x = Vec::with_capacity(spec.input_dim);
    x.extend_from_slice(&cv1.e);
    x.extend_from_slice(&cv2.e);
    for i in 0..dim {
        x.push(cv1.e[i] - cv2.e[i]);
    }
    for i in 0..dim {
        x.push(cv1.e[i] * cv2.e[i]);
    }
    x.extend_from_slice(&ad.a);
    let h1 = h(cv1);
    let h2 = h(cv2);
    x.extend_from_slice(&h1);
    x.extend_from_slice(&h2);
    for i in 0..dim {
        x.push(h1[i] - h2[i]);
    }
    x.extend_from_slice(&stats(cv1));
    x.extend_from_slice(&stats(cv2));
    x
}

/// Fused-variant input for an ordered pair (design 18 §18.5):
/// `[a; e1; e2; c1; c2; x1; x2]` — the raw embedding blocks unchanged,
/// plus the per-cv evidence scalars. Requires both docs to carry an `x`
/// block (the extractor fills it when the spec variant is `Fused`).
pub fn assemble_fused_pair(spec: &FeatureSpec, ad: &DocFeatures, cv1: &DocFeatures, cv2: &DocFeatures) -> Vec<f32> {
    debug_assert_eq!(spec.variant, FeatureVariant::Fused);
    assert_eq!(
        cv1.x.len(),
        X_BLOCK_LEN,
        "fused pair needs an evidence x block on cv1 (variant fused requires evidence extraction)"
    );
    assert_eq!(
        cv2.x.len(),
        X_BLOCK_LEN,
        "fused pair needs an evidence x block on cv2 (variant fused requires evidence extraction)"
    );
    let mut x = assemble_raw_pair_inner(ad, cv1, cv2);
    x.extend_from_slice(&cv1.x);
    x.extend_from_slice(&cv2.x);
    x
}

fn assemble_raw_pair_inner(ad: &DocFeatures, cv1: &DocFeatures, cv2: &DocFeatures) -> Vec<f32> {
    let mut x = Vec::with_capacity(5 * ad.a.len());
    x.extend_from_slice(&ad.a);
    x.extend_from_slice(&cv1.e);
    x.extend_from_slice(&cv2.e);
    x.extend_from_slice(&cv1.c);
    x.extend_from_slice(&cv2.c);
    x
}

/// Assemble the pair input according to the spec's variant.
pub fn assemble_pair(spec: &FeatureSpec, ad: &DocFeatures, cv1: &DocFeatures, cv2: &DocFeatures) -> Vec<f32> {
    match spec.variant {
        FeatureVariant::Raw => assemble_raw_pair(spec, ad, cv1, cv2),
        FeatureVariant::Engineered => assemble_engineered_pair(spec, ad, cv1, cv2),
        FeatureVariant::Fused => assemble_fused_pair(spec, ad, cv1, cv2),
    }
}

/// Pointwise (m0) input for one document: `x = [a; e; c]`, `3*dim` dims.
pub fn assemble_pointwise(spec: &FeatureSpec, ad: &DocFeatures, cv: &DocFeatures) -> Vec<f32> {
    let mut x = Vec::with_capacity(spec.input_dim);
    x.extend_from_slice(&ad.a);
    x.extend_from_slice(&cv.e);
    x.extend_from_slice(&cv.c);
    x
}

/// Block layout for the raw pair input: a, e1, e2, c1, c2.
pub fn raw_blocks(dim: usize) -> Vec<Block> {
    let names = ["a", "e1", "e2", "c1", "c2"];
    names
        .iter()
        .enumerate()
        .map(|(i, name)| Block {
            name: name.to_string(),
            start: i * dim,
            len: dim,
        })
        .collect()
}

/// Block layout for the fused pair input (design 18 §18.5):
/// raw blocks + one evidence block per cv.
pub fn fused_blocks(dim: usize) -> Vec<Block> {
    let mut blocks = raw_blocks(dim);
    let mut start = 5 * dim;
    for name in ["x1", "x2"] {
        blocks.push(Block {
            name: name.to_string(),
            start,
            len: X_BLOCK_LEN,
        });
        start += X_BLOCK_LEN;
    }
    blocks
}

/// Block layout for the engineered pair input (15.4).
pub fn engineered_blocks(dim: usize) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut start = 0;
    for name in ["e1", "e2", "e1-e2", "e1*e2", "a", "h1", "h2", "h1-h2"] {
        blocks.push(Block {
            name: name.to_string(),
            start,
            len: dim,
        });
        start += dim;
    }
    for name in ["s1", "s2"] {
        blocks.push(Block {
            name: name.to_string(),
            start,
            len: 4,
        });
        start += 4;
    }
    blocks
}

/// Block layout for the pointwise input: a, e, c.
pub fn pointwise_blocks(dim: usize) -> Vec<Block> {
    let names = ["a", "e", "c"];
    names
        .iter()
        .enumerate()
        .map(|(i, name)| Block {
            name: name.to_string(),
            start: i * dim,
            len: dim,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{ChunkingMode, Contextual};

    fn chunk(v: &[f64]) -> Vec<f64> {
        v.to_vec()
    }

    #[test]
    fn best_chunk_is_most_similar_to_ad() {
        // ad along axis 0
        let job = vec![chunk(&[1.0, 0.0])];
        // cv: chunk 2 is most aligned with the ad
        let cv = vec![chunk(&[0.0, 1.0]), chunk(&[1.0, 0.0]), chunk(&[0.6, 0.8])];
        let d = doc_features(&job, &cv).unwrap();
        // c must be the normalized [1, 0] chunk
        let nc = ((d.c[0] * d.c[0] + d.c[1] * d.c[1]) as f64).sqrt();
        assert!((nc - 1.0).abs() < 1e-5);
        assert!(d.c[0] > 0.999);
        assert_eq!(d.n_chunks, 3);
        assert_eq!(d.chunk_sims.len(), 3);
        assert!(d.chunk_sims[1] > d.chunk_sims[0]);
    }

    #[test]
    fn single_chunk_means_c_equals_e() {
        let job = vec![chunk(&[1.0, 0.0])];
        let cv = vec![chunk(&[3.0, 4.0])];
        let d = doc_features(&job, &cv).unwrap();
        assert!((d.c[0] - d.e[0]).abs() < 1e-6);
        assert!((d.c[1] - d.e[1]).abs() < 1e-6);
    }

    #[test]
    fn raw_pair_layout_matches_blocks() {
        let spec = FeatureSpec::raw_pair("m", 2, ChunkingMode::CtxSize, None, Contextual::Off, TextPrep::Lowercase);
        let job = DocFeatures { a: vec![1.0, 1.0], e: vec![0.0; 2], c: vec![0.0; 2], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let cv1 = DocFeatures { a: vec![0.0; 2], e: vec![2.0, 2.0], c: vec![3.0, 3.0], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let cv2 = DocFeatures { a: vec![0.0; 2], e: vec![4.0, 4.0], c: vec![5.0, 5.0], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let x = assemble_raw_pair(&spec, &job, &cv1, &cv2);
        assert_eq!(x.len(), 10);
        assert_eq!(&x[0..2], &[1.0, 1.0]);       // a
        assert_eq!(&x[2..4], &[2.0, 2.0]);       // e1
        assert_eq!(&x[4..6], &[4.0, 4.0]);       // e2
        assert_eq!(&x[6..8], &[3.0, 3.0]);       // c1
        assert_eq!(&x[8..10], &[5.0, 5.0]);      // c2
        for b in &spec.blocks {
            assert_eq!(b.end() <= 10, true);
        }
    }

    #[test]
    fn fused_pair_layout_matches_blocks() {
        let spec = FeatureSpec::fused_pair("m", 2, ChunkingMode::CtxSize, None, Contextual::Off, TextPrep::Lowercase);
        assert_eq!(spec.variant, FeatureVariant::Fused);
        assert_eq!(spec.input_dim, 5 * 2 + 2 * crate::evidence::X_BLOCK_LEN);
        let job = DocFeatures { a: vec![1.0, 1.0], e: vec![0.0; 2], c: vec![0.0; 2], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let x1 = vec![0.5f32; crate::evidence::X_BLOCK_LEN];
        let x2 = vec![0.25f32; crate::evidence::X_BLOCK_LEN];
        let cv1 = DocFeatures { a: vec![0.0; 2], e: vec![2.0, 2.0], c: vec![3.0, 3.0], n_chunks: 1, chunk_sims: vec![0.0], x: x1 };
        let cv2 = DocFeatures { a: vec![0.0; 2], e: vec![4.0, 4.0], c: vec![5.0, 5.0], n_chunks: 1, chunk_sims: vec![0.0], x: x2 };
        let x = assemble_pair(&spec, &job, &cv1, &cv2);
        assert_eq!(x.len(), spec.input_dim);
        let raw = assemble_raw_pair(
            &FeatureSpec::raw_pair("m", 2, ChunkingMode::CtxSize, None, Contextual::Off, TextPrep::Lowercase),
            &job, &cv1, &cv2,
        );
        assert_eq!(&x[..10], &raw[..]);
        let xw = crate::evidence::X_BLOCK_LEN;
        assert!(x[10..10 + xw].iter().all(|&v| v == 0.5));
        assert!(x[10 + xw..].iter().all(|&v| v == 0.25));
        // blocks tile the input exactly
        let mut covered = 0usize;
        for b in &spec.blocks {
            assert_eq!(b.start, covered);
            covered = b.end();
        }
        assert_eq!(covered, spec.input_dim);
    }

    #[test]
    fn engineered_pair_layout_and_stats() {
        let dim = 2;
        let spec = FeatureSpec::engineered_pair("m", dim, ChunkingMode::CtxSize, None, Contextual::Off, TextPrep::Lowercase, 0.5);
        let job = DocFeatures { a: vec![1.0, 0.0], e: vec![0.0; 2], c: vec![0.0; 2], n_chunks: 2, chunk_sims: vec![0.9, 0.2], x: vec![] };
        let cv1 = DocFeatures { a: vec![0.0; 2], e: vec![1.0, 0.0], c: vec![0.0; 2], n_chunks: 2, chunk_sims: vec![0.9, 0.2], x: vec![] };
        let cv2 = DocFeatures { a: vec![0.0; 2], e: vec![0.0, 1.0], c: vec![0.0; 2], n_chunks: 2, chunk_sims: vec![0.2, 0.9], x: vec![] };
        let x = assemble_engineered_pair(&spec, &job, &cv1, &cv2);
        assert_eq!(x.len(), 8 * dim + 8);
        // e1 block
        assert_eq!(&x[0..2], &[1.0, 0.0]);
        // e2 block
        assert_eq!(&x[2..4], &[0.0, 1.0]);
        // e1-e2
        assert_eq!(&x[4..6], &[1.0, -1.0]);
        // e1*e2
        assert_eq!(&x[6..8], &[0.0, 0.0]);
        // a
        assert_eq!(&x[8..10], &[1.0, 0.0]);
        // stats blocks at the end: [max, mean, top3, coverage] each
        let s1 = &x[x.len() - 8..x.len() - 4];
        assert!((s1[0] - 0.9).abs() < 1e-6);              // max
        assert!((s1[1] - 0.55).abs() < 1e-6);             // mean
        assert!((s1[2] - 0.55).abs() < 1e-6);             // top3 = mean of both
        assert!((s1[3] - 0.5).abs() < 1e-6);              // coverage at tau .5
        // h1 = a * e1 = [1, 0]
        assert_eq!(&x[10..12], &[1.0, 0.0]);
    }

    #[test]
    fn pointwise_layout() {
        let spec = FeatureSpec::pointwise("m", 3, ChunkingMode::None, None, Contextual::Off, TextPrep::Lowercase);
        let job = DocFeatures { a: vec![1.0, 2.0, 3.0], e: vec![0.0; 3], c: vec![0.0; 3], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let cv = DocFeatures { a: vec![0.0; 3], e: vec![7.0, 8.0, 9.0], c: vec![1.0, 1.0, 1.0], n_chunks: 1, chunk_sims: vec![0.0], x: vec![] };
        let x = assemble_pointwise(&spec, &job, &cv);
        assert_eq!(x, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 1.0, 1.0, 1.0]);
    }
}
