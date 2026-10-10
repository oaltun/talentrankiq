//! Frozen-embedder judge: artifacts, feature assembly and hand-rolled
//! inference for the pairwise/pointwise judge methods (bench design step 15,
//! `bench/bench1/design/15-pairwise-judge.md`).
//!
//! Crate layout follows the contract-lib pattern from `NOAI.md`: both
//! consumers (`judgetrainer`, which trains with burn and writes artifacts, and
//! `bench`, which scores with the methods `pointwise-judge` /
//! `pairwise-judge`) link this crate so the artifact schema, the feature
//! assembly and the forward pass cannot drift between training and inference.
//!
//! burn is deliberately NOT a dependency: training exports plain serde weight
//! arrays, and this crate reproduces the network with a hand-rolled forward
//! pass. Burn-to-inference parity is enforced by the fixture vectors embedded
//! in every artifact (see `JudgeArtifact::verify_parity`).

pub mod artifact;
pub mod datamaker_import;
pub mod evidence;
pub mod extract;
pub mod features;
pub mod forward;
pub mod tournament;

pub use artifact::{ArtifactMeta, FeatureSpec, JudgeArtifact, LayerWeights, ParityPair};
pub use datamaker_import::{load_datamaker_export, DatamakerExportConfig, DatamakerJobGroup, DatamakerDoc, JudgeRow};
pub use extract::{ChunkCache, EvidenceCache, FeatureExtractor, TextEmbedder};
pub use features::{
    assemble_engineered_pair, assemble_fused_pair, assemble_raw_pair, doc_features, DocFeatures,
    TextPrep,
};
pub use forward::JudgeModel;
pub use tournament::{bradley_terry, cycle_rate, pairwise_prob_matrix, win_rate_scores};

/// Error type for judge loading / inference / feature building.
#[derive(thiserror::Error, Debug)]
pub enum JudgeError {
    #[error("invalid artifact: {0}")]
    InvalidArtifact(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("parity fixture failed on vector {index}: burn={burn} hand-rolled={hand} (tol 1e-4)")]
    ParityFailed {
        index: usize,
        burn: f64,
        hand: f64,
    },
    #[error("{0}")]
    Other(String),
}

/// L2-normalize a vector in place; zero vectors stay zero.
pub fn l2_normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two vectors (0.0 if either is zero).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
