//! Artifact schema: the JSON file a trained judge is shipped as
//! (`judge_<model>.json`, design step 15.6). Plain serde types, no burn types.

use serde::{Deserialize, Serialize};

use crate::features::TextPrep;
use crate::JudgeError;

/// Artifact schema version. Bump on any breaking change to this file's shape
/// or to feature semantics.
pub const SPEC_VERSION: u32 = 1;

/// Historic default: artifacts before the `dropout` meta field existed were
/// all trained with the hard-coded 0.1.
fn default_dropout() -> f64 {
    0.1
}

/// Which ranking scheme the artifact's score feeds (15.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeKind {
    /// m0: one forward per (ad, cv), ranked by P(Good Fit). No tournament.
    Pointwise,
    /// m1-m3: pairwise verdicts aggregated by win-rate (15.7).
    Pairwise,
}

/// Human-readable architecture label (informational; the layer dims in
/// `JudgeArtifact.layers` define the actual computation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    /// m0: 3-class softmax head on [a; e; c].
    PointwiseLinear3,
    /// m1: logistic regression (linear judge).
    Linear,
    /// m2: MLP with one hidden layer.
    Mlp1,
    /// m3: MLP with two hidden layers.
    Mlp2,
}

/// Feature variant (15.4). `raw` is the default; `engineered` is the
/// ablation-only variant that also hosts the linear rung (m1); `fused`
/// adds the structured-evidence block (design 18 §18.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureVariant {
    Raw,
    Engineered,
    Fused,
}

/// Chunking mode, mirroring bench step 14.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkingMode {
    None,
    CtxSize,
    Fixed,
}

/// Contextual prefix mode, mirroring bench step 14.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Contextual {
    Off,
    Static,
}

/// One named contiguous slice of the judge input vector. Recorded so
/// introspection (per-block weight mass, zero-out ablation) and config
/// validation can be data-driven instead of hard-coded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub name: String,
    pub start: usize,
    pub len: usize,
}

impl Block {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// Full description of how judge inputs are built from raw text. Travels
/// inside the artifact; bench validates its `--config embedder=...` (if
/// given) against `embedder` here, and rebuilds features from the rest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub spec_version: u32,
    pub variant: FeatureVariant,
    pub embedder: String,
    pub dim: usize,
    pub chunking: ChunkingMode,
    /// Only meaningful when `chunking` is `fixed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    pub contextual: Contextual,
    pub text_preprocessing: TextPrep,
    /// Coverage threshold tau for the engineered stats block (15.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau: Option<f64>,
    pub blocks: Vec<Block>,
    pub input_dim: usize,
}

/// Per-layer weights, `w[out][in]`, straight from the trained burn model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerWeights {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

/// (input, score) pair recorded straight from the burn model at save time.
/// `libjudge` tests and any consumer can assert the hand-rolled forward
/// reproduces them within 1e-4 (design 15.6: parity enforced by test).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParityPair {
    pub input: Vec<f32>,
    pub score: f32,
}

/// Per-block first-layer diagnostics (15.6 introspection).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockMass {
    pub name: String,
    /// L2 norm of the first layer's incoming weights for this block.
    pub mass: f64,
}

/// Zero-out ablation result (15.6): the lowest-mass block was zeroed on val
/// inputs and val pairwise accuracy re-measured.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroOutAblation {
    pub block: String,
    pub val_pairwise_accuracy: f64,
}

/// Training / evaluation bookkeeping stored alongside the weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMeta {
    pub trained_at: String,
    pub git_rev: String,
    /// Branch the run started from (empty for artifacts pre-dating the field).
    #[serde(default)]
    pub git_branch: String,
    pub seed: u64,
    /// Resolved batch-order shuffle seed (None for artifacts pre-dating the
    /// field; unrecorded fresh entropy is the default there).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_seed: Option<u64>,
    pub epochs_requested: u32,
    pub epochs_trained: u32,
    pub best_epoch: u32,
    pub early_stopped: bool,
    pub lr: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub train_pairs: usize,
    pub val_pairs: usize,
    pub jobs_train: usize,
    pub jobs_val: usize,
    /// Headline comparable metric (15.6), higher is better.
    pub val_pairwise_accuracy: f64,
    /// Cycle rate of the val tournament (15.7); pairwise models only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val_cycle_rate: Option<f64>,
    /// m0 diagnostic: 3-class accuracy on val docs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val_class_accuracy: Option<f64>,
    /// Dropout probability used at train time (0.0 = off; historic
    /// artifacts pre-date the field and were trained at the hard-coded 0.1).
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    pub block_mass: Vec<BlockMass>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_out: Option<ZeroOutAblation>,
    /// Embedding call stats during feature extraction.
    pub extraction_cache_hits: usize,
    pub extraction_cache_misses: usize,
    #[serde(default)]
    pub notes: String,
}

impl FeatureSpec {
    /// Raw pair-input spec `[a; e1; e2; c1; c2]` (default, 15.4).
    pub fn raw_pair(embedder: &str, dim: usize, chunking: ChunkingMode, chunk_size: Option<usize>, contextual: Contextual, prep: crate::features::TextPrep) -> Self {
        Self {
            spec_version: SPEC_VERSION,
            variant: FeatureVariant::Raw,
            embedder: embedder.to_string(),
            dim,
            chunking,
            chunk_size: chunk_size.filter(|_| chunking == ChunkingMode::Fixed),
            contextual,
            text_preprocessing: prep,
            tau: None,
            blocks: crate::features::raw_blocks(dim),
            input_dim: 5 * dim,
        }
    }

    /// Engineered pair-input spec (ablation, 15.4).
    pub fn engineered_pair(embedder: &str, dim: usize, chunking: ChunkingMode, chunk_size: Option<usize>, contextual: Contextual, prep: crate::features::TextPrep, tau: f64) -> Self {
        Self {
            spec_version: SPEC_VERSION,
            variant: FeatureVariant::Engineered,
            embedder: embedder.to_string(),
            dim,
            chunking,
            chunk_size: chunk_size.filter(|_| chunking == ChunkingMode::Fixed),
            contextual,
            text_preprocessing: prep,
            tau: Some(tau),
            blocks: crate::features::engineered_blocks(dim),
            input_dim: 8 * dim + 8,
        }
    }

    /// Fused pair-input spec (design 18 §18.5): raw embedding blocks
    /// `[a; e1; e2; c1; c2]` plus one evidence scalar block per cv.
    pub fn fused_pair(embedder: &str, dim: usize, chunking: ChunkingMode, chunk_size: Option<usize>, contextual: Contextual, prep: crate::features::TextPrep) -> Self {
        Self {
            spec_version: SPEC_VERSION,
            variant: FeatureVariant::Fused,
            embedder: embedder.to_string(),
            dim,
            chunking,
            chunk_size: chunk_size.filter(|_| chunking == ChunkingMode::Fixed),
            contextual,
            text_preprocessing: prep,
            tau: None,
            blocks: crate::features::fused_blocks(dim),
            input_dim: 5 * dim + 2 * crate::evidence::X_BLOCK_LEN,
        }
    }

    /// Pointwise spec `[a; e; c]` (m0).
    pub fn pointwise(embedder: &str, dim: usize, chunking: ChunkingMode, chunk_size: Option<usize>, contextual: Contextual, prep: crate::features::TextPrep) -> Self {
        Self {
            spec_version: SPEC_VERSION,
            variant: FeatureVariant::Raw,
            embedder: embedder.to_string(),
            dim,
            chunking,
            chunk_size: chunk_size.filter(|_| chunking == ChunkingMode::Fixed),
            contextual,
            text_preprocessing: prep,
            tau: None,
            blocks: crate::features::pointwise_blocks(dim),
            input_dim: 3 * dim,
        }
    }
}

/// The full judge artifact: feature spec + weights + parity fixtures + meta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JudgeArtifact {
    pub spec_version: u32,
    pub kind: JudgeKind,
    pub architecture: Architecture,
    pub feature_spec: FeatureSpec,
    pub layers: Vec<LayerWeights>,
    pub meta: ArtifactMeta,
    pub parity: Vec<ParityPair>,
}

impl JudgeArtifact {
    /// Validate shape consistency (dims chain, blocks cover the input).
    pub fn validate(&self) -> Result<(), JudgeError> {
        if self.spec_version != SPEC_VERSION {
            return Err(JudgeError::InvalidArtifact(format!(
                "spec_version {} != supported {}",
                self.spec_version, SPEC_VERSION
            )));
        }
        if self.layers.is_empty() {
            return Err(JudgeError::InvalidArtifact("no layers".into()));
        }
        let mut expected_in = self.feature_spec.input_dim;
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.b.len() != layer.w.len() {
                return Err(JudgeError::InvalidArtifact(format!(
                    "layer {i}: {} biases for {} rows",
                    layer.b.len(),
                    layer.w.len()
                )));
            }
            for (r, row) in layer.w.iter().enumerate() {
                if row.len() != expected_in {
                    return Err(JudgeError::InvalidArtifact(format!(
                        "layer {i} row {r}: {} inputs, expected {expected_in}",
                        row.len()
                    )));
                }
            }
            expected_in = layer.w.len();
        }
        for block in &self.feature_spec.blocks {
            if block.end() > self.feature_spec.input_dim {
                return Err(JudgeError::InvalidArtifact(format!(
                    "block {} [{}, {}) exceeds input_dim {}",
                    block.name,
                    block.start,
                    block.end(),
                    self.feature_spec.input_dim
                )));
            }
        }
        let last_out = self.layers.last().unwrap().w.len();
        let want_out = match self.kind {
            JudgeKind::Pointwise => 3,
            JudgeKind::Pairwise => 1,
        };
        if last_out != want_out {
            return Err(JudgeError::InvalidArtifact(format!(
                "kind {kind:?} needs {want_out} output units, artifact has {last_out}",
                kind = self.kind
            )));
        }
        Ok(())
    }

    /// Assert the hand-rolled forward pass reproduces the recorded burn
    /// outputs within 1e-4 (design 15.6).
    pub fn verify_parity(&self) -> Result<(), JudgeError> {
        let model = crate::forward::JudgeModel::from_artifact(self)?;
        for (i, pair) in self.parity.iter().enumerate() {
            let score = model.score(pair.input.clone());
            if (score - pair.score as f64).abs() > 1e-4 {
                return Err(JudgeError::ParityFailed {
                    index: i,
                    burn: pair.score as f64,
                    hand: score,
                });
            }
        }
        Ok(())
    }

    /// Load + validate + parity-check from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, JudgeError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| JudgeError::Other(format!("cannot read {}: {e}", path.display())))?;
        Self::load_from_str(&data)
    }

    /// Load + validate + parity-check from a JSON string.
    pub fn load_from_str(data: &str) -> Result<Self, JudgeError> {
        let artifact: JudgeArtifact = serde_json::from_str(data)
            .map_err(|e| JudgeError::InvalidArtifact(format!("JSON parse: {e}")))?;
        artifact.validate()?;
        artifact.verify_parity()?;
        Ok(artifact)
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> Result<String, JudgeError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| JudgeError::Other(format!("serialize: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: JudgeKind) -> FeatureSpec {
        match kind {
            JudgeKind::Pairwise => FeatureSpec::raw_pair("m", 4, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase),
            JudgeKind::Pointwise => FeatureSpec::pointwise("m", 4, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase),
        }
    }

    fn linear_artifact(kind: JudgeKind) -> JudgeArtifact {
        let s = spec(kind);
        let out = match kind {
            JudgeKind::Pairwise => 1,
            JudgeKind::Pointwise => 3,
        };
        let w = (0..out).map(|o| (0..s.input_dim).map(|i| ((o + i) % 7) as f32 - 3.0).collect()).collect();
        let b = vec![0.5f32; out];
        JudgeArtifact {
            spec_version: SPEC_VERSION,
            kind,
            architecture: if kind == JudgeKind::Pairwise { Architecture::Linear } else { Architecture::PointwiseLinear3 },
            feature_spec: s,
            layers: vec![LayerWeights { w, b }],
            meta: meta(),
            parity: vec![],
        }
    }

    fn meta() -> ArtifactMeta {
        ArtifactMeta {
            trained_at: "2026-09-03T00:00:00Z".into(),
            git_rev: "test".into(),
            git_branch: "test".into(),
            seed: 42,
            shuffle_seed: Some(7),
            epochs_requested: 50,
            epochs_trained: 7,
            best_epoch: 5,
            early_stopped: true,
            lr: 1e-3,
            weight_decay: 1e-4,
            batch_size: 64,
            train_pairs: 100,
            val_pairs: 20,
            jobs_train: 8,
            jobs_val: 2,
            val_pairwise_accuracy: 0.8,
            val_cycle_rate: None,
            val_class_accuracy: None,
            dropout: 0.1,
            block_mass: vec![],
            zero_out: None,
            extraction_cache_hits: 0,
            extraction_cache_misses: 0,
            notes: String::new(),
        }
    }

    #[test]
    fn roundtrip_and_validate_ok() {
        let a = linear_artifact(JudgeKind::Pairwise);
        let json = a.to_json().unwrap();
        let b: JudgeArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(b.kind, JudgeKind::Pairwise);
        assert_eq!(b.feature_spec.input_dim, 20);
        b.validate().unwrap();
    }

    #[test]
    fn validate_rejects_wrong_out_dim() {
        let mut a = linear_artifact(JudgeKind::Pairwise);
        a.layers[0].w.insert(0, vec![0.0; a.feature_spec.input_dim]);
        a.layers[0].b.insert(0, 0.0);
        assert!(a.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_block() {
        let mut a = linear_artifact(JudgeKind::Pairwise);
        a.feature_spec.blocks[0].start = 100;
        assert!(a.validate().is_err());
    }

    #[test]
    fn parity_check_detects_drift() {
        let mut a = linear_artifact(JudgeKind::Pairwise);
        let model = crate::forward::JudgeModel::from_artifact(&a).unwrap();
        let x = vec![0.25f32; a.feature_spec.input_dim];
        let score = model.score_ref(&x) as f32;
        a.parity = vec![ParityPair { input: x.clone(), score }];
        a.verify_parity().unwrap();

        let mut bad = a.clone();
        bad.parity[0].score += 0.01;
        assert!(matches!(
            bad.verify_parity(),
            Err(JudgeError::ParityFailed { .. })
        ));
    }

    #[test]
    fn spec_constructors_dims() {
        let raw = FeatureSpec::raw_pair("m", 1024, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase);
        assert_eq!(raw.input_dim, 5120);
        assert_eq!(raw.blocks.len(), 5);
        let eng = FeatureSpec::engineered_pair("m", 1024, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase, 0.5);
        assert_eq!(eng.input_dim, 8 * 1024 + 8);
        let pw = FeatureSpec::pointwise("m", 1024, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase);
        assert_eq!(pw.input_dim, 3072);
        let fused = FeatureSpec::fused_pair("m", 1024, ChunkingMode::CtxSize, None, Contextual::Off, crate::features::TextPrep::Lowercase);
        assert_eq!(fused.input_dim, 5120 + 2 * crate::evidence::X_BLOCK_LEN);
        assert_eq!(fused.variant, FeatureVariant::Fused);
        // serde roundtrip keeps the new variant
        let json = serde_json::to_string(&fused).unwrap();
        let back: FeatureSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fused);
    }
}
