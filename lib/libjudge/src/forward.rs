//! Hand-rolled forward pass. This is the production inference path — burn
//! never leaves `judgetrainer`. Weights are f32 (as exported); math runs in
//! f64. Parity with burn is asserted by the artifact's fixture vectors
//! (design 15.6).

use crate::artifact::{JudgeArtifact, JudgeKind};
use crate::{sigmoid, JudgeError};
use rayon::prelude::*;

/// A loaded judge network: consecutive affine layers, ReLU between hidden
/// layers. Output is interpreted by `kind` (softmax over 3 for pointwise,
/// sigmoid over 1 logit for pairwise).
pub struct JudgeModel {
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    kind: JudgeKind,
    input_dim: usize,
}

impl JudgeModel {
    pub fn from_artifact(artifact: &JudgeArtifact) -> Result<Self, JudgeError> {
        artifact.validate()?;
        let layers = artifact
            .layers
            .iter()
            .map(|l| {
                (
                    l.w.iter()
                        .map(|row| row.iter().map(|&x| x as f64).collect())
                        .collect(),
                    l.b.iter().map(|&x| x as f64).collect(),
                )
            })
            .collect();
        Ok(Self {
            layers,
            kind: artifact.kind,
            input_dim: artifact.feature_spec.input_dim,
        })
    }

    pub fn kind(&self) -> JudgeKind {
        self.kind
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Raw scalar ranking score. Pairwise: the comparison logit (monotone in
    /// p(i>j)). Pointwise: P(Good Fit).
    pub fn score(&self, x: Vec<f32>) -> f64 {
        self.score_ref(&x)
    }

    /// All output logits (pre-softmax) for one row. Pointwise models expose
    /// the 3 class logits here; pairwise models return the single comparison
    /// logit.
    pub fn logits(&self, x: &[f32]) -> Vec<f64> {
        self.logits_row(x)
    }

    /// The per-row forward pass. Every entry point funnels through this, so
    /// batched and scalar scoring cannot drift: same f64 order, same result,
    /// bit for bit (bench design 15.13).
    fn logits_row(&self, x: &[f32]) -> Vec<f64> {
        let mut h: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let last = self.layers.len() - 1;
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let mut next = vec![0.0f64; w.len()];
            for (o, row) in w.iter().enumerate() {
                let mut acc = b[o];
                for (wi, &xi) in row.iter().zip(h.iter()) {
                    acc += wi * xi;
                }
                next[o] = acc;
            }
            if i < last {
                for v in next.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0; // ReLU
                    }
                }
            }
            h = next;
        }
        h
    }

    /// Score `rows` feature rows in one call, parallel across rows.
    ///
    /// `flat` is row-major: `rows * input_dim` f32 values (the caller's
    /// feature buffer, so no per-row `Vec` churn). The result is row-major
    /// too, `rows * layers.last().len()` logits. Results are **bitwise
    /// identical** to calling [`Self::logits`] per row — only the driver
    /// differs.
    pub fn logits_batch(&self, flat: &[f32], rows: usize) -> Vec<f64> {
        let dim = self.input_dim;
        assert_eq!(
            flat.len(),
            rows * dim,
            "logits_batch: flat must hold exactly {rows} rows of {dim}"
        );
        let out_dim = self.layers.last().map_or(0, |(w, _)| w.len());
        if rows == 0 || out_dim == 0 {
            return Vec::new();
        }
        let scored: Vec<Vec<f64>> = if rows == 1 {
            vec![self.logits_row(&flat[..dim])]
        } else {
            flat.par_chunks_exact(dim)
                .map(|row| self.logits_row(row))
                .collect()
        };
        scored.into_iter().flatten().collect()
    }

    /// Pairwise probability per row for a row-major batch — the batched twin
    /// of [`Self::pair_prob`], and the entry point the tournament uses
    /// (bench design 15.13). Bitwise identical to the scalar path.
    pub fn pair_probs_batch(&self, flat: &[f32], rows: usize) -> Vec<f64> {
        debug_assert_eq!(self.kind, JudgeKind::Pairwise);
        self.logits_batch(flat, rows)
            .into_iter()
            .map(sigmoid)
            .collect()
    }

    /// Same as [`score`](Self::score) without taking ownership.
    pub fn score_ref(&self, x: &[f32]) -> f64 {
        let h = self.logits_row(x);
        match self.kind {
            JudgeKind::Pairwise => h[0],
            JudgeKind::Pointwise => {
                // softmax over 3 classes, score = P(Good Fit) = class 2
                let max = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = h.iter().map(|&v| (v - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps[2] / sum
            }
        }
    }

    /// Probability that cv1 beats cv2 for the given ad, from the comparison
    /// logit. Pointwise models don't produce pairwise probabilities.
    pub fn pair_prob(&self, x: Vec<f32>) -> f64 {
        debug_assert_eq!(self.kind, JudgeKind::Pairwise);
        sigmoid(self.score(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{Architecture, FeatureSpec, LayerWeights, ParityPair};
    use crate::features::TextPrep;

    fn pair_spec(dim: usize) -> FeatureSpec {
        FeatureSpec::raw_pair("m", dim, crate::artifact::ChunkingMode::CtxSize, None, crate::artifact::Contextual::Off, TextPrep::Lowercase)
    }

    fn artifact(kind: JudgeKind, layers: Vec<LayerWeights>, input_dim: usize) -> JudgeArtifact {
        let mut spec = pair_spec(1);
        spec.input_dim = input_dim;
        spec.blocks = vec![crate::artifact::Block { name: "x".into(), start: 0, len: input_dim }];
        let architecture = match (kind, layers.len()) {
            (JudgeKind::Pairwise, 1) => Architecture::Linear,
            (JudgeKind::Pairwise, 2) => Architecture::Mlp1,
            (JudgeKind::Pairwise, _) => Architecture::Mlp2,
            (JudgeKind::Pointwise, _) => Architecture::PointwiseLinear3,
        };
        JudgeArtifact {
            spec_version: crate::artifact::SPEC_VERSION,
            kind,
            architecture,
            feature_spec: spec,
            layers,
            meta: crate::artifact::ArtifactMeta {
                trained_at: String::new(),
                git_rev: String::new(),
                git_branch: String::new(),
                seed: 42,
                shuffle_seed: None,
                epochs_requested: 0,
                epochs_trained: 0,
                best_epoch: 0,
                early_stopped: false,
                lr: 0.0,
                weight_decay: 0.0,
                batch_size: 1,
                train_pairs: 0,
                val_pairs: 0,
                jobs_train: 0,
                jobs_val: 0,
                val_pairwise_accuracy: 0.0,
                val_cycle_rate: None,
                val_class_accuracy: None,
                dropout: 0.1,
                block_mass: vec![],
                zero_out: None,
                extraction_cache_hits: 0,
                extraction_cache_misses: 0,
                notes: String::new(),
            },
            parity: vec![],
        }
    }

    #[test]
    fn linear_logit_hand_computed() {
        // score = 1*x0 + (-2)*x1 + 0.5
        let layers = vec![LayerWeights { w: vec![vec![1.0, -2.0]], b: vec![0.5] }];
        let a = artifact(JudgeKind::Pairwise, layers, 2);
        let m = JudgeModel::from_artifact(&a).unwrap();
        let score = m.score_ref(&[2.0, 3.0]);
        assert!((score - (2.0 - 6.0 + 0.5)).abs() < 1e-9);
        let p = m.pair_prob(vec![2.0, 3.0]);
        assert!((p - sigmoid(-3.5)).abs() < 1e-9);
    }

    #[test]
    fn mlp_relu_hidden_layer() {
        // h = relu(x . w1 + b1); out = h . w2 + b2
        let l1 = LayerWeights { w: vec![vec![1.0, -1.0], vec![1.0, 1.0]], b: vec![0.0, 0.0] }; // 2->2
        let l2 = LayerWeights { w: vec![vec![2.0, 3.0]], b: vec![-1.0] }; // 2->1
        let a = artifact(JudgeKind::Pairwise, vec![l1, l2], 2);
        let m = JudgeModel::from_artifact(&a).unwrap();
        // x = [1, 2]: h = [max(0, 1-2), max(0, 1+2)] = [0, 3]; out = 0*2 + 3*3 - 1 = 8
        assert!((m.score_ref(&[1.0, 2.0]) - 8.0).abs() < 1e-9);
        // x = [3, 1]: h = [2, 4]; out = 4 + 12 - 1 = 15
        assert!((m.score_ref(&[3.0, 1.0]) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn pointwise_softmax_picks_good_fit_class() {
        // 3 classes; make class 2 the largest
        let layers = vec![LayerWeights {
            w: vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]],
            b: vec![0.0, 0.0, 2.0],
        }];
        let a = artifact(JudgeKind::Pointwise, layers, 2);
        let m = JudgeModel::from_artifact(&a).unwrap();
        let score = m.score_ref(&[1.0, 1.0]);
        // logits = [0, 0, 4] -> p2 = e4 / (2 + e4)
        let e4 = (4.0f64).exp();
        assert!((score - e4 / (2.0 + e4)).abs() < 1e-9);
        assert!(score > 0.9);
    }

    #[test]
    fn artifact_roundtrip_parity_file() {
        let layers = vec![LayerWeights { w: vec![vec![1.0, -2.0]], b: vec![0.5] }];
        let mut a = artifact(JudgeKind::Pairwise, layers, 2);
        let m = JudgeModel::from_artifact(&a).unwrap();
        a.parity = vec![ParityPair { input: vec![2.0, 3.0], score: m.score_ref(&[2.0, 3.0]) as f32 }];
        let json = a.to_json().unwrap();
        let loaded = JudgeArtifact::load_from_str(&json).unwrap();
        assert!(loaded.verify_parity().is_ok());
    }

    /// Batched scoring must be **bitwise** the scalar path (design 15.13):
    /// results are cached and compared across runs, so a rounding-level
    /// difference would invalidate every stored verdict.
    #[test]
    fn batched_scoring_is_bitwise_identical_to_scalar() {
        // Two layers so ReLU and a hidden matmul are in the path; a wide
        // input so the row driver is actually chunking.
        let dim = 64;
        let hidden = 48;
        let layers = vec![
            LayerWeights {
                w: (0..hidden)
                    .map(|o| (0..dim).map(|i| ((o * 31 + i * 7) % 23) as f32 * 0.01 - 0.1).collect())
                    .collect(),
                b: (0..hidden).map(|o| (o as f32) * 0.001).collect(),
            },
            LayerWeights {
                w: vec![(0..hidden).map(|i| ((i * 13) % 17) as f32 * 0.05 - 0.2).collect()],
                b: vec![0.25],
            },
        ];
        let a = artifact(JudgeKind::Pairwise, layers, dim);
        let m = JudgeModel::from_artifact(&a).unwrap();

        let rows: Vec<Vec<f32>> = (0..257)
            .map(|r| (0..dim).map(|i| ((r * 17 + i * 5) % 29) as f32 * 0.02 - 0.3).collect())
            .collect();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();

        let batched = m.pair_probs_batch(&flat, rows.len());
        assert_eq!(batched.len(), rows.len());
        for (r, row) in rows.iter().enumerate() {
            let scalar = m.pair_prob(row.clone());
            assert_eq!(
                batched[r].to_bits(),
                scalar.to_bits(),
                "row {r}: batched {batched:?} != scalar {scalar}"
            );
        }
        // Same for the raw logits (multi-output kinds), and the single-row
        // path that skips the parallel driver.
        let single = m.logits_batch(&flat[..dim], 1);
        assert_eq!(single[0].to_bits(), m.logits(&rows[0])[0].to_bits());
        let pair_batch = m.logits_batch(&flat, rows.len());
        for (r, row) in rows.iter().enumerate() {
            assert_eq!(pair_batch[r].to_bits(), m.logits(row)[0].to_bits());
        }
    }
}
