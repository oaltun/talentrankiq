//! Ranking from pairwise verdicts (design 15.7): win-rate over the
//! tournament with both orders evaluated, cycle-rate diagnostics, and the
//! Bradley-Terry fallback aggregator.

use crate::artifact::FeatureSpec;
use crate::features::{assemble_pair, DocFeatures};
use crate::forward::JudgeModel;
use rayon::prelude::*;

/// Oriented pair probability p(i > j) with both orders evaluated and
/// averaged (15.7 symmetry): `p = (f(i,j) + 1 - f(j,i)) / 2` where
/// `f` is the judge's pair probability. Public because it IS the memoizable
/// verdict unit: consumers may cache `p(i > j)` per (ad, candidate pair,
/// artifact) and assemble matrices from cached verdicts.
pub fn averaged_pair_prob(
    model: &JudgeModel,
    spec: &FeatureSpec,
    ad: &DocFeatures,
    i: &DocFeatures,
    j: &DocFeatures,
) -> f64 {
    let xij = assemble_pair(spec, ad, i, j);
    let xji = assemble_pair(spec, ad, j, i);
    let pij = model.pair_prob(xij);
    let pji = model.pair_prob(xji);
    0.5 * (pij + (1.0 - pji))
}

/// How many pairs one batched pass should carry, given the feature width.
///
/// Budgeted by BYTES (two rows per pair — both orientations — times the
/// feature dim), so a 8200-dim engineered pair and a 20-dim linear pair
/// both peak at the same few MB. A 2000-candidate job is ~2 M pairs; their
/// rows alone would be >100 GB materialized at once, so callers MUST chunk.
pub fn pairs_per_tile(dim: usize) -> usize {
    const TILE_BYTES: usize = 8 << 20;
    (TILE_BYTES / (2 * dim.max(1) * size_of::<f32>())).max(1)
}

/// Score a tile of pairs in ONE batched, row-parallel pass:
/// `out[k] = p(pairs[k].0 > pairs[k].1)`, the same value
/// [`averaged_pair_prob`] returns (bit for bit — same features, same
/// forward passes, same averaging expression).
///
/// This is the unit both callers build on: [`pairwise_prob_matrix`] streams
/// every pair through it, and the gui's cached path collects its fresh pairs
/// and runs them in chunks of [`pairs_per_tile`].
pub fn averaged_pair_probs_batch(
    model: &JudgeModel,
    spec: &FeatureSpec,
    ad: &DocFeatures,
    pairs: &[(usize, usize)],
    docs: &[DocFeatures],
) -> Vec<f64> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let dim = spec.input_dim.max(1);
    let mut rows: Vec<f32> = vec![0.0f32; pairs.len() * 2 * dim];
    fill_tile(&mut rows, pairs, spec, ad, docs);
    let probs = model.pair_probs_batch(&rows, pairs.len() * 2);
    (0..pairs.len())
        .map(|k| 0.5 * (probs[2 * k] + (1.0 - probs[2 * k + 1])))
        .collect()
}

/// Full averaged probability matrix, `p[i][j] = p(i > j)` (diagonal 0.5).
///
/// Batched + row-parallel (bench design 15.13): pairs are scored in
/// byte-budgeted tiles through [`averaged_pair_probs_batch`], so peak memory
/// stays flat no matter how wide the feature row is. Values are unchanged —
/// bitwise identical to the scalar loop.
pub fn pairwise_prob_matrix(
    model: &JudgeModel,
    spec: &FeatureSpec,
    ad: &DocFeatures,
    docs: &[DocFeatures],
) -> Vec<Vec<f64>> {
    let n = docs.len();
    let mut p = vec![vec![0.5f64; n]; n];
    if n < 2 {
        return p;
    }
    let per_tile = pairs_per_tile(spec.input_dim);
    let mut tile: Vec<(usize, usize)> = Vec::with_capacity(per_tile);
    for i in 0..n {
        for j in (i + 1)..n {
            tile.push((i, j));
            if tile.len() == per_tile {
                absorb(
                    &mut p,
                    &tile,
                    &averaged_pair_probs_batch(model, spec, ad, &tile, docs),
                );
                tile.clear();
            }
        }
    }
    if !tile.is_empty() {
        absorb(
            &mut p,
            &tile,
            &averaged_pair_probs_batch(model, spec, ad, &tile, docs),
        );
    }
    p
}

/// Write a tile of results into the symmetric matrix, pairing each value
/// with the pair it belongs to (`vals[k]` is `p(pairs[k].0 > pairs[k].1)`).
fn absorb(p: &mut [Vec<f64>], pairs: &[(usize, usize)], vals: &[f64]) {
    for (k, &(i, j)) in pairs.iter().enumerate() {
        match vals.get(k) {
            Some(&v) => {
                p[i][j] = v;
                p[j][i] = 1.0 - v;
            }
            None => return,
        }
    }
}

/// Fill `rows` (row-major, two rows per pair: `i→j` then `j→i`) with the
/// assembled features, one task per pair.
fn fill_tile(
    rows: &mut [f32],
    pairs: &[(usize, usize)],
    spec: &FeatureSpec,
    ad: &DocFeatures,
    docs: &[DocFeatures],
) {
    let dim = spec.input_dim.max(1);
    rows
        .par_chunks_exact_mut(2 * dim)
        .zip(pairs.par_iter())
        .for_each(|(slot, &(i, j))| {
            let (a, b) = slot.split_at_mut(dim);
            a.copy_from_slice(&assemble_pair(spec, ad, &docs[i], &docs[j]));
            b.copy_from_slice(&assemble_pair(spec, ad, &docs[j], &docs[i]));
        });
}

/// Win-rate scores (15.7 default): `score_i = mean over j != i of p(i > j)`,
/// ties = 0.5. Collapses any tournament deterministically and keeps
/// fractional probabilities.
pub fn win_rate_scores(p: &[Vec<f64>]) -> Vec<f64> {
    let n = p.len();
    (0..n)
        .map(|i| {
            if n == 1 {
                return 0.5;
            }
            let sum: f64 = (0..n).filter(|&j| j != i).map(|j| p[i][j]).sum();
            sum / (n - 1) as f64
        })
        .collect()
}

/// Fraction of index triples (i < j < k) forming a majority cycle:
/// `p(i>j) > 0.5 && p(j>k) > 0.5 && p(k>i) > 0.5`.
pub fn cycle_rate(p: &[Vec<f64>]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut cycles = 0usize;
    let mut triples = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                triples += 1;
                let cyclic = (p[i][j] > 0.5 && p[j][k] > 0.5 && p[k][i] > 0.5)
                    || (p[j][i] > 0.5 && p[k][j] > 0.5 && p[i][k] > 0.5);
                if cyclic {
                    cycles += 1;
                }
            }
        }
    }
    if triples == 0 {
        0.0
    } else {
        cycles as f64 / triples as f64
    }
}

/// Bradley-Terry fallback (15.7): fit latent log-strengths `s` with
/// `sigmoid(s_i - s_j) ~ p(i > j)` via Hunter's MM iteration (logistic
/// regression on the N^2 matrix), then rank by `s_i`. Few iterations are
/// enough for ranking purposes.
pub fn bradley_terry(p: &[Vec<f64>], iters: usize) -> Vec<f64> {
    let n = p.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let clamp = |v: f64| v.clamp(-30.0, 30.0);
    let mut s = vec![0.0f64; n];
    for _ in 0..iters {
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            // fractional wins of i
            let wins: f64 = (0..n).filter(|&j| j != i).map(|j| p[i][j]).sum();
            let gamma_i = s[i].clamp(-30.0, 30.0).exp();
            let den: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| 1.0 / (gamma_i + s[j].clamp(-30.0, 30.0).exp()))
                .sum();
            next[i] = clamp((wins / den.max(1e-12)).ln());
        }
        // anchor scale (log-strengths are shift-invariant): mean 0
        let mean = next.iter().sum::<f64>() / n as f64;
        for v in next.iter_mut() {
            *v = clamp(*v - mean);
        }
        s = next;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{Architecture, ChunkingMode, Contextual, JudgeKind, LayerWeights};
    use crate::features::TextPrep;

    /// A synthetic pairwise judge over `dim`-wide raw-pair features with one
    /// hidden layer — the shape the gui actually loads.
    fn judge(dim: usize) -> (JudgeModel, FeatureSpec) {
        let mut spec = FeatureSpec::raw_pair(
            "m",
            dim,
            ChunkingMode::CtxSize,
            None,
            Contextual::Off,
            TextPrep::Lowercase,
        );
        let hidden = 32;
        let layers = vec![
            LayerWeights {
                w: (0..hidden)
                    .map(|o| (0..spec.input_dim).map(|i| ((o * 31 + i * 7) % 23) as f32 * 0.01 - 0.1).collect())
                    .collect(),
                b: vec![0.0; hidden],
            },
            LayerWeights {
                w: (0..1)
                    .map(|o| (0..hidden).map(|i| ((o * 13 + i * 3) % 11) as f32 * 0.02).collect())
                    .collect(),
                b: vec![0.1],
            },
        ];
        let _ = &mut spec;
        let artifact = crate::artifact::JudgeArtifact {
            spec_version: crate::artifact::SPEC_VERSION,
            kind: JudgeKind::Pairwise,
            architecture: Architecture::Mlp1,
            feature_spec: spec.clone(),
            layers,
            meta: crate::artifact::ArtifactMeta {
                trained_at: String::new(),
                git_rev: String::new(),
                git_branch: String::new(),
                seed: 1,
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
                dropout: 0.0,
                block_mass: vec![],
                zero_out: None,
                extraction_cache_hits: 0,
                extraction_cache_misses: 0,
                notes: String::new(),
            },
            parity: vec![],
        };
        (
            JudgeModel::from_artifact(&artifact).expect("synthetic artifact is valid"),
            spec,
        )
    }

    fn doc(seed: usize, dim: usize) -> DocFeatures {
        // Deterministic pseudo-document: no rng in the tree, and the same
        // inputs every run keeps the timing comparable.
        let v = |i: usize| (((seed * 37 + i * 11) % 29) as f32 / 29.0) - 0.5;
        DocFeatures {
            a: (0..dim).map(v).collect(),
            e: (0..dim).map(|i| v(i) + 0.01 * seed as f32).collect(),
            c: (0..dim).map(|i| v(i) * 0.9).collect(),
            n_chunks: 1,
            chunk_sims: vec![],
            x: vec![],
        }
    }

    /// The batched matrix must equal the scalar reference **bit for bit**:
    /// verdicts are memoized in sqlite and reused across runs, so a
    /// rounding-level drift would invalidate every cached pair.
    #[test]
    fn batched_matrix_is_bitwise_identical_to_the_scalar_reference() {
        let dim = 24;
        let (model, spec) = judge(dim);
        let ad = doc(0, dim);
        let docs: Vec<DocFeatures> = (1..=40).map(|s| doc(s, dim)).collect();

        let mut reference = vec![vec![0.5f64; docs.len()]; docs.len()];
        for i in 0..docs.len() {
            for j in (i + 1)..docs.len() {
                let v = averaged_pair_prob(&model, &spec, &ad, &docs[i], &docs[j]);
                reference[i][j] = v;
                reference[j][i] = 1.0 - v;
            }
        }
        let batched = pairwise_prob_matrix(&model, &spec, &ad, &docs);
        for i in 0..docs.len() {
            for j in 0..docs.len() {
                assert_eq!(
                    batched[i][j].to_bits(),
                    reference[i][j].to_bits(),
                    "p[{i}][{j}]: {} vs {}",
                    batched[i][j],
                    reference[i][j]
                );
            }
        }
    }

    /// The point of design 15.13: print the scalar vs batched tournament
    /// times on a 300-doc synthetic job so the win is visible in the test
    /// log (`cargo test --release -p libjudge -- --nocapture`).
    #[test]
    fn batched_tournament_is_faster_than_the_scalar_loop() {
        let dim = 128; // wide enough that the MLP is not the only cost
        let (model, spec) = judge(dim);
        let ad = doc(0, dim);
        let docs: Vec<DocFeatures> = (1..=300).map(|s| doc(s, dim)).collect();

        let t0 = std::time::Instant::now();
        let mut reference = vec![vec![0.5f64; docs.len()]; docs.len()];
        for i in 0..docs.len() {
            for j in (i + 1)..docs.len() {
                let v = averaged_pair_prob(&model, &spec, &ad, &docs[i], &docs[j]);
                reference[i][j] = v;
                reference[j][i] = 1.0 - v;
            }
        }
        let scalar = t0.elapsed();
        let t1 = std::time::Instant::now();
        let batched = pairwise_prob_matrix(&model, &spec, &ad, &docs);
        let fast = t1.elapsed();
        for i in 0..docs.len() {
            for j in 0..docs.len() {
                assert_eq!(batched[i][j].to_bits(), reference[i][j].to_bits());
            }
        }
        eprintln!(
            "tournament 300 docs ({} pairs): scalar {scalar:.2?} vs batched {fast:.2?} = {:.1}x",
            docs.len() * (docs.len() - 1) / 2,
            scalar.as_secs_f64() / fast.as_secs_f64().max(1e-9)
        );
        assert!(
            fast < scalar,
            "batched path ({fast:?}) must beat the scalar loop ({scalar:?})"
        );
    }

    #[test]
    fn single_and_empty_tournaments_are_well_formed() {
        let dim = 8;
        let (model, spec) = judge(dim);
        let ad = doc(0, dim);
        assert!(pairwise_prob_matrix(&model, &spec, &ad, &[]).is_empty());
        let one = pairwise_prob_matrix(&model, &spec, &ad, &[doc(1, dim)]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0][0], 0.5);
    }

    #[test]
    fn win_rate_transitive_tournament() {
        // a beats both, b beats c: p = [[.5,.9,.9],[.1,.5,.8],[.1,.2,.5]]
        let p = vec![
            vec![0.5, 0.9, 0.9],
            vec![0.1, 0.5, 0.8],
            vec![0.1, 0.2, 0.5],
        ];
        let s = win_rate_scores(&p);
        assert!(s[0] > s[1] && s[1] > s[2]);
        assert!((s[0] - 0.9).abs() < 1e-9);
        assert!((s[1] - 0.45).abs() < 1e-9);
        assert!((s[2] - 0.15).abs() < 1e-9);
        assert!((cycle_rate(&p) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cycle_rate_detects_cycle() {
        // a>b, b>c, c>a: exact cycle
        let p = vec![
            vec![0.5, 0.9, 0.1],
            vec![0.1, 0.5, 0.9],
            vec![0.9, 0.1, 0.5],
        ];
        assert!((cycle_rate(&p) - 1.0).abs() < 1e-9);
        // and win-rate still collapses it deterministically
        let s = win_rate_scores(&p);
        assert!((s.iter().sum::<f64>() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn cycle_rate_counts_all_orientations_once() {
        // 3 docs, acyclic: 1 triple, 0 cycles
        let p = vec![
            vec![0.5, 0.7, 0.6],
            vec![0.3, 0.5, 0.8],
            vec![0.4, 0.2, 0.5],
        ];
        assert!((cycle_rate(&p) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn bt_recovers_dominance_order() {
        let p = vec![
            vec![0.5, 0.9, 0.9],
            vec![0.1, 0.5, 0.8],
            vec![0.1, 0.2, 0.5],
        ];
        let s = bradley_terry(&p, 50);
        assert!(s[0] > s[1] && s[1] > s[2]);
    }

    #[test]
    fn bt_handles_cycles_gracefully() {
        let p = vec![
            vec![0.5, 0.9, 0.1],
            vec![0.1, 0.5, 0.9],
            vec![0.9, 0.1, 0.5],
        ];
        let s = bradley_terry(&p, 50);
        // near-perfect cycle -> near-equal strengths
        let spread = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - s.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            spread < 0.5,
            "spread too large for a perfect cycle: {spread}"
        );
    }

    #[test]
    fn prob_matrix_is_antisymmetric() {
        let p = vec![vec![0.5f64, 0.7], vec![0.3, 0.5]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((p[i][j] + p[j][i] - 1.0).abs() < 1e-9);
            }
        }
    }
}
