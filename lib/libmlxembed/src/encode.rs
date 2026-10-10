//! The loaded-embedder handle the engine talks to.
//!
//! Mirrors what `onnxwrapperrs::SessionManager` offers an embedder — load
//! once, embed many — so the backend ladder can hold either arm behind one
//! interface (design 15 §15.4). The engine tokenizes: ids and the padding mask
//! come in from `libtokenizer`, so both runtimes see the same tokens and a
//! parity test compares models rather than tokenizers.

use std::path::Path;

use mlxcel_core::weights::load_weights_from_dir_with_subfolders;

use crate::checkpoint::{self, Checkpoint, Family, Mask, Pooling};
use crate::pooling::{normalize_l2, pool, to_host};
use crate::{gemma3::Gemma3Encoder, qwen3::Qwen3Encoder};

#[derive(Debug)]
pub enum EmbedError {
    /// No local checkpoint, or no encoder for this family — the ladder skips
    /// the rung rather than failing.
    Unsupported(String),
    Load(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Unsupported(m) => write!(f, "mlx rung unavailable: {m}"),
            EmbedError::Load(m) => write!(f, "mlx load failed: {m}"),
        }
    }
}

enum Backend {
    Qwen3(Box<Qwen3Encoder>),
    Gemma3(Box<Gemma3Encoder>),
}

pub struct Embedder {
    backend: Backend,
    checkpoint: Checkpoint,
    dim: usize,
    pooling: Pooling,
}

impl Embedder {
    /// Load a checkpoint directory. `Unsupported` means "this rung cannot
    /// serve this model", which the ladder treats as a skip.
    pub fn load(dir: &Path) -> Result<Self, EmbedError> {
        let cp = checkpoint::load(dir).map_err(EmbedError::Load)?;
        if cp.family == Family::Unsupported {
            return Err(EmbedError::Unsupported(format!(
                "no encoder for model_type in {}",
                dir.display()
            )));
        }
        let pooling = match cp.pooling {
            Pooling::Default => checkpoint::default_pooling(cp.family),
            declared => declared,
        };
        let weights = load_weights_from_dir_with_subfolders(dir)
            .map_err(|e| EmbedError::Load(e.to_string()))?;
        let backend = match cp.family {
            Family::Qwen3 => Backend::Qwen3(Box::new(
                Qwen3Encoder::load(&weights, &cp).map_err(EmbedError::Load)?,
            )),
            Family::Gemma3 => Backend::Gemma3(Box::new(
                Gemma3Encoder::load(&weights, &cp).map_err(EmbedError::Load)?,
            )),
            Family::Unsupported => unreachable!("checked above"),
        };
        Ok(Self {
            backend,
            dim: cp.hidden_size,
            checkpoint: cp,
            pooling,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// What this embedder will report at load time, so the engine's load
    /// response can name the runtime instead of implying "accelerated".
    pub fn describe(&self) -> String {
        format!(
            "mlx:{} dim={} pooling={:?} mask={:?} layers={}",
            match self.checkpoint.family {
                Family::Qwen3 => "qwen3",
                Family::Gemma3 => "gemma3_text",
                Family::Unsupported => "?",
            },
            self.dim,
            self.pooling,
            self.checkpoint.mask,
            self.checkpoint.num_hidden_layers,
        )
    }

    /// Embed tokenized rows. `rows` are right-padded here: the padding mask
    /// is built from each row's true length, and the mask — not the padded
    /// width — decides what pooling can see.
    pub fn embed(&self, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, mask) = to_padded_arrays(rows);
        let hidden = match &self.backend {
            Backend::Qwen3(m) => {
                let attn = match self.checkpoint.mask {
                    Mask::Causal => mlxcel_core::utils::create_causal_padding_mask(&mask, 0),
                    Mask::Bidirectional => {
                        mlxcel_core::utils::create_bidirectional_padding_mask(&mask)
                    }
                };
                m.forward_hidden(&ids, Some(&attn))
            }
            Backend::Gemma3(m) => m.forward_hidden(&ids, &mask, &self.checkpoint),
        };
        let pooled = pool(&hidden, &mask, self.pooling);
        let out = normalize_l2(&pooled);
        Ok(to_host(&out))
    }
}

/// Right-pad to the batch's longest row. Pad id 0 is fine: every key it could
/// be attended as is masked out, and last-token pooling reads positions from
/// the mask.
fn to_padded_arrays(rows: &[Vec<u32>]) -> (mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) {
    let b = rows.len();
    let l = rows.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
    let mut ids = vec![0i32; b * l];
    let mut mask = vec![0i32; b * l];
    for (i, r) in rows.iter().enumerate() {
        for (j, t) in r.iter().enumerate() {
            ids[i * l + j] = *t as i32;
            mask[i * l + j] = 1;
        }
    }
    let shape = [b as i32, l as i32];
    (
        mlxcel_core::from_slice_i32(&ids, &shape),
        mlxcel_core::from_slice_i32(&mask, &shape),
    )
}