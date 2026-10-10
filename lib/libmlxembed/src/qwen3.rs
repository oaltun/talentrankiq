//! The qwen3 encoder slice: fused QKV, q/k RMS-norm, RoPE, masked SDPA,
//! SwiGLU, last-token pooling.
//!
//! Measured (design 15 §15.5.3.3): this composition matches `mlxcel embed
//! --pooling last` at cosine 0.999999 on harrier-mlx-600m, and runs at 45.7
//! texts/s against 1.56 texts/s for our ONNX CPU path — 29×.

use mlxcel_core::layers::{FusedQKVLinear, RMSNorm, UnifiedEmbedding, UnifiedLinear, attention};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::checkpoint::Checkpoint;

pub struct Qwen3Encoder {
    pub embed: UnifiedEmbedding,
    layers: Vec<Block>,
    pub norm: RMSNorm,
}

struct Block {
    attn: Attn,
    mlp: Mlp,
    input_norm: RMSNorm,
    post_norm: RMSNorm,
}

struct Attn {
    qkv: FusedQKVLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_base: f32,
}

impl Attn {
    fn load(w: &WeightMap, prefix: &str, c: &Checkpoint) -> Result<Self, String> {
        let get = |name: &str| {
            w.get(name)
                .map(|a| mlxcel_core::copy(a))
                .ok_or_else(|| format!("qwen3: missing weight {name}"))
        };
        Ok(Self {
            qkv: FusedQKVLinear::from_weights_separate(
                w,
                prefix,
                c.group_size,
                c.bits,
                c.num_attention_heads as i32,
                c.num_key_value_heads as i32,
                c.head_dim as i32,
            )?,
            o_proj: UnifiedLinear::from_weights(
                w,
                &format!("{prefix}.o_proj"),
                c.group_size,
                c.bits,
            )?,
            q_norm: RMSNorm::new(get(&format!("{prefix}.q_norm.weight"))?, c.rms_norm_eps),
            k_norm: RMSNorm::new(get(&format!("{prefix}.k_norm.weight"))?, c.rms_norm_eps),
            n_heads: c.num_attention_heads as i32,
            n_kv_heads: c.num_key_value_heads as i32,
            head_dim: c.head_dim as i32,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope_base: c.rope_theta,
        })
    }

    /// `x` is `[B, L, H]`; `mask` is the additive mask from
    /// `create_causal_padding_mask` (`[B, 1, L, T]`).
    fn forward(&self, x: &MlxArray, mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (q, k, v) = self.qkv.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);
        // qwen3 normalizes BEFORE the transpose.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        let q = mlxcel_core::fast_rope(&q, self.head_dim, false, self.rope_base, 1.0, 0);
        let k = mlxcel_core::fast_rope(&k, self.head_dim, false, self.rope_base, 1.0, 0);
        let out = attention(&q, &k, &v, self.scale, mask, 0.0, 0);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate: UnifiedLinear,
    up: UnifiedLinear,
    down: UnifiedLinear,
}

impl Mlp {
    fn load(w: &WeightMap, prefix: &str, c: &Checkpoint) -> Result<Self, String> {
        Ok(Self {
            gate: UnifiedLinear::from_weights(
                w,
                &format!("{prefix}.gate_proj"),
                c.group_size,
                c.bits,
            )?,
            up: UnifiedLinear::from_weights(
                w,
                &format!("{prefix}.up_proj"),
                c.group_size,
                c.bits,
            )?,
            down: UnifiedLinear::from_weights(
                w,
                &format!("{prefix}.down_proj"),
                c.group_size,
                c.bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate.forward(x);
        let up = self.up.forward(x);
        let act = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down.forward(&act)
    }
}

impl Qwen3Encoder {
    pub fn load(w: &WeightMap, c: &Checkpoint) -> Result<Self, String> {
        let get = |name: &str| {
            w.get(name)
                .map(|a| mlxcel_core::copy(a))
                .ok_or_else(|| format!("qwen3: missing weight {name}"))
        };
        let embed = UnifiedEmbedding::from_weights(w, "model.embed_tokens", c.group_size, c.bits)
            .map_err(|e| format!("qwen3 embed_tokens: {e}"))?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(Block {
                attn: Attn::load(w, &format!("{prefix}.self_attn"), c)?,
                mlp: Mlp::load(w, &format!("{prefix}.mlp"), c)?,
                input_norm: RMSNorm::new(
                    get(&format!("{prefix}.input_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
                post_norm: RMSNorm::new(
                    get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
            });
        }
        let norm = RMSNorm::new(get("model.norm.weight")?, c.rms_norm_eps);
        Ok(Self {
            embed,
            layers,
            norm,
        })
    }

    /// `[B, L]` ids + mask -> `[B, L, H]` hidden states (no head).
    pub fn forward_hidden(
        &self,
        ids: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embed.forward(ids);
        for layer in &self.layers {
            let normed = layer.input_norm.forward(&h);
            let attn_out = layer.attn.forward(&normed, mask);
            let x = mlxcel_core::add(&h, &attn_out);
            let normed = layer.post_norm.forward(&x);
            let ff = layer.mlp.forward(&normed);
            h = mlxcel_core::add(&x, &ff);
        }
        self.norm.forward(&h)
    }
}