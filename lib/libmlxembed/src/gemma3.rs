//! The gemma3_text encoder slice.
//!
//! Not the qwen3 encoder at a different width — five things differ and each
//! one changes the vectors: `GemmaRMSNorm`'s `1 + w`, a `sqrt(hidden_size)`
//! scale on the token embeddings, **GeGLU** instead of SwiGLU, **four** norms
//! per layer with *clipped* residuals, and the bidirectional/causal choice,
//! which the checkpoint declares rather than the family assuming (design 15
//! §15.5.3.4: copying the runtime's unconditional bidirectional mask drops
//! agreement with our ONNX export from 0.937 to 0.703).
//!
//! Measured: matches `mlxcel embed --pooling last` at cosine 0.999998, and
//! runs at ~100 texts/s against 5.46 texts/s for our ONNX CPU path.

use mlxcel_core::layers::{
    FusedQKVLinear, GemmaRMSNorm, UnifiedEmbedding, UnifiedLinear, attention,
};
use mlxcel_core::utils::{create_bidirectional_padding_mask, create_bidirectional_window_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::checkpoint::{self, Checkpoint, Mask};

pub struct Gemma3Encoder {
    pub embed: UnifiedEmbedding,
    layers: Vec<Block>,
    pub norm: GemmaRMSNorm,
    hidden_size: usize,
}

struct Block {
    attn: Attn,
    mlp: Mlp,
    input_norm: GemmaRMSNorm,
    post_attn_norm: GemmaRMSNorm,
    pre_ffn_norm: GemmaRMSNorm,
    post_ffn_norm: GemmaRMSNorm,
}

struct Attn {
    qkv: FusedQKVLinear,
    o_proj: UnifiedLinear,
    q_norm: GemmaRMSNorm,
    k_norm: GemmaRMSNorm,
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
                .map(|x| mlxcel_core::copy(x))
                .ok_or_else(|| format!("gemma3: missing weight {name}"))
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
            q_norm: GemmaRMSNorm::new(get(&format!("{prefix}.q_norm.weight"))?, c.rms_norm_eps),
            k_norm: GemmaRMSNorm::new(get(&format!("{prefix}.k_norm.weight"))?, c.rms_norm_eps),
            n_heads: c.num_attention_heads as i32,
            n_kv_heads: c.num_key_value_heads as i32,
            head_dim: c.head_dim as i32,
            scale: 1.0 / (c.head_dim as f32).sqrt(),
            rope_base: c.rope_theta,
        })
    }

    fn forward(&self, x: &MlxArray, mask: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (q, k, v) = self.qkv.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.n_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.n_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.n_kv_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        // gemma3 normalizes q/k AFTER the transpose, unlike qwen3.
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
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
        let act = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        self.down.forward(&act)
    }
}

impl Gemma3Encoder {
    pub fn load(w: &WeightMap, c: &Checkpoint) -> Result<Self, String> {
        let get = |name: &str| {
            w.get(name)
                .map(|x| mlxcel_core::copy(x))
                .ok_or_else(|| format!("gemma3: missing weight {name}"))
        };
        let embed = UnifiedEmbedding::from_weights(w, "model.embed_tokens", c.group_size, c.bits)
            .map_err(|e| format!("gemma3 embed_tokens: {e}"))?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(Block {
                attn: Attn::load(w, &format!("{prefix}.self_attn"), c)?,
                mlp: Mlp::load(w, &format!("{prefix}.mlp"), c)?,
                input_norm: GemmaRMSNorm::new(
                    get(&format!("{prefix}.input_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
                post_attn_norm: GemmaRMSNorm::new(
                    get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
                pre_ffn_norm: GemmaRMSNorm::new(
                    get(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
                post_ffn_norm: GemmaRMSNorm::new(
                    get(&format!("{prefix}.post_feedforward_layernorm.weight"))?,
                    c.rms_norm_eps,
                ),
            });
        }
        let norm = GemmaRMSNorm::new(get("model.norm.weight")?, c.rms_norm_eps);
        Ok(Self {
            embed,
            layers,
            norm,
            hidden_size: c.hidden_size,
        })
    }

    /// `[B, L]` ids + `[B, L]` padding mask -> `[B, L, H]` hidden states.
    pub fn forward_hidden(
        &self,
        ids: &MlxArray,
        pad_mask: &MlxArray,
        c: &Checkpoint,
    ) -> UniquePtr<MlxArray> {
        // Gemma scales token embeddings by sqrt(hidden_size).
        let mut h = self.embed.forward(ids);
        h = mlxcel_core::multiply_scalar(&h, (self.hidden_size as f32).sqrt());

        let full = create_bidirectional_padding_mask(pad_mask);
        let causal = if c.mask == Mask::Causal {
            Some(mlxcel_core::utils::create_causal_padding_mask(pad_mask, 0))
        } else {
            None
        };
        let sliding = if c.sliding_window >= 1
            && (0..self.layers.len()).any(|i| !checkpoint::is_full_attention(c.sliding_window_pattern, i))
        {
            Some(create_bidirectional_window_mask(pad_mask, c.sliding_window))
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            // A causal layer inside a bidirectional model still gets the
            // bidirectional mask; only an explicit Causal checkpoint narrows.
            let windowed = !checkpoint::is_full_attention(c.sliding_window_pattern, i);
            let m: &MlxArray = match (&causal, windowed) {
                (Some(c), _) => c,
                (None, true) => sliding.as_deref().unwrap_or(&full),
                (None, false) => &full,
            };
            let normed = layer.input_norm.forward(&h);
            let attn_out = layer.attn.forward(&normed, Some(m));
            let post = layer.post_attn_norm.forward(&attn_out);
            let h2 = mlxcel_core::compiled_clip_residual(&h, &post);
            let normed = layer.pre_ffn_norm.forward(&h2);
            let ff = layer.mlp.forward(&normed);
            let post = layer.post_ffn_norm.forward(&ff);
            h = mlxcel_core::compiled_clip_residual(&h2, &post);
        }
        self.norm.forward(&h)
    }
}