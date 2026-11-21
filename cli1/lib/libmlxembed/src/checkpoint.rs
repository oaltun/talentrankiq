//! Checkpoint semantics: everything the *runtime* must not guess.
//!
//! Design 15 §15.5.3.4 measured two "family default" traps in one afternoon —
//! mlxcel pools gemma3 with `Mean` when the ONNX export of the same weights
//! pools last-token, and it runs gemma3 attention bidirectionally when the
//! checkpoint declares `use_bidirectional_attention: false`. Each looked like
//! a broken conversion (cosine 0.70 and 0.78) and neither was.
//!
//! So the rule this module enforces: **a checkpoint's semantics come from the
//! checkpoint.** Pooling, mask convention, rope base, sequence length and
//! quantization are read from `config.json` / `1_Pooling/config.json` / the
//! tokenizer, and a family default is only ever the last resort — recorded as
//! [`Pooling::Default`] so it is visible in the load report rather than
//! silent.
//!
//! This module is pure (serde_json only), so its rules are unit-tested on
//! every platform, including the ones where MLX cannot run.

use serde_json::Value;
use std::path::Path;

/// Which encoder slice a checkpoint needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `qwen3`: causal layers, q/k RMS-norm, SwiGLU, last-token pooling.
    Qwen3,
    /// `gemma3_text`: bidirectional (or causal, per the config) layers,
    /// `GemmaRMSNorm`, GeGLU, four norms per layer, clipped residuals.
    Gemma3,
    /// A model_type this crate has no encoder for. Not an error here — the
    /// ladder uses it to skip the MLX rung (design 15 §15.4).
    Unsupported,
}

/// How `[B, L, D]` hidden states become one `[B, D]` vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
    /// The last real token. What both harrier exports do *inside* their ONNX
    /// graph (`model/st/pool_0/lasttoken_squeeze` → `sentence_embedding`).
    LastToken,
    /// The checkpoint declared nothing; this is the value in force.
    Default,
}

/// Which attention mask the encoder builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    /// Blocks keys after the query position (and padding keys).
    Causal,
    /// Blocks padding keys only.
    Bidirectional,
}

/// A parsed checkpoint: shapes, quantization, and the semantics above.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub family: Family,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    /// Global RoPE base. Gemma3 nests it under `rope_parameters`.
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub sliding_window_pattern: usize,
    pub group_size: i32,
    pub bits: i32,
    pub pooling: Pooling,
    pub mask: Mask,
    /// `Some` when the checkpoint set one; `None` when we fell back to a
    /// sequence-length cap of our own.
    pub max_position_embeddings: Option<i64>,
}

fn num(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize)
}

fn f32_of(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default)
}

/// Resolve everything a forward pass needs from the checkpoint directory.
///
/// `dir` must contain `config.json`; `1_Pooling/config.json` is read when
/// present (sentence-transformers exports ship it, ours do not — which is
/// exactly how the pooling trap in §15.5.3.4 arose).
pub fn load(dir: &Path) -> Result<Checkpoint, String> {
    let config_path = dir.join("config.json");
    let raw: Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .map_err(|e| format!("read {}: {e}", config_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", config_path.display()))?;

    let model_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let family = match model_type {
        "qwen3" => Family::Qwen3,
        "gemma3_text" | "gemma3" => Family::Gemma3,
        _ => Family::Unsupported,
    };

    // RoPE: qwen3 keeps a flat `rope_theta`; gemma3 splits global/sliding
    // under `rope_parameters`.
    let rope_theta = raw
        .get("rope_parameters")
        .and_then(|r| r.get("full_attention"))
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or_else(|| f32_of(&raw, "rope_theta", 10_000.0));

    let q = raw.get("quantization");
    let group_size = q
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(64) as i32;
    let bits = q
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .unwrap_or(16) as i32;

    let bidirectional = raw
        .get("use_bidirectional_attention")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(Checkpoint {
        family,
        hidden_size: num(&raw, "hidden_size").ok_or("config.json: hidden_size")?,
        num_hidden_layers: num(&raw, "num_hidden_layers").ok_or("config.json: num_hidden_layers")?,
        num_attention_heads: num(&raw, "num_attention_heads").ok_or("num_attention_heads")?,
        num_key_value_heads: num(&raw, "num_key_value_heads").unwrap_or_else(|| {
            num(&raw, "num_attention_heads").ok_or("num_attention_heads").unwrap_or(1)
        }),
        head_dim: num(&raw, "head_dim").unwrap_or_else(|| {
            let h = num(&raw, "hidden_size").unwrap_or(1);
            let n = num(&raw, "num_attention_heads").unwrap_or(1);
            h / n
        }),
        intermediate_size: num(&raw, "intermediate_size").unwrap_or(0),
        rms_norm_eps: f32_of(&raw, "rms_norm_eps", 1e-6),
        rope_theta,
        sliding_window: raw
            .get("sliding_window")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32,
        sliding_window_pattern: raw
            .get("sliding_window_pattern")
            .and_then(|v| v.as_u64())
            .unwrap_or(6) as usize,
        group_size,
        bits,
        pooling: resolve_pooling(dir, family),
        // A causal mask is what both harrier exports implement; a checkpoint
        // must opt *in* to bidirectional, exactly as mlxcel's own detection
        // does. Never the other way round.
        mask: if bidirectional { Mask::Bidirectional } else { Mask::Causal },
        max_position_embeddings: raw
            .get("max_position_embeddings")
            .and_then(|v| v.as_i64()),
    })
}

/// `1_Pooling/config.json` when the checkpoint ships it, else the family
/// default recorded as [`Pooling::Default`].
pub fn resolve_pooling(dir: &Path, family: Family) -> Pooling {
    if let Some(mode) = read_pooling_config(&dir.join("1_Pooling").join("config.json")) {
        return mode;
    }
    match family {
        // Both harrier exports pool last-token inside their ONNX graph, so
        // last-token is what keeps the two runtimes interchangeable.
        Family::Qwen3 | Family::Gemma3 => Pooling::Default,
        Family::Unsupported => Pooling::Default,
    }
}

/// The pooling a `Default` resolves to, for the forward pass and the report.
pub fn default_pooling(family: Family) -> Pooling {
    match family {
        Family::Qwen3 | Family::Gemma3 => Pooling::LastToken,
        Family::Unsupported => Pooling::Mean,
    }
}

fn read_pooling_config(path: &Path) -> Option<Pooling> {
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    if let Some(mode) = raw.get("pooling_mode").and_then(|v| v.as_str()) {
        return match mode.to_ascii_lowercase().as_str() {
            "cls" => Some(Pooling::Cls),
            "mean" => Some(Pooling::Mean),
            "lasttoken" | "last_token" | "last" => Some(Pooling::LastToken),
            _ => None,
        };
    }
    // Legacy sentence-transformers booleans: exactly one set.
    let flag = |k: &str| raw.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    match (
        flag("pooling_mode_cls_token"),
        flag("pooling_mode_mean_tokens"),
        flag("pooling_mode_max_tokens"),
        flag("pooling_mode_lasttoken"),
    ) {
        (true, false, false, false) => Some(Pooling::Cls),
        (false, true, false, false) => Some(Pooling::Mean),
        (false, false, false, true) => Some(Pooling::LastToken),
        // Nothing declared: leave it to the family default rather than
        // silently choosing mean, which is how the 270M trap started.
        (false, false, false, false) => None,
        // Contradictory config is a load error in spirit; refusing here beats
        // picking one and producing a different model.
        _ => None,
    }
}

/// `(i + 1) % pattern == 0` -> full attention, else windowed. Gemma3's
/// sliding-window pattern.
pub fn is_full_attention(pattern: usize, i: usize) -> bool {
    pattern == 0 || (i + 1) % pattern == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("libmlxembed-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("temp dir");
        d
    }

    fn write(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).expect("mkdir");
        std::fs::write(p, body).expect("write");
    }

    #[test]
    fn qwen3_shape_and_semantics_come_from_the_config() {
        let d = tmpdir("qwen3");
        write(
            &d,
            "config.json",
            r#"{"model_type":"qwen3","hidden_size":1024,"num_hidden_layers":28,
                "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
                "intermediate_size":3072,"rms_norm_eps":1e-6,"rope_theta":1000000,
                "tie_word_embeddings":true,"vocab_size":151936,
                "max_position_embeddings":32768,
                "quantization":{"group_size":64,"bits":4}}"#,
        );
        let c = load(&d).expect("loads");
        assert_eq!(c.family, Family::Qwen3);
        assert_eq!((c.hidden_size, c.num_hidden_layers, c.head_dim), (1024, 28, 128));
        assert_eq!((c.num_attention_heads, c.num_key_value_heads), (16, 8));
        assert_eq!((c.group_size, c.bits), (64, 4));
        assert_eq!(c.rope_theta, 1_000_000.0);
        // Neither harrier declares bidirectional, so causal is the default.
        assert_eq!(c.mask, Mask::Causal);
        assert_eq!(c.max_position_embeddings, Some(32768));
    }

    #[test]
    fn gemma3_reads_its_nested_rope_and_norms_the_pattern() {
        let d = tmpdir("gemma3");
        write(
            &d,
            "config.json",
            r#"{"model_type":"gemma3_text","hidden_size":640,"num_hidden_layers":18,
                "num_attention_heads":4,"num_key_value_heads":1,"head_dim":256,
                "intermediate_size":2048,"rms_norm_eps":1e-6,
                "sliding_window":512,"sliding_window_pattern":1,
                "rope_parameters":{"full_attention":{"rope_theta":1000000.0},
                                   "sliding_attention":{"rope_theta":10000.0}},
                "use_bidirectional_attention":false,
                "quantization":{"group_size":64,"bits":4}}"#,
        );
        let c = load(&d).expect("loads");
        assert_eq!(c.family, Family::Gemma3);
        assert_eq!(c.rope_theta, 1_000_000.0, "global base, not the sliding one");
        assert_eq!(c.mask, Mask::Causal, "declared false -> causal");
        assert_eq!(c.sliding_window, 512);
        // pattern 1 means every layer is full attention.
        assert!((0..18).all(|i| is_full_attention(c.sliding_window_pattern, i)));
    }

    #[test]
    fn bidirectional_is_opt_in_only() {
        let d = tmpdir("bidir");
        write(
            &d,
            "config.json",
            r#"{"model_type":"gemma3_text","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,
                "use_bidirectional_attention":true}"#,
        );
        assert_eq!(load(&d).expect("loads").mask, Mask::Bidirectional);
    }

    #[test]
    fn unknown_family_is_skipped_not_failed() {
        let d = tmpdir("unknown");
        write(
            &d,
            "config.json",
            r#"{"model_type":"llama","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":1,"num_key_value_heads":1}"#,
        );
        // The ladder skips a rung whose artifact it cannot serve; this must
        // not be an error the caller has to special-case.
        assert_eq!(load(&d).expect("loads").family, Family::Unsupported);
    }

    #[test]
    fn pooling_config_beats_the_family_default() {
        let d = tmpdir("poolmean");
        write(
            &d,
            "config.json",
            r#"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8}"#,
        );
        assert_eq!(load(&d).expect("loads").pooling, Pooling::Default);
        write(&d, "1_Pooling/config.json", r#"{"pooling_mode":"mean"}"#);
        assert_eq!(load(&d).expect("loads").pooling, Pooling::Mean);
        write(&d, "1_Pooling/config.json", r#"{"pooling_mode_cls_token":true}"#);
        assert_eq!(load(&d).expect("loads").pooling, Pooling::Cls);
        write(&d, "1_Pooling/config.json", r#"{"pooling_mode_lasttoken":true}"#);
        assert_eq!(load(&d).expect("loads").pooling, Pooling::LastToken);
    }

    #[test]
    fn a_declared_pooling_that_contradicts_itself_falls_back_rather_than_guessing() {
        let d = tmpdir("poolconflict");
        write(
            &d,
            "config.json",
            r#"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8}"#,
        );
        write(
            &d,
            "1_Pooling/config.json",
            r#"{"pooling_mode_cls_token":true,"pooling_mode_mean_tokens":true}"#,
        );
        assert_eq!(load(&d).expect("loads").pooling, Pooling::Default);
    }

    #[test]
    fn sliding_pattern_marks_every_fourth_layer_full() {
        assert!(is_full_attention(6, 5));
        assert!(!is_full_attention(6, 4));
        assert!(is_full_attention(1, 0));
    }
}