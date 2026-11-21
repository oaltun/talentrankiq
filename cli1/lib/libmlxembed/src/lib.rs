//! `libmlxembed` — the MLX/Metal arm of the engine's backend ladder
//! (design 15 §15.5.3).
//!
//! One binary, no Python, no subprocess: the encoders are built directly on
//! [`mlxcel_core`] primitives, so the host still owns a single dedicated
//! wrapper thread and `run_stdio_over`'s JSONL contract does not move.
//!
//! What this crate is *not*: a copy of mlxcel. Their library ships the layer
//! primitives (fused QKV, unified linear/embedding, RMS norms, rope, masked
//! SDPA, safetensors loading); the embedding *models* live in their
//! binaries-only package. So the encoder composition — qwen3 and
//! gemma3_text — is ours, and the measured cost of writing it is recorded in
//! design 15 §15.5.3.1/§15.5.3.3.
//!
//! The rule that shapes the API is in [`checkpoint`]: a checkpoint's pooling,
//! mask convention, rope base and quantization come from the checkpoint, never
//! from a runtime family default. Two family defaults (mlxcel's gemma3
//! `Mean` pooling and its unconditional bidirectional mask) were measured
//! silently producing a *different model* from the same weights.

pub mod checkpoint;

// The encoder modules are gated on `macos` *and* the `mlx` feature: the
// feature is what actually brings the vendored dependency in, so gating on it
// keeps a default build of this crate to pure-Rust config parsing.


#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod gemma3;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod pooling;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod protocol;
#[cfg(all(target_os = "macos", feature = "mlx"))]
pub mod qwen3;

#[cfg(all(target_os = "macos", feature = "mlx"))]
mod encode;

#[cfg(all(target_os = "macos", feature = "mlx"))]
pub use encode::{Embedder, EmbedError};

#[cfg(all(target_os = "macos", feature = "mlx"))]
pub use protocol::run_stdio_over;

/// Serve the JSONL loop on a build that has no MLX arm.
///
/// Present so a host can hold the same call site unconditionally: every
/// request is answered with an `unsupported_build` error rather than the
/// process hanging on a loop that would never answer. Reaching this means the
/// ladder asked for Metal in a build that does not have it, which
/// [`AVAILABLE`] is meant to prevent — so it logs loudly and says so on the
/// wire instead of failing open.
#[cfg(not(all(target_os = "macos", feature = "mlx")))]
pub fn run_stdio_over(input: impl std::io::BufRead, mut output: impl std::io::Write) {
    log::error!("libmlxembed: MLX arm not built in (macOS + --features mlx required)");
    for line in input.lines() {
        let id = line
            .ok()
            .and_then(|l| serde_json::from_str::<serde_json::Value>(&l).ok())
            .and_then(|v| v.get("id").and_then(|i| i.as_str()).map(str::to_string));
        let out = serde_json::json!({
            "id": id,
            "type": "error",
            "payload": { "code": "unsupported_build", "msg": "this build has no MLX arm" },
        });
        let _ = writeln!(output, "{out}");
        let _ = output.flush();
    }
}

/// Whether this build can serve the MLX rung at all.
///
/// False off macOS, and false when the crate is built without `--features
/// mlx`. The ladder asks before it tries, so an unavailable arm reports
/// "unavailable" instead of a load failure — and a non-accelerated build of
/// this crate stays a few seconds of pure-Rust config parsing rather than a
/// vendored MLX compile.
pub const AVAILABLE: bool = cfg!(all(target_os = "macos", feature = "mlx"));