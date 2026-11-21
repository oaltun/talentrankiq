//! Parity gate for the MLX arm against a reference runtime.
//!
//! Needs `--features mlx` (the arm is opt-in; see Cargo.toml) plus a local
//! checkpoint, and the second gate needs a reference implementation:
//!
//! 1. `IK_MLX_PARITY_MODEL=<checkpoint dir>` — embed and check the vectors are
//!    unit-norm and finite, then dump them to `IK_MLX_PARITY_DUMP`.
//! 2. `IK_MLX_PARITY_REF=<vectors json>` — the dumped vectors are compared to
//!    that reference at `IK_MLX_PARITY_MIN_COSINE` (default 0.999), which is
//!    the shape of the mlxcel-CLI comparison in design 15 §15.5.3.3.
//!
//! **Which reference matters, per family.** For the qwen3 600M, `mlxcel embed
//! --pooling last` *is* the reference: 0.999999. For the gemma3 270M it is
//! not, and treating it as one would "fail" a correct implementation at 0.947
//! — because mlxcel builds bidirectional masks for that family while the
//! checkpoint declares `use_bidirectional_attention: false`. Our vectors match
//! a causal run to 1.000000 and our ONNX export at 0.937. The reference for
//! gemma3 is our own ONNX export (or a causal mlxcel run); see design 15
//! §15.5.3.4.
//!
//! Why opt-in rather than a fixture in-tree: the reference vectors come from a
//! *runtime*, not from us, so the gate only means something where both a
//! checkpoint and that runtime exist. The measurements that justify the
//! thresholds live in the design doc; this test keeps them from rotting.

#![cfg(all(target_os = "macos", feature = "mlx"))]

use std::path::PathBuf;

use libmlxembed::Embedder;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn texts() -> Vec<String> {
    vec![
        "Deneyimli makine öğrenmesi mühendislik rolü için beş yıllık deneyim".to_string(),
        "Senior backend developer with five years of distributed systems".to_string(),
        "İstanbul ofisinde satış temsilcisi olarak görev yaptım".to_string(),
        "Certified Kubernetes administrator operating production clusters"
            .to_string(),
    ]
}

fn tokenize(model_dir: &std::path::Path, texts: &[String]) -> Vec<Vec<u32>> {
    // The same crate the engine's libtokenizer uses to produce ids, so this
    // gate never measures tokenizer drift (design 15 §15.5.3.3).
    let mut tk = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("tokenizer.json loads");
    tk.with_padding(None);
    let _ = tk.with_truncation(None);
    texts
        .iter()
        .map(|t| tk.encode(t.as_str(), true).expect("encode").get_ids().to_vec())
        .collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
}

#[test]
fn mlx_vectors_are_unit_norm_and_match_the_reference() {
    let model_dir = env_path("IK_MLX_PARITY_MODEL");
    let Some(model_dir) = model_dir else {
        eprintln!(
            "skipped: set IK_MLX_PARITY_MODEL=<checkpoint dir> (and optionally \
             IK_MLX_PARITY_REF / IK_MLX_PARITY_DUMP / IK_MLX_PARITY_MIN_COSINE)"
        );
        return;
    };

    let embedder = Embedder::load(&model_dir).expect("checkpoint loads on the MLX arm");
    eprintln!("loaded: {}", embedder.describe());

    let texts = texts();
    let rows = tokenize(&model_dir, &texts);
    let vectors = embedder.embed(&rows).expect("embed runs");

    assert_eq!(vectors.len(), texts.len());
    for (v, t) in vectors.iter().zip(texts.iter()) {
        assert_eq!(v.len(), embedder.dim(), "dim for {t:?}");
        assert!(v.iter().all(|x| x.is_finite()), "finite for {t:?}");
        assert!(
            (norm(v) - 1.0).abs() < 1e-3,
            "L2-normalized for {t:?}: got {}",
            norm(v)
        );
    }
    // A degenerate encoder (no attention, collapsed state) can still emit
    // unit vectors; distinct texts must not share one.
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            assert!(
                cosine(&vectors[i], &vectors[j]) < 0.999_9,
                "texts {i} and {j} collapse to the same vector"
            );
        }
    }

    if let Some(dump) = env_path("IK_MLX_PARITY_DUMP") {
        std::fs::write(&dump, serde_json::to_string(&vectors).unwrap())
            .expect("dump vectors");
        eprintln!("dumped {} vectors to {}", vectors.len(), dump.display());
    }

    let Some(ref_path) = env_path("IK_MLX_PARITY_REF") else {
        return;
    };
    let min: f32 = std::env::var("IK_MLX_PARITY_MIN_COSINE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.999);
    let reference: Vec<Vec<f32>> =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).expect("reference reads"))
            .expect("reference parses");
    assert_eq!(reference.len(), vectors.len(), "same text count");

    let mut worst = f32::INFINITY;
    for (i, (mine, theirs)) in vectors.iter().zip(reference.iter()).enumerate() {
        let c = cosine(mine, theirs);
        assert!(
            c >= min,
            "text {i} ({:?}) cosine {c:.6} below the {min} gate",
            texts[i]
        );
        worst = worst.min(c);
    }
    eprintln!("parity vs reference: worst per-text cosine {worst:.6} (gate {min})");
}