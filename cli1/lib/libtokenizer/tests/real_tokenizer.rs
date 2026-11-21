use std::path::PathBuf;

use libtokenizer::TokenizerHandle;

fn find_real_tokenizer() -> Option<PathBuf> {
    let candidates = [
        "multilingual-e5-large_model_qint8_avx512_vnni/repo/onnx/tokenizer.json",
        "nomic-embed-text-v1.5/repo/onnx/tokenizer.json",
        "harrier-oss-v1-270m-ONNX_model_q4f16/repo/onnx/tokenizer.json",
    ];
    let base = dirs_home().join(".cache/ik/models");
    for rel in &candidates {
        let p = base.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp"))
}

#[test]
fn real_tokenizer_load_and_count() {
    let path = match find_real_tokenizer() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no cached tokenizer found under ~/.cache/ik/models/");
            return;
        }
    };
    eprintln!("using real tokenizer: {}", path.display());
    let t = TokenizerHandle::from_file(&path).unwrap();
    assert!(t.count_tokens("hello world") >= 2);
    assert!(t.count_tokens("the quick brown fox jumps over the lazy dog") > 0);
}

#[test]
fn real_tokenizer_chunk_long_text() {
    let path = match find_real_tokenizer() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no cached tokenizer found");
            return;
        }
    };
    let t = TokenizerHandle::from_file(&path).unwrap();
    let long_text = "Senior software engineer with 10 years of experience in distributed systems, "
        .repeat(50);
    let total_tokens = t.count_tokens(&long_text);
    assert!(total_tokens > 100, "expected >100 tokens, got {total_tokens}");

    let chunk_size = 128;
    let overlap = 19;
    let chunks = t.chunk_text(&long_text, chunk_size, overlap).unwrap();
    assert!(chunks.len() > 1, "long text should produce multiple chunks");

    for (i, c) in chunks.iter().enumerate() {
        assert!(
            c.token_count <= chunk_size,
            "chunk {i}: token_count {} > chunk_size {chunk_size}",
            c.token_count
        );
        assert!(
            !c.text.is_empty() || c.token_count == 0,
            "chunk {i}: non-zero tokens but empty text"
        );
    }

    for w in chunks.windows(2) {
        let overlap_bytes = w[0].char_end.saturating_sub(w[1].char_start);
        assert!(
            overlap_bytes > 0,
            "consecutive chunks should overlap: chunk_end={} > next_start={}",
            w[0].char_end,
            w[1].char_start
        );
    }
}

#[test]
fn real_tokenizer_encode_returns_consistent_fields() {
    let path = match find_real_tokenizer() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no cached tokenizer found");
            return;
        }
    };
    let t = TokenizerHandle::from_file(&path).unwrap();
    let enc = t.encode("hello world", true).unwrap();
    assert_eq!(enc.ids.len(), enc.attention_mask.len());
    assert_eq!(enc.ids.len(), enc.type_ids.len());
    assert_eq!(enc.ids.len(), enc.offsets.len());
    assert!(enc.ids.len() >= 2);
}

#[test]
fn real_tokenizer_encode_truncated_respects_limit() {
    let path = match find_real_tokenizer() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no cached tokenizer found");
            return;
        }
    };
    let t = TokenizerHandle::from_file(&path).unwrap();
    let text = "software engineer ".repeat(200);
    let max = 64;
    let enc = t.encode_truncated(&text, true, max).unwrap();
    assert_eq!(enc.ids.len(), max);
    assert_eq!(enc.attention_mask.len(), max);
}

#[test]
fn real_tokenizer_short_text_no_chunk() {
    let path = match find_real_tokenizer() {
        Some(p) => p,
        None => {
            eprintln!("skipping: no cached tokenizer found");
            return;
        }
    };
    let t = TokenizerHandle::from_file(&path).unwrap();
    let chunks = t.chunk_text("short", 512, 0).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "short");
}