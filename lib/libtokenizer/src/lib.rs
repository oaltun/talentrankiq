use std::path::Path;

use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct TokenizerHandle {
    inner: Tokenizer,
}

#[derive(thiserror::Error, Debug)]
pub enum TokenizerError {
    #[error("failed to load tokenizer from {path}: {reason}")]
    Load { path: String, reason: String },
    #[error("tokenization failed: {0}")]
    Tokenize(String),
}

impl TokenizerHandle {
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|e| TokenizerError::Load {
            path: path.to_string_lossy().to_string(),
            reason: e.to_string(),
        })?;
        Ok(Self { inner })
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.len(),
            Err(_) => 0,
        }
    }

    pub fn chunk_text(
        &self,
        text: &str,
        max_tokens: usize,
        overlap_tokens: usize,
    ) -> Result<Vec<Chunk>, TokenizerError> {
        if max_tokens == 0 {
            return Err(TokenizerError::Tokenize("max_tokens must be > 0".into()));
        }

        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Tokenize(e.to_string()))?;

        let ids: Vec<u32> = encoded.get_ids().to_vec();
        let offsets: Vec<(usize, usize)> =
            encoded.get_offsets().iter().map(|&(s, e)| (s, e)).collect();

        if ids.len() <= max_tokens {
            return Ok(vec![Chunk {
                text: text.to_string(),
                token_count: ids.len(),
                char_start: 0,
                char_end: text.len(),
            }]);
        }

        let mut chunks = Vec::new();
        let mut pos = 0usize;

        while pos < ids.len() {
            let end = (pos + max_tokens).min(ids.len());
            let char_start = offsets[pos].0;
            let char_end = offsets[end - 1].1;
            let chunk_text = text[char_start..char_end].to_string();
            let token_count = end - pos;

            chunks.push(Chunk {
                text: chunk_text,
                token_count,
                char_start,
                char_end,
            });

            if end >= ids.len() {
                break;
            }

            let next = if overlap_tokens >= max_tokens {
                end
            } else {
                (end - overlap_tokens).max(pos + 1)
            };

            if next <= pos {
                break;
            }
            pos = next;
        }

        Ok(chunks)
    }
}

pub struct TokenEncoding {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
}

impl TokenizerHandle {
    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenEncoding, TokenizerError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Tokenize(e.to_string()))?;
        Ok(TokenEncoding {
            ids: enc.get_ids().to_vec(),
            attention_mask: enc.get_attention_mask().to_vec(),
            type_ids: enc.get_type_ids().to_vec(),
            offsets: enc.get_offsets().iter().map(|&(s, e)| (s, e)).collect(),
        })
    }

    pub fn encode_truncated(
        &self,
        text: &str,
        add_special_tokens: bool,
        max_len: usize,
    ) -> Result<TokenEncoding, TokenizerError> {
        let mut enc = self.encode(text, add_special_tokens)?;
        if enc.ids.len() > max_len {
            enc.ids.truncate(max_len);
            enc.attention_mask.truncate(max_len);
            enc.type_ids.truncate(max_len);
            enc.offsets.truncate(max_len);
        }
        Ok(enc)
    }

    /// Detokenize ids back to text (special tokens skipped when asked).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Tokenize(e.to_string()))
    }

    /// Vocabulary string of one token id (for logit masking).
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Total vocabulary size (mask buffers, iteration bounds).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

pub struct Chunk {
    pub text: String,
    pub token_count: usize,
    pub char_start: usize,
    pub char_end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_dir() -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tid = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("libtokenizer_test_{tid}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let vocab = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 200,
                "vocab": {
                    "[UNK]": 0, "the": 1, "quick": 2, "brown": 3,
                    "fox": 4, "jumps": 5, "over": 6, "lazy": 7,
                    "dog": 8, "cat": 9, "sits": 10, "on": 11, "mat": 12
                }
            }
        });
        let json = serde_json::to_string(&vocab).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn count_tokens_simple() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        assert_eq!(t.count_tokens("the quick brown fox"), 4);
    }

    #[test]
    fn count_tokens_unknown() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        assert_eq!(t.count_tokens("the xyz fox"), 3);
    }

    #[test]
    fn chunk_text_fits_in_one_chunk() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        let chunks = t.chunk_text("the quick brown fox", 10, 0).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "the quick brown fox");
        assert_eq!(chunks[0].token_count, 4);
    }

    #[test]
    fn chunk_text_splits_correctly() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        let text = "the quick brown fox jumps over the lazy dog";
        let chunks = t.chunk_text(text, 4, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].token_count, 4);
        assert_eq!(chunks[0].text, "the quick brown fox");
        assert_eq!(chunks[1].token_count, 4);
        assert_eq!(chunks[2].token_count, 1);
        assert_eq!(chunks[2].text, "dog");
    }

    #[test]
    fn chunk_text_with_overlap() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        let text = "the quick brown fox jumps over the lazy dog";
        let chunks = t.chunk_text(text, 4, 1).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].token_count, 4);
        assert_eq!(chunks[1].token_count, 4);
        let overlap_text = &text[chunks[1].char_start..chunks[0].char_end];
        assert!(!overlap_text.is_empty());
    }

    #[test]
    fn chunk_text_empty_string() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        let chunks = t.chunk_text("", 10, 0).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "");
        assert_eq!(chunks[0].token_count, 0);
    }

    #[test]
    fn chunk_text_zero_max_tokens_errors() {
        let path = make_test_dir();
        let t = TokenizerHandle::from_file(&path).unwrap();
        let result = t.chunk_text("hello", 0, 0);
        assert!(result.is_err());
    }
}
