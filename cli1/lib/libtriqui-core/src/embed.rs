//! Embedding vector serialization helpers.

/// A backend that turns text into embedding vectors.
pub trait Embedder: Send + Sync {
    fn model_id(&self) -> &str;
    fn embed(&self, texts: &[&str]) -> Result<Vec<f32>, String>;
}

/// Serialize a float vector to little-endian bytes for BLOB storage.
pub fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Deserialize a little-endian byte BLOB back into a float vector.
pub fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}