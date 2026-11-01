pub use libembedrank::cosine_similarity;
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    cosine_similarity(a, b)
}
