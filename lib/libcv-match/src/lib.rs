pub use libembedrank::cosine_similarity;
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    cosine_similarity(a, b)
}

pub mod cosine {
    pub use super::cosine;
}
pub mod types {
    pub use super::*;
}