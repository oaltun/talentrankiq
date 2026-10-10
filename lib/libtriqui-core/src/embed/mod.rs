//! Text embedding backends for Baseline ranking.
//!
//! The ranking pipeline is backend-agnostic behind the [`Embedder`] trait. The
//! production backend is [`ModelCallerEmbedder`], which drives the `modelcaller`
//! orchestrator over a stdio JSONL pipe (no HTTP, no port); [`MockEmbedder`] is a
//! deterministic stand-in for unit tests.
//!
//! Vectors are stored as raw **little-endian f32** byte arrays — see
//! [`vector_to_bytes`] / [`bytes_to_vector`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::types::DomainError;
use libmodelcallerclient::ModelCallerClient;

/// Error from an embedding backend.
#[derive(Debug)]
pub enum EmbedError {
    /// An HTTP/transport failure talking to the engine.
    Transport(String),
    /// The engine responded with an error status or an unexpected body.
    Decode(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Transport(m) => write!(f, "embedding transport error: {m}"),
            EmbedError::Decode(m) => write!(f, "embedding decode error: {m}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Map an embedding failure onto the app's typed domain error.
/// Transport faults become IO errors; decode faults become parse errors. (The
/// "engine not ready" case is surfaced directly as a `ValidationError` by the
/// runtime accessor, so it never round-trips through `EmbedError`.)
impl From<EmbedError> for DomainError {
    fn from(e: EmbedError) -> Self {
        match e {
            EmbedError::Transport(m) => DomainError::IoError { message: m },
            EmbedError::Decode(m) => DomainError::ParseError { message: m },
        }
    }
}

/// A text-embedding backend. Blocking — runs on a background `std::thread`.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts; returns one vector per input, in order.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
    /// The model id (e.g. `nomic-embed-text-v1.5`).
    fn model_id(&self) -> &str;
    /// Vector dimension; `0` until established by the first successful response.
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// busy tracking (edge-triggered, for the status-bar engine chip)
// ---------------------------------------------------------------------------

/// Transition callback: `true` = became busy, `false` = became idle.
pub type BusyHook = Arc<dyn Fn(bool) + Send + Sync>;

/// In-flight-work counter with an optional transition hook. Fires `true` on the
/// idle→busy edge and `false` on busy→idle, so an import with dozens of batch
/// embeds produces exactly two notifications instead of one per request.
#[derive(Default)]
pub struct BusyFlag {
    count: AtomicUsize,
    hook: Mutex<Option<BusyHook>>,
}

impl BusyFlag {
    /// Register the transition hook (replaces any previous one).
    pub fn set_hook(&self, hook: BusyHook) {
        *self.hook.lock().expect("busy hook lock") = Some(hook);
    }

    /// Whether any embed is currently in flight.
    pub fn is_busy(&self) -> bool {
        self.count.load(Ordering::SeqCst) > 0
    }

    fn enter(&self) -> BusyGuard<'_> {
        if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
            self.fire(true);
        }
        BusyGuard { flag: self }
    }

    fn fire(&self, busy: bool) {
        // Clone the hook out so the callback never runs under the lock.
        let hook = self.hook.lock().expect("busy hook lock").clone();
        if let Some(hook) = hook {
            hook(busy);
        }
    }
}

/// Decrements the counter on drop and fires the busy→idle edge at zero.
/// Drop-based so early returns in `embed` can't strand the busy state.
struct BusyGuard<'a> {
    flag: &'a BusyFlag,
}

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        if self.flag.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.flag.fire(false);
        }
    }
}

// ---------------------------------------------------------------------------
// modelcaller stdio backend (shared libmodelcallerclient)
// ---------------------------------------------------------------------------

/// Embeds text via the `modelcaller` orchestrator over a stdio JSONL pipe, using
/// the shared `libmodelcallerclient` (so the protocol can't drift from `bench`).
pub struct ModelCallerEmbedder {
    client: Arc<ModelCallerClient>,
    model: String,
    dim: OnceLock<usize>,
    busy: Arc<BusyFlag>,
}

impl ModelCallerEmbedder {
    pub fn new(client: Arc<ModelCallerClient>, model: String) -> Self {
        Self {
            client,
            model,
            dim: OnceLock::new(),
            busy: Arc::new(BusyFlag::default()),
        }
    }

    /// Shared busy flag: the runtime installs a transition hook so the UI can
    /// show Working/Ready edges.
    pub fn busy_flag(&self) -> Arc<BusyFlag> {
        self.busy.clone()
    }

    /// Whether an embed is currently in flight.
    pub fn is_busy(&self) -> bool {
        self.busy.is_busy()
    }
}

impl Embedder for ModelCallerEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let _busy = self.busy.enter();
        // Lowercase before embedding: this is the preprocessing every gui
        // vector is produced with, matching bench defaults and the judge
        // artifacts' `text_preprocessing: lowercase` training spec.
        let lowered: Vec<String> = texts.iter().map(|t| t.to_lowercase()).collect();
        let refs: Vec<&str> = lowered.iter().map(|s| s.as_str()).collect();
        // `texts_role` is omitted: the engine always ranks CVs/documents against a
        // job ad, so every text is treated as a document.
        let embeddings = self
            .client
            .embed(&self.model, &refs, None)
            .map_err(|e| EmbedError::Transport(e.to_string()))?;

        if embeddings.len() != texts.len() {
            return Err(EmbedError::Decode(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }

        let mut out = Vec::with_capacity(embeddings.len());
        for v in embeddings {
            // `OnceLock::set` only succeeds the first time, so this records the
            // model's dimension from the very first response and then ignores it.
            let _ = self.dim.set(v.len());
            let mut v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
            normalize(&mut v);
            out.push(v);
        }
        Ok(out)
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        // `OnceLock` is `Sync`; read directly. Returns 0 until the first response.
        match self.dim.get() {
            Some(d) => *d,
            None => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// deterministic test backend
// ---------------------------------------------------------------------------

/// Deterministic embedding backend for tests: a stable hash of the text mapped to a
/// fixed-dimension unit vector. Not semantically meaningful — only for exercising the
/// ranking pipeline without a network or model.
#[cfg(test)]
pub struct MockEmbedder {
    model: String,
    dim: usize,
}

#[cfg(test)]
impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            model: "mock".to_string(),
            dim,
        }
    }

    fn hash_vec(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for (i, b) in text.as_bytes().iter().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            v[i % self.dim] += sign * (*b as f32);
        }
        normalize(&mut v);
        v
    }
}

#[cfg(test)]
impl Embedder for MockEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| self.hash_vec(t)).collect())
    }
    fn model_id(&self) -> &str {
        &self.model
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// vector math + (de)serialization helpers (shared with ranker/storage)
// ---------------------------------------------------------------------------

/// L2-normalize a vector in place. The zero vector is left as-is (norm 0).
pub fn normalize(v: &mut [f32]) {
    let norm = (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
    if norm > 0.0 {
        let n = norm as f32;
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Cosine similarity for two equal-length vectors. Assumes nothing about norm;
/// returns 0.0 if either vector is zero-length.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na = (a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
    let nb = (b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

/// Alias for `cosine` for callers expecting the name `cosine_similarity`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    cosine(a, b)
}

/// In-place `acc += coef * v` for equal-length vectors; a no-op on a length mismatch.
pub fn add_weighted(acc: &mut [f32], v: &[f32], coef: f32) {
    if acc.len() != v.len() {
        return;
    }
    for (a, x) in acc.iter_mut().zip(v.iter()) {
        *a += coef * x;
    }
}

/// Componentwise mean of a set of vectors (`None` for an empty set). Used by the
/// feedback learner to average candidate vectors into gain/loss centroids.
pub fn centroid(vectors: &[&[f32]]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }
    let dim = vectors[0].len();
    let mut acc = vec![0.0_f32; dim];
    for v in vectors {
        add_weighted(&mut acc, v, 1.0);
    }
    let n = vectors.len() as f32;
    for x in acc.iter_mut() {
        *x /= n;
    }
    Some(acc)
}

/// Encode a vector as raw little-endian f32 bytes (the on-disk BLOB format).
pub fn vector_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Decode raw little-endian f32 bytes into a vector.
/// If `expected_dim` is given, the decoded dimension must match.
pub fn bytes_to_vector(bytes: &[u8], expected_dim: Option<usize>) -> Result<Vec<f32>, EmbedError> {
    if bytes.len() % 4 != 0 {
        return Err(EmbedError::Decode(format!(
            "vector blob length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let dim = bytes.len() / 4;
    if let Some(expected) = expected_dim {
        if dim != expected {
            return Err(EmbedError::Decode(format!(
                "vector dimension {dim} != expected {expected}"
            )));
        }
    }
    let mut v = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact(4) yields 4 bytes");
        v.push(f32::from_le_bytes(arr));
    }
    Ok(v)
}

/// SHA-256 of the *preprocessed* (lowercased) text as lowercase hex — a stable
/// id for embedding caching + dedup. Lowercasing here (not just at the embedder)
/// keeps cache keys in lockstep with the text actually embedded, matching the
/// judge artifacts' `text_preprocessing: lowercase` training spec. Texts
/// embedded before this change have old-key cache rows and re-embed once.
pub fn content_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.to_lowercase().as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_makes_unit_vector() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_zero_stays_zero() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![0.1, 0.2, 0.3];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn bytes_roundtrip_preserves_vector() {
        let v = vec![0.12, -0.34, 0.56, 1.0, -2.5];
        let bytes = vector_to_bytes(&v);
        let back = bytes_to_vector(&bytes, Some(v.len())).unwrap();
        for (x, y) in v.iter().zip(back.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn bytes_rejects_wrong_dim() {
        let v = vec![0.0; 4];
        let bytes = vector_to_bytes(&v);
        assert!(bytes_to_vector(&bytes, Some(8)).is_err());
    }

    #[test]
    fn mock_embedder_is_deterministic_and_dim_correct() {
        let e = MockEmbedder::new(16);
        // embed() returns one vector per input (a batch); take the first.
        let a = e
            .embed(&["hello world"])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let b = e
            .embed(&["hello world"])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(a.len(), 16);
        assert_eq!(a, b);
        let norm = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn content_hash_is_deterministic_hex() {
        let a = content_hash("hello");
        let b = content_hash("hello");
        let c = content_hash("world");
        assert_eq!(a.len(), 64);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// BusyFlag fires the hook once on the idle→busy edge and once on
    /// busy→idle, no matter how many overlapping embeds entered.
    #[test]
    fn busy_flag_fires_only_on_transitions() {
        let flag = Arc::new(BusyFlag::default());
        let events: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        flag.set_hook(Arc::new(move |busy| sink.lock().unwrap().push(busy)));

        assert!(!flag.is_busy());

        let g1 = flag.enter();
        assert!(flag.is_busy());
        let g2 = flag.enter();
        assert!(flag.is_busy());
        drop(g1);
        assert!(flag.is_busy(), "still busy while one guard remains");
        drop(g2);
        assert!(!flag.is_busy());

        assert_eq!(*events.lock().unwrap(), vec![true, false]);
    }
}
