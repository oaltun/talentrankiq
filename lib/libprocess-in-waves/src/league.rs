use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
struct ScoredItem {
    cv_hash: String,
    score: f64,
}

impl PartialEq for ScoredItem {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.cv_hash == other.cv_hash
    }
}
impl Eq for ScoredItem {}
impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap by score, tie-break by cv_hash ascending for determinism
        match self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal) {
            Ordering::Equal => other.cv_hash.cmp(&self.cv_hash),
            o => o,
        }
    }
}

pub struct MaxPriorityQueueByScore {
    heap: BinaryHeap<ScoredItem>,
}

impl MaxPriorityQueueByScore {
    pub fn new() -> Self { Self { heap: BinaryHeap::new() } }
    pub fn push(&mut self, cv_hash: String, score: f64) {
        self.heap.push(ScoredItem { cv_hash, score });
    }
    pub fn pop(&mut self) -> Option<String> {
        self.heap.pop().map(|s| s.cv_hash)
    }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.heap.iter().map(|s| &s.cv_hash)
    }
    pub fn contains(&self, cv_hash: &str) -> bool {
        self.heap.iter().any(|s| s.cv_hash == cv_hash)
    }
    pub fn to_vec(&self) -> Vec<String> {
        self.heap.iter().map(|s| s.cv_hash.clone()).collect()
    }
}

impl Default for MaxPriorityQueueByScore {
    fn default() -> Self { Self::new() }
}
