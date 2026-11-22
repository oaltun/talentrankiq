use std::time::Duration;

pub mod mem {
    pub fn available_memory(_total: u64) -> u64 { 0 }
}

pub struct Inventory;
impl Inventory {
    pub fn global() -> Self { Self }
}

pub struct LoadSnapshot;
pub struct LoadSampler;
impl LoadSampler {
    pub fn new() -> Self { Self }
    pub fn sample(&mut self, _min_interval: Duration) -> LoadSnapshot { LoadSnapshot }
}

pub struct Budget;

pub fn budget<S, E>(_inv: Inventory, _load: &LoadSnapshot, _syscap: &S, _engine: &E) -> Budget { Budget }

pub fn effective_intra_threads<T>(_engine: &T, _b: &Budget) -> usize { 1 }
pub fn effective_inter_threads<T>(_engine: &T, _b: &Budget) -> usize { 1 }
