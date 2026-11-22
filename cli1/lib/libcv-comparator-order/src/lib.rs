use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlickoState {
    pub rating: f64,
    pub rd: f64,
}

#[derive(Debug, Clone)]
pub struct ComparatorOrderState {
    pub glicko: std::collections::HashMap<(String, u32), GlickoState>,
    pub comparisons: HashSet<(String, String)>,
    pub provisional_sim_rank: Vec<(String, f64)>,
    pub job_version: u32,
    pub entitled: bool,
}

pub fn next_pair(_order_state: &ComparatorOrderState) -> Option<(String, String)> {
    None
}
