//! Cascade config type used by ServeState.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CascadeConfig {
    pub waves: u32,
    pub candidates_per_wave: usize,
}

impl CascadeConfig {
    pub fn quick(&self) -> Option<String> {
        None
    }
}