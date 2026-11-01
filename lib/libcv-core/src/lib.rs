pub mod types;
pub use types::*;

pub mod ids {
    pub use super::{CvId, JobId, CriterionId, FavId, PairKey};
}
pub mod events {}

pub type CvId = String;
pub type JobId = String;
pub type CriterionId = String;
pub type FavId = String;
pub type PairKey = (CvId, CvId);
