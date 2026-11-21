use std::path::PathBuf;
use std::sync::Arc;

use libembedrank::RetrievalStrategy;
use libtriqui_core::runtime::EngineRuntime;

use libdb_manager::Db;

#[derive(Clone, Debug)]
pub struct JobConfig {
    pub id: String,
    pub title: String,
    pub job_ad_text: String,
    pub job_ad_hash: String,
    pub criteria: Vec<String>,
    pub import_directory_path: Option<PathBuf>,
    pub quick_model: String,
    pub deep_model: String,
    pub comparator_model_path: Option<PathBuf>,
    pub similarity_method: RetrievalStrategy,
}

#[derive(Clone, Debug)]
pub struct ProcessConfig {
    pub wave_size_deep: usize,
    pub wave_size_comparison: usize,
    pub max_waves: Option<usize>,
    pub poll_interval_ms: u64,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            wave_size_deep: 10,
            wave_size_comparison: 5,
            max_waves: None,
            poll_interval_ms: 2000,
        }
    }
}

pub struct ProcessContext {
    pub db: std::sync::Arc<Db>,
    pub watch_dir: PathBuf,
    pub job: JobConfig,
    pub runtime: Arc<EngineRuntime>,
    pub comparator_artifact: Option<libcomparator::ComparatorArtifact>,
}

impl ProcessContext {
    pub fn job_ad_hash(&self) -> String {
        self.job.job_ad_hash.clone()
    }
}
