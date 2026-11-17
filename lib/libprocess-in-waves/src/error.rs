use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("DB error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("embed rank error: {0}")]
    EmbedRank(#[from] libembedrank::EmbedRankError),
    #[error("comparator error: {0}")]
    Comparator(#[from] libcomparator::ComparatorError),
    #[error("model caller error: {0}")]
    ModelCaller(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CV parse error: {0}")]
    CvParse(#[from] libcvparser::CvParseError),
    #[error("config error: {0}")]
    Config(String),
    #[error("domain error: {0}")]
    Domain(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl From<libtriqui_core::types::DomainError> for ProcessError {
    fn from(e: libtriqui_core::types::DomainError) -> Self {
        ProcessError::Domain(e.to_string())
    }
}

impl From<libmodelcaller::ModelError> for ProcessError {
    fn from(e: libmodelcaller::ModelError) -> Self {
        ProcessError::ModelCaller(e.to_string())
    }
}
