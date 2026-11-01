use libcv_state::Event;

pub trait Worker {
    type In;
    type Out: Into<Event>;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError>;
}

#[derive(Debug)]
pub enum WorkerError {
    Failed(String),
}

pub struct EffectRunner;

impl EffectRunner {
    pub fn new() -> Self {
        Self
    }
}

pub struct TextWorker;
pub struct ScoutWorker;
pub struct HeavyWorker;
pub struct MatchWorker;
pub struct JudgeWorker;
pub struct Orchestrator;
