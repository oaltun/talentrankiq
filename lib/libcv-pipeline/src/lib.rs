use std::collections::HashSet;
use std::env;

use libcv_state::{Event, State};

pub trait Worker {
    type In;
    type Out: Into<Event>;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError>;
}

#[derive(Debug)]
pub enum WorkerError {
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkItem {
    TextExtract { cv_id: String },
    Scout { cv_id: String },
    Heavy { cv_id: String },
    Match { cv_id: String, job_id: String },
    Judge { pair: (String, String) },
}

pub struct EffectRunner {
    workers: usize,
}

impl EffectRunner {
    pub fn new() -> Self {
        Self {
            workers: env::var("CV_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1),
        }
    }

    pub fn with_workers(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
        }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Scheduling decisions derived from post-reducer state and the triggering event.
    pub fn schedule(&self, state: &State, event: &Event) -> Vec<WorkItem> {
        self.effects(state, event)
    }

    pub fn effects(&self, state: &State, event: &Event) -> Vec<WorkItem> {
        let mut out = Vec::new();
        match event {
            Event::CvAdded { cv_id, .. } => {
                out.push(WorkItem::TextExtract { cv_id: cv_id.clone() });
            }
            Event::TextExtracted { cv_id, .. } => {
                out.push(WorkItem::Scout { cv_id: cv_id.clone() });
                out.push(WorkItem::Heavy { cv_id: cv_id.clone() });
            }
            Event::HeavyCompleted { cv_id, .. } => {
                // Match work is enqueued when a job ad version is known.
                for (job_id, job) in &state.jobs {
                    // Only if this CV belongs to the job.
                    if let Some(cv) = state.cvs.get(cv_id) {
                        if cv.job_id == *job_id {
                            out.push(WorkItem::Match {
                                cv_id: cv_id.clone(),
                                job_id: job_id.clone(),
                            });
                        }
                    }
                    let _ = job;
                }
            }
            Event::JobAdUpdated { job_id, .. } => {
                for cv in state.cvs.values() {
                    if cv.job_id == *job_id && cv.heavy_vector.is_some() {
                        out.push(WorkItem::Match {
                            cv_id: cv.id.clone(),
                            job_id: job_id.clone(),
                        });
                    }
                }
            }
            Event::HeavyMatched { .. } => {
                // Ask judge orderer for next pair (handled by caller with state).
            }
            Event::ComparisonCompleted { .. } => {}
            _ => {}
        }
        // Judge scheduling: after HeavyMatched accumulation, consult next_pair.
        // Delegated to caller; here we leave judge work to explicit enqueue.
        out
    }

    /// Decide judge work given current state (call after HeavyMatched/ComparisonCompleted).
    pub fn next_judge_work(&self, state: &State) -> Option<WorkItem> {
        use libcv_judge_order::{JudgeOrderState, next_pair};
        let mut provisional: Vec<(String, f64)> = state
            .cvs
            .values()
            .filter(|c| !c.deleted)
            .map(|c| (c.id.clone(), c.heavy_similarity.unwrap_or(0.0)))
            .collect();
        provisional.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let comparisons: HashSet<(String, String)> = state
            .comparisons
            .iter()
            .map(|(a, b, _, _)| {
                if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                }
            })
            .collect();
        let job_version = state.jobs.values().next().map(|j| j.version).unwrap_or(0);
        let order_state = JudgeOrderState {
            glicko: state
                .glicko
                .iter()
                .map(|((cv, ver), g)| {
                    (
                        (cv.clone(), *ver),
                        libjudge::tournament::GlickoState { rating: g.rating, rd: g.rd },
                    )
                })
                .collect(),
            comparisons,
            provisional_sim_rank: provisional,
            job_version,
        };
        next_pair(&order_state).map(|p| WorkItem::Judge { pair: p })
    }
}

pub struct TextWorker;
pub struct ScoutWorker;
pub struct HeavyWorker;
pub struct MatchWorker;
pub struct JudgeWorker;

impl Worker for TextWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        Ok(Event::TextExtracted {
            cv_id: item,
            text: String::new(),
            status: libcv_state::ExtractionStatus::Extracted,
        })
    }
}

impl Worker for ScoutWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        Ok(Event::ScoutCompleted {
            cv_id: item,
            vector: Vec::new(),
            model_bundle_version: 0,
        })
    }
}

impl Worker for HeavyWorker {
    type In = String;
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        Ok(Event::HeavyCompleted {
            cv_id: item,
            vector: Vec::new(),
            model_bundle_version: 0,
        })
    }
}

impl Worker for MatchWorker {
    type In = (String, String);
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        Ok(Event::HeavyMatched {
            cv_id: item.0,
            job_version: 0,
            similarity: 0.0,
        })
    }
}

impl Worker for JudgeWorker {
    type In = (String, String);
    type Out = Event;
    fn process(&mut self, item: Self::In) -> Result<Self::Out, WorkerError> {
        Ok(Event::ComparisonCompleted {
            pair: item,
            job_version: 0,
            model_bundle_version: 0,
            result: 0.5,
        })
    }
}

pub struct Orchestrator {
    workers: usize,
}

impl Orchestrator {
    pub fn new() -> Self {
        Self {
            workers: env::var("CV_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1),
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EffectRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_runner_default_one_worker() {
        let r = EffectRunner::new();
        assert_eq!(r.workers(), 1);
    }

    #[test]
    fn cv_added_schedules_text_extract() {
        let mut s = State::default();
        let e = Event::CvAdded {
            cv_id: "c1".into(),
            job_id: "j1".into(),
            file_path: "/tmp/c1.pdf".into(),
            file_name: "c1.pdf".into(),
            file_type: "Pdf".into(),
        };
        s = libcv_state::reduce(s, &e);
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert!(work.iter().any(|w| matches!(w, WorkItem::TextExtract { cv_id } if cv_id == "c1")));
    }

    #[test]
    fn text_extracted_schedules_scout_and_heavy() {
        let e = Event::TextExtracted {
            cv_id: "c1".into(),
            text: "hi".into(),
            status: libcv_state::ExtractionStatus::Extracted,
        };
        let s = State::default();
        let r = EffectRunner::new();
        let work = r.schedule(&s, &e);
        assert!(work.iter().any(|w| matches!(w, WorkItem::Scout { .. })));
        assert!(work.iter().any(|w| matches!(w, WorkItem::Heavy { .. })));
    }

    #[test]
    fn orchestrator_reads_env() {
        std::env::set_var("CV_PIPELINE_WORKERS", "4");
        let o = Orchestrator::new();
        assert_eq!(o.worker_count(), 4);
        std::env::remove_var("CV_PIPELINE_WORKERS");
    }
}
