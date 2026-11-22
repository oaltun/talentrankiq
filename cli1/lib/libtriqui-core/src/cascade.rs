//! Progressive wave orchestration under a system budget.
//!
//! A cascade ranks candidates in bounded waves so a big import never pins the
//! machine and progress is observable through `rank_state`. Each wave scores a
//! batch of candidates and rewrites the rubric, so the ranking improves wave by
//! wave and a cancel at a wave boundary leaves a valid partial ranking behind.

use std::sync::Arc;

use crate::runtime::EngineRuntime;
use crate::sort::{self, RubricQuestion, WriteBackOutcome};

/// Stage-0.5 scout model, if configured.
pub fn scout_model(_runtime: &crate::runtime::EngineRuntime) -> Option<String> {
    None
}

/// Wave configuration for a run.
#[derive(Debug, Clone)]
pub struct WaveConfig {
    /// How many candidates to score per wave.
    pub candidates_per_wave: usize,
    /// Soft ceiling on waves (0 = unbounded, stop when candidates exhausted).
    pub max_waves: u32,
    /// Wave scale from the system budget (1.0 full, 0.5 polite, 0.25 backoff,
    /// 0.0 hold). A hold (0.0) processes at least one candidate per wave so a
    /// run still makes progress.
    pub wave_scale: f64,
}

impl Default for WaveConfig {
    fn default() -> Self {
        Self {
            candidates_per_wave: 100,
            max_waves: 0,
            wave_scale: 1.0,
        }
    }
}

impl WaveConfig {
    /// Effective batch size after applying the system-budget scale.
    pub fn effective_batch(&self) -> usize {
        let base = self.candidates_per_wave.max(1);
        let scaled = (base as f64 * self.wave_scale).floor() as usize;
        scaled.max(1)
    }
}

/// The result of a cascade run, including coverage info.
#[derive(Debug, Clone, Default)]
pub struct CascadeOutcome {
    pub waves: u32,
    pub ranked: usize,
    pub candidates: usize,
    pub skipped_missing_vector: usize,
    pub rubric_vectors: usize,
    pub cancelled: bool,
}

/// Run the cascade for a job in waves. `is_cancelled` is polled at every wave
/// boundary; `on_wave` reports `(wave_number, candidates_scored_so_far,
/// total_candidates)` for progress emission.
pub fn run(
    runtime: &Arc<EngineRuntime>,
    job_id: &str,
    cfg: &WaveConfig,
    is_cancelled: &dyn Fn() -> bool,
    on_wave: &dyn Fn(u32, usize, usize),
) -> Result<CascadeOutcome, String> {
    // Scoring is a join between candidate centroids and rubric vectors; supply
    // any missing side in the active space before reading either.
    crate::ensure::ensure_inputs(runtime, job_id)?;

    let conn = runtime.open_conn()?;

    let rubric: Vec<RubricQuestion> = sort::rubric_for_job(&conn, job_id)?;
    let model = resolve_model(runtime, &conn, job_id);

    // Candidate ids in a stable order.
    let candidates: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT id FROM candidates
                 WHERE job_id = ?1 AND extraction_status = 'Extracted' AND deleted = 0
                 ORDER BY imported_at, id",
            )
            .map_err(|e| e.to_string())?;
        let mut iter = stmt
            .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
            .map_err(|e| e.to_string())?;
        let mut v = Vec::new();
        while let Some(r) = iter.next().transpose().map_err(|e| e.to_string())? {
            v.push(r);
        }
        v
    };
    let total = candidates.len();

    // Rubric vectors for the resolved space.
    let mut rubric_vectors: Vec<(String, Vec<f32>)> = Vec::new();
    if let Some(model) = &model {
        for (qid, bytes) in crate::db::rubric_vectors(&conn, job_id, model)? {
            rubric_vectors.push((qid, crate::embed::bytes_to_vector(&bytes)));
        }
    }
    let rubric_vectors_count = rubric_vectors.len();

    // Candidate centroids keyed by id, for the resolved space.
    let mut centroids: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    if let Some(model) = &model {
        for (cid, vec_bytes, _hash) in
            crate::db::candidates_with_centroids(&conn, job_id, model, None)?
        {
            centroids.insert(cid, crate::embed::bytes_to_vector(&vec_bytes));
        }
    }

    let batch = cfg.effective_batch();
    let mut scores: std::collections::HashMap<(String, String), f64> =
        std::collections::HashMap::new();
    let mut waves: u32 = 0;
    let mut scored: usize = 0;
    let mut skipped_missing_vector: usize = 0;

    for chunk in candidates.chunks(batch) {
        if is_cancelled() {
            // Write back what we have as a cancelled partial ranking.
            let _ = sort::write_back(&conn, job_id, &rubric, &scores, &std::collections::HashMap::new())?;
            return Ok(CascadeOutcome {
                waves,
                ranked: scored,
                candidates: total,
                skipped_missing_vector,
                rubric_vectors: rubric_vectors_count,
                cancelled: true,
            });
        }

        for cid in chunk {
            if let Some(cvec) = centroids.get(cid) {
                for (qid, cos) in sort::score_candidate(&rubric_vectors, cvec) {
                    scores.insert((cid.clone(), qid), cos);
                }
                scored += 1;
            } else {
                skipped_missing_vector += 1;
            }
        }

        waves += 1;
        // Rewrite the rubric each wave so rank/points improve progressively.
        sort::write_back(&conn, job_id, &rubric, &scores, &std::collections::HashMap::new())?;
        on_wave(waves, scored, total);

        if cfg.max_waves > 0 && waves >= cfg.max_waves {
            break;
        }
    }

    // Ensure a final write-back even for an empty candidate set.
    let _ = if waves == 0 {
        sort::write_back(&conn, job_id, &rubric, &scores, &std::collections::HashMap::new())?
    } else {
        // Get the outcome from the last write_back call
        WriteBackOutcome {
            ranked: scored,
            candidates: total,
            capacity: 0.0,
            rubric_questions: rubric_vectors_count,
            skipped: skipped_missing_vector,
        }
    };

    Ok(CascadeOutcome {
        waves,
        ranked: scored,
        candidates: total,
        skipped_missing_vector,
        rubric_vectors: rubric_vectors_count,
        cancelled: false,
    })
}

fn resolve_model(rt: &EngineRuntime, conn: &rusqlite::Connection, job_id: &str) -> Option<String> {
    if let Ok(e) = rt.embedder() {
        return Some(e.model_id().to_string());
    }
    conn.query_row(
        "SELECT model FROM rubric_vectors WHERE job_id = ?1 LIMIT 1",
        rusqlite::params![job_id],
        |r| r.get::<_, String>(0),
    )
    .ok()
    .or_else(|| {
        conn.query_row(
            "SELECT vector_model FROM candidates WHERE job_id = ?1 AND vector_model IS NOT NULL LIMIT 1",
            rusqlite::params![job_id],
            |r| r.get::<_, String>(0),
        )
        .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_batch_scales_with_wave_scale() {
        let cfg = WaveConfig {
            candidates_per_wave: 100,
            max_waves: 0,
            wave_scale: 0.5,
        };
        assert_eq!(cfg.effective_batch(), 50);
    }

    #[test]
    fn effective_batch_never_zero() {
        let cfg = WaveConfig {
            candidates_per_wave: 100,
            max_waves: 0,
            wave_scale: 0.0,
        };
        assert_eq!(cfg.effective_batch(), 1, "a hold still makes progress");
    }
}