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
    // Capture the input version this run scores *before* any preparation work,
    // so the completion marker records the version actually scored rather than
    // whatever the column reads when the last wave writes back. A bump landing
    // mid-run therefore leaves `ranked_version` at the captured value and the
    // job stale, which is exactly what drives the follow-up run.
    let scored_inputs_version: i64 = {
        let conn = runtime.open_conn()?;
        conn.query_row(
            "SELECT inputs_version FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .unwrap_or(0)
    };

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

    // Per-chunk rubric embeddings (coverage scoring) for the resolved space.
    // Questions without persisted chunks fall back to the whole-question vector.
    let mut rubric_chunks_by_question: std::collections::HashMap<
        String,
        Vec<crate::chunks::ChunkRow>,
    > = std::collections::HashMap::new();
    if let Some(model) = &model {
        for q in &rubric {
            let rcs = crate::chunks::rubric_chunks_for(&conn, job_id, &q.question_id, model)
                .unwrap_or_default();
            if !rcs.is_empty() {
                rubric_chunks_by_question.insert(q.question_id.clone(), rcs);
            }
        }
    }

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

    // Extracted text per candidate, used as the explanation when a candidate
    // has no persisted chunks and falls back to the centroid path.
    let extracted: std::collections::HashMap<String, String> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, COALESCE(extracted_text, '') FROM candidates
                 WHERE job_id = ?1 AND extraction_status = 'Extracted' AND deleted = 0",
            )
            .map_err(|e| e.to_string())?;
        let mut iter = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        let mut v = std::collections::HashMap::new();
        while let Some(r) = iter.next().transpose().map_err(|e| e.to_string())? {
            v.insert(r.0, r.1);
        }
        v
    };

    let batch = cfg.effective_batch();
    let mut scores: std::collections::HashMap<(String, String), f64> =
        std::collections::HashMap::new();
    let mut chunk_texts: std::collections::HashMap<(String, String), String> =
        std::collections::HashMap::new();
    let mut rubric_chunk_map: std::collections::HashMap<
        (String, String),
        (String, usize, usize),
    > = std::collections::HashMap::new();
    let mut waves: u32 = 0;
    let mut scored: usize = 0;
    let mut skipped_missing_vector: usize = 0;

    for chunk in candidates.chunks(batch) {
        if is_cancelled() {
            // Write back what we have as a cancelled partial ranking.
            let _ = sort::write_back(
                &conn,
                job_id,
                &rubric,
                &scores,
                &chunk_texts,
                &rubric_chunk_map,
                scored_inputs_version,
            )?;
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
            let chunks: Vec<(String, Vec<f32>)> = match &model {
                Some(model) => crate::chunks::candidate_chunks_for(&conn, cid, model)?
                    .into_iter()
                    .map(|c| (c.chunk_text, c.vector))
                    .collect(),
                None => Vec::new(),
            };
            let centroid = centroids.get(cid).map(|v| v.as_slice());
            if chunks.is_empty() && centroid.is_none() {
                skipped_missing_vector += 1;
                continue;
            }
            let whole = extracted.get(cid).cloned().unwrap_or_default();
            let scored_candidate = sort::score_candidate_against_rubric(
                &rubric_vectors,
                &rubric_chunks_by_question,
                &chunks,
                centroid,
                &whole,
            );
            for (qid, s) in scored_candidate.scores {
                scores.insert((cid.clone(), qid), s);
            }
            for (qid, t) in scored_candidate.chunk_texts {
                chunk_texts.insert((cid.clone(), qid), t);
            }
            for (qid, r) in scored_candidate.rubric_chunks {
                rubric_chunk_map.insert((cid.clone(), qid), r);
            }
            scored += 1;
        }

        waves += 1;
        // Rewrite the rubric each wave so rank/points improve progressively.
        sort::write_back(
            &conn,
            job_id,
            &rubric,
            &scores,
            &chunk_texts,
            &rubric_chunk_map,
            scored_inputs_version,
        )?;
        on_wave(waves, scored, total);

        if cfg.max_waves > 0 && waves >= cfg.max_waves {
            break;
        }
    }

    // Ensure a final write-back even for an empty candidate set.
    let _ = if waves == 0 {
        sort::write_back(
            &conn,
            job_id,
            &rubric,
            &scores,
            &chunk_texts,
            &rubric_chunk_map,
            scored_inputs_version,
        )?
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

    // 1.2 — a candidate without any centroid or chunks is counted in
    // skipped_missing_vector, not in ranked.
    #[test]
    fn candidate_without_centroid_is_skipped_not_ranked() {
        use crate::db::open_single_connection_at;
        use crate::runtime::{EngineRuntime, RuntimeConfig};

        let mut path = std::env::temp_dir();
        path.push(format!("triq-cascade-skip-{}.db", uuid::Uuid::new_v4()));
        let _ = std::fs::remove_file(&path);
        let conn = open_single_connection_at(&path).expect("open db");

        conn.execute("INSERT INTO jobs (id, title, job_ad_text, ad_weight) VALUES ('job1', 'j', 'ad', 10.0)", [])
            .unwrap();
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, extracted_text)
                VALUES ('c1', 'job1', 'p/c1.txt', 'c1.txt', 'txt', 'Extracted', 'candidate one text')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, extracted_text)
                VALUES ('c2', 'job1', 'p/c2.txt', 'c2.txt', 'txt', 'Extracted', 'candidate two text')",
            [],
        )
        .unwrap();

        // No rubric vectors, no candidate centroids → everything is skipped.
        // No model loaded; resolve_model falls back to stored vectors which are absent.
        let runtime = std::sync::Arc::new(EngineRuntime::new(
            RuntimeConfig::discover(None, None),
            path.clone(),
        ));
        let cfg = WaveConfig::default();
        let cancelled = || false;
        let on_wave = |_w: u32, _d: usize, _t: usize| {};

        let outcome = crate::cascade::run(
            &runtime,
            "job1",
            &cfg,
            &cancelled,
            &on_wave,
        )
        .expect("cascade run");

        assert_eq!(outcome.ranked, 0, "no candidate had a centroid or chunks → none ranked");
        assert_eq!(outcome.skipped_missing_vector, 2, "both candidates counted as skipped_missing_vector");
        assert_eq!(outcome.candidates, 2);
        assert!(!outcome.cancelled);
    }
}