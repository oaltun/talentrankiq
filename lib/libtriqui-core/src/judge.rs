//! Gui judge stage (design: thin-gui "Job fit (judge)" rubric row).
//!
//! Replaces the rubric's "Job ad" cosine row with a win-rate from a trained
//! pairwise judge artifact (`libjudge`, design 15). Uses only vectors the gui
//! already stores — no modelcaller calls at rank time:
//! - the job ad's fresh embedding becomes the doc's `a` block;
//! - each candidate's stored chunk embeddings (or its centroid vector when no
//!   chunks exist) become `e` (centroid) and `c` (best chunk) via
//!   [`libjudge::features::doc_features`] — the same structure the artifact
//!   was trained on.
//!
//! The stage is active only when the artifact loads, is pairwise, is not
//! fused (no LLM evidence extraction in the gui), and its feature spec's
//! embedder + dim match the currently selected gui embedder. Otherwise the
//! caller falls back to the legacy cosine ad row.

use std::path::Path;

use libjudge::artifact::{FeatureSpec, FeatureVariant, JudgeArtifact, JudgeKind};
use libjudge::features::{doc_features, DocFeatures};
use libjudge::forward::JudgeModel;
use libjudge::tournament::{pairwise_prob_matrix, win_rate_scores};

use crate::embed::content_hash;
use crate::judge_cache::{self, PairVerdictStats};

/// Loaded pairwise judge + the feature spec it was trained with.
pub struct JudgeStage {
    spec: FeatureSpec,
    model: JudgeModel,
    /// SHA-256 of the artifact's canonical JSON — memoization key component:
    /// verdicts are facts about *this* artifact, a retrained one invalidates.
    content_hash: String,
}

/// Length floors from the training distribution (cnam train): resumes never
/// fall below ~900 chars (min 897); ads go down to ~72. Inputs below the
/// resume floor are out-of-distribution for the judge head — it can return
/// confidently wrong verdicts on short/synthetic CVs (verified: the artifact
/// prefers an obvious non-fit over a match on 2-sentence CVs while ad cosine
/// ranks correctly). Jobs below the floors use the cosine ad row instead.
pub const RESUME_MIN_CHARS: usize = 900;
pub const AD_MIN_CHARS: usize = 70;

/// Whether the judge stage may score this job: the ad must exist and be at
/// least `AD_MIN_CHARS`, and every candidate's extracted text must be at
/// least `RESUME_MIN_CHARS` (`None` = no candidates yet → not eligible).
pub fn job_eligible(ad_text: Option<&str>, min_resume_chars: Option<usize>) -> bool {
    let ad_ok = ad_text.map(|t| t.trim().chars().count()).unwrap_or(0) >= AD_MIN_CHARS;
    let resumes_ok = min_resume_chars
        .map(|m| m >= RESUME_MIN_CHARS)
        .unwrap_or(false);
    ad_ok && resumes_ok
}

impl JudgeStage {
    /// Wrap a validated pairwise artifact (see [`JudgeArtifact::load`]).
    pub fn from_artifact(artifact: JudgeArtifact) -> Result<Self, String> {
        if artifact.kind != JudgeKind::Pairwise {
            return Err(format!(
                "artifact is a {:?} model, the gui judge stage needs pairwise",
                artifact.kind
            ));
        }
        if artifact.feature_spec.variant == FeatureVariant::Fused {
            return Err(
                "fused artifacts need LLM evidence extraction, which the gui does not run".into(),
            );
        }
        let json = artifact.to_json().map_err(|e| e.to_string())?;
        Ok(Self {
            spec: artifact.feature_spec.clone(),
            model: JudgeModel::from_artifact(&artifact).map_err(|e| e.to_string())?,
            content_hash: content_hash(&json),
        })
    }

    /// Load + validate + burn-parity-check an artifact from disk.
    pub fn load(path: &Path) -> Result<Self, String> {
        let artifact = JudgeArtifact::load(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_artifact(artifact)
    }

    /// The artifact only ranks vectors from the embedder it was trained on.
    pub fn matches(&self, embedder: &str, dim: usize) -> bool {
        self.spec.embedder == embedder && self.spec.dim == dim
    }

    /// The artifact's training embedder slug (for fallback diagnostics).
    pub fn spec_embedder(&self) -> &str {
        &self.spec.embedder
    }

    /// The artifact's training embedding dimension (for fallback diagnostics).
    pub fn spec_dim(&self) -> usize {
        self.spec.dim
    }

    /// Rubric row label replacing "Job ad" when the stage is active.
    pub const AD_LABEL: &str = "Job fit (judge)";

    /// SHA-256 of the artifact's canonical JSON (memoization key component).
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    /// Win-rate `s_i ∈ [0,1]` per candidate, memoized per pair verdict.
    ///
    /// Same result as [`Self::win_rates`] (bit-identical: fresh pairs run the
    /// same forward passes, cached pairs round-trip f64 exactly), but every
    /// canonical pair `p(a > b)` is looked up in — and stored to — the
    /// `judge_pair_verdicts` table keyed by `(embedder model, artifact hash,
    /// ad hash, doc_a, doc_b)`. A re-rank whose judge inputs (ad text, CV
    /// texts, model, artifact) are unchanged runs zero forward passes; a new
    /// candidate only pays for its own pairs.
    ///
    /// `doc_keys[i]` is the content hash of candidate i's extracted text (the
    /// same lowercase-preprocessed hashing the embedding cache uses, so keys
    /// track exactly the text the vectors were built from). Returns the
    /// win-rates aligned with `cv_chunks` plus what the pass did.
    ///
    /// `rt` (the cascade passes `Some`) only feeds the status line: the judge
    /// never touches the engine, but the gui wants "judging a.pdf vs b.pdf
    /// pair 210/1120" while tiles run.
    pub fn win_rates_cached(
        &self,
        conn: &rusqlite::Connection,
        rt: Option<&std::sync::Arc<crate::runtime::EngineRuntime>>,
        model: &str,
        ad_hash: &str,
        doc_keys: &[String],
        ead: &[f32],
        cv_chunks: &[&[Vec<f64>]],
    ) -> Result<(Vec<f64>, PairVerdictStats), String> {
        if doc_keys.len() != cv_chunks.len() {
            return Err(format!(
                "doc_keys ({}) and cv_chunks ({}) length mismatch",
                doc_keys.len(),
                cv_chunks.len()
            ));
        }
        let artifact_hash = self.content_hash().to_string();
        let ad_chunks = vec![ead.iter().map(|&x| x as f64).collect::<Vec<f64>>()];
        let ad = doc_features(&ad_chunks, &ad_chunks).map_err(|e| e.to_string())?;
        let mut docs: Vec<DocFeatures> = Vec::with_capacity(cv_chunks.len());
        for chunks in cv_chunks {
            docs.push(doc_features(&ad_chunks, chunks).map_err(|e| e.to_string())?);
        }

        let n = docs.len();
        let mut p = vec![vec![0.5f64; n]; n];
        let mut stats = PairVerdictStats::default();
        // Collect, then batch, then store (bench design 15.13). The cache
        // probe stays a cheap serial lookup; the FRESH pairs — the only ones
        // that need a forward pass — go through one row-parallel batched
        // pass per tile instead of one serial pass per pair.
        let mut fresh: Vec<(usize, usize, bool)> = Vec::new(); // (i, j, i is canonical)
        for i in 0..n {
            for j in (i + 1)..n {
                // Canonical (sorted) key order; the stored prob belongs to the
                // canonical orientation, so mirror it when needed.
                let (key_a, key_b) = if doc_keys[i] <= doc_keys[j] {
                    (&doc_keys[i], &doc_keys[j])
                } else {
                    (&doc_keys[j], &doc_keys[i])
                };
                let i_canonical = key_a == &doc_keys[i];
                let hit =
                    judge_cache::lookup_pair(conn, model, &artifact_hash, ad_hash, key_a, key_b)
                        .map_err(|e| e.to_string())?;
                match hit {
                    Some(v) => {
                        stats.cached += 1;
                        let pij = if i_canonical { v } else { 1.0 - v };
                        p[i][j] = pij;
                        p[j][i] = 1.0 - pij;
                    }
                    None => {
                        stats.fresh += 1;
                        fresh.push((i, j, i_canonical));
                    }
                }
            }
        }

        let per_tile = libjudge::tournament::pairs_per_tile(self.spec.input_dim);
        // Content hash → file name, for the status line only ("judging a.pdf
        // vs b.pdf"). Hashes are content-derived, so the same text under two
        // names resolves to one of them — fine for a progress subject.
        let names: std::collections::HashMap<String, String> = match rt {
            Some(_) if !fresh.is_empty() => {
                let mut stmt = conn
                    .prepare("SELECT content_hash, file_name FROM candidates WHERE content_hash IS NOT NULL")
                    .map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                    .map_err(|e| e.to_string())?;
                let mut m = std::collections::HashMap::new();
                for row in rows {
                    let (h, f) = row.map_err(|e| e.to_string())?;
                    m.entry(h).or_insert(f);
                }
                m
            }
            _ => Default::default(),
        };
        let name_of = |i: usize| -> String {
            doc_keys
                .get(i)
                .and_then(|k| names.get(k))
                .cloned()
                .filter(|f| !f.is_empty())
                .unwrap_or_else(|| format!("candidate {i}"))
        };
        let total_pairs = fresh.len();
        let mut judged_pairs = 0usize;
        for chunk in fresh.chunks(per_tile) {
            let (si, sj, _) = chunk[0];
            let subject = format!("{} vs {}", name_of(si), name_of(sj));
            if let Some(rt) = rt {
                rt.set_activity(
                    "judge",
                    &subject,
                    Some(format!("pair {}/{}", judged_pairs + 1, total_pairs)),
                    Some(model),
                );
            }
            let pairs: Vec<(usize, usize)> = chunk.iter().map(|&(i, j, _)| (i, j)).collect();
            let vals = libjudge::tournament::averaged_pair_probs_batch(
                &self.model,
                &self.spec,
                &ad,
                &pairs,
                &docs,
            );
            judged_pairs += chunk.len();
            if let Some(rt) = rt {
                rt.set_activity(
                    "judge",
                    &subject,
                    Some(format!("pair {}/{}", judged_pairs, total_pairs)),
                    Some(model),
                );
            }
            for (k, &(i, j, i_canonical)) in chunk.iter().enumerate() {
                // `v` = p(i > j) for this loop's orientation; the table stores
                // it under canonical keys, so re-orient.
                let v = vals[k];
                let (key_a, key_b) = if i_canonical {
                    (&doc_keys[i], &doc_keys[j])
                } else {
                    (&doc_keys[j], &doc_keys[i])
                };
                let canonical = if i_canonical { v } else { 1.0 - v };
                judge_cache::store_pair(
                    conn,
                    model,
                    &artifact_hash,
                    ad_hash,
                    key_a,
                    key_b,
                    canonical,
                )
                .map_err(|e| e.to_string())?;
                p[i][j] = v;
                p[j][i] = 1.0 - v;
            }
        }
        if let Some(rt) = rt {
            rt.clear_activity();
        }
        Ok((win_rate_scores(&p), stats))
    }

    /// Win-rate `s_i ∈ [0,1]` per candidate (0.5 = indifferent), aligned with
    /// `cv_chunks`. `ead` is the job ad's (normalized) embedding; each
    /// `cv_chunks[i]` is that candidate's chunk embeddings in the same space
    /// (or a single centroid vector when no chunks are stored).
    pub fn win_rates(&self, ead: &[f32], cv_chunks: &[&[Vec<f64>]]) -> Vec<f64> {
        let ad_chunks = vec![ead.iter().map(|&x| x as f64).collect::<Vec<f64>>()];
        let ad = match doc_features(&ad_chunks, &ad_chunks) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let mut docs: Vec<DocFeatures> = Vec::with_capacity(cv_chunks.len());
        for chunks in cv_chunks {
            match doc_features(&ad_chunks, chunks) {
                Ok(d) => docs.push(d),
                // a malformed candidate drops the whole tournament (it is
                // set-relative; scoring a subset would shift everyone else)
                Err(_) => return Vec::new(),
            }
        }
        let p = pairwise_prob_matrix(&self.model, &self.spec, &ad, &docs);
        win_rate_scores(&p)
    }
}

/// The judge stage usable for the runtime's current embedder, or `None` when no
/// artifact is configured, it fails to load, or it does not match.
pub fn stage_for(rt: &std::sync::Arc<crate::runtime::EngineRuntime>) -> Option<JudgeStage> {
    let model = rt.model();
    let path = rt.judge_artifact_for(&model)?;
    match JudgeStage::load(&path) {
        Ok(stage) => Some(stage),
        Err(e) => {
            log::warn!("judge: artifact {path:?} failed to load, stage disabled: {e}");
            None
        }
    }
}

/// One pass of the judge over a job's frozen pool.
///
/// The pool is read from `judge_pool` — never re-picked — because a win-rate is
/// set-relative: with a pool that changed every wave, a CV's grade would move for
/// reasons that have nothing to do with the CV. Members the heavy embedder has not
/// reached are skipped (there are no vectors to compare), and members that already
/// have a grade are left alone, so a wave's judge pass costs nothing once the
/// tournament is memoized.
///
/// The result is one `candidate_scores` row per member (`question_id =
/// 'ad_judge'`), which is the judge's only write in the system: rescore reads those
/// rows as the ad slot's value and never touches them.
pub fn run_pool(
    rt: &std::sync::Arc<crate::runtime::EngineRuntime>,
    job_id: &str,
) -> Result<Option<crate::judge_cache::PairVerdictStats>, String> {
    let model = rt.model();
    let Some(stage) = stage_for(rt) else {
        return Ok(None);
    };
    let conn = rt.open_conn().map_err(|e| e.to_string())?;

    let pool = crate::db::judge_pool_ids(&conn, job_id)?;
    if pool.is_empty() {
        // A pool of nobody is a configuration, not a failure: `pool_size: 0`
        // turns the judge off and reads as "did nothing" — never as "no limit",
        // which is what a `0` budget used to mean here.
        return Ok(None);
    }

    // Every member goes through the tournament, graded or not: `win_rates_cached`
    // memoizes per pair, so an unchanged pool costs one indexed lookup per pair and
    // no forward pass — and it keeps `judge_pairs_cached` a truthful count of *pair
    // verdicts* rather than of members. Short-circuiting the already-graded members
    // would report `cached: 0` for a pool whose verdicts are all in the cache, which
    // reads as "the judge did nothing" instead of "it ran and found nothing new".
    let mut stats = crate::judge_cache::PairVerdictStats::default();
    let pending: Vec<&String> = pool.iter().collect();

    // The ad, in the heavy space. Its dimension is also the artifact match check:
    // a judge trained for another embedder's dimension must not score here.
    let ad_hash = {
        let ad_text: Option<String> = conn
            .query_row(
                "SELECT job_ad_text FROM jobs WHERE id = ?1",
                rusqlite::params![job_id],
                |r| r.get(0),
            )
            .ok()
            .flatten();
        if !job_eligible(ad_text.as_deref(), min_resume_chars(&conn, job_id)?) {
            log::info!(
                "judge: pool of {} not eligible (ad >= {AD_MIN_CHARS} chars, every CV >= {RESUME_MIN_CHARS}) — using the embedder's cosine",
                pending.len()
            );
            return Ok(None);
        }
        crate::embed::content_hash(ad_text.as_deref().unwrap_or_default())
    };
    let ead: Vec<f32> = match rubric_ad_vector(&conn, job_id, &model)? {
        Some(v) => v,
        None => return Ok(None),
    };
    if !stage.matches(&model, ead.len()) {
        log::info!(
            "judge: artifact trained for embedder '{}' (dim {}), selected is '{model}' (dim {}) — using the cosine ad row",
            stage.spec_embedder(),
            stage.spec_dim(),
            ead.len()
        );
        return Ok(None);
    }

    // Documents: the candidate's chunks where it has them, its centroid otherwise.
    // `ids` stays parallel to `doc_keys` so a grade can be written back to the CV
    // it belongs to.
    let mut ids: Vec<String> = Vec::with_capacity(pending.len());
    let mut doc_keys: Vec<String> = Vec::with_capacity(pending.len());
    let mut chunk_refs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(pending.len());
    for cid in &pending {
        let key: Option<String> = conn
            .query_row(
                "SELECT COALESCE(content_hash, id) FROM candidates WHERE id = ?1",
                rusqlite::params![cid.as_str()],
                |r| r.get(0),
            )
            .ok();
        let Some(key) = key else { continue };
        let chunks = crate::rubric::centroid_in_space(&conn, cid, &model)
            .map_err(|e| e.to_string())?
            .map(|c| vec![c])
            .unwrap_or_default();
        if chunks.is_empty() {
            continue;
        }
        ids.push((*cid).clone());
        doc_keys.push(key);
        chunk_refs.push(chunks);
    }
    if ids.len() < 2 {
        // A single candidate cannot be compared with anything. The neutral 0.5 is
        // what the tournament would produce anyway, so store it rather than
        // leaving the pool member looking ungraded.
        if ids.len() == 1 {
            write_grades(&conn, job_id, &ids, &[0.5])?;
        }
        return Ok(Some(stats));
    }

    let refs: Vec<&[Vec<f64>]> = chunk_refs.iter().map(|v| v.as_slice()).collect();
    let (wins, pass_stats) =
        stage.win_rates_cached(&conn, Some(rt), &model, &ad_hash, &doc_keys, &ead, &refs)?;
    stats.fresh += pass_stats.fresh;
    stats.cached += pass_stats.cached;
    if wins.len() != ids.len() {
        return Err(format!(
            "judge: win-rates cover {} of {} pool members",
            wins.len(),
            ids.len()
        ));
    }
    write_grades(&conn, job_id, &ids, &wins)?;
    Ok(Some(stats))
}


/// The shortest extracted CV in the job — the judge's resume-length floor.
fn min_resume_chars(conn: &rusqlite::Connection, job_id: &str) -> Result<Option<usize>, String> {
    conn.query_row(
        "SELECT MIN(length(extracted_text)) FROM candidates
         WHERE job_id = ?1 AND extracted_text IS NOT NULL",
        rusqlite::params![job_id],
        |r| r.get::<_, Option<i64>>(0).map(|v| v.map(|n| n as usize)),
    )
    .map_err(|e| e.to_string())
}

/// The job ad's vector in the heavy space, from `rubric_vectors`.
fn rubric_ad_vector(
    conn: &rusqlite::Connection,
    job_id: &str,
    model: &str,
) -> Result<Option<Vec<f32>>, String> {
    let blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT vector FROM rubric_vectors
             WHERE job_id = ?1 AND question_id = 'ad' AND model = ?2",
            rusqlite::params![job_id, model],
            |r| r.get(0),
        )
        .ok();
    match blob {
        Some(b) => crate::embed::bytes_to_vector(&b, None)
            .map(Some)
            .map_err(|e| e.to_string()),
        None => Ok(None),
    }
}

/// Write one `ad_judge` row per pool member, in one transaction.
///
/// `points / max_points` is the win-rate itself, so rescore can read the grade
/// straight off the row without a second column to keep in sync.
fn write_grades(
    conn: &rusqlite::Connection,
    job_id: &str,
    ids: &[String],
    wins: &[f64],
) -> Result<(), String> {
    if ids.is_empty() {
        return Ok(());
    }
    let ad_weight: f64 = conn
        .query_row(
            "SELECT COALESCE(ad_weight, 1.0) FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    for (i, cid) in ids.iter().enumerate() {
        tx.execute(
            "DELETE FROM candidate_scores WHERE candidate_id = ?1 AND question_id = ?2",
            rusqlite::params![cid, crate::rescore::JUDGE_Q],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO candidate_scores
               (id, candidate_id, question_id, kind, label, points, max_points, sign)
             VALUES (?1, ?2, ?3, 'AdJudge', ?4, ?5, ?6, 'Positive')",
            rusqlite::params![
                format!("cs_{}", uuid::Uuid::new_v4()),
                cid,
                crate::rescore::JUDGE_Q,
                JudgeStage::AD_LABEL,
                ad_weight * wins[i],
                ad_weight
            ],
        )
        .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use libjudge::artifact::{Architecture, ArtifactMeta, ChunkingMode, Contextual, LayerWeights};

    /// Single-layer linear pairwise artifact over a raw 5·4 pair input:
    /// logit = a0 + e1_0 − e2_0 (blocks: [a; e1; e2; c1; c2]).
    fn linear_stage() -> JudgeStage {
        let spec = FeatureSpec::raw_pair(
            "fake",
            4,
            ChunkingMode::None,
            None,
            Contextual::Off,
            libjudge::features::TextPrep::Lowercase,
        );
        let mut w = vec![0.0f32; spec.input_dim];
        w[0] = 1.0; // a
        w[4] = 1.0; // e1
        w[8] = -1.0; // e2
        let artifact = JudgeArtifact {
            spec_version: libjudge::artifact::SPEC_VERSION,
            kind: JudgeKind::Pairwise,
            architecture: Architecture::Linear,
            feature_spec: spec,
            layers: vec![LayerWeights {
                w: vec![w],
                b: vec![0.0],
            }],
            meta: ArtifactMeta {
                trained_at: "test".into(),
                git_rev: "test".into(),
                git_branch: "test".into(),
                seed: 0,
                shuffle_seed: None,
                epochs_requested: 1,
                epochs_trained: 1,
                best_epoch: 1,
                early_stopped: false,
                lr: 1e-3,
                weight_decay: 0.0,
                batch_size: 8,
                train_pairs: 0,
                val_pairs: 0,
                jobs_train: 0,
                jobs_val: 0,
                val_pairwise_accuracy: 0.0,
                val_cycle_rate: None,
                val_class_accuracy: None,
                dropout: 0.0,
                block_mass: vec![],
                zero_out: None,
                extraction_cache_hits: 0,
                extraction_cache_misses: 0,
                notes: String::new(),
            },
            parity: vec![],
        };
        JudgeStage::from_artifact(artifact).expect("linear artifact is valid")
    }

    fn unit(v: &[f32]) -> Vec<f64> {
        let mut v: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        libjudge::l2_normalize(&mut v);
        v
    }

    #[test]
    fn job_eligibility_needs_ad_and_long_resumes() {
        let ad = "a valid job ad that is definitely long enough for the judge stage to engage";
        assert!(ad.chars().count() >= AD_MIN_CHARS);
        assert!(job_eligible(Some(ad), Some(900)));
        assert!(!job_eligible(Some("short ad"), Some(5000)));
        assert!(!job_eligible(None, Some(5000)));
        assert!(!job_eligible(Some(ad), Some(899)));
        assert!(!job_eligible(Some(ad), None));
    }

    #[test]
    fn aligned_cv_beats_orthogonal() {
        let stage = linear_stage();
        assert!(stage.matches("fake", 4));
        assert!(!stage.matches("fake", 7));
        assert!(!stage.matches("other", 4));

        let ead = [1.0, 0.0, 0.0, 0.0];
        let good = vec![unit(&[1.0, 0.0, 0.0, 0.0])];
        let bad = vec![unit(&[0.0, 1.0, 0.0, 0.0])];
        let s = stage.win_rates(&ead, &[&good, &bad]);
        assert_eq!(s.len(), 2);
        assert!(s[0] > s[1], "aligned {s:?} must beat orthogonal");
        for v in &s {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn identical_candidates_are_indifferent() {
        let stage = linear_stage();
        let ead = [1.0, 0.0, 0.0, 0.0];
        let a = vec![unit(&[1.0, 0.0, 0.0, 0.0])];
        let s = stage.win_rates(&ead, &[&a, &a]);
        assert!(s.iter().all(|v| (v - 0.5).abs() < 1e-9), "{s:?}");
    }

    #[test]
    fn single_candidate_is_neutral() {
        let stage = linear_stage();
        let ead = [1.0, 0.0, 0.0, 0.0];
        let a = vec![unit(&[1.0, 0.0, 0.0, 0.0])];
        let s = stage.win_rates(&ead, &[&a]);
        assert!(s.len() == 1 && (s[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn multiple_chunks_use_centroid_and_best_chunk() {
        let stage = linear_stage();
        let ead = [1.0, 0.0, 0.0, 0.0];
        // chunks average to an aligned doc even though none is aligned alone
        let mixed = vec![
            unit(&[0.0, 1.0, 0.0, 0.0]),
            unit(&[1.0, 0.0, 0.0, 0.0]),
            unit(&[0.0, 1.0, 0.0, 0.0]),
        ];
        let orthogonal = vec![unit(&[0.0, 0.0, 0.0, 1.0])];
        let s = stage.win_rates(&ead, &[&mixed, &orthogonal]);
        assert!(s[0] > s[1], "centroid-aligned {s:?} must beat orthogonal");
    }

    // --- state-contract: judge verdicts are memoized per (ad, cv_i, cv_j) ---
    //
    // The judge tournament is O(N²) pairwise forward passes (2 per pair). A
    // re-rank that changed nothing the judge reads (e.g. a criterion was
    // added) must run ZERO new verdicts; a new candidate must only pay for
    // its own pairs; a changed ad must invalidate everything.

    fn cached_db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(crate::db::SCHEMA).expect("schema");
        conn
    }

    #[test]
    fn verdict_cache_skips_recomputation_and_matches_uncached_values() {
        let stage = linear_stage();
        let conn = cached_db();

        let ead = [1.0f32, 0.0, 0.0, 0.0];
        let good = vec![unit(&[1.0, 0.0, 0.0, 0.0])];
        let bad = vec![unit(&[0.0, 1.0, 0.0, 0.0])];
        let keys = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let k2 = keys(&["cv-good", "cv-bad"]);

        // 1st pass: the single pair is computed fresh and matches the
        // uncached tournament exactly.
        let (wins, stats) = stage
            .win_rates_cached(&conn, None, "fake", "ad-1", &k2, &ead, &[&good, &bad])
            .expect("cached win rates");
        assert_eq!((stats.fresh, stats.cached), (1, 0), "one fresh pair");
        assert_eq!(wins, stage.win_rates(&ead, &[&good, &bad]));

        // 2nd pass, identical inputs (e.g. a criterion was added): zero
        // forward passes — every verdict comes from the cache.
        let (wins2, stats2) = stage
            .win_rates_cached(&conn, None, "fake", "ad-1", &k2, &ead, &[&good, &bad])
            .expect("cached win rates");
        assert_eq!((stats2.fresh, stats2.cached), (0, 1), "fully memoized");
        assert_eq!(wins2, wins, "memoized verdicts must be bit-identical");

        // 3rd candidate joins: only its own 2 pairs are computed; the old
        // pair stays cached, and the values still match the uncached run.
        let ugly = vec![unit(&[0.0, 0.0, 1.0, 0.0])];
        let k3 = keys(&["cv-good", "cv-bad", "cv-ugly"]);
        let (wins3, stats3) = stage
            .win_rates_cached(
                &conn,
                None,
                "fake",
                "ad-1",
                &k3,
                &ead,
                &[&good, &bad, &ugly],
            )
            .expect("cached win rates");
        assert_eq!((stats3.fresh, stats3.cached), (2, 1), "only new pairs");
        assert_eq!(wins3, stage.win_rates(&ead, &[&good, &bad, &ugly]));

        // Changed ad: every verdict involving it is invalid — full recompute.
        let (_, stats4) = stage
            .win_rates_cached(
                &conn,
                None,
                "fake",
                "ad-2",
                &k3,
                &ead,
                &[&good, &bad, &ugly],
            )
            .expect("cached win rates");
        assert_eq!(
            (stats4.fresh, stats4.cached),
            (3, 0),
            "ad change invalidates"
        );

        // Changed embedder: same — verdicts are vectors-of-that-model facts.
        let (_, stats5) = stage
            .win_rates_cached(
                &conn,
                None,
                "other",
                "ad-2",
                &k3,
                &ead,
                &[&good, &bad, &ugly],
            )
            .expect("cached win rates");
        assert_eq!(
            (stats5.fresh, stats5.cached),
            (3, 0),
            "embedder change invalidates"
        );
    }
}
