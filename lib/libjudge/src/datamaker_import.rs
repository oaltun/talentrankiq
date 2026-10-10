//! Streaming reader for `datamaker.judge.v1` exports.
//!
//! Reads judge_results.jsonl + bundles/index.jsonl + bundle zips to reconstruct
//! the full text of each judgment. Produces JobGroup/Doc structs compatible
//! with judgetrainer's dataset module.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

/// One line of `judge_results.jsonl` (fields as exported by `datamaker export-judge`).
#[derive(Debug, Clone, Deserialize)]
pub struct JudgeRow {
    #[serde(default)]
    pub schema: String,
    pub pair_id: String,
    pub jobad_id: String,
    pub cv_a_id: String,
    pub cv_b_id: String,
    pub a_is: String,
    #[serde(default)]
    pub verdict: String,
    #[serde(default)]
    pub winner_doc: String,
    #[serde(default)]
    pub preferred: Option<String>,
    #[serde(default)]
    pub confidence: f64,
    pub model: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub reasons: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub split: String,
}

/// One line of `bundles/index.jsonl`.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexRow {
    pub sha256: String,
    pub doc_id: String,
    pub kind: String,
    pub bundle: String,
}

/// A document text loaded from a bundle zip.
#[derive(Debug, Clone)]
pub struct DocText {
    pub sha256: String,
    pub text: String,
}

/// Job group with its CVs, built from datamaker pairwise judgments.
/// Mirrors judgetrainer::dataset::JobGroup but with pairwise labels.
#[derive(Debug, Clone)]
pub struct DatamakerJobGroup {
    pub job_id: String,
    pub job_text: String,
    pub docs: Vec<DatamakerDoc>,
}

/// One CV under a job, with its pairwise comparison data.
#[derive(Debug, Clone)]
pub struct DatamakerDoc {
    pub resume_id: String,
    pub resume_text: String,
    /// For pairwise training: list of (other_resume_id, preferred) where this doc won
    pub wins_against: Vec<(String, f64)>,
    /// For pairwise training: list of (other_resume_id, preferred) where this doc lost
    pub loses_to: Vec<(String, f64)>,
}

/// Configuration for loading a datamaker export.
#[derive(Debug, Clone)]
pub struct DatamakerExportConfig {
    pub export_dir: PathBuf,
    /// Minimum confidence threshold for including judgments
    pub min_confidence: f64,
    /// Filter by judge model (None = all models)
    pub model_filter: Option<String>,
    /// Only include status == "ok"
    pub only_ok: bool,
}

impl Default for DatamakerExportConfig {
    fn default() -> Self {
        Self {
            export_dir: PathBuf::new(),
            min_confidence: 0.0,
            model_filter: None,
            only_ok: true,
        }
    }
}

/// Load a datamaker export and build job groups with pairwise comparison data.
pub fn load_datamaker_export(config: DatamakerExportConfig) -> Result<Vec<DatamakerJobGroup>> {
    let export_dir = &config.export_dir;
    
    // 1. Load index.jsonl to map sha256 -> (bundle, doc_id)
    let index_path = export_dir.join("bundles").join("index.jsonl");
    let index = load_index(&index_path)?;
    
    // 2. Load judge_results.jsonl and collect all needed sha256s
    let results_path = export_dir.join("judge_results.jsonl");
    let (rows, needed_shas) = load_judgments(&results_path, &config)?;
    
    // 3. Extract needed documents from bundle zips
    let doc_texts = extract_docs(export_dir, &index, &needed_shas)?;
    
    // 4. Build job groups from judgments
    let job_groups = build_job_groups(&rows, &doc_texts, &config)?;
    
    Ok(job_groups)
}

/// Load the bundle index: sha256 -> IndexRow
fn load_index(index_path: &Path) -> Result<BTreeMap<String, IndexRow>> {
    let mut map = BTreeMap::new();
    let reader = BufReader::with_capacity(128 * 1024, File::open(index_path)?);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: IndexRow = serde_json::from_str(&line)
            .with_context(|| format!("bad JSON line in {}", index_path.display()))?;
        map.insert(entry.sha256.clone(), entry);
    }
    Ok(map)
}

/// Load judgments from judge_results.jsonl, filtering by config.
/// Returns (filtered_rows, set of needed sha256s).
fn load_judgments(
    results_path: &Path,
    config: &DatamakerExportConfig,
) -> Result<(Vec<JudgeRow>, Vec<String>)> {
    let mut rows = Vec::new();
    let mut needed_shas = Vec::new();
    let mut seen_shas = std::collections::HashSet::new();
    
    let reader = BufReader::with_capacity(128 * 1024, File::open(results_path)?);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: JudgeRow = serde_json::from_str(&line)
            .with_context(|| format!("bad JSON line in {}", results_path.display()))?;
        
        // Filter
        if config.only_ok && row.status != "ok" {
            continue;
        }
        if row.confidence < config.min_confidence {
            continue;
        }
        if let Some(ref model) = config.model_filter {
            if &row.model != model {
                continue;
            }
        }
        
        // Collect needed sha256s
        for sha in [&row.jobad_id, &row.cv_a_id, &row.cv_b_id] {
            if seen_shas.insert(sha.clone()) {
                needed_shas.push(sha.clone());
            }
        }
        
        rows.push(row);
    }
    
    Ok((rows, needed_shas))
}

/// Extract needed documents from bundle zips, verifying sha256.
fn extract_docs(
    export_dir: &Path,
    index: &BTreeMap<String, IndexRow>,
    needed_shas: &[String],
) -> Result<HashMap<String, String>> {
    // Group needed shas by bundle
    let mut by_bundle: BTreeMap<String, Vec<&String>> = BTreeMap::new();
    for sha in needed_shas {
        let entry = index.get(sha)
            .with_context(|| format!("sha256 not in index: {}", sha))?;
        by_bundle.entry(entry.bundle.clone()).or_default().push(sha);
    }
    
    let mut texts = HashMap::new();
    
    for (bundle_name, shas) in by_bundle {
        let zip_path = export_dir.join("bundles").join(&bundle_name);
        let file = File::open(&zip_path)
            .with_context(|| format!("open bundle {}", zip_path.display()))?;
        let mut zf = ZipArchive::new(file)
            .with_context(|| format!("read zip {}", zip_path.display()))?;
        
        for sha in shas {
            // Try .txt extension (files are stored as <sha256>.<lang>.txt)
            // We need to find the actual member name in the zip
            let mut found = false;
            for i in 0..zf.len() {
                let mut member = zf.by_index(i)?;
                let name = member.name().to_string();
                if name.starts_with(&format!("{}.", sha)) && name.ends_with(".txt") {
                    let mut raw = Vec::new();
                    member.read_to_end(&mut raw)?;
                    let got = sha256_hex(&raw);
                    if got != *sha {
                        bail!("sha256 mismatch for {} in {}: expected {}, got {}", name, bundle_name, sha, got);
                    }
                    texts.insert(sha.clone(), String::from_utf8_lossy(&raw).into_owned());
                    found = true;
                    break;
                }
            }
            if !found {
                bail!("document {} not found in bundle {}", sha, bundle_name);
            }
        }
    }
    
    Ok(texts)
}

/// Build job groups from judgments and document texts.
fn build_job_groups(
    rows: &[JudgeRow],
    doc_texts: &HashMap<String, String>,
    _config: &DatamakerExportConfig,
) -> Result<Vec<DatamakerJobGroup>> {
    // Group by jobad_id
    let mut job_map: HashMap<String, DatamakerJobGroupBuilder> = HashMap::new();
    
    for row in rows {
        let job_text = doc_texts.get(&row.jobad_id)
            .with_context(|| format!("jobad text not found: {}", row.jobad_id))?;
        let cv_a_text = doc_texts.get(&row.cv_a_id)
            .with_context(|| format!("cv_a text not found: {}", row.cv_a_id))?;
        let cv_b_text = doc_texts.get(&row.cv_b_id)
            .with_context(|| format!("cv_b text not found: {}", row.cv_b_id))?;
        
        let preferred = row.preferred.as_deref().unwrap_or("");
        if preferred.is_empty() {
            continue; // skip ties/refusals
        }
        
        // Determine winner and loser
        let (winner_id, loser_id, winner_text, loser_text) = if preferred == "a" {
            (&row.cv_a_id, &row.cv_b_id, cv_a_text, cv_b_text)
        } else {
            (&row.cv_b_id, &row.cv_a_id, cv_b_text, cv_a_text)
        };
        
        let job = job_map.entry(row.jobad_id.clone())
            .or_insert_with(|| DatamakerJobGroupBuilder::new(row.jobad_id.clone(), job_text.clone()));
        
        // Add/update winner
        job.add_or_update_doc(winner_id, winner_text, true, loser_id, row.confidence);
        // Add/update loser
        job.add_or_update_doc(loser_id, loser_text, false, winner_id, row.confidence);
    }
    
    // Convert builders to final job groups
    let mut job_groups: Vec<DatamakerJobGroup> = job_map
        .into_iter()
        .map(|(_, builder)| builder.build())
        .collect();
    
    // Sort for determinism
    job_groups.sort_by(|a, b| a.job_id.cmp(&b.job_id));
    for job in &mut job_groups {
        job.docs.sort_by(|a, b| a.resume_id.cmp(&b.resume_id));
    }
    
    Ok(job_groups)
}

/// Helper to build a job group incrementally.
struct DatamakerJobGroupBuilder {
    job_id: String,
    job_text: String,
    docs: HashMap<String, DatamakerDocBuilder>,
}

struct DatamakerDocBuilder {
    resume_id: String,
    resume_text: String,
    wins_against: Vec<(String, f64)>,
    loses_to: Vec<(String, f64)>,
}

impl DatamakerJobGroupBuilder {
    fn new(job_id: String, job_text: String) -> Self {
        Self {
            job_id,
            job_text,
            docs: HashMap::new(),
        }
    }
    
    fn add_or_update_doc(
        &mut self,
        resume_id: &str,
        resume_text: &str,
        is_winner: bool,
        other_id: &str,
        confidence: f64,
    ) {
        let doc = self.docs.entry(resume_id.to_string())
            .or_insert_with(|| DatamakerDocBuilder {
                resume_id: resume_id.to_string(),
                resume_text: resume_text.to_string(),
                wins_against: Vec::new(),
                loses_to: Vec::new(),
            });
        
        if is_winner {
            doc.wins_against.push((other_id.to_string(), confidence));
        } else {
            doc.loses_to.push((other_id.to_string(), confidence));
        }
    }
    
    fn build(self) -> DatamakerJobGroup {
        let docs = self.docs.into_iter()
            .map(|(_, b)| DatamakerDoc {
                resume_id: b.resume_id,
                resume_text: b.resume_text,
                wins_against: b.wins_against,
                loses_to: b.loses_to,
            })
            .collect();
        
        DatamakerJobGroup {
            job_id: self.job_id,
            job_text: self.job_text,
            docs,
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[test]
    fn test_sha256_hex() {
        let hash = sha256_hex(b"hello");
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }
}