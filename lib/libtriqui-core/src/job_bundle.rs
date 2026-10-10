//! Job bundles (`.triquijob`): a portable zip holding one job's setup, an
//! optional results snapshot, and the original CV files. Layout:
//!
//! ```text
//! manifest.json   format_version, app version, contents summary
//! job.json        title, ad text?, triqui size?, weights?, selected criteria
//! results.json    optional per-candidate rank/points/liked + per-question scores
//! candidates/     the original CV files (flat, de-duplicated names)
//! ```
//!
//! Embeddings and learned feedback vectors are deliberately NOT exported — they
//! are model-specific and recomputed on the receiving machine. The bundle
//! carries only what survives a machine change: text, numbers, flags, files.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

/// Current bundle format. Importers refuse bundles from newer versions.
pub const FORMAT_VERSION: u32 = 1;
/// Custom file extension (the file is an ordinary zip inside).
pub const EXTENSION: &str = "triquijob";

const MANIFEST_NAME: &str = "manifest.json";
const JOB_NAME: &str = "job.json";
const RESULTS_NAME: &str = "results.json";
const CANDIDATES_PREFIX: &str = "candidates/";

// ---------------------------------------------------------------------------
// Options + summary — the wire-contract DTOs live in `libtriqui_client`
// and are re-exported through `crate::types` (also the Tauri IPC shape).
// ---------------------------------------------------------------------------

pub use crate::types::{BundleExportOptions, BundleExportSummary};

// ---------------------------------------------------------------------------
// Bundle-internal JSON payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub format_version: u32,
    pub app: String,
    pub app_version: String,
    pub exported_at: String,
    pub title: String,
    pub candidates: i64,
    pub has_results: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleCriterion {
    /// Original criterion id — the join key for `results.json` scores.
    pub id: String,
    pub text: String,
    /// `"Positive" | "Negative"` (normalized on import).
    pub sign: String,
    /// `None` when point weights were excluded from the export.
    pub weight: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleJob {
    pub title: String,
    #[serde(default)]
    pub job_ad_text: Option<String>,
    #[serde(default)]
    pub triqui_size: Option<i64>,
    /// `None` when point weights were excluded from the export.
    #[serde(default)]
    pub ad_weight: Option<f64>,
    #[serde(default)]
    pub criteria: Vec<BundleCriterion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleScore {
    /// `"ad"` (the embedder's job-fit cosine), `"ad_judge"` (the judge's
    /// win-rate, judge mode only), or the original criterion id (feedback
    /// questions are never exported).
    pub question_id: String,
    /// `"Ad" | "AdJudge" | "Criterion"` (as stored in `candidate_scores` —
    /// `AdJudge` is the judge-mode job-ad row, also carrying `question_id: "ad"`).
    pub kind: String,
    pub sign: String,
    pub label: String,
    pub points: f64,
    pub max_points: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleResultRow {
    /// File name inside `candidates/` (de-duplicated at export time).
    pub file: String,
    pub file_type: String,
    pub status: String,
    pub rank: Option<i64>,
    pub points: Option<f64>,
    pub points_max: Option<f64>,
    /// `false` when likes were excluded from the export.
    pub liked: bool,
    #[serde(default)]
    pub scores: Vec<BundleScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleResults {
    pub rows: Vec<BundleResultRow>,
}

/// Everything `read_bundle` pulls out of a bundle file.
#[derive(Debug, Clone)]
pub struct BundleMetadata {
    pub manifest: BundleManifest,
    pub job: BundleJob,
    pub results: Option<BundleResults>,
}

/// Result of creating a job from a bundle: the new ids the caller needs to
/// finish the import (results application + pipeline spawn).
#[derive(Debug, Clone)]
pub struct BundleJobCreated {
    pub job_id: String,
    /// Original criterion id → new criterion id.
    pub criterion_map: HashMap<String, String>,
}

/// Outcome of applying a results snapshot to an imported job.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, specta::Type)]
pub struct BundleApplyStats {
    pub rows_matched: i64,
    pub rows_unmatched: i64,
    pub liked_applied: i64,
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Export a job + selected setup + optional results snapshot + CV files to
/// `out_path` (a `.triquijob` zip). Missing source files are skipped and
/// reported in the summary, not fatal.
pub fn export_job(
    conn: &Connection,
    job_id: &str,
    opts: &BundleExportOptions,
    out_path: &Path,
    app_version: &str,
) -> Result<BundleExportSummary, String> {
    let (title, job_ad_text, triqui_size, ad_weight): (String, Option<String>, i64, f64) = conn
        .query_row(
            "SELECT title, job_ad_text, triqui_size, ad_weight FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .map_err(|e| format!("Job not found: {e}"))?;

    // Selected criteria, creation order preserved.
    let mut criteria = Vec::new();
    for cid in &opts.criterion_ids {
        let row: Option<(String, String, f64)> = conn
            .query_row(
                "SELECT text, sign, weight FROM criteria WHERE id = ?1 AND job_id = ?2",
                rusqlite::params![cid, job_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .ok();
        if let Some((text, sign, weight)) = row {
            criteria.push(BundleCriterion {
                id: cid.clone(),
                text,
                sign,
                weight: opts.include_weights.then_some(weight),
            });
        }
    }

    // Candidates (ranked first), with a de-duplicated bundle file name each.
    let mut stmt = conn
        .prepare(
            "SELECT id, file_name, file_type, extraction_status, rank, points, points_max, liked, source_file_path
             FROM candidates WHERE job_id = ?1
             ORDER BY (rank IS NULL), rank, file_name",
        )
        .map_err(|e| e.to_string())?;
    let rows: Vec<(
        String,
        String,
        String,
        String,
        Option<i64>,
        Option<f64>,
        Option<f64>,
        bool,
        String,
    )> = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
                r.get::<_, i64>(7)? != 0,
                r.get(8)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    drop(stmt);

    // Optional top-N cut. Rows are ordered best-rank first, so truncating keeps
    // the top N (unranked candidates sort last and are dropped first).
    let mut rows = rows;
    if let Some(n) = opts.top_n {
        rows.truncate(n.max(1) as usize);
    }

    // Per-question scores keyed by candidate id (never the feedback questions).
    let mut scores_by_candidate: HashMap<String, Vec<BundleScore>> = HashMap::new();
    if opts.include_results {
        let mut stmt = conn
            .prepare(
                "SELECT s.candidate_id, s.question_id, s.kind, s.sign, s.label, s.points, s.max_points
                 FROM candidate_scores s JOIN candidates c ON c.id = s.candidate_id
                 WHERE c.job_id = ?1 AND s.kind IN ('Ad', 'AdJudge', 'Criterion')",
            )
            .map_err(|e| e.to_string())?;
        let rows: Vec<(String, String, String, String, String, f64, f64)> = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        for (cid, question_id, kind, sign, label, points, max_points) in rows {
            // Keep only questions the bundle manifest actually carries: the
            // ad rows when the ad is included (the embedder's cosine and, in
            // judge mode, the judge's win-rate), criterion rows for the selected
            // criteria only — an excluded criterion's label (its text) must
            // not leak through the scores.
            let included = if question_id == crate::sort::AD_QUESTION_ID
                || question_id == crate::sort::JUDGE_QUESTION_ID
            {
                opts.include_ad
            } else {
                opts.criterion_ids.iter().any(|id| *id == question_id)
            };
            if !included {
                continue;
            }
            scores_by_candidate
                .entry(cid)
                .or_default()
                .push(BundleScore {
                    question_id,
                    kind,
                    sign,
                    label,
                    points,
                    max_points,
                });
        }
    }

    // Build the zip.
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create output directory: {e}"))?;
    }
    let file = std::fs::File::create(out_path)
        .map_err(|e| format!("Cannot create {}: {e}", out_path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let mut result_rows = Vec::new();
    let mut used_names = std::collections::HashSet::new();
    let mut files_missing = 0i64;
    let mut files_written = 0i64;

    for (cid, file_name, file_type, status, rank, points, points_max, liked, src) in &rows {
        let bundle_name = unique_name(&safe_name(file_name), &mut used_names);
        let src_path = Path::new(src);
        if src_path.is_file() {
            let mut f = std::fs::File::open(src_path)
                .map_err(|e| format!("Cannot open {}: {e}", src_path.display()))?;
            zip.start_file(format!("{CANDIDATES_PREFIX}{bundle_name}"), options)
                .map_err(|e| format!("Zip write failed: {e}"))?;
            std::io::copy(&mut f, &mut zip).map_err(|e| format!("Zip write failed: {e}"))?;
            files_written += 1;
        } else {
            files_missing += 1;
            log::warn!("bundle export: source file missing, skipping: {src}");
        }

        result_rows.push(BundleResultRow {
            file: bundle_name,
            file_type: file_type.clone(),
            status: status.clone(),
            rank: *rank,
            points: *points,
            points_max: *points_max,
            liked: opts.include_likes && *liked,
            scores: scores_by_candidate.get(cid).cloned().unwrap_or_default(),
        });
    }

    let has_results = opts.include_results || opts.include_likes;
    let manifest = BundleManifest {
        format_version: FORMAT_VERSION,
        app: "triqui".into(),
        app_version: app_version.to_string(),
        exported_at: conn
            .query_row("SELECT datetime('now')", [], |r| r.get(0))
            .unwrap_or_default(),
        title: title.clone(),
        candidates: files_written,
        has_results,
    };
    let job_payload = BundleJob {
        title,
        job_ad_text: opts.include_ad.then_some(job_ad_text).flatten(),
        triqui_size: Some(triqui_size),
        ad_weight: opts.include_weights.then_some(ad_weight),
        criteria,
    };

    fn write_json<T: Serialize>(
        zip: &mut zip::ZipWriter<std::fs::File>,
        name: &str,
        value: &T,
        options: zip::write::SimpleFileOptions,
    ) -> Result<(), String> {
        let json = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
        zip.start_file(name, options)
            .map_err(|e| format!("Zip write failed: {e}"))?;
        zip.write_all(json.as_bytes())
            .map_err(|e| format!("Zip write failed: {e}"))
    }
    write_json(&mut zip, MANIFEST_NAME, &manifest, options)?;
    write_json(&mut zip, JOB_NAME, &job_payload, options)?;
    if has_results {
        write_json(
            &mut zip,
            RESULTS_NAME,
            &BundleResults { rows: result_rows },
            options,
        )?;
    }
    zip.finish()
        .map_err(|e| format!("Zip finish failed: {e}"))?;

    Ok(BundleExportSummary {
        candidates_total: rows.len() as i64,
        files_missing,
        criteria_included: job_payload.criteria.len(),
    })
}

/// Replace path separators and other unsafe characters; keep the name readable.
fn safe_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '\0' => '_',
            _ => c,
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "unnamed".into()
    } else {
        trimmed.to_string()
    }
}

/// `jane.pdf` → `jane.pdf`, `jane (2).pdf`, `jane (3).pdf`, …
fn unique_name(name: &str, used: &mut std::collections::HashSet<String>) -> String {
    if used.insert(name.to_string()) {
        return name.to_string();
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) if !s.is_empty() => (s.to_string(), format!(".{e}")),
        _ => (name.to_string(), String::new()),
    };
    for i in 2.. {
        let candidate = format!("{stem} ({i}){ext}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!()
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Read and validate the JSON members of a bundle. CV files are extracted
/// separately via [`extract_bundle_candidates`].
pub fn read_bundle(path: &Path) -> Result<BundleMetadata, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("{} is not a readable bundle: {e}", path.display()))?;

    let manifest: BundleManifest = read_json_member(&mut archive, MANIFEST_NAME)?;
    if manifest.format_version > FORMAT_VERSION {
        return Err(format!(
            "This bundle was created by a newer version of TalentRankIQ (format {v} > {FORMAT_VERSION}). Please update the app.",
            v = manifest.format_version
        ));
    }
    let job: BundleJob = read_json_member(&mut archive, JOB_NAME)?;
    let results: Option<BundleResults> = {
        let names: Vec<String> = (0..archive.len())
            .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
            .collect();
        if names.iter().any(|n| n == RESULTS_NAME) {
            Some(read_json_member(&mut archive, RESULTS_NAME)?)
        } else {
            None
        }
    };
    Ok(BundleMetadata {
        manifest,
        job,
        results,
    })
}

fn read_json_member<T: serde::de::DeserializeOwned>(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
) -> Result<T, String> {
    let mut member = archive
        .by_name(name)
        .map_err(|e| format!("Bundle is missing {name}: {e}"))?;
    let mut buf = String::new();
    member
        .read_to_string(&mut buf)
        .map_err(|e| format!("Cannot read {name}: {e}"))?;
    serde_json::from_str(&buf).map_err(|e| format!("{name} is invalid: {e}"))
}

/// Extract every `candidates/` entry of the bundle into `dest_dir` (flat).
/// Entries trying to escape the directory (zip-slip) or nested below it are
/// skipped. Returns `(extracted, skipped)`.
pub fn extract_bundle_candidates(path: &Path, dest_dir: &Path) -> Result<(i64, i64), String> {
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| format!("Cannot create {}: {e}", dest_dir.display()))?;
    let file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {e}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("{} is not a readable bundle: {e}", path.display()))?;

    let mut extracted = 0i64;
    let mut skipped = 0i64;
    for i in 0..archive.len() {
        let mut member = match archive.by_index(i) {
            Ok(m) => m,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let name = member.name().to_string();
        let Some(rest) = name.strip_prefix(CANDIDATES_PREFIX) else {
            continue;
        };
        // Flat files only: no nested directories, no traversal.
        if rest.is_empty()
            || name.ends_with('/')
            || rest.contains('/')
            || rest.contains('\\')
            || rest == ".."
            || rest.contains('\0')
        {
            log::warn!("bundle import: skipping unsafe/nested entry {name:?}");
            skipped += 1;
            continue;
        }
        let dest = dest_dir.join(safe_name(rest));
        let mut out = match std::fs::File::create(&dest) {
            Ok(f) => f,
            Err(e) => {
                log::warn!("bundle import: cannot create {}: {e}", dest.display());
                skipped += 1;
                continue;
            }
        };
        match std::io::copy(&mut member, &mut out) {
            Ok(_) => extracted += 1,
            Err(e) => {
                log::warn!("bundle import: cannot extract {name:?}: {e}");
                skipped += 1;
            }
        }
    }
    Ok((extracted, skipped))
}

/// Insert the job + criteria rows for an imported bundle. Weights absent from
/// the bundle fall back to the `ikapp.yaml` rubric defaults; feedback weights
/// are never imported (personal). `cv_dir` becomes the job's watched CV
/// directory (the staging dir the bundle's CV files were extracted into).
pub fn create_job_from_bundle(
    conn: &Connection,
    meta: &BundleMetadata,
    cv_dir: &str,
) -> Result<BundleJobCreated, String> {
    let rubric = crate::app_config::rubric_defaults();
    let job_id = format!("job_{}", uuid::Uuid::new_v4());

    // De-duplicate the title so the Jobs list stays unambiguous.
    let mut title = meta.job.title.trim().to_string();
    if title.is_empty() {
        title = "Imported job".into();
    }
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM jobs WHERE title = ?1 LIMIT 1",
            rusqlite::params![&title],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if exists {
        title = format!("{title} (imported)");
    }

    conn.execute(
        "INSERT INTO jobs (id, title, job_ad_text, import_directory_path, triqui_size, ad_weight, fb_job_weight, fb_profile_weight)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            &job_id,
            &title,
            &meta.job.job_ad_text,
            cv_dir,
            meta.job.triqui_size.unwrap_or(10),
            meta.job.ad_weight.unwrap_or(rubric.ad_points),
            rubric.fb_job_points,
            rubric.fb_profile_points,
        ],
    )
    .map_err(|e| format!("Cannot create job: {e}"))?;

    let mut criterion_map = HashMap::new();
    for c in &meta.job.criteria {
        let new_id = format!("crit_{}", uuid::Uuid::new_v4());
        let sign = if c.sign.eq_ignore_ascii_case("negative") {
            "Negative"
        } else {
            "Positive"
        };
        conn.execute(
            "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                &new_id,
                &job_id,
                &c.text,
                sign,
                c.weight.unwrap_or(rubric.criterion_points),
            ],
        )
        .map_err(|e| format!("Cannot import criterion: {e}"))?;
        criterion_map.insert(c.id.clone(), new_id);
    }

    Ok(BundleJobCreated {
        job_id,
        criterion_map,
    })
}

/// Apply the results snapshot (ranks, points, likes, per-question scores) to an
/// imported job. Rows are matched by bundle file name, which equals the staged
/// file name the pipeline imported. Excluded criteria have no mapping and their
/// score columns are dropped; feedback scores are never imported. Called after
/// the extract/embed pipeline so the snapshot wins over the auto-rank.
pub fn apply_bundle_results(
    conn: &Connection,
    job_id: &str,
    results: &BundleResults,
    criterion_map: &HashMap<String, String>,
    include_likes: bool,
) -> Result<BundleApplyStats, String> {
    let mut stats = BundleApplyStats::default();
    let mut any_rank = false;

    for row in &results.rows {
        let candidate_id: Option<String> = conn
            .query_row(
                "SELECT id FROM candidates WHERE job_id = ?1 AND file_name = ?2 LIMIT 1",
                rusqlite::params![job_id, &row.file],
                |r| r.get(0),
            )
            .ok();
        let Some(candidate_id) = candidate_id else {
            stats.rows_unmatched += 1;
            continue;
        };
        stats.rows_matched += 1;

        if row.rank.is_some() || row.points.is_some() || row.points_max.is_some() {
            conn.execute(
                "UPDATE candidates SET
                    rank = COALESCE(?1, rank),
                    points = COALESCE(?2, points),
                    points_max = COALESCE(?3, points_max)
                 WHERE id = ?4",
                rusqlite::params![row.rank, row.points, row.points_max, &candidate_id],
            )
            .map_err(|e| e.to_string())?;
            any_rank |= row.rank.is_some();
        }
        if include_likes && row.liked {
            conn.execute(
                "UPDATE candidates SET liked = 1 WHERE id = ?1",
                rusqlite::params![&candidate_id],
            )
            .map_err(|e| e.to_string())?;
            stats.liked_applied += 1;
        }

        if !row.scores.is_empty() {
            conn.execute(
                "DELETE FROM candidate_scores WHERE candidate_id = ?1",
                rusqlite::params![&candidate_id],
            )
            .map_err(|e| e.to_string())?;
            for s in &row.scores {
                // Map to the imported rubric; drop questions that did not make it.
                // The ad stage grades keep their own ids: the embedder's cosine
                // (`ad`) and the judge's win-rate (`ad_judge`) are distinct rows.
                let question_id = if s.question_id == crate::sort::AD_QUESTION_ID
                    || s.question_id == crate::sort::JUDGE_QUESTION_ID
                {
                    s.question_id.clone()
                } else {
                    match criterion_map.get(&s.question_id) {
                        Some(id) => id.clone(),
                        None => continue,
                    }
                };
                conn.execute(
                    "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        &format!("score_{}", uuid::Uuid::new_v4()),
                        &candidate_id,
                        &question_id,
                        &s.kind,
                        &s.sign,
                        &s.label,
                        &s.points,
                        &s.max_points,
                    ],
                )
                .map_err(|e| e.to_string())?;
            }
        }
    }

    if any_rank {
        // The visible triqui now matches the imported inputs — not stale.
        conn.execute(
            "UPDATE jobs SET last_ranked_at = datetime('now') WHERE id = ?1",
            rusqlite::params![job_id],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_conn() -> Connection {
        // A real (temp) file DB so `open_connection_at` runs the full migration.
        let path = std::env::temp_dir().join(format!("bundle-test-db-{}.db", uuid::Uuid::new_v4()));
        crate::db::open_connection_at(&path).expect("test db")
    }

    /// Seed a job with 2 criteria, 2 ranked candidates (one liked), scores for
    /// both ad + criteria questions, and real source files in `dir`.
    fn seed_job(conn: &Connection, dir: &Path) -> String {
        conn.execute(
            "INSERT INTO jobs (id, title, job_ad_text, triqui_size, ad_weight) VALUES ('job_1', 'Rust Engineer', 'Build things', 10, 50)",
            [],
        )
        .unwrap();
        for (id, text, weight) in [
            ("crit_a", "Knows Rust", 10.0),
            ("crit_b", "German fluent", 5.0),
        ] {
            conn.execute(
                "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, 'job_1', ?2, 'Positive', ?3)",
                rusqlite::params![id, text, weight],
            )
            .unwrap();
        }
        std::fs::create_dir_all(dir).unwrap();
        for (id, name, rank, points, liked) in [
            ("cand_1", "jane.pdf", 1, 42.5, true),
            ("cand_2", "jane.pdf", 2, 10.0, false), // duplicate name on purpose
        ] {
            let src = dir.join(id);
            std::fs::write(&src, format!("cv text of {id}")).unwrap();
            conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, rank, points, points_max, liked)
                 VALUES (?1, 'job_1', ?2, ?3, 'Pdf', 'Extracted', ?4, ?5, 65.0, ?6)",
                rusqlite::params![id, src.to_string_lossy(), name, rank, points, liked],
            )
            .unwrap();
            for (qid, kind, pts) in [
                ("ad", "Ad", 30.0),
                ("crit_a", "Criterion", 12.5),
                ("crit_b", "Criterion", 3.0), // excluded from the export below
            ] {
                conn.execute(
                    "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
                     VALUES (?1, ?2, ?3, ?4, 'Positive', 'label', ?5, 50.0)",
                    rusqlite::params![format!("score_{id}_{qid}"), id, qid, kind, pts],
                )
                .unwrap();
            }
        }
        "job_1".into()
    }

    #[test]
    fn export_round_trips_through_import() {
        let conn = test_conn();
        let src_dir =
            std::env::temp_dir().join(format!("bundle-test-src-{}", uuid::Uuid::new_v4()));
        let job_id = seed_job(&conn, &src_dir);

        let out =
            std::env::temp_dir().join(format!("bundle-test-{}.triquijob", uuid::Uuid::new_v4()));
        let opts = BundleExportOptions {
            include_ad: true,
            include_weights: true,
            include_results: true,
            include_likes: true,
            criterion_ids: vec!["crit_a".into()], // crit_b excluded
            top_n: None,
        };
        let summary = export_job(&conn, &job_id, &opts, &out, "0.0.0-test").unwrap();
        assert_eq!(summary.candidates_total, 2);
        assert_eq!(summary.files_missing, 0);
        assert_eq!(summary.criteria_included, 1);

        let meta = read_bundle(&out).unwrap();
        assert_eq!(meta.manifest.format_version, 1);
        assert_eq!(meta.job.title, "Rust Engineer");
        assert_eq!(meta.job.job_ad_text.as_deref(), Some("Build things"));
        assert_eq!(meta.job.criteria.len(), 1);
        assert_eq!(meta.job.criteria[0].weight, Some(10.0));
        let results = meta.results.as_ref().unwrap();
        assert_eq!(results.rows.len(), 2);
        // Duplicate source names get unique bundle names.
        assert_ne!(results.rows[0].file, results.rows[1].file);
        assert!(results.rows[0].liked);
        assert_eq!(results.rows[0].scores.len(), 2);
        // The excluded criterion's score rows never enter the file — its
        // label (the criterion text) must not leak via the results block.
        assert!(
            results
                .rows
                .iter()
                .all(|r| r.scores.iter().all(|s| s.question_id != "crit_b")),
            "unselected criterion scores must not export"
        );

        // Import into a fresh DB.
        let dest = test_conn();
        let stage =
            std::env::temp_dir().join(format!("bundle-test-stage-{}", uuid::Uuid::new_v4()));
        let (extracted, skipped) = extract_bundle_candidates(&out, &stage).unwrap();
        assert_eq!((extracted, skipped), (2, 0));
        let created = create_job_from_bundle(&dest, &meta, stage.to_str().unwrap()).unwrap();
        let new_job = &created.job_id;

        // Re-run the "pipeline" by hand: rows named after the staged files.
        for entry in std::fs::read_dir(&stage).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            dest.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
                 VALUES ('new_' || ?1, ?2, ?3, ?4, 'Pdf', 'Extracted')",
                rusqlite::params![name, new_job, entry.path().to_string_lossy(), name],
            )
            .unwrap();
        }
        let stats =
            apply_bundle_results(&dest, new_job, results, &created.criterion_map, true).unwrap();
        assert_eq!(stats.rows_matched, 2);
        assert_eq!(stats.rows_unmatched, 0);
        assert_eq!(stats.liked_applied, 1);

        let (title, dir_set, rank1, pts1, liked1): (String, String, i64, f64, bool) = dest
            .query_row(
                "SELECT j.title, j.import_directory_path, c.rank, c.points, c.liked
                 FROM jobs j JOIN candidates c ON c.job_id = j.id WHERE c.rank = 1",
                [],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get::<_, i64>(4)? != 0,
                    ))
                },
            )
            .unwrap();
        assert_eq!(title, "Rust Engineer");
        assert_eq!(dir_set, stage.to_str().unwrap());
        assert_eq!(rank1, 1);
        assert_eq!(pts1, 42.5);
        assert!(liked1);

        // Excluded criterion was not imported; its score column was dropped.
        let crit_count: i64 = dest
            .query_row(
                "SELECT COUNT(*) FROM criteria WHERE job_id = ?1",
                rusqlite::params![new_job],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(crit_count, 1);
        let scores: i64 = dest
            .query_row(
                "SELECT COUNT(*) FROM candidate_scores WHERE question_id = 'ad'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(scores, 2);

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&src_dir);
        let _ = std::fs::remove_dir_all(&stage);
    }

    #[test]
    fn top_n_exports_only_the_best_ranked() {
        let conn = test_conn();
        let src_dir =
            std::env::temp_dir().join(format!("bundle-test-topn-{}", uuid::Uuid::new_v4()));
        let job_id = seed_job(&conn, &src_dir);

        let out = std::env::temp_dir().join(format!(
            "bundle-test-topn-{}.triquijob",
            uuid::Uuid::new_v4()
        ));
        let opts = BundleExportOptions {
            include_results: true,
            include_likes: true,
            top_n: Some(1),
            ..Default::default()
        };
        let summary = export_job(&conn, &job_id, &opts, &out, "0.0.0-test").unwrap();
        assert_eq!(summary.candidates_total, 1);

        let meta = read_bundle(&out).unwrap();
        let results = meta.results.as_ref().unwrap();
        assert_eq!(results.rows.len(), 1);
        assert_eq!(results.rows[0].rank, Some(1));
        assert!(results.rows[0].liked);
        // The job setup still carries the sender's triqui size.
        assert_eq!(meta.job.triqui_size, Some(10));

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&src_dir);
    }

    #[test]
    fn rejects_newer_format_versions() {
        let conn = test_conn();
        let src_dir =
            std::env::temp_dir().join(format!("bundle-test-src2-{}", uuid::Uuid::new_v4()));
        let job_id = seed_job(&conn, &src_dir);
        let out = std::env::temp_dir().join(format!(
            "bundle-test2-{}.triquijob",
            uuid::Uuid::new_v4()
        ));
        export_job(
            &conn,
            &job_id,
            &BundleExportOptions::default(),
            &out,
            "0.0.0-test",
        )
        .unwrap();

        // Rewrite the manifest with a future version.
        let file = std::fs::File::open(&out).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let path = out.with_extension("zip");
        let writer = std::fs::File::create(&path).unwrap();
        let mut new_zip = zip::ZipWriter::new(writer);
        for i in 0..archive.len() {
            let mut member = archive.by_index(i).unwrap();
            let name = member.name().to_string();
            new_zip
                .start_file(
                    &name,
                    zip::write::SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Deflated),
                )
                .unwrap();
            if name == MANIFEST_NAME {
                let mut buf = String::new();
                member.read_to_string(&mut buf).unwrap();
                let mut manifest: BundleManifest = serde_json::from_str(&buf).unwrap();
                manifest.format_version = FORMAT_VERSION + 1;
                new_zip
                    .write_all(serde_json::to_string(&manifest).unwrap().as_bytes())
                    .unwrap();
            } else {
                std::io::copy(&mut member, &mut new_zip).unwrap();
            }
        }
        new_zip.finish().unwrap();
        let err = read_bundle(&path).unwrap_err();
        assert!(err.contains("newer version"), "unexpected error: {err}");

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&src_dir);
    }

    #[test]
    fn zip_slip_entries_are_skipped() {
        let path =
            std::env::temp_dir().join(format!("bundle-slip-{}.triquijob", uuid::Uuid::new_v4()));
        let writer = std::fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(writer);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("candidates/../../evil.txt", options)
            .unwrap();
        zip.write_all(b"boom").unwrap();
        zip.start_file("candidates/nested/sub.txt", options)
            .unwrap();
        zip.write_all(b"boom").unwrap();
        zip.start_file("candidates/ok.txt", options).unwrap();
        zip.write_all(b"fine").unwrap();
        zip.finish().unwrap();

        let dest = std::env::temp_dir().join(format!("bundle-slip-dest-{}", uuid::Uuid::new_v4()));
        let (extracted, skipped) = extract_bundle_candidates(&path, &dest).unwrap();
        assert_eq!((extracted, skipped), (1, 2));
        assert!(dest.join("ok.txt").is_file());
        assert!(!dest.parent().unwrap().join("evil.txt").exists());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dest);
    }
}
