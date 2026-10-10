//! The gui↔cli wire contract: every DTO that crosses the protocol.
//!
//! These types live here (not in `triqui-core`) so the Tauri gui and
//! `triqui-cli` share one definition — neither side can drift. The
//! Tauri gui uses them in `#[tauri::command]` signatures, so they all
//! derive `specta::Type` (directly or transitively) exactly as before the
//! thin-gui migration; generated `bindings.ts` shapes are unchanged.

use regex::Regex;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::sync::LazyLock;

/// Default shortcut for the quick pane
pub const DEFAULT_QUICK_PANE_SHORTCUT: &str = "CommandOrControl+Shift+.";

/// Maximum size for recovery data files (10MB)
pub const MAX_RECOVERY_DATA_BYTES: u32 = 10_485_760;

/// Pre-compiled regex pattern for filename validation.
/// Only allows alphanumeric characters, dashes, underscores, and a single extension.
pub static FILENAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9]+)?$")
        .expect("Failed to compile filename regex pattern")
});

// ============================================================================
// Preferences
// ============================================================================

/// Application preferences that persist to disk.
/// Only contains settings that should be saved between sessions.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AppPreferences {
    pub theme: String,
    /// Color scheme (palette) applied on top of light/dark mode: "default",
    /// "slate", "ocean", "forest", "amber", or "contrast"
    #[serde(default = "default_color_theme")]
    pub color_theme: String,
    /// Global shortcut for quick pane (e.g., "CommandOrControl+Shift+.")
    /// If None, uses the default shortcut
    pub quick_pane_shortcut: Option<String>,
    /// User's preferred language (e.g., "en", "es", "de")
    /// If None, uses system locale detection
    pub language: Option<String>,
    /// Selected embedding model (e.g. "nomic-embed-text-v1.5"). If None, falls back to
    /// the runtime default or the TRIQI_EMBED_MODEL env var.
    pub embedding_model: Option<String>,
    /// Selected text-inference model. If None, falls back to the runtime default.
    pub inference_model: Option<String>,
    /// Root folder for new jobs' default CV directories (`<root>/job-N/cv`),
    /// set from Settings → General → CV directory. If None, serve falls back
    /// to `gui_cv_dir_root` from `ikapp.yaml`, then the platform Documents dir
    /// + `jobs`.
    pub cv_dir_root: Option<String>,
}

fn default_color_theme() -> String {
    "default".to_string()
}

impl Default for AppPreferences {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
            color_theme: default_color_theme(),
            quick_pane_shortcut: None, // None means use default
            language: None,            // None means use system locale
            embedding_model: None,
            inference_model: None,
            cv_dir_root: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AppPreferences;

    /// `preferences.json` files written before a field existed must keep
    /// parsing (serve's prefs loader rejects nothing but real corruption).
    #[test]
    fn legacy_preferences_without_cv_dir_root_parse() {
        let legacy = r#"{"theme":"dark","quick_pane_shortcut":null,"language":null,"embedding_model":null,"inference_model":null}"#;
        let prefs: AppPreferences = serde_json::from_str(legacy).expect("legacy prefs parse");
        assert_eq!(prefs.theme, "dark");
        assert!(prefs.cv_dir_root.is_none());
    }

    #[test]
    fn cv_dir_root_round_trips() {
        let prefs = AppPreferences {
            cv_dir_root: Some("/tmp/cv-root".into()),
            ..Default::default()
        };
        let back: AppPreferences =
            serde_json::from_str(&serde_json::to_string(&prefs).expect("serialize"))
                .expect("parse");
        assert_eq!(back.cv_dir_root.as_deref(), Some("/tmp/cv-root"));
    }
}

// ============================================================================
// Recovery Errors
// ============================================================================

/// Error types for recovery operations (typed for frontend matching)
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "type")]
pub enum RecoveryError {
    /// File does not exist (expected case, not a failure)
    FileNotFound,
    /// Filename validation failed
    ValidationError { message: String },
    /// Data exceeds size limit
    DataTooLarge { max_bytes: u32 },
    /// File system read/write error
    IoError { message: String },
    /// JSON serialization/deserialization error
    ParseError { message: String },
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryError::FileNotFound => write!(f, "File not found"),
            RecoveryError::ValidationError { message } => write!(f, "Validation error: {message}"),
            RecoveryError::DataTooLarge { max_bytes } => {
                write!(f, "Data too large (max {max_bytes} bytes)")
            }
            RecoveryError::IoError { message } => write!(f, "IO error: {message}"),
            RecoveryError::ParseError { message } => write!(f, "Parse error: {message}"),
        }
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates a filename for safe file system operations.
/// Only allows alphanumeric characters, dashes, underscores, and a single extension.
pub fn validate_filename(filename: &str) -> Result<(), String> {
    if filename.is_empty() {
        return Err("Filename cannot be empty".to_string());
    }

    if filename.chars().count() > 100 {
        return Err("Filename too long (max 100 characters)".to_string());
    }

    if !FILENAME_PATTERN.is_match(filename) {
        return Err(
            "Invalid filename: only alphanumeric characters, dashes, underscores, and dots allowed"
                .to_string(),
        );
    }

    Ok(())
}

/// Validates string input length (by character count, not bytes).
pub fn validate_string_input(input: &str, max_len: usize, field_name: &str) -> Result<(), String> {
    let char_count = input.chars().count();
    if char_count > max_len {
        return Err(format!("{field_name} too long (max {max_len} characters)"));
    }
    Ok(())
}

/// Validates theme value.
pub fn validate_theme(theme: &str) -> Result<(), String> {
    match theme {
        "light" | "dark" | "system" => Ok(()),
        _ => Err("Invalid theme: must be 'light', 'dark', or 'system'".to_string()),
    }
}

/// Validates the color-scheme preference (palette overlay on light/dark).
pub fn validate_color_theme(color_theme: &str) -> Result<(), String> {
    match color_theme {
        "default" | "slate" | "ocean" | "forest" | "amber" | "contrast" => Ok(()),
        _ => Err(
            "Invalid color theme: must be 'default', 'slate', 'ocean', 'forest', 'amber', or 'contrast'"
                .to_string(),
        ),
    }
}

// ============================================================================
// Domain: TalentRankIQ (Baseline ranking — vectors, scores, explanations)
// ============================================================================

/// Supported CV file types.
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq, Eq)]
pub enum FileType {
    Doc,
    Docx,
    Pdf,
    Txt,
    Other,
}

/// Per-candidate extraction lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq, Eq)]
pub enum ExtractionStatus {
    Queued,
    Extracting,
    Extracted,
    Failed,
}

impl FileType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::Doc => "Doc",
            FileType::Docx => "Docx",
            FileType::Pdf => "Pdf",
            FileType::Txt => "Txt",
            FileType::Other => "Other",
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "Docx" => FileType::Docx,
            "Pdf" => FileType::Pdf,
            "Txt" => FileType::Txt,
            "Doc" => FileType::Doc,
            _ => FileType::Other,
        }
    }
}

impl ExtractionStatus {
    #[allow(dead_code)] // Phase-2 ranking writes will use it
    pub fn as_str(&self) -> &'static str {
        match self {
            ExtractionStatus::Queued => "Queued",
            ExtractionStatus::Extracting => "Extracting",
            ExtractionStatus::Extracted => "Extracted",
            ExtractionStatus::Failed => "Failed",
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "Extracting" => ExtractionStatus::Extracting,
            "Failed" => ExtractionStatus::Failed,
            "Extracted" => ExtractionStatus::Extracted,
            _ => ExtractionStatus::Queued,
        }
    }
}

/// A single hiring opening the HR user is screening for.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct Job {
    pub id: String,
    pub title: String,
    pub job_ad_text: Option<String>,
    pub import_directory_path: Option<String>,
    pub triqui_size: i32,
    pub archived: bool,
    pub created_at: String,
    pub updated_at: String,
    pub last_ranked_at: Option<String>,
    /// When ranking inputs (ad / N / criteria) last changed. Drives the stale badge
    /// (NOT bumped on a mere title edit).
    #[serde(default)]
    pub inputs_updated_at: Option<String>,
    /// Rubric weight: max points for the job-ad question.
    #[serde(default = "default_ad_weight")]
    pub ad_weight: f64,
    /// Rubric weight: max points for the per-job feedback question.
    #[serde(default = "default_fb_job_weight")]
    pub fb_job_weight: f64,
    /// Rubric weight: max points for the cross-job feedback question.
    #[serde(default = "default_fb_profile_weight")]
    pub fb_profile_weight: f64,
    /// The job-ad **file** this job reads its ad from, absolute. Resolved on
    /// first read to `<cv-root>/<job-slug>/job-ad.txt` and created empty; the
    /// user can point it elsewhere ("Change"). The ad itself lives in that file
    /// — this app has no ad editor — and edits on disk are picked up and
    /// re-ranked. `None` only until it has been resolved once.
    #[serde(default)]
    pub job_ad_path: Option<String>,
}

/// The job-ad **file**: where it is and what it says.
///
/// The ad is a plain text file the user edits in their own editor — the app
/// has no ad editor. Serve hands this to the gui so the path shown is always
/// the resolved one, and `created` lets the gui say "we made this empty file"
/// exactly once instead of guessing.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct JobAdFile {
    /// Absolute path to the ad file. Created empty on first resolution.
    pub path: String,
    /// The file's contents; empty when the user has not written an ad yet.
    #[serde(default)]
    pub text: String,
    /// True when this call had to create the file.
    #[serde(default)]
    pub created: bool,
}

fn default_ad_weight() -> f64 {
    50.0
}

fn default_fb_job_weight() -> f64 {
    10.0
}

fn default_fb_profile_weight() -> f64 {
    10.0
}

/// An extra requirement beyond the job ad (always positive — negative criteria removed).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct Criterion {
    pub id: String,
    pub job_id: String,
    pub text: String,
    pub weight: f64,
    pub created_at: String,
}

/// One applicant's CV within a Job.
#[allow(dead_code)] // canonical domain entity; Phase-2 full-row queries use it
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct Candidate {
    pub id: String,
    pub job_id: String,
    pub source_file_path: String,
    pub file_name: String,
    pub file_type: FileType,
    pub extraction_status: ExtractionStatus,
    pub extracted_text: Option<String>,
    pub rank: Option<i32>,
    pub imported_at: String,
    pub ranked_at: Option<String>,
}

/// A ranked triqui row (no extracted text — fetched on overlay open).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct TriqiItem {
    pub id: String,
    pub file_name: String,
    pub file_type: FileType,
    pub extraction_status: ExtractionStatus,
    pub rank: i32,
    /// Total rubric points. None until ranked.
    #[serde(default)]
    pub points: Option<f32>,
    /// Sum of question max-points (the rubric capacity). None until ranked.
    #[serde(default)]
    pub points_max: Option<f32>,
    /// Whether the user has liked this CV (feedback signal).
    #[serde(default)]
    pub liked: bool,
    /// When the user last opened this CV in the inspector (read/unread
    /// marking; NULL = unread).
    #[serde(default)]
    pub viewed_at: Option<String>,
}

/// Which kind of rubric question a row represents.
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq, Eq)]
pub enum QuestionKind {
    Ad,
    /// "Job fit (judge)" — pairwise-judge win-rate replacing the `Ad` cosine
    /// row when the gui judge stage is active (`gui_judge_artifact`).
    AdJudge,
    Criterion,
    FeedbackJob,
    FeedbackProfile,
    /// Stage-0.5 scout row — the cheap multilingual embedder's similarity to
    /// the job ad, in the ad row's currency. Scored for *every* candidate, so
    /// the results table can show what the scout thought of a CV that the
    /// heavy embedder has not reached yet.
    Scout,
}

/// Which cascade stage currently backs a candidate's total (the total's ad
/// Which embedding space a candidate's *criteria and feedback* similarities were
/// computed in (scoring v3, `gui/design/app1/cascade-simple.md`).
///
/// The heavy embedder's space is better than the scout's, so the sheet's default
/// order is heavy first, then total within a space. It is a separate axis from
/// [`AdSource`]: a candidate can have scout-space criteria scores and a judge
/// win-rate on its ad slot at the same time, and collapsing that into one "basis"
/// chip is what made the old ordering hard to reason about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum GradeSpace {
    /// The cheap scout's space — the candidate has not been reached by the heavy
    /// embedder yet. Every candidate has a complete scorecard here, because the
    /// scout scores the whole rubric rather than only the ad.
    Scout,
    /// The selected embedder's space.
    Heavy,
}

/// What backs the ad slot of a candidate's total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Type)]
#[serde(rename_all = "lowercase")]
pub enum AdSource {
    /// The scout's ad cosine.
    Scout,
    /// The heavy embedder's ad cosine.
    Heavy,
    /// The judge's pairwise win-rate. It replaces the ad slot's *value* and never
    /// its weight, so it changes no row's max-points.
    Judge,
}

impl GradeSpace {
    /// The space that backs a candidate's rubric scores, from its score rows.
    pub fn from_scores(scores: &[RubricScore]) -> Option<GradeSpace> {
        // The heavy embedder's ad grade exists exactly when this candidate has
        // been embedded, so it is the whole test.
        if scores.iter().any(|s| s.kind == QuestionKind::Ad) {
            Some(GradeSpace::Heavy)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            GradeSpace::Scout => "scout",
            GradeSpace::Heavy => "heavy",
        }
    }
}

impl AdSource {
    /// What backs the ad slot, from a candidate's score rows.
    pub fn from_scores(scores: &[RubricScore]) -> Option<AdSource> {
        let has = |kind: QuestionKind| scores.iter().any(|s| s.kind == kind);
        if has(QuestionKind::AdJudge) {
            Some(AdSource::Judge)
        } else if has(QuestionKind::Ad) {
            Some(AdSource::Heavy)
        } else if has(QuestionKind::Scout) {
            Some(AdSource::Scout)
        } else {
            None
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AdSource::Scout => "scout",
            AdSource::Heavy => "heavy",
            AdSource::Judge => "judge",
        }
    }
}

impl QuestionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuestionKind::Ad => "Ad",
            QuestionKind::AdJudge => "AdJudge",
            QuestionKind::Criterion => "Criterion",
            QuestionKind::FeedbackJob => "FeedbackJob",
            QuestionKind::FeedbackProfile => "FeedbackProfile",
            QuestionKind::Scout => "Scout",
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "AdJudge" => QuestionKind::AdJudge,
            "Criterion" => QuestionKind::Criterion,
            "FeedbackJob" => QuestionKind::FeedbackJob,
            "FeedbackProfile" => QuestionKind::FeedbackProfile,
            "Scout" => QuestionKind::Scout,
            _ => QuestionKind::Ad,
        }
    }

    /// Whether rows of this kind are stage grades *outside* the rubric the
    /// total is made of. The job ad (`Ad`) is the embed stage's grade **and** a
    /// rubric row, so it is not one: the Total's capacity must count its weight
    /// exactly once, which is also true in judge mode, where the `AdJudge` row
    /// replaces it in the total while keeping the same weight.
    pub fn is_stage(&self) -> bool {
        matches!(self, QuestionKind::Scout | QuestionKind::AdJudge)
    }
}

/// One question in a job's rubric (the scoring model) — no per-candidate value.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RubricQuestion {
    pub question_id: String,
    pub label: String,
    pub kind: QuestionKind,
    /// Max points (weight) for this question.
    pub max_points: f32,
}

/// One row of a candidate's rubric score: points earned on a single question.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RubricScore {
    pub question_id: String,
    pub label: String,
    pub kind: QuestionKind,
    /// Points earned.
    pub points: f32,
    /// Max points (weight) for this question.
    pub max_points: f32,
    /// Top matching CV chunk texts with similarity scores (maxsim explanation).
    #[serde(default)]
    pub chunk_explanations: Vec<ChunkExplanation>,
}

/// One chunk that contributed to a rubric score via maxsim.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChunkExplanation {
    /// The chunk text from the CV.
    pub text: String,
    /// Cosine similarity of this chunk against the question vector.
    pub similarity: f64,
}

/// Which text the CV chunks are compared against in the Match Lens
/// (CV Inspector). Serialized externally tagged: `"JobAd"`,
/// `{"Criterion": {"id": "..."}}`.
#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq, Eq)]
pub enum ChunkMatchTarget {
    /// The job advertisement text.
    JobAd,
    /// One criterion, by id.
    Criterion { id: String },
}

/// One CV chunk with its char range in the extracted text and its cosine
/// similarity to the target.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChunkSim {
    /// Char index of the chunk start in the candidate's extracted text.
    pub char_start: usize,
    /// Char index one past the chunk end.
    pub char_end: usize,
    /// Cosine similarity of the chunk embedding against the target embedding.
    pub similarity: f64,
}

/// Match Lens response: EVERY chunk of the candidate at the requested size,
/// in document order. The frontend picks the top-n and ranks, so changing n
/// re-ranks client-side without a refetch.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ChunkMatchResult {
    /// Model slug that produced the vectors.
    pub model: String,
    /// All chunks, in document order.
    pub chunks: Vec<ChunkSim>,
}

/// One row in the Jobs list — precomputed facts (no status enum).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct JobSummary {
    pub id: String,
    pub title: String,
    pub triqui_size: i32,
    pub archived: bool,
    pub cv_imported: i32,
    pub cv_failed: i32,
    pub want_count: i32,
    pub avoid_count: i32,
    pub ad_set: bool,
    pub importing: bool,
    pub last_ranked_at: Option<String>,
    pub created_at: String,
}

/// Full Job-view payload.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct JobDetail {
    pub job: Job,
    pub criteria: Vec<Criterion>,
    /// The results-table columns, in display order: the cascade's stage grades
    /// in pipeline order (lexical, scout, the job-ad cosine the heavy embedder
    /// gives, the judge's win-rate), then the rubric the total is made of
    /// (criteria and the two feedback rows).
    pub rubric: Vec<RubricQuestion>,
    /// Σ |max_points| over the questions that count towards `points` — the
    /// rubric's whole weight. Backend-derived so the sheet never has to
    /// re-decide which columns are rubric and which are stages. The sheet
    /// shows each score as `points / max_points × 100`, so it reads the
    /// Average out of 100 and does not need this capacity as a caption.
    #[serde(default)]
    pub rubric_capacity: f32,
    pub triqui: Vec<TriqiItem>,
    pub cv_imported: i32,
    pub cv_failed: i32,
    pub importing: bool,
}

/// Candidate overlay payload.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CandidateDetail {
    pub id: String,
    pub job_id: String,
    pub file_name: String,
    pub file_type: FileType,
    pub rank: Option<i32>,
    pub extraction_status: ExtractionStatus,
    pub extracted_text: Option<String>,
    pub source_file_path: String,
    pub original_available: bool,
    /// Total rubric points. None until ranked.
    #[serde(default)]
    pub points: Option<f32>,
    /// Sum of question max-points (the rubric capacity). None until ranked.
    #[serde(default)]
    pub points_max: Option<f32>,
    /// Per-question rubric breakdown. Empty until ranked.
    #[serde(default)]
    pub rubric_scores: Vec<RubricScore>,
    /// Whether the user has starred this CV (feedback signal).
    #[serde(default)]
    pub liked: bool,
    /// When the user last opened this CV in the inspector (read/unread
    /// marking; NULL = unread).
    #[serde(default)]
    pub viewed_at: Option<String>,
}

/// One row in the candidate browser (all of a job's candidates, not just the ranked
/// triqui). Carries the per-question breakdown for the spreadsheet columns.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CandidateSummary {
    pub id: String,
    pub file_name: String,
    pub file_type: FileType,
    pub extraction_status: ExtractionStatus,
    #[serde(default)]
    pub rank: Option<i32>,
    /// Total rubric points. None until ranked.
    #[serde(default)]
    pub points: Option<f32>,
    /// Sum of question max-points (the rubric capacity). None until ranked.
    #[serde(default)]
    pub points_max: Option<f32>,
    /// Per-question points (one row per rubric question). Empty until ranked.
    #[serde(default)]
    pub scores: Vec<RubricScore>,
    /// Which embedding space this candidate's criteria/feedback similarities come
    /// from. `None` until something has graded it at all.
    #[serde(default)]
    pub grade_space: Option<GradeSpace>,
    /// What backs the ad slot of this candidate's total.
    #[serde(default)]
    pub ad_source: Option<AdSource>,
    /// Whether the user has liked this CV (feedback signal).
    #[serde(default)]
    pub liked: bool,
    /// When the user last opened this CV in the inspector (read/unread
    /// marking; NULL = unread).
    #[serde(default)]
    pub viewed_at: Option<String>,
}

/// A page of candidate results for the browser.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CandidatePage {
    pub items: Vec<CandidateSummary>,
    pub total: i32,
}

/// Prev/next candidate ids in the job's global ordering (for overlay navigation
/// across all candidates, not just the triqui).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CandidateNeighbors {
    #[serde(default)]
    pub prev_id: Option<String>,
    #[serde(default)]
    pub next_id: Option<String>,
}

/// What the engine is working on right now — one human-readable status line.
/// Backend-derived: the gui renders it (i18n-mapping the stage code) and
/// never reconstructs it.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct Activity {
    /// Stable stage code: `scout` | `embed` | `judge` | `inspect`.
    pub stage: String,
    /// The thing being worked on — a CV file name, or `a.pdf vs b.pdf`.
    pub subject: String,
    /// Fine-grained position within the subject, e.g. `chunk 3/17` or
    /// `pair 210/1120`.
    #[serde(default)]
    #[specta(optional)]
    pub detail: Option<String>,
    /// Model slug serving this work (the embedder, the scout, or the judge's
    /// embedder).
    #[serde(default)]
    #[specta(optional)]
    pub model: Option<String>,
}

/// Snapshot of the modelcaller engine state for the UI.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RuntimeStatusInfo {
    /// `"Starting"` | `"Running"` | `"Error"` | `"Stopped"`.
    pub status: String,
    pub model: Option<String>,
    /// Vector dimension; `0` until the first embedding response.
    #[serde(default)]
    pub dim: u32,
    /// An embed is currently in flight (only meaningful while `Running`).
    #[serde(default)]
    pub busy: bool,
    #[serde(default)]
    pub error: Option<String>,
    /// The arm serving `model` as its load response reported it: `"ort"` or
    /// `"mlx"` (design 15 §15.4). `None` until the first load — the value
    /// comes from the arm, not from what the host asked for.
    #[serde(default)]
    #[specta(optional)]
    pub backend: Option<String>,
    /// What the engine is working on right now, when it is.
    #[serde(default)]
    #[specta(optional)]
    pub activity: Option<Activity>,
}

/// Which arm serves an embedder slug on this machine, and whether its weights
/// are on disk — resolved engine-free (no model load, no wrapper). The gui
/// uses this to name the effective variant in a consent-to-download prompt
/// and to show the arm next to the model.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct EmbedderVariants {
    pub model: String,
    /// `"ort"` or `"mlx"` — the arm `embed` would be served by.
    pub runtime: String,
    /// The effective arm's weights exist on disk.
    pub present: bool,
    /// The ONNX graph exists (the arm-independent fallback).
    pub onnx_present: bool,
    /// The MLX variant exists (`None` when the model declares no variant).
    #[serde(default)]
    #[specta(optional)]
    pub mlx_present: Option<bool>,
}

/// Backend-owned enablement state for a job's Re-rank button — the gui is a
/// pure projection of this (reason codes map to strings; it computes nothing).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct JobUiState {
    /// Engine status is `Running` (a rank may need it for re-embedding).
    pub engine_ready: bool,
    /// At least one extracted CV — without it ranking is meaningless.
    pub sort_allowed: bool,
    /// Ranking inputs changed since the last rank (or never ranked).
    /// This is the "Re-rank becomes active" signal.
    pub rerank_dirty: bool,
    /// Stable reason code for the current state, precedence:
    /// `no-cvs` > `no-jobad` > `engine-not-ready` > `stale` > `clean`.
    /// `no-jobad`: CVs are imported but the job has no job-ad text, so a rank
    /// has nothing to score against.
    pub reason: Option<String>,
    /// One-shot guidance code, backend-decided: `add-job-ad` when the job
    /// holds embedded CVs but no job ad **and** the import auto-start's
    /// embed-prefill cascade has converged — all the ad-independent work is
    /// done, so ranking is the only missing step. The gui surfaces this as a
    /// dismissible banner; `None` = nothing to nudge about.
    #[serde(default)]
    #[specta(optional)]
    pub nudge: Option<String>,
    /// Progressive-ranking cascade state (live run + last-rank coverage).
    #[serde(default = "default_rank_state")]
    pub rank: RankState,
}

fn default_rank_state() -> RankState {
    RankState {
        running: false,
        ranking: false,
        stage: None,
        wave: None,
        embedded_total: 0,
        embedded_done: 0,
        judged: 0,
        candidates: 0,
        converged: false,
        reason: None,
    }
}

/// Read-model of the last rank of a job: what the judge stage actually did.
/// `judge_mode`: `ran` (≥1 fresh verdict), `cached` (engaged, zero fresh —
/// every verdict memoized), `skipped` (stage not engaged).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SortReport {
    pub job_id: String,
    pub judge_mode: String,
    pub judge_pairs_fresh: u32,
    pub judge_pairs_cached: u32,
    /// How the last rank ran: `full` (one blocking pass) or `waves`
    /// (progressive cascade).
    #[serde(default)]
    pub mode: String,
    /// Waves completed by the last rank (0 for full passes).
    #[serde(default)]
    pub waves: u32,
    /// The last rank covered every candidate (not cancelled mid-run).
    #[serde(default)]
    pub complete: bool,
    /// The last rank was cancelled mid-run (partial coverage).
    #[serde(default)]
    pub cancelled: bool,
    pub created_at: String,
}

/// Backend-derived state of a job's progressive-ranking cascade — the gui is
/// a pure projection of this (reason codes map to strings; it computes
/// nothing).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RankState {
    /// A cascade run is active for this job right now. Note that the
    /// import auto-start runs ad-less jobs as an **embed-prefill** cascade
    /// (design §9), which reports `running: true` without producing a
    /// ranking — the button-facing flag is [`RankState::ranking`].
    pub running: bool,
    /// The active cascade is actually ranking this job: in flight **and**
    /// the job has a job ad to score against (an ad landing mid-run flips
    /// this on at the next poll). The rank buttons spin on this, not on
    /// [`RankState::running`] — background embedding of an ad-less job is
    /// status-bar activity, not a ranking in progress.
    #[serde(default)]
    pub ranking: bool,
    /// Live stage of the active run: `lex` | `embed` | `sort`.
    pub stage: Option<String>,
    /// Current wave number (1-based) while running waves.
    pub wave: Option<u32>,
    /// Candidates with embeddable text — the embed stage's total.
    pub embedded_total: u32,
    /// Candidates embedded with the current model so far.
    pub embedded_done: u32,
    /// Candidates in the judge pool so far.
    pub judged: u32,
    /// Extracted candidates overall.
    pub candidates: u32,
    /// The last completed rank covers every candidate and inputs are clean.
    pub converged: bool,
    /// Stable reason code for the cascade state, precedence:
    /// `running` > `cancelled` > `stale` > `converged` > `partial`.
    pub reason: Option<String>,
}

/// Progress event payload for a running cascade (`rank-progress` event).
/// Emitted per stage edge and throttled per wave; `running: false` marks the
/// terminal event (`stage`: `done` | `cancelled` | `error`).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct RankProgress {
    pub job_id: String,
    pub stage: String,
    pub wave: u32,
    pub embedded_done: u32,
    pub embedded_total: u32,
    pub judged: u32,
    pub candidates: u32,
    pub running: bool,
    /// Stable reason code while the run is alive but not making progress:
    /// `holding` = the system-capacity budget says hold (low memory —
    /// libsyscap design §4); the gui shows a visible notice. `None` while
    /// waves run normally.
    #[serde(default)]
    #[specta(optional)]
    pub reason: Option<String>,
}

/// Backend-derived snapshot of the system-capacity budget (libsyscap):
/// how much of the machine this serve may use right now. `holding: true`
/// (verdict `hold`) is the only state the gui must surface — a visible
/// notice; `polite`/`backoff` adapt silently. A general load indicator is
/// phase 2.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SystemBudget {
    /// `full` | `polite` | `backoff` | `hold` (snake_case verdict code).
    pub verdict: String,
    /// Cascade wave multiplier currently applied (1.0 / 0.5 / 0.25 / 0.0).
    pub wave_scale: f64,
    /// Effective intra-op thread cap the budget hands out.
    pub thread_cap: u32,
    /// Effective embed batch cap (`None` = config sets none — consumer
    /// default stays).
    #[specta(optional)]
    pub embed_batch: Option<u32>,
    /// Convenience mirror of `verdict == "hold"`.
    pub holding: bool,
    /// The load sample the verdict was computed from.
    pub load: SystemLoad,
}

/// One live-load sample (libsyscap `LoadSnapshot`), for provenance/UX.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SystemLoad {
    /// Whole-system CPU utilization, 0..=100.
    pub cpu_pct: f64,
    /// 1-minute load average.
    pub loadavg_1: f64,
    pub mem_available_bytes: u64,
    pub mem_total_bytes: u64,
    /// `None` when the host has no readable battery.
    #[specta(optional)]
    pub on_battery: Option<bool>,
}

/// Fallible-domain errors (typed for frontend matching via `#[serde(tag = "type")]`).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(tag = "type")]
pub enum DomainError {
    NotFound { entity: String, id: String },
    ValidationError { message: String },
    DbError { message: String },
    IoError { message: String },
    ParseError { message: String },
}

impl std::fmt::Display for DomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DomainError::NotFound { entity, id } => write!(f, "{entity} '{id}' not found"),
            DomainError::ValidationError { message } => write!(f, "Validation error: {message}"),
            DomainError::DbError { message } => write!(f, "Database error: {message}"),
            DomainError::IoError { message } => write!(f, "IO error: {message}"),
            DomainError::ParseError { message } => write!(f, "Parse error: {message}"),
        }
    }
}

// ============================================================================
// Models management (was gui `commands/models.rs`)
// ============================================================================

/// Which kind of model an operation targets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Type, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    Embedding,
    TextGeneration,
}

impl ModelKind {
    pub fn is_embedding(&self) -> bool {
        matches!(self, ModelKind::Embedding)
    }
}

/// Progress payload emitted during an ONNX model download. Mirrors the frontend's
/// `model-download-progress` listener (`repo_id`, `filename`, `done`, `total`).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct ModelDownloadProgress {
    pub repo_id: String,
    pub filename: String,
    pub done: u64,
    pub total: Option<u64>,
}

/// One ONNX embedder from `ikapp.yaml` `gui_embedders`, enriched with on-disk state.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct GuiEmbedder {
    /// Model slug (also the cache directory name), e.g. `nomic-embed-text-v1.5`.
    pub slug: String,
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Approximate RAM footprint in MB.
    pub memory_mb: u32,
    /// True when the ONNX files are present in the model cache.
    pub downloaded: bool,
    /// True when this is the active default embedder (per saved preferences).
    pub is_default: bool,
    /// Embedding dimension.
    pub dim: u32,
    /// Max context window in tokens.
    pub max_context_window: u32,
    /// Hugging Face source (`hf:org/repo`) used by `modeldownloader`.
    pub source: String,
    /// On-disk size of the cached model, in bytes (0 when not downloaded).
    pub size_bytes: u64,
}

/// Result of a single pair-wise cosine comparison.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CosinePair {
    pub i: usize,
    pub j: usize,
    pub label_i: String,
    pub label_j: String,
    pub similarity: f64,
}

/// Result of the "try model" benchmark.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct TryModelResult {
    /// Per-CV processing times in milliseconds (one per CV).
    pub per_cv_ms: Vec<f64>,
    /// Median processing time per CV in milliseconds.
    pub median_ms: f64,
    /// All pair-wise cosine similarities (45 pairs for 10 CVs).
    pub pairs: Vec<CosinePair>,
    /// CV labels (short descriptions for the matrix).
    pub labels: Vec<String>,
}

/// Per-CV progress event emitted during `try_model`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TryModelProgress {
    pub cv_index: usize,
    pub cv_total: usize,
    pub cv_label: String,
}

// ============================================================================
// Vetted embedders (was gui `commands/vetted_embedders.rs`)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct VettedEmbedder {
    pub slug: String,
    pub name: String,
    pub memory_mb: u32,
    pub dim: u32,
    pub max_context_window: u32,
    pub downloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct VettedEmbeddersResponse {
    pub embedders: Vec<VettedEmbedder>,
    pub system_ram_mb: u64,
    pub recommended_slug: String,
}

// ============================================================================
// Progress event payloads (were gui `commands/import.rs` / `commands/embed.rs`)
// ============================================================================

/// Payload for the `job-import-progress` event. Also reused by the job-bundle
/// import so both flows drive the same UI progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportProgress {
    pub job_id: String,
    pub done: i64,
    pub total: i64,
    pub failed: i64,
    pub active: bool,
    /// Which sub-phase is running: "extracting" | "embedding" | "ranking" |
    /// "engine-unavailable" | "done". Lets the UI show one continuous pipeline instead
    /// of declaring the import "finished" before embedding/ranking have run.
    pub phase: String,
}

/// Payload for the `job-embedding-progress` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProgress {
    pub job_id: String,
    pub done: i64,
    pub total: i64,
    pub failed: i64,
    pub active: bool,
}

// ============================================================================
// Results CSV export (was `libtriqui_core::results_csv::ResultsCsvSummary`)
// ============================================================================

/// Outcome of a results CSV export (surfaced as a toast).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Type)]
pub struct ResultsCsvSummary {
    /// Data rows written (ranked candidates).
    pub rows: i64,
    /// Rubric question columns (job ad + criteria).
    pub questions: usize,
}

// ============================================================================
// Job bundle export/import (contract DTOs from `libtriqui_core::job_bundle`)
// ============================================================================

/// What to include in an export. CV files are always included — without them
/// the bundle is just names and numbers, which the app cannot interpret (there
/// is no NER extracting candidate identity).
#[derive(Debug, Clone, Default, Serialize, Deserialize, Type)]
pub struct BundleExportOptions {
    #[serde(default)]
    pub include_ad: bool,
    #[serde(default)]
    pub include_weights: bool,
    #[serde(default)]
    pub include_results: bool,
    #[serde(default)]
    pub include_likes: bool,
    /// Ids of the criteria to include (order preserved).
    #[serde(default)]
    pub criterion_ids: Vec<String>,
    /// Export only the top N ranked candidates (CV files + result rows);
    /// `None` = all. Unranked candidates sort last and are dropped first.
    #[serde(default)]
    pub top_n: Option<u32>,
}

/// Outcome of an export (surfaced as a toast / log).
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct BundleExportSummary {
    pub candidates_total: i64,
    pub files_missing: i64,
    pub criteria_included: usize,
}
