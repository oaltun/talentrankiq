//! Request `params` for every serve command, shared by both sides of the
//! protocol so neither can drift. Field names are the wire keys the gui shell
//! serializes; `triqui-cli` deserializes exactly these structs.
//!
//! Responses reuse [`crate::types`] DTOs; events reuse the progress payloads
//! there too.

use serde::{Deserialize, Serialize};

use crate::types::{AppPreferences, BundleExportOptions, ChunkMatchTarget, ModelKind};

macro_rules! params {
    (
        $(
            $(#[$meta:meta])*
            struct $name:ident { $( $(#[$fmeta:meta])* $field:ident : $ty:ty ),* $(,)? }
        )*
    ) => {
        $(
            $(#[$meta])*
            #[derive(Debug, Clone, Serialize, Deserialize)]
            #[serde(deny_unknown_fields)]
            pub struct $name { $( $(#[$fmeta:meta])* pub $field : $ty ),* }
        )*
    };
}

params! {
    /// No parameters.
    struct EmptyParams {}

    struct SavePreferencesParams { preferences: AppPreferences }

    struct GreetParams { name: String }

    struct CreateJobParams {}

    struct GetJobDetailParams { job_id: String }

    struct UpdateJobParams {
        job_id: String,
        title: String,
        job_ad_text: Option<String>,
        triqui_size: i32,
    }

    struct UpdateJobWeightsParams {
        job_id: String,
        ad_weight: f64,
        fb_job_weight: f64,
        fb_profile_weight: f64,
    }

    struct SetJobCvDirParams { job_id: String, directory: String }

    /// Make `job_id` the only watched job: drops all other CV-dir watchers,
    /// creates the job's directory if missing, and runs a catch-up import
    /// scan. The gui sends this whenever the viewed job changes.
    struct WatchJobParams { job_id: String }

    struct ArchiveJobParams { job_id: String, archived: bool }

    struct EmptyCandidatesParams { job_id: String }

    struct DeleteJobParams { job_id: String }

    struct AddCriterionParams {
        job_id: String,
        text: String,
        weight: Option<f64>,
    }

    struct UpdateCriterionParams { id: String, text: String, weight: f64 }

    struct DeleteCriterionParams { id: String }

    struct ImportCvDirectoryParams { job_id: String, directory: String }

    struct GetJobAdParams { job_id: String }

    struct SetJobAdPathParams { job_id: String, path: String }

    struct ReloadJobAdParams { job_id: String }

    struct ExportJobBundleParams {
        job_id: String,
        options: BundleExportOptions,
        out_path: String,
    }

    struct ImportJobBundleParams { path: String }

    struct ExportResultsCsvParams { job_id: String, out_path: String }

    struct SortJobParams { job_id: String }

    /// Start a progressive-ranking cascade run for the job (start/track: the
    /// response returns a run id immediately; progress arrives as
    /// `rank-progress` events; a second concurrent run for the same job is a
    /// validation error).
    struct StartRankParams { job_id: String }

    /// Request cancellation of the job's running cascade. The run stops at
    /// the next phase boundary and reports `cancelled` in its sort report.
    struct CancelRankParams { job_id: String }

    /// Stamp `viewed_at` on a candidate (CV Inspector opened / navigated to).
    struct MarkCandidateViewedParams { candidate_id: String }

    struct GetCandidateParams { candidate_id: String }

    struct ListCandidatesParams {
        job_id: String,
        offset: i32,
        limit: i32,
        query: Option<String>,
        sort_by: Option<String>,
        sort_dir: Option<bool>,
    }

    struct CandidateNeighborsParams {
        job_id: String,
        candidate_id: String,
        query: Option<String>,
        sort_by: Option<String>,
        sort_dir: Option<bool>,
    }

    struct ApplyFeedbackParams {
        job_id: String,
        ordered_candidate_ids: Vec<String>,
        liked_candidate_ids: Vec<String>,
    }

    struct ResetFeedbackParams { job_id: String }

    struct ListLikedCandidateIdsParams { job_id: String }

    struct EmbedJobParams { job_id: String }

    /// Match Lens request. `chunk_size` is the chunk size in tokens, bounded
    /// by `8 <= chunk_size <= 2048` on the serve side.
    struct ChunkMatchParams {
        candidate_id: String,
        target: ChunkMatchTarget,
        chunk_size: usize,
    }

    struct DownloadEmbedderParams { slug: String }

    struct RemoveEmbedderParams { slug: String }

    struct SetDefaultModelParams { kind: ModelKind, repo_id: String }

    struct SetEmbeddingModelParams { slug: String }

    /// Which arm would serve `model`, and are its weights on disk — engine-free.
    struct ModelVariantsParams { model: String }

    struct TryModelParams { slug: String }

    struct SaveEmergencyDataParams { filename: String, data: serde_json::Value }

    struct LoadEmergencyDataParams { filename: String }

    /// Activate a judge license with a license key (downloads + saves locally).
    struct LicenseActivateParams { license_key: String }

    /// Check whether a license authorizes a given artifact id.
    struct LicenseCheckArtifactParams { artifact_id: String }
}
