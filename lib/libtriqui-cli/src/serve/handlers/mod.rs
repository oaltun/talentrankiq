//! Serve-side command handlers — one module per domain, mirroring the
//! pre-migration gui command modules. Every handler is a thin function over
//! `triqui-core` (plus file/system work that belongs outside a gui):
//! params deserialize from the shared [`libtriqui_client::params`] structs,
//! results serialize back as the `ok` payload, and fallible-domain errors are
//! wrapped so the gui shell can rehydrate the exact typed error the frontend
//! expects.

pub mod candidates;
pub mod criteria;
pub mod embed;
pub mod feedback;
pub mod import;
pub mod job_ad;
pub mod job_bundle;
pub mod jobs;
pub mod models;
pub mod prefs;
pub mod rank;
pub mod recovery;
pub mod results_csv;
pub mod sort;
pub mod vetted_embedders;

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use libtriqui_client::protocol::{ProtocolError, Request, Response};
use libtriqui_client::types::DomainError;

use crate::serve::ServeState;

/// Handle one request: dispatch on a dedicated thread (see `serve::run`),
/// converting panics into protocol errors so no waiter is ever left hanging.
pub fn handle(state: &Arc<ServeState>, request: &Request) -> Response {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        dispatch(state, &request.cmd, &request.params)
    }));
    match outcome {
        Ok(Ok(value)) => Response::ok(request.id, value),
        Ok(Err(err)) => Response::err(request.id, err),
        Err(_) => Response::err(
            request.id,
            ProtocolError::internal("internal error: handler panicked"),
        ),
    }
}

fn dispatch(state: &Arc<ServeState>, cmd: &str, params: &Value) -> Result<Value, ProtocolError> {
    use libtriqui_client::params::*;
    match cmd {
        // Preferences
        "load_preferences" => prefs::load_preferences(state),
        "save_preferences" => {
            let p: SavePreferencesParams = from_params(params)?;
            prefs::save_preferences(state, p.preferences).map(|_| Value::Null)
        }
        "get_cv_dir_root" => prefs::get_cv_dir_root(state),

        // Jobs
        "create_job" => jobs::create_job(state),
        "list_job_summaries" => jobs::list_job_summaries(state),
        "get_job_detail" => {
            let p: GetJobDetailParams = from_params(params)?;
            jobs::get_job_detail(state, &p.job_id)
        }
        "update_job" => {
            let p: UpdateJobParams = from_params(params)?;
            jobs::update_job(state, &p.job_id, &p.title, p.job_ad_text, p.triqui_size)
        }
        "update_job_weights" => {
            let p: UpdateJobWeightsParams = from_params(params)?;
            jobs::update_job_weights(
                state,
                &p.job_id,
                p.ad_weight,
                p.fb_job_weight,
                p.fb_profile_weight,
            )
        }
        "set_job_cv_dir" => {
            let p: SetJobCvDirParams = from_params(params)?;
            jobs::set_job_cv_dir(state, &p.job_id, &p.directory)
        }
        "watch_job" => {
            let p: WatchJobParams = from_params(params)?;
            jobs::watch_job(state, &p.job_id)
        }
        "archive_job" => {
            let p: ArchiveJobParams = from_params(params)?;
            jobs::archive_job(state, &p.job_id, p.archived).map(|_| Value::Null)
        }
        "empty_candidates" => {
            let p: EmptyCandidatesParams = from_params(params)?;
            jobs::empty_candidates(state, &p.job_id)
        }
        "delete_job" => {
            let p: DeleteJobParams = from_params(params)?;
            jobs::delete_job(state, &p.job_id).map(|_| Value::Null)
        }
        "job_ui_state" => {
            let p: GetJobDetailParams = from_params(params)?;
            jobs::job_ui_state(state, &p.job_id)
        }
        "get_sort_report" => {
            let p: GetJobDetailParams = from_params(params)?;
            jobs::get_sort_report(state, &p.job_id)
        }

        // Criteria
        "add_criterion" => {
            let p: AddCriterionParams = from_params(params)?;
            criteria::add_criterion(state, &p.job_id, &p.text, p.weight)
        }
        "update_criterion" => {
            let p: UpdateCriterionParams = from_params(params)?;
            criteria::update_criterion(state, &p.id, &p.text, p.weight)
        }
        "delete_criterion" => {
            let p: DeleteCriterionParams = from_params(params)?;
            criteria::delete_criterion(state, &p.id).map(|_| Value::Null)
        }

        // Import
        "import_cv_directory" => {
            let p: ImportCvDirectoryParams = from_params(params)?;
            import::import_cv_directory(state, &p.job_id, &p.directory)
        }

        // Job bundles
        "export_job_bundle" => {
            let p: ExportJobBundleParams = from_params(params)?;
            job_bundle::export_job_bundle(state, &p.job_id, &p.options, &p.out_path)
        }
        "import_job_bundle" => {
            let p: ImportJobBundleParams = from_params(params)?;
            job_bundle::import_job_bundle(state, &p.path)
        }

        // Job ad (a text file the user edits; no in-app editor)
        "get_job_ad" => {
            let p: GetJobAdParams = from_params(params)?;
            job_ad::get_job_ad(state, &p.job_id)
        }
        "set_job_ad_path" => {
            let p: SetJobAdPathParams = from_params(params)?;
            job_ad::set_job_ad_path(state, &p.job_id, &p.path)
        }
        "reload_job_ad" => {
            let p: ReloadJobAdParams = from_params(params)?;
            job_ad::reload_job_ad(state, &p.job_id)
        }

        // Results export
        "export_results_csv" => {
            let p: ExportResultsCsvParams = from_params(params)?;
            results_csv::export_results_csv(state, &p.job_id, &p.out_path)
        }

        // Ranking / candidates / feedback / cascade
        "sort_job" => {
            let p: SortJobParams = from_params(params)?;
            sort::sort_job(state, &p.job_id)
        }
        "start_rank" => {
            let p: StartRankParams = from_params(params)?;
            rank::start_rank(state, &p.job_id)
        }
        "cancel_rank" => {
            let p: CancelRankParams = from_params(params)?;
            rank::cancel_rank(state, &p.job_id)
        }
        "get_rank_state" => {
            let p: GetJobDetailParams = from_params(params)?;
            rank::get_rank_state(state, &p.job_id)
        }
        "system_budget" => rank::system_budget(state),
        "mark_candidate_viewed" => {
            let p: MarkCandidateViewedParams = from_params(params)?;
            rank::mark_candidate_viewed(state, &p.candidate_id)
        }
        "get_candidate" => {
            let p: GetCandidateParams = from_params(params)?;
            candidates::get_candidate(state, &p.candidate_id)
        }
        "list_candidates" => {
            let p: ListCandidatesParams = from_params(params)?;
            candidates::list_candidates(
                state, &p.job_id, p.offset, p.limit, p.query, p.sort_by, p.sort_dir,
            )
        }
        "candidate_neighbors" => {
            let p: CandidateNeighborsParams = from_params(params)?;
            candidates::candidate_neighbors(
                state,
                &p.job_id,
                &p.candidate_id,
                p.query,
                p.sort_by,
                p.sort_dir,
            )
        }
        "apply_feedback" => {
            let p: ApplyFeedbackParams = from_params(params)?;
            feedback::apply_feedback(
                state,
                &p.job_id,
                p.ordered_candidate_ids,
                p.liked_candidate_ids,
            )
        }
        "reset_feedback" => {
            let p: ResetFeedbackParams = from_params(params)?;
            feedback::reset_feedback(state, &p.job_id)
        }
        "list_liked_candidate_ids" => {
            let p: ListLikedCandidateIdsParams = from_params(params)?;
            feedback::list_liked_candidate_ids(state, &p.job_id)
        }

        // Engine runtime
        "start_runtime" => embed::start_runtime(state),
        "runtime_status" => embed::runtime_status(state),
        "embed_job" => {
            let p: EmbedJobParams = from_params(params)?;
            embed::embed_job(state, &p.job_id)
        }
        "chunk_match" => {
            let p: ChunkMatchParams = from_params(params)?;
            embed::chunk_match(state, &p)
        }

        // Models
        "list_embedders" => models::list_embedders(state),
        "model_variants" => {
            let p: ModelVariantsParams = from_params(params)?;
            models::model_variants(state, &p)
        }
        "download_embedder" => {
            let p: DownloadEmbedderParams = from_params(params)?;
            models::download_embedder(state, &p.slug).map(|_| Value::Null)
        }
        "remove_embedder" => {
            let p: RemoveEmbedderParams = from_params(params)?;
            models::remove_embedder(&p.slug).map(|_| Value::Null)
        }
        "set_default_model" => {
            let p: SetDefaultModelParams = from_params(params)?;
            models::set_default_model(state, p.kind, &p.repo_id).map(|_| Value::Null)
        }
        "set_embedding_model" => {
            let p: SetEmbeddingModelParams = from_params(params)?;
            models::set_embedding_model(state, &p.slug).map(|_| Value::Null)
        }
        "try_model" => {
            let p: TryModelParams = from_params(params)?;
            models::try_model(state, &p.slug)
        }

        // Vetted embedders (bench-leaderboard enriched cards)
        "list_vetted_embedders" => vetted_embedders::list_vetted_embedders(state),

        // Recovery
        "save_emergency_data" => {
            let p: SaveEmergencyDataParams = from_params(params)?;
            recovery::save_emergency_data(state, &p.filename, &p.data)
        }
        "load_emergency_data" => {
            let p: LoadEmergencyDataParams = from_params(params)?;
            recovery::load_emergency_data(state, &p.filename)
        }
        "cleanup_old_recovery_files" => recovery::cleanup_old_recovery_files(state),

        other => Err(ProtocolError::internal(format!(
            "unknown command '{other}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Handler helpers
// ---------------------------------------------------------------------------

fn from_params<T: DeserializeOwned>(params: &Value) -> Result<T, ProtocolError> {
    serde_json::from_value(params.clone())
        .map_err(|e| ProtocolError::internal(format!("bad params: {e}")))
}

/// Serialize an `ok` payload (any DTO from `libtriqui_client::types`).
pub fn ok_json<T: Serialize>(value: T) -> Result<Value, ProtocolError> {
    serde_json::to_value(value).map_err(|e| ProtocolError::internal(e.to_string()))
}

/// Wrap a fallible-domain error so the gui shell can rehydrate its type.
pub fn domain<E: Serialize + std::fmt::Display>(err: &E) -> ProtocolError {
    ProtocolError::domain(err)
}

/// Wrap a plain `String` error (commands whose frontend signature is
/// `Result<_, String>`).
pub fn string_err(message: impl Into<String>) -> ProtocolError {
    ProtocolError::string(message)
}

pub fn lock_err(e: impl std::fmt::Display) -> DomainError {
    DomainError::DbError {
        message: format!("DB lock poisoned: {e}"),
    }
}
