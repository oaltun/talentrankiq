//! `modeldownloader` — ONNX model cache fetcher, as a library.
//!
//! There is no `modeldownloader` binary: every consumer calls the library
//! in-process — the gui's serve loop downloads embedders via
//! [`download_url`], modelcaller's auto-download via [`download_url`] too,
//! and benchdriverrs pulls bench models via
//! [`Downloader::run`]. Progress comes back typed through [`DownloadEvent`]
//! instead of parsed stderr log lines.

pub mod cli;
pub mod downloader;
pub mod hf_url;
pub mod modelcache;
pub mod source;

pub use downloader::{DownloadEvent, DownloadReport, Downloader, ProgressSink};
pub use modelcache::Exit;

use std::sync::{Arc, Mutex};

/// One URL-mode download: a HuggingFace blob url plus its destination slug —
/// the exact shape the gui embedder list and modelcaller's auto-download use.
/// Files land under `<model_dir>/<slug>/repo/` (the layout `resolve_onnx`
/// prefers), with `extra_files` (projection weights etc.) fetched alongside.
#[derive(Debug, Clone)]
pub struct UrlDownload {
    /// Full HF blob url to a single ONNX file (e.g. from `ikapp.yaml`
    /// `models[].onnx_url`).
    pub url: String,
    /// Directory slug (the model id) used under `model_dir`.
    pub slug: String,
    /// Cache root.
    pub model_dir: std::path::PathBuf,
    /// HF revision (tag/branch/commit).
    pub revision: String,
    /// Additional repo-relative files fetched with paths preserved.
    pub extra_files: Vec<String>,
    /// Re-download even if all files exist and match size.
    pub force: bool,
    /// sha256 checksum of the primary artifact (model.onnx).
    /// When present, the download is verified against this hash.
    pub sha256: Option<String>,
}

impl UrlDownload {
    pub fn new(url: impl Into<String>, slug: impl Into<String>, model_dir: std::path::PathBuf, sha256: Option<String>) -> Self {
        Self {
            url: url.into(),
            slug: slug.into(),
            model_dir,
            revision: "main".to_string(),
            extra_files: Vec::new(),
            force: false,
            sha256,
        }
    }
}

/// Run a URL-mode download in-process, forwarding structured progress to
/// `progress`. Returns the on-disk slug dir on success. Equivalent to the
/// binary's `--url/--slug/--model-dir/--extra-file` invocation.
pub fn download_url(
    req: UrlDownload,
    progress: impl FnMut(DownloadEvent) + Send + 'static,
) -> Result<std::path::PathBuf, Exit> {
    let cli = cli::Cli {
        config: None,
        model: None,
        source: None,
        url: Some(req.url),
        slug: Some(req.slug),
        revision: req.revision,
        model_dir: Some(req.model_dir),
        force: req.force,
        dry_run: false,
        files: vec![],
        extra_files: req.extra_files,
        // URL mode is the single-ONNX path by construction; an MLX fetch is a
        // repo fetch planned from the tree.
        artifact: cli::Artifact::Onnx,
        sha256: req.sha256,
        verbose: false,
        logfile: None,
    };
    let sink: ProgressSink = Arc::new(Mutex::new(progress));
    Downloader::run_with_progress(cli, Some(sink))
}
