use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// The artifact shape a fetch plans for. Named rather than inferred from the
/// repo because both shapes can live in one HuggingFace repo family, and
/// because the verification step differs: an ONNX fetch must produce a
/// `.onnx`, an MLX fetch a `.safetensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Artifact {
    #[default]
    Onnx,
    Mlx,
}

#[derive(Parser, Debug)]
#[command(
    name = "modeldownloader",
    about = "Fetch model files (ONNX graphs, or MLX safetensors) into the modelcaller cache"
)]
pub struct Cli {
    /// path to shared ikapp.yaml config
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// logical model id (e.g. nomic-embed-text-v1). use with --source.
    #[arg(short, long)]
    pub model: Option<String>,

    /// where to fetch: hf:<org>/<repo>, https://..., file://...
    #[arg(short, long)]
    pub source: Option<String>,

    /// full HF blob url to a single ONNX file. use with --slug.
    /// e.g. https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/blob/main/onnx/model_fp16.onnx
    #[arg(long)]
    pub url: Option<String>,

    /// directory slug used as the model id when --url is given.
    /// files are written to <model-dir>/<slug>/repo/
    #[arg(long)]
    pub slug: Option<String>,

    /// HF revision (tag/branch/commit)
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// cache root
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// re-download even if all files exist and match size
    #[arg(long)]
    pub force: bool,

    /// show what would be downloaded without downloading
    #[arg(long)]
    pub dry_run: bool,

    /// specific files to fetch
    #[arg(long, value_delimiter = ',')]
    pub files: Vec<String>,

    /// additional repo-relative files fetched alongside the onnx, into the
    /// same model dir with paths preserved (e.g. `1_Dense/model.safetensors`
    /// projection weights the runtime applies post-forward). repeatable.
    #[arg(long = "extra-file")]
    pub extra_files: Vec<String>,

    /// which artifact shape to plan for when the repo listing decides the
    /// file list. `onnx` (default) is today's behaviour; `mlx` fetches a
    /// Metal-backend checkpoint instead — safetensors shards plus the config
    /// and tokenizer `libmlxembed` reads (design 15 §15.5.3).
    #[arg(long, value_enum, default_value_t = Artifact::Onnx)]
    pub artifact: Artifact,

    /// sha256 checksum of the primary artifact file (model.onnx or model.safetensors).
    /// When provided, the download is verified against this hash.
    #[arg(long)]
    pub sha256: Option<String>,

    /// enable verbose debug logging
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// write logs to this file instead of stderr
    #[arg(long)]
    pub logfile: Option<PathBuf>,
}

impl Cli {
    pub fn config_overrides(&self) -> libconfigreader::CliOverrides {
        let mut ovr = libconfigreader::CliOverrides::new();
        ovr.model_dir = self.model_dir.clone();
        ovr.verbose = if self.verbose { Some(true) } else { None };
        ovr.logfile = self.logfile.clone();
        ovr.revision = Some(self.revision.clone());
        ovr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_args() {
        let cli = Cli::try_parse_from([
            "modeldownloader",
            "--model",
            "nomic-embed-text-v1",
            "--source",
            "hf:nomic-ai/nomic-embed-text-v1-onnx",
            "--model-dir",
            "/tmp/mdcache",
        ])
        .unwrap();
        assert_eq!(cli.model.as_deref(), Some("nomic-embed-text-v1"));
        assert_eq!(
            cli.source.as_deref(),
            Some("hf:nomic-ai/nomic-embed-text-v1-onnx")
        );
        assert_eq!(cli.revision, "main");
        assert_eq!(
            cli.model_dir.as_deref(),
            Some(std::path::Path::new("/tmp/mdcache"))
        );
        assert!(!cli.force);
        assert!(!cli.dry_run);
        assert!(cli.files.is_empty());
    }

    #[test]
    fn parses_files_csv_and_force() {
        let cli = Cli::try_parse_from([
            "modeldownloader",
            "--model",
            "m",
            "--source",
            "file:///tmp/src",
            "--force",
            "--files",
            "model.onnx,config.json",
        ])
        .unwrap();
        assert!(cli.force);
        assert_eq!(
            cli.files,
            vec!["model.onnx".to_string(), "config.json".to_string()]
        );
    }

    #[test]
    fn rejects_missing_model_and_url() {
        let r = Cli::try_parse_from(["modeldownloader", "--source", "hf:org/repo"]);
        assert!(r.is_ok());
        let cli = r.unwrap();
        assert!(crate::downloader::Downloader::run(cli).is_err());
    }

    #[test]
    fn parses_extra_files_repeatable() {
        let cli = Cli::try_parse_from([
            "modeldownloader",
            "--url",
            "https://huggingface.co/org/repo/blob/main/model.onnx",
            "--slug",
            "m",
            "--extra-file",
            "1_Dense/model.safetensors",
            "--extra-file",
            "2_Dense/model.safetensors",
        ])
        .unwrap();
        assert_eq!(
            cli.extra_files,
            vec![
                "1_Dense/model.safetensors".to_string(),
                "2_Dense/model.safetensors".to_string()
            ]
        );
    }

    #[test]
    fn dry_run_flag() {
        let cli = Cli::try_parse_from([
            "modeldownloader",
            "--model",
            "m",
            "--source",
            "hf:org/repo",
            "--dry-run",
        ])
        .unwrap();
        assert!(cli.dry_run);
    }
}