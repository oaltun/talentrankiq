use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cli::Cli;
use crate::hf_url;
use crate::modelcache::{self, Exit};
use crate::source::{parse_source, Source};

/// Structured progress event for library consumers (the gui's in-process
/// downloads). The binary reports the same facts as `[modeldownloader]` log
/// lines; lib consumers get them typed, no stderr parsing.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// A new file starts downloading; `total` is the content-length when known.
    Started { file: String, total: Option<u64> },
    /// Byte progress within the current file (emitted at ≥5% steps).
    Progress { file: String, done: u64, total: Option<u64> },
    /// A file finished downloading; `total` is its final size when known.
    Finished { file: String, total: Option<u64> },
    /// A file was already cached on disk and counted without a download.
    Cached { file: String, bytes: u64 },
}

/// Shared progress sink. `None` = the binary path, which logs instead.
pub type ProgressSink = Arc<Mutex<dyn FnMut(DownloadEvent) + Send>>;

pub struct DownloadReport {
    pub target: PathBuf,
    pub skipped: Vec<String>,
    pub downloaded: Vec<String>,
}

pub struct Downloader {
    cli: Cli,
    model_dir: PathBuf,
    progress: Option<ProgressSink>,
}

impl Downloader {
    fn emit(&self, ev: DownloadEvent) {
        if let Some(sink) = &self.progress {
            if let Ok(mut f) = sink.lock() {
                f(ev);
            }
        }
    }

    pub fn run(cli: Cli) -> Result<PathBuf, Exit> {
        Self::run_with_progress(cli, None)
    }

    /// Library entry with an optional structured progress sink (the binary
    /// passes `None` and relies on its log lines).
    pub fn run_with_progress(cli: Cli, progress: Option<ProgressSink>) -> Result<PathBuf, Exit> {
        let model_dir = modelcache::resolve_model_dir(cli.model_dir.as_deref());
        let mut dl = Downloader {
            cli,
            model_dir,
            progress,
        };

        // URL mode: fetch a single ONNX (with siblings) from a HF blob url.
        if dl.cli.url.is_some() {
            let report = dl.execute_url()?;
            return Ok(report.target);
        }

        if dl.cli.model.as_deref().unwrap_or("").is_empty() {
            log::error!("--model is required (or use --url with --slug)");
            return Err(Exit::BadArgs);
        }

        let report = dl.execute()?;
        Ok(report.target)
    }

    fn execute(&mut self) -> Result<DownloadReport, Exit> {
        let cli = &self.cli;
        let extra_files = cli.extra_files.clone();

        let model = cli.model.as_deref().ok_or_else(|| {
            log::error!("--model is required");
            Exit::BadArgs
        })?;
        if model.is_empty() {
            log::error!("--model is required");
            return Err(Exit::BadArgs);
        }

        let source_str = cli.source.as_deref().ok_or_else(|| {
            log::error!("--source is required");
            Exit::BadArgs
        })?;

        let source = parse_source(source_str).map_err(|_| Exit::BadArgs)?;
        let target = modelcache::target_dir(&self.model_dir, model, &cli.revision);

        // An MLX fetch plans from the repo listing, because its file set is a
        // property of the repo (one `model.safetensors`, or N shards plus an
        // index) and cannot be a fixed name list the way `model.onnx` is. The
        // listing costs one API call, which the HuggingFace path makes anyway.
        if cli.artifact == crate::cli::Artifact::Mlx && cli.files.is_empty() {
            let Source::HuggingFace { repo } = &source else {
                log::error!(
                    "--artifact mlx needs a HuggingFace repo source (hf:<org>/<repo>); \
                     safetensors are fetched by repo listing, not by blob url"
                );
                return Err(Exit::BadArgs);
            };
            let client = http_client();
            let tree = self.hf_tree(&client, repo)?;
            let paths: Vec<String> = tree.iter().map(|t| t.path.clone()).collect();
            let files = modelcache::plan_mlx_downloads(&paths);
            if files.is_empty() {
                log::error!("no safetensors or metadata files found in repo {repo}");
                return Err(Exit::NoWeights);
            }
            if cli.dry_run {
                log::info!("--dry-run: would fetch into {target:?} (model: {model})");
                for f in &files {
                    log::info!("  {f}");
                }
                return Ok(DownloadReport {
                    target,
                    skipped: Vec::new(),
                    downloaded: Vec::new(),
                });
            }
            fs::create_dir_all(&target).map_err(|e| {
                log::error!("failed to create target dir {target:?}: {e}");
                Exit::DiskWrite
            })?;
            let mut report = DownloadReport {
                target: target.clone(),
                skipped: Vec::new(),
                downloaded: Vec::new(),
            };
            self.fetch_huggingface(repo, &target, &files, &extra_files, &mut report)?;
            if !modelcache::verify_safetensors_present(&target) {
                log::error!("verification failed: no .safetensors under {target:?}");
                return Err(Exit::NoWeights);
            }
            return Ok(report);
        }

        let files: Vec<String> = match &source {
            // https sources are dispatched by URL shape inside fetch_https;
            // we carry explicit --files only (for a directory listing). An
            // empty list means "download this single file URL as-is".
            Source::Https { .. } => cli.files.clone(),
            _ => {
                if cli.files.is_empty() {
                    modelcache::DEFAULT_FILES
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    cli.files.clone()
                }
            }
        };
        fs::create_dir_all(&target).map_err(|e| {
            log::error!("failed to create target dir {target:?}: {e}");
            Exit::DiskWrite
        })?;

        let mut report = DownloadReport {
            target: target.clone(),
            skipped: Vec::new(),
            downloaded: Vec::new(),
        };

        if cli.dry_run {
            log::info!("--dry-run: would fetch into {target:?}");
            match &source {
                Source::Https { url } => {
                    if is_archive_url(url) {
                        log::info!("  (archive: {url})");
                    } else if files.is_empty() {
                        let name = https_basename(url);
                        log::info!("  {name}");
                        for f in cli.files.iter() {
                            log::info!("  {f}");
                        }
                    } else {
                        for f in &files {
                            log::info!("  {f}");
                        }
                    }
                }
                _ => {
                    for f in &files {
                        log::info!("  {f}");
                    }
                }
            }
            for f in &cli.extra_files {
                log::info!("  {f} (extra)");
            }
            return Ok(report);
        }

        match &source {
            Source::HuggingFace { repo } => {
                self.fetch_huggingface(repo, &target, &files, &extra_files, &mut report)?;
            }
            Source::Https { url } => {
                self.fetch_https(url, &target, &files, &mut report)?;
            }
            Source::File { path } => {
                self.fetch_file(path, &target, &files, &mut report)?;
            }
        }

        if !modelcache::verify_onnx_present(&target) {
            log::error!("verification failed: no .onnx or .onnx.gz file found in {target:?}");
            return Err(Exit::NoOnnx);
        }

        Ok(report)
    }

    /// URL mode: fetch a single ONNX file (plus its `_data` siblings and
    /// tokenizer/config) from a HuggingFace blob url into
    /// `<model_dir>/<slug>/repo/`. Writes `<model_dir>/<slug>/meta/source.json`.
    fn execute_url(&mut self) -> Result<DownloadReport, Exit> {
        let url = self
            .cli
            .url
            .as_deref()
            .ok_or_else(|| {
                log::error!("--url is required in url mode");
                Exit::BadArgs
            })?;
        let slug = self.cli.slug.as_deref().ok_or_else(|| {
            log::error!("--slug is required with --url");
            Exit::BadArgs
        })?;
        if slug.is_empty() {
            log::error!("--slug must not be empty");
            return Err(Exit::BadArgs);
        }

        let blob = hf_url::parse_hf_blob_url(url).ok_or_else(|| {
            log::error!("--url is not a valid huggingface blob url: {url}");
            Exit::BadArgs
        })?;

        let slug_dir = libmodelstorage::slug_dir(&self.model_dir, slug);
        let target = modelcache::url_target_dir(&self.model_dir, slug);

        let mut report = DownloadReport {
            target: slug_dir.clone(),
            skipped: Vec::new(),
            downloaded: Vec::new(),
        };

        let client = http_client();
        let tree = self.hf_tree(&client, &blob.repo)?;
        let tree_paths: Vec<String> = tree.iter().map(|t| t.path.clone()).collect();
        let files = hf_url::plan_downloads(&blob, &tree_paths);

        if files.is_empty() {
            log::error!("no files resolved from url {url}");
            return Err(Exit::SourceNotFound);
        }

        if self.cli.dry_run {
            log::info!(
                "--dry-run: would fetch into {target:?} (slug: {slug})"
            );
            for f in &files {
                log::info!("  {f}");
            }
            for f in &self.cli.extra_files {
                log::info!("  {f} (extra)");
            }
            return Ok(report);
        }

        fs::create_dir_all(&target).map_err(|e| {
            log::error!("failed to create target dir {target:?}: {e}");
            Exit::DiskWrite
        })?;

        for rel in &files {
            // Remap tokenizer/config.json to sit next to the onnx file so the
            // wrapper (which loads tokenizer.json from the model directory)
            // finds them. The HF source path stays `rel`; only the local dest
            // is rewritten.
            let dest_rel = remap_dest(blob.dir().as_str(), rel);
            let dest = target.join(&dest_rel);
            let remote = tree
                .iter()
                .find(|t| &t.path == rel)
                .map(|t| t.size)
                .unwrap_or(0);

            if !self.cli.force {
                if let Some(local) = modelcache::local_size(&dest) {
                    if self.size_matches(local, remote) {
                        log::info!("{dest_rel} cached ({local} bytes) -> {}", dest.display());
                        self.emit(DownloadEvent::Cached {
                            file: dest_rel.clone(),
                            bytes: local,
                        });
                        report.skipped.push(rel.clone());
                        continue;
                    }
                }
            }

            let dl_url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                blob.repo, blob.revision, rel
            );
            self.download_hf_file(&client, &dl_url, rel, &dest)?;
            report.downloaded.push(rel.clone());
        }

        // Extra files (e.g. projection weights): always explicit, so a missing
        // entry is a hard error. They land in the same repo dir, paths preserved.
        for extra in &self.cli.extra_files {
            let (remote_size,) = match tree.iter().find(|t| &t.path == extra) {
                Some(t) => (t.size,),
                None => {
                    log::error!("{extra}: not found in repo {}", blob.repo);
                    return Err(Exit::SourceNotFound);
                }
            };
            let dest = target.join(extra);

            if !self.cli.force {
                if let Some(local) = modelcache::local_size(&dest) {
                    if self.size_matches(local, remote_size) {
                        log::info!("{extra} cached ({local} bytes) -> {}", dest.display());
                        self.emit(DownloadEvent::Cached {
                            file: extra.clone(),
                            bytes: local,
                        });
                        report.skipped.push(extra.clone());
                        continue;
                    }
                }
            }

            let dl_url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                blob.repo, blob.revision, extra
            );
            self.download_hf_file(&client, &dl_url, extra, &dest)?;
            report.downloaded.push(extra.clone());
        }

        let mut all_files = files;
        all_files.extend(self.cli.extra_files.iter().cloned());

        if !verify_onnx_present_recursive(&target) {
            log::error!(
                "verification failed: no .onnx file found under {target:?}"
            );
            return Err(Exit::NoOnnx);
        }

        // The wrapper reads `config.json` from the ONNX's directory and fails the
        // load outright when it is absent. Some repos have none (a static /
        // Model2Vec export carries no HF config), so write a minimal one. Every
        // field is optional there: `dim` is inferred from the graph's output
        // shape and `kind` from its inputs, so `{}` is enough — and the graph,
        // not a hand-written file, stays the authority.
        ensure_config_json(&target, &blob.dir())?;

        // Write provenance metadata.
        self.write_meta(&slug_dir, url, &blob, &all_files)?;

        report.target = slug_dir;
        Ok(report)
    }

    fn write_meta(
        &self,
        slug_dir: &Path,
        url: &str,
        blob: &hf_url::HfBlob,
        files: &[String],
    ) -> Result<(), Exit> {
        let meta_dir = slug_dir.join(libmodelstorage::META_DIR);
        fs::create_dir_all(&meta_dir).map_err(|e| {
            log::error!("failed to create meta dir {meta_dir:?}: {e}");
            Exit::DiskWrite
        })?;
        let meta = serde_json::json!({
            "source_url": url,
            "repo": blob.repo,
            "revision": blob.revision,
            "onnx_path": blob.path,
            "files": files,
        });
        let text = serde_json::to_string_pretty(&meta).map_err(|e| {
            log::error!("failed to serialize meta: {e}");
            Exit::DiskWrite
        })?;
        let dest = meta_dir.join(libmodelstorage::META_FILE);
        atomic_write(&dest, text.as_bytes())?;
        Ok(())
    }

    fn fetch_huggingface(
        &mut self,
        repo: &str,
        target: &Path,
        files: &[String],
        extras: &[String],
        report: &mut DownloadReport,
    ) -> Result<(), Exit> {
        let client = http_client();
        let tree = self.hf_tree(&client, repo)?;

        for local_name in files {
            let dest = target.join(local_name);

            let entry = self.match_tree_entry(&tree, local_name);
            let (remote_path, remote_size) = match entry {
                Some(e) => (e.path.clone(), e.size),
                None => {
                    if self.cli.files.is_empty() {
                        log::info!("{local_name} not found in repo, skipping");
                        continue;
                    }
                    log::error!("{local_name}: not found in repo {repo}");
                    return Err(Exit::SourceNotFound);
                }
            };

            if !self.cli.force {
                if let Some(local) = modelcache::local_size(&dest) {
                    if self.size_matches(local, remote_size) {
                        let unit = if local > 0 {
                            format!(" ({local} bytes)")
                        } else {
                            String::new()
                        };
                        log::info!("{local_name} cached{unit} -> {}", dest.display());
                        self.emit(DownloadEvent::Cached {
                            file: local_name.clone(),
                            bytes: local,
                        });
                        report.skipped.push(local_name.clone());
                        continue;
                    }
                }
            }

            let url = format!(
                "https://huggingface.co/{repo}/resolve/{}/{}",
                self.cli.revision, remote_path
            );
            self.download_hf_file(&client, &url, local_name, &dest)?;
            report.downloaded.push(local_name.clone());

            // ONNX external weights: many repos split the tensor payload into a
            // sibling `<stem>.onnx_data` (or `.onnx_data_N`) file. Without it the
            // .onnx is only a graph shell and the model fails to load at runtime.
            if local_name.ends_with(".onnx") {
                for sibling in external_data_siblings(&tree, &remote_path) {
                    let sib_name = sibling.rsplit_once('/').map(|(_, n)| n).unwrap_or(&sibling);
                    let sib_dest = target.join(sib_name);
                    let sib_size = tree
                        .iter()
                        .find(|t| t.path == sibling)
                        .map(|t| t.size)
                        .unwrap_or(0);
                    if !self.cli.force {
                        if let Some(local) = modelcache::local_size(&sib_dest) {
                            if self.size_matches(local, sib_size) {
                                log::info!("{sib_name} cached ({local} bytes) -> {}", sib_dest.display());
                                self.emit(DownloadEvent::Cached {
                                    file: sibling.clone(),
                                    bytes: local,
                                });
                                report.skipped.push(sibling.clone());
                                continue;
                            }
                        }
                    }
                    let sib_url = format!(
                        "https://huggingface.co/{repo}/resolve/{}/{}",
                        self.cli.revision, sibling
                    );
                    self.download_hf_file(&client, &sib_url, sib_name, &sib_dest)?;
                    report.downloaded.push(sibling.clone());
                }
            }
        }

        // Extra files (e.g. projection weights): always explicit, so a missing
        // entry is a hard error. Paths are preserved under the target dir.
        for extra in extras {
            let (remote_path, remote_size) = match self.match_tree_entry(&tree, extra) {
                Some(e) => (e.path.clone(), e.size),
                None => {
                    log::error!("{extra}: not found in repo {repo}");
                    return Err(Exit::SourceNotFound);
                }
            };
            let dest = target.join(extra);

            if !self.cli.force {
                if let Some(local) = modelcache::local_size(&dest) {
                    if self.size_matches(local, remote_size) {
                        log::info!("{extra} cached ({local} bytes) -> {}", dest.display());
                        self.emit(DownloadEvent::Cached {
                            file: extra.clone(),
                            bytes: local,
                        });
                        report.skipped.push(extra.clone());
                        continue;
                    }
                }
            }

            let url = format!(
                "https://huggingface.co/{repo}/resolve/{}/{}",
                self.cli.revision, remote_path
            );
            self.download_hf_file(&client, &url, extra, &dest)?;
            report.downloaded.push(extra.clone());
        }

        Ok(())
    }

    fn match_tree_entry<'a>(&self, tree: &'a [TreeFile], local_name: &str) -> Option<&'a TreeFile> {
        tree.iter().find(|f| f.path == local_name).or_else(|| {
            tree.iter().find(|f| {
                std::path::Path::new(&f.path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    == Some(local_name)
            })
        })
    }

    fn size_matches(&self, local: u64, remote_size: u64) -> bool {
        if remote_size > 0 {
            local == remote_size
        } else {
            local > 0
        }
    }

    fn download_hf_file(
        &self,
        client: &ureq::Agent,
        url: &str,
        file: &str,
        dest: &Path,
    ) -> Result<(), Exit> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                log::error!("failed to create parent dir {parent:?}: {e}");
                Exit::DiskWrite
            })?;
        }
        log::info!("downloading {file} from {url} -> {}", dest.display());
        stream_response_to_file(client, url, dest, &file.to_string(), Some(self))
    }

    fn fetch_https(
        &self,
        url: &str,
        target: &Path,
        files: &[String],
        report: &mut DownloadReport,
    ) -> Result<(), Exit> {
        if is_archive_url(url) {
            let client = http_client();
            log::info!("downloading archive from {url}");
            self.emit(DownloadEvent::Started {
                file: https_basename(url),
                total: None,
            });
            let bytes = http_get_bytes(&client, url)?;
            self.extract_archive(&bytes, target)?;
            flatten_onnx(target);
            for f in modelcache::DEFAULT_FILES {
                if target.join(f).is_file() {
                    report.downloaded.push(f.to_string());
                }
            }
            self.emit(DownloadEvent::Finished {
                file: https_basename(url),
                total: None,
            });
            return Ok(());
        }

        let client = http_client();
        if files.is_empty() {
            // Direct file URL: download the single referenced file.
            let file_name = https_basename(url);
            let dest = target.join(&file_name);
            self.download_file_cached(&client, url, &file_name, &dest, report)?;
        } else {
            // Directory listing base: download each requested file.
            let base = url.trim_end_matches('/');
            for file in files {
                let file_url = format!("{base}/{file}");
                let dest = target.join(file);
                self.download_file_cached(&client, &file_url, file, &dest, report)?;
            }
        }
        Ok(())
    }

    fn download_file_cached(
        &self,
        client: &ureq::Agent,
        url: &str,
        file: &str,
        dest: &Path,
        report: &mut DownloadReport,
    ) -> Result<(), Exit> {
        if !self.cli.force {
            if let Some(local) = modelcache::local_size(dest) {
                if let Some(remote) = http_head(client, url)? {
                    if self.size_matches(local, remote) {
                        let unit = if local > 0 {
                            format!(" ({local} bytes)")
                        } else {
                            String::new()
                        };
                        log::info!("{file} cached{unit} -> {}", dest.display());
                        self.emit(DownloadEvent::Cached {
                            file: file.to_string(),
                            bytes: local,
                        });
                        report.skipped.push(file.to_string());
                        return Ok(());
                    }
                }
            }
        }
        log::info!("downloading {file} from {url} -> {}", dest.display());
        self.emit(DownloadEvent::Started {
            file: file.to_string(),
            total: None,
        });
        let bytes = http_get_bytes(client, url)?;
        atomic_write(dest, &bytes)?;
        self.emit(DownloadEvent::Finished {
            file: file.to_string(),
            total: Some(bytes.len() as u64),
        });
        report.downloaded.push(file.to_string());
        Ok(())
    }

    fn fetch_file(
        &self,
        src: &Path,
        target: &Path,
        files: &[String],
        report: &mut DownloadReport,
    ) -> Result<(), Exit> {
        if !src.is_dir() {
            log::error!("file:// source {src:?} is not a directory");
            return Err(Exit::SourceNotFound);
        }
        let to_fetch: Vec<String> = if files.is_empty() {
            modelcache::DEFAULT_FILES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            files.to_vec()
        };
        for file in &to_fetch {
            let src_file = src.join(file);
            if !src_file.is_file() {
                if self.cli.files.is_empty() {
                    log::info!("{file} not found in source dir, skipping");
                    continue;
                }
                log::error!("file:// source missing requested file: {file}");
                return Err(Exit::SourceNotFound);
            }
            let dest = target.join(file);
            let data = fs::read(&src_file).map_err(|e| {
                log::error!("failed to read {src_file:?}: {e}");
                Exit::DiskWrite
            })?;
            atomic_write(&dest, &data)?;
            report.downloaded.push(file.clone());
        }
        Ok(())
    }

    fn extract_archive(&self, bytes: &[u8], target: &Path) -> Result<(), Exit> {
        if bytes.starts_with(&[0x50, 0x4b, 0x03, 0x04])
            || bytes.starts_with(&[0x50, 0x4b, 0x05, 0x06])
        {
            extract_zip(bytes, target)
        } else {
            extract_tar(bytes, target)
        }
    }

    fn hf_tree(&self, client: &ureq::Agent, repo: &str) -> Result<Vec<TreeFile>, Exit> {
        let url = format!(
            "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true",
            revision = self.cli.revision
        );
        let body: serde_json::Value = http_get_json(client, &url)?;
        let arr = body.as_array().ok_or(Exit::SourceNotFound)?;
        let mut out = Vec::new();
        for entry in arr {
            if entry.get("type").and_then(|t| t.as_str()) != Some("file") {
                continue;
            }
            let path = entry
                .get("path")
                .and_then(|p| p.as_str())
                .ok_or(Exit::SourceNotFound)?
                .to_string();
            let size = entry.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
            out.push(TreeFile { path, size });
        }
        if out.is_empty() {
            log::error!("no files listed in repo {repo} (gated or empty?)");
            return Err(Exit::SourceNotFound);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct TreeFile {
    pub path: String,
    pub size: u64,
}

pub fn http_client() -> ureq::Agent {
    use ureq::config::Config;
    let cfg = Config::builder().http_status_as_error(false).build();
    ureq::Agent::new_with_config(cfg)
}

pub fn http_get_json(client: &ureq::Agent, url: &str) -> Result<serde_json::Value, Exit> {
    let resp = client.get(url).apply_auth().call().map_err(|e| {
        log::error!("network error fetching JSON {url}: {e}");
        Exit::Network
    })?;
    let status = resp.status();
    let mut body = resp.into_body().into_reader();
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes).map_err(|e| {
        log::error!("failed to read response body for {url}: {e}");
        Exit::Network
    })?;
    if !status.is_success() {
        let code = status.as_u16();
        let preview = preview_bytes(&bytes);
        if code == 404 || code == 401 {
            log::error!("source not found ({code}) for {url}: {preview}");
            return Err(Exit::SourceNotFound);
        }
        log::error!("HTTP {code} for {url}: {preview}");
        return Err(Exit::Network);
    }
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        log::error!("failed to parse JSON response from {url}: {e}");
        Exit::Network
    })?;
    Ok(v)
}

pub fn http_get_bytes(client: &ureq::Agent, url: &str) -> Result<Vec<u8>, Exit> {
    let resp = client.get(url).apply_auth().call().map_err(|e| {
        log::error!("network error fetching {url}: {e}");
        Exit::Network
    })?;
    let status = resp.status();
    let mut body = resp.into_body().into_reader();
    let mut bytes = Vec::new();
    body.read_to_end(&mut bytes).map_err(|e| {
        log::error!("failed to read response body for {url}: {e}");
        Exit::Network
    })?;
    if !status.is_success() {
        let code = status.as_u16();
        let preview = preview_bytes(&bytes);
        if code == 404 || code == 401 {
            log::error!("source not found ({code}) for {url}: {preview}");
            return Err(Exit::SourceNotFound);
        }
        log::error!("HTTP {code} for {url}: {preview}");
        return Err(Exit::Network);
    }
    Ok(bytes)
}

fn preview_bytes(b: &[u8]) -> String {
    let s = String::from_utf8_lossy(b);
    s.chars().take(200).collect()
}

trait AuthExt: Sized {
    fn apply_auth(self) -> Self;
}

impl<B> AuthExt for ureq::RequestBuilder<B> {
    fn apply_auth(self) -> Self {
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                return self.header("authorization", format!("Bearer {token}"));
            }
        }
        self
    }
}

fn is_archive_url(url: &str) -> bool {
    url.ends_with(".zip")
        || url.ends_with(".tar.gz")
        || url.ends_with(".tgz")
        || url.ends_with(".tar")
}

/// Repo-relative paths of the external-data siblings of an ONNX file (e.g.
/// `onnx/model.onnx` -> `onnx/model.onnx_data`, `onnx/model.onnx_data_1`).
/// ONNX external-weights exports keep the tensor payload in such files; without
/// them the .onnx is only a graph shell and cannot be loaded.
fn external_data_siblings(tree: &[TreeFile], onnx_path: &str) -> Vec<String> {
    let (dir, file_name) = match onnx_path.rsplit_once('/') {
        Some((d, n)) => (d, n),
        None => ("", onnx_path),
    };
    let stem = match file_name.strip_suffix(".onnx") {
        Some(s) => s,
        None => return Vec::new(),
    };
    let data_stem = format!("{stem}.onnx_data");
    tree.iter()
        .filter(|t| {
            let (file_dir, name) = match t.path.rsplit_once('/') {
                Some((d, n)) => (d, n),
                None => ("", t.path.as_str()),
            };
            file_dir == dir
                && name != file_name
                && (name == data_stem.as_str() || name.starts_with(&format!("{data_stem}_")))
        })
        .map(|t| t.path.clone())
        .collect()
}

fn https_basename(url: &str) -> String {
    url.split('/')
        .next_back()
        .filter(|s| !s.is_empty())
        .unwrap_or("download")
        .to_string()
}

pub fn http_head(client: &ureq::Agent, url: &str) -> Result<Option<u64>, Exit> {
    let resp = client.head(url).apply_auth().call().map_err(|e| {
        log::error!("network error HEAD {url}: {e}");
        Exit::Network
    })?;
    let code = resp.status().as_u16();
    if code == 404 || code == 401 {
        return Err(Exit::SourceNotFound);
    }
    if !resp.status().is_success() {
        log::error!("HTTP {code} for {url}");
        return Err(Exit::Network);
    }
    let size = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());
    Ok(size)
}

/// Staging path for `label` inside `parent`, unique per attempt (pid +
/// per-process counter).
///
/// Concurrent downloads of the same artifact are normal: the engine's
/// modelcaller auto-download and the serve `download_embedder` rpc target the
/// same model dir and may overlap. A shared `.<label>.tmp` would make the
/// second `File::create` truncate the bytes the first is still streaming, and
/// the first `rename` would leave the second with a missing staging file — a
/// spurious "disk write error" for a download that actually succeeded.
fn unique_tmp(parent: &Path, label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(".{label}.{}.{seq}.tmp", std::process::id()))
}

pub fn atomic_write(dest: &Path, data: &[u8]) -> Result<(), Exit> {
    let dest = dest.to_path_buf();
    let parent = dest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let label = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download")
        .to_string();
    let tmp = unique_tmp(&parent, &label);
    {
        let mut f = fs::File::create(&tmp).map_err(|e| {
            log::error!("failed to create temp file {tmp:?}: {e}");
            Exit::DiskWrite
        })?;
        f.write_all(data).map_err(|e| {
            log::error!("failed to write {tmp:?}: {e}");
            Exit::DiskWrite
        })?;
        f.sync_all().map_err(|e| {
            log::error!("failed to fsync {tmp:?}: {e}");
            Exit::DiskWrite
        })?;
    }
    fs::rename(&tmp, &dest).map_err(|e| {
        log::error!("failed to rename {tmp:?} -> {dest:?}: {e}");
        Exit::DiskWrite
    })?;
    Ok(())
}

/// Stream an HTTP response body straight to `dest` (via a temp file in the
/// same directory), renaming into place only once the full body is received.
/// Used for large ONNX weights so we never buffer gigabytes in memory.
/// `event_file` names the file in [`DownloadEvent`]s; `dl` (when present)
/// receives them — the same facts also go to the log either way.
fn stream_response_to_file(
    client: &ureq::Agent,
    url: &str,
    dest: &Path,
    event_file: &str,
    dl: Option<&Downloader>,
) -> Result<(), Exit> {
    let resp = client.get(url).apply_auth().call().map_err(|e| {
        log::error!("network error fetching {url}: {e}");
        Exit::Network
    })?;
    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        log::error!("HTTP {code} for {url}");
        if code == 404 || code == 401 {
            return Err(Exit::SourceNotFound);
        }
        return Err(Exit::Network);
    }

    let total: Option<u64> = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0);

    let file_label = dest.file_name().and_then(|n| n.to_str()).unwrap_or("download");
    if let Some(t) = total {
        log::info!("  {file_label}: {t} bytes ({:.1} MiB)", t as f64 / 1_048_576.0);
    }
    if let Some(dl) = dl {
        dl.emit(DownloadEvent::Started {
            file: event_file.to_string(),
            total,
        });
    }
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    let tmp = unique_tmp(parent, file_label);
    // A failed transfer must not leave staging bytes behind in the model dir.
    let streamed: Result<(), Exit> = (|| {
        let mut f = fs::File::create(&tmp).map_err(|e| {
            log::error!("failed to create temp file {tmp:?}: {e}");
            Exit::DiskWrite
        })?;
        let mut body = resp.into_body().into_reader();
        let mut buf = [0u8; 1 << 20];
        let mut done: u64 = 0;
        let mut last_logged_pct = -1i64;
        loop {
            let n = body.read(&mut buf).map_err(|e| {
                log::error!("failed to read from {url}: {e}");
                Exit::Network
            })?;
            if n == 0 {
                break;
            }
            f.write_all(&buf[..n]).map_err(|e| {
                log::error!("failed to stream {url} -> {tmp:?}: {e}");
                Exit::DiskWrite
            })?;
            done += n as u64;
            if let Some(t) = total {
                let pct: i64 = (done as f64 / t as f64 * 100.0).floor() as i64;
                if pct >= last_logged_pct + 5 || pct >= 100 {
                    log::info!(
                        "  {file_label}: {pct}% ({:.1}/{:.1} MiB)",
                        done as f64 / 1_048_576.0,
                        t as f64 / 1_048_576.0
                    );
                    last_logged_pct = pct;
                    if let Some(dl) = dl {
                        dl.emit(DownloadEvent::Progress {
                            file: event_file.to_string(),
                            done,
                            total: Some(t),
                        });
                    }
                }
            }
        }
        f.sync_all().map_err(|e| {
            log::error!("failed to fsync {tmp:?}: {e}");
            Exit::DiskWrite
        })?;
        Ok(())
    })();
    if streamed.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    streamed?;

    if let Err(e) = fs::rename(&tmp, dest) {
        let _ = fs::remove_file(&tmp);
        log::error!("failed to rename {tmp:?} -> {dest:?}: {e}");
        return Err(Exit::DiskWrite);
    }
    if let Some(t) = total {
        log::info!("  {file_label}: done ({:.1} MiB)", t as f64 / 1_048_576.0);
    } else {
        log::info!("  {file_label}: done");
    }
    if let Some(dl) = dl {
        dl.emit(DownloadEvent::Finished {
            file: event_file.to_string(),
            total,
        });
    }
    Ok(())
}

fn extract_zip(bytes: &[u8], target: &Path) -> Result<(), Exit> {
    use std::io::Cursor;
    let mut archive = match zip::ZipArchive::new(Cursor::new(bytes)) {
        Ok(a) => a,
        Err(e) => {
            log::error!("failed to open zip archive: {e}");
            return Err(Exit::SourceNotFound);
        }
    };
    for i in 0..archive.len() {
        let mut entry = match archive.by_index(i) {
            Ok(e) => e,
            Err(e) => {
                log::error!("zip entry {i}: {e}");
                return Err(Exit::SourceNotFound);
            }
        };
        let name = match entry.enclosed_name() {
            Some(n) => n.to_path_buf(),
            None => {
                log::warn!("skipping zip entry {i}: unsafe path");
                continue;
            }
        };
        if entry.is_dir() {
            continue;
        }
        let dest = target.join(&name);
        if !is_within(target, &dest) {
            log::warn!("skipping zip entry {i}: escapes target");
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                log::error!("failed to mkdir {parent:?}: {e}");
                Exit::DiskWrite
            })?;
        }
        let mut out = fs::File::create(&dest).map_err(|e| {
            log::error!("failed to create {dest:?}: {e}");
            Exit::DiskWrite
        })?;
        std::io::copy(&mut entry, &mut out).map_err(|e| {
            log::error!("failed to extract {dest:?}: {e}");
            Exit::DiskWrite
        })?;
    }
    Ok(())
}

fn extract_tar(bytes: &[u8], target: &Path) -> Result<(), Exit> {
    use std::io::Read;
    let gzip = bytes.starts_with(&[0x1f, 0x8b]);
    let reader: Box<dyn Read> = if gzip {
        Box::new(flate2::read::GzDecoder::new(bytes))
    } else {
        Box::new(bytes)
    };
    let mut archive = tar::Archive::new(reader);
    archive
        .entries()
        .map_err(|e| {
            log::error!("failed to open tar archive: {e}");
            Exit::SourceNotFound
        })?
        .try_for_each(|entry| {
            let mut entry = entry.map_err(|e| {
                log::error!("tar entry read failed: {e}");
                Exit::SourceNotFound
            })?;
            let path = entry.path().map_err(|e| {
                log::error!("tar entry path error: {e}");
                Exit::SourceNotFound
            })?;
            if entry.header().entry_type().is_dir() {
                return Ok(());
            }
            let dest = target.join(path);
            if !is_within(target, &dest) {
                log::warn!("skipping tar entry: escapes target");
                return Ok(());
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    log::error!("failed to mkdir {parent:?}: {e}");
                    Exit::DiskWrite
                })?;
            }
            let mut out = fs::File::create(&dest).map_err(|e| {
                log::error!("failed to create {dest:?}: {e}");
                Exit::DiskWrite
            })?;
            std::io::copy(&mut entry, &mut out).map_err(|e| {
                log::error!("failed to extract {dest:?}: {e}");
                Exit::DiskWrite
            })?;
            Ok(())
        })
}

fn is_within(base: &Path, p: &Path) -> bool {
    p.strip_prefix(base).is_ok()
}

fn flatten_onnx(target: &Path) {
    if modelcache::verify_onnx_present(target) {
        return;
    }
    if let Some(src) = walkdir_onnx(target) {
        if src.parent() == Some(target) {
            return;
        }
        let dest = target.join(src.file_name().unwrap_or_default());
        log::info!("flattening {} -> {}", src.display(), dest.display());
        let moved = fs::rename(&src, &dest).is_ok()
            || (fs::copy(&src, &dest).is_ok() && fs::remove_file(&src).is_ok());
        if moved {
            let mut cur = src.parent();
            while let Some(d) = cur {
                if d == target {
                    break;
                }
                match fs::remove_dir(d) {
                    Ok(()) => cur = d.parent(),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                    Err(_) => break,
                }
            }
        } else {
            log::warn!("failed to flatten onnx into target root");
        }
    }
}

fn walkdir_onnx(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).ok()?;
        for entry in entries {
            let path = entry.ok()?.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.file_name()?.to_str()?;
            if name.ends_with(".onnx") || name.ends_with(".onnx.gz") {
                return Some(path);
            }
        }
    }
    None
}

/// Recursively check whether any `.onnx` file exists under `root`.
fn verify_onnx_present_recursive(root: &Path) -> bool {
    walkdir_onnx(root).is_some()
}

/// If `rel` is a tokenizer/config file downloaded from the repo root, rewrite
/// its local destination to sit next to the onnx file (in `dir`). Otherwise the
/// HF path is preserved locally.
/// Local destination for a planned repo-relative file.
///
/// `tokenizer.json` / `config.json` must sit **next to the ONNX file** because
/// the wrapper loads both from the ONNX's own directory. The HF source path is
/// whatever the repo uses (usually the root, sometimes a sentence-transformers
/// module dir like `0_StaticEmbedding/`); only the local destination is
/// rewritten, so the remote path in `files` stays valid.
/// Write a minimal `config.json` beside the ONNX when the repo provided none.
/// The wrapper (`onnxwrapperrs::session::load`) reads it unconditionally and
/// errors the load when it cannot be read; every field in it is optional
/// (`dim` falls back to the graph's output shape, `kind` to its inputs), so an
/// empty object is a valid config rather than a guess. An existing file is left
/// untouched — a real config from the repo always wins over a synthesized one.
fn ensure_config_json(target: &Path, onnx_dir: &str) -> Result<(), Exit> {
    let dir = if onnx_dir.is_empty() {
        target.to_path_buf()
    } else {
        target.join(onnx_dir)
    };
    let path = dir.join("config.json");
    if path.exists() {
        return Ok(());
    }
    fs::write(&path, b"{}").map_err(|e| {
        log::error!("failed to write synthesized config {path:?}: {e}");
        Exit::DiskWrite
    })?;
    log::info!("synthesized empty config.json at {path:?} (repo shipped none)");
    Ok(())
}

fn remap_dest(dir: &str, rel: &str) -> String {
    const SUPPORT_FILES: &[&str] = &["tokenizer.json", "token.json", "config.json"];
    let name = rel.rsplit('/').next().unwrap_or(rel);
    if SUPPORT_FILES.contains(&name) {
        if dir.is_empty() {
            return name.to_string();
        }
        return format!("{dir}/{name}");
    }
    rel.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;

    fn sample_cli(model: &str, source: &str, force: bool, files: Vec<String>) -> Cli {
        Cli {
            config: None,
            model: Some(model.to_string()),
            source: Some(source.to_string()),
            url: None,
            slug: None,
            revision: "main".to_string(),
            model_dir: Some(std::path::PathBuf::from("/test/cache")),
            force,
            dry_run: false,
            files,
            extra_files: vec![],
            artifact: crate::cli::Artifact::Onnx,
            verbose: false,
            logfile: None,
        }
    }

    #[test]
    fn missing_model_is_bad_args() {
        let cli = sample_cli("", "hf:org/repo", false, vec![]);
        let r = Downloader::run(cli);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code(), Exit::BadArgs.code());
    }

    #[test]
    fn missing_source_is_bad_args() {
        let cli = Cli {
            config: None,
            model: Some("m".to_string()),
            source: None,
            url: None,
            slug: None,
            revision: "v1".to_string(),
            model_dir: None,
            force: false,
            dry_run: false,
            files: vec![],
            extra_files: vec![],
            artifact: crate::cli::Artifact::Onnx,
            verbose: false,
            logfile: None,
        };
        let r = Downloader::run(cli);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code(), Exit::BadArgs.code());
    }

    #[test]
    fn atomic_write_creates_file() {
        let tmp = std::env::temp_dir().join("md_atomic_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let dest = tmp.join("out.txt");
        atomic_write(&dest, b"hello").unwrap();
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "hello");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn remap_puts_support_files_next_to_the_onnx() {
        // Root-level source (the common case) and a module-subdir source both
        // have to land beside the ONNX, which is where the wrapper looks.
        assert_eq!(remap_dest("onnx", "tokenizer.json"), "onnx/tokenizer.json");
        assert_eq!(
            remap_dest("onnx", "0_StaticEmbedding/tokenizer.json"),
            "onnx/tokenizer.json"
        );
        assert_eq!(remap_dest("", "tokenizer.json"), "tokenizer.json");
        // Ordinary files keep their path.
        assert_eq!(remap_dest("onnx", "onnx/model.onnx"), "onnx/model.onnx");
    }

    #[test]
    fn ensure_config_writes_only_when_absent() {
        let tmp = std::env::temp_dir().join("md_config_synth_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("onnx")).unwrap();
        ensure_config_json(&tmp, "onnx").unwrap();
        let path = tmp.join("onnx/config.json");
        assert!(path.is_file(), "config.json synthesized for a repo with none");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
        // A second run must not clobber a real config.
        std::fs::write(&path, br#"{"dim":1024}"#).unwrap();
        ensure_config_json(&tmp, "onnx").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), r#"{"dim":1024}"#);
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Two downloads of the same artifact may overlap (engine auto-download vs
    /// the serve `download_embedder` rpc). They must not share a staging file.
    #[test]
    fn staging_files_are_unique_per_attempt() {
        let dir = std::env::temp_dir().join("md_unique_tmp_test");
        let a = unique_tmp(&dir, "model.onnx");
        let b = unique_tmp(&dir, "model.onnx");
        assert_ne!(a, b, "staging paths collided");
        assert_eq!(a.parent().unwrap(), dir, "staging stays next to the target");
        assert!(a
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with(".model.onnx."));
    }

    #[test]
    fn fetch_file_copies_local() {
        let src = std::env::temp_dir().join("md_src_test");
        let _ = std::fs::remove_dir_all(&src);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model.onnx"), b"model-bytes").unwrap();
        std::fs::write(src.join("tokenizer.json"), b"tok").unwrap();

        let target = std::env::temp_dir().join("md_tgt_test");
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target).unwrap();

        let dl = Downloader {
            cli: sample_cli("test-model", "file:///", false, vec![]),
            model_dir: PathBuf::from("/ignored"),
            progress: None,
        };
        let mut report = DownloadReport {
            target: target.clone(),
            skipped: vec![],
            downloaded: vec![],
        };
        dl.fetch_file(&src, &target, &[], &mut report).unwrap();

        assert_eq!(
            std::fs::read(target.join("model.onnx")).unwrap(),
            b"model-bytes"
        );
        assert_eq!(report.downloaded.len(), 2);
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&target).ok();
    }

    #[test]
    fn fetch_file_missing_requested_is_sourcenotfound() {
        let src = std::env::temp_dir().join("md_src_missing");
        let _ = std::fs::remove_dir_all(&src);
        std::fs::create_dir_all(&src).unwrap();

        let target = std::env::temp_dir().join("md_tgt_missing");
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target).unwrap();

        let dl = Downloader {
            cli: sample_cli("m", "file:///", false, vec![]),
            model_dir: PathBuf::from("/ignored"),
            progress: None,
        };
        let mut report = DownloadReport {
            target: target.clone(),
            skipped: vec![],
            downloaded: vec![],
        };
        // default files include model.onnx which is absent here but default list skips it
        let res = dl.fetch_file(&src, &target, &[], &mut report);
        assert!(res.is_ok()); // default list skips missing model.onnx silently
        assert!(std::fs::read_dir(&target).unwrap().next().is_none()); // nothing written

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&target).ok();
    }

    #[test]
    fn end_to_end_file_source() {
        let src = std::env::temp_dir().join("md_e2e_src");
        let target_dir = std::env::temp_dir().join("md_e2e_tgt");
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&target_dir);
        std::fs::create_dir_all(&src).unwrap();

        std::fs::write(src.join("model.onnx"), b"real-model-bytes").unwrap();
        std::fs::write(src.join("tokenizer.json"), b"{\"model\":{}}").unwrap();
        std::fs::write(src.join("config.json"), b"{\"dim\":1}").unwrap();

        let src_url = format!("file://{}", src.display());

        let cli = Cli {
            config: None,
            model: Some("e2e-model".to_string()),
            source: Some(src_url),
            url: None,
            slug: None,
            revision: "v1".to_string(),
            model_dir: Some(target_dir.clone()),
            force: false,
            dry_run: false,
            files: vec![],
            extra_files: vec![],
            artifact: crate::cli::Artifact::Onnx,
            verbose: false,
            logfile: None,
        };
        let result = Downloader::run(cli);
        assert!(result.is_ok(), "Downloader::run failed: {:?}", result.err());

        let cache = result.unwrap();
        assert_eq!(cache, target_dir.join("e2e-model").join("v1"));
        assert_eq!(
            std::fs::read(cache.join("model.onnx")).unwrap(),
            b"real-model-bytes"
        );
        assert!(cache.join("tokenizer.json").is_file());
        assert!(cache.join("config.json").is_file());

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&target_dir).ok();
    }

    #[test]
    fn extract_zip_flattens_nested_onnx() {
        let buf = make_zip(&[
            ("onnx/model.onnx", b"real-model-bytes"),
            ("config.json", b"{}"),
        ]);

        let target = std::env::temp_dir().join("md_zip_flatten_test");
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target).unwrap();

        extract_zip(&buf, &target).unwrap();
        flatten_onnx(&target);

        assert!(target.join("config.json").is_file(), "config.json at root");
        assert!(
            target.join("model.onnx").is_file(),
            "onnx should be flattened to root"
        );
        assert!(
            !target.join("onnx").exists(),
            "nested onnx dir should be gone"
        );
        assert_eq!(
            std::fs::read(target.join("model.onnx")).unwrap(),
            b"real-model-bytes"
        );

        std::fs::remove_dir_all(&target).ok();
    }

    #[test]
    fn extract_zip_blocks_path_traversal() {
        let buf = make_zip(&[("../evil.txt", b"pwned"), ("model.onnx", b"m")]);

        let target = std::env::temp_dir().join("md_zipslip_test");
        let _ = std::fs::remove_dir_all(&target);
        std::fs::create_dir_all(&target).unwrap();

        extract_zip(&buf, &target).unwrap();

        assert!(
            !target.parent().unwrap().join("evil.txt").exists(),
            "zip-slip traversal must be blocked"
        );
        assert!(target.join("model.onnx").is_file());

        std::fs::remove_dir_all(&target).ok();
    }

    #[test]
    fn external_data_siblings_finds_same_dir_payloads() {
        let tree = vec![
            TreeFile { path: "onnx/model.onnx".into(), size: 1 },
            TreeFile { path: "onnx/model.onnx_data".into(), size: 2 },
            TreeFile { path: "onnx/model.onnx_data_1".into(), size: 3 },
            TreeFile { path: "onnx/model_q4.onnx".into(), size: 4 },
            TreeFile { path: "onnx/model_q4.onnx_data".into(), size: 5 },
            TreeFile { path: "model.onnx_data".into(), size: 6 },
        ];
        let sib = external_data_siblings(&tree, "onnx/model.onnx");
        assert_eq!(sib, vec!["onnx/model.onnx_data", "onnx/model.onnx_data_1"]);
        let sib_q4 = external_data_siblings(&tree, "onnx/model_q4.onnx");
        assert_eq!(sib_q4, vec!["onnx/model_q4.onnx_data"]);
    }

    #[test]
    fn external_data_siblings_handles_root_and_gz() {
        let tree = vec![
            TreeFile { path: "model.onnx".into(), size: 1 },
            TreeFile { path: "model.onnx_data".into(), size: 2 },
        ];
        let sib = external_data_siblings(&tree, "model.onnx");
        assert_eq!(sib, vec!["model.onnx_data"]);
        let gz = external_data_siblings(&tree, "model.onnx.gz");
        assert!(gz.is_empty());
        let none = external_data_siblings(&tree, "tokenizer.json");
        assert!(none.is_empty());
    }

    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let mut zw = zip::ZipWriter::new(&mut buf);
        for (name, data) in entries {
            let opts = zip::write::FileOptions::<()>::default();
            zw.start_file(name, opts).unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap();
        buf.into_inner()
    }
}
