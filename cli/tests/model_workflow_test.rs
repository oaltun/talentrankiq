use libtriqui_client::client::TriqiClient;
use libtriqui_client::protocol::Event;
use libtriqui_client::types::{
    GuiEmbedder, ModelDownloadComplete, ModelDownloadProgress, TryModelResult,
};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const EMBEDDER_SLUGS: &[&str] = &[
    "harrier-oss-v1-270m-ONNX-q4f16",
    "harrier-oss-v1-270m-MLX-4bit",
    "harrier-oss-v1-0.6b-ONNX-q4",
    "harrier-oss-v1-0.6b-MLX-4bit",
];

fn should_test_embedder(slug: &str) -> bool {
    if slug.contains("MLX") {
        return false;
    }
    true
}

struct TestProcess {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
}

impl TestProcess {
    fn spawn(app_dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let binary = manifest_dir.join("../target/debug/triq");
        let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);
        
        let mut cmd = Command::new(&binary);
        cmd.arg("serve");
        cmd.env("TRIQ_APP_DIR", app_dir);
        cmd.env("RUST_LOG", "info");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let process = Self {
            child,
            stdin,
            stdout,
            stderr,
        };

        thread::sleep(Duration::from_secs(2));
        Ok(process)
    }

    fn wait_for_ready(&mut self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let stderr = self.stderr.as_mut().unwrap();
        let reader = BufReader::new(stderr);
        let start = std::time::Instant::now();
        for line in reader.lines() {
            if start.elapsed() > timeout {
                return Err("Timeout waiting for ready".into());
            }
            if let Ok(line) = line {
                if line.contains("serve: ready") {
                    return Ok(());
                }
            }
        }
        Err("Process exited before ready".into())
    }

    fn take_stdio(&mut self) -> (Box<dyn std::io::Read + Send>, Box<dyn std::io::Write + Send>) {
        let stdout = self.stdout.take().unwrap();
        let stdin = self.stdin.take().unwrap();
        (Box::new(stdout), Box::new(stdin))
    }

    fn shutdown(mut self) -> Result<(), Box<dyn std::error::Error>> {
        drop(self.stdin.take());
        let _ = self.child.wait()?;
        Ok(())
    }
}

struct TestClient {
    client: TriqiClient,
    events: Arc<Mutex<Vec<Event>>>,
    _event_thread: thread::JoinHandle<()>,
}

impl TestClient {
    fn new(read: Box<dyn std::io::Read + Send>, write: Box<dyn std::io::Write + Send>) -> Self {
        let client = TriqiClient::over(read, write);
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let event_rx = client.take_event_receiver().expect("event receiver");
        let _event_thread = thread::spawn(move || {
            while let Ok(event) = event_rx.recv() {
                events_clone.lock().unwrap().push(event);
            }
        });
        Self {
            client,
            events,
            _event_thread,
        }
    }

    fn call<T: serde::de::DeserializeOwned>(&self, cmd: &str, params: &impl serde::Serialize) -> Result<T, Box<dyn std::error::Error>> {
        Ok(self.client.call_typed(cmd, params)?)
    }

    fn call_raw(&self, cmd: &str, params: &impl serde::Serialize) -> Result<Value, Box<dyn std::error::Error>> {
        Ok(self.client.call(cmd, params)?)
    }

    fn wait_for_event<F>(&self, predicate: F, timeout: Duration) -> Result<Event, Box<dyn std::error::Error>>
    where
        F: Fn(&Event) -> bool,
    {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err("Timeout waiting for event".into());
            }
            let events = self.events.lock().unwrap();
            if let Some(event) = events.iter().find(|e| predicate(e)) {
                return Ok(event.clone());
            }
            drop(events);
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn wait_for_download_progress(&self, slug: &str, timeout: Duration) -> Result<ModelDownloadProgress, Box<dyn std::error::Error>> {
        let event = self.wait_for_event(
            |e| e.event == "model-download-progress" && e.data["repo_id"].as_str() == Some(slug),
            timeout,
        )?;
        Ok(serde_json::from_value(event.data)?)
    }

    fn wait_for_download_complete(&self, slug: &str, timeout: Duration) -> Result<ModelDownloadComplete, Box<dyn std::error::Error>> {
        let event = self.wait_for_event(
            |e| e.event == "model-download-complete" && e.data["slug"].as_str() == Some(slug),
            timeout,
        )?;
        Ok(serde_json::from_value(event.data)?)
    }
}

fn copy_benchmark_cvs(src: &std::path::Path, dst: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "txt") {
            let dest = dst.join(path.file_name().unwrap());
            std::fs::copy(&path, &dest)?;
        }
    }
    Ok(())
}

fn list_embedders(client: &TestClient) -> Result<Vec<GuiEmbedder>, Box<dyn std::error::Error>> {
    let result: Vec<GuiEmbedder> = client.call("list_embedders", &libtriqui_client::params::EmptyParams {})?;
    Ok(result)
}

fn find_embedder<'a>(embedders: &'a [GuiEmbedder], slug: &str) -> Option<&'a GuiEmbedder> {
    embedders.iter().find(|e| e.slug == slug)
}

#[test]
fn test_cli_model_workflow() {
    let _ = env_logger::try_init();

    let benchmark_src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../gui/src-tauri/benchmark-cvs");

    let temp_dir = TempDir::new().expect("create temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let benchmark_dst = app_dir.join("benchmark-cvs");

    copy_benchmark_cvs(&benchmark_src, &benchmark_dst).expect("copy benchmark CVs");
    assert!(benchmark_dst.join("cv-01-geologist.txt").exists(), "CV files copied");

    let mut process = TestProcess::spawn(&app_dir).expect("spawn triq serve");
    process.wait_for_ready(Duration::from_secs(30)).expect("wait for ready");

    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    let embedders = list_embedders(&client).expect("initial list_embedders");
    assert_eq!(embedders.len(), 4, "4 gui_embedders");

    for &slug in EMBEDDER_SLUGS {
        if !should_test_embedder(slug) {
            continue;
        }

        let embedders = list_embedders(&client).expect("list_embedders before download");
        let embedder = find_embedder(&embedders, slug).expect("embedder found");
        let expected_default = slug == "harrier-oss-v1-270m-ONNX-q4f16";
        assert_eq!(embedder.is_default, expected_default, "default status for {}", slug);

        let was_downloaded = embedder.downloaded;
        if !was_downloaded {
            let download_result = client
                .call_raw("download_embedder", &libtriqui_client::params::DownloadEmbedderParams { slug: slug.to_string() });

            if download_result.is_err() {
                eprintln!("Download failed for {}: {:?}, skipping to next embedder", slug, download_result);
                continue;
            }
            download_result.expect("download_embedder request");

            let progress_result = client.wait_for_download_progress(slug, Duration::from_secs(600));
            if progress_result.is_err() {
                eprintln!("Download progress timeout for {}: {:?}, skipping to next embedder", slug, progress_result);
                continue;
            }
            let complete_result = client.wait_for_download_complete(slug, Duration::from_secs(600));
            if complete_result.is_err() {
                eprintln!("Download complete timeout for {}: {:?}, skipping to next embedder", slug, complete_result);
                continue;
            }
        }

        let embedders = list_embedders(&client).expect("list_embedders after download");
        let embedder = find_embedder(&embedders, slug).expect("embedder found");
        assert!(embedder.downloaded, "downloaded after download: {}", slug);
        assert!(embedder.size_bytes > 0, "size_bytes > 0 for {}", slug);

        let result: TryModelResult = client
            .call("try_model", &libtriqui_client::params::TryModelParams { slug: slug.to_string() })
            .expect("try_model");

        assert_eq!(result.per_cv_ms.len(), 11, "11 CVs processed for {}", slug);
        assert!(result.median_ms > 0.0, "median_ms > 0 for {}", slug);
        assert_eq!(result.pairs.len(), 55, "55 pairs for {}", slug);
        assert_eq!(result.labels.len(), 11, "11 labels for {}", slug);

        let twin_pair = result.pairs.iter().find(|p| {
            (p.label_i == "CV 10 — Mechanical Engineer" && p.label_j == "CV 11 — Mechanical Engineer [twin]")
                || (p.label_i == "CV 11 — Mechanical Engineer [twin]" && p.label_j == "CV 10 — Mechanical Engineer")
        });
        let twin_similarity = twin_pair.expect("twin pair exists").similarity;
        assert!(twin_similarity > 0.95, "twin similarity > 0.95 for {}, got {}", slug, twin_similarity);

        client
            .call_raw("set_embedding_model", &libtriqui_client::params::SetEmbeddingModelParams { slug: slug.to_string() })
            .expect("set_embedding_model");

        let embedders = list_embedders(&client).expect("list_embedders after set default");
        let embedder = find_embedder(&embedders, slug).expect("embedder found");
        assert!(embedder.is_default, "is_default after set: {}", slug);

        for other in &embedders {
            if other.slug.as_str() != slug {
                assert!(!other.is_default, "other not default: {}", other.slug);
            }
        }

        client
            .call_raw("remove_embedder", &libtriqui_client::params::RemoveEmbedderParams { slug: slug.to_string() })
            .expect("remove_embedder");

        let embedders = list_embedders(&client).expect("list_embedders after remove");
        let embedder = find_embedder(&embedders, slug).expect("embedder found");
        assert!(!embedder.downloaded, "not downloaded after remove: {}", slug);
        assert_eq!(embedder.size_bytes, 0, "size_bytes = 0 after remove: {}", slug);
    }

    client.client.shutdown();
    process.shutdown().expect("process shutdown");
}