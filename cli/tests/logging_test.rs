//! E2E tests for the `triq serve --log` per-request log files.
//!
//! Spawns the real `triq` binary, drives the real serve JSONL protocol over
//! stdio, and asserts on the timestamped per-request log files the serve loop
//! writes into the resolved log directory.

use libtriqui_client::{ClientError, TriqiClient};
use serde_json::json;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

struct ServeProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
}

impl ServeProcess {
    fn spawn(app_dir: &Path, args: &[&str]) -> Self {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let binary = manifest_dir.join("../target/debug/triq");
        let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);

        let mut cmd = Command::new(&binary);
        cmd.args(args);
        cmd.env("TRIQ_APP_DIR", app_dir);
        cmd.env("RUST_LOG", "info");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().expect("spawn triq serve");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        Self {
            child,
            stdin,
            stdout,
            stderr,
        }
    }

    fn wait_for_ready(&mut self, timeout: Duration) -> Result<(), String> {
        let stderr = self.stderr.as_mut().expect("stderr piped");
        let reader = BufReader::new(stderr);
        let start = Instant::now();
        for line in reader.lines() {
            let line = line.expect("read stderr line");
            if start.elapsed() > timeout {
                return Err("timeout waiting for serve ready".to_string());
            }
            if line.contains("serve: ready") {
                return Ok(());
            }
        }
        Err("process exited before ready".to_string())
    }

    fn take_stdio(
        &mut self,
    ) -> (
        Box<dyn std::io::Read + Send>,
        Box<dyn std::io::Write + Send>,
    ) {
        let stdout = self.stdout.take().expect("stdout piped");
        let stdin = self.stdin.take().expect("stdin piped");
        (Box::new(stdout), Box::new(stdin))
    }

    fn shutdown(mut self) -> Result<(), String> {
        drop(self.stdin.take());
        let status = self.child.wait().map_err(|e| format!("wait: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("serve exited with {status}"))
        }
    }
}

struct TestClient {
    client: TriqiClient,
}

impl TestClient {
    fn new(read: Box<dyn std::io::Read + Send>, write: Box<dyn std::io::Write + Send>) -> Self {
        Self {
            client: TriqiClient::over(read, write),
        }
    }

    fn call(
        &self,
        cmd: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, ClientError> {
        self.client.call(cmd, params)
    }

    fn shutdown(&self) {
        self.client.shutdown();
    }
}

/// Poll the log dir for the per-request file of `request` and return its content.
/// Waits until every expected substring is present in the file.
fn wait_for_request_log(
    log_dir: &Path,
    request: &str,
    expect: &[&str],
    timeout: Duration,
) -> String {
    let start = Instant::now();
    loop {
        if let Ok(entries) = std::fs::read_dir(log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with("triq__") || !name.ends_with(".log") {
                    continue;
                }
                if !name.contains(&format!("__{request}.")) {
                    continue;
                }
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if expect.iter().all(|e| content.contains(e)) {
                        return content;
                    }
                }
            }
        }
        if start.elapsed() > timeout {
            panic!("timed out waiting for log file for request '{request}'");
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn list_log_files(log_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(log_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("triq__") && name.ends_with(".log") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn serve_writes_per_request_log_files() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let log_dir = temp_dir.path().join("logs");

    let mut process = ServeProcess::spawn(
        &app_dir,
        &[
            "serve",
            "--log-dir",
            log_dir.to_str().expect("log dir utf8"),
        ],
    );
    process
        .wait_for_ready(Duration::from_secs(60))
        .expect("serve ready");
    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    client
        .call("load_preferences", &json!({}))
        .expect("load_preferences response");

    let content = wait_for_request_log(
        &log_dir,
        "load-preferences",
        &[
            "load_preferences",
            "preferences file not found, using defaults",
        ],
        Duration::from_secs(30),
    );
    assert!(
        content.contains("load_preferences"),
        "log has span: {content}"
    );

    client
        .call("create_job", &json!({}))
        .expect("create_job response");
    let create_content = wait_for_request_log(
        &log_dir,
        "create-job",
        &["created job"],
        Duration::from_secs(30),
    );
    assert!(
        create_content.contains("created job"),
        "log has create event: {create_content}"
    );

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}

#[test]
fn serve_writes_wave_events_to_request_log() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let log_dir = temp_dir.path().join("logs");

    let mut process = ServeProcess::spawn(
        &app_dir,
        &[
            "serve",
            "--log-dir",
            log_dir.to_str().expect("log dir utf8"),
        ],
    );
    process
        .wait_for_ready(Duration::from_secs(60))
        .expect("serve ready");
    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    let create: serde_json::Value = client
        .call("create_job", &json!({}))
        .expect("create_job response");
    let job_id = create["id"].as_str().expect("job id").to_string();

    let _ = client
        .call("process_in_waves", &json!({ "job_id": job_id }))
        .expect("process_in_waves response");

    let content = wait_for_request_log(
        &log_dir,
        "process-in-waves",
        &[
            "wave 0",
            "scan done",
            "quick done",
            "deep done",
            "league intro start",
            "comparisons done",
        ],
        Duration::from_secs(60),
    );
    for signal in [
        "wave 0",
        "scan done",
        "quick done",
        "deep done",
        "league intro start",
        "comparisons done",
    ] {
        assert!(
            content.contains(signal),
            "wave log missing {signal}: {content}"
        );
    }

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}

#[test]
fn serve_no_log_flag_writes_no_files() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let log_dir = temp_dir.path().join("logs");

    let mut process = ServeProcess::spawn(
        &app_dir,
        &[
            "serve",
            "--no-log",
            "--log-dir",
            log_dir.to_str().expect("log dir utf8"),
        ],
    );
    process
        .wait_for_ready(Duration::from_secs(60))
        .expect("serve ready");
    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    client
        .call("load_preferences", &json!({}))
        .expect("load_preferences response");

    thread::sleep(Duration::from_secs(2));

    let files = list_log_files(&log_dir);
    assert!(
        files.is_empty(),
        "--no-log must not create log files, found: {:?}",
        files
    );

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}
