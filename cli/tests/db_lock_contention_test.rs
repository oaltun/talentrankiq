//! E2E test for DB lock contention bug (talentrankiq/bug.md).
//!
//! The bug: when a CV-directory scan runs (triggered by `watch_job`), it holds
//! the single shared SQLite connection (Mutex<Connection>) while doing one SELECT
//! per file to filter already-imported files. This blocks concurrent readers like
//! `get_job_detail` for 10-30 seconds on 2000 files.
//!
//! This test reproduces the contention by:
//! 1. Creating a job with a CV directory containing many files
//! 2. Importing them so they're marked as "Extracted" in the DB
//! 3. Triggering a catch-up scan (which checks each file via SELECT)
//! 4. Calling `get_job_detail` concurrently and measuring the delay

use libtriqui_client::{ClientError, TriqiClient};
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
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
    fn spawn(app_dir: &Path, model_dir: Option<&Path>) -> Self {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        // From cli/ (CARGO_MANIFEST_DIR) -> ../target/debug/triq
        let binary = manifest_dir.join("../target/debug/triq");
        let binary = fs::canonicalize(&binary).unwrap_or_else(|_| binary);

        let mut cmd = Command::new(&binary);
        cmd.arg("serve");
        cmd.env("TRIQ_APP_DIR", app_dir);
        if let Some(md) = model_dir {
            cmd.env("TRIQI_MODEL_DIR", md);
        }
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

impl std::fmt::Debug for TestClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestClient").finish()
    }
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

/// Create N dummy CV files in a directory
fn create_cv_files(dir: &Path, count: usize) {
    fs::create_dir_all(dir).expect("create cv dir");
    for i in 0..count {
        let content = format!(
            "Candidate {} resume content. Skills: Rust, TypeScript, React. Experience: {} years.",
            i, i
        );
        fs::write(dir.join(format!("cv-{:04}.txt", i)), content).expect("write cv file");
    }
}

#[test]
fn db_lock_contention_during_scan() {
    // This test reproduces the bug where a CV-directory scan holds the DB lock
    // while doing per-file SELECT queries, blocking concurrent get_job_detail calls.
    //
    // Expected behavior with bug: get_job_detail calls take seconds each
    // Expected behavior after fix: get_job_detail calls take milliseconds

    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let cv_dir = app_dir.join("cvs");
    let model_dir = temp_dir.path().join("models");

    // Create 2000 CV files to match the bug report (2000 files = 10-30s stall)
    const NUM_CVS: usize = 2000;
    create_cv_files(&cv_dir, NUM_CVS);

    // Provide a fake quick model so the workspace is reachable
    let quick_slug = "static-similarity-mrl-multilingual-v1";
    let quick_onnx = model_dir.join(quick_slug).join("repo").join("onnx");
    fs::create_dir_all(&quick_onnx).expect("create model dir");
    fs::write(quick_onnx.join("model.onnx"), "fake").expect("write fake model");

    let mut process = ServeProcess::spawn(&app_dir, Some(&model_dir));
    process
        .wait_for_ready(Duration::from_secs(60))
        .expect("serve ready");

    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    // Create a job
    let create_resp = client
        .call("create_job", &json!({}))
        .expect("create_job response");
    let job_id = create_resp["id"].as_str().expect("job id").to_string();
    println!("Created job: {}", job_id);

    // Set the CV directory to our test folder
    let set_cv_resp = client
        .call(
            "set_job_cv_dir",
            &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
        )
        .expect("set_job_cv_dir response");
    println!("Set CV dir: {:?}", set_cv_resp);

    // Wait for the catch-up scan to complete the initial import
    // For 2000 text files, extraction is fast but embedding takes time.
    // We wait longer to ensure candidates are inserted with extraction_status='Extracted'
    thread::sleep(Duration::from_secs(30));

    // Verify files were imported by checking job detail
    let detail_before = client
        .call("get_job_detail", &json!({ "job_id": job_id }))
        .expect("get_job_detail response");
    let cv_imported_before = detail_before["cv_imported"].as_i64().unwrap_or(0);
    println!("CVs imported before re-scan: {}", cv_imported_before);
    assert!(cv_imported_before > 0, "Expected some CVs to be imported");

    // Now trigger another catch-up scan by calling watch_job
    // This will scan the directory and check each file via SELECT
    // Since files are already imported, each SELECT will find a row
    let watch_resp = client
        .call("watch_job", &json!({ "job_id": job_id }))
        .expect("watch_job response");
    println!("Triggered watch_job: {:?}", watch_resp);

    // Give the scan a moment to start (debounce is 1s)
    thread::sleep(Duration::from_millis(1500));

    // Now call get_job_detail repeatedly while the scan holds the DB lock
    // Each call should be slow because the scan is doing 2000 SELECT queries
    // while holding the mutex
    let mut durations = Vec::new();
    for i in 0..10 {
        let start = Instant::now();
        let detail_resp = client
            .call("get_job_detail", &json!({ "job_id": job_id }))
            .expect("get_job_detail response");
        let elapsed = start.elapsed();
        durations.push(elapsed);
        println!("get_job_detail call {}: {:?}", i + 1, elapsed);
        assert!(detail_resp["cv_imported"].as_i64().unwrap_or(0) > 0);
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let max_duration = durations.iter().max().copied().unwrap_or(Duration::ZERO);
    println!("Average get_job_detail duration: {:?}", avg_duration);
    println!("Max get_job_detail duration: {:?}", max_duration);

    // With the bug, each get_job_detail should take hundreds of milliseconds
    // because the scan holds the DB lock for seconds (2000 SELECT queries).
    // After the fix, they should take < 50ms each.
    //
    // This assertion should FAIL with the current buggy implementation.
    assert!(
        max_duration < Duration::from_millis(500),
        "get_job_detail took {:?} (max), expected < 500ms. This indicates DB lock contention from the scan.",
        max_duration
    );

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}

#[test]
fn db_lock_contention_initial_page_load() {
    // This test reproduces the bug where opening a job (which calls watch_job
    // and get_job_detail in parallel) stalls because the scan holds the DB lock.
    //
    // The GUI's WorkspaceActivity.loadJob does:
    //   Promise.all([watchJob(jobId), getJobDetail(jobId)])
    //
    // If the scan from watch_job holds the lock, get_job_detail blocks.

    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let cv_dir = app_dir.join("cvs");
    let model_dir = temp_dir.path().join("models");

    const NUM_CVS: usize = 2000;
    create_cv_files(&cv_dir, NUM_CVS);

    let quick_slug = "static-similarity-mrl-multilingual-v1";
    let quick_onnx = model_dir.join(quick_slug).join("repo").join("onnx");
    fs::create_dir_all(&quick_onnx).expect("create model dir");
    fs::write(quick_onnx.join("model.onnx"), "fake").expect("write fake model");

    let mut process = ServeProcess::spawn(&app_dir, Some(&model_dir));
    process
        .wait_for_ready(Duration::from_secs(60))
        .expect("serve ready");

    let (read, write) = process.take_stdio();
    let client = TestClient::new(read, write);

    // Create a job
    let create_resp = client
        .call("create_job", &json!({}))
        .expect("create_job response");
    let job_id = create_resp["id"].as_str().expect("job id").to_string();

    // Set CV directory (triggers initial scan + import)
    client
        .call(
            "set_job_cv_dir",
            &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
        )
        .expect("set_job_cv_dir response");

    // Wait for import to complete (2000 files takes longer)
    thread::sleep(Duration::from_secs(30));

    // Now simulate the GUI's loadJob: call watch_job and get_job_detail in parallel
    // In the real GUI, these are Promise.all'd. Here we spawn threads.
    let job_id_clone = job_id.clone();
    let client_clone = Arc::new(Mutex::new(client));

    let client_for_watch = Arc::clone(&client_clone);
    let watch_handle = thread::spawn(move || {
        let client = client_for_watch.lock().unwrap();
        let start = Instant::now();
        let resp = client.call("watch_job", &json!({ "job_id": job_id_clone }));
        let elapsed = start.elapsed();
        println!("watch_job took: {:?}", elapsed);
        resp
    });

    let job_id_clone2 = job_id.clone();
    let client_for_detail = Arc::clone(&client_clone);
    let detail_handle = thread::spawn(move || {
        let client = client_for_detail.lock().unwrap();
        let start = Instant::now();
        let resp = client.call("get_job_detail", &json!({ "job_id": job_id_clone2 }));
        let elapsed = start.elapsed();
        println!("get_job_detail took: {:?}", elapsed);
        (resp, elapsed)
    });

    let watch_result = watch_handle.join().expect("watch thread panicked");
    let (detail_result, detail_duration) = detail_handle.join().expect("detail thread panicked");

    watch_result.expect("watch_job failed");
    let detail = detail_result.expect("get_job_detail failed");

    println!("get_job_detail duration in parallel: {:?}", detail_duration);
    assert!(detail["cv_imported"].as_i64().unwrap_or(0) > 0);

    // With the bug, get_job_detail should be slow because the scan from
    // watch_job holds the DB lock. The scan does 2000 SELECT queries.
    // After the fix, it should be fast (< 500ms).
    //
    // This assertion should FAIL with the current buggy implementation.
    assert!(
        detail_duration < Duration::from_millis(500),
        "get_job_detail took {:?} in parallel with watch_job, expected < 500ms. This indicates DB lock contention.",
        detail_duration
    );

    let client = Arc::try_unwrap(client_clone).unwrap().into_inner().unwrap();
    client.shutdown();
    process.shutdown().expect("serve shutdown");
}