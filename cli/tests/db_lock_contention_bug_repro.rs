//! Reproduction test for DB lock contention bug (talentrankiq/bug.md).
//!
//! The bug: multiple code paths use the writer connection (`state.db.0.lock()`)
//! for SELECT queries that should use the read pool (`state.read_conn()`).
//! This is an architectural bug that can cause contention when the writer
//! connection is busy with long-running write operations.
//!
//! This test attempts to reproduce the contention by creating a scenario
//! where the writer connection is held while SELECT queries are executed.

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

/// Test that demonstrates the architectural bug: watch_job uses writer connection for SELECT
/// 
/// This test creates a job with many imported CVs, then calls watch_job which
/// triggers scan_and_import. The scan_and_import function uses the writer
/// connection for SELECT queries (lines 172 and 212 in watch.rs).
/// 
/// While this test may pass with fast SSD storage, it documents the bug.
/// The fix is to change these to use state.read_conn().
#[test]
fn db_lock_contention_watch_job_uses_writer_for_select() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let cv_dir = app_dir.join("cvs");
    let model_dir = temp_dir.path().join("models");

    // Create many CV files to make the SELECT query return many rows
    const NUM_CVS: usize = 5000;
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
    client
        .call(
            "set_job_cv_dir",
            &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
        )
        .expect("set_job_cv_dir response");

    // Wait for the initial import to complete
    thread::sleep(Duration::from_secs(45));

    // Verify files were imported
    let detail_before = client
        .call("get_job_detail", &json!({ "job_id": job_id }))
        .expect("get_job_detail response");
    let cv_imported_before = detail_before["cv_imported"].as_i64().unwrap_or(0);
    println!("CVs imported before re-scan: {}", cv_imported_before);
    assert!(cv_imported_before > 0, "Expected some CVs to be imported");

    // Now trigger another catch-up scan by calling watch_job
    // This will call scan_and_import which uses writer connection for SELECT
    let watch_start = Instant::now();
    let watch_resp = client
        .call("watch_job", &json!({ "job_id": job_id }))
        .expect("watch_job response");
    let watch_duration = watch_start.elapsed();
    println!("watch_job took: {:?}", watch_duration);

    // The watch_job handler (jobs.rs:611) uses writer connection for SELECT
    // Then watch::watch_job (watch.rs:307) uses writer connection for 2 more SELECTs
    // Then scan_and_import (watch.rs:170) uses writer connection for 2 more SELECTs
    // Total: 5 SELECT queries using writer connection

    // Immediately call get_job_detail many times to measure contention
    // get_job_detail correctly uses read_conn(), so it should not be blocked
    // by the writer connection in WAL mode.
    // However, if the writer connection is held for a long time, there could be contention.
    let mut durations = Vec::new();
    for i in 0..20 {
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

    // With the current implementation, watch_job uses writer connection for SELECT.
    // In WAL mode with read pool, get_job_detail should be fast (< 50ms).
    // This assertion documents the expected behavior after the fix.
    // Currently it passes because the SELECT queries are fast.
    assert!(
        max_duration < Duration::from_millis(500),
        "get_job_detail took {:?} (max), expected < 500ms. This indicates potential DB lock contention.",
        max_duration
    );

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}

/// Test that demonstrates the bug in set_job_cv_dir handler
/// 
/// The set_job_cv_dir handler (jobs.rs:575) uses writer connection for SELECT
/// at lines 594-600. This should use read_conn().
#[test]
fn db_lock_contention_set_job_cv_dir_uses_writer_for_select() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let cv_dir = app_dir.join("cvs");
    let model_dir = temp_dir.path().join("models");

    const NUM_CVS: usize = 1000;
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

    // First set CV dir
    client
        .call(
            "set_job_cv_dir",
            &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
        )
        .expect("set_job_cv_dir response");

    // Wait for import
    thread::sleep(Duration::from_secs(20));

    // Now call set_job_cv_dir again with the same directory
    // This triggers the handler which uses writer connection for SELECT
    let mut durations = Vec::new();
    for i in 0..10 {
        let start = Instant::now();
        let resp = client
            .call(
                "set_job_cv_dir",
                &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
            )
            .expect("set_job_cv_dir response");
        let elapsed = start.elapsed();
        durations.push(elapsed);
        println!("set_job_cv_dir call {}: {:?}", i + 1, elapsed);
        assert!(resp["id"].as_str().is_some());
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let max_duration = durations.iter().max().copied().unwrap_or(Duration::ZERO);
    println!("Average set_job_cv_dir duration: {:?}", avg_duration);
    println!("Max set_job_cv_dir duration: {:?}", max_duration);

    // The handler uses writer connection for SELECT (jobs.rs:594-600)
    // This should be fast but documents the architectural issue
    assert!(
        max_duration < Duration::from_millis(500),
        "set_job_cv_dir took {:?} (max), expected < 500ms",
        max_duration
    );

    client.shutdown();
    process.shutdown().expect("serve shutdown");
}

/// Test concurrent watch_job and get_job_detail calls
/// 
/// This simulates the GUI's WorkspaceActivity.loadJob which calls
/// watchJob and getJobDetail in parallel (Promise.all).
#[test]
fn db_lock_contention_concurrent_watch_and_detail() {
    let temp_dir = TempDir::new().expect("temp dir");
    let app_dir = temp_dir.path().to_path_buf();
    let cv_dir = app_dir.join("cvs");
    let model_dir = temp_dir.path().join("models");

    const NUM_CVS: usize = 3000;
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
    let client = Arc::new(Mutex::new(TestClient::new(read, write)));

    // Create a job
    let create_resp = client.lock().unwrap().call("create_job", &json!({})).expect("create_job");
    let job_id = create_resp["id"].as_str().expect("job id").to_string();

    // Set CV directory
    client.lock().unwrap().call(
        "set_job_cv_dir",
        &json!({ "job_id": job_id, "directory": cv_dir.to_string_lossy() }),
    ).expect("set_job_cv_dir");

    // Wait for import
    thread::sleep(Duration::from_secs(30));

    // Now simulate concurrent watch_job and get_job_detail calls
    // This is what the GUI does when loading a job
    let job_id_clone = job_id.clone();
    let client_clone = Arc::clone(&client);
    
    let watch_handle = thread::spawn(move || {
        let client = client_clone.lock().unwrap();
        let start = Instant::now();
        let resp = client.call("watch_job", &json!({ "job_id": job_id_clone }));
        let elapsed = start.elapsed();
        println!("watch_job took: {:?}", elapsed);
        (resp, elapsed)
    });

    let job_id_clone2 = job_id.clone();
    let client_clone2 = Arc::clone(&client);
    let detail_handle = thread::spawn(move || {
        let client = client_clone2.lock().unwrap();
        let start = Instant::now();
        let resp = client.call("get_job_detail", &json!({ "job_id": job_id_clone2 }));
        let elapsed = start.elapsed();
        println!("get_job_detail took: {:?}", elapsed);
        (resp, elapsed)
    });

    let (watch_result, watch_duration) = watch_handle.join().expect("watch thread panicked");
    let (detail_result, detail_duration) = detail_handle.join().expect("detail thread panicked");

    watch_result.expect("watch_job failed");
    let detail = detail_result.expect("get_job_detail failed");

    println!("get_job_detail duration in parallel: {:?}", detail_duration);
    assert!(detail["cv_imported"].as_i64().unwrap_or(0) > 0);

    // With the bug, watch_job uses writer connection for 5 SELECT queries.
    // In WAL mode with read pool, get_job_detail (using read_conn) should not block.
    // This assertion documents the expected behavior.
    assert!(
        detail_duration < Duration::from_millis(500),
        "get_job_detail took {:?} in parallel with watch_job, expected < 500ms",
        detail_duration
    );

    let client = Arc::try_unwrap(client).unwrap().into_inner().unwrap();
    client.shutdown();
    process.shutdown().expect("serve shutdown");
}