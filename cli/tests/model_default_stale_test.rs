//! Regression test for the bug described in `talentrankiq_gui/bug.md`:
//! "Job Details page is unreachable when the default embedding model is missing".
//!
//! Root cause chain (CLI side):
//!   1. `make_default_embedder` records the preference even when the model is
//!      not downloaded yet.
//!   2. `remove_embedder` deletes the cache directory but never clears
//!      `preferences.embedding_model` when the removed model was the default.
//!   3. `list_embedders` derives `is_default` only from the saved preference, so
//!      it can return `{ is_default: true, downloaded: false }`.
//!
//! That last state is the contract mismatch the GUI trips over: a saved default
//! that points at a model with no files on disk.
//!
//! This test uses the real `triq serve` process and the real protocol (no mocks).
//! It fakes a downloaded model by dropping a real `.onnx` file into the model
//! cache so `onnx_exists()` reports it present, drives the bug path, and asserts
//! the CLI never reports a default that is not downloaded.

use std::io::BufRead;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use libtriqui_client::client::TriqiClient;
use libtriqui_client::types::GuiEmbedder;
use serde_json::Value;
use tempfile::TempDir;

const TARGET_SLUG: &str = "harrier-oss-v1-270m-ONNX-q4f16";

struct TestProcess {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: Option<std::process::ChildStdout>,
    ready_rx: mpsc::Receiver<()>,
    _stderr_thread: thread::JoinHandle<()>,
}

impl TestProcess {
    fn spawn(app_dir: &Path, model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let binary = manifest_dir.join("../target/debug/triq");
        let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);

        let mut cmd = Command::new(&binary);
        cmd.arg("serve");
        cmd.env("TRIQ_APP_DIR", app_dir);
        cmd.env("TRIQI_MODEL_DIR", model_dir);
        cmd.env("RUST_LOG", "info");
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take().expect("stderr piped");

        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let stderr_thread = thread::Builder::new()
            .name("triq-stderr-drain".into())
            .spawn(move || {
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines() {
                    match line {
                        Ok(line) => {
                            eprintln!("[triq] {line}");
                            if line.contains("serve: ready") {
                                let _ = ready_tx.send(());
                            }
                        }
                        Err(_) => break,
                    }
                }
            })?;

        let process = Self {
            child,
            stdin,
            stdout,
            ready_rx,
            _stderr_thread: stderr_thread,
        };
        thread::sleep(Duration::from_secs(2));
        Ok(process)
    }

    fn wait_for_ready(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        match self.ready_rx.recv_timeout(timeout) {
            Ok(()) => Ok(()),
            Err(mpsc::RecvTimeoutError::Timeout) => Err("Timeout waiting for ready".into()),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err("Process exited before ready".into())
            }
        }
    }

    fn take_stdio(
        &mut self,
    ) -> (
        Box<dyn std::io::Read + Send>,
        Box<dyn std::io::Write + Send>,
    ) {
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

fn call_raw(
    client: &TriqiClient,
    cmd: &str,
    params: &impl serde::Serialize,
) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(client.call(cmd, params)?)
}

fn list_embedders(client: &TriqiClient) -> Result<Vec<GuiEmbedder>, Box<dyn std::error::Error>> {
    let result: Vec<GuiEmbedder> = client.call_typed(
        "list_embedders",
        &libtriqui_client::params::EmptyParams {},
    )?;
    Ok(result)
}

/// Drop a single dummy `.onnx` file under the model's repo cache dir so that
/// `libmodelstorage::onnx_exists()` reports the model as downloaded. No engine
/// is ever loaded: `list_embedders` / `remove_embedder` only check the filesystem.
fn fake_downloaded(model_dir: &Path, slug: &str) {
    let onnx_dir = model_dir.join(slug).join("repo").join("onnx");
    std::fs::create_dir_all(&onnx_dir).expect("create fake onnx dir");
    std::fs::write(onnx_dir.join("model_q4f16.onnx"), b"fake-onnx-bytes")
        .expect("write fake onnx file");
}

#[test]
fn remove_embedder_clears_stale_default_preference() {
    let _ = tracing_subscriber::fmt::try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let app_dir = temp_dir.path().join("app");
    let model_dir = temp_dir.path().join("models");
    std::fs::create_dir_all(&app_dir).expect("create app dir");
    std::fs::create_dir_all(&model_dir).expect("create model dir");

    let mut process =
        TestProcess::spawn(&app_dir, &model_dir).expect("spawn triq serve");
    process
        .wait_for_ready(Duration::from_secs(30))
        .expect("wait for ready");

    let (read, write) = process.take_stdio();
    let client = TriqiClient::over(read, write);

    // Pretend the deep model is downloaded so `set_embedding_model` accepts it.
    fake_downloaded(&model_dir, TARGET_SLUG);

    // Make the target model the default (persists the preference).
    call_raw(
        &client,
        "set_embedding_model",
        &libtriqui_client::params::SetEmbeddingModelParams {
            slug: TARGET_SLUG.to_string(),
        },
    )
    .expect("set_embedding_model");

    let embedders = list_embedders(&client).expect("list_embedders after set default");
    let target = embedders
        .iter()
        .find(|e| e.slug == TARGET_SLUG)
        .expect("target embedder present");
    assert!(
        target.is_default && target.downloaded,
        "target should be default + downloaded after set default"
    );

    // Remove the model. This is the bug: `remove_embedder` deletes the cache
    // directory but leaves `preferences.embedding_model` pointing at the slug.
    call_raw(
        &client,
        "remove_embedder",
        &libtriqui_client::params::RemoveEmbedderParams {
            slug: TARGET_SLUG.to_string(),
        },
    )
    .expect("remove_embedder");

    let embedders = list_embedders(&client).expect("list_embedders after remove");
    let target = embedders
        .iter()
        .find(|e| e.slug == TARGET_SLUG)
        .expect("target embedder present");

    // The removed model must no longer be reported as downloaded...
    assert!(
        !target.downloaded,
        "removed model should not be downloaded"
    );

    // ...and it must never be reported as the default. A default that is not
    // downloaded is exactly the state that makes the GUI redirect the whole
    // workspace to the Models page.
    assert!(
        !target.is_default,
        "removed model must not remain the default (stale preference)"
    );

    // No embedder should ever be simultaneously default and not downloaded.
    for e in &embedders {
        assert!(
            !(e.is_default && !e.downloaded),
            "embedder '{}' reported is_default=true, downloaded=false",
            e.slug
        );
    }

    client.shutdown();
    process.shutdown().expect("process shutdown");
}

/// Test for scenario 2 from the bug report:
/// Fresh install with no saved preference, where the config's default model
/// is not downloaded. The CLI should never report a default that is not downloaded.
#[test]
fn fresh_install_config_default_not_downloaded() {
    let _ = tracing_subscriber::fmt::try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let app_dir = temp_dir.path().join("app");
    let model_dir = temp_dir.path().join("models");
    std::fs::create_dir_all(&app_dir).expect("create app dir");
    std::fs::create_dir_all(&model_dir).expect("create model dir");

    let mut process =
        TestProcess::spawn(&app_dir, &model_dir).expect("spawn triq serve");
    process
        .wait_for_ready(Duration::from_secs(30))
        .expect("wait for ready");

    let (read, write) = process.take_stdio();
    let client = TriqiClient::over(read, write);

    // Fresh install: no saved preference, config default is NOT downloaded
    // (we do NOT call fake_downloaded for the default model)
    let embedders = list_embedders(&client).expect("list_embedders on fresh install");

    // Find the config default embedder (harrier-oss-v1-270m-ONNX-q4f16)
    let default_embedder = embedders
        .iter()
        .find(|e| e.slug == TARGET_SLUG)
        .expect("config default embedder present");

    // The default embedder should NOT be reported as downloaded
    assert!(
        !default_embedder.downloaded,
        "config default should not be downloaded on fresh install"
    );

    // The default embedder should NOT be reported as is_default
    // because it's not downloaded. This is the fix for the bug.
    assert!(
        !default_embedder.is_default,
        "config default must not be reported as default when not downloaded (stale default)"
    );

    // No embedder should ever be simultaneously default and not downloaded.
    for e in &embedders {
        assert!(
            !(e.is_default && !e.downloaded),
            "embedder '{}' reported is_default=true, downloaded=false",
            e.slug
        );
    }

    client.shutdown();
    process.shutdown().expect("process shutdown");
}