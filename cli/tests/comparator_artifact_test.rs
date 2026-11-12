//! E2E tests for the `load_comparator_artifact` / `remove_comparator_artifact` serve
//! commands. No mocks: each test spawns the real `triq serve` binary with a
//! temp `TRIQ_APP_DIR` and drives it over the JSONL stdio protocol with
//! `TriqiClient`, exactly like the gui does.
//!
//! Test contract (see plans/comparator-add-validation):
//! - wrong file name for the slug's mapped comparator artifact -> err "must be named"
//! - right name but `feature_spec.dim` != the embedder's registry dim -> err "dim"
//! - unparseable JSON / missing `feature_spec.dim` -> err "dim"
//! - unknown `embedder_slug` -> err "unknown embedder"
//! - correct file + slug -> ok, `list_embedders` reports `comparator.exists`
//! - no `embedder_slug` -> active-embedder fallback, same gates
//! - `remove_comparator_artifact` -> ok, file gone, `comparator.exists == false`

use libcli_client::client::{ClientError, TriqiClient};
use libcli_client::params::{EmptyParams, LoadComparatorArtifactParams, RemoveComparatorArtifactParams};
use libcli_client::types::GuiEmbedder;
use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

const SLUG_270M_ONNX: &str = "harrier-oss-v1-270m-ONNX-q4f16";
const SLUG_270M_MLX: &str = "harrier-oss-v1-270m-MLX-4bit";
const SLUG_600M_ONNX: &str = "harrier-oss-v1-0.6b-ONNX-q4";
const SLUG_600M_MLX: &str = "harrier-oss-v1-0.6b-MLX-4bit";

const JUDGE_270M: &str = "comparator-harrier-270m-v1.json";
const JUDGE_600M: &str = "comparator-harrier-600m-v1.json";

const DIM_270M: usize = 640;
const DIM_600M: usize = 1024;

/// Spawn serve on a fresh temp app dir and return (client, app_dir, child).
fn spawn_serve() -> Result<(TriqiClient, TempDir, Child), Box<dyn std::error::Error>> {
    let app_dir = TempDir::new()?;
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = manifest_dir.join("../target/debug/triq");
    let binary = fs::canonicalize(&binary).unwrap_or(binary);

    let mut cmd = Command::new(&binary);
    cmd.arg("serve");
    cmd.env("TRIQ_APP_DIR", app_dir.path());
    cmd.env("RUST_LOG", "info");
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;

    {
        let stderr = child.stderr.take().unwrap();
        let reader = BufReader::new(stderr);
        let start = std::time::Instant::now();
        let mut ready = false;
        for line in reader.lines() {
            if start.elapsed() > Duration::from_secs(60) {
                break;
            }
            match line {
                Ok(l) if l.contains("serve: ready") => {
                    ready = true;
                    break;
                }
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        if !ready {
            return Err("triq serve did not become ready".into());
        }
    }

    let stdout = child.stdout.take().unwrap();
    let stdin = child.stdin.take().unwrap();
    let client = TriqiClient::over(Box::new(stdout), Box::new(stdin));
    // Drain events so the channel never backs up.
    if let Some(rx) = client.take_event_receiver() {
        thread::spawn(move || {
            while let Ok(_ev) = rx.recv() {}
        });
    }
    Ok((client, app_dir, child))
}

fn load_comparator(
    client: &TriqiClient,
    source_path: &Path,
    embedder_slug: Option<&str>,
) -> Result<Value, ClientError> {
    client.call(
        "load_comparator_artifact",
        &LoadComparatorArtifactParams {
            source_path: source_path.to_string_lossy().to_string(),
            embedder_slug: embedder_slug.map(|s| s.to_string()),
        },
    )
}

fn list_embedders(client: &TriqiClient) -> Result<Vec<GuiEmbedder>, ClientError> {
    client.call_typed("list_embedders", &EmptyParams {})
}

fn expect_err_containing(result: Result<Value, ClientError>, needles: &[&str]) {
    match result {
        Err(e) => {
            let msg = e.to_string();
            for needle in needles {
                assert!(
                    msg.contains(needle),
                    "expected error containing {needle:?}, got: {msg}"
                );
            }
        }
        Ok(v) => panic!("expected error containing {needles:?}, got ok: {v}"),
    }
}

fn write_fixture(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, body).unwrap();
    p
}

fn comparator_json(dim: usize) -> String {
    format!(r#"{{"feature_spec":{{"dim":{dim}}}}}"#)
}

#[test]
fn rejects_mismatched_comparator_filename() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    // The 600M artifact picked for the 270M embedder: wrong file name.
    let src = write_fixture(
        fixtures.path(),
        JUDGE_600M,
        &comparator_json(DIM_600M),
    );

    let res = load_comparator(&client, &src, Some(SLUG_270M_ONNX));
    expect_err_containing(res, &["must be named", JUDGE_270M, JUDGE_600M]);
    assert!(
        !app_dir.path().join(JUDGE_270M).exists(),
        "rejected pick must not be copied into the app dir"
    );

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn rejects_wrong_dim_with_correct_filename() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    // Named like the 270M artifact but carrying the 600M dim.
    let src = write_fixture(fixtures.path(), JUDGE_270M, &comparator_json(DIM_600M));

    let res = load_comparator(&client, &src, Some(SLUG_270M_ONNX));
    expect_err_containing(res, &["dim"]);
    assert!(!app_dir.path().join(JUDGE_270M).exists());

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn rejects_missing_dim_and_unparseable_json() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    let missing_dim_dir = fixtures.path().join("missing-dim");
    let garbage_dir = fixtures.path().join("garbage");
    fs::create_dir_all(&missing_dim_dir).unwrap();
    fs::create_dir_all(&garbage_dir).unwrap();

    let missing_dim = write_fixture(&missing_dim_dir, JUDGE_270M, r#"{"feature_spec":{}}"#);
    expect_err_containing(load_comparator(&client, &missing_dim, Some(SLUG_270M_ONNX)), &["dim"]);

    let garbage = write_fixture(&garbage_dir, JUDGE_270M, "this is not json");
    expect_err_containing(load_comparator(&client, &garbage, Some(SLUG_270M_ONNX)), &["dim"]);

    assert!(!app_dir.path().join(JUDGE_270M).exists());

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn rejects_unknown_embedder_slug() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    let src = write_fixture(fixtures.path(), JUDGE_270M, &comparator_json(DIM_270M));

    let res = load_comparator(&client, &src, Some("no-such-embedder"));
    expect_err_containing(res, &["unknown embedder", "no-such-embedder"]);
    assert!(!app_dir.path().join(JUDGE_270M).exists());
    assert!(!app_dir.path().join(JUDGE_600M).exists());

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn accepts_correct_artifact_and_lists_exists() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    let src = write_fixture(fixtures.path(), JUDGE_270M, &comparator_json(DIM_270M));

    load_comparator(&client, &src, Some(SLUG_270M_ONNX))
        .expect("correct 270M artifact for 270M embedder must be accepted");

    let dest = app_dir.path().join(JUDGE_270M);
    assert!(dest.exists(), "artifact must land at app_dir/{JUDGE_270M}");
    assert_eq!(fs::read_to_string(&dest).unwrap(), comparator_json(DIM_270M));

    let embedders = list_embedders(&client).unwrap();
    let comparator_of = |slug: &str| {
        embedders
            .iter()
            .find(|e| e.slug == slug)
            .unwrap_or_else(|| panic!("embedder {slug} missing"))
            .comparator
            .clone()
            .expect("comparator mapping must be present")
    };
    // Both 270M variants map to the same file, so both report exists.
    assert!(comparator_of(SLUG_270M_ONNX).exists, "270M ONNX comparator.exists");
    assert!(comparator_of(SLUG_270M_MLX).exists, "270M MLX shares the 270M file");
    assert!(!comparator_of(SLUG_600M_ONNX).exists, "600M ONNX comparator.exists");
    assert!(!comparator_of(SLUG_600M_MLX).exists, "600M MLX comparator.exists");

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn active_embedder_fallback_applies_same_gates() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    // Default active embedder is the 270M ONNX; no slug -> same gates.
    let wrong = write_fixture(fixtures.path(), JUDGE_600M, &comparator_json(DIM_600M));
    expect_err_containing(load_comparator(&client, &wrong, None), &["must be named"]);
    assert!(!app_dir.path().join(JUDGE_270M).exists());

    let right_dir = fixtures.path().join("right");
    fs::create_dir_all(&right_dir).unwrap();
    let right = write_fixture(&right_dir, JUDGE_270M, &comparator_json(DIM_270M));
    load_comparator(&client, &right, None).expect("correct artifact via fallback must be accepted");
    assert!(app_dir.path().join(JUDGE_270M).exists());

    let embedders = list_embedders(&client).unwrap();
    let e = embedders.iter().find(|e| e.slug == SLUG_270M_ONNX).unwrap();
    assert!(e.comparator.as_ref().unwrap().exists);

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn remove_comparator_artifact_roundtrip() {
    let (client, app_dir, mut child) = spawn_serve().unwrap();
    let fixtures = TempDir::new().unwrap();
    let src = write_fixture(fixtures.path(), JUDGE_270M, &comparator_json(DIM_270M));

    load_comparator(&client, &src, Some(SLUG_270M_ONNX)).unwrap();
    assert!(app_dir.path().join(JUDGE_270M).exists());

    client
        .call(
            "remove_comparator_artifact",
            &RemoveComparatorArtifactParams {
                slug: SLUG_270M_ONNX.to_string(),
            },
        )
        .expect("remove_comparator_artifact must succeed");
    assert!(
        !app_dir.path().join(JUDGE_270M).exists(),
        "app-dir copy must be deleted"
    );
    // The originally picked source file is untouched.
    assert!(src.exists());

    let embedders = list_embedders(&client).unwrap();
    let e = embedders.iter().find(|e| e.slug == SLUG_270M_ONNX).unwrap();
    assert!(!e.comparator.as_ref().unwrap().exists);

    drop(client);
    let _ = child.kill();
    let _ = child.wait();
}
