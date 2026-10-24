use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use tempfile::TempDir;

#[test]
fn test_cli_download_onnx_model() {
    let _ = env_logger::try_init();

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = manifest_dir.join("../target/debug/triq");
    let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);

    let temp_dir = TempDir::new().expect("create temp dir");
    let model_dir = temp_dir.path().join("models");

    // Test 270M ONNX download
    let mut cmd = Command::new(&binary);
    cmd.args([
        "download",
        "--slug", "harrier-oss-v1-270m-ONNX-q4f16",
        "--url", "https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx",
        "--model-dir", model_dir.to_str().unwrap(),
    ]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn download command");

    let stderr = child.stderr.take().unwrap();
    thread::spawn(move || {
        let reader = std::io::BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[download stderr] {}", line);
            }
        }
    });

    let status = child.wait().expect("wait for download");
    assert!(status.success(), "download command failed");

    // Verify files exist
    let slug_dir = model_dir.join("harrier-oss-v1-270m-ONNX-q4f16");
    assert!(slug_dir.join("repo/onnx/model_q4f16.onnx").exists(), "ONNX model file missing");
    assert!(slug_dir.join("repo/onnx/model_q4f16.onnx_data").exists(), "ONNX data file missing");
    assert!(slug_dir.join("repo/onnx/tokenizer.json").exists(), "tokenizer.json missing");
    assert!(slug_dir.join("repo/onnx/config.json").exists(), "config.json missing");

    // Test 600M ONNX download
    let temp_dir2 = TempDir::new().expect("create temp dir 2");
    let model_dir2 = temp_dir2.path().join("models");

    let mut cmd = Command::new(&binary);
    cmd.args([
        "download",
        "--slug", "harrier-oss-v1-0.6b-ONNX-q4",
        "--url", "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_q4.onnx",
        "--model-dir", model_dir2.to_str().unwrap(),
    ]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn download command 2");

    let stderr = child.stderr.take().unwrap();
    thread::spawn(move || {
        let reader = std::io::BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("[download stderr] {}", line);
            }
        }
    });

    let status = child.wait().expect("wait for download 2");
    assert!(status.success(), "download command 2 failed");

    // Verify files exist
    let slug_dir = model_dir2.join("harrier-oss-v1-0.6b-ONNX-q4");
    assert!(slug_dir.join("repo/onnx/model_q4.onnx").exists(), "ONNX model file missing");
    assert!(slug_dir.join("repo/onnx/model_q4.onnx_data").exists(), "ONNX data file missing");
    assert!(slug_dir.join("repo/onnx/tokenizer.json").exists(), "tokenizer.json missing");
    assert!(slug_dir.join("repo/onnx/config.json").exists(), "config.json missing");
}

#[test]
fn test_cli_list_embedders() {
    let _ = env_logger::try_init();

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = manifest_dir.join("../target/debug/triq");
    let binary = std::fs::canonicalize(&binary).unwrap_or_else(|_| binary);

    let mut cmd = Command::new(&binary);
    cmd.arg("list-embedders");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output().expect("run list-embedders");
    assert!(output.status.success(), "list-embedders command failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("harrier-oss-v1-270m-ONNX-q4f16"), "missing 270M ONNX");
    assert!(stdout.contains("harrier-oss-v1-270m-MLX-4bit"), "missing 270M MLX");
    assert!(stdout.contains("harrier-oss-v1-0.6b-ONNX-q4"), "missing 600M ONNX");
    assert!(stdout.contains("harrier-oss-v1-0.6b-MLX-4bit"), "missing 600M MLX");
}