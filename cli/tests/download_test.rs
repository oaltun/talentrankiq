use std::process::{Command, Stdio};

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