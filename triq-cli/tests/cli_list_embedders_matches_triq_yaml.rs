//! Integration test: verify the CLI's `list_embedders` command returns
//! the same gui_embedders list as defined in talentrankiq/gui/triq.yaml.
//!
//! This test builds the release binary and runs it as a subprocess,
//! sending a JSONL request over stdin and verifying the stdout response.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

// Get the directory of this test file (crate root is the current working directory when test runs)
fn crate_root() -> PathBuf {
    // When running tests, the current working directory is the crate root (triq-cli/)
    std::env::current_dir().unwrap()
}

// Get cargo path from PATH env var
fn cargo_path() -> PathBuf {
    // First try to get from PATH
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            let cargo_bin = PathBuf::from(dir).join("cargo");
            if cargo_bin.exists() {
                return cargo_bin;
            }
        }
    }
    // Fallback to common locations
    if let Ok(home) = std::env::var("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo/bin/cargo");
        if cargo_bin.exists() {
            return cargo_bin;
        }
    }
    // Last resort - just use "cargo" and let the OS find it in PATH
    PathBuf::from("cargo")
}

#[test]
fn cli_list_embedders_matches_triq_yaml() {
    // 1. Build the release binary
    let binary_path = build_release_binary();
    
    // 2. Locate the gui triq.yaml
    let config_path = find_gui_triq_yaml();
    
    // 3. Parse expected gui_embedders from triq.yaml
    let expected = parse_gui_embedders_from_yaml(&config_path);
    
    // 4. Run the CLI binary as subprocess with the config
    let mut child = Command::new(&binary_path)
        .env("TRIQ_CONFIG", &config_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn triq CLI");
    
    // Read stderr in background to prevent blocking
    let stderr = child.stderr.take().expect("failed to open stderr");
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            line.clear();
        }
    });
    
    // Give the CLI time to start up and initialize
    std::thread::sleep(std::time::Duration::from_millis(2000));
    
    // 5. Send JSONL request for list_embedders
    let request = r#"{"id":1,"cmd":"list_embedders","params":{}}"#;
    let mut stdin = child.stdin.take().expect("failed to open stdin");
    stdin.write_all(request.as_bytes()).expect("failed to write request");
    stdin.write_all(b"\n").expect("failed to write newline");
    stdin.flush().expect("failed to flush stdin");
    // Keep stdin open - the serve loop reads until EOF
    // We'll drop it after getting the response
    let stdin_handle = stdin;
    
    // 6. Read JSONL response from stdout (skip events, find the response)
    let stdout = child.stdout.take().expect("failed to open stdout");
    let mut reader = BufReader::new(stdout);
    let mut response_line = String::new();
    let response: serde_json::Value;
    let start = std::time::Instant::now();
    loop {
        response_line.clear();
        let bytes_read = reader.read_line(&mut response_line).expect("failed to read response");
        if bytes_read == 0 {
            // Read stderr to see what happened
            let mut stderr = String::new();
            let _ = child.stderr.as_mut().expect("no stderr").read_to_string(&mut stderr);
            panic!("EOF before receiving response (elapsed: {:?}): {}", start.elapsed(), stderr);
        }
        let line = response_line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: serde_json::Value = serde_json::from_str(line)
            .expect("response line is not valid JSON");
        // Check if it's a response (has "id" and "ok" or "err") vs event (has "event")
        if parsed.get("id").is_some() && (parsed.get("ok").is_some() || parsed.get("err").is_some()) {
            response = parsed;
            break;
        }
        // Skip events
        if start.elapsed().as_secs() > 30 {
            panic!("Timeout waiting for response");
        }
    }
    
    // 7. Close stdin and wait for process to exit
    drop(stdin_handle);
    let output = child.wait_with_output().expect("failed to wait on child");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("CLI exited with error: {}", stderr);
    }
    
    // 8. Extract the embedders list from response
    let actual_embedders = extract_embedders_from_response(&response);
    
    // 9. Compare with expected (only fields present in both YAML and CLI response)
    assert_eq!(actual_embedders.len(), expected.len(), "Number of embedders differs");
    for (actual, expected) in actual_embedders.iter().zip(expected.iter()) {
        assert_eq!(actual.slug, expected.slug, "slug mismatch");
        assert_eq!(actual.name, expected.name, "name mismatch");
        assert_eq!(actual.memory_mb, expected.memory_mb, "memory_mb mismatch");
        assert_eq!(actual.default, expected.default, "default mismatch");
        // chunk_size is not in CLI response, skip
    }
}

fn build_release_binary() -> PathBuf {
    // crate_root() = triq-cli/ (crate root)
    // parent() = talentrankiq/ (repo root)
    let cli_dir = crate_root();
    let repo_root = cli_dir.parent().expect("repo root");
    let manifest_path = cli_dir.join("Cargo.toml");
    let cargo = cargo_path();
    
    let status = Command::new(&cargo)
        .args(["build", "--release", "--manifest-path", manifest_path.to_str().unwrap()])
        .current_dir(&repo_root)
        .status()
        .expect("failed to run cargo build");
    
    assert!(status.success(), "cargo build --release failed");
    
    // Binary is built at repo_root/target/release/triq (workspace-level target)
    repo_root.join("target/release/triq")
}

fn find_gui_triq_yaml() -> PathBuf {
    let cli_dir = crate_root();
    let repo_root = cli_dir.parent().expect("repo root");
    let path = repo_root.join("gui/triq.yaml");
    assert!(path.exists(), "gui/triq.yaml not found at {}", path.display());
    path
}

fn parse_gui_embedders_from_yaml(path: &PathBuf) -> Vec<GuiEmbedder> {
    let content = std::fs::read_to_string(path).expect("failed to read triq.yaml");
    let yaml: serde_yaml::Value = serde_yaml::from_str(&content).expect("invalid YAML");
    
    let gui_embedders = yaml["gui_embedders"]
        .as_sequence()
        .expect("gui_embedders not found or not a sequence");
    
    gui_embedders.iter().map(|v| GuiEmbedder {
        slug: v["slug"].as_str().unwrap().to_string(),
        name: v["name"].as_str().unwrap().to_string(),
        memory_mb: v["memory_mb"].as_u64().unwrap() as u32,
        default: v["default"].as_bool().unwrap_or(false),
        chunk_size: v["chunk_size"].as_u64().unwrap() as u32,
    }).collect()
}

fn extract_embedders_from_response(response: &serde_json::Value) -> Vec<GuiEmbedder> {
    let ok = response["ok"].as_array().expect("response.ok is not an array");
    ok.iter().map(|v| GuiEmbedder {
        slug: v["slug"].as_str().unwrap().to_string(),
        name: v["name"].as_str().unwrap().to_string(),
        memory_mb: v["memory_mb"].as_u64().unwrap() as u32,
        default: v["is_default"].as_bool().unwrap_or(false),
        chunk_size: 0, // Not in CLI response
    }).collect()
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct GuiEmbedder {
    slug: String,
    name: String,
    memory_mb: u32,
    default: bool,
    chunk_size: u32,
}