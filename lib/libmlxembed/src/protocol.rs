//! The MLX arm's `run_stdio_over`: the same JSONL contract the ONNX wrapper
//! serves, so the ladder can swap backends without any caller knowing
//! (design 15 §15.4).
//!
//! Three things differ from the ONNX arm on purpose:
//!
//! 1. **It serves fewer commands.** `embed` is the whole point; `rerank` and
//!    `generate` have no MLX implementation here and answer with a clear
//!    `unsupported_cmd` naming the arm. Silence would be read as "ran and
//!    returned nothing", which is how a broken rung passes a test.
//! 2. **`load` names the runtime.** The response carries
//!    `runtime: "mlx"` (and `family`, `pooling`, `mask`, `mask_mode`) so a
//!    report can never imply "accelerated" for something that quietly ran on
//!    the CPU — the trap design 15 §15.4.1 exists to prevent.
//! 3. **Tokenization is the engine's, not a second one.** Texts go through
//!    `libtokenizer` with the same truncation the ONNX arm applies, so the two
//!    arms see identical ids and a parity test compares models rather than
//!    tokenizers (design 15 §15.5.3.3).

use std::io::{BufRead, Write};
use std::panic;
use std::path::Path;

use libtokenizer::TokenizerHandle;
use serde_json::Value;

use crate::{EmbedError, Embedder};

type CmdResult = Result<serde_json::Map<String, Value>, (String, String)>;
type Payload = serde_json::Map<String, Value>;

/// Sessions the arm can hold. One model at a time is deliberate: MLX models
/// are GPU-resident, and the cascade's own rule is that only one embedder is
/// resident at a time (the scout unloads before the heavy model loads).
pub struct SessionManager {
    sessions: std::collections::HashMap<u32, Loaded>,
    next_handle: u32,
}

struct Loaded {
    embedder: Embedder,
    tokenizer: TokenizerHandle,
    /// Truncation cap, from the load payload — identical to the ONNX arm's.
    max_ctx: usize,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            next_handle: 1,
        }
    }

    /// Errors come back as `(code, msg)` because that is the wire shape: a
    /// caller mapping a failure onto the protocol must not have to invent a
    /// code, and `unsupported_model` (skip this rung) must stay distinct from
    /// `load_failed` (something is wrong).
    pub fn load(
        &mut self,
        model_path: &str,
        max_ctx: usize,
    ) -> Result<(u32, LoadReport), (String, String)> {
        let dir = Path::new(model_path);
        // Wrap in catch_unwind to catch any panics from C++ exceptions
        // that cross the FFI boundary (cxx bridge converts them to panics).
        let embedder = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Embedder::load(dir)
        }))
        .map_err(|_| ("load_failed".to_string(), "MLX embedder load panicked (likely unhandled C++ exception)".to_string()))?
        .map_err(|e| match e {
            EmbedError::Unsupported(m) => ("unsupported_model".to_string(), m),
            EmbedError::Load(m) => ("load_failed".to_string(), m),
        })?;
        // The tokenizer sits next to the weights, as in the ONNX layout. A
        // missing one is a load error, not an empty-vocabulary fallback: ids
        // are the input to everything this arm computes.
        let tokenizer = TokenizerHandle::from_file(&dir.join("tokenizer.json")).map_err(|e| {
            ("load_failed".to_string(), format!("tokenizer.json next to the weights: {e}"))
        })?;
        let report = LoadReport::from(&embedder);
        let handle = self.next_handle;
        self.next_handle += 1;
        self.sessions.insert(
            handle,
            Loaded {
                embedder,
                tokenizer,
                max_ctx: max_ctx.max(1),
            },
        );
        Ok((handle, report))
    }

    pub fn unload(&mut self, handle: u32) -> Result<(), (String, String)> {
        self.sessions.remove(&handle).map(|_| ()).ok_or_else(|| {
            ("no_session".to_string(), format!("session {handle} not loaded"))
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn embed(
        &self,
        handle: u32,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, (String, String)> {
        let loaded = self.sessions.get(&handle).ok_or_else(|| {
            ("no_session".to_string(), format!("session {handle} not loaded"))
        })?;
        // Same call, same cap, same `add_special_tokens=true` as the ONNX arm:
        // for a last-token-pooling family the appended token IS the pooled
        // position, so dropping it changes every vector.
        let rows: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| {
                loaded
                    .tokenizer
                    .encode_truncated(t, true, loaded.max_ctx)
                    .map(|enc| enc.ids)
                    .unwrap_or_default()
            })
            .collect();
        // Wrap in catch_unwind to catch any panics from C++ exceptions
        // that cross the FFI boundary during embedding.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loaded.embedder.embed(&rows)
        }))
        .map_err(|_| ("embed_failed".to_string(), "MLX embedder embed panicked (likely unhandled C++ exception)".to_string()))?
        .map_err(|e| ("embed_failed".to_string(), format!("{e}")))
    }
}

/// What `load` reports back. Every field here is something the gui shows or a
/// test asserts on, so "which runtime actually served this" is never a guess.
#[derive(Debug)]
pub struct LoadReport {
    pub runtime: &'static str,
    pub dim: usize,
    pub family: String,
    pub pooling: String,
    pub mask: String,
    pub layers: usize,
}

impl LoadReport {
    fn from(e: &Embedder) -> Self {
        // `describe()` is the single place that knows the family names; parse
        // it once here rather than widening Embedder's public surface.
        let d = e.describe();
        let field = |k: &str| {
            d.split_whitespace()
                .find_map(|w| w.strip_prefix(&format!("{k}=")))
                .unwrap_or("?")
                .to_string()
        };
        Self {
            runtime: "mlx",
            dim: e.dim(),
            family: field("mlx").replace([':', ','], ""),
            pooling: field("pooling"),
            mask: field("mask"),
            layers: field("layers").parse().unwrap_or(0),
        }
    }
}

pub fn run_stdio_over(input: impl BufRead, mut output: impl Write) {
    // Install a panic hook to capture any panics in this thread
    // This helps diagnose crashes from unhandled C++ exceptions
    let orig_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            format!("MLX embedder panic: {}", s)
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            format!("MLX embedder panic: {}", s)
        } else {
            "MLX embedder panic: unknown payload".to_string()
        };
        let loc = info.location().map(|l| format!(" at {}:{}:{}", l.file(), l.line(), l.column())).unwrap_or_default();
        let full_msg = format!("{}", msg);
        eprintln!("{}", full_msg);
        eprintln!("  Location:{}", loc);
        
        // Also write to a crash log file for post-mortem analysis
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/talentrankiq_mlx_crash.log") {
            use std::io::Write;
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(file, "[{}] {}", timestamp, full_msg);
            let _ = writeln!(file, "  Location:{}", loc);
        }
        
        // Also call the original hook
        orig_hook(info);
    }));

    log::info!("libmlxembed session loop starting (Metal)");
    let mut sessions = SessionManager::new();

    for (line_num, line_result) in input.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                log::error!("failed to read line {line_num}: {e}");
                break;
            }
        };
        let resp = match serde_json::from_str::<Value>(&line) {
            Ok(v) => {
                let id = v.get("id").and_then(|i| i.as_str()).map(str::to_string);
                match v.get("payload") {
                    Some(p) if p.is_object() => {
                        let payload = p.as_object().cloned().unwrap_or_default();
                        match handle_request(&payload, &mut sessions) {
                            Ok(p) => envelope(id, "response", p),
                            Err((code, msg)) => error_envelope(id, &code, &msg),
                        }
                    }
                    _ => error_envelope(id, "bad_cmd", "payload missing \"cmd\" field"),
                }
            }
            Err(e) => error_envelope(
                None,
                "bad_req",
                &format!("invalid JSON on line {line_num}: {e}"),
            ),
        };
        let _ = writeln!(output, "{resp}");
        let _ = output.flush();
    }
    log::info!("EOF on input, dropping sessions and exiting loop");
}

fn envelope(id: Option<String>, kind: &str, payload: Payload) -> String {
    let v = serde_json::json!({ "id": id, "type": kind, "payload": payload });
    v.to_string()
}

fn error_envelope(id: Option<String>, code: &str, msg: &str) -> String {
    envelope(
        id,
        "error",
        serde_json::json!({ "code": code, "msg": msg })
            .as_object()
            .cloned()
            .unwrap_or_default(),
    )
}

fn handle_request(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let cmd = match payload.get("cmd").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return Err(("bad_cmd".into(), "payload missing \"cmd\" field".into())),
    };
    match cmd {
        "probe" => handle_probe(),
        "eplist" => Ok(serde_json::json!({ "eps": ["mlx"] })
            .as_object()
            .cloned()
            .unwrap()),
        "load" => handle_load(payload, sessions),
        "embed" => handle_embed(payload, sessions),
        "unload" => handle_unload(payload, sessions),
        // Named, not silently ignored: a caller that gets no error here would
        // read "ran, returned nothing" and cache that as a real result.
        "rerank" | "generate" | "generate_constrained" => Err((
            "unsupported_cmd".into(),
            format!("{cmd} is ONNX-only; the mlx arm serves embed"),
        )),
        other => Err(("unknown_cmd".into(), format!("unknown cmd: {other}"))),
    }
}

fn handle_probe() -> CmdResult {
    Ok(serde_json::json!({
        "runtime": "mlx",
        "available": true,
        "commands": ["probe", "eplist", "load", "embed", "unload"],
        "unsupported_commands": ["rerank", "generate", "generate_constrained"],
    })
    .as_object()
    .cloned()
    .unwrap())
}

fn handle_load(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let model_path = payload
        .get("model_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ("bad_req".into(), "load requires \"model_path\"".into()))?;
    // The engine's context cap travels with the load, so both arms truncate
    // identically. A disagreement here is invisible in the vectors and looks
    // exactly like a broken conversion (design 15 §15.5.3.3).
    let max_ctx = payload
        .get("max_ctx")
        .and_then(|v| v.as_u64())
        .unwrap_or(512) as usize;
    let (handle, report) = sessions.load(model_path, max_ctx)?;
    Ok(serde_json::json!({
        "session": handle,
        "kind": "embedder",
        "dim": report.dim,
        "runtime": report.runtime,
        "family": report.family,
        "pooling": report.pooling,
        "mask": report.mask,
        "layers": report.layers,
        // The ONNX arm's fields, present and truthful: this arm has one
        // session and no thread pool, so it reports 1 rather than omitting
        // them and leaving a caller to guess.
        "embed_sessions": 1,
        "threads_per_session": 1,
    })
    .as_object()
    .cloned()
    .unwrap())
}

fn handle_embed(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let handle = payload
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ("bad_req".into(), "embed requires \"session\"".into()))?
        as u32;
    let texts: Vec<&str> = payload
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ("bad_req".into(), "embed requires \"texts\" array".into()))?
        .iter()
        .map(|v| v.as_str().unwrap_or(""))
        .collect();
    let embeddings = sessions.embed(handle, &texts)?;
    Ok(serde_json::json!({ "embeddings": embeddings })
        .as_object()
        .cloned()
        .unwrap())
}

fn handle_unload(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let handle = payload
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ("bad_req".into(), "unload requires \"session\"".into()))?
        as u32;
    sessions.unload(handle)?;
    Ok(serde_json::json!({ "ok": true })
        .as_object()
        .cloned()
        .unwrap())
}

#[cfg(all(test, target_os = "macos", feature = "mlx"))]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    fn call(input: &str) -> Vec<Value> {
        let mut out = Vec::new();
        run_stdio_over(std::io::Cursor::new(input), &mut out);
        String::from_utf8(out)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("json line"))
            .collect()
    }

    #[test]
    fn probe_and_eplist_serve_the_same_contract_shape() {
        let lines = call(concat!(
            r#"{"id":"1","type":"request","payload":{"cmd":"probe"}}"#,
            "
",
            r#"{"id":"2","type":"request","payload":{"cmd":"eplist"}}"#,
            "
",
        ));
        assert_eq!(lines[0]["id"], "1");
        assert_eq!(lines[0]["type"], "response");
        assert_eq!(lines[0]["payload"]["runtime"], "mlx");
        assert_eq!(lines[1]["payload"]["eps"][0], "mlx");
    }

    #[test]
    fn commands_this_arm_does_not_serve_say_so_instead_of_returning_nothing() {
        for cmd in ["rerank", "generate", "generate_constrained"] {
            let line = format!(r#"{{"id":"1","type":"request","payload":{{"cmd":"{cmd}"}}}}"#);
            let out = call(&line);
            assert_eq!(out[0]["type"], "error", "{cmd} must not look like it ran");
            assert_eq!(out[0]["payload"]["code"], "unsupported_cmd", "{cmd}");
            assert!(
                out[0]["payload"]["msg"].as_str().unwrap().contains("embed"),
                "the error names what the arm does serve: {cmd}"
            );
        }
    }

    #[test]
    fn an_unknown_command_is_unknown_not_unsupported() {
        let out = call(r#"{"id":"1","type":"request","payload":{"cmd":"embedd"}}"#);
        assert_eq!(out[0]["payload"]["code"], "unknown_cmd");
    }

    #[test]
    fn embedding_an_unloaded_session_is_an_error_not_a_crash() {
        let out = call(r#"{"id":"1","type":"request","payload":{"cmd":"embed","session":7,"texts":["x"]}}"#);
        assert_eq!(out[0]["type"], "error");
        assert_eq!(out[0]["payload"]["code"], "no_session");
    }

    /// Test that reproduces the MLX embedder crash scenario:
    /// When loading a non-existent or corrupt model directory, the catch_unwind
    /// wrapper in SessionManager::load should catch any panic (including those
    /// from unhandled C++ exceptions converted by the cxx bridge) and return
    /// a proper error instead of crashing the process via std::terminate().
    #[test]
    fn load_nonexistent_model_returns_error_not_crash() {
        let mut sessions = SessionManager::new();
        let result = sessions.load("/nonexistent/model/path", 512);
        assert!(result.is_err());
        let (code, msg) = result.unwrap_err();
        assert_eq!(code, "load_failed");
        assert!(msg.contains("load failed") || msg.contains("No such file") || msg.contains("not found") || msg.contains("Failed to load"));
    }

    #[test]
    fn load_corrupt_model_missing_tokenizer_returns_error() {
        let temp_dir = std::env::temp_dir().join("libmlxembed_test");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let mut sessions = SessionManager::new();
        let result = sessions.load(temp_dir.to_str().unwrap(), 512);
        assert!(result.is_err());
        let (code, _msg) = result.unwrap_err();
        assert_eq!(code, "load_failed");
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn catch_unwind_catches_panic_in_embedder_load() {
        let mut sessions = SessionManager::new();
        let catch_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sessions.load("/definitely/does/not/exist", 512)
        }));
        assert!(catch_result.is_ok());
        let load_result = catch_result.unwrap();
        assert!(load_result.is_err());
        let (code, _msg) = load_result.unwrap_err();
        assert_eq!(code, "load_failed");
    }
}
