//! `onnxwrapperrs` — the only component that links ONNX Runtime.
//!
//! One wire protocol (envelope JSONL, one request/response per line), always
//! served in-process: [`run_stdio_over`] runs on a dedicated thread over an
//! OS-pipe pair inside the host process (modelcaller's `wrapper.rs` spawns
//! it — there is no wrapper binary anymore). All ORT work (session load,
//! embed, rerank, generate) happens synchronously on that thread, so the
//! host's main thread stays responsive; EOF on the request pipe shuts the
//! loop (and the ORT sessions) down.

pub mod generate;
pub mod probe;
pub mod projection;
pub mod protocol;
pub mod session;

use std::io::{BufRead, Write};

use protocol::{Envelope, Payload};
use session::SessionManager;

type CmdResult = Result<Payload, (String, String)>;

/// Serve the wrapper protocol over arbitrary streams — the library core. The
/// host passes an OS-pipe pair connected to a dedicated wrapper thread.
/// Returns when `input` hits EOF, which is how a host shuts the wrapper down
/// (closing the pipe drops the ORT sessions with it).
pub fn run_stdio_over(input: impl BufRead, mut output: impl Write) {
    log::info!("onnxwrapperrs session loop starting");

    let mut sessions = SessionManager::new();

    for (line_num, line_result) in input.lines().enumerate() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                log::error!("failed to read line {line_num}: {e}");
                break;
            }
        };

        let req: Envelope = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let err = Envelope::error(
                    None,
                    "bad_req",
                    &format!("invalid JSON on line {line_num}: {e}"),
                );
                let _ = writeln!(output, "{err}");
                let _ = output.flush();
                continue;
            }
        };

        let resp = handle_request(&req, &mut sessions);
        let out = serde_json::to_string(&resp).unwrap();
        let _ = writeln!(output, "{out}");
        let _ = output.flush();
    }

    log::info!("EOF on input, dropping sessions and exiting loop");
}

fn handle_request(req: &Envelope, sessions: &mut SessionManager) -> Envelope {
    let id = req.id.clone();
    let payload = &req.payload;

    let cmd_str = match payload.get("cmd").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => {
            return Envelope::error(id, "bad_cmd", "payload missing \"cmd\" field");
        }
    };

    let result = match cmd_str {
        "probe" => handle_probe(),
        "load" => handle_load(payload, sessions),
        "embed" => handle_embed(payload, sessions),
        "rerank" => handle_rerank(payload, sessions),
        "generate" => handle_generate(payload, sessions, false),
        "generate_constrained" => handle_generate(payload, sessions, true),
        "unload" => handle_unload(payload, sessions),
        "eplist" => handle_eplist(),
        _ => {
            return Envelope::error(
                id.clone(),
                "unknown_cmd",
                &format!("unknown cmd: {cmd_str}"),
            );
        }
    };

    match result {
        Ok(payload) => Envelope::response(id, payload),
        Err((code, msg)) => Envelope::error(id, &code, &msg),
    }
}

fn handle_probe() -> CmdResult {
    let caps = probe::probe_capabilities();
    let val = serde_json::to_value(&caps).unwrap();
    Ok(val.as_object().cloned().unwrap())
}

fn handle_load(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let model_path = payload
        .get("model_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ("bad_req".into(), "load requires \"model_path\"".into()))?;
    let exec_provider = payload
        .get("exec_provider")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");
    let precision = payload
        .get("precision")
        .and_then(|v| v.as_str())
        .unwrap_or("fp32");
    let intra_threads = payload
        .get("intra_threads")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let inter_threads = payload
        .get("inter_threads")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    // Sessions a batched embed is spread over (design 15 §15.7). Absent or 0
    // = auto: the wrapper picks by the model's weight footprint (a
    // cache-resident model is compute-bound and scales across sessions; a big
    // one is bandwidth-bound and does not). The total thread budget is divided
    // across the pool, so this never adds silicon — it only changes the shape.
    let embed_sessions = payload
        .get("embed_sessions")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).min(64))
        .unwrap_or(0);
    // Raise ORT's logging for this load so node placement is visible: the only
    // way to tell a working provider from a silent CPU fallback (design 15
    // §15.4.1). Opt-in, because it is verbose and per-load.
    let provider_logging = payload
        .get("provider_logging")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let info = sessions.load(
        model_path,
        exec_provider,
        precision,
        intra_threads,
        inter_threads,
        embed_sessions,
        provider_logging,
    )?;
    Ok(serde_json::json!({
        "session": info.handle,
        "kind": info.kind,
        "dim": info.dim,
        "embed_sessions": info.embed_sessions,
        "threads_per_session": info.threads_per_session,
        "exec_provider": info.exec_provider,
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
    let texts = payload
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ("bad_req".into(), "embed requires \"texts\" array".into()))?
        .iter()
        .map(|v| v.as_str().unwrap_or(""))
        .collect::<Vec<_>>();
    let opts = payload.get("opts").and_then(|v| v.as_object()).cloned();

    let multi_vector = opts
        .as_ref()
        .and_then(|o| o.get("multi_vector"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if multi_vector {
        let mv = sessions.embed_multi(handle, &texts, opts.as_ref())?;
        let embeddings_b64: Vec<String> = mv.blobs.iter().map(|b| encode_b64(b)).collect();
        return Ok(serde_json::json!({
            "multi_vector": true,
            "dim": mv.dim,
            "counts": mv.counts,
            "embeddings_b64": embeddings_b64,
        })
        .as_object()
        .cloned()
        .unwrap());
    }

    let embeddings = sessions.embed(handle, &texts, opts.as_ref())?;
    Ok(serde_json::json!({ "embeddings": embeddings })
        .as_object()
        .cloned()
        .unwrap())
}

fn encode_b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn handle_rerank(payload: &Payload, sessions: &mut SessionManager) -> CmdResult {
    let handle = payload
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ("bad_req".into(), "rerank requires \"session\"".into()))?
        as u32;
    let query = payload
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ("bad_req".into(), "rerank requires \"query\"".into()))?;
    let docs = payload
        .get("docs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ("bad_req".into(), "rerank requires \"docs\" array".into()))?
        .iter()
        .map(|v| v.as_str().unwrap_or(""))
        .collect::<Vec<_>>();
    let opts = payload.get("opts").and_then(|v| v.as_object()).cloned();

    let scores = sessions.rerank(handle, query, &docs, opts.as_ref())?;
    Ok(serde_json::json!({ "scores": scores })
        .as_object()
        .cloned()
        .unwrap())
}

fn handle_generate(
    payload: &Payload,
    sessions: &mut SessionManager,
    constrained: bool,
) -> CmdResult {
    let handle = payload
        .get("session")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ("bad_req".into(), "generate requires \"session\"".into()))?
        as u32;
    let prompt = payload
        .get("prompt")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ("bad_req".into(), "generate requires \"prompt\"".into()))?;
    let opts = payload.get("opts").and_then(|v| v.as_object());
    let max_new_tokens = opts
        .and_then(|o| o.get("max_new_tokens"))
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(256);
    let stop_ids: Vec<u32> = opts
        .and_then(|o| o.get("stop_ids"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default();
    let stop_strings: Vec<String> = opts
        .and_then(|o| o.get("stop_strings"))
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let schema = payload.get("schema");
    if constrained && schema.is_none() {
        return Err((
            "bad_req".into(),
            "generate_constrained requires \"schema\"".into(),
        ));
    }

    let out = sessions.generate(
        handle,
        prompt,
        max_new_tokens,
        &stop_ids,
        &stop_strings,
        schema,
    )?;
    Ok(serde_json::json!({
        "text": out.text,
        "n_new_tokens": out.n_new_tokens,
        "eos": out.finished,
    })
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

fn handle_eplist() -> CmdResult {
    Ok(serde_json::json!({
        "eps": session::available_eps()
    })
    .as_object()
    .cloned()
    .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The protocol loop serves request/response pairs over arbitrary
    /// streams — here an in-memory buffer pair (the in-process gui path does
    /// the same over OS pipes).
    #[test]
    fn run_stdio_over_serves_probe_over_memory_streams() {
        let input = concat!(
            r#"{"id":"1","type":"request","payload":{"cmd":"probe"}}"#,
            "\n",
            r#"{"id":"2","type":"request","payload":{"cmd":"eplist"}}"#,
            "\n",
        );
        let mut output = Vec::new();
        run_stdio_over(std::io::Cursor::new(input), &mut output);
        let out = String::from_utf8(output).expect("utf8 output");
        let mut lines = out.lines();
        let probe: serde_json::Value =
            serde_json::from_str(lines.next().expect("probe response")).expect("json");
        assert_eq!(probe["id"], "1");
        assert_eq!(probe["type"], "response");
        assert!(probe["payload"].is_object(), "probe returns capabilities");
        let eps: serde_json::Value =
            serde_json::from_str(lines.next().expect("eplist response")).expect("json");
        assert_eq!(eps["id"], "2");
        assert!(eps["payload"]["eps"].is_array(), "eplist returns providers");
    }

    /// Malformed lines answer with a `bad_req` error envelope (no id) and the
    /// loop keeps going.
    #[test]
    fn run_stdio_over_survives_bad_lines() {
        let input = "not json\n{\"id\":\"7\",\"type\":\"request\",\"payload\":{}}\n";
        let mut output = Vec::new();
        run_stdio_over(std::io::Cursor::new(input), &mut output);
        let out = String::from_utf8(output).expect("utf8 output");
        let mut lines = out.lines();
        let bad: serde_json::Value =
            serde_json::from_str(lines.next().expect("bad_req response")).expect("json");
        assert_eq!(bad["type"], "error");
        assert_eq!(bad["payload"]["code"], "bad_req");
        let noid: serde_json::Value =
            serde_json::from_str(lines.next().expect("bad_cmd response")).expect("json");
        assert_eq!(noid["id"], "7");
        assert_eq!(noid["payload"]["code"], "bad_cmd");
    }
}
