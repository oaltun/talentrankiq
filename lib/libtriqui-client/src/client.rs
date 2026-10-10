//! Client half of the gui↔serve protocol.
//!
//! Concurrency rules (see `gui/design/app1/thin-gui.md`): requests are fire-and-forget
//! into the transport; a dedicated reader thread drains responses forever and
//! demuxes by id into per-request one-shot channels, so a slow or long-running
//! command never blocks another. Events go to a single channel the host drains
//! (the Tauri bridge forwards them as Tauri events).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

use crate::protocol::{Event, Inbound, ProtocolError, Request, MAX_LINE_BYTES};

#[derive(Debug, Error, Clone)]
pub enum ClientError {
    #[error("failed to write to triqui-cli: {0}")]
    Write(String),
    #[error("serialization error: {0}")]
    Json(String),
    #[error("triqui-cli serve has exited")]
    Closed,
    #[error("triqui-cli error: {0}")]
    Protocol(String),
    /// The serve process answered with a structured error envelope — carries
    /// the full [`ProtocolError`] so the host can rehydrate typed domain
    /// errors (e.g. `DomainError` / `RecoveryError`) from `domain`.
    #[error("triqui-cli error: {0}")]
    Domain(ProtocolError),
}

struct Inner {
    /// The request side of the transport: the in-memory writer handed to
    /// [`TriqiClient::over`]. Dropping it is EOF for serve.
    stdin: Mutex<Option<Box<dyn Write + Send>>>,
    pending: Mutex<HashMap<u64, std::sync::mpsc::Sender<Result<Value, ClientError>>>>,
    next_id: AtomicU64,
    closed: AtomicBool,
    events_tx: std::sync::mpsc::Sender<Event>,
    events_rx: Mutex<Option<std::sync::mpsc::Receiver<Event>>>,
}

/// A live connection to one serve loop. Cloneable; all clones share the
/// transport, the id counter, and the event channel.
pub struct TriqiClient {
    inner: Arc<Inner>,
}

impl Clone for TriqiClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl TriqiClient {
    /// Connect to an already-running serve loop over an arbitrary transport
    /// (e.g. [`crate::transport::duplex`] for the gui's in-process proxy —
    /// one binary, no child). Taking ownership of `write` is the shutdown
    /// signal: dropping it (via [`TriqiClient::shutdown`] or the last
    /// client) is EOF for the serve reader, which then tears the engine down.
    pub fn over(read: Box<dyn Read + Send>, write: Box<dyn Write + Send>) -> Self {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let inner = Arc::new(Inner {
            stdin: Mutex::new(Some(write)),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            events_tx,
            events_rx: Mutex::new(Some(events_rx)),
        });
        {
            let inner = Arc::clone(&inner);
            thread::Builder::new()
                .name("triqui-cli-reader".into())
                .spawn(move || reader_loop(inner, read))
                .expect("spawn triqui-cli reader");
        }
        Self { inner }
    }

    /// Send a request and block until its response arrives. Concurrent calls
    /// are safe: the reader demuxes by id, so a pending long command does not
    /// delay others (this is what keeps the GUI responsive while an import
    /// runs).
    pub fn call(&self, cmd: &str, params: &impl serde::Serialize) -> Result<Value, ClientError> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(ClientError::Closed);
        }
        let id = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel();
        self.inner.pending.lock().unwrap().insert(id, tx);

        let request = Request {
            id,
            cmd: cmd.to_string(),
            params: serde_json::to_value(params).map_err(|e| ClientError::Json(e.to_string()))?,
        };
        let line = serde_json::to_string(&request).map_err(|e| ClientError::Json(e.to_string()))?;

        {
            let mut stdin = self.inner.stdin.lock().unwrap();
            if self.inner.closed.load(Ordering::SeqCst) {
                self.inner.pending.lock().unwrap().remove(&id);
                return Err(ClientError::Closed);
            }
            match stdin.as_mut() {
                Some(w) => {
                    if let Err(e) = writeln!(w, "{line}").and_then(|_| w.flush()) {
                        self.inner.pending.lock().unwrap().remove(&id);
                        return Err(ClientError::Write(e.to_string()));
                    }
                }
                None => {
                    self.inner.pending.lock().unwrap().remove(&id);
                    return Err(ClientError::Closed);
                }
            }
        }

        match rx.recv() {
            Ok(res) => res,
            // Sender dropped without replying: the reader thread gave up
            // (process died) — every waiter is failed there, so this only
            // happens on a bug; treat as closed.
            Err(_) => Err(ClientError::Closed),
        }
    }

    /// [`call`] + deserialize the `ok` payload.
    pub fn call_typed<T: DeserializeOwned>(
        &self,
        cmd: &str,
        params: &impl serde::Serialize,
    ) -> Result<T, ClientError> {
        let value = self.call(cmd, params)?;
        serde_json::from_value(value).map_err(|e| ClientError::Json(e.to_string()))
    }

    /// Take the event receiver (once). The host must drain it continuously;
    /// events are fire-and-forget progress/status traffic.
    pub fn take_event_receiver(&self) -> Option<std::sync::mpsc::Receiver<Event>> {
        self.inner.events_rx.lock().unwrap().take()
    }

    /// True once the serve process's stdout has closed (crash or exit).
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::SeqCst)
    }

    /// Graceful shutdown: drop the write end (serve exits on EOF). Safe to
    /// call multiple times.
    pub fn shutdown(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.fail_all_pending(ClientError::Closed);
        *self.inner.stdin.lock().unwrap() = None;
    }

    fn fail_all_pending(&self, err: ClientError) {
        let mut pending = self.inner.pending.lock().unwrap();
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(err.clone()));
        }
    }
}

/// Reader loop: parse each line, route responses to waiters, push events.
/// On EOF mark the client closed and fail every waiter with [`ClientError::Closed`].
fn reader_loop(inner: Arc<Inner>, stdout: Box<dyn Read + Send>) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut buf = Vec::new();
        // Bounded read so a protocol bug fails loudly instead of consuming
        // memory (`BufRead` is implemented for `Take<_>`).
        let n = match reader
            .by_ref()
            .take(MAX_LINE_BYTES)
            .read_until(b'\n', &mut buf)
        {
            Ok(n) => n,
            Err(e) => {
                log::warn!("triqui-cli stdout read error: {e}");
                break;
            }
        };
        if n == 0 {
            break; // EOF
        }
        if n as u64 >= MAX_LINE_BYTES {
            log::error!("triqui-cli sent an over-long line; dropping it");
            continue;
        }
        let line = match std::str::from_utf8(&buf[..n]) {
            Ok(s) => s.trim(),
            Err(_) => continue,
        };
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Inbound>(line) {
            Ok(Inbound::Response(resp)) => {
                let payload = match (&resp.ok, &resp.err) {
                    (Some(v), _) => Ok(v.clone()),
                    (None, Some(err)) => Err(ClientError::Domain(err.clone())),
                    (None, None) => Ok(Value::Null),
                };
                if let Some(tx) = inner.pending.lock().unwrap().remove(&resp.id) {
                    let _ = tx.send(payload);
                }
            }
            Ok(Inbound::Event(event)) => {
                // Unbounded std channel; the cli throttles progress sources
                // (≤10 Hz per operation) so this cannot flood.
                let _ = inner.events_tx.send(event);
            }
            Err(e) => {
                log::warn!("triqui-cli sent an unparsable line: {e}");
            }
        }
    }
    inner.closed.store(true, Ordering::SeqCst);
    let mut pending = inner.pending.lock().unwrap();
    for (_, tx) in pending.drain() {
        let _ = tx.send(Err(ClientError::Closed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Response;

    #[test]
    fn response_routes_to_waiter_shape() {
        // Parse-level check of the demux contract (no process needed). The
        // full call/demux round-trip is covered by the integration tests,
        // which run the real serve loop in-process.
        let resp: Inbound = serde_json::from_str(r#"{"id":42,"ok":{"v":1}}"#).unwrap();
        match resp {
            Inbound::Response(Response { id, ok, err }) => {
                assert_eq!(id, 42);
                assert_eq!(ok.unwrap()["v"], 1);
                assert!(err.is_none());
            }
            _ => panic!("expected response"),
        }
    }
}
