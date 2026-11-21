use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::error::{ModelError, Result};
use crate::protocol::Envelope;
pub struct Wrapper {
    /// Request pipe to the wrapper thread. `None` after [`Wrapper::kill`]:
    /// dropping it is the shutdown signal (EOF ends the wrapper loop).
    stdin: Option<Box<dyn Write + Send>>,
    stdout: BufReader<Box<dyn std::io::Read + Send>>,
    next_id: u64,
    /// Set by the wrapper thread right before it returns. `kill` only sends
    /// EOF — the wrapper loop finishes any in-flight request (a session load
    /// takes seconds) before it sees the EOF — so a teardown that must not
    /// race ONNX Runtime's global cleanup waits on this instead of assuming
    /// the thread is already gone.
    exited: Arc<(Mutex<bool>, Condvar)>,
}

impl Wrapper {
    /// Bring an arm up in-process: the chosen backend's `run_stdio_over` runs
    /// on a dedicated thread over an OS-pipe pair. All blocking inference —
    /// session load, embed, rerank, generate — happens on that thread, so the
    /// host's loop thread (and everything above it) never waits on it.
    /// There is nothing to force-kill: shutdown is EOF on the request pipe,
    /// after which the thread drops its sessions.
    ///
    /// The two arms serve the same JSONL contract (design 15 §15.4), so the
    /// only thing that differs here is which loop runs and what the thread is
    /// named for — which is why the ladder can pick per model without any
    /// caller seeing a different protocol.
    pub fn spawn(runtime: crate::commands::Runtime) -> Result<Self> {
        let thread_name = match runtime {
            crate::commands::Runtime::Ort => "onnxwrapperrs",
            crate::commands::Runtime::Mlx => "libmlxembed",
        };
        let pipe = |what: &str| {
            std::io::pipe().map_err(|e| {
                ModelError::fatal(
                    "wrapper_spawn",
                    format!("failed to create wrapper {what} pipe: {e}"),
                    1,
                )
            })
        };
        let (req_read, req_write) = pipe("request")?;
        let (resp_read, resp_write) = pipe("response")?;

        // Dropping the JoinHandle detaches the thread: shutdown is EOF on the
        // request pipe (see `kill`), never a join. `exited` carries the one
        // bit a joining teardown needs — "the loop has returned".
        let exited = Arc::new((Mutex::new(false), Condvar::new()));
        let flag = Arc::clone(&exited);
        let _worker = std::thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                match runtime {
                    crate::commands::Runtime::Ort => {
                        libonnxwrapperrs::run_stdio_over(BufReader::new(req_read), resp_write)
                    }
                    crate::commands::Runtime::Mlx => {
                        // Unreachable unless something spawned the Metal arm in
                        // a build without it. `spawn` is only called with Mlx
                        // when `libmlxembed::AVAILABLE`, so this is a loud
                        // refusal rather than a silent ONNX-shaped no-op.
                        if !libmlxembed::AVAILABLE {
                            log::error!("mlx wrapper requested but this build has no MLX arm");
                        } else {
                            libmlxembed::run_stdio_over(BufReader::new(req_read), resp_write);
                        }
                    }
                }
                let (lock, cvar) = &*flag;
                *lock.lock().expect("wrapper exited flag poisoned") = true;
                cvar.notify_all();
            })
            .map_err(|e| {
                ModelError::fatal(
                    "wrapper_spawn",
                    format!("failed to spawn wrapper thread: {e}"),
                    1,
                )
            })?;

        log::info!("wrapper running in-process on dedicated thread");
        Ok(Wrapper {
            stdin: Some(Box::new(req_write)),
            stdout: BufReader::new(Box::new(resp_read)),
            next_id: 1,
            exited,
        })
    }

    pub fn send(&mut self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let id = self.next_id.to_string();
        self.next_id += 1;

        let env = Envelope {
            id: Some(id.clone()),
            r#type: "request".to_string(),
            payload,
        };
        let line = env.to_line().map_err(ModelError::load_err)?;
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| ModelError::fatal("wrapper_exit", "wrapper stdin already closed", 1))?;
        writeln!(stdin, "{line}")
            .map_err(|e| ModelError::load_err(format!("failed to write to wrapper stdin: {e}")))?;
        stdin
            .flush()
            .map_err(|e| ModelError::load_err(format!("failed to flush wrapper stdin: {e}")))?;

        loop {
            let mut buf = String::new();
            match self.stdout.read_line(&mut buf) {
                Ok(0) => {
                    return Err(ModelError::fatal(
                        "wrapper_exit",
                        "wrapper closed stdout before responding",
                        1,
                    ))
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(ModelError::load_err(format!(
                        "failed to read wrapper stdout: {e}"
                    )))
                }
            }
            let buf = buf.trim();
            if buf.is_empty() {
                continue;
            }
            let resp = Envelope::parse_line(buf).map_err(|e| {
                ModelError::load_err(format!("unparseable wrapper response: {e}: {buf}"))
            })?;
            if resp.id.as_deref() == Some(&id) {
                return Ok(resp.payload);
            }
        }
    }

    /// Shut the wrapper down: close the request pipe — the wrapper loop
    /// finishes any in-flight request, sees EOF, exits, and drops the ORT
    /// sessions.
    pub fn kill(&mut self) {
        self.stdin = None;
    }

    /// Wait (bounded) for the wrapper thread to finish its in-flight request
    /// and exit after [`Wrapper::kill`]. True if it exited within `timeout`.
    /// Always pair a `kill` with this on teardown: process exit while the
    /// thread is mid-request races ONNX Runtime's global teardown (observed
    /// as a SIGSEGV inside `CreateSession`).
    pub fn wait_exited(&self, timeout: Duration) -> bool {
        let (lock, cvar) = &*self.exited;
        let done = lock.lock().expect("wrapper exited flag poisoned");
        let (done, res) = cvar
            .wait_timeout_while(done, timeout, |exited| !*exited)
            .expect("wrapper exited flag poisoned");
        !(res.timed_out() && !*done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_line_rejects_nonsense() {
        // sanity: protocol parse used inside send loop
        assert!(Envelope::parse_line("not json").is_err());
        assert!(Envelope::parse_line("").is_err());
    }

    /// `spawn` brings the wrapper up in-process: `probe` and `eplist`
    /// round-trip over the pipes without any child process, and `kill` (EOF)
    /// ends the loop thread.
    #[test]
    fn spawn_in_process_round_trips_and_kills() {
        let mut w = Wrapper::spawn(crate::commands::Runtime::Ort).expect("in-process wrapper");
        let caps = w
            .send(serde_json::json!({ "cmd": "probe" }))
            .expect("probe round-trip");
        assert!(caps.is_object(), "probe returns capabilities");
        let eps = w
            .send(serde_json::json!({ "cmd": "eplist" }))
            .expect("eplist round-trip");
        assert!(eps["eps"].is_array(), "eplist returns providers");
        w.kill();
        // After kill (stdin dropped), further sends fail instead of hanging,
        // and the loop thread exits — wait_exited confirms it drained.
        assert!(w.send(serde_json::json!({ "cmd": "eplist" })).is_err());
        assert!(w.wait_exited(Duration::from_secs(5)));
    }
}
