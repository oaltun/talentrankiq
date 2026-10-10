# step 7 — process management (caller perspective)

## depends on

- `01-protocol.md` (readiness/ping, shutdown via stdin EOF)
- `06-cli.md` (how to launch)

## 7.1 startup

how a caller starts modelcaller and knows it's ready:

- spawn: caller creates a child process with piped stdin, piped stdout,
  and inherited (or piped) stderr. stdin is for sending requests,
  stdout for reading responses, stderr for diagnostics. the child is
  launched with the `--stdio` flag (and any other flags the caller needs).
  the caller must **begin reading stdout before writing any request** to
  avoid pipe-buffer deadlock (see `01-protocol.md` 1.4).
- readiness signal: modelcaller emits a `ping` response on stdout when
  it has finished its startup sequence (hardware probe, wrapper spawn,
  wrapper probe). callers detect readiness by sending a `ping` request
  immediately after spawn and waiting for the matching `ping` response.
  the `ping` response includes `"ok":true,"version":"..."`.
  callers MUST NOT parse stderr for readiness — stderr output order is
  not guaranteed.
- timeout: callers set a startup deadline. if the `ping` response is
  not received within **30 seconds** (configurable by the caller), the
  caller treats modelcaller as failed-to-start and kills it
  (see 7.2). this covers: model binary missing, wrapper spawn failure
  (exit code 3), hardware probe hang, or a misconfigured `--wrapper`
  path. 30s is generous for cold-start model load; a pure readiness
  ping (no model preloaded) should respond in <1s.

## 7.2 shutdown

- graceful: **close stdin (send EOF)**. per `01-protocol.md` 1.3, EOF
  on stdin triggers graceful shutdown: modelcaller finishes any
  in-flight request, flushes stdout, releases all loaded models, then
  exits 0. the caller can also send an explicit `{"cmd":"shutdown"}`
  request, which produces a `{"ok":true}` response followed by the
  same exit-0 cleanup. both paths are equivalent; EOF is preferred
  because it requires no protocol knowledge and works even if the
  orchestrator is wedged mid-parse.
- forced: if modelcaller does not exit within **10 seconds** after stdin
  EOF (or shutdown response), the caller sends **SIGTERM**. if it still
  does not exit within **5 more seconds**, the caller sends **SIGKILL**.
  SIGKILL is last-resort — the OS reclaims all resources (memory, child
  wrapper process, file locks) so no permanent state is leaked.
- cleanup: on graceful exit, modelcaller (1) sends stdin EOF to its
  wrapper child, waits for it to exit, (2) releases all in-memory model
  sessions, (3) flushes and closes stdout. on SIGTERM/SIGKILL, the OS
  handles cleanup. modelcaller writes no temp files to disk during
  normal operation (model cache files are read-only after download), so
  no temp-file cleanup is needed.

## 7.3 crash / hang handling

- caller detects dead process: the caller reads from stdout. if the
  read returns **EOF (pipe closed)** unexpectedly (before the caller
  closed stdin), the child has exited or crashed. the caller then
  **reaps the child** (waitpid / equivalent) and inspects the exit code
  (see `06-cli.md` 6.4). exit code 0 = clean shutdown the caller didn't
  initiate (treat as crash). exit code 1 = panic (unrecoverable).
  exit codes 2–4 = startup/configuration failures. the caller's
  response is context-dependent: a desktop app (ik5) shows an error to
  the user; a benchmark (bench4) aborts the run.
- caller detects no response: each request sent has a **per-request
  timeout**. the caller starts a timer when it writes a request to
  stdin; if the matching response (by `id`) is not received within the
  timeout, the caller considers the request failed with a local
  `timeout` error. the default timeout is **60 seconds** (configurable
  by the caller). the caller does NOT kill modelcaller over a single
  timeout — the process may still be processing other pipelined
  requests or recovering. if **N consecutive requests** time out
  (caller-configurable, default 3), the caller treats modelcaller as
  hung and initiates forced shutdown (7.2).
- restart policy: **modelcaller does not self-restart or self-heal.**
  if the wrapper child dies, modelcaller exits with code 1 (it cannot
  function without the wrapper). restart is entirely the caller's
  responsibility. the recommended pattern: spawn a fresh modelcaller
  instance. callers that need persistence (ik5) respawn on crash;
  callers that run once (bench4) treat a crash as a fatal error.
  there is no built-in watchdog or supervisor inside modelcaller.

## 7.4 no ports

modelcaller uses stdio, no tcp port — avoids the lemonade port conflict
problem from start.md problem #5. confirm: the entire external interface
is stdin/stdout JSONL (see `06-cli.md` 6.2). there is no `--serve` flag,
no TCP listener, no unix socket, no D-Bus, no shared memory. the only
inter-process communication is the pipe pair between caller↔modelcaller
and the pipe pair between modelcaller↔wrapper. this eliminates: port
collisions between ik5 and bench4, "is it already running?" ambiguity,
firewall/permissions issues, and the need for lifecycle management
beyond child process spawn/kill. the wrapper is similarly stdio-only
(see `../onnxwrapperrs/design/01-wrapper.md`).

## 7.5 example

pseudo-code for a caller spawning modelcaller, using it, and shutting
down:

```
process = spawn("modelcaller --stdio --model-dir /cache")
reader_thread = async read_lines(process.stdout)

process.stdin.write(jsonl({id:"0", type:"request",
                           payload:{cmd:"ping"}}))
resp = await reader_thread.next(timeout=30s)
assert resp.id == "0" && resp.payload.ok == true

process.stdin.write(jsonl({id:"1", type:"request",
  payload:{cmd:"embed", model:"nomic-embed-text-v1",
            texts:["rust dev","python dev"]}}))
resp = await reader_thread.next(timeout=60s)
assert resp.id == "1"
vectors = resp.payload.embeddings

process.stdin.close()          # EOF -> graceful shutdown
await process.wait(timeout=10s) # exit 0
```
