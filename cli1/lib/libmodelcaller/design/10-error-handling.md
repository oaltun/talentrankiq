# step 10 — error handling

## depends on

- `01-protocol.md` (where errors appear: stdout error type vs stderr)
- `02-interface.md` (per-command failure modes)

## 10.1 error taxonomy

| code      | meaning                  | recoverable by caller? | behavior                                  |
| --------- | ------------------------ | ---------------------- | ----------------------------------------- |
| `bad_req` | malformed/unknown cmd    | no (fix caller)        | malformed JSON, unknown `cmd`, unknown `opts` key. reply with `id` echoed (or `id:null` if parse failed), continue serving. |
| `no_model`| model id not found       | maybe (load first)     | id not in registry or not in cache and no source configured. reply, continue. |
| `load_err`| model load failed        | maybe (retry)          | wrapper `load` failed: OOM, bad ONNX file, EP unavailable. reply, continue — session not created. |
| `infer`   | inference/runtime error  | maybe (retry)          | wrapper `embed`/`rerank` failed mid-run: ONNX error, tensor shape mismatch, non-finite output. reply, continue. |
| `timeout` | request took too long    | yes                    | internal deadline exceeded (see `07-process-management.md` 7.3). reply, continue. |
| `internal`| unexpected panic         | no (restart)           | unreachable in normal operation. if it happens: write one error line with `id:null` on stdout, then exit 1. |

## 10.2 error envelope

```json
{"id":"<echo>","type":"error","payload":{"code":"<code>","msg":"<text>"}}
```

- errors go on **stdout** as `type:"error"` envelopes, correlated by
  `id` to the originating request (or `id:null` if the request could
  not be parsed). this is the only place callers should look for
  failures — stderr is diagnostic only (see `01-protocol.md` 1.5).
- panic -> exit code: **exit 1**. modelcaller writes one final error
  line to stdout (`"id":null,"type":"error","payload":{"code":"internal","msg":"..."}}`),
  flushes stdout, then exits 1. callers detect this via the unexpected
  EOF on stdout (see `07-process-management.md` 7.3). the panic
  backtrace goes to stderr, not stdout.

## 10.3 caller recovery

- per-request error: the caller reads the error envelope, matches it by
  `id`, and decides based on `code`: `bad_req` -> fix the caller code;
  `no_model` -> run `modeldownloader` then retry; `load_err` -> try a different EP
  or free memory and retry; `infer` -> retry once (transient ONNX
  glitch), then give up; `timeout` -> retry with a larger batch or
  smaller model. in all cases the caller does NOT kill modelcaller —
  the process is still serving.
- process death: if stdout returns EOF unexpectedly (see
  `07-process-management.md` 7.3), the caller inspects the exit code.
  exit 0 = clean shutdown (caller didn't initiate). exit 1 = panic.
  exit 2–4 = startup failure. the caller's response is
  context-dependent: ik5 shows an error to the user and offers retry;
  bench4 aborts the benchmark run. the caller spawns a fresh
  modelcaller to recover.

## 10.4 example

```jsonl
{"id":"2","type":"error","payload":{"code":"no_model","msg":"nomic-embed-text-v1 not loaded"}}
```

malformed request (unparseable JSON):
```jsonl
{"id":null,"type":"error","payload":{"code":"bad_req","msg":"invalid JSON on line 5: expected '{' at column 1"}}
```
