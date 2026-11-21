# step 6 — CLI

## depends on

- `02-interface.md` (commands)
- `01-protocol.md` (stdio JSONL framing)
- `08-config.md` (config sources / env vars)
- `07-process-management.md` (how callers spawn it)
- `../onnxwrapperrs/design/01-wrapper.md` (the wrapper it spawns)

## 6.1 binary

- name: `modelcaller` — the **orchestrator** (project dir `infer1/`).
- invoked by: the tauri app (ik5) and bench4, both as a **child
  process**. it is a CLI, no GUI.
- at startup it **spawns `onnxwrapperrs`** (the wrapper, project dir
  `onnxwrapperrs/`) as its own child and delegates all ONNX execution to it.
  callers never launch the wrapper directly (see `02-interface.md`).

## 6.2 modes

| mode      | flag       | behavior                                                     |
| --------- | ---------- | ------------------------------------------------------------ |
| stdio     | `--stdio`  | read JSONL requests on stdin, write JSONL on stdout, logs on stderr. **the only mode.** |

- **no `--serve` / TCP server.** a TCP server would reintroduce the
  ports / lifecycle / "is it running?" problems that start.md set out
  to eliminate (problems #2, #5). communication is stdin/stdout only.
- EOF on stdin = graceful shutdown (finish in-flight, flush, exit 0).
  the `shutdown` command does the same explicitly.

## 6.3 flags

| flag                   | meaning                                                      | default              |
| ---------------------- | ------------------------------------------------------------ | -------------------- |
| `--stdio`              | enable the stdin/stdout JSONL loop (required to run)         | off                  |
| `--model-dir <path>`   | model cache / search root (see `08-config.md`)               | `~/.cache/modelcaller` |
| `--exec-provider <ep>` | global default EP for the wrapper (`auto`/`cuda`/`tensorrt`/`coreml`/`directml`/`openvino`/`rocm`/`nnapi`); per-request `opts` override | `auto` |
| `--threads <n>`        | default ONNX thread count (overridable per request)          | ORT built-in         |
| `--log-level <lvl>`    | stderr verbosity (`trace`/`debug`/`info`/`warn`/`error`)     | `info`               |
| `--wrapper <path>`     | path to the `onnxwrapperrs` binary to spawn                     | resolved next to self |
| `--version`, `--help`  | standard                                                     | —                    |

env vars mirror the flags (higher precedence than built-in defaults,
lower than explicit flags): `MODELCALLER_MODEL_DIR`,
`MODELCALLER_EXEC_PROVIDER`, `MODELCALLER_THREADS`,
`MODELCALLER_LOG_LEVEL`, `MODELCALLER_WRAPPER`.

## 6.4 exit codes

| code | meaning                                                              |
| ---- | -------------------------------------------------------------------- |
| 0    | clean shutdown (stdin EOF or `shutdown` command)                     |
| 2    | bad CLI arguments / unknown flag                                     |
| 3    | failed to spawn or communicate with the `onnxwrapperrs` wrapper at startup (fatal — orchestrator cannot run without it) |
| 4    | fatal startup error (stdio setup, config load)                       |

**request-level failures do NOT exit.** a bad `embed`/`rerank`/etc. is
reported as an `error` envelope on stdout, id-correlated, and the
process keeps serving (per `01-protocol.md` and `10-error-handling.md`).
only startup / configuration failures exit nonzero.

## 6.5 example

```bash
modelcaller --stdio --model-dir ~/.cache/modelcaller --exec-provider auto
```
- the orchestrator spawns `onnxwrapperrs` automatically; the caller then
  writes JSONL requests to its stdin and reads JSONL responses from its
  stdout (see `07-process-management.md` for spawn/read details).
- sending EOF (closing stdin) shuts it down cleanly (exit 0).
