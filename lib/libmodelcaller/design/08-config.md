# step 8 — configuration

## depends on

- `03-models.md` (model resolution)
- `06-cli.md` (flags as overrides)

## 8.1 config sources (priority order)

1. **CLI flags** — highest precedence. each flag maps to one config key
   (see `06-cli.md` 6.3). a flag explicitly provided on the command line
   overrides all other sources.
2. **Environment variables** — `MODELCALLER_MODEL_DIR`,
   `MODELCALLER_EXEC_PROVIDER`, `MODELCALLER_THREADS`,
   `MODELCALLER_LOG_LEVEL`, `MODELCALLER_WRAPPER`. lower precedence than
   flags, higher than config file / built-ins. naming pattern:
   `MODELCALLER_` prefix + flag name uppercased.
3. **Config file** — optional TOML file at
   `$MODELCALLER_CONFIG` or `~/.config/modelcaller/config.toml`. contains
   the same keys as env vars without the prefix. used for persistent
   settings a caller doesn't want to pass every launch (e.g. a default
   model-dir for all invocations). if the file is missing, it is silently
   skipped — no error.
4. **Built-in defaults** — lowest precedence. `model_dir`:
   `~/.cache/modelcaller`. `exec_provider`: `auto`. `threads`: unset
   (ORT built-in). `log_level`: `info`. `wrapper`: resolved next to the
   modelcaller binary itself (same directory).

a key not present in any source uses its built-in default. there is no
required config — modelcaller works with zero config files and zero
env vars.

## 8.2 model resolution

production models are **ONNX on HuggingFace**; the separate `modeldownloader`
utility fetches them (see `../modeldownloader/design/`). GGUF->ONNX conversion and publishing to
HF is the job of the separate **modelconverter** project (dev/bench-time
only, run by bench4), so this config never references GGUF.

where model files are found:

- search paths: `~/.cache/modelcaller`, bundled dir (for dev `file://` ONNX)
- download/cache: from HuggingFace via `modeldownloader` (`hf:<repo>`), cached locally
- format: `.onnx` / `.onnx.gz` + tokenizer + `config.json` (see `03-models.md` §3.3)

## 8.3 onnx runtime options

- **execution provider — compile ALL, auto-select at runtime.**
  the wrapper (`../onnxwrapperrs/design/01-wrapper.md`) is built with **every ONNX Runtime EP
  feature enabled** (see `12-implementation.md`), so it can use
  **whatever accelerator the host exposes — GPU, TPU, or NPU**. models
  are CPU-heavy, so at load time the wrapper **probes the host and picks
  the strongest accelerator available**, registering EPs in this priority
  order and falling back:

  1. NVIDIA discrete GPU -> `tensorrt` (preferred) / `cuda`
  2. Apple Silicon -> `coreml`
  3. Windows (any GPU: NVIDIA/AMD/Intel) -> `directml`
  4. Intel CPU/GPU -> `openvino`
  5. AMD GPU (Linux) -> `rocm`
  6. mobile/embedded NPU+TPU (Android) -> `nnapi`; Arm `armnn`/`acl`
  7. fallback -> `cpu` (`xnnpack` for fast neon CPU path)

  an EP may be compiled in but unavailable at runtime (e.g. `cuda` with
  no NVIDIA driver, `nnapi` on a non-Android host) — modelcaller must
  **degrade gracefully** to the next EP rather than fail the load. the
  chosen EP is reported in the `load`/`embed`/`rerank` response (`ep`
  field) so callers can see what is actually running. (Google Coral
  Edge TPU has no mainline ONNX Runtime EP; treat as out-of-scope /
  optional custom EP.)

- **explicit override**: `exec_provider` (request `opts` / env
  `MODELCALLER_EXEC_PROVIDER`) forces a specific EP. used by bench4 to
  pin a fixed EP so benchmark numbers are reproducible across runs.
  forcing an unavailable EP -> `load_err`.

- intra/inter thread counts: default = ONNX Runtime built-in (scales to
  cores). overridable via `threads` opt for resource governance (don't
  starve the UI in ik5; pin for reproducible bench4 runs).
- graph optimization level: ONNX Runtime applies this automatically
  (fusion, constant folding) — **no knob exposed**; leave at ONNX
  Runtime default.

## 8.4 defaults

- default embedder: `nomic-embed-text-v1` — selected by `recommend`
  when the caller omits `model` on `embed`. chosen by the profiler as the
  highest-quality embedder that fits the host's resources (see
  `11-profiling.md`). for hosts that cannot fit it, the tiny embedder
  is the fallback.
- default reranker: `bge-reranker-v2-m3` — selected by `recommend`
  when the caller omits `model` on `rerank`. same selection logic.

these are not hardcoded defaults in config; they are the output of the
profiling/selection step (`11-profiling.md`). the built-in model
registry (`03-models.md` 3.5) defines the candidate set and their
footprints; the profiler picks from that set at startup.

## 8.5 example

```bash
MODELCALLER_MODEL_DIR=~/.cache/modelcaller MODELCALLER_EXEC_PROVIDER=cpu \
  modelcaller --stdio
```
resolved config (what modelcaller sees after merging all sources):

| key             | value                          | source       |
| --------------- | ------------------------------ | ------------ |
| model_dir       | `~/.cache/modelcaller`         | env var      |
| exec_provider   | `cpu`                          | env var      |
| threads         | *(ORT default)*                | built-in     |
| log_level       | `info`                         | built-in     |
| wrapper         | `<same-dir>/onnxwrapperrs`         | built-in     |