# step 2 — command interface

## depends on

- `01-protocol.md` (every command is a `payload` of a `request`/`response`
  envelope; `id` correlates request↔response; errors are `type:"error"`).
- references `03-models.md` (the `model` field), `04-embeddings.md`
  (`embed`), `05-inference.md` (`rerank`/`generate`), `09-caching.md`
  (lazy load), `07-process-management.md` (`ping`/`shutdown` readiness).

## 2.1 command set

every request `payload` carries a `cmd` string. the supported commands:

| cmd         | purpose                                       | needs `model`? |
| ----------- | --------------------------------------------- | -------------- |
| `load`      | explicitly preload a model into memory        | yes            |
| `unload`    | release a loaded model (free memory)          | yes            |
| `embed`     | produce embeddings for a list of texts        | yes            |
| `rerank`    | score query+doc pairs (cross-encoder)         | yes            |
| `models`    | query model metadata (dim, context window)    | no (or filter) |
| `probe`     | report host capabilities (hw scan)            | no             |
| `recommend` | best-fit model for a task (from profiling)    | no (`task`)    |
| `generate`  | text generation (deferred — see 2.7)          | yes            |
| `list`      | list loaded / discoverable models             | no             |
| `ping`      | readiness / liveness probe                    | no             |
| `shutdown`  | explicit graceful shutdown                    | no             |

design note: inference commands (`embed`/`rerank`/`generate`) carry
their own `model` id and **lazily load + cache** the model on first use
(see `09-caching.md`). `load` is an *optional* explicit preload so a
caller can fail fast / warm up before a benchmark run; it is not a
required handshake. `pull` is acquisition-only: it downloads model
files into the local cache so that `load`/`embed` can run fully offline
(see `08-config.md` for registry/source config). modelcaller has **no network code** — model
acquisition (downloading ONNX from HuggingFace) is done by the
separate `modeldownloader` utility (`../modeldownloader/design/`)
before modelcaller starts. modelcaller expects models already present
in the cache and runs fully offline.

this is the **orchestrator's** external protocol (the CLI that ik5 /
bench4 spawn). the orchestrator does NOT run ONNX directly — it
delegates every tensor computation to the **wrapper CLI** (`../onnxwrapperrs/design/01-wrapper.md`),
which is the only component that touches ONNX Runtime.

## 2.2 common request fields

fields that may appear on any request (all optional unless noted):

- `model`: `<string>` model id (see `03-models.md`). required by
  `load`/`unload`/`generate`. for `embed`/`rerank` it is **optional**:
  if omitted, modelcaller uses the profiling-selected best-fit model
  for that task (see `11-profiling.md` + `recommend`). if provided, it
  is validated strictly (unknown id -> `no_model`).
- `opts`: `<object>` per-request overrides. recognized keys:
  - `normalize`: `bool` — L2-normalize output vectors (embed).
  - `preprocess`: `"lowercase" | "none"` — text prep (embed).
  - `batch_size`: `int` — max texts per inference run. **default derived from
    host capabilities** (see `11-profiling.md`) when omitted.
  - `precision`: `"fp32" | "fp16" | "int8"` — compute precision.
    **default derived from GPU capability** (fp16 if supported, else
    int8, else fp32) when omitted.
  - `exec_provider`: `"cpu" | "cuda" | "tensorrt" | "coreml" | "directml" |
    "openvino" | "rocm" | "nnapi"` — **optional explicit override**. when
    omitted, modelcaller **auto-detects the best available accelerator**
    (GPU/TPU/NPU) on the user's machine and uses it (see `08-config.md`).
    provided so benchmarks (bench4) can pin a fixed EP for reproducibility.
  - `threads`: `int` — ONNX Runtime thread count override (load).

unknown `opts` keys are **rejected** with a `bad_req` error (see
`10-error-handling.md`), not silently ignored. reason: the only
callers are internal/co-versioned (ik5, bench4), so forward-compat via
ignoring is not needed; a misspelled option (e.g. `normalise` vs
`normalize`) would otherwise silently change behavior and cause hard-
to-trace correctness bugs. **do not panic** on an unknown key — that
would kill the whole process mid-stream; reply with a per-request error
instead so the process and other in-flight requests survive.

## 2.3 per-command input / output

### load
request:
```json
{"cmd":"load","model":"nomic-embed-text-v1","opts":{"exec_provider":"cpu"}}
```
response:
```json
{"status":"loaded","model":"nomic-embed-text-v1","kind":"embedder","dim":768,"ep":"coreml"}
```
errors: `load_err`, `no_model` (unresolvable id).

### unload
request: `{"cmd":"unload","model":"nomic-embed-text-v1"}`
response: `{"status":"unloaded","model":"nomic-embed-text-v1"}`
errors: `no_model` (not loaded).

### pull (removed)
model download has been extracted to the separate `modeldownloader`
utility (`../modeldownloader/design/`). callers run `modeldownloader`
before starting modelcaller. modelcaller has no `pull` command and no
network code. if a model is not in the cache, `load`/`embed` returns
`no_model`.


### probe
returns the cached host capabilities from the startup hardware scan
(see `11-profiling.md`). no network, no model needed.

request: `{"cmd":"probe"}`
response:
```json
{"capabilities":{
  "os":"linux","arch":"x86_64",
  "cpu":{"vendor":"intel","physical_cores":8,"logical_cores":16,"simd":["avx2","avx512"]},
  "ram_total":17179869184,"ram_free":8589934592,
  "gpus":[{"index":0,"vendor":"nvidia","name":"RTX 4070","vram_total":8589934592,
           "fp16":true,"int8":true,"compute_capability":"8.9"}],
  "disk_cache_free":107374182400,
  "selected_ep":"cuda"
}}
```
a probe subsystem that can't be read (e.g. no NVML) simply omits its
field; modelcaller degrades to conservative CPU defaults. never
crashes on a probe miss.

### recommend
returns the profiling-selected best-fit model id for a task, plus the
reason (so UIs can explain "why this model"). drives the auto-select
used when `embed`/`rerank` omit `model`.

request:
```json
{"cmd":"recommend","task":"embed"}
```
response:
```json
{"task":"embed","model":"nomic-embed-text-v1","ep":"cuda","reason":"fits_vram+quality"}
```
- `task`: `"embed" | "rerank"`.
- `reason`: short tag — `fits_vram+quality`, `low_vram`, `cpu_only`,
  `latency_budget`, etc.

### embed
request:
```json
{"cmd":"embed","model":"nomic-embed-text-v1",
 "texts":["rust developer","python developer"],
 "opts":{"normalize":true,"preprocess":"lowercase","batch_size":32}}
```
response:
```json
{"model":"nomic-embed-text-v1","dim":768,"ep":"coreml",
 "embeddings":[[0.01,-0.02,0.03],[0.04,0.05,-0.06]]}
```
- `model` is **optional**: if omitted, the profiling-selected model for
  `embed` (see `recommend` / `11-profiling.md`) is used. if provided, it
  is validated strictly.
- `embeddings`: array of vectors, **one per input text, in the same
  order**. each vector length == `dim`, f32 values.
- empty `texts` -> empty `embeddings` array.
- length mismatch / non-finite values -> `infer` error.
- **truncation**: inputs longer than the model's `max_context_window`
  (from `models`) are truncated before tokenization — centralized here
  so callers don't each re-implement it (bench4 currently hard-codes
  this per method). exact unit (chars vs tokens) is model-specific, see
  `04-embeddings.md`.
- **parameter defaults**: `batch_size` / `precision` / `exec_provider`
  default to profiling-derived values when not in `opts` (see
  `11-profiling.md`).
- lazy-loads the model if not already in memory (see 2.1).

### rerank
request:
```json
{"cmd":"rerank","model":"bge-reranker-v2-m3",
 "query":"senior rust developer","docs":["cv A text","cv B text"],
 "opts":{"batch_size":16}}
```
response:
```json
{"model":"bge-reranker-v2-m3","ep":"cuda","scores":[0.91,0.33]}
```
- `scores`: array of f32, **one per `docs` entry, in the same order**.
  higher = more relevant to `query`. (lemonade returns the same as
  `results[].{index,relevance_score}`; we return an order-aligned array
  for simpler caller handling.)
- **truncation**: each `doc` longer than the reranker's
  `max_context_window` is truncated before scoring (bench4 currently
  hard-codes `MAX_RERANK_CHARS = 12000`; modelcaller centralizes this).
  see `05-inference.md`.
- lazy-loads the model if needed.

### models
queries model metadata. mirrors lemonade's `GET /v1/models` (bench4's
`model_max_ctx` uses it to size context-window truncation). returns the
properties a caller needs to truncate inputs correctly and to know
output shape.

request (optional `model` filter; omit to list all known/cached):
```json
{"cmd":"models","model":"nomic-embed-text-v1"}
```
response:
```json
{"models":[
  {"model":"nomic-embed-text-v1","kind":"embedder","dim":768,"max_context_window":8192},
  {"model":"bge-reranker-v2-m3","kind":"reranker","dim":0,"max_context_window":8192}
]}
```
- `dim`: embedding dimension for embedders (`0` for rerankers, which
  output a scalar score, not a vector).
- `max_context_window`: token budget; used by `embed`/`rerank` to
  truncate inputs (see `04-embeddings.md` / `05-inference.md`).
- unknown model -> `no_model` error (or omitted from list).

### list
request: `{"cmd":"list"}`
response:
```json
{"loaded":[{"model":"nomic-embed-text-v1","kind":"embedder","dim":768}],
 "available":["nomic-embed-text-v1","bge-reranker-v2-m3"]}
```
- `loaded`: models currently in memory.
- `available`: models discoverable from search paths (see `08-config.md`).

### ping
request: `{"cmd":"ping"}`
response: `{"ok":true,"version":"0.1.0"}`
used by callers to detect startup readiness (see `07-process-management.md`).

### shutdown
request: `{"cmd":"shutdown"}`
response: `{"ok":true}` — then modelcaller exits 0 (same as stdin EOF).
gives callers an explicit alternative to closing stdin.

## 2.4 deferred commands

### generate (deferred)
reserved in the protocol but **not implemented yet**: no concrete
generative model is selected. placeholder shape:
```json
{"cmd":"generate","model":"<id>","prompt":"...","opts":{"max_tokens":256}}
-> {"text":"..."}
```
reason: start.md scopes modelcaller to embedders + rerankers
(centralizing ik5/bench4 duplication). generative/rubric scoring may
come later; the `cmd` string is reserved so callers can plan for it
without a protocol change.

## 2.5 example (full round-trip)

setup (one-time, before starting modelcaller):
```bash
modeldownloader --model nomic-embed-text-v1 --source hf:nomic-ai/nomic-embed-text-v1-onnx
# stdout: /Users/apple1/.cache/modelcaller/nomic-embed-text-v1/main
```

runtime (offline):
```jsonl
{"id":"1","type":"request","payload":{"cmd":"ping"}}
{"id":"2","type":"request","payload":{"cmd":"embed","model":"nomic-embed-text-v1","texts":["rust dev","python dev"],"opts":{"normalize":true}}}
{"id":"3","type":"request","payload":{"cmd":"rerank","model":"bge-reranker-v2-m3","query":"rust dev","docs":["cv A","cv B"]}}
```

modelcaller -> stdout:
```jsonl
{"id":"1","type":"response","payload":{"ok":true,"version":"0.1.0"}}
{"id":"2","type":"response","payload":{"model":"nomic-embed-text-v1","dim":768,"embeddings":[[0.01,-0.02,0.03],[0.04,0.05,-0.06]]}}
{"id":"3","type":"response","payload":{"model":"bge-reranker-v2-m3","ep":"cuda","scores":[0.91,0.33]}}
```
