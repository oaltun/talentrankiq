# onnx runtime wrapper cli

## depends on

- `02-protocol.md` (the wrapper reuses the same JSONL framing)
- `03-implementation.md` (how it is built against an ONNX Runtime binding)

## 1.1 purpose

the wrapper is the **only** component that touches ONNX Runtime. it is
a thin, language-neutral CLI that loads ONNX model files (with their
tokenizers) and runs **model-specific preprocessing + tensor
computation**, returning the semantic output: an embedding **vector**
(for embedders) or a relevance **score** (for rerankers). everything
else in the system (the **orchestrator**) is pure logic — protocol,
model registry, selection, caching, batching, download — and delegates
actual inference to this wrapper.

this isolates the ONNX-Runtime-specific code (the binding, execution
providers, tensor layout, tokenizers, hardware probing) so the rest of
the design is independent of any implementation language or binding.
you can write the orchestrator in Rust or Zig; you implement the wrapper
against whatever ONNX Runtime binding exists for that language (e.g.
Rust `ort`, Zig `onnxruntime.zig`).

> relaxation: the wrapper is **not** strictly tensors-in/tensors-out.
> to keep inter-process IO tiny (no giant float tensors crossing the
> pipe), the wrapper owns tokenization, pooling/aggregation, and
> L2-normalize, and returns vectors/scores. see `infer1/design/04-embeddings.md`.

## 1.2 relationship to the orchestrator

- the **orchestrator** (`infer1/`) is what external callers (ik5,
  bench4) spawn. it speaks the external protocol in
  `infer1/design/02-interface.md`.
- the orchestrator **spawns the wrapper as a child process** at startup
  and talks to it over stdin/stdout JSONL, using the same framing as
  `02-protocol.md`.
- relaxation (2026-09-23): the wrapper is also a **library**
  (`onnxwrapperrs::run_stdio_over` over arbitrary streams). modelcaller's
  `inprocess-wrapper` feature (enabled by shortlist-core) runs that loop on a
  dedicated thread over OS pipes inside the host process — the single-binary
  gui path (`gui/design/app1/thin-gui.md`); shutdown is EOF on the request pipe. the
  spawned-child mode stays the default for the modelcaller binary (bench),
  where ORT crash isolation matters.
- the wrapper is long-lived: loaded sessions persist across `embed` /
  `rerank` calls for the orchestrator's lifetime.
- the wrapper never sees the orchestrator's external protocol or
  high-level concepts (`pull` / model selection / caching); it only
  knows models, text, and execution providers.

## 1.3 command set (wrapper's own protocol)

| cmd      | purpose                                              |
| -------- | ---------------------------------------------------- |
| `load`   | load an ONNX model file + its tokenizer into a session |
| `embed`  | embed texts -> one vector per text                   |
| `rerank` | score query+doc pairs -> one score per doc           |
| `unload` | free a session                                       |
| `probe`  | report host hardware / usable execution providers    |
| `eplist` | list execution providers this build supports         |

all messages use the `02-protocol.md` envelope (`id` / `type` /
`payload`); `stderr` is diagnostic only.

### load
request:
```json
{"id":"w1","type":"request","payload":{"cmd":"load",
 "model_path":"/cache/nomic-embed-text-v1/model.onnx",
 "exec_provider":"auto","precision":"fp16","intra_threads":4,"inter_threads":1}}
```
- `exec_provider`: `"auto"` (the binding picks the best available EP,
  see `infer1/design/08-config.md` EP priority) or an explicit EP name.
- `precision`: `"fp32" | "fp16" | "int8"` — session compute precision.
- response: `{"session":<handle>,"kind":"embedder|reranker",
  "dim":768}` — `kind` tells the orchestrator how to call `embed`/`rerank`.

### embed
request:
```json
{"id":"w2","type":"request","payload":{"cmd":"embed","session":1,
 "texts":["rust developer","python developer"],
 "opts":{"preprocess":"lowercase","normalize":true,"max_context_window":8192}}}
```
- `texts`: raw input texts. the wrapper **tokenizes**, truncates each to
  `max_context_window` (token-level), runs the model, **mean-pools**
  over tokens, and **L2-normalizes** (unless `normalize:false`).
- response: `{"embeddings":[[0.01,0.02,...],[...]]}` — one f32 vector per
  text, in order, length == `dim`.

### rerank
request:
```json
{"id":"w3","type":"request","payload":{"cmd":"rerank","session":1,
 "query":"senior rust developer","docs":["cv A","cv B"],
 "opts":{"max_context_window":8192}}}
```
- the wrapper tokenizes each (query, doc) pair, runs the cross-encoder,
  and returns one relevance score per doc.
- response: `{"scores":[0.91,0.33]}` — one f32 per doc, in order.

### probe
returns host capabilities (GPU vendor / VRAM / fp16 / int8 support, CPU
info). this is the ONNX-Runtime-specific hardware scan that feeds
the orchestrator's `probe` / `recommend`
(`infer1/design/11-profiling.md`). the wrapper owns it because only the
binding knows which EPs are actually usable on the host.

### unload / eplist
self-explanatory.

## 1.4 what stays OUT of the wrapper

model registry / footprint / **selection**, **caching**, **batching
decisions**, download (`pull`), and the external `embed`/`rerank`/
`probe`/`recommend` protocol all live in the orchestrator
(`infer1/design`). the wrapper is a model server: **text in, vector /
score out**, owning only tokenization, tensor compute, pooling, and
normalize.

## 1.5 example

orchestrator -> wrapper stdin:
```jsonl
{"id":"w1","type":"request","payload":{"cmd":"load","model_path":"/cache/nomic.onnx","exec_provider":"auto","precision":"fp16"}}
{"id":"w2","type":"request","payload":{"cmd":"embed","session":1,"texts":["rust dev","python dev"],"opts":{"normalize":true,"max_context_window":8192}}}
```
wrapper -> orchestrator stdout:
```jsonl
{"id":"w1","type":"response","payload":{"session":1,"kind":"embedder","dim":768}}
{"id":"w2","type":"response","payload":{"embeddings":[[0.01,0.02,...],[...]]}}
```

## 1.6 generation (step 18 m3 — designed in 04-generation.md)

step 18 (`bench/bench1/design/18-structured-evidence.md` §18.4) adds token
generation for cv-evidence extraction:

- `generate` — greedy decode with kv-cache (the design's "no kv-cache
  in v1" note is corrected there: at qwen2's 151 936-token vocabulary
  a full re-run materializes a ~1.2 GB logits tensor per step, so the
  exports' past/present kv inputs are the simple path, not a perf
  nicety).
- `generate_constrained` — the same loop driven by a schema-derived
  json state machine; logits are masked to admitted tokens so invalid
  json is structurally impossible.

sessions of the new `extractor` kind serve these commands; prompt
rendering, validation, and caching stay in modelcaller. full spec:
`04-generation.md`.
