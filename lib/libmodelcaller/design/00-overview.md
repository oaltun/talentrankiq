# modelcaller design process

> **Status update (2026-09-23).** The original two-binary design below is
> history: the `modelcaller` binary was deleted and `onnxwrapperrs` is
> lib-only. modelcaller is now a **library** (`run_stdio_over` + `Registry`)
> that always runs the wrapper in-process on a dedicated thread over
> `std::io::pipe()` pairs. Every consumer (gui, bench, tests) hosts it via
> `libmodelcallerclient::ModelCallerClient::spawn_inprocess`. The protocol,
> registry, caching, and pipeline contracts below are unchanged.

## goal

design a standalone CLI binary ("modelcaller") that owns all model
loading and model calling (inference) in this workspace.

today the ik5 tauri app and bench4 both talk to a `lemonade-embeddable`
server over HTTP and each re-implements the embedding/reranking logic.
that is duplicated (not DRY), confusing (two lemonade installs on the
system), and ships a server to customer machines we don't want to run.

modelcaller replaces that. it is actually **two cooperating CLIs**
that together own all model loading/calling:

1. **the orchestrator (`modelcaller`)** — the part external callers
   (tauri app, bench4) actually spawn. it speaks the JSONL protocol in
   `02-interface.md` and owns all the logic: command dispatch, model
   registry + selection, hardware profiling, caching, and the
   embeddings/reranking pipelines (tokenization, pooling, normalize,
   truncation). it is **implementation-language-neutral**.
2. **the wrapper (`onnxwrap`)** — a thin CLI that is the **only** place
   ONNX Runtime is touched. it loads ONNX model files into inference
   sessions and runs raw tensor computations (`../onnxwrapperrs/design/01-wrapper.md`). the
   orchestrator spawns it as a child process and delegates every
   inference to it.

shared properties:

- both communicate over **stdin / stdout / stderr** using **JSONL**
  (`01-protocol.md`); both are child processes with no external server.
- model loading goes through **ONNX Runtime via an ONNX Runtime binding**
  (e.g. Rust `ort`, Zig `onnxruntime.zig`) — isolated entirely inside
  the wrapper, so the rest of the design is independent of any language
  or binding. no lemonade / no server on customer machines.
- general (a generic model loader/runner) but with specifics for
  well-known models (embedders, rerankers).
- centralizes the embedding/reranking logic that ik5 and bench4
  currently duplicate.

benchdriver2 is unaffected: it only calls bench4, never modelcaller.

> **a third, dev-only project `modelconverter`** handles GGUF->ONNX
> conversion + HuggingFace upload (`modelconverter/design/00-overview.md`).
> it is run by bench4 at benchmark time and is **never shipped to
> customers**. production (ik5 / infer1) consumes only ONNX from HF, so
> infer1 has no GGUF/conversion code.

## design steps

each step is a form in this directory. complete them in order. later
steps depend on earlier ones.

### step 1 — `01-protocol.md`
define the JSONL wire protocol: how one request is framed on stdin, how
responses are framed on stdout, and how diagnostics/errors flow on
stderr. this is the foundation — every caller depends on it.

### step 2 — `02-interface.md`
define the command/request set modelcaller accepts (load model, embed,
generate/rerank, unload, etc.) and each command's input + output schema.

### step 3 — `03-models.md`
define how models are represented: a general loader plus specifics for
well-known models (embedders like nomic, rerankers like bge). what
identifies a model, where files come from, how variants are named.

### step 4 — `04-embeddings.md`
define the centralized embedding logic (the part ik5 and bench4
duplicate today): batching, text preprocessing, normalization, pooling,
and the canonical output shape.

### step 5 — `05-inference.md`
define non-embedding inference (e.g. cross-encoder reranker score
generation, token classification) and how it differs from embeddings.

### step 6 — `06-cli.md`
define CLI invocation: binary name, subcommands/flags, stdin/stdout
mode selection, version/help, exit codes.

### step 7 — `07-process-management.md`
define the child-process lifecycle from the caller's perspective: how a
caller starts modelcaller, detects readiness, sends shutdown, and what
happens on crash or timeout.

### step 8 — `08-config.md`
define configuration: model search paths, onnx execution providers /
threads, default models, environment variables.

### step 9 — `09-caching.md`
define caching: model load/reuse in memory and on-disk cache for
embeddings, keyed by model + text + preprocessing.

### step 10 — `10-error-handling.md`
define error taxonomy, stderr error envelope, exit codes, and how a
caller recovers.

### step 11 — `11-profiling.md`
define host hardware profiling: what to scan (VRAM, GPU, RAM, cores,
disk), the capabilities report, how it drives **default model
selection** (pick the best model that fits and is fast enough) and
**parameter tuning** (batch size, threads, precision). this is what lets
modelcaller default users to a larger model when they can handle it, and
avoid a model that would be too slow or fail to load.

### step 12 — `12-implementation.md`
define the orchestrator implementation plan: implementation language +
JSON/CLI libs (no ONNX Runtime dependency), profiler orchestration,
model selection, pipelines, project structure, build, tests, and the
migration path from lemonade for ik5/bench4.

> the **ONNX Runtime wrapper** is a separate project, documented under
> `onnxwrapperrs/design/` (`00-overview.md`, `01-wrapper.md`,
> `03-implementation.md`). the orchestrator spawns it and delegates all
> tensor computation to it.

## rules

- **design-first**: when changing anything (new command, model, CLI
  flag, protocol change), update the relevant design md first, then
  implement. the design is the single source of truth.
- each step references earlier steps explicitly where it depends on them.
- each form has fields marked with `<!-- TODO -->` to fill in.
- each form has an "example" section where applicable to show concrete
  expected values.
- do not skip steps. if a step is not needed yet, mark it "deferred"
  with a reason.
- these md files are read directly (no preview). keep tables narrow
  (<= ~60 chars width); use lists for wider content.
- no comments in source code unless asked. (design notes go here.)
