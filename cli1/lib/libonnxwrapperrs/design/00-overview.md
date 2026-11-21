# onnxwrapperrs — ONNX Runtime wrapper design

> **Status update (2026-09-23).** `onnxwrapperrs` is lib-only: the binary was
> deleted and `run_stdio_over` always runs on a dedicated thread inside the
> host process (gui, bench, tests), driven by modelcaller's in-process
> wrapper. The session/protocol/generation contracts below are unchanged.

## purpose

`onnxwrapperrs` is the **wrapper CLI**: the *only* component that touches
ONNX Runtime. it is spawned as a child process by the orchestrator
(`infer1/`) and exposes a minimal JSONL command set — `load` / `run` /
`unload` / `probe` / `eplist` — that performs raw tensor computation. it
is language/binding-neutral in design; the reference implementation uses
an ONNX Runtime binding (e.g. Rust `ort`, Zig `onnxruntime.zig`).

## design steps

- `00-overview.md` — this file.
- `01-wrapper.md` — the wrapper's JSONL command set and tensor contract.
- `02-protocol.md` — the shared JSONL framing (identical to the
  orchestrator's; copied here so this design is self-contained).
- `03-implementation.md` — building against an ONNX Runtime binding, EP
  features, and the session / run / probe structure.

## relationship to the orchestrator

the orchestrator (`infer1/design`) owns all high-level logic (protocol,
model registry + selection, profiling, caching, embeddings/reranking
pipelines) and **spawns `onnxwrapperrs`, delegating every inference to it**.
see `infer1/design/02-interface.md` for the orchestrator's external
protocol and `infer1/design/11-profiling.md` for how the orchestrator
drives model selection from the wrapper's `probe` output.
