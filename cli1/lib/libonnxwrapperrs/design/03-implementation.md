# onnxwrapperrs implementation

## depends on

- `01-wrapper.md` (command set + tensor contract)
- `02-protocol.md` (JSONL framing)

## 3.1 language & deps

- an **ONNX Runtime binding** (e.g. Rust `ort`, Zig `onnxruntime.zig`).
  this is the only place such a dependency exists; the orchestrator has
  none (see `infer1/design/12-implementation.md`).
- **enable ALL execution-provider features** so the wrapper can use
  whatever accelerator the host has (GPU/TPU/NPU): `cuda`, `tensorrt`,
  `coreml`, `directml`, `openvino`, `rocm`, `nnapi` (Android TPU/NPU),
  `xnnpack`, `armnn`, `cann` (Huawei Ascend), `acl` (Arm Compute Lib).
  - **build caveat**: each EP pulls its native SDK at *compile time*
    (e.g. `cuda` needs the CUDA toolkit, `coreml` needs macOS). a build
    with all features **fails on a machine missing an SDK**. plan: build
    the release binary on a host with every SDK (or use the binding's
    prebuilt EP artifacts), or gate EPs behind build-system features
    with a "maximal" set for the shipped binary; dev builds may use a
    reduced `cpu`-only set.

## 3.2 project structure

```
onnxwrapperrs/
├── <manifest>            # e.g. Cargo.toml / build.zig
└── src/
    ├── main.<ext>        # JSONL loop: load / embed / rerank / unload / probe / eplist
    ├── session.<ext>     # ONNX model load + embed/rerank via the binding
    └── probe.<ext>       # host / EP capability scan
```

## 3.3 build & test

- build the release wrapper binary.
- unit tests: tensor round-trip (feed known inputs, assert outputs
  against a reference), EP-selection / fallback logic.
- integration: spawn `--stdio`, send `load` + `run`, assert outputs;
  send `probe`, assert capabilities.

the orchestrator-side implementation (JSON/CLI libs, profiler
orchestration, model selection, pipelines, caching) lives in
`infer1/design/12-implementation.md`.
