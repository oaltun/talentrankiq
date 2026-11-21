# step 13 — implementation plan

## depends on

- all prior steps (protocol, interface, models, embeddings, inference,
  cli, process mgmt, config, caching, errors, profiling, wrapper).

## 12.1 language & deps

modelcaller is split into two binaries (see `00-overview.md`): the
**orchestrator** (language-neutral logic) and the **wrapper** (the only
place ONNX Runtime is used). pick one implementation language per
binary; they need not be the same. examples below use Rust, but Zig
(with `onnxruntime.zig`) or any language with an ONNX Runtime binding
works.

- **wrapper** — an ONNX Runtime binding (e.g. Rust `ort`, Zig
  `onnxruntime.zig`) — **enable ALL execution-provider features** so the
  wrapper can use whatever accelerator the host has (GPU/TPU/NPU):
  `cuda`, `tensorrt`, `coreml`, `directml`, `openvino`, `rocm`,
  `nnapi` (Android TPU/NPU), `xnnpack`, `armnn`, `cann` (Huawei
  Ascend), `acl` (Arm Compute Library).
  - **build caveat**: each EP pulls its native SDK at *compile time*
    (e.g. `cuda` needs the CUDA toolkit, `coreml` needs macOS). a build
    with all features will **fail on a machine missing an SDK**. plan:
    build the release binary on a host that has every SDK (or use the
    binding's prebuilt EP artifacts), or gate EPs behind build-system
    features with a "maximal" feature set used only for the shipped
    binary. dev builds can use a reduced `cpu`-only feature set for fast
    iteration.
- **orchestrator** — JSON/CLI libs only (JSONL framing, arg parsing).
  no ONNX Runtime dependency.
- **profiling** (see `11-profiling.md`, implemented in the orchestrator
  by calling the wrapper's `probe`):
  - a system-info library (e.g. Rust `sysinfo`) — CPU vendor/cores, RAM
    total/free, OS/arch, disk free.
  - an NVML binding (e.g. Rust `nvml-wrapper`, or parse `nvidia-smi` on
    Linux) — NVIDIA GPU name, VRAM, compute capability, fp16/int8 support.
  - platform-native GPU queries: Metal (on macOS) for Apple Silicon,
    DXGI on Windows, DRM/`sysfs` on Linux for non-NVIDIA GPUs. prefer
    probing via the wrapper's `probe` (the binding's EP capability query)
    as a fallback so a missing vendor SDK degrades gracefully.
  - all probes must be **best-effort**: a failing probe omits the field,
    never crashes startup.

## 12.2 project structure

this is the **orchestrator** project (`infer1/`). the wrapper is a
separate project — see `../onnxwrapperrs/design/03-implementation.md`.
names illustrative; language-agnostic:

```
infer1/                           # orchestrator repo root
├── <manifest>                    # e.g. Cargo.toml / package.json
└── src/
    ├── main.<ext>                # CLI parse, spawn stdio loop + wrapper child
    ├── protocol.<ext>            # JSONL framing (01-protocol.md)
    ├── commands.<ext>            # cmd dispatch (02-interface.md)
    ├── models/                   # general loader + well-known adapters (03)
    │   ├── mod.<ext>
    │   ├── embedder.<ext>
    │   └── reranker.<ext>
    ├── embeddings.<ext>          # centralized embed pipeline (04)
    ├── inference.<ext>           # rerank/generate (05)
    ├── profiler.<ext>            # hw scan orchestration (11), calls wrapper probe
    ├── selection.<ext>           # model + param selection (11)
    ├── config.<ext>              # 08
    ├── cache.<ext>               # 09
    └── error.<ext>               # 10
```

## 12.3 build & test

- build the release orchestrator binary — `cargo build --release` (Rust)
  or equivalent. the wrapper binary is built separately in its own
  project directory (`../onnxwrapperrs/`). the orchestrator's build
  produces a single binary `modelcaller`. at ship time, the wrapper
  binary is placed next to it (or the path is configured via `--wrapper`).
- unit tests: protocol round-trip (serialize request -> parse, verify
  envelope fields), config merge (CLI > env > file > defaults), cache
  key computation (deterministic SHA-256, same input = same key),
  model registry lookup (known ids resolve, unknown -> no_model),
  selection algorithm (footprint filtering + quality ranking). these
  need no wrapper child — pure logic.
- integration: spawn modelcaller with `--stdio`, send a `ping`, assert
  `ok:true` response. send an `embed` with a known text and model,
  assert the response has the correct `dim` and finite f32 values.
  verify the wrapper child was spawned (check stderr for wrapper
  startup line). send stdin EOF, assert exit 0. these tests require
  the wrapper binary and a model file on disk; gate them behind a
  feature flag or `--ignored` so CI can run unit tests without model
  files.

## 12.4 migration from lemonade

how ik5 and bench4 switch over (start.md problems #1,#2,#4):

- ik5: replace `runtime.rs` lemonade sidecar with modelcaller child —
  change `runtime.rs` to spawn `modelcaller --stdio` instead of
  lemonade, send `embed`/`rerank` requests over stdin, read responses
  from stdout on a dedicated thread. the lemonade health-check,
  port-discovery, and retry logic are all deleted. the OpenAI-
  compatible `/v1/embeddings` HTTP shim (if any) is removed — ik5's
  Rust code calls modelcaller directly via the pipe. the model
  download flow changes from lemonade's internal download to a one-
  time `modeldownloader` at app first-run.
- bench4: replace `lemonade/` HTTP client with modelcaller child —
  bench4's embedding method currently POSTs to `http://localhost:23187/v1/embeddings`.
  replace with spawning `modelcaller --stdio` (once per benchmark
  run, not per-document) and sending `embed` requests over the pipe.
  the `--exec-provider` flag replaces the implicit lemonade CUDA/CPU
  selection. benchdriver2's lemonade startup logic is replaced with
  ensuring modelcaller is on PATH (no port polling needed).
- benchdriver2: unchanged (only calls bench4, never modelcaller
  directly). benchdriver2's lemonade startup/polling code is removed
  since bench4 no longer needs lemonade running.

## 12.5 example

first runnable milestone: load + embed one text via stdio.

```bash
# terminal 1: modelcaller (orchestrator + wrapper)
modelcaller --stdio --model-dir ~/.cache/modelcaller

# terminal 2: pipe a request
printf '{"id":"1","type":"request","payload":{"cmd":"embed","model":"nomic-embed-text-v1","texts":["hello world"]}}\n' \
  | modelcaller --stdio --model-dir ~/.cache/modelcaller
# expected stdout:
# {"id":"1","type":"response","payload":{"model":"nomic-embed-text-v1","dim":768,"ep":"cuda","embeddings":[[0.012,...768 values...]]}}
```
