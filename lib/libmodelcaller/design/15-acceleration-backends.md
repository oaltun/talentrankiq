# 15 — acceleration backends (per-arch provider ladder + CPU data-parallel fan-out)

## depends on

- `01-protocol.md` (the JSONL command contract — unchanged by this design)
- `08-config.md` (`ikapp.yaml` `engine:` section)
- `../onnxwrapperrs/design/01-wrapper.md` (the in-process wrapper: one
  dedicated thread, pipe pair, ORT sessions)
- `../../../lib/libsyscap/design/design.md` (the live budget: capacity,
  thread caps, verdicts)
- `../../../bench/bench1/design/15-pairwise-judge.md` §15.7 (the pairwise
  tournament — the second hot path this design accelerates)

## 15.1 purpose

make the engine use the silicon the machine actually has, per arch, with
CPU as the always-present fallback — without breaking the two rules the
whole stack rests on: **one binary, zero spawned children**, and **bench,
gui and tests drive the identical in-process stack**.

The motivation is measured, not assumed (this box, M5 Pro 6 P + 12 E):

- `exec_provider` is hardcoded `"auto"` (`lib/shortlist-core/src/runtime.rs`,
  `lib/modelcaller/src/cli.rs`), and `"auto"` *skips* provider registration
  entirely (`lib/onnxwrapperrs/src/session.rs`) → **CPU EP only**.
- the `coreml` feature is declared but off (`lib/onnxwrapperrs/Cargo.toml`
  `default = []`), so CoreML is not even linked.
- during the heaviest storm we could produce (4 concurrent full-graph LTO
  rebuilds + the 39-test engine suite, CPU 87 % user), `ioreg` reported
  AGX `Device Utilization % = 8..15` — the desktop compositor, nothing else.
  **The GPU and the 16-core ANE are idle in every embed, rerank and judge
  pass.**

## 15.2 what is already in place (and what is therefore *not* the win)

- **Quantization is banked.** The zoo is community-quantized already
  (`q4f16`, `quint8`, `int8`, `q4` slugs; the scout is a distilled MRL
  model). A pre-converted 8-bit MLX checkpoint of the same 270 M model is
  the *same quantization decision in another container* — MLX's marginal
  value is Metal execution, not the bit width.
- **The EP plumbing exists.** `ort` 2.0.0-rc.13 ships `coreml`,
  `directml`, `xnnpack`, `webgpu`, `cuda`, `openvino`, `rocm`, and
  `lib/onnxwrapperrs/Cargo.toml` already maps a feature per EP. Enabling a
  provider is a Cargo feature plus a config line.
- **The seam is clean.** `modelcaller::Wrapper` spawns
  `onnxwrapperrs::run_stdio_over` over a pipe pair and owns the JSONL
  protocol. A backend is anything that offers the same `run_stdio_over`
  shape; the contract does not move.

## 15.3 the axis that decides everything: split vs batch

CPU and accelerators want opposite things, so the two strategies are
**substitutes, not complements**:

| | CPU (today) | GPU / ANE (CoreML, MLX) |
|---|---|---|
| scales by | **N independent jobs on N cores** | **few, large kernels** |
| cost of splitting | one session (and its weights) per worker | launch overhead + contention on a single device |
| wants | small batch, many workers | big batch (32..256), one pipeline |
| bad at | one forward pass, many threads | 16 concurrent tiny embeddings |

The per-core-split instinct is therefore a *CPU-native* strategy: moving
to an accelerator pushes the design toward batching and away from
splitting. Both can coexist (N sessions × batch), but the GPU half of a
unified-memory box is one device — concurrent tiny kernels just queue.

**We already have evidence the CPU path leaves cores idle.** Thread-scaling
A/B on the same suite (all green, this box): 2 threads → 308 s, 5 → 219 s,
16 → 167 s. Scaling 2→5 was ~2×, 5→16 only ~1.3×: sublinear intra-op
scaling is the signature of a workload that does not saturate cores inside
one forward pass — the gap fan-out fills.

## 15.4 the backend ladder

```
modelcaller (protocol owner — JSONL unchanged, DTOs unchanged)
└── backend: run_stdio_over(in, out) + capability descriptor
    ├── ort / cpu          today, always available
    ├── ort / coreml       macOS: ANE + GPU
    ├── ort / xnnpack      linux-amd64
    ├── ort / directml     windows (legacy path; see 15.5)
    └── mlx (mlx-c FFI)    macOS, later, batch-shaped, only if 15.8 says so
```

- **Config, not code**: `engine.backend: auto | ort | coreml | mlx` and an
  ordered `engine.provider_ladder` (first rung that *loads* wins). This is
  our own automatic-EP selection — strictly better than a platform
  framework's, because `libsyscap` already knows the machine's load.
- **Availability is per slug**: `libmodelstorage` grows a
  `backends(slug)` reporting which runtimes have local artifacts (ONNX
  present? MLX dir present?). A missing artifact **skips that rung**; it is
  not an error.
- **Fallback happens at load time only** (provider absent, session load
  fails, shape unsupported) → next rung. Never a silent CPU-only run.
- **The load response reports the winner**: backend + provider + actual
  intra/inter threads. This is the repo's testing rule (assert on reported
  behavior, not instrumentation) and it is what makes every A/B below
  trustworthy.
- **Comparability**: `runtime + provider + precision` must land in the
  bench result metadata (`libbenchresults` config key). Quantized-GPU
  scores must never silently mix with CPU scores on a leaderboard.
  - **Pin-hole postmortem (2026-10-01)**: bench's ort pin was silently dead —
    `spawn_inprocess_with_engine` built `StdioArgs.backend`/`exec_provider`
    from the raw config instead of the effective section, so every
    "ort-pinned" A/B of a model with an `mlx:` variant served Metal (caught
    only because the "loading … on the mlx arm" load line named the arm).
    Fixed in `libmodelcallerclient` (client prefers the effective section,
    `effective_engine_io`) and defense-in-depth in modelcaller `State::new`
    (section wins over wire args). Rule: **the effective engine section is
    the single source of backend truth**; wire args are fallback only.
  - **Quality A/B (cnam test.csv, `embedder-cosine`, mean NDCG@10 over 71
    jobs)**: 0.6B — mlx 0.6322 vs ort 0.6374; 270m — mlx 0.6186 vs ort
    0.6174. Deltas ≤ 0.005, 95 % CIs overlap ⇒ Metal arm is
    ranking-quality-equivalent to the ONNX arm for both harrier models;
    `backend: auto` is safe for the gui.

### 15.4.1 the CoreML trap (why "it loaded" is not a result)

CoreML/ANE/DirectML want **static shapes** and recompile per shape. Two
consequences, both design requirements:

1. **Fallback is silent by default.** ORT logs node placement only when
   logging is raised; a session that places three nodes on CoreML and
   silently CPU-falls-back the rest *loads fine and looks fast-ish*. So the
   acceptance gate is **"N of M nodes placed on <provider>"** read from the
   placement log — never "the session loaded".
2. **Our op set is unverified.** Community exports (especially
   `q4f16`/quantized graphs) may carry ops or an opset the provider cannot
   take. A static read of the opset is inconclusive for these files (they
   use the implicit default domain), so this is settled by the placement
   log at load time, per model, recorded in bench metadata.

## 15.5 per-arch ladders

- **ANE is out of scope for this app (decided 2026-10-01).** It is a separate
  block reachable only by compiling a CoreML model, with static shapes only,
  a limited op set, LUT-quantized weights and a ~1–5 W budget — built for
  always-on background work, not throughput. Our encoder is the opposite
  shape (262k vocab, variable lengths, mean pooling) and the LUT path is
  lossy. It also cannot be split across at runtime: the CPU/GPU/ANE division
  is decided when the model is *compiled*, which is exactly the mechanism
  that measured 11.9× slower for us (§15.8 step 2). Metal and the GPU are the
  same thing — Metal is the API — so "the GPU" is the only accelerated target
  we are chasing, plus whatever the host CPU does in parallel beside it.
- **macOS**: `cpu → coreml → mlx`. `coreml` is compiled in on macOS and
  **measured rejected** (§15.5.2); `mlx` is the live rung (macOS-only, see
  §15.5.3.1). Battery politeness is a syscap question, not a backend
  question (deviation: it is off for this laptop).
- **linux/amd64**: `cpu → xnnpack` (already a feature flag). No GPU in scope.
- **windows**: `cpu → cuda / tensorrt → directml`. All three are **Cargo
  feature flags only** — `ort` ships prebuilt ONNX Runtime binaries and its
  build just places the DLLs next to the executable (`ort-sys`
  `build/dynamic_link.rs`), so **no compiler toolchain, CUDA toolkit or
  Xcode analogue is needed to build them**; MSVC and the prebuilt bundle are
  the whole story. (If DirectML's generic D3D12 path is ever adopted it needs
  the same placement gate as everything else — see below — because a
  partitioner can half-work, and on macOS it did: 11.9× slower.)
- **Every rung, every arch, passes the placement gate** before it is
  believed: a provider that silently CPU-falls-back is indistinguishable
  from one that works, and the harness prints the bitwise-identity tell
  (§15.8 step 2).
- **python sidecar**: rejected outright — it breaks the zero-children rule
  and adds a runtime to a single-binary app. A throwaway python probe is
  acceptable for measuring an upper bound; it never ships.

### 15.5.0 how the macOS-only rung stays out of other platforms' builds

The vendored MLX dependency would otherwise break every non-macOS build (no
cmake/Metal, and MLX has no Windows target upstream at all). So it is a
**target-scoped dependency**, exactly like the CoreML feature is a
`cfg(target_os = "macos")` gate today:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
mlxcel-core = { path = "vendor/mlxcel/src/lib/mlxcel-core", features = ["metal", "accelerate"] }
```

with the rung's code behind the same `cfg`. A Windows or Linux build never
sees the dependency, never invokes its cmake, and never needs a Metal
toolchain. (mlxcel's `cuda`/`rocm` features are Linux-only — a Linux+NVIDIA
MLX rung is conceivable in principle, but it is not in scope here.)

### 15.5.1 mlx-rs, assessed (2026-10-01)

[`oxiglade/mlx-rs`](https://github.com/oxiglade/mlx-rs) is the first MLX path
that fits this architecture, and it settles the objection that ruled MLX out
in §15.5: `mlx-sys` **vendors `mlx-c` and links it statically**
(`cmake` on a bundled `src/mlx-c`, then `rustc-link-lib=static=mlx` +
`framework=Metal`), so there is no Python and no subprocess — the
single-binary rule holds. It is on crates.io (`mlx-rs`, tracking MLX 0.32.2 /
mlx-c 0.6.0), MIT/Apache-2.0, with an `mlx-lm` crate and a Mistral example.

Three things to know before treating it as the macOS rung:

- **It is GPU, not ANE.** MLX's own feature list says "for now the CPU and
  GPU". ANE access through MLX is thin/experimental. So MLX buys Metal
  throughput, not the 16-core Neural Engine — which is the right medicine
  for the bandwidth-bound encoder we measured anyway (§15.7), and the weaker
  half of the original pitch.
- **It cannot read our models.** MLX loads MLX-format safetensors, not ONNX.
  Adopting it means a **second model zoo**: our ~30 slugs are ONNX with
  community quantizations (`q4f16`, `quint8`, `int8`), and an MLX version of
  any given embedder is a third-party conversion we would be trusting for
  ranking quality. That is a data-quality dependency, not an engineering one,
  and it is the real cost — not the binding.
- **Build weight.** cmake + bindgen + a static Metal library in the build
  graph of every machine that builds the gui (the vendored C++ trade-off,
  accepted, plus the toolchain requirements), on an *unofficial, actively
  developed* API whose versions track MLX releases (so MLX bumps are breaking
  changes for us).

**Decision: keep it as the evaluated macOS rung (step 3), not step 1.** The
step-2 rung (`ort`'s `coreml` feature, already a Cargo feature in
`lib/onnxwrapperrs/Cargo.toml`) answers the same question — does the GPU help
these artifacts, and do the ops actually land on it — for one flag and no new
model supply, and its two failure modes are both informative: nodes land and
it is fast (MLX never needs to exist), or placement is poor (that is the
evidence that justifies mlx-rs, with a target number to beat instead of a
hope).

### 15.5.2 the CoreML rung, measured negative at both ends

With the ladder in place (§15.8 step 2), the scout was probed too, because
the failure could have been specific to the 0.6 B:

| model (108 MB scout, 1500 texts, batch 16) | 1 session | 8 sessions |
|---|---|---|
| cpu | 851 ms (1762 texts/s) | 631 ms (2379 texts/s) — **1.35×** |
| coreml | 847 ms (1771 texts/s) | 630 ms (2379 texts/s) |

`bitwise_identical = true` for **every** coreml run: the placement gate fires
and CoreML placed **nothing** on the scout — identical timings because it *is*
the CPU. So CoreML EP is now measured at both ends of our size range: a
total no-op on the small model, **11.9× slower** on the 0.6 B (partial
placement). It is dead for our zoo. The config knob, the reporting and the
bucketing stay (any future rung needs them); `exec_provider` stays at `auto`.

### 15.5.3 mlxcel — the shape that actually fits

[`lablup/mlxcel`](https://github.com/lablup/mlxcel) (v0.7.0, Apache-2.0,
1 889 commits) is a Rust CLI + server that runs **MLX SafeTensors checkpoints
through native MLX C++ bindings, "without Python in the request path"**, and
ships exactly our workload: `mlxcel embed` / `rerank`, an OpenAI-compatible
`/v1/embeddings`, embedding families (BERT, ModernBERT, SigLIP, Qwen3,
cross-encoders…), and — decisively — **`mlxcel-core` is a library crate in
its workspace**, so it can be linked in-process behind the same session
interface as `onnxwrapperrs` rather than spawned. Its README notes many
standard SafeTensors embedding checkpoints "load without conversion".

Measured on this box (CV-length texts, batch 16, matched model pairs):

| model | ONNX + CPU (ours) | MLX + Metal (`mlxcel` 0.7.0 CLI) | speedup |
|---|---|---|---|
| harrier 270M (gemma3, 640) | 5.46 texts/s | **29.7 texts/s** | 5.4× |
| harrier 600M (qwen3, 1024) = our gui default | 1.60 texts/s | **8.7 texts/s** | 5.5× |

`user 0.33 s` for a 2.1 s wall on the 270M: the CPU is idle, Metal is
working. At 8.7 texts/s a 2000-CV embed pass is ~3.8 min instead of ~21.

**But the CLI is not the ceiling — our own in-process port is 4× faster than
it** (§15.5.3.3, measured after the port landed): **45.7 texts/s** on the same
600M, same CV-length texts, same batch 16, against our ONNX CPU path's
**1.56 texts/s** — a **29× speedup**, not 5.5×. 2000 CVs go from ~21 min to
~44 s. The gap to the CLI is overhead we simply do not pay (their CLI does
length-sorted micro-batching, format conversion and text formatting per call;
we already have ids in hand). So the honest headline for this rung is
**~29× on the 0.6 B embed**, and the CLI measurement should be read as a lower
bound on what the library path delivers.

**The quality gate** (16 identical texts, per-text cosine between the two
conversions of the same model):

| pair | per-text cosine | pairwise cosine delta |
|---|---|---|
| 600M / 0.6B | **0.966–0.969** | mean 0.010, max 0.029 |
| 270M, MLX default pooling (mean) | ~0.55–0.78 | mean 0.109 |
| 270M, MLX `--pooling last` | **0.895–0.961** (mean 0.930) | — |

**The 270M "near-orthogonal" verdict was our own pooling mistake, and it is
retracted.** Both ONNX exports pool **inside the graph** — the node names are
in the file (`model/st/pool_0/lasttoken_squeeze`, `last_idx_clamp`,
`sentence_embedding`), so the wrapper receives an already-pooled, already
L2-normalized `[B, D]` vector and applies no pooling of its own. mlxcel's
gemma3 family default is `Mean`, so the first comparison measured
last-token-vs-mean. Forcing `--pooling last` on the same checkpoint moves the
270M from 0.78 → **0.930** mean per-text cosine: the same space, ordinary
conversion drift, the same magnitude as the 600M pair.

The lesson generalizes beyond the 270M: **`1_Pooling/config.json` is absent
from both MLX checkpoints** (only `modules.json` names the module), so
mlxcel silently falls back to a family default — and the two families differ
(gemma3 `Mean`, qwen3 `LastToken`). A rung that adopts family defaults instead
of reading the checkpoint's declared pooling will look like a broken
conversion when it is a config mismatch. Pooling must therefore come from the
checkpoint, with the ONNX export as the reference for what it is.

What survives as a real caveat: 0.93/0.97 is not 1.0, so **ranking parity
still has to be bench-verified** rather than assumed. And cached vectors are
keyed by model, so a runtime switch invalidates them by construction (§15.4).

Also assessed and rejected: **Anemll** (1.7k stars, MIT) ports *generative*
LLMs to the ANE — Python 3.9 + coremltools conversion, Swift runtime, no
embedding path, beta caveats ("LUT4 quality is fairly low"), and a Gemma-3
FP16 overflow story. Right tool for an on-device chat app, wrong shape here.

**Decided (2026-10-01): `mlxcel-core`**, with `mlx-rs` (§15.5.3.2) as the
documented fallback if their MLX build or `layers.rs` churn proves
unworkable. The tie-breaker is breadth: `mlxcel-core` already carries the
encoder primitives for every family we might add later (and its `cuda`/`rocm`
features are the only path to a Linux+NVIDIA rung later), at the cost of a
layer-only port of the two backbones we actually need.

#### 15.5.3.1 what integrating mlxcel actually costs (measured, 2026-10-01)

`mlxcel-core` **is** a library crate (`crate-type = ["lib"]`, features
`metal` / `accelerate` / `cuda` / `rocm`) — but Cargo cannot point a git
dependency at a subdirectory, so it means **vendoring** (the worktree is
100 MB, ~79 MB of it source; only `src/lib/mlxcel-core` + `src/lib/mlx-cpp`
are needed, and `mlx-cpp` is a CMake shim that **FetchContent-downloads MLX
C++ at build time**, so the build wants network + cmake + the Metal
toolchain).

**Build feasibility, measured on this box** (probe crate outside the repo):

1. ✅ the build script wires up correctly — it invokes cmake with
   `-DMLX_BUILD_METAL=ON -DMLX_BUILD_ACCELERATE=ON` and arm64 flags.
2. ✅ **`cmake` was not installed** here; installed (a standard, reversible
   dev tool).
3. ✅ **full Xcode installed, and the build now succeeds end to end.** The
   last three steps were all toolchain setup, in order:
   `sudo xcode-select -s /Applications/Xcode.app` (until then `xcrun` resolves
   to CommandLineTools and cannot find `metal`), `sudo xcodebuild -license
   accept`, and `xcodebuild -runFirstLaunch` followed by
   `xcodebuild -downloadComponent MetalToolchain` — the Metal compiler is a
   **separate 839 MB component** even in a full Xcode 27.0 install, and
   `xcrun metal` fails with "missing Metal Toolchain" until it is there.
   `mlxcel-core` then builds from source in **2 m 45 s** and links, and an
   array round-trips through Metal (`from_slice_f32` → `try_eval` →
   `array_to_vec_f32`). So the vendored MLX dependency is real, not a
   promise.

**The cost every machine building the gui now carries**: a first build needs
network (FetchContent clones MLX C++ at a pinned commit), cmake ≥ 3.16, a C++20
compiler, and the Metal toolchain component on macOS. Subsequent builds use
MLX's cached `_deps/mlx-src`. This is the vendored-C++ trade-off the design
accepted, now with a measured number attached.

Note the runtime itself is fine — the **Homebrew `mlxcel` bottle runs here**,
which is exactly how the 5.4× numbers were measured without building anything.
So the blocker was *building* MLX from source, not running it.
`mlx-rs` does **not** dodge it — its `build.rs` also cmake-builds a
vendored `mlx-c` with `MLX_BUILD_METAL=ON` and compiles a metallib.

**The vendored subset is smaller than it looks**: the 79 MB worktree is mostly
docs/benches/webui. Only `src/lib/mlxcel-core` (6.3 MB) and `src/lib/mlx-cpp`
(3.3 MB, a cmake shim with no `Cargo.toml`) are needed — **~10 MB**, pinned to
MLX commit `81ba1c6a` via `mlx-cpp/CMakeLists.txt`'s `GIT_TAG`, which
`mlxcel-core/build.rs` parses as the single source of truth for the pin.

The sharper finding: **`mlxcel-core` has no embedding path.** It ships the
building blocks — `layers.rs` (9.8 k lines: `UnifiedEmbedding`,
`UnifiedLinear`, `Attention`, `RMSNorm`/`GemmaRMSNorm`, `SwiGLUMLP`),
`weights::WeightMap` (safetensors loading), `ops`, `utils`
(`create_bidirectional_padding_mask`, `create_causal_padding_mask`,
`create_bidirectional_window_mask`), `dtype` — but the embedding *models* and
the `/v1/embeddings` logic live in the root `mlxcel` package, which is
binaries-only (no `[lib]`). So the integration is **not** "adopt their
runtime", it is:

- **Shape A — build the encoder on their primitives** (recommended):
  vendor `mlxcel-core` (+`mlx-cpp`), and write the part that is actually
  missing for us: stack `gemma3_text` / `qwen3` encoder layers from
  `layers`, run with an attention mask, pool, L2-normalize.
  Bounded, and it depends only on the *library* — so it survives mlxcel's
  fast-moving CLI/server surface.
- **Shape B — vendor the whole repo and add a lib target** exposing their
  embed path: less new code, but we carry a patched fork of a
  v0.7.0-plus-unreleased tree and its whole release profile (their README:
  4–6 min release rebuilds) as a dependency.

**Shape A's real size, measured against their source (not the "~300 lines"
first guess).** The glue is genuinely lean: `src/embeddings/{model, pooling,
normalize, tokenize, limits, engine}.rs` is ~1.8 k lines touching only
`mlxcel_core::{MlxArray, UniquePtr, ops::*, utils::*, weights::WeightMap,
dtype}` — no private API. The weight is the **family backbones**, and they are
not in the library:

| file | lines | what a port drags in |
|---|---|---|
| `src/models/qwen3.rs` | 1 186 | `mlxcel_core::generate::LanguageModel`, `DecodeBatchContext`, `cache::*`, paged decode |
| `src/models/gemma3.rs` | 2 048 | the above plus `distributed::pipeline`, `kv_snapshot`, `model_owned` (a 65-entry module tree) |
| `src/models/qwen3_embedding.rs` | 159 | the `impl EmbeddingModel` is 41 lines; `Qwen3Model::forward_hidden` is `pub(crate)` |
| `src/models/gemma3_embedding.rs` | 331 | the `impl EmbeddingModel` is 42 lines |

So Shape A means porting a **layer-only slice** of each backbone (embedding +
attention + MLP + norm + mask + rope) and dropping the generation half. That
is still bounded, but it is a real port, not a 300-line shim — and it is the
one genuinely expensive step in this plan.

**What we do NOT have to port** (measured, and it shrinks the job further):
the tokenizer. `LoadedEmbeddingModel` carries mlxcel's own
`MlxcelTokenizer` (3 backends, ~3.5 k lines, pulling `sentencepiece` +
`hf-hub`), but we already tokenize with `libtokenizer` over the same
`tokenizer.json`, and our ONNX path already owns mask construction, pooling
and L2-normalization. The rung should therefore take **token ids from
`libtokenizer`**, not a second tokenizer — one tokenizer, one set of ids, and
a parity test that compares vectors from identical ids.

Either way the rung sits behind the same session interface as
`onnxwrapperrs` (design §15.4), ONNX stays the fallback, and the bench parity
run is the gate.

#### 15.5.3.2 fallback: `mlx-rs`

If `mlxcel-core` turns out to be unworkable (their MLX C++ fetch at build
time, upstream churn in `layers.rs`, or a break we cannot patch around
without forking), the documented fallback is
[`oxiglade/mlx-rs`](https://github.com/oxiglade/mlx-rs) (§15.5.1): same
vendored-MLX, but only the raw bindings — **no embedding models**, so we
would own the encoder, pooling and quantization handling outright. That is
strictly more code than Shape A and no less MLX build risk, which is why it
is the fallback and not the plan.

#### 15.5.3.3 the port, measured (2026-10-01)

Shape A's encoder was built and gated outside the repo, on `mlxcel-core` 0.7.0
primitives only — `UnifiedEmbedding`, `FusedQKVLinear::from_weights_separate`,
`UnifiedLinear`, `RMSNorm`, `fast_rope`, `layers::attention`,
`create_causal_padding_mask`, `weights::load_weights_from_dir_with_subfolders`
— plus pooling and L2 written against the same op sequence as mlxcel's
`pool(PoolingMode::LastToken)`. No code from mlxcel's binary crate.

**Gate 1 — port fidelity (does our composition equal theirs?).** 16 texts,
`harrier-mlx-600m`, our port vs `mlxcel embed --pooling last`:

| metric | value |
|---|---|
| per-text cosine | **min 0.999999, mean 0.999999** |

**Gate 2 — runtime parity (does Metal agree with our ONNX?).** 64 CV-length
texts (~2.5 k chars, ~630 tokens), 600M, in-process MLX vs our ONNX CPU path:

| metric | value |
|---|---|
| per-text cosine | min 0.9658, mean **0.9694** |
| pairwise cosine delta | mean 0.013, max 0.042 |
| same nearest neighbour | 50/64 (78 %) |
| mean absolute rank shift | 3.2 of 63 candidates |
| mean top-10 overlap | **9.41/10** |

**Gate 3 — throughput (the point of the rung).** 16 CV-length texts, batch 16,
one session, warm model cache:

| runtime | texts/s | 2000 CVs |
|---|---|---|
| our ONNX + CPU EP (ORT, 16 threads) | 1.56 | ~21 min |
| `mlxcel` CLI 0.7.0 (Metal) | 8.93 | ~3.7 min |
| **our in-process MLX port (Metal)** | **45.7** | **~44 s** |

**29× over the ONNX path, 5.1× over the `mlxcel` CLI** on identical input. The
CPU-side ceiling from §15.7 is what makes the ratio so large: 1.56 texts/s was
already proven to be bandwidth-bound and insensitive to threads, so the win is
the weights living on the GPU's path, exactly as §15.8 predicted.

Two findings from building it that the design did not anticipate:

1. **Truncation dominates the parity number, not the runtime.** With a probe
   cap of 512 tokens the 600M pair measured 0.427 mean cosine; raising the cap
   to 2048 (past the inputs' real length) moved it to 0.969. A truncation
   mismatch between two runtimes of the same weights looks exactly like a
   broken conversion. The rung must derive `max_length` from the *same* place
   `onnxwrapperrs` does, or the parity gate is measuring the harness.
2. **Special tokens are load-bearing for last-token pooling.** Qwen3-Embedding
   pools the appended `<|endoftext|>`. Encoding with
   `add_special_tokens = false` still produces plausible vectors and still
   matches nothing: cosine 0.58 against the reference. A pooling gate that
   tolerates "roughly similar" would have shipped that.

Neither is a defect in the plan — both are the repo's existing rule (one
tokenizer, one set of ids) being load-bearing in a new place.

#### 15.5.3.4 the gemma3 slice, and a second family-default trap

The 270M harrier is `gemma3_text`, which is not the qwen3 encoder with a
different width. Four differences, all of which the port has to get right:
`GemmaRMSNorm` (the `1 + w` convention), a `sqrt(hidden_size)` scale on the
token embeddings, **GeGLU** instead of SwiGLU, **four** norms per layer with
*clipped* residuals (`compiled_clip_residual`), and bidirectional rather than
causal attention. It also nests its rope bases under `rope_parameters`
(`full_attention.rope_theta = 1e6`), not a flat `rope_theta`.

Gate 1 (port fidelity, 16 CV-length texts, vs `mlxcel embed --pooling last`):

| metric | value |
|---|---|
| per-text cosine | min 0.999996, mean **0.999998** |

Gate 3 (throughput): **~100 texts/s** in-process vs **5.46 texts/s** on our
ONNX CPU path — **~18×**, the same order as the 0.6 B's 29×.

**The attention mask is the second family-default trap, and it is worse than
the pooling one.** `harrier-mlx-270m` declares
`"use_bidirectional_attention": false`, and so does our ONNX export — both run
**causal** attention. mlxcel's `Gemma3EmbeddingModel` builds **bidirectional**
masks unconditionally. Measured against our ONNX vectors on identical texts:

| mask used by the MLX side | per-text cosine vs our ONNX 270M |
|---|---|
| bidirectional (mlxcel's family behaviour) | 0.703 |
| **causal (what the checkpoint declares)** | **0.9368** |

So a rung that copies mlxcel's family behaviour would ship a *different model*
for this checkpoint — 0.70 agreement, and a silently worse ranking — while
looking completely healthy. Together with the pooling default, the rule is
now explicit and belongs in the code, not in a doc:

> **Nothing about a checkpoint's semantics comes from the runtime's family
> default.** Pooling, mask convention, rope base, max length and special-token
> handling are read from the checkpoint (`config.json`, `tokenizer.json`, the
> ONNX graph's own pooling nodes). A family default is only a fallback for a
> checkpoint that declares nothing, and the parity test must cover every
> semantic we derive rather than one pooled vector.

#### 15.5.3.5 what landed: `lib/libmlxembed`

The port is now a crate in the repo rather than a probe outside it:

```
lib/libmlxembed/
  src/checkpoint.rs   pure serde_json: family, shapes, quantization, and the
                      semantics (pooling, mask, rope base) read from the
                      checkpoint — unit-tested on every platform
  src/qwen3.rs        fused QKV, q/k RMS-norm, RoPE, masked SDPA, SwiGLU
  src/gemma3.rs       GemmaRMSNorm, sqrt(hidden) embed scale, GeGLU, four
                      norms + clipped residuals, declared mask, sliding window
  src/pooling.rs      pool (cls/mean/last, mask-driven) + L2, op-for-op the
                      same sequence as the reference runtime
  src/encode.rs       Embedder::load / embed, the handle the ladder holds
  tests/parity.rs     the opt-in gate against a reference runtime
vendor/mlxcel/        `mlxcel-core` 0.7.0 + the `mlx-cpp` cmake shim (9.7 MB),
                      Apache-2.0, pinned to MLX commit 81ba1c6a
```

Both families re-verified through the crate's own API (not the probe):

| check | qwen3 600M | gemma3 270M |
|---|---|---|
| vs `mlxcel embed --pooling last` | **0.999998** | 0.947 — *by design*, see below |
| vs a causal reference run | — | **1.000000** |
| vs our ONNX export | 0.969 | 0.937 |
| load report | `mlx:qwen3 dim=1024 pooling=LastToken mask=Causal layers=28` | `mlx:gemma3_text dim=640 pooling=LastToken mask=Causal layers=18` |

The gemma3 row is the family-default rule earning its keep: the crate reads
`use_bidirectional_attention: false` and runs causal, so it *disagrees* with
`mlxcel` by 0.947 while agreeing with our ONNX export — the divergence is the
feature. A future change that makes this test "pass" against mlxcel would be
introducing the bug the design documented.

**The arm is opt-in (`--features mlx`), not just macOS-gated.** `libbuild`
discovers every crate under `lib/`, and building this crate with MLX compiles
MLX from source: ~2m40s and ~2.1 GB under `target/` on first use. An
always-on dependency would tax every `dev.py` run on this laptop for a rung
that is not the default backend. So the default build of `libmlxembed` is a
few seconds of pure-Rust config parsing (7 tests, no MLX), and
`libmlxembed::AVAILABLE` is the constant the ladder asks before it tries.

#### 15.5.3.6 wiring it up, and the bug class it exposed

The arm is now the default on macOS: `modelcaller` depends on `libmlxembed`
with `--features mlx` *on macOS only*, so "Metal by default" is not a flag
somebody has to remember to pass. `engine.backend` resolves per model —
`auto` takes Metal where the model declares an `mlx:` variant, `ort` pins the
CPU floor, `mlx` demands Metal and fails loudly without it.

Two design points that are not obvious:

1. **Bench pins `ort` unconditionally** (`meta::syscap::clamp_for_bench`).
   `engine.backend: auto` is a gui-facing default; a bench run that followed it
   would redefine the floor every recorded result was measured against. An
   *explicit* `backend: mlx` is honoured, which is how the parity gate measures
   Metal without a second mechanism.
2. **Cached vectors are keyed by model slug, and the MLX variant is a different
   slug.** Switching a job's default embedder therefore does not silently reuse
   ONNX vectors; it re-embeds. That is the §15.4 rule working, not a migration.

**The bug the wiring exposed, and why it is worth writing down.** Session
handles are numbered per arm — each wrapper starts at 1. The first version of
this routed by a single global `session -> runtime` map, and with the scout on
ONNX and the heavy embedder on MLX both holding handle 2, the 640-dim 270M's
embed was routed to the scout's session and came back **1024-dimensional**. No
error, no warning, a plausible vector of the wrong width. It surfaced only
because six serve-contract tests asserted on grades the cascade produces.

The rule: **an identifier minted inside a component is not globally unique, and
anything that keys on it across components must re-key it.** modelcaller now
allocates its own global session id and rewrites it to the arm's own handle on
the way out; `arm_session` has no default arm, because defaulting would send a
request to an arm that does not hold the model — which answers with *someone
else's* vectors. Two tests pin it.

**Also fixed while measuring: the memory signal was dead on this machine.**
`sysinfo::available_memory()` returns **0** here (verified on 0.34 and 0.37),
and `libsyscap` derives `mem_pressure = 1 - avail/total` from it — so the whole
politeness policy was pinned at `verdict=Hold, thread_cap=1, wave_scale=0` on a
box with ~5.7 GiB free. Nothing looked broken; every number was just wrong.
`libsyscap::mem` now reads `vm_stat` on macOS (free + inactive + speculative,
wired excluded), with a captured-output test and a live non-zero guard. A
silently-wrong capacity signal is worse than none, because every downstream
consumer still gets a plausible number.

### 15.5.4 how big a model can this machine actually hold?
`rerank`/`generate` stay ONNX-only for now; the MLX arm answers them with
`unsupported_cmd` naming what it does serve, so a caller can never read "ran
and returned nothing" as a result.

### 15.5.4 how big a model can this machine actually hold?

Measured throughput per parameter class (§15.5.3.3) sets the ceiling: the
0.6 B does 45.7 texts/s on Metal in-process, so a 27 B (~45×) would be
~1 text/s — about half an hour for a 2000-CV pass — and it would not even be
resident: `harrier-mlx-27b` is **8-bit affine, hidden 5376, 62 layers = 27 GB on
disk** against **16 GB of unified memory**. MLX mmaps safetensors, so it
would technically run by paging every layer from disk, which is far worse
than the arithmetic suggests.

So the ladder's size ceiling on this box is set by *residency*, not by FLOPs:
4-bit weights up to ~8–10 B parameters fit with room for activations and the
desktop. A 27 B is not a candidate for bulk embedding here — it would be a
reranker over a top-10 shortlist at best, and even that needs a smaller
quantization than the 8-bit checkpoint on disk. If a much larger embedder is
wanted, the lever is **distillation** (a smaller model trained for the job),
not scale — which is also §15.8's "model size is the dominant lever".

## 15.6 shape bucketing (the enabler, backend-independent)

Every accelerator rung needs fixed shapes, and we feed variable batch
(1..capacity) with variable chunk token counts. So before any provider
work: bucket the sequence length to a small fixed ladder (pad/truncate to
the next bucket) and use fixed batch buckets.

*Implemented* as `onnxwrapperrs::session::bucket_len` (ladder 128/256/512/768/1024,
never past the model's context window) and applied **only when a provider is
registered** — on the CPU EP the padding is pure waste. It is not sufficient
on its own: §15.8 step 2 measured CoreML at 11.9× slower *with* bucketing
(8.5× without), so shape churn was not that provider's problem — partial
placement was. Bucketing is still the right thing for any future rung that
really does compile per shape (an MLX/ANE-style static-shape runtime).

## 15.7 CPU data-parallel fan-out (the cheap rung) — MEASURED

The CPU-native strategy. `N` sessions of the same model, each with
`capacity / N` intra-op threads, each taking a disjoint slice of the pending
texts (`onnxwrapperrs::SessionManager::load(.., fan_out)`, threaded through
the `embed_sessions` load-payload field; the total thread budget is divided,
never multiplied).

**What we measured** (6P+12E box, batch 16, via the fan-out A/B harness that
used to live at `tests/tests/embed_fanout_bench.rs`, now deleted — see the end
of this section):

| model (weights) | 1×16 | 2×8 | 4×4 | 8×2 | verdict |
|---|---|---|---|---|---|
| `static-similarity-mrl` (108 MB) | 827 ms | 725 ms | 697 ms | **609 ms** | **1.36×** — cache-resident, compute-bound |
| `harrier-oss-v1-0.6b-ONNX-q4` (~600 MB, 600 texts) | 390 ms | 404 ms | 393 ms | 389 ms | **1.00×** — bandwidth-bound |

Embeddings are identical across shapes (min cosine 0.999999 — ORT picks
different kernels per shape, so this is a cosine gate, not a bitwise one).

**The conclusion the design was built to test:** the harrier 0.6 B embed does
**not** need more cores, and it does not want them split either — 1×16 ≈
8×2 means intra-op threads buy nothing past ~2 for this model, because every
forward pass streams the whole weight set from unified memory. Splitting the
batch across sessions splits the same bandwidth. This is why the earlier
gui-suite scaling was sublinear (5 → 16 threads ≈ 1.3×): it was never
core-bound, it was bandwidth-bound.

So the auto rule is **model-size dependent**, and the wrapper now owns it
(`auto_fan_out`): a model whose weights fit the cache budget (≤ 384 MB
today) fans out (8 sessions), a bigger one keeps a single session. Both
numbers above are the reason; `embed_sessions` in the load payload overrides
the count for a specific model.

Consequences that follow from the measurement, not from taste:

- **The 0.6 B's remaining CPU levers are batch size and silicon.** Bigger
  batches amortize the weight streaming (measured separately, §15.8 step 1b);
  past that, only a different runtime (CoreML/MLX — weights resident on the
  accelerator) or a smaller/distilled model changes the number. Note the
  270 M variant is ~4× smaller and the scout is 108 MB — model choice is a
  bigger lever than any thread count on this box.
- Fan-out is *not* a general speedup, so it must not be sold as one, and it
  must stay off for big models (8 × 600 MB would be 4.8 GB of sessions for
  nothing).
- The scout, which every cascade runs first, is in the winning regime: it
  gets 1.36× for free.

**The harness is gone** — the question above is answered, so the A/B test was
deleted rather than cached and replayed: its timed legs were minutes of real
inference per arm (600 texts × 4 session counts on the 600 MB 0.6 B — ~390 s
each on a cache miss), which made `dev.py` runs unpredictable for a number
nobody re-reads. The table above is the record; re-measure by hand instead of
on every suite run (`embedder_thread_probe --ignored` is the sanctioned manual
probe). The pure logic behind the auto rule stays gated by cheap unit tests in
`onnxwrapperrs`: `fan_out_splits_the_thread_budget` (the budget is divided,
never multiplied — floor 1 per pool member) and
`auto_fan_out_follows_the_weight_footprint` (cache-resident → 8 sessions,
bandwidth-bound → 1). What is no longer asserted automatically is the
provider-placement gate (bit-identity vs the CPU run); re-derive it from a
hand run when a new provider rung lands.

## 15.8 measurement plan — each step decides the next

Every number below is on the giant job, warm model cache, recorded in the
result yaml (runtime/provider/precision/threads in the metadata).

1. **Step 0 — plumbing (no behavior change)**: report backend+provider+actual
   threads in the load response; runtime/provider/precision in bench
   metadata; shape buckets behind a flag (default off). Makes every later
   comparison honest.
2. **Step 1 — CPU fan-out A/B**: 1×16 vs 2×8 vs 4×4 vs 8×2 sessions on the
   2000-input harrier embed. **DONE — the answer is in §15.7: 1.00× on the
   0.6 B (bandwidth-bound), 1.36× on a 108 MB model (compute-bound).** The
   box is NOT core-starved; the 0.6 B is bandwidth-bound and its levers are
   batch size and silicon, not cores.
   1b. **Batch-size curve on the 0.6 B** (16/32/64/128 at one session): if
   throughput climbs with batch, weight-streaming amortization is the live
   CPU lever and the derived `embed_batch` (currently the thread cap) should
   follow the curve instead. **DONE — the opposite: 1.54 → 1.46 → 1.38 →
   1.05 texts/s at batch 16 / 32 / 64 / 128** (400 texts, one session, 16
   threads). Bigger batches are monotonically *worse* — attention and
   activation footprint grow faster than any amortization, and the working
   set stops fitting cache. So the derived batch (currently the thread cap,
   16) is already at the good end of the curve; raising it would hurt.

**After steps 1 and 1b the CPU side is exhausted for this model.** Threads
past ~2 do nothing (1×16 ≈ 8×2), splitting the batch does nothing (1.00×),
and bigger batches hurt (0.68× at 128). The harrier 0.6 B q4 sits at
**~1.5 texts/s** on this box — about 22 minutes to embed 2000 CVs — while
the 108 MB scout does **2463 texts/s** on the same texts. Two consequences,
in order of how much they actually matter:

1. **Model size is the dominant lever** (orders of magnitude), and it is a
   bench/quality decision, not an engineering one: the 270 M variant is ~4×
   smaller, a distilled embedder is smaller still, and the cascade already
   starts from the cheap scout ordering. Any "make embedding fast" work
   should start by measuring quality at a smaller model.
2. **Silicon is the remaining engineering lever** — weights resident on the
   GPU/ANE is the one thing that changes the 1.5 texts/s number, which is
   what steps 2–3 (CoreML EP, then MLX) are for. They are now justified by
   elimination rather than by promise, and the gate stays the same: node
   placement must actually land on the provider (§15.4.1).
3. **Step 2 — CoreML rung**: enable the feature behind the ladder, run
   with logging raised, record node placement per model. **DONE, and it is a
   negative result.** `engine.exec_provider: coreml` (CoreML compiled in on
   macOS by a target-scoped `ort` dependency), 200 CV-length texts, batch 16,
   one session:

   | provider | time | throughput |
   |---|---|---|
   | cpu (auto) | 124.7 s | 1.60 texts/s |
   | coreml | 1482.7 s | 0.13 texts/s = **11.9× slower** |

   The placement gate says this is NOT a silent no-op: the vectors differ
   from the CPU run (min cosine 0.999999, `bitwise_identical=false`), so
   something did run on CoreML. Shape bucketing (§15.6, implemented as
   `bucket_len` — provider-conditional, since padding is waste on CPU) did
   **not** rescue it: the same run unbucketed was 1350.6 s, i.e. already
   8.5× slower, and the shape count barely changed because the tokenizer
   truncates to the model's context window anyway. So the cost is not
   per-shape compilation — it is **partial placement**: the expensive ops stay
   on the CPU (the op/opset coverage limit), the cheap ones move to CoreML,
   and the partition boundary breaks the CPU-side graph fusion the plain CPU
   EP had. Worst of both worlds, which is the failure mode §15.4.1 warned
   about — except that here the tell was speed, not silence.

   **Decision: CoreML EP is rejected for our embedders.** The ladder stays in
   the code (it is cheap, provider-conditional, and any future rung needs it),
   `engine.exec_provider` stays as config with `auto` as the default, and
   `load` keeps reporting the provider so this can never be mistaken for
   "accelerated" again.
4. **Step 3 — MLX (only if 1 says bandwidth-bound and 2 says CoreML can't
   take our ops)**: `mlx-c` C API over FFI, in-process, vendored C++/Metal
   (accepted trade-off), MLX-converted checkpoints (`mlx-community`
   coverage first), **batch-shaped** (32..256), never per-core split.
   Start with the models where a quality conversion exists.
   **GATE PASSED and the rung is built (§15.5.3.3).** Steps 1 and 1b said
   bandwidth-bound, step 2 said CoreML cannot take our ops — exactly the
   condition this step was conditional on. On `mlxcel-core` 0.7.0 the
   in-process 600M embedder measures **45.7 texts/s vs 1.56 texts/s** on our
   ONNX CPU path (**29×**) at **0.969** per-text cosine, with port fidelity
   against `mlxcel embed` at **0.999999**; the gemma3 slice clears the same
   gates at 0.999998 fidelity, 0.937 parity and **~18×** (§15.5.3.4). What is
   left is not "is MLX faster" but the production wiring: vendor the crate,
   take ids from `libtokenizer`, read pooling/mask/length from the checkpoint
   rather than from a family default, put the rung behind the session
   interface, and bench the ranking.

## 15.9 non-goals and risks

- Not replacing ORT as the default; CPU EP stays the floor every arch can
  fall back to, and bench comparability depends on it.
- Not a general plugin system: the ladder is a fixed, ordered list per
  arch in `ikapp.yaml`, not user-supplied provider strings.
- Risks: quantization drift between runtimes (bench-visible, hence §15.4
  metadata); shape recompile storms under a derived batch (hence §15.6);
  silent CPU fallback (hence the placement-log gate); a vendored C++
  dependency if MLX is reached (accepted, but only behind step 3's gate —
  now paid: MLX builds and links on this box, §15.5.3.1);
  **build-time network access**, since `mlx-cpp` FetchContent-downloads MLX
  C++ at first build; **truncation/special-token drift** between the two
  runtimes, which is invisible in the vectors unless the parity harness feeds
  identical ids (measured: both, §15.5.3.3).

## 15.10 open questions

- Which model is the first MLX candidate? The 0.6 B embedder is the
  throughput bottleneck; the 270 M is the cheap win. (The 4 B evidence LLM
  is not used in the gui path and is out of scope.) **Answered in practice:
  the 600M, because it is the gui default and the rank the cascade is stuck
  behind** — it is the model the port was gated on. The 270M still needs its
  gemma3 slice ported (sliding-window layers, bidirectional masks,
  `GemmaRMSNorm`, the dense stack) before it can be offered.
- Does the fan-out scheduler live in `onnxwrapperrs` (owns the sessions) or
  in `modelcaller` (owns the protocol)? Default: `onnxwrapperrs`, so the
  JSONL contract and every caller stay untouched.
- Target fan-out cap: memory-bound on 16 GB, or verdict-aware (use
  `capacity / N` threads per session, floor 1)?
