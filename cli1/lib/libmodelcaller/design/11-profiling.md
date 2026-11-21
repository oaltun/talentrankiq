# step 11 — hardware profiling & model selection

## depends on

- `03-models.md` (model footprints / tiers)
- `08-config.md` (capabilities feed config defaults)
- `02-interface.md` (`probe` / `recommend` commands; `model` auto-select)
- `../onnxwrapperrs/design/01-wrapper.md` (the wrapper owns the ONNX-Runtime-specific hardware scan)
- `12-implementation.md` (how probing is implemented, cross-platform)

## 12.1 purpose

at startup (and on `probe`) modelcaller scans the host and produces a
**capabilities** report. that report drives two things:

1. **model selection** — pick the *best* model a user can actually run:
   default them to a larger/higher-quality model when they have the
   VRAM/RAM/compute for it; never default to a model that would be
   unbearably slow or fail to load (start.md problem of "forever to
   respond / can't respond at all").
2. **parameter tuning** — derive sensible `batch_size`, `threads`,
   `precision` from the hardware instead of hard-coding them.

the result is the set of defaults used whenever a request omits an
`opts` override (see `02-interface.md`).

## 12.2 what to probe

| property            | source (cross-platform notes)                  | drives                  |
| ------------------- | ---------------------------------------------- | ----------------------- |
| os / arch           | `std::env` / `std::mem::size_of`               | EP + model build choice |
| cpu vendor / cores  | `sysinfo` (physical + logical)                 | `threads` default       |
| simd                | cpuid / `std::arch`                            | CPU EP speed            |
| ram total / free    | `sysinfo`                                      | model footprint filter  |
| gpus                | NVML/`nvidia-smi` (NV), Metal (macOS),         | `selected_ep`, vram     |
|                     | DXGI (Win), DRM/`sysfs` (Linux)                |                         |
| gpu vram total      | per-GPU query                                  | model + precision fit   |
| gpu fp16 / int8     | EP capability query (wrapper, `../onnxwrapperrs/design/01-wrapper.md`) | `precision` default     |
| disk cache free     | cache dir stat                                 | can we download a model? |

an EP may be compiled in but its hardware absent — probing confirms
what is *actually* usable, complementing the EP auto-select in
`08-config.md`.

## 12.3 capabilities report

exposed verbatim by the `probe` command (see `02-interface.md`):

```json
{"capabilities":{
  "os":"linux","arch":"x86_64",
  "cpu":{"vendor":"intel","physical_cores":8,"logical_cores":16,"simd":["avx2","avx512"]},
  "ram_total":17179869184,"ram_free":8589934592,
  "gpus":[{"index":0,"vendor":"nvidia","name":"RTX 4070",
           "vram_total":8589934592,"fp16":true,"int8":true,"compute_capability":"8.9"}],
  "disk_cache_free":107374182400,
  "selected_ep":"cuda"
}}
```

## 12.4 model tiers & selection

each well-known model (see `03-models.md`) carries a **footprint**:

- `min_vram`, `min_ram` (bytes) — hard fit filter.
- `rel_speed`: `"fast" | "medium" | "slow"` — expected latency class.
- `quality`: 1..5 — ranking/embedding quality.

selection algorithm (for a task = `embed` or `rerank`):

1. filter models of that `kind` to those whose `min_vram <=
   available_vram` and `min_ram <= available_ram`.
2. of the survivors, drop any that exceed a **latency budget** (a slow
   model on a weak host is rejected even if it fits in RAM).
3. pick the **highest `quality`** remaining. tie-break by `rel_speed`
   (prefer faster).
4. if nothing fits (extreme low-end), fall back to the smallest known
   model for that task (guaranteed to load).

the chosen id is what `embed`/`rerank` use when the caller omits
`model` (auto-select), and what `recommend` returns. it is also flagged
`recommended:true` in the `models` listing.

example tiers (illustrative; fill real numbers in `03-models.md`):
- embed: `nomic-embed-text-v1` (medium, q4) vs a tiny `all-MiniLM`
  (fast, q2) for low-VRAM hosts.
- rerank: `bge-reranker-v2-m3` (medium, q4) vs `jina-reranker-v1-tiny`
  (fast, q3).

## 12.5 parameter defaults from capabilities

when a request omits an `opts` key, modelcaller fills it from the
probe:

- `threads`: `min(logical_cores - 1, N)` on a desktop caller (ik5) so
  the UI isn't starved; full `logical_cores` for benchmark caller
  (bench4) for max throughput. (caller type known via launch / `opts`.)
- `batch_size`: scaled to `available_vram` (GPU) or `ram_free` (CPU),
  capped to keep latency bounded.
- `precision`: `fp16` if a GPU EP reports `fp16`; else `int8` if
  supported; else `fp32`. (precision is a new `opts` key — see
  `02-interface.md`.)
- `exec_provider`: the `selected_ep` from `08-config.md` auto-select.

explicit `opts` always override these (so bench4 can pin for
reproducibility, per `02-interface.md`).

## 12.6 lifecycle

- probe runs once at startup, cached for the process lifetime (hardware
  doesn't change mid-run).
- `probe` command re-emits the cached report on demand.
- if probing fails for a subsystem (e.g. no NVML), that field is
  omitted and selection degrades to conservative CPU defaults —
  never crash on a probe miss.

## 12.7 example

caller (ik5, low-end laptop) -> `{"cmd":"recommend","task":"rerank"}`
-> `{"model":"jina-reranker-v1-tiny-en","ep":"cpu","reason":"low_vram"}`

caller (bench4, RTX 4070) -> `{"cmd":"recommend","task":"embed"}`
-> `{"model":"nomic-embed-text-v1","ep":"cuda","reason":"fits_vram+quality"}`

then `{"cmd":"embed","texts":[...]}` (no `model`) runs on the
recommended model with probe-derived `batch_size`/`precision`.
