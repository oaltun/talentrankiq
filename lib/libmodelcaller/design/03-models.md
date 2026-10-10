# step 3 — models

## depends on

- `02-interface.md` (the `model` field, `load`/`embed`/`rerank`/`models`/`pull`)
- `08-config.md` (search paths / cache dir)
- `11-profiling.md` (footprints drive selection)
- `../onnxwrapperrs/design/01-wrapper.md` (the wrapper executes a resolved model)

this step defines the **model catalog**: how a model is identified,
versioned, resolved to files, and registered. it is the single
specification both projects implement — the orchestrator uses it for
**resolution + selection + caching**; the wrapper uses the per-model
**adapter** (tokenizer / pooling / IO mapping) to execute. the JSONL
protocol stays unchanged when models are added.

## 3.1 model identity

a model is identified by a **logical id**, not a file path or on-disk
format:

- format: lowercase `family-name` with an optional version token, e.g.
  `nomic-embed-text-v1`, `bge-reranker-v2-m3`, `jina-reranker-v1-tiny-en`.
  hyphens separate tokens; no slashes, no `.onnx`/`.gguf` suffix.
- this is the string callers pass in `model` (see `02-interface.md`) and
  what `load`/`embed`/`rerank` accept. an unknown id -> `no_model`
  (strict, per `02-interface.md`).

> **GGUF vs ONNX (important):** today these models ship as **GGUF**
> files run by lemonade. modelcaller runs them through **ONNX Runtime**
> (via a binding such as Rust `ort` or Zig `onnxruntime.zig`), so the
> artifact is an **ONNX** model (`.onnx`/`.onnx.gz`) plus tokenizer +
> config files. the production `modeldownloader` **downloads the ONNX model from
> HuggingFace**; GGUF->ONNX conversion + HF upload are performed
> upstream by the separate **modelconverter** project (dev/bench-time only,
> `modelconverter/design/00-overview.md`), so infer1 never touches GGUF.
> the logical id (e.g.
> `nomic-embed-text-v1`) stays the same across both formats — callers
> never see the on-disk format.

## 3.2 versioning

- the id embeds the major version (`-v1`). a breaking change to a model
  (different vector space, different dim) gets a new id.
- for reproducibility (bench4), pin a specific **revision**: `modeldownloader --revision` takes
  `source` + `revision` (e.g. HF tag/commit) and stores the result under
  `~/.cache/modelcaller/<id>/<revision>` (or content-hash). the
  orchestrator records the resolved revision alongside the id.
- changing a model's version changes its output space, so the embedding
  cache (`09-caching.md`) must key on **id + revision**, not id alone.

## 3.3 resolution (where files live)

a logical id resolves to a **model directory** under the cache root. two
layouts are supported; the orchestrator tries the new layout first, then
the legacy flat layout.

### 3.3.1 new layout (HuggingFace ONNX repos, multiple quantizations)

HuggingFace ONNX repos routinely ship the artifact in an `onnx/`
subdirectory (e.g. `onnx/model.onnx`) and may offer several
quantizations (`model_fp16.onnx`, `model_q4.onnx`, `model_q4f16.onnx`,
`model_quantized.onnx`, each with sibling `*.onnx_data` external-data
files). to keep one quantization per logical id, each id gets its own
directory and the download preserves the repo's directory structure:

```
~/.cache/modelcaller/<id>/
├── repo/                          # exact HuggingFace tree (unmodified)
│   ├── onnx/
│   │   ├── model_fp16.onnx        # the single ONNX for this id
│   │   ├── model_fp16.onnx_data    # external weights (resolved by ORT
│   │   ├── model_fp16.onnx_data_1  #   relative to the .onnx file)
│   │   ├── tokenizer.json          # remapped next to the .onnx so the
│   │   └── config.json             #   wrapper finds it
└── meta/
    └── source.json                 # provenance: source_url, repo, revision,
                                    #   onnx_path, files[]
```

- `<id>` is a **slug** chosen by the caller (e.g.
  `harrier-oss-v1-0.6b-ONNX-fp16`). there is **no `<revision>` level** in
  this layout; the slug already pins a specific quantization.
- the orchestrator **scans `repo/` recursively** for `*.onnx` (excluding
  `*_data` external-data files) and requires **exactly one**. finding
  zero -> `no_model`; finding more than one -> `ambiguous_onnx` (cache
  only one quantization per id). ORT resolves the sibling `*_data` files
  automatically because they sit next to the `.onnx`.
- `tokenizer.json` / `config.json` are downloaded from the repo root but
  **remapped to sit next to the `.onnx`** so the wrapper (which loads the
  tokenizer from the model's own directory) finds them.

### 3.3.2 legacy layout (flat ONNX at repo root)

```
~/.cache/modelcaller/<id>/<revision>/
├── model.onnx            # or model.onnx.gz
├── tokenizer.json
└── config.json
```

### 3.3.3 resolution order (orchestrator side)

1. explicit path in `load` opts (if any),
2. new layout `~/.cache/modelcaller/<id>/repo/` (scan for `*.onnx`),
3. legacy layout `~/.cache/modelcaller/<id>/<revision>/model.onnx`,
4. not found -> `no_model` error (caller must run `modeldownloader` first).

the orchestrator resolves id -> `model_path` (the `.onnx`) and passes it
to the wrapper's `load`; the wrapper loads the tokenizer from the same
directory. `config.json` is the source of truth for `dim` and
`max_context_window` used by `models` / truncation / selection.

## 3.4 the model registry (catalog + adapter split)

- **orchestrator registry** — a table keyed by logical id:

  ```json
  { "id":"nomic-embed-text-v1", "kind":"embedder", "dim":768,
    "max_context_window":8192,
    "footprint":{"min_vram":1073741824,"min_ram":2147483648,
                 "rel_speed":"medium","quality":4} }
  ```

  used for resolution, `models` output, selection (`11-profiling.md`),
  and cache keys. no ONNX internals here.

- **wrapper adapter** (in `onnxwrapperrs`) — keyed by the same id, says how
  to *run* it: which tokenizer, input/output tensor names + dtypes
  (discovered from the ONNX graph), pooling method, and normalize flag.
  this is the "general ONNX loader" — generic session load in
  `onnxwrapperrs/design/03-implementation.md`, specialized per model by the
  adapter.

both projects implement the registry from this spec (it is data, not
logic, so the small duplication keeps the projects independent). a
logical id present in the orchestrator registry but missing a wrapper
adapter is a `load_err`.

## 3.5 well-known model specifics

concrete models we ship, with their adapter summary. footprints are in
3.3b.

| logical id                   | kind      | dim  | max_ctx | notes                                    |
| ---------------------------- | --------- | ---- | ------- | ---------------------------------------- |
| `nomic-embed-text-v1`        | embedder  | 768  | 8192    | ik5 default (`runtime.rs` `DEFAULT_MODEL`) |
| `bge-reranker-v2-m3`         | reranker  | —    | 8192    | bench4 default reranker                  |
| `jina-reranker-v1-tiny-en`   | reranker  | —    | 8192    | bench4 search-space alt (smaller/faster) |
| `<tiny-embedder e.g. MiniLM>`| embedder  | 384  | 512     | low-end fallback (fast, lower quality)   |

per-model adapter spec:

- **`nomic-embed-text-v1`** (embedder): tokenizer (BPE/wordpiece via
  `tokenizer.json`); preprocessing `lowercase`; **mean-pool** over token
  embeddings; **L2-normalize** (default; overridable via `opts.normalize`);
  output = one f32 vector of length `dim` (768) per text.
- **`bge-reranker-v2-m3`** (reranker): tokenizer; cross-encoder takes
  `(query, doc)` concatenated with the model's separator; output = one
  f32 relevance logit per pair (higher = more relevant). `dim` is N/A
  (it emits a scalar score, not a vector).
- **`jina-reranker-v1-tiny-en`** (reranker): same shape as bge, smaller
  network -> faster, slightly lower quality.

### 3.5b footprints & tiers

every well-known model carries a resource footprint so the profiler can
pick the best one the host can actually run (start.md: don't default to
a model that's too slow or won't load; do default to a larger one when
the user can handle it).

| logical id                   | kind     | min_vram | min_ram | rel_speed | quality |
| ---------------------------- | -------- | -------- | ------- | --------- | ------- |
| `nomic-embed-text-v1`        | embedder | 1 GiB    | 2 GiB   | medium    | 4       |
| `<tiny-embedder e.g. MiniLM>`| embedder | 256 MiB  | 512 MiB | fast      | 2       |
| `bge-reranker-v2-m3`         | reranker | 1.5 GiB  | 2 GiB   | medium    | 4       |
| `jina-reranker-v1-tiny-en`   | reranker | 512 MiB  | 1 GiB   | fast      | 3       |

selection algorithm (detail in `11-profiling.md`): filter by fit
(`min_vram <= available_vram` and `min_ram <= available_ram`), drop
anything exceeding the latency budget, then pick the highest `quality`
(smallest model is the guaranteed fallback). the winner is flagged
`recommended:true` in `models` and returned by `recommend`.

## 3.6 registration (adding a model)

to add a well-known model without a protocol change:

1. add a `ModelRecord` to the **orchestrator registry** (id, kind, dim,
   `max_context_window`, footprint) — feeds `models` / selection / cache.
2. add a matching **adapter** to the **wrapper** (tokenizer, IO names,
   pooling, normalize) keyed by the same id.
3. the `embed`/`rerank` commands are generic, so no protocol edit is
   needed. unknown id stays a strict `no_model` error.

## 3.7 example

```
model id:  nomic-embed-text-v1
resolve -> ~/.cache/modelcaller/nomic-embed-text-v1/main/
            model.onnx + tokenizer.json + config.json
orchestrator -> wrapper load:
  {"cmd":"load","model_path":"/.../model.onnx","exec_provider":"auto","precision":"fp16"}
wrapper -> {"session":1,"kind":"embedder","dim":768}
embed -> wrapper returns one L2-normalized 768-f32 vector per text.
```
