# step 9 — caching

## depends on

- `04-embeddings.md` (embedding cache boundary)
- `08-config.md` (cache dir)

## 9.1 in-memory model cache

- loaded models kept alive across requests: once a model is loaded (via
  explicit `load` or lazy-load on first `embed`/`rerank`), its ONNX
  session stays in the wrapper's memory for the lifetime of the
  modelcaller process. the wrapper tracks sessions by a numeric handle
  (see `../onnxwrapperrs/design/01-wrapper.md`). the orchestrator tracks the
  mapping from logical model id to wrapper session handle.
- keyed by: logical model id + revision. a model can only be loaded once
  per process (loading the same id+revision again is a no-op and returns
  the existing session handle). loading the same id with a different
  revision is an error — the caller must `unload` first or use a
  different modelcaller instance. loading with different `opts`
 (different EP, precision, threads) also requires an `unload` first; a
  session's EP and precision are fixed at load time.
- eviction: **explicit only** — the `unload` command. there is no LRU,
  no auto-eviction, no memory-pressure-triggered unload. reasons: (a)
  the number of models in use at once is small (typically 1–2: one
  embedder, one reranker), so unbounded growth is not a risk. (b) auto-
  eviction would silently change behavior between requests. (c) the
  caller (ik5, bench4) knows its own memory budget and can `unload`
  when switching tasks. if the host runs out of memory, the OS OOM
  killer handles it — modelcaller does not try to outguess the OS.

## 9.2 embedding cache (on disk)

the duplicate-prone work ik5/bench4 each cached separately. centralize.

- cache key: `SHA-256( model_id || "||" || revision || "||" || normalize || "||" ||
  preprocess || "||" || lowercase(text) )` for each input text. `lowercase(text)`
  is used (not raw text) so that caching is insensitive to input
  casing when `preprocess` is `"lowercase"`; the hash already includes
  the `preprocess` value so non-lowercased embeddings get a different
  key. `revision` is included so a model update invalidates old cache
  entries (see `03-models.md` 3.2). `batch_size` and `exec_provider` are
  NOT part of the key — they affect performance, not output correctness,
  and an fp32 vector on CPU equals an fp16 vector on CUDA (within
  floating-point tolerance).
- storage: **one file per hash** under
  `$MODELCALLER_MODEL_DIR/embeddings/<model_id>/<first 2 hex chars of hash>/<hash>.bin`.
  each file is a flat binary blob: `[dim:u32_le][n_vectors:u32_le][f32_le * dim * n_vectors]`.
  single-text files have `n_vectors=1`. batch results are stored as one
  file per text (the orchestrator splits batch results into per-text
  cache entries). this layout avoids any database dependency (no lmdb,
  no sqlite) — it is just files on disk, trivially inspectable and
  cacheable by the OS page cache. the two-level hash directory avoids
  huge directory listings.
- hit path: orchestrator computes the cache key(s) for the input text(s).
  for each text, if the `.bin` file exists and is readable, the
  orchestrator reads the vector from disk and returns it, skipping the
  wrapper call entirely. the hit is reported on stderr at `debug` level.
- miss path: orchestrator sends the text(s) to the wrapper as normal,
  receives the vectors, writes each vector to its `.bin` file (creating
  parent directories as needed), and returns the vectors to the caller.
  write failures (disk full, permissions) are logged on stderr but do
  NOT fail the request — the caller still gets its vectors.
- invalidation: cache entries are invalidated by (a) changing the model
  revision (different hash prefix), (b) deleting the cache directory
  manually, or (c) re-running `modeldownloader --force` which clears the embedding cache for
  that model. there is no TTL — cached vectors for the same model+text
  are always valid. `MODELCALLER_NO_CACHE=1` env var disables disk
  caching entirely (hits are never checked, misses are never stored).

## 9.3 sharing across callers

- tauri app and bench4 both call modelcaller; cache is inside one
  process. note: separate modelcaller processes (e.g. ik5 has one, bench4
  has another) have separate in-memory model sessions but **share the
  same on-disk embedding cache** (they use the same `model_dir`). this is
  safe because cache files are write-once (keyed by content hash — the
  same text always produces the same file path) and never mutated. two
  processes writing the same cache entry simultaneously write identical
  bytes, so there is no corruption risk.

## 9.4 example

cache key computation for text `"Rust developer"` with model
`nomic-embed-text-v1`, revision `main`, `normalize=true`,
`preprocess=lowercase`:

```
preimage = "nomic-embed-text-v1||main||true||lowercase||rust developer"
hash    = SHA-256(preimage)  # e.g. a3f1b2c4...
path    = ~/.cache/modelcaller/embeddings/nomic-embed-text-v1/a3/a3f1b2c4....bin
contents = [768 as u32_le][1 as u32_le][0.012, -0.034, ... 768 floats]
```

hit: file exists -> read 3084 bytes (8 + 3072) -> return vector, skip wrapper.
miss: file absent -> call wrapper `embed` -> write 3084 bytes -> return vector.
10k CVs, 90% cache hit: 9000 disk reads (~0.1s), 1000 wrapper calls.
