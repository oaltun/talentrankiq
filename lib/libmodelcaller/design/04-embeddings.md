# step 4 — embeddings (centralized logic)

## depends on

- `03-models.md` (which embedder models exist)
- `02-interface.md` (`embed` command)
- `../onnxwrapperrs/design/01-wrapper.md` (the wrapper owns tokenization +
  pooling + normalize)

## 4.1 goal

this is the logic ik5 and bench4 currently duplicate against lemonade.
the orchestrator becomes the single owner. the heavy ONNX work (and the
model-specific preprocessing) lives in the wrapper; the orchestrator
only forwards text and gets vectors back, so inter-process IO stays
tiny (no giant float tensors cross the pipe).

## 4.2 request -> vector pipeline

text(s) in -> vectors out. the orchestrator owns model selection,
batching, caching and the external protocol; the **wrapper owns
tokenization, pooling and normalize**. step by step:

1. text preprocessing decision: lowercase? (per `opts.preprocess`) — the
   choice is forwarded to the wrapper; the orchestrator does not tokenize.
2. batching: group texts into batches up to `opts.batch_size`.
3. call wrapper `embed` with the batch of texts + opts (`preprocess`,
   `normalize`, `max_context_window`). the wrapper tokenizes, truncates
   each text to `max_context_window` (token-level), runs ONNX,
   mean-pools over tokens, and L2-normalizes, returning one f32 vector
   per text.
4. output shape: `[n_texts, dim]` f32 (dim from `models`, e.g. 768),
   already normalized unless `opts.normalize=false`.

## 4.2b truncation (centralized)

bench4 currently hard-codes context truncation per method. modelcaller
owns it here instead:

- the orchestrator passes each model's `max_context_window` (from
  `models`) to the wrapper as an opt; the wrapper applies **token-level**
  truncation before running. for nomic-embed, truncate per-text; for
  reranker docs, see `05-inference.md` (bench4 used
  `MAX_RERANK_CHARS=12000`).
- truncation happens in one place so ik5 + bench4 get identical
  behavior (fixes the DRY gap from start.md problem #4).

## 4.3 dimensionality

- fixed per model: e.g. nomic = 768.
- mismatch handling: error if a wrapper response's dim != expected.

## 4.4 caching boundary

the embedding cache (`09-caching.md`) is keyed by model id + normalized
text + preprocessing + opts hash, and sits **in the orchestrator**,
before the wrapper call. a cache hit skips the wrapper entirely.

## 4.5 example

```json
{ "cmd":"embed","model":"nomic-embed-text-v1","texts":["rust dev","python dev"],"opts":{"normalize":true} }
```
- orchestrator forwards to wrapper `embed`; receives
  `[[...768...],[...768...]]` L2-normalized.
- concrete: 2 input texts, 768-dim output. each vector is 3072 bytes (768 × f32).
  wrapper returns both vectors as a flat JSON array of arrays.
