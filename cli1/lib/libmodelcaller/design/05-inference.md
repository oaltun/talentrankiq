# step 5 — inference (non-embedding)

## depends on

- `03-models.md` (reranker / generative models)
- `02-interface.md` (`rerank`, `generate` commands)
- `../onnxwrapperrs/design/01-wrapper.md` (wrapper owns tokenization + scoring)

## 5.1 rerank (cross-encoder)

- input: query + N docs — paired internally into (query, doc) pairs;
  one score per doc, order-aligned to input `docs`.
- model: bge-reranker-v2-m3 / jina-reranker-v1-tiny-en (see
  `03-models.md`); single relevance-score logit per pair.
- output: one f32 score per doc, higher = more relevant.
- batching: run pairs in batches up to `opts.batch_size`.
- the **orchestrator forwards `query` + `docs` text to the wrapper's
  `rerank` command**; the wrapper tokenizes each pair, runs the
  cross-encoder, and returns the score. the orchestrator does no tensor
  work — only pairing and score ordering.
- truncation: each `doc` is truncated to the reranker's
  `max_context_window` (passed to the wrapper) before scoring. bench4
  hard-codes `MAX_RERANK_CHARS = 12000`; modelcaller centralizes this
  (fixes the DRY gap from start.md problem #4). query is also truncated.

## 5.2 generate (optional)

only if a generative model is needed (e.g. rubric scoring).
- input: `prompt` (string), `max_tokens` (int, optional, default 256).
- output: `text` (string) — the generated completion.
- streaming: not supported in v1. the full response is returned in a
  single envelope. streaming would require a new envelope type
  (`"type":"chunk"`) and is deferred to a future protocol revision.
  reason: the only current consumer (ik5 rubric scoring) needs the
  full response, and streaming adds complexity (partial JSON, error
  mid-stream) without benefit for short rubric prompts.

## 5.3 relationship to embeddings

why rerank is separate from embed: different model class (cross-encoder
vs bi-encoder), no vector reuse, different output shape (scalar score
vs vector). both delegate the actual compute to the wrapper; the
orchestrator only handles text, batching, caching and the external
protocol.

## 5.4 example

```json
{ "cmd":"rerank","model":"bge-reranker-v2-m3","query":"rust dev","docs":["cv A","cv B"] }
```
- output: `{"model":"bge-reranker-v2-m3","ep":"cuda","scores":[0.91,0.33]}`
- 2 docs -> 2 scores, order-aligned. score 0.91 means cv A is more
  relevant than cv B (0.33) to the query "rust dev".
