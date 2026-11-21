# 04 — token generation (step 18 m3: cv-evidence extraction)

## depends on

- `01-wrapper.md` §1.6 (planned command set extension)
- `02-protocol.md` (envelope framing)
- `../bench/bench1/design/18-structured-evidence.md` §18.4 (the consumer: llm
  schema discovery + constrained extraction)
- `../libcvevidence/` (`ExtractionSchema` — the constraint source)

## 4.1 purpose

step 18's m3 milestone turns the wrapper from an encoder into a small
decoder host: a 1–4b quantized instruct model (qwen2.5-1.5b-instruct
q4 today) runs twice per extraction pipeline — schema discovery (job
ad → field list) and constrained fill (cv + schema → extensions json).
the wrapper stays a model server: **prompt in, json text out**. prompt
rendering (chat template), schema state machines for masking, retries,
caches, and validation all live in modelcaller / libcvparser.

## 4.2 deviation from 18 §18.4: kv-cache in v1

§18.4 said "no kv-cache in v1". that is **wrong for this model class**
and corrected here: qwen2's vocabulary is 151 936, so a full re-run at
a 2k-token context materializes a `[1, seq, vocab]` logits tensor —
~1.2 GB per decode step, ~500 steps per extraction. the HF exports
already expose `past_key_values` inputs / `present` outputs; feeding
them is the *simple* path (prefill once, then one new token per step,
logits `[1, 1, vocab]`). kv-cache in v1, not a later perf step.

## 4.3 session kind: `extractor`

`load` infers the session kind from the graph, as today:

| observable | kind | commands |
| --- | --- | --- |
| output named `logits` | `extractor` | `generate`, `generate_constrained` |
| inputs `query*` / seq-score output | `reranker` | `rerank` |
| otherwise | `embedder` | `embed` |

an `extractor` session rejects `embed`/`rerank` and vice versa
(`bad_kind`), like the existing kinds. `dim` is meaningless for a
decoder; `load` reports `dim: 0` and `max_context_window` comes from
the model dir's `config.json` (`max_position_embeddings`, patched by
modeldownloader registration if absent).

## 4.4 commands

### generate (greedy decode)

request:

```json
{"id":"g1","type":"request","payload":{"cmd":"generate","session":1,
 "prompt":"<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
 "opts":{"max_new_tokens":256,"stop_ids":[151645],"stop_strings":["<|im_end|>"]}}}
```

- `prompt` is the **fully rendered** prompt including chat markers —
  modelcaller owns template rendering (its registry `usage` adapters
  already own query/doc formatting; same ownership rule). the wrapper
  only tokenizes with `add_special_tokens:false`.
- loop: prefill the prompt (kv cached), then per step argmax the last
  logits row, append, re-feed only the new id + past kv.
- stop: any `stop_ids`, any generated token whose decoded text
  contains a `stop_strings` member, or `max_new_tokens` (default 256).
  the stop token itself is not included in the output.
- response:

```json
{"id":"g1","type":"response","payload":{"text":"...","n_new_tokens":87,"eos":true}}
```

### generate_constrained

same request shape plus a `schema` object — the `ExtractionSchema`
json from `libcvevidence` (`base`, `fields[{name, type, description,
required, enum}]`). the model does **structured filling, not free
generation**: the wrapper emits the json object skeleton itself and
lets the model generate only the value atoms, masking logits to the
tokens the state machine admits at each step.

state machine (deterministic, derived from the schema):

1. `{` `"` field-name `"` `:` — emitted verbatim by the machine, not
   the model (field order = schema order; required fields always
   emitted, optional fields may be skipped — the machine offers `}`
   as an alternative continuation after each optional field's comma).
2. value generation per field type:
   - `numeric`: mask to digits, `-`, `.`, then the closing
     `,`/`"`-continuation.
   - `bool`: mask to the token spellings of `true` / `false`.
   - `string`: free until the closing quote — any token whose text
     contains no `"`, `\`, or control character, or the closing
     `"` continuation.
   - `string` + `enum`: only token continuations inside the quoted
     enum literals (prefix-closed over `"<value>"`).
   - `list_string`: `[` then quoted strings separated by `,`, empty
     list allowed.
3. after the last field: `}` then the stop token.

masking = `-inf` on rejected logits before argmax (greedy, so no
sampling effects to reason about). a malformed continuation is
therefore structurally impossible; modelcaller's repair pass
(strip fences, truncate to last balanced brace) remains
belt-and-braces for the unconstrained path only.

response: same shape as `generate`; `text` is the filled json object.

## 4.5 performance envelope (v1, cpu)

prefill ~2k tokens + ~200–400 constrained steps on 1.5b q4:
order 5–20 s/cv single-threaded, better with xnnpack. per-job
(71 ads) and per-cv caches in modelcaller/bench make this a one-time
cost per (model, ad-schema, cv); no batching in v1 (batch 1 keeps the
loop simple; batched decode is a later perf step alongside continuous
batching if m3 throughput ever matters).

## 4.6 what stays out

registry, model selection, download, prompt templates, schema
discovery prompts, extraction prompts, validation, caching: all
modelcaller. the wrapper knows tokens and logits only.

## 4.7 testing

- unit: state-machine transitions per field type (feed token texts,
  assert admission sets); stop handling; mask shape.
- integration (`ONNXWRAPPERRS_TEST_MODEL` pointing at the qwen dir):
  `generate` returns coherent text for a one-line prompt;
  `generate_constrained` with a 3-field schema returns parseable json
  validating against `ExtractionSchema`.
