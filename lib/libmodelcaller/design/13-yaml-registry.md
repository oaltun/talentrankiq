# step 3a — YAML model registry

## depends on

- `03-models.md` (supersedes the hardcoded `WELL_KNOWN` table in section 3.5)
- `06-cli.md` (new `registry` subcommand, `--registry` flag)
- `08-config.md` (registry path is a CLI arg, not env var)

## problem

modelcaller's model catalog is a hardcoded `const WELL_KNOWN` array in Rust.
adding a model requires recompilation. download source info is duplicated in
benchdriverrs' `MODEL_HF_REPO` table. usage patterns (instruction prefixes,
pooling) are not tracked, so texts are embedded raw regardless of model
requirements.

## solution

a single `models.yaml` file is the **only** source of truth. no built-in
defaults. every model, family, and usage pattern is explicit.

## 3a.1 registry file location

passed via `--registry <path>` to modelcaller. for now it lives in the
dev directory alongside the projects. no env var fallback.

## 3a.2 schema

```yaml
families:
  - family: nomic-embed-text-v1
    usage: nomic-prefix
  - family: bge-small-en-v1.5
    usage: bge-prefix
  - family: multilingual-e5-large
    usage: e5-prefix
  - family: harrier-oss-v1
    usage: harrier-instruct
  - family: bge-reranker-v2-m3
    usage: plain-pair
  - family: jina-reranker-v1-tiny-en
    usage: plain-pair
  - family: granite-embedding-311m-multilingual-r2
    usage: plain

models:
  - slug: nomic-embed-text-v1
    family: nomic-embed-text-v1
    kind: embedder
    dim: 768
    max_context_window: 8192
    onnx_url: https://huggingface.co/nomic-ai/nomic-embed-text-v1/resolve/main/onnx/model.onnx
    # optional `extra_files:` — repo-relative files fetched alongside the
    # onnx with paths preserved (passed to modeldownloader --extra-file by
    # auto-download). used for weights the wrapper applies itself, e.g.
    # ColBERT projection safetensors (1_Dense/model.safetensors, ...).
    # optional `multi_vector: true` — late-interaction (ColBERT-style)
    # model: `embed` returns one vector per token (dim = per-token size)
    # instead of one pooled vector; onnxwrapperrs skips pooling and applies
    # the projection chain from `extra_files` post-forward. embed responses
    # carry `multi_vector: true`, `counts`, `embeddings_b64`. see
    # bench/bench1/design/17-late-interaction.md.
  - slug: bge-small-en-v1.5
    family: bge-small-en-v1.5
    kind: embedder
    dim: 384
    max_context_window: 512
    onnx_url: https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx
  - slug: multilingual-e5-large_model_qint8_avx512_vnni
    family: multilingual-e5-large
    kind: embedder
    dim: 1024
    max_context_window: 512
    onnx_url: https://huggingface.co/Xenova/multilingual-e5-large/resolve/main/onnx/model_quantized.onnx
  - slug: harrier-oss-v1-0.6b-ONNX-q4
    family: harrier-oss-v1
    kind: embedder
    dim: 1024
    max_context_window: 32768
    onnx_url: https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_q4.onnx
  - slug: harrier-oss-v1-270m-q4f16
    family: harrier-oss-v1
    kind: embedder
    dim: 640
    max_context_window: 32768
    onnx_url: https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx
  - slug: bge-reranker-v2-m3
    family: bge-reranker-v2-m3
    kind: reranker
    dim: 0
    max_context_window: 512
    onnx_url: https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/onnx/model.onnx
  - slug: jina-reranker-v1-tiny-en
    family: jina-reranker-v1-tiny-en
    kind: reranker
    dim: 0
    max_context_window: 8192
    onnx_url: https://huggingface.co/cross-encoder/jina-reranker-v1-tiny-en/resolve/main/onnx/model.onnx
```

## 3a.3 usage patterns

a `usage` string names a code adapter in modelcaller that transforms text
before tokenization. each adapter has two methods:

- `format_query(text) -> String`
- `format_document(text) -> String`

for rerankers, a single `format_pair(query, doc) -> (String, String)`.

implemented as a `match` in modelcaller. if a family's usage is not recognized,
modelcaller errors with `unknown_usage`. we add adapters one at a time.

| usage id           | embedder query prefix                                       | embedder doc prefix | reranker
|--------------------|-------------------------------------------------------------|--------------------| --------
| `plain`            | none                                                         | none               | raw (q, d) pair
| `nomic-prefix`     | `search_query: `                                              | `search_document: `| N/A
| `bge-prefix`       | `Represent this sentence for searching relevant passages: `  | none               | N/A
| `e5-prefix`        | `query: `                                                    | `passage: `         | N/A
| `harrier-instruct` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` | none | N/A
| `colbert-prefix`   | `[Q] `                                                       | `[D] `              | N/A
| `plain-pair`       | N/A                                                          | N/A                | raw (q, d) pair

notes:
- `plain` = no transformation at all. suitable for models that don't
  need prefixes, or when the caller has already formatted the text.
- `nomic-prefix` and `harrier-instruct` are mandatory — performance
  degrades significantly without the prefix.
- `bge-prefix` is optional for v1.5 (slight improvement with it), but
  the adapter applies it consistently.
- `plain-pair` = rerankers that take raw (query, document) as input.
- `colbert-prefix` = PyLate/ColBERT marker tokens (`[Q]`, `[D]`) added
  as text; the tokenizer maps them to the model's special prefix ids.
  mandatory for PyLate-trained models (e.g. lightonai/mLateOn).

## 3a.4 auto-download

modelcaller gains `--auto-download-models`. when set:

1. on `load <slug>`: if the model is not found in `--model-dir`, modelcaller
   invokes `modeldownloader --model <slug> --source <onnx_url> --model-dir <model-dir>`
   as a child process, then retries the load. any `extra_files:` on the model
   entry are passed as repeatable `--extra-file <path>` arguments.
2. the `onnx_url` from `models.yaml` is passed as `--source` to modeldownloader.
   modeldownloader's `--source` already supports direct URLs.
3. if `--auto-download-models` is off (default for now), a missing model
   returns the usual `no_model` error.

modelcaller learns the modeldownloader binary path via a new `--modeldownloader`
arg (compulsory, no default — same convention as `--wrapper`).

## 3a.5 CLI changes

### new flag: `--registry <path>` (compulsory)

path to the `models.yaml` file. required for stdio mode.

### new flag: `--auto-download-models`

when set, auto-invoke modeldownloader for missing models.

### new flag: `--modeldownloader <path>`

path to the modeldownloader binary. required when `--auto-download-models`
is set.

### new subcommand: `modelcaller registry`

operates on the yaml file directly (no stdio mode needed):

```
modelcaller --registry models.yaml registry list
modelcaller --registry models.yaml registry add --slug <slug> --family <family> --kind embedder --dim 768 --max-ctx 8192 --onnx-url <url>
modelcaller --registry models.yaml registry remove <slug>
```

`list` prints a table. `add` appends to the file. `remove` deletes the
entry with the given slug.

## 3a.6 changes to other projects

### benchdriverrs
- remove the hardcoded `MODEL_HF_REPO` table.
- read model slugs from `models.yaml` (or from its config's `embed_models`
  list, which now must match slugs in `models.yaml`).
- pass `--registry models.yaml` to bench (which passes it to modelcaller).

### bench
- pass `--registry` through to modelcaller.

## 3a.7 migration

1. create `models.yaml` in the workspace root.
2. modelcaller reads it instead of the hardcoded `WELL_KNOWN`.
3. remove `WELL_KNOWN` const from `model_registry.rs`.
4. add usage adapters in modelcaller's embed/rerank path.
5. add `--auto-download-models` and `--modeldownloader` args.
6. add `registry` subcommand.
7. update benchdriverrs to pass `--registry` and remove `MODEL_HF_REPO`.

## 3a.8 generation kinds (step 18 m3 — designed in 14-evidence-extraction.md)

step 18 (`bench/bench1/design/18-structured-evidence.md` §18.4) adds two
generation kinds to this registry, same entry shape (`slug`, `family`,
`kind`, `onnx_url`, …):

- `kind: schema_discoverer` (`usage: extraction-json`) — job ad →
  typed `ExtractionSchema` field list; runs once per job ad.
- `kind: extractor` (`usage: extraction-json`) — cv + schema →
  extensions via constrained decoding in onnxwrapperrs.

the default model for both is a small quantized instruct model
(qwen2.5-1.5b-instruct-onnx-q4 working target). modelcaller grows
`cmd: discover_schema` and `cmd: extract_constrained` alongside them.

the concrete registration (family `qwen-instruct`, usage
`extraction-json`, onnx url, dim 0), the command payloads, the prompt
ownership rule, and the schema/evidence caches are specified in
`14-evidence-extraction.md`; the wrapper side in
`../onnxwrapperrs/design/04-generation.md`.