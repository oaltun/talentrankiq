# modeldownloader — ONNX model fetcher

## purpose

a **small, short-lived CLI utility** that downloads ONNX model files
(model weights + tokenizer + config) from HuggingFace into the local
cache directory. it is the **one-off acquisition step** that runs before
modelcaller needs to serve a model. it is **shipped to customers** (ik5
calls it at install / first-run).

modelcaller does **not** download models itself — it expects them
already present in the cache. this separation keeps the long-lived
server free of network code and download state.

## invoked by

- **ik5** at install or first-run (production — ships to customers).
- **bench4** before a benchmark run that needs an ONNX model not yet
cached.
- **developers** manually when adding a new model to test.

## what it downloads

ONNX files only. the output is a **model directory** matching the
layout modelcaller expects (see `../modelcaller/design/03-models.md` 3.3):

```
<model-dir>/<id>/<revision>/
├── model.onnx            # or model.onnx.gz (kept as-is from HF)
├── tokenizer.json        # or vocab.txt / merges.txt / tokenizer.model
└── config.json           # { "kind", "dim", "max_context_window", ... }
```

the HF repo is expected to contain these files. modeldownloader
fetches them verbatim — no conversion, no validation beyond checking
that `model.onnx` (or `.onnx.gz`) exists after download.

## source

- **primary: HuggingFace** — `hf:<org>/<repo>` (e.g.
  `hf:nomic-ai/nomic-embed-text-v1-onnx`). uses the HuggingFace Hub API
to download files. no authentication required for public repos; honors
`HF_TOKEN` for private repos.
- **secondary: `https://` URL** — points to a zip/tarball or a directory
listing of ONNX files. for dev / air-gapped testing.
- **tertiary: `file://` path** — copies from a local directory. for
dev / testing with a locally-baked ONNX (from modelconverter output).

the source is specified per-invocation; there is no persistent registry
of sources. the caller (ik5, bench4) knows which model maps to which
HF repo.

## relationship to other projects

- **modelcaller** (`../modelcaller/design/`) — consumes the downloaded
  files. modeldownloader writes the cache layout; modelcaller reads it.
  no code dependency.
- **modelconverter** (`../modelconverter/design/`) — the dev-only GGUF→ONNX
  converter that *publishes* ONNX repos to HF. modelconverter produces;
  modeldownloader fetches. they are opposite ends of the same pipeline.
- **onnxbinrs** (`../onnxbinrs/design/`) — never interacts with
  modeldownloader directly. onnxbinrs loads ONNX files from the path
  modelcaller gives it.

## design steps (this project)

- `00-overview.md` — this file.
- `01-cli.md` — CLI flags, exit codes, download behavior, idempotency.
