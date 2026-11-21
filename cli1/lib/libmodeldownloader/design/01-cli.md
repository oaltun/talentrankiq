# CLI

## depends on

- `00-overview.md` (purpose, cache layout, sources)
- `../modelcaller/design/03-models.md` 3.3 (the directory layout written)

## 1.1 binary

- name: `modeldownloader`.
- mode: run once, exit. no server, no daemon, no stdin protocol.
- short-lived: downloads files, writes them to the cache, exits.

## 1.2 flags

| flag | meaning | default |
| --- | --- | --- |
| `--model <id>` | logical model id (e.g. `nomic-embed-text-v1`). use with `--source`. | **required**\* |
| `--source <src>` | where to fetch: `hf:<org>/<repo>`, `https://...`, `file://...`. use with `--model`. | resolved from config |
| `--url <url>` | full HuggingFace blob url to a single ONNX file. use with `--slug`. e.g. `https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/blob/main/onnx/model_fp16.onnx` | — |
| `--slug <slug>` | directory slug used as the model id when `--url` is given. files written to `<model-dir>/<slug>/repo/`. | — |
| `--revision <tag>` | HF revision (tag/branch/commit) | `main` |
| `--model-dir <path>` | cache root | `~/.cache/modelcaller` |
| `--force` | re-download even if all files exist and match size | off |
| `--dry-run` | show what would be downloaded without downloading | off |
| `--files <list>` | specific files to fetch (e.g. `model.onnx,tokenizer.json,config.json`). fetch all known files if omitted | all |
| `--extra-file <path>` | additional repo-relative file fetched alongside the onnx, written into the same model dir with the repo path preserved (repeatable). e.g. `1_Dense/model.safetensors` ColBERT projection weights the runtime applies post-forward. a missing extra file is an error. | none |
| `--version`, `--help` | standard | — |

\* either (`--model` + `--source`) **or** (`--url` + `--slug`) is required.

## 1.2.1 url mode (`--url` + `--slug`)

many HuggingFace ONNX repos keep the artifact under an `onnx/`
subdirectory and offer several quantizations. to pin one quantization to
one logical id without ambiguity:

1. pass the full blob url of the target ONNX, e.g.
   `https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/blob/main/onnx/model_fp16.onnx`.
2. pass a `--slug` that becomes the model id, e.g.
   `harrier-oss-v1-0.6b-ONNX-fp16`.

behavior:

- the url is parsed into `repo` + `revision` + file path. the HF tree
  API lists the repo; the download set is computed automatically:
  - the target ONNX itself,
  - any sibling `_data` external-weight files (`<stem>.onnx_data`,
    `<stem>.onnx_data_1`, …) in the same directory,
  - `tokenizer.json` / `config.json` from the repo root, **remapped to sit
    next to the `.onnx`** so the wrapper finds them.
- files are written into `<model-dir>/<slug>/repo/`, **preserving the
  repo's directory structure** (no flattening, no symlinks). ORT resolves
  external `_data` weights relative to the `.onnx`.
- `<model-dir>/<slug>/meta/source.json` records provenance:
  `source_url`, `repo`, `revision`, `onnx_path`, `files` (extras included).
- the model id for `modelcaller` / `bench` is exactly `<slug>`.

example:

```bash
modeldownloader \
  --url https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/blob/main/onnx/model_fp16.onnx \
  --slug harrier-oss-v1-0.6b-ONNX-fp16
# writes ~/.cache/modelcaller/harrier-oss-v1-0.6b-ONNX-fp16/{repo,meta}/...
# stdout: /Users/apple1/.cache/modelcaller/harrier-oss-v1-0.6b-ONNX-fp16
```

env vars (lower precedence than flags, higher than defaults):
`MODELDOWNLOADER_MODEL_DIR`, `HF_TOKEN` (HuggingFace auth).

## 1.3 behavior

1. resolve target directory: `<model-dir>/<id>/<revision>/`.
2. determine the list of files to fetch. default set:
   `model.onnx` (or `model.onnx.gz`), `tokenizer.json` (or
   `vocab.txt` + `merges.txt` / `tokenizer.model`), `config.json`.
   for `hf:` sources, list the repo's files and pick the ones matching
   these names. for `https://` / `file://`, list the source directory
   or extract the archive and match.
3. skip check (unless `--force`): if every target file already exists in
   the cache directory and its size matches the remote, skip download
   (cache hit). report which files were skipped.
4. download each missing file to a temp file in the target directory,
   then atomically rename. this prevents partial downloads from
   looking like valid cache entries.
5. verify: after all files are written, check that at least one ONNX
   file (`model.onnx` or `model.onnx.gz`) exists. if not, exit `5`.
6. print the target directory path to stdout (one line). print progress
   to stderr.

## 1.4 idempotency

running twice with the same args is a no-op (unless `--force`). the
skip check compares file sizes; for stronger validation, a future
version may compare SHA-256 but size is sufficient for the common case
(a failed partial download will have a wrong size and be re-fetched).

## 1.5 source details

### hf:<org>/<repo>
- uses the HuggingFace Hub file-download API. no git clone.
- respects `HF_TOKEN` / `huggingface-cli login` for private repos.
- for each file, requests the size header first, compares with local,
  skips if match (unless `--force`).
- HF repos are expected to be flat (files at root, no subdirectories).

### https://<url>
- if the URL is a zip/tarball, download and extract into the target
  directory.
- if the URL points to a directory listing (HTML with file links),
  parse it and download matching files.
- for a direct file URL, download the single file.

### file://<path>
- copy files from the local directory to the cache directory.
- useful for testing with modelconverter output without pushing to HF.

## 1.6 output

- **stdout**: exactly one line on success — the target directory path.
  e.g. `/Users/apple1/.cache/modelcaller/nomic-embed-text-v1/main`
- **stderr**: progress (file names, bytes downloaded, skip notices).
  callers must not parse stderr.

## 1.7 exit codes

| code | meaning |
| --- | --- |
| 0 | success (all files present, newly downloaded or cached) |
| 1 | network error (unreachable, timeout, auth failure) |
| 2 | bad arguments (missing `--model`, invalid `--source`) |
| 3 | source not found (HF repo 404, URL 404, file:// dir missing) |
| 4 | disk write error (permissions, disk full) |
| 5 | post-download verification failed (no ONNX file found) |

## 1.8 example

```bash
# first download
modeldownloader --model nomic-embed-text-v1 \
  --source hf:nomic-ai/nomic-embed-text-v1-onnx
# stderr: [info] downloading model.onnx (134 MiB)
# stderr: [info] downloading tokenizer.json (466 KiB)
# stderr: [info] downloading config.json (128 B)
# stdout:  /Users/apple1/.cache/modelcaller/nomic-embed-text-v1/main

# second run (cached)
modeldownloader --model nomic-embed-text-v1 \
  --source hf:nomic-ai/nomic-embed-text-v1-onnx
# stderr: [info] model.onnx cached (134 MiB)
# stderr: [info] tokenizer.json cached (466 KiB)
# stderr: [info] config.json cached (128 B)
# stdout:  /Users/apple1/.cache/modelcaller/nomic-embed-text-v1/main

# force re-download
modeldownloader --model nomic-embed-text-v1 --force \
  --source hf:nomic-ai/nomic-embed-text-v1-onnx --revision v2

# local file copy (dev)
modeldownloader --model test-embed \
  --source file:///tmp/baked-output/
```