# step 3b — shared config (libconfigreader)

## depends on

- `libmodelstorage/` (path resolution, used for default model_dir)
- `models.yaml` (model registry — moved into the shared config)

## problem

5 CLIs share several concepts (`model_dir`, `registry`, `verbose`,
`logfile`, `revision`) but each defines them independently with
inconsistent types (`PathBuf` vs `String`). benchdriverrs uses a
hand-rolled YAML parser. there is no validation that config values
agree across tools. a typo in one tool's `--model-dir` causes a silent
mismatch.

furthermore, `models.yaml` lives at the workspace root, unowned by
any single crate. it should be part of the config system.

## solution

a single `config.yaml` that all CLIs can read. a shared crate
`libconfigreader` parses it, validates it, and provides typed accessors.

### config file location

passed via `--config <path>` to each CLI (compulsory per AGENTS.md).
no env var fallback, no default path.

### schema

```yaml
# general section — read by all CLIs
model_dir: /Users/apple1/.cache/modelcaller
registry: ./models.yaml
verbose: false
logfile: null
revision: main

# binary paths — read by CLIs that spawn other CLIs
binaries:
  modelcaller: /path/to/modelcaller/target/release/modelcaller
  modeldownloader: /path/to/modeldownloader/target/release/modeldownloader
  onnxwrapperrs: /path/to/onnxwrapperrs/target/release/onnxwrapperrs
  bench: /path/to/bench/bench1/target/release/bench1

# model registry — read by modelcaller and benchdriverrs
# (the contents of models.yaml move here)
models:
  - slug: nomic-embed-text-v1
    family: nomic-embed-text-v1
    kind: embedder
    dim: 768
    max_context_window: 8192
    source: hf:nomic-ai/nomic-embed-text-v1
    onnx_url: https://huggingface.co/nomic-ai/nomic-embed-text-v1/resolve/main/onnx/model.onnx
    # optional: repo-relative files downloaded alongside the onnx with paths
    # preserved (modelcaller auto-download, shortlist-cli and benchdriverrs
    # pass them to modeldownloader --extra-file). used for weights the
    # runtime applies itself, e.g. ColBERT projection safetensors.
    # extra_files:
    #   - 1_Dense/model.safetensors
    # optional `multi_vector: true` — late-interaction (ColBERT-style)
    # model: embed returns one vector per token (dim = per-token size);
    # onnxwrapperrs skips pooling and (if projection files are given)
    # applies the projection chain. see bench/bench1/design/17-late-interaction.md.
    # multi_vector: true
  ...

families:
  - family: nomic-embed-text-v1
    usage: nomic-prefix
  ...

# bench-specific section — only bench and benchdriverrs read these
bench:
  dataset: /path/to/test.csv
  task: cnam-test
  results_dir: bench/results
  cache_dir: /Users/apple1/.cache/ik/modelcaller/embeddings
  baselines:
    - random
    - keyword-tfidf
    - bm25
  embed_models:
    - multilingual-e5-large_model_qint8_avx512_vnni
    - harrier-oss-v1-0.6b-ONNX-q4
  timezone: Europe/Istanbul
```

### sections

| section | readers | required | notes |
|---------|---------|---------|-------|
| (root) | all CLIs | yes | `model_dir`, `registry`, `verbose`, `logfile`, `revision` |
| `binaries` | all CLIs that spawn children | no | binary paths; modelcaller needs modeldownloader, bench needs modelcaller+wrapper |
| `models` | modelcaller, benchdriverrs | yes | the current models.yaml content |
| `families` | modelcaller, benchdriverrs | yes | usage patterns |
| `bench` | bench, benchdriverrs | no (only if present) | dataset, task, baselines, etc. |

### CLI arg vs config conflict rule

if a CLI arg is set explicitly, it must match the config value.
if both are set and disagree, the CLI errors out:

```
error: --model-dir '/a' conflicts with config value '/b'
```

if only one is set, that value is used. if neither is set,
the field uses its hard-coded default (if any) or the config value.

### what libconfigreader provides

```rust
pub struct Config {
    pub model_dir: PathBuf,        // from root or libmodelstorage default
    pub registry: PathBuf,         // from root (required)
    pub verbose: bool,             // from root
    pub logfile: Option<PathBuf>, // from root
    pub revision: String,          // from root, default "main"
    pub binaries: BinaryPaths,    // from binaries section
    pub models: Vec<ModelDef>,     // from models section
    pub families: Vec<FamilyDef>, // from families section
    pub bench: Option<BenchSection>, // from bench section
}

pub fn load(path: &Path, cli_overrides: &CliOverrides) -> Result<Config, ConfigError>;
```

`CliOverrides` is a struct of `Option<T>` for each general field.
`load` parses the YAML, applies CLI overrides (with conflict
detection), validates required fields, and returns the typed config.

### path resolution

`load` normalizes every path-valued key against the config file's own
directory, so a config that travels between machines (or is read from
a tool whose cwd is somewhere else entirely, e.g. a spawned release
binary) resolves the same way everywhere:

- a leading `~` expands to the home directory (`gui_cv_dir_root`,
  `bench.results_dir`, `bench.cache_dir`)
- a relative path joins onto the config file's directory (`model_dir`,
  `logfile`, `binaries.*`, `gui_cv_dir_root`, `gui_judge_artifact.*`,
  `bench.results_dir`, `bench.cache_dir`)
- everything else is kept verbatim (absolute paths)

`bench.results_dir` is the committed result corpus, so it is
conventionally relative (`bench/results`) and lands inside the repo.
`bench.cache_dir` and `model_dir` are reproducible/large, so they stay
absolute and out of the repo.

### what stays in each CLI

each CLI still has its own clap args for CLI-specific options.
only shared/general args come from the config. CLIs call
`libconfigreader::load` with their clap values as overrides.

for example, bench's `--method` and `--seed` stay as clap args.
modeldownloader's `--url` and `--force` stay as clap args.

### validation rules

1. `registry` must exist and be parseable (delegate to modelcaller's
   yaml parser)
2. all `embed_models` slugs in `bench` section must exist in `models`
3. all `binaries` paths that the CLI needs must be set
4. `model_dir` defaults to `libmodelstorage::default_model_dir()` if not set
5. unknown sections are allowed (forward compatibility)
6. unknown keys within known sections are rejected

### migration

1. create `libconfigreader/` crate with the YAML parser and validator
2. merge `models.yaml` into `config.yaml` (keep `models.yaml` as a
   symlink or alias for backward compat during transition)
3. each CLI adds `--config` arg, calls `libconfigreader::load` at startup
4. remove duplicated arg definitions from each CLI's clap struct
5. benchdriverrs deletes its hand-rolled YAML parser
6. remove `--registry` from bench (comes from config)
7. remove `--model-dir` from modelcaller/bench/modeldownloader (comes from config)
