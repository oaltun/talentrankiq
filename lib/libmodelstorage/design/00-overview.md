# modelstorage — shared model cache contract

## problem

modelcaller (reader) and modeldownloader (writer) both implement
path construction, ONNX file scanning, and directory layout logic
independently. changing the layout in one breaks the other —
"integration decay".

## solution

a shared Rust crate `modelstorage` that both depend on. it is the
single source of truth for:

- **path construction**: where model files live on disk
- **ONNX file scanning**: finding `.onnx` files, excluding `*_data`
- **resolution order**: repo layout first, then legacy
- **directory naming**: `repo/`, `meta/`, `source.json`
- **model_dir resolution**: CLI arg > env var > default `~/.cache/modelcaller`

## API surface

```rust
// layout constants
pub const DEFAULT_REVISION: &str = "main";
pub const REPO_DIR: &str = "repo";
pub const META_DIR: &str = "meta";
pub const META_FILE: &str = "source.json";
pub const DEFAULT_FILES: &[&str] = &["model.onnx", "model.onnx.gz", "tokenizer.json", "config.json"];

// path construction
pub fn default_model_dir() -> PathBuf;          // ~/.cache/modelcaller
pub fn resolve_model_dir(cli_dir: Option<&Path>) -> PathBuf;  // CLI > env > default
pub fn slug_dir(model_dir: &Path, slug: &str) -> PathBuf;     // <model_dir>/<slug>
pub fn repo_dir(model_dir: &Path, slug: &str) -> PathBuf;     // <model_dir>/<slug>/repo
pub fn legacy_dir(model_dir: &Path, slug: &str, revision: &str) -> PathBuf;  // <model_dir>/<slug>/<revision>
pub fn meta_dir(model_dir: &Path, slug: &str) -> PathBuf;     // <model_dir>/<slug>/meta

// ONNX scanning
pub fn is_onnx_file(path: &Path) -> bool;         // .onnx && !*_data
pub fn scan_single_onnx(dir: &Path) -> Result<PathBuf, ResolveError>;  // exactly one .onnx
pub fn collect_onnx(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), ResolveError>;

// resolution (what modelcaller calls)
pub fn resolve_onnx(model_dir: &Path, slug: &str, revision: &str) -> Result<Resolution, ResolveError>;
pub fn onnx_exists(model_dir: &Path, slug: &str, revision: &str) -> bool;
pub fn local_size(path: &Path) -> Option<u64>;

// types
pub enum Resolution { Repo(PathBuf), Legacy(PathBuf) }
pub enum ResolveError { NotFound, Ambiguous(String), Io(String) }
```

## resolution order

`resolve_onnx` tries, in order:
1. `<model_dir>/<slug>/repo/` — scan recursively for exactly one `*.onnx` (excluding `*_data`)
2. `<model_dir>/<slug>/<revision>/` — same scan
3. `NotFound`

multiple `.onnx` files → `Ambiguous` error.

## what stays in each consumer

### modelstorage (this crate)
- path construction functions
- ONNX scanning logic
- resolution algorithm
- layout constants

### modeldownloader
- `Exit` error codes (its own concern)
- `verify_onnx_present` (flat `model.onnx`/`model.onnx.gz` check for legacy mode)
- HTTP download, HF API, extraction
- `DEFAULT_FILES` used for file:// source listing

### modelcaller
- JSONL protocol
- model registry (YAML)
- usage adapters (prefix formatting)
- wrapper spawning
- maps `ResolveError` → `ModelError`