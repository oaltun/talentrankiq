#[derive(Debug, Clone)]
pub struct HfBlob {
    pub repo: String,
    pub revision: String,
    pub path: String,
}

impl HfBlob {
    /// The directory (within the repo) containing the target file.
    /// e.g. for `onnx/model_fp16.onnx` this is `onnx`.
    pub fn dir(&self) -> String {
        match self.path.rfind('/') {
            Some(i) => self.path[..i].to_string(),
            None => String::new(),
        }
    }

    /// The bare filename, e.g. `model_fp16.onnx`.
    pub fn filename(&self) -> String {
        match self.path.rfind('/') {
            Some(i) => self.path[i + 1..].to_string(),
            None => self.path.clone(),
        }
    }

    /// The file stem (no extension), e.g. `model_fp16`.
    pub fn stem(&self) -> String {
        let name = self.filename();
        match name.rsplit_once(".onnx") {
            Some((stem, _)) => stem.to_string(),
            None => name,
        }
    }
}

/// Parse a HuggingFace URL of the form:
///   https://huggingface.co/<org>/<repo>/blob/<revision>/<path>
///   https://huggingface.co/<org>/<repo>/resolve/<revision>/<path>
pub fn parse_hf_blob_url(url: &str) -> Option<HfBlob> {
    // strip scheme if present
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);

    // drop query / fragment
    let rest = rest.split(['?', '#']).next().unwrap_or(rest);

    // expect `huggingface.co/<org>/<repo>/{blob,resolve}/<revision>/<path...>`
    let after_host = rest.strip_prefix("huggingface.co/")?;
    let parts: Vec<&str> = after_host.split('/').collect();
    // parts: [org, repo, "blob"|"resolve", revision, path...]
    if parts.len() < 5 {
        return None;
    }
    if parts[2] != "blob" && parts[2] != "resolve" {
        return None;
    }
    let org = parts[0];
    let repo = parts[1];
    if org.is_empty() || repo.is_empty() || repo.contains('/') {
        return None;
    }
    let revision = parts[3].to_string();
    let path = parts[4..].join("/");
    if path.is_empty() || !path.ends_with(".onnx") {
        return None;
    }
    Some(HfBlob {
        repo: format!("{org}/{repo}"),
        revision,
        path,
    })
}

/// Build the list of repo-relative file paths to download for a given target
/// ONNX file, given the repo tree (list of all file paths).
///
/// We pull:
///  - the target ONNX file itself
///  - any sibling `_data*` files (same directory, same stem, e.g.
///    `model_fp16.onnx_data`, `model_fp16.onnx_data_1`)
///  - `tokenizer.json` and `config.json` if present at the repo root
///
/// The root is the HF convention, but not a universal one: a sentence-transformers
/// model that wraps its weights in a module directory (e.g. Model2Vec's
/// `0_StaticEmbedding/tokenizer.json`) keeps them there instead. So when a
/// root-level file is missing we fall back to the shallowest match in a
/// subdirectory; `remap_dest` puts it next to the ONNX either way.
pub fn plan_downloads(blob: &HfBlob, tree: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let dir = blob.dir();
    let stem = blob.stem();

    // target onnx
    out.push(blob.path.clone());

    let data_stem = format!("{stem}.onnx_data");
    for f in tree {
        if f == &blob.path {
            continue;
        }
        // sibling external-data files: same dir, filename starts with "<stem>.onnx_data"
        let name = match f.rsplit_once('/') {
            Some((file_dir, file_name)) => {
                if file_dir != dir {
                    continue;
                }
                file_name
            }
            None => {
                if !dir.is_empty() {
                    continue;
                }
                f.as_str()
            }
        };
        if name == data_stem.as_str() || name.starts_with(&format!("{data_stem}_")) {
            if !out.contains(f) {
                out.push(f.clone());
            }
        }
    }

    // tokenizer + config: source paths are at the repo root (HF convention).
    // execute_url remaps their local destination to sit next to the onnx file
    // so the wrapper (which loads tokenizer.json from the model directory) finds
    // them. A module-subdirectory copy is used as the fallback.
    for root_file in ["tokenizer.json", "token.json", "config.json"] {
        if let Some(found) = find_support_file(root_file, tree) {
            if !out.contains(&found) {
                out.push(found);
            }
        }
    }

    out
}

/// The repo-relative path of `root_file`: the root-level one if present,
/// otherwise the shallowest match in a subdirectory (a sentence-transformers
/// module directory). `None` when the repo has no such file at all.
fn find_support_file(root_file: &str, tree: &[String]) -> Option<String> {
    if tree.iter().any(|t| t == root_file) {
        return Some(root_file.to_string());
    }
    tree.iter()
        .filter_map(|f| {
            let (d, name) = f.rsplit_once('/')?;
            (name == root_file).then_some((d.matches('/').count(), f.clone()))
        })
        .min_by_key(|(depth, _)| *depth)
        .map(|(_, f)| f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let b = parse_hf_blob_url(
            "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/blob/main/onnx/model_fp16.onnx",
        )
        .unwrap();
        assert_eq!(b.repo, "onnx-community/harrier-oss-v1-0.6b-ONNX");
        assert_eq!(b.revision, "main");
        assert_eq!(b.path, "onnx/model_fp16.onnx");
        assert_eq!(b.dir(), "onnx");
        assert_eq!(b.filename(), "model_fp16.onnx");
        assert_eq!(b.stem(), "model_fp16");
    }

    #[test]
    fn parse_root_path() {
        let b = parse_hf_blob_url(
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1-onnx/blob/main/model.onnx",
        )
        .unwrap();
        assert_eq!(b.repo, "nomic-ai/nomic-embed-text-v1-onnx");
        assert_eq!(b.path, "model.onnx");
        assert_eq!(b.dir(), "");
        assert_eq!(b.stem(), "model");
    }

    #[test]
    fn parse_rejects_non_blob() {
        assert!(parse_hf_blob_url("https://example.com/foo/bar/blob/main/model.onnx").is_none());
        assert!(parse_hf_blob_url("not a url").is_none());
    }

    #[test]
    fn parse_accepts_resolve_url() {
        let b = parse_hf_blob_url(
            "https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX/resolve/main/onnx/model_q4f16.onnx",
        )
        .unwrap();
        assert_eq!(b.repo, "onnx-community/harrier-oss-v1-270m-ONNX");
        assert_eq!(b.revision, "main");
        assert_eq!(b.path, "onnx/model_q4f16.onnx");
        assert_eq!(b.stem(), "model_q4f16");
    }

    #[test]
    fn plan_finds_siblings_and_root() {
        let b = HfBlob {
            repo: "onnx-community/harrier-oss-v1-0.6b-ONNX".to_string(),
            revision: "main".to_string(),
            path: "onnx/model_fp16.onnx".to_string(),
        };
        let tree = vec![
            "onnx/model_fp16.onnx".to_string(),
            "onnx/model_fp16.onnx_data".to_string(),
            "onnx/model_fp16.onnx_data_1".to_string(),
            "onnx/model_q4.onnx".to_string(),
            "tokenizer.json".to_string(),
            "config.json".to_string(),
            "README.md".to_string(),
        ];
        let plan = plan_downloads(&b, &tree);
        assert!(plan.contains(&"onnx/model_fp16.onnx".to_string()));
        assert!(plan.contains(&"onnx/model_fp16.onnx_data".to_string()));
        assert!(plan.contains(&"onnx/model_fp16.onnx_data_1".to_string()));
        assert!(plan.contains(&"tokenizer.json".to_string()));
        assert!(plan.contains(&"config.json".to_string()));
        // must NOT include the q4 variant
        assert!(!plan.contains(&"onnx/model_q4.onnx".to_string()));
        // must NOT include unrelated files
        assert!(!plan.contains(&"README.md".to_string()));
    }

    #[test]
    fn plan_falls_back_to_module_subdir_tokenizer() {
        // A Model2Vec-style repo: no root config.json, tokenizer in a module dir.
        let b = HfBlob {
            repo: "sentence-transformers/static-similarity-mrl-multilingual-v1".to_string(),
            revision: "main".to_string(),
            path: "onnx/model_int8.onnx".to_string(),
        };
        let tree = vec![
            "0_StaticEmbedding/model.safetensors".to_string(),
            "0_StaticEmbedding/tokenizer.json".to_string(),
            "README.md".to_string(),
            "modules.json".to_string(),
            "onnx/model.onnx".to_string(),
            "onnx/model_int8.onnx".to_string(),
        ];
        let plan = plan_downloads(&b, &tree);
        assert!(plan.contains(&"onnx/model_int8.onnx".to_string()));
        assert!(
            plan.contains(&"0_StaticEmbedding/tokenizer.json".to_string()),
            "must find the tokenizer in the module dir"
        );
        assert!(
            !plan.contains(&"0_StaticEmbedding/model.safetensors".to_string()),
            "weights are not needed: the ONNX already carries them"
        );
    }

    #[test]
    fn root_support_file_wins_over_subdir() {
        let b = HfBlob {
            repo: "org/model".to_string(),
            revision: "main".to_string(),
            path: "onnx/model.onnx".to_string(),
        };
        let tree = vec![
            "tokenizer.json".to_string(),
            "0_StaticEmbedding/tokenizer.json".to_string(),
            "onnx/model.onnx".to_string(),
        ];
        let plan = plan_downloads(&b, &tree);
        assert!(plan.contains(&"tokenizer.json".to_string()));
        assert!(!plan.contains(&"0_StaticEmbedding/tokenizer.json".to_string()));
    }

    #[test]
    fn shallowest_subdir_wins() {
        let b = HfBlob {
            repo: "org/model".to_string(),
            revision: "main".to_string(),
            path: "model.onnx".to_string(),
        };
        let tree = vec![
            "a/b/c/tokenizer.json".to_string(),
            "a/tokenizer.json".to_string(),
            "model.onnx".to_string(),
        ];
        let plan = plan_downloads(&b, &tree);
        assert!(plan.contains(&"a/tokenizer.json".to_string()));
        assert!(!plan.contains(&"a/b/c/tokenizer.json".to_string()));
    }

    #[test]
    fn missing_support_files_are_simply_absent() {
        let b = HfBlob {
            repo: "org/model".to_string(),
            revision: "main".to_string(),
            path: "onnx/model.onnx".to_string(),
        };
        let plan = plan_downloads(&b, &["onnx/model.onnx".to_string()]);
        assert_eq!(plan, vec!["onnx/model.onnx".to_string()]);
    }
}
