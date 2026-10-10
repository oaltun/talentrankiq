use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FamilyDef {
    pub family: String,
    pub usage: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelDef {
    pub slug: String,
    pub family: String,
    pub kind: String,
    pub dim: u32,
    pub max_context_window: u32,
    pub source: String,
    pub onnx_url: String,
    #[serde(default)]
    pub extra_files: Vec<String>,
    #[serde(default)]
    pub multi_vector: bool,
    /// The Metal-backend variant declared on this model in `ikapp.yaml`
    /// (`models[].mlx`). Mirrors `libconfigreader::ModelDef::mlx`.
    #[serde(default)]
    pub mlx: Option<crate::cli::MlxArtifact>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RegistryFile {
    pub families: Vec<FamilyDef>,
    pub models: Vec<ModelDef>,
}

pub struct Registry {
    models: Vec<ModelDef>,
    families: HashMap<String, String>,
}

impl Registry {
    pub fn from_config(cfg: &libconfigreader::Config) -> Self {
        let mut families = HashMap::new();
        for f in &cfg.families {
            families.insert(f.family.clone(), f.usage.clone());
        }
        let models = cfg
            .models
            .iter()
            .map(|m| ModelDef {
                slug: m.slug.clone(),
                family: m.family.clone(),
                kind: m.kind.clone(),
                dim: m.dim,
                max_context_window: m.max_context_window,
                source: m.source.clone(),
                onnx_url: m.onnx_url.clone(),
                extra_files: m.extra_files.clone(),
                multi_vector: m.multi_vector,
                mlx: m
                    .mlx
                    .as_ref()
                    .map(|v| crate::cli::MlxArtifact {
                        slug: v.slug.clone(),
                        source: v.source.clone(),
                    }),
            })
            .collect();
        Registry { models, families }
    }

    pub fn find(&self, id: &str) -> Option<ResolvedModel> {
        let m = self.models.iter().find(|m| m.slug == id)?;
        let usage = self.families.get(&m.family).cloned().unwrap_or_default();
        Some(ResolvedModel {
            slug: m.slug.clone(),
            kind: m.kind.clone(),
            dim: m.dim,
            max_context_window: m.max_context_window,
            source: m.source.clone(),
            onnx_url: m.onnx_url.clone(),
            extra_files: m.extra_files.clone(),
            multi_vector: m.multi_vector,
            family: m.family.clone(),
            usage,
            mlx: m.mlx.as_ref().map(|v| crate::cli::MlxArtifact {
                slug: v.slug.clone(),
                source: v.source.clone(),
            }),
        })
    }

    pub fn all(&self) -> &[ModelDef] {
        &self.models
    }

    pub fn remove(&mut self, slug: &str) -> bool {
        let len_before = self.models.len();
        self.models.retain(|m| m.slug != slug);
        self.models.len() != len_before
    }

    pub fn add(&mut self, model: ModelDef) {
        self.models.push(model);
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let families: Vec<FamilyDef> = self
            .families
            .iter()
            .map(|(family, usage)| FamilyDef {
                family: family.clone(),
                usage: usage.clone(),
            })
            .collect();
        let file = RegistryFile {
            families,
            models: self.models.clone(),
        };
        let yaml = serde_yaml::to_string(&file)
            .map_err(|e| format!("failed to serialize registry: {}", e))?;
        std::fs::write(path, yaml).map_err(|e| format!("failed to write registry: {}", e))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub slug: String,
    pub kind: String,
    pub dim: u32,
    pub max_context_window: u32,
    /// hf source (`hf:<org>/<repo>`) kept for registry parity with
    /// `ModelDef`; downloads go through `onnx_url` (URL mode) instead.
    #[allow(dead_code)]
    pub source: String,
    pub onnx_url: String,
    /// repo-relative files fetched alongside the onnx (projection weights)
    pub extra_files: Vec<String>,
    /// multi-vector (late interaction) model: embed returns token vectors
    pub multi_vector: bool,
    #[allow(dead_code)]
    pub family: String,
    pub usage: String,
    /// The Metal-backend variant of this model, when the config declares one
    /// (`ikapp.yaml` `models[].mlx`). `None` means this model has no MLX
    /// artifact, so the ladder keeps it on ONNX even where the arm exists —
    /// which is why the scout needs no port (design 15 §15.5.3).
    pub mlx: Option<crate::cli::MlxArtifact>,
}

pub fn format_query(text: &str, usage: &str) -> String {
    match usage {
        "nomic-prefix" => format!("search_query: {}", text),
        "bge-prefix" => format!("Represent this sentence for searching relevant passages: {}", text),
        "e5-prefix" => format!("query: {}", text),
        "harrier-instruct" => format!(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {}",
            text
        ),
        "colbert-prefix" => format!("[Q] {}", text),
        "plain" | "plain-pair" | _ => text.to_string(),
    }
}

pub fn format_document(text: &str, usage: &str) -> String {
    match usage {
        "nomic-prefix" => format!("search_document: {}", text),
        "e5-prefix" => format!("passage: {}", text),
        "colbert-prefix" => format!("[D] {}", text),
        _ => text.to_string(),
    }
}

pub fn unknown_usage_err(usage: &str) -> String {
    format!(
        "unknown usage pattern '{}' for model family; supported: plain, nomic-prefix, bge-prefix, e5-prefix, harrier-instruct, colbert-prefix, plain-pair, extraction-json",
        usage
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_registry() {
        let yaml = "families: []\nmodels: []\n";
        let reg: RegistryFile = serde_yaml::from_str(yaml).unwrap();
        assert!(reg.models.is_empty());
        assert!(reg.families.is_empty());
    }

    #[test]
    fn parse_full_registry() {
        let yaml = r#"
families:
  - family: test-fam
    usage: plain
models:
  - slug: test-model
    family: test-fam
    kind: embedder
    dim: 768
    max_context_window: 8192
    source: hf:example/test-model
    onnx_url: https://example.com/model.onnx
"#;
        let reg = Registry::load_from_str(yaml).unwrap();
        let m = reg.find("test-model").unwrap();
        assert_eq!(m.dim, 768);
        assert_eq!(m.usage, "plain");
    }

    #[test]
    fn find_unknown_is_none() {
        let reg = Registry::load_from_str("families: []\nmodels: []\n").unwrap();
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn usage_formatters() {
        assert_eq!(format_query("hello", "nomic-prefix"), "search_query: hello");
        assert_eq!(
            format_document("doc", "nomic-prefix"),
            "search_document: doc"
        );
        assert_eq!(format_query("hello", "e5-prefix"), "query: hello");
        assert_eq!(format_document("doc", "e5-prefix"), "passage: doc");
        assert_eq!(format_query("hello", "plain"), "hello");
        assert_eq!(format_document("doc", "plain"), "doc");
        assert_eq!(format_query("hello", "colbert-prefix"), "[Q] hello");
        assert_eq!(format_document("doc", "colbert-prefix"), "[D] doc");
    }

    impl Registry {
        pub fn load_from_str(s: &str) -> Result<Self, String> {
            let file: RegistryFile = serde_yaml::from_str(s).unwrap();
            let mut families = HashMap::new();
            for f in &file.families {
                families.insert(f.family.clone(), f.usage.clone());
            }
            Ok(Registry {
                models: file.models,
                families,
            })
        }
    }
}
