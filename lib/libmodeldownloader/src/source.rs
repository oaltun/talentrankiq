use std::path::PathBuf;

#[derive(Debug)]
pub enum Source {
    HuggingFace { repo: String },
    Https { url: String },
    File { path: PathBuf },
}

#[derive(Debug)]
pub struct SourceParseError(pub String);

impl std::fmt::Display for SourceParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid --source: {}", self.0)
    }
}

impl std::error::Error for SourceParseError {}

pub fn parse_source(src: &str) -> Result<Source, SourceParseError> {
    if let Some(rest) = src.strip_prefix("hf:") {
        let rest = rest.trim();
        if !rest.contains('/') || rest.starts_with('/') || rest.ends_with('/') {
            return Err(SourceParseError(format!(
                "hf source must be <org>/<repo>, got '{rest}'"
            )));
        }
        Ok(Source::HuggingFace {
            repo: rest.to_string(),
        })
    } else if let Some(rest) = src.strip_prefix("https://") {
        Ok(Source::Https {
            url: format!("https://{rest}"),
        })
    } else if let Some(rest) = src.strip_prefix("http://") {
        Ok(Source::Https {
            url: format!("http://{rest}"),
        })
    } else if let Some(rest) = src.strip_prefix("file://") {
        let path = PathBuf::from(rest);
        if !path.is_absolute() {
            return Err(SourceParseError(format!(
                "file:// source must be an absolute path, got '{rest}'"
            )));
        }
        Ok(Source::File { path })
    } else {
        Err(SourceParseError(format!(
            "unknown source scheme; expected hf:<repo>, https://..., or file://... : got '{src}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf() {
        match parse_source("hf:nomic-ai/nomic-embed-text-v1-onnx") {
            Ok(Source::HuggingFace { repo }) => {
                assert_eq!(repo, "nomic-ai/nomic-embed-text-v1-onnx");
            }
            other => panic!("expected HuggingFace source, got {other:?}"),
        }
    }

    #[test]
    fn parse_hf_rejects_leading_slash() {
        assert!(parse_source("hf:/foo/bar").is_err());
        assert!(parse_source("hf:foo/bar/").is_err());
    }

    #[test]
    fn parse_https() {
        match parse_source("https://example.com/model.onnx") {
            Ok(Source::Https { url }) => assert_eq!(url, "https://example.com/model.onnx"),
            other => panic!("expected Https, got {other:?}"),
        }
    }

    #[test]
    fn parse_file() {
        match parse_source("file:///tmp/models/") {
            Ok(Source::File { path }) => assert_eq!(path, PathBuf::from("/tmp/models")),
            other => panic!("expected File, got {other:?}"),
        }
    }

    #[test]
    fn parse_file_requires_absolute() {
        assert!(parse_source("file://relative/path").is_err());
    }

    #[test]
    fn parse_unknown_scheme() {
        assert!(parse_source("ftp://example.com/x").is_err());
        assert!(parse_source("just-a-string").is_err());
    }
}
