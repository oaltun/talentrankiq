use std::path::Path;

pub use anydoc;

pub mod evidence;
pub use evidence::{
    contains_term, parse_evidence, parse_layout, DatePoint, DateRange, EvidenceOptions, ParsedCV,
    Section, SectionKind,
};

#[derive(Debug, thiserror::Error)]
pub enum CvParseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Document is encrypted or password-protected")]
    Encrypted,
    #[error("Document needs OCR (scanned/image-only pages): {pages:?}")]
    NeedsOcr { pages: Vec<u32> },
    #[error("Malformed document: {0}")]
    Malformed(String),
    #[error("Anydoc conversion failed: {0}")]
    Other(String),
}

fn map_anydoc_err(e: anydoc::ConvertError) -> CvParseError {
    match e {
        anydoc::ConvertError::Unsupported(msg) => CvParseError::UnsupportedFormat(msg),
        anydoc::ConvertError::Encrypted => CvParseError::Encrypted,
        anydoc::ConvertError::NeedsOcr { pages, .. } => CvParseError::NeedsOcr { pages },
        anydoc::ConvertError::Malformed { .. } => CvParseError::Malformed("malformed document".to_string()),
        other => CvParseError::Other(other.to_string()),
    }
}

fn strip_markdown_to_text(md: &str) -> String {
    let mut out = String::with_capacity(md.len());
    for line in md.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let text = trimmed.trim_start_matches('#').trim();
            if !text.is_empty() {
                out.push_str(text);
                out.push('\n');
            }
        } else if trimmed.starts_with('|') {
            let cells: Vec<String> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|cell| strip_inline_markdown(cell.trim()))
                .filter(|cell| {
                    !cell.is_empty()
                        && !cell.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
                })
                .collect();
            if !cells.is_empty() {
                out.push_str(&cells.join(" "));
                out.push('\n');
            }
        } else if trimmed.starts_with("```") {
            continue;
        } else if trimmed.starts_with('!') && trimmed.contains('[') {
            let text = trimmed.trim_start_matches('!');
            if let Some(start) = text.find('[') {
                if let Some(end) = text[start..].find(']') {
                    let alt = &text[start + 1..start + end];
                    if !alt.is_empty() {
                        out.push_str(alt);
                        out.push('\n');
                    }
                    continue;
                }
            }
            out.push_str(trimmed);
            out.push('\n');
        } else {
            let cleaned = strip_inline_markdown(trimmed);
            out.push_str(&cleaned);
            out.push('\n');
        }
    }
    let s = out.replace("\n\n\n+", "\n\n");
    s.trim().to_string()
}

fn strip_inline_markdown(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let ch = s[i..].chars().next().unwrap();
        if ch == '[' {
            if let Some(end_bracket) = s[i..].find(']') {
                let after_bracket = i + end_bracket + 1;
                if after_bracket < s.len() && s[after_bracket..].starts_with('(') {
                    if let Some(close_paren) = s[after_bracket..].find(')') {
                        let link_text = &s[i + 1..i + end_bracket];
                        result.push_str(link_text);
                        i = after_bracket + close_paren + 1;
                        continue;
                    }
                }
                let link_text = &s[i + 1..i + end_bracket];
                result.push_str(link_text);
                i = i + end_bracket + 1;
                continue;
            }
        }
        if ch == '*' || ch == '_' || ch == '`' || ch == '~' {
            i += ch.len_utf8();
            continue;
        }
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

pub fn extract_text(path: &Path) -> Result<String, CvParseError> {
    let md = anydoc::to_markdown(path).map_err(map_anydoc_err)?;
    Ok(strip_markdown_to_text(&md))
}

pub fn extract_text_from_bytes(bytes: &[u8], format_hint: Option<&str>) -> Result<String, CvParseError> {
    let format = match format_hint {
        Some(hint) => anydoc::Format::from_extension(hint),
        None => anydoc::Format::from_bytes(bytes),
    };
    let md = anydoc::to_markdown_bytes(bytes, format).map_err(map_anydoc_err)?;
    Ok(strip_markdown_to_text(&md))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_inline_markdown_basic() {
        assert_eq!(strip_inline_markdown("hello **world**"), "hello world");
        assert_eq!(strip_inline_markdown("some *italic* text"), "some italic text");
        assert_eq!(strip_inline_markdown("`code` here"), "code here");
        assert_eq!(strip_inline_markdown("~~strikethrough~~"), "strikethrough");
    }

    #[test]
    fn test_strip_inline_markdown_links() {
        assert_eq!(strip_inline_markdown("[link](http://example.com)"), "link");
        assert_eq!(strip_inline_markdown("see [docs](https://a.com) now"), "see docs now");
    }

    #[test]
    fn test_strip_inline_markdown_unicode() {
        assert_eq!(strip_inline_markdown("60% – in java"), "60% – in java");
        assert_eq!(strip_inline_markdown("עברית **מעוצבת**"), "עברית מעוצבת");
    }

    #[test]
    fn test_strip_markdown_to_text() {
        let input = "# Heading\n\nSome **bold** text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nMore text.";
        let result = strip_markdown_to_text(input);
        assert!(result.contains("Heading"));
        assert!(result.contains("Some bold text."));
        assert!(result.contains("a b"));
        assert!(result.contains("1 2"));
        assert!(result.contains("More text."));
        assert!(!result.contains("|"));
        assert!(!result.contains("---"));
    }

    #[test]
    fn test_extract_text_from_bytes_unsupported() {
        let result = extract_text_from_bytes(b"not a real document", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_detection() {
        assert!(anydoc::Format::from_extension("pdf").is_some());
        assert!(anydoc::Format::from_extension("docx").is_some());
        assert!(anydoc::Format::from_extension("doc").is_some());
        assert!(anydoc::Format::from_extension("xyz").is_none());
    }
}
