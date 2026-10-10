use std::fmt;

#[derive(Debug, Clone)]
pub struct ModelError {
    pub code: &'static str,
    pub msg: String,
    pub exit_code: Option<u8>,
}

impl ModelError {
    pub fn new(code: &'static str, msg: impl Into<String>) -> Self {
        ModelError {
            code,
            msg: msg.into(),
            exit_code: None,
        }
    }

    pub fn no_model(id: &str) -> Self {
        ModelError::new(
            "no_model",
            format!("unknown model id '{id}' (run modeldownloader first)"),
        )
    }

    pub fn ambiguous_onnx(dir: &str) -> Self {
        ModelError::new(
            "ambiguous_onnx",
            format!("multiple .onnx files found under '{dir}'; cache only one quantization per model"),
        )
    }

    pub fn bad_req(msg: impl Into<String>) -> Self {
        ModelError::new("bad_req", msg)
    }

    pub fn load_err(msg: impl Into<String>) -> Self {
        ModelError::new("load_err", msg)
    }

    /// model output could not be parsed/validated into the target contract
    /// (design 14 §14.2: callers fall back to hand-curated schemas or
    /// base-only evidence)
    pub fn invalid_output(msg: impl Into<String>) -> Self {
        ModelError::new("invalid_output", msg)
    }

    #[allow(dead_code)]
    pub fn infer(msg: impl Into<String>) -> Self {
        ModelError::new("infer", msg)
    }

    pub fn fatal(code: &'static str, msg: impl Into<String>, exit_code: u8) -> Self {
        ModelError {
            code,
            msg: msg.into(),
            exit_code: Some(exit_code),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.msg)
    }
}

impl From<String> for ModelError {
    fn from(s: String) -> Self {
        ModelError::new("registry", s)
    }
}

impl std::error::Error for ModelError {}

pub type Result<T> = std::result::Result<T, ModelError>;
