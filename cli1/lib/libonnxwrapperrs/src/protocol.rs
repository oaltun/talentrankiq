use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub payload: Payload,
}

pub type Payload = serde_json::Map<String, Value>;

impl Envelope {
    pub fn request(id: &str, payload: Payload) -> Self {
        Self {
            id: Some(id.to_string()),
            msg_type: "request".into(),
            payload,
        }
    }

    pub fn response(id: Option<String>, payload: Payload) -> Self {
        Self {
            id,
            msg_type: "response".into(),
            payload,
        }
    }

    pub fn error(id: Option<String>, code: &str, msg: &str) -> Self {
        Self {
            id,
            msg_type: "error".into(),
            payload: serde_json::json!({
                "code": code,
                "msg": msg,
            })
            .as_object()
            .unwrap()
            .clone(),
        }
    }
}

impl fmt::Display for Envelope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "{s}")
    }
}
