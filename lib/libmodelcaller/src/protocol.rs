use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Envelope {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

impl Envelope {
    pub fn parse_line(line: &str) -> Result<Envelope, String> {
        let line = line.trim();
        if line.is_empty() {
            return Err("empty line.".to_string());
        }
        serde_json::from_str::<Envelope>(line).map_err(|e| format!("invalid JSON: {e}"))
    }

    pub fn response(id: &str, payload: Value) -> Self {
        Envelope {
            id: Some(id.to_string()),
            r#type: "response".to_string(),
            payload,
        }
    }

    pub fn error(id: Option<&str>, code: &str, msg: impl Into<String>) -> Self {
        Envelope {
            id: id.map(|s| s.to_string()),
            r#type: "error".to_string(),
            payload: json!({ "code": code, "msg": msg.into() }),
        }
    }

    pub fn to_line(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("failed to serialize response: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip() {
        let line = r#"{"id":"1","type":"request","payload":{"cmd":"ping"}}"#;
        let env = Envelope::parse_line(line).unwrap();
        assert_eq!(env.id.as_deref(), Some("1"));
        assert_eq!(env.r#type, "request");
        assert_eq!(env.payload["cmd"], "ping");
    }

    #[test]
    fn parse_invalid_json() {
        assert!(Envelope::parse_line("{not json").is_err());
    }

    #[test]
    fn parse_empty_line() {
        assert!(Envelope::parse_line("   ").is_err());
    }

    #[test]
    fn response_carries_id_and_payload() {
        let r = Envelope::response("7", json!({"ok": true}));
        let line = r.to_line().unwrap();
        let parsed = Envelope::parse_line(&line).unwrap();
        assert_eq!(parsed.id.as_deref(), Some("7"));
        assert_eq!(parsed.r#type, "response");
        assert_eq!(parsed.payload["ok"], true);
    }

    #[test]
    fn error_uses_null_id_for_unparseable() {
        let e = Envelope::error(None, "bad_req", "bad");
        assert!(e.id.is_none());
        assert_eq!(e.r#type, "error");
        assert_eq!(e.payload["code"], "bad_req");
    }

    #[test]
    fn unparseable_input_yields_null_id() {
        // caller behavior: parse error -> error envelope with id null
        let e = Envelope::parse_line("{bad");
        assert!(e.is_err());
        let err_env = Envelope::error(None, "bad_req", e.unwrap_err());
        assert!(err_env.id.is_none());
    }
}
