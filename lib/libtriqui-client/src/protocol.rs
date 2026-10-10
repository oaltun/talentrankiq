//! JSONL framing for the gui↔`triqui-cli serve` stdio protocol.
//!
//! One JSON object per line:
//! - request  gui → cli: `{"id":7,"cmd":"list_jobs","params":{...}}`
//! - response cli → gui: `{"id":7,"ok":{...}}` or `{"id":7,"err":{...}}`
//! - event    cli → gui: `{"event":"job-import-progress","data":{...}}`
//!
//! Long operations follow the start/track rule: the command's response returns
//! immediately with a handle, progress/completion arrive as events (or the
//! request simply stays open when the frontend awaits completion — either way
//! the request loop never blocks on modelcaller/modeldownloader).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Defensive cap on one protocol line (16 MB). Bulk data must go through
/// files (path in the message), never through the pipe.
pub const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;

/// A request from the gui to the serve process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub cmd: String,
    #[serde(default)]
    pub params: Value,
}

/// A protocol error. `domain` carries the serialized typed error (e.g.
/// [`DomainError`] / [`crate::types::RecoveryError`]) for commands whose
/// frontend signature matches on it; `message` is always the display form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolError {
    /// `"domain"` when `domain` is set, `"internal"` otherwise.
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<Value>,
}

impl ProtocolError {
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: "internal".to_string(),
            message: message.into(),
            domain: None,
        }
    }

    /// Wrap a fallible-domain error (serialized, so the gui shell can
    /// rehydrate the exact typed error the frontend expects).
    pub fn domain<E: Serialize + std::fmt::Display>(err: &E) -> Self {
        Self {
            code: "domain".to_string(),
            message: err.to_string(),
            domain: Some(serde_json::to_value(err).unwrap_or(Value::Null)),
        }
    }

    /// From a plain string error (commands that returned `Result<_, String>`).
    pub fn string(message: impl Into<String>) -> Self {
        Self::internal(message)
    }
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// `?`-friendly conversion so handlers can propagate fallible-domain errors
/// (`Result<_, DomainError>`) directly into `Result<_, ProtocolError>`.
impl From<crate::types::DomainError> for ProtocolError {
    fn from(err: crate::types::DomainError) -> Self {
        Self::domain(&err)
    }
}

impl From<crate::types::RecoveryError> for ProtocolError {
    fn from(err: crate::types::RecoveryError) -> Self {
        Self::domain(&err)
    }
}

/// A response to one request. Exactly one of `ok`/`err` is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub err: Option<ProtocolError>,
}

impl Response {
    pub fn ok(id: u64, value: Value) -> Self {
        Self {
            id,
            ok: Some(value),
            err: None,
        }
    }

    pub fn err(id: u64, err: ProtocolError) -> Self {
        Self {
            id,
            ok: None,
            err: Some(err),
        }
    }
}

/// An unsolicited event from the serve process (progress, status, ...).
/// The gui bridge forwards these as Tauri events with the same name/payload,
/// so frontend listeners are unchanged by the migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event: String,
    pub data: Value,
}

/// Anything the serve process can send on stdout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Inbound {
    Response(Response),
    Event(Event),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DomainError;

    #[test]
    fn response_round_trip_ok() {
        let r = Response::ok(7, serde_json::json!({"import_id": "abc"}));
        let line = serde_json::to_string(&r).unwrap();
        let back: Response = serde_json::from_str(&line).unwrap();
        assert_eq!(back.id, 7);
        assert_eq!(back.ok.unwrap()["import_id"], "abc");
        assert!(back.err.is_none());
    }

    #[test]
    fn response_round_trip_domain_error() {
        let e = DomainError::NotFound {
            entity: "Job".into(),
            id: "j1".into(),
        };
        let r = Response::err(3, ProtocolError::domain(&e));
        let line = serde_json::to_string(&r).unwrap();
        assert!(!line.contains("\"ok\""), "err-only response must omit ok");
        let back: Response = serde_json::from_str(&line).unwrap();
        let err = back.err.unwrap();
        assert_eq!(err.code, "domain");
        let de: DomainError = serde_json::from_value(err.domain.unwrap()).unwrap();
        assert!(matches!(de, DomainError::NotFound { ref id, .. } if id == "j1"));
    }

    #[test]
    fn inbound_demux() {
        let resp: Inbound = serde_json::from_str(r#"{"id":1,"ok":null}"#).unwrap();
        assert!(matches!(resp, Inbound::Response(_)));
        let event: Inbound = serde_json::from_str(
            r#"{"event":"job-import-progress","data":{"job_id":"j","done":1,"total":2,"failed":0,"active":true,"phase":"extracting"}}"#,
        )
        .unwrap();
        assert!(matches!(event, Inbound::Event(_)));
    }
}
