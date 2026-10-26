//! `libcli_client` — JSONL wire protocol crate for `triq serve`.
//!
//! This crate defines the contract between clients and the `triq serve` loop
//! (the CLI server). Clients subprocess-spawn `triq serve` and communicate
//! via JSONL over stdin/stdout.
//!
//! The crate exports:
//! - **Envelope types**: [`protocol::Request`], [`protocol::Response`],
//!   [`protocol::Event`], [`protocol::Inbound`], [`protocol::ProtocolError`]
//! - **Command params**: all structs in [`params`]
//! - **Wire DTOs**: all types in [`types`] (every type that crosses the JSONL wire)
//! - **Client**: [`client::TriqiClient`] for connecting to a serve loop
//! - **Transport**: [`transport::duplex`], [`transport::DuplexReader`],
//!   [`transport::DuplexWriter`] for in-process testing
//!
//! JSON Schema and TypeScript bindings are generated at build time via
//! `build.rs` and written to `protocol-schema.json` and `protocol.ts`
//! in the crate root.

pub mod client;
pub mod params;
pub mod protocol;
pub mod transport;
pub mod types;

pub use client::{ClientError, TriqiClient};
pub use params::*;
pub use protocol::{Event, Inbound, ProtocolError, Request, Response, MAX_LINE_BYTES};
pub use transport::{duplex, DuplexReader, DuplexWriter};
pub use types::*;

#[cfg(test)]
mod schema_tests {
    use std::fs;

    /// Test that protocol-schema.json exists and is valid JSON
    #[test]
    fn protocol_schema_exists_and_valid() {
        let schema_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol-schema.json");
        let content = fs::read_to_string(schema_path).expect("protocol-schema.json should exist");
        
        // Parse as JSON
        let json: serde_json::Value = serde_json::from_str(&content)
            .expect("protocol-schema.json should be valid JSON");
        
        // Verify it's a JSON Schema (has definitions)
        assert!(json.get("definitions").is_some(), "Schema should have definitions");
        
        // Verify key types are present
        let definitions = json["definitions"].as_object().expect("definitions should be object");
        assert!(definitions.contains_key("Request"), "Schema should define Request");
        assert!(definitions.contains_key("Response"), "Schema should define Response");
        assert!(definitions.contains_key("Event"), "Schema should define Event");
        assert!(definitions.contains_key("Inbound"), "Schema should define Inbound");
        assert!(definitions.contains_key("ProtocolError"), "Schema should define ProtocolError");
        assert!(definitions.contains_key("DomainError"), "Schema should define DomainError");
        assert!(definitions.contains_key("RecoveryError"), "Schema should define RecoveryError");
        assert!(definitions.contains_key("AppPreferences"), "Schema should define AppPreferences");
        assert!(definitions.contains_key("Job"), "Schema should define Job");
        assert!(definitions.contains_key("Candidate"), "Schema should define Candidate");
        assert!(definitions.contains_key("CandidateSummary"), "Schema should define CandidateSummary");
        assert!(definitions.contains_key("ModelKind"), "Schema should define ModelKind");
        assert!(definitions.contains_key("TryModelProgress"), "Schema should define TryModelProgress");
    }

    /// Test that protocol-schema.json can be parsed as JSON Schema draft 07
    #[test]
    fn protocol_schema_valid_json_schema() {
        let schema_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol-schema.json");
        let content = fs::read_to_string(schema_path).expect("protocol-schema.json should exist");
        let json: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
        
        // Should have $schema or be valid draft 07
        // At minimum, definitions should be objects with type properties
        let definitions = json["definitions"].as_object().expect("definitions should be object");
        
        // Check Request structure
        let request = &definitions["Request"];
        assert_eq!(request["type"], "object");
        let props = request["properties"].as_object().expect("Request should have properties");
        assert!(props.contains_key("id"), "Request should have id");
        assert!(props.contains_key("cmd"), "Request should have cmd");
        assert!(props.contains_key("params"), "Request should have params");
    }

    /// Test that protocol.ts exists and contains expected TypeScript types
    #[test]
    fn protocol_ts_exists_and_valid() {
        let ts_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol.ts");
        let content = fs::read_to_string(ts_path).expect("protocol.ts should exist");
        
        // Should have auto-generated header
        assert!(content.contains("Auto-generated CLI protocol types"), "Should have auto-generated header");
        
        // Verify key types are exported
        assert!(content.contains("export type Request"), "Should export Request");
        assert!(content.contains("export type Response"), "Should export Response");
        assert!(content.contains("export type Event"), "Should export Event");
        assert!(content.contains("export type Inbound"), "Should export Inbound");
        assert!(content.contains("export type ProtocolError"), "Should export ProtocolError");
        assert!(content.contains("export type DomainError"), "Should export DomainError");
        assert!(content.contains("export type RecoveryError"), "Should export RecoveryError");
        assert!(content.contains("export type AppPreferences"), "Should export AppPreferences");
        assert!(content.contains("export type Job"), "Should export Job");
        assert!(content.contains("export type Candidate"), "Should export Candidate");
        assert!(content.contains("export type CandidateSummary"), "Should export CandidateSummary");
        assert!(content.contains("export type ModelKind"), "Should export ModelKind");
        assert!(content.contains("export type TryModelProgress"), "Should export TryModelProgress");
        assert!(content.contains("export type TryModelResult"), "Should export TryModelResult");
        assert!(content.contains("export type ImportProgress"), "Should export ImportProgress");
        assert!(content.contains("export type EmbeddingProgress"), "Should export EmbeddingProgress");
    }

    /// Test that TypeScript types have correct structure
    #[test]
    fn protocol_ts_types_structure() {
        let ts_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol.ts");
        let content = fs::read_to_string(ts_path).expect("protocol.ts should exist");
        
        // Request should have id, cmd, params
        assert!(content.contains("id: number"), "Request should have id: number");
        assert!(content.contains("cmd: string"), "Request should have cmd: string");
        assert!(content.contains("params?"), "Request should have optional params");
        
        // Response should have id, ok, err
        assert!(content.contains("ok?"), "Response should have optional ok");
        assert!(content.contains("err?"), "Response should have optional err");
        
        // ProtocolError should have code, message, domain
        assert!(content.contains("code: string"), "ProtocolError should have code");
        assert!(content.contains("message: string"), "ProtocolError should have message");
        
        // ModelKind should be enum
        assert!(content.contains("type ModelKind ="), "ModelKind should be type alias");
    }

    /// Test that both files are in sync (same types defined)
    #[test]
    fn schema_and_ts_in_sync() {
        let schema_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol-schema.json");
        let ts_path = concat!(env!("CARGO_MANIFEST_DIR"), "/protocol.ts");
        
        let schema_content = fs::read_to_string(schema_path).expect("schema should exist");
        let ts_content = fs::read_to_string(ts_path).expect("ts should exist");
        
        let schema: serde_json::Value = serde_json::from_str(&schema_content).expect("valid schema JSON");
        let definitions = schema["definitions"].as_object().expect("definitions object");
        
        // Every definition in JSON Schema should have corresponding TypeScript export
        for def_name in definitions.keys() {
            // Skip internal/private types
            if def_name.starts_with('_') {
                continue;
            }
            let ts_pattern = format!("export type {} =", def_name);
            assert!(
                ts_content.contains(&ts_pattern),
                "TypeScript missing export for {} (found in JSON Schema)",
                def_name
            );
        }
    }
}