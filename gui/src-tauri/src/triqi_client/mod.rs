//! `triqi_client` — inlined copy of `libtriqui_client` so the GUI has zero
//! local crate dependencies.
//!
//! This module is the contract between the Tauri gui and the headless serve
//! loop (thin gui pattern). Both sides share the envelope ([`protocol`]),
//! the command params ([`params`]), and every DTO that crosses the wire
//! ([`types`]), so the contract cannot drift.
//!
//! The host links only this module plus the `triq` CLI binary that hosts
//! the serve loop as a subprocess; the DB, triq.yaml, engine runtime,
//! model management, and CV watching all live behind it.

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
