//! `libtriqui_client` — client side of the gui↔serve protocol.
//!
//! This crate is the contract between the Tauri gui and the headless
//! serve loop (thin gui pattern, `NOAI.md` / `gui/design/app1/thin-gui.md`). Both
//! sides share the envelope ([`protocol`]), the command params
//! ([`params`]), and every DTO that crosses the wire ([`types`]), so the
//! contract cannot drift (contract-lib pattern, same role
//! `libmodelcallerclient` plays for `modelcaller`).
//!
//! The envelope runs in-process: serve runs on a thread inside the host
//! (the gui, or the e2e contract tests), connected through
//! [`transport::duplex`] and [`client::TriqiClient::over`] — no child
//! process, no `triqui-cli` binary (thin gui pattern).
//!
//! The host links only this crate plus the `triqui-cli` library that hosts
//! the serve loop; the DB, triq.yaml, engine runtime, model management, and
//! CV watching all live behind it.

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
