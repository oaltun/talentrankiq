//! Shared TalentRankIQ types.
//!
//! Since the thin-gui migration these are the gui↔cli wire contract and live
//! in `libtriqui_client` (see `gui/design/app1/thin-gui.md`); core re-exports them
//! under `crate::types` so all domain modules keep their paths.

pub use libtriqui_client::types::*;
