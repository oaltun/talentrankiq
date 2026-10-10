//! `modelcaller` — ONNX inference orchestrator, as a library.
//!
//! There is no `modelcaller` binary: every consumer runs the JSONL loop
//! in-process via [`run_stdio_over`] on a dedicated thread — triqui-core
//! and `libmodelcallerclient::ModelCallerClient::spawn_inprocess` (bench,
//! judgetrainer) behind the gui (thin gui pattern, `gui/design/app1/thin-gui.md`).
//! The wire protocol contract lives in `libmodelcallerclient` on the client
//! side.
//!
//! Below modelcaller, the ORT wrapper (`onnxwrapperrs`) also runs in-process
//! on its own thread over OS pipes, so inference never blocks the host's
//! loop thread.

pub mod cli;
pub mod commands;
pub mod error;
pub mod model_registry;
pub mod protocol;
pub mod wrapper;

pub use cli::StdioArgs;
pub use commands::{run_stdio_over, variant_status, VariantStatus};
pub use error::{ModelError, Result};
pub use model_registry::Registry;
