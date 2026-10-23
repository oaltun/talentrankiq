//! Host capability + live-load budget.
//!
//! Three questions, one small library:
//! 1. what does this machine HAVE — [`inventory::Inventory`] (cpu topology
//!    including the Apple Silicon P/E split, RAM, GPUs via Metal, accelerator
//!    hints, power state, disk free);
//! 2. what is it doing RIGHT NOW — [`load::LoadSampler`] (cpu %, loadavg,
//!    memory available/total, swap, a memory-pressure score);
//! 3. how much of it may this process use — [`budget::budget`], a pure policy
//!    that turns (1) + (2) + the `triq.yaml` knobs into a [`budget::Budget`].
//!
//! Invariants (see `design/design.md`):
//! - no subprocesses, no threads of its own, no panics on probe failure, no
//!   writes to disk or config — the gui runs this in its zero-children binary;
//! - the budget only ever LOWERS the configured `engine:` caps (config stays
//!   the enforcement; a busy machine must not surprise the user, an idle one
//!   should not be wasted);
//! - config comes from `triq.yaml` only (`syscap:` section via
//!   `libconfigreader`); no env-var config.

pub mod budget;
pub mod guard;
pub mod inventory;
pub mod load;
pub mod mem;

pub use budget::{
    build_jobs, build_jobs_now, budget, effective_inter_threads, effective_intra_threads, Budget,
    Verdict,
};
pub use guard::machine_busy;
pub use inventory::{AccelInfo, CpuInfo, CpuVendor, DiskInfo, Gpu, GpuVendor, Inventory, MemInfo, PowerInfo};
pub use load::{LoadSampler, LoadSnapshot};
