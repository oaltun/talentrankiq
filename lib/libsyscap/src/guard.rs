//! Consumer-side gates built on the budget.

use std::time::Duration;

use crate::budget::{budget, Verdict};
use crate::inventory::Inventory;
use crate::load::LoadSampler;

/// True when the machine is already busy enough that heavy work should be
/// skipped: the default-config budget verdicts `Backoff` or `Hold` (another
/// job is eating the box — datamaker, kilo, a bench run, the gui).
///
/// Blocks up to one sysinfo cpu-refresh interval (~200 ms) on the first
/// sample. Uses the default `syscap:` knobs and no configured engine caps
/// (gates want to know about the MACHINE, not about one consumer's caps).
pub fn machine_busy() -> bool {
    let inv = Inventory::global();
    let mut sampler = LoadSampler::new();
    let snap = sampler.sample(Duration::from_millis(
        libconfigreader::DEFAULT_SYSCAP_SAMPLE_MIN_INTERVAL_MS,
    ));
    let cfg = libconfigreader::SyscapSection::default();
    let caps = libconfigreader::EngineSection::default();
    matches!(
        budget(inv, &snap, &cfg, &caps).verdict,
        Verdict::Backoff | Verdict::Hold
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_busy_probes_without_panicking() {
        // Result is machine-state-dependent (true when the box is loaded), so
        // only the full probe->sample->budget path must hold.
        let _ = machine_busy();
    }
}
