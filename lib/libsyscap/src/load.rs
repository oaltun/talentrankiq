//! Live host load: what the machine is doing RIGHT NOW. Pull-based — the
//! sampler owns the sysinfo state (cpu % is a delta between refreshes) and
//! never spawns threads; consumers sample at natural job boundaries (start of
//! a wave, a batch, a test). No background watcher in v1.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// One load reading of the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadSnapshot {
    /// Whole-system cpu utilization, 0.0..=100.0.
    pub cpu_pct: f32,
    /// 1 / 5 / 15 minute load averages.
    pub loadavg: [f64; 3],
    pub mem_total_bytes: u64,
    pub mem_available_bytes: u64,
    pub swap_used_bytes: u64,
    /// Memory pressure score 0.0..=1.0. v1 derivation: the share of RAM NOT
    /// immediately available (swap pressure folds into future work; the mach
    /// pressure level is phase 2).
    pub mem_pressure: f64,
}

impl LoadSnapshot {
    /// Free-RAM fraction (1.0 = empty machine, 0.0 = RAM fully committed).
    pub fn free_mem_frac(&self) -> f64 {
        if self.mem_total_bytes == 0 {
            1.0
        } else {
            (self.mem_available_bytes as f64 / self.mem_total_bytes as f64).clamp(0.0, 1.0)
        }
    }
}

/// Pull-based load sampler. The first [`LoadSampler::sample`] takes one
/// sysinfo cpu-refresh interval (~200 ms) to establish a delta; subsequent
/// calls inside `min_interval` (and inside sysinfo's own
/// `MINIMUM_CPU_UPDATE_INTERVAL`, below which cpu % is meaningless) return the
/// previous snapshot unchanged, so the budget cannot flip-flop.
#[derive(Debug)]
pub struct LoadSampler {
    sys: sysinfo::System,
    last: Option<(Instant, LoadSnapshot)>,
}

impl LoadSampler {
    pub fn new() -> Self {
        Self {
            sys: sysinfo::System::new(),
            last: None,
        }
    }

    pub fn sample(&mut self, min_interval: Duration) -> LoadSnapshot {
        let now = Instant::now();
        let min_gap = min_interval.max(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        if let Some((at, snap)) = &self.last {
            if now.duration_since(*at) < min_gap {
                return snap.clone();
            }
        }
        match &self.last {
            Some((at, _)) => {
                let elapsed = now.duration_since(*at);
                if elapsed < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
                    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL - elapsed);
                }
            }
            None => {
                self.sys.refresh_cpu_usage();
                std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
            }
        }
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let cpu_pct = self.sys.global_cpu_usage().clamp(0.0, 100.0);
        let la = sysinfo::System::load_average();
        let total = self.sys.total_memory();
        // Not `sysinfo::available_memory()`: it returns 0 on this macOS build,
        // which would read as "RAM exhausted" and pin the whole policy at Hold
        // (see `crate::mem`).
        let available = crate::mem::available_memory(total);
        let pressure = if total == 0 {
            0.0
        } else {
            (1.0 - available as f64 / total as f64).clamp(0.0, 1.0)
        };
        let snap = LoadSnapshot {
            cpu_pct,
            loadavg: [la.one, la.five, la.fifteen],
            mem_total_bytes: total,
            mem_available_bytes: available,
            swap_used_bytes: self.sys.used_swap(),
            mem_pressure: pressure,
        };
        self.last = Some((Instant::now(), snap.clone()));
        snap
    }
}

impl Default for LoadSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_returns_sane_ranges() {
        let mut s = LoadSampler::new();
        let a = s.sample(Duration::from_millis(250));
        std::thread::sleep(Duration::from_millis(250));
        let b = s.sample(Duration::from_millis(250));
        for snap in [a, b] {
            assert!((0.0..=100.0).contains(&snap.cpu_pct));
            assert!(snap.loadavg.iter().all(|v| v.is_finite() && *v >= 0.0));
            assert!(snap.mem_total_bytes > 0);
            assert!(snap.mem_available_bytes <= snap.mem_total_bytes);
            assert!((0.0..=1.0).contains(&snap.mem_pressure));
            assert_eq!(snap.free_mem_frac(), snap.mem_available_bytes as f64 / snap.mem_total_bytes as f64);
        }
    }

    #[test]
    fn sampler_rate_limits_within_interval() {
        let mut s = LoadSampler::new();
        let a = s.sample(Duration::from_secs(10));
        let b = s.sample(Duration::from_secs(10));
        assert_eq!(a, b, "inside min_interval the last snapshot is returned");
    }
}
