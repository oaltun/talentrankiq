//! The budget policy: given what the host has ([`Inventory`]), what it is
//! doing ([`LoadSnapshot`]), the `ikapp.yaml` `syscap:` knobs and the
//! configured `engine:` caps, decide how much of the machine this process may
//! use RIGHT NOW.
//!
//! Pure and table-tested — no I/O, no threads, no state. The output is always
//! ≤ the configured caps: this library recommends; `ikapp.yaml` enforces.

use serde::{Deserialize, Serialize};

use crate::inventory::Inventory;
use crate::load::{LoadSampler, LoadSnapshot};

const GIB: f64 = (1u64 << 30) as f64;

/// How hard the process may work right now, from most to least aggressive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Machine quiet: hand out the configured caps.
    Full,
    /// Somewhat loaded or on battery: halve the knobs, shrink the waves.
    Polite,
    /// Overloaded or low on RAM: floor values, barely-visible progress.
    Backoff,
    /// Nearly out of RAM: defer heavy work entirely. The ONLY verdict the gui
    /// must surface to the user (a `holding` reason code + visible notice);
    /// Polite/Backoff adapt silently.
    Hold,
}

impl Verdict {
    /// Multiplier for wave-sized consumers (cascade k1/k2): 1.0 / 0.5 / 0.25
    /// / 0.0. Consumers floor their `config_value * wave_scale` and keep the
    /// config value as the ceiling.
    pub fn wave_scale(self) -> f64 {
        match self {
            Verdict::Full => 1.0,
            Verdict::Polite => 0.5,
            Verdict::Backoff => 0.25,
            Verdict::Hold => 0.0,
        }
    }

    /// `false` only for [`Verdict::Hold`] (pause heavy work).
    pub fn heavy_ok(self) -> bool {
        !matches!(self, Verdict::Hold)
    }

    /// Snake_case code matching the serde wire form — for logs and reason
    /// codes (`full`, `polite`, `backoff`, `hold`).
    pub fn code(self) -> &'static str {
        match self {
            Verdict::Full => "full",
            Verdict::Polite => "polite",
            Verdict::Backoff => "backoff",
            Verdict::Hold => "hold",
        }
    }
}

/// What a consumer may do right now. `thread_cap`/`embed_batch` are the
/// configured engine caps after the verdict scaled them down; `None` batch
/// means the config set none (keep the consumer's own default — only
/// Polite/Backoff force a concrete floor, see below). `mem_cap_bytes` is an
/// advisory ceiling for this process' allocations right now (`u64::MAX` =
/// uncapped, `0` = hold). `capacity` is the machine's core budget
/// (`P + E − reserve`) — what the hardware offers right now, independent of
/// the configured engine caps (build jobs scale on it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub verdict: Verdict,
    pub thread_cap: usize,
    pub embed_batch: Option<usize>,
    pub wave_scale: f64,
    pub mem_cap_bytes: u64,
    pub capacity: usize,
}

/// Compute the budget. `cfg` is the resolved `syscap:` section (absent yaml =
/// defaults, per `libconfigreader`), `caps` the `engine:` section the caller
/// runs with.
pub fn budget(
    inv: &Inventory,
    load: &LoadSnapshot,
    cfg: &libconfigreader::SyscapSection,
    caps: &libconfigreader::EngineSection,
) -> Budget {
    let capacity = capacity_cores(inv, cfg);
    if !cfg.enabled() {
        // Disabled: report the configured caps untouched, sample nothing.
        // No hardware clamp either — `enabled: false` must reproduce the
        // exact pre-syscap behavior (that's what makes A/B runs comparable).
        return Budget {
            verdict: Verdict::Full,
            thread_cap: caps.intra_threads.unwrap_or(capacity),
            embed_batch: caps.embed_batch,
            wave_scale: 1.0,
            mem_cap_bytes: u64::MAX,
            capacity,
        };
    }
    let verdict = classify(inv, load, cfg);
    let full_threads = caps.intra_threads.unwrap_or(usize::MAX).min(capacity);
    let (thread_cap, embed_batch) = match verdict {
        Verdict::Full => (full_threads, caps.embed_batch),
        Verdict::Polite => (half(full_threads), halve(caps.embed_batch)),
        Verdict::Backoff | Verdict::Hold => (1, caps.embed_batch.map(|_| 1)),
    };
    Budget {
        verdict,
        thread_cap,
        embed_batch,
        wave_scale: verdict.wave_scale(),
        mem_cap_bytes: mem_cap_for(verdict, load, cfg),
        capacity,
    }
}

/// Throughput capacity in cores: every core the machine has, minus the user's
/// headroom reserve (floor 1). E-cores count FULLY — the earlier half-weight
/// was a hedge for mixed-core throughput, but on an idle box the only cost of
/// using them is thermal, and the verdict ladder (not the ceiling) is what
/// yields when someone else needs the machine. Machines without a P/E split
/// use logical cores as the base.
fn capacity_cores(inv: &Inventory, cfg: &libconfigreader::SyscapSection) -> usize {
    let logical = inv.cpu.logical.max(1) as usize;
    let reserve = (cfg.reserve_cores() as usize)
        .max((cfg.reserve_core_frac() * logical as f64).ceil() as usize);
    let base = match inv.cpu.perf_cores {
        Some(p) => p as usize + inv.cpu.eff_cores.unwrap_or(0) as usize,
        None => logical,
    };
    base.saturating_sub(reserve).max(1)
}

fn half(n: usize) -> usize {
    (n / 2).max(1)
}

fn halve(batch: Option<usize>) -> Option<usize> {
    batch.map(|b| (b / 2).max(1))
}

fn classify(
    inv: &Inventory,
    load: &LoadSnapshot,
    cfg: &libconfigreader::SyscapSection,
) -> Verdict {
    let logical = inv.cpu.logical.max(1) as f64;
    let free_frac = load.free_mem_frac();
    let load_ratio = load.loadavg[0] / logical;
    if free_frac < cfg.hold_free_mem_frac() {
        return Verdict::Hold;
    }
    // Backoff = genuinely squeezed: RAM short, or the run queue twice the size
    // of the machine (thrashing). Polite = the queue outgrew the machine
    // (`load_backoff`, default 1.0/core) or we are on battery. Sub-1.0 load is
    // NOT a reason to yield: on an 18-core box one background compile — or a
    // consumer's own threads — crosses the old 0.70 threshold while the
    // desktop is idle, which wasted idle cores and made an embedder throttle
    // itself mid-wave (its own threads are inside the load average).
    if free_frac < cfg.min_free_mem_frac() || load_ratio > cfg.load_stall() {
        return Verdict::Backoff;
    }
    let battery_polite = cfg.on_battery_polite() && inv.power.on_battery == Some(true);
    if battery_polite || load_ratio > cfg.load_backoff() {
        return Verdict::Polite;
    }
    Verdict::Full
}

/// Advisory allocation ceiling: free RAM beyond the user's reserve, scaled by
/// the verdict (Full keeps the whole slice, Polite half, Backoff a quarter,
/// Hold nothing).
fn mem_cap_for(
    verdict: Verdict,
    load: &LoadSnapshot,
    cfg: &libconfigreader::SyscapSection,
) -> u64 {
    let reserve = (cfg.min_free_mem_frac() * load.mem_total_bytes as f64)
        .max(2.0 * GIB)
        .max(0.0) as u64;
    let free_beyond_reserve = load.mem_available_bytes.saturating_sub(reserve);
    match verdict {
        Verdict::Full => free_beyond_reserve,
        Verdict::Polite => free_beyond_reserve / 2,
        Verdict::Backoff => free_beyond_reserve / 4,
        Verdict::Hold => 0,
    }
}

/// Effective ONNX intra-op thread count: the configured `engine.intra_threads`
/// cap, or the budget's (`capacity`-derived when the config sets none) — the
/// config can only lower, never raise. Consumers spawn the session with this
/// value; unset caps no longer fall through to ONNX Runtime's own default
/// (one worker per physical core), which would bypass the syscap policy.
pub fn effective_intra_threads(caps: &libconfigreader::EngineSection, b: &Budget) -> usize {
    caps.intra_threads.unwrap_or(b.thread_cap).min(b.thread_cap)
}

/// Effective ONNX inter-op (graph) parallelism: the configured
/// `engine.inter_threads`, defaulting to 1 — a model-shape choice, not a
/// machine-size one (these embedding graphs gain nothing from node-level
/// parallelism and a second worker pool only fights the intra-op one) — then
/// capped by the budget.
pub fn effective_inter_threads(caps: &libconfigreader::EngineSection, b: &Budget) -> usize {
    caps.inter_threads.unwrap_or(1).min(b.thread_cap)
}

/// Map the build-jobs spec onto a budget (§12 of the design): `Fixed(n)` wins
/// verbatim (floor 1 — the human override). `Auto` uses the machine's
/// *capacity* (`P + E/2 − reserve`), not the engine thread cap — compilation
/// parallelizes well and may use E-cores, unlike ORT sessions: `Full` hands
/// out the whole capacity, `Polite` half of it (floor 2), and
/// `Backoff`/`Hold` a single unit (links are the RAM peak; free-RAM pressure
/// flips the verdict on the next invocation — builds sample once at start,
/// the per-crate re-sample is watcher territory).
pub fn build_jobs(spec: &libconfigreader::BuildJobs, b: &Budget) -> u32 {
    match spec {
        libconfigreader::BuildJobs::Fixed(n) => (*n).max(1),
        libconfigreader::BuildJobs::Auto => {
            let full = b.capacity as u32;
            match b.verdict {
                Verdict::Full => full,
                Verdict::Polite => (full / 2).max(2),
                Verdict::Backoff | Verdict::Hold => 1,
            }
        }
    }
}

/// One-shot convenience for the build drivers (`ikbuild`, `benchdriverrs`):
/// fresh inventory + load sample -> budget -> resolved jobs. Returns the
/// budget too so callers can log the verdict.
pub fn build_jobs_now(
    cfg: &libconfigreader::SyscapSection,
    caps: &libconfigreader::EngineSection,
) -> (Budget, u32) {
    let inv = Inventory::global();
    let mut sampler = LoadSampler::new();
    let load =
        sampler.sample(std::time::Duration::from_millis(cfg.sample_min_interval_ms()));
    let b = budget(inv, &load, cfg, caps);
    let jobs = build_jobs(&cfg.build_jobs(), &b);
    (b, jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{AccelInfo, CpuInfo, CpuVendor, DiskInfo, MemInfo, PowerInfo};

    fn inv(perf: Option<u32>, logical: u32, on_battery: Option<bool>) -> Inventory {
        Inventory {
            cpu: CpuInfo {
                vendor: if perf.is_some() { CpuVendor::Apple } else { CpuVendor::Intel },
                brand: "test".into(),
                logical,
                physical: logical,
                perf_cores: perf,
                eff_cores: None,
                simd: vec!["neon".into()],
            },
            mem: MemInfo { total_bytes: 16 << 30 },
            gpus: Vec::new(),
            accel: AccelInfo { ane_present: None, coreml_available: false },
            power: PowerInfo { on_battery },
            disk: DiskInfo { home_free_bytes: 100 << 30 },
        }
    }

    fn load(avail_gib: u64, loadavg1: f64) -> LoadSnapshot {
        LoadSnapshot {
            cpu_pct: 0.0,
            loadavg: [loadavg1, loadavg1, loadavg1],
            mem_total_bytes: 16 << 30,
            mem_available_bytes: avail_gib << 30,
            swap_used_bytes: 0,
            mem_pressure: 1.0 - (avail_gib << 30) as f64 / (16u64 << 30) as f64,
        }
    }

    fn caps(intra: usize, batch: usize) -> libconfigreader::EngineSection {
        libconfigreader::EngineSection {
            intra_threads: Some(intra),
            inter_threads: Some(1),
            embed_batch: Some(batch),
            embed_sessions: None,
            exec_provider: None,
            provider_logging: None,
            max_cores: None,
            backend: None,
            no_cache: None,
        }
    }

    fn syscap() -> libconfigreader::SyscapSection {
        libconfigreader::SyscapSection::default()
    }

    #[test]
    fn full_when_idle_hands_out_configured_caps() {
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps(2, 4));
        // reserve = max(1, ceil(0.25 * 8)) = 2; usable = 4 perf - 2 = 2.
        assert_eq!(b.verdict, Verdict::Full);
        assert_eq!(b.thread_cap, 2);
        assert_eq!(b.embed_batch, Some(4));
        assert_eq!(b.wave_scale, 1.0);
        // mem_cap = 8 GiB free - max(2 GiB, 20% of 16 GiB = 3.2 GiB).
        assert_eq!(b.mem_cap_bytes, (8 << 30) - 3_435_973_836);
        assert!(b.verdict.heavy_ok());
    }

    #[test]
    fn polite_on_battery_halves_the_knobs() {
        let b = budget(&inv(Some(4), 8, Some(true)), &load(8, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Polite);
        assert_eq!(b.thread_cap, 1);
        assert_eq!(b.embed_batch, Some(2));
        assert_eq!(b.wave_scale, 0.5);
    }

    #[test]
    fn polite_at_moderate_load() {
        // 9.0 loadavg on 8 logical cores = 1.125/core: the run queue outgrew
        // the machine (default load_backoff 1.0), but it is not thrashing yet.
        let b = budget(&inv(Some(4), 8, None), &load(8, 9.0), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Polite);
    }

    #[test]
    fn backoff_on_overload_floors_everything() {
        // 17 loadavg on 8 logical cores = 2.1/core: twice oversubscribed
        // (default load_stall 2.0).
        let b = budget(&inv(Some(4), 8, None), &load(8, 17.0), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Backoff);
        assert_eq!(b.thread_cap, 1);
        assert_eq!(b.embed_batch, Some(1));
        assert_eq!(b.wave_scale, 0.25);
    }

    #[test]
    fn backoff_on_low_free_ram() {
        // 12.5% free < min_free_mem_frac 0.20, load still quiet.
        let b = budget(&inv(Some(4), 8, None), &load(2, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Backoff);
    }

    #[test]
    fn hold_on_critical_ram_pauses_heavy_work() {
        // 3.1% free < hold_free_mem_frac 0.05.
        let b = budget(&inv(Some(4), 8, None), &load(0, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Hold);
        assert!(!b.verdict.heavy_ok());
        assert_eq!(b.wave_scale, 0.0);
        assert_eq!(b.mem_cap_bytes, 0);
        assert_eq!(b.thread_cap, 1);
        assert_eq!(b.embed_batch, Some(1));
    }

    #[test]
    fn budget_never_exceeds_configured_caps_or_hardware() {
        // Config under hardware: config wins.
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps(1, 2));
        assert_eq!(b.thread_cap, 1);
        assert_eq!(b.embed_batch, Some(2));
        // Config over hardware: hardware+headroom wins (never raise the caps
        // past what the machine can politely give).
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps(99, 4));
        assert_eq!(b.thread_cap, 3);
        // Polite never drops below 1 thread even with a tiny config.
        let b = budget(&inv(Some(4), 8, Some(true)), &load(8, 0.5), &syscap(), &caps(1, 2));
        assert_eq!(b.thread_cap, 1);
    }

    #[test]
    fn unconfigured_caps_fall_back_to_hardware() {
        let engine = libconfigreader::EngineSection::default();
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &engine);
        assert_eq!(b.verdict, Verdict::Full);
        assert_eq!(b.thread_cap, 3, "no config cap -> hardware capacity (4P - reserve 1)");
        assert_eq!(b.embed_batch, None, "no config batch -> leave it to the consumer");
    }

    #[test]
    fn disabled_config_reports_caps_without_sampling() {
        let cfg = libconfigreader::SyscapSection { enabled: Some(false), ..Default::default() };
        let b = budget(&inv(Some(4), 8, Some(true)), &load(0, 50.0), &cfg, &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Full);
        assert_eq!(b.thread_cap, 2);
        assert_eq!(b.embed_batch, Some(4));
        assert_eq!(b.mem_cap_bytes, u64::MAX);
        // Disabled means untouched: a configured cap ABOVE the hardware
        // usable-core estimate must survive (pre-syscap behavior).
        let b = budget(&inv(Some(4), 8, Some(true)), &load(0, 50.0), &cfg, &caps(8, 4));
        assert_eq!(b.thread_cap, 8);
    }

    #[test]
    fn machines_without_pe_split_use_logical_cores() {
        // Intel-style: no perf_cores -> usable = logical - reserve.
        // reserve = max(1, ceil(0.25 * 4)) = 1 -> usable 3.
        let b = budget(&inv(None, 4, None), &load(8, 0.5), &syscap(), &caps(8, 4));
        assert_eq!(b.thread_cap, 3);
    }

    #[test]
    fn tiny_machine_still_gets_one_core() {
        let b = budget(&inv(Some(1), 2, None), &load(8, 0.5), &syscap(), &caps(8, 4));
        assert_eq!(b.thread_cap, 1, "reserve must never eat the last core");
    }

    fn caps_no_intra() -> libconfigreader::EngineSection {
        libconfigreader::EngineSection {
            intra_threads: None,
            inter_threads: Some(1),
            embed_batch: Some(4),
            embed_sessions: None,
            exec_provider: None,
            provider_logging: None,
            max_cores: None,
            backend: None,
            no_cache: None,
        }
    }

    #[test]
    fn build_jobs_auto_follows_the_budget_ladder() {
        let spec = libconfigreader::BuildJobs::Auto;
        // Full -> capacity-driven (4 P-cores - reserve 1 = 3), not the
        // engine thread cap.
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Full);
        assert_eq!(b.capacity, 3);
        assert_eq!(build_jobs(&spec, &b), 3);
        // Polite (battery) -> half the build jobs, floor 2.
        let b = budget(&inv(Some(4), 8, Some(true)), &load(8, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Polite);
        assert_eq!(build_jobs(&spec, &b), 2);
        // Backoff (low RAM) -> 1.
        let b = budget(&inv(Some(4), 8, None), &load(2, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Backoff);
        assert_eq!(build_jobs(&spec, &b), 1);
        // Hold (nearly out of RAM) -> 1.
        let b = budget(&inv(Some(4), 8, None), &load(0, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Hold);
        assert_eq!(build_jobs(&spec, &b), 1);
    }

    /// This box: 6 P + 12 E cores, 18 logical. Every core counts, reserve =
    /// max(1, ceil(0.10 * 18)) = 2 -> capacity 6 + 12 - 2 = 16.
    fn inv_this_box() -> Inventory {
        let mut i = inv(Some(6), 18, None);
        i.cpu.eff_cores = Some(12);
        i
    }

    #[test]
    fn build_jobs_auto_is_aggressive_on_the_real_box() {
        let spec = libconfigreader::BuildJobs::Auto;
        // Full -> the whole capacity (16), no constant ceiling.
        let b = budget(&inv_this_box(), &load(12, 0.5), &syscap(), &caps(4, 8));
        assert_eq!(b.capacity, 16);
        assert_eq!(b.thread_cap, 4, "a configured cap still binds the runtime");
        assert_eq!(build_jobs(&spec, &b), 16);
        // Polite -> half the capacity, floor 2.
        let mut on_battery = inv_this_box();
        on_battery.power.on_battery = Some(true);
        let b = budget(&on_battery, &load(12, 0.5), &syscap(), &caps(4, 8));
        assert_eq!(b.verdict, Verdict::Polite);
        assert_eq!(build_jobs(&spec, &b), 8);
    }

    #[test]
    fn a_busy_compile_at_three_fourths_of_the_machine_is_still_full() {
        // The regression this policy exists for: on this box the OLD 0.70/core
        // threshold tripped at loadavg 12.6, which ONE background compile (or
        // the consumer's own threads) crosses while the desktop is idle — the
        // A/B runs measured `polite` on an idle desktop for exactly that
        // reason. loadavg 13.5 = 0.75/core must now be Full, and only the run
        // queue outgrowing the machine (>= 1.0/core) yields.
        let spec = libconfigreader::BuildJobs::Auto;
        let b = budget(&inv_this_box(), &load(12, 13.5), &syscap(), &caps_no_intra());
        assert_eq!(b.verdict, Verdict::Full);
        assert_eq!(b.capacity, 16);
        assert_eq!(effective_intra_threads(&caps_no_intra(), &b), 16);
        assert_eq!(build_jobs(&spec, &b), 16);
        // Past 1.0/core -> Polite (half), past 2.0/core -> Backoff (floor).
        let b = budget(&inv_this_box(), &load(12, 19.0), &syscap(), &caps_no_intra());
        assert_eq!(b.verdict, Verdict::Polite);
        assert_eq!(effective_intra_threads(&caps_no_intra(), &b), 8);
        let b = budget(&inv_this_box(), &load(12, 37.0), &syscap(), &caps_no_intra());
        assert_eq!(b.verdict, Verdict::Backoff);
        assert_eq!(effective_intra_threads(&caps_no_intra(), &b), 1);
    }

    #[test]
    fn build_jobs_auto_uses_capacity_when_caps_unset_and_has_no_ceiling() {
        let spec = libconfigreader::BuildJobs::Auto;
        // Unset engine.intra_threads: thread_cap falls back to capacity, and
        // the build ladder uses capacity either way.
        let b = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps_no_intra());
        assert_eq!(b.thread_cap, 3);
        assert_eq!(build_jobs(&spec, &b), 3);
        // Big machine without caps: capacity 28, handed out in full — there
        // is no hardcoded ceiling; Backoff/Hold are the RAM guard.
        let b = budget(&inv(Some(32), 32, None), &load(64, 0.5), &syscap(), &caps_no_intra());
        assert_eq!(b.thread_cap, 28);
        assert_eq!(build_jobs(&spec, &b), 28);
    }

    #[test]
    fn effective_threads_derive_from_the_budget_and_config_can_only_lower() {
        // Full on the fixture box: capacity 3 (4 P − reserve 1).
        let full = budget(&inv(Some(4), 8, None), &load(8, 0.5), &syscap(), &caps_no_intra());
        assert_eq!(effective_intra_threads(&caps_no_intra(), &full), 3, "unset -> capacity");
        assert_eq!(effective_intra_threads(&caps(2, 4), &full), 2, "config under budget wins");
        assert_eq!(effective_intra_threads(&caps(99, 4), &full), 3, "config can never raise");
        assert_eq!(effective_inter_threads(&caps_no_intra(), &full), 1, "unset inter -> 1");
        // Polite (battery): the derived value shrinks with the budget.
        let polite = budget(&inv(Some(4), 8, Some(true)), &load(8, 0.5), &syscap(), &caps_no_intra());
        assert_eq!(polite.thread_cap, 1);
        assert_eq!(effective_intra_threads(&caps_no_intra(), &polite), 1);
        // Backoff: floored at one.
        let backoff = budget(&inv(Some(4), 8, None), &load(2, 0.5), &syscap(), &caps(8, 4));
        assert_eq!(effective_intra_threads(&caps(8, 4), &backoff), 1);
        assert_eq!(effective_inter_threads(&caps(8, 4), &backoff), 1);
    }

    #[test]
    fn build_jobs_fixed_is_the_human_override() {
        let b = budget(&inv(Some(4), 8, None), &load(0, 0.5), &syscap(), &caps(2, 4));
        assert_eq!(b.verdict, Verdict::Hold);
        // Override wins even under Hold; zero clamps to one.
        assert_eq!(build_jobs(&libconfigreader::BuildJobs::Fixed(6), &b), 6);
        assert_eq!(build_jobs(&libconfigreader::BuildJobs::Fixed(0), &b), 1);
    }

    #[test]
    fn build_jobs_now_is_consistent_with_the_budget_it_returns() {
        let (b, jobs) = build_jobs_now(&syscap(), &caps(2, 4));
        assert!(jobs >= 1);
        assert_eq!(jobs, build_jobs(&syscap().build_jobs(), &b));
    }
}
