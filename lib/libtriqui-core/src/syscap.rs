//! System-capacity budget for the cascade (libsyscap design §7.3, M5c).
//!
//! The progressive-ranking cascade is the one long-running consumer that must
//! adapt to a busy machine: its waves shrink under load and pause entirely
//! when the verdict is `hold`. The `triq.yaml` `cascade.k1`/`k2` (and
//! `engine.embed_batch`) stay the ceiling — the budget can only lower them.
//! Pull-based: the caller samples (`budget_now`) at every wave, so waves
//! scale down AND back up as the machine's load changes; no watcher thread.
//!
//! The `holding` reason code lives here so gui, serve, and tests cannot
//! drift: `RankState.reason` / `RankProgress.reason` carry
//! [`HOLDING`] exactly when [`holding_code`] says so, and the gui i18n-maps
//! it to a visible notice. Polite/Backoff stay silent by design.

use libconfigreader::{EngineConfig, SyscapConfig};
use libsyscap::{budget, Budget, LoadSampler, LoadSnapshot};

/// Reason code carried by `RankState.reason` / `RankProgress.reason` while
/// the system budget holds heavy work back (waves paused). The gui maps it
/// to a visible notice; it clears itself when the verdict lifts.
pub const HOLDING: &str = "holding";

/// The `holding` reason code for a budget — `Some` only for
/// [`libsyscap::Verdict::Hold`], `None` for Full/Polite/Backoff (those adapt
/// silently).
pub fn holding_code(b: &Budget) -> Option<String> {
    if b.verdict.heavy_ok() {
        None
    } else {
        Some(HOLDING.to_string())
    }
}

/// Sample the host once and return both the budget and the raw load sample
/// it was computed from (for DTOs that surface the load).
pub fn sample_budget(cfg: &SyscapConfig, caps: &EngineConfig) -> (Budget, LoadSnapshot) {
    let mut sampler = LoadSampler::new();
    let load: LoadSnapshot = sampler.sample(std::time::Duration::from_millis(0));
    let b = budget(libsyscap::Inventory::global(), &load, cfg, caps);
    (b, load)
}

/// Sample the host once and compute the budget. Fails soft: a probe error
/// inside libsyscap already degrades to conservative defaults, and a config
/// load failure upstream yields the `Default` sections (enabled syscap,
/// unset engine caps).
pub fn budget_now(cfg: &SyscapConfig, caps: &EngineConfig) -> Budget {
    sample_budget(cfg, caps).0
}

/// This wave's slice size: the configured `cascade.k1_embed_per_wave` scaled by
/// the budget's `wave_scale`, floored at 1 (a wave always makes *some* progress
/// unless the verdict is `hold` — the caller pauses instead of running
/// zero-slice waves). `Full` hands out the configured value untouched.
///
/// The judge pool is **not** scaled: it is frozen once per run, so the load
/// verdict has nothing to say about it — a pool that grew and shrank with the
/// machine would move every member's set-relative grade.
pub fn wave_budget(k1: usize, pool_size: usize, b: &Budget) -> crate::cascade::CascadeBudget {
    let scale = |k: usize| ((k as f64 * b.wave_scale).floor() as usize).max(1);
    crate::cascade::CascadeBudget {
        wave1_budget: scale(k1),
        pool_size,
    }
}

/// The effective embed batch from a budget: the budget's batch when it has
/// an opinion (a configured `engine.embed_batch`, scaled by the verdict),
/// else the thread cap — which is capacity-derived when the config sets no
/// caps (10 on this box at Full, halved under Polite, 1 under Backoff/Hold).
/// Never zero.
pub fn effective_embed_batch(b: &Budget) -> usize {
    b.embed_batch.unwrap_or_else(|| b.thread_cap.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsyscap::Verdict;

    fn budget_of(verdict: Verdict, wave_scale: f64, embed_batch: Option<usize>) -> Budget {
        Budget {
            verdict,
            thread_cap: 2,
            embed_batch,
            wave_scale,
            mem_cap_bytes: 0,
            capacity: 2,
        }
    }

    #[test]
    fn holding_code_only_for_hold() {
        assert_eq!(holding_code(&budget_of(Verdict::Hold, 0.0, None)).as_deref(), Some(HOLDING));
        assert_eq!(holding_code(&budget_of(Verdict::Full, 1.0, None)), None);
        assert_eq!(holding_code(&budget_of(Verdict::Polite, 0.5, None)), None);
        assert_eq!(holding_code(&budget_of(Verdict::Backoff, 0.25, None)), None);
    }

    #[test]
    fn budget_now_fails_soft_on_default_sections() {
        // Defaults (enabled syscap, unset caps) on a real host must produce a
        // sane budget without panicking — whatever this box is doing right
        // now, the wave scale is one of the four known values and threads
        // never drop to zero.
        let b = budget_now(&SyscapConfig::default(), &EngineConfig::default());
        assert!(matches!(
            b.verdict,
            Verdict::Full | Verdict::Polite | Verdict::Backoff | Verdict::Hold
        ));
        assert!(b.thread_cap >= 1);
    }

    #[test]
    fn effective_embed_batch_prefers_the_budget_and_floors_at_one() {
        let b = budget_of(Verdict::Polite, 0.5, Some(4));
        assert_eq!(effective_embed_batch(&b), 4, "a configured batch scales via the budget");
        let none = budget_of(Verdict::Full, 1.0, None);
        assert_eq!(effective_embed_batch(&none), 2, "no config batch -> the thread cap");
    }

    #[test]
    fn wave_budget_scales_the_slice_but_never_the_pool() {
        // Full: the configured slice, untouched.
        let w = wave_budget(8, 4, &budget_of(Verdict::Full, 1.0, None));
        assert_eq!((w.wave1_budget, w.pool_size), (8, 4));
        // Polite halves, Backoff quarters — but the pool is frozen per run, so a
        // busy machine must not shrink it: the win-rates in it are set-relative,
        // and a pool that moved between waves would move every member's grade.
        let w = wave_budget(8, 4, &budget_of(Verdict::Polite, 0.5, None));
        assert_eq!((w.wave1_budget, w.pool_size), (4, 4));
        let w = wave_budget(8, 4, &budget_of(Verdict::Backoff, 0.25, None));
        assert_eq!((w.wave1_budget, w.pool_size), (2, 4));
        // The floor: a tiny slice never collapses to a zero-slice wave (the caller
        // pauses on hold instead of scaling to 0).
        let w = wave_budget(1, 1, &budget_of(Verdict::Backoff, 0.25, None));
        assert_eq!((w.wave1_budget, w.pool_size), (1, 1));
        // Rounding: 5 * 0.5 = 2.5 floors to 2.
        let w = wave_budget(5, 5, &budget_of(Verdict::Polite, 0.5, None));
        assert_eq!((w.wave1_budget, w.pool_size), (2, 5));
        // A pool of nobody is a configuration, and stays a pool of nobody.
        let w = wave_budget(8, 0, &budget_of(Verdict::Full, 1.0, None));
        assert_eq!((w.wave1_budget, w.pool_size), (8, 0));
    }
}
