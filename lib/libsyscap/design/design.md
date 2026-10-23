# libsyscap — host capability + live-load budget library (design)

Status: **draft for human review — no code written yet.**
Consumers: gui (through shortlist-core/serve + a libshortlistclient DTO), bench (bench1),
tests crate. Proposed home: `lib/libsyscap/` (edition 2021, default `src/lib.rs`,
libbuild auto-discovers it — no registration needed).

## 1. Problem

- `triq.yaml` `engine:` caps (`intra_threads` / `inter_threads` / `embed_batch`)
  are static and deliberately conservative (this box: 4 cores / 16 GB, datamaker +
  kilo running in background). They cannot see *when* the machine is actually idle
  (wasted capability) or *when* the user is active (annoyance risk).
- `lib/onnxwrapperrs/src/probe.rs` already does a **static** hardware probe
  (sysinfo: cpu vendor/cores, ram total/free, simd; gpus via `nvidia-smi` +
  `system_profiler` subprocesses; disk free). Gaps:
  - no **live load** signal at all (cpu %, loadavg, memory pressure),
  - no Apple Silicon **P/E core split**,
  - subprocess probes (`system_profiler` costs seconds) clash with the gui's
    "single binary, zero spawned children" invariant,
  - it lives behind onnxwrapperrs, so non-engine consumers would have to link ORT
    to read the host's specs.
- Wanted: one small shared lib that knows (a) what the host **has**, (b) what the
  host is **doing right now**, and (c) turns both into a **budget**: how many
  threads / batch size / wave size this process may use *now*. Not too low (use
  unused capability with breathing room), not too high (the user must not feel it).

## 2. Goals / non-goals

**Goals (v1)**
- Static inventory: cpu (logical, physical, P/E split on Apple Silicon, simd),
  RAM total, GPU list (count, names, memory *budget*), accelerator hints
  (Apple Neural Engine presence via chip identity, CoreML EP availability),
  power state (on battery vs AC), disk free at the cache dir.
- Live load: cpu utilization, loadavg 1/5/15, memory available/total, swap used,
  a memory-pressure score.
- **Budget policy as pure functions**: `Inventory + LoadSnapshot + SyscapConfig
  + configured engine caps → Budget`. Fully table-testable, no I/O.
- Config comes from `triq.yaml` only (new optional `syscap:` section via
  libconfigreader). No env-var config (NOAI.md).
- **Clamp invariant**: recommendations are always ≤ the caller's configured caps.
  The lib *recommends*; triq.yaml stays the enforcement (AGENTS.md: don't raise
  engine caps).
- No spawned processes, no panics on probe failure, no threads of its own (v1).

**Non-goals (v1)**
- GPU utilization sampling (macOS has no public API; IOReport is private).
- Process attribution ("who is loading the box").
- Thermal state (phase 2), migrating `onnxwrapperrs::probe` (phase 2), changing
  ikbuild's hardcoded 2-job parallelism, benchdriverrs orchestration parallelism,
  tuning any non-macOS platform (cross-platform via sysinfo, but only darwin is
  tuned/tested).

## 3. Alternatives considered

| Option | Verdict |
| --- | --- |
| Extend `onnxwrapperrs::probe` | rejected — load/budget is not an ONNX concern; every consumer would link ORT; onnxwrapperrs is the only crate allowed to touch ORT |
| Put it in shortlist-core | rejected — bench/tests need it without the gui domain |
| New small lib crate | **chosen** — contract-lib pattern, one focused job |

## 4. Module sketch / API

```rust
// ---- inventory.rs : static, probe once, cache in a OnceLock ----
pub struct Inventory {
    pub cpu:   CpuInfo,       // logical, physical, perf/eff cores (Option), vendor, brand, simd
    pub mem:   MemInfo,       // total_bytes
    pub gpus:  Vec<Gpu>,      // vendor, name, mem_budget (Option<u64>), low_power, headless
    pub accel: AccelInfo,     // ane_present: Option<bool>, coreml_available: bool
    pub power: PowerInfo,     // on_battery: Option<bool> (sysinfo Batteries; None = desktop)
    pub disk:  DiskInfo,      // cache_dir_free_bytes
}
impl Inventory {
    pub fn probe() -> Inventory;          // fresh probe (ms-cheap: sysctls + frameworks, no subprocesses)
    pub fn global() -> &'static Inventory; // cached OnceLock
}

// ---- load.rs : pull-based sampler; owns sysinfo::System (cpu% needs deltas) ----
pub struct LoadSampler { /* sysinfo::System + last-refresh instant */ }
pub struct LoadSnapshot {
    pub cpu_pct: f32,
    pub loadavg: [f64; 3],        // 1/5/15 min
    pub mem_total: u64, pub mem_available: u64,
    pub swap_used: u64,
    pub mem_pressure: f64,        // 0.0..1.0 (v1: derived; phase 2: mach memorystatus level)
}
impl LoadSampler {
    pub fn new() -> Self;
    pub fn sample(&mut self, min_interval: Duration) -> LoadSnapshot;
    // sysinfo needs >= MINIMUM_CPU_UPDATE_INTERVAL (200 ms) between refreshes
    // for a meaningful cpu%; shorter waits return the last snapshot.
}

// ---- budget.rs : pure policy, the heart of the crate ----
pub enum Verdict { Full, Polite, Backoff, Hold }
pub struct Budget {
    pub thread_cap:    usize,   // configured engine.intra_threads, or capacity when unset
    pub embed_batch:   Option<usize>, // configured engine.embed_batch, verdict-scaled
    pub wave_scale:    f64,     // multiplies cascade k1/k2 (0.0..=1.0, consumers floor it)
    pub mem_cap:       u64,     // bytes this process should not exceed right now
    pub capacity:      usize,   // P + E − reserve — what the hardware offers now
    pub verdict:       Verdict,
}
pub fn budget(inv: &Inventory, load: &LoadSnapshot, cfg: &SyscapConfig,
              caps: &EngineCaps /* caller's triq.yaml engine values */) -> Budget;
pub fn effective_intra_threads(caps: &EngineCaps, b: &Budget) -> usize; // cap or capacity, min
pub fn effective_inter_threads(caps: &EngineCaps, b: &Budget) -> usize; // cap or 1, min
```

**Default policy (all knobs overridable in `syscap:`)** — as re-arranged by
deviation 10:
- Full: free RAM ≥ max(2 GiB, 20 % of total) AND loadavg1m ≤ 1.0 × logical
  cores AND not discharging → hand out the full capacity (or the configured
  caps, if set — config only lowers).
- Headroom: capacity counts every core — `P + E` (E-cores are real cores; the
  old half-weight was a mixed-core throughput hedge that only cost idle
  throughput) minus `max(reserve_cores, reserve_core_frac × logical)` reserved
  for the user (floor 1). This box: 6 P + 12 E → 6 + 12 − 2 = 16. Machines
  without a P/E split use logical cores as the base.
- Polite: on battery (`on_battery_polite`, default on), or loadavg1m in
  (1.0 ..= 2.0) × logical → half the threads/batch, wave_scale 0.5.
- Backoff: free RAM < 20 % or loadavg1m > 2.0 × logical (`load_stall`) → floor
  values (1 thread, batch 1, wave_scale 0.25).
- Hold: free RAM < 5 % → verdict Hold; consumers defer heavy work (waves pause,
  expensive tests skip). Hold is the ONLY gear the user must see: the gui
  receives a `holding` reason code and shows a visible "paused: system low on
  memory" notice. Polite/Backoff adapt silently — no noise for the user.
- Sub-1.0 load is NOT politeness: one background compile — or the consumer's
  own threads, which are inside the same load average — crosses 0.7 × logical
  on an 18-core box while the desktop sits idle. Yielding there wasted idle
  cores and made an embedder throttle itself mid-wave.
- Verdicts step multiplicatively (½ / ×2) so the budget is stable; hysteresis is
  a phase-2 watcher concern — v1 consumers sample at natural job boundaries.

## 5. Config (`triq.yaml` + libconfigreader)

New optional `SyscapSection` in libconfigreader (Option fields, defaults as
consts next to it — same pattern as `DEFAULT_SCOUT_CHUNK_SIZE`). Absent section =
defaults. Sketch:

```yaml
syscap:
  enabled: true                 # false = budget == configured caps, no load sampled
  reserve_cores: 1              # cores kept for the user
  reserve_core_frac: 0.10       # or 10% of logical cores, whichever is larger
  min_free_mem_frac: 0.20       # below this free-RAM fraction -> Backoff
  hold_free_mem_frac: 0.05      # below this -> Hold
  load_backoff: 1.00            # loadavg1m/logical > this -> Polite (queue outgrows the machine)
  load_stall: 2.00              # ...> this -> Backoff (twice oversubscribed)
  on_battery_polite: true
  sample_min_interval_ms: 250   # sysinfo cpu% needs >=200ms between refreshes
```

## 6. Dependencies

- `sysinfo` (already in the tree at 0.34/0.36 — pick one, pin it): cpu/mem/loadavg,
  batteries (macOS), disk? (no — reuse the statvfs approach from probe.rs).
- `libc`: `sysctlbyname` for `hw.perflevel0/1.logicalcpu`, `hw.ncpu`,
  `hw.physicalcpu`, `machdep.cpu.brand_string`; `getloadavg` fallback.
- `metal` (macOS-only dep, `cfg(target_os = "macos")`): `Device::all()` → gpu
  count/names, `recommended_max_working_set_size()` (= practical memory budget on
  unified memory), `is_low_power`/`is_headless`. Zero subprocesses, milliseconds.
  ANE is not directly queryable: inferred `ane_present` from Apple-Silicon brand.
- `serde` (DTOs), `log`. No tokio, no new heavy deps.

**Hard rule: the lib never spawns a process.** `nvidia-smi`/`system_profiler`
subprocess tricks from probe.rs are not carried over (gui invariant). On this Mac
that loses nothing: Metal covers GPU enumeration.

## 7. Consumer wiring (each its own review-sized step)

1. **tests crate**: gate expensive tests on load — alongside `IK_SKIP_EXPENSIVE`,
   skip when `budget(...).verdict` is `Backoff`/`Hold` (machine already busy).
   Reported in test output, not silent.
2. **bench1**: record `verdict` + key budget numbers into the result yaml
   (provenance: "ran in polite mode"); pick `embed_batch`/threads via `min(budget,
   config)` so a busy machine can't distort or stall a bench run. Design note in
   `bench/bench1/design/` updated first (design-first rule).
3. **gui (via shortlist-core + serve + libshortlistclient)**: cascade uses
   `wave_scale` to shrink k1/k2 under load and restore when idle (config values
   stay the ceiling); a `system_budget` DTO command surfaces verdict + load for
   the gui to i18n-map. Domain-derived state stays backend-side per
   `gui/AGENTS.md`; gui links only the contract lib. A general *load indicator*
   is phase 2, but surfacing `hold` is a v1 requirement: while waves are paused,
   rank_state / progress carries a `holding` reason code and the gui shows a
   visible notice ("paused: low system memory"). Computed backend-side, text via
   i18n (standard reason-code pattern); it clears itself when the verdict lifts.

## 8. Testing

- `budget::tests::policy_table_*` — synthetic Inventory/Load → expected
  Verdict/Budget (the core value; pure, fast).
- `budget::tests::never_exceeds_configured_caps` — randomized configs, clamp
  invariant.
- `inventory::tests::darwin_probe_plausible` (cfg macos) — logical ≥ 1,
  total RAM > 0, ≥ 1 GPU, plausible ranges (no exact-value asserts).
- `load::tests::sampler_returns_sane_ranges` — two samples ≥ interval apart,
  cpu_pct in 0..=100, mem_available ≤ total.
- libconfigreader: `syscap:` roundtrip + defaults-when-absent.
- shortlist-core (M5c): `Budget{verdict: Hold} → rank_state reason code
  `holding`` mapping test, and that non-Hold verdicts produce no holding code
  (assert on reported behavior, not internals).
- All cheap (< 30 s) → they always run under dev.py's cheap/expensive policy. No
  engine, no downloads, no expensive tests.

## 9. Invariants (checked in review + tests)

1. No subprocesses, no panics on probe failure (Option fields + conservative
   defaults), no writes to config or disk (read-only probing), no env-var config.
2. Budget output ≤ caller-configured caps, always. The lib never raises
   `engine:` caps; enforcement stays in triq.yaml consumers.
3. No threads of its own in v1 (pull-based). Phase 2 watcher is opt-in.
4. Anything that only the gui needs goes through the libshortlistclient contract,
   never a direct domain-lib link.

## 10. Milestones (check these off in order)

- [x] **M0 — design approved** (this doc; Q5 resolved: hold must be surfaced)
- [x] **M1 — scaffold**: `lib/libsyscap` (edition 2021), `.gitignore`,
      `-w lib/libsyscap/src` in `watch-build.sh`. libbuild auto-discovers the
      crate; nothing to register.
- [x] **M2 — inventory**: sysinfo base + sysctl P/E split + Metal GPUs +
      battery (`starship-battery`; sysinfo 0.34 dropped its battery module) +
      disk free. Real-probe tests green on this box.
- [x] **M3 — load sampler**: `LoadSampler` + rate-limit + sane-range tests.
- [x] **M4 — budget + config**: policy fns + `SyscapSection` in libconfigreader
      + `syscap:` docs in triq.yaml (commented defaults) + 11 policy table
      tests (verdict ladder, clamp invariant, disabled mode, no-P/E machines).
- [x] **M5a — consumers: tests-crate busy-gate**: the two expensive test
      binaries' `expensive_disabled()` helpers now also skip when
      `libsyscap::machine_busy()` (verdict Backoff/Hold), reported on stderr.
- [x] **M5b — consumers: bench1** provenance/knobs: `run` probes once at
      start (`meta::syscap::plan`), clamps the engine spawn via
      `spawn_inprocess_with_engine` (new libmodelcallerclient variant;
      `spawn_inprocess` keeps its meaning), records `meta.syscap`
      (verdict/thread_cap/embed_batch/wave_scale/load) into the result yaml.
      Hold does not refuse the run (Q6 proposal adopted). Bench design
      02 §2.9 documents it.
- [x] **M5c — consumers: gui**: `shortlist_core::syscap` (wave scaling,
      `holding` code, embed-batch clamp) + the serve runner samples per wave
      (scales down AND restores; `hold` pauses waves with a `holding`
      progress event and re-checks every 5 s), `get_rank_state` reports
      `reason: "holding"`, the `system_budget` command runs through
      libshortlistclient → serve → tauri → bindings, and CandidatesPane
      shows the i18n'd hold notice (`rank.holding`, en/fr/ar).
- [ ] **M6 — phase-2 backlog (not v1)**: opt-in watcher thread with hysteresis,
      thermal state (`NSProcessInfo.thermalState`), real memory-pressure via mach,
      migrate `onnxwrapperrs::probe` onto this crate (onnxwrapperrs may depend on
      libsyscap; never the reverse), benchdriverrs/ikbuild parallelism
      (**build-parallelism slice done — see §12**; watcher / thermal / mach /
      probe migration remain).

### Implementation deviations from this doc (M1–M5a)

1. **Dependency direction**: `libsyscap` depends on `libconfigreader` (leaf
   stays leaf); `budget()` takes `&SyscapSection` + `&EngineSection` directly
   instead of a separately-resolved `SyscapConfig` — no duplicated defaults,
   same pattern as `modelcaller` depending on the config reader.
2. **Battery probing**: sysinfo 0.34 removed `Batteries`; the maintained
   wrapper it used internally arrived as `starship-battery = "0.10"` (unix
   target only). `metal = "0.33"`, `sysinfo = "0.34"` (same major as
   onnxwrapperrs already pins).
3. **Budget shape**: `embed_batch: Option<usize>` (None = config set none —
   keep the consumer's own default; only Polite/Backoff force a concrete
   floor), `mem_cap_bytes` advisory ceiling per verdict (Full keeps free RAM
   beyond the user reserve, Polite ½, Backoff ¼, Hold 0).
4. **M5b**: bench1 does not batch by `engine.embed_batch` today, so the
   batch clamp is recorded in `meta.syscap` but not applied there — the
   consumer that actually batches (shortlist-core `EMBED_BATCH`) applies it
   in M5c.
5. **M5c**: the `system_budget` DTO carries `verdict` as the snake_case
   code string (contract-lib style, like `judge_mode`) plus a `holding`
   bool mirror; the load sample rides along for provenance/phase-2 badge.
   Small jobs (≤ `small_job_threshold`) skip wave *scaling* (one wave of
   everything anyway) but not the `hold` pause — low memory stops every
   job. The embed-batch clamp samples once per process (the value is read
   into a `LazyLock`); wave sizes re-sample every wave, which is the
   user-visible adaptation. The runner's pause loop is not cheap-testable
   (needs a real `hold`) — the pure pieces (scaling table, reason mapping,
   DTO shape) are, plus a wire test for `system_budget`.
6. **M6 build slice**: build jobs inherit the *engine* thread budget
   (`Full` = configured `engine.intra_threads`, not raw usable cores), so
   the §9 clamp invariant "budget output always ≤ configured caps" holds
   for builds too. With this box's `intra_threads: 2` that means auto
   = 2 when idle (exactly today's hardcoded value) and 1 under
   Polite/Backoff/Hold — the win is yielding on a busy machine, not
   speeding up idle builds; `build_jobs: N` is the explicit escape hatch.
7. **Disabled = truly untouched**: the `enabled: false` branch originally
   still clamped configured caps by the usable-core estimate
   (`min(usable)`), contradicting "reports the configured engine caps
   untouched" — and making `enabled: false` useless as an A/B baseline
   (this box: 6 P + 12 E cores, 25 % reserve → usable 1 → `-j 1` even
   disabled). It now honors configured caps verbatim (unset still falls
   back to `usable`), test-locked. Timed A/B on the same touch-rebuild
   cascade incl. all test suites: disabled (`-j 2`) 77.0 s vs enabled
   while the desktop is busy (Polite → `-j 1`) 116.7 s.
8. **Aggression rework (user direction: "this machine has much to offer")**:
   the original headroom was doubly lenient — 25 % of *logical* cores
   (5 of 18 here) reserved out of only the 6 P-cores → capacity 1, and
   `engine.intra_threads: 2` then capped every runtime consumer below
   even that. Now: capacity counts E-cores at half weight
   (`P + E/2 − reserve`), `reserve_core_frac` default drops to 0.10
   (capacity 10 here), `triq.yaml` engine caps are raised deliberately
   (`intra_threads: 4`, `embed_batch: 8` — the low caps predate libsyscap
   and were guarding against *unbounded* ORT threads; the budget now
   provides that enforcement dynamically, halving under Polite and
   flooring under Backoff), and the build ladder uses capacity directly
   (8/4/1), not the engine cap. Resolves open question 4 (defaults) by
   replacement. `enabled: false` still reproduces the literal configured
   caps for A/B comparability (deviation 7).
9. **No hardcoded machine-size defaults remain (user direction: "remove
   any hard coded default everywhere")**: the `triq.yaml` engine caps are
   unset — `intra_threads`/`embed_batch` now DERIVE from the budget
   (`libsyscap::effective_intra_threads`/`effective_inter_threads`;
   shortlist-core `effective_embed_batch`), so Full uses the whole
   capacity (10 threads / batch 10 here), Polite halves, Backoff floors.
   `modelcaller::run_stdio_over` gained the `syscap:` section and resolves
   unset caps itself before spawning a session — previously unset caps
   fell through to ONNX Runtime's own default (one worker per physical
   core), an unpoliced machine-size default hiding outside our config.
   `DEFAULT_EMBED_BATCH` (4) and `BUILD_JOBS_CAP` (8) are deleted; the
   build ladder is `Full` = capacity, Polite = half (floor 2),
   Backoff/Hold = 1. `inter_threads: 1` stays the one pinned value — a
   model-shape choice, not a machine-size one. The engine contract test
   asserts derivation properties (≥ 1, ≤ hardware, ≤ configured cap)
   instead of pinning numbers.
10. **Ladder re-arranged after the A/B (user: "this desktop was def not
   busy. re arrange auto to be more aggressive")**: the A/B timed two legs
   back-to-back and BOTH read `polite` on an idle desktop — diagnosis: the
   box is a MacBook Pro (M5 Pro, 6 P + 12 E) and the `polite` verdict came
   from `on_battery_polite`, but the load arms were miscalibrated too:
   `load_backoff` 0.70/core means "yield once 70 % of the machine is busy",
   which on 18 cores is loadavg 12.6 — one background compile crosses that,
   and so does the engine's own wave (its threads are inside the same load
   average, so a full-width wave could halve itself). Three changes:
   (a) `load_backoff` 0.70 → **1.0** (yield only when the run queue exceeds
   the machine) and the Backoff CPU arm 1.0 → **2.0**, now the configurable
   `load_stall` knob instead of a hardcoded constant;
   (b) capacity counts E-cores at **full** weight again: `P + E − reserve`
   = 16 here (was 10) — the half-weight was a mixed-core throughput hedge,
   but the verdict ladder, not the ceiling, is what yields when the machine
   is busy;
   (c) `on_battery_polite` stays a knob (default true); on battery the box
   runs Polite (8 threads / `-j 8`), on AC it gets the full 16. Timed on the
   battery box after the change: `serve_gui_contract` 219 s → 189 s,
   `try_model_benchmarks` 23 s → 18 s (both vs the previous capacity-10
   polite leg), 39/39 green. Locked by
   `budget::tests::a_busy_compile_at_three_fourths_of_the_machine_is_still_full`.
   Measured after that change (`serve_gui_contract`, 39 tests, all green each
   time): 308 s on the old hardcoded defaults → 219 s at capacity 10 (polite)
   → 167 s at capacity 16 (full) → 140 s at capacity 18 with four concurrent
   full-graph LTO rebuilds sharing the box. `embed_batch: 64` (stress-only)
   was *slower* than the derived 16 (170–195 s) — activation memory and
   attention cost grow faster than the batch saves, so the derived
   thread-cap-sized batch is the better default.

## 12. M6 slice — build parallelism (`syscap.build_jobs`)

`ikbuild` and `benchdriverrs` hardcode cargo `-j 2` (AGENTS: bound RAM on
this 4-core/16 GB box). This slice replaces the constant with the same
budget every other consumer uses — one machine policy, no special cases.

**Config** (`triq.yaml`, parsed in `libconfigreader::SyscapSection`):

```yaml
syscap:
  build_jobs: auto   # auto (default) | fixed count
```

`BuildJobs` is `Auto` (default, absent section included) or `Fixed(u32)`;
an unrecognized value is a config error, not a silent fallback.

**Policy** (`libsyscap::build_jobs`, pure + table-tested):

- `Fixed(n)` → `max(1, n)` verbatim — the human override wins.
- `Auto` → capacity-driven ladder (builds may use E-cores — compilation
  parallelizes, unlike ORT sessions, so this is deliberately decoupled from
  `engine.intra_threads`):
  - `Full` → the whole `capacity` — this box: 10.
  - `Polite` → `max(2, capacity / 2)` — this box: 5.
  - `Backoff` / `Hold` → `1` (links are the RAM peak; a nearly spent box
    builds one unit at a time).
- There is NO constant ceiling (deviation 9 removed `BUILD_JOBS_CAP`):
  free-RAM pressure flips the verdict to Backoff/Hold on the *next*
  invocation (builds sample once at start — the per-crate re-sample is
  watcher territory), and `build_jobs: N` is the explicit override.

`build_jobs_now(cfg, caps)` is the one-shot convenience both drivers call:
fresh `Inventory::global()` + `LoadSampler` sample → `budget()` → policy,
returning `(Budget, u32)` so callers can log the verdict.

**Callers** — `libbuild` stays dependency-free and untouched; only the
`jobs: u32` argument changes:

- `ikbuild`: new optional `--config-file PATH` flag (default
  `<repo_root>/triq.yaml`). If the config cannot be loaded it warns on
  stderr and falls back to today's `2` — the build tool must stay usable
  even with a broken config. Logs `syscap: verdict=… jobs=N`.
- `benchdriverrs`: `build_all_binaries(cfg, log)` resolves the same way
  and logs the verdict. The bench **matrix itself stays strictly
  sequential** — run wall-times are recorded results, and concurrent
  benches would distort them. Only the build step adapts.

**Out of scope / deferred**: per-crate parallelism across crates
(`libbuild` builds sequentially by design; `-j` bounds intra-crate
units); dev.py test timings are unaffected (crates' tests run one at a
time regardless of `-j`).

## 11. Open questions — please check

1. **Crate name**: `libsyscap` (proposed) vs `libsysbudget` vs `libhoststate`?
   (Dir move is trivial before M1.)
2. **`metal` crate dependency** for zero-spawn GPU probing — OK? (Alternative:
   keep `system_profiler` subprocess, bench-only, gui gets no GPU info. Not
   recommended.)
3. **Battery politeness** (`on_battery_polite`) — wanted, or desktop-only box?
4. ~~**Default headroom numbers** (§4): reserve 1 core / 25 %, 20 % free-RAM floor,
   backoff at loadavg 0.7×logical — sane starting point?~~ **superseded** by
   deviation 8: E-cores count at half weight, reserve frac 0.10, engine caps
   raised — see there.
5. ~~gui v1: backend-only or visible badge?~~ **resolved**: backend-only scaling
   in v1, plus a mandatory visible notice when the verdict is `hold` (§4, §7.3);
   general load badge stays phase 2.
6. Should bench **pause** entirely on `Hold`, or just floor its knobs (proposed)?
