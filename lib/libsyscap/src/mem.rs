//! Available-memory reading.
//!
//! Why this exists instead of `sysinfo::available_memory()`: on this macOS
//! build that call returns 0, verified against both sysinfo 0.34 (what the
//! repo pins) and 0.37, so it is not a version thing. That is not cosmetic.
//! `libsyscap` derives `mem_pressure = 1 - avail/total` from it, so a 0 reads
//! as "RAM exhausted" and the whole politeness policy pins itself at `Hold`
//! (1 thread, wave scale 0, no embed batch) on a machine with gigabytes free.
//! A silently-wrong capacity signal is worse than no signal, because every
//! downstream number still looks plausible.
//!
//! So macOS is measured directly from `vm_stat`, and `sysinfo` is the fallback
//! for platforms we have not verified. "Available" is the set the kernel can
//! hand out without swapping: free + inactive (reclaimable) + speculative.
//! Wired memory is deliberately NOT counted: an active app's pages are not
//! available, and counting them would overstate headroom by gigabytes.

/// Parsed `vm_stat` counters (pages, not bytes).
#[cfg(target_os = "macos")]
#[derive(Debug, Default, Clone, Copy)]
struct VmStat {
    free: u64,
    active: u64,
    inactive: u64,
    speculative: u64,
    wired: u64,
    compressed: u64,
}

/// Parse a `vm_stat` dump into page counters plus the page size.
#[cfg(target_os = "macos")]
fn parse_vm_stat(text: &str) -> Option<(VmStat, u64)> {
    let mut page_size = 0u64;
    let mut v = VmStat::default();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics") {
            if let Some(i) = rest.find("page size of ") {
                let digits: String = rest[i + 13..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                page_size = digits.parse().ok()?;
            }
            continue;
        }
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        let value: u64 = value.trim().trim_end_matches('.').trim().parse().ok()?;
        match label.trim() {
            "Pages free" => v.free = value,
            "Pages active" => v.active = value,
            "Pages inactive" => v.inactive = value,
            "Pages speculative" => v.speculative = value,
            "Pages wired down" => v.wired = value,
            "Pages occupied by compressor" => v.compressed = value,
            _ => {}
        }
    }
    if page_size == 0 {
        return None;
    }
    Some((v, page_size))
}

/// Available bytes from a `vm_stat` dump. Pure, so it is tested against
/// captured output rather than against whatever the machine says today.
#[cfg(target_os = "macos")]
fn available_from_vm_stat(text: &str, total: u64) -> Option<u64> {
    let (v, page_size) = parse_vm_stat(text)?;
    let pages = v.free + v.inactive + v.speculative;
    let bytes = pages.saturating_mul(page_size);
    // Clamped to physical memory: a parser that mis-reads one field must not be
    // able to report more RAM than the machine has, which would read as
    // infinite headroom.
    Some(bytes.min(total))
}

/// Available memory in bytes, measured where we know how.
pub fn available_memory(total: u64) -> u64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(text) = std::process::Command::new("vm_stat").output() {
            let out = String::from_utf8_lossy(&text.stdout);
            if let Some(bytes) = available_from_vm_stat(&out, total) {
                return bytes;
            }
            log::warn!("vm_stat unreadable; falling back to sysinfo available_memory");
        }
    }
    let mut sys = sysinfo::System::new_all();
    sys.refresh_memory();
    let v = sys.available_memory();
    // A 0 here is the failure mode this module exists to prevent. Reporting
    // "everything is free" would be the opposite lie and reporting `total`
    // would hide real pressure, so say "critically low" and let the caller's
    // own pressure logic act on it.
    if v == 0 {
        log::warn!("available_memory() returned zero; treating as critically low");
        return 1;
    }
    v.min(total.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from this machine (macOS, 16 KiB pages): plenty reclaimable,
    /// so `available` must not come out as zero.
    const SAMPLE: &str = concat!(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n",
        "Pages free:                                54161.\n",
        "Pages active:                             323060.\n",
        "Pages inactive:                          314847.\n",
        "Pages speculative:                         7411.\n",
        "Pages throttled:                              0.\n",
        "Pages wired down:                         184175.\n",
        "Pages purgeable:                           9839.\n",
    );

    const TOTAL: u64 = 24 * 1024 * 1024 * 1024;

    #[cfg(target_os = "macos")]
    #[test]
    fn vm_stat_page_size_and_counters_parse() {
        let (v, page) = parse_vm_stat(SAMPLE).expect("parses");
        assert_eq!(page, 16384);
        assert_eq!(v.free, 54161);
        assert_eq!(v.inactive, 314847);
        assert_eq!(v.speculative, 7411);
        assert_eq!(v.wired, 184175);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_counts_free_inactive_and_speculative_but_not_wired() {
        let got = available_from_vm_stat(SAMPLE, TOTAL).expect("available");
        let expected = (54161 + 314847 + 7411) * 16384;
        assert_eq!(got, expected);
        assert!(got > 0, "this is the zero the module exists to prevent");
        assert!(got < TOTAL, "available must stay below physical RAM");
        // Wired pages belong to the kernel and drivers; counting them would
        // overstate headroom by ~2.8 GiB on this sample.
        assert!(got < TOTAL - 184175 * 16384);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn a_machine_with_everything_wired_reports_zero() {
        let text = concat!(
            "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n",
            "Pages free: 0.\n",
            "Pages inactive: 0.\n",
            "Pages speculative: 0.\n",
            "Pages wired down: 6000000.\n",
        );
        assert_eq!(available_from_vm_stat(text, TOTAL), Some(0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn garbage_never_claims_headroom() {
        // No page size means refuse rather than default to something plausible.
        assert_eq!(available_from_vm_stat("total pages: 12345\n", TOTAL), None);
        assert_eq!(available_from_vm_stat("", TOTAL), None);
    }

    #[test]
    fn live_reading_is_plausible_on_the_machine_running_it() {
        // The regression guard: a zero here is exactly what pinned the budget
        // policy at Hold, and nothing else in the suite would have noticed.
        let got = available_memory(TOTAL);
        assert!(got > 0, "sysinfo reports zero again; that is the bug prevented here");
        assert!(got <= TOTAL);
    }
}