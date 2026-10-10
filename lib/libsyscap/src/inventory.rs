//! Static host inventory: what the machine HAS. Probed via syscalls and
//! frameworks only — this crate never spawns a process (`system_profiler` /
//! `nvidia-smi` tricks from onnxwrapperrs' older probe are deliberately not
//! carried over: the gui must stay a zero-children binary).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Static hardware snapshot of the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    pub cpu: CpuInfo,
    pub mem: MemInfo,
    pub gpus: Vec<Gpu>,
    pub accel: AccelInfo,
    pub power: PowerInfo,
    pub disk: DiskInfo,
}

impl Inventory {
    /// Probe the host fresh. Milliseconds-cheap (sysctls + frameworks, no
    /// subprocesses); prefer [`Inventory::global`] when one snapshot is all
    /// you need.
    pub fn probe() -> Inventory {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        let cpus = sys.cpus();
        let brand = cpus
            .first()
            .map(|c| c.brand().trim().to_string())
            .unwrap_or_default();
        let logical = cpus.len().max(1) as u32;
        let physical = sysinfo::System::physical_core_count().unwrap_or(0).max(1) as u32;
        let vendor = CpuVendor::from_brand(&brand);
        let (perf_cores, eff_cores) = pe_cores(vendor);
        let total = sys.total_memory();
        Inventory {
            cpu: CpuInfo {
                vendor,
                brand,
                logical,
                physical,
                perf_cores,
                eff_cores,
                simd: detect_simd(),
            },
            mem: MemInfo {
                total_bytes: total,
            },
            gpus: detect_gpus(),
            accel: AccelInfo {
                ane_present: Some(vendor == CpuVendor::Apple && cfg!(target_arch = "aarch64")),
                coreml_available: cfg!(target_os = "macos"),
            },
            power: PowerInfo {
                on_battery: detect_on_battery(),
            },
            disk: DiskInfo {
                home_free_bytes: free_bytes(&home_dir()).unwrap_or(0),
            },
        }
    }

    /// Process-wide cached snapshot (the hardware does not change mid-run).
    pub fn global() -> &'static Inventory {
        static INV: OnceLock<Inventory> = OnceLock::new();
        INV.get_or_init(Inventory::probe)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CpuVendor {
    Apple,
    Intel,
    Amd,
    Unknown,
}

impl CpuVendor {
    fn from_brand(brand: &str) -> Self {
        let b = brand.to_lowercase();
        if b.contains("apple") {
            CpuVendor::Apple
        } else if b.contains("intel") {
            CpuVendor::Intel
        } else if b.contains("amd") {
            CpuVendor::Amd
        } else {
            CpuVendor::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub vendor: CpuVendor,
    pub brand: String,
    pub logical: u32,
    pub physical: u32,
    /// Fast performance cores on Apple Silicon (`hw.perflevel0`); `None` when
    /// the platform exposes no P/E split (Intel macs, Linux, Windows).
    pub perf_cores: Option<u32>,
    /// Efficiency cores on Apple Silicon (`hw.perflevel1`).
    pub eff_cores: Option<u32>,
    pub simd: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemInfo {
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub vendor: GpuVendor,
    pub name: String,
    /// Practical device memory budget. On Apple Silicon (unified memory) this
    /// is Metal's `recommendedMaxWorkingSetSize`, not a separate VRAM size.
    pub mem_budget_bytes: Option<u64>,
    pub low_power: bool,
    pub headless: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelInfo {
    /// Apple Neural Engine. There is no direct query, so this is inferred:
    /// `Some(true)` on Apple Silicon, `Some(false)` on non-Apple hosts.
    pub ane_present: Option<bool>,
    pub coreml_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowerInfo {
    /// `Some(true)` when a battery is present and discharging, `Some(false)`
    /// when a battery is present but charging/full, `None` when there is no
    /// battery (desktop / unknown).
    pub on_battery: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiskInfo {
    /// Free bytes on the disk hosting the home dir — where `~/.cache` and
    /// therefore the model/vector caches live. `0` = unknown.
    pub home_free_bytes: u64,
}

fn pe_cores(vendor: CpuVendor) -> (Option<u32>, Option<u32>) {
    #[cfg(target_os = "macos")]
    {
        if vendor == CpuVendor::Apple {
            return (
                sysctl_u32("hw.perflevel0.logicalcpu"),
                sysctl_u32("hw.perflevel1.logicalcpu"),
            );
        }
    }
    #[allow(unreachable_code)]
    (None, None)
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Option<u32> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut value: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: `cname` is a valid NUL-terminated string and oldp/oldlenp
    // describe a writable u32 slot; sysctlbyname only writes on success.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::addr_of_mut!(value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len as usize == std::mem::size_of::<u32>()).then_some(value)
}

/// Always empty — see the note on [`Gpu`].
///
/// GPU enumeration used to shell through the `metal` crate here. It was removed
/// (2026-10-04): nothing ever *read* the result. The budget in `budget.rs`
/// decides from cores, memory and battery only, no DTO or serve command carries
/// a GPU, and the gui never saw one. Meanwhile `metal` drags in `block 0.1.6`,
/// whose `static _NSConcreteStackBlock: Class` trips rustc's
/// `uninhabited_static` future-incompat lint and will become a hard error. There
/// is no fix available: `block` has exactly one published version (0.1.6) and
/// every `metal` release from 0.29 to 0.33 depends on it.
///
/// Reinstating this means either a small `objc2`-style FFI shim for the three
/// MTLDevice facts, or accepting the lint until `metal` drops `block`. Re-add it
/// when something actually consumes [`Inventory::gpus`].
fn detect_gpus() -> Vec<Gpu> {
    Vec::new()
}

fn detect_simd() -> Vec<String> {
    let mut simd = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        simd.push("neon".into());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            simd.push("avx512".into());
        }
        if is_x86_feature_detected!("avx2") {
            simd.push("avx2".into());
        }
        if is_x86_feature_detected!("sse4.1") {
            simd.push("sse4.1".into());
        }
    }
    simd
}

fn detect_on_battery() -> Option<bool> {
    let manager = starship_battery::Manager::new().ok()?;
    let mut present = false;
    let mut discharging = false;
    for battery in manager.batteries().ok()? {
        let Ok(b) = battery else { continue };
        present = true;
        if matches!(b.state(), starship_battery::State::Discharging) {
            discharging = true;
        }
    }
    if present { Some(discharging) } else { None }
}

/// Free bytes on the filesystem containing `path`. `None` = unknown.
pub fn free_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let c = CString::new(path.to_str()?).ok()?;
        let mut stat: std::mem::MaybeUninit<libc::statvfs> = std::mem::MaybeUninit::zeroed();
        // SAFETY: `stat` points at a zeroed statvfs buffer the call fills.
        if unsafe { libc::statvfs(c.as_ptr(), stat.as_mut_ptr()) } != 0 {
            return None;
        }
        let stat = unsafe { stat.assume_init() };
        Some((stat.f_bavail as u64) * (stat.f_bsize as u64))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_plausible_host() {
        let inv = Inventory::probe();
        assert!(inv.cpu.logical >= 1);
        assert!(inv.cpu.physical >= 1);
        assert!(inv.mem.total_bytes > 0);
        assert!(inv.disk.home_free_bytes > 0);
        assert!(matches!(
            inv.cpu.vendor,
            CpuVendor::Apple | CpuVendor::Intel | CpuVendor::Amd | CpuVendor::Unknown
        ));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_exposes_pe_split() {
        let inv = Inventory::probe();
        assert_eq!(inv.cpu.vendor, CpuVendor::Apple);
        assert!(matches!(inv.cpu.perf_cores, Some(p) if p >= 1));
        assert!(matches!(inv.cpu.eff_cores, Some(e) if e >= 1));
        assert_eq!(inv.accel.ane_present, Some(true));
        assert!(inv.accel.coreml_available);
    }

    #[test]
    fn global_is_cached() {
        assert!(std::ptr::eq(Inventory::global(), Inventory::global()));
    }
}
