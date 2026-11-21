#[cfg_attr(not(target_os = "macos"), allow(unused_imports))]
use serde_json::Value;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Capabilities {
    pub os: String,
    pub arch: String,
    pub cpu: CpuInfo,
    pub ram_total: u64,
    pub ram_free: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpus: Vec<GpuInfo>,
    pub disk_cache_free: u64,
    pub selected_ep: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuInfo {
    pub vendor: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub simd: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub vendor: String,
    pub name: String,
    pub vram_total: u64,
    pub fp16: bool,
    pub int8: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_capability: Option<String>,
}

pub fn probe_capabilities() -> Capabilities {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let physical_cores = sysinfo::System::physical_core_count().unwrap_or(0) as u32;
    let logical_cores = sys.cpus().len() as u32;

    let vendor = detect_cpu_vendor();
    let simd = detect_simd();

    let ram_total = sys.total_memory();
    // Not `sysinfo::available_memory()`: it returns 0 on this macOS build,
    // which would report a machine with gigabytes free as having none.
    // libsyscap owns that measurement for the whole repo (see its `mem`
    // module) so the probe and the budget policy cannot disagree.
    let ram_free = libsyscap::mem::available_memory(ram_total);

    let gpus = detect_gpus();

    let selected_ep = select_ep(&gpus, &os);

    let disk_cache_free = disk_cache_free();

    Capabilities {
        os,
        arch,
        cpu: CpuInfo {
            vendor,
            physical_cores,
            logical_cores,
            simd,
        },
        ram_total,
        ram_free,
        gpus,
        disk_cache_free,
        selected_ep,
    }
}

fn detect_cpu_vendor() -> String {
    let info = sysinfo::System::new_all();
    let brand = info
        .cpus()
        .first()
        .map(|c| c.brand().to_lowercase())
        .unwrap_or_default();
    if brand.contains("intel") {
        "intel".into()
    } else if brand.contains("amd") || brand.contains("authenticamd") {
        "amd".into()
    } else if brand.contains("apple") || brand.contains("arm") {
        "arm".into()
    } else {
        "unknown".into()
    }
}

fn detect_simd() -> Vec<String> {
    let mut simd = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            simd.push("avx512".into());
        }
        if is_x86_feature_detected!("avx2") {
            simd.push("avx2".into());
        }
        if is_x86_feature_detected!("sse4.1") {
            simd.push("sse4_1".into());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        simd.push("neon".into());
    }
    simd
}

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    if let Some(nvidia_gpus) = detect_nvidia() {
        gpus.extend(nvidia_gpus);
    }

    if let Some(apple_gpu) = detect_apple_gpu() {
        gpus.push(apple_gpu);
    }

    if let Some(vulkan_gpus) = detect_vulkan() {
        gpus.extend(vulkan_gpus);
    }

    gpus
}

fn detect_nvidia() -> Option<Vec<GpuInfo>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim().is_empty() {
        return None;
    }

    let mut gpus = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let index: u32 = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_string();
        let vram_mb: u64 = parts[2].parse().unwrap_or(0);
        let vram_total = vram_mb * 1024 * 1024;

        let compute_capability = get_nvidia_compute_capability(index);

        gpus.push(GpuInfo {
            index,
            vendor: "nvidia".into(),
            name,
            vram_total,
            fp16: compute_capability
                .as_ref()
                .map(|cc| cc.parse::<f32>().unwrap_or(0.0) >= 5.3)
                .unwrap_or(false),
            int8: compute_capability
                .as_ref()
                .map(|cc| cc.parse::<f32>().unwrap_or(0.0) >= 6.1)
                .unwrap_or(false),
            compute_capability,
        });
    }
    if gpus.is_empty() { None } else { Some(gpus) }
}

fn get_nvidia_compute_capability(gpu_index: u32) -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
            "--id",
            &gpu_index.to_string(),
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn detect_apple_gpu() -> Option<GpuInfo> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .ok()?;
        let v: Value = serde_json::from_slice(&output.stdout).ok()?;
        let displays = v.get("SPDisplaysDataType")?.as_array()?;
        for (i, disp) in displays.iter().enumerate() {
            let name = disp.get("chipset_model")?.as_str()?.to_string();
            let vram_str = disp.get("vram")?.as_str()?;
            let vram_total = parse_apple_vram(vram_str);
            if vram_total == 0 {
                continue;
            }
            return Some(GpuInfo {
                index: i as u32,
                vendor: "apple".into(),
                name,
                vram_total,
                fp16: true,
                int8: false,
                compute_capability: None,
            });
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn parse_apple_vram(s: &str) -> u64 {
    let s = s.trim();
    if let Some(n) = s.strip_suffix(" GB") {
        n.trim().parse::<u64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else if let Some(n) = s.strip_suffix(" MB") {
        n.trim().parse::<u64>().unwrap_or(0) * 1024 * 1024
    } else if let Some(n) = s.strip_suffix(" TB") {
        n.trim().parse::<u64>().unwrap_or(0) * 1024 * 1024 * 1024 * 1024
    } else {
        0
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn parse_apple_vram(_s: &str) -> u64 {
    0
}

fn detect_vulkan() -> Option<Vec<GpuInfo>> {
    None
}

fn select_ep(gpus: &[GpuInfo], os: &str) -> String {
    if !gpus.is_empty() {
        let gpu = &gpus[0];
        match gpu.vendor.as_str() {
            "nvidia" => return "cuda".into(),
            "apple" if os == "macos" => return "coreml".into(),
            _ => {}
        }
    }
    if cfg!(target_os = "windows") {
        return "directml".into();
    }
    "cpu".into()
}

fn disk_cache_free() -> u64 {
    use std::path::Path;
    let cache_str = dirs_cache_dir();
    let cache_dir = Path::new(&cache_str);
    let _ = std::fs::create_dir_all(cache_dir);
    fs2_available_space(cache_dir).unwrap_or(0)
}

fn fs2_available_space(path: &std::path::Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        let c_path = CString::new(path.to_str()?).ok()?;
        let mut stat: std::mem::MaybeUninit<libc::statvfs> = std::mem::MaybeUninit::zeroed();
        if unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
            return None;
        }
        let stat = unsafe { stat.assume_init() };
        Some((stat.f_bavail as u64) * (stat.f_bsize as u64))
    }
    #[cfg(windows)]
    {
        None
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

fn dirs_cache_dir() -> String {
    if let Ok(home) = std::env::var("HOME") {
        return format!("{home}/.cache/modelcaller");
    }
    ".cache/modelcaller".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_valid_structure() {
        let caps = probe_capabilities();
        assert!(!caps.os.is_empty());
        assert!(!caps.arch.is_empty());
        assert!(caps.cpu.logical_cores >= 1);
        assert!(caps.ram_total > 0);
        assert!(caps.ram_free > 0);
        assert!(caps.ram_free <= caps.ram_total);
        assert!(!caps.selected_ep.is_empty());
    }

    #[test]
    fn probe_serializes_to_json() {
        let caps = probe_capabilities();
        let json = serde_json::to_value(&caps).unwrap();
        assert!(json.get("os").is_some());
        assert!(json.get("arch").is_some());
        assert!(json.get("cpu").is_some());
        assert!(json.get("ram_total").is_some());
        assert!(json.get("selected_ep").is_some());
    }

    #[test]
    fn cpu_vendor_detected() {
        let caps = probe_capabilities();
        assert!(matches!(
            caps.cpu.vendor.as_str(),
            "intel" | "amd" | "arm" | "unknown"
        ));
    }

    #[test]
    fn simd_detection_no_panic() {
        let caps = probe_capabilities();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert!(
                caps.cpu.simd.contains(&"avx2".into()) || !caps.cpu.simd.contains(&"avx512".into())
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(caps.cpu.simd.contains(&"neon".into()));
        }
    }
}
