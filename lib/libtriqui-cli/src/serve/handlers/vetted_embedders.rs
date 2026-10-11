//! Manually vetted embedder cards for the onboarding recommendation.
//! Reads `ikapp.yaml` via `libtriqui_core::app_config` (repo-root discovery).

use std::collections::HashMap;

use libtriqui_client::protocol::ProtocolError;
use libtriqui_client::types::{VettedEmbedder, VettedEmbeddersResponse};

use crate::serve::handlers::{ok_json, string_err};
use crate::serve::ServeState;

fn system_ram_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
            for line in info.lines() {
                if let Some(kb) = line.strip_prefix("MemTotal:") {
                    let num: u64 = kb.trim().trim_end_matches(" kB").parse().unwrap_or(0);
                    return num / 1024;
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Tail expression (no `;`): the block's value is the return value —
        // a `;` here made this a discarded statement and always returned 0.
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|bytes| bytes / (1024 * 1024))
            .unwrap_or(0)
    }
}

fn pick_recommended(embedders: &[VettedEmbedder], system_ram_mb: u64) -> String {
    let available = system_ram_mb as f64 * 0.5;
    let mut best: Option<&VettedEmbedder> = None;
    let mut best_score: f64 = -1.0;

    for e in embedders {
        if e.memory_mb as f64 > available {
            continue;
        }
        let score = 1.0 / (e.memory_mb as f64).max(1.0);
        if score > best_score {
            best_score = score;
            best = Some(e);
        }
    }

    match best {
        Some(e) => e.slug.clone(),
        None => embedders
            .iter()
            .min_by_key(|e| e.memory_mb)
            .map(|e| e.slug.clone())
            .unwrap_or_default(),
    }
}

fn is_downloaded(model_dir: &std::path::Path, slug: &str) -> bool {
    let model_path = model_dir.join(slug);
    model_path.exists()
}

pub fn list_vetted_embedders(
    _state: &std::sync::Arc<ServeState>,
) -> Result<serde_json::Value, ProtocolError> {
    let cfg = libtriqui_core::app_config::load_ikapp()
        .ok_or_else(|| string_err("config: triq.yaml not found"))?;

    let ram_mb = system_ram_mb();

    let model_map: HashMap<_, _> = cfg.models.iter().map(|m| (m.slug.as_str(), m)).collect();

    let mut embedders: Vec<VettedEmbedder> = Vec::new();
    for ge in &cfg.gui_embedders {
        let model_def = model_map
            .get(ge.slug.as_str())
            .ok_or_else(|| string_err(format!("gui_embedders slug '{}' not in models", ge.slug)))?;
        embedders.push(VettedEmbedder {
            slug: ge.slug.clone(),
            name: ge.name.clone(),
            memory_mb: ge.memory_mb,
            dim: model_def.dim,
            max_context_window: model_def.max_context_window,
            downloaded: is_downloaded(&cfg.model_dir, &ge.slug),
        });
    }

    let recommended = pick_recommended(&embedders, ram_mb);

    ok_json(VettedEmbeddersResponse {
        embedders,
        system_ram_mb: ram_mb,
        recommended_slug: recommended,
    })
}
