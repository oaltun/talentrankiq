use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const METRIC_FIELDS: &[&str] = &[
    "ndcg_at_5", "ndcg_at_10", "ndcg_at_20",
    "recall_at_10", "recall_at_20", "map", "spearman",
];

const METRIC_LABELS: &[(&str, &str)] = &[
    ("ndcg_at_5", "nDCG@5"),
    ("ndcg_at_10", "nDCG@10"),
    ("ndcg_at_20", "nDCG@20"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_20", "Recall@20"),
    ("map", "MAP"),
    ("spearman", "Spearman"),
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricDetail {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub iqr: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CostInfo {
    pub total_wall_time_s: f64,
    pub compute_time_s: Option<f64>,
    pub per_observation_mean_ms: f64,
    pub per_compute_observation_mean_ms: Option<f64>,
    pub compute_observations: Option<usize>,
    pub cached_observations: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LeaderboardEntry {
    pub method: String,
    pub config: HashMap<String, serde_yaml::Value>,
    pub result_file: String,
    /// Name of the judgetrainer run script that produced this result
    /// (bench `--run-script`, e.g. "0995-wd5.nu"); None for results that
    /// pre-date the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_script: Option<String>,
    /// Full git commit hash the run started from
    /// (provenance.git_commit); None for results that pre-date it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    pub metrics: HashMap<String, f64>,
    pub metric_details: HashMap<String, MetricDetail>,
    pub is_baseline: bool,
    pub observations_count: usize,
    pub cost: CostInfo,
    pub display_time_s: f64,
    #[serde(with = "timestamp_serde")]
    pub modified: std::time::SystemTime,
}

mod timestamp_serde {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let dur = t
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        s.serialize_u64(dur)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs: u64 = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

#[derive(Debug, Error)]
pub enum ResultError {
    #[error("failed to read directory '{path}': {source}")]
    ReadDir {
        path: String,
        source: std::io::Error,
    },

    #[error("failed to parse YAML in '{path}': {source}")]
    YamlParse {
        path: String,
        source: serde_yaml::Error,
    },

    #[error("YAML serialization failed: {0}")]
    YamlSerialize(#[from] serde_yaml::Error),

    #[error("failed to write '{path}': {source}")]
    WriteFile {
        path: String,
        source: std::io::Error,
    },
}

fn parse_f64(v: &serde_yaml::Value) -> Option<f64> {
    match v {
        serde_yaml::Value::String(s) => s.parse().ok(),
        serde_yaml::Value::Number(n) => n.as_f64(),
        serde_yaml::Value::Null => None,
        _ => None,
    }
}

fn config_key(config: &HashMap<String, serde_yaml::Value>) -> String {
    if config.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = config
        .iter()
        .map(|(_k, v)| match v {
            serde_yaml::Value::String(s) => s.clone(),
            _ => String::new(),
        })
        .collect();
    parts.sort();
    parts.join("\x00")
}

pub fn load_results(dir: &str) -> Result<Vec<LeaderboardEntry>, ResultError> {
    let mut all_entries: Vec<LeaderboardEntry> = Vec::new();

    let entries_dir =
        fs::read_dir(dir).map_err(|e| ResultError::ReadDir {
            path: dir.to_string(),
            source: e,
        })?;

    for entry in entries_dir {
        let entry = entry.map_err(|e| ResultError::ReadDir {
            path: dir.to_string(),
            source: e,
        })?;
        let path = entry.path();
        if !path
            .extension()
            .is_some_and(|ext| ext == "yaml" || ext == "yml")
        {
            continue;
        }

        let modified = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let doc: serde_yaml::Value = match serde_yaml::from_str(&content) {
            Ok(d) => d,
            Err(e) => {
                log::warn!(
                    "skipping unparseable YAML '{}': {}",
                    path.display(),
                    e
                );
                continue;
            }
        };

        if !doc.get("metrics").is_some() {
            continue;
        }

        let meta = doc.get("meta").unwrap_or(&doc);
        let method = meta
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let config = meta
            .get("config")
            .and_then(|v| v.as_mapping())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.clone())))
                    .collect()
            })
            .unwrap_or_default();

        let metrics_section = doc.get("metrics").unwrap_or(&doc);
        let mut metrics = HashMap::new();
        let mut metric_details = HashMap::new();

        for &field in METRIC_FIELDS {
            if let Some(m) = metrics_section.get(field) {
                let mean = m.get("mean").and_then(parse_f64).unwrap_or(0.0);
                let std = m.get("std").and_then(parse_f64).unwrap_or(0.0);
                let median = m.get("median").and_then(parse_f64).unwrap_or(0.0);
                let iqr = m.get("iqr").and_then(parse_f64).unwrap_or(0.0);
                let ci_lower = m.get("ci_95_lower").and_then(parse_f64).unwrap_or(0.0);
                let ci_upper = m.get("ci_95_upper").and_then(parse_f64).unwrap_or(0.0);
                metrics.insert(field.to_string(), mean);
                metric_details.insert(
                    field.to_string(),
                    MetricDetail {
                        mean,
                        median,
                        std,
                        iqr,
                        ci_lower,
                        ci_upper,
                    },
                );
            }
        }

        let observations_count = meta
            .get("observations_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let run_script = meta
            .get("run_script")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let git_commit = meta
            .get("provenance")
            .and_then(|p| p.get("git_commit"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let cost = meta.get("cost");
        let total_wall_time_s = cost
            .and_then(|c| c.get("total_wall_time_s"))
            .and_then(parse_f64)
            .unwrap_or(0.0);
        let compute_time_s = cost.and_then(|c| c.get("compute_time_s")).and_then(parse_f64);
        let per_observation_mean_ms = cost
            .and_then(|c| c.get("per_observation_mean_ms"))
            .and_then(parse_f64)
            .unwrap_or(0.0);
        let per_compute_observation_mean_ms = cost
            .and_then(|c| c.get("per_compute_observation_mean_ms"))
            .and_then(parse_f64);
        let compute_observations = cost
            .and_then(|c| c.get("compute_observations"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let cached_observations = cost
            .and_then(|c| c.get("cached_observations"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let display_time_s = compute_time_s.unwrap_or(total_wall_time_s);

        all_entries.push(LeaderboardEntry {
            method,
            config,
            result_file: path.to_string_lossy().to_string(),
            run_script,
            git_commit,
            metrics,
            metric_details,
            is_baseline: false,
            observations_count,
            cost: CostInfo {
                total_wall_time_s,
                compute_time_s,
                per_observation_mean_ms,
                per_compute_observation_mean_ms,
                compute_observations,
                cached_observations,
            },
            display_time_s,
            modified,
        });
    }

    dedup_keep_newest(&mut all_entries);

    Ok(all_entries)
}

fn dedup_keep_newest(entries: &mut Vec<LeaderboardEntry>) {
    let mut best: HashMap<String, usize> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        let group_key = format!("{}\x00{}", entry.method, config_key(&entry.config));
        let is_newer = match best.get(&group_key) {
            Some(&prev_i) => entry.modified > entries[prev_i].modified,
            None => true,
        };
        if is_newer {
            best.insert(group_key, i);
        }
    }

    let mut used: Vec<bool> = vec![false; entries.len()];
    for &idx in best.values() {
        used[idx] = true;
    }
    let kept: Vec<LeaderboardEntry> = entries
        .iter()
        .zip(used.iter())
        .filter(|(_, keep)| **keep)
        .map(|(e, _)| e.clone())
        .collect();
    *entries = kept;
}

fn method_label(method: &str, config: &HashMap<String, serde_yaml::Value>) -> String {
    if config.is_empty() {
        return method.to_string();
    }
    let parts: Vec<String> = config
        .iter()
        .map(|(k, v)| {
            let val = match v {
                serde_yaml::Value::String(s) => s.clone(),
                serde_yaml::Value::Number(n) => n.to_string(),
                serde_yaml::Value::Bool(b) => b.to_string(),
                _ => String::new(),
            };
            let val_truncated = if k == "embedder" {
                &val // never truncate the model name, even if long
            } else if val.len() > 25 {
                &val[..25]
            } else {
                &val
            };
            format!("{k}: {val_truncated}")
        })
        .collect();
    format!("{method} ({})", parts.join(", "))
}

fn config_summary(config: &HashMap<String, serde_yaml::Value>) -> String {
    if config.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = config
        .iter()
        .map(|(k, v)| {
            let val = match v {
                serde_yaml::Value::String(s) => s.clone(),
                serde_yaml::Value::Number(n) => n.to_string(),
                serde_yaml::Value::Bool(b) => b.to_string(),
                _ => String::new(),
            };
            let val_truncated = if k == "embedder" {
                &val // never truncate the model name, even if long
            } else if val.len() > 25 {
                &val[..25]
            } else {
                &val
            };
            format!("{k}: {val_truncated}")
        })
        .collect();
    parts.join(", ")
}

fn fmt_cost_count(v: Option<usize>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "-".to_string(),
    }
}

fn fmt_cost_ms(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{:.1}", n),
        None => "-".to_string(),
    }
}

fn fmt_cost_s(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{:.1}", n),
        None => "-".to_string(),
    }
}

/// Human date for the leaderboard "added" column: `YYYY-MM-DD HH:MM` UTC
/// (civil-from-days, Howard Hinnant's algorithm; keeps this crate dep-free).
fn fmt_added(modified: std::time::SystemTime) -> String {
    let secs = modified
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (h, m) = (rem / 3600, (rem % 3600) / 60);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}")
}

pub fn sort_entries(entries: &mut [LeaderboardEntry], sort_by: &str) {
    entries.sort_by(|a, b| {
        let va = b.metrics.get(sort_by).copied().unwrap_or(0.0);
        let vb = a.metrics.get(sort_by).copied().unwrap_or(0.0);
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub fn mark_baselines(entries: &mut [LeaderboardEntry], baseline: &Option<String>) {
    if let Some(bl) = baseline {
        for entry in entries.iter_mut() {
            entry.is_baseline = entry.method == *bl;
        }
    }
}

pub fn generate_html(
    entries: &mut [LeaderboardEntry],
    sort_by: &str,
    baseline: &Option<String>,
) -> String {
    sort_entries(entries, sort_by);
    mark_baselines(entries, baseline);

    let best_per_metric: HashMap<String, f64> = METRIC_FIELDS
        .iter()
        .filter_map(|&f| {
            let best = entries
                .iter()
                .filter_map(|e| e.metrics.get(f).copied())
                .fold(0.0f64, f64::max);
            if best > 0.0 {
                Some((f.to_string(), best))
            } else {
                None
            }
        })
        .collect();

    let sort_label = METRIC_LABELS
        .iter()
        .find(|(k, _)| *k == sort_by)
        .map(|(_, l)| *l)
        .unwrap_or(sort_by);

    let mut rows = String::new();
    let mut configs_section = String::new();
    for entry in entries {
        let row_class = if entry.is_baseline {
            " class=\"baseline\""
        } else {
            ""
        };

        let mut cells = String::new();
        let label = method_label(&entry.method, &entry.config);
        cells.push_str(&format!(
            "      <td class=\"method\" title=\"{}\">{}</td>\n",
            config_summary(&entry.config),
            label
        ));
        // the run script that produced this result (one file = one run in
        // judge/judgetrainer/runs/; the numeric prefix keeps them in run order)
        cells.push_str(&format!(
            "      <td class=\"script\">{}</td>\n",
            entry.run_script.as_deref().unwrap_or("-")
        ));
        // the commit the run started from; short hash displayed, full hash
        // in the hover title (libgitstate clean-tree gate guarantees it
        // names the exact code that produced the result)
        let commit_short = entry
            .git_commit
            .as_deref()
            .map(|c| c[..8.min(c.len())].to_string())
            .unwrap_or_else(|| "-".to_string());
        cells.push_str(&format!(
            "      <td class=\"commit\" title=\"{}\">{}</td>\n",
            entry.git_commit.as_deref().unwrap_or(""),
            commit_short
        ));
        let added_secs = entry
            .modified
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        cells.push_str(&format!(
            "      <td class=\"added\" data-ts=\"{added_secs}\">{}</td>\n",
            fmt_added(entry.modified)
        ));

        for &(field, _label) in METRIC_LABELS {
            let detail = entry.metric_details.get(field);
            let val = detail
                .map(|d| format!("{:.3}", d.mean))
                .unwrap_or_else(|| "-".to_string());
            let best = best_per_metric.get(field).copied().unwrap_or(0.0);
            let is_best =
                detail.is_some_and(|d| (d.mean - best).abs() < 1e-6 && best > 0.0);
            let cell_class = if is_best {
                " class=\"best\""
            } else {
                ""
            };
            cells.push_str(&format!("      <td{cell_class}>{val}</td>\n"));
        }

        let cost = &entry.cost;
        cells.push_str(&format!(
            "      <td>{}</td>\n",
            fmt_cost_count(cost.compute_observations)
        ));
        cells.push_str(&format!(
            "      <td>{}</td>\n",
            fmt_cost_count(cost.cached_observations)
        ));
        cells.push_str(&format!(
            "      <td>{:.1}</td>\n",
            cost.per_observation_mean_ms
        ));
        cells.push_str(&format!(
            "      <td>{}</td>\n",
            fmt_cost_ms(cost.per_compute_observation_mean_ms)
        ));
        cells.push_str(&format!(
            "      <td>{}</td>\n",
            fmt_cost_s(cost.compute_time_s)
        ));
        cells.push_str(&format!(
            "      <td>{:.1}</td>\n",
            cost.total_wall_time_s
        ));
        rows.push_str(&format!(
            "    <tr{row_class}>\n{cells}    </tr>\n\n",
            row_class = row_class,
            cells = cells
        ));

        if !entry.config.is_empty() {
            let config_str = config_summary(&entry.config);
            configs_section.push_str(&format!(
                "  <div class=\"config-entry\"><strong>{}</strong>: {}</div>\n",
                entry.method, config_str
            ));
        }
    }

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>bench1 leaderboard</title>
<style>
  :root {{
    --bg: #1a1a2e;
    --bg2: #16213e;
    --bg3: #1a1a3e;
    --bg4: #162040;
    --text: #e0e0e0;
    --text2: #e0e0ff;
    --text3: #888;
    --border: #333;
    --best: #90ee90;
    --hover: #1a1a4e;
  }}
  [data-theme="light"] {{
    --bg: #ffffff;
    --bg2: #f5f5f5;
    --bg3: #e8e8e8;
    --bg4: #e0e0e0;
    --text: #333333;
    --text2: #000000;
    --text3: #666666;
    --border: #cccccc;
    --best: #006600;
    --hover: #d0d0d0;
  }}
  body {{ font-family: monospace; margin: 2em; background: var(--bg); color: var(--text); }}
  h1 {{ color: var(--text2); font-size: 1.4em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ position: sticky; top: 0; background: var(--bg2); cursor: pointer; user-select: none; }}
  th:hover {{ background: var(--hover); }}
  tr:hover {{ background: var(--bg3); }}
  tr.baseline {{ font-style: italic; background: var(--bg4); }}
  td.best {{ font-weight: bold; color: var(--best); }}
  td.added {{ white-space: nowrap; }}
  td.method {{ white-space: normal; word-break: break-word; max-width: 38em; vertical-align: top; }}
  .configs {{ margin-top: 1em; }}
  .config-entry {{ font-size: 0.85em; color: var(--text3); margin-bottom: 0.3em; }}
  .notes {{ margin-top: 1em; font-size: 0.85em; color: var(--text3); }}
  .theme-toggle {{
    position: fixed; top: 1em; right: 1em; padding: 0.4em 0.8em;
    background: var(--bg2); color: var(--text); border: 1px solid var(--border);
    cursor: pointer; font-family: monospace; font-size: 0.9em;
  }}
  .theme-toggle:hover {{ background: var(--hover); }}
</style>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()">light/dark</button>
<h1>leaderboard (sorted by {sort_label})</h1>
<table>
<thead>
<tr>
  <th>method</th>
  <th>script</th>
  <th>commit</th>
  <th data-type="date">added (UTC)</th>
  <th data-type="num">nDCG@5</th>
  <th data-type="num">nDCG@10</th>
  <th data-type="num">nDCG@20</th>
  <th data-type="num">Recall@10</th>
  <th data-type="num">Recall@20</th>
  <th data-type="num">MAP</th>
  <th data-type="num">Spearman</th>
  <th data-type="num">computed obs</th>
  <th data-type="num">cached obs</th>
  <th data-type="num">ms/obs</th>
  <th data-type="num">ms/computed</th>
  <th data-type="num">compute (s)</th>
  <th data-type="num">wall (s)</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<div class="configs">
<h3>configurations</h3>
{configs_section}
</div>
<div class="notes">
  entries loaded from result YAML files | best per column in <span style="color:var(--best)">green</span> | baseline row in <span style="font-style:italic">italic</span>
  <p><strong>script</strong>: the judgetrainer run script that produced this result (judge/judgetrainer/runs/&lt;script&gt; — one file = one run, committed with the result; "-" for results benched by hand).</p>
  <p><strong>commit</strong>: git commit the run started from (hover for the full hash). runs are refused from a dirty tree, so this is the exact code that produced the result.</p>
  <p><strong>added (UTC)</strong>: when this result was written to the results dir. click the column once for newest-first.</p>
</div>
<h3>metrics explained</h3>
<div class="notes">
  <p><strong>nDCG@k</strong> (Normalized Discounted Cumulative Gain at k): measures ranking quality at the top-k positions. considers graded relevance (0=No Fit, 1=Potential Fit, 3=Good Fit). higher is better. typical range: 0-1.</p>
  <p><strong>Recall@k</strong>: fraction of "Good Fit" resumes found in the top-k. answers "did we miss anyone good?" higher is better. typical range: 0-1.</p>
  <p><strong>MAP</strong> (Mean Average Precision): treats "Good Fit" as relevant (1), everything else as non-relevant (0). ignores Potential Fit. higher is better. typical range: 0-1.</p>
  <p><strong>Spearman</strong> (Spearman's rank correlation): correlation between predicted ranking order and ground truth order. measures overall ordering quality. higher is better. typical range: -1 to 1.</p>
  <p><strong>mean</strong>: arithmetic average across all observations (job postings). sensitive to outliers.</p>
  <p><strong>median</strong> (med): middle value when sorted. robust to outlier observations.</p>
  <p><strong>iqr</strong> (Interquartile Range): difference between 75th and 25th percentile. measures spread of the central 50% of values. robust to outliers. not shown in table (available in YAML output).</p>
  <p><strong>[lo, hi]</strong>: 95% bootstrap confidence interval. if two methods' intervals don't overlap, the difference is likely statistically significant.</p>
  <p><strong>computed obs</strong>: number of observations that required actual computation (not fully cached).</p>
  <p><strong>cached obs</strong>: number of observations where all embeddings were served from cache.</p>
  <p><strong>ms/obs</strong>: mean milliseconds per observation (wall time / total observations).</p>
  <p><strong>ms/computed</strong>: mean milliseconds per computed observation (compute time / computed observations). "-" if none computed.</p>
  <p><strong>compute (s)</strong>: wall-clock seconds spent on computed observations only. "-" if none computed.</p>
  <p><strong>wall (s)</strong>: total wall-clock seconds for the entire run.</p>
</div>
<script>
function toggleTheme() {{
  var body = document.body;
  if (body.getAttribute('data-theme') === 'light') {{
    body.removeAttribute('data-theme');
  }} else {{
    body.setAttribute('data-theme', 'light');
  }}
}}
document.querySelectorAll("th[data-type]").forEach(function(th) {{
  // dates start newest-first on the first click; numbers start ascending
  var asc = th.getAttribute("data-type") === "date" ? false : true;
  th.addEventListener("click", function() {{
    var table = th.closest("table");
    var tbody = table.querySelector("tbody");
    var rows = Array.from(tbody.querySelectorAll("tr"));
    var col = Array.from(th.parentNode.children).indexOf(th);
    var isDate = th.getAttribute("data-type") === "date";
    rows.sort(function(a,b) {{
      var av, bv;
      if (isDate) {{
        av = parseFloat(a.children[col].getAttribute("data-ts")) || 0;
        bv = parseFloat(b.children[col].getAttribute("data-ts")) || 0;
      }} else {{
        av = parseFloat(a.children[col].textContent) || 0;
        bv = parseFloat(b.children[col].textContent) || 0;
      }}
      return asc ? av-bv : bv-av;
    }});
    asc = !asc;
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }});
}});
</script>
</body>
</html>"#,
        sort_label = sort_label,
        rows = rows,
        configs_section = configs_section,
    )
}

pub fn generate_text_table(entries: &mut [LeaderboardEntry], sort_by: &str) -> String {
    sort_entries(entries, sort_by);

    let mut lines = Vec::new();
    lines.push(format!("leaderboard — sorted by {sort_by}"));
    lines.push("-".repeat(90));

    let header = format!(
        "{:<55} {:>8} {:>8} {:>8}",
        "method", "nDCG@10", "Recall@10", "time"
    );
    lines.push(header);

    for entry in entries {
        let ndcg = entry.metrics.get("ndcg_at_10").copied().unwrap_or(0.0);
        let recall = entry
            .metrics
            .get("recall_at_10")
            .copied()
            .unwrap_or(0.0);
        let label = method_label(&entry.method, &entry.config);
        let marker = if entry.is_baseline { " *" } else { "" };
        let computed = entry
            .cost
            .compute_observations
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let cached = entry
            .cost
            .cached_observations
            .map(|n| n.to_string())
            .unwrap_or_else(|| "-".to_string());
        let ms_computed = entry.cost.per_compute_observation_mean_ms
            .map(|n| format!("{:.0}", n))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "{:<55} {:>8.3} {:>8.3} {:>4.1}s  comp:{} cached:{} ms/comp:{}{}",
            label,
            ndcg,
            recall,
            entry.display_time_s,
            computed,
            cached,
            ms_computed,
            marker
        ));
    }

    lines.push("-".repeat(60));
    lines.join("\n")
}

pub fn generate_yaml(
    entries: &mut [LeaderboardEntry],
    sort_by: &str,
    baseline: &Option<String>,
) -> Result<String, ResultError> {
    sort_entries(entries, sort_by);
    mark_baselines(entries, baseline);

    let yaml_entries: Vec<serde_yaml::Value> = entries.iter().map(|e| {
        let mut metrics_map = serde_yaml::Mapping::new();
        for &(field, _) in METRIC_LABELS {
            if let Some(detail) = e.metric_details.get(field) {
                metrics_map.insert(
                    serde_yaml::Value::String(field.to_string()),
                    serde_yaml::Value::Mapping({
                        let mut m = serde_yaml::Mapping::new();
                        m.insert(
                            serde_yaml::Value::String("mean".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.mean)),
                        );
                        m.insert(
                            serde_yaml::Value::String("median".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.median)),
                        );
                        m.insert(
                            serde_yaml::Value::String("std".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.std)),
                        );
                        m.insert(
                            serde_yaml::Value::String("iqr".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.iqr)),
                        );
                        m.insert(
                            serde_yaml::Value::String("ci_95_lower".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.ci_lower)),
                        );
                        m.insert(
                            serde_yaml::Value::String("ci_95_upper".to_string()),
                            serde_yaml::Value::String(format!("{:.4}", detail.ci_upper)),
                        );
                        m
                    }),
                );
            }
        }

        let mut entry_map = serde_yaml::Mapping::new();
        entry_map.insert(
            serde_yaml::Value::String("method".to_string()),
            serde_yaml::Value::String(e.method.clone()),
        );
        if !e.config.is_empty() {
            entry_map.insert(
                serde_yaml::Value::String("config".to_string()),
                serde_yaml::Value::Mapping(
                    e.config
                        .iter()
                        .map(|(k, v)| {
                            (serde_yaml::Value::String(k.clone()), v.clone())
                        })
                        .collect(),
                ),
            );
        }
        entry_map.insert(
            serde_yaml::Value::String("result_file".to_string()),
            serde_yaml::Value::String(e.result_file.clone()),
        );
        entry_map.insert(
            serde_yaml::Value::String("metrics".to_string()),
            serde_yaml::Value::Mapping(metrics_map),
        );
        if e.is_baseline {
            entry_map.insert(
                serde_yaml::Value::String("is_baseline".to_string()),
                serde_yaml::Value::Bool(true),
            );
        }
        entry_map.insert(
            serde_yaml::Value::String("observations".to_string()),
            serde_yaml::Value::Number(serde_yaml::Number::from(e.observations_count)),
        );
        let cost = &e.cost;
        let mut cost_map = serde_yaml::Mapping::new();
        cost_map.insert(
            serde_yaml::Value::String("total_wall_time_s".to_string()),
            serde_yaml::Value::String(format!("{:.1}", cost.total_wall_time_s)),
        );
        if let Some(v) = cost.compute_time_s {
            cost_map.insert(
                serde_yaml::Value::String("compute_time_s".to_string()),
                serde_yaml::Value::String(format!("{:.1}", v)),
            );
        }
        cost_map.insert(
            serde_yaml::Value::String("per_observation_mean_ms".to_string()),
            serde_yaml::Value::String(format!("{:.4}", cost.per_observation_mean_ms)),
        );
        if let Some(v) = cost.per_compute_observation_mean_ms {
            cost_map.insert(
                serde_yaml::Value::String("per_compute_observation_mean_ms".to_string()),
                serde_yaml::Value::String(format!("{:.4}", v)),
            );
        }
        if let Some(v) = cost.compute_observations {
            cost_map.insert(
                serde_yaml::Value::String("compute_observations".to_string()),
                serde_yaml::Value::Number(serde_yaml::Number::from(v)),
            );
        }
        if let Some(v) = cost.cached_observations {
            cost_map.insert(
                serde_yaml::Value::String("cached_observations".to_string()),
                serde_yaml::Value::Number(serde_yaml::Number::from(v)),
            );
        }
        entry_map.insert(
            serde_yaml::Value::String("cost".to_string()),
            serde_yaml::Value::Mapping(cost_map),
        );
        serde_yaml::Value::Mapping(entry_map)
    }).collect();

    let mut doc = serde_yaml::Mapping::new();
    doc.insert(
        serde_yaml::Value::String("sort_by".to_string()),
        serde_yaml::Value::String(sort_by.to_string()),
    );
    doc.insert(
        serde_yaml::Value::String("entries_count".to_string()),
        serde_yaml::Value::Number(serde_yaml::Number::from(entries.len())),
    );
    doc.insert(
        serde_yaml::Value::String("entries".to_string()),
        serde_yaml::Value::Sequence(yaml_entries),
    );

    serde_yaml::to_string(&doc).map_err(ResultError::YamlSerialize)
}

pub fn write_html(
    entries: &mut [LeaderboardEntry],
    sort_by: &str,
    baseline: &Option<String>,
    path: &str,
) -> Result<(), ResultError> {
    let html = generate_html(entries, sort_by, baseline);
    fs::write(path, &html).map_err(|e| ResultError::WriteFile {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}

pub fn write_text(
    entries: &mut [LeaderboardEntry],
    sort_by: &str,
) -> String {
    generate_text_table(entries, sort_by)
}

pub fn write_yaml(
    entries: &mut [LeaderboardEntry],
    sort_by: &str,
    baseline: &Option<String>,
    path: &str,
) -> Result<(), ResultError> {
    let yaml = generate_yaml(entries, sort_by, baseline)?;
    fs::write(path, &yaml).map_err(|e| ResultError::WriteFile {
        path: path.to_string(),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    fn sample_entry(method: &str) -> LeaderboardEntry {
        let mut metrics = HashMap::new();
        metrics.insert("ndcg_at_10".to_string(), 0.721);
        metrics.insert("ndcg_at_5".to_string(), 0.600);
        metrics.insert("recall_at_10".to_string(), 0.85);

        let mut metric_details = HashMap::new();
        metric_details.insert(
            "ndcg_at_10".to_string(),
            MetricDetail {
                mean: 0.721,
                median: 0.710,
                std: 0.05,
                iqr: 0.03,
                ci_lower: 0.68,
                ci_upper: 0.76,
            },
        );

        LeaderboardEntry {
            method: method.to_string(),
            config: HashMap::new(),
            result_file: format!("/tmp/{method}.yaml"),
            run_script: Some("0995-wd5.nu".to_string()),
            git_commit: Some("ae2eef5d7c62f72806aead9bbc30bd13ebb7d409".to_string()),
            metrics,
            metric_details,
            is_baseline: false,
            observations_count: 100,
            cost: CostInfo {
                total_wall_time_s: 1.5,
                compute_time_s: Some(1.0),
                per_observation_mean_ms: 15.0,
                per_compute_observation_mean_ms: Some(10.0),
                compute_observations: Some(100),
                cached_observations: Some(0),
            },
            display_time_s: 1.0,
            modified: UNIX_EPOCH,
        }
    }

    #[test]
    fn test_parse_f64_string() {
        assert_eq!(
            parse_f64(&serde_yaml::Value::String("0.721".to_string())),
            Some(0.721)
        );
    }

    #[test]
    fn test_fmt_added_known_timestamps() {
        use std::time::{Duration, UNIX_EPOCH};
        // 2026-01-01 00:00 UTC
        assert_eq!(
            fmt_added(UNIX_EPOCH + Duration::from_secs(1767225600)),
            "2026-01-01 00:00"
        );
        // epoch itself
        assert_eq!(fmt_added(UNIX_EPOCH), "1970-01-01 00:00");
        // leap year: 2024-02-29 15:05 UTC = 1709219100
        assert_eq!(
            fmt_added(UNIX_EPOCH + Duration::from_secs(1709219100)),
            "2024-02-29 15:05"
        );
    }

    #[test]
    fn test_generate_html_has_added_column() {
        let mut entries = vec![sample_entry("bm25")];
        let html = generate_html(&mut entries, "ndcg_at_10", &None);
        assert!(html.contains(">added (UTC)<"));
        assert!(html.contains("class=\"added\" data-ts=\"0\">1970-01-01 00:00<"));
    }

    #[test]
    fn test_generate_html_has_script_column() {
        // run script flows from bench --run-script through meta.run_script
        // into the leaderboard table; results without one render as "-"
        let mut with = vec![sample_entry("bm25")];
        let html = generate_html(&mut with, "ndcg_at_10", &None);
        assert!(html.contains("<th>script</th>"));
        assert!(html.contains("class=\"script\">0995-wd5.nu<"));

        let mut without = vec![sample_entry("bm25")];
        without[0].run_script = None;
        without[0].git_commit = None;
        let html = generate_html(&mut without, "ndcg_at_10", &None);
        assert!(html.contains("class=\"script\">-<"));
    }

    #[test]
    fn test_generate_html_has_commit_column() {
        // provenance.git_commit renders short in the cell, full in the
        // hover title; results without one render as "-"
        let mut with = vec![sample_entry("bm25")];
        let html = generate_html(&mut with, "ndcg_at_10", &None);
        assert!(html.contains("<th>commit</th>"));
        assert!(html.contains("class=\"commit\" title=\"ae2eef5d7c62f72806aead9bbc30bd13ebb7d409\">ae2eef5d<"));

        let mut without = vec![sample_entry("bm25")];
        without[0].git_commit = None;
        let html = generate_html(&mut without, "ndcg_at_10", &None);
        assert!(html.contains("class=\"commit\" title=\"\">-<"));
    }

    #[test]
    fn test_parse_f64_null() {
        assert_eq!(parse_f64(&serde_yaml::Value::Null), None);
    }

    #[test]
    fn test_sort_entries() {
        let mut entries = vec![sample_entry("bm25"), sample_entry("random")];
        sort_entries(&mut entries, "ndcg_at_10");
        assert_eq!(entries[0].method, "bm25");
        assert_eq!(entries[1].method, "random");
    }

    #[test]
    fn test_mark_baselines() {
        let mut entries = vec![sample_entry("bm25"), sample_entry("random")];
        mark_baselines(&mut entries, &Some("random".to_string()));
        assert!(!entries[0].is_baseline);
        assert!(entries[1].is_baseline);
    }

    #[test]
    fn test_generate_yaml() {
        let mut entries = vec![sample_entry("bm25")];
        mark_baselines(&mut entries, &Some("bm25".to_string()));
        let yaml = generate_yaml(&mut entries, "ndcg_at_10", &Some("bm25".to_string()))
            .expect("yaml generation should succeed");
        assert!(yaml.contains("sort_by: ndcg_at_10"));
        assert!(yaml.contains("method: bm25"));
        assert!(yaml.contains("is_baseline: true"));
    }

    #[test]
    fn test_generate_html() {
        let mut entries = vec![sample_entry("bm25")];
        let html = generate_html(&mut entries, "ndcg_at_10", &None);
        assert!(html.contains("leaderboard"));
        assert!(html.contains("bm25"));
        assert!(html.contains("0.721"));
    }

    #[test]
    fn test_generate_text_table() {
        let mut entries = vec![sample_entry("bm25")];
        let text = generate_text_table(&mut entries, "ndcg_at_10");
        assert!(text.contains("leaderboard"));
        assert!(text.contains("bm25"));
        assert!(text.contains("0.721"));
    }

    #[test]
    fn test_dedup_keeps_newest() {
        let mut entries = vec![
            LeaderboardEntry {
                method: "bm25".to_string(),
                config: HashMap::new(),
                result_file: "/tmp/a.yaml".to_string(),
                run_script: None,
                git_commit: None,
                metrics: HashMap::new(),
                metric_details: HashMap::new(),
                is_baseline: false,
                observations_count: 0,
                cost: CostInfo {
                    total_wall_time_s: 0.0,
                    compute_time_s: None,
                    per_observation_mean_ms: 0.0,
                    per_compute_observation_mean_ms: None,
                    compute_observations: None,
                    cached_observations: None,
                },
                display_time_s: 0.0,
                modified: UNIX_EPOCH,
            },
            LeaderboardEntry {
                method: "bm25".to_string(),
                config: HashMap::new(),
                result_file: "/tmp/b.yaml".to_string(),
                run_script: Some("0996-s44.nu".to_string()),
                git_commit: None,
                metrics: HashMap::new(),
                metric_details: HashMap::new(),
                is_baseline: false,
                observations_count: 0,
                cost: CostInfo {
                    total_wall_time_s: 0.0,
                    compute_time_s: None,
                    per_observation_mean_ms: 0.0,
                    per_compute_observation_mean_ms: None,
                    compute_observations: None,
                    cached_observations: None,
                },
                display_time_s: 0.0,
                modified: UNIX_EPOCH + std::time::Duration::from_secs(1),
            },
        ];
        dedup_keep_newest(&mut entries);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].result_file, "/tmp/b.yaml");
    }

    #[test]
    fn test_config_key_empty() {
        let config: HashMap<String, serde_yaml::Value> = HashMap::new();
        assert_eq!(config_key(&config), "");
    }

    #[test]
    fn test_config_key_non_empty() {
        let mut config: HashMap<String, serde_yaml::Value> = HashMap::new();
        config.insert(
            "embedder".to_string(),
            serde_yaml::Value::String("nomic".to_string()),
        );
        let key = config_key(&config);
        assert!(!key.is_empty());
    }
}
