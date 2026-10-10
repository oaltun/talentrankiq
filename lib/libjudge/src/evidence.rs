//! Evidence `x`-block features (design 18-structured-evidence.md §18.5):
//! pair-level scalars from (job ad, cv evidence), the non-similarity signal
//! fused into the `fused` judge variant. One code path for judgetrainer and
//! bench — the same function feeds the trainer's extraction and the
//! `evidence-rules` stage-1 method (rules only add fixed weights on top).

use libcvevidence::CvEvidence;
use std::collections::HashSet;

/// Width of the fused evidence block appended per cv (`x1`/`x2`).
pub const X_BLOCK_LEN: usize = 12;

/// Feature order inside the x block (design 18 §18.5 table).
pub const X_FEATURES: [&str; X_BLOCK_LEN] = [
    "years_matching_max",
    "years_total",
    "skill_coverage_depth",
    "skill_coverage_mention",
    "project_relevance",
    "seniority",
    "title_sim",
    "age",
    "birth_year_present",
    "coverage_roles",
    "coverage_projects",
    "coverage_skills",
];

/// Stopword list for vocabulary/terms (mirrors bench's tokenizer).
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "i", "you", "your", "we", "our", "us", "my",
    "me", "him", "her", "his", "its", "our", "can", "could", "would", "should", "may", "might",
    "shall", "must", "have", "has", "had", "do", "does", "did", "doing", "done", "been", "being",
    "am", "from", "also", "just", "about", "up", "out", "how", "what", "when", "where", "which",
    "who", "whom", "so", "than", "too", "very", "more", "most", "only", "own", "same", "other",
    "each", "every", "all", "any", "some", "few", "many", "much", "well", "back", "even",
    "still", "way", "take", "come", "get", "make", "know", "like", "long", "look", "most", "new",
    "now", "also", "one", "two", "first", "last", "over", "after", "before",
];

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| !STOPWORDS.contains(s))
        .map(|s| s.to_string())
        .collect()
}

/// Corpus-derived skill vocabulary (§18.3 step 3): top-k discriminative
/// terms of the given texts (df within [2, 70% of docs]), deterministic
/// order (df desc, term asc).
pub fn build_vocabulary(texts: &[String], k: usize) -> Vec<String> {
    let n = texts.len();
    if n == 0 {
        return Vec::new();
    }
    let token_lists: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
    let mut df: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for tokens in &token_lists {
        let seen: HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        for term in seen {
            *df.entry(term.to_string()).or_insert(0) += 1;
        }
    }
    let max_df = ((n as f64) * 0.7).ceil() as usize;
    let mut terms: Vec<(String, usize)> = df
        .into_iter()
        .filter(|(t, d)| {
            *d >= 2 && *d <= max_df && t.len() >= 3 && !t.chars().all(|c| c.is_ascii_digit())
        })
        .collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    terms.into_iter().take(k).map(|(t, _)| t).collect()
}

/// Terms of `vocabulary` present in the ad; if too few (<5), fall back to
/// the tokenized ad's unique terms so short/sparse ads still score. The
/// result is sorted + deduped (required by [`compute_features`]).
pub fn ad_terms(job_text: &str, vocabulary: &[String]) -> Vec<String> {
    let lower = job_text.to_lowercase();
    let mut terms: Vec<String> = vocabulary
        .iter()
        .filter(|t| libcvparser::contains_term(&lower, t))
        .cloned()
        .collect();
    if terms.len() < 5 {
        for t in tokenize(job_text) {
            if !terms.contains(&t) {
                terms.push(t);
            }
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

fn seniority_score(title: &str) -> f64 {
    let t = title.to_lowercase();
    let lead = [
        "lead", "principal", "staff", "head", "director", "manager", "architect", "chef",
        "responsable",
    ];
    let senior = ["senior", "sr.", "sr ", "snr", "expert", "confirmé"];
    let junior = [
        "junior", "jr.", "jr ", "intern", "stage", "stagiaire", "trainee", "apprentice",
        "assistant",
    ];
    if lead.iter().any(|w| t.contains(w)) {
        1.0
    } else if senior.iter().any(|w| t.contains(w)) {
        0.75
    } else if junior.iter().any(|w| t.contains(w)) {
        0.25
    } else {
        0.5
    }
}

fn title_similarity(title: &str, job_lower: &str) -> f64 {
    let tokens = tokenize(title);
    if tokens.is_empty() {
        return 0.0;
    }
    let hits = tokens
        .iter()
        .filter(|t| libcvparser::contains_term(job_lower, t))
        .count();
    hits as f64 / tokens.len() as f64
}

/// One cv's evidence features against one ad's term list (§18.5 `x` block).
/// All values are bounded to [0, 1]; `ad_terms` must be sorted.
pub struct PairFeatures {
    pub years_matching_max: f64,
    pub years_total: f64,
    pub skill_coverage_depth: f64,
    pub skill_coverage_mention: f64,
    pub project_relevance: f64,
    pub seniority: f64,
    pub title_sim: f64,
    pub age: f64,
    pub birth_year_present: f64,
    pub coverage_roles: f64,
    pub coverage_projects: f64,
    pub coverage_skills: f64,
}

pub fn compute_features(job_text: &str, ev: &CvEvidence, ad_terms: &[String]) -> PairFeatures {
    debug_assert!(
        ad_terms.windows(2).all(|w| w[0] <= w[1]),
        "ad_terms must be sorted"
    );
    let job_lower = job_text.to_lowercase();
    let n_terms = ad_terms.len().max(1) as f64;

    // depth sets: skills evidenced inside roles / projects
    let mut role_skills: HashSet<&str> = HashSet::new();
    for r in &ev.roles {
        for s in &r.skills {
            role_skills.insert(s.as_str());
        }
    }
    let mut proj_tech: HashSet<&str> = HashSet::new();
    for p in &ev.projects {
        for t in &p.tech {
            proj_tech.insert(t.as_str());
        }
        // project text evidences terms too (§18.5: fraction of ad skill
        // terms evidenced in projects)
        let p_lower = p.text.to_lowercase();
        for t in ad_terms {
            if libcvparser::contains_term(&p_lower, t) {
                proj_tech.insert(t.as_str());
            }
        }
    }
    let flat: HashSet<&str> = ev.skills_flat.iter().map(|s| s.as_str()).collect();

    let frac = |set: &HashSet<&str>| -> f64 {
        ad_terms.iter().filter(|t| set.contains(t.as_str())).count() as f64 / n_terms
    };
    let skill_coverage_depth = frac(&role_skills);
    let skill_coverage_mention = frac(&flat);
    let project_relevance = frac(&proj_tech);

    // years matching: max months evidenced under any ad term
    let mut years_matching_max = 0.0f64;
    for r in &ev.roles {
        let evidences = r
            .skills
            .iter()
            .any(|s| ad_terms.binary_search(s).is_ok());
        if let Some(m) = r.months {
            if evidences {
                years_matching_max = years_matching_max.max(m as f64 / 12.0);
            }
        }
    }
    years_matching_max = (years_matching_max / 10.0).min(1.0);

    let years_total = (ev.total_months as f64 / 120.0).min(1.0);

    // seniority + title sim of the most recent ad-matching role
    let matching = ev
        .roles
        .iter()
        .filter(|r| r.skills.iter().any(|s| ad_terms.binary_search(s).is_ok()))
        .collect::<Vec<_>>();
    let (seniority, title_sim) = if let Some(mr) = matching
        .iter()
        .max_by_key(|r| r.start.clone().unwrap_or_default())
    {
        (seniority_score(&mr.title), title_similarity(&mr.title, &job_lower))
    } else {
        (0.0, 0.0)
    };

    let (age, birth_year_present) = match ev.birth_year {
        Some(y) => (((today_utc().0 - y) as f64 / 100.0).min(1.0), 1.0),
        None => (0.0, 0.0),
    };

    PairFeatures {
        years_matching_max,
        years_total,
        skill_coverage_depth,
        skill_coverage_mention,
        project_relevance,
        seniority,
        title_sim,
        age,
        birth_year_present,
        coverage_roles: if ev.coverage.roles { 1.0 } else { 0.0 },
        coverage_projects: if ev.coverage.projects { 1.0 } else { 0.0 },
        coverage_skills: if ev.coverage.skills { 1.0 } else { 0.0 },
    }
}

/// The fused x block for one cv, in [`X_FEATURES`] order. Unlike the
/// rules score, no fixed weighting or coverage discount here — the raw
/// features go in and the judge learns the blend (coverage flags are part
/// of the block for exactly that reason).
pub fn x_block(f: &PairFeatures) -> Vec<f32> {
    vec![
        f.years_matching_max as f32,
        f.years_total as f32,
        f.skill_coverage_depth as f32,
        f.skill_coverage_mention as f32,
        f.project_relevance as f32,
        f.seniority as f32,
        f.title_sim as f32,
        f.age as f32,
        f.birth_year_present as f32,
        f.coverage_roles as f32,
        f.coverage_projects as f32,
        f.coverage_skills as f32,
    ]
}

/// (year, month) from the system clock — the "present" pin (§18.2).
pub fn today_utc() -> (i32, u32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = secs.div_euclid(86_400);
    // Howard Hinnant's civil_from_days
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y } as i32, m as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_with_role() -> CvEvidence {
        serde_json::from_str(
            r#"{
                "schema": 1,
                "roles": [
                    {"title": "senior java developer", "org": "acme",
                     "start": "2019-03", "end": "2022-06", "months": 40,
                     "skills": ["java", "spring"]},
                    {"title": "junior developer", "start": "2015-01", "end": "2019-01",
                     "months": 48, "skills": ["python"]}
                ],
                "projects": [{"name": "billing", "tech": ["kafka"], "text": "kafka streaming"}],
                "skills_flat": ["java", "docker"],
                "education": [{"degree": "bsc computer science", "year": "2015"}],
                "birth_year": 1988,
                "total_months": 88,
                "coverage": {"roles": true, "projects": true, "skills": true}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn depth_beats_mention() {
        let terms = vec!["java".to_string(), "spring".to_string()];
        let deep = ev_with_role();
        let mut mention_only = ev_with_role();
        mention_only.roles = Vec::new(); // java only in the flat list
        let f_deep = compute_features("java spring job", &deep, &terms);
        let f_mention = compute_features("java spring job", &mention_only, &terms);
        assert!(f_deep.skill_coverage_depth > f_mention.skill_coverage_depth);
    }

    #[test]
    fn seniority_reads_the_most_recent_matching_role() {
        let terms = vec!["java".to_string()];
        let f = compute_features("java job", &ev_with_role(), &terms);
        assert!((f.seniority - 0.75).abs() < 1e-9, "{}", f.seniority);
    }

    #[test]
    fn recency_years_and_age_are_bounded() {
        let terms = vec!["java".to_string()];
        let f = compute_features("java job", &ev_with_role(), &terms);
        assert!(f.years_matching_max > 0.0 && f.years_matching_max <= 1.0);
        assert!(f.years_total > 0.0 && f.years_total <= 1.0);
        assert!(f.age > 0.3 && f.age < 0.45); // 2026-1988 = 38
        assert_eq!(f.birth_year_present, 1.0);
    }

    #[test]
    fn empty_evidence_gives_all_zero_features() {
        let terms = vec!["java".to_string()];
        let ev = CvEvidence::new();
        let f = compute_features("java job", &ev, &terms);
        let x = x_block(&f);
        assert_eq!(x.len(), X_BLOCK_LEN);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn x_block_matches_feature_order() {
        let terms = vec!["java".to_string()];
        let f = compute_features("java job", &ev_with_role(), &terms);
        let x = x_block(&f);
        assert!((x[0] - f.years_matching_max as f32).abs() < 1e-6);
        assert_eq!(x[8], 1.0); // birth_year_present
        assert_eq!(x[9] + x[10] + x[11], 3.0); // coverage flags
    }

    #[test]
    fn vocabulary_is_deterministic_and_filtered() {
        let texts: Vec<String> = (0..6)
            .map(|i| match i {
                0 | 2 | 4 => "java services with kafka pipelines and spring".to_string(),
                1 | 3 => "python scripts with kafka tools and spring".to_string(),
                _ => "go services with terraform and spring".to_string(),
            })
            .collect();
        let v1 = build_vocabulary(&texts, 10);
        let v2 = build_vocabulary(&texts, 10);
        assert_eq!(v1, v2);
        // "java": df 3 in [2, ceil(4.2)] ✓; "with" is a stopword; "terraform"
        // df 1 < 2 → dropped
        assert!(v1.contains(&"java".to_string()));
        assert!(v1.contains(&"kafka".to_string()));
        assert!(!v1.iter().any(|t| t == "with"));
        assert!(!v1.iter().any(|t| t == "terraform"));
    }

    #[test]
    fn ad_terms_fall_back_on_sparse_ads() {
        let vocab = vec!["java".to_string(), "kafka".to_string()];
        let terms = ad_terms("golang rust developer", &vocab);
        assert!(terms.contains(&"golang".to_string()));
        assert!(terms.contains(&"rust".to_string()));
        assert!(terms.windows(2).all(|w| w[0] <= w[1]), "sorted");
        // 2 vocab hits < 5 → fallback still merges the ad's own tokens
        let hit = ad_terms("senior java kafka", &vocab);
        assert_eq!(hit, vec!["java".to_string(), "kafka".to_string(), "senior".to_string()]);
    }
}
