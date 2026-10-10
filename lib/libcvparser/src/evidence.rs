//! stage-a evidence extraction (bench/bench1/design/18-structured-evidence.md §18.3):
//! deterministic text → [`CvEvidence`], no model. the cv corpus is mostly
//! glued single-blob text (headings fused into content like
//! `ExperienceAcme-Sr. Engineer…`, date ranges like `04/2018toCurrent`), so
//! all detection works on character offsets, not on lines.

use libcvevidence::{months_between, Coverage, CvEvidence, Education, Project, Role};
use regex::Regex;
use std::sync::OnceLock;

/// one detected heading and the span of its content. offsets are byte
/// offsets into the source text; `start_char` points just after the heading
/// phrase (content span), `end_char` at the next heading (or end of text).
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub heading: String,
    pub kind: SectionKind,
    pub start_line: usize,
    pub end_line: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    Header,
    Experience,
    Projects,
    Education,
    Skills,
    Certifications,
    Other,
}

/// a cv split into its layout map (§18.3). `text` is the full cv text;
/// `sections` lists every heading found (the text before the first heading
/// is the implicit `Header` section).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCV {
    pub text: String,
    pub sections: Vec<Section>,
}

impl ParsedCV {
    pub fn sections_of(&self, kind: SectionKind) -> Vec<&Section> {
        self.sections.iter().filter(|s| s.kind == kind).collect()
    }

    /// concatenated content of all sections of one kind
    pub fn content_of(&self, kind: SectionKind) -> String {
        let mut out = String::new();
        for s in self.sections_of(kind) {
            out.push_str(&self.text[s.start_char..s.end_char]);
            out.push('\n');
        }
        out
    }

    /// the implicit header (text before the first heading), if non-empty
    pub fn header_text(&self) -> Option<&str> {
        self.sections
            .first()
            .filter(|s| s.kind == SectionKind::Header)
            .map(|s| &self.text[s.start_char..s.end_char])
    }
}

// ---------------------------------------------------------------------------
// heading detection
// ---------------------------------------------------------------------------

/// (kind, phrases) — phrases are matched as accent-stripped lowercase
/// prefixes of a line (cvs glue headings into the following content:
/// `ExperienceHoneywell-…`). longest match wins.
const HEADINGS: &[(SectionKind, &[&str])] = &[
    (
        SectionKind::Experience,
        &[
            "professional experience",
            "work experience",
            "employment history",
            "work history",
            "career history",
            "experience professionnelle",
            "parcours professionnel",
            "parcours",
            "experience",
            "employement",
            "employment",
        ],
    ),
    (
        SectionKind::Projects,
        &[
            "selected projects",
            "personal projects",
            "key projects",
            "projects",
            "projets",
        ],
    ),
    (
        SectionKind::Education,
        &[
            "academic background",
            "academic history",
            "education & training",
            "education",
            "educations",
            "formation",
        ],
    ),
    (
        SectionKind::Skills,
        &[
            "technical skills",
            "skill highlights",
            "skills & expertise",
            "core skills",
            "key skills",
            "technical expertise",
            "competences",
            "skills",
        ],
    ),
    (
        SectionKind::Certifications,
        &["certifications", "certificates", "certification"],
    ),
    (
        SectionKind::Other,
        &[
            "professional summary",
            "executive profile",
            "professional background",
            "additional information",
            "summary",
            "profile",
            "objective",
            "about",
            "languages",
            "langues",
            "centres d'interet",
            "interests",
            "hobbies",
            "loisirs",
            "references",
            "awards",
            "publications",
            "volunteer",
        ],
    ),
];

fn strip_accents(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'à' | 'â' | 'ä' | 'á' | 'ã' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'î' | 'ï' | 'í' => 'i',
            'ô' | 'ö' | 'ó' | 'õ' => 'o',
            'ù' | 'û' | 'ü' | 'ú' => 'u',
            'ç' => 'c',
            'œ' => 'o',
            _ => c,
        })
        .collect()
}

fn normalize(s: &str) -> String {
    strip_accents(&s.to_lowercase())
}

/// find the heading phrase a line starts with (longest match wins)
fn match_heading(normalized_line: &str) -> Option<(&'static str, SectionKind)> {
    let mut best: Option<(&'static str, SectionKind)> = None;
    for (kind, phrases) in HEADINGS {
        for phrase in *phrases {
            if normalized_line.starts_with(phrase)
                && best.is_none_or(|(b, _)| phrase.len() > b.len())
            {
                best = Some((phrase, *kind));
            }
        }
    }
    best
}

/// split a cv into sections (heading dictionary, glue-tolerant)
pub fn parse_layout(text: &str) -> ParsedCV {
    // (line index, byte offset, content)
    let lines: Vec<(usize, usize, &str)> = {
        let mut out = Vec::new();
        let mut off = 0usize;
        for (idx, line) in text.split('\n').enumerate() {
            out.push((idx, off, line));
            off += line.len() + 1; // +1 for the removed '\n'
        }
        out
    };

    // heading lines in order
    let mut found: Vec<Section> = Vec::new();
    for &(idx, off, line) in &lines {
        let t = line.trim_start();
        if t.is_empty() {
            continue;
        }
        if let Some((phrase, kind)) = match_heading(&normalize(t)) {
            found.push(Section {
                heading: phrase.to_string(),
                kind,
                start_line: idx,
                end_line: idx,
                start_char: (off + (line.len() - t.len()) + phrase.len()).min(text.len()),
                end_char: off + line.len(),
            });
        }
    }

    // each section's content extends to the end of the line before the next
    // heading (or end of text)
    for i in 0..found.len() {
        let end_line = found
            .get(i + 1)
            .map(|next| next.start_line.saturating_sub(1))
            .unwrap_or(lines.len().saturating_sub(1));
        let end_char = lines
            .get(end_line)
            .map(|&(_, off, line)| off + line.len())
            .unwrap_or(text.len());
        found[i].end_line = end_line;
        found[i].end_char = end_char.max(found[i].start_char);
    }

    // implicit header: text before the first heading line, if non-blank
    if let Some(first) = found.first() {
        let header_end = lines
            .get(first.start_line)
            .map(|&(_, off, _)| off)
            .unwrap_or(0);
        if text[..header_end].trim().is_empty() == false {
            let header_lines = lines
                .iter()
                .take_while(|&&(_, off, _)| off < header_end)
                .count();
            found.insert(
                0,
                Section {
                    heading: "(header)".to_string(),
                    kind: SectionKind::Header,
                    start_line: 0,
                    end_line: header_lines.saturating_sub(1),
                    start_char: 0,
                    end_char: header_end,
                },
            );
        }
    }

    ParsedCV {
        text: text.to_string(),
        sections: found,
    }
}

// ---------------------------------------------------------------------------
// date ranges
// ---------------------------------------------------------------------------

const MONTHS_EN_FR: &[(&str, u32)] = &[
    ("january", 1), ("february", 2), ("march", 3), ("april", 4), ("may", 5),
    ("june", 6), ("july", 7), ("august", 8), ("september", 9), ("october", 10),
    ("november", 11), ("december", 12),
    ("janvier", 1), ("fevrier", 2), ("mars", 3), ("avril", 4), ("juin", 6),
    ("juillet", 7), ("aout", 8), ("septembre", 9), ("octobre", 10),
    ("novembre", 11), ("decembre", 12),
    ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4), ("jun", 6), ("jul", 7),
    ("aug", 8), ("sept", 9), ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12),
];

fn month_number(name: &str) -> Option<u32> {
    let n = normalize(name);
    MONTHS_EN_FR
        .iter()
        .filter(|(m, _)| n.starts_with(m))
        .max_by_key(|(m, _)| m.len())
        .map(|(_, num)| *num)
}

/// a concrete parsed date point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatePoint {
    pub year: i32,
    /// 1-12; year-only dates pin to mid-year (6) per design §18.2
    pub month: u32,
}

/// one full date-range match: entry start + byte extent of the match text
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DateRange {
    pub start: DatePoint,
    /// `present` when the range says current/present (growing role)
    pub end: DatePoint,
    pub end_present: bool,
    pub match_start: usize,
    pub match_end: usize,
}

fn date_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // glued-tolerant: `05/2020-Current`, `04/2018to04/2015`, `2005to04/2006`,
    // `mar 2019 - present`, `2019 - 2022`. separators may be glued to the
    // dates (`04/2018toCurrent`).
    RE.get_or_init(|| {
        Regex::new(
            r"(?ix)
            (?:(?P<m1>[a-zà-ÿ]+)\.?\s*)?                # optional month name
            (?:
                (?:(?P<mo1>\d{1,2})[/.](?P<y1>(?:19|20)\d{2}))   # mm/yyyy
                |
                (?P<y2>(?:19|20)\d{2})                           # or bare year
            )
            \s*(?:–|—|-|to|au|à|jusqu'|jusqu'a)\s*      # separator (glued ok)
            (?:
                (?P<present>current|present|now|today|présent|actual|aujourd|en\s*poste|ce\s*jour)
                |
                (?:(?P<m2>[a-zà-ÿ]+)\.?\s*)?
                (?:
                    (?:(?P<mo2>\d{1,2})[/.](?P<y3>(?:19|20)\d{2}))
                    |
                    (?P<y4>(?:19|20)\d{2})
                )
            )",
        )
        .unwrap()
    })
}

fn present_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // prefix match on purpose: cvs glue "Current" into the next word
    // ("04/2018toCurrentSenior Analyst…")
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:current|present|now|today|présent|actual|aujourd|en\s*poste|ce\s*jour)")
            .unwrap()
    })
}

fn resolve_date(year: i32, mm: Option<&str>, month_name: Option<&str>) -> Option<DatePoint> {
    if !(1850..=2200).contains(&year) {
        return None;
    }
    let month = match (mm, month_name) {
        (Some(m), _) => m.parse::<u32>().ok().filter(|m| (1..=12).contains(m)),
        (None, n) => n.and_then(month_number),
    };
    Some(DatePoint {
        year,
        month: month.unwrap_or(6), // year-only → mid-year per design
    })
}

/// all date ranges (role entry starts) in `text`, ordered by position.
/// overlapping/nested matches are skipped (regex `find_iter` is
/// non-overlapping already).
pub fn find_date_ranges(text: &str) -> Vec<DateRange> {
    let re = date_regex();
    let mut out = Vec::new();
    for caps in re.captures_iter(text) {
        let whole = caps.get(0).unwrap();
        let start = resolve_date(
            caps.name("y1")
                .or_else(|| caps.name("y2"))
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0),
            caps.name("mo1").map(|m| m.as_str()),
            caps.name("m1").map(|m| m.as_str()),
        );
        let Some(start) = start else { continue };

        // end side: "current/present" (possibly glued to the next word), or
        // an explicit month/year
        let end_year = caps
            .name("y3")
            .or_else(|| caps.name("y4"))
            .and_then(|m| m.as_str().parse().ok());
        let end;
        let end_present;
        match end_year {
            Some(y) => {
                end_present = false;
                match resolve_date(y, caps.name("mo2").map(|m| m.as_str()), caps.name("m2").map(|m| m.as_str())) {
                    Some(d) => end = d,
                    None => continue,
                }
            }
            None => {
                let word = caps.name("present").map(|m| m.as_str()).unwrap_or("");
                if !present_regex().is_match(word) {
                    continue;
                }
                end_present = true;
                end = DatePoint { year: 0, month: 0 };
            }
        }
        if !end_present && (end.year, end.month) < (start.year, start.month) {
            continue;
        }
        out.push(DateRange {
            start,
            end,
            end_present,
            match_start: whole.start(),
            match_end: whole.end(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// role title / org heuristics
// ---------------------------------------------------------------------------

/// split glued text on lower→Upper transitions ("GovernanceMilliman–" →
/// 2 words); plain words with spaces stay one chunk
fn split_camel(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for ch in s.chars() {
        if ch.is_uppercase() && prev_lower && !cur.is_empty() {
            words.push(cur.trim().to_string());
            cur.clear();
        }
        prev_lower = ch.is_lowercase() || ch.is_ascii_digit();
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        words.push(cur.trim().to_string());
    }
    words.into_iter().filter(|w| !w.is_empty()).collect()
}

/// strip a trailing "City, ST," / "City,ST" pattern
fn strip_city_state(s: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // middle is lowercase-only so the match can't swallow preceding
        // title words ("…Support EngineerGladstone,MO," keeps the title)
        Regex::new(r"[A-ZÀ-Ü][a-zà-ÿ.' ]*[a-zà-ÿ.]\s*,\s*[A-ZÀ-Ü]{2}\.?\s*,?\s*$").unwrap()
    });
    re.replace(s, "").to_string()
}

fn clean_title(s: &str) -> String {
    s.trim()
        .trim_matches(|c: char| c.is_whitespace() || ",;:|-–—/".contains(c))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(100)
        .collect::<String>()
        .trim()
        .to_string()
}

fn tail_chars(s: &str, n: usize) -> String {
    let skip = s.char_indices().rev().nth(n.saturating_sub(1)).map(|(i, _)| i).unwrap_or(0);
    s[skip..].to_string()
}

/// first word of a glued org chunk, cut at the city separator
/// ("Honeywell–Memphis,TN,Responsible…" → "Honeywell")
fn clean_org(word: &str) -> Option<String> {
    let head = word.split(['–', '—', '(', ',']).next().unwrap_or(word);
    // the org may be glued to preceding content ("…ExperienceHoneywell") —
    // camel-split keeps the final word
    let last = split_camel(head).pop().unwrap_or_else(|| head.to_string());
    let last = clean_title(&last);
    if last.chars().count() > 1 {
        Some(last)
    } else {
        None
    }
}

/// words that start bullet sentences, not titles — format-B title guard
const BULLET_VERBS: &[&str] = &[
    "built", "created", "working", "worked", "responsible", "prepared",
    "managed", "developed", "developing", "experience", "proficient",
    "delivered", "conducted", "monitor", "leading", "lead", "uses", "using",
    "skills", "ability", "excels",
];

/// extract (title, org) for a role from the text around its date range.
/// format B: date first — `04/2018toCurrentSenior Analyst …GovernanceMilliman–Pasadena,CA,`
/// format A: date last  — `Honeywell-Sr. Data Analyst/Support Engineer…,05/2020-Current`
fn extract_title_org(before: &str, after: &str) -> (String, Option<String>) {
    // --- format B: the text right after the range is the title (+org) ----
    let after_clean = strip_city_state(after);
    let after_trim: String = after_clean
        .trim()
        .trim_matches(|c: char| c.is_whitespace() || ",;:".contains(c))
        .to_string();
    if after_trim.chars().count() >= 3 {
        let words = split_camel(&after_trim);
        let title_candidate = words.first().cloned().unwrap_or_default();
        let first_word = title_candidate
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        let wc = title_candidate.split_whitespace().count();
        let is_titleish = !title_candidate.is_empty()
            && wc <= 6
            && title_candidate.chars().count() <= 80
            && title_candidate.chars().filter(|c| c.is_alphabetic()).count() >= 3
            && !title_candidate.contains(':')
            && !title_candidate.contains('@')
            && !BULLET_VERBS.contains(&first_word.as_str());
        if is_titleish {
            let title = clean_title(&title_candidate);
            let org = words.get(1).and_then(|w| clean_org(w));
            if !title.is_empty() {
                return (title, org);
            }
        }
    }

    // --- format A: `Org-Title` comes from the text before the range ------
    let tail = strip_city_state(&tail_chars(before, 120));
    let tail = tail
        .trim()
        .trim_start_matches(|c: char| c.is_whitespace() || ",;:".contains(c));
    if let Some((org_part, title_part)) = tail
        .split_once('-')
        .or_else(|| tail.split_once('–'))
    {
        let t = clean_title(title_part);
        if t.chars().count() > 2 {
            let o = org_part
                .split_whitespace()
                .last()
                .map(clean_org)
                .unwrap_or(None);
            return (t, o);
        }
    }
    let t = clean_title(tail);
    if t.chars().count() > 2 {
        (t, None)
    } else {
        ("unknown".to_string(), None)
    }
}

// ---------------------------------------------------------------------------
// vocabulary matching
// ---------------------------------------------------------------------------

/// word-boundary-aware substring match on an already-lowercased haystack
pub fn contains_term(hay_lower: &str, term: &str) -> bool {
    let term = term.trim();
    if term.len() < 2 {
        return false;
    }
    let mut start = 0usize;
    while let Some(rel) = hay_lower[start..].find(term) {
        let abs = start + rel;
        let end = abs + term.len();
        let before_ok = hay_lower[..abs]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let after_ok = hay_lower[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        start = end;
    }
    false
}

fn matches_any<'a>(hay_lower: &str, vocabulary: &'a [String]) -> Vec<String> {
    vocabulary
        .iter()
        .filter(|t| contains_term(hay_lower, t))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// birth year (§18.3 step 5 — header only, explicit statements only)
// ---------------------------------------------------------------------------

fn extract_birth_year(header: &str, now: (i32, u32)) -> Option<i32> {
    static BIRTH_DATE: OnceLock<Regex> = OnceLock::new();
    static AGE_EXPLICIT: OnceLock<Regex> = OnceLock::new();
    // née le 12/03/1985, date de naissance : 12.03.1985, born on 3/4/1985
    let birth_date = BIRTH_DATE.get_or_init(|| {
        Regex::new(r"(?ix)
            (?:née?s?\s*(?:le|:)?|date\s*de\s*naissance\s*:?|born(?:\s*on)?\s*:?|geboren(?:\s*am)?\s*:?)
            [^0-9]{0,10}
            \d{1,4}\s*[/.-]\s*\d{1,4}\s*[/.-]\s*((?:19|20)\d{2})")
            .unwrap()
    });
    // only unambiguous forms: "age: 34", "aged 34 years", "34 years old".
    // bare "34 ans" is rejected (matches "25 ans d'expérience")
    let age_explicit = AGE_EXPLICIT.get_or_init(|| {
        Regex::new(r"(?ix)
            (?:
                (?:age|âge)\s*[:\-]?\s*(\d{2})
                |
                (?:aged|âge?e?)\s*(?:de\s+)?(\d{2})\s*(?:years|ans)\b
                |
                (\d{2})\s*years\s*old
            )")
        .unwrap()
    });
    if let Some(caps) = birth_date.captures(header) {
        if let Ok(y) = caps[1].parse::<i32>() {
            if (1900..=now.0).contains(&y) {
                return Some(y);
            }
        }
    }
    if let Some(caps) = age_explicit.captures(header) {
        let a = caps
            .iter()
            .skip(1)
            .flatten()
            .find_map(|m| m.as_str().parse::<i32>().ok())
            .filter(|a| (16..=80).contains(a));
        if let Some(a) = a {
            return Some(now.0 - a);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// the extractor
// ---------------------------------------------------------------------------

/// configuration for [`parse_evidence`]
pub struct EvidenceOptions<'a> {
    /// corpus-derived skill vocabulary (lowercase terms); role/project skill
    /// attribution matches against it
    pub vocabulary: &'a [String],
    /// "present"-pinned extraction date (year, month 1-12)
    pub now: (i32, u32),
}

impl Default for EvidenceOptions<'static> {
    fn default() -> Self {
        Self {
            vocabulary: &[],
            now: (2026, 1),
        }
    }
}

fn iso(year: i32, month: u32) -> String {
    format!("{year}-{month:02}")
}

/// deterministic cv text → [`CvEvidence`] (stage a). never fails: whatever
/// cannot be parsed is reflected in `coverage` instead.
pub fn parse_evidence(text: &str, opts: &EvidenceOptions) -> CvEvidence {
    let parsed = parse_layout(text);
    let lower = |s: &str| strip_accents(&s.to_lowercase());

    // --- roles ------------------------------------------------------------
    let mut roles: Vec<Role> = Vec::new();
    let exp_text = parsed.content_of(SectionKind::Experience);
    if !exp_text.trim().is_empty() {
        let ranges = find_date_ranges(&exp_text);
        for (i, r) in ranges.iter().enumerate() {
            let span_start = r.match_end;
            let span_end = ranges.get(i + 1).map_or(exp_text.len(), |n| n.match_start);
            // before = text leading up to the date match (format A), limited
            // to the tail so previous roles' bullets don't leak in
            let before = tail_chars(&exp_text[..r.match_start], 120);
            let after = &exp_text[span_start..span_end];

            let (title, org) = extract_title_org(&before, after);
            let span_lower = lower(&exp_text[r.match_start..span_end]);
            let skills = matches_any(&span_lower, &opts.vocabulary);

            let start_fmt = iso(start_year(&r), r.start.month);
            let end_opt = if r.end_present {
                iso(opts.now.0, opts.now.1)
            } else {
                iso(r.end.year, r.end.month)
            };
            let months = months_between(&start_fmt, &end_opt);
            roles.push(Role {
                title,
                org,
                start: Some(start_fmt),
                end: Some(end_opt),
                months,
                skills,
            });
        }
    }

    // --- projects ---------------------------------------------------------
    let proj_text = parsed.content_of(SectionKind::Projects);
    let mut projects: Vec<Project> = Vec::new();
    if !proj_text.trim().is_empty() {
        for chunk in split_chunks(&proj_text).into_iter().take(20) {
            let chunk_lower = lower(&chunk);
            let tech = matches_any(&chunk_lower, &opts.vocabulary);
            let name: String = {
                let mut w = chunk
                    .split_whitespace()
                    .take(6)
                    .collect::<Vec<_>>()
                    .join(" ");
                if w.chars().count() > 60 {
                    w = w.chars().take(60).collect();
                }
                if w.is_empty() {
                    "project".to_string()
                } else {
                    w
                }
            };
            projects.push(Project {
                name,
                tech,
                months: None,
                text: chunk.chars().take(400).collect(),
            });
        }
    }

    // --- skills_flat --------------------------------------------------------
    let skills_text = parsed.content_of(SectionKind::Skills);
    let mut skills_flat: Vec<String> = Vec::new();
    for part in skills_text.split(|c: char| c == ',' || c == ';' || c == '\n' || c == '|') {
        let p = part
            .trim()
            .trim_matches(|c: char| c == ':' || c == '.' || c == '*')
            .trim();
        let key = normalize(p);
        if (2..=60).contains(&p.len())
            && !key.is_empty()
            && skills_flat.iter().all(|s| normalize(s) != key)
        {
            skills_flat.push(p.to_string());
        }
        if skills_flat.len() >= 80 {
            break;
        }
    }

    // --- education ----------------------------------------------------------
    static DEGREE: OnceLock<Regex> = OnceLock::new();
    let degree_re = DEGREE.get_or_init(|| {
        Regex::new(r"(?ix)\b(
            b\.?\s*c\.?\s*s|b\.?\s*sc|bachelor|licence|license|m\.?\s*c\.?\s*s|m\.?\s*sc|
            master|mba|ph\.?\s*d|doctorate|doctorat|ingénieur|ingenieur|engineer|diploma|
            diplôme|diplome|associate|high\s*school|bac\b
        )")
        .unwrap()
    });
    static YEAR: OnceLock<Regex> = OnceLock::new();
    let year_re = YEAR.get_or_init(|| Regex::new(r"\b((?:19|20)\d{2})\b").unwrap());
    let edu_text = parsed.content_of(SectionKind::Education);
    let mut education: Vec<Education> = Vec::new();
    for line in edu_text.split('\n') {
        if let Some(m) = degree_re.find(line) {
            // the year may sit before the degree phrase ("Expected in 05/2013
            // — Bachelor of Science…") or after it ("Master, 2015")
            let year = year_re.find(line).map(|y| y.as_str().to_string());
            let end = (m.end() + 60).min(line.len());
            education.push(Education {
                degree: clean_title(&line[m.start()..end]),
                year,
            });
        }
        if education.len() >= 10 {
            break;
        }
    }

    // --- birth year (header only, explicit statements only) -----------------
    let birth_year = parsed
        .header_text()
        .and_then(|h| extract_birth_year(h, opts.now));

    let total_months: u32 = roles.iter().filter_map(|r| r.months).sum();

    let ev = CvEvidence {
        schema: libcvevidence::SCHEMA_VERSION,
        coverage: Coverage {
            roles: !roles.is_empty(),
            projects: !projects.is_empty(),
            skills: !skills_flat.is_empty(),
        },
        roles,
        projects,
        skills_flat,
        education,
        birth_year,
        total_months,
        extensions: None,
        extensions_schema: None,
    };
    debug_assert!(ev.validate().is_ok(), "stage-a evidence must validate");
    ev
}

fn start_year(r: &DateRange) -> i32 {
    r.start.year
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn split_chunks(s: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in s.split('\n') {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let is_bullet = t.starts_with(['-', '•', '*', '·', '▪']);
        if (is_bullet || t.chars().count() > 200) && !cur.trim().is_empty() {
            chunks.push(cur.clone());
            cur.clear();
        }
        cur.push_str(t.trim_start_matches(['-', '•', '*', '·', '▪']));
        cur.push(' ');
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks.into_iter().filter(|c| c.trim().chars().count() > 10).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts<'a>(vocab: &[&'a str]) -> EvidenceOptions<'a> {
        let owned: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        EvidenceOptions {
            vocabulary: Box::leak(owned.into_boxed_slice()),
            now: (2026, 9),
        }
    }

    const FORMAT_B: &str = "Professional BackgroundAnalyst versed in data analysis and reporting.  Excels at cultivating relationships\n\
        Skill HighlightsMicrosoft Access, Microsoft Excel, SAS, SQL Server Manager, Cognos, Crystal ReportsBusiness Objects, SQL, Tableau Server and Desktop, Project Management, Data Analysis\n\
        Professional Experience04/2018toCurrentSenior Analyst Data Quality & GovernanceMilliman–Pasadena,CA,Create policies and procedures for actuarial and analytics groupMonitor adherence to policies with scheduled reviewsConduct training on quality and governance policiesUses SAS and SQL daily04/2012toCurrentHealthcare Data AnalystGeneral Motors–Detroit,MI,Experience with process analysis and other analytic functionsPrepares monthly, quarterly, and annual reports of established metrics02/2008to04/2012Business AnalystHoneywell–Memphis,TN,Responsible for identifying processes to streamline";

    #[test]
    fn layout_detects_glued_headings() {
        let parsed = parse_layout(FORMAT_B);
        let kinds: Vec<SectionKind> = parsed.sections.iter().map(|s| s.kind).collect();
        assert!(kinds.contains(&SectionKind::Skills), "{kinds:?}");
        assert!(kinds.contains(&SectionKind::Experience), "{kinds:?}");
        let exp = parsed.content_of(SectionKind::Experience);
        assert!(exp.starts_with("04/2018toCurrent"), "{exp}");
        assert!(exp.contains("Business Analyst"), "{exp}");
    }

    #[test]
    fn date_ranges_glued_and_present() {
        let ranges = find_date_ranges("04/2018toCurrentSenior Analyst04/2012to04/2015X02/2008to04/2012Y");
        assert_eq!(ranges.len(), 3, "{ranges:?}");
        assert!(ranges[0].end_present);
        assert_eq!((ranges[0].start.year, ranges[0].start.month), (2018, 4));
        assert_eq!((ranges[1].end.year, ranges[1].end.month), (2015, 4));
        assert_eq!((ranges[2].start.year, ranges[2].start.month), (2008, 2));
    }

    #[test]
    fn date_ranges_year_only_and_names() {
        let ranges = find_date_ranges("2019 - 2022 Engineermar 2019 — Present Stuff2005to04/2006 Z");
        assert_eq!(ranges.len(), 3, "{ranges:?}");
        assert_eq!(ranges[0].start.month, 6); // year-only → mid-year
        assert!(ranges[1].end_present);
        assert_eq!(ranges[2].start.year, 2005);
    }

    #[test]
    fn format_b_roles() {
        let ev = parse_evidence(FORMAT_B, &opts(&["sas", "sql", "process analysis"]));
        assert!(ev.validate().is_ok(), "{ev:?}");
        assert!(ev.coverage.roles);
        assert_eq!(ev.roles.len(), 3, "{:?}", ev.roles);
        let first = &ev.roles[0];
        assert_eq!(first.title, "Senior Analyst Data Quality & Governance");
        assert_eq!(first.org.as_deref(), Some("Milliman"));
        assert_eq!(first.start.as_deref(), Some("2018-04"));
        assert_eq!(first.end.as_deref(), Some("2026-09"));
        assert!(first.skills.contains(&"sql".to_string()), "{:?}", first.skills);
        let last = &ev.roles[2];
        assert_eq!(last.title, "Business Analyst");
        assert_eq!(last.org.as_deref(), Some("Honeywell"));
        assert_eq!(last.months, Some(51)); // 2008-02..2012-04 inclusive
        assert!(ev.total_months > 100);
        assert!(ev.coverage.skills);
        assert!(ev.skills_flat.iter().any(|s| s.contains("SAS")));
    }

    const FORMAT_A: &str = "Summary7+ years of experience as a BI developer.\n\
        EducationGMR Institute of TechnologyRajam, India,Expected in05/2013––Bachelor of Science:Information Technology-GPA:\n\
        ExperienceHoneywell-Sr. Data Analyst/Support EngineerGladstone,MO,05/2020-CurrentClient:Citi Bank, Irving, TxWorking as Level2 Support Engineer for Data Pipelines\n\
        Cognizant-BI DeveloperDallas,TX,03/2017-04/2020Built ETL packages in SSIS for the health care domain";

    #[test]
    fn format_a_roles_and_education() {
        let ev = parse_evidence(FORMAT_A, &opts(&["ssis", "etl"]));
        assert!(ev.validate().is_ok(), "{ev:?}");
        assert_eq!(ev.roles.len(), 2, "{:?}", ev.roles);
        let first = &ev.roles[0];
        assert_eq!(first.title, "Sr. Data Analyst/Support Engineer");
        assert_eq!(first.org.as_deref(), Some("Honeywell"));
        assert_eq!(first.start.as_deref(), Some("2020-05"));
        assert_eq!(first.end.as_deref(), Some("2026-09"));
        let second = &ev.roles[1];
        assert_eq!(second.title, "BI Developer");
        assert_eq!(second.org.as_deref(), Some("Cognizant"));
        assert_eq!(second.months, Some(38)); // 2017-03..2020-04 inclusive
        assert_eq!(ev.education.len(), 1, "{:?}", ev.education);
        assert!(ev.education[0].degree.contains("Bachelor of Science"));
        assert_eq!(ev.education[0].year.as_deref(), Some("2013"));
    }

    #[test]
    fn no_dates_cv_has_empty_roles_and_coverage_false() {
        let ev = parse_evidence(
            "Skillsjava, python, docker\nA career spent building java services without dates",
            &opts(&["java"]),
        );
        assert!(ev.validate().is_ok());
        assert!(!ev.coverage.roles);
        assert!(ev.roles.is_empty());
        assert!(ev.coverage.skills);
    }

    #[test]
    fn birth_year_french_and_english() {
        let ev = parse_evidence(
            "Jean Dupont, né le 12/03/1985, Paris\nExperience04/2018toCurrentDev at Acme",
            &opts(&[]),
        );
        assert_eq!(ev.birth_year, Some(1985));
        let ev2 = parse_evidence(
            "Jane Roe, age: 34\nExperience04/2018toCurrentDev at Acme",
            &opts(&[]),
        );
        assert_eq!(ev2.birth_year, Some(2026 - 34));
        let ev3 = parse_evidence(
            "7 ans d'expérience en java\nExperience04/2018toCurrentDev at Acme",
            &opts(&[]),
        );
        assert_eq!(ev3.birth_year, None, "bare 'N ans' must not become an age");
    }

    #[test]
    fn projects_section_parsed() {
        let cv = "Projects\n- Billing migration: java, kafka, kubernetes\n- Data platform on python and airflow";
        let ev = parse_evidence(cv, &opts(&["java", "kafka", "python"]));
        assert!(ev.coverage.projects, "{ev:?}");
        assert!(ev.projects.iter().any(|p| p.tech.contains(&"kafka".to_string())));
    }

    #[test]
    fn contains_term_boundaries() {
        assert!(contains_term("senior java developer", "java"));
        assert!(!contains_term("javascript developer", "java"));
        assert!(contains_term("migrated to microservices", "microservices"));
        assert!(!contains_term("java", "javascript"));
    }
}
