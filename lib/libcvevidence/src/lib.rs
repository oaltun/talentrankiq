//! CvEvidence schema contract (bench/bench1/design/18-structured-evidence.md §18.2).
//!
//! One cv → one `CvEvidence` json: a strict standard base (v1) plus an
//! optional job-ad-derived `extensions` block validated against an
//! `ExtractionSchema` (§18.4). The base is produced deterministically by
//! `libcvparser` (stage A) and can be filled/upgraded by the llm extractor
//! (stage B) later without downstream changes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// current base schema version (`schema` field)
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum EvidenceError {
    #[error("invalid evidence: {0}")]
    Invalid(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// one employment entry. dates are ISO-ish "YYYY-MM"; `months` is derived
/// by the parser from start/end ("present" pinned at extraction time).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Role {
    pub title: String,
    #[serde(default)]
    pub org: Option<String>,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub months: Option<u32>,
    /// skills evidenced *inside this role's text span* (the depth signal:
    /// java under a role ≠ java in a 30-item skill list)
    #[serde(default)]
    pub skills: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Project {
    pub name: String,
    #[serde(default)]
    pub tech: Vec<String>,
    #[serde(default)]
    pub months: Option<u32>,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Education {
    pub degree: String,
    #[serde(default)]
    pub year: Option<String>,
}

/// which sections were found — a feature in itself (a cv without
/// parseable history is a weaker signal, not a parse error)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct Coverage {
    #[serde(default)]
    pub roles: bool,
    #[serde(default)]
    pub projects: bool,
    #[serde(default)]
    pub skills: bool,
}

/// the evidence contract. base fields are strict (unknown fields rejected);
/// `extensions` is schema-bound (see [`ExtractionSchema`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CvEvidence {
    #[serde(default = "default_schema")]
    pub schema: u32,
    #[serde(default)]
    pub roles: Vec<Role>,
    #[serde(default)]
    pub projects: Vec<Project>,
    #[serde(default)]
    pub skills_flat: Vec<String>,
    #[serde(default)]
    pub education: Vec<Education>,
    /// only from an explicit birth date / stated age; never inferred
    #[serde(default)]
    pub birth_year: Option<i32>,
    #[serde(default)]
    pub total_months: u32,
    #[serde(default)]
    pub coverage: Coverage,
    /// job-ad-specific values (null when no discovered schema is in play)
    #[serde(default)]
    pub extensions: Option<serde_json::Map<String, serde_json::Value>>,
    /// hash of the ExtractionSchema the extensions were validated against
    #[serde(default)]
    pub extensions_schema: Option<String>,
}

fn default_schema() -> u32 {
    SCHEMA_VERSION
}

impl CvEvidence {
    pub fn new() -> Self {
        serde_json::from_str("{}").expect("empty evidence is valid")
    }

    /// structural validation (§18.2 rules): schema version, date formats,
    /// months consistency with start/end.
    pub fn validate(&self) -> Result<(), EvidenceError> {
        let mut issues: Vec<String> = Vec::new();
        if self.schema != SCHEMA_VERSION {
            issues.push(format!(
                "unsupported schema version {} (expected {SCHEMA_VERSION})",
                self.schema
            ));
        }
        for (i, role) in self.roles.iter().enumerate() {
            if role.title.trim().is_empty() {
                issues.push(format!("roles[{i}].title is empty"));
            }
            for (which, date) in [("start", &role.start), ("end", &role.end)] {
                if let Some(d) = date {
                    if parse_year_month(d).is_none() {
                        issues.push(format!("roles[{i}].{which} '{d}' is not YYYY-MM"));
                    }
                }
            }
            if let (Some(s), Some(e)) = (&role.start, &role.end) {
                if let (Some((sy, sm)), Some((ey, em))) = (parse_year_month(s), parse_year_month(e)) {
                    if (ey, em) < (sy, sm) {
                        issues.push(format!("roles[{i}] end {e} before start {s}"));
                    }
                    let derived = month_diff(sy, sm, ey, em);
                    if let Some(m) = role.months {
                        if m as i64 != derived as i64 {
                            issues.push(format!(
                                "roles[{i}].months {m} != derived {derived} from {s}..{e}"
                            ));
                        }
                    }
                }
            }
        }
        if self.birth_year.is_some_and(|y| !(1900..=2100).contains(&y)) {
            issues.push(format!("birth_year {:?} out of range", self.birth_year));
        }
        if self.extensions.is_some() != self.extensions_schema.is_some() {
            issues.push(
                "extensions and extensions_schema must be set together (or both null)"
                    .to_string(),
            );
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(EvidenceError::Invalid(issues.join("; ")))
        }
    }

    /// json round-trip through the strict contract (rejects unknown fields)
    pub fn to_json(&self) -> Result<String, EvidenceError> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self, EvidenceError> {
        let ev: CvEvidence = serde_json::from_str(json)?;
        ev.validate()?;
        Ok(ev)
    }
}

impl Default for CvEvidence {
    fn default() -> Self {
        Self::new()
    }
}

/// parse "YYYY-MM" (also accepts "YYYY-MM-DD", truncated to the month)
pub fn parse_year_month(s: &str) -> Option<(i32, u32)> {
    let s = s.trim();
    let mut parts = s.split('-');
    let year: i32 = parts.next()?.trim().parse().ok()?;
    let month: u32 = match parts.next() {
        Some(m) => m.trim().parse().ok()?,
        None => return None,
    };
    if !(1..=12).contains(&month) || !(1850..=2200).contains(&year) {
        return None;
    }
    Some((year, month))
}

/// inclusive month count between two year/month pairs (2019-01..2019-12 → 12)
fn month_diff(sy: i32, sm: u32, ey: i32, em: u32) -> u32 {
    ((ey as i64 - sy as i64) * 12 + em as i64 - sm as i64 + 1) as u32
}

/// inclusive month count between two "YYYY-MM" dates (e.g. 2019-01..2019-12 → 12)
pub fn months_between(start: &str, end: &str) -> Option<u32> {
    let (sy, sm) = parse_year_month(start)?;
    let (ey, em) = parse_year_month(end)?;
    if (ey, em) < (sy, sm) {
        return None;
    }
    Some(month_diff(sy, sm, ey, em))
}

// ---------------------------------------------------------------------------
// ExtractionSchema (§18.4 step 1): job ad → typed field list, discovered once
// per ad, constraining extraction for every cv of that job
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    Numeric,
    String,
    Bool,
    ListString,
}

impl FieldType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FieldType::Numeric => "numeric",
            FieldType::String => "string",
            FieldType::Bool => "bool",
            FieldType::ListString => "list_string",
        }
    }

    pub fn check(&self, value: &serde_json::Value) -> bool {
        match self {
            FieldType::Numeric => value.is_number(),
            FieldType::String => value.is_string(),
            FieldType::Bool => value.is_boolean(),
            FieldType::ListString => value
                .as_array()
                .is_some_and(|a| a.iter().all(|v| v.is_string())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchemaField {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: FieldType,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
    #[serde(default, rename = "enum")]
    pub enum_values: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractionSchema {
    /// base schema this extends (always "v1" today)
    pub base: String,
    pub fields: Vec<SchemaField>,
}

impl ExtractionSchema {
    pub fn new(fields: Vec<SchemaField>) -> Self {
        Self { base: "v1".to_string(), fields }
    }

    /// stable content hash (first 6 hex of sha256 of the canonical json),
    /// stored as `extensions_schema` on every evidence extended by it
    pub fn hash(&self) -> String {
        let canonical = serde_json::to_string(self).unwrap_or_default();
        let digest = Sha256::digest(canonical.as_bytes());
        digest.iter().take(3).map(|b| format!("{b:02x}")).collect()
    }

    /// validate an extensions object against this schema (§18.2 rules):
    /// invalid values DROP the field (never the whole evidence); unknown
    /// keys are dropped; missing required fields are reported as issues.
    /// returns the cleaned map, the dropped keys, and the issues.
    pub fn validate_extensions(
        &self,
        ext: &serde_json::Map<String, serde_json::Value>,
    ) -> (serde_json::Map<String, serde_json::Value>, Vec<String>, Vec<String>) {
        let mut cleaned = serde_json::Map::new();
        let mut dropped = Vec::new();
        let mut issues = Vec::new();
        let known: Vec<&str> = self.fields.iter().map(|f| f.name.as_str()).collect();

        for field in &self.fields {
            match ext.get(&field.name) {
                None => {
                    if field.required {
                        issues.push(format!("missing required field '{}'", field.name));
                    }
                }
                Some(value) => {
                    let mut ok = field.kind.check(value);
                    if ok {
                        if let (Some(allowed), Some(s)) =
                            (&field.enum_values, value.as_str())
                        {
                            ok = allowed.iter().any(|a| a == s);
                            if !ok {
                                issues.push(format!(
                                    "field '{}' value '{s}' not in enum",
                                    field.name
                                ));
                            }
                        }
                    } else {
                        issues.push(format!(
                            "field '{}' expected {}",
                            field.name,
                            field.kind.as_str()
                        ));
                    }
                    if ok {
                        cleaned.insert(field.name.clone(), value.clone());
                    } else {
                        dropped.push(field.name.clone());
                    }
                }
            }
        }
        for key in ext.keys() {
            if !known.contains(&key.as_str()) {
                dropped.push(key.clone());
            }
        }
        (cleaned, dropped, issues)
    }

    pub fn from_json(json: &str) -> Result<Self, EvidenceError> {
        let schema: ExtractionSchema = serde_json::from_str(json)?;
        for field in &schema.fields {
            if field.name.trim().is_empty() {
                return Err(EvidenceError::Invalid("empty field name".to_string()));
            }
        }
        Ok(schema)
    }

    pub fn to_json(&self) -> Result<String, EvidenceError> {
        Ok(serde_json::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> CvEvidence {
        CvEvidence {
            schema: SCHEMA_VERSION,
            roles: vec![Role {
                title: "senior java developer".to_string(),
                org: Some("acme".to_string()),
                start: Some("2019-03".to_string()),
                end: Some("2022-06".to_string()),
                months: Some(40),
                skills: vec!["java".to_string(), "spring".to_string()],
            }],
            projects: vec![Project {
                name: "billing".to_string(),
                tech: vec!["java".to_string(), "kubernetes".to_string()],
                months: None,
                text: "migrated the billing monolith".to_string(),
            }],
            skills_flat: vec!["java".to_string(), "docker".to_string()],
            education: vec![Education {
                degree: "bsc computer science".to_string(),
                year: Some("2015".to_string()),
            }],
            birth_year: Some(1985),
            total_months: 96,
            coverage: Coverage { roles: true, projects: true, skills: true },
            extensions: None,
            extensions_schema: None,
        }
    }

    #[test]
    fn round_trip() {
        let ev = sample();
        let json = ev.to_json().unwrap();
        let back = CvEvidence::from_json(&json).unwrap();
        assert_eq!(ev, back);
    }

    #[test]
    fn empty_is_valid() {
        let ev = CvEvidence::new();
        assert!(ev.validate().is_ok());
        assert_eq!(ev.schema, SCHEMA_VERSION);
        assert!(ev.extensions.is_none());
    }

    #[test]
    fn unknown_base_field_rejected() {
        let err = serde_json::from_str::<CvEvidence>(r#"{"schema":1,"bogus":1}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn wrong_schema_version_rejected() {
        let mut ev = sample();
        ev.schema = 99;
        assert!(ev.validate().is_err());
    }

    #[test]
    fn months_must_match_dates() {
        let mut ev = sample();
        ev.roles[0].months = Some(12);
        assert!(ev.validate().is_err());
        let mut ev2 = sample();
        ev2.roles[0].end = Some("2019-01".to_string());
        assert!(ev2.validate().is_err());
    }

    #[test]
    fn extensions_require_schema_hash() {
        let mut ev = sample();
        ev.extensions = Some(json!({"a": 1}).as_object().unwrap().clone());
        assert!(ev.validate().is_err());
        ev.extensions_schema = Some("deadbe".to_string());
        assert!(ev.validate().is_ok());
    }

    #[test]
    fn months_between_basics() {
        assert_eq!(months_between("2019-01", "2019-12"), Some(12));
        assert_eq!(months_between("2019-03", "2022-06"), Some(40));
        assert_eq!(months_between("2022-06", "2019-03"), None);
        assert_eq!(months_between("2019", "2022"), None);
    }

    fn sample_schema() -> ExtractionSchema {
        ExtractionSchema::new(vec![
            SchemaField {
                name: "years_professional_java".to_string(),
                kind: FieldType::Numeric,
                description: Some("years building java services".to_string()),
                required: true,
                enum_values: None,
            },
            SchemaField {
                name: "leadership_roles".to_string(),
                kind: FieldType::ListString,
                description: None,
                required: false,
                enum_values: None,
            },
            SchemaField {
                name: "seniority_band".to_string(),
                kind: FieldType::String,
                description: None,
                required: false,
                enum_values: Some(vec!["junior".into(), "senior".into()]),
            },
        ])
    }

    #[test]
    fn extension_validation_drops_invalid_keeps_valid() {
        let schema = sample_schema();
        let ext = json!({
            "years_professional_java": 6,
            "leadership_roles": ["tech lead"],
            "seniority_band": "principal",
            "made_up": true
        })
        .as_object()
        .unwrap()
        .clone();
        let (cleaned, dropped, issues) = schema.validate_extensions(&ext);
        assert_eq!(cleaned.get("years_professional_java"), Some(&json!(6)));
        assert!(cleaned.contains_key("leadership_roles"));
        assert!(!cleaned.contains_key("seniority_band"));
        assert!(!cleaned.contains_key("made_up"));
        assert!(dropped.contains(&"seniority_band".to_string()));
        assert!(dropped.contains(&"made_up".to_string()));
        assert!(issues.iter().any(|i| i.contains("enum")));
    }

    #[test]
    fn extension_missing_required_is_reported() {
        let schema = sample_schema();
        let ext = json!({"leadership_roles": []}).as_object().unwrap().clone();
        let (cleaned, dropped, issues) = schema.validate_extensions(&ext);
        assert!(cleaned.contains_key("leadership_roles"));
        assert!(dropped.is_empty());
        assert!(issues.iter().any(|i| i.contains("required")));

        let (cleaned2, _, issues2) =
            schema.validate_extensions(&serde_json::Map::new());
        assert!(cleaned2.is_empty());
        assert!(issues2.iter().any(|i| i.contains("required")));
    }

    #[test]
    fn schema_hash_is_stable_and_short() {
        let schema = sample_schema();
        let h1 = schema.hash();
        let h2 = ExtractionSchema::from_json(&schema.to_json().unwrap())
            .unwrap()
            .hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 6);
    }

    #[test]
    fn evidence_with_validated_extensions_round_trips() {
        let schema = sample_schema();
        let ext = json!({"years_professional_java": 6, "leadership_roles": ["tech lead"]})
            .as_object()
            .unwrap()
            .clone();
        let (cleaned, dropped, _) = schema.validate_extensions(&ext);
        assert!(dropped.is_empty());
        let mut ev = sample();
        ev.extensions = Some(cleaned);
        ev.extensions_schema = Some(schema.hash());
        let back = CvEvidence::from_json(&ev.to_json().unwrap()).unwrap();
        assert_eq!(ev, back);
    }
}
