//! Candidate detail + browser queries (no Tauri). All functions take a
//! `&Connection` so callers decide the locking/ownership model (the GUI locks
//! its managed `DbConnection`; the CLI opens its own).

use std::path::Path;

use rusqlite::Connection;

use crate::db::db_err;
use crate::types::{
    AdSource, CandidateDetail, CandidateNeighbors, CandidatePage, CandidateSummary,
    ChunkExplanation, DomainError, ExtractionStatus, FileType, GradeSpace, QuestionKind, RubricScore,
};

/// Column list for the browser + triqui rows (positional read by
/// `candidate_summary_from_row` / `triqui_from_row`-style mappers).
const SUMMARY_COLS: &str =
    "id, file_name, file_type, extraction_status, rank, points, points_max, liked, viewed_at";

fn candidate_summary_from_row(row: &rusqlite::Row) -> rusqlite::Result<CandidateSummary> {
    Ok(CandidateSummary {
        id: row.get(0)?,
        file_name: row.get(1)?,
        file_type: FileType::from_str(row.get::<_, String>(2)?.as_str()),
        extraction_status: ExtractionStatus::from_str(row.get::<_, String>(3)?.as_str()),
        rank: row.get(4)?,
        points: row.get(5)?,
        points_max: row.get(6)?,
        scores: vec![],
        grade_space: None,
        ad_source: None,
        liked: row.get::<_, i64>(7)? != 0,
        viewed_at: row.get(8)?,
    })
}

/// Build a case-insensitive `LIKE` pattern from a raw search string, escaping the
/// SQL wildcard/escape characters (`\`, `%`, `_`) so user input matches literally.
/// Leads/trails are left for a "contains" match.
fn like_pattern(q: &str) -> String {
    let escaped = q
        .to_lowercase()
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    format!("%{escaped}%")
}

/// Which column the browser is sorted by.
enum SortCol {
    /// The grade space, then the total within it. **The default**, and the honest
    /// order: a scout-space total and a heavy-space total are both complete and
    /// comparable, but they are not equally *good*, so the better space leads and
    /// the total breaks the tie inside it.
    Space,
    Name,
    Rank,
    /// The ad source, then the total: judge-backed rows lead, then the heavy
    /// embedder's cosine, then the scout's.
    AdSourceT,
    Question(String),
}

/// Resolve `sort_by` into a sort target. `None`, empty, or `"__space"` → grade
/// space then total (the default order); `"__total"` → total points alone;
/// `"__name"` → file name; `"__rank"` → stored rank; `"__source"` → ad-source
/// tier; any other value is treated as a rubric-question id.
fn resolve_sort(sort_by: &Option<String>) -> SortCol {
    match sort_by.as_deref() {
        None | Some("") | Some("__space") | Some("__basis") => SortCol::Space,
        Some("__total") => SortCol::Space,
        Some("__name") => SortCol::Name,
        Some("__rank") => SortCol::Rank,
        Some("__source") => SortCol::AdSourceT,
        Some(q) => SortCol::Question(q.to_string()),
    }
}

/// Build the shared `WHERE … ORDER BY …` SQL for the browser, plus flags telling the
/// caller which optional params it must bind — in this order: `job_id`, then `like`
/// (if `has_like`), then the sort-by question id (if `has_question`). Extracted so the
/// list + neighbors sites stay in lockstep. `asc` flips the direction.
fn browser_clause(like: &Option<String>, sort: &SortCol, asc: bool) -> (String, bool, bool) {
    let has_like = like.is_some();
    let dir = if asc { "ASC" } else { "DESC" };
    let mut where_sql = String::from("WHERE job_id = ?");
    if has_like {
        where_sql.push_str(" AND lower(extracted_text) LIKE ? ESCAPE '\\'");
    }
    // One EXISTS probe per quality axis. The heavy embedder's ad row exists
    // exactly when the candidate has been embedded, and the judge row exactly when
    // it has been compared, so those two probes *are* the axes — no separate
    // bookkeeping column to fall out of sync with the scores.
    let in_heavy = "(EXISTS (SELECT 1 FROM candidate_scores cs WHERE cs.candidate_id = candidates.id \
                      AND cs.kind = 'Ad'))";
    let judged = "(EXISTS (SELECT 1 FROM candidate_scores cs WHERE cs.candidate_id = candidates.id \
                   AND cs.kind = 'AdJudge'))";
    let order = match sort {
        // Space tier first, then total. A better space always leads, because its
        // total is the better estimate — even when it is numerically lower.
        SortCol::Space => format!(
            "{in_heavy} {dir}, points {dir} NULLS LAST, file_name"
        ),
        SortCol::Name => format!("file_name {dir}"),
        SortCol::Rank => format!("rank {dir} NULLS LAST, points DESC, file_name"),
        SortCol::AdSourceT => format!(
            "{judged} {dir}, {in_heavy} {dir}, \
             (EXISTS (SELECT 1 FROM candidate_scores cs WHERE cs.candidate_id = candidates.id \
              AND cs.kind = 'Scout')) {dir}, \
             points DESC NULLS LAST, file_name"
        ),
        SortCol::Question(_) => format!(
            "(SELECT cs.points FROM candidate_scores cs WHERE cs.candidate_id = candidates.id AND cs.question_id = ?) {dir} NULLS LAST, file_name"
        ),
    };
    let has_question = matches!(sort, SortCol::Question(_));
    (
        format!("{where_sql} ORDER BY {order}"),
        has_like,
        has_question,
    )
}

/// Full candidate detail (extracted text + rubric breakdown) for the overlay.
pub fn get_candidate(
    conn: &Connection,
    candidate_id: &str,
) -> Result<CandidateDetail, DomainError> {
    let mut detail = match conn.query_row(
        "SELECT id, job_id, file_name, file_type, rank, extraction_status, extracted_text,
                source_file_path, points, points_max, liked, viewed_at
         FROM candidates WHERE id = ?1",
        rusqlite::params![&candidate_id],
        |row| {
            Ok(CandidateDetail {
                id: row.get(0)?,
                job_id: row.get(1)?,
                file_name: row.get(2)?,
                file_type: FileType::from_str(row.get::<_, String>(3)?.as_str()),
                rank: row.get(4)?,
                extraction_status: ExtractionStatus::from_str(row.get::<_, String>(5)?.as_str()),
                extracted_text: row.get(6)?,
                source_file_path: row.get::<_, String>(7)?,
                original_available: false,
                points: row.get(8)?,
                points_max: row.get(9)?,
                rubric_scores: vec![],
                liked: row.get::<_, i64>(10)? != 0,
                viewed_at: row.get::<_, Option<String>>(11)?,
            })
        },
    ) {
        Ok(d) => d,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(DomainError::NotFound {
                entity: "Candidate".into(),
                id: candidate_id.to_string(),
            });
        }
        Err(e) => return Err(db_err(e)),
    };
    // The original may have moved/been deleted since import.
    detail.original_available = Path::new(&detail.source_file_path).exists();

    // Per-question rubric breakdown (ad → criteria → feedback → the judge
    // stand-in for the ad → the cascade's stage grades, matching the results
    // table's column order).
    let mut sstmt = conn
        .prepare(
            "SELECT question_id, label, kind, points, max_points, chunk_texts
             FROM candidate_scores WHERE candidate_id = ?1
             ORDER BY CASE kind WHEN 'Ad' THEN 0 WHEN 'Criterion' THEN 1
                                 WHEN 'FeedbackJob' THEN 2 WHEN 'FeedbackProfile' THEN 3
                                 WHEN 'AdJudge' THEN 4 WHEN 'Lexical' THEN 5 ELSE 6 END,
                      question_id",
        )
        .map_err(db_err)?;
    let scores: Vec<RubricScore> = sstmt
        .query_map(rusqlite::params![&candidate_id], |row| {
            let chunk_json: Option<String> = row.get(5)?;
            let chunk_explanations: Vec<ChunkExplanation> = chunk_json
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default();
            Ok(RubricScore {
                question_id: row.get(0)?,
                label: row.get(1)?,
                kind: QuestionKind::from_str(row.get::<_, String>(2)?.as_str()),
                points: row.get(3)?,
                max_points: row.get(4)?,
                chunk_explanations,
            })
        })
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;
    detail.rubric_scores = scores;

    Ok(detail)
}

/// List all of a job's candidates as a page, sorted by total rubric points (default)
/// or by a single column (`sort_by`: `None`/`"__total"` = total points, `"__name"` =
/// file name, `"__rank"` = stored rank, else a rubric-question id). `sort_dir` = `true`
/// sorts ascending; `false`/None sorts descending. `query` (when set) filters by the
/// imported CV **content** (substring, ASCII case-insensitive — SQLite `lower()` is
/// ASCII-only). Each row carries its per-question `scores` for the spreadsheet columns.
pub fn list_candidates(
    conn: &Connection,
    job_id: &str,
    offset: i32,
    limit: i32,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<CandidatePage, DomainError> {
    let offset = offset.max(0) as i64;
    let limit = limit.clamp(1, 5000) as i64;

    let like = query
        .as_ref()
        .filter(|q| !q.trim().is_empty())
        .map(|q| like_pattern(q));
    let sort = resolve_sort(&sort_by);
    let asc = sort_dir.unwrap_or(false);

    let total: i64 = match &like {
        Some(l) => conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1 AND lower(extracted_text) LIKE ?2 ESCAPE '\\'",
                rusqlite::params![&job_id, l],
                |r| r.get(0),
            )
            .map_err(db_err)?,
        None => conn
            .query_row(
                "SELECT COUNT(*) FROM candidates WHERE job_id = ?1",
                rusqlite::params![&job_id],
                |r| r.get(0),
            )
            .map_err(db_err)?,
    };
    let total = total.clamp(0, i32::MAX as i64) as i32;

    let (clause, has_like, has_q) = browser_clause(&like, &sort, asc);
    let sql = format!("SELECT {SUMMARY_COLS} FROM candidates {clause} LIMIT ? OFFSET ?");
    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&job_id];
    if has_like {
        params.push(like.as_ref().unwrap());
    }
    if has_q {
        if let SortCol::Question(qid) = &sort {
            params.push(qid);
        }
    }
    params.push(&limit);
    params.push(&offset);

    let mut stmt = conn.prepare(&sql).map_err(db_err)?;
    let mut items: Vec<CandidateSummary> = stmt
        .query_map(params.as_slice(), candidate_summary_from_row)
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;

    // Attach per-question scores (one RubricScore per rubric question) in a single
    // batch query so each row can render its own column in the spreadsheet.
    // Ordered like the results table (ad → criteria → feedback → judge → the
    // cascade's stage grades) so a page's rows are deterministic.
    if !items.is_empty() {
        let ids: Vec<String> = items.iter().map(|i| i.id.clone()).collect();
        let placeholders = vec!["?"; ids.len()].join(", ");
        let sql = format!(
            "SELECT candidate_id, question_id, label, kind, points, max_points
             FROM candidate_scores WHERE candidate_id IN ({placeholders})
             ORDER BY candidate_id,
                       CASE kind WHEN 'Ad' THEN 0 WHEN 'Criterion' THEN 1
                                 WHEN 'FeedbackJob' THEN 2 WHEN 'FeedbackProfile' THEN 3
                                 WHEN 'AdJudge' THEN 4 ELSE 5 END,
                       question_id"
        );
        let mut score_stmt = conn.prepare(&sql).map_err(db_err)?;
        let score_params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
        let rows = score_stmt
            .query_map(score_params.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    RubricScore {
                        question_id: row.get(1)?,
                        label: row.get(2)?,
                        kind: QuestionKind::from_str(row.get::<_, String>(3)?.as_str()),
                        points: row.get(4)?,
                        max_points: row.get(5)?,
                        chunk_explanations: vec![],
                    },
                ))
            })
            .map_err(db_err)?
            .collect::<Result<Vec<(String, RubricScore)>, _>>()
            .map_err(db_err)?;
        let mut by_cand: std::collections::HashMap<String, Vec<RubricScore>> =
            std::collections::HashMap::new();
        for (cid, sc) in rows {
            by_cand.entry(cid).or_default().push(sc);
        }
        for item in items.iter_mut() {
            if let Some(scores) = by_cand.get(&item.id) {
                item.scores = scores.clone();
            }
            item.grade_space = Some(
                GradeSpace::from_scores(&item.scores).unwrap_or(GradeSpace::Scout),
            );
            item.ad_source = AdSource::from_scores(&item.scores);
        }
    }

    Ok(CandidatePage { items, total })
}

/// Prev/next candidate ids in the job's browser ordering (same sort as `list_candidates`,
/// defaulting to total points) so the overlay's `<-`/`->` buttons walk every CV in the
/// current sort. `query` (when set) restricts the walk to the filtered set.
pub fn candidate_neighbors(
    conn: &Connection,
    job_id: &str,
    candidate_id: &str,
    query: Option<String>,
    sort_by: Option<String>,
    sort_dir: Option<bool>,
) -> Result<CandidateNeighbors, DomainError> {
    let like = query
        .as_ref()
        .filter(|q| !q.trim().is_empty())
        .map(|q| like_pattern(q));
    let sort = resolve_sort(&sort_by);
    let asc = sort_dir.unwrap_or(false);

    let (clause, has_like, has_q) = browser_clause(&like, &sort, asc);
    let sql = format!("SELECT id FROM candidates {clause}");
    let mut params: Vec<&dyn rusqlite::ToSql> = vec![&job_id];
    if has_like {
        params.push(like.as_ref().unwrap());
    }
    if has_q {
        if let SortCol::Question(qid) = &sort {
            params.push(qid);
        }
    }
    let mut stmt = conn.prepare(&sql).map_err(db_err)?;
    let ordered: Vec<String> = stmt
        .query_map(params.as_slice(), |r| r.get::<_, String>(0))
        .map_err(db_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;

    let idx = ordered.iter().position(|id| id == candidate_id);
    let (prev_id, next_id) = match idx {
        Some(i) => (
            if i > 0 {
                ordered.get(i - 1).cloned()
            } else {
                None
            },
            if i + 1 < ordered.len() {
                ordered.get(i + 1).cloned()
            } else {
                None
            },
        ),
        None => (None, None),
    };

    Ok(CandidateNeighbors { prev_id, next_id })
}

#[cfg(test)]
mod tests {
    use super::{like_pattern, resolve_sort, SortCol};

    #[test]
    fn like_pattern_lowercases_and_wraps() {
        assert_eq!(like_pattern("Rust"), "%rust%");
    }

    #[test]
    fn like_pattern_escapes_wildcards() {
        // `%`, `_` and `\` must be escaped so they match literally, not as SQL wildcards.
        assert_eq!(like_pattern("C++%_x"), "%c++\\%\\_x%");
        assert_eq!(like_pattern("a\\b"), "%a\\\\b%");
    }

    #[test]
    fn like_pattern_leaves_substring_anchoring() {
        assert_eq!(like_pattern("foo bar"), "%foo bar%");
    }

    #[test]
    fn resolve_sort_defaults_to_the_grade_space_then_total() {
        // The default order is the honest one: a better space always leads, and the
        // total breaks the tie inside it. An empty sort and the legacy `__total`
        // both land there, so an old gui keeps working and gets the better order.
        assert!(matches!(resolve_sort(&None), SortCol::Space));
        assert!(matches!(resolve_sort(&Some(String::new())), SortCol::Space));
        for key in ["__space", "__total", "__basis"] {
            assert!(
                matches!(resolve_sort(&Some(key.into())), SortCol::Space),
                "{key} must resolve to the space-first order"
            );
        }
    }

    #[test]
    fn resolve_sort_reads_the_ad_source_axis() {
        assert!(matches!(
            resolve_sort(&Some("__source".into())),
            SortCol::AdSourceT
        ));
    }

    #[test]
    fn resolve_sort_special_columns() {
        assert!(matches!(
            resolve_sort(&Some("__name".into())),
            SortCol::Name
        ));
        assert!(matches!(
            resolve_sort(&Some("__rank".into())),
            SortCol::Rank
        ));
    }

    #[test]
    fn resolve_sort_passes_through_question_id() {
        assert!(matches!(
            resolve_sort(&Some("ad".into())),
            SortCol::Question(_)
        ));
        assert!(matches!(
            resolve_sort(&Some("crit_123".into())),
            SortCol::Question(_)
        ));
    }
}
