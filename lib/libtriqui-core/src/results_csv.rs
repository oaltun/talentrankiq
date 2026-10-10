//! Results CSV export: one row per ranked candidate, one column per rubric
//! question (job ad + criteria) — the on-screen results table as data.
//! Feedback questions are personal and never exported.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use rusqlite::Connection;

// The summary DTO is part of the gui<->cli wire contract: `crate::types`
// re-exports it from `libtriqui_client`.
pub use crate::types::ResultsCsvSummary;

/// A rubric question column: the job ad first, then the criteria in creation
/// order. `max` is the per-question maximum from the stored scores (absent
/// when no candidate was scored on it).
type Question = (String, String, Option<f64>);

/// Export every ranked candidate of `job_id` to `out_path` as CSV: rank, file
/// identity, total points, liked flag, and one points column per question.
pub fn export_results_csv(
    conn: &Connection,
    job_id: &str,
    out_path: &Path,
) -> Result<ResultsCsvSummary, String> {
    let mut questions: Vec<Question> = vec![("ad".into(), "Job ad".into(), None)];
    {
        let mut stmt = conn
            .prepare("SELECT id, text FROM criteria WHERE job_id = ?1 ORDER BY created_at, rowid")
            .map_err(|e| e.to_string())?;
        let rows: Vec<(String, String)> = stmt
            .query_map(rusqlite::params![job_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        questions.extend(rows.into_iter().map(|(id, text)| (id, text, None)));
    }
    let column: HashMap<String, usize> = questions
        .iter()
        .enumerate()
        .map(|(i, (id, _, _))| (id.clone(), i))
        .collect();

    // Earned points per candidate and question, plus the canonical max per
    // question (uniform across candidates; first score wins).
    let mut points: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT s.candidate_id, s.question_id, s.points, s.max_points
                 FROM candidate_scores s JOIN candidates c ON c.id = s.candidate_id
                 WHERE c.job_id = ?1 AND s.kind IN ('Ad', 'AdJudge', 'Criterion')",
            )
            .map_err(|e| e.to_string())?;
        let rows: Vec<(String, String, f64, f64)> = stmt
            .query_map(rusqlite::params![job_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        drop(stmt);
        // The judge's win-rate and the embedder's cosine both answer "how well
        // does this CV fit the ad", so they share the Job ad column — and the
        // judge wins when both are present, because that is the grade the
        // ranking used. Whichever row is read first must not decide it, so a
        // written cell records which grade it holds.
        let mut written: HashMap<(String, usize), u8> = HashMap::new();
        for (candidate_id, question_id, pts, max) in rows {
            let is_judge = question_id == crate::sort::JUDGE_QUESTION_ID;
            let Some(&i) = column.get(if is_judge {
                crate::sort::AD_QUESTION_ID
            } else {
                question_id.as_str()
            }) else {
                continue;
            };
            let key = (candidate_id.clone(), i);
            let already = written
                .get(&key)
                .is_some_and(|&rank| rank >= is_judge as u8);
            if already {
                continue;
            }
            let entry = points.entry(candidate_id).or_default();
            if entry.len() <= i {
                entry.resize(i + 1, None);
            }
            entry[i] = Some(pts);
            if questions[i].2.is_none() {
                questions[i].2 = Some(max);
            }
            written.insert(key, is_judge as u8);
        }
    }

    // Ranked candidates, best first.
    let mut stmt = conn
        .prepare(
            "SELECT id, rank, file_name, file_type, extraction_status, points, points_max, liked
             FROM candidates WHERE job_id = ?1 AND rank IS NOT NULL
             ORDER BY rank, file_name",
        )
        .map_err(|e| e.to_string())?;
    let rows: Vec<(
        String,
        i64,
        String,
        String,
        String,
        Option<f64>,
        Option<f64>,
        bool,
    )> = stmt
        .query_map(rusqlite::params![job_id], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
                r.get::<_, i64>(7)? != 0,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    drop(stmt);

    let mut header: Vec<String> = vec![
        "rank".into(),
        "file".into(),
        "type".into(),
        "status".into(),
        "total".into(),
        "max".into(),
        "liked".into(),
    ];
    header.extend(questions.iter().map(|(_, label, max)| match max {
        Some(m) => format!("{label} (max {})", fmt_num(*m)),
        None => label.clone(),
    }));

    let mut csv = header
        .iter()
        .map(|h| csv_cell(h))
        .collect::<Vec<_>>()
        .join(",");
    for (cid, rank, file, file_type, status, total, total_max, liked) in &rows {
        let question_cols = questions
            .iter()
            .enumerate()
            .map(|(i, _)| {
                points
                    .get(cid)
                    .and_then(|p| p.get(i).and_then(|v| *v))
                    .map(fmt_num)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let cells = vec![
            rank.to_string(),
            csv_cell(file),
            csv_cell(file_type),
            csv_cell(status),
            total.map(fmt_num).unwrap_or_default(),
            total_max.map(fmt_num).unwrap_or_default(),
            liked.to_string(),
        ];
        csv.push('\n');
        csv.push_str(
            &cells
                .into_iter()
                .chain(question_cols)
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create output directory: {e}"))?;
    }
    let mut f = std::fs::File::create(out_path)
        .map_err(|e| format!("Cannot create {}: {e}", out_path.display()))?;
    f.write_all(csv.as_bytes())
        .map_err(|e| format!("Cannot write {}: {e}", out_path.display()))?;

    Ok(ResultsCsvSummary {
        rows: rows.len() as i64,
        questions: questions.len(),
    })
}

/// `42.5` → `"42.5"`, `10.0` → `"10"` — compact spreadsheet-friendly numbers.
fn fmt_num(v: f64) -> String {
    let s = format!("{v:.1}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Quote a cell only when needed (comma, quote, newline) — RFC 4180 style.
fn csv_cell(v: &str) -> String {
    if v.contains(',') || v.contains('"') || v.contains('\n') {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_conn() -> Connection {
        // A real (temp) file DB so `open_connection_at` runs the full migration.
        let path = std::env::temp_dir().join(format!("csv-test-db-{}.db", uuid::Uuid::new_v4()));
        crate::db::open_connection_at(&path).expect("test db")
    }

    #[test]
    fn csv_includes_all_ranked_and_question_columns() {
        let conn = test_conn();
        conn.execute(
            "INSERT INTO jobs (id, title, triqui_size) VALUES ('job_1', 'Rust Engineer', 1)",
            [],
        )
        .unwrap();
        // Insertion order defines the column order (created_at, rowid).
        for (id, text) in [
            ("crit_b", "Knows Rust, deeply"),
            ("crit_a", "German fluent"),
        ] {
            conn.execute(
                "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, 'job_1', ?2, 'Positive', 10)",
                rusqlite::params![id, text],
            )
            .unwrap();
        }
        for (id, name, rank, pts, liked) in [
            ("cand_1", "jane.pdf", 1, 42.5, true),
            ("cand_2", "bob, \"the best\".pdf", 2, 10.0, false),
        ] {
            conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, rank, points, points_max, liked)
                 VALUES (?1, 'job_1', ?2, ?3, 'Pdf', 'Extracted', ?4, ?5, 65.0, ?6)",
                rusqlite::params![id, format!("/tmp/{id}"), name, rank, pts, liked],
            )
            .unwrap();
        }
        // Scores: ad + both criteria for the winner; only ad for the runner-up.
        for (cid, qid, kind, pts) in [
            ("cand_1", "ad", "Ad", 30.0),
            ("cand_1", "crit_b", "Criterion", 0.0),
            ("cand_1", "crit_a", "Criterion", 12.5),
            ("cand_2", "ad", "Ad", 10.0),
        ] {
            conn.execute(
                "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
                 VALUES (?1, ?2, ?3, ?4, 'Positive', 'label', ?5, 50.0)",
                rusqlite::params![format!("score_{cid}_{qid}"), cid, qid, kind, pts],
            )
            .unwrap();
        }

        let out = std::env::temp_dir().join(format!("csv-test-{}.csv", uuid::Uuid::new_v4()));
        let summary = export_results_csv(&conn, "job_1", &out).unwrap();
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.questions, 3);

        let csv = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[0],
            "rank,file,type,status,total,max,liked,Job ad (max 50),\"Knows Rust, deeply (max 50)\",German fluent (max 50)"
        );
        assert_eq!(lines[1], "1,jane.pdf,Pdf,Extracted,42.5,65,true,30,0,12.5");
        assert_eq!(
            lines[2],
            "2,\"bob, \"\"the best\"\".pdf\",Pdf,Extracted,10,65,false,10,,"
        );

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn unranked_candidates_are_not_exported() {
        let conn = test_conn();
        conn.execute(
            "INSERT INTO jobs (id, title, triqui_size) VALUES ('job_1', 'Rust Engineer', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES ('cand_1', 'job_1', '/tmp/cand_1', 'jane.pdf', 'Pdf', 'Extracted')",
            [],
        )
        .unwrap();

        let out = std::env::temp_dir().join(format!("csv-test-{}.csv", uuid::Uuid::new_v4()));
        let summary = export_results_csv(&conn, "job_1", &out).unwrap();
        assert_eq!(summary.rows, 0);
        assert_eq!(summary.questions, 1); // just the job ad, without a max

        let csv = std::fs::read_to_string(&out).unwrap();
        assert_eq!(
            csv.lines().next(),
            Some("rank,file,type,status,total,max,liked,Job ad")
        );

        let _ = std::fs::remove_file(&out);
    }

    /// With the judge stage active the job-ad row is stored as `kind =
    /// 'AdJudge'` (win-rate), not `'Ad'` — it must still land in the "Job ad"
    /// column instead of silently leaving it empty.
    #[test]
    fn judge_mode_ad_row_fills_the_job_ad_column() {
        let conn = test_conn();
        conn.execute(
            "INSERT INTO jobs (id, title, triqui_size) VALUES ('job_1', 'Rust Engineer', 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, rank, points, points_max, liked)
             VALUES ('cand_1', 'job_1', '/tmp/cand_1', 'jane.pdf', 'Pdf', 'Extracted', 1, 42.5, 65.0, 0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
             VALUES ('score_1', 'cand_1', 'ad', 'AdJudge', 'Positive', 'Job fit (judge)', 37.5, 50.0)",
            [],
        )
        .unwrap();

        let out = std::env::temp_dir().join(format!("csv-test-{}.csv", uuid::Uuid::new_v4()));
        let summary = export_results_csv(&conn, "job_1", &out).unwrap();
        assert_eq!(summary.rows, 1);
        assert_eq!(summary.questions, 1);

        let csv = std::fs::read_to_string(&out).unwrap();
        assert_eq!(
            csv.lines().nth(1),
            Some("1,jane.pdf,Pdf,Extracted,42.5,65,false,37.5"),
            "the AdJudge win-rate must fill the Job ad column"
        );

        let _ = std::fs::remove_file(&out);
    }

    /// Judge mode stores *two* job-ad grades: the embedder's cosine (`ad`, kept
    /// for the results table's stage column) and the judge's win-rate
    /// (`ad_judge`, the grade the rank actually used). Both answer the same
    /// question, so they share the "Job ad" column — and the judge's must win
    /// whichever order SQLite hands the rows back in.
    #[test]
    fn judge_grade_outranks_the_embed_cosine_in_the_job_ad_column() {
        for judge_first in [true, false] {
            let conn = test_conn();
            conn.execute(
                "INSERT INTO jobs (id, title, triqui_size) VALUES ('job_1', 'Rust Engineer', 1)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status, rank, points, points_max, liked)
                 VALUES ('cand_1', 'job_1', '/tmp/cand_1', 'jane.pdf', 'Pdf', 'Extracted', 1, 40.0, 50.0, 0)",
                [],
            )
            .unwrap();
            let stored = [
                ("ad", "Ad", 21.0, "s1"),
                ("ad_judge", "AdJudge", 40.0, "s2"),
            ];
            for (qid, kind, pts, sid) in stored {
                conn.execute(
                    "INSERT INTO candidate_scores (id, candidate_id, question_id, kind, sign, label, points, max_points)
                     VALUES (?1, 'cand_1', ?2, ?3, 'Positive', 'label', ?4, 50.0)",
                    rusqlite::params![sid, qid, kind, pts],
                )
                .unwrap();
            }
            if judge_first {
                // Flip the physical row order: `rowid` follows insertion.
                conn.execute("UPDATE candidate_scores SET rowid = 9 WHERE id = 's2'", [])
                    .unwrap();
                conn.execute("UPDATE candidate_scores SET rowid = 2 WHERE id = 's1'", [])
                    .unwrap();
            }

            let out = std::env::temp_dir().join(format!("csv-test-{}.csv", uuid::Uuid::new_v4()));
            let summary = export_results_csv(&conn, "job_1", &out).unwrap();
            assert_eq!(summary.questions, 1, "both grades share one column");
            let csv = std::fs::read_to_string(&out).unwrap();
            assert_eq!(
                csv.lines().nth(1),
                Some("1,jane.pdf,Pdf,Extracted,40,50,false,40"),
                "the judge grade fills the Job ad column (judge_first={judge_first})"
            );

            let _ = std::fs::remove_file(&out);
        }
    }
}
