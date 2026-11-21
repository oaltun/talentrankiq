use libprocess_in_waves::WavesDb;
use libprocess_in_waves::league::MaxPriorityQueueByScore;

#[test]
fn league_orders_by_score() {
    let mut q = MaxPriorityQueueByScore::new();
    q.push("a".into(), 1.0);
    q.push("b".into(), 3.0);
    q.push("c".into(), 2.0);
    assert_eq!(q.pop(), Some("b".into()));
    assert_eq!(q.pop(), Some("c".into()));
    assert_eq!(q.pop(), Some("a".into()));
}

#[test]
fn db_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let dbpath = dir.path().join("t.db");
    let db = libprocess_in_waves::DbHandle::open_at(&dbpath).unwrap();
    db.insert_cv("h1", "text").unwrap();
    assert!(db.cv_has("h1").unwrap());
    db.insert_job_cv("job1", "h1").unwrap();
    assert_eq!(
        db.get_all_cv_hashes_for_job("job1").unwrap(),
        vec!["h1".to_string()]
    );
    assert_eq!(db.get_job_cv_count("job1").unwrap(), 1);
    assert_eq!(db.get_job_cv_embedded_count("job1", "m").unwrap(), 0);
}

/// The GUI table reads candidates/candidate_scores; the waves pipeline scores
/// cv/cv_similarity. `write_back_candidate_scores` bridges the two (bug.md):
/// deep-embedded candidates get an 'Ad' score row, quick-only ones a 'Scout'
/// row, points are the job-ad similarity on a 0..100 scale, and rank orders
/// by points desc with the file-name tie-break.
#[test]
fn write_back_populates_points_and_rank_from_waves_similarity() {
    let dir = tempfile::tempdir().unwrap();
    let dbpath = dir.path().join("triqui.db");
    // Waves schema first (file_state / cv / cv_similarity / ...).
    let db = libprocess_in_waves::DbHandle::open_at(&dbpath).unwrap();
    // The serve process additionally runs the libtriqui-core schema+migrations;
    // replicate the parts the write-back touches.
    let conn = rusqlite::Connection::open(&dbpath).unwrap();
    conn.execute_batch(libtriqui_core::db::SCHEMA).unwrap();
    conn.execute_batch(
        "ALTER TABLE candidates ADD COLUMN points REAL;
         ALTER TABLE candidates ADD COLUMN points_max REAL;",
    )
    .unwrap();

    conn.execute("INSERT INTO jobs (id, title) VALUES ('j1', 'job')", [])
        .unwrap();
    let cands = [
        ("cand_a", "/cv/a.txt", "a.txt", "ha"),
        ("cand_b", "/cv/b.txt", "b.txt", "hb"),
        ("cand_c", "/cv/c.txt", "c.txt", "hc"),
        ("cand_d", "/cv/d.txt", "d.txt", "hd"),
    ];
    for (id, path, name, hash) in cands {
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES (?1, 'j1', ?2, ?3, 'Txt', 'Extracted')",
            rusqlite::params![id, path, name],
        )
        .unwrap();
        db.upsert_file_state(path, hash).unwrap();
        db.insert_cv(hash, &format!("text {hash}")).unwrap();
    }

    // Deep + quick similarity to the job ad for a; deep-only for b;
    // quick-only for c; d has none.
    db.upsert_similarity("ha", "deep", "ad1", 0.9).unwrap();
    db.upsert_similarity("ha", "quick", "ad1", 0.5).unwrap();
    db.upsert_similarity("hb", "deep", "ad1", 0.7).unwrap();
    db.upsert_similarity("hc", "quick", "ad1", 0.4).unwrap();

    let written = db
        .write_back_candidate_scores("j1", "quick", "deep", "ad1")
        .unwrap();
    assert_eq!(written, 3);

    let points = |id: &str| -> Option<f64> {
        conn.query_row("SELECT points FROM candidates WHERE id = ?1", [id], |r| {
            r.get(0)
        })
        .unwrap()
    };
    let rank = |id: &str| -> Option<i64> {
        conn.query_row("SELECT rank FROM candidates WHERE id = ?1", [id], |r| {
            r.get(0)
        })
        .unwrap()
    };
    let score_rows = |id: &str| -> Vec<(String, f64, f64)> {
        let mut stmt = conn
            .prepare(
                "SELECT kind, points, max_points FROM candidate_scores
                  WHERE candidate_id = ?1 AND question_id = 'ad'",
            )
            .unwrap();
        stmt.query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };

    assert_eq!(points("cand_a"), Some(90.0));
    assert_eq!(points("cand_b"), Some(70.0));
    assert_eq!(points("cand_c"), Some(40.0));
    assert_eq!(points("cand_d"), None, "no similarity means no points");
    assert_eq!(rank("cand_a"), Some(1));
    assert_eq!(rank("cand_b"), Some(2));
    assert_eq!(rank("cand_c"), Some(3));
    assert_eq!(rank("cand_d"), None);
    assert_eq!(
        score_rows("cand_a"),
        vec![("Ad".to_string(), 90.0, 100.0)],
        "deep-embedded candidate gets an Ad row"
    );
    assert_eq!(
        score_rows("cand_c"),
        vec![("Scout".to_string(), 40.0, 100.0)],
        "quick-only candidate gets a Scout row"
    );
    assert!(score_rows("cand_d").is_empty());

    // Idempotent: a second pass replaces the same rows instead of duplicating.
    let written = db
        .write_back_candidate_scores("j1", "quick", "deep", "ad1")
        .unwrap();
    assert_eq!(written, 3);
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidate_scores cs
              JOIN candidates c ON c.id = cs.candidate_id
             WHERE c.job_id = 'j1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 3);
    assert_eq!(rank("cand_a"), Some(1));
}

/// bug.md: the waves write-back only persists the job-ad similarity, so a
/// criterion the pipeline computed and stored in `cv_similarity` never reaches
/// `candidate_scores`, cannot change `points`/`rank`, and the total sits on a
/// hard-coded 0..100 scale instead of the job's rubric currency.
///
/// Rubric here: ad 50 + feedback-job 10 + feedback-profile 5 + criterion 5 = 70.
#[test]
fn write_back_scores_criteria_and_uses_the_job_rubric_currency() {
    let dir = tempfile::tempdir().unwrap();
    let dbpath = dir.path().join("triqui.db");
    // The serve process runs the libtriqui-core schema first (it owns `criteria`
    // with `sign`); the waves schema then adds its own tables around it.
    let conn = rusqlite::Connection::open(&dbpath).unwrap();
    conn.execute_batch(libtriqui_core::db::SCHEMA).unwrap();
    let db = libprocess_in_waves::DbHandle::open_at(&dbpath).unwrap();
    conn.execute_batch(
        "ALTER TABLE candidates ADD COLUMN points REAL;
         ALTER TABLE candidates ADD COLUMN points_max REAL;
         ALTER TABLE jobs ADD COLUMN ad_weight REAL NOT NULL DEFAULT 1.0;
         ALTER TABLE jobs ADD COLUMN fb_job_weight REAL NOT NULL DEFAULT 0.0;
         ALTER TABLE jobs ADD COLUMN fb_profile_weight REAL NOT NULL DEFAULT 0.0;",
    )
    .unwrap();

    conn.execute(
        "INSERT INTO jobs (id, title, ad_weight, fb_job_weight, fb_profile_weight)
         VALUES ('j1', 'job', 50.0, 10.0, 5.0)",
        [],
    )
    .unwrap();
    let crit_text = "Kubernetes";
    let crit_id = "crit_1";
    conn.execute(
        "INSERT INTO criteria (id, job_id, text, sign, weight) VALUES (?1, 'j1', ?2, 'Positive', 5.0)",
        rusqlite::params![crit_id, crit_text],
    )
    .unwrap();

    let cands = [
        ("cand_a", "/cv/a.txt", "a.txt", "ha"),
        ("cand_b", "/cv/b.txt", "b.txt", "hb"),
        ("cand_c", "/cv/c.txt", "c.txt", "hc"),
    ];
    for (id, path, name, hash) in cands {
        conn.execute(
            "INSERT INTO candidates (id, job_id, source_file_path, file_name, file_type, extraction_status)
             VALUES (?1, 'j1', ?2, ?3, 'Txt', 'Extracted')",
            rusqlite::params![id, path, name],
        )
        .unwrap();
        db.upsert_file_state(path, hash).unwrap();
        db.insert_cv(hash, &format!("text {hash}")).unwrap();
    }

    let ad_hash = "ad1";
    let crit_hash = libprocess_in_waves::hash_text(crit_text);
    // deep similarity to (ad, criterion) per candidate.
    db.upsert_similarity("ha", "deep", ad_hash, 0.8).unwrap();
    db.upsert_similarity("ha", "deep", &crit_hash, 0.6).unwrap();
    db.upsert_similarity("hb", "deep", ad_hash, 0.7).unwrap();
    db.upsert_similarity("hb", "deep", &crit_hash, 1.0).unwrap();
    db.upsert_similarity("hc", "deep", ad_hash, 0.75).unwrap();
    db.upsert_similarity("hc", "deep", &crit_hash, 0.0).unwrap();

    db.write_back_candidate_scores("j1", "quick", "deep", ad_hash)
        .unwrap();

    let points = |id: &str| -> Option<f64> {
        conn.query_row("SELECT points FROM candidates WHERE id = ?1", [id], |r| {
            r.get(0)
        })
        .unwrap()
    };
    let max_points = |id: &str| -> Option<f64> {
        conn.query_row(
            "SELECT points_max FROM candidates WHERE id = ?1",
            [id],
            |r| r.get(0),
        )
        .unwrap()
    };
    let rank = |id: &str| -> Option<i64> {
        conn.query_row("SELECT rank FROM candidates WHERE id = ?1", [id], |r| {
            r.get(0)
        })
        .unwrap()
    };
    let score_rows = |id: &str| -> Vec<(String, String, f64, f64)> {
        let mut stmt = conn
            .prepare(
                "SELECT question_id, kind, points, max_points FROM candidate_scores
                  WHERE candidate_id = ?1 ORDER BY question_id",
            )
            .unwrap();
        stmt.query_map([id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
    };

    // Each candidate carries the ad row and the criterion row, on the rubric
    // scale: ad_weight · sim + criterion weight · sim.
    assert_eq!(
        score_rows("cand_a"),
        vec![
            ("ad".to_string(), "Ad".to_string(), 40.0, 50.0),
            (
                crit_id.to_string(),
                "Criterion".to_string(),
                3.0,
                5.0
            ),
        ],
        "the criterion's similarity must reach candidate_scores"
    );
    assert_eq!(points("cand_a"), Some(43.0));
    assert_eq!(points("cand_b"), Some(40.0));
    assert_eq!(points("cand_c"), Some(37.5));
    assert_eq!(
        max_points("cand_a"),
        Some(70.0),
        "points_max is the job's rubric capacity (ad 50 + fb 10 + fb 5 + crit 5)"
    );

    // Criteria move the order: ad-only ranking is a(40) > c(37.5) > b(35),
    // but with the criterion it is a(43) > b(40) > c(37.5).
    assert_eq!(rank("cand_a"), Some(1));
    assert_eq!(rank("cand_b"), Some(2));
    assert_eq!(rank("cand_c"), Some(3));

    // Idempotent: the criterion row is replaced, not duplicated.
    db.write_back_candidate_scores("j1", "quick", "deep", ad_hash)
        .unwrap();
    assert_eq!(score_rows("cand_a"), score_rows("cand_a"));
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM candidate_scores cs
              JOIN candidates c ON c.id = cs.candidate_id
             WHERE c.job_id = 'j1' AND cs.question_id = ?1",
            [crit_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 3, "one criterion row per candidate");
}

#[test]
fn write_back_is_noop_without_candidates_table() {
    // A waves-only database (no libtriqui-core schema) must not fail the wave.
    let dir = tempfile::tempdir().unwrap();
    let db = libprocess_in_waves::DbHandle::open_at(&dir.path().join("triqui.db")).unwrap();
    let written = db
        .write_back_candidate_scores("j1", "quick", "deep", "ad1")
        .unwrap();
    assert_eq!(written, 0);
}
