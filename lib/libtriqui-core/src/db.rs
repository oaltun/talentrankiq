//! SQLite persistence for the TalentRankIQ domain (Baseline ranking).
//!
//! One shared connection (`DbConnection`) serves command-thread reads/writes;
//! long background imports open their own connection (WAL allows concurrent
//! readers + one writer). Timestamps are produced by SQL `datetime('now')`,
//! so no time crate is needed in Rust.
//!
//! This module is Tauri-free: callers open a connection at an explicit path
//! ([`open_connection_at`]) or via [`default_db_path`]. The GUI keeps its own
//! `db_path(app)` (Tauri app-data dir) and delegates to [`open_connection_at`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OptionalExtension};

/// Shared SQLite connection for command handlers.
pub struct DbConnection(pub Mutex<Connection>);

/// Jobs with an active background import (drives `importing`). The `Arc` lets
/// the import thread hold onto the set while the Tauri-managed state stays put.
pub struct ImportState(pub Arc<Mutex<HashSet<String>>>);

impl ImportState {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HashSet::new())))
    }
}

impl Default for ImportState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create/migrate the schema. Idempotent — run on every connection open.
pub const SCHEMA: &str = "
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS jobs (
  id                   TEXT PRIMARY KEY,
  title                TEXT NOT NULL,
  job_ad_text          TEXT,
  import_directory_path TEXT,
  triqui_size       INTEGER NOT NULL DEFAULT 10,
  archived             INTEGER NOT NULL DEFAULT 0,
  created_at           TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at           TEXT NOT NULL DEFAULT (datetime('now')),
  last_ranked_at       TEXT,
  job_ad_path          TEXT,
  inputs_updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS criteria (
  id         TEXT PRIMARY KEY,
  job_id     TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  text       TEXT NOT NULL,
  sign       TEXT NOT NULL,
  weight     REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS candidates (
  id               TEXT PRIMARY KEY,
  job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  source_file_path TEXT NOT NULL,
  file_name        TEXT NOT NULL,
  file_type        TEXT NOT NULL,
  extraction_status TEXT NOT NULL,
  extracted_text   TEXT,
  rank             INTEGER,
  imported_at      TEXT NOT NULL DEFAULT (datetime('now')),
  ranked_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_candidates_job  ON candidates(job_id);
CREATE INDEX IF NOT EXISTS idx_criteria_job    ON criteria(job_id);
CREATE INDEX IF NOT EXISTS idx_candidates_rank ON candidates(job_id, rank);

-- Baseline ranking: per-candidate explanation rows (one per Candidate x Criterion).
CREATE TABLE IF NOT EXISTS criterion_matches (
  id           TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  criterion_id TEXT NOT NULL REFERENCES criteria(id)   ON DELETE CASCADE,
  verdict      TEXT NOT NULL,
  contribution REAL NOT NULL,
  evidence     TEXT,
  evaluated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cmatches_candidate ON criterion_matches(candidate_id);

-- Rubric scoring: per-candidate, per-question points (one row per Candidate x Question).
-- A question is the job ad ('ad'), a criterion (criterion id), or a feedback signal
-- ('fb_job' / 'fb_profile'). Rebuilt wholesale on every sort.
CREATE TABLE IF NOT EXISTS candidate_scores (
  id           TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  question_id  TEXT NOT NULL,
  kind         TEXT NOT NULL,
  sign         TEXT NOT NULL,
  label        TEXT NOT NULL,
  points       REAL NOT NULL,
  max_points   REAL NOT NULL,
  evaluated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cs_candidate ON candidate_scores(candidate_id);

-- Learned preference state. MVP has one implicit Profile; eprofile/b NULL = zero vector.
CREATE TABLE IF NOT EXISTS profiles (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  eprofile   BLOB,
  b          BLOB,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- modelcaller engine runtime/config (singleton row).
CREATE TABLE IF NOT EXISTS runtime (
  id          TEXT PRIMARY KEY CHECK (id = 'singleton'),
  status      TEXT NOT NULL DEFAULT 'Stopped',
  model       TEXT,
  config_json TEXT,
  error       TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Global embedding cache keyed by (model, content_hash). Lets identical documents
-- (same text, even across jobs/re-imports) reuse a vector instead of re-embedding.
CREATE TABLE IF NOT EXISTS embedding_cache (
  model        TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector       BLOB NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (model, content_hash)
);

-- Which backend arm (ort / mlx) last wrote a model's vectors. Vectors from
-- the two arms are NOT interchangeable (the 270m MLX/ONNX cosine agreement is
-- ~0.94, not 1.0), so when the recorded arm differs from the arm about to
-- serve, every slug-keyed vector artifact is cleared before any of it is
-- read. The scout's slug never flips (it declares no mlx variant), which is
-- what keeps scout rows safe by construction.
CREATE TABLE IF NOT EXISTS model_runtime (
  model      TEXT PRIMARY KEY,
  backend    TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Judge verdict memoization: one row per pairwise tournament verdict
-- p(doc_a > doc_b | ad, artifact, embedder). Keys are content hashes (the ad
-- text, the two candidate texts) plus the embedding model and the judge
-- artifact content hash, so identical CVs across jobs share verdicts and any
-- input change misses. Sorted pairs are stored canonically (doc_a < doc_b);
-- the reverse orientation is 1 - prob.
CREATE TABLE IF NOT EXISTS judge_pair_verdicts (
  model         TEXT NOT NULL,
  artifact_hash TEXT NOT NULL,
  ad_hash       TEXT NOT NULL,
  doc_a         TEXT NOT NULL,
  doc_b         TEXT NOT NULL,
  prob          REAL NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (model, artifact_hash, ad_hash, doc_a, doc_b)
);

-- Report of the last rank of a job (what ran, what was memoized). Read-model
-- for the state contract: tests and the gui assert on reported behavior.
CREATE TABLE IF NOT EXISTS sort_reports (
  job_id              TEXT PRIMARY KEY,
  judge_mode          TEXT NOT NULL,
  judge_pairs_fresh   INTEGER NOT NULL DEFAULT 0,
  judge_pairs_cached  INTEGER NOT NULL DEFAULT 0,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cascade rank state per job (progressive-ranking design §10).
-- `sort` is the last completed SortState (Stage0, Stage1, Stage2).
-- `sort_reports_json` stores the array of SortReport for the gui to
-- reconstruct the wave-by-wave history. Updated atomically by the
-- background cascade task so the gui sees a clean, monotonic sequence.
CREATE TABLE IF NOT EXISTS rank_state (
  job_id           TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  sort             TEXT NOT NULL,
  sort_reports_json TEXT NOT NULL DEFAULT '[]',
  complete         INTEGER NOT NULL DEFAULT 0,
  cancelled        INTEGER NOT NULL DEFAULT 0,
  updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-candidate chunk embeddings (long texts split by libembedrank chunking).
-- Written at import time so Re-rank is pure scoring (cosine/maxsim + sort) with
-- no model calls. `vector` holds the raw little-endian f32 chunk vector;
-- `content_hash` keys the chunk text into `embedding_cache`. The candidate
-- centroid is still stored in `candidates.vector` as a fallback for models
-- without a tokenizer (non-chunked path).
-- Keyed by (candidate, MODEL, index): the cascade's scout embedder and the
-- full embedder each keep their own chunks for the same candidate side by side
-- (progressive-ranking design §7), so the key must include `vector_model`.
CREATE TABLE IF NOT EXISTS candidate_chunks (
  id           TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  chunk_index  INTEGER NOT NULL,
  chunk_text   TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector       BLOB NOT NULL,
  vector_model TEXT NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (candidate_id, vector_model, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_cc_candidate ON candidate_chunks(candidate_id);
CREATE INDEX IF NOT EXISTS idx_cc_candidate_model ON candidate_chunks(candidate_id, vector_model);

-- Learned feedback vectors, one row per (job | profile) x embedding model.
-- Different models' vectors are not comparable, so each model learns and ranks
-- against its own vectors; switching models never mixes or clobbers them.
CREATE TABLE IF NOT EXISTS job_feedback_vectors (
  job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  model        TEXT NOT NULL,
  ejobfeedback BLOB,
  updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (job_id, model)
);

CREATE TABLE IF NOT EXISTS profile_vectors (
  profile_id TEXT NOT NULL,
  model      TEXT NOT NULL,
  eprofile   BLOB,
  b          BLOB,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (profile_id, model)
);

-- ---------------------------------------------------------------------------
-- Scoring v3 (gui/design/app1/cascade-simple.md): one table per fact.
-- ---------------------------------------------------------------------------

-- Rubric question vectors, one row per (job, question, space). A 'question' is
-- the job ad ('ad'), a feedback signal ('fb_job' / 'fb_profile') or a criterion
-- (its own id); a 'space' is the model slug whose vectors the row holds.
--
-- This replaces four single-space slots (jobs.ead, criteria.vector,
-- job_feedback_vectors.ejobfeedback, profile_vectors.eprofile) with one
-- multi-space store, which is what lets the cheap scout score the WHOLE rubric
-- and not just the ad. Keyed by space so swapping models never mixes two
-- embeddings.
CREATE TABLE IF NOT EXISTS rubric_vectors (
  job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  question_id TEXT NOT NULL,
  model       TEXT NOT NULL,
  vector      BLOB NOT NULL,
  updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (job_id, question_id, model)
);

-- One grade per (candidate, space, question): the cosine of the question's
-- vector against the candidate's best chunk in that space. The 'ad' row of a
-- space is that space's fit score — the ad slot of the total, unless the
-- judge has produced a win-rate for the candidate.
--
-- Replaces scout_scores (which could only hold the scout's ad similarity) so
-- there is exactly one place a stage grade lives. Rescore reads this table and
-- never writes it; each stage writes only its own space.
CREATE TABLE IF NOT EXISTS stage_scores (
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  model        TEXT NOT NULL,
  question_id  TEXT NOT NULL,
  score        REAL NOT NULL,
  updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (candidate_id, model, question_id)
);
CREATE INDEX IF NOT EXISTS idx_stage_question ON stage_scores(model, question_id);

-- The judge pool: the candidates a run compares against each other. Frozen when
-- the run opens it, because judge win-rates are SET-RELATIVE — a pool that
-- re-picked itself every wave would move a candidate's grade for reasons that
-- have nothing to do with the candidate.
CREATE TABLE IF NOT EXISTS judge_pool (
  job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
  position     INTEGER NOT NULL,
  PRIMARY KEY (job_id, candidate_id)
);
";

/// The Tauri app identifier — the GUI's app-data dir is `<data_root>/<identifier>`.
pub const APP_IDENTIFIER: &str = "com.oaltun.triqui";

/// Default DB location matching the GUI's Tauri app-data dir:
/// `<data_root>/com.oaltun.triqui/triqui.db`, where `<data_root>` follows
/// the per-OS convention (`$XDG_DATA_HOME`/`~/.local/share` on Linux,
/// `~/Library/Application Support` on macOS, `%APPDATA%` on Windows).
/// Does not create anything; use [`open_connection_at`] to create dirs.
pub fn default_db_path() -> PathBuf {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
            })
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join(APP_IDENTIFIER)
        .join("triqui.db")
}

/// Open a connection at `path` (creating parent dirs) and ensure the schema exists.
pub fn open_connection_at(path: &Path) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create database directory: {e}"))?;
    }
    let conn = Connection::open(path).map_err(|e| format!("Failed to open database: {e}"))?;
    conn.execute_batch(SCHEMA)
        .map_err(|e| format!("Failed to migrate database: {e}"))?;
    migrate(&conn)?;
    Ok(conn)
}

/// Add a column only if it is not already present.
/// `CREATE TABLE IF NOT EXISTS` (in `SCHEMA`) handles new tables; columns added to
/// existing tables need this guard because rusqlite errors on a duplicate column.
fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    type_def: &str,
) -> Result<(), String> {
    let exists: bool = {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|e| e.to_string())?;
        // Collect names first so the statement borrow ends before any ALTER.
        let names: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(1))
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();
        names.iter().any(|n| n == column)
    };
    if !exists {
        conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {type_def}"),
            [],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Column names of every UNIQUE index on `table`, in declared order.
fn unique_index_columns(conn: &Connection, table: &str) -> Result<Vec<Vec<String>>, String> {
    let names: Vec<String> = {
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_list({table})"))
            .map_err(|e| e.to_string())?;
        // Collect fully inside this block so the statement borrow ends before
        // the next PRAGMA. `unique` is column 2 of index_list.
        let mut out: Vec<String> = stmt
            .query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, i64>(2)?)))
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .filter(|(_, uniq)| *uniq == 1)
            .map(|(name, _)| name)
            .collect();
        out.sort();
        out
    };
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_info({name})"))
            .map_err(|e| e.to_string())?;
        let cols: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(2))
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();
        out.push(cols);
    }
    Ok(out)
}

/// Make `candidate_chunks` multi-model.
///
/// The table shipped `UNIQUE (candidate_id, chunk_index)`, which silently makes
/// it single-model: the cascade's scout embedder cannot store its own chunks for
/// a candidate that already has full-embedder chunks (progressive-ranking
/// design §7 has both live at once). SQLite cannot drop a UNIQUE constraint
/// with ALTER, so rebuild the table with `(candidate_id, vector_model,
/// chunk_index)`, preserving every row.
///
/// Idempotent: returns immediately when the new key is already present, so it
/// is safe to run on every connection open.
fn migrate_candidate_chunks_multimodel(conn: &Connection) -> Result<(), String> {
    const NEW_KEY: [&str; 3] = ["candidate_id", "vector_model", "chunk_index"];
    for cols in unique_index_columns(conn, "candidate_chunks")? {
        if cols.len() == NEW_KEY.len() && cols.iter().zip(NEW_KEY).all(|(a, b)| a == b) {
            return Ok(()); // already multi-model
        }
    }
    log::info!("db: rebuilding candidate_chunks for multi-model chunk storage");
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    tx.execute_batch(
        "CREATE TABLE candidate_chunks_new (
           id           TEXT PRIMARY KEY,
           candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
           chunk_index  INTEGER NOT NULL,
           chunk_text   TEXT NOT NULL,
           content_hash TEXT NOT NULL,
           vector       BLOB NOT NULL,
           vector_model TEXT NOT NULL,
           created_at   TEXT NOT NULL DEFAULT (datetime('now')),
           UNIQUE (candidate_id, vector_model, chunk_index)
         );
         INSERT INTO candidate_chunks_new
           (id, candidate_id, chunk_index, chunk_text, content_hash, vector, vector_model, created_at)
         SELECT id, candidate_id, chunk_index, chunk_text, content_hash, vector, vector_model, created_at
           FROM candidate_chunks;
         DROP TABLE candidate_chunks;
         ALTER TABLE candidate_chunks_new RENAME TO candidate_chunks;
         CREATE INDEX IF NOT EXISTS idx_cc_candidate ON candidate_chunks(candidate_id);
         CREATE INDEX IF NOT EXISTS idx_cc_candidate_model ON candidate_chunks(candidate_id, vector_model);",
    )
    .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())
}

/// Migrate the scoring facts into their multi-space homes (scoring v3,
/// `gui/design/app1/cascade-simple.md`).
///
/// Four single-space slots become one multi-space table (`rubric_vectors`), and
/// `scout_scores` — which could only ever hold the scout's ad similarity —
/// becomes a `(candidate, space, question)` row of `stage_scores`.
///
/// Every statement is `INSERT OR IGNORE` off an existing value, so the whole
/// migration is idempotent and re-running it never overwrites a fresher vector
/// written by a live stage. The legacy columns are only cleared once their
/// contents are known to be in `rubric_vectors`, and are left in place (unused)
/// rather than dropped: SQLite cannot drop a column cheaply and they are harmless
/// once nothing reads them.
fn migrate_scoring_v3(conn: &Connection) -> Result<(), String> {
    // Job ad.
    conn.execute(
        "INSERT OR IGNORE INTO rubric_vectors (job_id, question_id, model, vector)
         SELECT id, 'ad', ead_model, ead FROM jobs
         WHERE ead IS NOT NULL AND ead_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    // Criteria.
    conn.execute(
        "INSERT OR IGNORE INTO rubric_vectors (job_id, question_id, model, vector)
         SELECT c.job_id, c.id, c.vector_model, c.vector FROM criteria c
         WHERE c.vector IS NOT NULL AND c.vector_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    // The per-job learned feedback row. This table is already per-model, so it is
    // copied so `rubric_vectors` is the single read path for a question's vector.
    //
    // The *profile* row is deliberately NOT copied: it is one global vector, not
    // one per job, and `rubric_vectors.job_id` is a foreign key — copying it under
    // the literal profile id would fail the constraint on the next connection open.
    // `rubric::ensure_vectors` copies it into a job's rubric lazily, which is also
    // the only way that avoids writing every job on every like.
    conn.execute(
        "INSERT OR IGNORE INTO rubric_vectors (job_id, question_id, model, vector)
         SELECT job_id, 'fb_job', model, ejobfeedback FROM job_feedback_vectors
         WHERE ejobfeedback IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    // The scout's ad similarity, as a stage grade of the scout space.
    if conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='scout_scores'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0
    {
        conn.execute(
            "INSERT OR IGNORE INTO stage_scores (candidate_id, model, question_id, score)
             SELECT candidate_id, model, 'ad', score FROM scout_scores",
            [],
        )
        .map_err(|e| e.to_string())?;
        conn.execute("DROP TABLE scout_scores", [])
            .map_err(|e| e.to_string())?;
        log::info!("db: scoring v3 — scout_scores folded into stage_scores");
    }
    Ok(())
}

/// Idempotent migration run once per connection open, after `SCHEMA`.
///
/// Ranking vectors are stored as raw little-endian f32 byte arrays; NULL means
/// "not embedded yet" (candidate) or "zero" (eprofile/b). `vector_model`/`ead_model`
/// stamp which embedding model produced a vector so a model swap is detectable.
pub(crate) fn migrate(conn: &Connection) -> Result<(), String> {
    add_column_if_missing(conn, "candidates", "vector", "BLOB")?;
    add_column_if_missing(conn, "candidates", "vector_model", "TEXT")?;
    add_column_if_missing(conn, "candidates", "match_score", "REAL")?;
    add_column_if_missing(conn, "candidates", "explanation_summary", "TEXT")?;
    add_column_if_missing(conn, "candidates", "content_hash", "TEXT")?;
    // `inputs_updated_at` is bumped only when ranking inputs change (ad / N / criteria),
    // not on a title edit — so title changes don't mark the triqui stale.
    add_column_if_missing(conn, "jobs", "inputs_updated_at", "TEXT")?;
    let _ = conn.execute(
        "UPDATE jobs SET inputs_updated_at = COALESCE(inputs_updated_at, created_at)",
        [],
    );
    // Monotone dirty-tracking versions: `inputs_version` bumps with every
    // ranking-input change; `sort` records `ranked_version = inputs_version`.
    // `inputs_version != ranked_version` (or NULL ranked_version) = dirty.
    // Versions, not timestamps: `datetime('now')` has 1-second resolution, so
    // an edit in the same second as a rank would wrongly read as clean.
    add_column_if_missing(conn, "jobs", "inputs_version", "INTEGER NOT NULL DEFAULT 0")?;
    add_column_if_missing(conn, "jobs", "ranked_version", "INTEGER")?;
    // The job ad lives in a plain text file the user edits in their own editor
    // (see `job_ad`); this is where the chosen path is remembered. NULL means
    // "the default `<cv-root>/<job>/job-ad.txt` has not been resolved yet".
    add_column_if_missing(conn, "jobs", "job_ad_path", "TEXT")?;
    add_column_if_missing(conn, "jobs", "ead", "BLOB")?;
    add_column_if_missing(conn, "jobs", "ead_model", "TEXT")?;
    add_column_if_missing(conn, "jobs", "ejobguess", "BLOB")?;
    add_column_if_missing(conn, "criteria", "vector", "BLOB")?;
    add_column_if_missing(conn, "criteria", "vector_model", "TEXT")?;
    // Rubric weights (job-ad max-points + the two feedback rows) + per-job feedback vector.
    add_column_if_missing(conn, "jobs", "ad_weight", "REAL NOT NULL DEFAULT 1.0")?;
    add_column_if_missing(conn, "jobs", "fb_job_weight", "REAL NOT NULL DEFAULT 0.0")?;
    add_column_if_missing(
        conn,
        "jobs",
        "fb_profile_weight",
        "REAL NOT NULL DEFAULT 0.0",
    )?;
    add_column_if_missing(conn, "jobs", "ejobfeedback", "BLOB")?;
    add_column_if_missing(conn, "jobs", "ejobfeedback_model", "TEXT")?;
    // Rubric points per candidate (replaces the single match_score in the read models).
    add_column_if_missing(conn, "candidates", "points", "REAL")?;
    add_column_if_missing(conn, "candidates", "points_max", "REAL")?;
    add_column_if_missing(conn, "candidates", "liked", "INTEGER NOT NULL DEFAULT 0")?;
    // Cross-job feedback vector (parity with ead_model / ejobfeedback_model).
    add_column_if_missing(conn, "profiles", "eprofile_model", "TEXT")?;
    // Maxsim chunk explanation texts (JSON array) per rubric score row.
    add_column_if_missing(conn, "candidate_scores", "chunk_texts", "TEXT")?;
    // Read/unread marking (CV Inspector opened) — progressive-ranking design §13.
    add_column_if_missing(conn, "candidates", "viewed_at", "TEXT")?;
    // Cascade read-model: how the last rank ran (§10 of
    // gui/design/app1/progressive-ranking.md). `mode`: full | waves.
    add_column_if_missing(conn, "sort_reports", "mode", "TEXT NOT NULL DEFAULT 'full'")?;
    add_column_if_missing(conn, "sort_reports", "waves", "INTEGER NOT NULL DEFAULT 0")?;
    add_column_if_missing(
        conn,
        "sort_reports",
        "complete",
        "INTEGER NOT NULL DEFAULT 1",
    )?;
    add_column_if_missing(
        conn,
        "sort_reports",
        "cancelled",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    migrate_candidate_chunks_multimodel(conn)?;

    // The lexical stage-0 tables are gone (progressive-ranking design §21):
    // drop the raw-score table and the display rows it wrote.
    conn.execute("DROP TABLE IF EXISTS lex_scores", [])
        .map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM candidate_scores WHERE question_id = 'lex'", [])
        .map_err(|e| e.to_string())?;

    migrate_scoring_v3(conn)?;

    // rank_state columns (table created by SCHEMA; columns may exist if
    // a stale DB had an earlier CREATE with fewer columns)
    add_column_if_missing(conn, "rank_state", "sort", "TEXT NOT NULL DEFAULT 'Stage0'")?;
    add_column_if_missing(
        conn,
        "rank_state",
        "sort_reports_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    add_column_if_missing(conn, "rank_state", "complete", "INTEGER NOT NULL DEFAULT 0")?;
    add_column_if_missing(
        conn,
        "rank_state",
        "cancelled",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    add_column_if_missing(
        conn,
        "rank_state",
        "updated_at",
        "TEXT NOT NULL DEFAULT (datetime('now'))",
    )?;

    // One implicit Profile (MVP); eprofile/b stay NULL => treated as the zero vector.
    conn.execute(
        "INSERT OR IGNORE INTO profiles (id, name) VALUES ('profile_default', 'Default')",
        [],
    )
    .map_err(|e| e.to_string())?;
    // One-time migration: legacy single-slot learned vectors (one blob, model-tagged)
    // → per-model rows. Learning under a second model used to overwrite the first
    // model's vectors; per-model rows keep every model's feedback separate. The
    // UPDATE makes the copy idempotent (sources are NULL afterwards).
    conn.execute(
        "INSERT OR IGNORE INTO job_feedback_vectors (job_id, model, ejobfeedback)
         SELECT id, ejobfeedback_model, ejobfeedback FROM jobs
         WHERE ejobfeedback IS NOT NULL AND ejobfeedback_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE jobs SET ejobfeedback = NULL, ejobfeedback_model = NULL
         WHERE ejobfeedback IS NOT NULL AND ejobfeedback_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT OR IGNORE INTO profile_vectors (profile_id, model, eprofile, b)
         SELECT id, eprofile_model, eprofile, b FROM profiles
         WHERE eprofile IS NOT NULL AND eprofile_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE profiles SET eprofile = NULL, eprofile_model = NULL
         WHERE eprofile IS NOT NULL AND eprofile_model IS NOT NULL",
        [],
    )
    .map_err(|e| e.to_string())?;
    // Runtime singleton (model/config persistence; live status is in-memory).
    conn.execute(
        "INSERT OR IGNORE INTO runtime (id, status) VALUES ('singleton', 'Stopped')",
        [],
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// Map a generic SQLite error to a `DomainError::DbError`.
pub fn db_err(e: rusqlite::Error) -> crate::types::DomainError {
    crate::types::DomainError::DbError {
        message: e.to_string(),
    }
}

/// Convenience: count rows in `table` (used by status-style queries).
pub fn count(conn: &Connection, table: &str) -> Result<i64, String> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    conn.query_row(&sql, [], |r| r.get(0))
        .map_err(|e| e.to_string())
}

/// Ids of all jobs with basic counts, for status-style listings.
pub fn job_overview(conn: &Connection) -> Result<Vec<JobOverview>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT j.id, j.title, j.archived,
                    (SELECT COUNT(*) FROM candidates c WHERE c.job_id = j.id),
                    (SELECT COUNT(*) FROM candidates c WHERE c.job_id = j.id AND c.rank IS NOT NULL)
             FROM jobs j ORDER BY j.created_at",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)? != 0,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

/// Row shape of [`job_overview`].
pub type JobOverview = (String, String, bool, i64, i64);

/// Read the job's ad_weight (default 1.0 if column missing).
pub fn job_weight(conn: &Connection, job_id: &str) -> Result<f64, String> {
    let w: f64 = conn
        .query_row(
            "SELECT COALESCE(ad_weight, 1.0) FROM jobs WHERE id = ?1",
            rusqlite::params![job_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    Ok(w)
}

/// Upsert the cascade rank state for a job.
pub fn set_rank_state(
    conn: &Connection,
    job_id: &str,
    sort: &str,
    sort_reports_json: &str,
    complete: bool,
    cancelled: bool,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO rank_state (job_id, sort, sort_reports_json, complete, cancelled, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
         ON CONFLICT(job_id) DO UPDATE SET
            sort = excluded.sort,
            sort_reports_json = excluded.sort_reports_json,
            complete = excluded.complete,
            cancelled = excluded.cancelled,
            updated_at = datetime('now')",
        rusqlite::params![
            job_id,
            sort,
            sort_reports_json,
            complete as i32,
            cancelled as i32
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Read the cascade rank state for a job.
pub fn rank_state_for_job(conn: &Connection, job_id: &str) -> Result<RankStateRow, String> {
    let row = conn
        .query_row(
            "SELECT sort, sort_reports_json, complete, cancelled FROM rank_state WHERE job_id = ?1",
            rusqlite::params![job_id],
            |r| {
                Ok(RankStateRow {
                    sort: r.get(0)?,
                    sort_reports_json: r.get(1)?,
                    complete: r.get::<_, i32>(2)? != 0,
                    cancelled: r.get::<_, i32>(3)? != 0,
                })
            },
        )
        .map_err(|e| e.to_string())?;
    Ok(row)
}

/// Row returned by [`rank_state_for_job`].
pub struct RankStateRow {
    pub sort: String,
    pub sort_reports_json: String,
    pub complete: bool,
    pub cancelled: bool,
}

/// Mark an in-flight cascade for `job_id` as cancelled, so it stops at the next
/// wave boundary.
///
/// Called when a ranking input changes (criterion, weights, ad, CV set, embedder,
/// judge artifact). A running cascade is computing a ranking for inputs that no
/// longer exist, so the result would be stale the moment it landed; cancelling
/// is strictly better than finishing work nobody wants. The stored vectors and
/// stage scores are untouched — they stay valid and cache-keyed, so the
/// re-cascade picks up where this one stopped.
///
/// Returns true when a running cascade was actually marked (i.e. the caller had
/// something to cancel).
pub fn cancel_running_cascade(conn: &Connection, job_id: &str) -> Result<bool, String> {
    let affected = conn
        .execute(
            "UPDATE rank_state SET cancelled = 1, updated_at = datetime('now')
             WHERE job_id = ?1 AND complete = 0 AND cancelled = 0",
            rusqlite::params![job_id],
        )
        .map_err(|e| e.to_string())?;
    Ok(affected > 0)
}

/// At serve start no cascade thread exists yet, so any `rank_state` row that
/// still claims a run in flight (`complete = 0 AND cancelled = 0`) is a
/// leftover from a previous process — a quit or crash mid-cascade. Left as-is
/// it wedges the job: `get_rank_state` reports `running` forever and
/// `spawn_cascade` refuses to start a new run. Mark those rows cancelled (the
/// same terminal state a stop-short run writes) so every job starts unstuck.
///
/// Returns the number of stale rows recovered.
pub fn recover_stale_running_cascades(conn: &Connection) -> Result<usize, String> {
    let affected = conn
        .execute(
            "UPDATE rank_state SET cancelled = 1, updated_at = datetime('now')
             WHERE complete = 0 AND cancelled = 0",
            rusqlite::params![],
        )
        .map_err(|e| e.to_string())?;
    Ok(affected)
}

/// The `sort_reports` row for `job_id`, if one has been written.
pub fn last_sort_report(
    rt: &Arc<crate::runtime::EngineRuntime>,
    job_id: &str,
) -> Result<Option<crate::types::SortReport>, String> {
    let conn = rt.open_conn()?;
    conn.query_row(
        "SELECT judge_mode, judge_pairs_fresh, judge_pairs_cached, mode, waves, complete, cancelled, created_at
         FROM sort_reports WHERE job_id = ?1",
        rusqlite::params![job_id],
        |r| {
            Ok(crate::types::SortReport {
                job_id: job_id.to_string(),
                judge_mode: r.get(0)?,
                judge_pairs_fresh: r.get(1)?,
                judge_pairs_cached: r.get(2)?,
                mode: r.get(3)?,
                waves: r.get(4)?,
                complete: r.get(5)?,
                cancelled: r.get(6)?,
                created_at: r.get(7)?,
            })
        },
    )
    .optional()
    .map_err(|e| e.to_string())
}


// ---------------------------------------------------------------------------
// Scoring v3 accessors. `rubric_vectors` / `stage_scores` / `judge_pool` are the
// three facts rescore and the stages read; each has exactly one writer.
// ---------------------------------------------------------------------------

/// Upsert a question's vector for one space.
pub fn put_rubric_vector(
    conn: &Connection,
    job_id: &str,
    question_id: &str,
    model: &str,
    vector: &[f32],
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO rubric_vectors (job_id, question_id, model, vector)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(job_id, question_id, model) DO UPDATE SET
            vector = excluded.vector, updated_at = datetime('now')",
        rusqlite::params![
            job_id,
            question_id,
            model,
            crate::embed::vector_to_bytes(vector)
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// A job's question vectors for one space, one row per question id.
pub fn rubric_vectors(
    conn: &Connection,
    job_id: &str,
    model: &str,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut stmt = conn
        .prepare("SELECT question_id, vector FROM rubric_vectors WHERE job_id = ?1 AND model = ?2")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![job_id, model], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}

/// Upsert one stage grade — the cosine of a question against a candidate in one
/// space. Idempotent, so a stage re-running a pass overwrites its own numbers
/// and nobody else's.
pub fn put_stage_score(
    conn: &Connection,
    candidate_id: &str,
    model: &str,
    question_id: &str,
    score: f64,
) -> Result<(), String> {
    conn.execute(
        "INSERT INTO stage_scores (candidate_id, model, question_id, score)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(candidate_id, model, question_id) DO UPDATE SET
            score = excluded.score, updated_at = datetime('now')",
        rusqlite::params![candidate_id, model, question_id, score],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Bulk upsert of stage grades, in one transaction. A stage's whole pass is one
/// call, so a 2000-candidate × 5-question scout pass is 10 000 upserts in a
/// single commit rather than 10 000 round trips.
pub fn put_stage_scores(
    conn: &Connection,
    rows: &[(String, (String, String), f64)],
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    {
        let mut stmt = tx
            .prepare(
                "INSERT INTO stage_scores (candidate_id, model, question_id, score)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(candidate_id, model, question_id) DO UPDATE SET
                    score = excluded.score, updated_at = datetime('now')",
            )
            .map_err(|e| e.to_string())?;
        for (cid, (model, qid), score) in rows {
            stmt.execute(rusqlite::params![cid, model, qid, score])
                .map_err(|e| e.to_string())?;
        }
    }
    tx.commit().map_err(|e| e.to_string())
}

/// Every stage grade of every candidate of a job, in one space, keyed
/// `(candidate_id, question_id)`. This is what rescore reads: the whole
/// per-space scorecard in one query.
pub fn stage_scores_for_job(
    conn: &Connection,
    job_id: &str,
    model: &str,
) -> Result<std::collections::HashMap<(String, String), f64>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT ss.candidate_id, ss.question_id, ss.score
             FROM stage_scores ss
             JOIN candidates c ON c.id = ss.candidate_id
             WHERE c.job_id = ?1 AND ss.model = ?2",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![job_id, model], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|(cid, qid, s)| ((cid, qid), s))
        .collect())
}

/// Drop one space's grades and rubric vectors — used when a model is switched
/// away or its backend arm flips, since vectors (and therefore cosines) from the
/// other arm are not interchangeable.
pub fn delete_space(conn: &Connection, model: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM stage_scores WHERE model = ?1",
        rusqlite::params![model],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "DELETE FROM rubric_vectors WHERE model = ?1",
        rusqlite::params![model],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Freeze the judge pool: the top `size` candidates by current total, in rank
/// order. Replacing the pool is the *only* thing that changes it — a run never
/// re-picks it mid-flight, which is what keeps a win-rate meaning the same thing
/// at two different moments. `size == 0` is a pool of nobody, which the judge
/// stage reads as "do nothing" (not "no limit").
pub fn freeze_judge_pool(conn: &Connection, job_id: &str, size: usize) -> Result<usize, String> {
    let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM judge_pool WHERE job_id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(|e| e.to_string())?;
    let inserted = if size == 0 {
        0
    } else {
        tx.execute(
            "INSERT INTO judge_pool (job_id, candidate_id, position)
             SELECT job_id, id, ROW_NUMBER() OVER (ORDER BY points DESC, file_name, id) - 1
             FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted'
               AND extracted_text IS NOT NULL AND points IS NOT NULL
             ORDER BY points DESC, file_name, id
             LIMIT ?2",
            rusqlite::params![job_id, size as i64],
        )
        .map_err(|e| e.to_string())?
    };
    tx.commit().map_err(|e| e.to_string())?;
    Ok(inserted)
}

/// The frozen pool's candidate ids, in pool order.
pub fn judge_pool_ids(conn: &Connection, job_id: &str) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .prepare("SELECT candidate_id FROM judge_pool WHERE job_id = ?1 ORDER BY position")
        .map_err(|e| e.to_string())?;
    let ids = stmt
        .query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(ids)
}

/// How big a job's frozen pool is.
pub fn judge_pool_size(conn: &Connection, job_id: &str) -> Result<i64, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM judge_pool WHERE job_id = ?1",
        rusqlite::params![job_id],
        |r| r.get(0),
    )
    .map_err(|e| e.to_string())
}

/// Empty a job's pool — the ad changed, so every cached verdict (whose key
/// includes the ad hash) is stale and the pool has to be chosen again.
pub fn clear_judge_pool(conn: &Connection, job_id: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM judge_pool WHERE job_id = ?1",
        rusqlite::params![job_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Candidates of a job that hold a centroid vector in `space`, with the content
/// hash the judge memoizes on. `ids` restricts the set.
pub fn candidates_with_centroids(
    conn: &Connection,
    job_id: &str,
    model: &str,
    ids: Option<&[String]>,
) -> Result<Vec<(String, Vec<u8>, String)>, String> {
    let (sql, params): (String, Vec<Box<dyn rusqlite::ToSql>>) = match ids {
        Some(list) if !list.is_empty() => {
            let holes = std::iter::repeat_n("?", list.len())
                .collect::<Vec<_>>()
                .join(",");
            let mut p: Vec<Box<dyn rusqlite::ToSql>> = vec![
                Box::new(job_id.to_string()) as Box<dyn rusqlite::ToSql>,
                Box::new(model.to_string()),
            ];
            p.extend(
                list.iter()
                    .map(|i| Box::new(i.clone()) as Box<dyn rusqlite::ToSql>),
            );
            (
                format!(
                    "SELECT id, vector, COALESCE(content_hash, id) FROM candidates
                     WHERE job_id = ?1 AND extraction_status = 'Extracted'
                       AND extracted_text IS NOT NULL
                       AND vector IS NOT NULL AND vector_model = ?2 AND id IN ({holes})"
                ),
                p,
            )
        }
        _ => (
            "SELECT id, vector, COALESCE(content_hash, id) FROM candidates
             WHERE job_id = ?1 AND extraction_status = 'Extracted'
               AND extracted_text IS NOT NULL
               AND vector IS NOT NULL AND vector_model = ?2"
                .to_string(),
            vec![
                Box::new(job_id.to_string()) as Box<dyn rusqlite::ToSql>,
                Box::new(model.to_string()),
            ],
        ),
    };
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(params.iter().map(|b| b.as_ref())), |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Vec<u8>>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok(rows)
}
