use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS file_state (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cv (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS cv_embedding (
    cv_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    vector BLOB,
    chunk_vectors BLOB,
    PRIMARY KEY(cv_hash, model_id)
);

CREATE TABLE IF NOT EXISTS cv_similarity (
    cv_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY(cv_hash, model_id, target_hash)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT,
    job_ad_text TEXT,
    import_directory_path TEXT
);

CREATE TABLE IF NOT EXISTS criteria (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    text TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS job_cv (
    job_id TEXT NOT NULL,
    cv_hash TEXT NOT NULL,
    PRIMARY KEY (job_id, cv_hash)
);

CREATE TABLE IF NOT EXISTS cv_comparison (
    job_id TEXT NOT NULL,
    a_hash TEXT NOT NULL,
    b_hash TEXT NOT NULL,
    prob REAL NOT NULL,
    PRIMARY KEY (job_id, a_hash, b_hash)
);

CREATE TABLE IF NOT EXISTS waves_rank_state (
    job_id TEXT PRIMARY KEY,
    running INTEGER NOT NULL DEFAULT 0,
    wave INTEGER NOT NULL DEFAULT 0,
    embedded_total INTEGER NOT NULL DEFAULT 0,
    embedded_done INTEGER NOT NULL DEFAULT 0,
    compared INTEGER NOT NULL DEFAULT 0,
    candidates INTEGER NOT NULL DEFAULT 0,
    converged INTEGER NOT NULL DEFAULT 0,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_cv_similarity_model ON cv_similarity(model_id, target_hash);
CREATE INDEX IF NOT EXISTS idx_cv_embedding_model ON cv_embedding(model_id);
"#;

#[derive(Clone)]
pub struct DbHandle {
    read_pool: Arc<Mutex<Vec<Connection>>>,
    writer: Arc<Mutex<Connection>>,
    path: std::path::PathBuf,
}

impl DbHandle {
    pub fn open_connection_at(path: &Path) -> Result<Self, rusqlite::Error> {
        Self::open_at(path)
    }
    pub fn open_at(path: &Path) -> Result<Self, rusqlite::Error> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // writer
        let w = Connection::open(path)?;
        w.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        w.execute_batch(SCHEMA)?;
        // read pool
        let mut pool = Vec::new();
        for _ in 0..4 {
            let c = Connection::open(path)?;
            c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            pool.push(c);
        }
        Ok(Self {
            read_pool: Arc::new(Mutex::new(pool)),
            writer: Arc::new(Mutex::new(w)),
            path: path.to_path_buf(),
        })
    }

    pub fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(SCHEMA)?;
        Ok(())
    }

    // ---- writer helpers ----
    pub fn with_writer<F, R>(&self, f: F) -> Result<R, rusqlite::Error>
    where
        F: FnOnce(&mut Connection) -> Result<R, rusqlite::Error>,
    {
        let mut g = self.writer.lock().unwrap();
        f(&mut g)
    }

    pub fn with_reader<F, R>(&self, f: F) -> Result<R, rusqlite::Error>
    where
        F: FnOnce(&mut Connection) -> Result<R, rusqlite::Error>,
    {
        let mut pool = self.read_pool.lock().unwrap();
        if pool.is_empty() {
            let mut c = Connection::open(&self.path)?;
            c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            let r = f(&mut c)?;
            pool.push(c);
            return Ok(r);
        }
        let idx = pool.len() - 1;
        let c = &mut pool[idx];
        f(c)
    }

    // ---- queries ----
    /// All CV hashes belonging to one job (via the job-scoped membership table).
    pub fn get_all_cv_hashes_for_job(&self, job_id: &str) -> Result<Vec<String>, rusqlite::Error> {
        self.with_reader(|c| {
            let mut stmt =
                c.prepare("SELECT cv_hash FROM job_cv WHERE job_id = ?1 ORDER BY cv_hash")?;
            let rows = stmt.query_map(rusqlite::params![job_id], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
    }

    /// How many CVs the job currently has on disk (membership rows).
    pub fn get_job_cv_count(&self, job_id: &str) -> Result<i64, rusqlite::Error> {
        self.with_reader(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM job_cv WHERE job_id = ?1",
                [job_id],
                |r| r.get::<_, i64>(0),
            )
        })
    }

    /// How many of the job's CVs hold an embedding for `model_id`.
    pub fn get_job_cv_embedded_count(
        &self,
        job_id: &str,
        model_id: &str,
    ) -> Result<i64, rusqlite::Error> {
        self.with_reader(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM job_cv j
                 JOIN cv_embedding e ON e.cv_hash = j.cv_hash AND e.model_id = ?2
                 WHERE j.job_id = ?1",
                rusqlite::params![job_id, model_id],
                |r| r.get::<_, i64>(0),
            )
        })
    }

    pub fn get_cv_score(&self, cv_hash: &str) -> Result<f64, rusqlite::Error> {
        self.with_reader(|c| {
            let s: f64 = c
                .query_row("SELECT score FROM cv WHERE hash = ?1", [cv_hash], |r| {
                    r.get(0)
                })
                .unwrap_or(0.0);
            Ok(s)
        })
    }

    pub fn upsert_file_state(&self, path: &str, hash: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO file_state (path, hash) VALUES (?1, ?2)",
                rusqlite::params![path, hash],
            )?;
            Ok(())
        })
    }

    pub fn file_state_has(&self, path: &str) -> Result<bool, rusqlite::Error> {
        self.with_reader(|c| {
            let n: i64 = c.query_row(
                "SELECT COUNT(*) FROM file_state WHERE path = ?1",
                [path],
                |r| r.get(0),
            )?;
            Ok(n > 0)
        })
    }

    /// Drop all membership rows for a job (watch-dir reconciliation).
    pub fn clear_job_cv(&self, job_id: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute("DELETE FROM job_cv WHERE job_id = ?1", [job_id])?;
            Ok(())
        })
    }

    /// Record a job→cv membership (content-hash dedupe across jobs is fine).
    pub fn insert_job_cv(&self, job_id: &str, cv_hash: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR IGNORE INTO job_cv (job_id, cv_hash) VALUES (?1, ?2)",
                rusqlite::params![job_id, cv_hash],
            )?;
            Ok(())
        })
    }

    pub fn cv_has(&self, hash: &str) -> Result<bool, rusqlite::Error> {
        self.with_reader(|c| {
            let n: i64 = c.query_row("SELECT COUNT(*) FROM cv WHERE hash = ?1", [hash], |r| {
                r.get(0)
            })?;
            Ok(n > 0)
        })
    }

    pub fn insert_cv(&self, hash: &str, text: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR IGNORE INTO cv (hash, text, score) VALUES (?1, ?2, 0.0)",
                rusqlite::params![hash, text],
            )?;
            Ok(())
        })
    }

    pub fn get_deep_queue_for_job(
        &self,
        job_id: &str,
        quick_model_id: &str,
        deep_model_id: &str,
        limit: usize,
    ) -> Result<Vec<String>, rusqlite::Error> {
        // The job's CVs missing a deep embedding, ordered by quick similarity to the job ad desc.
        self.with_reader(|c| {
            let sql = r#"
                SELECT c.hash FROM cv c
                JOIN job_cv j ON j.cv_hash = c.hash AND j.job_id = ?1
                WHERE NOT EXISTS (SELECT 1 FROM cv_embedding e WHERE e.cv_hash=c.hash AND e.model_id=?3)
                ORDER BY (SELECT COALESCE(MAX(s.similarity),0) FROM cv_similarity s WHERE s.cv_hash=c.hash AND s.model_id=?2) DESC,
                         c.hash
                LIMIT ?4
            "#;
            let mut stmt = c.prepare(sql)?;
            let rows = stmt.query_map(rusqlite::params![job_id, quick_model_id, deep_model_id, limit as i64], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
    }

    pub fn get_cvs_for_comparison_for_job(
        &self,
        job_id: &str,
        deep_model_id: &str,
        jobad_hash: &str,
    ) -> Result<Vec<String>, rusqlite::Error> {
        self.with_reader(|c| {
            let sql = r#"
                SELECT c.hash FROM cv c
                JOIN job_cv j ON j.cv_hash = c.hash AND j.job_id = ?1
                WHERE EXISTS (SELECT 1 FROM cv_embedding e WHERE e.cv_hash=c.hash AND e.model_id=?2)
                  AND EXISTS (SELECT 1 FROM cv_similarity s WHERE s.cv_hash=c.hash AND s.model_id=?2 AND s.target_hash=?3)
                ORDER BY c.score DESC, c.hash
            "#;
            let mut stmt = c.prepare(sql)?;
            let rows = stmt.query_map(rusqlite::params![job_id, deep_model_id, jobad_hash], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
    }

    pub fn insert_comparison(
        &self,
        job_id: &str,
        a_hash: &str,
        b_hash: &str,
        prob: f64,
    ) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO cv_comparison (job_id, a_hash, b_hash, prob) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![job_id, a_hash, b_hash, prob],
            )?;
            Ok(())
        })
    }

    pub fn get_comparison_count(&self, job_id: &str) -> Result<i64, rusqlite::Error> {
        self.with_reader(|c| {
            c.query_row(
                "SELECT COUNT(*) FROM cv_comparison WHERE job_id = ?1",
                [job_id],
                |r| r.get::<_, i64>(0),
            )
        })
    }

    pub fn upsert_embedding(
        &self,
        cv_hash: &str,
        model_id: &str,
        vector: &[u8],
        chunk_vectors: Option<&[u8]>,
    ) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO cv_embedding (cv_hash, model_id, vector, chunk_vectors) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![cv_hash, model_id, vector, chunk_vectors],
            )?;
            Ok(())
        })
    }

    pub fn get_embedding(
        &self,
        cv_hash: &str,
        model_id: &str,
    ) -> Result<Option<Vec<u8>>, rusqlite::Error> {
        self.with_reader(|c| {
            let mut stmt =
                c.prepare("SELECT vector FROM cv_embedding WHERE cv_hash=?1 AND model_id=?2")?;
            let mut rows = stmt.query(rusqlite::params![cv_hash, model_id])?;
            if let Some(row) = rows.next()? {
                let v: Vec<u8> = row.get(0)?;
                Ok(Some(v))
            } else {
                Ok(None)
            }
        })
    }

    pub fn has_embedding(&self, cv_hash: &str, model_id: &str) -> Result<bool, rusqlite::Error> {
        self.with_reader(|c| {
            let n: i64 = c.query_row(
                "SELECT COUNT(*) FROM cv_embedding WHERE cv_hash=?1 AND model_id=?2",
                rusqlite::params![cv_hash, model_id],
                |r| r.get(0),
            )?;
            Ok(n > 0)
        })
    }

    pub fn upsert_similarity(
        &self,
        cv_hash: &str,
        model_id: &str,
        target_hash: &str,
        sim: f64,
    ) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO cv_similarity (cv_hash, model_id, target_hash, similarity) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![cv_hash, model_id, target_hash, sim],
            )?;
            Ok(())
        })
    }

    pub fn rescore_cv(&self, cv_hash: &str) -> Result<(), rusqlite::Error> {
        // recompute cv.score = max similarity to jobad across models? Use deep model similarity if present else quick.
        self.with_writer(|c| {
            let score: f64 = c
                .query_row(
                    "SELECT COALESCE(MAX(similarity), 0.0) FROM cv_similarity WHERE cv_hash = ?1",
                    [cv_hash],
                    |r| r.get(0),
                )
                .unwrap_or(0.0);
            c.execute(
                "UPDATE cv SET score = ?1 WHERE hash = ?2",
                rusqlite::params![score, cv_hash],
            )?;
            Ok(())
        })
    }

    pub fn update_waves_rank_state(
        &self,
        job_id: &str,
        wave: i64,
        embedded_total: i64,
        embedded_done: i64,
        compared: i64,
        candidates: i64,
        converged: i64,
        reason: Option<&str>,
        running: bool,
    ) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO waves_rank_state (job_id, running, wave, embedded_total, embedded_done, compared, candidates, converged, reason)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![job_id, running as i64, wave, embedded_total, embedded_done, compared, candidates, converged, reason],
            )?;
            Ok(())
        })
    }
}

pub fn open_connection_at(path: &std::path::Path) -> Result<DbHandle, rusqlite::Error> {
    DbHandle::open_at(path)
}

pub fn init_schema(conn: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
    DbHandle::init_schema(conn)
}
