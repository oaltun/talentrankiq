use std::sync::Mutex;
use std::path::Path;

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

CREATE TABLE IF NOT EXISTS rank_state (
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

pub struct DbHandle {
    pub read_pool: Mutex<Vec<Connection>>,
    pub writer: Mutex<Connection>,
    pub path: std::path::PathBuf,
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
        let mut w = Connection::open(path)?;
        w.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        w.execute_batch(SCHEMA)?;
        // read pool
        let mut pool = Vec::new();
        for _ in 0..4 {
            let mut c = Connection::open(path)?;
            c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            // read-only-ish: still need schema present (already created)
            pool.push(c);
        }
        Ok(Self {
            read_pool: Mutex::new(pool),
            writer: Mutex::new(w),
            path: path.to_path_buf(),
        })
    }

    pub fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(SCHEMA)?;
        Ok(())
    }

    // ---- writer helpers ----
    pub fn with_writer<F, R>(&self, f: F) -> Result<R, rusqlite::Error>
    where F: FnOnce(&mut Connection) -> Result<R, rusqlite::Error> {
        let mut g = self.writer.lock().unwrap();
        f(&mut g)
    }

    pub fn with_reader<F, R>(&self, f: F) -> Result<R, rusqlite::Error>
    where F: FnOnce(&mut Connection) -> Result<R, rusqlite::Error> {
        let mut pool = self.read_pool.lock().unwrap();
        // borrow first available
        if pool.is_empty() {
            let mut c = Connection::open(&self.path)?;
            c.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
            let r = f(&mut c)?;
            pool.push(c);
            return Ok(r);
        }
        // use last
        let idx = pool.len() - 1;
        let c = &mut pool[idx];
        f(c)
    }

    // ---- queries ----
    pub fn get_all_cv_hashes(&self) -> Result<Vec<String>, rusqlite::Error> {
        self.with_reader(|c| {
            let mut stmt = c.prepare("SELECT hash FROM cv ORDER BY hash")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
    }

    pub fn get_cv_score(&self, cv_hash: &str) -> Result<f64, rusqlite::Error> {
        self.with_reader(|c| {
            let s: f64 = c.query_row("SELECT score FROM cv WHERE hash = ?1", [cv_hash], |r| r.get(0)).unwrap_or(0.0);
            Ok(s)
        })
    }

    pub fn upsert_file_state(&self, path: &str, hash: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute("INSERT OR REPLACE INTO file_state (path, hash) VALUES (?1, ?2)", rusqlite::params![path, hash])?;
            Ok(())
        })
    }

    pub fn file_state_has(&self, path: &str) -> Result<bool, rusqlite::Error> {
        self.with_reader(|c| {
            let n: i64 = c.query_row("SELECT COUNT(*) FROM file_state WHERE path = ?1", [path], |r| r.get(0))?;
            Ok(n > 0)
        })
    }

    pub fn cv_has(&self, hash: &str) -> Result<bool, rusqlite::Error> {
        self.with_reader(|c| {
            let n: i64 = c.query_row("SELECT COUNT(*) FROM cv WHERE hash = ?1", [hash], |r| r.get(0))?;
            Ok(n > 0)
        })
    }

    pub fn insert_cv(&self, hash: &str, text: &str) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute("INSERT OR IGNORE INTO cv (hash, text, score) VALUES (?1, ?2, 0.0)", rusqlite::params![hash, text])?;
            Ok(())
        })
    }

    pub fn get_deep_queue(&self, quick_model_id: &str, deep_model_id: &str, limit: usize) -> Result<Vec<String>, rusqlite::Error> {
        // CVs missing deep embedding, ordered by quick similarity to jobad desc
        // We don't have jobad hash here; use score desc as proxy ordering like pseudocode (top by quick score)
        self.with_reader(|c| {
            let sql = r#"
                SELECT c.hash FROM cv c
                WHERE EXISTS (SELECT 1 FROM cv_similarity s WHERE s.cv_hash=c.hash AND s.model_id=?1)
                  AND NOT EXISTS (SELECT 1 FROM cv_embedding e WHERE e.cv_hash=c.hash AND e.model_id=?2)
                ORDER BY (SELECT COALESCE(MAX(s.similarity),0) FROM cv_similarity s WHERE s.cv_hash=c.hash AND s.model_id=?1) DESC,
                         c.hash
                LIMIT ?3
            "#;
            let mut stmt = c.prepare(sql)?;
            let rows = stmt.query_map(rusqlite::params![quick_model_id, deep_model_id, limit as i64], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
    }

    pub fn get_cvs_for_comparison(&self, deep_model_id: &str, jobad_hash: &str) -> Result<Vec<String>, rusqlite::Error> {
        self.with_reader(|c| {
            let sql = r#"
                SELECT c.hash FROM cv c
                WHERE EXISTS (SELECT 1 FROM cv_embedding e WHERE e.cv_hash=c.hash AND e.model_id=?1)
                  AND EXISTS (SELECT 1 FROM cv_similarity s WHERE s.cv_hash=c.hash AND s.model_id=?1 AND s.target_hash=?2)
                ORDER BY c.score DESC, c.hash
            "#;
            let mut stmt = c.prepare(sql)?;
            let rows = stmt.query_map(rusqlite::params![deep_model_id, jobad_hash], |r| r.get::<_, String>(0))?;
            let mut out = Vec::new();
            for r in rows { out.push(r?); }
            Ok(out)
        })
    }

    pub fn upsert_embedding(&self, cv_hash: &str, model_id: &str, vector: &[u8], chunk_vectors: Option<&[u8]>) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO cv_embedding (cv_hash, model_id, vector, chunk_vectors) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![cv_hash, model_id, vector, chunk_vectors],
            )?;
            Ok(())
        })
    }

    pub fn get_embedding(&self, cv_hash: &str, model_id: &str) -> Result<Option<Vec<u8>>, rusqlite::Error> {
        self.with_reader(|c| {
            let mut stmt = c.prepare("SELECT vector FROM cv_embedding WHERE cv_hash=?1 AND model_id=?2")?;
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
            let n: i64 = c.query_row("SELECT COUNT(*) FROM cv_embedding WHERE cv_hash=?1 AND model_id=?2", rusqlite::params![cv_hash, model_id], |r| r.get(0))?;
            Ok(n > 0)
        })
    }

    pub fn upsert_similarity(&self, cv_hash: &str, model_id: &str, target_hash: &str, sim: f64) -> Result<(), rusqlite::Error> {
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
            let score: f64 = c.query_row(
                "SELECT COALESCE(MAX(similarity), 0.0) FROM cv_similarity WHERE cv_hash = ?1",
                [cv_hash],
                |r| r.get(0),
            ).unwrap_or(0.0);
            c.execute("UPDATE cv SET score = ?1 WHERE hash = ?2", rusqlite::params![score, cv_hash])?;
            Ok(())
        })
    }

    pub fn update_rank_state(&self, job_id: &str, wave: i64, embedded_total: i64, embedded_done: i64, compared: i64, candidates: i64, converged: i64, reason: Option<&str>, running: bool) -> Result<(), rusqlite::Error> {
        self.with_writer(|c| {
            c.execute(
                "INSERT OR REPLACE INTO rank_state (job_id, running, wave, embedded_total, embedded_done, compared, candidates, converged, reason)
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
