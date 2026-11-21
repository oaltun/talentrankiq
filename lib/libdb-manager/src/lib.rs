//! Unified SQLite connection pool for TalentRankIQ.
//!
//! One [`Db`] instance owns a single writer (`Mutex<Connection>`) and a
//! round-robin read pool (`Vec<Mutex<Connection>>`). All database access
//! across the CLI, waves pipeline, and core crates goes through
//! [`Db::read`] / [`Db::write`] closures, so long-running writes never
//! block concurrent reads (WAL mode).

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

/// Number of read-only connections in the pool.
const READ_POOL_SIZE: usize = 4;

/// Round-robin read pool of query-only connections.
pub struct ReadPool {
    conns: Vec<Mutex<Connection>>,
    next: AtomicUsize,
}

impl ReadPool {
    fn new(conns: Vec<Mutex<Connection>>) -> Self {
        Self {
            conns,
            next: AtomicUsize::new(0),
        }
    }

    fn acquire(&self) -> &Mutex<Connection> {
        let n = self.conns.len();
        let i = self.next.fetch_add(1, Ordering::Relaxed) % n;
        &self.conns[i]
    }
}

/// Unified database handle: one writer + a read pool, shared via `Arc<Db>`.
pub struct Db {
    writer: Mutex<Connection>,
    read_pool: ReadPool,
}

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("{0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Other(String),
}

impl Db {
    /// Open (or create) the database at `path` and return a shared handle.
    ///
    /// The writer connection gets `PRAGMA journal_mode=WAL` and
    /// `PRAGMA busy_timeout=5000`. Read-pool connections are opened
    /// read/write but restricted with `PRAGMA query_only` so they can
    /// participate in WAL recovery while rejecting writes.
    pub fn open_at(path: &Path) -> Result<Arc<Db>, DbError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| DbError::Other(format!("Failed to create database directory: {e}")))?;
        }

        let writer = Connection::open(path)?;
        writer.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;

        let mut read_conns = Vec::with_capacity(READ_POOL_SIZE);
        for _ in 0..READ_POOL_SIZE {
            let conn = Connection::open(path)?;
            conn.execute_batch("PRAGMA query_only=ON;")?;
            read_conns.push(Mutex::new(conn));
        }

        Ok(Arc::new(Db {
            writer: Mutex::new(writer),
            read_pool: ReadPool::new(read_conns),
        }))
    }

    /// Run a read closure on the next pooled read-only connection (round-robin).
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Connection) -> R,
    {
        let cell = self.read_pool.acquire();
        let guard = cell.lock().expect("read pool mutex poisoned");
        f(&guard)
    }

    /// Run a write closure on the single writer connection (serialized).
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Connection) -> R,
    {
        let mut guard = self.writer.lock().expect("writer mutex poisoned");
        f(&mut guard)
    }

    /// Acquire a guard on the next pooled read-only connection (round-robin).
    ///
    /// Prefer [`Db::read`] for new code; this exists for handlers whose query
    /// body spans many statements and would be awkward to wrap in a closure.
    pub fn read_guard(&self) -> std::sync::MutexGuard<'_, Connection> {
        let cell = self.read_pool.acquire();
        cell.lock().expect("read pool mutex poisoned")
    }

    /// Acquire the single writer guard. Prefer [`Db::write`] for new code.
    pub fn write_guard(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.writer.lock().expect("writer mutex poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_read_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Db::open_at(&path).unwrap();

        db.write(|conn| {
            conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
                .unwrap();
            conn.execute("INSERT INTO t (val) VALUES ('hello')", [])
                .unwrap();
        });

        let val: String = db.read(|conn| {
            conn.query_row("SELECT val FROM t WHERE id = 1", [], |r| r.get(0))
                .unwrap()
        });
        assert_eq!(val, "hello");
    }

    #[test]
    fn concurrent_reads_during_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Db::open_at(&path).unwrap();

        db.write(|conn| {
            conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY);")
                .unwrap();
        });

        let db2 = Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            for _ in 0..100 {
                let _: i64 = db2.read(|conn| {
                    conn.query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
                        .unwrap()
                });
            }
        });

        for i in 0..100 {
            db.write(|conn| {
                conn.execute("INSERT INTO t (id) VALUES (?1)", [i]).unwrap();
            });
        }

        handle.join().unwrap();
        let count: i64 = db.read(|conn| {
            conn.query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
                .unwrap()
        });
        assert_eq!(count, 100);
    }
}
