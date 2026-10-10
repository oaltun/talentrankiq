//! Judge verdict memoization (state contract: the judge runs only when
//! necessary).
//!
//! The gui judge stage is a pairwise tournament: with N candidates it
//! evaluates N·(N−1)/2 canonical pairs × 2 forward passes each (~4M passes at
//! 2000 CVs). None of that depends on criteria or feedback — only on the job
//! ad text, the CV texts, the embedding model, and the judge artifact. A
//! re-rank that changed none of those must run zero verdicts, and a new
//! candidate must only pay for its own pairs.
//!
//! One row per canonical pair verdict `p(a > b)`, keyed by content hashes
//! (ad text, both candidate texts) plus the embedding model slug and the
//! judge artifact content hash. Identical CV texts across jobs share rows, so
//! the table doubles as a portable verdict store for future consumers.

use rusqlite::{Connection, OptionalExtension};

/// What a `win_rates_cached` pass actually did: verdicts computed fresh
/// (forward passes) vs served from the memoization table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PairVerdictStats {
    pub fresh: u32,
    pub cached: u32,
}

/// Look up the canonical pair verdict `p(doc_a > doc_b)`; `None` on a miss.
/// `doc_a`/`doc_b` must be in canonical (sorted) order.
///
/// When `engine.no_cache` is set (benchmarking) this always misses: every
/// verdict is recomputed while [`store_pair`] keeps the memo warm for
/// cache-enabled runs.
pub fn lookup_pair(
    conn: &Connection,
    model: &str,
    artifact_hash: &str,
    ad_hash: &str,
    doc_a: &str,
    doc_b: &str,
) -> rusqlite::Result<Option<f64>> {
    debug_assert!(doc_a <= doc_b, "canonical (sorted) pair order required");
    if crate::app_config::engine_no_cache() {
        return Ok(None);
    }
    conn.query_row(
        "SELECT prob FROM judge_pair_verdicts
         WHERE model = ?1 AND artifact_hash = ?2 AND ad_hash = ?3 AND doc_a = ?4 AND doc_b = ?5",
        rusqlite::params![model, artifact_hash, ad_hash, doc_a, doc_b],
        |r| r.get(0),
    )
    .optional()
}

/// Store a canonical pair verdict `p(doc_a > doc_b)` (INSERT OR REPLACE —
/// recompute overwrites; cache hits never write).
pub fn store_pair(
    conn: &Connection,
    model: &str,
    artifact_hash: &str,
    ad_hash: &str,
    doc_a: &str,
    doc_b: &str,
    prob: f64,
) -> rusqlite::Result<()> {
    debug_assert!(doc_a <= doc_b, "canonical (sorted) pair order required");
    conn.execute(
        "INSERT OR REPLACE INTO judge_pair_verdicts
         (model, artifact_hash, ad_hash, doc_a, doc_b, prob) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![model, artifact_hash, ad_hash, doc_a, doc_b, prob],
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::db::SCHEMA).unwrap();
        conn
    }

    #[test]
    fn pair_roundtrip_and_miss() {
        let conn = db();
        assert_eq!(
            lookup_pair(&conn, "m", "art", "ad", "a", "b").unwrap(),
            None,
            "empty table misses"
        );
        store_pair(&conn, "m", "art", "ad", "a", "b", 0.75).unwrap();
        let v = lookup_pair(&conn, "m", "art", "ad", "a", "b")
            .unwrap()
            .expect("hit after store");
        assert!((v - 0.75).abs() < 1e-12);
        // any key component change must miss (different verdict)
        assert_eq!(
            lookup_pair(&conn, "other", "art", "ad", "a", "b").unwrap(),
            None
        );
        assert_eq!(
            lookup_pair(&conn, "m", "art2", "ad", "a", "b").unwrap(),
            None
        );
        assert_eq!(
            lookup_pair(&conn, "m", "art", "ad2", "a", "b").unwrap(),
            None
        );
        assert_eq!(
            lookup_pair(&conn, "m", "art", "ad", "a", "c").unwrap(),
            None
        );
        // replace overwrites
        store_pair(&conn, "m", "art", "ad", "a", "b", 0.25).unwrap();
        let v = lookup_pair(&conn, "m", "art", "ad", "a", "b")
            .unwrap()
            .expect("hit");
        assert!((v - 0.25).abs() < 1e-12);
    }
}
