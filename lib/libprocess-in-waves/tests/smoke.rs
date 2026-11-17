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
    assert_eq!(db.get_all_cv_hashes().unwrap(), vec!["h1".to_string()]);
}
