use std::collections::HashMap;

use anima_embed::SparseVector;
use rusqlite::{params, Connection};

/// Insert a sparse vector for a memory, storing both the blob and inverted postings.
/// If the sparse vector is empty, this is a no-op.
pub fn insert_sparse(
    conn: &Connection,
    memory_id: &str,
    namespace: &str,
    sparse: &SparseVector,
) -> rusqlite::Result<()> {
    if sparse.is_empty() {
        return Ok(());
    }

    let blob = sparse.to_bytes();
    conn.execute(
        "INSERT INTO sparse_vectors (memory_id, vector) VALUES (?1, ?2)",
        params![memory_id, blob],
    )?;

    let mut stmt = conn.prepare(
        "INSERT INTO sparse_postings (namespace, token_id, memory_id, weight)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for &(token_id, weight) in &sparse.0 {
        stmt.execute(params![namespace, token_id as i64, memory_id, weight as f64])?;
    }

    Ok(())
}

/// Delete sparse data for a memory from both tables.
pub fn delete_sparse(conn: &Connection, memory_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM sparse_postings WHERE memory_id = ?1",
        params![memory_id],
    )?;
    conn.execute(
        "DELETE FROM sparse_vectors WHERE memory_id = ?1",
        params![memory_id],
    )?;
    Ok(())
}

/// Update sparse data: delete then re-insert.
pub fn update_sparse(
    conn: &Connection,
    memory_id: &str,
    namespace: &str,
    sparse: &SparseVector,
) -> rusqlite::Result<()> {
    delete_sparse(conn, memory_id)?;
    insert_sparse(conn, memory_id, namespace, sparse)
}

/// Search sparse postings within a namespace, returning (memory_id, score) pairs
/// sorted by descending dot-product score.
pub fn search_sparse(
    conn: &Connection,
    query_sparse: &SparseVector,
    namespace: &str,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    if query_sparse.is_empty() {
        return Ok(Vec::new());
    }

    let query_weights: HashMap<u32, f32> = query_sparse.0.iter().copied().collect();
    let token_ids: Vec<i64> = query_weights.keys().map(|&k| k as i64).collect();

    // Build parameterized IN clause: ?1 = namespace, ?2..?N = token_ids
    let placeholders: Vec<String> = (2..=token_ids.len() + 1)
        .map(|i| format!("?{i}"))
        .collect();
    let sql = format!(
        "SELECT memory_id, token_id, weight FROM sparse_postings
         WHERE namespace = ?1 AND token_id IN ({})
        ",
        placeholders.join(", ")
    );

    let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    sql_params.push(Box::new(namespace.to_string()));
    for tid in &token_ids {
        sql_params.push(Box::new(*tid));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        sql_params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut rows = stmt.query(param_refs.as_slice())?;
    while let Some(row) = rows.next()? {
        let mem_id: String = row.get(0)?;
        let token_id: i64 = row.get(1)?;
        let doc_weight: f64 = row.get(2)?;
        if let Some(&qw) = query_weights.get(&(token_id as u32)) {
            *scores.entry(mem_id).or_insert(0.0) += qw as f64 * doc_weight;
        }
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Search sparse postings globally (all namespaces), returning (memory_id, score) pairs
/// sorted by descending dot-product score.
pub fn search_sparse_global(
    conn: &Connection,
    query_sparse: &SparseVector,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    if query_sparse.is_empty() {
        return Ok(Vec::new());
    }

    let query_weights: HashMap<u32, f32> = query_sparse.0.iter().copied().collect();
    let token_ids: Vec<i64> = query_weights.keys().map(|&k| k as i64).collect();

    let placeholders: Vec<String> = (1..=token_ids.len()).map(|i| format!("?{i}")).collect();
    let sql = format!(
        "SELECT memory_id, token_id, weight FROM sparse_postings
         WHERE token_id IN ({})
        ",
        placeholders.join(", ")
    );

    let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for tid in &token_ids {
        sql_params.push(Box::new(*tid));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        sql_params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut rows = stmt.query(param_refs.as_slice())?;
    while let Some(row) = rows.next()? {
        let mem_id: String = row.get(0)?;
        let token_id: i64 = row.get(1)?;
        let doc_weight: f64 = row.get(2)?;
        if let Some(&qw) = query_weights.get(&(token_id as u32)) {
            *scores.entry(mem_id).or_insert(0.0) += qw as f64 * doc_weight;
        }
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Count of rows in sparse_vectors table.
pub fn sparse_count(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row("SELECT count(*) FROM sparse_vectors", [], |row| row.get(0))
}

/// Delete all data from both sparse tables.
pub fn force_rebuild_sparse(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute("DELETE FROM sparse_postings", [])?;
    conn.execute("DELETE FROM sparse_vectors", [])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;

    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        schema::initialize(&conn).unwrap();
        conn
    }

    /// Insert a minimal row into memories so FK constraints are satisfied.
    fn insert_test_memory(conn: &Connection, id: &str, namespace: &str) {
        conn.execute(
            "INSERT INTO memories (id, namespace, content, metadata, embedding, status, created_at, updated_at, accessed_at, access_count, hash)
             VALUES (?1, ?2, 'test', NULL, X'00', 'active', '2025-01-01', '2025-01-01', '2025-01-01', 0, 'h')",
            params![id, namespace],
        )
        .unwrap();
    }

    #[test]
    fn test_sparse_insert_search() {
        let conn = setup_db();
        insert_test_memory(&conn, "m1", "ns1");

        let sv = SparseVector(vec![(10, 0.5), (20, 1.0), (30, 0.3)]);
        insert_sparse(&conn, "m1", "ns1", &sv).unwrap();

        assert_eq!(sparse_count(&conn).unwrap(), 1);

        // Search with overlapping tokens: query has tokens 10 and 20
        let query = SparseVector(vec![(10, 1.0), (20, 2.0)]);
        let results = search_sparse(&conn, &query, "ns1", 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "m1");
        // Expected score: 1.0*0.5 + 2.0*1.0 = 2.5
        assert!((results[0].1 - 2.5).abs() < 1e-6);

        // Delete and verify gone
        delete_sparse(&conn, "m1").unwrap();
        assert_eq!(sparse_count(&conn).unwrap(), 0);

        let results = search_sparse(&conn, &query, "ns1", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_namespace_isolation() {
        let conn = setup_db();
        insert_test_memory(&conn, "m1", "ns_a");
        insert_test_memory(&conn, "m2", "ns_b");

        let sv1 = SparseVector(vec![(100, 1.0)]);
        let sv2 = SparseVector(vec![(100, 2.0)]);
        insert_sparse(&conn, "m1", "ns_a", &sv1).unwrap();
        insert_sparse(&conn, "m2", "ns_b", &sv2).unwrap();

        let query = SparseVector(vec![(100, 1.0)]);

        // Namespace-scoped search returns only matching namespace
        let results_a = search_sparse(&conn, &query, "ns_a", 10).unwrap();
        assert_eq!(results_a.len(), 1);
        assert_eq!(results_a[0].0, "m1");

        let results_b = search_sparse(&conn, &query, "ns_b", 10).unwrap();
        assert_eq!(results_b.len(), 1);
        assert_eq!(results_b[0].0, "m2");

        // Global search returns both
        let results_global = search_sparse_global(&conn, &query, 10).unwrap();
        assert_eq!(results_global.len(), 2);
        // m2 should come first (higher score: 2.0 vs 1.0)
        assert_eq!(results_global[0].0, "m2");
        assert_eq!(results_global[1].0, "m1");
    }

    #[test]
    fn test_sparse_empty_vector() {
        let conn = setup_db();
        insert_test_memory(&conn, "m1", "ns1");

        let empty = SparseVector(vec![]);
        insert_sparse(&conn, "m1", "ns1", &empty).unwrap();

        // Empty vector is a no-op: nothing inserted
        assert_eq!(sparse_count(&conn).unwrap(), 0);
    }
}
