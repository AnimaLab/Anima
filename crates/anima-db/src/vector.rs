use rusqlite::Connection;

/// Initialize the vec0 virtual table for vector search.
/// sqlite-vec must be loaded as an extension before calling this.
/// If the table exists with a different schema, it will be recreated and
/// automatically repopulated from the memories table (matching dimension only).
pub fn initialize_vec_table(conn: &Connection, dimension: usize) -> rusqlite::Result<()> {
    let table_exists: bool = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_memories'")?
        .exists([])?;

    let mut needs_rebuild = false;

    if table_exists {
        // Check if the table has the namespace metadata column (v2 schema).
        // If not, drop and recreate to add it.
        let has_namespace = conn
            .prepare("SELECT 1 FROM vec_memories WHERE namespace = '__schema_check__' LIMIT 0")
            .is_ok();

        if !has_namespace {
            tracing::info!("Upgrading vec_memories to v2 schema (adding namespace metadata column)");
            conn.execute_batch("DROP TABLE vec_memories;")?;
            needs_rebuild = true;
        } else {
            // Verify dimension compatibility
            let test_blob: Vec<u8> = vec![0u8; dimension * 4];
            let dim_ok = conn
                .execute(
                    "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES ('__dim_check__', ?1, '__test__')",
                    rusqlite::params![test_blob],
                )
                .is_ok();

            if dim_ok {
                conn.execute(
                    "DELETE FROM vec_memories WHERE memory_id = '__dim_check__'",
                    [],
                )?;
            } else {
                tracing::warn!(
                    "Vec table dimension mismatch, recreating with dimension={dimension}."
                );
                conn.execute_batch("DROP TABLE vec_memories;")?;
                needs_rebuild = true;
            }
        }
    } else {
        needs_rebuild = true;
    }

    // v2 schema: namespace as a metadata column for filtered KNN search.
    // sqlite-vec metadata columns support = and IN operators during MATCH queries.
    let sql = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding float[{dimension}],
            namespace TEXT
        );"
    );
    conn.execute_batch(&sql)?;

    if needs_rebuild {
        rebuild_vec_index(conn, dimension)?;
    }

    Ok(())
}

/// Repopulate vec_memories from the memories table.
/// Only includes active memories whose embedding size matches the expected dimension.
fn rebuild_vec_index(conn: &Connection, dimension: usize) -> rusqlite::Result<()> {
    let expected_bytes = dimension * 4;
    let mut stmt = conn.prepare(
        "SELECT id, embedding, namespace FROM memories WHERE status = 'active' AND length(embedding) = ?1",
    )?;
    let mut count = 0u64;
    let mut rows = stmt.query(rusqlite::params![expected_bytes as i64])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        let namespace: String = row.get(2)?;
        if let Err(e) = conn.execute(
            "INSERT OR IGNORE INTO vec_memories(memory_id, embedding, namespace) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, blob, namespace],
        ) {
            tracing::warn!("Failed to rebuild vec for {}: {}", id, e);
        } else {
            count += 1;
        }
    }
    tracing::info!("Rebuilt vec_memories index: {count} vectors inserted (dim={dimension})");
    Ok(())
}

/// Insert a vector embedding into vec_memories.
pub fn insert_embedding(
    conn: &Connection,
    memory_id: &str,
    embedding: &[f32],
    namespace: &str,
) -> rusqlite::Result<()> {
    let blob = embedding_to_blob(embedding);
    conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES (?1, ?2, ?3)",
        rusqlite::params![memory_id, blob, namespace],
    )?;
    Ok(())
}

/// Delete a vector embedding from vec_memories.
pub fn delete_embedding(conn: &Connection, memory_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM vec_memories WHERE memory_id = ?1",
        rusqlite::params![memory_id],
    )?;
    Ok(())
}

/// Search for nearest neighbors by vector similarity, filtered by namespace.
/// The namespace filter uses exact match (=) on the vec0 metadata column,
/// which sqlite-vec evaluates during ANN search (pre-filtering).
/// Returns (memory_id, distance) pairs ordered by distance ascending.
pub fn search_vectors_filtered(
    conn: &Connection,
    query_embedding: &[f32],
    namespace: &str,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = embedding_to_blob(query_embedding);
    let mut stmt = conn.prepare(
        "SELECT memory_id, distance
         FROM vec_memories
         WHERE embedding MATCH ?1
           AND k = ?2
           AND namespace = ?3
         ORDER BY distance",
    )?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64, namespace], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Search for nearest neighbors without namespace filter (global search).
/// Used when the namespace is hierarchical and requires LIKE prefix matching,
/// which sqlite-vec metadata columns don't support.
pub fn search_vectors(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = embedding_to_blob(query_embedding);
    let mut stmt = conn.prepare(
        "SELECT memory_id, distance
         FROM vec_memories
         WHERE embedding MATCH ?1
           AND k = ?2
         ORDER BY distance",
    )?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Update a vector embedding (delete + re-insert).
pub fn update_embedding(
    conn: &Connection,
    memory_id: &str,
    embedding: &[f32],
    namespace: &str,
) -> rusqlite::Result<()> {
    delete_embedding(conn, memory_id)?;
    insert_embedding(conn, memory_id, embedding, namespace)?;
    Ok(())
}

/// Convert f32 slice to raw byte blob for sqlite-vec.
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::DbPool;

    #[test]
    fn test_vec0_crud() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let w = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;

            let embedding: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            insert_embedding(&w, "test-id-1", &embedding, "default").unwrap();

            // Global search should find it
            let results = search_vectors(&w, &embedding, 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "test-id-1");

            // Filtered search should find it in the right namespace
            let results = search_vectors_filtered(&w, &embedding, "default", 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "test-id-1");

            // Filtered search should NOT find it in wrong namespace
            let results = search_vectors_filtered(&w, &embedding, "other", 5).unwrap();
            assert_eq!(results.len(), 0);

            // Delete
            delete_embedding(&w, "test-id-1").unwrap();
            let results = search_vectors(&w, &embedding, 5).unwrap();
            assert_eq!(results.len(), 0);

            // Update
            insert_embedding(&w, "test-id-2", &embedding, "ns1").unwrap();
            let new_embedding: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 / dim as f32).collect();
            update_embedding(&w, "test-id-2", &new_embedding, "ns1").unwrap();

            let results = search_vectors_filtered(&w, &new_embedding, "ns1", 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "test-id-2");
        });
    }

    #[test]
    fn test_vec0_namespace_isolation() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let w = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;

            let emb1: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let emb2: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 / dim as f32).collect();

            insert_embedding(&w, "mem-a", &emb1, "alpha").unwrap();
            insert_embedding(&w, "mem-b", &emb2, "beta").unwrap();

            // Global search returns both
            let all = search_vectors(&w, &emb1, 10).unwrap();
            assert_eq!(all.len(), 2);

            // Filtered search returns only the matching namespace
            let alpha = search_vectors_filtered(&w, &emb1, "alpha", 10).unwrap();
            assert_eq!(alpha.len(), 1);
            assert_eq!(alpha[0].0, "mem-a");

            let beta = search_vectors_filtered(&w, &emb1, "beta", 10).unwrap();
            assert_eq!(beta.len(), 1);
            assert_eq!(beta[0].0, "mem-b");
        });
    }

    #[test]
    fn test_vec0_delete_in_transaction() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let w = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;

            let embedding: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            insert_embedding(&w, "tx-test-1", &embedding, "default").unwrap();

            let tx = w.unchecked_transaction().unwrap();
            let result = delete_embedding(&tx, "tx-test-1");
            tx.commit().unwrap();

            assert!(result.is_ok(), "Delete in tx failed: {:?}", result.err());
        });
    }
}
