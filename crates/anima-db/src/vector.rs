use rusqlite::Connection;

/// Result of initializing the vec table.
#[derive(Debug, Clone, PartialEq)]
pub enum VecTableStatus {
    /// Table is ready with matching dimension.
    Ready,
    /// Table was created fresh (new database).
    Created,
    /// Schema was upgraded (e.g. namespace column added) and rebuilt from existing embeddings.
    Upgraded,
    /// Vec table was rebuilt with int8 quantization (previously float).
    QuantizationUpgraded,
    /// Dimension mismatch detected. Vec index is stale — memories need re-embedding.
    /// The old table is kept intact so existing search still works (at old dimension).
    DimensionMismatch { existing: usize, requested: usize },
}

/// Initialize the vec0 virtual table for vector search.
/// sqlite-vec must be loaded as an extension before calling this.
///
/// On dimension mismatch, does NOT auto-drop the table. Returns
/// `DimensionMismatch` so the server can surface it in the UI
/// and let the user trigger re-indexing explicitly.
pub fn initialize_vec_table(conn: &Connection, dimension: usize) -> rusqlite::Result<VecTableStatus> {
    let table_exists: bool = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_memories'")?
        .exists([])?;

    if !table_exists {
        // Fresh database — create the table.
        let sql = format!(
            "CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding int8[{dimension}],
                namespace TEXT
            );"
        );
        conn.execute_batch(&sql)?;
        return Ok(VecTableStatus::Created);
    }

    // Check if the table has the namespace metadata column (v2 schema).
    let has_namespace = conn
        .prepare("SELECT 1 FROM vec_memories WHERE namespace = '__schema_check__' LIMIT 0")
        .is_ok();

    if !has_namespace {
        // v1 → v2 schema upgrade: add namespace column. Must drop and recreate.
        tracing::info!("Upgrading vec_memories to v2 schema (adding namespace metadata column)");
        conn.execute_batch("DROP TABLE vec_memories;")?;
        let sql = format!(
            "CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding int8[{dimension}],
                namespace TEXT
            );"
        );
        conn.execute_batch(&sql)?;
        rebuild_vec_index(conn, dimension)?;
        return Ok(VecTableStatus::Upgraded);
    }

    // Verify dimension compatibility.
    let test_blob: Vec<u8> = vec![0u8; dimension];
    let dim_ok = conn
        .execute(
            "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES ('__dim_check__', vec_int8(?1), '__test__')",
            rusqlite::params![test_blob],
        )
        .is_ok();

    if dim_ok {
        conn.execute(
            "DELETE FROM vec_memories WHERE memory_id = '__dim_check__'",
            [],
        )?;
        return Ok(VecTableStatus::Ready);
    }

    // int8 probe failed — check if the table is still float-format (pre-quantization).
    let float_probe: Vec<u8> = vec![0u8; dimension * 4];
    let is_float_format = conn
        .execute(
            "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES ('__dim_check__', ?1, '__test__')",
            rusqlite::params![float_probe],
        )
        .is_ok();

    if is_float_format {
        // Clean up probe row
        conn.execute("DELETE FROM vec_memories WHERE memory_id = '__dim_check__'", [])?;
        // Upgrade: drop float table, recreate as int8, rebuild from f32 source
        tracing::info!("Upgrading vec_memories from float to int8 quantization");
        conn.execute_batch("DROP TABLE vec_memories;")?;
        let sql = format!(
            "CREATE VIRTUAL TABLE vec_memories USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding int8[{dimension}],
                namespace TEXT
            );"
        );
        conn.execute_batch(&sql)?;
        rebuild_vec_index(conn, dimension)?;
        return Ok(VecTableStatus::QuantizationUpgraded);
    }

    // Dimension mismatch — figure out the existing dimension.
    let existing_dim = detect_existing_dimension(conn).unwrap_or(0);
    tracing::warn!(
        "Vec index dimension mismatch: index has {existing_dim}d, config requests {dimension}d. \
         Search will use the existing index until re-indexed from Settings."
    );
    Ok(VecTableStatus::DimensionMismatch {
        existing: existing_dim,
        requested: dimension,
    })
}

/// Detect the dimension of the existing vec_memories table by reading one embedding.
fn detect_existing_dimension(conn: &Connection) -> Option<usize> {
    conn.prepare("SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1")
        .ok()?
        .query_row([], |row| row.get::<_, usize>(0))
        .ok()
        .map(|bytes| bytes / 4) // f32 = 4 bytes
}

/// Force re-index: drop and recreate the vec table with the given dimension,
/// then repopulate from memories table. Call this from the reindex API endpoint.
pub fn force_reindex(conn: &Connection, dimension: usize) -> rusqlite::Result<usize> {
    conn.execute_batch("DROP TABLE IF EXISTS vec_memories;")?;
    let sql = format!(
        "CREATE VIRTUAL TABLE vec_memories USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding int8[{dimension}],
            namespace TEXT
        );"
    );
    conn.execute_batch(&sql)?;
    rebuild_vec_index(conn, dimension)
}

/// Repopulate vec_memories from the memories table.
/// Only includes active memories whose embedding size matches the expected dimension.
fn rebuild_vec_index(conn: &Connection, dimension: usize) -> rusqlite::Result<usize> {
    let expected_bytes = dimension * 4; // memories stores f32
    let mut stmt = conn.prepare(
        "SELECT id, embedding, namespace FROM memories WHERE status = 'active' AND length(embedding) = ?1",
    )?;
    let mut count = 0usize;
    let mut rows = stmt.query(rusqlite::params![expected_bytes as i64])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let f32_blob: Vec<u8> = row.get(1)?;
        let namespace: String = row.get(2)?;
        // Decode f32 blob → quantize → int8 blob
        let floats: Vec<f32> = f32_blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let int8_blob = crate::quantize::embedding_to_int8_blob(&floats);
        if let Err(e) = conn.execute(
            "INSERT OR IGNORE INTO vec_memories(memory_id, embedding, namespace) VALUES (?1, vec_int8(?2), ?3)",
            rusqlite::params![id, int8_blob, namespace],
        ) {
            tracing::warn!("Failed to rebuild vec for {}: {}", id, e);
        } else {
            count += 1;
        }
    }
    tracing::info!("Rebuilt vec_memories index: {count} vectors inserted (dim={dimension}, int8)");
    Ok(count)
}

/// Insert a vector embedding into vec_memories.
pub fn insert_embedding(
    conn: &Connection,
    memory_id: &str,
    embedding: &[f32],
    namespace: &str,
) -> rusqlite::Result<()> {
    let blob = crate::quantize::embedding_to_int8_blob(embedding);
    conn.execute(
        "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES (?1, vec_int8(?2), ?3)",
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

/// Scale factor for int8 L2 distances.
/// sqlite-vec computes L2 distance on raw int8 values (range [-127, 127]).
/// Since we quantize by multiplying by 127, int8 distances are 127x larger
/// than the equivalent float distances. Dividing by this factor normalizes
/// distances back to the [0, 2] range expected for unit vectors, preserving
/// the `similarity = 1.0 - (distance / 2.0)` conversion used by callers.
pub const INT8_DISTANCE_SCALE: f64 = 127.0;

/// Search for nearest neighbors by vector similarity, filtered by namespace.
pub fn search_vectors_filtered(
    conn: &Connection,
    query_embedding: &[f32],
    namespace: &str,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = crate::quantize::embedding_to_int8_blob(query_embedding);
    let mut stmt = conn.prepare(
        "SELECT memory_id, distance
         FROM vec_memories
         WHERE embedding MATCH vec_int8(?1)
           AND k = ?2
           AND namespace = ?3
         ORDER BY distance",
    )?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64, namespace], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? / INT8_DISTANCE_SCALE))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Search for nearest neighbors without namespace filter (global search).
pub fn search_vectors(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = crate::quantize::embedding_to_int8_blob(query_embedding);
    let mut stmt = conn.prepare(
        "SELECT memory_id, distance
         FROM vec_memories
         WHERE embedding MATCH vec_int8(?1)
           AND k = ?2
         ORDER BY distance",
    )?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? / INT8_DISTANCE_SCALE))
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

// ---------------------------------------------------------------------------
// Named vector table functions
// ---------------------------------------------------------------------------

/// Initialize a named vec0 table (e.g., vec_content, vec_summary).
/// Uses int8 quantization matching the existing vec_memories table.
pub fn initialize_named_vec_table(
    conn: &Connection,
    vector_name: &str,
    dimension: usize,
) -> rusqlite::Result<VecTableStatus> {
    let table = format!("vec_{vector_name}");
    let table_exists: bool = conn
        .prepare(&format!(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}'"
        ))?
        .exists([])?;

    if !table_exists {
        let sql = format!(
            "CREATE VIRTUAL TABLE \"{table}\" USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding int8[{dimension}],
                namespace TEXT
            );"
        );
        conn.execute_batch(&sql)?;
        return Ok(VecTableStatus::Created);
    }

    // Verify dimension by probe insert
    let test_blob: Vec<u8> = vec![0u8; dimension]; // int8 = 1 byte per dim
    let dim_ok = conn
        .execute(
            &format!(
                "INSERT INTO \"{table}\"(memory_id, embedding, namespace) VALUES ('__dim_check__', vec_int8(?1), '__test__')"
            ),
            rusqlite::params![test_blob],
        )
        .is_ok();

    if dim_ok {
        conn.execute(
            &format!("DELETE FROM \"{table}\" WHERE memory_id = '__dim_check__'"),
            [],
        )?;
        return Ok(VecTableStatus::Ready);
    }

    let existing_dim = detect_existing_dimension(conn).unwrap_or(0);
    Ok(VecTableStatus::DimensionMismatch {
        existing: existing_dim,
        requested: dimension,
    })
}

/// Insert a vector embedding into a named vec table.
pub fn insert_named_embedding(
    conn: &Connection,
    table_name: &str,
    memory_id: &str,
    embedding: &[f32],
    namespace: &str,
) -> rusqlite::Result<()> {
    let blob = crate::quantize::embedding_to_int8_blob(embedding);
    conn.execute(
        &format!(
            "INSERT INTO \"{table_name}\"(memory_id, embedding, namespace) VALUES (?1, vec_int8(?2), ?3)"
        ),
        rusqlite::params![memory_id, blob, namespace],
    )?;
    Ok(())
}

/// Delete a vector embedding from a named vec table.
pub fn delete_named_embedding(
    conn: &Connection,
    table_name: &str,
    memory_id: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        &format!("DELETE FROM \"{table_name}\" WHERE memory_id = ?1"),
        rusqlite::params![memory_id],
    )?;
    Ok(())
}

/// Search nearest neighbors in a named vec table, filtered by namespace.
pub fn search_named_vectors_filtered(
    conn: &Connection,
    table_name: &str,
    query_embedding: &[f32],
    namespace: &str,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = crate::quantize::embedding_to_int8_blob(query_embedding);
    let mut stmt = conn.prepare(&format!(
        "SELECT memory_id, distance
         FROM \"{table_name}\"
         WHERE embedding MATCH vec_int8(?1)
           AND k = ?2
           AND namespace = ?3
         ORDER BY distance"
    ))?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64, namespace], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Search nearest neighbors in a named vec table, global (no namespace filter).
pub fn search_named_vectors(
    conn: &Connection,
    table_name: &str,
    query_embedding: &[f32],
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let blob = crate::quantize::embedding_to_int8_blob(query_embedding);
    let mut stmt = conn.prepare(&format!(
        "SELECT memory_id, distance
         FROM \"{table_name}\"
         WHERE embedding MATCH vec_int8(?1)
           AND k = ?2
         ORDER BY distance"
    ))?;

    let rows = stmt.query_map(rusqlite::params![blob, limit as i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Force reindex a named vec table from memory_vectors.
pub fn force_reindex_named(
    conn: &Connection,
    vector_name: &str,
    dimension: usize,
) -> rusqlite::Result<usize> {
    let table = format!("vec_{vector_name}");
    conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{table}\";"))?;
    let sql = format!(
        "CREATE VIRTUAL TABLE \"{table}\" USING vec0(
            memory_id TEXT PRIMARY KEY,
            embedding int8[{dimension}],
            namespace TEXT
        );"
    );
    conn.execute_batch(&sql)?;

    // Rebuild from memory_vectors
    let expected_bytes = dimension * 4; // f32
    let mut stmt = conn.prepare(
        "SELECT mv.memory_id, mv.embedding, mv.namespace
         FROM memory_vectors mv
         JOIN memories m ON m.id = mv.memory_id
         WHERE mv.vector_name = ?1
           AND m.status = 'active'
           AND length(mv.embedding) = ?2"
    )?;
    let mut count = 0usize;
    let mut rows = stmt.query(rusqlite::params![vector_name, expected_bytes as i64])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let f32_blob: Vec<u8> = row.get(1)?;
        let namespace: String = row.get(2)?;
        let floats: Vec<f32> = f32_blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let int8_blob = crate::quantize::embedding_to_int8_blob(&floats);
        if let Err(e) = conn.execute(
            &format!(
                "INSERT OR IGNORE INTO \"{table}\"(memory_id, embedding, namespace) VALUES (?1, vec_int8(?2), ?3)"
            ),
            rusqlite::params![id, int8_blob, namespace],
        ) {
            tracing::warn!("Failed to rebuild {table} for {id}: {e}");
        } else {
            count += 1;
        }
    }
    tracing::info!("Rebuilt {table} index: {count} vectors (dim={dimension}, int8)");
    Ok(count)
}

// ---------------------------------------------------------------------------
// memory_vectors CRUD helpers
// ---------------------------------------------------------------------------

/// Insert a raw f32 embedding into memory_vectors (source of truth for reindex).
pub fn insert_memory_vector(
    conn: &Connection,
    memory_id: &str,
    vector_name: &str,
    embedding: &[f32],
    namespace: &str,
) -> rusqlite::Result<()> {
    let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    conn.execute(
        "INSERT OR REPLACE INTO memory_vectors(memory_id, vector_name, embedding, namespace)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![memory_id, vector_name, blob, namespace],
    )?;
    Ok(())
}

/// Delete all named vectors for a memory from memory_vectors.
pub fn delete_memory_vectors(conn: &Connection, memory_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM memory_vectors WHERE memory_id = ?1",
        rusqlite::params![memory_id],
    )?;
    Ok(())
}

/// Fetch a single f32 embedding for a (memory, vector_name) pair from memory_vectors.
/// Returns None if the embedding doesn't exist.
pub fn get_f32_embedding(
    conn: &Connection,
    memory_id: &str,
    vector_name: &str,
) -> rusqlite::Result<Option<Vec<f32>>> {
    let mut stmt = conn.prepare_cached(
        "SELECT embedding FROM memory_vectors WHERE memory_id = ?1 AND vector_name = ?2"
    )?;
    let result = stmt.query_row(rusqlite::params![memory_id, vector_name], |row| {
        let blob: Vec<u8> = row.get(0)?;
        let floats: Vec<f32> = blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(floats)
    });
    match result {
        Ok(emb) => Ok(Some(emb)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Fetch f32 embeddings for a memory from memory_vectors.
/// Returns a map of vector_name → f32 embedding.
pub fn get_memory_embeddings(
    conn: &Connection,
    memory_id: &str,
) -> rusqlite::Result<std::collections::HashMap<String, Vec<f32>>> {
    let mut stmt = conn.prepare(
        "SELECT vector_name, embedding FROM memory_vectors WHERE memory_id = ?1"
    )?;
    let mut map = std::collections::HashMap::new();
    let mut rows = stmt.query(rusqlite::params![memory_id])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        let floats: Vec<f32> = blob
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        map.insert(name, floats);
    }
    Ok(map)
}

/// Count how many memories have a given vector_name populated.
pub fn count_named_vectors(
    conn: &Connection,
    vector_name: &str,
) -> rusqlite::Result<u64> {
    conn.prepare(
        "SELECT COUNT(*) FROM memory_vectors WHERE vector_name = ?1"
    )?
    .query_row(rusqlite::params![vector_name], |row| row.get(0))
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

            let results = search_vectors(&w, &embedding, 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "test-id-1");

            let results = search_vectors_filtered(&w, &embedding, "default", 5).unwrap();
            assert_eq!(results.len(), 1);

            let results = search_vectors_filtered(&w, &embedding, "other", 5).unwrap();
            assert_eq!(results.len(), 0);

            delete_embedding(&w, "test-id-1").unwrap();
            let results = search_vectors(&w, &embedding, 5).unwrap();
            assert_eq!(results.len(), 0);

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

            let all = search_vectors(&w, &emb1, 10).unwrap();
            assert_eq!(all.len(), 2);

            let alpha = search_vectors_filtered(&w, &emb1, "alpha", 10).unwrap();
            assert_eq!(alpha.len(), 1);
            assert_eq!(alpha[0].0, "mem-a");

            let beta = search_vectors_filtered(&w, &emb1, "beta", 10).unwrap();
            assert_eq!(beta.len(), 1);
            assert_eq!(beta[0].0, "mem-b");
        });
    }

    #[test]
    fn test_float_to_int8_migration() {
        // Simulate an old float-format database, then verify migration
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let w = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;

            // Insert a memory with f32 embedding into the memories table
            let emb: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let emb: Vec<f32> = emb.iter().map(|x| x / norm).collect();
            let f32_blob: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();

            w.execute(
                "INSERT INTO memories(id, namespace, content, embedding, status, created_at, updated_at, accessed_at, hash)
                 VALUES ('m1', 'default', 'test', ?1, 'active', '2026-01-01', '2026-01-01', '2026-01-01', 'h1')",
                rusqlite::params![f32_blob],
            ).unwrap();

            // Drop int8 table, recreate as float to simulate old schema
            w.execute_batch("DROP TABLE vec_memories;").unwrap();
            let sql = format!(
                "CREATE VIRTUAL TABLE vec_memories USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    embedding float[{dim}],
                    namespace TEXT
                );"
            );
            w.execute_batch(&sql).unwrap();

            // Insert into float table
            w.execute(
                "INSERT INTO vec_memories(memory_id, embedding, namespace) VALUES ('m1', ?1, 'default')",
                rusqlite::params![f32_blob],
            ).unwrap();

            // Now run initialize_vec_table — should detect float and upgrade
            let status = initialize_vec_table(&w, dim).unwrap();
            assert_eq!(status, VecTableStatus::QuantizationUpgraded);

            // Verify search works with int8 table
            let results = search_vectors(&w, &emb, 5).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "m1");
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
