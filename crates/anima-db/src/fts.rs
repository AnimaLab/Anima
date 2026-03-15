use rusqlite::Connection;

/// Insert content into the FTS5 index.
pub fn insert_fts(
    conn: &Connection,
    memory_id: &str,
    namespace: &str,
    content: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO fts_memories(memory_id, namespace, content) VALUES (?1, ?2, ?3)",
        rusqlite::params![memory_id, namespace, content],
    )?;
    Ok(())
}

/// Delete content from the FTS5 index.
pub fn delete_fts(conn: &Connection, memory_id: &str) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM fts_memories WHERE memory_id = ?1",
        rusqlite::params![memory_id],
    )?;
    Ok(())
}

/// Search FTS5 for matching memories.
/// Returns (memory_id, bm25_score) pairs ordered by relevance.
/// BM25 scores are negated so higher = better match.
pub fn search_fts(
    conn: &Connection,
    query: &str,
    namespace_pattern: &str,
    limit: usize,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let sanitized = sanitize_fts_query(query);
    if sanitized.is_empty() {
        return Ok(vec![]);
    }

    // Join with memories table for namespace filtering and status check
    let mut stmt = conn.prepare(
        "SELECT f.memory_id, -rank as score
         FROM fts_memories f
         JOIN memories m ON f.memory_id = m.id
         WHERE fts_memories MATCH ?1
           AND m.namespace LIKE ?2
           AND m.status = 'active'
         ORDER BY rank
         LIMIT ?3",
    )?;

    let rows = stmt.query_map(
        rusqlite::params![sanitized, namespace_pattern, limit as i64],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
    )?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Common English stopwords to filter from FTS queries.
/// These words appear in nearly every document and add no search value.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "his", "her", "its", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "after", "before",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "user", // common prefix in memory content
];

/// Sanitize user input for FTS5 query syntax.
/// Wraps each word in double quotes to prevent FTS5 syntax errors.
/// Filters out stopwords that would match too many documents.
fn sanitize_fts_query(input: &str) -> String {
    let words: Vec<_> = input
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .filter(|w| !STOPWORDS.contains(&w.to_lowercase().as_str()))
        .map(|word| {
            let escaped = word.replace('"', "\"\"");
            format!("\"{escaped}\"")
        })
        .collect();

    // If all words were stopwords, fall back to the original (better than empty)
    if words.is_empty() && !input.trim().is_empty() {
        return input
            .split_whitespace()
            .map(|word| {
                let escaped = word.replace('"', "\"\"");
                format!("\"{escaped}\"")
            })
            .collect::<Vec<_>>()
            .join(" ");
    }

    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_fts_query() {
        assert_eq!(sanitize_fts_query("hello world"), "\"hello\" \"world\"");
        assert_eq!(sanitize_fts_query(""), "");
        assert_eq!(sanitize_fts_query("  spaces  "), "\"spaces\"");
    }

    #[test]
    fn test_stopword_filtering() {
        // Stopwords should be removed
        assert_eq!(sanitize_fts_query("the cat"), "\"cat\"");
        assert_eq!(sanitize_fts_query("User is vegetarian"), "\"vegetarian\"");
        // If ALL words are stopwords, keep them (fallback)
        assert_eq!(sanitize_fts_query("the"), "\"the\"");
        assert_eq!(sanitize_fts_query("a the is"), "\"a\" \"the\" \"is\"");
    }

    #[test]
    fn test_sanitize_fts_query_special_chars() {
        assert_eq!(
            sanitize_fts_query("hello \"world\""),
            "\"hello\" \"\"\"world\"\"\""
        );
    }

    #[test]
    fn test_fts_insert_delete() {
        use crate::pool::DbPool;

        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let w = pool.writer().await;

            insert_fts(&w, "mem-1", "test/ns", "User prefers dark mode").unwrap();

            // Delete should work
            let result = delete_fts(&w, "mem-1");
            assert!(result.is_ok(), "FTS delete failed: {:?}", result.err());
        });
    }
}
