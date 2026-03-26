use std::path::Path;
use std::sync::Arc;

use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::Connection;
use sqlite_vec::sqlite3_vec_init;
use tokio::sync::Mutex;

use crate::schema;

/// Register sqlite-vec as an auto-extension. Safe to call multiple times.
fn register_sqlite_vec() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    });
}

/// Database pool with a single writer and multiple reader connections.
/// SQLite WAL mode allows concurrent reads with a single writer.
pub struct DbPool {
    writer: Mutex<Connection>,
    db_path: String,
}

impl DbPool {
    /// Default embedding dimension for local test/in-memory stores.
    pub const DEFAULT_DIMENSION: usize = 1024;

    /// Open (or create) a database at the given path.
    /// Returns the pool and the vec table initialization status.
    pub fn open(path: impl AsRef<Path>, dimension: usize) -> Result<(Arc<Self>, crate::vector::VecTableStatus), DbError> {
        register_sqlite_vec();

        let path_str = path.as_ref().to_string_lossy().to_string();

        let writer = Connection::open(&path_str).map_err(DbError::Sqlite)?;
        schema::initialize(&writer).map_err(DbError::Sqlite)?;
        let vec_status = crate::vector::initialize_vec_table(&writer, dimension).map_err(DbError::Sqlite)?;

        Ok((Arc::new(Self {
            writer: Mutex::new(writer),
            db_path: path_str,
        }), vec_status))
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Arc<Self>, DbError> {
        register_sqlite_vec();

        let writer = Connection::open_in_memory().map_err(DbError::Sqlite)?;
        schema::initialize(&writer).map_err(DbError::Sqlite)?;
        let _ = crate::vector::initialize_vec_table(&writer, Self::DEFAULT_DIMENSION)
            .map_err(DbError::Sqlite)?;

        Ok(Arc::new(Self {
            writer: Mutex::new(writer),
            db_path: ":memory:".into(),
        }))
    }

    /// Acquire the write connection. Holds the lock until the guard is dropped.
    pub async fn writer(&self) -> tokio::sync::MutexGuard<'_, Connection> {
        self.writer.lock().await
    }

    /// Open a new read-only connection. Each call creates a fresh connection.
    pub fn reader(&self) -> Result<Connection, DbError> {
        let conn = Connection::open_with_flags(
            &self.db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(DbError::Sqlite)?;

        Ok(conn)
    }

    /// Quick health check: run a trivial query to verify the DB is accessible.
    pub async fn ping(&self) -> Result<(), DbError> {
        let conn = self.writer.lock().await;
        conn.execute_batch("SELECT 1").map_err(DbError::Sqlite)?;
        Ok(())
    }

    pub fn db_path(&self) -> &str {
        &self.db_path
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("database error: {0}")]
    Other(String),
}
