use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoreError {
    #[error("invalid namespace: {0}")]
    InvalidNamespace(String),

    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("invalid content: {0}")]
    InvalidContent(String),
}
