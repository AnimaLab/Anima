pub mod download;
pub mod model;
pub mod openai;

/// Trait for embedding models. Implemented by both local ONNX models
/// and remote API-based embedders (e.g. OpenAI text-embedding-3-small).
pub trait Embedder: Send + Sync {
    /// Embed a document (no instruction prefix).
    fn embed(&self, text: &str) -> Result<Vec<f32>, model::EmbedError>;

    /// Embed a search query (with instruction prefix if applicable).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, model::EmbedError>;

    /// Embed multiple texts at once.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, model::EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Embedding dimension.
    fn dimension(&self) -> usize;
}

impl<T: Embedder + ?Sized> Embedder for std::sync::Arc<T> {
    fn embed(&self, text: &str) -> Result<Vec<f32>, model::EmbedError> {
        (**self).embed(text)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, model::EmbedError> {
        (**self).embed_query(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, model::EmbedError> {
        (**self).embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }
}

impl Embedder for model::EmbeddingModel {
    fn embed(&self, text: &str) -> Result<Vec<f32>, model::EmbedError> {
        self.embed(text)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, model::EmbedError> {
        self.embed_query(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, model::EmbedError> {
        self.embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        self.dimension()
    }
}
