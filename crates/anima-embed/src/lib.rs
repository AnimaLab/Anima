pub mod download;
pub mod model;
pub mod openai;
pub mod reranker;

/// Sparse vector: sorted list of (token_id, weight) pairs with non-zero weights.
/// Produced by BGE-M3's sparse linear head. Typically 100-300 entries.
#[derive(Debug, Clone, Default)]
pub struct SparseVector(pub Vec<(u32, f32)>);

impl SparseVector {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Encode to compact binary: [u32 count][(u32 token_id, f32 weight) × count]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.0.len() * 8);
        buf.extend_from_slice(&(self.0.len() as u32).to_le_bytes());
        for &(tid, w) in &self.0 {
            buf.extend_from_slice(&tid.to_le_bytes());
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Decode from compact binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("sparse vector blob too short".into());
        }
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let expected = 4 + count * 8;
        if data.len() < expected {
            return Err(format!("sparse vector blob: expected {expected} bytes, got {}", data.len()));
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * 8;
            let tid = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            let w = f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap());
            entries.push((tid, w));
        }
        Ok(Self(entries))
    }
}

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

    fn embed_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        Ok((self.embed(text)?, SparseVector::default()))
    }

    fn embed_query_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        Ok((self.embed_query(text)?, SparseVector::default()))
    }
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

    fn embed_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        (**self).embed_with_sparse(text)
    }

    fn embed_query_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        (**self).embed_query_with_sparse(text)
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

    fn embed_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        model::EmbeddingModel::embed_with_sparse(self, text)
    }

    fn embed_query_with_sparse(&self, text: &str) -> Result<(Vec<f32>, SparseVector), model::EmbedError> {
        model::EmbeddingModel::embed_query_with_sparse(self, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_vector_roundtrip() {
        let sv = SparseVector(vec![(5, 0.3), (100, 1.2), (999, 0.01)]);
        let bytes = sv.to_bytes();
        let decoded = SparseVector::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.0.len(), 3);
        assert_eq!(decoded.0[0].0, 5);
        assert!((decoded.0[0].1 - 0.3).abs() < 1e-6);
        assert_eq!(decoded.0[2].0, 999);
    }

    #[test]
    fn sparse_vector_empty() {
        let sv = SparseVector::default();
        assert!(sv.is_empty());
        let bytes = sv.to_bytes();
        assert_eq!(bytes.len(), 4);
        let decoded = SparseVector::from_bytes(&bytes).unwrap();
        assert!(decoded.is_empty());
    }
}
