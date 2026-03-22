//! Cross-encoder re-ranker using ONNX Runtime.
//!
//! Takes (query, document) pairs and returns relevance scores.
//! Uses bge-reranker-v2-m3 INT8 by default — a true cross-encoder that outputs
//! a single logit per pair (not a generative model).

use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::{Session, SessionInputValue};
use ort::value::Value;
use tokenizers::Tokenizer;

use crate::model::EmbedError;

/// Cross-encoder re-ranker model.
/// Thread-safe via internal Mutex on the session.
pub struct Reranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
}

impl Reranker {
    /// Load a cross-encoder ONNX model + tokenizer.
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        max_length: usize,
    ) -> Result<Self, EmbedError> {
        let session = Session::builder()
            .map_err(ort::Error::from)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(4)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)
            .map_err(ort::Error::from)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length,
        })
    }

    /// Score (query, document) pairs. Returns one relevance score per pair.
    ///
    /// For long documents, splits into overlapping chunks of `max_length` tokens,
    /// scores each chunk, and returns the max score per document.
    pub fn score_pairs(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f64>, EmbedError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Build (query, document_chunk) pairs. Long documents get split into
        // overlapping windows so no content is truncated.
        let mut pair_texts: Vec<(usize, String, String)> = Vec::new(); // (doc_index, query, chunk)
        for (doc_idx, doc) in documents.iter().enumerate() {
            let chunks = self.chunk_document(query, doc);
            for chunk in chunks {
                pair_texts.push((doc_idx, query.to_string(), chunk));
            }
        }

        // Score all pairs
        let pair_scores = self.score_raw_pairs(&pair_texts)?;

        // Aggregate: max score per document
        let mut doc_scores = vec![f64::NEG_INFINITY; documents.len()];
        for (i, &(doc_idx, _, _)) in pair_texts.iter().enumerate() {
            if pair_scores[i] > doc_scores[doc_idx] {
                doc_scores[doc_idx] = pair_scores[i];
            }
        }

        Ok(doc_scores)
    }

    /// Split a document into chunks that fit within max_length tokens
    /// (accounting for query tokens + special tokens).
    fn chunk_document(&self, query: &str, document: &str) -> Vec<String> {
        // Estimate query token count
        let query_tokens = self.tokenizer.encode(query, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0);

        // Reserve tokens for query + [CLS] + [SEP] + [SEP]
        let doc_budget = self.max_length.saturating_sub(query_tokens + 3);
        if doc_budget == 0 {
            return vec![document.to_string()];
        }

        let doc_encoding = match self.tokenizer.encode(document, false) {
            Ok(e) => e,
            Err(_) => return vec![document.to_string()],
        };

        let doc_token_count = doc_encoding.get_ids().len();
        if doc_token_count <= doc_budget {
            return vec![document.to_string()];
        }

        // Split into overlapping chunks by token offsets
        let offsets = doc_encoding.get_offsets();
        let overlap = doc_budget / 4; // 25% overlap
        let step = doc_budget.saturating_sub(overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < doc_token_count {
            let end = (start + doc_budget).min(doc_token_count);
            let char_start = offsets[start].0;
            let char_end = if end < offsets.len() { offsets[end - 1].1 } else { document.len() };
            let chunk = &document[char_start..char_end.min(document.len())];
            if !chunk.trim().is_empty() {
                chunks.push(chunk.to_string());
            }
            if end >= doc_token_count {
                break;
            }
            start += step;
        }

        if chunks.is_empty() {
            chunks.push(document.to_string());
        }
        chunks
    }

    /// Run ONNX inference on pre-built (query, document) pairs.
    fn score_raw_pairs(
        &self,
        pairs: &[(usize, String, String)],
    ) -> Result<Vec<f64>, EmbedError> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all pairs as (query, document) sentence pairs
        let encoded: Vec<_> = pairs
            .iter()
            .map(|(_, q, d)| {
                self.tokenizer
                    .encode((q.as_str(), d.as_str()), true)
                    .map_err(|e| EmbedError::Tokenizer(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let batch_size = encoded.len();
        let max_len = encoded.iter().map(|e| e.get_ids().len().min(self.max_length)).max().unwrap_or(0);

        // Build padded input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, enc) in encoded.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(max_len);
            for j in 0..len {
                input_ids[i * max_len + j] = ids[j] as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
            }
        }

        let input_ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| EmbedError::Shape(e.to_string()))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, max_len), attention_mask)
            .map_err(|e| EmbedError::Shape(e.to_string()))?;

        let mut session = self.session.lock().unwrap();

        let input_ids_value = Value::from_array(input_ids_array)
            .map_err(ort::Error::from)?;
        let attention_mask_value = Value::from_array(attention_mask_array)
            .map_err(ort::Error::from)?;

        let inputs: Vec<(String, SessionInputValue<'_>)> = vec![
            ("input_ids".to_string(), SessionInputValue::from(input_ids_value)),
            ("attention_mask".to_string(), SessionInputValue::from(attention_mask_value)),
        ];

        let outputs = session.run(inputs).map_err(ort::Error::from)?;

        // Cross-encoder output: logits [batch, 1] or [batch]
        // Apply sigmoid to get [0, 1] relevance scores
        let (_out_shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(ort::Error::from)?;

        let all_scores: Vec<f64> = logits_data
            .iter()
            .map(|&logit| sigmoid(logit as f64))
            .collect();

        // If output is [batch, num_classes], take the last logit per row
        // (cross-encoders output a single relevance logit)
        if all_scores.len() == batch_size {
            Ok(all_scores)
        } else {
            // Multi-output: take one score per batch item (last value per row)
            let cols = all_scores.len() / batch_size;
            Ok((0..batch_size).map(|i| all_scores[i * cols + cols - 1]).collect())
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
