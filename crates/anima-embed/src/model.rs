use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use ort::session::{Session, SessionInputValue};
use ort::value::Value;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    Mean,
    LastToken,
}

impl PoolingStrategy {
    pub fn parse(value: &str) -> Result<Self, EmbedError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "last_token" | "last-token" | "lasttoken" => Ok(Self::LastToken),
            other => Err(EmbedError::Config(format!(
                "invalid embedding pooling '{other}' (expected 'mean' or 'last_token')"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::LastToken => "last_token",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("ort error: {0}")]
    Ort(#[from] ort::Error<()>),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("shape error: {0}")]
    Shape(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("config error: {0}")]
    Config(String),
}

/// Helper to convert typed ort errors to EmbedError
fn map_ort<T, R>(r: Result<T, ort::Error<R>>) -> Result<T, EmbedError>
where ort::Error<()>: From<ort::Error<R>> {
    r.map_err(|e| EmbedError::Ort(ort::Error::from(e)))
}

/// Default instruction prefix for query embeddings (Qwen3-Embedding style).
/// Documents are embedded without instruction; queries use this prefix for
/// asymmetric retrieval as recommended by the model card.
/// Customize via config `embedding.query_instruction` for your domain.
const _DEFAULT_QUERY_INSTRUCTION: &str =
    "Instruct: Given a question about someone's memories or past conversations, retrieve the most relevant memory excerpts\nQuery:";

/// ONNX-based sentence embedding model.
/// Thread-safe via internal Mutex on the session.
pub struct EmbeddingModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dimension: usize,
    pooling: PoolingStrategy,
    /// Optional instruction prefix prepended to queries (not documents).
    query_instruction: Option<String>,
}

impl EmbeddingModel {
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        dimension: usize,
        pooling: PoolingStrategy,
        query_instruction: Option<String>,
    ) -> Result<Self, EmbedError> {
        let session = Session::builder()
            .map_err(ort::Error::from)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)
            .map_err(ort::Error::from)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimension,
            pooling,
            query_instruction,
        })
    }

    /// Embed a document (no instruction prefix).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let results = self.embed_single(text)?;
        Ok(results)
    }

    /// Embed a search query (with instruction prefix if configured).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if let Some(ref instruction) = self.query_instruction {
            let prefixed = format!("{}{}", instruction, text);
            let results = self.embed_batch(&[&prefixed])?;
            Ok(results.into_iter().next().unwrap())
        } else {
            self.embed(text)
        }
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Process each text individually to avoid padding-induced numerical
        // differences.  Batched ONNX inference with padding produces slightly
        // different hidden states (cosine ~0.992 instead of 1.0), which
        // degrades retrieval when queries are embedded one-at-a-time via
        // embed().  The embedding step is not the bottleneck (LLM calls are),
        // so the marginal latency is negligible.
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_single(text)?);
        }
        Ok(results)
    }

    /// Core single-text ONNX inference. Both embed() and embed_batch() use this.
    fn embed_single(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let texts = &[text];
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut input_ids_vec = vec![0i64; batch_size * max_len];
        let mut attn_mask_vec = vec![0i64; batch_size * max_len];
        let mut token_type_vec = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();
            for (j, (&id, (&m, &t))) in ids.iter().zip(mask.iter().zip(types.iter())).enumerate() {
                let idx = i * max_len + j;
                input_ids_vec[idx] = id as i64;
                attn_mask_vec[idx] = m as i64;
                token_type_vec[idx] = t as i64;
            }
        }

        let shape = vec![batch_size, max_len];

        let input_ids_val: Value<_> = map_ort(Value::from_array((shape.clone(), input_ids_vec)))?;
        let attn_mask_val: Value<_> = map_ort(Value::from_array((shape.clone(), attn_mask_vec.clone())))?;
        let token_type_val: Value<_> = map_ort(Value::from_array((shape.clone(), token_type_vec)))?;

        let mut session = self.session.lock().map_err(|e| EmbedError::Shape(format!("lock: {e}")))?;

        // Build inputs dynamically based on what the model expects.
        // Encoder-only models (e.g. BGE-M3) need 2-3 inputs.
        // Decoder-based embedding models (e.g. Qwen3-Embedding) also need
        // past_key_values.*.key/value tensors initialized to empty.
        let mut inputs_map: Vec<(String, SessionInputValue<'_>)> = Vec::new();

        // Collect output names before run (to avoid borrow conflict)
        let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();

        // Check if model needs decoder-style inputs (past_key_values, position_ids)
        let _has_kv_cache = session.inputs().iter().any(|i| i.name().starts_with("past_key_values"));

        for input in session.inputs() {
            let name = input.name().to_string();
            if name == "input_ids" {
                inputs_map.push((name, SessionInputValue::from(input_ids_val.clone())));
            } else if name == "attention_mask" {
                inputs_map.push((name, SessionInputValue::from(attn_mask_val.clone())));
            } else if name == "token_type_ids" {
                inputs_map.push((name, SessionInputValue::from(token_type_val.clone())));
            } else if name == "position_ids" {
                // Generate position IDs: [0, 1, 2, ..., seq_len-1] for each batch item
                let mut pos_ids = vec![0i64; batch_size * max_len];
                for b in 0..batch_size {
                    for s in 0..max_len {
                        pos_ids[b * max_len + s] = s as i64;
                    }
                }
                let pos_val: Value<_> = map_ort(Value::from_array((shape.clone(), pos_ids)))?;
                inputs_map.push((name, SessionInputValue::from(pos_val)));
            } else if name.starts_with("past_key_values") {
                // KV cache: provide zero-filled tensors for first-pass inference.
                // Shape: [batch, num_kv_heads, 0, head_dim] — 0 past tokens.
                // ORT's from_array rejects 0-length dims, so use ndarray with IxDyn.
                if let ort::value::ValueType::Tensor { ty: _, shape, .. } = input.dtype() {
                    let dims: Vec<usize> = shape.iter().enumerate().map(|(i, &d)| {
                        if d > 0 { d as usize }
                        else if i == 0 { batch_size }
                        else { 0 }  // past sequence length = 0
                    }).collect();
                    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&dims));
                    let val: Value<_> = map_ort(Value::from_array(arr))?;
                    inputs_map.push((name, SessionInputValue::from(val)));
                }
            }
        }

        let outputs = map_ort(session.run(inputs_map))?;

        // Find the right output: prefer "last_hidden_state", fall back to first 3D tensor.
        // Decoder models with KV cache output present.*.key/value alongside hidden states.
        let hidden_idx = output_names.iter()
            .position(|n| n == "last_hidden_state")
            .unwrap_or(0);

        let (out_shape, out_data) = outputs[hidden_idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedError::Shape(format!("extract output '{}' failed: {e}", output_names.get(hidden_idx).unwrap_or(&"?".to_string()))))?;

        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 3 {
            return Err(EmbedError::Shape(format!("expected 3D output, got {:?} for '{}'", dims, output_names.get(hidden_idx).unwrap_or(&"?".to_string()))));
        }

        let (bs, seq_len, hidden) = (dims[0], dims[1], dims[2]);

        let token_embeddings = Array3::from_shape_vec((bs, seq_len, hidden), out_data.to_vec())
            .map_err(|e| EmbedError::Shape(format!("reshape: {e}")))?;

        let attention_mask = Array2::from_shape_vec(
            (batch_size, max_len),
            attn_mask_vec,
        )
        .map_err(|e| EmbedError::Shape(format!("attn mask reshape: {e}")))?;

        let pooled = match self.pooling {
            PoolingStrategy::Mean => mean_pool(&token_embeddings, &attention_mask),
            PoolingStrategy::LastToken => last_token_pool(&token_embeddings, &attention_mask),
        };
        let projected = project_dimension(&pooled, self.dimension)?;
        let normalized = l2_normalize(&projected);

        let all: Vec<Vec<f32>> = normalized.outer_iter().map(|row| row.to_vec()).collect();
        Ok(all.into_iter().next().unwrap())
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn pooling(&self) -> PoolingStrategy {
        self.pooling
    }
}

fn mean_pool(token_embeddings: &Array3<f32>, attention_mask: &Array2<i64>) -> Array2<f32> {
    let batch_size = token_embeddings.shape()[0];
    let hidden_dim = token_embeddings.shape()[2];
    let mut result = Array2::<f32>::zeros((batch_size, hidden_dim));

    for b in 0..batch_size {
        let mut sum = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;
        for s in 0..token_embeddings.shape()[1] {
            let m = attention_mask[[b, s]] as f32;
            if m > 0.0 {
                for d in 0..hidden_dim {
                    sum[d] += token_embeddings[[b, s, d]] * m;
                }
                count += m;
            }
        }
        if count > 0.0 {
            for d in 0..hidden_dim {
                result[[b, d]] = sum[d] / count;
            }
        }
    }
    result
}

fn last_token_pool(token_embeddings: &Array3<f32>, attention_mask: &Array2<i64>) -> Array2<f32> {
    let batch_size = token_embeddings.shape()[0];
    let seq_len = token_embeddings.shape()[1];
    let hidden_dim = token_embeddings.shape()[2];
    let mut result = Array2::<f32>::zeros((batch_size, hidden_dim));

    for b in 0..batch_size {
        let mut last_idx = 0usize;
        for s in (0..seq_len).rev() {
            if attention_mask[[b, s]] > 0 {
                last_idx = s;
                break;
            }
        }
        for d in 0..hidden_dim {
            result[[b, d]] = token_embeddings[[b, last_idx, d]];
        }
    }

    result
}

fn project_dimension(embeddings: &Array2<f32>, dimension: usize) -> Result<Array2<f32>, EmbedError> {
    if dimension == 0 {
        return Err(EmbedError::Config(
            "embedding dimension must be greater than zero".to_string(),
        ));
    }
    let hidden_dim = embeddings.shape()[1];
    if dimension == hidden_dim {
        return Ok(embeddings.clone());
    }
    if dimension > hidden_dim {
        return Err(EmbedError::Config(format!(
            "requested embedding dimension {dimension} exceeds model hidden size {hidden_dim}"
        )));
    }

    let rows = embeddings.shape()[0];
    let mut out = Array2::<f32>::zeros((rows, dimension));
    for b in 0..rows {
        for d in 0..dimension {
            out[[b, d]] = embeddings[[b, d]];
        }
    }
    Ok(out)
}

fn l2_normalize(embeddings: &Array2<f32>) -> Array2<f32> {
    let mut result = embeddings.clone();
    for mut row in result.outer_iter_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
    result
}

unsafe impl Send for EmbeddingModel {}
unsafe impl Sync for EmbeddingModel {}
