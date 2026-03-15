//! OpenAI-compatible embedding API client.
//!
//! Supports `text-embedding-3-small`, `text-embedding-3-large`, and any
//! OpenAI-compatible embedding endpoint.

use crate::model::EmbedError;
use std::sync::Mutex;

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    /// Must be accessible from sync context, so we store the runtime handle.
    rt: Mutex<tokio::runtime::Handle>,
    base_url: String,
    model: String,
    api_key: String,
    dimension: usize,
}

impl OpenAiEmbedder {
    pub fn new(
        base_url: String,
        model: String,
        api_key: String,
        dimension: usize,
    ) -> Self {
        let rt = tokio::runtime::Handle::current();
        Self {
            client: reqwest::Client::new(),
            rt: Mutex::new(rt),
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            api_key,
            dimension,
        }
    }

    fn call_api(&self, input: &str) -> Result<Vec<f32>, EmbedError> {
        let body = serde_json::json!({
            "input": input,
            "model": &self.model,
            "dimensions": self.dimension,
        });

        let url = format!("{}/embeddings", &self.base_url);
        let rt = self.rt.lock().map_err(|e| EmbedError::Config(format!("runtime lock: {e}")))?;

        let resp: serde_json::Value = rt.block_on(async {
            let resp = self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", &self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| EmbedError::Config(format!("OpenAI embedding request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(EmbedError::Config(format!(
                    "OpenAI embedding API returned {status}: {text}"
                )));
            }

            resp.json::<serde_json::Value>()
                .await
                .map_err(|e| EmbedError::Config(format!("failed to parse embedding response: {e}")))
        })?;

        let embedding = resp["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| EmbedError::Config("missing embedding in API response".into()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<f32>>();

        if embedding.len() != self.dimension {
            return Err(EmbedError::Config(format!(
                "expected dimension {} but API returned {}",
                self.dimension,
                embedding.len()
            )));
        }

        Ok(embedding)
    }
}

impl crate::Embedder for OpenAiEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.call_api(text)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // OpenAI embeddings don't use query/document distinction
        self.call_api(text)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// Safety: reqwest::Client is Send+Sync, Mutex<Handle> is Send+Sync
unsafe impl Send for OpenAiEmbedder {}
unsafe impl Sync for OpenAiEmbedder {}
