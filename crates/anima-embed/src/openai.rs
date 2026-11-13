//! OpenAI-compatible embedding API client.
//!
//! Supports `text-embedding-3-small`, `text-embedding-3-large`, and any
//! OpenAI-compatible embedding endpoint.

use std::future::Future;

use crate::model::EmbedError;

/// Run `fut` to completion on `handle` from a thread that is not part of any
/// runtime.
///
/// The `Embedder` trait is synchronous and the server calls it from async
/// handlers, that is from tokio worker threads, where `Handle::block_on`
/// panics ("Cannot start a runtime from within a runtime"). A scoped thread
/// sits outside the runtime, so this works from worker threads, blocking
/// threads and plain threads alike.
pub(crate) fn block_on_outside_runtime<T, F>(handle: &tokio::runtime::Handle, fut: F) -> T
where
    T: Send,
    F: Future<Output = T> + Send,
{
    std::thread::scope(|s| {
        s.spawn(|| handle.block_on(fut))
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

/// POST a JSON body with a bearer token and return the JSON response.
pub(crate) async fn post_json(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    what: &str,
) -> Result<serde_json::Value, EmbedError> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(body)
        .send()
        .await
        .map_err(|e| EmbedError::Config(format!("{what} request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(EmbedError::Config(format!("{what} API returned {status}: {text}")));
    }

    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| EmbedError::Config(format!("failed to parse {what} response: {e}")))
}

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    rt: tokio::runtime::Handle,
    base_url: String,
    model: String,
    api_key: String,
    dimension: usize,
}

impl OpenAiEmbedder {
    /// Must be called from within a tokio runtime: requests are driven on it.
    pub fn new(
        base_url: String,
        model: String,
        api_key: String,
        dimension: usize,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            rt: tokio::runtime::Handle::current(),
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

        let resp = block_on_outside_runtime(
            &self.rt,
            post_json(&self.client, &url, &self.api_key, &body, "embedding"),
        )?;

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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::Embedder;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve `responses.len()` HTTP requests, answering each with the next JSON
    /// body, and hand back the request bodies received. Returns the base URL.
    pub(crate) async fn mock_server(
        responses: Vec<serde_json::Value>,
    ) -> (String, tokio::task::JoinHandle<Vec<serde_json::Value>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let mut received = Vec::new();
            for body in responses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // Read headers, then exactly Content-Length bytes of body.
                let (head_end, len) = loop {
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .map(|v| v.trim().parse::<usize>().unwrap())
                            .unwrap_or(0);
                        break (pos + 4, len);
                    }
                };
                while buf.len() < head_end + len {
                    let n = sock.read(&mut chunk).await.unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                }
                received.push(serde_json::from_slice(&buf[head_end..head_end + len]).unwrap());
                let payload = body.to_string();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                sock.write_all(resp.as_bytes()).await.unwrap();
            }
            received
        });
        (base, handle)
    }

    /// Regression, v0.3.0: `call_api` ran `Handle::block_on` on the tokio
    /// worker thread that called it, so the first embedding request panicked
    /// and poisoned the runtime lock for every request after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embed_from_a_runtime_worker_does_not_panic() {
        let (base, server) =
            mock_server(vec![serde_json::json!({"data": [{"embedding": [0.5, 0.25, 0.125]}]})]).await;
        let embedder = OpenAiEmbedder::new(base, "m".into(), "k".into(), 3);

        let v = tokio::spawn(async move { embedder.embed("hello") })
            .await
            .expect("embedding task panicked")
            .unwrap();

        assert_eq!(v, vec![0.5, 0.25, 0.125]);
        let received = server.await.unwrap();
        assert_eq!(received[0]["input"], "hello");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrong_dimension_is_an_error() {
        let (base, _server) =
            mock_server(vec![serde_json::json!({"data": [{"embedding": [0.5]}]})]).await;
        let embedder = OpenAiEmbedder::new(base, "m".into(), "k".into(), 3);
        let r = tokio::spawn(async move { embedder.embed("hello") }).await.unwrap();
        assert!(r.is_err());
    }
}
