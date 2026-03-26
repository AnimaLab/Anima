use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("llm returned error: {0}")]
    LlmResponse(String),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LlmUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct LlmCompletion {
    pub content: String,
    pub usage: Option<LlmUsage>,
}

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete_with_usage(&self, prompt: &str) -> Result<LlmCompletion, LlmError>;

    async fn complete(&self, prompt: &str) -> Result<String, LlmError> {
        let out = self.complete_with_usage(prompt).await?;
        Ok(out.content)
    }
}

fn llm_mock_mode_enabled() -> bool {
    std::env::var("ANIMA_LLM_MODE")
        .or_else(|_| std::env::var("MEMI_LLM_MODE"))
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "mock"
            )
        })
        .unwrap_or(false)
}

fn llm_fixture_dir() -> Option<PathBuf> {
    std::env::var("ANIMA_LLM_FIXTURES_DIR")
        .or_else(|_| std::env::var("MEMI_LLM_FIXTURES_DIR"))
        .ok()
        .map(PathBuf::from)
}

fn fixture_key_for_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn load_mock_fixture_text(prompt: &str) -> Option<String> {
    let dir = llm_fixture_dir()?;
    let key = fixture_key_for_text(prompt);
    let txt = dir.join(format!("{key}.txt"));
    if let Ok(data) = std::fs::read_to_string(&txt) {
        return Some(data);
    }
    let json = dir.join(format!("{key}.json"));
    std::fs::read_to_string(&json).ok()
}

fn mock_completion_for_prompt(prompt: &str) -> String {
    if let Some(data) = load_mock_fixture_text(prompt) {
        return data;
    }

    if prompt.contains("memory consolidation engine using predict-calibrate learning") {
        return r#"{"action":"create","target_id":null,"merged_content":null,"novel_content":null,"reasoning":"mock mode"}"#.to_string();
    }
    if prompt.contains("memory reflection engine") {
        return r#"{"facts":[]}"#.to_string();
    }
    if prompt.contains("memory deduction engine") {
        return r#"{"deductions":[]}"#.to_string();
    }
    if prompt.contains("memory induction engine") {
        return r#"{"patterns":[]}"#.to_string();
    }
    "{}".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[tokio::test]
    async fn mock_mode_loads_prompt_fixture() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let prompt = "fixture-test-prompt";
        let key = fixture_key_for_text(prompt);
        let fixture = dir.path().join(format!("{key}.txt"));
        std::fs::write(&fixture, "{\"facts\":[{\"content\":\"fixture\"}]}").unwrap();

        std::env::set_var("ANIMA_LLM_MODE", "mock");
        std::env::set_var("ANIMA_LLM_FIXTURES_DIR", dir.path());

        let client = OpenAiCompatClient::new(
            "http://127.0.0.1:1".to_string(),
            None,
            "mock".to_string(),
        );
        let out = client.complete(prompt).await.unwrap();
        assert!(out.contains("fixture"));

        std::env::remove_var("ANIMA_LLM_FIXTURES_DIR");
        std::env::remove_var("ANIMA_LLM_MODE");
    }

    #[test]
    fn parses_openai_usage_payload() {
        let payload = r#"{
            "choices": [{"message": {"content": "{\"facts\":[]}"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        }"#;
        let parsed: ChatResponse = serde_json::from_str(payload).unwrap();
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(12));
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(19));
    }

    #[test]
    fn parses_ollama_usage_payload() {
        let payload = r#"{
            "response": "{\"facts\":[]}",
            "prompt_eval_count": 23,
            "eval_count": 11
        }"#;
        let parsed: OllamaResponse = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed.prompt_eval_count, Some(23));
        assert_eq!(parsed.eval_count, Some(11));
    }
}

/// Ollama-compatible HTTP client.
pub struct OllamaClient {
    base_url: String,
    model: String,
    temperature: f64,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            base_url,
            model,
            temperature: 0.2,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    format: String,
    temperature: f64,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

#[async_trait::async_trait]
impl LlmClient for OllamaClient {
    async fn complete_with_usage(&self, prompt: &str) -> Result<LlmCompletion, LlmError> {
        if llm_mock_mode_enabled() {
            return Ok(LlmCompletion {
                content: mock_completion_for_prompt(prompt),
                usage: None,
            });
        }

        let req = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            format: "json".to_string(),
            temperature: self.temperature,
        };

        let resp = self
            .http
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<OllamaResponse>()
            .await?;

        let usage = if resp.prompt_eval_count.is_some() || resp.eval_count.is_some() {
            let prompt_tokens = resp.prompt_eval_count.unwrap_or(0) as usize;
            let completion_tokens = resp.eval_count.unwrap_or(0) as usize;
            Some(LlmUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            })
        } else {
            None
        };

        Ok(LlmCompletion {
            content: resp.response,
            usage,
        })
    }
}

/// OpenAI-compatible API client.
pub struct OpenAiCompatClient {
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f64,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: String, api_key: Option<String>, model: String) -> Self {
        Self {
            base_url,
            api_key,
            model,
            temperature: 0.2,
            http: reqwest::Client::new(),
        }
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[async_trait::async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn complete_with_usage(&self, prompt: &str) -> Result<LlmCompletion, LlmError> {
        if llm_mock_mode_enabled() {
            return Ok(LlmCompletion {
                content: mock_completion_for_prompt(prompt),
                usage: None,
            });
        }

        let is_openai = self.base_url.contains("openai.com");
        let req = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.temperature,
            max_tokens: if is_openai { None } else { Some(4096) },
            max_completion_tokens: if is_openai { Some(4096) } else { None },
        };

        let mut request = self
            .http
            .post(format!("{}/chat/completions", self.base_url));

        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let resp = request
            .json(&req)
            .send()
            .await?
            .error_for_status()?
            .json::<ChatResponse>()
            .await?;

        let content = resp
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::LlmResponse("empty response from LLM".into()))?;

        let usage = resp.usage.map(|u| {
            let prompt_tokens = u.prompt_tokens.unwrap_or(0) as usize;
            let completion_tokens = u.completion_tokens.unwrap_or(0) as usize;
            let total_tokens = u
                .total_tokens
                .map(|v| v as usize)
                .unwrap_or(prompt_tokens + completion_tokens);
            LlmUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }
        });

        Ok(LlmCompletion { content, usage })
    }
}
