use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub consolidation: ConsolidationConfig,
    #[serde(default)]
    pub llm: LlmServerConfig,
    #[serde(default)]
    pub processor: ProcessorLlmConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding backend: "local" (default, ONNX) or "openai" (API-based).
    #[serde(default = "default_embedding_backend")]
    pub backend: String,
    // --- Local ONNX settings ---
    #[serde(default)]
    pub model_dir: String,
    #[serde(default)]
    pub model_url: String,
    #[serde(default)]
    pub tokenizer_url: String,
    pub dimension: usize,
    #[serde(default = "default_embedding_pooling")]
    pub pooling: String,
    /// Instruction prefix prepended to search queries (not documents).
    /// For Qwen3-Embedding: "Instruct: Given a memory query, retrieve relevant passages that answer the query\nQuery:"
    /// Set to empty string or omit to disable.
    #[serde(default)]
    pub query_instruction: Option<String>,
    // --- OpenAI API settings (used when backend = "openai") ---
    /// API base URL (e.g. "https://api.openai.com/v1").
    #[serde(default)]
    pub api_base_url: Option<String>,
    /// Model name (e.g. "text-embedding-3-small").
    #[serde(default)]
    pub api_model: Option<String>,
    /// API key. Falls back to OPENAI_API_KEY env var.
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_embedding_backend() -> String {
    "local".into()
}

fn default_embedding_pooling() -> String {
    "mean".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    pub rrf_k: f64,
    pub weight_vector: f64,
    pub weight_keyword: f64,
    pub temporal_weight: f64,
    pub temporal_lambda: f64,
    pub similarity_threshold: f64,
    #[serde(default = "default_access_weight")]
    pub access_weight: f64,
    #[serde(default = "default_importance_weight")]
    pub importance_weight: f64,
    #[serde(default = "default_tier_boost")]
    pub tier_boost: f64,
    #[serde(default = "default_min_vector_similarity")]
    pub min_vector_similarity: f64,
    #[serde(default = "default_min_score_spread")]
    pub min_score_spread: f64,
    /// Maximum memory tier to include in search (1=raw, 2=+reflected, 3=+deduced, 4=all).
    #[serde(default = "default_max_tier")]
    pub max_tier: i32,
}

fn default_access_weight() -> f64 {
    0.02
}

fn default_importance_weight() -> f64 {
    0.03
}

fn default_tier_boost() -> f64 {
    0.0
}

fn default_max_tier() -> i32 {
    4
}

fn default_min_vector_similarity() -> f64 {
    0.55
}

fn default_min_score_spread() -> f64 {
    0.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConsolidationConfig {
    pub enabled: bool,
    pub backend: String,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub openai_compat: OpenAiCompatConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "qwen3.5:4b".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiCompatConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl Default for OpenAiCompatConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.groq.com/openai/v1".into(),
            api_key: String::new(),
            model: "qwen/qwen3-32b".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmServerConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

impl Default for LlmServerConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".into(),
            model: "qwen3.5:4b".into(),
            api_key: String::new(),
        }
    }
}

/// LLM config for background processor (reflection + deduction).
/// Defaults to Qwen3-32B on Groq. Falls back to consolidation LLM if disabled.
#[derive(Debug, Clone, Deserialize)]
pub struct ProcessorLlmConfig {
    #[serde(default = "processor_default_enabled")]
    pub enabled: bool,
    #[serde(default = "processor_default_base_url")]
    pub base_url: String,
    #[serde(default = "processor_default_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

fn processor_default_enabled() -> bool { true }
fn processor_default_base_url() -> String { "https://api.groq.com/openai/v1".into() }
fn processor_default_model() -> String { "qwen/qwen3-32b".into() }

impl Default for ProcessorLlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: processor_default_base_url(),
            model: processor_default_model(),
            api_key: String::new(),
        }
    }
}

impl ProcessorLlmConfig {
    /// Resolve the API key: config value > PROCESSOR_API_KEY env > OPENAI_API_KEY env.
    pub fn resolve_api_key(&self) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        if let Ok(key) = std::env::var("PROCESSOR_API_KEY") {
            if !key.is_empty() { return Some(key); }
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            if !key.is_empty() { return Some(key); }
        }
        None
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 3000,
            },
            database: DatabaseConfig {
                path: "./anima.db".into(),
            },
            embedding: EmbeddingConfig {
                backend: "local".into(),
                model_dir: "./models/qwen3-embedding-0.6b".into(),
                model_url: "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/onnx/model_int8.onnx".into(),
                tokenizer_url: "https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/tokenizer.json".into(),
                dimension: 1024,
                pooling: "last_token".into(),
                query_instruction: None,
                api_base_url: None,
                api_model: None,
                api_key: None,
            },
            search: SearchConfig {
                rrf_k: 10.0,
                weight_vector: 0.6,
                weight_keyword: 0.4,
                temporal_weight: 0.2,
                temporal_lambda: 0.001,
                similarity_threshold: 0.85,
                access_weight: 0.02,
                importance_weight: 0.03,
                tier_boost: 0.0,
                min_vector_similarity: 0.55,
                min_score_spread: 0.055,
                max_tier: 4,
            },
            consolidation: ConsolidationConfig {
                enabled: true,
                backend: "ollama".into(),
                ollama: OllamaConfig {
                    base_url: "http://localhost:11434".into(),
                    model: "qwen3.5:4b".into(),
                },
                openai_compat: OpenAiCompatConfig {
                    base_url: "https://api.openai.com/v1".into(),
                    api_key: String::new(),
                    model: "gpt-4o-mini".into(),
                },
            },
            llm: LlmServerConfig::default(),
            processor: ProcessorLlmConfig::default(),
        }
    }
}

impl AppConfig {
    /// Load config from file, falling back to defaults.
    pub fn load(path: Option<&str>) -> Self {
        if let Some(p) = path {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(config) = toml::from_str(&content) {
                    return config;
                }
                tracing::warn!("Failed to parse config file {p}, using defaults");
            }
        }

        // Try default locations
        for candidate in &["config.toml", "config.default.toml"] {
            if let Ok(content) = std::fs::read_to_string(candidate) {
                if let Ok(config) = toml::from_str(&content) {
                    return config;
                }
            }
        }

        Self::default()
    }
}
