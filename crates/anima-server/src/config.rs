use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Cross-encoder re-ranker for improved retrieval quality.
    #[serde(default)]
    pub reranker: RerankerConfig,
    /// User-defined memory categories with custom decay rates.
    /// Keys are category names, values configure the decay lambda.
    /// Built-in defaults (identity, preference, environment, routine, task, inferred, general)
    /// are always available — entries here override or extend them.
    #[serde(default)]
    pub categories: HashMap<String, CategoryConfig>,
    /// Named LLM provider profiles. Each profile is an OpenAI-compatible endpoint.
    /// When defined, `[routing]` controls which profile each operation uses.
    /// Multiple profiles can share the same base_url with different models.
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
    /// Routes operations to named profiles.
    #[serde(default)]
    pub routing: RoutingConfig,
}

/// A named LLM provider profile (OpenAI-compatible endpoint).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

impl ProfileConfig {
    /// Resolve the API key: config value > {PROFILE_NAME}_API_KEY env > OPENAI_API_KEY env.
    pub fn resolve_api_key(&self, profile_name: &str) -> Option<String> {
        if !self.api_key.is_empty() {
            return Some(self.api_key.clone());
        }
        let env_key = format!("{}_API_KEY", profile_name.to_uppercase().replace('-', "_"));
        if let Ok(key) = std::env::var(&env_key) {
            if !key.is_empty() { return Some(key); }
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            if !key.is_empty() { return Some(key); }
        }
        None
    }
}

/// Routes operations to named profiles.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RoutingConfig {
    /// Profile for /ask answer generation.
    #[serde(default)]
    pub ask: Option<String>,
    /// Profile for /chat replies.
    #[serde(default)]
    pub chat: Option<String>,
    /// Profile for background reflection/deduction.
    #[serde(default)]
    pub processor: Option<String>,
    /// Profile for memory consolidation.
    #[serde(default)]
    pub consolidation: Option<String>,
}

/// Resolved profiles + routing, computed once at startup.
#[derive(Debug, Clone)]
pub struct ResolvedProfiles {
    pub profiles: HashMap<String, ProfileConfig>,
    pub routing: RoutingConfig,
}

/// Configuration for a memory category.
#[derive(Debug, Clone, Deserialize)]
pub struct CategoryConfig {
    /// Half-life in days. After this many days, a memory's temporal score drops to 50%.
    /// Examples: 3000 (near-permanent), 29 (default), 6 (fast-expiring task).
    /// Takes priority over `lambda` if both are set.
    #[serde(default)]
    pub half_life_days: Option<f64>,
    /// Raw decay rate (advanced). Most users should use `half_life_days` instead.
    #[serde(default)]
    pub lambda: Option<f64>,
}

impl CategoryConfig {
    /// Resolve to a lambda value. half_life_days takes priority.
    pub fn resolve_lambda(&self) -> Option<f64> {
        if let Some(days) = self.half_life_days {
            if days > 0.0 {
                // lambda = ln(2) / (days * 24 hours)
                return Some(std::f64::consts::LN_2 / (days * 24.0));
            }
        }
        self.lambda
    }
}

impl AppConfig {
    /// Look up the decay lambda for a category.
    /// Priority: user config > built-in defaults > global search.temporal_lambda.
    pub fn category_lambda(&self, category: &str) -> f64 {
        // User-defined override
        if let Some(cfg) = self.categories.get(category) {
            if let Some(lambda) = cfg.resolve_lambda() {
                return lambda;
            }
        }
        // Built-in default
        if let Some(lambda) = anima_core::memory::builtin_category_lambda(category) {
            return lambda;
        }
        // Unknown category: use global lambda
        self.search.temporal_lambda
    }
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
    /// Embedding backend: "local" (default, ONNX), "openai", or "openai_compat"
    /// (any OpenAI-compatible embedding API — Cohere, Voyage, Together, vLLM, etc.).
    /// "openai" and "openai_compat" are equivalent — both use the OpenAI embedding
    /// API protocol and support custom base URLs.
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
    // --- OpenAI / OpenAI-compatible API settings (used when backend = "openai" or "openai_compat") ---
    /// API base URL (e.g. "https://api.openai.com/v1", "https://api.cohere.com/compatibility/v1",
    /// "https://api.voyageai.com/v1").
    #[serde(default)]
    pub api_base_url: Option<String>,
    /// Model name (e.g. "text-embedding-3-small", "embed-v4.0", "voyage-3-large").
    #[serde(default)]
    pub api_model: Option<String>,
    /// API key. Falls back to EMBEDDING_API_KEY env var, then OPENAI_API_KEY env var.
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

#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "telemetry_default_enabled")]
    pub enabled: bool,
    #[serde(default = "telemetry_default_endpoint")]
    pub endpoint: String,
    #[serde(default = "telemetry_default_interval_secs")]
    pub interval_secs: u64,
}

fn telemetry_default_enabled() -> bool { true }
fn telemetry_default_endpoint() -> String { "https://telemetry.anima-memory.dev/v1/report".into() }
fn telemetry_default_interval_secs() -> u64 { 86400 }

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: telemetry_default_enabled(),
            endpoint: telemetry_default_endpoint(),
            interval_secs: telemetry_default_interval_secs(),
        }
    }
}

/// Cross-encoder re-ranker configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RerankerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "reranker_default_model_dir")]
    pub model_dir: String,
    #[serde(default = "reranker_default_model_url")]
    pub model_url: String,
    #[serde(default = "reranker_default_tokenizer_url")]
    pub tokenizer_url: String,
    /// Max tokens per (query + document) pair. Default 512.
    #[serde(default = "reranker_default_max_length")]
    pub max_length: usize,
    /// Max candidate pairs to re-rank. Default 15.
    #[serde(default = "reranker_default_top_n")]
    pub top_n: usize,
}

fn reranker_default_model_dir() -> String { "./models/bge-reranker-v2-m3".into() }
fn reranker_default_model_url() -> String {
    "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/onnx/model_int8.onnx".into()
}
fn reranker_default_tokenizer_url() -> String {
    "https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/main/tokenizer.json".into()
}
fn reranker_default_max_length() -> usize { 512 }
fn reranker_default_top_n() -> usize { 10 }

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_dir: reranker_default_model_dir(),
            model_url: reranker_default_model_url(),
            tokenizer_url: reranker_default_tokenizer_url(),
            max_length: reranker_default_max_length(),
            top_n: reranker_default_top_n(),
        }
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
            telemetry: TelemetryConfig::default(),
            reranker: RerankerConfig::default(),
            categories: HashMap::new(),
            profiles: HashMap::new(),
            routing: RoutingConfig::default(),
        }
    }
}

impl AppConfig {
    /// Resolve named profiles from config. If `[profiles]` is defined, use them.
    /// Otherwise, synthesize profiles from legacy `[llm]`, `[processor]`, `[consolidation]` sections.
    pub fn resolve_profiles(&self) -> ResolvedProfiles {
        if !self.profiles.is_empty() {
            // Explicit profiles — use as-is, apply routing defaults
            let mut routing = self.routing.clone();
            let first_name = self.profiles.keys().next().cloned();
            if routing.ask.is_none() { routing.ask = first_name.clone(); }
            if routing.chat.is_none() { routing.chat = first_name.clone(); }
            if routing.processor.is_none() { routing.processor = first_name.clone(); }
            if routing.consolidation.is_none() { routing.consolidation = first_name; }

            // Warn about routing references to non-existent profiles
            for (op, name) in [
                ("ask", &routing.ask), ("chat", &routing.chat),
                ("processor", &routing.processor), ("consolidation", &routing.consolidation),
            ] {
                if let Some(n) = name {
                    if !self.profiles.contains_key(n) {
                        tracing::warn!("Routing '{op}' references unknown profile '{n}'");
                    }
                }
            }

            return ResolvedProfiles { profiles: self.profiles.clone(), routing };
        }

        // Legacy mode: synthesize profiles from [llm], [processor], [consolidation]
        let mut profiles = HashMap::new();
        let mut routing = RoutingConfig::default();

        // "default" from [llm]
        profiles.insert("default".into(), ProfileConfig {
            base_url: self.llm.base_url.clone(),
            model: self.llm.model.clone(),
            api_key: self.llm.api_key.clone(),
        });
        routing.ask = Some("default".into());
        routing.chat = Some("default".into());

        // "processor" from [processor] if it differs from [llm]
        if self.processor.enabled
            && (self.processor.base_url != self.llm.base_url || self.processor.model != self.llm.model)
        {
            profiles.insert("processor".into(), ProfileConfig {
                base_url: self.processor.base_url.clone(),
                model: self.processor.model.clone(),
                api_key: self.processor.api_key.clone(),
            });
            routing.processor = Some("processor".into());
        } else {
            routing.processor = Some("default".into());
        }

        // "consolidation" from [consolidation]
        if self.consolidation.enabled {
            match self.consolidation.backend.as_str() {
                "ollama" => {
                    // Ollama's OpenAI-compat endpoint is at /v1
                    let base = self.consolidation.ollama.base_url.trim_end_matches('/');
                    profiles.insert("consolidation".into(), ProfileConfig {
                        base_url: format!("{base}/v1"),
                        model: self.consolidation.ollama.model.clone(),
                        api_key: String::new(),
                    });
                }
                "openai_compat" => {
                    profiles.insert("consolidation".into(), ProfileConfig {
                        base_url: self.consolidation.openai_compat.base_url.clone(),
                        model: self.consolidation.openai_compat.model.clone(),
                        api_key: self.consolidation.openai_compat.api_key.clone(),
                    });
                }
                _ => {}
            }
            routing.consolidation = Some(
                if profiles.contains_key("consolidation") { "consolidation" } else { "default" }.into()
            );
        } else {
            routing.consolidation = Some("default".into());
        }

        ResolvedProfiles { profiles, routing }
    }

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
