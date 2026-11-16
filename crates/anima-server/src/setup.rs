//! Model setup from the web UI: the first-run dialog and Settings > Models.
//!
//! `GET /api/v1/setup` reports the models in use, `POST /api/v1/setup/models`
//! lists the models an OpenAI-compatible server offers, and `PUT /api/v1/setup`
//! writes the choice into `config.toml` (keeping its comments and every other
//! key) and restarts the server so it takes effect.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use toml_edit::{DocumentMut, Item, Table, value};

use crate::app::{AppError, AppState};
use crate::config::AppConfig;

/// Shipped template, the base of a new `config.toml` when none exists.
const DEFAULT_CONFIG: &str = include_str!("../../../config.default.toml");

/// BGE-M3, the default local embedder (as in `config.default.toml`).
const LOCAL_EMBEDDING: [(&str, &str); 4] = [
    ("model_dir", "./models/bge-m3"),
    ("model_url", "https://huggingface.co/aapot/bge-m3-onnx/resolve/main/model.onnx"),
    ("tokenizer_url", "https://huggingface.co/aapot/bge-m3-onnx/resolve/main/tokenizer.json"),
    ("pooling", "mean"),
];
const LOCAL_EMBEDDING_DIMENSION: i64 = 1024;
const REMOTE: &str = "openai_compat";

/// Where the server's configuration lives, and whether the user made it.
#[derive(Debug, Clone)]
pub struct ConfigSource {
    /// File the running config was read from; `None` for built-in defaults.
    pub loaded_from: Option<PathBuf>,
}

impl ConfigSource {
    /// A server running on the shipped template or the built-in defaults has
    /// not been set up yet.
    pub fn configured(&self) -> bool {
        self.loaded_from
            .as_deref()
            .is_some_and(|p| p.file_name().is_some_and(|n| n != "config.default.toml"))
    }

    /// The file a save writes: the loaded file, or `config.toml` next to the
    /// template (or in the working directory).
    pub fn target(&self) -> PathBuf {
        match &self.loaded_from {
            Some(p) if self.configured() => p.clone(),
            Some(p) => p.with_file_name("config.toml"),
            None => PathBuf::from("config.toml"),
        }
    }
}

// ---------------------------------------------------------------- GET

#[derive(Debug, Serialize)]
pub struct SetupStatus {
    pub configured: bool,
    pub config_file: String,
    /// The server restarts itself after a save.
    pub restart_supported: bool,
    /// Active memories, embedded with the current model.
    pub memories: u64,
    pub embedding: EmbeddingStatus,
    pub llm: LlmStatus,
    pub reranker: RerankerStatus,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingStatus {
    /// `local` or `openai_compat`.
    pub backend: String,
    pub base_url: Option<String>,
    pub model: String,
    pub dimension: usize,
    pub has_key: bool,
}

#[derive(Debug, Serialize)]
pub struct LlmStatus {
    pub base_url: String,
    pub model: String,
    pub has_key: bool,
    /// `[profiles]` in the config file decides the LLMs; the form edits `[llm]` only.
    pub from_profiles: bool,
}

#[derive(Debug, Serialize)]
pub struct RerankerStatus {
    pub enabled: bool,
    pub backend: String,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub has_key: bool,
}

fn env_key(names: &[&str]) -> bool {
    names.iter().any(|n| std::env::var(n).is_ok_and(|v| !v.trim().is_empty()))
}

fn is_remote(backend: &str) -> bool {
    matches!(backend, "openai" | "openai_compat")
}

fn some_key(k: &Option<String>) -> bool {
    k.as_deref().is_some_and(|k| !k.trim().is_empty())
}

pub fn status(config: &AppConfig, source: &ConfigSource, memories: u64) -> SetupStatus {
    let e = &config.embedding;
    let embedding = if is_remote(&e.backend) {
        EmbeddingStatus {
            backend: REMOTE.into(),
            base_url: e.api_base_url.clone(),
            model: e.api_model.clone().unwrap_or_default(),
            dimension: e.dimension,
            has_key: some_key(&e.api_key) || env_key(&["EMBEDDING_API_KEY", "OPENAI_API_KEY"]),
        }
    } else {
        EmbeddingStatus {
            backend: "local".into(),
            base_url: None,
            model: Path::new(&e.model_dir)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| e.model_dir.clone()),
            dimension: e.dimension,
            has_key: false,
        }
    };
    let r = &config.reranker;
    let remote_rerank = is_remote(&r.backend);
    SetupStatus {
        configured: source.configured(),
        config_file: source.target().display().to_string(),
        restart_supported: cfg!(unix) || cfg!(windows),
        memories,
        embedding,
        llm: LlmStatus {
            base_url: config.llm.base_url.clone(),
            model: config.llm.model.clone(),
            has_key: !config.llm.api_key.trim().is_empty() || env_key(&["OPENAI_API_KEY"]),
            from_profiles: !config.profiles.is_empty(),
        },
        reranker: RerankerStatus {
            enabled: r.enabled,
            backend: if remote_rerank { REMOTE.into() } else { "local".into() },
            base_url: remote_rerank.then(|| r.api_base_url.clone()).flatten(),
            model: if remote_rerank { r.api_model.clone() } else { Some("bge-reranker-v2-m3".into()) },
            has_key: remote_rerank
                && (some_key(&r.api_key) || env_key(&["RERANKER_API_KEY", "OPENAI_API_KEY"])),
        },
    }
}

/// GET /api/v1/setup
pub async fn get_setup(State(state): State<Arc<AppState>>) -> Result<Json<SetupStatus>, AppError> {
    let memories = state
        .store
        .list_namespaces()
        .await?
        .iter()
        .map(|n| n.active_count as u64)
        .sum();
    Ok(Json(status(&state.config, &state.config_source, memories)))
}

// ---------------------------------------------------------------- model list

#[derive(Debug, Deserialize)]
pub struct ModelsRequest {
    pub base_url: String,
    /// Empty: use a key already saved for this server, or the environment.
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub models: Vec<String>,
}

/// A key already known for `base_url`: from the config, then the environment.
fn saved_key(config: &AppConfig, base_url: &str) -> Option<String> {
    let same = |u: &str| u.trim_end_matches('/') == base_url.trim_end_matches('/');
    let e = &config.embedding;
    let r = &config.reranker;
    let mut candidates: Vec<Option<String>> = Vec::new();
    if same(&config.llm.base_url) {
        candidates.push(Some(config.llm.api_key.clone()));
    }
    if e.api_base_url.as_deref().is_some_and(same) {
        candidates.push(e.api_key.clone());
        candidates.push(std::env::var("EMBEDDING_API_KEY").ok());
    }
    if r.api_base_url.as_deref().is_some_and(same) {
        candidates.push(r.api_key.clone());
        candidates.push(std::env::var("RERANKER_API_KEY").ok());
    }
    candidates.push(std::env::var("OPENAI_API_KEY").ok());
    candidates.into_iter().flatten().map(|k| k.trim().to_string()).find(|k| !k.is_empty())
}

fn pick_key(config: &AppConfig, base_url: &str, given: &Option<String>) -> Option<String> {
    given
        .as_deref()
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string)
        .or_else(|| saved_key(config, base_url))
}

fn check_base_url(base_url: &str) -> Result<String, AppError> {
    let trimmed = base_url.trim().trim_end_matches('/').to_string();
    let parsed = url::Url::parse(&trimmed)
        .map_err(|_| AppError::BadRequest(format!("'{base_url}' is not a URL")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(AppError::BadRequest("the server URL must start with http:// or https://".into()));
    }
    Ok(trimmed)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_default()
}

/// POST /api/v1/setup/models — the models an OpenAI-compatible server lists.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelsRequest>,
) -> Result<Json<ModelsResponse>, AppError> {
    let base = check_base_url(&req.base_url)?;
    let mut call = client().get(format!("{base}/models"));
    if let Some(key) = pick_key(&state.config, &base, &req.api_key) {
        call = call.bearer_auth(key);
    }
    let resp = call
        .send()
        .await
        .map_err(|e| AppError::BadRequest(format!("could not reach {base}: {e}")))?;
    if !resp.status().is_success() {
        return Err(AppError::BadRequest(format!("{base}/models answered {}", resp.status())));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AppError::BadRequest(format!("{base}/models is not a model list: {e}")))?;
    let mut models: Vec<String> = body["data"]
        .as_array()
        .or_else(|| body["models"].as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|m| m["id"].as_str().or_else(|| m["name"].as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    models.sort();
    models.dedup();
    Ok(Json(ModelsResponse { models }))
}

// ---------------------------------------------------------------- PUT

#[derive(Debug, Deserialize)]
pub struct SetupRequest {
    pub embedding: EmbeddingChoice,
    #[serde(default)]
    pub llm: Option<LlmChoice>,
    pub reranker: RerankerChoice,
    /// Restart the server once the file is written (default true).
    #[serde(default = "yes")]
    pub restart: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingChoice {
    /// `local` or `openai_compat`.
    pub backend: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// Empty or absent: keep the saved key.
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LlmChoice {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RerankerChoice {
    pub enabled: bool,
    /// `local` or `openai_compat`.
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SetupSaved {
    pub config_file: String,
    pub restarting: bool,
}

/// What a save writes, once remote models are checked.
#[derive(Debug, Clone, PartialEq)]
pub enum Embedding {
    Local,
    Remote { base_url: String, model: String, dimension: usize, api_key: Option<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Llm {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reranker {
    Off,
    Local,
    Remote { base_url: String, model: String, api_key: Option<String> },
}

fn table<'a>(doc: &'a mut DocumentMut, path: &[&str]) -> &'a mut Table {
    let mut t = doc.as_table_mut();
    for key in path {
        let item = t.entry(key).or_insert_with(|| Item::Table(Table::new()));
        if !item.is_table() {
            *item = Item::Table(Table::new());
        }
        t = item.as_table_mut().expect("just made a table");
    }
    t
}

/// Set `api_key` when a new one is given; an empty one keeps the saved key.
fn set_key(t: &mut Table, new_key: &Option<String>) {
    if let Some(k) = new_key.as_deref().map(str::trim).filter(|k| !k.is_empty()) {
        t["api_key"] = value(k);
    }
}

/// Write the choice into the config text. Keeps comments, order and every
/// key the choice does not cover.
pub fn apply(current: &str, embedding: &Embedding, llm: Option<&Llm>, reranker: &Reranker) -> Result<String, String> {
    let mut doc: DocumentMut = current.parse().map_err(|e| format!("the config file does not parse: {e}"))?;

    {
        let was_remote = doc
            .get("embedding")
            .and_then(|e| e.get("backend"))
            .and_then(|b| b.as_str())
            .is_some_and(is_remote);
        let old_model = doc
            .get("embedding")
            .and_then(|e| e.get("api_model"))
            .and_then(|m| m.as_str())
            .map(str::to_string);
        let e = table(&mut doc, &["embedding"]);
        match embedding {
            Embedding::Local => {
                if was_remote || e.get("model_dir").is_none() {
                    for (k, v) in LOCAL_EMBEDDING {
                        e[k] = value(v);
                    }
                    e["dimension"] = value(LOCAL_EMBEDDING_DIMENSION);
                }
                e["backend"] = value("local");
                for k in ["api_base_url", "api_model", "request_dimensions", "query_instruction"] {
                    if was_remote {
                        e.remove(k);
                    }
                }
            }
            Embedding::Remote { base_url, model, dimension, api_key } => {
                e["backend"] = value(REMOTE);
                e["api_base_url"] = value(base_url.as_str());
                e["api_model"] = value(model.as_str());
                e["dimension"] = value(*dimension as i64);
                // The dimension was read from the model itself: no need to ask for it.
                e["request_dimensions"] = value(false);
                if old_model.as_deref() != Some(model.as_str()) {
                    // An instruction belongs to the model it was written for.
                    e.remove("query_instruction");
                }
                set_key(e, api_key);
            }
        }
        // Sparse vectors come from the local BGE-M3 model.
        table(&mut doc, &["sparse"])["enabled"] = value(matches!(embedding, Embedding::Local));
    }

    if let Some(llm) = llm {
        for section in [&["llm"][..], &["processor"][..], &["consolidation", "openai_compat"][..]] {
            let t = table(&mut doc, section);
            t["base_url"] = value(llm.base_url.as_str());
            t["model"] = value(llm.model.as_str());
            set_key(t, &llm.api_key);
            if t.get("api_key").is_none() {
                t["api_key"] = value("");
            }
        }
        let c = table(&mut doc, &["consolidation"]);
        c["backend"] = value(REMOTE);
        if c.get("enabled").is_none() {
            c["enabled"] = value(true);
        }
    }

    {
        let r = table(&mut doc, &["reranker"]);
        match reranker {
            Reranker::Off => r["enabled"] = value(false),
            Reranker::Local => {
                r["enabled"] = value(true);
                r["backend"] = value("local");
                r.remove("api_base_url");
                r.remove("api_model");
            }
            Reranker::Remote { base_url, model, api_key } => {
                r["enabled"] = value(true);
                r["backend"] = value(REMOTE);
                r["api_base_url"] = value(base_url.as_str());
                r["api_model"] = value(model.as_str());
                set_key(r, api_key);
            }
        }
    }

    let out = doc.to_string();
    toml::from_str::<AppConfig>(&out).map_err(|e| format!("the new config would not load: {e}"))?;
    Ok(out)
}

/// Embed one string to check the model answers, and learn its dimension.
async fn probe_embedding(base_url: &str, model: &str, key: Option<&str>) -> Result<usize, AppError> {
    let mut call = client()
        .post(format!("{base_url}/embeddings"))
        .json(&serde_json::json!({ "model": model, "input": ["Anima setup check"] }));
    if let Some(k) = key {
        call = call.bearer_auth(k);
    }
    let resp = call
        .send()
        .await
        .map_err(|e| AppError::BadRequest(format!("could not reach {base_url}: {e}")))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let detail = body["error"]["message"].as_str().or_else(|| body["detail"].as_str()).unwrap_or("");
        return Err(AppError::BadRequest(format!(
            "the embedding model '{model}' answered {status}{}{detail}",
            if detail.is_empty() { "" } else { ": " }
        )));
    }
    body["data"][0]["embedding"]
        .as_array()
        .map(|v| v.len())
        .filter(|n| *n > 0)
        .ok_or_else(|| AppError::BadRequest(format!("'{model}' did not return an embedding")))
}

fn required(v: &Option<String>, what: &str) -> Result<String, AppError> {
    v.as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| AppError::BadRequest(format!("{what} is required")))
}

/// PUT /api/v1/setup
pub async fn put_setup(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetupRequest>,
) -> Result<Json<SetupSaved>, AppError> {
    let config = &state.config;

    let embedding = match req.embedding.backend.as_str() {
        "local" => Embedding::Local,
        b if is_remote(b) => {
            let base_url = check_base_url(&required(&req.embedding.base_url, "the embedding server URL")?)?;
            let model = required(&req.embedding.model, "the embedding model")?;
            let key = pick_key(config, &base_url, &req.embedding.api_key);
            let dimension = probe_embedding(&base_url, &model, key.as_deref()).await?;
            Embedding::Remote { base_url, model, dimension, api_key: req.embedding.api_key.clone() }
        }
        other => return Err(AppError::BadRequest(format!("unknown embedding backend '{other}'"))),
    };

    let llm = match &req.llm {
        Some(l) => {
            let model = l.model.trim();
            if model.is_empty() {
                return Err(AppError::BadRequest("the chat model is required".into()));
            }
            Some(Llm {
                base_url: check_base_url(&l.base_url)?,
                model: model.to_string(),
                api_key: l.api_key.clone(),
            })
        }
        None => None,
    };

    let reranker = if !req.reranker.enabled {
        Reranker::Off
    } else {
        match req.reranker.backend.as_deref().unwrap_or("local") {
            "local" => Reranker::Local,
            b if is_remote(b) => Reranker::Remote {
                base_url: check_base_url(&required(&req.reranker.base_url, "the re-ranker server URL")?)?,
                model: required(&req.reranker.model, "the re-ranker model")?,
                api_key: req.reranker.api_key.clone(),
            },
            other => return Err(AppError::BadRequest(format!("unknown re-ranker backend '{other}'"))),
        }
    };

    let target = state.config_source.target();
    let current = match &state.config_source.loaded_from {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| AppError::Internal(format!("could not read {}: {e}", p.display())))?,
        None => DEFAULT_CONFIG.to_string(),
    };
    let updated = apply(&current, &embedding, llm.as_ref(), &reranker).map_err(AppError::BadRequest)?;
    write_private(&target, &updated)
        .map_err(|e| AppError::Internal(format!("could not write {}: {e}", target.display())))?;
    tracing::info!(domain = "setup", "Saved the model setup to {}", target.display());

    if req.restart {
        let state = state.clone();
        tokio::spawn(async move {
            // Let the response reach the browser first.
            tokio::time::sleep(Duration::from_millis(300)).await;
            state.request_restart();
        });
    }
    Ok(Json(SetupSaved { config_file: target.display().to_string(), restarting: req.restart }))
}

/// Write a file readable by its owner only: it may hold API keys.
fn write_private(path: &Path, content: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)
}

/// Start this binary again with the same arguments, after the server stopped.
/// On Unix the process image is replaced, so the PID stays (service managers
/// and scripts keep tracking it).
pub fn relaunch() -> std::io::Error {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return e,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    tracing::info!(domain = "setup", "Restarting with the new configuration");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        std::process::Command::new(exe).args(args).exec()
    }
    #[cfg(not(unix))]
    {
        match std::process::Command::new(exe).args(args).spawn() {
            Ok(_) => std::process::exit(0),
            Err(e) => e,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> AppConfig {
        toml::from_str(text).unwrap()
    }

    fn remote_llm() -> Llm {
        Llm { base_url: "https://llm.example/v1".into(), model: "chat-1".into(), api_key: Some("sk-chat".into()) }
    }

    #[test]
    fn a_remote_setup_keeps_comments_and_other_keys() {
        let embedding = Embedding::Remote {
            base_url: "https://llm.example/v1".into(),
            model: "embed-1".into(),
            dimension: 2560,
            api_key: None,
        };
        let reranker = Reranker::Remote {
            base_url: "https://llm.example/v1".into(),
            model: "rerank-1".into(),
            api_key: Some(" sk-r ".into()),
        };
        let out = apply(DEFAULT_CONFIG, &embedding, Some(&remote_llm()), &reranker).unwrap();
        let c = load(&out);
        assert_eq!(c.embedding.backend, "openai_compat");
        assert_eq!(c.embedding.api_model.as_deref(), Some("embed-1"));
        assert_eq!(c.embedding.dimension, 2560);
        assert!(!c.embedding.request_dimensions);
        assert!(c.embedding.api_key.is_none(), "an empty key keeps the saved one (none here)");
        assert!(!c.sparse.enabled, "sparse vectors need the local BGE-M3");
        assert_eq!((c.llm.base_url.as_str(), c.llm.model.as_str()), ("https://llm.example/v1", "chat-1"));
        assert_eq!(c.processor.model, "chat-1");
        assert_eq!(c.consolidation.backend, "openai_compat");
        assert_eq!(c.consolidation.openai_compat.api_key, "sk-chat");
        assert_eq!(c.reranker.api_key.as_deref(), Some("sk-r"));
        assert_eq!(c.server.port, 3000);
        // The template's comments are still there.
        let first_comment = DEFAULT_CONFIG.lines().find(|l| l.starts_with('#')).unwrap();
        assert!(out.contains(first_comment));
    }

    #[test]
    fn going_back_to_local_restores_bge_m3_and_drops_remote_keys() {
        let remote = r#"
[server]
host = "127.0.0.1"
port = 3000
[database]
path = "./anima.db"
[embedding]
backend = "openai_compat"
api_base_url = "https://llm.example/v1"
api_model = "embed-1"
dimension = 2560
query_instruction = "Instruct: x"
[search]
rrf_k = 60.0
weight_vector = 0.6
weight_keyword = 0.4
temporal_weight = 0.2
temporal_lambda = 0.001
similarity_threshold = 0.85
[consolidation]
enabled = true
backend = "openai_compat"
[consolidation.openai_compat]
base_url = "https://llm.example/v1"
api_key = ""
model = "chat-1"
"#;
        let out = apply(remote, &Embedding::Local, None, &Reranker::Local).unwrap();
        let c = load(&out);
        assert_eq!(c.embedding.backend, "local");
        assert_eq!(c.embedding.model_dir, "./models/bge-m3");
        assert_eq!(c.embedding.dimension, 1024);
        assert!(c.embedding.api_base_url.is_none() && c.embedding.query_instruction.is_none());
        assert!(c.sparse.enabled);
        assert!(c.reranker.enabled && c.reranker.backend == "local");
        // No LLM in the request: its sections are left as they were.
        assert_eq!(c.consolidation.openai_compat.model, "chat-1");
    }

    #[test]
    fn the_instruction_stays_with_its_model() {
        let base = r#"
[server]
host = "127.0.0.1"
port = 3000
[database]
path = "./anima.db"
[embedding]
backend = "openai_compat"
api_base_url = "https://llm.example/v1"
api_model = "embed-1"
api_key = "sk-saved"
dimension = 2560
query_instruction = "Instruct: x"
[search]
rrf_k = 60.0
weight_vector = 0.6
weight_keyword = 0.4
temporal_weight = 0.2
temporal_lambda = 0.001
similarity_threshold = 0.85
[consolidation]
enabled = false
backend = "ollama"
"#;
        let same = Embedding::Remote {
            base_url: "https://llm.example/v1".into(),
            model: "embed-1".into(),
            dimension: 2560,
            api_key: Some(String::new()),
        };
        let c = load(&apply(base, &same, None, &Reranker::Off).unwrap());
        assert_eq!(c.embedding.query_instruction.as_deref(), Some("Instruct: x"));
        assert_eq!(c.embedding.api_key.as_deref(), Some("sk-saved"), "an empty key keeps the saved one");
        assert!(!c.reranker.enabled);

        let other = Embedding::Remote {
            base_url: "https://llm.example/v1".into(),
            model: "embed-2".into(),
            dimension: 1024,
            api_key: None,
        };
        let c = load(&apply(base, &other, None, &Reranker::Off).unwrap());
        assert!(c.embedding.query_instruction.is_none());
        assert_eq!(c.embedding.dimension, 1024);
    }

    #[test]
    fn a_config_that_would_not_load_is_refused() {
        assert!(apply("[server]\nport = 3000\n", &Embedding::Local, None, &Reranker::Off).is_err());
        assert!(apply("not = [toml", &Embedding::Local, None, &Reranker::Off).is_err());
    }

    #[test]
    fn only_a_users_own_file_counts_as_configured() {
        let src = |p: Option<&str>| ConfigSource { loaded_from: p.map(PathBuf::from) };
        assert!(!src(None).configured());
        assert_eq!(src(None).target(), PathBuf::from("config.toml"));
        assert!(!src(Some("config.default.toml")).configured());
        assert_eq!(src(Some("/opt/anima/config.default.toml")).target(), PathBuf::from("/opt/anima/config.toml"));
        assert!(src(Some("config.toml")).configured());
        assert_eq!(src(Some("/etc/anima/custom.toml")).target(), PathBuf::from("/etc/anima/custom.toml"));
    }

    #[test]
    fn a_saved_key_is_found_by_server_url() {
        let mut c = AppConfig::default();
        c.llm.base_url = "https://llm.example/v1/".into();
        c.llm.api_key = "sk-llm".into();
        assert_eq!(saved_key(&c, "https://llm.example/v1").as_deref(), Some("sk-llm"));
        assert_eq!(pick_key(&c, "https://llm.example/v1", &Some(" sk-new ".into())).as_deref(), Some("sk-new"));
    }
}
