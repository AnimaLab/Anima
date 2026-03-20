use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;
use sha2::{Digest, Sha256};
use url::Url;

use crate::app::AppState;

#[derive(Debug, Serialize)]
pub struct TelemetryPayload {
    pub schema_version: u32,
    pub installation_id: String,
    pub app_version: String,
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub timestamp: String,
    pub uptime_secs: u64,
    pub config: ConfigPayload,
    pub feature_flags: FeatureFlags,
    pub usage: UsagePayload,
    pub processor_metrics: Option<ProcessorMetricsPayload>,
}

#[derive(Debug, Serialize)]
pub struct ConfigPayload {
    pub embedding_backend: String,
    pub embedding_dimension: usize,
    pub consolidation_enabled: bool,
    pub consolidation_backend: String,
    pub llm_model: String,
    pub llm_provider_domain: String,
    pub processor_enabled: bool,
    pub processor_model: String,
    pub processor_provider_domain: String,
}

#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct FeatureFlags {
    #[serde(default)]
    pub vision: bool,
    #[serde(default = "default_true")]
    pub tool_use: bool,
    #[serde(default = "default_true")]
    pub streaming: bool,
    #[serde(default = "default_true")]
    pub reflection_enabled: bool,
    #[serde(default = "default_true")]
    pub deduction_enabled: bool,
    #[serde(default = "default_true")]
    pub induction_enabled: bool,
}

fn default_true() -> bool { true }

#[derive(Debug, Serialize)]
pub struct UsagePayload {
    pub namespace_count: usize,
    pub total_memories: u64,
    pub active_memories: u64,
    pub memories_by_type: HashMap<String, u64>,
}

#[derive(Debug, Serialize)]
pub struct ProcessorMetricsPayload {
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub reflected_created: usize,
    pub deduced_created: usize,
    pub llm_calls: usize,
    pub llm_total_tokens: usize,
}

fn extract_domain(raw_url: &str) -> String {
    Url::parse(raw_url)
        .ok()
        .and_then(|u| u.host_str().map(String::from))
        .unwrap_or_default()
}

fn home_dir() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

fn installation_id() -> String {
    let dir = home_dir().join(".anima");
    let path = dir.join("telemetry_id");

    if let Ok(existing) = std::fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            let hash = Sha256::digest(trimmed.as_bytes());
            return format!("{hash:x}");
        }
    }

    let _ = std::fs::create_dir_all(&dir);
    let id = uuid::Uuid::new_v4().to_string();
    let _ = std::fs::write(&path, &id);
    let hash = Sha256::digest(id.as_bytes());
    format!("{hash:x}")
}

fn os_version() -> String {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("PRETTY_NAME="))
                    .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string())
            })
            .unwrap_or_default()
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("OS").unwrap_or_default()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        String::new()
    }
}

pub async fn collect_payload(state: &AppState) -> TelemetryPayload {
    let cfg = &state.config;

    // Gather namespace/memory counts
    let namespaces = state.store.list_namespaces().await.unwrap_or_default();
    let namespace_count = namespaces.len();
    let total_memories: u64 = namespaces.iter().map(|n| n.total_count).sum();
    let active_memories: u64 = namespaces.iter().map(|n| n.active_count).sum();

    // We don't have per-type counts readily available across all namespaces
    // without a custom query, so we send totals only.
    let memories_by_type = HashMap::new();

    let processor_metrics = state.processor.as_ref().map(|p| {
        let m = p.metrics_snapshot();
        ProcessorMetricsPayload {
            completed_jobs: m.completed_jobs,
            failed_jobs: m.failed_jobs,
            reflected_created: m.reflected_created,
            deduced_created: m.deduced_created,
            llm_calls: m.llm_calls,
            llm_total_tokens: m.llm_total_tokens,
        }
    });

    let feature_flags = state.telemetry_feature_flags.read().await.clone();

    TelemetryPayload {
        schema_version: 1,
        installation_id: installation_id(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
        os_version: os_version(),
        arch: std::env::consts::ARCH.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_secs: state.ingested_started_at.elapsed().as_secs(),
        config: ConfigPayload {
            embedding_backend: cfg.embedding.backend.clone(),
            embedding_dimension: cfg.embedding.dimension,
            consolidation_enabled: cfg.consolidation.enabled,
            consolidation_backend: cfg.consolidation.backend.clone(),
            llm_model: cfg.llm.model.clone(),
            llm_provider_domain: extract_domain(&cfg.llm.base_url),
            processor_enabled: cfg.processor.enabled,
            processor_model: cfg.processor.model.clone(),
            processor_provider_domain: extract_domain(&cfg.processor.base_url),
        },
        feature_flags,
        usage: UsagePayload {
            namespace_count,
            total_memories,
            active_memories,
            memories_by_type,
        },
        processor_metrics,
    }
}

pub fn spawn_telemetry_loop(state: Arc<AppState>) {
    let endpoint = state.config.telemetry.endpoint.clone();
    let interval_secs = state.config.telemetry.interval_secs.max(60);

    tokio::spawn(async move {
        // Initial delay to let the server stabilize
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        loop {
            if !state.telemetry_enabled() {
                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
                continue;
            }

            let payload = collect_payload(&state).await;

            match client.post(&endpoint).json(&payload).send().await {
                Ok(resp) => {
                    tracing::debug!("Telemetry sent (status={})", resp.status());
                }
                Err(e) => {
                    tracing::debug!("Telemetry send failed: {e}");
                }
            }

            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
        }
    });

    tracing::info!("Telemetry enabled (interval={}s)", interval_secs);
}
