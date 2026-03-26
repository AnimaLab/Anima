use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use axum::extract::FromRequestParts;
use axum::http::Method;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use tower_http::services::{ServeDir, ServeFile};

use anima_consolidate::consolidator::Consolidator;
use anima_consolidate::llm_client::{LlmClient, OpenAiCompatClient};
use anima_core::namespace::Namespace;
use anima_core::search::ScorerConfig;
use anima_db::store::MemoryStore;
use anima_embed::Embedder;

use crate::config::{AppConfig, ProfileConfig, ResolvedProfiles};
use crate::dto::ErrorResponse;
use crate::handlers;
use crate::processor::BackgroundProcessor;
use crate::telemetry::FeatureFlags;

/// Shared application state.
pub struct AppState {
    pub store: MemoryStore,
    pub embedder: Arc<dyn Embedder>,
    pub consolidator: Option<Arc<Consolidator>>,
    pub processor: Option<BackgroundProcessor>,
    /// Hybrid search scorer configuration. Wrapped in RwLock so the background
    /// auto-tuner can update weights from calibration observations.
    pub scorer_config: tokio::sync::RwLock<ScorerConfig>,
    pub reranker: Option<Arc<anima_embed::reranker::Reranker>>,
    pub config: AppConfig,
    /// Resolved LLM provider profiles and operation routing.
    pub resolved_profiles: ResolvedProfiles,
    /// Ingestion tracking
    pub ingested_count: AtomicU64,
    pub ingested_started_at: std::time::Instant,
    /// Telemetry
    pub telemetry_enabled: AtomicBool,
    pub telemetry_feature_flags: tokio::sync::RwLock<FeatureFlags>,
    /// Vec index status — DimensionMismatch means re-embedding is needed.
    pub vec_status: tokio::sync::RwLock<anima_db::vector::VecTableStatus>,
}

impl AppState {
    pub fn record_ingestion(&self, count: u64) {
        self.ingested_count.fetch_add(count, Ordering::Relaxed);
    }

    pub fn ingestion_stats(&self) -> (u64, f64) {
        let count = self.ingested_count.load(Ordering::Relaxed);
        let elapsed = self.ingested_started_at.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 { count as f64 / elapsed } else { 0.0 };
        (count, rate)
    }

    pub fn telemetry_enabled(&self) -> bool {
        self.telemetry_enabled.load(Ordering::Relaxed)
    }
}

impl AppState {
    /// Get the resolved profile for a given operation.
    pub fn profile_for(&self, operation: &str) -> Option<&ProfileConfig> {
        let name = match operation {
            "ask" => self.resolved_profiles.routing.ask.as_deref(),
            "chat" => self.resolved_profiles.routing.chat.as_deref(),
            "processor" => self.resolved_profiles.routing.processor.as_deref(),
            "consolidation" => self.resolved_profiles.routing.consolidation.as_deref(),
            _ => None,
        };
        name.and_then(|n| self.resolved_profiles.profiles.get(n))
    }

    /// Get the LLM client for the processor (reflection/deduction).
    /// Priority: resolved profile → legacy [processor] config → consolidation LLM.
    pub fn get_processor_llm(&self) -> Option<Arc<dyn LlmClient>> {
        // Try resolved profile first
        if let Some(profile) = self.profile_for("processor") {
            let profile_name = self.resolved_profiles.routing.processor.as_deref().unwrap_or("processor");
            if let Some(key) = profile.resolve_api_key(profile_name) {
                return Some(Arc::new(OpenAiCompatClient::new(
                    profile.base_url.clone(),
                    Some(key),
                    profile.model.clone(),
                ).with_temperature(0.2)));
            }
            // Profile exists but no API key — if base_url is local, no key needed
            if profile.base_url.contains("localhost") || profile.base_url.contains("127.0.0.1") {
                return Some(Arc::new(OpenAiCompatClient::new(
                    profile.base_url.clone(),
                    None,
                    profile.model.clone(),
                ).with_temperature(0.2)));
            }
        }
        // Fall back to legacy [processor] config
        let cfg = &self.config.processor;
        if cfg.enabled {
            if let Some(key) = cfg.resolve_api_key() {
                return Some(Arc::new(OpenAiCompatClient::new(
                    cfg.base_url.clone(),
                    Some(key),
                    cfg.model.clone(),
                ).with_temperature(0.2)));
            }
        }
        // Fall back to consolidation LLM
        if let Some(consolidator) = &self.consolidator {
            return Some(consolidator.llm_client());
        }
        None
    }
}

/// Build the Axum router.
pub fn build_router(state: Arc<AppState>) -> Router {
    // SPA fallback: serve index.html for all non-API routes (client-side routing).
    // ServeDir.not_found_service preserves a 404 status which causes console errors,
    // so we use fallback_service directly with ServeFile which returns 200.
    let spa = ServeDir::new("web/dist").fallback(ServeFile::new("web/dist/index.html"));

    Router::new()
        .route("/health", get(handlers::health))
        .route("/health/ready", get(handlers::health_ready))
        .route("/api/v1/calibration/metrics", get(handlers::calibration_metrics))
        .route("/api/v1/calibration/weights", get(handlers::get_hybrid_weights))
        .route("/api/v1/profiles", get(handlers::get_profiles))
        .route("/api/v1/models", get(handlers::list_models))
        .route(
            "/api/v1/memories",
            get(handlers::list_memories).post(handlers::add_memory),
        )
        .route("/api/v1/memories/batch", post(handlers::add_memories_batch))
        .route(
            "/api/v1/working-memory",
            get(handlers::list_working_memory).post(handlers::add_working_memory),
        )
        .route(
            "/api/v1/working-memory/commit",
            post(handlers::commit_working_memory),
        )
        .route("/api/v1/memories/search", post(handlers::search_memories))
        .route("/api/v1/memories/merge", post(handlers::merge_memories))
        .route("/api/v1/corrections", post(handlers::capture_correction))
        .route("/api/v1/contradictions", get(handlers::list_contradictions))
        .route("/api/v1/audit/events", get(handlers::list_audit_events))
        .route("/api/v1/identities/entities", post(handlers::create_identity_entity))
        .route(
            "/api/v1/identities/entities/{id}/aliases",
            post(handlers::add_identity_alias),
        )
        .route("/api/v1/identities/resolve", get(handlers::resolve_identity))
        .route(
            "/api/v1/plans",
            get(handlers::list_plans).post(handlers::create_plan),
        )
        .route(
            "/api/v1/plans/{id}/checkpoints",
            get(handlers::list_plan_checkpoints).post(handlers::add_plan_checkpoint),
        )
        .route(
            "/api/v1/plans/checkpoints/{id}/status",
            post(handlers::update_plan_checkpoint_status),
        )
        .route("/api/v1/plans/{id}/outcome", post(handlers::set_plan_outcome))
        .route(
            "/api/v1/plans/{id}/branches",
            get(handlers::list_plan_recovery_branches).post(handlers::add_plan_recovery_branch),
        )
        .route(
            "/api/v1/plans/branches/{id}/resolve",
            post(handlers::resolve_plan_recovery_branch),
        )
        .route(
            "/api/v1/plans/{id}/procedures",
            get(handlers::list_plan_procedure_bindings).post(handlers::bind_procedure_to_plan),
        )
        .route(
            "/api/v1/transitions",
            get(handlers::list_state_transitions).post(handlers::upsert_state_transition),
        )
        .route(
            "/api/v1/counterfactuals/simulate",
            post(handlers::simulate_counterfactual),
        )
        .route(
            "/api/v1/memories/top-accessed",
            get(handlers::top_accessed),
        )
        .route("/api/v1/memories/{id}", get(handlers::get_memory))
        .route("/api/v1/memories/{id}", put(handlers::update_memory))
        .route("/api/v1/memories/{id}", delete(handlers::delete_memory))
        .route("/api/v1/memories/{id}/patch", post(handlers::patch_memory))
        .route("/api/v1/memories/{id}/rollback", post(handlers::rollback_memory))
        .route("/api/v1/memories/{id}/revisions", get(handlers::list_claim_revisions))
        .route("/api/v1/memories/{id}/history", get(handlers::get_memory_history))
        .route("/api/v1/memories/purge", post(handlers::purge_deleted_memories))
        .route("/api/v1/stats", get(handlers::get_stats))
        .route("/api/v1/namespaces", get(handlers::list_namespaces).delete(handlers::delete_namespace))
        .route("/api/v1/namespaces/rename", post(handlers::rename_namespace))
        .route("/api/v1/vec/status", get(handlers::get_vec_status))
        .route("/api/v1/vec/reindex", post(handlers::reindex_embeddings))
        .route("/api/v1/graph", get(handlers::get_graph))
        .route("/api/v1/embeddings", get(handlers::get_embeddings))
        .route("/api/v1/chat", post(handlers::chat))
        .route("/api/v1/chat/stream", post(handlers::chat_stream))
        .route("/api/v1/ask", post(handlers::ask))
        .route("/api/v1/reflect", post(handlers::reflect))
        .route("/api/v1/processor/status", get(handlers::processor_status))
        .route("/api/v1/processor/log", get(handlers::list_processor_log))
        .route(
            "/api/v1/processor/reconsolidate",
            post(handlers::reconsolidate_memories),
        )
        .route(
            "/api/v1/processor/retention/run",
            post(handlers::run_retention),
        )
        .route(
            "/api/v1/procedures/{name}/revisions",
            get(handlers::list_procedure_revisions).post(handlers::upsert_procedure_revision),
        )
        .route(
            "/api/v1/conversations",
            get(handlers::list_conversations).post(handlers::create_conversation),
        )
        .route(
            "/api/v1/conversations/{id}",
            get(handlers::get_conversation)
                .put(handlers::update_conversation)
                .delete(handlers::delete_conversation),
        )
        .route(
            "/api/v1/conversations/{id}/title",
            post(handlers::generate_title),
        )
        .route(
            "/api/v1/telemetry/config",
            get(handlers::get_telemetry_config).put(handlers::set_telemetry_config),
        )
        .route("/api/v1/backup", get(handlers::export_backup))
        .route("/api/v1/restore", post(handlers::import_backup)
            .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024)))
        .route("/api/v1/restore/sqlite", post(handlers::import_backup_sqlite)
            .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024)))
        .with_state(state)
        .fallback_service(spa)
}

/// Extract namespace from X-Anima-Namespace header.
pub struct ExtractNamespace(pub Namespace);

impl FromRequestParts<Arc<AppState>> for ExtractNamespace {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get("X-Anima-Namespace")
            .ok_or_else(|| AppError::BadRequest(
                "missing namespace header (expected X-Anima-Namespace)".into(),
            ))?
            .to_str()
            .map_err(|_| AppError::BadRequest("invalid namespace header".into()))?;

        let ns = Namespace::parse(header)
            .map_err(|e| AppError::BadRequest(format!("invalid namespace: {e}")))?;

        let principal = parts
            .headers
            .get("X-Anima-Principal")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("anonymous");
        let access_mode = AccessMode::from_method(&parts.method);
        if !namespace_acl_allows(principal, ns.as_str(), access_mode) {
            return Err(AppError::Forbidden(format!(
                "principal '{principal}' is not allowed to {} namespace '{}'",
                access_mode.as_str(),
                ns.as_str()
            )));
        }

        Ok(ExtractNamespace(ns))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessMode {
    Read,
    Write,
}

impl AccessMode {
    fn from_method(method: &Method) -> Self {
        match *method {
            Method::GET | Method::HEAD | Method::OPTIONS => Self::Read,
            _ => Self::Write,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

#[derive(Debug, Clone)]
struct AclEntry {
    pattern: String,
    can_read: bool,
    can_write: bool,
}

#[derive(Debug, Clone)]
struct AclRule {
    principal: String,
    entries: Vec<AclEntry>,
}

fn namespace_acl_allows(principal: &str, namespace: &str, mode: AccessMode) -> bool {
    let raw = std::env::var("ANIMA_NAMESPACE_ACL").unwrap_or_default();
    if raw.trim().is_empty() {
        return true;
    }
    let rules = parse_namespace_acl(&raw);
    if rules.is_empty() {
        return true;
    }

    for rule in rules {
        if rule.principal != principal && rule.principal != "*" {
            continue;
        }
        for entry in &rule.entries {
            if namespace_pattern_matches(&entry.pattern, namespace) {
                let allowed = match mode {
                    AccessMode::Read => entry.can_read,
                    AccessMode::Write => entry.can_write,
                };
                if allowed {
                    return true;
                }
            }
        }
    }
    false
}

fn parse_namespace_acl(raw: &str) -> Vec<AclRule> {
    let mut rules = Vec::new();
    for part in raw.split(';') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((principal, entries_raw)) = trimmed.split_once('=') else {
            continue;
        };
        let principal = principal.trim();
        if principal.is_empty() {
            continue;
        }
        let mut entries = Vec::new();
        for item in entries_raw.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let (pattern_raw, mode_raw) = if let Some((pattern, mode)) = item.split_once(':') {
                (pattern.trim(), mode.trim().to_ascii_lowercase())
            } else {
                (item, "rw".to_string())
            };
            if pattern_raw.is_empty() {
                continue;
            }
            let (can_read, can_write) = match mode_raw.as_str() {
                "r" => (true, false),
                "w" => (false, true),
                "rw" | "wr" | "" => (true, true),
                _ => (true, true),
            };
            entries.push(AclEntry {
                pattern: pattern_raw.to_string(),
                can_read,
                can_write,
            });
        }
        if !entries.is_empty() {
            rules.push(AclRule {
                principal: principal.to_string(),
                entries,
            });
        }
    }
    rules
}

fn namespace_pattern_matches(pattern: &str, namespace: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("/*") {
        return namespace == prefix || namespace.starts_with(&format!("{prefix}/"));
    }
    pattern == namespace
}

/// Unified error type for API responses.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    BadRequest(String),

    #[error("{0}")]
    Forbidden(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    /// Return the error domain for structured logging.
    fn domain(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "retrieval",
            Self::BadRequest(_) => "request",
            Self::Forbidden(_) => "auth",
            Self::Database(_) => "db",
            Self::Embedding(_) => "embedding",
            Self::Internal(_) => "internal",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            Self::Forbidden(msg) => (StatusCode::FORBIDDEN, msg.clone()),
            Self::Database(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Self::Embedding(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        // Structured log with error domain for filtering
        match status {
            StatusCode::INTERNAL_SERVER_ERROR => {
                tracing::error!(domain = self.domain(), "{message}");
            }
            StatusCode::FORBIDDEN => {
                tracing::warn!(domain = self.domain(), "{message}");
            }
            _ => {
                tracing::debug!(domain = self.domain(), "{message}");
            }
        }

        let body = Json(ErrorResponse { error: message });
        (status, body).into_response()
    }
}

impl From<anima_db::pool::DbError> for AppError {
    fn from(e: anima_db::pool::DbError) -> Self {
        Self::Database(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn acl_disabled_is_permissive() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("ANIMA_NAMESPACE_ACL");
        assert!(namespace_acl_allows("anonymous", "default", AccessMode::Read));
        assert!(namespace_acl_allows("anonymous", "default/private", AccessMode::Write));
    }

    #[test]
    fn acl_respects_principal_namespace_and_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(
            "ANIMA_NAMESPACE_ACL",
            "alice=default/*:rw,private/*:r;bob=default:r;*=public/*:r",
        );

        assert!(namespace_acl_allows("alice", "default", AccessMode::Read));
        assert!(namespace_acl_allows("alice", "default/team", AccessMode::Write));
        assert!(namespace_acl_allows("alice", "private/reports", AccessMode::Read));
        assert!(!namespace_acl_allows("alice", "private/reports", AccessMode::Write));

        assert!(namespace_acl_allows("bob", "default", AccessMode::Read));
        assert!(!namespace_acl_allows("bob", "default", AccessMode::Write));
        assert!(!namespace_acl_allows("bob", "private/reports", AccessMode::Read));

        assert!(namespace_acl_allows("charlie", "public/docs", AccessMode::Read));
        assert!(!namespace_acl_allows("charlie", "public/docs", AccessMode::Write));

        std::env::remove_var("ANIMA_NAMESPACE_ACL");
    }
}
