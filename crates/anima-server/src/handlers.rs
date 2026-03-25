use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use chrono::{Duration, Utc};
use ndarray::{Array1, Array2, Axis};

use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, Sse};
use axum::Json;
use futures::stream::Stream;
use sha2::{Digest, Sha256};
use tokio_stream::StreamExt;

use anima_consolidate::actions::ConsolidationActionType;
use anima_core::memory::{content_hash, Memory};
use anima_core::search::SearchMode;
use anima_db::store::{MemoryPatch, PredictionKind};
use tracing::warn;

use crate::app::{AppError, AppState, ExtractNamespace};
use crate::dto::*;
use crate::telemetry::FeatureFlags;

/// Liveness probe — is the process alive and responding?
/// Served at `/health` (primary) and `/health/live` (explicit).
pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "anima",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Readiness probe — are all dependencies healthy?
pub async fn health_ready(
    State(state): State<Arc<AppState>>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let mut checks = serde_json::Map::new();
    let mut all_ok = true;

    // DB check
    match state.store.ping().await {
        Ok(()) => { checks.insert("db".into(), "ok".into()); }
        Err(e) => {
            all_ok = false;
            checks.insert("db".into(), serde_json::json!({"error": e.to_string()}));
            tracing::warn!(domain = "db", "Readiness check failed — database: {e}");
        }
    }

    // Embedder check — run a trivial embedding
    match state.embedder.embed("health check") {
        Ok(_) => { checks.insert("embedder".into(), "ok".into()); }
        Err(e) => {
            all_ok = false;
            checks.insert("embedder".into(), serde_json::json!({"error": e.to_string()}));
            tracing::warn!(domain = "embedding", "Readiness check failed — embedder: {e}");
        }
    }

    // Processor check
    match &state.processor {
        Some(p) => {
            checks.insert("processor".into(), serde_json::json!({
                "status": "running",
                "queue_depth": p.queue_depth(),
                "in_flight": p.in_flight(),
            }));
        }
        None => {
            checks.insert("processor".into(), "disabled".into());
        }
    }

    let status = if all_ok { "ready" } else { "degraded" };
    let http_status = if all_ok {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };

    (http_status, Json(serde_json::json!({
        "status": status,
        "service": "anima",
        "version": env!("CARGO_PKG_VERSION"),
        "checks": checks,
    })))
}

pub async fn processor_status(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let (ingested_count, ingested_rate) = state.ingestion_stats();
    match &state.processor {
        Some(p) => {
            let queue_depth = p.queue_depth();
            let in_flight = p.in_flight();
            let dead_letter_count = p.dead_letter_count();
            let m = p.metrics_snapshot();
            Json(serde_json::json!({
                "enabled": true,
                "queue_depth": queue_depth,
                "in_flight": in_flight,
                "dead_letter_count": dead_letter_count,
                "idle": queue_depth == 0 && in_flight == 0,
                "ingestion": {
                    "total": ingested_count,
                    "rate_per_sec": (ingested_rate * 100.0).round() / 100.0,
                    "uptime_secs": state.ingested_started_at.elapsed().as_secs(),
                },
                "metrics": {
                    "completed_jobs": m.completed_jobs,
                    "failed_jobs": m.failed_jobs,
                    "reflected_created": m.reflected_created,
                    "deduced_created": m.deduced_created,
                    "committed_evaluated": m.committed_evaluated,
                    "committed_written": m.committed_written,
                    "recon_processed": m.recon_processed,
                    "recon_superseded": m.recon_superseded,
                    "retention_processed": m.retention_processed,
                    "retention_softened": m.retention_softened,
                    "llm_calls": m.llm_calls,
                    "llm_usage_missing": m.llm_usage_missing,
                    "llm_prompt_tokens": m.llm_prompt_tokens,
                    "llm_completion_tokens": m.llm_completion_tokens,
                    "llm_total_tokens": m.llm_total_tokens,
                    "reflection_llm_calls": m.reflection_llm_calls,
                    "reflection_usage_missing": m.reflection_usage_missing,
                    "reflection_prompt_tokens": m.reflection_prompt_tokens,
                    "reflection_completion_tokens": m.reflection_completion_tokens,
                    "reflection_total_tokens": m.reflection_total_tokens,
                    "deduction_llm_calls": m.deduction_llm_calls,
                    "deduction_usage_missing": m.deduction_usage_missing,
                    "deduction_prompt_tokens": m.deduction_prompt_tokens,
                    "deduction_completion_tokens": m.deduction_completion_tokens,
                    "deduction_total_tokens": m.deduction_total_tokens,
                    "induction_llm_calls": m.induction_llm_calls,
                    "induction_usage_missing": m.induction_usage_missing,
                    "induction_prompt_tokens": m.induction_prompt_tokens,
                    "induction_completion_tokens": m.induction_completion_tokens,
                    "induction_total_tokens": m.induction_total_tokens,
                },
            }))
        }
        None => Json(serde_json::json!({
            "enabled": false,
            "queue_depth": 0,
            "in_flight": 0,
            "dead_letter_count": 0,
            "idle": true,
            "ingestion": {
                "total": ingested_count,
                "rate_per_sec": (ingested_rate * 100.0).round() / 100.0,
                "uptime_secs": state.ingested_started_at.elapsed().as_secs(),
            },
            "metrics": {
                "completed_jobs": 0,
                "failed_jobs": 0,
                "reflected_created": 0,
                "deduced_created": 0,
                "committed_evaluated": 0,
                "committed_written": 0,
                "recon_processed": 0,
                "recon_superseded": 0,
                "retention_processed": 0,
                "retention_softened": 0,
                "llm_calls": 0,
                "llm_usage_missing": 0,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_total_tokens": 0,
                "reflection_llm_calls": 0,
                "reflection_usage_missing": 0,
                "reflection_prompt_tokens": 0,
                "reflection_completion_tokens": 0,
                "reflection_total_tokens": 0,
                "deduction_llm_calls": 0,
                "deduction_usage_missing": 0,
                "deduction_prompt_tokens": 0,
                "deduction_completion_tokens": 0,
                "deduction_total_tokens": 0,
                "induction_llm_calls": 0,
                "induction_usage_missing": 0,
                "induction_prompt_tokens": 0,
                "induction_completion_tokens": 0,
                "induction_total_tokens": 0
            },
        })),
    }
}

pub async fn calibration_metrics(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
) -> Result<Json<anima_db::store::CalibrationMetrics>, AppError> {
    let metrics = state.store.calibration_metrics(&ns).await?;
    Ok(Json(metrics))
}

/// GET /api/v1/calibration/weights — return current hybrid search weights
/// (possibly auto-tuned from calibration observations).
pub async fn get_hybrid_weights(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let cfg = state.scorer_config.read().await;
    Json(serde_json::json!({
        "weight_vector": cfg.weight_vector,
        "weight_keyword": cfg.weight_keyword,
    }))
}

/// GET /api/v1/profiles — return resolved LLM provider profiles and routing (masks API keys).
pub async fn get_profiles(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let profiles: serde_json::Map<String, serde_json::Value> = state.resolved_profiles.profiles.iter()
        .map(|(name, p)| {
            (name.clone(), serde_json::json!({
                "base_url": p.base_url,
                "model": p.model,
                "has_api_key": !p.api_key.is_empty() || p.resolve_api_key(name).is_some(),
            }))
        })
        .collect();
    Json(serde_json::json!({
        "profiles": profiles,
        "routing": {
            "ask": state.resolved_profiles.routing.ask,
            "chat": state.resolved_profiles.routing.chat,
            "processor": state.resolved_profiles.routing.processor,
            "consolidation": state.resolved_profiles.routing.consolidation,
        }
    }))
}

fn to_working_memory_dto(entry: anima_db::store::WorkingMemoryEntry) -> WorkingMemoryEntryDto {
    WorkingMemoryEntryDto {
        id: entry.id,
        namespace: entry.namespace,
        content: entry.content,
        metadata: entry.metadata,
        provisional_score: entry.provisional_score,
        status: entry.status,
        conversation_id: entry.conversation_id,
        expires_at: entry.expires_at,
        committed_memory_id: entry.committed_memory_id,
        created_at: entry.created_at,
        updated_at: entry.updated_at,
    }
}

struct QueryEmbeddingCache {
    map: HashMap<String, Vec<f32>>,
    fifo: VecDeque<String>,
    capacity: usize,
}

impl QueryEmbeddingCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            fifo: VecDeque::new(),
            capacity: capacity.max(32),
        }
    }

    fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.map.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: Vec<f32>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, value);
            return;
        }
        self.fifo.push_back(key.clone());
        self.map.insert(key, value);
        while self.map.len() > self.capacity {
            if let Some(oldest) = self.fifo.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
    }
}

static QUERY_EMBED_CACHE: OnceLock<Mutex<QueryEmbeddingCache>> = OnceLock::new();
static REFLECT_BATCH_BUFFER: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();

fn embed_query_cached(state: &AppState, query: &str) -> Result<Vec<f32>, AppError> {
    let key = query.trim().to_ascii_lowercase();
    let cacheable = key.len() <= 256 && !key.is_empty();
    if cacheable {
        let cache = QUERY_EMBED_CACHE.get_or_init(|| {
            let cap = std::env::var("ANIMA_QUERY_EMBED_CACHE_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256);
            Mutex::new(QueryEmbeddingCache::new(cap))
        });
        if let Ok(guard) = cache.lock() {
            if let Some(hit) = guard.get(&key) {
                return Ok(hit);
            }
        }
    }

    let embedding = state
        .embedder
        .embed_query(query)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    if cacheable {
        if let Some(cache) = QUERY_EMBED_CACHE.get() {
            if let Ok(mut guard) = cache.lock() {
                guard.insert(key, embedding.clone());
            }
        }
    }

    Ok(embedding)
}

fn reflect_batch_size() -> usize {
    std::env::var("ANIMA_REFLECT_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .clamp(8, 1024)
}

fn reflect_batch_wait_ms() -> u64 {
    std::env::var("ANIMA_REFLECT_BATCH_WAIT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(80)
        .clamp(10, 2000)
}

fn enqueue_reflect_batched(state: &Arc<AppState>, namespace: &str, memory_ids: Vec<String>) {
    if memory_ids.is_empty() {
        return;
    }
    let Some(processor) = &state.processor else {
        return;
    };

    let batch_size = reflect_batch_size();
    let wait_ms = reflect_batch_wait_ms();
    let mut immediate: Vec<Vec<String>> = Vec::new();
    let mut schedule_delayed_flush = false;

    {
        let buffer = REFLECT_BATCH_BUFFER.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = match buffer.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        let entry = guard.entry(namespace.to_string()).or_insert_with(Vec::new);
        let was_empty = entry.is_empty();
        entry.extend(memory_ids);

        while entry.len() >= batch_size {
            let batch: Vec<String> = entry.drain(..batch_size).collect();
            immediate.push(batch);
        }

        if was_empty && !entry.is_empty() {
            schedule_delayed_flush = true;
        }
    }

    for ids in immediate {
        processor.enqueue(crate::processor::ProcessingJob::Reflect {
            namespace: namespace.to_string(),
            memory_ids: ids,
        });
    }

    if schedule_delayed_flush {
        let namespace = namespace.to_string();
        let processor = processor.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
            let ids = {
                let Some(buffer) = REFLECT_BATCH_BUFFER.get() else {
                    return;
                };
                let mut guard = match buffer.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                let Some(entry) = guard.get_mut(&namespace) else {
                    return;
                };
                if entry.is_empty() {
                    vec![]
                } else {
                    std::mem::take(entry)
                }
            };
            if !ids.is_empty() {
                processor.enqueue(crate::processor::ProcessingJob::Reflect {
                    namespace,
                    memory_ids: ids,
                });
            }
        });
    }
}

fn namespace_contains(ns: &anima_core::namespace::Namespace, memory_namespace: &str) -> bool {
    memory_namespace == ns.as_str() || memory_namespace.starts_with(&format!("{}/", ns.as_str()))
}

fn identity_stop_words() -> HashSet<&'static str> {
    [
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "which",
        "does",
        "did",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "not",
        "but",
        "about",
        "been",
        "being",
        "into",
        "than",
        "then",
        "there",
        "their",
        "they",
        "them",
        "some",
        "also",
        "just",
        "only",
        "like",
        "my",
        "your",
        "our",
        "his",
        "her",
        "its",
        "we",
        "i",
        "you",
        "me",
        "us",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
    ]
    .into_iter()
    .collect()
}

fn clean_entity_token(raw: &str) -> String {
    raw.trim_matches(|c: char| {
        c.is_whitespace() || ",.;:!?\"'()[]{}<>".contains(c)
    })
    .to_string()
}

fn is_entity_like_token(token: &str, stop_words: &HashSet<&str>) -> bool {
    if token.len() < 2 {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    if stop_words.contains(lower.as_str()) {
        return false;
    }
    let has_alpha = token.chars().any(|c| c.is_alphabetic());
    if !has_alpha {
        return false;
    }
    let first_upper = token
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false);
    let has_non_ascii = token.chars().any(|c| !c.is_ascii() && c.is_alphabetic());
    first_upper || has_non_ascii
}

/// Extract keyword search queries from a question without using an LLM.
/// Generates up to 5 queries from content words for broader recall.
fn extract_keyword_queries(question: &str) -> Vec<String> {
    static STOP: &[&str] = &[
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "above", "below", "and", "or",
        "but", "if", "while", "that", "which", "who", "whom", "this", "these",
        "those", "what", "when", "where", "how", "why", "not", "no", "so",
        "than", "too", "very", "just", "also", "then", "more", "most", "some",
        "any", "each", "every", "all", "both", "few", "many", "much", "own",
        "other", "such", "only", "same", "her", "his", "its", "my", "our",
        "your", "their", "she", "he", "it", "we", "they", "i", "you", "me",
        "him", "us", "them", "s", "likely", "recently", "go", "went",
    ];
    let words: Vec<&str> = question
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .filter(|w| !STOP.contains(&w.to_ascii_lowercase().as_str()))
        .collect();

    let mut queries = Vec::new();
    let mut seen = HashSet::new();

    let entities: Vec<&str> = words.iter().copied().filter(|w| {
        w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
    }).collect();
    let content: Vec<&str> = words.iter().copied().filter(|w| {
        w.chars().next().map(|c| !c.is_uppercase()).unwrap_or(true)
    }).collect();

    // 1. All content words (full semantic)
    if words.len() >= 2 {
        let q = words.join(" ");
        if seen.insert(q.to_ascii_lowercase()) { queries.push(q); }
    }

    // 2. Each entity paired with ALL content words
    for e in &entities {
        if !content.is_empty() {
            let mut parts = vec![*e];
            parts.extend(content.iter());
            let q = parts.join(" ");
            if seen.insert(q.to_ascii_lowercase()) { queries.push(q); }
        }
    }

    // 3. Each entity paired with each individual content word
    'outer: for e in &entities {
        for c in &content {
            let q = format!("{} {}", e, c);
            if seen.insert(q.to_ascii_lowercase()) { queries.push(q); }
            if queries.len() >= 8 { break 'outer; }
        }
    }

    // 4. Content-word pairs (captures related concepts not near entities)
    // e.g. "concert birthday", "pottery workshop", "colors patterns"
    if content.len() >= 2 {
        for pair in content.windows(2) {
            let q = format!("{} {}", pair[0], pair[1]);
            if seen.insert(q.to_ascii_lowercase()) { queries.push(q); }
            if queries.len() >= 10 { break; }
        }
    }

    queries.truncate(10);
    queries
}

/// Detect temporal intent in a question and extract date hints.
/// Returns (date_start, date_end) if temporal patterns are found.
/// This supplements (not replaces) the main search by suggesting date-range filters.
fn detect_temporal_dates(question: &str) -> Option<(Option<String>, Option<String>)> {
    let lower = question.to_lowercase();

    // Month name → number mapping
    let month_map: &[(&str, u32)] = &[
        ("january", 1), ("february", 2), ("march", 3), ("april", 4),
        ("may", 5), ("june", 6), ("july", 7), ("august", 8),
        ("september", 9), ("october", 10), ("november", 11), ("december", 12),
        ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
        ("jun", 6), ("jul", 7), ("aug", 8), ("sep", 9),
        ("oct", 10), ("nov", 11), ("dec", 12),
    ];

    // Pattern: "in [Month] [Year]" or "[Month] [Year]" or "in [Month] of [Year]"
    for &(month_name, month_num) in month_map {
        // Match "in May 2023", "May 2023", "in May of 2023"
        let patterns = [
            format!("in {} ", month_name),
            format!("{} ", month_name),
        ];
        for pat in &patterns {
            if let Some(pos) = lower.find(pat) {
                let after = &lower[pos + pat.len()..];
                // Skip "of " if present
                let after = after.strip_prefix("of ").unwrap_or(after);
                // Try to parse a 4-digit year
                if let Some(year_str) = after.split_whitespace().next() {
                    if let Ok(year) = year_str.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>() {
                        if (1900..=2100).contains(&year) {
                            let days = match month_num {
                                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                                4 | 6 | 9 | 11 => 30,
                                2 => if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { 28 },
                                _ => 31,
                            };
                            return Some((
                                Some(format!("{year:04}-{month_num:02}-01")),
                                Some(format!("{year:04}-{month_num:02}-{days:02}")),
                            ));
                        }
                    }
                }
            }
        }
    }

    // Pattern: "in [Year]" (just a year)
    let year_re_patterns = ["in ", "during ", "around "];
    for prefix in &year_re_patterns {
        if let Some(pos) = lower.find(prefix) {
            let after = &lower[pos + prefix.len()..];
            if let Some(year_str) = after.split_whitespace().next() {
                if let Ok(year) = year_str.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>() {
                    if (1900..=2100).contains(&year) && year_str.len() == 4 {
                        return Some((
                            Some(format!("{year:04}-01-01")),
                            Some(format!("{year:04}-12-31")),
                        ));
                    }
                }
            }
        }
    }

    // Pattern: explicit date "on [DD] [Month] [Year]" or "[DD] [Month] [Year]"
    // Also "[Month] [DD], [Year]"
    for &(month_name, month_num) in month_map {
        // "8 May 2023" or "on 8 May 2023"
        let words: Vec<&str> = lower.split_whitespace().collect();
        for i in 0..words.len().saturating_sub(2) {
            let w0 = words[i].trim_end_matches(|c: char| !c.is_ascii_digit());
            let w1 = words[i + 1].trim_end_matches(|c: char| !c.is_alphanumeric());
            let w2 = words.get(i + 2).map(|w| w.trim_end_matches(|c: char| !c.is_ascii_digit())).unwrap_or("");

            if let Ok(day) = w0.parse::<u32>() {
                if (1..=31).contains(&day) && w1 == month_name {
                    if let Ok(year) = w2.parse::<u32>() {
                        if (1900..=2100).contains(&year) {
                            let date = format!("{year:04}-{month_num:02}-{day:02}");
                            return Some((Some(date.clone()), Some(date)));
                        }
                    }
                }
            }

            // "May 8, 2023"
            if w0 == month_name {
                let day_str = w1.trim_end_matches(',');
                if let Ok(day) = day_str.parse::<u32>() {
                    if (1..=31).contains(&day) {
                        if let Ok(year) = w2.parse::<u32>() {
                            if (1900..=2100).contains(&year) {
                                let date = format!("{year:04}-{month_num:02}-{day:02}");
                                return Some((Some(date.clone()), Some(date)));
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

fn extract_candidate_entities(text: &str) -> Vec<String> {
    let stop_words = identity_stop_words();
    let mut entities = Vec::new();
    let mut current = Vec::new();
    for raw in text.split_whitespace() {
        let token = clean_entity_token(raw);
        if token.is_empty() {
            continue;
        }
        if is_entity_like_token(&token, &stop_words) {
            current.push(token);
            continue;
        }
        if !current.is_empty() {
            entities.push(current.join(" "));
            current.clear();
        }
    }
    if !current.is_empty() {
        entities.push(current.join(" "));
    }

    // include single-token candidates when phrase extraction misses them
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-') {
        let token = clean_entity_token(raw);
        if token.is_empty() || !is_entity_like_token(&token, &stop_words) {
            continue;
        }
        entities.push(token);
    }

    let mut seen = HashSet::new();
    entities
        .into_iter()
        .filter(|e| seen.insert(e.to_ascii_lowercase()))
        .collect()
}

async fn ingest_identity_hints(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    text: &str,
    confidence: f64,
    memory_id: Option<&str>,
) {
    for entity_name in extract_candidate_entities(text).into_iter().take(8) {
        let entity = match state
            .store
            .upsert_identity_entity(
                ns,
                &entity_name,
                None,
                confidence.clamp(0.2, 0.95),
                Some(serde_json::json!({ "source": "memory_ingest" })),
            )
            .await
        {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("identity upsert failed for '{entity_name}': {e}");
                continue;
            }
        };
        if let Err(e) = state
            .store
            .add_identity_alias(
                ns,
                &entity.id,
                &entity_name,
                None,
                (confidence + 0.05).clamp(0.2, 0.99),
            )
            .await
        {
            tracing::warn!("identity alias add failed for '{entity_name}': {e}");
        }
        // Create entity→memory edge for graph-based retrieval
        if let Some(mid) = memory_id {
            if let Err(e) = state
                .store
                .upsert_causal_edge(
                    ns,
                    &entity.id,
                    mid,
                    "mentioned_in",
                    confidence.clamp(0.2, 0.95),
                    None,
                )
                .await
            {
                tracing::warn!("causal edge failed for entity '{entity_name}' → memory '{mid}': {e}");
            }
        }
    }
}

fn redaction_enabled() -> bool {
    std::env::var("ANIMA_REDACTION_ENABLED")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn sensitive_metadata_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "cookie",
        "session",
        "ssn",
        "credit_card",
        "card_number",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn redact_metadata_value(value: &mut serde_json::Value) -> bool {
    let mut changed = false;
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if sensitive_metadata_key(k) {
                    if !v.is_string() || v.as_str() != Some("[redacted]") {
                        *v = serde_json::Value::String("[redacted]".to_string());
                        changed = true;
                    }
                } else if redact_metadata_value(v) {
                    changed = true;
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if redact_metadata_value(item) {
                    changed = true;
                }
            }
        }
        _ => {}
    }
    changed
}

fn looks_like_email(token: &str) -> bool {
    let token = token.trim_matches(|c: char| ",;:()[]{}<>\"'".contains(c));
    let mut parts = token.split('@');
    let local = parts.next().unwrap_or_default();
    let domain = parts.next().unwrap_or_default();
    !local.is_empty() && domain.contains('.') && parts.next().is_none()
}

fn looks_like_ssn(token: &str) -> bool {
    let t = token.trim_matches(|c: char| ",;:()[]{}<>\"'".contains(c));
    if t.len() != 11 {
        return false;
    }
    let b = t.as_bytes();
    b[3] == b'-'
        && b[6] == b'-'
        && b[..3].iter().all(|c| c.is_ascii_digit())
        && b[4..6].iter().all(|c| c.is_ascii_digit())
        && b[7..].iter().all(|c| c.is_ascii_digit())
}

fn looks_like_api_key(token: &str) -> bool {
    let t = token.trim_matches(|c: char| ",;:()[]{}<>\"'".contains(c));
    (t.starts_with("sk-") && t.len() >= 20)
        || (t.starts_with("gsk_") && t.len() >= 20)
        || (t.starts_with("ghp_") && t.len() >= 20)
}

fn looks_like_credit_card(token: &str) -> bool {
    let digits: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.len() >= 13 && digits.len() <= 19
}

fn redact_sensitive_text(text: &str) -> (String, bool) {
    if !redaction_enabled() {
        return (text.to_string(), false);
    }
    let mut changed = false;
    let mut out = Vec::new();
    for token in text.split_whitespace() {
        let replacement = if looks_like_email(token) {
            Some("[redacted:email]")
        } else if looks_like_ssn(token) {
            Some("[redacted:ssn]")
        } else if looks_like_api_key(token) {
            Some("[redacted:key]")
        } else if looks_like_credit_card(token) {
            Some("[redacted:card]")
        } else {
            None
        };
        if let Some(masked) = replacement {
            changed = true;
            out.push(masked.to_string());
        } else {
            out.push(token.to_string());
        }
    }
    (out.join(" "), changed)
}

fn redact_content_and_metadata(
    content: &str,
    metadata: Option<serde_json::Value>,
) -> (String, Option<serde_json::Value>, bool) {
    let (content, mut changed) = redact_sensitive_text(content);
    let mut metadata_out = metadata;
    if let Some(ref mut m) = metadata_out {
        if redact_metadata_value(m) {
            changed = true;
        }
    }
    (content, metadata_out, changed)
}

fn build_chat_provenance(
    ns: &anima_core::namespace::Namespace,
    memories_used: &[MemoryContext],
    redaction_applied: bool,
) -> ResponseProvenance {
    let mut source_ids = Vec::new();
    let mut sources = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for mem in memories_used {
        if seen.insert(mem.id.clone()) {
            source_ids.push(mem.id.clone());
            sources.push(ProvenanceSource {
                memory_id: mem.id.clone(),
                score: Some(mem.score),
                source: mem.source.clone(),
                memory_type: Some(mem.memory_type.clone()),
                created_at: None,
            });
        }
    }
    ResponseProvenance {
        namespace: ns.as_str().to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        source_ids,
        redaction_applied,
        sources,
    }
}

fn build_ask_provenance(
    ns: &anima_core::namespace::Namespace,
    memories_referenced: &[AskMemoryRef],
    redaction_applied: bool,
) -> ResponseProvenance {
    let mut source_ids = Vec::new();
    let mut sources = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for mem in memories_referenced {
        if seen.insert(mem.id.clone()) {
            source_ids.push(mem.id.clone());
            sources.push(ProvenanceSource {
                memory_id: mem.id.clone(),
                score: Some(mem.score),
                source: Some("search".to_string()),
                memory_type: None,
                created_at: Some(mem.created_at.clone()),
            });
        }
    }
    ResponseProvenance {
        namespace: ns.as_str().to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        source_ids,
        redaction_applied,
        sources,
    }
}

fn make_chat_response(
    ns: &anima_core::namespace::Namespace,
    reply: String,
    mut memories_used: Vec<MemoryContext>,
    mut memories_added: Vec<AddedMemory>,
    mode: String,
    degraded: bool,
) -> ChatResponse {
    let (reply, mut redaction_applied) = redact_sensitive_text(&reply);
    for mem in &mut memories_used {
        let (redacted, changed) = redact_sensitive_text(&mem.content);
        mem.content = redacted;
        redaction_applied |= changed;
    }
    for mem in &mut memories_added {
        let (redacted, changed) = redact_sensitive_text(&mem.content);
        mem.content = redacted;
        redaction_applied |= changed;
    }
    let provenance = build_chat_provenance(ns, &memories_used, redaction_applied);
    ChatResponse {
        reply,
        memories_used,
        memories_added,
        mode,
        provenance,
        degraded,
    }
}

fn make_ask_response(
    ns: &anima_core::namespace::Namespace,
    answer: String,
    queries_used: Vec<String>,
    mut memories_referenced: Vec<AskMemoryRef>,
    total_search_results: usize,
    elapsed_ms: f64,
    needs_confirmation: Vec<ConfirmationQuestion>,
    degraded: bool,
    conflicts: Vec<ConflictNote>,
) -> AskResponse {
    let (answer, mut redaction_applied) = redact_sensitive_text(&answer);
    for mem in &mut memories_referenced {
        let (redacted, changed) = redact_sensitive_text(&mem.content);
        mem.content = redacted;
        redaction_applied |= changed;
    }
    let provenance = build_ask_provenance(ns, &memories_referenced, redaction_applied);
    AskResponse {
        answer,
        queries_used,
        memories_referenced,
        total_search_results,
        elapsed_ms,
        provenance,
        needs_confirmation,
        degraded,
        conflicts,
    }
}

pub async fn add_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<AddMemoryRequest>,
) -> Result<Json<AddMemoryResponse>, AppError> {
    let AddMemoryRequest {
        content,
        metadata,
        consolidate,
        tags,
        episode_id,
        category,
        source,
    } = req;
    let parsed_category = category.unwrap_or_else(|| "general".to_string());
    let parsed_source = source.filter(|s| !s.is_empty()).unwrap_or_else(|| "user_stated".to_string());
    let source_confidence = anima_core::memory::default_confidence_for_source(&parsed_source);
    let (content, metadata, _) = redact_content_and_metadata(&content, metadata);

    if content.trim().is_empty() {
        return Err(AppError::BadRequest("content cannot be empty".into()));
    }

    // Embed the content
    let embedding = state
        .embedder
        .embed(&content)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    // Step 1: Exact dedup by hash
    let hash = content_hash(&content);
    if let Some(existing) = state.store.find_by_hash(&ns, &hash).await? {
        return Ok(Json(AddMemoryResponse {
            id: existing.id,
            action: "deduplicated".into(),
            merged_into: None,
        }));
    }

    // Step 2: Consolidation (if enabled and requested)
    // novel_content_override is set by predict-calibrate when only a subset of the input is new.
    let mut novel_content_override: Option<String> = None;
    if consolidate {
        if let Some(consolidator) = &state.consolidator {
            let similar = state
                .store
                .find_similar(
                    &embedding,
                    &ns,
                    5,
                    consolidator.similarity_threshold(),
                )
                .await?;

            if !similar.is_empty() {
                let decision = consolidator
                    .decide(&content, &similar)
                    .await
                    .map_err(|e| AppError::Internal(format!("consolidation error: {e}")))?;

                match decision.action {
                    ConsolidationActionType::NoChange => {
                        // Memory already captured by existing ones
                        let id = similar[0].0.id.clone();
                        return Ok(Json(AddMemoryResponse {
                            id,
                            action: "no_change".into(),
                            merged_into: None,
                        }));
                    }
                    ConsolidationActionType::Update => {
                        // Merge into existing memory
                        if let (Some(target_id), Some(merged_content)) =
                            (&decision.target_id, &decision.merged_content)
                        {
                            let new_embedding = state
                                .embedder
                                .embed(merged_content)
                                .map_err(|e| AppError::Embedding(e.to_string()))?;
                            state
                                .store
                                .update_content(target_id, merged_content, &new_embedding)
                                .await?;
                            return Ok(Json(AddMemoryResponse {
                                id: target_id.clone(),
                                action: "updated".into(),
                                merged_into: Some(target_id.clone()),
                            }));
                        }
                        // Fallthrough to create if target/content missing
                    }
                    ConsolidationActionType::Supersede => {
                        // Create the new memory first so we have its ID
                        let mut memory = Memory::new(
                            ns.as_str().to_string(),
                            content,
                            metadata,
                            tags,
                            None,
                        );
                        memory.category = parsed_category.clone();
        memory.confidence = source_confidence;
        memory.source = parsed_source.clone();
                        // Resolve episode_id for supersede path
                        memory.episode_id = episode_id.clone().or_else(|| {
                            memory.metadata.as_ref()
                                .and_then(|m| m.get("session"))
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        });
                        let new_id = memory.id.clone();
                        state.store.insert(&memory, &embedding).await?;
                        ingest_identity_hints(&state, &ns, &memory.content, 0.72, Some(&new_id)).await;

                        // Mark old as superseded, linking to the new memory
                        if let Some(target_id) = &decision.target_id {
                            state
                                .store
                                .mark_superseded(target_id, &new_id)
                                .await?;
                        }

                        return Ok(Json(AddMemoryResponse {
                            id: new_id,
                            action: "created".into(),
                            merged_into: None,
                        }));
                    }
                    ConsolidationActionType::Create => {
                        // Predict-calibrate: if the LLM extracted only the novel claims,
                        // store those instead of the full raw input.
                        // Safety net: if novel_content is less than 40% the length of the
                        // original, the LLM likely over-stripped specific details — fall back
                        // to the full content to avoid losing named entities / exact facts.
                        novel_content_override = decision
                            .novel_content
                            .filter(|s| !s.trim().is_empty())
                            .filter(|novel| {
                                let orig_len = content.len();
                                orig_len == 0 || novel.len() * 100 / orig_len >= 40
                            });
                    }
                }
            }
        }
    }

    // Step 3: Create new memory
    // Use novel_content if predict-calibrate identified only a subset as new.
    let (final_content, final_embedding) = if let Some(novel) = novel_content_override {
        let emb = state
            .embedder
            .embed(&novel)
            .map_err(|e| AppError::Embedding(e.to_string()))?;
        (novel, emb)
    } else {
        (content, embedding)
    };
    // Resolve episode_id: explicit field > metadata.session
    let resolved_episode = episode_id.or_else(|| {
        metadata.as_ref()
            .and_then(|m| m.get("session"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });
    let mut memory = Memory::new(ns.as_str().to_string(), final_content, metadata, tags, None);
    memory.episode_id = resolved_episode;
    memory.category = parsed_category;
    memory.confidence = source_confidence;
    memory.source = parsed_source;
    let id = memory.id.clone();
    state.store.insert(&memory, &final_embedding).await?;
    state.record_ingestion(1);
    ingest_identity_hints(&state, &ns, &memory.content, 0.72, Some(&id)).await;

    // Enqueue background reflection (micro-batched).
    enqueue_reflect_batched(&state, ns.as_str(), vec![id.clone()]);

    Ok(Json(AddMemoryResponse {
        id,
        action: "created".into(),
        merged_into: None,
    }))
}

pub async fn add_memories_batch(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<AddMemoryBatchRequest>,
) -> Result<Json<AddMemoryBatchResponse>, AppError> {
    if req.items.is_empty() {
        return Err(AppError::BadRequest("items cannot be empty".into()));
    }
    if req.items.len() > 2000 {
        return Err(AppError::BadRequest(
            "items exceeds max batch size (2000)".into(),
        ));
    }

    let start = Instant::now();
    let mut entries: Vec<(Memory, Vec<f32>)> = Vec::with_capacity(req.items.len());
    let mut ids = Vec::with_capacity(req.items.len());

    for item in req.items {
        let (content, metadata, _) = redact_content_and_metadata(&item.content, item.metadata);
        if content.trim().is_empty() {
            return Err(AppError::BadRequest(
                "batch item content cannot be empty".into(),
            ));
        }
        let embedding = state
            .embedder
            .embed(&content)
            .map_err(|e| AppError::Embedding(e.to_string()))?;
        let ep_id = item.episode_id.or_else(|| {
            metadata.as_ref()
                .and_then(|m| m.get("session"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
        let mut memory = Memory::new(
            ns.as_str().to_string(),
            content,
            metadata,
            item.tags,
            Some("raw".to_string()),
        );
        memory.episode_id = ep_id;
        ids.push(memory.id.clone());
        entries.push((memory, embedding));
    }

    state.store.insert_many(&entries).await?;
    state.record_ingestion(ids.len() as u64);

    if req.reflect {
        enqueue_reflect_batched(&state, ns.as_str(), ids.clone());
    }

    Ok(Json(AddMemoryBatchResponse {
        created: ids.len(),
        ids,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// Run the multi-stage /ask retrieval pipeline (keyword expansion, entity resolution,
/// temporal supplement, episode expansion, entity-linked retrieval) and return ranked
/// (Memory, score) pairs. Reusable by both /ask and search_mode=ask_retrieval.
async fn run_ask_retrieval_pipeline(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    query: &str,
    limit: usize,
    scorer_config: &anima_core::search::ScorerConfig,
) -> Result<Vec<(anima_core::memory::Memory, f64)>, AppError> {
    let embedding = embed_query_cached(state, query)?;

    // 1. Initial hybrid search
    let results = state.store.search(&embedding, query, ns, &SearchMode::Hybrid, limit, scorer_config)
        .await
        .map_err(|e| AppError::Database(e.to_string()))?;
    record_retrieval_observations(state, ns, &results, 0.15, "ask_retrieval_hybrid").await;

    // 2. Keyword expansion
    let keyword_queries = extract_keyword_queries(query);
    let mut expanded_results = Vec::new();
    for kq in &keyword_queries {
        let kq_embedding = embed_query_cached(state, kq)?;
        if let Ok(kw_results) = state
            .store
            .search(&kq_embedding, kq, ns, &SearchMode::Hybrid, limit, scorer_config)
            .await
        {
            expanded_results.extend(kw_results);
        }
    }

    // 3. Identity-aware query expansion
    let entities = extract_candidate_entities(query);
    let mut resolved_entity_queries: Vec<String> = Vec::new();
    for entity in &entities {
        match state.store.resolve_identity(ns, entity, 3).await {
            Ok(resolution) => {
                if resolution.candidates.is_empty() {
                    resolved_entity_queries.push(entity.clone());
                    continue;
                }
                if resolution.best_confidence >= 0.62 {
                    let mut added_bases = HashSet::new();
                    for c in &resolution.candidates {
                        let base = c.canonical_name
                            .split_whitespace().next().unwrap_or(&c.canonical_name)
                            .trim_end_matches("'s").trim_end_matches("\u{2019}s")
                            .to_string();
                        if added_bases.insert(base.to_ascii_lowercase()) {
                            resolved_entity_queries.push(base);
                        }
                    }
                } else {
                    resolved_entity_queries.push(entity.clone());
                }
            }
            Err(e) => {
                warn!("identity resolution failed for '{entity}': {e}");
                resolved_entity_queries.push(entity.clone());
            }
        }
    }
    if resolved_entity_queries.is_empty() {
        resolved_entity_queries = entities.clone();
    }
    let mut dedup = HashSet::new();
    resolved_entity_queries.retain(|q| dedup.insert(q.to_ascii_lowercase()));

    // 4. Entity keyword searches
    let mut entity_results = Vec::new();
    let topic_words: Vec<&str> = {
        let stop: HashSet<&str> = ["a","an","the","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could","should","may","might",
            "shall","can","to","of","in","for","on","with","at","by","from","as","into","about",
            "and","or","but","if","that","which","who","this","what","when","where","how","why",
            "not","no","so","long","much","many","her","his","their","my","your","they","she",
            "he","it","we","you","i","me","him","them","us"].into_iter().collect();
        let entity_lower: HashSet<String> = resolved_entity_queries.iter()
            .flat_map(|e| e.split_whitespace().map(|w| w.to_ascii_lowercase()))
            .collect();
        query.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty() && w.len() > 2)
            .filter(|w| !stop.contains(w.to_ascii_lowercase().as_str()))
            .filter(|w| !entity_lower.contains(&w.to_ascii_lowercase()))
            .collect()
    };
    for entity in &resolved_entity_queries {
        if let Ok(kw_results) = state
            .store
            .search(&embedding, entity, ns, &SearchMode::Keyword, limit / 2, scorer_config)
            .await
        {
            record_retrieval_observations(state, ns, &kw_results, 0.15, "ask_retrieval_entity_kw").await;
            entity_results.extend(kw_results);
        }
        for &topic in &topic_words {
            let combined = format!("{entity} {topic}");
            if let Ok(kw_results) = state
                .store
                .search(&embedding, &combined, ns, &SearchMode::Keyword, limit / 3, scorer_config)
                .await
            {
                entity_results.extend(kw_results);
            }
        }
    }

    // 5. Merge all sources: best score per memory_id
    let mut best_scores: HashMap<String, f64> = HashMap::new();
    for sr in results.iter().chain(expanded_results.iter()).chain(entity_results.iter()) {
        let entry = best_scores.entry(sr.memory_id.clone()).or_insert(0.0);
        if sr.score > *entry { *entry = sr.score; }
    }
    let mut scored_ids: Vec<(String, f64)> = best_scores.into_iter().collect();
    scored_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut all_results: Vec<(anima_core::memory::Memory, f64)> = Vec::new();
    let mut seen_contents: HashSet<String> = HashSet::new();
    for (memory_id, score) in scored_ids {
        if let Ok(Some(memory)) = state.store.get(&memory_id).await {
            if seen_contents.insert(memory.content.clone()) {
                all_results.push((memory, score));
            }
        }
    }

    // Drop results scoring below 25% of top hit
    if let Some((_, top_score)) = all_results.first() {
        let cutoff = top_score * 0.25;
        all_results.retain(|(_, s)| *s >= cutoff);
    }
    all_results.truncate(limit);

    // 6. Temporal date-range supplement
    if let Some((date_start, date_end)) = detect_temporal_dates(query) {
        let start = date_start.as_deref().unwrap_or("0000-01-01");
        let end = date_end.as_deref().unwrap_or("9999-12-31");
        if let Ok(date_mems) = state.store.find_by_date_range(ns, start, end, 20).await {
            for mem in date_mems {
                if seen_contents.insert(mem.content.clone()) {
                    all_results.push((mem, 0.25));
                }
            }
        }
    }

    // 7. Episode expansion
    {
        let mut episode_ids_seen: HashSet<String> = HashSet::new();
        for (mem, score) in &all_results {
            if *score < 0.3 { continue; }
            if let Some(ep_id) = &mem.episode_id {
                if !ep_id.is_empty() { episode_ids_seen.insert(ep_id.clone()); }
            } else if let Some(meta) = &mem.metadata {
                if let Some(session) = meta.get("session").and_then(|v| v.as_str()) {
                    episode_ids_seen.insert(session.to_string());
                }
            }
        }
        let mut episodes_expanded = 0;
        for ep_id in &episode_ids_seen {
            if episodes_expanded >= 3 { break; }
            if let Ok(co_mems) = state.store.find_by_episode(ns, ep_id, 10).await {
                for mem in co_mems {
                    if seen_contents.insert(mem.content.clone()) {
                        all_results.push((mem, 0.3));
                    }
                }
            }
            episodes_expanded += 1;
        }
    }

    // 8. Entity-linked retrieval
    {
        let mut entity_ids: Vec<String> = Vec::new();
        for entity in &resolved_entity_queries {
            if let Ok(resolution) = state.store.resolve_identity(ns, entity, 3).await {
                for c in &resolution.candidates {
                    if c.score >= 0.3 { entity_ids.push(c.entity_id.clone()); }
                }
            }
        }
        entity_ids.sort();
        entity_ids.dedup();
        if !entity_ids.is_empty() {
            if let Ok(linked_ids) = state.store.find_memories_by_entity_ids(ns, &entity_ids, 30).await {
                for mid in linked_ids {
                    if let Ok(Some(mem)) = state.store.get(&mid).await {
                        if seen_contents.insert(mem.content.clone()) {
                            all_results.push((mem, 0.28));
                        }
                    }
                }
            }
        }
    }

    Ok(all_results)
}

/// Run query rewriting: extract keywords from a natural-language query and run expanded
/// searches, merging results with the original. Returns merged ScoredResults.
async fn run_query_rewrite(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    query: &str,
    _embedding: &[f32],
    mode: &SearchMode,
    limit: usize,
    scorer_config: &anima_core::search::ScorerConfig,
    original_results: &[anima_core::search::ScoredResult],
) -> Result<Vec<anima_core::search::ScoredResult>, AppError> {
    let keyword_queries = extract_keyword_queries(query);
    if keyword_queries.is_empty() {
        return Ok(original_results.to_vec());
    }

    let mut best_scores: HashMap<String, anima_core::search::ScoredResult> = HashMap::new();
    for sr in original_results {
        best_scores.insert(sr.memory_id.clone(), sr.clone());
    }

    for kq in &keyword_queries {
        let kq_embedding = embed_query_cached(state, kq)?;
        if let Ok(kw_results) = state
            .store
            .search(&kq_embedding, kq, ns, mode, limit, scorer_config)
            .await
        {
            for sr in kw_results {
                let entry = best_scores.entry(sr.memory_id.clone()).or_insert_with(|| sr.clone());
                if sr.score > entry.score {
                    *entry = sr;
                }
            }
        }
    }

    let mut merged: Vec<anima_core::search::ScoredResult> = best_scores.into_values().collect();
    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(merged)
}

pub async fn search_memories(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    if req.query.trim().is_empty() {
        return Err(AppError::BadRequest("query cannot be empty".into()));
    }
    if is_likely_noisy_or_adversarial_input(&req.query) {
        return Ok(Json(SearchResponse {
            results: vec![],
            query_time_ms: 0.0,
        }));
    }

    let limit = req.limit.min(100).max(1);
    let start = Instant::now();

    // Parse search mode
    let mode = match req.search_mode.as_deref() {
        Some("vector") => SearchMode::Vector,
        Some("keyword") => SearchMode::Keyword,
        Some("ask_retrieval") => SearchMode::AskRetrieval,
        _ => SearchMode::Hybrid,
    };

    // Override scorer config from request
    let mut scorer_config = state.scorer_config.read().await.clone();
    if let Some(tw) = req.temporal_weight {
        scorer_config.temporal_weight = tw.clamp(0.0, 1.0);
    }
    if let Some(mt) = req.max_tier {
        scorer_config.max_tier = mt.clamp(1, 4);
    }
    scorer_config.date_start = req.date_start;
    scorer_config.date_end = req.date_end;

    // ── ask_retrieval mode: full multi-stage pipeline, return as search results ──
    if matches!(mode, SearchMode::AskRetrieval) {
        let all_results = run_ask_retrieval_pipeline(&state, &ns, &req.query, limit, &scorer_config).await?;
        let mut results = Vec::with_capacity(all_results.len().min(limit));
        for (memory, score) in all_results.into_iter().take(limit) {
            let (content, metadata, _) = redact_content_and_metadata(&memory.content, memory.metadata);
            results.push(SearchResultDto {
                id: memory.id,
                content,
                metadata,
                tags: memory.tags,
                memory_type: memory.memory_type,
                category: memory.category.clone(),
                confidence: memory.confidence,
                source: memory.source.clone(),
                score,
                vector_score: None,
                keyword_score: None,
                temporal_score: None,
                created_at: memory.created_at.to_rfc3339(),
                updated_at: memory.updated_at.to_rfc3339(),
            });
        }
        return Ok(Json(SearchResponse {
            results,
            query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }));
    }

    // ── Standard search modes (hybrid/vector/keyword) ──
    // Embed query
    let embedding = embed_query_cached(&state, &req.query)?;

    // Search — fetch more candidates if reranker is enabled so it has room to reorder
    let reranker_top_n = state.config.reranker.top_n;
    let fetch_limit = if state.reranker.is_some() { limit.max(reranker_top_n) } else { limit };
    let mut scored = state
        .store
        .search(&embedding, &req.query, &ns, &mode, fetch_limit, &scorer_config)
        .await?;

    // Feature 2: Query rewriting — expand with keyword extraction if enabled
    if req.query_rewrite {
        scored = run_query_rewrite(&state, &ns, &req.query, &embedding, &mode, fetch_limit, &scorer_config, &scored).await?;
    }

    // Re-rank with cross-encoder if enabled.
    // Skip if the top result is already high-confidence (> 0.50 from initial retrieval)
    let skip_rerank = scored.first().map_or(false, |r| r.score > 0.50);
    if let Some(ref reranker) = state.reranker {
        let top_n = reranker_top_n.min(scored.len());
        if top_n > 0 && !skip_rerank {
            // Fetch content for top-N candidates
            let mut contents: Vec<String> = Vec::with_capacity(top_n);
            for sr in scored.iter().take(top_n) {
                let content = state.store.get(&sr.memory_id).await?
                    .map(|m| m.content)
                    .unwrap_or_default();
                contents.push(content);
            }
            let doc_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
            match reranker.score_pairs(&req.query, &doc_refs) {
                Ok(rerank_scores) => {
                    for (i, score) in rerank_scores.iter().enumerate() {
                        if i < scored.len() {
                            // Blend reranker score with a small keyword-match bonus
                            // so results that actually contain query terms aren't buried
                            // by cross-encoder noise on short documents.
                            let kw_bonus = if scored[i].keyword_score.unwrap_or(0.0) > 0.0 {
                                0.03
                            } else {
                                0.0
                            };
                            scored[i].score = (*score + kw_bonus).min(1.0);
                        }
                    }
                    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                }
                Err(e) => {
                    tracing::warn!(domain = "reranker", "Reranker failed, using original scores: {e}");
                }
            }
        }
    }

    // Trim to requested limit
    scored.truncate(limit);

    // Fetch full memory data for results
    let mut results = Vec::with_capacity(scored.len());

    for sr in &scored {
        if let Some(memory) = state.store.get(&sr.memory_id).await? {
            let (content, metadata, _) = redact_content_and_metadata(&memory.content, memory.metadata);
            results.push(SearchResultDto {
                id: memory.id,
                content,
                metadata,
                tags: memory.tags,
                memory_type: memory.memory_type,
                category: memory.category.clone(),
                confidence: memory.confidence,
                source: memory.source.clone(),
                score: sr.score,
                vector_score: sr.vector_score,
                keyword_score: sr.keyword_score,
                temporal_score: sr.temporal_score,
                created_at: memory.created_at.to_rfc3339(),
                updated_at: memory.updated_at.to_rfc3339(),
            });
        }
    }

    let elapsed = start.elapsed();

    Ok(Json(SearchResponse {
        results,
        query_time_ms: elapsed.as_secs_f64() * 1000.0,
    }))
}

pub async fn get_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
) -> Result<Json<MemoryResponse>, AppError> {
    let memory = state
        .store
        .get(&id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {id} not found")))?;
    if !namespace_contains(&ns, &memory.namespace) {
        return Err(AppError::Forbidden(format!(
            "memory {id} is not accessible in namespace '{}'",
            ns.as_str()
        )));
    }
    let (content, metadata, _) = redact_content_and_metadata(&memory.content, memory.metadata);

    Ok(Json(MemoryResponse {
        id: memory.id,
        namespace: memory.namespace,
        content,
        metadata,
        tags: memory.tags,
        memory_type: memory.memory_type,
        category: memory.category.clone(),
        confidence: memory.confidence,
        source: memory.source.clone(),
        status: memory.status.as_str().to_string(),
        created_at: memory.created_at.to_rfc3339(),
        updated_at: memory.updated_at.to_rfc3339(),
        access_count: memory.access_count,
        importance: memory.importance,
    }))
}

pub async fn update_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<Json<UpdateMemoryResponse>, AppError> {
    let target = state
        .store
        .get(&id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {id} not found")))?;
    if !namespace_contains(&ns, &target.namespace) {
        return Err(AppError::Forbidden(format!(
            "memory {id} is not writable in namespace '{}'",
            ns.as_str()
        )));
    }

    // Update content if provided (requires re-embedding)
    if let Some(ref content) = req.content {
        let (redacted_content, _, _) = redact_content_and_metadata(content, None);
        if redacted_content.trim().is_empty() {
            return Err(AppError::BadRequest("content cannot be empty".into()));
        }

        let embedding = state
            .embedder
            .embed(&redacted_content)
            .map_err(|e| AppError::Embedding(e.to_string()))?;

        let updated = state
            .store
            .update_content(&id, &redacted_content, &embedding)
            .await?;
        if !updated {
            return Err(AppError::NotFound(format!("memory {id} not found")));
        }
    }

    // Update metadata fields if provided (no re-embedding needed)
    if req.importance.is_some() || req.tags.is_some() {
        let updated = state
            .store
            .update_metadata(
                &id,
                None,
                req.importance,
                req.tags.as_deref(),
            )
            .await?;
        if !updated && req.content.is_none() {
            return Err(AppError::NotFound(format!("memory {id} not found")));
        }
    }

    // At least one field must be provided
    if req.content.is_none() && req.importance.is_none() && req.tags.is_none() {
        return Err(AppError::BadRequest("at least one field to update is required".into()));
    }

    Ok(Json(UpdateMemoryResponse {
        id,
        updated: true,
    }))
}

pub async fn patch_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
    Json(req): Json<PatchMemoryRequest>,
) -> Result<Json<PatchMemoryResponse>, AppError> {
    let target = state
        .store
        .get(&id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {id} not found")))?;
    if !namespace_contains(&ns, &target.namespace) {
        return Err(AppError::Forbidden(format!(
            "memory {id} is not writable in namespace '{}'",
            ns.as_str()
        )));
    }

    let has_any = req.content.is_some()
        || req.metadata.is_some()
        || req.memory_type.is_some()
        || req.importance.is_some()
        || req.tags.is_some();
    if !has_any {
        return Err(AppError::BadRequest(
            "patch requires at least one field".into(),
        ));
    }

    let mut metadata = req.metadata.clone();
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let content = if let Some(content) = req.content.as_deref() {
        let (redacted_content, _, _) = redact_content_and_metadata(content, None);
        if redacted_content.trim().is_empty() {
            return Err(AppError::BadRequest("content cannot be empty".into()));
        }
        Some(redacted_content)
    } else {
        None
    };
    let embedding = if let Some(ref c) = content {
        Some(
            state
                .embedder
                .embed(c)
                .map_err(|e| AppError::Embedding(e.to_string()))?,
        )
    } else {
        None
    };

    let patch = MemoryPatch {
        content,
        metadata,
        memory_type: req.memory_type.clone(),
        importance: req.importance,
        tags: req.tags.clone(),
    };
    let actor = headers
        .get("X-Anima-Principal")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("api");
    let result = state
        .store
        .patch_memory(
            &id,
            &patch,
            embedding.as_deref(),
            Some(actor),
            req.reason.as_deref(),
        )
        .await?;
    let Some(patched) = result else {
        return Err(AppError::NotFound(format!("memory {id} not found")));
    };

    Ok(Json(PatchMemoryResponse {
        id: patched.memory.id,
        patched: true,
        revision_number: patched.revision_number,
    }))
}

pub async fn rollback_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RollbackMemoryRequest>,
) -> Result<Json<RollbackMemoryResponse>, AppError> {
    if req.revision_number <= 0 {
        return Err(AppError::BadRequest(
            "revision_number must be a positive integer".into(),
        ));
    }
    let target = state
        .store
        .get(&id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {id} not found")))?;
    if !namespace_contains(&ns, &target.namespace) {
        return Err(AppError::Forbidden(format!(
            "memory {id} is not writable in namespace '{}'",
            ns.as_str()
        )));
    }

    let actor = headers
        .get("X-Anima-Principal")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("api");
    let rolled_back = state
        .store
        .rollback_memory_to_revision(&id, req.revision_number, Some(actor), req.reason.as_deref())
        .await?
        .ok_or_else(|| AppError::NotFound(format!(
            "revision {} for memory {id} not found",
            req.revision_number
        )))?;

    Ok(Json(RollbackMemoryResponse {
        id: rolled_back.memory.id,
        rolled_back_to_revision: req.revision_number,
        revision_number: rolled_back.revision_number,
    }))
}

pub async fn merge_memories(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    headers: axum::http::HeaderMap,
    Json(req): Json<MergeMemoriesRequest>,
) -> Result<Json<MergeMemoriesResponse>, AppError> {
    if req.source_memory_ids.is_empty() {
        return Err(AppError::BadRequest(
            "source_memory_ids cannot be empty".into(),
        ));
    }
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content cannot be empty".into()));
    }

    let mut source_memories = Vec::new();
    for source_id in &req.source_memory_ids {
        let source = state
            .store
            .get(source_id)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("source memory {source_id} not found")))?;
        if !namespace_contains(&ns, &source.namespace) {
            return Err(AppError::Forbidden(format!(
                "source memory {source_id} is outside namespace '{}'",
                ns.as_str()
            )));
        }
        source_memories.push(source);
    }

    let tags = if let Some(t) = req.tags.clone() {
        t
    } else {
        let mut merged_tags = HashSet::new();
        for s in &source_memories {
            for tag in &s.tags {
                merged_tags.insert(tag.clone());
            }
        }
        merged_tags.into_iter().collect()
    };
    let mut metadata = req.metadata.clone().unwrap_or_else(|| serde_json::json!({}));
    if !metadata.is_object() {
        metadata = serde_json::json!({});
    }
    metadata["merge"] = serde_json::json!({
        "source_memory_ids": req.source_memory_ids,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });
    let _ = redact_metadata_value(&mut metadata);
    let (content, _, _) = redact_content_and_metadata(&req.content, None);
    let embedding = state
        .embedder
        .embed(&content)
        .map_err(|e| AppError::Embedding(e.to_string()))?;
    let mut merged = Memory::new(
        ns.as_str().to_string(),
        content,
        Some(metadata),
        tags,
        Some(
            req.memory_type
                .clone()
                .unwrap_or_else(|| "merged".to_string()),
        ),
    );
    let avg_importance = if source_memories.is_empty() {
        5
    } else {
        let sum: i32 = source_memories.iter().map(|m| m.importance).sum();
        (sum / source_memories.len() as i32).clamp(1, 10)
    };
    merged.importance = req.importance.unwrap_or(avg_importance).clamp(1, 10);

    let actor = headers
        .get("X-Anima-Principal")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("api");
    let result = state
        .store
        .merge_memories(
            &ns,
            &req.source_memory_ids,
            &merged,
            &embedding,
            Some(actor),
            req.reason.as_deref(),
        )
        .await?;
    Ok(Json(MergeMemoriesResponse {
        merged_memory_id: result.merged_memory_id,
        superseded_source_ids: result.superseded_source_ids,
        revision_number: result.merged_revision_number,
    }))
}

pub async fn list_claim_revisions(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
    Query(params): Query<RevisionListParams>,
) -> Result<Json<Vec<anima_db::store::ClaimRevision>>, AppError> {
    let limit = params.limit.unwrap_or(50).clamp(1, 500);
    let revisions = state.store.list_claim_revisions(&ns, &id, limit).await?;
    if revisions.is_empty() {
        return Err(AppError::NotFound(format!(
            "no revisions found for memory {id}"
        )));
    }
    Ok(Json(revisions))
}

pub async fn list_audit_events(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<AuditListParams>,
) -> Result<Json<Vec<anima_db::store::AuditEvent>>, AppError> {
    let limit = params.limit.unwrap_or(100).clamp(1, 1000);
    let events = state
        .store
        .list_audit_events(
            &ns,
            params.entity_type.as_deref(),
            params.entity_id.as_deref(),
            limit,
        )
        .await?;
    Ok(Json(events))
}

/// GET /api/v1/contradictions — list contradiction ledger entries for the namespace.
pub async fn list_contradictions(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<ContradictionListParams>,
) -> Result<Json<Vec<anima_db::store::ContradictionEntry>>, AppError> {
    let limit = params.limit.unwrap_or(50).clamp(1, 500);
    let offset = params.offset.unwrap_or(0);
    let entries = state.store.list_contradictions(&ns, limit, offset).await?;
    Ok(Json(entries))
}

/// GET /api/v1/memories/{id}/history — get supersession chain for a memory.
pub async fn get_memory_history(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
) -> Result<Json<Vec<anima_db::store::SupersessionLink>>, AppError> {
    let chain = state.store.get_supersession_chain(&ns, &id).await?;
    Ok(Json(chain))
}

pub async fn upsert_procedure_revision(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(name): Path<String>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ProcedureRevisionRequest>,
) -> Result<Json<anima_db::store::ProcedureRevision>, AppError> {
    if name.trim().is_empty() {
        return Err(AppError::BadRequest("procedure name cannot be empty".into()));
    }
    let actor = headers
        .get("X-Anima-Principal")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("api");
    let revision = state
        .store
        .upsert_procedure_revision(
            &ns,
            &name,
            &req.operation,
            &req.spec,
            Some(actor),
            req.reason.as_deref(),
        )
        .await?;
    Ok(Json(revision))
}

pub async fn list_procedure_revisions(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(name): Path<String>,
    Query(params): Query<RevisionListParams>,
) -> Result<Json<Vec<anima_db::store::ProcedureRevision>>, AppError> {
    if name.trim().is_empty() {
        return Err(AppError::BadRequest("procedure name cannot be empty".into()));
    }
    let limit = params.limit.unwrap_or(50).clamp(1, 500);
    let revisions = state
        .store
        .list_procedure_revisions(&ns, &name, limit)
        .await?;
    Ok(Json(revisions))
}

pub async fn create_identity_entity(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CreateIdentityEntityRequest>,
) -> Result<Json<anima_db::store::IdentityEntity>, AppError> {
    if req.canonical_name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "canonical_name cannot be empty".into(),
        ));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let entity = state
        .store
        .upsert_identity_entity(
            &ns,
            &req.canonical_name,
            req.language.as_deref(),
            req.confidence.unwrap_or(0.8).clamp(0.0, 1.0),
            metadata,
        )
        .await?;
    Ok(Json(entity))
}

pub async fn add_identity_alias(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
    Json(req): Json<AddIdentityAliasRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if req.alias.trim().is_empty() {
        return Err(AppError::BadRequest("alias cannot be empty".into()));
    }
    state
        .store
        .add_identity_alias(
            &ns,
            &id,
            &req.alias,
            req.language.as_deref(),
            req.confidence.unwrap_or(0.8).clamp(0.0, 1.0),
        )
        .await?;
    Ok(Json(serde_json::json!({
        "entity_id": id,
        "alias": req.alias,
        "linked": true
    })))
}

pub async fn resolve_identity(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<IdentityResolveParams>,
) -> Result<Json<anima_db::store::IdentityResolution>, AppError> {
    if params.query.trim().is_empty() {
        return Err(AppError::BadRequest("query cannot be empty".into()));
    }
    let resolution = state
        .store
        .resolve_identity(&ns, &params.query, params.limit.unwrap_or(5).clamp(1, 20))
        .await?;
    Ok(Json(resolution))
}

pub async fn create_plan(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CreatePlanRequest>,
) -> Result<Json<anima_db::store::PlanTrace>, AppError> {
    if req.goal.trim().is_empty() {
        return Err(AppError::BadRequest("goal cannot be empty".into()));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let plan = state
        .store
        .create_plan_trace(
            &ns,
            &req.goal,
            req.priority.unwrap_or(5).clamp(1, 10),
            req.due_at.as_deref(),
            metadata,
        )
        .await?;
    Ok(Json(plan))
}

pub async fn list_plans(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<ListPlansParams>,
) -> Result<Json<Vec<anima_db::store::PlanTrace>>, AppError> {
    let plans = state
        .store
        .list_plan_traces(
            &ns,
            params.status.as_deref(),
            params.limit.unwrap_or(100).clamp(1, 500),
        )
        .await?;
    Ok(Json(plans))
}

pub async fn add_plan_checkpoint(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
    Json(req): Json<AddPlanCheckpointRequest>,
) -> Result<Json<anima_db::store::PlanCheckpoint>, AppError> {
    if req.checkpoint_key.trim().is_empty() || req.title.trim().is_empty() {
        return Err(AppError::BadRequest(
            "checkpoint_key and title are required".into(),
        ));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let checkpoint = state
        .store
        .add_plan_checkpoint(
            &ns,
            &plan_id,
            &req.checkpoint_key,
            &req.title,
            req.order_index.unwrap_or(0).max(0),
            req.expected_by.as_deref(),
            metadata,
        )
        .await?;
    Ok(Json(checkpoint))
}

pub async fn list_plan_checkpoints(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
) -> Result<Json<Vec<anima_db::store::PlanCheckpoint>>, AppError> {
    let checkpoints = state.store.list_plan_checkpoints(&ns, &plan_id).await?;
    Ok(Json(checkpoints))
}

pub async fn update_plan_checkpoint_status(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(checkpoint_id): Path<String>,
    Json(req): Json<UpdatePlanCheckpointRequest>,
) -> Result<Json<anima_db::store::PlanCheckpoint>, AppError> {
    if req.status.trim().is_empty() {
        return Err(AppError::BadRequest("status cannot be empty".into()));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let checkpoint = state
        .store
        .update_plan_checkpoint_status(
            &ns,
            &checkpoint_id,
            &req.status,
            req.evidence.as_deref(),
            metadata,
        )
        .await?
        .ok_or_else(|| AppError::NotFound(format!("checkpoint {checkpoint_id} not found")))?;
    Ok(Json(checkpoint))
}

pub async fn set_plan_outcome(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
    Json(req): Json<SetPlanOutcomeRequest>,
) -> Result<Json<anima_db::store::PlanTrace>, AppError> {
    if req.status.trim().is_empty() || req.outcome.trim().is_empty() {
        return Err(AppError::BadRequest(
            "status and outcome are required".into(),
        ));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let plan = state
        .store
        .set_plan_outcome(
            &ns,
            &plan_id,
            &req.status,
            &req.outcome,
            req.outcome_confidence.map(|v| v.clamp(0.0, 1.0)),
            metadata,
        )
        .await?
        .ok_or_else(|| AppError::NotFound(format!("plan {plan_id} not found")))?;
    Ok(Json(plan))
}

pub async fn add_plan_recovery_branch(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
    Json(req): Json<AddPlanRecoveryBranchRequest>,
) -> Result<Json<anima_db::store::PlanRecoveryBranch>, AppError> {
    if req.branch_label.trim().is_empty() || req.trigger_reason.trim().is_empty() {
        return Err(AppError::BadRequest(
            "branch_label and trigger_reason are required".into(),
        ));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let branch = state
        .store
        .add_plan_recovery_branch(
            &ns,
            &plan_id,
            req.source_checkpoint_id.as_deref(),
            &req.branch_label,
            &req.trigger_reason,
            req.branch_plan,
            metadata,
        )
        .await?;
    Ok(Json(branch))
}

pub async fn list_plan_recovery_branches(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
) -> Result<Json<Vec<anima_db::store::PlanRecoveryBranch>>, AppError> {
    let branches = state.store.list_plan_recovery_branches(&ns, &plan_id).await?;
    Ok(Json(branches))
}

pub async fn resolve_plan_recovery_branch(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(branch_id): Path<String>,
    Json(req): Json<ResolvePlanRecoveryBranchRequest>,
) -> Result<Json<anima_db::store::PlanRecoveryBranch>, AppError> {
    if req.status.trim().is_empty() {
        return Err(AppError::BadRequest("status cannot be empty".into()));
    }
    let branch = state
        .store
        .resolve_plan_recovery_branch(&ns, &branch_id, &req.status, req.resolution_notes.as_deref())
        .await?
        .ok_or_else(|| AppError::NotFound(format!("branch {branch_id} not found")))?;
    Ok(Json(branch))
}

pub async fn bind_procedure_to_plan(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
    Json(req): Json<BindProcedureToPlanRequest>,
) -> Result<Json<anima_db::store::PlanProcedureBinding>, AppError> {
    if req.procedure_name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "procedure_name cannot be empty".into(),
        ));
    }
    let mut metadata = req.metadata;
    if let Some(ref mut m) = metadata {
        let _ = redact_metadata_value(m);
    }
    let binding = state
        .store
        .bind_procedure_to_plan(
            &ns,
            &plan_id,
            &req.procedure_name,
            req.binding_role.as_deref().unwrap_or("primary"),
            req.confidence.unwrap_or(0.75).clamp(0.0, 1.0),
            metadata,
        )
        .await?;
    Ok(Json(binding))
}

pub async fn list_plan_procedure_bindings(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(plan_id): Path<String>,
) -> Result<Json<Vec<anima_db::store::PlanProcedureBinding>>, AppError> {
    let bindings = state
        .store
        .list_plan_procedure_bindings(&ns, &plan_id)
        .await?;
    Ok(Json(bindings))
}

pub async fn upsert_state_transition(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<UpsertTransitionRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if req.from_memory_id.trim().is_empty() || req.to_memory_id.trim().is_empty() {
        return Err(AppError::BadRequest(
            "from_memory_id and to_memory_id are required".into(),
        ));
    }
    if req.from_memory_id == req.to_memory_id {
        return Err(AppError::BadRequest(
            "from_memory_id and to_memory_id must be different".into(),
        ));
    }
    let from = state
        .store
        .get(&req.from_memory_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {} not found", req.from_memory_id)))?;
    let to = state
        .store
        .get(&req.to_memory_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {} not found", req.to_memory_id)))?;
    if !namespace_contains(&ns, &from.namespace) || !namespace_contains(&ns, &to.namespace) {
        return Err(AppError::Forbidden(
            "transition endpoints can only reference memories in current namespace".into(),
        ));
    }
    state
        .store
        .upsert_state_transition(
            &ns,
            &req.from_memory_id,
            &req.to_memory_id,
            req.transition_type.as_deref().unwrap_or("temporal"),
            req.confidence.unwrap_or(0.75).clamp(0.0, 1.0),
            req.evidence.as_deref(),
        )
        .await?;
    Ok(Json(serde_json::json!({
        "ok": true,
        "from_memory_id": req.from_memory_id,
        "to_memory_id": req.to_memory_id
    })))
}

pub async fn list_state_transitions(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<ListTransitionsParams>,
) -> Result<Json<Vec<anima_db::store::StateTransitionEdge>>, AppError> {
    let transitions = state
        .store
        .list_state_transitions(&ns, params.limit.unwrap_or(100).clamp(1, 500))
        .await?;
    Ok(Json(transitions))
}

pub async fn simulate_counterfactual(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CounterfactualRequest>,
) -> Result<Json<anima_db::store::CounterfactualSimulation>, AppError> {
    let intervention = req.intervention.trim();
    if intervention.is_empty() {
        return Err(AppError::BadRequest(
            "intervention cannot be empty".into(),
        ));
    }

    let mut seed_ids = Vec::new();
    if !req.seed_memory_ids.is_empty() {
        for seed in req.seed_memory_ids {
            let memory = state
                .store
                .get(&seed)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("seed memory {seed} not found")))?;
            if !namespace_contains(&ns, &memory.namespace) {
                return Err(AppError::Forbidden(format!(
                    "seed memory {seed} is outside namespace '{}'",
                    ns.as_str()
                )));
            }
            seed_ids.push(seed);
        }
    } else {
        let query = req.query.clone().unwrap_or_else(|| intervention.to_string());
        let embedding = embed_query_cached(&state, &query)?;
        let cf_scorer_config = state.scorer_config.read().await.clone();
        let scored = state
            .store
            .search(
                &embedding,
                &query,
                &ns,
                &SearchMode::Hybrid,
                8,
                &cf_scorer_config,
            )
            .await?;
        for sr in scored {
            if sr.score >= 0.15 {
                seed_ids.push(sr.memory_id);
            }
        }
    }
    seed_ids.sort();
    seed_ids.dedup();

    let simulation = state
        .store
        .simulate_counterfactual(
            &ns,
            intervention,
            &seed_ids,
            req.max_hops.unwrap_or(2),
            req.top_k.unwrap_or(6),
            req.include_correlational,
            req.include_transitions,
        )
        .await?;
    Ok(Json(simulation))
}

pub async fn capture_correction(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CorrectionRequest>,
) -> Result<Json<CorrectionResponse>, AppError> {
    if req.target_memory_id.trim().is_empty() {
        return Err(AppError::BadRequest("target_memory_id is required".into()));
    }
    let (corrected_content, correction_metadata, _) =
        redact_content_and_metadata(&req.corrected_content, req.metadata.clone());
    if corrected_content.trim().is_empty() {
        return Err(AppError::BadRequest("corrected_content cannot be empty".into()));
    }

    let target = state
        .store
        .get(&req.target_memory_id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {} not found", req.target_memory_id)))?;

    let ns_prefix = ns.as_str();
    let in_namespace = target.namespace == ns_prefix
        || target.namespace.starts_with(&format!("{ns_prefix}/"));
    if !in_namespace {
        return Err(AppError::BadRequest(
            "target memory does not belong to the current namespace".into(),
        ));
    }

    let target_ns = anima_core::namespace::Namespace::parse(&target.namespace)
        .map_err(|e| AppError::BadRequest(format!("invalid target namespace: {e}")))?;
    let embedding = state
        .embedder
        .embed(&corrected_content)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    let tier = target
        .metadata
        .as_ref()
        .and_then(|m| m.get("tier"))
        .and_then(|v| v.as_i64())
        .unwrap_or(1);
    let mut correction_meta = serde_json::json!({
        "tier": tier,
        "confidence": 1.0,
        "correction": {
            "target_memory_id": target.id.clone(),
            "reason": req.reason.clone(),
            "captured_at": chrono::Utc::now().to_rfc3339(),
        }
    });
    if let Some(extra) = correction_metadata.clone() {
        correction_meta["correction"]["metadata"] = extra;
    }

    let memory = Memory::new(
        target.namespace.clone(),
        corrected_content.clone(),
        Some(correction_meta.clone()),
        req.tags.clone().unwrap_or_else(|| target.tags.clone()),
        Some(target.memory_type.clone()),
    );
    let new_id = memory.id.clone();
    state.store.insert(&memory, &embedding).await?;
    ingest_identity_hints(&state, &target_ns, &memory.content, 0.95, Some(&new_id)).await;

    if let Some(importance) = req.importance {
        state
            .store
            .set_importance(&new_id, importance.clamp(1, 10))
            .await?;
    } else {
        state
            .store
            .set_importance(&new_id, target.importance)
            .await?;
    }

    state
        .store
        .mark_superseded(&req.target_memory_id, &new_id)
        .await?;

    let correction_event = state
        .store
        .record_correction_event(
            &target_ns,
            &req.target_memory_id,
            &new_id,
            req.reason.as_deref(),
            correction_metadata.clone(),
        )
        .await?;

    state
        .store
        .record_contradiction_resolution(
            &target_ns,
            &req.target_memory_id,
            &new_id,
            "user_correction_supersede",
            Some(serde_json::json!({
                "source": "api",
                "reason": req.reason.clone(),
                "correction_event_id": correction_event.id.clone(),
                "correction_metadata": correction_metadata.clone(),
            })),
        )
        .await?;

    // Immediate belief re-scoring signal: corrected old belief becomes negative feedback.
    let old_conf = target
        .metadata
        .as_ref()
        .and_then(|m| m.get("confidence"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.7)
        .clamp(0.0, 1.0);
    let kind = match tier {
        3 => PredictionKind::Deduction,
        4 => PredictionKind::Induction,
        _ => PredictionKind::Extraction,
    };
    let _ = state
        .store
        .record_calibration_observation(
            &target_ns,
            kind,
            Some(&req.target_memory_id),
            old_conf,
            Some(0.0),
            Some(serde_json::json!({
                "source": "correction_api",
                "reason": req.reason.clone(),
                "new_memory_id": new_id.clone(),
            })),
        )
        .await;
    let _ = state
        .store
        .record_calibration_observation(
            &target_ns,
            kind,
            Some(&new_id),
            1.0,
            Some(1.0),
            Some(serde_json::json!({
                "source": "correction_api",
                "reason": req.reason.clone(),
                "supersedes": req.target_memory_id.clone(),
            })),
        )
        .await;
    let recalibrated = state.store.recompute_calibration_models().await.is_ok();

    if let Some(processor) = &state.processor {
        processor.enqueue(crate::processor::ProcessingJob::Induce {
            namespace: target_ns.as_str().to_string(),
        });
    }

    Ok(Json(CorrectionResponse {
        correction_event_id: correction_event.id,
        superseded_memory_id: req.target_memory_id,
        new_memory_id: new_id,
        contradiction_logged: true,
        recalibrated,
    }))
}

pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let target = state
        .store
        .get(&id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("memory {id} not found")))?;
    if !namespace_contains(&ns, &target.namespace) {
        return Err(AppError::Forbidden(format!(
            "memory {id} is not writable in namespace '{}'",
            ns.as_str()
        )));
    }

    let deleted = state.store.hard_delete(&id).await?;

    if !deleted {
        return Err(AppError::NotFound(format!("memory {id} not found")));
    }

    Ok(Json(serde_json::json!({
        "id": id,
        "deleted": true
    })))
}

pub async fn purge_deleted_memories(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let count = state.store.purge_deleted().await?;
    Ok(Json(serde_json::json!({ "purged": count })))
}

pub async fn add_working_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<AddWorkingMemoryRequest>,
) -> Result<Json<AddWorkingMemoryResponse>, AppError> {
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content cannot be empty".into()));
    }
    let ttl_secs = req.ttl_seconds.unwrap_or(7200).clamp(30, 604800);
    let expires_at = (Utc::now() + Duration::seconds(ttl_secs as i64)).to_rfc3339();
    let entry = state
        .store
        .add_working_memory(
            &ns,
            req.content.trim(),
            req.provisional_score.unwrap_or(0.5),
            req.metadata,
            req.conversation_id.as_deref(),
            Some(&expires_at),
        )
        .await?;
    Ok(Json(AddWorkingMemoryResponse {
        entry: to_working_memory_dto(entry),
    }))
}

pub async fn list_working_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<ListWorkingMemoryParams>,
) -> Result<Json<ListWorkingMemoryResponse>, AppError> {
    let entries = state
        .store
        .list_working_memories(
            &ns,
            params.status.as_deref(),
            params.conversation_id.as_deref(),
            params.limit.unwrap_or(200).min(1000),
        )
        .await?;
    Ok(Json(ListWorkingMemoryResponse {
        entries: entries.into_iter().map(to_working_memory_dto).collect(),
    }))
}

pub async fn commit_working_memory(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CommitWorkingMemoryRequest>,
) -> Result<Json<CommitWorkingMemoryResponse>, AppError> {
    let limit = req.limit.unwrap_or(64).min(512);
    let min_score = req.min_score.unwrap_or(0.55).clamp(0.0, 1.0);

    if req.r#async {
        if let Some(processor) = &state.processor {
            processor.enqueue(crate::processor::ProcessingJob::CommitWorkingMemory {
                namespace: ns.as_str().to_string(),
                conversation_id: req.conversation_id.clone(),
                limit,
                min_score,
            });
            return Ok(Json(CommitWorkingMemoryResponse {
                status: "queued".to_string(),
                evaluated: 0,
                committed: 0,
                committed_memory_ids: vec![],
            }));
        }
        return Err(AppError::Internal("background processor not available".into()));
    }

    let result = crate::processor::commit_working_memory_sync(
        &state.store,
        &state.embedder,
        ns.as_str(),
        req.conversation_id.as_deref(),
        limit,
        min_score,
    )
    .await
    .map_err(|e| AppError::Internal(format!("working-memory commit failed: {e}")))?;

    if !result.committed_memory_ids.is_empty() {
        enqueue_reflect_batched(&state, ns.as_str(), result.committed_memory_ids.clone());
    }

    Ok(Json(CommitWorkingMemoryResponse {
        status: "completed".to_string(),
        evaluated: result.evaluated,
        committed: result.committed_memory_ids.len(),
        committed_memory_ids: result.committed_memory_ids,
    }))
}

pub async fn reconsolidate_memories(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<ReconsolidateRequest>,
) -> Result<Json<ReconsolidateResponse>, AppError> {
    if req.memory_ids.is_empty() {
        return Err(AppError::BadRequest("memory_ids cannot be empty".into()));
    }

    if req.r#async {
        if let Some(processor) = &state.processor {
            processor.enqueue(crate::processor::ProcessingJob::Reconsolidate {
                namespace: ns.as_str().to_string(),
                memory_ids: req.memory_ids,
            });
            return Ok(Json(ReconsolidateResponse {
                status: "queued".to_string(),
                processed: 0,
                superseded: 0,
            }));
        }
        return Err(AppError::Internal("background processor not available".into()));
    }

    let result = crate::processor::reconsolidate_sync(&state.store, ns.as_str(), &req.memory_ids)
        .await
        .map_err(|e| AppError::Internal(format!("reconsolidation failed: {e}")))?;
    Ok(Json(ReconsolidateResponse {
        status: "completed".to_string(),
        processed: result.processed,
        superseded: result.superseded,
    }))
}

pub async fn run_retention(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<RetentionRunRequest>,
) -> Result<Json<RetentionRunResponse>, AppError> {
    let result = crate::processor::run_retention_sync(
        &state.store,
        Some(ns.as_str()),
        req.limit_per_namespace.unwrap_or(1000).min(5000),
    )
    .await
    .map_err(|e| AppError::Internal(format!("retention failed: {e}")))?;

    Ok(Json(RetentionRunResponse {
        processed: result.processed,
        softened: result.softened,
    }))
}

pub async fn list_memories(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<ListParams>,
) -> Result<Json<ListMemoriesResponse>, AppError> {
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(50).min(200);
    let status_filter = params.status.as_deref();
    let type_filter = params.memory_type.as_deref();
    let category_filter = params.category.as_deref();

    let (memories, total) = state.store.list(&ns, status_filter, type_filter, category_filter, offset, limit).await?;

    let memory_responses: Vec<MemoryResponse> = memories
        .into_iter()
        .map(|m| {
            let (content, metadata, _) = redact_content_and_metadata(&m.content, m.metadata);
            MemoryResponse {
                id: m.id,
                namespace: m.namespace,
                content,
                metadata,
                tags: m.tags,
                memory_type: m.memory_type,
                category: m.category.clone(),
                confidence: m.confidence,
                source: m.source.clone(),
                status: m.status.as_str().to_string(),
                created_at: m.created_at.to_rfc3339(),
                updated_at: m.updated_at.to_rfc3339(),
                access_count: m.access_count,
                importance: m.importance,
            }
        })
        .collect();

    Ok(Json(ListMemoriesResponse {
        memories: memory_responses,
        total,
        offset,
        limit,
    }))
}

pub async fn get_stats(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
) -> Result<Json<anima_db::store::NamespaceStats>, AppError> {
    let stats = state.store.stats(&ns).await?;
    Ok(Json(stats))
}

pub async fn list_namespaces(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<anima_db::store::NamespaceInfo>>, AppError> {
    let namespaces = state.store.list_namespaces().await?;
    Ok(Json(namespaces))
}

pub async fn get_graph(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<GraphParams>,
) -> Result<Json<anima_db::store::GraphData>, AppError> {
    let threshold = params.threshold.unwrap_or(0.5);
    let max_nodes = params.limit.unwrap_or(100).min(500);
    let graph = state.store.similarity_graph(&ns, threshold, max_nodes).await?;
    Ok(Json(graph))
}

pub async fn get_embeddings(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<EmbeddingsResponse>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(300)
        .min(1000);

    let raw_all = state.store.get_raw_embeddings(&ns, limit).await?;

    if raw_all.is_empty() {
        return Ok(Json(EmbeddingsResponse { points: vec![] }));
    }

    // Use the dominant (most frequent) embedding dimension; discard outliers from old models
    let d = {
        let mut counts = std::collections::HashMap::<usize, usize>::new();
        for r in &raw_all {
            if r.embedding.len() > 0 {
                *counts.entry(r.embedding.len()).or_insert(0) += 1;
            }
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(d, _)| d).unwrap_or(0)
    };

    if d == 0 {
        return Ok(Json(EmbeddingsResponse { points: vec![] }));
    }

    let raw: Vec<_> = raw_all
        .into_iter()
        .filter(|r| r.embedding.len() == d)
        .collect();

    if raw.is_empty() {
        return Ok(Json(EmbeddingsResponse { points: vec![] }));
    }

    let n = raw.len();

    // Build n×d matrix
    let flat: Vec<f32> = raw.iter().flat_map(|r| r.embedding.iter().cloned()).collect();
    let mut m = Array2::<f32>::from_shape_vec((n, d), flat)
        .map_err(|e| AppError::Internal(format!("pca shape error: {e}")))?;

    // Center columns (subtract per-dimension mean)
    let mean = m.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(d));
    for mut row in m.outer_iter_mut() {
        row -= &mean;
    }

    // Power-iteration PCA for top-3 components
    let mut coords = vec![[0f32; 3]; n];
    for k in 0..3usize {
        // Initialize with uniform vector (avoids basis-vector bias)
        let mut v = Array1::<f32>::from_elem(d, 1.0 / (d as f32).sqrt());

        for _ in 0..60 {
            let mv = m.dot(&v);         // shape [n]
            let mtmv = m.t().dot(&mv);  // shape [d]
            let norm: f32 = mtmv.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                v = mtmv.mapv(|x| x / norm);
            }
        }

        let scores = m.dot(&v); // shape [n]
        for (i, &s) in scores.iter().enumerate() {
            coords[i][k] = s;
        }

        // Deflate: M = M − scores ⊗ v
        for (i, mut row) in m.outer_iter_mut().enumerate() {
            let s = scores[i];
            row.zip_mut_with(&v, |a, &b| *a -= s * b);
        }
    }

    let points = raw
        .iter()
        .zip(coords.iter())
        .map(|(r, &[x, y, z])| EmbeddingPointDto {
            id: r.id.clone(),
            content: redact_sensitive_text(&r.content).0,
            memory_type: r.memory_type.clone(),
            x,
            y,
            z,
        })
        .collect();

    Ok(Json(EmbeddingsResponse { points }))
}

pub async fn top_accessed(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Query(params): Query<AccessRankingParams>,
) -> Result<Json<Vec<MemoryResponse>>, AppError> {
    let ascending = params.order.as_deref() == Some("asc");
    let limit = params.limit.unwrap_or(20).min(100);
    let memories = state.store.access_ranking(&ns, ascending, limit).await?;

    let responses: Vec<MemoryResponse> = memories
        .into_iter()
        .map(|m| {
            let (content, metadata, _) = redact_content_and_metadata(&m.content, m.metadata);
            MemoryResponse {
                id: m.id,
                namespace: m.namespace,
                content,
                metadata,
                tags: m.tags,
                memory_type: m.memory_type,
                category: m.category.clone(),
                confidence: m.confidence,
                source: m.source.clone(),
                status: m.status.as_str().to_string(),
                created_at: m.created_at.to_rfc3339(),
                updated_at: m.updated_at.to_rfc3339(),
                access_count: m.access_count,
                importance: m.importance,
            }
        })
        .collect();

    Ok(Json(responses))
}

/// Resolve an optional client-provided LLM config, falling back to the
/// operation's profile (from `[profiles]` + `[routing]`) or server-side `[llm]` config.
fn resolve_llm_config(llm: Option<LlmConfig>, state: &AppState, operation: &str) -> LlmConfig {
    let profile = state.profile_for(operation);
    let profile_name = match operation {
        "ask" => state.resolved_profiles.routing.ask.as_deref(),
        "chat" => state.resolved_profiles.routing.chat.as_deref(),
        _ => None,
    }.unwrap_or("default");

    match llm {
        Some(mut client_llm) => {
            let server_url = profile.map(|p| &p.base_url).unwrap_or(&state.config.llm.base_url);
            if client_llm.base_url.contains("localhost") || client_llm.base_url.contains("127.0.0.1") {
                if !server_url.contains("localhost") && !server_url.contains("127.0.0.1") {
                    client_llm.base_url = server_url.clone();
                }
            }
            if client_llm.api_key.as_ref().map_or(true, |k| k.is_empty()) {
                client_llm.api_key = profile
                    .and_then(|p| p.resolve_api_key(profile_name))
                    .or_else(|| Some(state.config.llm.api_key.clone()).filter(|k| !k.is_empty()))
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()));
            }
            if client_llm.model.is_empty() {
                client_llm.model = profile
                    .map(|p| p.model.clone())
                    .unwrap_or_else(|| state.config.llm.model.clone());
            }
            client_llm
        }
        None => {
            if let Some(p) = profile {
                LlmConfig {
                    base_url: p.base_url.clone(),
                    model: p.model.clone(),
                    api_key: p.resolve_api_key(profile_name),
                    temperature: None,
                    max_tokens: None,
                    system_prompt: None,
                    vision: false,
                    tool_use: false,
                    streaming: false,
                }
            } else {
                let cfg = &state.config.llm;
                let api_key = Some(cfg.api_key.clone())
                    .filter(|k| !k.is_empty())
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()));
                LlmConfig {
                    base_url: cfg.base_url.clone(),
                    model: cfg.model.clone(),
                    api_key,
                    temperature: None,
                    max_tokens: None,
                    system_prompt: None,
                    vision: false,
                    tool_use: false,
                    streaming: false,
                }
            }
        }
    }
}

pub async fn chat(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(mut req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    if req.message.trim().is_empty() {
        return Err(AppError::BadRequest("message cannot be empty".into()));
    }
    if is_likely_noisy_or_adversarial_input(&req.message) {
        return Ok(Json(make_chat_response(
            &ns,
            "Input looks noisy or adversarial. Please rewrite your message with a clear, specific request.".to_string(),
            vec![],
            vec![],
            req.mode.clone(),
            false,
        )));
    }

    let resolved_llm = resolve_llm_config(req.llm.take(), &state, "chat");
    req.llm = Some(resolved_llm);

    let client = reqwest::Client::new();

    match req.mode.as_str() {
        "tool" if req.llm.as_ref().is_some_and(|l| l.tool_use) => chat_tool_mode(&state, &client, &ns, &req).await,
        _ => chat_rag_mode(&state, &client, &ns, &req).await,
    }
}

/// Expand search results by traversing graph neighbors for additional context.
/// Returns (memory_context_string, all_memory_contexts) with deduplication.
async fn expand_with_graph_neighbors(
    state: &AppState,
    scored: &[anima_core::search::ScoredResult],
    max_total: usize,
) -> Result<(String, Vec<MemoryContext>), AppError> {
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut memories_used = Vec::new();
    let mut memory_context = String::new();

    // Phase 1: Direct search results
    for sr in scored {
        if let Some(mem) = state.store.get(&sr.memory_id).await? {
            let (content, _) = redact_sensitive_text(&mem.content);
            seen_ids.insert(mem.id.clone());
            memory_context.push_str(&format!("- [{}] {}\n", mem.memory_type, content));
            memories_used.push(MemoryContext {
                id: mem.id,
                content,
                score: sr.score,
                source: Some("search".into()),
                memory_type: mem.memory_type,
                importance: mem.importance,
            });
        }
    }

    // Phase 2: Graph neighbor expansion
    if memories_used.len() < max_total {
        let remaining = max_total - memories_used.len();
        let mut neighbor_candidates: Vec<(anima_core::memory::Memory, f64)> = Vec::new();

        for sr in scored {
            let neighbors = state
                .store
                .find_neighbors(&sr.memory_id, 3, 0.5)
                .await?;
            for (mem, sim) in neighbors {
                if !seen_ids.contains(&mem.id) {
                    seen_ids.insert(mem.id.clone());
                    neighbor_candidates.push((mem, sim));
                }
            }
        }

        // Sort by similarity descending, take top remaining
        neighbor_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbor_candidates.truncate(remaining);

        for (mem, sim) in neighbor_candidates {
            let (content, _) = redact_sensitive_text(&mem.content);
            memory_context.push_str(&format!("- [{}] {}\n", mem.memory_type, content));
            memories_used.push(MemoryContext {
                id: mem.id,
                content,
                score: sim,
                source: Some("graph".into()),
                memory_type: mem.memory_type,
                importance: mem.importance,
            });
        }
    }

    Ok((memory_context, memories_used))
}

async fn record_retrieval_observations(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    scored: &[anima_core::search::ScoredResult],
    threshold: f64,
    source: &str,
) {
    for (rank, sr) in scored.iter().enumerate() {
        let outcome = if sr.score >= threshold { 1.0 } else { 0.0 };
        let _ = state
            .store
            .record_calibration_observation(
                ns,
                PredictionKind::RetrievalRelevance,
                Some(&sr.memory_id),
                sr.score,
                Some(outcome),
                Some(serde_json::json!({
                    "source": source,
                    "rank": rank + 1,
                    "threshold": threshold,
                    "vector_score": sr.vector_score,
                    "keyword_score": sr.keyword_score,
                })),
            )
            .await;
    }
}

fn procedure_confidence_hint(fn_name: &str, args: &serde_json::Value) -> f64 {
    let arg_len = args.to_string().len();
    let arg_bonus: f64 = if arg_len > 400 {
        0.03
    } else if arg_len < 8 {
        -0.06
    } else {
        0.0
    };

    let base: f64 = match fn_name {
        "memory_search" => 0.84,
        "memory_add" => 0.79,
        "memory_update" => 0.78,
        _ => 0.55,
    };
    (base + arg_bonus).clamp(0.05, 0.99)
}

async fn record_procedure_observation(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    fn_name: &str,
    predicted_confidence: f64,
    outcome: f64,
    metadata: serde_json::Value,
) {
    let _ = state
        .store
        .record_calibration_observation(
            ns,
            PredictionKind::ProcedureSelection,
            None,
            predicted_confidence,
            Some(outcome),
            Some(serde_json::json!({
                "procedure": fn_name,
                "meta": metadata,
            })),
        )
        .await;
}

fn safe_fallback_chat_reply(
    user_message: &str,
    memories_used: &[MemoryContext],
    failure: &str,
) -> String {
    tracing::warn!("Chat LLM fallback triggered: {failure}");
    if memories_used.is_empty() {
        return format!(
            "Safe fallback mode is active because the LLM backend is unavailable. I can still store and retrieve memories. Please retry your request: \"{user_message}\"."
        );
    }

    let context = memories_used
        .iter()
        .take(3)
        .map(|m| format!("[{}] {}", m.memory_type, m.content))
        .collect::<Vec<_>>()
        .join(" | ");

    format!(
        "Safe fallback mode is active because the LLM backend is unavailable. Relevant memories: {context}. Please retry for a full composed answer."
    )
}

fn is_likely_noisy_or_adversarial_input(text: &str) -> bool {
    let s = text.trim();
    if s.is_empty() {
        return false;
    }
    if s.len() > 4096 {
        return true;
    }

    // Excessive repeated characters often indicates junk input.
    let mut run = 1usize;
    let mut prev = '\0';
    for ch in s.chars() {
        if ch == prev {
            run += 1;
            if run >= 14 {
                return true;
            }
        } else {
            prev = ch;
            run = 1;
        }
    }

    let total_chars = s.chars().count().max(1);
    let alnum_chars = s.chars().filter(|c| c.is_alphanumeric()).count();
    let alnum_ratio = alnum_chars as f64 / total_chars as f64;
    if total_chars >= 24 && alnum_ratio < 0.20 {
        return true;
    }

    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.len() >= 8 {
        let unique = tokens.iter().collect::<std::collections::HashSet<_>>().len();
        let unique_ratio = unique as f64 / tokens.len() as f64;
        if unique_ratio < 0.25 {
            return true;
        }
    }

    false
}

fn should_escalate_uncertainty(top_score: Option<f64>, result_count: usize) -> bool {
    const MIN_TOP_SCORE: f64 = 0.33;
    const MIN_RESULT_COUNT: usize = 2;
    result_count < MIN_RESULT_COUNT || top_score.unwrap_or(0.0) < MIN_TOP_SCORE
}

/// Enriched retrieval pipeline shared by /chat and /ask.
/// Performs hybrid search, keyword expansion, entity-linked retrieval,
/// episode expansion, and graph neighbor expansion.
async fn enriched_retrieval(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    query: &str,
    search_limit: usize,
    max_results: usize,
    label: &str,
) -> Result<(String, Vec<MemoryContext>), AppError> {
    let embedding = embed_query_cached(state, query)?;
    let scorer_cfg = state.scorer_config.read().await.clone();

    // 1. Primary hybrid search
    let results = state.store
        .search(&embedding, query, ns, &SearchMode::Hybrid, search_limit, &scorer_cfg)
        .await
        .map_err(|e| AppError::Database(e.to_string()))?;
    record_retrieval_observations(state, ns, &results, 0.15, label).await;

    // 2. Keyword expansion (zero-latency, pure DB queries)
    let keyword_queries = extract_keyword_queries(query);
    let mut expanded_results = Vec::new();
    for kq in &keyword_queries {
        let kq_embedding = embed_query_cached(state, kq)?;
        if let Ok(kw_results) = state.store
            .search(&kq_embedding, kq, ns, &SearchMode::Hybrid, search_limit, &scorer_cfg)
            .await
        {
            expanded_results.extend(kw_results);
        }
    }

    // 3. Entity-aware expansion
    let entities = extract_candidate_entities(query);
    let mut resolved_entity_queries: Vec<String> = Vec::new();
    for entity in &entities {
        match state.store.resolve_identity(ns, entity, 3).await {
            Ok(resolution) => {
                if resolution.candidates.is_empty() || resolution.best_confidence < 0.62 {
                    resolved_entity_queries.push(entity.clone());
                } else {
                    let mut added_bases = HashSet::new();
                    for c in &resolution.candidates {
                        let base = c.canonical_name
                            .split_whitespace().next().unwrap_or(&c.canonical_name)
                            .trim_end_matches("'s").trim_end_matches("\u{2019}s")
                            .to_string();
                        if added_bases.insert(base.to_ascii_lowercase()) {
                            resolved_entity_queries.push(base);
                        }
                    }
                }
            }
            Err(_) => {
                resolved_entity_queries.push(entity.clone());
            }
        }
    }
    if resolved_entity_queries.is_empty() {
        resolved_entity_queries = entities;
    }
    let mut dedup = HashSet::new();
    resolved_entity_queries.retain(|q| dedup.insert(q.to_ascii_lowercase()));

    // Entity keyword searches
    let mut entity_results = Vec::new();
    for entity in &resolved_entity_queries {
        if let Ok(kw_results) = state.store
            .search(&embedding, entity, ns, &SearchMode::Keyword, search_limit / 2, &scorer_cfg)
            .await
        {
            entity_results.extend(kw_results);
        }
    }

    // 4. Merge all sources: keep best score per memory_id
    let mut best_scores: HashMap<String, f64> = HashMap::new();
    for sr in results.iter().chain(expanded_results.iter()).chain(entity_results.iter()) {
        let entry = best_scores.entry(sr.memory_id.clone()).or_insert(0.0);
        if sr.score > *entry {
            *entry = sr.score;
        }
    }

    let mut scored_ids: Vec<(String, f64)> = best_scores.into_iter().collect();
    scored_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut all_mems: Vec<(anima_core::memory::Memory, f64)> = Vec::new();
    let mut seen_contents: HashSet<String> = HashSet::new();

    for (memory_id, score) in scored_ids {
        if let Ok(Some(memory)) = state.store.get(&memory_id).await {
            if seen_contents.insert(memory.content.clone()) {
                all_mems.push((memory, score));
            }
        }
    }

    // Drop results scoring below 25% of the top hit
    if let Some((_, top_score)) = all_mems.first() {
        let cutoff = top_score * 0.25;
        all_mems.retain(|(_, s)| *s >= cutoff);
    }
    all_mems.truncate(max_results);

    // 5. Episode expansion
    {
        let mut episode_ids_seen: HashSet<String> = HashSet::new();
        for (mem, score) in &all_mems {
            if *score < 0.3 { continue; }
            if let Some(ep_id) = &mem.episode_id {
                if !ep_id.is_empty() {
                    episode_ids_seen.insert(ep_id.clone());
                }
            } else if let Some(meta) = &mem.metadata {
                if let Some(session) = meta.get("session").and_then(|v| v.as_str()) {
                    episode_ids_seen.insert(session.to_string());
                }
            }
        }
        let mut episodes_expanded = 0;
        for ep_id in &episode_ids_seen {
            if episodes_expanded >= 3 { break; }
            if let Ok(co_mems) = state.store.find_by_episode(ns, ep_id, 10).await {
                for mem in co_mems {
                    if seen_contents.insert(mem.content.clone()) {
                        all_mems.push((mem, 0.3));
                    }
                }
            }
            episodes_expanded += 1;
        }
    }

    // 6. Entity-linked retrieval
    {
        let mut entity_ids: Vec<String> = Vec::new();
        for entity in &resolved_entity_queries {
            if let Ok(resolution) = state.store.resolve_identity(ns, entity, 3).await {
                for c in &resolution.candidates {
                    if c.score >= 0.3 {
                        entity_ids.push(c.entity_id.clone());
                    }
                }
            }
        }
        entity_ids.sort();
        entity_ids.dedup();
        if !entity_ids.is_empty() {
            if let Ok(linked_ids) = state.store.find_memories_by_entity_ids(ns, &entity_ids, 30).await {
                for mid in linked_ids {
                    if let Ok(Some(mem)) = state.store.get(&mid).await {
                        if seen_contents.insert(mem.content.clone()) {
                            all_mems.push((mem, 0.28));
                        }
                    }
                }
            }
        }
    }

    // 7. Build context string and MemoryContext vec
    let mut memory_context = String::new();
    let mut memories_used = Vec::new();
    for (mem, score) in &all_mems {
        let (content, _) = redact_sensitive_text(&mem.content);
        memory_context.push_str(&format!("- [{}] {}\n", mem.memory_type, content));
        memories_used.push(MemoryContext {
            id: mem.id.clone(),
            content,
            score: *score,
            source: Some("search".into()),
            memory_type: mem.memory_type.clone(),
            importance: mem.importance,
        });
    }

    Ok((memory_context, memories_used))
}

async fn chat_rag_mode(
    state: &AppState,
    client: &reqwest::Client,
    ns: &anima_core::namespace::Namespace,
    req: &ChatRequest,
) -> Result<Json<ChatResponse>, AppError> {
    let llm = req.llm.as_ref().expect("llm must be resolved before calling chat_rag_mode");

    // 1. Enriched retrieval (same pipeline as /ask)
    let (memory_context, memories_used) =
        enriched_retrieval(state, ns, &req.message, 20, 15, "chat_rag").await?;

    // 2. Build messages with memory-enriched system prompt
    let today = chrono::Utc::now().format("%A, %B %-d, %Y");
    let base_prompt = llm.system_prompt.as_deref().unwrap_or(
        "You are a helpful assistant."
    );
    let base_prompt = format!("{base_prompt}\n\nToday's date: {today}.");
    let system_prompt = if memory_context.is_empty() {
        base_prompt
    } else {
        format!(
            "{base_prompt}\n\nThe following are memories about the user. Only reference them if they are \
             directly relevant to what the user is currently asking about. Do not mention or allude to \
             them otherwise.\n\n{memory_context}"
        )
    };

    let mut messages = vec![serde_json::json!({"role": "system", "content": system_prompt})];
    for msg in &req.history {
        messages.push(serde_json::json!({"role": msg.role, "content": msg.content}));
    }
    messages.push(build_user_message(&req.message, &req.attachments, llm.vision));

    // 3. Call LLM
    let (reply, chat_degraded) = match call_llm(client, llm, &messages, None).await {
        Ok(r) => (r, false),
        Err(e) => {
            tracing::warn!("LLM unavailable during /chat (rag mode), degrading to retrieval-only: {e}");
            (safe_fallback_chat_reply(&req.message, &memories_used, &e.to_string()), true)
        }
    };

    // 4. Extract memorable facts from the exchange and store them (skip if degraded)
    let memories_added = if chat_degraded {
        vec![]
    } else {
        extract_and_store_facts(
            state, client, llm, ns, &req.message, &reply, &req.history,
        ).await
    };

    Ok(Json(make_chat_response(
        ns,
        reply,
        memories_used,
        memories_added,
        "rag".into(),
        chat_degraded,
    )))
}

async fn chat_tool_mode(
    state: &AppState,
    client: &reqwest::Client,
    ns: &anima_core::namespace::Namespace,
    req: &ChatRequest,
) -> Result<Json<ChatResponse>, AppError> {
    let llm = req.llm.as_ref().expect("llm must be resolved before calling chat_tool_mode");

    let tools = serde_json::json!([
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search the user's personal memory store. Call this PROACTIVELY to recall facts, preferences, relationships, or history about the user. Use short focused queries like 'diet', 'job', 'pets'. Always search before answering personal questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Short search query about the user (e.g., 'allergies', 'family', 'coding preferences')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Store a personal memory about the user. This can be a fact, preference, story, experience, event, decision, or reflection. \
                For facts, keep them atomic (one piece of info). For stories and experiences, capture the full narrative. \
                Do NOT store questions, greetings, or general knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory to store — a fact, story, experience, preference, event, or reflection about the user"
                        }
                    },
                    "required": ["content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "memory_update",
                "description": "Update or correct an existing memory. Use when the user corrects information (e.g., 'actually I have 3 cats, not 2'). Search first to find the memory ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The memory ID to update (from a previous memory_search result)"
                        },
                        "content": {
                            "type": "string",
                            "description": "The corrected memory content"
                        }
                    },
                    "required": ["id", "content"]
                }
            }
        }),
    ]);

    let today = chrono::Utc::now().format("%A, %B %-d, %Y");
    let base_prompt = llm.system_prompt.as_deref().unwrap_or(
        "You are a helpful assistant."
    );
    let base_prompt = format!("{base_prompt}\n\nToday's date: {today}.");
    let system_prompt = format!(
        "{base_prompt}\n\nYou have access to a persistent memory system about the user.\n\n\
        MEMORY TOOLS:\n\
        - memory_search: Search the user's stored memories. Call this PROACTIVELY whenever:\n\
          * The user asks something that might relate to their personal info, preferences, or history\n\
          * The user references something they told you before (\"as I mentioned\", \"remember when\")\n\
          * You need context about the user to give a personalized answer\n\
          * The conversation touches on topics where user preferences matter (food, tech, hobbies)\n\
          Use short, focused queries (e.g., \"allergies\", \"programming languages\", \"family\").\n\n\
        - memory_add: Store a personal memory about the user. Use for facts, stories, experiences, preferences, events, decisions, reflections, goals, relationships, emotions, habits, beliefs, skills, or locations.\n\
          For facts, keep them atomic (ONE piece of info). For stories and experiences, capture the full narrative.\n\
          Include the subject (e.g., \"User's wife Sarah...\").\n\
          Do NOT store: questions, greetings, general knowledge, AI instructions, or ephemeral info.\n\n\
        - memory_update: Correct or update an existing memory. Search first to find the memory ID, then update its content.\n\
          Use when the user corrects information (\"actually I have 3 cats, not 2\").\n\n\
        When in doubt, search first — it's better to check memories than to miss relevant context.\n\
        Be natural and helpful."
    );

    let mut messages: Vec<serde_json::Value> =
        vec![serde_json::json!({"role": "system", "content": system_prompt})];
    for msg in &req.history {
        messages.push(serde_json::json!({"role": msg.role, "content": msg.content}));
    }
    messages.push(build_user_message(&req.message, &req.attachments, llm.vision));

    let mut memories_used = Vec::new();
    let mut memories_added: Vec<AddedMemory> = Vec::new();

    // Tool call loop (max 5 rounds)
    for _ in 0..5 {
        let body = build_llm_body(llm, &messages, Some(&tools));
        let resp = match send_llm_request(client, llm, &body).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM unavailable during /chat (tool mode), degrading: {e}");
                let reply = safe_fallback_chat_reply(&req.message, &memories_used, &e.to_string());
                return Ok(Json(make_chat_response(
                    ns,
                    reply,
                    memories_used,
                    memories_added,
                    "tool".into(),
                    true,
                )));
            }
        };

        let choice = &resp["choices"][0];
        let finish_reason = choice["finish_reason"].as_str().unwrap_or("stop");

        if finish_reason != "tool_calls" {
            // Final response
            let reply = choice["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            return Ok(Json(make_chat_response(
                ns,
                reply,
                memories_used,
                memories_added,
                "tool".into(),
                false,
            )));
        }

        // Process tool calls
        let assistant_msg = choice["message"].clone();
        messages.push(assistant_msg.clone());

        if let Some(tool_calls) = assistant_msg["tool_calls"].as_array() {
            for tc in tool_calls {
                let fn_name = tc["function"]["name"].as_str().unwrap_or("");
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let args: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                let tc_id = tc["id"].as_str().unwrap_or("");
                let raw_conf = procedure_confidence_hint(fn_name, &args);
                let calibrated_conf = state
                    .store
                    .calibrate_confidence(ns, PredictionKind::ProcedureSelection, raw_conf)
                    .await
                    .unwrap_or(raw_conf);

                let result = match fn_name {
                    "memory_search" => {
                        let query = args["query"].as_str().unwrap_or("");
                        match handle_tool_search(state, ns, query).await {
                            Ok((text, contexts)) => {
                                memories_used.extend(contexts);
                                text
                            }
                            Err(e) => format!("Error searching memories: {e}"),
                        }
                    }
                    "memory_add" => {
                        let content = args["content"].as_str().unwrap_or("");
                        match handle_tool_add(state, ns, content, vec![]).await {
                            Ok(id) => {
                                memories_added.push(AddedMemory { id: id.clone(), content: content.to_string() });
                                format!("Memory stored successfully (id: {id})")
                            }
                            Err(e) => format!("Error storing memory: {e}"),
                        }
                    }
                    "memory_update" => {
                        let mem_id = args["id"].as_str().unwrap_or("");
                        let content = args["content"].as_str().unwrap_or("");
                        match handle_tool_update(state, mem_id, content).await {
                            Ok(()) => format!("Memory {mem_id} updated successfully"),
                            Err(e) => format!("Error updating memory: {e}"),
                        }
                    }
                    _ => format!("Unknown tool: {fn_name}"),
                };
                let outcome = if result.starts_with("Error ") || result.starts_with("Unknown tool") {
                    0.0
                } else {
                    1.0
                };
                record_procedure_observation(
                    state,
                    ns,
                    fn_name,
                    raw_conf,
                    outcome,
                    serde_json::json!({
                        "calibrated_confidence": calibrated_conf,
                        "tool_call_id": tc_id,
                        "args_len": args_str.len(),
                    }),
                )
                .await;

                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result
                }));
            }
        }
    }

    // If we exhausted rounds, get a final reply without tools
    let (reply, tool_degraded) = match call_llm(client, llm, &messages, None).await {
        Ok(r) => (r, false),
        Err(e) => {
            tracing::warn!("LLM unavailable during /chat (tool mode final), degrading: {e}");
            (safe_fallback_chat_reply(&req.message, &memories_used, &e.to_string()), true)
        }
    };
    Ok(Json(make_chat_response(
        ns,
        reply,
        memories_used,
        memories_added,
        "tool".into(),
        tool_degraded,
    )))
}

async fn handle_tool_search(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    query: &str,
) -> Result<(String, Vec<MemoryContext>), AppError> {
    let embedding = state
        .embedder
        .embed_query(query)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    let tool_scorer = state.scorer_config.read().await.clone();
    let scored = state
        .store
        .search(
            &embedding,
            query,
            ns,
            &SearchMode::Hybrid,
            5,
            &tool_scorer,
        )
        .await?;
    record_retrieval_observations(state, ns, &scored, 0.15, "tool_memory_search").await;

    let (_, contexts) = expand_with_graph_neighbors(state, &scored, 10).await?;

    let text_parts: Vec<String> = contexts.iter().map(|ctx| {
        let label = ctx.source.as_deref().unwrap_or("search");
        format!("- {} (score: {:.2}, {})", ctx.content, ctx.score, label)
    }).collect();

    let text = if text_parts.is_empty() {
        "No relevant memories found.".to_string()
    } else {
        format!("Found {} memories:\n{}", text_parts.len(), text_parts.join("\n"))
    };

    Ok((text, contexts))
}

async fn handle_tool_add(
    state: &AppState,
    ns: &anima_core::namespace::Namespace,
    content: &str,
    tags: Vec<String>,
) -> Result<String, AppError> {
    let (content, _, _) = redact_content_and_metadata(content, None);
    if content.trim().is_empty() {
        return Err(AppError::BadRequest("content cannot be empty".into()));
    }

    let embedding = state
        .embedder
        .embed(&content)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    // Step 1: Exact dedup by hash
    let hash = content_hash(&content);
    if let Some(existing) = state.store.find_by_hash(ns, &hash).await? {
        return Ok(existing.id);
    }

    // Step 2: Semantic dedup via consolidation
    if let Some(consolidator) = &state.consolidator {
        let similar = state
            .store
            .find_similar(&embedding, ns, 5, consolidator.similarity_threshold())
            .await?;

        if !similar.is_empty() {
            let decision = consolidator
                .decide(&content, &similar)
                .await
                .map_err(|e| AppError::Internal(format!("consolidation error: {e}")))?;

            match decision.action {
                ConsolidationActionType::NoChange => {
                    return Ok(similar[0].0.id.clone());
                }
                ConsolidationActionType::Update => {
                    if let (Some(target_id), Some(merged_content)) =
                        (&decision.target_id, &decision.merged_content)
                    {
                        let new_embedding = state
                            .embedder
                            .embed(merged_content)
                            .map_err(|e| AppError::Embedding(e.to_string()))?;
                        state
                            .store
                            .update_content(target_id, merged_content, &new_embedding)
                            .await?;
                        return Ok(target_id.clone());
                    }
                    // Fallthrough to create if target/content missing
                }
                ConsolidationActionType::Supersede => {
                    let memory = Memory::new(
                            ns.as_str().to_string(),
                            content.clone(),
                            None,
                            tags,
                            None,
                    );
                    let new_id = memory.id.clone();
                    state.store.insert(&memory, &embedding).await?;
                    ingest_identity_hints(state, ns, &memory.content, 0.7, Some(&new_id)).await;
                    if let Some(target_id) = &decision.target_id {
                        state.store.mark_superseded(target_id, &new_id).await?;
                    }
                    return Ok(new_id);
                }
                ConsolidationActionType::Create => {
                    // Predict-calibrate: store only novel claims if the LLM extracted them.
                    // Safety net: if novel_content is < 40% of original length, the LLM
                    // likely over-stripped specific details — fall through to use full content.
                    if let Some(novel) = decision.novel_content.filter(|s| !s.trim().is_empty()).filter(|novel| {
                        let orig_len = content.len();
                        orig_len == 0 || novel.len() * 100 / orig_len >= 40
                    }) {
                        let novel_embedding = state
                            .embedder
                            .embed(&novel)
                            .map_err(|e| AppError::Embedding(e.to_string()))?;
                        let memory = Memory::new(
                            ns.as_str().to_string(),
                            novel,
                            None,
                            tags,
                            None,
                        );
                        let id = memory.id.clone();
                        state.store.insert(&memory, &novel_embedding).await?;
                        ingest_identity_hints(state, ns, &memory.content, 0.68, Some(&id)).await;
                        return Ok(id);
                    }
                    // Fall through to create with original content
                }
            }
        }
    }

    // Step 3: Create new memory
    let memory = Memory::new(ns.as_str().to_string(), content.to_string(), None, tags, None);
    let id = memory.id.clone();
    state.store.insert(&memory, &embedding).await?;
    ingest_identity_hints(state, ns, &memory.content, 0.68, Some(&id)).await;
    Ok(id)
}

async fn handle_tool_update(
    state: &AppState,
    id: &str,
    new_content: &str,
) -> Result<(), AppError> {
    let (new_content, _, _) = redact_content_and_metadata(new_content, None);
    if new_content.trim().is_empty() {
        return Err(AppError::BadRequest("content cannot be empty".into()));
    }

    let embedding = state
        .embedder
        .embed(&new_content)
        .map_err(|e| AppError::Embedding(e.to_string()))?;

    let updated = state.store.update_content(id, &new_content, &embedding).await?;
    if !updated {
        return Err(AppError::NotFound(format!("memory {id} not found")));
    }

    Ok(())
}

// =============================================================================
// Ask (extract-then-answer pipeline)
// =============================================================================

pub async fn reflect(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<ReflectRequest>,
) -> Result<Json<ReflectResponse>, AppError> {
    let start = Instant::now();

    // Resolve memory IDs to reflect on
    let memory_ids = if req.memory_ids.is_empty() {
        // Auto-find unreflected raw memories
        let unreflected = state.store.find_unreflected_raw(&ns, req.limit).await?;
        unreflected.into_iter().map(|m| m.id).collect::<Vec<_>>()
    } else {
        req.memory_ids
    };

    if memory_ids.is_empty() {
        return Ok(Json(ReflectResponse {
            status: "no_work".into(),
            raw_processed: Some(0),
            facts_extracted: Some(0),
            elapsed_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
            reflected: vec![],
        }));
    }

    // Async mode: enqueue to background processor
    if req.r#async {
        if let Some(processor) = &state.processor {
            processor.enqueue(crate::processor::ProcessingJob::Reflect {
                namespace: ns.as_str().to_string(),
                memory_ids: memory_ids.clone(),
            });
            return Ok(Json(ReflectResponse {
                status: "queued".into(),
                raw_processed: Some(memory_ids.len()),
                facts_extracted: None,
                elapsed_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
                reflected: vec![],
            }));
        } else {
            return Err(AppError::Internal("background processor not available".into()));
        }
    }

    // Sync mode: process now
    let llm = state.get_processor_llm()
        .ok_or_else(|| AppError::Internal("no processor LLM configured".into()))?;

    let (facts, reflected_ids) = crate::processor::reflect_sync(
        &state.store,
        &state.embedder,
        &llm,
        ns.as_str(),
        &memory_ids,
    )
    .await
    .map_err(|e| AppError::Internal(format!("reflection error: {e}")))?;

    let reflected: Vec<ReflectedFactDto> = facts
        .iter()
        .map(|f| ReflectedFactDto {
            content: f.content.clone(),
            confidence: f.confidence,
            source_ids: f.source_ids.clone(),
            corrections: f.corrections.clone(),
        })
        .collect();

    // After sync reflection, enqueue reconsolidation + deduction.
    if !reflected_ids.is_empty() {
        if let Some(processor) = &state.processor {
            processor.enqueue(crate::processor::ProcessingJob::Reconsolidate {
                namespace: ns.as_str().to_string(),
                memory_ids: reflected_ids.clone(),
            });
        }
    }
    if reflected_ids.len() >= 2 {
        if let Some(processor) = &state.processor {
            processor.enqueue(crate::processor::ProcessingJob::Deduce {
                namespace: ns.as_str().to_string(),
                reflected_ids,
            });
        }
    }

    Ok(Json(ReflectResponse {
        status: "completed".into(),
        raw_processed: Some(memory_ids.len()),
        facts_extracted: Some(reflected.len()),
        elapsed_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
        reflected,
    }))
}

pub async fn ask(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<AskRequest>,
) -> Result<Json<AskResponse>, AppError> {
    if req.question.trim().is_empty() {
        return Err(AppError::BadRequest("question cannot be empty".into()));
    }
    if is_likely_noisy_or_adversarial_input(&req.question) {
        return Ok(Json(make_ask_response(
            &ns,
            "Input looks noisy or adversarial. Please ask a clear question so I can answer reliably.".to_string(),
            vec![req.question],
            vec![],
            0,
            0.0,
            vec![ConfirmationQuestion {
                question: "Could you restate your question in one clear sentence?".to_string(),
                source_memory_id: "uncertainty".to_string(),
                confidence: 0.0,
            }],
            false,
            vec![],
        )));
    }

    let start = Instant::now();
    let client = reqwest::Client::new();

    // Build LlmConfig from server config or client override
    let base_llm = resolve_llm_config(req.llm, &state, "ask");

    // --- Step 1: Multi-stage retrieval pipeline (no LLM) ---
    let mut ask_scorer_config = state.scorer_config.read().await.clone();
    if let Some(mt) = req.max_tier {
        ask_scorer_config.max_tier = mt.clamp(1, 4);
    }
    ask_scorer_config.date_start = req.date_start;
    ask_scorer_config.date_end = req.date_end;

    let mut all_results = run_ask_retrieval_pipeline(&state, &ns, &req.question, req.max_results, &ask_scorer_config).await?;

    // Apply memory_types filter if specified
    let type_filter = &req.memory_types;
    if !type_filter.is_empty() {
        all_results.retain(|(mem, _)| type_filter.contains(&mem.memory_type));
    }

    // Identity confirmation questions (ask-specific — low-confidence entity disambiguation)
    let entities = extract_candidate_entities(&req.question);
    let mut identity_needs_confirmation: Vec<ConfirmationQuestion> = Vec::new();
    let mut resolved_entity_queries: Vec<String> = Vec::new();
    for entity in &entities {
        match state.store.resolve_identity(&ns, entity, 3).await {
            Ok(resolution) => {
                if resolution.candidates.is_empty() {
                    resolved_entity_queries.push(entity.clone());
                    continue;
                }
                if resolution.best_confidence >= 0.62 {
                    let mut added_bases = HashSet::new();
                    for c in &resolution.candidates {
                        let base = c.canonical_name
                            .split_whitespace().next().unwrap_or(&c.canonical_name)
                            .trim_end_matches("'s").trim_end_matches("\u{2019}s")
                            .to_string();
                        if added_bases.insert(base.to_ascii_lowercase()) {
                            resolved_entity_queries.push(base);
                        }
                    }
                } else {
                    let options: Vec<String> = resolution
                        .candidates
                        .iter()
                        .take(3)
                        .map(|c| c.canonical_name.clone())
                        .collect();
                    if !options.is_empty() {
                        identity_needs_confirmation.push(ConfirmationQuestion {
                            question: format!(
                                "When you say \"{entity}\", did you mean {}?",
                                options.join(" / ")
                            ),
                            source_memory_id: format!(
                                "identity:{}",
                                resolution.candidates[0].entity_id
                            ),
                            confidence: resolution.best_confidence,
                        });
                    }
                    resolved_entity_queries.push(entity.clone());
                }
            }
            Err(_) => {
                resolved_entity_queries.push(entity.clone());
            }
        }
    }
    if resolved_entity_queries.is_empty() {
        resolved_entity_queries = entities.clone();
    }
    let mut dedup = HashSet::new();
    resolved_entity_queries.retain(|q| dedup.insert(q.to_ascii_lowercase()));

    let total_search_results = all_results.len();

    // --- Step 2: Answer ---
    let mut degraded = false;
    let answer = if req.skip_llm {
        // No LLM — just return top memory content (for benchmarking search speed)
        all_results.first().map(|(m, _)| m.content.clone()).unwrap_or_else(|| "No relevant memories found.".into())
    } else {
        let context_block: String = all_results
            .iter()
            .enumerate()
            .map(|(i, (mem, _))| {
                let date = mem.created_at.format("%d %B %Y");
                let (content, _) = redact_sensitive_text(&mem.content);
                format!("{}. [{}] {}", i + 1, date, content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let answer_config = LlmConfig {
            temperature: Some(0.0),
            max_tokens: Some(4096),
            ..base_llm.clone()
        };
        // Extract the latest year mentioned in memory content for temporal context
        let latest_year: Option<u16> = all_results.iter()
            .flat_map(|(m, _)| {
                m.content.as_bytes().windows(4).filter_map(|w| {
                    if w[0] == b'2' && w[1] == b'0' && w[2].is_ascii_digit() && w[3].is_ascii_digit() {
                        let s = std::str::from_utf8(w).ok()?;
                        s.parse::<u16>().ok().filter(|&y| y >= 2020 && y <= 2030)
                    } else {
                        None
                    }
                })
            })
            .max();
        let date_context = match latest_year {
            Some(year) => format!("\nThese memories are from {year}. Answer time-relative questions (\"how long ago\", \"how old\") from {year}'s perspective, NOT from today.\n"),
            None => String::new(),
        };
        // Add date instruction for "when" questions
        let q_lower = req.question.to_ascii_lowercase();
        let date_hint = if q_lower.starts_with("when ") || q_lower.contains("what date") || q_lower.contains("what day") {
            "\nREMINDER: For date questions, answer with ONLY relative phrasing from the memory's date prefix. Example: if a memory from [25 May 2023] says \"last Saturday\", answer \"the Saturday before 25 May 2023\". Do NOT compute or add absolute dates like \"May 20\" or \"July 10\". Give ONLY the relative expression.\n"
        } else {
            ""
        };
        let answer_messages = vec![
            serde_json::json!({"role": "system", "content": crate::prompts::ASK_DIRECT_PROMPT}),
            serde_json::json!({"role": "user", "content": format!(
                "{}Memories:\n{}\n\nQuestion: {}{}", date_context, context_block, req.question, date_hint
            )}),
        ];
        let answer_raw = match call_llm(&client, &answer_config, &answer_messages, None).await {
            Ok(r) => r,
            Err(e) => {
                // Graceful degradation: return retrieval results without LLM synthesis
                tracing::warn!("LLM unavailable during /ask, degrading to retrieval-only: {e}");
                degraded = true;
                build_retrieval_only_answer(&all_results)
            }
        };
        let mut answer_cleaned = strip_think_blocks(&answer_raw);
        if answer_cleaned.is_empty() && !answer_raw.is_empty() {
            tracing::warn!("LLM answer stripped to empty! Raw len={}, first 200 chars: {:?}", answer_raw.len(), &answer_raw[..answer_raw.len().min(200)]);
        }
        // Retry once if LLM says "I don't know" but top memories scored well
        // (skip retry if already degraded — LLM is down)
        if !degraded {
            let idk = answer_cleaned.to_ascii_lowercase();
            let has_idk = idk.contains("i don't know") || idk.contains("i don\u{2019}t know")
                || idk.contains("do not specify") || idk.contains("not specified") || idk.contains("not explicitly")
                || idk.contains("do not mention") || idk.contains("not mentioned")
                || idk.contains("no information") || idk.contains("no relevant")
                || idk.contains("no specific mention") || idk.contains("does not state")
                || idk.contains("do not state") || idk.contains("no mention");
            if has_idk && !all_results.is_empty() && all_results[0].1 > 0.4 {
                tracing::info!("Retrying answer because LLM said IDK with high-scoring memories (top={:.3})", all_results[0].1);
                let retry_system = "You are a memory assistant. The user previously asked a question and I said I don't know, but the answer IS somewhere in the memories. Your job: find ANY detail in the memories that could answer the question, even partially. Look for indirect mentions, related context, and implied information. NEVER say \"I don't know\" — always give your best answer from the available memories.";
                let mut retry_messages = vec![
                    serde_json::json!({"role": "system", "content": retry_system}),
                ];
                for msg in &answer_messages {
                    if msg.get("role").and_then(|v| v.as_str()) == Some("user") {
                        retry_messages.push(msg.clone());
                    }
                }
                let retry_config = LlmConfig {
                    temperature: Some(0.3),
                    ..answer_config.clone()
                };
                if let Ok(retry_raw) = call_llm(&client, &retry_config, &retry_messages, None).await {
                    let retry_cleaned = strip_think_blocks(&retry_raw);
                    let retry_lower = retry_cleaned.to_ascii_lowercase();
                    let still_idk = retry_lower.contains("i don't know")
                        || retry_lower.contains("not specified") || retry_lower.contains("not explicitly")
                        || retry_lower.contains("no information") || retry_lower.contains("do not state")
                        || retry_lower.contains("no mention") || retry_lower.contains("not mentioned");
                    if !retry_cleaned.is_empty() && !still_idk {
                        answer_cleaned = retry_cleaned;
                    }
                }
            }
        }
        answer_cleaned
    };

    // --- Build response ---
    let memories_referenced: Vec<AskMemoryRef> = all_results
        .iter()
        .map(|(mem, score)| AskMemoryRef {
            id: mem.id.clone(),
            content: mem.content.clone(),
            memory_type: mem.memory_type.clone(),
            score: *score,
            created_at: mem.created_at.to_rfc3339(),
        })
        .collect();

    // Scan deduced memories in results for needs_confirmation
    let mut needs_confirmation: Vec<ConfirmationQuestion> = all_results
        .iter()
        .filter_map(|(mem, _)| {
            let meta = mem.metadata.as_ref()?;
            let tier = meta.get("tier")?.as_u64()?;
            if tier != 3 { return None; }
            let question = meta.get("needs_confirmation")?.as_str()?.to_string();
            if question.is_empty() || question == "null" { return None; }
            let confidence = meta.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some(ConfirmationQuestion {
                question,
                source_memory_id: mem.id.clone(),
                confidence,
            })
        })
        .collect();
    needs_confirmation.extend(identity_needs_confirmation);
    let top_score = all_results.first().map(|(_, score)| *score);
    if should_escalate_uncertainty(top_score, total_search_results) {
        needs_confirmation.push(ConfirmationQuestion {
            question: "I may be uncertain about this answer. Can you confirm key details or provide more context?".to_string(),
            source_memory_id: "uncertainty".to_string(),
            confidence: top_score.unwrap_or(0.0),
        });
    }

    // --- Conflict detection: find supersession history for retrieved memories ---
    let retrieved_ids: Vec<String> = all_results.iter().map(|(m, _)| m.id.clone()).collect();
    let conflicts = match state.store.find_contradictions_for_memories(&ns, &retrieved_ids).await {
        Ok(entries) => {
            entries.into_iter().filter_map(|e| {
                let old_content = e.old_content.as_deref().unwrap_or("[deleted]");
                let new_content = e.new_content.as_deref().unwrap_or("[deleted]");
                // Only surface if old and new content are meaningfully different
                if old_content == new_content { return None; }
                // Determine which side is the retrieved memory
                let (memory_id, conflicting_id) = if retrieved_ids.contains(&e.new_memory_id) {
                    (e.new_memory_id.clone(), e.old_memory_id.clone())
                } else {
                    (e.old_memory_id.clone(), e.new_memory_id.clone())
                };
                Some(ConflictNote {
                    memory_id,
                    conflicting_memory_id: conflicting_id,
                    old_content: old_content.to_string(),
                    new_content: new_content.to_string(),
                    resolution: e.resolution,
                    resolved_at: e.created_at,
                })
            }).collect()
        }
        Err(e) => {
            tracing::warn!("Failed to detect conflicts for /ask response: {e}");
            vec![]
        }
    };

    Ok(Json(make_ask_response(
        &ns,
        answer,
        std::iter::once(req.question.clone())
            .chain(resolved_entity_queries.into_iter())
            .collect(),
        memories_referenced,
        total_search_results,
        start.elapsed().as_secs_f64() * 1000.0,
        needs_confirmation,
        degraded,
        conflicts,
    )))
}

/// Build a human-readable answer from retrieval results when the LLM is unavailable.
fn build_retrieval_only_answer(results: &[(anima_core::memory::Memory, f64)]) -> String {
    if results.is_empty() {
        return "No relevant memories found.".to_string();
    }
    let mut answer = String::from("Based on retrieved memories:\n\n");
    for (i, (mem, _score)) in results.iter().take(5).enumerate() {
        let date = mem.created_at.format("%d %B %Y");
        let (content, _) = redact_sensitive_text(&mem.content);
        answer.push_str(&format!("{}. [{}] {}\n", i + 1, date, content));
    }
    if results.len() > 5 {
        answer.push_str(&format!("\n({} more results available)", results.len() - 5));
    }
    answer
}

/// A structured fact extracted by the LLM.
#[derive(Debug, serde::Deserialize)]
struct ExtractedFact {
    fact: String,
    #[allow(dead_code)]
    #[serde(default = "default_fact_type")]
    r#type: String,
    #[serde(default)]
    tags: Vec<String>,
    /// 1-10 importance score (5 = default, 10 = critical like allergies)
    #[serde(default = "default_importance")]
    importance: i32,
    /// If this fact contradicts an existing memory, the content of that memory
    #[serde(default)]
    supersedes: Option<String>,
}

fn default_fact_type() -> String {
    "fact".to_string()
}

fn default_importance() -> i32 {
    5
}

/// Strip `<think>...</think>` blocks from LLM output.
fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</think>") {
            remaining = &remaining[start + end + 8..];
        } else {
            // Unclosed think block — strip everything after <think>
            return result.trim().to_string();
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
}

/// Ask the LLM to extract memorable facts from a user-assistant exchange.
/// Includes conversation history for coreference resolution and contradiction detection.
/// Returns IDs of any memories that were stored.
async fn extract_and_store_facts(
    state: &AppState,
    client: &reqwest::Client,
    llm_config: &LlmConfig,
    ns: &anima_core::namespace::Namespace,
    user_msg: &str,
    assistant_reply: &str,
    history: &[ChatMessage],
) -> Vec<AddedMemory> {
    // Fetch recent memories so the extractor knows what's already stored
    let existing_context = match state.store.list(ns, Some("active"), None, None, 0, 30).await {
        Ok((memories, _)) if !memories.is_empty() => {
            let items: Vec<String> = memories.iter().map(|m| format!("- {}", m.content)).collect();
            format!("\n\nEXISTING MEMORIES (do NOT duplicate these — if a new fact CONTRADICTS one of these, include \"supersedes\" with the exact text of the contradicted memory):\n{}", items.join("\n"))
        }
        _ => String::new(),
    };

    // Build conversation context from recent history for coreference resolution
    let history_context = if history.is_empty() {
        String::new()
    } else {
        // Take last 8 messages for context (4 exchanges)
        let recent: Vec<String> = history
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|m| format!("{}: {}", m.role, truncate_str(&m.content, 200)))
            .collect();
        format!(
            "\n\nCONVERSATION CONTEXT (recent messages for reference resolution — extract facts ONLY from the latest exchange below, not from this context):\n{}",
            recent.join("\n")
        )
    };

    let today = chrono::Utc::now().format("%A, %B %-d, %Y");
    let system_prompt = format!(
        "You extract meaningful personal memories from a conversation with a human user. \
        You receive the latest user message + AI reply, plus recent conversation context for resolving references like \"it\", \"there\", \"she\".\n\n\
        TODAY'S DATE: {today}\n\n\
        TEMPORAL RESOLUTION — CRITICAL:\n\
        ALWAYS resolve relative dates and times to absolute dates in the stored memory.\n\
        - \"yesterday\" → the actual date (e.g., \"on February 27, 2026\")\n\
        - \"last week\" → approximate date (e.g., \"around February 21, 2026\")\n\
        - \"last month\" → \"in January 2026\"\n\
        - \"last year\" → \"in 2025\"\n\
        - \"two days ago\" → the actual date\n\
        - \"this morning\" → \"on the morning of {today}\"\n\
        - \"next Friday\" → the actual date\n\
        Relative time references become meaningless once stored — always anchor them to real dates.\n\n\
        WHAT TO EXTRACT — anything the user shares that is worth remembering:\n\
        - Facts: identity, job, objective information\n\
        - Stories & experiences: anecdotes, things that happened, memorable moments\n\
        - Events: life changes, milestones, plans (moved, started a job, upcoming trip)\n\
        - Opinions & reflections: thoughts they share, lessons learned\n\
        - Context: people they know, pets, projects, tools they use\n\
        - Decisions: choices they've made or are considering\n\
        - Goals & aspirations: what they want to achieve, plans for the future\n\
        - Relationships: who people are to them (family, friends, colleagues)\n\
        - Emotions: how they feel about things, emotional experiences\n\
        - Habits & routines: recurring patterns, daily practices\n\
        - Beliefs & principles: worldviews, values, convictions\n\
        - Skills & expertise: what they know, what they're learning\n\
        - Locations: where they live, work, travel, or care about\n\n\
        MEMORY TYPES:\n\
        - \"fact\" — objective info (\"I work at Google\", \"My cat's name is Luna\")\n\
        - \"preference\" — likes, dislikes (\"I prefer dark mode\", \"I'm vegetarian\")\n\
        - \"event\" — something that happened or will happen (\"I got promoted\", \"We're going to Japan in March\")\n\
        - \"story\" — an anecdote, experience, or narrative (\"Last week I was hiking and saw a bear\")\n\
        - \"decision\" — choices or commitments (\"I decided to switch to Rust\")\n\
        - \"reflection\" — opinions, lessons, values (\"I think remote work is better for creativity\")\n\
        - \"context\" — background info useful for future conversations\n\
        - \"goal\" — aspirations, plans, targets (\"I want to learn piano by summer\", \"I'm saving up for a house\")\n\
        - \"relationship\" — people and how they relate (\"Sarah is my sister\", \"My manager is Tom\")\n\
        - \"emotion\" — feelings about something (\"I was really frustrated with that interview\", \"I love my new job\")\n\
        - \"habit\" — recurring patterns, routines (\"I run every morning at 6am\", \"I journal before bed\")\n\
        - \"belief\" — subjective views, principles (\"I believe in open source\", \"I think AI will change everything\")\n\
        - \"skill\" — abilities, expertise, learning (\"I'm fluent in French\", \"I'm learning Rust\")\n\
        - \"location\" — places that matter (\"I live in Paris\", \"My office is in La Défense\")\n\n\
        STORIES & EXPERIENCES — IMPORTANT:\n\
        When the user tells a story or shares an experience, capture the FULL narrative, not just atomic facts.\n\
        A story should be stored as ONE memory that preserves the narrative flow and emotional context.\n\
        Do NOT split stories into atomic facts — the story IS the memory.\n\n\
        ATOMIC FACTS — for non-story factual information:\n\
        NEVER combine multiple unrelated facts into one. Each factual JSON object MUST contain exactly ONE piece of information.\n\
        If a sentence contains multiple facts (joined by \"and\", commas, or listing multiple items), split them into separate JSON objects.\n\n\
        MANDATORY SPLITS (facts only, NOT stories):\n\
        - \"My wife's name is Sarah and she works as a doctor\" → TWO separate facts:\n\
          {{\"fact\": \"User's wife's name is Sarah\"}} AND {{\"fact\": \"User's wife Sarah works as a doctor\"}}\n\
        - \"We have two cats named Luna and Mochi\" → TWO separate facts:\n\
          {{\"fact\": \"User has a cat named Luna\"}} AND {{\"fact\": \"User has a cat named Mochi\"}}\n\n\
        Include the linking entity (e.g., \"Sarah\", \"User\") in each memory so related facts connect in the knowledge graph.\n\n\
        NEVER EXTRACT — return [] for:\n\
        - Questions the user asks (\"What is X?\", \"How do I Y?\")\n\
        - Typos, corrections, or meta-conversation (\"I meant X\", \"sorry, typo\")\n\
        - Greetings, thanks, or small talk (\"hi\", \"thanks\", \"ok\")\n\
        - Instructions to the AI (\"your name is Bob\", \"be concise\")\n\
        - Facts about the AI assistant itself\n\
        - General knowledge from the AI's reply\n\
        - Trivial or ephemeral things not worth remembering long-term\n\
        - Anything the user didn't explicitly say about themselves or their world\n\
        - Facts that are already in the existing memories list below\n\n\
        CONTRADICTION DETECTION:\n\
        If the user states something that contradicts an existing memory (e.g., \"I work at Meta\" when existing memory says \"User works at Google\"), \
        include a \"supersedes\" field with the EXACT text of the contradicted memory from the list below.\n\n\
        IMPORTANCE SCORING (1-10):\n\
        - 1-3: Minor preferences, casual mentions (\"I like blue\")\n\
        - 4-6: Normal facts, small stories (\"I work as a developer\", \"Had a funny thing happen at work\")\n\
        - 7-8: Important personal facts, meaningful experiences (\"I'm getting married in June\", \"I survived a car accident\")\n\
        - 9-10: Critical safety/health facts, life-defining moments (\"I'm allergic to penicillin\", \"I lost my father last year\")\n\n\
        Be selective but NOT overly restrictive. If the user shares something personal or tells a story, it's worth remembering.\n\n\
        Output ONLY a JSON array:\n\
        [{{\"fact\": \"...\", \"type\": \"fact|preference|event|story|decision|reflection|context|goal|relationship|emotion|habit|belief|skill|location\", \"tags\": [\"...\"], \"importance\": 5, \"supersedes\": null}}]\n\n\
        Examples:\n\
        - \"I'm a vegetarian\" → [{{\"fact\": \"User is vegetarian\", \"type\": \"preference\", \"tags\": [\"food\"], \"importance\": 6}}]\n\
        - \"My wife's name is Sarah and she works as a doctor\" → [\n\
            {{\"fact\": \"User's wife's name is Sarah\", \"type\": \"fact\", \"tags\": [\"family\"], \"importance\": 5}},\n\
            {{\"fact\": \"User's wife Sarah works as a doctor\", \"type\": \"fact\", \"tags\": [\"family\", \"career\"], \"importance\": 5}}\n\
          ]\n\
        - \"Yesterday I saw a bear while hiking\" → [\n\
            {{\"fact\": \"User saw a bear while hiking on February 27, 2026\", \"type\": \"event\", \"tags\": [\"experience\", \"hiking\", \"outdoors\"], \"importance\": 5}}\n\
          ]\n\
        - \"Last week I was hiking near Tahoe and a bear walked right across the trail in front of me. I froze and it just looked at me and kept walking. Scariest moment of my life.\" → [\n\
            {{\"fact\": \"User had a close encounter with a bear while hiking near Lake Tahoe around February 21, 2026. The bear walked across the trail right in front of them — user froze and the bear just looked at them and kept walking. User described it as the scariest moment of their life.\", \"type\": \"story\", \"tags\": [\"experience\", \"hiking\", \"outdoors\"], \"importance\": 6}}\n\
          ]\n\
        - \"I've been thinking a lot about switching careers. I love coding but I feel like I want to do something more creative.\" → [\n\
            {{\"fact\": \"User is considering a career change — loves coding but wants to do something more creative\", \"type\": \"reflection\", \"tags\": [\"career\", \"personal\"], \"importance\": 6}}\n\
          ]\n\
        - \"We just moved to Seattle\" (existing: \"User lives in Portland\") → [{{\"fact\": \"User recently moved to Seattle\", \"type\": \"event\", \"tags\": [\"location\"], \"importance\": 6, \"supersedes\": \"User lives in Portland\"}}]\n\
        - \"I'm severely allergic to shellfish\" → [{{\"fact\": \"User is severely allergic to shellfish\", \"type\": \"fact\", \"tags\": [\"health\", \"allergy\"], \"importance\": 9}}]\n\n\
        Return [] for:\n\
        - \"What is Python?\" → []\n\
        - \"I meant can not cna\" → []\n\
        - \"Thanks that helps\" → []\n\
        - \"Can you explain this code?\" → []\n\
        - \"Your name is Bob\" → []{existing_context}{history_context}"
    );

    let extract_messages = vec![
        serde_json::json!({
            "role": "system",
            "content": system_prompt
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("[HUMAN USER SAID]:\n{user_msg}\n\n[AI ASSISTANT REPLIED]:\n{assistant_reply}")
        }),
    ];

    let raw = match call_llm(client, llm_config, &extract_messages, None).await {
        Ok(r) => r,
        Err(e) => {
            warn!("Failed to extract facts: {e}");
            return vec![];
        }
    };

    // Strip think blocks and parse as structured facts, fall back to plain strings
    let cleaned = strip_think_blocks(&raw);
    let json_str = cleaned.trim();
    let json_str = if let Some(start) = json_str.find('[') {
        if let Some(end) = json_str.rfind(']') {
            &json_str[start..=end]
        } else {
            return vec![];
        }
    } else {
        return vec![];
    };

    // Try structured format first
    let extracted: Vec<ExtractedFact> = match serde_json::from_str(json_str) {
        Ok(facts) => facts,
        Err(_) => {
            // Fall back to plain string array
            let plain: Vec<String> = serde_json::from_str(json_str).unwrap_or_default();
            plain
                .into_iter()
                .map(|f| ExtractedFact {
                    fact: f,
                    r#type: "fact".to_string(),
                    tags: vec![],
                    importance: 5,
                    supersedes: None,
                })
                .collect()
        }
    };

    let mut added = Vec::new();
    for item in extracted {
        let fact = item.fact.trim().to_string();
        if fact.is_empty() {
            continue;
        }

        let importance = item.importance.clamp(1, 10);

        match handle_tool_add(state, ns, &fact, item.tags).await {
            Ok(id) => {
                // Update importance if non-default
                if importance != 5 {
                    if let Err(e) = state.store.set_importance(&id, importance).await {
                        warn!("Failed to set importance for {id}: {e}");
                    }
                }

                // Supersede contradicted memory now that we have the new ID
                if let Some(ref superseded_content) = item.supersedes {
                    let trimmed = superseded_content.trim();
                    if !trimmed.is_empty() {
                        if let Ok(embedding) = state.embedder.embed(trimmed) {
                            if let Ok(similar) =
                                state.store.find_similar(&embedding, ns, 3, 0.7).await
                            {
                                for (mem, sim) in &similar {
                                    if mem.id == id {
                                        continue; // don't supersede self
                                    }
                                    if mem.content.to_lowercase().contains(&trimmed.to_lowercase())
                                        || trimmed
                                            .to_lowercase()
                                            .contains(&mem.content.to_lowercase())
                                        || *sim > 0.85
                                    {
                                        let _ = state
                                            .store
                                            .mark_superseded(&mem.id, &id)
                                            .await;
                                        tracing::info!(
                                            "Superseded '{}' (id={}) → '{}' (id={})",
                                            mem.content, mem.id, fact, id
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                added.push(AddedMemory {
                    id,
                    content: fact,
                });
            }
            Err(e) => warn!("Failed to store fact: {e}"),
        }
    }
    added
}

/// Truncate a string to max chars, adding "..." if truncated.
fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}...")
    }
}

/// Build the user message JSON value.
/// When `vision` is true and images are attached, uses the OpenAI multimodal content array.
/// When `vision` is false, images are noted as text placeholders.
fn build_user_message(text: &str, attachments: &[FileAttachment], vision: bool) -> serde_json::Value {
    let mut text_prefix = String::new();
    let mut images: Vec<&FileAttachment> = Vec::new();

    for att in attachments {
        match att.r#type.as_str() {
            "image" => images.push(att),
            _ => {
                let name = att.name.as_deref().unwrap_or("file");
                text_prefix.push_str(&format!("<file name=\"{name}\">\n{}\n</file>\n\n", att.data));
            }
        }
    }

    let full_text = if text_prefix.is_empty() {
        text.to_string()
    } else {
        format!("{text_prefix}{text}")
    };

    if !vision || images.is_empty() {
        serde_json::json!({"role": "user", "content": full_text})
    } else {
        // Multimodal message (OpenAI vision format)
        let mut content_parts: Vec<serde_json::Value> = Vec::new();

        for img in &images {
            let media_type = img.media_type.as_deref().unwrap_or("image/png");
            let data_url = if img.data.starts_with("data:") {
                img.data.clone()
            } else {
                format!("data:{media_type};base64,{}", img.data)
            };
            content_parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": data_url }
            }));
        }

        if !full_text.is_empty() {
            content_parts.push(serde_json::json!({
                "type": "text",
                "text": full_text
            }));
        }

        serde_json::json!({"role": "user", "content": content_parts})
    }
}

fn build_llm_body(
    config: &LlmConfig,
    messages: &[serde_json::Value],
    tools: Option<&serde_json::Value>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": config.model,
        "messages": messages,
    });
    if let Some(temp) = config.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tok) = config.max_tokens {
        body["max_tokens"] = serde_json::json!(max_tok);
    }
    // Set seed for reproducibility when temperature is 0
    if config.temperature == Some(0.0) {
        body["seed"] = serde_json::json!(42);
    }
    if let Some(t) = tools {
        body["tools"] = t.clone();
    }
    // Disable extended thinking for Qwen3+ models to prevent reasoning tokens
    // from consuming the entire max_tokens budget.
    // chat_template_kwargs is Ollama-specific; cloud APIs (Groq, OpenAI) reject it.
    // Disable extended thinking for Qwen3+ models to prevent reasoning tokens
    // from consuming the entire max_tokens budget.
    if config.model.to_lowercase().contains("qwen3") {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    }
    body
}

async fn send_llm_request(
    client: &reqwest::Client,
    config: &LlmConfig,
    body: &serde_json::Value,
) -> Result<serde_json::Value, AppError> {
    if llm_mock_mode_enabled() {
        if let Some(fixture) = load_mock_fixture_json(body) {
            return Ok(fixture);
        }
        return Ok(mock_chat_completion(body));
    }

    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let mut req = client.post(&url).json(body);
    if let Some(key) = &config.api_key {
        if !key.is_empty() {
            req = req.bearer_auth(key);
        }
    }

    let resp = req
        .send()
        .await
        .map_err(|e| AppError::Internal(format!("LLM request failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        warn!("LLM API error {status}: {text}");
        return Err(AppError::Internal(format!("LLM API returned {status}: {text}")));
    }

    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to parse LLM response: {e}")))
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

fn fixture_key_for_json(body: &serde_json::Value) -> String {
    let serialized = serde_json::to_string(body).unwrap_or_else(|_| body.to_string());
    let mut hasher = Sha256::new();
    hasher.update(serialized.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn load_mock_fixture_json(body: &serde_json::Value) -> Option<serde_json::Value> {
    let dir = llm_fixture_dir()?;
    let key = fixture_key_for_json(body);
    let path = dir.join(format!("{key}.json"));
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn mock_chat_completion(body: &serde_json::Value) -> serde_json::Value {
    let mut system_text = String::new();
    let mut user_text = String::new();

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for m in messages {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let content = m.get("content");

            let as_text = match content {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };

            if role == "system" {
                system_text.push_str(&as_text);
                system_text.push('\n');
            } else if role == "user" {
                user_text = as_text;
            }
        }
    }

    let content = if system_text.contains("Output ONLY a JSON array:") {
        "[]".to_string()
    } else if system_text.contains("Generate a short, descriptive title") {
        "Anima Conversation".to_string()
    } else if system_text.contains("answering questions about someone's memories") {
        if user_text.contains("Memories:") {
            "I don't know.".to_string()
        } else {
            "Mock answer from Anima.".to_string()
        }
    } else {
        "Mock reply from Anima.".to_string()
    };

    serde_json::json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": content
            }
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn loads_body_hash_fixture_in_mock_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let body = serde_json::json!({
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "hello fixture"}]
        });
        let key = fixture_key_for_json(&body);
        let path = dir.path().join(format!("{key}.json"));
        let fixture = serde_json::json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "fixture reply"}
            }]
        });
        std::fs::write(&path, fixture.to_string()).unwrap();

        std::env::set_var("ANIMA_LLM_MODE", "mock");
        std::env::set_var("ANIMA_LLM_FIXTURES_DIR", dir.path());

        let loaded = load_mock_fixture_json(&body).unwrap();
        assert_eq!(loaded["choices"][0]["message"]["content"], "fixture reply");

        std::env::remove_var("ANIMA_LLM_FIXTURES_DIR");
        std::env::remove_var("ANIMA_LLM_MODE");
    }

    #[test]
    fn noisy_input_detection_flags_gibberish() {
        let noisy = "!!!! #### ???? !!!!! #### ???? !!!!! #### ???? !!!!!";
        assert!(is_likely_noisy_or_adversarial_input(noisy));
        assert!(is_likely_noisy_or_adversarial_input("aaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    }

    #[test]
    fn noisy_input_detection_allows_normal_queries() {
        assert!(!is_likely_noisy_or_adversarial_input(
            "Where did I say I wanted to move next year?"
        ));
        assert!(!is_likely_noisy_or_adversarial_input(
            "Please summarize what you remember about my project timeline."
        ));
    }

    #[test]
    fn uncertainty_escalation_thresholds_work() {
        assert!(should_escalate_uncertainty(None, 0));
        assert!(should_escalate_uncertainty(Some(0.2), 3));
        assert!(should_escalate_uncertainty(Some(0.6), 1));
        assert!(!should_escalate_uncertainty(Some(0.7), 4));
    }

    #[test]
    fn redaction_masks_sensitive_text_and_metadata() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("ANIMA_REDACTION_ENABLED", "true");

        let metadata = serde_json::json!({
            "api_key": "sk-secret-value",
            "nested": {"password": "hunter2"},
            "safe": "ok"
        });
        let (content, metadata, changed) = redact_content_and_metadata(
            "contact me at jane@example.com using sk-super-secret-token",
            Some(metadata),
        );
        assert!(changed);
        assert!(!content.contains("jane@example.com"));
        assert!(!content.contains("sk-super-secret-token"));
        let metadata = metadata.unwrap();
        assert_eq!(metadata["api_key"], "[redacted]");
        assert_eq!(metadata["nested"]["password"], "[redacted]");
        assert_eq!(metadata["safe"], "ok");

        std::env::remove_var("ANIMA_REDACTION_ENABLED");
    }

    #[test]
    fn chat_provenance_contains_sources() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("ANIMA_REDACTION_ENABLED", "true");

        let ns = anima_core::namespace::Namespace::parse("default").unwrap();
        let resp = make_chat_response(
            &ns,
            "My email is jane@example.com".to_string(),
            vec![MemoryContext {
                id: "m1".to_string(),
                content: "User email is jane@example.com".to_string(),
                score: 0.91,
                source: Some("search".to_string()),
                memory_type: "fact".to_string(),
                importance: 5,
            }],
            vec![],
            "rag".to_string(),
            false,
        );

        assert_eq!(resp.provenance.namespace, "default");
        assert_eq!(resp.provenance.source_ids, vec!["m1".to_string()]);
        assert!(resp.provenance.redaction_applied);
        assert_eq!(resp.provenance.sources.len(), 1);
        assert!(!resp.reply.contains("jane@example.com"));

        std::env::remove_var("ANIMA_REDACTION_ENABLED");
    }

    #[test]
    fn namespace_contains_subnamespace() {
        let ns = anima_core::namespace::Namespace::parse("team").unwrap();
        assert!(namespace_contains(&ns, "team"));
        assert!(namespace_contains(&ns, "team/alice"));
        assert!(!namespace_contains(&ns, "other/team"));
    }

    #[test]
    fn extract_candidate_entities_handles_multilingual_names() {
        let entities = extract_candidate_entities("I spoke with José Álvarez and Marie Curie today");
        assert!(entities.iter().any(|e| e.contains("José")));
        assert!(entities.iter().any(|e| e.contains("Marie Curie")));
    }

    #[test]
    fn extract_candidate_entities_ignores_question_stop_words() {
        let entities = extract_candidate_entities("What did you say about the project timeline?");
        assert!(entities.is_empty());
    }
}

async fn call_llm(
    client: &reqwest::Client,
    config: &LlmConfig,
    messages: &[serde_json::Value],
    tools: Option<&serde_json::Value>,
) -> Result<String, AppError> {
    let body = build_llm_body(config, messages, tools);
    let resp = send_llm_request(client, config, &body).await?;
    let msg = &resp["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    if content.is_empty() {
        tracing::warn!(
            "LLM returned empty content. finish_reason={}, message keys: {:?}, content_type={:?}",
            resp["choices"][0]["finish_reason"].as_str().unwrap_or("?"),
            msg.as_object().map(|o| o.keys().collect::<Vec<_>>()),
            msg["content"],
        );
    }
    Ok(content.to_string())
}

// =============================================================================
// Streaming Chat (SSE)
// =============================================================================

pub async fn chat_stream(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(mut req): Json<ChatStreamRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    if req.message.trim().is_empty() {
        return Err(AppError::BadRequest("message cannot be empty".into()));
    }
    if is_likely_noisy_or_adversarial_input(&req.message) {
        return Err(AppError::BadRequest(
            "input looks noisy or adversarial; please rewrite clearly".into(),
        ));
    }

    let resolved_llm = resolve_llm_config(req.llm.take(), &state, "chat");
    req.llm = Some(resolved_llm);

    // Search anima for relevant memories
    let embedding = embed_query_cached(&state, &req.message)?;
    let chat_scorer = state.scorer_config.read().await.clone();
    let scored = state
        .store
        .search(&embedding, &req.message, &ns, &SearchMode::Hybrid, 5, &chat_scorer)
        .await?;

    // Gate: only include memories with meaningful relevance scores
    const MIN_RETRIEVAL_SCORE: f64 = 0.15;
    record_retrieval_observations(
        &state,
        &ns,
        &scored,
        MIN_RETRIEVAL_SCORE,
        "chat_stream",
    )
    .await;
    let relevant: Vec<_> = scored
        .into_iter()
        .filter(|sr| sr.score >= MIN_RETRIEVAL_SCORE)
        .collect();

    // Graph-enhanced context expansion
    let (memory_context, memories_used) =
        expand_with_graph_neighbors(&state, &relevant, 12).await?;

    let is_tool_mode = req.mode == "tool";

    // Build messages
    let today = chrono::Utc::now().format("%A, %B %-d, %Y");
    let llm_ref = req.llm.as_ref().expect("llm must be resolved");
    let base_prompt = llm_ref.system_prompt.as_deref().unwrap_or(
        "You are a helpful assistant."
    );
    let base_prompt = format!("{base_prompt}\n\nToday's date: {today}.");
    let system_prompt = if is_tool_mode {
        format!(
            "{base_prompt}\n\nYou have access to a persistent memory system about the user.\n\n\
            MEMORY TOOLS:\n\
            - memory_search: Search the user's stored memories. Call this PROACTIVELY whenever:\n\
              * The user asks something that might relate to their personal info, preferences, or history\n\
              * The user references something they told you before (\"as I mentioned\", \"remember when\")\n\
              * You need context about the user to give a personalized answer\n\
              * The conversation touches on topics where user preferences matter (food, tech, hobbies)\n\
              Use short, focused queries (e.g., \"allergies\", \"programming languages\", \"family\").\n\n\
            - memory_add: Store a personal memory about the user. Use for facts, stories, experiences, preferences, events, decisions, reflections, goals, relationships, emotions, habits, beliefs, skills, or locations.\n\
              For facts, keep them atomic (ONE piece of info). For stories and experiences, capture the full narrative.\n\
              Include the subject (e.g., \"User's wife Sarah...\").\n\
              Do NOT store: questions, greetings, general knowledge, AI instructions, or ephemeral info.\n\n\
            - memory_update: Correct or update an existing memory. Search first to find the memory ID, then update its content.\n\
              Use when the user corrects information (\"actually I have 3 cats, not 2\").\n\n\
            When in doubt, search first — it's better to check memories than to miss relevant context.\n\
            Be natural and helpful."
        )
    } else if memory_context.is_empty() {
        base_prompt.clone()
    } else {
        format!(
            "{base_prompt}\n\nThe following are memories about the user. Only reference them if they are \
             directly relevant to what the user is currently asking about. Do not mention or allude to \
             them otherwise.\n\n{memory_context}"
        )
    };

    let mut messages = vec![serde_json::json!({"role": "system", "content": system_prompt})];
    for msg in &req.history {
        messages.push(serde_json::json!({"role": msg.role, "content": msg.content}));
    }
    messages.push(build_user_message(&req.message, &req.attachments, llm_ref.vision));

    // Build tools for tool mode
    let tools: Option<serde_json::Value> = if is_tool_mode {
        let tools_vec = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search the user's personal memory store. Call this PROACTIVELY to recall facts, preferences, relationships, or history about the user. Use short focused queries like 'diet', 'job', 'pets'. Always search before answering personal questions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Short search query about the user (e.g., 'allergies', 'family', 'coding preferences')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "memory_add",
                    "description": "Store a personal memory about the user. This can be a fact, preference, story, experience, event, decision, or reflection. \
                    For facts, keep them atomic (one piece of info). For stories and experiences, capture the full narrative. \
                    Do NOT store questions, greetings, or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory to store — a fact, story, experience, preference, event, or reflection about the user"
                            }
                        },
                        "required": ["content"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "memory_update",
                    "description": "Update or correct an existing memory. Use when the user corrects information (e.g., 'actually I have 3 cats, not 2'). Search first to find the memory ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The memory ID to update (from a previous memory_search result)"
                            },
                            "content": {
                                "type": "string",
                                "description": "The corrected memory content"
                            }
                        },
                        "required": ["id", "content"]
                    }
                }
            }),
        ];
        Some(serde_json::Value::Array(tools_vec))
    } else {
        None
    };

    let llm_config = req.llm.expect("llm must be resolved");
    let state_clone = state.clone();
    let user_message = req.message.clone();
    let history_clone = req.history.clone();
    let ns_clone = ns.clone();
    let memories_for_fallback = memories_used.clone();

    // Send memories_used as the first SSE event
    let memories_event = serde_json::json!({
        "type": "memories",
        "memories_used": memories_used,
    });

    let stream = async_stream::stream! {
        yield Ok(Event::default().data(memories_event.to_string()));

        if llm_mock_mode_enabled() {
            let full_reply = "Mock reply from Anima.".to_string();
            let token_event = serde_json::json!({
                "type": "token",
                "content": full_reply,
            });
            yield Ok(Event::default().data(token_event.to_string()));

            let extract_client = reqwest::Client::new();
            let _facts = extract_and_store_facts(
                &state_clone, &extract_client, &llm_config, &ns_clone,
                &user_message, &full_reply, &history_clone,
            ).await;

            let done_event = serde_json::json!({
                "type": "done",
                "full_reply": full_reply,
            });
            yield Ok(Event::default().data(done_event.to_string()));
            return;
        }

        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", llm_config.base_url.trim_end_matches('/'));
        let mut full_reply = String::new();

        // Tool-call loop (max 5 rounds)
        let max_rounds = if is_tool_mode { 5 } else { 1 };
        for _round in 0..max_rounds {
            let mut body = serde_json::json!({
                "model": llm_config.model,
                "messages": messages,
                "stream": true,
            });
            if let Some(temp) = llm_config.temperature {
                body["temperature"] = serde_json::json!(temp);
            }
            if let Some(max_tok) = llm_config.max_tokens {
                body["max_tokens"] = serde_json::json!(max_tok);
            }
            if let Some(ref t) = tools {
                body["tools"] = t.clone();
            }

            let mut req_builder = client.post(&url).json(&body);
            if let Some(key) = &llm_config.api_key {
                if !key.is_empty() {
                    req_builder = req_builder.bearer_auth(key);
                }
            }

            let resp = match req_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    let fallback = safe_fallback_chat_reply(
                        &user_message,
                        &memories_for_fallback,
                        &format!("llm_connection_failed: {e}"),
                    );
                    let token_event = serde_json::json!({"type": "token", "content": fallback});
                    yield Ok(Event::default().data(token_event.to_string()));
                    let done_event = serde_json::json!({"type": "done", "full_reply": fallback});
                    yield Ok(Event::default().data(done_event.to_string()));
                    return;
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                let fallback = safe_fallback_chat_reply(
                    &user_message,
                    &memories_for_fallback,
                    &format!("llm_api_error_{status}: {text}"),
                );
                let token_event = serde_json::json!({"type": "token", "content": fallback});
                yield Ok(Event::default().data(token_event.to_string()));
                let done_event = serde_json::json!({"type": "done", "full_reply": fallback});
                yield Ok(Event::default().data(done_event.to_string()));
                return;
            }

            let mut byte_stream = resp.bytes_stream();
            let mut buffer = String::new();
            let mut round_content = String::new();
            let mut finish_reason = String::new();

            // Accumulated tool calls from streaming deltas
            // Key: tool call index -> (id, function_name, arguments_buffer)
            let mut tool_calls_acc: std::collections::BTreeMap<u64, (String, String, String)> =
                std::collections::BTreeMap::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].trim().to_string();
                            buffer = buffer[pos + 1..].to_string();

                            if line.is_empty() || line == "data: [DONE]" {
                                continue;
                            }
                            if let Some(data) = line.strip_prefix("data: ") {
                                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                                    let choice = &parsed["choices"][0];

                                    // Check finish reason
                                    if let Some(fr) = choice["finish_reason"].as_str() {
                                        if !fr.is_empty() && fr != "null" {
                                            finish_reason = fr.to_string();
                                        }
                                    }

                                    // Accumulate text content
                                    if let Some(content) = choice["delta"]["content"].as_str() {
                                        if !content.is_empty() {
                                            round_content.push_str(content);
                                            let token_event = serde_json::json!({"type": "token", "content": content});
                                            yield Ok(Event::default().data(token_event.to_string()));
                                        }
                                    }

                                    // Accumulate tool call deltas
                                    if let Some(tcs) = choice["delta"]["tool_calls"].as_array() {
                                        for tc in tcs {
                                            let idx = tc["index"].as_u64().unwrap_or(0);
                                            let entry = tool_calls_acc.entry(idx).or_insert_with(|| {
                                                (String::new(), String::new(), String::new())
                                            });
                                            if let Some(id) = tc["id"].as_str() {
                                                entry.0 = id.to_string();
                                            }
                                            if let Some(name) = tc["function"]["name"].as_str() {
                                                entry.1 = name.to_string();
                                            }
                                            if let Some(args) = tc["function"]["arguments"].as_str() {
                                                entry.2.push_str(args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let fallback = safe_fallback_chat_reply(
                            &user_message,
                            &memories_for_fallback,
                            &format!("stream_error: {e}"),
                        );
                        let token_event = serde_json::json!({"type": "token", "content": fallback});
                        yield Ok(Event::default().data(token_event.to_string()));
                        let done_event = serde_json::json!({"type": "done", "full_reply": fallback});
                        yield Ok(Event::default().data(done_event.to_string()));
                        return;
                    }
                }
            }

            full_reply.push_str(&round_content);

            // If no tool calls, we're done
            if finish_reason != "tool_calls" || tool_calls_acc.is_empty() {
                break;
            }

            // Process tool calls
            // Build the assistant message with tool_calls for the conversation
            let tc_array: Vec<serde_json::Value> = tool_calls_acc.iter().map(|(_, (id, name, args))| {
                serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                })
            }).collect();

            messages.push(serde_json::json!({
                "role": "assistant",
                "content": if round_content.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(round_content) },
                "tool_calls": tc_array
            }));

            // Execute each tool call and add results
            for (_, (tc_id, fn_name, args_str)) in &tool_calls_acc {
                let args: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                let raw_conf = procedure_confidence_hint(fn_name, &args);
                let calibrated_conf = state_clone
                    .store
                    .calibrate_confidence(&ns_clone, PredictionKind::ProcedureSelection, raw_conf)
                    .await
                    .unwrap_or(raw_conf);

                let result = match fn_name.as_str() {
                    "memory_search" => {
                        let query = args["query"].as_str().unwrap_or("");
                        match handle_tool_search(&state_clone, &ns_clone, query).await {
                            Ok((text, _contexts)) => text,
                            Err(e) => format!("Error searching memories: {e}"),
                        }
                    }
                    "memory_add" => {
                        let content = args["content"].as_str().unwrap_or("");
                        match handle_tool_add(&state_clone, &ns_clone, content, vec![]).await {
                            Ok(id) => format!("Memory stored (id: {id})"),
                            Err(e) => format!("Error storing memory: {e}"),
                        }
                    }
                    "memory_update" => {
                        let mem_id = args["id"].as_str().unwrap_or("");
                        let content = args["content"].as_str().unwrap_or("");
                        match handle_tool_update(&state_clone, mem_id, content).await {
                            Ok(()) => format!("Memory {mem_id} updated successfully"),
                            Err(e) => format!("Error updating memory: {e}"),
                        }
                    }
                    _ => format!("Unknown tool: {fn_name}"),
                };
                let outcome = if result.starts_with("Error ") || result.starts_with("Unknown tool") {
                    0.0
                } else {
                    1.0
                };
                record_procedure_observation(
                    &state_clone,
                    &ns_clone,
                    fn_name,
                    raw_conf,
                    outcome,
                    serde_json::json!({
                        "calibrated_confidence": calibrated_conf,
                        "tool_call_id": tc_id,
                        "args_len": args_str.len(),
                        "streaming": true,
                    }),
                )
                .await;

                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result
                }));
            }
        }

        // Extract and store memorable facts from the exchange
        let extract_client = reqwest::Client::new();
        let _facts = extract_and_store_facts(
            &state_clone, &extract_client, &llm_config, &ns_clone,
            &user_message, &full_reply, &history_clone,
        ).await;

        let done_event = serde_json::json!({
            "type": "done",
            "full_reply": full_reply,
        });
        yield Ok(Event::default().data(done_event.to_string()));
    };

    Ok(Sse::new(stream))
}

// =============================================================================
// Auto-generate conversation title using embeddings + LLM
// =============================================================================

pub async fn generate_title(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(_ns): ExtractNamespace,
    Path(conv_id): Path<String>,
    Json(req): Json<LlmConfig>,
) -> Result<Json<serde_json::Value>, AppError> {
    let conv = state.store.get_conversation(&conv_id).await?
        .ok_or_else(|| AppError::NotFound("conversation not found".into()))?;

    tracing::info!("generate_title: conv_id={conv_id}, messages_bytes={}", conv.messages.len());

    let messages: Vec<serde_json::Value> = serde_json::from_str(&conv.messages).unwrap_or_default();

    tracing::info!("generate_title: parsed {} messages", messages.len());

    // Collect conversation messages for context — use the full conversation
    // but truncate individual messages to keep the prompt manageable
    let conv_snippet: Vec<String> = messages.iter()
        .filter(|m| matches!(m["role"].as_str(), Some("user" | "assistant")))
        .filter_map(|m| {
            let role = m["role"].as_str()?;
            let content = m["content"].as_str()?;
            let truncated: String = content.chars().take(150).collect();
            Some(format!("{}: {}", role, truncated))
        })
        .collect();

    // Cap total context to ~3000 chars to avoid huge prompts
    let mut context = String::new();
    for snippet in &conv_snippet {
        if context.len() + snippet.len() > 3000 { break; }
        if !context.is_empty() { context.push('\n'); }
        context.push_str(snippet);
    }

    let client = reqwest::Client::new();
    let title_messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "Generate a short, descriptive title (3-6 words) for this conversation based on what the user is asking about. Focus on the user's intent and topic. Reply with ONLY the title — no quotes, no punctuation, no explanation."
        }),
        serde_json::json!({
            "role": "user",
            "content": context,
        }),
    ];

    let fallback_title = messages.iter()
        .find(|m| m["role"].as_str() == Some("user"))
        .and_then(|m| m["content"].as_str())
        .unwrap_or("New Chat");

    let raw_title = match call_llm(&client, &req, &title_messages, None).await {
        Ok(t) => {
            tracing::info!("generate_title: LLM returned: {t:?}");
            t
        }
        Err(e) => {
            tracing::warn!("generate_title: LLM call failed: {e}, using fallback");
            fallback_title.chars().take(50).collect::<String>()
        }
    };

    // Strip <think>...</think> blocks that some models produce
    let title = strip_think_blocks(&raw_title).trim().trim_matches('"').to_string();

    state.store.update_conversation(&conv_id, Some(&title), None).await?;

    Ok(Json(serde_json::json!({"title": title})))
}

// =============================================================================
// Conversation CRUD
// =============================================================================

pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<CreateConversationRequest>,
) -> Result<Json<anima_db::store::Conversation>, AppError> {
    let conv = state.store.create_conversation(&ns, &req.title, &req.mode).await?;
    Ok(Json(conv))
}

pub async fn list_conversations(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
) -> Result<Json<Vec<anima_db::store::ConversationSummary>>, AppError> {
    let convs = state.store.list_conversations(&ns).await?;
    Ok(Json(convs))
}

pub async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<anima_db::store::Conversation>, AppError> {
    let conv = state.store.get_conversation(&id).await?
        .ok_or_else(|| AppError::NotFound("conversation not found".into()))?;
    Ok(Json(conv))
}

pub async fn update_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateConversationRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let messages_str = req.messages.map(|m| m.to_string());
    let updated = state.store.update_conversation(
        &id,
        req.title.as_deref(),
        messages_str.as_deref(),
    ).await?;
    if !updated {
        return Err(AppError::NotFound("conversation not found".into()));
    }
    Ok(Json(serde_json::json!({"updated": true})))
}

pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.store.delete_conversation(&id).await?;
    if !deleted {
        return Err(AppError::NotFound("conversation not found".into()));
    }
    Ok(Json(serde_json::json!({"deleted": true})))
}

// ── Telemetry ──────────────────────────────────────────────

pub async fn get_telemetry_config(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "enabled": state.telemetry_enabled()
    }))
}

#[derive(serde::Deserialize)]
pub struct SetTelemetryConfigRequest {
    pub enabled: bool,
    #[serde(default)]
    pub feature_flags: Option<FeatureFlags>,
}

pub async fn set_telemetry_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SetTelemetryConfigRequest>,
) -> Json<serde_json::Value> {
    state
        .telemetry_enabled
        .store(body.enabled, std::sync::atomic::Ordering::Relaxed);

    if let Some(flags) = body.feature_flags {
        let mut current = state.telemetry_feature_flags.write().await;
        *current = flags;
    }

    Json(serde_json::json!({"updated": true}))
}
