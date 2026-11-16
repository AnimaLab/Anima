//! An in-memory server state for handler tests: a SQLite database in memory,
//! an embedder that turns words into a deterministic vector, and an LLM that
//! answers a fixed text and counts its calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anima_consolidate::consolidator::Consolidator;
use anima_consolidate::llm_client::{LlmClient, LlmCompletion, LlmError};
use anima_core::memory::Memory;
use anima_core::search::ScorerConfig;
use anima_db::pool::DbPool;
use anima_db::store::MemoryStore;
use anima_embed::Embedder;
use anima_embed::model::EmbedError;

use crate::app::AppState;
use crate::config::AppConfig;

/// Bag of words hashed into `DbPool::DEFAULT_DIMENSION` buckets, normalised:
/// texts sharing words are close, unrelated texts are far.
pub struct WordEmbedder;

impl Embedder for WordEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let dim = DbPool::DEFAULT_DIMENSION;
        let mut v = vec![0f32; dim];
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 1) {
            let h = word.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
            v[(h % dim as u64) as usize] += 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        Ok(v.into_iter().map(|x| x / norm).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed(text)
    }

    fn dimension(&self) -> usize {
        DbPool::DEFAULT_DIMENSION
    }
}

/// Answers `reply` to every prompt and counts the calls.
pub struct FixedLlm {
    pub reply: String,
    pub calls: AtomicUsize,
}

#[async_trait::async_trait]
impl LlmClient for FixedLlm {
    async fn complete_with_usage(&self, _prompt: &str) -> Result<LlmCompletion, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(LlmCompletion { content: self.reply.clone(), usage: None })
    }
}

/// A server state on an empty in-memory database, with a consolidator that
/// uses `llm` when one is given.
pub fn state(llm: Option<Arc<FixedLlm>>) -> Arc<AppState> {
    let pool = DbPool::open_in_memory().expect("in-memory database");
    let config = AppConfig::default();
    Arc::new(AppState {
        store: MemoryStore::new(pool),
        embedder: Arc::new(WordEmbedder),
        consolidator: llm.map(|l| Arc::new(Consolidator::new(l as Arc<dyn LlmClient>, 0.5))),
        processor: None,
        scorer_config: tokio::sync::RwLock::new(ScorerConfig { min_score_spread: 0.0, ..Default::default() }),
        reranker: None,
        resolved_profiles: config.resolve_profiles(),
        config,
        ingested_count: AtomicU64::new(0),
        ingested_started_at: std::time::Instant::now(),
        telemetry_enabled: AtomicBool::new(false),
        telemetry_feature_flags: tokio::sync::RwLock::new(Default::default()),
        vector_configs: vec![("content".to_string(), 1.0)],
        vec_status: tokio::sync::RwLock::new(anima_db::vector::VecTableStatus::Ready),
        latest_release: tokio::sync::RwLock::new(None),
        config_source: crate::setup::ConfigSource { loaded_from: None },
        restart_requested: AtomicBool::new(false),
        restart: tokio::sync::Notify::new(),
    })
}

/// Store a memory directly, with its category.
pub async fn put(state: &AppState, namespace: &str, content: &str, category: &str) -> String {
    let mut m = Memory::new(namespace.to_string(), content.to_string(), None, vec![], Some("raw".to_string()));
    m.category = category.to_string();
    let embedding = state.embedder.embed(content).unwrap();
    state.store.insert(&m, &embedding).await.unwrap();
    m.id
}
