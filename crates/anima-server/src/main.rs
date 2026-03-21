mod app;
mod config;
mod dto;
mod handlers;
mod processor;
mod prompts;
mod service;
mod telemetry;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use anima_consolidate::consolidator::Consolidator;
use anima_consolidate::llm_client::{OllamaClient, OpenAiCompatClient};
use anima_core::search::ScorerConfig;
use anima_db::pool::DbPool;
use anima_db::store::MemoryStore;
use anima_embed::Embedder;
use anima_embed::download::ensure_model_files;
use anima_embed::model::{EmbeddingModel, PoolingStrategy};
use anima_embed::openai::OpenAiEmbedder;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::app::{AppState, build_router};
use crate::config::AppConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Handle service management commands before anything else.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--install") {
        return service::run(service::ServiceAction::Install);
    }
    if args.iter().any(|a| a == "--uninstall") {
        return service::run(service::ServiceAction::Uninstall);
    }
    if args.iter().any(|a| a == "--service-status") {
        return service::run(service::ServiceAction::Status);
    }

    // Load .env file (if present)
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "anima_server=info,anima_db=info,anima_embed=info".into()),
        )
        .init();

    // Load config — skip flags when looking for config path
    let config_path = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned();
    let config = AppConfig::load(config_path.as_deref());

    tracing::info!("Starting Anima v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Database: {}", config.database.path);

    // --- Startup self-test: validate dependencies before serving ---

    // 1. Database
    let pool = DbPool::open(&config.database.path, config.embedding.dimension)
        .map_err(|e| {
            tracing::error!(domain = "db", "Startup check failed — database: {e}");
            e
        })?;
    let store = MemoryStore::new(pool);
    store.ping().await.map_err(|e| {
        tracing::error!(domain = "db", "Startup check failed — database ping: {e}");
        anyhow::anyhow!("database unreachable: {e}")
    })?;
    tracing::info!(domain = "db", "Startup check passed — database OK");

    // 2. Embedding model
    let embedder: Arc<dyn Embedder> = match config.embedding.backend.as_str() {
        "openai" => {
            let base_url = config.embedding.api_base_url.clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".into());
            let model = config.embedding.api_model.clone()
                .unwrap_or_else(|| "text-embedding-3-small".into());
            let api_key = config.embedding.api_key.clone()
                .filter(|k| !k.is_empty())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()))
                .ok_or_else(|| {
                    tracing::error!(domain = "embedding", "Startup check failed — OpenAI API key missing");
                    anyhow::anyhow!(
                        "OpenAI embedding backend requires api_key in config or OPENAI_API_KEY env var"
                    )
                })?;
            tracing::info!(
                domain = "embedding",
                "Using OpenAI embedding: model={}, dim={}, base_url={}",
                model, config.embedding.dimension, base_url
            );
            Arc::new(OpenAiEmbedder::new(base_url, model, api_key, config.embedding.dimension))
        }
        _ => {
            // Local ONNX backend (default)
            let model_dir = Path::new(&config.embedding.model_dir);
            let (model_path, tokenizer_path) = ensure_model_files(
                model_dir,
                &config.embedding.model_url,
                &config.embedding.tokenizer_url,
            )
            .await
            .map_err(|e| {
                tracing::error!(domain = "embedding", "Startup check failed — model download: {e}");
                e
            })?;

            tracing::info!("Loading local embedding model from {}", model_path.display());
            let pooling = PoolingStrategy::parse(&config.embedding.pooling)
                .map_err(|e| {
                    tracing::error!(domain = "embedding", "Startup check failed — invalid pooling: {e}");
                    anyhow::anyhow!("invalid embedding config: {e}")
                })?;
            let query_instruction = config.embedding.query_instruction.clone()
                .filter(|s| !s.is_empty());
            let embedder = EmbeddingModel::load(
                &model_path,
                &tokenizer_path,
                config.embedding.dimension,
                pooling,
                query_instruction.clone(),
            )
            .map_err(|e| {
                tracing::error!(domain = "embedding", "Startup check failed — model load: {e}");
                anyhow::anyhow!("failed to load embedding model: {e}")
            })?;

            // Smoke test: embed a short string to verify the model works
            embedder.embed("startup self-test").map_err(|e| {
                tracing::error!(domain = "embedding", "Startup check failed — embedding smoke test: {e}");
                anyhow::anyhow!("embedding model smoke test failed: {e}")
            })?;

            tracing::info!(
                domain = "embedding",
                "Startup check passed — embedding OK (dim={}, pooling={}, query_instruction={})",
                config.embedding.dimension,
                pooling.as_str(),
                if query_instruction.is_some() { "yes" } else { "none" }
            );
            Arc::new(embedder)
        }
    };

    // 3. Consolidation LLM (optional — warn but don't fail)
    let consolidator = if config.consolidation.enabled {
        let llm_client: Arc<dyn anima_consolidate::llm_client::LlmClient> =
            match config.consolidation.backend.as_str() {
                "ollama" => Arc::new(OllamaClient::new(
                    config.consolidation.ollama.base_url.clone(),
                    config.consolidation.ollama.model.clone(),
                ).with_temperature(0.2)),
                "openai_compat" => {
                    let api_key = Some(config.consolidation.openai_compat.api_key.clone())
                        .filter(|k| !k.is_empty())
                        .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()));
                    Arc::new(OpenAiCompatClient::new(
                        config.consolidation.openai_compat.base_url.clone(),
                        api_key,
                        config.consolidation.openai_compat.model.clone(),
                    ).with_temperature(0.2))
                }
                other => {
                    tracing::warn!(domain = "llm", "Unknown consolidation backend '{other}', disabling");
                    return Ok(start_server(store, embedder, None, &config).await?);
                }
            };
        tracing::info!(domain = "llm", "Startup check passed — consolidation LLM configured");
        Some(Arc::new(Consolidator::new(
            llm_client,
            config.search.similarity_threshold,
        )))
    } else {
        tracing::info!("Memory consolidation disabled");
        None
    };

    start_server(store, embedder, consolidator, &config).await
}

async fn start_server(
    store: MemoryStore,
    embedder: Arc<dyn Embedder>,
    consolidator: Option<Arc<Consolidator>>,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let scorer_config = ScorerConfig {
        rrf_k: config.search.rrf_k,
        weight_vector: config.search.weight_vector,
        weight_keyword: config.search.weight_keyword,
        temporal_weight: config.search.temporal_weight,
        lambda: config.search.temporal_lambda,
        access_weight: config.search.access_weight,
        importance_weight: config.search.importance_weight,
        tier_boost: config.search.tier_boost,
        min_vector_similarity: config.search.min_vector_similarity,
        min_score_spread: config.search.min_score_spread,
        max_tier: config.search.max_tier,
        date_start: None,
        date_end: None,
    };

    // Spawn background processor for reflection/deduction
    let bg_processor = if config.processor.enabled {
        let api_key = config.processor.resolve_api_key();
        if let Some(key) = api_key {
            let llm: Arc<dyn anima_consolidate::llm_client::LlmClient> =
                Arc::new(OpenAiCompatClient::new(
                    config.processor.base_url.clone(),
                    Some(key),
                    config.processor.model.clone(),
                ).with_temperature(0.2));
            tracing::info!(
                "Processor LLM: {} ({})",
                config.processor.model,
                config.processor.base_url
            );
            Some(processor::BackgroundProcessor::spawn(
                store.clone(),
                embedder.clone(),
                llm,
            ))
        } else if let Some(ref consolidator) = consolidator {
            tracing::info!("Processor LLM: falling back to consolidation LLM");
            let llm = consolidator.llm_client();
            Some(processor::BackgroundProcessor::spawn(
                store.clone(),
                embedder.clone(),
                llm,
            ))
        } else {
            tracing::info!("Background processor disabled (no API key configured)");
            None
        }
    } else {
        tracing::info!("Background processor disabled by config");
        None
    };

    // At startup, enqueue induction for every namespace that already has
    // reflected or deduced facts but may be missing induced patterns.
    if let Some(ref proc) = bg_processor {
        match store.list_namespaces().await {
            Ok(namespaces) => {
                for ns in namespaces {
                    tracing::info!("Queueing startup induction for namespace: {}", ns.namespace);
                    proc.enqueue(processor::ProcessingJob::Induce {
                        namespace: ns.namespace,
                    });
                }
            }
            Err(e) => tracing::warn!("Could not list namespaces for startup induction: {e}"),
        }
    }

    let state = Arc::new(AppState {
        store,
        embedder,
        consolidator,
        processor: bg_processor,
        scorer_config,
        telemetry_enabled: std::sync::atomic::AtomicBool::new(config.telemetry.enabled),
        telemetry_feature_flags: tokio::sync::RwLock::new(telemetry::FeatureFlags::default()),
        config: config.clone(),
        ingested_count: std::sync::atomic::AtomicU64::new(0),
        ingested_started_at: std::time::Instant::now(),
    });

    // Spawn telemetry loop
    if config.telemetry.enabled {
        telemetry::spawn_telemetry_loop(state.clone());
    } else {
        tracing::info!("Telemetry disabled by config");
    }

    // Periodic calibration model updater
    let calibrate_secs = std::env::var("ANIMA_CALIBRATION_RECOMPUTE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300)
        .max(1);
    {
        let store = state.store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(calibrate_secs));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                if let Err(e) = store.recompute_calibration_models().await {
                    tracing::warn!(domain = "db", "Calibration recompute failed: {e}");
                } else {
                    tracing::debug!("Calibration recompute completed");
                }
            }
        });
    }

    // Periodic retention loop
    let retention_secs = std::env::var("ANIMA_RETENTION_RECOMPUTE_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(900)
        .max(10);
    let retention_limit = std::env::var("ANIMA_RETENTION_LIMIT_PER_NAMESPACE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1000)
        .max(1);
    {
        let store = state.store.clone();
        let processor = state.processor.as_ref().map(|p| p.clone());
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(retention_secs));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                if let Some(proc) = &processor {
                    proc.enqueue(processor::ProcessingJob::Retain {
                        namespace: None,
                        limit_per_namespace: retention_limit,
                    });
                } else if let Err(e) =
                    processor::run_retention_sync(&store, None, retention_limit).await
                {
                    tracing::warn!(domain = "db", "Retention run failed: {e}");
                } else {
                    tracing::debug!("Retention run completed");
                }
            }
        });
    }

    let app = build_router(state.clone())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address: {e}"))?;

    tracing::info!("Listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // --- Graceful shutdown ---
    // Wait for SIGINT (ctrl-c) or SIGTERM, then drain in-flight work.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;

    Ok(())
}

/// Wait for a shutdown signal (ctrl-c on all platforms, SIGTERM on Unix).
/// Once received, drain the background processor with a timeout.
async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received SIGINT, starting graceful shutdown..."),
        _ = terminate => tracing::info!("Received SIGTERM, starting graceful shutdown..."),
    }

    // Drain the background processor: drop the sender so workers see channel closed,
    // then wait for in-flight jobs to finish (with a timeout).
    if let Some(ref proc) = state.processor {
        let in_flight = proc.in_flight();
        let queued = proc.queue_depth();
        if in_flight > 0 || queued > 0 {
            tracing::info!(
                "Waiting for background processor to drain ({in_flight} in-flight, {queued} queued)..."
            );
            let drain_timeout = std::time::Duration::from_secs(30);
            let start = std::time::Instant::now();
            loop {
                if proc.in_flight() == 0 {
                    tracing::info!("Background processor drained successfully");
                    break;
                }
                if start.elapsed() > drain_timeout {
                    tracing::warn!(
                        "Drain timeout (30s) — {} jobs still in-flight, shutting down anyway",
                        proc.in_flight()
                    );
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }

    tracing::info!("Shutdown complete");
}
