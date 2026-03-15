//! Background processor for memory reflection and deduction.
//!
//! Runs as a tokio task with bounded queue + dead-letter fallback so that
//! pressure is visible and dropped jobs can be inspected/replayed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use chrono::Utc;
use serde::Deserialize;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;

use anima_consolidate::llm_client::{LlmClient, LlmUsage};
use anima_core::memory::Memory;
use anima_core::namespace::Namespace;
use anima_db::store::{MemoryStore, PredictionKind};
use anima_embed::Embedder;

/// A job for the background processor.
#[derive(Debug, Clone)]
pub enum ProcessingJob {
    /// Extract structured facts from raw memories.
    Reflect {
        namespace: String,
        memory_ids: Vec<String>,
    },
    /// Cross-memory inference from reflected facts.
    Deduce {
        namespace: String,
        reflected_ids: Vec<String>,
    },
    /// Pattern generalisation across all deduced facts in a namespace.
    Induce {
        namespace: String,
    },
    /// Commit high-value items from working memory into long-term memory.
    CommitWorkingMemory {
        namespace: String,
        conversation_id: Option<String>,
        limit: usize,
        min_score: f64,
    },
    /// Supersede duplicate/obsolete claims when new related evidence appears.
    Reconsolidate {
        namespace: String,
        memory_ids: Vec<String>,
    },
    /// Periodic retention job to soften stale low-value memories.
    Retain {
        namespace: Option<String>,
        limit_per_namespace: usize,
    },
}

#[derive(Debug, Clone)]
pub struct WorkingMemoryCommitResult {
    pub evaluated: usize,
    pub committed_memory_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReconsolidationResult {
    pub processed: usize,
    pub superseded: usize,
}

#[derive(Debug, Clone)]
pub struct RetentionResult {
    pub processed: usize,
    pub softened: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct LlmStageUsage {
    llm_calls: usize,
    usage_missing: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl LlmStageUsage {
    fn record_call(&mut self, usage: Option<LlmUsage>) {
        self.llm_calls += 1;
        if let Some(u) = usage {
            self.prompt_tokens += u.prompt_tokens;
            self.completion_tokens += u.completion_tokens;
            self.total_tokens += u.total_tokens;
        } else {
            self.usage_missing += 1;
        }
    }
}

#[derive(Debug, Clone)]
struct ReflectionRunResult {
    reflected_ids: Vec<String>,
    usage: LlmStageUsage,
}

#[derive(Debug, Clone)]
struct DeductionRunResult {
    deduced_ids: Vec<String>,
    usage: LlmStageUsage,
}

#[derive(Debug, Clone)]
struct InductionRunResult {
    usage: LlmStageUsage,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DeadLetterJob {
    job: ProcessingJob,
    reason: String,
    at: String,
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessorMetricsSnapshot {
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub reflected_created: usize,
    pub deduced_created: usize,
    pub committed_evaluated: usize,
    pub committed_written: usize,
    pub recon_processed: usize,
    pub recon_superseded: usize,
    pub retention_processed: usize,
    pub retention_softened: usize,
    pub llm_calls: usize,
    pub llm_usage_missing: usize,
    pub llm_prompt_tokens: usize,
    pub llm_completion_tokens: usize,
    pub llm_total_tokens: usize,
    pub reflection_llm_calls: usize,
    pub reflection_usage_missing: usize,
    pub reflection_prompt_tokens: usize,
    pub reflection_completion_tokens: usize,
    pub reflection_total_tokens: usize,
    pub deduction_llm_calls: usize,
    pub deduction_usage_missing: usize,
    pub deduction_prompt_tokens: usize,
    pub deduction_completion_tokens: usize,
    pub deduction_total_tokens: usize,
    pub induction_llm_calls: usize,
    pub induction_usage_missing: usize,
    pub induction_prompt_tokens: usize,
    pub induction_completion_tokens: usize,
    pub induction_total_tokens: usize,
}

/// Background processor that handles reflection/deduction asynchronously.
#[derive(Clone)]
pub struct BackgroundProcessor {
    tx: mpsc::Sender<ProcessingJob>,
    /// Jobs enqueued but not yet picked up by the worker.
    queued: Arc<AtomicUsize>,
    /// Jobs currently being executed by the worker.
    in_flight: Arc<AtomicUsize>,
    /// Count of jobs routed to dead-letter queue.
    dead_letter_count: Arc<AtomicUsize>,
    completed_jobs: Arc<AtomicUsize>,
    failed_jobs: Arc<AtomicUsize>,
    reflected_created: Arc<AtomicUsize>,
    deduced_created: Arc<AtomicUsize>,
    committed_evaluated: Arc<AtomicUsize>,
    committed_written: Arc<AtomicUsize>,
    recon_processed: Arc<AtomicUsize>,
    recon_superseded: Arc<AtomicUsize>,
    retention_processed: Arc<AtomicUsize>,
    retention_softened: Arc<AtomicUsize>,
    llm_calls: Arc<AtomicUsize>,
    llm_usage_missing: Arc<AtomicUsize>,
    llm_prompt_tokens: Arc<AtomicUsize>,
    llm_completion_tokens: Arc<AtomicUsize>,
    llm_total_tokens: Arc<AtomicUsize>,
    reflection_llm_calls: Arc<AtomicUsize>,
    reflection_usage_missing: Arc<AtomicUsize>,
    reflection_prompt_tokens: Arc<AtomicUsize>,
    reflection_completion_tokens: Arc<AtomicUsize>,
    reflection_total_tokens: Arc<AtomicUsize>,
    deduction_llm_calls: Arc<AtomicUsize>,
    deduction_usage_missing: Arc<AtomicUsize>,
    deduction_prompt_tokens: Arc<AtomicUsize>,
    deduction_completion_tokens: Arc<AtomicUsize>,
    deduction_total_tokens: Arc<AtomicUsize>,
    induction_llm_calls: Arc<AtomicUsize>,
    induction_usage_missing: Arc<AtomicUsize>,
    induction_prompt_tokens: Arc<AtomicUsize>,
    induction_completion_tokens: Arc<AtomicUsize>,
    induction_total_tokens: Arc<AtomicUsize>,
    dead_letters: Arc<std::sync::Mutex<Vec<DeadLetterJob>>>,
}

impl BackgroundProcessor {
    /// Spawn the background worker and return a handle for enqueuing jobs.
    pub fn spawn(
        store: MemoryStore,
        embedder: Arc<dyn Embedder>,
        llm: Arc<dyn LlmClient>,
    ) -> Self {
        let capacity = std::env::var("ANIMA_PROCESSOR_QUEUE_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            .max(8);
        let (tx, rx) = mpsc::channel::<ProcessingJob>(capacity);
        let queued = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let dead_letter_count = Arc::new(AtomicUsize::new(0));
        let completed_jobs = Arc::new(AtomicUsize::new(0));
        let failed_jobs = Arc::new(AtomicUsize::new(0));
        let reflected_created = Arc::new(AtomicUsize::new(0));
        let deduced_created = Arc::new(AtomicUsize::new(0));
        let committed_evaluated = Arc::new(AtomicUsize::new(0));
        let committed_written = Arc::new(AtomicUsize::new(0));
        let recon_processed = Arc::new(AtomicUsize::new(0));
        let recon_superseded = Arc::new(AtomicUsize::new(0));
        let retention_processed = Arc::new(AtomicUsize::new(0));
        let retention_softened = Arc::new(AtomicUsize::new(0));
        let llm_calls = Arc::new(AtomicUsize::new(0));
        let llm_usage_missing = Arc::new(AtomicUsize::new(0));
        let llm_prompt_tokens = Arc::new(AtomicUsize::new(0));
        let llm_completion_tokens = Arc::new(AtomicUsize::new(0));
        let llm_total_tokens = Arc::new(AtomicUsize::new(0));
        let reflection_llm_calls = Arc::new(AtomicUsize::new(0));
        let reflection_usage_missing = Arc::new(AtomicUsize::new(0));
        let reflection_prompt_tokens = Arc::new(AtomicUsize::new(0));
        let reflection_completion_tokens = Arc::new(AtomicUsize::new(0));
        let reflection_total_tokens = Arc::new(AtomicUsize::new(0));
        let deduction_llm_calls = Arc::new(AtomicUsize::new(0));
        let deduction_usage_missing = Arc::new(AtomicUsize::new(0));
        let deduction_prompt_tokens = Arc::new(AtomicUsize::new(0));
        let deduction_completion_tokens = Arc::new(AtomicUsize::new(0));
        let deduction_total_tokens = Arc::new(AtomicUsize::new(0));
        let induction_llm_calls = Arc::new(AtomicUsize::new(0));
        let induction_usage_missing = Arc::new(AtomicUsize::new(0));
        let induction_prompt_tokens = Arc::new(AtomicUsize::new(0));
        let induction_completion_tokens = Arc::new(AtomicUsize::new(0));
        let induction_total_tokens = Arc::new(AtomicUsize::new(0));
        let dead_letters = Arc::new(std::sync::Mutex::new(Vec::new()));

        let num_workers = processor_llm_concurrency();
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        for worker_id in 0..num_workers {
            let rx = rx.clone();
            let store = store.clone();
            let embedder = embedder.clone();
            let llm = llm.clone();
            let queued_worker = queued.clone();
            let in_flight_worker = in_flight.clone();
            let completed_jobs_worker = completed_jobs.clone();
            let failed_jobs_worker = failed_jobs.clone();
            let reflected_created_worker = reflected_created.clone();
            let deduced_created_worker = deduced_created.clone();
            let committed_evaluated_worker = committed_evaluated.clone();
            let committed_written_worker = committed_written.clone();
            let recon_processed_worker = recon_processed.clone();
            let recon_superseded_worker = recon_superseded.clone();
            let retention_processed_worker = retention_processed.clone();
            let retention_softened_worker = retention_softened.clone();
            let llm_calls_worker = llm_calls.clone();
            let llm_usage_missing_worker = llm_usage_missing.clone();
            let llm_prompt_tokens_worker = llm_prompt_tokens.clone();
            let llm_completion_tokens_worker = llm_completion_tokens.clone();
            let llm_total_tokens_worker = llm_total_tokens.clone();
            let reflection_llm_calls_worker = reflection_llm_calls.clone();
            let reflection_usage_missing_worker = reflection_usage_missing.clone();
            let reflection_prompt_tokens_worker = reflection_prompt_tokens.clone();
            let reflection_completion_tokens_worker = reflection_completion_tokens.clone();
            let reflection_total_tokens_worker = reflection_total_tokens.clone();
            let deduction_llm_calls_worker = deduction_llm_calls.clone();
            let deduction_usage_missing_worker = deduction_usage_missing.clone();
            let deduction_prompt_tokens_worker = deduction_prompt_tokens.clone();
            let deduction_completion_tokens_worker = deduction_completion_tokens.clone();
            let deduction_total_tokens_worker = deduction_total_tokens.clone();
            let induction_llm_calls_worker = induction_llm_calls.clone();
            let induction_usage_missing_worker = induction_usage_missing.clone();
            let induction_prompt_tokens_worker = induction_prompt_tokens.clone();
            let induction_completion_tokens_worker = induction_completion_tokens.clone();
            let induction_total_tokens_worker = induction_total_tokens.clone();
            tokio::spawn(async move {
                tracing::info!("Background processor worker {worker_id} started");
                run_worker(
                    rx,
                    store,
                    embedder,
                    llm,
                    queued_worker,
                    in_flight_worker,
                    completed_jobs_worker,
                    failed_jobs_worker,
                    reflected_created_worker,
                    deduced_created_worker,
                    committed_evaluated_worker,
                    committed_written_worker,
                    recon_processed_worker,
                    recon_superseded_worker,
                    retention_processed_worker,
                    retention_softened_worker,
                    llm_calls_worker,
                    llm_usage_missing_worker,
                    llm_prompt_tokens_worker,
                    llm_completion_tokens_worker,
                    llm_total_tokens_worker,
                    reflection_llm_calls_worker,
                    reflection_usage_missing_worker,
                    reflection_prompt_tokens_worker,
                    reflection_completion_tokens_worker,
                    reflection_total_tokens_worker,
                    deduction_llm_calls_worker,
                    deduction_usage_missing_worker,
                    deduction_prompt_tokens_worker,
                    deduction_completion_tokens_worker,
                    deduction_total_tokens_worker,
                    induction_llm_calls_worker,
                    induction_usage_missing_worker,
                    induction_prompt_tokens_worker,
                    induction_completion_tokens_worker,
                    induction_total_tokens_worker,
                )
                .await;
            });
        }

        tracing::info!("Background processor started (bounded queue, capacity={capacity}, workers={num_workers})");
        Self {
            tx,
            queued,
            in_flight,
            dead_letter_count,
            completed_jobs,
            failed_jobs,
            reflected_created,
            deduced_created,
            committed_evaluated,
            committed_written,
            recon_processed,
            recon_superseded,
            retention_processed,
            retention_softened,
            llm_calls,
            llm_usage_missing,
            llm_prompt_tokens,
            llm_completion_tokens,
            llm_total_tokens,
            reflection_llm_calls,
            reflection_usage_missing,
            reflection_prompt_tokens,
            reflection_completion_tokens,
            reflection_total_tokens,
            deduction_llm_calls,
            deduction_usage_missing,
            deduction_prompt_tokens,
            deduction_completion_tokens,
            deduction_total_tokens,
            induction_llm_calls,
            induction_usage_missing,
            induction_prompt_tokens,
            induction_completion_tokens,
            induction_total_tokens,
            dead_letters,
        }
    }

    /// Enqueue a job. If queue is full/closed, route to dead-letter storage.
    pub fn enqueue(&self, job: ProcessingJob) {
        match self.tx.try_send(job.clone()) {
            Ok(()) => {
                self.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(j)) => {
                self.record_dead_letter(j, "queue_full");
                tracing::warn!("Background processor queue full — routed to dead-letter");
            }
            Err(mpsc::error::TrySendError::Closed(j)) => {
                self.record_dead_letter(j, "queue_closed");
                tracing::warn!("Background processor channel closed — routed to dead-letter");
            }
        }
    }

    /// Number of jobs waiting in the channel (not yet picked up by the worker).
    pub fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    /// Number of jobs currently being executed by the worker.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn dead_letter_count(&self) -> usize {
        self.dead_letter_count.load(Ordering::Relaxed)
    }

    pub fn metrics_snapshot(&self) -> ProcessorMetricsSnapshot {
        ProcessorMetricsSnapshot {
            completed_jobs: self.completed_jobs.load(Ordering::Relaxed),
            failed_jobs: self.failed_jobs.load(Ordering::Relaxed),
            reflected_created: self.reflected_created.load(Ordering::Relaxed),
            deduced_created: self.deduced_created.load(Ordering::Relaxed),
            committed_evaluated: self.committed_evaluated.load(Ordering::Relaxed),
            committed_written: self.committed_written.load(Ordering::Relaxed),
            recon_processed: self.recon_processed.load(Ordering::Relaxed),
            recon_superseded: self.recon_superseded.load(Ordering::Relaxed),
            retention_processed: self.retention_processed.load(Ordering::Relaxed),
            retention_softened: self.retention_softened.load(Ordering::Relaxed),
            llm_calls: self.llm_calls.load(Ordering::Relaxed),
            llm_usage_missing: self.llm_usage_missing.load(Ordering::Relaxed),
            llm_prompt_tokens: self.llm_prompt_tokens.load(Ordering::Relaxed),
            llm_completion_tokens: self.llm_completion_tokens.load(Ordering::Relaxed),
            llm_total_tokens: self.llm_total_tokens.load(Ordering::Relaxed),
            reflection_llm_calls: self.reflection_llm_calls.load(Ordering::Relaxed),
            reflection_usage_missing: self.reflection_usage_missing.load(Ordering::Relaxed),
            reflection_prompt_tokens: self.reflection_prompt_tokens.load(Ordering::Relaxed),
            reflection_completion_tokens: self.reflection_completion_tokens.load(Ordering::Relaxed),
            reflection_total_tokens: self.reflection_total_tokens.load(Ordering::Relaxed),
            deduction_llm_calls: self.deduction_llm_calls.load(Ordering::Relaxed),
            deduction_usage_missing: self.deduction_usage_missing.load(Ordering::Relaxed),
            deduction_prompt_tokens: self.deduction_prompt_tokens.load(Ordering::Relaxed),
            deduction_completion_tokens: self.deduction_completion_tokens.load(Ordering::Relaxed),
            deduction_total_tokens: self.deduction_total_tokens.load(Ordering::Relaxed),
            induction_llm_calls: self.induction_llm_calls.load(Ordering::Relaxed),
            induction_usage_missing: self.induction_usage_missing.load(Ordering::Relaxed),
            induction_prompt_tokens: self.induction_prompt_tokens.load(Ordering::Relaxed),
            induction_completion_tokens: self.induction_completion_tokens.load(Ordering::Relaxed),
            induction_total_tokens: self.induction_total_tokens.load(Ordering::Relaxed),
        }
    }

    fn record_dead_letter(&self, job: ProcessingJob, reason: &str) {
        self.dead_letter_count.fetch_add(1, Ordering::Relaxed);
        let mut dlq = self.dead_letters.lock().unwrap();
        dlq.push(DeadLetterJob {
            job,
            reason: reason.to_string(),
            at: Utc::now().to_rfc3339(),
        });
        // Keep memory bounded.
        const MAX_DLQ: usize = 2000;
        if dlq.len() > MAX_DLQ {
            let overflow = dlq.len() - MAX_DLQ;
            dlq.drain(0..overflow);
        }
    }
}

fn apply_usage_metrics(
    usage: LlmStageUsage,
    llm_calls: &Arc<AtomicUsize>,
    llm_usage_missing: &Arc<AtomicUsize>,
    llm_prompt_tokens: &Arc<AtomicUsize>,
    llm_completion_tokens: &Arc<AtomicUsize>,
    llm_total_tokens: &Arc<AtomicUsize>,
    stage_calls: &Arc<AtomicUsize>,
    stage_usage_missing: &Arc<AtomicUsize>,
    stage_prompt_tokens: &Arc<AtomicUsize>,
    stage_completion_tokens: &Arc<AtomicUsize>,
    stage_total_tokens: &Arc<AtomicUsize>,
) {
    if usage.llm_calls == 0 {
        return;
    }

    llm_calls.fetch_add(usage.llm_calls, Ordering::Relaxed);
    llm_usage_missing.fetch_add(usage.usage_missing, Ordering::Relaxed);
    llm_prompt_tokens.fetch_add(usage.prompt_tokens, Ordering::Relaxed);
    llm_completion_tokens.fetch_add(usage.completion_tokens, Ordering::Relaxed);
    llm_total_tokens.fetch_add(usage.total_tokens, Ordering::Relaxed);

    stage_calls.fetch_add(usage.llm_calls, Ordering::Relaxed);
    stage_usage_missing.fetch_add(usage.usage_missing, Ordering::Relaxed);
    stage_prompt_tokens.fetch_add(usage.prompt_tokens, Ordering::Relaxed);
    stage_completion_tokens.fetch_add(usage.completion_tokens, Ordering::Relaxed);
    stage_total_tokens.fetch_add(usage.total_tokens, Ordering::Relaxed);
}

fn induction_trigger_new_deduced() -> usize {
    std::env::var("ANIMA_INDUCTION_TRIGGER_NEW_DEDUCED")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(12)
        .clamp(1, 10_000)
}

fn processor_llm_concurrency() -> usize {
    std::env::var("ANIMA_PROCESSOR_LLM_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 16)
}

async fn run_worker(
    rx: Arc<tokio::sync::Mutex<mpsc::Receiver<ProcessingJob>>>,
    store: MemoryStore,
    embedder: Arc<dyn Embedder>,
    llm: Arc<dyn LlmClient>,
    queued: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
    completed_jobs: Arc<AtomicUsize>,
    failed_jobs: Arc<AtomicUsize>,
    reflected_created: Arc<AtomicUsize>,
    deduced_created: Arc<AtomicUsize>,
    committed_evaluated: Arc<AtomicUsize>,
    committed_written: Arc<AtomicUsize>,
    recon_processed: Arc<AtomicUsize>,
    recon_superseded: Arc<AtomicUsize>,
    retention_processed: Arc<AtomicUsize>,
    retention_softened: Arc<AtomicUsize>,
    llm_calls: Arc<AtomicUsize>,
    llm_usage_missing: Arc<AtomicUsize>,
    llm_prompt_tokens: Arc<AtomicUsize>,
    llm_completion_tokens: Arc<AtomicUsize>,
    llm_total_tokens: Arc<AtomicUsize>,
    reflection_llm_calls: Arc<AtomicUsize>,
    reflection_usage_missing: Arc<AtomicUsize>,
    reflection_prompt_tokens: Arc<AtomicUsize>,
    reflection_completion_tokens: Arc<AtomicUsize>,
    reflection_total_tokens: Arc<AtomicUsize>,
    deduction_llm_calls: Arc<AtomicUsize>,
    deduction_usage_missing: Arc<AtomicUsize>,
    deduction_prompt_tokens: Arc<AtomicUsize>,
    deduction_completion_tokens: Arc<AtomicUsize>,
    deduction_total_tokens: Arc<AtomicUsize>,
    induction_llm_calls: Arc<AtomicUsize>,
    induction_usage_missing: Arc<AtomicUsize>,
    induction_prompt_tokens: Arc<AtomicUsize>,
    induction_completion_tokens: Arc<AtomicUsize>,
    induction_total_tokens: Arc<AtomicUsize>,
) {
    // Track newly deduced facts per namespace and trigger induction only once
    // enough net-new deductions have accumulated.
    let mut pending_induction_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let induction_threshold = induction_trigger_new_deduced();

    loop {
        let job = {
            let mut guard = rx.lock().await;
            guard.recv().await
        };
        let Some(job) = job else { break };
        queued.fetch_sub(1, Ordering::Relaxed);
        in_flight.fetch_add(1, Ordering::Relaxed);
        let mut job_failed = false;

        match job {
            ProcessingJob::Reflect {
                namespace,
                memory_ids,
            } => {
                match process_reflection(&store, &embedder, &llm, &namespace, &memory_ids).await {
                    Ok(reflection) => {
                        apply_usage_metrics(
                            reflection.usage,
                            &llm_calls,
                            &llm_usage_missing,
                            &llm_prompt_tokens,
                            &llm_completion_tokens,
                            &llm_total_tokens,
                            &reflection_llm_calls,
                            &reflection_usage_missing,
                            &reflection_prompt_tokens,
                            &reflection_completion_tokens,
                            &reflection_total_tokens,
                        );
                        let reflected_ids = reflection.reflected_ids;
                        reflected_created.fetch_add(reflected_ids.len(), Ordering::Relaxed);
                        if !reflected_ids.is_empty() {
                            match process_reconsolidation(&store, &namespace, &reflected_ids).await {
                                Ok(recon) => {
                                    recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                                    recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    job_failed = true;
                                    tracing::error!("Reconsolidation failed for {namespace}: {e}");
                                }
                            }
                        }
                        if reflected_ids.len() >= 2 {
                            tracing::info!("Auto-triggering deduction on {} reflected facts", reflected_ids.len());
                            match process_deduction(&store, &embedder, &llm, &namespace, &reflected_ids).await {
                                Ok(deduction) => {
                                    apply_usage_metrics(
                                        deduction.usage,
                                        &llm_calls,
                                        &llm_usage_missing,
                                        &llm_prompt_tokens,
                                        &llm_completion_tokens,
                                        &llm_total_tokens,
                                        &deduction_llm_calls,
                                        &deduction_usage_missing,
                                        &deduction_prompt_tokens,
                                        &deduction_completion_tokens,
                                        &deduction_total_tokens,
                                    );
                                    let deduced_ids = deduction.deduced_ids;
                                    if !deduced_ids.is_empty() {
                                        deduced_created.fetch_add(deduced_ids.len(), Ordering::Relaxed);
                                        match process_reconsolidation(&store, &namespace, &deduced_ids).await {
                                            Ok(recon) => {
                                                recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                                                recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                                            }
                                            Err(e) => {
                                                job_failed = true;
                                                tracing::error!("Reconsolidation failed for {namespace}: {e}");
                                            }
                                        }
                                        let counter = pending_induction_counts
                                            .entry(namespace.clone())
                                            .or_insert(0);
                                        *counter += deduced_ids.len();
                                    }
                                }
                                Err(e) => {
                                    job_failed = true;
                                    tracing::error!("Deduction failed for {namespace}: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Reflection failed for {namespace}: {e}");
                    }
                }
            }
            ProcessingJob::Deduce {
                namespace,
                reflected_ids,
            } => {
                match process_deduction(&store, &embedder, &llm, &namespace, &reflected_ids).await {
                    Ok(deduction) => {
                        apply_usage_metrics(
                            deduction.usage,
                            &llm_calls,
                            &llm_usage_missing,
                            &llm_prompt_tokens,
                            &llm_completion_tokens,
                            &llm_total_tokens,
                            &deduction_llm_calls,
                            &deduction_usage_missing,
                            &deduction_prompt_tokens,
                            &deduction_completion_tokens,
                            &deduction_total_tokens,
                        );
                        let deduced_ids = deduction.deduced_ids;
                        if !deduced_ids.is_empty() {
                            deduced_created.fetch_add(deduced_ids.len(), Ordering::Relaxed);
                            match process_reconsolidation(&store, &namespace, &deduced_ids).await {
                                Ok(recon) => {
                                    recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                                    recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    job_failed = true;
                                    tracing::error!("Reconsolidation failed for {namespace}: {e}");
                                }
                            }
                            let counter = pending_induction_counts
                                .entry(namespace.clone())
                                .or_insert(0);
                            *counter += deduced_ids.len();
                        }
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Deduction failed for {namespace}: {e}");
                    }
                }
            }
            ProcessingJob::Induce { namespace } => {
                // Explicit induction request (e.g. from the API).
                match process_induction(&store, &embedder, &llm, &namespace).await {
                    Ok(induction) => {
                        apply_usage_metrics(
                            induction.usage,
                            &llm_calls,
                            &llm_usage_missing,
                            &llm_prompt_tokens,
                            &llm_completion_tokens,
                            &llm_total_tokens,
                            &induction_llm_calls,
                            &induction_usage_missing,
                            &induction_prompt_tokens,
                            &induction_completion_tokens,
                            &induction_total_tokens,
                        );
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Induction failed for {namespace}: {e}");
                    }
                }
                pending_induction_counts.remove(&namespace);
            }
            ProcessingJob::CommitWorkingMemory {
                namespace,
                conversation_id,
                limit,
                min_score,
            } => {
                match commit_working_memory_sync(
                    &store,
                    &embedder,
                    &namespace,
                    conversation_id.as_deref(),
                    limit,
                    min_score,
                )
                .await
                {
                    Ok(result) => {
                        committed_evaluated.fetch_add(result.evaluated, Ordering::Relaxed);
                        committed_written.fetch_add(result.committed_memory_ids.len(), Ordering::Relaxed);
                        if !result.committed_memory_ids.is_empty() {
                            match process_reflection(
                                &store,
                                &embedder,
                                &llm,
                                &namespace,
                                &result.committed_memory_ids,
                            )
                            .await
                            {
                                Ok(reflection) => {
                                    apply_usage_metrics(
                                        reflection.usage,
                                        &llm_calls,
                                        &llm_usage_missing,
                                        &llm_prompt_tokens,
                                        &llm_completion_tokens,
                                        &llm_total_tokens,
                                        &reflection_llm_calls,
                                        &reflection_usage_missing,
                                        &reflection_prompt_tokens,
                                        &reflection_completion_tokens,
                                        &reflection_total_tokens,
                                    );
                                    let reflected_ids = reflection.reflected_ids;
                                    reflected_created.fetch_add(reflected_ids.len(), Ordering::Relaxed);
                                    if !reflected_ids.is_empty() {
                                        match process_reconsolidation(&store, &namespace, &reflected_ids).await {
                                            Ok(recon) => {
                                                recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                                                recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                                            }
                                            Err(e) => {
                                                job_failed = true;
                                                tracing::error!("Reconsolidation failed for {namespace}: {e}");
                                            }
                                        }
                                    }
                                    if reflected_ids.len() >= 2 {
                                        match process_deduction(
                                            &store,
                                            &embedder,
                                            &llm,
                                            &namespace,
                                            &reflected_ids,
                                        )
                                        .await
                                        {
                                            Ok(deduction) => {
                                                apply_usage_metrics(
                                                    deduction.usage,
                                                    &llm_calls,
                                                    &llm_usage_missing,
                                                    &llm_prompt_tokens,
                                                    &llm_completion_tokens,
                                                    &llm_total_tokens,
                                                    &deduction_llm_calls,
                                                    &deduction_usage_missing,
                                                    &deduction_prompt_tokens,
                                                    &deduction_completion_tokens,
                                                    &deduction_total_tokens,
                                                );
                                                let deduced_ids = deduction.deduced_ids;
                                                if !deduced_ids.is_empty() {
                                                    deduced_created.fetch_add(deduced_ids.len(), Ordering::Relaxed);
                                                    match process_reconsolidation(&store, &namespace, &deduced_ids).await {
                                                        Ok(recon) => {
                                                            recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                                                            recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                                                        }
                                                        Err(e) => {
                                                            job_failed = true;
                                                            tracing::error!("Reconsolidation failed for {namespace}: {e}");
                                                        }
                                                    }
                                                    let counter = pending_induction_counts
                                                        .entry(namespace.clone())
                                                        .or_insert(0);
                                                    *counter += deduced_ids.len();
                                                }
                                            }
                                            Err(e) => {
                                                job_failed = true;
                                                tracing::error!("Deduction failed for {namespace}: {e}");
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    job_failed = true;
                                    tracing::error!("Reflection failed for {namespace}: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Working-memory commit failed for {namespace}: {e}");
                    }
                }
            }
            ProcessingJob::Reconsolidate {
                namespace,
                memory_ids,
            } => {
                match process_reconsolidation(&store, &namespace, &memory_ids).await {
                    Ok(recon) => {
                        recon_processed.fetch_add(recon.processed, Ordering::Relaxed);
                        recon_superseded.fetch_add(recon.superseded, Ordering::Relaxed);
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Reconsolidation failed for {namespace}: {e}");
                    }
                }
            }
            ProcessingJob::Retain {
                namespace,
                limit_per_namespace,
            } => {
                match run_retention_sync(&store, namespace.as_deref(), limit_per_namespace).await {
                    Ok(ret) => {
                        retention_processed.fetch_add(ret.processed, Ordering::Relaxed);
                        retention_softened.fetch_add(ret.softened, Ordering::Relaxed);
                    }
                    Err(e) => {
                        job_failed = true;
                        tracing::error!("Retention run failed: {e}");
                    }
                }
            }
        }

        if job_failed {
            failed_jobs.fetch_add(1, Ordering::Relaxed);
        } else {
            completed_jobs.fetch_add(1, Ordering::Relaxed);
        }
        in_flight.fetch_sub(1, Ordering::Relaxed);

        // When the queue is fully drained, run induction only for namespaces
        // that reached the configured new-deduced threshold.
        if queued.load(Ordering::Relaxed) == 0 && !pending_induction_counts.is_empty() {
            let ready: Vec<(String, usize)> = pending_induction_counts
                .iter()
                .filter_map(|(ns, count)| {
                    if *count >= induction_threshold {
                        Some((ns.clone(), *count))
                    } else {
                        None
                    }
                })
                .collect();
            for (ns, count) in ready {
                pending_induction_counts.remove(&ns);
                in_flight.fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    "Queue drained — running induction for {ns} (new_deduced={count}, threshold={induction_threshold})"
                );
                match process_induction(&store, &embedder, &llm, &ns).await {
                    Ok(induction) => {
                        apply_usage_metrics(
                            induction.usage,
                            &llm_calls,
                            &llm_usage_missing,
                            &llm_prompt_tokens,
                            &llm_completion_tokens,
                            &llm_total_tokens,
                            &induction_llm_calls,
                            &induction_usage_missing,
                            &induction_prompt_tokens,
                            &induction_completion_tokens,
                            &induction_total_tokens,
                        );
                        completed_jobs.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        failed_jobs.fetch_add(1, Ordering::Relaxed);
                        tracing::error!("Induction failed for {}: {e}", ns);
                    }
                }
                in_flight.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
    tracing::info!("Background processor shutting down");
}

/// Similarity threshold for grouping memories into the same reflection cluster.
const CLUSTER_THRESHOLD: f64 = 0.60;
/// Maximum memories per reflection cluster (keeps LLM context manageable).
const MAX_CLUSTER_SIZE: usize = 10;

/// Group memories into topic-coherent clusters using greedy centroid matching.
///
/// Memories with similar embeddings are reflected together so the LLM has
/// coherent context when extracting facts. Falls back to individual clusters
/// for any memory whose embedding fails.
fn semantic_cluster(memories: Vec<Memory>, embedder: &dyn Embedder) -> Vec<Vec<Memory>> {
    // Re-embed each memory (local ONNX, fast). None means embed failed.
    let embeddings: Vec<Option<Vec<f32>>> = memories
        .iter()
        .map(|m| embedder.embed(&m.content).ok())
        .collect();

    let mut assigned = vec![false; memories.len()];
    let mut clusters: Vec<Vec<Memory>> = Vec::new();

    for i in 0..memories.len() {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;

        let seed_emb = match &embeddings[i] {
            Some(e) => e,
            None => {
                clusters.push(vec![memories[i].clone()]);
                continue;
            }
        };

        let mut cluster_indices = vec![i];

        for j in (i + 1)..memories.len() {
            if assigned[j] || cluster_indices.len() >= MAX_CLUSTER_SIZE {
                continue;
            }
            if let Some(emb_j) = &embeddings[j] {
                if cosine_sim(seed_emb, emb_j) >= CLUSTER_THRESHOLD {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        clusters.push(
            cluster_indices
                .iter()
                .map(|&idx| memories[idx].clone())
                .collect(),
        );
    }

    clusters
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)) as f64
    }
}

fn working_memory_commit_score(base_score: f64, content: &str) -> f64 {
    let mut score = base_score.clamp(0.0, 1.0);
    let lower = content.to_ascii_lowercase();
    let stable_markers = [
        "always",
        "prefer",
        "allergic",
        "i live",
        "my name is",
        "my dog's name",
        "i use",
        "i check",
    ];
    let uncertain_markers = ["maybe", "not sure", "i think", "guess", "probably"];

    for marker in stable_markers {
        if lower.contains(marker) {
            score += 0.08;
        }
    }
    for marker in uncertain_markers {
        if lower.contains(marker) {
            score -= 0.12;
        }
    }
    if content.trim().len() > 64 {
        score += 0.03;
    }
    score.clamp(0.0, 1.0)
}

fn memory_tier(memory: &Memory) -> i32 {
    memory
        .metadata
        .as_ref()
        .and_then(|m| m.get("tier"))
        .and_then(|v| v.as_i64())
        .unwrap_or(1) as i32
}

fn normalized_claim_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_jaccard(a: &str, b: &str) -> f64 {
    let ta: std::collections::HashSet<String> =
        a.split_whitespace().map(|s| s.to_string()).collect();
    let tb: std::collections::HashSet<String> =
        b.split_whitespace().map(|s| s.to_string()).collect();
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn should_supersede(existing: &Memory, newer: &Memory) -> bool {
    if existing.id == newer.id {
        return false;
    }
    let existing_norm = normalized_claim_text(&existing.content);
    let newer_norm = normalized_claim_text(&newer.content);
    if existing_norm.is_empty() || newer_norm.is_empty() {
        return false;
    }
    if existing_norm == newer_norm {
        return true;
    }
    let jac = token_jaccard(&existing_norm, &newer_norm);
    jac >= 0.92
        && (existing_norm.len() as i64 - newer_norm.len() as i64).abs() <= 16
        && existing_norm.len() >= 24
        && newer_norm.len() >= 24
}

pub async fn commit_working_memory_sync(
    store: &MemoryStore,
    embedder: &Arc<dyn Embedder>,
    namespace: &str,
    conversation_id: Option<&str>,
    limit: usize,
    min_score: f64,
) -> anyhow::Result<WorkingMemoryCommitResult> {
    let ns = Namespace::parse(namespace)?;
    let now = Utc::now().to_rfc3339();
    let _ = store.expire_working_memories(&now).await;
    let entries = store
        .list_working_memories(&ns, Some("pending"), conversation_id, limit.max(1))
        .await?;
    let mut committed_ids = Vec::new();

    // Filter and build memories, discard low-score entries
    struct PendingCommit {
        memory: Memory,
        score: f64,
        wm_entry_id: String,
    }
    let mut pending: Vec<PendingCommit> = Vec::new();
    for entry in &entries {
        let score = working_memory_commit_score(entry.provisional_score, &entry.content);
        if score < min_score {
            let _ = store
                .update_working_memory_state(&entry.id, Some(score), Some("discarded"), None)
                .await;
            continue;
        }

        let mut metadata = entry.metadata.clone().unwrap_or_else(|| serde_json::json!({}));
        if !metadata.is_object() {
            metadata = serde_json::json!({
                "raw_metadata": metadata,
            });
        }
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert("tier".to_string(), serde_json::json!(1));
            obj.insert("source".to_string(), serde_json::json!("working_memory_commit"));
            obj.insert("working_memory_id".to_string(), serde_json::json!(entry.id));
            obj.insert("commit_score".to_string(), serde_json::json!(score));
        }

        let mut memory = Memory::new(
            namespace.to_string(),
            entry.content.clone(),
            Some(metadata),
            vec!["working_memory".to_string(), "raw".to_string()],
            Some("raw".to_string()),
        );
        // Inherit episode_id from working memory's conversation_id
        memory.episode_id = entry.conversation_id.clone();
        pending.push(PendingCommit { memory, score, wm_entry_id: entry.id.clone() });
    }

    // Batch-embed all accepted working memories at once
    if !pending.is_empty() {
        let texts: Vec<&str> = pending.iter().map(|p| p.memory.content.as_str()).collect();
        let embeddings = match embedder.embed_batch(&texts) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Batch embedding failed for {} working memories: {e}", texts.len());
                Vec::new()
            }
        };

        for (i, commit) in pending.into_iter().enumerate() {
            let embedding = match embeddings.get(i) {
                Some(e) => e.clone(),
                None => {
                    tracing::warn!("Missing embedding for working memory {}", commit.memory.id);
                    continue;
                }
            };
            let mem_id = commit.memory.id.clone();
            if store.insert(&commit.memory, &embedding).await.is_ok() {
                let importance = (3.0 + (commit.score * 6.0)).round() as i32;
                let _ = store.set_importance(&mem_id, importance.clamp(1, 10)).await;
                let _ = store
                    .update_working_memory_state(&commit.wm_entry_id, Some(commit.score), Some("committed"), Some(&mem_id))
                    .await;
                committed_ids.push(mem_id);
            }
        }
    }

    Ok(WorkingMemoryCommitResult {
        evaluated: entries.len(),
        committed_memory_ids: committed_ids,
    })
}

pub async fn reconsolidate_sync(
    store: &MemoryStore,
    namespace: &str,
    memory_ids: &[String],
) -> anyhow::Result<ReconsolidationResult> {
    process_reconsolidation(store, namespace, memory_ids).await
}

async fn process_reconsolidation(
    store: &MemoryStore,
    namespace: &str,
    memory_ids: &[String],
) -> anyhow::Result<ReconsolidationResult> {
    let ns = Namespace::parse(namespace)?;
    let mut processed = 0usize;
    let mut superseded = 0usize;

    for id in memory_ids {
        let Some(newer) = store.get(id).await? else {
            continue;
        };
        if newer.status != anima_core::memory::MemoryStatus::Active {
            continue;
        }
        let tier = memory_tier(&newer);
        if tier < 2 {
            continue;
        }
        processed += 1;

        let candidates = store.find_by_tier(&ns, tier, 400).await?;
        for existing in candidates {
            if existing.id == newer.id
                || existing.status != anima_core::memory::MemoryStatus::Active
                || existing.created_at >= newer.created_at
            {
                continue;
            }
            if should_supersede(&existing, &newer) {
                let marked = store.mark_superseded(&existing.id, &newer.id).await.unwrap_or(false);
                if marked {
                    superseded += 1;
                    let _ = store
                        .upsert_state_transition(
                            &ns,
                            &existing.id,
                            &newer.id,
                            "supersession",
                            0.93,
                            Some("reconsolidation duplicate or near-duplicate"),
                        )
                        .await;
                }
            }
        }
    }

    Ok(ReconsolidationResult {
        processed,
        superseded,
    })
}

pub async fn run_retention_sync(
    store: &MemoryStore,
    namespace: Option<&str>,
    limit_per_namespace: usize,
) -> anyhow::Result<RetentionResult> {
    let now = Utc::now();
    let cutoff_days = std::env::var("ANIMA_RETENTION_CUTOFF_DAYS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(45)
        .max(1);
    let strong_cold_days = std::env::var("ANIMA_RETENTION_COLD_DAYS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(180)
        .max(cutoff_days + 1);

    let mut namespaces: Vec<Namespace> = Vec::new();
    if let Some(ns_str) = namespace {
        namespaces.push(Namespace::parse(ns_str)?);
    } else {
        for info in store.list_namespaces().await? {
            if let Ok(ns) = Namespace::parse(&info.namespace) {
                namespaces.push(ns);
            }
        }
    }

    let mut processed = 0usize;
    let mut softened = 0usize;

    for ns in namespaces {
        let (memories, _) = store
            .list(&ns, Some("active"), None, 0, limit_per_namespace.max(1))
            .await?;
        for memory in memories {
            let age_days = (now - memory.created_at).num_days();
            if age_days < cutoff_days {
                continue;
            }
            processed += 1;

            if memory.access_count >= 3 && memory.importance >= 4 {
                continue;
            }
            let tier = memory_tier(&memory);
            let floor = match tier {
                4..=10 => 3,
                3 => 2,
                _ => 1,
            };
            let drop = if age_days >= strong_cold_days { 2 } else { 1 };
            let next_importance = (memory.importance - drop).max(floor);
            if next_importance < memory.importance {
                store.set_importance(&memory.id, next_importance).await?;
                softened += 1;
                let mut meta = memory.metadata.unwrap_or_else(|| serde_json::json!({}));
                if !meta.is_object() {
                    meta = serde_json::json!({ "raw_metadata": meta });
                }
                if let Some(obj) = meta.as_object_mut() {
                    obj.insert("retention_last_run".into(), serde_json::json!(now.to_rfc3339()));
                    obj.insert("retention_age_days".into(), serde_json::json!(age_days));
                    obj.insert("retention_importance_before".into(), serde_json::json!(memory.importance));
                    obj.insert("retention_importance_after".into(), serde_json::json!(next_importance));
                }
                let _ = store.update_json_metadata(&memory.id, meta).await;
            }
        }
    }

    Ok(RetentionResult { processed, softened })
}

/// Process reflection for a batch of raw memory IDs.
/// Returns the IDs of newly created reflected memories.
async fn process_reflection(
    store: &MemoryStore,
    embedder: &Arc<dyn Embedder>,
    llm: &Arc<dyn LlmClient>,
    namespace: &str,
    memory_ids: &[String],
) -> anyhow::Result<ReflectionRunResult> {
    // Fetch the raw memories
    let mut raw_memories: Vec<Memory> = Vec::new();
    for id in memory_ids {
        if let Ok(Some(mem)) = store.get(id).await {
            if mem.status == anima_core::memory::MemoryStatus::Active {
                raw_memories.push(mem);
            }
        }
    }

    if raw_memories.is_empty() {
        return Ok(ReflectionRunResult {
            reflected_ids: vec![],
            usage: LlmStageUsage::default(),
        });
    }

    tracing::info!(
        "Reflecting on {} raw memories in {namespace}",
        raw_memories.len()
    );

    let mut reflected_ids = Vec::new();
    let mut usage = LlmStageUsage::default();
    let ns = anima_core::namespace::Namespace::parse(namespace)?;

    // Cluster into semantically coherent groups before reflecting.
    // LLM calls are bounded-concurrent for throughput, then DB writes run in
    // deterministic input order to keep behavior stable.
    let llm_concurrency = processor_llm_concurrency();
    let semaphore = Arc::new(Semaphore::new(llm_concurrency));
    let mut join_set = JoinSet::new();
    for (idx, chunk) in semantic_cluster(raw_memories, embedder).into_iter().enumerate() {
        let llm = llm.clone();
        let prompt = build_reflection_prompt(&chunk);
        let permit = semaphore.clone().acquire_owned().await?;
        join_set.spawn(async move {
            let _permit = permit;
            (idx, chunk, llm.complete_with_usage(&prompt).await)
        });
    }

    let mut llm_batches = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(item) => llm_batches.push(item),
            Err(e) => tracing::warn!("Reflection worker task failed: {e}"),
        }
    }
    llm_batches.sort_by_key(|(idx, _, _)| *idx);

    for (_idx, chunk, completion) in llm_batches {
        let completion = match completion {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM reflection call failed: {e}");
                continue;
            }
        };
        usage.record_call(completion.usage);
        let response = completion.content;

        let facts = match parse_reflection_response(&response) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("Failed to parse reflection response: {e}");
                continue;
            }
        };

        // Filter facts by confidence threshold, build memories
        let mut pending_memories: Vec<(Memory, f64, usize)> = Vec::new(); // (memory, raw_conf, source_count)
        for fact in &facts {
            let raw_conf = fact.confidence.clamp(0.0, 1.0);
            let calibrated_conf = store
                .calibrate_confidence(&ns, PredictionKind::Extraction, raw_conf)
                .await
                .unwrap_or(raw_conf);
            if calibrated_conf < 0.5 {
                let _ = store
                    .record_calibration_observation(
                        &ns,
                        PredictionKind::Extraction,
                        None,
                        raw_conf,
                        Some(0.0),
                        Some(serde_json::json!({
                            "stage": "reflection",
                            "reason": "below_threshold",
                            "source_count": fact.source_ids.len(),
                        })),
                    )
                    .await;
                continue;
            }

            let metadata = serde_json::json!({
                "tier": 2,
                "confidence": calibrated_conf,
                "raw_confidence": raw_conf,
                "source_ids": fact.source_ids,
                "corrections": fact.corrections,
                "event_date": fact.event_date,
                "location": fact.location,
            });

            let mut memory = Memory::new(
                namespace.to_string(),
                fact.content.clone(),
                Some(metadata),
                vec!["reflected".to_string()],
                Some("reflected".to_string()),
            );
            // Inherit episode_id from source chunk (most common)
            {
                let mut ep_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                for m in chunk.iter() {
                    if let Some(ep) = m.episode_id.as_deref() {
                        *ep_counts.entry(ep).or_insert(0) += 1;
                    }
                }
                if let Some((&best_ep, _)) = ep_counts.iter().max_by_key(|(_, c)| *c) {
                    memory.episode_id = Some(best_ep.to_string());
                }
            }
            pending_memories.push((memory, raw_conf, fact.source_ids.len()));
        }

        // Batch-embed all accepted facts at once
        if !pending_memories.is_empty() {
            let texts: Vec<&str> = pending_memories.iter().map(|(m, _, _)| m.content.as_str()).collect();
            let embeddings = match embedder.embed_batch(&texts) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Batch embedding failed for {} reflected facts: {e}", texts.len());
                    Vec::new()
                }
            };

            // Batch insert with pre-computed embeddings
            let mut batch_entries: Vec<(Memory, Vec<f32>)> = Vec::new();
            for (i, (memory, raw_conf, source_count)) in pending_memories.into_iter().enumerate() {
                let embedding = match embeddings.get(i) {
                    Some(e) => e.clone(),
                    None => {
                        tracing::warn!("Missing embedding for reflected fact {}", memory.id);
                        continue;
                    }
                };
                batch_entries.push((memory, embedding));
                // Record calibration for accepted facts (raw_conf, source_count stored for later)
                let _ = raw_conf;
                let _ = source_count;
            }

            // Use batch insert for efficiency
            let batch_ids: Vec<String> = batch_entries.iter().map(|(m, _)| m.id.clone()).collect();
            if let Err(e) = store.insert_many(&batch_entries).await {
                tracing::warn!("Batch insert of {} reflected memories failed: {e}", batch_entries.len());
            } else {
                reflected_ids.extend(batch_ids);
            }
        }

        // Mark source raw memories as reflected
        for source in &chunk {
            let meta = serde_json::json!({
                "tier": 1,
                "confidence": 1.0,
                "reflected": true,
            });
            let _ = store.update_json_metadata(&source.id, meta).await;
        }

        tracing::info!(
            "Extracted {} facts from {} raw memories in {namespace}",
            facts.len(),
            chunk.len()
        );
    }

    Ok(ReflectionRunResult {
        reflected_ids,
        usage,
    })
}

// --- Reflection prompt and parsing ---

fn reflection_prompt_template() -> &'static str {
    static TEMPLATE: OnceLock<String> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        r#"Task: extract atomic factual claims from RAW_MEMORIES.

Output JSON only (no markdown, no prose):
{"facts":[{"content":"","confidence":0.0,"source_ids":[""],"corrections":null,"event_date":null,"location":null}]}

Rules:
- Keep only factual claims with confidence >= 0.5.
- One sentence per fact, deduplicated.
- source_ids must reference input IDs.
- event_date: YYYY-MM-DD only for specific dated events, else null.
- Convert relative time references to absolute dates using the memory timestamp.
- location: most specific place string available, else null.
- Max 12 facts.
- If no valid facts, return {"facts":[]}.

CRITICAL — attribution:
- Always name WHO did/said/owns the thing. Copy the speaker name exactly from the source.
- WRONG: "made a black and white bowl" — WHO made it?
- RIGHT: "Melanie made a black and white bowl in her pottery class."
- If two people are mentioned, be precise about which one is the subject.

BANNED content — reject any fact that:
- Describes your own reasoning process ("I need to...", "The instruction to convert...")
- Contains meta-commentary about the task or prompt
- Is a vague summary without specific names, dates, or details

RAW_MEMORIES:"#
            .to_string()
    })
}

fn deduction_prompt_template() -> &'static str {
    static TEMPLATE: OnceLock<String> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        r#"Task: infer NEW concrete facts from REFLECTED_FACTS by combining 2+ facts.

Output JSON only:
{"deductions":[{"content":"","confidence":0.0,"source_ids":["",""],"reasoning":"","needs_confirmation":null,"relation_type":"correlational","cause_ids":[],"effect_ids":[]}]}

Rules:
- Each deduction must combine >=2 source_ids to produce a NEW specific fact.
- Do not restate existing facts; generate only net-new inferences.
- confidence in [0.5, 0.9].
- relation_type = "causal" only with clear cause->effect evidence; otherwise "correlational".
- If confidence < 0.85, set needs_confirmation to a short question; else null.
- reasoning must be concise (max 18 words).
- Max 8 deductions.

BANNED patterns (reject any deduction matching these):
- "X correlates with Y" — vague correlation is not a useful fact.
- "X may ..." / "X might ..." / "X could ..." — speculation is not a fact.
- "X caused Y" unless there is direct evidence of causation in the source facts.
- Truisms or tautologies ("feeling happy correlates with feeling thankful").
- Restating a single fact with different words.

DATE HANDLING:
- Each fact has a date context from its source memories. Use THOSE dates for arithmetic, NOT today's date.
- "7 years" in a fact from 2023 means 2016, NOT 7 years before today.
- Always anchor relative durations to the source fact's date context.

GOOD deductions combine facts to derive something NEW and SPECIFIC:
- "Caroline moved to the US 4 years ago" (from: "Caroline has had friends for 4 years" + "Caroline moved from Sweden")
- "Melanie's son was about 5 during the camping trip" (from: age fact + trip date)

If no valid deductions exist, return {"deductions":[]}.

REFLECTED_FACTS:"#
            .to_string()
    })
}

fn induction_prompt_template() -> &'static str {
    static TEMPLATE: OnceLock<String> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        r#"Task: infer stable long-term patterns from FACTS (tier-2 and tier-3).

Output JSON only:
{"patterns":[{"content":"","confidence":0.0,"source_ids":["","",""],"pattern_type":"behavioral_pattern"}]}

Rules:
- Keep only stable patterns supported by >=3 independent source_ids.
- confidence in [0.5, 0.95].
- pattern_type must be one of:
  personality_trait, behavioral_pattern, core_value, recurring_interest, relationship_pattern, lifestyle_habit
- One concise present-tense sentence per pattern.
- Do not restate individual facts; synthesize across facts.
- Max 6 patterns.
- If no valid patterns, return {"patterns":[]}.

FACTS:"#
            .to_string()
    })
}

fn compact_prompt_text(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_leading_bracketed_timestamp(input: &str) -> &str {
    let s = input.trim();
    if s.starts_with('[') {
        if let Some(end) = s.find("] ") {
            return &s[end + 2..];
        }
    }
    s
}

/// Reject low-quality generated memory content (reflections, deductions, induced).
/// Returns true if the content should be REJECTED.
fn is_noisy_generated_content(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();

    // Meta-junk: LLM echoed prompt instructions as facts
    if lower.contains("i need to") || lower.contains("the instruction to")
        || lower.contains("i should") || lower.contains("let me")
        || lower.contains("the task") || lower.contains("as instructed")
    {
        return true;
    }

    // Vague correlational fluff (deductions)
    if lower.contains("correlates with") || lower.contains("correlate with")
        || lower.contains("may correlate")
    {
        return true;
    }

    // Too short to be useful
    if content.len() < 15 {
        return true;
    }

    false
}

/// Jaccard similarity between two strings (word-level).
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let set_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let set_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

fn truncate_prompt_text(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut out = String::new();
    for ch in input.chars().take(max_chars) {
        out.push(ch);
    }
    if let Some((prefix, _)) = out.rsplit_once(' ') {
        if prefix.len() >= (max_chars / 2) {
            return format!("{prefix}...");
        }
    }
    format!("{out}...")
}

fn build_reflection_prompt(memories: &[Memory]) -> String {
    let memories_block: String = memories
        .iter()
        .map(|m| {
            format!("[ID:{}] {}", m.id, compact_prompt_text(&m.content))
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{}\n{memories_block}", reflection_prompt_template())
}

#[derive(Debug, Deserialize)]
struct ReflectionResponse {
    facts: Vec<ReflectedFact>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReflectedFact {
    pub content: String,
    pub confidence: f64,
    pub source_ids: Vec<String>,
    pub corrections: Option<String>,
    /// ISO 8601 date (YYYY-MM-DD) if this fact describes a specific dated event; null otherwise.
    #[serde(default)]
    pub event_date: Option<String>,
    /// Most specific location mentioned (city, state/region, country — any that apply).
    /// Format: "City, State, Country" or subset. Null if no location is mentioned.
    #[serde(default)]
    pub location: Option<String>,
}

fn parse_reflection_response(response: &str) -> anyhow::Result<Vec<ReflectedFact>> {
    let mut s = response.trim();

    // Strip <think>...</think> blocks (Qwen thinking mode)
    if let Some(think_end) = s.find("</think>") {
        s = s[think_end + 8..].trim();
    }

    // Strip plain-text thinking preamble (e.g. "Thinking Process:\n...")
    // by finding the first '{' which starts the JSON.
    let json_str = if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            &s[start..=end]
        } else {
            &s[start..]
        }
    } else {
        s
    };

    // Try parsing; on trailing-chars error, use streaming deserializer to get first object.
    let parsed: ReflectionResponse = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            let mut de = serde_json::Deserializer::from_str(json_str).into_iter::<ReflectionResponse>();
            de.next()
                .ok_or_else(|| anyhow::anyhow!("No JSON object found in response: {}", &s[..s.len().min(500)]))?
                .map_err(|e| anyhow::anyhow!("JSON parse error: {e}\nResponse: {}", &s[..s.len().min(500)]))?
        }
    };

    // Filter out low-confidence facts
    Ok(parsed
        .facts
        .into_iter()
        .filter(|f| f.confidence >= 0.5 && f.content.len() >= 10)
        .take(12)
        .collect())
}

/// Synchronous reflection for the /api/v1/reflect endpoint (non-background).
/// Returns (extracted facts, IDs of created reflected memories).
pub async fn reflect_sync(
    store: &MemoryStore,
    embedder: &Arc<dyn Embedder>,
    llm: &Arc<dyn LlmClient>,
    namespace: &str,
    memory_ids: &[String],
) -> anyhow::Result<(Vec<ReflectedFact>, Vec<String>)> {
    let mut raw_memories: Vec<Memory> = Vec::new();
    for id in memory_ids {
        if let Ok(Some(mem)) = store.get(id).await {
            if mem.status == anima_core::memory::MemoryStatus::Active {
                raw_memories.push(mem);
            }
        }
    }

    if raw_memories.is_empty() {
        return Ok((vec![], vec![]));
    }

    let mut all_facts = Vec::new();
    let mut all_reflected_ids = Vec::new();
    let ns = anima_core::namespace::Namespace::parse(namespace)?;

    let clusters = semantic_cluster(raw_memories, embedder);
    for chunk in &clusters {
        let prompt = build_reflection_prompt(chunk);

        let response = llm
            .complete(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("LLM error: {e}"))?;

        let facts = parse_reflection_response(&response)?;

        for fact in &facts {
            // Reject noisy/meta-junk content
            if is_noisy_generated_content(&fact.content) {
                tracing::debug!("Rejected noisy reflection: {}", &fact.content[..fact.content.len().min(60)]);
                continue;
            }

            let raw_conf = fact.confidence.clamp(0.0, 1.0);
            let calibrated_conf = store
                .calibrate_confidence(&ns, PredictionKind::Extraction, raw_conf)
                .await
                .unwrap_or(raw_conf);
            if calibrated_conf < 0.5 {
                let _ = store
                    .record_calibration_observation(
                        &ns,
                        PredictionKind::Extraction,
                        None,
                        raw_conf,
                        Some(0.0),
                        Some(serde_json::json!({
                            "stage": "reflect_sync",
                            "reason": "below_threshold",
                            "source_count": fact.source_ids.len(),
                        })),
                    )
                    .await;
                continue;
            }

            let metadata = serde_json::json!({
                "tier": 2,
                "confidence": calibrated_conf,
                "raw_confidence": raw_conf,
                "source_ids": fact.source_ids,
                "corrections": fact.corrections,
                "event_date": fact.event_date,
                "location": fact.location,
            });

            let mut memory = Memory::new(
                namespace.to_string(),
                fact.content.clone(),
                Some(metadata),
                vec!["reflected".to_string()],
                Some("reflected".to_string()),
            );
            // Inherit episode_id from source memories (most common in chunk)
            {
                let mut ep_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
                for m in chunk.iter() {
                    if let Some(ep) = m.episode_id.as_deref() {
                        *ep_counts.entry(ep).or_insert(0) += 1;
                    }
                }
                if let Some((&best_ep, _)) = ep_counts.iter().max_by_key(|(_, c)| *c) {
                    memory.episode_id = Some(best_ep.to_string());
                }
            }

            let mem_id = memory.id.clone();
            let embedding = embedder
                .embed(&fact.content)
                .map_err(|e| anyhow::anyhow!("Embedding error: {e}"))?;

            store
                .insert(&memory, &embedding)
                .await
                .map_err(|e| anyhow::anyhow!("DB insert error: {e}"))?;
            all_reflected_ids.push(mem_id);
            let _ = store
                .record_calibration_observation(
                    &ns,
                    PredictionKind::Extraction,
                    all_reflected_ids.last().map(|s| s.as_str()),
                    raw_conf,
                    Some(1.0),
                    Some(serde_json::json!({
                        "stage": "reflect_sync",
                        "source_count": fact.source_ids.len(),
                    })),
                )
                .await;
        }

        // Mark sources as reflected
        for m in chunk {
            let meta = serde_json::json!({
                "tier": 1,
                "confidence": 1.0,
                "reflected": true,
            });
            let _ = store.update_json_metadata(&m.id, meta).await;
        }

        all_facts.extend(facts);
    }

    Ok((all_facts, all_reflected_ids))
}

// =============================================================================
// Deduction (Tier 3): Cross-memory inference from reflected facts
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct DeducedFact {
    pub content: String,
    pub confidence: f64,
    pub source_ids: Vec<String>,
    pub reasoning: String,
    pub needs_confirmation: Option<String>,
    #[serde(default = "default_relation_type")]
    pub relation_type: String,
    #[serde(default)]
    pub cause_ids: Vec<String>,
    #[serde(default)]
    pub effect_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DeductionResponse {
    deductions: Vec<DeducedFact>,
}

fn default_relation_type() -> String {
    "correlational".to_string()
}

/// Process deduction for a batch of reflected memory IDs.
/// Returns the IDs of newly created deduced memories.
async fn process_deduction(
    store: &MemoryStore,
    embedder: &Arc<dyn Embedder>,
    llm: &Arc<dyn LlmClient>,
    namespace: &str,
    reflected_ids: &[String],
) -> anyhow::Result<DeductionRunResult> {
    let ns = anima_core::namespace::Namespace::parse(namespace)?;
    let mut reflected: Vec<Memory> = Vec::new();
    for id in reflected_ids {
        if let Ok(Some(mem)) = store.get(id).await {
            if mem.status == anima_core::memory::MemoryStatus::Active {
                reflected.push(mem);
            }
        }
    }

    if reflected.len() < 2 {
        return Ok(DeductionRunResult {
            deduced_ids: vec![],
            usage: LlmStageUsage::default(),
        }); // Need at least 2 facts to deduce
    }

    tracing::info!(
        "Deducing from {} reflected facts in {namespace}",
        reflected.len()
    );

    let mut deduced_ids = Vec::new();
    let mut usage = LlmStageUsage::default();

    // Process chunked deduction calls with bounded concurrency.
    let llm_concurrency = processor_llm_concurrency();
    let semaphore = Arc::new(Semaphore::new(llm_concurrency));
    let mut join_set = JoinSet::new();
    for (idx, chunk) in reflected.chunks(20).map(|chunk| chunk.to_vec()).enumerate() {
        let llm = llm.clone();
        let prompt = build_deduction_prompt(&chunk);
        let permit = semaphore.clone().acquire_owned().await?;
        join_set.spawn(async move {
            let _permit = permit;
            (idx, chunk, llm.complete_with_usage(&prompt).await)
        });
    }

    let mut llm_batches = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(item) => llm_batches.push(item),
            Err(e) => tracing::warn!("Deduction worker task failed: {e}"),
        }
    }
    llm_batches.sort_by_key(|(idx, _, _)| *idx);

    for (_idx, chunk, completion) in llm_batches {
        let completion = match completion {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM deduction call failed: {e}");
                continue;
            }
        };
        usage.record_call(completion.usage);
        let response = completion.content;

        let deductions = match parse_deduction_response(&response) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to parse deduction response: {e}");
                continue;
            }
        };

        // Filter deductions by quality and confidence, build memories
        let mut pending_memories: Vec<Memory> = Vec::new();
        for deduction in &deductions {
            // Reject noisy content (correlates, meta-junk, etc.)
            if is_noisy_generated_content(&deduction.content) {
                tracing::debug!("Rejected noisy deduction: {}", &deduction.content[..deduction.content.len().min(60)]);
                continue;
            }

            let raw_conf = deduction.confidence.clamp(0.0, 1.0);
            let calibrated_conf = store
                .calibrate_confidence(&ns, PredictionKind::Deduction, raw_conf)
                .await
                .unwrap_or(raw_conf);
            if calibrated_conf < 0.5 {
                let _ = store
                    .record_calibration_observation(
                        &ns,
                        PredictionKind::Deduction,
                        None,
                        raw_conf,
                        Some(0.0),
                        Some(serde_json::json!({
                            "stage": "deduction",
                            "reason": "below_threshold",
                            "source_count": deduction.source_ids.len(),
                        })),
                    )
                    .await;
                continue;
            }

            let relation_type = match deduction.relation_type.trim().to_ascii_lowercase().as_str() {
                "causal" | "cause" | "causes" | "caused" => "causal",
                _ => "correlational",
            };
            let metadata = serde_json::json!({
                "tier": 3,
                "confidence": calibrated_conf,
                "raw_confidence": raw_conf,
                "source_ids": deduction.source_ids,
                "reasoning": deduction.reasoning,
                "needs_confirmation": deduction.needs_confirmation,
                "relation_type": relation_type,
                "cause_ids": deduction.cause_ids,
                "effect_ids": deduction.effect_ids,
            });

            let memory = Memory::new(
                namespace.to_string(),
                deduction.content.clone(),
                Some(metadata),
                vec!["deduced".to_string()],
                Some("deduced".to_string()),
            );
            pending_memories.push(memory);
        }

        // Batch-embed all accepted deductions at once
        if !pending_memories.is_empty() {
            let texts: Vec<&str> = pending_memories.iter().map(|m| m.content.as_str()).collect();
            let embeddings = match embedder.embed_batch(&texts) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Batch embedding failed for {} deduced facts: {e}", texts.len());
                    Vec::new()
                }
            };

            let mut batch_entries: Vec<(Memory, Vec<f32>)> = Vec::new();
            for (i, memory) in pending_memories.into_iter().enumerate() {
                let embedding = match embeddings.get(i) {
                    Some(e) => e.clone(),
                    None => {
                        tracing::warn!("Missing embedding for deduced fact {}", memory.id);
                        continue;
                    }
                };
                batch_entries.push((memory, embedding));
            }

            // Collect IDs and causal edge info before batch insert
            struct DeducedEntry {
                id: String,
                relation_type: String,
                confidence: f64,
                reasoning: String,
                cause_ids: Vec<String>,
                effect_ids: Vec<String>,
                source_ids: Vec<String>,
            }
            let mut edge_info: Vec<DeducedEntry> = Vec::new();
            for (_i, (memory, _)) in batch_entries.iter().enumerate() {
                // Find the matching deduction by content
                if let Some(deduction) = deductions.iter().find(|d| d.content == memory.content) {
                    let raw_conf = deduction.confidence.clamp(0.0, 1.0);
                    let calibrated_conf = store
                        .calibrate_confidence(&ns, PredictionKind::Deduction, raw_conf)
                        .await
                        .unwrap_or(raw_conf);
                    let relation_type = match deduction.relation_type.trim().to_ascii_lowercase().as_str() {
                        "causal" | "cause" | "causes" | "caused" => "causal",
                        _ => "correlational",
                    };
                    edge_info.push(DeducedEntry {
                        id: memory.id.clone(),
                        relation_type: relation_type.to_string(),
                        confidence: calibrated_conf,
                        reasoning: deduction.reasoning.clone(),
                        cause_ids: deduction.cause_ids.clone(),
                        effect_ids: deduction.effect_ids.clone(),
                        source_ids: deduction.source_ids.clone(),
                    });
                }
            }

            let batch_ids: Vec<String> = batch_entries.iter().map(|(m, _)| m.id.clone()).collect();
            if let Err(e) = store.insert_many(&batch_entries).await {
                tracing::warn!("Batch insert of {} deduced memories failed: {e}", batch_entries.len());
            } else {
                deduced_ids.extend(batch_ids);

                // Record causal edges after successful insert
                for entry in &edge_info {
                    let mut wrote_relation = false;
                    if !entry.cause_ids.is_empty() && !entry.effect_ids.is_empty() {
                        for cause in &entry.cause_ids {
                            for effect in &entry.effect_ids {
                                let _ = store
                                    .upsert_causal_edge(
                                        &ns, cause, effect,
                                        &entry.relation_type, entry.confidence,
                                        Some(&entry.reasoning),
                                    )
                                    .await;
                                wrote_relation = true;
                            }
                        }
                    }
                    if !wrote_relation {
                        for src in &entry.source_ids {
                            let _ = store
                                .upsert_causal_edge(
                                    &ns, src, &entry.id,
                                    &entry.relation_type, entry.confidence,
                                    Some(&entry.reasoning),
                                )
                                .await;
                        }
                    }
                }
            }
        }

        tracing::info!(
            "Deduced {} facts from {} reflected memories in {namespace}",
            deductions.len(),
            chunk.len()
        );
    }

    Ok(DeductionRunResult {
        deduced_ids,
        usage,
    })
}

fn build_deduction_prompt(facts: &[Memory]) -> String {
    let facts_block: String = facts
        .iter()
        .map(|m| {
            format!(
                "[ID:{}|c:{:.2}|date:{}] {}",
                m.id,
                extract_confidence(&m.metadata),
                m.created_at.format("%Y-%m"),
                compact_prompt_text(&m.content)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{}\n{facts_block}", deduction_prompt_template())
}

fn extract_confidence(metadata: &Option<serde_json::Value>) -> f64 {
    metadata
        .as_ref()
        .and_then(|v| v.get("confidence"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0)
}

fn parse_deduction_response(response: &str) -> anyhow::Result<Vec<DeducedFact>> {
    let mut s = response.trim();

    // Strip <think>...</think> blocks
    if let Some(think_end) = s.find("</think>") {
        s = s[think_end + 8..].trim();
    }

    let json_str = if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            &s[start..=end]
        } else {
            &s[start..]
        }
    } else {
        s
    };

    let parsed: DeductionResponse = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            let mut de = serde_json::Deserializer::from_str(json_str).into_iter::<DeductionResponse>();
            de.next()
                .ok_or_else(|| anyhow::anyhow!("No JSON object found in response: {}", &s[..s.len().min(500)]))?
                .map_err(|e| anyhow::anyhow!("JSON parse error: {e}\nResponse: {}", &s[..s.len().min(500)]))?
        }
    };

    // Filter: require at least 2 source IDs, confidence >= 0.5
    Ok(parsed
        .deductions
        .into_iter()
        .filter(|d| d.source_ids.len() >= 2 && d.confidence >= 0.5 && d.content.len() >= 10)
        .take(8)
        .collect())
}

// =============================================================================
// Induction (Tier 4): Behavioural patterns and personality profiles
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct InducedFact {
    pub content: String,
    pub confidence: f64,
    pub source_ids: Vec<String>,
    pub pattern_type: String,
}

#[derive(Debug, Deserialize)]
struct InductionResponse {
    patterns: Vec<InducedFact>,
}

/// Minimum growth ratio of the fact pool before re-running induction.
/// 1.25 means we only re-induce when there are ≥25% more facts than last time.
const INDUCTION_REGROWTH_THRESHOLD: f64 = 1.25;

#[derive(Debug, Clone)]
struct PromptFact {
    id: String,
    tier: i32,
    content: String,
}

fn induction_max_facts_for_prompt() -> usize {
    std::env::var("ANIMA_INDUCTION_MAX_FACTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(24)
        .clamp(8, 96)
}

fn induction_max_chars_per_fact() -> usize {
    std::env::var("ANIMA_INDUCTION_MAX_CHARS_PER_FACT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(160)
        .clamp(64, 512)
}

fn induction_dedup_jaccard_threshold() -> f64 {
    std::env::var("ANIMA_INDUCTION_DEDUP_JACCARD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.92)
        .clamp(0.75, 0.99)
}

fn compress_induction_facts_for_prompt(facts: &[Memory]) -> Vec<PromptFact> {
    let max_facts = induction_max_facts_for_prompt();
    let max_chars = induction_max_chars_per_fact();
    let dedup_threshold = induction_dedup_jaccard_threshold();

    let mut out: Vec<PromptFact> = Vec::new();
    let mut seen_norm: Vec<String> = Vec::new();

    for fact in facts {
        if out.len() >= max_facts {
            break;
        }

        let normalized = normalized_claim_text(&fact.content);
        if normalized.is_empty() {
            continue;
        }

        let is_dup = seen_norm
            .iter()
            .any(|prev| prev == &normalized || token_jaccard(prev, &normalized) >= dedup_threshold);
        if is_dup {
            continue;
        }
        seen_norm.push(normalized);

        let no_ts = strip_leading_bracketed_timestamp(&fact.content);
        let compact = compact_prompt_text(no_ts);
        let shortened = truncate_prompt_text(&compact, max_chars);
        if shortened.len() < 10 {
            continue;
        }

        out.push(PromptFact {
            id: fact.id.clone(),
            tier: memory_tier(fact),
            content: shortened,
        });
    }

    out
}

/// Process induction for a namespace: find recurring patterns across ALL
/// deduced facts and produce high-level personality/behavioural profiles.
///
/// Skips if existing induced memories already cover the current fact pool
/// (checked via `facts_used_count` stored in each induced memory's metadata).
async fn process_induction(
    store: &MemoryStore,
    embedder: &Arc<dyn Embedder>,
    llm: &Arc<dyn LlmClient>,
    namespace: &str,
) -> anyhow::Result<InductionRunResult> {
    let mut usage = LlmStageUsage::default();
    let ns = anima_core::namespace::Namespace::parse(namespace)?;

    // Gather all tier-2 and tier-3 memories in the namespace
    let deduced = store.find_by_tier(&ns, 3, 200).await?;
    let reflected = store.find_by_tier(&ns, 2, 100).await?;

    let mut all_facts: Vec<Memory> = reflected;
    all_facts.extend(deduced);

    if all_facts.len() < 5 {
        tracing::info!("Skipping induction for {namespace}: only {} facts", all_facts.len());
        return Ok(InductionRunResult { usage });
    }

    // Check if existing induced memories already cover this fact pool.
    // We look at the highest `facts_used_count` among existing tier-4 memories.
    let existing_induced = store.find_by_tier(&ns, 4, 500).await?;
    let prev_facts_used = existing_induced
        .iter()
        .filter_map(|m| {
            m.metadata.as_ref()
                .and_then(|v| v.get("facts_used_count"))
                .and_then(|v| v.as_u64())
        })
        .max()
        .unwrap_or(0) as usize;

    let min_new_count = ((prev_facts_used as f64) * INDUCTION_REGROWTH_THRESHOLD) as usize;
    if !existing_induced.is_empty() && all_facts.len() <= min_new_count {
        tracing::info!(
            "Skipping induction for {namespace}: {} facts (last run used {}, threshold {})",
            all_facts.len(), prev_facts_used, min_new_count
        );
        return Ok(InductionRunResult { usage });
    }

    tracing::info!(
        "Inducing patterns from {} facts in {namespace} (prev: {})",
        all_facts.len(), prev_facts_used
    );

    // Take highest-confidence facts, then compress/deduplicate for prompt budget.
    all_facts.sort_by(|a, b| {
        let ca = extract_confidence(&a.metadata);
        let cb = extract_confidence(&b.metadata);
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_facts = &all_facts[..all_facts.len().min(96)];
    let prompt_facts = compress_induction_facts_for_prompt(top_facts);
    let facts_used_count = prompt_facts.len();
    if facts_used_count < 5 {
        tracing::info!(
            "Skipping induction for {namespace}: only {} compressed facts after dedup",
            facts_used_count
        );
        return Ok(InductionRunResult { usage });
    }

    let prompt = build_induction_prompt(&prompt_facts);

    let completion = match llm.complete_with_usage(&prompt).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("LLM induction call failed: {e}");
            return Ok(InductionRunResult { usage });
        }
    };
    usage.record_call(completion.usage);
    let response = completion.content;

    let patterns = match parse_induction_response(&response) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Failed to parse induction response: {e}");
            return Ok(InductionRunResult { usage });
        }
    };

    // Build list of existing induced content for dedup
    let existing_content: Vec<String> = existing_induced.iter().map(|m| m.content.clone()).collect();

    // Filter patterns by quality, dedup, and confidence
    let mut pending_memories: Vec<Memory> = Vec::new();
    for pattern in &patterns {
        // Reject noisy content
        if is_noisy_generated_content(&pattern.content) {
            tracing::debug!("Rejected noisy induced: {}", &pattern.content[..pattern.content.len().min(60)]);
            continue;
        }

        // Dedup against existing induced memories (Jaccard > 0.85 = duplicate)
        let is_dup = existing_content.iter().any(|existing| {
            jaccard_similarity(&pattern.content, existing) > 0.85
        });
        if is_dup {
            tracing::debug!("Skipped duplicate induced: {}", &pattern.content[..pattern.content.len().min(60)]);
            continue;
        }

        let raw_conf = pattern.confidence.clamp(0.0, 1.0);
        let calibrated_conf = store
            .calibrate_confidence(&ns, PredictionKind::Induction, raw_conf)
            .await
            .unwrap_or(raw_conf);
        if calibrated_conf < 0.5 {
            let _ = store
                .record_calibration_observation(
                    &ns,
                    PredictionKind::Induction,
                    None,
                    raw_conf,
                    Some(0.0),
                    Some(serde_json::json!({
                        "stage": "induction",
                        "reason": "below_threshold",
                        "source_count": pattern.source_ids.len(),
                    })),
                )
                .await;
            continue;
        }

        let metadata = serde_json::json!({
            "tier": 4,
            "confidence": calibrated_conf,
            "raw_confidence": raw_conf,
            "source_ids": pattern.source_ids,
            "pattern_type": pattern.pattern_type,
            "facts_used_count": facts_used_count,
        });

        let memory = Memory::new(
            namespace.to_string(),
            pattern.content.clone(),
            Some(metadata),
            vec!["induced".to_string()],
            Some("induced".to_string()),
        );
        pending_memories.push(memory);
    }

    // Batch-embed all accepted patterns at once
    if !pending_memories.is_empty() {
        let texts: Vec<&str> = pending_memories.iter().map(|m| m.content.as_str()).collect();
        let embeddings = match embedder.embed_batch(&texts) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Batch embedding failed for {} induced patterns: {e}", texts.len());
                Vec::new()
            }
        };

        let mut batch_entries: Vec<(Memory, Vec<f32>)> = Vec::new();
        for (i, memory) in pending_memories.into_iter().enumerate() {
            let embedding = match embeddings.get(i) {
                Some(e) => e.clone(),
                None => {
                    tracing::warn!("Missing embedding for induced pattern {}", memory.id);
                    continue;
                }
            };
            batch_entries.push((memory, embedding));
        }

        if let Err(e) = store.insert_many(&batch_entries).await {
            tracing::warn!("Batch insert of {} induced memories failed: {e}", batch_entries.len());
        }
    }

    tracing::info!(
        "Induced {} patterns from {} facts in {namespace}",
        patterns.len(),
        facts_used_count
    );

    Ok(InductionRunResult { usage })
}

fn build_induction_prompt(facts: &[PromptFact]) -> String {
    let facts_block: String = facts
        .iter()
        .map(|m| {
            format!("[ID:{}|t:{}] {}", m.id, m.tier, m.content)
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{}\n{facts_block}", induction_prompt_template())
}

fn parse_induction_response(response: &str) -> anyhow::Result<Vec<InducedFact>> {
    let mut s = response.trim();

    if let Some(think_end) = s.find("</think>") {
        s = s[think_end + 8..].trim();
    }

    let json_str = if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            &s[start..=end]
        } else {
            s
        }
    } else {
        s
    };

    let parsed: InductionResponse = serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("JSON parse error: {e}\nResponse: {}", &s[..s.len().min(500)]))?;

    Ok(parsed
        .patterns
        .into_iter()
        .filter(|p| p.source_ids.len() >= 3 && p.confidence >= 0.5 && p.content.len() >= 10)
        .take(6)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anima_core::memory::MemoryStatus;

    #[test]
    fn working_commit_score_rewards_stable_claims() {
        let stable = working_memory_commit_score(0.45, "I always check battery telemetry before navigation.");
        let uncertain = working_memory_commit_score(0.45, "I think maybe I like battery telemetry.");
        assert!(stable > uncertain, "stable claim score should exceed uncertain phrasing");
        assert!(stable > 0.5);
    }

    #[test]
    fn supersede_match_for_near_duplicates() {
        let now = Utc::now();
        let older = Memory {
            id: "a".into(),
            namespace: "test/ns".into(),
            content: "User prefers Rust for robotics control software.".into(),
            metadata: Some(serde_json::json!({"tier": 2})),
            tags: vec![],
            memory_type: "reflected".into(),
            status: MemoryStatus::Active,
            created_at: now - chrono::Duration::seconds(10),
            updated_at: now,
            accessed_at: now,
            access_count: 0,
            importance: 5,
            episode_id: None,
            event_date: None,
            hash: "h1".into(),
        };
        let newer = Memory {
            id: "b".into(),
            namespace: "test/ns".into(),
            content: "User prefers Rust for robotics control software".into(),
            metadata: Some(serde_json::json!({"tier": 2})),
            tags: vec![],
            memory_type: "reflected".into(),
            status: MemoryStatus::Active,
            created_at: now,
            updated_at: now,
            accessed_at: now,
            access_count: 0,
            importance: 5,
            episode_id: None,
            event_date: None,
            hash: "h2".into(),
        };
        assert!(should_supersede(&older, &newer));
    }
}
