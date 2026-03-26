use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use anima_core::memory::{content_hash, Memory, MemoryStatus};
use anima_core::namespace::Namespace;
use anima_core::search::{HybridScorer, ScoredResult, ScorerConfig, SearchMode};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::fts;
use crate::pool::{DbError, DbPool};
use crate::vector;

// --- New types for web UI API ---

#[derive(Debug, Serialize)]
pub struct NamespaceStats {
    pub total: u64,
    pub active: u64,
    pub superseded: u64,
    pub deleted: u64,
    pub avg_access_count: f64,
    pub max_access_count: u64,
    pub oldest_memory: Option<String>,
    pub newest_memory: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct NamespaceInfo {
    pub namespace: String,
    pub total_count: u64,
    pub active_count: u64,
}

#[derive(Debug, Serialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub content: String,
    pub namespace: String,
    pub status: String,
    pub memory_type: String,
    pub importance: i32,
    pub created_at: String,
    pub access_count: u64,
}

#[derive(Debug, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub similarity: f64,
    pub edge_type: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CausalEdge {
    pub id: String,
    pub namespace: String,
    pub source_memory_id: String,
    pub target_memory_id: String,
    pub relation_type: String,
    pub confidence: f64,
    pub evidence: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StateTransitionEdge {
    pub id: String,
    pub namespace: String,
    pub from_memory_id: String,
    pub to_memory_id: String,
    pub transition_type: String,
    pub confidence: f64,
    pub evidence: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CounterfactualHypothesis {
    pub memory_id: String,
    pub statement: String,
    pub confidence_low: f64,
    pub confidence_mid: f64,
    pub confidence_high: f64,
    pub evidence_ids: Vec<String>,
    pub source_kinds: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CounterfactualSimulation {
    pub intervention: String,
    pub seed_memory_ids: Vec<String>,
    pub hypotheses: Vec<CounterfactualHypothesis>,
    pub considered_nodes: usize,
    pub generated_at: String,
}

/// Raw embedding data for visualization / PCA.
#[derive(Debug)]
pub struct RawEmbeddingData {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub embedding: Vec<f32>,
}

// --- Conversation types ---

#[derive(Debug, Clone, Serialize)]
pub struct Conversation {
    pub id: String,
    pub namespace: String,
    pub title: String,
    pub mode: String,
    pub messages: String, // JSON array
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub mode: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkingMemoryEntry {
    pub id: String,
    pub namespace: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub provisional_score: f64,
    pub status: String,
    pub conversation_id: Option<String>,
    pub expires_at: Option<String>,
    pub committed_memory_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// Calibrated prediction domains.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionKind {
    Extraction,
    Deduction,
    Induction,
    ProcedureSelection,
    RetrievalRelevance,
}

impl PredictionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Extraction => "extraction",
            Self::Deduction => "deduction",
            Self::Induction => "induction",
            Self::ProcedureSelection => "procedure_selection",
            Self::RetrievalRelevance => "retrieval_relevance",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationModel {
    pub namespace: String,
    pub prediction_kind: String,
    pub samples: u64,
    pub avg_prediction: f64,
    pub avg_outcome: f64,
    pub mse: f64,
    pub slope: f64,
    pub intercept: f64,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    pub namespace: String,
    pub prediction_kind: String,
    pub bin_index: i32,
    pub sample_count: u64,
    pub avg_prediction: f64,
    pub avg_outcome: f64,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationMetrics {
    pub namespace: String,
    pub models: Vec<CalibrationModel>,
    pub bins: Vec<CalibrationBin>,
    pub unresolved_observations: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrectionEvent {
    pub id: String,
    pub namespace: String,
    pub target_memory_id: String,
    pub new_memory_id: String,
    pub reason: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClaimRevision {
    pub id: String,
    pub namespace: String,
    pub memory_id: String,
    pub revision_number: i64,
    pub operation: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    pub memory_type: String,
    pub importance: i32,
    pub status: String,
    pub superseded_by: Option<String>,
    pub hash: String,
    pub actor: Option<String>,
    pub reason: Option<String>,
    pub provenance: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcedureRevision {
    pub id: String,
    pub namespace: String,
    pub procedure_name: String,
    pub revision_number: i64,
    pub operation: String,
    pub spec: serde_json::Value,
    pub actor: Option<String>,
    pub reason: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    pub id: String,
    pub namespace: String,
    pub entity_type: String,
    pub entity_id: String,
    pub operation: String,
    pub actor: Option<String>,
    pub reason: Option<String>,
    pub details: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContradictionEntry {
    pub id: String,
    pub namespace: String,
    pub old_memory_id: String,
    pub new_memory_id: String,
    pub resolution: String,
    pub provenance: Option<serde_json::Value>,
    pub created_at: String,
    /// Content of the old (superseded) memory, if still in DB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_content: Option<String>,
    /// Content of the new (superseding) memory, if still in DB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SupersessionLink {
    pub memory_id: String,
    pub content: String,
    pub status: String,
    pub superseded_by: Option<String>,
    pub confidence: f64,
    pub source: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IdentityEntity {
    pub id: String,
    pub namespace: String,
    pub canonical_name: String,
    pub normalized_name: String,
    pub language: String,
    pub confidence: f64,
    pub ambiguity: f64,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IdentityCandidate {
    pub entity_id: String,
    pub canonical_name: String,
    pub matched_alias: String,
    pub language: String,
    pub match_kind: String,
    pub score: f64,
    pub entity_confidence: f64,
    pub alias_confidence: f64,
    pub ambiguity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IdentityResolution {
    pub query: String,
    pub normalized_query: String,
    pub best_confidence: f64,
    pub ambiguous: bool,
    pub candidates: Vec<IdentityCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanTrace {
    pub id: String,
    pub namespace: String,
    pub goal: String,
    pub status: String,
    pub priority: i32,
    pub due_at: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub outcome: Option<String>,
    pub outcome_confidence: Option<f64>,
    pub finished_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanCheckpoint {
    pub id: String,
    pub namespace: String,
    pub plan_id: String,
    pub checkpoint_key: String,
    pub title: String,
    pub order_index: i32,
    pub status: String,
    pub expected_by: Option<String>,
    pub completed_at: Option<String>,
    pub evidence: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanRecoveryBranch {
    pub id: String,
    pub namespace: String,
    pub plan_id: String,
    pub source_checkpoint_id: Option<String>,
    pub branch_label: String,
    pub trigger_reason: String,
    pub status: String,
    pub branch_plan: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub resolution_notes: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub resolved_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanProcedureBinding {
    pub id: String,
    pub namespace: String,
    pub plan_id: String,
    pub procedure_name: String,
    pub binding_role: String,
    pub confidence: f64,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryPatch {
    pub content: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub memory_type: Option<String>,
    pub importance: Option<i32>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryMutationResult {
    pub memory: Memory,
    pub revision_number: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryMergeResult {
    pub merged_memory_id: String,
    pub superseded_source_ids: Vec<String>,
    pub merged_revision_number: i64,
}

/// High-level memory store operations.
#[derive(Clone)]
pub struct MemoryStore {
    pool: Arc<DbPool>,
}

impl MemoryStore {
    pub fn new(pool: Arc<DbPool>) -> Self {
        Self { pool }
    }

    /// Quick health check: verify the database is accessible.
    pub async fn ping(&self) -> Result<(), DbError> {
        self.pool.ping().await
    }

    /// Insert a new memory with three-table sync.
    pub async fn insert(
        &self,
        memory: &Memory,
        embedding: &[f32],
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        insert_memory_sync(&conn, memory, embedding)
    }

    /// Insert many memories in a single transaction (high-throughput ingest path).
    pub async fn insert_many(
        &self,
        entries: &[(Memory, Vec<f32>)],
    ) -> Result<(), DbError> {
        if entries.is_empty() {
            return Ok(());
        }
        let conn = self.pool.writer().await;
        insert_many_memories_sync(&conn, entries)
    }

    /// Check for exact duplicate by content hash within a namespace.
    pub async fn find_by_hash(
        &self,
        namespace: &Namespace,
        hash: &str,
    ) -> Result<Option<Memory>, DbError> {
        let conn = self.pool.writer().await;
        find_by_hash_sync(&conn, namespace, hash)
    }

    /// Get a single memory by ID.
    pub async fn get(&self, id: &str) -> Result<Option<Memory>, DbError> {
        let conn = self.pool.writer().await;
        get_memory_sync(&conn, id)
    }

    /// Update a memory's content (re-embeds and syncs all three tables).
    pub async fn update_content(
        &self,
        id: &str,
        new_content: &str,
        new_embedding: &[f32],
    ) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        update_content_sync(&conn, id, new_content, new_embedding)
    }

    /// Update a memory's metadata (type, importance, tags) without re-embedding.
    pub async fn update_metadata(
        &self,
        id: &str,
        memory_type: Option<&str>,
        importance: Option<i32>,
        tags: Option<&[String]>,
    ) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        update_metadata_sync(&conn, id, memory_type, importance, tags)
    }

    /// Soft-delete a memory (sets status to 'deleted', removes from vec + fts).
    pub async fn soft_delete(&self, id: &str) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        soft_delete_sync(&conn, id)
    }

    /// Hard-delete a memory (removes row entirely from db, vec + fts).
    pub async fn hard_delete(&self, id: &str) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        hard_delete_sync(&conn, id)
    }

    /// Purge all soft-deleted memories from the database.
    pub async fn purge_deleted(&self) -> Result<u64, DbError> {
        let conn = self.pool.writer().await;
        let count = conn.execute(
            "DELETE FROM memories WHERE status = 'deleted'",
            [],
        ).map_err(DbError::Sqlite)?;
        Ok(count as u64)
    }

    /// Mark a memory as superseded (for consolidation).
    pub async fn mark_superseded(&self, id: &str, superseded_by: &str) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        mark_superseded_sync(&conn, id, superseded_by)
    }

    /// Hybrid search: vector + FTS5 + temporal decay.
    pub async fn search(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        namespace: &Namespace,
        mode: &SearchMode,
        limit: usize,
        scorer_config: &ScorerConfig,
    ) -> Result<Vec<ScoredResult>, DbError> {
        if self.pool.db_path() == ":memory:" {
            let conn = self.pool.writer().await;
            search_sync(
                &conn,
                query_embedding,
                query_text,
                namespace,
                mode,
                limit,
                scorer_config,
            )
        } else {
            let conn = self.pool.reader()?;
            search_sync(
                &conn,
                query_embedding,
                query_text,
                namespace,
                mode,
                limit,
                scorer_config,
            )
        }
    }

    /// Record a prediction/outcome pair (or pending prediction when outcome is None).
    pub async fn record_calibration_observation(
        &self,
        namespace: &Namespace,
        kind: PredictionKind,
        prediction_id: Option<&str>,
        predicted_confidence: f64,
        outcome: Option<f64>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        insert_calibration_observation_sync(
            &conn,
            namespace,
            kind,
            prediction_id,
            predicted_confidence,
            outcome,
            metadata,
        )
    }

    /// Recompute calibration models and reliability bins from observations.
    pub async fn recompute_calibration_models(&self) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        recompute_calibration_models_sync(&conn)
    }

    /// Apply calibrated confidence for a given prediction domain.
    pub async fn calibrate_confidence(
        &self,
        namespace: &Namespace,
        kind: PredictionKind,
        raw_confidence: f64,
    ) -> Result<f64, DbError> {
        let conn = self.pool.writer().await;
        Ok(calibrate_confidence_sync(&conn, namespace, kind, raw_confidence)?)
    }

    /// Return calibration summary for dashboards/API.
    pub async fn calibration_metrics(
        &self,
        namespace: &Namespace,
    ) -> Result<CalibrationMetrics, DbError> {
        let conn = self.pool.writer().await;
        calibration_metrics_sync(&conn, namespace)
    }

    /// Compute optimal hybrid weights (weight_vector, weight_keyword) from calibration
    /// observations. Analyzes retrieval observations with component scores (vector_score,
    /// keyword_score) and computes the correlation between each component and positive
    /// outcomes. Returns None if insufficient data (<50 observations with component scores).
    pub async fn compute_optimal_hybrid_weights(
        &self,
    ) -> Result<Option<(f64, f64)>, DbError> {
        let conn = self.pool.writer().await;
        compute_optimal_hybrid_weights_sync(&conn)
    }

    pub async fn record_correction_event(
        &self,
        namespace: &Namespace,
        target_memory_id: &str,
        new_memory_id: &str,
        reason: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<CorrectionEvent, DbError> {
        let conn = self.pool.writer().await;
        insert_correction_event_sync(
            &conn,
            namespace,
            target_memory_id,
            new_memory_id,
            reason,
            metadata,
        )
    }

    pub async fn record_contradiction_resolution(
        &self,
        namespace: &Namespace,
        old_memory_id: &str,
        new_memory_id: &str,
        resolution: &str,
        provenance: Option<serde_json::Value>,
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        insert_contradiction_ledger_sync(
            &conn,
            namespace,
            old_memory_id,
            new_memory_id,
            resolution,
            provenance,
        )
    }

    /// List contradiction ledger entries for a namespace, ordered by most recent first.
    pub async fn list_contradictions(
        &self,
        namespace: &Namespace,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<ContradictionEntry>, DbError> {
        let conn = self.pool.writer().await;
        list_contradictions_sync(&conn, namespace, limit, offset)
    }

    /// Find contradictions involving any of the given memory IDs
    /// (as either old or new side). Returns entries where the retrieved
    /// memory was superseded OR superseded something else.
    pub async fn find_contradictions_for_memories(
        &self,
        namespace: &Namespace,
        memory_ids: &[String],
    ) -> Result<Vec<ContradictionEntry>, DbError> {
        if memory_ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.pool.writer().await;
        find_contradictions_for_memories_sync(&conn, namespace, memory_ids)
    }

    /// Get the supersession chain for a memory: all predecessors and successors.
    pub async fn get_supersession_chain(
        &self,
        namespace: &Namespace,
        memory_id: &str,
    ) -> Result<Vec<SupersessionLink>, DbError> {
        let conn = self.pool.writer().await;
        get_supersession_chain_sync(&conn, namespace, memory_id)
    }

    pub async fn upsert_causal_edge(
        &self,
        namespace: &Namespace,
        source_memory_id: &str,
        target_memory_id: &str,
        relation_type: &str,
        confidence: f64,
        evidence: Option<&str>,
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        upsert_causal_edge_sync(
            &conn,
            namespace,
            source_memory_id,
            target_memory_id,
            relation_type,
            confidence,
            evidence,
        )
    }

    pub async fn list_causal_edges(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<CausalEdge>, DbError> {
        let conn = self.pool.writer().await;
        list_causal_edges_sync(&conn, namespace, limit)
    }

    /// Find memory IDs linked to given entity IDs via causal_edges (relation_type = "mentioned_in").
    /// Returns up to `limit` unique target_memory_ids.
    pub async fn find_memories_by_entity_ids(
        &self,
        namespace: &Namespace,
        entity_ids: &[String],
        limit: usize,
    ) -> Result<Vec<String>, DbError> {
        let conn = self.pool.reader()?;
        find_memories_by_entity_ids_sync(&conn, namespace, entity_ids, limit)
    }

    pub async fn upsert_state_transition(
        &self,
        namespace: &Namespace,
        from_memory_id: &str,
        to_memory_id: &str,
        transition_type: &str,
        confidence: f64,
        evidence: Option<&str>,
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        upsert_state_transition_sync(
            &conn,
            namespace,
            from_memory_id,
            to_memory_id,
            transition_type,
            confidence,
            evidence,
        )
    }

    pub async fn list_state_transitions(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<StateTransitionEdge>, DbError> {
        let conn = self.pool.writer().await;
        list_state_transitions_sync(&conn, namespace, limit)
    }

    pub async fn simulate_counterfactual(
        &self,
        namespace: &Namespace,
        intervention: &str,
        seed_memory_ids: &[String],
        max_hops: usize,
        top_k: usize,
        include_correlational: bool,
        include_transitions: bool,
    ) -> Result<CounterfactualSimulation, DbError> {
        if self.pool.db_path() == ":memory:" {
            let conn = self.pool.writer().await;
            simulate_counterfactual_sync(
                &conn,
                namespace,
                intervention,
                seed_memory_ids,
                max_hops,
                top_k,
                include_correlational,
                include_transitions,
            )
        } else {
            let conn = self.pool.reader()?;
            simulate_counterfactual_sync(
                &conn,
                namespace,
                intervention,
                seed_memory_ids,
                max_hops,
                top_k,
                include_correlational,
                include_transitions,
            )
        }
    }

    /// Find similar memories by vector search (for consolidation).
    pub async fn find_similar(
        &self,
        embedding: &[f32],
        namespace: &Namespace,
        limit: usize,
        threshold: f64,
    ) -> Result<Vec<(Memory, f64)>, DbError> {
        let conn = self.pool.writer().await;
        find_similar_sync(&conn, embedding, namespace, limit, threshold)
    }

    /// Find graph neighbors of a memory by its stored embedding.
    pub async fn find_neighbors(
        &self,
        memory_id: &str,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<(Memory, f64)>, DbError> {
        let conn = self.pool.writer().await;
        find_neighbors_sync(&conn, memory_id, limit, min_similarity)
    }

    /// List memories with pagination and optional status/type filter.
    pub async fn list(
        &self,
        namespace: &Namespace,
        status: Option<&str>,
        memory_type: Option<&str>,
        category: Option<&str>,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<Memory>, u64), DbError> {
        let conn = self.pool.writer().await;
        list_sync(&conn, namespace, status, memory_type, category, offset, limit)
    }

    /// Get namespace stats.
    pub async fn stats(&self, namespace: &Namespace) -> Result<NamespaceStats, DbError> {
        let conn = self.pool.writer().await;
        stats_sync(&conn, namespace)
    }

    /// Get all namespaces with counts.
    pub async fn list_namespaces(&self) -> Result<Vec<NamespaceInfo>, DbError> {
        let conn = self.pool.writer().await;
        list_namespaces_sync(&conn)
    }

    /// Update just the embedding blob for a memory (for re-indexing).
    pub async fn update_embedding_blob(&self, id: &str, embedding: &[f32]) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            params![blob, id],
        )?;
        Ok(())
    }

    /// Force rebuild the vec index with a new dimension.
    pub async fn force_reindex(&self, dimension: usize) -> Result<usize, DbError> {
        let conn = self.pool.writer().await;
        vector::force_reindex(&conn, dimension).map_err(DbError::Sqlite)
    }

    /// Delete an entire namespace and all its data across all tables.
    pub async fn delete_namespace(&self, namespace: &Namespace) -> Result<u64, DbError> {
        let conn = self.pool.writer().await;
        delete_namespace_sync(&conn, namespace)
    }

    /// Rename a namespace: update all records from old_ns to new_ns across all tables.
    pub async fn rename_namespace(&self, old_ns: &Namespace, new_ns: &Namespace) -> Result<u64, DbError> {
        let conn = self.pool.writer().await;
        rename_namespace_sync(&conn, old_ns, new_ns)
    }

    // --- Working memory methods ---

    pub async fn add_working_memory(
        &self,
        namespace: &Namespace,
        content: &str,
        provisional_score: f64,
        metadata: Option<serde_json::Value>,
        conversation_id: Option<&str>,
        expires_at: Option<&str>,
    ) -> Result<WorkingMemoryEntry, DbError> {
        let conn = self.pool.writer().await;
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now().to_rfc3339();
        let metadata_json = metadata
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        conn.execute(
            "INSERT INTO working_memories
                (id, namespace, content, metadata, provisional_score, status, conversation_id, expires_at, committed_memory_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 'pending', ?6, ?7, NULL, ?8, ?8)",
            params![
                id,
                namespace.as_str(),
                content,
                metadata_json,
                provisional_score.clamp(0.0, 1.0),
                conversation_id,
                expires_at,
                now
            ],
        )
        .map_err(DbError::Sqlite)?;
        Ok(WorkingMemoryEntry {
            id,
            namespace: namespace.as_str().to_string(),
            content: content.to_string(),
            metadata,
            provisional_score: provisional_score.clamp(0.0, 1.0),
            status: "pending".to_string(),
            conversation_id: conversation_id.map(|s| s.to_string()),
            expires_at: expires_at.map(|s| s.to_string()),
            committed_memory_id: None,
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub async fn list_working_memories(
        &self,
        namespace: &Namespace,
        status: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<WorkingMemoryEntry>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let status_value = status.unwrap_or("");
        let conversation_value = conversation_id.unwrap_or("");
        let mut stmt = conn
            .prepare(
                "SELECT id, namespace, content, metadata, provisional_score, status, conversation_id, expires_at, committed_memory_id, created_at, updated_at
                 FROM working_memories
                 WHERE namespace LIKE ?1
                   AND (?2 = '' OR status = ?2)
                   AND (?3 = '' OR conversation_id = ?3)
                 ORDER BY created_at ASC
                 LIMIT ?4",
            )
            .map_err(DbError::Sqlite)?;
        let rows = stmt
            .query_map(
                params![ns_pattern, status_value, conversation_value, limit as i64],
                |row| {
                    let metadata: Option<String> = row.get(3)?;
                    Ok(WorkingMemoryEntry {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        content: row.get(2)?,
                        metadata: metadata.and_then(|s| serde_json::from_str(&s).ok()),
                        provisional_score: row.get(4)?,
                        status: row.get(5)?,
                        conversation_id: row.get(6)?,
                        expires_at: row.get(7)?,
                        committed_memory_id: row.get(8)?,
                        created_at: row.get(9)?,
                        updated_at: row.get(10)?,
                    })
                },
            )
            .map_err(DbError::Sqlite)?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub async fn update_working_memory_state(
        &self,
        id: &str,
        provisional_score: Option<f64>,
        status: Option<&str>,
        committed_memory_id: Option<&str>,
    ) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        let score_value = provisional_score.unwrap_or(-1.0);
        let status_value = status.unwrap_or("");
        let now = Utc::now().to_rfc3339();
        let rows = conn
            .execute(
                "UPDATE working_memories
                 SET provisional_score = CASE WHEN ?1 < 0.0 THEN provisional_score ELSE ?1 END,
                     status = CASE WHEN ?2 = '' THEN status ELSE ?2 END,
                     committed_memory_id = COALESCE(?3, committed_memory_id),
                     updated_at = ?4
                 WHERE id = ?5",
                params![score_value, status_value, committed_memory_id, now, id],
            )
            .map_err(DbError::Sqlite)?;
        Ok(rows > 0)
    }

    pub async fn expire_working_memories(&self, now_iso: &str) -> Result<u64, DbError> {
        let conn = self.pool.writer().await;
        let rows = conn
            .execute(
                "UPDATE working_memories
                 SET status = 'expired', updated_at = ?1
                 WHERE status = 'pending'
                   AND expires_at IS NOT NULL
                   AND expires_at <= ?2",
                params![now_iso, now_iso],
            )
            .map_err(DbError::Sqlite)?;
        Ok(rows as u64)
    }

    /// Build similarity graph data.
    pub async fn similarity_graph(
        &self,
        namespace: &Namespace,
        threshold: f64,
        max_nodes: usize,
    ) -> Result<GraphData, DbError> {
        let conn = self.pool.writer().await;
        similarity_graph_sync(&conn, namespace, threshold, max_nodes)
    }

    /// Get top/bottom accessed memories.
    pub async fn access_ranking(
        &self,
        namespace: &Namespace,
        ascending: bool,
        limit: usize,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        access_ranking_sync(&conn, namespace, ascending, limit)
    }

    // --- Conversation methods ---

    pub async fn create_conversation(
        &self,
        namespace: &Namespace,
        title: &str,
        mode: &str,
    ) -> Result<Conversation, DbError> {
        let conn = self.pool.writer().await;
        let id = ulid::Ulid::new().to_string();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO conversations (id, namespace, title, mode, messages, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, '[]', ?5, ?5)",
            params![id, namespace.as_str(), title, mode, now],
        ).map_err(DbError::Sqlite)?;
        Ok(Conversation {
            id,
            namespace: namespace.as_str().to_string(),
            title: title.to_string(),
            mode: mode.to_string(),
            messages: "[]".to_string(),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    pub async fn list_conversations(
        &self,
        namespace: &Namespace,
    ) -> Result<Vec<ConversationSummary>, DbError> {
        let conn = self.pool.writer().await;
        let mut stmt = conn.prepare(
            "SELECT id, title, mode, created_at, updated_at
             FROM conversations
             WHERE namespace LIKE ?1
             ORDER BY updated_at DESC"
        ).map_err(DbError::Sqlite)?;
        let rows = stmt.query_map(
            params![namespace.like_pattern()],
            |row: &rusqlite::Row| Ok(ConversationSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                mode: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
            }),
        ).map_err(DbError::Sqlite)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(DbError::Sqlite)?);
        }
        Ok(results)
    }

    pub async fn get_conversation(&self, id: &str) -> Result<Option<Conversation>, DbError> {
        let conn = self.pool.writer().await;
        let mut stmt = conn.prepare(
            "SELECT id, namespace, title, mode, messages, created_at, updated_at
             FROM conversations WHERE id = ?1"
        ).map_err(DbError::Sqlite)?;
        let result = stmt.query_row(params![id], |row: &rusqlite::Row| {
            Ok(Conversation {
                id: row.get(0)?,
                namespace: row.get(1)?,
                title: row.get(2)?,
                mode: row.get(3)?,
                messages: row.get(4)?,
                created_at: row.get(5)?,
                updated_at: row.get(6)?,
            })
        });
        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DbError::Sqlite(e)),
        }
    }

    pub async fn update_conversation(
        &self,
        id: &str,
        title: Option<&str>,
        messages: Option<&str>,
    ) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        let now = Utc::now().to_rfc3339();
        let rows = match (title, messages) {
            (Some(t), Some(m)) => conn.execute(
                "UPDATE conversations SET title = ?1, messages = ?2, updated_at = ?3 WHERE id = ?4",
                params![t, m, now, id],
            ),
            (Some(t), None) => conn.execute(
                "UPDATE conversations SET title = ?1, updated_at = ?2 WHERE id = ?3",
                params![t, now, id],
            ),
            (None, Some(m)) => conn.execute(
                "UPDATE conversations SET messages = ?1, updated_at = ?2 WHERE id = ?3",
                params![m, now, id],
            ),
            (None, None) => conn.execute(
                "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
                params![now, id],
            ),
        }.map_err(DbError::Sqlite)?;
        Ok(rows > 0)
    }

    pub async fn delete_conversation(&self, id: &str) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        let rows = conn.execute(
            "DELETE FROM conversations WHERE id = ?1",
            params![id],
        ).map_err(DbError::Sqlite)?;
        Ok(rows > 0)
    }

    /// Set the importance score for a memory (1-10, default 5).
    pub async fn set_importance(&self, id: &str, importance: i32) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        conn.execute(
            "UPDATE memories SET importance = ?1 WHERE id = ?2",
            params![importance.clamp(1, 10), id],
        )
        .map_err(DbError::Sqlite)?;
        Ok(())
    }

    /// Update the JSON metadata column for a memory.
    pub async fn update_json_metadata(&self, id: &str, metadata: serde_json::Value) -> Result<bool, DbError> {
        let conn = self.pool.writer().await;
        let json_str = serde_json::to_string(&metadata).unwrap_or_default();
        let now = Utc::now().to_rfc3339();
        let rows = conn.execute(
            "UPDATE memories SET metadata = ?1, updated_at = ?2 WHERE id = ?3",
            params![json_str, now, id],
        ).map_err(DbError::Sqlite)?;
        if rows > 0 {
            snapshot_claim_revision_sync(
                &conn,
                id,
                "update_metadata_json",
                None,
                None,
                Some(serde_json::json!({"source": "update_json_metadata"})),
            )?;
        }
        Ok(rows > 0)
    }

    pub async fn patch_memory(
        &self,
        id: &str,
        patch: &MemoryPatch,
        new_embedding: Option<&[f32]>,
        actor: Option<&str>,
        reason: Option<&str>,
    ) -> Result<Option<MemoryMutationResult>, DbError> {
        let conn = self.pool.writer().await;
        patch_memory_sync(&conn, id, patch, new_embedding, actor, reason)
    }

    pub async fn rollback_memory_to_revision(
        &self,
        id: &str,
        revision_number: i64,
        actor: Option<&str>,
        reason: Option<&str>,
    ) -> Result<Option<MemoryMutationResult>, DbError> {
        let conn = self.pool.writer().await;
        rollback_memory_to_revision_sync(&conn, id, revision_number, actor, reason)
    }

    pub async fn merge_memories(
        &self,
        namespace: &Namespace,
        source_memory_ids: &[String],
        merged_memory: &Memory,
        merged_embedding: &[f32],
        actor: Option<&str>,
        reason: Option<&str>,
    ) -> Result<MemoryMergeResult, DbError> {
        let conn = self.pool.writer().await;
        merge_memories_sync(
            &conn,
            namespace,
            source_memory_ids,
            merged_memory,
            merged_embedding,
            actor,
            reason,
        )
    }

    pub async fn list_claim_revisions(
        &self,
        namespace: &Namespace,
        memory_id: &str,
        limit: usize,
    ) -> Result<Vec<ClaimRevision>, DbError> {
        let conn = self.pool.writer().await;
        list_claim_revisions_sync(&conn, namespace, memory_id, limit)
    }

    pub async fn upsert_procedure_revision(
        &self,
        namespace: &Namespace,
        procedure_name: &str,
        operation: &str,
        spec: &serde_json::Value,
        actor: Option<&str>,
        reason: Option<&str>,
    ) -> Result<ProcedureRevision, DbError> {
        let conn = self.pool.writer().await;
        upsert_procedure_revision_sync(
            &conn,
            namespace,
            procedure_name,
            operation,
            spec,
            actor,
            reason,
        )
    }

    pub async fn list_procedure_revisions(
        &self,
        namespace: &Namespace,
        procedure_name: &str,
        limit: usize,
    ) -> Result<Vec<ProcedureRevision>, DbError> {
        let conn = self.pool.writer().await;
        list_procedure_revisions_sync(&conn, namespace, procedure_name, limit)
    }

    pub async fn list_audit_events(
        &self,
        namespace: &Namespace,
        entity_type: Option<&str>,
        entity_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<AuditEvent>, DbError> {
        let conn = self.pool.writer().await;
        list_audit_events_sync(&conn, namespace, entity_type, entity_id, limit)
    }

    pub async fn upsert_identity_entity(
        &self,
        namespace: &Namespace,
        canonical_name: &str,
        language: Option<&str>,
        confidence: f64,
        metadata: Option<serde_json::Value>,
    ) -> Result<IdentityEntity, DbError> {
        let conn = self.pool.writer().await;
        upsert_identity_entity_sync(
            &conn,
            namespace,
            canonical_name,
            language,
            confidence,
            metadata,
        )
    }

    pub async fn add_identity_alias(
        &self,
        namespace: &Namespace,
        entity_id: &str,
        alias: &str,
        language: Option<&str>,
        confidence: f64,
    ) -> Result<(), DbError> {
        let conn = self.pool.writer().await;
        add_identity_alias_sync(&conn, namespace, entity_id, alias, language, confidence)
    }

    pub async fn resolve_identity(
        &self,
        namespace: &Namespace,
        query: &str,
        limit: usize,
    ) -> Result<IdentityResolution, DbError> {
        let conn = self.pool.writer().await;
        resolve_identity_sync(&conn, namespace, query, limit)
    }

    pub async fn create_plan_trace(
        &self,
        namespace: &Namespace,
        goal: &str,
        priority: i32,
        due_at: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<PlanTrace, DbError> {
        let conn = self.pool.writer().await;
        create_plan_trace_sync(&conn, namespace, goal, priority, due_at, metadata)
    }

    pub async fn list_plan_traces(
        &self,
        namespace: &Namespace,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<PlanTrace>, DbError> {
        let conn = self.pool.writer().await;
        list_plan_traces_sync(&conn, namespace, status, limit)
    }

    pub async fn add_plan_checkpoint(
        &self,
        namespace: &Namespace,
        plan_id: &str,
        checkpoint_key: &str,
        title: &str,
        order_index: i32,
        expected_by: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<PlanCheckpoint, DbError> {
        let conn = self.pool.writer().await;
        add_plan_checkpoint_sync(
            &conn,
            namespace,
            plan_id,
            checkpoint_key,
            title,
            order_index,
            expected_by,
            metadata,
        )
    }

    pub async fn list_plan_checkpoints(
        &self,
        namespace: &Namespace,
        plan_id: &str,
    ) -> Result<Vec<PlanCheckpoint>, DbError> {
        let conn = self.pool.writer().await;
        list_plan_checkpoints_sync(&conn, namespace, plan_id)
    }

    pub async fn update_plan_checkpoint_status(
        &self,
        namespace: &Namespace,
        checkpoint_id: &str,
        status: &str,
        evidence: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Option<PlanCheckpoint>, DbError> {
        let conn = self.pool.writer().await;
        update_plan_checkpoint_status_sync(
            &conn,
            namespace,
            checkpoint_id,
            status,
            evidence,
            metadata,
        )
    }

    pub async fn set_plan_outcome(
        &self,
        namespace: &Namespace,
        plan_id: &str,
        status: &str,
        outcome: &str,
        outcome_confidence: Option<f64>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Option<PlanTrace>, DbError> {
        let conn = self.pool.writer().await;
        set_plan_outcome_sync(
            &conn,
            namespace,
            plan_id,
            status,
            outcome,
            outcome_confidence,
            metadata,
        )
    }

    pub async fn add_plan_recovery_branch(
        &self,
        namespace: &Namespace,
        plan_id: &str,
        source_checkpoint_id: Option<&str>,
        branch_label: &str,
        trigger_reason: &str,
        branch_plan: Option<serde_json::Value>,
        metadata: Option<serde_json::Value>,
    ) -> Result<PlanRecoveryBranch, DbError> {
        let conn = self.pool.writer().await;
        add_plan_recovery_branch_sync(
            &conn,
            namespace,
            plan_id,
            source_checkpoint_id,
            branch_label,
            trigger_reason,
            branch_plan,
            metadata,
        )
    }

    pub async fn list_plan_recovery_branches(
        &self,
        namespace: &Namespace,
        plan_id: &str,
    ) -> Result<Vec<PlanRecoveryBranch>, DbError> {
        let conn = self.pool.writer().await;
        list_plan_recovery_branches_sync(&conn, namespace, plan_id)
    }

    pub async fn resolve_plan_recovery_branch(
        &self,
        namespace: &Namespace,
        branch_id: &str,
        status: &str,
        resolution_notes: Option<&str>,
    ) -> Result<Option<PlanRecoveryBranch>, DbError> {
        let conn = self.pool.writer().await;
        resolve_plan_recovery_branch_sync(&conn, namespace, branch_id, status, resolution_notes)
    }

    pub async fn bind_procedure_to_plan(
        &self,
        namespace: &Namespace,
        plan_id: &str,
        procedure_name: &str,
        binding_role: &str,
        confidence: f64,
        metadata: Option<serde_json::Value>,
    ) -> Result<PlanProcedureBinding, DbError> {
        let conn = self.pool.writer().await;
        bind_procedure_to_plan_sync(
            &conn,
            namespace,
            plan_id,
            procedure_name,
            binding_role,
            confidence,
            metadata,
        )
    }

    pub async fn list_plan_procedure_bindings(
        &self,
        namespace: &Namespace,
        plan_id: &str,
    ) -> Result<Vec<PlanProcedureBinding>, DbError> {
        let conn = self.pool.writer().await;
        list_plan_procedure_bindings_sync(&conn, namespace, plan_id)
    }

    /// Find raw memories that haven't been reflected yet.
    pub async fn find_unreflected_raw(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let mut stmt = conn.prepare(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
             FROM memories
             WHERE namespace LIKE ?1
               AND status = 'active'
               AND (json_extract(metadata, '$.reflected') IS NULL OR json_extract(metadata, '$.reflected') = 0)
               AND (json_extract(metadata, '$.tier') IS NULL OR json_extract(metadata, '$.tier') = 1)
               AND memory_type != 'reflected'
             ORDER BY created_at ASC
             LIMIT ?2",
        ).map_err(DbError::Sqlite)?;

        let rows = stmt.query_map(params![ns_pattern, limit as i64], row_to_memory)
            .map_err(DbError::Sqlite)?;
        let memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();
        Ok(memories)
    }

    /// Find all active memories at a specific tier level in a namespace.
    pub async fn find_by_tier(
        &self,
        namespace: &Namespace,
        tier: i32,
        limit: usize,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let mut stmt = conn.prepare(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
             FROM memories
             WHERE namespace LIKE ?1
               AND status = 'active'
               AND CAST(json_extract(metadata, '$.tier') AS INTEGER) = ?2
             ORDER BY created_at ASC
             LIMIT ?3",
        ).map_err(DbError::Sqlite)?;

        let rows = stmt.query_map(params![ns_pattern, tier, limit as i64], row_to_memory)
            .map_err(DbError::Sqlite)?;
        let memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();
        Ok(memories)
    }

    /// Find memories that have a structured event_date within a date range.
    /// `start` and `end` are ISO 8601 dates ("YYYY-MM-DD").
    pub async fn find_by_date_range(
        &self,
        namespace: &Namespace,
        start: &str,
        end: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let mut stmt = conn.prepare(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
             FROM memories
             WHERE namespace LIKE ?1
               AND status = 'active'
               AND event_date IS NOT NULL
               AND event_date >= ?2
               AND event_date <= ?3
             ORDER BY event_date ASC
             LIMIT ?4",
        ).map_err(DbError::Sqlite)?;

        let rows = stmt.query_map(params![ns_pattern, start, end, limit as i64], row_to_memory)
            .map_err(DbError::Sqlite)?;
        let memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();
        Ok(memories)
    }

    /// Fetch raw embeddings for visualization (PCA input).
    pub async fn get_raw_embeddings(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<RawEmbeddingData>, DbError> {
        let conn = self.pool.writer().await;
        get_raw_embeddings_sync(&conn, namespace, limit)
    }

    /// Update accessed_at and increment access_count for retrieved memories.
    pub async fn touch(&self, ids: &[String]) -> Result<(), DbError> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn: tokio::sync::MutexGuard<'_, rusqlite::Connection> = self.pool.writer().await;
        let now = Utc::now().to_rfc3339();
        for id in ids {
            conn.execute(
                "UPDATE memories SET accessed_at = ?1, access_count = access_count + 1 WHERE id = ?2",
                params![now, id],
            ).map_err(DbError::Sqlite)?;
        }
        Ok(())
    }

    /// Fetch all active memories in the same episode.
    pub async fn find_by_episode(
        &self,
        namespace: &Namespace,
        episode_id: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let mut stmt = conn.prepare(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at,
                    accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
             FROM memories
             WHERE namespace LIKE ?1
               AND episode_id = ?2
               AND status = 'active'
             ORDER BY created_at ASC
             LIMIT ?3",
        ).map_err(DbError::Sqlite)?;
        let rows = stmt
            .query_map(params![ns_pattern, episode_id, limit as i64], row_to_memory)
            .map_err(DbError::Sqlite)?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Backfill episode_id from metadata.session for existing memories.
    pub async fn backfill_episode_ids(&self) -> Result<u64, DbError> {
        let conn = self.pool.writer().await;
        let updated = conn.execute(
            "UPDATE memories
             SET episode_id = json_extract(metadata, '$.session')
             WHERE episode_id IS NULL
               AND json_extract(metadata, '$.session') IS NOT NULL",
            [],
        ).map_err(DbError::Sqlite)?;
        tracing::info!("Backfilled episode_id for {updated} memories");
        Ok(updated as u64)
    }

    /// Export all memories for backup. No pagination — returns everything.
    /// Filters by namespace (use wildcard namespace for all).
    pub async fn export_all(
        &self,
        namespace: &Namespace,
    ) -> Result<Vec<Memory>, DbError> {
        let conn = self.pool.writer().await;
        let ns_pattern = namespace.like_pattern();
        let mut stmt = conn.prepare(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at, \
             accessed_at, access_count, hash, tags, memory_type, importance, episode_id, \
             event_date, category, confidence, source \
             FROM memories WHERE namespace LIKE ?1 \
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![ns_pattern], row_to_memory)?;
        let mut memories = Vec::new();
        for row in rows {
            memories.push(row?);
        }
        Ok(memories)
    }

    /// Return the database file size in bytes. Returns 0 for in-memory DBs.
    pub fn db_size_bytes(&self) -> u64 {
        if self.pool.db_path() == ":memory:" {
            return 0;
        }
        std::fs::metadata(self.pool.db_path())
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Return the database file path.
    pub fn db_path(&self) -> &str {
        self.pool.db_path()
    }

    /// Acquire the writer connection (for WAL checkpoint).
    pub async fn writer_conn(&self) -> tokio::sync::MutexGuard<'_, rusqlite::Connection> {
        self.pool.writer().await
    }
}

// --- Synchronous implementations (called within mutex guard) ---

fn insert_memory_sync(
    conn: &Connection,
    memory: &Memory,
    embedding: &[f32],
) -> Result<(), DbError> {
    let metadata_json = memory
        .metadata
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let embedding_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

    let tags_json = serde_json::to_string(&memory.tags).unwrap_or_else(|_| "[]".to_string());

    // Resolve event_date: explicit field > metadata.event_date
    let event_date = memory.event_date.clone().or_else(|| {
        memory.metadata.as_ref()
            .and_then(|m| m.get("event_date"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    tx.execute(
        "INSERT INTO memories (id, namespace, content, metadata, embedding, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, episode_id, event_date, category, confidence, source)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        params![
            memory.id,
            memory.namespace,
            memory.content,
            metadata_json,
            embedding_blob,
            memory.status.as_str(),
            memory.created_at.to_rfc3339(),
            memory.updated_at.to_rfc3339(),
            memory.accessed_at.to_rfc3339(),
            memory.access_count,
            memory.hash,
            tags_json,
            memory.memory_type,
            memory.episode_id,
            event_date,
            memory.category.as_str(),
            memory.confidence,
            memory.source,
        ],
    )?;

    // Sync to vec_memories
    vector::insert_embedding(&tx, &memory.id, embedding, &memory.namespace)?;

    // Sync to fts_memories
    fts::insert_fts(&tx, &memory.id, &memory.namespace, &memory.content)?;

    snapshot_claim_revision_sync(
        &tx,
        &memory.id,
        "create",
        None,
        None,
        Some(serde_json::json!({"source": "insert_memory"})),
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(())
}

fn insert_many_memories_sync(
    conn: &Connection,
    entries: &[(Memory, Vec<f32>)],
) -> Result<(), DbError> {
    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    for (memory, embedding) in entries {
        let metadata_json = memory
            .metadata
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        let embedding_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tags_json = serde_json::to_string(&memory.tags).unwrap_or_else(|_| "[]".to_string());

        let event_date = memory.event_date.clone().or_else(|| {
            memory.metadata.as_ref()
                .and_then(|m| m.get("event_date"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

        tx.execute(
            "INSERT INTO memories (id, namespace, content, metadata, embedding, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, episode_id, event_date, category, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
            params![
                memory.id,
                memory.namespace,
                memory.content,
                metadata_json,
                embedding_blob,
                memory.status.as_str(),
                memory.created_at.to_rfc3339(),
                memory.updated_at.to_rfc3339(),
                memory.accessed_at.to_rfc3339(),
                memory.access_count,
                memory.hash,
                tags_json,
                memory.memory_type,
                memory.episode_id,
                event_date,
                memory.category.as_str(),
                memory.confidence,
                memory.source,
            ],
        )?;

        vector::insert_embedding(&tx, &memory.id, embedding, &memory.namespace)?;
        fts::insert_fts(&tx, &memory.id, &memory.namespace, &memory.content)?;

        snapshot_claim_revision_sync(
            &tx,
            &memory.id,
            "create",
            None,
            None,
            Some(serde_json::json!({"source": "insert_many_memories"})),
        )?;
    }

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(())
}

fn find_by_hash_sync(
    conn: &Connection,
    namespace: &Namespace,
    hash: &str,
) -> Result<Option<Memory>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
         FROM memories
         WHERE namespace LIKE ?1 AND hash = ?2 AND status = 'active'
         LIMIT 1",
    )?;

    let result = stmt
        .query_row(params![namespace.like_pattern(), hash], row_to_memory)
        .optional()?;

    Ok(result)
}

fn get_memory_sync(conn: &Connection, id: &str) -> Result<Option<Memory>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
         FROM memories WHERE id = ?1 AND status = 'active'",
    )?;

    let result = stmt.query_row(params![id], row_to_memory).optional()?;
    Ok(result)
}

fn update_content_sync(
    conn: &Connection,
    id: &str,
    new_content: &str,
    new_embedding: &[f32],
) -> Result<bool, DbError> {
    // Get old memory for FTS delete
    let old = match get_memory_sync(conn, id)? {
        Some(m) => m,
        None => return Ok(false),
    };

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    let now = Utc::now().to_rfc3339();
    let new_hash = content_hash(new_content);
    let embedding_blob: Vec<u8> = new_embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

    tx.execute(
        "UPDATE memories SET content = ?1, embedding = ?2, hash = ?3, updated_at = ?4 WHERE id = ?5",
        params![new_content, embedding_blob, new_hash, now, id],
    )?;

    // Sync vec_memories
    vector::update_embedding(&tx, id, new_embedding, &old.namespace)?;

    // Sync fts_memories: delete old, insert new
    fts::delete_fts(&tx, &old.id)?;
    fts::insert_fts(&tx, id, &old.namespace, new_content)?;

    snapshot_claim_revision_sync(
        &tx,
        id,
        "update_content",
        None,
        None,
        Some(serde_json::json!({"source": "update_content"})),
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(true)
}

fn update_metadata_sync(
    conn: &Connection,
    id: &str,
    memory_type: Option<&str>,
    importance: Option<i32>,
    tags: Option<&[String]>,
) -> Result<bool, DbError> {
    if get_memory_sync(conn, id)?.is_none() {
        return Ok(false);
    }

    let now = Utc::now().to_rfc3339();

    if let Some(mt) = memory_type {
        conn.execute(
            "UPDATE memories SET memory_type = ?1, updated_at = ?2 WHERE id = ?3",
            params![mt, now, id],
        ).map_err(DbError::Sqlite)?;
    }

    if let Some(imp) = importance {
        conn.execute(
            "UPDATE memories SET importance = ?1, updated_at = ?2 WHERE id = ?3",
            params![imp.clamp(1, 10), now, id],
        ).map_err(DbError::Sqlite)?;
    }

    if let Some(t) = tags {
        let tags_json = serde_json::to_string(t).unwrap_or_else(|_| "[]".to_string());
        conn.execute(
            "UPDATE memories SET tags = ?1, updated_at = ?2 WHERE id = ?3",
            params![tags_json, now, id],
        ).map_err(DbError::Sqlite)?;
    }

    snapshot_claim_revision_sync(
        conn,
        id,
        "update_metadata",
        None,
        None,
        Some(serde_json::json!({"source": "update_metadata"})),
    )?;

    Ok(true)
}

fn soft_delete_sync(conn: &Connection, id: &str) -> Result<bool, DbError> {
    // Check existence
    if get_memory_sync(conn, id)?.is_none() {
        return Ok(false);
    }

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    let now = Utc::now().to_rfc3339();
    tx.execute(
        "UPDATE memories SET status = 'deleted', updated_at = ?1 WHERE id = ?2",
        params![now, id],
    )?;

    // Remove from search indexes
    vector::delete_embedding(&tx, id)?;
    fts::delete_fts(&tx, id)?;

    snapshot_claim_revision_sync(
        &tx,
        id,
        "soft_delete",
        None,
        None,
        Some(serde_json::json!({"source": "soft_delete"})),
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(true)
}

fn hard_delete_sync(conn: &Connection, id: &str) -> Result<bool, DbError> {
    if get_memory_sync(conn, id)?.is_none() {
        return Ok(false);
    }

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    snapshot_claim_revision_sync(
        &tx,
        id,
        "hard_delete",
        None,
        None,
        Some(serde_json::json!({"source": "hard_delete"})),
    )?;

    vector::delete_embedding(&tx, id)?;
    fts::delete_fts(&tx, id)?;
    tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(true)
}

fn mark_superseded_sync(conn: &Connection, id: &str, superseded_by: &str) -> Result<bool, DbError> {
    if get_memory_sync(conn, id)?.is_none() {
        return Ok(false);
    }

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    let now = Utc::now().to_rfc3339();
    tx.execute(
        "UPDATE memories SET status = 'superseded', updated_at = ?1, superseded_by = ?3 WHERE id = ?2",
        params![now, id, superseded_by],
    )?;

    vector::delete_embedding(&tx, id)?;
    fts::delete_fts(&tx, id)?;

    snapshot_claim_revision_sync(
        &tx,
        id,
        "supersede",
        None,
        Some("replaced by newer memory"),
        Some(serde_json::json!({ "superseded_by": superseded_by })),
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(true)
}

fn search_sync(
    conn: &Connection,
    query_embedding: &[f32],
    query_text: &str,
    namespace: &Namespace,
    mode: &SearchMode,
    limit: usize,
    scorer_config: &ScorerConfig,
) -> Result<Vec<ScoredResult>, DbError> {
    let candidate_limit = limit * 5;
    let ns_pattern = namespace.like_pattern();
    let now = Utc::now();

    let mut vector_scored: Vec<(String, f64)> = vec![];
    let mut keyword_scored: Vec<(String, f64)> = vec![];

    // Vector search — filter out low-similarity results.
    // Configurable via search.min_vector_similarity (default 0.35).
    let min_sim = scorer_config.min_vector_similarity;

    // Determine if we can use pre-filtered vector search.
    // sqlite-vec metadata columns support exact = but not LIKE.
    // Use pre-filtered search for exact namespace matches (no '/' hierarchy).
    // Fall back to global search + post-filter for hierarchical namespaces.
    let ns_str = namespace.as_str();
    let is_exact_ns = !ns_str.contains('/');

    match mode {
        SearchMode::Hybrid | SearchMode::Vector | SearchMode::AskRetrieval => {
            let raw = if is_exact_ns {
                // Pre-filtered: sqlite-vec filters by namespace during ANN search.
                // Only vectors in this namespace are considered — no wasted comparisons.
                vector::search_vectors_filtered(conn, query_embedding, ns_str, candidate_limit)?
            } else {
                // Hierarchical namespace — global search, post-filter below.
                vector::search_vectors(conn, query_embedding, candidate_limit)?
            };

            // Score spread check on RAW results (before any post-filter).
            let mut spread_ok = true;
            if raw.len() >= 3 && scorer_config.min_score_spread > 0.0 {
                let best_sim = 1.0 - (raw.first().unwrap().1 / 2.0);
                let worst_sim = 1.0 - (raw.last().unwrap().1 / 2.0);
                let spread = best_sim - worst_sim;
                if spread < scorer_config.min_score_spread {
                    tracing::debug!(
                        "Vector score spread too narrow ({:.4}), likely noise query — skipping vector",
                        spread
                    );
                    spread_ok = false;
                }
            }

            if spread_ok {
                if is_exact_ns {
                    // Pre-filtered results: only need similarity threshold + active status check.
                    // Namespace already matched by sqlite-vec.
                    for (id, dist) in &raw {
                        let similarity = 1.0 - (dist / 2.0);
                        if similarity < min_sim {
                            continue;
                        }
                        let mut stmt = conn.prepare_cached(
                            "SELECT 1 FROM memories WHERE id = ?1 AND status = 'active'",
                        )?;
                        if stmt.exists(params![id])? {
                            vector_scored.push((id.clone(), similarity));
                        }
                    }
                } else {
                    // Global results: post-filter by namespace (LIKE) and active status.
                    for (id, dist) in &raw {
                        let similarity = 1.0 - (dist / 2.0);
                        if similarity < min_sim {
                            continue;
                        }
                        let mut stmt = conn.prepare_cached(
                            "SELECT 1 FROM memories WHERE id = ?1 AND namespace LIKE ?2 AND status = 'active'",
                        )?;
                        if stmt.exists(params![id, ns_pattern])? {
                            vector_scored.push((id.clone(), similarity));
                        }
                    }
                }
            }

            // Short/single-token queries produce low-signal embeddings — require
            // a higher similarity floor to avoid returning noise.
            if query_text.trim().len() <= 2 && !vector_scored.is_empty() {
                let short_query_min = (min_sim + 0.10).min(0.70);
                let before = vector_scored.len();
                vector_scored.retain(|(_, s)| *s >= short_query_min);
                if vector_scored.len() < before {
                    tracing::debug!(
                        "Short query filter: removed {} low-confidence results (min_sim={:.2})",
                        before - vector_scored.len(),
                        short_query_min
                    );
                }
            }
        }
        SearchMode::Keyword => {}
    }

    // Keyword search
    match mode {
        SearchMode::Hybrid | SearchMode::Keyword | SearchMode::AskRetrieval => {
            let raw = fts::search_fts(conn, query_text, &ns_pattern, candidate_limit)?;
            keyword_scored = raw;
        }
        SearchMode::Vector => {}
    }

    // Collect timestamps, access counts, importances, tiers, and event_dates for scoring
    let all_ids: Vec<&String> = vector_scored.iter().map(|(id, _)| id)
        .chain(keyword_scored.iter().map(|(id, _)| id))
        .collect();
    let mut timestamps: HashMap<String, DateTime<Utc>> = HashMap::new();
    let mut access_counts: HashMap<String, u64> = HashMap::new();
    let mut importances: HashMap<String, i32> = HashMap::new();
    let mut tiers: HashMap<String, i32> = HashMap::new();
    let mut categories: HashMap<String, String> = HashMap::new();
    let mut confidences: HashMap<String, f64> = HashMap::new();
    for id in &all_ids {
        if !timestamps.contains_key(*id) {
            if let Some((ts, ac, imp, tier, event_date, category, confidence)) = get_scoring_metadata(conn, id)? {
                // Prefer event_date over updated_at for temporal scoring —
                // updated_at is ingestion time (useless when all memories ingested together),
                // event_date is when the event actually happened.
                timestamps.insert((*id).clone(), event_date.unwrap_or(ts));
                access_counts.insert((*id).clone(), ac);
                importances.insert((*id).clone(), imp);
                tiers.insert((*id).clone(), tier);
                categories.insert((*id).clone(), category);
                confidences.insert((*id).clone(), confidence);
            }
        }
    }

    // Filter out memories above max_tier
    let max_tier = scorer_config.max_tier;
    if max_tier < 4 {
        let over_tier: HashSet<&String> = tiers.iter()
            .filter(|(_, &t)| t > max_tier)
            .map(|(id, _)| id)
            .collect();
        if !over_tier.is_empty() {
            vector_scored.retain(|(id, _)| !over_tier.contains(id));
            keyword_scored.retain(|(id, _)| !over_tier.contains(id));
        }
    }

    // Filter by event_date range if specified
    if scorer_config.date_start.is_some() || scorer_config.date_end.is_some() {
        let date_start = scorer_config.date_start.as_deref();
        let date_end = scorer_config.date_end.as_deref();
        let all_candidate_ids: Vec<String> = vector_scored.iter().map(|(id, _)| id.clone())
            .chain(keyword_scored.iter().map(|(id, _)| id.clone()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let mut excluded: HashSet<String> = HashSet::new();
        for id in &all_candidate_ids {
            let event_date: Option<String> = conn.query_row(
                "SELECT event_date FROM memories WHERE id = ?1",
                params![id],
                |row| row.get(0),
            ).unwrap_or(None);
            if let Some(ed) = &event_date {
                if let Some(start) = date_start {
                    if ed.as_str() < start { excluded.insert(id.clone()); continue; }
                }
                if let Some(end) = date_end {
                    if ed.as_str() > end { excluded.insert(id.clone()); }
                }
            }
            // Memories without event_date pass through (not excluded)
        }
        if !excluded.is_empty() {
            vector_scored.retain(|(id, _)| !excluded.contains(id));
            keyword_scored.retain(|(id, _)| !excluded.contains(id));
        }
    }

    // Fuse with RRF + temporal decay
    let scorer = HybridScorer::new(scorer_config.clone());
    let mut results = scorer.fuse(&vector_scored, &[], &keyword_scored, &timestamps, now);

    // Apply access frequency, importance, and tier boosts
    scorer.apply_boosts(&mut results, &access_counts, &importances, &tiers);

    // Confidence boost: high-confidence memories get a small additive bonus,
    // low-confidence memories get a penalty. Centered at 0.7 so typical user
    // memories (1.0) get a slight boost and inferred facts (0.5) get a penalty.
    for r in results.iter_mut() {
        let conf = confidences.get(&r.memory_id).copied().unwrap_or(1.0);
        let confidence_bonus = 0.03 * (conf - 0.7);
        r.score = (r.score + confidence_bonus).clamp(0.0, 1.0);
    }

    // Per-category temporal decay correction.
    // The initial fuse used a single global lambda. Here we adjust scores for
    // memories whose category implies a different decay rate.
    // Identity memories get a boost (they should barely decay), task memories get penalized.
    if scorer_config.temporal_weight > 0.0 && !scorer_config.category_lambdas.is_empty() {
        let global_lambda = scorer_config.lambda;
        for r in results.iter_mut() {
            let cat_name = categories.get(&r.memory_id).cloned().unwrap_or_else(|| "general".to_string());
            let cat_lambda = scorer_config.category_lambdas.get(&cat_name).copied().unwrap_or(global_lambda);
            if (cat_lambda - global_lambda).abs() > 1e-9 {
                if let Some(ts) = timestamps.get(&r.memory_id) {
                    let age_hours = (now - *ts).num_seconds() as f64 / 3600.0;
                    let age_hours = age_hours.max(0.0);
                    let global_decay = anima_core::temporal::exponential_decay(age_hours, global_lambda);
                    let cat_decay = anima_core::temporal::exponential_decay(age_hours, cat_lambda);
                    // Apply the difference between category-specific and global decay
                    if global_decay > 1e-9 {
                        let tw = scorer_config.temporal_weight;
                        let temporal_delta = tw * (cat_decay - global_decay);
                        r.score = (r.score + temporal_delta).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    // Graph-aware retrieval: boost memories connected through causal/entity edges.
    // Full boost for causal queries ("why/because"), scaled down for other queries.
    if !results.is_empty() {
        let seeds: Vec<String> = results
            .iter()
            .take(8)
            .map(|r| r.memory_id.clone())
            .collect();
        let edge_boosts = causal_boosts_for_seed_ids_sync(conn, namespace, &seeds)?;
        if !edge_boosts.is_empty() {
            let scale = if is_causal_query(query_text) { 1.0 } else { 0.3 };
            for r in &mut results {
                if let Some(boost) = edge_boosts.get(&r.memory_id) {
                    r.score = (r.score + boost * scale).clamp(0.0, 1.0);
                }
            }
        }
    }

    // NOTE: Retrieval calibration disabled — the linear model (slope=0.2, intercept=0.5)
    // compresses all scores into [0.50, 0.70], destroying differentiation.
    // TODO: retrain with proper relevance labels before re-enabling.
    // let retrieval_kind = PredictionKind::RetrievalRelevance;
    // for r in results.iter_mut() {
    //     r.score = calibrate_confidence_sync(conn, namespace, retrieval_kind, r.score)?;
    // }

    // Content-length discount: short memories (reflected facts, deduced summaries)
    // get inflated cosine similarity because their embeddings are semantically broad.
    // Discount memories with fewer than `min_tokens` words so they don't crowd out
    // longer, detail-rich raw memories.
    let min_tokens: usize = 20;
    for r in results.iter_mut() {
        let content_len = get_content_length(conn, &r.memory_id).unwrap_or(min_tokens);
        if content_len < min_tokens {
            // Linear ramp: 1 word → 0.5x, 10 words → 0.75x, 20 words → 1.0x
            let factor = 0.5 + 0.5 * (content_len as f64 / min_tokens as f64);
            r.score *= factor;
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(limit);

    Ok(results)
}

fn is_causal_query(query: &str) -> bool {
    let q = query.to_ascii_lowercase();
    [
        "why ",
        "why?",
        "how come",
        "because",
        "cause",
        "caused",
        "causal",
        "reason",
        "led to",
        "resulted in",
        "effect",
        "impact",
        "due to",
    ]
    .iter()
    .any(|needle| q.contains(needle))
}

fn causal_boosts_for_seed_ids_sync(
    conn: &Connection,
    namespace: &Namespace,
    seed_ids: &[String],
) -> Result<HashMap<String, f64>, DbError> {
    if seed_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut boosts: HashMap<String, f64> = HashMap::new();
    let mut stmt = conn.prepare_cached(
        "SELECT source_memory_id, target_memory_id, relation_type, confidence
         FROM causal_edges
         WHERE namespace = ?1
           AND (source_memory_id = ?2 OR target_memory_id = ?2)",
    )?;
    for seed in seed_ids {
        let rows = stmt.query_map(params![namespace.as_str(), seed], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        for row in rows {
            let (source, target, relation_type, confidence) = row?;
            let other = if source == *seed { target } else { source };
            if other == *seed {
                continue;
            }
            let type_weight = match relation_type.as_str() {
                "causal" => 1.0,
                "correlational" => 0.55,
                _ => 0.4,
            };
            let boost = (confidence.clamp(0.0, 1.0) * type_weight * 0.12).clamp(0.0, 0.2);
            let entry = boosts.entry(other).or_insert(0.0);
            if boost > *entry {
                *entry = boost;
            }
        }
    }
    Ok(boosts)
}

fn find_similar_sync(
    conn: &Connection,
    embedding: &[f32],
    namespace: &Namespace,
    limit: usize,
    threshold: f64,
) -> Result<Vec<(Memory, f64)>, DbError> {
    let raw = vector::search_vectors(conn, embedding, limit * 3)?;
    let ns_pattern = namespace.like_pattern();
    let mut results = Vec::new();

    for (id, distance) in raw {
        // Convert L2 distance to cosine similarity (vectors are pre-normalized)
        let similarity = 1.0 - (distance / 2.0);
        if similarity < threshold {
            continue;
        }

        let mut stmt = conn.prepare_cached(
            "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
             FROM memories
             WHERE id = ?1 AND namespace LIKE ?2 AND status = 'active'",
        )?;

        if let Some(memory) = stmt.query_row(params![id, ns_pattern], row_to_memory).optional()? {
            results.push((memory, similarity));
        }

        if results.len() >= limit {
            break;
        }
    }

    Ok(results)
}

fn find_neighbors_sync(
    conn: &Connection,
    memory_id: &str,
    limit: usize,
    min_similarity: f64,
) -> Result<Vec<(Memory, f64)>, DbError> {
    // Load embedding for this memory
    let blob: Option<Vec<u8>> = conn
        .prepare_cached("SELECT embedding FROM memories WHERE id = ?1")?
        .query_row(params![memory_id], |row| row.get(0))
        .optional()?;

    let blob = match blob {
        Some(b) => b,
        None => return Ok(vec![]),
    };

    let embedding: Vec<f32> = blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Find nearest neighbors
    let raw = vector::search_vectors(&conn, &embedding, limit + 1)?;
    let mut results = Vec::new();

    for (id, distance) in raw {
        if id == memory_id {
            continue;
        }
        let similarity = 1.0 - (distance / 2.0);
        if similarity < min_similarity {
            continue;
        }
        if let Some(memory) = get_memory_sync(&conn, &id)? {
            results.push((memory, similarity));
        }
        if results.len() >= limit {
            break;
        }
    }

    Ok(results)
}

fn list_sync(
    conn: &Connection,
    namespace: &Namespace,
    status: Option<&str>,
    memory_type: Option<&str>,
    category: Option<&str>,
    offset: usize,
    limit: usize,
) -> Result<(Vec<Memory>, u64), DbError> {
    let ns_pattern = namespace.like_pattern();

    // Build dynamic WHERE clause
    let mut conditions = vec!["namespace LIKE ?1".to_string()];
    let mut count_params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(ns_pattern.clone())];

    if let Some(st) = status {
        conditions.push(format!("status = ?{}", count_params.len() + 1));
        count_params.push(Box::new(st.to_string()));
    }
    if let Some(mt) = memory_type {
        conditions.push(format!("memory_type = ?{}", count_params.len() + 1));
        count_params.push(Box::new(mt.to_string()));
    }
    if let Some(cat) = category {
        conditions.push(format!("category = ?{}", count_params.len() + 1));
        count_params.push(Box::new(cat.to_string()));
    }

    let where_clause = conditions.join(" AND ");
    let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {where_clause}");
    let data_sql = format!(
        "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
         FROM memories WHERE {where_clause}
         ORDER BY created_at DESC LIMIT ?{} OFFSET ?{}",
        count_params.len() + 1,
        count_params.len() + 2,
    );

    let count_refs: Vec<&dyn rusqlite::types::ToSql> = count_params.iter().map(|b| b.as_ref()).collect();
    let total: u64 = conn
        .prepare(&count_sql)?
        .query_row(count_refs.as_slice(), |row| row.get(0))?;

    let mut data_params = count_params;
    data_params.push(Box::new(limit as i64));
    data_params.push(Box::new(offset as i64));
    let data_refs: Vec<&dyn rusqlite::types::ToSql> = data_params.iter().map(|b| b.as_ref()).collect();

    let mut stmt = conn.prepare(&data_sql)?;
    let rows = stmt.query_map(data_refs.as_slice(), row_to_memory)?;
    let memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();

    Ok((memories, total))
}

fn stats_sync(conn: &Connection, namespace: &Namespace) -> Result<NamespaceStats, DbError> {
    let ns_pattern = namespace.like_pattern();
    let mut stmt = conn.prepare(
        "SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
            SUM(CASE WHEN status = 'superseded' THEN 1 ELSE 0 END) as superseded,
            SUM(CASE WHEN status = 'deleted' THEN 1 ELSE 0 END) as deleted,
            COALESCE(AVG(access_count), 0.0) as avg_access,
            COALESCE(MAX(access_count), 0) as max_access,
            MIN(created_at) as oldest,
            MAX(created_at) as newest
         FROM memories WHERE namespace LIKE ?1",
    )?;

    let stats = stmt.query_row(params![ns_pattern], |row| {
        Ok(NamespaceStats {
            total: row.get::<_, u64>(0)?,
            active: row.get::<_, u64>(1)?,
            superseded: row.get::<_, u64>(2)?,
            deleted: row.get::<_, u64>(3)?,
            avg_access_count: row.get::<_, f64>(4)?,
            max_access_count: row.get::<_, u64>(5)?,
            oldest_memory: row.get::<_, Option<String>>(6)?,
            newest_memory: row.get::<_, Option<String>>(7)?,
        })
    })?;

    Ok(stats)
}

fn list_namespaces_sync(conn: &Connection) -> Result<Vec<NamespaceInfo>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT namespace, COUNT(*) as total,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active
         FROM memories GROUP BY namespace",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(NamespaceInfo {
            namespace: row.get(0)?,
            total_count: row.get(1)?,
            active_count: row.get(2)?,
        })
    })?;

    let infos: Vec<NamespaceInfo> = rows.filter_map(|r| r.ok()).collect();
    Ok(infos)
}

fn rename_namespace_sync(conn: &Connection, old_ns: &Namespace, new_ns: &Namespace) -> Result<u64, DbError> {
    let old = old_ns.as_str();
    let new = new_ns.as_str();
    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    let count: u64 = tx
        .prepare_cached("SELECT COUNT(*) FROM memories WHERE namespace = ?1")?
        .query_row(params![old], |r| r.get(0))?;

    // Update vec_memories namespace metadata column
    tx.execute(
        "UPDATE vec_memories SET namespace = ?1 WHERE namespace = ?2",
        params![new, old],
    )?;

    // Update all namespace-scoped tables
    let tables = [
        "memories",
        "conversations",
        "working_memories",
        "claim_revisions",
        "correction_events",
        "contradiction_ledger",
        "causal_edges",
        "state_transitions",
        "calibration_observations",
        "calibration_models",
        "calibration_bins",
        "procedure_revisions",
        "memory_audit_log",
        "identity_entities",
        "identity_aliases",
        "plan_traces",
        "plan_checkpoints",
        "plan_recovery_branches",
        "plan_procedure_bindings",
    ];
    for table in tables {
        tx.execute(
            &format!("UPDATE {table} SET namespace = ?1 WHERE namespace = ?2"),
            params![new, old],
        )?;
    }

    // Update FTS namespace
    tx.execute(
        "UPDATE fts_memories SET namespace = ?1 WHERE namespace = ?2",
        params![new, old],
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(count)
}

fn delete_namespace_sync(conn: &Connection, namespace: &Namespace) -> Result<u64, DbError> {
    let ns = namespace.as_str();
    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    // Count memories to report
    let count: u64 = tx
        .prepare_cached("SELECT COUNT(*) FROM memories WHERE namespace = ?1")?
        .query_row(params![ns], |r| r.get(0))?;

    // Delete from vec and fts indexes for all memories in the namespace
    let ids: Vec<String> = {
        let mut stmt = tx.prepare("SELECT id FROM memories WHERE namespace = ?1")?;
        let rows = stmt.query_map(params![ns], |r| r.get(0))?;
        rows.filter_map(|r| r.ok()).collect()
    };
    for id in &ids {
        let _ = crate::vector::delete_embedding(&tx, id);
        let _ = crate::fts::delete_fts(&tx, id);
    }

    // Delete from all namespace-scoped tables
    let tables = [
        "memories",
        "conversations",
        "working_memories",
        "claim_revisions",
        "correction_events",
        "contradiction_ledger",
        "causal_edges",
        "state_transitions",
        "calibration_observations",
        "calibration_models",
        "calibration_bins",
        "procedure_revisions",
        "memory_audit_log",
        "identity_entities",
        "identity_aliases",
        "plan_traces",
        "plan_checkpoints",
        "plan_recovery_branches",
        "plan_procedure_bindings",
    ];
    for table in tables {
        tx.execute(
            &format!("DELETE FROM {table} WHERE namespace = ?1"),
            params![ns],
        )?;
    }

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(count)
}

fn graph_node_from_memory(memory: &Memory) -> GraphNode {
    let truncated = if memory.content.len() > 100 {
        format!("{}...", &memory.content[..100])
    } else {
        memory.content.clone()
    };
    GraphNode {
        id: memory.id.clone(),
        content: truncated,
        namespace: memory.namespace.clone(),
        status: memory.status.as_str().to_string(),
        memory_type: memory.memory_type.clone(),
        importance: memory.importance,
        created_at: memory.created_at.to_rfc3339(),
        access_count: memory.access_count,
    }
}

fn similarity_graph_sync(
    conn: &Connection,
    namespace: &Namespace,
    threshold: f64,
    max_nodes: usize,
) -> Result<GraphData, DbError> {
    let ns_pattern = namespace.like_pattern();

    // 1. Fetch active memories with embeddings
    let mut stmt = conn.prepare(
        "SELECT id, content, namespace, status, created_at, access_count, embedding, memory_type, importance
         FROM memories
         WHERE namespace LIKE ?1 AND status = 'active'
           AND embedding IS NOT NULL AND length(embedding) > 0
         LIMIT ?2",
    )?;

    struct MemWithEmbed {
        node: GraphNode,
        embedding: Vec<f32>,
    }

    let rows = stmt.query_map(params![ns_pattern, max_nodes as i64], |row| {
        let content: String = row.get(1)?;
        let truncated = if content.len() > 100 {
            format!("{}...", &content[..100])
        } else {
            content
        };
        let blob: Vec<u8> = row.get(6)?;
        let embedding: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(MemWithEmbed {
            node: GraphNode {
                id: row.get(0)?,
                content: truncated,
                namespace: row.get(2)?,
                status: row.get(3)?,
                memory_type: row.get(7)?,
                importance: row.get(8)?,
                created_at: row.get(4)?,
                access_count: row.get(5)?,
            },
            embedding,
        })
    })?;

    let all_mems: Vec<MemWithEmbed> = rows
        .filter_map(|r| r.ok())
        .filter(|m| !m.embedding.is_empty())
        .collect();

    // Use only the dominant embedding dimension to avoid cross-model dimension mismatches
    let dominant_dim = {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for m in &all_mems {
            *counts.entry(m.embedding.len()).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(d, _)| d).unwrap_or(0)
    };
    let mems: Vec<MemWithEmbed> = all_mems
        .into_iter()
        .filter(|m| m.embedding.len() == dominant_dim)
        .collect();

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    let mut edge_set: HashSet<(String, String)> = HashSet::new();
    let mut node_ids: HashSet<String> = mems.iter().map(|m| m.node.id.clone()).collect();

    // 2. For each memory, find similar ones
    for mem in &mems {
        let results = match vector::search_vectors(conn, &mem.embedding, 10) {
            Ok(r) => r,
            Err(_) => continue, // skip if dimension mismatch or other vector error
        };

        for (neighbor_id, distance) in results {
            if neighbor_id == mem.node.id {
                continue;
            }
            if !node_ids.contains(&neighbor_id) {
                continue;
            }

            let similarity = 1.0 - (distance / 2.0);
            if similarity < threshold {
                continue;
            }

            // Deduplicate edges (A-B = B-A)
            let key = if mem.node.id < neighbor_id {
                (mem.node.id.clone(), neighbor_id.clone())
            } else {
                (neighbor_id.clone(), mem.node.id.clone())
            };

            if edge_set.insert(key.clone()) {
                edges.push(GraphEdge {
                    source: key.0,
                    target: key.1,
                    similarity,
                    edge_type: "similarity".into(),
                });
            }
        }
    }

    // Collect nodes
    for mem in mems {
        nodes.push(mem.node);
    }

    // 3. Fetch superseded memories and add superseded_by edges
    let mut sup_stmt = conn.prepare(
        "SELECT id, content, namespace, status, created_at, access_count, superseded_by, memory_type, importance
         FROM memories
         WHERE namespace LIKE ?1 AND status = 'superseded' AND superseded_by IS NOT NULL
         LIMIT ?2",
    )?;

    let sup_rows = sup_stmt.query_map(params![ns_pattern, max_nodes as i64], |row| {
        let content: String = row.get(1)?;
        let truncated = if content.len() > 100 {
            format!("{}...", &content[..100])
        } else {
            content
        };
        Ok((
            GraphNode {
                id: row.get(0)?,
                content: truncated,
                namespace: row.get(2)?,
                status: row.get(3)?,
                memory_type: row.get(7)?,
                importance: row.get(8)?,
                created_at: row.get(4)?,
                access_count: row.get(5)?,
            },
            row.get::<_, String>(6)?,
        ))
    })?;

    for row in sup_rows {
        if let Ok((node, superseded_by_id)) = row {
            let source_id = node.id.clone();
            node_ids.insert(source_id.clone());
            nodes.push(node);
            edges.push(GraphEdge {
                source: source_id,
                target: superseded_by_id,
                similarity: 1.0,
                edge_type: "superseded_by".into(),
            });
        }
    }

    // 4. Fetch causal/correlational edges and surface them in the graph.
    let mut causal_stmt = conn.prepare(
        "SELECT source_memory_id, target_memory_id, relation_type, confidence
         FROM causal_edges
         WHERE namespace LIKE ?1
         ORDER BY confidence DESC
         LIMIT ?2",
    )?;
    let causal_rows = causal_stmt.query_map(params![ns_pattern, (max_nodes * 4) as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, f64>(3)?,
        ))
    })?;
    let mut typed_edge_set: HashSet<(String, String, String)> = HashSet::new();
    for row in causal_rows {
        let (source, target, relation_type, confidence) = row?;
        if source == target {
            continue;
        }
        if !node_ids.contains(&source) {
            if let Some(mem) = get_memory_sync(conn, &source)? {
                let node = graph_node_from_memory(&mem);
                node_ids.insert(node.id.clone());
                nodes.push(node);
            }
        }
        if !node_ids.contains(&target) {
            if let Some(mem) = get_memory_sync(conn, &target)? {
                let node = graph_node_from_memory(&mem);
                node_ids.insert(node.id.clone());
                nodes.push(node);
            }
        }
        if !(node_ids.contains(&source) && node_ids.contains(&target)) {
            continue;
        }
        let key = (source.clone(), target.clone(), relation_type.clone());
        if typed_edge_set.insert(key) {
            edges.push(GraphEdge {
                source,
                target,
                similarity: confidence.clamp(0.0, 1.0),
                edge_type: relation_type,
            });
        }
    }

    Ok(GraphData { nodes, edges })
}

fn access_ranking_sync(
    conn: &Connection,
    namespace: &Namespace,
    ascending: bool,
    limit: usize,
) -> Result<Vec<Memory>, DbError> {
    let ns_pattern = namespace.like_pattern();
    let order = if ascending { "ASC" } else { "DESC" };
    let sql = format!(
        "SELECT id, namespace, content, metadata, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance, episode_id, event_date, category, confidence, source
         FROM memories
         WHERE namespace LIKE ?1 AND status = 'active'
         ORDER BY access_count {order}
         LIMIT ?2"
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params![ns_pattern, limit as i64], row_to_memory)?;
    let memories: Vec<Memory> = rows.filter_map(|r| r.ok()).collect();
    Ok(memories)
}

fn insert_calibration_observation_sync(
    conn: &Connection,
    namespace: &Namespace,
    kind: PredictionKind,
    prediction_id: Option<&str>,
    predicted_confidence: f64,
    outcome: Option<f64>,
    metadata: Option<serde_json::Value>,
) -> Result<(), DbError> {
    let now = Utc::now().to_rfc3339();
    let predicted = predicted_confidence.clamp(0.0, 1.0);
    let outcome = outcome.map(|v| v.clamp(0.0, 1.0));
    let resolved_at = outcome.map(|_| now.clone());
    let metadata_json = metadata.map(|m| serde_json::to_string(&m).unwrap_or_default());

    conn.execute(
        "INSERT INTO calibration_observations
            (namespace, prediction_kind, prediction_id, predicted_confidence, outcome, metadata, observed_at, resolved_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            namespace.as_str(),
            kind.as_str(),
            prediction_id,
            predicted,
            outcome,
            metadata_json,
            now,
            resolved_at,
        ],
    )
    .map_err(DbError::Sqlite)?;

    Ok(())
}

fn recompute_calibration_models_sync(conn: &Connection) -> Result<(), DbError> {
    let now = Utc::now().to_rfc3339();
    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;

    tx.execute("DELETE FROM calibration_models", [])
        .map_err(DbError::Sqlite)?;
    tx.execute("DELETE FROM calibration_bins", [])
        .map_err(DbError::Sqlite)?;

    {
        let mut model_stmt = tx
            .prepare(
                "SELECT
                    namespace,
                    prediction_kind,
                    COUNT(*) AS n,
                    AVG(predicted_confidence) AS avg_pred,
                    AVG(outcome) AS avg_out,
                    AVG((outcome - predicted_confidence) * (outcome - predicted_confidence)) AS mse,
                    SUM(predicted_confidence) AS sum_x,
                    SUM(outcome) AS sum_y,
                    SUM(predicted_confidence * predicted_confidence) AS sum_xx,
                    SUM(predicted_confidence * outcome) AS sum_xy
                 FROM calibration_observations
                 WHERE outcome IS NOT NULL
                 GROUP BY namespace, prediction_kind",
            )
            .map_err(DbError::Sqlite)?;
        let model_rows = model_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, u64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                    row.get::<_, f64>(6)?,
                    row.get::<_, f64>(7)?,
                    row.get::<_, f64>(8)?,
                    row.get::<_, f64>(9)?,
                ))
            })
            .map_err(DbError::Sqlite)?;

        for row in model_rows {
            let (namespace, kind, n, avg_pred, avg_out, mse, sum_x, sum_y, sum_xx, sum_xy) =
                row.map_err(DbError::Sqlite)?;
            let n_f = n as f64;
            let denom = n_f * sum_xx - sum_x * sum_x;

            let (raw_slope, raw_intercept) = if n >= 2 && denom.abs() > 1e-9 {
                let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
                let intercept = (sum_y - slope * sum_x) / n_f;
                (slope, intercept)
            } else {
                (1.0, 0.0)
            };

            let mut slope = if raw_slope.is_finite() {
                raw_slope.clamp(0.2, 2.5)
            } else {
                1.0
            };
            let mut intercept = if raw_intercept.is_finite() {
                raw_intercept.clamp(-0.5, 0.5)
            } else {
                0.0
            };

            // For low sample counts, stay close to identity calibration.
            let blend = (n_f / 50.0).min(1.0);
            slope = 1.0 + blend * (slope - 1.0);
            intercept *= blend;

            tx.execute(
                "INSERT INTO calibration_models
                    (namespace, prediction_kind, samples, avg_prediction, avg_outcome, mse, slope, intercept, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    namespace,
                    kind,
                    n as i64,
                    avg_pred,
                    avg_out,
                    mse,
                    slope,
                    intercept,
                    now,
                ],
            )
            .map_err(DbError::Sqlite)?;
        }
    }

    {
        let mut bins_stmt = tx
            .prepare(
                "SELECT
                    namespace,
                    prediction_kind,
                    CASE
                        WHEN predicted_confidence >= 1.0 THEN 9
                        WHEN predicted_confidence <= 0.0 THEN 0
                        ELSE CAST(predicted_confidence * 10 AS INTEGER)
                    END AS bin_index,
                    COUNT(*) AS n,
                    AVG(predicted_confidence) AS avg_pred,
                    AVG(outcome) AS avg_out
                 FROM calibration_observations
                 WHERE outcome IS NOT NULL
                 GROUP BY namespace, prediction_kind, bin_index",
            )
            .map_err(DbError::Sqlite)?;
        let bins_rows = bins_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i32>(2)?,
                    row.get::<_, u64>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                ))
            })
            .map_err(DbError::Sqlite)?;

        for row in bins_rows {
            let (namespace, kind, bin_index, n, avg_pred, avg_out) = row.map_err(DbError::Sqlite)?;
            tx.execute(
                "INSERT INTO calibration_bins
                    (namespace, prediction_kind, bin_index, sample_count, avg_prediction, avg_outcome, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![namespace, kind, bin_index, n as i64, avg_pred, avg_out, now],
            )
            .map_err(DbError::Sqlite)?;
        }
    }

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(())
}

fn calibrate_confidence_sync(
    conn: &Connection,
    namespace: &Namespace,
    kind: PredictionKind,
    raw_confidence: f64,
) -> Result<f64, DbError> {
    let raw = raw_confidence.clamp(0.0, 1.0);
    let mut stmt = conn
        .prepare_cached(
            "SELECT samples, slope, intercept
             FROM calibration_models
             WHERE namespace = ?1 AND prediction_kind = ?2
             LIMIT 1",
        )
        .map_err(DbError::Sqlite)?;

    let model = stmt
        .query_row(params![namespace.as_str(), kind.as_str()], |row| {
            Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })
        .optional()
        .map_err(DbError::Sqlite)?;

    let Some((samples, slope, intercept)) = model else {
        return Ok(raw);
    };

    let calibrated = (slope * raw + intercept).clamp(0.0, 1.0);
    let blend = (samples as f64 / 50.0).min(1.0);
    Ok((raw * (1.0 - blend) + calibrated * blend).clamp(0.0, 1.0))
}

fn calibration_metrics_sync(
    conn: &Connection,
    namespace: &Namespace,
) -> Result<CalibrationMetrics, DbError> {
    let ns_pattern = namespace.like_pattern();

    let mut models_stmt = conn
        .prepare(
            "SELECT namespace, prediction_kind, samples, avg_prediction, avg_outcome, mse, slope, intercept, updated_at
             FROM calibration_models
             WHERE namespace LIKE ?1
             ORDER BY prediction_kind ASC, namespace ASC",
        )
        .map_err(DbError::Sqlite)?;
    let model_rows = models_stmt
        .query_map(params![ns_pattern.clone()], |row| {
            Ok(CalibrationModel {
                namespace: row.get(0)?,
                prediction_kind: row.get(1)?,
                samples: row.get(2)?,
                avg_prediction: row.get(3)?,
                avg_outcome: row.get(4)?,
                mse: row.get(5)?,
                slope: row.get(6)?,
                intercept: row.get(7)?,
                updated_at: row.get(8)?,
            })
        })
        .map_err(DbError::Sqlite)?;
    let models: Vec<CalibrationModel> = model_rows.filter_map(|r| r.ok()).collect();

    let mut bins_stmt = conn
        .prepare(
            "SELECT namespace, prediction_kind, bin_index, sample_count, avg_prediction, avg_outcome, updated_at
             FROM calibration_bins
             WHERE namespace LIKE ?1
             ORDER BY prediction_kind ASC, bin_index ASC, namespace ASC",
        )
        .map_err(DbError::Sqlite)?;
    let bins_rows = bins_stmt
        .query_map(params![ns_pattern.clone()], |row| {
            Ok(CalibrationBin {
                namespace: row.get(0)?,
                prediction_kind: row.get(1)?,
                bin_index: row.get(2)?,
                sample_count: row.get(3)?,
                avg_prediction: row.get(4)?,
                avg_outcome: row.get(5)?,
                updated_at: row.get(6)?,
            })
        })
        .map_err(DbError::Sqlite)?;
    let bins: Vec<CalibrationBin> = bins_rows.filter_map(|r| r.ok()).collect();

    let unresolved: u64 = conn
        .prepare(
            "SELECT COUNT(*)
             FROM calibration_observations
             WHERE namespace LIKE ?1 AND outcome IS NULL",
        )
        .map_err(DbError::Sqlite)?
        .query_row(params![ns_pattern], |row| row.get(0))
        .map_err(DbError::Sqlite)?;

    Ok(CalibrationMetrics {
        namespace: namespace.as_str().to_string(),
        models,
        bins,
        unresolved_observations: unresolved,
    })
}

/// Compute optimal hybrid search weights from calibration observations.
///
/// Analyzes retrieval observations that have both vector_score and keyword_score
/// in their metadata, correlating each component score with positive outcomes
/// (score >= threshold). Returns `(weight_vector, weight_keyword)` normalized
/// to sum to 1.0, or None if insufficient data.
fn compute_optimal_hybrid_weights_sync(
    conn: &Connection,
) -> Result<Option<(f64, f64)>, DbError> {
    // Fetch recent retrieval observations that have component scores in metadata
    let mut stmt = conn
        .prepare(
            "SELECT metadata, outcome
             FROM calibration_observations
             WHERE prediction_kind = 'retrieval_relevance'
               AND outcome IS NOT NULL
               AND metadata IS NOT NULL
             ORDER BY created_at DESC
             LIMIT 5000",
        )
        .map_err(DbError::Sqlite)?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, f64>(1)?,
            ))
        })
        .map_err(DbError::Sqlite)?;

    let mut vec_score_sum = 0.0_f64;
    let mut kw_score_sum = 0.0_f64;
    let mut vec_outcome_sum = 0.0_f64;
    let mut kw_outcome_sum = 0.0_f64;
    let mut count = 0_usize;

    for row in rows {
        let (meta_str, outcome) = row.map_err(DbError::Sqlite)?;
        let Some(meta_str) = meta_str else { continue };
        let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) else { continue };

        let vec_score = meta.get("vector_score").and_then(|v| v.as_f64());
        let kw_score = meta.get("keyword_score").and_then(|v| v.as_f64());

        // Only use observations that have at least one component score
        if vec_score.is_none() && kw_score.is_none() {
            continue;
        }

        let vs = vec_score.unwrap_or(0.0);
        let ks = kw_score.unwrap_or(0.0);

        vec_score_sum += vs;
        kw_score_sum += ks;
        // Weight each component's contribution by whether the outcome was positive
        vec_outcome_sum += vs * outcome;
        kw_outcome_sum += ks * outcome;
        count += 1;
    }

    // Need at least 50 observations with component scores
    if count < 50 {
        return Ok(None);
    }

    // Compute the "contribution efficiency" of each source:
    // What fraction of each source's total score went to positive outcomes?
    let vec_efficiency = if vec_score_sum > 0.0 {
        vec_outcome_sum / vec_score_sum
    } else {
        0.0
    };
    let kw_efficiency = if kw_score_sum > 0.0 {
        kw_outcome_sum / kw_score_sum
    } else {
        0.0
    };

    let total = vec_efficiency + kw_efficiency;
    if total <= 0.0 {
        return Ok(None);
    }

    // Normalize to sum to 1.0, with floor of 0.1 to prevent zeroing out either source
    let raw_wv = vec_efficiency / total;
    let raw_wk = kw_efficiency / total;
    let wv = raw_wv.max(0.1);
    let wk = raw_wk.max(0.1);
    let norm = wv + wk;

    Ok(Some((wv / norm, wk / norm)))
}

fn insert_correction_event_sync(
    conn: &Connection,
    namespace: &Namespace,
    target_memory_id: &str,
    new_memory_id: &str,
    reason: Option<&str>,
    metadata: Option<serde_json::Value>,
) -> Result<CorrectionEvent, DbError> {
    let id = ulid::Ulid::new().to_string();
    let created_at = Utc::now().to_rfc3339();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());

    conn.execute(
        "INSERT INTO correction_events
            (id, namespace, target_memory_id, new_memory_id, reason, metadata, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            id,
            namespace.as_str(),
            target_memory_id,
            new_memory_id,
            reason,
            metadata_json,
            created_at,
        ],
    )
    .map_err(DbError::Sqlite)?;

    Ok(CorrectionEvent {
        id,
        namespace: namespace.as_str().to_string(),
        target_memory_id: target_memory_id.to_string(),
        new_memory_id: new_memory_id.to_string(),
        reason: reason.map(|r| r.to_string()),
        metadata,
        created_at,
    })
}

fn insert_contradiction_ledger_sync(
    conn: &Connection,
    namespace: &Namespace,
    old_memory_id: &str,
    new_memory_id: &str,
    resolution: &str,
    provenance: Option<serde_json::Value>,
) -> Result<(), DbError> {
    let id = ulid::Ulid::new().to_string();
    let created_at = Utc::now().to_rfc3339();
    let provenance_json = provenance
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());

    conn.execute(
        "INSERT INTO contradiction_ledger
            (id, namespace, old_memory_id, new_memory_id, resolution, provenance, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            id,
            namespace.as_str(),
            old_memory_id,
            new_memory_id,
            resolution,
            provenance_json,
            created_at,
        ],
    )
    .map_err(DbError::Sqlite)?;

    Ok(())
}

fn list_contradictions_sync(
    conn: &Connection,
    namespace: &Namespace,
    limit: usize,
    offset: usize,
) -> Result<Vec<ContradictionEntry>, DbError> {
    let ns_pattern = namespace.like_pattern();
    let mut stmt = conn.prepare(
        "SELECT c.id, c.namespace, c.old_memory_id, c.new_memory_id,
                c.resolution, c.provenance, c.created_at,
                m_old.content, m_new.content
         FROM contradiction_ledger c
         LEFT JOIN memories m_old ON m_old.id = c.old_memory_id
         LEFT JOIN memories m_new ON m_new.id = c.new_memory_id
         WHERE c.namespace LIKE ?1
         ORDER BY c.created_at DESC
         LIMIT ?2 OFFSET ?3",
    ).map_err(DbError::Sqlite)?;

    let rows = stmt.query_map(params![ns_pattern, limit as i64, offset as i64], |row| {
        let provenance_str: Option<String> = row.get(5)?;
        let provenance = provenance_str
            .and_then(|s| serde_json::from_str(&s).ok());
        Ok(ContradictionEntry {
            id: row.get(0)?,
            namespace: row.get(1)?,
            old_memory_id: row.get(2)?,
            new_memory_id: row.get(3)?,
            resolution: row.get(4)?,
            provenance,
            created_at: row.get(6)?,
            old_content: row.get(7)?,
            new_content: row.get(8)?,
        })
    }).map_err(DbError::Sqlite)?;

    rows.collect::<Result<Vec<_>, _>>().map_err(DbError::Sqlite)
}

fn find_contradictions_for_memories_sync(
    conn: &Connection,
    namespace: &Namespace,
    memory_ids: &[String],
) -> Result<Vec<ContradictionEntry>, DbError> {
    let ns_pattern = namespace.like_pattern();
    // Build IN clause with placeholders
    let placeholders: Vec<String> = (0..memory_ids.len()).map(|i| format!("?{}", i + 2)).collect();
    let in_clause = placeholders.join(",");
    let sql = format!(
        "SELECT c.id, c.namespace, c.old_memory_id, c.new_memory_id,
                c.resolution, c.provenance, c.created_at,
                m_old.content, m_new.content
         FROM contradiction_ledger c
         LEFT JOIN memories m_old ON m_old.id = c.old_memory_id
         LEFT JOIN memories m_new ON m_new.id = c.new_memory_id
         WHERE c.namespace LIKE ?1
           AND (c.old_memory_id IN ({in_clause}) OR c.new_memory_id IN ({in_clause}))
         ORDER BY c.created_at DESC"
    );
    let mut stmt = conn.prepare(&sql).map_err(DbError::Sqlite)?;

    // Build params: namespace pattern + memory_ids twice (for both IN clauses)
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params_vec.push(Box::new(ns_pattern));
    for id in memory_ids {
        params_vec.push(Box::new(id.clone()));
    }
    for id in memory_ids {
        params_vec.push(Box::new(id.clone()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        let provenance_str: Option<String> = row.get(5)?;
        let provenance = provenance_str
            .and_then(|s| serde_json::from_str(&s).ok());
        Ok(ContradictionEntry {
            id: row.get(0)?,
            namespace: row.get(1)?,
            old_memory_id: row.get(2)?,
            new_memory_id: row.get(3)?,
            resolution: row.get(4)?,
            provenance,
            created_at: row.get(6)?,
            old_content: row.get(7)?,
            new_content: row.get(8)?,
        })
    }).map_err(DbError::Sqlite)?;

    rows.collect::<Result<Vec<_>, _>>().map_err(DbError::Sqlite)
}

fn get_supersession_chain_sync(
    conn: &Connection,
    namespace: &Namespace,
    memory_id: &str,
) -> Result<Vec<SupersessionLink>, DbError> {
    let ns_pattern = namespace.like_pattern();
    let mut chain = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Walk backwards: find predecessors (memories that this one superseded)
    let mut current_id = memory_id.to_string();
    loop {
        if !seen.insert(current_id.clone()) { break; }
        let predecessor: Option<String> = conn.prepare(
            "SELECT old_memory_id FROM contradiction_ledger
             WHERE namespace LIKE ?1 AND new_memory_id = ?2
             ORDER BY created_at DESC LIMIT 1",
        ).map_err(DbError::Sqlite)?
        .query_row(params![ns_pattern, current_id], |row| row.get(0))
        .optional()
        .map_err(DbError::Sqlite)?;

        match predecessor {
            Some(pred_id) => { current_id = pred_id; }
            None => break,
        }
    }

    // Now walk forward from the earliest ancestor
    let start_id = current_id;
    let mut walk_id = start_id;
    seen.clear();

    loop {
        if !seen.insert(walk_id.clone()) { break; }
        // Fetch memory data
        let mem_data: Option<(String, String, Option<String>, f64, String, String)> = conn.prepare(
            "SELECT content, status, superseded_by, confidence, source, created_at
             FROM memories WHERE id = ?1 AND namespace LIKE ?2",
        ).map_err(DbError::Sqlite)?
        .query_row(params![walk_id, ns_pattern], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get::<_, f64>(3).unwrap_or(0.5),
                row.get::<_, String>(4).unwrap_or_else(|_| "unknown".into()),
                row.get(5)?,
            ))
        })
        .optional()
        .map_err(DbError::Sqlite)?;

        match mem_data {
            Some((content, status, superseded_by, confidence, source, created_at)) => {
                let next = superseded_by.clone();
                chain.push(SupersessionLink {
                    memory_id: walk_id.clone(),
                    content,
                    status,
                    superseded_by: superseded_by.clone(),
                    confidence,
                    source,
                    created_at,
                });
                match next {
                    Some(next_id) if !next_id.is_empty() => { walk_id = next_id; }
                    _ => break,
                }
            }
            None => break,
        }
    }

    Ok(chain)
}

#[derive(Debug, Clone)]
struct ClaimSnapshotRow {
    namespace: String,
    memory_id: String,
    content: String,
    metadata: Option<String>,
    tags: String,
    memory_type: String,
    importance: i32,
    status: String,
    superseded_by: Option<String>,
    hash: String,
    embedding: Option<Vec<u8>>,
}

fn get_claim_snapshot_sync(conn: &Connection, memory_id: &str) -> Result<Option<ClaimSnapshotRow>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT namespace, id, content, metadata, tags, memory_type, importance, status, superseded_by, hash, embedding
         FROM memories
         WHERE id = ?1
         LIMIT 1",
    )?;
    let row = stmt
        .query_row(params![memory_id], |row| {
            Ok(ClaimSnapshotRow {
                namespace: row.get(0)?,
                memory_id: row.get(1)?,
                content: row.get(2)?,
                metadata: row.get(3)?,
                tags: row.get::<_, Option<String>>(4)?.unwrap_or_else(|| "[]".to_string()),
                memory_type: row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "fact".to_string()),
                importance: row.get::<_, Option<i32>>(6)?.unwrap_or(5),
                status: row.get::<_, Option<String>>(7)?.unwrap_or_else(|| "active".to_string()),
                superseded_by: row.get(8)?,
                hash: row.get(9)?,
                embedding: row.get(10)?,
            })
        })
        .optional()?;
    Ok(row)
}

fn next_claim_revision_number_sync(
    conn: &Connection,
    namespace: &str,
    memory_id: &str,
) -> Result<i64, DbError> {
    let rev: i64 = conn
        .prepare_cached(
            "SELECT COALESCE(MAX(revision_number), 0) + 1
             FROM claim_revisions
             WHERE namespace = ?1 AND memory_id = ?2",
        )?
        .query_row(params![namespace, memory_id], |row| row.get(0))?;
    Ok(rev)
}

fn insert_audit_event_sync(
    conn: &Connection,
    namespace: &str,
    entity_type: &str,
    entity_id: &str,
    operation: &str,
    actor: Option<&str>,
    reason: Option<&str>,
    details: Option<serde_json::Value>,
) -> Result<(), DbError> {
    let id = ulid::Ulid::new().to_string();
    let now = Utc::now().to_rfc3339();
    let details_json = details
        .as_ref()
        .map(|d| serde_json::to_string(d).unwrap_or_default());
    conn.execute(
        "INSERT INTO memory_audit_log
            (id, namespace, entity_type, entity_id, operation, actor, reason, details, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            id,
            namespace,
            entity_type,
            entity_id,
            operation,
            actor,
            reason,
            details_json,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;
    Ok(())
}

fn snapshot_claim_revision_sync(
    conn: &Connection,
    memory_id: &str,
    operation: &str,
    actor: Option<&str>,
    reason: Option<&str>,
    provenance: Option<serde_json::Value>,
) -> Result<i64, DbError> {
    let Some(snapshot) = get_claim_snapshot_sync(conn, memory_id)? else {
        return Err(DbError::Other(format!(
            "cannot snapshot claim revision for missing memory_id={memory_id}"
        )));
    };
    let revision_number =
        next_claim_revision_number_sync(conn, &snapshot.namespace, &snapshot.memory_id)?;
    let created_at = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let provenance_json = provenance
        .as_ref()
        .map(|p| serde_json::to_string(p).unwrap_or_default());
    conn.execute(
        "INSERT INTO claim_revisions
            (id, namespace, memory_id, revision_number, operation, content, metadata, tags, memory_type, importance, status, superseded_by, hash, embedding, actor, reason, provenance, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        params![
            id,
            snapshot.namespace,
            snapshot.memory_id,
            revision_number,
            operation,
            snapshot.content,
            snapshot.metadata,
            snapshot.tags,
            snapshot.memory_type,
            snapshot.importance,
            snapshot.status,
            snapshot.superseded_by,
            snapshot.hash,
            snapshot.embedding,
            actor,
            reason,
            provenance_json,
            created_at,
        ],
    )
    .map_err(DbError::Sqlite)?;

    let mut details = serde_json::json!({
        "revision_number": revision_number,
    });
    if let Some(p) = provenance {
        details["provenance"] = p;
    }
    insert_audit_event_sync(
        conn,
        &snapshot.namespace,
        "claim",
        &snapshot.memory_id,
        operation,
        actor,
        reason,
        Some(details),
    )?;
    Ok(revision_number)
}

fn list_claim_revisions_sync(
    conn: &Connection,
    namespace: &Namespace,
    memory_id: &str,
    limit: usize,
) -> Result<Vec<ClaimRevision>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, memory_id, revision_number, operation, content, metadata, tags, memory_type, importance, status, superseded_by, hash, actor, reason, provenance, created_at
         FROM claim_revisions
         WHERE namespace LIKE ?1 AND memory_id = ?2
         ORDER BY revision_number DESC
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![namespace.like_pattern(), memory_id, limit as i64], |row| {
        let metadata: Option<String> = row.get(6)?;
        let tags_json: String = row.get::<_, Option<String>>(7)?.unwrap_or_else(|| "[]".to_string());
        let provenance: Option<String> = row.get(15)?;
        Ok(ClaimRevision {
            id: row.get(0)?,
            namespace: row.get(1)?,
            memory_id: row.get(2)?,
            revision_number: row.get(3)?,
            operation: row.get(4)?,
            content: row.get(5)?,
            metadata: metadata.and_then(|v| serde_json::from_str(&v).ok()),
            tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            memory_type: row.get(8)?,
            importance: row.get(9)?,
            status: row.get(10)?,
            superseded_by: row.get(11)?,
            hash: row.get(12)?,
            actor: row.get(13)?,
            reason: row.get(14)?,
            provenance: provenance.and_then(|v| serde_json::from_str(&v).ok()),
            created_at: row.get(16)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn patch_memory_sync(
    conn: &Connection,
    id: &str,
    patch: &MemoryPatch,
    new_embedding: Option<&[f32]>,
    actor: Option<&str>,
    reason: Option<&str>,
) -> Result<Option<MemoryMutationResult>, DbError> {
    let Some(before) = get_claim_snapshot_sync(conn, id)? else {
        return Ok(None);
    };
    if before.status != "active" {
        return Err(DbError::Other(format!(
            "cannot patch non-active memory {id} (status={})",
            before.status
        )));
    }
    if patch.content.is_none()
        && patch.metadata.is_none()
        && patch.memory_type.is_none()
        && patch.importance.is_none()
        && patch.tags.is_none()
    {
        return Err(DbError::Other("patch requires at least one field".to_string()));
    }
    if patch.content.is_some() && new_embedding.is_none() {
        return Err(DbError::Other(
            "new_embedding is required when patching content".to_string(),
        ));
    }

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;
    let now = Utc::now().to_rfc3339();
    let mut content_changed = false;
    let mut patched_content = before.content.clone();

    if let Some(content) = &patch.content {
        patched_content = content.clone();
        let patched_hash = content_hash(&patched_content);
        let embedding_blob: Vec<u8> = new_embedding
            .unwrap_or_default()
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tx.execute(
            "UPDATE memories
             SET content = ?1, embedding = ?2, hash = ?3, updated_at = ?4
             WHERE id = ?5",
            params![patched_content, embedding_blob, patched_hash, now, id],
        )?;
        content_changed = true;
    }

    if let Some(metadata) = &patch.metadata {
        let metadata_json = serde_json::to_string(metadata).unwrap_or_default();
        tx.execute(
            "UPDATE memories SET metadata = ?1, updated_at = ?2 WHERE id = ?3",
            params![metadata_json, now, id],
        )?;
    }

    if let Some(memory_type) = &patch.memory_type {
        tx.execute(
            "UPDATE memories SET memory_type = ?1, updated_at = ?2 WHERE id = ?3",
            params![memory_type, now, id],
        )?;
    }

    if let Some(importance) = patch.importance {
        tx.execute(
            "UPDATE memories SET importance = ?1, updated_at = ?2 WHERE id = ?3",
            params![importance.clamp(1, 10), now, id],
        )?;
    }

    if let Some(tags) = &patch.tags {
        let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".to_string());
        tx.execute(
            "UPDATE memories SET tags = ?1, updated_at = ?2 WHERE id = ?3",
            params![tags_json, now, id],
        )?;
    }

    if content_changed {
        let embedding = new_embedding.unwrap_or_default();
        vector::update_embedding(&tx, id, embedding, &before.namespace)?;
        fts::delete_fts(&tx, id)?;
        fts::insert_fts(&tx, id, &before.namespace, &patched_content)?;
    }

    let revision_number = snapshot_claim_revision_sync(
        &tx,
        id,
        "patch",
        actor,
        reason,
        Some(serde_json::json!({
            "content_changed": content_changed,
            "metadata_changed": patch.metadata.is_some(),
            "memory_type_changed": patch.memory_type.is_some(),
            "importance_changed": patch.importance.is_some(),
            "tags_changed": patch.tags.is_some(),
        })),
    )?;

    let memory = get_memory_sync(&tx, id)?.ok_or_else(|| {
        DbError::Other(format!("patched memory {id} is unexpectedly unavailable"))
    })?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(Some(MemoryMutationResult {
        memory,
        revision_number,
    }))
}

fn rollback_memory_to_revision_sync(
    conn: &Connection,
    id: &str,
    revision_number: i64,
    actor: Option<&str>,
    reason: Option<&str>,
) -> Result<Option<MemoryMutationResult>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT namespace, content, metadata, tags, memory_type, importance, hash, embedding
         FROM claim_revisions
         WHERE memory_id = ?1 AND revision_number = ?2
         LIMIT 1",
    )?;
    let row: Option<(String, String, Option<String>, String, String, i32, String, Option<Vec<u8>>)> =
        stmt.query_row(params![id, revision_number], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
            ))
        }).optional()?;

    let Some((namespace, content, metadata, tags_json, memory_type, importance, hash, embedding_blob)) =
        row
    else {
        return Ok(None);
    };
    let embedding_blob = embedding_blob.ok_or_else(|| {
        DbError::Other(format!(
            "revision {revision_number} for memory {id} has no embedding snapshot"
        ))
    })?;
    if embedding_blob.len() % 4 != 0 || embedding_blob.is_empty() {
        return Err(DbError::Other(format!(
            "revision {revision_number} for memory {id} has invalid embedding payload"
        )));
    }
    let embedding: Vec<f32> = embedding_blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;
    let now = Utc::now().to_rfc3339();
    let rows = tx.execute(
        "UPDATE memories
         SET content = ?1,
             metadata = ?2,
             tags = ?3,
             memory_type = ?4,
             importance = ?5,
             hash = ?6,
             embedding = ?7,
             status = 'active',
             superseded_by = NULL,
             updated_at = ?8
         WHERE id = ?9",
        params![
            content,
            metadata,
            tags_json,
            memory_type,
            importance,
            hash,
            embedding_blob,
            now,
            id,
        ],
    )?;
    if rows == 0 {
        return Ok(None);
    }

    vector::update_embedding(&tx, id, &embedding, &namespace)?;
    fts::delete_fts(&tx, id)?;
    let restored_content: String = tx
        .prepare_cached("SELECT content FROM memories WHERE id = ?1")?
        .query_row(params![id], |row| row.get(0))?;
    fts::insert_fts(&tx, id, &namespace, &restored_content)?;

    let new_revision_number = snapshot_claim_revision_sync(
        &tx,
        id,
        "rollback",
        actor,
        reason,
        Some(serde_json::json!({ "rolled_back_to_revision": revision_number })),
    )?;

    let memory = get_memory_sync(&tx, id)?.ok_or_else(|| {
        DbError::Other(format!("rolled back memory {id} is unexpectedly unavailable"))
    })?;
    tx.commit().map_err(DbError::Sqlite)?;
    Ok(Some(MemoryMutationResult {
        memory,
        revision_number: new_revision_number,
    }))
}

fn merge_memories_sync(
    conn: &Connection,
    namespace: &Namespace,
    source_memory_ids: &[String],
    merged_memory: &Memory,
    merged_embedding: &[f32],
    actor: Option<&str>,
    reason: Option<&str>,
) -> Result<MemoryMergeResult, DbError> {
    if source_memory_ids.is_empty() {
        return Err(DbError::Other(
            "merge requires at least one source memory id".to_string(),
        ));
    }
    let mut unique: HashSet<&String> = HashSet::new();
    for source_id in source_memory_ids {
        if !unique.insert(source_id) {
            return Err(DbError::Other(format!(
                "duplicate source memory id in merge request: {source_id}"
            )));
        }
    }

    let tx = conn.unchecked_transaction().map_err(DbError::Sqlite)?;
    for source_id in source_memory_ids {
        let Some(source) = get_claim_snapshot_sync(&tx, source_id)? else {
            return Err(DbError::Other(format!(
                "cannot merge missing source memory {source_id}"
            )));
        };
        if source.status != "active" {
            return Err(DbError::Other(format!(
                "cannot merge non-active source memory {source_id} (status={})",
                source.status
            )));
        }
        if !namespace_contains_str(namespace, &source.namespace) {
            return Err(DbError::Other(format!(
                "source memory {source_id} is outside namespace '{}'",
                namespace.as_str()
            )));
        }
    }

    let merged_metadata_json = merged_memory
        .metadata
        .as_ref()
        .map(|v| serde_json::to_string(v).unwrap_or_default());
    let merged_tags_json =
        serde_json::to_string(&merged_memory.tags).unwrap_or_else(|_| "[]".to_string());
    let embedding_blob: Vec<u8> = merged_embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    tx.execute(
        "INSERT INTO memories (id, namespace, content, metadata, embedding, status, created_at, updated_at, accessed_at, access_count, hash, tags, memory_type, importance)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            merged_memory.id,
            merged_memory.namespace,
            merged_memory.content,
            merged_metadata_json,
            embedding_blob,
            merged_memory.status.as_str(),
            merged_memory.created_at.to_rfc3339(),
            merged_memory.updated_at.to_rfc3339(),
            merged_memory.accessed_at.to_rfc3339(),
            merged_memory.access_count,
            merged_memory.hash,
            merged_tags_json,
            merged_memory.memory_type,
            merged_memory.importance,
        ],
    )?;
    vector::insert_embedding(&tx, &merged_memory.id, merged_embedding, &merged_memory.namespace)?;
    fts::insert_fts(
        &tx,
        &merged_memory.id,
        &merged_memory.namespace,
        &merged_memory.content,
    )?;
    let merged_revision_number = snapshot_claim_revision_sync(
        &tx,
        &merged_memory.id,
        "merge_created",
        actor,
        reason,
        Some(serde_json::json!({ "source_memory_ids": source_memory_ids })),
    )?;

    let now = Utc::now().to_rfc3339();
    for source_id in source_memory_ids {
        tx.execute(
            "UPDATE memories
             SET status = 'superseded', superseded_by = ?1, updated_at = ?2
             WHERE id = ?3",
            params![merged_memory.id, now, source_id],
        )?;
        vector::delete_embedding(&tx, source_id)?;
        fts::delete_fts(&tx, source_id)?;
        snapshot_claim_revision_sync(
            &tx,
            source_id,
            "merge_source",
            actor,
            reason,
            Some(serde_json::json!({ "merged_into": merged_memory.id })),
        )?;
    }

    insert_audit_event_sync(
        &tx,
        namespace.as_str(),
        "claim_merge",
        &merged_memory.id,
        "merge",
        actor,
        reason,
        Some(serde_json::json!({
            "source_memory_ids": source_memory_ids,
            "merged_memory_id": merged_memory.id,
            "merged_revision_number": merged_revision_number,
        })),
    )?;

    tx.commit().map_err(DbError::Sqlite)?;
    Ok(MemoryMergeResult {
        merged_memory_id: merged_memory.id.clone(),
        superseded_source_ids: source_memory_ids.to_vec(),
        merged_revision_number,
    })
}

fn upsert_procedure_revision_sync(
    conn: &Connection,
    namespace: &Namespace,
    procedure_name: &str,
    operation: &str,
    spec: &serde_json::Value,
    actor: Option<&str>,
    reason: Option<&str>,
) -> Result<ProcedureRevision, DbError> {
    let revision_number: i64 = conn
        .prepare_cached(
            "SELECT COALESCE(MAX(revision_number), 0) + 1
             FROM procedure_revisions
             WHERE namespace = ?1 AND procedure_name = ?2",
        )?
        .query_row(params![namespace.as_str(), procedure_name], |row| row.get(0))?;
    let spec_json = serde_json::to_string(spec).unwrap_or_else(|_| "{}".to_string());
    let id = ulid::Ulid::new().to_string();
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO procedure_revisions
            (id, namespace, procedure_name, revision_number, operation, spec, actor, reason, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            id,
            namespace.as_str(),
            procedure_name,
            revision_number,
            operation,
            spec_json,
            actor,
            reason,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "procedure",
        procedure_name,
        operation,
        actor,
        reason,
        Some(serde_json::json!({ "revision_number": revision_number })),
    )?;
    Ok(ProcedureRevision {
        id,
        namespace: namespace.as_str().to_string(),
        procedure_name: procedure_name.to_string(),
        revision_number,
        operation: operation.to_string(),
        spec: spec.clone(),
        actor: actor.map(|v| v.to_string()),
        reason: reason.map(|v| v.to_string()),
        created_at: now,
    })
}

fn list_procedure_revisions_sync(
    conn: &Connection,
    namespace: &Namespace,
    procedure_name: &str,
    limit: usize,
) -> Result<Vec<ProcedureRevision>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, procedure_name, revision_number, operation, spec, actor, reason, created_at
         FROM procedure_revisions
         WHERE namespace LIKE ?1 AND procedure_name = ?2
         ORDER BY revision_number DESC
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![namespace.like_pattern(), procedure_name, limit as i64], |row| {
        let spec_json: String = row.get(5)?;
        Ok(ProcedureRevision {
            id: row.get(0)?,
            namespace: row.get(1)?,
            procedure_name: row.get(2)?,
            revision_number: row.get(3)?,
            operation: row.get(4)?,
            spec: serde_json::from_str(&spec_json).unwrap_or_else(|_| serde_json::json!({})),
            actor: row.get(6)?,
            reason: row.get(7)?,
            created_at: row.get(8)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn list_audit_events_sync(
    conn: &Connection,
    namespace: &Namespace,
    entity_type: Option<&str>,
    entity_id: Option<&str>,
    limit: usize,
) -> Result<Vec<AuditEvent>, DbError> {
    let ns_pattern = namespace.like_pattern();
    let mut rows_out = Vec::new();

    let (sql, params_any): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match (entity_type, entity_id) {
        (Some(t), Some(id)) => (
            "SELECT id, namespace, entity_type, entity_id, operation, actor, reason, details, created_at
             FROM memory_audit_log
             WHERE namespace LIKE ?1 AND entity_type = ?2 AND entity_id = ?3
             ORDER BY created_at DESC
             LIMIT ?4".to_string(),
            vec![Box::new(ns_pattern), Box::new(t.to_string()), Box::new(id.to_string()), Box::new(limit as i64)],
        ),
        (Some(t), None) => (
            "SELECT id, namespace, entity_type, entity_id, operation, actor, reason, details, created_at
             FROM memory_audit_log
             WHERE namespace LIKE ?1 AND entity_type = ?2
             ORDER BY created_at DESC
             LIMIT ?3".to_string(),
            vec![Box::new(ns_pattern), Box::new(t.to_string()), Box::new(limit as i64)],
        ),
        (None, Some(id)) => (
            "SELECT id, namespace, entity_type, entity_id, operation, actor, reason, details, created_at
             FROM memory_audit_log
             WHERE namespace LIKE ?1 AND entity_id = ?2
             ORDER BY created_at DESC
             LIMIT ?3".to_string(),
            vec![Box::new(ns_pattern), Box::new(id.to_string()), Box::new(limit as i64)],
        ),
        (None, None) => (
            "SELECT id, namespace, entity_type, entity_id, operation, actor, reason, details, created_at
             FROM memory_audit_log
             WHERE namespace LIKE ?1
             ORDER BY created_at DESC
             LIMIT ?2".to_string(),
            vec![Box::new(ns_pattern), Box::new(limit as i64)],
        ),
    };
    let params_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_any.iter().map(|p| p.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_refs.as_slice(), |row| {
        let details_json: Option<String> = row.get(7)?;
        Ok(AuditEvent {
            id: row.get(0)?,
            namespace: row.get(1)?,
            entity_type: row.get(2)?,
            entity_id: row.get(3)?,
            operation: row.get(4)?,
            actor: row.get(5)?,
            reason: row.get(6)?,
            details: details_json.and_then(|v| serde_json::from_str(&v).ok()),
            created_at: row.get(8)?,
        })
    })?;
    for row in rows {
        rows_out.push(row.map_err(DbError::Sqlite)?);
    }
    Ok(rows_out)
}

fn normalize_identity_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.trim().to_lowercase().chars() {
        match ch {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => out.push('a'),
            'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => out.push('c'),
            'ď' | 'đ' => out.push('d'),
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => out.push('e'),
            'ĝ' | 'ğ' | 'ġ' | 'ģ' => out.push('g'),
            'ĥ' | 'ħ' => out.push('h'),
            'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' => out.push('i'),
            'ĵ' => out.push('j'),
            'ķ' => out.push('k'),
            'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => out.push('l'),
            'ñ' | 'ń' | 'ņ' | 'ň' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => out.push('o'),
            'ŕ' | 'ŗ' | 'ř' => out.push('r'),
            'ś' | 'ŝ' | 'ş' | 'š' | 'ș' => out.push('s'),
            'ţ' | 'ť' | 'ŧ' | 'ț' => out.push('t'),
            'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => out.push('u'),
            'ŵ' => out.push('w'),
            'ý' | 'ÿ' | 'ŷ' => out.push('y'),
            'ź' | 'ż' | 'ž' => out.push('z'),
            'ß' => {
                out.push('s');
                out.push('s');
            }
            'æ' => {
                out.push('a');
                out.push('e');
            }
            'œ' => {
                out.push('o');
                out.push('e');
            }
            '\'' | '"' | '.' | ',' | ':' | ';' | '-' | '_' | '/' | '\\' | '(' | ')' | '['
            | ']' | '{' | '}' | '|' | '\t' | '\n' | '\r' => out.push(' '),
            _ if ch.is_alphanumeric() || ch.is_whitespace() => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a.is_empty() {
        return b.chars().count();
    }
    if b.is_empty() {
        return a.chars().count();
    }
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut cur = vec![0; b_chars.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let substitution = if ca == *cb { prev[j] } else { prev[j] + 1 };
            let insertion = cur[j] + 1;
            let deletion = prev[j + 1] + 1;
            cur[j + 1] = substitution.min(insertion.min(deletion));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b_chars.len()]
}

fn similarity_ratio(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let d = levenshtein_distance(a, b) as f64;
    (1.0 - d / max_len as f64).clamp(0.0, 1.0)
}

fn upsert_identity_entity_sync(
    conn: &Connection,
    namespace: &Namespace,
    canonical_name: &str,
    language: Option<&str>,
    confidence: f64,
    metadata: Option<serde_json::Value>,
) -> Result<IdentityEntity, DbError> {
    let canonical = canonical_name.trim();
    if canonical.is_empty() {
        return Err(DbError::Other("canonical_name cannot be empty".to_string()));
    }
    let normalized = normalize_identity_text(canonical);
    if normalized.is_empty() {
        return Err(DbError::Other(
            "canonical_name normalized to empty value".to_string(),
        ));
    }
    let lang = language.unwrap_or("und");
    let conf = confidence.clamp(0.0, 1.0);
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());

    conn.execute(
        "INSERT INTO identity_entities
            (id, namespace, canonical_name, normalized_name, language, confidence, ambiguity, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.0, ?7, ?8, ?8)
         ON CONFLICT(namespace, normalized_name)
         DO UPDATE SET
            canonical_name = excluded.canonical_name,
            language = CASE WHEN excluded.language = 'und' THEN identity_entities.language ELSE excluded.language END,
            confidence = (identity_entities.confidence * 0.7) + (excluded.confidence * 0.3),
            metadata = COALESCE(excluded.metadata, identity_entities.metadata),
            updated_at = excluded.updated_at",
        params![
            id,
            namespace.as_str(),
            canonical,
            normalized,
            lang,
            conf,
            metadata_json,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;

    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, canonical_name, normalized_name, language, confidence, ambiguity, metadata, created_at, updated_at
         FROM identity_entities
         WHERE namespace = ?1 AND normalized_name = ?2
         LIMIT 1",
    )?;
    let entity: IdentityEntity = stmt
        .query_row(params![namespace.as_str(), normalized], |row| {
            let metadata_str: Option<String> = row.get(7)?;
            Ok(IdentityEntity {
                id: row.get(0)?,
                namespace: row.get(1)?,
                canonical_name: row.get(2)?,
                normalized_name: row.get(3)?,
                language: row.get(4)?,
                confidence: row.get::<_, f64>(5)?.clamp(0.0, 1.0),
                ambiguity: row.get::<_, f64>(6)?.clamp(0.0, 1.0),
                metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                created_at: row.get(8)?,
                updated_at: row.get(9)?,
            })
        })
        .map_err(DbError::Sqlite)?;
    add_identity_alias_sync(
        conn,
        namespace,
        &entity.id,
        &entity.canonical_name,
        Some(&entity.language),
        0.98,
    )?;
    Ok(entity)
}

fn recompute_entity_ambiguity_sync(
    conn: &Connection,
    namespace: &Namespace,
    entity_id: &str,
) -> Result<(), DbError> {
    let max_competitor_count: i64 = conn
        .prepare_cached(
            "SELECT COALESCE(MAX(alias_counts.cnt), 1)
             FROM identity_aliases ia
             JOIN (
                SELECT normalized_alias, COUNT(DISTINCT entity_id) AS cnt
                FROM identity_aliases
                WHERE namespace = ?1
                GROUP BY normalized_alias
             ) alias_counts
               ON alias_counts.normalized_alias = ia.normalized_alias
             WHERE ia.namespace = ?1 AND ia.entity_id = ?2",
        )?
        .query_row(params![namespace.as_str(), entity_id], |row| row.get(0))
        .map_err(DbError::Sqlite)?;
    let ambiguity = if max_competitor_count <= 1 {
        0.0
    } else {
        ((max_competitor_count as f64 - 1.0) / max_competitor_count as f64).clamp(0.0, 1.0)
    };
    conn.execute(
        "UPDATE identity_entities
         SET ambiguity = ?1, updated_at = ?2
         WHERE namespace = ?3 AND id = ?4",
        params![
            ambiguity,
            Utc::now().to_rfc3339(),
            namespace.as_str(),
            entity_id,
        ],
    )
    .map_err(DbError::Sqlite)?;
    Ok(())
}

fn add_identity_alias_sync(
    conn: &Connection,
    namespace: &Namespace,
    entity_id: &str,
    alias: &str,
    language: Option<&str>,
    confidence: f64,
) -> Result<(), DbError> {
    let alias = alias.trim();
    if alias.is_empty() {
        return Err(DbError::Other("alias cannot be empty".to_string()));
    }
    let normalized_alias = normalize_identity_text(alias);
    if normalized_alias.is_empty() {
        return Err(DbError::Other("alias normalized to empty value".to_string()));
    }
    let exists: bool = conn
        .prepare_cached(
            "SELECT 1 FROM identity_entities
             WHERE namespace = ?1 AND id = ?2",
        )?
        .exists(params![namespace.as_str(), entity_id])
        .map_err(DbError::Sqlite)?;
    if !exists {
        return Err(DbError::Other(format!(
            "entity {entity_id} does not exist in namespace {}",
            namespace.as_str()
        )));
    }

    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let lang = language.unwrap_or("und");
    let conf = confidence.clamp(0.0, 1.0);
    conn.execute(
        "INSERT INTO identity_aliases
            (id, namespace, entity_id, alias, normalized_alias, language, confidence, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
         ON CONFLICT(namespace, normalized_alias, entity_id)
         DO UPDATE SET
            alias = excluded.alias,
            language = CASE WHEN excluded.language = 'und' THEN identity_aliases.language ELSE excluded.language END,
            confidence = MAX(identity_aliases.confidence, excluded.confidence),
            updated_at = excluded.updated_at",
        params![
            id,
            namespace.as_str(),
            entity_id,
            alias,
            normalized_alias,
            lang,
            conf,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;

    let mut related_entities = Vec::new();
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT entity_id
         FROM identity_aliases
         WHERE namespace = ?1 AND normalized_alias = ?2",
    )?;
    let rows = stmt
        .query_map(params![namespace.as_str(), normalized_alias], |row| {
            row.get::<_, String>(0)
        })
        .map_err(DbError::Sqlite)?;
    for row in rows {
        related_entities.push(row.map_err(DbError::Sqlite)?);
    }
    for eid in related_entities {
        recompute_entity_ambiguity_sync(conn, namespace, &eid)?;
    }
    Ok(())
}

fn resolve_identity_sync(
    conn: &Connection,
    namespace: &Namespace,
    query: &str,
    limit: usize,
) -> Result<IdentityResolution, DbError> {
    let normalized_query = normalize_identity_text(query);
    if normalized_query.is_empty() {
        return Ok(IdentityResolution {
            query: query.to_string(),
            normalized_query,
            best_confidence: 0.0,
            ambiguous: false,
            candidates: Vec::new(),
        });
    }
    let limit = limit.max(1).min(20);
    let mut candidate_map: HashMap<String, IdentityCandidate> = HashMap::new();

    let mut exact_stmt = conn.prepare(
        "SELECT e.id, e.canonical_name, a.alias, e.language, e.confidence, a.confidence, e.ambiguity
         FROM identity_aliases a
         JOIN identity_entities e ON e.id = a.entity_id
         WHERE a.namespace LIKE ?1 AND a.normalized_alias = ?2
         ORDER BY a.confidence DESC, e.confidence DESC
         LIMIT ?3",
    )?;
    let exact_rows = exact_stmt.query_map(
        params![namespace.like_pattern(), normalized_query, (limit * 4) as i64],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?.clamp(0.0, 1.0),
                row.get::<_, f64>(5)?.clamp(0.0, 1.0),
                row.get::<_, f64>(6)?.clamp(0.0, 1.0),
            ))
        },
    )?;
    for row in exact_rows {
        let (entity_id, canonical_name, matched_alias, language, entity_conf, alias_conf, ambiguity) =
            row.map_err(DbError::Sqlite)?;
        let match_score = 1.0;
        let final_score =
            (0.55 * match_score + 0.25 * alias_conf + 0.20 * entity_conf - 0.20 * ambiguity)
                .clamp(0.0, 1.0);
        let candidate = IdentityCandidate {
            entity_id: entity_id.clone(),
            canonical_name,
            matched_alias,
            language,
            match_kind: "exact".to_string(),
            score: final_score,
            entity_confidence: entity_conf,
            alias_confidence: alias_conf,
            ambiguity,
        };
        candidate_map
            .entry(entity_id)
            .and_modify(|current| {
                if candidate.score > current.score {
                    *current = candidate.clone();
                }
            })
            .or_insert(candidate);
    }

    if candidate_map.len() < limit {
        let prefix_like = format!("{normalized_query}%");
        let mut prefix_stmt = conn.prepare(
            "SELECT e.id, e.canonical_name, a.alias, e.language, e.confidence, a.confidence, e.ambiguity, a.normalized_alias
             FROM identity_aliases a
             JOIN identity_entities e ON e.id = a.entity_id
             WHERE a.namespace LIKE ?1 AND a.normalized_alias LIKE ?2
             ORDER BY a.confidence DESC, e.confidence DESC
             LIMIT ?3",
        )?;
        let prefix_rows = prefix_stmt.query_map(
            params![namespace.like_pattern(), prefix_like, (limit * 8) as i64],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?.clamp(0.0, 1.0),
                    row.get::<_, f64>(5)?.clamp(0.0, 1.0),
                    row.get::<_, f64>(6)?.clamp(0.0, 1.0),
                    row.get::<_, String>(7)?,
                ))
            },
        )?;
        for row in prefix_rows {
            let (
                entity_id,
                canonical_name,
                matched_alias,
                language,
                entity_conf,
                alias_conf,
                ambiguity,
                normalized_alias,
            ) = row.map_err(DbError::Sqlite)?;
            let match_score = if normalized_alias == normalized_query {
                1.0
            } else if normalized_alias.starts_with(&normalized_query) {
                0.86
            } else {
                0.72
            };
            let final_score =
                (0.55 * match_score + 0.25 * alias_conf + 0.20 * entity_conf - 0.20 * ambiguity)
                    .clamp(0.0, 1.0);
            let candidate = IdentityCandidate {
                entity_id: entity_id.clone(),
                canonical_name,
                matched_alias,
                language,
                match_kind: "prefix".to_string(),
                score: final_score,
                entity_confidence: entity_conf,
                alias_confidence: alias_conf,
                ambiguity,
            };
            candidate_map
                .entry(entity_id)
                .and_modify(|current| {
                    if candidate.score > current.score {
                        *current = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }
    }

    if candidate_map.len() < limit {
        let mut fuzzy_stmt = conn.prepare(
            "SELECT e.id, e.canonical_name, a.alias, e.language, e.confidence, a.confidence, e.ambiguity, a.normalized_alias
             FROM identity_aliases a
             JOIN identity_entities e ON e.id = a.entity_id
             WHERE a.namespace LIKE ?1
             ORDER BY a.updated_at DESC
             LIMIT 200",
        )?;
        let fuzzy_rows = fuzzy_stmt.query_map(params![namespace.like_pattern()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?.clamp(0.0, 1.0),
                row.get::<_, f64>(5)?.clamp(0.0, 1.0),
                row.get::<_, f64>(6)?.clamp(0.0, 1.0),
                row.get::<_, String>(7)?,
            ))
        })?;
        for row in fuzzy_rows {
            let (
                entity_id,
                canonical_name,
                matched_alias,
                language,
                entity_conf,
                alias_conf,
                ambiguity,
                normalized_alias,
            ) = row.map_err(DbError::Sqlite)?;
            let sim = similarity_ratio(&normalized_alias, &normalized_query);
            if sim < 0.68 {
                continue;
            }
            let final_score =
                (0.55 * sim + 0.25 * alias_conf + 0.20 * entity_conf - 0.20 * ambiguity)
                    .clamp(0.0, 1.0);
            let candidate = IdentityCandidate {
                entity_id: entity_id.clone(),
                canonical_name,
                matched_alias,
                language,
                match_kind: "fuzzy".to_string(),
                score: final_score,
                entity_confidence: entity_conf,
                alias_confidence: alias_conf,
                ambiguity,
            };
            candidate_map
                .entry(entity_id)
                .and_modify(|current| {
                    if candidate.score > current.score {
                        *current = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }
    }

    let mut candidates: Vec<IdentityCandidate> = candidate_map.into_values().collect();
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(limit);

    let best_confidence = candidates.first().map(|c| c.score).unwrap_or(0.0);
    let ambiguous = if candidates.len() < 2 {
        best_confidence < 0.62
    } else {
        let second = candidates[1].score;
        let margin = best_confidence - second;
        best_confidence < 0.62 || margin < 0.10 || candidates[0].ambiguity > 0.35
    };
    Ok(IdentityResolution {
        query: query.to_string(),
        normalized_query,
        best_confidence,
        ambiguous,
        candidates,
    })
}

fn normalize_plan_status(status: &str) -> &'static str {
    match status.trim().to_ascii_lowercase().as_str() {
        "active" | "in_progress" | "running" => "active",
        "completed" | "done" | "success" => "completed",
        "failed" | "error" => "failed",
        "abandoned" | "cancelled" | "canceled" => "abandoned",
        _ => "active",
    }
}

fn normalize_checkpoint_status(status: &str) -> &'static str {
    match status.trim().to_ascii_lowercase().as_str() {
        "pending" | "todo" => "pending",
        "in_progress" | "running" | "active" => "in_progress",
        "completed" | "done" | "success" => "completed",
        "failed" | "error" => "failed",
        "skipped" => "skipped",
        _ => "pending",
    }
}

fn normalize_branch_status(status: &str) -> &'static str {
    match status.trim().to_ascii_lowercase().as_str() {
        "active" | "open" | "pending" => "active",
        "resolved" | "completed" | "done" => "resolved",
        "abandoned" | "cancelled" | "canceled" => "abandoned",
        _ => "active",
    }
}

fn normalize_binding_role(role: &str) -> &'static str {
    match role.trim().to_ascii_lowercase().as_str() {
        "primary" | "default" => "primary",
        "fallback" | "recovery" => "fallback",
        "rollback" => "rollback",
        "monitor" | "observer" => "monitor",
        _ => "primary",
    }
}

fn row_to_plan_trace(row: &rusqlite::Row<'_>) -> rusqlite::Result<PlanTrace> {
    let metadata_str: Option<String> = row.get(6)?;
    Ok(PlanTrace {
        id: row.get(0)?,
        namespace: row.get(1)?,
        goal: row.get(2)?,
        status: row.get(3)?,
        priority: row.get(4)?,
        due_at: row.get(5)?,
        metadata: metadata_str.and_then(|v| serde_json::from_str(&v).ok()),
        outcome: row.get(7)?,
        outcome_confidence: row.get(8)?,
        finished_at: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
    })
}

fn row_to_plan_checkpoint(row: &rusqlite::Row<'_>) -> rusqlite::Result<PlanCheckpoint> {
    let metadata_str: Option<String> = row.get(10)?;
    Ok(PlanCheckpoint {
        id: row.get(0)?,
        namespace: row.get(1)?,
        plan_id: row.get(2)?,
        checkpoint_key: row.get(3)?,
        title: row.get(4)?,
        order_index: row.get(5)?,
        status: row.get(6)?,
        expected_by: row.get(7)?,
        completed_at: row.get(8)?,
        evidence: row.get(9)?,
        metadata: metadata_str.and_then(|v| serde_json::from_str(&v).ok()),
        created_at: row.get(11)?,
        updated_at: row.get(12)?,
    })
}

fn row_to_plan_recovery_branch(row: &rusqlite::Row<'_>) -> rusqlite::Result<PlanRecoveryBranch> {
    let branch_plan_str: Option<String> = row.get(7)?;
    let metadata_str: Option<String> = row.get(8)?;
    Ok(PlanRecoveryBranch {
        id: row.get(0)?,
        namespace: row.get(1)?,
        plan_id: row.get(2)?,
        source_checkpoint_id: row.get(3)?,
        branch_label: row.get(4)?,
        trigger_reason: row.get(5)?,
        status: row.get(6)?,
        branch_plan: branch_plan_str.and_then(|v| serde_json::from_str(&v).ok()),
        metadata: metadata_str.and_then(|v| serde_json::from_str(&v).ok()),
        resolution_notes: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
        resolved_at: row.get(12)?,
    })
}

fn row_to_plan_procedure_binding(row: &rusqlite::Row<'_>) -> rusqlite::Result<PlanProcedureBinding> {
    let metadata_str: Option<String> = row.get(6)?;
    Ok(PlanProcedureBinding {
        id: row.get(0)?,
        namespace: row.get(1)?,
        plan_id: row.get(2)?,
        procedure_name: row.get(3)?,
        binding_role: row.get(4)?,
        confidence: row.get::<_, f64>(5)?.clamp(0.0, 1.0),
        metadata: metadata_str.and_then(|v| serde_json::from_str(&v).ok()),
        created_at: row.get(7)?,
        updated_at: row.get(8)?,
    })
}

fn create_plan_trace_sync(
    conn: &Connection,
    namespace: &Namespace,
    goal: &str,
    priority: i32,
    due_at: Option<&str>,
    metadata: Option<serde_json::Value>,
) -> Result<PlanTrace, DbError> {
    let goal = goal.trim();
    if goal.is_empty() {
        return Err(DbError::Other("plan goal cannot be empty".to_string()));
    }
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "INSERT INTO plan_traces
            (id, namespace, goal, status, priority, due_at, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, 'active', ?4, ?5, ?6, ?7, ?7)",
        params![
            id,
            namespace.as_str(),
            goal,
            priority.clamp(1, 10),
            due_at,
            metadata_json,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;

    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan",
        &id,
        "create",
        None,
        None,
        Some(serde_json::json!({ "goal": goal })),
    )?;

    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, goal, status, priority, due_at, metadata, outcome, outcome_confidence, finished_at, created_at, updated_at
         FROM plan_traces WHERE id = ?1",
    )?;
    let plan = stmt
        .query_row(params![id], row_to_plan_trace)
        .map_err(DbError::Sqlite)?;
    Ok(plan)
}

fn list_plan_traces_sync(
    conn: &Connection,
    namespace: &Namespace,
    status: Option<&str>,
    limit: usize,
) -> Result<Vec<PlanTrace>, DbError> {
    let limit = limit.clamp(1, 500);
    let mut plans = Vec::new();
    if let Some(status) = status {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, goal, status, priority, due_at, metadata, outcome, outcome_confidence, finished_at, created_at, updated_at
             FROM plan_traces
             WHERE namespace LIKE ?1 AND status = ?2
             ORDER BY updated_at DESC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![namespace.like_pattern(), normalize_plan_status(status), limit as i64],
            row_to_plan_trace,
        )?;
        for row in rows {
            plans.push(row.map_err(DbError::Sqlite)?);
        }
    } else {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, goal, status, priority, due_at, metadata, outcome, outcome_confidence, finished_at, created_at, updated_at
             FROM plan_traces
             WHERE namespace LIKE ?1
             ORDER BY updated_at DESC
             LIMIT ?2",
        )?;
        let rows =
            stmt.query_map(params![namespace.like_pattern(), limit as i64], row_to_plan_trace)?;
        for row in rows {
            plans.push(row.map_err(DbError::Sqlite)?);
        }
    }
    Ok(plans)
}

fn add_plan_checkpoint_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
    checkpoint_key: &str,
    title: &str,
    order_index: i32,
    expected_by: Option<&str>,
    metadata: Option<serde_json::Value>,
) -> Result<PlanCheckpoint, DbError> {
    let key = checkpoint_key.trim();
    let title = title.trim();
    if key.is_empty() || title.is_empty() {
        return Err(DbError::Other(
            "checkpoint_key and title are required".to_string(),
        ));
    }
    let exists: bool = conn
        .prepare_cached(
            "SELECT 1 FROM plan_traces
             WHERE namespace = ?1 AND id = ?2",
        )?
        .exists(params![namespace.as_str(), plan_id])?;
    if !exists {
        return Err(DbError::Other(format!(
            "plan {plan_id} does not exist in namespace {}",
            namespace.as_str()
        )));
    }

    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "INSERT INTO plan_checkpoints
            (id, namespace, plan_id, checkpoint_key, title, order_index, status, expected_by, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9, ?9)",
        params![
            id,
            namespace.as_str(),
            plan_id,
            key,
            title,
            order_index.max(0),
            expected_by,
            metadata_json,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan_checkpoint",
        &id,
        "create",
        None,
        None,
        Some(serde_json::json!({
            "plan_id": plan_id,
            "checkpoint_key": key,
            "title": title
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, plan_id, checkpoint_key, title, order_index, status, expected_by, completed_at, evidence, metadata, created_at, updated_at
         FROM plan_checkpoints WHERE id = ?1",
    )?;
    let checkpoint = stmt
        .query_row(params![id], row_to_plan_checkpoint)
        .map_err(DbError::Sqlite)?;
    Ok(checkpoint)
}

fn list_plan_checkpoints_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
) -> Result<Vec<PlanCheckpoint>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, plan_id, checkpoint_key, title, order_index, status, expected_by, completed_at, evidence, metadata, created_at, updated_at
         FROM plan_checkpoints
         WHERE namespace LIKE ?1 AND plan_id = ?2
         ORDER BY order_index ASC, created_at ASC",
    )?;
    let rows = stmt.query_map(params![namespace.like_pattern(), plan_id], row_to_plan_checkpoint)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(DbError::Sqlite)?);
    }
    Ok(out)
}

fn update_plan_checkpoint_status_sync(
    conn: &Connection,
    namespace: &Namespace,
    checkpoint_id: &str,
    status: &str,
    evidence: Option<&str>,
    metadata: Option<serde_json::Value>,
) -> Result<Option<PlanCheckpoint>, DbError> {
    let normalized = normalize_checkpoint_status(status);
    let now = Utc::now().to_rfc3339();
    let completed_at = if normalized == "completed" || normalized == "failed" {
        Some(now.clone())
    } else {
        None
    };
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    let rows = conn.execute(
        "UPDATE plan_checkpoints
         SET status = ?1,
             evidence = COALESCE(?2, evidence),
             metadata = COALESCE(?3, metadata),
             completed_at = CASE WHEN ?4 IS NULL THEN completed_at ELSE ?4 END,
             updated_at = ?5
         WHERE namespace = ?6 AND id = ?7",
        params![
            normalized,
            evidence,
            metadata_json,
            completed_at,
            now,
            namespace.as_str(),
            checkpoint_id,
        ],
    )?;
    if rows == 0 {
        return Ok(None);
    }
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan_checkpoint",
        checkpoint_id,
        "status_update",
        None,
        None,
        Some(serde_json::json!({
            "status": normalized,
            "evidence": evidence
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, plan_id, checkpoint_key, title, order_index, status, expected_by, completed_at, evidence, metadata, created_at, updated_at
         FROM plan_checkpoints WHERE id = ?1",
    )?;
    let checkpoint = stmt
        .query_row(params![checkpoint_id], row_to_plan_checkpoint)
        .optional()?;
    Ok(checkpoint)
}

fn set_plan_outcome_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
    status: &str,
    outcome: &str,
    outcome_confidence: Option<f64>,
    metadata: Option<serde_json::Value>,
) -> Result<Option<PlanTrace>, DbError> {
    let normalized = normalize_plan_status(status);
    if outcome.trim().is_empty() {
        return Err(DbError::Other("outcome cannot be empty".to_string()));
    }
    let now = Utc::now().to_rfc3339();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    let finished_at = if normalized == "completed" || normalized == "failed" || normalized == "abandoned" {
        Some(now.clone())
    } else {
        None
    };
    let rows = conn.execute(
        "UPDATE plan_traces
         SET status = ?1,
             outcome = ?2,
             outcome_confidence = ?3,
             metadata = COALESCE(?4, metadata),
             finished_at = CASE WHEN ?5 IS NULL THEN finished_at ELSE ?5 END,
             updated_at = ?6
         WHERE namespace = ?7 AND id = ?8",
        params![
            normalized,
            outcome.trim(),
            outcome_confidence.map(|v| v.clamp(0.0, 1.0)),
            metadata_json,
            finished_at,
            now,
            namespace.as_str(),
            plan_id,
        ],
    )?;
    if rows == 0 {
        return Ok(None);
    }
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan",
        plan_id,
        "set_outcome",
        None,
        None,
        Some(serde_json::json!({
            "status": normalized,
            "outcome": outcome.trim(),
            "outcome_confidence": outcome_confidence,
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, goal, status, priority, due_at, metadata, outcome, outcome_confidence, finished_at, created_at, updated_at
         FROM plan_traces WHERE id = ?1",
    )?;
    let plan = stmt.query_row(params![plan_id], row_to_plan_trace).optional()?;
    Ok(plan)
}

fn add_plan_recovery_branch_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
    source_checkpoint_id: Option<&str>,
    branch_label: &str,
    trigger_reason: &str,
    branch_plan: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
) -> Result<PlanRecoveryBranch, DbError> {
    if branch_label.trim().is_empty() || trigger_reason.trim().is_empty() {
        return Err(DbError::Other(
            "branch_label and trigger_reason are required".to_string(),
        ));
    }
    let plan_exists: bool = conn
        .prepare_cached(
            "SELECT 1 FROM plan_traces
             WHERE namespace = ?1 AND id = ?2",
        )?
        .exists(params![namespace.as_str(), plan_id])?;
    if !plan_exists {
        return Err(DbError::Other(format!(
            "plan {plan_id} does not exist in namespace {}",
            namespace.as_str()
        )));
    }
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let branch_plan_json = branch_plan
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "INSERT INTO plan_recovery_branches
            (id, namespace, plan_id, source_checkpoint_id, branch_label, trigger_reason, status, branch_plan, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?8, ?9, ?9)",
        params![
            id,
            namespace.as_str(),
            plan_id,
            source_checkpoint_id,
            branch_label.trim(),
            trigger_reason.trim(),
            branch_plan_json,
            metadata_json,
            now,
        ],
    )?;
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan_recovery_branch",
        &id,
        "create",
        None,
        None,
        Some(serde_json::json!({
            "plan_id": plan_id,
            "source_checkpoint_id": source_checkpoint_id,
            "label": branch_label.trim(),
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, plan_id, source_checkpoint_id, branch_label, trigger_reason, status, branch_plan, metadata, resolution_notes, created_at, updated_at, resolved_at
         FROM plan_recovery_branches
         WHERE id = ?1",
    )?;
    let branch = stmt
        .query_row(params![id], row_to_plan_recovery_branch)
        .map_err(DbError::Sqlite)?;
    Ok(branch)
}

fn list_plan_recovery_branches_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
) -> Result<Vec<PlanRecoveryBranch>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, plan_id, source_checkpoint_id, branch_label, trigger_reason, status, branch_plan, metadata, resolution_notes, created_at, updated_at, resolved_at
         FROM plan_recovery_branches
         WHERE namespace LIKE ?1 AND plan_id = ?2
         ORDER BY created_at DESC",
    )?;
    let rows =
        stmt.query_map(params![namespace.like_pattern(), plan_id], row_to_plan_recovery_branch)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(DbError::Sqlite)?);
    }
    Ok(out)
}

fn resolve_plan_recovery_branch_sync(
    conn: &Connection,
    namespace: &Namespace,
    branch_id: &str,
    status: &str,
    resolution_notes: Option<&str>,
) -> Result<Option<PlanRecoveryBranch>, DbError> {
    let normalized = normalize_branch_status(status);
    let now = Utc::now().to_rfc3339();
    let resolved_at = if normalized == "resolved" || normalized == "abandoned" {
        Some(now.clone())
    } else {
        None
    };
    let rows = conn.execute(
        "UPDATE plan_recovery_branches
         SET status = ?1,
             resolution_notes = COALESCE(?2, resolution_notes),
             resolved_at = CASE WHEN ?3 IS NULL THEN resolved_at ELSE ?3 END,
             updated_at = ?4
         WHERE namespace = ?5 AND id = ?6",
        params![
            normalized,
            resolution_notes,
            resolved_at,
            now,
            namespace.as_str(),
            branch_id,
        ],
    )?;
    if rows == 0 {
        return Ok(None);
    }
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan_recovery_branch",
        branch_id,
        "resolve",
        None,
        None,
        Some(serde_json::json!({
            "status": normalized,
            "resolution_notes": resolution_notes
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, plan_id, source_checkpoint_id, branch_label, trigger_reason, status, branch_plan, metadata, resolution_notes, created_at, updated_at, resolved_at
         FROM plan_recovery_branches
         WHERE id = ?1",
    )?;
    let branch = stmt
        .query_row(params![branch_id], row_to_plan_recovery_branch)
        .optional()?;
    Ok(branch)
}

fn bind_procedure_to_plan_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
    procedure_name: &str,
    binding_role: &str,
    confidence: f64,
    metadata: Option<serde_json::Value>,
) -> Result<PlanProcedureBinding, DbError> {
    if procedure_name.trim().is_empty() {
        return Err(DbError::Other("procedure_name cannot be empty".to_string()));
    }
    let plan_exists: bool = conn
        .prepare_cached(
            "SELECT 1 FROM plan_traces
             WHERE namespace = ?1 AND id = ?2",
        )?
        .exists(params![namespace.as_str(), plan_id])?;
    if !plan_exists {
        return Err(DbError::Other(format!(
            "plan {plan_id} does not exist in namespace {}",
            namespace.as_str()
        )));
    }
    let role = normalize_binding_role(binding_role);
    let conf = confidence.clamp(0.0, 1.0);
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let metadata_json = metadata
        .as_ref()
        .map(|m| serde_json::to_string(m).unwrap_or_default());
    conn.execute(
        "INSERT INTO plan_procedure_bindings
            (id, namespace, plan_id, procedure_name, binding_role, confidence, metadata, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
         ON CONFLICT(namespace, plan_id, procedure_name, binding_role)
         DO UPDATE SET
            confidence = excluded.confidence,
            metadata = COALESCE(excluded.metadata, plan_procedure_bindings.metadata),
            updated_at = excluded.updated_at",
        params![
            id,
            namespace.as_str(),
            plan_id,
            procedure_name.trim(),
            role,
            conf,
            metadata_json,
            now,
        ],
    )?;
    insert_audit_event_sync(
        conn,
        namespace.as_str(),
        "plan_procedure_binding",
        plan_id,
        "bind_procedure",
        None,
        None,
        Some(serde_json::json!({
            "procedure_name": procedure_name.trim(),
            "binding_role": role,
            "confidence": conf
        })),
    )?;
    let mut stmt = conn.prepare_cached(
        "SELECT id, namespace, plan_id, procedure_name, binding_role, confidence, metadata, created_at, updated_at
         FROM plan_procedure_bindings
         WHERE namespace = ?1 AND plan_id = ?2 AND procedure_name = ?3 AND binding_role = ?4
         LIMIT 1",
    )?;
    let binding = stmt
        .query_row(
            params![namespace.as_str(), plan_id, procedure_name.trim(), role],
            row_to_plan_procedure_binding,
        )
        .map_err(DbError::Sqlite)?;
    Ok(binding)
}

fn list_plan_procedure_bindings_sync(
    conn: &Connection,
    namespace: &Namespace,
    plan_id: &str,
) -> Result<Vec<PlanProcedureBinding>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, plan_id, procedure_name, binding_role, confidence, metadata, created_at, updated_at
         FROM plan_procedure_bindings
         WHERE namespace LIKE ?1 AND plan_id = ?2
         ORDER BY updated_at DESC",
    )?;
    let rows =
        stmt.query_map(params![namespace.like_pattern(), plan_id], row_to_plan_procedure_binding)?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(DbError::Sqlite)?);
    }
    Ok(out)
}

fn namespace_contains_str(namespace: &Namespace, memory_namespace: &str) -> bool {
    memory_namespace == namespace.as_str()
        || memory_namespace.starts_with(&format!("{}/", namespace.as_str()))
}

fn normalize_relation_type(input: &str) -> &'static str {
    match input.trim().to_ascii_lowercase().as_str() {
        "causal" | "cause" | "causes" | "caused" => "causal",
        "correlational" | "correlation" | "associated" | "association" => "correlational",
        _ => "correlational",
    }
}

fn upsert_causal_edge_sync(
    conn: &Connection,
    namespace: &Namespace,
    source_memory_id: &str,
    target_memory_id: &str,
    relation_type: &str,
    confidence: f64,
    evidence: Option<&str>,
) -> Result<(), DbError> {
    if source_memory_id == target_memory_id {
        return Ok(());
    }
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let rel = normalize_relation_type(relation_type);
    let conf = confidence.clamp(0.0, 1.0);

    conn.execute(
        "INSERT INTO causal_edges
            (id, namespace, source_memory_id, target_memory_id, relation_type, confidence, evidence, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(namespace, source_memory_id, target_memory_id, relation_type)
         DO UPDATE SET
            confidence = excluded.confidence,
            evidence = COALESCE(excluded.evidence, causal_edges.evidence),
            updated_at = excluded.updated_at",
        params![
            id,
            namespace.as_str(),
            source_memory_id,
            target_memory_id,
            rel,
            conf,
            evidence,
            now,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;

    Ok(())
}

fn find_memories_by_entity_ids_sync(
    conn: &Connection,
    namespace: &Namespace,
    entity_ids: &[String],
    limit: usize,
) -> Result<Vec<String>, DbError> {
    if entity_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders: Vec<String> = entity_ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 2)).collect();
    let sql = format!(
        "SELECT DISTINCT target_memory_id FROM causal_edges
         WHERE namespace LIKE ?1 AND source_memory_id IN ({})
         ORDER BY confidence DESC
         LIMIT {}",
        placeholders.join(","),
        limit
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params_vec.push(Box::new(namespace.like_pattern()));
    for eid in entity_ids {
        params_vec.push(Box::new(eid.clone()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        row.get::<_, String>(0)
    })?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row.map_err(DbError::Sqlite)?);
    }
    Ok(result)
}

fn list_causal_edges_sync(
    conn: &Connection,
    namespace: &Namespace,
    limit: usize,
) -> Result<Vec<CausalEdge>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, source_memory_id, target_memory_id, relation_type, confidence, evidence, created_at, updated_at
         FROM causal_edges
         WHERE namespace LIKE ?1
         ORDER BY confidence DESC, updated_at DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![namespace.like_pattern(), limit as i64], |row| {
        Ok(CausalEdge {
            id: row.get(0)?,
            namespace: row.get(1)?,
            source_memory_id: row.get(2)?,
            target_memory_id: row.get(3)?,
            relation_type: row.get(4)?,
            confidence: row.get(5)?,
            evidence: row.get(6)?,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn normalize_transition_type(input: &str) -> &'static str {
    match input.trim().to_ascii_lowercase().as_str() {
        "temporal" | "time" | "before_after" => "temporal",
        "state_change" | "state" => "state_change",
        "dependency" | "depends_on" => "dependency",
        "supersession" | "superseded_by" => "supersession",
        _ => "temporal",
    }
}

fn upsert_state_transition_sync(
    conn: &Connection,
    namespace: &Namespace,
    from_memory_id: &str,
    to_memory_id: &str,
    transition_type: &str,
    confidence: f64,
    evidence: Option<&str>,
) -> Result<(), DbError> {
    if from_memory_id == to_memory_id {
        return Ok(());
    }
    let now = Utc::now().to_rfc3339();
    let id = ulid::Ulid::new().to_string();
    let kind = normalize_transition_type(transition_type);
    conn.execute(
        "INSERT INTO state_transitions
            (id, namespace, from_memory_id, to_memory_id, transition_type, confidence, evidence, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
         ON CONFLICT(namespace, from_memory_id, to_memory_id, transition_type)
         DO UPDATE SET
            confidence = excluded.confidence,
            evidence = COALESCE(excluded.evidence, state_transitions.evidence),
            updated_at = excluded.updated_at",
        params![
            id,
            namespace.as_str(),
            from_memory_id,
            to_memory_id,
            kind,
            confidence.clamp(0.0, 1.0),
            evidence,
            now,
        ],
    )
    .map_err(DbError::Sqlite)?;
    Ok(())
}

fn list_state_transitions_sync(
    conn: &Connection,
    namespace: &Namespace,
    limit: usize,
) -> Result<Vec<StateTransitionEdge>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, namespace, from_memory_id, to_memory_id, transition_type, confidence, evidence, created_at, updated_at
         FROM state_transitions
         WHERE namespace LIKE ?1
         ORDER BY confidence DESC, updated_at DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![namespace.like_pattern(), limit as i64], |row| {
        Ok(StateTransitionEdge {
            id: row.get(0)?,
            namespace: row.get(1)?,
            from_memory_id: row.get(2)?,
            to_memory_id: row.get(3)?,
            transition_type: row.get(4)?,
            confidence: row.get::<_, f64>(5)?.clamp(0.0, 1.0),
            evidence: row.get(6)?,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
        })
    })?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn simulate_counterfactual_sync(
    conn: &Connection,
    namespace: &Namespace,
    intervention: &str,
    seed_memory_ids: &[String],
    max_hops: usize,
    top_k: usize,
    include_correlational: bool,
    include_transitions: bool,
) -> Result<CounterfactualSimulation, DbError> {
    let max_hops = max_hops.clamp(1, 4);
    let top_k = top_k.clamp(1, 20);
    let intervention = intervention.trim();
    if intervention.is_empty() {
        return Err(DbError::Other(
            "counterfactual intervention cannot be empty".to_string(),
        ));
    }
    if seed_memory_ids.is_empty() {
        return Ok(CounterfactualSimulation {
            intervention: intervention.to_string(),
            seed_memory_ids: Vec::new(),
            hypotheses: Vec::new(),
            considered_nodes: 0,
            generated_at: Utc::now().to_rfc3339(),
        });
    }

    #[derive(Clone)]
    struct FrontierNode {
        id: String,
        score: f64,
        hop: usize,
        evidence_ids: Vec<String>,
        source_kinds: Vec<String>,
    }

    let mut queue = VecDeque::new();
    let mut best_scores: HashMap<String, f64> = HashMap::new();
    let mut evidences: HashMap<String, Vec<String>> = HashMap::new();
    let mut kinds: HashMap<String, Vec<String>> = HashMap::new();
    let mut visited_pairs: HashSet<(String, usize)> = HashSet::new();

    for seed in seed_memory_ids {
        queue.push_back(FrontierNode {
            id: seed.clone(),
            score: 1.0,
            hop: 0,
            evidence_ids: vec![seed.clone()],
            source_kinds: vec!["seed".to_string()],
        });
    }

    while let Some(node) = queue.pop_front() {
        if node.hop >= max_hops {
            continue;
        }
        if !visited_pairs.insert((node.id.clone(), node.hop)) {
            continue;
        }
        let next_hop = node.hop + 1;
        let hop_decay = 0.82f64.powi(next_hop as i32);

        let mut causal_stmt = conn.prepare_cached(
            "SELECT source_memory_id, target_memory_id, relation_type, confidence
             FROM causal_edges
             WHERE namespace = ?1
               AND (source_memory_id = ?2 OR target_memory_id = ?2)",
        )?;
        let causal_rows = causal_stmt.query_map(params![namespace.as_str(), node.id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?.clamp(0.0, 1.0),
            ))
        })?;
        for row in causal_rows {
            let (source, target, relation_type, confidence) = row?;
            if relation_type == "correlational" && !include_correlational {
                continue;
            }
            let other = if source == node.id { target } else { source };
            if other == node.id {
                continue;
            }
            let type_weight = match relation_type.as_str() {
                "causal" => 1.0,
                "correlational" => 0.6,
                _ => 0.5,
            };
            let propagated = (node.score * confidence * type_weight * hop_decay).clamp(0.0, 1.0);
            if propagated <= 0.01 {
                continue;
            }
            let current = best_scores.entry(other.clone()).or_insert(0.0);
            if propagated > *current {
                *current = propagated;
                let mut ev = node.evidence_ids.clone();
                ev.push(other.clone());
                evidences.insert(other.clone(), ev);
                let mut sk = node.source_kinds.clone();
                sk.push(relation_type.clone());
                kinds.insert(other.clone(), sk);
                let next_evidence = evidences
                    .get(&other)
                    .cloned()
                    .unwrap_or_else(|| node.evidence_ids.clone());
                let next_kinds = kinds
                    .get(&other)
                    .cloned()
                    .unwrap_or_else(|| node.source_kinds.clone());
                queue.push_back(FrontierNode {
                    id: other,
                    score: propagated,
                    hop: next_hop,
                    evidence_ids: next_evidence,
                    source_kinds: next_kinds,
                });
            }
        }

        if include_transitions {
            let mut tr_stmt = conn.prepare_cached(
                "SELECT from_memory_id, to_memory_id, transition_type, confidence
                 FROM state_transitions
                 WHERE namespace = ?1
                   AND (from_memory_id = ?2 OR to_memory_id = ?2)",
            )?;
            let tr_rows = tr_stmt.query_map(params![namespace.as_str(), node.id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?.clamp(0.0, 1.0),
                ))
            })?;
            for row in tr_rows {
                let (from_id, to_id, transition_type, confidence) = row?;
                let other = if from_id == node.id { to_id } else { from_id };
                if other == node.id {
                    continue;
                }
                let type_weight = match transition_type.as_str() {
                    "temporal" => 0.75,
                    "state_change" => 0.85,
                    "dependency" => 0.9,
                    "supersession" => 0.65,
                    _ => 0.7,
                };
                let propagated = (node.score * confidence * type_weight * hop_decay).clamp(0.0, 1.0);
                if propagated <= 0.01 {
                    continue;
                }
                let current = best_scores.entry(other.clone()).or_insert(0.0);
                if propagated > *current {
                    *current = propagated;
                    let mut ev = node.evidence_ids.clone();
                    ev.push(other.clone());
                    evidences.insert(other.clone(), ev);
                    let mut sk = node.source_kinds.clone();
                    sk.push(format!("transition:{transition_type}"));
                    kinds.insert(other.clone(), sk);
                    let next_evidence = evidences
                        .get(&other)
                        .cloned()
                        .unwrap_or_else(|| node.evidence_ids.clone());
                    let next_kinds = kinds
                        .get(&other)
                        .cloned()
                        .unwrap_or_else(|| node.source_kinds.clone());
                    queue.push_back(FrontierNode {
                        id: other,
                        score: propagated,
                        hop: next_hop,
                        evidence_ids: next_evidence,
                        source_kinds: next_kinds,
                    });
                }
            }
        }
    }

    for seed in seed_memory_ids {
        best_scores.remove(seed);
        evidences.remove(seed);
        kinds.remove(seed);
    }

    let mut scored_ids: Vec<(String, f64)> = best_scores.into_iter().collect();
    scored_ids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored_ids.truncate(top_k);

    let mut mem_stmt = conn.prepare_cached(
        "SELECT id, content
         FROM memories
         WHERE id = ?1 AND namespace LIKE ?2 AND status = 'active'
         LIMIT 1",
    )?;
    let mut hypotheses = Vec::new();
    for (memory_id, score) in scored_ids {
        let maybe_row = mem_stmt
            .query_row(params![memory_id, namespace.like_pattern()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .optional()?;
        let Some((id, content)) = maybe_row else {
            continue;
        };
        let mid = score.clamp(0.0, 1.0);
        let low = (mid - 0.14).clamp(0.0, 1.0);
        let high = (mid + 0.11).clamp(0.0, 1.0);
        let statement = format!(
            "If intervention '{}' holds, this is likely affected: {}",
            intervention, content
        );
        hypotheses.push(CounterfactualHypothesis {
            memory_id: id.clone(),
            statement,
            confidence_low: low,
            confidence_mid: mid,
            confidence_high: high,
            evidence_ids: evidences.remove(&id).unwrap_or_default(),
            source_kinds: kinds.remove(&id).unwrap_or_default(),
        });
    }

    Ok(CounterfactualSimulation {
        intervention: intervention.to_string(),
        seed_memory_ids: seed_memory_ids.to_vec(),
        considered_nodes: hypotheses.len(),
        hypotheses,
        generated_at: Utc::now().to_rfc3339(),
    })
}

/// Fetch updated_at, access_count, importance, and tier for scoring.
fn get_content_length(conn: &Connection, id: &str) -> Option<usize> {
    conn.prepare_cached("SELECT content FROM memories WHERE id = ?1")
        .ok()?
        .query_row(params![id], |row| {
            let content: String = row.get(0)?;
            Ok(content.split_whitespace().count())
        })
        .ok()
}

/// Scoring metadata: (updated_at, access_count, importance, tier, event_date)
fn get_scoring_metadata(
    conn: &Connection,
    id: &str,
) -> Result<Option<(DateTime<Utc>, u64, i32, i32, Option<DateTime<Utc>>, String, f64)>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT updated_at, access_count, importance,
                COALESCE(CAST(json_extract(metadata, '$.tier') AS INTEGER), 1),
                event_date, category, confidence, source
         FROM memories WHERE id = ?1",
    )?;
    let result = stmt
        .query_row(params![id], |row| {
            let ts_str: String = row.get(0)?;
            let ac: u64 = row.get(1)?;
            let imp: i32 = row.get(2)?;
            let tier: i32 = row.get(3)?;
            let event_date_str: Option<String> = row.get(4)?;
            let cat_str: String = row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "general".to_string());
            let conf: f64 = row.get::<_, Option<f64>>(6)?.unwrap_or(1.0);
            Ok((ts_str, ac, imp, tier, event_date_str, cat_str, conf))
        })
        .optional()?;

    Ok(result.and_then(|(ts_str, ac, imp, tier, ed_str, cat_str, conf)| {
        DateTime::parse_from_rfc3339(&ts_str)
            .ok()
            .map(|dt| {
                // Parse event_date (ISO 8601 date like "2023-08-15") into DateTime
                let event_date = ed_str.and_then(|s| {
                    // Try full RFC3339 first, then date-only
                    DateTime::parse_from_rfc3339(&s)
                        .map(|d| d.with_timezone(&Utc))
                        .ok()
                        .or_else(|| {
                            chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                                .ok()
                                .map(|d| d.and_hms_opt(12, 0, 0).unwrap().and_utc())
                        })
                });
                (dt.with_timezone(&Utc), ac, imp, tier, event_date, cat_str, conf)
            })
    }))
}

fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let status_str: String = row.get(4)?;
    let created_str: String = row.get(5)?;
    let updated_str: String = row.get(6)?;
    let accessed_str: String = row.get(7)?;
    let metadata_str: Option<String> = row.get(3)?;
    let tags_str: String = row.get::<_, Option<String>>(10)?.unwrap_or_else(|| "[]".to_string());
    let memory_type: String = row.get::<_, Option<String>>(11)?.unwrap_or_else(|| "fact".to_string());
    let importance: i32 = row.get::<_, Option<i32>>(12)?.unwrap_or(5);

    let episode_id: Option<String> = row.get::<_, Option<String>>(13).unwrap_or(None);
    let event_date: Option<String> = row.get::<_, Option<String>>(14).unwrap_or(None);
    let category_str: String = row.get::<_, Option<String>>(15)?.unwrap_or_else(|| "general".to_string());
    let confidence: f64 = row.get::<_, Option<f64>>(16)?.unwrap_or(1.0);
    let source: String = row.get::<_, Option<String>>(17)?.unwrap_or_else(|| "user_stated".to_string());

    Ok(Memory {
        id: row.get(0)?,
        namespace: row.get(1)?,
        content: row.get(2)?,
        metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
        tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        memory_type,
        status: MemoryStatus::from_str(&status_str).unwrap_or(MemoryStatus::Active),
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .unwrap_or_default()
            .with_timezone(&Utc),
        updated_at: DateTime::parse_from_rfc3339(&updated_str)
            .unwrap_or_default()
            .with_timezone(&Utc),
        accessed_at: DateTime::parse_from_rfc3339(&accessed_str)
            .unwrap_or_default()
            .with_timezone(&Utc),
        access_count: row.get(8)?,
        importance,
        episode_id,
        event_date,
        hash: row.get(9)?,
        category: category_str,
        confidence,
        source,
    })
}

/// Extension trait for rusqlite::Result<T> -> Option<T>
trait OptionalExt<T> {
    fn optional(self) -> rusqlite::Result<Option<T>>;
}

impl<T> OptionalExt<T> for rusqlite::Result<T> {
    fn optional(self) -> rusqlite::Result<Option<T>> {
        match self {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

fn get_raw_embeddings_sync(
    conn: &Connection,
    namespace: &Namespace,
    limit: usize,
) -> Result<Vec<RawEmbeddingData>, DbError> {
    let ns_pattern = namespace.like_pattern();
    let mut stmt = conn
        .prepare(
            "SELECT id, content, memory_type, embedding
             FROM memories
             WHERE namespace LIKE ?1 AND status = 'active'
               AND embedding IS NOT NULL AND length(embedding) > 0
             ORDER BY created_at DESC
             LIMIT ?2",
        )
        .map_err(DbError::Sqlite)?;

    let rows = stmt
        .query_map(params![ns_pattern, limit as i64], |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let memory_type: String = row.get(2)?;
            let blob: Option<Vec<u8>> = row.get(3)?;
            Ok((id, content, memory_type, blob))
        })
        .map_err(DbError::Sqlite)?;

    let mut result = Vec::new();
    for row in rows {
        let (id, content, memory_type, blob) = row.map_err(DbError::Sqlite)?;
        let Some(blob) = blob else { continue };
        if blob.len() % 4 != 0 || blob.is_empty() { continue }
        let embedding: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        result.push(RawEmbeddingData { id, content, memory_type, embedding });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::DbPool;

    /// Create a test memory with the given content in a namespace, insert it, and return its ID.
    fn insert_test_memory(conn: &Connection, ns: &str, content: &str, embedding: &[f32]) -> String {
        let id = ulid::Ulid::new().to_string().to_lowercase();
        let now = Utc::now();
        let mem = Memory {
            id: id.clone(),
            namespace: ns.to_string(),
            content: content.to_string(),
            metadata: None,
            tags: vec![],
            memory_type: "raw".to_string(),
            status: MemoryStatus::Active,
            created_at: now,
            updated_at: now,
            accessed_at: now,
            access_count: 0,
            importance: 5,
            hash: content_hash(content),
            episode_id: None,
            event_date: None,
            category: "general".to_string(),
            confidence: 1.0,
            source: "user_stated".to_string(),
        };
        insert_memory_sync(conn, &mem, embedding).unwrap();
        id
    }

    /// Generate a slightly varied embedding based on a seed, so different memories
    /// have different but similar embeddings (simulating real dense namespace behavior).
    fn make_embedding(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let base = (i as f32) / (dim as f32);
                let noise = ((seed * 7 + i * 13) % 100) as f32 / 10000.0;
                base + noise
            })
            .collect()
    }

    #[test]
    fn search_returns_vector_results_in_dense_namespace() {
        // Regression test: vector search must return results even when many memories
        // have similar embeddings (tight score spread). Previously, a min_score_spread
        // filter was clearing all vector results in dense namespaces.
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;
            let ns = "test/dense";

            // Insert 50 memories with similar embeddings (simulates dense namespace)
            for i in 0..50 {
                let emb = make_embedding(dim, i);
                insert_test_memory(&conn, ns, &format!("Memory about topic {i}"), &emb);
            }

            // Search with a query embedding similar to the inserted ones
            let query_emb = make_embedding(dim, 25);
            let namespace = Namespace::parse(ns).unwrap();
            let config = ScorerConfig {
                min_score_spread: 0.0, // disabled (current default)
                ..Default::default()
            };

            let results = search_sync(
                &conn, &query_emb, "Memory about topic", &namespace,
                &SearchMode::Vector, 10, &config,
            ).unwrap();

            assert!(!results.is_empty(), "Vector search should return results in dense namespace");
            assert!(results.len() >= 5, "Expected at least 5 results, got {}", results.len());

            // All results should have vector scores
            for r in &results {
                assert!(r.vector_score.is_some(), "Vector-mode results should have vector_score");
                assert!(r.score > 0.0, "Score should be positive");
            }
        });
    }

    #[test]
    fn insert_many_writes_all_rows_in_one_call() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/insert-many").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let mut batch = Vec::new();
            for i in 0..8 {
                let mut m = Memory::new(
                    ns.as_str().to_string(),
                    format!("Batch memory item {i}"),
                    Some(serde_json::json!({"tier": 1})),
                    vec!["batch".to_string()],
                    Some("raw".to_string()),
                );
                m.importance = 5;
                batch.push((m, make_embedding(dim, i)));
            }
            store.insert_many(&batch).await.unwrap();

            let (memories, total) = store.list(&ns, Some("active"), None, None, 0, 50).await.unwrap();
            assert_eq!(total, 8);
            assert_eq!(memories.len(), 8);
        });
    }

    #[test]
    fn search_hybrid_includes_vector_scores() {
        // Regression test: hybrid search must include vector component scores,
        // not just keyword scores.
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;
            let ns = "test/hybrid";

            // Insert memories with content that matches both vector and keyword
            for i in 0..20 {
                let emb = make_embedding(dim, i);
                insert_test_memory(&conn, ns, &format!("Caroline loves camping trip number {i}"), &emb);
            }

            let query_emb = make_embedding(dim, 10);
            let namespace = Namespace::parse(ns).unwrap();
            let config = ScorerConfig::default();

            let results = search_sync(
                &conn, &query_emb, "camping", &namespace,
                &SearchMode::Hybrid, 10, &config,
            ).unwrap();

            assert!(!results.is_empty(), "Hybrid search should return results");

            // At least some results should have keyword scores (from FTS "camping" match)
            let has_kw = results.iter().any(|r| r.keyword_score.is_some());
            assert!(has_kw, "Hybrid results should include keyword matches for 'camping'");
        });
    }

    #[test]
    fn search_spread_filter_only_blocks_noise_queries() {
        // When min_score_spread > 0, it should only filter truly noisy queries
        // (where raw candidate pool has tight spread), not legitimate queries
        // in dense namespaces.
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;
            let ns = "test/spread";

            // Insert diverse memories (different embeddings)
            for i in 0..30 {
                let emb = make_embedding(dim, i * 50); // larger gaps between embeddings
                insert_test_memory(&conn, ns, &format!("Diverse content {i}"), &emb);
            }

            let query_emb = make_embedding(dim, 0);
            let namespace = Namespace::parse(ns).unwrap();

            // With spread filter enabled but checking raw results (not namespace-filtered)
            let config_with_spread = ScorerConfig {
                min_score_spread: 0.055,
                ..Default::default()
            };

            let results = search_sync(
                &conn, &query_emb, "content", &namespace,
                &SearchMode::Vector, 10, &config_with_spread,
            ).unwrap();

            // With diverse enough embeddings, the raw candidate pool should have
            // enough spread to pass the filter
            // (This test documents the behavior - if it fails, the spread threshold
            // needs adjustment for the embedding model being used)
            if results.is_empty() {
                // If empty, verify it's because of spread filter by testing without it
                let config_no_spread = ScorerConfig {
                    min_score_spread: 0.0,
                    ..Default::default()
                };
                let results_no_spread = search_sync(
                    &conn, &query_emb, "content", &namespace,
                    &SearchMode::Vector, 10, &config_no_spread,
                ).unwrap();
                assert!(results_no_spread.is_empty(),
                    "If spread filter cleared results, disabling it should recover them. \
                     This means the spread threshold is too aggressive for this embedding model.");
            }
        });
    }

    #[test]
    fn search_keyword_only_works() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let dim = DbPool::DEFAULT_DIMENSION;
            let ns = "test/keyword";

            for i in 0..5 {
                let emb = make_embedding(dim, i);
                insert_test_memory(&conn, ns, &format!("Caroline went camping in June {i}"), &emb);
            }

            let query_emb = make_embedding(dim, 99); // unrelated embedding
            let namespace = Namespace::parse(ns).unwrap();
            let config = ScorerConfig::default();

            let results = search_sync(
                &conn, &query_emb, "camping", &namespace,
                &SearchMode::Keyword, 5, &config,
            ).unwrap();

            assert!(!results.is_empty(), "Keyword search should find 'camping'");
            for r in &results {
                assert!(r.keyword_score.is_some());
                assert!(r.vector_score.is_none());
            }
        });
    }

    #[test]
    fn calibration_recompute_and_metrics_work() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/calibration").unwrap();

            for i in 0..40 {
                let pred = 0.05 + (i as f64 * 0.9 / 39.0);
                let outcome = (0.5 * pred + 0.1).clamp(0.0, 1.0);
                store
                    .record_calibration_observation(
                        &ns,
                        PredictionKind::RetrievalRelevance,
                        None,
                        pred,
                        Some(outcome),
                        None,
                    )
                    .await
                    .unwrap();
            }

            store.recompute_calibration_models().await.unwrap();

            let calibrated = store
                .calibrate_confidence(&ns, PredictionKind::RetrievalRelevance, 0.9)
                .await
                .unwrap();
            assert!(calibrated < 0.9, "expected downward calibration, got {calibrated}");
            assert!(calibrated > 0.5, "calibration should remain in plausible range, got {calibrated}");

            let metrics = store.calibration_metrics(&ns).await.unwrap();
            assert!(!metrics.models.is_empty(), "expected fitted model rows");
            assert!(!metrics.bins.is_empty(), "expected calibration bin rows");
        });
    }

    #[test]
    fn search_applies_retrieval_calibration_model() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let ns = "test/search-cal";
            let namespace = Namespace::parse(ns).unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let emb = make_embedding(dim, 7);
            insert_test_memory(&conn, ns, "Very specific retrievable memory", &emb);

            // Preload a strong calibration model that compresses relevance scores.
            conn.execute(
                "INSERT INTO calibration_models
                    (namespace, prediction_kind, samples, avg_prediction, avg_outcome, mse, slope, intercept, updated_at)
                 VALUES (?1, 'retrieval_relevance', 200, 0.7, 0.35, 0.02, 0.5, 0.0, ?2)",
                params![ns, Utc::now().to_rfc3339()],
            )
            .unwrap();

            let mut cfg = ScorerConfig::default();
            cfg.temporal_weight = 0.0;
            cfg.min_vector_similarity = 0.0;
            cfg.min_score_spread = 0.0;

            let results = search_sync(
                &conn,
                &emb,
                "specific retrievable",
                &namespace,
                &SearchMode::Vector,
                5,
                &cfg,
            )
            .unwrap();
            assert!(!results.is_empty());
            assert!(
                results[0].score <= 0.6,
                "calibrated top score should be compressed, got {}",
                results[0].score
            );
        });
    }

    #[test]
    fn correction_flow_updates_belief_and_ledger() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let dim = DbPool::DEFAULT_DIMENSION;
            let ns = Namespace::parse("test/correction").unwrap();
            let (old_id, new_id) = {
                let conn = pool.writer().await;
                let old_id = insert_test_memory(
                    &conn,
                    ns.as_str(),
                    "User lives in Portland",
                    &make_embedding(dim, 1),
                );
                let new_mem = Memory::new(
                    ns.as_str().to_string(),
                    "User moved to Seattle".to_string(),
                    Some(serde_json::json!({
                        "tier": 2,
                        "confidence": 1.0,
                        "correction": { "source": "test" }
                    })),
                    vec!["location".to_string()],
                    Some("event".to_string()),
                );
                let new_id = new_mem.id.clone();
                insert_memory_sync(&conn, &new_mem, &make_embedding(dim, 2)).unwrap();
                mark_superseded_sync(&conn, &old_id, &new_id).unwrap();
                (old_id, new_id)
            };

            let event = store
                .record_correction_event(
                    &ns,
                    &old_id,
                    &new_id,
                    Some("user corrected location"),
                    Some(serde_json::json!({"actor": "user"})),
                )
                .await
                .unwrap();
            assert_eq!(event.target_memory_id, old_id);
            assert_eq!(event.new_memory_id, new_id);

            store
                .record_contradiction_resolution(
                    &ns,
                    &old_id,
                    &new_id,
                    "user_correction_supersede",
                    Some(serde_json::json!({"source": "test"})),
                )
                .await
                .unwrap();

            store
                .record_calibration_observation(
                    &ns,
                    PredictionKind::Extraction,
                    Some(&old_id),
                    0.9,
                    Some(0.0),
                    Some(serde_json::json!({"source": "correction"})),
                )
                .await
                .unwrap();
            store
                .record_calibration_observation(
                    &ns,
                    PredictionKind::Extraction,
                    Some(&new_id),
                    1.0,
                    Some(1.0),
                    Some(serde_json::json!({"source": "correction"})),
                )
                .await
                .unwrap();
            store.recompute_calibration_models().await.unwrap();

            let old_active = store.get(&old_id).await.unwrap();
            let new_active = store.get(&new_id).await.unwrap();
            assert!(old_active.is_none(), "old belief should be superseded (inactive)");
            assert!(new_active.is_some(), "new corrected belief should be active");

            let conn = pool.writer().await;
            let correction_rows: u64 = conn
                .prepare("SELECT COUNT(*) FROM correction_events WHERE namespace = ?1")
                .unwrap()
                .query_row(params![ns.as_str()], |row| row.get(0))
                .unwrap();
            assert_eq!(correction_rows, 1);

            let contradiction_rows: u64 = conn
                .prepare("SELECT COUNT(*) FROM contradiction_ledger WHERE namespace = ?1")
                .unwrap()
                .query_row(params![ns.as_str()], |row| row.get(0))
                .unwrap();
            assert_eq!(contradiction_rows, 1);
        });
    }

    #[test]
    fn claim_patch_and_rollback_revision_flow() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/revision").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let emb = make_embedding(dim, 1);
            let id = insert_test_memory(&conn, ns.as_str(), "Original claim", &emb);
            drop(conn);

            let patch = MemoryPatch {
                content: Some("Patched claim".to_string()),
                tags: Some(vec!["patched".to_string()]),
                ..Default::default()
            };
            let patched = store
                .patch_memory(
                    &id,
                    &patch,
                    Some(&make_embedding(dim, 2)),
                    Some("test"),
                    Some("fix wording"),
                )
                .await
                .unwrap()
                .expect("expected patched memory");
            assert_eq!(patched.memory.content, "Patched claim");

            let revisions = store.list_claim_revisions(&ns, &id, 10).await.unwrap();
            assert!(
                revisions.iter().any(|r| r.operation == "patch"),
                "expected patch revision entry"
            );

            let rolled_back = store
                .rollback_memory_to_revision(&id, 1, Some("test"), Some("revert"))
                .await
                .unwrap()
                .expect("expected rollback result");
            assert_eq!(rolled_back.memory.content, "Original claim");

            let revisions_after = store.list_claim_revisions(&ns, &id, 10).await.unwrap();
            assert!(
                revisions_after.iter().any(|r| r.operation == "rollback"),
                "expected rollback revision entry"
            );
        });
    }

    #[test]
    fn merge_memories_supersedes_sources_and_records_audit() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/merge").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let id_a = insert_test_memory(&conn, ns.as_str(), "Claim A", &make_embedding(dim, 3));
            let id_b = insert_test_memory(&conn, ns.as_str(), "Claim B", &make_embedding(dim, 4));
            drop(conn);

            let mut merged = Memory::new(
                ns.as_str().to_string(),
                "Merged claim AB".to_string(),
                Some(serde_json::json!({"source": "test"})),
                vec!["merged".to_string()],
                Some("merged".to_string()),
            );
            merged.importance = 8;

            let result = store
                .merge_memories(
                    &ns,
                    &[id_a.clone(), id_b.clone()],
                    &merged,
                    &make_embedding(dim, 5),
                    Some("test"),
                    Some("merge duplicates"),
                )
                .await
                .unwrap();

            assert_eq!(result.superseded_source_ids.len(), 2);
            assert_eq!(result.merged_memory_id, merged.id);
            assert!(store.get(&id_a).await.unwrap().is_none());
            assert!(store.get(&id_b).await.unwrap().is_none());
            assert!(store.get(&merged.id).await.unwrap().is_some());

            let source_revisions = store.list_claim_revisions(&ns, &id_a, 10).await.unwrap();
            assert!(
                source_revisions.iter().any(|r| r.operation == "merge_source"),
                "source should include merge_source revision"
            );

            let merged_revisions = store.list_claim_revisions(&ns, &merged.id, 10).await.unwrap();
            assert!(
                merged_revisions.iter().any(|r| r.operation == "merge_created"),
                "merged memory should include merge_created revision"
            );

            let merge_audit = store
                .list_audit_events(&ns, Some("claim_merge"), Some(&merged.id), 10)
                .await
                .unwrap();
            assert!(
                !merge_audit.is_empty(),
                "merge should emit audit trail event"
            );
        });
    }

    #[test]
    fn procedure_revisions_and_audit_are_recorded() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/procedure").unwrap();
            let name = "memory_search";

            store
                .upsert_procedure_revision(
                    &ns,
                    name,
                    "create",
                    &serde_json::json!({"temperature": 0.2}),
                    Some("test"),
                    Some("initial"),
                )
                .await
                .unwrap();
            store
                .upsert_procedure_revision(
                    &ns,
                    name,
                    "patch",
                    &serde_json::json!({"temperature": 0.1}),
                    Some("test"),
                    Some("tighten"),
                )
                .await
                .unwrap();

            let revisions = store.list_procedure_revisions(&ns, name, 10).await.unwrap();
            assert_eq!(revisions.len(), 2);
            assert_eq!(revisions[0].revision_number, 2);
            assert_eq!(revisions[1].revision_number, 1);

            let audit = store
                .list_audit_events(&ns, Some("procedure"), Some(name), 10)
                .await
                .unwrap();
            assert_eq!(audit.len(), 2);
        });
    }

    #[test]
    fn identity_resolution_handles_aliases_and_multilingual_normalization() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/identity").unwrap();

            let entity = store
                .upsert_identity_entity(
                    &ns,
                    "Jose Alvarez",
                    Some("es"),
                    0.9,
                    Some(serde_json::json!({"role": "friend"})),
                )
                .await
                .unwrap();
            store
                .add_identity_alias(&ns, &entity.id, "José Álvarez", Some("es"), 0.95)
                .await
                .unwrap();
            store
                .add_identity_alias(&ns, &entity.id, "Pepe", Some("es"), 0.8)
                .await
                .unwrap();

            let resolved = store.resolve_identity(&ns, "José Alvarez", 5).await.unwrap();
            assert!(!resolved.candidates.is_empty());
            assert_eq!(resolved.candidates[0].entity_id, entity.id);
            assert!(
                resolved.best_confidence > 0.7,
                "expected strong confidence for accent-folded exact alias, got {}",
                resolved.best_confidence
            );

            let by_alias = store.resolve_identity(&ns, "Pepe", 5).await.unwrap();
            assert!(!by_alias.ambiguous);
            assert_eq!(by_alias.candidates[0].entity_id, entity.id);
        });
    }

    #[test]
    fn identity_resolution_marks_ambiguous_aliases() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/identity-ambiguity").unwrap();

            let alice = store
                .upsert_identity_entity(&ns, "Alice Johnson", Some("en"), 0.82, None)
                .await
                .unwrap();
            let alicia = store
                .upsert_identity_entity(&ns, "Alicia Jones", Some("en"), 0.8, None)
                .await
                .unwrap();
            store
                .add_identity_alias(&ns, &alice.id, "Ali", Some("en"), 0.9)
                .await
                .unwrap();
            store
                .add_identity_alias(&ns, &alicia.id, "Ali", Some("en"), 0.88)
                .await
                .unwrap();

            let resolution = store.resolve_identity(&ns, "Ali", 5).await.unwrap();
            assert!(resolution.ambiguous, "shared alias should be ambiguous");
            assert!(resolution.candidates.len() >= 2);
            assert!(
                resolution.candidates[0].ambiguity >= 0.4,
                "expected ambiguity to be reflected in candidate score metadata"
            );
        });
    }

    #[test]
    fn planning_trace_checkpoint_and_outcome_flow() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/planning").unwrap();

            let plan = store
                .create_plan_trace(
                    &ns,
                    "Ship embedded memory update",
                    8,
                    Some("2026-03-31"),
                    Some(serde_json::json!({"owner":"robotics"})),
                )
                .await
                .unwrap();
            assert_eq!(plan.status, "active");

            let cp1 = store
                .add_plan_checkpoint(
                    &ns,
                    &plan.id,
                    "design",
                    "Finalize design",
                    1,
                    None,
                    None,
                )
                .await
                .unwrap();
            let _cp2 = store
                .add_plan_checkpoint(
                    &ns,
                    &plan.id,
                    "implement",
                    "Implement and verify",
                    2,
                    None,
                    None,
                )
                .await
                .unwrap();

            let cp1_done = store
                .update_plan_checkpoint_status(
                    &ns,
                    &cp1.id,
                    "completed",
                    Some("design doc accepted"),
                    Some(serde_json::json!({"reviewers":2})),
                )
                .await
                .unwrap()
                .expect("checkpoint should exist");
            assert_eq!(cp1_done.status, "completed");
            assert!(cp1_done.completed_at.is_some());

            let completed = store
                .set_plan_outcome(
                    &ns,
                    &plan.id,
                    "completed",
                    "Delivered with low latency",
                    Some(0.91),
                    Some(serde_json::json!({"p95_ms": 28})),
                )
                .await
                .unwrap()
                .expect("plan should exist");
            assert_eq!(completed.status, "completed");
            assert_eq!(completed.outcome.as_deref(), Some("Delivered with low latency"));
            assert!(completed.finished_at.is_some());

            let checkpoints = store.list_plan_checkpoints(&ns, &plan.id).await.unwrap();
            assert_eq!(checkpoints.len(), 2);
            assert_eq!(checkpoints[0].checkpoint_key, "design");

            let completed_plans = store.list_plan_traces(&ns, Some("completed"), 10).await.unwrap();
            assert_eq!(completed_plans.len(), 1);
            assert_eq!(completed_plans[0].id, plan.id);
        });
    }

    #[test]
    fn planning_recovery_branches_and_procedure_bindings_flow() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/planning-recovery").unwrap();

            let plan = store
                .create_plan_trace(&ns, "Deploy robot update", 7, None, None)
                .await
                .unwrap();
            let cp = store
                .add_plan_checkpoint(
                    &ns,
                    &plan.id,
                    "field_test",
                    "Run field test",
                    1,
                    None,
                    None,
                )
                .await
                .unwrap();

            let cp_failed = store
                .update_plan_checkpoint_status(
                    &ns,
                    &cp.id,
                    "failed",
                    Some("battery thermal event"),
                    None,
                )
                .await
                .unwrap()
                .expect("checkpoint should exist");
            assert_eq!(cp_failed.status, "failed");

            let branch = store
                .add_plan_recovery_branch(
                    &ns,
                    &plan.id,
                    Some(&cp.id),
                    "thermal-mitigation",
                    "Field test failure due to thermal event",
                    Some(serde_json::json!({
                        "steps": ["limit peak current", "repeat field test"]
                    })),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(branch.status, "active");

            let primary = store
                .bind_procedure_to_plan(
                    &ns,
                    &plan.id,
                    "memory_search",
                    "primary",
                    0.87,
                    Some(serde_json::json!({"phase":"execution"})),
                )
                .await
                .unwrap();
            assert_eq!(primary.binding_role, "primary");
            let rollback = store
                .bind_procedure_to_plan(
                    &ns,
                    &plan.id,
                    "memory_update",
                    "rollback",
                    0.72,
                    Some(serde_json::json!({"phase":"recovery"})),
                )
                .await
                .unwrap();
            assert_eq!(rollback.binding_role, "rollback");

            let resolved_branch = store
                .resolve_plan_recovery_branch(
                    &ns,
                    &branch.id,
                    "resolved",
                    Some("thermal strategy validated"),
                )
                .await
                .unwrap()
                .expect("branch should exist");
            assert_eq!(resolved_branch.status, "resolved");
            assert!(resolved_branch.resolved_at.is_some());

            let branches = store
                .list_plan_recovery_branches(&ns, &plan.id)
                .await
                .unwrap();
            assert_eq!(branches.len(), 1);
            assert_eq!(branches[0].id, branch.id);

            let bindings = store
                .list_plan_procedure_bindings(&ns, &plan.id)
                .await
                .unwrap();
            assert_eq!(bindings.len(), 2);
            assert!(bindings.iter().any(|b| b.binding_role == "primary"));
            assert!(bindings.iter().any(|b| b.binding_role == "rollback"));
        });
    }

    #[test]
    fn counterfactual_simulator_prioritizes_causal_paths() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/counterfactual").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let a = insert_test_memory(&conn, ns.as_str(), "Power surge occurs", &make_embedding(dim, 11));
            let b = insert_test_memory(&conn, ns.as_str(), "Controller restarts unexpectedly", &make_embedding(dim, 12));
            let c = insert_test_memory(&conn, ns.as_str(), "Mission timeline slips by one day", &make_embedding(dim, 13));
            let d = insert_test_memory(&conn, ns.as_str(), "Weather was cloudy", &make_embedding(dim, 14));
            drop(conn);

            store
                .upsert_causal_edge(&ns, &a, &b, "causal", 0.95, Some("direct electrical effect"))
                .await
                .unwrap();
            store
                .upsert_causal_edge(&ns, &b, &c, "causal", 0.88, Some("restart delays execution"))
                .await
                .unwrap();
            store
                .upsert_causal_edge(&ns, &a, &d, "correlational", 0.7, Some("same day"))
                .await
                .unwrap();

            let sim = store
                .simulate_counterfactual(&ns, "What if the power surge did not happen?", &[a.clone()], 3, 5, true, false)
                .await
                .unwrap();
            assert!(!sim.hypotheses.is_empty(), "simulation should return hypotheses");
            assert_eq!(sim.seed_memory_ids, vec![a.clone()]);
            assert_eq!(sim.hypotheses[0].memory_id, b, "direct causal node should rank first");
            let has_c = sim.hypotheses.iter().any(|h| h.memory_id == c);
            assert!(has_c, "downstream causal consequence should appear");
        });
    }

    #[test]
    fn counterfactual_simulator_uses_transition_edges() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/counterfactual-transition").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let start = insert_test_memory(&conn, ns.as_str(), "Battery health is nominal", &make_embedding(dim, 31));
            let next = insert_test_memory(&conn, ns.as_str(), "Battery enters thermal throttle mode", &make_embedding(dim, 32));
            drop(conn);

            store
                .upsert_state_transition(
                    &ns,
                    &start,
                    &next,
                    "state_change",
                    0.9,
                    Some("telemetry transition"),
                )
                .await
                .unwrap();

            let transitions = store.list_state_transitions(&ns, 10).await.unwrap();
            assert_eq!(transitions.len(), 1);
            assert_eq!(transitions[0].transition_type, "state_change");

            let sim = store
                .simulate_counterfactual(
                    &ns,
                    "If the battery temperature were capped earlier",
                    &[start.clone()],
                    2,
                    5,
                    false,
                    true,
                )
                .await
                .unwrap();
            assert!(
                sim.hypotheses.iter().any(|h| h.memory_id == next),
                "state transition target should be included in hypotheses"
            );
        });
    }


    #[test]
    fn causal_edges_upsert_and_list() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/causal").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let a = insert_test_memory(&conn, ns.as_str(), "Heavy rain happened", &make_embedding(dim, 1));
            let b = insert_test_memory(&conn, ns.as_str(), "Flooding happened", &make_embedding(dim, 2));

            upsert_causal_edge_sync(
                &conn,
                &ns,
                &a,
                &b,
                "causal",
                0.8,
                Some("rain -> flood"),
            )
            .unwrap();
            upsert_causal_edge_sync(
                &conn,
                &ns,
                &a,
                &b,
                "causal",
                0.9,
                None,
            )
            .unwrap();

            let edges = list_causal_edges_sync(&conn, &ns, 10).unwrap();
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].relation_type, "causal");
            assert!(edges[0].confidence > 0.85);
            assert_eq!(edges[0].evidence.as_deref(), Some("rain -> flood"));
        });
    }

    #[test]
    fn causal_query_boost_distinguishes_causal_vs_correlational() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let conn = pool.writer().await;
            let ns = Namespace::parse("test/causal-boost").unwrap();
            let dim = DbPool::DEFAULT_DIMENSION;
            let emb = vec![0.01f32; dim];

            let cause = insert_test_memory(
                &conn,
                ns.as_str(),
                "Heavy rain caused flooding",
                &emb,
            );
            let effect = insert_test_memory(
                &conn,
                ns.as_str(),
                "Flooding disrupted city traffic",
                &emb,
            );
            let corr = insert_test_memory(
                &conn,
                ns.as_str(),
                "Traffic and festivals often happen in summer",
                &emb,
            );

            upsert_causal_edge_sync(
                &conn,
                &ns,
                &cause,
                &effect,
                "causal",
                1.0,
                Some("direct mechanism"),
            )
            .unwrap();
            upsert_causal_edge_sync(
                &conn,
                &ns,
                &cause,
                &corr,
                "correlational",
                1.0,
                Some("co-occurrence only"),
            )
            .unwrap();

            let boosts = causal_boosts_for_seed_ids_sync(&conn, &ns, &[cause.clone()]).unwrap();
            let effect_score = boosts.get(&effect).copied().unwrap_or(0.0);
            let corr_score = boosts.get(&corr).copied().unwrap_or(0.0);
            assert!(
                effect_score > corr_score,
                "causal edge should outrank correlational edge for causal query: {effect_score} vs {corr_score}"
            );
        });
    }

    #[test]
    fn working_memory_lifecycle_and_expiry() {
        let pool = DbPool::open_in_memory().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let store = MemoryStore::new(pool.clone());
            let ns = Namespace::parse("test/working").unwrap();

            let expired_at = (Utc::now() - chrono::Duration::seconds(5)).to_rfc3339();
            let pending = store
                .add_working_memory(
                    &ns,
                    "I might move cities next year",
                    0.42,
                    Some(serde_json::json!({"source":"unit"})),
                    Some("conv-1"),
                    Some(&expired_at),
                )
                .await
                .unwrap();

            let active_at = (Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
            let surviving = store
                .add_working_memory(
                    &ns,
                    "I always charge batteries before navigation.",
                    0.73,
                    None,
                    Some("conv-1"),
                    Some(&active_at),
                )
                .await
                .unwrap();

            let before = store
                .list_working_memories(&ns, Some("pending"), Some("conv-1"), 20)
                .await
                .unwrap();
            assert_eq!(before.len(), 2);

            let expired = store
                .expire_working_memories(&Utc::now().to_rfc3339())
                .await
                .unwrap();
            assert!(expired >= 1);

            let after = store
                .list_working_memories(&ns, Some("pending"), Some("conv-1"), 20)
                .await
                .unwrap();
            assert_eq!(after.len(), 1);
            assert_eq!(after[0].id, surviving.id);

            let updated = store
                .update_working_memory_state(
                    &surviving.id,
                    Some(0.91),
                    Some("committed"),
                    Some("mem-123"),
                )
                .await
                .unwrap();
            assert!(updated);

            let committed = store
                .list_working_memories(&ns, Some("committed"), Some("conv-1"), 20)
                .await
                .unwrap();
            assert_eq!(committed.len(), 1);
            assert_eq!(committed[0].committed_memory_id.as_deref(), Some("mem-123"));
            assert!(committed[0].provisional_score > 0.9);

            let expired_rows = store
                .list_working_memories(&ns, Some("expired"), Some("conv-1"), 20)
                .await
                .unwrap();
            assert_eq!(expired_rows.len(), 1);
            assert_eq!(expired_rows[0].id, pending.id);
        });
    }
}
