use serde::{Deserialize, Serialize};

// --- Add Memory ---

#[derive(Debug, Deserialize)]
pub struct AddMemoryRequest {
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default = "default_true")]
    pub consolidate: bool,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub episode_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    pub id: String,
    pub action: String,
    pub merged_into: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddMemoryBatchItem {
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub episode_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddMemoryBatchRequest {
    pub items: Vec<AddMemoryBatchItem>,
    /// Enqueue reflection for inserted IDs (batched).
    #[serde(default = "default_true")]
    pub reflect: bool,
}

#[derive(Debug, Serialize)]
pub struct AddMemoryBatchResponse {
    pub created: usize,
    pub ids: Vec<String>,
    pub elapsed_ms: f64,
}

// --- Search ---

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[allow(dead_code)]
    #[serde(default)]
    pub metadata_filter: Option<serde_json::Value>,
    #[serde(default)]
    pub search_mode: Option<String>,
    #[serde(default)]
    pub temporal_weight: Option<f64>,
    /// Maximum memory tier to include in results (1=raw, 2=reflected, 3=deduced, 4=induced).
    /// Defaults to server-side config. Set to 4 to include all tiers.
    #[serde(default)]
    pub max_tier: Option<i32>,
    /// ISO 8601 date filter (inclusive start, e.g. "2023-01-01").
    #[serde(default)]
    pub date_start: Option<String>,
    /// ISO 8601 date filter (inclusive end, e.g. "2023-12-31").
    #[serde(default)]
    pub date_end: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultDto>,
    pub query_time_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchResultDto {
    pub id: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    pub memory_type: String,
    pub score: f64,
    pub vector_score: Option<f64>,
    pub keyword_score: Option<f64>,
    pub temporal_score: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

// --- Get Memory ---

#[derive(Debug, Serialize)]
pub struct MemoryResponse {
    pub id: String,
    pub namespace: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    pub memory_type: String,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub access_count: u64,
    pub importance: i32,
}

// --- Update Memory ---

#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub content: Option<String>,
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
    pub importance: Option<i32>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct UpdateMemoryResponse {
    pub id: String,
    pub updated: bool,
}

// --- Memory Versioning ---

#[derive(Debug, Deserialize)]
pub struct PatchMemoryRequest {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub importance: Option<i32>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PatchMemoryResponse {
    pub id: String,
    pub patched: bool,
    pub revision_number: i64,
}

#[derive(Debug, Deserialize)]
pub struct RollbackMemoryRequest {
    pub revision_number: i64,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RollbackMemoryResponse {
    pub id: String,
    pub rolled_back_to_revision: i64,
    pub revision_number: i64,
}

#[derive(Debug, Deserialize)]
pub struct MergeMemoriesRequest {
    pub source_memory_ids: Vec<String>,
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub importance: Option<i32>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MergeMemoriesResponse {
    pub merged_memory_id: String,
    pub superseded_source_ids: Vec<String>,
    pub revision_number: i64,
}

#[derive(Debug, Deserialize)]
pub struct RevisionListParams {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AuditListParams {
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ProcedureRevisionRequest {
    pub operation: String,
    pub spec: serde_json::Value,
    #[serde(default)]
    pub reason: Option<String>,
}

// --- Identity ---

#[derive(Debug, Deserialize)]
pub struct CreateIdentityEntityRequest {
    pub canonical_name: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AddIdentityAliasRequest {
    pub alias: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct IdentityResolveParams {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

// --- Planning ---

#[derive(Debug, Deserialize)]
pub struct CreatePlanRequest {
    pub goal: String,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default)]
    pub due_at: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ListPlansParams {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AddPlanCheckpointRequest {
    pub checkpoint_key: String,
    pub title: String,
    #[serde(default)]
    pub order_index: Option<i32>,
    #[serde(default)]
    pub expected_by: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpdatePlanCheckpointRequest {
    pub status: String,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct SetPlanOutcomeRequest {
    pub status: String,
    pub outcome: String,
    #[serde(default)]
    pub outcome_confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AddPlanRecoveryBranchRequest {
    #[serde(default)]
    pub source_checkpoint_id: Option<String>,
    pub branch_label: String,
    pub trigger_reason: String,
    #[serde(default)]
    pub branch_plan: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ResolvePlanRecoveryBranchRequest {
    pub status: String,
    #[serde(default)]
    pub resolution_notes: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BindProcedureToPlanRequest {
    pub procedure_name: String,
    #[serde(default)]
    pub binding_role: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

// --- Counterfactual ---

#[derive(Debug, Deserialize)]
pub struct UpsertTransitionRequest {
    pub from_memory_id: String,
    pub to_memory_id: String,
    #[serde(default)]
    pub transition_type: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub evidence: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListTransitionsParams {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CounterfactualRequest {
    pub intervention: String,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub seed_memory_ids: Vec<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_hops: Option<usize>,
    #[serde(default = "default_true")]
    pub include_correlational: bool,
    #[serde(default = "default_true")]
    pub include_transitions: bool,
}

// --- Corrections ---

#[derive(Debug, Deserialize)]
pub struct CorrectionRequest {
    pub target_memory_id: String,
    pub corrected_content: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub importance: Option<i32>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct CorrectionResponse {
    pub correction_event_id: String,
    pub superseded_memory_id: String,
    pub new_memory_id: String,
    pub contradiction_logged: bool,
    pub recalibrated: bool,
}

// --- Error ---

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// --- List Memories ---

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    pub status: Option<String>,
    pub memory_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListMemoriesResponse {
    pub memories: Vec<MemoryResponse>,
    pub total: u64,
    pub offset: usize,
    pub limit: usize,
}

// --- Working Memory ---

#[derive(Debug, Deserialize)]
pub struct AddWorkingMemoryRequest {
    pub content: String,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub provisional_score: Option<f64>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct WorkingMemoryEntryDto {
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

#[derive(Debug, Serialize)]
pub struct AddWorkingMemoryResponse {
    pub entry: WorkingMemoryEntryDto,
}

#[derive(Debug, Deserialize)]
pub struct ListWorkingMemoryParams {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ListWorkingMemoryResponse {
    pub entries: Vec<WorkingMemoryEntryDto>,
}

#[derive(Debug, Deserialize)]
pub struct CommitWorkingMemoryRequest {
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub min_score: Option<f64>,
    #[serde(default)]
    pub r#async: bool,
}

#[derive(Debug, Serialize)]
pub struct CommitWorkingMemoryResponse {
    pub status: String,
    pub evaluated: usize,
    pub committed: usize,
    pub committed_memory_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ReconsolidateRequest {
    pub memory_ids: Vec<String>,
    #[serde(default)]
    pub r#async: bool,
}

#[derive(Debug, Serialize)]
pub struct ReconsolidateResponse {
    pub status: String,
    pub processed: usize,
    pub superseded: usize,
}

#[derive(Debug, Deserialize)]
pub struct RetentionRunRequest {
    #[serde(default)]
    pub limit_per_namespace: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RetentionRunResponse {
    pub processed: usize,
    pub softened: usize,
}

// --- Graph ---

#[derive(Debug, Deserialize)]
pub struct GraphParams {
    pub threshold: Option<f64>,
    pub limit: Option<usize>,
}

// --- Access Ranking ---

#[derive(Debug, Deserialize)]
pub struct AccessRankingParams {
    pub order: Option<String>,
    pub limit: Option<usize>,
}

// --- Chat ---

#[derive(Debug, Clone, Deserialize)]
pub struct FileAttachment {
    /// "image" or "text"
    pub r#type: String,
    /// For images: base64 data. For text: the extracted text content.
    pub data: String,
    /// MIME type (e.g. "image/png", "text/plain")
    #[serde(default)]
    pub media_type: Option<String>,
    /// Original file name
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    #[serde(default = "default_chat_mode")]
    pub mode: String, // "rag" or "tool"
    /// Optional LLM override. If omitted, uses server-side [llm] config.
    #[serde(default)]
    pub llm: Option<LlmConfig>,
    #[serde(default)]
    pub attachments: Vec<FileAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Whether the model supports vision/image inputs
    #[serde(default)]
    pub vision: bool,
    /// Whether the model supports tool/function calling
    #[serde(default = "default_true")]
    pub tool_use: bool,
    /// Whether the model supports streaming
    #[allow(dead_code)]
    #[serde(default = "default_true")]
    pub streaming: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub reply: String,
    pub memories_used: Vec<MemoryContext>,
    pub memories_added: Vec<AddedMemory>,
    pub mode: String,
    pub provenance: ResponseProvenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContext {
    pub id: String,
    pub content: String,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub memory_type: String,
    pub importance: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AddedMemory {
    pub id: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceSource {
    pub memory_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseProvenance {
    pub namespace: String,
    pub generated_at: String,
    pub source_ids: Vec<String>,
    pub redaction_applied: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<ProvenanceSource>,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamRequest {
    pub message: String,
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    #[serde(default = "default_chat_mode")]
    pub mode: String,
    /// Optional LLM override. If omitted, uses server-side [llm] config.
    #[serde(default)]
    pub llm: Option<LlmConfig>,
    #[allow(dead_code)]
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub attachments: Vec<FileAttachment>,
}

fn default_chat_mode() -> String {
    "rag".into()
}

// --- Conversations ---

#[derive(Debug, Deserialize)]
pub struct CreateConversationRequest {
    #[serde(default = "default_conversation_title")]
    pub title: String,
    #[serde(default = "default_chat_mode")]
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateConversationRequest {
    pub title: Option<String>,
    pub messages: Option<serde_json::Value>,
}

fn default_conversation_title() -> String {
    "New Chat".into()
}

// --- Defaults ---

fn default_true() -> bool {
    true
}

fn default_limit() -> usize {
    10
}

// --- Ask (direct answer from memories) ---

#[derive(Debug, Deserialize)]
pub struct AskRequest {
    pub question: String,
    /// Optional LLM override. If omitted, uses server-side [llm] config.
    #[serde(default)]
    pub llm: Option<LlmConfig>,
    #[serde(default = "default_ask_search_limit")]
    pub search_limit: usize,
    #[serde(default = "default_ask_max_results")]
    pub max_results: usize,
    /// Skip LLM — return top memory content as the answer (for benchmarking)
    #[serde(default)]
    pub skip_llm: bool,
    /// Filter which memory types to include (e.g. ["event", "reflected", "deduced"]).
    /// If empty or omitted, all types are included.
    #[serde(default)]
    pub memory_types: Vec<String>,
    /// Maximum memory tier to include (1=raw, 2=+reflected, 3=+deduced, 4=all).
    #[serde(default)]
    pub max_tier: Option<i32>,
    /// ISO 8601 date filter (inclusive start).
    #[serde(default)]
    pub date_start: Option<String>,
    /// ISO 8601 date filter (inclusive end).
    #[serde(default)]
    pub date_end: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AskResponse {
    pub answer: String,
    pub queries_used: Vec<String>,
    pub memories_referenced: Vec<AskMemoryRef>,
    pub total_search_results: usize,
    pub elapsed_ms: f64,
    pub provenance: ResponseProvenance,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub needs_confirmation: Vec<ConfirmationQuestion>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfirmationQuestion {
    pub question: String,
    pub source_memory_id: String,
    pub confidence: f64,
}

#[derive(Debug, Serialize)]
pub struct AskMemoryRef {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub score: f64,
    pub created_at: String,
}

// --- Reflect (structured fact extraction) ---

#[derive(Debug, Deserialize)]
pub struct ReflectRequest {
    /// Specific memory IDs to reflect on. If empty, auto-finds unreflected raw memories.
    #[serde(default)]
    pub memory_ids: Vec<String>,
    /// Max raw memories to process (when auto-finding).
    #[serde(default = "default_reflect_limit")]
    pub limit: usize,
    /// If true, enqueue as background job and return immediately.
    #[serde(default)]
    pub r#async: bool,
}

#[derive(Debug, Serialize)]
pub struct ReflectResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_processed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facts_extracted: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reflected: Vec<ReflectedFactDto>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReflectedFactDto {
    pub content: String,
    pub confidence: f64,
    pub source_ids: Vec<String>,
    pub corrections: Option<String>,
}

fn default_reflect_limit() -> usize {
    50
}

fn default_ask_search_limit() -> usize {
    30
}

fn default_ask_max_results() -> usize {
    30
}

// --- Embeddings (3D visualization) ---

#[derive(Debug, Serialize)]
pub struct EmbeddingPointDto {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub points: Vec<EmbeddingPointDto>,
}
