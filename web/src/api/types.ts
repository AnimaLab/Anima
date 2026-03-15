export interface Memory {
  id: string
  namespace: string
  content: string
  metadata: Record<string, unknown> | null
  tags: string[]
  memory_type: string
  status: string
  created_at: string
  updated_at: string
  access_count: number
  importance: number
  event_date?: string
}

export interface AddMemoryRequest {
  content: string
  metadata?: Record<string, unknown>
  consolidate: boolean
  tags?: string[]
}

export interface AddMemoryResponse {
  id: string
  action: string
  merged_into: string | null
}

export interface SearchRequest {
  query: string
  limit?: number
  search_mode?: 'hybrid' | 'vector' | 'keyword'
  temporal_weight?: number
  max_tier?: number
  date_start?: string
  date_end?: string
}

export interface SearchResult {
  id: string
  content: string
  metadata: Record<string, unknown> | null
  tags: string[]
  memory_type: string
  score: number
  vector_score: number | null
  keyword_score: number | null
  temporal_score: number | null
  created_at: string
  updated_at: string
  importance: number
}

export interface SearchResponse {
  results: SearchResult[]
  query_time_ms: number
}

export interface ListMemoriesResponse {
  memories: Memory[]
  total: number
  offset: number
  limit: number
}

export interface NamespaceStats {
  total: number
  active: number
  superseded: number
  deleted: number
  avg_access_count: number
  max_access_count: number
  oldest_memory: string | null
  newest_memory: string | null
}

export interface NamespaceInfo {
  namespace: string
  total_count: number
  active_count: number
}

export interface GraphNode {
  id: string
  content: string
  namespace: string
  status: string
  memory_type: string
  importance: number
  created_at: string
  access_count: number
}

export interface GraphEdge {
  source: string
  target: string
  similarity: number
  edge_type: 'similarity' | 'superseded'
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// --- Chat ---

export interface LlmConfig {
  base_url: string
  model: string
  api_key?: string
  temperature?: number
  max_tokens?: number
  system_prompt?: string
  vision?: boolean
  tool_use?: boolean
  streaming?: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface FileAttachment {
  type: 'text' | 'image'
  data: string           // text content or base64 image data
  media_type?: string    // e.g. "image/png"
  name?: string          // original file name
}

export interface ChatRequest {
  message: string
  history: ChatMessage[]
  mode: 'rag' | 'tool'
  llm: LlmConfig
  attachments?: FileAttachment[]
}

export interface MemoryContext {
  id: string
  content: string
  score: number
  source?: string
  memory_type: string
  importance: number
}

export interface ChatResponse {
  reply: string
  memories_used: MemoryContext[]
  memories_added: { id: string; content: string }[]
  mode: string
}

// --- Conversations ---

export interface Conversation {
  id: string
  namespace: string
  title: string
  mode: string
  messages: string // JSON array
  created_at: string
  updated_at: string
}

export interface ConversationSummary {
  id: string
  title: string
  mode: string
  created_at: string
  updated_at: string
}

// --- SSE Stream Events ---

export interface StreamMemoriesEvent {
  type: 'memories'
  memories_used: MemoryContext[]
}

export interface StreamTokenEvent {
  type: 'token'
  content: string
}

export interface StreamDoneEvent {
  type: 'done'
  full_reply: string
}

export interface StreamErrorEvent {
  type: 'error'
  error: string
}

export type StreamEvent = StreamMemoriesEvent | StreamTokenEvent | StreamDoneEvent | StreamErrorEvent

// --- Embeddings (3D visualization) ---

export interface EmbeddingPoint {
  id: string
  content: string
  memory_type: string
  x: number
  y: number
  z: number
}

export interface EmbeddingsResponse {
  points: EmbeddingPoint[]
}

// --- Ask (direct answer from memories) ---

export interface AskRequest {
  question: string
  llm?: LlmConfig
  search_limit?: number
  max_results?: number
  max_tier?: number
  date_start?: string
  date_end?: string
}

export interface AskMemoryRef {
  id: string
  content: string
  score: number
  created_at: string
}

export interface AskResponse {
  answer: string
  queries_used: string[]
  memories_referenced: AskMemoryRef[]
  total_search_results: number
  elapsed_ms: number
}
