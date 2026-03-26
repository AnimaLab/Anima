import type {
  AddMemoryRequest, AddMemoryResponse, SearchRequest, SearchResponse,
  ListMemoriesResponse, NamespaceStats, NamespaceInfo, GraphData, Memory,
  ChatRequest, ChatResponse, Conversation, ConversationSummary,
  LlmConfig, ChatMessage, StreamEvent, FileAttachment,
  AskRequest, AskResponse, EmbeddingsResponse,
  ContradictionEntry, SupersessionLink, ProfilesResponse
} from './types'

let currentNamespace = 'default'

export function setNamespace(ns: string) {
  currentNamespace = ns
}

export function getNamespace() {
  return currentNamespace
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((options.headers as Record<string, string>) || {}),
  }
  if (!path.startsWith('/health') && !path.startsWith('/api/v1/telemetry')) {
    // Allow per-request override (e.g. deleteNamespace, renameNamespace pass a specific namespace)
    if (!headers['X-Anima-Namespace']) {
      headers['X-Anima-Namespace'] = currentNamespace
    }
  }
  const res = await fetch(path, { ...options, headers })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  addMemory: (req: AddMemoryRequest) =>
    request<AddMemoryResponse>('/api/v1/memories', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  listMemories: (offset = 0, limit = 50, status?: string, memory_type?: string) =>
    request<ListMemoriesResponse>(
      `/api/v1/memories?offset=${offset}&limit=${limit}${status ? `&status=${status}` : ''}${memory_type ? `&memory_type=${memory_type}` : ''}`
    ),

  getMemory: (id: string) =>
    request<Memory>(`/api/v1/memories/${id}`),

  updateMemory: (id: string, data: { content?: string; memory_type?: string; importance?: number; tags?: string[] }) =>
    request<{ id: string; updated: boolean }>(`/api/v1/memories/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteMemory: (id: string) =>
    request<{ id: string; deleted: boolean }>(`/api/v1/memories/${id}`, {
      method: 'DELETE',
    }),

  purgeDeletedMemories: () =>
    request<{ purged: number }>('/api/v1/memories/purge', {
      method: 'POST',
    }),

  search: (req: SearchRequest) =>
    request<SearchResponse>('/api/v1/memories/search', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getStats: () => request<NamespaceStats>('/api/v1/stats'),
  getVecStatus: () => request<{ status: string; needs_reindex: boolean; existing_dimension?: number; requested_dimension?: number }>('/api/v1/vec/status'),
  reindex: () => request<{ reindexed: number; dimension: number }>('/api/v1/vec/reindex', { method: 'POST' }),
  listNamespaces: () => request<NamespaceInfo[]>('/api/v1/namespaces'),
  deleteNamespace: (ns: string) =>
    request<{ namespace: string; deleted_memories: number }>('/api/v1/namespaces', {
      method: 'DELETE',
      headers: { 'X-Anima-Namespace': ns } as Record<string, string>,
    }),
  renameNamespace: (oldNs: string, newName: string) =>
    request<{ old_namespace: string; new_namespace: string; renamed_memories: number }>('/api/v1/namespaces/rename', {
      method: 'POST',
      headers: { 'X-Anima-Namespace': oldNs } as Record<string, string>,
      body: JSON.stringify({ name: newName }),
    }),
  getGraph: (threshold = 0.7, limit = 200) =>
    request<GraphData>(`/api/v1/graph?threshold=${threshold}&limit=${limit}`),
  getEmbeddings: (limit = 300) =>
    request<EmbeddingsResponse>(`/api/v1/embeddings?limit=${limit}`),
  topAccessed: (order = 'most', limit = 10) =>
    request<Memory[]>(`/api/v1/memories/top-accessed?order=${order}&limit=${limit}`),

  // Ask (extract-then-answer pipeline)
  ask: (req: AskRequest) =>
    request<AskResponse>('/api/v1/ask', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  // Non-streaming chat (kept for tool mode)
  chat: (req: ChatRequest) =>
    request<ChatResponse>('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  // Streaming chat via SSE
  chatStream: async (
    message: string,
    history: ChatMessage[],
    mode: string,
    llm: LlmConfig,
    conversationId?: string,
    onEvent?: (event: StreamEvent) => void,
    attachments?: FileAttachment[],
    signal?: AbortSignal,
  ): Promise<void> => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Anima-Namespace': currentNamespace,
    }
    const body = JSON.stringify({
      message, history, mode, llm,
      conversation_id: conversationId,
      attachments: attachments?.length ? attachments : undefined,
    })
    const res = await fetch('/api/v1/chat/stream', { method: 'POST', headers, body, signal })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }))
      throw new Error(err.error || res.statusText)
    }
    const reader = res.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || !trimmed.startsWith('data: ')) continue
        const data = trimmed.slice(6)
        if (data === '[DONE]') continue
        try {
          const event: StreamEvent = JSON.parse(data)
          onEvent?.(event)
        } catch {
          // ignore parse errors
        }
      }
    }
  },

  // Conversations
  createConversation: (title = 'New Chat', mode = 'rag') =>
    request<Conversation>('/api/v1/conversations', {
      method: 'POST',
      body: JSON.stringify({ title, mode }),
    }),

  listConversations: () =>
    request<ConversationSummary[]>('/api/v1/conversations'),

  getConversation: (id: string) =>
    request<Conversation>(`/api/v1/conversations/${id}`),

  updateConversation: (id: string, data: { title?: string; messages?: unknown }) =>
    request<{ updated: boolean }>(`/api/v1/conversations/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteConversation: (id: string) =>
    request<{ deleted: boolean }>(`/api/v1/conversations/${id}`, {
      method: 'DELETE',
    }),

  generateTitle: (convId: string, llm: LlmConfig) =>
    request<{ title: string }>(`/api/v1/conversations/${convId}/title`, {
      method: 'POST',
      body: JSON.stringify(llm),
    }),

  // Profiles
  getProfiles: () =>
    request<ProfilesResponse>('/api/v1/profiles'),

  // Conflicts & Contradictions
  listContradictions: (limit = 50, offset = 0) =>
    request<ContradictionEntry[]>(`/api/v1/contradictions?limit=${limit}&offset=${offset}`),

  getMemoryHistory: (id: string) =>
    request<SupersessionLink[]>(`/api/v1/memories/${id}/history`),

  // Processor log
  getProcessorStatus: () =>
    request<{
      queue_depth: number; in_flight: number; dead_letter_count: number;
      idle: boolean; metrics: { completed_jobs: number; failed_jobs: number };
    }>('/api/v1/processor/status'),

  getProcessorLog: (limit = 50, offset = 0) =>
    request<Array<{
      id: string; namespace: string; pipeline: string; status: string;
      input_count: number; output_count: number;
      prompt_tokens: number; completion_tokens: number; total_tokens: number;
      elapsed_ms: number; details: unknown; created_at: string;
    }>>(`/api/v1/processor/log?limit=${limit}&offset=${offset}`),

  // Telemetry
  getTelemetryConfig: () =>
    request<{ enabled: boolean }>('/api/v1/telemetry/config'),

  setTelemetryConfig: (enabled: boolean, featureFlags?: Record<string, boolean>) =>
    request<{ updated: boolean }>('/api/v1/telemetry/config', {
      method: 'PUT',
      body: JSON.stringify({ enabled, feature_flags: featureFlags }),
    }),

  // Backup & Restore
  exportBackupJson: async (namespace?: string): Promise<Blob> => {
    const ns = namespace || currentNamespace
    const res = await fetch(`/api/v1/backup?format=json`, {
      headers: { 'X-Anima-Namespace': ns },
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }))
      throw new Error(err.error || res.statusText)
    }
    return res.blob()
  },

  exportBackupSqlite: async (): Promise<Blob> => {
    const res = await fetch(`/api/v1/backup?format=sqlite`, {
      headers: { 'X-Anima-Namespace': currentNamespace },
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }))
      throw new Error(err.error || res.statusText)
    }
    return res.blob()
  },

  importBackup: (backup: unknown, mode: 'merge' | 'replace' = 'merge') =>
    request<{ imported: number; skipped: number; total: number; elapsed_ms: number }>(
      '/api/v1/restore',
      {
        method: 'POST',
        body: JSON.stringify({ mode, backup }),
      }
    ),

  importBackupSqlite: async (file: File): Promise<void> => {
    const buffer = await file.arrayBuffer()
    const res = await fetch('/api/v1/restore/sqlite', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Anima-Namespace': currentNamespace,
      },
      body: buffer,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }))
      throw new Error(err.error || res.statusText)
    }
  },
}
