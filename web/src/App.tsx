import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { NamespaceContext } from './hooks/useNamespace'
import { ChatContext, type DisplayMessage, type SendMessageParams } from './hooks/useChat'
import { setNamespace as setApiNamespace, api } from './api/client'
import type { LlmConfig, ConversationSummary, FileAttachment, ChatMessage, MemoryContext, StreamEvent } from './api/types'
import { Layout } from './components/Layout'
import { DashboardPage } from './pages/DashboardPage'
import { MemoriesPage } from './pages/MemoriesPage'
import { SearchPage } from './pages/SearchPage'
import { GraphPage } from './pages/GraphPage'
import { Graph3DPage } from './pages/Graph3DPage'
import { EmbeddingPage } from './pages/EmbeddingPage'
import { ChatPage } from './pages/ChatPage'
import { SettingsPage } from './pages/SettingsPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 10000 },
  },
})

const DEFAULT_LLM_CONFIG: LlmConfig = {
  base_url: 'http://localhost:11434/v1',
  model: '',
  api_key: '',
}

export default function App() {
  const [namespace, setNamespaceState] = useState(() => {
    try {
      return localStorage.getItem('anima-namespace') || 'default'
    } catch {
      return 'default'
    }
  })

  // Chat state lifted to App so it survives tab switches
  const [chatMessages, setChatMessages] = useState<DisplayMessage[]>([])
  const [chatMode, setChatMode] = useState<'rag' | 'tool'>('rag')
  const [chatConfig, setChatConfig] = useState<LlmConfig>(() => {
    try {
      const saved = localStorage.getItem('anima-llm-config')
      return saved ? JSON.parse(saved) : DEFAULT_LLM_CONFIG
    } catch {
      return DEFAULT_LLM_CONFIG
    }
  })
  const [conversationId, setConversationIdRaw] = useState<string | null>(() => {
    try {
      return localStorage.getItem('anima-conversation-id') || null
    } catch {
      return null
    }
  })
  const setConversationId = useCallback((id: string | null) => {
    setConversationIdRaw(id)
    try {
      if (id) localStorage.setItem('anima-conversation-id', id)
      else localStorage.removeItem('anima-conversation-id')
    } catch { /* ignore */ }
  }, [])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])

  // Per-conversation loading tracking (Set of convKeys currently streaming)
  const [loadingConvIds, setLoadingConvIds] = useState<Set<string>>(new Set())
  const [streamingContent, setStreamingContent] = useState('')
  const [streamingMemories, setStreamingMemories] = useState<MemoryContext[]>([])

  // Track which conversation is currently streaming (for visible UI)
  const streamingConvIdRef = useRef<string | null>(null)

  // Per-conversation abort controllers
  const abortsRef = useRef<Map<string, AbortController>>(new Map())

  // Per-conversation message queues (for same-conv sequential messages)
  const queuesRef = useRef<Map<string, SendMessageParams[]>>(new Map())

  // Per-conversation API history
  const apiHistoriesRef = useRef<Map<string, ChatMessage[]>>(new Map())
  const fromQueueRef = useRef(false)

  // Refs for latest values inside the async sendMessage closure
  const loadingConvIdsRef = useRef(loadingConvIds)
  loadingConvIdsRef.current = loadingConvIds
  const messagesRef = useRef(chatMessages)
  messagesRef.current = chatMessages
  const modeRef = useRef(chatMode)
  modeRef.current = chatMode
  const configRef = useRef(chatConfig)
  configRef.current = chatConfig
  const convIdRef = useRef(conversationId)
  convIdRef.current = conversationId
  const conversationsRef = useRef(conversations)
  conversationsRef.current = conversations
  const streamingContentRef = useRef(streamingContent)
  streamingContentRef.current = streamingContent
  const sendMessageRef = useRef<(params: SendMessageParams) => Promise<void>>(null!)

  // Reset API history when chat is cleared
  useEffect(() => {
    if (chatMessages.length === 0) apiHistoriesRef.current.delete('__new__')
  }, [chatMessages.length])

  // Restore conversation messages on mount if we have a saved conversationId
  useEffect(() => {
    if (!conversationId) return
    api.getConversation(conversationId).then(conv => {
      try {
        const parsed: DisplayMessage[] = JSON.parse(conv.messages)
        if (parsed.length > 0) {
          setChatMessages(parsed)
          messagesRef.current = parsed
          setChatMode(conv.mode as 'rag' | 'tool')
          // Rebuild API history from loaded messages
          const history = parsed
            .filter(m => m.role === 'user' || m.role === 'assistant')
            .map(m => ({ role: m.role, content: m.content }))
          apiHistoriesRef.current.set(conv.id, history)
        }
      } catch { /* ignore parse errors */ }
    }).catch(() => {
      // Conversation no longer exists, clear the stored ID
      setConversationId(null)
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Rebuild API history when switching conversations (from Layout's loadConversation)
  useEffect(() => {
    if (conversationId && !loadingConvIds.has(conversationId)) {
      const history = chatMessages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content }))
      apiHistoriesRef.current.set(conversationId, history)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId])

  // Process queued messages for the current conversation when it stops loading
  useEffect(() => {
    const convKey = conversationId || '__new__'
    const queue = queuesRef.current.get(convKey)
    if (!loadingConvIds.has(convKey) && queue && queue.length > 0) {
      const next = queue.shift()!
      fromQueueRef.current = true
      sendMessageRef.current(next)
    }
  }, [loadingConvIds, conversationId])

  const saveMessages = useCallback(async (convId: string, msgs: DisplayMessage[]) => {
    const serialized = msgs.map(m => ({
      role: m.role,
      content: m.content,
      memoriesUsed: m.memoriesUsed,
      memoriesAdded: m.memoriesAdded,
      attachments: m.attachments,
    }))
    await api.updateConversation(convId, { messages: serialized }).catch(() => {})
  }, [])

  const stopGeneration = useCallback(() => {
    const convKey = convIdRef.current || '__new__'
    const abort = abortsRef.current.get(convKey)
    if (abort) {
      abort.abort()
      abortsRef.current.delete(convKey)
    }
  }, [])

  const sendMessage = useCallback(async ({ text, attachedFiles }: SendMessageParams) => {
    if (!text && attachedFiles.length === 0) return

    let convKey = convIdRef.current || '__new__'

    // If THIS conversation is already loading, queue the message
    if (loadingConvIdsRef.current.has(convKey)) {
      const queue = queuesRef.current.get(convKey) || []
      queue.push({ text, attachedFiles })
      queuesRef.current.set(convKey, queue)
      const imageAttachments = attachedFiles
        .filter(f => f.type === 'image')
        .map(f => `data:${f.mediaType || 'image/png'};base64,${f.data}`)
      const queuedMsg: DisplayMessage = {
        role: 'user',
        content: text,
        attachments: attachedFiles.length > 0 ? attachedFiles.map(f => f.name) : undefined,
        imageDataUrls: imageAttachments.length > 0 ? imageAttachments : undefined,
      }
      const updated = [...messagesRef.current, queuedMsg]
      messagesRef.current = updated
      setChatMessages(updated)
      return
    }

    // Check if this was dispatched from the queue (user message already displayed)
    const fromQueue = fromQueueRef.current
    fromQueueRef.current = false

    const mode = modeRef.current
    const config = configRef.current

    // Auto-create conversation if none
    let convId = convIdRef.current
    if (!convId) {
      try {
        const conv = await api.createConversation('New Chat', mode)
        convId = conv.id
        setConversationId(conv.id)
        setConversations(prev => [{ id: conv.id, title: conv.title, mode: conv.mode, created_at: conv.created_at, updated_at: conv.updated_at }, ...prev])
        // Migrate state from __new__ to actual convId
        const existingHistory = apiHistoriesRef.current.get('__new__') || []
        apiHistoriesRef.current.set(conv.id, existingHistory)
        apiHistoriesRef.current.delete('__new__')
        const existingQueue = queuesRef.current.get('__new__')
        if (existingQueue) {
          queuesRef.current.set(conv.id, existingQueue)
          queuesRef.current.delete('__new__')
        }
        convKey = conv.id
      } catch {
        // continue without persistence
      }
    }

    // Build attachments for the API — skip image data when vision is off
    const visionEnabled = (config as any).vision ?? false
    const apiAttachments: FileAttachment[] = attachedFiles
      .filter(f => f.type !== 'image' || visionEnabled)
      .map(f => ({
        type: f.type,
        data: f.data,
        media_type: f.mediaType,
        name: f.name,
      }))

    // Add user message to display if not already shown (queued messages are pre-displayed)
    if (!fromQueue) {
      const imageAttachments = attachedFiles
        .filter(f => f.type === 'image')
        .map(f => `data:${f.mediaType || 'image/png'};base64,${f.data}`)
      const userMsg: DisplayMessage = {
        role: 'user',
        content: text,
        attachments: attachedFiles.length > 0 ? attachedFiles.map(f => f.name) : undefined,
        imageDataUrls: imageAttachments.length > 0 ? imageAttachments : undefined,
      }
      const updated = [...messagesRef.current, userMsg]
      messagesRef.current = updated
      setChatMessages(updated)
    }

    // Persist user message immediately so it survives conversation switches
    if (convId) saveMessages(convId, messagesRef.current)

    // Mark this conversation as loading
    setLoadingConvIds(prev => { const next = new Set(prev); next.add(convKey); return next })

    // Helper: is the user still viewing this conversation?
    const isVisible = () => convIdRef.current === convId

    // Set up streaming state only for visible conversation
    if (isVisible()) {
      streamingConvIdRef.current = convId
      setStreamingContent('')
      setStreamingMemories([])
    }

    // Use per-conversation API history
    const history: ChatMessage[] = [...(apiHistoriesRef.current.get(convKey) || [])]
    const isFirstExchange = history.length === 0

    const useStreaming = mode !== 'tool' && ((config as any).streaming ?? true)

    // Create abort controller for this conversation
    const abort = new AbortController()
    abortsRef.current.set(convKey, abort)

    // Track streamed content locally (accessible in catch block)
    let fullReply = ''

    try {
      if (!useStreaming) {
        // Non-streaming path
        const resp = await api.chat({
          message: text, history, mode, llm: config,
          attachments: apiAttachments.length > 0 ? apiAttachments : undefined,
        })
        fullReply = resp.reply
        const assistantMsg: DisplayMessage = {
          role: 'assistant',
          content: resp.reply,
          memoriesUsed: resp.memories_used,
          memoriesAdded: resp.memories_added,
        }
        if (isVisible()) {
          const saved = [...messagesRef.current, assistantMsg]
          messagesRef.current = saved
          setChatMessages(saved)
          if (convId) await saveMessages(convId, saved)
        } else if (convId) {
          const saved = [...history.map(h => ({ role: h.role as 'user' | 'assistant', content: h.content })), { role: 'user' as const, content: text }, assistantMsg]
          await saveMessages(convId, saved)
        }
        // Update per-conv API history
        const convHistory = apiHistoriesRef.current.get(convKey) || []
        apiHistoriesRef.current.set(convKey, [...convHistory, { role: 'user', content: text }, { role: 'assistant', content: resp.reply }])
      } else {
        // Streaming via SSE
        let memories: MemoryContext[] = []

        await api.chatStream(
          text, history, mode, config, convId || undefined,
          (event: StreamEvent) => {
            switch (event.type) {
              case 'memories':
                memories = event.memories_used
                if (isVisible()) setStreamingMemories(event.memories_used)
                break
              case 'token':
                fullReply += event.content
                if (isVisible()) setStreamingContent(prev => prev + event.content)
                break
              case 'done':
                break
              case 'error':
                fullReply = `Error: ${event.error}`
                break
            }
          },
          apiAttachments.length > 0 ? apiAttachments : undefined,
          abort.signal,
        )

        const assistantMsg: DisplayMessage = {
          role: 'assistant',
          content: fullReply,
          memoriesUsed: memories,
        }
        if (isVisible()) {
          const saved = [...messagesRef.current, assistantMsg]
          messagesRef.current = saved
          setChatMessages(saved)
          if (convId) await saveMessages(convId, saved)
        } else if (convId) {
          const saved = [...history.map(h => ({ role: h.role as 'user' | 'assistant', content: h.content })), { role: 'user' as const, content: text }, assistantMsg]
          await saveMessages(convId, saved)
        }
        // Update per-conv API history
        const convHistory = apiHistoriesRef.current.get(convKey) || []
        apiHistoriesRef.current.set(convKey, [...convHistory, { role: 'user', content: text }, { role: 'assistant', content: fullReply }])

        if (isVisible()) {
          setStreamingContent('')
          setStreamingMemories([])
        }
      }

      // Auto-generate title after first exchange
      if (isFirstExchange && convId) {
        try {
          const { title } = await api.generateTitle(convId, config)
          if (title) {
            setConversations(prev => prev.map(c => c.id === convId ? { ...c, title } : c))
          }
        } catch (e) {
          console.error('Title generation failed:', e)
        }
      }
    } catch (err) {
      // If aborted, save whatever was streamed so far as the assistant message
      if (abort.signal.aborted) {
        const partial = fullReply || (isVisible() ? streamingContentRef.current : '')
        if (partial && isVisible()) {
          const assistantMsg: DisplayMessage = {
            role: 'assistant',
            content: partial,
          }
          const saved = [...messagesRef.current, assistantMsg]
          messagesRef.current = saved
          setChatMessages(saved)
          const convHistory = apiHistoriesRef.current.get(convKey) || []
          apiHistoriesRef.current.set(convKey, [...convHistory, { role: 'user', content: text }, { role: 'assistant', content: partial }])
          if (convId) await saveMessages(convId, saved)
        }
        if (isVisible()) {
          setStreamingContent('')
          setStreamingMemories([])
        }
      } else {
        if (isVisible()) {
          const errorMsg: DisplayMessage = {
            role: 'assistant',
            content: `Error: ${err instanceof Error ? err.message : 'Failed to get response'}`,
          }
          const updated = [...messagesRef.current, errorMsg]
          messagesRef.current = updated
          setChatMessages(updated)
          setStreamingContent('')
        }
      }
    } finally {
      abortsRef.current.delete(convKey)
      if (streamingConvIdRef.current === convId) {
        streamingConvIdRef.current = null
      }
      setLoadingConvIds(prev => { const next = new Set(prev); next.delete(convKey); return next })
    }
  }, [saveMessages, setConversationId, setConversations])

  sendMessageRef.current = sendMessage

  // Initialize API namespace from persisted value on mount
  useEffect(() => {
    setApiNamespace(namespace)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const nsContext = useMemo(() => ({
    namespace,
    setNamespace: (ns: string) => {
      setNamespaceState(ns)
      setApiNamespace(ns)
      try { localStorage.setItem('anima-namespace', ns) } catch { /* ignore */ }
      queryClient.invalidateQueries()
    },
  }), [namespace])

  // Derive visible streaming/loading state for the current conversation
  const visibleLoading = loadingConvIds.has(conversationId || '__new__')
  const isStreamingHere = streamingConvIdRef.current === conversationId
  const visibleStreamingContent = isStreamingHere ? streamingContent : ''
  const visibleStreamingMemories = isStreamingHere ? streamingMemories : []

  const chatContext = useMemo(() => ({
    messages: chatMessages,
    setMessages: setChatMessages,
    mode: chatMode,
    setMode: setChatMode,
    config: chatConfig,
    setConfig: (cfg: LlmConfig) => {
      setChatConfig(cfg)
      localStorage.setItem('anima-llm-config', JSON.stringify(cfg))
    },
    conversationId,
    setConversationId,
    conversations,
    setConversations,
    loading: visibleLoading,
    streamingContent: visibleStreamingContent,
    streamingMemories: visibleStreamingMemories,
    sendMessage,
    stopGeneration,
  }), [chatMessages, chatMode, chatConfig, conversationId, conversations, visibleLoading, visibleStreamingContent, visibleStreamingMemories, sendMessage, stopGeneration])

  return (
    <QueryClientProvider client={queryClient}>
      <NamespaceContext.Provider value={nsContext}>
        <ChatContext.Provider value={chatContext}>
          <BrowserRouter>
            <Layout>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/memories" element={<MemoriesPage />} />
                <Route path="/search" element={<SearchPage />} />
                <Route path="/graph" element={<GraphPage />} />
                <Route path="/graph3d" element={<Graph3DPage />} />
                <Route path="/embeddings" element={<EmbeddingPage />} />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </Layout>
          </BrowserRouter>
        </ChatContext.Provider>
      </NamespaceContext.Provider>
    </QueryClientProvider>
  )
}
