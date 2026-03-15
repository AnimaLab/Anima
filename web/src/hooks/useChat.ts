import { createContext, useContext } from 'react'
import type { LlmConfig, MemoryContext, ConversationSummary, FileAttachment } from '../api/types'

export interface DisplayMessage {
  role: 'user' | 'assistant'
  content: string
  memoriesUsed?: MemoryContext[]
  memoriesAdded?: { id: string; content: string }[]
  attachments?: string[]       // file names attached to this message
  imageDataUrls?: string[]     // base64 data URLs for image previews
}

export interface SendMessageParams {
  text: string
  attachedFiles: {
    name: string
    type: 'text' | 'image'
    data: string
    mediaType?: string
  }[]
}

export interface ChatState {
  messages: DisplayMessage[]
  setMessages: React.Dispatch<React.SetStateAction<DisplayMessage[]>>
  mode: 'rag' | 'tool'
  setMode: (mode: 'rag' | 'tool') => void
  config: LlmConfig
  setConfig: (config: LlmConfig) => void
  conversationId: string | null
  setConversationId: (id: string | null) => void
  conversations: ConversationSummary[]
  setConversations: React.Dispatch<React.SetStateAction<ConversationSummary[]>>

  // Streaming state — lives in context so it survives navigation
  loading: boolean
  streamingContent: string
  streamingMemories: MemoryContext[]

  // Send a message — runs in context, survives page changes
  sendMessage: (params: SendMessageParams) => Promise<void>

  // Stop the current LLM generation
  stopGeneration: () => void
}

export const ChatContext = createContext<ChatState | null>(null)

export function useChat(): ChatState {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within ChatContext.Provider')
  return ctx
}
