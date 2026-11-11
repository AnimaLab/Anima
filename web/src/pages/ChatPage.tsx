import { useState, useRef, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Send, Brain, Wrench, ChevronDown, ChevronUp, ChevronRight, Sparkles, X, Copy, Check, Paperclip, File as FileIcon, Image as ImageIcon, ArrowDown, Square, RefreshCw, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import * as pdfjsLib from 'pdfjs-dist'
import { useChat, type DisplayMessage, type ChatSegment } from '../hooks/useChat'
import { api } from '../api/client'

const typeColor: Record<string, string> = {
  preference: 'bg-purple-500/15 text-purple-700',
  fact: 'bg-blue-500/15 text-blue-700',
  event: 'bg-cyan-500/15 text-cyan-700',
  decision: 'bg-orange-500/15 text-orange-700',
  story: 'bg-orange-500/15 text-orange-700',
  reflection: 'bg-indigo-500/15 text-indigo-700',
  context: 'bg-stone-500/15 text-stone-600',
  goal: 'bg-emerald-500/15 text-emerald-700',
  relationship: 'bg-amber-500/15 text-amber-700',
  emotion: 'bg-red-500/15 text-red-700',
  habit: 'bg-teal-500/15 text-teal-700',
  belief: 'bg-amber-500/15 text-amber-700',
  skill: 'bg-lime-500/15 text-lime-700',
  location: 'bg-sky-500/15 text-sky-700',
}

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.mjs',
  import.meta.url,
).toString()

export function ChatPage() {
  const { id: urlConvId } = useParams<{ id?: string }>()
  const navigate = useNavigate()
  const {
    messages, setMessages, features, setFeatures, config, setConfig,
    conversationId, setConversationId, conversations, setConversations,
    loading, streamingContent, streamingMemories, streamingSegments,
    sendMessage: contextSendMessage,
    stopGeneration,
  } = useChat()
  const [regenningTitle, setRegenningTitle] = useState(false)
  const [input, setInput] = useState('')

  // Load conversation from URL param on mount
  useEffect(() => {
    if (urlConvId && urlConvId !== conversationId) {
      api.getConversation(urlConvId).then(conv => {
        setConversationId(conv.id)
        try {
          const parsed: DisplayMessage[] = JSON.parse(conv.messages)
          if (parsed.length > 0) setMessages(parsed)
        } catch { /* ignore */ }
      }).catch(() => {
        // Invalid ID — redirect to new chat
        navigate('/chat', { replace: true })
      })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlConvId])

  // Reset title regen state when switching conversations
  useEffect(() => { setRegenningTitle(false) }, [conversationId])
  const [models, setModels] = useState<{ id: string }[]>([])
  const [loadingModels, setLoadingModels] = useState(false)

  const fetchModels = useCallback(async () => {
    const baseUrl = config.base_url?.trim()
    if (!baseUrl) return
    setLoadingModels(true)
    try {
      const url = `${baseUrl.replace(/\/+$/, '')}/models`
      const headers: Record<string, string> = {}
      if (config.api_key) headers['Authorization'] = `Bearer ${config.api_key}`
      const res = await fetch(url, { headers })
      if (!res.ok) throw new Error(`${res.status}`)
      const json = await res.json()
      setModels((json.data || []).map((m: { id: string }) => ({ id: m.id })))
    } catch {
      setModels([])
    } finally {
      setLoadingModels(false)
    }
  }, [config.base_url, config.api_key])

  useEffect(() => { fetchModels() }, [fetchModels])
  const [attachedFiles, setAttachedFiles] = useState<{
    name: string
    type: 'text' | 'image'
    data: string          // text content or base64 data
    mediaType?: string    // MIME type for images
    previewUrl?: string   // object URL for image thumbnails
  }[]>([])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isScrollPinned, setIsScrollPinned] = useState(true)
  const convSwitchRef = useRef(false)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  // Mark conversation switches so we can scroll instantly instead of smooth
  useEffect(() => {
    convSwitchRef.current = true
  }, [conversationId])

  // Auto-scroll only when pinned — instant on conversation switch, smooth for new messages
  useEffect(() => {
    if (isScrollPinned) {
      const behavior = convSwitchRef.current ? 'instant' as const : 'smooth' as const
      convSwitchRef.current = false
      messagesEndRef.current?.scrollIntoView({ behavior })
    }
  }, [messages, streamingContent, isScrollPinned])

  // Track scroll position to pin/unpin
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    // Pin when within 80px of bottom
    setIsScrollPinned(distanceFromBottom < 80)
  }, [])

  // Focus input when loading finishes
  useEffect(() => {
    if (!loading) inputRef.current?.focus()
  }, [loading])

  const readFileAsBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        resolve(result.split(',')[1] || '')
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  const extractPdfText = async (file: File): Promise<string> => {
    const buffer = await file.arrayBuffer()
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise
    const pages: string[] = []
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i)
      const content = await page.getTextContent()
      const text = content.items
        .map((item) => ('str' in item ? item.str : ''))
        .join(' ')
      if (text.trim()) pages.push(text)
    }
    return pages.join('\n\n')
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    for (const file of Array.from(files)) {
      try {
        if (file.type.startsWith('image/')) {
          const base64 = await readFileAsBase64(file)
          const previewUrl = URL.createObjectURL(file)
          setAttachedFiles(prev => [...prev, {
            name: file.name,
            type: 'image',
            data: base64,
            mediaType: file.type,
            previewUrl,
          }])
        } else if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
          const text = await extractPdfText(file)
          setAttachedFiles(prev => [...prev, {
            name: file.name,
            type: 'text',
            data: text,
          }])
        } else {
          const content = await file.text()
          setAttachedFiles(prev => [...prev, {
            name: file.name,
            type: 'text',
            data: content,
          }])
        }
      } catch {
        // skip files that can't be read
      }
    }
    e.target.value = ''
  }

  const removeFile = (index: number) => {
    setAttachedFiles(prev => {
      const file = prev[index]
      if (file.previewUrl) URL.revokeObjectURL(file.previewUrl)
      return prev.filter((_, i) => i !== index)
    })
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text && attachedFiles.length === 0) return

    // Prepare files for context-level send (strip previewUrls)
    const filesToSend = attachedFiles.map(f => ({
      name: f.name,
      type: f.type,
      data: f.data,
      mediaType: f.mediaType,
    }))

    setInput('')
    // Clean up preview URLs before clearing
    attachedFiles.forEach(f => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl) })
    setAttachedFiles([])

    // Delegate to context — this survives navigation
    await contextSendMessage({ text, attachedFiles: filesToSend })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const hasConversation = messages.length > 0 || !!streamingContent
  const title = conversationId
    ? conversations.find(c => c.id === conversationId)?.title || 'Chat'
    : 'New chat'
  const featureHint = features.recall && features.tools
    ? 'Memories are recalled automatically, and the model can search and store them itself.'
    : features.recall
    ? 'Your memories are searched automatically to help the conversation.'
    : features.tools
    ? 'The model can search and store memories using tools.'
    : 'Plain chat: no memory features are enabled.'

  const applySuggestion = (text: string) => {
    setInput(text)
    inputRef.current?.focus()
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 pb-3 border-b border-warm-border shrink-0">
        <div className="flex items-center gap-1 min-w-0">
          <h1 className="page-title truncate">{title}</h1>
          {conversationId && messages.length > 0 && (
            <button
              onClick={async () => {
                if (regenningTitle) return
                setRegenningTitle(true)
                try {
                  const { title } = await api.generateTitle(conversationId, config)
                  if (title) {
                    setConversations(prev => prev.map(c => c.id === conversationId ? { ...c, title } : c))
                  }
                } catch { /* ignore */ }
                setRegenningTitle(false)
              }}
              className="btn btn-ghost btn-icon shrink-0"
              title="Regenerate title"
              aria-label="Regenerate title"
            >
              {regenningTitle ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
            </button>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 min-w-0">
          {/* Model */}
          <div className="flex items-center gap-0.5 min-w-0">
            {models.length > 0 ? (
              <select
                value={config.model}
                onChange={e => setConfig({ ...config, model: e.target.value })}
                aria-label="Model"
                title="Model"
                className="field w-auto max-w-[200px] sm:max-w-[240px] truncate"
              >
                {!models.find(m => m.id === config.model) && (
                  <option value={config.model}>{config.model}</option>
                )}
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
              </select>
            ) : (
              <span className="text-[13px] text-ink-muted font-mono truncate max-w-[200px] sm:max-w-[240px] px-1" title="Model">
                {config.model || 'No model set'}
              </span>
            )}
            <button
              onClick={fetchModels}
              disabled={loadingModels}
              className="btn btn-ghost btn-icon shrink-0"
              title="Refresh model list"
              aria-label="Refresh model list"
            >
              {loadingModels ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
            </button>
          </div>

          {/* Memory features: independent toggles */}
          <div className="segmented" role="group" aria-label="Memory features">
            <button
              type="button"
              onClick={() => setFeatures({ ...features, recall: !features.recall })}
              aria-pressed={features.recall}
              title="Automatically retrieve relevant memories for each message"
              className="inline-flex items-center gap-1.5"
            >
              <Brain size={14} /> Recall
            </button>
            <button
              type="button"
              onClick={() => setFeatures({ ...features, tools: !features.tools })}
              aria-pressed={features.tools}
              title="Let the model search and store memories with tools"
              className="inline-flex items-center gap-1.5"
            >
              <Wrench size={14} /> Tools
            </button>
          </div>

          {messages.length > 0 && (
            <button
              onClick={() => { setMessages([]); setConversationId(null) }}
              className="btn btn-ghost btn-icon"
              title="Clear chat"
              aria-label="Clear chat"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div className="relative flex-1 min-h-0">
        <div
          className="h-full overflow-y-auto"
          ref={scrollContainerRef}
          onScroll={handleScroll}
          role="log"
          aria-label="Conversation"
        >
          <div className={`max-w-3xl mx-auto px-1 ${hasConversation ? 'py-6 space-y-6' : 'h-full'}`}>
            {!hasConversation && (
              <div className="flex flex-col items-center justify-center h-full text-center py-8">
                <div className="w-12 h-12 rounded-full bg-accent-light text-accent flex items-center justify-center mb-4">
                  <Sparkles size={22} />
                </div>
                <p className="text-lg font-semibold text-ink tracking-tight">What's on your mind?</p>
                <p className="text-sm text-ink-muted mt-1 max-w-md">{featureHint}</p>
                <div className="flex flex-wrap justify-center gap-2 mt-5">
                  {['What do you remember about me?', 'Summarise what I am working on', 'Remember that I prefer short answers'].map(s => (
                    <button key={s} type="button" onClick={() => applySuggestion(s)} className="btn btn-secondary rounded-full font-normal">
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}

            {/* Streaming message */}
            {streamingContent && (
              <AssistantRow>
                <div className="text-[15px] leading-relaxed text-ink">
                  {streamingSegments.length > 1 ? (
                    <SegmentedContent segments={streamingSegments.filter(s => s.type === 'action' || (s.type === 'text' && s.content))} />
                  ) : (
                    <AssistantContent content={streamingContent} />
                  )}
                </div>
                {streamingMemories.length > 0 && (
                  <p className="text-xs text-ink-faint mt-2">{streamingMemories.length} memories used</p>
                )}
              </AssistantRow>
            )}

            {loading && !streamingContent && (
              <AssistantRow>
                <div className="flex items-center gap-1 h-6" role="status" aria-label="Generating a reply">
                  <span className="w-1.5 h-1.5 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </AssistantRow>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Scroll to bottom */}
        {!isScrollPinned && hasConversation && (
          <div className="absolute bottom-3 inset-x-0 flex justify-center pointer-events-none">
            <button
              onClick={() => { setIsScrollPinned(true); scrollToBottom() }}
              className="btn btn-secondary rounded-full pointer-events-auto shadow-[var(--shadow-overlay)]"
            >
              <ArrowDown size={14} />
              Scroll to bottom
            </button>
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="shrink-0 pt-3">
        <div className="max-w-3xl mx-auto">
          <div className="card rounded-xl p-2 focus-within:border-accent transition-colors">
            {/* Attached files */}
            {attachedFiles.length > 0 && (
              <div className="px-1.5 pt-1 pb-2">
                <ul className="flex flex-wrap gap-1.5">
                  {attachedFiles.map((file, i) => (
                    <li
                      key={i}
                      className="inline-flex items-center gap-1.5 bg-paper-deep text-ink-light text-[13px] pl-2 pr-0.5 py-0.5 rounded-md max-w-full"
                    >
                      {file.previewUrl ? (
                        <img src={file.previewUrl} alt="" className="w-6 h-6 rounded object-cover shrink-0" />
                      ) : file.type === 'image' ? (
                        <ImageIcon size={14} className="text-ink-muted shrink-0" />
                      ) : (
                        <FileIcon size={14} className="text-ink-muted shrink-0" />
                      )}
                      <span className="truncate max-w-[200px]">{file.name}</span>
                      <button
                        onClick={() => removeFile(i)}
                        className="w-6 h-6 flex items-center justify-center rounded text-ink-muted hover:text-ink hover:bg-paper-warm transition-colors shrink-0"
                        title={`Remove ${file.name}`}
                        aria-label={`Remove ${file.name}`}
                      >
                        <X size={13} />
                      </button>
                    </li>
                  ))}
                </ul>
                {!(config.vision ?? false) && attachedFiles.some(f => f.type === 'image') && (
                  <p className="text-xs text-warning mt-2">
                    Vision is off, so images won't be sent to the model. Enable it in Settings.
                  </p>
                )}
              </div>
            )}

            <textarea
              ref={inputRef}
              value={input}
              onChange={e => {
                setInput(e.target.value)
                // Auto-resize to content, max 17 lines
                const el = e.target
                el.style.height = 'auto'
                const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 20
                const maxHeight = lineHeight * 17
                el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
              }}
              onKeyDown={handleKeyDown}
              placeholder={loading ? 'Type to queue a message…' : 'Message Anima…'}
              aria-label="Message"
              rows={1}
              className="block w-full bg-transparent text-ink text-[15px] leading-6 resize-none px-2 py-1.5 placeholder:text-ink-faint focus:outline-none focus-visible:outline-none"
              style={{ minHeight: '2.25rem' }}
            />

            <div className="flex items-center justify-between gap-2 pt-1">
              <div className="flex items-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="text/*,application/json,application/pdf,image/*,.csv,.md,.yaml,.yml,.toml,.xml,.html,.css,.js,.ts,.py,.rs,.go,.java,.c,.cpp,.h,.rb,.sh"
                  className="hidden"
                  onChange={handleFileSelect}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-ghost btn-icon"
                  title="Attach files"
                  aria-label="Attach files"
                >
                  <Paperclip size={16} />
                </button>
              </div>
              <div className="flex items-center gap-1.5">
                {loading && (
                  <button
                    onClick={stopGeneration}
                    className="btn btn-danger btn-icon"
                    title="Stop generating"
                    aria-label="Stop generating"
                  >
                    <Square size={14} />
                  </button>
                )}
                <button
                  onClick={handleSend}
                  disabled={!input.trim() && attachedFiles.length === 0}
                  className="btn btn-primary btn-icon"
                  title={loading ? 'Queue message' : 'Send message'}
                  aria-label={loading ? 'Queue message' : 'Send message'}
                >
                  <Send size={15} />
                </button>
              </div>
            </div>
          </div>
          <p className="hidden sm:block text-xs text-ink-faint text-center mt-2">
            Enter to send · Shift + Enter for a new line
          </p>
        </div>
      </div>
    </div>
  )
}

/** Assistant turn: a small marker in the gutter, content across the full column. */
function AssistantRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="w-7 h-7 rounded-full bg-accent-light text-accent flex items-center justify-center shrink-0 mt-0.5" aria-hidden="true">
        <Brain size={14} />
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  )
}

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative group my-3 rounded-lg overflow-hidden border border-warm-border">
      <div className="flex items-center justify-between bg-paper-deep pl-3 pr-1 text-xs text-ink-muted">
        <span>{language || 'code'}</span>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 min-h-8 px-2 rounded-md hover:text-ink transition-colors"
          aria-live="polite"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, borderRadius: 0, fontSize: '0.8125rem' }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  )
}

/** Parse think blocks out of content, returning alternating segments. */
function parseThinkBlocks(content: string): { type: 'text' | 'think'; content: string }[] {
  const parts: { type: 'text' | 'think'; content: string }[] = []
  const regex = /<think>([\s\S]*?)(<\/think>|$)/gi
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      const text = content.slice(lastIndex, match.index).trim()
      if (text) parts.push({ type: 'text', content: text })
    }
    const thinkContent = match[1].trim()
    if (thinkContent) parts.push({ type: 'think', content: thinkContent })
    lastIndex = regex.lastIndex
  }

  if (lastIndex < content.length) {
    const text = content.slice(lastIndex).trim()
    if (text) parts.push({ type: 'text', content: text })
  }

  return parts.length > 0 ? parts : [{ type: 'text', content }]
}

function ThinkBlock({ content }: { content: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="my-2 border border-warm-border rounded-lg text-[13px]">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex items-center gap-1.5 w-full min-h-8 px-2.5 text-left text-ink-muted hover:text-ink transition-colors rounded-lg"
      >
        {open ? <ChevronDown size={14} className="shrink-0" /> : <ChevronRight size={14} className="shrink-0" />}
        <span className="font-medium shrink-0">Thinking</span>
        {!open && <span className="text-ink-faint truncate min-w-0">{content.slice(0, 80)}{content.length > 80 ? '…' : ''}</span>}
      </button>
      {open && (
        <div className="px-3 pb-2.5 pt-0.5 text-ink-muted leading-relaxed whitespace-pre-wrap break-words">
          {content}
        </div>
      )}
    </div>
  )
}

function AssistantContent({ content }: { content: string }) {
  const parts = parseThinkBlocks(content)
  return (
    <>
      {parts.map((part, i) =>
        part.type === 'think'
          ? <ThinkBlock key={i} content={part.content} />
          : <MarkdownContent key={i} content={part.content} />
      )}
    </>
  )
}

function MarkdownContent({ content }: { content: string }) {
  // Clean up stray bullet characters (•) on their own lines that some LLMs produce
  const cleaned = content.replace(/^[•·]\s*$/gm, '')
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '')
          const codeStr = String(children).replace(/\n$/, '')
          if (match) {
            return <CodeBlock language={match[1]}>{codeStr}</CodeBlock>
          }
          return (
            <code className="bg-paper-deep text-ink px-1.5 py-0.5 rounded text-[13px] break-words" {...props}>
              {children}
            </code>
          )
        },
        p({ children }) {
          return <p className="mb-3 last:mb-0">{children}</p>
        },
        ul({ children }) {
          return <ul className="list-disc pl-5 mb-3 last:mb-0 space-y-1">{children}</ul>
        },
        ol({ children }) {
          return <ol className="list-decimal pl-5 mb-3 last:mb-0 space-y-1">{children}</ol>
        },
        h1({ children }) {
          return <h2 className="text-lg font-semibold tracking-tight mt-4 first:mt-0 mb-2">{children}</h2>
        },
        h2({ children }) {
          return <h3 className="text-base font-semibold mt-4 first:mt-0 mb-2">{children}</h3>
        },
        h3({ children }) {
          return <h4 className="text-[15px] font-semibold mt-3 first:mt-0 mb-1">{children}</h4>
        },
        blockquote({ children }) {
          return <blockquote className="border-l-2 border-warm-border-strong pl-3 text-ink-muted my-3">{children}</blockquote>
        },
        a({ href, children }) {
          return <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-2 hover:text-accent-hover">{children}</a>
        },
        table({ children }) {
          return <div className="my-3 overflow-x-auto"><table className="border-collapse text-[13px] w-full">{children}</table></div>
        },
        th({ children }) {
          return <th className="border border-warm-border px-2.5 py-1.5 bg-paper-deep text-left font-medium">{children}</th>
        },
        td({ children }) {
          return <td className="border border-warm-border px-2.5 py-1.5">{children}</td>
        },
        hr() {
          return <hr className="border-warm-border my-4" />
        },
        pre({ children }) {
          return <>{children}</>
        },
      }}
    >
      {cleaned}
    </ReactMarkdown>
  )
}

function ActionIndicator({ segment }: { segment: ChatSegment }) {
  const [expanded, setExpanded] = useState(false)
  const icons: Record<string, string> = {
    memory_search: '🔍',
    memory_add: '💾',
    memory_update: '✏️',
    send_message: '💬',
  }
  const hasDetails = !!segment.details && segment.details.length > 0
  return (
    <div className="border border-warm-border rounded-lg text-[13px]">
      <button
        onClick={() => setExpanded(!expanded)}
        aria-expanded={hasDetails ? expanded : undefined}
        className="flex items-center gap-2 w-full min-h-8 px-2.5 text-left text-ink-muted hover:text-ink transition-colors rounded-lg"
      >
        <span aria-hidden="true">{icons[segment.tool || ''] || '⚙️'}</span>
        <span className="flex-1 min-w-0 truncate">{segment.summary}</span>
        {hasDetails && (
          expanded ? <ChevronUp size={14} className="shrink-0" /> : <ChevronDown size={14} className="shrink-0" />
        )}
      </button>
      {expanded && segment.details && (
        <div className="px-3 pb-2.5 text-xs text-ink-muted space-y-1 max-h-40 overflow-y-auto">
          {segment.details.map((d, i) => (
            <div key={i} className="truncate">{typeof d === 'string' ? d : JSON.stringify(d)}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function SegmentedContent({ segments }: { segments: ChatSegment[] }) {
  return (
    <div className="space-y-3">
      {segments.map((seg, i) => {
        if (seg.type === 'action') {
          return <ActionIndicator key={i} segment={seg} />
        }
        if (seg.type === 'text' && seg.content) {
          return (
            <div key={i}>
              <AssistantContent content={seg.content} />
            </div>
          )
        }
        return null
      })}
    </div>
  )
}

function MessageBubble({ message }: { message: DisplayMessage }) {
  const [showMemories, setShowMemories] = useState(false)
  const isUser = message.role === 'user'
  const usedCount = message.memoriesUsed?.length || 0
  const addedCount = message.memoriesAdded?.length || 0
  const hasMemoryInfo = usedCount > 0 || addedCount > 0

  if (isUser) {
    const fileNames = (message.attachments ?? []).filter(name => !/\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(name))
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] sm:max-w-[80%] bg-paper-deep text-ink rounded-2xl rounded-br-md px-4 py-2.5 text-[15px] leading-relaxed">
          {message.imageDataUrls && message.imageDataUrls.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {message.imageDataUrls.map((url, i) => (
                <img key={i} src={url} alt="" className="max-w-[200px] max-h-[150px] rounded-lg object-contain" />
              ))}
            </div>
          )}
          {fileNames.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-1.5">
              {fileNames.map((name, i) => (
                <span key={i} className="inline-flex items-center gap-1 bg-card text-ink-light text-xs px-2 py-0.5 rounded-md max-w-full">
                  <FileIcon size={12} className="shrink-0 text-ink-muted" />
                  <span className="truncate">{name}</span>
                </span>
              ))}
            </div>
          )}
          {message.content && <p className="whitespace-pre-wrap break-words">{message.content}</p>}
        </div>
      </div>
    )
  }

  return (
    <AssistantRow>
      <div className="text-[15px] leading-relaxed text-ink break-words">
        {message.segments && message.segments.length > 1 ? (
          <SegmentedContent segments={message.segments} />
        ) : (
          <AssistantContent content={message.content} />
        )}
      </div>

      {hasMemoryInfo && (
        <div className="mt-3 border border-warm-border rounded-lg text-[13px]">
          <button
            onClick={() => setShowMemories(!showMemories)}
            aria-expanded={showMemories}
            className="flex items-center gap-1.5 w-full min-h-8 px-2.5 text-left text-ink-muted hover:text-ink transition-colors rounded-lg"
          >
            {showMemories ? <ChevronUp size={14} className="shrink-0" /> : <ChevronDown size={14} className="shrink-0" />}
            <Brain size={13} className="shrink-0 text-ink-faint" />
            <span>
              {usedCount} {usedCount === 1 ? 'memory' : 'memories'} used
              {addedCount > 0 && `, ${addedCount} added`}
            </span>
          </button>

          {showMemories && (
            <div className="px-3 pb-3 pt-1 space-y-3">
              {usedCount > 0 && (
                <div>
                  <p className="section-label mb-1.5">Retrieved</p>
                  <ul className="space-y-1.5">
                    {message.memoriesUsed!.map((m, i) => (
                      <li key={i} className="flex items-baseline gap-2 text-ink-light">
                        <span className="text-xs text-ink-muted font-mono tabular-nums shrink-0">{m.score.toFixed(3)}</span>
                        {m.memory_type && m.memory_type !== 'raw' && (
                          <span className={`badge shrink-0 ${typeColor[m.memory_type] || 'badge-neutral'}`}>
                            {m.memory_type}
                          </span>
                        )}
                        <span className="min-w-0 break-words">{m.content}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {addedCount > 0 && (
                <div>
                  <p className="section-label mb-1.5">Stored</p>
                  <ul className="space-y-1">
                    {message.memoriesAdded!.map((m, i) => (
                      <li key={i} className="flex items-baseline gap-2 text-success">
                        <span aria-hidden="true">+</span>
                        <span className="min-w-0 break-words">{typeof m === 'string' ? m : m.content}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </AssistantRow>
  )
}
