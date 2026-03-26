import { useState, useRef, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Send, Brain, Wrench, ChevronDown, ChevronUp, ChevronRight, Sparkles, X, Copy, Check, Paperclip, File as FileIcon, Image as ImageIcon, ArrowDown, Square, RefreshCw, Loader2, User } from 'lucide-react'
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
  story: 'bg-rose-500/15 text-rose-700',
  reflection: 'bg-indigo-500/15 text-indigo-700',
  context: 'bg-stone-500/15 text-stone-600',
  goal: 'bg-emerald-500/15 text-emerald-700',
  relationship: 'bg-pink-500/15 text-pink-700',
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

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-3rem)] md:max-h-[calc(100vh-3rem)]">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-warm-border shrink-0">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-ink truncate">
              {conversationId
                ? conversations.find(c => c.id === conversationId)?.title || 'Chat'
                : 'New Chat'}
            </h2>
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
                className="text-ink-faint hover:text-ink-muted transition-colors shrink-0"
                title="Regenerate title"
              >
                {regenningTitle ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              </button>
            )}
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            {models.length > 0 ? (
              <select
                value={config.model}
                onChange={e => setConfig({ ...config, model: e.target.value })}
                className="bg-transparent text-xs text-ink-muted hover:text-ink-light cursor-pointer focus:outline-none appearance-none pr-4 max-w-[240px] truncate"
                style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0 center' }}
              >
                {!models.find(m => m.id === config.model) && (
                  <option value={config.model}>{config.model}</option>
                )}
                {models.map(m => (
                  <option key={m.id} value={m.id}>{m.id}</option>
                ))}
              </select>
            ) : (
              <span className="text-xs text-ink-muted truncate max-w-[240px]">{config.model}</span>
            )}
            <button
              onClick={fetchModels}
              disabled={loadingModels}
              className="text-ink-faint hover:text-ink-muted transition-colors shrink-0"
              title="Refresh models"
            >
              {loadingModels ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 cursor-pointer" title="Auto-retrieve relevant memories">
              <input type="checkbox" checked={features.recall}
                onChange={e => setFeatures({ ...features, recall: e.target.checked })}
                className="sr-only peer" />
              <div className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                features.recall ? 'bg-accent text-white' : 'bg-paper-deep text-ink-muted'
              }`}>
                <Brain size={13} />
                Recall
              </div>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer" title="Let the model use memory tools">
              <input type="checkbox" checked={features.tools}
                onChange={e => setFeatures({ ...features, tools: e.target.checked })}
                className="sr-only peer" />
              <div className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                features.tools ? 'bg-[#8B7DB8] text-white' : 'bg-paper-deep text-ink-muted'
              }`}>
                <Wrench size={13} />
                Tools
              </div>
            </label>
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => { setMessages([]); setConversationId(null) }}
              className="p-2 text-ink-muted hover:bg-paper-deep hover:text-ink rounded-lg transition-colors"
              title="Clear chat"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto mt-4 space-y-4 min-h-0 relative pr-2" ref={scrollContainerRef} onScroll={handleScroll}>
        {messages.length === 0 && !streamingContent && (
          <div className="flex flex-col items-center justify-center h-full text-ink-faint">
            <Sparkles size={32} className="mb-3 text-ink-faint" />
            <p className="text-base font-medium text-ink-muted">What's on your mind?</p>
            <p className="text-sm text-ink-faint mt-1">
              {features.recall && features.tools
                ? 'Memories are recalled and the model can actively search and store'
                : features.recall
                ? 'Your memories are searched automatically to help the conversation'
                : features.tools
                ? 'The model can search and store memories using tools'
                : 'Plain chat — no memory features enabled'}
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {/* Streaming message */}
        {streamingContent && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-accent-light flex items-center justify-center shrink-0">
              <Brain size={16} className="text-accent" />
            </div>
            <div className="max-w-[90%] sm:max-w-[75%]">
              <div className="bg-card border border-warm-border rounded-lg px-4 py-2.5 text-sm leading-relaxed text-ink">
                {streamingSegments.length > 1 ? (
                  <SegmentedContent segments={streamingSegments.filter(s => s.type === 'action' || (s.type === 'text' && s.content))} />
                ) : (
                  <AssistantContent content={streamingContent} />
                )}
              </div>
              {streamingMemories.length > 0 && (
                <p className="text-xs text-ink-faint mt-1">{streamingMemories.length} memories used</p>
              )}
            </div>
          </div>
        )}

        {loading && !streamingContent && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-accent-light flex items-center justify-center shrink-0">
              <Brain size={16} className="text-accent" />
            </div>
            <div className="bg-card border border-warm-border rounded-lg px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-ink-faint rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Scroll to bottom button */}
      {!isScrollPinned && (messages.length > 0 || streamingContent) && (
        <div className="flex justify-center -mt-5 mb-1 relative z-10">
          <button
            onClick={() => { setIsScrollPinned(true); scrollToBottom() }}
            className="flex items-center gap-1 px-3 py-1.5 bg-paper-deep border border-warm-border-strong text-ink-light text-xs rounded-full hover:bg-paper-deep hover:text-ink transition-colors shadow-lg"
          >
            <ArrowDown size={12} />
            Scroll to bottom
          </button>
        </div>
      )}

      {/* Input */}
      <div className="py-3 shrink-0">
        {/* Attached files */}
        {attachedFiles.length > 0 && (
          <div className="mb-2 px-1">
            <div className="flex flex-wrap gap-1.5">
              {attachedFiles.map((file, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1.5 bg-paper-deep border border-warm-border-strong text-ink-light text-xs px-2.5 py-1 rounded-md"
                >
                  {file.previewUrl ? (
                    <img src={file.previewUrl} alt="" className="w-6 h-6 rounded object-cover shrink-0" />
                  ) : file.type === 'image' ? (
                    <ImageIcon size={12} className="text-ink-muted shrink-0" />
                  ) : (
                    <FileIcon size={12} className="text-ink-muted shrink-0" />
                  )}
                  <span className="truncate max-w-[200px]">{file.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-ink-muted hover:text-ink-light transition-colors"
                  >
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
            {!(config.vision ?? false) && attachedFiles.some(f => f.type === 'image') && (
              <p className="text-[11px] text-amber-600 mt-1.5">
                Vision is off — images won't be sent to the model. Enable it in Settings.
              </p>
            )}
          </div>
        )}
        <div className="flex gap-2 items-end bg-card border border-warm-border rounded-lg p-2">
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
            className="p-2 text-ink-muted hover:text-ink hover:bg-paper-deep rounded-lg transition-colors shrink-0"
            title="Attach files"
          >
            <Paperclip size={16} />
          </button>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={loading ? "Type to queue a message..." : "Type a message..."}
            rows={1}
            className="flex-1 bg-transparent text-ink text-sm resize-none focus:outline-none max-h-32 py-1.5 px-2"
            style={{ minHeight: '2rem' }}
          />
          {loading && (
            <button
              onClick={stopGeneration}
              className="p-2 bg-[#C25B4E] hover:bg-[#B5503F] text-white rounded-lg transition-colors shrink-0"
              title="Stop generating"
            >
              <Square size={14} />
            </button>
          )}
          <button
            onClick={handleSend}
            disabled={!input.trim() && attachedFiles.length === 0}
            className="p-2 bg-accent hover:bg-accent-hover disabled:bg-paper-deep disabled:text-ink-faint text-white rounded-lg transition-colors shrink-0"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
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
    <div className="relative group my-2">
      <div className="flex items-center justify-between bg-paper-deep rounded-t-lg px-3 py-1 text-xs text-ink-muted">
        <span>{language || 'code'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-ink transition-colors"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{ margin: 0, borderTopLeftRadius: 0, borderTopRightRadius: 0, fontSize: '0.8rem' }}
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
    <div className="my-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink-light transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="font-medium">Thinking</span>
        {!open && <span className="text-ink-faint truncate max-w-[300px]">{content.slice(0, 80)}{content.length > 80 ? '...' : ''}</span>}
      </button>
      {open && (
        <div className="mt-1.5 pl-4 border-l-2 border-warm-border-strong text-xs text-ink-muted leading-relaxed whitespace-pre-wrap">
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
            <code className="bg-paper-deep text-accent px-1.5 py-0.5 rounded text-xs" {...props}>
              {children}
            </code>
          )
        },
        p({ children }) {
          return <p className="mb-2 last:mb-0">{children}</p>
        },
        ul({ children }) {
          return <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
        },
        ol({ children }) {
          return <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
        },
        h1({ children }) {
          return <h1 className="text-lg font-bold mb-2">{children}</h1>
        },
        h2({ children }) {
          return <h2 className="text-base font-bold mb-2">{children}</h2>
        },
        h3({ children }) {
          return <h3 className="text-sm font-bold mb-1">{children}</h3>
        },
        blockquote({ children }) {
          return <blockquote className="border-l-2 border-warm-border-strong pl-3 italic text-ink-muted my-2">{children}</blockquote>
        },
        a({ href, children }) {
          return <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">{children}</a>
        },
        table({ children }) {
          return <table className="border-collapse my-2 text-xs w-full">{children}</table>
        },
        th({ children }) {
          return <th className="border border-warm-border px-2 py-1 bg-paper-deep text-left font-medium">{children}</th>
        },
        td({ children }) {
          return <td className="border border-warm-border px-2 py-1">{children}</td>
        },
        hr() {
          return <hr className="border-warm-border my-3" />
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
  return (
    <div className="py-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-ink-faint hover:text-ink-muted transition-colors"
      >
        <span>{icons[segment.tool || ''] || '⚙️'}</span>
        <span>{segment.summary}</span>
        {segment.details && segment.details.length > 0 && (
          expanded ? <ChevronUp size={10} /> : <ChevronDown size={10} />
        )}
      </button>
      {expanded && segment.details && (
        <div className="mt-1.5 ml-5 p-2 bg-paper-deep rounded-lg text-[11px] text-ink-muted space-y-1 max-h-40 overflow-y-auto">
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
            <div key={i} className={i > 0 ? 'border-t border-warm-border pt-3' : ''}>
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
  const hasMemoryInfo =
    (message.memoriesUsed && message.memoriesUsed.length > 0) ||
    (message.memoriesAdded && message.memoriesAdded.length > 0)

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
          isUser ? 'bg-paper-deep' : 'bg-accent-light'
        }`}
      >
        {isUser ? (
          <User size={14} className="text-ink-light" />
        ) : (
          <Brain size={16} className="text-accent" />
        )}
      </div>
      <div className={`max-w-[90%] sm:max-w-[75%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`rounded-lg px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? 'bg-accent text-white'
              : 'bg-card border border-warm-border text-ink'
          }`}
        >
          {isUser ? (
            <>
              {message.imageDataUrls && message.imageDataUrls.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {message.imageDataUrls.map((url, i) => (
                    <img key={i} src={url} alt="" className="max-w-[200px] max-h-[150px] rounded object-contain" />
                  ))}
                </div>
              )}
              {message.attachments && message.attachments.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1.5">
                  {message.attachments
                    .filter(name => !/\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(name))
                    .map((name, i) => (
                      <span key={i} className="inline-flex items-center gap-1 bg-accent/20 text-accent text-xs px-2 py-0.5 rounded">
                        <FileIcon size={10} />
                        {name}
                      </span>
                    ))}
                </div>
              )}
              {message.content && <p className="whitespace-pre-wrap">{message.content}</p>}
            </>
          ) : (
            <>
              {message.segments && message.segments.length > 1 ? (
                <SegmentedContent segments={message.segments} />
              ) : (
                <AssistantContent content={message.content} />
              )}
            </>
          )}
        </div>

        {hasMemoryInfo && (
          <div className="mt-1">
            <button
              onClick={() => setShowMemories(!showMemories)}
              className="flex items-center gap-1 text-xs text-ink-faint hover:text-ink-muted transition-colors"
            >
              {showMemories ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              {message.memoriesUsed?.length || 0} memories used
              {message.memoriesAdded && message.memoriesAdded.length > 0 &&
                `, ${message.memoriesAdded.length} added`}
            </button>

            {showMemories && (
              <div className="mt-2 p-3 bg-card/80 border border-warm-border rounded-lg space-y-2 text-left">
                {message.memoriesUsed && message.memoriesUsed.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-ink-muted mb-1">Memories Retrieved</p>
                    {message.memoriesUsed.map((m, i) => (
                      <div key={i} className="flex items-start gap-2 text-xs text-ink-muted py-1">
                        <span className="text-accent/60 shrink-0">{m.score.toFixed(3)}</span>
                        {m.memory_type && (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full shrink-0 ${typeColor[m.memory_type] || typeColor.fact}`}>
                            {m.memory_type}
                          </span>
                        )}
                        <span>{m.content}</span>
                      </div>
                    ))}
                  </div>
                )}
                {message.memoriesAdded && message.memoriesAdded.length > 0 && (
                  <div>
                    <p className="text-xs font-medium text-ink-muted mb-1">Memories Stored</p>
                    {message.memoriesAdded.map((m, i) => (
                      <p key={i} className="text-xs text-[#5B8C5A] py-0.5">+ {typeof m === 'string' ? m : m.content}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
