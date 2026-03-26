import { useState, useCallback, useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { LayoutDashboard, Database, Search, Share2, Network, MessageSquare, Trash2, PanelLeftClose, PanelLeftOpen, ChevronDown, ChevronRight, Settings, Menu, X, Layers, Box, GitCompareArrows, Activity } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { useNamespace } from '../hooks/useNamespace'
import { api } from '../api/client'
import type { DisplayMessage } from '../hooks/useChat'
import type { NamespaceInfo } from '../api/types'

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/memories', label: 'Memories', icon: Database },
  { path: '/search', label: 'Search', icon: Search },
  { path: '/graph', label: 'Graph', icon: Share2 },
  { path: '/graph3d', label: 'Embeddings', icon: Network },
  { path: '/conflicts', label: 'Conflicts', icon: GitCompareArrows },
  { path: '/processor', label: 'Processor', icon: Activity },
]

export function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(true)
  const {
    setMessages, mode, setMode,
    conversationId, setConversationId,
    conversations, setConversations,
  } = useChat()

  const { namespace, setNamespace } = useNamespace()
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])

  useEffect(() => {
    api.listNamespaces().then(setNamespaces).catch(() => {})
  }, [])

  const isOnChat = location.pathname === '/chat' || location.pathname.startsWith('/chat/')

  // Close mobile sidebar on navigation
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  // Load conversations on mount
  useEffect(() => {
    api.listConversations().then(setConversations).catch(() => {})
  }, [setConversations])

  const startNewChat = useCallback(() => {
    setConversationId(null)
    setMessages([])
    navigate('/chat')
  }, [setConversationId, setMessages, navigate])

  const loadConversation = useCallback(async (id: string) => {
    try {
      const conv = await api.getConversation(id)
      setConversationId(conv.id)
      const parsed: DisplayMessage[] = JSON.parse(conv.messages)
      setMessages(parsed)
      setMode(conv.mode as 'rag' | 'tool')
      navigate(`/chat/${conv.id}`)
    } catch {
      // ignore
    }
  }, [setConversationId, setMessages, setMode, navigate])

  const deleteConv = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await api.deleteConversation(id).catch(() => {})
    setConversations(prev => prev.filter(c => c.id !== id))
    if (conversationId === id) {
      setConversationId(null)
      setMessages([])
    }
  }, [conversationId, setConversationId, setMessages, setConversations])

  // Whether to show labels (expanded desktop or mobile drawer)
  const showLabels = !collapsed || mobileOpen

  const sidebarContent = (
    <>
      {/* Header */}
      <div className={`flex items-center ${showLabels ? 'justify-between p-3' : 'justify-center p-2'} border-b border-warm-border`}>
        {showLabels ? (
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-bold text-ink tracking-tight">Anima</h1>
          </div>
        ) : (
          <span className="text-sm font-bold text-ink">A</span>
        )}
        <button
          onClick={() => {
            if (mobileOpen) setMobileOpen(false)
            else setCollapsed(!collapsed)
          }}
          className={`text-ink-muted hover:text-ink transition-colors ${showLabels ? 'p-1' : 'w-8 h-8 flex items-center justify-center'}`}
        >
          {mobileOpen ? <X size={18} /> : collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-1.5 space-y-0.5 overflow-y-auto">
        {navItems.map(({ path, label, icon: Icon }) => {
          const active = location.pathname === path
          return (
            <Link
              key={path}
              to={path}
              title={!showLabels ? label : undefined}
              className={`flex items-center py-2 rounded-lg text-sm transition-colors ${
                showLabels ? 'gap-3 px-3' : 'justify-center w-9 mx-auto'
              } ${
                active
                  ? 'bg-accent-light text-accent'
                  : 'text-ink-light hover:bg-paper-deep hover:text-ink'
              }`}
            >
              <Icon size={18} className="shrink-0" />
              {showLabels && label}
            </Link>
          )
        })}

        {/* Chat nav item + expandable history */}
        <div>
          <button
            onClick={(e) => {
              e.preventDefault()
              startNewChat()
            }}
            title={!showLabels ? 'Chat' : undefined}
            className={`w-full flex items-center py-2 rounded-lg text-sm transition-colors ${
              showLabels ? 'gap-3 px-3' : 'justify-center w-9 mx-auto'
            } ${
              isOnChat
                ? 'bg-accent-light text-accent'
                : 'text-ink-light hover:bg-paper-deep hover:text-ink'
            }`}
          >
            <MessageSquare size={18} className="shrink-0" />
            {showLabels && <>
              <span className="flex-1 text-left">Chat</span>
              <span
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setChatOpen(!chatOpen) }}
                className="p-0.5 text-ink-muted hover:text-ink transition-colors"
              >
                {chatOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </span>
            </>}
          </button>

          {/* Chat history list */}
          {showLabels && chatOpen && (
            <div className="ml-3 mt-1 space-y-0.5 border-l border-warm-border pl-2">
              {conversations.map(conv => (
                <div
                  key={conv.id}
                  className={`group flex items-center gap-1.5 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                    conversationId === conv.id
                      ? 'bg-paper-deep text-ink'
                      : 'text-ink-muted hover:bg-paper-deep/50 hover:text-ink-light'
                  }`}
                  onClick={() => { loadConversation(conv.id) }}
                >
                  <span className="truncate flex-1">{conv.title}</span>
                  <button
                    onClick={e => deleteConv(conv.id, e)}
                    className="opacity-0 group-hover:opacity-100 text-ink-faint hover:text-red-600 transition-all shrink-0"
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </nav>

      {/* Namespace selector */}
      {showLabels ? (
        <div className="px-3 py-2 border-t border-warm-border">
          <label className="text-xs text-ink-muted block mb-1">Namespace</label>
          <select
            value={namespace}
            onChange={(e) => setNamespace(e.target.value)}
            className="w-full bg-input border border-warm-border rounded-md px-2 py-1.5 text-sm text-ink focus:outline-none focus:ring-1 focus:ring-accent"
          >
            {namespaces.length === 0 && (
              <option value={namespace}>{namespace}</option>
            )}
            {namespaces.map((ns) => (
              <option key={ns.namespace} value={ns.namespace}>
                {ns.namespace} ({ns.active_count})
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="border-t border-warm-border py-2 flex justify-center">
          <div title={`Namespace: ${namespace}`} className="w-9 flex items-center justify-center text-ink-muted">
            <Layers size={16} />
          </div>
        </div>
      )}

      {/* Settings */}
      <div className="p-1.5 border-t border-warm-border">
        <Link
          to="/settings"
          title={!showLabels ? 'Settings' : undefined}
          className={`flex items-center py-2 rounded-lg text-sm transition-colors ${
            showLabels ? 'gap-3 px-3' : 'justify-center w-9 mx-auto'
          } ${
            location.pathname === '/settings'
              ? 'bg-accent-light text-accent'
              : 'text-ink-light hover:bg-paper-deep hover:text-ink'
          }`}
        >
          <Settings size={18} className="shrink-0" />
          {showLabels && 'Settings'}
        </Link>
      </div>
    </>
  )

  return (
    <div className="flex h-screen">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-ink/30 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar — desktop */}
      <aside className={`${collapsed ? 'w-12' : 'w-56'} bg-paper-warm border-r border-warm-border flex-col shrink-0 transition-all duration-200 hidden md:flex`}>
        {sidebarContent}
      </aside>

      {/* Sidebar — mobile drawer */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-paper-warm border-r border-warm-border flex flex-col transition-transform duration-200 md:hidden ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        {sidebarContent}
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar */}
        <div className="flex items-center gap-3 px-4 py-2 bg-paper-warm border-b border-warm-border md:hidden shrink-0">
          <button
            onClick={() => setMobileOpen(true)}
            className="p-1 text-ink-muted hover:text-ink transition-colors"
          >
            <Menu size={20} />
          </button>
          <h1 className="text-sm font-bold text-ink">Anima</h1>
        </div>

        <main className="flex-1 overflow-auto bg-paper p-4 sm:p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
