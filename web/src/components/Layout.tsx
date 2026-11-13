import { useState, useCallback, useEffect, useRef } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { LayoutDashboard, Database, Search, Share2, Network, MessageSquare, Trash2, PanelLeftClose, PanelLeftOpen, ChevronDown, ChevronRight, Settings, Menu, X, Layers, GitCompareArrows, Activity, Sun, Moon, Monitor, Plus, BookOpen, Info } from 'lucide-react'
import { useTheme, setThemePreference, type ThemePreference } from '../lib/theme'
import { AboutAnima, aboutSeen } from './AboutAnima'
import { useChat } from '../hooks/useChat'
import { useNamespace } from '../hooks/useNamespace'
import { useToast } from '../hooks/useToast'
import { api } from '../api/client'
import type { DisplayMessage } from '../hooks/useChat'
import type { NamespaceInfo } from '../api/types'

const navGroups = [
  {
    label: 'Memory',
    items: [
      { path: '/', label: 'Dashboard', icon: LayoutDashboard },
      { path: '/memories', label: 'Memories', icon: Database },
      { path: '/search', label: 'Search', icon: Search },
      { path: '/skills', label: 'Skills', icon: BookOpen },
    ],
  },
  {
    label: 'Explore',
    items: [
      { path: '/graph', label: 'Graph', icon: Share2 },
      { path: '/graph3d', label: 'Embeddings', icon: Network },
      { path: '/conflicts', label: 'Conflicts', icon: GitCompareArrows },
    ],
  },
  {
    label: 'System',
    items: [
      { path: '/processor', label: 'Processor', icon: Activity },
      { path: '/settings', label: 'Settings', icon: Settings },
    ],
  },
]
const navItems = navGroups.flatMap(g => g.items)

const THEME_OPTIONS: { value: ThemePreference; label: string; icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'system', label: 'System', icon: Monitor },
  { value: 'dark', label: 'Dark', icon: Moon },
]

/** Brand mark: an angular "A" sheltering one memory cell. */
function Logo({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true" className="shrink-0 text-accent">
      <path d="M12 2 22 21h-5.2L12 11.6 7.2 21H2L12 2Z" fill="currentColor" />
      <rect x="10" y="16" width="4" height="4" fill="currentColor" />
    </svg>
  )
}

export function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const navigate = useNavigate()
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(true)
  // The introduction opens by itself on a browser's first visit.
  const [aboutOpen, setAboutOpen] = useState(() => !aboutSeen())
  const { preference: themePreference } = useTheme()

  // Give each page its own tab title so history and multiple tabs are usable.
  useEffect(() => {
    const path = location.pathname
    const title = path === '/'
      ? 'Dashboard'
      : path.startsWith('/chat')
        ? 'Chat'
        : [...navItems].sort((a, b) => b.path.length - a.path.length).find(item => item.path !== '/' && path.startsWith(item.path))?.label
    document.title = title ? `${title} · Anima` : 'Anima'
  }, [location.pathname])
  const {
    setMessages, mode, setMode,
    conversationId, setConversationId,
    conversations, setConversations,
  } = useChat()

  const { namespace, setNamespace } = useNamespace()
  const { show: showToast } = useToast()
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])
  const prevFailedRef = useRef<number | null>(null)

  useEffect(() => {
    api.listNamespaces().then(setNamespaces).catch(() => {})
  }, [])

  // Poll processor status for LLM errors
  useEffect(() => {
    const check = async () => {
      try {
        const status = await api.getProcessorStatus()
        const failed = status.metrics.failed_jobs
        if (prevFailedRef.current !== null && failed > prevFailedRef.current) {
          // Fetch latest failed log entry for details
          try {
            const logs = await api.getProcessorLog(5, 0)
            const failedLog = logs.find(l => l.status === 'failed')
            const detail = failedLog?.details
              ? String(typeof failedLog.details === 'object' ? (failedLog.details as any).error || JSON.stringify(failedLog.details) : failedLog.details)
              : 'Unknown error'
            showToast(`Processor error: ${detail}`, true)
          } catch {
            showToast(`Processor: ${failed - prevFailedRef.current} job(s) failed`, true)
          }
        }
        prevFailedRef.current = failed
      } catch { /* server unreachable, ignore */ }
    }
    check()
    const interval = setInterval(check, 15_000)
    return () => clearInterval(interval)
  }, [showToast])

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
      {/* Brand */}
      <div className={`flex items-center h-14 shrink-0 ${showLabels ? 'justify-between px-3' : 'justify-center'}`}>
        {showLabels && (
          <Link to="/" className="flex items-center gap-2.5 rounded-md" aria-label="Anima home">
            <Logo />
            <span className="text-base font-semibold text-ink tracking-wide uppercase font-[family-name:var(--font-display)]">Anima</span>
          </Link>
        )}
        <button
          onClick={() => {
            if (mobileOpen) setMobileOpen(false)
            else setCollapsed(!collapsed)
          }}
          aria-label={mobileOpen ? 'Close menu' : collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={mobileOpen ? 'Close menu' : collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="btn btn-ghost btn-icon"
        >
          {mobileOpen ? <X size={18} /> : collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      {/* Namespace switcher: the scope everything below applies to */}
      {showLabels ? (
        <div className="px-3 pb-3">
          <label htmlFor="namespace-select" className="section-label block mb-1.5">Namespace</label>
          <div className="relative">
            <Layers size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none" />
            <select
              id="namespace-select"
              value={namespace}
              onChange={(e) => setNamespace(e.target.value)}
              className="field pl-8 font-medium"
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
        </div>
      ) : (
        <div className="pb-2 flex justify-center">
          <div title={`Namespace: ${namespace}`} className="w-8 h-8 flex items-center justify-center text-ink-muted">
            <Layers size={16} />
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav aria-label="Main" className="flex-1 px-2 pb-2 overflow-y-auto space-y-4">
        {navGroups.map((group, gi) => (
          <div key={group.label}>
            {showLabels
              ? <p className="section-label px-2 mb-1">{group.label}</p>
              : gi > 0 && <div className="mx-2 mb-2 border-t border-warm-border" />}
            <ul className="space-y-0.5">
              {group.items.map(({ path, label, icon: Icon }) => {
                const active = location.pathname === path
                return (
                  <li key={path}>
                    <Link
                      to={path}
                      aria-current={active ? 'page' : undefined}
                      aria-label={!showLabels ? label : undefined}
                      title={!showLabels ? label : undefined}
                      className={`nav-item ${showLabels ? 'gap-2.5 px-2' : 'justify-center w-8 mx-auto'}`}
                    >
                      <Icon size={16} className="shrink-0" />
                      {showLabels && label}
                    </Link>
                  </li>
                )
              })}
            </ul>

            {/* Chat lives with the memory tools, with its history underneath */}
            {gi === 0 && (
              <div className="mt-0.5">
                <div className="flex items-center gap-0.5">
                  <Link
                    to="/chat"
                    onClick={(e) => { e.preventDefault(); startNewChat() }}
                    aria-current={isOnChat ? 'page' : undefined}
                    aria-label={!showLabels ? 'Chat' : undefined}
                    title={!showLabels ? 'Chat' : undefined}
                    className={`nav-item flex-1 ${showLabels ? 'gap-2.5 px-2' : 'justify-center w-8 mx-auto flex-none'}`}
                  >
                    <MessageSquare size={16} className="shrink-0" />
                    {showLabels && 'Chat'}
                  </Link>
                  {showLabels && (
                    <>
                      <button
                        onClick={startNewChat}
                        aria-label="New chat"
                        title="New chat"
                        className="btn btn-ghost btn-icon"
                      >
                        <Plus size={15} />
                      </button>
                      {conversations.length > 0 && (
                        <button
                          onClick={() => setChatOpen(!chatOpen)}
                          aria-expanded={chatOpen}
                          aria-label={chatOpen ? 'Hide chat history' : 'Show chat history'}
                          title={chatOpen ? 'Hide chat history' : 'Show chat history'}
                          className="btn btn-ghost btn-icon"
                        >
                          {chatOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                        </button>
                      )}
                    </>
                  )}
                </div>

                {showLabels && chatOpen && conversations.length > 0 && (
                  <ul className="ml-4 mt-1 space-y-0.5 border-l border-warm-border pl-2">
                    {conversations.map(conv => (
                      <li key={conv.id} className="group flex items-center">
                        <button
                          onClick={() => { loadConversation(conv.id) }}
                          aria-current={conversationId === conv.id ? 'page' : undefined}
                          className={`flex-1 min-w-0 text-left truncate px-2 py-1.5 rounded-md text-[13px] transition-colors ${
                            conversationId === conv.id
                              ? 'bg-paper-deep text-ink font-medium'
                              : 'text-ink-muted hover:bg-paper-deep hover:text-ink'
                          }`}
                        >
                          {conv.title}
                        </button>
                        <button
                          onClick={e => deleteConv(conv.id, e)}
                          aria-label={`Delete conversation ${conv.title}`}
                          title="Delete conversation"
                          className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 w-6 h-6 flex items-center justify-center rounded-md text-ink-faint hover:text-danger hover:bg-danger-light transition-all shrink-0"
                        >
                          <Trash2 size={13} />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        ))}
      </nav>

      {/* About */}
      <div className={`px-2 pb-2 ${showLabels ? '' : 'flex justify-center'}`}>
        <button
          onClick={() => setAboutOpen(true)}
          aria-label={!showLabels ? 'About Anima' : undefined}
          title={!showLabels ? 'About Anima' : undefined}
          className={`nav-item w-full ${showLabels ? 'gap-2.5 px-2' : 'justify-center w-8'}`}
        >
          <Info size={16} className="shrink-0" />
          {showLabels && 'About Anima'}
        </button>
      </div>

      {/* Theme */}
      <div className={`border-t border-warm-border ${showLabels ? 'p-3' : 'p-2 flex justify-center'}`}>
        {showLabels ? (
          <div className="segmented w-full" role="group" aria-label="Theme">
            {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                onClick={() => setThemePreference(value)}
                aria-pressed={themePreference === value}
                title={`${label} theme`}
                className="flex-1 inline-flex items-center justify-center gap-1.5"
              >
                <Icon size={14} />
                <span className="text-xs">{label}</span>
              </button>
            ))}
          </div>
        ) : (
          <button
            onClick={() => setThemePreference(themePreference === 'dark' ? 'light' : 'dark')}
            aria-label="Toggle dark theme"
            title="Toggle dark theme"
            className="btn btn-ghost btn-icon"
          >
            {themePreference === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        )}
      </div>
    </>
  )

  return (
    <div className="app-shell flex h-screen bg-paper text-ink">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <AboutAnima open={aboutOpen} onClose={() => setAboutOpen(false)} />

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar — desktop */}
      <aside className={`${collapsed ? 'w-14' : 'w-60'} bg-paper-warm border-r border-warm-border flex-col shrink-0 transition-[width] duration-200 hidden md:flex`}>
        {sidebarContent}
      </aside>

      {/* Sidebar — mobile drawer */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] bg-paper-warm border-r border-warm-border flex flex-col transition-transform duration-200 md:hidden ${mobileOpen ? 'translate-x-0 shadow-[var(--shadow-overlay)]' : '-translate-x-full'}`}
        aria-hidden={!mobileOpen}
        inert={!mobileOpen}
      >
        {sidebarContent}
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar */}
        <div className="flex items-center gap-2 px-3 h-14 bg-paper-warm border-b border-warm-border md:hidden shrink-0">
          <button
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
            className="btn btn-ghost btn-icon"
          >
            <Menu size={20} />
          </button>
          <Logo size={22} />
          <span className="text-base font-semibold text-ink tracking-wide uppercase font-[family-name:var(--font-display)]">Anima</span>
        </div>

        <main id="main-content" tabIndex={-1} className="flex-1 overflow-auto bg-paper p-4 sm:p-6 lg:p-8 focus:outline-none">
          {children}
        </main>
      </div>
    </div>
  )
}
