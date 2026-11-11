import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import type { ContradictionEntry, SupersessionLink } from '../api/types'
import { GitCompareArrows, ArrowRight, ArrowDown, ChevronDown, ChevronRight, AlertTriangle, RefreshCw } from 'lucide-react'
import { sourceLabel } from '../lib/memoryMeta'

function relativeDate(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

const resolutionLabel = (r: string) => {
  const map: Record<string, string> = {
    user_correction_supersede: 'User correction',
    reconsolidation_supersede: 'Reconsolidation',
  }
  return map[r] || r.replace(/_/g, ' ')
}

function Header({ count }: { count?: number }) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2.5">
        <h1 className="page-title">Conflicts</h1>
        {count !== undefined && count > 0 && (
          <span className="badge badge-neutral tabular-nums">{count} on this page</span>
        )}
      </div>
      <p className="page-description">
        The ledger of memories that replaced or contradicted an earlier one, with what changed and why
      </p>
    </div>
  )
}

export function ConflictsPage() {
  const [entries, setEntries] = useState<ContradictionEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null)
  const [expandedEntryId, setExpandedEntryId] = useState<string | null>(null)
  const [chain, setChain] = useState<SupersessionLink[]>([])
  const [chainLoading, setChainLoading] = useState(false)
  const [offset, setOffset] = useState(0)
  const limit = 30

  const loadContradictions = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.listContradictions(limit, offset)
      setEntries(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [offset])

  useEffect(() => { loadContradictions() }, [loadContradictions])

  const loadChain = useCallback(async (memoryId: string) => {
    setSelectedMemoryId(memoryId)
    setChainLoading(true)
    try {
      const data = await api.getMemoryHistory(memoryId)
      setChain(data)
    } catch {
      setChain([])
    } finally {
      setChainLoading(false)
    }
  }, [])

  const toggleHistory = (entry: ContradictionEntry) => {
    if (expandedEntryId === entry.id) {
      setExpandedEntryId(null)
      setSelectedMemoryId(null)
      return
    }
    setExpandedEntryId(entry.id)
    loadChain(entry.new_memory_id)
  }

  if (loading) {
    return (
      <div className="page space-y-5">
        <Header />
        <div className="space-y-3" aria-busy="true" aria-label="Loading conflicts">
          {[0, 1, 2].map(i => <div key={i} className="card h-[132px] animate-pulse bg-paper-warm" />)}
        </div>
      </div>
    )
  }

  if (error && entries.length === 0) {
    return (
      <div className="page space-y-5">
        <Header />
        <div className="card flex flex-col items-center text-center px-6 py-12" role="alert">
          <div className="w-11 h-11 rounded-full bg-danger-light text-danger flex items-center justify-center mb-3">
            <AlertTriangle size={20} />
          </div>
          <p className="text-sm font-medium text-ink">Could not load conflicts</p>
          <p className="text-[13px] text-danger mt-1 break-words max-w-full">{error}</p>
          <button onClick={() => loadContradictions()} className="btn btn-secondary mt-4">
            <RefreshCw size={15} /> Retry
          </button>
        </div>
      </div>
    )
  }

  // Empty state — no conflicts
  if (entries.length === 0 && offset === 0) {
    return (
      <div className="page space-y-5">
        <Header />
        <div className="card flex flex-col items-center justify-center py-16 px-6 text-center">
          <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
            <GitCompareArrows size={20} />
          </div>
          <p className="text-sm font-medium text-ink">No conflicts</p>
          <p className="text-[13px] text-ink-muted mt-1 max-w-sm">
            When a new memory corrects or supersedes an older one, both versions are kept and shown here
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="page space-y-5">
      <Header count={entries.length} />

      {error && (
        <div className="card border-danger/40 px-4 py-3 text-[13px] text-danger flex items-center gap-2" role="alert">
          <AlertTriangle size={15} className="shrink-0" />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      )}

      <ul className="space-y-3">
        {entries.map((e) => {
          const expanded = expandedEntryId === e.id
          return (
            <li key={e.id} className={`card overflow-hidden ${expanded ? 'border-warm-border-strong' : ''}`}>
              {/* Card header */}
              <div className="flex flex-wrap items-center justify-between gap-2 px-4 sm:px-5 py-2.5 border-b border-warm-border">
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="badge badge-warning">{resolutionLabel(e.resolution)}</span>
                  <span className="text-xs text-ink-faint" title={e.created_at}>{relativeDate(e.created_at)}</span>
                </div>
                <button
                  onClick={() => toggleHistory(e)}
                  aria-expanded={expanded}
                  className="btn btn-ghost -mr-2"
                >
                  {expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                  History
                </button>
              </div>

              {/* Before / after */}
              <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-stretch">
                <div className="bg-danger-light/50 px-4 sm:px-5 py-3.5">
                  <p className="section-label mb-1.5">Replaced</p>
                  <p className="text-sm text-ink-muted leading-relaxed line-through decoration-danger/40 break-words">
                    {e.old_content || '[deleted]'}
                  </p>
                </div>
                <div className="flex items-center justify-center px-1 py-1 md:py-0 text-ink-faint" aria-hidden="true">
                  <ArrowRight size={16} className="hidden md:block" />
                  <ArrowDown size={16} className="md:hidden" />
                </div>
                <div className="bg-success-light/50 px-4 sm:px-5 py-3.5">
                  <p className="section-label mb-1.5">Current</p>
                  <p className="text-sm text-ink leading-relaxed break-words">
                    {e.new_content || '[deleted]'}
                  </p>
                </div>
              </div>

              {/* Supersession chain */}
              {expanded && (
                <div className="border-t border-warm-border px-4 sm:px-5 py-4">
                  <p className="section-label mb-3">History of this memory</p>
                  {chainLoading ? (
                    <div className="space-y-2" aria-busy="true">
                      {[0, 1].map(i => <div key={i} className="h-10 rounded-lg bg-paper-warm animate-pulse" />)}
                    </div>
                  ) : chain.length === 0 || selectedMemoryId !== e.new_memory_id ? (
                    <p className="text-[13px] text-ink-muted">No earlier versions found.</p>
                  ) : (
                    <ol>
                      {chain.map((link, i) => {
                        const active = link.status === 'active'
                        return (
                          <li key={link.memory_id} className="relative flex gap-3 pb-4 last:pb-0">
                            {/* Timeline rail */}
                            <div className="flex flex-col items-center shrink-0 pt-1.5" aria-hidden="true">
                              <span className={`w-2.5 h-2.5 rounded-full border-2 ${active ? 'bg-success border-success' : 'bg-card border-warm-border-strong'}`} />
                              {i < chain.length - 1 && <span className="flex-1 w-px bg-warm-border mt-1" />}
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                                <span className={`badge ${active ? 'badge-success' : 'badge-neutral'}`}>
                                  {active ? 'Current' : 'Superseded'}
                                </span>
                                <span className="text-xs text-ink-faint">
                                  {sourceLabel(link.source)} · {Math.round(link.confidence * 100)}% confidence · {relativeDate(link.created_at)}
                                </span>
                              </div>
                              <p
                                className={`mt-1 text-[13px] leading-relaxed break-words ${active ? 'text-ink' : 'text-ink-muted line-through'}`}
                                title={link.content}
                              >
                                {link.content.length > 240 ? link.content.slice(0, 240) + '…' : link.content}
                              </p>
                            </div>
                          </li>
                        )
                      })}
                    </ol>
                  )}
                </div>
              )}
            </li>
          )
        })}
      </ul>

      {/* Pagination */}
      {(offset > 0 || entries.length >= limit) && (
        <nav aria-label="Pagination" className="flex items-center gap-3 justify-center pt-1">
          <button
            onClick={() => setOffset(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="btn btn-secondary"
          >
            Previous
          </button>
          <span className="text-[13px] text-ink-muted tabular-nums">
            {entries.length > 0 ? <>{offset + 1}&ndash;{offset + entries.length}</> : 'No more entries'}
          </span>
          <button
            onClick={() => setOffset(offset + limit)}
            disabled={entries.length < limit}
            className="btn btn-secondary"
          >
            Next
          </button>
        </nav>
      )}
    </div>
  )
}
