import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import type { ContradictionEntry, SupersessionLink } from '../api/types'
import { GitCompareArrows, ArrowDown } from 'lucide-react'

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

export function ConflictsPage() {
  const [entries, setEntries] = useState<ContradictionEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null)
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

  if (loading) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-ink">Conflicts</h2>
        <p className="text-sm text-ink-muted py-8 text-center">Loading...</p>
      </div>
    )
  }

  // Empty state — no conflicts
  if (!error && entries.length === 0 && offset === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-ink">Conflicts</h2>
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <GitCompareArrows size={32} className="text-ink-faint mb-3" />
          <p className="text-sm text-ink-muted">No conflicts</p>
          <p className="text-xs text-ink-faint mt-1">
            When memories contradict or supersede each other, they'll appear here
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-ink">Conflicts</h2>
        <p className="text-xs text-ink-muted mt-0.5">{entries.length} contradiction{entries.length !== 1 ? 's' : ''}</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="flex gap-6">
        {/* Contradiction list */}
        <div className="flex-1 min-w-0 space-y-1.5">
          {entries.map((e) => (
            <div
              key={e.id}
              className={`bg-card border rounded-lg p-3 cursor-pointer transition-colors ${
                selectedMemoryId === e.new_memory_id || selectedMemoryId === e.old_memory_id
                  ? 'border-accent bg-accent-light/30'
                  : 'border-warm-border hover:border-warm-border-strong'
              }`}
              onClick={() => loadChain(e.new_memory_id)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">
                  {resolutionLabel(e.resolution)}
                </span>
                <span className="text-[10px] text-ink-faint">{relativeDate(e.created_at)}</span>
              </div>
              <div className="space-y-1">
                <div className="flex gap-2 text-sm">
                  <span className="text-red-500 shrink-0 text-[10px] font-medium mt-0.5 w-7">OLD</span>
                  <span className="text-ink-muted line-through decoration-red-300/50">{e.old_content || '[deleted]'}</span>
                </div>
                <div className="flex gap-2 text-sm">
                  <span className="text-green-600 shrink-0 text-[10px] font-medium mt-0.5 w-7">NEW</span>
                  <span className="text-ink">{e.new_content || '[deleted]'}</span>
                </div>
              </div>
            </div>
          ))}

          {/* Pagination */}
          {(offset > 0 || entries.length >= limit) && (
            <div className="flex justify-between items-center pt-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="text-xs text-ink-muted hover:text-ink disabled:opacity-30 transition-colors"
              >
                Previous
              </button>
              <span className="text-xs text-ink-faint tabular-nums">
                {offset + 1}&ndash;{offset + entries.length}
              </span>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={entries.length < limit}
                className="text-xs text-ink-muted hover:text-ink disabled:opacity-30 transition-colors"
              >
                Next
              </button>
            </div>
          )}
        </div>

        {/* Supersession chain — only show when selected */}
        {selectedMemoryId && (
          <div className="w-64 shrink-0">
            <h3 className="text-xs font-medium text-ink-muted uppercase tracking-wider mb-3">History</h3>
            {chainLoading ? (
              <p className="text-xs text-ink-faint">Loading...</p>
            ) : chain.length === 0 ? (
              <p className="text-xs text-ink-faint">No chain found.</p>
            ) : (
              <div className="space-y-0">
                {chain.map((link, i) => (
                  <div key={link.memory_id}>
                    <div
                      className={`border rounded-lg p-2.5 text-xs ${
                        link.status === 'active'
                          ? 'border-green-300 bg-green-50'
                          : 'border-warm-border bg-paper-deep'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={`font-medium ${link.status === 'active' ? 'text-green-700' : 'text-ink-muted'}`}>
                          {link.status === 'active' ? 'Current' : 'Superseded'}
                        </span>
                        <span className="text-ink-faint">{link.source}</span>
                      </div>
                      <p className={`leading-relaxed ${link.status === 'active' ? 'text-ink' : 'text-ink-muted line-through'}`}>
                        {link.content.length > 100 ? link.content.slice(0, 100) + '...' : link.content}
                      </p>
                      <div className="flex items-center justify-between mt-1.5 text-ink-faint">
                        <span>conf {link.confidence.toFixed(2)}</span>
                        <span>{relativeDate(link.created_at)}</span>
                      </div>
                    </div>
                    {i < chain.length - 1 && (
                      <div className="flex justify-center py-1">
                        <ArrowDown size={12} className="text-ink-faint" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
