import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import type { ContradictionEntry, SupersessionLink } from '../api/types'

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
      setError(e instanceof Error ? e.message : 'Failed to load contradictions')
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

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    } catch { return iso }
  }

  const resolutionLabel = (r: string) => {
    const map: Record<string, string> = {
      user_correction_supersede: 'User correction',
      reconsolidation_supersede: 'Reconsolidation',
    }
    return map[r] || r.replace(/_/g, ' ')
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-ink">Conflict Inspector</h2>
        <p className="text-sm text-ink-muted mt-1">
          Supersession history &mdash; when one memory replaces another due to corrections or reconsolidation.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="flex gap-6">
        {/* Left: contradiction list */}
        <div className="flex-1 min-w-0 space-y-2">
          {loading ? (
            <div className="text-sm text-ink-muted py-8 text-center">Loading...</div>
          ) : entries.length === 0 ? (
            <div className="text-sm text-ink-muted py-8 text-center">No contradictions found in this namespace.</div>
          ) : (
            <>
              {entries.map((e) => (
                <div
                  key={e.id}
                  className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                    selectedMemoryId === e.new_memory_id || selectedMemoryId === e.old_memory_id
                      ? 'border-accent bg-accent-light/30'
                      : 'border-warm-border hover:border-accent/50'
                  }`}
                  onClick={() => loadChain(e.new_memory_id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">
                      {resolutionLabel(e.resolution)}
                    </span>
                    <span className="text-xs text-ink-muted">{formatDate(e.created_at)}</span>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex gap-2 text-sm">
                      <span className="text-red-400 shrink-0 font-mono text-xs mt-0.5">OLD</span>
                      <span className="text-ink-light line-through decoration-red-300">{e.old_content || '[deleted]'}</span>
                    </div>
                    <div className="flex gap-2 text-sm">
                      <span className="text-green-500 shrink-0 font-mono text-xs mt-0.5">NEW</span>
                      <span className="text-ink">{e.new_content || '[deleted]'}</span>
                    </div>
                  </div>
                </div>
              ))}

              {/* Pagination */}
              <div className="flex justify-between items-center pt-2">
                <button
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                  disabled={offset === 0}
                  className="text-sm text-accent disabled:text-ink-faint disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <span className="text-xs text-ink-muted">
                  {offset + 1}&ndash;{offset + entries.length}
                </span>
                <button
                  onClick={() => setOffset(offset + limit)}
                  disabled={entries.length < limit}
                  className="text-sm text-accent disabled:text-ink-faint disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </>
          )}
        </div>

        {/* Right: supersession chain */}
        <div className="w-72 shrink-0">
          <h3 className="text-sm font-medium text-ink mb-3">Supersession Chain</h3>
          {!selectedMemoryId ? (
            <p className="text-xs text-ink-muted">Click a contradiction to see the full chain.</p>
          ) : chainLoading ? (
            <p className="text-xs text-ink-muted">Loading...</p>
          ) : chain.length === 0 ? (
            <p className="text-xs text-ink-muted">No chain found.</p>
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
                    <p className={`${link.status === 'active' ? 'text-ink' : 'text-ink-muted line-through'}`}>
                      {link.content.length > 120 ? link.content.slice(0, 120) + '...' : link.content}
                    </p>
                    <div className="flex items-center justify-between mt-1.5 text-ink-faint">
                      <span>conf: {link.confidence.toFixed(2)}</span>
                      <span>{formatDate(link.created_at)}</span>
                    </div>
                  </div>
                  {i < chain.length - 1 && (
                    <div className="flex justify-center py-1">
                      <svg width="12" height="16" viewBox="0 0 12 16" className="text-ink-muted">
                        <path d="M6 0 L6 12 L2 8 M6 12 L10 8" fill="none" stroke="currentColor" strokeWidth="1.5" />
                      </svg>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
