import { useState, useEffect, useCallback, useRef } from 'react'
import { RefreshCw, Activity } from 'lucide-react'
import { api } from '../api/client'

type ProcessorLogEntry = {
  id: string
  namespace: string
  pipeline: string
  status: string
  input_count: number
  output_count: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  elapsed_ms: number
  details: unknown
  created_at: string
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

const pipelineColors: Record<string, string> = {
  reflection: 'bg-blue-100 text-blue-800',
  deduction: 'bg-purple-100 text-purple-800',
  induction: 'bg-green-100 text-green-800',
  reconsolidation: 'bg-amber-100 text-amber-800',
  retention: 'bg-stone-100 text-stone-800',
}

const statusColors: Record<string, string> = {
  completed: 'text-green-700',
  failed: 'text-red-700',
  skipped: 'text-ink-faint',
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function ProcessorLogPage() {
  const [entries, setEntries] = useState<ProcessorLogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchEntries = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) setRefreshing(true)
    setError(null)
    try {
      const data = await api.getProcessorLog(50, 0)
      setEntries(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchEntries()
    intervalRef.current = setInterval(() => fetchEntries(), 10000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchEntries])

  if (loading) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-ink">Processor Log</h2>
        <p className="text-sm text-ink-muted py-8 text-center">Loading...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-ink">Processor Log</h2>
        <p className="text-sm text-red-600 py-8 text-center">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity size={20} className="text-ink-muted" />
          <h2 className="text-lg font-semibold text-ink">Processor Log</h2>
          <span className="text-xs text-ink-faint">auto-refreshes every 10s</span>
        </div>
        <button
          onClick={() => fetchEntries(true)}
          disabled={refreshing}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-ink-light hover:text-ink border border-warm-border rounded-lg hover:bg-paper-deep transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {entries.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Activity size={32} className="text-ink-faint mb-3" />
          <p className="text-sm text-ink-muted">No processor activity yet</p>
          <p className="text-xs text-ink-faint mt-1">Run reflection, deduction, or retention to see entries here</p>
        </div>
      ) : (
        <div className="bg-card border border-warm-border rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-warm-border bg-paper-deep">
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Time</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Pipeline</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Namespace</th>
                  <th className="text-left px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Status</th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Input</th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Output</th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Tokens</th>
                  <th className="text-right px-4 py-2.5 text-xs font-medium text-ink-muted uppercase tracking-wide">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-warm-border">
                {entries.map(entry => (
                  <tr key={entry.id} className="hover:bg-paper-deep/50 transition-colors">
                    <td className="px-4 py-3 text-xs text-ink-muted whitespace-nowrap" title={entry.created_at}>
                      {timeAgo(entry.created_at)}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${pipelineColors[entry.pipeline] ?? 'bg-stone-100 text-stone-700'}`}>
                        {entry.pipeline}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-ink-light font-mono truncate max-w-[120px]" title={entry.namespace}>
                      {entry.namespace}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs font-medium ${statusColors[entry.status] ?? 'text-ink-muted'}`}>
                        {entry.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-light tabular-nums">
                      {entry.input_count > 0 ? entry.input_count : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-light tabular-nums">
                      {entry.output_count > 0 ? entry.output_count : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-light tabular-nums">
                      {entry.total_tokens > 0 ? entry.total_tokens.toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-ink-muted tabular-nums whitespace-nowrap">
                      {formatDuration(entry.elapsed_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
