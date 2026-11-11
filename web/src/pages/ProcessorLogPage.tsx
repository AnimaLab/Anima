import { useState, useEffect, useCallback, useRef } from 'react'
import { RefreshCw, Activity, AlertTriangle } from 'lucide-react'
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

const pipelineBadge: Record<string, string> = {
  reflection: 'badge-info',
  deduction: 'badge-warning',
  induction: 'badge-success',
  reconsolidation: 'badge-warning',
  retention: 'badge-neutral',
}

const statusStyle: Record<string, { text: string; dot: string }> = {
  completed: { text: 'text-success', dot: 'bg-success' },
  failed: { text: 'text-danger', dot: 'bg-danger' },
  skipped: { text: 'text-ink-faint', dot: 'bg-ink-faint' },
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

/** Human-readable error for a failed run, when the entry carries one. */
function failureDetail(entry: ProcessorLogEntry): string | null {
  if (entry.status !== 'failed' || entry.details == null) return null
  const d = entry.details
  if (typeof d === 'string') return d
  if (typeof d === 'object') {
    const err = (d as Record<string, unknown>).error
    if (typeof err === 'string') return err
    return JSON.stringify(d)
  }
  return String(d)
}

const COLUMNS: { label: string; align: 'left' | 'right' }[] = [
  { label: 'Time', align: 'left' },
  { label: 'Pipeline', align: 'left' },
  { label: 'Namespace', align: 'left' },
  { label: 'Status', align: 'left' },
  { label: 'Input', align: 'right' },
  { label: 'Output', align: 'right' },
  { label: 'Tokens', align: 'right' },
  { label: 'Duration', align: 'right' },
]

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

  const completed = entries.filter(e => e.status === 'completed').length
  const failed = entries.filter(e => e.status === 'failed').length
  const avgMs = entries.length > 0
    ? entries.reduce((sum, e) => sum + e.elapsed_ms, 0) / entries.length
    : 0

  return (
    <div className="page space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="page-title">Processor</h1>
          <p className="page-description">
            Background jobs that maintain memory: reflection, deduction, reconsolidation and retention
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-ink-faint">Refreshes every 10s</span>
          <button
            onClick={() => fetchEntries(true)}
            disabled={refreshing || loading}
            className="btn btn-secondary"
          >
            <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-4" aria-busy="true" aria-label="Loading processor log">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
            {[0, 1, 2, 3].map(i => <div key={i} className="card h-[84px] animate-pulse bg-paper-warm" />)}
          </div>
          <div className="card h-64 animate-pulse bg-paper-warm" />
        </div>
      ) : error && entries.length === 0 ? (
        <div className="card flex flex-col items-center text-center px-6 py-12" role="alert">
          <div className="w-11 h-11 rounded-full bg-danger-light text-danger flex items-center justify-center mb-3">
            <AlertTriangle size={20} />
          </div>
          <p className="text-sm font-medium text-ink">Could not load the processor log</p>
          <p className="text-[13px] text-danger mt-1 break-words max-w-full">{error}</p>
          <button onClick={() => fetchEntries(true)} disabled={refreshing} className="btn btn-secondary mt-4">
            <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} /> Retry
          </button>
        </div>
      ) : entries.length === 0 ? (
        <div className="card flex flex-col items-center justify-center py-16 px-6 text-center">
          <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
            <Activity size={20} />
          </div>
          <p className="text-sm font-medium text-ink">No processor activity yet</p>
          <p className="text-[13px] text-ink-muted mt-1 max-w-sm">
            Runs appear here when reflection, deduction or retention processes memories in the background
          </p>
        </div>
      ) : (
        <>
          {error && (
            <div className="card border-danger/40 px-4 py-3 text-[13px] text-danger flex items-center gap-2" role="alert">
              <AlertTriangle size={15} className="shrink-0" />
              <span className="min-w-0 break-words">Last refresh failed: {error}</span>
            </div>
          )}

          {/* Summary of the runs listed below */}
          <dl className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
            <Stat label="Runs shown" value={entries.length} sub="Most recent first" />
            <Stat label="Completed" value={completed} />
            <Stat label="Failed" value={failed} tone={failed > 0 ? 'danger' : undefined} />
            <Stat label="Average duration" value={formatDuration(avgMs)} />
          </dl>

          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">Recent processor runs</caption>
                <thead>
                  <tr className="border-b border-warm-border bg-paper-warm">
                    {COLUMNS.map(col => (
                      <th
                        key={col.label}
                        scope="col"
                        className={`px-4 py-2.5 text-xs font-semibold text-ink-faint uppercase tracking-wide whitespace-nowrap ${col.align === 'right' ? 'text-right' : 'text-left'}`}
                      >
                        {col.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-warm-border">
                  {entries.map(entry => {
                    const status = statusStyle[entry.status] ?? { text: 'text-ink-muted', dot: 'bg-ink-faint' }
                    const detail = failureDetail(entry)
                    return (
                      <tr key={entry.id} className="hover:bg-paper/60 transition-colors align-top">
                        <td className="px-4 h-11 py-3 text-[13px] text-ink-muted whitespace-nowrap" title={entry.created_at}>
                          {timeAgo(entry.created_at)}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`badge ${pipelineBadge[entry.pipeline] ?? 'badge-neutral'}`}>
                            {entry.pipeline}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs text-ink-light font-mono truncate max-w-[160px]" title={entry.namespace}>
                          {entry.namespace}
                        </td>
                        <td className="px-4 py-3 min-w-[120px]">
                          <span className={`inline-flex items-center gap-1.5 text-[13px] font-medium capitalize ${status.text}`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} aria-hidden="true" />
                            {entry.status}
                          </span>
                          {detail && (
                            <p className="mt-1 text-xs text-danger break-words max-w-xs" title={detail}>
                              {detail.length > 160 ? detail.slice(0, 160) + '…' : detail}
                            </p>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right text-[13px] text-ink-light tabular-nums">
                          {entry.input_count > 0 ? entry.input_count : '—'}
                        </td>
                        <td className="px-4 py-3 text-right text-[13px] text-ink-light tabular-nums">
                          {entry.output_count > 0 ? entry.output_count : '—'}
                        </td>
                        <td className="px-4 py-3 text-right text-[13px] text-ink-light tabular-nums">
                          {entry.total_tokens > 0 ? entry.total_tokens.toLocaleString() : '—'}
                        </td>
                        <td className="px-4 py-3 text-right text-[13px] text-ink-muted tabular-nums whitespace-nowrap">
                          {formatDuration(entry.elapsed_ms)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, sub, tone }: {
  label: string
  value: string | number
  sub?: string
  tone?: 'danger'
}) {
  return (
    <div className="card p-4">
      <dt className="text-[13px] font-medium text-ink-muted">{label}</dt>
      <dd className={`mt-1.5 text-2xl leading-8 font-semibold tracking-tight tabular-nums ${tone === 'danger' ? 'text-danger' : 'text-ink'}`}>
        {value}
      </dd>
      {sub && <p className="text-xs text-ink-faint mt-0.5">{sub}</p>}
    </div>
  )
}
