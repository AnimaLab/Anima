import { useQuery } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { useNamespace } from '../hooks/useNamespace'
import { Database, Clock, TrendingUp, Layers } from 'lucide-react'
import type { NamespaceStats, NamespaceInfo, Memory } from '../api/types'

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return 'never'
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

const STATUS_COLORS: Record<string, string> = {
  active: '#5B8C5A',
  superseded: '#C49A3C',
  deleted: '#C25B4E',
}

const TYPE_INFO: Record<string, { color: string; label: string; desc: string }> = {
  raw:       { color: '#9C9488', label: 'Raw',       desc: 'Original input' },
  reflected: { color: '#8B7DB8', label: 'Reflected', desc: 'Extracted facts' },
  deduced:   { color: '#5B9CA6', label: 'Deduced',   desc: 'Inferred' },
  induced:   { color: '#5BA88C', label: 'Induced',   desc: 'Synthesized' },
}

export function DashboardPage() {
  const { namespace } = useNamespace()

  const { data: stats } = useQuery({
    queryKey: ['stats', getNamespace()],
    queryFn: () => api.getStats(),
    refetchInterval: 30000,
  })
  const { data: namespaces } = useQuery({
    queryKey: ['namespaces'],
    queryFn: () => api.listNamespaces(),
  })
  // Fetch memory type counts via listing (small queries)
  const { data: allMemories } = useQuery({
    queryKey: ['memories-types', getNamespace()],
    queryFn: () => api.listMemories(0, 1, 'active', undefined),
    staleTime: 30000,
  })
  // Fetch top accessed
  const { data: topAccessed } = useQuery({
    queryKey: ['top-accessed', 'most', getNamespace()],
    queryFn: () => api.topAccessed('most', 5),
  })
  // Fetch type breakdown from graph data (has memory_type per node)
  const { data: graphData } = useQuery({
    queryKey: ['graph', getNamespace(), 0.99],
    queryFn: () => api.getGraph(0.99, 500),
    staleTime: 60000,
  })

  const typeCounts = graphData
    ? graphData.nodes.reduce((acc, n) => {
        acc[n.memory_type] = (acc[n.memory_type] || 0) + 1
        return acc
      }, {} as Record<string, number>)
    : {}

  const total = stats?.total ?? 0
  const hasAccess = topAccessed && topAccessed.some(m => m.access_count > 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-ink">Dashboard</h2>
        <p className="text-sm text-ink-muted mt-0.5">
          {total > 0
            ? <>{total} memories in <span className="font-medium text-ink-light">{namespace}</span></>
            : <>No memories yet in <span className="font-medium text-ink-light">{namespace}</span></>
          }
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard icon={Database} label="Total" value={stats?.total ?? '-'} />
        <StatCard
          icon={TrendingUp}
          label="Active"
          value={stats?.active ?? '-'}
          sub={stats && stats.superseded > 0 ? `${stats.superseded} superseded` : undefined}
        />
        <StatCard icon={Clock} label="Newest" value={stats ? timeAgo(stats.newest_memory) : '-'} />
        <StatCard icon={Layers} label="Namespaces" value={namespaces?.length ?? '-'} />
      </div>

      {/* Middle row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Memory Breakdown */}
        <div className="bg-card border border-warm-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-ink-light mb-4">Memory Breakdown</h3>

          {/* Status bar */}
          {stats && total > 0 && (
            <div className="mb-5">
              <p className="text-xs text-ink-muted mb-2">By status</p>
              <div className="flex h-2.5 rounded-full overflow-hidden bg-paper-deep">
                {(['active', 'superseded', 'deleted'] as const).map(s => {
                  const val = stats[s]
                  if (!val) return null
                  return (
                    <div
                      key={s}
                      className="h-full first:rounded-l-full last:rounded-r-full"
                      style={{ width: `${(val / total) * 100}%`, background: STATUS_COLORS[s] }}
                    />
                  )
                })}
              </div>
              <div className="flex gap-4 mt-2">
                {(['active', 'superseded', 'deleted'] as const).map(s => {
                  const val = stats[s]
                  if (!val) return null
                  return (
                    <div key={s} className="flex items-center gap-1.5 text-xs text-ink-muted">
                      <div className="w-2 h-2 rounded-full" style={{ background: STATUS_COLORS[s] }} />
                      <span className="capitalize">{s}</span>
                      <span className="text-ink-faint">{val}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Type breakdown */}
          <div>
            <p className="text-xs text-ink-muted mb-2">By type</p>
            <div className="space-y-2">
              {Object.entries(TYPE_INFO).map(([key, { color, label, desc }]) => {
                const count = typeCounts[key] || 0
                const pct = total > 0 ? (count / total) * 100 : 0
                return (
                  <div key={key} className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                    <span className="text-xs text-ink-light w-16">{label}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-paper-deep overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${Math.max(pct, count > 0 ? 2 : 0)}%`, background: color }} />
                    </div>
                    <span className="text-xs text-ink-muted tabular-nums w-8 text-right">{count}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {total === 0 && (
            <p className="text-sm text-ink-faint text-center py-6">No memories yet</p>
          )}
        </div>

        {/* Access Patterns */}
        <div className="bg-card border border-warm-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-ink-light mb-4">Most Accessed</h3>
          {hasAccess ? (
            <div className="space-y-2.5">
              {topAccessed!.filter(m => m.access_count > 0).slice(0, 7).map((m, i) => (
                <div key={m.id} className="flex items-start gap-3">
                  <span className="text-xs text-ink-faint tabular-nums w-4 text-right mt-0.5 shrink-0">{i + 1}</span>
                  <p className="text-xs text-ink-light flex-1 leading-relaxed line-clamp-2">{m.content}</p>
                  <span className="text-xs text-accent tabular-nums font-medium shrink-0">{m.access_count}x</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <TrendingUp size={24} className="text-ink-faint mb-2" />
              <p className="text-sm text-ink-muted">No access data yet</p>
              <p className="text-xs text-ink-faint mt-1">Memories will appear here as you search and chat</p>
            </div>
          )}
        </div>
      </div>

      {/* Namespaces */}
      {namespaces && namespaces.length > 0 && (
        <div className="bg-card border border-warm-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-ink-light mb-3">Namespaces</h3>
          <div className="space-y-2.5">
            {namespaces.map(ns => {
              const max = Math.max(...namespaces.map(n => n.total_count), 1)
              const pct = (ns.total_count / max) * 100
              const isActive = ns.namespace === namespace
              return (
                <div key={ns.namespace} className="flex items-center gap-3">
                  <span className={`text-xs w-28 truncate shrink-0 ${isActive ? 'text-accent font-medium' : 'text-ink-light'}`}>
                    {ns.namespace}
                  </span>
                  <div className="flex-1 h-2 rounded-full bg-paper-deep overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${Math.max(pct, 2)}%`, background: isActive ? '#C47B3B' : '#D4CBC0' }}
                    />
                  </div>
                  <span className="text-xs text-ink-muted tabular-nums w-12 text-right">
                    {ns.active_count}<span className="text-ink-faint">/{ns.total_count}</span>
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ icon: Icon, label, value, sub }: {
  icon: React.ComponentType<{ size?: number; className?: string }>
  label: string
  value: string | number
  sub?: string
}) {
  return (
    <div className="bg-card border border-warm-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-1.5">
        <Icon size={14} className="text-accent" />
        <span className="text-xs text-ink-muted">{label}</span>
      </div>
      <p className="text-2xl font-semibold text-ink">{value}</p>
      {sub && <p className="text-[11px] text-ink-faint mt-0.5">{sub}</p>}
    </div>
  )
}
