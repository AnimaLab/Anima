import { Link } from 'react-router-dom'
import { useQuery, useQueries } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { useNamespace } from '../hooks/useNamespace'
import { Database, Clock, CheckCircle2, Layers, TrendingUp, Plus, ArrowRight, Search } from 'lucide-react'
import { CATEGORIES, categoryColor, provenanceText } from '../lib/memoryMeta'

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

// Data-visualisation hues. Mid-tones chosen to read on both themes.
const STATUS_COLORS: Record<string, string> = {
  active: '#4F9A6B',
  superseded: '#C49A3C',
  deleted: '#D0604F',
}

const TYPE_INFO: Record<string, { color: string; label: string; desc: string }> = {
  raw:       { color: '#8A94A0', label: 'Raw',       desc: 'Original input' },
  reflected: { color: '#8B7DB8', label: 'Reflected', desc: 'Facts extracted from raw memories' },
  deduced:   { color: '#5B9CA6', label: 'Deduced',   desc: 'Inferred by combining facts' },
  induced:   { color: '#5BA88C', label: 'Induced',   desc: 'Patterns synthesised over time' },
}

const CATEGORY_BAR: Record<string, string> = {
  identity: '#8E9AAF',
  preference: '#F2994A',
  routine: '#7BC96F',
  environment: '#3FB6A8',
  task: '#D99A1E',
  inferred: '#B59ADB',
  general: '#8A94A0',
}

export function DashboardPage() {
  const { namespace } = useNamespace()
  const ns = getNamespace()

  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats', ns],
    queryFn: () => api.getStats(),
    refetchInterval: 30000,
  })
  const { data: namespaces } = useQuery({
    queryKey: ['namespaces'],
    queryFn: () => api.listNamespaces(),
  })
  const { data: recent } = useQuery({
    queryKey: ['memories', ns, 'recent'],
    queryFn: () => api.listMemories(0, 5, 'active'),
    staleTime: 15000,
  })
  const { data: topAccessed } = useQuery({
    queryKey: ['top-accessed', 'most', ns],
    queryFn: () => api.topAccessed('most', 5),
  })
  // Type breakdown comes from graph data, which carries memory_type per node.
  const { data: graphData } = useQuery({
    queryKey: ['graph', ns, 0.99],
    queryFn: () => api.getGraph(0.99, 500),
    staleTime: 60000,
  })
  // One tiny request per category: only the total is read.
  const categoryQueries = useQueries({
    queries: CATEGORIES.map(category => ({
      queryKey: ['category-count', ns, category],
      queryFn: () => api.listMemories(0, 1, 'active', undefined, category),
      staleTime: 30000,
    })),
  })

  const typeCounts = graphData
    ? graphData.nodes.reduce((acc, n) => {
        acc[n.memory_type] = (acc[n.memory_type] || 0) + 1
        return acc
      }, {} as Record<string, number>)
    : {}

  const categoryCounts = CATEGORIES
    .map((category, i) => ({ category, count: Number(categoryQueries[i].data?.total ?? 0) }))
    .sort((a, b) => b.count - a.count)
  const maxCategory = Math.max(...categoryCounts.map(c => c.count), 1)

  const total = stats?.total ?? 0
  const accessed = (topAccessed ?? []).filter(m => m.access_count > 0)
  const isEmpty = !isLoading && stats !== undefined && total === 0

  return (
    <div className="page space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-description">
            What Anima remembers in <span className="font-medium text-ink">{namespace}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/search" className="btn btn-secondary"><Search size={15} /> Search</Link>
          <Link to="/memories" className="btn btn-primary"><Plus size={15} /> Add memory</Link>
        </div>
      </div>

      {isEmpty ? (
        <div className="card flex flex-col items-center text-center px-6 py-16">
          <div className="w-12 h-12 rounded-full bg-accent-light text-accent flex items-center justify-center mb-4">
            <Database size={22} />
          </div>
          <h2 className="text-base font-semibold text-ink">No memories yet</h2>
          <p className="text-sm text-ink-muted mt-1 max-w-sm">
            Add a fact, a preference or a decision. Anima will find it again by meaning, and track how it changes over time.
          </p>
          <Link to="/memories" className="btn btn-primary mt-5"><Plus size={15} /> Add your first memory</Link>
        </div>
      ) : (
        <>
          {/* Key figures */}
          <dl className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
            <StatCard icon={Database} label="Total memories" value={stats?.total ?? '–'} />
            <StatCard
              icon={CheckCircle2}
              label="Active"
              value={stats?.active ?? '–'}
              sub={stats
                ? [stats.superseded > 0 && `${stats.superseded} superseded`, stats.deleted > 0 && `${stats.deleted} deleted`].filter(Boolean).join(' · ') || 'Nothing superseded'
                : undefined}
            />
            <StatCard icon={Clock} label="Last added" value={stats ? timeAgo(stats.newest_memory) : '–'} />
            <StatCard icon={Layers} label="Namespaces" value={namespaces?.length ?? '–'} />
          </dl>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Recent memories */}
            <section className="card lg:col-span-2 flex flex-col" aria-labelledby="recent-heading">
              <div className="flex items-center justify-between px-5 pt-4 pb-3">
                <h2 id="recent-heading" className="text-sm font-semibold text-ink">Recent memories</h2>
                <Link to="/memories" className="text-[13px] font-medium text-accent hover:text-accent-hover inline-flex items-center gap-1 rounded">
                  View all <ArrowRight size={14} />
                </Link>
              </div>
              {(recent?.memories?.length ?? 0) > 0 ? (
                <ul className="divide-y divide-warm-border border-t border-warm-border">
                  {recent!.memories.map(m => (
                    <li key={m.id} className="px-5 py-3">
                      <p className="text-sm text-ink leading-relaxed line-clamp-2">{m.content}</p>
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-1.5 text-xs text-ink-faint">
                        {m.category && m.category !== 'general' && (
                          <span className={`badge ${categoryColor(m.category)}`}>{m.category}</span>
                        )}
                        {provenanceText(m.source, m.confidence) && <span>{provenanceText(m.source, m.confidence)}</span>}
                        <span>{timeAgo(m.created_at)}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="px-5 pb-5 text-sm text-ink-muted">No active memories in this namespace.</p>
              )}
            </section>

            {/* Categories */}
            <section className="card p-5" aria-labelledby="category-heading">
              <h2 id="category-heading" className="text-sm font-semibold text-ink">By category</h2>
              <p className="text-xs text-ink-muted mt-0.5 mb-4">Category sets how fast a memory fades from ranking</p>
              <ul className="space-y-2.5">
                {categoryCounts.map(({ category, count }) => (
                  <li key={category} className="flex items-center gap-3">
                    <span className="text-[13px] text-ink-light capitalize w-24 shrink-0">{category}</span>
                    <div className="flex-1 h-2 rounded-full bg-paper-deep overflow-hidden" aria-hidden="true">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${count > 0 ? Math.max((count / maxCategory) * 100, 4) : 0}%`, background: CATEGORY_BAR[category] }}
                      />
                    </div>
                    <span className="text-[13px] text-ink-muted tabular-nums w-8 text-right">{count}</span>
                  </li>
                ))}
              </ul>
            </section>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Lifecycle */}
            <section className="card p-5" aria-labelledby="lifecycle-heading">
              <h2 id="lifecycle-heading" className="text-sm font-semibold text-ink mb-4">Lifecycle</h2>

              <p className="section-label mb-2">Status</p>
              {stats && total > 0 && (
                <>
                  <div className="flex h-2 rounded-full overflow-hidden bg-paper-deep gap-px" aria-hidden="true">
                    {(['active', 'superseded', 'deleted'] as const).map(s => {
                      const val = stats[s]
                      if (!val) return null
                      return <div key={s} className="h-full" style={{ width: `${(val / total) * 100}%`, background: STATUS_COLORS[s] }} />
                    })}
                  </div>
                  <ul className="flex flex-wrap gap-x-4 gap-y-1 mt-2.5">
                    {(['active', 'superseded', 'deleted'] as const).map(s => (
                      <li key={s} className="flex items-center gap-1.5 text-xs text-ink-muted">
                        <span className="w-2 h-2 rounded-full" style={{ background: STATUS_COLORS[s] }} aria-hidden="true" />
                        <span className="capitalize">{s}</span>
                        <span className="text-ink tabular-nums font-medium">{stats[s]}</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              <p className="section-label mt-5 mb-2">Type</p>
              <ul className="space-y-2">
                {Object.entries(TYPE_INFO).map(([key, { color, label, desc }]) => {
                  const count = typeCounts[key] || 0
                  const pct = total > 0 ? (count / total) * 100 : 0
                  return (
                    <li key={key} className="flex items-center gap-3" title={desc}>
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} aria-hidden="true" />
                      <span className="text-[13px] text-ink-light w-20 shrink-0">{label}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-paper-deep overflow-hidden" aria-hidden="true">
                        <div className="h-full rounded-full" style={{ width: `${Math.max(pct, count > 0 ? 3 : 0)}%`, background: color }} />
                      </div>
                      <span className="text-[13px] text-ink-muted tabular-nums w-8 text-right">{count}</span>
                    </li>
                  )
                })}
              </ul>
            </section>

            {/* Most accessed */}
            <section className="card p-5" aria-labelledby="accessed-heading">
              <h2 id="accessed-heading" className="text-sm font-semibold text-ink mb-4">Most accessed</h2>
              {accessed.length > 0 ? (
                <ol className="space-y-3">
                  {accessed.slice(0, 5).map((m, i) => (
                    <li key={m.id} className="flex items-start gap-3">
                      <span className="text-xs text-ink-faint tabular-nums w-4 text-right mt-0.5 shrink-0">{i + 1}</span>
                      <p className="text-[13px] text-ink-light flex-1 leading-relaxed line-clamp-2">{m.content}</p>
                      <span className="badge badge-neutral tabular-nums shrink-0">{m.access_count}×</span>
                    </li>
                  ))}
                </ol>
              ) : (
                <div className="flex flex-col items-center justify-center py-6 text-center">
                  <TrendingUp size={22} className="text-ink-faint mb-2" />
                  <p className="text-sm text-ink-muted">No access data yet</p>
                  <p className="text-xs text-ink-faint mt-1">Memories appear here once search or chat retrieves them</p>
                </div>
              )}
            </section>

            {/* Namespaces */}
            <section className="card p-5" aria-labelledby="ns-heading">
              <h2 id="ns-heading" className="text-sm font-semibold text-ink mb-4">Namespaces</h2>
              <ul className="space-y-3">
                {(namespaces ?? []).map(n => {
                  const max = Math.max(...(namespaces ?? []).map(x => x.total_count), 1)
                  const isCurrent = n.namespace === namespace
                  return (
                    <li key={n.namespace}>
                      <div className="flex items-baseline justify-between gap-2 mb-1">
                        <span className={`text-[13px] truncate ${isCurrent ? 'text-ink font-semibold' : 'text-ink-light'}`}>
                          {n.namespace}
                          {isCurrent && <span className="ml-1.5 text-xs font-normal text-ink-faint">current</span>}
                        </span>
                        <span className="text-xs text-ink-muted tabular-nums shrink-0">
                          {n.active_count} active<span className="text-ink-faint"> / {n.total_count}</span>
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-paper-deep overflow-hidden" aria-hidden="true">
                        <div
                          className={`h-full rounded-full ${isCurrent ? 'bg-accent' : 'bg-warm-border-strong'}`}
                          style={{ width: `${Math.max((n.total_count / max) * 100, 3)}%` }}
                        />
                      </div>
                    </li>
                  )
                })}
              </ul>
            </section>
          </div>
        </>
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
    <div className="card p-4 sm:p-5">
      <div className="flex items-start justify-between gap-2">
        <dt className="text-[13px] font-medium text-ink-muted">{label}</dt>
        <Icon size={16} className="text-ink-faint shrink-0" />
      </div>
      <dd className="mt-2 text-[28px] leading-8 font-semibold text-ink tracking-tight tabular-nums">{value}</dd>
      {sub && <p className="text-xs text-ink-faint mt-1">{sub}</p>}
    </div>
  )
}
