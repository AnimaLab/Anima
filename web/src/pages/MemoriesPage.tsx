import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { Trash2, Plus, Tag, Flame, Pencil, Check, Database } from 'lucide-react'
import { CATEGORIES, categoryColor, provenanceText } from '../lib/memoryMeta'
import { useToast } from '../hooks/useToast'

const STATUS_TABS = ['all', 'active', 'superseded', 'deleted'] as const
const TYPE_TABS = ['all', 'raw', 'reflected', 'deduced', 'induced'] as const

const typeColor: Record<string, string> = {
  raw:       'bg-stone-500/15 text-stone-600',
  reflected: 'bg-violet-500/15 text-violet-600',
  deduced:   'bg-sky-500/15 text-sky-600',
  induced:   'bg-emerald-500/15 text-emerald-600',
}

const statusColor: Record<string, string> = {
  active: 'badge-success',
  superseded: 'badge-warning',
  deleted: 'badge-danger',
}

function relativeDate(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function MemoriesPage() {
  const [status, setStatus] = useState<string>('all')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [category, setCategory] = useState<string>('all')
  const [newCategory, setNewCategory] = useState<string>('')
  const [offset, setOffset] = useState(0)
  const [showAdd, setShowAdd] = useState(false)
  const [newContent, setNewContent] = useState('')
  const [newTags, setNewTags] = useState('')
  const [consolidate, setConsolidate] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [editImportance, setEditImportance] = useState(5)
  const limit = 20
  const queryClient = useQueryClient()
  const toast = useToast()

  // Counts in the filter tabs and sidebar come from stats, so refresh both.
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['memories'] })
    queryClient.invalidateQueries({ queryKey: ['stats'] })
    queryClient.invalidateQueries({ queryKey: ['namespaces'] })
  }
  const onError = (action: string) => (e: Error) => toast.show(`${action} failed: ${e.message}`, true)

  const { data, isLoading } = useQuery({
    queryKey: ['memories', getNamespace(), status, typeFilter, category, offset],
    queryFn: () => api.listMemories(
      offset,
      limit,
      status === 'all' ? undefined : status,
      typeFilter === 'all' ? undefined : typeFilter,
      category === 'all' ? undefined : category,
    ),
  })

  // Fetch stats for filter counts
  const { data: stats } = useQuery({
    queryKey: ['stats', getNamespace()],
    queryFn: () => api.getStats(),
    staleTime: 30000,
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.deleteMemory(id),
    onSuccess: () => { refresh(); toast.show('Memory moved to Deleted') },
    onError: onError('Delete'),
  })

  const purgeMut = useMutation({
    mutationFn: () => api.purgeDeletedMemories(),
    onSuccess: refresh,
    onError: onError('Purge'),
  })

  const addMut = useMutation({
    mutationFn: () => api.addMemory({
      content: newContent,
      consolidate,
      tags: newTags.split(',').map(t => t.trim()).filter(Boolean),
      category: newCategory || undefined,
    }),
    onSuccess: () => {
      setNewContent('')
      setNewTags('')
      setNewCategory('')
      setShowAdd(false)
      refresh()
    },
    onError: onError('Save'),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: { content?: string; importance?: number } }) =>
      api.updateMemory(id, data),
    onSuccess: () => {
      setEditingId(null)
      refresh()
    },
    onError: onError('Update'),
  })

  const startEdit = (m: { id: string; content: string; importance: number }) => {
    setEditingId(m.id)
    setEditContent(m.content)
    setEditImportance(m.importance)
  }

  const total = data?.total ?? 0
  const statusCounts: Record<string, number> = {
    all: stats?.total ?? 0,
    active: stats?.active ?? 0,
    superseded: stats?.superseded ?? 0,
    deleted: stats?.deleted ?? 0,
  }

  return (
    <div className="page space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="page-title">Memories</h1>
          <p className="page-description">
            {data ? `${total} ${total === 1 ? 'memory' : 'memories'} match the current filters` : 'Everything stored in this namespace'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {status === 'deleted' && (data?.memories?.length ?? 0) > 0 && (
            <button
              onClick={() => purgeMut.mutate()}
              disabled={purgeMut.isPending}
              className="btn btn-danger"
              title="Permanently remove all deleted memories. This cannot be undone."
            >
              <Flame size={15} />
              {purgeMut.isPending ? 'Purging...' : 'Purge deleted'}
            </button>
          )}
          <button onClick={() => setShowAdd(!showAdd)} aria-expanded={showAdd} className="btn btn-primary">
            <Plus size={15} /> Add memory
          </button>
        </div>
      </div>

      {/* Add memory form */}
      {showAdd && (
        <div className="card p-4 space-y-3">
          <label htmlFor="new-memory" className="text-[13px] font-medium text-ink">New memory</label>
          <textarea
            id="new-memory"
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="What do you want to remember?"
            className="field resize-y min-h-20"
            autoFocus
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Tag size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none" />
              <input
                type="text"
                value={newTags}
                onChange={(e) => setNewTags(e.target.value)}
                placeholder="Tags, comma-separated"
                aria-label="Tags"
                className="field pl-8"
              />
            </div>
            <select
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              aria-label="Category"
              title="Category sets how fast the memory fades from ranking"
              className="field w-auto capitalize"
            >
              <option value="">Auto category</option>
              {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <label className="flex items-center gap-2 text-[13px] text-ink-light min-h-8" title="Check for duplicates and merge if similar">
              <input type="checkbox" checked={consolidate} onChange={(e) => setConsolidate(e.target.checked)} className="rounded" />
              Deduplicate
            </label>
            <div className="flex gap-1.5 ml-auto">
              <button onClick={() => setShowAdd(false)} className="btn btn-ghost">
                Cancel
              </button>
              <button onClick={() => addMut.mutate()} disabled={!newContent.trim() || addMut.isPending} className="btn btn-primary">
                {addMut.isPending ? 'Saving…' : 'Save memory'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        <div className="segmented" role="group" aria-label="Filter by status">
          {STATUS_TABS.map(tab => {
            const count = statusCounts[tab]
            return (
              <button key={tab} onClick={() => { setStatus(tab); setOffset(0) }} aria-pressed={status === tab}>
                {tab}{count > 0 && tab !== 'all' ? <span className="ml-1.5 text-ink-faint tabular-nums">{count}</span> : null}
              </button>
            )
          })}
        </div>
        <div className="segmented flex-wrap" role="group" aria-label="Filter by type">
          {TYPE_TABS.map(t => (
            <button key={t} onClick={() => { setTypeFilter(t); setOffset(0) }} aria-pressed={typeFilter === t}>
              {t === 'all' ? 'All types' : t}
            </button>
          ))}
        </div>
        <select
          value={category}
          onChange={(e) => { setCategory(e.target.value); setOffset(0) }}
          aria-label="Filter by category"
          className={`field w-auto sm:ml-auto capitalize ${category === 'all' ? '' : 'font-medium'}`}
        >
          <option value="all">All categories</option>
          {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {/* Memory list */}
      {isLoading ? (
        <div className="space-y-2" aria-busy="true" aria-label="Loading memories">
          {[0, 1, 2, 3].map(i => <div key={i} className="card h-[68px] animate-pulse bg-paper-warm" />)}
        </div>
      ) : (data?.memories?.length ?? 0) === 0 ? (
        <div className="card flex flex-col items-center justify-center py-16 px-6 text-center">
          <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
            <Database size={20} />
          </div>
          <p className="text-sm font-medium text-ink">No memories found</p>
          <p className="text-[13px] text-ink-muted mt-1">
            {status !== 'all' || typeFilter !== 'all' || category !== 'all'
              ? 'Try changing the filters'
              : 'Add your first memory to get started'}
          </p>
        </div>
      ) : (
        <ul className="card divide-y divide-warm-border overflow-hidden">
          {(data?.memories || []).map(m => (
            <li key={m.id} className="group px-4 sm:px-5 py-3.5 hover:bg-paper/60 transition-colors">
              {editingId === m.id ? (
                <div className="space-y-3">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    aria-label="Memory content"
                    className="field resize-y min-h-[60px]"
                    rows={3}
                  />
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] text-ink-muted">Importance</span>
                      <input type="range" min="1" max="10" value={editImportance}
                        onChange={(e) => setEditImportance(parseInt(e.target.value))}
                        aria-label="Importance" className="w-24" />
                      <span className="text-[13px] text-ink-muted tabular-nums">{editImportance}/10</span>
                    </div>
                    <div className="flex gap-1.5 ml-auto">
                      <button onClick={() => setEditingId(null)} className="btn btn-ghost">Cancel</button>
                      <button
                        onClick={() => updateMut.mutate({
                          id: m.id,
                          data: {
                            content: editContent !== m.content ? editContent : undefined,
                            importance: editImportance !== m.importance ? editImportance : undefined,
                          }
                        })}
                        disabled={updateMut.isPending}
                        className="btn btn-primary"
                      >
                        <Check size={15} /> Save
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-ink leading-relaxed">{m.content}</p>
                    <div className="flex items-center gap-x-2 gap-y-1 mt-2 flex-wrap text-xs text-ink-faint">
                      {/* Active and raw are the defaults: only the exceptions get a badge */}
                      {m.status !== 'active' && (
                        <span className={`badge ${statusColor[m.status] || 'badge-neutral'}`}>{m.status}</span>
                      )}
                      {m.memory_type !== 'raw' && (
                        <span className={`badge ${typeColor[m.memory_type] || 'badge-neutral'}`}>{m.memory_type}</span>
                      )}
                      {m.category && m.category !== 'general' && (
                        <button
                          type="button"
                          onClick={() => { setCategory(category === m.category ? 'all' : m.category!); setOffset(0) }}
                          title={`Filter by ${m.category}`}
                          className={`badge hover:opacity-80 transition-opacity ${categoryColor(m.category)}`}
                        >
                          {m.category}
                        </button>
                      )}
                      {provenanceText(m.source, m.confidence) && (
                        <span className="text-xs text-ink-muted" title="How this memory was created and how much it is trusted">
                          {provenanceText(m.source, m.confidence)}
                        </span>
                      )}
                      {/* Only show importance when non-default */}
                      {m.importance !== 5 && (
                        <span className={`text-xs ${m.importance >= 7 ? 'text-amber-600' : 'text-ink-faint'}`}>
                          {m.importance >= 9 ? 'Critical' : m.importance >= 7 ? 'Important' : `imp ${m.importance}`}
                        </span>
                      )}
                      {m.tags.length > 0 && m.tags.map(tag => (
                        <span key={tag} className="badge badge-neutral normal-case">
                          <Tag size={11} />{tag}
                        </span>
                      ))}
                      <span className="text-xs text-ink-faint">{relativeDate(m.created_at)}</span>
                      {/* Only show access count when > 0 */}
                      {m.access_count > 0 && (
                        <span className="text-xs text-accent">{m.access_count}x accessed</span>
                      )}
                    </div>
                  </div>
                  {m.status === 'active' && (
                    <div className="flex items-center gap-0.5 shrink-0 -mr-1.5 -mt-1">
                      <button onClick={() => startEdit(m)} className="btn btn-ghost btn-icon" title="Edit" aria-label="Edit memory">
                        <Pencil size={15} />
                      </button>
                      <button onClick={() => deleteMut.mutate(m.id)} disabled={deleteMut.isPending} className="btn btn-ghost btn-icon hover:text-danger" title="Delete" aria-label="Delete memory">
                        <Trash2 size={15} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {/* Pagination */}
      {data && data.total > limit && (
        <nav aria-label="Pagination" className="flex items-center gap-3 justify-center pt-1">
          <button onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0}
            className="btn btn-secondary">
            Previous
          </button>
          <span className="text-xs text-ink-muted tabular-nums">{offset + 1}–{Math.min(offset + limit, Number(data.total))} of {data.total}</span>
          <button onClick={() => setOffset(offset + limit)} disabled={offset + limit >= Number(data.total)}
            className="btn btn-secondary">
            Next
          </button>
        </nav>
      )}
    </div>
  )
}
