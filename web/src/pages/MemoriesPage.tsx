import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { Trash2, Plus, Tag, Flame, Pencil, Check, X, Database } from 'lucide-react'

const STATUS_TABS = ['all', 'active', 'superseded', 'deleted'] as const
const TYPE_TABS = ['all', 'raw', 'reflected', 'deduced', 'induced'] as const

const typeColor: Record<string, string> = {
  raw:       'bg-stone-500/15 text-stone-600',
  reflected: 'bg-violet-500/15 text-violet-600',
  deduced:   'bg-sky-500/15 text-sky-600',
  induced:   'bg-emerald-500/15 text-emerald-600',
}

const statusColor: Record<string, string> = {
  active: 'bg-green-500/15 text-green-700',
  superseded: 'bg-amber-500/15 text-amber-700',
  deleted: 'bg-red-500/15 text-red-700',
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

  const { data, isLoading } = useQuery({
    queryKey: ['memories', getNamespace(), status, typeFilter, offset],
    queryFn: () => api.listMemories(offset, limit, status === 'all' ? undefined : status, typeFilter === 'all' ? undefined : typeFilter),
  })

  // Fetch stats for filter counts
  const { data: stats } = useQuery({
    queryKey: ['stats', getNamespace()],
    queryFn: () => api.getStats(),
    staleTime: 30000,
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.deleteMemory(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['memories'] }),
  })

  const purgeMut = useMutation({
    mutationFn: () => api.purgeDeletedMemories(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['memories'] }),
  })

  const addMut = useMutation({
    mutationFn: () => api.addMemory({
      content: newContent,
      consolidate,
      tags: newTags.split(',').map(t => t.trim()).filter(Boolean),
    }),
    onSuccess: () => {
      setNewContent('')
      setNewTags('')
      setShowAdd(false)
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
  })

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: { content?: string; importance?: number } }) =>
      api.updateMemory(id, data),
    onSuccess: () => {
      setEditingId(null)
      queryClient.invalidateQueries({ queryKey: ['memories'] })
    },
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
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-ink">Memories</h2>
          {data && (
            <p className="text-xs text-ink-muted mt-0.5">{total} total</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {status === 'deleted' && (data?.memories?.length ?? 0) > 0 && (
            <button
              onClick={() => purgeMut.mutate()}
              disabled={purgeMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-[#C25B4E] hover:bg-[#B5503F] disabled:opacity-50 rounded-lg text-sm text-white transition-colors"
            >
              <Flame size={14} />
              {purgeMut.isPending ? 'Purging...' : 'Purge deleted'}
            </button>
          )}
          <button onClick={() => setShowAdd(!showAdd)} className="flex items-center gap-1.5 px-3 py-1.5 bg-accent hover:bg-accent-hover rounded-lg text-sm text-white transition-colors">
            <Plus size={16} /> Add Memory
          </button>
        </div>
      </div>

      {/* Add memory form */}
      {showAdd && (
        <div className="bg-card border border-warm-border rounded-xl p-4 space-y-3">
          <textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="What do you want to remember?"
            className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink resize-none h-20 focus:outline-none focus:ring-1 focus:ring-accent placeholder:text-ink-faint"
            autoFocus
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5 flex-1 min-w-[200px]">
              <Tag size={12} className="text-ink-faint shrink-0" />
              <input
                type="text"
                value={newTags}
                onChange={(e) => setNewTags(e.target.value)}
                placeholder="Tags (comma-separated)"
                className="flex-1 bg-transparent text-sm text-ink focus:outline-none placeholder:text-ink-faint"
              />
            </div>
            <label className="flex items-center gap-1.5 text-xs text-ink-muted cursor-pointer" title="Check for duplicates and merge if similar">
              <input type="checkbox" checked={consolidate} onChange={(e) => setConsolidate(e.target.checked)} className="rounded" />
              Deduplicate
            </label>
            <div className="flex gap-1.5 ml-auto">
              <button onClick={() => setShowAdd(false)} className="px-3 py-1.5 text-xs text-ink-muted hover:text-ink transition-colors">
                Cancel
              </button>
              <button onClick={() => addMut.mutate()} disabled={!newContent.trim()} className="px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:opacity-50 rounded-lg text-sm text-white">
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-4">
        <div className="flex flex-wrap gap-1">
          {STATUS_TABS.map(tab => {
            const count = statusCounts[tab]
            return (
              <button key={tab} onClick={() => { setStatus(tab); setOffset(0) }}
                className={`px-2.5 py-1 rounded-md text-xs capitalize transition-colors ${
                  status === tab ? 'bg-paper-deep text-ink font-medium' : 'text-ink-muted hover:text-ink'
                }`}>
                {tab}{count > 0 && tab !== 'all' ? ` (${count})` : ''}
              </button>
            )
          })}
        </div>
        <div className="hidden sm:block w-px h-4 bg-warm-border" />
        <div className="flex flex-wrap gap-1">
          {TYPE_TABS.map(t => (
            <button key={t} onClick={() => { setTypeFilter(t); setOffset(0) }}
              className={`px-2.5 py-1 rounded-md text-xs capitalize transition-colors ${
                typeFilter === t ? 'bg-paper-deep text-ink font-medium' : 'text-ink-muted hover:text-ink'
              }`}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Memory list */}
      {isLoading ? (
        <p className="text-ink-muted text-sm py-8 text-center">Loading...</p>
      ) : (data?.memories?.length ?? 0) === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Database size={28} className="text-ink-faint mb-3" />
          <p className="text-sm text-ink-muted">No memories found</p>
          <p className="text-xs text-ink-faint mt-1">
            {status !== 'all' || typeFilter !== 'all'
              ? 'Try changing the filters'
              : 'Add your first memory to get started'}
          </p>
        </div>
      ) : (
        <div className="space-y-1.5">
          {(data?.memories || []).map(m => (
            <div key={m.id} className="bg-card border border-warm-border rounded-lg px-4 py-3">
              {editingId === m.id ? (
                <div className="space-y-3">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-accent"
                    rows={3}
                  />
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-ink-muted">Importance:</span>
                      <input type="range" min="1" max="10" value={editImportance}
                        onChange={(e) => setEditImportance(parseInt(e.target.value))}
                        className="w-20 accent-accent" />
                      <span className="text-xs text-ink-muted tabular-nums">{editImportance}/10</span>
                    </div>
                    <div className="flex gap-1.5 ml-auto">
                      <button onClick={() => setEditingId(null)} className="px-2.5 py-1 text-xs text-ink-muted hover:text-ink transition-colors">Cancel</button>
                      <button
                        onClick={() => updateMut.mutate({
                          id: m.id,
                          data: {
                            content: editContent !== m.content ? editContent : undefined,
                            importance: editImportance !== m.importance ? editImportance : undefined,
                          }
                        })}
                        disabled={updateMut.isPending}
                        className="flex items-center gap-1 px-2.5 py-1 bg-accent hover:bg-accent-hover disabled:opacity-50 text-white text-xs rounded-lg"
                      >
                        <Check size={12} /> Save
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-ink leading-relaxed">{m.content}</p>
                    <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                      {/* Only show status badge if not filtering by a specific status, or if status != active */}
                      {(status === 'all' || m.status !== 'active') && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${statusColor[m.status] || ''}`}>{m.status}</span>
                      )}
                      {/* Only show type badge if not filtering by a specific type */}
                      {typeFilter === 'all' && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${typeColor[m.memory_type] || typeColor.raw}`}>{m.memory_type}</span>
                      )}
                      {/* Only show importance when non-default */}
                      {m.importance !== 5 && (
                        <span className={`text-[10px] ${m.importance >= 7 ? 'text-amber-600' : 'text-ink-faint'}`}>
                          {m.importance >= 9 ? 'Critical' : m.importance >= 7 ? 'Important' : `imp ${m.importance}`}
                        </span>
                      )}
                      {m.tags.length > 0 && m.tags.map(tag => (
                        <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-paper-deep text-ink-muted flex items-center gap-0.5">
                          <Tag size={8} />{tag}
                        </span>
                      ))}
                      <span className="text-[10px] text-ink-faint">{relativeDate(m.created_at)}</span>
                      {/* Only show access count when > 0 */}
                      {m.access_count > 0 && (
                        <span className="text-[10px] text-accent">{m.access_count}x accessed</span>
                      )}
                    </div>
                  </div>
                  {m.status === 'active' && (
                    <div className="flex items-center gap-0.5 shrink-0">
                      <button onClick={() => startEdit(m)} className="text-ink-faint hover:text-accent transition-colors p-1.5" title="Edit">
                        <Pencil size={14} />
                      </button>
                      <button onClick={() => deleteMut.mutate(m.id)} className="text-ink-faint hover:text-[#C25B4E] transition-colors p-1.5" title="Delete">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {data && data.total > limit && (
        <div className="flex items-center gap-3 justify-center pt-2">
          <button onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0}
            className="px-3 py-1 rounded-md text-xs text-ink-muted hover:text-ink hover:bg-paper-deep disabled:opacity-30 transition-colors">
            Prev
          </button>
          <span className="text-xs text-ink-muted tabular-nums">{offset + 1}–{Math.min(offset + limit, Number(data.total))} of {data.total}</span>
          <button onClick={() => setOffset(offset + limit)} disabled={offset + limit >= Number(data.total)}
            className="px-3 py-1 rounded-md text-xs text-ink-muted hover:text-ink hover:bg-paper-deep disabled:opacity-30 transition-colors">
            Next
          </button>
        </div>
      )}
    </div>
  )
}
