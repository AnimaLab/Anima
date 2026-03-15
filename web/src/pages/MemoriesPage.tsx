import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { Trash2, Plus, Tag, Flame, Pencil, Check, X } from 'lucide-react'

const STATUS_TABS = ['all', 'active', 'superseded', 'deleted'] as const
const TYPE_TABS = ['all', 'raw', 'reflected', 'deduced', 'induced'] as const

const typeColor: Record<string, string> = {
  raw:       'bg-gray-500/20 text-gray-400',
  reflected: 'bg-violet-500/20 text-violet-400',
  deduced:   'bg-sky-500/20 text-sky-400',
  induced:   'bg-emerald-500/20 text-emerald-400',
}

const importanceLabel = (v: number) =>
  v >= 9 ? 'Critical' : v >= 7 ? 'Important' : v >= 4 ? 'Normal' : 'Minor'

const importanceColor = (v: number) =>
  v >= 9 ? 'text-red-400' : v >= 7 ? 'text-amber-400' : v >= 4 ? 'text-gray-400' : 'text-gray-600'

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

  const statusColor: Record<string, string> = {
    active: 'bg-green-500/20 text-green-400',
    superseded: 'bg-amber-500/20 text-amber-400',
    deleted: 'bg-red-500/20 text-red-400',
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Memories</h2>
        <div className="flex items-center gap-2">
          {status === 'deleted' && (data?.memories?.length ?? 0) > 0 && (
            <button
              onClick={() => purgeMut.mutate()}
              disabled={purgeMut.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600/80 hover:bg-red-500 disabled:opacity-50 rounded-lg text-sm text-white transition-colors"
            >
              <Flame size={14} />
              {purgeMut.isPending ? 'Purging...' : 'Purge deleted'}
            </button>
          )}
          <button onClick={() => setShowAdd(!showAdd)} className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm text-white transition-colors">
            <Plus size={16} /> Add Memory
          </button>
        </div>
      </div>

      {showAdd && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
          <textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="Memory content..."
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 resize-none h-24 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <div>
            <label className="block text-xs text-gray-500 mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={newTags}
              onChange={(e) => setNewTags(e.target.value)}
              placeholder="food, preferences, work"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-400">
              <input type="checkbox" checked={consolidate} onChange={(e) => setConsolidate(e.target.checked)} className="rounded" />
              Consolidate
            </label>
            <button onClick={() => addMut.mutate()} disabled={!newContent.trim()} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg text-sm text-white">
              Save
            </button>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-4">
        <div className="flex flex-wrap gap-1">
          {STATUS_TABS.map(tab => (
            <button key={tab} onClick={() => { setStatus(tab); setOffset(0) }}
              className={`px-3 py-1 rounded-md text-xs capitalize transition-colors ${status === tab ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
              {tab}
            </button>
          ))}
        </div>
        <div className="hidden sm:block w-px h-4 bg-gray-700" />
        <div className="flex flex-wrap gap-1">
          {TYPE_TABS.map(t => (
            <button key={t} onClick={() => { setTypeFilter(t); setOffset(0) }}
              className={`px-2.5 py-1 rounded-md text-xs capitalize transition-colors ${typeFilter === t ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <p className="text-gray-500 text-sm">Loading...</p>
      ) : (
        <div className="space-y-2">
          {(data?.memories || []).map(m => (
            <div key={m.id} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
              {editingId === m.id ? (
                <div className="space-y-3">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 resize-y min-h-[60px] focus:outline-none focus:ring-1 focus:ring-blue-500"
                    rows={3}
                  />
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">Importance:</span>
                      <input type="range" min="1" max="10" value={editImportance}
                        onChange={(e) => setEditImportance(parseInt(e.target.value))}
                        className="w-20 accent-blue-500" />
                      <span className={`text-xs font-mono ${importanceColor(editImportance)}`}>
                        {editImportance} ({importanceLabel(editImportance)})
                      </span>
                    </div>
                    <div className="flex gap-1.5 ml-auto">
                      <button
                        onClick={() => updateMut.mutate({
                          id: m.id,
                          data: {
                            content: editContent !== m.content ? editContent : undefined,
                            importance: editImportance !== m.importance ? editImportance : undefined,
                          }
                        })}
                        disabled={updateMut.isPending}
                        className="flex items-center gap-1 px-2.5 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs rounded-lg"
                      >
                        <Check size={12} /> Save
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="flex items-center gap-1 px-2.5 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs rounded-lg"
                      >
                        <X size={12} /> Cancel
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-200 leading-relaxed">{m.content}</p>
                    <div className="flex items-center gap-2 mt-2 flex-wrap">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${statusColor[m.status] || ''}`}>{m.status}</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${typeColor[m.memory_type] || typeColor.raw}`}>{m.memory_type}</span>
                      <span className={`text-[10px] ${importanceColor(m.importance)}`}>{importanceLabel(m.importance)} ({m.importance})</span>
                      {m.tags.length > 0 && m.tags.map(tag => (
                        <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-700/50 text-gray-400 flex items-center gap-0.5">
                          <Tag size={8} />{tag}
                        </span>
                      ))}
                      <span className="text-[10px] text-gray-600">{new Date(m.created_at).toLocaleDateString()}</span>
                      <span className="text-[10px] text-gray-600">accessed {m.access_count}x</span>
                    </div>
                  </div>
                  {m.status === 'active' && (
                    <div className="flex items-center gap-1">
                      <button onClick={() => startEdit(m)} className="text-gray-600 hover:text-blue-400 transition-colors p-1" title="Edit">
                        <Pencil size={14} />
                      </button>
                      <button onClick={() => deleteMut.mutate(m.id)} className="text-gray-600 hover:text-red-400 transition-colors p-1" title="Delete">
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

      {data && data.total > limit && (
        <div className="flex items-center gap-2 justify-center">
          <button onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0} className="px-3 py-1 rounded text-xs text-gray-400 hover:text-white disabled:opacity-30">Prev</button>
          <span className="text-xs text-gray-500">{offset + 1}–{Math.min(offset + limit, Number(data.total))} of {data.total}</span>
          <button onClick={() => setOffset(offset + limit)} disabled={offset + limit >= Number(data.total)} className="px-3 py-1 rounded text-xs text-gray-400 hover:text-white disabled:opacity-30">Next</button>
        </div>
      )}
    </div>
  )
}
