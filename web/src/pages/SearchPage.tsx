import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import type { SearchResponse, SearchResult, GraphData, AskResponse } from '../api/types'
import { loadCognitiveConfig } from './SettingsPage'
import { Search, Tag, Clock, Zap, BookOpen, ArrowUpDown, Shield, Link2, ChevronDown, ChevronRight, MessageSquare, SlidersHorizontal, Compass, X } from 'lucide-react'

type SortField = 'score' | 'vector_score' | 'keyword_score' | 'temporal_score' | 'importance' | 'created_at'
type SortDir = 'desc' | 'asc'
type AdjEntry = { id: string; content: string; similarity: number; edgeType: string }

const MEMORY_TYPES = ['all', 'fact', 'preference', 'event', 'story', 'decision', 'reflection', 'context', 'goal', 'relationship', 'emotion', 'habit', 'belief', 'skill', 'location'] as const

const typeColor: Record<string, string> = {
  preference: 'bg-purple-500/15 text-purple-700 border-purple-500/25',
  fact: 'bg-blue-500/15 text-blue-700 border-blue-500/25',
  event: 'bg-cyan-500/15 text-cyan-700 border-cyan-500/25',
  decision: 'bg-orange-500/15 text-orange-700 border-orange-500/25',
  story: 'bg-rose-500/15 text-rose-700 border-rose-500/25',
  reflection: 'bg-indigo-500/15 text-indigo-700 border-indigo-500/25',
  context: 'bg-stone-500/15 text-stone-600 border-stone-500/25',
  goal: 'bg-emerald-500/15 text-emerald-700 border-emerald-500/25',
  relationship: 'bg-pink-500/15 text-pink-700 border-pink-500/25',
  emotion: 'bg-red-500/15 text-red-700 border-red-500/25',
  habit: 'bg-teal-500/15 text-teal-700 border-teal-500/25',
  belief: 'bg-amber-500/15 text-amber-700 border-amber-500/25',
  skill: 'bg-lime-500/15 text-lime-700 border-lime-500/25',
  location: 'bg-sky-500/15 text-sky-700 border-sky-500/25',
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

function importanceLabel(imp: number): { text: string; color: string } {
  if (imp >= 9) return { text: 'Critical', color: 'text-red-600' }
  if (imp >= 7) return { text: 'Important', color: 'text-orange-600' }
  if (imp >= 4) return { text: 'Normal', color: 'text-ink-muted' }
  return { text: 'Minor', color: 'text-ink-faint' }
}

function ImportanceDots({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          className={`w-1 h-3 rounded-sm ${
            i < value
              ? value >= 9 ? 'bg-red-600' : value >= 7 ? 'bg-orange-500' : value >= 4 ? 'bg-accent' : 'bg-ink-faint'
              : 'bg-paper-deep'
          }`}
        />
      ))}
    </div>
  )
}

/** Build adjacency map from graph data */
function buildAdjacency(graph: GraphData) {
  const adj = new Map<string, AdjEntry[]>()
  for (const edge of graph.edges) {
    const sourceNode = graph.nodes.find(n => n.id === edge.source)
    const targetNode = graph.nodes.find(n => n.id === edge.target)
    if (sourceNode && targetNode) {
      if (!adj.has(edge.source)) adj.set(edge.source, [])
      adj.get(edge.source)!.push({
        id: edge.target,
        content: targetNode.content,
        similarity: edge.similarity,
        edgeType: edge.edge_type,
      })
      if (!adj.has(edge.target)) adj.set(edge.target, [])
      adj.get(edge.target)!.push({
        id: edge.source,
        content: sourceNode.content,
        similarity: edge.similarity,
        edgeType: edge.edge_type,
      })
    }
  }
  return adj
}

export function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchType, setSearchType] = useState<'search' | 'ask'>('search')
  const [mode, setMode] = useState<string>('hybrid')
  const [temporalWeight, setTemporalWeight] = useState(0.2)
  const [limit, setLimit] = useState(20)
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [results, setResults] = useState<SearchResponse | null>(null)
  const [askResult, setAskResult] = useState<AskResponse | null>(null)
  const [sortField, setSortField] = useState<SortField>('score')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  // Fetch graph data for neighbor relationships (low threshold to catch links)
  const { data: graphData } = useQuery({
    queryKey: ['graph', getNamespace(), 0.4],
    queryFn: () => api.getGraph(0.4, 500),
    staleTime: 30000,
  })

  const adjacency = useMemo(() => graphData ? buildAdjacency(graphData) : new Map<string, AdjEntry[]>(), [graphData])

  const searchMut = useMutation({
    mutationFn: (params: { q: string; m: string; lim: number; tw: number }) => {
      const cogConfig = loadCognitiveConfig()
      return api.search({
        query: params.q,
        limit: params.lim,
        search_mode: params.m as 'hybrid',
        temporal_weight: params.tw,
        max_tier: cogConfig.max_tier,
      })
    },
    onSuccess: (data) => {
      setResults(data)
      setAskResult(null)
      setSortField('score')
      setSortDir('desc')
      setExpandedId(null)
    },
  })

  const askMut = useMutation({
    mutationFn: (question: string) => {
      const cogConfig = loadCognitiveConfig()
      return api.ask({
        question,
        max_tier: cogConfig.max_tier,
      })
    },
    onSuccess: (data) => {
      setAskResult(data)
      setResults(null)
    },
  })

  // Discovery mode state
  const [discoverSource, setDiscoverSource] = useState<{ id: string; content: string } | null>(null)

  const discoverMut = useMutation({
    mutationFn: (memoryId: string) => api.discover({ positive_ids: [memoryId], limit: 20 }),
    onSuccess: (data) => {
      setResults(data)
      setAskResult(null)
      setSortField('score')
      setSortDir('desc')
      setExpandedId(null)
    },
  })

  const doSearch = useCallback((q: string, m: string, lim: number, tw: number) => {
    if (q.trim()) searchMut.mutate({ q, m, lim, tw })
  }, [searchMut])

  // Debounced auto-search on query change (only for classic search)
  useEffect(() => {
    if (searchType !== 'search') return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (!query.trim()) return
    debounceRef.current = setTimeout(() => {
      doSearch(query, mode, limit, temporalWeight)
    }, 350)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [query]) // only debounce on query text changes

  // Immediate re-search when options change (if query exists)
  const handleModeSwitch = (m: string) => {
    setMode(m)
    if (query.trim()) doSearch(query, m, limit, temporalWeight)
  }

  const handleLimitChange = (lim: number) => {
    setLimit(lim)
    if (query.trim()) doSearch(query, mode, lim, temporalWeight)
  }

  const handleTemporalChange = (tw: number) => {
    setTemporalWeight(tw)
    if (query.trim()) doSearch(query, mode, limit, tw)
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    if (searchType === 'ask') {
      askMut.mutate(query)
    } else {
      doSearch(query, mode, limit, temporalWeight)
    }
  }

  // Filter by type then sort
  const filteredResults = results ? results.results.filter(r => typeFilter === 'all' || r.memory_type === typeFilter) : []
  const sortedResults = filteredResults.length > 0 ? [...filteredResults].sort((a, b) => {
    let aVal: number, bVal: number
    switch (sortField) {
      case 'score': aVal = a.score; bVal = b.score; break
      case 'vector_score': aVal = a.vector_score ?? 0; bVal = b.vector_score ?? 0; break
      case 'keyword_score': aVal = a.keyword_score ?? 0; bVal = b.keyword_score ?? 0; break
      case 'temporal_score': aVal = a.temporal_score ?? 0; bVal = b.temporal_score ?? 0; break
      case 'importance': aVal = a.importance ?? 5; bVal = b.importance ?? 5; break
      case 'created_at': aVal = new Date(a.created_at).getTime(); bVal = new Date(b.created_at).getTime(); break
      default: aVal = a.score; bVal = b.score
    }
    return sortDir === 'desc' ? bVal - aVal : aVal - bVal
  }) : [] as SearchResult[]

  // Find graph-expanded context (like the LLM sees): results + their neighbors not already in results
  const resultIds = useMemo(() => new Set(sortedResults.map(r => r.id)), [sortedResults])
  const graphNeighbors = useMemo(() => {
    const neighbors = new Map<string, { id: string; content: string; similarity: number; parentId: string; edgeType: string }[]>()
    for (const r of sortedResults) {
      const adj = adjacency.get(r.id) || []
      const linked = adj
        .filter(n => !resultIds.has(n.id))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3)
      if (linked.length > 0) {
        neighbors.set(r.id, linked.map(n => ({ ...n, parentId: r.id })))
      }
    }
    return neighbors
  }, [sortedResults, adjacency, resultIds])

  // Count total context size (what the LLM would see)
  const totalContext = sortedResults.length + Array.from(graphNeighbors.values()).reduce((sum, arr) => sum + arr.length, 0)

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-ink">Search</h2>
        <div className="flex items-center gap-3">
          {results && (
            <>
              <span className="text-xs text-ink-muted">
                {results.results.length} direct + {totalContext - results.results.length} graph = {totalContext} total
              </span>
              <span className="text-[10px] text-ink-faint">{results.query_time_ms.toFixed(1)}ms</span>
            </>
          )}
          {askResult && (
            <>
              <span className="text-xs text-ink-muted">
                {askResult.total_search_results} memories searched
              </span>
              <span className="text-[10px] text-ink-faint">{askResult.elapsed_ms.toFixed(0)}ms</span>
            </>
          )}
        </div>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            {searchType === 'ask'
              ? <MessageSquare size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-emerald-600" />
              : <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-muted" />
            }
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={searchType === 'ask' ? "Ask a question about memories... (press Enter)" : "Search memories... (try: 'allergies', 'family', 'work')"}
              className={`w-full bg-card border rounded-lg pl-9 pr-3 py-2.5 text-sm text-ink placeholder:text-ink-faint focus:outline-none focus:ring-1 ${
                searchType === 'ask'
                  ? 'border-emerald-600/40 focus:ring-emerald-600 focus:border-emerald-600'
                  : 'border-warm-border-strong focus:ring-accent focus:border-accent'
              }`}
            />
          </div>
          {(searchMut.isPending || askMut.isPending) && (
            <div className="flex items-center px-3 text-ink-muted">
              <div className={`w-4 h-4 border-2 border-ink-faint rounded-full animate-spin ${
                searchType === 'ask' ? 'border-t-emerald-600' : 'border-t-accent'
              }`} />
            </div>
          )}
        </div>

        {/* Search type toggle + mode tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-6">
          {/* Search vs Ask toggle */}
          <div className="flex gap-1 bg-card rounded-lg p-1">
            <button
              type="button"
              onClick={() => setSearchType('search')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                searchType === 'search'
                  ? 'bg-accent-light text-accent shadow-sm'
                  : 'text-ink-muted hover:text-ink hover:bg-paper-deep'
              }`}
            >
              <Search size={12} />
              Search
            </button>
            <button
              type="button"
              onClick={() => setSearchType('ask')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                searchType === 'ask'
                  ? 'bg-emerald-600/15 text-emerald-600 shadow-sm'
                  : 'text-ink-muted hover:text-ink hover:bg-paper-deep'
              }`}
            >
              <MessageSquare size={12} />
              Ask
            </button>
          </div>

          {/* Classic search controls (hidden in ask mode) */}
          {searchType === 'search' && (
            <>
              <div className="flex gap-1 bg-card rounded-lg p-1">
                {['hybrid', 'vector', 'keyword'].map(m => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => handleModeSwitch(m)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium capitalize transition-colors ${
                      mode === m
                        ? 'bg-accent-light text-accent shadow-sm'
                        : 'text-ink-muted hover:text-ink hover:bg-paper-deep'
                    }`}
                  >
                    {m === 'hybrid' && <Zap size={12} />}
                    {m === 'vector' && <BookOpen size={12} />}
                    {m === 'keyword' && <Search size={12} />}
                    {m}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <Clock size={12} className="text-ink-muted" />
                <span className="text-xs text-ink-muted">Recency:</span>
                <input type="range" min="0" max="1" step="0.05" value={temporalWeight}
                  onChange={(e) => handleTemporalChange(parseFloat(e.target.value))}
                  className="w-20 accent-[#5B9CA6]" />
                <span className="text-xs text-ink-muted font-mono w-8">{temporalWeight.toFixed(2)}</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-ink-muted">Limit:</span>
                <select value={limit} onChange={(e) => handleLimitChange(parseInt(e.target.value))}
                  className="bg-card border border-warm-border-strong rounded px-2 py-1 text-xs text-ink-light focus:outline-none">
                  {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </>
          )}
        </div>

        {/* Type filter — only show when a non-all filter is active, or toggle via the type badge */}
        {searchType === 'search' && typeFilter !== 'all' && (
          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-[11px] text-ink-faint mr-1">Type:</span>
            {MEMORY_TYPES.map(t => (
              <button key={t} type="button" onClick={() => setTypeFilter(t)}
                className={`px-2.5 py-1 rounded-md text-xs capitalize transition-colors ${typeFilter === t ? 'bg-paper-deep text-ink font-medium' : 'text-ink-muted hover:text-ink'}`}>
                {t}
              </button>
            ))}
          </div>
        )}
      </form>

      {/* Ask result */}
      {askResult && (
        <div className="space-y-4">
          {/* Answer card */}
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <MessageSquare size={18} className="text-emerald-600 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-ink leading-relaxed whitespace-pre-wrap">{askResult.answer}</p>
              </div>
            </div>
          </div>

          {/* Referenced memories */}
          {askResult.memories_referenced.length > 0 && (
            <div>
              <span className="text-[10px] text-ink-muted block mb-2">
                {askResult.memories_referenced.length} memories referenced
              </span>
              <div className="space-y-1">
                {askResult.memories_referenced.slice(0, 20).map((mem, i) => (
                  <div key={mem.id} className="bg-card/80 border border-warm-border/60 rounded-md px-3 py-2 flex items-center gap-3">
                    <span className="text-[10px] text-ink-faint shrink-0 font-mono w-5 text-right">{i + 1}</span>
                    <p className="text-xs text-ink-muted flex-1 min-w-0 truncate">{mem.content}</p>
                    <span className="text-[10px] text-ink-faint font-mono shrink-0">{mem.score.toFixed(3)}</span>
                  </div>
                ))}
                {askResult.memories_referenced.length > 20 && (
                  <p className="text-[10px] text-ink-faint pl-8">...and {askResult.memories_referenced.length - 20} more</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Ask error */}
      {askMut.isError && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
          <p className="text-sm text-red-600">{(askMut.error as Error).message}</p>
          <p className="text-xs text-red-500/70 mt-1">Make sure the LLM is configured in Settings</p>
        </div>
      )}

      {/* Empty state (search mode only) */}
      {results && results.results.length === 0 && (
        <div className="text-center py-12">
          <Search size={32} className="mx-auto text-ink-faint mb-3" />
          <p className="text-ink-muted text-sm">No memories found for "{query}"</p>
          <p className="text-ink-faint text-xs mt-1">Try a different query or switch modes</p>
        </div>
      )}

      {/* Discovery mode header */}
      {discoverSource && (
        <div className="flex items-center gap-2 bg-accent/10 border border-accent/20 rounded-lg px-4 py-2.5">
          <Compass size={14} className="text-accent shrink-0" />
          <span className="text-xs text-accent">Similar to:</span>
          <span className="text-xs text-ink truncate flex-1">{discoverSource.content}</span>
          <button
            onClick={() => { setDiscoverSource(null); setResults(null) }}
            className="p-1 text-ink-muted hover:text-ink rounded transition-colors shrink-0"
            title="Clear discovery"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Results */}
      {sortedResults.length > 0 && (
        <div className="space-y-3">
          {/* Context summary */}
          <div className="flex items-center gap-2 text-xs text-ink-muted">
            <Zap size={12} className="text-accent" />
            <span>{sortedResults.length} results</span>
            {totalContext > sortedResults.length && (
              <span className="text-ink-faint">+ {totalContext - sortedResults.length} graph neighbors</span>
            )}
          </div>

          {/* Sort controls */}
          <div className="flex items-center gap-1 flex-wrap">
            <ArrowUpDown size={12} className="text-ink-faint mr-1" />
            <span className="text-[11px] text-ink-faint mr-1">Sort:</span>
            {([
              ['score', 'Score'],
              ['vector_score', 'Vector'],
              ['keyword_score', 'Keyword'],
              ['temporal_score', 'Temporal'],
              ['importance', 'Importance'],
              ['created_at', 'Date'],
            ] as [SortField, string][]).map(([field, label]) => (
              <button
                key={field}
                onClick={() => toggleSort(field)}
                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                  sortField === field ? 'bg-accent-light text-accent' : 'text-ink-faint hover:text-ink-muted'
                }`}
              >
                {label}{sortField === field && (sortDir === 'desc' ? ' ↓' : ' ↑')}
              </button>
            ))}
          </div>

          {/* Result cards with graph neighbors */}
          {sortedResults.map((r, i) => {
            const neighbors = graphNeighbors.get(r.id) || []
            const isExpanded = expandedId === r.id
            // Check if this result is a neighbor of a previous result
            const mutualLinks = sortedResults
              .filter(other => other.id !== r.id)
              .filter(other => {
                const adj = adjacency.get(r.id) || []
                return adj.some(n => n.id === other.id)
              })
              .map(other => {
                const adj = adjacency.get(r.id) || []
                const link = adj.find(n => n.id === other.id)
                return { id: other.id, content: other.content, similarity: link?.similarity ?? 0 }
              })

            const isNoise = r.score < 0.01

            return (
              <div key={r.id} className={`space-y-0 ${isNoise ? 'opacity-40' : ''}`}>
                {/* Main result card */}
                <div
                  className={`bg-card border rounded-lg overflow-hidden transition-colors cursor-pointer ${
                    isExpanded ? 'border-accent/40' : 'border-warm-border hover:border-warm-border-strong'
                  }`}
                  onClick={() => setExpandedId(isExpanded ? null : r.id)}
                >
                  {/* Score indicator bar */}
                  <div className="h-0.5 bg-paper-deep">
                    <div
                      className={`h-full transition-all ${
                        r.score > 0.7 ? 'bg-green-600' : r.score > 0.4 ? 'bg-accent' : r.score > 0.2 ? 'bg-amber-500' : 'bg-ink-faint'
                      }`}
                      style={{ width: `${Math.min((r.score / (sortedResults[0]?.score || 1)) * 100, 100)}%` }}
                    />
                  </div>

                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {/* Rank number */}
                      <span className="text-base font-bold text-ink-faint tabular-nums w-6 text-right shrink-0 mt-0.5">{i + 1}</span>

                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-ink leading-relaxed">{r.content}</p>

                        {/* Tags + metadata */}
                        <div className="flex items-center gap-2 mt-2.5 flex-wrap">
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); setTypeFilter(typeFilter === r.memory_type ? 'all' : r.memory_type) }}
                            className={`text-[10px] px-1.5 py-0.5 rounded-full border cursor-pointer hover:opacity-80 transition-opacity ${typeColor[r.memory_type] || typeColor.fact}`}
                            title={`Filter by ${r.memory_type}`}
                          >
                            {r.memory_type}
                          </button>
                          {r.tags.map(tag => (
                            <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-paper-deep text-ink-muted flex items-center gap-0.5 border border-warm-border-strong/50">
                              <Tag size={8} />{tag}
                            </span>
                          ))}
                          {(neighbors.length > 0 || mutualLinks.length > 0) && (
                            <span className="text-[10px] text-accent/70 flex items-center gap-0.5">
                              <Link2 size={9} />
                              {neighbors.length + mutualLinks.length} linked
                            </span>
                          )}
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              setDiscoverSource({ id: r.id, content: r.content })
                              discoverMut.mutate(r.id)
                            }}
                            className="text-[10px] px-1.5 py-0.5 rounded-full border border-accent/30 text-accent hover:bg-accent/10 transition-colors flex items-center gap-0.5"
                            title="Find similar memories"
                          >
                            <Compass size={9} />similar
                          </button>
                          <span className="text-[10px] text-ink-faint">{relativeDate(r.created_at)}</span>
                        </div>
                      </div>

                      {/* Score + importance */}
                      <div className="text-right shrink-0 space-y-0.5">
                        <div className="text-sm font-mono font-bold text-ink">{r.score.toFixed(2)}</div>
                        {(r.importance ?? 5) !== 5 && (
                          <div className={`text-[10px] ${importanceLabel(r.importance ?? 5).color}`}>
                            {importanceLabel(r.importance ?? 5).text}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Expanded details */}
                    {isExpanded && (
                      <div className="mt-4 pt-3 border-t border-warm-border space-y-3">
                        {/* Score breakdown */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                          <ScoreCard label="Combined" value={r.score} desc="Final score after fusion + temporal + boosts" range="0–1" color="blue" />
                          {r.vector_score != null && (
                            <ScoreCard label="Vector" value={r.vector_score} desc="Cosine similarity between embeddings" range="0–1, >0.5 strong" color="purple" dim={mode === 'keyword'} />
                          )}
                          {r.keyword_score != null && (
                            <ScoreCard label="Keyword" value={r.keyword_score} desc="BM25 term-frequency score" range="0–∞, raw BM25" color="green" dim={mode === 'vector'} maxValue={Math.max(r.keyword_score, 1)} />
                          )}
                          {r.temporal_score != null && (
                            <ScoreCard label="Temporal" value={r.temporal_score} desc="Exponential decay by age" range="0–1, 1 = just now" color="teal" />
                          )}
                        </div>

                        <div className="flex items-center gap-3">
                          <span className="text-xs text-ink-muted">Importance:</span>
                          <ImportanceDots value={r.importance ?? 5} />
                          <span className="text-xs text-ink-muted font-mono">{r.importance ?? 5}/10</span>
                        </div>

                        {/* Mutual links to other search results */}
                        {mutualLinks.length > 0 && (
                          <div>
                            <span className="text-[10px] text-ink-muted mb-1 block">Linked to other results:</span>
                            <div className="flex flex-wrap gap-1.5">
                              {mutualLinks.map(link => (
                                <button
                                  key={link.id}
                                  onClick={(e) => { e.stopPropagation(); setExpandedId(link.id) }}
                                  className="text-[10px] px-2 py-1 rounded bg-accent/10 text-accent border border-accent/20 hover:bg-accent/15 transition-colors truncate max-w-[200px]"
                                >
                                  ↔ {link.content} ({link.similarity.toFixed(2)})
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="text-[10px] text-ink-faint">
                          ID: {r.id} · Created: {new Date(r.created_at).toLocaleString()} · Updated: {new Date(r.updated_at).toLocaleString()}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Graph neighbor cards (always visible if they exist) */}
                {neighbors.length > 0 && (
                  <NeighborCards neighbors={neighbors} />
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function NeighborCards({ neighbors }: {
  neighbors: { id: string; content: string; similarity: number; edgeType: string }[]
}) {
  const [collapsed, setCollapsed] = useState(true)

  return (
    <div className="ml-9 pl-4 border-l-2 border-accent/20">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center gap-1 text-[10px] text-accent/60 hover:text-accent py-1 transition-colors"
      >
        {collapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
        <Link2 size={9} />
        {neighbors.length} graph neighbor{neighbors.length > 1 ? 's' : ''}
      </button>

      {!collapsed && (
        <div className="space-y-1 pb-2">
          {neighbors.map(n => (
            <div
              key={n.id}
              className="bg-card/80 border border-warm-border/60 rounded-md px-3 py-2 flex items-center gap-3"
            >
              <span className="text-[9px] text-accent/40 shrink-0 font-mono">graph</span>
              <p className="text-xs text-ink-muted flex-1 min-w-0 truncate">{n.content}</p>
              <span className="text-[10px] text-ink-faint font-mono shrink-0">
                {n.edgeType === 'superseded_by' ? 'superseded' : n.similarity.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ScoreCard({ label, value, desc, range, color, dim, maxValue }: {
  label: string; value: number; desc: string; range: string; color: string; dim?: boolean; maxValue?: number
}) {
  const colors: Record<string, { bg: string; bar: string; text: string }> = {
    blue: { bg: 'bg-accent/8', bar: 'bg-accent', text: 'text-accent' },
    purple: { bg: 'bg-purple-500/10', bar: 'bg-purple-500', text: 'text-purple-600' },
    green: { bg: 'bg-green-500/5', bar: 'bg-green-500', text: 'text-green-600' },
    teal: { bg: 'bg-teal-500/5', bar: 'bg-teal-500', text: 'text-teal-600' },
  }
  const c = colors[color] || colors.blue
  const normalizedMax = maxValue ?? 1
  const pct = Math.min((value / normalizedMax) * 100, 100)

  return (
    <div className={`rounded-lg p-2.5 ${c.bg} ${dim ? 'opacity-30' : ''}`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`text-[10px] font-medium ${c.text}`}>{label}</span>
        <span className="text-xs font-mono text-ink-light">{value.toFixed(3)}</span>
      </div>
      <div className="w-full h-1 bg-paper-deep rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${c.bar}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[9px] text-ink-faint mt-1">{desc}</p>
      <p className="text-[8px] text-ink-faint mt-0.5">{range}</p>
    </div>
  )
}
