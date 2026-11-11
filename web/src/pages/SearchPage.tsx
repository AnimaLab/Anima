import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { categoryColor, provenanceText } from '../lib/memoryMeta'
import type { SearchResponse, SearchResult, GraphData, AskResponse } from '../api/types'
import { loadCognitiveConfig } from './SettingsPage'
import { Search, Tag, Clock, Zap, BookOpen, ArrowUpDown, Shield, Link2, ChevronDown, ChevronRight, MessageSquare, Compass, X, Loader2 } from 'lucide-react'

type SortField = 'score' | 'vector_score' | 'keyword_score' | 'temporal_score' | 'importance' | 'created_at'
type SortDir = 'desc' | 'asc'
type AdjEntry = { id: string; content: string; similarity: number; edgeType: string }

const MEMORY_TYPES = ['all', 'fact', 'preference', 'event', 'story', 'decision', 'reflection', 'context', 'goal', 'relationship', 'emotion', 'habit', 'belief', 'skill', 'location'] as const

const typeColor: Record<string, string> = {
  preference: 'bg-purple-500/15 text-purple-700',
  fact: 'bg-blue-500/15 text-blue-700',
  event: 'bg-cyan-500/15 text-cyan-700',
  decision: 'bg-orange-500/15 text-orange-700',
  story: 'bg-orange-500/15 text-orange-700',
  reflection: 'bg-indigo-500/15 text-indigo-700',
  context: 'bg-stone-500/15 text-stone-600',
  goal: 'bg-emerald-500/15 text-emerald-700',
  relationship: 'bg-amber-500/15 text-amber-700',
  emotion: 'bg-red-500/15 text-red-700',
  habit: 'bg-teal-500/15 text-teal-700',
  belief: 'bg-amber-500/15 text-amber-700',
  skill: 'bg-lime-500/15 text-lime-700',
  location: 'bg-sky-500/15 text-sky-700',
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
  if (imp >= 9) return { text: 'Critical', color: 'text-danger' }
  if (imp >= 7) return { text: 'Important', color: 'text-warning' }
  if (imp >= 4) return { text: 'Normal', color: 'text-ink-muted' }
  return { text: 'Minor', color: 'text-ink-faint' }
}

function ImportanceDots({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-0.5" aria-hidden="true">
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          className={`w-1 h-3 rounded-sm ${
            i < value
              ? value >= 9 ? 'bg-danger' : value >= 7 ? 'bg-warning' : value >= 4 ? 'bg-accent' : 'bg-ink-faint'
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

  const isPending = searchMut.isPending || askMut.isPending || discoverMut.isPending
  const neighborCount = totalContext - sortedResults.length
  // /ask may report conflicts and a degraded (no-LLM) answer; neither is typed yet.
  const askExtras = askResult as unknown as { conflicts?: unknown[]; degraded?: boolean } | null
  const askConflictCount = Array.isArray(askExtras?.conflicts) ? askExtras!.conflicts!.length : 0
  const topScore = sortedResults[0]?.score || 1

  const SORT_OPTIONS: [SortField, string][] = [
    ['score', 'Score'],
    ['vector_score', 'Vector'],
    ['keyword_score', 'Keyword'],
    ['temporal_score', 'Temporal'],
    ['importance', 'Importance'],
    ['created_at', 'Date'],
  ]

  return (
    <div className="page space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
        <div>
          <h1 className="page-title">Search</h1>
          <p className="page-description">
            {searchType === 'ask'
              ? 'Ask a question and get an answer drawn from your memories'
              : 'Find memories by meaning, keywords and recency'}
          </p>
        </div>
        <p className="text-xs text-ink-faint tabular-nums" aria-live="polite">
          {results && (
            <>
              <span className="text-ink-muted">
                {results.results.length} direct + {totalContext - results.results.length} graph = {totalContext} total
              </span>
              {' · '}
              <span className="font-mono">{results.query_time_ms.toFixed(1)}ms</span>
            </>
          )}
          {askResult && (
            <>
              <span className="text-ink-muted">{askResult.total_search_results} memories searched</span>
              {' · '}
              <span className="font-mono">{askResult.elapsed_ms.toFixed(0)}ms</span>
            </>
          )}
        </p>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="space-y-3" role="search">
        <div className="relative">
          {searchType === 'ask'
            ? <MessageSquare size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none" />
            : <Search size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-faint pointer-events-none" />
          }
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label={searchType === 'ask' ? 'Question' : 'Search query'}
            placeholder={searchType === 'ask' ? 'Ask a question about your memories…' : 'Search memories… try “allergies”, “family”, “work”'}
            className="field h-11 pl-11 pr-28 text-[15px]"
          />
          <button
            type="submit"
            disabled={!query.trim() || isPending}
            className="btn btn-primary absolute right-1.5 top-1/2 -translate-y-1/2"
          >
            {isPending
              ? <Loader2 size={15} className="animate-spin" />
              : searchType === 'ask' ? <MessageSquare size={15} /> : <Search size={15} />}
            {searchType === 'ask' ? 'Ask' : 'Search'}
          </button>
        </div>

        {/* Toolbar: what kind of query, how to match, and tuning */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <div className="segmented" role="group" aria-label="Query type">
            <button type="button" onClick={() => setSearchType('search')} aria-pressed={searchType === 'search'} className="inline-flex items-center gap-1.5">
              <Search size={14} /> Search
            </button>
            <button type="button" onClick={() => setSearchType('ask')} aria-pressed={searchType === 'ask'} className="inline-flex items-center gap-1.5">
              <MessageSquare size={14} /> Ask
            </button>
          </div>

          {/* Matching and tuning only apply to classic search */}
          {searchType === 'search' && (
            <>
              <div className="segmented" role="group" aria-label="Search mode">
                {['hybrid', 'vector', 'keyword'].map(m => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => handleModeSwitch(m)}
                    aria-pressed={mode === m}
                    className="inline-flex items-center gap-1.5"
                  >
                    {m === 'hybrid' && <Zap size={14} />}
                    {m === 'vector' && <BookOpen size={14} />}
                    {m === 'keyword' && <Search size={14} />}
                    {m}
                  </button>
                ))}
              </div>

              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 sm:ml-auto">
                <label className="flex items-center gap-2 text-[13px] text-ink-muted min-h-8" title="How much newer memories are favoured">
                  <Clock size={14} className="text-ink-faint" />
                  Recency
                  <input
                    type="range" min="0" max="1" step="0.05" value={temporalWeight}
                    onChange={(e) => handleTemporalChange(parseFloat(e.target.value))}
                    className="w-24"
                  />
                  <span className="text-xs text-ink font-mono tabular-nums w-8">{temporalWeight.toFixed(2)}</span>
                </label>

                <label className="flex items-center gap-2 text-[13px] text-ink-muted">
                  Limit
                  <select
                    value={limit}
                    onChange={(e) => handleLimitChange(parseInt(e.target.value))}
                    className="field w-auto"
                  >
                    {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
              </div>
            </>
          )}
        </div>

        {/* Type filter: appears once a type badge has been clicked */}
        {searchType === 'search' && typeFilter !== 'all' && (
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by type">
            <span className="section-label mr-1">Type</span>
            {MEMORY_TYPES.map(t => (
              <button
                key={t}
                type="button"
                onClick={() => setTypeFilter(t)}
                aria-pressed={typeFilter === t}
                className={`min-h-8 px-2.5 rounded-md text-[13px] capitalize transition-colors ${
                  typeFilter === t ? 'bg-paper-deep text-ink font-medium' : 'text-ink-muted hover:text-ink hover:bg-paper-deep'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        )}
      </form>

      {/* Ask result */}
      {askResult && (
        <div className="space-y-4">
          <section className="card p-5" aria-label="Answer">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-7 h-7 rounded-full bg-accent-light text-accent flex items-center justify-center shrink-0">
                <MessageSquare size={14} />
              </span>
              <h2 className="text-sm font-semibold text-ink">Answer</h2>
              {askExtras?.degraded && (
                <span className="badge badge-warning normal-case" title="The LLM backend was unreachable, so this is retrieval output rather than a generated answer">
                  No LLM: retrieval only
                </span>
              )}
            </div>
            <p className="text-[15px] text-ink leading-relaxed whitespace-pre-wrap">{askResult.answer}</p>

            {askConflictCount > 0 && (
              <div className="mt-4 flex items-start gap-2 rounded-lg bg-warning-light px-3 py-2.5">
                <Shield size={15} className="text-warning shrink-0 mt-0.5" />
                <p className="text-[13px] text-warning">
                  {askConflictCount} conflicting {askConflictCount === 1 ? 'memory was' : 'memories were'} found among the sources.
                  Review them on the Conflicts page before relying on this answer.
                </p>
              </div>
            )}
          </section>

          {askResult.memories_referenced.length > 0 && (
            <section aria-label="Sources">
              <h2 className="section-label mb-2">
                Sources · {askResult.memories_referenced.length}
              </h2>
              <ol className="card divide-y divide-warm-border overflow-hidden">
                {askResult.memories_referenced.slice(0, 20).map((mem, i) => (
                  <li key={mem.id} className="flex items-baseline gap-3 px-4 py-2.5">
                    <span className="text-xs text-ink-faint font-mono tabular-nums w-5 text-right shrink-0">{i + 1}</span>
                    <p className="text-[13px] text-ink-light flex-1 min-w-0 truncate" title={mem.content}>{mem.content}</p>
                    <span className="text-xs text-ink-muted font-mono tabular-nums shrink-0">{mem.score.toFixed(3)}</span>
                  </li>
                ))}
              </ol>
              {askResult.memories_referenced.length > 20 && (
                <p className="text-xs text-ink-faint mt-2">and {askResult.memories_referenced.length - 20} more</p>
              )}
            </section>
          )}
        </div>
      )}

      {/* Ask error */}
      {askMut.isError && (
        <div className="flex items-start gap-2 rounded-lg bg-danger-light px-4 py-3" role="alert">
          <Shield size={15} className="text-danger shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-danger font-medium">{(askMut.error as Error).message}</p>
            <p className="text-[13px] text-ink-muted mt-0.5">Make sure the LLM is configured in Settings.</p>
          </div>
        </div>
      )}

      {/* Initial state: suggest what to try */}
      {!results && !askResult && !isPending && (
        <div className="card flex flex-col items-center text-center px-6 py-14">
          <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
            {searchType === 'ask' ? <MessageSquare size={20} /> : <Search size={20} />}
          </div>
          <p className="text-sm font-medium text-ink">
            {searchType === 'ask' ? 'Ask a question and get an answer drawn from your memories' : 'Search by meaning, not just keywords'}
          </p>
          <p className="text-[13px] text-ink-muted mt-1">Try one of these to get started</p>
          <div className="flex flex-wrap justify-center gap-2 mt-4">
            {(searchType === 'ask'
              ? ['What do you know about me?', 'What am I working on?', 'What are my preferences?']
              : ['work', 'preferences', 'daily routine', 'current projects']
            ).map(example => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setQuery(example)
                  if (searchType === 'ask') askMut.mutate(example)
                  else doSearch(example, mode, limit, temporalWeight)
                }}
                className="btn btn-secondary rounded-full font-normal"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* No results (search mode only) */}
      {results && results.results.length === 0 && (
        <div className="card flex flex-col items-center text-center px-6 py-14">
          <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
            <Search size={20} />
          </div>
          <p className="text-sm font-medium text-ink break-words max-w-full">No memories found for “{query}”</p>
          <p className="text-[13px] text-ink-muted mt-1">Try a different query or switch search mode.</p>
        </div>
      )}

      {/* Discovery mode banner */}
      {discoverSource && (
        <div className="flex items-center gap-2 rounded-lg bg-accent-light pl-4 pr-1.5 py-1.5">
          <Compass size={15} className="text-accent shrink-0" />
          <span className="text-[13px] font-medium text-accent shrink-0">Similar to</span>
          <span className="text-[13px] text-ink truncate flex-1 min-w-0">{discoverSource.content}</span>
          <button
            type="button"
            onClick={() => { setDiscoverSource(null); setResults(null) }}
            className="btn btn-ghost btn-icon shrink-0"
            title="Clear discovery"
            aria-label="Clear discovery"
          >
            <X size={15} />
          </button>
        </div>
      )}

      {/* Results */}
      {sortedResults.length > 0 && (
        <section className="space-y-3" aria-label="Results">
          {/* Summary and sort */}
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <p className="text-[13px] text-ink-muted">
              <span className="font-medium text-ink">{sortedResults.length} {sortedResults.length === 1 ? 'result' : 'results'}</span>
              {neighborCount > 0 && <> · {neighborCount} graph {neighborCount === 1 ? 'neighbor' : 'neighbors'}</>}
            </p>
            <div className="flex items-center gap-1.5">
              <label htmlFor="search-sort" className="text-[13px] text-ink-muted flex items-center gap-1.5">
                <ArrowUpDown size={14} className="text-ink-faint" /> Sort
              </label>
              <select
                id="search-sort"
                value={sortField}
                onChange={(e) => { setSortField(e.target.value as SortField); setSortDir('desc') }}
                className="field w-auto"
              >
                {SORT_OPTIONS.map(([field, label]) => <option key={field} value={field}>{label}</option>)}
              </select>
              <button
                type="button"
                onClick={() => toggleSort(sortField)}
                className="btn btn-secondary btn-icon"
                title={sortDir === 'desc' ? 'Highest first. Click for lowest first' : 'Lowest first. Click for highest first'}
                aria-label={sortDir === 'desc' ? 'Sorted highest first, switch to lowest first' : 'Sorted lowest first, switch to highest first'}
              >
                <span aria-hidden="true" className="text-sm leading-none">{sortDir === 'desc' ? '↓' : '↑'}</span>
              </button>
            </div>
          </div>

          {/* One list, one surface */}
          <ul className="card divide-y divide-warm-border overflow-hidden">
            {sortedResults.map((r, i) => {
              const neighbors = graphNeighbors.get(r.id) || []
              const isExpanded = expandedId === r.id
              // Other results this one is linked to in the graph
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
              const provenance = provenanceText(r.source, r.confidence)
              const importance = r.importance ?? 5

              return (
                <li
                  key={r.id}
                  className={`px-4 sm:px-5 py-4 transition-colors ${isExpanded ? 'bg-paper-warm' : 'hover:bg-paper/60'} ${isNoise ? 'opacity-50' : ''}`}
                >
                  <div className="flex items-start gap-3">
                    {/* Rank */}
                    <span className="text-[13px] font-medium text-ink-faint tabular-nums w-5 text-right shrink-0 mt-0.5">{i + 1}</span>

                    <div className="flex-1 min-w-0">
                      <button
                        type="button"
                        onClick={() => setExpandedId(isExpanded ? null : r.id)}
                        aria-expanded={isExpanded}
                        title={isExpanded ? 'Hide score details' : 'Show score details'}
                        className="block w-full text-left text-sm text-ink leading-relaxed rounded break-words"
                      >
                        {r.content}
                      </button>

                      {/* Meta */}
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 mt-2 text-xs text-ink-faint">
                        {r.memory_type !== 'raw' && (
                          <button
                            type="button"
                            onClick={() => setTypeFilter(typeFilter === r.memory_type ? 'all' : r.memory_type)}
                            className={`badge hover:opacity-80 transition-opacity ${typeColor[r.memory_type] || 'badge-neutral'}`}
                            title={`Filter by ${r.memory_type}`}
                          >
                            {r.memory_type}
                          </button>
                        )}
                        {r.category && r.category !== 'general' && (
                          <span className={`badge ${categoryColor(r.category)}`}>{r.category}</span>
                        )}
                        {provenance && <span className="text-ink-muted">{provenance}</span>}
                        {r.tags.map(tag => (
                          <span key={tag} className="badge badge-neutral normal-case">
                            <Tag size={11} />{tag}
                          </span>
                        ))}
                        {(neighbors.length > 0 || mutualLinks.length > 0) && (
                          <span className="inline-flex items-center gap-1 text-ink-muted">
                            <Link2 size={12} />
                            {neighbors.length + mutualLinks.length} linked
                          </span>
                        )}
                        <span>{relativeDate(r.created_at)}</span>
                        <button
                          type="button"
                          onClick={() => {
                            setDiscoverSource({ id: r.id, content: r.content })
                            discoverMut.mutate(r.id)
                          }}
                          className="inline-flex items-center gap-1 min-h-6 px-1.5 -my-1 rounded-md font-medium text-accent hover:bg-accent-light transition-colors"
                          title="Find similar memories"
                        >
                          <Compass size={12} /> Find similar
                        </button>
                      </div>
                    </div>

                    {/* Score */}
                    <div className="w-14 shrink-0 text-right">
                      <div className="text-sm font-mono font-semibold text-ink tabular-nums">{r.score.toFixed(2)}</div>
                      <div className="mt-1 h-1 rounded-full bg-paper-deep overflow-hidden" aria-hidden="true">
                        <div
                          className={`h-full rounded-full ${r.score > 0.7 ? 'bg-success' : r.score > 0.2 ? 'bg-accent' : 'bg-ink-faint'}`}
                          style={{ width: `${Math.min((r.score / topScore) * 100, 100)}%` }}
                        />
                      </div>
                      {importance !== 5 && (
                        <div className={`text-xs mt-1 ${importanceLabel(importance).color}`}>
                          {importanceLabel(importance).text}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Expanded details */}
                  {isExpanded && (
                    <div className="mt-4 ml-8 pt-4 border-t border-warm-border space-y-4">
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                        <ScoreCard label="Combined" value={r.score} desc="Final score after fusion, recency and boosts" range="0–1" color="blue" />
                        {r.vector_score != null && (
                          <ScoreCard label="Vector" value={r.vector_score} desc="Cosine similarity between embeddings" range="0–1, above 0.5 is strong" color="purple" dim={mode === 'keyword'} />
                        )}
                        {r.keyword_score != null && (
                          <ScoreCard label="Keyword" value={r.keyword_score} desc="BM25 term-frequency score" range="0–∞, raw BM25" color="green" dim={mode === 'vector'} maxValue={Math.max(r.keyword_score, 1)} />
                        )}
                        {r.temporal_score != null && (
                          <ScoreCard label="Temporal" value={r.temporal_score} desc="Exponential decay by age" range="0–1, 1 is just now" color="teal" />
                        )}
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-[13px] text-ink-muted">Importance</span>
                        <ImportanceDots value={importance} />
                        <span className="text-xs text-ink-muted font-mono tabular-nums">{importance}/10</span>
                      </div>

                      {mutualLinks.length > 0 && (
                        <div>
                          <p className="section-label mb-1.5">Linked to other results</p>
                          <div className="flex flex-wrap gap-1.5">
                            {mutualLinks.map(link => (
                              <button
                                key={link.id}
                                type="button"
                                onClick={() => setExpandedId(link.id)}
                                className="btn btn-secondary font-normal max-w-full sm:max-w-[260px]"
                                title={link.content}
                              >
                                <Link2 size={13} className="shrink-0 text-ink-faint" />
                                <span className="truncate">{link.content}</span>
                                <span className="font-mono text-xs text-ink-muted shrink-0">{link.similarity.toFixed(2)}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-faint">
                        <div className="flex gap-1.5 min-w-0"><dt>ID</dt><dd className="font-mono text-ink-muted break-all">{r.id}</dd></div>
                        <div className="flex gap-1.5"><dt>Created</dt><dd className="text-ink-muted">{new Date(r.created_at).toLocaleString()}</dd></div>
                        <div className="flex gap-1.5"><dt>Updated</dt><dd className="text-ink-muted">{new Date(r.updated_at).toLocaleString()}</dd></div>
                      </dl>
                    </div>
                  )}

                  {/* Graph neighbours: context the model would also see */}
                  {neighbors.length > 0 && <NeighborCards neighbors={neighbors} />}
                </li>
              )
            })}
          </ul>
        </section>
      )}
    </div>
  )
}

function NeighborCards({ neighbors }: {
  neighbors: { id: string; content: string; similarity: number; edgeType: string }[]
}) {
  const [collapsed, setCollapsed] = useState(true)

  return (
    <div className="ml-8 mt-2">
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        className="inline-flex items-center gap-1.5 min-h-8 px-1.5 -ml-1.5 rounded-md text-[13px] text-ink-muted hover:text-ink hover:bg-paper-deep transition-colors"
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
        <Link2 size={13} className="text-ink-faint" />
        {neighbors.length} graph neighbor{neighbors.length > 1 ? 's' : ''}
      </button>

      {!collapsed && (
        <ul className="mt-1 border-l-2 border-warm-border pl-3 space-y-1.5 pb-1">
          {neighbors.map(n => (
            <li key={n.id} className="flex items-baseline gap-3">
              <p className="text-[13px] text-ink-light flex-1 min-w-0 truncate" title={n.content}>{n.content}</p>
              {n.edgeType === 'superseded_by'
                ? <span className="badge badge-warning shrink-0">superseded</span>
                : <span className="text-xs text-ink-muted font-mono tabular-nums shrink-0">{n.similarity.toFixed(2)}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ScoreCard({ label, value, desc, range, color, dim, maxValue }: {
  label: string; value: number; desc: string; range: string; color: string; dim?: boolean; maxValue?: number
}) {
  const bars: Record<string, string> = {
    blue: 'bg-accent',
    purple: 'bg-info',
    green: 'bg-success',
    teal: 'bg-warning',
  }
  const bar = bars[color] || bars.blue
  const normalizedMax = maxValue ?? 1
  const pct = Math.min((value / normalizedMax) * 100, 100)

  return (
    <div className={`rounded-lg border border-warm-border bg-card p-3 ${dim ? 'opacity-50' : ''}`}>
      <div className="flex items-baseline justify-between gap-2 mb-1.5">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        <span className="text-[13px] font-mono tabular-nums text-ink">{value.toFixed(3)}</span>
      </div>
      <div className="w-full h-1 bg-paper-deep rounded-full overflow-hidden" aria-hidden="true">
        <div className={`h-full rounded-full ${bar}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-ink-muted mt-2">{desc}</p>
      <p className="text-xs text-ink-faint mt-0.5">{range}</p>
    </div>
  )
}
