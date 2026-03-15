import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import type { SearchResponse, SearchResult, GraphData, AskResponse } from '../api/types'
import { loadCognitiveConfig } from './SettingsPage'
import { Search, Tag, Clock, Zap, BookOpen, ArrowUpDown, Shield, Link2, ChevronDown, ChevronRight, MessageSquare } from 'lucide-react'

type SortField = 'score' | 'vector_score' | 'keyword_score' | 'temporal_score' | 'importance' | 'created_at'
type SortDir = 'desc' | 'asc'
type AdjEntry = { id: string; content: string; similarity: number; edgeType: string }

const MEMORY_TYPES = ['all', 'fact', 'preference', 'event', 'story', 'decision', 'reflection', 'context', 'goal', 'relationship', 'emotion', 'habit', 'belief', 'skill', 'location'] as const

const typeColor: Record<string, string> = {
  preference: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  fact: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  event: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  decision: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  story: 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  reflection: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
  context: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  goal: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  relationship: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  emotion: 'bg-red-500/20 text-red-400 border-red-500/30',
  habit: 'bg-teal-500/20 text-teal-400 border-teal-500/30',
  belief: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  skill: 'bg-lime-500/20 text-lime-400 border-lime-500/30',
  location: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
}

const modeDescriptions: Record<string, string> = {
  hybrid: 'Combines semantic understanding with keyword matching for best overall results',
  vector: 'Pure semantic similarity — finds conceptually related memories even with different wording',
  keyword: 'Full-text search using Porter stemming — exact and partial word matches',
}

function importanceLabel(imp: number): { text: string; color: string } {
  if (imp >= 9) return { text: 'Critical', color: 'text-red-400' }
  if (imp >= 7) return { text: 'Important', color: 'text-orange-400' }
  if (imp >= 4) return { text: 'Normal', color: 'text-gray-400' }
  return { text: 'Minor', color: 'text-gray-600' }
}

function ImportanceDots({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          className={`w-1 h-3 rounded-sm ${
            i < value
              ? value >= 9 ? 'bg-red-500' : value >= 7 ? 'bg-orange-500' : value >= 4 ? 'bg-blue-500' : 'bg-gray-600'
              : 'bg-gray-800'
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
        <h2 className="text-lg font-semibold text-white">Memory Search</h2>
        <div className="flex items-center gap-3">
          {results && (
            <>
              <span className="text-xs text-gray-500">
                {results.results.length} direct + {totalContext - results.results.length} graph = {totalContext} total
              </span>
              <span className="text-[10px] text-gray-600">{results.query_time_ms.toFixed(1)}ms</span>
            </>
          )}
          {askResult && (
            <>
              <span className="text-xs text-gray-500">
                {askResult.total_search_results} memories searched
              </span>
              <span className="text-[10px] text-gray-600">{askResult.elapsed_ms.toFixed(0)}ms</span>
            </>
          )}
        </div>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            {searchType === 'ask'
              ? <MessageSquare size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-emerald-500" />
              : <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
            }
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={searchType === 'ask' ? "Ask a question about memories... (press Enter)" : "Search memories... (try: 'allergies', 'family', 'work')"}
              className={`w-full bg-gray-900 border rounded-lg pl-9 pr-3 py-2.5 text-sm text-gray-200 placeholder:text-gray-600 focus:outline-none focus:ring-1 ${
                searchType === 'ask'
                  ? 'border-emerald-700/50 focus:ring-emerald-500 focus:border-emerald-500'
                  : 'border-gray-700 focus:ring-blue-500 focus:border-blue-500'
              }`}
            />
          </div>
          {(searchMut.isPending || askMut.isPending) && (
            <div className="flex items-center px-3 text-gray-500">
              <div className={`w-4 h-4 border-2 border-gray-600 rounded-full animate-spin ${
                searchType === 'ask' ? 'border-t-emerald-500' : 'border-t-blue-500'
              }`} />
            </div>
          )}
        </div>

        {/* Search type toggle + mode tabs */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-6">
          {/* Search vs Ask toggle */}
          <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
            <button
              type="button"
              onClick={() => setSearchType('search')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                searchType === 'search'
                  ? 'bg-blue-600/20 text-blue-400 shadow-sm'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
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
                  ? 'bg-emerald-600/20 text-emerald-400 shadow-sm'
                  : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
              }`}
            >
              <MessageSquare size={12} />
              Ask
            </button>
          </div>

          {/* Classic search controls (hidden in ask mode) */}
          {searchType === 'search' && (
            <>
              <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
                {['hybrid', 'vector', 'keyword'].map(m => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => handleModeSwitch(m)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium capitalize transition-colors ${
                      mode === m
                        ? 'bg-blue-600/20 text-blue-400 shadow-sm'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
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
                <Clock size={12} className="text-gray-500" />
                <span className="text-xs text-gray-500">Recency:</span>
                <input type="range" min="0" max="1" step="0.05" value={temporalWeight}
                  onChange={(e) => handleTemporalChange(parseFloat(e.target.value))}
                  className="w-20 accent-teal-500" />
                <span className="text-xs text-gray-400 font-mono w-8">{temporalWeight.toFixed(2)}</span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Limit:</span>
                <select value={limit} onChange={(e) => handleLimitChange(parseInt(e.target.value))}
                  className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:outline-none">
                  {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
            </>
          )}
        </div>

        {searchType === 'search' && <p className="text-[11px] text-gray-600">{modeDescriptions[mode]}</p>}
        {searchType === 'ask' && <p className="text-[11px] text-gray-600">Ask a question and get a direct answer from your memories. Press Enter to ask.</p>}

        {/* Type filter (search mode only) */}
        {searchType === 'search' && (
          <div className="flex flex-wrap gap-1">
            {MEMORY_TYPES.map(t => (
              <button key={t} type="button" onClick={() => setTypeFilter(t)}
                className={`px-2.5 py-1 rounded-md text-xs capitalize transition-colors ${typeFilter === t ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
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
          <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-5">
            <div className="flex items-start gap-3">
              <MessageSquare size={18} className="text-emerald-400 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{askResult.answer}</p>
              </div>
            </div>
          </div>

          {/* Referenced memories */}
          {askResult.memories_referenced.length > 0 && (
            <div>
              <span className="text-[10px] text-gray-500 block mb-2">
                {askResult.memories_referenced.length} memories referenced
              </span>
              <div className="space-y-1">
                {askResult.memories_referenced.slice(0, 20).map((mem, i) => (
                  <div key={mem.id} className="bg-gray-900/60 border border-gray-800/60 rounded-md px-3 py-2 flex items-center gap-3">
                    <span className="text-[10px] text-gray-700 shrink-0 font-mono w-5 text-right">{i + 1}</span>
                    <p className="text-xs text-gray-400 flex-1 min-w-0 truncate">{mem.content}</p>
                    <span className="text-[10px] text-gray-600 font-mono shrink-0">{mem.score.toFixed(3)}</span>
                  </div>
                ))}
                {askResult.memories_referenced.length > 20 && (
                  <p className="text-[10px] text-gray-600 pl-8">...and {askResult.memories_referenced.length - 20} more</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Ask error */}
      {askMut.isError && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
          <p className="text-sm text-red-400">{(askMut.error as Error).message}</p>
          <p className="text-xs text-red-500/60 mt-1">Make sure the LLM is configured in Settings</p>
        </div>
      )}

      {/* Empty state (search mode only) */}
      {results && results.results.length === 0 && (
        <div className="text-center py-12">
          <Search size={32} className="mx-auto text-gray-700 mb-3" />
          <p className="text-gray-500 text-sm">No memories found for "{query}"</p>
          <p className="text-gray-600 text-xs mt-1">Try a different query or switch modes</p>
        </div>
      )}

      {/* Results */}
      {sortedResults.length > 0 && (
        <div className="space-y-3">
          {/* LLM context preview banner */}
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg px-4 py-2.5 flex items-center gap-3">
            <Zap size={14} className="text-blue-400 shrink-0" />
            <div className="text-xs text-blue-300">
              <span className="font-medium">LLM would see:</span>{' '}
              {sortedResults.length} search results + {totalContext - sortedResults.length} graph neighbors = {totalContext} memories in context
            </div>
          </div>

          {/* Sort controls */}
          <div className="flex items-center gap-1 flex-wrap">
            <ArrowUpDown size={12} className="text-gray-600 mr-1" />
            <span className="text-[11px] text-gray-600 mr-1">Sort:</span>
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
                  sortField === field ? 'bg-blue-600/20 text-blue-400' : 'text-gray-600 hover:text-gray-400'
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

            return (
              <div key={r.id} className="space-y-0">
                {/* Main result card */}
                <div
                  className={`bg-gray-900 border rounded-lg overflow-hidden transition-colors cursor-pointer ${
                    isExpanded ? 'border-blue-500/40' : 'border-gray-800 hover:border-gray-700'
                  }`}
                  onClick={() => setExpandedId(isExpanded ? null : r.id)}
                >
                  {/* Score indicator bar */}
                  <div className="h-0.5 bg-gray-800">
                    <div
                      className={`h-full transition-all ${
                        r.score > 0.7 ? 'bg-green-500' : r.score > 0.4 ? 'bg-blue-500' : r.score > 0.2 ? 'bg-amber-500' : 'bg-gray-600'
                      }`}
                      style={{ width: `${Math.min((r.score / (sortedResults[0]?.score || 1)) * 100, 100)}%` }}
                    />
                  </div>

                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {/* Rank number */}
                      <div className="flex flex-col items-center gap-1 shrink-0">
                        <span className="text-lg font-bold text-gray-700 tabular-nums w-6 text-right">{i + 1}</span>
                        <span className="text-[9px] text-gray-700">search</span>
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-gray-200 leading-relaxed">{r.content}</p>

                        {/* Tags + metadata */}
                        <div className="flex items-center gap-2 mt-2.5 flex-wrap">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${typeColor[r.memory_type] || typeColor.fact}`}>
                            {r.memory_type}
                          </span>
                          {r.tags.map(tag => (
                            <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-800 text-gray-500 flex items-center gap-0.5 border border-gray-700/50">
                              <Tag size={8} />{tag}
                            </span>
                          ))}
                          {(neighbors.length > 0 || mutualLinks.length > 0) && (
                            <span className="text-[10px] text-blue-400/70 flex items-center gap-0.5">
                              <Link2 size={9} />
                              {neighbors.length + mutualLinks.length} linked
                            </span>
                          )}
                          <span className="text-[10px] text-gray-600">{new Date(r.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>

                      {/* Score + importance */}
                      <div className="text-right shrink-0 space-y-1">
                        <div className="text-sm font-mono font-bold text-white">{r.score.toFixed(3)}</div>
                        <div className={`flex items-center gap-1 justify-end text-[10px] ${importanceLabel(r.importance ?? 5).color}`}>
                          <Shield size={9} />
                          {importanceLabel(r.importance ?? 5).text}
                        </div>
                      </div>
                    </div>

                    {/* Expanded details */}
                    {isExpanded && (
                      <div className="mt-4 pt-3 border-t border-gray-800 space-y-3">
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
                          <span className="text-xs text-gray-500">Importance:</span>
                          <ImportanceDots value={r.importance ?? 5} />
                          <span className="text-xs text-gray-400 font-mono">{r.importance ?? 5}/10</span>
                        </div>

                        {/* Mutual links to other search results */}
                        {mutualLinks.length > 0 && (
                          <div>
                            <span className="text-[10px] text-gray-500 mb-1 block">Linked to other results:</span>
                            <div className="flex flex-wrap gap-1.5">
                              {mutualLinks.map(link => (
                                <button
                                  key={link.id}
                                  onClick={(e) => { e.stopPropagation(); setExpandedId(link.id) }}
                                  className="text-[10px] px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 transition-colors truncate max-w-[200px]"
                                >
                                  ↔ {link.content} ({link.similarity.toFixed(2)})
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="text-[10px] text-gray-700">
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
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="ml-9 pl-4 border-l-2 border-blue-500/20">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center gap-1 text-[10px] text-blue-400/60 hover:text-blue-400 py-1 transition-colors"
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
              className="bg-gray-900/60 border border-gray-800/60 rounded-md px-3 py-2 flex items-center gap-3"
            >
              <span className="text-[9px] text-blue-400/40 shrink-0 font-mono">graph</span>
              <p className="text-xs text-gray-400 flex-1 min-w-0 truncate">{n.content}</p>
              <span className="text-[10px] text-gray-600 font-mono shrink-0">
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
    blue: { bg: 'bg-blue-500/5', bar: 'bg-blue-500', text: 'text-blue-400' },
    purple: { bg: 'bg-purple-500/5', bar: 'bg-purple-500', text: 'text-purple-400' },
    green: { bg: 'bg-green-500/5', bar: 'bg-green-500', text: 'text-green-400' },
    teal: { bg: 'bg-teal-500/5', bar: 'bg-teal-500', text: 'text-teal-400' },
  }
  const c = colors[color] || colors.blue
  const normalizedMax = maxValue ?? 1
  const pct = Math.min((value / normalizedMax) * 100, 100)

  return (
    <div className={`rounded-lg p-2.5 ${c.bg} ${dim ? 'opacity-30' : ''}`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`text-[10px] font-medium ${c.text}`}>{label}</span>
        <span className="text-xs font-mono text-gray-300">{value.toFixed(3)}</span>
      </div>
      <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${c.bar}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[9px] text-gray-600 mt-1">{desc}</p>
      <p className="text-[8px] text-gray-700 mt-0.5">{range}</p>
    </div>
  )
}
