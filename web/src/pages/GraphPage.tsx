import { useState, useCallback, useMemo, useRef, useEffect, useLayoutEffect, memo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { cssVar } from '../lib/theme'
import ForceGraph2D from 'react-force-graph-2d'
import { X, Tag, Pencil, Trash2, Check, Maximize2, Share2 } from 'lucide-react'
import { categoryColor, provenanceText } from '../lib/memoryMeta'
import type { GraphData, Memory } from '../api/types'

const STATUS_COLORS: Record<string, string> = {
  active: '#5B8C5A',
  superseded: '#C49A3C',
  deleted: '#C25B4E',
}

const TYPE_NODE_COLOR: Record<string, string> = {
  raw:       '#8A94A0',
  reflected: '#8B7DB8',
  deduced:   '#5B9CA6',
  induced:   '#5BA88C',
}

const TYPE_COLOR: Record<string, string> = {
  raw:       'bg-stone-500/15 text-stone-600',
  reflected: 'bg-violet-500/15 text-violet-600',
  deduced:   'bg-sky-500/15 text-sky-600',
  induced:   'bg-emerald-500/15 text-emerald-600',
}

const STATUS_BADGE: Record<string, string> = {
  active: 'badge-success',
  superseded: 'badge-warning',
  deleted: 'badge-danger',
}

/** Word-wrap text into lines that fit within maxWidth (canvas pixels). */
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number, maxLines: number): string[] {
  const words = text.split(/\s+/)
  const lines: string[] = []
  let current = ''

  for (const word of words) {
    const test = current ? `${current} ${word}` : word
    if (ctx.measureText(test).width > maxWidth && current) {
      lines.push(current)
      if (lines.length >= maxLines) break
      current = word
    } else {
      current = test
    }
  }

  if (current && lines.length < maxLines) {
    lines.push(current)
  }

  if (lines.length === maxLines) {
    const last = lines[maxLines - 1]
    if (ctx.measureText(last).width > maxWidth || words.length > lines.join(' ').split(/\s+/).length) {
      let trimmed = last
      while (trimmed && ctx.measureText(trimmed + '…').width > maxWidth) {
        trimmed = trimmed.slice(0, -1)
      }
      lines[maxLines - 1] = trimmed + '…'
    }
  }

  return lines
}

const nodeCanvasObjectModeReplace = () => 'replace' as const

const MemoGraph = memo(function MemoGraph({
  graphRef, graphData, dimensions, nodeColor, nodeSize, linkColor, linkWidth, nodeCanvasObject, nodePointerAreaPaint, onNodeClick, onBackgroundClick, onEngineStop,
}: {
  graphRef: React.RefObject<any>
  graphData: { nodes: any[]; links: any[] }
  dimensions: { width: number; height: number }
  nodeColor: (node: any) => string
  nodeSize: (node: any) => number
  linkColor: (link: any) => string
  linkWidth: (link: any) => number
  nodeCanvasObject: (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => void
  nodePointerAreaPaint: (node: any, color: string, ctx: CanvasRenderingContext2D) => void
  onNodeClick: (node: any) => void
  onBackgroundClick: () => void
  onEngineStop: () => void
}) {
  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={graphData}
      width={dimensions.width}
      height={dimensions.height}
      nodeLabel=""
      nodeColor={nodeColor}
      nodeVal={nodeSize}
      linkColor={linkColor}
      linkWidth={linkWidth}
      linkDirectionalArrowLength={0}
      backgroundColor="transparent"
      onNodeClick={onNodeClick}
      onBackgroundClick={onBackgroundClick}
      onEngineStop={onEngineStop}
      enableNodeDrag={false}
      cooldownTicks={100}
      d3AlphaDecay={0.1}
      nodeCanvasObjectMode={nodeCanvasObjectModeReplace}
      nodeCanvasObject={nodeCanvasObject}
      nodePointerAreaPaint={nodePointerAreaPaint}
    />
  )
})

export function GraphPage() {
  const [threshold, setThreshold] = useState(0.75)
  const [showSuperseded, setShowSuperseded] = useState(true)
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null)
  const [loadingMemory, setLoadingMemory] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editContent, setEditContent] = useState('')
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const graphRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })
  const queryClient = useQueryClient()

  const updateMutation = useMutation({
    mutationFn: ({ id, content }: { id: string; content: string }) => api.updateMemory(id, { content }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] })
      if (selectedMemory) api.getMemory(selectedMemory.id).then(setSelectedMemory)
      setIsEditing(false)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteMemory(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graph'] })
      setSelectedMemory(null)
      setShowDeleteConfirm(false)
    },
  })

  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      if (width > 0 && height > 0) setDimensions({ width, height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const { data, isLoading } = useQuery({
    queryKey: ['graph', getNamespace(), threshold],
    queryFn: () => api.getGraph(threshold, 200),
  })

  const graphData = useMemo(
    () => data ? transformGraphData(data, showSuperseded) : { nodes: [], links: [] },
    [data, showSuperseded],
  )

  const onEngineStop = useCallback(() => {
    graphRef.current?.zoomToFit(400, 40)
  }, [])

  const handleBackgroundClick = useCallback(() => {
    setSelectedMemory(null)
    graphRef.current?.zoomToFit(400, 40)
  }, [])

  const handleNodeClick = useCallback((node: any) => {
    if (!node.id) return
    setIsEditing(false)
    setShowDeleteConfirm(false)
    const fg = graphRef.current
    if (fg && node.x !== undefined && node.y !== undefined) {
      fg.centerAt(node.x, node.y, 300)
      fg.zoom(4, 300)
    }
    setLoadingMemory(true)
    api.getMemory(node.id)
      .then(memory => setSelectedMemory(memory))
      .catch(() => setSelectedMemory(null))
      .finally(() => setLoadingMemory(false))
  }, [])

  const nodeColor = useCallback((node: any) => {
    if (node.status === 'superseded') return STATUS_COLORS.superseded
    return TYPE_NODE_COLOR[node.memory_type] || TYPE_NODE_COLOR.raw
  }, [])
  const nodeSize = useCallback((node: any) => Math.max(3, Math.min(12, 3 + (node.access_count || 0))), [])

  const linkColor = useCallback((link: any) => {
    if (link.edge_type === 'superseded') return 'rgba(239,68,68,0.5)'
    return `rgba(59,130,246,${0.1 + (link.similarity || 0) * 0.5})`
  }, [])

  const linkWidth = useCallback((link: any) => {
    return link.edge_type === 'superseded' ? 2 : 0.5 + (link.similarity || 0) * 3
  }, [])

  const nodePointerAreaPaint = useCallback((node: any, color: string, ctx: CanvasRenderingContext2D) => {
    const r = Math.sqrt(Math.max(3, Math.min(12, 3 + (node.access_count || 0)))) * 2 + 4
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI)
    ctx.fill()
  }, [])

  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const nodeR = Math.sqrt(Math.max(3, Math.min(12, 3 + (node.access_count || 0)))) * 2
    const color = node.status === 'superseded'
      ? STATUS_COLORS.superseded
      : (TYPE_NODE_COLOR[node.memory_type] || TYPE_NODE_COLOR.raw)

    ctx.beginPath()
    ctx.arc(node.x!, node.y!, nodeR, 0, 2 * Math.PI)
    ctx.fillStyle = color
    ctx.fill()

    ctx.strokeStyle = color
    ctx.globalAlpha = 0.3
    ctx.lineWidth = 1.5 / globalScale
    ctx.stroke()
    ctx.globalAlpha = 1

    if (globalScale < 0.8) return
    const content = (node.content || '').trim()
    if (!content) return

    const fontSize = Math.min(12, Math.max(8, 10 / Math.sqrt(globalScale))) / globalScale
    const maxWidth = Math.max(60, 40 + nodeR * 8) / globalScale

    ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'

    const lines = wrapText(ctx, content, maxWidth, 2)
    const lineHeight = fontSize * 1.3
    const labelY = node.y! + nodeR + 3 / globalScale
    const textWidth = Math.max(...lines.map(l => ctx.measureText(l).width))
    const padX = 3 / globalScale, padY = 2 / globalScale
    const bgHeight = lines.length * lineHeight + padY * 2
    const bgWidth = textWidth + padX * 2
    const bgRadius = 2 / globalScale
    const bgX = node.x! - bgWidth / 2, bgY = labelY - padY

    ctx.fillStyle = cssVar('color-card')
    ctx.beginPath()
    ctx.moveTo(bgX + bgRadius, bgY)
    ctx.lineTo(bgX + bgWidth - bgRadius, bgY)
    ctx.quadraticCurveTo(bgX + bgWidth, bgY, bgX + bgWidth, bgY + bgRadius)
    ctx.lineTo(bgX + bgWidth, bgY + bgHeight - bgRadius)
    ctx.quadraticCurveTo(bgX + bgWidth, bgY + bgHeight, bgX + bgWidth - bgRadius, bgY + bgHeight)
    ctx.lineTo(bgX + bgRadius, bgY + bgHeight)
    ctx.quadraticCurveTo(bgX, bgY + bgHeight, bgX, bgY + bgHeight - bgRadius)
    ctx.lineTo(bgX, bgY + bgRadius)
    ctx.quadraticCurveTo(bgX, bgY, bgX + bgRadius, bgY)
    ctx.closePath()
    ctx.fill()

    ctx.fillStyle = cssVar('color-ink')
    lines.forEach((line, i) => { ctx.fillText(line, node.x!, labelY + i * lineHeight) })
  }, [])

  const closePanel = () => { setSelectedMemory(null); setIsEditing(false); setShowDeleteConfirm(false) }
  const panelOpen = !!selectedMemory || loadingMemory
  const isEmpty = !isLoading && graphData.nodes.length === 0

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Header */}
      <div className="shrink-0">
        <div className="flex flex-wrap items-center gap-2.5">
          <h1 className="page-title">Graph</h1>
          {data && (
            <>
              <span className="badge badge-neutral tabular-nums">{graphData.nodes.length} nodes</span>
              <span className="badge badge-neutral tabular-nums">{graphData.links.length} edges</span>
            </>
          )}
        </div>
        <p className="page-description">
          Memories linked by similarity. Scroll to zoom, drag to pan, click a node to inspect it.
        </p>
      </div>

      <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-4">
        {/* Canvas */}
        <section className="card flex-1 min-w-0 min-h-[360px] flex flex-col overflow-hidden" aria-label="Memory graph">
          {/* Toolbar: everything that changes what the canvas shows */}
          <div className="border-b border-warm-border px-4 py-2 flex flex-wrap items-center gap-x-5 gap-y-2 shrink-0">
            <label className="flex items-center gap-2 text-[13px] text-ink-muted min-h-8">
              <span>Similarity</span>
              <input
                type="range" min="0.3" max="0.9" step="0.05" value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-28"
                title="Minimum similarity for two memories to be linked"
              />
              <span className="font-mono text-xs text-ink tabular-nums w-8">{threshold.toFixed(2)}</span>
            </label>
            <label className="flex items-center gap-2 text-[13px] text-ink-muted min-h-8">
              <input type="checkbox" checked={showSuperseded} onChange={(e) => setShowSuperseded(e.target.checked)} />
              Show superseded
            </label>
            <button onClick={() => graphRef.current?.zoomToFit(400, 40)} className="btn btn-secondary ml-auto">
              <Maximize2 size={14} /> Fit
            </button>
          </div>

          <div ref={containerRef} className="flex-1 min-h-0 relative overflow-hidden">
            {isLoading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-ink-muted text-sm">
                <div className="w-6 h-6 border-2 border-warm-border-strong border-t-accent rounded-full animate-spin" />
                Loading graph…
              </div>
            ) : (
              <MemoGraph
                graphRef={graphRef}
                graphData={graphData}
                dimensions={dimensions}
                nodeColor={nodeColor}
                nodeSize={nodeSize}
                linkColor={linkColor}
                linkWidth={linkWidth}
                nodeCanvasObject={nodeCanvasObject}
                nodePointerAreaPaint={nodePointerAreaPaint}
                onNodeClick={handleNodeClick}
                onBackgroundClick={handleBackgroundClick}
                onEngineStop={onEngineStop}
              />
            )}

            {isEmpty && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 pointer-events-none">
                <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
                  <Share2 size={20} />
                </div>
                <p className="text-sm font-medium text-ink">Nothing to draw yet</p>
                <p className="text-[13px] text-ink-muted mt-1 max-w-xs">
                  Add a few memories and they will appear here, linked when their meaning is close.
                </p>
              </div>
            )}

            {/* Legend */}
            {!isLoading && !isEmpty && (
              <div className="absolute bottom-3 left-3 card px-3 py-2 text-xs max-w-[calc(100%-1.5rem)]">
                <ul className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  {Object.entries(TYPE_NODE_COLOR).map(([type, color]) => (
                    <li key={type} className="flex items-center gap-1.5 text-ink-muted capitalize">
                      <span className="w-2 h-2 rounded-full" style={{ background: color }} aria-hidden="true" />
                      {type}
                    </li>
                  ))}
                  <li className="flex items-center gap-1.5 text-ink-muted">
                    <span className="w-2 h-2 rounded-full" style={{ background: STATUS_COLORS.superseded }} aria-hidden="true" />
                    Superseded
                  </li>
                </ul>
              </div>
            )}
          </div>
        </section>

        {/* Details panel */}
        {panelOpen && (
          <aside className="card lg:w-80 shrink-0 flex flex-col lg:min-h-0 overflow-hidden" aria-label="Memory details">
            <div className="flex items-center justify-between gap-2 pl-4 pr-2 py-2 border-b border-warm-border shrink-0">
              <h2 className="text-sm font-semibold text-ink">Memory</h2>
              <button onClick={closePanel} className="btn btn-ghost btn-icon" aria-label="Close details" title="Close">
                <X size={16} />
              </button>
            </div>

            {loadingMemory ? (
              <div className="p-4 space-y-2" aria-busy="true">
                <div className="h-4 rounded bg-paper-warm animate-pulse" />
                <div className="h-4 rounded bg-paper-warm animate-pulse w-4/5" />
                <div className="h-4 rounded bg-paper-warm animate-pulse w-3/5" />
              </div>
            ) : selectedMemory && (
              <>
                <div className="flex-1 lg:min-h-0 lg:overflow-y-auto p-4 space-y-4">
                  {isEditing ? (
                    <textarea
                      value={editContent}
                      onChange={(e) => setEditContent(e.target.value)}
                      aria-label="Memory content"
                      className="field resize-y min-h-[96px]"
                      rows={5}
                      autoFocus
                    />
                  ) : (
                    <p className="text-sm text-ink leading-relaxed break-words">{selectedMemory.content}</p>
                  )}

                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className={`badge ${STATUS_BADGE[selectedMemory.status] || 'badge-neutral'}`}>{selectedMemory.status}</span>
                    <span className={`badge ${TYPE_COLOR[selectedMemory.memory_type] || TYPE_COLOR.raw}`}>{selectedMemory.memory_type}</span>
                    {selectedMemory.category && (
                      <span className={`badge ${categoryColor(selectedMemory.category)}`}>{selectedMemory.category}</span>
                    )}
                    {selectedMemory.tags.map(tag => (
                      <span key={tag} className="badge badge-neutral normal-case">
                        <Tag size={11} />{tag}
                      </span>
                    ))}
                  </div>

                  {provenanceText(selectedMemory.source, selectedMemory.confidence) && (
                    <p className="text-xs text-ink-muted">{provenanceText(selectedMemory.source, selectedMemory.confidence)}</p>
                  )}

                  <div>
                    <p className="section-label mb-2">Details</p>
                    <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-[13px]">
                      <div><dt className="text-xs text-ink-faint">Created</dt><dd className="text-ink-light">{new Date(selectedMemory.created_at).toLocaleDateString()}</dd></div>
                      <div><dt className="text-xs text-ink-faint">Updated</dt><dd className="text-ink-light">{new Date(selectedMemory.updated_at).toLocaleDateString()}</dd></div>
                      <div><dt className="text-xs text-ink-faint">Accessed</dt><dd className="text-ink-light tabular-nums">{selectedMemory.access_count}×</dd></div>
                      <div className="min-w-0"><dt className="text-xs text-ink-faint">Namespace</dt><dd className="text-ink-light truncate" title={selectedMemory.namespace}>{selectedMemory.namespace}</dd></div>
                    </dl>
                  </div>

                  {selectedMemory.metadata && Object.keys(selectedMemory.metadata).length > 0 && (
                    <div>
                      <p className="section-label mb-2">Metadata</p>
                      <pre className="text-xs text-ink-muted bg-paper-warm border border-warm-border rounded-lg p-2.5 overflow-x-auto">
                        {JSON.stringify(selectedMemory.metadata, null, 2)}
                      </pre>
                    </div>
                  )}

                  <p className="text-xs text-ink-faint font-mono truncate" title={selectedMemory.id}>{selectedMemory.id}</p>
                </div>

                {/* Actions */}
                {isEditing ? (
                  <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-warm-border shrink-0">
                    <button onClick={() => setIsEditing(false)} className="btn btn-ghost">Cancel</button>
                    <button
                      onClick={() => updateMutation.mutate({ id: selectedMemory.id, content: editContent })}
                      disabled={updateMutation.isPending || editContent.trim() === ''}
                      className="btn btn-primary"
                    >
                      <Check size={15} /> {updateMutation.isPending ? 'Saving…' : 'Save'}
                    </button>
                  </div>
                ) : showDeleteConfirm ? (
                  <div className="px-4 py-3 border-t border-warm-border bg-danger-light/50 shrink-0" role="alertdialog" aria-label="Confirm delete">
                    <p className="text-[13px] text-ink">Delete this memory? It moves to Deleted and leaves the graph.</p>
                    <div className="flex items-center justify-end gap-2 mt-2.5">
                      <button onClick={() => setShowDeleteConfirm(false)} className="btn btn-secondary">Cancel</button>
                      <button onClick={() => deleteMutation.mutate(selectedMemory.id)} disabled={deleteMutation.isPending} className="btn btn-danger">
                        <Trash2 size={15} /> {deleteMutation.isPending ? 'Deleting…' : 'Delete'}
                      </button>
                    </div>
                  </div>
                ) : selectedMemory.status === 'active' && (
                  <div className="flex items-center gap-2 px-4 py-3 border-t border-warm-border shrink-0">
                    <button onClick={() => { setEditContent(selectedMemory.content); setIsEditing(true) }} className="btn btn-secondary flex-1">
                      <Pencil size={15} /> Edit
                    </button>
                    <button onClick={() => setShowDeleteConfirm(true)} className="btn btn-danger flex-1">
                      <Trash2 size={15} /> Delete
                    </button>
                  </div>
                )}
              </>
            )}
          </aside>
        )}
      </div>
    </div>
  )
}

function transformGraphData(data: GraphData, showSuperseded: boolean) {
  const nodes = showSuperseded ? data.nodes : data.nodes.filter(n => n.status !== 'superseded')
  const nodeIds = new Set(nodes.map(n => n.id))
  const links = data.edges
    .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
    .map(e => ({ ...e, source: e.source, target: e.target }))
  return { nodes, links }
}
