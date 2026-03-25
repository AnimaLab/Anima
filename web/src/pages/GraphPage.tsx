import { useState, useCallback, useMemo, useRef, useEffect, useLayoutEffect, memo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import ForceGraph2D from 'react-force-graph-2d'
import { X, Tag, Pencil, Trash2, Check } from 'lucide-react'
import type { GraphData, Memory } from '../api/types'

const STATUS_COLORS: Record<string, string> = {
  active: '#5B8C5A',
  superseded: '#C49A3C',
  deleted: '#C25B4E',
}

const TYPE_NODE_COLOR: Record<string, string> = {
  raw:       '#9C9488',
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
  active: 'bg-green-500/15 text-green-700',
  superseded: 'bg-amber-500/15 text-amber-700',
  deleted: 'bg-red-500/15 text-red-700',
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

    ctx.fillStyle = 'rgba(255, 255, 255, 0.92)'
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

    ctx.fillStyle = 'rgba(45, 42, 38, 0.9)'
    lines.forEach((line, i) => { ctx.fillText(line, node.x!, labelY + i * lineHeight) })
  }, [])

  return (
    <div className="space-y-4 h-full max-h-[calc(100vh-3rem)] flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-ink">Graph</h2>
            {data && <span className="text-xs text-ink-faint">{data.nodes.length} nodes, {data.edges.length} edges</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 sm:gap-4">
          {/* Type legend */}
          <div className="hidden sm:flex items-center gap-2">
            {Object.entries(TYPE_NODE_COLOR).map(([type, color]) => (
              <div key={type} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                <span className="text-[10px] text-ink-faint capitalize">{type}</span>
              </div>
            ))}
          </div>
          <div className="hidden sm:block w-px h-3 bg-warm-border" />
          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-muted">Sim:</span>
            <input type="range" min="0.3" max="0.9" step="0.05" value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-20 accent-[#C47B3B]" />
            <span className="text-xs text-ink-muted font-mono">{threshold.toFixed(2)}</span>
          </div>
          <label className="flex items-center gap-1.5 text-xs text-ink-muted">
            <input type="checkbox" checked={showSuperseded} onChange={(e) => setShowSuperseded(e.target.checked)} />
            Superseded
          </label>
          <button onClick={() => graphRef.current?.zoomToFit(400, 40)} className="text-xs text-ink-muted hover:text-ink">
            Fit
          </button>
        </div>
      </div>

      <div ref={containerRef} className="flex-1 bg-card border border-warm-border rounded-xl overflow-hidden relative">
        {isLoading ? (
          <div className="flex items-center justify-center h-full text-ink-muted text-sm">Loading graph...</div>
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

        {(selectedMemory || loadingMemory) && (
          <div className="absolute top-3 right-3 w-80 max-w-[calc(100%-1.5rem)] bg-card border border-warm-border-strong rounded-xl shadow-2xl z-10">
            <div className="flex items-center justify-between px-4 py-3 border-b border-warm-border">
              <span className="text-sm font-medium text-ink">Memory Detail</span>
              <div className="flex items-center gap-1">
                {selectedMemory && selectedMemory.status === 'active' && !isEditing && (
                  <>
                    <button onClick={() => { setEditContent(selectedMemory.content); setIsEditing(true) }} className="p-1 text-ink-muted hover:text-accent transition-colors" title="Edit">
                      <Pencil size={14} />
                    </button>
                    <button onClick={() => setShowDeleteConfirm(true)} className="p-1 text-ink-muted hover:text-[#C25B4E] transition-colors" title="Delete">
                      <Trash2 size={14} />
                    </button>
                  </>
                )}
                <button onClick={() => { setSelectedMemory(null); setIsEditing(false); setShowDeleteConfirm(false) }} className="p-1 text-ink-muted hover:text-ink transition-colors">
                  <X size={14} />
                </button>
              </div>
            </div>
            {loadingMemory ? (
              <div className="px-4 py-6 text-center text-ink-muted text-sm">Loading...</div>
            ) : selectedMemory && (
              <div className="px-4 py-3 space-y-3 max-h-[60vh] overflow-y-auto">
                {isEditing ? (
                  <div className="space-y-2">
                    <textarea value={editContent} onChange={(e) => setEditContent(e.target.value)}
                      className="w-full bg-card border border-warm-border-strong rounded-lg p-2 text-sm text-ink resize-y min-h-[80px] focus:outline-none focus:border-accent" rows={4} />
                    <div className="flex gap-2">
                      <button onClick={() => updateMutation.mutate({ id: selectedMemory.id, content: editContent })}
                        disabled={updateMutation.isPending || editContent.trim() === ''}
                        className="flex items-center gap-1 px-3 py-1 bg-accent hover:bg-accent-hover disabled:opacity-50 text-white text-xs rounded-lg">
                        <Check size={12} /> Save
                      </button>
                      <button onClick={() => setIsEditing(false)} className="px-3 py-1 bg-paper-deep hover:bg-paper-deep text-ink-light text-xs rounded-lg">Cancel</button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-ink leading-relaxed">{selectedMemory.content}</p>
                )}

                {showDeleteConfirm && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3 space-y-2">
                    <p className="text-xs text-red-600">Delete this memory?</p>
                    <div className="flex gap-2">
                      <button onClick={() => deleteMutation.mutate(selectedMemory.id)} disabled={deleteMutation.isPending}
                        className="px-3 py-1 bg-[#C25B4E] hover:bg-[#B5503F] disabled:opacity-50 text-white text-xs rounded-lg">
                        {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
                      </button>
                      <button onClick={() => setShowDeleteConfirm(false)} className="px-3 py-1 bg-paper-deep hover:bg-paper-deep text-ink-light text-xs rounded-lg">Cancel</button>
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-1.5">
                  <span className={`text-[10px] px-2 py-0.5 rounded-full ${STATUS_BADGE[selectedMemory.status] || ''}`}>{selectedMemory.status}</span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full ${TYPE_COLOR[selectedMemory.memory_type] || TYPE_COLOR.raw}`}>{selectedMemory.memory_type}</span>
                  {selectedMemory.tags.length > 0 && selectedMemory.tags.map(tag => (
                    <span key={tag} className="text-[10px] px-2 py-0.5 rounded-full bg-paper-deep/50 text-ink-muted inline-flex items-center gap-0.5">
                      <Tag size={8} />{tag}
                    </span>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                  <div><span className="text-ink-faint">Created</span><p className="text-ink-muted">{new Date(selectedMemory.created_at).toLocaleDateString()}</p></div>
                  <div><span className="text-ink-faint">Updated</span><p className="text-ink-muted">{new Date(selectedMemory.updated_at).toLocaleDateString()}</p></div>
                  <div><span className="text-ink-faint">Accessed</span><p className="text-ink-muted">{selectedMemory.access_count}x</p></div>
                  <div><span className="text-ink-faint">Namespace</span><p className="text-ink-muted truncate">{selectedMemory.namespace}</p></div>
                </div>

                {selectedMemory.metadata && Object.keys(selectedMemory.metadata).length > 0 && (
                  <div>
                    <span className="text-[10px] text-ink-faint uppercase tracking-wider">Metadata</span>
                    <pre className="mt-1 text-[11px] text-ink-muted bg-card rounded-lg p-2 overflow-x-auto">
                      {JSON.stringify(selectedMemory.metadata, null, 2)}
                    </pre>
                  </div>
                )}

                <p className="text-[10px] text-ink-faint font-mono truncate" title={selectedMemory.id}>{selectedMemory.id}</p>
              </div>
            )}
          </div>
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
