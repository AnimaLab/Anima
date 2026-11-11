import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { cssVar } from '../lib/theme'
import { useNamespace } from '../hooks/useNamespace'
import type { EmbeddingPoint, GraphEdge } from '../api/types'
import { X, Maximize2, Network, AlertTriangle } from 'lucide-react'

const TIER_COLORS: Record<string, string> = {
  raw:       '#8A94A0',
  reflected: '#8B7DB8',
  deduced:   '#5B9CA6',
  induced:   '#5BA88C',
}

function getColor(type: string): string {
  return TIER_COLORS[type] ?? '#64748b'
}

interface Projected {
  sx: number
  sy: number
  depth: number
  r: number
  point: EmbeddingPoint
}

function project3D(
  x: number, y: number, z: number,
  rx: number, ry: number,
  cx: number, cy: number,
  scale: number, fov: number,
): { sx: number; sy: number; depth: number } | null {
  const cosY = Math.cos(ry), sinY = Math.sin(ry)
  const x1 = x * cosY - z * sinY
  const z1 = x * sinY + z * cosY
  const cosX = Math.cos(rx), sinX = Math.sin(rx)
  const y2 = y * cosX - z1 * sinX
  const z2 = y * sinX + z1 * cosX
  const depth = z2 + fov
  if (depth < 0.01) return null
  return { sx: cx + (x1 / depth) * scale, sy: cy + (y2 / depth) * scale, depth: z2 }
}

function projectPoint(p: EmbeddingPoint, rx: number, ry: number, cx: number, cy: number, scale: number, fov: number): Projected | null {
  const r = project3D(p.x, p.y, p.z, rx, ry, cx, cy, scale, fov)
  if (!r) return null
  return { ...r, r: Math.min(6, Math.max(1.5, 3.5 * (fov / (r.depth + fov)))), point: p }
}

export function Graph3DPage() {
  const { namespace } = useNamespace()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [points, setPoints] = useState<EmbeddingPoint[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<EmbeddingPoint | null>(null)
  const [threshold, setThreshold] = useState(0.75)
  const [showSuperseded, setShowSuperseded] = useState(true)

  const rotRef = useRef({ x: 0.35, y: 0.5 })
  const zoomRef = useRef(1.0)
  const dragRef = useRef(false)
  const lastMouseRef = useRef({ x: 0, y: 0 })
  const projectedRef = useRef<Projected[]>([])
  const hoveredRef = useRef<Projected | null>(null)
  const pointsRef = useRef<EmbeddingPoint[]>([])
  const edgesRef = useRef<GraphEdge[]>([])
  const selectedRef = useRef<EmbeddingPoint | null>(null)
  const thresholdRef = useRef(0.75)
  const showSupersededRef = useRef(true)
  const rafRef = useRef<number>(0)

  pointsRef.current = points
  edgesRef.current = edges
  selectedRef.current = selected
  thresholdRef.current = threshold
  showSupersededRef.current = showSuperseded

  useEffect(() => {
    setLoading(true)
    setError(null)
    setSelected(null)
    setPoints([])
    setEdges([])
    Promise.all([
      api.getEmbeddings(500),
      api.getGraph(0.4, 800),
    ]).then(([embData, graphData]) => {
      const pts = embData.points
      if (pts.length > 0) {
        const maxAbs = (arr: number[]) => Math.max(...arr.map(Math.abs)) || 1
        const xs = pts.map(p => p.x), ys = pts.map(p => p.y), zs = pts.map(p => p.z)
        const sx = maxAbs(xs), sy = maxAbs(ys), sz = maxAbs(zs)
        setPoints(pts.map(p => ({ ...p, x: p.x / sx, y: p.y / sy, z: p.z / sz })))
      } else {
        setPoints(pts)
      }
      setEdges(graphData.edges)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [namespace])

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const pts = pointsRef.current
    if (pts.length === 0) return

    const dpr = window.devicePixelRatio || 1
    const cssW = canvas.clientWidth
    const cssH = canvas.clientHeight
    if (cssW === 0 || cssH === 0) return
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr)
      canvas.height = Math.round(cssH * dpr)
    }

    const ctx = canvas.getContext('2d')!
    ctx.save()
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, cssW, cssH)
    ctx.fillStyle = cssVar('color-paper')
    ctx.fillRect(0, 0, cssW, cssH)

    const cx = cssW / 2
    const cy = cssH / 2
    const scale = Math.min(cssW, cssH) * 0.38 * zoomRef.current
    const fov = 3.5
    const { x: rx, y: ry } = rotRef.current
    const hov = hoveredRef.current
    const selId = selectedRef.current?.id ?? null
    const thresh = thresholdRef.current
    const showSup = showSupersededRef.current

    // Project all points
    const projected: Projected[] = []
    const projMap = new Map<string, { sx: number; sy: number; depth: number }>()
    for (const p of pts) {
      const pp = projectPoint(p, rx, ry, cx, cy, scale, fov)
      if (pp) {
        projected.push(pp)
        projMap.set(p.id, { sx: pp.sx, sy: pp.sy, depth: pp.depth })
      }
    }
    projected.sort((a, b) => a.depth - b.depth)
    projectedRef.current = projected

    // Draw edges (behind nodes)
    ctx.globalAlpha = 1
    for (const edge of edgesRef.current) {
      if (edge.similarity < thresh) continue
      if (!showSup && edge.edge_type === 'superseded') continue
      const a = projMap.get(edge.source)
      const b = projMap.get(edge.target)
      if (!a || !b) continue
      const t = (edge.similarity - thresh) / Math.max(1 - thresh, 0.001)
      const alpha = 0.2 + t * 0.55
      if (edge.edge_type === 'superseded') {
        ctx.strokeStyle = `rgba(196,154,60,${alpha.toFixed(3)})`
        ctx.lineWidth = 1.5
      } else {
        ctx.strokeStyle = `rgba(139,125,184,${alpha.toFixed(3)})`
        ctx.lineWidth = 1
      }
      ctx.beginPath()
      ctx.moveTo(a.sx, a.sy)
      ctx.lineTo(b.sx, b.sy)
      ctx.stroke()
    }

    // Draw nodes (front to back already sorted back-to-front by depth)
    for (const pp of projected) {
      const isSel = pp.point.id === selId
      const isHov = hov?.point.id === pp.point.id
      const r = isSel ? pp.r * 2.2 : isHov ? pp.r * 1.7 : pp.r
      ctx.beginPath()
      ctx.arc(pp.sx, pp.sy, r, 0, Math.PI * 2)
      ctx.globalAlpha = isSel || isHov ? 1.0 : 0.75
      ctx.fillStyle = getColor(pp.point.memory_type)
      ctx.fill()
      if (isSel || isHov) {
        ctx.strokeStyle = cssVar(isSel ? 'color-ink' : 'color-ink-muted')
        ctx.lineWidth = isSel ? 1.5 : 1
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1.0

    // Hover tooltip
    if (hov) {
      const { sx, sy } = hov
      const text = hov.point.content.length > 72 ? hov.point.content.slice(0, 72) + '…' : hov.point.content
      ctx.font = '12px "Inter Variable", system-ui, sans-serif'
      const tw = ctx.measureText(text).width
      const pad = 6, th = 16
      let tx = sx + 10
      let ty = sy - 24
      if (tx + tw + pad * 2 > cssW) tx = sx - tw - pad * 2 - 10
      if (ty < 4) ty = sy + 10
      ctx.fillStyle = cssVar('color-card')
      ctx.strokeStyle = cssVar('color-warm-border-strong')
      ctx.lineWidth = 1
      ctx.beginPath()
      const rx2 = 4
      ctx.roundRect(tx, ty, tw + pad * 2, th + pad, rx2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = cssVar('color-ink')
      ctx.fillText(text, tx + pad, ty + pad + th - 4)
    }

    ctx.restore()
  }, [])

  useEffect(() => {
    let running = true
    function loop() { if (!running) return; render(); rafRef.current = requestAnimationFrame(loop) }
    rafRef.current = requestAnimationFrame(loop)
    return () => { running = false; cancelAnimationFrame(rafRef.current) }
  }, [render])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = true
    lastMouseRef.current = { x: e.clientX, y: e.clientY }
  }, [])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragRef.current) {
      const dx = e.clientX - lastMouseRef.current.x
      const dy = e.clientY - lastMouseRef.current.y
      rotRef.current.y += dx * 0.005
      rotRef.current.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotRef.current.x + dy * 0.005))
      lastMouseRef.current = { x: e.clientX, y: e.clientY }
      return
    }
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    let closest: Projected | null = null
    let closestDist = 14
    for (const pp of projectedRef.current) {
      const d = Math.hypot(pp.sx - mx, pp.sy - my)
      if (d < closestDist) { closestDist = d; closest = pp }
    }
    hoveredRef.current = closest
  }, [])

  const onMouseUp = useCallback(() => { dragRef.current = false }, [])

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    zoomRef.current = Math.max(0.2, Math.min(20, zoomRef.current * (e.deltaY < 0 ? 1.1 : 0.91)))
  }, [])

  const onClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    let closest: EmbeddingPoint | null = null
    let closestDist = 14
    for (const pp of projectedRef.current) {
      const d = Math.hypot(pp.sx - mx, pp.sy - my)
      if (d < closestDist) { closestDist = d; closest = pp.point }
    }
    setSelected(prev => prev?.id === closest?.id ? null : closest)
  }, [])

  const visibleEdges = edges.filter(e =>
    e.similarity >= threshold && (showSuperseded || e.edge_type !== 'superseded')
  )
  const simEdgeCount = visibleEdges.filter(e => e.edge_type === 'similarity').length
  const supEdgeCount = visibleEdges.filter(e => e.edge_type === 'superseded').length
  const tierTypes = ['reflected', 'deduced', 'induced', 'raw'] as const

  const resetView = () => {
    rotRef.current = { x: 0.35, y: 0.5 }
    zoomRef.current = 1.0
  }
  const isEmpty = !loading && !error && points.length === 0

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Header */}
      <div className="shrink-0">
        <div className="flex flex-wrap items-center gap-2.5">
          <h1 className="page-title">Embeddings</h1>
          {!loading && !error && (
            <>
              <span className="badge badge-neutral tabular-nums">{points.length} nodes</span>
              <span className="badge badge-neutral tabular-nums">{visibleEdges.length} edges</span>
            </>
          )}
        </div>
        <p className="page-description">
          Memories placed by meaning in vector space. Drag to orbit, scroll to zoom, click a point to inspect it.
        </p>
      </div>

      <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-4">
        {/* Canvas */}
        <section className="card flex-1 min-w-0 min-h-[360px] flex flex-col overflow-hidden" aria-label="Embedding space">
          {/* Toolbar: everything that changes what the canvas shows */}
          <div className="border-b border-warm-border px-4 py-2 flex flex-wrap items-center gap-x-5 gap-y-2 shrink-0">
            <label className="flex items-center gap-2 text-[13px] text-ink-muted min-h-8">
              <span>Edge threshold</span>
              <input
                type="range" min="0.4" max="0.95" step="0.05"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                className="w-28"
                title="Minimum similarity for an edge to be drawn"
              />
              <span className="font-mono text-xs text-ink tabular-nums w-8">{threshold.toFixed(2)}</span>
            </label>
            <label className="flex items-center gap-2 text-[13px] text-ink-muted min-h-8">
              <input
                type="checkbox"
                checked={showSuperseded}
                onChange={e => setShowSuperseded(e.target.checked)}
              />
              Show superseded
            </label>
            <button onClick={resetView} className="btn btn-secondary ml-auto">
              <Maximize2 size={14} /> Reset view
            </button>
          </div>

          <div className="flex-1 min-h-0 relative overflow-hidden bg-paper">
            {loading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-ink-muted text-sm">
                <div className="w-6 h-6 border-2 border-warm-border-strong border-t-accent rounded-full animate-spin" />
                Loading embeddings…
              </div>
            )}

            {error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6" role="alert">
                <div className="w-11 h-11 rounded-full bg-danger-light text-danger flex items-center justify-center mb-3">
                  <AlertTriangle size={20} />
                </div>
                <p className="text-sm font-medium text-ink">Could not load embeddings</p>
                <p className="text-[13px] text-danger mt-1 break-words max-w-full">{error}</p>
              </div>
            )}

            {!loading && !error && (
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing"
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={onMouseUp}
                onClick={onClick}
                onWheel={onWheel}
              />
            )}

            {isEmpty && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 pointer-events-none">
                <div className="w-11 h-11 rounded-full bg-paper-deep text-ink-muted flex items-center justify-center mb-3">
                  <Network size={20} />
                </div>
                <p className="text-sm font-medium text-ink">Nothing to plot yet</p>
                <p className="text-[13px] text-ink-muted mt-1 max-w-xs">
                  Add a few memories and they will appear here, close together when their meaning is close.
                </p>
              </div>
            )}

            {/* Legend */}
            {!loading && !error && !isEmpty && (
              <div className="absolute bottom-3 left-3 card px-3 py-2 text-xs max-w-[calc(100%-1.5rem)] pointer-events-none">
                <ul className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  {tierTypes.map(type => (
                    <li key={type} className="flex items-center gap-1.5 text-ink-muted capitalize">
                      <span className="w-2 h-2 rounded-full" style={{ background: getColor(type) }} aria-hidden="true" />
                      {type}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>

        {/* Details panel */}
        <aside className="card lg:w-80 shrink-0 flex flex-col lg:min-h-0 overflow-hidden" aria-label="Details">
          <div className="flex-1 lg:min-h-0 lg:overflow-y-auto p-4 space-y-5">
            {/* Selection */}
            <div>
              <div className="flex items-center justify-between gap-2 mb-2 min-h-8">
                <p className="section-label">Selected</p>
                {selected && (
                  <button onClick={() => setSelected(null)} className="btn btn-ghost btn-icon -mr-2" aria-label="Clear selection" title="Clear selection">
                    <X size={16} />
                  </button>
                )}
              </div>
              {selected ? (
                <>
                  <span className="badge badge-neutral">
                    <span className="w-2 h-2 rounded-full" style={{ background: getColor(selected.memory_type) }} aria-hidden="true" />
                    {selected.memory_type}
                  </span>
                  <p className="mt-2 text-sm text-ink leading-relaxed break-words">{selected.content}</p>
                </>
              ) : (
                <p className="text-[13px] text-ink-muted">Click a point to read the memory behind it.</p>
              )}
            </div>

            {/* Node types */}
            <div className="border-t border-warm-border pt-4">
              <p className="section-label mb-2">Nodes</p>
              <ul className="space-y-1.5">
                {tierTypes.map(type => {
                  const count = points.filter(p => p.memory_type === type).length
                  if (count === 0) return null
                  return (
                    <li key={type} className="flex items-center gap-2 text-[13px]">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(type) }} aria-hidden="true" />
                      <span className="text-ink-light flex-1 capitalize">{type}</span>
                      <span className="text-ink-muted tabular-nums">{count}</span>
                    </li>
                  )
                })}
                {points.length === 0 && <li className="text-[13px] text-ink-muted">None</li>}
              </ul>
            </div>

            {/* Edge types */}
            <div className="border-t border-warm-border pt-4">
              <p className="section-label mb-2">Edges</p>
              <ul className="space-y-1.5">
                <li className="flex items-center gap-2 text-[13px]">
                  <span className="w-3 h-0.5 shrink-0 rounded" style={{ background: 'rgba(139,125,184,0.8)' }} aria-hidden="true" />
                  <span className="text-ink-light flex-1">Similarity</span>
                  <span className="text-ink-muted tabular-nums">{simEdgeCount}</span>
                </li>
                {supEdgeCount > 0 && (
                  <li className="flex items-center gap-2 text-[13px]">
                    <span className="w-3 h-0.5 shrink-0 rounded" style={{ background: 'rgba(196,154,60,0.8)' }} aria-hidden="true" />
                    <span className="text-ink-light flex-1">Superseded</span>
                    <span className="text-ink-muted tabular-nums">{supEdgeCount}</span>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
