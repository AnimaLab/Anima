import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { useNamespace } from '../hooks/useNamespace'
import type { EmbeddingPoint, GraphEdge } from '../api/types'

const TIER_COLORS: Record<string, string> = {
  raw:       '#9ca3af',
  reflected: '#a78bfa',
  deduced:   '#38bdf8',
  induced:   '#34d399',
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
  const [threshold, setThreshold] = useState(0.4)
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
  const thresholdRef = useRef(0.5)
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
    ctx.fillStyle = '#030712'
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
        ctx.strokeStyle = `rgba(251,191,36,${alpha.toFixed(3)})`
        ctx.lineWidth = 1.5
      } else {
        ctx.strokeStyle = `rgba(99,102,241,${alpha.toFixed(3)})`
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
        ctx.strokeStyle = isSel ? '#fff' : 'rgba(255,255,255,0.6)'
        ctx.lineWidth = isSel ? 1.5 : 1
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1.0

    // Hover tooltip
    if (hov) {
      const { sx, sy } = hov
      const text = hov.point.content.length > 72 ? hov.point.content.slice(0, 72) + '…' : hov.point.content
      ctx.font = '11px system-ui, sans-serif'
      const tw = ctx.measureText(text).width
      const pad = 6, th = 16
      let tx = sx + 10
      let ty = sy - 24
      if (tx + tw + pad * 2 > cssW) tx = sx - tw - pad * 2 - 10
      if (ty < 4) ty = sy + 10
      ctx.fillStyle = 'rgba(15,23,42,0.92)'
      ctx.strokeStyle = 'rgba(99,102,241,0.6)'
      ctx.lineWidth = 1
      ctx.beginPath()
      const rx2 = 4
      ctx.roundRect(tx, ty, tw + pad * 2, th + pad, rx2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#e2e8f0'
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

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <h2 className="text-lg font-semibold text-white">Graph 3D</h2>
          <p className="text-xs text-gray-500">PCA positions · drag to orbit · scroll to zoom · click to inspect</p>
        </div>
        {!loading && <span className="text-xs text-gray-600">{points.length} nodes · {visibleEdges.length} edges</span>}
      </div>

      {loading && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
          <div className="w-6 h-6 border-2 border-gray-700 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-sm text-gray-500">Loading graph…</span>
        </div>
      )}
      {error && <div className="text-red-400 text-sm bg-red-950/30 rounded-lg px-4 py-3">Error: {error}</div>}

      {!loading && !error && (
        <div style={{ flex: 1, display: 'flex', gap: '12px', minHeight: 0 }}>
          {/* Canvas */}
          <div style={{ flex: 1, position: 'relative', borderRadius: '12px', overflow: 'hidden', background: '#030712', border: '1px solid rgba(31,41,55,0.6)' }}>
            <canvas
              ref={canvasRef}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'grab' }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
              onClick={onClick}
              onWheel={onWheel}
            />
          </div>

          {/* Side panel */}
          <div style={{ width: '180px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
            {/* Threshold */}
            <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60">
              <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider">Edge Threshold</p>
              <input
                type="range" min="0.4" max="0.95" step="0.05"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between mt-1">
                <span className="text-[10px] text-gray-600">0.40</span>
                <span className="text-[10px] font-mono text-gray-300">{threshold.toFixed(2)}</span>
                <span className="text-[10px] text-gray-600">0.95</span>
              </div>
              <label className="flex items-center gap-1.5 mt-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showSuperseded}
                  onChange={e => setShowSuperseded(e.target.checked)}
                  className="rounded"
                />
                <span className="text-xs text-gray-400">Show superseded</span>
              </label>
            </div>

            {/* Node types */}
            <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60">
              <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider">Nodes</p>
              {tierTypes.map(type => {
                const count = points.filter(p => p.memory_type === type).length
                if (count === 0) return null
                return (
                  <div key={type} className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(type) }} />
                    <span className="text-xs text-gray-300 flex-1 capitalize">{type}</span>
                    <span className="text-xs text-gray-600 tabular-nums">{count}</span>
                  </div>
                )
              })}
            </div>

            {/* Edge types */}
            <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60">
              <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider">Edges</p>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-0.5 shrink-0 rounded" style={{ background: 'rgba(99,102,241,0.8)' }} />
                <span className="text-xs text-gray-300 flex-1">Similarity</span>
                <span className="text-xs text-gray-600 tabular-nums">{simEdgeCount}</span>
              </div>
              {supEdgeCount > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-0.5 shrink-0 rounded" style={{ background: 'rgba(251,191,36,0.8)' }} />
                  <span className="text-xs text-gray-300 flex-1">Superseded</span>
                  <span className="text-xs text-gray-600 tabular-nums">{supEdgeCount}</span>
                </div>
              )}
            </div>

            {/* Selected node */}
            {selected && (
              <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60 overflow-auto" style={{ maxHeight: '35%', minHeight: '80px' }}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">Selected</p>
                  <button onClick={() => setSelected(null)} className="text-gray-600 hover:text-gray-400 text-sm leading-none">✕</button>
                </div>
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(selected.memory_type) }} />
                  <span className="text-xs text-gray-400 capitalize">{selected.memory_type}</span>
                </div>
                <p className="text-xs text-gray-200 leading-relaxed">{selected.content}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
