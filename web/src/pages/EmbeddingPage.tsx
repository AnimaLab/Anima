import { useState, useEffect, useRef, useCallback } from 'react'
import { api } from '../api/client'
import { useNamespace } from '../hooks/useNamespace'
import type { EmbeddingPoint } from '../api/types'

// Tier 1 — raw user input
const RAW_COLORS: Record<string, string> = {
  raw: '#9ca3af',
}

// Tier 2/3/4 — processor-generated
const PROCESSED_COLORS: Record<string, string> = {
  reflected: '#a78bfa',
  deduced:   '#38bdf8',
  induced:   '#34d399',
}

const ALL_COLORS = { ...RAW_COLORS, ...PROCESSED_COLORS }

function getColor(type: string): string {
  return ALL_COLORS[type] ?? '#64748b'
}

interface Projected {
  sx: number
  sy: number
  depth: number
  r: number
  point: EmbeddingPoint
}

// Project a raw 3D point (no EmbeddingPoint wrapper)
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

export function EmbeddingPage() {
  const { namespace } = useNamespace()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [points, setPoints] = useState<EmbeddingPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<EmbeddingPoint | null>(null)

  const rotRef = useRef({ x: 0.35, y: 0.5 })
  const zoomRef = useRef(1.0)
  const dragRef = useRef(false)
  const lastMouseRef = useRef({ x: 0, y: 0 })
  const projectedRef = useRef<Projected[]>([])
  const hoveredRef = useRef<Projected | null>(null)
  const pointsRef = useRef<EmbeddingPoint[]>([])
  const selectedRef = useRef<EmbeddingPoint | null>(null)
  const rafRef = useRef<number>(0)

  pointsRef.current = points
  selectedRef.current = selected

  useEffect(() => {
    setLoading(true)
    setError(null)
    setSelected(null)
    api.getEmbeddings(400).then(data => {
      const pts = data.points
      if (pts.length > 0) {
        const maxAbs = (arr: number[]) => Math.max(...arr.map(Math.abs)) || 1
        const xs = pts.map(p => p.x), ys = pts.map(p => p.y), zs = pts.map(p => p.z)
        const sx = maxAbs(xs), sy = maxAbs(ys), sz = maxAbs(zs)
        setPoints(pts.map(p => ({ ...p, x: p.x / sx, y: p.y / sy, z: p.z / sz })))
      } else {
        setPoints(pts)
      }
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

    // --- 3D Axes ---
    const axes = [
      { end: [1.35, 0, 0] as [number,number,number], color: '#ef4444', label: 'PC1' },
      { end: [0, 1.35, 0] as [number,number,number], color: '#22c55e', label: 'PC2' },
      { end: [0, 0, 1.35] as [number,number,number], color: '#3b82f6', label: 'PC3' },
    ]
    const origin = project3D(0, 0, 0, rx, ry, cx, cy, scale, fov)
    if (origin) {
      for (const { end, color, label } of axes) {
        const tip = project3D(...end, rx, ry, cx, cy, scale, fov)
        if (!tip) continue
        ctx.save()
        ctx.strokeStyle = color
        ctx.lineWidth = 1
        ctx.globalAlpha = 0.5
        ctx.beginPath()
        ctx.moveTo(origin.sx, origin.sy)
        ctx.lineTo(tip.sx, tip.sy)
        ctx.stroke()
        // Arrowhead
        const dx = tip.sx - origin.sx, dy = tip.sy - origin.sy
        const len = Math.hypot(dx, dy)
        if (len > 0) {
          const ux = dx / len, uy = dy / len
          ctx.fillStyle = color
          ctx.globalAlpha = 0.6
          ctx.beginPath()
          ctx.moveTo(tip.sx, tip.sy)
          ctx.lineTo(tip.sx - ux * 7 - uy * 3, tip.sy - uy * 7 + ux * 3)
          ctx.lineTo(tip.sx - ux * 7 + uy * 3, tip.sy - uy * 7 - ux * 3)
          ctx.closePath()
          ctx.fill()
        }
        // Label
        ctx.globalAlpha = 0.8
        ctx.fillStyle = color
        ctx.font = 'bold 11px system-ui, sans-serif'
        ctx.fillText(label, tip.sx + 6, tip.sy + 4)
        ctx.restore()
      }
    }

    // --- Points ---
    const projected: Projected[] = []
    for (const p of pts) {
      const pp = projectPoint(p, rx, ry, cx, cy, scale, fov)
      if (pp) projected.push(pp)
    }
    projected.sort((a, b) => a.depth - b.depth)
    projectedRef.current = projected

    for (const pp of projected) {
      const isSel = pp.point.id === selId
      const isHov = hov?.point.id === pp.point.id
      const r = isSel ? pp.r * 2.2 : isHov ? pp.r * 1.7 : pp.r
      ctx.beginPath()
      ctx.arc(pp.sx, pp.sy, r, 0, Math.PI * 2)
      ctx.globalAlpha = isSel || isHov ? 1.0 : 0.72
      ctx.fillStyle = getColor(pp.point.memory_type)
      ctx.fill()
      if (isSel || isHov) {
        ctx.strokeStyle = isSel ? '#fff' : 'rgba(255,255,255,0.6)'
        ctx.lineWidth = isSel ? 1.5 : 1
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1.0

    // --- Hover tooltip ---
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

  const rawPresent = Object.keys(RAW_COLORS).filter(t => points.some(p => p.memory_type === t))
  const processedPresent = Object.keys(PROCESSED_COLORS).filter(t => points.some(p => p.memory_type === t))

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <h2 className="text-lg font-semibold text-white">Vector Space</h2>
          <p className="text-xs text-gray-500">PCA · drag to orbit · scroll to zoom · click to inspect</p>
        </div>
        {!loading && <span className="text-xs text-gray-600">{points.length} memories</span>}
      </div>

      {loading && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
          <div className="w-6 h-6 border-2 border-gray-700 border-t-blue-500 rounded-full animate-spin" />
          <span className="text-sm text-gray-500">Computing PCA projection…</span>
        </div>
      )}
      {error && <div className="text-red-400 text-sm bg-red-950/30 rounded-lg px-4 py-3">Error: {error}</div>}

      {!loading && !error && (
        <div style={{ flex: 1, display: 'flex', gap: '12px', minHeight: 0 }}>
          {/* Canvas */}
          <div style={{ flex: 1, position: 'relative', borderRadius: '12px', overflow: 'hidden', background: '#030712', border: '1px solid rgba(31,41,55,0.6)' }}>
            <canvas
              ref={canvasRef}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: dragRef.current ? 'grabbing' : 'grab' }}
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
            {/* Axes legend */}
            <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60">
              <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider">Axes</p>
              {[['PC1','#ef4444','Most variance'], ['PC2','#22c55e','2nd variance'], ['PC3','#3b82f6','3rd variance']].map(([label, color, desc]) => (
                <div key={label} className="flex items-center gap-2 mb-1.5">
                  <div className="w-3 h-0.5 shrink-0" style={{ background: color }} />
                  <span className="text-xs font-mono text-gray-300">{label}</span>
                  <span className="text-xs text-gray-600 truncate">{desc}</span>
                </div>
              ))}
            </div>

            {/* Types legend */}
            <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60 flex-1 overflow-auto" style={{ minHeight: 0 }}>
              {processedPresent.length > 0 && <>
                <p className="text-xs text-gray-500 mb-1.5 font-medium uppercase tracking-wider">Processed</p>
                {processedPresent.map(type => (
                  <div key={type} className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(type) }} />
                    <span className="text-xs text-gray-300 flex-1 capitalize">{type}</span>
                    <span className="text-xs text-gray-600 tabular-nums">{points.filter(p => p.memory_type === type).length}</span>
                  </div>
                ))}
                {rawPresent.length > 0 && <div className="border-t border-gray-800 my-2" />}
              </>}
              {rawPresent.length > 0 && <>
                <p className="text-xs text-gray-500 mb-1.5 font-medium uppercase tracking-wider">Raw</p>
                {rawPresent.map(type => (
                  <div key={type} className="flex items-center gap-2 mb-1">
                    <div className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(type) }} />
                    <span className="text-xs text-gray-400 flex-1 capitalize">{type}</span>
                    <span className="text-xs text-gray-600 tabular-nums">{points.filter(p => p.memory_type === type).length}</span>
                  </div>
                ))}
              </>}
            </div>

            {/* Selected memory */}
            {selected && (
              <div className="bg-gray-900 rounded-xl p-3 border border-gray-800/60 overflow-auto" style={{ maxHeight: '35%', minHeight: '80px' }}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">Memory</p>
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
