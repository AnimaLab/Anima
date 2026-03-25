import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { NamespaceInfo } from '../../api/types'

export function NamespaceBreakdown({ namespaces }: { namespaces: NamespaceInfo[] | undefined }) {
  if (!namespaces || namespaces.length === 0) return null

  return (
    <div className="bg-card border border-warm-border rounded-xl p-4">
      <h3 className="text-sm font-medium text-ink-muted mb-3">Namespace Breakdown</h3>
      <ResponsiveContainer width="100%" height={Math.max(100, namespaces.length * 36)}>
        <BarChart data={namespaces} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" tick={{ fill: '#9C9488', fontSize: 11 }} />
          <YAxis type="category" dataKey="namespace" tick={{ fill: '#6B6259', fontSize: 11 }} width={80} />
          <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #E0D8CB', borderRadius: 8, color: '#2D2A26' }} />
          <Bar dataKey="active_count" fill="#C47B3B" name="Active" radius={[0, 4, 4, 0]} />
          <Bar dataKey="total_count" fill="#E0D8CB" name="Total" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
