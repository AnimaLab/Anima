import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { NamespaceInfo } from '../../api/types'

export function NamespaceBreakdown({ namespaces }: { namespaces: NamespaceInfo[] | undefined }) {
  if (!namespaces || namespaces.length === 0) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Namespace Breakdown</h3>
      <ResponsiveContainer width="100%" height={Math.max(100, namespaces.length * 36)}>
        <BarChart data={namespaces} layout="vertical" margin={{ left: 80 }}>
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <YAxis type="category" dataKey="namespace" tick={{ fill: '#d1d5db', fontSize: 11 }} width={80} />
          <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
          <Bar dataKey="active_count" fill="#3b82f6" name="Active" radius={[0, 4, 4, 0]} />
          <Bar dataKey="total_count" fill="#374151" name="Total" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
