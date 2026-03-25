import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import type { NamespaceStats } from '../../api/types'

const COLORS = { active: '#5B8C5A', superseded: '#C49A3C', deleted: '#C25B4E' }

export function LifecyclePie({ stats }: { stats: NamespaceStats | undefined }) {
  if (!stats || stats.total === 0) return <div className="text-ink-muted text-sm">No data</div>

  const data = [
    { name: 'Active', value: stats.active },
    { name: 'Superseded', value: stats.superseded },
    { name: 'Deleted', value: stats.deleted },
  ].filter(d => d.value > 0)

  return (
    <div className="bg-card border border-warm-border rounded-xl p-4">
      <h3 className="text-sm font-medium text-ink-muted mb-3">Memory Lifecycle</h3>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" nameKey="name" stroke="none">
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #E0D8CB', borderRadius: 8, color: '#2D2A26' }} />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-4 mt-2">
        {data.map(d => (
          <div key={d.name} className="flex items-center gap-1.5 text-xs text-ink-muted">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS[d.name.toLowerCase() as keyof typeof COLORS] }} />
            {d.name} ({d.value})
          </div>
        ))}
      </div>
    </div>
  )
}
