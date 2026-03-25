import type { NamespaceStats } from '../../api/types'
import { Database, Activity, Eye, Clock } from 'lucide-react'

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return 'N/A'
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export function StatCards({ stats }: { stats: NamespaceStats | undefined }) {
  const cards = [
    { label: 'Total Memories', value: stats?.total ?? '-', icon: Database, color: 'text-accent' },
    { label: 'Active', value: stats?.active ?? '-', icon: Activity, color: 'text-[#5B8C5A]' },
    { label: 'Avg Access', value: stats ? stats.avg_access_count.toFixed(1) : '-', icon: Eye, color: 'text-[#8B7DB8]' },
    { label: 'Newest', value: stats ? timeAgo(stats.newest_memory) : '-', icon: Clock, color: 'text-amber-600' },
  ]

  return (
    <div className="grid grid-cols-4 gap-4">
      {cards.map(({ label, value, icon: Icon, color }) => (
        <div key={label} className="bg-card border border-warm-border rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <Icon size={16} className={color} />
            <span className="text-xs text-ink-muted">{label}</span>
          </div>
          <p className="text-2xl font-semibold text-ink">{value}</p>
        </div>
      ))}
    </div>
  )
}
