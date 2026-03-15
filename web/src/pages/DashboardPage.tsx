import { useQuery } from '@tanstack/react-query'
import { api, getNamespace } from '../api/client'
import { StatCards } from '../components/dashboard/StatCards'
import { LifecyclePie } from '../components/dashboard/LifecyclePie'
import { AccessPatterns } from '../components/dashboard/AccessPatterns'
import { NamespaceBreakdown } from '../components/dashboard/NamespaceBreakdown'

export function DashboardPage() {
  const { data: stats } = useQuery({
    queryKey: ['stats', getNamespace()],
    queryFn: () => api.getStats(),
    refetchInterval: 30000,
  })
  const { data: namespaces } = useQuery({
    queryKey: ['namespaces'],
    queryFn: () => api.listNamespaces(),
  })

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-white">Dashboard</h2>
      <StatCards stats={stats} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <LifecyclePie stats={stats} />
        <AccessPatterns />
      </div>
      <NamespaceBreakdown namespaces={namespaces} />
    </div>
  )
}
