import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/client'
import { getNamespace } from '../../api/client'

export function AccessPatterns() {
  const { data: most } = useQuery({
    queryKey: ['top-accessed', 'most', getNamespace()],
    queryFn: () => api.topAccessed('most', 10),
  })
  const { data: least } = useQuery({
    queryKey: ['top-accessed', 'least', getNamespace()],
    queryFn: () => api.topAccessed('least', 10),
  })

  return (
    <div className="bg-card border border-warm-border rounded-xl p-4">
      <h3 className="text-sm font-medium text-ink-muted mb-3">Access Patterns</h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-ink-muted mb-2">Most Accessed</p>
          <div className="space-y-1.5">
            {(most || []).slice(0, 5).map(m => (
              <div key={m.id} className="flex items-center justify-between text-xs">
                <span className="text-ink-light truncate flex-1 mr-2">{m.content.slice(0, 50)}</span>
                <span className="text-accent font-mono shrink-0">{m.access_count}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <p className="text-xs text-ink-muted mb-2">Least Accessed</p>
          <div className="space-y-1.5">
            {(least || []).slice(0, 5).map(m => (
              <div key={m.id} className="flex items-center justify-between text-xs">
                <span className="text-ink-light truncate flex-1 mr-2">{m.content.slice(0, 50)}</span>
                <span className="text-amber-600 font-mono shrink-0">{m.access_count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
