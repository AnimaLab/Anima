import { useEffect, useState } from 'react'
import { useNamespace } from '../hooks/useNamespace'
import { api } from '../api/client'
import type { NamespaceInfo } from '../api/types'

export function NamespacePicker() {
  const { namespace, setNamespace, nsVersion } = useNamespace()
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])

  useEffect(() => {
    api.listNamespaces().then(setNamespaces).catch(() => {})
  }, [nsVersion])

  return (
    <div>
      <label className="text-xs text-ink-muted block mb-1">Namespace</label>
      <select
        value={namespace}
        onChange={(e) => setNamespace(e.target.value)}
        className="w-full bg-paper-deep border border-warm-border-strong rounded-md px-2 py-1.5 text-sm text-ink focus:outline-none focus:ring-1 focus:ring-accent"
      >
        {namespaces.length === 0 && (
          <option value={namespace}>{namespace}</option>
        )}
        {namespaces.map((ns) => (
          <option key={ns.namespace} value={ns.namespace}>
            {ns.namespace} ({ns.active_count})
          </option>
        ))}
      </select>
    </div>
  )
}
