import { useEffect, useState } from 'react'
import { useNamespace } from '../hooks/useNamespace'
import { api } from '../api/client'
import type { NamespaceInfo } from '../api/types'

export function NamespacePicker() {
  const { namespace, setNamespace } = useNamespace()
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])

  useEffect(() => {
    api.listNamespaces().then(setNamespaces).catch(() => {})
  }, [])

  return (
    <div>
      <label className="text-xs text-gray-500 block mb-1">Namespace</label>
      <select
        value={namespace}
        onChange={(e) => setNamespace(e.target.value)}
        className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
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
