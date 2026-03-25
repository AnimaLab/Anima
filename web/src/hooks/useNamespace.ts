import { createContext, useContext } from 'react'

interface NamespaceContextType {
  namespace: string
  setNamespace: (ns: string) => void
  /** Bump to force NamespacePicker to refetch the namespace list */
  nsVersion: number
  refreshNamespaces: () => void
}

export const NamespaceContext = createContext<NamespaceContextType>({
  namespace: 'default',
  setNamespace: () => {},
  nsVersion: 0,
  refreshNamespaces: () => {},
})

export function useNamespace() {
  return useContext(NamespaceContext)
}
