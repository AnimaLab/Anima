import { createContext, useContext } from 'react'

interface NamespaceContextType {
  namespace: string
  setNamespace: (ns: string) => void
}

export const NamespaceContext = createContext<NamespaceContextType>({
  namespace: 'default',
  setNamespace: () => {},
})

export function useNamespace() {
  return useContext(NamespaceContext)
}
