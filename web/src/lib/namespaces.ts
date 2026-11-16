import type { NamespaceInfo } from '../api/types'

/** Whether `ns` is `root` or one of its descendants. */
export function isWithin(ns: string, root: string): boolean {
  return ns === root || ns.startsWith(`${root}/`)
}

/** Namespaces sorted so each one comes right after its parent. */
export function sortedTree(namespaces: NamespaceInfo[]): NamespaceInfo[] {
  return [...namespaces].sort((a, b) => a.namespace.split('/').join('\u0000').localeCompare(b.namespace.split('/').join('\u0000')))
}

/** A namespace's label in a flat list: its last segment, indented by depth. */
export function treeLabel(ns: string): string {
  const parts = ns.split('/')
  if (parts.length === 1) return ns
  return `${'  '.repeat(parts.length - 1)}└ ${parts[parts.length - 1]}`
}

/** The namespaces and memories a rename or delete of `root` takes with it. */
export function subtree(namespaces: NamespaceInfo[], root: string) {
  const inside = namespaces.filter(n => isWithin(n.namespace, root))
  return {
    descendants: inside.filter(n => n.namespace !== root).length,
    memories: inside.reduce((sum, n) => sum + n.total_count, 0),
  }
}

/** `ns` after `from` is renamed to `to`. */
export function renamed(ns: string, from: string, to: string): string {
  return isWithin(ns, from) ? to + ns.slice(from.length) : ns
}
