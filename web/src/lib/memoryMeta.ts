/** Shared presentation helpers for memory category, source and confidence. */

export const CATEGORIES = [
  'identity',
  'preference',
  'routine',
  'environment',
  'task',
  'inferred',
  'general',
] as const

const categoryColors: Record<string, string> = {
  identity: 'bg-indigo-500/15 text-indigo-700',
  preference: 'bg-orange-500/15 text-orange-700',
  routine: 'bg-teal-500/15 text-teal-700',
  environment: 'bg-sky-500/15 text-sky-700',
  task: 'bg-amber-500/15 text-amber-700',
  inferred: 'bg-violet-500/15 text-violet-600',
  general: 'bg-stone-500/15 text-stone-600',
}

/** Badge classes for a category; custom categories get a neutral style. */
export function categoryColor(category: string | undefined): string {
  return categoryColors[category ?? 'general'] ?? 'bg-paper-deep text-ink-muted'
}

const sourceLabels: Record<string, string> = {
  user_stated: 'stated by user',
  promoted: 'promoted',
  agent_observed: 'observed by agent',
  reflected: 'reflected',
  deduced: 'deduced',
  inferred: 'inferred',
}

export function sourceLabel(source: string | undefined): string {
  if (!source) return ''
  return sourceLabels[source] ?? source.replace(/_/g, ' ')
}

/**
 * Provenance worth showing: anything the user did not state directly, or
 * stated with less than full confidence. Returns null for the default case
 * so cards stay quiet.
 */
export function provenanceText(source: string | undefined, confidence: number | undefined): string | null {
  const nonDefaultSource = !!source && source !== 'user_stated'
  const lowConfidence = confidence !== undefined && confidence < 1
  if (!nonDefaultSource && !lowConfidence) return null
  const parts: string[] = []
  if (nonDefaultSource) parts.push(sourceLabel(source))
  if (confidence !== undefined) parts.push(`${Math.round(confidence * 100)}% confidence`)
  return parts.join(' · ')
}
