import { useState, useEffect, useCallback } from 'react'
import { BookOpen, Plus, Save, Trash2, RotateCcw } from 'lucide-react'
import { api } from '../api/client'
import type { Skill, SkillSummary } from '../api/types'
import { useNamespace } from '../hooks/useNamespace'
import { useToast } from '../hooks/useToast'

const NAME_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/

/** What the editor holds. `original` is null while creating a new skill. */
interface Draft {
  original: Skill | null
  name: string
  description: string
  content: string
}

const emptyDraft: Draft = { original: null, name: '', description: '', content: '' }

function draftOf(skill: Skill): Draft {
  return { original: skill, name: skill.name, description: skill.description, content: skill.content }
}

export function SkillsPage() {
  const { namespace } = useNamespace()
  const toast = useToast()
  const [skills, setSkills] = useState<SkillSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setSkills((await api.listSkills()).skills)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load skills')
    } finally {
      setLoading(false)
    }
  }, [])

  // Skills belong to a namespace: reload and drop the editor when it changes.
  useEffect(() => {
    setDraft(null)
    load()
  }, [namespace, load])

  const open = async (name: string) => {
    try {
      setDraft(draftOf(await api.getSkill(name)))
    } catch (e) {
      toast.show(e instanceof Error ? e.message : 'Could not open the skill', true)
    }
  }

  const isNew = draft !== null && draft.original === null
  const builtin = draft?.original?.builtin === true
  const nameValid = draft !== null && NAME_PATTERN.test(draft.name) && draft.name.length <= 64
  const changed = draft !== null && (
    draft.original === null ||
    draft.description !== draft.original.description ||
    draft.content !== draft.original.content
  )
  const canSave = draft !== null && nameValid && changed &&
    draft.description.trim() !== '' && draft.content.trim() !== '' && !saving

  const save = async () => {
    if (!draft || !canSave) return
    setSaving(true)
    try {
      const saved = await api.saveSkill(draft.name, draft.description, draft.content)
      setDraft(draftOf(saved))
      toast.show(`Saved ${saved.name} (v${saved.version})`)
      await load()
    } catch (e) {
      toast.show(e instanceof Error ? e.message : 'Could not save the skill', true)
    } finally {
      setSaving(false)
    }
  }

  const remove = async () => {
    if (!draft?.original || builtin) return
    const restores = draft.original.name === 'anima'
    const question = restores
      ? 'Remove your version of the anima skill and go back to the built-in one?'
      : `Delete the skill "${draft.original.name}"? This cannot be undone.`
    if (!window.confirm(question)) return
    try {
      await api.deleteSkill(draft.original.name)
      toast.show(restores ? 'Built-in anima skill restored' : `Deleted ${draft.original.name}`)
      setDraft(null)
      await load()
    } catch (e) {
      toast.show(e instanceof Error ? e.message : 'Could not delete the skill', true)
    }
  }

  return (
    <div className="page">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="page-title">Skills</h1>
            {skills.length > 0 && <span className="badge badge-neutral tabular-nums">{skills.length}</span>}
          </div>
          <p className="page-description">
            Reusable instructions your assistants load on demand. Only the name and description
            are sent up front, so keep the description precise about when to use the skill.
          </p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setDraft({ ...emptyDraft })}>
          <Plus size={16} aria-hidden="true" />
          New skill
        </button>
      </div>

      {error && (
        <div className="card mt-6 flex flex-wrap items-center justify-between gap-3 p-4" role="alert">
          <p className="text-sm text-danger">{error}</p>
          <button type="button" className="btn btn-secondary" onClick={load}>
            <RotateCcw size={14} aria-hidden="true" />
            Retry
          </button>
        </div>
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <section aria-label="Stored skills" className="card self-start overflow-hidden">
          {loading && skills.length === 0 ? (
            <p className="p-5 text-sm text-ink-muted">Loading skills…</p>
          ) : (
            <ul className="divide-y divide-warm-border">
              {skills.map(skill => {
                const active = draft?.original?.name === skill.name
                return (
                  <li key={skill.name}>
                    <button
                      type="button"
                      onClick={() => open(skill.name)}
                      aria-current={active ? 'true' : undefined}
                      className={`block w-full px-5 py-3.5 text-left transition-colors hover:bg-paper-deep focus-visible:outline-2 focus-visible:-outline-offset-2 ${active ? 'bg-paper-deep' : ''}`}
                    >
                      <span className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-sm font-medium text-ink">{skill.name}</span>
                        {skill.builtin && <span className="badge badge-info">Built in</span>}
                        <span className="ml-auto text-xs tabular-nums text-ink-faint">v{skill.version}</span>
                      </span>
                      <span className="mt-1 line-clamp-2 text-xs leading-relaxed text-ink-muted">
                        {skill.description}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </section>

        {draft === null ? (
          <section className="card flex min-h-64 flex-col items-center justify-center gap-2 p-8 text-center">
            <BookOpen size={28} className="text-ink-faint" aria-hidden="true" />
            <p className="text-sm font-medium text-ink">Select a skill to read or edit it</p>
            <p className="max-w-sm text-xs leading-relaxed text-ink-muted">
              Assistants can also save skills themselves with the skill_save tool, and you can load
              SKILL.md folders with <span className="font-mono">anima-server skills add</span>.
            </p>
          </section>
        ) : (
          <form
            className="card flex flex-col gap-5 p-5"
            onSubmit={e => { e.preventDefault(); save() }}
            aria-label={isNew ? 'New skill' : `Skill ${draft.name}`}
          >
            <div>
              <label htmlFor="skill-name" className="mb-1.5 block text-sm font-medium text-ink">Name</label>
              <input
                id="skill-name"
                className="field font-mono"
                value={draft.name}
                disabled={!isNew}
                onChange={e => setDraft({ ...draft, name: e.target.value })}
                placeholder="release-checklist"
                aria-describedby="skill-name-help"
                aria-invalid={draft.name !== '' && !nameValid}
                autoFocus={isNew}
              />
              <p id="skill-name-help" className={`mt-1.5 text-xs ${draft.name !== '' && !nameValid ? 'text-danger' : 'text-ink-muted'}`}>
                {isNew
                  ? 'Lowercase letters, digits and hyphens. It cannot be changed later.'
                  : 'The name identifies the skill and cannot be changed.'}
              </p>
            </div>

            <div>
              <label htmlFor="skill-description" className="mb-1.5 block text-sm font-medium text-ink">Description</label>
              <textarea
                id="skill-description"
                className="field"
                rows={2}
                value={draft.description}
                onChange={e => setDraft({ ...draft, description: e.target.value })}
                placeholder="What the skill does and when to use it, in one or two sentences."
                aria-describedby="skill-description-help"
              />
              <p id="skill-description-help" className="mt-1.5 text-xs text-ink-muted">
                This is all an assistant sees before deciding to load the skill.
              </p>
            </div>

            <div>
              <label htmlFor="skill-content" className="mb-1.5 block text-sm font-medium text-ink">Instructions</label>
              <textarea
                id="skill-content"
                className="field font-mono text-xs leading-relaxed"
                rows={18}
                value={draft.content}
                onChange={e => setDraft({ ...draft, content: e.target.value })}
                placeholder={'# Title\n\n1. First step\n2. Second step'}
                spellCheck={false}
              />
              <p className="mt-1.5 text-xs tabular-nums text-ink-muted">
                {draft.content.length.toLocaleString()} characters, loaded only when the skill is used.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 border-t border-warm-border pt-4">
              <button type="submit" className="btn btn-primary" disabled={!canSave}>
                <Save size={16} aria-hidden="true" />
                {saving ? 'Saving…' : builtin ? 'Save my version' : isNew ? 'Create skill' : 'Save changes'}
              </button>
              <button type="button" className="btn btn-ghost" onClick={() => setDraft(null)}>
                Close
              </button>
              {draft.original && !builtin && (
                <button type="button" className="btn btn-ghost ml-auto text-danger" onClick={remove}>
                  <Trash2 size={16} aria-hidden="true" />
                  {draft.original.name === 'anima' ? 'Restore built-in' : 'Delete'}
                </button>
              )}
              {builtin && (
                <span className="ml-auto text-xs text-ink-muted">Built in. Saving stores your own version.</span>
              )}
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
