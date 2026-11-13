import { useEffect, useRef } from 'react'
import { X, BookOpen, Search, Layers, Plug, Sparkles } from 'lucide-react'

const STORAGE_KEY = 'anima-about-seen'

/** Whether this browser has already seen the introduction. */
export function aboutSeen(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    return true
  }
}

function markAboutSeen() {
  try {
    localStorage.setItem(STORAGE_KEY, '1')
  } catch { /* storage unavailable: show it again next time */ }
}

const TIERS: { name: string; tier: number; text: string }[] = [
  { tier: 1, name: 'Raw', text: 'What an assistant or you stored, word for word. Everything else is built on it.' },
  { tier: 2, name: 'Reflection', text: 'Reads new raw memories and extracts standalone facts: who you are, what you prefer, what you decided.' },
  { tier: 3, name: 'Deduction', text: 'Combines several facts into conclusions none of them states alone.' },
  { tier: 4, name: 'Induction', text: 'Looks across many facts for recurring patterns: habits, ways of working, lasting preferences.' },
]

function Block({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <section className="flex gap-3">
      <div className="mt-0.5 shrink-0 text-accent">{icon}</div>
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        <div className="text-[13px] text-ink-muted mt-1 space-y-2">{children}</div>
      </div>
    </section>
  )
}

/** Introduction to Anima: what it is, how memory is processed, search and skills. */
export function AboutAnima({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)

  const close = () => {
    markAboutSeen()
    onClose()
  }

  useEffect(() => {
    if (!open) return
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-anima-title"
        className="bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3 px-6 pt-5 pb-3">
          <div className="min-w-0">
            <h2 id="about-anima-title" className="text-lg font-semibold text-ink">Anima is your memory ledger</h2>
            <p className="text-[13px] text-ink-muted mt-1">
              One record of what your assistants learn about you and your work, kept on your machine.
            </p>
          </div>
          <button ref={closeRef} onClick={close} aria-label="Close" className="btn btn-ghost btn-icon -mr-2">
            <X size={16} />
          </button>
        </div>

        <div className="px-6 pb-6 overflow-y-auto space-y-5">
          <Block icon={<Plug size={18} />} title="Whatever AI or harness you use">
            <p>
              Claude Code, Codex, Cursor, any MCP client or a script over the HTTP API: each connected
              assistant reads and writes the same memory. What one learns, the next one knows, and changing
              tools or models never means starting over.
            </p>
            <p>
              At the start of a session, a connected assistant receives a short briefing (who you are, what
              matters for the current project, which skills exist), then the memories relevant to each message.
            </p>
          </Block>

          <Block icon={<Sparkles size={18} />} title="Memory that works in the background">
            <p>Stored memories are processed by an LLM in four tiers, each built on the one before:</p>
            <ol className="space-y-1.5">
              {TIERS.map(t => (
                <li key={t.tier} className="flex gap-2">
                  <span className="shrink-0 w-5 h-5 rounded-full bg-paper-deep text-ink text-[11px] font-semibold flex items-center justify-center tabular-nums">{t.tier}</span>
                  <span><span className="font-medium text-ink">{t.name}.</span> {t.text}</span>
                </li>
              ))}
            </ol>
            <p>
              When a new fact contradicts an older one, the older one is superseded and the change stays
              visible as a conflict. Memories fade at a pace set by their category: identity lasts, a task
              is forgotten within days. Every change keeps its history and can be rolled back.
            </p>
          </Block>

          <Block icon={<Search size={18} />} title="Search and answers">
            <p>
              Search combines meaning (embeddings) and exact words, then re-ranks the results. Ask composes
              an answer from your memories, cites the ones it used, and points out the ones that disagree.
            </p>
          </Block>

          <Block icon={<BookOpen size={18} />} title="Skills">
            <p>
              Reusable instructions (a release checklist, a review procedure, a house style) stored once and
              available to every assistant. Only their names and descriptions go into the briefing; an
              assistant loads the full text when it needs it.
            </p>
          </Block>

          <Block icon={<Layers size={18} />} title="Namespaces">
            <p>
              Memories live in namespaces such as <code className="font-mono text-ink">work/project</code>. A
              namespace includes the ones below it, and access can be limited per namespace.
            </p>
          </Block>
        </div>

        <div className="flex justify-end px-6 py-3 border-t border-warm-border bg-paper-warm">
          <button onClick={close} className="btn btn-primary">Got it</button>
        </div>
      </div>
    </div>
  )
}
