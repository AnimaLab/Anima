import { useCallback, useEffect, useRef, useState } from 'react'
import {
  X, ArrowLeft, ArrowRight, Check, BookOpen, Target, ListChecks, NotebookPen, Users, FolderTree,
  HardDrive, ShieldCheck, History, GitBranch, Search,
} from 'lucide-react'

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

/** Fades and lifts its content in, `step` × 90 ms after the step appears. */
function Rise({ step = 0, className = '', children }: { step?: number; className?: string; children: React.ReactNode }) {
  return <div className={`anima-rise ${className}`} style={{ animationDelay: `${120 + step * 90}ms` }}>{children}</div>
}

/** A growing bar, for tiers and scores. */
function Bar({ width, step = 0, faint }: { width: string; step?: number; faint?: boolean }) {
  return (
    <div className="h-2 flex-1 rounded-full bg-warm-border/60 overflow-hidden">
      <div className={`anima-grow h-full rounded-full ${faint ? 'bg-ink-faint' : 'bg-accent'}`}
        style={{ width, animationDelay: `${200 + step * 120}ms` }} />
    </div>
  )
}

function Chip({ children, step }: { children: React.ReactNode; step: number }) {
  return (
    <Rise step={step}>
      <span className="inline-flex items-center rounded-full border border-warm-border bg-card px-2.5 py-1 text-[12px] font-medium text-ink whitespace-nowrap">
        {children}
      </span>
    </Rise>
  )
}

function Mark({ size = 44 }: { size?: number }) {
  return (
    <span className="anima-glow inline-flex items-center justify-center rounded-lg bg-[#0B0B0C]" style={{ width: size, height: size }}>
      <svg viewBox="0 0 24 24" width={size * 0.62} height={size * 0.62} fill="#FCEE0A" aria-hidden="true">
        <path d="M12 2 22 21h-5.2L12 11.6 7.2 21H2L12 2Z" />
        <rect x="10" y="16" width="4" height="4" />
      </svg>
    </span>
  )
}

// ---------------------------------------------------------------- illustrations

function LedgerArt() {
  const tools = ['Claude Code', 'Codex', 'Cursor', 'Hermes', 'Any MCP client', 'HTTP API']
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex flex-wrap justify-center gap-2 max-w-md">
        {tools.map((t, i) => <Chip key={t} step={i}>{t}</Chip>)}
      </div>
      <svg width="340" height="34" viewBox="0 0 340 34" className="text-accent" aria-hidden="true">
        {[20, 100, 170, 240, 320].map(x => (
          <path key={x} d={`M${x} 0 Q ${x} 20 170 34`} fill="none" stroke="currentColor" strokeWidth="1.5" className="anima-flow" opacity="0.7" />
        ))}
      </svg>
      <Rise step={6}><Mark /></Rise>
    </div>
  )
}

function BriefingArt() {
  const rows = [
    ['Profile', 'Prefers short answers, works in Go and Rust'],
    ['This project', 'Releases are tagged X.Y.Z; CI must be green'],
    ['Skills', 'release-checklist · code-review · house-style'],
  ]
  return (
    <div className="w-full max-w-md space-y-2">
      <Rise className="text-[11px] font-semibold uppercase tracking-wide text-ink-muted">Session start · briefing</Rise>
      {rows.map(([k, v], i) => (
        <Rise key={k} step={i + 1}>
          <div className="flex gap-3 rounded-lg border border-warm-border bg-card px-3 py-2 text-[12px]">
            <span className="w-24 shrink-0 font-medium text-ink">{k}</span>
            <span className="text-ink-muted truncate">{v}</span>
          </div>
        </Rise>
      ))}
      <Rise step={5} className="flex items-center gap-2 text-[12px] text-ink-muted pt-1">
        <ArrowRight size={13} className="text-accent" /> then, on each message, the memories that match it
      </Rise>
    </div>
  )
}

function TiersArt() {
  const tiers = [
    ['1', 'Raw', 'what was said', '100%'],
    ['2', 'Reflection', 'standalone facts', '78%'],
    ['3', 'Deduction', 'conclusions', '52%'],
    ['4', 'Induction', 'patterns, habits', '34%'],
  ]
  return (
    <div className="w-full max-w-md space-y-2.5">
      {tiers.map(([n, name, what, w], i) => (
        <Rise key={n} step={i} className="flex items-center gap-3">
          <span className="w-5 h-5 shrink-0 rounded-full bg-card border border-warm-border text-[11px] font-semibold text-ink flex items-center justify-center tabular-nums">{n}</span>
          <span className="w-48 shrink-0 text-[12px] whitespace-nowrap"><span className="font-medium text-ink">{name}</span> <span className="text-ink-muted">· {what}</span></span>
          <Bar width={w} step={i} />
        </Rise>
      ))}
    </div>
  )
}

function HonestArt() {
  return (
    <div className="w-full max-w-md space-y-3 text-[12px]">
      <Rise className="flex items-center gap-2">
        <span className="rounded-md border border-warm-border bg-card px-2 py-1 text-ink-muted line-through">Deploys go out on Fridays</span>
        <ArrowRight size={13} className="text-accent shrink-0" />
        <span className="rounded-md border border-accent bg-card px-2 py-1 text-ink">Deploys go out on Tuesdays</span>
      </Rise>
      <Rise step={1} className="flex items-center gap-2 text-ink-muted">
        <GitBranch size={13} className="text-accent" /> The older fact is superseded, and the conflict stays visible.
      </Rise>
      <Rise step={2} className="space-y-1.5 pt-1">
        {[['Identity', '100%'], ['Preference', '70%'], ['Task', '18%']].map(([c, w], i) => (
          <div key={c} className="flex items-center gap-3">
            <span className="w-20 shrink-0 text-ink-muted">{c}</span>
            <Bar width={w} step={i + 2} faint={c === 'Task'} />
          </div>
        ))}
      </Rise>
      <Rise step={4} className="flex items-center gap-2 text-ink-muted">
        <History size={13} className="text-accent" /> Every change has a revision you can roll back.
      </Rise>
    </div>
  )
}

function SearchArt() {
  return (
    <div className="w-full max-w-md space-y-2.5 text-[12px]">
      <Rise className="flex items-center gap-2 rounded-lg border border-warm-border bg-card px-3 py-2">
        <Search size={13} className="text-ink-muted" /><span className="text-ink">What did we decide about the database?</span>
      </Rise>
      {[['Meaning', '82%'], ['Exact words', '55%'], ['Sparse terms', '40%']].map(([k, w], i) => (
        <Rise key={k} step={i + 1} className="flex items-center gap-3">
          <span className="w-24 shrink-0 text-ink-muted">{k}</span><Bar width={w} step={i + 1} />
        </Rise>
      ))}
      <Rise step={4} className="text-ink-muted">Re-ranked, then answered with its sources:</Rise>
      <Rise step={5} className="rounded-lg border border-accent/60 bg-card px-3 py-2 text-ink">
        Postgres 17 for staging, decided in March <span className="text-accent font-medium">[1]</span> <span className="text-accent font-medium">[2]</span>
      </Rise>
    </div>
  )
}

function GoalsArt() {
  return (
    <div className="w-full max-w-md grid grid-cols-2 gap-2 text-[12px]">
      <Rise className="rounded-lg border border-warm-border bg-card p-2.5 col-span-2">
        <div className="flex items-center gap-2 font-medium text-ink"><Target size={13} className="text-accent" /> Migrate staging to Postgres 17</div>
        <div className="flex items-center gap-1.5 mt-2">
          {['Back up', 'Dry run', 'Switch', 'Verify'].map((c, i) => (
            <span key={c} className={`flex items-center gap-1 rounded-full px-2 py-0.5 border ${i < 2 ? 'border-accent text-ink' : 'border-warm-border text-ink-muted'}`}>
              {i < 2 && <Check size={11} className="text-accent" />}{c}
            </span>
          ))}
        </div>
      </Rise>
      <Rise step={1} className="rounded-lg border border-warm-border bg-card p-2.5">
        <div className="flex items-center gap-2 font-medium text-ink"><BookOpen size={13} className="text-accent" /> Skills</div>
        <div className="text-ink-muted mt-1">release-checklist</div>
      </Rise>
      <Rise step={2} className="rounded-lg border border-warm-border bg-card p-2.5">
        <div className="flex items-center gap-2 font-medium text-ink"><ListChecks size={13} className="text-accent" /> Procedures</div>
        <div className="text-ink-muted mt-1">deploy · v1 v2 <span className="text-ink">v3</span></div>
      </Rise>
      <Rise step={3} className="rounded-lg border border-warm-border bg-card p-2.5">
        <div className="flex items-center gap-2 font-medium text-ink"><NotebookPen size={13} className="text-accent" /> Working memory</div>
        <div className="text-ink-muted mt-1">notes, kept if they matter</div>
      </Rise>
      <Rise step={4} className="rounded-lg border border-warm-border bg-card p-2.5">
        <div className="flex items-center gap-2 font-medium text-ink"><Users size={13} className="text-accent" /> Identities</div>
        <div className="text-ink-muted mt-1">"Sam", "S. Lee" → one person</div>
      </Rise>
    </div>
  )
}

function YoursArt() {
  return (
    <div className="w-full max-w-md grid grid-cols-2 gap-3 text-[12px]">
      <Rise className="rounded-lg border border-warm-border bg-card p-3 font-mono leading-relaxed">
        <div className="flex items-center gap-2 font-sans font-medium text-ink mb-1"><FolderTree size={13} className="text-accent" /> Namespaces</div>
        <div className="text-ink">work</div>
        <div className="text-ink-muted pl-3">└ alpha</div>
        <div className="text-ink-muted pl-3">└ beta</div>
        <div className="text-ink">personal</div>
      </Rise>
      <div className="space-y-2">
        {[
          [<HardDrive size={13} className="text-accent" />, 'One SQLite file on your machine'],
          [<ShieldCheck size={13} className="text-accent" />, 'API key, and access per namespace'],
          [<History size={13} className="text-accent" />, 'Export, import and backups'],
        ].map(([icon, text], i) => (
          <Rise key={i} step={i + 1} className="flex items-center gap-2 rounded-lg border border-warm-border bg-card px-3 py-2 text-ink">
            {icon}<span>{text}</span>
          </Rise>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- steps

interface Step {
  title: string
  art: () => React.ReactNode
  body: React.ReactNode
}

const STEPS: Step[] = [
  {
    title: 'Anima is your memory ledger',
    art: LedgerArt,
    body: (
      <p>
        One record of what your assistants learn about you and your work, kept on your machine. Every connected
        assistant reads and writes the same memory: what one learns, the next one knows, and changing tools or
        models never means starting over.
      </p>
    ),
  },
  {
    title: 'Every session starts informed',
    art: BriefingArt,
    body: (
      <p>
        At the start of a session, a connected assistant receives a short briefing: who you are, what matters for
        the current project, and which skills exist. On each message it then gets the memories relevant to it.
        <code className="ml-1 font-mono text-ink">anima-server connect</code> sets this up for Claude Code, Codex,
        Cursor and Hermes.
      </p>
    ),
  },
  {
    title: 'Memory that works in the background',
    art: TiersArt,
    body: (
      <p>
        An LLM processes stored memories in four tiers, each built on the one before. Raw memories are what was
        said. Reflection extracts standalone facts, deduction combines facts into conclusions, and induction finds
        the patterns that recur across many of them.
      </p>
    ),
  },
  {
    title: 'Kept current, kept honest',
    art: HonestArt,
    body: (
      <p>
        When a new fact contradicts an older one, the older one is superseded and the conflict stays visible.
        Memories fade at the pace of their category: identity lasts, a task is forgotten within days. Every change
        is audited, keeps its history and can be rolled back.
      </p>
    ),
  },
  {
    title: 'Search, and answers with sources',
    art: SearchArt,
    body: (
      <p>
        Search combines meaning (embeddings) and exact words, plus sparse terms with BGE-M3, then re-ranks the best results. Ask
        composes an answer from your memories, cites the ones it used, and points out the ones that disagree.
      </p>
    ),
  },
  {
    title: 'Goals, skills and procedures',
    art: GoalsArt,
    body: (
      <>
        <p>
          <span className="text-ink font-medium">Plans</span> track a goal across sessions: checkpoints, recovery
          branches when a step fails, and the final outcome. <span className="text-ink font-medium">Skills</span> are
          reusable instructions shared by every assistant; <span className="text-ink font-medium">procedures</span> are
          versioned ways of doing a task.
        </p>
        <p>
          <span className="text-ink font-medium">Working memory</span> holds provisional notes and keeps those that
          matter. <span className="text-ink font-medium">Identities</span> resolve several names to one person or
          thing. Skills live in the Skills page; plans, procedures, working memory and identities are used by agents
          through the HTTP API.
        </p>
      </>
    ),
  },
  {
    title: 'Yours, and organised',
    art: YoursArt,
    body: (
      <p>
        Memories live in namespaces such as <code className="font-mono text-ink">work/alpha</code>; a namespace
        includes the ones below it, and access can be limited per namespace. Everything sits in one SQLite file on
        your machine, and moves to another one with export and import.
      </p>
    ),
  },
]

/** Introduction to Anima, in steps. */
export function AboutAnima({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [index, setIndex] = useState(0)
  const [direction, setDirection] = useState<'next' | 'back'>('next')
  const nextRef = useRef<HTMLButtonElement>(null)
  const last = index === STEPS.length - 1

  const close = useCallback(() => {
    markAboutSeen()
    onClose()
  }, [onClose])

  const go = useCallback((to: number) => {
    if (to < 0 || to >= STEPS.length) return
    setDirection(to > index ? 'next' : 'back')
    setIndex(to)
  }, [index])

  useEffect(() => { if (open) { setIndex(0); setDirection('next') } }, [open])

  useEffect(() => {
    if (!open) return
    nextRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
      else if (e.key === 'ArrowRight') go(index + 1)
      else if (e.key === 'ArrowLeft') go(index - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, index, go, close])

  if (!open) return null
  const step = STEPS[index]
  const Art = step.art

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={close}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-anima-title"
        aria-describedby="about-anima-body"
        className="bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 px-6 pt-4">
          <span className="text-[12px] font-medium text-ink-muted tabular-nums" aria-live="polite">
            Step {index + 1} of {STEPS.length}
          </span>
          <button onClick={close} aria-label="Close" className="btn btn-ghost btn-icon -mr-2">
            <X size={16} />
          </button>
        </div>

        <div key={index} className={`flex-1 overflow-y-auto px-6 pb-5 ${direction === 'next' ? 'anima-step-next' : 'anima-step-back'}`}>
          <div className="mt-2 rounded-xl bg-paper-deep/70 border border-warm-border px-5 py-6 h-[236px] overflow-hidden flex items-center justify-center">
            <Art />
          </div>
          <h2 id="about-anima-title" className="text-lg font-semibold text-ink mt-5">{step.title}</h2>
          <div id="about-anima-body" className="text-[13px] text-ink-muted mt-1.5 space-y-2 leading-relaxed min-h-[88px]">{step.body}</div>
        </div>

        <div className="flex items-center justify-between gap-3 px-6 py-3 border-t border-warm-border bg-paper-warm">
          <div className="flex items-center gap-1.5" role="tablist" aria-label="Steps">
            {STEPS.map((s, i) => (
              <button
                key={s.title}
                role="tab"
                aria-selected={i === index}
                aria-label={`Step ${i + 1}: ${s.title}`}
                onClick={() => go(i)}
                className={`h-2 rounded-full transition-all duration-300 ${i === index ? 'w-6 bg-accent' : 'w-2 bg-warm-border-strong hover:bg-ink-faint'}`}
              />
            ))}
          </div>
          <div className="flex items-center gap-2">
            {index > 0 ? (
              <button onClick={() => go(index - 1)} className="btn btn-ghost"><ArrowLeft size={15} /> Back</button>
            ) : (
              <button onClick={close} className="btn btn-ghost">Skip</button>
            )}
            <button ref={nextRef} onClick={() => (last ? close() : go(index + 1))} className="btn btn-primary">
              {last ? <>Get started <Check size={15} /></> : <>Next <ArrowRight size={15} /></>}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
