import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, RefreshCw, Eye, EyeOff, Cpu, Cloud, Check, AlertTriangle } from 'lucide-react'
import { api } from '../api/client'
import type { SetupRequest, SetupStatus } from '../api/types'

/** Ollama, for a setup that runs everything on this machine. */
const OLLAMA_URL = 'http://localhost:11434/v1'
const OLLAMA_MODEL = 'qwen3.5:4b'

/** What the built-in models download on the first start that needs them. */
const BUILTIN_EMBEDDING = { name: 'BGE-M3', size: '2.3 GB', gb: 2.3 }
const BUILTIN_RERANKER = { name: 'bge-reranker-v2-m3', size: '570 MB', gb: 0.57 }

const LATER_KEY = 'anima-setup-later'
const CAN_MASK_TEXT = typeof CSS !== 'undefined' && CSS.supports('-webkit-text-security', 'disc')

const fieldLabel = 'block text-[13px] font-medium text-ink mb-1.5'
const helpText = 'text-xs text-ink-muted mt-1.5'

/** Whether this browser chose to set Anima up later. */
export function setupPostponed(): boolean {
  try { return sessionStorage.getItem(LATER_KEY) === '1' } catch { return false }
}
function postponeSetup() {
  try { sessionStorage.setItem(LATER_KEY, '1') } catch { /* ask again next time */ }
}

const host = (url: string | null) => {
  try { return url ? new URL(url).host : '' } catch { return url ?? '' }
}
const sameUrl = (a: string | null, b: string) => !!a && a.replace(/\/+$/, '') === b.trim().replace(/\/+$/, '')

/**
 * One choice in a model list: the built-in model, none, the current remote
 * model on another server, or a model of the server in the form.
 */
type Choice = 'builtin' | 'none' | 'keep' | `model:${string}`

function initialChoices(s: SetupStatus, fresh = false) {
  // A first run starts from an empty form, not from the shipped defaults.
  if (fresh) return { url: '', chat: '', emb: 'builtin' as Choice, rr: 'none' as Choice }
  const url = s.llm.base_url
  const emb: Choice = s.embedding.backend === 'local'
    ? 'builtin'
    : sameUrl(s.embedding.base_url, url) ? `model:${s.embedding.model}` : 'keep'
  const rr: Choice = !s.reranker.enabled
    ? 'none'
    : s.reranker.backend === 'local'
      ? 'builtin'
      : sameUrl(s.reranker.base_url, url) ? `model:${s.reranker.model ?? ''}` : 'keep'
  return { url, chat: s.llm.model, emb, rr }
}

/** The state of the model form, and the request it saves. */
export function useModelForm(status: SetupStatus | null, fresh = false) {
  const [url, setUrl] = useState('')
  const [key, setKey] = useState('')
  const [chat, setChat] = useState('')
  const [emb, setEmb] = useState<Choice>('builtin')
  const [rr, setRr] = useState<Choice>('builtin')
  const [models, setModels] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)

  const reset = useCallback(() => {
    if (!status) return
    const c = initialChoices(status, fresh)
    setUrl(c.url); setChat(c.chat); setEmb(c.emb); setRr(c.rr); setKey('')
  }, [status, fresh])
  useEffect(reset, [reset])

  /** Ask the server for its models; pick sensible ones when none is chosen yet. */
  const loadModels = useCallback(async (pick = false) => {
    if (!url.trim()) return
    setLoading(true); setError(null)
    try {
      const { models } = await api.listSetupModels(url.trim(), key.trim() || undefined)
      setModels(models)
      setLoaded(true)
      if (pick) {
        const embed = models.find(m => /embed/i.test(m))
        const rerank = models.find(m => /rerank/i.test(m))
        const chatModel = models.find(m => !/embed|rerank|whisper|tts|guard/i.test(m))
        if (embed) setEmb(`model:${embed}`)
        if (rerank) setRr(`model:${rerank}`)
        if (chatModel && !models.includes(chat)) setChat(chatModel)
      }
    } catch (e) {
      setModels([]); setLoaded(true)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [url, key, chat])

  // An existing setup lists its server's models right away, with the saved key.
  const autoLoaded = useRef(false)
  useEffect(() => {
    if (!status || fresh || autoLoaded.current || !url.trim()) return
    autoLoaded.current = true
    loadModels(false)
  }, [status, fresh, url, loadModels])

  const toRequest = useCallback((): SetupRequest | null => {
    if (!status) return null
    const k = key.trim() || undefined
    const u = url.trim()
    const embedding: SetupRequest['embedding'] = emb === 'builtin'
      ? { backend: 'local' }
      : emb === 'keep'
        ? { backend: 'openai_compat', base_url: status.embedding.base_url ?? '', model: status.embedding.model }
        : { backend: 'openai_compat', base_url: u, model: emb.slice(6), api_key: k }
    const reranker: SetupRequest['reranker'] = rr === 'none'
      ? { enabled: false }
      : rr === 'builtin'
        ? { enabled: true, backend: 'local' }
        : rr === 'keep'
          ? { enabled: true, backend: 'openai_compat', base_url: status.reranker.base_url ?? '', model: status.reranker.model ?? '' }
          : { enabled: true, backend: 'openai_compat', base_url: u, model: rr.slice(6), api_key: k }
    return {
      embedding,
      llm: status.llm.from_profiles ? undefined : { base_url: u, model: chat.trim(), api_key: k },
      reranker,
    }
  }, [status, url, key, chat, emb, rr])

  const embeddingChanged = !!status && (() => {
    const c = initialChoices(status, fresh)
    return emb !== c.emb || (emb.startsWith('model:') && !sameUrl(status.embedding.base_url, url))
  })()
  const dirty = !!status && (() => {
    const c = initialChoices(status, fresh)
    return url !== c.url || chat !== c.chat || emb !== c.emb || rr !== c.rr || key.trim() !== ''
  })()

  return {
    status, fresh, url, setUrl, key, setKey, chat, setChat, emb, setEmb, rr, setRr,
    models, loading, error, loaded, loadModels, toRequest, reset, embeddingChanged, dirty,
  }
}

type ModelForm = ReturnType<typeof useModelForm>

function ModelSelect({ id, label, help, value, onChange, form, builtin, allowNone, current }: {
  id: string
  label: string
  help: string
  value: Choice
  onChange: (c: Choice) => void
  form: ModelForm
  builtin: string
  allowNone?: boolean
  current?: { model: string | null; base_url: string | null }
}) {
  const ids = [...form.models]
  if (value.startsWith('model:') && !ids.includes(value.slice(6))) ids.unshift(value.slice(6))
  return (
    <div>
      <label htmlFor={id} className={fieldLabel}>{label}</label>
      <select id={id} value={value} onChange={e => onChange(e.target.value as Choice)} className="field w-full">
        <option value="builtin">{builtin}</option>
        {allowNone && <option value="none">None</option>}
        {current?.model && current.base_url && (
          <option value="keep">{current.model} on {host(current.base_url)} (current)</option>
        )}
        {ids.length > 0 && (
          <optgroup label={host(form.url) || 'Server models'}>
            {ids.map(m => <option key={m} value={`model:${m}`}>{m}</option>)}
          </optgroup>
        )}
      </select>
      <p className={helpText}>{help}</p>
    </div>
  )
}

/** Server, key and the three models. Used by the first-run dialog and Settings. */
export function ModelFields({ form }: { form: ModelForm }) {
  const [showKey, setShowKey] = useState(false)
  const s = form.status
  if (!s) return null
  const otherEmbedding = s.embedding.backend !== 'local' && !sameUrl(s.embedding.base_url, form.url)
  const otherReranker = s.reranker.enabled && s.reranker.backend !== 'local' && !sameUrl(s.reranker.base_url, form.url)
  const chatIds = form.models.filter(m => !/embed|rerank/i.test(m))
  const chatListed = chatIds.includes(form.chat)

  return (
    <div className="@container space-y-4">
      <div className="grid @lg:grid-cols-2 gap-4">
        <div>
          <label htmlFor="setup-url" className={fieldLabel}>Server URL</label>
          <input id="setup-url" type="text" value={form.url} onChange={e => form.setUrl(e.target.value)}
            onBlur={() => form.url.trim() && !form.loaded && form.loadModels(true)}
            placeholder="https://api.openai.com/v1" className="field w-full" />
          <p className={helpText}>Any OpenAI-compatible API: OpenAI, Groq, vLLM, Ollama…</p>
        </div>
        <div>
          <label htmlFor="setup-key" className={fieldLabel}>API key</label>
          <div className="flex gap-2">
            <input id="setup-key" name="anima-setup-api-key"
              type={CAN_MASK_TEXT || showKey ? 'text' : 'password'}
              style={CAN_MASK_TEXT && !showKey ? ({ WebkitTextSecurity: 'disc' } as React.CSSProperties) : undefined}
              value={form.key} onChange={e => form.setKey(e.target.value)}
              placeholder={s.llm.has_key && !form.fresh ? 'Leave empty to keep the current key' : 'Optional'}
              autoComplete="off" autoCorrect="off" autoCapitalize="off" spellCheck={false}
              data-1p-ignore data-lpignore="true" data-form-type="other"
              className="field flex-1 min-w-0" />
            <button type="button" onClick={() => setShowKey(v => !v)} aria-pressed={showKey}
              aria-label={showKey ? 'Hide API key' : 'Show API key'} className="btn btn-secondary btn-icon shrink-0">
              {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
          <p className={helpText}>Saved in {s.config_file}.</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button type="button" onClick={() => form.loadModels(true)} disabled={form.loading || !form.url.trim()}
          className="btn btn-secondary">
          {form.loading ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
          Load models
        </button>
        <p role="status" className={`text-xs ${form.error ? 'text-warning' : 'text-ink-muted'}`}>
          {form.loading
            ? 'Asking the server for its models…'
            : form.error
              ? `Could not list the models: ${form.error}`
              : form.loaded
                ? `${form.models.length} model${form.models.length === 1 ? '' : 's'} found.`
                : ''}
        </p>
      </div>

      <div className="grid @lg:grid-cols-3 gap-4">
        <div>
          <label htmlFor="setup-chat" className={fieldLabel}>Chat model</label>
          {s.llm.from_profiles ? (
            <>
              <input id="setup-chat" value={s.llm.model} disabled className="field w-full" />
              <p className={helpText}>Set by [profiles] in the config file.</p>
            </>
          ) : chatIds.length > 0 ? (
            <>
              <select id="setup-chat" value={form.chat} onChange={e => form.setChat(e.target.value)} className="field w-full">
                {!chatListed && <option value={form.chat}>{form.chat || 'Select a model'}</option>}
                {chatIds.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <p className={helpText}>Answers in chat, and derives new memories.</p>
            </>
          ) : (
            <>
              <input id="setup-chat" type="text" value={form.chat} onChange={e => form.setChat(e.target.value)}
                placeholder="model name" className="field w-full" />
              <p className={helpText}>Answers in chat, and derives new memories.</p>
            </>
          )}
        </div>
        <ModelSelect id="setup-embedding" label="Embedding model" form={form}
          value={form.emb} onChange={form.setEmb}
          builtin={`Built-in ${BUILTIN_EMBEDDING.name}`}
          current={otherEmbedding ? { model: s.embedding.model, base_url: s.embedding.base_url } : undefined}
          help="Turns memories into vectors for search." />
        <ModelSelect id="setup-reranker" label="Re-ranking model" form={form}
          value={form.rr} onChange={form.setRr} allowNone
          builtin={`Built-in ${BUILTIN_RERANKER.name}`}
          current={otherReranker ? { model: s.reranker.model, base_url: s.reranker.base_url } : undefined}
          help="Sorts the best search results again, more precisely." />
      </div>

      {(form.emb === 'builtin' || form.rr === 'builtin') && (
        <p className="text-xs text-ink-muted">
          Built-in models run on this machine. The first start that needs them downloads{' '}
          {[
            form.emb === 'builtin' && `${BUILTIN_EMBEDDING.name} (${BUILTIN_EMBEDDING.size})`,
            form.rr === 'builtin' && `${BUILTIN_RERANKER.name} (${BUILTIN_RERANKER.size})`,
          ].filter(Boolean).join(' and ')}{' '}
          from Hugging Face, which takes a few minutes.
        </p>
      )}
      {form.embeddingChanged && s.memories > 0 && (
        <p className="flex gap-2 text-[13px] text-warning">
          <AlertTriangle size={15} className="shrink-0 mt-0.5" />
          <span>
            Your {s.memories} memories were embedded with {s.embedding.model}. After the restart, Settings offers to
            re-index them with the new model; search stays limited until then.
          </span>
        </p>
      )}
    </div>
  )
}

/** Full-page notice while the server restarts; reloads the page once it is back. */
export function RestartWait({ slow }: { slow?: boolean }) {
  useEffect(() => {
    const started = Date.now()
    let sawDown = false
    let stop = false
    const tick = async () => {
      if (stop) return
      try {
        const res = await fetch('/health', { cache: 'no-store' })
        if (res.ok && (sawDown || Date.now() - started > 8000)) {
          window.location.reload()
          return
        }
        if (!res.ok) sawDown = true
      } catch {
        sawDown = true
      }
      setTimeout(tick, 1000)
    }
    const first = setTimeout(tick, 1000)
    return () => { stop = true; clearTimeout(first) }
  }, [])
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" role="alertdialog" aria-live="polite" aria-label="Restarting Anima">
      <div className="bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] px-6 py-5 max-w-sm text-center">
        <Loader2 size={22} className="animate-spin text-accent mx-auto" />
        <p className="text-sm font-semibold text-ink mt-3">Restarting Anima</p>
        <p className="text-[13px] text-ink-muted mt-1">
          {slow
            ? 'Downloading the built-in models on this first start. This can take a few minutes.'
            : 'The page reloads when the server is back.'}
        </p>
      </div>
    </div>
  )
}

/** Save the form and restart; returns an error message, or null once restarting. */
export async function saveAndRestart(body: SetupRequest): Promise<string | null> {
  try {
    await api.saveSetup({ ...body, restart: true })
    return null
  } catch (e) {
    return e instanceof Error ? e.message : String(e)
  }
}

/** First-run dialog: where Anima's models run. */
export function SetupDialog({ status, onClose }: { status: SetupStatus; onClose: () => void }) {
  const [mode, setMode] = useState<'local' | 'api'>('api')
  const [localReranker, setLocalReranker] = useState(true)
  const [ollamaModel, setOllamaModel] = useState(OLLAMA_MODEL)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [restarting, setRestarting] = useState<null | { slow: boolean }>(null)
  const form = useModelForm(status, true)
  const titleRef = useRef<HTMLHeadingElement>(null)

  useEffect(() => { titleRef.current?.focus() }, [])

  const later = () => { postponeSetup(); onClose() }

  const start = async () => {
    setSaving(true); setError(null)
    const body: SetupRequest | null = mode === 'local'
      ? {
          embedding: { backend: 'local' },
          llm: { base_url: OLLAMA_URL, model: ollamaModel.trim() },
          reranker: localReranker ? { enabled: true, backend: 'local' } : { enabled: false },
        }
      : form.toRequest()
    if (!body) { setSaving(false); return }
    const err = await saveAndRestart(body)
    setSaving(false)
    if (err) { setError(err); return }
    setRestarting({ slow: body.embedding.backend === 'local' || body.reranker.backend === 'local' })
  }

  if (restarting) return <RestartWait slow={restarting.slow} />

  const card = (value: 'local' | 'api', icon: React.ReactNode, title: string, text: string) => (
    <label className={`flex gap-3 rounded-lg border p-4 cursor-pointer transition-colors ${
      mode === value ? 'border-accent bg-accent-light/40' : 'border-warm-border hover:bg-paper-deep'
    }`}>
      <input type="radio" name="setup-mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="sr-only" />
      <span className="shrink-0 text-accent mt-0.5">{icon}</span>
      <span className="min-w-0">
        <span className="flex items-center gap-2 text-sm font-semibold text-ink">
          {title}{mode === value && <Check size={14} className="text-accent" />}
        </span>
        <span className="block text-[13px] text-ink-muted mt-0.5">{text}</span>
      </span>
    </label>
  )

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div role="dialog" aria-modal="true" aria-labelledby="setup-title"
        className="bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden">
        <div className="px-6 pt-5 pb-3">
          <h2 id="setup-title" ref={titleRef} tabIndex={-1} className="text-lg font-semibold text-ink outline-none">Set up Anima</h2>
          <p className="text-[13px] text-ink-muted mt-0.5">
            Choose where Anima's models run. You can change it later in Settings.
          </p>
        </div>

        <div className="px-6 pb-5 overflow-y-auto space-y-4">
          <div role="radiogroup" aria-label="Where the models run" className="grid sm:grid-cols-2 gap-3">
            {card('api', <Cloud size={18} />, 'OpenAI-compatible API',
              'One server URL and key for every model: OpenAI, Groq, vLLM…')}
            {card('local', <Cpu size={18} />, 'On this machine',
              'Built-in search models, and a chat model served by Ollama. Nothing leaves this computer.')}
          </div>

          {mode === 'local' ? (
            <div className="space-y-4">
              <fieldset>
                <legend className="text-[13px] font-medium text-ink mb-2">What Anima downloads</legend>
                <ul className="rounded-lg border border-warm-border divide-y divide-warm-border">
                  <li>
                    <label className="flex items-start gap-3 px-3 py-2.5 cursor-not-allowed">
                      <input type="checkbox" checked disabled className="mt-1" />
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13px] font-medium text-ink">Search model: {BUILTIN_EMBEDDING.name}</span>
                        <span className="block text-xs text-ink-muted">Turns memories into vectors, in 100+ languages. Required to search.</span>
                      </span>
                      <span className="text-xs text-ink-muted tabular-nums whitespace-nowrap">{BUILTIN_EMBEDDING.size}</span>
                    </label>
                  </li>
                  <li>
                    <label className="flex items-start gap-3 px-3 py-2.5 cursor-pointer">
                      <input type="checkbox" checked={localReranker} onChange={e => setLocalReranker(e.target.checked)} className="mt-1" />
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13px] font-medium text-ink">Re-ranking model: {BUILTIN_RERANKER.name}</span>
                        <span className="block text-xs text-ink-muted">Sorts the best search results again, more precisely. Optional.</span>
                      </span>
                      <span className="text-xs text-ink-muted tabular-nums whitespace-nowrap">{BUILTIN_RERANKER.size}</span>
                    </label>
                  </li>
                </ul>
                <p className={helpText}>
                  About {(BUILTIN_EMBEDDING.gb + (localReranker ? BUILTIN_RERANKER.gb : 0)).toFixed(1)} GB from Hugging Face,
                  into the <code className="bg-paper-deep px-1 py-0.5 rounded">models</code> folder, on the first start. It takes a few minutes.
                </p>
              </fieldset>
              <div>
                <label htmlFor="setup-ollama" className={fieldLabel}>Chat model (Ollama)</label>
                <input id="setup-ollama" type="text" value={ollamaModel} onChange={e => setOllamaModel(e.target.value)}
                  className="field w-full sm:w-1/2" />
                <p className={helpText}>
                  Served by Ollama at {OLLAMA_URL}. Ollama downloads it once:{' '}
                  <code className="bg-paper-deep px-1 py-0.5 rounded">ollama pull {ollamaModel.trim() || OLLAMA_MODEL}</code>
                </p>
              </div>
            </div>
          ) : (
            <ModelFields form={form} />
          )}

          {error && (
            <p role="alert" className="flex gap-2 text-[13px] text-danger">
              <AlertTriangle size={15} className="shrink-0 mt-0.5" /><span>{error}</span>
            </p>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 px-6 py-3 border-t border-warm-border bg-paper-warm">
          <button onClick={later} className="btn btn-ghost">Later</button>
          <button onClick={start} disabled={saving || (mode === 'api' ? (!form.url.trim() || !form.chat.trim()) : !ollamaModel.trim())}
            className="btn btn-primary">
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Check size={15} />}
            {saving ? 'Checking the models…' : 'Start Anima'}
          </button>
        </div>
      </div>
    </div>
  )
}
