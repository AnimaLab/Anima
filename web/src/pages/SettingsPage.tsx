import { useState, useEffect, useCallback, useRef } from 'react'
import { Plus, RefreshCw, Loader2, Download, Upload, X, AlertTriangle, Check, Pencil, Trash2 } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { useNamespace } from '../hooks/useNamespace'
import { useToast } from '../hooks/useToast'
import { api, apiFetch } from '../api/client'
import type { NamespaceInfo, ProfilesResponse } from '../api/types'

export interface CognitiveConfig {
  max_tier: number
  reflection_enabled: boolean
  deduction_enabled: boolean
  induction_enabled: boolean
}

const DEFAULT_COGNITIVE_CONFIG: CognitiveConfig = {
  max_tier: 4,
  reflection_enabled: true,
  deduction_enabled: true,
  induction_enabled: true,
}

export function loadCognitiveConfig(): CognitiveConfig {
  try {
    const saved = localStorage.getItem('anima-cognitive-config')
    return saved ? { ...DEFAULT_COGNITIVE_CONFIG, ...JSON.parse(saved) } : DEFAULT_COGNITIVE_CONFIG
  } catch {
    return DEFAULT_COGNITIVE_CONFIG
  }
}

export function saveCognitiveConfig(config: CognitiveConfig) {
  localStorage.setItem('anima-cognitive-config', JSON.stringify(config))
}

interface ModelInfo {
  id: string
  owned_by?: string
}

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <div className="relative shrink-0">
      <input type="checkbox" role="switch" checked={checked} onChange={e => onChange(e.target.checked)} disabled={disabled} className="sr-only peer" />
      <div className={`w-9 h-5 rounded-full transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent ${disabled ? 'bg-accent opacity-60' : 'bg-warm-border-strong peer-checked:bg-accent'}`} />
      <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-card shadow-sm transition-transform ${disabled ? 'translate-x-4' : 'peer-checked:translate-x-4'}`} />
    </div>
  )
}

/** One settings section: header, body and an optional footer bar for actions. */
function Section({ id, title, description, action, footer, children }: {
  id: string
  title: string
  description?: React.ReactNode
  action?: React.ReactNode
  footer?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="card scroll-mt-4 overflow-hidden">
      <div className="flex items-start justify-between gap-3 px-5 pt-5">
        <div className="min-w-0">
          <h2 id={`${id}-title`} className="text-sm font-semibold text-ink">{title}</h2>
          {description && <p className="text-[13px] text-ink-muted mt-0.5">{description}</p>}
        </div>
        {action}
      </div>
      <div className="p-5 space-y-4">{children}</div>
      {footer && (
        <div className="border-t border-warm-border px-5 py-3 flex flex-wrap justify-end gap-2">{footer}</div>
      )}
    </section>
  )
}

/** Label and description on the left, switch on the right. The whole row toggles. */
function ToggleRow({ label, description, checked, onChange, disabled }: {
  label: string
  description: string
  checked: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <label className={`flex items-center justify-between gap-4 py-3 first:pt-0 last:pb-0 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}>
      <span className="min-w-0">
        <span className="block text-sm font-medium text-ink">{label}</span>
        <span className="block text-[13px] text-ink-muted">{description}</span>
      </span>
      <Toggle checked={checked} onChange={onChange} disabled={disabled} />
    </label>
  )
}

const fieldLabel = 'block text-[13px] font-medium text-ink mb-1.5'
const helpText = 'text-xs text-ink-muted mt-1.5'

/** Modal shell shared by the confirmation dialogs. */
function Dialog({ titleId, title, icon, onClose, children, footer, wide }: {
  titleId: string
  title: string
  icon: React.ReactNode
  onClose: () => void
  children: React.ReactNode
  footer: React.ReactNode
  wide?: boolean
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`bg-card border border-warm-border rounded-xl shadow-[var(--shadow-overlay)] w-full ${wide ? 'max-w-md' : 'max-w-sm'} overflow-hidden`}
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 px-5 pt-4 pb-2">
          <div className="flex items-center gap-2 min-w-0">
            {icon}
            <h2 id={titleId} className="text-base font-semibold text-ink truncate">{title}</h2>
          </div>
          <button onClick={onClose} aria-label="Close" className="btn btn-ghost btn-icon -mr-2">
            <X size={16} />
          </button>
        </div>
        <div className="px-5 pb-5 space-y-4">{children}</div>
        <div className="flex flex-wrap justify-end gap-2 px-5 py-3 border-t border-warm-border bg-paper-warm">{footer}</div>
      </div>
    </div>
  )
}

export function SettingsPage() {
  const { show: showToast } = useToast()
  const { config, setConfig: setConfigRaw } = useChat()
  const { namespace, setNamespace: setNamespaceRaw, refreshNamespaces: refreshSidebar } = useNamespace()

  const setConfig = (c: typeof config) => { setConfigRaw(c); showToast('Settings saved') }
  const setNamespace = (ns: string) => { setNamespaceRaw(ns); showToast(`Switched to ${ns}`) }
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])
  const [newNs, setNewNs] = useState('')
  const [loadingNs, setLoadingNs] = useState(false)
  const [deleteNsModal, setDeleteNsModal] = useState<NamespaceInfo | null>(null)
  const [deleteConfirmText, setDeleteConfirmText] = useState('')
  const [deletingNs, setDeletingNs] = useState(false)
  const [renameNsModal, setRenameNsModal] = useState<NamespaceInfo | null>(null)
  const [renameNewName, setRenameNewName] = useState('')
  const [renamingNs, setRenamingNs] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [cognitive, setCognitiveState] = useState<CognitiveConfig>(loadCognitiveConfig)
  const [telemetryEnabled, setTelemetryEnabled] = useState(true)
  const [telemetryLocked, setTelemetryLocked] = useState(false)
  const [profiles, setProfiles] = useState<ProfilesResponse | null>(null)
  const setCognitive = (cfg: CognitiveConfig) => {
    setCognitiveState(cfg)
    saveCognitiveConfig(cfg)
    showToast('Settings saved')
  }

  const [userName, setUserName] = useState(() => localStorage.getItem('anima-user-name') || '')
  const [agentName, setAgentName] = useState(() => localStorage.getItem('anima-agent-name') || 'Anima')
  const [agentPersona, setAgentPersona] = useState(() => localStorage.getItem('anima-agent-persona') || '')

  const saveIdentity = () => {
    localStorage.setItem('anima-user-name', userName)
    localStorage.setItem('anima-agent-name', agentName)
    localStorage.setItem('anima-agent-persona', agentPersona)
    showToast('Identity saved')
  }

  const fetchModels = useCallback(async () => {
    setLoadingModels(true)
    setModelError(null)
    try {
      const res = await apiFetch('/api/v1/models')
      if (!res.ok) throw new Error(`${res.status}`)
      const json = await res.json()
      const list: ModelInfo[] = (json.data || []).map((m: { id: string; owned_by?: string }) => ({
        id: m.id,
        owned_by: m.owned_by,
      }))
      setModels(list)
    } catch (e) {
      setModelError(e instanceof Error ? e.message : 'Failed to fetch')
      setModels([])
    } finally {
      setLoadingModels(false)
    }
  }, [])

  useEffect(() => { fetchModels() }, [fetchModels])

  const refreshNamespaces = () => {
    setLoadingNs(true)
    api.listNamespaces().then(setNamespaces).catch(() => {}).finally(() => setLoadingNs(false))
  }

  useEffect(() => { refreshNamespaces() }, [])

  useEffect(() => { api.getProfiles().then(setProfiles).catch(() => {}) }, [])
  useEffect(() => {
    api.getTelemetryConfig().then(res => {
      setTelemetryEnabled(res.enabled)
      setTelemetryLocked(res.locked ?? false)
    }).catch(() => {})
  }, [])

  const toggleTelemetry = (enabled: boolean) => {
    setTelemetryEnabled(enabled)
    api.setTelemetryConfig(enabled, {
      vision: config.vision ?? false,
      tool_use: config.tool_use ?? true,
      streaming: config.streaming ?? true,
      reflection_enabled: cognitive.reflection_enabled,
      deduction_enabled: cognitive.deduction_enabled,
      induction_enabled: cognitive.induction_enabled,
    }).then(res => {
      // The server has the last word: config.toml can keep telemetry off.
      const actual = res.enabled ?? enabled
      setTelemetryEnabled(actual)
      setTelemetryLocked(res.locked ?? false)
      if (res.locked) showToast('Telemetry is turned off in config.toml')
      else if (res.persisted === false) showToast(actual ? 'Telemetry enabled until restart' : 'Telemetry disabled until restart')
      else showToast(actual ? 'Telemetry enabled' : 'Telemetry disabled')
    }).catch(() => showToast('Could not change telemetry'))
  }

  const [backupLoading, setBackupLoading] = useState<'json' | 'sqlite' | null>(null)
  const [importLoading, setImportLoading] = useState(false)
  const [importResult, setImportResult] = useState<{ imported: number; skipped: number; elapsed_ms: number } | null>(null)
  const [backupError, setBackupError] = useState<string | null>(null)
  const [dbSize, setDbSize] = useState<number | null>(null)
  const [importModal, setImportModal] = useState<{ file: File; format: 'json' | 'sqlite' } | null>(null)
  const [importMode, setImportMode] = useState<'merge' | 'replace'>('merge')
  const [vecStatus, setVecStatus] = useState<{
    needs_reindex: boolean;
    existing_dimension?: number;
    requested_dimension?: number;
    sparse_indexed?: boolean;
    sparse_count?: number;
    memory_count?: number;
  } | null>(null)
  const [reindexing, setReindexing] = useState(false)

  useEffect(() => {
    api.getStats().then(s => setDbSize(s.total)).catch(() => {})
    api.getVecStatus().then(setVecStatus).catch(() => {})
  }, [])

  const downloadBackup = async (format: 'json' | 'sqlite') => {
    setBackupLoading(format)
    setBackupError(null)
    try {
      const blob = format === 'sqlite'
        ? await api.exportBackupSqlite()
        : await api.exportBackupJson()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const date = new Date().toISOString().slice(0, 10)
      a.download = `anima-backup-${date}.${format === 'sqlite' ? 'db' : 'json'}`
      a.click()
      URL.revokeObjectURL(url)
      showToast(`${format.toUpperCase()} backup exported`)
    } catch (e: unknown) {
      setBackupError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setBackupLoading(null)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''

    const isJson = file.name.endsWith('.json')
    const isSqlite = file.name.endsWith('.db') || file.name.endsWith('.sqlite')
    if (!isJson && !isSqlite) {
      setBackupError('Unsupported file type. Use a .json or .db backup file.')
      return
    }
    setBackupError(null)
    setImportResult(null)
    setImportMode('merge')
    setImportModal({ file, format: isJson ? 'json' : 'sqlite' })
  }

  const runImport = async () => {
    if (!importModal) return
    const { file, format } = importModal
    setImportModal(null)
    setImportLoading(true)
    setImportResult(null)
    setBackupError(null)
    try {
      if (format === 'sqlite') {
        await api.importBackupSqlite(file)
        setImportResult({ imported: -1, skipped: 0, elapsed_ms: 0 })
      } else {
        const text = await file.text()
        const backup = JSON.parse(text)
        const result = await api.importBackup(backup, importMode)
        setImportResult(result)
      }
    } catch (e: unknown) {
      setBackupError(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setImportLoading(false)
    }
  }

  const runReindex = async () => {
    setReindexing(true)
    try {
      const result = await api.reindex()
      setVecStatus({ needs_reindex: false })
      showToast(`Re-indexed ${result.reindexed} memories (${result.dimension}d)`)
    } catch (e: unknown) {
      showToast(e instanceof Error ? e.message : 'Re-index failed', true)
    } finally {
      setReindexing(false)
    }
  }

  const createNamespace = () => {
    const ns = newNs.trim()
    if (!ns) return
    setNamespaceRaw(ns)
    setNewNs('')
    if (!namespaces.find(n => n.namespace === ns)) {
      setNamespaces(prev => [...prev, { namespace: ns, total_count: 0, active_count: 0 }])
    }
    showToast(`Created namespace "${ns}"`)
    refreshSidebar()
  }

  const confirmDeleteNamespace = async () => {
    if (!deleteNsModal) return
    setDeletingNs(true)
    try {
      const result = await api.deleteNamespace(deleteNsModal.namespace)
      setNamespaces(prev => prev.filter(n => n.namespace !== deleteNsModal.namespace))
      if (namespace === deleteNsModal.namespace) {
        const remaining = namespaces.filter(n => n.namespace !== deleteNsModal.namespace)
        setNamespaceRaw(remaining[0]?.namespace || 'default')
      }
      showToast(`Deleted "${deleteNsModal.namespace}" (${result.deleted_memories} memories)`)
      refreshSidebar()
    } catch (e: unknown) {
      showToast(e instanceof Error ? e.message : 'Delete failed', true)
    } finally {
      setDeletingNs(false)
      setDeleteNsModal(null)
      setDeleteConfirmText('')
    }
  }

  const confirmRenameNamespace = async () => {
    if (!renameNsModal || !renameNewName.trim()) return
    setRenamingNs(true)
    try {
      await api.renameNamespace(renameNsModal.namespace, renameNewName.trim())
      setNamespaces(prev => prev.map(n =>
        n.namespace === renameNsModal.namespace ? { ...n, namespace: renameNewName.trim() } : n
      ))
      if (namespace === renameNsModal.namespace) setNamespaceRaw(renameNewName.trim())
      showToast(`Renamed to "${renameNewName.trim()}"`)
      refreshSidebar()
    } catch (e: unknown) {
      showToast(e instanceof Error ? e.message : 'Rename failed', true)
    } finally {
      setRenamingNs(false)
      setRenameNsModal(null)
      setRenameNewName('')
    }
  }

  const hasProfiles = !!profiles && Object.keys(profiles.profiles).length > 1
  const sections = [
    { id: 'namespaces', label: 'Namespaces' },
    { id: 'identity', label: 'Identity' },
    { id: 'model', label: 'Chat model' },
    { id: 'capabilities', label: 'Capabilities' },
    { id: 'tiers', label: 'Search tiers' },
    { id: 'processing', label: 'Background processing' },
    ...(hasProfiles ? [{ id: 'profiles', label: 'Server profiles' }] : []),
    { id: 'privacy', label: 'Privacy' },
    { id: 'backup', label: 'Backup and restore' },
    { id: 'danger', label: 'Danger zone' },
  ]
  const currentNs = namespaces.find(n => n.namespace === namespace)
    ?? { namespace, total_count: 0, active_count: 0 }
  const importIsDestructive = !!importModal && (importMode === 'replace' || importModal.format === 'sqlite')

  return (
    <div className="page">
      <div className="mb-6">
        <h1 className="page-title">Settings</h1>
        <p className="page-description">Namespaces, assistant identity, model, memory pipeline and data.</p>
      </div>

      <div className="lg:grid lg:grid-cols-[200px_minmax(0,720px)] lg:gap-10">
        {/* In-page navigation: sticky list on desktop, scrollable row below */}
        <nav aria-label="Settings sections" className="mb-4 lg:mb-0 lg:sticky lg:top-0 lg:self-start">
          <ul className="flex lg:flex-col gap-1 overflow-x-auto pb-1 lg:pb-0">
            {sections.map(s => (
              <li key={s.id} className="shrink-0">
                <a
                  href={`#${s.id}`}
                  className={`flex items-center min-h-8 px-3 rounded-lg text-[13px] font-medium whitespace-nowrap transition-colors hover:bg-paper-deep ${
                    s.id === 'danger' ? 'text-danger' : 'text-ink-muted hover:text-ink'
                  }`}
                >
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="space-y-6 min-w-0">
          {/* Index warnings come first: they affect search quality right now */}
          {vecStatus?.needs_reindex && (
            <div role="alert" className="rounded-xl border border-warm-border bg-warning-light p-5">
              <div className="flex items-start gap-3">
                <AlertTriangle size={18} className="text-warning shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <h2 className="text-sm font-semibold text-ink">Embedding model changed</h2>
                  <p className="text-[13px] text-ink-light mt-1">
                    The vector index uses {vecStatus.existing_dimension}d embeddings but the current model produces {vecStatus.requested_dimension}d.
                    Search results may be degraded. Re-index to re-embed all memories with the new model.
                  </p>
                  <button onClick={runReindex} disabled={reindexing} className="btn btn-primary mt-3">
                    {reindexing ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                    {reindexing ? 'Re-indexing…' : 'Re-index all memories'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {vecStatus && !vecStatus.needs_reindex && vecStatus.sparse_indexed === false && (
            <div role="alert" className="rounded-xl border border-warm-border bg-warning-light p-5">
              <div className="flex items-start gap-3">
                <AlertTriangle size={18} className="text-warning shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <h2 className="text-sm font-semibold text-ink">Sparse vectors not indexed</h2>
                  <p className="text-[13px] text-ink-light mt-1">
                    {vecStatus.sparse_count ?? 0} of {vecStatus.memory_count ?? 0} memories have sparse vectors.
                    Re-index to enable improved keyword-semantic matching.
                  </p>
                  <button onClick={runReindex} disabled={reindexing} className="btn btn-primary mt-3">
                    {reindexing ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                    {reindexing ? 'Re-indexing…' : 'Re-index all memories'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ── Namespaces ── */}
          <Section
            id="namespaces"
            title="Namespaces"
            description="Each namespace is an isolated set of memories. Select one to work in it."
            action={
              <button onClick={refreshNamespaces} disabled={loadingNs} aria-label="Refresh namespaces" title="Refresh" className="btn btn-ghost btn-icon -mr-2 -mt-1">
                <RefreshCw size={15} className={loadingNs ? 'animate-spin' : ''} />
              </button>
            }
          >
            <div>
              <label htmlFor="new-namespace" className={fieldLabel}>New namespace</label>
              <div className="flex gap-2">
                <input id="new-namespace" type="text" value={newNs} onChange={e => setNewNs(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && createNamespace()}
                  placeholder="e.g. work/project-x"
                  className="field flex-1 min-w-0" />
                <button onClick={createNamespace} disabled={!newNs.trim()} className="btn btn-primary shrink-0">
                  <Plus size={15} /> Add
                </button>
              </div>
            </div>

            {namespaces.length > 0 && (
              <ul className="border border-warm-border rounded-lg divide-y divide-warm-border max-h-72 overflow-y-auto">
                {namespaces.map(ns => {
                  const isCurrent = namespace === ns.namespace
                  return (
                    <li key={ns.namespace} className="flex items-center gap-2 pl-1 pr-2 hover:bg-paper transition-colors">
                      <button
                        onClick={() => setNamespace(ns.namespace)}
                        aria-current={isCurrent ? 'true' : undefined}
                        title={isCurrent ? 'Current namespace' : `Switch to ${ns.namespace}`}
                        className="flex-1 min-w-0 flex items-center gap-2 min-h-11 px-2 text-left rounded-md"
                      >
                        <span className={`truncate text-sm ${isCurrent ? 'font-semibold text-ink' : 'text-ink-light'}`}>{ns.namespace}</span>
                        {isCurrent && <span className="badge badge-success shrink-0">current</span>}
                      </button>
                      <span className="badge badge-neutral tabular-nums shrink-0 normal-case" title={`${ns.active_count} active of ${ns.total_count} memories`}>
                        {ns.active_count} / {ns.total_count}
                      </span>
                      <button
                        onClick={() => { setRenameNsModal(ns); setRenameNewName(ns.namespace) }}
                        className="btn btn-ghost btn-icon shrink-0"
                        aria-label={`Rename namespace ${ns.namespace}`}
                        title="Rename"
                      >
                        <Pencil size={15} />
                      </button>
                      <button
                        onClick={() => { setDeleteNsModal(ns); setDeleteConfirmText('') }}
                        className="btn btn-ghost btn-icon shrink-0 hover:text-danger"
                        aria-label={`Delete namespace ${ns.namespace}`}
                        title="Delete"
                      >
                        <Trash2 size={15} />
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </Section>

          {/* ── Identity ── */}
          <Section
            id="identity"
            title="Identity"
            description="How the assistant addresses you and how it behaves in chat."
            footer={
              <button onClick={saveIdentity} className="btn btn-primary">
                <Check size={15} /> Save identity
              </button>
            }
          >
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="user-name" className={fieldLabel}>Your name</label>
                <input id="user-name" type="text" value={userName}
                  onChange={e => setUserName(e.target.value)}
                  placeholder="e.g. Zhen"
                  className="field" />
              </div>
              <div>
                <label htmlFor="agent-name" className={fieldLabel}>Agent name</label>
                <input id="agent-name" type="text" value={agentName}
                  onChange={e => setAgentName(e.target.value)}
                  placeholder="e.g. Anima"
                  className="field" />
              </div>
            </div>
            <div>
              <label htmlFor="agent-persona" className={fieldLabel}>Instructions</label>
              <textarea id="agent-persona" value={agentPersona}
                onChange={e => setAgentPersona(e.target.value)}
                placeholder="e.g. You are a warm, thoughtful personal assistant who helps me reflect on my life and remember what matters."
                rows={3}
                aria-describedby="agent-persona-help"
                className="field resize-y min-h-20" />
              <p id="agent-persona-help" className={helpText}>Persona, tone, and behavior. Sent as the system prompt with every message.</p>
            </div>
          </Section>

          {/* ── Chat model ── */}
          <Section
            id="model"
            title="Chat model"
            description="The model that answers in chat. Changes are saved as you make them."
          >
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="base-url" className={fieldLabel}>API base URL</label>
                <input id="base-url" type="text" value={config.base_url} onChange={e => setConfig({ ...config, base_url: e.target.value })}
                  placeholder="http://localhost:11434/v1"
                  className="field" />
              </div>
              <div>
                <label htmlFor="model-name" className={fieldLabel}>Model</label>
                <div className="flex gap-2">
                  {models.length > 0 ? (
                    <select id="model-name" value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })}
                      className="field flex-1 min-w-0">
                      {!models.find(m => m.id === config.model) && <option value={config.model}>{config.model}</option>}
                      {models.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}
                    </select>
                  ) : (
                    <input id="model-name" type="text" value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })}
                      placeholder="model name"
                      className="field flex-1 min-w-0" />
                  )}
                  <button onClick={fetchModels} disabled={loadingModels} aria-label="Refresh model list" title="Refresh model list"
                    className="btn btn-secondary btn-icon shrink-0">
                    {loadingModels ? <Loader2 size={15} className="animate-spin" /> : <RefreshCw size={15} />}
                  </button>
                </div>
                {modelError && <p className="text-xs text-warning mt-1.5">Could not fetch models. Enter the name manually.</p>}
              </div>
            </div>

            <div>
              <label htmlFor="api-key" className={fieldLabel}>API key</label>
              <input id="api-key" type="password" value={config.api_key || ''} onChange={e => setConfig({ ...config, api_key: e.target.value || undefined })}
                placeholder="Optional"
                autoComplete="off"
                aria-describedby="api-key-help"
                className="field" />
              <p id="api-key-help" className={helpText}>Leave empty for local models. Stored in this browser only.</p>
            </div>

            <div className="grid sm:grid-cols-2 gap-x-4 gap-y-5 pt-1">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="temperature" className="text-[13px] font-medium text-ink">Temperature</label>
                  <span className="text-[13px] text-ink font-mono tabular-nums">{(config.temperature ?? 0.7).toFixed(2)}</span>
                </div>
                <input id="temperature" type="range" min="0" max="2" step="0.05" value={config.temperature ?? 0.7}
                  onChange={e => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                  className="w-full" />
                <div className="flex justify-between text-xs text-ink-muted mt-1">
                  <span>Precise</span><span>Creative</span>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="max-tokens" className="text-[13px] font-medium text-ink">Max tokens</label>
                  <span className="text-[13px] text-ink font-mono tabular-nums">{config.max_tokens ?? 4096}</span>
                </div>
                <input id="max-tokens" type="range" min="256" max="16384" step="256" value={config.max_tokens ?? 4096}
                  onChange={e => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
                  className="w-full" />
                <div className="flex justify-between text-xs text-ink-muted mt-1 tabular-nums">
                  <span>256</span><span>16384</span>
                </div>
              </div>
            </div>
          </Section>

          {/* ── Capabilities ── */}
          <Section id="capabilities" title="Capabilities" description="What the chat model is allowed to do.">
            <div className="divide-y divide-warm-border">
              {([
                { key: 'vision' as const, label: 'Vision', desc: 'Send images to the model', defaultVal: false },
                { key: 'tool_use' as const, label: 'Tool use', desc: 'Allow function calling (memory search and add)', defaultVal: true },
                { key: 'streaming' as const, label: 'Streaming', desc: 'Stream responses token by token', defaultVal: true },
              ]).map(cap => (
                <ToggleRow
                  key={cap.key}
                  label={cap.label}
                  description={cap.desc}
                  checked={config[cap.key] ?? cap.defaultVal}
                  onChange={v => setConfig({ ...config, [cap.key]: v })}
                />
              ))}
            </div>
          </Section>

          {/* ── Search tiers ── */}
          <Section id="tiers" title="Search tiers" description="Which memory tiers search results may include. Each tier builds on the one before it.">
            <div className="divide-y divide-warm-border">
              {([
                { tier: 1, label: 'Raw', desc: 'Original input. Always included.' },
                { tier: 2, label: 'Reflected', desc: 'Extracted facts' },
                { tier: 3, label: 'Deduced', desc: 'Cross-memory inferences' },
                { tier: 4, label: 'Induced', desc: 'Synthesized patterns' },
              ] as const).map(({ tier, label, desc }) => (
                <ToggleRow
                  key={tier}
                  label={label}
                  description={desc}
                  checked={cognitive.max_tier >= tier}
                  onChange={() => {
                    const newMaxTier = cognitive.max_tier >= tier ? tier - 1 : tier
                    setCognitive({ ...cognitive, max_tier: Math.max(1, newMaxTier) })
                  }}
                  disabled={tier === 1}
                />
              ))}
            </div>
          </Section>

          {/* ── Background processing ── */}
          <Section id="processing" title="Background processing" description="Pipelines that derive new memories from existing ones.">
            <div className="divide-y divide-warm-border">
              {([
                { key: 'reflection_enabled' as const, label: 'Reflection', desc: 'Extract facts from raw memories' },
                { key: 'deduction_enabled' as const, label: 'Deduction', desc: 'Infer new facts by combining reflected facts' },
                { key: 'induction_enabled' as const, label: 'Induction', desc: 'Synthesize patterns from 3 or more facts' },
              ]).map(({ key, label, desc }) => (
                <ToggleRow
                  key={key}
                  label={label}
                  description={desc}
                  checked={cognitive[key]}
                  onChange={v => setCognitive({ ...cognitive, [key]: v })}
                />
              ))}
            </div>
          </Section>

          {/* ── Server profiles (read-only) ── */}
          {hasProfiles && profiles && (
            <Section
              id="profiles"
              title="Server profiles"
              description={<>Read-only. Configured in <code className="bg-paper-deep px-1 py-0.5 rounded text-xs">config.toml</code>.</>}
            >
              <ul className="grid sm:grid-cols-2 gap-2">
                {Object.entries(profiles.profiles).map(([name, p]) => (
                  <li key={name} className="border border-warm-border rounded-lg px-3 py-2 min-w-0">
                    <div className="text-[13px] font-medium text-ink truncate">{name}</div>
                    <div className="text-xs text-ink-muted font-mono truncate">{p.model}</div>
                  </li>
                ))}
              </ul>
              <div>
                <p className="section-label mb-2">Routing</p>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
                  {(['ask', 'chat', 'processor', 'consolidation'] as const).map(op => profiles.routing[op] && (
                    <div key={op} className="contents">
                      <dt className="text-ink-muted">{op}</dt>
                      <dd className="text-ink font-medium truncate">{profiles.routing[op]}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </Section>
          )}

          {/* ── Privacy ── */}
          <Section id="privacy" title="Privacy" description="What this server reports back to the Anima project.">
            <ToggleRow
              label="Anonymous telemetry"
              description={telemetryLocked
                ? 'Turned off in config.toml ([telemetry] enabled = false).'
                : 'Model names, memory counts, OS info. No personal data. Your choice is kept across restarts.'}
              checked={telemetryEnabled}
              onChange={toggleTelemetry}
              disabled={telemetryLocked}
            />
          </Section>

          {/* ── Backup and restore ── */}
          <Section
            id="backup"
            title="Backup and restore"
            description={dbSize !== null ? `${dbSize.toLocaleString()} memories in the current namespace.` : 'Export your memories or bring a backup back in.'}
          >
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Export</p>
                <p className="text-[13px] text-ink-muted">JSON for the current namespace, SQLite for the whole database.</p>
              </div>
              <div className="flex gap-2 shrink-0">
                <button onClick={() => downloadBackup('json')} disabled={backupLoading !== null} className="btn btn-secondary">
                  {backupLoading === 'json' ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
                  JSON
                </button>
                <button onClick={() => downloadBackup('sqlite')} disabled={backupLoading !== null} className="btn btn-secondary">
                  {backupLoading === 'sqlite' ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
                  SQLite
                </button>
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-warm-border">
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">Import</p>
                <p className="text-[13px] text-ink-muted">Accepts .json or .db files. You choose how to import before anything changes.</p>
              </div>
              <label className={`btn btn-secondary shrink-0 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent ${importLoading ? 'opacity-50 pointer-events-none' : ''}`}>
                {importLoading ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
                Choose file
                <input type="file" accept=".json,.db,.sqlite" onChange={handleFileSelect} className="sr-only" disabled={importLoading} />
              </label>
            </div>

            {importResult && (
              <div role="status" className="flex items-start gap-2 text-[13px] text-success bg-success-light rounded-lg px-3 py-2">
                <Check size={15} className="shrink-0 mt-0.5" />
                <span>
                  {importResult.imported === -1
                    ? 'Database restored from SQLite backup. Reload the page to see changes.'
                    : importResult.imported > 0
                      ? `Imported ${importResult.imported} memories${importResult.skipped > 0 ? `, skipped ${importResult.skipped} duplicates` : ''} (${(importResult.elapsed_ms / 1000).toFixed(1)}s)`
                      : `All ${importResult.skipped} memories already exist. Nothing to import.`
                  }
                </span>
              </div>
            )}

            {backupError && (
              <div role="alert" className="flex items-start gap-2 text-[13px] text-danger bg-danger-light rounded-lg px-3 py-2">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <span>{backupError}</span>
              </div>
            )}
          </Section>

          {/* ── Danger zone ── */}
          <section id="danger" aria-labelledby="danger-title" className="card scroll-mt-4 overflow-hidden border-danger/40">
            <div className="px-5 pt-5">
              <h2 id="danger-title" className="text-sm font-semibold text-danger">Danger zone</h2>
              <p className="text-[13px] text-ink-muted mt-0.5">These actions cannot be undone.</p>
            </div>
            <div className="p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-ink">Delete the current namespace</p>
                  <p className="text-[13px] text-ink-muted">
                    Permanently deletes <strong className="font-semibold text-ink">{currentNs.namespace}</strong> and its {currentNs.total_count} memories.
                  </p>
                </div>
                <button
                  onClick={() => { setDeleteNsModal(currentNs); setDeleteConfirmText('') }}
                  className="btn btn-danger shrink-0"
                >
                  <Trash2 size={15} /> Delete namespace
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>

      {/* Import confirmation */}
      {importModal && (
        <Dialog
          titleId="import-dialog-title"
          title="Import backup"
          icon={<Upload size={16} className="text-ink-muted shrink-0" />}
          onClose={() => setImportModal(null)}
          wide
          footer={
            <>
              <button onClick={() => setImportModal(null)} className="btn btn-secondary">Cancel</button>
              <button onClick={runImport} className={`btn ${importIsDestructive ? 'btn-danger' : 'btn-primary'}`}>
                {importModal.format === 'sqlite' ? 'Restore database' : importMode === 'replace' ? 'Replace and import' : 'Import'}
              </button>
            </>
          }
        >
          <div className="border border-warm-border rounded-lg px-3 py-2.5">
            <div className="text-[13px] text-ink font-medium break-all">{importModal.file.name}</div>
            <div className="text-xs text-ink-muted mt-0.5">
              {(importModal.file.size / 1024).toFixed(1)} KB · {importModal.format.toUpperCase()} format
            </div>
          </div>

          {importModal.format === 'json' && (
            <fieldset>
              <legend className={fieldLabel}>Import mode</legend>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setImportMode('merge')}
                  aria-pressed={importMode === 'merge'}
                  className={`px-3 py-2 text-left rounded-lg border-2 transition-colors ${importMode === 'merge' ? 'border-accent bg-accent-light' : 'border-warm-border hover:border-warm-border-strong'}`}
                >
                  <span className="block text-[13px] font-medium text-ink">Merge</span>
                  <span className="block text-xs text-ink-muted mt-0.5">Skip duplicates</span>
                </button>
                <button
                  onClick={() => setImportMode('replace')}
                  aria-pressed={importMode === 'replace'}
                  className={`px-3 py-2 text-left rounded-lg border-2 transition-colors ${importMode === 'replace' ? 'border-danger bg-danger-light' : 'border-warm-border hover:border-warm-border-strong'}`}
                >
                  <span className={`block text-[13px] font-medium ${importMode === 'replace' ? 'text-danger' : 'text-ink'}`}>Replace</span>
                  <span className="block text-xs text-ink-muted mt-0.5">Delete existing first</span>
                </button>
              </div>
            </fieldset>
          )}

          {importIsDestructive && (
            <div role="alert" className="flex items-start gap-2 bg-warning-light rounded-lg px-3 py-2.5">
              <AlertTriangle size={15} className="text-warning mt-0.5 shrink-0" />
              <p className="text-[13px] text-ink-light">
                {importModal.format === 'sqlite'
                  ? 'This will replace your entire database. A backup of the current database will be created automatically.'
                  : 'This will delete all existing memories in the current namespace before importing.'}
              </p>
            </div>
          )}

          {importModal.format === 'json' && importMode === 'merge' && (
            <p className="text-[13px] text-ink-muted">
              Memories will be added to the current namespace. Duplicates are detected by content and skipped. Embeddings are re-generated on import.
            </p>
          )}
        </Dialog>
      )}

      {/* Rename namespace */}
      {renameNsModal && (
        <Dialog
          titleId="rename-dialog-title"
          title="Rename namespace"
          icon={<Pencil size={16} className="text-ink-muted shrink-0" />}
          onClose={() => setRenameNsModal(null)}
          footer={
            <>
              <button onClick={() => setRenameNsModal(null)} className="btn btn-secondary">Cancel</button>
              <button
                onClick={confirmRenameNamespace}
                disabled={!renameNewName.trim() || renameNewName.trim() === renameNsModal.namespace || renamingNs}
                className="btn btn-primary"
              >
                {renamingNs && <Loader2 size={15} className="animate-spin" />}
                Rename
              </button>
            </>
          }
        >
          <div>
            <label htmlFor="rename-namespace" className={fieldLabel}>New name</label>
            <input
              id="rename-namespace"
              type="text"
              value={renameNewName}
              onChange={e => setRenameNewName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && renameNewName.trim() && renameNewName.trim() !== renameNsModal.namespace && confirmRenameNamespace()}
              className="field"
              autoFocus
            />
            <p className={helpText}>
              Renaming <strong className="font-semibold text-ink">{renameNsModal.namespace}</strong> ({renameNsModal.total_count} memories)
            </p>
          </div>
        </Dialog>
      )}

      {/* Delete namespace */}
      {deleteNsModal && (
        <Dialog
          titleId="delete-dialog-title"
          title="Delete namespace"
          icon={<AlertTriangle size={16} className="text-danger shrink-0" />}
          onClose={() => setDeleteNsModal(null)}
          footer={
            <>
              <button onClick={() => setDeleteNsModal(null)} className="btn btn-secondary">Cancel</button>
              <button
                onClick={confirmDeleteNamespace}
                disabled={deleteConfirmText !== deleteNsModal.namespace || deletingNs}
                className="btn btn-danger"
              >
                {deletingNs && <Loader2 size={15} className="animate-spin" />}
                Delete forever
              </button>
            </>
          }
        >
          <p className="text-[13px] text-ink-light">
            This will permanently delete <strong className="font-semibold text-ink">{deleteNsModal.namespace}</strong> and all <strong className="font-semibold text-ink">{deleteNsModal.total_count}</strong> memories in it. This cannot be undone.
          </p>
          <div>
            <label htmlFor="delete-confirm" className={fieldLabel}>
              Type <strong className="font-semibold">{deleteNsModal.namespace}</strong> to confirm
            </label>
            <input
              id="delete-confirm"
              type="text"
              value={deleteConfirmText}
              onChange={e => setDeleteConfirmText(e.target.value)}
              placeholder={deleteNsModal.namespace}
              autoComplete="off"
              className="field"
              autoFocus
            />
          </div>
        </Dialog>
      )}
    </div>
  )
}
