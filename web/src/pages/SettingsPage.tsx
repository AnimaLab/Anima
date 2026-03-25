import { useState, useEffect, useCallback } from 'react'
import { Plus, RefreshCw, ChevronDown, ChevronRight, Loader2, Brain, Download, Upload, Database, X, AlertTriangle } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { useNamespace } from '../hooks/useNamespace'
import { api } from '../api/client'
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
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} disabled={disabled} className="sr-only peer" />
      <div className={`w-8 h-[18px] rounded-full transition-colors ${disabled ? 'bg-accent opacity-60' : 'bg-paper-deep peer-checked:bg-accent'}`} />
      <div className={`absolute top-[2px] left-[2px] w-[14px] h-[14px] rounded-full transition-all ${disabled ? 'translate-x-[14px] bg-white opacity-60' : 'bg-ink-faint peer-checked:translate-x-[14px] peer-checked:bg-white'}`} />
    </div>
  )
}

export function SettingsPage() {
  const { config, setConfig } = useChat()
  const { namespace, setNamespace } = useNamespace()
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([])
  const [newNs, setNewNs] = useState('')
  const [loadingNs, setLoadingNs] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [cognitive, setCognitiveState] = useState<CognitiveConfig>(loadCognitiveConfig)
  const [telemetryEnabled, setTelemetryEnabled] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [profiles, setProfiles] = useState<ProfilesResponse | null>(null)
  const setCognitive = (cfg: CognitiveConfig) => {
    setCognitiveState(cfg)
    saveCognitiveConfig(cfg)
  }

  const [userName, setUserName] = useState(() => localStorage.getItem('anima-user-name') || '')
  const [agentName, setAgentName] = useState(() => localStorage.getItem('anima-agent-name') || 'Anima')
  const [agentPersona, setAgentPersona] = useState(() => localStorage.getItem('anima-agent-persona') || '')

  const saveIdentity = (key: string, value: string) => {
    localStorage.setItem(key, value)
  }

  const fetchModels = useCallback(async () => {
    const baseUrl = config.base_url?.trim()
    if (!baseUrl) return
    setLoadingModels(true)
    setModelError(null)
    try {
      const url = `${baseUrl.replace(/\/+$/, '')}/models`
      const headers: Record<string, string> = {}
      if (config.api_key) headers['Authorization'] = `Bearer ${config.api_key}`
      const res = await fetch(url, { headers })
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
  }, [config.base_url, config.api_key])

  useEffect(() => { fetchModels() }, [fetchModels])

  const refreshNamespaces = () => {
    setLoadingNs(true)
    api.listNamespaces().then(setNamespaces).catch(() => {}).finally(() => setLoadingNs(false))
  }

  useEffect(() => { refreshNamespaces() }, [])

  useEffect(() => { api.getProfiles().then(setProfiles).catch(() => {}) }, [])
  useEffect(() => { api.getTelemetryConfig().then(res => setTelemetryEnabled(res.enabled)).catch(() => {}) }, [])

  const toggleTelemetry = (enabled: boolean) => {
    setTelemetryEnabled(enabled)
    api.setTelemetryConfig(enabled, {
      vision: config.vision ?? false,
      tool_use: config.tool_use ?? true,
      streaming: config.streaming ?? true,
      reflection_enabled: cognitive.reflection_enabled,
      deduction_enabled: cognitive.deduction_enabled,
      induction_enabled: cognitive.induction_enabled,
    }).catch(() => {})
  }

  const [backupLoading, setBackupLoading] = useState<'json' | 'sqlite' | null>(null)
  const [importLoading, setImportLoading] = useState(false)
  const [importResult, setImportResult] = useState<{ imported: number; skipped: number; elapsed_ms: number } | null>(null)
  const [backupError, setBackupError] = useState<string | null>(null)
  const [dbSize, setDbSize] = useState<number | null>(null)
  const [importModal, setImportModal] = useState<{ file: File; format: 'json' | 'sqlite' } | null>(null)
  const [importMode, setImportMode] = useState<'merge' | 'replace'>('merge')

  useEffect(() => {
    api.getStats().then(s => setDbSize(s.total)).catch(() => {})
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

  const createNamespace = () => {
    const ns = newNs.trim()
    if (!ns) return
    setNamespace(ns)
    setNewNs('')
    if (!namespaces.find(n => n.namespace === ns)) {
      setNamespaces(prev => [...prev, { namespace: ns, total_count: 0, active_count: 0 }])
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold text-ink">Settings</h2>

      {/* ── Namespaces (most commonly accessed) ── */}
      <section className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-ink-light">Namespaces</h3>
          <button onClick={refreshNamespaces} disabled={loadingNs}
            className="flex items-center gap-1 text-[11px] text-ink-muted hover:text-ink transition-colors">
            <RefreshCw size={10} className={loadingNs ? 'animate-spin' : ''} />
          </button>
        </div>
        <div className="flex gap-2 pb-3 border-b border-warm-border">
          <input type="text" value={newNs} onChange={e => setNewNs(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && createNamespace()}
            placeholder="new-namespace"
            className="flex-1 bg-input border border-warm-border rounded-lg px-3 py-1.5 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
          <button onClick={createNamespace} disabled={!newNs.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:bg-paper-deep disabled:text-ink-faint text-white text-sm rounded-lg transition-colors">
            <Plus size={14} /> Add
          </button>
        </div>
        <div className="relative">
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {namespaces.map(ns => (
              <div key={ns.namespace} onClick={() => setNamespace(ns.namespace)}
                className={`flex items-center justify-between px-3 py-2 rounded-lg text-sm cursor-pointer transition-colors ${
                  namespace === ns.namespace
                    ? 'bg-accent-light text-accent font-medium'
                    : 'text-ink-muted hover:bg-paper-deep hover:text-ink'
                }`}>
                <span className="truncate">{ns.namespace}</span>
                <span className="text-[11px] text-ink-faint shrink-0 ml-2 tabular-nums">{ns.active_count}/{ns.total_count}</span>
              </div>
            ))}
          </div>
          {namespaces.length > 4 && (
            <div className="absolute bottom-0 left-0 right-0 h-6 bg-linear-to-t from-card to-transparent pointer-events-none rounded-b-lg" />
          )}
        </div>
      </section>

      {/* ── Identity ── */}
      <section className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-medium text-ink-light">Identity</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-ink-muted mb-1">Your name</label>
            <input type="text" value={userName}
              onChange={e => { setUserName(e.target.value); saveIdentity('anima-user-name', e.target.value) }}
              placeholder="e.g. Zhen"
              className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
          </div>
          <div>
            <label className="block text-xs text-ink-muted mb-1">Agent name</label>
            <input type="text" value={agentName}
              onChange={e => { setAgentName(e.target.value); saveIdentity('anima-agent-name', e.target.value) }}
              placeholder="e.g. Anima"
              className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
          </div>
        </div>
        <div>
          <label className="block text-xs text-ink-muted mb-1">Instructions</label>
          <textarea value={agentPersona}
            onChange={e => { setAgentPersona(e.target.value); saveIdentity('anima-agent-persona', e.target.value) }}
            placeholder="e.g. You are a warm, thoughtful personal assistant who helps me reflect on my life and remember what matters."
            rows={3}
            className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 resize-y min-h-16" />
          <p className="text-[11px] text-ink-faint mt-1">Persona, tone, and behavior. Sent as the system prompt with every message.</p>
        </div>
      </section>

      {/* ── Chat Model ── */}
      <section className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-medium text-ink-light">Chat Model</h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-ink-muted mb-1">API Base URL</label>
            <input type="text" value={config.base_url} onChange={e => setConfig({ ...config, base_url: e.target.value })}
              placeholder="http://localhost:11434/v1"
              className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-ink-muted">Model</label>
              <button onClick={fetchModels} disabled={loadingModels}
                className="text-[11px] text-ink-faint hover:text-ink transition-colors">
                {loadingModels ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
              </button>
            </div>
            {models.length > 0 ? (
              <div className="relative">
                <select value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })}
                  className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink appearance-none focus:outline-none focus:ring-1 focus:ring-accent/50 cursor-pointer">
                  {!models.find(m => m.id === config.model) && <option value={config.model}>{config.model}</option>}
                  {models.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted pointer-events-none" />
              </div>
            ) : (
              <input type="text" value={config.model} onChange={e => setConfig({ ...config, model: e.target.value })}
                placeholder="model name"
                className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
            )}
            {modelError && <p className="text-[11px] text-amber-600 mt-1">Could not fetch models — enter manually</p>}
          </div>
        </div>

        <div>
          <label className="block text-xs text-ink-muted mb-1">API Key</label>
          <input type="password" value={config.api_key || ''} onChange={e => setConfig({ ...config, api_key: e.target.value || undefined })}
            placeholder="Optional — leave empty for local models"
            className="w-full bg-input border border-warm-border rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
        </div>

        {/* Temperature + Max Tokens */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="flex items-center justify-between text-xs text-ink-muted mb-1.5">
              <span>Temperature</span>
              <span className="text-accent font-mono tabular-nums text-[11px]">{(config.temperature ?? 0.7).toFixed(2)}</span>
            </label>
            <input type="range" min="0" max="2" step="0.05" value={config.temperature ?? 0.7}
              onChange={e => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
              className="w-full accent-accent h-1.5" />
            <div className="flex justify-between text-[10px] text-ink-faint mt-0.5">
              <span>Precise</span><span>Creative</span>
            </div>
          </div>
          <div>
            <label className="flex items-center justify-between text-xs text-ink-muted mb-1.5">
              <span>Max Tokens</span>
              <span className="text-accent font-mono tabular-nums text-[11px]">{config.max_tokens ?? 4096}</span>
            </label>
            <input type="range" min="256" max="16384" step="256" value={config.max_tokens ?? 4096}
              onChange={e => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
              className="w-full accent-accent h-1.5" />
            <div className="flex justify-between text-[10px] text-ink-faint mt-0.5">
              <span>256</span><span>16384</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Advanced ── */}
      <section>
        <button onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors mb-3">
          {showAdvanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          Advanced
        </button>

        {showAdvanced && (
          <div className="space-y-4">
            {/* Capabilities */}
            <div className="bg-card border border-warm-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-medium text-ink-light">Capabilities</h3>
              {([
                { key: 'vision' as const, label: 'Vision', desc: 'Send images to the model', defaultVal: false },
                { key: 'tool_use' as const, label: 'Tool Use', desc: 'Allow function calling (memory search/add)', defaultVal: true },
                { key: 'streaming' as const, label: 'Streaming', desc: 'Stream responses token-by-token', defaultVal: true },
              ]).map(cap => (
                <label key={cap.key} className="flex items-center gap-3 cursor-pointer group">
                  <Toggle checked={config[cap.key] ?? cap.defaultVal} onChange={v => setConfig({ ...config, [cap.key]: v })} />
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">{cap.label}</span>
                    <p className="text-[11px] text-ink-faint">{cap.desc}</p>
                  </div>
                </label>
              ))}
            </div>

            {/* Search Tiers */}
            <div className="bg-card border border-warm-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-medium text-ink-light">Search Tiers</h3>
              <p className="text-[11px] text-ink-faint">Which memory tiers to include in search results.</p>
              {([
                { tier: 1, label: 'Raw', desc: 'Original input' },
                { tier: 2, label: 'Reflected', desc: 'Extracted facts' },
                { tier: 3, label: 'Deduced', desc: 'Cross-memory inferences' },
                { tier: 4, label: 'Induced', desc: 'Synthesized patterns' },
              ] as const).map(({ tier, label, desc }) => (
                <label key={tier} className="flex items-center gap-3 cursor-pointer group">
                  <Toggle
                    checked={cognitive.max_tier >= tier}
                    onChange={() => {
                      const newMaxTier = cognitive.max_tier >= tier ? tier - 1 : tier
                      setCognitive({ ...cognitive, max_tier: Math.max(1, newMaxTier) })
                    }}
                    disabled={tier === 1}
                  />
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">{label}</span>
                    <span className="text-[11px] text-ink-faint ml-1.5">{desc}</span>
                  </div>
                </label>
              ))}
            </div>

            {/* Background Processing */}
            <div className="bg-card border border-warm-border rounded-xl p-5 space-y-3">
              <h3 className="text-sm font-medium text-ink-light">Background Processing</h3>
              {([
                { key: 'reflection_enabled' as const, label: 'Reflection', desc: 'Extract facts from raw memories' },
                { key: 'deduction_enabled' as const, label: 'Deduction', desc: 'Infer new facts by combining reflected facts' },
                { key: 'induction_enabled' as const, label: 'Induction', desc: 'Synthesize patterns from 3+ facts' },
              ]).map(({ key, label, desc }) => (
                <label key={key} className="flex items-center gap-3 cursor-pointer group">
                  <Toggle checked={cognitive[key]} onChange={v => setCognitive({ ...cognitive, [key]: v })} />
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">{label}</span>
                    <p className="text-[11px] text-ink-faint">{desc}</p>
                  </div>
                </label>
              ))}
            </div>

            {/* Server Profiles (read-only) */}
            {profiles && Object.keys(profiles.profiles).length > 1 && (
              <div className="bg-card border border-warm-border rounded-xl p-5 space-y-3">
                <h3 className="text-sm font-medium text-ink-light">Server Profiles</h3>
                <p className="text-[11px] text-ink-faint">Configured in <code className="bg-paper-deep px-1 rounded">config.toml</code></p>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(profiles.profiles).map(([name, p]) => (
                    <div key={name} className="bg-paper-deep rounded-lg px-3 py-2">
                      <div className="text-xs font-medium text-ink">{name}</div>
                      <div className="text-[11px] text-ink-muted truncate">{p.model}</div>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-faint">
                  {profiles.routing.ask && <span><strong>ask</strong> → {profiles.routing.ask}</span>}
                  {profiles.routing.chat && <span><strong>chat</strong> → {profiles.routing.chat}</span>}
                  {profiles.routing.processor && <span><strong>processor</strong> → {profiles.routing.processor}</span>}
                  {profiles.routing.consolidation && <span><strong>consolidation</strong> → {profiles.routing.consolidation}</span>}
                </div>
              </div>
            )}

            {/* Telemetry — compact */}
            <div className="bg-card border border-warm-border rounded-xl px-5 py-4">
              <label className="flex items-center gap-3 cursor-pointer group">
                <Toggle checked={telemetryEnabled} onChange={toggleTelemetry} />
                <div>
                  <span className="text-sm text-ink-light group-hover:text-ink transition-colors">Anonymous Telemetry</span>
                  <p className="text-[11px] text-ink-faint">Model names, memory counts, OS info. No personal data.</p>
                </div>
              </label>
            </div>

            {/* Backup & Restore */}
            <div className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-ink-faint" />
                <h3 className="text-sm font-medium text-ink-light">Backup & Restore</h3>
              </div>

              {dbSize !== null && (
                <p className="text-[11px] text-ink-faint">{dbSize.toLocaleString()} memories in database</p>
              )}

              <div className="space-y-2">
                <p className="text-xs text-ink-muted">Export</p>
                <div className="flex gap-2">
                  <button
                    onClick={() => downloadBackup('json')}
                    disabled={backupLoading !== null}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-paper-deep hover:bg-warm-border text-ink-light rounded-lg transition-colors disabled:opacity-50"
                  >
                    {backupLoading === 'json' ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                    JSON
                  </button>
                  <button
                    onClick={() => downloadBackup('sqlite')}
                    disabled={backupLoading !== null}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-paper-deep hover:bg-warm-border text-ink-light rounded-lg transition-colors disabled:opacity-50"
                  >
                    {backupLoading === 'sqlite' ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                    SQLite
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-xs text-ink-muted">Import</p>
                <label className={`flex items-center gap-1.5 px-3 py-1.5 text-xs bg-paper-deep hover:bg-warm-border text-ink-light rounded-lg transition-colors cursor-pointer w-fit ${importLoading ? 'opacity-50 pointer-events-none' : ''}`}>
                  {importLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Upload className="w-3 h-3" />}
                  Import backup
                  <input type="file" accept=".json,.db,.sqlite" onChange={handleFileSelect} className="hidden" disabled={importLoading} />
                </label>
                <p className="text-[11px] text-ink-faint">Accepts .json or .db files</p>
              </div>

              {importResult && (
                <div className="text-xs text-green-900 bg-green-100 border border-green-300 rounded-lg px-3 py-2">
                  {importResult.imported === -1
                    ? 'Database restored from SQLite backup. Reload the page to see changes.'
                    : importResult.imported > 0
                      ? `Imported ${importResult.imported} memories${importResult.skipped > 0 ? `, skipped ${importResult.skipped} duplicates` : ''} (${(importResult.elapsed_ms / 1000).toFixed(1)}s)`
                      : `All ${importResult.skipped} memories already exist — nothing to import`
                  }
                </div>
              )}

              {backupError && (
                <div className="text-xs text-red-900 bg-red-100 border border-red-300 rounded-lg px-3 py-2">
                  {backupError}
                </div>
              )}
            </div>

            {/* Import Confirmation Modal */}
            {importModal && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm" onClick={() => setImportModal(null)}>
                <div className="bg-card border border-warm-border rounded-2xl shadow-xl w-full max-w-md mx-4 overflow-hidden" onClick={e => e.stopPropagation()}>
                  <div className="flex items-center justify-between px-5 pt-5 pb-3">
                    <div className="flex items-center gap-2">
                      <Upload className="w-4 h-4 text-ink-faint" />
                      <h3 className="text-sm font-semibold text-ink">Import Backup</h3>
                    </div>
                    <button onClick={() => setImportModal(null)} className="text-ink-faint hover:text-ink transition-colors p-1 rounded-lg hover:bg-paper-deep">
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="px-5 pb-4 space-y-4">
                    <div className="bg-paper-deep rounded-lg px-3 py-2.5">
                      <div className="text-xs text-ink-light font-medium">{importModal.file.name}</div>
                      <div className="text-[11px] text-ink-faint mt-0.5">
                        {(importModal.file.size / 1024).toFixed(1)} KB · {importModal.format.toUpperCase()} format
                      </div>
                    </div>

                    {importModal.format === 'json' && (
                      <div className="space-y-2">
                        <p className="text-xs text-ink-muted font-medium">Import mode</p>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setImportMode('merge')}
                            className={`flex-1 px-3 py-2 text-xs rounded-lg border-2 transition-colors ${importMode === 'merge' ? 'border-accent bg-accent-light text-ink' : 'border-warm-border text-ink-muted hover:border-ink-faint'}`}
                          >
                            <div className="font-medium">Merge</div>
                            <div className="text-[11px] mt-0.5 opacity-70">Skip duplicates</div>
                          </button>
                          <button
                            onClick={() => setImportMode('replace')}
                            className={`flex-1 px-3 py-2 text-xs rounded-lg border-2 transition-colors ${importMode === 'replace' ? 'border-red-600 bg-red-100 text-red-800' : 'border-warm-border text-ink-muted hover:border-ink-faint'}`}
                          >
                            <div className="font-medium">Replace</div>
                            <div className="text-[11px] mt-0.5 opacity-70">Delete existing first</div>
                          </button>
                        </div>
                      </div>
                    )}

                    {(importMode === 'replace' || importModal.format === 'sqlite') && (
                      <div className="flex items-start gap-2 bg-amber-100 text-amber-900 rounded-lg px-3 py-2.5 border border-amber-300">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <p className="text-[11px]">
                          {importModal.format === 'sqlite'
                            ? 'This will replace your entire database. A backup of the current database will be created automatically.'
                            : 'This will delete all existing memories in the current namespace before importing.'}
                        </p>
                      </div>
                    )}

                    {importModal.format === 'json' && importMode === 'merge' && (
                      <p className="text-[11px] text-ink-faint">
                        Memories will be added to the current namespace. Duplicates are detected by content and skipped. Embeddings are re-generated on import.
                      </p>
                    )}
                  </div>

                  <div className="flex justify-end gap-2 px-5 py-3 border-t border-warm-border bg-paper-deep/50">
                    <button onClick={() => setImportModal(null)} className="px-3 py-1.5 text-xs text-ink-muted hover:text-ink rounded-lg transition-colors">
                      Cancel
                    </button>
                    <button
                      onClick={runImport}
                      className={`px-4 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                        importMode === 'replace' || importModal.format === 'sqlite'
                          ? 'bg-red-500 hover:bg-red-600 text-white'
                          : 'bg-accent hover:bg-accent/90 text-white'
                      }`}
                    >
                      {importModal.format === 'sqlite' ? 'Restore Database' : importMode === 'replace' ? 'Replace & Import' : 'Import'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  )
}
