import { useState, useEffect, useCallback } from 'react'
import { Plus, RefreshCw, ChevronDown, ChevronRight, Loader2, Brain, Download, Upload, Database } from 'lucide-react'
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

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Reset file input so the same file can be re-selected
    e.target.value = ''

    if (!file.name.endsWith('.json')) {
      setBackupError('Only JSON backup files can be imported from the UI. For SQLite restore, replace the database file on the server.')
      return
    }

    const confirmMsg = `Import ${file.name}?\n\nThis will add memories from the backup to the current namespace. Duplicates will be skipped.\n\nThis may take a while for large backups (embedding generation).`
    if (!confirm(confirmMsg)) return

    setImportLoading(true)
    setImportResult(null)
    setBackupError(null)
    try {
      const text = await file.text()
      const backup = JSON.parse(text)
      const result = await api.importBackup(backup, 'merge')
      setImportResult(result)
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
        <div className="flex gap-2 pt-3 border-t border-warm-border">
          <input type="text" value={newNs} onChange={e => setNewNs(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && createNamespace()}
            placeholder="new-namespace"
            className="flex-1 bg-input border border-warm-border rounded-lg px-3 py-1.5 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50" />
          <button onClick={createNamespace} disabled={!newNs.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:bg-paper-deep disabled:text-ink-faint text-white text-sm rounded-lg transition-colors">
            <Plus size={14} /> Add
          </button>
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
                  Import JSON backup
                  <input type="file" accept=".json" onChange={handleImport} className="hidden" disabled={importLoading} />
                </label>
              </div>

              {importResult && (
                <div className="text-xs text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400 rounded-lg px-3 py-2">
                  Imported {importResult.imported} memories, skipped {importResult.skipped} duplicates ({(importResult.elapsed_ms / 1000).toFixed(1)}s)
                </div>
              )}

              {backupError && (
                <div className="text-xs text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400 rounded-lg px-3 py-2">
                  {backupError}
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
