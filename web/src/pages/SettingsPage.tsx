import { useState, useEffect, useCallback } from 'react'
import { Plus, RefreshCw, ChevronDown, Loader2, Brain } from 'lucide-react'
import { useChat } from '../hooks/useChat'
import { useNamespace } from '../hooks/useNamespace'
import { api } from '../api/client'
import type { NamespaceInfo, ProfilesResponse } from '../api/types'

export interface CognitiveConfig {
  max_tier: number  // 1-4: which tiers to include in search
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
  const [profiles, setProfiles] = useState<ProfilesResponse | null>(null)
  const setCognitive = (cfg: CognitiveConfig) => {
    setCognitiveState(cfg)
    saveCognitiveConfig(cfg)
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

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  const refreshNamespaces = () => {
    setLoadingNs(true)
    api.listNamespaces()
      .then(setNamespaces)
      .catch(() => {})
      .finally(() => setLoadingNs(false))
  }

  useEffect(() => {
    refreshNamespaces()
  }, [])

  // Fetch profiles and telemetry config on mount
  useEffect(() => {
    api.getProfiles().then(setProfiles).catch(() => {})
  }, [])
  useEffect(() => {
    api.getTelemetryConfig()
      .then(res => setTelemetryEnabled(res.enabled))
      .catch(() => {})
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
    }).catch(() => {})
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
    <div className="max-w-2xl mx-auto space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-ink">Settings</h2>
        <p className="text-sm text-ink-muted mt-1">Configure model, namespaces, and system prompt</p>
      </div>

      {/* Server Profiles (read-only info from config.toml) */}
      {profiles && Object.keys(profiles.profiles).length > 1 && (
        <section className="space-y-3">
          <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Server Profiles</h3>
          <div className="bg-card border border-warm-border rounded-xl p-4 space-y-3">
            <p className="text-xs text-ink-muted">
              Configured in <code className="bg-paper-deep px-1 rounded">config.toml</code> &mdash; controls which model the server uses per operation.
            </p>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(profiles.profiles).map(([name, p]) => (
                <div key={name} className="bg-paper-deep rounded-lg px-3 py-2">
                  <div className="text-xs font-medium text-ink">{name}</div>
                  <div className="text-[11px] text-ink-muted truncate">{p.model}</div>
                  <div className="text-[10px] text-ink-faint truncate">{p.base_url}</div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-muted pt-1 border-t border-warm-border">
              {profiles.routing.ask && <span><strong>ask</strong> &rarr; {profiles.routing.ask}</span>}
              {profiles.routing.chat && <span><strong>chat</strong> &rarr; {profiles.routing.chat}</span>}
              {profiles.routing.processor && <span><strong>processor</strong> &rarr; {profiles.routing.processor}</span>}
              {profiles.routing.consolidation && <span><strong>consolidation</strong> &rarr; {profiles.routing.consolidation}</span>}
            </div>
          </div>
        </section>
      )}

      {/* Model Configuration */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Model Configuration</h3>
        <div className="bg-card border border-warm-border rounded-xl p-5 space-y-5">
          {/* API Base URL */}
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1.5">API Base URL</label>
            <input
              type="text"
              value={config.base_url}
              onChange={e => setConfig({ ...config, base_url: e.target.value })}
              placeholder="http://localhost:11434/v1"
              className="w-full bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all"
            />
            <p className="text-[11px] text-ink-faint mt-1">OpenAI-compatible API endpoint</p>
          </div>

          {/* Model Selector */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-ink-muted">Model</label>
              <button
                onClick={fetchModels}
                disabled={loadingModels}
                className="flex items-center gap-1 text-[11px] text-ink-muted hover:text-ink transition-colors"
              >
                {loadingModels ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
                Refresh
              </button>
            </div>
            {models.length > 0 ? (
              <div className="relative">
                <select
                  value={config.model}
                  onChange={e => setConfig({ ...config, model: e.target.value })}
                  className="w-full bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-2 text-sm text-ink appearance-none focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all cursor-pointer"
                >
                  {!models.find(m => m.id === config.model) && (
                    <option value={config.model}>{config.model}</option>
                  )}
                  {models.map(m => (
                    <option key={m.id} value={m.id}>{m.id}</option>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-muted pointer-events-none" />
              </div>
            ) : (
              <input
                type="text"
                value={config.model}
                onChange={e => setConfig({ ...config, model: e.target.value })}
                placeholder="model name"
                className="w-full bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all"
              />
            )}
            {modelError && (
              <p className="text-[11px] text-amber-600 mt-1">Could not fetch models ({modelError}) — enter manually</p>
            )}
          </div>

          {/* API Key */}
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-1.5">API Key</label>
            <input
              type="password"
              value={config.api_key || ''}
              onChange={e => setConfig({ ...config, api_key: e.target.value || undefined })}
              placeholder="Optional — leave empty for local models"
              className="w-full bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all"
            />
          </div>

          {/* Temperature + Max Tokens */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div>
              <label className="flex items-center justify-between text-xs font-medium text-ink-muted mb-2">
                <span>Temperature</span>
                <span className="text-accent font-mono tabular-nums">{(config.temperature ?? 0.7).toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={config.temperature ?? 0.7}
                onChange={e => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                className="w-full accent-[#C47B3B] h-1.5"
              />
              <div className="flex justify-between text-[10px] text-ink-faint mt-1">
                <span>Precise</span>
                <span>Balanced</span>
                <span>Creative</span>
              </div>
            </div>
            <div>
              <label className="flex items-center justify-between text-xs font-medium text-ink-muted mb-2">
                <span>Max Tokens</span>
                <span className="text-accent font-mono tabular-nums">{config.max_tokens ?? 4096}</span>
              </label>
              <input
                type="range"
                min="256"
                max="16384"
                step="256"
                value={config.max_tokens ?? 4096}
                onChange={e => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
                className="w-full accent-[#C47B3B] h-1.5"
              />
              <div className="flex justify-between text-[10px] text-ink-faint mt-1">
                <span>256</span>
                <span>4096</span>
                <span>16384</span>
              </div>
              <p className="text-[11px] text-ink-faint mt-1">Maximum length of the model's response</p>
            </div>
          </div>

          {/* Model Capabilities */}
          <div>
            <label className="block text-xs font-medium text-ink-muted mb-2">Capabilities</label>
            <div className="space-y-3">
              {([
                { key: 'vision' as const, label: 'Vision', desc: 'Send images to the model using multimodal format. Disable for text-only models.', defaultVal: false },
                { key: 'tool_use' as const, label: 'Tool Use', desc: 'Allow the model to call functions (memory search/add). Disable if unsupported.', defaultVal: true },
                { key: 'streaming' as const, label: 'Streaming', desc: 'Stream responses token-by-token. Disable if the endpoint does not support SSE.', defaultVal: true },
              ]).map(cap => (
                <label key={cap.key} className="flex items-center gap-3 cursor-pointer group">
                  <div className="relative shrink-0">
                    <input
                      type="checkbox"
                      checked={config[cap.key] ?? cap.defaultVal}
                      onChange={e => setConfig({ ...config, [cap.key]: e.target.checked })}
                      className="sr-only peer"
                    />
                    <div className="w-8 h-[18px] bg-paper-deep rounded-full peer-checked:bg-accent transition-colors" />
                    <div className="absolute top-[2px] left-[2px] w-[14px] h-[14px] bg-ink-faint rounded-full peer-checked:translate-x-[14px] peer-checked:bg-white transition-all" />
                  </div>
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">{cap.label}</span>
                    <p className="text-[11px] text-ink-faint">{cap.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* System Prompt */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">System Prompt</h3>
        <div className="bg-card border border-warm-border rounded-xl p-5">
          <textarea
            value={config.system_prompt || ''}
            onChange={e => setConfig({ ...config, system_prompt: e.target.value || undefined })}
            placeholder="You are a helpful assistant."
            rows={4}
            className="w-full bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-2 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all resize-y min-h-[5rem]"
          />
          <p className="text-[11px] text-ink-faint mt-2">
            Prepended to every chat message. Memory context is appended automatically.
          </p>
        </div>
      </section>

      {/* Cognitive Functions */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Cognitive Functions</h3>
        <div className="bg-card border border-warm-border rounded-xl p-5 space-y-5">
          {/* Search Tier Filter */}
          <div>
            <label className="flex items-center gap-2 text-xs font-medium text-ink-muted mb-3">
              <Brain size={14} />
              Search Tiers
            </label>
            <p className="text-[11px] text-ink-faint mb-3">
              Which memory tiers to include in search and ask results. Higher tiers are more processed but may add noise.
            </p>
            <div className="space-y-2">
              {([
                { tier: 1, label: 'Raw', desc: 'Original user input — highest fidelity' },
                { tier: 2, label: 'Reflected', desc: 'Extracted atomic facts from raw memories' },
                { tier: 3, label: 'Deduced', desc: 'Cross-memory inferences (may be speculative)' },
                { tier: 4, label: 'Induced', desc: 'Synthesized patterns and personality traits' },
              ] as const).map(({ tier, label, desc }) => (
                <label key={tier} className="flex items-center gap-3 cursor-pointer group">
                  <div className="relative shrink-0">
                    <input
                      type="checkbox"
                      checked={cognitive.max_tier >= tier}
                      onChange={() => {
                        // Toggle: if this tier was included, set max_tier to tier-1
                        // If excluded, set max_tier to this tier
                        const newMaxTier = cognitive.max_tier >= tier ? tier - 1 : tier
                        setCognitive({ ...cognitive, max_tier: Math.max(1, newMaxTier) })
                      }}
                      disabled={tier === 1}
                      className="sr-only peer"
                    />
                    <div className={`w-8 h-[18px] rounded-full transition-colors ${tier === 1 ? 'bg-accent opacity-60' : 'bg-paper-deep peer-checked:bg-accent'}`} />
                    <div className={`absolute top-[2px] left-[2px] w-[14px] h-[14px] rounded-full transition-all ${tier === 1 ? 'translate-x-[14px] bg-white opacity-60' : 'bg-ink-faint peer-checked:translate-x-[14px] peer-checked:bg-white'}`} />
                  </div>
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">
                      Tier {tier}: {label}
                    </span>
                    <p className="text-[11px] text-ink-faint">{desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Background Processing Toggles */}
          <div className="pt-4 border-t border-warm-border">
            <label className="block text-xs font-medium text-ink-muted mb-3">Background Processing</label>
            <p className="text-[11px] text-ink-faint mb-3">
              Controls which cognitive processes run automatically when memories are added. These are client-side flags — the server processes them if enabled.
            </p>
            <div className="space-y-3">
              {([
                { key: 'reflection_enabled' as const, label: 'Reflection', desc: 'Extract atomic facts from raw memories (Tier 1 → Tier 2)' },
                { key: 'deduction_enabled' as const, label: 'Deduction', desc: 'Infer new facts by combining 2+ reflected facts (Tier 2 → Tier 3)' },
                { key: 'induction_enabled' as const, label: 'Induction', desc: 'Synthesize stable patterns from 3+ facts (→ Tier 4)' },
              ]).map(({ key, label, desc }) => (
                <label key={key} className="flex items-center gap-3 cursor-pointer group">
                  <div className="relative shrink-0">
                    <input
                      type="checkbox"
                      checked={cognitive[key]}
                      onChange={e => setCognitive({ ...cognitive, [key]: e.target.checked })}
                      className="sr-only peer"
                    />
                    <div className="w-8 h-[18px] bg-paper-deep rounded-full peer-checked:bg-accent transition-colors" />
                    <div className="absolute top-[2px] left-[2px] w-[14px] h-[14px] bg-ink-faint rounded-full peer-checked:translate-x-[14px] peer-checked:bg-white transition-all" />
                  </div>
                  <div>
                    <span className="text-sm text-ink-light group-hover:text-ink transition-colors">{label}</span>
                    <p className="text-[11px] text-ink-faint">{desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Telemetry */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Telemetry</h3>
        <div className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
          <label className="flex items-center gap-3 cursor-pointer group">
            <div className="relative shrink-0">
              <input
                type="checkbox"
                checked={telemetryEnabled}
                onChange={e => toggleTelemetry(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-8 h-[18px] bg-paper-deep rounded-full peer-checked:bg-accent transition-colors" />
              <div className="absolute top-[2px] left-[2px] w-[14px] h-[14px] bg-ink-faint rounded-full peer-checked:translate-x-[14px] peer-checked:bg-white transition-all" />
            </div>
            <div>
              <span className="text-sm text-ink-light group-hover:text-ink transition-colors">Usage Analytics</span>
              <p className="text-[11px] text-ink-faint">Help improve Anima by sharing anonymous usage data</p>
            </div>
          </label>
          <p className="text-[11px] text-ink-faint leading-relaxed">
            Anima collects anonymous usage metrics — model names (not keys), memory counts (not content),
            OS info, and feature settings. No personal data, namespace names, or memory content is ever sent.
            You can disable this at any time.
          </p>
        </div>
      </section>

      {/* Namespace Management */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Namespaces</h3>
          <button
            onClick={refreshNamespaces}
            disabled={loadingNs}
            className="flex items-center gap-1 text-[11px] text-ink-muted hover:text-ink transition-colors"
            title="Refresh"
          >
            <RefreshCw size={10} className={loadingNs ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
        <div className="bg-card border border-warm-border rounded-xl p-5 space-y-4">
          {/* Active namespace */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-muted">Active:</span>
            <span className="text-sm text-accent font-medium bg-accent-light px-2 py-0.5 rounded">{namespace}</span>
          </div>

          {/* Namespace list */}
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {namespaces.map(ns => (
              <div
                key={ns.namespace}
                onClick={() => setNamespace(ns.namespace)}
                className={`flex items-center justify-between px-3 py-2 rounded-lg text-sm cursor-pointer transition-colors ${
                  namespace === ns.namespace
                    ? 'bg-accent-light text-accent border border-accent/20'
                    : 'text-ink-muted hover:bg-paper-deep hover:text-ink'
                }`}
              >
                <span className="truncate">{ns.namespace}</span>
                <span className="text-[11px] text-ink-faint shrink-0 ml-2">
                  {ns.active_count} / {ns.total_count}
                </span>
              </div>
            ))}
            {namespaces.length === 0 && (
              <p className="text-xs text-ink-faint py-3 text-center">No namespaces found</p>
            )}
          </div>

          {/* Create new */}
          <div className="flex gap-2 pt-3 border-t border-warm-border">
            <input
              type="text"
              value={newNs}
              onChange={e => setNewNs(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && createNamespace()}
              placeholder="org/project/user"
              className="flex-1 bg-paper-deep border border-warm-border-strong rounded-lg px-3 py-1.5 text-sm text-ink placeholder-ink-faint focus:outline-none focus:ring-1 focus:ring-accent/50 focus:border-accent/50 transition-all"
            />
            <button
              onClick={createNamespace}
              disabled={!newNs.trim()}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-accent hover:bg-accent-hover disabled:bg-paper-deep disabled:text-ink-faint text-white text-sm rounded-lg transition-colors"
            >
              <Plus size={14} />
              Add
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}
