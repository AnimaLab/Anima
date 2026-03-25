# Anima Roadmap PRD

## 1. Typed Memory with Semantic Categories — SHIPPED (v0.3.0)

**Shipped:**

- `category` column on memories table: `identity`, `preference`, `environment`, `routine`, `task`, `inferred`, `general`
- Per-category temporal decay rates (identity ~permanent, task ~6 day half-life)
- Category-aware scoring in the search pipeline — per-category decay correction applied after fusion
- Idempotent DB migration — existing memories backfill as `general`
- `category` field on REST API: add, get, list (with filter), search results
- MCP `memory_add` tool gains `category` parameter
- Reflection pipeline auto-classifies extracted facts into categories via LLM

**Still open:**

- Per-category dedup thresholds (preferences: aggressive merge, environment: keep distinct)
- Category-aware retrieval mode in `/ask` — weight categories based on query intent detection

---

## 2. Memory Confidence + Conflict Resolution — SHIPPED (v0.3.0)

**Shipped:**

- `confidence` (0.0–1.0) and `source` (string) columns on memories table
- Source-based default confidence: user_stated=1.0, agent_observed=0.7, inferred=0.5, reflected/deduced/induced=calibrated
- Confidence boost in search scoring — higher-confidence memories rank higher
- Reflection/deduction/induction pipelines set confidence from calibrated values and source from tier
- Reconsolidation now logs every supersession to contradiction_ledger with old/new content and confidence
- API exposes confidence and source on add, get, list, and search responses

**Shipped (v0.4.0):**

- **`/ask` conflict detection.** When retrieved memories have supersession history, the response includes a `conflicts` array with `ConflictNote` entries showing old vs new content, resolution method, and timestamp. Clients/agents can surface "Note: you previously said X but later said Y" from this data.
- **`GET /api/v1/contradictions` endpoint.** Lists contradiction ledger entries with pagination, joining old/new memory content for display. Supports offset/limit.
- **`GET /api/v1/memories/{id}/history` endpoint.** Returns the full supersession chain for a memory — walks backward to the earliest ancestor, then forward through all successors, returning content, status, confidence, and timestamps for each link.
- **Web UI: Conflicts page.** Sidebar nav with conflict inspector showing contradiction list (old/new diff, resolution type, date) and supersession chain viewer (click any entry to see the full version history with active/superseded status).

---

## 3. Retrieval Quality — SHIPPED

**Shipped (v0.3.0):**

- **Cross-encoder re-ranking pass.** bge-reranker-v2-m3 INT8 (571MB ONNX), configurable via `[reranker]`. Scores top-N candidates after initial retrieval. High-confidence queries skip the reranker (~55ms), ambiguous queries use it (~200ms). Average ~160ms. Auto-downloads on first use.
- **Confidence-aware scoring.** Higher-confidence memories get a small ranking boost in search results.
- **Lowered min_vector_similarity to 0.40.** Catches more paraphrased natural language queries that were previously filtered out.

**Shipped (v0.4.0):**

- **`search_mode: "ask_retrieval"` on `/search`.** Exposes the full multi-stage `/ask` retrieval pipeline (keyword expansion, entity resolution, temporal supplement, episode expansion, entity-linked retrieval) as a search mode — returns ranked results without forcing an LLM answer. Shared `run_ask_retrieval_pipeline()` function used by both `/ask` and `/search`.
- **Query rewriting for `/search`.** New `query_rewrite: true` parameter on search requests. Runs the same zero-LLM keyword extraction from `/ask` (`extract_keyword_queries`) to expand queries before searching. Works with all standard search modes (hybrid/vector/keyword).
- **Configurable embedding backend.** `"openai_compat"` backend supports any OpenAI-compatible embedding API (Cohere, Voyage, Together, vLLM, etc.) via custom `api_base_url`. API key resolution chain: config > `EMBEDDING_API_KEY` env > `OPENAI_API_KEY` env.
- **Hybrid weight auto-tuning.** Background auto-tuner runs every calibration cycle (default 5 min), analyzes retrieval observations with component scores (vector_score, keyword_score), computes optimal RRF weights via outcome correlation, and applies changes if >5% shift (0.1 floor on either weight). `scorer_config` is now `RwLock<ScorerConfig>` for live updates. New `GET /api/v1/calibration/weights` endpoint to inspect current weights.

---

## 4. Provider / Auth Abstraction

**What exists:** Trait-based `LlmClient` (Ollama or OpenAI-compatible), `Embedder` (local or OpenAI). No OAuth. No token refresh. Config uses raw API keys.

**What's actually needed:** Not full OAuth — Anima is a local-first daemon, not a SaaS product.

- **Provider-aware parameter filtering.** Different LLM backends accept different parameters. The current OpenAI-compatible client sends everything and hopes for the best. It should know which params each backend supports and strip the rest.
- **Token/key management.** Support key rotation, environment variable references in config (`api_key = "$OPENAI_API_KEY"`), and credential file references. Not OAuth flows — just clean secret management for a daemon.
- ~~**Provider profiles.**~~ **SHIPPED (v0.4.0).** Named `[profiles.*]` sections in config.toml (arbitrary names, each with `base_url`, `model`, `api_key`). Multiple profiles can share the same endpoint with different models. `[routing]` table maps operations (`ask`, `chat`, `processor`, `consolidation`) to profile names. Backward compatible — legacy `[llm]`/`[processor]`/`[consolidation]` synthesized into profiles when `[profiles]` is absent. Per-request client overrides still win. `GET /api/v1/profiles` endpoint exposes routing for clients. Web UI shows server profiles in Settings.

Skip OAuth adapters — proxy/bridge approach works fine for non-standard providers.

---

## 5. Operational Hardening — SHIPPED

**Shipped in v0.2.0:**

- **Structured logging with error domains.** Every API error logs with `domain = db|embedding|llm|auth|retrieval|request|internal`. 500s at `error`, 403s at `warn`, 4xx at `debug`.
- **Health endpoints.** `GET /health` (liveness), `GET /health/ready` (checks DB, embedder, processor — returns 503 if degraded).
- **Startup self-test.** DB ping, embedding smoke test, LLM config validation — fails fast with clear diagnostics before serving.
- **Graceful shutdown.** SIGINT/SIGTERM handler, 30s drain timeout for in-flight processor jobs.
- **Cross-platform service install.** `--install`/`--uninstall`/`--service-status` for systemd (Linux), launchd (macOS), NSSM/schtasks (Windows).
- **Service templates.** Shipped in `service/` (systemd unit + launchd plist).
- **Telemetry opt-out notice.** Startup log explains what's collected and how to disable.

**Shipped (v0.4.0):**

- **Graceful degradation.** When the LLM backend is unavailable, `/ask` returns structured retrieval results (numbered, dated) with `degraded: true` instead of failing. `/chat` does the same via `safe_fallback_chat_reply` with `degraded: true`. Fact extraction is skipped when degraded. The retry-on-IDK logic is also skipped when already degraded to avoid waiting on a down LLM twice.

---

## 6. Agent Integration Layer

**What exists:** MCP server with 8 tools. Claude can search, add, update, delete, ask, and trigger reflection.

**What's needed:**

- **Write policies.** Add a `memory_policy` tool that returns the current write policy (e.g., "always store identity facts, always store preferences, skip transient task chatter, require user confirmation for sensitive topics").
- **Read policies.** Add a `should_consult_memory(query)` lightweight classifier — or document heuristics for agents.
- **Memory verbs beyond CRUD:**
  - `memory_replace(old_id, new_content)` — explicit supersession with provenance
  - `memory_temporary(content, ttl)` — auto-expiring task memory
  - `memory_promote(id)` — move from local/session to shared namespace
  - `memory_forget(query)` — find and delete by semantic match, not just ID
- **Namespace hierarchy for agent isolation.** Spawned agents get `{parent_namespace}/agent-{id}` by default, with read access to parent namespace but write access only to their own. Promotion = copy to parent namespace.

---

## 7. Shared vs Private Memory (Multi-Agent)

**What exists:** Namespace isolation. Prefix-matching for hierarchical access. No cross-namespace promotion mechanism.

**Concrete design:**

```
Namespace hierarchy:
  zhen/                     ← global shared (identity, preferences, environment)
  zhen/memi/                ← project-level shared
  zhen/memi/agent-abc/      ← agent-private working memory
```

- **Read policy:** Agents can read their own namespace + all ancestors (already supported via prefix matching).
- **Write policy:** Agents write to their own namespace by default.
- **Promotion:** New endpoint `POST /memories/{id}/promote` moves a memory from agent namespace to parent. Requires confidence >= threshold or explicit user approval.
- **Garbage collection:** Agent namespaces with no access in N days get auto-archived.

---

## Priority Order


| #   | Item                                                 | Impact | Effort     | Status     |
| --- | ---------------------------------------------------- | ------ | ---------- | ---------- |
| 1   | Re-ranking pass in retrieval                         | High   | Medium     | **v0.3.0** |
| 2   | Typed memory categories + per-category decay         | High   | Medium     | **v0.3.0** |
| 3   | Confidence + source tracking                         | High   | Medium     | **v0.3.0** |
| 4   | /ask-grade retrieval, query rewrite, embedding backends, weight auto-tune | High | Low | **v0.4.0** |
| 5   | Provider profiles (fast/strong/cheap)                | Medium | Low        | **v0.4.0** |
| 6   | Agent write/read policies + memory verbs             | Medium | Medium     | Pending    |
| 7   | Conflict detection + inspector                       | Medium | Medium     | **v0.4.0** |
| 8   | Namespace promotion for multi-agent                  | Medium | Medium     | Pending    |
| 9   | Operational hardening (health, logging, degradation) | Medium | Low-Medium | **v0.2.0** |
| 10  | Provider parameter filtering                         | Low    | Low        | Pending    |


---

## What was cut and why

- **Full OAuth adapters** — Anima is a local daemon, not a SaaS. Proxy/bridge pattern works. Provider-aware param filtering solves the real pain.
- **"Memory permission boundaries" as a separate system** — namespace hierarchy + category + confidence already covers this. Adding a separate ACL layer is over-engineering for a single-user system.
- **"Inspectability" as a feature** — the web UI + claim_revisions + audit_log already provide this. The conflict inspector is the only missing piece.
- **"Passive background memory" as a feature request** — this is the consolidation pipeline. It exists. The gap is category-aware write policy, not a new system.

