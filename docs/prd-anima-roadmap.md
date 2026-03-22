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

**Still open:**

- `/ask` should surface conflicts when relevant ("Note: you previously said X but later said Y")
- Web UI: conflict inspector panel showing supersession chains

---

## 3. Retrieval Quality — PARTIALLY SHIPPED

**Shipped:**

- **Cross-encoder re-ranking pass.** bge-reranker-v2-m3 INT8 (571MB ONNX), configurable via `[reranker]`. Scores top-N candidates after initial retrieval. High-confidence queries skip the reranker (~55ms), ambiguous queries use it (~200ms). Average ~160ms. Auto-downloads on first use.
- **Confidence-aware scoring.** Higher-confidence memories get a small ranking boost in search results.
- **Lowered min_vector_similarity to 0.40.** Catches more paraphrased natural language queries that were previously filtered out.

**Still open:**

- **Expose `/ask`-grade retrieval as a search mode.** Query expansion, entity resolution, and temporal supplement stages should be available without forcing an LLM answer.
- **Query rewriting for recall.** Formalize the `/ask` keyword extraction into a rewrite stage for the basic search endpoint.
- **Configurable embedding backend.** Add Cohere, Voyage, or any OpenAI-compatible embedding API as backends.
- **Hybrid weight auto-tuning.** Close the loop from calibration observations to RRF weights.

---

## 4. Provider / Auth Abstraction

**What exists:** Trait-based `LlmClient` (Ollama or OpenAI-compatible), `Embedder` (local or OpenAI). No OAuth. No token refresh. Config uses raw API keys.

**What's actually needed:** Not full OAuth — Anima is a local-first daemon, not a SaaS product.

- **Provider-aware parameter filtering.** Different LLM backends accept different parameters. The current OpenAI-compatible client sends everything and hopes for the best. It should know which params each backend supports and strip the rest.
- **Token/key management.** Support key rotation, environment variable references in config (`api_key = "$OPENAI_API_KEY"`), and credential file references. Not OAuth flows — just clean secret management for a daemon.
- **Provider profiles.** Instead of one `[llm]` section, support named profiles (`[llm.fast]`, `[llm.strong]`, `[llm.cheap]`) and let different operations use different profiles. Reflection might use a cheap model, `/ask` might use a strong one.

Skip OAuth adapters — proxy/bridge approach works fine for non-standard providers.

---

## 5. Operational Hardening — SHIPPED (v0.2.0)

**Shipped in v0.2.0:**

- **Structured logging with error domains.** Every API error logs with `domain = db|embedding|llm|auth|retrieval|request|internal`. 500s at `error`, 403s at `warn`, 4xx at `debug`.
- **Health endpoints.** `GET /health` (liveness), `GET /health/ready` (checks DB, embedder, processor — returns 503 if degraded).
- **Startup self-test.** DB ping, embedding smoke test, LLM config validation — fails fast with clear diagnostics before serving.
- **Graceful shutdown.** SIGINT/SIGTERM handler, 30s drain timeout for in-flight processor jobs.
- **Cross-platform service install.** `--install`/`--uninstall`/`--service-status` for systemd (Linux), launchd (macOS), NSSM/schtasks (Windows).
- **Service templates.** Shipped in `service/` (systemd unit + launchd plist).
- **Telemetry opt-out notice.** Startup log explains what's collected and how to disable.

**Still open:**

- **Graceful degradation.** If the LLM is down, `/ask` should still return retrieval results with `skip_llm` fallback instead of failing entirely.

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


| #   | Item                                                 | Impact | Effort     | Status  |
| --- | ---------------------------------------------------- | ------ | ---------- | ------- |
| 1   | Re-ranking pass in retrieval                         | High   | Medium     | **v0.3.0** |
| 2   | Typed memory categories + per-category decay         | High   | Medium     | **v0.3.0** |
| 3   | Confidence + source tracking                         | High   | Medium     | **v0.3.0** |
| 4   | Expose /ask-grade retrieval without LLM              | High   | Low        | Pending |
| 5   | Provider profiles (fast/strong/cheap)                | Medium | Low        | Pending |
| 6   | Agent write/read policies + memory verbs             | Medium | Medium     | Pending |
| 7   | Conflict detection + inspector                       | Medium | Medium     | Pending |
| 8   | Namespace promotion for multi-agent                  | Medium | Medium     | Pending |
| 9   | Operational hardening (health, logging, degradation) | Medium | Low-Medium | **v0.2.0** |
| 10  | Provider parameter filtering                         | Low    | Low        | Pending |


---

## What was cut and why

- **Full OAuth adapters** — Anima is a local daemon, not a SaaS. Proxy/bridge pattern works. Provider-aware param filtering solves the real pain.
- **"Memory permission boundaries" as a separate system** — namespace hierarchy + category + confidence already covers this. Adding a separate ACL layer is over-engineering for a single-user system.
- **"Inspectability" as a feature** — the web UI + claim_revisions + audit_log already provide this. The conflict inspector is the only missing piece.
- **"Passive background memory" as a feature request** — this is the consolidation pipeline. It exists. The gap is category-aware write policy, not a new system.

