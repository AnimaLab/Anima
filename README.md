# Anima

[License: Apache 2.0](LICENSE)
[Benchmarks](https://github.com/AnimaLab/Anima-Benchmarks)

Anima is a local-first memory engine for AI assistants and agent workflows. It
helps systems remember facts, preferences, decisions, and prior context across
sessions without relying on a hosted memory backend.

Under the hood, Anima combines a Rust API server, SQLite storage, hybrid
retrieval, local embeddings, a TypeScript MCP bridge, and a web UI for
inspecting and managing memory.

It is built for teams that want persistent, auditable memory with minimal
infrastructure and full data control. The current public release is text-first:
it supports memory ingestion, retrieval, revisions, audit trails, and
REST/MCP-based integration. Audio, image, and video ingestion are roadmap
items, not shipped features.

Your memories belong to you, not your LLM provider.

## Benchmark

**87.73% on LOCOMO** — the academic benchmark for long-term conversational
memory. Human baseline: 87.9%. Gap: 0.17%.

Full results and methodology live in
[Anima-Benchmarks](https://github.com/AnimaLab/Anima-Benchmarks). Headline
numbers:

- **87.7%** full-dataset LoCoMo accuracy across all 10 conversations, 1,540 questions
- **93.4%** on `conv-26` only, 152 questions, single-conversation focused run
- **0.864** broad continuous judge score

Dataset guidance lives in [DATASETS.md](DATASETS.md). See also
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR expectations.

## Why Anima

- **Local-first storage**: memories live in a SQLite database file under your
control.
- **Cloud-optional reasoning**: start in smoke mode with no remote LLM, or wire
Anima to Ollama or another OpenAI-compatible backend for deeper reasoning.
- **Hybrid retrieval**: semantic, keyword, and temporal ranking work together
instead of forcing a vector-only workflow.
- **Inspectable memory**: revisions, rollback, audit events, graphs, and
embedding views make the system easier to trust and debug.
- **Multiple integration surfaces**: REST API, MCP server, and web UI are all
part of the repo.

## What Ships Today


| Capability                | Status  | Notes                                                       |
| ------------------------- | ------- | ----------------------------------------------------------- |
| Persistent memory API     | Shipped | Add, search, list, update, merge, and delete memories       |
| Hybrid retrieval          | Shipped | Vector + keyword + temporal ranking with auto-tuned weights |
| Multi-vector search       | Shipped | Named vectors per memory with query routing heuristics      |
| Scalar quantization       | Shipped | Int8 quantized vec indexes with f32 source for reindexing   |
| Discovery mode            | Shipped | "Find similar" — example-based search with pos/neg vectors  |
| Result grouping           | Shipped | Deduplicate by episode, category, or source via `group_by`  |
| Cross-encoder re-ranking  | Shipped | Optional bge-reranker-v2-m3 for better natural language     |
| Confidence + source       | Shipped | Per-memory trust scoring and provenance tracking            |
| Typed memory categories   | Shipped | Per-category decay rates and filtering                      |
| Revisions and audit trail | Shipped | Revision history, rollback, and audit events                |
| Conflict detection        | Shipped | Supersession chains, contradiction ledger, web inspector    |
| Namespace isolation       | Shipped | `X-Anima-Namespace` plus namespace ACL support              |
| Background consolidation  | Shipped | Reflection and deduction pipeline with auto-classification  |
| Provider profiles         | Shipped | Named LLM profiles with per-operation routing               |
| Configurable embeddings   | Shipped | Local ONNX or any OpenAI-compatible API (Cohere, Voyage)    |
| Graceful degradation      | Shipped | Returns retrieval results when LLM backend is down          |
| MCP bridge                | Shipped | Tool surface for assistants and coding agents               |
| Web UI                    | Shipped | Search, chat, graphs, conflicts, embeddings, and settings   |
| Multimodal ingestion      | Roadmap | Audio, image, and video are not public-server features yet  |


## Install

Pick whichever method gets you running fastest.

### Option A: Download a release (no Rust needed)

Download the latest binary for your platform from
[GitHub Releases](https://github.com/AnimaLab/Anima/releases):

```bash
# macOS (Apple Silicon)
curl -L https://github.com/AnimaLab/Anima/releases/latest/download/anima-server-latest-aarch64-apple-darwin.tar.gz | tar xz
cd anima-server-*
./anima-server
```

```bash
# Linux (x86_64)
curl -L https://github.com/AnimaLab/Anima/releases/latest/download/anima-server-latest-x86_64-unknown-linux-gnu.tar.gz | tar xz
cd anima-server-*
./anima-server
```

The release archive includes the binary, `config.default.toml`, and service
templates. No Rust toolchain required.

### Option B: Docker (no Rust needed)

```bash
docker run -p 3000:3000 -v ./anima.db:/app/anima.db ghcr.io/animalab/anima:latest
```

### Option C: Build from source

```bash
cargo build --release -p anima-server
./target/release/anima-server
```

In all cases, Anima starts on `http://localhost:3000`. On first run it downloads
the local embedding models automatically. The web UI is served at the
same address.

## Quickstart

Once Anima is running, the core loop is three steps:

### 1. Store a memory

```bash
curl -X POST http://localhost:3000/api/v1/memories \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"content": "I love hiking on weekends.", "tags": ["preference"]}'
```

### 2. Search

```bash
curl -X POST http://localhost:3000/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"query": "What does the user like to do on weekends?", "limit": 5}'
```

### 3. Ask (requires an LLM backend)

```bash
curl -X POST http://localhost:3000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"question": "What are the user'\''s hobbies?"}'
```

Store and search work with zero configuration. The `/ask` and `/chat` endpoints
need an LLM backend — see the next section.

## LLM Setup

By default Anima expects Ollama running locally. Pull a model and you're set:

```bash
ollama serve
ollama pull qwen3.5:4b
```

To use a cloud API instead (OpenAI, Groq, Together, etc.), edit
`config.default.toml`:

```toml
[llm]
base_url = "https://api.groq.com/openai/v1"
model = "qwen/qwen3-32b"
api_key = ""  # or set OPENAI_API_KEY env var
```

Or use a `.env` file in the project root (loaded automatically):

```
OPENAI_API_KEY=sk-...
```

Without an LLM backend, store and search still work. The `/ask` endpoint
degrades gracefully — it returns retrieval results directly instead of an
LLM-generated answer (the response includes `"degraded": true`).

## Provider Profiles

Use different models for different operations. Define named profiles in
`config.toml` and route each operation to the right one:

```toml
[profiles.fast]
base_url = "http://localhost:11434/v1"
model = "qwen3.5:4b"

[profiles.strong]
base_url = "http://localhost:11434/v1"
model = "qwen3:32b"

[profiles.cloud]
base_url = "https://api.groq.com/openai/v1"
model = "qwen/qwen3-32b"
api_key = ""  # or set CLOUD_API_KEY env var

[routing]
ask = "strong"
chat = "fast"
processor = "cloud"
consolidation = "fast"
```

Multiple profiles can point to the same endpoint with different models (e.g. same
Ollama, different model sizes). Profile names are arbitrary.

When `[profiles]` is absent, the legacy `[llm]`, `[processor]`, and
`[consolidation]` sections work exactly as before. Per-request `llm` overrides in
`/chat` and `/ask` still take priority over the routed profile.

API key resolution per profile: config value > `{PROFILE_NAME}_API_KEY` env >
`OPENAI_API_KEY` env.

## Memory Categories

Every memory has a semantic `category` that controls how fast it decays and how
it's ranked in search. Set it on ingestion or let the reflection pipeline
classify automatically.

**Built-in categories:**


| Category      | Decay half-life | Use case                                         |
| ------------- | --------------- | ------------------------------------------------ |
| `identity`    | ~2,888 days     | Who the user is, names, relationships            |
| `preference`  | ~289 days       | Likes, dislikes, communication style             |
| `routine`     | ~96 days        | Recurring tasks, schedules, habits               |
| `environment` | ~58 days        | Ports, paths, services, technical infra          |
| `inferred`    | ~29 days        | Conclusions Anima drew but user hasn't confirmed |
| `general`     | ~29 days        | Default — uncategorized memories                 |
| `task`        | ~6 days         | Current work, temporary context                  |


**Custom categories:** define your own in `config.toml` with a name and half-life:

```toml
[categories.health]
half_life_days = 144   # temporal score drops to 50% after 144 days

[categories.finance]
half_life_days = 58
```

```bash
# Set category on ingestion (built-in or custom)
curl -X POST "$BASE_URL/api/v1/memories" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"content": "Blood pressure 120/80", "category": "health"}'

# Filter by category
curl "$BASE_URL/api/v1/memories?category=health" \
  -H "X-Anima-Namespace: default"
```

When the reflection pipeline extracts facts from raw memories, it auto-classifies
each fact into the appropriate category. The MCP `memory_add` tool also accepts
`category` as an optional parameter. Unknown categories default to the global
decay rate.

## Re-ranking

Anima includes a cross-encoder re-ranker that dramatically improves retrieval
quality for natural language queries. Enabled by default. On first startup Anima
downloads
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) INT8
(571 MB). The re-ranker scores the top candidates from the initial retrieval
pass using a cross-encoder that sees query and document together, catching
paraphrases and semantic matches that embedding similarity alone misses.

Performance: high-confidence queries skip the re-ranker (about 55ms). Ambiguous
queries use it (about 200ms). Average across mixed workloads is around 160ms.

Tune `top_n` (default 10) to trade latency for quality — fewer candidates =
faster, more = better ordering.

## Confidence and Source Tracking

Every memory has a `confidence` score (0.0–1.0) and a `source` field that
tracks how it was created. Confidence affects search ranking — higher-confidence
memories score higher for the same relevance.


| Source           | Default confidence | When used                             |
| ---------------- | ------------------ | ------------------------------------- |
| `user_stated`    | 1.0                | User explicitly provided this fact    |
| `promoted`       | 0.8                | Promoted from agent working memory    |
| `agent_observed` | 0.7                | Agent noticed this from behavior      |
| `reflected`      | 0.6 (calibrated)   | Extracted by the reflection pipeline  |
| `deduced`        | 0.5 (calibrated)   | Inferred by combining reflected facts |
| `inferred`       | 0.5                | General inference, not user-confirmed |


```bash
# Store with explicit source
curl -X POST "$BASE_URL/api/v1/memories" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"content": "User seems to prefer dark mode", "source": "agent_observed"}'
```

When one memory supersedes another (via correction or reconsolidation), the
conflict is logged in the contradiction ledger for auditability. The `/ask`
endpoint automatically detects conflicts among retrieved memories and includes
them in the response as a `conflicts` array. The web UI has a Conflicts page
for inspecting supersession chains.

## Advanced Search

The `/search` endpoint supports several modes beyond basic hybrid search:

```bash
# Ask-grade retrieval without LLM — runs the full /ask pipeline
# (keyword expansion, entity resolution, temporal supplement, episode expansion)
# but returns search results instead of an LLM-generated answer.
curl -X POST "$BASE_URL/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"query": "What does Zhen like to eat?", "search_mode": "ask_retrieval", "limit": 10}'

# Query rewriting — extract keywords and expand the query before searching.
# Works with hybrid, vector, or keyword modes.
curl -X POST "$BASE_URL/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"query": "weekend hiking plans", "query_rewrite": true, "limit": 10}'

# Discovery — find memories similar to examples (with optional negative examples)
curl -X POST "$BASE_URL/api/v1/memories/discover" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"positive_ids": ["mem-id-1"], "negative_ids": [], "query": "optional topic bias", "limit": 10}'
```

## Embedding Backend

The default embedding backend is a local ONNX model (Qwen3-Embedding-0.6B).
To use a cloud embedding API instead, set the backend to `openai_compat` and
point it at any OpenAI-compatible embedding endpoint:

```toml
[embedding]
backend = "openai_compat"
api_base_url = "https://api.cohere.com/compatibility/v1"
api_model = "embed-v4.0"
api_key = ""  # or set EMBEDDING_API_KEY env var
dimension = 1024
```

Works with OpenAI, Cohere, Voyage, Together, vLLM, or any provider that
implements the `/embeddings` endpoint. API key resolution: config value >
`EMBEDDING_API_KEY` env > `OPENAI_API_KEY` env.

## Interfaces

### REST API

The HTTP server exposes memory CRUD, search, revisions, audit events, chat,
ask, graph, embeddings, planning, and conversation endpoints.

Key routes:


| Route                                 | Description                                                               |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `GET /health`                         | Liveness probe — always 200 if the process is alive                       |
| `GET /health/ready`                   | Readiness probe — checks DB, embedder, processor; returns 503 if degraded |
| `POST /api/v1/memories`               | Add a memory (supports `category` field)                                  |
| `POST /api/v1/memories/search`        | Hybrid search (`ask_retrieval` mode, `query_rewrite` option)              |
| `POST /api/v1/memories/discover`      | Example-based discovery with positive/negative IDs and optional query     |
| `GET /api/v1/memories?category=X`     | List memories filtered by category                                        |
| `GET /api/v1/memories/{id}/revisions` | Revision history                                                          |
| `GET /api/v1/memories/{id}/history`   | Supersession chain (full conflict history for a memory)                   |
| `GET /api/v1/audit/events`            | Audit trail                                                               |
| `GET /api/v1/contradictions`          | Contradiction ledger (all supersessions with old/new content)             |
| `POST /api/v1/ask`                    | Retrieval-augmented answer (includes `conflicts` and `degraded` fields)   |
| `POST /api/v1/chat`                   | Conversational interface (graceful degradation when LLM is down)          |
| `GET /api/v1/profiles`                | Resolved LLM provider profiles and operation routing                      |
| `GET /api/v1/calibration/weights`     | Current hybrid search weights (auto-tuned from observations)              |
| `GET /api/v1/processor/status`        | Background processor metrics                                              |


### MCP bridge

The MCP server wraps Anima as tools such as `memory_search`, `memory_add`,
`memory_update`, `memory_delete`, `memory_stats`, `memory_ask`, and
`memory_reflect`. To use with Claude Code or Claude Desktop, add to your MCP
config:

```json
{
  "mcpServers": {
    "anima": {
      "command": "node",
      "args": ["path/to/mcp/dist/index.js"],
      "env": {
        "ANIMA_BASE_URL": "http://localhost:3000",
        "ANIMA_NAMESPACE": "default"
      }
    }
  }
}
```

### Web UI

The web app gives you a visual surface for memories, search results, graph
relationships, conflicts, chat, and namespace-aware inspection. Open
`http://localhost:3000` in your browser after starting the server.

## Deployment

### Docker

```bash
docker build -t anima-server .
docker run -p 3000:3000 -v ./anima.db:/app/anima.db anima-server
```

The Dockerfile uses a multi-stage build (Rust compile → Debian slim runtime).
The default config is patched to bind `0.0.0.0:3000` and point Ollama URLs at
`host.docker.internal`.

### Running as a Service

Anima can install itself as a system service that starts on boot.

```bash
# Install (auto-detects platform)
anima-server --install

# Check status
anima-server --service-status

# Uninstall
anima-server --uninstall
```


| Platform | Backend | Details                                                                                   |
| -------- | ------- | ----------------------------------------------------------------------------------------- |
| Linux    | systemd | Unit at `/etc/systemd/system/anima-server.service`, logs via `journalctl -u anima-server` |
| macOS    | launchd | Plist at `~/Library/LaunchAgents/dev.anima.server.plist`, starts on login                 |


The install command uses the current working directory and binary path, so run
it from the directory where your `config.toml` and `anima.db` live.

A pre-built systemd unit template is also available at  
[service/anima-server.service](service/anima-server.service) for manual setup.

## Development

### Rust

```bash
cargo test --workspace
```

### Web UI

```bash
cd web
npm install
npm run build
```

### MCP bridge

```bash
cd mcp
npm install
npm run build
```

## Repository Map

- [crates/](crates): Rust workspace — core types, database, embeddings, consolidation, server
- [mcp/](mcp): TypeScript MCP server for tool-based integrations
- [web/](web): React frontend for memory browsing and inspection
- [scripts/](scripts): local startup helpers
- [service/](service): systemd and launchd service templates
- [benchmarks/](benchmarks): benchmark harnesses and curated notes

## Telemetry

Anima collects anonymous usage analytics to help improve the product. This
includes model names (not keys), memory counts (not content), OS and
architecture info, and feature flag settings. No personal data, namespace
names, or memory content is ever transmitted.

Telemetry is enabled by default. To opt out permanently, set `enabled = false`
in `config.toml`:

```toml
[telemetry]
enabled = false
```

The web UI Settings page can also toggle telemetry off, but that only lasts
until the next restart. Edit the config file for a persistent opt-out.

## Contact

For partnerships, pilots, or product questions: [contact@runanima.com](mailto:contact@runanima.com)

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).