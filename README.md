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


| Capability                | Status  | Notes                                                      |
| ------------------------- | ------- | ---------------------------------------------------------- |
| Persistent memory API     | Shipped | Add, search, list, update, merge, and delete memories      |
| Hybrid retrieval          | Shipped | Vector + keyword + temporal ranking                        |
| Typed memory categories   | Shipped | Per-category decay rates and filtering                     |
| Revisions and audit trail | Shipped | Revision history, rollback, and audit events               |
| Namespace isolation       | Shipped | `X-Anima-Namespace` plus namespace ACL support             |
| Background consolidation  | Shipped | Reflection and deduction pipeline with auto-classification |
| MCP bridge                | Shipped | Tool surface for assistants and coding agents              |
| Web UI                    | Shipped | Search, chat, graphs, embeddings, and memory inspection    |
| Multimodal ingestion      | Roadmap | Audio, image, and video are not public-server features yet |


## Quickstart

### 1. Start Anima in smoke mode

Smoke mode is the fastest path from clone to working API. No Ollama, no API
keys, no LLM required — just Rust and curl. It disables the features that need
an LLM backend (reflection, deduction, answer generation) so you can test the
core memory loop immediately: store, search, and inspect.

```bash
cargo build --release -p anima-server
eval "$(./scripts/start_local.sh --detached --print-env)"
until curl -fsS "$BASE_URL/health" >/dev/null; do sleep 1; done
curl -sS "$BASE_URL/health"
```

### 2. Store a memory

```bash
curl -X POST "$BASE_URL/api/v1/memories" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{
    "content": "I love hiking on weekends.",
    "tags": ["preference", "outdoors"],
    "consolidate": true
  }'
```

### 3. Retrieve it with hybrid search

```bash
curl -X POST "$BASE_URL/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{
    "query": "What does the user like to do on weekends?",
    "search_mode": "hybrid",
    "limit": 5
  }'
```

That is enough to validate the core loop: write memory, search memory, inspect
results. No extra services are required for this smoke path.

## Full Cognitive Mode

For reflection, deduction, and answer generation, run the server with the
default config and point it at the models or API backends you want to use.

```bash
ollama serve
ollama pull qwen3.5:4b
cargo run -p anima-server -- config.default.toml
```

Notes:

- On first run, Anima downloads the local embedding model automatically.
- The default config uses Ollama for both consolidation and LLM reasoning.
- To use a remote backend instead, edit `config.default.toml` and point the
`[llm]` and `[processor]` sections to any OpenAI-compatible endpoint. Set
credentials via `PROCESSOR_API_KEY` or `OPENAI_API_KEY`.
- Environment variables can also be set in a `.env` file in the project root
(loaded automatically via dotenvy). See `.env.example` for available keys.

## Memory Categories

Every memory has a semantic `category` that controls how fast it decays and how
it's ranked in search. Set it on ingestion or let the reflection pipeline
classify automatically.

| Category      | Decay half-life | Use case                                           |
| ------------- | --------------- | -------------------------------------------------- |
| `identity`    | ~2,888 days     | Who the user is, names, relationships              |
| `preference`  | ~289 days       | Likes, dislikes, communication style               |
| `routine`     | ~96 days        | Recurring tasks, schedules, habits                 |
| `environment` | ~58 days        | Ports, paths, services, technical infra            |
| `inferred`    | ~29 days        | Conclusions Anima drew but user hasn't confirmed   |
| `general`     | ~29 days        | Default — uncategorized memories                   |
| `task`        | ~6 days         | Current work, temporary context                    |

```bash
# Set category on ingestion
curl -X POST "$BASE_URL/api/v1/memories" \
  -H "Content-Type: application/json" \
  -H "X-Anima-Namespace: default" \
  -d '{"content": "Gateway runs on port 18789", "category": "environment"}'

# Filter by category
curl "$BASE_URL/api/v1/memories?category=identity" \
  -H "X-Anima-Namespace: default"
```

When the reflection pipeline extracts facts from raw memories, it auto-classifies
each fact into the appropriate category. The MCP `memory_add` tool also accepts
`category` as an optional parameter.

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
| `POST /api/v1/memories/search`        | Hybrid search (results include `category`)                                |
| `GET /api/v1/memories?category=X`     | List memories filtered by category                                        |
| `GET /api/v1/memories/{id}/revisions` | Revision history                                                          |
| `GET /api/v1/audit/events`            | Audit trail                                                               |
| `POST /api/v1/ask`                    | Retrieval-augmented answer                                                |
| `POST /api/v1/chat`                   | Conversational interface                                                  |
| `GET /api/v1/processor/status`        | Background processor metrics                                              |


### MCP bridge

The MCP server wraps Anima as tools such as `memory_search`, `memory_add`,
`memory_update`, `memory_delete`, `memory_stats`, `memory_ask`, and
`memory_reflect`.

### Web UI

The web app gives you a visual surface for memories, search results, graph
relationships, embedding space, chat, and namespace-aware inspection. By
default it expects the Anima API at `http://localhost:3000`.

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


| Platform | Backend         | Details                                                                                   |
| -------- | --------------- | ----------------------------------------------------------------------------------------- |
| Linux    | systemd         | Unit at `/etc/systemd/system/anima-server.service`, logs via `journalctl -u anima-server` |
| macOS    | launchd         | Plist at `~/Library/LaunchAgents/dev.anima.server.plist`, starts on login                 |
| Windows  | NSSM / schtasks | Uses [NSSM](https://nssm.cc) if available, otherwise prints a `schtasks` command          |


The install command uses the current working directory and binary path, so run
it from the directory where your `config.toml` and `anima.db` live.

A pre-built systemd unit template is also available at
[service/anima-server.service](service/anima-server.service) for manual setup.

### Graceful Shutdown

Anima handles SIGINT (ctrl-c) and SIGTERM cleanly:

1. Stops accepting new HTTP connections
2. Finishes in-flight requests
3. Waits up to 30 seconds for background processor jobs to drain
4. Exits

This means `systemctl stop` and `docker stop` work without losing in-flight
work.

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