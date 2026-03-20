# Anima

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-LoCoMo_87.7%25-green.svg)](https://github.com/AnimaLab/Anima-Benchmarks)

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
**87.73% on LOCOMO** — the academic benchmark for long-term conversational memory. Human baseline: 87.9%. Gap: 0.17%.

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
| Revisions and audit trail | Shipped | Revision history, rollback, and audit events               |
| Namespace isolation       | Shipped | `X-Anima-Namespace` plus namespace ACL support             |
| Background consolidation  | Shipped | Reflection and deduction pipeline                          |
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

## What Makes Anima Different


| Area               | Anima                                                                      |
| ------------------ | -------------------------------------------------------------------------- |
| Storage model      | Local-first SQLite database you can move, back up, and inspect             |
| Retrieval          | Hybrid search with vector, keyword, and temporal signals                   |
| Inspection         | Revision history, rollback, audit trail, graph view, embedding view        |
| Interfaces         | REST API, MCP bridge, and web UI in one repo                               |
| Deployment posture | Start locally, then decide how much reasoning you want on-device or remote |


## Interfaces

### REST API

The HTTP server exposes health, memory CRUD, search, revisions, audit events,
chat, ask, graph, embeddings, planning, and conversation endpoints.

Key routes include:

- `GET /health`
- `POST /api/v1/memories`
- `POST /api/v1/memories/search`
- `GET /api/v1/memories/{id}/revisions`
- `GET /api/v1/audit/events`
- `POST /api/v1/ask`
- `POST /api/v1/chat`

### MCP bridge

The MCP server wraps Anima as tools such as `memory_search`, `memory_add`,
`memory_update`, `memory_delete`, `memory_stats`, `memory_ask`, and
`memory_reflect`.

```bash
cd mcp
npm install
npm run build
```

### Web UI

The web app gives you a visual surface for memories, search results, graph
relationships, embedding space, chat, and namespace-aware inspection.

```bash
cd web
npm install
npm run dev
```

By default, the UI expects the Anima API at `http://localhost:3000`.

## Benchmarks

Anima's public benchmark harnesses and curated reports live in
[anima-benches](https://github.com/AnimaLab/Anima-Benchmarks). This repo keeps
the benchmark code under [`benchmarks/`](benchmarks) for local development, but
the public benchmark surface and headline reporting are maintained in the
dedicated benchmark repo.

Current headline public LoCoMo results:

- **93.4%** on `conv-26` only, 152 questions, single-conversation focused run
- **87.7%** best published full-dataset LoCoMo accuracy across all 10 conversations, 1,540 questions
- **86.0%** curated March full-dataset LoCoMo report across all 10 conversations
- **0.864** best published broad continuous judge score

Dataset guidance lives in [DATASETS.md](DATASETS.md). See also
[CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR expectations.

## Repository Map

- `[crates/](crates)`: Rust workspace for core memory types, database,
embeddings, consolidation, and server code
- `[mcp/](mcp)`: TypeScript MCP server for tool-based integrations
- `[web/](web)`: React frontend for memory browsing and inspection
- `[scripts/](scripts)`: local startup helpers
- `[benchmarks/](benchmarks)`: benchmark harnesses and curated benchmark notes

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

## Telemetry

Anima collects anonymous usage analytics to help improve the product. This
includes model names (not keys), memory counts (not content), OS and
architecture info, and feature flag settings. No personal data, namespace
names, or memory content is ever transmitted.

Telemetry is enabled by default. To opt out, set `enabled = false` in
`config.toml`:

```toml
[telemetry]
enabled = false
```

Or toggle it off in the Settings page of the web UI.

## Contact

For partnerships, pilots, or product questions: [contact@runanima.com](mailto:contact@runanima.com)

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
