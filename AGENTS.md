# AGENTS.md

Guidance for coding agents working in this repository. `CONTRIBUTING.md` holds the setup and the common checks.

## Keep the docs in step with every PR

Each pull request updates the documentation it affects, in the same PR. The docs are the Mintlify site in `docs/` (navigation in `docs/docs.json`) and `README.md`.

- **New or changed endpoint** (`crates/anima-server/src/app.rs`, `handlers.rs`, `dto.rs`): add or update its page under `docs/api-reference/`. Its fields, defaults, limits, status codes and JSON examples follow the request and response structs. A new page gets an entry in `docs/docs.json`.
- **New or changed config key or environment variable** (`config.rs`, `config.default.toml`): update `docs/configuration.mdx`.
- **CLI, install or release change** (`cli/`, `.github/workflows/release.yml`): update `docs/quickstart.mdx`, `docs/integrations/connect.mdx` and `README.md`.
- **MCP tool change** (`mcp/src`, `crates/anima-server/src/cli/mcp.rs`): update `docs/integrations/mcp.mdx`.
- **Web UI page added or removed** (`web/src/App.tsx`): update `docs/integrations/web-ui.mdx`.
- **Retrieval, namespaces, memory types, briefing or skills behaviour**: update the matching page under `docs/concepts/`.
- **Security-relevant change** (authentication, CORS, ACL, Host check): update `docs/security.mdx`.

Before opening the PR:
- Run `npx mint broken-links` in `docs/`.
- Preview the touched pages with `npx mint dev` in `docs/`.
- State in the PR description which docs pages changed. When no doc applies, say so in one line.
