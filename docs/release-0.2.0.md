# Anima v0.2.0 — Operational Hardening

## Graceful Shutdown

Anima now handles SIGINT (ctrl-c) and SIGTERM cleanly on all platforms. On shutdown it stops accepting new connections, finishes in-flight HTTP requests, and waits up to 30 seconds for background processor jobs to drain before exiting. `systemctl stop`, `docker stop`, and manual ctrl-c all work without losing in-flight work.

## Health Endpoints

- **`GET /health`** — liveness probe, always returns 200 if the process is alive
- **`GET /health/ready`** — readiness probe that checks database connectivity, embedding model health, and background processor state. Returns 503 with per-component detail when degraded:

```json
{
  "status": "ready",
  "checks": {
    "db": "ok",
    "embedder": "ok",
    "processor": { "status": "running", "queue_depth": 0, "in_flight": 0 }
  }
}
```

## Startup Self-Test

The server now validates all dependencies before accepting traffic:

- **Database**: opens and pings with `SELECT 1`
- **Embedding model**: loads and runs a smoke-test embedding
- **LLM**: logs configuration state (warns if unreachable, does not block startup)

Failures are logged with `domain=` tags (`db`, `embedding`, `llm`) and the process exits immediately with a clear error instead of silently serving with broken dependencies.

## Structured Error Logging

All API error responses now log with a `domain` field for easy filtering:

| Domain | Errors |
|--------|--------|
| `db` | Database failures |
| `embedding` | Embedding model errors |
| `llm` | LLM connectivity / response issues |
| `auth` | ACL / permission denials |
| `retrieval` | Memory not found |
| `request` | Bad request / validation |

500s log at `error`, 403s at `warn`, other 4xx at `debug`.

## Cross-Platform Service Install

Install Anima as an auto-start system service from the command line:

```bash
anima-server --install        # auto-detects platform
anima-server --service-status # check if running
anima-server --uninstall      # remove the service
```

| Platform | Backend | What it does |
|----------|---------|--------------|
| Linux | systemd | Writes unit to `/etc/systemd/system/`, enables and starts |
| macOS | launchd | Writes plist to `~/Library/LaunchAgents/`, loads agent |
| Windows | NSSM / schtasks | Uses NSSM if available, otherwise prints schtasks command |

Pre-built service templates are also shipped in `service/` for manual setup.

## Windows Release Builds

Release artifacts now include a Windows build (`x86_64-pc-windows-msvc`) packaged as `.zip`. All archives now include the `service/` templates.

## Telemetry Opt-Out Notice

Startup now prints a clear message explaining what telemetry collects and how to opt out, matching the disclosure pattern used by Next.js and similar tools.

## Other Changes

- README restructured: removed duplicate sections, added Deployment docs (Docker, service install, graceful shutdown, health checks)
- All crates bumped from 0.1.0 to 0.2.0
