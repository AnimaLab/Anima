# Contributing

Thanks for your interest in Anima.

## Development Setup

- Rust stable toolchain
- Node.js 22+ for `web/` and `mcp/`
- Python 3.11+ for benchmark harnesses
- Optional local Ollama instance for full cognitive-mode testing

## Common Checks

```bash
cargo test --workspace

cd web && npm install && npm run build
cd mcp && npm install && npm run build
```

## Pull Request Expectations

- Keep changes focused and easy to review.
- Include tests or a short verification note for behavior changes.
- Update docs when user-facing behavior or setup changes.
- Do not commit generated artifacts, local databases, model downloads, benchmark
dumps, or machine-specific config.

## Benchmarks And Data

- Benchmark harnesses are public.
- Raw benchmark outputs and fine-tuning corpora are intentionally excluded from
version control.
- Follow `DATASETS.md` for dataset provenance and download expectations.
