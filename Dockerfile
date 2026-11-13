# Multi-stage build for anima-server
# Usage: docker build -t anima-server .

FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin anima-server

# Build web UI
FROM node:22-slim AS web-builder

WORKDIR /app
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ .
RUN npm run build

# Runtime image — minimal
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/anima-server /usr/local/bin/

WORKDIR /app

COPY config.default.toml ./config.default.toml
# BGE-M3 sparse head: never downloaded, read from [sparse] weights_path.
COPY models/bge-m3/sparse_linear.bin ./models/bge-m3/sparse_linear.bin
COPY --from=web-builder /app/dist/ ./web/dist/
RUN sed -i 's/host = "127.0.0.1"/host = "0.0.0.0"/' config.default.toml && \
    sed -i 's|http://localhost:|http://host.docker.internal:|g' config.default.toml

# The container binds 0.0.0.0, so the server requires ANIMA_API_KEY at runtime:
#   docker run -e ANIMA_API_KEY=... -p 3000:3000 anima-server
EXPOSE 3000

ENTRYPOINT ["anima-server"]
