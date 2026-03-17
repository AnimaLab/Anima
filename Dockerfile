# Multi-stage build for anima-server
# Usage: docker build -t anima-server .

FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin anima-server

# Runtime image — minimal
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/anima-server /usr/local/bin/

WORKDIR /app

EXPOSE 3000

ENTRYPOINT ["anima-server"]
