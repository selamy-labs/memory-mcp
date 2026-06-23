# Shared semantic memory MCP server (networked, pgvector backend).
#
# Builds a slim image that runs `memory-mcp-server` over streamable-http for the
# in-cluster service, and can also run `python -m memory_mcp` for the reindex
# CronJob. Credentials are NEVER baked in: the Postgres DSN and any embedding API
# key are provided at runtime via env from a mounted Secret (ExternalSecrets <-
# Google Secret Manager). The image carries `git` so the reindex job can read a
# checked-out markdown source tree's commit dates.
FROM python:3.12-slim AS base

# git: used by the indexer's LocalSourceReader to read per-file commit dates.
# libpq is bundled by psycopg[binary], so no system libpq is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package with the mcp + pg + otel extras. Copy the build inputs
# first so the layer caches on source changes only.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[mcp,pg,otel]"

# Run as a non-root user (least privilege; the service needs no write access).
RUN useradd --create-home --uid 10001 memory
USER memory

# Default service config: networked transport, bind all interfaces, pgvector.
ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8080 \
    MEMORY_BACKEND=pgvector \
    MEMORY_EMBEDDER=hashing

EXPOSE 8080

# The Deployment runs the server; the reindex CronJob overrides command to
# `python -m memory_mcp` with MEMORY_SOURCES set.
ENTRYPOINT ["memory-mcp-server"]
