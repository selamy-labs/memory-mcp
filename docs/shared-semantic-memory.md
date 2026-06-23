# Shared fleet semantic memory (in-cluster service)

This repo ships **two** things:

1. **`memory-mcp`** — the local, stdio, markdown-file read/write server (run via
   `uvx` against a local `MEMORY_ROOT`). See the top-level [README](../README.md).
2. **`memory-mcp-server`** — the **networked shared semantic memory**: a
   long-running service over a Postgres + pgvector index that the whole fleet and
   the orchestrator read/write. This doc covers (2).

It is Phase-1 of the shared-fleet-memory design: replace siloed per-agent
markdown memory with ONE shared store, namespaced by `group_id` (scope).
Markdown-in-git stays the **source of truth**; this store is a **rebuildable
index** over it (drop the table and re-run the indexer to rebuild from git).

## Tools

| Tool | Purpose |
| --- | --- |
| `add_memory(name, description, type, body, group_id?, updated_at?)` | Index one memory into a scope (default `fleet`). Idempotent on `(group_id, name)`. `updated_at` is an ISO date used as the recency anchor. |
| `search_memory(query, group_ids?, include_fleet?, type?, limit?)` | Blended **semantic + recency + keyword** search across one or more scopes (always including `fleet` unless `include_fleet=false`). |
| `get_memory(name, group_id?)` | One indexed memory's verbatim record. |

### Scopes (`group_id`)

`fleet` is the shared common ground every agent and the orchestrator read/write.
Per-domain scopes (e.g. `trading`, `infra`, `matchpoint`, `career`) isolate
domain knowledge; a `search_memory` over a domain scope still sees `fleet` by
default, so an agent gets both the common ground and its domain.

## How clients connect

The service has **no external surface**: a `ClusterIP` Service plus a
`NetworkPolicy` that admits only the agent namespaces. There is **no Ingress**.

### Agents (in-cluster) — cluster DNS

The server listens on `memory-mcp.memory.svc:8080` over MCP streamable-http. An
in-cluster MCP client points at:

```
http://memory-mcp.memory.svc:8080/mcp
```

The agent's namespace must be in the chart's `networkPolicy.allowedNamespaces`.

### Orchestrator (off-cluster) — port-forward, never a public endpoint

The orchestrator runs outside the cluster, so it reaches the service through a
`kubectl port-forward`, not a public URL:

```bash
# 1. Get cluster credentials (run where gcloud is authed, e.g. ssh pselamy).
gcloud container clusters get-credentials selamy-agents-prod \
  --zone us-central1-a --project patrick-agents-prod

# 2. Forward the service to localhost.
kubectl -n memory port-forward svc/memory-mcp 8080:8080

# 3. Point the MCP client at the forwarded port.
#    http://127.0.0.1:8080/mcp
```

A scoped, longer-lived path (e.g. an authenticated reverse proxy) can replace the
port-forward later; Phase-1 deliberately keeps the surface minimal.

## Configuration (environment, resolved at start)

| Variable | Effect |
| --- | --- |
| `MEMORY_BACKEND` | `pgvector` (default in-cluster) or `memory` (in-RAM smoke test). |
| `MEMORY_PG_HOST` / `MEMORY_PG_PORT` / `MEMORY_PG_USER` / `MEMORY_PG_DB` / `MEMORY_PG_PASSWORD` | Discrete Postgres connection params (preferred). The chart sets these; the password is mounted from the synced Secret. Discrete params are used (not a URL DSN) so a base64/special-char password can never be misparsed. |
| `MEMORY_PG_DSN` | Fallback URL DSN for `pgvector` when `MEMORY_PG_HOST` is unset. |
| `MEMORY_EMBEDDER` | `hashing` (default; zero-cost, no API key, self-hosted) or `openai`. |
| `MEMORY_EMBEDDING_*` | `openai` embedder config (key/base-url/model/dim), read at call time. |
| `MEMORY_ENSURE_SCHEMA` | `1` to create the pgvector schema on start (idempotent). |
| `MCP_TRANSPORT` / `MCP_HOST` / `MCP_PORT` | `streamable-http` on `0.0.0.0:8080` by default. |

**No credentials are baked into the image.** The Postgres password and any
embedding API key are provided at runtime from a Kubernetes Secret synced by
External Secrets from Google Secret Manager.

## Rebuilding the index from git

The index is rebuildable, so PVC loss is "reindex", not "data loss". The reindex
CronJob (and a manual run) executes the indexer:

```bash
MEMORY_SOURCES="/path/to/memory:fleet,/path/to/architecture:infra" \
MEMORY_BACKEND=pgvector MEMORY_PG_DSN=... \
  python -m memory_mcp
```

Recency is seeded from each file's own date (git last-commit date → a date in the
filename → file mtime), so importing the existing corpus does not make old
memories look brand-new.
