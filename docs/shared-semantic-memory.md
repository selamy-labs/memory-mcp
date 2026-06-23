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
in-cluster MCP client (Claude Code) declares it as a remote HTTP MCP server:

```json
{
  "mcpServers": {
    "fleet-memory": {
      "type": "http",
      "url": "http://memory-mcp.memory.svc:8080/mcp"
    }
  }
}
```

Use a **distinct name** (`fleet-memory`) so it never collides with the local
stdio `memory` server. The agent's namespace must be in the chart's
`networkPolicy.allowedNamespaces`.

### Orchestrator (off-cluster) — tunnel, never a public endpoint

The orchestrator runs outside the cluster and reaches it only through a jump host
that has `kubectl` credentials. Two hops are needed, and there is a port gotcha:
the SSH `-L` forward binds its far-end port on the jump host, so `kubectl
port-forward` must bind a *different* port there. The helper script encodes this:

```bash
# Opens dev:18080 -> jump:18081 -> memory-mcp.memory.svc:8080 (foreground; Ctrl-C to stop).
JUMP_HOST=pselamy LOCAL_PORT=18080 scripts/orchestrator-connect.sh
```

Then point the orchestrator's MCP client (e.g. `~/.claude.json`) at the local end:

```json
{
  "mcpServers": {
    "fleet-memory": {
      "type": "http",
      "url": "http://127.0.0.1:18080/mcp"
    }
  }
}
```

A scoped, longer-lived path (an authenticated reverse proxy) can replace the
tunnel later; Phase-1 deliberately keeps the surface minimal. The script reads no
secrets — it only forwards a port.

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

## Embedder: $0 default, with a true-semantic upgrade path

The default embedder is `hashing` — **$0, no API key, fully self-hosted**. It
makes the semantic plumbing real (shared vocabulary pulls vectors together) but
is not a learned model, so recall is "good enough", not state-of-the-art.

To upgrade to true semantic recall, set `server.embedder=openai` (the adapter
speaks any OpenAI-compatible `/embeddings` API) and choose a backend:

1. **Self-hosted local model — RECOMMENDED, $0.** Run an in-cluster embedding
   server that exposes an OpenAI-compatible endpoint (e.g. HuggingFace
   [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
   serving a small model like `BAAI/bge-small-en-v1.5`, 384-dim) and point the
   server at it — no API key, no per-call cost, no data leaving the cluster:

   ```yaml
   server:
     embedder: openai
     embedding:
       baseUrl: "http://text-embeddings.memory.svc:80/v1"
       model: "BAAI/bge-small-en-v1.5"
       dim: 384
   ```

   A no-auth local endpoint needs no key (the adapter omits the `Authorization`
   header when none is set).

2. **Paid hosted provider (e.g. OpenAI `text-embedding-3-small`) — COSTS MONEY.**
   This incurs per-token embedding charges and sends memory text to a third
   party. **Do not enable without an explicit cost decision from the owner.**
   When chosen, the key is provided via `externalSecret.embeddingKey` (GSM), never
   image-baked.

> **Switching the embedder requires a reindex.** The embedding dimension is fixed
> per pgvector column, and different models produce different-dimension,
> non-comparable vectors. To switch: update `embedder`/`embedding.dim`, drop +
> recreate the table (`MEMORY_ENSURE_SCHEMA=1` creates it; an existing table must
> be dropped first), then run the reindex CronJob to repopulate from git.
