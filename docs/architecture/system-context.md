# System context

`memory-mcp` exposes memory through two runtime modes. The local stdio server
reads and carefully updates a markdown memory root. The shared in-cluster HTTP
server reads and writes a rebuildable Postgres/pgvector index populated from
markdown-in-git sources. The modes share Python domain logic but use different
storage backends.

```mermaid
flowchart LR
    localClient[Local MCP client]
    clusterClient[In-cluster agent MCP client]
    orchestrator[Off-cluster orchestrator]
    memoryMcp[memory-mcp]
    markdown[Local markdown memory root]
    gitSources[GitHub markdown-memory repositories]
    pgvector[Postgres with pgvector]
    embedder[Optional OpenAI-compatible embedder]
    secrets[External Secrets and Kubernetes Secrets]

    localClient -->|MCP over stdio: search, read, and careful write| memoryMcp
    clusterClient -->|MCP over private HTTP: add, search, and get| memoryMcp
    orchestrator -->|MCP over kubectl and SSH port forwarding| memoryMcp
    memoryMcp -->|Local mode: read and update markdown plus MEMORY.md| markdown
    gitSources -->|Reindex job clones and parses source memories| memoryMcp
    memoryMcp -->|Shared mode: upsert and query the rebuildable index| pgvector
    memoryMcp -->|Optional shared mode: request embeddings| embedder
    secrets -->|Supply database and optional embedding credentials| memoryMcp
```

The shared service has no public ingress. Its Helm chart restricts access with
Kubernetes `NetworkPolicy`; off-cluster access uses the documented private
port-forward path. The default embedder is the in-process hashing implementation,
so the external embedding edge exists only when the optional OpenAI-compatible
adapter is configured.

Sources of truth: [`src/memory_mcp/`](../../src/memory_mcp/),
[`templates/`](../../templates/), [`values.yaml`](../../values.yaml), and
[`docs/shared-semantic-memory.md`](../shared-semantic-memory.md).
