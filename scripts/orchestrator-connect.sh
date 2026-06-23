#!/usr/bin/env bash
# Open a tunnel from this (off-cluster) box to the in-cluster shared semantic
# memory service, so an MCP client here can reach it at http://127.0.0.1:<port>/mcp.
#
# The service has NO external surface (ClusterIP + NetworkPolicy). The orchestrator
# runs off-cluster and reaches the cluster only through a jump host that has
# `kubectl` credentials (here: the SSH alias `pselamy`). This script therefore
# nests two hops:
#
#   dev box  --(ssh -L LOCAL_PORT:127.0.0.1:REMOTE_PORT)-->  jump host
#   jump host  --(kubectl -n memory port-forward REMOTE_PORT:8080)-->  Service
#
# IMPORTANT (the gotcha this script encodes): the SSH `-L` forward binds
# REMOTE_PORT on the jump host as the tunnel's far end, so `kubectl port-forward`
# MUST bind a DIFFERENT port on the jump host. We use LOCAL_PORT on this box and
# REMOTE_PORT (= LOCAL_PORT+1 by default) on the jump host.
#
# Usage:
#   scripts/orchestrator-connect.sh                 # foreground; Ctrl-C to stop
#   JUMP_HOST=pselamy LOCAL_PORT=18080 scripts/orchestrator-connect.sh
#
# Then point an MCP client (e.g. ~/.claude.json) at:
#   { "type": "http", "url": "http://127.0.0.1:${LOCAL_PORT}/mcp" }
#
# This is a connection helper, NOT a deployment step. It reads no secrets.
set -euo pipefail

JUMP_HOST="${JUMP_HOST:-pselamy}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
REMOTE_PORT="${REMOTE_PORT:-$((LOCAL_PORT + 1))}"
NAMESPACE="${MEMORY_NAMESPACE:-memory}"
SERVICE="${MEMORY_SERVICE:-memory-mcp}"
KUBE_CONTEXT="${KUBE_CONTEXT:-gke_patrick-agents-prod_us-central1-a_selamy-agents-prod}"

echo "[orchestrator-connect] dev:${LOCAL_PORT} -> ${JUMP_HOST}:${REMOTE_PORT} -> ${SERVICE}.${NAMESPACE}.svc:8080" >&2
echo "[orchestrator-connect] MCP url once up:  http://127.0.0.1:${LOCAL_PORT}/mcp" >&2

# A single SSH session: establish the local forward, then run kubectl on the jump
# host bound to REMOTE_PORT. Cleaning up the kubectl child on SSH exit keeps the
# jump host tidy across reconnects.
exec ssh -o ExitOnForwardFailure=yes \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${JUMP_HOST}" \
  "kubectl config use-context '${KUBE_CONTEXT}' >/dev/null 2>&1; \
   trap 'kill %1 2>/dev/null || true' EXIT; \
   kubectl -n '${NAMESPACE}' port-forward 'svc/${SERVICE}' '${REMOTE_PORT}:8080'"
