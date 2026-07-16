# Image publication gate

Image construction in this repository is validation-only. Pull requests,
pushes to `main`, and manual dispatches build the Dockerfile with `push: false`;
the workflow has read-only permissions and no registry authentication or image
tag generation.

## Prior publication discrepancy

PR [#33](https://github.com/selamy-labs/memory-mcp/pull/33) merged as source
commit `1403877fa8c2de8a728300386777cf2f95722070`. The pre-existing push workflow
then authenticated to GHCR and [run
29471199264](https://github.com/selamy-labs/memory-mcp/actions/runs/29471199264)
published digest
`sha256:a5ed34acbf6b82e92079470b9c9aa32ad2a193db9f15cfca23e26bac7bce8aed`
under both `sha-1403877fa8c2de8a728300386777cf2f95722070` and mutable tag
`latest`. That publication contradicted the source-only boundary recorded for
issue #32.

The publication did not deploy or promote the image to E5. The GitOps
Application and E5 bootstrap values remained pinned to the older explicit tag
`sha-ba1bb55faddad2d0acfd542ccd526913dfd0d251`. This is non-deployment
evidence only: issue [#34](https://github.com/selamy-labs/memory-mcp/issues/34)
does not delete, retag, or otherwise remediate the already published package.

## Publication ownership

Publication may return only through issue #32 follow-up E after independent
review of an immutable-image contract covering the exact source commit,
dependency lock, base-image digest, SBOM and provenance, signing, registry
custody, immutable digest/tag policy, and proof that no mutable reference is
deployed. Production rollout remains owned by follow-ups F/G and is outside
this gate.
