from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from memory_mcp.authorization import (
    AuthenticatedPrincipal,
    AuthorizationDecisionSet,
    AuthorizationRequest,
    CapabilityDecision,
    Effect,
    Operation,
    principal_from_claims,
    validate_decision_set,
)

CLAIMS = {
    "trust_domain": "selamy.dev",
    "issuer": "https://issuer.example",
    "subject": "agent-1",
    "actor_id": "actor-1",
    "delegated_by": None,
    "tenant_id": "tenant-1",
    "audience": "memory-mcp",
    "credential_id": "credential-1",
    "credential_binding_id": "binding-1",
    "grant_version": "v1",
}


def test_principal_conversion_is_closed_frozen_and_binds_trusted_scalars():
    principal = principal_from_claims(
        CLAIMS,
        expected_issuer="https://issuer.example",
        expected_audience="memory-mcp",
        expected_tenant_id="tenant-1",
    )

    assert principal == AuthenticatedPrincipal(**CLAIMS)
    with pytest.raises(FrozenInstanceError):
        principal.actor_id = "swapped"  # type: ignore[misc]

    for changed in ("issuer", "audience", "tenant_id"):
        claims = {**CLAIMS, changed: "mismatch"}
        with pytest.raises(ValueError, match="memory authorization denied"):
            principal_from_claims(
                claims,
                expected_issuer="https://issuer.example",
                expected_audience="memory-mcp",
                expected_tenant_id="tenant-1",
            )


@pytest.mark.parametrize("field", sorted(CLAIMS))
@pytest.mark.parametrize(
    "mutation", [pytest.param(None, id="null"), pytest.param("", id="empty"), pytest.param(1, id="type")]
)
def test_every_claim_is_required_nonempty_and_exactly_typed(field, mutation):
    claims = {**CLAIMS, field: mutation}
    if field == "delegated_by" and mutation is None:
        claims.pop(field)
    with pytest.raises(ValueError, match="memory authorization denied"):
        principal_from_claims(
            claims,
            expected_issuer="https://issuer.example",
            expected_audience="memory-mcp",
            expected_tenant_id="tenant-1",
        )


@pytest.mark.parametrize("bad", ["e\u0301", "nul\0", "line\n", "x" * 257])
def test_claim_string_normalization_control_and_size_bounds(bad):
    with pytest.raises(ValueError, match="memory authorization denied"):
        principal_from_claims(
            {**CLAIMS, "actor_id": bad},
            expected_issuer="https://issuer.example",
            expected_audience="memory-mcp",
            expected_tenant_id="tenant-1",
        )


def test_non_null_delegation_and_exact_string_boundaries_are_accepted():
    principal = principal_from_claims(
        {**CLAIMS, "delegated_by": "delegator", "actor_id": "😀" * 256},
        expected_issuer="https://issuer.example",
        expected_audience="memory-mcp",
        expected_tenant_id="tenant-1",
    )
    assert principal.delegated_by == "delegator"
    assert len(principal.actor_id.encode("utf-8")) == 1024


@pytest.mark.parametrize("grant_version", ["has space", "é", "x" * 65, "v/1", "line\n"])
def test_grant_version_has_closed_printable_ascii_grammar(grant_version):
    with pytest.raises(ValueError, match="memory authorization denied"):
        principal_from_claims(
            {**CLAIMS, "grant_version": grant_version},
            expected_issuer="https://issuer.example",
            expected_audience="memory-mcp",
            expected_tenant_id="tenant-1",
        )


def _decision_values():
    principal = AuthenticatedPrincipal(**CLAIMS)
    request = AuthorizationRequest(
        invocation_id=b"i" * 16,
        operation=Operation.SEARCH,
        service_tenant_id="tenant-1",
        requested_scopes=("fleet", "infra"),
    )
    children = tuple(
        CapabilityDecision(
            effect=Effect.ALLOW,
            operation=request.operation,
            tenant_id=request.service_tenant_id,
            exact_scope=scope,
            principal=principal,
            request=request,
            snapshot_id="snapshot-1",
            grant_version="v1",
        )
        for scope in request.requested_scopes
    )
    decision_set = AuthorizationDecisionSet(
        principal=principal,
        request=request,
        snapshot_id="snapshot-1",
        grant_version="v1",
        decisions=children,
    )
    return principal, request, decision_set


def test_complete_decision_set_requires_exact_children_and_deny_overrides():
    principal, request, decision_set = _decision_values()
    assert validate_decision_set(decision_set, principal=principal, request=request) is decision_set
    with pytest.raises(FrozenInstanceError):
        request.operation = Operation.GET  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision_set.snapshot_id = "swapped"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision_set.decisions[0].effect = Effect.DENY  # type: ignore[misc]

    first, second = decision_set.decisions
    malformed = [
        None,
        replace(decision_set, principal=replace(principal, actor_id="other")),
        replace(decision_set, request=replace(request, invocation_id=b"x" * 16)),
        replace(decision_set, snapshot_id="other"),
        replace(decision_set, grant_version="v2"),
        replace(decision_set, decisions=(first,)),
        replace(decision_set, decisions=(first, first)),
        replace(decision_set, decisions=(first, second, second)),
        replace(decision_set, decisions=(first, replace(second, effect=Effect.DENY))),
    ]
    for candidate in malformed:
        with pytest.raises(ValueError, match="memory authorization denied"):
            validate_decision_set(candidate, principal=principal, request=request)


def test_every_child_principal_request_and_binding_field_is_revalidated():
    principal, request, decision_set = _decision_values()
    first, second = decision_set.decisions
    principal_mutations = [
        replace(principal, **{field: ("delegate" if field == "delegated_by" else "swapped")}) for field in CLAIMS
    ]
    request_mutations = [
        replace(request, invocation_id=b"z" * 16),
        replace(request, operation=Operation.GET),
        replace(request, service_tenant_id="other"),
        replace(request, requested_scopes=("fleet",)),
    ]
    child_mutations = [
        replace(first, effect=Effect.DENY),
        replace(first, operation=Operation.GET),
        replace(first, tenant_id="other"),
        replace(first, exact_scope="other"),
        replace(first, snapshot_id="other"),
        replace(first, grant_version="v2"),
        replace(first, effect="allow"),  # type: ignore[arg-type]
    ]
    child_mutations.extend(replace(first, principal=changed) for changed in principal_mutations)
    child_mutations.extend(replace(first, request=changed) for changed in request_mutations)

    for changed in principal_mutations:
        with pytest.raises(ValueError, match="memory authorization denied"):
            validate_decision_set(replace(decision_set, principal=changed), principal=principal, request=request)
    for changed in request_mutations:
        with pytest.raises(ValueError, match="memory authorization denied"):
            validate_decision_set(replace(decision_set, request=changed), principal=principal, request=request)
    for changed in child_mutations:
        candidate = replace(decision_set, decisions=(changed, second))
        with pytest.raises(ValueError, match="memory authorization denied"):
            validate_decision_set(candidate, principal=principal, request=request)


def test_decision_set_rejects_non_tuple_and_real_generator_is_only_bounded_by_type_and_length():
    from memory_mcp.authorized_server import generate_invocation_id

    principal, request, decision_set = _decision_values()
    with pytest.raises(ValueError, match="memory authorization denied"):
        validate_decision_set(
            replace(decision_set, decisions=list(decision_set.decisions)),  # type: ignore[arg-type]
            principal=principal,
            request=request,
        )
    draws = [generate_invocation_id() for _ in range(4)]
    assert all(type(value) is bytes and len(value) == 16 for value in draws)

    with pytest.raises(ValueError, match="memory authorization denied"):
        principal_from_claims(
            {**CLAIMS, "unexpected": "value"},
            expected_issuer="https://issuer.example",
            expected_audience="memory-mcp",
            expected_tenant_id="tenant-1",
        )
