"""Closed, invocation-local authorization values for the shared MCP server."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

DENIAL_MESSAGE = "memory authorization denied"

_PRINCIPAL_FIELDS = frozenset(
    {
        "trust_domain",
        "issuer",
        "subject",
        "actor_id",
        "delegated_by",
        "tenant_id",
        "audience",
        "credential_id",
        "credential_binding_id",
        "grant_version",
    }
)
_GRANT_VERSION = re.compile(r"[A-Za-z0-9._:-]{1,64}\Z")


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    trust_domain: str
    issuer: str
    subject: str
    actor_id: str
    delegated_by: str | None
    tenant_id: str
    audience: str
    credential_id: str
    credential_binding_id: str
    grant_version: str


class Operation(str, Enum):
    SEARCH = "search"
    GET = "get"
    ADD = "add"


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    invocation_id: bytes
    operation: Operation
    service_tenant_id: str
    requested_scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    effect: Effect
    operation: Operation
    tenant_id: str
    exact_scope: str
    principal: AuthenticatedPrincipal
    request: AuthorizationRequest
    snapshot_id: str
    grant_version: str


@dataclass(frozen=True, slots=True)
class AuthorizationDecisionSet:
    principal: AuthenticatedPrincipal
    request: AuthorizationRequest
    snapshot_id: str
    grant_version: str
    decisions: tuple[CapabilityDecision, ...]


class GrantProvider(Protocol):
    async def authorize(
        self,
        principal: AuthenticatedPrincipal,
        request: AuthorizationRequest,
    ) -> AuthorizationDecisionSet: ...


def _deny() -> ValueError:
    return ValueError(DENIAL_MESSAGE)


def validate_closed_string(value: object) -> str:
    """Validate one contract-bounded, normalized Unicode configuration value."""
    if type(value) is not str or not value or unicodedata.normalize("NFC", value) != value:
        raise _deny()
    if len(value) > 256 or any(unicodedata.category(character) == "Cc" for character in value):
        raise _deny()
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _deny() from error
    if len(encoded) > 1024:
        raise _deny()
    return value


def principal_from_claims(
    claims: object,
    *,
    expected_issuer: str,
    expected_audience: str,
    expected_tenant_id: str,
) -> AuthenticatedPrincipal:
    """Convert only the verifier's exact normalized claims into a principal."""
    if type(claims) is not dict or any(type(key) is not str for key in claims) or set(claims) != _PRINCIPAL_FIELDS:
        raise _deny()

    values: dict[str, Any] = {}
    for field in _PRINCIPAL_FIELDS - {"delegated_by", "grant_version"}:
        values[field] = validate_closed_string(claims[field])

    delegated_by = claims["delegated_by"]
    if delegated_by is not None:
        delegated_by = validate_closed_string(delegated_by)
    values["delegated_by"] = delegated_by

    grant_version = claims["grant_version"]
    if type(grant_version) is not str or _GRANT_VERSION.fullmatch(grant_version) is None:
        raise _deny()
    values["grant_version"] = grant_version

    if (
        values["issuer"] != expected_issuer
        or values["audience"] != expected_audience
        or values["tenant_id"] != expected_tenant_id
    ):
        raise _deny()
    return AuthenticatedPrincipal(**values)


def validate_decision_set(
    value: object,
    *,
    principal: AuthenticatedPrincipal,
    request: AuthorizationRequest,
) -> AuthorizationDecisionSet:
    """Accept one complete, indivisible decision set or deny the invocation."""
    if type(value) is not AuthorizationDecisionSet:
        raise _deny()
    if type(value.principal) is not AuthenticatedPrincipal or type(value.request) is not AuthorizationRequest:
        raise _deny()
    if value.principal != principal or value.request != request:
        raise _deny()
    if type(value.snapshot_id) is not str or not value.snapshot_id or type(value.grant_version) is not str:
        raise _deny()
    if value.grant_version != principal.grant_version:
        raise _deny()
    if type(value.decisions) is not tuple or len(value.decisions) != len(request.requested_scopes):
        raise _deny()

    by_scope: dict[str, CapabilityDecision] = {}
    for decision in value.decisions:
        _validate_child(decision, principal, request, value.snapshot_id, value.grant_version)
        if decision.exact_scope in by_scope:
            raise _deny()
        by_scope[decision.exact_scope] = decision
    if set(by_scope) != set(request.requested_scopes):
        raise _deny()
    return value


def _validate_child(
    decision: object,
    principal: AuthenticatedPrincipal,
    request: AuthorizationRequest,
    snapshot_id: str,
    grant_version: str,
) -> None:
    if type(decision) is not CapabilityDecision or type(decision.exact_scope) is not str:
        raise _deny()
    if type(decision.principal) is not AuthenticatedPrincipal or type(decision.request) is not AuthorizationRequest:
        raise _deny()
    if type(decision.snapshot_id) is not str or type(decision.grant_version) is not str:
        raise _deny()
    if (
        decision.effect is not Effect.ALLOW
        or decision.principal != principal
        or decision.request != request
        or decision.operation is not request.operation
        or type(decision.tenant_id) is not str
        or decision.tenant_id != request.service_tenant_id
        or decision.snapshot_id != snapshot_id
        or decision.grant_version != grant_version
    ):
        raise _deny()
