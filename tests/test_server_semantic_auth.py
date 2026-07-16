from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from dataclasses import replace
from datetime import datetime, timezone

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools.base import Tool
from mcp.shared.context import RequestContext
from pydantic import AnyHttpUrl
from starlette.requests import Request

import memory_mcp.authorized_server as authorized_server
from memory_mcp.authorization import AuthorizationDecisionSet, CapabilityDecision, Effect
from memory_mcp.authorized_server import ApplicationJSONAdapter
from memory_mcp.embeddings import HashingEmbedder
from memory_mcp.pgvector_store import PgVectorStore
from memory_mcp.semantic import SemanticMemory
from memory_mcp.server_semantic import AuthorizedMemoryMCP, build_authorized_server
from memory_mcp.vector_store import InMemoryVectorStore
from tests.test_pgvector_store import FakeConnection

ISSUER = "https://issuer.example/"
RESOURCE = "https://memory.example/mcp"
AUDIENCE = "memory-mcp"
TENANT = "tenant-1"

BOUNDARY_VERSIONS = {
    "mcp": "1.28.1",
    "starlette": "1.3.1",
    "pydantic": "2.13.4",
    "anyio": "4.14.2",
    "httpx": "0.28.1",
    "sse-starlette": "3.4.5",
    "pydantic-settings": "2.14.2",
    "uvicorn": "0.51.0",
    "python-multipart": "0.0.32",
    "authlib": "1.6.9",
    "cryptography": "49.0.0",
}

PROVENANCE_SCALAR_FIELDS = (
    "source_repo",
    "source_path",
    "source_commit",
    "source_url",
    "author",
    "session_id",
    "issue",
    "evidence",
    "privacy",
    "retention",
)
PROVENANCE_LIST_FIELDS = ("supersedes", "superseded_by")


def claims(actor_id="actor-1"):
    return {
        "trust_domain": "selamy.dev",
        "issuer": ISSUER,
        "subject": actor_id,
        "actor_id": actor_id,
        "delegated_by": None,
        "tenant_id": TENANT,
        "audience": AUDIENCE,
        "credential_id": f"credential-{actor_id}",
        "credential_binding_id": f"binding-{actor_id}",
        "grant_version": "v1",
    }


class Verifier:
    def __init__(self):
        self.calls = []

    async def verify_token(self, token: str):
        self.calls.append(token)
        if token == "invalid":
            return None
        return AccessToken(token=token, client_id="client", scopes=[], resource=RESOURCE, claims=claims(token))


class Provider:
    def __init__(self):
        self.calls = []

    async def authorize(self, principal, request):
        self.calls.append((principal, request))
        children = tuple(
            CapabilityDecision(
                effect=Effect.ALLOW,
                operation=request.operation,
                tenant_id=request.service_tenant_id,
                exact_scope=scope,
                principal=principal,
                request=request,
                snapshot_id="snapshot-1",
                grant_version=principal.grant_version,
            )
            for scope in request.requested_scopes
        )
        return AuthorizationDecisionSet(
            principal=principal,
            request=request,
            snapshot_id="snapshot-1",
            grant_version=principal.grant_version,
            decisions=children,
        )


def settings():
    return AuthSettings(issuer_url=AnyHttpUrl(ISSUER), resource_server_url=AnyHttpUrl(RESOURCE), required_scopes=[])


def build(**overrides):
    values = {
        "token_verifier": Verifier(),
        "auth_settings": settings(),
        "expected_issuer": ISSUER,
        "service_resource_server": RESOURCE,
        "service_audience": AUDIENCE,
        "service_tenant_id": TENANT,
        "application_body_limit_bytes": 32_768,
        "grant_provider": Provider(),
        "memory_factory": lambda: pytest.fail("memory factory must not run"),
        "invocation_id_factory": lambda: b"i" * 16,
    }
    values.update(overrides)
    return build_authorized_server(**values)


def memory():
    return SemanticMemory(HashingEmbedder(dim=64), InMemoryVectorStore())


def client_factory(app):
    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    return factory


def current_context(actor_id="actor-1", *, token_claims=None, resource=RESOURCE):
    access_token = AccessToken(
        token=actor_id,
        client_id="client",
        scopes=[],
        resource=resource,
        claims=claims(actor_id) if token_claims is None else token_claims,
    )
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "scheme": "https",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("memory.example", 443),
            "user": AuthenticatedUser(access_token),
        }
    )
    request_context = RequestContext(
        request_id=1,
        meta=None,
        session=object(),
        lifespan_context=None,
        request=request,
    )
    return Context(request_context=request_context)


async def registered_call(server, name, arguments, *, context):
    """Exercise the private registered Tool as FastMCP does; not a wrapper surface."""
    return await server._tools[name].run(arguments, context=context)


def test_factory_builds_only_the_closed_authenticated_wrapper_and_exact_schemas():
    server = build()
    assert type(server) is AuthorizedMemoryMCP
    assert server.app is not None
    assert not any(hasattr(server, name) for name in ("settings", "fastmcp", "add_tool", "streamable_http_app", "run"))

    schemas = {tool["name"]: tool["inputSchema"] for tool in server.list_tools()}
    assert set(schemas) == {"add_memory", "get_memory", "search_memory"}
    assert set(schemas["add_memory"]["properties"]) == {
        "name",
        "description",
        "type",
        "body",
        "group_id",
        "updated_at",
        "provenance",
    }
    assert schemas["add_memory"]["required"] == ["name", "description", "type", "body", "group_id"]
    assert schemas["get_memory"]["required"] == ["name", "group_id"]
    assert schemas["search_memory"]["required"] == ["query", "group_ids"]
    assert all(schema["additionalProperties"] is False for schema in schemas.values())
    assert all(
        "context" not in schema["properties"] and "ctx" not in schema["properties"] for schema in schemas.values()
    )
    provenance = schemas["add_memory"]["properties"]["provenance"]
    assert provenance == {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    **{name: {"anyOf": [{"type": "string"}, {"type": "null"}]} for name in PROVENANCE_SCALAR_FIELDS},
                    **{name: {"type": "array", "items": {"type": "string"}} for name in PROVENANCE_LIST_FIELDS},
                },
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }


@pytest.mark.asyncio
async def test_pin_level_composition_initializes_once_and_has_no_exported_bypass(monkeypatch):
    calls = 0
    original = FastMCP.streamable_http_app

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(FastMCP, "streamable_http_app", counted)
    server = build()
    assert {package: importlib.metadata.version(package) for package in BOUNDARY_VERSIONS} == BOUNDARY_VERSIONS
    assert calls == 1
    assert {name for name in dir(AuthorizedMemoryMCP) if not name.startswith("_")} == {
        "app",
        "call_tool",
        "list_tools",
    }
    mcp_route = next(route for route in server.app.routes if getattr(route, "path", None) == "/mcp")
    assert type(mcp_route.endpoint.app) is ApplicationJSONAdapter

    async def permissive(name: str):
        return name

    default_tool = Tool.from_function(permissive)
    assert default_tool.parameters.get("additionalProperties") is not False
    assert await default_tool.run({"name": "n", "forged": "ignored"}) == "n"


@pytest.mark.asyncio
async def test_pin_boundary_lifespan_routes_sessions_streaming_headers_batch_and_non_tool_forwarding():
    provider = Provider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    assert [route.path for route in server.app.routes] == [
        "/mcp",
        "/.well-known/oauth-protected-resource/mcp",
    ]
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    headers = {
        "Authorization": "Bearer actor-1",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            initialized = await client.post("/mcp", json=initialize, headers=headers)
            session_id = initialized.headers["mcp-session-id"]
            ping = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
                headers={**headers, "mcp-session-id": session_id},
            )
            batch = await client.post("/mcp", json=[initialize], headers=headers)
            sse_only = await client.post("/mcp", json=initialize, headers={**headers, "Accept": "text/event-stream"})
    assert initialized.status_code == 200
    assert initialized.headers["content-type"] == "application/json"
    assert type(session_id) is str and session_id
    assert ping.status_code == 200
    assert ping.headers["content-type"] == "application/json"
    assert ping.headers["mcp-session-id"] == session_id
    assert batch.status_code == 400
    assert batch.json() == {"error": "invalid tool input"}
    assert sse_only.status_code == 406
    assert sse_only.headers["content-type"] == "application/json"
    assert provider.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
async def test_application_json_adapter_reads_once_and_replays_canonical_body_to_captured_app():
    received = []
    sent = []

    async def captured_app(scope, receive, send):
        received.append((scope, await receive(), await receive()))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    adapter = ApplicationJSONAdapter(captured_app, 512)
    messages = iter(
        (
            {"type": "http.request", "body": b'{ "params": {"arguments": {"group_id":', "more_body": True},
            {"type": "http.request", "body": b'"fleet", "name":"n"}}, "method":"tools/call"}', "more_body": False},
        )
    )

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [(b"x-test", b"preserved")]}
    await adapter(scope, receive, send)
    assert received == [
        (
            scope,
            {
                "type": "http.request",
                "body": b'{"method":"tools/call","params":{"arguments":{"group_id":"fleet","name":"n"}}}',
                "more_body": False,
            },
            {"type": "http.disconnect"},
        )
    ]
    assert sent == [
        {"type": "http.response.start", "status": 204, "headers": []},
        {"type": "http.response.body", "body": b""},
    ]


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "100"])
def test_factory_rejects_non_exact_positive_body_limits(limit):
    with pytest.raises((TypeError, ValueError)):
        build(application_body_limit_bytes=limit)


@pytest.mark.parametrize(
    "dependency",
    [
        "token_verifier",
        "auth_settings",
        "expected_issuer",
        "service_resource_server",
        "service_audience",
        "service_tenant_id",
        "grant_provider",
        "memory_factory",
    ],
)
def test_factory_refuses_every_null_dependency(dependency):
    with pytest.raises((TypeError, ValueError)):
        build(**{dependency: None})


def test_authoritative_factory_inputs_have_no_defaults():
    signature = inspect.signature(build_authorized_server)
    for name in (
        "token_verifier",
        "auth_settings",
        "expected_issuer",
        "service_resource_server",
        "service_audience",
        "service_tenant_id",
        "application_body_limit_bytes",
        "grant_provider",
        "memory_factory",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "bad_settings",
    [
        AuthSettings(issuer_url=AnyHttpUrl(ISSUER), resource_server_url=None),
        AuthSettings(
            issuer_url=AnyHttpUrl(ISSUER), resource_server_url=AnyHttpUrl(RESOURCE), required_scopes=["token-scope"]
        ),
        AuthSettings(
            issuer_url=AnyHttpUrl(ISSUER),
            resource_server_url=AnyHttpUrl(RESOURCE),
            service_documentation_url=AnyHttpUrl("https://docs.example/"),
        ),
        AuthSettings(
            issuer_url=AnyHttpUrl(ISSUER),
            resource_server_url=AnyHttpUrl(RESOURCE),
            client_registration_options=ClientRegistrationOptions(),
        ),
        AuthSettings(
            issuer_url=AnyHttpUrl(ISSUER),
            resource_server_url=AnyHttpUrl(RESOURCE),
            revocation_options=RevocationOptions(),
        ),
    ],
)
def test_factory_refuses_non_verifier_only_auth_settings(bad_settings):
    with pytest.raises(ValueError):
        build(auth_settings=bad_settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_issuer", "https://other.example/"),
        ("service_resource_server", "https://other.example/mcp"),
        ("service_audience", ""),
        ("service_tenant_id", ""),
        ("service_audience", 1),
        ("service_tenant_id", True),
    ],
)
def test_factory_refuses_every_trusted_scalar_mismatch_or_empty_value(field, value):
    with pytest.raises(ValueError):
        build(**{field: value})


def test_factory_refuses_wrong_dependency_types_and_auth_settings_subclass():
    class SettingsSubclass(AuthSettings):
        pass

    subclass = SettingsSubclass(issuer_url=AnyHttpUrl(ISSUER), resource_server_url=AnyHttpUrl(RESOURCE))
    for override in (
        {"auth_settings": subclass},
        {"token_verifier": object()},
        {"grant_provider": object()},
        {"memory_factory": object()},
    ):
        with pytest.raises(TypeError):
            build(**override)


@pytest.mark.asyncio
async def test_mutating_input_settings_cannot_change_authentication_or_local_bindings():
    supplied = settings()
    provider = Provider()
    store = memory()
    store.add_memory("n", "d", "reference", "b", group_id="fleet")
    server = build(auth_settings=supplied, grant_provider=provider, memory_factory=lambda: store)
    supplied.issuer_url = AnyHttpUrl("https://mutated.example/")
    supplied.resource_server_url = AnyHttpUrl("https://mutated.example/mcp")
    supplied.required_scopes = ["mutated"]
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("get_memory", {"name": "n", "group_id": "fleet"})
    assert not result.isError
    assert provider.calls[0][0].issuer == ISSUER


@pytest.mark.asyncio
async def test_direct_calls_reject_raw_input_then_fail_closed_without_current_context():
    provider = Provider()
    server = build(grant_provider=provider)
    forged = {"name": "n", "description": "d", "type": "reference", "body": "b", "group_id": "fleet", "actor_id": "x"}
    with pytest.raises(ToolError, match="invalid tool input"):
        await server.call_tool("add_memory", forged)
    with pytest.raises(ToolError, match="memory authorization denied"):
        await server.call_tool("get_memory", {"name": "n", "group_id": "fleet"})
    with pytest.raises(TypeError):
        await server.call_tool(  # type: ignore[call-arg]
            "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context()
        )
    assert provider.calls == []


ALIASES = (
    "identity",
    "actor",
    "actor_id",
    "principal",
    "tenant",
    "tenant_id",
    "grant",
    "grant_version",
    "capability",
    "claims",
    "auth",
    "authorization",
    "ctx",
    "context",
    "scope",
    "scope_id",
    "scopes",
    "group",
    "groups",
    "include_fleet",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ALIASES)
async def test_every_identity_grant_context_and_scope_alias_is_unknown(alias):
    provider = Provider()
    server = build(grant_provider=provider)
    arguments = {"name": "n", "group_id": "fleet", alias: "forged"}
    with pytest.raises(ToolError, match="invalid tool input"):
        await server.call_tool("get_memory", arguments)
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("add_memory", {"description": "d", "type": "reference", "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": None, "description": "d", "type": "reference", "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": 1, "type": "reference", "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": "d", "type": 1, "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": "d", "type": "reference", "body": 1, "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": "d", "type": "reference", "body": "b", "group_id": 1}),
        (
            "add_memory",
            {
                "name": "n",
                "description": "d",
                "type": "reference",
                "body": "b",
                "group_id": "fleet",
                "updated_at": 1,
            },
        ),
        (
            "add_memory",
            {
                "name": "n",
                "description": "d",
                "type": "reference",
                "body": "b",
                "group_id": "fleet",
                "provenance": [],
            },
        ),
        ("get_memory", {"name": "n", "group_id": None}),
        ("get_memory", {"name": 1, "group_id": "fleet"}),
        ("get_memory", {"name": "n", "group_id": "fleet", "extra": "x"}),
        ("search_memory", {"query": 1, "group_ids": ["fleet"]}),
        ("search_memory", {"query": "q", "group_ids": "fleet"}),
        ("search_memory", {"query": "q", "group_ids": ("fleet",)}),
        ("search_memory", {"query": "q", "group_ids": ["fleet", 1]}),
        ("search_memory", {"query": "q", "group_ids": []}),
        ("search_memory", {"query": "q", "group_ids": ["fleet"], "type": 1}),
        ("search_memory", {"query": "q", "group_ids": ["fleet"], "limit": True}),
        ("search_memory", {"query": "q", "group_ids": ["fleet"], "limit": None}),
        ("search_memory", {"query": "q", "group_ids": ["fleet"], "limit": "10"}),
        ("search_memory", {"query": "q", "group_ids": [" fleet ", "fleet"]}),
    ],
)
async def test_representable_missing_null_unknown_type_container_element_boolean_and_duplicate_scope_table(
    tool, arguments
):
    provider = Provider()
    server = build(grant_provider=provider)
    with pytest.raises(ToolError, match="invalid tool input"):
        await server.call_tool(tool, arguments)
    assert provider.calls == []


@pytest.mark.asyncio
async def test_direct_python_subclasses_are_not_exact_builtin_scalars_or_containers():
    class StringSubclass(str):
        pass

    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    server = build()
    cases = [
        ("get_memory", {"name": StringSubclass("n"), "group_id": "fleet"}),
        ("get_memory", DictSubclass(name="n", group_id="fleet")),
        ("search_memory", {"query": "q", "group_ids": ListSubclass(["fleet"])}),
    ]
    for tool, arguments in cases:
        with pytest.raises(ToolError, match="invalid tool input"):
            await server.call_tool(tool, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("add_memory", {"name": "", "description": "d", "type": "reference", "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": "", "type": "reference", "body": "b", "group_id": "fleet"}),
        ("add_memory", {"name": "n", "description": "d", "type": "", "body": "b", "group_id": "fleet"}),
        (
            "add_memory",
            {
                "name": "n",
                "description": "d",
                "type": "reference",
                "body": "b",
                "group_id": "fleet",
                "updated_at": "not-a-date",
            },
        ),
        (
            "add_memory",
            {
                "name": "n",
                "description": "d",
                "type": "reference",
                "body": "b",
                "group_id": "fleet",
                "provenance": {"supersedes": None},
            },
        ),
        ("get_memory", {"name": "", "group_id": "fleet"}),
        ("get_memory", {"name": "n", "group_id": "has space"}),
        ("get_memory", {"name": "n", "group_id": "x" * 65}),
        ("search_memory", {"query": "", "group_ids": ["fleet"]}),
        ("search_memory", {"query": "q", "group_ids": [f"scope-{index}" for index in range(17)]}),
        ("search_memory", {"query": "q", "group_ids": ["fleet"], "limit": 0}),
    ],
)
async def test_typed_semantic_validation_completes_before_context_provider_or_memory(tool, arguments):
    provider = Provider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    with pytest.raises(ToolError, match="invalid tool input"):
        await server.call_tool(tool, arguments)
    assert provider.calls == []
    assert factory_calls == []


def add_arguments(provenance):
    return {
        "name": "n",
        "description": "d",
        "type": "reference",
        "body": "b",
        "group_id": "fleet",
        "provenance": provenance,
    }


VALID_PROVENANCE_CASES = [(None, "source_repo", None), ({}, "source_repo", None)]
VALID_PROVENANCE_CASES.extend(({field: None}, field, None) for field in PROVENANCE_SCALAR_FIELDS)
VALID_PROVENANCE_CASES.extend(
    ({field: f"value-{field}"}, field, f"value-{field}") for field in PROVENANCE_SCALAR_FIELDS
)
VALID_PROVENANCE_CASES.extend(
    ({field: ["memory-a", "memory-b"]}, field, ["memory-a", "memory-b"]) for field in PROVENANCE_LIST_FIELDS
)
VALID_PROVENANCE_CASES.extend(({field: []}, field, []) for field in PROVENANCE_LIST_FIELDS)

MALFORMED_PROVENANCE_CASES = [({"unknown": "value"}, "unknown-member")]
MALFORMED_PROVENANCE_CASES.extend(
    ({field: value}, f"{field}-{type(value).__name__}")
    for field in PROVENANCE_SCALAR_FIELDS
    for value in (1, 1.5, True, [], {})
)
MALFORMED_PROVENANCE_CASES.extend(
    ({field: value}, f"{field}-{type(value).__name__}")
    for field in PROVENANCE_LIST_FIELDS
    for value in ("memory-a", {"memory-a": True}, True, 1, 1.5, None)
)
MALFORMED_PROVENANCE_CASES.extend(
    ({field: [value]}, f"{field}-element-{type(value).__name__}")
    for field in PROVENANCE_LIST_FIELDS
    for value in (True, 1, 1.5, None, {}, [])
)


class FailureLayerSpy:
    def __init__(self):
        self.calls = {
            name: 0
            for name in (
                "context",
                "provider",
                "factory",
                "embedder",
                "schema",
                "store",
                "sql",
                "result",
                "observation",
            )
        }

    def memory_factory(self):
        self.calls["factory"] += 1
        raise AssertionError("memory factory ran for invalid provenance")


@pytest.mark.asyncio
@pytest.mark.parametrize(("provenance", "field", "expected"), VALID_PROVENANCE_CASES)
async def test_direct_strict_tool_accepts_every_closed_provenance_field(provenance, field, expected):
    provider = Provider()
    store = memory()
    server = build(grant_provider=provider, memory_factory=lambda: store)
    result = await registered_call(server, "add_memory", add_arguments(provenance), context=current_context())
    assert result["provenance"][field] == expected
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provenance", "case"), MALFORMED_PROVENANCE_CASES, ids=lambda value: value if isinstance(value, str) else None
)
async def test_direct_strict_tool_rejects_every_malformed_provenance_before_all_layers(provenance, case, monkeypatch):
    del case
    layers = FailureLayerSpy()
    provider = Provider()

    def forbidden_context(*args, **kwargs):
        del args, kwargs
        layers.calls["context"] += 1
        raise AssertionError("context extraction ran for invalid provenance")

    monkeypatch.setattr(authorized_server, "_current_principal", forbidden_context)
    server = build(grant_provider=provider, memory_factory=layers.memory_factory)
    with pytest.raises(ToolError, match="invalid tool input"):
        await registered_call(server, "add_memory", add_arguments(provenance), context=object())
    layers.calls["provider"] = len(provider.calls)
    assert layers.calls == {name: 0 for name in layers.calls}


@pytest.mark.asyncio
async def test_direct_strict_tool_rejects_nested_builtin_subclasses_before_context_provider_or_memory(monkeypatch):
    class StringSubclass(str):
        pass

    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    layers = FailureLayerSpy()
    provider = Provider()

    def forbidden_context(*args, **kwargs):
        del args, kwargs
        layers.calls["context"] += 1
        raise AssertionError("context extraction ran for invalid provenance")

    monkeypatch.setattr(authorized_server, "_current_principal", forbidden_context)
    server = build(grant_provider=provider, memory_factory=layers.memory_factory)
    cases = (
        DictSubclass(source_repo="repo"),
        {"source_repo": StringSubclass("repo")},
        {"source_repo": b"repo"},
        {"supersedes": ListSubclass(["memory-a"])},
        {"supersedes": ("memory-a",)},
        {"supersedes": [StringSubclass("memory-a")]},
    )
    for provenance in cases:
        with pytest.raises(ToolError, match="invalid tool input"):
            await registered_call(server, "add_memory", add_arguments(provenance), context=object())
    layers.calls["provider"] = len(provider.calls)
    assert layers.calls == {name: 0 for name in layers.calls}


@pytest.mark.asyncio
async def test_wrapped_asgi_accepts_every_closed_provenance_field():
    provider = Provider()
    store = memory()
    server = build(grant_provider=provider, memory_factory=lambda: store)
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    for index, (provenance, field, expected) in enumerate(VALID_PROVENANCE_CASES):
                        result = await session.call_tool(
                            "add_memory", {**add_arguments(provenance), "name": f"memory-{index}"}
                        )
                        assert not result.isError
                        stored = store.get_memory(f"memory-{index}", group_id="fleet")
                        assert stored["provenance"][field] == expected
    assert len(provider.calls) == len(VALID_PROVENANCE_CASES)


@pytest.mark.asyncio
async def test_wrapped_asgi_rejects_every_malformed_provenance_before_all_layers(monkeypatch):
    layers = FailureLayerSpy()
    provider = Provider()

    def forbidden_context(*args, **kwargs):
        del args, kwargs
        layers.calls["context"] += 1
        raise AssertionError("context extraction ran for invalid provenance")

    monkeypatch.setattr(authorized_server, "_current_principal", forbidden_context)
    server = build(grant_provider=provider, memory_factory=layers.memory_factory)
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    results = [
                        await session.call_tool("add_memory", add_arguments(provenance))
                        for provenance, _ in MALFORMED_PROVENANCE_CASES
                    ]
    assert all(result.isError for result in results)
    layers.calls["provider"] = len(provider.calls)
    assert layers.calls == {name: 0 for name in layers.calls}


@pytest.mark.asyncio
async def test_issue_32_authorization_table_add_get_search_runs_only_after_exact_authorization():
    provider = Provider()
    store = memory()
    ids = iter((b"a" * 16, b"b" * 16, b"c" * 16))
    server = build(grant_provider=provider, memory_factory=lambda: store, invocation_id_factory=lambda: next(ids))

    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    added = await session.call_tool(
                        "add_memory",
                        {
                            "name": "fact",
                            "description": "shared fact",
                            "type": "reference",
                            "body": "body",
                            "group_id": "fleet",
                        },
                    )
                    got = await session.call_tool("get_memory", {"name": "fact", "group_id": "fleet"})
                    searched = await session.call_tool(
                        "search_memory", {"query": "shared", "group_ids": ["infra", "fleet"], "limit": 10}
                    )

    assert not added.isError and not got.isError and not searched.isError
    assert {tool.name for tool in listed.tools} == {"add_memory", "get_memory", "search_memory"}
    assert all(tool.inputSchema["additionalProperties"] is False for tool in listed.tools)
    assert all("include_fleet" not in tool.inputSchema["properties"] for tool in listed.tools)
    assert [call[1].invocation_id for call in provider.calls] == [b"a" * 16, b"b" * 16, b"c" * 16]
    assert provider.calls[0][1].requested_scopes == ("fleet",)
    assert provider.calls[2][1].requested_scopes == ("fleet", "infra")
    assert all(call[0].actor_id == "actor-1" for call in provider.calls)


@pytest.mark.asyncio
async def test_fake_pgvector_get_success_occurs_only_after_exact_authorization():
    connection = FakeConnection()
    connection.next_one = (
        "fleet",
        "pg-fact",
        "reference",
        "from pg",
        "body",
        "[0.5,0.5,0.5,0.5]",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        {},
    )
    semantic_memory = SemanticMemory(HashingEmbedder(dim=4), PgVectorStore(lambda: connection, dim=4))
    provider = Provider()
    server = build(grant_provider=provider, memory_factory=lambda: semantic_memory)
    result = await registered_call(
        server, "get_memory", {"name": "pg-fact", "group_id": "fleet"}, context=current_context()
    )
    assert result["name"] == "pg-fact"
    assert len(provider.calls) == 1
    assert connection.executed and "SELECT" in connection.executed[0][0]


@pytest.mark.asyncio
async def test_unauthenticated_and_duplicate_argument_bodies_stop_before_provider_and_memory(monkeypatch):
    provider = Provider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    adapter_calls = 0
    original_adapter = ApplicationJSONAdapter.__call__

    async def counted_adapter(self, scope, receive, send):
        nonlocal adapter_calls
        adapter_calls += 1
        await original_adapter(self, scope, receive, send)

    monkeypatch.setattr(ApplicationJSONAdapter, "__call__", counted_adapter)
    headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    body = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_memory",'
        b'"arguments":{"name":"a","name":"b","group_id":"fleet"}}}'
    )

    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            unauthenticated = await client.post("/mcp", content=body, headers=headers)
            invalid = await client.post("/mcp", content=body, headers={**headers, "Authorization": "Bearer invalid"})
            assert adapter_calls == 0
            duplicate = await client.post("/mcp", content=body, headers={**headers, "Authorization": "Bearer actor-1"})

    assert unauthenticated.status_code == 401
    assert invalid.status_code == 401
    assert duplicate.status_code == 400
    assert adapter_calls == 1
    assert provider.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_claims", "resource"),
    [
        (None, RESOURCE),
        ({key: value for key, value in claims().items() if key != "actor_id"}, RESOURCE),
        ({**claims(), "extra": "forged"}, RESOURCE),
        ({**claims(), "actor_id": 1}, RESOURCE),
        ({**claims(), "tenant_id": "other"}, RESOURCE),
        (claims(), "https://other.example/mcp"),
    ],
)
async def test_missing_extra_wrong_type_mismatched_claims_and_resource_stop_before_provider_and_all_work(
    token_claims, resource
):
    provider = Provider()
    layers = {name: 0 for name in ("factory", "embedder", "schema", "store", "sql", "result", "observation")}

    def forbidden_factory():
        layers["factory"] += 1
        raise AssertionError("memory/component work ran before authorization")

    server = build(grant_provider=provider, memory_factory=forbidden_factory)
    context = current_context(token_claims=token_claims, resource=resource)
    if token_claims is None:
        context = current_context(token_claims=None, resource=resource)
        context.request_context.request.scope["user"].access_token.claims = None
    context.request_context.request.scope["headers"] = [
        (b"x-actor-id", b"fallback-actor"),
        (b"x-tenant-id", b"fallback-tenant"),
        (b"x-grant-version", b"fallback-grant"),
    ]
    with pytest.raises(ToolError, match="memory authorization denied"):
        await registered_call(server, "get_memory", {"name": "n", "group_id": "fleet"}, context=context)
    assert provider.calls == []
    assert layers == {name: 0 for name in layers}


class BrokenProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    async def authorize(self, principal, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [BrokenProvider(), BrokenProvider(error=RuntimeError("offline"))])
async def test_provider_exception_or_malformed_set_denies_after_one_atomic_call_and_before_memory(provider):
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    with pytest.raises(ToolError, match="memory authorization denied"):
        await registered_call(
            server,
            "search_memory",
            {"query": "q", "group_ids": ["fleet", "infra"]},
            context=current_context(),
        )
    assert provider.calls == 1
    assert factory_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "id_factory",
    [lambda: b"short", lambda: bytearray(16), lambda: (_ for _ in ()).throw(RuntimeError("rng failed"))],
)
async def test_invalid_or_failed_invocation_generation_denies_before_provider_and_memory(id_factory):
    provider = Provider()
    factory_calls = []
    server = build(
        grant_provider=provider,
        memory_factory=lambda: factory_calls.append(True),
        invocation_id_factory=id_factory,
    )
    with pytest.raises(ToolError, match="memory authorization denied"):
        await registered_call(server, "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context())
    assert provider.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
async def test_wrapped_transport_rejects_every_alias_and_raw_type_case_before_provider_or_memory():
    provider = Provider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    cases = [("get_memory", {"name": "n", "group_id": "fleet", alias: "forged"}) for alias in ALIASES]
    cases.extend(
        [
            ("get_memory", {"name": "n", "group_id": None}),
            ("get_memory", {"name": "n", "group_id": "fleet", "unexpected": "x"}),
            ("get_memory", {"name": "n"}),
            (
                "add_memory",
                {"name": "n", "description": "d", "type": "reference", "body": "b", "group_id": None},
            ),
            ("search_memory", {"query": "q", "group_ids": "fleet"}),
            ("search_memory", {"query": "q", "group_ids": ["fleet", 1]}),
            ("search_memory", {"query": "q", "group_ids": ["fleet"], "limit": True}),
            ("search_memory", {"query": "q", "group_ids": [" fleet ", "fleet"]}),
        ]
    )
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    results = [await session.call_tool(tool, arguments) for tool, arguments in cases]
    assert all(result.isError for result in results)
    assert provider.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_memory",'
        b'"arguments":{"name":"a","group_id":"fleet"},"arguments":{"name":"b","group_id":"fleet"}}}',
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_memory",'
        b'"arguments":{"name":"a","group_id":"fleet","actor_id":"forged"}}}',
    ],
)
async def test_wrapped_raw_duplicate_arguments_key_and_valid_plus_alias_fail_closed(body):
    provider = Provider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    headers = {
        "Authorization": "Bearer actor-1",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            response = await client.post("/mcp", content=body, headers=headers)
    assert response.status_code in (200, 400)
    assert provider.calls == []
    assert factory_calls == []


@pytest.mark.asyncio
async def test_wrapped_asgi_rejects_duplicate_nested_provenance_member_before_provider_and_all_memory_layers():
    layers = FailureLayerSpy()
    provider = Provider()
    server = build(grant_provider=provider, memory_factory=layers.memory_factory)
    body = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add_memory","arguments":'
        b'{"name":"n","description":"d","type":"reference","body":"b","group_id":"fleet",'
        b'"provenance":{"source_repo":"first","source_repo":"second"}}}}'
    )
    headers = {
        "Authorization": "Bearer actor-1",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            response = await client.post("/mcp", content=body, headers=headers)
    assert response.status_code == 400
    layers.calls["provider"] = len(provider.calls)
    assert layers.calls == {name: 0 for name in layers.calls}


@pytest.mark.asyncio
async def test_authenticated_application_body_limit_fails_closed_before_manager_work():
    provider = Provider()
    factory_calls = []
    server = build(
        application_body_limit_bytes=16,
        grant_provider=provider,
        memory_factory=lambda: factory_calls.append(True),
    )
    headers = {
        "Authorization": "Bearer actor-1",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            response = await client.post("/mcp", content=b"{" + b"x" * 17, headers=headers)
    assert response.status_code == 413
    assert provider.calls == []
    assert factory_calls == []


class DenyProvider(Provider):
    async def authorize(self, principal, request):
        allowed = await super().authorize(principal, request)
        return replace_decision_effect(allowed, request.requested_scopes[-1], Effect.DENY)


def replace_decision_effect(decision_set, scope, effect):
    return replace(
        decision_set,
        decisions=tuple(
            replace(decision, effect=effect) if decision.exact_scope == scope else decision
            for decision in decision_set.decisions
        ),
    )


@pytest.mark.asyncio
async def test_unknown_and_known_unauthorized_scopes_share_bounded_denial_and_atomic_search():
    provider = DenyProvider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    for scopes in (["unknown"], ["fleet"], ["fleet", "infra"]):
        with pytest.raises(ToolError) as denied:
            await registered_call(
                server,
                "search_memory",
                {"query": "q", "group_ids": scopes},
                context=current_context(),
            )
        assert str(denied.value) == "memory authorization denied"
    assert len(provider.calls) == 3
    assert factory_calls == []


@pytest.mark.asyncio
async def test_wrapped_denial_text_is_exact_and_discloses_no_identity_scope_grant_or_existence():
    provider = DenyProvider()
    server = build(grant_provider=provider, memory_factory=lambda: pytest.fail("memory work must not run"))
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    denied = await session.call_tool(
                        "search_memory", {"query": "secret", "group_ids": ["unknown-scope"]}
                    )
    assert denied.isError
    assert len(denied.content) == 1
    assert denied.content[0].text == "memory authorization denied"


class RotatingProvider(Provider):
    async def authorize(self, principal, request):
        value = await super().authorize(principal, request)
        snapshot = f"snapshot-{len(self.calls)}"
        return replace(
            value,
            snapshot_id=snapshot,
            decisions=tuple(replace(decision, snapshot_id=snapshot) for decision in value.decisions),
        )


@pytest.mark.asyncio
async def test_retry_draws_and_binds_a_new_invocation_and_provider_snapshot():
    provider = RotatingProvider()
    values = iter((b"1" * 16, b"2" * 16))
    store = memory()
    store.add_memory("n", "d", "reference", "b", group_id="fleet")
    server = build(
        grant_provider=provider,
        memory_factory=lambda: store,
        invocation_id_factory=lambda: next(values),
    )
    for _ in range(2):
        result = await registered_call(
            server, "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context()
        )
        assert result["name"] == "n"
    assert [request.invocation_id for _, request in provider.calls] == [b"1" * 16, b"2" * 16]
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_wrapped_retry_reauthenticates_and_reauthorizes_without_session_authority():
    verifier = Verifier()
    provider = RotatingProvider()
    values = iter((b"1" * 16, b"2" * 16))
    store = memory()
    store.add_memory("n", "d", "reference", "b", group_id="fleet")
    server = build(
        token_verifier=verifier,
        grant_provider=provider,
        memory_factory=lambda: store,
        invocation_id_factory=lambda: next(values),
    )
    async with server.app.router.lifespan_context(server.app):
        async with client_factory(server.app)(headers={"Authorization": "Bearer actor-1"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    first = await session.call_tool("get_memory", {"name": "n", "group_id": "fleet"})
                    second = await session.call_tool("get_memory", {"name": "n", "group_id": "fleet"})
    assert not first.isError and not second.isError
    assert len(provider.calls) == 2
    assert [request.invocation_id for _, request in provider.calls] == [b"1" * 16, b"2" * 16]
    assert verifier.calls.count("actor-1") >= 3


class OverlapProvider(Provider):
    def __init__(self):
        super().__init__()
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    async def authorize(self, principal, request):
        self.calls.append((principal, request))
        if len(self.calls) == 2:
            self.both_started.set()
        await self.release.wait()
        self.calls.pop()
        value = await super().authorize(principal, request)
        return value


@pytest.mark.asyncio
async def test_parallel_overlapping_principals_keep_authority_invocation_local():
    provider = OverlapProvider()
    store = memory()
    store.add_memory("n", "d", "reference", "b", group_id="fleet")
    ids = iter((b"a" * 16, b"b" * 16))
    server = build(grant_provider=provider, memory_factory=lambda: store, invocation_id_factory=lambda: next(ids))
    first = asyncio.create_task(
        registered_call(server, "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context("actor-a"))
    )
    second = asyncio.create_task(
        registered_call(server, "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context("actor-b"))
    )
    await provider.both_started.wait()
    observed = {(principal.actor_id, request.invocation_id) for principal, request in provider.calls}
    assert observed == {("actor-a", b"a" * 16), ("actor-b", b"b" * 16)}
    provider.release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_wrapped_parallel_requests_keep_current_authenticated_principals_isolated():
    provider = OverlapProvider()
    store = memory()
    store.add_memory("n", "d", "reference", "b", group_id="fleet")
    ids = iter((b"a" * 16, b"b" * 16))
    server = build(grant_provider=provider, memory_factory=lambda: store, invocation_id_factory=lambda: next(ids))

    async def invoke(actor_id):
        async with client_factory(server.app)(headers={"Authorization": f"Bearer {actor_id}"}) as http_client:
            async with streamable_http_client("http://localhost:8000/mcp", http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool("get_memory", {"name": "n", "group_id": "fleet"})

    async with server.app.router.lifespan_context(server.app):
        first = asyncio.create_task(invoke("actor-a"))
        second = asyncio.create_task(invoke("actor-b"))
        await provider.both_started.wait()
        observed = {(principal.actor_id, request.invocation_id) for principal, request in provider.calls}
        assert observed == {("actor-a", b"a" * 16), ("actor-b", b"b" * 16)}
        provider.release.set()
        results = await asyncio.gather(first, second)
    assert all(not result.isError for result in results)


class BlockingProvider:
    def __init__(self):
        self.started = asyncio.Event()

    async def authorize(self, principal, request):
        self.started.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cancellation_during_provider_propagates_with_zero_memory_work():
    provider = BlockingProvider()
    factory_calls = []
    server = build(grant_provider=provider, memory_factory=lambda: factory_calls.append(True))
    task = asyncio.create_task(
        registered_call(server, "get_memory", {"name": "n", "group_id": "fleet"}, context=current_context())
    )
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert factory_calls == []


class BlockingVerifier:
    def __init__(self):
        self.started = asyncio.Event()

    async def verify_token(self, token):
        self.started.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_cancellation_during_verifier_propagates_before_adapter_provider_or_memory():
    verifier = BlockingVerifier()
    provider = Provider()
    factory_calls = []
    server = build(
        token_verifier=verifier,
        grant_provider=provider,
        memory_factory=lambda: factory_calls.append(True),
    )
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    }
    headers = {
        "Authorization": "Bearer actor-1",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://localhost:8000"
        ) as client:
            task = asyncio.create_task(client.post("/mcp", json=initialize, headers=headers))
            await verifier.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert provider.calls == []
    assert factory_calls == []
