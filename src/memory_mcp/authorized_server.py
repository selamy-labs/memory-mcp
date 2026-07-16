"""Default-deny authenticated ASGI composition for shared semantic memory."""

from __future__ import annotations

import copy
import json
import secrets
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser, BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.fastmcp.tools.base import Tool
from pydantic import AnyHttpUrl, PrivateAttr
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from memory_mcp.authorization import (
    DENIAL_MESSAGE,
    AuthenticatedPrincipal,
    AuthorizationRequest,
    GrantProvider,
    Operation,
    principal_from_claims,
    validate_closed_string,
    validate_decision_set,
)
from memory_mcp.semantic import SemanticMemory, SemanticMemoryError, _coerce_updated_at, _validate_scope
from memory_mcp.vector_store import Provenance

INVALID_INPUT_MESSAGE = "invalid tool input"
INSTRUCTIONS = (
    "Authorized shared semantic memory. add_memory and get_memory require one explicit group_id; "
    "search_memory requires one through sixteen explicit group_ids. Scopes identify requested resources only "
    "and never establish identity, tenant, grants, or authority."
)


class InvalidToolInput(ValueError):
    pass


class BodyTooLarge(InvalidToolInput):
    pass


def _invalid() -> InvalidToolInput:
    return InvalidToolInput(INVALID_INPUT_MESSAGE)


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    required: bool
    python_types: tuple[type, ...]
    schema: Mapping[str, Any]


_STRING = {"type": "string"}
_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_NULLABLE_OBJECT = {"anyOf": [{"type": "object"}, {"type": "null"}]}
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
_INTEGER = {"type": "integer"}

_FIELDS: dict[str, tuple[_Field, ...]] = {
    "add_memory": (
        _Field("name", True, (str,), _STRING),
        _Field("description", True, (str,), _STRING),
        _Field("type", True, (str,), _STRING),
        _Field("body", True, (str,), _STRING),
        _Field("group_id", True, (str,), _STRING),
        _Field("updated_at", False, (str, type(None)), _NULLABLE_STRING),
        _Field("provenance", False, (dict, type(None)), _NULLABLE_OBJECT),
    ),
    "get_memory": (
        _Field("name", True, (str,), _STRING),
        _Field("group_id", True, (str,), _STRING),
    ),
    "search_memory": (
        _Field("query", True, (str,), _STRING),
        _Field("group_ids", True, (list,), _STRING_ARRAY),
        _Field("type", False, (str, type(None)), _NULLABLE_STRING),
        _Field("limit", False, (int,), _INTEGER),
    ),
}


def _schema(fields: tuple[_Field, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field.name: copy.deepcopy(dict(field.schema)) for field in fields},
        "required": [field.name for field in fields if field.required],
        "additionalProperties": False,
    }


def generate_invocation_id() -> bytes:
    """Draw one fresh service-side invocation value."""
    return secrets.token_bytes(16)


def _validate_raw(tool_name: str, arguments: object) -> dict[str, Any]:
    if type(arguments) is not dict or any(type(key) is not str for key in arguments):
        raise _invalid()
    fields = _FIELDS[tool_name]
    allowed = {field.name for field in fields}
    required = {field.name for field in fields if field.required}
    if set(arguments) - allowed or not required.issubset(arguments):
        raise _invalid()
    output: dict[str, Any] = {}
    for field in fields:
        if field.name not in arguments:
            continue
        value = arguments[field.name]
        if not any(type(value) is accepted for accepted in field.python_types):
            raise _invalid()
        output[field.name] = value
    if tool_name == "search_memory":
        groups = output["group_ids"]
        if not 1 <= len(groups) <= 16 or any(type(group) is not str for group in groups):
            raise _invalid()
    return output


def _scope(value: str) -> str:
    try:
        scope = _validate_scope(value)
    except (SemanticMemoryError, UnicodeError) as error:
        raise _invalid() from error
    if len(scope.encode("utf-8")) > 256:
        raise _invalid()
    return scope


def _nonempty(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise _invalid()
    return cleaned


def _validate_semantic(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        if tool_name == "add_memory":
            normalized = {
                **arguments,
                "name": _nonempty(arguments["name"]),
                "description": _nonempty(arguments["description"]),
                "type": _nonempty(arguments["type"]),
                "group_id": _scope(arguments["group_id"]),
                "updated_at": (
                    _coerce_updated_at(arguments["updated_at"]) if arguments.get("updated_at") is not None else None
                ),
                "provenance": Provenance.from_mapping(arguments.get("provenance")),
            }
            return normalized, (normalized["group_id"],)
        if tool_name == "get_memory":
            normalized = {"name": _nonempty(arguments["name"]), "group_id": _scope(arguments["group_id"])}
            return normalized, (normalized["group_id"],)

        limit = arguments.get("limit", 10)
        if type(limit) is not int or limit < 1:
            raise _invalid()
        scopes = tuple(_scope(group) for group in arguments["group_ids"])
        if len(set(scopes)) != len(scopes):
            raise _invalid()
        normalized = {
            "query": _nonempty(arguments["query"]),
            "group_ids": list(scopes),
            "type": arguments.get("type"),
            "limit": min(limit, 100),
        }
        return normalized, tuple(sorted(scopes, key=lambda item: item.encode("utf-8")))
    except InvalidToolInput:
        raise
    except (TypeError, ValueError, SemanticMemoryError, UnicodeError) as error:
        raise _invalid() from error


def _current_principal(
    context: object,
    *,
    expected_issuer: str,
    service_resource_server: str,
    service_audience: str,
    service_tenant_id: str,
) -> AuthenticatedPrincipal:
    if type(context) is not Context:
        raise ValueError(DENIAL_MESSAGE)
    try:
        request = context.request_context.request
    except (AttributeError, ValueError) as error:
        raise ValueError(DENIAL_MESSAGE) from error
    if type(request) is not Request:
        raise ValueError(DENIAL_MESSAGE)
    user = request.scope.get("user")
    if type(user) is not AuthenticatedUser or type(user.access_token) is not AccessToken:
        raise ValueError(DENIAL_MESSAGE)
    access_token = user.access_token
    if type(access_token.resource) is not str or access_token.resource != service_resource_server:
        raise ValueError(DENIAL_MESSAGE)
    return principal_from_claims(
        access_token.claims,
        expected_issuer=expected_issuer,
        expected_audience=service_audience,
        expected_tenant_id=service_tenant_id,
    )


class StrictToolDispatcher:
    """One closed raw/semantic/authenticated dispatch path for all three tools."""

    __slots__ = (
        "_expected_issuer",
        "_grant_provider",
        "_invocation_id_factory",
        "_memory_factory",
        "_service_audience",
        "_service_resource_server",
        "_service_tenant_id",
    )

    def __init__(
        self,
        *,
        expected_issuer: str,
        service_resource_server: str,
        service_audience: str,
        service_tenant_id: str,
        grant_provider: GrantProvider,
        memory_factory: Callable[[], SemanticMemory],
        invocation_id_factory: Callable[[], bytes],
    ) -> None:
        self._expected_issuer = expected_issuer
        self._service_resource_server = service_resource_server
        self._service_audience = service_audience
        self._service_tenant_id = service_tenant_id
        self._grant_provider = grant_provider
        self._memory_factory = memory_factory
        self._invocation_id_factory = invocation_id_factory

    async def dispatch(self, tool_name: str, arguments: object, context: object) -> dict[str, Any]:
        raw = _validate_raw(tool_name, arguments)
        normalized, scopes = _validate_semantic(tool_name, raw)
        principal = _current_principal(
            context,
            expected_issuer=self._expected_issuer,
            service_resource_server=self._service_resource_server,
            service_audience=self._service_audience,
            service_tenant_id=self._service_tenant_id,
        )
        try:
            invocation_id = self._invocation_id_factory()
        except Exception as error:
            raise ValueError(DENIAL_MESSAGE) from error
        if type(invocation_id) is not bytes or len(invocation_id) != 16:
            raise ValueError(DENIAL_MESSAGE)
        operation = {
            "add_memory": Operation.ADD,
            "get_memory": Operation.GET,
            "search_memory": Operation.SEARCH,
        }[tool_name]
        request = AuthorizationRequest(invocation_id, operation, self._service_tenant_id, scopes)
        try:
            decision_set = await self._grant_provider.authorize(principal, request)
        except Exception as error:
            raise ValueError(DENIAL_MESSAGE) from error
        validate_decision_set(decision_set, principal=principal, request=request)

        memory = self._memory_factory()
        if tool_name == "add_memory":
            return memory.add_memory(
                normalized["name"],
                normalized["description"],
                normalized["type"],
                normalized["body"],
                group_id=normalized["group_id"],
                updated_at=normalized["updated_at"],
                provenance=normalized["provenance"],
            )
        if tool_name == "get_memory":
            return memory.get_memory(normalized["name"], group_id=normalized["group_id"])
        return memory.search_memory(
            normalized["query"],
            group_ids=normalized["group_ids"],
            include_fleet=False,
            type=normalized["type"],
            limit=normalized["limit"],
        )


async def _metadata_function() -> dict[str, Any]:
    return {}


class StrictTool(Tool):
    """MCP 1.28.1 Tool whose public run path always uses strict dispatch."""

    _dispatcher: StrictToolDispatcher = PrivateAttr()

    @classmethod
    def create(cls, name: str, dispatcher: StrictToolDispatcher) -> StrictTool:
        base = Tool.from_function(_metadata_function, name=name, description=f"Authorized {name}")
        tool = cls(
            fn=base.fn,
            name=name,
            description=base.description,
            parameters=_schema(_FIELDS[name]),
            fn_metadata=base.fn_metadata,
            is_async=True,
            context_kwarg=None,
        )
        tool._dispatcher = dispatcher
        return tool

    async def run(self, arguments: dict[str, Any], context: Context | None = None, convert_result: bool = False) -> Any:
        try:
            return await self._dispatcher.dispatch(self.name, arguments, context)
        except InvalidToolInput as error:
            raise ToolError(INVALID_INPUT_MESSAGE) from error
        except ToolError:
            raise
        except SemanticMemoryError as error:
            raise ToolError(str(error)) from error
        except ValueError as error:
            if str(error) == DENIAL_MESSAGE:
                raise ToolError(DENIAL_MESSAGE) from error
            raise ToolError(str(error)) from error


class _Pairs(list[tuple[str, Any]]):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, _Pairs):
        return {key: _plain(item) for key, item in value}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _member(value: _Pairs, name: str) -> list[Any]:
    return [item for key, item in value if key == name]


class ApplicationJSONAdapter:
    """Bounded pair-preserving JSON adapter before MCP's ordinary decoder."""

    __slots__ = ("_app", "_limit")

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self._app = app
        self._limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return
        try:
            replay = _reencode_application_json(await self._read_body(receive))
        except BodyTooLarge:
            await JSONResponse({"error": INVALID_INPUT_MESSAGE}, status_code=413)(scope, receive, send)
            return
        except (UnicodeDecodeError, json.JSONDecodeError, InvalidToolInput, TypeError, ValueError):
            await JSONResponse({"error": INVALID_INPUT_MESSAGE}, status_code=400)(scope, receive, send)
            return

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": replay, "more_body": False}
            return {"type": "http.disconnect"}

        await self._app(scope, replay_receive, send)

    async def _read_body(self, receive: Receive) -> bytes:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                raise _invalid()
            body.extend(message.get("body", b""))
            if len(body) > self._limit:
                raise BodyTooLarge(INVALID_INPUT_MESSAGE)
            if not message.get("more_body", False):
                return bytes(body)


def _reencode_application_json(body: bytes) -> bytes:
    parsed = json.loads(body, object_pairs_hook=_Pairs)
    if not isinstance(parsed, _Pairs):
        raise _invalid()
    methods = _member(parsed, "method")
    if methods and methods[-1] == "tools/call":
        _reject_duplicate_arguments(parsed)
    return json.dumps(_plain(parsed), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _reject_duplicate_arguments(envelope: _Pairs) -> None:
    params_values = _member(envelope, "params")
    if not params_values or not isinstance(params_values[-1], _Pairs):
        return
    argument_values = _member(params_values[-1], "arguments")
    if len(argument_values) > 1:
        raise _invalid()
    if argument_values and isinstance(argument_values[0], _Pairs):
        names = [key for key, _ in argument_values[0]]
        if len(names) != len(set(names)):
            raise _invalid()


class _CapturedManagerEndpoint:
    __slots__ = ("_manager",)

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._manager.handle_request(scope, receive, send)


class AuthorizedMemoryMCP:
    """Closed supported surface: captured ASGI app plus strict direct calls."""

    __slots__ = ("_app", "_tools")

    def __init__(self, app: Starlette, tools: tuple[StrictTool, ...]) -> None:
        self._app = app
        self._tools = {tool.name: tool for tool in tools}

    @property
    def app(self) -> Starlette:
        return self._app

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "inputSchema": copy.deepcopy(tool.parameters)}
            for tool in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        return await tool.run(arguments, context=None)


def _required_dependency(value: object, name: str) -> None:
    if value is None:
        raise ValueError(f"{name} is required")


def build_authorized_server(
    *,
    token_verifier: TokenVerifier,
    auth_settings: AuthSettings,
    expected_issuer: str,
    service_resource_server: str,
    service_audience: str,
    service_tenant_id: str,
    application_body_limit_bytes: int,
    grant_provider: GrantProvider,
    memory_factory: Callable[[], SemanticMemory],
    invocation_id_factory: Callable[[], bytes] | None = None,
) -> AuthorizedMemoryMCP:
    """Build the one-time, non-bypass authenticated MCP application."""
    for value, name in (
        (token_verifier, "token_verifier"),
        (auth_settings, "auth_settings"),
        (expected_issuer, "expected_issuer"),
        (service_resource_server, "service_resource_server"),
        (service_audience, "service_audience"),
        (service_tenant_id, "service_tenant_id"),
        (application_body_limit_bytes, "application_body_limit_bytes"),
        (grant_provider, "grant_provider"),
        (memory_factory, "memory_factory"),
    ):
        _required_dependency(value, name)
    if type(auth_settings) is not AuthSettings:
        raise TypeError("auth_settings must be the pinned AuthSettings type")
    if type(application_body_limit_bytes) is not int or application_body_limit_bytes <= 0:
        raise ValueError("application_body_limit_bytes must be a positive exact integer")
    if not callable(getattr(token_verifier, "verify_token", None)):
        raise TypeError("token_verifier must implement verify_token")
    if not callable(getattr(grant_provider, "authorize", None)):
        raise TypeError("grant_provider must implement authorize")
    if not callable(memory_factory):
        raise TypeError("memory_factory must be callable")

    bindings = tuple(
        validate_closed_string(value)
        for value in (expected_issuer, service_resource_server, service_audience, service_tenant_id)
    )
    issuer, resource, audience, tenant = bindings
    if (
        auth_settings.resource_server_url is None
        or str(auth_settings.issuer_url) != issuer
        or str(auth_settings.resource_server_url) != resource
        or auth_settings.required_scopes not in (None, [])
        or auth_settings.service_documentation_url is not None
        or auth_settings.client_registration_options is not None
        or auth_settings.revocation_options is not None
    ):
        raise ValueError("auth_settings is not exact verifier-only configuration")

    copied_auth = AuthSettings(
        issuer_url=AnyHttpUrl(issuer),
        resource_server_url=AnyHttpUrl(resource),
        required_scopes=[],
    )
    if str(copied_auth.issuer_url) != issuer or str(copied_auth.resource_server_url) != resource:
        raise ValueError("copied authentication settings mismatch")

    id_factory = generate_invocation_id if invocation_id_factory is None else invocation_id_factory
    if not callable(id_factory):
        raise TypeError("invocation_id_factory must be callable")
    dispatcher = StrictToolDispatcher(
        expected_issuer=issuer,
        service_resource_server=resource,
        service_audience=audience,
        service_tenant_id=tenant,
        grant_provider=grant_provider,
        memory_factory=memory_factory,
        invocation_id_factory=id_factory,
    )
    tools = tuple(StrictTool.create(name, dispatcher) for name in ("add_memory", "get_memory", "search_memory"))
    private_fastmcp = FastMCP(
        "memory-mcp-shared",
        instructions=INSTRUCTIONS,
        tools=list(tools),
        token_verifier=token_verifier,
        auth=copied_auth,
        json_response=True,
    )
    private_fastmcp.streamable_http_app()
    manager = private_fastmcp.session_manager
    endpoint = _CapturedManagerEndpoint(manager)
    adapter = ApplicationJSONAdapter(endpoint, application_body_limit_bytes)
    resource_url = AnyHttpUrl(resource)
    issuer_url = AnyHttpUrl(issuer)
    routes = [
        Route(
            "/mcp",
            endpoint=RequireAuthMiddleware(
                adapter, required_scopes=[], resource_metadata_url=build_resource_metadata_url(resource_url)
            ),
        ),
        *create_protected_resource_routes(
            resource_url=resource_url, authorization_servers=[issuer_url], scopes_supported=[]
        ),
    ]

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with manager.run():
            yield

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(token_verifier)),
            Middleware(AuthContextMiddleware),
        ],
        lifespan=lifespan,
    )
    return AuthorizedMemoryMCP(app, tools)
