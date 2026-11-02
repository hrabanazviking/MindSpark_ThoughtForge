"""
TF-153 — the service shape, filed: ThoughtForge
callable as a library AND as a local service
(loopback only by default) — one contract.

TF-153's roadmap text: *Service shape:
ThoughtForge callable as a library AND as a
local service (loopback only by default) — one
contract.* It opens Part IX (Bindings).

**What this slice files**

  * One contract, as plain JSON-safe dicts. A
    request is ``{"format":
    "thoughtforge-service-request/1",
    "contract_version": <TF-034's API_VERSION>,
    "op": <op>, ...}`` and a response is
    ``{"format":
    "thoughtforge-service-response/1",
    "contract_version": ..., "op": ..., "ok":
    bool, ...}``. The contract version is
    consumed from TF-034
    (``thoughtforge.cognition.api_surface``'s
    ``API_VERSION`` / ``api_manifest()`` /
    ``verify_api_surface()``) — this module
    invents no competing version.
  * ``SERVICE_OPS`` — three ops as literal
    data: ``health``, ``contract`` (returns the
    TF-034 api manifest plus this service
    contract) and ``turn`` (turn-in / turn-out
    through ``ThoughtForgeCore.think()``,
    returning the response text, the turn and
    response ids, the mode, the enforcement
    verdict and the citations — the fields of
    the real ``FinalResponseRecord``). TF-154
    deepens the cognition API; TF-153 files
    only this shape.
  * The library shape: ``ThoughtForgeService``
    wrapping a ``ThoughtForgeCore`` (lazy by
    default, an injectable core / core factory
    / turn callable for tests), exposing
    ``handle(request: dict) -> dict``.
  * The service shape: a stdlib-only
    ``http.server`` server —
    ``make_server(service, host="127.0.0.1",
    port=0)`` with routes ``GET /health``,
    ``GET /contract`` and ``POST /turn``. The
    HTTP layer calls the SAME ``handle()`` the
    library caller calls, with the same request
    and response dicts. ``DEFAULT_BIND`` is
    ``"127.0.0.1"``, filed as data; binding a
    non-loopback host requires an explicit
    ``allow_non_loopback=True`` and that act is
    recorded on the server object as
    ``non_loopback_acknowledged``. TF-163 owns
    the logged proof in depth; TF-153 files
    the default and the explicit gate.
  * ``SERVICE_RULES`` — the shape's law as
    literal data, and ``service_shape_problems()``
    — a checker that re-measures the live tree:
    the filings, the TF-034 surface, the default
    bind, a library round-trip and a REAL
    loopback HTTP round-trip (ephemeral port,
    in-process thread, urllib client) returning
    identical response dicts for health and
    contract, and a turn against a
    knowledge-only core (the TF-084 fixture
    shape: ``ThoughtForgeCore(memory_dir=
    tmp/memory, db_path=tmp/nonexistent.db,
    model_path=None)``).
  * Refusals are data, never raises: a
    malformed request (not a dict, missing /
    wrong format, unknown op, missing
    ``user_text`` for ``turn``, or an
    incompatible contract major version)
    returns a truthful refusal response dict
    (``ok=False``, ``error`` naming the problem)
    from ``handle()``, and the HTTP layer maps
    it to status 400 (200 on ok, 404 on an
    unknown route, 405 on a wrong method).
    TF-162 owns versioning enforcement in
    depth; TF-153 files the shape and the
    major-mismatch refusal.

**Docket**: TF-153 has no share of TF-016 and
finds no new defect in existing code; it does
NOT edit ``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``.
TF-016 totals stand unchanged (137 entries —
32 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 4
CLOSED-BY-VERDICT), filed in the open here and
in the render.

The human-readable doc is
``docs/architecture/SERVICE_SHAPE.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.service_shape``).
Proof grade: FIXTURE — a knowledge-only core
plus real loopback HTTP on this VM; a live
model turn through the service is NOT PROVEN,
never claimed.
"""

from __future__ import annotations

import ipaddress
import json
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)

SERVICE_SHAPE_FORMAT = "thoughtforge-service-shape/1"
SERVICE_SHAPE_DOC = "docs/architecture/SERVICE_SHAPE.md"
REQUEST_FORMAT = "thoughtforge-service-request/1"
RESPONSE_FORMAT = "thoughtforge-service-response/1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The default bind: loopback only. Filed as
#: data; the checker pins that it never moves.
DEFAULT_BIND = "127.0.0.1"

#: Hostnames that name the loopback interface
#: without parsing as an IP address.
_LOOPBACK_NAMES: tuple[str, ...] = ("localhost",)


# ── The ops, as literal data ─────────────────────────────────────────────────

SERVICE_OPS: tuple[dict[str, Any], ...] = (
    {"id": "health",
     "route": "GET /health",
     "summary": "the service is up and files its "
                "contract version and op ids",
     "request_fields": ["format", "contract_version", "op"],
     "response_fields": ["format", "contract_version", "op",
                         "ok", "status", "ops"]},
    {"id": "contract",
     "route": "GET /contract",
     "summary": "the service contract data plus the TF-034 "
                "cognition api manifest it is versioned by",
     "request_fields": ["format", "contract_version", "op"],
     "response_fields": ["format", "contract_version", "op",
                         "ok", "manifest", "contract"]},
    {"id": "turn",
     "route": "POST /turn",
     "summary": "one cognition turn in / out through "
                "ThoughtForgeCore.think(); the response "
                "fields are the FinalResponseRecord's",
     "request_fields": ["format", "contract_version", "op",
                        "user_text"],
     "optional_fields": ["retrieval_path", "num_drafts", "history"],
     "response_fields": ["format", "contract_version", "op",
                         "ok", "text", "response_id", "turn_id",
                         "mode", "enforcement_passed", "citations",
                         "token_count"]},
)

OP_IDS: tuple[str, ...] = tuple(op["id"] for op in SERVICE_OPS)


# ── The routes, as literal data ──────────────────────────────────────────────

SERVICE_ROUTES: tuple[dict[str, Any], ...] = (
    {"op": "health", "method": "GET", "path": "/health"},
    {"op": "contract", "method": "GET", "path": "/contract"},
    {"op": "turn", "method": "POST", "path": "/turn"},
)


# ── The rules, as literal data ───────────────────────────────────────────────

SERVICE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "one_contract_both_shapes",
     "rule": "the library handle() and the HTTP routes speak "
             "the SAME request and response dicts — one "
             "contract, two shapes; the HTTP layer adds no "
             "field and removes none"},
    {"id": "loopback_by_default",
     "rule": "DEFAULT_BIND is 127.0.0.1: the service binds "
             "the loopback interface unless a caller "
             "explicitly says otherwise"},
    {"id": "non_loopback_is_explicit",
     "rule": "binding a non-loopback host requires "
             "allow_non_loopback=True, judged before any "
             "socket is bound, and the act is recorded on "
             "the server object as "
             "non_loopback_acknowledged (TF-163 owns the "
             "logged proof in depth)"},
    {"id": "refusals_are_data_never_raises",
     "rule": "a malformed request returns a refusal "
             "response dict (ok=False, error naming the "
             "problem); handle() never raises on a "
             "malformed request and never silently "
             "corrupts one into a different op"},
    {"id": "contract_version_is_tf034",
     "rule": "the contract_version is TF-034's "
             "API_VERSION from "
             "thoughtforge.cognition.api_surface — no "
             "competing version is invented here; a "
             "request on an incompatible contract major "
             "version is refused (TF-162 owns versioning "
             "enforcement in depth)"},
    {"id": "turn_fields_from_final_response",
     "rule": "the turn response's fields are the real "
             "FinalResponseRecord's: text, response_id, "
             "turn_id, mode, enforcement_passed, citations "
             "and token_count — sizes and fields, filed "
             "from the record the core actually returns"},
)


# ── The contract as plain data ───────────────────────────────────────────────

def service_contract() -> dict[str, Any]:
    """The service contract as plain JSON-safe data,
    built from the filed literals above and TF-034's
    version. A fresh copy each call."""
    return {
        "format": SERVICE_SHAPE_FORMAT,
        "request_format": REQUEST_FORMAT,
        "response_format": RESPONSE_FORMAT,
        "contract_version": API_VERSION,
        "default_bind": DEFAULT_BIND,
        "ops": [dict(op) for op in SERVICE_OPS],
        "routes": [dict(r) for r in SERVICE_ROUTES],
        "rules": [dict(r) for r in SERVICE_RULES],
    }


def make_request(op: str, **fields: Any) -> dict[str, Any]:
    """Build a contract request dict for an op."""
    return {"format": REQUEST_FORMAT,
            "contract_version": API_VERSION,
            "op": op, **fields}


def _response(op: Any, ok: bool, **fields: Any) -> dict[str, Any]:
    return {"format": RESPONSE_FORMAT,
            "contract_version": API_VERSION,
            "op": op if isinstance(op, str) else "",
            "ok": ok, **fields}


def _refusal(op: Any, error: str) -> dict[str, Any]:
    return _response(op, False, error=error)


def _contract_major(version: Any) -> int | None:
    """The major component of a contract version
    string, or None when it does not parse."""
    if not isinstance(version, str) or not version:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


# ── Pure validators ──────────────────────────────────────────────────────────

def service_op_problems(row: Any) -> list[str]:
    """Judge one op filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["op: the row is empty"]
    problems: list[str] = []
    for key in ("id", "route", "summary"):
        if not row.get(key):
            problems.append(f"op {row.get('id')}: no {key} filed")
    for key in ("request_fields", "response_fields"):
        if not row.get(key) or not isinstance(row.get(key), list):
            problems.append(
                f"op {row.get('id')}: no {key} filed as a list")
    return problems


def service_rule_problems(row: Any) -> list[str]:
    """Judge one rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("rule: the row files no id")
    if not row.get("rule"):
        problems.append(f"rule {row.get('id')}: no rule filed")
    return problems


def service_route_problems(row: Any) -> list[str]:
    """Judge one route filing, purely: it must name
    a filed op, a method and a path. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["route: the row is empty"]
    problems: list[str] = []
    if row.get("op") not in OP_IDS:
        problems.append(
            f"route {row.get('path')}: op is {row.get('op')!r}, "
            f"not one of {OP_IDS}")
    if row.get("method") not in ("GET", "POST"):
        problems.append(
            f"route {row.get('path')}: method is "
            f"{row.get('method')!r}")
    if not str(row.get("path", "")).startswith("/"):
        problems.append(f"route: path {row.get('path')!r} is not a path")
    return problems


def service_response_problems(resp: Any) -> list[str]:
    """Judge one response dict against the contract,
    purely. Empty-safe."""
    if not isinstance(resp, dict) or not resp:
        return ["response: the response is empty"]
    problems: list[str] = []
    if resp.get("format") != RESPONSE_FORMAT:
        problems.append(
            f"response: format is {resp.get('format')!r}, "
            f"not {RESPONSE_FORMAT!r}")
    if resp.get("contract_version") != API_VERSION:
        problems.append(
            f"response: contract_version is "
            f"{resp.get('contract_version')!r}, not TF-034's "
            f"{API_VERSION!r}")
    if not isinstance(resp.get("ok"), bool):
        problems.append("response: ok is not a boolean")
    if resp.get("ok") is False and not resp.get("error"):
        problems.append(
            "response: a refusal files no error naming the problem")
    try:
        json.dumps(resp)
    except (TypeError, ValueError) as exc:
        problems.append(f"response: not JSON-safe: {exc}")
    return problems


def is_loopback_host(host: Any) -> bool:
    """True when a bind host names only the loopback
    interface (127.0.0.0/8, ::1, or localhost)."""
    if not isinstance(host, str) or not host:
        return False
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def bind_problems(host: Any, allow_non_loopback: bool = False) -> list[str]:
    """Judge a bind configuration, purely — this is
    the validation make_server() raises on, judged
    WITHOUT binding anything. Empty-safe."""
    if not isinstance(host, str) or not host:
        return ["bind: the host is empty or not a string"]
    if is_loopback_host(host):
        return []
    if not allow_non_loopback:
        return [f"bind: non-loopback host {host!r} requires "
                "allow_non_loopback=True — loopback is the "
                "default and leaving it is an explicit act"]
    return []


# ── The library shape ────────────────────────────────────────────────────────

def _record_fields(record: Any) -> dict[str, Any]:
    """The turn response fields, read from a real
    FinalResponseRecord (or a mapping with the same
    fields, as an injected turn callable may return)."""
    def get(name: str, default: Any = "") -> Any:
        if isinstance(record, dict):
            return record.get(name, default)
        return getattr(record, name, default)

    text = get("text", get("response_text", ""))
    citations = get("citations", [])
    if not isinstance(citations, list):
        citations = [str(citations)]
    return {
        "text": str(text),
        "response_id": str(get("response_id")),
        "turn_id": str(get("turn_id")),
        "mode": str(get("mode")),
        "enforcement_passed": bool(get("enforcement_passed", False)),
        "citations": [str(c) for c in citations],
        "token_count": int(get("token_count", 0) or 0),
    }


class ThoughtForgeService:
    """ThoughtForge behind the one contract.

    Wraps a ThoughtForgeCore: pass ``core`` to inject
    one, ``core_factory`` to build one lazily, or
    ``turn_callable`` to inject the turn itself (a
    callable taking ``user_text`` and returning a
    FinalResponseRecord or a mapping with its
    fields). With none of those, a default
    ThoughtForgeCore is constructed lazily on the
    first turn — health and contract never build it.
    """

    def __init__(
        self,
        core: Any = None,
        *,
        core_factory: Callable[[], Any] | None = None,
        turn_callable: Callable[..., Any] | None = None,
    ) -> None:
        self._core = core
        self._core_factory = core_factory
        self._turn_callable = turn_callable

    def _think(self, request: dict[str, Any]) -> Any:
        if self._turn_callable is not None:
            return self._turn_callable(request["user_text"])
        if self._core is None:
            if self._core_factory is not None:
                self._core = self._core_factory()
            else:
                from thoughtforge.cognition.core import ThoughtForgeCore

                self._core = ThoughtForgeCore()
        kwargs: dict[str, Any] = {}
        for key in ("retrieval_path", "num_drafts", "history"):
            if key in request:
                kwargs[key] = request[key]
        return self._core.think(request["user_text"], **kwargs)

    def handle(self, request: Any) -> dict[str, Any]:
        """The one contract entry point — the SAME
        function the HTTP layer calls. Malformed
        requests return a refusal dict; this method
        never raises on a malformed request."""
        if not isinstance(request, dict):
            return _refusal(
                "", "request is not a dict — a service request "
                    "is a JSON object with format, "
                    "contract_version and op")
        op = request.get("op")
        if request.get("format") != REQUEST_FORMAT:
            return _refusal(
                op, f"request format is {request.get('format')!r}, "
                    f"not {REQUEST_FORMAT!r}")
        version = request.get("contract_version")
        if version is None:
            return _refusal(
                op, "request files no contract_version — the "
                    "contract is versioned by TF-034's "
                    f"API_VERSION ({API_VERSION})")
        major = _contract_major(version)
        if major is None:
            return _refusal(
                op, f"contract_version {version!r} does not parse "
                    "as a version")
        expected_major = _contract_major(API_VERSION)
        if major != expected_major:
            return _refusal(
                op, f"contract_version {version!r} has major "
                    f"{major}, incompatible with this service's "
                    f"contract major {expected_major} "
                    f"({API_VERSION}) — a truthful refusal, "
                    "never silent corruption (TF-162 owns "
                    "versioning enforcement in depth)")
        if op not in OP_IDS:
            return _refusal(
                op, f"unknown op {op!r} — the filed ops are "
                    f"{list(OP_IDS)}")
        if op == "health":
            return _response(
                op, True, status="ok", ops=list(OP_IDS))
        if op == "contract":
            return _response(
                op, True, manifest=api_manifest(),
                contract=service_contract())
        # op == "turn"
        user_text = request.get("user_text")
        if not isinstance(user_text, str) or not user_text:
            return _refusal(
                op, "turn request files no user_text — a turn "
                    "needs the user's text as a non-empty string")
        try:
            record = self._think(request)
            fields = _record_fields(record)
        except Exception as exc:  # a failed turn is data too
            return _refusal(
                op, f"turn failed: {type(exc).__name__}: {exc}")
        return _response(op, True, **fields)


# ── The service shape (stdlib http.server only) ──────────────────────────────

class _ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    service: ThoughtForgeService
    bind_host: str = DEFAULT_BIND
    non_loopback_acknowledged: bool = False


class _Handler(BaseHTTPRequestHandler):
    server: _ServiceHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send(self, resp: dict[str, Any], status: int) -> None:
        body = json.dumps(resp).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path
        routes = {r["path"]: r for r in SERVICE_ROUTES}
        route = routes.get(path)
        if route is None:
            self._send(_refusal(
                "", f"unknown route {path!r} — the filed routes "
                    f"are {[r['path'] for r in SERVICE_ROUTES]}"), 404)
            return
        if route["method"] != method:
            self._send(_refusal(
                route["op"],
                f"method {method} on {path!r} — this route is "
                f"{route['method']}"), 405)
            return
        service = self.server.service
        if method == "GET":
            resp = service.handle(make_request(route["op"]))
        else:  # POST /turn
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                body: Any = {}
            else:
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    self._send(_refusal(
                        route["op"],
                        "request body is not valid JSON — the "
                        "body is the service request dict"), 400)
                    return
            if isinstance(body, dict) and "op" not in body:
                body["op"] = route["op"]  # the route supplies the op
            resp = service.handle(body)
        self._send(resp, 200 if resp.get("ok") else 400)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")


def make_server(
    service: ThoughtForgeService | None = None,
    host: str = DEFAULT_BIND,
    port: int = 0,
    allow_non_loopback: bool = False,
) -> _ServiceHTTPServer:
    """Build (and bind) the local service. The bind
    configuration is validated BEFORE any socket is
    bound: a non-loopback host without
    ``allow_non_loopback=True`` raises ValueError
    naming the gate, and nothing is bound."""
    problems = bind_problems(host, allow_non_loopback)
    if problems:
        raise ValueError("; ".join(problems))
    server = _ServiceHTTPServer((host, port), _Handler)
    server.service = service if service is not None \
        else ThoughtForgeService()
    server.bind_host = host
    server.non_loopback_acknowledged = bool(
        allow_non_loopback and not is_loopback_host(host))
    return server


def serve(
    service: ThoughtForgeService | None = None,
    host: str = DEFAULT_BIND,
    port: int = 8765,
    allow_non_loopback: bool = False,
) -> int:
    """Serve forever (the local-service entry point)."""
    server = make_server(service, host, port, allow_non_loopback)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# ── Round-trip helpers (checker + probes) ────────────────────────────────────

def _knowledge_only_service() -> ThoughtForgeService:
    """A service over a knowledge-only core — the
    TF-084 fixture shape, in a fresh temp home."""
    from thoughtforge.cognition.core import ThoughtForgeCore

    tmp = tempfile.mkdtemp(prefix="tf153-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    return ThoughtForgeService(core=core)


def _http_json(
    url: str, payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _http_round_trip(
    service: ThoughtForgeService,
) -> dict[str, tuple[int, dict[str, Any]]]:
    """A REAL loopback round-trip: ephemeral port,
    in-process thread, urllib client. Returns the
    (status, response dict) per route."""
    server = make_server(service, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        return {
            "health": _http_json(f"{base}/health"),
            "contract": _http_json(f"{base}/contract"),
            "turn": _http_json(
                f"{base}/turn",
                make_request("turn", user_text="What time does "
                                               "the ferry leave?")),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ── The filed probes ─────────────────────────────────────────────────────────

SERVICE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "library_health_ok", "kind": "library_health",
     "detail": "a library handle() health request returns "
               "ok with the filed op ids"},
    {"id": "library_contract_is_tf034", "kind": "library_contract",
     "detail": "a library contract request returns TF-034's "
               "api_manifest() verbatim and this service "
               "contract at TF-034's API_VERSION"},
    {"id": "library_turn_knowledge_only", "kind": "library_turn",
     "detail": "a library turn against a knowledge-only "
               "core returns ok with the FinalResponseRecord "
               "fields (text / ids / mode / enforcement / "
               "citations)"},
    {"id": "http_equals_library", "kind": "http_equals",
     "detail": "a real loopback HTTP round-trip returns "
               "response dicts identical to library handle() "
               "for health and contract, status 200"},
    {"id": "refusal_on_unknown_op", "kind": "unknown_op",
     "detail": "an unknown op returns a refusal dict naming "
               "it — handle() does not raise"},
    {"id": "refusal_on_bad_version_major", "kind": "bad_version",
     "detail": "a request on an incompatible contract major "
               "version returns a refusal dict naming the "
               "mismatch"},
    {"id": "non_loopback_requires_explicit_flag",
     "kind": "non_loopback_gate",
     "detail": "bind_problems() names a non-loopback host "
               "without the flag, clears it with the flag, "
               "and make_server() raises ValueError naming "
               "the gate before binding anything"},
)


def run_service_probes() -> list[dict[str, Any]]:
    """Run the filed probes; each returns its verdict."""
    results: list[dict[str, Any]] = []
    plain = ThoughtForgeService(
        turn_callable=lambda user_text: {  # never used by health/contract
            "text": "", "response_id": "", "turn_id": "",
            "mode": "", "enforcement_passed": False,
            "citations": [], "token_count": 0})
    for probe in SERVICE_PROBES:
        kind = probe["kind"]
        try:
            if kind == "library_health":
                resp = plain.handle(make_request("health"))
                passed = resp.get("ok") is True \
                    and resp.get("ops") == list(OP_IDS) \
                    and service_response_problems(resp) == []
            elif kind == "library_contract":
                resp = plain.handle(make_request("contract"))
                passed = resp.get("ok") is True \
                    and resp.get("manifest") == api_manifest() \
                    and resp.get("contract", {}).get(
                        "contract_version") == API_VERSION
            elif kind == "library_turn":
                resp = _knowledge_only_service().handle(make_request(
                    "turn", user_text="What time does the ferry "
                                      "leave?"))
                passed = resp.get("ok") is True \
                    and bool(resp.get("text")) \
                    and bool(resp.get("response_id")) \
                    and bool(resp.get("turn_id")) \
                    and resp.get("mode") == "knowledge_only" \
                    and isinstance(resp.get("citations"), list) \
                    and service_response_problems(resp) == []
            elif kind == "http_equals":
                trip = _http_round_trip(plain)
                passed = all(
                    trip[op][0] == 200
                    and trip[op][1] == plain.handle(make_request(op))
                    for op in ("health", "contract"))
            elif kind == "unknown_op":
                resp = plain.handle(make_request("explode"))
                passed = resp.get("ok") is False \
                    and "unknown op" in str(resp.get("error", ""))
            elif kind == "bad_version":
                req = make_request("health")
                req["contract_version"] = "99.0.0"
                resp = plain.handle(req)
                passed = resp.get("ok") is False \
                    and "major" in str(resp.get("error", ""))
            elif kind == "non_loopback_gate":
                named = bool(bind_problems("0.0.0.0", False))
                cleared = bind_problems("0.0.0.0", True) == []
                raised = False
                try:
                    make_server(plain, host="0.0.0.0")
                except ValueError as exc:
                    raised = "allow_non_loopback" in str(exc)
                passed = named and cleared and raised
            else:  # pragma: no cover — the checker names unknown kinds
                passed = False
        except Exception:
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the service-shape doc, verbatim) ────────────────────

def render_service_shape_markdown() -> str:
    """Render the service-shape doc deterministically
    — the exact text filed as
    docs/architecture/SERVICE_SHAPE.md."""
    lines: list[str] = []
    lines.append("# Service Shape — Library AND Local Service, "
                 "One Contract (TF-153)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/service_shape.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.service_shape`. The "
        "checker (`service_shape_problems()`) re-derives "
        "every claim live from the tree and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{SERVICE_SHAPE_FORMAT}`")
    lines.append(f"Request format: `{REQUEST_FORMAT}`")
    lines.append(f"Response format: `{RESPONSE_FORMAT}`")
    lines.append(
        f"Contract version: `{API_VERSION}` (TF-034's "
        "`API_VERSION` — consumed, not reinvented)")
    lines.append(f"Default bind: `{DEFAULT_BIND}` (loopback only)")
    lines.append(f"Ops: {len(SERVICE_OPS)}; routes: "
                 f"{len(SERVICE_ROUTES)}; rules: "
                 f"{len(SERVICE_RULES)}")
    lines.append(f"Probes: {len(SERVICE_PROBES)}/{len(SERVICE_PROBES)}")
    lines.append("")
    lines.append("## 1. One contract")
    lines.append("")
    lines.append(
        "A request is a plain JSON-safe dict: "
        '`{"format": "thoughtforge-service-request/1", '
        '"contract_version": "<TF-034 API_VERSION>", '
        '"op": "<op>", ...}`. A response is a plain '
        "JSON-safe dict: "
        '`{"format": "thoughtforge-service-response/1", '
        '"contract_version": ..., "op": ..., "ok": true, '
        "...}` — and a refusal is the same dict with "
        '`"ok": false` and an `"error"` naming the '
        "problem. The library caller and the HTTP caller "
        "send and receive these exact dicts: "
        "`ThoughtForgeService.handle(request)` is the "
        "one entry point, and the HTTP routes call it "
        "unchanged.")
    lines.append("")
    lines.append("## 2. The ops")
    lines.append("")
    lines.append("| Op | Route | What it does |")
    lines.append("|---|---|---|")
    for op in SERVICE_OPS:
        lines.append(f"| {op['id']} | {op['route']} | {op['summary']} |")
    lines.append("")
    lines.append(
        "The `turn` op's response fields are the real "
        "`FinalResponseRecord`'s: `text`, `response_id`, "
        "`turn_id`, `mode`, `enforcement_passed`, "
        "`citations`, `token_count`. TF-154 deepens the "
        "cognition API (traces, TF-084); TF-153 files "
        "only this shape.")
    lines.append("")
    lines.append("## 3. The two shapes")
    lines.append("")
    lines.append(
        "**Library**: `ThoughtForgeService(core)` — or "
        "an injected core factory / turn callable — and "
        "`handle(request) -> response`. The core is "
        "built lazily, so health and contract never "
        "build one.")
    lines.append("")
    lines.append(
        "**Local service**: `make_server(service, "
        'host="127.0.0.1", port=0)` — stdlib '
        "`http.server` only, no new dependency. Routes: "
        "`GET /health`, `GET /contract`, `POST /turn` "
        "(the JSON body is the same request dict; if "
        "`op` is omitted in the body, the route supplies "
        "it). HTTP status: 200 on ok, 400 on a refusal, "
        "404 on an unknown route, 405 on a wrong method "
        "— errors still arrive as the contract response "
        "dict where applicable.")
    lines.append("")
    lines.append("## 4. The rules")
    lines.append("")
    for rule in SERVICE_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## 5. Docket")
    lines.append("")
    lines.append(
        "TF-153 has **no share** of TF-016 and found no "
        "new defect in existing code, so this slice does "
        "not edit the docket and adds no entry. TF-016 "
        "totals stand unchanged (137 entries — 32 OPEN / "
        "1 OPEN-ENVIRONMENT / 100 FIXED / 4 "
        "CLOSED-BY-VERDICT), filed in the open here.")
    lines.append("")
    lines.append("## 6. The probes")
    lines.append("")
    for probe in SERVICE_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Proof grade FIXTURE: the checker and probes "
        "run a knowledge-only core (the TF-084 fixture "
        "shape) plus a real loopback HTTP round-trip on "
        "an ephemeral port on this VM. A live model turn "
        "through the service is NOT PROVEN, never "
        "claimed.")
    lines.append(
        "- Versioning enforcement in depth is TF-162's; "
        "local-only proof (the logged non-loopback "
        "record) is TF-163's. TF-153 files the shape, "
        "the major-mismatch refusal, the loopback "
        "default and the explicit gate.")
    lines.append(
        "- No canon data was touched (TF-017); no "
        "README.md anywhere was touched; no Docker was "
        "used.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_service_shape_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / SERVICE_SHAPE_DOC
    doc.write_text(render_service_shape_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.service_shape`."""
    doc = write_service_shape_doc()
    print(f"service shape doc regenerated: -> {SERVICE_SHAPE_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def service_shape_problems() -> list[str]:
    """Re-derive TF-153's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    op_ids = [op["id"] for op in SERVICE_OPS]
    if op_ids != ["health", "contract", "turn"] \
            or len(op_ids) != len(set(op_ids)):
        problems.append(
            f"the SERVICE_OPS filing is not health/contract/turn: {op_ids}")
    for op in SERVICE_OPS:
        problems.extend(service_op_problems(op))
    rule_ids = [r["id"] for r in SERVICE_RULES]
    if len(rule_ids) != len(set(rule_ids)) or len(SERVICE_RULES) != 6:
        problems.append("the SERVICE_RULES filing is not six unique rules")
    for rule in SERVICE_RULES:
        problems.extend(service_rule_problems(rule))
    for route in SERVICE_ROUTES:
        problems.extend(service_route_problems(route))
    if len(SERVICE_ROUTES) != 3:
        problems.append("the SERVICE_ROUTES filing is not three routes")

    # The version is TF-034's, and its surface is clean.
    contract = service_contract()
    if contract["contract_version"] != API_VERSION \
            or API_VERSION != api_manifest().get("version"):
        problems.append(
            "the filed contract_version is not TF-034's API_VERSION")
    surface = verify_api_surface()
    if surface:
        problems.append(
            f"TF-034: verify_api_surface() is not clean: {surface[:3]}")

    # The default bind is loopback, filed as data.
    if DEFAULT_BIND != "127.0.0.1" or not is_loopback_host(DEFAULT_BIND) \
            or contract["default_bind"] != DEFAULT_BIND:
        problems.append(
            f"the default bind is {DEFAULT_BIND!r}, not filed "
            "loopback 127.0.0.1")
    if bind_problems("0.0.0.0", False) == [] \
            or bind_problems("0.0.0.0", True) != []:
        problems.append("the non-loopback gate does not judge as filed")

    # A library round-trip and a REAL loopback HTTP
    # round-trip agree on health and contract; a turn
    # against a knowledge-only core answers in shape.
    service = _knowledge_only_service()
    for op in ("health", "contract"):
        resp = service.handle(make_request(op))
        problems.extend(
            f"library {op}: {p}" for p in service_response_problems(resp))
        if resp.get("ok") is not True:
            problems.append(f"library {op}: not ok: {resp}")
    try:
        trip = _http_round_trip(service)
        for op in ("health", "contract"):
            status, resp = trip[op]
            if status != 200 or resp != service.handle(make_request(op)):
                problems.append(
                    f"http {op}: status {status} / response differs "
                    "from the library handle() response — one "
                    "contract, both shapes")
        status, turn_resp = trip["turn"]
        if status != 200 or turn_resp.get("ok") is not True \
                or turn_resp.get("mode") != "knowledge_only" \
                or not turn_resp.get("text"):
            problems.append(
                f"http turn: status {status}, response not an ok "
                f"knowledge-only turn: {turn_resp}")
        problems.extend(
            f"http turn: {p}"
            for p in service_response_problems(turn_resp))
    except Exception as exc:
        problems.append(f"http round-trip failed: {exc}")

    # The probes all behave.
    probes = run_service_probes()
    if len(probes) != len(SERVICE_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the service probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The docket stands unchanged.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, 100 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 32 OPEN / 1 OPEN-ENVIRONMENT / 100 "
            "FIXED; TF-153 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / SERVICE_SHAPE_DOC
    if not doc.exists():
        problems.append(f"the service doc {SERVICE_SHAPE_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_service_shape_markdown():
        problems.append(
            f"the filed doc {SERVICE_SHAPE_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.service_shape)")
    return problems


def service_shape_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_service_probes()
    return {
        "format": SERVICE_SHAPE_FORMAT,
        "doc": SERVICE_SHAPE_DOC,
        "contract": service_contract(),
        "manifest": api_manifest(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "DEFAULT_BIND",
    "OP_IDS",
    "REQUEST_FORMAT",
    "RESPONSE_FORMAT",
    "SERVICE_OPS",
    "SERVICE_PROBES",
    "SERVICE_ROUTES",
    "SERVICE_RULES",
    "SERVICE_SHAPE_DOC",
    "SERVICE_SHAPE_FORMAT",
    "ThoughtForgeService",
    "bind_problems",
    "is_loopback_host",
    "main",
    "make_request",
    "make_server",
    "render_service_shape_markdown",
    "run_service_probes",
    "serve",
    "service_contract",
    "service_op_problems",
    "service_response_problems",
    "service_route_problems",
    "service_rule_problems",
    "service_shape_as_data",
    "service_shape_problems",
    "write_service_shape_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
