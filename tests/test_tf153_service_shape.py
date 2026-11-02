"""
TF-153 — Service shape: ThoughtForge callable as
a library AND as a local service (loopback only
by default) — one contract.

The proof lives as verifiable data in
``src/thoughtforge/service_shape.py``
(``SERVICE_OPS``, ``SERVICE_ROUTES``,
``SERVICE_RULES``, ``SERVICE_PROBES``, checked
by ``service_shape_problems()``); the doc
``docs/architecture/SERVICE_SHAPE.md`` is filed
verbatim as the module's render.

Pinned here:

  * one contract for both shapes: the library
    ``ThoughtForgeService.handle()`` and a REAL
    loopback HTTP round-trip (ephemeral port)
    return identical response dicts for health
    and contract,
  * the contract version is TF-034's
    ``API_VERSION`` (consumed, never reinvented)
    and the contract op returns TF-034's
    ``api_manifest()``,
  * the turn op against a knowledge-only core
    (the TF-084 fixture shape) returns the real
    FinalResponseRecord fields,
  * refusals are data, never raises: not a
    dict, wrong / missing format, missing
    version, unknown op, missing user_text, and
    an incompatible contract major version,
  * loopback by default: ``DEFAULT_BIND`` is
    127.0.0.1, and a non-loopback bind without
    the explicit flag is refused before any
    socket is bound,
  * HTTP status: 200 ok / 400 refusal / 404
    unknown route / 405 wrong method,
  * every response is JSON-safe,
  * the service-shape doc matches its render
    verbatim, the checker is clean, the probes
    pass 7/7, and TF-016's docket totals are
    unchanged (TF-153 has no docket share).

Proof grade: FIXTURE — knowledge-only core +
real loopback HTTP on this VM; a live model
turn through the service is NOT PROVEN, never
claimed. All fixtures synthetic (TF-017); no
canon.
"""

from __future__ import annotations

import http.client
import json
import re
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.api_surface import (
    API_VERSION,
    api_manifest,
    verify_api_surface,
)
from thoughtforge.service_shape import (
    DEFAULT_BIND,
    OP_IDS,
    REQUEST_FORMAT,
    RESPONSE_FORMAT,
    SERVICE_OPS,
    SERVICE_PROBES,
    SERVICE_ROUTES,
    SERVICE_RULES,
    SERVICE_SHAPE_DOC,
    SERVICE_SHAPE_FORMAT,
    ThoughtForgeService,
    bind_problems,
    is_loopback_host,
    make_request,
    make_server,
    render_service_shape_markdown,
    run_service_probes,
    service_contract,
    service_op_problems,
    service_response_problems,
    service_route_problems,
    service_rule_problems,
    service_shape_as_data,
    service_shape_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _stub_service() -> ThoughtForgeService:
    def turn(user_text: str):
        return SimpleNamespace(
            text=f"stub answer to: {user_text}",
            response_id="resp_stub", turn_id="turn_stub",
            mode="stub_mode", enforcement_passed=True,
            citations=["frag-1"], token_count=4)

    return ThoughtForgeService(turn_callable=turn)


def _knowledge_service() -> ThoughtForgeService:
    from thoughtforge.cognition.core import ThoughtForgeCore

    tmp = tempfile.mkdtemp(prefix="tf153-test-")
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    return ThoughtForgeService(core=core)


class _Server:
    """A real loopback server on an ephemeral port."""

    def __enter__(self):
        self.server = make_server(_stub_service(), host="127.0.0.1", port=0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method: str, path: str, payload=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = raw if raw is not None else (
            json.dumps(payload) if payload is not None else None)
        conn.request(method, path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, data


# ── The filed estate ─────────────────────────────────────────────────────────


class TestFiling:
    def test_problems_empty(self) -> None:
        assert service_shape_problems() == []

    def test_formats_are_versioned(self) -> None:
        assert SERVICE_SHAPE_FORMAT == "thoughtforge-service-shape/1"
        assert REQUEST_FORMAT == "thoughtforge-service-request/1"
        assert RESPONSE_FORMAT == "thoughtforge-service-response/1"

    def test_ops_are_three_unique_and_complete(self) -> None:
        assert OP_IDS == ("health", "contract", "turn")
        assert len({op["id"] for op in SERVICE_OPS}) == 3
        for op in SERVICE_OPS:
            assert service_op_problems(op) == []
        turn = next(op for op in SERVICE_OPS if op["id"] == "turn")
        assert "user_text" in turn["request_fields"]
        for field in ("text", "response_id", "turn_id", "mode",
                      "enforcement_passed", "citations"):
            assert field in turn["response_fields"]

    def test_rules_are_six_unique(self) -> None:
        ids = [r["id"] for r in SERVICE_RULES]
        assert len(ids) == len(set(ids)) == 6
        assert set(ids) == {
            "one_contract_both_shapes", "loopback_by_default",
            "non_loopback_is_explicit",
            "refusals_are_data_never_raises",
            "contract_version_is_tf034",
            "turn_fields_from_final_response"}
        for rule in SERVICE_RULES:
            assert service_rule_problems(rule) == []

    def test_routes_match_ops(self) -> None:
        assert {(r["method"], r["path"], r["op"])
                for r in SERVICE_ROUTES} == {
            ("GET", "/health", "health"),
            ("GET", "/contract", "contract"),
            ("POST", "/turn", "turn")}
        for route in SERVICE_ROUTES:
            assert service_route_problems(route) == []

    def test_validators_are_empty_safe(self) -> None:
        assert service_op_problems(None)
        assert service_rule_problems(None)
        assert service_route_problems(None)
        assert service_response_problems(None)
        assert bind_problems("")

    def test_contract_data_is_tf034_versioned(self) -> None:
        contract = service_contract()
        assert contract["contract_version"] == API_VERSION
        assert contract["default_bind"] == DEFAULT_BIND == "127.0.0.1"
        assert contract["request_format"] == REQUEST_FORMAT
        assert json.loads(json.dumps(contract)) == contract
        assert verify_api_surface() == []
        data = service_shape_as_data()
        assert data["format"] == SERVICE_SHAPE_FORMAT
        assert data["doc"] == SERVICE_SHAPE_DOC
        assert data["manifest"] == api_manifest()
        assert data["measured"] == {"probes_passed": 7, "probes_total": 7}


# ── The library shape ────────────────────────────────────────────────────────


class TestLibrary:
    def test_health(self) -> None:
        resp = _stub_service().handle(make_request("health"))
        assert resp["ok"] is True
        assert resp["status"] == "ok"
        assert resp["ops"] == list(OP_IDS)
        assert service_response_problems(resp) == []

    def test_contract_returns_tf034_manifest(self) -> None:
        resp = _stub_service().handle(make_request("contract"))
        assert resp["ok"] is True
        assert resp["manifest"] == api_manifest()
        assert resp["manifest"]["version"] == API_VERSION
        assert resp["contract"]["contract_version"] == API_VERSION

    def test_turn_with_injected_callable(self) -> None:
        resp = _stub_service().handle(
            make_request("turn", user_text="hello forge"))
        assert resp["ok"] is True
        assert resp["text"] == "stub answer to: hello forge"
        assert resp["response_id"] == "resp_stub"
        assert resp["turn_id"] == "turn_stub"
        assert resp["mode"] == "stub_mode"
        assert resp["enforcement_passed"] is True
        assert resp["citations"] == ["frag-1"]
        assert resp["token_count"] == 4

    def test_turn_with_injected_core_factory_is_lazy(self) -> None:
        built: list[int] = []

        def factory():
            built.append(1)
            return _knowledge_service()._core

        service = ThoughtForgeService(core_factory=factory)
        assert service.handle(make_request("health"))["ok"] is True
        assert built == []  # health never builds the core
        resp = service.handle(make_request("turn", user_text="hello?"))
        assert resp["ok"] is True and built == [1]

    def test_turn_knowledge_only_core(self) -> None:
        resp = _knowledge_service().handle(make_request(
            "turn", user_text="What time does the ferry leave?"))
        assert resp["ok"] is True
        assert resp["mode"] == "knowledge_only"
        assert resp["text"]
        assert resp["response_id"] and resp["turn_id"]
        assert isinstance(resp["citations"], list)
        assert isinstance(resp["enforcement_passed"], bool)
        assert service_response_problems(resp) == []

    def test_every_response_is_json_safe(self) -> None:
        service = _stub_service()
        requests = [
            make_request("health"), make_request("contract"),
            make_request("turn", user_text="hi"),
            make_request("nope"), "not a dict", {},
            make_request("turn"),
        ]
        for req in requests:
            resp = service.handle(req)
            assert json.loads(json.dumps(resp)) == resp


# ── Refusals are data, never raises ──────────────────────────────────────────


class TestRefusals:
    def test_not_a_dict(self) -> None:
        for bad in ("text", 42, ["health"], None):
            resp = _stub_service().handle(bad)
            assert resp["ok"] is False
            assert "not a dict" in resp["error"]
            assert service_response_problems(resp) == []

    def test_wrong_and_missing_format(self) -> None:
        req = make_request("health")
        req["format"] = "something-else/9"
        resp = _stub_service().handle(req)
        assert resp["ok"] is False and "format" in resp["error"]
        req = make_request("health")
        del req["format"]
        assert _stub_service().handle(req)["ok"] is False

    def test_unknown_op(self) -> None:
        resp = _stub_service().handle(make_request("explode"))
        assert resp["ok"] is False
        assert "unknown op" in resp["error"] and "explode" in resp["error"]

    def test_missing_op(self) -> None:
        req = make_request("health")
        del req["op"]
        assert _stub_service().handle(req)["ok"] is False

    def test_turn_missing_user_text(self) -> None:
        for req in (make_request("turn"),
                    make_request("turn", user_text=""),
                    make_request("turn", user_text=42)):
            resp = _stub_service().handle(req)
            assert resp["ok"] is False
            assert "user_text" in resp["error"]

    def test_turn_failure_is_data(self) -> None:
        def boom(user_text: str):
            raise RuntimeError("engine gone")

        resp = ThoughtForgeService(turn_callable=boom).handle(
            make_request("turn", user_text="hi"))
        assert resp["ok"] is False
        assert "engine gone" in resp["error"]


class TestVersionGate:
    def test_bad_major_is_refused(self) -> None:
        req = make_request("health")
        req["contract_version"] = "2.0.0"
        resp = _stub_service().handle(req)
        assert resp["ok"] is False
        assert "major" in resp["error"] and "2.0.0" in resp["error"]

    def test_missing_and_unparseable_version_refused(self) -> None:
        req = make_request("health")
        del req["contract_version"]
        assert _stub_service().handle(req)["ok"] is False
        req = make_request("health")
        req["contract_version"] = "banana"
        resp = _stub_service().handle(req)
        assert resp["ok"] is False and "parse" in resp["error"]

    def test_same_major_minor_difference_is_accepted(self) -> None:
        req = make_request("health")
        req["contract_version"] = "1.99.0"
        assert _stub_service().handle(req)["ok"] is True


# ── The service shape (real loopback HTTP) ───────────────────────────────────


class TestHttp:
    def test_http_equals_library_for_health_and_contract(self) -> None:
        service = _stub_service()
        with _Server() as srv:
            srv.server.service = service
            for path, op in (("/health", "health"),
                             ("/contract", "contract")):
                status, resp = srv.request("GET", path)
                assert status == 200
                assert resp == service.handle(make_request(op))

    def test_http_turn_route_supplies_op(self) -> None:
        with _Server() as srv:
            body = make_request("turn", user_text="hello forge")
            del body["op"]  # the route supplies it
            status, resp = srv.request("POST", "/turn", payload=body)
            assert status == 200
            assert resp["ok"] is True and resp["op"] == "turn"
            assert resp["text"] == "stub answer to: hello forge"
            # And with the op carried in the body, as filed.
            status, resp = srv.request(
                "POST", "/turn",
                payload=make_request("turn", user_text="hi"))
            assert status == 200 and resp["ok"] is True

    def test_http_refusal_is_400_with_contract_dict(self) -> None:
        with _Server() as srv:
            status, resp = srv.request(
                "POST", "/turn", payload=make_request("turn"))
            assert status == 400
            assert resp["ok"] is False and "user_text" in resp["error"]
            assert resp["format"] == RESPONSE_FORMAT

    def test_http_unknown_route_is_404(self) -> None:
        with _Server() as srv:
            status, resp = srv.request("GET", "/nope")
            assert status == 404
            assert resp["ok"] is False and "unknown route" in resp["error"]

    def test_http_wrong_method_is_405(self) -> None:
        with _Server() as srv:
            status, resp = srv.request(
                "POST", "/health", payload=make_request("health"))
            assert status == 405 and resp["ok"] is False
            status, resp = srv.request("GET", "/turn")
            assert status == 405 and resp["ok"] is False
            status, resp = srv.request("PUT", "/health", payload={})
            assert status == 405 and resp["ok"] is False

    def test_http_invalid_json_body_is_400(self) -> None:
        with _Server() as srv:
            status, resp = srv.request("POST", "/turn", raw="{not json")
            assert status == 400
            assert resp["ok"] is False and "JSON" in resp["error"]

    def test_http_knowledge_only_turn(self) -> None:
        server = make_server(_knowledge_service(), host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/turn",
                data=json.dumps(make_request(
                    "turn", user_text="What time does the ferry leave?")
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                assert resp.status == 200
            assert data["ok"] is True
            assert data["mode"] == "knowledge_only"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


# ── Loopback by default ──────────────────────────────────────────────────────


class TestBindGate:
    def test_default_bind_is_loopback(self) -> None:
        assert DEFAULT_BIND == "127.0.0.1"
        assert is_loopback_host(DEFAULT_BIND)
        assert is_loopback_host("::1") and is_loopback_host("localhost")
        assert is_loopback_host("127.0.0.53")
        assert not is_loopback_host("0.0.0.0")
        assert not is_loopback_host("192.168.1.10")
        assert bind_problems(DEFAULT_BIND) == []

    def test_non_loopback_requires_explicit_flag(self) -> None:
        assert bind_problems("0.0.0.0")
        assert "allow_non_loopback" in bind_problems("0.0.0.0")[0]
        assert bind_problems("0.0.0.0", allow_non_loopback=True) == []
        with pytest.raises(ValueError, match="allow_non_loopback"):
            make_server(_stub_service(), host="0.0.0.0")

    def test_loopback_server_records_no_acknowledgement(self) -> None:
        server = make_server(_stub_service(), host="127.0.0.1", port=0)
        try:
            assert server.non_loopback_acknowledged is False
            assert server.bind_host == "127.0.0.1"
            assert server.server_address[1] > 0  # ephemeral port bound
        finally:
            server.server_close()


# ── State, probes, doc, docket ───────────────────────────────────────────────


class TestStateProbesDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_service_probes()
        assert len(probes) == len(SERVICE_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_doc_matches_render(self) -> None:
        assert _read(SERVICE_SHAPE_DOC) == render_service_shape_markdown()
        doc = _read(SERVICE_SHAPE_DOC)
        assert "One Contract" in doc
        assert "127.0.0.1" in doc
        assert "NOT PROVEN" in doc
        assert API_VERSION in doc
        assert "TF-154" in doc and "TF-162" in doc and "TF-163" in doc

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized
