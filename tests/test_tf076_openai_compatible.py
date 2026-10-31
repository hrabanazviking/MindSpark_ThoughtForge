"""
TF-076 — OpenAI-compatible backend: any server speaking
the common shape binds under the same (TF-065) contract.

The binding lives in
``src/thoughtforge/inference/openai_compatible.py``
(``OpenAICompatibleBackend`` + ``OPENAI_SHAPE`` /
``OPENAI_RULES`` / ``KNOWN_SERVERS`` as data, the shared
family parser ``chat_completion_outcome``, format
``thoughtforge-openai-compatible-backend/1``).
Write-up: ``docs/architecture/OPENAI_COMPATIBLE_BACKEND.md``.

Measured start: the shape was reachable only as a side
effect of LMStudioBackend (its docstring's claim + the
factory's ``openai_compatible`` alias), and probes of
that leg found D-OAI-1 (finish_reason passthrough
outside the contract vocabulary + content:null landing
as text=None + the cryptic empty-choices index error),
D-OAI-2 (a /v1-ending base URL doubled to
/v1/v1/chat/completions), and D-OAI-3 (no usage block
-> tokens_generated 0 over real text) — all FIXED in
scope in both legs through the one shared parser.

Deliberate in-scope amendments: ``REQUESTS_CONSUMERS``
grew to six in ``backend_contract.py`` (this module is
an announced requests consumer) and the TF-065 test pin
for its length was updated; the TF-071 lmstudio
counting-surface note was amended for the D-OAI-3
fallback. ``SHIPPED_BACKENDS`` is NOT amended — the
factory set stays TF-065's four; TF-153 owns admitting
new backends there.

Pinned here:

  * the filed estate: shape/rule/server ids + statuses
    (every known server stays shape-only), both checkers
    clean, TF-065 conformance of the new backend judged
    by the contract's own checker,
  * the wire: payload keys exactly the shape's sent set;
    seed/stop/model present only when carried or
    configured; the Authorization header only when a
    key is configured,
  * base normalisation: bare, /v1, and /v1/ bases all
    produce the one route URL (D-OAI-2),
  * responses: stop/length pass; content_filter and
    tool_calls are named error data; null content is a
    named error, never a None text; no-choices is a
    named error (D-OAI-1),
  * tokens: usage count when reported, the declared
    contract estimate otherwise (D-OAI-3),
  * errors: both server error shapes parsed with the
    status named; a raising transport becomes error
    data; generate() never raises,
  * health: exactly 200 is True; 500 and a raising
    transport are False; model_listed answers True /
    False / None separately,
  * the LM Studio preset: the shared parser/normaliser
    identity pinned; D-OAI-1/2/3 regression pins on the
    shipped leg; payload parity between the two legs,
  * a real HTTP round trip against a stdlib fixture
    server on loopback (generate + health + listing),
  * token counting: the contract estimate, equal to the
    other backends on the TF-071 corpus,
  * the core leg: a core holding the generic backend
    over a fake transport answers the turn through it.
"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.backend_contract import (
    REQUESTS_CONSUMERS,
    backend_contract_problems,
    estimate_tokens,
)
from thoughtforge.inference.lmstudio_backend import LMStudioBackend
from thoughtforge.inference.openai_compatible import (
    KNOWN_SERVERS,
    OPENAI_RULES,
    OPENAI_SHAPE,
    FakeOpenAIServer,
    OpenAICompatibleBackend,
    chat_completion_outcome,
    normalize_base_url,
    openai_as_data,
    openai_live_problems,
    openai_problems,
    server_error_message,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    UnifiedBackend,
)

POST = "thoughtforge.inference.openai_compatible.requests.post"
GET = "thoughtforge.inference.openai_compatible.requests.get"
LM_POST = "thoughtforge.inference.lmstudio_backend.requests.post"


def capture(backend, server, request=None):
    with patch(POST, side_effect=server.post):
        resp = backend.generate(request or GenerationRequest(
            prompt="What is two plus two?"))
    return resp


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert openai_problems() == []
        assert openai_live_problems() == []

    def test_as_data_shape(self):
        data = openai_as_data()
        assert data["format"] == \
            "thoughtforge-openai-compatible-backend/1"
        assert len(data["shape"]) == len(OPENAI_SHAPE) == 5
        assert len(data["rules"]) == len(OPENAI_RULES) == 10
        assert len(data["known_servers"]) == \
            len(KNOWN_SERVERS) == 6
        assert data["served_finish_reasons"] == ["stop", "length"]

    def test_known_servers_stay_shape_only(self):
        for row in KNOWN_SERVERS:
            assert row["status"] == "shape_bound_not_live_proven"

    def test_announced_requests_consumer(self):
        assert "src/thoughtforge/inference/openai_compatible.py" \
            in REQUESTS_CONSUMERS
        assert len(REQUESTS_CONSUMERS) == 6

    def test_contract_checker_clean_over_the_new_backend(self):
        assert backend_contract_problems(
            [OpenAICompatibleBackend()]) == []

    def test_is_a_unified_backend_with_simulated_streaming(self):
        backend = OpenAICompatibleBackend()
        assert isinstance(backend, UnifiedBackend)
        assert (OpenAICompatibleBackend.generate_stream
                is UnifiedBackend.generate_stream)
        caps = backend.capabilities()
        assert caps.streaming == "simulated"
        assert caps.token_counting == "estimate"
        assert caps.seed == "honoured"
        assert caps.placement == "local_or_remote"


# ── The wire ───────────────────────────────────────────────────


class TestWire:
    def test_payload_keys_exact_with_all_optionals(self):
        server = FakeOpenAIServer()
        backend = OpenAICompatibleBackend(model="fake-model")
        capture(backend, server, GenerationRequest(
            prompt="hi", seed=7, stop=["\n\n"]))
        payload = server.calls[0]["json"]
        assert set(payload) == {
            "model", "messages", "temperature", "max_tokens",
            "stream", "stop", "seed"}
        assert payload["stream"] is False
        assert payload["seed"] == 7
        assert payload["stop"] == ["\n\n"]

    def test_payload_omits_uncarried_optionals(self):
        server = FakeOpenAIServer()
        capture(OpenAICompatibleBackend(), server)
        payload = server.calls[0]["json"]
        assert set(payload) == {
            "messages", "temperature", "max_tokens", "stream"}

    def test_auth_header_only_when_configured(self):
        server = FakeOpenAIServer()
        capture(OpenAICompatibleBackend(), server)
        assert "Authorization" not in server.calls[0]["headers"]
        server2 = FakeOpenAIServer()
        capture(OpenAICompatibleBackend(api_key="k-1"), server2)
        assert server2.calls[0]["headers"]["Authorization"] == \
            "Bearer " + "k-1"

    @pytest.mark.parametrize("base", [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/v1/",
    ])
    def test_base_v1_never_doubled(self, base):
        server = FakeOpenAIServer()
        capture(OpenAICompatibleBackend(base_url=base), server)
        assert server.calls[0]["url"] == \
            "http://127.0.0.1:8000/v1/chat/completions"
        assert normalize_base_url(base) == "http://127.0.0.1:8000"

    def test_backend_name_identity(self):
        assert OpenAICompatibleBackend(model="m").backend_name() \
            == "openai-compatible:m"
        assert OpenAICompatibleBackend(
            base_url="http://127.0.0.1:8000/v1").backend_name() \
            == "openai-compatible:127.0.0.1:8000"


# ── Responses: the parser's rules (D-OAI-1 / D-OAI-3) ─────────


class TestResponses:
    def test_stop_and_length_pass(self):
        for finish in ("stop", "length"):
            server = FakeOpenAIServer(finish=finish)
            resp = capture(OpenAICompatibleBackend(), server)
            assert resp.finish_reason == finish
            assert resp.text == "Four"
            assert resp.tokens_generated == 1

    @pytest.mark.parametrize("finish", [
        "content_filter", "tool_calls", "function_call"])
    def test_other_finishes_are_named_error_data(self, finish):
        server = FakeOpenAIServer(finish=finish)
        resp = capture(OpenAICompatibleBackend(), server)
        assert resp.finish_reason == "error"
        assert resp.text == ""
        assert finish in resp.error

    def test_null_content_is_named_error_never_none_text(self):
        server = FakeOpenAIServer(chat_payload={
            "choices": [{"message": {"role": "assistant",
                                     "content": None},
                         "finish_reason": "stop"}]})
        resp = capture(OpenAICompatibleBackend(), server)
        assert resp.finish_reason == "error"
        assert resp.text == ""
        assert "NoneType" in resp.error

    def test_no_choices_is_named_error(self):
        for payload in ({"choices": []}, {"id": "x"}):
            server = FakeOpenAIServer(chat_payload=payload)
            resp = capture(OpenAICompatibleBackend(), server)
            assert resp.finish_reason == "error"
            assert "no choices" in resp.error

    def test_no_usage_falls_back_to_declared_estimate(self):
        server = FakeOpenAIServer(text="the ferry", usage=False)
        backend = OpenAICompatibleBackend()
        resp = capture(backend, server)
        assert resp.tokens_generated == \
            backend.count_tokens("the ferry") == \
            estimate_tokens("the ferry")
        assert resp.tokens_generated > 0

    def test_parser_directly(self):
        outcome = chat_completion_outcome({
            "choices": [{"message": {"content": "x"},
                         "finish_reason": "length"}],
            "usage": {"completion_tokens": 5}})
        assert outcome == {"ok": True, "text": "x",
                           "finish_reason": "length",
                           "completion_tokens": 5}
        assert chat_completion_outcome("nope")["ok"] is False


# ── Errors are data, with the server's reason ──────────────────


class _StatusResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class TestErrors:
    @pytest.mark.parametrize("payload", [
        {"error": {"message": "bad key", "code": "invalid"}},
        {"error": "bad key"},
    ])
    def test_http_error_names_status_and_server_message(self, payload):
        backend = OpenAICompatibleBackend()
        with patch(POST, return_value=_StatusResponse(401, payload)):
            resp = backend.generate(GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error"
        assert "401" in resp.error and "bad key" in resp.error

    def test_server_error_message_shapes(self):
        assert server_error_message(
            _StatusResponse(400, {"error": {"message": "m1"}})) == "m1"
        assert server_error_message(
            _StatusResponse(400, {"error": "m2"})) == "m2"
        assert server_error_message(
            _StatusResponse(400, {})) == ""

    def test_raising_transport_becomes_error_data(self):
        backend = OpenAICompatibleBackend()
        with patch(POST, side_effect=ConnectionError("down")):
            resp = backend.generate(GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error"
        assert "down" in resp.error


# ── Health + listing, separately ───────────────────────────────


class TestHealthAndListing:
    def test_health_exactly_200(self):
        server = FakeOpenAIServer()
        backend = OpenAICompatibleBackend(model="fake-model")
        with patch(GET, side_effect=server.get):
            assert backend.health_check() is True
        down = FakeOpenAIServer(models_status=500)
        with patch(GET, side_effect=down.get):
            assert backend.health_check() is False
        with patch(GET, side_effect=ConnectionError("down")):
            assert backend.health_check() is False

    def test_model_listed_variants(self):
        server = FakeOpenAIServer()
        with patch(GET, side_effect=server.get):
            assert OpenAICompatibleBackend(
                model="fake-model").model_listed() is True
            assert OpenAICompatibleBackend(
                model="absent").model_listed() is False
            assert OpenAICompatibleBackend().model_listed() is None
        with patch(GET, side_effect=ConnectionError("down")):
            assert OpenAICompatibleBackend(
                model="fake-model").model_listed() is None

    def test_list_models_filters_malformed(self):
        server = FakeOpenAIServer()
        backend = OpenAICompatibleBackend()
        with patch(GET, side_effect=server.get):
            assert backend.list_models() == ["fake-model"]


# ── The LM Studio preset: shared parser + regressions ──────────


class TestLMStudioPreset:
    def test_shared_parser_and_normaliser_identity(self):
        from thoughtforge.inference import lmstudio_backend
        assert lmstudio_backend.chat_completion_outcome is \
            chat_completion_outcome
        assert lmstudio_backend.normalize_base_url is \
            normalize_base_url

    def test_lmstudio_base_v1_not_doubled(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return _StatusResponse(200, {
                "choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}})

        backend = LMStudioBackend(base_url="http://host:9/v1")
        with patch(LM_POST, side_effect=fake_post):
            resp = backend.generate(GenerationRequest(prompt="hi"))
        assert captured["url"] == "http://host:9/v1/chat/completions"
        assert resp.text == "ok"

    def test_lmstudio_finish_and_null_content_fixed(self):
        for payload, needle in (
            ({"choices": [{"message": {"content": "p"},
                           "finish_reason": "content_filter"}]},
             "content_filter"),
            ({"choices": [{"message": {"content": None},
                           "finish_reason": "stop"}]},
             "NoneType"),
            ({"choices": []}, "no choices"),
        ):
            backend = LMStudioBackend()
            with patch(LM_POST, return_value=_StatusResponse(
                    200, payload)):
                resp = backend.generate(
                    GenerationRequest(prompt="hi"))
            assert resp.finish_reason == "error"
            assert needle in resp.error
            assert resp.text == ""

    def test_lmstudio_no_usage_estimate_fallback(self):
        backend = LMStudioBackend()
        with patch(LM_POST, return_value=_StatusResponse(200, {
                "choices": [{"message": {"content": "the ferry"},
                             "finish_reason": "stop"}]})):
            resp = backend.generate(GenerationRequest(prompt="hi"))
        assert resp.tokens_generated == \
            backend.count_tokens("the ferry") > 0

    def test_payload_parity_between_the_legs(self):
        req = GenerationRequest(prompt="hi", seed=3, stop=["x"])
        generic_server = FakeOpenAIServer()
        capture(OpenAICompatibleBackend(
            base_url="http://h:1", model="m",
            api_key="lm-studio"), generic_server, req)
        lm_calls = {}

        def fake_post(url, **kwargs):
            lm_calls.update(kwargs, url=url)
            return _StatusResponse(200, {
                "choices": [{"message": {"content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}})

        with patch(LM_POST, side_effect=fake_post):
            LMStudioBackend(base_url="http://h:1", model="m").generate(req)
        assert lm_calls["json"] == generic_server.calls[0]["json"]
        assert lm_calls["url"] == generic_server.calls[0]["url"]


# ── A real HTTP round trip (loopback, stdlib fixture server) ───


class _ShapeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = {"object": "list",
                    "data": [{"id": "fixture-model"}]}
            self._send(200, body)
        else:
            self._send(404, {"error": {"message": "no route"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            self._send(200, {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "Four, over real HTTP."},
                    "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4}})
        else:
            self._send(404, {"error": {"message": "no route"}})

    def _send(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class TestRealHttpRoundTrip:
    def test_generate_health_and_listing_over_real_http(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ShapeHandler)
        thread = threading.Thread(target=server.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            backend = OpenAICompatibleBackend(
                base_url=f"http://127.0.0.1:{port}/v1",
                model="fixture-model")
            resp = backend.generate(
                GenerationRequest(prompt="two plus two?"))
            assert resp.text == "Four, over real HTTP."
            assert resp.tokens_generated == 4
            assert backend.health_check() is True
            assert backend.model_listed() is True
            assert backend.list_models() == ["fixture-model"]
        finally:
            server.shutdown()
            server.server_close()


# ── Token counting + the core leg ──────────────────────────────


class TestCountingAndCore:
    def test_count_is_the_contract_estimate(self):
        from thoughtforge.inference.tokenizer_discipline import (
            CORPUS,
        )

        backend = OpenAICompatibleBackend()
        for text in CORPUS.values():
            assert backend.count_tokens(text) == \
                estimate_tokens(text)

    def test_core_answers_through_the_generic_backend(self):
        server = FakeOpenAIServer(text=(
            "The River Sarn ferry crossing takes eleven "
            "minutes and the tide bell rings at dawn "
            "for the first boat."))
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
        )
        core._unified_backend = OpenAICompatibleBackend(
            model="fake-model")
        with patch(POST, side_effect=server.post):
            final = core.think(
                "What is the River Sarn ferry crossing?")
        assert final.mode != "failed"
        turn = core.last_turn_state
        assert turn is not None and turn.candidates
        assert server.calls and server.calls[0]["url"].endswith(
            "/v1/chat/completions")
