"""
TF-074 — The Aesir backend: Project Aesir bound as a
backend under TF-065's contract, its capability ledger
respected.

The binding lives in
``src/thoughtforge/inference/aesir_backend.py``
(``AesirBackend`` + ``AESIR_LEDGER`` / ``AESIR_RULES`` as
data, format ``thoughtforge-aesir-backend/1``), written
against Aesir's own versioned statements (API matrix v1,
CAPABILITY_LEDGER AES-SRV-006/008, NATIVE_SERVICE; Aesir
repo HEAD 1eed9706, the pin Rúnmál's P-009/F-02 used).
Write-up: ``docs/architecture/AESIR_BACKEND.md``.

Deliberate in-scope amendments: ``REQUESTS_CONSUMERS``
grew to five in ``backend_contract.py`` (this module is
an announced requests consumer) and the TF-065 test pin
for its length was updated to match. ``SHIPPED_BACKENDS``
is NOT amended — the factory set stays TF-065's four;
TF-153 owns admitting new backends there.

Pinned here:

  * the filed estate: ledger/rule ids and statuses (the
    not-served rows stay not-served), both checkers clean,
  * TF-065 conformance of the new backend judged by the
    contract's own checker (capabilities clean, native
    streaming override real, seed in source, count_tokens
    behaviour, health returns a bool without raising),
  * the wire: payload keys are exactly the served chat
    subset, seed forwarded only when carried, the bearer
    header in native mode, no rejected field ever sent,
  * stop: refused as error data before any transport call,
    in generate() and generate_stream() alike,
  * admission bounds: 129 messages / oversize field /
    oversize body / bad role / non-string content /
    assistant-final — each refused client-side,
  * responses: usage tokens when reported, the declared
    estimate otherwise; finish length passes; an
    unexpected finish reason is error data,
  * errors: the native error shape parsed with its status
    named; 422 named admission; 404 named non-resident and
    never retried; a raising transport becomes error data,
  * health: native readiness is exactly 200 + ready true;
    503 / ready-false / unreachable are False; ollama mode
    reads /v1/models; readiness_check surfaces the
    resident payload and states the ollama-mode gap,
  * catalog vs readiness: list_models returns ids while
    readiness on the same fake server says not ready,
  * streaming: SSE deltas join in order; EOF without a
    terminal raises; an SSE error event raises; an HTTP
    error before the stream raises,
  * token counting: the contract estimate, equal to the
    other backends on the TF-071 corpus (agreement holds),
  * the core leg: a core holding the Aesir backend over a
    fake transport answers the turn through it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.aesir_backend import (
    AESIR_BOUNDS,
    AESIR_CHAT_FIELDS,
    AESIR_FORMAT,
    AESIR_LEDGER,
    AESIR_RULES,
    AesirBackend,
    aesir_as_data,
    aesir_live_problems,
    aesir_problems,
)
from thoughtforge.inference.backend_contract import (
    SHIPPED_BACKENDS,
    backend_contract_problems,
    capabilities_problems,
    estimate_tokens,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    UnifiedBackend,
)

POST = "thoughtforge.inference.aesir_backend.requests.post"
GET = "thoughtforge.inference.aesir_backend.requests.get"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, lines=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._lines = lines or []

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def chat_payload(text="Four", finish="stop", usage=True):
    payload = {
        "id": "chatcmpl-aesir", "object": "chat.completion",
        "created": 1, "model": "gemma4-e2b:latest",
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": text},
                     "finish_reason": finish}],
    }
    if usage:
        payload["usage"] = {"prompt_tokens": 9,
                            "completion_tokens": 3,
                            "total_tokens": 12}
    return payload


def capture_post(response):
    calls = []

    def fake(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if isinstance(response, Exception):
            raise response
        return response

    return calls, fake


SSE_COMPLETE = [
    'data: {"choices":[{"delta":{"role":"assistant"},'
    '"finish_reason":null}]}',
    'data: {"choices":[{"delta":{"content":"Fo"},'
    '"finish_reason":null}]}',
    "",
    'data: {"choices":[{"delta":{"content":"ur"},'
    '"finish_reason":null}]}',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
]


# ── The filed estate ───────────────────────────────────────────


class TestFiledEstate:
    def test_format_and_data_shape(self):
        data = aesir_as_data()
        assert data["format"] == AESIR_FORMAT
        assert AESIR_FORMAT == "thoughtforge-aesir-backend/1"
        assert [e["id"] for e in data["ledger"]] == \
            [e["id"] for e in AESIR_LEDGER]
        assert data["bounds"]["max_messages"] == 128
        assert data["bounds"]["max_body_bytes"] == 131072

    def test_ledger_statuses(self):
        statuses = {e["id"]: e["status"] for e in AESIR_LEDGER}
        for not_served in ("embeddings_not_served",
                           "vision_rejected",
                           "general_tools_not_served"):
            assert statuses[not_served] == "not_served"
        assert statuses["native_generate_full_samplers"] == \
            "served_not_bound"
        assert all(e["source"] for e in AESIR_LEDGER)

    def test_rule_ids_exact(self):
        assert [r["id"] for r in AESIR_RULES] == [
            "bind_served_subset_only",
            "stop_refused_never_dropped",
            "health_is_resident_readiness",
            "catalog_is_not_readiness",
            "admission_bounds_checked_before_sending",
            "incomplete_stream_is_failure",
            "errors_are_data_with_status",
            "token_counting_stays_estimate",
            "one_resident_model",
            "live_proof_not_claimed",
        ]
        assert all(r["rule"] and r["measured_by"]
                   for r in AESIR_RULES)

    def test_chat_fields_table(self):
        fields = {f["field"]: f["served"]
                  for f in AESIR_CHAT_FIELDS}
        assert fields == {"temperature": True, "max_tokens": True,
                          "seed": True, "stop": False}

    def test_checkers_clean(self):
        assert aesir_problems() == []
        assert aesir_live_problems() == []

    def test_bounds_are_aesirs_own(self):
        assert AESIR_BOUNDS["max_field_bytes"] == 64 * 1024
        assert AESIR_BOUNDS["roles"] == \
            ("system", "user", "assistant")


# ── TF-065 conformance ─────────────────────────────────────────


class TestContractConformance:
    def test_contract_checker_clean_for_aesir(self):
        backend = AesirBackend(model="gemma4-e2b:latest")
        assert backend_contract_problems(backends=[backend]) == []

    def test_capability_record(self):
        caps = AesirBackend(model="m1").capabilities()
        assert capabilities_problems(caps.to_dict()) == []
        assert caps.name == "aesir:m1"
        assert caps.streaming == "native"
        assert caps.token_counting == "estimate"
        assert caps.seed == "honoured"
        assert caps.placement == "local"
        assert any("embeddings are not served" in note
                   for note in caps.notes)

    def test_not_in_shipped_factory_set(self):
        # TF-153 owns admitting new backends to the factory;
        # TF-065's shipped set stays the four it pinned.
        assert [r["id"] for r in SHIPPED_BACKENDS] == [
            "ollama", "lmstudio", "huggingface", "turboquant"]
        assert isinstance(AesirBackend(), UnifiedBackend)

    def test_base_url_normalisation(self):
        assert AesirBackend().base_url == "http://127.0.0.1:18434"
        assert AesirBackend(
            base_url="http://127.0.0.1:18434/v1").base_url == \
            "http://127.0.0.1:18434"
        assert AesirBackend().backend_name() == "aesir:resident"

    def test_bad_mode_refused(self):
        with pytest.raises(ValueError):
            AesirBackend(mode="cloud")


# ── The wire ───────────────────────────────────────────────────


class TestWireShape:
    def test_payload_is_exactly_the_served_subset(self):
        calls, fake = capture_post(FakeResponse(200, chat_payload()))
        backend = AesirBackend(model="gemma4-e2b:latest",
                               api_key="k3y")
        with patch(POST, side_effect=fake):
            resp = backend.generate(GenerationRequest(
                prompt="What is two plus two?", temperature=0.2,
                max_tokens=16, seed=42))
        assert resp.text == "Four"
        assert calls[0]["url"] == \
            "http://127.0.0.1:18434/v1/chat/completions"
        payload = calls[0]["json"]
        assert set(payload) == {"model", "messages", "stream",
                                "max_tokens", "temperature", "seed"}
        assert payload["stream"] is False
        assert payload["seed"] == 42
        assert payload["messages"] == [
            {"role": "user", "content": "What is two plus two?"}]
        assert calls[0]["headers"]["Authorization"] == "Bearer k3y"

    def test_no_seed_no_key_and_no_model(self):
        calls, fake = capture_post(FakeResponse(200, chat_payload()))
        with patch(POST, side_effect=fake):
            AesirBackend().generate(GenerationRequest(prompt="hi"))
        payload = calls[0]["json"]
        assert "seed" not in payload and "model" not in payload
        assert "Authorization" not in calls[0]["headers"]

    def test_system_prompt_and_carried_messages(self):
        calls, fake = capture_post(FakeResponse(200, chat_payload()))
        backend = AesirBackend(model="gemma4-e2b:latest")
        with patch(POST, side_effect=fake):
            backend.generate(GenerationRequest(
                prompt="q", system_prompt="Be terse."))
        assert calls[0]["json"]["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "q"}]

    def test_usage_tokens_and_finish(self):
        calls, fake = capture_post(FakeResponse(
            200, chat_payload(text="abc", finish="length")))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(
                GenerationRequest(prompt="hi"))
        assert resp.tokens_generated == 3  # Aesir's own usage
        assert resp.finish_reason == "length"
        assert resp.backend_used == "aesir:m"

    def test_no_usage_falls_back_to_declared_estimate(self):
        calls, fake = capture_post(FakeResponse(
            200, chat_payload(text="the ferry", usage=False)))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(
                GenerationRequest(prompt="hi"))
        assert resp.tokens_generated == estimate_tokens("the ferry")

    def test_unexpected_finish_reason_is_error_data(self):
        calls, fake = capture_post(FakeResponse(
            200, chat_payload(finish="tool_calls")))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(
                GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error"
        assert "tool_calls" in resp.error and resp.text == ""


# ── Stop refusal + admission bounds ────────────────────────────


class TestRefusals:
    def test_stop_refused_before_any_call(self):
        calls, fake = capture_post(FakeResponse(200, chat_payload()))
        backend = AesirBackend(model="m")
        with patch(POST, side_effect=fake):
            resp = backend.generate(GenerationRequest(
                prompt="hi", stop=["\n\n"]))
        assert resp.finish_reason == "error"
        assert "stop" in resp.error and "refused" in resp.error
        assert calls == []

    def test_stop_refused_in_stream_before_any_call(self):
        calls, fake = capture_post(FakeResponse(200, lines=[]))
        backend = AesirBackend(model="m")
        with patch(POST, side_effect=fake):
            with pytest.raises(RuntimeError, match="stop"):
                list(backend.generate_stream(GenerationRequest(
                    prompt="hi", stop=["x"])))
        assert calls == []

    def _refused(self, request):
        calls, fake = capture_post(FakeResponse(200, chat_payload()))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(request)
        assert resp.finish_reason == "error"
        assert resp.error.startswith("aesir admission:")
        assert calls == []
        return resp

    def test_too_many_messages_refused(self):
        msgs = [{"role": "user", "content": f"m{i}"}
                for i in range(129)]
        self._refused(GenerationRequest(prompt="", messages=msgs))

    def test_oversize_field_refused(self):
        self._refused(GenerationRequest(
            prompt="x" * (64 * 1024 + 1)))

    def test_bad_role_refused(self):
        self._refused(GenerationRequest(prompt="", messages=[
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "hi"}]))

    def test_non_string_content_refused(self):
        self._refused(GenerationRequest(prompt="", messages=[
            {"role": "user",
             "content": [{"type": "image", "data": "..."}]}]))

    def test_assistant_final_refused(self):
        self._refused(GenerationRequest(prompt="", messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]))


# ── Errors are data ────────────────────────────────────────────


class TestErrorMapping:
    @pytest.mark.parametrize("status,needle", [
        (400, "HTTP 400"), (401, "HTTP 401"),
        (404, "not resident"), (422, "admission"),
        (503, "HTTP 503"),
    ])
    def test_http_errors_named_with_server_message(self, status,
                                                    needle):
        body = {"error": {"code": status,
                          "message": f"server says {status}"}}
        calls, fake = capture_post(FakeResponse(status, body))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(
                GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error" and resp.text == ""
        assert needle in resp.error
        assert f"server says {status}" in resp.error
        assert resp.backend_used == "aesir:m"

    def test_404_never_retried_against_another_model(self):
        body = {"error": {"code": 404, "message": "not loaded"}}
        calls, fake = capture_post(FakeResponse(404, body))
        with patch(POST, side_effect=fake):
            AesirBackend(model="only:this").generate(
                GenerationRequest(prompt="hi"))
        assert len(calls) == 1
        assert calls[0]["json"]["model"] == "only:this"

    def test_ollama_mode_error_shape(self):
        calls, fake = capture_post(
            FakeResponse(500, {"error": "engine died"}))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m", mode="ollama").generate(
                GenerationRequest(prompt="hi"))
        assert "engine died" in resp.error

    def test_transport_exception_is_error_data(self):
        calls, fake = capture_post(ConnectionError("refused"))
        with patch(POST, side_effect=fake):
            resp = AesirBackend(model="m").generate(
                GenerationRequest(prompt="hi"))
        assert resp.finish_reason == "error"
        assert "refused" in resp.error


# ── Health, readiness, catalog ─────────────────────────────────


class TestHealthAndCatalog:
    def _get(self, response):
        calls = []

        def fake(url, **kwargs):
            calls.append(url)
            if isinstance(response, Exception):
                raise response
            return response

        return calls, fake

    def test_native_health_is_readiness_exactly(self):
        calls, fake = self._get(FakeResponse(
            200, {"ready": True, "status": "ready"}))
        with patch(GET, side_effect=fake):
            assert AesirBackend().health_check() is True
        assert calls == ["http://127.0.0.1:18434/health/ready"]

    @pytest.mark.parametrize("response", [
        FakeResponse(200, {"ready": False, "status": "not_ready"}),
        FakeResponse(503, {"ready": False, "status": "not_ready"}),
        FakeResponse(200, {"status": "ok"}),   # live != ready
        ConnectionError("down"),
    ])
    def test_native_health_false_shapes(self, response):
        calls, fake = self._get(response)
        with patch(GET, side_effect=fake):
            assert AesirBackend().health_check() is False

    def test_ollama_mode_health_is_models_reachability(self):
        calls, fake = self._get(FakeResponse(200, {"data": []}))
        backend = AesirBackend(base_url="http://127.0.0.1:11434",
                               mode="ollama")
        with patch(GET, side_effect=fake):
            assert backend.health_check() is True
        assert calls == ["http://127.0.0.1:11434/v1/models"]

    def test_readiness_check_surfaces_resident_payload(self):
        payload = {"ready": True, "status": "ready",
                   "model": "gemma4-e2b:latest", "context": 16384}
        calls, fake = self._get(FakeResponse(200, payload))
        with patch(GET, side_effect=fake):
            report = AesirBackend().readiness_check()
        assert report["ready"] is True
        assert report["payload"]["model"] == "gemma4-e2b:latest"
        assert report["http_status"] == 200

    def test_readiness_check_ollama_mode_states_the_gap(self):
        report = AesirBackend(mode="ollama").readiness_check()
        assert report["served"] is False
        assert report["ready"] is None

    def test_readiness_check_unreachable_is_unknown(self):
        calls, fake = self._get(ConnectionError("down"))
        with patch(GET, side_effect=fake):
            report = AesirBackend().readiness_check()
        assert report["ready"] is None and "error" in report

    def test_catalog_is_not_readiness(self):
        def fake(url, **kwargs):
            if url.endswith("/v1/models"):
                return FakeResponse(200, {"data": [
                    {"id": "gemma4-e2b:latest"},
                    {"id": "other:model"}]})
            return FakeResponse(
                503, {"ready": False, "status": "not_ready"})

        backend = AesirBackend(model="gemma4-e2b:latest")
        with patch(GET, side_effect=fake):
            assert backend.list_models() == [
                "gemma4-e2b:latest", "other:model"]
            assert backend.health_check() is False


# ── Streaming ──────────────────────────────────────────────────


class TestStreaming:
    def test_complete_stream_yields_in_order(self):
        calls, fake = capture_post(
            FakeResponse(200, lines=SSE_COMPLETE))
        backend = AesirBackend(model="m")
        with patch(POST, side_effect=fake):
            chunks = list(backend.generate_stream(
                GenerationRequest(prompt="hi", seed=3)))
        assert chunks == ["Fo", "ur"]
        assert calls[0]["json"]["stream"] is True
        assert calls[0]["json"]["seed"] == 3

    def test_done_alone_is_a_terminal(self):
        lines = ['data: {"choices":[{"delta":{"content":"x"},'
                 '"finish_reason":null}]}', "data: [DONE]"]
        calls, fake = capture_post(FakeResponse(200, lines=lines))
        with patch(POST, side_effect=fake):
            assert list(AesirBackend(model="m").generate_stream(
                GenerationRequest(prompt="hi"))) == ["x"]

    def test_eof_without_terminal_raises(self):
        calls, fake = capture_post(
            FakeResponse(200, lines=SSE_COMPLETE[:2]))
        gen = AesirBackend(model="m").generate_stream(
            GenerationRequest(prompt="hi"))
        with patch(POST, side_effect=fake):
            assert next(gen) == "Fo"
            with pytest.raises(RuntimeError, match="terminal"):
                next(gen)

    def test_sse_error_event_raises(self):
        lines = ['data: {"error":{"code":500,'
                 '"message":"execution failed"}}']
        calls, fake = capture_post(FakeResponse(200, lines=lines))
        with patch(POST, side_effect=fake):
            with pytest.raises(RuntimeError,
                               match="execution failed"):
                list(AesirBackend(model="m").generate_stream(
                    GenerationRequest(prompt="hi")))

    def test_http_error_before_stream_raises(self):
        body = {"error": {"code": 422, "message": "too long"}}
        calls, fake = capture_post(FakeResponse(422, body))
        with patch(POST, side_effect=fake):
            with pytest.raises(RuntimeError, match="admission"):
                list(AesirBackend(model="m").generate_stream(
                    GenerationRequest(prompt="hi")))


# ── Token counting agreement (TF-071) ──────────────────────────


class TestTokenCounting:
    def test_count_is_the_contract_estimate(self):
        from thoughtforge.inference.tokenizer_discipline import (
            CORPUS,
        )

        backend = AesirBackend(model="m")
        for text in CORPUS.values():
            assert backend.count_tokens(text) == \
                estimate_tokens(text)
        assert backend.count_tokens("") == 0


# ── The core leg ───────────────────────────────────────────────


class TestCoreLeg:
    def test_core_answers_through_aesir(self):
        calls, fake = capture_post(FakeResponse(200, chat_payload(
            text=("The River Sarn ferry crossing takes eleven "
                  "minutes and the tide bell rings at dawn "
                  "for the first boat."))))
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
        )
        core._unified_backend = AesirBackend(
            model="gemma4-e2b:latest")
        with patch(POST, side_effect=fake):
            final = core.think(
                "What is the River Sarn ferry crossing?")
        assert final.mode != "failed"
        turn = core.last_turn_state
        assert turn is not None and turn.candidates
        assert calls and calls[0]["url"].endswith(
            "/v1/chat/completions")
