"""
TF-073 — Backend health & fallback: a dead backend
degrades to the next per the stated policy, announced
honestly.

The policy lives in
``src/thoughtforge/inference/backend_fallback.py``
(FALLBACK_RULES + HEALTH_SEMANTICS as data, format
``thoughtforge-backend-fallback/1``), implemented by
``FallbackBackend`` — a UnifiedBackend wrapping an ordered
chain, one ``utils/retry.CircuitBreaker`` per member
(D-ORPHAN-2's production wiring). Fixes in scope:
D-HEALTH-1 in ``hf_backend.py`` (health was any status
< 500; now exactly 200) and D-ORPHAN-2 itself.
Write-up: ``docs/architecture/BACKEND_HEALTH_AND_FALLBACK.md``.

Pinned here:

  * the filed estate: format, rule ids, attempt outcomes,
    the health-semantics table; both checkers clean,
  * order + health gate: a healthy primary's fallbacks are
    never called; a health-False member is skipped with no
    generate call,
  * failure shapes: error response (its decoy text never
    served), empty response, and a raising generate() are
    all passed over and recorded,
  * announcement: last_report names every attempt, a
    fallback serve is marked degraded; an all-failed chain
    returns one error naming every member and its reason,
  * the breaker: threshold failures open it, the member is
    then skipped with no health/generate call while the
    chain keeps serving; a success resets the count,
  * chain health: any-member, short-circuiting, a raising
    member counts False, never raises,
  * streaming: fallback only before the first chunk — an
    empty or pre-chunk-failing stream degrades, a
    mid-stream failure after a chunk propagates with the
    fallback untouched,
  * D-HEALTH-1: HF health is True exactly on a mocked 200,
  * the core leg: a core holding the chain answers through
    the fallback instead of failing the turn.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.inference.backend_fallback import (
    ATTEMPT_OUTCOMES,
    FALLBACK_FORMAT,
    FALLBACK_RULES,
    HEALTH_SEMANTICS,
    FallbackBackend,
    fallback_as_data,
    fallback_live_problems,
    fallback_problems,
)
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)

REQ = GenerationRequest(prompt="prompt")


# ── Stubs ──────────────────────────────────────────────────────


class StubBackend(UnifiedBackend):
    def __init__(self, name, *, health=True, mode="serve", text=None,
                 chunks=None, health_raises=False):
        self._name = name
        self._health = health
        self._mode = mode
        self._text = text if text is not None else f"text from {name}"
        self._chunks = chunks
        self._health_raises = health_raises
        self.generate_calls = 0
        self.health_calls = 0
        self.stream_calls = 0

    def generate(self, request):
        self.generate_calls += 1
        if self._mode == "raise":
            raise RuntimeError(f"{self._name} exploded")
        if self._mode == "error":
            return GenerationResponse(
                text="decoy text", error=f"{self._name} is dead",
                finish_reason="error", backend_used=self._name)
        if self._mode == "empty":
            return GenerationResponse(text="  ", backend_used=self._name)
        return GenerationResponse(text=self._text,
                                  backend_used=self._name)

    def generate_stream(self, request):
        self.stream_calls += 1
        if self._mode == "stream_raise":
            raise RuntimeError("stream died before first chunk")
            yield  # pragma: no cover — makes this a generator
        if self._mode == "stream_mid_fail":
            yield "first "
            raise RuntimeError("stream died mid-way")
        yield from (self._chunks or [])

    def health_check(self):
        self.health_calls += 1
        if self._health_raises:
            raise ConnectionError("health exploded")
        return self._health

    def backend_name(self):
        return self._name


# ── The filed estate ───────────────────────────────────────────


class TestFiledEstate:
    def test_format_and_data_shape(self):
        data = fallback_as_data()
        assert data["format"] == FALLBACK_FORMAT
        assert FALLBACK_FORMAT == "thoughtforge-backend-fallback/1"
        assert [r["id"] for r in data["rules"]] == [
            r["id"] for r in FALLBACK_RULES]
        assert [s["backend"] for s in data["health_semantics"]] == [
            "huggingface", "ollama", "lmstudio", "turboquant"]

    def test_rule_ids_exact(self):
        assert [r["id"] for r in FALLBACK_RULES] == [
            "order_is_the_policy",
            "health_gate_before_generate",
            "error_response_is_failure",
            "empty_response_is_failure",
            "exception_is_failure_data",
            "circuit_breaker_skips_known_dead",
            "announced_never_silent",
            "all_failed_is_stated_whole",
            "stream_fallback_before_first_chunk_only",
            "chain_health_is_any_member",
        ]
        assert all(r["rule"] and r["measured_by"] for r in FALLBACK_RULES)

    def test_attempt_outcomes_cover_report(self):
        for outcome in ("served", "error", "empty", "exception",
                        "health_false", "circuit_open"):
            assert outcome in ATTEMPT_OUTCOMES
        assert all(s["proves"] and s["does_not_prove"]
                   for s in HEALTH_SEMANTICS)

    def test_checkers_clean(self):
        assert fallback_problems() == []
        assert fallback_live_problems() == []

    def test_empty_chain_refused(self):
        with pytest.raises(ValueError):
            FallbackBackend([])

    def test_wrapper_is_a_unified_backend(self):
        chain = FallbackBackend([StubBackend("a:1")])
        assert isinstance(chain, UnifiedBackend)
        assert chain.backend_name() == "fallback:a:1"
        assert chain.chain_names() == ["a:1"]


# ── Order, health gate, failure shapes ─────────────────────────


class TestGeneratePolicy:
    def test_primary_serves_fallbacks_untouched(self):
        primary = StubBackend("good:1")
        spare = StubBackend("spare:2")
        chain = FallbackBackend([primary, spare])
        resp = chain.generate(REQ)
        assert resp.text == "text from good:1"
        assert spare.generate_calls == 0 and spare.health_calls == 0
        report = chain.last_report
        assert report.served_by == "good:1"
        assert report.degraded is False and report.all_failed is False
        assert [a.outcome for a in report.attempts] == ["served"]

    def test_health_gate_skips_without_generate(self):
        dead = StubBackend("dead:1", health=False)
        good = StubBackend("good:2")
        chain = FallbackBackend([dead, good])
        resp = chain.generate(REQ)
        assert resp.text == "text from good:2"
        assert dead.generate_calls == 0
        assert [a.outcome for a in chain.last_report.attempts] == [
            "health_false", "served"]
        assert chain.last_report.degraded is True

    def test_health_raising_member_is_skipped(self):
        sick = StubBackend("sick:1", health_raises=True)
        good = StubBackend("good:2")
        chain = FallbackBackend([sick, good])
        assert chain.generate(REQ).text == "text from good:2"
        assert chain.last_report.attempts[0].outcome == "health_raised"

    def test_error_response_decoy_text_never_served(self):
        err = StubBackend("err:1", mode="error")
        good = StubBackend("good:2")
        chain = FallbackBackend([err, good])
        resp = chain.generate(REQ)
        assert resp.text == "text from good:2"
        assert chain.last_report.attempts[0].outcome == "error"
        assert "err:1 is dead" in chain.last_report.attempts[0].detail

    def test_empty_response_is_failure(self):
        empty = StubBackend("empty:1", mode="empty")
        good = StubBackend("good:2")
        chain = FallbackBackend([empty, good])
        assert chain.generate(REQ).text == "text from good:2"
        assert chain.last_report.attempts[0].outcome == "empty"

    def test_raising_generate_is_failure_data(self):
        boom = StubBackend("boom:1", mode="raise")
        good = StubBackend("good:2")
        chain = FallbackBackend([boom, good])
        resp = chain.generate(REQ)  # the wrapper itself never raises
        assert resp.text == "text from good:2"
        assert chain.last_report.attempts[0].outcome == "exception"

    def test_all_failed_states_the_whole_chain(self):
        chain = FallbackBackend([
            StubBackend("dead:1", health=False),
            StubBackend("err:2", mode="error"),
        ])
        resp = chain.generate(REQ)
        assert resp.text == "" and resp.finish_reason == "error"
        assert resp.backend_used == chain.backend_name()
        assert resp.error.startswith("all backends failed:")
        assert "dead:1" in resp.error and "health_false" in resp.error
        assert "err:2" in resp.error and "err:2 is dead" in resp.error
        assert chain.last_report.all_failed is True


# ── The circuit breaker (D-ORPHAN-2) ───────────────────────────


class TestCircuitBreaker:
    def test_open_breaker_skips_without_calls(self):
        flaky = StubBackend("flaky:1", mode="error")
        good = StubBackend("good:2")
        chain = FallbackBackend([flaky, good], failure_threshold=2,
                                recovery_timeout=60.0)
        chain.generate(REQ)
        chain.generate(REQ)
        assert chain.breaker_states()["flaky:1"] == "open"
        calls, health_calls = flaky.generate_calls, flaky.health_calls
        resp = chain.generate(REQ)
        assert resp.text == "text from good:2"
        assert flaky.generate_calls == calls
        assert flaky.health_calls == health_calls
        assert chain.last_report.attempts[0].outcome == "circuit_open"

    def test_success_resets_failure_count(self):
        switch = StubBackend("switch:1", mode="error")
        good = StubBackend("good:2")
        chain = FallbackBackend([switch, good], failure_threshold=2,
                                recovery_timeout=60.0)
        chain.generate(REQ)              # one failure recorded
        switch._mode = "serve"
        assert chain.generate(REQ).text == "text from switch:1"
        switch._mode = "error"
        chain.generate(REQ)              # one failure since success
        assert chain.breaker_states()["switch:1"] == "closed"


# ── Chain health ───────────────────────────────────────────────


class TestChainHealth:
    def test_any_member_short_circuits(self):
        first = StubBackend("a:1", health=True)
        second = StubBackend("b:2", health=True)
        chain = FallbackBackend([first, second])
        assert chain.health_check() is True
        assert second.health_calls == 0

    def test_all_unhealthy_is_false(self):
        chain = FallbackBackend([
            StubBackend("a:1", health=False),
            StubBackend("b:2", health_raises=True),
        ])
        assert chain.health_check() is False

    def test_second_member_can_carry_health(self):
        chain = FallbackBackend([
            StubBackend("a:1", health=False),
            StubBackend("b:2", health=True),
        ])
        assert chain.health_check() is True


# ── Streaming: takeover only before the first chunk ────────────


class TestStreamingPolicy:
    def test_primary_stream_fallback_untouched(self):
        primary = StubBackend("p:1", chunks=["one ", "two"])
        spare = StubBackend("s:2", chunks=["spare"])
        chain = FallbackBackend([primary, spare])
        assert "".join(chain.generate_stream(REQ)) == "one two"
        assert spare.stream_calls == 0
        assert chain.last_report.served_by == "p:1"

    def test_empty_stream_degrades(self):
        empty = StubBackend("e:1", chunks=[])
        good = StubBackend("g:2", chunks=["real ", "text"])
        chain = FallbackBackend([empty, good])
        assert "".join(chain.generate_stream(REQ)) == "real text"
        assert [a.outcome for a in chain.last_report.attempts] == [
            "stream_empty", "served"]
        assert chain.last_report.degraded is True

    def test_pre_chunk_stream_error_degrades(self):
        dying = StubBackend("d:1", mode="stream_raise")
        good = StubBackend("g:2", chunks=["whole"])
        chain = FallbackBackend([dying, good])
        assert "".join(chain.generate_stream(REQ)) == "whole"
        assert chain.last_report.attempts[0].outcome == "stream_error"

    def test_mid_stream_failure_propagates_fallback_untouched(self):
        dying = StubBackend("d:1", mode="stream_mid_fail")
        spare = StubBackend("s:2", chunks=["spare"])
        chain = FallbackBackend([dying, spare])
        gen = chain.generate_stream(REQ)
        assert next(gen) == "first "
        with pytest.raises(RuntimeError):
            next(gen)
        assert spare.stream_calls == 0

    def test_stream_health_gate(self):
        dead = StubBackend("dead:1", health=False, chunks=["x"])
        good = StubBackend("g:2", chunks=["y"])
        chain = FallbackBackend([dead, good])
        assert "".join(chain.generate_stream(REQ)) == "y"
        assert dead.stream_calls == 0


# ── Capabilities / token counting ──────────────────────────────


class TestWrapperSurface:
    def test_capabilities_announce_the_chain(self):
        chain = FallbackBackend([StubBackend("a:1"), StubBackend("b:2")])
        caps = chain.capabilities()
        assert caps.name == "fallback:a:1+b:2"
        assert any("a:1 -> b:2" in note for note in caps.notes)

    def test_count_tokens_delegates_to_primary(self):
        chain = FallbackBackend([StubBackend("a:1"), StubBackend("b:2")])
        assert chain.count_tokens("") == 0
        assert chain.count_tokens("the ferry leaves at dawn") > 0


# ── D-HEALTH-1: HF health is exactly 200 ───────────────────────


class TestHFHealthSemantics:
    @pytest.mark.parametrize("status,expected", [
        (200, True), (401, False), (403, False), (404, False),
        (500, False), (503, False),
    ])
    def test_health_is_exactly_200(self, status, expected):
        from thoughtforge.inference.hf_backend import HuggingFaceBackend

        backend = HuggingFaceBackend(model="no/such-model")
        resp = MagicMock()
        resp.status_code = status
        with patch("thoughtforge.inference.hf_backend.requests.get",
                   return_value=resp):
            assert backend.health_check() is expected


# ── The core leg ───────────────────────────────────────────────


class TestCoreLeg:
    def test_core_answers_through_the_fallback(self):
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
        )
        core._unified_backend = FallbackBackend([
            StubBackend("dead:1", mode="error"),
            StubBackend("good:2", text=(
                "The River Sarn ferry crossing takes eleven minutes "
                "and the tide bell rings at dawn for the first boat.")),
        ])
        final = core.think("What is the River Sarn ferry crossing?")
        assert final.mode != "failed"
        turn = core.last_turn_state
        assert turn is not None and turn.candidates
        report = core._unified_backend.last_report
        assert report.degraded is True
        assert report.served_by == "good:2"
