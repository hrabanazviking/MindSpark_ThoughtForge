"""
TF-084 — Inference traces join cognition traces: one
exportable story per turn.

The estate lives in
``src/thoughtforge/inference/joined_trace.py``
(``TRACE_RULES`` as data, ``InferenceRecorder``,
``TracingBackend``, ``join_traces`` /
``joined_trace_problems``, formats
``thoughtforge-inference-trace/1`` +
``thoughtforge-joined-trace/1``); the core wiring is
``ThoughtForgeCore.last_turn_joined_trace()`` plus a
recording call at every leaf generation site in
``cognition/core.py``. The D-HON-1 fix lives in
``inference/unified_backend.py`` (the default stream's
``last_stream_response`` stash) and
``inference/turboquant_backend.py`` (the load-failure
stream stash), consumed by the core's streamed path.
Write-up: ``docs/architecture/JOINED_TRACES.md``.

Pinned here:

  * the filed estate: formats, sections, rule ids
    unique, the export's proof grades stated, and the
    post-freeze additions registry clean (the core
    method + module recorded, TF-034),
  * one story per turn, measured through real turns:
    knowledge-only (zero calls, source stated none),
    stub-backend plain + streamed, stub-engine plain /
    native-stream / generate-in-stream, a repair-pass
    failure, a model-load failure, an interrupted turn
    — the checker clean on every one, the two
    reconciliation identities holding on every one,
  * D-HON-1 FIXED: a streamed backend failure carries
    the backend's exact error string, identical to the
    plain path's record for the same backend; the
    default stream still yields nothing on an error;
    TurboQuantBackend's load-failure stream stashes
    its error the same way,
  * sizes, never texts: no inference call carries a
    prompt or response text field, and the response
    text appears nowhere in the inference section,
  * latency named, never conflated: the recorder's
    wall and the backend's reported latency ride as
    separate fields,
  * fallback attempts ride the call: a FallbackBackend
    chain's per-member outcomes are in the record,
  * TracingBackend standalone: generate + stream
    recorded, the wrapped response unmodified, a
    raising backend recorded then re-raised,
  * the checker catches, by name: wrong format, a
    missing section, a cognition problem propagated,
    a turn_id disagreement, a tampered
    reconciliation, a broken token identity, a broken
    failure identity, an error call with no detail,
    a served call carrying an error, a bad sequence,
  * the named gap is filed: D-TRACE-1 in the rules
    (salvage refine passes, TF-097's road).

Proof grade: FIXTURE generation sources on this VM
(stub backends / engines over real core turns). No
live model was joined; a live model's turn differs
only in who wrote the GenerationResults.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.core_docs import additions_problems
from thoughtforge.cognition.interruption import InterruptToken, TurnInterrupted
from thoughtforge.inference.backend_fallback import FallbackBackend
from thoughtforge.inference.joined_trace import (
    CALL_KINDS,
    CALL_OUTCOMES,
    INFERENCE_FORMAT,
    JOINED_FORMAT,
    JOINED_SECTIONS,
    TRACE_RULES,
    InferenceRecorder,
    TracingBackend,
    inference_problems,
    join_traces,
    joined_as_data,
    joined_trace_problems,
    record_engine_result,
    write_joined_trace,
)
from thoughtforge.inference.turboquant_backend import TurboQuantBackend
from thoughtforge.inference.unified_backend import (
    GenerationRequest,
    GenerationResponse,
    UnifiedBackend,
)

TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)
LOW_TEXT = "Bananas ripen slowly inside wooden crates during winter storage months."
QUERY = "What time does the ferry leave?"


# ── Fixture generation sources ────────────────────────────────────────────────


class StubBackend(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(
            text=TEXT, tokens_generated=len(TEXT.split()),
            backend_used="tf084-stub", latency_ms=3.5,
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf084-stub"


class BadBackend(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(
            text="", error="boom-engine-dead",
            finish_reason="error", backend_used="tf084-bad",
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf084-bad"


class EmptyBackend(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(text="", backend_used="tf084-empty")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf084-empty"


class RaisingBackend(UnifiedBackend):
    def generate(self, request):
        raise ConnectionError("backend unreachable")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf084-raising"


class StubEngine:
    """The real GenerationResult shape, both modes."""

    def generate(self, prompt, params=None):
        return SimpleNamespace(
            text=TEXT, tokens_generated=len(TEXT.split()),
            tokens_prompt=42, total_tokens=70, duration_ms=12,
            tokens_per_second=100.0, stopped_reason="eos",
            model_path="/models/test-q4.gguf",
            profile_id="desktop_cpu",
        )


class NativeStreamEngine:
    def generate_stream(self, prompt, params=None):
        yield TEXT[:50]
        yield TEXT[50:]

    def generate(self, prompt, params=None):  # pragma: no cover
        raise AssertionError("generate() must not be used")


class RepairFailingEngine:
    """Candidates answer (badly); the repair prompt raises."""

    def generate(self, prompt, params=None):
        if "Keep it specific and useful" in prompt:
            raise RuntimeError("repair died")
        return SimpleNamespace(
            text=LOW_TEXT, tokens_generated=len(LOW_TEXT.split())
        )


def _core(**kwargs) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
        **kwargs,
    )


def _joined(core: ThoughtForgeCore) -> dict:
    joined = core.last_turn_joined_trace()
    assert joined is not None
    assert joined_trace_problems(joined) == []
    return joined


# ── The filed estate ──────────────────────────────────────────────────────────


class TestEstate:
    def test_formats_sections_and_rules(self) -> None:
        assert JOINED_FORMAT == "thoughtforge-joined-trace/1"
        assert INFERENCE_FORMAT == "thoughtforge-inference-trace/1"
        assert JOINED_SECTIONS == (
            "cognition", "inference", "reconciliation")
        assert set(CALL_KINDS) == {
            "backend_generate", "backend_stream", "engine_generate",
            "engine_stream", "engine_repair", "model_load",
        }
        assert "interrupted" in CALL_OUTCOMES
        ids = [r["id"] for r in TRACE_RULES]
        assert len(ids) == len(set(ids)) == 11
        data = joined_as_data()
        assert data["format"] == JOINED_FORMAT
        assert data["proof"]["live_model_join"] == "not_proven"
        assert data["proof"]["d_hon_1_stream_error_detail"] == "fixed"

    def test_named_gap_is_filed(self) -> None:
        rules = {r["id"]: r["rule"] for r in TRACE_RULES}
        assert "D-TRACE-1" in rules["salvage_refine_passes_are_a_named_gap"]
        assert "TF-097" in rules["salvage_refine_passes_are_a_named_gap"]
        assert "NOT PROVEN" in rules["live_model_join_not_proven"]

    def test_post_freeze_additions_recorded(self) -> None:
        # TF-034: the core method + the module are in
        # core_docs' registry, and the registry is clean.
        assert additions_problems() == []
        assert callable(ThoughtForgeCore.last_turn_joined_trace)

    def test_none_before_the_first_turn(self) -> None:
        assert _core().last_turn_joined_trace() is None


# ── One story per turn, measured ──────────────────────────────────────────────


class TestKnowledgeOnlyJoin:
    def test_knowledge_only_turn_has_zero_calls_stated(self) -> None:
        core = _core()
        core.think(QUERY)
        joined = _joined(core)
        assert joined["inference"]["calls"] == []
        assert joined["inference"]["generation_source"] == "none"
        recon = joined["reconciliation"]
        assert recon["inference_call_count"] == 0
        assert recon["candidate_count"] == 0
        assert joined["turn_id"] == joined["cognition"]["turn_id"]


class TestBackendJoin:
    def test_plain_backend_turn(self) -> None:
        core = _core(backend=StubBackend())
        core.think(QUERY)
        joined = _joined(core)
        assert joined["inference"]["generation_source"] == "unified_backend"
        (call,) = joined["inference"]["calls"]
        assert call["kind"] == "backend_generate"
        assert call["outcome"] == "served"
        assert call["backend"] == "tf084-stub"
        assert call["response"]["tokens_generated"] == len(TEXT.split())
        assert call["response"]["tokens_basis"] == "backend_reported"
        # Latency named, never conflated: the backend's
        # own 3.5 rides separately from the recorder wall.
        assert call["response"]["reported_latency_ms"] == 3.5
        assert call["response"]["latency_ms"] >= 0
        recon = joined["reconciliation"]
        assert recon["candidate_tokens_recorded"] == \
            recon["candidate_tokens_in_cognition"] == len(TEXT.split())

    def test_no_texts_in_the_inference_section(self) -> None:
        core = _core(backend=StubBackend())
        core.think(QUERY)
        joined = _joined(core)
        inference_json = json.dumps(joined["inference"])
        assert TEXT not in inference_json
        for call in joined["inference"]["calls"]:
            assert "prompt" not in call["request"] or \
                isinstance(call["request"].get("prompt"), type(None))
            assert set(call["request"]) == {
                "prompt_chars", "prompt_tokens", "prompt_tokens_basis",
                "temperature", "max_tokens", "seed", "streamed", "native",
            }
            assert call["request"]["prompt_chars"] > len(QUERY)

    def test_streamed_backend_turn(self) -> None:
        core = _core(backend=StubBackend())
        list(core.think_stream(QUERY))
        joined = _joined(core)
        (call,) = joined["inference"]["calls"]
        assert call["kind"] == "backend_stream"
        assert call["outcome"] == "served"
        assert call["request"]["streamed"] is True
        assert call["request"]["native"] is False  # the default stream
        assert call["chunks"] == 1
        assert call["response"]["tokens_basis"] == "word_count_after_stream"
        assert call["response"]["tokens_generated"] == len(TEXT.split())

    def test_json_round_trip_and_file_write(self, tmp_path) -> None:
        core = _core(backend=StubBackend())
        core.think(QUERY)
        joined = _joined(core)
        assert json.loads(json.dumps(joined)) == joined
        target = write_joined_trace(joined, tmp_path / "deep" / "j.json")
        assert json.loads(target.read_text(encoding="utf-8")) == joined


class TestEngineJoin:
    def test_plain_engine_turn_two_modes(self) -> None:
        core = _core()
        core._engine = StubEngine()
        core.think(QUERY)
        joined = _joined(core)
        assert joined["inference"]["generation_source"] == "turboquant_engine"
        calls = joined["inference"]["calls"]
        assert [c["kind"] for c in calls] == [
            "engine_generate", "engine_generate"]
        assert {c["mode"] for c in calls} == {
            m for m in joined["cognition"]["scaffold"]["candidate_modes"][:2]}
        for call in calls:
            assert call["response"]["tokens_basis"] == "engine_reported"
            assert call["response"]["reported_latency_ms"] == 12
            assert call["request"]["prompt_tokens"] == 42
            assert call["request"]["prompt_tokens_basis"] == "engine_usage"
            assert call["model"] == "test-q4.gguf"
            assert call["profile_id"] == "desktop_cpu"

    def test_native_stream_engine_turn(self) -> None:
        core = _core()
        core._engine = NativeStreamEngine()
        list(core.think_stream(QUERY))
        joined = _joined(core)
        calls = joined["inference"]["calls"]
        assert calls and all(c["kind"] == "engine_stream" for c in calls)
        for call in calls:
            assert call["request"]["native"] is True
            assert call["chunks"] == 2
            assert call["response"]["tokens_basis"] == \
                "word_count_after_stream"

    def test_generate_inside_stream_engine_turn(self) -> None:
        core = _core()
        core._engine = StubEngine()  # no generate_stream: generate() serves
        list(core.think_stream(QUERY))
        joined = _joined(core)
        calls = joined["inference"]["calls"]
        assert calls and all(c["kind"] == "engine_stream" for c in calls)
        for call in calls:
            assert call["request"]["native"] is False
            assert call["response"]["tokens_basis"] == "engine_reported"

    def test_repair_failure_rides_the_join(self) -> None:
        core = _core()
        core._engine = RepairFailingEngine()
        core.think(QUERY)
        joined = _joined(core)
        kinds = [c["kind"] for c in joined["inference"]["calls"]]
        assert "engine_repair" in kinds
        repair = [c for c in joined["inference"]["calls"]
                  if c["kind"] == "engine_repair"][0]
        assert repair["outcome"] == "exception"
        assert "repair died" in repair["error"]
        recon = joined["reconciliation"]
        assert recon["failed_call_count"] == \
            recon["generation_failure_count"] == 1
        assert recon["repair_tokens_recorded"] == 0

    def test_model_load_failure_rides_the_join(self) -> None:
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=Path(tmp) / "no-such-model.gguf",
        )
        core.think(QUERY)
        joined = _joined(core)
        (call,) = joined["inference"]["calls"]
        assert call["kind"] == "model_load"
        assert call["outcome"] == "exception"
        assert call["error"]
        assert joined["reconciliation"]["failed_call_count"] == 1

    def test_interrupted_turn_join(self) -> None:
        token = InterruptToken()
        token.interrupt("caller stopped it")
        core = _core(backend=StubBackend())
        with pytest.raises(TurnInterrupted):
            core.think(QUERY, interrupt=token)
        # Cut before generation: no call was ever made.
        joined = _joined(core)
        assert joined["inference"]["calls"] == []
        assert joined["cognition"]["turn"]["is_interrupted"] is True


# ── D-HON-1: the streamed error is the plain error ────────────────────────────


class TestHon1Fixed:
    def test_streamed_failure_matches_plain_failure_exactly(self) -> None:
        plain = _core(backend=BadBackend())
        plain.think(QUERY)
        plain_joined = _joined(plain)
        streamed = _core(backend=BadBackend())
        list(streamed.think_stream(QUERY))
        streamed_joined = _joined(streamed)
        plain_call = plain_joined["inference"]["calls"][0]
        stream_call = streamed_joined["inference"]["calls"][0]
        assert plain_call["outcome"] == stream_call["outcome"] == "error"
        assert plain_call["error"] == stream_call["error"] == \
            "boom-engine-dead"
        # And the cognition failure records agree, detail for detail.
        assert plain_joined["cognition"]["generation_failures"] == \
            streamed_joined["cognition"]["generation_failures"] == [
                {"stage": "backend", "mode": "practical",
                 "detail": "boom-engine-dead"}
            ]

    def test_default_stream_still_yields_nothing_but_stashes(self) -> None:
        backend = BadBackend()
        assert list(backend.generate_stream(
            GenerationRequest(prompt="hi"))) == []
        stash = backend.last_stream_response
        assert stash.error == "boom-engine-dead"
        assert stash.finish_reason == "error"

    def test_turboquant_load_failure_stream_stashes(self) -> None:
        backend = TurboQuantBackend(
            model_path=Path("/nonexistent/x.gguf"))
        assert list(backend.generate_stream(
            GenerationRequest(prompt="hi"))) == []
        assert backend.last_stream_response.error != ""
        assert backend.last_stream_response.finish_reason == "error"

    def test_detail_less_empty_keeps_the_generic_line(self) -> None:
        core = _core(backend=EmptyBackend())
        list(core.think_stream(QUERY))
        joined = _joined(core)
        (call,) = joined["inference"]["calls"]
        assert call["outcome"] == "empty"
        assert call["error"] == "backend returned no text (error or empty)"
        assert joined["cognition"]["generation_failures"][0]["detail"] == \
            "backend returned no text (error or empty)"


# ── Fallback attempts + the tracing wrapper ───────────────────────────────────


class TestFallbackAndWrapper:
    def test_fallback_attempts_ride_the_call(self) -> None:
        chain = FallbackBackend([BadBackend(), StubBackend()])
        core = _core(backend=chain)
        core.think(QUERY)
        joined = _joined(core)
        (call,) = joined["inference"]["calls"]
        assert call["outcome"] == "served"
        assert call["backend"] == "tf084-stub"
        attempts = call["fallback_attempts"]
        assert [(a["backend"], a["outcome"]) for a in attempts] == [
            ("tf084-bad", "error"), ("tf084-stub", "served")]
        assert attempts[0]["detail"] == "boom-engine-dead"

    def test_tracing_backend_records_and_delegates(self) -> None:
        recorder = InferenceRecorder(turn_id="turn_wrap")
        wrapped = TracingBackend(StubBackend(), recorder)
        resp = wrapped.generate(GenerationRequest(prompt="hello world"))
        assert resp.text == TEXT  # the wrapped response, unmodified
        chunks = list(wrapped.generate_stream(
            GenerationRequest(prompt="hello")))
        assert chunks == [TEXT]
        data = recorder.as_data()
        assert inference_problems(data) == []
        assert [c["outcome"] for c in data["calls"]] == ["served", "served"]
        assert data["turn_id"] == "turn_wrap"

    def test_tracing_backend_records_a_raise_then_reraises(self) -> None:
        recorder = InferenceRecorder()
        wrapped = TracingBackend(RaisingBackend(), recorder)
        with pytest.raises(ConnectionError):
            wrapped.generate(GenerationRequest(prompt="hi"))
        (call,) = recorder.calls
        assert call["outcome"] == "exception"
        assert "backend unreachable" in call["error"]

    def test_tracing_backend_stream_error_detail(self) -> None:
        recorder = InferenceRecorder()
        wrapped = TracingBackend(BadBackend(), recorder)
        assert list(wrapped.generate_stream(
            GenerationRequest(prompt="hi"))) == []
        (call,) = recorder.calls
        assert call["outcome"] == "error"
        assert call["error"] == "boom-engine-dead"

    def test_record_engine_result_helper(self) -> None:
        recorder = InferenceRecorder(turn_id="turn_helper")
        result = SimpleNamespace(
            text=TEXT, tokens_generated=9, tokens_prompt=4,
            duration_ms=7, stopped_reason="eos",
            model_path="/m/x.gguf", profile_id="pi_5",
        )
        call = record_engine_result(
            recorder, result=result, prompt="a prompt", mode="practical",
            backend="cpu", wall_ms=9.0)
        assert call["sequence"] == 1
        assert call["response"]["tokens_generated"] == 9
        assert call["model"] == "x.gguf"
        assert inference_problems(recorder.as_data()) == []


# ── The checker catches tampering ─────────────────────────────────────────────


def _backend_joined() -> dict:
    core = _core(backend=StubBackend())
    core.think(QUERY)
    joined = core.last_turn_joined_trace()
    assert joined is not None
    return joined


class TestJoinedProblems:
    def test_not_a_mapping(self) -> None:
        assert joined_trace_problems([1]) == ["joined trace is not a mapping"]

    def test_wrong_format_and_missing_section(self) -> None:
        joined = _backend_joined()
        joined["format"] = "something-else/9"
        del joined["inference"]
        problems = joined_trace_problems(joined)
        assert any("format" in p for p in problems)
        assert any("missing section 'inference'" in p for p in problems)

    def test_turn_id_disagreement(self) -> None:
        joined = _backend_joined()
        joined["inference"]["turn_id"] = "turn_someone_else"
        assert any("disagrees" in p for p in joined_trace_problems(joined))

    def test_tampered_reconciliation(self) -> None:
        joined = _backend_joined()
        joined["reconciliation"]["inference_call_count"] = 99
        assert any("reconciliation disagrees" in p
                   for p in joined_trace_problems(joined))

    def test_broken_token_identity(self) -> None:
        joined = _backend_joined()
        joined["cognition"]["candidates"][0]["token_estimate"] += 5
        # Recompute the filed reconciliation to match the
        # tampered sections — the identity itself must fail.
        from thoughtforge.inference.joined_trace import _reconciliation
        joined["reconciliation"] = _reconciliation(
            joined["cognition"], joined["inference"])
        assert any("token" in p for p in joined_trace_problems(joined))

    def test_broken_failure_identity(self) -> None:
        joined = _backend_joined()
        joined["cognition"]["generation_failures"] = [
            {"stage": "backend", "mode": "practical", "detail": "ghost"}
        ]
        from thoughtforge.inference.joined_trace import _reconciliation
        joined["reconciliation"] = _reconciliation(
            joined["cognition"], joined["inference"])
        assert any("generation" in p and "failures" in p
                   for p in joined_trace_problems(joined))

    def test_error_call_without_detail_and_served_with_error(self) -> None:
        joined = _backend_joined()
        call = joined["inference"]["calls"][0]
        call["outcome"] = "error"
        call["error"] = ""
        problems = joined_trace_problems(joined)
        assert any("no error detail (D-HON-1)" in p for p in problems)
        joined = _backend_joined()
        joined["inference"]["calls"][0]["error"] = "surprise"
        problems = joined_trace_problems(joined)
        assert any("served call carries an error" in p for p in problems)

    def test_bad_sequence(self) -> None:
        joined = _backend_joined()
        joined["inference"]["calls"][0]["sequence"] = 7
        assert any("sequence" in p for p in joined_trace_problems(joined))

    def test_join_of_recorder_without_core(self) -> None:
        recorder = InferenceRecorder(turn_id="turn_solo")
        TracingBackend(StubBackend(), recorder).generate(
            GenerationRequest(prompt="hi"))
        joined = join_traces(None, recorder)
        assert joined["cognition"] is None
        assert joined["reconciliation"]["candidate_count"] is None
        assert joined_trace_problems(joined) == []
        detached = copy.deepcopy(joined)
        detached["inference"]["calls"][0]["outcome"] = "empty"
        assert joined_trace_problems(detached) != []
