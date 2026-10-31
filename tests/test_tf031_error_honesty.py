"""
TF-031 — Error honesty: a failed turn states its failure.

The discipline lives in ``src/thoughtforge/cognition/error_honesty.py``
(failures as data, the stated-failure / degraded-knowledge /
partial-disclosure texts); the write-up is
``docs/architecture/ERROR_HONESTY.md``.

Pinned here:

  * the failure record: stage validated against FAILURE_STAGES,
    detail is "ClassName: message" capped at MAX_DETAIL_CHARS,
    dict round-trip exact,
  * the texts: a stated failure names stage + detail and says no
    model answer was generated; the degraded-knowledge text labels
    its body as retrieved knowledge only; the partial disclosure
    names the failed mode(s); empty / repair-only failure lists
    raise ValueError rather than fabricate a disclosure,
  * apply_partial_disclosure appends exactly once, recomputes the
    token count, and names the failure in the record's notes,
  * the core wiring end-to-end, through real think()/think_stream()
    turns on stub engines/backends:
      - every mode failing + empty store → mode "failed", a stated
        failure, and the empty-store text (which blames the
        knowledge) nowhere in it; both failures on the turn record,
      - model load failure → one model_load failure, stated,
      - an engine returning empty text → a failure, not a candidate,
      - one mode failing, one surviving → the real composed answer
        PLUS the disclosure; exactly one failure recorded,
      - a unified backend error (error channel or exception) and
        an empty backend text → backend failures, stated,
      - streamed: a dying stream yields candidate_error events and
        the terminal final is the stated failure, not a fabricated
        completion; a half-dying stream's final carries the
        disclosure,
      - total failure WITH retrieved knowledge → the knowledge is
        shown labelled as retrieved, under the stated failure
        (the bundle is injected at the retrieval seam: the
        existence gate checks the default DB path, not the
        configured one — CORE_LOOP F2, owned by TF-041),
      - a repair-pass failure is recorded on the turn but NOT
        announced on the answer (the composed answer is real),
      - a deliberate no-model core is untouched: no failures, the
        pre-TF-031 knowledge-only answer byte-identical,
      - an interrupted turn (TF-027) records no failure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.error_honesty import (
    DISCLOSURE_MARKER,
    FAILURE_STAGES,
    MAX_DETAIL_CHARS,
    GenerationFailure,
    apply_partial_disclosure,
    degraded_knowledge_text,
    failure_detail,
    failure_text,
    generation_failures,
    partial_disclosure,
    turn_failures,
)
from thoughtforge.cognition.interruption import InterruptToken, TurnInterrupted
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.models import (
    ActivatedRecord,
    FinalResponseRecord,
    MemoryActivationBundle,
)

QUERY = "How long does the River Sarn ferry crossing take?"
GOOD_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water. Wrenna the lantern-keeper checks the river level each "
    "morning and chalk-marks it on the mill door."
)
LOW_TEXT = "Bananas ripen slowly inside wooden crates during winter storage months."


# ── Stub engines / backends ────────────────────────────────────────────────────


class _FailingEngine:
    """Every generate() raises."""

    def generate(self, prompt, params=None):
        raise RuntimeError("engine exploded")


class _EmptyEngine:
    """Every generate() returns whitespace — a non-answer."""

    def generate(self, prompt, params=None):
        return SimpleNamespace(text="   ", tokens_generated=0)


class _HalfFailingEngine:
    """First generate() raises; later calls answer."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, params=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first mode died")
        return SimpleNamespace(
            text=GOOD_TEXT, tokens_generated=len(GOOD_TEXT.split())
        )


class _RepairFailingEngine:
    """Candidates answer (badly); the repair prompt raises."""

    def generate(self, prompt, params=None):
        if "Keep it specific and useful" in prompt:
            raise RuntimeError("repair died")
        return SimpleNamespace(
            text=LOW_TEXT, tokens_generated=len(LOW_TEXT.split())
        )


class _StreamFailEngine:
    """Every stream yields one delta, then dies mid-way."""

    def generate_stream(self, prompt, params=None):
        yield "partial words "
        raise RuntimeError("stream broke mid-way")

    def generate(self, prompt, params=None):  # pragma: no cover
        raise AssertionError("generate() must not be used on the stream path")


class _HalfStreamEngine:
    """First stream dies mid-way; the second completes."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_stream(self, prompt, params=None):
        self.calls += 1
        if self.calls == 1:
            yield GOOD_TEXT[:20]
            raise RuntimeError("stream broke mid-way")
        yield GOOD_TEXT

    def generate(self, prompt, params=None):
        return SimpleNamespace(
            text=GOOD_TEXT, tokens_generated=len(GOOD_TEXT.split())
        )


class _ErrorBackend(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(text="", error="backend exploded")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "stub-error"


class _RaisingBackend(UnifiedBackend):
    def generate(self, request):
        raise ConnectionError("backend unreachable")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "stub-raising"


class _EmptyBackend(UnifiedBackend):
    def generate(self, request):
        return GenerationResponse(text="", tokens_generated=0)

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "stub-empty"


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )


def _core_with_engine(engine) -> ThoughtForgeCore:
    core = _core()
    core._engine = engine
    return core


# ── The failure record ─────────────────────────────────────────────────────────


class TestFailureRecord:
    def test_stages_are_the_four(self) -> None:
        assert FAILURE_STAGES == ("model_load", "candidate", "backend", "repair")

    def test_from_exception_detail_is_class_and_message(self) -> None:
        failure = GenerationFailure.from_exception(
            "candidate", RuntimeError("engine exploded"), mode="practical"
        )
        assert failure.detail == "RuntimeError: engine exploded"
        assert failure.describe() == (
            "candidate (mode practical): RuntimeError: engine exploded"
        )

    def test_describe_without_mode(self) -> None:
        failure = GenerationFailure(stage="model_load", detail="FileNotFoundError: x.gguf")
        assert failure.describe() == "model_load: FileNotFoundError: x.gguf"

    def test_unknown_stage_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationFailure(stage="elsewhere", detail="boom")

    def test_detail_capped_at_max(self) -> None:
        failure = GenerationFailure(stage="backend", detail="x" * 500)
        assert len(failure.detail) == MAX_DETAIL_CHARS
        assert failure.detail.endswith("…")
        assert len(failure_detail(RuntimeError("y" * 500))) == MAX_DETAIL_CHARS

    def test_bare_exception_has_no_dangling_colon(self) -> None:
        assert failure_detail(RuntimeError()) == "RuntimeError"

    def test_dict_round_trip(self) -> None:
        failure = GenerationFailure.from_exception(
            "backend", ConnectionError("down"), mode="direct"
        )
        assert GenerationFailure.from_dict(failure.to_dict()).to_dict() == failure.to_dict()


# ── The texts ──────────────────────────────────────────────────────────────────


def _failures() -> list[GenerationFailure]:
    return [
        GenerationFailure.from_exception("candidate", RuntimeError("engine exploded"), mode="practical"),
        GenerationFailure.from_exception("candidate", RuntimeError("engine exploded"), mode="reflective"),
    ]


class TestTexts:
    def test_failure_text_states_no_answer_was_generated(self) -> None:
        text = failure_text(_failures())
        assert text.startswith(DISCLOSURE_MARKER)
        assert "RuntimeError: engine exploded" in text
        assert "mode practical" in text and "mode reflective" in text
        assert "No model answer was generated" in text

    def test_failure_text_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            failure_text([])

    def test_failure_text_repair_only_raises(self) -> None:
        repair = [GenerationFailure.from_exception("repair", RuntimeError("repair died"))]
        with pytest.raises(ValueError):
            failure_text(repair)
        assert generation_failures(repair) == []
        assert generation_failures(_failures() + repair) == _failures()

    def test_degraded_knowledge_text_labels_the_body(self) -> None:
        text = degraded_knowledge_text(_failures(), "• a retrieved cue")
        assert text.startswith(DISCLOSURE_MARKER)
        assert "retrieved knowledge only" in text
        assert "not a model-written answer" in text
        assert text.endswith("• a retrieved cue")

    def test_degraded_knowledge_text_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            degraded_knowledge_text([], "body")

    def test_partial_disclosure_names_failed_modes(self) -> None:
        text = partial_disclosure(_failures()[:1])
        assert "candidate generation failed" in text
        assert "mode practical" in text
        assert "surviving candidate(s) only" in text


class TestApplyPartialDisclosure:
    def test_appends_once_and_recounts_tokens(self) -> None:
        final = FinalResponseRecord(response_id="r1", text="A real answer.")
        assert apply_partial_disclosure(final, _failures()) is True
        assert final.text.startswith("A real answer.")
        assert final.text.count(DISCLOSURE_MARKER) == 1
        assert final.token_count == len(final.text.split())
        assert "degraded turn" in final.enforcement_notes
        # exactly once, however many paths reach for it
        assert apply_partial_disclosure(final, _failures()) is False
        assert final.text.count(DISCLOSURE_MARKER) == 1

    def test_noop_without_visible_failures(self) -> None:
        final = FinalResponseRecord(response_id="r2", text="A real answer.")
        assert apply_partial_disclosure(final, []) is False
        repair = [GenerationFailure.from_exception("repair", RuntimeError("x"))]
        assert apply_partial_disclosure(final, repair) is False
        assert final.text == "A real answer."


# ── Core wiring: total failure ─────────────────────────────────────────────────


class TestTotalFailure:
    def test_all_modes_fail_empty_store_states_failure(self) -> None:
        core = _core_with_engine(_FailingEngine())
        final = core.think(QUERY)
        assert final.mode == "failed"
        assert final.salvage_path == "model_failed"
        assert final.enforcement_passed is False
        assert final.text.startswith(DISCLOSURE_MARKER)
        assert "RuntimeError: engine exploded" in final.text
        assert "No model answer was generated" in final.text
        # the pre-TF-031 text blamed the knowledge for the model
        assert "no knowledge relevant" not in final.text
        assert final.token_count == len(final.text.split())
        turn = core.last_turn_state
        assert turn is not None
        failures = turn_failures(turn)
        assert len(failures) == 2  # both candidate modes attempted
        assert all(f.stage == "candidate" for f in failures)
        assert {f.mode for f in failures} == set(
            turn.scaffold.candidate_modes[:2]  # type: ignore[union-attr]
        )

    def test_model_load_failure_is_stated(self) -> None:
        tmp = tempfile.mkdtemp()
        core = ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=Path(tmp) / "no-such-model.gguf",
        )
        final = core.think(QUERY)
        assert final.mode == "failed"
        failures = turn_failures(core.last_turn_state)  # type: ignore[arg-type]
        assert [f.stage for f in failures] == ["model_load"]
        assert "model_load" in final.text

    def test_empty_engine_text_is_failure_not_candidate(self) -> None:
        core = _core_with_engine(_EmptyEngine())
        final = core.think(QUERY)
        assert final.mode == "failed"
        assert "empty generation" in final.text
        turn = core.last_turn_state
        assert turn is not None and turn.candidates == []
        assert len(turn_failures(turn)) == 2

    def test_backend_error_channel_is_stated(self) -> None:
        core = _core()
        core._unified_backend = _ErrorBackend()
        final = core.think(QUERY)
        assert final.mode == "failed"
        assert "backend exploded" in final.text
        failures = turn_failures(core.last_turn_state)  # type: ignore[arg-type]
        assert [f.stage for f in failures] == ["backend"]
        assert failures[0].detail == "backend exploded"

    def test_backend_exception_is_stated(self) -> None:
        core = _core()
        core._unified_backend = _RaisingBackend()
        final = core.think(QUERY)
        assert final.mode == "failed"
        assert "ConnectionError: backend unreachable" in final.text
        failures = turn_failures(core.last_turn_state)  # type: ignore[arg-type]
        assert [f.stage for f in failures] == ["backend"]

    def test_backend_empty_text_is_stated(self) -> None:
        core = _core()
        core._unified_backend = _EmptyBackend()
        final = core.think(QUERY)
        assert final.mode == "failed"
        assert "empty generation" in final.text

    def test_total_failure_with_knowledge_labels_it_retrieved(self) -> None:
        core = _core_with_engine(_FailingEngine())
        bundle = MemoryActivationBundle(
            activated_records=[
                ActivatedRecord(
                    record_id="syn_fact_001",
                    record_type="fact",
                    score=0.9,
                    cue="the River Sarn ferry crossing takes eleven minutes",
                    raw={"qid": "Q999"},
                )
            ],
            total_retrieved=1,
            retrieval_confidence=0.9,
        )
        core._retrieve = lambda sketch: bundle  # retrieval seam (TF-041 owns F2)
        final = core.think(QUERY)
        assert final.mode == "knowledge_only"  # the body is knowledge, honestly named
        assert final.text.startswith(DISCLOSURE_MARKER)
        assert "retrieved knowledge only" in final.text
        assert "the River Sarn ferry crossing takes eleven minutes" in final.text
        assert "model failure" in final.enforcement_notes


# ── Core wiring: partial failure ───────────────────────────────────────────────


class TestPartialFailure:
    def test_surviving_candidate_carries_the_disclosure(self) -> None:
        core = _core_with_engine(_HalfFailingEngine())
        final = core.think(QUERY)
        assert "eleven minutes" in final.text  # the real composed answer stands
        assert final.text.count(DISCLOSURE_MARKER) == 1
        assert "first mode died" in final.text
        assert "surviving candidate(s) only" in final.text
        assert "degraded turn" in final.enforcement_notes
        assert final.token_count == len(final.text.split())
        failures = turn_failures(core.last_turn_state)  # type: ignore[arg-type]
        visible = generation_failures(failures)
        assert len(visible) == 1 and visible[0].stage == "candidate"


# ── Core wiring: streaming ─────────────────────────────────────────────────────


class TestStreamedHonesty:
    def test_dying_stream_final_is_the_stated_failure(self) -> None:
        core = _core_with_engine(_StreamFailEngine())
        events = list(core.think_stream(QUERY))
        kinds = [e.kind for e in events]
        assert kinds.count("candidate_error") == 2
        assert "candidate_end" not in kinds
        final_event = events[-1]
        assert final_event.kind == "final"
        assert final_event.final is not None
        assert final_event.final.mode == "failed"
        assert "stream broke mid-way" in final_event.final.text
        assert "No model answer was generated" in final_event.final.text

    def test_half_dying_stream_final_carries_disclosure(self) -> None:
        core = _core_with_engine(_HalfStreamEngine())
        events = list(core.think_stream(QUERY))
        kinds = [e.kind for e in events]
        assert kinds.count("candidate_error") == 1
        assert kinds.count("candidate_end") == 1
        final_event = events[-1]
        assert final_event.kind == "final"
        assert final_event.final is not None
        assert "eleven minutes" in final_event.final.text
        assert final_event.final.text.count(DISCLOSURE_MARKER) == 1


# ── Core wiring: what is NOT a failure ─────────────────────────────────────────


class TestNotAFailure:
    def test_deliberate_no_model_turn_is_untouched(self) -> None:
        core = _core()
        final = core.think(QUERY)
        assert final.mode == "knowledge_only"
        assert final.text == "The forge has no knowledge relevant to this query."
        turn = core.last_turn_state
        assert turn is not None and turn_failures(turn) == []

    def test_repair_failure_recorded_not_announced(self) -> None:
        core = _core_with_engine(_RepairFailingEngine())
        final = core.think(QUERY)
        turn = core.last_turn_state
        assert turn is not None and turn.repair_attempted is True
        failures = turn_failures(turn)
        assert [f.stage for f in failures] == ["repair"]
        assert DISCLOSURE_MARKER not in final.text

    def test_interrupted_turn_records_no_failure(self) -> None:
        core = _core_with_engine(_FailingEngine())
        token = InterruptToken()
        token.interrupt("caller stopped it")
        with pytest.raises(TurnInterrupted):
            core.think(QUERY, interrupt=token)
        turn = core.last_turn_state
        assert turn is not None and turn_failures(turn) == []
