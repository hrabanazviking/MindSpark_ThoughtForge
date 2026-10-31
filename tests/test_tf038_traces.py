"""
TF-038 — Logging & traces: a turn's full cognition trace
exportable, and the logging surface pinned directly.

The discipline lives in ``src/thoughtforge/cognition/traces.py``
(the trace format, the exporter, the JSON writer, the pure
checker) plus the core's ``last_turn_trace()`` and the direct
tests of ``thoughtforge.utils.logging_setup`` below (TF-016's
D-COV-3 logging_setup share). Write-up:
``docs/architecture/LOGGING_AND_TRACES.md``.

Pinned here:

  * the export carries every section in TRACE_SECTIONS order —
    turn / conversation / route / retrieval / scaffold /
    candidates / fragments / generation_failures / final /
    writeback / timings — as detached JSON-safe data: mutating an
    export never touches the live turn state, and a non-JSON
    value in a store record's raw payload becomes its str(),
    visible, never dropped,
  * trace ≠ fingerprint: the trace INCLUDES timestamps and
    timings (TF-036's fingerprint excludes them by ruling) and
    exports an unfinished turn honestly (final None, phase
    stated) where the fingerprint refuses,
  * measured through real turns: knowledge-only, stub-backend
    (candidates + fragments + salvage path, every cross-reference
    resolving), stated failure (TF-031's records riding the
    trace), and an interrupted turn (TF-027's cut, recorded),
  * the pure checker catches, by name: wrong format, a missing
    section, an unknown phase, complete-but-no-final, a final
    whose turn id disagrees, dangling source candidate /
    fragment references, a fragment naming an absent candidate,
    a duplicated candidate id, a negative timing,
  * logging_setup (D-COV-3's share): one-shot configuration —
    level honoured, an unknown level falling back to INFO,
    console / file switches honoured, the rotating file created
    where the config says, and a second call changing nothing.

Proof grade: FIXTURE generation sources on this VM (knowledge-
only core, stub backend / engine). No live model was traced; a
live model's turn differs only in who wrote the candidates.
"""

from __future__ import annotations

import copy
import json
import logging
import logging.handlers
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.determinism import turn_fingerprint
from thoughtforge.cognition.interruption import InterruptToken, TurnInterrupted
from thoughtforge.cognition.traces import (
    TRACE_FORMAT,
    TRACE_SECTIONS,
    export_turn_trace,
    trace_problems,
    trace_to_json,
    write_turn_trace,
)
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.models import (
    ActivatedRecord,
    MemoryActivationBundle,
    RuntimeTurnState,
    WritebackRecord,
)
from thoughtforge.utils import logging_setup

TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)
FERRY = "What time does the ferry leave?"


# ── Fixture generation sources ────────────────────────────────────────────────


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=TEXT, tokens_generated=len(TEXT.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf038-stub"


class BadBackend(UnifiedBackend):
    """Fails through the backend error channel (TF-031's shape)."""

    def generate(self, request):
        return GenerationResponse(text="", error="boom-engine-dead")

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf038-bad"


class _FireOnGenerateEngine:
    """Fires the interrupt token during its first generate()."""

    def __init__(self, token: InterruptToken) -> None:
        self.token = token
        self.calls = 0

    def generate(self, prompt, params=None):
        self.calls += 1
        if self.calls == 1:
            self.token.interrupt("user pressed stop")
        return SimpleNamespace(
            text=TEXT, tokens_generated=len(TEXT.split())
        )


def _core(**kwargs) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
        **kwargs,
    )


def _backend_trace() -> dict:
    core = _core(backend=StubBackend())
    core.think(FERRY)
    trace = core.last_turn_trace()
    assert trace is not None
    return trace


# ── Format & sections ─────────────────────────────────────────────────────────


class TestFormatAndSections:
    def test_format_and_section_tables(self) -> None:
        assert TRACE_FORMAT == "thoughtforge-turn-trace/1"
        assert TRACE_SECTIONS == (
            "turn",
            "conversation",
            "route",
            "retrieval",
            "scaffold",
            "candidates",
            "fragments",
            "generation_failures",
            "final",
            "writeback",
            "timings",
        )

    def test_export_keys_exactly_format_turn_id_sections(self) -> None:
        trace = export_turn_trace(RuntimeTurnState(turn_id="turn_bare"))
        assert list(trace.keys()) == [
            "format",
            "turn_id",
            *TRACE_SECTIONS,
        ]
        assert trace["format"] == TRACE_FORMAT

    def test_bare_turn_exports_honestly(self) -> None:
        trace = export_turn_trace(RuntimeTurnState(turn_id="turn_bare"))
        assert trace["conversation"] is None
        assert trace["route"] is None
        assert trace["retrieval"] is None
        assert trace["scaffold"] is None
        assert trace["candidates"] == []
        assert trace["final"] is None
        assert trace["writeback"] is None
        assert trace["turn"]["phase"] == "created"
        assert trace["turn"]["is_complete"] is False
        assert trace_problems(trace) == []


# ── Real turns, measured ──────────────────────────────────────────────────────


class TestKnowledgeOnlyTrace:
    def test_knowledge_only_turn_trace(self) -> None:
        core = _core()
        assert core.last_turn_trace() is None  # before the first turn
        core.think(FERRY)
        trace = core.last_turn_trace()
        assert trace is not None
        assert trace_problems(trace) == []
        assert trace["turn_id"] == core.last_turn_state.turn_id
        assert trace["turn"]["is_complete"] is True
        assert trace["turn"]["started_at"]
        assert trace["turn"]["completed_at"]
        assert trace["route"]["raw_text"] == FERRY
        assert trace["retrieval"]["activated_records"] == []
        assert trace["candidates"] == []
        assert trace["fragments"] == []
        assert trace["generation_failures"] == []
        assert trace["final"]["mode"] == "knowledge_only"
        assert trace["final"]["turn_id"] == trace["turn_id"]
        assert trace["timings"]["salvage_ms"] == 0  # did not run
        # The shipped personality fallback is in force (TF-030's
        # discovery): the trace names it.
        assert trace["conversation"]["persona_id"] == "pers_001"
        assert trace["conversation"]["profile_id"] is None

    def test_conversation_section_carries_selections(self) -> None:
        core = _core(
            config={
                "conversation": {"profile_id": "phone_low"},
                "determinism": {"seed": 7},
            }
        )
        core.think(FERRY)
        conversation = core.last_turn_trace()["conversation"]
        assert conversation["profile_id"] == "phone_low"
        assert conversation["determinism_seed"] == 7


class TestBackendTrace:
    def test_candidates_fragments_final_whole(self) -> None:
        trace = _backend_trace()
        assert trace_problems(trace) == []
        assert len(trace["candidates"]) == 1
        cand = trace["candidates"][0]
        assert cand["text"] == TEXT
        assert cand["scores"]["composite"] == pytest.approx(
            _live_composite(trace)
        )
        assert trace["fragments"], "salvage produced fragments"
        kept = [f for f in trace["fragments"] if f["keep"]]
        assert kept, "the composed answer kept at least one fragment"
        final = trace["final"]
        assert final["source_candidate_ids"] == [cand["candidate_id"]]
        assert set(final["source_fragment_ids"]) <= {
            f["fragment_id"] for f in trace["fragments"]
        }
        assert final["scores"]["quality_tier"] in {
            "excellent",
            "strong",
            "usable",
            "weak",
            "reject",
        }
        assert final["enforcement_passed"] is True

    def test_json_round_trip_and_file_write(self, tmp_path) -> None:
        trace = _backend_trace()
        assert json.loads(trace_to_json(trace)) == trace
        target = write_turn_trace(trace, tmp_path / "deep" / "trace.json")
        assert target == tmp_path / "deep" / "trace.json"
        assert json.loads(target.read_text(encoding="utf-8")) == trace


def _live_composite(trace) -> float:
    scores = trace["candidates"][0]["scores"]
    return (
        scores["relevance"] * 0.28
        + scores["clarity"] * 0.18
        + scores["coherence"] * 0.16
        + scores["personality_fit"] * 0.16
        + scores["specificity"] * 0.14
        + scores["goal_fit"] * 0.08
        - scores["genericness_penalty"] * 0.10
    )


class TestFailureTrace:
    def test_stated_failure_rides_the_trace(self) -> None:
        core = _core(backend=BadBackend())
        core.think(FERRY)
        trace = core.last_turn_trace()
        assert trace_problems(trace) == []
        assert trace["generation_failures"] == [
            {
                "stage": "backend",
                "mode": "practical",
                "detail": "boom-engine-dead",
            }
        ]
        assert trace["final"]["mode"] == "failed"
        assert trace["candidates"] == []


class TestInterruptedTrace:
    def test_interrupted_turn_exports_cut_not_completion(self) -> None:
        token = InterruptToken()
        core = _core()
        core._engine = _FireOnGenerateEngine(token)
        with pytest.raises(TurnInterrupted):
            core.think(FERRY, interrupt=token)
        trace = core.last_turn_trace()
        assert trace_problems(trace) == []
        assert trace["turn"]["is_interrupted"] is True
        assert trace["turn"]["interrupt_reason"] == "user pressed stop"
        assert trace["turn"]["is_complete"] is False
        assert trace["final"] is None
        assert trace["turn"]["completed_at"] == ""


# ── Detachment & JSON safety ──────────────────────────────────────────────────


class TestDetachment:
    def test_mutating_the_export_never_touches_the_live_turn(self) -> None:
        core = _core(backend=StubBackend())
        core.think(FERRY)
        trace = core.last_turn_trace()
        trace["candidates"][0]["text"] = "TAMPERED"
        trace["scaffold"]["tone"].append("TAMPERED")
        trace["final"]["text"] = "TAMPERED"
        live = core.last_turn_state
        assert live.candidates[0].text == TEXT
        assert "TAMPERED" not in live.scaffold.tone
        assert live.final_response.text != "TAMPERED"
        fresh = core.last_turn_trace()
        assert fresh["candidates"][0]["text"] == TEXT
        assert trace_problems(fresh) == []

    def test_non_json_raw_becomes_visible_str(self) -> None:
        turn = RuntimeTurnState(turn_id="turn_populated")
        marker = object()
        turn.memory_bundle = MemoryActivationBundle(
            personality_core_id="pers_001",
            activated_records=[
                ActivatedRecord(
                    record_id="rec_1",
                    record_type="fact",
                    score=0.9,
                    cue="ferry eleven minutes",
                    raw={"nested": {"a": [1, 2]}, "weird": marker},
                )
            ],
            total_retrieved=1,
            retrieval_confidence=0.9,
        )
        turn.writeback = WritebackRecord(
            turn_id="turn_populated", new_facts=[{"fact": "ferry"}]
        )
        trace = export_turn_trace(turn)
        raw = trace["retrieval"]["activated_records"][0]["raw"]
        assert raw["nested"] == {"a": [1, 2]}
        assert raw["weird"] == str(marker)  # visible, never dropped
        assert trace["writeback"]["new_facts"] == [{"fact": "ferry"}]
        assert json.loads(trace_to_json(trace)) == trace
        assert trace_problems(trace) == []


# ── Trace ≠ fingerprint ───────────────────────────────────────────────────────


class TestTraceIsNotFingerprint:
    def test_trace_includes_what_fingerprint_excludes(self) -> None:
        core = _core(backend=StubBackend())
        core.think(FERRY)
        fingerprint = turn_fingerprint(core.last_turn_state)
        assert "timings" not in fingerprint
        assert "started_at" not in json.dumps(fingerprint)
        trace = core.last_turn_trace()
        assert trace["timings"]["retrieval_ms"] >= 0
        assert trace["turn"]["started_at"]

    def test_fingerprint_refuses_unfinished_export_does_not(self) -> None:
        turn = RuntimeTurnState(turn_id="turn_cut")
        with pytest.raises(ValueError):
            turn_fingerprint(turn)
        trace = export_turn_trace(turn)
        assert trace["final"] is None
        assert trace_problems(trace) == []


# ── The checker catches tampering ─────────────────────────────────────────────


class TestTraceProblems:
    def test_not_a_mapping(self) -> None:
        assert trace_problems([1, 2]) == ["trace is not a mapping"]

    def test_wrong_format_and_missing_section(self) -> None:
        trace = _backend_trace()
        trace["format"] = "something-else/9"
        del trace["fragments"]
        problems = trace_problems(trace)
        assert any("format" in p for p in problems)
        assert any("missing section 'fragments'" in p for p in problems)

    def test_unknown_phase(self) -> None:
        trace = _backend_trace()
        trace["turn"]["phase"] = "sideways"
        assert any("not a known turn phase" in p for p in trace_problems(trace))

    def test_complete_without_final(self) -> None:
        trace = _backend_trace()
        trace["final"] = None
        assert any(
            "marked complete but carries no final" in p
            for p in trace_problems(trace)
        )

    def test_final_turn_id_disagreement(self) -> None:
        trace = _backend_trace()
        trace["final"]["turn_id"] = "turn_someone_else"
        assert any("disagrees" in p for p in trace_problems(trace))

    def test_dangling_references(self) -> None:
        trace = _backend_trace()
        broken = copy.deepcopy(trace)
        broken["final"]["source_candidate_ids"] = ["cand_ghost"]
        assert any(
            "source candidate 'cand_ghost'" in p
            for p in trace_problems(broken)
        )
        broken = copy.deepcopy(trace)
        broken["final"]["source_fragment_ids"] = ["frag_ghost"]
        assert any(
            "source fragment 'frag_ghost'" in p
            for p in trace_problems(broken)
        )
        broken = copy.deepcopy(trace)
        broken["fragments"][0]["source_candidate_id"] = "cand_ghost"
        assert any(
            "names source candidate 'cand_ghost'" in p
            for p in trace_problems(broken)
        )

    def test_duplicate_candidate_and_negative_timing(self) -> None:
        trace = _backend_trace()
        trace["candidates"].append(copy.deepcopy(trace["candidates"][0]))
        trace["timings"]["generation_ms"] = -1
        problems = trace_problems(trace)
        assert any("duplicated candidate_id" in p for p in problems)
        assert any("negative" in p for p in problems)


# ── logging_setup, directly (TF-016 D-COV-3, logging_setup share) ────────────


@pytest.fixture()
def root_logger_guard():
    """Preserve the root logger + the module's one-shot flag.

    Pytest's logging plugin keeps its own capture handlers on the
    root logger during a test, so the tests below measure handler
    DELTAS against the baseline they capture at their start; this
    fixture only guarantees nothing setup_logging added survives
    the test, and the level / flag are restored.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_flag = logging_setup._configured
    logging_setup._configured = False
    yield root
    for handler in root.handlers[:]:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)
    logging_setup._configured = saved_flag


class TestLoggingSetup:
    def test_level_honoured_no_handlers_when_both_switches_off(
        self, root_logger_guard
    ) -> None:
        root = root_logger_guard
        baseline = root.handlers[:]
        logging_setup.setup_logging(
            {"logging": {"level": "DEBUG", "log_to_file": False,
                         "log_to_console": False}}
        )
        assert root.level == logging.DEBUG
        assert root.handlers == baseline  # neither switch added one
        assert logging_setup._configured is True

    def test_second_call_changes_nothing(self, root_logger_guard) -> None:
        root = root_logger_guard
        logging_setup.setup_logging(
            {"logging": {"level": "DEBUG", "log_to_file": False,
                         "log_to_console": False}}
        )
        after_first = root.handlers[:]
        logging_setup.setup_logging(
            {"logging": {"level": "CRITICAL", "log_to_file": False,
                         "log_to_console": True}}
        )
        assert root.level == logging.DEBUG  # one-shot: the first stands
        assert root.handlers == after_first

    def test_unknown_level_falls_back_to_info(self, root_logger_guard) -> None:
        root = root_logger_guard
        logging_setup.setup_logging(
            {"logging": {"level": "SHOUTY", "log_to_file": False,
                         "log_to_console": False}}
        )
        assert root.level == logging.INFO

    def test_console_switch_adds_stream_handler(self, root_logger_guard) -> None:
        root = root_logger_guard
        baseline = root.handlers[:]
        logging_setup.setup_logging(
            {"logging": {"level": "INFO", "log_to_file": False,
                         "log_to_console": True}}
        )
        added = [h for h in root.handlers if h not in baseline]
        assert len(added) == 1
        assert type(added[0]) is logging.StreamHandler

    def test_file_switch_creates_rotating_file(
        self, root_logger_guard, tmp_path
    ) -> None:
        root = root_logger_guard
        logging_setup.setup_logging(
            {"logging": {"level": "INFO", "log_to_file": True,
                         "log_to_console": False,
                         "log_dir": str(tmp_path)}}
        )
        assert (tmp_path / "thoughtforge.log").exists()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 10 * 1024 * 1024
        assert file_handlers[0].backupCount == 5
