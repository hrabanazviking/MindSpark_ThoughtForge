"""
TF-025 — Turn State: a typed turn-state object replaces ad-hoc
argument bundles; the state is phased, timed, and retained.

The discipline lives in ``src/thoughtforge/cognition/turn_state.py``
(phases as data + ``advance_phase``) and is carried by
``RuntimeTurnState`` (knowledge/models.py §14); the shared-helper
half (D-DUP-1) lives in ``src/thoughtforge/utils/common.py``.
Write-up: ``docs/architecture/TURN_STATE.md``.

Pinned here:

  * phases: the 7-phase order, forward-only movement, equal-phase
    no-op, backward / unknown moves raising,
  * RuntimeTurnState's new fields' defaults (phase "created",
    repair_attempted False, salvage_ms 0, is_complete False),
  * think() end-to-end on the knowledge-only path: the finished
    state is retained on ``last_turn_state``, is the object whose
    final_response IS the returned record, phase completed,
    salvage_ms exactly 0 (no salvage ran), tokens mirror the final,
  * think() with a stub engine: candidates exist, so salvage_ms is
    MEASURED (an int >= 0 recorded from a real compose call — the
    perf tracker also gains a think.salvage sample),
  * a fresh core has no last turn state; a second think() replaces
    the first turn's state (turn ids differ),
  * D-DUP-1: exactly one implementation per copied helper — the old
    private names in models / chat_history / core / salvage /
    ollama / lmstudio ARE the shared functions (identity, not just
    equal behaviour), and the shared functions behave as the copies
    did (sentence splitting, message building, ISO timestamps).
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition import core as core_module
from thoughtforge.cognition.chat_history import ChatMessage
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.turn_state import PHASE_ORDER, advance_phase, phase_index
from thoughtforge.inference import lmstudio_backend, ollama_backend
from thoughtforge.inference.unified_backend import GenerationRequest
from thoughtforge.knowledge import models as models_module
from thoughtforge.knowledge.models import RuntimeTurnState
from thoughtforge.refinement import salvage as salvage_module
from thoughtforge.utils.common import build_messages, now_iso, split_sentences
from thoughtforge.utils.perf import get_perf_tracker


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )


# ── Phases ─────────────────────────────────────────────────────────────────────


class TestPhaseLaw:
    def test_phase_order_is_the_seven_phases(self) -> None:
        assert PHASE_ORDER == (
            "created", "routed", "retrieved", "scaffolded",
            "generated", "composed", "completed",
        )

    def test_phase_index_positions(self) -> None:
        assert phase_index("created") == 0
        assert phase_index("completed") == len(PHASE_ORDER) - 1

    def test_phase_index_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            phase_index("teleported")

    def test_advance_forward_moves_and_reports(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        assert advance_phase(turn, "routed") is True
        assert turn.phase == "routed"

    def test_advance_can_skip_phases_forward(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        assert advance_phase(turn, "generated") is True
        assert turn.phase == "generated"

    def test_advance_equal_is_noop(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        advance_phase(turn, "routed")
        assert advance_phase(turn, "routed") is False
        assert turn.phase == "routed"

    def test_advance_backward_raises_and_keeps_phase(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        advance_phase(turn, "composed")
        with pytest.raises(ValueError):
            advance_phase(turn, "routed")
        assert turn.phase == "composed"

    def test_advance_unknown_raises(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        with pytest.raises(ValueError):
            advance_phase(turn, "vanished")


# ── RuntimeTurnState defaults ──────────────────────────────────────────────────


class TestTurnStateDefaults:
    def test_new_turn_defaults(self) -> None:
        turn = RuntimeTurnState(turn_id="turn_x")
        assert turn.phase == "created"
        assert turn.repair_attempted is False
        assert turn.salvage_ms == 0
        assert turn.is_complete is False
        assert turn.completed_at == ""

    def test_is_complete_needs_phase_and_timestamp(self) -> None:
        turn = RuntimeTurnState(turn_id="turn_x")
        advance_phase(turn, "completed")
        assert turn.is_complete is False          # no timestamp yet
        turn.completed_at = now_iso()
        assert turn.is_complete is True


# ── think(): the state is retained, phased, timed ─────────────────────────────


class TestThinkRetainsState:
    def test_fresh_core_has_no_last_turn(self) -> None:
        assert _core().last_turn_state is None

    def test_knowledge_only_turn_state(self) -> None:
        core = _core()
        result = core.think("What is Yggdrasil?")
        turn = core.last_turn_state
        assert turn is not None
        assert turn.turn_id == result.turn_id
        assert turn.phase == "completed"
        assert turn.is_complete is True
        assert turn.final_response is result
        assert turn.candidates == []
        # Knowledge-only: no salvage ran, so salvage_ms is exactly 0 —
        # the TF-025 ruling: 0 means "did not run", never "unmeasured".
        assert turn.salvage_ms == 0
        assert turn.repair_attempted is False
        assert turn.total_tokens_used == result.token_count
        assert turn.input_sketch is not None
        assert turn.memory_bundle is not None
        assert turn.scaffold is not None

    def test_second_think_replaces_state(self) -> None:
        core = _core()
        core.think("First question about the world tree?")
        first = core.last_turn_state
        core.think("Second question about the world tree?")
        second = core.last_turn_state
        assert first is not None and second is not None
        assert second is not first
        assert second.turn_id != first.turn_id
        assert first.phase == "completed"        # the old object is untouched

    def test_stub_engine_turn_measures_salvage(self) -> None:
        core = _core()
        text = (
            "Yggdrasil is the world tree that joins the nine realms. "
            "Its roots reach into deep wells of wisdom and fate. "
            "Odin hung upon its branches for nine nights seeking runes."
        )

        class _StubEngine:
            def generate(self, prompt, params=None):
                return SimpleNamespace(text=text, tokens_generated=len(text.split()))

        core._engine = _StubEngine()
        tracker = get_perf_tracker()
        before = tracker.summary(last_n=100000).events.get(
            "think.salvage", SimpleNamespace(count=0)).count
        result = core.think("What is Yggdrasil?")
        turn = core.last_turn_state
        assert turn is not None
        assert turn.candidates, "stub engine should have produced candidates"
        assert turn.final_response is result
        assert turn.phase == "completed"
        # Measured, not the old never-assigned default-by-accident:
        # an int recorded around a real compose call, and the perf
        # tracker gained exactly one think.salvage sample for it.
        assert isinstance(turn.salvage_ms, int) and turn.salvage_ms >= 0
        after = tracker.summary(last_n=100000).events.get(
            "think.salvage", SimpleNamespace(count=0)).count
        assert after == before + 1


# ── D-DUP-1: one shared home per copied helper ────────────────────────────────


class TestSharedHelpers:
    def test_now_iso_is_one_function(self) -> None:
        assert models_module._now_iso is now_iso
        assert core_module._now_iso is now_iso
        from thoughtforge.cognition import chat_history
        assert chat_history._now_iso is now_iso

    def test_split_sentences_is_one_function(self) -> None:
        assert core_module._split_sentences is split_sentences
        assert salvage_module._split_sentences is split_sentences

    def test_build_messages_is_one_function(self) -> None:
        assert ollama_backend._build_messages is build_messages
        assert lmstudio_backend._build_messages is build_messages

    def test_now_iso_is_tz_aware_iso(self) -> None:
        parsed = datetime.fromisoformat(now_iso())
        assert parsed.tzinfo is not None

    def test_split_sentences_behaviour_unchanged(self) -> None:
        assert split_sentences("First one. Second one! Third?") == [
            "First one.", "Second one!", "Third?",
        ]
        assert split_sentences("Just one sentence") == ["Just one sentence"]
        assert split_sentences("") == []

    def test_build_messages_passthrough(self) -> None:
        msgs = [{"role": "user", "content": "already built"}]
        req = GenerationRequest(prompt="ignored", messages=msgs)
        assert build_messages(req) is msgs

    def test_build_messages_system_and_user(self) -> None:
        req = GenerationRequest(prompt="hello", system_prompt="be terse")
        assert build_messages(req) == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]

    def test_build_messages_user_only(self) -> None:
        req = GenerationRequest(prompt="hello")
        assert build_messages(req) == [{"role": "user", "content": "hello"}]

    def test_chat_message_still_timestamps_via_alias(self) -> None:
        msg = ChatMessage(role="user", content="hi")
        assert datetime.fromisoformat(msg.timestamp).tzinfo is not None
