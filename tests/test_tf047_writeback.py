"""TF-047 — conversation → memory: the gated writeback leg.

Pins knowledge/writeback.py against the live tree: the
enforcement gate (a failed turn writes nothing, every item
refused by name), the TF-042 contract gate and the provenance
stamp it closes, the TF-044 score gates at their boundaries,
reinforce / archive / delete by id (identity unreachable),
thread updates, and the core wiring in _finish_turn.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.knowledge import scoring
from thoughtforge.knowledge.ingestion_contract import contract_problems
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    UserFactRecord,
    UserPreferenceRecord,
    WritebackRecord,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.writeback import (
    SCORE_GATES,
    WRITEBACK_SOURCE,
    WritebackOutcome,
    apply_writeback,
    writeback_problems,
)

FULL = {
    "durability": 1.0, "novelty": 1.0, "confidence": 1.0,
    "importance": 1.0, "future_relevance": 1.0,
}
EP_FULL = {
    "novelty": 1.0, "confidence": 1.0, "importance": 1.0,
    "future_relevance": 1.0, "thread_relevance": 1.0,
}
PAT_FULL = {
    "final_response_quality": 1.0, "pattern_reusability": 1.0,
    "future_relevance": 1.0,
}


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path)


def _wb(**kwargs) -> WritebackRecord:
    return WritebackRecord(turn_id="turn_t", **kwargs)


# ── Self-checks and gate data ────────────────────────────────────────────────


class TestSelfCheck:
    def test_writeback_problems_empty(self) -> None:
        assert writeback_problems() == []

    def test_contract_problems_empty_after_stamp(self) -> None:
        # TF-042's pin flipped by TF-047: the four writeback kinds
        # carry the stamp; identity and thread state do not.
        assert contract_problems() == []

    def test_score_gates_are_scorings_own(self) -> None:
        assert SCORE_GATES["user_preference"][1] == scoring.MIN_PREFERENCE_WRITEBACK_SCORE == 0.72
        assert SCORE_GATES["episode"][1] == scoring.MIN_EPISODE_WRITEBACK_SCORE == 0.58
        assert SCORE_GATES["response_pattern"][1] == scoring.MIN_PATTERN_LEARNING_SCORE == 0.78
        assert WRITEBACK_SOURCE == "conversation_writeback"


# ── The enforcement gate ─────────────────────────────────────────────────────


class TestEnforcementGate:
    def test_failed_turn_writes_nothing(self, store: MemoryStore) -> None:
        wb = _wb(
            new_facts=[{"summary": "the ferry leaves at dawn"}],
            new_preferences=[{"category": "tone", "summary": "plain", "components": FULL}],
            new_episodes=[{"summary": "a moment", "components": EP_FULL}],
            new_patterns=[{"summary": "shape", "components": PAT_FULL}],
            thread_updates={"topic": "ferries"},
            records_to_reinforce=["x"], records_to_archive=["y"],
            records_to_delete=["z"],
        )
        outcome = apply_writeback(store, wb, enforcement_passed=False)
        assert outcome.enforcement_gated is True
        assert outcome.applied == []
        # Every requested item refused by name: 4 new + thread + 3 lists.
        assert len(outcome.refused) == 8
        assert all("did not pass enforcement" in r["reasons"][0] for r in outcome.refused)
        assert store.load_profile_records() == []
        assert store.load_episodes() == []
        assert store.load_patterns() == []
        assert store.load_thread_state() is None

    def test_empty_writeback_is_a_clean_noop(self, store: MemoryStore) -> None:
        outcome = apply_writeback(store, _wb(), enforcement_passed=True)
        assert outcome == WritebackOutcome(turn_id="turn_t")
        assert outcome.written is False


# ── Facts: contract gate + provenance ────────────────────────────────────────


class TestFacts:
    def test_fact_written_with_default_stamp(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store, _wb(new_facts=[{"summary": "the ferry leaves at dawn"}]),
            enforcement_passed=True,
        )
        assert len(outcome.applied) == 1
        assert outcome.applied[0]["score"] is None  # facts carry no formula
        (fact,) = [r for r in store.load_profile_records() if isinstance(r, UserFactRecord)]
        assert fact.summary == "the ferry leaves at dawn"
        assert fact.source == "conversation_writeback"

    def test_fact_keeps_its_own_valid_stamp(self, store: MemoryStore) -> None:
        apply_writeback(
            store,
            _wb(new_facts=[{"summary": "user said so", "source": "user_stated"}]),
            enforcement_passed=True,
        )
        (fact,) = store.load_profile_records()
        assert fact.source == "user_stated"

    def test_fact_with_foreign_source_refused(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store,
            _wb(new_facts=[{"summary": "from mars", "source": "mars"}]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert "provenance" in outcome.refused[0]["reasons"][0]
        assert store.load_profile_records() == []

    def test_contentless_fact_refused(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store, _wb(new_facts=[{"summary": "  "}]), enforcement_passed=True
        )
        assert outcome.applied == []
        assert any("no content" in r for r in outcome.refused[0]["reasons"])


# ── Score gates at their boundaries ──────────────────────────────────────────


def _uniform(value: float, keys: tuple[str, ...]) -> dict[str, float]:
    # The writeback weights sum to 1.0 per formula (TF-041), so a
    # uniform component set scores exactly its value.
    return {k: value for k in keys}


class TestScoreGates:
    def test_preference_at_gate_applied(self, store: MemoryStore) -> None:
        entry = {
            "category": "tone", "summary": "plain speech",
            "components": _uniform(0.72, tuple(FULL)),
        }
        outcome = apply_writeback(store, _wb(new_preferences=[entry]), enforcement_passed=True)
        assert len(outcome.applied) == 1
        assert outcome.applied[0]["score"] == pytest.approx(0.72)
        (pref,) = store.load_profile_records()
        assert isinstance(pref, UserPreferenceRecord) and pref.source == WRITEBACK_SOURCE

    def test_preference_below_gate_refused_with_score(self, store: MemoryStore) -> None:
        entry = {
            "category": "tone", "summary": "plain speech",
            "components": _uniform(0.71, tuple(FULL)),
        }
        outcome = apply_writeback(store, _wb(new_preferences=[entry]), enforcement_passed=True)
        assert outcome.applied == []
        assert "0.710" in outcome.refused[0]["reasons"][0]
        assert "0.72" in outcome.refused[0]["reasons"][0]
        assert store.load_profile_records() == []

    def test_preference_without_category_refused_by_contract(
        self, store: MemoryStore
    ) -> None:
        entry = {"summary": "plain speech", "components": FULL}
        outcome = apply_writeback(store, _wb(new_preferences=[entry]), enforcement_passed=True)
        assert outcome.applied == []
        assert any("category" in r for r in outcome.refused[0]["reasons"])

    def test_bad_component_refused(self, store: MemoryStore) -> None:
        entry = {
            "category": "tone", "summary": "plain speech",
            "components": {**FULL, "novelty": 1.5},
        }
        outcome = apply_writeback(store, _wb(new_preferences=[entry]), enforcement_passed=True)
        assert outcome.applied == []
        assert any("novelty" in r for r in outcome.refused[0]["reasons"])

    def test_episode_at_gate_applied_and_links_carried(self, store: MemoryStore) -> None:
        entry = {
            "summary": "planned the crossing",
            "components": _uniform(0.58, tuple(EP_FULL)),
            "linked_facts": ["fct_x"], "linked_preferences": ["usr_x"],
        }
        outcome = apply_writeback(store, _wb(new_episodes=[entry]), enforcement_passed=True)
        assert len(outcome.applied) == 1
        (ep,) = store.load_episodes()
        assert isinstance(ep, EpisodicMemoryRecord)
        assert ep.linked_facts == ["fct_x"] and ep.source == WRITEBACK_SOURCE

    def test_episode_below_gate_refused(self, store: MemoryStore) -> None:
        entry = {"summary": "a shrug", "components": _uniform(0.57, tuple(EP_FULL))}
        outcome = apply_writeback(store, _wb(new_episodes=[entry]), enforcement_passed=True)
        assert outcome.applied == [] and store.load_episodes() == []

    def test_pattern_quality_field_feeds_its_component(self, store: MemoryStore) -> None:
        # No explicit final_response_quality: the entry's quality
        # field feeds it; the rest explicit → score exactly 1.0.
        entry = {
            "summary": "answer, then evidence",
            "pattern_shape": ["answer", "evidence"],
            "quality": 1.0,
            "components": {"pattern_reusability": 1.0, "future_relevance": 1.0},
        }
        outcome = apply_writeback(store, _wb(new_patterns=[entry]), enforcement_passed=True)
        assert len(outcome.applied) == 1
        assert outcome.applied[0]["score"] == pytest.approx(1.0)

    def test_pattern_neutral_fallback_stays_below_gate(self, store: MemoryStore) -> None:
        # quality defaults 0.6, other components neutral 0.5:
        # 0.45*0.6 + 0.30*0.5 + 0.25*0.5 = 0.545 < 0.78.
        entry = {"summary": "some shape", "pattern_shape": ["a"]}
        outcome = apply_writeback(store, _wb(new_patterns=[entry]), enforcement_passed=True)
        assert outcome.applied == []
        assert "0.545" in outcome.refused[0]["reasons"][0]


# ── Reinforce / archive / delete ─────────────────────────────────────────────


class TestMutations:
    def _seed(self, store: MemoryStore) -> tuple[str, str]:
        store.append_profile_record(
            UserFactRecord(record_id="fct_seed", summary="seeded fact")
        )
        store.append_episode(
            EpisodicMemoryRecord(record_id="ep_seed", summary="seeded moment")
        )
        return "fct_seed", "ep_seed"

    def test_reinforce_increments_and_marks(self, store: MemoryStore) -> None:
        fact_id, _ = self._seed(store)
        outcome = apply_writeback(
            store, _wb(records_to_reinforce=[fact_id]), enforcement_passed=True
        )
        assert outcome.applied_ids == [fact_id]
        (fact,) = store.load_profile_records()
        assert fact.times_reinforced == 1 and fact.status == "reinforced"

    def test_archive_drops_record_from_file(self, store: MemoryStore) -> None:
        fact_id, _ = self._seed(store)
        outcome = apply_writeback(
            store, _wb(records_to_archive=[fact_id]), enforcement_passed=True
        )
        assert outcome.applied_ids == [fact_id]
        assert store.load_profile_records() == []

    def test_delete_drops_episode_from_file(self, store: MemoryStore) -> None:
        _, ep_id = self._seed(store)
        apply_writeback(store, _wb(records_to_delete=[ep_id]), enforcement_passed=True)
        assert store.load_episodes() == []

    def test_unknown_id_refused_not_silent(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store, _wb(records_to_delete=["nope"]), enforcement_passed=True
        )
        assert outcome.applied == []
        assert "not found" in outcome.refused[0]["reasons"][0]

    def test_identity_unreachable_by_writeback(self, store: MemoryStore) -> None:
        store.save_personality_core(PersonalityCoreRecord(record_id="pers_seed"))
        outcome = apply_writeback(
            store,
            _wb(records_to_delete=["pers_seed"], records_to_archive=["pers_seed"]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert len(outcome.refused) == 2
        assert store.load_personality_core() is not None


# ── Thread updates ───────────────────────────────────────────────────────────


class TestThreadUpdates:
    def test_creates_thread_with_topic(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store,
            _wb(thread_updates={"topic": "ferries", "summary": "crossing plans"}),
            enforcement_passed=True,
        )
        assert len(outcome.applied) == 1
        thread = store.load_thread_state()
        assert thread is not None
        assert thread.topic == "ferries" and thread.summary == "crossing plans"

    def test_topicless_creation_refused(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store, _wb(thread_updates={"summary": "no topic"}), enforcement_passed=True
        )
        assert outcome.applied == []
        assert store.load_thread_state() is None

    def test_update_replaces_named_fields_and_resets_counter(
        self, store: MemoryStore
    ) -> None:
        apply_writeback(
            store, _wb(thread_updates={"topic": "ferries"}), enforcement_passed=True
        )
        thread = store.load_thread_state()
        assert thread is not None
        thread.turns_since_touch = 4
        store.save_thread_state(thread)
        apply_writeback(
            store, _wb(thread_updates={"summary": "new summary"}),
            enforcement_passed=True,
        )
        thread = store.load_thread_state()
        assert thread is not None
        assert thread.summary == "new summary" and thread.topic == "ferries"
        assert thread.turns_since_touch == 0

    def test_unknown_key_refused(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store, _wb(thread_updates={"mood": "calm"}), enforcement_passed=True
        )
        assert outcome.applied == []
        assert any("unknown thread update keys" in r for r in outcome.refused[0]["reasons"])


# ── Core wiring ──────────────────────────────────────────────────────────────


class TestCoreWiring:
    @pytest.fixture()
    def core(self, tmp_path: Path) -> ThoughtForgeCore:
        return ThoughtForgeCore(
            config={}, memory_dir=tmp_path, db_path=tmp_path / "no.db",
        )

    def test_core_apply_writeback_lands_and_is_retained(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        outcome = core.apply_writeback(
            _wb(new_facts=[{"summary": "the ferry leaves at dawn"}]),
            enforcement_passed=True,
        )
        assert core.last_writeback_outcome is outcome
        store = MemoryStore(memory_dir=tmp_path)
        (fact,) = store.load_profile_records()
        assert fact.source == WRITEBACK_SOURCE

    def test_core_apply_writeback_gated_by_enforcement(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        outcome = core.apply_writeback(
            _wb(new_facts=[{"summary": "must not land"}]),
            enforcement_passed=False,
        )
        assert outcome.enforcement_gated is True
        assert MemoryStore(memory_dir=tmp_path).load_profile_records() == []

    def test_plain_think_writes_nothing_and_reports_nothing(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        core.think("hello there")
        assert core.last_writeback_outcome is None
        store = MemoryStore(memory_dir=tmp_path)
        assert store.load_profile_records() == []
        assert store.load_episodes() == [] and store.load_patterns() == []

    def test_freeze_additions_record_stands(self) -> None:
        from thoughtforge.cognition.core_docs import additions_problems

        assert additions_problems() == []
