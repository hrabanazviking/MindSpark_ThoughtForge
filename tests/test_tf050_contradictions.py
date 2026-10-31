"""TF-050 — contradiction handling: new fact vs stored fact.

Pins knowledge/contradictions.py against the live tree: the
three detection rules (fact_value slot, preference value,
negated claim), the scope guard (spec §27.3), duplicates as
reinforcement rather than conflict, the strength/score
arithmetic and the resolution rule at its margins
(supersede / stored stands / preserve-both-flagged), the
store application of each resolution, and the TF-047
writeback wiring that routes facts and preferences through
it and surfaces every conflict on the outcome.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.knowledge.contradictions import (
    CLOSE_CALL_CONFIDENCE_FACTOR,
    CONTRADICTION_RULES,
    RESOLUTION_RULE,
    RULE_STRENGTH,
    SOURCE_AUTHORITY,
    WIN_MARGIN,
    ConflictRecord,
    ContradictionReport,
    apply_contradiction_resolution,
    assess_contradiction,
    contradiction_problems,
    record_strength,
)
from thoughtforge.knowledge.ingestion_contract import MEMORY_PROVENANCE
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    UserFactRecord,
    UserPreferenceRecord,
    WritebackRecord,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.writeback import (
    WritebackOutcome,
    apply_writeback,
    writeback_problems,
)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path)


def fact(rid: str, summary: str, fv: dict | None = None, conf: float = 0.6,
         source: str = "", tags: list[str] | None = None,
         reinforced: int = 0) -> UserFactRecord:
    return UserFactRecord(
        record_id=rid, summary=summary, fact_value=fv or {},
        confidence=conf, source=source, tags=tags or [],
        times_reinforced=reinforced,
    )


def pref(rid: str, category: str, value: str, conf: float = 0.6,
         source: str = "", tags: list[str] | None = None) -> UserPreferenceRecord:
    return UserPreferenceRecord(
        record_id=rid, category=category, summary=f"{category}: {value}",
        value=value, confidence=conf, source=source, tags=tags or [],
    )


# ── Self-check and rule data ─────────────────────────────────────────────────


class TestSelfCheck:
    def test_contradiction_problems_empty(self) -> None:
        assert contradiction_problems() == []

    def test_writeback_problems_still_empty(self) -> None:
        assert writeback_problems() == []

    def test_rule_data_covers_stated_rules(self) -> None:
        names = [row["name"] for row in CONTRADICTION_RULES]
        assert names == ["fact_value_slot", "preference_value", "negated_claim"]
        assert set(RULE_STRENGTH) == set(names)
        assert set(SOURCE_AUTHORITY) == set(MEMORY_PROVENANCE)
        assert RESOLUTION_RULE["margin"] == "new_strength - stored_strength"

    def test_outcome_carries_contradictions_surface(self) -> None:
        assert WritebackOutcome(turn_id="t").contradictions == []


# ── Strength arithmetic ──────────────────────────────────────────────────────


class TestStrength:
    def test_strength_formula(self) -> None:
        new = fact("a", "x", conf=1.0, source="user_stated")
        assert record_strength(new, stored=False) == pytest.approx(1.0)
        old = fact("b", "x", conf=0.4, source="import")
        assert record_strength(old, stored=True) == pytest.approx(0.6 * 0.4 + 0.4 * 0.5)

    def test_reinforcement_bonus_capped(self) -> None:
        base = fact("b", "x", conf=0.5, source="user_stated", reinforced=0)
        many = fact("b", "x", conf=0.5, source="user_stated", reinforced=99)
        assert record_strength(many, stored=True) == pytest.approx(
            record_strength(base, stored=True) + 0.10
        )
        # The bonus is a stored-side property only.
        assert record_strength(many, stored=False) == record_strength(base, stored=False)

    def test_unstamped_authority_is_lowest(self) -> None:
        legacy = fact("b", "x", conf=0.5, source="")
        stamped = fact("b", "x", conf=0.5, source="synthetic_fixture")
        assert record_strength(legacy, stored=True) < record_strength(stamped, stored=True)


# ── Detection: facts ─────────────────────────────────────────────────────────


class TestFactDetection:
    def test_slot_contradiction_detected_and_scored(self) -> None:
        stored = fact("f_old", "the ferry leaves at dawn",
                      {"departure": "07:30"}, conf=0.4, source="import")
        new = fact("f_new", "the ferry now leaves later",
                   {"departure": "09:00"}, conf=1.0, source="user_stated")
        report = assess_contradiction(new, [stored])
        (conflict,) = report.conflicts
        assert conflict.rule == "fact_value_slot" and conflict.slot == "departure"
        assert conflict.new_value == "09:00" and conflict.stored_value == "07:30"
        assert conflict.new_strength == pytest.approx(1.0)
        assert conflict.stored_strength == pytest.approx(0.44)
        assert conflict.margin == pytest.approx(0.56)
        assert conflict.score == pytest.approx(0.5 * 1.0 + 0.25 * 1.0 + 0.25 * 0.44)
        assert conflict.resolution == "new_supersedes_stored"
        assert not report.blocked and report.duplicate_ids == []

    def test_slot_key_and_value_normalisation(self) -> None:
        stored = fact("f_old", "berths", {"Berth Count": 30})
        new = fact("f_new", "berths", {"berth_count": "30"})
        report = assess_contradiction(new, [stored])
        assert report.conflicts == []          # 30 == "30" numerically
        assert report.duplicate_ids == ["f_old"]

    def test_different_slots_no_conflict(self) -> None:
        stored = fact("f_old", "the ferry", {"departure": "07:30"})
        new = fact("f_new", "the ferry", {"arrival": "09:00"})
        report = assess_contradiction(new, [stored])
        assert report.clean

    def test_expanding_fact_is_neither_duplicate_nor_conflict(self) -> None:
        # Same summary, a slot the stored record lacks: the new
        # record expands the old one (spec §10.2) — it must land,
        # not be reinforced away with its new slot lost.
        stored = fact("f_old", "the ferry", {"departure": "07:30"})
        new = fact("f_new", "the ferry", {"departure": "07:30", "arrival": "09:00"})
        report = assess_contradiction(new, [stored])
        assert report.clean

    def test_scope_guard_disjoint_tags(self) -> None:
        stored = fact("f_old", "winter ferry", {"departure": "07:30"}, tags=["winter"])
        new = fact("f_new", "summer ferry", {"departure": "09:00"}, tags=["summer"])
        report = assess_contradiction(new, [stored])
        assert report.clean  # different scopes may hold different values

    def test_scope_guard_shared_tag_still_conflicts(self) -> None:
        stored = fact("f_old", "ferry", {"departure": "07:30"}, tags=["winter", "ferry"])
        new = fact("f_new", "ferry", {"departure": "09:00"}, tags=["ferry"])
        report = assess_contradiction(new, [stored])
        assert len(report.conflicts) == 1

    def test_identical_summary_is_duplicate_not_conflict(self) -> None:
        stored = fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"})
        new = fact("f_new", "The Ferry Leaves At Dawn", {"departure": "07:30"})
        report = assess_contradiction(new, [stored])
        assert report.conflicts == [] and report.duplicate_ids == ["f_old"]

    def test_contradicting_pair_is_never_also_duplicate(self) -> None:
        stored = fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"})
        new = fact("f_new", "the ferry leaves at dawn", {"departure": "09:00"})
        report = assess_contradiction(new, [stored])
        assert len(report.conflicts) == 1 and report.duplicate_ids == []

    def test_negated_claim_detected(self) -> None:
        stored = fact("f_pos", "the ferry runs on sundays every week")
        new = fact("f_neg", "the ferry does not run on sundays")
        report = assess_contradiction(new, [stored])
        (conflict,) = report.conflicts
        assert conflict.rule == "negated_claim" and conflict.slot == "summary"

    def test_both_negated_is_not_a_contradiction(self) -> None:
        stored = fact("f_a", "the ferry does not run on sundays in winter")
        new = fact("f_b", "the ferry never runs on sundays in winter")
        report = assess_contradiction(new, [stored])
        assert report.conflicts == []

    def test_negation_without_shared_subject_no_conflict(self) -> None:
        stored = fact("f_a", "the harbour office opens at nine each morning")
        new = fact("f_b", "the ferry does not run on sundays")
        report = assess_contradiction(new, [stored])
        assert report.conflicts == []

    def test_unrelated_facts_clean(self) -> None:
        stored = fact("f_a", "the harbourmaster keeps a grey cat")
        new = fact("f_b", "the ferry carries forty cars per crossing")
        assert assess_contradiction(new, [stored]).clean

    def test_skips_other_kinds_archived_and_self(self) -> None:
        stored_fact = fact("f_old", "the ferry", {"departure": "07:30"})
        stored_fact.status = "archived"
        new = fact("f_new", "the ferry", {"departure": "09:00"})
        others = [pref("p1", "tone", "concise"), stored_fact, new]
        assert assess_contradiction(new, others).clean

    def test_wrong_record_type_raises(self) -> None:
        with pytest.raises(TypeError):
            assess_contradiction(  # type: ignore[arg-type]
                EpisodicMemoryRecord(record_id="ep1", summary="a moment"), [],
            )


# ── Detection: preferences ───────────────────────────────────────────────────


class TestPreferenceDetection:
    def test_value_contradiction(self) -> None:
        stored = pref("p_old", "tone", "concise")
        new = pref("p_new", "tone", "detailed")
        report = assess_contradiction(new, [stored])
        (conflict,) = report.conflicts
        assert conflict.rule == "preference_value" and conflict.slot == "category"
        assert conflict.new_value == "detailed" and conflict.stored_value == "concise"

    def test_same_value_is_duplicate(self) -> None:
        stored = pref("p_old", "Tone", "Concise")
        new = pref("p_new", "tone", "concise")
        report = assess_contradiction(new, [stored])
        assert report.conflicts == [] and report.duplicate_ids == ["p_old"]

    def test_different_category_no_conflict(self) -> None:
        stored = pref("p_old", "tone", "concise")
        new = pref("p_new", "depth", "concise")
        assert assess_contradiction(new, [stored]).clean

    def test_scope_guard_applies_to_preferences(self) -> None:
        # Spec §27.3's own example: concise generally, detailed
        # for project docs — different scopes, no contradiction.
        stored = pref("p_old", "tone", "concise", tags=["general"])
        new = pref("p_new", "tone", "detailed", tags=["project_docs"])
        assert assess_contradiction(new, [stored]).clean


# ── The resolution rule at its margins ───────────────────────────────────────


class TestResolutionMargins:
    def _slot_report(self, new_conf: float, stored_conf: float,
                     source: str = "user_stated") -> ContradictionReport:
        stored = fact("f_old", "the ferry", {"departure": "07:30"},
                      conf=stored_conf, source=source)
        new = fact("f_new", "the ferry", {"departure": "09:00"},
                   conf=new_conf, source=source)
        return assess_contradiction(new, [stored])

    def test_margin_at_win_margin_new_supersedes(self) -> None:
        report = self._slot_report(0.85, 0.60)  # margin exactly +0.15
        assert report.conflicts[0].margin == pytest.approx(WIN_MARGIN)
        assert report.conflicts[0].resolution == "new_supersedes_stored"

    def test_margin_at_negative_win_margin_stored_stands(self) -> None:
        report = self._slot_report(0.60, 0.85)  # margin exactly -0.15
        assert report.conflicts[0].resolution == "stored_stands"
        assert report.blocked

    def test_close_margin_preserves_both(self) -> None:
        report = self._slot_report(0.70, 0.65)
        assert report.conflicts[0].resolution == "preserve_both_flagged"
        assert not report.blocked


# ── Applying resolutions to a store ──────────────────────────────────────────


class TestApply:
    def test_supersede_archives_stored_and_lands_new(self, store: MemoryStore) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"},
                 conf=0.4, source="import")
        )
        new = fact("f_new", "the ferry now leaves later", {"departure": "09:00"},
                   conf=1.0, source="user_stated")
        report = assess_contradiction(new, store.load_profile_records())
        assert apply_contradiction_resolution(store, new, report) == "f_new"
        live = store.load_profile_records()
        assert [r.record_id for r in live] == ["f_new"]  # old archived out

    def test_stored_stands_apply_raises_and_store_untouched(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"},
                 conf=0.9, source="user_stated")
        )
        new = fact("f_new", "the ferry leaves later", {"departure": "09:00"},
                   conf=0.3, source="conversation_writeback")
        report = assess_contradiction(new, store.load_profile_records())
        assert report.blocked
        with pytest.raises(ValueError, match="blocked"):
            apply_contradiction_resolution(store, new, report)
        (only,) = store.load_profile_records()
        assert only.record_id == "f_old" and only.confidence == 0.9

    def test_close_call_downgrades_both_and_keeps_both(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry", {"departure": "07:30"},
                 conf=0.6, source="conversation_writeback")
        )
        new = fact("f_new", "the ferry", {"departure": "09:00"},
                   conf=0.6, source="conversation_writeback")
        report = assess_contradiction(new, store.load_profile_records())
        apply_contradiction_resolution(store, new, report)
        live = {r.record_id: r for r in store.load_profile_records()}
        assert set(live) == {"f_old", "f_new"}
        for record in live.values():
            assert record.confidence == pytest.approx(0.6 * CLOSE_CALL_CONFIDENCE_FACTOR)

    def test_duplicate_reinforces_without_appending(self, store: MemoryStore) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"})
        )
        new = fact("f_new", "the ferry leaves at dawn", {"departure": "07:30"})
        report = assess_contradiction(new, store.load_profile_records())
        assert apply_contradiction_resolution(store, new, report) == "f_old"
        (only,) = store.load_profile_records()
        assert only.times_reinforced == 1 and only.status == "reinforced"

    def test_clean_report_is_a_plain_append(self, store: MemoryStore) -> None:
        new = fact("f_new", "a brand new claim about the harbour")
        report = assess_contradiction(new, [])
        assert report.clean
        assert apply_contradiction_resolution(store, new, report) == "f_new"
        assert [r.record_id for r in store.load_profile_records()] == ["f_new"]

    def test_report_to_dict_shape(self) -> None:
        report = ContradictionReport(
            new_record_id="n", kind="user_fact",
            conflicts=[ConflictRecord(
                new_record_id="n", stored_record_id="s", kind="user_fact",
                rule="fact_value_slot", slot="departure", new_value="09:00",
                stored_value="07:30", score=0.86, new_strength=1.0,
                stored_strength=0.44, margin=0.56,
                resolution="new_supersedes_stored",
            )],
        )
        data = report.to_dict()
        assert data["blocked"] is False
        assert data["conflicts"][0]["slot"] == "departure"


# ── Writeback wiring ─────────────────────────────────────────────────────────


def _wb(**kwargs) -> WritebackRecord:
    return WritebackRecord(turn_id="turn_t", **kwargs)


FULL = {
    "durability": 1.0, "novelty": 1.0, "confidence": 1.0,
    "importance": 1.0, "future_relevance": 1.0,
}


class TestWritebackWiring:
    def test_fact_blocked_by_standing_stored_fact(self, store: MemoryStore) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"},
                 conf=0.9, source="user_stated")
        )
        outcome = apply_writeback(
            store,
            _wb(new_facts=[{
                "summary": "the ferry leaves later now",
                "fact_value": {"departure": "09:00"},
                "confidence": 0.3, "source": "conversation_writeback",
            }]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert "contradiction" in outcome.refused[0]["reasons"][0]
        assert outcome.contradictions[0]["resolution"] == "stored_stands"
        assert outcome.contradictions[0]["stored_record_id"] == "f_old"
        (only,) = store.load_profile_records()
        assert only.record_id == "f_old" and only.confidence == 0.9

    def test_fact_superseding_via_writeback(self, store: MemoryStore) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"},
                 conf=0.4, source="import")
        )
        outcome = apply_writeback(
            store,
            _wb(new_facts=[{
                "summary": "the ferry now leaves later",
                "fact_value": {"departure": "09:00"},
                "confidence": 1.0, "source": "user_stated",
            }]),
            enforcement_passed=True,
        )
        assert len(outcome.applied) == 1
        assert outcome.contradictions[0]["resolution"] == "new_supersedes_stored"
        live = store.load_profile_records()
        assert len(live) == 1 and live[0].record_id == outcome.applied[0]["record_id"]
        assert live[0].fact_value == {"departure": "09:00"}

    def test_close_call_via_writeback_keeps_and_downgrades_both(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry", {"departure": "07:30"},
                 conf=0.6, source="conversation_writeback")
        )
        outcome = apply_writeback(
            store,
            _wb(new_facts=[{
                "summary": "the ferry", "fact_value": {"departure": "09:00"},
                "confidence": 0.6, "source": "conversation_writeback",
            }]),
            enforcement_passed=True,
        )
        assert len(outcome.applied) == 1
        assert outcome.contradictions[0]["resolution"] == "preserve_both_flagged"
        live = store.load_profile_records()
        assert len(live) == 2
        assert all(r.confidence == pytest.approx(0.48) for r in live)

    def test_duplicate_fact_via_writeback_reinforces(self, store: MemoryStore) -> None:
        entry = {"summary": "the ferry leaves at dawn",
                 "fact_value": {"departure": "07:30"}}
        first = apply_writeback(store, _wb(new_facts=[entry]), enforcement_passed=True)
        second = apply_writeback(store, _wb(new_facts=[entry]), enforcement_passed=True)
        assert second.applied[0]["record_id"] == first.applied[0]["record_id"]
        (only,) = store.load_profile_records()
        assert only.times_reinforced == 1

    def test_preference_blocked_by_standing_stored_preference(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            pref("p_old", "tone", "concise", conf=0.9, source="user_stated")
        )
        outcome = apply_writeback(
            store,
            _wb(new_preferences=[{
                "category": "tone", "summary": "detailed answers",
                "value": "detailed", "confidence": 0.3, "components": FULL,
            }]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert "contradiction" in outcome.refused[0]["reasons"][0]
        assert outcome.contradictions[0]["rule"] == "preference_value"
        (only,) = store.load_profile_records()
        assert only.record_id == "p_old"

    def test_enforcement_gate_fires_before_contradiction(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn", {"departure": "07:30"},
                 conf=0.4, source="import")
        )
        outcome = apply_writeback(
            store,
            _wb(new_facts=[{
                "summary": "the ferry now leaves later",
                "fact_value": {"departure": "09:00"},
                "confidence": 1.0, "source": "user_stated",
            }]),
            enforcement_passed=False,
        )
        assert outcome.enforcement_gated is True
        assert outcome.contradictions == []  # assessed against nothing
        (only,) = store.load_profile_records()
        assert only.record_id == "f_old"
