"""TF-046 — memory tiers: working / episodic / semantic.

Pins knowledge/memory_tiers.py against the live tree: the
verdicts (working separate, episodic separate, semantic unified
in the personal store and split across stores), the outside
rows (identity, procedural), the half-life ladder as the tier
ordering, cross-tier links, the unified surfaces, a synthetic
store census per tier, and the honest consumption pins — the
core reads working + identity per construction/turn and the
knowledge DB per retrieval, and no core code loads the personal
episodic/semantic/procedural tiers today.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.knowledge.memory_tiers import (
    CROSS_TIER_LINKS,
    HISTORY_WINDOW,
    OUTSIDE_TIERS,
    TIERS,
    TIER_BY_RECORD_TYPE,
    TIER_NAMES,
    TRANSIENT_WORKING,
    UNIFIED_SURFACES,
    WORLD_SEMANTIC_TABLES,
    classify_records,
    tier_census,
    tier_of_record,
    tier_of_record_type,
    tier_problems,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.scoring import HALF_LIFE_DAYS, compute_recency_score
from thoughtforge.knowledge.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path)


def _populated(store: MemoryStore) -> MemoryStore:
    store.save_personality_core(PersonalityCoreRecord(record_id="pers_t"))
    store.append_profile_record(
        UserFactRecord(record_id="fct_t", summary="the ferry leaves at dawn")
    )
    store.append_profile_record(
        UserPreferenceRecord(record_id="usr_t", category="tone", summary="plain speech")
    )
    store.append_episode(
        EpisodicMemoryRecord(
            record_id="ep_t", summary="planned the crossing",
            linked_facts=["fct_t"], linked_preferences=["usr_t"],
        )
    )
    store.append_pattern(
        ResponsePatternRecord(record_id="pat_t", summary="answer, then evidence")
    )
    store.save_thread_state(ActiveThreadStateRecord(record_id="thr_t", topic="ferry"))
    return store


# ── The verdicts as data ─────────────────────────────────────────────────────

class TestTierData:
    def test_problems_empty(self) -> None:
        assert tier_problems() == []

    def test_three_named_tiers_in_order(self) -> None:
        assert TIER_NAMES == ("working", "episodic", "semantic")
        assert [row["tier"] for row in TIERS] == list(TIER_NAMES)

    def test_verdicts(self) -> None:
        verdicts = {row["tier"]: row["verdict"] for row in TIERS}
        assert verdicts["working"] == "separate"
        assert verdicts["episodic"] == "separate"
        assert verdicts["semantic"] == "unified-personal / split-world"

    def test_every_persisted_type_mapped(self) -> None:
        assert set(TIER_BY_RECORD_TYPE) == {
            "active_thread_state", "episode", "user_fact",
            "user_preference", "personality_core", "response_pattern",
        }

    def test_outside_rows(self) -> None:
        outside = {row["tier"] for row in OUTSIDE_TIERS}
        assert outside == {"identity", "procedural"}
        assert TIER_BY_RECORD_TYPE["personality_core"] == "identity"
        assert TIER_BY_RECORD_TYPE["response_pattern"] == "procedural"

    def test_tier_of_record_type(self) -> None:
        assert tier_of_record_type("active_thread_state") == "working"
        assert tier_of_record_type("episode") == "episodic"
        assert tier_of_record_type("user_fact") == "semantic"
        assert tier_of_record_type("user_preference") == "semantic"
        assert tier_of_record_type("nope") is None

    def test_tier_of_record_instances(self) -> None:
        assert tier_of_record(ActiveThreadStateRecord(record_id="t")) == "working"
        assert tier_of_record(EpisodicMemoryRecord(record_id="e")) == "episodic"
        assert tier_of_record(UserFactRecord(record_id="f")) == "semantic"
        assert tier_of_record(UserPreferenceRecord(record_id="p", category="c")) == "semantic"
        assert tier_of_record(ResponsePatternRecord(record_id="r")) == "procedural"
        # Personality carries no record_type — read by class.
        assert tier_of_record(PersonalityCoreRecord(record_id="i")) == "identity"
        assert tier_of_record(object()) is None

    def test_unified_and_links_and_world_rows(self) -> None:
        assert len(UNIFIED_SURFACES) == 7
        link = CROSS_TIER_LINKS[0]
        assert (link["from"], link["to"]) == ("episodic", "semantic")
        assert set(link["fields"]) == {"linked_preferences", "linked_facts"}
        assert "entities" in WORLD_SEMANTIC_TABLES
        assert "reference_chunks" in WORLD_SEMANTIC_TABLES

    def test_transient_working_surfaces(self) -> None:
        assert "RuntimeTurnState" in TRANSIENT_WORKING
        assert HISTORY_WINDOW == {
            "surface": "ChatHistory (caller-owned, cognition/chat_history.py)",
            "max_turns": 20, "max_tokens": 1000, "max_chars": 2000,
            "owner": "TF-024 budgets, enforced on append",
        }


# ── The ladder is the tier ordering ──────────────────────────────────────────

class TestLadder:
    def test_half_life_order(self) -> None:
        order = [
            HALF_LIFE_DAYS[k] for k in (
                "active_thread_state", "episode", "response_pattern",
                "user_preference", "user_fact", "personality_core",
            )
        ]
        assert order == [0.07, 7.0, 30.0, 90.0, 180.0, 9999.0]

    def test_recency_at_one_day_by_tier(self) -> None:
        # After one day the working tier has essentially decayed,
        # the semantic fact has essentially not.
        assert compute_recency_score(1.0, "active_thread_state") < 0.001
        assert compute_recency_score(1.0, "episode") < 0.95
        assert compute_recency_score(1.0, "user_fact") > 0.99


# ── Behaviour on a synthetic store ───────────────────────────────────────────

class TestStoreCensus:
    def test_empty_store_census(self, store: MemoryStore) -> None:
        assert tier_census(store) == {
            "working": 0, "identity": 0, "episodic": 0,
            "procedural": 0, "semantic": 0,
        }

    def test_populated_store_census(self, store: MemoryStore) -> None:
        _populated(store)
        assert tier_census(store) == {
            "working": 1, "identity": 1, "episodic": 1,
            "procedural": 1, "semantic": 2,
        }

    def test_classify_records(self, store: MemoryStore) -> None:
        _populated(store)
        all_records = store.load_all()
        flat = (
            [all_records["personality_core"], all_records["thread_state"]]
            + all_records["profile_records"]
            + all_records["episodes"]
            + all_records["patterns"]
        )
        assert classify_records(flat) == {
            "identity": 1, "working": 1, "semantic": 2,
            "episodic": 1, "procedural": 1,
        }
        assert classify_records([object()]) == {"unknown": 1}

    def test_semantic_tier_is_one_file(self, store: MemoryStore) -> None:
        # The unified verdict, behaviourally: a fact and a
        # preference land interleaved in the SAME file and come
        # back from ONE loader, in append order.
        _populated(store)
        lines = (store.dir / "user_profile_store.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        assert len(lines) == 2
        loaded = store.load_profile_records()
        assert [r.record_type for r in loaded] == ["user_fact", "user_preference"]

    def test_separate_tiers_separate_files(self, store: MemoryStore) -> None:
        _populated(store)
        assert (store.dir / "episodic_store.jsonl").exists()
        assert (store.dir / "active_thread_state.json").exists()
        episode_lines = (store.dir / "episodic_store.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        assert len(episode_lines) == 1
        assert '"episode"' in episode_lines[0]

    def test_episode_links_point_across_tiers(self, store: MemoryStore) -> None:
        _populated(store)
        episode = store.load_episodes()[0]
        assert episode.linked_facts == ["fct_t"]
        assert episode.linked_preferences == ["usr_t"]
        semantic_ids = {r.record_id for r in store.load_profile_records()}
        assert set(episode.linked_facts + episode.linked_preferences) <= semantic_ids

    def test_working_tier_expires_fastest(self, store: MemoryStore) -> None:
        # The working record cools to stale within its 10-turn
        # expiry (TF-045's cool path) while the other tiers'
        # records are untouched by cooling.
        _populated(store)
        thread = store.load_thread_state()
        assert thread is not None
        for _ in range(10):
            thread.cool_down()
            if thread.status == "stale":
                break
        assert thread.status == "stale"
        assert store.load_episodes()[0].status == "created"
        assert store.load_profile_records()[0].status == "created"

    def test_tag_index_unifies_across_tiers(self, store: MemoryStore) -> None:
        store.append_profile_record(
            UserFactRecord(record_id="fct_x", summary="s", tags=["ferry"])
        )
        store.append_episode(
            EpisodicMemoryRecord(record_id="ep_x", summary="s", tags=["ferry"])
        )
        store.append_pattern(
            ResponsePatternRecord(record_id="pat_x", summary="s", tags=["ferry"])
        )
        index = store.rebuild_tag_index()
        assert set(index["ferry"]) == {"fct_x", "ep_x", "pat_x"}


# ── Consumption: who actually reads which tier ───────────────────────────────

class TestConsumption:
    def test_core_reads_working_and_identity(self) -> None:
        import inspect

        from thoughtforge.cognition import core as core_module

        assert "load_thread_state" in inspect.getsource(
            core_module.ThoughtForgeCore._run_turn
        )
        assert "load_personality_core" in inspect.getsource(core_module)

    def test_core_retrieval_reads_world_semantic_only(self) -> None:
        import inspect

        from thoughtforge.cognition import core as core_module

        retrieve_src = inspect.getsource(core_module.ThoughtForgeCore._retrieve)
        assert "self._knowledge.retrieve" in retrieve_src
        assert "_store" not in retrieve_src

    def test_personal_tiers_have_no_core_loader(self) -> None:
        # The pinned gap: until a wiring slice flips this with
        # evidence, episodes / patterns / profile records are
        # storage + lifecycle only.
        import inspect

        from thoughtforge.cognition import core as core_module

        core_src = inspect.getsource(core_module)
        for loader in ("load_episodes", "load_patterns", "load_profile_records"):
            assert loader not in core_src
