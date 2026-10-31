"""TF-045 — lifecycle: facts age, decay, and are retired by rule.

Pins knowledge/lifecycle_rules.py against the live lifecycle:
the prune score's floor/ceiling and the reachability table they
force, threshold decisions at their boundaries with engineered
records, dedup / cap / protected behaviour at the edges,
reinforcement arithmetic (incl. the pattern-crash regression),
thread cooling, and the D-ORPHAN-1 wiring into the core
(per-turn cooling + prune_memory). Synthetic records only.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.knowledge.lifecycle import (
    AGE_PARSE_DEFAULT_DAYS,
    MemoryLifecycle,
    _age_days,
)
from thoughtforge.knowledge.lifecycle_rules import (
    KIND_BY_RECORD_TYPE,
    MODES,
    PRUNE_SCORE_CEILING,
    PRUNE_SCORE_FLOOR,
    WIRING,
    age_for_prune_score,
    decision,
    lifecycle_problems,
    score_reachability,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore

NOW = datetime.now(timezone.utc)


def _ts(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _worst_fact(rid: str, days: float = 3650.0, **kw) -> UserFactRecord:
    # Distinct tag per record: facts share the profile dedup key
    # (category|tags) and identical keys would dedupe each other
    # before the score path ever sees them.
    base = dict(
        summary="worst", confidence=0.0, importance=0.0, tags=[rid],
        last_used_at=_ts(days), updated_at=_ts(days),
    )
    base.update(kw)
    return UserFactRecord(record_id=rid, **base)


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path)


@pytest.fixture()
def lifecycle(store: MemoryStore) -> MemoryLifecycle:
    return MemoryLifecycle(store=store)


# ── The rules as data ────────────────────────────────────────────────────────

class TestRulesData:
    def test_problems_empty(self) -> None:
        assert lifecycle_problems() == []

    def test_bounds_are_the_measured_pair(self) -> None:
        assert math.isclose(PRUNE_SCORE_FLOOR, 0.05, abs_tol=1e-9)
        assert math.isclose(PRUNE_SCORE_CEILING, 0.75, abs_tol=1e-9)

    def test_reachability_table_exact(self) -> None:
        reach = score_reachability()
        for kind in ("preference", "fact", "episode", "pattern"):
            assert reach["light"][kind] == {"archive": False, "delete": False}
            assert reach["routine"][kind] == {"archive": True, "delete": False}
            assert reach["emergency"][kind] == {"archive": True, "delete": True}
        # Heavy delete: reachable for fact/episode/pattern; the
        # preference cell sits exactly at the ceiling.
        assert reach["heavy"]["fact"]["delete"] is True
        assert reach["heavy"]["episode"]["delete"] is True
        assert reach["heavy"]["pattern"]["delete"] is True

    def test_modes_and_kinds(self) -> None:
        assert MODES == ("light", "routine", "heavy", "emergency")
        assert KIND_BY_RECORD_TYPE["user_fact"] == "fact"
        assert KIND_BY_RECORD_TYPE["episode"] == "episode"
        assert len(WIRING) == 3

    def test_decision_boundaries_inclusive(self) -> None:
        assert decision(0.86, "routine", "fact") == "deleted"
        assert decision(0.72, "routine", "fact") == "archived"
        assert decision(0.719999, "routine", "fact") == "kept"
        assert decision(0.75, "heavy", "preference") == "deleted"
        assert decision(0.05, "emergency", "episode") == "kept"


# ── Age edges ────────────────────────────────────────────────────────────────

class TestAgeEdges:
    def test_unparseable_ages_by_default(self) -> None:
        assert AGE_PARSE_DEFAULT_DAYS == 30.0
        assert _age_days("not a timestamp") == 30.0
        assert _age_days("") == 30.0

    def test_future_timestamp_ages_zero(self) -> None:
        assert _age_days(_ts(-5.0)) == 0.0

    def test_age_inverse_round_trip(self) -> None:
        age = age_for_prune_score(0.72, "user_fact")
        assert age is not None and age > 0
        assert age_for_prune_score(0.86, "user_fact") is None  # routine delete


# ── Score-path outcomes by mode (engineered records) ─────────────────────────

class TestScorePath:
    def test_light_never_scores_out_a_record(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        for i in range(3):
            store.append_profile_record(_worst_fact(f"f{i}"))
        counts = lifecycle.prune("light")
        assert counts == {"archived": 0, "deleted": 0, "deduplicated": 0}
        assert len(store.load_profile_records()) == 3

    def test_routine_archives_but_never_deletes_by_score(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        for i in range(3):
            store.append_profile_record(_worst_fact(f"f{i}"))
        counts = lifecycle.prune("routine")
        assert counts["archived"] == 3
        assert counts["deleted"] == 0
        # Retirement is physical: archived records leave the file.
        assert store.load_profile_records() == []

    def test_emergency_deletes_worst_episode(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_episode(EpisodicMemoryRecord(
            record_id="ep1", summary="worst", confidence=0.0,
            importance=0.0, quality=0.0,
            last_used_at=_ts(3650), updated_at=_ts(3650),
        ))
        counts = lifecycle.prune("emergency")
        assert counts["deleted"] == 1
        assert store.load_episodes() == []

    def test_engineered_boundary_fact(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        # Routine fact archive threshold is 0.72: a zero-signal
        # fact engineered to score 0.73 archives; one at 0.71 is kept.
        above = age_for_prune_score(0.73, "user_fact")
        below = age_for_prune_score(0.71, "user_fact")
        assert above is not None and below is not None
        store.append_profile_record(_worst_fact("above", days=above))
        store.append_profile_record(_worst_fact("below", days=below))
        counts = lifecycle.prune("routine")
        assert counts["archived"] == 1
        assert [r.record_id for r in store.load_profile_records()] == ["below"]

    def test_invalid_mode_refused_up_front(
        self, lifecycle: MemoryLifecycle
    ) -> None:
        with pytest.raises(ValueError, match="valid modes"):
            lifecycle.prune("bogus")  # type: ignore[arg-type]


# ── Dedup, caps, protection ──────────────────────────────────────────────────

class TestDedupCapsProtection:
    def test_dedup_keeps_higher_confidence(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_profile_record(UserPreferenceRecord(
            record_id="low", category="tone", tags=["x"], summary="a",
            confidence=0.3, importance=0.5,
        ))
        store.append_profile_record(UserPreferenceRecord(
            record_id="high", category="tone", tags=["x"], summary="b",
            confidence=0.9, importance=0.5,
        ))
        counts = lifecycle.prune("light")
        assert counts["deduplicated"] == 1
        assert [r.record_id for r in store.load_profile_records()] == ["high"]

    def test_cap_deletes_lowest_survival_unprotected(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_profile_record(UserPreferenceRecord(
            record_id="ancient", category="old", tags=["old"], summary="s",
            confidence=0.0, importance=0.0,
            last_used_at=_ts(3650), updated_at=_ts(3650),
        ))
        for i in range(256):
            store.append_profile_record(UserPreferenceRecord(
                record_id=f"p{i:03d}", category=f"c{i}", tags=[f"t{i}"],
                summary="s", confidence=0.9, importance=0.9,
            ))
        counts = lifecycle.prune("light")
        ids = [r.record_id for r in store.load_profile_records()]
        assert counts["deleted"] == 1
        assert "ancient" not in ids and len(ids) == 256

    def test_cap_never_deletes_protected_regression(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        # TF-045 probe: before the fix this protected record —
        # the lowest-survival record in an over-cap store — was
        # cap-deleted by an ordinary light prune.
        store.append_profile_record(UserPreferenceRecord(
            record_id="prot", category="protcat", tags=["prot"], summary="s",
            confidence=0.0, importance=0.0, retention_class="protected",
            last_used_at=_ts(3650), updated_at=_ts(3650),
        ))
        for i in range(256):
            store.append_profile_record(UserPreferenceRecord(
                record_id=f"p{i:03d}", category=f"c{i}", tags=[f"t{i}"],
                summary="s", confidence=0.9, importance=0.9,
            ))
        lifecycle.prune("light")
        ids = [r.record_id for r in store.load_profile_records()]
        assert "prot" in ids and len(ids) == 256

    def test_protected_survives_emergency_score_path(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            _worst_fact("prot", retention_class="protected")
        )
        counts = lifecycle.prune("emergency")
        assert counts == {"archived": 0, "deleted": 0, "deduplicated": 0}
        assert len(store.load_profile_records()) == 1

    def test_open_loops_trimmed_to_soft_cap(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.save_thread_state(ActiveThreadStateRecord(
            record_id="thr", topic="t",
            open_loops=[f"loop{i}" for i in range(10)],
        ))
        lifecycle.prune("light")
        thread = store.load_thread_state()
        assert thread is not None and len(thread.open_loops) == 6

    def test_tag_index_rebuilt_except_in_light(
        self, lifecycle: MemoryLifecycle, store: MemoryStore, tmp_path: Path
    ) -> None:
        store.append_profile_record(UserPreferenceRecord(
            record_id="p1", category="tone", tags=["direct"], summary="s",
        ))
        lifecycle.prune("light")
        assert not (tmp_path / "tag_index.json").exists()
        lifecycle.prune("routine")
        assert store.load_tag_index() == {"direct": ["p1"]}

    def test_empty_store_prunes_clean(self, lifecycle: MemoryLifecycle) -> None:
        for mode in MODES:
            assert lifecycle.prune(mode) == {
                "archived": 0, "deleted": 0, "deduplicated": 0,
            }


# ── Reinforcement ────────────────────────────────────────────────────────────

class TestReinforce:
    def test_profile_deltas_exact(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_profile_record(UserPreferenceRecord(
            record_id="p1", category="tone", tags=[], summary="s",
            confidence=0.5, importance=0.5,
        ))
        assert lifecycle.reinforce("p1", signal_strength=1.0) is True
        r = store.load_profile_records()[0]
        assert r.confidence == pytest.approx(0.55)
        assert r.importance == pytest.approx(0.53)
        assert r.times_reinforced == 1 and r.status == "reinforced"

    def test_signal_strength_scales_and_caps_at_one(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.append_episode(EpisodicMemoryRecord(
            record_id="ep1", summary="s", confidence=0.98,
            importance=0.99, quality=0.99,
        ))
        assert lifecycle.reinforce("ep1", signal_strength=5.0) is True
        ep = store.load_episodes()[0]
        assert (ep.confidence, ep.importance, ep.quality) == (1.0, 1.0, 1.0)

    def test_pattern_reinforce_regression(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        # Before TF-045 this raised AttributeError: the pattern
        # record had no times_reinforced field to increment.
        store.append_pattern(ResponsePatternRecord(
            record_id="pat1", summary="s", pattern_shape=["a"], quality=0.6,
        ))
        assert lifecycle.reinforce("pat1") is True
        p = store.load_patterns()[0]
        assert p.quality == pytest.approx(0.62)
        assert p.times_successful == 1 and p.times_reinforced == 1
        assert p.status == "reinforced"

    def test_missing_record_returns_false(
        self, lifecycle: MemoryLifecycle
    ) -> None:
        assert lifecycle.reinforce("nope") is False


# ── Thread cooling ───────────────────────────────────────────────────────────

class TestCoolThread:
    def test_cooling_arithmetic_measured(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.save_thread_state(ActiveThreadStateRecord(
            record_id="thr", topic="t", priority=0.9, expires_after_turns=10,
        ))
        lifecycle.cool_thread()
        t = store.load_thread_state()
        assert t is not None
        assert (round(t.priority, 2), t.turns_since_touch, t.expires_after_turns) == (0.82, 1, 9)
        lifecycle.cool_thread()
        t = store.load_thread_state()
        assert t is not None
        assert (round(t.priority, 2), t.turns_since_touch, t.expires_after_turns) == (0.66, 2, 7)

    def test_stale_thread_is_cleared(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        store.save_thread_state(ActiveThreadStateRecord(
            record_id="thr", topic="t", priority=0.25, expires_after_turns=1,
        ))
        lifecycle.cool_thread()
        assert store.load_thread_state() is None

    def test_no_thread_is_a_noop(
        self, lifecycle: MemoryLifecycle, store: MemoryStore
    ) -> None:
        lifecycle.cool_thread()
        assert store.load_thread_state() is None


# ── The wiring (D-ORPHAN-1) ──────────────────────────────────────────────────

class TestCoreWiring:
    @pytest.fixture()
    def core(self, tmp_path: Path) -> ThoughtForgeCore:
        return ThoughtForgeCore(
            config={}, memory_dir=tmp_path, db_path=tmp_path / "no.db",
        )

    def test_think_cools_the_thread(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        store.save_thread_state(ActiveThreadStateRecord(
            record_id="thr", topic="ferries", priority=0.9,
            expires_after_turns=10,
        ))
        core.think("hello there")
        t = store.load_thread_state()
        assert t is not None
        assert t.turns_since_touch == 1 and round(t.priority, 2) == 0.82

    def test_think_clears_a_thread_that_goes_stale(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        store.save_thread_state(ActiveThreadStateRecord(
            record_id="thr", topic="old", priority=0.25,
            expires_after_turns=1,
        ))
        core.think("hello there")
        assert store.load_thread_state() is None

    def test_prune_memory_surface(
        self, core: ThoughtForgeCore, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        store.append_profile_record(_worst_fact("ancient"))
        counts = core.prune_memory("routine")
        assert counts["archived"] == 1
        assert store.load_profile_records() == []
        with pytest.raises(ValueError, match="valid modes"):
            core.prune_memory("bogus")

    def test_freeze_additions_record_stands(self) -> None:
        from thoughtforge.cognition.core_docs import additions_problems

        assert additions_problems() == []

    def test_store_map_k10_stands(self) -> None:
        from thoughtforge.knowledge.store_map import store_map_problems

        assert store_map_problems() == []
