"""TF-056 — memory budgets: store size + retrieval cost capped
per hardware profile; eviction by the lifecycle rules.

Pins knowledge/memory_budgets.py against the live tree: budgets
resolved live from the shipped profile JSONs, the flat default
proven to be the desktop budget, plans that only cap, store
size measured as counts-per-kind + bytes-per-file, eviction of
an over-budget store to exactly its profile's hard caps by a
light lifecycle pass (lowest survival first, protected records
never cap-evicted), and the wiring — lifecycle profile caps,
the core's profile lifecycle + capped retrieval, the forge's
optional sql/vector caps (None = pre-TF-056 behaviour).
Synthetic records only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.router import InputSketch
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.lifecycle import STORE_CAPS, MemoryLifecycle
from thoughtforge.knowledge.memory_budgets import (
    BUDGETED_KINDS,
    DEFAULT_REQUESTED_TOP_K,
    all_budgets,
    budget_for_profile,
    budget_report,
    enforce_budget,
    measure_store,
    memory_budgets_problems,
    retrieval_plan,
)
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.retrieval_precision import build_fixture_db
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

NOW = datetime.now(timezone.utc)


def _ts(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _fact(rid: str, **kw) -> UserFactRecord:
    base = dict(summary=f"fact {rid}", tags=[rid])
    base.update(kw)
    return UserFactRecord(record_id=rid, **base)


def _store_with_facts(tmp_path: Path, n: int, **kw) -> MemoryStore:
    store = MemoryStore(memory_dir=tmp_path)
    for i in range(n):
        store.append_profile_record(_fact(f"f{i:03d}", **kw))
    return store


def _core(tmp_path: Path, profile_id: str | None = None) -> ThoughtForgeCore:
    config: dict = {"cognition": {}}
    if profile_id is not None:
        config["conversation"] = {"profile_id": profile_id}
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "kb.db"
    db.touch()
    return ThoughtForgeCore(
        config=config, memory_dir=tmp_path / "memory", db_path=db
    )


# ── Budgets as data, read live ─────────────────────────────────────────────────


class TestBudgets:
    def test_problems_empty_on_live_tree(self) -> None:
        assert memory_budgets_problems() == []

    def test_budgets_cover_exactly_the_shipped_profiles(self) -> None:
        assert sorted(all_budgets()) == sorted(list_hardware_profiles())

    def test_caps_read_live_from_the_jsons(self) -> None:
        for pid in list_hardware_profiles():
            raw = load_hardware_profile(pid)["memory"]
            budget = budget_for_profile(pid)
            assert budget.cap("user_fact", "hard") == raw["user_facts_cap_hard"]
            assert budget.cap("episode", "soft") == raw["episodic_store_cap_soft"]
            assert budget.max_memory_bundle_facts == raw["max_memory_bundle_facts"]

    def test_spot_values_across_the_ladder(self) -> None:
        assert budget_for_profile("pi_zero").cap("user_fact") == 32
        assert budget_for_profile("phone_low").cap("user_fact") == 64
        assert budget_for_profile("pi_5").cap("episode") == 256
        assert budget_for_profile("server_gpu").cap("episode") == 1024
        assert budget_for_profile("phone_low").sql_max_results == 5
        assert budget_for_profile("phone_low").vector_top_k == 3

    def test_unknown_profile_raises_naming_valid(self) -> None:
        with pytest.raises(ValueError, match="phone_low"):
            budget_for_profile("quantum_phone")

    def test_flat_default_is_the_desktop_budget(self) -> None:
        for pid in ("desktop_cpu", "desktop_gpu"):
            budget = budget_for_profile(pid)
            for kind in BUDGETED_KINDS:
                assert budget.caps[kind] == STORE_CAPS[kind]

    def test_ladder_is_monotonic_per_kind(self) -> None:
        ladder = ["pi_zero", "phone_low", "pi_5", "desktop_cpu", "server_gpu"]
        for kind in BUDGETED_KINDS:
            hards = [budget_for_profile(p).cap(kind) for p in ladder]
            assert hards == sorted(hards)


# ── Retrieval plans only cap ───────────────────────────────────────────────────


class TestRetrievalPlan:
    def test_bundle_capped_per_profile(self) -> None:
        assert retrieval_plan("phone_low").bundle_top_k == 5
        assert retrieval_plan("pi_zero").bundle_top_k == 3
        assert retrieval_plan("pi_5").bundle_top_k == 10

    def test_plan_never_raises_a_request(self) -> None:
        # desktop's bundle cap is 15 and server's 30 — a standing
        # request of 10 passes through untouched on both.
        assert retrieval_plan("desktop_cpu").bundle_top_k == DEFAULT_REQUESTED_TOP_K
        assert retrieval_plan("server_gpu").bundle_top_k == DEFAULT_REQUESTED_TOP_K
        # …and a larger request under a larger cap passes too.
        assert retrieval_plan("server_gpu", requested_top_k=25).bundle_top_k == 25

    def test_plan_carries_the_retrieval_caps(self) -> None:
        plan = retrieval_plan("phone_low")
        assert plan.sql_max_results == 5
        assert plan.vector_top_k == 3
        assert plan.retrieval_timeout_ms == 500

    def test_zero_request_refused(self) -> None:
        with pytest.raises(ValueError):
            retrieval_plan("phone_low", requested_top_k=0)


# ── Store size + budget reports ────────────────────────────────────────────────


class TestStoreSize:
    def test_measure_counts_per_kind_and_bytes_per_file(
        self, tmp_path: Path
    ) -> None:
        store = _store_with_facts(tmp_path, 3)
        store.append_episode(EpisodicMemoryRecord(record_id="ep1", summary="e"))
        size = measure_store(store)
        assert size.counts["user_fact"] == 3
        assert size.counts["episode"] == 1
        assert size.total_records == 4
        assert size.file_bytes["user_profile_store.jsonl"] > 0
        assert size.total_bytes == sum(size.file_bytes.values())

    def test_report_statuses(self, tmp_path: Path) -> None:
        store = _store_with_facts(tmp_path / "a", 10)
        report = budget_report(store, "phone_low")
        assert report.rows["user_fact"]["status"] == "within_soft"
        assert report.within_budget

        store = _store_with_facts(tmp_path / "b", 40)
        report = budget_report(store, "phone_low")
        assert report.rows["user_fact"]["status"] == "over_soft"
        assert report.within_budget  # soft is a report line, not eviction

        store = _store_with_facts(tmp_path / "c", 70)
        report = budget_report(store, "phone_low")
        assert report.rows["user_fact"]["status"] == "over_hard"
        assert report.over_hard_kinds == ["user_fact"]
        assert not report.within_budget


# ── Eviction by the lifecycle rules ────────────────────────────────────────────


class TestEviction:
    def test_phone_budget_evicts_facts_to_exactly_its_hard_cap(
        self, tmp_path: Path
    ) -> None:
        store = _store_with_facts(tmp_path, 70)
        counts, after = enforce_budget(store, "phone_low", mode="light")
        assert len(store.load_profile_records()) == 64
        assert counts["deleted"] == 6
        assert after.within_budget

    def test_default_lifecycle_keeps_the_same_store(
        self, tmp_path: Path
    ) -> None:
        # The measured starting state: the flat (desktop) cap
        # leaves a phone-over-budget store entirely alone.
        store = _store_with_facts(tmp_path, 70)
        MemoryLifecycle(store=store).prune("light")
        assert len(store.load_profile_records()) == 70

    def test_lowest_survival_is_evicted_first(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        for i in range(64):
            store.append_profile_record(_fact(
                f"keep{i:02d}", confidence=1.0, importance=1.0,
                times_reinforced=10,
            ))
        for i in range(6):
            store.append_profile_record(_fact(
                f"drop{i}", confidence=0.0, importance=0.0,
                last_used_at=_ts(3650), updated_at=_ts(3650),
            ))
        enforce_budget(store, "phone_low", mode="light")
        remaining = {r.record_id for r in store.load_profile_records()}
        assert len(remaining) == 64
        assert not any(rid.startswith("drop") for rid in remaining)

    def test_protected_surplus_stays_and_the_report_says_so(
        self, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        for i in range(68):
            store.append_profile_record(_fact(
                f"prot{i:02d}", retention_class="protected",
            ))
        for i in range(2):
            store.append_profile_record(_fact(f"plain{i}"))
        _, after = enforce_budget(store, "phone_low", mode="light")
        remaining = store.load_profile_records()
        assert len(remaining) == 68  # only the 2 unprotected could go
        assert all(r.retention_class == "protected" for r in remaining)
        assert after.rows["user_fact"]["status"] == "over_hard"
        assert not after.within_budget

    def test_episode_and_pattern_caps_follow_the_profile(
        self, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        for i in range(70):
            store.append_episode(EpisodicMemoryRecord(
                record_id=f"ep{i:03d}", summary=f"episode {i}", tags=[f"e{i}"],
            ))
            store.append_pattern(ResponsePatternRecord(
                record_id=f"pat{i:03d}", summary=f"pattern {i}", tags=[f"p{i}"],
            ))
        enforce_budget(store, "pi_zero", mode="light")
        assert len(store.load_episodes()) == 64
        assert len(store.load_patterns()) == 32

    def test_lifecycle_profile_caps_and_unknown_profile(
        self, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path)
        lifecycle = MemoryLifecycle(store=store, profile_id="phone_low")
        assert lifecycle.caps["user_fact"] == {"soft": 32, "hard": 64}
        default = MemoryLifecycle(store=store)
        assert default.caps == STORE_CAPS
        with pytest.raises(ValueError):
            MemoryLifecycle(store=store, profile_id="bogus")


# ── The forge's caps ───────────────────────────────────────────────────────────


class TestForgeCaps:
    @pytest.fixture()
    def forge(self, tmp_path: Path) -> KnowledgeForge:
        return KnowledgeForge(db_path=build_fixture_db(tmp_path / "fx.db"))

    def test_sql_leg_capped(self, forge: KnowledgeForge) -> None:
        seen: dict[str, int] = {}
        original = forge.sql_retrieve

        def spy(query, top_k=15, sources=None):
            seen["top_k"] = top_k
            return original(query, top_k=top_k, sources=sources)

        forge.sql_retrieve = spy  # type: ignore[method-assign]
        bundle = forge.retrieve(
            "Lantern Hollow ferry", path="sql", top_k=10, sql_max_results=5
        )
        assert seen["top_k"] == 5  # not the standing 2 × top_k = 20
        assert bundle.total_retrieved <= 10

    def test_no_caps_is_the_standing_scaling(self, forge: KnowledgeForge) -> None:
        seen: dict[str, int] = {}
        original = forge.sql_retrieve

        def spy(query, top_k=15, sources=None):
            seen["top_k"] = top_k
            return original(query, top_k=top_k, sources=sources)

        forge.sql_retrieve = spy  # type: ignore[method-assign]
        forge.retrieve("Lantern Hollow ferry", path="sql", top_k=10)
        assert seen["top_k"] == 20


# ── Core wiring ────────────────────────────────────────────────────────────────


class TestCoreWiring:
    def test_memory_budget_property(self, tmp_path: Path) -> None:
        assert _core(tmp_path / "a").memory_budget is None
        budget = _core(tmp_path / "b", "phone_low").memory_budget
        assert budget is not None and budget.profile_id == "phone_low"
        assert budget.cap("user_fact") == 64

    def test_core_lifecycle_enforces_the_selected_profile(
        self, tmp_path: Path
    ) -> None:
        core = _core(tmp_path, "phone_low")
        store = MemoryStore(memory_dir=tmp_path / "memory")
        for i in range(70):
            store.append_profile_record(_fact(f"f{i:03d}"))
        core.prune_memory("light")
        assert len(store.load_profile_records()) == 64

    def test_core_without_profile_keeps_the_flat_caps(
        self, tmp_path: Path
    ) -> None:
        core = _core(tmp_path)
        store = MemoryStore(memory_dir=tmp_path / "memory")
        for i in range(70):
            store.append_profile_record(_fact(f"f{i:03d}"))
        core.prune_memory("light")
        assert len(store.load_profile_records()) == 70

    def test_core_retrieval_is_capped_by_the_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        core = _core(tmp_path, "phone_low")
        seen: dict = {}

        def fake_retrieve(**kwargs):
            seen.update(kwargs)
            from thoughtforge.knowledge.models import MemoryActivationBundle

            return MemoryActivationBundle()

        monkeypatch.setattr(core._knowledge, "retrieve", fake_retrieve)
        sketch = InputSketch(raw_text="anything", retrieval_path="sql")
        core._retrieve(sketch)
        assert seen["top_k"] == 5
        assert seen["sql_max_results"] == 5
        assert seen["vector_top_k"] == 3

    def test_core_retrieval_without_profile_is_the_standing_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        core = _core(tmp_path)
        seen: dict = {}

        def fake_retrieve(**kwargs):
            seen.update(kwargs)
            from thoughtforge.knowledge.models import MemoryActivationBundle

            return MemoryActivationBundle()

        monkeypatch.setattr(core._knowledge, "retrieve", fake_retrieve)
        sketch = InputSketch(raw_text="anything", retrieval_path="sql")
        core._retrieve(sketch)
        assert seen == {
            "query": "anything", "path": "sql", "top_k": 10,
        }

    def test_freeze_additions_record_stands(self) -> None:
        from thoughtforge.cognition.core_docs import additions_problems

        assert additions_problems() == []
