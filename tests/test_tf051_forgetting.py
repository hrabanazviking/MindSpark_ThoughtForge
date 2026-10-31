"""TF-051 — forgetting: user-ordered deletion, total + provable.

Pins knowledge/forgetting.py against the live tree: the
forgettable surfaces as data, forget_record across every
home store, the derived-surface cleanup (tag index rebuild,
recent-queue purge, episode-link scrub), retention vs user
order, the personality-core refusal, residue-only cleanup,
the verify_forgotten proof (including its failure modes),
and the TF-047 writeback delete leg routed through here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.knowledge.forgetting import (
    EXCLUDED_SURFACES,
    FORGET_STATUSES,
    FORGET_SURFACES,
    PROOF_CHECKS,
    ForgetProof,
    ForgetReport,
    forget_record,
    forgetting_problems,
    verify_forgotten,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
    WritebackRecord,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.knowledge.writeback import apply_writeback, writeback_problems


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path)


def seed_fact_with_residues(store: MemoryStore) -> tuple[str, str]:
    """A fact, an episode linking to it, a tag index and a
    recent queue that both name it — the TF-051 probe shape."""
    store.append_profile_record(
        UserFactRecord(
            record_id="fct_target", tags=["ferry"],
            summary="ferry leaves at 07:30",
            fact_value={"departure": "07:30"}, source="user_stated",
        )
    )
    store.append_episode(
        EpisodicMemoryRecord(
            record_id="ep_link", tags=["ferry"],
            summary="talked about the ferry",
            linked_facts=["fct_target"], source="user_stated",
        )
    )
    store.rebuild_tag_index()
    store.push_to_recent_queue("fct_target")
    return "fct_target", "ep_link"


# ── Self-check and surface data ──────────────────────────────────────────────


class TestSelfCheck:
    def test_forgetting_problems_empty(self) -> None:
        assert forgetting_problems() == []

    def test_writeback_problems_still_empty(self) -> None:
        assert writeback_problems() == []

    def test_surfaces_cover_store_indices_caches(self) -> None:
        names = [row["name"] for row in FORGET_SURFACES]
        assert names == [
            "profile_store", "episodic_store", "patterns_store",
            "thread_state", "tag_index", "recent_queue", "episode_links",
        ]
        files = {row["file"] for row in FORGET_SURFACES}
        assert "user_profile_store.jsonl" in files
        assert "tag_index.json" in files
        assert "recent_queue.json" in files

    def test_excluded_surfaces_named_with_reasons(self) -> None:
        names = {row["name"] for row in EXCLUDED_SURFACES}
        assert names == {"personality_core", "knowledge_db"}
        assert all(row["reason"] for row in EXCLUDED_SURFACES)

    def test_proof_checks_and_statuses_as_data(self) -> None:
        assert "raw_files_free_of_id" in PROOF_CHECKS
        assert "absent_from_tag_index" in PROOF_CHECKS
        assert "absent_from_recent_queue" in PROOF_CHECKS
        assert set(FORGET_STATUSES) == {
            "forgotten", "not_found", "refused_protected", "not_proven",
        }


# ── Total deletion from each home ────────────────────────────────────────────


class TestForgetHomes:
    def test_fact_forgetting_is_total(self, store: MemoryStore) -> None:
        fact_id, ep_id = seed_fact_with_residues(store)
        report = forget_record(store, fact_id)
        assert isinstance(report, ForgetReport)
        assert report.status == "forgotten" and report.forgotten
        assert report.home == "profile_store"
        assert set(report.removed_from) == {
            "profile_store", "episode_links", "recent_queue", "tag_index",
        }
        assert report.links_scrubbed == 1
        # The record is gone; the linking episode survives, unlinked.
        assert store.load_profile_records() == []
        (episode,) = store.load_episodes()
        assert episode.record_id == ep_id and episode.linked_facts == []
        # Indices and caches carry no trace.
        assert all(fact_id not in ids for ids in store.load_tag_index().values())
        assert fact_id not in store.load_recent_queue()
        # The proof passed, on a raw scan of real files.
        assert report.proof is not None and report.proof.forgotten
        assert "user_profile_store.jsonl" in report.proof.files_scanned

    def test_preference_forgotten(self, store: MemoryStore) -> None:
        store.append_profile_record(
            UserPreferenceRecord(
                record_id="usr_target", category="tone",
                summary="tone: terse", value="terse", source="user_stated",
            )
        )
        report = forget_record(store, "usr_target")
        assert report.forgotten and report.home == "profile_store"
        assert store.load_profile_records() == []

    def test_episode_forgotten_from_its_home(self, store: MemoryStore) -> None:
        store.append_episode(
            EpisodicMemoryRecord(record_id="ep_target", summary="a moment")
        )
        report = forget_record(store, "ep_target")
        assert report.forgotten and report.home == "episodic_store"
        assert store.load_episodes() == []

    def test_pattern_forgotten(self, store: MemoryStore) -> None:
        store.append_pattern(
            ResponsePatternRecord(record_id="pat_target", summary="a shape")
        )
        report = forget_record(store, "pat_target")
        assert report.forgotten and report.home == "patterns_store"
        assert store.load_patterns() == []

    def test_thread_state_forgotten_by_direct_api(self, store: MemoryStore) -> None:
        store.save_thread_state(
            ActiveThreadStateRecord(record_id="thr_target", topic="ferries")
        )
        report = forget_record(store, "thr_target")
        assert report.forgotten and report.home == "thread_state"
        assert store.load_thread_state() is None

    def test_thread_out_of_reach_when_excluded(self, store: MemoryStore) -> None:
        store.save_thread_state(
            ActiveThreadStateRecord(record_id="thr_target", topic="ferries")
        )
        report = forget_record(store, "thr_target", include_thread=False)
        assert report.status == "not_found"
        assert store.load_thread_state() is not None

    def test_protected_retention_yields_to_user_order(
        self, store: MemoryStore
    ) -> None:
        store.append_profile_record(
            UserFactRecord(
                record_id="fct_protected", summary="guarded by prune rules",
                retention_class="protected",
            )
        )
        report = forget_record(store, "fct_protected")
        assert report.forgotten
        assert store.load_profile_records() == []

    def test_other_records_untouched(self, store: MemoryStore) -> None:
        seed_fact_with_residues(store)
        store.append_profile_record(
            UserFactRecord(record_id="fct_keep", summary="unrelated", tags=["other"])
        )
        forget_record(store, "fct_target")
        (kept,) = store.load_profile_records()
        assert kept.record_id == "fct_keep"
        assert store.load_tag_index().get("other") == ["fct_keep"]


# ── Refusals, absence, residues ──────────────────────────────────────────────


class TestOutcomes:
    def test_personality_core_refused_by_name(self, store: MemoryStore) -> None:
        store.save_personality_core(PersonalityCoreRecord(record_id="pers_seed"))
        report = forget_record(store, "pers_seed")
        assert report.status == "refused_protected"
        assert "identity" in report.reason
        assert not report.forgotten
        assert store.load_personality_core() is not None
        # The proof honestly shows the record still present.
        assert report.proof is not None
        assert report.proof.checks["absent_from_personality_core"] is False

    def test_unknown_id_is_not_found_not_silent(self, store: MemoryStore) -> None:
        report = forget_record(store, "nope")
        assert report.status == "not_found"
        assert "not found" in report.reason
        # Absence is proven too: the id is nowhere.
        assert report.proof is not None and report.proof.forgotten

    def test_residue_only_cleanup_is_forgetting(self, store: MemoryStore) -> None:
        # No record anywhere — but the cache and index hold the id.
        store.push_to_recent_queue("ghost_id")
        store.save_tag_index({"stale": ["ghost_id"]})
        report = forget_record(store, "ghost_id")
        assert report.status == "forgotten" and report.home is None
        assert set(report.removed_from) == {"recent_queue", "tag_index"}
        assert store.load_recent_queue() == []
        assert store.load_tag_index() == {}

    def test_second_forget_is_not_found(self, store: MemoryStore) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        assert forget_record(store, fact_id).forgotten
        assert forget_record(store, fact_id).status == "not_found"

    def test_report_dict_shape(self, store: MemoryStore) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        data = forget_record(store, fact_id).to_dict()
        assert data["status"] == "forgotten" and data["forgotten"] is True
        assert data["proof"]["forgotten"] is True


# ── The proof itself ─────────────────────────────────────────────────────────


class TestProof:
    def test_proof_fails_before_deletion(self, store: MemoryStore) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        proof = verify_forgotten(store, fact_id)
        assert isinstance(proof, ForgetProof)
        assert not proof.forgotten
        assert proof.checks["absent_from_profile_store"] is False
        assert proof.checks["absent_from_tag_index"] is False
        assert proof.checks["absent_from_recent_queue"] is False
        assert proof.checks["no_dangling_episode_links"] is False
        assert proof.checks["raw_files_free_of_id"] is False
        # Surfaces the fact never touched still pass their checks.
        assert proof.checks["absent_from_patterns_store"] is True

    def test_proof_catches_reinserted_residue(self, store: MemoryStore) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        assert forget_record(store, fact_id).forgotten
        store.push_to_recent_queue(fact_id)  # a residue reappears
        proof = verify_forgotten(store, fact_id)
        assert not proof.forgotten
        assert proof.failed_checks == [
            "absent_from_recent_queue", "raw_files_free_of_id",
        ]

    def test_proof_scans_every_file(self, store: MemoryStore, tmp_path: Path) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        forget_record(store, fact_id)
        proof = verify_forgotten(store, fact_id)
        assert proof.forgotten
        assert set(proof.files_scanned) == {
            p.name for p in tmp_path.iterdir() if p.is_file()
        }


# ── Writeback wiring ─────────────────────────────────────────────────────────


class TestWritebackWiring:
    def test_writeback_delete_is_total(self, store: MemoryStore) -> None:
        fact_id, _ = seed_fact_with_residues(store)
        outcome = apply_writeback(
            store,
            WritebackRecord(turn_id="t1", records_to_delete=[fact_id]),
            enforcement_passed=True,
        )
        assert outcome.applied_ids == [fact_id]
        assert store.load_profile_records() == []
        assert fact_id not in store.load_recent_queue()
        assert all(fact_id not in ids for ids in store.load_tag_index().values())
        (episode,) = store.load_episodes()
        assert episode.linked_facts == []
        assert verify_forgotten(store, fact_id).forgotten

    def test_writeback_delete_unknown_refused(self, store: MemoryStore) -> None:
        outcome = apply_writeback(
            store,
            WritebackRecord(turn_id="t1", records_to_delete=["nope"]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert "not found" in outcome.refused[0]["reasons"][0]

    def test_writeback_delete_identity_refused(self, store: MemoryStore) -> None:
        store.save_personality_core(PersonalityCoreRecord(record_id="pers_seed"))
        outcome = apply_writeback(
            store,
            WritebackRecord(turn_id="t1", records_to_delete=["pers_seed"]),
            enforcement_passed=True,
        )
        assert outcome.applied == []
        assert "identity" in outcome.refused[0]["reasons"][0]
        assert store.load_personality_core() is not None

    def test_writeback_archive_is_not_forgetting(self, store: MemoryStore) -> None:
        # Archive keeps its retirement shape (TF-047): the
        # record leaves its file; forgetting's proof is a
        # separate, stronger claim this leg does not make.
        fact_id, _ = seed_fact_with_residues(store)
        outcome = apply_writeback(
            store,
            WritebackRecord(turn_id="t1", records_to_archive=[fact_id]),
            enforcement_passed=True,
        )
        assert outcome.applied_ids == [fact_id]
        assert store.load_profile_records() == []
