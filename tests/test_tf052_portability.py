"""TF-052 — store portability: a whole store as a bundle.

Pins knowledge/portability.py against the live tree: the
bundle shape as data, export with manifest counts and
checksums, verify_bundle's gate (tampering, invalid
records, duplicate ids, wrong format/version), both
import modes (replace is whole, merge never overwrites),
the refused-import guarantee (target untouched), the
import provenance stamp, the queue filter, and the tag
index rebuilt rather than carried.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.portability import (
    BUNDLE_FORMAT,
    BUNDLE_SECTIONS,
    BUNDLE_VERSION,
    EXCLUDED_SURFACES,
    IMPORT_MODES,
    IMPORT_SOURCE,
    IMPORT_STATUSES,
    ImportReport,
    export_store,
    import_store,
    portability_problems,
    read_bundle,
    verify_bundle,
    write_bundle,
)
from thoughtforge.knowledge.store import MemoryStore


def seed_store(store: MemoryStore) -> MemoryStore:
    store.save_personality_core(
        PersonalityCoreRecord(record_id="pers_seed", traits=["calm"])
    )
    store.append_profile_record(
        UserPreferenceRecord(
            record_id="usr_tone", category="tone", tags=["style"],
            summary="tone: terse", value="terse", source="user_stated",
        )
    )
    store.append_profile_record(
        UserFactRecord(
            record_id="fct_ferry", tags=["ferry"],
            summary="ferry leaves at 07:30",
            fact_value={"departure": "07:30"}, source="user_stated",
        )
    )
    store.append_episode(
        EpisodicMemoryRecord(
            record_id="ep_ferry", tags=["ferry"], summary="talked ferries",
            linked_facts=["fct_ferry"], source="conversation_writeback",
        )
    )
    store.append_pattern(
        ResponsePatternRecord(
            record_id="pat_shape", summary="a shape",
            pattern_shape=["ack", "answer"], source="manual_entry",
        )
    )
    store.save_thread_state(
        ActiveThreadStateRecord(record_id="thr_now", topic="ferries")
    )
    store.rebuild_tag_index()
    store.push_to_recent_queue("fct_ferry")
    store.push_to_recent_queue("ep_ferry")
    return store


@pytest.fixture()
def source(tmp_path: Path) -> MemoryStore:
    return seed_store(MemoryStore(memory_dir=tmp_path / "src"))


@pytest.fixture()
def target(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path / "dst")


def dir_bytes(store: MemoryStore) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in sorted(store.dir.iterdir()) if p.is_file()
    }


# ── Self-check and bundle data ───────────────────────────────────────────────


class TestSelfCheck:
    def test_portability_problems_empty(self) -> None:
        assert portability_problems() == []

    def test_sections_and_exclusions_as_data(self) -> None:
        names = [row["name"] for row in BUNDLE_SECTIONS]
        assert names == [
            "personality_core", "profile_records", "episodes",
            "patterns", "thread_state", "recent_queue",
        ]
        excluded = {row["name"] for row in EXCLUDED_SURFACES}
        assert excluded == {"tag_index", "knowledge_db"}
        assert all(row["reason"] for row in EXCLUDED_SURFACES)

    def test_modes_and_statuses_as_data(self) -> None:
        assert IMPORT_MODES == ("replace", "merge")
        assert set(IMPORT_STATUSES) == {"imported", "refused"}
        assert IMPORT_SOURCE == "import"


# ── Export ───────────────────────────────────────────────────────────────────


class TestExport:
    def test_bundle_shape_and_manifest(self, source: MemoryStore) -> None:
        bundle = export_store(source)
        assert bundle["format"] == BUNDLE_FORMAT
        assert bundle["version"] == BUNDLE_VERSION
        counts = bundle["manifest"]["counts"]
        assert counts == {
            "personality_core": 1, "profile_records": 2, "episodes": 1,
            "patterns": 1, "thread_state": 1, "recent_queue": 2,
        }
        assert bundle["manifest"]["total_records"] == 6
        assert set(bundle["manifest"]["checksums"]) == set(counts)
        assert verify_bundle(bundle) == []

    def test_tag_index_not_carried(self, source: MemoryStore) -> None:
        bundle = export_store(source)
        assert "tag_index" not in bundle["sections"]
        assert "tag_index" not in json.dumps(bundle)

    def test_empty_store_exports_and_verifies(self, target: MemoryStore) -> None:
        bundle = export_store(target)
        assert bundle["manifest"]["total_records"] == 0
        assert bundle["sections"]["personality_core"] is None
        assert bundle["sections"]["thread_state"] is None
        assert verify_bundle(bundle) == []

    def test_bundle_file_round_trip(
        self, source: MemoryStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "bundle.json"
        written = write_bundle(source, path)
        loaded = read_bundle(path)
        assert loaded == written
        assert verify_bundle(loaded) == []


# ── The gate ─────────────────────────────────────────────────────────────────


class TestVerify:
    def test_tampered_record_fails_checksum_by_section(
        self, source: MemoryStore
    ) -> None:
        bundle = export_store(source)
        bundle["sections"]["profile_records"][0]["summary"] = "tone: verbose"
        problems = verify_bundle(bundle)
        assert any("checksum" in p and "profile_records" in p for p in problems)

    def test_tampered_count_is_caught(self, source: MemoryStore) -> None:
        bundle = export_store(source)
        bundle["manifest"]["counts"]["episodes"] = 99
        problems = verify_bundle(bundle)
        assert any("count" in p and "episodes" in p for p in problems)

    def test_wrong_format_and_version_refused(self, source: MemoryStore) -> None:
        bundle = export_store(source)
        bundle["format"] = "something-else"
        bundle["version"] = 99
        problems = verify_bundle(bundle)
        assert any("format" in p for p in problems)
        assert any("version" in p for p in problems)

    def test_duplicate_ids_refused(self, source: MemoryStore) -> None:
        bundle = export_store(source)
        bundle["sections"]["episodes"].append(
            copy.deepcopy(bundle["sections"]["profile_records"][1])
            | {"record_type": "episode", "summary": "dup"}
        )
        # fix manifest so only the duplicate fault remains visible
        bundle = _remanifest(bundle)
        problems = verify_bundle(bundle)
        assert any("duplicate" in p and "fct_ferry" in p for p in problems)

    def test_invalid_record_refused_by_contract(
        self, source: MemoryStore
    ) -> None:
        bundle = export_store(source)
        bundle["sections"]["profile_records"][1]["summary"] = ""
        bundle["sections"]["profile_records"][1]["fact_value"] = {}
        bundle = _remanifest(bundle)
        problems = verify_bundle(bundle)
        assert any("fct_ferry" in p for p in problems)

    def test_non_mapping_bundle_is_a_problem(self) -> None:
        assert verify_bundle(["not", "a", "bundle"])


def _remanifest(bundle: dict) -> dict:
    """Recompute counts/checksums after a deliberate edit,
    so a test can isolate the fault it planted."""
    from thoughtforge.knowledge.portability import _checksum, _section_count

    sections = bundle["sections"]
    for row in BUNDLE_SECTIONS:
        name = row["name"]
        bundle["manifest"]["counts"][name] = _section_count(
            sections[name], row["kind"]
        )
        bundle["manifest"]["checksums"][name] = _checksum(sections[name])
    bundle["manifest"]["total_records"] = sum(
        bundle["manifest"]["counts"][n]
        for n in ("personality_core", "profile_records", "episodes",
                  "patterns", "thread_state")
    )
    return bundle


# ── Import: replace ──────────────────────────────────────────────────────────


class TestReplace:
    def test_round_trip_lands_everything(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        bundle = export_store(source)
        report = import_store(target, bundle, mode="replace")
        assert isinstance(report, ImportReport)
        assert report.imported and report.status == "imported"
        assert report.landed == {
            "personality_core": 1, "profile_records": 2, "episodes": 1,
            "patterns": 1, "thread_state": 1,
        }
        assert target.load_personality_core().record_id == "pers_seed"
        assert target.load_thread_state().record_id == "thr_now"
        assert {r.record_id for r in target.load_profile_records()} == {
            "usr_tone", "fct_ferry",
        }
        (episode,) = target.load_episodes()
        assert episode.linked_facts == ["fct_ferry"]
        # The queue landed in order; the index was rebuilt.
        assert target.load_recent_queue() == ["ep_ferry", "fct_ferry"]
        assert target.load_tag_index().get("ferry") == [
            "fct_ferry", "ep_ferry",
        ] or set(target.load_tag_index().get("ferry", [])) == {
            "fct_ferry", "ep_ferry",
        }

    def test_import_stamps_provenance_only(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        bundle = export_store(source)
        import_store(target, bundle, mode="replace")
        for record in (
            target.load_profile_records() + target.load_episodes()
            + target.load_patterns()
        ):
            assert record.source == "import"
        # Everything but the stamp is verbatim.
        again = export_store(target)
        for section in ("profile_records", "episodes", "patterns"):
            for landed, original in zip(
                again["sections"][section], bundle["sections"][section]
            ):
                assert {k: v for k, v in landed.items() if k != "source"} == {
                    k: v for k, v in original.items() if k != "source"
                }
        assert again["sections"]["personality_core"] == (
            bundle["sections"]["personality_core"]
        )
        assert again["sections"]["thread_state"] == (
            bundle["sections"]["thread_state"]
        )

    def test_second_round_trip_is_exactly_stable(
        self, source: MemoryStore, target: MemoryStore, tmp_path: Path
    ) -> None:
        import_store(target, export_store(source), mode="replace")
        third = MemoryStore(memory_dir=tmp_path / "third")
        import_store(third, export_store(target), mode="replace")
        assert export_store(third)["sections"] == export_store(target)["sections"]

    def test_replace_removes_what_the_bundle_lacks(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        seed_store(target)
        empty = MemoryStore(memory_dir=target.dir.parent / "empty")
        report = import_store(target, export_store(empty), mode="replace")
        assert report.imported
        assert target.load_profile_records() == []
        assert target.load_episodes() == []
        assert target.load_patterns() == []
        assert target.load_personality_core() is None
        assert target.load_thread_state() is None
        assert source is not target  # the seeded source is untouched

    def test_refused_import_leaves_target_untouched(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        seed_store(target)
        before = dir_bytes(target)
        bundle = export_store(source)
        bundle["sections"]["patterns"][0]["summary"] = "tampered"
        report = import_store(target, bundle, mode="replace")
        assert report.status == "refused" and not report.imported
        assert report.problems
        assert dir_bytes(target) == before


# ── Import: merge ────────────────────────────────────────────────────────────


class TestMerge:
    def test_merge_adds_new_records(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        target.append_profile_record(
            UserFactRecord(record_id="fct_local", summary="local fact")
        )
        report = import_store(target, export_store(source), mode="merge")
        assert report.imported and report.skipped_collisions == []
        assert {r.record_id for r in target.load_profile_records()} == {
            "fct_local", "usr_tone", "fct_ferry",
        }

    def test_merge_collision_skipped_never_overwritten(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        target.append_profile_record(
            UserFactRecord(
                record_id="fct_ferry", summary="the local truth",
                fact_value={"departure": "09:00"}, source="user_stated",
            )
        )
        report = import_store(target, export_store(source), mode="merge")
        assert report.imported
        assert report.skipped_collisions == ["fct_ferry"]
        (kept,) = [
            r for r in target.load_profile_records()
            if r.record_id == "fct_ferry"
        ]
        assert kept.summary == "the local truth"
        assert kept.source == "user_stated"

    def test_merge_personality_collision_keeps_target_core(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        target.save_personality_core(
            PersonalityCoreRecord(record_id="pers_seed", traits=["local"])
        )
        report = import_store(target, export_store(source), mode="merge")
        assert "pers_seed" in report.skipped_collisions
        assert target.load_personality_core().traits == ["local"]

    def test_queue_filtered_to_landed_ids(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        bundle = export_store(source)
        bundle["sections"]["recent_queue"] = ["fct_ferry", "ghost_id"]
        bundle = _remanifest(bundle)
        report = import_store(target, bundle, mode="merge")
        assert report.imported
        assert report.queue_dropped == 1
        assert target.load_recent_queue() == ["fct_ferry"]

    def test_mode_must_be_named(self, source: MemoryStore,
                                 target: MemoryStore) -> None:
        with pytest.raises(ValueError, match="mode"):
            import_store(target, export_store(source), mode="overwrite")

    def test_report_dict_shape(
        self, source: MemoryStore, target: MemoryStore
    ) -> None:
        data = import_store(
            target, export_store(source), mode="replace"
        ).to_dict()
        assert data["status"] == "imported" and data["imported"] is True
        assert data["mode"] == "replace"
