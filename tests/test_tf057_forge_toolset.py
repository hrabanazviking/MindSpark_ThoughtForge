"""TF-057 — knowledge forge UI-shape: the forge_memory toolset.

Pins knowledge/forge_toolset.py against the live tree: the
operations and add-kinds as data (exactly TF-042's stamped
kinds), manual adds stamped source='manual_entry' by
construction and landed through the store gate (invalid and
duplicate refused with nothing written), inspect as a proven
read-only leg (whole-store report, filters, find), prune as
the lifecycle's reported act (unknown mode refused; profile
caps through the toolset), export as a verified TF-052
bundle, and the forge_memory CLI wiring (four new commands,
nine DB commands intact, refusals as exit-1 errors).
Synthetic records only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from thoughtforge.knowledge.forge_toolset import (
    ADD_KIND_NAMES,
    ADD_KINDS,
    MANUAL_SOURCE,
    OPERATIONS,
    add_record,
    export_store_bundle,
    find_record,
    forge_toolset_problems,
    inspect_records,
    inspect_store,
    prune_store,
)
from thoughtforge.knowledge.ingestion_contract import (
    FACT_KINDS,
    MEMORY_PROVENANCE,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    PersonalityCoreRecord,
)
from thoughtforge.knowledge.portability import read_bundle, verify_bundle
from thoughtforge.knowledge.store import MemoryStore

REPO_ROOT = Path(__file__).parent.parent


_forge_memory_mod = None


def _load_forge_memory():
    """Load forge_memory LAZILY, on first CLI test use — never at
    import time. forge_memory configures logging at its top level
    (setup_logging on import, a CLI entry point's normal act);
    loading it during collection would leave a RotatingFileHandler
    on the root logger that TF-038's logging tests count. Loading
    here, at execution time, keeps this file side-effect-free at
    collection."""
    global _forge_memory_mod
    if _forge_memory_mod is None:
        spec = importlib.util.spec_from_file_location(
            "forge_memory_tf057", REPO_ROOT / "forge_memory.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _forge_memory_mod = mod
    return _forge_memory_mod


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(memory_dir=tmp_path / "store")


def _dir_bytes(store: MemoryStore) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(store.dir.iterdir())
            if p.is_file()}


# ── The toolset as data ──────────────────────────────────────────────────────


class TestToolsetData:
    def test_problems_empty_on_live_tree(self) -> None:
        assert forge_toolset_problems() == []

    def test_operations_exactly_the_four(self) -> None:
        assert [r["name"] for r in OPERATIONS] == [
            "add", "inspect", "prune", "export"]
        assert [r["cli"] for r in OPERATIONS] == [
            "add", "inspect", "prune", "export"]
        writes = {r["name"]: r["writes_store"] for r in OPERATIONS}
        assert writes == {"add": True, "inspect": False,
                          "prune": True, "export": False}

    def test_add_kinds_are_tf042s_stamped_kinds(self) -> None:
        stamped = {
            r["record_type"] for r in FACT_KINDS
            if r["store"] == "memory" and r["provenance_field"] == "source"
        }
        assert set(ADD_KIND_NAMES) == stamped
        assert [r["kind"] for r in ADD_KINDS] == list(ADD_KIND_NAMES)

    def test_manual_source_is_the_contracts_vocabulary(self) -> None:
        assert MANUAL_SOURCE == "manual_entry"
        assert MANUAL_SOURCE in MEMORY_PROVENANCE


# ── Add ──────────────────────────────────────────────────────────────────────


class TestAdd:
    def test_each_kind_lands_stamped(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_preference", summary="tone: terse",
                   category="tone", value="terse", record_id="usr_a")
        add_record(store, "user_fact", summary="ferry at 07:30",
                   record_id="fct_a")
        add_record(store, "episode", summary="talked ferries",
                   record_id="ep_a")
        add_record(store, "response_pattern", summary="a shape",
                   pattern_shape=["ack", "answer"], record_id="pat_a")
        loaded = (
            store.load_profile_records() + store.load_episodes()
            + store.load_patterns()
        )
        assert len(loaded) == 4
        assert all(r.source == "manual_entry" for r in loaded)

    def test_kind_defaults_mirror_the_dataclasses(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        pref = add_record(store, "user_preference", summary="s",
                          category="c")["record"]
        fact = add_record(store, "user_fact", summary="s")["record"]
        ep = add_record(store, "episode", summary="s")["record"]
        pat = add_record(store, "response_pattern",
                         pattern_shape=["a"])["record"]
        assert (pref["importance"], pref["confidence"]) == (0.5, 0.5)
        assert (fact["importance"], fact["confidence"]) == (0.5, 0.6)
        assert (ep["importance"], ep["confidence"], ep["quality"]) == (
            0.4, 0.5, 0.5)
        assert (pat["quality"], pat["confidence"]) == (0.6, 0.6)

    def test_generated_ids_use_the_store_factories(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert add_record(store, "user_fact", summary="x")[
            "record_id"].startswith("fct_")
        assert add_record(store, "user_preference", summary="x",
                          category="c")["record_id"].startswith("usr_")
        assert add_record(store, "episode", summary="x")[
            "record_id"].startswith("ep_")
        assert add_record(store, "response_pattern",
                          pattern_shape=["a"])[
            "record_id"].startswith("pat_")

    def test_add_is_immediately_findable(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_fact", summary="ferry at 07:30",
                   tags=["ferry"], record_id="fct_find")
        assert find_record(store, "fct_find")["found"] is True
        assert store.load_tag_index()["ferry"] == ["fct_find"]
        assert store.load_recent_queue()[0] == "fct_find"

    def test_duplicate_id_refused_across_homes(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_fact", summary="one", record_id="dup_1")
        with pytest.raises(ValueError, match="already exists"):
            add_record(store, "episode", summary="two", record_id="dup_1")
        assert len(store.load_episodes()) == 0

    def test_id_of_core_or_thread_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save_personality_core(
            PersonalityCoreRecord(record_id="pers_x", traits=["calm"]))
        store.save_thread_state(
            ActiveThreadStateRecord(record_id="thr_x", topic="t"))
        for taken in ("pers_x", "thr_x"):
            with pytest.raises(ValueError, match="already exists"):
                add_record(store, "user_fact", summary="s",
                           record_id=taken)

    def test_unknown_kind_refused_naming_valid(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="user_fact"):
            add_record(store, "personality_core", summary="s")
        assert inspect_store(store)["total_records"] == 0

    def test_invalid_record_refused_and_nothing_written(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError):
            add_record(store, "user_fact", summary="")
        with pytest.raises(ValueError):
            add_record(store, "user_fact", summary="x" * 181)
        with pytest.raises(ValueError):
            add_record(store, "user_fact", summary="ok", confidence=2.0)
        with pytest.raises(ValueError):
            add_record(store, "user_preference", summary="no category")
        assert store.load_profile_records() == []
        assert _dir_bytes(store) == {}


# ── Inspect ──────────────────────────────────────────────────────────────────


class TestInspect:
    def _seeded(self, tmp_path: Path) -> MemoryStore:
        store = _store(tmp_path)
        store.save_personality_core(
            PersonalityCoreRecord(record_id="pers_s", traits=["calm"]))
        add_record(store, "user_fact", summary="ferry at 07:30",
                   tags=["ferry"], record_id="fct_s")
        add_record(store, "episode", summary="talked ferries",
                   tags=["ferry"], record_id="ep_s")
        store.save_thread_state(
            ActiveThreadStateRecord(record_id="thr_s", topic="ferries"))
        return store

    def test_store_report(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        report = inspect_store(store)
        assert report["counts"] == {
            "user_preference": 0, "user_fact": 1,
            "episode": 1, "response_pattern": 0}
        assert report["total_records"] == 2
        assert report["counts_by_status"] == {"created": 2}
        assert report["personality_core"] == {
            "present": True, "record_id": "pers_s"}
        assert report["thread_state"]["topic"] == "ferries"
        assert report["recent_queue_length"] == 2
        assert report["tag_count"] == 1
        json.dumps(report)  # JSON-safe

    def test_budget_only_when_profile_named(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        assert "budget" not in inspect_store(store)
        with_budget = inspect_store(store, profile_id="phone_low")
        assert with_budget["budget"]["profile_id"] == "phone_low"
        assert with_budget["budget"]["within_budget"] is True

    def test_record_filters(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        assert [r["record_id"] for r in inspect_records(
            store, kind="user_fact")] == ["fct_s"]
        assert {r["record_id"] for r in inspect_records(
            store, tag="ferry")} == {"fct_s", "ep_s"}
        assert [r["record_id"] for r in inspect_records(
            store, text="FERRY")] == ["fct_s"]
        assert inspect_records(store, text="absent") == []
        rows = inspect_records(store)
        assert {r["kind"] for r in rows} == {
            "user_fact", "episode", "active_thread_state",
            "personality_core"}

    def test_unknown_filter_kind_refused(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        with pytest.raises(ValueError, match="user_fact"):
            inspect_records(store, kind="bogus_kind")

    def test_find_record_homes_and_miss(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        assert find_record(store, "fct_s")["home"] == "profile"
        assert find_record(store, "ep_s")["home"] == "episodes"
        assert find_record(store, "thr_s")["kind"] == "active_thread_state"
        assert find_record(store, "pers_s")["kind"] == "personality_core"
        assert find_record(store, "nope") == {
            "found": False, "record_id": "nope"}

    def test_inspect_is_read_only(self, tmp_path: Path) -> None:
        store = self._seeded(tmp_path)
        before = _dir_bytes(store)
        inspect_store(store, profile_id="phone_low")
        inspect_records(store, tag="ferry")
        find_record(store, "fct_s")
        assert _dir_bytes(store) == before


# ── Prune ────────────────────────────────────────────────────────────────────


class TestPrune:
    def test_unknown_mode_refused_before_any_write(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_fact", summary="keep me",
                   record_id="fct_k")
        before = _dir_bytes(store)
        with pytest.raises(ValueError, match="valid modes"):
            prune_store(store, mode="scorched")
        assert _dir_bytes(store) == before

    def test_fresh_store_prune_is_a_no_op_report(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_fact", summary="fresh", record_id="fct_f")
        report = prune_store(store, mode="routine")
        assert report["counts"] == {
            "archived": 0, "deleted": 0, "deduplicated": 0}
        assert report["before"] == report["after"]
        assert report["after"]["user_fact"] == 1

    def test_profile_caps_reach_the_prune(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for i in range(70):
            # Distinct tags: the lifecycle dedups profile records on
            # category|tags (TF-045), so untagged facts would be
            # duplicates of one another before any cap acted.
            add_record(store, "user_fact", summary=f"fact {i}",
                       tags=[f"t{i}"], record_id=f"fct_cap{i:03d}")
        report = prune_store(store, mode="light",
                             profile_id="phone_low")
        assert report["before"]["user_fact"] == 70
        assert report["after"]["user_fact"] == 64  # phone_low hard cap
        flat = prune_store(store, mode="light")
        assert flat["after"]["user_fact"] == 64  # already inside flat


# ── Export ───────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_writes_a_verified_bundle(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "user_fact", summary="ferry at 07:30",
                   record_id="fct_e")
        out = tmp_path / "bundle.json"
        report = export_store_bundle(store, out)
        assert report["verified"] is True
        assert report["problems"] == []
        assert report["counts"]["profile_records"] == 1
        assert report["total_records"] == 1
        bundle = read_bundle(out)
        assert verify_bundle(bundle) == []
        assert bundle["sections"]["profile_records"][0][
            "source"] == "manual_entry"

    def test_export_does_not_write_the_store(
            self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        add_record(store, "episode", summary="a moment",
                   record_id="ep_e")
        before = _dir_bytes(store)
        export_store_bundle(store, tmp_path / "out.json")
        assert _dir_bytes(store) == before


# ── CLI wiring ───────────────────────────────────────────────────────────────


class TestCli:
    def test_help_lists_toolset_and_db_commands(self) -> None:
        result = CliRunner().invoke(_load_forge_memory().cli, ["--help"])
        assert result.exit_code == 0
        for name in ("add", "inspect", "prune", "export",
                     "wikidata", "conceptnet", "geonames", "dbpedia",
                     "reference", "embeddings", "status", "init", "all"):
            assert name in result.output

    def test_cli_round_trip(self, tmp_path: Path) -> None:
        runner = CliRunner()
        mem = tmp_path / "mem"
        added = runner.invoke(_load_forge_memory().cli, [
            "--memory-dir", str(mem), "add", "--kind", "user_fact",
            "--summary", "ferry at 07:30", "--tag", "ferry",
            "--record-id", "fct_cli"])
        assert added.exit_code == 0, added.output
        assert json.loads(added.output)["source"] == "manual_entry"

        shown = runner.invoke(_load_forge_memory().cli, [
            "--memory-dir", str(mem), "inspect",
            "--record-id", "fct_cli"])
        assert json.loads(shown.output)["found"] is True

        exported = runner.invoke(_load_forge_memory().cli, [
            "--memory-dir", str(mem), "export",
            "--out", str(tmp_path / "b.json")])
        assert json.loads(exported.output)["verified"] is True

    def test_cli_refusal_is_an_error_not_a_write(
            self, tmp_path: Path) -> None:
        runner = CliRunner()
        mem = tmp_path / "mem"
        bad = runner.invoke(_load_forge_memory().cli, [
            "--memory-dir", str(mem), "add", "--kind", "user_fact",
            "--summary", ""])
        assert bad.exit_code == 1
        assert not (mem / "user_profile_store.jsonl").exists()
        bad_prune = runner.invoke(_load_forge_memory().cli, [
            "--memory-dir", str(mem), "prune", "--mode", "scorched"])
        assert bad_prune.exit_code == 1
        assert "valid modes" in bad_prune.output
