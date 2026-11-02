"""
TF-112 — Ingestion dry-run: a source can be
test-ingested into a scratch store with a full
report before touching a real store.

The driver lives as verifiable data in
``src/thoughtforge/etl/ingestion_dry_run.py``
(``BUILDER_SOURCES``, ``DRY_RUN_RULES``,
``dry_run_ingestion``, checked by
``dry_run_problems()`` and
``dry_run_live_problems()``). D-COV-2 (the ETL
modules had no direct test imports; its remaining
share was this dry-run's depth) is FIXED here;
D-ETL-2's TF-112 share (the dry-run consults the
TF-106 registry first) is discharged here — the
entry stays open for TF-119.

Pinned here:

  * the map covers TF-105's five builders exactly,
    each bound to its TF-106 primary registry entry,
  * a GeoNames dry run files the full story in one
    report: stats, scratch counts, the TF-108 gate
    ledger (one refusal, reason out_of_range), the
    TF-109 merge ledger (one conflict collision),
    overlaps, registry entry and measured bytes,
  * the named real store is proven untouched
    (hash + counts agree; a sentinel row survives),
    and an unnamed target files None honestly,
  * every builder dry-runs through the same road,
  * the reference builder passes its registry gate
    as quarantined (licence None, quarantine in the
    report), resolves by its registry id too, and
    files gate=None with its note — never a fake
    empty ledger,
  * an unregistered source raises by name and
    creates no scratch; a missing input raises
    before any scratch; a scratch equal to the
    target raises DryRunTargetError,
  * an unkept scratch is removed, a kept scratch
    stays at the reported path, and sequential
    appended runs compose one estate whose report
    names the cross-source overlap,
  * the embedding leg is not run without an
    encoder, and with an injected encoder (TF-111's
    filed first consumer) it runs through TF-111's
    driver over the scratch,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented
records in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic sources through
the real builders into scratch stores on this VM).
A real dump dry-run at scale is NOT PROVEN.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl.ingestion_dry_run import (
    BUILDER_SOURCES,
    DRY_RUN_FORMAT,
    DRY_RUN_RULES,
    SCRATCH_DB_NAME,
    DryRunTargetError,
    dry_run_as_data,
    dry_run_ingestion,
    dry_run_live_problems,
    dry_run_problems,
    report_as_data,
)
from thoughtforge.etl.schema import get_connection, initialize_schema


def _inputs(root: Path) -> dict[str, Path]:
    import json

    root.mkdir(parents=True, exist_ok=True)
    entity = {
        "id": "Q112001",
        "labels": {"en": {"value": "Lantern Town"}},
        "descriptions": {"en": {"value": "A synthetic town"}},
        "aliases": {"en": [{"value": "the lit town"}]},
        "sitelinks": {"enwiki": {}},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q112900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }
    dump = root / "dump.json"
    dump.write_text(json.dumps([entity]), encoding="utf-8")
    csv_path = root / "assertions.csv"
    csv_path.write_text(
        "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}\n",
        encoding="utf-8")
    base = ["112001", "Lantern Town", "Lantern Town", "", "1.5",
            "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
            "10", "10", "UTC", "2026-01-01"]
    changed = list(base); changed[1] = "Lantern Town Renamed"
    bad = list(base); bad[0] = "112002"; bad[4] = "999.0"
    txt_path = root / "places.txt"
    txt_path.write_text("\n".join(
        ["\t".join(base), "\t".join(changed), "\t".join(bad)]) + "\n",
        encoding="utf-8")
    single_path = root / "places_single.txt"
    single_path.write_text("\t".join(base) + "\n", encoding="utf-8")
    json_path = root / "abstracts.json"
    json_path.write_text(json.dumps({
        "uri": "http://example.org/Lantern_Town",
        "label": "Lantern Town", "abstract": "A synthetic town.",
        "type": "Place"}) + "\n", encoding="utf-8")
    ref_dir = root / "ref"
    ref_dir.mkdir(exist_ok=True)
    (ref_dir / "lantern_care.md").write_text(
        "# Lantern Care\n\nKeep the lantern trimmed.\n",
        encoding="utf-8")
    return {"wikidata": dump, "conceptnet": csv_path,
            "geonames": txt_path, "geonames_single": single_path,
            "dbpedia": json_path, "reference": ref_dir}


def _real_store(root: Path) -> Path:
    db = root / "real.db"
    initialize_schema(db)
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en) VALUES (?, ?)",
            ("Q999", "sentinel"))
        conn.commit()
    finally:
        conn.close()
    return db


def _fake_encoder(texts, model_name):
    vectors = []
    for text in texts:
        seed = int(hashlib.sha256(
            text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vectors.append(rng.random(384, dtype=np.float32))
    return np.array(vectors, dtype=np.float32)


class TestMapAndRules:
    def test_rules_are_unique_data(self) -> None:
        ids = [r["id"] for r in DRY_RUN_RULES]
        assert len(ids) == len(set(ids))
        assert "the_registry_is_consulted_first" in ids
        assert "the_target_is_proven_untouched" in ids
        assert "embeddings_only_with_an_injected_encoder" in ids

    def test_map_covers_the_five_builders(self) -> None:
        assert [e["builder_id"] for e in BUILDER_SOURCES] == [
            "wikidata", "conceptnet", "geonames", "dbpedia", "reference"]
        assert BUILDER_SOURCES[-1]["registry_id"] == "reference_library"

    def test_as_data_shape(self) -> None:
        data = dry_run_as_data()
        assert data["format"] == DRY_RUN_FORMAT
        assert len(data["sources"]) == 5
        assert len(data["rules"]) == len(DRY_RUN_RULES)


class TestGeoNamesFullReport:
    def test_full_story_in_one_report(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        real = _real_store(tmp_path)
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], target_db_path=real)
        data = report_as_data(report)
        assert data["format"] == DRY_RUN_FORMAT
        assert report.stats == {"inserted": 2, "skipped": 1}
        assert report.store_counts["geonames_places"] == 1
        assert report.gate is not None
        assert report.gate["refused"] == 1
        assert report.gate["by_reason"] == {"out_of_range": 1}
        assert report.gate["by_source"] == {"geonames": 1}
        assert report.merge["collisions"] == 1
        assert report.merge["by_outcome"] == {"conflict_overwrite": 1}
        assert report.overlaps["group_count"] == 0
        assert report.registry["licence"] == "CC BY 4.0"
        assert report.registry["use_checked"] == "ingest_private"
        assert report.scratch_size_bytes > 0
        assert report.embeddings["status"] == "not_run"

    def test_the_real_store_is_proven_untouched(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        real = _real_store(tmp_path)
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], target_db_path=real)
        assert report.target_untouched is True
        assert report.target is not None
        before, after = report.target["before"], report.target["after"]
        assert before["sha256"] == after["sha256"]
        assert before["counts"] == after["counts"]
        conn = get_connection(real, read_only=True)
        try:
            rows = conn.execute(
                "SELECT qid, label_en FROM entities").fetchall()
        finally:
            conn.close()
        assert [tuple(r) for r in rows] == [("Q999", "sentinel")]

    def test_no_target_files_none(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion("geonames", inputs["geonames"])
        assert report.target is None
        assert report.target_untouched is None

    def test_a_target_that_does_not_exist_stays_absent(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        ghost = tmp_path / "ghost.db"
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], target_db_path=ghost)
        assert report.target_untouched is True
        assert not ghost.exists()


class TestEveryBuilder:
    @pytest.mark.parametrize("source,stats,table,count", [
        ("wikidata", {"entities": 1, "statements": 1, "labels": 1,
                      "skipped": 0}, "entities", 1),
        ("conceptnet", {"inserted": 1, "skipped": 0},
         "conceptnet_edges", 1),
        ("dbpedia", {"inserted": 1, "skipped": 0},
         "dbpedia_entities", 1),
        ("reference", {"files": 1, "chunks": 1, "skipped": 0},
         "reference_chunks", 1),
    ])
    def test_each_builder_dry_runs(self, tmp_path: Path, source,
                                    stats, table, count) -> None:
        inputs = _inputs(tmp_path / "in")
        real = _real_store(tmp_path)
        report = dry_run_ingestion(source, inputs[source],
                                   target_db_path=real)
        assert report.stats == stats
        assert report.store_counts[table] == count
        assert report.target_untouched is True
        assert not Path(report.scratch_path).exists()


class TestRegistryGate:
    def test_reference_is_quarantined_in_the_report(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion("reference", inputs["reference"])
        assert report.registry["status"] == "quarantined_unverified"
        assert report.registry["licence"] is None
        assert "no_redistribution" in report.registry["obligations"]

    def test_reference_resolves_by_registry_id(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion("reference_library",
                                   inputs["reference"])
        assert report.source == "reference"
        assert report.registry_id == "reference_library"

    def test_reference_files_no_gate(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion("reference", inputs["reference"])
        assert report.gate is None
        assert report.gate_note and "no schema gate" in report.gate_note

    def test_unregistered_raises_and_creates_nothing(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        scratch_dir = tmp_path / "never"
        with pytest.raises(ValueError, match="no_such_source_tf112"):
            dry_run_ingestion("no_such_source_tf112",
                              inputs["geonames"],
                              scratch_dir=scratch_dir)
        assert not scratch_dir.exists()


class TestGuards:
    def test_missing_input_raises_before_scratch(
        self, tmp_path: Path,
    ) -> None:
        scratch_dir = tmp_path / "never"
        with pytest.raises(FileNotFoundError, match="absent"):
            dry_run_ingestion("geonames", tmp_path / "absent.txt",
                              scratch_dir=scratch_dir)
        assert not scratch_dir.exists()

    def test_scratch_equal_to_target_raises(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        with pytest.raises(DryRunTargetError):
            dry_run_ingestion(
                "geonames", inputs["geonames"],
                scratch_dir=tmp_path,
                target_db_path=tmp_path / SCRATCH_DB_NAME)
        assert not (tmp_path / SCRATCH_DB_NAME).exists()

    def test_wrong_input_kind_raises(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        with pytest.raises(FileNotFoundError, match="not a directory"):
            dry_run_ingestion("reference", inputs["geonames"])


class TestScratchDiscipline:
    def test_unkept_scratch_is_removed(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        scratch_dir = tmp_path / "scratch"
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], scratch_dir=scratch_dir)
        assert report.scratch_kept is False
        assert not (scratch_dir / SCRATCH_DB_NAME).exists()

    def test_kept_scratch_stays(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        scratch_dir = tmp_path / "scratch"
        report = dry_run_ingestion(
            "geonames", inputs["geonames"], scratch_dir=scratch_dir,
            keep_scratch=True)
        assert report.scratch_kept is True
        assert Path(report.scratch_path).is_file()
        assert report.store_counts["geonames_places"] == 1

    def test_appended_runs_compose_and_name_the_overlap(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        shared = tmp_path / "shared"
        dry_run_ingestion("wikidata", inputs["wikidata"],
                          scratch_dir=shared, keep_scratch=True)
        composed = dry_run_ingestion(
            "geonames", inputs["geonames_single"],
            scratch_dir=shared, keep_scratch=True, append=True)
        assert composed.appended is True
        names = [g["name"] for g in composed.overlaps["groups"]]
        assert "lantern town" in names
        group = next(g for g in composed.overlaps["groups"]
                     if g["name"] == "lantern town")
        assert {m["table"] for m in group["members"]} == {
            "entities", "geonames_places"}

    def test_non_append_run_starts_fresh(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        shared = tmp_path / "shared"
        dry_run_ingestion("wikidata", inputs["wikidata"],
                          scratch_dir=shared, keep_scratch=True)
        fresh = dry_run_ingestion(
            "geonames", inputs["geonames_single"],
            scratch_dir=shared, keep_scratch=True)
        assert fresh.appended is False
        assert fresh.store_counts["entities"] == 0
        assert fresh.store_counts["geonames_places"] == 1


class TestEmbeddingLeg:
    def test_injected_encoder_runs_tf111_over_the_scratch(
        self, tmp_path: Path,
    ) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion(
            "wikidata", inputs["wikidata"], encoder=_fake_encoder)
        assert report.embeddings["status"] == "run"
        assert report.embeddings["entities_built"] == 1
        assert report.embeddings["reference_built"] == 0
        assert report.store_counts["embeddings"] == 1

    def test_no_encoder_no_leg(self, tmp_path: Path) -> None:
        inputs = _inputs(tmp_path / "in")
        report = dry_run_ingestion("wikidata", inputs["wikidata"])
        assert report.embeddings["status"] == "not_run"
        assert report.store_counts["embeddings"] == 0


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert dry_run_problems() == []

    def test_live_checker_clean(self, tmp_path: Path) -> None:
        assert dry_run_live_problems(tmp_path / "live") == []
