"""
TF-115 — Ingestion metrics: records/sec, refusal
rate, store growth per source — reported per run.

The metrics live as verifiable data in
``src/thoughtforge/etl/ingestion_metrics.py``
(``SOURCE_METRIC_SHAPES``, ``METRICS_RULES``,
``store_snapshot``, ``measure_ingestion``,
``summarise_runs``, checked by ``metrics_problems()``
and ``metrics_live_problems()``); TF-105's living map
was updated in the open (its ``metrics``
orchestration entry). D-MET-1 (no per-run metrics
report existed), D-MET-2 (no build records its
elapsed, so no rate could be computed after the
fact) and D-MET-3 (stats count accepted records the
store never landed, and nothing named the
difference) were found by this slice's probes and
are FIXED here.

Pinned here:

  * the shapes bind exactly to TF-105's builders /
    target tables and TF-112's bindings; the rules as
    data; the report format + doc,
  * snapshots: a missing store is measured without
    being created; an existing store's counts + bytes
    (db + sidecars named apart),
  * the pure computation: rate only from a positive
    measured elapsed (zero / None file None),
    refusal rate = gate refused / seen with filter
    skips filed apart as other_skips, an ungated
    builder files no rate, growth deltas (negative
    growth filed as measured), accepted-vs-landed,
  * through the REAL builders under a fake clock:
    GeoNames (new + conflict + refusal in one run;
    an identical re-run's zero growth), wikidata
    (gate + update ledgers ride), reference (ungated:
    no rate; update ledger present; growth in
    chunks), DBpedia + ConceptNet through the same
    road,
  * a missing input and an unregistered source raise
    before anything is measured or written; a failed
    build files no report,
  * aggregation: per-source sums with rates
    recomputed from the sums, unmeasured runs counted,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented
lantern records in a temp dir, never a real dump or
canon.

Proof grade: FIXTURE (synthetic sources through the
real builders on this VM, rates deterministic under
a fake clock). A real dump's ingestion rate is
NOT PROVEN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.etl.etl_map import SOURCE_BUILDERS
from thoughtforge.etl.ingestion_dry_run import BUILDER_SOURCES
from thoughtforge.etl.ingestion_metrics import (
    METRICS_DOC,
    METRICS_FORMAT,
    METRICS_RULES,
    SOURCE_METRIC_SHAPES,
    build_run_metrics,
    measure_ingestion,
    metrics_as_data,
    metrics_live_problems,
    metrics_problems,
    store_snapshot,
    summarise_runs,
)


def _clock(*values: float):
    ticks = iter(values)
    return lambda: next(ticks)


def _geonames_file(tmp_path: Path) -> Path:
    base = ["115001", "Lantern Town", "Lantern Town", "", "1.5",
            "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "", "100",
            "10", "10", "UTC", "2026-01-01"]
    changed = list(base); changed[1] = "Lantern Town Renamed"
    bad = list(base); bad[0] = "115002"; bad[4] = "999.0"
    txt_path = tmp_path / "places.txt"
    txt_path.write_text("\n".join(
        ["\t".join(base), "\t".join(changed), "\t".join(bad)]) + "\n",
        encoding="utf-8")
    return txt_path


def _snapshot(counts: dict | None, bytes_: int = 0,
              exists: bool = True) -> dict:
    return {"path": "x.db", "exists": exists, "bytes": bytes_,
            "file_bytes": {"db": bytes_, "wal": 0, "shm": 0},
            "counts": counts}


class TestMetricsData:
    def test_format_and_doc(self) -> None:
        assert METRICS_FORMAT == "thoughtforge-ingestion-metrics/1"
        assert METRICS_DOC == "docs/architecture/INGESTION_METRICS.md"
        assert Path(METRICS_DOC).exists()

    def test_shapes_bind_to_tf105_and_tf112(self) -> None:
        shaped = [e["builder_id"] for e in SOURCE_METRIC_SHAPES]
        assert shaped == [b["id"] for b in SOURCE_BUILDERS]
        assert shaped == [e["builder_id"] for e in BUILDER_SOURCES]
        filed = {b["id"]: tuple(b["tables"]) for b in SOURCE_BUILDERS}
        for entry in SOURCE_METRIC_SHAPES:
            assert entry["target_tables"] == filed[entry["builder_id"]]
            assert entry["primary_table"] in entry["target_tables"]

    def test_rules_unique_and_complete(self) -> None:
        ids = [r["id"] for r in METRICS_RULES]
        assert len(ids) == len(set(ids)) == 12
        for needed in (
                "metrics_are_measured_never_derived_from_stats_alone",
                "refusal_rate_is_gate_refused_over_gate_seen",
                "store_growth_is_a_delta_of_snapshots",
                "accepted_is_not_landed_and_the_difference_is_filed"):
            assert needed in ids

    def test_as_data_shape(self) -> None:
        data = metrics_as_data()
        assert data["format"] == METRICS_FORMAT
        assert len(data["sources"]) == 5
        assert len(data["rules"]) == 12


class TestSnapshot:
    def test_missing_store_measured_not_created(
            self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost.db"
        snap = store_snapshot(ghost)
        assert snap["exists"] is False
        assert snap["bytes"] == 0
        assert snap["counts"] is None
        assert not ghost.exists()

    def test_existing_store_counts_and_bytes(
            self, tmp_path: Path) -> None:
        db = tmp_path / "g.db"
        measure_ingestion("geonames", _geonames_file(tmp_path), db,
                          clock=_clock(0.0, 1.0))
        snap = store_snapshot(db)
        assert snap["exists"] is True
        assert snap["counts"]["geonames_places"] == 1
        assert snap["bytes"] == snap["file_bytes"]["db"] + \
            snap["file_bytes"]["wal"] + snap["file_bytes"]["shm"]
        assert snap["bytes"] > 0


class TestPureComputation:
    def _gate(self, seen: int = 4, refused: int = 1) -> dict:
        return {"format": "thoughtforge-schema-gate/1", "seen": seen,
                "refused": refused, "by_reason": {"out_of_range": 1},
                "by_source": {"geonames": 1}, "refusals": []}

    def test_rate_and_refusal_rate(self) -> None:
        report = build_run_metrics(
            "geonames", {"inserted": 3, "skipped": 1}, self._gate(),
            {"collisions": 0}, None, "note",
            _snapshot({"geonames_places": 0}),
            _snapshot({"geonames_places": 3}, bytes_=500),
            elapsed_seconds=2.0)
        assert report.records == {"unit": "places", "accepted": 3,
                                  "skipped": 1, "presented": 4,
                                  "output_rows": 3}
        assert report.records_per_second == 2.0
        assert report.refusals["refusal_rate"] == 0.25
        assert report.refusals["other_skips"] == 0
        assert report.growth["counts"]["geonames_places"] == 3
        assert report.growth["bytes"] == 500
        assert report.growth["accepted_vs_primary_growth"] == 0

    def test_zero_and_unmeasured_elapsed_file_no_rate(self) -> None:
        for elapsed in (0.0, None):
            report = build_run_metrics(
                "geonames", {"inserted": 1, "skipped": 0}, self._gate(),
                {"collisions": 0}, None, None,
                _snapshot({"geonames_places": 0}),
                _snapshot({"geonames_places": 1}),
                elapsed_seconds=elapsed)
            assert report.records_per_second is None

    def test_filter_skips_are_not_refusals(self) -> None:
        report = build_run_metrics(
            "geonames", {"inserted": 2, "skipped": 3}, self._gate(),
            {"collisions": 0}, None, None,
            _snapshot({"geonames_places": 0}),
            _snapshot({"geonames_places": 2}),
            elapsed_seconds=1.0)
        assert report.refusals["refused"] == 1
        assert report.refusals["other_skips"] == 2

    def test_ungated_files_no_rate(self) -> None:
        report = build_run_metrics(
            "reference", {"files": 1, "chunks": 2, "skipped": 0}, None,
            {"collisions": 0}, {"updates": 0}, None,
            _snapshot({"reference_chunks": 0}),
            _snapshot({"reference_chunks": 2}),
            elapsed_seconds=1.0)
        assert report.refusals["gate_present"] is False
        assert report.refusals["refusal_rate"] is None
        assert report.records["accepted"] == 1
        assert report.records["output_rows"] == 2
        assert report.growth["accepted_vs_primary_growth"] == 0

    def test_negative_growth_filed_as_measured(self) -> None:
        report = build_run_metrics(
            "reference", {"files": 1, "chunks": 0, "skipped": 1}, None,
            {"collisions": 0}, {"updates": 2}, None,
            _snapshot({"reference_chunks": 2}, bytes_=900),
            _snapshot({"reference_chunks": 0}, bytes_=400),
            elapsed_seconds=1.0)
        assert report.growth["counts"]["reference_chunks"] == -2
        assert report.growth["total_rows"] == -2
        assert report.growth["bytes"] == -500
        assert report.growth["accepted_vs_primary_growth"] == 2


class TestMeasuredRuns:
    def test_geonames_full_story_one_run(self, tmp_path: Path) -> None:
        db = tmp_path / "g.db"
        report = measure_ingestion(
            "geonames", _geonames_file(tmp_path), db,
            clock=_clock(10.0, 12.0))
        data = report.to_dict()
        assert data["format"] == METRICS_FORMAT
        assert report.stats == {"inserted": 2, "skipped": 1}
        assert report.records["presented"] == 3
        assert report.elapsed_seconds == 2.0
        assert report.records_per_second == 1.5
        assert report.refusals["refused"] == 1
        assert report.refusals["by_reason"] == {"out_of_range": 1}
        assert report.refusals["refusal_rate"] == pytest.approx(1 / 3)
        assert report.refusals["other_skips"] == 0
        assert report.merge["by_outcome"] == {"conflict_overwrite": 1}
        assert report.update is None and report.update_note
        assert report.store_before["exists"] is False
        assert report.growth["counts"]["geonames_places"] == 1
        assert report.growth["accepted_vs_primary_growth"] == 1
        assert report.growth["bytes"] > 0

    def test_identical_rerun_zero_growth(self, tmp_path: Path) -> None:
        db = tmp_path / "g.db"
        txt_path = _geonames_file(tmp_path)
        measure_ingestion("geonames", txt_path, db, clock=_clock(0.0, 1.0))
        rerun = measure_ingestion("geonames", txt_path, db,
                                  clock=_clock(5.0, 6.0))
        assert rerun.growth["counts"]["geonames_places"] == 0
        assert rerun.growth["accepted_vs_primary_growth"] == 2
        assert rerun.store_after["counts"]["geonames_places"] == 1

    def test_wikidata_ledgers_ride(self, tmp_path: Path) -> None:
        entity = {
            "id": "Q115001",
            "labels": {"en": {"value": "Lantern Town"}},
            "descriptions": {"en": {"value": "A synthetic town"}},
            "aliases": {"en": [{"value": "the lit town"}]},
            "sitelinks": {"enwiki": {}},
            "claims": {"P31": [{
                "mainsnak": {"snaktype": "value",
                             "datavalue": {"type": "wikibase-entityid",
                                           "value": {"id": "Q115900"}}},
                "rank": "normal", "qualifiers": {}, "references": [],
            }]},
        }
        bad = {"id": 115002}
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps([entity, bad]), encoding="utf-8")
        report = measure_ingestion(
            "wikidata", dump, tmp_path / "w.db",
            builder_kwargs={"resume": False}, clock=_clock(0.0, 2.0))
        assert report.stats == {"entities": 1, "statements": 1,
                                "labels": 1, "skipped": 1}
        assert report.records["presented"] == 2
        assert report.records["output_rows"] == 3
        assert report.records_per_second == 1.0
        assert report.refusals["refused"] == 1
        assert report.refusals["refusal_rate"] == 0.5
        assert report.update is not None
        assert report.update["updates"] == 0
        assert report.growth["counts"]["entities"] == 1
        assert report.growth["accepted_vs_primary_growth"] == 0

    def test_reference_ungated_with_update_ledger(
            self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        (ref_dir / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        report = measure_ingestion("reference", ref_dir, tmp_path / "r.db",
                                   clock=_clock(0.0, 1.0))
        assert report.stats == {"files": 1, "chunks": 1, "skipped": 0}
        assert report.refusals["gate_present"] is False
        assert report.refusals["refusal_rate"] is None
        assert report.update is not None
        assert report.growth["counts"]["reference_chunks"] == 1
        assert report.growth["accepted_vs_primary_growth"] == 0

    def test_conceptnet_and_dbpedia_same_road(
            self, tmp_path: Path) -> None:
        csv_path = tmp_path / "a.csv"
        csv_path.write_text(
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t"
            "{\"weight\": 2.0}\n", encoding="utf-8")
        cn = measure_ingestion("conceptnet", csv_path, tmp_path / "c.db",
                               clock=_clock(0.0, 1.0))
        assert cn.growth["counts"]["conceptnet_edges"] == 1
        assert cn.refusals["refusal_rate"] == 0.0

        json_path = tmp_path / "a.json"
        json_path.write_text(json.dumps({
            "uri": "http://example.org/Lantern", "label": "Lantern",
            "abstract": "A synthetic light.", "type": "Device"}) + "\n",
            encoding="utf-8")
        dp = measure_ingestion("dbpedia", json_path, tmp_path / "d.db",
                               clock=_clock(0.0, 1.0))
        assert dp.growth["counts"]["dbpedia_entities"] == 1
        assert dp.growth["accepted_vs_primary_growth"] == 0

    def test_missing_input_and_unknown_source_raise_first(
            self, tmp_path: Path) -> None:
        db = tmp_path / "g.db"
        with pytest.raises(FileNotFoundError):
            measure_ingestion("geonames", tmp_path / "nope.txt", db)
        assert not db.exists()
        with pytest.raises(ValueError, match="unregistered source"):
            measure_ingestion("no_such_source_tf115",
                              _geonames_file(tmp_path), db)
        assert not db.exists()


class TestAggregation:
    def test_rates_recomputed_from_sums(self, tmp_path: Path) -> None:
        db = tmp_path / "g.db"
        txt_path = _geonames_file(tmp_path)
        first = measure_ingestion("geonames", txt_path, db,
                                  clock=_clock(10.0, 12.0))
        second = measure_ingestion("geonames", txt_path, db,
                                   clock=_clock(20.0, 21.0))
        summary = summarise_runs([first, second])
        geo = summary["sources"]["geonames"]
        assert geo["runs"] == 2
        assert geo["presented"] == 6
        assert geo["elapsed_seconds"] == 3.0
        assert geo["records_per_second"] == 2.0
        assert geo["gate_refused"] == 2
        assert geo["refusal_rate"] == pytest.approx(2 / 6)
        assert geo["growth_rows"] == 1
        assert summary["totals"]["runs"] == 2
        assert summary["format"] == METRICS_FORMAT

    def test_unmeasured_runs_counted(self) -> None:
        report = build_run_metrics(
            "geonames", {"inserted": 1, "skipped": 0},
            {"seen": 1, "refused": 0, "by_reason": {}, "by_source": {}},
            {"collisions": 0}, None, None,
            _snapshot({"geonames_places": 0}),
            _snapshot({"geonames_places": 1}),
            elapsed_seconds=None)
        summary = summarise_runs([report])
        assert summary["sources"]["geonames"]["unmeasured_runs"] == 1
        assert summary["sources"]["geonames"]["records_per_second"] is None


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert metrics_problems() == []

    def test_live_checker_clean(self) -> None:
        assert metrics_live_problems() == []
