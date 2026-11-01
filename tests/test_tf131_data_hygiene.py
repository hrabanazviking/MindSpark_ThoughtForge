"""
TF-131 — Benchmark data hygiene: all fixtures
synthetic or licensed; canon never a benchmark
input.

The proof lives as verifiable data in
``benchmarks/data_hygiene.py`` (``HYGIENE_ROWS``,
``HYGIENE_PROBES``, ``HYGIENE_RULES``, checked by
``data_hygiene_problems()``), registered in
``benchmarks/harness.py`` with the pinned
baseline ``data_hygiene_synthetic_v1``; D-BENCH-16
(the benchmark estate carried no per-fixture
provenance filing at all — TF-017's guards scan
tests/ only, so a benchmark corpus could have
drifted to real material and nothing running
would have named it) was fixed by this slice in
the measuring sense.

Pinned here:

  * the filed inventory: 21 inputs — 15
    synthetic-invented, 0 licensed, 6
    derived-live — every harness fixture
    covered exactly, every binding resolving
    live, every proof in order,
  * the canon scan: TF-017's guard shapes
    (assembled from parts in this file too)
    find no canon-access shape anywhere in
    the benchmark code estate,
  * the verdict shape: full inventory PASS,
    partial and empty selections REVIEW (no
    vacuous pass), an unclean source row
    failing its derived rows by name,
  * the probes: 7/7 against the pure checker
    parts, proving the checker can fail,
  * the checker clean, the harness
    registration (20 / 13 / 13 / 109, rules
    13 — counts moved in the open by TF-132,
    whose new synthetic fixture grew this
    inventory 20 -> 21 inputs, 14 -> 15
    synthetic, and re-pinned this baseline in
    the open for exactly that reason), the new
    baseline re-measured live
    twice — with every earlier pinned
    baseline preserved exactly and TF-129's
    gated set untouched (the hygiene baseline
    is ungated, in the open),
  * TF-017 compliance: no canon access —
    risky markers assembled from parts.

All fixtures are synthetic (TF-017). Proof
grade: FIXTURE — no canon was opened and no
live model was read, never claimed (TF-173).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import benchmarks.data_hygiene as hygiene
import benchmarks.harness as harness
from benchmarks.data_hygiene import (
    ALL_ROW_IDS,
    DATA_HYGIENE_BASELINE_ID,
    DATA_HYGIENE_DOC,
    DATA_HYGIENE_FORMAT,
    HYGIENE_PROBES,
    HYGIENE_ROWS,
    HYGIENE_RULES,
    ORIGIN_CLASSES,
    TOTAL_INPUTS,
    coverage_problems,
    data_hygiene_as_data,
    data_hygiene_problems,
    row_problems,
    run_data_hygiene,
    run_hygiene_probes,
    scan_source_text,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_data_hygiene_baseline,
)
from benchmarks.regression_gate import (
    GATED_BASELINE_IDS,
    UNGATED_BASELINE_IDS,
    run_regression_gate,
)

REPO_ROOT = Path(__file__).parent.parent


def _by_id(rows):
    return {r["id"]: r for r in rows}


# ── The filed inventory ─────────────────────────────────────────────────────


class TestFiledInventory:
    def test_twentyone_inputs_classified_15_0_6(self) -> None:
        assert TOTAL_INPUTS == 21
        assert len(HYGIENE_ROWS) == 21
        ids = [r["id"] for r in HYGIENE_ROWS]
        assert len(ids) == len(set(ids))
        classes = [r["origin_class"] for r in HYGIENE_ROWS]
        assert classes.count("synthetic") == 15
        assert classes.count("licensed") == 0
        assert classes.count("derived") == 6
        assert set(classes) <= set(ORIGIN_CLASSES)
        assert list(ALL_ROW_IDS) == [
            r["id"] for r in HYGIENE_ROWS
            if r["id"] != "data_hygiene_corpus"]

    def test_every_row_binds_and_files_its_proof(self) -> None:
        for row in HYGIENE_ROWS:
            assert row["subject"].strip()
            assert row["proof_kind"].strip()
            if row["bind_kind"] == "module":
                assert row["bind_module"].strip()
                assert row["bind_attribute"].strip()
            else:
                assert row["bind_kind"] == "files"
                assert row["files"]
            if row["origin_class"] in ("synthetic", "licensed"):
                assert row["licence_or_source"].strip()
            if row["origin_class"] == "derived" and \
                    row["id"] != "data_hygiene_corpus":
                assert row["derived_from"]
                for source in row["derived_from"]:
                    assert source in ALL_ROW_IDS

    def test_coverage_equals_the_harness_fixtures(self) -> None:
        filed = [r["harness_fixture"] for r in HYGIENE_ROWS
                 if r["harness_fixture"] is not None]
        assert coverage_problems(
            filed, [f["id"] for f in FIXTURES]) == []
        assert len(filed) == len(FIXTURES) == 13


# ── Live evaluation ─────────────────────────────────────────────────────────


class TestLiveEvaluation:
    def test_full_inventory_passes_clean(self) -> None:
        result = run_data_hygiene()
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.total_inputs == 21
        assert result.synthetic_inputs == 15
        assert result.licensed_inputs == 0
        assert result.derived_inputs == 6
        assert result.clean_inputs == 21
        assert result.failures == []

    def test_bindings_resolve_to_the_real_things(self) -> None:
        rows = _by_id(run_data_hygiene().rows)
        assert rows["persona_fixture_responses"]["problems"] == []
        assert rows["tf067_quant_fixture_corpus"]["problems"] == []
        assert rows["tf043_retrieval_fixture_corpus"]["problems"] == []
        assert rows["hardware_profile_files"]["problems"] == []
        assert rows["tests_fixtures_chaos_corpus"]["problems"] == []

    def test_empty_selection_evaluates_zero_and_reviews(self) -> None:
        empty = run_data_hygiene(row_ids=())
        assert empty.total_inputs == 0
        assert empty.clean_inputs == 0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_partial_selection_evaluates_and_reviews(self) -> None:
        subset = run_data_hygiene(
            row_ids=("persona_fixture_responses",))
        assert subset.total_inputs == 1
        assert subset.clean_inputs == 1, subset.failures
        assert subset.verdict == "REVIEW"
        assert subset.passes is False

    def test_an_unclean_source_fails_its_derived_rows(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = copy.deepcopy([dict(r) for r in HYGIENE_ROWS])
        for row in broken:
            if row["id"] == "tf067_quant_fixture_corpus":
                row["origin_class"] = "canon"
        monkeypatch.setattr(
            hygiene, "HYGIENE_ROWS", tuple(broken))
        result = run_data_hygiene()
        assert result.verdict == "FAIL"
        assert result.passes is False
        rows = _by_id(result.rows)
        assert rows["tf067_quant_fixture_corpus"]["problems"]
        assert any("derived from unclean row" in p
                   for p in rows["quant_panel_corpus"]["problems"])
        assert any("derived from unclean row" in p
                   for p in rows["spot_checks_corpus"]["problems"])


# ── The canon scan ──────────────────────────────────────────────────────────


class TestCanonScan:
    def test_the_benchmark_code_estate_scans_clean(self) -> None:
        benchmarks_dir = REPO_ROOT / "benchmarks"
        for path in sorted(benchmarks_dir.glob("*.py")):
            assert scan_source_text(
                path.read_text(encoding="utf-8")) == [], path.name

    def test_the_scan_names_each_forbidden_shape(self) -> None:
        kr = "knowledge_" + "reference"
        assert "canon_reference_marker" in scan_source_text(kr)
        assert "canon_data_dir_helper" in scan_source_text(
            "get_" + "data_dir()")
        assert "canon_persona_name:Sig" + "rid" in scan_source_text(
            "Sig" + "rid")
        assert scan_source_text("invented synthetic text") == []


# ── The probes ──────────────────────────────────────────────────────────────


class TestProbes:
    def test_seven_probes_all_behave(self) -> None:
        probes = run_hygiene_probes()
        assert len(probes) == len(HYGIENE_PROBES) == 7
        assert all(p["passed"] for p in probes)
        by_id = {p["id"]: p for p in probes}
        for failing in ("canon_origin_row_fails",
                        "licensed_without_licence_fails",
                        "synthetic_without_proof_fails",
                        "unresolvable_binding_fails",
                        "canon_access_source_fails",
                        "unregistered_fixture_fails"):
            assert by_id[failing]["problems"], failing
        assert by_id["clean_synthetic_row_passes"]["problems"] == []

    def test_the_pure_parts_fail_by_name(self) -> None:
        row = dict(HYGIENE_ROWS[0])
        row["origin_class"] = "canon"
        problems = row_problems(row)
        assert any("origin class" in p for p in problems)
        assert coverage_problems(("a",), ("a", "b")) != []
        assert coverage_problems(("a", "b"), ("a", "b")) == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_data_hygiene_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "data_hygiene"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.data_hygiene"
        assert entry[0]["bind_attribute"] == "run_data_hygiene"
        assert entry[0]["fixture"] == "data_hygiene_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence",
                                           "will")
        assert any("D-BENCH-16 FIXED by TF-131" in d
                   for d in entry[0]["docket"])
        fixture = [f for f in FIXTURES
                   if f["id"] == "data_hygiene_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        assert fixture[0]["bind_attribute"] == "HYGIENE_ROWS"
        baseline = [b for b in BASELINES
                    if b["id"] == "data_hygiene_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_inputs"] == 21
        assert baseline[0]["metrics"]["clean_inputs"] == 21
        assert baseline[0]["metrics"]["licensed_inputs"] == 0
        assert baseline[0]["targets"]["clean_inputs"] == {
            "target": 21, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for metric in ("total_inputs", "synthetic_inputs",
                       "licensed_inputs", "derived_inputs",
                       "clean_inputs", "probes_passed",
                       "probes_total"):
            assert ("data_hygiene", metric) in defined
        assert len(BENCHMARKS) == 20
        assert len(FIXTURES) == 13
        assert len(BASELINES) == 13
        assert len(METRIC_DEFINITIONS) == 109
        assert len(HARNESS_RULES) == 13
        assert "benchmark_inputs_are_synthetic_or_licensed_never_canon" \
            in {r["id"] for r in HARNESS_RULES}
        assert harness.harness_problems() == []

    def test_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_data_hygiene_baseline()
        second = measure_data_hygiene_baseline()
        assert first == second
        assert first["total_inputs"] == 21
        assert first["synthetic_inputs"] == 15
        assert first["licensed_inputs"] == 0
        assert first["derived_inputs"] == 6
        assert first["clean_inputs"] == 21
        assert first["probes_passed"] == 7
        assert first["probes_total"] == 7
        assert baseline_problems(
            "data_hygiene_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        assert harness.measure_presence_suite_baseline()[
            "overall_score"] == pytest.approx(0.7777777777777778)
        assert harness.measure_regression_gate_baseline()[
            "gate_failures"] == 0
        assert harness.measure_spot_checks_baseline()[
            "total_transcripts"] == 8
        assert baseline_problems() == []

    def test_the_hygiene_baseline_is_not_gated(self) -> None:
        assert DATA_HYGIENE_BASELINE_ID == \
            "data_hygiene_synthetic_v1"
        assert DATA_HYGIENE_BASELINE_ID not in GATED_BASELINE_IDS
        assert DATA_HYGIENE_BASELINE_ID in UNGATED_BASELINE_IDS
        assert len(GATED_BASELINE_IDS) == 9
        assert run_regression_gate().verdict == "PASS"


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert DATA_HYGIENE_FORMAT == "thoughtforge-data-hygiene/1"
        assert DATA_HYGIENE_DOC == \
            "docs/architecture/BENCHMARK_DATA_HYGIENE.md"
        assert (REPO_ROOT / DATA_HYGIENE_DOC).exists()
        ids = [r["id"] for r in HYGIENE_RULES]
        assert len(ids) == len(set(ids)) == 9
        for needed in (
                "every_benchmark_input_carries_a_hygiene_row",
                "canon_is_never_a_benchmark_input",
                "synthetic_rows_carry_their_proof",
                "pass_needs_the_full_inventory_clean",
                "the_data_hygiene_baseline_is_not_gated_by_tf129"):
            assert needed in ids
        data = data_hygiene_as_data()
        assert data["format"] == DATA_HYGIENE_FORMAT
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_inputs"] == 21
        assert data["measured"]["clean_inputs"] == 21
        assert data["measured"]["probes_passed"] == 7

    def test_the_doc_files_every_inventory_row(self) -> None:
        doc_text = (REPO_ROOT / DATA_HYGIENE_DOC).read_text(
            encoding="utf-8")
        for row in HYGIENE_ROWS:
            assert row["id"] in doc_text

    def test_data_hygiene_checker_clean_on_the_real_estate(
            self) -> None:
        assert data_hygiene_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hygiene, "HYGIENE_ROWS", ())
        problems = data_hygiene_problems()
        assert any("inputs" in p or "HYGIENE" in p
                   for p in problems)


# ── TF-017 canon guard ──────────────────────────────────────────────────────


class TestCanonGuard:
    def test_no_canon_markers_in_this_test_file(self) -> None:
        text = Path(__file__).read_text(encoding="utf-8")
        canon_name = "Sig" + "rid"
        assert canon_name not in text
        data_read = "REPO_ROOT / " + '"data"'
        assert data_read not in text
        assert not re.search(r'["\']data/(knowledge|knowledge\.db)',
                             text)

    def test_the_inventory_world_is_the_estates_filed_fixtures(
            self) -> None:
        data = data_hygiene_as_data()
        assert data["measured"]["licensed_inputs"] == 0
        assert len(data["rows"]) == 21
        assert {r["origin_class"] for r in data["rows"]} == {
            "synthetic", "derived"}
