"""
TF-129 — Regression gates: benchmark deltas
beyond tolerance fail the campaign's slice gate
(thresholds from baseline noise, measured).

The proof lives as verifiable data in
``benchmarks/regression_gate.py`` (``NOISE_ROWS``,
``TOLERANCES``, ``GATE_PROBES``, ``GATE_RULES``,
checked by ``regression_gate_problems()``),
registered in ``benchmarks/harness.py`` with
the pinned baseline
``regression_gate_synthetic_v1``; D-BENCH-14
(pinned baselines existed and
baseline_problems named moves, but no gate
consumed them — nothing classified deltas,
nothing could fail a slice, and tolerances
were uniform / asserted rather than
noise-derived) was fixed by this slice.

Pinned here:

  * the measured noise: the nine gated
    baselines re-measured 5 times in one
    process through the harness's own
    measurers — every spread 0.0, the dict
    metric never changed,
  * the tolerance derivation: per float metric
    max(measured spread, the 1e-9 floor TF-121
    filed); counts and the dict metric exact,
  * the gate: PASS live on the full set
    (9 baselines / 83 metrics / 0 failures),
    partial + empty selections REVIEW, and the
    seven filed probes behaving against the
    pure evaluator (regression / improvement /
    count change / unmeasurable),
  * the checker clean, the harness registration
    (17 / 10 / 10 / 89, rules 10), the new gate
    baseline re-measured live twice — with
    every earlier pinned baseline preserved
    exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof
grade: FIXTURE — no live model is gated, never
claimed (TF-173).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import benchmarks.regression_gate as gate
import benchmarks.harness as harness
from benchmarks.regression_gate import (
    GATE_BASELINE_ID,
    GATE_PROBES,
    GATE_RULES,
    GATED_BASELINE_IDS,
    NOISE_REPEATS,
    NOISE_ROWS,
    REGRESSION_GATE_DOC,
    REGRESSION_GATE_FORMAT,
    TOLERANCES,
    TOLERANCE_FLOOR,
    TOTAL_PINNED_METRICS,
    compute_noise_rows,
    direction_for,
    evaluate_gate,
    regression_gate_as_data,
    regression_gate_problems,
    run_gate_probes,
    run_regression_gate,
    tolerances_for,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_regression_gate_baseline,
)

REPO_ROOT = Path(__file__).parent.parent


def _pins() -> dict[str, dict]:
    return {
        b["id"]: copy.deepcopy(dict(b["metrics"]))
        for b in BASELINES if b["id"] in GATED_BASELINE_IDS}


def _row(evaluation, baseline, metric):
    return [r for r in evaluation["rows"]
            if r["baseline"] == baseline
            and r["metric"] == metric][0]


# ── The measured noise ──────────────────────────────────────────────────────


class TestMeasuredNoise:
    def test_noise_rows_cover_the_nine_gated_baselines(self) -> None:
        assert [r["baseline"] for r in NOISE_ROWS] == \
            list(GATED_BASELINE_IDS)
        assert len(GATED_BASELINE_IDS) == 9
        assert GATE_BASELINE_ID == \
            "regression_gate_synthetic_v1"
        assert GATE_BASELINE_ID not in GATED_BASELINE_IDS
        harness_ids = [b["id"] for b in BASELINES
                       if b["id"] not in gate.UNGATED_BASELINE_IDS]
        assert list(GATED_BASELINE_IDS) == harness_ids
        for row in NOISE_ROWS:
            assert row["repeats"] == NOISE_REPEATS == 5
            baseline = [b for b in BASELINES
                        if b["id"] == row["baseline"]][0]
            assert row["metrics_measured"] == \
                len(baseline["metrics"])
            assert row["benchmark"] == baseline["benchmark"]
        assert TOTAL_PINNED_METRICS == 83
        assert sum(r["metrics_measured"] for r in NOISE_ROWS) == 83

    def test_every_measured_spread_is_zero_on_this_tree(self) -> None:
        for row in NOISE_ROWS:
            assert row["max_abs_spread"] == 0.0
            assert row["metrics_with_nonzero_spread"] == []
            for metric, spread in row["spreads"].items():
                if metric == "quality_tiers":
                    assert spread is None  # non-numeric: change-or-not
                else:
                    assert spread == 0.0

    def test_noise_rows_recompute_live(self) -> None:
        live = compute_noise_rows()
        assert [dict(r) for r in live] == \
            [dict(r) for r in NOISE_ROWS]


# ── Tolerance derivation ────────────────────────────────────────────────────


class TestToleranceDerivation:
    def test_tolerances_derive_from_the_filed_noise(self) -> None:
        assert tolerances_for(NOISE_ROWS) == TOLERANCES
        assert set(TOLERANCES) == set(GATED_BASELINE_IDS)
        for baseline_id, per_metric in TOLERANCES.items():
            baseline = [b for b in BASELINES
                        if b["id"] == baseline_id][0]
            assert set(per_metric) == set(baseline["metrics"])

    def test_float_tolerances_file_the_floor_because_noise_is_zero(
            self) -> None:
        assert TOLERANCE_FLOOR == 1e-9
        for per_metric in TOLERANCES.values():
            for row in per_metric.values():
                if not row["exact"]:
                    # max(measured spread 0.0, the floor)
                    assert row["tolerance"] == pytest.approx(1e-9)

    def test_counts_and_the_dict_metric_are_exact(self) -> None:
        persona = TOLERANCES["persona_consistency_synthetic_v1"]
        assert persona["total_turns"] == {
            "tolerance": 0.0, "exact": True}
        assert persona["generic_phrase_hits"]["exact"] is True
        profile = TOLERANCES["profile_benchmark_empty_store_v1"]
        assert profile["quality_tiers"] == {
            "tolerance": 0.0, "exact": True}
        assert profile["avg_response_words"]["exact"] is True
        assert profile["avg_confidence"]["exact"] is False

    def test_noise_can_only_widen_a_tolerance_to_what_was_observed(
            self) -> None:
        widened = copy.deepcopy([dict(r) for r in NOISE_ROWS])
        target = widened[0]
        target["spreads"]["consistency_score"] = 0.25
        target["max_abs_spread"] = 0.25
        target["metrics_with_nonzero_spread"] = ["consistency_score"]
        derived = tolerances_for(widened)
        row = derived["persona_consistency_synthetic_v1"][
            "consistency_score"]
        assert row["tolerance"] == pytest.approx(0.25)
        assert row["exact"] is False
        # An untouched metric keeps the floor — nothing invented.
        other = derived["persona_suite_synthetic_v1"]["overall_score"]
        assert other["tolerance"] == pytest.approx(1e-9)

    def test_direction_table(self) -> None:
        assert direction_for("overall_score", 0.8) == \
            "higher_is_better"
        assert direction_for("q4_k_m_loss", 0.2) == \
            "lower_is_better"
        assert direction_for("total_cases", 36) == "count"
        assert direction_for("generic_phrase_hits", 7) == "count"
        assert direction_for("avg_response_words", 9.0) == "count"
        assert direction_for("quality_tiers", {"weak": 10}) == \
            "exact"
        assert direction_for("delta", 0.3) == "higher_is_better"


# ── The pure evaluator + the probes ─────────────────────────────────────────


class TestEvaluatorAndProbes:
    def test_the_pins_evaluate_pass_shaped(self) -> None:
        evaluation = evaluate_gate(_pins())
        assert evaluation["outcome"] == "PASS"
        assert evaluation["failures"] == 0
        assert evaluation["metrics_compared"] == 83
        assert all(r["classification"] == "within_tolerance"
                   for r in evaluation["rows"])

    def test_a_dropped_score_is_a_regression_failure(self) -> None:
        measured = _pins()
        measured["persona_suite_synthetic_v1"]["overall_score"] -= 0.1
        evaluation = evaluate_gate(measured)
        assert evaluation["outcome"] == "FAIL"
        row = _row(evaluation, "persona_suite_synthetic_v1",
                   "overall_score")
        assert row["classification"] == "regression"
        assert row["beyond_tolerance"] is True
        assert row["delta"] == pytest.approx(-0.1)

    def test_a_raised_score_is_an_improvement_failure(self) -> None:
        measured = _pins()
        measured["persona_suite_synthetic_v1"]["overall_score"] += 0.1
        evaluation = evaluate_gate(measured)
        assert evaluation["outcome"] == "FAIL"
        assert _row(evaluation, "persona_suite_synthetic_v1",
                    "overall_score")["classification"] == \
            "improvement"

    def test_a_changed_count_and_dict_file_count_changes(self) -> None:
        measured = _pins()
        measured["persona_suite_synthetic_v1"]["total_turns"] += 1
        measured["profile_benchmark_empty_store_v1"][
            "quality_tiers"] = {"weak": 9, "moderate": 1}
        evaluation = evaluate_gate(measured)
        assert evaluation["outcome"] == "FAIL"
        assert evaluation["failures"] == 2
        assert _row(evaluation, "persona_suite_synthetic_v1",
                    "total_turns")["classification"] == "count_change"
        assert _row(evaluation,
                    "profile_benchmark_empty_store_v1",
                    "quality_tiers")["classification"] == \
            "count_change"

    def test_a_dropped_loss_is_an_improvement_failure(self) -> None:
        measured = _pins()
        measured["quant_panel_synthetic_v1"]["q4_k_m_loss"] -= 0.1
        evaluation = evaluate_gate(measured)
        assert _row(evaluation, "quant_panel_synthetic_v1",
                    "q4_k_m_loss")["classification"] == "improvement"
        measured["quant_panel_synthetic_v1"]["q4_k_m_loss"] += 0.2
        evaluation = evaluate_gate(measured)
        assert _row(evaluation, "quant_panel_synthetic_v1",
                    "q4_k_m_loss")["classification"] == "regression"

    def test_a_move_inside_tolerance_passes(self) -> None:
        measured = _pins()
        measured["persona_consistency_synthetic_v1"][
            "consistency_score"] += 5e-10
        evaluation = evaluate_gate(measured)
        assert evaluation["outcome"] == "PASS"
        assert _row(evaluation,
                    "persona_consistency_synthetic_v1",
                    "consistency_score")["classification"] == \
            "within_tolerance"

    def test_a_missing_baseline_is_unmeasurable_and_fails(
            self) -> None:
        measured = _pins()
        del measured["memory_suite_synthetic_v1"]
        evaluation = evaluate_gate(measured)
        assert evaluation["outcome"] == "FAIL"
        rows = [r for r in evaluation["rows"]
                if r["baseline"] == "memory_suite_synthetic_v1"]
        assert rows and all(
            r["classification"] == "unmeasurable" for r in rows)

    def test_all_filed_probes_behave(self) -> None:
        assert len(GATE_PROBES) == 7
        results = run_gate_probes()
        assert len(results) == 7
        assert all(r["passed"] for r in results)
        by_id = {r["id"]: r for r in results}
        assert by_id["score_dropped_is_a_regression_failure"][
            "classification"] == "regression"
        assert by_id["score_raised_is_an_improvement_failure"][
            "classification"] == "improvement"
        assert by_id["count_changed_is_a_count_change_failure"][
            "classification"] == "count_change"
        assert by_id[
            "a_move_inside_tolerance_passes_the_evaluation"][
                "outcome"] == "PASS"
        assert by_id[
            "a_missing_baseline_is_unmeasurable_and_fails"][
                "classification"] == "unmeasurable"


# ── The gate run + its pinned baseline ──────────────────────────────────────


class TestGateRun:
    def test_full_gate_passes_live(self) -> None:
        result = run_regression_gate()
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.baselines_gated == 9
        assert result.metrics_compared == 83
        assert result.failures == 0
        assert result.overall_score == pytest.approx(1.0)
        assert all(r["classification"] == "within_tolerance"
                   for r in result.rows)

    def test_empty_selection_gates_zero_and_reviews(self) -> None:
        empty = run_regression_gate(baseline_ids=())
        assert empty.baselines_gated == 0
        assert empty.metrics_compared == 0
        assert empty.overall_score == 0.0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_partial_selection_measures_and_reviews(self) -> None:
        subset = run_regression_gate(
            baseline_ids=("persona_consistency_synthetic_v1",))
        assert subset.baselines_gated == 1
        assert subset.metrics_compared == 7
        assert subset.failures == 0
        assert subset.verdict == "REVIEW"
        assert subset.passes is False

    def test_gate_baseline_remeasures_live_twice_identical(
            self) -> None:
        first = measure_regression_gate_baseline()
        second = measure_regression_gate_baseline()
        assert first == second
        assert first["baselines_gated"] == 9
        assert first["metrics_compared"] == 83
        assert first["max_noise_spread"] == 0.0
        assert first["gate_failures"] == 0
        assert first["probes_passed"] == first["probes_total"] == 7
        assert baseline_problems(
            "regression_gate_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        depth = harness.measure_depth_suite_baseline()
        assert depth["overall_on_score"] == \
            pytest.approx(0.8055555555555557)
        assert harness.measure_presence_suite_baseline()[
            "overall_score"] == pytest.approx(0.7777777777777778)
        assert harness.measure_honesty_panel_baseline()[
            "mid_delta"] == pytest.approx(0.24004629629629637)
        assert harness.measure_quant_panel_baseline()[
            "q4_k_m_score"] == pytest.approx(0.8125)
        assert harness.measure_profile_panel_baseline()[
            "desktop_cpu_suite_score"] == \
            pytest.approx(0.8085648148148149)
        assert harness.measure_profile_baseline()[
            "avg_confidence"] == pytest.approx(0.408)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_gate_is_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "regression_gate"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "gate"
        assert entry[0]["bind_module"] == \
            "benchmarks.regression_gate"
        assert entry[0]["bind_attribute"] == "run_regression_gate"
        assert entry[0]["fixture"] == "regression_gate_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence",
                                           "will")
        assert any("D-BENCH-14 FIXED by TF-129" in d
                   for d in entry[0]["docket"])
        fixture = [f for f in FIXTURES
                   if f["id"] == "regression_gate_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        assert fixture[0]["bind_attribute"] == "NOISE_ROWS"
        baseline = [b for b in BASELINES
                    if b["id"] == "regression_gate_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["baselines_gated"] == 9
        assert baseline[0]["metrics"]["metrics_compared"] == 83
        assert baseline[0]["metrics"]["gate_failures"] == 0
        assert baseline[0]["targets"]["gate_failures"] == {
            "target": 0, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for metric in ("baselines_gated", "metrics_compared",
                       "max_noise_spread", "gate_failures",
                       "probes_passed", "probes_total"):
            assert ("regression_gate", metric) in defined
        assert len(BENCHMARKS) == 21
        assert len(FIXTURES) == 13
        assert len(BASELINES) == 14
        assert len(METRIC_DEFINITIONS) == 119
        assert len(HARNESS_RULES) == 14
        assert "regression_gates_fail_on_moved_baselines" in {
            r["id"] for r in HARNESS_RULES}
        assert harness.harness_problems() == []


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert REGRESSION_GATE_FORMAT == \
            "thoughtforge-regression-gate/1"
        assert REGRESSION_GATE_DOC == \
            "docs/architecture/REGRESSION_GATES.md"
        assert (REPO_ROOT / REGRESSION_GATE_DOC).exists()
        ids = [r["id"] for r in GATE_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in (
                "noise_is_measured_never_assumed",
                "tolerances_derive_from_noise_over_a_filed_floor",
                "any_beyond_tolerance_move_fails_the_gate",
                "probes_prove_the_gate_can_fail",
                "a_checker_a_slice_runs_not_a_ci_wiring"):
            assert needed in ids
        data = regression_gate_as_data()
        assert data["format"] == REGRESSION_GATE_FORMAT
        assert data["gated_baselines"] == list(GATED_BASELINE_IDS)
        assert data["noise_repeats"] == 5
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["baselines_gated"] == 9
        assert data["measured"]["metrics_compared"] == 83
        assert data["measured"]["gate_failures"] == 0
        assert data["measured"]["max_noise_spread"] == 0.0
        assert data["measured"]["probes_passed"] == 7
        assert data["measured"]["probes_total"] == 7

    def test_regression_gate_checker_clean_on_the_real_estate(
            self) -> None:
        assert regression_gate_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gate, "NOISE_ROWS", ())
        problems = regression_gate_problems()
        assert any("NOISE_ROWS" in p or "TOLERANCES" in p
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

    def test_gate_world_is_the_harness_pins_and_synthetic_probes(
            self) -> None:
        data = regression_gate_as_data()
        assert data["gated_baselines"] == list(GATED_BASELINE_IDS)
        assert all(p["passed"] for p in data["probes"])
