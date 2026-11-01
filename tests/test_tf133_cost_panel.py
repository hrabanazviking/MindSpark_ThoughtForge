"""
TF-133 — Cost panel: tokens + time per suite
per profile — sharpness per watt, the
edge-hardware point of the project.

The proof lives as verifiable data in
``benchmarks/cost_panel.py`` (``PANEL_ROWS``,
``FINDING``, ``PANEL_PROBES``,
``COST_PANEL_RULES``, checked by
``cost_panel_problems()``), registered in
``benchmarks/harness.py`` with the pinned
baseline ``cost_panel_synthetic_v1``;
D-BENCH-18 (no cost panel existed — tokens /
time per suite per profile measured by
nothing) was fixed by this slice in the
measuring sense, and D-BENCH-7's consumption
share was fixed here (the panel consumes
``PerfTracker``).

Pinned here:

  * the deterministic rows: 24 cells (4 suites
    x 6 shipped profiles), token budgets from
    the real runners' unit counts x the
    profiles' filed max_response_tokens
    (total 766800 over 1296 units), sharpness
    derived only from the real suite scores,
    pi_zero sharpest / server_gpu least sharp,
    pi_zero's depth cell refused_by_budget,
  * the watt honesty call: no profile declares
    a power figure (scanned live), every row
    files power_watts None +
    not_proven_not_measured, and a claimed
    watt fails the pure checker,
  * time: repeat-measured live through
    PerfTracker (5 repeats per cell, judged by
    shape, never pinned — no time metric in
    the baseline),
  * the verdict shape: full matrix PASS,
    partial and empty selections REVIEW (no
    vacuous pass),
  * the probes: 7/7 against the pure checker
    parts, proving the checker can fail,
  * the checker clean, the harness
    registration (21 / 13 / 14 / 119, rules
    14), the new baseline re-measured live
    twice — with every earlier pinned baseline
    preserved exactly (no exception: the
    hygiene inventory does NOT move, because
    the panel adds no benchmark input) and
    TF-129's gated set untouched (the cost
    baseline is ungated, in the open),
  * TF-017 compliance: no canon access —
    risky markers assembled from parts.

All fixtures are synthetic (TF-017). Proof
grade: FIXTURE — no canon was opened, no live
model was read and no watt was measured,
never claimed (TF-173).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import benchmarks.cost_panel as panel
import benchmarks.harness as harness
from benchmarks.cost_panel import (
    COST_PANEL_BASELINE_ID,
    COST_PANEL_DOC,
    COST_PANEL_FORMAT,
    COST_PANEL_RULES,
    FINDING,
    PANEL_PROBES,
    PANEL_PROFILES,
    PANEL_ROWS,
    PANEL_SUITES,
    POWER_STATUS,
    TIME_REPEATS,
    TOTAL_CELLS,
    build_cell,
    cell_problems,
    compute_cost_rows,
    cost_panel_as_data,
    cost_panel_problems,
    measure_timing,
    render_cost_panel_markdown,
    run_cost_panel,
    run_panel_probes,
    sharpness_for,
    timing_problems,
)
from benchmarks.data_hygiene import run_data_hygiene
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_cost_panel_baseline,
)
from benchmarks.regression_gate import (
    GATED_BASELINE_IDS,
    UNGATED_BASELINE_IDS,
    run_regression_gate,
)

REPO_ROOT = Path(__file__).parent.parent


def _cell(profile: str, suite: str):
    return [r for r in PANEL_ROWS
            if r["profile"] == profile
            and r["suite"] == suite][0]


# ── The deterministic rows ──────────────────────────────────────────────────


class TestDeterministicRows:
    def test_twenty_four_cells_over_six_profiles_four_suites(
            self) -> None:
        assert TOTAL_CELLS == 24
        assert len(PANEL_ROWS) == 24
        assert list(PANEL_PROFILES) == [
            "desktop_cpu", "desktop_gpu", "phone_low",
            "pi_5", "pi_zero", "server_gpu"]
        assert list(PANEL_SUITES) == [
            "persona", "memory", "depth", "presence"]
        ids = [f"{r['profile']}:{r['suite']}" for r in PANEL_ROWS]
        assert len(ids) == len(set(ids)) == 24

    def test_token_budgets_are_units_times_filed_budgets(
            self) -> None:
        for row in PANEL_ROWS:
            assert row["token_budget"] == \
                row["units"] * row["max_response_tokens"]
            assert cell_problems(row) == []
        assert sum(r["units"] for r in PANEL_ROWS) == 1296
        assert sum(r["token_budget"] for r in PANEL_ROWS) == 766800
        assert _cell("pi_zero", "persona")["token_budget"] == 12960
        assert _cell("server_gpu", "persona")["token_budget"] == 216000
        assert _cell("desktop_cpu", "memory")["units"] == 36
        assert _cell("desktop_cpu", "persona")["units"] == 108

    def test_sharpness_derives_only_from_real_scores(self) -> None:
        for row in PANEL_ROWS:
            assert row["sharpness_per_1000_tokens"] == \
                pytest.approx(sharpness_for(
                    row["suite_score"], row["token_budget"]))
        assert sharpness_for(0.8, 0) == 0.0
        assert FINDING["sharpest_profile"] == "pi_zero"
        assert FINDING["least_sharp_profile"] == "server_gpu"
        assert FINDING["total_token_budget"] == 766800
        assert FINDING["suite_scores_vary_by_profile"] is True
        assert FINDING[
            "suite_scores_vary_without_the_refused_cell"] is False

    def test_the_depth_refusal_is_carried_from_tf128(self) -> None:
        refused = [r for r in PANEL_ROWS
                   if r["status"] == "refused_by_budget"]
        assert len(refused) == 1
        assert refused[0]["profile"] == "pi_zero"
        assert refused[0]["suite"] == "depth"
        assert refused[0]["suite_score"] == \
            pytest.approx(0.47222222222222227)
        assert FINDING["refused_cells"] == [
            ["pi_zero", "depth", "max_refinement_passes=0"]]
        built = build_cell("pi_zero", "depth", 36, 120, 0,
                           0.47222222222222227, "refused_by_budget")
        assert built == refused[0]


# ── The watt honesty call ───────────────────────────────────────────────────


class TestWattHonesty:
    def test_no_profile_declares_a_power_figure(self) -> None:
        assert panel._profile_power_keys() == []
        assert FINDING["declared_power_profiles"] == []
        assert FINDING["sharpness_per_watt"] == POWER_STATUS

    def test_every_row_files_not_proven_power(self) -> None:
        assert POWER_STATUS == "not_proven_not_measured"
        for row in PANEL_ROWS:
            assert row["power_watts"] is None
            assert row["power_status"] == POWER_STATUS
        assert set(FINDING["power_status_by_profile"].values()) == \
            {POWER_STATUS}

    def test_a_claimed_watt_fails_the_pure_checker(self) -> None:
        cell = dict(PANEL_ROWS[0])
        cell["power_watts"] = 5.0
        assert any("power_watts" in p for p in cell_problems(cell))
        cell = dict(PANEL_ROWS[0])
        cell["power_status"] = "measured"
        assert any("power status" in p for p in cell_problems(cell))


# ── Live evaluation + timing ────────────────────────────────────────────────


class TestLiveEvaluation:
    def test_full_panel_passes_with_filed_totals(self) -> None:
        result = run_cost_panel()
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.total_cells == 24
        assert result.total_token_budget == 766800
        assert result.mean_sharpness == pytest.approx(
            0.06650563779657573)
        assert result.overall_score == pytest.approx(
            0.794675925925926)
        assert result.rows_reproduce is True
        assert result.timings_well_shaped is True

    def test_every_cell_timing_is_well_shaped(self) -> None:
        result = run_cost_panel()
        for cell in result.cells:
            assert timing_problems(cell["timing"]) == []
            assert cell["timing"]["repeats"] == TIME_REPEATS == 5
            assert cell["timing"]["median_ms"] > 0

    def test_timing_is_measured_through_the_perf_tracker(
            self) -> None:
        timing = measure_timing("desktop_cpu", "depth")
        assert timing_problems(timing) == []
        assert timing["repeats"] == 5
        assert timing["min_ms"] <= timing["median_ms"] \
            <= timing["max_ms"]

    def test_empty_selection_scores_zero_and_reviews(self) -> None:
        empty = run_cost_panel(profiles=())
        assert empty.total_cells == 0
        assert empty.total_token_budget == 0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_partial_selection_measures_and_reviews(self) -> None:
        subset = run_cost_panel(profiles=("desktop_cpu",))
        assert subset.total_cells == 4
        assert subset.total_token_budget == 86400
        assert subset.verdict == "REVIEW"
        assert subset.passes is False


# ── The probes ──────────────────────────────────────────────────────────────


class TestProbes:
    def test_seven_probes_all_behave(self) -> None:
        probes = run_panel_probes()
        assert len(probes) == len(PANEL_PROBES) == 7
        assert all(p["passed"] for p in probes)
        by_id = {p["id"]: p for p in probes}
        assert by_id["wrong_token_budget_is_named"]["problems"]
        assert by_id["wrong_sharpness_is_named"]["problems"]
        assert by_id["invented_watts_are_named"]["problems"]
        assert by_id["bad_timing_shape_is_named"]["problems"]
        assert by_id["clean_filed_cell_passes"]["problems"] == []

    def test_the_pure_checker_fails_by_shape(self) -> None:
        cell = dict(PANEL_ROWS[0])
        cell["token_budget"] += 1
        assert cell_problems(cell) != []
        bad_timing = {"repeats": 5, "mean_ms": 1.0,
                      "median_ms": 3.0, "min_ms": 1.0,
                      "max_ms": 2.0}
        assert timing_problems(bad_timing) != []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_cost_panel_registered_without_a_new_fixture(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "cost_panel"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.cost_panel"
        assert entry[0]["bind_attribute"] == "run_cost_panel"
        # No new fixture: the panel adds no benchmark
        # input (TF-131 coupling, filed in the open).
        assert entry[0]["fixture"] is None
        assert len(FIXTURES) == 13
        assert any("D-BENCH-18 FIXED by TF-133" in d
                   for d in entry[0]["docket"])
        assert any("D-BENCH-7 FIXED by TF-133" in d
                   for d in entry[0]["docket"])
        tracker = [b for b in BENCHMARKS
                   if b["id"] == "perf_tracker"]
        assert any("D-BENCH-7 FIXED by TF-133" in d
                   for d in tracker[0]["docket"])
        baseline = [b for b in BASELINES
                    if b["id"] == "cost_panel_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cells"] == 24
        assert baseline[0]["metrics"]["total_token_budget"] == 766800
        assert baseline[0]["targets"][
            "pi_zero_mean_sharpness_per_1000_tokens"] == {
                "target": 0.13526306012802927, "met": True}
        # Time and watts are excluded, never pinned.
        assert not any("latency" in m or "time" in m
                       for m in baseline[0]["metrics"])
        assert "cell_wall_time_ms" in \
            baseline[0]["excluded_metrics"]
        assert "power_watts" in baseline[0]["excluded_metrics"]

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for metric in ("total_cells", "total_units",
                       "total_token_budget",
                       "pi_zero_token_budget",
                       "server_gpu_token_budget",
                       "mean_sharpness_per_1000_tokens",
                       "pi_zero_mean_sharpness_per_1000_tokens",
                       "server_gpu_mean_sharpness_per_1000_tokens",
                       "probes_passed", "probes_total"):
            assert ("cost_panel", metric) in defined
        assert len(BENCHMARKS) == 23
        assert len(FIXTURES) == 13
        assert len(BASELINES) == 16
        assert len(METRIC_DEFINITIONS) == 132
        assert len(HARNESS_RULES) == 16
        assert "the_cost_panel_files_tokens_time_and_no_watts" \
            in {r["id"] for r in HARNESS_RULES}
        assert harness.harness_problems() == []

    def test_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_cost_panel_baseline()
        second = measure_cost_panel_baseline()
        assert first == second
        assert first["total_cells"] == 24
        assert first["total_units"] == 1296
        assert first["total_token_budget"] == 766800
        assert first["pi_zero_token_budget"] == 25920
        assert first["server_gpu_token_budget"] == 432000
        assert first["pi_zero_mean_sharpness_per_1000_tokens"] == \
            pytest.approx(0.13526306012802927)
        assert first["probes_passed"] == 7
        assert first["probes_total"] == 7
        assert baseline_problems("cost_panel_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_profile_panel_baseline()[
            "total_cells"] == 30
        assert harness.measure_wyrd_panel_baseline()[
            "overall_score"] == pytest.approx(0.5555555555555556)
        assert baseline_problems() == []

    def test_the_hygiene_inventory_does_not_move(self) -> None:
        # TF-133's filed exception-that-isn't: no new
        # benchmark input, so TF-131's inventory and
        # its pinned baseline stand exactly as filed.
        result = run_data_hygiene()
        assert result.verdict == "PASS"
        assert result.total_inputs == 21
        assert result.synthetic_inputs == 15
        assert result.derived_inputs == 6
        assert baseline_problems(
            "data_hygiene_synthetic_v1") == []

    def test_the_cost_baseline_is_not_gated(self) -> None:
        assert COST_PANEL_BASELINE_ID == "cost_panel_synthetic_v1"
        assert COST_PANEL_BASELINE_ID not in GATED_BASELINE_IDS
        assert COST_PANEL_BASELINE_ID in UNGATED_BASELINE_IDS
        assert len(GATED_BASELINE_IDS) == 9
        assert run_regression_gate().verdict == "PASS"


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert COST_PANEL_FORMAT == "thoughtforge-cost-panel/1"
        assert COST_PANEL_DOC == "docs/architecture/COST_PANEL.md"
        assert (REPO_ROOT / COST_PANEL_DOC).exists()
        ids = [r["id"] for r in COST_PANEL_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in (
                "tokens_are_budgets_from_real_runners_and_profiles",
                "the_perf_tracker_is_consumed_here",
                "watts_are_not_proven_and_never_invented",
                "no_new_benchmark_input_was_added",
                "the_cost_panel_baseline_is_not_gated_by_tf129"):
            assert needed in ids
        data = cost_panel_as_data()
        assert data["format"] == COST_PANEL_FORMAT
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_cells"] == 24
        assert data["measured"]["probes_passed"] == 7
        assert data["finding"]["sharpness_per_watt"] == \
            "not_proven_not_measured"

    def test_the_doc_files_the_rendered_panel(self) -> None:
        doc_text = (REPO_ROOT / COST_PANEL_DOC).read_text(
            encoding="utf-8")
        assert render_cost_panel_markdown().strip() in doc_text
        for row in PANEL_ROWS:
            assert row["profile"] in doc_text
            assert str(row["token_budget"]) in doc_text

    def test_cost_panel_checker_clean_on_the_real_estate(
            self) -> None:
        assert cost_panel_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = copy.deepcopy([dict(r) for r in PANEL_ROWS])
        broken[0]["token_budget"] = 1
        monkeypatch.setattr(panel, "PANEL_ROWS", tuple(broken))
        problems = cost_panel_problems()
        assert problems != []


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

    def test_the_panel_consumes_only_filed_inputs(self) -> None:
        data = cost_panel_as_data()
        assert data["profiles"] == list(PANEL_PROFILES)
        assert data["suites"] == list(PANEL_SUITES)
        assert len(data["rows"]) == 24
