"""
TF-128 — Profile panels: suites per hardware
profile with the profile's budgets enforced
during the run.

The proof lives as verifiable data in
``benchmarks/profile_panel.py`` (``PANEL_ROWS``,
``FINDING``, ``PANEL_RULES``, checked by
``profile_panel_problems()``), registered in
``benchmarks/harness.py`` with the pinned
baseline ``profile_panel_synthetic_v1``;
D-BENCH-1 (the profile never applied — every
profile measured the same ephemeral empty-store
core) was fixed by this slice in the measuring
sense: its validation share was TF-121's, its
application share is discharged here.

Pinned here:

  * the measured panel: six shipped profiles x
    five filed cells (30, one refused by budget),
    every row's budgets read live from the
    profile JSONs, TF-069's boot + cap rows
    reproduced (server_gpu exceeds its ceiling;
    phone_low / pi_zero unprovable),
  * the enforcement: per profile, the benchmark
    runs on a core whose resolved tokens / passes
    + lifecycle caps equal the profile's filed
    budgets (budgets_applied on all six),
  * the finding from the numbers: no score
    varies by profile except through pi_zero's
    refused depth ON road (its 0-pass refinement
    budget; its depth score is its OFF score),
  * the checker clean, the harness registration
    (16 / 9 / 9 / 83), the new panel baseline
    re-measured live twice — with every earlier
    pinned baseline preserved exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof
grade: FIXTURE/SIMULATION — no live model, no
real per-profile hardware, never claimed
(TF-082 / TF-173).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.profile_panel as panel
import benchmarks.harness as harness
from benchmarks.profile_panel import (
    CELLS_PER_PROFILE,
    FINDING,
    PANEL_PROFILES,
    PANEL_ROWS,
    PANEL_RULES,
    PANEL_SUITES,
    PROFILE_PANEL_DOC,
    PROFILE_PANEL_FORMAT,
    TOTAL_CELLS,
    compute_panel_rows,
    filed_budgets,
    finding_for,
    measure_profile_row,
    profile_panel_as_data,
    profile_panel_problems,
    run_profile_panel,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_profile_panel_baseline,
)
from thoughtforge.inference.profile_runner import (
    BOOT_VERDICTS,
    RAM_ROWS,
    check_model_fit,
)
from thoughtforge.utils.config import load_hardware_profile

REPO_ROOT = Path(__file__).parent.parent


def _row(profile):
    return [r for r in PANEL_ROWS if r["profile"] == profile][0]


# ── The measured panel + the finding ────────────────────────────────────────


class TestMeasuredPanel:
    def test_panel_covers_the_shipped_six_with_30_cells(self) -> None:
        assert [r["profile"] for r in PANEL_ROWS] == \
            list(PANEL_PROFILES)
        assert PANEL_PROFILES == (
            "desktop_cpu", "desktop_gpu", "phone_low", "pi_5",
            "pi_zero", "server_gpu")
        assert PANEL_SUITES == ("persona", "memory", "depth",
                                "presence")
        assert CELLS_PER_PROFILE == 5
        assert TOTAL_CELLS == 30

    def test_budgets_are_the_profile_jsons_own_values(self) -> None:
        for row in PANEL_ROWS:
            profile = load_hardware_profile(row["profile"])
            assert row["ram_gb"] == profile["hardware"]["ram_gb"]
            assert row["max_response_tokens"] == \
                profile["generation"]["max_response_tokens"]
            assert row["max_refinement_passes"] == \
                profile["generation"]["max_refinement_passes"]
            assert row["bundle_facts"] == \
                profile["memory"]["max_memory_bundle_facts"]
            assert filed_budgets(row["profile"])["ceiling_gb"] == \
                row["ceiling_gb"]
        assert _row("pi_zero")["max_response_tokens"] == 120
        assert _row("server_gpu")["max_response_tokens"] == 2000
        assert _row("pi_zero")["bundle_facts"] == 3
        assert _row("server_gpu")["bundle_facts"] == 30

    def test_tf069_boot_and_cap_rows_reproduce(self) -> None:
        for row in PANEL_ROWS:
            assert row["boot_verdict"] == BOOT_VERDICTS[row["profile"]]
            assert row["cap_status"] == RAM_ROWS[row["profile"]]
            cap = check_model_fit(
                row["profile"], row["max_params_b"],
                row["quantization"])
            assert row["cap_status"] == cap.status
            assert row["cap_accounted_gb"] == cap.accounted_gb
        assert _row("desktop_gpu")["boot_verdict"] == "boots"
        assert _row("server_gpu")["cap_status"] == \
            "exceeds_ceiling"
        assert _row("server_gpu")["cap_accounted_gb"] == 140.0
        assert _row("server_gpu")["ceiling_gb"] == 88.0
        for pid in ("phone_low", "pi_zero"):
            assert _row(pid)["cap_status"] == "unprovable"
            assert _row(pid)["cap_accounted_gb"] is None
            assert _row(pid)["cap_basis"] == "uncalibrated"

    def test_suite_scores_reproduce_the_pinned_suite_baselines(
            self) -> None:
        for row in PANEL_ROWS:
            assert row["persona_score"] == \
                pytest.approx(0.8453703703703704)
            assert row["memory_score"] == \
                pytest.approx(0.8055555555555557)
            assert row["presence_score"] == \
                pytest.approx(0.7777777777777778)
            assert row["depth_off_score"] == \
                pytest.approx(0.47222222222222227)
        for pid in ("desktop_cpu", "desktop_gpu", "phone_low",
                    "pi_5", "server_gpu"):
            assert _row(pid)["depth_on_score"] == \
                pytest.approx(0.8055555555555557)
            assert _row(pid)["suite_score"] == \
                pytest.approx(0.8085648148148149)

    def test_the_finding_from_the_numbers(self) -> None:
        assert FINDING["budgets_applied_on_all_profiles"] is True
        assert FINDING["bench_scores_vary_by_profile"] is False
        assert FINDING["suite_scores_vary_by_profile"] is True
        assert FINDING[
            "suite_scores_vary_without_the_refused_cell"] is False
        assert FINDING["ceiling_gap_profiles"] == [
            "phone_low", "pi_zero", "server_gpu"]
        assert FINDING["widest_profile"] == "server_gpu"
        assert FINDING["narrowest_profile"] == "pi_zero"
        measured = {r["suite_score"] for r in PANEL_ROWS
                    if r["depth_on_status"] == "measured"}
        assert len(measured) == 1


# ── Enforcement during the run ──────────────────────────────────────────────


class TestEnforcement:
    def test_budgets_applied_on_every_profile(self) -> None:
        for row in PANEL_ROWS:
            assert row["budgets_applied"] is True
            assert row["core_caps_match_budget"] is True
            assert row["core_max_response_tokens"] == \
                row["max_response_tokens"]
            assert row["core_max_refinement_passes"] == \
                row["max_refinement_passes"]

    def test_benchmark_cell_files_the_empty_store_shape(self) -> None:
        for row in PANEL_ROWS:
            assert row["bench_successful_turns"] == 10
            assert row["bench_total_turns"] == 10
            assert row["bench_citation_accuracy"] == 0.0
            assert row["bench_enforcement_pass_rate"] == 0.0
            assert row["bench_avg_response_words"] == 9
            assert row["bench_avg_confidence"] == \
                pytest.approx(0.408)

    def test_pi_zero_depth_on_is_refused_by_budget(self) -> None:
        row = _row("pi_zero")
        assert row["max_refinement_passes"] == 0
        assert row["depth_on_status"] == "refused_by_budget"
        assert row["depth_on_score"] is None
        # Its depth headline is the single-pass OFF
        # score, filed as exactly that.
        assert row["suite_score"] == pytest.approx(
            0.7252314814814815)
        assert FINDING["refused_cells"] == [
            ["pi_zero", "depth_on", "max_refinement_passes=0"]]
        refused = [r for r in PANEL_ROWS
                   if r["depth_on_status"] == "refused_by_budget"]
        assert [r["profile"] for r in refused] == ["pi_zero"]

    def test_a_measured_row_recomputes_live(self) -> None:
        live = measure_profile_row("pi_5")
        assert live == dict(_row("pi_5"))
        assert live["boot_verdict"] == "boots_with_ceiling_gaps"


# ── The panel run + its pinned baseline ─────────────────────────────────────


class TestPanelRun:
    def test_full_run_passes_with_30_cells_one_refused(self) -> None:
        result = run_profile_panel()
        assert result.total_cells == 30
        assert result.refused_cells == 1
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.budgets_applied_everywhere is True
        assert result.boots_reproduce is True
        assert result.caps_reproduce is True
        assert result.rows_reproduce is True
        assert result.overall_score == pytest.approx(
            0.794675925925926)
        assert [dict(r) for r in result.rows] == \
            [dict(r) for r in PANEL_ROWS]
        assert result.finding == FINDING

    def test_empty_panel_scores_zero_and_reviews(self) -> None:
        empty = run_profile_panel(profiles=())
        assert empty.total_cells == 0
        assert empty.overall_score == 0.0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_single_profile_selection_measures_and_reviews(
            self) -> None:
        subset = run_profile_panel(profiles=("desktop_cpu",))
        assert subset.total_cells == CELLS_PER_PROFILE == 5
        assert subset.verdict == "REVIEW"
        assert subset.passes is False
        assert subset.rows[0]["profile"] == "desktop_cpu"

    def test_panel_is_deterministic(self) -> None:
        assert compute_panel_rows() == compute_panel_rows()
        assert finding_for(compute_panel_rows()) == FINDING

    def test_panel_baseline_remeasures_live_twice_identical(
            self) -> None:
        first = measure_profile_panel_baseline()
        second = measure_profile_panel_baseline()
        assert first == second
        assert first["total_cells"] == 30
        assert first["desktop_cpu_suite_score"] == pytest.approx(
            0.8085648148148149)
        assert first["pi_zero_suite_score"] == pytest.approx(
            0.7252314814814815)
        assert first["server_gpu_bench_avg_words"] == 9
        assert baseline_problems(
            "profile_panel_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703703704)
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
        assert harness.measure_profile_baseline()[
            "avg_confidence"] == pytest.approx(0.408)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_panel_is_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "profile_panel"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.profile_panel"
        assert entry[0]["bind_attribute"] == "run_profile_panel"
        assert entry[0]["fixture"] == "profile_panel_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence",
                                           "will")
        fixture = [f for f in FIXTURES
                   if f["id"] == "profile_panel_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "profile_panel_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cells"] == 30
        assert baseline[0]["metrics"]["pi_zero_suite_score"] == \
            pytest.approx(0.7252314814814815)
        assert baseline[0]["targets"][
            "desktop_cpu_suite_score"] == {
                "target": 0.8085648148148149, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_profile_benchmark_entry_files_tf128_done(self) -> None:
        entry = [b for b in BENCHMARKS
                 if b["id"] == "profile_benchmark"][0]
        assert entry["suite_slice"].startswith("TF-128 DONE")
        assert "profile_panel.py" in entry["suite_slice"]
        assert any("D-BENCH-1 FIXED by TF-128" in d
                   for d in entry["docket"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for profile in PANEL_PROFILES:
            assert ("profile_panel",
                    f"{profile}_suite_score") in defined
            assert ("profile_panel",
                    f"{profile}_bench_avg_words") in defined
        assert ("profile_panel", "total_cells") in defined
        assert len(BENCHMARKS) == 16
        assert len(FIXTURES) == 9
        assert len(BASELINES) == 9
        assert len(METRIC_DEFINITIONS) == 83
        assert harness.harness_problems() == []


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert PROFILE_PANEL_FORMAT == \
            "thoughtforge-profile-panel/1"
        assert PROFILE_PANEL_DOC == \
            "docs/architecture/PROFILE_PANEL.md"
        assert (REPO_ROOT / PROFILE_PANEL_DOC).exists()
        ids = [r["id"] for r in PANEL_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in ("budgets_enforced_during_the_run",
                       "ram_ceiling_is_tf069s_machinery",
                       "depth_on_refused_where_no_refinement_is_budgeted",
                       "live_per_profile_hardware_is_not_proven"):
            assert needed in ids
        data = profile_panel_as_data()
        assert data["format"] == PROFILE_PANEL_FORMAT
        assert len(data["rows"]) == 6
        assert data["finding"]["refused_cells"] == [
            ["pi_zero", "depth_on", "max_refinement_passes=0"]]
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_cells"] == 30
        assert data["measured"]["refused_cells"] == 1

    def test_profile_panel_checker_clean_on_the_real_estate(
            self) -> None:
        assert profile_panel_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(panel, "PANEL_ROWS", ())
        problems = profile_panel_problems()
        assert any("PANEL_ROWS" in p or "refused" in p
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

    def test_panel_world_is_the_shipped_profiles_and_suites(
            self) -> None:
        data = profile_panel_as_data()
        assert data["profiles"] == list(PANEL_PROFILES)
        assert data["suites"] == list(PANEL_SUITES)
