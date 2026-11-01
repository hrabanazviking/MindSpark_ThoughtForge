"""
TF-127 — Quant-vs-quality panel (consumes TF-067
mechanically).

The proof lives as verifiable data in
``benchmarks/quant_panel.py`` (``PANEL_ROWS``,
``KIND_ROWS``, ``FINDING``, ``PANEL_RULES``,
checked by ``quant_panel_problems()``),
registered in ``benchmarks/harness.py`` with the
pinned baseline ``quant_panel_synthetic_v1``;
D-BENCH-13 (quant-vs-quality measured only as
TF-067's single overall curve in ``src/`` — never
as a Part VII panel with a per-kind breakdown or
a harness baseline; the gap assigned to TF-127 in
TF-067's doc without ever being numbered) was
added and fixed by this slice in the measuring
sense.

Pinned here:

  * the measured panel: TF-067's overall curve in
    row form (48 cells) reproduced exactly —
    rounded to 4 places against TF-067's own
    CURVE — and the 18 per-kind rows (fact /
    reasoning / instruction per level) with their
    losses against each kind's own fp16 mean,
  * the finding from the numbers: reasoning
    carries the largest loss at both mid levels
    and plateaus at 0.5 from q2_k through q5_k_m;
    instruction collapses to 0.0 at q2_k; the
    non-monotone q4/q5 step is carried,
  * the mechanical consumption: the panel's rows
    equal TF-067's ``fixture_panel()`` rows, a
    custom generate callable flows through
    TF-067's runner, a fact-only prompt subset
    measures its cells and files REVIEW, and a
    selection without fp16 raises TF-067's own
    refusal,
  * the checker clean, the harness registration
    (15 / 8 / 8 / 70), the new panel baseline
    re-measured live twice — with every earlier
    pinned baseline preserved exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no live model, never claimed; live
quant-vs-quality is NOT PROVEN (TF-173).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.quant_panel as panel
import benchmarks.harness as harness
from benchmarks.quant_panel import (
    FINDING,
    KIND_PROMPT_COUNTS,
    KIND_ROWS,
    PANEL_KINDS,
    PANEL_ROWS,
    PANEL_RULES,
    QUANT_PANEL_DOC,
    QUANT_PANEL_FORMAT,
    TOTAL_CELLS,
    compute_quant_panel,
    finding_for,
    kind_rows_for,
    quant_panel_as_data,
    quant_panel_problems,
    run_quant_panel,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_quant_panel_baseline,
)
from thoughtforge.inference.quant_quality import (
    CURVE,
    NON_MONOTONE_STEPS,
    QUANT_LEVELS,
    QUANT_PROMPTS,
    fixture_generate,
    fixture_panel,
    quant_quality_problems,
    score_answer,
)

REPO_ROOT = Path(__file__).parent.parent


def _row(quant):
    return [r for r in PANEL_ROWS if r["quant"] == quant][0]


def _kind(quant, kind):
    return [r for r in KIND_ROWS
            if r["quant"] == quant and r["kind"] == kind][0]


# ── The measured panel + the finding ────────────────────────────────────────


class TestMeasuredPanel:
    def test_panel_rows_reproduce_tf067_curve_exactly(self) -> None:
        assert len(PANEL_ROWS) == 6
        assert [r["quant"] for r in PANEL_ROWS] == list(QUANT_LEVELS)
        published = {
            (q, round(m, 4), round(l, 4), p, c)
            for q, m, l, p, c in CURVE}
        filed = {
            (r["quant"], round(r["mean_score"], 4),
             round(r["loss_vs_reference"], 4), r["perfect"],
             r["contradiction_prompts"]) for r in PANEL_ROWS}
        assert filed == published
        assert _row("iq2_xs")["mean_score"] == 0.0
        assert _row("iq2_xs")["contradiction_prompts"] == 7
        assert _row("q2_k")["mean_score"] == pytest.approx(0.4375)
        assert _row("q4_k_m")["mean_score"] == pytest.approx(0.8125)
        assert _row("q5_k_m")["mean_score"] == pytest.approx(
            0.7083333333333334)
        assert _row("q8_0")["mean_score"] == 1.0
        assert _row("fp16")["loss_vs_reference"] == 0.0
        assert TOTAL_CELLS == 48
        assert sum(r["prompts"] for r in PANEL_ROWS) == 48

    def test_kind_rows_are_the_filed_breakdown(self) -> None:
        assert len(KIND_ROWS) == 18
        assert [(r["quant"], r["kind"]) for r in KIND_ROWS] == [
            (q, k) for q in QUANT_LEVELS for k in PANEL_KINDS]
        assert PANEL_KINDS == ("fact", "reasoning", "instruction")
        assert KIND_PROMPT_COUNTS == {
            "fact": 5, "reasoning": 2, "instruction": 1}
        for row in KIND_ROWS:
            assert row["prompts"] == KIND_PROMPT_COUNTS[row["kind"]]
        # Kind rows aggregate back to the overall rows.
        for quant in QUANT_LEVELS:
            cells = [r for r in KIND_ROWS if r["quant"] == quant]
            total = sum(r["prompts"] for r in cells)
            mean = sum(r["mean_score"] * r["prompts"]
                       for r in cells) / total
            assert mean == pytest.approx(_row(quant)["mean_score"])
            assert sum(r["perfect"] for r in cells) == \
                _row(quant)["perfect"]
            assert sum(r["contradiction_prompts"] for r in cells) == \
                _row(quant)["contradiction_prompts"]

    def test_per_kind_numbers_are_filed_exactly(self) -> None:
        assert _kind("q4_k_m", "fact")["mean_score"] == \
            pytest.approx(0.9)
        assert _kind("q4_k_m", "fact")["loss_vs_fp16"] == \
            pytest.approx(0.09999999999999998)
        assert _kind("q5_k_m", "fact")["mean_score"] == \
            pytest.approx(0.8)
        assert _kind("q5_k_m", "instruction")["mean_score"] == \
            pytest.approx(0.6666666666666666)
        assert _kind("iq2_xs", "fact")["contradiction_prompts"] == 5
        assert _kind("iq2_xs", "reasoning")[
            "contradiction_prompts"] == 2
        assert _kind("q8_0", "reasoning")["perfect"] == 2
        for kind in PANEL_KINDS:
            assert _kind("fp16", kind)["loss_vs_fp16"] == 0.0

    def test_the_finding_from_the_numbers(self) -> None:
        assert FINDING["largest_loss_kind_at_q4_k_m"] == "reasoning"
        assert FINDING["largest_loss_kind_at_q5_k_m"] == "reasoning"
        assert FINDING["reasoning_plateau_levels"] == [
            "q2_k", "q4_k_m", "q5_k_m"]
        assert FINDING["reasoning_plateau_score"] == \
            pytest.approx(0.5)
        for quant in FINDING["reasoning_plateau_levels"]:
            assert _kind(quant, "reasoning")["mean_score"] == \
                pytest.approx(0.5)
        assert _kind("q8_0", "reasoning")["mean_score"] == 1.0
        assert FINDING["instruction_score_at_q2_k"] == 0.0
        assert _kind("q2_k", "fact")["mean_score"] == \
            pytest.approx(0.5)

    def test_non_monotone_step_is_carried_not_smoothed(self) -> None:
        assert _row("q4_k_m")["mean_score"] > \
            _row("q5_k_m")["mean_score"]
        assert FINDING["non_monotone_steps"] == [
            [a, b] for a, b in NON_MONOTONE_STEPS]
        assert NON_MONOTONE_STEPS == (("q4_k_m", "q5_k_m"),)
        # Per kind, the step is fact's + instruction's;
        # reasoning is flat across it.
        assert _kind("q4_k_m", "fact")["mean_score"] > \
            _kind("q5_k_m", "fact")["mean_score"]
        assert _kind("q4_k_m", "reasoning")["mean_score"] == \
            _kind("q5_k_m", "reasoning")["mean_score"]


# ── Mechanical consumption of TF-067 ────────────────────────────────────────


class TestMechanicalConsumption:
    def test_rows_equal_tf067_fixture_panel_rows(self) -> None:
        live = fixture_panel()
        assert [(r.quant, r.mean_score, r.loss_vs_reference,
                 r.perfect, r.contradiction_prompts)
                for r in live.rows] == [
            (r["quant"], r["mean_score"], r["loss_vs_reference"],
             r["perfect"], r["contradiction_prompts"])
            for r in PANEL_ROWS]
        assert kind_rows_for(live) == tuple(
            dict(r) for r in KIND_ROWS)
        assert quant_quality_problems() == []

    def test_a_custom_generate_flows_through_tf067s_runner(
            self) -> None:
        def perfect(quant, prompt):
            return prompt.reference

        result = run_quant_panel(generate=perfect)
        assert result.total_cells == 48
        assert all(r["mean_score"] == 1.0 for r in result.rows)
        # A perfect generator does not reproduce the
        # filed fixture curve — REVIEW, honestly.
        assert result.curve_reproduces is False
        assert result.verdict == "REVIEW"
        reference = QUANT_PROMPTS[0]
        assert score_answer(reference, perfect("fp16", reference)
                            ).score == 1.0

    def test_a_fact_only_subset_measures_and_reviews(self) -> None:
        fact_only = tuple(p for p in QUANT_PROMPTS
                          if p.kind == "fact")
        result = run_quant_panel(prompts=fact_only)
        assert result.total_cells == 6 * 5 == 30
        assert result.verdict == "REVIEW"
        assert result.passes is False
        by_quant = {r["quant"]: r for r in result.rows}
        assert by_quant["q4_k_m"]["mean_score"] == pytest.approx(0.9)
        assert by_quant["fp16"]["mean_score"] == 1.0
        assert result.finding["largest_loss_kind_at_q4_k_m"] == \
            "fact"
        assert result.finding["reasoning_plateau_score"] is None

    def test_a_selection_without_fp16_raises_tf067s_refusal(
            self) -> None:
        with pytest.raises(ValueError, match="reference"):
            run_quant_panel(levels=("q4_k_m", "q8_0"))
        with pytest.raises(ValueError, match="reference"):
            compute_quant_panel(levels=("q2_k",))

    def test_generate_is_tf067s_fixture_generate_by_default(
            self) -> None:
        prompt = QUANT_PROMPTS[0]
        assert fixture_generate("q2_k", prompt) != \
            fixture_generate("fp16", prompt)
        rows, kinds = compute_quant_panel()
        assert rows == tuple(dict(r) for r in PANEL_ROWS)
        assert kinds == tuple(dict(r) for r in KIND_ROWS)


# ── The panel run + its pinned baseline ─────────────────────────────────────


class TestPanelRun:
    def test_full_run_passes_with_48_cells(self) -> None:
        result = run_quant_panel()
        assert result.total_cells == 48
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.curve_reproduces is True
        assert result.kind_rows_reproduce is True
        assert result.overall_score == pytest.approx(
            0.6597222222222222)
        assert [dict(r) for r in result.rows] == \
            [dict(r) for r in PANEL_ROWS]
        assert result.finding == FINDING

    def test_empty_panel_scores_zero_and_reviews(self) -> None:
        for empty in (run_quant_panel(levels=()),
                      run_quant_panel(prompts=())):
            assert empty.total_cells == 0
            assert empty.overall_score == 0.0
            assert empty.verdict == "REVIEW"
            assert empty.passes is False

    def test_panel_is_deterministic(self) -> None:
        assert compute_quant_panel() == compute_quant_panel()
        assert finding_for(*compute_quant_panel()) == FINDING

    def test_panel_baseline_remeasures_live_twice_identical(
            self) -> None:
        first = measure_quant_panel_baseline()
        second = measure_quant_panel_baseline()
        assert first == second
        assert first["total_cells"] == 48
        assert first["q4_k_m_score"] == pytest.approx(0.8125)
        assert first["q5_k_m_loss"] == pytest.approx(
            0.29166666666666663)
        assert first["fact_mean_score"] == pytest.approx(
            0.7000000000000001)
        assert first["reasoning_mean_score"] == pytest.approx(
            0.5833333333333334)
        assert first["instruction_mean_score"] == pytest.approx(
            0.611111111111111)
        assert baseline_problems("quant_panel_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        depth = harness.measure_depth_suite_baseline()
        assert depth["overall_off_score"] == \
            pytest.approx(0.47222222222222227)
        assert depth["overall_on_score"] == \
            pytest.approx(0.8055555555555557)
        assert harness.measure_presence_suite_baseline()[
            "overall_score"] == pytest.approx(0.7777777777777778)
        honesty = harness.measure_honesty_panel_baseline()
        assert honesty["mid_delta"] == pytest.approx(
            0.24004629629629637)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_panel_is_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "quant_panel"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.quant_panel"
        assert entry[0]["bind_attribute"] == "run_quant_panel"
        assert entry[0]["fixture"] == "quant_panel_corpus"
        assert entry[0]["claims_axes"] == ("depth",)
        fixture = [f for f in FIXTURES
                   if f["id"] == "quant_panel_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "quant_panel_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cells"] == 48
        assert baseline[0]["targets"]["fp16_score"] == \
            {"target": 1.0, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_quant_quality_neighbour_files_tf127_done(self) -> None:
        entry = [b for b in BENCHMARKS
                 if b["id"] == "quant_quality"][0]
        assert entry["suite_slice"].startswith("TF-127 DONE")
        assert "quant_panel.py" in entry["suite_slice"]

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for level in QUANT_LEVELS:
            assert ("quant_panel", f"{level}_score") in defined
            assert ("quant_panel", f"{level}_loss") in defined
        for kind in PANEL_KINDS:
            assert ("quant_panel", f"{kind}_mean_score") in defined
        assert ("quant_panel", "total_cells") in defined
        assert len(BENCHMARKS) == 15
        assert len(FIXTURES) == 8
        assert len(BASELINES) == 8
        assert len(METRIC_DEFINITIONS) == 70
        assert harness.harness_problems() == []


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert QUANT_PANEL_FORMAT == "thoughtforge-quant-panel/1"
        assert QUANT_PANEL_DOC == "docs/architecture/QUANT_PANEL.md"
        assert (REPO_ROOT / QUANT_PANEL_DOC).exists()
        ids = [r["id"] for r in PANEL_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in ("consumes_tf067_mechanically",
                       "non_monotone_step_carried_not_smoothed",
                       "neighbouring_levels_are_not_ranked",
                       "live_quant_quality_is_not_proven"):
            assert needed in ids
        data = quant_panel_as_data()
        assert data["format"] == QUANT_PANEL_FORMAT
        assert len(data["rows"]) == 6
        assert len(data["kind_rows"]) == 18
        assert data["finding"]["largest_loss_kind_at_q4_k_m"] == \
            "reasoning"
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_cells"] == 48

    def test_quant_panel_checker_clean_on_the_real_estate(
            self) -> None:
        assert quant_panel_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(panel, "KIND_ROWS", ())
        problems = quant_panel_problems()
        assert any("KIND_ROWS" in p for p in problems)


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

    def test_panel_world_is_tf067s_synthetic_fixture_corpus(
            self) -> None:
        data = quant_panel_as_data()
        assert data["levels"] == list(QUANT_LEVELS)
        assert data["kind_prompt_counts"] == {
            "fact": 5, "reasoning": 2, "instruction": 1}
