"""
TF-124 — Depth suite: multi-step reasoning with
scaffolds on/off — the scaffold value measured, not
assumed.

The proof lives as verifiable data in
``benchmarks/depth_suite.py`` (``CHAIN_CASES`` /
``PLAN_CASES`` / ``CHECK_CASES``, ``SUITE_AXES``,
``SUITE_RULES``, checked by
``depth_suite_problems()``), registered in
``benchmarks/harness.py`` with the pinned baseline
``depth_suite_synthetic_v1``; D-BENCH-10 (scaffold
value measured by nothing; TF-022's execution
wiring assigned unnumbered) was added and fixed by
this slice in the measuring sense.

Pinned here:

  * the measured delta, per axis and overall, with
    its filed honesty shapes: no-gain cases on both
    roads and scaffold-loss cases (OFF holds, ON
    fails) scoring exactly as filed,
  * ON requires the working: the right answer
    without the filed intermediates does not hold,
    and a plan out of order does not hold,
  * the real-code leg: TF-022's selection returns
    the filed strategy per case, the OFF prompt is
    the single-pass candidate prompt (no "Working
    so far"), and every ON step prompt opens with
    its real instruction and carries the working
    section where the step consumes prior output,
  * the filed corpus: 36 scored problems, the filed
    on/off expectations true case by case,
  * the checker clean, the harness registration,
    and the new suite baseline re-measured live
    twice — with every earlier pinned baseline
    preserved exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no live model, never claimed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.depth_suite as suite
import benchmarks.harness as harness
from benchmarks.depth_suite import (
    ALL_SUITE_CASES,
    AXIS_PASS_THRESHOLD,
    CASES_PER_AXIS,
    CHAIN_CASES,
    CHECK_CASES,
    DEPTH_SUITE_DOC,
    DEPTH_SUITE_FORMAT,
    PLAN_CASES,
    SUITE_AXES,
    SUITE_RULES,
    depth_suite_as_data,
    depth_suite_problems,
    run_depth_suite,
    scaffold_prompts_for,
    score_chain_cases,
    score_check_cases,
    score_plan_cases,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_depth_suite_baseline,
)
from thoughtforge.cognition.scaffold_strategies import select_strategy

REPO_ROOT = Path(__file__).parent.parent


def _case(cases, case_id):
    return [c for c in cases if c["id"] == case_id][0]


# ── The measured delta + its honesty shapes ─────────────────────────────────


class TestMeasuredDelta:
    def test_axis_scores_are_the_filed_measurements(self) -> None:
        chain = score_chain_cases()
        assert (chain["off_held"], chain["on_held"]) == (5, 10)
        assert chain["off_score"] == pytest.approx(0.4166666666666667)
        assert chain["on_score"] == pytest.approx(0.8333333333333334)
        assert chain["delta"] == pytest.approx(0.4166666666666667)
        plan = score_plan_cases()
        assert (plan["off_held"], plan["on_held"]) == (6, 9)
        assert plan["off_score"] == pytest.approx(0.5)
        assert plan["on_score"] == pytest.approx(0.75)
        assert plan["delta"] == pytest.approx(0.25)
        check = score_check_cases()
        assert (check["off_held"], check["on_held"]) == (6, 10)
        assert check["off_score"] == pytest.approx(0.5)
        assert check["on_score"] == pytest.approx(0.8333333333333334)
        assert check["delta"] == pytest.approx(0.3333333333333333)

    def test_scaffold_loss_cases_score_as_filed(self) -> None:
        loss_chain = _case(CHAIN_CASES, "c11_scaffold_loss_slip")
        assert score_chain_cases([loss_chain])["off_held"] == 1
        assert score_chain_cases([loss_chain])["on_held"] == 0
        loss_plan = _case(PLAN_CASES, "p11_scaffold_loss_order")
        assert score_plan_cases([loss_plan])["off_held"] == 1
        assert score_plan_cases([loss_plan])["on_held"] == 0
        loss_check = _case(CHECK_CASES, "s09_scaffold_loss_invented")
        assert score_check_cases([loss_check])["off_held"] == 1
        assert score_check_cases([loss_check])["on_held"] == 0

    def test_no_gain_cases_score_as_filed(self) -> None:
        both_fail = _case(CHAIN_CASES, "c12_no_gain_transcription")
        axis = score_chain_cases([both_fail])
        assert axis["off_held"] == axis["on_held"] == 0
        missed = _case(CHECK_CASES, "s06_check_misses_error")
        axis = score_check_cases([missed])
        assert axis["off_held"] == axis["on_held"] == 0
        both_pass = _case(CHAIN_CASES, "c03_percent_then_add")
        axis = score_chain_cases([both_pass])
        assert axis["off_held"] == axis["on_held"] == 1

    def test_on_requires_the_working_material(self) -> None:
        case = dict(_case(CHAIN_CASES, "c01_precedence_chain"))
        case["working"] = "The answer is worked out somehow."
        assert score_chain_cases([case])["on_held"] == 0
        case["on_answer"] = case["expected"]
        assert score_chain_cases([case])["on_held"] == 0

    def test_plan_intermediates_must_be_in_order(self) -> None:
        case = dict(_case(PLAN_CASES, "p01_fete_order"))
        case["working"] = ("Plan: 1. send invites. 2. book hall. "
                            "3. order food. 4. set up hall.")
        assert score_plan_cases([case])["on_held"] == 0

    def test_a_wrong_strategy_scores_no_on_hold(self) -> None:
        case = dict(_case(CHAIN_CASES, "c01_precedence_chain"))
        case["intent"] = "planning_request"  # selects plan_then_answer
        assert score_chain_cases([case])["on_held"] == 0


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert DEPTH_SUITE_FORMAT == "thoughtforge-depth-suite/1"
        assert DEPTH_SUITE_DOC == "docs/architecture/DEPTH_SUITE.md"
        assert (REPO_ROOT / DEPTH_SUITE_DOC).exists()
        ids = [r["id"] for r in SUITE_RULES]
        assert len(ids) == len(set(ids)) == 7
        assert [a["id"] for a in SUITE_AXES] == [
            "chain_of_thought", "plan_then_answer", "self_check"]
        data = depth_suite_as_data()
        assert data["format"] == DEPTH_SUITE_FORMAT
        assert data["case_counts"]["total"] == 36

    def test_corpus_is_36_cases_balanced_per_axis(self) -> None:
        assert CASES_PER_AXIS == 12
        assert len(CHAIN_CASES) == len(PLAN_CASES) == \
            len(CHECK_CASES) == 12
        assert len(ALL_SUITE_CASES) == 36
        assert sum(c["off_holds"] for c in CHAIN_CASES) == 5
        assert sum(c["on_holds"] for c in CHAIN_CASES) == 10
        assert sum(c["off_holds"] for c in PLAN_CASES) == 6
        assert sum(c["on_holds"] for c in PLAN_CASES) == 9
        assert sum(c["off_holds"] for c in CHECK_CASES) == 6
        assert sum(c["on_holds"] for c in CHECK_CASES) == 10
        ids = [c["id"] for c in ALL_SUITE_CASES]
        assert len(ids) == len(set(ids))

    def test_expectations_hold_case_by_case(self) -> None:
        for cases, scorer in ((CHAIN_CASES, score_chain_cases),
                              (PLAN_CASES, score_plan_cases),
                              (CHECK_CASES, score_check_cases)):
            for case in cases:
                single = scorer([case])
                assert bool(single["off_held"]) is case["off_holds"], \
                    case["id"]
                assert bool(single["on_held"]) is case["on_holds"], \
                    case["id"]

    def test_every_case_names_its_tf022_strategy(self) -> None:
        for case in ALL_SUITE_CASES:
            strategy = select_strategy(case["intent"])
            assert strategy is not None
            assert strategy.name == case["strategy"]


# ── The real-code prompt leg ────────────────────────────────────────────────


class TestPromptLeg:
    def test_off_prompt_is_single_pass_without_working(self) -> None:
        for case in ALL_SUITE_CASES:
            built = scaffold_prompts_for(case)
            assert "Working so far" not in built["off_prompt"]
            assert built["off_prompt"].startswith(
                "Write one") or built["off_prompt"].startswith(
                "Rewrite")

    def test_on_step_prompts_carry_instructions_and_working(self) -> None:
        for case in ALL_SUITE_CASES:
            built = scaffold_prompts_for(case)
            strategy = built["strategy"]
            assert len(built["step_prompts"]) == len(strategy.steps)
            for step, prompt in zip(strategy.steps,
                                    built["step_prompts"]):
                assert prompt.startswith(step.instruction)
                if step.requires_prior:
                    assert "Working so far" in prompt
                    assert case["working"][:40] in prompt
                else:
                    assert "Working so far" not in prompt

    def test_the_three_strategies_have_their_real_shapes(self) -> None:
        chain = scaffold_prompts_for(CHAIN_CASES[0])
        assert [s.name for s in chain["strategy"].steps] == [
            "reason_stepwise", "answer_from_reasoning"]
        plan = scaffold_prompts_for(PLAN_CASES[0])
        assert [s.name for s in plan["strategy"].steps] == [
            "outline_plan", "execute_plan"]
        check = scaffold_prompts_for(CHECK_CASES[0])
        assert [s.name for s in check["strategy"].steps] == [
            "draft_answer", "check_answer", "revise_from_check"]


# ── The suite run + its pinned baseline ─────────────────────────────────────


class TestSuiteRun:
    def test_full_run_passes_with_filed_scores(self) -> None:
        result = run_depth_suite()
        assert result.total_cases == 36
        assert result.chain_off == pytest.approx(0.4166666666666667)
        assert result.chain_on == pytest.approx(0.8333333333333334)
        assert result.plan_off == pytest.approx(0.5)
        assert result.plan_on == pytest.approx(0.75)
        assert result.check_off == pytest.approx(0.5)
        assert result.check_on == pytest.approx(0.8333333333333334)
        assert result.overall_off == pytest.approx(0.47222222222222227)
        assert result.overall_on == pytest.approx(0.8055555555555557)
        assert result.delta == pytest.approx(0.3333333333333334)
        assert result.verdict == "PASS"
        assert result.passes is True
        assert AXIS_PASS_THRESHOLD == 0.75

    def test_empty_suite_scores_zero_and_reviews(self) -> None:
        result = run_depth_suite([], [], [])
        assert result.total_cases == 0
        assert result.overall_off == result.overall_on == 0.0
        assert result.delta == 0.0
        assert result.verdict == "REVIEW"
        assert result.passes is False
        assert score_chain_cases([])["on_score"] == 0.0
        assert score_plan_cases([])["delta"] == 0.0
        assert score_check_cases([])["off_score"] == 0.0

    def test_no_positive_delta_keeps_the_suite_at_review(self) -> None:
        losses = [_case(CHAIN_CASES, "c11_scaffold_loss_slip")] * 4
        result = run_depth_suite(chain_cases=losses)
        assert result.chain_on == 0.0
        assert result.verdict == "REVIEW"

    def test_a_failing_axis_keeps_the_suite_at_review(self) -> None:
        bad = [_case(PLAN_CASES, "p08_plan_misses_step")] * 4
        result = run_depth_suite(plan_cases=bad)
        assert result.plan_on == 0.0
        assert result.verdict == "REVIEW"

    def test_suite_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_depth_suite_baseline()
        second = measure_depth_suite_baseline()
        assert first == second
        assert first["overall_on_score"] == \
            pytest.approx(0.8055555555555557)
        assert first["overall_off_score"] == \
            pytest.approx(0.47222222222222227)
        assert first["delta"] == pytest.approx(0.3333333333333334)
        assert first["total_cases"] == 36
        assert baseline_problems("depth_suite_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_suite_is_registered_with_fixture_and_baseline(self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "depth_suite"]
        assert len(entry) == 1
        assert entry[0]["bind_module"] == "benchmarks.depth_suite"
        assert entry[0]["fixture"] == "depth_suite_corpus"
        fixture = [f for f in FIXTURES
                   if f["id"] == "depth_suite_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "depth_suite_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cases"] == 36
        assert baseline[0]["targets"]["overall_on_score"] == \
            {"target": 0.75, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined(self) -> None:
        defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
        for metric in ("chain_off_score", "chain_on_score",
                       "plan_off_score", "plan_on_score",
                       "check_off_score", "check_on_score",
                       "overall_off_score", "overall_on_score",
                       "delta", "total_cases"):
            assert ("depth_suite", metric) in defined
        assert len(BENCHMARKS) == 21  # + presence_suite (TF-125) + honesty_panel (TF-126) + quant_panel (TF-127) + profile_panel (TF-128) + regression_gate (TF-129) + spot_checks (TF-130) + data_hygiene (TF-131) + wyrd_panel (TF-132) + cost_panel (TF-133), in the open
        assert len(FIXTURES) == 13  # + presence_suite_corpus (TF-125) + honesty_panel_corpus (TF-126) + quant_panel_corpus (TF-127) + profile_panel_corpus (TF-128) + regression_gate_corpus (TF-129) + spot_checks_corpus (TF-130) + data_hygiene_corpus (TF-131) + wyrd_panel_corpus (TF-132), in the open
        assert len(BASELINES) == 14  # + presence_suite_synthetic_v1 (TF-125) + honesty_panel_synthetic_v1 (TF-126) + quant_panel_synthetic_v1 (TF-127) + profile_panel_synthetic_v1 (TF-128) + regression_gate_synthetic_v1 (TF-129) + spot_checks_synthetic_v1 (TF-130) + data_hygiene_synthetic_v1 (TF-131) + wyrd_panel_synthetic_v1 (TF-132) + cost_panel_synthetic_v1 (TF-133), in the open
        assert len(METRIC_DEFINITIONS) == 119  # + 5 presence_suite metrics (TF-125) + 13 honesty_panel metrics (TF-126) + 16 quant_panel metrics (TF-127) + 13 profile_panel metrics (TF-128) + 6 regression_gate metrics (TF-129) + 6 spot_checks metrics (TF-130) + 7 data_hygiene metrics (TF-131) + 7 wyrd_panel metrics (TF-132) + 10 cost_panel metrics (TF-133), in the open

    def test_harness_checker_stays_clean(self) -> None:
        assert harness.harness_problems() == []
        assert baseline_problems() == []


# ── Checker ─────────────────────────────────────────────────────────────────


class TestChecker:
    def test_depth_suite_checker_clean_on_the_real_estate(self) -> None:
        assert depth_suite_problems() == []

    def test_a_broken_corpus_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(suite, "CHAIN_CASES", CHAIN_CASES[:6])
        problems = depth_suite_problems()
        assert any("CHAIN_CASES" in p for p in problems)


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

    def test_suite_world_is_synthetic_known_answer_problems(self) -> None:
        data = depth_suite_as_data()
        assert data["measured"]["delta"] == \
            pytest.approx(0.3333333333333334)
        joined = " ".join(c["question"] for c in ALL_SUITE_CASES)
        assert "Compute" in joined and "Order" in joined
