"""
TF-132 — The WYRD-shaped panel: world-state
consistency questions (feeds Part VIII's
boundary with evidence).

The proof lives as verifiable data in
``benchmarks/wyrd_panel.py`` (``WORLD_ENTITIES``,
``WORLD_EVENTS``, ``PANEL_CASES``,
``PANEL_PROBES``, ``WYRD_PANEL_RULES``, checked
by ``wyrd_panel_problems()``), registered in
``benchmarks/harness.py`` with the pinned
baseline ``wyrd_panel_synthetic_v1``; D-BENCH-17
(no world-state consistency panel existed — no
filed world with a timeline, no scorer judged
an answer against a stated state) was fixed by
this slice in the measuring sense.

Pinned here:

  * the filed world: Emberwick, synthetic
    invented (TF-017) — 8 entities, 6 timeline
    changes, replayed purely by
    ``world_state_at()`` (the ferry is back at
    the steps only after the return crossing),
  * the filed set: 18 cases over six axes —
    9 consistent, 8 contradicted, 1
    honest-unanswerable, overall 10/18 — every
    case reproducing its filed expectation
    through the panel's own scorer, every claim
    bound to its answer by its claim phrase,
  * the verdict shape: full set PASS, partial
    and empty selections REVIEW (no vacuous
    pass), a wrong filed expectation FAILs,
  * the probes: 7/7 against the pure scorer
    parts, proving the scorer can fail,
  * the checker clean, the harness
    registration (20 / 13 / 13 / 109, rules
    13), the new baseline re-measured live
    twice — with every earlier pinned baseline
    preserved exactly (the one filed exception:
    TF-131's hygiene baseline re-pinned in the
    open 20 -> 21 inputs for TF-132's new
    synthetic hygiene row) and TF-129's gated
    set untouched (the wyrd baseline is
    ungated, in the open),
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

import benchmarks.wyrd_panel as panel
import benchmarks.harness as harness
from benchmarks.data_hygiene import run_data_hygiene
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_data_hygiene_baseline,
    measure_wyrd_panel_baseline,
)
from benchmarks.regression_gate import (
    GATED_BASELINE_IDS,
    UNGATED_BASELINE_IDS,
    run_regression_gate,
)
from benchmarks.wyrd_panel import (
    PANEL_AXES,
    PANEL_CASES,
    PANEL_PROBES,
    TOTAL_CASES,
    TOTAL_ENTITIES,
    TOTAL_EVENTS,
    WORLD_ENTITIES,
    WORLD_EVENTS,
    WYRD_PANEL_BASELINE_ID,
    WYRD_PANEL_DOC,
    WYRD_PANEL_FORMAT,
    WYRD_PANEL_RULES,
    case_problems,
    render_wyrd_panel_markdown,
    run_panel_probes,
    run_wyrd_panel,
    score_case,
    world_state_at,
    wyrd_panel_as_data,
    wyrd_panel_problems,
)

REPO_ROOT = Path(__file__).parent.parent


def _by_id(rows):
    return {r["id"]: r for r in rows}


# ── The filed world ─────────────────────────────────────────────────────────


class TestFiledWorld:
    def test_eight_entities_six_changes(self) -> None:
        assert TOTAL_ENTITIES == 8
        assert len(WORLD_ENTITIES) == 8
        ids = [e["id"] for e in WORLD_ENTITIES]
        assert len(ids) == len(set(ids))
        assert TOTAL_EVENTS == 6
        assert len(WORLD_EVENTS) == 6
        for event in WORLD_EVENTS:
            for change in event["changes"]:
                assert change["subject"] in ids

    def test_the_timeline_replays_purely(self) -> None:
        dawn = world_state_at(0)
        assert dawn["skiff_wren"]["location"] == "ferry_steps"
        assert dawn["water_mill"]["wheel"] == "turning"
        assert dawn["great_bell"]["state"] == "silent"
        after_crossing = world_state_at(1)
        assert after_crossing["skiff_wren"]["location"] == \
            "far_bank"
        assert after_crossing["mara_venn"]["location"] == \
            "far_bank"
        after_break = world_state_at(2)
        assert after_break["water_mill"]["wheel"] == "stopped"
        end = world_state_at(6)
        assert end["skiff_wren"]["location"] == "ferry_steps"
        assert end["pip"]["location"] == "granary"
        # Replay is pure: the filed dawn state is untouched.
        assert WORLD_ENTITIES[5]["state"]["location"] == \
            "ferry_steps"


# ── The filed set + the scorer ──────────────────────────────────────────────


class TestFiledSet:
    def test_eighteen_cases_9_8_1_over_six_axes(self) -> None:
        assert TOTAL_CASES == 18
        assert len(PANEL_CASES) == 18
        ids = [c["id"] for c in PANEL_CASES]
        assert len(ids) == len(set(ids))
        assert list(PANEL_AXES) == [
            "state_at_point", "after_change", "relation",
            "count", "contradiction_probe", "unanswerable"]
        classes = [c["expected_class"] for c in PANEL_CASES]
        assert classes.count("consistent") == 9
        assert classes.count("contradicted") == 8
        assert classes.count("honest_unanswerable") == 1
        for case in PANEL_CASES:
            assert case["question"].strip()
            assert case["filed_fact"].strip()
            assert case["answer"].strip()
            assert 0 <= case["at"] <= len(WORLD_EVENTS)

    def test_every_case_reproduces_its_expectation(self) -> None:
        for case in PANEL_CASES:
            assert case_problems(case) == [], case["id"]
            scored = score_case(case)
            assert scored["classification"] == \
                case["expected_class"]
            assert scored["score"] == case["expected_score"]

    def test_the_scorer_judges_each_shape(self) -> None:
        cases = _by_id(PANEL_CASES)
        stale = score_case(cases["w06_wheel_stale_after_break"])
        assert stale["classification"] == "contradicted"
        assert stale["world_value"] == "stopped"
        honest = score_case(cases["w17_maras_age_unanswerable"])
        assert honest["classification"] == "honest_unanswerable"
        assert honest["answerable"] is False
        fabricated = score_case(
            cases["w18_ferry_colour_fabricated"])
        assert fabricated["classification"] == "contradicted"
        assert fabricated["score"] == 0.0
        counted = score_case(
            cases["w13_people_at_the_mill_after_bram"])
        assert counted["world_value"] == 2
        assert counted["classification"] == "consistent"


# ── Live evaluation ─────────────────────────────────────────────────────────


class TestLiveEvaluation:
    def test_full_panel_passes_with_filed_totals(self) -> None:
        result = run_wyrd_panel()
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.total_cases == 18
        assert result.consistent_cases == 9
        assert result.contradicted_cases == 8
        assert result.honest_unanswerable_cases == 1
        assert result.overall_score == \
            pytest.approx(0.5555555555555556)
        assert result.mismatches == 0

    def test_empty_selection_scores_zero_and_reviews(self) -> None:
        empty = run_wyrd_panel(case_ids=())
        assert empty.total_cases == 0
        assert empty.overall_score == 0.0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_partial_selection_scores_and_reviews(self) -> None:
        subset = run_wyrd_panel(case_ids=("w01_mara_at_dawn",))
        assert subset.total_cases == 1
        assert subset.consistent_cases == 1
        assert subset.verdict == "REVIEW"
        assert subset.passes is False

    def test_a_wrong_filed_expectation_fails_the_run(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = copy.deepcopy([dict(c) for c in PANEL_CASES])
        for case in broken:
            if case["id"] == "w01_mara_at_dawn":
                case["expected_class"] = "contradicted"
        monkeypatch.setattr(panel, "PANEL_CASES", tuple(broken))
        result = run_wyrd_panel()
        assert result.verdict == "FAIL"
        assert result.passes is False
        assert result.mismatches == 1


# ── The probes ──────────────────────────────────────────────────────────────


class TestProbes:
    def test_seven_probes_all_behave(self) -> None:
        probes = run_panel_probes()
        assert len(probes) == len(PANEL_PROBES) == 7
        assert all(p["passed"] for p in probes)
        by_id = {p["id"]: p for p in probes}
        assert by_id[
            "stale_pre_change_value_contradicts"][
                "classification"] == "contradicted"
        assert by_id[
            "fabricated_answer_to_unanswerable_contradicts"][
                "classification"] == "contradicted"
        assert by_id[
            "honest_unanswerable_scores_honest"][
                "classification"] == "honest_unanswerable"
        assert by_id[
            "wrong_filed_expectation_is_named"]["problems"]

    def test_the_pure_scorer_fails_by_shape(self) -> None:
        case = dict(PANEL_CASES[0])
        case["claim"] = "granary"
        case["claim_text"] = "granary"
        case["answer"] = "At dawn Mara Venn is at the granary."
        assert score_case(case)["classification"] == \
            "contradicted"
        assert case_problems(case) != []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_wyrd_panel_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "wyrd_panel"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.wyrd_panel"
        assert entry[0]["bind_attribute"] == "run_wyrd_panel"
        assert entry[0]["fixture"] == "wyrd_panel_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence",
                                           "will")
        assert any("D-BENCH-17 FIXED by TF-132" in d
                   for d in entry[0]["docket"])
        fixture = [f for f in FIXTURES
                   if f["id"] == "wyrd_panel_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        assert fixture[0]["bind_attribute"] == "PANEL_CASES"
        baseline = [b for b in BASELINES
                    if b["id"] == "wyrd_panel_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cases"] == 18
        assert baseline[0]["metrics"]["consistent_cases"] == 9
        assert baseline[0]["metrics"]["contradicted_cases"] == 8
        assert baseline[0]["targets"]["consistent_cases"] == {
            "target": 9, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for metric in ("total_cases", "consistent_cases",
                       "contradicted_cases",
                       "honest_unanswerable_cases",
                       "overall_score", "probes_passed",
                       "probes_total"):
            assert ("wyrd_panel", metric) in defined
        assert len(BENCHMARKS) == 23
        assert len(FIXTURES) == 13
        assert len(BASELINES) == 16
        assert len(METRIC_DEFINITIONS) == 132
        assert len(HARNESS_RULES) == 16
        assert "the_wyrd_panel_scores_world_state_consistency" \
            in {r["id"] for r in HARNESS_RULES}
        assert harness.harness_problems() == []

    def test_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_wyrd_panel_baseline()
        second = measure_wyrd_panel_baseline()
        assert first == second
        assert first["total_cases"] == 18
        assert first["consistent_cases"] == 9
        assert first["contradicted_cases"] == 8
        assert first["honest_unanswerable_cases"] == 1
        assert first["overall_score"] == \
            pytest.approx(0.5555555555555556)
        assert first["probes_passed"] == 7
        assert first["probes_total"] == 7
        assert baseline_problems("wyrd_panel_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        assert harness.measure_presence_suite_baseline()[
            "overall_score"] == pytest.approx(0.7777777777777778)
        assert harness.measure_spot_checks_baseline()[
            "total_transcripts"] == 8
        assert baseline_problems() == []

    def test_the_hygiene_baseline_repinned_in_the_open(self) -> None:
        # TF-132's filed exception: the new synthetic
        # fixture grew TF-131's inventory by one row.
        measured = measure_data_hygiene_baseline()
        assert measured["total_inputs"] == 21
        assert measured["synthetic_inputs"] == 15
        assert measured["derived_inputs"] == 6
        assert measured["clean_inputs"] == 21
        result = run_data_hygiene()
        assert result.verdict == "PASS"
        assert result.total_inputs == 21
        assert baseline_problems(
            "data_hygiene_synthetic_v1") == []

    def test_the_wyrd_baseline_is_not_gated(self) -> None:
        assert WYRD_PANEL_BASELINE_ID == "wyrd_panel_synthetic_v1"
        assert WYRD_PANEL_BASELINE_ID not in GATED_BASELINE_IDS
        assert WYRD_PANEL_BASELINE_ID in UNGATED_BASELINE_IDS
        assert len(GATED_BASELINE_IDS) == 9
        assert run_regression_gate().verdict == "PASS"


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert WYRD_PANEL_FORMAT == "thoughtforge-wyrd-panel/1"
        assert WYRD_PANEL_DOC == "docs/architecture/WYRD_PANEL.md"
        assert (REPO_ROOT / WYRD_PANEL_DOC).exists()
        ids = [r["id"] for r in WYRD_PANEL_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in (
                "the_world_is_filed_as_data_and_replayed",
                "every_case_is_judged_against_a_filed_fact",
                "claims_are_bound_to_their_answers",
                "honesty_about_the_unfiled_scores",
                "the_panel_feeds_tf145_it_does_not_decide_it"):
            assert needed in ids
        data = wyrd_panel_as_data()
        assert data["format"] == WYRD_PANEL_FORMAT
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_cases"] == 18
        assert data["measured"]["probes_passed"] == 7
        assert data["world"]["name"] == "Emberwick"

    def test_the_doc_files_the_rendered_panel(self) -> None:
        doc_text = (REPO_ROOT / WYRD_PANEL_DOC).read_text(
            encoding="utf-8")
        assert render_wyrd_panel_markdown().strip() in doc_text
        for case in PANEL_CASES:
            assert case["id"] in doc_text
            assert case["question"] in doc_text
            assert case["answer"] in doc_text

    def test_wyrd_panel_checker_clean_on_the_real_estate(
            self) -> None:
        assert wyrd_panel_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(panel, "PANEL_CASES", ())
        problems = wyrd_panel_problems()
        assert any("cases" in p or "PANEL" in p
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

    def test_the_panel_world_is_the_invented_one(self) -> None:
        data = wyrd_panel_as_data()
        assert data["world"]["name"] == "Emberwick"
        assert len(data["cases"]) == 18
        names = {e["name"] for e in data["world"]["entities"]}
        assert "Mara Venn" in names
        assert len(names) == 8
