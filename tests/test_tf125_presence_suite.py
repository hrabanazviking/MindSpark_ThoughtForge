"""
TF-125 — Presence suite: long-conversation coherence
+ the will layer's follow-through (TF-029) scored
over scripted sagas.

The proof lives as verifiable data in
``benchmarks/presence_suite.py`` (``COHERENCE_SAGAS``
/ ``STEERING_SAGAS`` / ``CLOSURE_SAGAS``,
``SUITE_AXES``, ``SUITE_RULES``, checked by
``presence_suite_problems()``), registered in
``benchmarks/harness.py`` with the pinned baseline
``presence_suite_synthetic_v1``; D-BENCH-11
(presence and will follow-through scored by
nothing; the gap filed as a harness rule and
assigned in TF-028's doc but never numbered in the
docket) was added and fixed by this slice in the
measuring sense, and TF-121's will-gap harness
rule now files TF-125 DONE.

Pinned here:

  * the measured scores per axis, with the filed
    honesty shapes: coherence breaks (onsets 15 /
    16, a truncated drifting saga), a dropped goal,
    a contradicted goal, and the still-steered
    closures (abandoned / completed / superseded)
    all scoring held=False by name,
  * the coherence hold floor is TF-028's own
    PASS_LATE_MIN over the whole-saga mean, with
    TF-028's stricter verdict carried as data,
  * the real-code legs: TF-028's fixture engine
    reproduces the onset sagas' transcripts, and
    TF-029's ledger + inject_will judge and steer
    over replays of the filed sagas,
  * the filed corpus: 36 scored sagas, the filed
    expectations true case by case,
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

import benchmarks.presence_suite as suite
import benchmarks.harness as harness
from benchmarks.presence_suite import (
    ALL_SUITE_CASES,
    AXIS_PASS_THRESHOLD,
    CASES_PER_AXIS,
    CLOSURE_SAGAS,
    COHERENCE_SAGAS,
    HOLD_MIN_SCORE,
    PRESENCE_SUITE_DOC,
    PRESENCE_SUITE_FORMAT,
    RESPONSE_GOAL,
    STEERING_SAGAS,
    SUITE_AXES,
    SUITE_RULES,
    coherence_saga_report,
    judge_will_saga,
    presence_suite_as_data,
    presence_suite_problems,
    run_presence_suite,
    score_closure_sagas,
    score_coherence_sagas,
    score_steering_sagas,
    transcript_for_saga,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_presence_suite_baseline,
)
from thoughtforge.cognition.coherence import (
    COHERENCE_SCRIPT,
    PASS_LATE_MIN,
    CoherenceFixtureEngine,
)
from thoughtforge.cognition.will import (
    MAX_UNTOUCHED_TURNS,
    GoalLedger,
    inject_will,
)
from thoughtforge.knowledge.models import (
    CognitionScaffold,
    InputSketch,
)

REPO_ROOT = Path(__file__).parent.parent


def _saga(sagas, saga_id):
    return [s for s in sagas if s["id"] == saga_id][0]


def _replay(saga, upto=None):
    ledger = GoalLedger()
    turns = saga["turns"] if upto is None else saga["turns"][:upto]
    for turn in turns:
        ledger.observe_turn(InputSketch(
            raw_text=turn["user"], intent=turn["intent"],
            topic=turn["topic"], tone_in="neutral",
            response_mode="conversational"), RESPONSE_GOAL)
    return ledger


# ── The measured scores + their honesty shapes ──────────────────────────────


class TestMeasuredScores:
    def test_axis_scores_are_the_filed_measurements(self) -> None:
        coherence = score_coherence_sagas()
        assert (coherence["cases"], coherence["held"]) == (12, 9)
        assert coherence["score"] == pytest.approx(0.75)
        steering = score_steering_sagas()
        assert (steering["cases"], steering["held"]) == (12, 10)
        assert steering["score"] == pytest.approx(0.8333333333333334)
        closure = score_closure_sagas()
        assert (closure["cases"], closure["held"]) == (12, 9)
        assert closure["score"] == pytest.approx(0.75)

    def test_coherence_breaks_score_as_filed(self) -> None:
        canonical = coherence_saga_report(
            _saga(COHERENCE_SAGAS, "k07_drift_from_15_break"))
        assert canonical["mean_score"] == pytest.approx(0.7)
        assert canonical["contradiction_turns"] == 6
        assert canonical["verdict"] == "REVIEW"
        onset16 = coherence_saga_report(
            _saga(COHERENCE_SAGAS, "k06_drift_from_16_break"))
        assert onset16["mean_score"] == pytest.approx(0.75)
        truncated = coherence_saga_report(_saga(
            COHERENCE_SAGAS, "k12_truncated_drift_from_12_break"))
        assert truncated["mean_score"] == pytest.approx(
            0.7857142857142857)
        for report in (canonical, onset16, truncated):
            assert report["held"] is False

    def test_coherence_hold_floor_is_tf028s(self) -> None:
        assert HOLD_MIN_SCORE == PASS_LATE_MIN == 0.8
        at_floor = coherence_saga_report(
            _saga(COHERENCE_SAGAS, "k05_drift_from_17"))
        assert at_floor["mean_score"] == pytest.approx(0.8)
        assert at_floor["held"] is True
        full = coherence_saga_report(
            _saga(COHERENCE_SAGAS, "k01_coherent_full"))
        assert full["mean_score"] == 1.0
        assert full["verdict"] == "PASS"

    def test_still_steered_closures_fail_by_name(self) -> None:
        for saga_id, status in (
                ("c10_abandoned_goal_still_steered_fails", "abandoned"),
                ("c11_completed_goal_still_steered_fails", "completed"),
                ("c12_superseded_goal_still_steered_fails",
                 "superseded")):
            judged = judge_will_saga(_saga(CLOSURE_SAGAS, saga_id))
            assert judged["held"] is False
            assert any(status in v and "still steers" in v
                       for v in judged["violations"]), saga_id

    def test_dropped_and_contradicted_goals_fail(self) -> None:
        for saga_id in ("w11_goal_dropped_fails",
                        "w12_goal_contradicted_fails"):
            judged = judge_will_saga(_saga(STEERING_SAGAS, saga_id))
            assert judged["held"] is False
            assert any("does not carry it" in v
                       for v in judged["violations"]), saga_id


# ── The will judge ──────────────────────────────────────────────────────────


class TestWillJudge:
    def test_steering_requires_the_statement(self) -> None:
        saga = _saga(STEERING_SAGAS, "w01_declared_goal_steers")
        turns = [dict(t) for t in saga["turns"]]
        turns[2]["assistant"] = "Wrenna keeps them, of course."
        broken = {**saga, "turns": tuple(turns)}
        assert judge_will_saga(broken)["held"] is False

    def test_end_states_match_the_filing_case_by_case(self) -> None:
        for saga in STEERING_SAGAS + CLOSURE_SAGAS:
            judged = judge_will_saga(saga)
            assert judged["end_active"] == saga["end_active"], saga["id"]
            assert judged["end_closed"] == list(saga["end_closed"]), \
                saga["id"]
            assert judged["held"] is saga["holds"], saga["id"]

    def test_redeclared_same_goal_touches_not_supersedes(self) -> None:
        judged = judge_will_saga(
            _saga(STEERING_SAGAS, "w06_redeclared_goal_touches"))
        assert judged["held"] is True
        assert judged["end_closed"] == []
        assert judged["end_active"] == "count the harbour lanterns"

    def test_stale_count_is_tf029s_filed_count(self) -> None:
        saga = _saga(CLOSURE_SAGAS, "c07_untouched_goal_goes_stale")
        assert len(saga["turns"]) == MAX_UNTOUCHED_TURNS + 3
        judged = judge_will_saga(saga)
        assert judged["held"] is True
        assert judged["end_closed"] == ["stale"]
        alive = _replay(saga, upto=MAX_UNTOUCHED_TURNS + 1)
        assert alive.active is not None  # at the limit: still steering
        stale = _replay(saga, upto=MAX_UNTOUCHED_TURNS + 2)
        assert stale.active is None


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert PRESENCE_SUITE_FORMAT == "thoughtforge-presence-suite/1"
        assert PRESENCE_SUITE_DOC == "docs/architecture/PRESENCE_SUITE.md"
        assert (REPO_ROOT / PRESENCE_SUITE_DOC).exists()
        ids = [r["id"] for r in SUITE_RULES]
        assert len(ids) == len(set(ids)) == 7
        assert [a["id"] for a in SUITE_AXES] == [
            "long_conversation_coherence", "will_steering",
            "will_closure"]
        data = presence_suite_as_data()
        assert data["format"] == PRESENCE_SUITE_FORMAT
        assert data["case_counts"]["total"] == 36
        assert len(data["coherence_reports"]) == 12

    def test_corpus_is_36_sagas_balanced_per_axis(self) -> None:
        assert CASES_PER_AXIS == 12
        assert len(COHERENCE_SAGAS) == len(STEERING_SAGAS) == \
            len(CLOSURE_SAGAS) == 12
        assert len(ALL_SUITE_CASES) == 36
        assert sum(s["holds"] for s in COHERENCE_SAGAS) == 9
        assert sum(s["holds"] for s in STEERING_SAGAS) == 10
        assert sum(s["holds"] for s in CLOSURE_SAGAS) == 9
        ids = [s["id"] for s in ALL_SUITE_CASES]
        assert len(ids) == len(set(ids))

    def test_expectations_hold_case_by_case(self) -> None:
        for sagas, scorer in ((COHERENCE_SAGAS, score_coherence_sagas),
                              (STEERING_SAGAS, score_steering_sagas),
                              (CLOSURE_SAGAS, score_closure_sagas)):
            for saga in sagas:
                single = scorer([saga])
                assert bool(single["held"]) is saga["holds"], saga["id"]

    def test_coherence_transcripts_come_from_tf028s_script(self) -> None:
        coherent = transcript_for_saga(COHERENCE_SAGAS[0])
        assert coherent == [(t.user, t.coherent)
                            for t in COHERENCE_SCRIPT]
        broken = transcript_for_saga(
            _saga(COHERENCE_SAGAS, "k07_drift_from_15_break"))
        assert broken[:14] == coherent[:14]
        assert broken[14:] == [(t.user, t.drifted)
                               for t in COHERENCE_SCRIPT[14:]]
        truncated = transcript_for_saga(
            _saga(COHERENCE_SAGAS, "k11_truncated_coherent_14"))
        assert len(truncated) == 14


# ── The real-code legs ──────────────────────────────────────────────────────


class TestRealCodeLegs:
    def test_fixture_engine_reproduces_onset_sagas(self) -> None:
        for saga_id, onset in (("k01_coherent_full", None),
                               ("k07_drift_from_15_break", 15)):
            engine = CoherenceFixtureEngine(drift_from_turn=onset)
            for user, answer in transcript_for_saga(
                    _saga(COHERENCE_SAGAS, saga_id)):
                got = engine.generate(f"User request: {user}\n").text
                assert got == answer, saga_id

    def test_will_line_is_the_ledgers_own(self) -> None:
        saga = _saga(STEERING_SAGAS, "w03_planning_goal_steers")
        ledger = _replay(saga, upto=1)
        assert ledger.will_line(RESPONSE_GOAL) is None  # setting turn
        ledger = _replay(saga, upto=2)
        assert ledger.will_line(RESPONSE_GOAL) == \
            "Standing goal: lantern festival route"

    def test_inject_will_over_a_replayed_saga(self) -> None:
        ledger = _replay(
            _saga(STEERING_SAGAS, "w01_declared_goal_steers"), upto=2)
        scaffold = CognitionScaffold(
            goal="turn goal", fact_block="Knowledge:\nx")
        assert inject_will(scaffold, ledger) is True
        assert scaffold.fact_block.startswith(
            "Standing goal: restore the ferry timetable")
        assert scaffold.goal == "turn goal"  # per-turn goal untouched
        assert inject_will(scaffold, ledger) is False  # never duplicated

    def test_replayed_ledger_survives_dict_round_trip(self) -> None:
        ledger = _replay(
            _saga(CLOSURE_SAGAS, "c05_new_declaration_supersedes"))
        twin = GoalLedger.from_dict(ledger.to_dict())
        assert twin.to_dict() == ledger.to_dict()
        assert [g.status for g in twin.closed] == ["superseded"]


# ── The suite run + its pinned baseline ─────────────────────────────────────


class TestSuiteRun:
    def test_full_run_passes_with_filed_scores(self) -> None:
        result = run_presence_suite()
        assert result.total_cases == 36
        assert result.coherence_score == pytest.approx(0.75)
        assert result.steering_score == pytest.approx(0.8333333333333334)
        assert result.closure_score == pytest.approx(0.75)
        assert result.overall_score == pytest.approx(0.7777777777777778)
        assert result.verdict == "PASS"
        assert result.passes is True
        assert AXIS_PASS_THRESHOLD == 0.75

    def test_empty_suite_scores_zero_and_reviews(self) -> None:
        result = run_presence_suite([], [], [])
        assert result.total_cases == 0
        assert result.overall_score == 0.0
        assert result.verdict == "REVIEW"
        assert result.passes is False
        assert score_coherence_sagas([])["score"] == 0.0
        assert score_steering_sagas([])["score"] == 0.0
        assert score_closure_sagas([])["score"] == 0.0

    def test_a_failing_axis_keeps_the_suite_at_review(self) -> None:
        bad = [_saga(CLOSURE_SAGAS,
                     "c10_abandoned_goal_still_steered_fails")] * 4
        result = run_presence_suite(closure_sagas=bad)
        assert result.closure_score == 0.0
        assert result.verdict == "REVIEW"

    def test_suite_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_presence_suite_baseline()
        second = measure_presence_suite_baseline()
        assert first == second
        assert first["coherence_score"] == pytest.approx(0.75)
        assert first["steering_score"] == pytest.approx(0.8333333333333334)
        assert first["closure_score"] == pytest.approx(0.75)
        assert first["overall_score"] == pytest.approx(0.7777777777777778)
        assert first["total_cases"] == 36
        assert baseline_problems("presence_suite_synthetic_v1") == []

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
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_suite_is_registered_with_fixture_and_baseline(self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "presence_suite"]
        assert len(entry) == 1
        assert entry[0]["bind_module"] == "benchmarks.presence_suite"
        assert entry[0]["fixture"] == "presence_suite_corpus"
        assert entry[0]["claims_axes"] == ("presence", "will")
        fixture = [f for f in FIXTURES
                   if f["id"] == "presence_suite_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "presence_suite_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cases"] == 36
        assert baseline[0]["targets"]["overall_score"] == \
            {"target": 0.75, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined(self) -> None:
        defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
        for metric in ("coherence_score", "steering_score",
                       "closure_score", "overall_score", "total_cases"):
            assert ("presence_suite", metric) in defined
        assert len(BENCHMARKS) == 13
        assert len(FIXTURES) == 6
        assert len(BASELINES) == 6
        assert len(METRIC_DEFINITIONS) == 41

    def test_will_gap_rule_now_files_tf125_done(self) -> None:
        rule_ids = {r["id"] for r in HARNESS_RULES}
        assert "the_will_axis_is_scored_by_tf125" in rule_ids
        assert "the_will_axis_has_no_scorer_yet" not in rule_ids
        assert harness.harness_problems() == []
        assert baseline_problems() == []


# ── Checker ─────────────────────────────────────────────────────────────────


class TestChecker:
    def test_presence_suite_checker_clean_on_the_real_estate(self) -> None:
        assert presence_suite_problems() == []

    def test_a_broken_corpus_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(suite, "STEERING_SAGAS", STEERING_SAGAS[:6])
        problems = presence_suite_problems()
        assert any("STEERING_SAGAS" in p for p in problems)


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

    def test_suite_world_is_synthetic_scripted_sagas(self) -> None:
        data = presence_suite_as_data()
        assert data["measured"]["overall_score"] == \
            pytest.approx(0.7777777777777778)
        joined = " ".join(s["id"] for s in ALL_SUITE_CASES)
        assert "drift" in joined and "goal" in joined
