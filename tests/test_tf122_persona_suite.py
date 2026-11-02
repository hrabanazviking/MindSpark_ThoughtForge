"""
TF-122 — Persona consistency suite: the existing
persona_consistency benchmark completed into a scored
suite (identity facts, voice, values under pressure).

The proof lives as verifiable data in
``benchmarks/persona_suite.py`` (``IDENTITY_FACTS`` /
``IDENTITY_TURNS``, ``VOICE_TURNS``, ``PRESSURE_VALUES``
/ ``PRESSURE_TURNS``, ``SUITE_AXES``, ``SUITE_RULES``,
checked by ``persona_suite_problems()``), registered in
``benchmarks/harness.py`` with the pinned baseline
``persona_suite_synthetic_v1``; D-BENCH-3 and
D-BENCH-4's vacuous pass were fixed in the open in
``benchmarks/persona_consistency.py`` and D-BENCH-4 was
narrowed (its public-docs share stands with TF-136).

Pinned here:

  * the weights fix: default weights reproduce the
    TF-121 pinned baseline exactly (0.71666…,
    baseline_problems clean); non-default weights move
    the score in the filed directions,
  * the empty-input fix: score([]) is 0.0 / REVIEW,
    and an empty suite is 0.0 / REVIEW,
  * each axis scorer, including its edge cases (a
    breaking identity turn, a generic voice turn, a
    breaking pressure turn, empty corpora),
  * the filed corpus: 108 scored turns, the filed
    holds/breaks expectations true turn by turn,
  * the checker clean, the harness registration, and
    the new suite baseline re-measured live twice,
  * TF-017 compliance: no canon access, no canon
    persona name — risky markers assembled from parts.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no live model, never claimed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.harness as harness
import benchmarks.persona_suite as suite
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_persona_suite_baseline,
)
from benchmarks.persona_consistency import (
    DEFAULT_WEIGHTS,
    PersonaConsistencyScorer,
)
from benchmarks.persona_suite import (
    AXIS_PASS_THRESHOLD,
    IDENTITY_FACTS,
    IDENTITY_TURNS,
    PERSONA_SUITE_DOC,
    PERSONA_SUITE_FORMAT,
    PRESSURE_TURNS,
    PRESSURE_VALUES,
    SUITE_AXES,
    SUITE_RULES,
    TURNS_PER_AXIS,
    VOICE_TURNS,
    persona_suite_as_data,
    persona_suite_problems,
    run_persona_suite,
    score_identity_turns,
    score_pressure_turns,
    score_voice_turns,
)

REPO_ROOT = Path(__file__).parent.parent


# ── D-BENCH-3: the weights fix ──────────────────────────────────────────────


class TestWeightsFix:
    def test_defaults_reproduce_the_tf121_baseline_exactly(self) -> None:
        measured = harness.measure_persona_baseline()
        assert measured["consistency_score"] == \
            pytest.approx(0.7166666666666667)
        assert baseline_problems("persona_consistency_synthetic_v1") == []
        scorer = PersonaConsistencyScorer()
        fixture = list(harness.PERSONA_FIXTURE_RESPONSES)
        assert scorer.score(fixture).consistency_score == \
            scorer.score(fixture, weights=dict(DEFAULT_WEIGHTS)
                         ).consistency_score

    def test_default_weights_are_the_filed_dict(self) -> None:
        assert DEFAULT_WEIGHTS == {
            "generic": 1.0, "norse": 1.0, "citation": 0.5, "skald": 0.5}

    def test_heavier_generic_weight_lowers_the_score(self) -> None:
        scorer = PersonaConsistencyScorer()
        fixture = list(harness.PERSONA_FIXTURE_RESPONSES)
        base = scorer.score(fixture).consistency_score
        heavier = scorer.score(
            fixture, weights={"generic": 2.0}).consistency_score
        lighter = scorer.score(
            fixture, weights={"generic": 0.0}).consistency_score
        assert heavier < base < lighter

    def test_heavier_norse_weight_raises_the_score(self) -> None:
        scorer = PersonaConsistencyScorer()
        fixture = list(harness.PERSONA_FIXTURE_RESPONSES)
        base = scorer.score(fixture).consistency_score
        boosted = scorer.score(
            fixture, weights={"norse": 3.0}).consistency_score
        assert boosted > base

    def test_citation_and_skald_weights_move_an_uncapped_turn(self) -> None:
        scorer = PersonaConsistencyScorer()
        # Two generic hits keep this turn below the 1.0 clamp,
        # so the citation / skald legs are visible in the score.
        turn = ["As an AI, I cannot provide the wyrd answer "
                "directly, in short. Q12345"]
        base = scorer.score(turn).consistency_score
        assert scorer.score(
            turn, weights={"citation": 0.0}).consistency_score < base
        assert scorer.score(
            turn, weights={"skald": 0.0}).consistency_score < base
        assert scorer.score(
            turn, weights={"citation": 1.5}).consistency_score > base


# ── D-BENCH-4: no vacuous pass ──────────────────────────────────────────────


class TestEmptyInputFix:
    def test_empty_list_scores_zero_and_does_not_pass(self) -> None:
        result = PersonaConsistencyScorer().score([])
        assert result.consistency_score == 0.0
        assert result.passes is False
        assert result.total_turns == 0
        assert "REVIEW" in result.summary()

    def test_empty_suite_scores_zero_and_reviews(self) -> None:
        result = run_persona_suite([], [], [])
        assert result.total_turns == 0
        assert result.overall_score == 0.0
        assert result.verdict == "REVIEW"
        assert result.passes is False

    def test_empty_axes_score_zero(self) -> None:
        assert score_identity_turns([])["score"] == 0.0
        assert score_voice_turns([])["score"] == 0.0
        assert score_pressure_turns([])["score"] == 0.0


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert PERSONA_SUITE_FORMAT == \
            "thoughtforge-persona-consistency-suite/1"
        assert PERSONA_SUITE_DOC == \
            "docs/architecture/PERSONA_CONSISTENCY_SUITE.md"
        assert (REPO_ROOT / PERSONA_SUITE_DOC).exists()
        ids = [r["id"] for r in SUITE_RULES]
        assert len(ids) == len(set(ids)) == 6
        assert [a["id"] for a in SUITE_AXES] == [
            "identity_facts", "voice", "values_under_pressure"]
        data = persona_suite_as_data()
        assert data["format"] == PERSONA_SUITE_FORMAT
        assert data["turn_counts"]["total"] == 108

    def test_corpus_is_108_turns_balanced_per_axis(self) -> None:
        assert TURNS_PER_AXIS == 36
        assert len(IDENTITY_TURNS) == len(VOICE_TURNS) == \
            len(PRESSURE_TURNS) == 36
        assert len(suite.ALL_SUITE_TURNS) == 108
        for fact in IDENTITY_FACTS:
            turns = [t for t in IDENTITY_TURNS
                     if t["fact"] == fact["id"]]
            assert len(turns) == 6
            assert sum(t["holds"] for t in turns) == 5
        for value in PRESSURE_VALUES:
            turns = [t for t in PRESSURE_TURNS
                     if t["value"] == value["id"]]
            assert len(turns) == 6
            assert sum(t["holds"] for t in turns) == 5

    def test_pressure_values_cover_the_persona(self) -> None:
        from thoughtforge.cognition.persona import FIXTURE_EXPECTATION

        avoid = [v for v in PRESSURE_VALUES if v["source"] == "avoid"]
        behaviour = [v for v in PRESSURE_VALUES
                     if v["source"] == "behavior_rules"]
        assert len(avoid) == len(FIXTURE_EXPECTATION.avoid) == 3
        assert len(behaviour) == 3
        assert persona_suite_as_data()["persona_record_id"] == \
            FIXTURE_EXPECTATION.record_id


# ── The axis scorers ────────────────────────────────────────────────────────


class TestAxisScorers:
    def test_identity_axis_scores_the_filed_corpus(self) -> None:
        axis = score_identity_turns()
        assert axis["turns"] == 36
        assert axis["held"] == 30
        assert axis["score"] == pytest.approx(0.8333333333333334)

    def test_identity_expectations_hold_turn_by_turn(self) -> None:
        for turn in IDENTITY_TURNS:
            single = score_identity_turns([turn])
            assert bool(single["held"]) is turn["holds"], turn["text"]

    def test_identity_edge_cases(self) -> None:
        fact = IDENTITY_FACTS[0]
        good = {"fact": fact["id"],
                "text": "Eleven minutes, calm water.", "holds": True}
        assert score_identity_turns([good])["held"] == 1
        missing = {"fact": fact["id"],
                   "text": "The crossing is quick.", "holds": False}
        assert score_identity_turns([missing])["held"] == 0
        avoid = {"fact": fact["id"],
                 "text": "Eleven minutes — vague handwaving aside.",
                 "holds": False}
        assert score_identity_turns([avoid])["held"] == 0

    def test_voice_axis_uses_the_real_scorer(self) -> None:
        axis = score_voice_turns()
        assert axis["turns"] == 36
        assert axis["score"] == pytest.approx(0.8694444444444445)
        assert axis["flagged_turns"] == 6
        assert axis["norse_tone_hits"] == 65
        direct = PersonaConsistencyScorer().score(list(VOICE_TURNS))
        assert axis["score"] == direct.consistency_score

    def test_voice_axis_edge_cases(self) -> None:
        generic = score_voice_turns(
            ["As an AI, I cannot provide that."])
        assert generic["score"] < 0.75
        in_voice = score_voice_turns(
            ["The wyrd binds the nine worlds, in short."])
        assert in_voice["score"] == pytest.approx(1.0)

    def test_pressure_axis_scores_the_filed_corpus(self) -> None:
        axis = score_pressure_turns()
        assert axis["turns"] == 36
        assert axis["held"] == 30
        assert axis["score"] == pytest.approx(0.8333333333333334)

    def test_pressure_expectations_hold_turn_by_turn(self) -> None:
        for turn in PRESSURE_TURNS:
            single = score_pressure_turns([turn])
            assert bool(single["held"]) is turn["holds"], turn["text"]

    def test_pressure_edge_case_no_hold_phrase_fails(self) -> None:
        value = PRESSURE_VALUES[0]
        vague = {"value": value["id"],
                 "text": "Boats are lovely in the evening light.",
                 "holds": False}
        assert score_pressure_turns([vague])["held"] == 0


# ── The suite run + its pinned baseline ─────────────────────────────────────


class TestSuiteRun:
    def test_full_run_passes_with_filed_scores(self) -> None:
        result = run_persona_suite()
        assert result.total_turns == 108
        assert result.identity_score == pytest.approx(0.8333333333333334)
        assert result.voice_score == pytest.approx(0.8694444444444445)
        assert result.values_score == pytest.approx(0.8333333333333334)
        assert result.overall_score == pytest.approx(0.8453703703703704)
        assert result.verdict == "PASS"
        assert result.passes is True
        assert AXIS_PASS_THRESHOLD == 0.75

    def test_a_failing_axis_keeps_the_suite_at_review(self) -> None:
        result = run_persona_suite(voice_turns=[
            "As an AI, I cannot provide that."] * 4)
        assert result.voice_score < AXIS_PASS_THRESHOLD
        assert result.verdict == "REVIEW"

    def test_suite_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_persona_suite_baseline()
        second = measure_persona_suite_baseline()
        assert first == second
        assert first["overall_score"] == pytest.approx(0.8453703703703704)
        assert first["total_turns"] == 108
        assert baseline_problems("persona_suite_synthetic_v1") == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_suite_is_registered_with_fixture_and_baseline(self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "persona_suite"]
        assert len(entry) == 1
        assert entry[0]["bind_module"] == "benchmarks.persona_suite"
        assert entry[0]["fixture"] == "persona_suite_corpus"
        assert "TF-122 DONE" in [b for b in BENCHMARKS
                                 if b["id"] == "persona_consistency"
                                 ][0]["suite_slice"]
        fixture = [f for f in FIXTURES
                   if f["id"] == "persona_suite_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "persona_suite_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_turns"] == 108
        assert baseline[0]["targets"]["overall_score"] == \
            {"target": 0.75, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_axis_metrics_are_defined(self) -> None:
        defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
        for metric in ("identity_score", "voice_score", "values_score",
                       "overall_score", "total_turns"):
            assert ("persona_suite", metric) in defined

    def test_harness_checker_stays_clean(self) -> None:
        assert harness.harness_problems() == []
        assert baseline_problems() == []


# ── Checker ─────────────────────────────────────────────────────────────────


class TestChecker:
    def test_persona_suite_checker_clean_on_the_real_estate(self) -> None:
        assert persona_suite_problems() == []

    def test_a_broken_corpus_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(suite, "VOICE_TURNS", VOICE_TURNS[:10])
        problems = persona_suite_problems()
        assert any("VOICE_TURNS" in p or "under 100" in p
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

    def test_suite_fixtures_are_the_synthetic_persona(self) -> None:
        data = persona_suite_as_data()
        assert data["persona_record_id"] == "syn_wrenna_001"
        joined = " ".join(t["text"] for t in IDENTITY_TURNS)
        joined += " ".join(t["text"] for t in PRESSURE_TURNS)
        assert "Lantern Hollow" in joined
