"""
TF-130 — Human-shaped spot checks: a small set
of transcripts Volmarr can read and judge
himself (the numbers' sanity anchor).

The proof lives as verifiable data in
``benchmarks/spot_checks.py`` (``TRANSCRIPT_SPECS``,
``SPOT_CHECK_RULES``, checked by
``spot_checks_problems()``), registered in
``benchmarks/harness.py`` with the pinned
baseline ``spot_checks_synthetic_v1``; D-BENCH-15
(the suites' numbers existed as pinned baselines
and checkers, but no artefact ever assembled
suite cases into transcripts a human could read
and judge — the numbers had no sanity anchor)
was fixed by this slice in the measuring sense.

Pinned here:

  * the filed set: 8 transcripts over 5 suite
    anchors, 56 verbatim turns, 4 holding / 4
    not, each carrying a judgement prompt,
  * verbatim fidelity: every transcript's turns
    equal the source suite's filed case data,
    re-derived from the source collections,
  * score reproduction: every score recomputed
    by its own suite's real per-case scorer and
    matching the filed expectation exactly,
  * the verdict shape: full set PASS, partial
    and empty selections REVIEW (no vacuous
    pass), a wrong expectation FAILs the full
    run,
  * the checker clean, the harness registration
    (18 / 11 / 11 / 95, rules 11), the new
    baseline re-measured live twice — with every
    earlier pinned baseline preserved exactly
    and TF-129's gated set untouched (the
    spot-check baseline is ungated, in the open),
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof
grade: FIXTURE — no live model was read, never
claimed (TF-173).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

import benchmarks.spot_checks as spot
import benchmarks.harness as harness
from benchmarks import depth_suite, memory_suite
from benchmarks import persona_suite, presence_suite
from benchmarks.spot_checks import (
    SPOT_CHECKS_BASELINE_ID,
    SPOT_CHECKS_DOC,
    SPOT_CHECKS_FORMAT,
    SPOT_CHECK_RULES,
    SPOT_CHECK_SUITES,
    TOTAL_TRANSCRIPTS,
    TRANSCRIPT_SPECS,
    build_spot_checks,
    render_spot_checks_markdown,
    run_spot_checks,
    spot_checks_as_data,
    spot_checks_problems,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_RULES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_spot_checks_baseline,
)
from benchmarks.regression_gate import (
    GATED_BASELINE_IDS,
    UNGATED_BASELINE_IDS,
    run_regression_gate,
)
from thoughtforge.inference.quant_quality import (
    QUANT_PROMPTS,
    fixture_generate,
    score_answer,
)

REPO_ROOT = Path(__file__).parent.parent


def _by_id(transcripts):
    return {t["id"]: t for t in transcripts}


# ── The filed set ───────────────────────────────────────────────────────────


class TestFiledSet:
    def test_eight_transcripts_over_five_anchors_balanced(self) -> None:
        assert TOTAL_TRANSCRIPTS == 8
        assert len(TRANSCRIPT_SPECS) == 8
        ids = [s["id"] for s in TRANSCRIPT_SPECS]
        assert len(ids) == len(set(ids))
        assert list(SPOT_CHECK_SUITES) == [
            "persona_suite", "memory_suite", "depth_suite",
            "presence_suite", "quant_quality"]
        holding = [s for s in TRANSCRIPT_SPECS if s["expected_held"]]
        assert len(holding) == 4
        assert len(TRANSCRIPT_SPECS) - len(holding) == 4

    def test_every_spec_carries_a_judgement_prompt(self) -> None:
        for spec in TRANSCRIPT_SPECS:
            assert spec["judgement_prompt"].strip()
            assert spec["case_ref"].strip()
            assert spec["title"].strip()


# ── Verbatim fidelity ───────────────────────────────────────────────────────


class TestVerbatimFidelity:
    def test_turns_equal_the_source_corpus_data(self) -> None:
        transcripts = _by_id(build_spot_checks())
        # persona identity: the filed turn itself.
        turn = persona_suite.IDENTITY_TURNS[0]
        assert transcripts["sc01_persona_identity_holds"]["turns"] == [
            {"speaker": "Assistant", "text": turn["text"]}]
        # persona voice: the filed line itself.
        assert transcripts["sc02_persona_voice_generic_slip"][
            "turns"][0]["text"] == persona_suite.VOICE_TURNS[30]
        # memory: the filed cases themselves.
        present = [c for c in memory_suite.PRESENT_CASES
                   if c["id"] == "p01_ferry_chunk_cited"][0]
        assert transcripts["sc03_memory_citation_present_and_true"][
            "turns"][0]["text"] == present["text"]
        true_case = [c for c in memory_suite.TRUE_CASES
                     if c["id"] == "t10_cited_but_false_value"][0]
        assert transcripts[
            "sc04_memory_citation_present_but_false"][
                "turns"][0]["text"] == true_case["text"]
        # depth: question + both roads' filed material.
        case = [c for c in depth_suite.CHAIN_CASES
                if c["id"] == "c01_precedence_chain"][0]
        texts = [t["text"] for t in transcripts[
            "sc05_depth_scaffold_lands_the_sum"]["turns"]]
        assert texts == [case["question"], case["off_answer"],
                         case["working"], case["on_answer"]]
        # presence coherence: TF-028's script via the suite builder.
        saga = [s for s in presence_suite.COHERENCE_SAGAS
                if s["id"] == "k07_drift_from_15_break"][0]
        expected: list[str] = []
        for user, assistant in presence_suite.transcript_for_saga(saga):
            expected.extend([user, assistant])
        texts = [t["text"] for t in transcripts[
            "sc06_presence_coherence_breaks_at_15"]["turns"]]
        assert texts == expected and len(texts) == 40
        # presence steering: the saga's filed turns.
        w01 = [s for s in presence_suite.STEERING_SAGAS
               if s["id"] == "w01_declared_goal_steers"][0]
        expected = []
        for turn in w01["turns"]:
            expected.extend([turn["user"], turn["assistant"]])
        texts = [t["text"] for t in transcripts[
            "sc07_presence_goal_keeps_steering"]["turns"]]
        assert texts == expected
        # quant: TF-067's prompt + its fixture generation.
        prompt = [p for p in QUANT_PROMPTS
                  if p.prompt_id == "fare_maths"][0]
        texts = [t["text"] for t in transcripts[
            "sc08_quant_plausible_working_wrong_total"]["turns"]]
        assert texts == [prompt.text,
                         fixture_generate("q4_k_m", prompt)]


# ── Score reproduction ──────────────────────────────────────────────────────


class TestScoreReproduction:
    def test_scores_recompute_through_the_real_scorers(self) -> None:
        transcripts = _by_id(build_spot_checks())
        identity = persona_suite.score_identity_turns(
            [persona_suite.IDENTITY_TURNS[0]])
        assert transcripts["sc01_persona_identity_holds"][
            "score"] == identity["score"] == 1.0
        voice = persona_suite.score_voice_turns(
            [persona_suite.VOICE_TURNS[30]])
        sc02 = transcripts["sc02_persona_voice_generic_slip"]
        assert sc02["score"] == voice["score"] == pytest.approx(0.3)
        assert sc02["detail"]["flagged_turns"] == 1
        assert sc02["held"] is False
        true_case = [c for c in memory_suite.TRUE_CASES
                     if c["id"] == "t10_cited_but_false_value"][0]
        assert memory_suite.score_true_cases([true_case])[
            "score"] == 0.0
        assert transcripts[
            "sc04_memory_citation_present_but_false"][
                "score"] == 0.0
        chain = [c for c in depth_suite.CHAIN_CASES
                 if c["id"] == "c01_precedence_chain"][0]
        scored = depth_suite.score_chain_cases([chain])
        sc05 = transcripts["sc05_depth_scaffold_lands_the_sum"]
        assert sc05["score"] == scored["on_score"] == 1.0
        assert sc05["detail"]["off_score"] == 0.0
        saga = [s for s in presence_suite.COHERENCE_SAGAS
                if s["id"] == "k07_drift_from_15_break"][0]
        report = presence_suite.coherence_saga_report(saga)
        sc06 = transcripts["sc06_presence_coherence_breaks_at_15"]
        assert sc06["score"] == report["mean_score"] == \
            pytest.approx(0.7)
        assert sc06["held"] is False
        assert sc06["detail"]["contradiction_turns"] == 6
        prompt = [p for p in QUANT_PROMPTS
                  if p.prompt_id == "fare_maths"][0]
        scored_q = score_answer(
            prompt, fixture_generate("q4_k_m", prompt))
        sc08 = transcripts[
            "sc08_quant_plausible_working_wrong_total"]
        assert sc08["score"] == scored_q.score == 0.0
        assert "18 coins" in sc08["detail"]["contradictions"]

    def test_every_transcript_matches_its_filed_expectation(
            self) -> None:
        for transcript in build_spot_checks():
            assert transcript["score"] == \
                transcript["expected_score"]
            assert transcript["held"] is \
                transcript["expected_held"]
            assert transcript["verdict"] == (
                "HOLDS" if transcript["held"]
                else "DOES NOT HOLD")

    def test_building_twice_is_identical(self) -> None:
        assert build_spot_checks() == build_spot_checks()


# ── The run + verdict shape ─────────────────────────────────────────────────


class TestRunShape:
    def test_full_set_passes_with_filed_counts(self) -> None:
        result = run_spot_checks()
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.total_transcripts == 8
        assert result.total_turns == 56
        assert result.suites_covered == 5
        assert result.holding == 4
        assert result.failing == 4
        assert result.mean_score == pytest.approx(0.625)
        assert result.mismatches == 0

    def test_empty_selection_builds_zero_and_reviews(self) -> None:
        empty = run_spot_checks(transcript_ids=())
        assert empty.total_transcripts == 0
        assert empty.total_turns == 0
        assert empty.mean_score == 0.0
        assert empty.verdict == "REVIEW"
        assert empty.passes is False

    def test_a_partial_selection_builds_and_reviews(self) -> None:
        subset = run_spot_checks(
            transcript_ids=("sc01_persona_identity_holds",))
        assert subset.total_transcripts == 1
        assert subset.holding == 1
        assert subset.verdict == "REVIEW"
        assert subset.passes is False

    def test_a_wrong_expectation_fails_the_full_run(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = copy.deepcopy([dict(s) for s in TRANSCRIPT_SPECS])
        broken[0]["expected_score"] = 0.5
        monkeypatch.setattr(spot, "TRANSCRIPT_SPECS", tuple(broken))
        result = run_spot_checks()
        assert result.mismatches == 1
        assert result.verdict == "FAIL"
        assert result.passes is False


# ── The render + the doc ────────────────────────────────────────────────────


class TestRenderAndDoc:
    def test_render_carries_every_turn_and_prompt(self) -> None:
        rendered = render_spot_checks_markdown()
        for transcript in build_spot_checks():
            assert transcript["id"] in rendered
            assert transcript["judgement_prompt"] in rendered
            for turn in transcript["turns"]:
                assert turn["text"] in rendered
        assert rendered.count("**Your judgement:**") == 8

    def test_the_doc_is_the_set_verbatim(self) -> None:
        doc_text = (REPO_ROOT / SPOT_CHECKS_DOC).read_text(
            encoding="utf-8")
        for transcript in build_spot_checks():
            assert transcript["id"] in doc_text
            for turn in transcript["turns"]:
                assert turn["text"] in doc_text


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_spot_checks_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "spot_checks"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.spot_checks"
        assert entry[0]["bind_attribute"] == "run_spot_checks"
        assert entry[0]["fixture"] == "spot_checks_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence",
                                           "will")
        assert any("D-BENCH-15 FIXED by TF-130" in d
                   for d in entry[0]["docket"])
        fixture = [f for f in FIXTURES
                   if f["id"] == "spot_checks_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        assert fixture[0]["bind_attribute"] == "TRANSCRIPT_SPECS"
        baseline = [b for b in BASELINES
                    if b["id"] == "spot_checks_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_transcripts"] == 8
        assert baseline[0]["metrics"]["total_turns"] == 56
        assert baseline[0]["metrics"]["mean_score"] == \
            pytest.approx(0.625)
        assert baseline[0]["targets"]["holding_transcripts"] == {
            "target": 4, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for metric in ("total_transcripts", "total_turns",
                       "suites_covered", "holding_transcripts",
                       "failing_transcripts", "mean_score"):
            assert ("spot_checks", metric) in defined
        assert len(BENCHMARKS) == 25
        assert len(FIXTURES) == 13
        assert len(BASELINES) == 18
        assert len(METRIC_DEFINITIONS) == 146
        assert len(HARNESS_RULES) == 18
        assert "spot_checks_anchor_the_numbers_in_human_judgement" \
            in {r["id"] for r in HARNESS_RULES}
        assert harness.harness_problems() == []

    def test_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_spot_checks_baseline()
        second = measure_spot_checks_baseline()
        assert first == second
        assert first["total_transcripts"] == 8
        assert first["total_turns"] == 56
        assert first["suites_covered"] == 5
        assert first["holding_transcripts"] == 4
        assert first["failing_transcripts"] == 4
        assert first["mean_score"] == pytest.approx(0.625)
        assert baseline_problems(
            "spot_checks_synthetic_v1") == []

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
        assert baseline_problems() == []

    def test_the_spot_check_baseline_is_not_gated(self) -> None:
        assert SPOT_CHECKS_BASELINE_ID == \
            "spot_checks_synthetic_v1"
        assert SPOT_CHECKS_BASELINE_ID not in GATED_BASELINE_IDS
        assert SPOT_CHECKS_BASELINE_ID in UNGATED_BASELINE_IDS
        assert len(GATED_BASELINE_IDS) == 9
        assert run_regression_gate().verdict == "PASS"


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert SPOT_CHECKS_FORMAT == "thoughtforge-spot-checks/1"
        assert SPOT_CHECKS_DOC == \
            "docs/architecture/SPOT_CHECKS.md"
        assert (REPO_ROOT / SPOT_CHECKS_DOC).exists()
        ids = [r["id"] for r in SPOT_CHECK_RULES]
        assert len(ids) == len(set(ids)) == 9
        for needed in (
                "transcripts_are_the_suites_own_cases_verbatim",
                "scores_are_recomputed_by_the_real_scorers",
                "the_set_holds_passes_and_failures_side_by_side",
                "the_doc_is_the_set",
                "the_spot_check_baseline_is_not_gated_by_tf129"):
            assert needed in ids
        data = spot_checks_as_data()
        assert data["format"] == SPOT_CHECKS_FORMAT
        assert data["suites"] == list(SPOT_CHECK_SUITES)
        assert data["measured"]["verdict"] == "PASS"
        assert data["measured"]["total_transcripts"] == 8
        assert data["measured"]["total_turns"] == 56
        assert data["measured"]["holding_transcripts"] == 4
        assert data["measured"]["failing_transcripts"] == 4
        assert data["measured"]["mean_score"] == \
            pytest.approx(0.625)

    def test_spot_checks_checker_clean_on_the_real_estate(
            self) -> None:
        assert spot_checks_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(spot, "TRANSCRIPT_SPECS", ())
        problems = spot_checks_problems()
        assert any("transcripts" in p or "TRANSCRIPT" in p
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

    def test_spot_check_world_is_the_suites_filed_corpora(
            self) -> None:
        data = spot_checks_as_data()
        assert data["suites"] == [
            "persona_suite", "memory_suite", "depth_suite",
            "presence_suite", "quant_quality"]
        assert all(t["turns"] for t in data["transcripts"])
