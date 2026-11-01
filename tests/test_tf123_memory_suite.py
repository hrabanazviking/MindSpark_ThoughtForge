"""
TF-123 — Memory fidelity suite: cite-or-explain
scored mechanically (citations present, citations
true).

The proof lives as verifiable data in
``benchmarks/memory_suite.py`` (``RECORDS`` /
``RECORD_MARKERS``, ``MEMORY_FACTS``, ``PRESENT_CASES``
/ ``TRUE_CASES`` / ``EXPLAIN_CASES``, ``SUITE_AXES``,
``SUITE_RULES``, checked by
``memory_suite_problems()``), registered in
``benchmarks/harness.py`` with the pinned baseline
``memory_suite_synthetic_v1``; D-BENCH-9 (citation
truth and the explain side scored by nothing) was
added and fixed by this slice in the scoring sense
TF-049's doc assigned.

Pinned here:

  * the D-BENCH-9 probe, both ways: the cited-but-
    false claim passes TF-049's presence check and
    fails this suite's truth judge,
  * each axis scorer, including its edge cases (a
    missing citation, a wrong-sentence citation, a
    fabricated citation, a wrong-record citation,
    an unowned assertion, a disguised memory claim,
    empty corpora),
  * the filed corpus: 36 scored cases, the filed
    holds expectations true case by case,
  * the checker clean (incl. TF-043's live leg),
    the harness registration, and the new suite
    baseline re-measured live twice — with every
    earlier pinned baseline preserved exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no live model, never claimed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.harness as harness
import benchmarks.memory_suite as suite
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_memory_suite_baseline,
)
from benchmarks.memory_suite import (
    ALL_SUITE_CASES,
    AXIS_PASS_THRESHOLD,
    CASES_PER_AXIS,
    EXPLAIN_CASES,
    MEMORY_FACTS,
    MEMORY_SUITE_DOC,
    MEMORY_SUITE_FORMAT,
    OWNERSHIP_PHRASES,
    PRESENT_CASES,
    RECORDS,
    RECORD_MARKERS,
    SUITE_AXES,
    SUITE_RULES,
    TRUE_CASES,
    citation_truth,
    memory_suite_as_data,
    memory_suite_problems,
    run_memory_suite,
    score_explain_cases,
    score_present_cases,
    score_true_cases,
)
from thoughtforge.knowledge.citations import citation_for_record
from thoughtforge.refinement.citation_enforcement import check_citations

REPO_ROOT = Path(__file__).parent.parent


def _case(cases, case_id):
    return [c for c in cases if c["id"] == case_id][0]


def _records_of(case):
    return [RECORDS[key] for key in case["records"]]


# ── D-BENCH-9: the probe, pinned both ways ──────────────────────────────────


class TestTruthGapProbe:
    def test_cited_but_false_passes_tf049_and_fails_truth(self) -> None:
        case = _case(TRUE_CASES, "t10_cited_but_false_value")
        records = _records_of(case)
        # Presence: TF-049 passes it (the filed gap).
        assert check_citations(case["text"], records).passed is True
        # Truth: this suite's judge fails it.
        verdict = citation_truth(case["text"], records)
        assert verdict["true"] is False
        assert verdict["verdicts"][0]["breaking_present"] is True

    def test_the_true_twin_passes_both(self) -> None:
        case = _case(TRUE_CASES, "t01_ferry_dawn_true")
        records = _records_of(case)
        assert check_citations(case["text"], records).passed is True
        assert citation_truth(case["text"], records)["true"] is True

    def test_unowned_nonmemory_passes_tf049_and_fails_explain(self) -> None:
        case = _case(EXPLAIN_CASES, "e11_unowned_assertion")
        records = _records_of(case)
        assert check_citations(case["text"], records).passed is True
        assert score_explain_cases([case])["held"] == 0


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert MEMORY_SUITE_FORMAT == \
            "thoughtforge-memory-fidelity-suite/1"
        assert MEMORY_SUITE_DOC == \
            "docs/architecture/MEMORY_FIDELITY_SUITE.md"
        assert (REPO_ROOT / MEMORY_SUITE_DOC).exists()
        ids = [r["id"] for r in SUITE_RULES]
        assert len(ids) == len(set(ids)) == 6
        assert [a["id"] for a in SUITE_AXES] == [
            "citations_present", "citations_true", "explain"]
        data = memory_suite_as_data()
        assert data["format"] == MEMORY_SUITE_FORMAT
        assert data["case_counts"]["total"] == 36

    def test_records_are_tf043s_corpus_with_tf048_markers(self) -> None:
        assert RECORD_MARKERS == {
            "ferry_entity": "QF4301", "mill_entity": "QF4302",
            "observatory_entity": "QF4303",
            "lighthouse_entity": "QF4304",
            "ferry_chunk": "ref:4301", "mill_chunk": "ref:4302",
            "observatory_chunk": "ref:4303",
            "place": "geonames:430001",
            "ferry_edge": "conceptnet:ferry|crosses|river"}
        for key, record in RECORDS.items():
            assert record.citation is not None
            assert citation_for_record(record).marker == \
                RECORD_MARKERS[key]

    def test_facts_bind_records_with_required_and_breaking(self) -> None:
        assert len(MEMORY_FACTS) == 6
        for fact in MEMORY_FACTS:
            assert fact["record"] in RECORDS
            assert fact["required"] and fact["breaking"]

    def test_corpus_is_36_cases_balanced_per_axis(self) -> None:
        assert CASES_PER_AXIS == 12
        assert len(PRESENT_CASES) == len(TRUE_CASES) == \
            len(EXPLAIN_CASES) == 12
        assert len(ALL_SUITE_CASES) == 36
        assert sum(c["holds"] for c in PRESENT_CASES) == 10
        assert sum(c["holds"] for c in TRUE_CASES) == 9
        assert sum(c["holds"] for c in EXPLAIN_CASES) == 10
        ids = [c["id"] for c in ALL_SUITE_CASES]
        assert len(ids) == len(set(ids))


# ── The axis scorers ────────────────────────────────────────────────────────


class TestAxisScorers:
    def test_present_axis_scores_the_filed_corpus(self) -> None:
        axis = score_present_cases()
        assert axis["cases"] == 12
        assert axis["held"] == 10
        assert axis["score"] == pytest.approx(0.8333333333333334)

    def test_present_expectations_hold_case_by_case(self) -> None:
        for case in PRESENT_CASES:
            single = score_present_cases([case])
            assert bool(single["held"]) is case["holds"], case["id"]

    def test_present_failing_shapes(self) -> None:
        missing = _case(PRESENT_CASES, "p11_missing_citation")
        assert score_present_cases([missing])["held"] == 0
        check = check_citations(missing["text"], _records_of(missing))
        assert check.uncited_record_ids == ["4301"]
        wrong_sentence = _case(
            PRESENT_CASES, "p12_citation_in_wrong_sentence")
        assert score_present_cases([wrong_sentence])["held"] == 0

    def test_true_axis_scores_the_filed_corpus(self) -> None:
        axis = score_true_cases()
        assert axis["cases"] == 12
        assert axis["held"] == 9
        assert axis["score"] == pytest.approx(0.75)

    def test_true_expectations_hold_case_by_case(self) -> None:
        for case in TRUE_CASES:
            single = score_true_cases([case])
            assert bool(single["held"]) is case["holds"], case["id"]

    def test_true_failing_shapes(self) -> None:
        fabricated = _case(TRUE_CASES, "t11_fabricated_citation")
        verdict = citation_truth(
            fabricated["text"], _records_of(fabricated))
        assert verdict["true"] is False
        assert verdict["foreign_markers"] == ["ref:9999"]
        wrong_record = _case(TRUE_CASES, "t12_wrong_record_citation")
        verdict = citation_truth(
            wrong_record["text"], _records_of(wrong_record))
        assert verdict["true"] is False
        ferry = [v for v in verdict["verdicts"]
                 if v["marker"] == "ref:4301"][0]
        assert ferry["resolves"] is True
        assert ferry["matches_claim"] is False

    def test_truth_requires_a_citation(self) -> None:
        verdict = citation_truth(
            "The lantern hollow ferry departs dawn.",
            [RECORDS["ferry_chunk"]])
        assert verdict["true"] is False
        assert verdict["cited_markers"] == []

    def test_explain_axis_scores_the_filed_corpus(self) -> None:
        axis = score_explain_cases()
        assert axis["cases"] == 12
        assert axis["held"] == 10
        assert axis["score"] == pytest.approx(0.8333333333333334)

    def test_explain_expectations_hold_case_by_case(self) -> None:
        for case in EXPLAIN_CASES:
            single = score_explain_cases([case])
            assert bool(single["held"]) is case["holds"], case["id"]

    def test_explain_failing_shapes(self) -> None:
        disguised = _case(EXPLAIN_CASES, "e12_memory_claim_disguised")
        assert score_explain_cases([disguised])["held"] == 0
        check = check_citations(
            disguised["text"], _records_of(disguised))
        assert check.matched_record_ids  # it IS a memory claim
        cited = {**_case(EXPLAIN_CASES, "e01_owned_weather"),
                 "text": "In my own reasoning, spring rain makes "
                         "the harbour path muddy [ref:4301]."}
        assert score_explain_cases([cited])["held"] == 0

    def test_ownership_phrases_are_the_filed_vocabulary(self) -> None:
        assert "in my own reasoning" in OWNERSHIP_PHRASES
        assert "my own inference" in OWNERSHIP_PHRASES
        assert len(OWNERSHIP_PHRASES) == len(set(OWNERSHIP_PHRASES))


# ── The suite run + its pinned baseline ─────────────────────────────────────


class TestSuiteRun:
    def test_full_run_passes_with_filed_scores(self) -> None:
        result = run_memory_suite()
        assert result.total_cases == 36
        assert result.present_score == pytest.approx(0.8333333333333334)
        assert result.true_score == pytest.approx(0.75)
        assert result.explain_score == pytest.approx(0.8333333333333334)
        assert result.overall_score == pytest.approx(0.8055555555555557)
        assert result.verdict == "PASS"
        assert result.passes is True
        assert AXIS_PASS_THRESHOLD == 0.75

    def test_empty_suite_scores_zero_and_reviews(self) -> None:
        result = run_memory_suite([], [], [])
        assert result.total_cases == 0
        assert result.overall_score == 0.0
        assert result.verdict == "REVIEW"
        assert result.passes is False
        assert score_present_cases([])["score"] == 0.0
        assert score_true_cases([])["score"] == 0.0
        assert score_explain_cases([])["score"] == 0.0

    def test_a_failing_axis_keeps_the_suite_at_review(self) -> None:
        bad = [_case(TRUE_CASES, "t10_cited_but_false_value")] * 4
        result = run_memory_suite(true_cases=bad)
        assert result.true_score == 0.0
        assert result.verdict == "REVIEW"

    def test_suite_baseline_remeasures_live_twice_identical(self) -> None:
        first = measure_memory_suite_baseline()
        second = measure_memory_suite_baseline()
        assert first == second
        assert first["overall_score"] == pytest.approx(0.8055555555555557)
        assert first["total_cases"] == 36
        assert baseline_problems("memory_suite_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_suite_is_registered_with_fixture_and_baseline(self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "memory_suite"]
        assert len(entry) == 1
        assert entry[0]["bind_module"] == "benchmarks.memory_suite"
        assert entry[0]["fixture"] == "memory_suite_corpus"
        retrieval = [b for b in BENCHMARKS
                     if b["id"] == "retrieval_precision"][0]
        assert "TF-123 DONE" in retrieval["suite_slice"]
        fixture = [f for f in FIXTURES
                   if f["id"] == "memory_suite_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "memory_suite_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cases"] == 36
        assert baseline[0]["targets"]["overall_score"] == \
            {"target": 0.75, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_axis_metrics_are_defined(self) -> None:
        defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
        for metric in ("present_score", "true_score", "explain_score",
                       "overall_score", "total_cases"):
            assert ("memory_suite", metric) in defined
        assert len(BENCHMARKS) == 18  # + depth_suite (TF-124) + presence_suite (TF-125) + honesty_panel (TF-126) + quant_panel (TF-127) + profile_panel (TF-128) + regression_gate (TF-129) + spot_checks (TF-130), in the open
        assert len(FIXTURES) == 11  # + depth_suite_corpus (TF-124) + presence_suite_corpus (TF-125) + honesty_panel_corpus (TF-126) + quant_panel_corpus (TF-127) + profile_panel_corpus (TF-128) + regression_gate_corpus (TF-129) + spot_checks_corpus (TF-130), in the open
        assert len(BASELINES) == 11  # + depth_suite_synthetic_v1 (TF-124) + presence_suite_synthetic_v1 (TF-125) + honesty_panel_synthetic_v1 (TF-126) + quant_panel_synthetic_v1 (TF-127) + profile_panel_synthetic_v1 (TF-128) + regression_gate_synthetic_v1 (TF-129) + spot_checks_synthetic_v1 (TF-130), in the open
        assert len(METRIC_DEFINITIONS) == 95  # + 10 depth_suite metrics (TF-124) + 5 presence_suite metrics (TF-125) + 13 honesty_panel metrics (TF-126) + 16 quant_panel metrics (TF-127) + 13 profile_panel metrics (TF-128) + 6 regression_gate metrics (TF-129) + 6 spot_checks metrics (TF-130), in the open

    def test_harness_checker_stays_clean(self) -> None:
        assert harness.harness_problems() == []
        assert baseline_problems() == []


# ── Checker ─────────────────────────────────────────────────────────────────


class TestChecker:
    def test_memory_suite_checker_clean_on_the_real_estate(self) -> None:
        assert memory_suite_problems() == []

    def test_a_broken_corpus_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(suite, "TRUE_CASES", TRUE_CASES[:6])
        problems = memory_suite_problems()
        assert any("TRUE_CASES" in p for p in problems)


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

    def test_suite_world_is_tf043s_synthetic_corpus(self) -> None:
        data = memory_suite_as_data()
        assert data["record_markers"]["ferry_chunk"] == "ref:4301"
        joined = " ".join(c["text"] for c in ALL_SUITE_CASES)
        assert "Lantern Hollow" in joined or "lantern hollow" in joined
