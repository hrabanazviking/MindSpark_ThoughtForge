"""
TF-043 — Retrieval precision fixtures: a synthetic corpus
with known right answers; recall/precision measured, baseline
pinned.

The fixture lives as data + measurement in
``src/thoughtforge/knowledge/retrieval_precision.py``
(``FIXTURE_ENTITIES`` / ``FIXTURE_REFERENCE_CHUNKS`` /
``FIXTURE_CONCEPTNET_EDGES`` / ``FIXTURE_GEONAMES`` /
``FIXTURE_QUERIES``, built by ``build_fixture_db`` and measured
by ``evaluate_retrieval``); the write-up is
``docs/architecture/RETRIEVAL_PRECISION.md``.

Pinned here:

  * the corpus and queries are complete as data: 8 entities
    (5 targets + 3 shared-term distractors), 4 reference
    chunks, 2 ConceptNet edges, 2 GeoNames places, filler as
    stated, and 9 queries whose relevant sets name bundle ids
    that exist in the corpus (chunk ids as strings, a
    ConceptNet concept1, a GeoNames name),
  * the metric rules are the stated ones (pure-function
    cases, including the negative-query convention: correctly
    returning nothing scores 1/1, returning junk scores
    recall 0),
  * at the defaults the core runs (sql, top_k 10, min_score
    0.45) the measured aggregates stand at the pinned
    BASELINE, every non-geonames known answer is recalled,
    and the first returned record is a right answer for every
    answered query (MRR 1.0),
  * the two findings are pinned live, not narrated: a
    GeoNames record's composite is exactly 0.285, so the
    place answer is absent at the defaults and present at
    min_score 0 (the geonames pass is threshold-dead at the
    defaults); the weight-4.0 ConceptNet ferry edge scores
    exactly 0.46 and rides every ferry/river query, holding
    three queries' precision to exactly 2/3,
  * the open setting (min_score 0) trades exactly as
    measured: recall rises to 1.000, precision falls to the
    pinned open baseline as the sub-threshold records enter,
  * ``baseline_problems`` names a regressed report and
    passes the real one (TF-064's gate consumes it), the
    measurement is deterministic across two independently
    built fixture DBs, and top_k truncates recall exactly
    as the cap says.

Proof grade: FIXTURE (synthetic corpus, SQL path, this VM).
Vector/hybrid quality NOT PROVEN — sentence-transformers is
absent here (D-ENV-1); TF-053 owns embedding availability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.retrieval_precision import (
    BASELINE,
    CONCEPTNET_FERRY_EDGE_SCORE,
    ENTITY_FILLER_COUNT,
    FIXTURE_CONCEPTNET_EDGES,
    FIXTURE_ENTITIES,
    FIXTURE_GEONAMES,
    FIXTURE_QUERIES,
    FIXTURE_REFERENCE_CHUNKS,
    GEONAMES_DEFAULT_SCORE,
    REFERENCE_FILLER_COUNT,
    baseline_problems,
    build_fixture_db,
    evaluate_retrieval,
    score_query,
)
from thoughtforge.knowledge.scoring import MIN_MEMORY_SCORE

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "architecture" / "RETRIEVAL_PRECISION.md"


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_db(tmp_path_factory.mktemp("tf043") / "fixture.db")


@pytest.fixture(scope="module")
def default_report(fixture_db: Path) -> dict:
    return evaluate_retrieval(fixture_db)


@pytest.fixture(scope="module")
def open_report(fixture_db: Path) -> dict:
    return evaluate_retrieval(fixture_db, min_score=0.0)


def by_id(report: dict) -> dict[str, dict]:
    return {q["id"]: q for q in report["per_query"]}


# ── The fixture as data ──────────────────────────────────────────────────────


class TestFixtureData:
    def test_corpus_shape(self) -> None:
        assert len(FIXTURE_ENTITIES) == 8
        assert len(FIXTURE_REFERENCE_CHUNKS) == 4
        assert len(FIXTURE_CONCEPTNET_EDGES) == 2
        assert len(FIXTURE_GEONAMES) == 2
        assert ENTITY_FILLER_COUNT == 60
        assert REFERENCE_FILLER_COUNT == 30

    def test_queries_shape_and_kinds(self) -> None:
        assert [q["id"] for q in FIXTURE_QUERIES] == [
            f"Q{i}" for i in range(1, 10)
        ]
        assert {q["kind"] for q in FIXTURE_QUERIES} == {
            "entity+reference", "entity", "shared-term", "negative",
            "conceptnet", "distractor-as-target", "geonames",
        }

    def test_relevant_ids_exist_in_the_corpus(self) -> None:
        known = (
            {row[0] for row in FIXTURE_ENTITIES}
            | {str(row[0]) for row in FIXTURE_REFERENCE_CHUNKS}
            | {row[0] for row in FIXTURE_CONCEPTNET_EDGES}
            | {row[1] for row in FIXTURE_GEONAMES}
        )
        for q in FIXTURE_QUERIES:
            assert set(q["relevant"]) <= known, q["id"]

    def test_negative_query_is_labelled_and_empty(self) -> None:
        negative = [q for q in FIXTURE_QUERIES if q["kind"] == "negative"]
        assert len(negative) == 1
        assert negative[0]["relevant"] == ()

    def test_doc_carries_the_baseline(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        assert "0.8889" in text and "0.9630" in text
        assert "threshold-dead" in text


# ── The metric rules (pure) ──────────────────────────────────────────────────


class TestScoreQuery:
    def test_perfect(self) -> None:
        s = score_query(["a", "b"], ("a", "b"))
        assert (s["precision"], s["recall"], s["f1"]) == (1.0, 1.0, 1.0)
        assert s["reciprocal_rank"] == 1.0

    def test_partial(self) -> None:
        s = score_query(["x", "a"], ("a", "b"))
        assert s["precision"] == 0.5 and s["recall"] == 0.5
        assert s["reciprocal_rank"] == 0.5

    def test_empty_return_against_answers_scores_zero(self) -> None:
        s = score_query([], ("a",))
        assert s["precision"] == 0.0 and s["recall"] == 0.0

    def test_negative_query_convention(self) -> None:
        assert score_query([], ())["precision"] == 1.0
        assert score_query([], ())["recall"] == 1.0
        assert score_query(["junk"], ())["recall"] == 0.0
        assert score_query(["junk"], ())["precision"] == 0.0


# ── The default baseline (the settings the core runs) ────────────────────────


class TestDefaultBaseline:
    def test_aggregates_match_the_pinned_baseline(
        self, default_report: dict
    ) -> None:
        agg = default_report["aggregate"]
        for metric, pinned in BASELINE["default"].items():
            assert agg[metric] == pytest.approx(pinned, abs=0.005), metric
        assert default_report["min_score"] == MIN_MEMORY_SCORE
        assert baseline_problems(default_report, setting="default") == []

    def test_every_answered_query_opens_with_a_right_answer(
        self, default_report: dict
    ) -> None:
        for q in default_report["per_query"]:
            if q["kind"] == "negative":
                assert q["returned"] == []
            else:
                assert q["reciprocal_rank"] == 1.0, q["id"]

    def test_recall_is_perfect_except_the_geonames_query(
        self, default_report: dict
    ) -> None:
        queries = by_id(default_report)
        for qid, q in queries.items():
            if qid == "Q9":
                assert q["recall"] == pytest.approx(2 / 3)
            else:
                assert q["recall"] == 1.0, qid
        assert default_report["aggregate"]["perfect_recall_queries"] == 8

    def test_conceptnet_edge_holds_three_queries_to_two_thirds(
        self, default_report: dict
    ) -> None:
        queries = by_id(default_report)
        for qid in ("Q1", "Q5", "Q8"):
            assert queries[qid]["precision"] == pytest.approx(2 / 3), qid
            assert queries[qid]["returned"][-1] == "ferry"
        for qid in ("Q2", "Q3", "Q4", "Q7"):
            assert queries[qid]["precision"] == 1.0, qid


# ── The two findings, pinned live ────────────────────────────────────────────


class TestFindings:
    def test_geonames_record_scores_exactly_its_ceiling(
        self, fixture_db: Path
    ) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("Lantern Hollow", path="sql", min_score=0.0)
        places = [r for r in bundle.activated_records
                  if r.record_id == "Lantern Hollow"]
        assert len(places) == 1
        assert places[0].score == GEONAMES_DEFAULT_SCORE
        assert GEONAMES_DEFAULT_SCORE < MIN_MEMORY_SCORE

    def test_geonames_answer_absent_at_defaults_present_open(
        self, default_report: dict, open_report: dict
    ) -> None:
        assert "Lantern Hollow" not in by_id(default_report)["Q9"]["returned"]
        assert "Lantern Hollow" in by_id(open_report)["Q9"]["returned"]
        assert by_id(open_report)["Q9"]["recall"] == 1.0

    def test_conceptnet_ferry_edge_scores_exactly_046(
        self, fixture_db: Path
    ) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("ferry crosses river", path="sql")
        assert [r.record_id for r in bundle.activated_records] == ["ferry"]
        assert bundle.activated_records[0].score == CONCEPTNET_FERRY_EDGE_SCORE


# ── The open setting: the threshold's trade, measured ────────────────────────


class TestOpenSetting:
    def test_open_aggregates_match_the_pinned_baseline(
        self, open_report: dict
    ) -> None:
        agg = open_report["aggregate"]
        for metric, pinned in BASELINE["open"].items():
            assert agg[metric] == pytest.approx(pinned, abs=0.005), metric
        assert baseline_problems(open_report, setting="open") == []

    def test_open_recovers_recall_and_loses_precision(
        self, default_report: dict, open_report: dict
    ) -> None:
        assert open_report["aggregate"]["mean_recall"] == 1.0
        assert (
            open_report["aggregate"]["mean_precision"]
            < default_report["aggregate"]["mean_precision"]
        )
        assert open_report["aggregate"]["perfect_recall_queries"] == 9


# ── The gate checker + measurement discipline ────────────────────────────────


class TestBaselineProblems:
    def test_regression_is_named(self, default_report: dict) -> None:
        regressed = {
            **default_report,
            "aggregate": {
                **default_report["aggregate"], "mean_recall": 0.5,
            },
        }
        problems = baseline_problems(regressed, setting="default")
        assert len(problems) == 1
        assert "mean_recall" in problems[0]

    def test_improvement_is_never_a_problem(self, default_report: dict) -> None:
        improved = {
            **default_report,
            "aggregate": {
                **default_report["aggregate"], "mean_precision": 1.0,
            },
        }
        assert baseline_problems(improved, setting="default") == []

    def test_unknown_setting_is_named(self, default_report: dict) -> None:
        assert baseline_problems(default_report, setting="bogus")


class TestMeasurementDiscipline:
    def test_deterministic_across_independent_builds(
        self, tmp_path: Path, default_report: dict
    ) -> None:
        other = evaluate_retrieval(build_fixture_db(tmp_path / "second.db"))
        assert other["aggregate"] == default_report["aggregate"]
        assert other["per_query"] == default_report["per_query"]

    def test_top_k_truncates_recall_exactly(self, fixture_db: Path) -> None:
        report = evaluate_retrieval(fixture_db, top_k=1)
        queries = by_id(report)
        # Q2 has two right answers; only the top one survives.
        assert queries["Q2"]["returned"] == ["QF4302"]
        assert queries["Q2"]["recall"] == 0.5
        assert queries["Q2"]["precision"] == 1.0
