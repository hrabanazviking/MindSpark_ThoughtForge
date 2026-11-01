"""
TF-121 — Harness architecture: benchmarks as code with
fixtures, scorers, and pinned baselines.

The proof lives as verifiable data in
``benchmarks/harness.py`` (``BENCHMARKS``, ``FIXTURES``,
``PERSONA_FIXTURE_RESPONSES``, ``METRIC_DEFINITIONS``,
``BASELINES``, ``HARNESS_RULES``, checked by
``harness_problems()`` / ``baseline_problems()``);
D-BENCH-2 and D-BENCH-6 were fixed in the estate's own
modules in the open (honest metric definitions +
MODEL_CARD's avg-words target now checked;
``quality_tiers`` populated, dedicated warm-up,
nearest-rank p95, the persona CLI's ``--report`` reach),
and D-BENCH-1 was narrowed (a bogus profile is refused
by name; application stays TF-128's).

Pinned here:

  * the filing as data: format, doc, rules, profiles,
  * the index: nine entries, every one bound to a real
    module / attribute / file; the locust fate filed as
    excluded; axis coverage depth + presence only,
  * the fixtures: synthetic, bound live, six persona
    lines carrying both penalty and bonus legs,
  * the baselines: pinned values re-measured live and
    equal (persona 0.7167 honestly below the 0.75 pass
    threshold; the empty-store profile baseline with
    every target unmet), latency pinned nowhere,
  * the negative cases: a moved baseline is named, a
    broken index entry is named, a bogus profile is
    named and refused,
  * the neighbours' fixes live: tiers populated,
    warm-up never a measured query, p95 nearest-rank,
    avg-words target judged, ``--report`` prints the
    markdown report,
  * the checker is clean on the real estate.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no loaded model, no seeded store, never
claimed.
"""

from __future__ import annotations

import io
import importlib
from pathlib import Path

import pytest

import benchmarks.harness as harness
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    HARNESS_DOC,
    HARNESS_FORMAT,
    HARNESS_RULES,
    KNOWN_PROFILES,
    METRIC_DEFINITIONS,
    PERSONA_FIXTURE_RESPONSES,
    baseline_problems,
    harness_as_data,
    harness_problems,
    measure_persona_baseline,
    measure_profile_baseline,
    validate_profile_id,
)
from benchmarks.benchmark_profiles import (
    WARM_UP_QUERY,
    BenchmarkQuery,
    BenchmarkResult,
    ProfileBenchmark,
    TurnResult,
    known_profile_ids,
)

REPO_ROOT = Path(__file__).parent.parent


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert HARNESS_FORMAT == "thoughtforge-benchmark-harness/1"
        assert HARNESS_DOC == "docs/architecture/HARNESS_ARCHITECTURE.md"
        assert (REPO_ROOT / HARNESS_DOC).exists()
        ids = [r["id"] for r in HARNESS_RULES]
        assert len(ids) == len(set(ids)) == 9
        data = harness_as_data()
        assert data["format"] == HARNESS_FORMAT
        assert data["doc"] == HARNESS_DOC
        assert [r["id"] for r in data["rules"]] == ids
        assert len(data["benchmarks"]) == len(BENCHMARKS)
        assert len(data["baselines"]) == len(BASELINES)
        assert data["persona_fixture_responses"] == \
            list(PERSONA_FIXTURE_RESPONSES)


# ── Profiles ────────────────────────────────────────────────────────────────


class TestProfiles:
    def test_known_profiles_match_the_shipped_files(self) -> None:
        assert tuple(known_profile_ids()) == KNOWN_PROFILES
        assert KNOWN_PROFILES == (
            "desktop_cpu", "desktop_gpu", "phone_low", "pi_5",
            "pi_zero", "server_gpu")
        for profile in KNOWN_PROFILES:
            assert validate_profile_id(profile) == []

    def test_a_bogus_profile_is_named_and_refused(self) -> None:
        problems = validate_profile_id("totally_bogus_profile")
        assert len(problems) == 1
        assert "totally_bogus_profile" in problems[0]
        with pytest.raises(ValueError, match="totally_bogus_profile"):
            ProfileBenchmark().run("totally_bogus_profile")


# ── The index ───────────────────────────────────────────────────────────────


class TestBenchmarkIndex:
    def test_nine_entries_cover_the_estate(self) -> None:
        ids = [b["id"] for b in BENCHMARKS]
        assert len(ids) == len(set(ids)) == 9
        for needed in ("persona_consistency", "profile_benchmark",
                       "locust_load_test", "retrieval_precision",
                       "scoring_honesty", "quant_quality",
                       "quality_marks", "determinism", "perf_tracker"):
            assert needed in ids

    def test_every_entry_binds_a_real_thing(self) -> None:
        for bench in BENCHMARKS:
            if "bind_module" in bench:
                module = importlib.import_module(bench["bind_module"])
                assert hasattr(module, bench["bind_attribute"]), bench["id"]
            else:
                text = (REPO_ROOT / bench["bind_file"]).read_text(
                    encoding="utf-8")
                for marker in bench["bind_file_markers"]:
                    assert marker in text, (bench["id"], marker)
            if bench["doc"] is not None:
                assert (REPO_ROOT / bench["doc"]).exists(), bench["id"]

    def test_locust_fate_is_exclusion(self) -> None:
        locust = [b for b in BENCHMARKS if b["id"] == "locust_load_test"][0]
        assert locust["status"] == "excluded_by_verdict"
        assert locust["claims_axes"] == ()
        assert any("D-BENCH-5" in d for d in locust["docket"])

    def test_axis_coverage_is_depth_and_presence_only(self) -> None:
        covered = {axis for b in BENCHMARKS for axis in b["claims_axes"]
                   if b["kind"] in ("scorer", "runner")}
        assert covered == {"depth", "presence"}
        assert "the_will_axis_has_no_scorer_yet" in {
            r["id"] for r in HARNESS_RULES}


# ── Fixtures ────────────────────────────────────────────────────────────────


class TestFixtures:
    def test_fixtures_are_synthetic_and_bind_live(self) -> None:
        assert len(FIXTURES) == 2
        for fixture in FIXTURES:
            assert fixture["synthetic"] is True
            module = importlib.import_module(fixture["bind_module"])
            assert getattr(module, fixture["bind_attribute"])

    def test_persona_fixture_carries_both_legs(self) -> None:
        assert len(PERSONA_FIXTURE_RESPONSES) == 6
        measured = measure_persona_baseline()
        assert measured["flagged_turns"] == 2       # the penalty leg
        assert measured["norse_tone_hits"] == 10    # the bonus leg
        assert measured["citation_turns"] == 1


# ── The pinned baselines ────────────────────────────────────────────────────


class TestBaselines:
    def test_persona_baseline_pins_the_measured_values(self) -> None:
        baseline = [b for b in BASELINES
                    if b["id"] == "persona_consistency_synthetic_v1"][0]
        assert baseline["metrics"]["consistency_score"] == \
            pytest.approx(0.7166666666666667)
        assert baseline["targets"]["consistency_score"] == \
            {"target": 0.75, "met": False}
        assert baseline_problems("persona_consistency_synthetic_v1") == []

    def test_profile_baseline_pins_the_empty_store_truth(self) -> None:
        baseline = [b for b in BASELINES
                    if b["id"] == "profile_benchmark_empty_store_v1"][0]
        metrics = baseline["metrics"]
        assert metrics["total_turns"] == 10
        assert metrics["successful_turns"] == 10
        assert metrics["citation_accuracy"] == 0.0
        assert metrics["avg_response_words"] == 9.0
        assert metrics["avg_confidence"] == pytest.approx(0.408)
        assert metrics["quality_tiers"] == {"weak": 10}
        assert metrics["min_words_met_rate"] == 0.0
        assert all(t["met"] is False for t in baseline["targets"].values())
        assert baseline_problems("profile_benchmark_empty_store_v1") == []

    def test_latency_is_pinned_nowhere(self) -> None:
        for baseline in BASELINES:
            assert not any("latency" in m for m in baseline["metrics"])
        profile = [b for b in BASELINES
                   if b["benchmark"] == "profile_benchmark"][0]
        assert "avg_latency_ms" in profile["excluded_metrics"]
        assert "p95_latency_ms" in profile["excluded_metrics"]

    def test_every_baseline_metric_is_defined(self) -> None:
        defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
        for baseline in BASELINES:
            for metric in baseline["metrics"]:
                assert (baseline["benchmark"], metric) in defined
        legacy = [m for m in METRIC_DEFINITIONS
                  if m["id"] == "min_words_met_rate"][0]
        assert legacy["legacy_name"] == "token_efficiency"

    def test_a_moved_baseline_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        doctored = tuple(
            {**b, "metrics": {**b["metrics"], "consistency_score": 0.5}}
            if b["id"] == "persona_consistency_synthetic_v1" else b
            for b in BASELINES)
        monkeypatch.setattr(harness, "BASELINES", doctored)
        problems = baseline_problems("persona_consistency_synthetic_v1")
        assert len(problems) == 1
        assert "consistency_score" in problems[0]
        assert "moved" in problems[0]


# ── The neighbours' fixes, live (D-BENCH-2 / D-BENCH-6) ─────────────────────


class _StubCore:
    """Records think() calls; returns an empty-store-shaped result."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def think(self, text: str):
        self.calls.append(text)

        class _Scores:
            composite = 0.408
            quality_tier = "weak"

        class _Result:
            text = "The forge has no knowledge relevant to this query."
            citations: list[str] = []
            enforcement_passed = False
            scores = _Scores()

        return _Result()


class TestNeighbourFixes:
    def test_warm_up_never_touches_a_measured_query(self) -> None:
        core = _StubCore()
        queries = [BenchmarkQuery("What is Yggdrasil?", "Norse mythology", 5),
                   BenchmarkQuery("Who is Odin?", "Norse mythology", 5)]
        result = ProfileBenchmark().run(
            "desktop_cpu", queries=queries, core=core, warm_up_turns=2)  # type: ignore[arg-type]
        assert core.calls[:2] == [WARM_UP_QUERY.text] * 2
        assert core.calls[2:] == [q.text for q in queries]
        assert result.quality_tiers == {"weak": 2}

    def test_p95_is_nearest_rank(self) -> None:
        bench = ProfileBenchmark()
        queries = [BenchmarkQuery(f"q{i}", "t", 0) for i in range(10)]
        turns = [TurnResult(query=q.text, response_text="x y",
                            response_words=2, latency_ms=float(i + 1),
                            citations=[], enforcement_passed=False,
                            confidence=0.5, quality_tier="usable")
                 for i, q in enumerate(queries)]
        result = bench._aggregate("desktop_cpu", queries, turns, [])
        assert result.p95_latency_ms == 10.0   # the old index filed 9.0
        assert result.quality_tiers == {"usable": 10}

    def test_avg_words_target_is_judged(self) -> None:
        base = dict(profile_id="test", total_turns=1, successful_turns=1,
                    failed_turns=0, citation_accuracy=0.0,
                    avg_latency_ms=1.0, median_latency_ms=1.0,
                    p95_latency_ms=1.0, token_efficiency=0.0,
                    enforcement_pass_rate=0.0, avg_confidence=0.5)
        assert BenchmarkResult(
            avg_response_words=9.0, **base).passes_avg_words_target() is False
        assert BenchmarkResult(
            avg_response_words=30.0, **base).passes_avg_words_target() is True
        assert BenchmarkResult.AVG_WORDS_TARGET == 30.0
        summary = BenchmarkResult(avg_response_words=9.0, **base).summary()
        assert "target ≥30" in summary

    def test_persona_cli_report_is_reachable(
            self, monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str]) -> None:
        from benchmarks.persona_consistency import main

        monkeypatch.setattr(
            "sys.stdin", io.StringIO("The wyrd binds. Q12345\n"))
        monkeypatch.setattr("sys.argv", ["persona_consistency", "--report"])
        main()
        out = capsys.readouterr().out
        assert "# ThoughtForge Persona Consistency Report" in out


# ── Checker ─────────────────────────────────────────────────────────────────


class TestChecker:
    def test_harness_checker_clean_on_the_real_estate(self) -> None:
        assert harness_problems() == []

    def test_a_broken_index_entry_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = tuple(
            {**b, "bind_attribute": "NoSuchScorerAnywhere"}
            if b["id"] == "persona_consistency" else b
            for b in BENCHMARKS)
        monkeypatch.setattr(harness, "BENCHMARKS", broken)
        problems = harness_problems()
        assert any("persona_consistency" in p and "NoSuchScorerAnywhere" in p
                   for p in problems)
