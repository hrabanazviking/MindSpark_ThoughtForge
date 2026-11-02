"""
benchmark_profiles.py — Per-hardware-profile benchmarking for ThoughtForge.

Measures:
  - Citation accuracy (target: >85% of retrieved QIDs cited)
  - Response length adequacy (words per turn; MODEL_CARD target
    avg >= 30 words, checked by BenchmarkResult.passes_avg_words_target)
  - Per-turn latency (ms)
  - Min-words met rate — filed in the result under the legacy field
    name `token_efficiency`: the fraction of successful turns whose
    response meets the query's min_response_words. It is NOT a
    useful-words/total-words ratio; no such ratio is measured
    anywhere in this tree (TF-121, D-BENCH-2).
  - Enforcement gate pass rate

TF-121 (D-BENCH-1 share): `profile_id` is validated against the
shipped hardware profiles (hardware_profiles/*.json) and an unknown
profile raises ValueError by name — it no longer runs silently.
The profile is still not APPLIED to the core (ThoughtForgeCore has
no profile parameter); that share is TF-128's.

Usage:
    from benchmarks.benchmark_profiles import ProfileBenchmark
    result = ProfileBenchmark().run("desktop_cpu")
    print(result.summary())
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thoughtforge.cognition.core import ThoughtForgeCore

logger = logging.getLogger(__name__)

# ── Default benchmark query set ───────────────────────────────────────────────

@dataclass
class BenchmarkQuery:
    """A single benchmark query with expected characteristics."""
    text: str
    topic: str
    min_response_words: int = 10


_DEFAULT_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery("What is Yggdrasil?", "Norse mythology", 15),
    BenchmarkQuery("Explain the concept of frith in Norse culture.", "Norse culture", 20),
    BenchmarkQuery("Who are the Vanir gods?", "Norse mythology", 15),
    BenchmarkQuery("What is the difference between Asgard and Midgard?", "Norse cosmology", 20),
    BenchmarkQuery("Describe the role of ravens in Norse mythology.", "Norse mythology", 15),
    BenchmarkQuery("What are the nine worlds connected by Yggdrasil?", "Norse cosmology", 20),
    BenchmarkQuery("Explain what a völva was in Norse society.", "Norse history", 15),
    BenchmarkQuery("What is the significance of Valhalla?", "Norse afterlife", 15),
    BenchmarkQuery("Who is Loki and what is his role in Norse myths?", "Norse mythology", 20),
    BenchmarkQuery("What does the word 'wyrd' mean in Old Norse context?", "Norse language", 15),
]

# Warm-up turns run this dedicated query, never a measured one
# (TF-121, D-BENCH-6: warm-up used to be drawn from the measured
# set itself, so query[0] was warmed and then measured again).
WARM_UP_QUERY = BenchmarkQuery("Warm up the forge.", "warm-up", 0)

_HARDWARE_PROFILES_DIR = Path(__file__).resolve().parent.parent / "hardware_profiles"


def known_profile_ids() -> list[str]:
    """The shipped hardware profile ids (hardware_profiles/*.json)."""
    if not _HARDWARE_PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _HARDWARE_PROFILES_DIR.glob("*.json"))


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result for a single benchmark turn."""
    query: str
    response_text: str
    response_words: int
    latency_ms: float
    citations: list[str]
    enforcement_passed: bool
    confidence: float
    error: str = ""
    quality_tier: str = ""   # the turn score's tier, when the core filed one


@dataclass
class BenchmarkResult:
    """Aggregate results of a profile benchmark run."""
    profile_id: str
    total_turns: int
    successful_turns: int
    failed_turns: int

    # Core metrics
    citation_accuracy: float        # fraction of successful turns carrying ≥1 citation
                                    # (denominator: all successful turns; any citation
                                    # string counts — TF-121, D-BENCH-2)
    avg_response_words: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    token_efficiency: float         # legacy name: the min-words met rate — fraction of
                                    # successful turns meeting their query's
                                    # min_response_words (TF-121, D-BENCH-2)
    enforcement_pass_rate: float    # fraction of turns enforcement_passed=True
    avg_confidence: float

    # Quality tier distribution
    quality_tiers: dict[str, int] = field(default_factory=dict)
    turn_results: list[TurnResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # Targets (from BUILD_PLAN_v1)
    CITATION_TARGET: float = 0.85
    ENFORCEMENT_TARGET: float = 0.90
    # MODEL_CARD's "avg response words ≥ 30" target — checked nowhere
    # before TF-121 (D-BENCH-2); now judged like the other two.
    AVG_WORDS_TARGET: float = 30.0

    def passes_citation_target(self) -> bool:
        return self.citation_accuracy >= self.CITATION_TARGET

    def passes_enforcement_target(self) -> bool:
        return self.enforcement_pass_rate >= self.ENFORCEMENT_TARGET

    def passes_avg_words_target(self) -> bool:
        return self.avg_response_words >= self.AVG_WORDS_TARGET

    def summary(self) -> str:
        """Human-readable benchmark summary."""
        lines = [
            f"═══ ThoughtForge Benchmark — {self.profile_id} ═══",
            f"  Turns        : {self.successful_turns}/{self.total_turns} successful",
            f"  Citation acc : {self.citation_accuracy:.1%}  (target ≥85%)"
            + (" ✓" if self.passes_citation_target() else " ✗"),
            f"  Enforcement  : {self.enforcement_pass_rate:.1%}  (target ≥90%)"
            + (" ✓" if self.passes_enforcement_target() else " ✗"),
            f"  Avg words    : {self.avg_response_words:.1f}  (target ≥30)"
            + (" ✓" if self.passes_avg_words_target() else " ✗"),
            f"  Avg latency  : {self.avg_latency_ms:.0f}ms  (p95: {self.p95_latency_ms:.0f}ms)",
            f"  Avg confidence: {self.avg_confidence:.3f}",
            f"  Token eff.   : {self.token_efficiency:.1%}",
        ]
        if self.notes:
            lines.append("  Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


# ── ProfileBenchmark ──────────────────────────────────────────────────────────

class ProfileBenchmark:
    """
    Runs the standard ThoughtForge benchmark across a hardware profile.

    Works in knowledge-only mode (no GGUF model required) — latency numbers
    will reflect pure retrieval + scaffold overhead rather than inference time.
    To benchmark with a model, pass `core` with model_path set.
    """

    def run(
        self,
        profile_id: str = "desktop_cpu",
        queries: list[BenchmarkQuery] | None = None,
        core: "ThoughtForgeCore | None" = None,
        warm_up_turns: int = 1,
    ) -> BenchmarkResult:
        """
        Run benchmark for the given profile.

        Args:
            profile_id:    Hardware profile to benchmark.
            queries:       Query set (default: 10 standard Norse-topic queries).
            core:          Pre-built ThoughtForgeCore (optional — creates ephemeral one if None).
            warm_up_turns: Number of warm-up turns to discard from timing stats.

        Returns:
            BenchmarkResult with all metrics populated.
        """
        known = known_profile_ids()
        if known and profile_id not in known:
            raise ValueError(
                f"unknown hardware profile {profile_id!r} — known profiles: "
                f"{', '.join(known)} (TF-121, D-BENCH-1: a bogus profile "
                "no longer runs silently)")
        queries = queries or _DEFAULT_QUERIES
        notes: list[str] = []

        if core is None:
            core = self._make_ephemeral_core(profile_id, notes)

        logger.info("ProfileBenchmark: running %d queries on profile=%s", len(queries), profile_id)

        # Warm-up: the dedicated warm-up query only — a measured
        # query is never warmed (TF-121, D-BENCH-6).
        for _ in range(warm_up_turns):
            try:
                core.think(WARM_UP_QUERY.text)
            except Exception:
                pass

        turn_results: list[TurnResult] = []
        for bq in queries:
            turn = self._run_turn(core, bq)
            turn_results.append(turn)
            logger.debug(
                "Benchmark turn: query=%r latency=%.0fms enforcement=%s",
                bq.text[:40], turn.latency_ms, turn.enforcement_passed,
            )

        return self._aggregate(profile_id, queries, turn_results, notes)

    # ── Core helpers ───────────────────────────────────────────────────────────

    def _run_turn(self, core: "ThoughtForgeCore", bq: BenchmarkQuery) -> TurnResult:
        """Run a single benchmark turn, capturing timing + metrics."""
        start = time.perf_counter()
        try:
            result = core.think(bq.text)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            text = getattr(result, "text", "") or ""
            word_count = len(text.split())
            citations = list(getattr(result, "citations", []) or [])
            enforcement_passed = bool(getattr(result, "enforcement_passed", False))
            scores = getattr(result, "scores", None)
            confidence = float(scores.composite if scores and hasattr(scores, "composite") else 0.0)
            quality_tier = str(getattr(scores, "quality_tier", "") or "") if scores else ""

            return TurnResult(
                query=bq.text,
                response_text=text,
                response_words=word_count,
                latency_ms=elapsed_ms,
                citations=citations,
                enforcement_passed=enforcement_passed,
                confidence=confidence,
                quality_tier=quality_tier,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("Benchmark turn failed: %s", e)
            return TurnResult(
                query=bq.text,
                response_text="",
                response_words=0,
                latency_ms=elapsed_ms,
                citations=[],
                enforcement_passed=False,
                confidence=0.0,
                error=str(e),
            )

    def _aggregate(
        self,
        profile_id: str,
        queries: list[BenchmarkQuery],
        turns: list[TurnResult],
        notes: list[str],
    ) -> BenchmarkResult:
        """Compute aggregate metrics from individual turn results."""
        successful = [t for t in turns if not t.error]
        failed = [t for t in turns if t.error]

        if not successful:
            notes.append("all turns failed")
            return BenchmarkResult(
                profile_id=profile_id,
                total_turns=len(turns),
                successful_turns=0,
                failed_turns=len(failed),
                citation_accuracy=0.0,
                avg_response_words=0.0,
                avg_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                token_efficiency=0.0,
                enforcement_pass_rate=0.0,
                avg_confidence=0.0,
                turn_results=turns,
                notes=notes,
            )

        # Citation accuracy: turns with ≥1 citation / total successful
        turns_with_citations = sum(1 for t in successful if t.citations)
        citation_accuracy = turns_with_citations / len(successful)

        # Response stats
        word_counts = [t.response_words for t in successful]
        avg_words = statistics.mean(word_counts)

        # Latency stats. p95 by nearest-rank (TF-121, D-BENCH-6: the
        # old index int(n*0.95)-1 picked ~p90 at n=10 and printed a
        # "p95" below the mean).
        latencies = [t.latency_ms for t in successful]
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        sorted_latencies = sorted(latencies)
        p95_idx = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
        p95_latency = sorted_latencies[p95_idx]

        # Min-words met rate (filed as token_efficiency, the legacy
        # field name — TF-121, D-BENCH-2): the fraction of successful
        # turns meeting their query's min_response_words.
        meets_min = sum(1 for t, bq in zip(successful, queries) if t.response_words >= bq.min_response_words)
        token_efficiency = meets_min / len(successful)

        # Quality tier distribution (TF-121, D-BENCH-6: the field was
        # declared and never populated — always {}).
        quality_tiers: dict[str, int] = {}
        for t in successful:
            tier = t.quality_tier or "unscored"
            quality_tiers[tier] = quality_tiers.get(tier, 0) + 1

        # Enforcement pass rate
        enforcement_pass_rate = sum(1 for t in successful if t.enforcement_passed) / len(successful)

        # Avg confidence
        avg_confidence = statistics.mean(t.confidence for t in successful)

        return BenchmarkResult(
            profile_id=profile_id,
            total_turns=len(turns),
            successful_turns=len(successful),
            failed_turns=len(failed),
            citation_accuracy=citation_accuracy,
            avg_response_words=avg_words,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            token_efficiency=token_efficiency,
            enforcement_pass_rate=enforcement_pass_rate,
            avg_confidence=avg_confidence,
            quality_tiers=quality_tiers,
            turn_results=turns,
            notes=notes,
        )

    def _make_ephemeral_core(self, profile_id: str, notes: list[str]) -> "ThoughtForgeCore":
        """Create an in-memory ThoughtForgeCore for benchmarking."""
        import tempfile
        from thoughtforge.cognition.core import ThoughtForgeCore

        tmp = tempfile.mkdtemp()
        notes.append(f"using ephemeral core (no DB) — latency = retrieval+scaffold overhead only")
        return ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "bench.db",
            model_path=None,
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Run benchmark from command line: python -m benchmarks.benchmark_profiles [profile]"""
    import sys
    profile = sys.argv[1] if len(sys.argv) > 1 else "desktop_cpu"
    try:
        result = ProfileBenchmark().run(profile_id=profile)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(result.summary())


if __name__ == "__main__":
    main()
