"""
Benchmark harness architecture, as data (TF-121).

The roadmap's rule: *harness architecture: benchmarks as
code with fixtures, scorers, and pinned baselines (the
`benchmarks/` estate, TF-010, unified).* Part VII's goal:
ThoughtForge's claims become numbers — a harness that
scores depth, presence, and will instead of asserting them.

Where this lives, and why: in the estate's own package,
``benchmarks/``. The estate TF-010 censused is a top-level
package imported as ``benchmarks.*`` by its tests; the
shipping library under ``src/thoughtforge/`` imports
nothing from it, and a harness filed under ``src/`` would
have to reach back across the root package to bind the
very modules it unifies. The neighbour scorers built by
earlier parts (TF-043 / TF-044 / TF-067 / TF-102 /
TF-036) are bound here by module + attribute and judged
live by the checker — bound, not copied.

The measured starting state, probed live this tick on
this tree before anything was built:

  * **TF-010's census stood exactly as filed**: the
    whole estate is 2 modules (``benchmark_profiles``,
    ``persona_consistency``), 0 suites — no fixture
    corpus, no seeded benchmark store, no results
    directory, no persisted output, and **no pinned
    baseline anywhere in the tree** to regress against.
  * **D-BENCH-1 stood open**: ``python3 -m
    benchmarks.benchmark_profiles totally_bogus_profile``
    ran to exit 0 with identical-shaped output — the
    profile is never validated and never applied.
  * **D-BENCH-2 stood open**: ``token_efficiency`` is
    documented "useful words / total words" but computes
    the fraction of turns meeting ``min_response_words``;
    ``citation_accuracy``'s comment says "when QIDs
    retrieved" but its denominator is all successful
    turns; MODEL_CARD's "avg response words >= 30"
    target was checked nowhere.
  * **D-BENCH-6 stood open, re-probed live**: a fresh
    ``ProfileBenchmark`` run filed ``quality_tiers ==
    {}`` (declared, never populated); warm-up turns were
    drawn from the measured set; the p95 index picked
    ~p90 at n=10 — this tick's CLI run printed avg 8ms
    with "p95: 2ms", below the mean; nothing persisted;
    the persona CLI could not reach ``generate_report()``.
  * **D-BENCH-5 / D-BENCH-7 stood open**: ``locust`` is
    not installed on this VM (a dev extra — import
    probed, ModuleNotFoundError), and the core's
    ``utils/perf.py`` tracker is consumed by no
    benchmark.

What this slice builds, filed as data here:

  * ``BENCHMARKS`` — the unified index: every benchmark
    and neighbour scorer in the estate, each bound to
    the real module + attribute (or, for the locust
    script, the real file + markers) it lives in, its
    fixture, its claims axes, its suite slice, and its
    docket state. Checked live by ``harness_problems()``.
  * ``FIXTURES`` + ``PERSONA_FIXTURE_RESPONSES`` — the
    first filed fixture corpus: six synthetic response
    lines written for this slice (TF-017: synthetic
    only, canon never a fixture), plus the profile
    benchmark's own default query set bound in place.
  * ``METRIC_DEFINITIONS`` — every metric the pinned
    baselines carry, with what it actually computes
    (D-BENCH-2's remedy at the harness level: canonical
    names beside the legacy field names).
  * ``BASELINES`` — the estate's first pinned baselines,
    measured live on this VM at banking (see the doc):
    the persona scorer over its filed fixture, and the
    profile benchmark over its ephemeral empty-store
    core. Latency is deliberately NOT pinned: it varied
    between probes (avg 7ms / 8ms, "p95" 5ms / 2ms) —
    walls under contention are not baselines; TF-133
    owns cost measurement.
  * ``baseline_problems()`` — re-measures both baselines
    live and names any metric that moved beyond its
    filed tolerance. Improvement and regression are both
    named; a moved baseline is a finding, never silently
    re-pinned.

Fixed in the open, in ``benchmarks/benchmark_profiles.py``
and ``benchmarks/persona_consistency.py``:

  * **D-BENCH-2 FIXED.** The metric comments and the
    module docstring now state what each metric actually
    computes; MODEL_CARD's avg-words target is filed as
    ``BenchmarkResult.AVG_WORDS_TARGET = 30.0``, judged
    by ``passes_avg_words_target()`` and printed in the
    summary beside the other two targets. The field name
    ``token_efficiency`` is kept (the phase-6 tests and
    the dataclass API pin it) and is filed in
    ``METRIC_DEFINITIONS`` under its canonical name
    ``min_words_met_rate`` with the legacy name named.
  * **D-BENCH-6 FIXED.** ``quality_tiers`` is populated
    from each successful turn's score tier (the empty-
    store baseline now files ``{"weak": 10}``); warm-up
    runs the dedicated ``WARM_UP_QUERY``, never a
    measured query; p95 is nearest-rank
    (``ceil(0.95 * n) - 1``); the pinned baselines here
    are the persistence the estate lacked (TF-134 owns
    the baseline book's regenerated form); and the
    persona CLI reaches ``generate_report()`` via
    ``--report``.
  * **D-BENCH-1 narrowed.** ``ProfileBenchmark.run``
    validates ``profile_id`` against the shipped
    ``hardware_profiles/*.json`` ids and raises
    ``ValueError`` naming the bogus id and the known
    set; the CLI exits 2 with that message. The profile
    is still not APPLIED — ``ThoughtForgeCore`` has no
    profile parameter; application is TF-128's share,
    and this entry stays OPEN on it.
  * **D-BENCH-5 narrowed (TF-121's fate share decided).**
    The locust file's fate is filed in ``BENCHMARKS``:
    EXCLUDED from the harness panel — it is a manual
    load script, not a scorer: unrunnable without the
    dev extra, no-HTTP by its own header, and it
    measures throughput, not a claim axis. TF-136 owns
    the stale TASK_PHASE6 description.
  * **D-BENCH-7 narrowed (TF-121's share).** Both
    latency systems are filed in one index with their
    roles stated: ``ProfileBenchmark``'s per-turn
    ``perf_counter`` is the harness's turn timing;
    ``utils/perf.py``'s ``PerfTracker`` is core
    instrumentation, filed as an instrument, not a
    scorer. Consuming the tracker is TF-133's share;
    the entry stays OPEN on it.

D-BENCH-3 and D-BENCH-4 were TF-122's (the persona
suite): TF-121 pinned the persona baseline under the
scorer exactly as it behaved then, dead ``weights``
and all. **TF-122 update, in the open:** D-BENCH-3 is
FIXED with the defaults reproducing the pinned
baseline exactly (``baseline_problems()`` still clean),
D-BENCH-4's vacuous pass is FIXED and its missing run
is filed as the suite's 108-turn corpus (the entry is
narrowed, its public-docs share standing with
TF-136), the suite is registered below
(``persona_suite``) with its own pinned baseline
(``persona_suite_synthetic_v1``), and the persona
entry's suite-slice note now files TF-122 done.

**TF-123 update, in the open:** the memory fidelity
suite is built (``benchmarks/memory_suite.py``) and
registered below (``memory_suite``) with its own
pinned baseline (``memory_suite_synthetic_v1``,
measured live twice, identical); the
``retrieval_precision`` entry's suite-slice note now
files TF-123 DONE — the suite consumes TF-043's
corpus + pinned baseline mechanically. TF-123's
probes found and filed **D-BENCH-9** (citation truth
and the explain side scored by nothing: a
cited-but-false claim passed TF-049's presence
check identically to the true claim) and FIXED it
in the scoring sense TF-049's doc assigned — the
suite's truth + explain axes now score both, with
the failing shapes filed as cases. All earlier
pinned baselines stand exactly as filed (never
re-pinned).

Filed limits, stated plainly:

  * The profile baseline measures an ephemeral core
    over an EMPTY store with ``model_path=None`` — the
    empty-store refusal (9 words, confidence 0.408),
    never inference, never a loaded model. A seeded
    store and a loaded model are NOT PROVEN here.
  * The will axis has no scorer anywhere in the estate
    yet: ``BENCHMARKS`` covers depth and presence only.
    TF-125 owns the will suite; the gap is filed as a
    rule below, not dressed as coverage.
  * No suite in Part VII existed at TF-121: TF-122..
    TF-125 build them. **TF-122 update:** the first
    suite (persona consistency) now exists and is
    registered below; TF-123..TF-125's suites, TF-129's
    regression gates, TF-134's baseline book and
    TF-137's one-command tool still stand as filed.
    **TF-123 update:** the second suite (memory
    fidelity) now exists and is registered below;
    TF-124..TF-125's suites, TF-129's regression
    gates, TF-134's baseline book and TF-137's
    one-command tool still stand as filed.

Docket (TF-016): D-BENCH-2 and D-BENCH-6 are FIXED by
this slice; D-BENCH-1, D-BENCH-5 and D-BENCH-7 are
narrowed by it, their remaining shares standing with
TF-128 / TF-136 / TF-133 as filed. No new entry: every
gap probed was already numbered by TF-010.

The human-readable write-up is
``docs/architecture/HARNESS_ARCHITECTURE.md``.
Proof grade: FIXTURE — synthetic fixtures through the
real scorers on this VM; no loaded model, no seeded
store, never claimed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

HARNESS_FORMAT = "thoughtforge-benchmark-harness/1"
HARNESS_DOC = "docs/architecture/HARNESS_ARCHITECTURE.md"

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── The shipped hardware profiles ───────────────────────────────────────────
#
# The only profile ids a benchmark run may name (TF-121,
# D-BENCH-1's validation share). Judged live against
# hardware_profiles/*.json by harness_problems().

KNOWN_PROFILES: tuple[str, ...] = (
    "desktop_cpu", "desktop_gpu", "phone_low", "pi_5", "pi_zero",
    "server_gpu",
)


def validate_profile_id(profile_id: str) -> list[str]:
    """Name an unknown profile; a known one files no problem."""
    if profile_id in KNOWN_PROFILES:
        return []
    return [
        f"unknown hardware profile {profile_id!r} — known profiles: "
        f"{', '.join(KNOWN_PROFILES)}"]


# ── The filed fixture corpus (TF-017: synthetic only) ───────────────────────
#
# Six response lines written for this slice: three in-voice
# lines (one carrying a QID citation), one plain factual
# line, and two deliberately generic AI-flavoured lines, so
# the pinned baseline exercises the scorer's penalty leg
# as well as its bonuses. Canon is never a fixture.

PERSONA_FIXTURE_RESPONSES: tuple[str, ...] = (
    "The wyrd binds the nine worlds, and Yggdrasil holds them; "
    "in short, the answer is the world tree. Q12345",
    "As an AI, I cannot provide that, and I am unable to recite "
    "the saga.",
    "Frith is kept concretely: the skald names the price "
    "directly, and the mead hall hears it.",
    "Odin hung on the tree for nine nights; the runes came at "
    "a price, specifically his eye.",
    "The ferry crosses at dawn, and the lantern keeper trims "
    "the wick before the crossing.",
    "Please note that I must remind you, as a language model, "
    "my training data ends there.",
)

FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "persona_fixture_responses", "benchmark": "persona_consistency",
     "bind_module": "benchmarks.harness",
     "bind_attribute": "PERSONA_FIXTURE_RESPONSES",
     "synthetic": True,
     "subject": "six synthetic response lines (3 in-voice, 1 plain, "
                "2 deliberately generic) scored by the persona scorer"},
    {"id": "profile_default_queries", "benchmark": "profile_benchmark",
     "bind_module": "benchmarks.benchmark_profiles",
     "bind_attribute": "_DEFAULT_QUERIES",
     "synthetic": True,
     "subject": "the profile benchmark's own 10 standard Norse-topic "
                "queries, bound in place"},
    {"id": "persona_suite_corpus", "benchmark": "persona_suite",
     "bind_module": "benchmarks.persona_suite",
     "bind_attribute": "ALL_SUITE_TURNS",
     "synthetic": True,
     "subject": "TF-122's scripted persona corpora: 36 identity "
                "turns + 36 voice turns + 36 pressure turns "
                "(108 scored turns) in TF-030's synthetic world"},
    {"id": "memory_suite_corpus", "benchmark": "memory_suite",
     "bind_module": "benchmarks.memory_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "synthetic": True,
     "subject": "TF-123's scripted memory corpora: 12 present "
                "cases + 12 true cases + 12 explain cases "
                "(36 scored cases) in TF-043's synthetic world"},
)


# ── The unified benchmark index ─────────────────────────────────────────────
#
# Every benchmark / scorer / instrument in the estate.
# Package entries bind "bind_module" + "bind_attribute"
# (import + getattr, judged live); the locust script binds
# "bind_file" + "bind_file_markers" (it is not importable
# without its dev extra — that is its filed state).

BENCHMARKS: tuple[dict[str, Any], ...] = (
    {"id": "persona_consistency", "kind": "scorer",
     "status": "existing_scorer",
     "bind_module": "benchmarks.persona_consistency",
     "bind_attribute": "PersonaConsistencyScorer",
     "fixture": "persona_fixture_responses",
     "claims_axes": ("presence",),
     "suite_slice": "TF-122 DONE (persona consistency suite "
                    "completed it: benchmarks/persona_suite.py)",
     "doc": "docs/audit/TF-010_BENCHMARK_ESTATE.md",
     "docket": ("D-BENCH-3 FIXED by TF-122",
                "D-BENCH-4 OPEN narrowed by TF-122 (vacuous pass "
                "fixed; 108-turn corpus filed; public-docs wording "
                "TF-136)"),
     "notes": "pure-text phrase scorer; runs with no model, store or "
              "network. TF-122 made its weights parameter real "
              "(components scaled by weight / DEFAULT_WEIGHTS; the "
              "defaults reproduce the pinned baseline exactly) and "
              "an empty list now scores 0.0 / REVIEW, never the "
              "vacuous 1.0"},
    {"id": "persona_suite", "kind": "suite",
     "status": "built_suite",
     "bind_module": "benchmarks.persona_suite",
     "bind_attribute": "run_persona_suite",
     "fixture": "persona_suite_corpus",
     "claims_axes": ("presence",),
     "suite_slice": "TF-122 (persona consistency suite — itself)",
     "doc": "docs/architecture/PERSONA_CONSISTENCY_SUITE.md",
     "docket": ("D-BENCH-3 FIXED by TF-122",
                "D-BENCH-4 OPEN narrowed by TF-122"),
     "notes": "TF-122's scored suite over the persona_consistency "
              "scorer + TF-030's cognition hooks: identity facts, "
              "voice, values under pressure — 108 scripted "
              "synthetic turns, per-axis scores + overall, "
              "PASS/REVIEW by filed thresholds, empty suite "
              "0.0 / REVIEW"},
    {"id": "memory_suite", "kind": "suite",
     "status": "built_suite",
     "bind_module": "benchmarks.memory_suite",
     "bind_attribute": "run_memory_suite",
     "fixture": "memory_suite_corpus",
     "claims_axes": ("depth",),
     "suite_slice": "TF-123 (memory fidelity suite — itself)",
     "doc": "docs/architecture/MEMORY_FIDELITY_SUITE.md",
     "docket": ("D-BENCH-9 FIXED by TF-123",),
     "notes": "TF-123's scored suite over TF-049's real "
              "check_citations / matcher and TF-043's corpus: "
              "citations present, citations true, explain — "
              "36 scripted synthetic cases, per-axis scores + "
              "overall, PASS/REVIEW by filed thresholds, "
              "empty suite 0.0 / REVIEW"},
    {"id": "profile_benchmark", "kind": "runner",
     "status": "existing_runner",
     "bind_module": "benchmarks.benchmark_profiles",
     "bind_attribute": "ProfileBenchmark",
     "fixture": "profile_default_queries",
     "claims_axes": ("depth",),
     "suite_slice": "TF-128 (profile panels apply the profile)",
     "doc": "docs/audit/TF-010_BENCHMARK_ESTATE.md",
     "docket": ("D-BENCH-1 OPEN narrowed by TF-121 (validation fixed; "
                "application TF-128)",
                "D-BENCH-2 FIXED by TF-121",
                "D-BENCH-6 FIXED by TF-121"),
     "notes": "runs the core over an ephemeral empty store "
              "(model_path=None): measures retrieval + scaffold "
              "overhead and the empty-store refusal, never inference. "
              "Latency filed as data per run, never pinned"},
    {"id": "locust_load_test", "kind": "load_instrument",
     "status": "excluded_by_verdict",
     "bind_file": "locustfile.py",
     "bind_file_markers": ("_LOCUST_AVAILABLE", "no-HTTP"),
     "fixture": None,
     "claims_axes": (),
     "suite_slice": "none — manual load script only",
     "doc": "docs/audit/TF-010_BENCHMARK_ESTATE.md",
     "docket": ("D-BENCH-5 OPEN narrowed by TF-121 (fate decided "
                "here; stale TASK_PHASE6 claim TF-136)",),
     "notes": "TF-121's fate verdict (D-BENCH-5 share): EXCLUDED "
              "from the harness panel. It measures throughput under "
              "load, not a claim axis; it is unrunnable without the "
              "locust dev extra (not installed on this VM, probed); "
              "its own header files it as no-HTTP. It stays a manual "
              "script a human runs on purpose, never a suite"},
    {"id": "retrieval_precision", "kind": "neighbour_scorer",
     "status": "bound_neighbour",
     "bind_module": "thoughtforge.knowledge.retrieval_precision",
     "bind_attribute": "evaluate_retrieval",
     "fixture": None,
     "claims_axes": ("depth",),
     "suite_slice": "TF-123 DONE (memory fidelity suite "
                    "consumes it: benchmarks/memory_suite.py)",
     "doc": "docs/architecture/RETRIEVAL_PRECISION.md",
     "docket": (),
     "notes": "TF-043's known-answer retrieval fixture + its own "
              "pinned BASELINE (that module's filing, bound here, "
              "not re-pinned): the shape this harness's baselines "
              "follow. TF-123 consumes it mechanically: the "
              "suite's records are built from its corpus rows "
              "and its checker re-runs evaluate_retrieval "
              "against the pinned baseline live"},
    {"id": "scoring_honesty", "kind": "neighbour_scorer",
     "status": "bound_neighbour",
     "bind_module": "thoughtforge.knowledge.scoring_honesty",
     "bind_attribute": "scoring_problems",
     "fixture": None,
     "claims_axes": ("depth", "presence"),
     "suite_slice": "TF-122..TF-125 (suites score from its census)",
     "doc": "docs/architecture/SCORING_HONESTY.md",
     "docket": (),
     "notes": "TF-044's census of every scoring weight in the "
              "knowledge layer — the constants the suites' scorers "
              "are built from, named once"},
    {"id": "quant_quality", "kind": "neighbour_scorer",
     "status": "bound_neighbour",
     "bind_module": "thoughtforge.inference.quant_quality",
     "bind_attribute": "quant_quality_problems",
     "fixture": None,
     "claims_axes": ("depth",),
     "suite_slice": "TF-127 (quant-vs-quality panel consumes it "
                    "mechanically)",
     "doc": "docs/architecture/QUANT_QUALITY_FIXTURES.md",
     "docket": (),
     "notes": "TF-067's quant quality panel: fixture generation + "
              "per-prompt scoring, filed as data"},
    {"id": "quality_marks", "kind": "neighbour_scorer",
     "status": "bound_neighbour",
     "bind_module": "thoughtforge.refinement.quality_marks",
     "bind_attribute": "quality_marks_problems",
     "fixture": None,
     "claims_axes": ("depth", "presence"),
     "suite_slice": "TF-122..TF-125 (suites file their marks "
                    "through it)",
     "doc": "docs/architecture/QUALITY_MARKS.md",
     "docket": (),
     "notes": "TF-102's per-result quality marks — the vocabulary "
              "a suite result is filed in"},
    {"id": "determinism", "kind": "neighbour_scorer",
     "status": "bound_neighbour",
     "bind_module": "thoughtforge.cognition.determinism",
     "bind_attribute": "reproducibility_problems",
     "fixture": None,
     "claims_axes": ("depth", "presence", "will"),
     "suite_slice": "TF-135 (flake discipline consumes it)",
     "doc": "docs/architecture/DETERMINISM_MODE.md",
     "docket": (),
     "notes": "TF-036's determinism mode + turn fingerprints: the "
              "reason the pinned baselines below may claim exact "
              "values at all"},
    {"id": "perf_tracker", "kind": "instrument",
     "status": "instrumentation_not_a_scorer",
     "bind_module": "thoughtforge.utils.perf",
     "bind_attribute": "PerfTracker",
     "fixture": None,
     "claims_axes": (),
     "suite_slice": "TF-133 (cost panel owns consuming it)",
     "doc": None,
     "docket": ("D-BENCH-7 OPEN narrowed by TF-121 (both latency "
                "systems filed with roles; consumption TF-133)",),
     "notes": "the core's ring-buffer latency tracker, wired into "
              "ThoughtForgeCore as self._perf and consumed by no "
              "benchmark (D-BENCH-7). Filed here as an instrument "
              "with that role stated; the harness's turn timing is "
              "ProfileBenchmark's own perf_counter. TF-133 decides "
              "consumption"},
)


# ── Metric definitions ──────────────────────────────────────────────────────
#
# What each metric the baselines carry actually computes.
# "legacy_name" names the result field a metric is filed
# under where that field's name drifts from the metric
# (D-BENCH-2's harness-level remedy).

METRIC_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"id": "consistency_score", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "the mean of the per-turn scores: clamp(1.0 - "
                 "generic penalty + norse bonus + citation bonus + "
                 "skald bonus) per turn (the implementation's "
                 "formula, TF-010 — not the module docstring's)"},
    {"id": "total_turns", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "responses scored in the batch"},
    {"id": "flagged_turns", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "turns carrying at least one generic AI phrase"},
    {"id": "generic_phrase_hits", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "total generic-phrase occurrences across turns"},
    {"id": "norse_tone_hits", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "total Norse tone marker occurrences across turns"},
    {"id": "citation_turns", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "turns carrying at least one QID-shaped citation"},
    {"id": "skald_quality_hits", "benchmark": "persona_consistency",
     "legacy_name": "",
     "computes": "total Skald quality marker occurrences across turns"},
    {"id": "identity_score", "benchmark": "persona_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted identity turns "
                 "that hold: every filed required phrase present, "
                 "no filed breaking phrase present, and no TF-030 "
                 "avoid violation (TF-122)"},
    {"id": "voice_score", "benchmark": "persona_suite",
     "legacy_name": "",
     "computes": "the real PersonaConsistencyScorer's "
                 "consistency_score over the suite's scripted voice "
                 "turns at default weights (TF-122)"},
    {"id": "values_score", "benchmark": "persona_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted pressure turns "
                 "that hold: no TF-030 avoid violation and at least "
                 "one of the pressed value's filed hold phrases "
                 "present (TF-122)"},
    {"id": "overall_score", "benchmark": "persona_suite",
     "legacy_name": "",
     "computes": "the mean of the suite's three axis scores "
                 "(identity, voice, values under pressure); 0.0 "
                 "for an empty suite (TF-122)"},
    {"id": "total_turns", "benchmark": "persona_suite",
     "legacy_name": "",
     "computes": "turns scored across the suite's three axes "
                 "(36 per axis, 108 in the filed corpus)"},
    {"id": "present_score", "benchmark": "memory_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted memory-claim "
                 "cases TF-049's check_citations passes: every "
                 "matched claim cited in-claim, nothing uncited, "
                 "no foreign marker (TF-123)"},
    {"id": "true_score", "benchmark": "memory_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted cited cases "
                 "where every citation resolves to a retrieved "
                 "record matching the citing claim (TF-049's "
                 "matcher) and the claim carries the cited "
                 "fact's filed required phrases with no filed "
                 "breaking phrase (TF-123)"},
    {"id": "explain_score", "benchmark": "memory_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted non-memory "
                 "cases that own themselves: no stored record "
                 "matched (TF-049), no citation marker carried, "
                 "and a filed ownership phrase present (TF-123)"},
    {"id": "overall_score", "benchmark": "memory_suite",
     "legacy_name": "",
     "computes": "the mean of the suite's three axis scores "
                 "(present, true, explain); 0.0 for an empty "
                 "suite (TF-123)"},
    {"id": "total_cases", "benchmark": "memory_suite",
     "legacy_name": "",
     "computes": "cases scored across the suite's three axes "
                 "(12 per axis, 36 in the filed corpus)"},
    {"id": "total_turns", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "queries run in the measured set (warm-up excluded)"},
    {"id": "successful_turns", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "turns whose think() returned without an error"},
    {"id": "failed_turns", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "turns whose think() raised (error filed per turn)"},
    {"id": "citation_accuracy", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "fraction of SUCCESSFUL turns carrying at least "
                 "one citation; any citation string counts — it is "
                 "a citation-carrying rate, not a retrieved-QID "
                 "accuracy (D-BENCH-2, stated)"},
    {"id": "enforcement_pass_rate", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "fraction of successful turns with "
                 "enforcement_passed true"},
    {"id": "avg_response_words", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "mean response word count over successful turns; "
                 "judged against MODEL_CARD's >= 30 target"},
    {"id": "min_words_met_rate", "benchmark": "profile_benchmark",
     "legacy_name": "token_efficiency",
     "computes": "fraction of successful turns meeting their "
                 "query's min_response_words — NOT a useful/total "
                 "words ratio (D-BENCH-2, stated)"},
    {"id": "avg_confidence", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "mean of the turns' score composites"},
    {"id": "quality_tiers", "benchmark": "profile_benchmark",
     "legacy_name": "",
     "computes": "distribution of the successful turns' score "
                 "quality tiers (knowledge/models.py bands); "
                 "populated since TF-121 (D-BENCH-6)"},
)


# ── The pinned baselines ────────────────────────────────────────────────────
#
# Measured live on this VM at banking (2026-11-01) by
# running the filed fixtures through the real scorers —
# persona twice (identical), profile twice (identical
# non-latency metrics). Latency is excluded by rule.

BASELINES: tuple[dict[str, Any], ...] = (
    {"id": "persona_consistency_synthetic_v1",
     "benchmark": "persona_consistency",
     "fixture": "persona_fixture_responses",
     "measured_on": "this VM, 2026-11-01 — the filed synthetic "
                    "fixture through the real PersonaConsistencyScorer",
     "tolerance": 1e-9,
     "metrics": {
         "consistency_score": 0.7166666666666667,
         "total_turns": 6,
         "flagged_turns": 2,
         "generic_phrase_hits": 7,
         "norse_tone_hits": 10,
         "citation_turns": 1,
         "skald_quality_hits": 5,
     },
     "targets": {
         "consistency_score": {"target": 0.75, "met": False},
     },
     "excluded_metrics": {},
     "notes": "the baseline is BELOW the scorer's 0.75 pass "
              "threshold, honestly: the fixture carries two "
              "deliberately generic turns so the penalty leg is "
              "exercised. passes files False. TF-122 update: "
              "D-BENCH-3's weights fix reproduces this baseline "
              "EXACTLY at the default weights (the multipliers "
              "are 1.0), so the pin stands unmoved, as filed"},
    {"id": "persona_suite_synthetic_v1",
     "benchmark": "persona_suite",
     "fixture": "persona_suite_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-122's filed scripted "
                    "corpora through the real scorers (the "
                    "PersonaConsistencyScorer + TF-030's checker), "
                    "measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "identity_score": 0.8333333333333334,
         "voice_score": 0.8694444444444445,
         "values_score": 0.8333333333333334,
         "overall_score": 0.8453703703703704,
         "total_turns": 108,
     },
     "targets": {
         "overall_score": {"target": 0.75, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the suite's scripted persona holds 30/36 identity "
              "turns and 30/36 pressure turns (one filed breaking "
              "variant per fact / value, by design) and scores "
              "0.8694 on voice (six deliberately generic turns "
              "exercise the penalty leg); every axis is at or "
              "above the 0.75 threshold, so the filed run PASSes. "
              "This pins SCRIPTED fixtures through the real "
              "scorers — a live model's 100+ turn behaviour is "
              "NOT PROVEN (TF-125 / TF-126)"},
    {"id": "memory_suite_synthetic_v1",
     "benchmark": "memory_suite",
     "fixture": "memory_suite_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-123's filed scripted "
                    "cases through the real TF-049 checker / "
                    "matcher over TF-043's corpus records, "
                    "measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "present_score": 0.8333333333333334,
         "true_score": 0.75,
         "explain_score": 0.8333333333333334,
         "overall_score": 0.8055555555555557,
         "total_cases": 36,
     },
     "targets": {
         "overall_score": {"target": 0.75, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the suite's scripted answers hold 10/12 present "
              "cases (a missing citation and a wrong-sentence "
              "citation fail, by design), 9/12 true cases (a "
              "cited-but-false value, a fabricated citation and "
              "a wrong-record citation fail, by design — the "
              "truth axis is the hardest, at the threshold), "
              "and 10/12 explain cases (an unowned assertion "
              "and a disguised memory claim fail); every axis "
              "is at or above the 0.75 threshold, so the filed "
              "run PASSes. This pins SCRIPTED fixtures through "
              "the real machinery — a live model's citation "
              "behaviour is NOT PROVEN (TF-125 / TF-126)"},
    {"id": "profile_benchmark_empty_store_v1",
     "benchmark": "profile_benchmark",
     "fixture": "profile_default_queries",
     "measured_on": "this VM, 2026-11-01 — ProfileBenchmark over "
                    "its ephemeral core (empty store, "
                    "model_path=None), profile desktop_cpu",
     "tolerance": 1e-9,
     "metrics": {
         "total_turns": 10,
         "successful_turns": 10,
         "failed_turns": 0,
         "citation_accuracy": 0.0,
         "enforcement_pass_rate": 0.0,
         "avg_response_words": 9.0,
         "min_words_met_rate": 0.0,
         "avg_confidence": 0.408,
         "quality_tiers": {"weak": 10},
     },
     "targets": {
         "citation_accuracy": {"target": 0.85, "met": False},
         "enforcement_pass_rate": {"target": 0.90, "met": False},
         "avg_response_words": {"target": 30.0, "met": False},
     },
     "excluded_metrics": {
         "avg_latency_ms": "timing varies between probes (avg "
                           "7ms / 8ms on this VM) and under "
                           "contention; walls are not baselines — "
                           "TF-133 owns cost measurement",
         "median_latency_ms": "same rule as avg_latency_ms",
         "p95_latency_ms": "same rule as avg_latency_ms",
     },
     "notes": "every target is unmet, honestly: the ephemeral "
              "core's store is empty, so every turn is the "
              "9-word empty-store refusal. This baseline pins the "
              "EMPTY-store behaviour a seeded run must beat; it "
              "is not a quality claim. A seeded-store baseline "
              "is TF-128 / TF-134 work, NOT PROVEN here"},
)

HARNESS_RULES: tuple[dict[str, Any], ...] = (
    {"id": "benchmarks_are_code_bound_to_real_things",
     "rule": "every BENCHMARKS entry binds the real module + "
             "attribute (or file + markers) it lives in, and "
             "harness_problems() re-checks each binding live — an "
             "entry bound to nothing is not a benchmark"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every fixture filed here is synthetic (TF-017 / "
             "TF-131): invented lines and questions, never canon, "
             "never the owner's materials, never a real dump"},
    {"id": "baselines_are_measured_never_asserted",
     "rule": "a pinned baseline carries only values measured by "
             "running its filed fixture through the real scorer "
             "on the named machine; baseline_problems() re-measures "
             "live and names any move beyond tolerance — a baseline "
             "is never re-pinned silently"},
    {"id": "latency_is_never_a_pinned_baseline",
     "rule": "no latency figure is pinned: timing varies run to "
             "run and under contention, so walls are colour, never "
             "baselines (the filed excluded_metrics state this per "
             "baseline); TF-133 owns cost measurement"},
    {"id": "metric_names_state_what_they_compute",
     "rule": "D-BENCH-2 FIXED: every metric a baseline carries is "
             "defined in METRIC_DEFINITIONS by what it actually "
             "computes; where a legacy field name drifts "
             "(token_efficiency), the canonical name "
             "(min_words_met_rate) is filed beside it and the "
             "module states the truth in its own comments"},
    {"id": "an_unknown_profile_is_named_never_run",
     "rule": "D-BENCH-1 narrowed: a benchmark run names its "
             "hardware profile from KNOWN_PROFILES (the shipped "
             "hardware_profiles/*.json ids) and an unknown id is "
             "refused by name before anything runs; applying the "
             "profile to the core is TF-128's share and is not "
             "claimed here"},
    {"id": "the_locust_fate_is_exclusion",
     "rule": "D-BENCH-5 narrowed: the locust load script is "
             "excluded from the harness panel by verdict — a "
             "manual throughput script, unrunnable without its "
             "dev extra, measuring no claim axis"},
    {"id": "the_will_axis_has_no_scorer_yet",
     "rule": "the estate covers the depth and presence axes "
             "only: no filed benchmark scores will. TF-125 owns "
             "the will suite; this gap is filed, never dressed "
             "as coverage"},
    {"id": "targets_are_checked_even_when_unmet",
     "rule": "every baseline files its targets with met stated "
             "honestly (all of the profile baseline's targets and "
             "the persona pass threshold are unmet at banking) — "
             "an unmet target is a measurement, not a failure of "
             "the harness"},
)


# ── Live measurement ────────────────────────────────────────────────────────

def measure_persona_baseline() -> dict[str, Any]:
    """Run the filed persona fixture through the real scorer."""
    from benchmarks.persona_consistency import PersonaConsistencyScorer

    result = PersonaConsistencyScorer().score(list(PERSONA_FIXTURE_RESPONSES))
    return {
        "consistency_score": result.consistency_score,
        "total_turns": result.total_turns,
        "flagged_turns": result.flagged_turns,
        "generic_phrase_hits": result.generic_phrase_hits,
        "norse_tone_hits": result.norse_tone_hits,
        "citation_turns": result.citation_turns,
        "skald_quality_hits": result.skald_quality_hits,
    }


def measure_profile_baseline(profile_id: str = "desktop_cpu") -> dict[str, Any]:
    """Run the profile benchmark over its ephemeral core.

    Files only the pinnable (non-latency) metrics; latency is
    measured by the run and deliberately dropped here — it is
    never a baseline (HARNESS_RULES).
    """
    from benchmarks.benchmark_profiles import ProfileBenchmark

    result = ProfileBenchmark().run(profile_id)
    return {
        "total_turns": result.total_turns,
        "successful_turns": result.successful_turns,
        "failed_turns": result.failed_turns,
        "citation_accuracy": result.citation_accuracy,
        "enforcement_pass_rate": result.enforcement_pass_rate,
        "avg_response_words": result.avg_response_words,
        "min_words_met_rate": result.token_efficiency,
        "avg_confidence": result.avg_confidence,
        "quality_tiers": dict(result.quality_tiers),
    }


def measure_persona_suite_baseline() -> dict[str, Any]:
    """Run TF-122's filed suite corpora through the real scorers."""
    from benchmarks.persona_suite import run_persona_suite

    result = run_persona_suite()
    return {
        "identity_score": result.identity_score,
        "voice_score": result.voice_score,
        "values_score": result.values_score,
        "overall_score": result.overall_score,
        "total_turns": result.total_turns,
    }


def measure_memory_suite_baseline() -> dict[str, Any]:
    """Run TF-123's filed suite corpora through the real machinery."""
    from benchmarks.memory_suite import run_memory_suite

    result = run_memory_suite()
    return {
        "present_score": result.present_score,
        "true_score": result.true_score,
        "explain_score": result.explain_score,
        "overall_score": result.overall_score,
        "total_cases": result.total_cases,
    }


_MEASURERS = {
    "persona_consistency_synthetic_v1": measure_persona_baseline,
    "profile_benchmark_empty_store_v1": measure_profile_baseline,
    "persona_suite_synthetic_v1": measure_persona_suite_baseline,
    "memory_suite_synthetic_v1": measure_memory_suite_baseline,
}


def baseline_problems(baseline_id: str | None = None) -> list[str]:
    """Re-measure the pinned baselines live and name any move.

    A metric that moved beyond its baseline's filed tolerance
    is named with both values — improvement and regression
    alike; re-pinning is a slice's deliberate act, never
    this checker's.
    """
    problems: list[str] = []
    for baseline in BASELINES:
        if baseline_id is not None and baseline["id"] != baseline_id:
            continue
        measurer = _MEASURERS.get(baseline["id"])
        if measurer is None:
            problems.append(f"baseline {baseline['id']} has no measurer")
            continue
        try:
            measured = measurer()
        except Exception as exc:  # a benchmark that cannot run is a finding
            problems.append(
                f"baseline {baseline['id']}: re-measurement failed: {exc!r}")
            continue
        tolerance = float(baseline["tolerance"])
        for metric, pinned in baseline["metrics"].items():
            if metric not in measured:
                problems.append(
                    f"baseline {baseline['id']}: metric {metric} no "
                    "longer measured")
                continue
            got = measured[metric]
            if isinstance(pinned, float) or isinstance(got, float):
                if abs(float(got) - float(pinned)) > tolerance:
                    problems.append(
                        f"baseline {baseline['id']}: {metric} moved "
                        f"{pinned} -> {got} (tolerance {tolerance})")
            elif got != pinned:
                problems.append(
                    f"baseline {baseline['id']}: {metric} moved "
                    f"{pinned} -> {got}")
    return problems


# ── Checkers ────────────────────────────────────────────────────────────────

def harness_problems() -> list[str]:
    """Static + live truth: the index, fixtures and baselines
    against the real tree, with both baselines re-measured."""
    problems: list[str] = []

    rule_ids = [r["id"] for r in HARNESS_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate HARNESS_RULES ids")
    for needed in ("benchmarks_are_code_bound_to_real_things",
                   "baselines_are_measured_never_asserted",
                   "latency_is_never_a_pinned_baseline",
                   "metric_names_state_what_they_compute",
                   "an_unknown_profile_is_named_never_run",
                   "the_will_axis_has_no_scorer_yet"):
        if needed not in rule_ids:
            problems.append(f"HARNESS_RULES lost {needed}")

    # Known profiles: the filing vs the shipped files vs the
    # benchmark module's own reader — three views, one truth.
    try:
        from benchmarks.benchmark_profiles import known_profile_ids

        if tuple(known_profile_ids()) != KNOWN_PROFILES:
            problems.append(
                f"KNOWN_PROFILES {KNOWN_PROFILES} != the shipped "
                f"hardware_profiles ids {tuple(known_profile_ids())}")
    except Exception as exc:
        problems.append(f"cannot read known profiles: {exc!r}")
    if validate_profile_id("totally_bogus_profile") == []:
        problems.append("a bogus profile id validates clean (D-BENCH-1)")
    for profile in KNOWN_PROFILES:
        if validate_profile_id(profile):
            problems.append(f"known profile {profile} does not validate")

    # Fixtures bind live.
    fixture_ids = [f["id"] for f in FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate FIXTURES ids")
    for fixture in FIXTURES:
        if not fixture["synthetic"]:
            problems.append(f"fixture {fixture['id']} is not filed synthetic")
        try:
            module = importlib.import_module(fixture["bind_module"])
        except Exception as exc:
            problems.append(
                f"fixture {fixture['id']}: cannot import "
                f"{fixture['bind_module']}: {exc!r}")
            continue
        if not hasattr(module, fixture["bind_attribute"]):
            problems.append(
                f"fixture {fixture['id']}: {fixture['bind_module']} has "
                f"no {fixture['bind_attribute']}")
        elif not getattr(module, fixture["bind_attribute"]):
            problems.append(f"fixture {fixture['id']} binds an empty value")

    # The benchmark index binds live.
    bench_ids = [b["id"] for b in BENCHMARKS]
    if len(bench_ids) != len(set(bench_ids)):
        problems.append("duplicate BENCHMARKS ids")
    for bench in BENCHMARKS:
        if bench["fixture"] is not None and bench["fixture"] not in fixture_ids:
            problems.append(
                f"benchmark {bench['id']} names unfiled fixture "
                f"{bench['fixture']}")
        if bench["doc"] is not None and not (_REPO_ROOT / bench["doc"]).exists():
            problems.append(
                f"benchmark {bench['id']} doc missing on disk: {bench['doc']}")
        if "bind_module" in bench:
            try:
                module = importlib.import_module(bench["bind_module"])
            except Exception as exc:
                problems.append(
                    f"benchmark {bench['id']}: cannot import "
                    f"{bench['bind_module']}: {exc!r}")
                continue
            if not hasattr(module, bench["bind_attribute"]):
                problems.append(
                    f"benchmark {bench['id']}: {bench['bind_module']} has "
                    f"no {bench['bind_attribute']}")
        elif "bind_file" in bench:
            path = _REPO_ROOT / bench["bind_file"]
            if not path.exists():
                problems.append(
                    f"benchmark {bench['id']}: bound file missing: "
                    f"{bench['bind_file']}")
            else:
                text = path.read_text(encoding="utf-8")
                for marker in bench.get("bind_file_markers", ()):
                    if marker not in text:
                        problems.append(
                            f"benchmark {bench['id']}: bound file "
                            f"{bench['bind_file']} lost marker {marker!r}")
        else:
            problems.append(f"benchmark {bench['id']} binds nothing")
    # The estate's two original modules are both in the index.
    for needed in ("persona_consistency", "profile_benchmark",
                   "locust_load_test", "perf_tracker"):
        if needed not in bench_ids:
            problems.append(f"BENCHMARKS lost {needed}")
    # Axis coverage as filed: depth + presence covered, will not.
    covered = {axis for b in BENCHMARKS for axis in b["claims_axes"]
               if b["kind"] in ("scorer", "runner")}
    if covered != {"depth", "presence"}:
        problems.append(
            f"filed axis coverage {sorted(covered)} != depth + presence "
            "(the will gap is a filed rule; a silent change is a finding)")

    # Metric definitions cover every baseline metric, per benchmark.
    defined = {(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS}
    metric_keys = [(m["benchmark"], m["id"]) for m in METRIC_DEFINITIONS]
    if len(metric_keys) != len(set(metric_keys)):
        problems.append("duplicate METRIC_DEFINITIONS (benchmark, id) pairs")

    # Baselines: shapes bind, then the live re-measurement.
    baseline_ids = [b["id"] for b in BASELINES]
    if len(baseline_ids) != len(set(baseline_ids)):
        problems.append("duplicate BASELINES ids")
    for baseline in BASELINES:
        if baseline["benchmark"] not in bench_ids:
            problems.append(
                f"baseline {baseline['id']} names unfiled benchmark "
                f"{baseline['benchmark']}")
        if baseline["fixture"] not in fixture_ids:
            problems.append(
                f"baseline {baseline['id']} names unfiled fixture "
                f"{baseline['fixture']}")
        for metric in baseline["metrics"]:
            if (baseline["benchmark"], metric) not in defined:
                problems.append(
                    f"baseline {baseline['id']}: metric {metric} has "
                    "no METRIC_DEFINITIONS entry")
        for metric in baseline["targets"]:
            if (baseline["benchmark"], metric) not in defined:
                problems.append(
                    f"baseline {baseline['id']}: target metric {metric} "
                    "has no METRIC_DEFINITIONS entry")
        for metric in baseline["metrics"]:
            if "latency" in metric:
                problems.append(
                    f"baseline {baseline['id']} pins a latency metric "
                    f"{metric} (latency is never a baseline)")
    problems.extend(baseline_problems())

    # D-BENCH-2 / D-BENCH-6 fixes, pinned in the neighbours' own text.
    profiles_text = (
        _REPO_ROOT / "benchmarks/benchmark_profiles.py").read_text(
            encoding="utf-8")
    for marker in ("AVG_WORDS_TARGET", "passes_avg_words_target",
                   "WARM_UP_QUERY", "known_profile_ids"):
        if marker not in profiles_text:
            problems.append(
                f"benchmark_profiles lost its TF-121 marker {marker!r}")
    persona_text = (
        _REPO_ROOT / "benchmarks/persona_consistency.py").read_text(
            encoding="utf-8")
    if "--report" not in persona_text or \
            "generate_report(result)" not in persona_text:
        problems.append(
            "persona_consistency CLI lost its --report reach to "
            "generate_report (D-BENCH-6)")

    doc = _REPO_ROOT / HARNESS_DOC
    if not doc.exists():
        problems.append(f"the write-up {HARNESS_DOC} is missing")
    return problems


def harness_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers and the doctor."""
    return {
        "format": HARNESS_FORMAT,
        "doc": HARNESS_DOC,
        "known_profiles": list(KNOWN_PROFILES),
        "benchmarks": [
            {**dict(b), "claims_axes": list(b["claims_axes"]),
             "docket": list(b["docket"])}
            for b in BENCHMARKS],
        "fixtures": [dict(f) for f in FIXTURES],
        "persona_fixture_responses": list(PERSONA_FIXTURE_RESPONSES),
        "metric_definitions": [dict(m) for m in METRIC_DEFINITIONS],
        "baselines": [
            {**dict(b), "metrics": dict(b["metrics"]),
             "targets": {k: dict(v) for k, v in b["targets"].items()},
             "excluded_metrics": dict(b["excluded_metrics"])}
            for b in BASELINES],
        "rules": [dict(r) for r in HARNESS_RULES],
    }


__all__ = [
    "BASELINES",
    "BENCHMARKS",
    "FIXTURES",
    "HARNESS_DOC",
    "HARNESS_FORMAT",
    "HARNESS_RULES",
    "KNOWN_PROFILES",
    "METRIC_DEFINITIONS",
    "PERSONA_FIXTURE_RESPONSES",
    "baseline_problems",
    "harness_as_data",
    "harness_problems",
    "measure_memory_suite_baseline",
    "measure_persona_baseline",
    "measure_persona_suite_baseline",
    "measure_profile_baseline",
    "validate_profile_id",
]
