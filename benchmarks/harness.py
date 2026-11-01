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

**TF-124 update, in the open:** the depth suite is
built (``benchmarks/depth_suite.py``) and registered
below (``depth_suite``) with its own pinned baseline
(``depth_suite_synthetic_v1``, measured live twice,
identical): 36 scripted multi-step problems scored
on both roads through TF-022's real selection table
and TF-021 / TF-022's real prompt builders — OFF
(single-pass) 0.4722, ON (scaffolded) 0.8056, the
measured delta +0.3333, filed with its no-gain and
scaffold-loss cases by name. TF-124's probes found
and filed **D-BENCH-10** (scaffold value measured by
nothing, and TF-022's core-execution wiring assigned
in its doc without ever being numbered) and FIXED
it in the measuring sense — the suite now scores
the on/off delta mechanically; core-loop strategy
execution is unchanged and NOT claimed. All earlier
pinned baselines stand exactly as filed (never
re-pinned).

**TF-125 update, in the open:** the presence suite
is built (``benchmarks/presence_suite.py``) and
registered below (``presence_suite``) with its own
pinned baseline (``presence_suite_synthetic_v1``,
measured live twice, identical): 36 scripted sagas
scored through TF-028's real coherence scorer and
TF-029's real ``GoalLedger`` — coherence 0.75,
will steering 0.8333, will closure 0.75, overall
0.7778, PASS. This discharges the will gap this
module filed at TF-121 as the rule
``the_will_axis_has_no_scorer_yet``: the rule is
replaced below by ``the_will_axis_is_scored_by_tf125``.
TF-125's probes found and filed **D-BENCH-11**
(presence and will follow-through scored by
nothing in the benchmarks estate, the gap filed
as a harness rule and assigned in TF-028's doc but
never numbered in the TF-016 docket — the
D-BENCH-9 / D-BENCH-10 pattern) and FIXED it in
the measuring sense. All earlier pinned baselines
stand exactly as filed (never re-pinned).

**TF-126 update, in the open:** the small-model
honesty panel is built
(``benchmarks/honesty_panel.py``) and registered
below (``honesty_panel``) with its own pinned
baseline (``honesty_panel_synthetic_v1``, measured
live twice, identical): the four suites' filed
corpora degraded by four filed size tiers
(reference 14B / mid 8B / small 3B / tiny 1.1B,
each anchored live to a catalogue model) and
scored through the suites' real runners, OFF
(degraded) against ON (degraded + the filed
per-suite forge repairs) — 16 cells, published as
literal data recomputed live by its checker. The
forge helps most at the mid tier overall (mean
delta +0.2400), not the smallest: per suite,
persona is helped most at tiny (+0.1531), memory
at small (+0.6111), depth and presence at mid —
and depth's delta goes negative at small
(-0.1944), filed by name. TF-126's probes found
and filed **D-BENCH-12** (the suites never run
across model sizes; the gap assigned to TF-126
by TF-122..TF-125's filed limits without ever
being numbered — the D-BENCH-9 / D-BENCH-10 /
D-BENCH-11 pattern) and FIXED it in the measuring
sense. The four suites' baseline notes below now
file TF-126 DONE in the measuring sense; live
per-size behaviour of real models stays NOT
PROVEN — no GGUF / llama_cpp on this VM (TF-078);
TF-173 owns the real-hardware road. All earlier
pinned baselines stand exactly as filed (never
re-pinned).

**TF-127 update, in the open:** the
quant-vs-quality panel is built
(``benchmarks/quant_panel.py``) and registered
below (``quant_panel``) with its own pinned
baseline (``quant_panel_synthetic_v1``, measured
live twice, identical): TF-067's fixture panel
consumed mechanically — its own
``run_quant_panel`` over its own
``fixture_generate`` / ``QUANT_LEVELS`` /
``QUANT_PROMPTS``, no corpus, scorer or curve
copied — published as the overall rows (TF-067's
``CURVE`` in row form) plus the panel's own
per-kind breakdown (18 rows: fact / reasoning /
instruction per level, each kind's loss against
its own fp16 mean), recomputed live by its
checker. The finding, filed whatever it was:
reasoning carries the largest per-kind loss at
both mid levels (0.5 at q4_k_m and q5_k_m) and
plateaus at 0.5 from q2_k through q5_k_m, while
instruction collapses to 0.0 at q2_k on its
single prompt; the filed non-monotone q4_k_m /
q5_k_m step is carried, never smoothed. TF-127's
probes found and filed **D-BENCH-13**
(quant-vs-quality measured only as TF-067's
single overall curve in ``src/`` — never as a
Part VII panel with a per-kind breakdown or a
harness baseline; the gap assigned to TF-127 in
TF-067's doc without ever being numbered — the
D-BENCH-9 / D-BENCH-10 / D-BENCH-11 / D-BENCH-12
pattern) and FIXED it in the measuring sense.
The ``quant_quality`` entry's suite-slice note
below now files TF-127 DONE in the measuring
sense. Live quant-vs-quality of real models
stays NOT PROVEN — no GGUF / llama_cpp on this
VM (TF-078) — so TF-068's live re-rank /
q2_k-calibration triggers stand open on
TF-173's road, discharged by nothing here. All
earlier pinned baselines stand exactly as filed
(never re-pinned).

**TF-128 update, in the open:** the profile
panels are built
(``benchmarks/profile_panel.py``) and registered
below (``profile_panel``) with their own pinned
baseline (``profile_panel_synthetic_v1``,
measured live twice, identical): the four
suites and the profile benchmark run per
shipped profile with the profile applied to a
real core and its budgets enforced during the
run — the applied core's resolved tokens /
passes and lifecycle caps asserted equal to the
profile JSON's filed budgets per profile,
TF-069's boot + ``check_model_fit`` RAM-ceiling
cap row re-run per profile (desktop_cpu /
desktop_gpu / pi_5 fit; server_gpu's cap
exceeds its own ceiling at 140.0 > 88 GB;
phone_low / pi_zero unprovable, q2_k
uncalibrated), and the depth suite's ON road
gated by the profile's refinement budget —
pi_zero budgets 0 passes, so its ON cell files
refused-by-budget and its depth score is its
OFF score. The measured finding: no score
varies by profile except through that refusal
— the benchmark cell is an empty-store run and
the suite scorers take no profile input, filed
plainly. This discharges **D-BENCH-1**'s
application share, FIXED in the measuring
sense; the ``profile_benchmark`` entry's
suite-slice note below now files TF-128 DONE
and the unknown-profile rule below files the
application discharged. Live per-profile
hardware behaviour stays NOT PROVEN (TF-082 /
TF-173), and ``benchmark_profiles``' default
ephemeral core is unchanged (the panel supplies
the applied core through its ``core=``
parameter). All earlier pinned baselines stand
exactly as filed (never re-pinned).

**TF-129 update, in the open:** the regression
gates are built
(``benchmarks/regression_gate.py``) and
registered below (``regression_gate``) with
their own pinned baseline
(``regression_gate_synthetic_v1``, measured
live twice, identical): every gated baseline
re-measured NOISE_REPEATS (5) times in one
process through this harness's own measurers —
every spread measured 0.0 on this tree, the
estate's determinism (TF-036) measured rather
than assumed — and the gate tolerances derived
from that noise by the filed rule (per float
metric, max(measured spread, this harness's
1e-9 filed floor); count and dict metrics
exact). The gate evaluates per-baseline /
per-metric deltas with classifications
(within tolerance / regression / improvement /
count change / unmeasurable) and files PASS
only on the full nine-baseline selection with
every metric within tolerance — any
beyond-tolerance move fails, improvements
included, and a partial or empty selection
files REVIEW. Seven filed probes prove the
gate can fail against its pure evaluator with
synthetic perturbations; real baselines are
never perturbed. TF-129's probes found and
filed **D-BENCH-14** (pinned baselines existed
and baseline_problems named moves, but no gate
consumed them — nothing classified deltas,
nothing could fail a slice, and tolerances
were uniform and asserted rather than
noise-derived; the gap stood filed as future
work in this module's own limits passages and
was assigned to TF-129 by the roadmap without
ever being numbered — the D-BENCH-9..13
pattern) and FIXED it. The gate judges these
pinned fixture baselines only — no live model
is gated (TF-173); it is a checker a slice
runs in-house, not a CI wiring, and it wires
no generation-time enforcement blocking
(D-BENCH-9's generation-time share is refused
openly in TF-129's doc). All earlier pinned
baselines stand exactly as filed (never
re-pinned).

**TF-130 update, in the open:** the human-shaped
spot checks are built
(``benchmarks/spot_checks.py``) and registered
below (``spot_checks``) with their own pinned
baseline (``spot_checks_synthetic_v1``,
measured live twice, identical): eight filed
transcripts assembled live from the real filed
corpora of the persona / memory / depth /
presence suites and TF-067's quant estate —
no corpus or scorer copied; every turn is the
suite's own case data rendered verbatim and
every machine score is recomputed by that
suite's real per-case scorer. Four transcripts
hold and four do not, as filed; each carries
its score + verdict beside a human-judgement
prompt (what Volmarr should look for), and the
rendered set IS the doc
(``docs/architecture/SPOT_CHECKS.md``) — the
numbers' sanity anchor, checked verbatim by
``spot_checks_problems()``. TF-130's probes
found and filed **D-BENCH-15** (the suites'
numbers existed as pinned baselines, but no
artefact ever assembled suite cases into
transcripts a human could read and judge —
assigned to TF-130 by the roadmap without ever
being numbered, the D-BENCH-9..14 pattern) and
FIXED it in the measuring sense. The
spot-check baseline is NOT one of TF-129's
gated baselines: the gate's gated set stays
the nine TF-121..TF-128 baselines exactly as
filed, and the exclusion is filed in the open
in ``benchmarks/regression_gate.py`` (the
shape TF-129 filed for its own baseline — a
verdict row judged by its own module's
checker). No human judgement is recorded yet
— the set is the instrument; live-model
transcripts are NOT PROVEN (TF-173). All
earlier pinned baselines stand exactly as
filed (never re-pinned).

Filed limits, stated plainly:

  * The profile baseline measures an ephemeral core
    over an EMPTY store with ``model_path=None`` — the
    empty-store refusal (9 words, confidence 0.408),
    never inference, never a loaded model. A seeded
    store and a loaded model are NOT PROVEN here.
  * The will axis had no scorer anywhere in the
    estate at TF-121: ``BENCHMARKS`` covered depth
    and presence only, and the gap was filed as a
    rule below, not dressed as coverage. **TF-125
    update:** the will axis now has its scorer —
    TF-125's presence suite scores will steering
    and will closure through TF-029's real ledger,
    the ``presence_suite`` entry below carries the
    will claims axis, and the rule below files
    TF-125 DONE. A live model's will follow-through
    is NOT PROVEN (TF-126). **TF-126 update:** the
    honesty panel now runs the will axes across
    filed size tiers in the measuring sense; a live
    model's follow-through is still NOT PROVEN
    (TF-173).
  * No suite in Part VII existed at TF-121: TF-122..
    TF-125 build them. **TF-122 update:** the first
    suite (persona consistency) now exists and is
    registered below; TF-123..TF-125's suites, TF-134's baseline book and
    TF-137's one-command tool still stand as filed.
    **TF-123 update:** the second suite (memory
    fidelity) now exists and is registered below;
    TF-124..TF-125's suites, TF-134's baseline book and TF-137's
    one-command tool still stand as filed.
    **TF-124 update:** the third suite (depth) now
    exists and is registered below; TF-125's suite,
    TF-134's baseline
    book and TF-137's one-command tool still stand
    as filed.
    **TF-125 update:** the fourth suite (presence)
    now exists and is registered below; TF-134's baseline book and
    TF-137's one-command tool still stand as filed.
    **TF-126 update:** the honesty panel over the
    four suites now exists and is registered below;
    TF-134's baseline
    book and TF-137's one-command tool still stand
    as filed.
    **TF-127 update:** the quant-vs-quality panel
    over TF-067's fixture estate now exists and is
    registered below; TF-134's baseline book and TF-137's one-command
    tool still stand as filed.
    **TF-128 update:** the profile panels over
    the six shipped profiles now exist and are
    registered below; TF-134's baseline book and TF-137's one-command
    tool still stand as filed.
    **TF-129 update:** the regression gates over
    the nine pinned baselines now exist and are
    registered below; TF-134's baseline book and
    TF-137's one-command tool still stand as filed.
    **TF-130 update:** the human-shaped spot
    checks over the suites' own filed corpora now
    exist and are registered below; TF-134's baseline book and
    TF-137's one-command tool still stand as filed.

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
    {"id": "depth_suite_corpus", "benchmark": "depth_suite",
     "bind_module": "benchmarks.depth_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "synthetic": True,
     "subject": "TF-124's scripted depth corpora: 12 chain "
                "cases + 12 plan cases + 12 check cases "
                "(36 scored problems, each on both roads)"},
    {"id": "presence_suite_corpus", "benchmark": "presence_suite",
     "bind_module": "benchmarks.presence_suite",
     "bind_attribute": "ALL_SUITE_CASES",
     "synthetic": True,
     "subject": "TF-125's scripted presence corpora: 12 "
                "coherence sagas + 12 will-steering sagas + "
                "12 will-closure sagas (36 scored sagas)"},
    {"id": "honesty_panel_corpus", "benchmark": "honesty_panel",
     "bind_module": "benchmarks.honesty_panel",
     "bind_attribute": "MODEL_TIERS",
     "synthetic": True,
     "subject": "TF-126's filed size tiers: 4 tiers "
                "(reference 14B / mid 8B / small 3B / "
                "tiny 1.1B) anchored to catalogue models, "
                "degrading the four suites' own filed "
                "corpora (16 scored cells)"},
    {"id": "quant_panel_corpus", "benchmark": "quant_panel",
     "bind_module": "benchmarks.quant_panel",
     "bind_attribute": "PANEL_ROWS",
     "synthetic": True,
     "subject": "TF-127's published panel rows over "
                "TF-067's synthetic fixture corpus: "
                "6 quant levels x 8 prompts (48 scored "
                "cells), bound in place — the corpus "
                "itself stays TF-067's, consumed "
                "mechanically, never copied"},
    {"id": "profile_panel_corpus", "benchmark": "profile_panel",
     "bind_module": "benchmarks.profile_panel",
     "bind_attribute": "PANEL_ROWS",
     "synthetic": True,
     "subject": "TF-128's published panel rows over "
                "the six shipped hardware profiles: "
                "per profile the applied-core "
                "benchmark over the profile "
                "benchmark's own filed queries plus "
                "the four suites' own filed corpora "
                "(30 filed cells, 1 refused by "
                "budget), bound in place — the "
                "corpora stay the suites', never "
                "copied"},
    {"id": "regression_gate_corpus", "benchmark": "regression_gate",
     "bind_module": "benchmarks.regression_gate",
     "bind_attribute": "NOISE_ROWS",
     "synthetic": True,
     "subject": "TF-129's filed noise rows over "
                "the 9 pinned baselines: every "
                "gated baseline re-measured 5 "
                "times in one process through the "
                "harness's own measurers, spreads "
                "filed per metric (all 0.0 on this "
                "tree), bound in place — the "
                "baselines stay the harness's, "
                "never copied"},
    {"id": "spot_checks_corpus", "benchmark": "spot_checks",
     "bind_module": "benchmarks.spot_checks",
     "bind_attribute": "TRANSCRIPT_SPECS",
     "synthetic": True,
     "subject": "TF-130's filed transcript specs: "
                "8 transcripts located in the real "
                "filed corpora of the persona / "
                "memory / depth / presence suites "
                "and TF-067's quant estate (56 "
                "turns; 4 holding, 4 not), bound in "
                "place — the corpora stay the "
                "suites', never copied; the "
                "transcripts are assembled live by "
                "build_spot_checks()"},
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
    {"id": "depth_suite", "kind": "suite",
     "status": "built_suite",
     "bind_module": "benchmarks.depth_suite",
     "bind_attribute": "run_depth_suite",
     "fixture": "depth_suite_corpus",
     "claims_axes": ("depth",),
     "suite_slice": "TF-124 (depth suite — itself)",
     "doc": "docs/architecture/DEPTH_SUITE.md",
     "docket": ("D-BENCH-10 FIXED by TF-124 (measuring sense)",),
     "notes": "TF-124's scored suite over TF-022's real "
              "selection table and TF-021 / TF-022's real "
              "prompt builders: multi-step reasoning with "
              "scaffolds on/off — 36 scripted synthetic "
              "problems, per-axis OFF / ON scores and the "
              "measured delta, PASS/REVIEW by filed "
              "thresholds + a positive delta, empty suite "
              "0.0 / REVIEW"},
    {"id": "presence_suite", "kind": "suite",
     "status": "built_suite",
     "bind_module": "benchmarks.presence_suite",
     "bind_attribute": "run_presence_suite",
     "fixture": "presence_suite_corpus",
     "claims_axes": ("presence", "will"),
     "suite_slice": "TF-125 (presence suite — itself)",
     "doc": "docs/architecture/PRESENCE_SUITE.md",
     "docket": ("D-BENCH-11 FIXED by TF-125 (measuring sense)",),
     "notes": "TF-125's scored suite over TF-028's real "
              "coherence scorer and TF-029's real "
              "GoalLedger: long-conversation coherence, "
              "will steering, will closure — 36 scripted "
              "synthetic sagas, per-axis scores + overall, "
              "PASS/REVIEW by filed thresholds, empty "
              "suite 0.0 / REVIEW. This entry carries the "
              "will claims axis TF-121 filed as scoreless"},
    {"id": "honesty_panel", "kind": "panel",
     "status": "built_panel",
     "bind_module": "benchmarks.honesty_panel",
     "bind_attribute": "run_honesty_panel",
     "fixture": "honesty_panel_corpus",
     "claims_axes": ("depth", "presence", "will"),
     "suite_slice": "TF-126 (small-model honesty panel — itself)",
     "doc": "docs/architecture/HONESTY_PANEL.md",
     "docket": ("D-BENCH-12 FIXED by TF-126 (measuring sense)",),
     "notes": "TF-126's panel over the four suites' real "
              "runners: their filed corpora degraded by "
              "four filed size tiers anchored live to "
              "catalogue models (reference 14B / mid 8B / "
              "small 3B / tiny 1.1B), scored OFF (degraded) "
              "against ON (degraded + the filed per-suite "
              "forge repairs) — 16 cells published as "
              "literal data, recomputed live; the forge "
              "helps most at the mid tier overall. A tier "
              "is a degradation profile, never a live "
              "model run: live per-size behaviour is NOT "
              "PROVEN (TF-173)"},
    {"id": "quant_panel", "kind": "panel",
     "status": "built_panel",
     "bind_module": "benchmarks.quant_panel",
     "bind_attribute": "run_quant_panel",
     "fixture": "quant_panel_corpus",
     "claims_axes": ("depth",),
     "suite_slice": "TF-127 (quant-vs-quality panel — itself)",
     "doc": "docs/architecture/QUANT_PANEL.md",
     "docket": ("D-BENCH-13 FIXED by TF-127 (measuring sense)",),
     "notes": "TF-127's panel over TF-067's real runner: "
              "the identical 8 synthetic prompts at the "
              "6 quant levels the profiles + catalogue "
              "span, scored by TF-067's own scorer over "
              "its fixture corpus — the overall curve "
              "(TF-067's CURVE in row form) plus the "
              "per-kind breakdown TF-067 never filed "
              "(fact / reasoning / instruction per "
              "level, 18 rows, 48 cells published as "
              "literal data, recomputed live). Reasoning "
              "plateaus at 0.5 from q2_k through q5_k_m; "
              "the non-monotone q4/q5 step is carried. "
              "No live model ran: live quant-vs-quality "
              "is NOT PROVEN (TF-173), and TF-068's live "
              "re-rank / calibration shares stand open"},
    {"id": "profile_panel", "kind": "panel",
     "status": "built_panel",
     "bind_module": "benchmarks.profile_panel",
     "bind_attribute": "run_profile_panel",
     "fixture": "profile_panel_corpus",
     "claims_axes": ("depth", "presence", "will"),
     "suite_slice": "TF-128 (profile panels — itself)",
     "doc": "docs/architecture/PROFILE_PANEL.md",
     "docket": ("D-BENCH-1 FIXED by TF-128 (measuring sense)",),
     "notes": "TF-128's panel over the six shipped "
              "profiles: per profile, the profile "
              "benchmark runs on a core built with "
              "the profile selected (its resolved "
              "tokens / passes + lifecycle caps "
              "asserted equal to the profile JSON's "
              "budgets during the run), TF-069's boot "
              "+ RAM-ceiling cap row re-run, and the "
              "four suites run through their real "
              "runners — 30 filed cells published as "
              "literal data, recomputed live. The "
              "depth ON road is refused by budget on "
              "pi_zero (0 refinement passes); no "
              "score varies by profile except through "
              "that refusal (empty store + scripted "
              "scorers, filed plainly). Live "
              "per-profile hardware behaviour is NOT "
              "PROVEN (TF-082 / TF-173)"},
    {"id": "regression_gate", "kind": "gate",
     "status": "built_gate",
     "bind_module": "benchmarks.regression_gate",
     "bind_attribute": "run_regression_gate",
     "fixture": "regression_gate_corpus",
     "claims_axes": ("depth", "presence", "will"),
     "suite_slice": "TF-129 (regression gates — itself)",
     "doc": "docs/architecture/REGRESSION_GATES.md",
     "docket": ("D-BENCH-14 FIXED by TF-129",),
     "notes": "TF-129's gate over the nine pinned "
              "baselines: their noise measured by "
              "re-measuring each 5 times in one "
              "process through this harness's own "
              "measurers (every spread 0.0 on this "
              "tree), tolerances derived from that "
              "noise over the filed 1e-9 floor "
              "(counts and the dict metric exact), "
              "and live re-measurement evaluated "
              "per metric with classifications "
              "(within tolerance / regression / "
              "improvement / count change / "
              "unmeasurable) — PASS only on the "
              "full selection within tolerance, "
              "any beyond-tolerance move FAILs "
              "(improvements included: a moved "
              "baseline is re-pinned deliberately, "
              "never passed silently), partial / "
              "empty selections REVIEW. Seven filed "
              "probes prove it can fail on "
              "synthetic perturbations. It gates "
              "fixture baselines only — no live "
              "model (TF-173), no CI wiring, no "
              "generation-time blocking"},
    {"id": "spot_checks", "kind": "panel",
     "status": "built_panel",
     "bind_module": "benchmarks.spot_checks",
     "bind_attribute": "run_spot_checks",
     "fixture": "spot_checks_corpus",
     "claims_axes": ("depth", "presence", "will"),
     "suite_slice": "TF-130 (human-shaped spot checks — itself)",
     "doc": "docs/architecture/SPOT_CHECKS.md",
     "docket": ("D-BENCH-15 FIXED by TF-130 (measuring sense)",),
     "notes": "TF-130's human-shaped spot checks: "
              "8 transcripts assembled live from "
              "the real filed corpora of the "
              "persona / memory / depth / presence "
              "suites and TF-067's quant estate — "
              "every turn verbatim suite case data, "
              "every machine score recomputed by "
              "that suite's real per-case scorer, "
              "each transcript carrying its score "
              "+ verdict beside a human-judgement "
              "prompt. Four hold and four do not, "
              "as filed. The rendered set IS the "
              "doc (SPOT_CHECKS.md). No human "
              "judgement is recorded yet — the set "
              "is the instrument; live-model "
              "transcripts are NOT PROVEN (TF-173)"},
    {"id": "profile_benchmark", "kind": "runner",
     "status": "existing_runner",
     "bind_module": "benchmarks.benchmark_profiles",
     "bind_attribute": "ProfileBenchmark",
     "fixture": "profile_default_queries",
     "claims_axes": ("depth",),
     "suite_slice": "TF-128 DONE (profile panels apply the "
                    "profile: benchmarks/profile_panel.py)",
     "doc": "docs/audit/TF-010_BENCHMARK_ESTATE.md",
     "docket": ("D-BENCH-1 FIXED by TF-128 (measuring sense; "
                "validation share FIXED by TF-121)",
                "D-BENCH-2 FIXED by TF-121",
                "D-BENCH-6 FIXED by TF-121"),
     "notes": "runs the core over an ephemeral empty store "
              "(model_path=None): measures retrieval + scaffold "
              "overhead and the empty-store refusal, never inference. "
              "Latency filed as data per run, never pinned. "
              "TF-128 update, in the open: the profile "
              "panels now run this benchmark per profile "
              "on a profile-applied core (supplied "
              "through this runner's core= parameter, "
              "budgets enforced during the run, "
              "D-BENCH-1 fixed in the measuring sense); "
              "this module's own default ephemeral core "
              "still applies no profile, stated openly"},
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
     "suite_slice": "TF-127 DONE (quant-vs-quality panel "
                    "consumes it mechanically: "
                    "benchmarks/quant_panel.py)",
     "doc": "docs/architecture/QUANT_QUALITY_FIXTURES.md",
     "docket": (),
     "notes": "TF-067's quant quality panel: fixture generation + "
              "per-prompt scoring, filed as data. TF-127 update, "
              "in the open: the Part VII panel now consumes this "
              "runner mechanically (benchmarks/quant_panel.py, "
              "D-BENCH-13) with the per-kind breakdown and a "
              "pinned harness baseline — over the fixture corpus "
              "only; this doc's live-model panel share stands "
              "open, live quant-vs-quality NOT PROVEN (TF-173)"},
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
    {"id": "chain_off_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted chain "
                 "problems whose direct (scaffold OFF) answer "
                 "equals the known answer (TF-124)"},
    {"id": "chain_on_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted chain "
                 "problems whose scaffolded (ON) run holds: "
                 "the reasoning carries every filed "
                 "intermediate and the final answer is exact "
                 "(TF-124)"},
    {"id": "plan_off_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted plan "
                 "problems whose direct sequence equals the "
                 "known order (TF-124)"},
    {"id": "plan_on_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted plan "
                 "problems whose scaffolded run holds: the "
                 "plan carries the filed steps in order and "
                 "the executed answer is exact (TF-124)"},
    {"id": "check_off_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted check "
                 "problems whose draft (scaffold OFF) equals "
                 "the known answer (TF-124)"},
    {"id": "check_on_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted check "
                 "problems whose scaffolded run holds: the "
                 "check names the filed error token (or its "
                 "clean verdict) and the revised answer is "
                 "exact (TF-124)"},
    {"id": "overall_off_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "the mean of the suite's three axis OFF "
                 "scores; 0.0 for an empty suite (TF-124)"},
    {"id": "overall_on_score", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "the mean of the suite's three axis ON "
                 "scores; 0.0 for an empty suite (TF-124)"},
    {"id": "delta", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "overall_on_score minus overall_off_score "
                 "— the measured scaffold value over the "
                 "filed corpus, never an asserted one "
                 "(TF-124)"},
    {"id": "total_cases", "benchmark": "depth_suite",
     "legacy_name": "",
     "computes": "problems scored across the suite's three "
                 "axes (12 per axis, 36 in the filed corpus, "
                 "each on both roads)"},
    {"id": "coherence_score", "benchmark": "presence_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted coherence "
                 "sagas TF-028's score_conversation scores at "
                 "a whole-saga mean at or above 0.80 "
                 "(TF-028's PASS_LATE_MIN) (TF-125)"},
    {"id": "steering_score", "benchmark": "presence_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted steering "
                 "sagas that follow through under TF-029's "
                 "GoalLedger replayed over the filed user "
                 "turns: every steered turn's answer carries "
                 "the standing statement, no closed statement "
                 "steers, the end state matches (TF-125)"},
    {"id": "closure_score", "benchmark": "presence_suite",
     "legacy_name": "",
     "computes": "fraction of the suite's scripted closure "
                 "sagas where completion / abandonment / "
                 "supersession / staleness ends the steering "
                 "exactly as filed under TF-029's ledger "
                 "replay — a still-steered closed goal fails "
                 "its saga (TF-125)"},
    {"id": "overall_score", "benchmark": "presence_suite",
     "legacy_name": "",
     "computes": "the mean of the suite's three axis scores "
                 "(coherence, will steering, will closure); "
                 "0.0 for an empty suite (TF-125)"},
    {"id": "total_cases", "benchmark": "presence_suite",
     "legacy_name": "",
     "computes": "sagas scored across the suite's three "
                 "axes (12 per axis, 36 in the filed corpus)"},
    {"id": "reference_off_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the reference (14B) tier's mean OFF score "
                 "across the four suites: their filed corpora "
                 "undegraded, scored by their real runners "
                 "(TF-126)"},
    {"id": "reference_on_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the reference tier's mean ON score across "
                 "the four suites (undegraded + the filed "
                 "per-suite forge repairs) (TF-126)"},
    {"id": "reference_delta", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "reference_on_score minus "
                 "reference_off_score — the measured forge "
                 "gain at the reference tier (TF-126)"},
    {"id": "mid_off_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the mid (8B) tier's mean OFF score across "
                 "the four suites: their filed corpora "
                 "degraded by citation_drop, scored by their "
                 "real runners (TF-126)"},
    {"id": "mid_on_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the mid tier's mean ON score across the "
                 "four suites (degraded + the filed "
                 "per-suite forge repairs) (TF-126)"},
    {"id": "mid_delta", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "mid_on_score minus mid_off_score — the "
                 "measured forge gain at the mid tier; the "
                 "panel's largest tier mean delta (TF-126)"},
    {"id": "small_off_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the small (3B) tier's mean OFF score "
                 "across the four suites: their filed corpora "
                 "degraded by citation_drop + "
                 "terse_first_sentence (and the coherence "
                 "drift shift), scored by their real "
                 "runners (TF-126)"},
    {"id": "small_on_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the small tier's mean ON score across "
                 "the four suites (degraded + the filed "
                 "per-suite forge repairs) (TF-126)"},
    {"id": "small_delta", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "small_on_score minus small_off_score — "
                 "the measured forge gain at the small "
                 "tier, carrying depth's negative cell "
                 "delta by name (TF-126)"},
    {"id": "tiny_off_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the tiny (1.1B) tier's mean OFF score "
                 "across the four suites: their filed "
                 "corpora degraded by all four filed "
                 "transforms (and the coherence drift "
                 "shift), scored by their real runners "
                 "(TF-126)"},
    {"id": "tiny_on_score", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "the tiny tier's mean ON score across the "
                 "four suites (degraded + the filed "
                 "per-suite forge repairs) (TF-126)"},
    {"id": "tiny_delta", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "tiny_on_score minus tiny_off_score — "
                 "the measured forge gain at the tiny "
                 "tier, the panel's smallest tier mean "
                 "delta (TF-126)"},
    {"id": "total_cells", "benchmark": "honesty_panel",
     "legacy_name": "",
     "computes": "panel cells measured (4 filed tiers x "
                 "4 suites, 16 in the filed panel) "
                 "(TF-126)"},
    {"id": "iq2_xs_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at iq2_xs, through "
                 "TF-067's run_quant_panel over its "
                 "fixture corpus (TF-127)"},
    {"id": "iq2_xs_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus the "
                 "iq2_xs mean — loss against a measured "
                 "reference row, never an assumed 1.0 "
                 "(TF-127)"},
    {"id": "q2_k_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at q2_k (TF-127)"},
    {"id": "q2_k_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus the "
                 "q2_k mean (TF-127)"},
    {"id": "q4_k_m_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at q4_k_m — above "
                 "q5_k_m on this corpus: the filed "
                 "non-monotone step, carried (TF-127)"},
    {"id": "q4_k_m_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus the "
                 "q4_k_m mean (TF-127)"},
    {"id": "q5_k_m_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at q5_k_m (TF-127)"},
    {"id": "q5_k_m_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus the "
                 "q5_k_m mean (TF-127)"},
    {"id": "q8_0_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at q8_0 (TF-127)"},
    {"id": "q8_0_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus the "
                 "q8_0 mean (TF-127)"},
    {"id": "fp16_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "TF-067's scorer's mean score over the "
                 "8 filed prompts at fp16 — the panel's "
                 "measured reference row (TF-127)"},
    {"id": "fp16_loss", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the panel's own fp16 mean minus "
                 "itself: 0.0 by construction of the "
                 "loss definition (TF-127)"},
    {"id": "fact_mean_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the mean of the per-level fact-kind "
                 "means (5 filed fact prompts per level, "
                 "6 levels) from the panel's kind rows "
                 "(TF-127)"},
    {"id": "reasoning_mean_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the mean of the per-level "
                 "reasoning-kind means (2 filed "
                 "reasoning prompts per level, 6 "
                 "levels) — the kind that plateaus at "
                 "0.5 across the mid levels (TF-127)"},
    {"id": "instruction_mean_score", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "the mean of the per-level "
                 "instruction-kind means (1 filed "
                 "instruction prompt per level, 6 "
                 "levels) — a single-prompt mean, filed "
                 "as exactly that (TF-127)"},
    {"id": "total_cells", "benchmark": "quant_panel",
     "legacy_name": "",
     "computes": "panel cells measured (6 filed levels "
                 "x 8 filed prompts, 48 in the filed "
                 "panel) (TF-127)"},
    {"id": "desktop_cpu_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of desktop_cpu's four suite "
                 "headline scores (persona / memory / "
                 "depth ON / presence) run per profile "
                 "through their real runners (TF-128)"},
    {"id": "desktop_cpu_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "desktop_cpu's profile-benchmark mean "
                 "response words over its filed queries "
                 "on a profile-applied ephemeral core "
                 "(empty store) (TF-128)"},
    {"id": "desktop_gpu_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of desktop_gpu's four suite "
                 "headline scores run per profile "
                 "through their real runners (TF-128)"},
    {"id": "desktop_gpu_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "desktop_gpu's profile-benchmark mean "
                 "response words on a profile-applied "
                 "ephemeral core (empty store) (TF-128)"},
    {"id": "phone_low_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of phone_low's four suite "
                 "headline scores run per profile "
                 "through their real runners (TF-128)"},
    {"id": "phone_low_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "phone_low's profile-benchmark mean "
                 "response words on a profile-applied "
                 "ephemeral core (empty store) (TF-128)"},
    {"id": "pi_5_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of pi_5's four suite headline "
                 "scores run per profile through their "
                 "real runners (TF-128)"},
    {"id": "pi_5_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "pi_5's profile-benchmark mean response "
                 "words on a profile-applied ephemeral "
                 "core (empty store) (TF-128)"},
    {"id": "pi_zero_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of pi_zero's four suite "
                 "headline scores — its depth headline "
                 "is the OFF score because its 0-pass "
                 "refinement budget refuses the ON "
                 "road (TF-128)"},
    {"id": "pi_zero_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "pi_zero's profile-benchmark mean "
                 "response words on a profile-applied "
                 "ephemeral core (empty store) (TF-128)"},
    {"id": "server_gpu_suite_score", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "the mean of server_gpu's four suite "
                 "headline scores run per profile "
                 "through their real runners (TF-128)"},
    {"id": "server_gpu_bench_avg_words", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "server_gpu's profile-benchmark mean "
                 "response words on a profile-applied "
                 "ephemeral core (empty store) (TF-128)"},
    {"id": "total_cells", "benchmark": "profile_panel",
     "legacy_name": "",
     "computes": "panel cells filed (6 shipped profiles "
                 "x the benchmark + 4 suites, 30 in the "
                 "filed panel; 1 of them refused by "
                 "budget) (TF-128)"},
    {"id": "baselines_gated", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "baselines the regression gate "
                 "re-measures and evaluates on a full "
                 "run: the nine TF-121..TF-128 pinned "
                 "baselines (the gate's own baseline "
                 "is not gated) (TF-129)"},
    {"id": "metrics_compared", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "per-metric delta rows the gate "
                 "evaluates on a full run: every "
                 "pinned metric of every gated "
                 "baseline, counted live from the "
                 "pins (83 on this tree) (TF-129)"},
    {"id": "max_noise_spread", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "the widest per-metric spread in "
                 "the gate's filed NOISE_ROWS: each "
                 "gated baseline re-measured 5 times "
                 "in one process through the harness "
                 "measurers, max-min per metric (0.0 "
                 "on this tree — TF-036's determinism, "
                 "measured) (TF-129)"},
    {"id": "gate_failures", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "delta rows beyond their derived "
                 "tolerance (or unmeasurable) on a "
                 "live full gate run — regressions, "
                 "improvements and count changes all "
                 "count; 0 on this tree, so the filed "
                 "run PASSes (TF-129)"},
    {"id": "probes_passed", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "filed GATE_PROBES whose synthetic "
                 "perturbation of the pins evaluates "
                 "with its filed outcome + "
                 "classification against the gate's "
                 "pure evaluator (TF-129)"},
    {"id": "probes_total", "benchmark": "regression_gate",
     "legacy_name": "",
     "computes": "filed GATE_PROBES in total: a "
                 "dropped score, a raised score, a "
                 "changed count, a dropped loss, a "
                 "changed dict metric, an "
                 "inside-tolerance move and a missing "
                 "baseline (7 on this tree) (TF-129)"},
    {"id": "total_transcripts", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "transcripts in TF-130's filed "
                 "spot-check set, each one a real "
                 "case from a real suite's filed "
                 "corpus (8 on this tree) (TF-130)"},
    {"id": "total_turns", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "verbatim turns across the filed "
                 "spot-check transcripts, counted "
                 "live from the assembled set (56 on "
                 "this tree: the 20-turn coherence "
                 "saga files 40 of them) (TF-130)"},
    {"id": "suites_covered", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "distinct suite anchors the set "
                 "draws from: persona_suite, "
                 "memory_suite, depth_suite, "
                 "presence_suite and TF-067's "
                 "quant_quality estate (5 on this "
                 "tree) (TF-130)"},
    {"id": "holding_transcripts", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "transcripts whose recomputed "
                 "score holds under their own "
                 "suite's real scorer, as filed in "
                 "their expectations (4 on this "
                 "tree) (TF-130)"},
    {"id": "failing_transcripts", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "transcripts whose recomputed "
                 "score does not hold — the failing "
                 "shapes filed by name beside the "
                 "passes (4 on this tree) (TF-130)"},
    {"id": "mean_score", "benchmark": "spot_checks",
     "legacy_name": "",
     "computes": "the descriptive mean of the "
                 "set's per-transcript scores across "
                 "five different scorers (0.625 on "
                 "this tree) — a summary only; the "
                 "anchor is the per-transcript "
                 "reading, never the mean (TF-130)"},
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
              "NOT PROVEN (TF-125 / TF-126). TF-126 update, in "
              "the open: the honesty panel now runs this suite "
              "across filed size tiers in the measuring sense "
              "(benchmarks/honesty_panel.py, D-BENCH-12) — a "
              "live model's behaviour at those sizes is still "
              "NOT PROVEN (TF-173)"},
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
              "behaviour is NOT PROVEN (TF-125 / TF-126). "
              "TF-126 update, in the open: the honesty panel "
              "now runs this suite across filed size tiers in "
              "the measuring sense (benchmarks/honesty_panel.py, "
              "D-BENCH-12) — a live model's behaviour at those "
              "sizes is still NOT PROVEN (TF-173)"},
    {"id": "depth_suite_synthetic_v1",
     "benchmark": "depth_suite",
     "fixture": "depth_suite_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-124's filed scripted "
                    "problems through TF-022's real selection "
                    "table and TF-021 / TF-022's real prompt "
                    "builders, measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "chain_off_score": 0.4166666666666667,
         "chain_on_score": 0.8333333333333334,
         "plan_off_score": 0.5,
         "plan_on_score": 0.75,
         "check_off_score": 0.5,
         "check_on_score": 0.8333333333333334,
         "overall_off_score": 0.47222222222222227,
         "overall_on_score": 0.8055555555555557,
         "delta": 0.3333333333333334,
         "total_cases": 36,
     },
     "targets": {
         "overall_on_score": {"target": 0.75, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the suite's scripted problems hold OFF 5/12 "
              "chain, 6/12 plan, 6/12 check and ON 10/12 "
              "chain, 9/12 plan, 10/12 check — the measured "
              "delta is +0.3333, and the failures are filed, "
              "not hidden: a reasoning slip and a plan "
              "locking a wrong order and a check inventing a "
              "problem are scaffold-LOSS cases (OFF holds, "
              "ON fails), and transcription / missed-error / "
              "missing-step cases fail on both roads; the "
              "plan axis's ON score is the lowest, exactly "
              "at the 0.75 threshold. This pins SCRIPTED "
              "fixtures through the real scaffold machinery "
              "— a live model's depth behaviour is NOT "
              "PROVEN (TF-126), and the core loop does not "
              "execute strategies automatically (TF-022's "
              "filed limit; D-BENCH-10). TF-126 update, in "
              "the open: the honesty panel now runs this "
              "suite across filed size tiers in the "
              "measuring sense (benchmarks/honesty_panel.py, "
              "D-BENCH-12) — a live model's behaviour at "
              "those sizes is still NOT PROVEN (TF-173)"},
    {"id": "presence_suite_synthetic_v1",
     "benchmark": "presence_suite",
     "fixture": "presence_suite_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-125's filed scripted "
                    "sagas through TF-028's real coherence "
                    "scorer and TF-029's real GoalLedger, "
                    "measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "coherence_score": 0.75,
         "steering_score": 0.8333333333333334,
         "closure_score": 0.75,
         "overall_score": 0.7777777777777778,
         "total_cases": 36,
     },
     "targets": {
         "overall_score": {"target": 0.75, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the suite's scripted sagas hold 9/12 coherence "
              "sagas (the onset-15 and onset-16 breaks and a "
              "truncated drifting saga fail, by design), 10/12 "
              "steering sagas (a dropped goal and a "
              "contradicted goal fail, by design) and 9/12 "
              "closure sagas (abandoned / completed / "
              "superseded goals still steered fail, by "
              "design); the coherence and closure axes sit "
              "exactly at the 0.75 threshold, so the filed run "
              "PASSes. This pins SCRIPTED fixtures through "
              "the real machinery — a live model's presence "
              "over real long conversations is NOT PROVEN "
              "(TF-126), and the core loop's live answers "
              "are not scored here (D-BENCH-11). TF-126 "
              "update, in the open: the honesty panel now "
              "runs this suite across filed size tiers in "
              "the measuring sense (benchmarks/honesty_panel.py, "
              "D-BENCH-12) — a live model's behaviour at "
              "those sizes is still NOT PROVEN (TF-173)"},
    {"id": "honesty_panel_synthetic_v1",
     "benchmark": "honesty_panel",
     "fixture": "honesty_panel_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-126's filed "
                    "size tiers degrading the four suites' "
                    "filed corpora through their real "
                    "runners, measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "reference_off_score": 0.7252314814814815,
         "reference_on_score": 0.8402777777777779,
         "reference_delta": 0.1150462962962963,
         "mid_off_score": 0.5932870370370371,
         "mid_on_score": 0.8333333333333335,
         "mid_delta": 0.24004629629629637,
         "small_off_score": 0.5840277777777778,
         "small_on_score": 0.712962962962963,
         "small_delta": 0.12893518518518515,
         "tiny_off_score": 0.40847222222222224,
         "tiny_on_score": 0.4675925925925926,
         "tiny_delta": 0.05912037037037035,
         "total_cells": 16,
     },
     "targets": {
         "reference_off_score": {"target": 0.7252314814814815,
                                 "met": True},
     },
     "excluded_metrics": {},
     "notes": "the panel's per-tier means over the four "
              "suites (16 cells, filed as literal rows in "
              "benchmarks/honesty_panel.py and recomputed "
              "live by its checker): the forge helps most "
              "at the mid tier overall (mean delta "
              "+0.2400), then small (+0.1289), reference "
              "(+0.1150), tiny (+0.0591) — per suite, "
              "persona is helped most at tiny, memory at "
              "small, depth and presence at mid, and "
              "depth's small-tier cell delta is negative "
              "(-0.1944), filed by name. The filed target "
              "is the reference-reproduction truth: the "
              "reference tier's OFF cells reproduce the "
              "four suites' pinned baselines exactly, so "
              "its mean OFF is pinned at the value those "
              "baselines imply. This pins FILED DEGRADATION "
              "PROFILES over scripted fixtures through the "
              "real machinery — live models' per-size "
              "behaviour is NOT PROVEN (no GGUF / "
              "llama_cpp on this VM, TF-078; TF-173 owns "
              "the real-hardware road)"},
    {"id": "quant_panel_synthetic_v1",
     "benchmark": "quant_panel",
     "fixture": "quant_panel_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-067's "
                    "filed fixture corpus through its "
                    "real run_quant_panel + scorer, "
                    "consumed mechanically by "
                    "TF-127's panel, measured live "
                    "twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "iq2_xs_score": 0.0,
         "iq2_xs_loss": 1.0,
         "q2_k_score": 0.4375,
         "q2_k_loss": 0.5625,
         "q4_k_m_score": 0.8125,
         "q4_k_m_loss": 0.1875,
         "q5_k_m_score": 0.7083333333333334,
         "q5_k_m_loss": 0.29166666666666663,
         "q8_0_score": 1.0,
         "q8_0_loss": 0.0,
         "fp16_score": 1.0,
         "fp16_loss": 0.0,
         "fact_mean_score": 0.7000000000000001,
         "reasoning_mean_score": 0.5833333333333334,
         "instruction_mean_score": 0.611111111111111,
         "total_cells": 48,
     },
     "targets": {
         "fp16_score": {"target": 1.0, "met": True},
     },
     "excluded_metrics": {},
     "notes": "TF-067's published curve pinned as a "
              "harness baseline through TF-127's panel "
              "(48 cells, filed as literal rows in "
              "benchmarks/quant_panel.py and recomputed "
              "live by its checker): steep loss at the "
              "bottom (iq2_xs 0.0, q2_k 0.4375), parity "
              "at the top (q8_0 and fp16 at 1.0), and "
              "the non-monotone q4_k_m 0.8125 > q5_k_m "
              "0.7083 step carried exactly as TF-067 "
              "filed it — 8 prompts cannot rank "
              "neighbouring levels. Per kind, reasoning "
              "plateaus at 0.5 from q2_k through q5_k_m "
              "and instruction collapses to 0.0 at q2_k "
              "on its single prompt. The filed target "
              "is the curve-reproduction truth: the "
              "fp16 reference row is pinned at its "
              "measured 1.0. This pins TF-067's "
              "SYNTHETIC FIXTURE corpus through its real "
              "scorer — live models' quant-vs-quality "
              "is NOT PROVEN (no GGUF / llama_cpp on "
              "this VM, TF-078; TF-173 owns the "
              "real-hardware road), and TF-068's live "
              "re-rank / q2_k calibration shares stand "
              "open on that road"},
    {"id": "profile_panel_synthetic_v1",
     "benchmark": "profile_panel",
     "fixture": "profile_panel_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-128's "
                    "filed panel: per shipped profile "
                    "the benchmark on a profile-applied "
                    "core + TF-069's boot and the four "
                    "suites' real runners, measured "
                    "live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "desktop_cpu_suite_score": 0.8085648148148149,
         "desktop_cpu_bench_avg_words": 9,
         "desktop_gpu_suite_score": 0.8085648148148149,
         "desktop_gpu_bench_avg_words": 9,
         "phone_low_suite_score": 0.8085648148148149,
         "phone_low_bench_avg_words": 9,
         "pi_5_suite_score": 0.8085648148148149,
         "pi_5_bench_avg_words": 9,
         "pi_zero_suite_score": 0.7252314814814815,
         "pi_zero_bench_avg_words": 9,
         "server_gpu_suite_score": 0.8085648148148149,
         "server_gpu_bench_avg_words": 9,
         "total_cells": 30,
     },
     "targets": {
         "desktop_cpu_suite_score": {"target": 0.8085648148148149,
                                     "met": True},
     },
     "excluded_metrics": {},
     "notes": "the panel's per-profile suite means + "
              "benchmark words (30 filed cells in "
              "benchmarks/profile_panel.py, recomputed "
              "live by its checker): the budgets apply "
              "exactly on all six profiles (tokens "
              "120..2000, passes 0..3, bundle 3..30), "
              "TF-069's cap rows stand in the rows "
              "(server_gpu exceeds its ceiling; "
              "phone_low / pi_zero unprovable), and no "
              "score varies by profile except pi_zero's "
              "suite mean — its depth ON road is "
              "refused by its 0-pass refinement budget, "
              "so its depth headline is the OFF score. "
              "The filed target is the reproduction "
              "truth: the per-profile suite runs "
              "reproduce the four suites' pinned "
              "baselines, so desktop_cpu's mean is "
              "pinned at the value those baselines "
              "imply. This pins SCRIPTED fixtures + an "
              "EMPTY-store core through the real "
              "machinery with profiles applied — live "
              "per-profile hardware behaviour is NOT "
              "PROVEN (TF-082 / TF-173)"},
    {"id": "regression_gate_synthetic_v1",
     "benchmark": "regression_gate",
     "fixture": "regression_gate_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-129's "
                    "gate over the nine pinned "
                    "baselines, re-measured live "
                    "through the harness measurers, "
                    "measured live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "baselines_gated": 9,
         "metrics_compared": 83,
         "max_noise_spread": 0.0,
         "gate_failures": 0,
         "probes_passed": 7,
         "probes_total": 7,
     },
     "targets": {
         "gate_failures": {"target": 0, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the gate's verdict row over the nine "
              "pinned baselines (83 pinned metrics, "
              "filed as literal NOISE_ROWS + derived "
              "TOLERANCES in "
              "benchmarks/regression_gate.py and "
              "recomputed live by its checker): the "
              "noise measurement re-ran every gated "
              "baseline 5 times in one process and "
              "every spread measured 0.0, so every "
              "float tolerance files the 1e-9 floor "
              "and counts / the dict metric file "
              "exact; the live full gate run PASSes "
              "with 0 failures, and all 7 filed "
              "probes behave against the pure "
              "evaluator (a dropped score fails as a "
              "regression, a raised score and a "
              "dropped loss as improvements, a "
              "changed count and a changed dict as "
              "count changes, a 5e-10 move passes, a "
              "missing baseline fails unmeasurable). "
              "The filed target is the gate truth: "
              "zero failures on the filed tree. This "
              "pins the gate over SYNTHETIC fixture "
              "baselines — no live model is gated, "
              "live behaviour NOT PROVEN (TF-173)"},
    {"id": "spot_checks_synthetic_v1",
     "benchmark": "spot_checks",
     "fixture": "spot_checks_corpus",
     "measured_on": "this VM, 2026-11-01 — TF-130's "
                    "filed spot-check set assembled "
                    "live from the suites' own filed "
                    "corpora through their real "
                    "per-case scorers, measured "
                    "live twice, identical",
     "tolerance": 1e-9,
     "metrics": {
         "total_transcripts": 8,
         "total_turns": 56,
         "suites_covered": 5,
         "holding_transcripts": 4,
         "failing_transcripts": 4,
         "mean_score": 0.625,
     },
     "targets": {
         "holding_transcripts": {"target": 4, "met": True},
     },
     "excluded_metrics": {},
     "notes": "the spot-check set's reproduction "
              "row (8 filed transcripts / 56 "
              "verbatim turns over 5 suite anchors, "
              "recomputed live by "
              "benchmarks/spot_checks.py's checker): "
              "persona identity 1.0 holds, persona "
              "voice generic slip 0.3 does not, "
              "memory citation present+true 1.0 "
              "holds, memory citation present but "
              "false 0.0 does not, depth chain "
              "scaffolded 1.0 holds (direct 0.0), "
              "presence coherence break mean 0.7 "
              "does not, presence will steering 1.0 "
              "holds, quant fare_maths at q4_k_m "
              "0.0 does not — 4 holding / 4 not, "
              "mean 0.625 filed as a descriptive "
              "summary across five scorers only. "
              "The filed target is the set's "
              "balance truth: half the anchor must "
              "show failures or it anchors nothing. "
              "This pins SCRIPTED fixture "
              "transcripts — no live model was "
              "read, live behaviour NOT PROVEN "
              "(TF-173), and no human judgement is "
              "recorded yet"},
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
     "rule": "D-BENCH-1 FIXED (validation by TF-121, "
             "application by TF-128 in the measuring "
             "sense): a benchmark run names its hardware "
             "profile from KNOWN_PROFILES (the shipped "
             "hardware_profiles/*.json ids) and an unknown "
             "id is refused by name before anything runs; "
             "TF-128's profile panels "
             "(benchmarks/profile_panel.py) run the suites "
             "+ this benchmark per profile on a core with "
             "the profile applied, its budgets enforced "
             "during the run — the module's own default "
             "ephemeral core still applies no profile, "
             "stated openly in TF-128's filing"},
    {"id": "the_locust_fate_is_exclusion",
     "rule": "D-BENCH-5 narrowed: the locust load script is "
             "excluded from the harness panel by verdict — a "
             "manual throughput script, unrunnable without its "
             "dev extra, measuring no claim axis"},
    {"id": "the_will_axis_is_scored_by_tf125",
     "rule": "TF-125 DONE: the estate covers depth, presence "
             "and will — TF-125's presence suite "
             "(benchmarks/presence_suite.py) scores will "
             "steering and will closure through TF-029's real "
             "GoalLedger over scripted sagas, with its own "
             "pinned baseline (D-BENCH-11, fixed in the "
             "measuring sense). TF-121 filed this slot as "
             "the_will_axis_has_no_scorer_yet; the gap is "
             "closed in the measuring sense, and a live "
             "model's follow-through stays NOT PROVEN "
             "(TF-126)"},
    {"id": "targets_are_checked_even_when_unmet",
     "rule": "every baseline files its targets with met stated "
             "honestly (all of the profile baseline's targets and "
             "the persona pass threshold are unmet at banking) — "
             "an unmet target is a measurement, not a failure of "
             "the harness"},
    {"id": "regression_gates_fail_on_moved_baselines",
     "rule": "TF-129 DONE: the pinned baselines have a "
             "gate (benchmarks/regression_gate.py, "
             "D-BENCH-14) — tolerances derived from "
             "measured baseline noise (5 in-process "
             "repeats per baseline; max(measured "
             "spread, the 1e-9 floor) per float "
             "metric, counts and the dict metric "
             "exact), deltas classified "
             "(regression / improvement / count "
             "change / unmeasurable), and the verdict "
             "PASS only on the full nine-baseline "
             "selection within tolerance: any "
             "beyond-tolerance move fails a slice "
             "gate, improvements included — a moved "
             "baseline is re-pinned deliberately by "
             "a slice, never passed silently. The "
             "gate is a checker a slice runs "
             "in-house, not a CI wiring, and it "
             "gates fixture baselines only (no live "
             "model, TF-173)"},
    {"id": "spot_checks_anchor_the_numbers_in_human_judgement",
     "rule": "TF-130 DONE: the estate's numbers "
             "have a human-shaped anchor "
             "(benchmarks/spot_checks.py, "
             "D-BENCH-15) — 8 transcripts assembled "
             "live from the suites' own filed "
             "corpora (no corpus or scorer copied), "
             "every turn verbatim, every machine "
             "score recomputed by its suite's real "
             "per-case scorer, each transcript "
             "carrying its score + verdict beside "
             "a human-judgement prompt, 4 holding "
             "and 4 not as filed; the rendered set "
             "IS the doc "
             "(docs/architecture/SPOT_CHECKS.md), "
             "checked verbatim by "
             "spot_checks_problems(). The "
             "spot-check baseline is judged by that "
             "checker, not gated by TF-129's gate "
             "(the gated set stays the nine "
             "TF-121..TF-128 baselines). No human "
             "judgement is recorded yet and "
             "live-model transcripts are NOT "
             "PROVEN (TF-173)"},
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


def measure_depth_suite_baseline() -> dict[str, Any]:
    """Run TF-124's filed suite corpora through the real machinery."""
    from benchmarks.depth_suite import run_depth_suite

    result = run_depth_suite()
    return {
        "chain_off_score": result.chain_off,
        "chain_on_score": result.chain_on,
        "plan_off_score": result.plan_off,
        "plan_on_score": result.plan_on,
        "check_off_score": result.check_off,
        "check_on_score": result.check_on,
        "overall_off_score": result.overall_off,
        "overall_on_score": result.overall_on,
        "delta": result.delta,
        "total_cases": result.total_cases,
    }


def measure_presence_suite_baseline() -> dict[str, Any]:
    """Run TF-125's filed suite corpora through the real machinery."""
    from benchmarks.presence_suite import run_presence_suite

    result = run_presence_suite()
    return {
        "coherence_score": result.coherence_score,
        "steering_score": result.steering_score,
        "closure_score": result.closure_score,
        "overall_score": result.overall_score,
        "total_cases": result.total_cases,
    }


def measure_honesty_panel_baseline() -> dict[str, Any]:
    """Run TF-126's filed panel through the real machinery."""
    from benchmarks.honesty_panel import MODEL_TIERS, run_honesty_panel

    result = run_honesty_panel()
    measured: dict[str, Any] = {"total_cells": result.total_cells}
    means = {m["tier"]: m for m in result.tier_means}
    for tier in MODEL_TIERS:
        prefix = tier["prefix"]
        mean = means[tier["id"]]
        measured[f"{prefix}_off_score"] = mean["mean_off"]
        measured[f"{prefix}_on_score"] = mean["mean_on"]
        measured[f"{prefix}_delta"] = mean["mean_delta"]
    return measured


def measure_quant_panel_baseline() -> dict[str, Any]:
    """Run TF-127's filed panel through TF-067's machinery."""
    from benchmarks.quant_panel import PANEL_KINDS, run_quant_panel

    result = run_quant_panel()
    measured: dict[str, Any] = {"total_cells": result.total_cells}
    for row in result.rows:
        measured[f"{row['quant']}_score"] = row["mean_score"]
        measured[f"{row['quant']}_loss"] = row["loss_vs_reference"]
    for kind in PANEL_KINDS:
        cells = [r for r in result.kind_rows if r["kind"] == kind]
        measured[f"{kind}_mean_score"] = (
            sum(r["mean_score"] for r in cells) / len(cells))
    return measured


def measure_profile_panel_baseline() -> dict[str, Any]:
    """Run TF-128's filed panel through the real machinery."""
    from benchmarks.profile_panel import run_profile_panel

    result = run_profile_panel()
    measured: dict[str, Any] = {"total_cells": result.total_cells}
    for row in result.rows:
        measured[f"{row['profile']}_suite_score"] = row["suite_score"]
        measured[f"{row['profile']}_bench_avg_words"] = \
            row["bench_avg_response_words"]
    return measured


def measure_regression_gate_baseline() -> dict[str, Any]:
    """Run TF-129's gate over the nine pinned baselines.

    The noise spread files the gate's published
    NOISE_ROWS summary (recomputed live by the
    gate's own checker); the gate verdict and the
    probes are measured live here.
    """
    from benchmarks.regression_gate import (
        NOISE_ROWS,
        run_gate_probes,
        run_regression_gate,
    )

    result = run_regression_gate()
    probes = run_gate_probes()
    return {
        "baselines_gated": result.baselines_gated,
        "metrics_compared": result.metrics_compared,
        "max_noise_spread": max(
            row["max_abs_spread"] for row in NOISE_ROWS),
        "gate_failures": result.failures,
        "probes_passed": sum(1 for p in probes if p["passed"]),
        "probes_total": len(probes),
    }


def measure_spot_checks_baseline() -> dict[str, Any]:
    """Run TF-130's filed spot-check set through the real machinery."""
    from benchmarks.spot_checks import run_spot_checks

    result = run_spot_checks()
    return {
        "total_transcripts": result.total_transcripts,
        "total_turns": result.total_turns,
        "suites_covered": result.suites_covered,
        "holding_transcripts": result.holding,
        "failing_transcripts": result.failing,
        "mean_score": result.mean_score,
    }


_MEASURERS = {
    "persona_consistency_synthetic_v1": measure_persona_baseline,
    "profile_benchmark_empty_store_v1": measure_profile_baseline,
    "profile_panel_synthetic_v1": measure_profile_panel_baseline,
    "regression_gate_synthetic_v1": measure_regression_gate_baseline,
    "spot_checks_synthetic_v1": measure_spot_checks_baseline,
    "persona_suite_synthetic_v1": measure_persona_suite_baseline,
    "memory_suite_synthetic_v1": measure_memory_suite_baseline,
    "depth_suite_synthetic_v1": measure_depth_suite_baseline,
    "presence_suite_synthetic_v1": measure_presence_suite_baseline,
    "honesty_panel_synthetic_v1": measure_honesty_panel_baseline,
    "quant_panel_synthetic_v1": measure_quant_panel_baseline,
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
                   "the_will_axis_is_scored_by_tf125",
                   "regression_gates_fail_on_moved_baselines",
                   "spot_checks_anchor_the_numbers_in_human_judgement"):
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
    # Axis coverage as filed: the scorer/runner kinds cover
    # depth + presence; suite-kind entries are not counted
    # here (TF-125's presence_suite entry carries the will
    # axis, filed in its own entry and in HARNESS_RULES).
    covered = {axis for b in BENCHMARKS for axis in b["claims_axes"]
               if b["kind"] in ("scorer", "runner")}
    if covered != {"depth", "presence"}:
        problems.append(
            f"filed axis coverage {sorted(covered)} != depth + presence "
            "(the scorer/runner coverage as filed; a silent change "
            "is a finding)")

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
    "measure_depth_suite_baseline",
    "measure_honesty_panel_baseline",
    "measure_memory_suite_baseline",
    "measure_persona_baseline",
    "measure_presence_suite_baseline",
    "measure_persona_suite_baseline",
    "measure_profile_baseline",
    "measure_profile_panel_baseline",
    "measure_quant_panel_baseline",
    "measure_regression_gate_baseline",
    "measure_spot_checks_baseline",
    "validate_profile_id",
]
