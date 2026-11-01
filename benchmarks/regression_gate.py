"""
Regression gates, as data (TF-129).

The roadmap's rule: *regression gates: benchmark
deltas beyond tolerance fail the campaign's
slice gate (thresholds from baseline noise,
measured).* TF-121's harness pinned the estate's
baselines and built ``baseline_problems()``,
which re-measures them live and *names* any
metric that moved beyond its filed tolerance.
Naming is not gating: no slice was ever failed
by a moved baseline, no verdict existed for a
gate to consume, a move was never classified as
a regression, an improvement or a count change,
and the tolerances themselves were one uniform
number asserted at pinning time. This slice
builds the gate on top of that estate, exactly
as TF-121's filed limits assigned it here.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and nothing gated
    it.** ``baseline_problems()`` returned []
    and ``harness_problems()`` returned [] over
    the nine pinned baselines (83 pinned metrics
    in total, counted live) — and that pair of
    checkers was the whole story: no verdict
    object, no PASS / FAIL semantics a slice
    gate could consume, and no classification
    of a move existed anywhere in the estate.
  * **The tolerances were uniform and asserted,
    never measured.** Every pinned baseline
    carries ``tolerance: 1e-9`` — a reasonable
    floor, but no slice had ever re-measured a
    baseline repeatedly and looked at the spread,
    so nothing in the tree said whether 1e-9 was
    the estate's noise, far above it, or below
    it.
  * **The gap stood filed as future work.**
    ``benchmarks/harness.py``'s filed-limits
    passages listed "TF-129's regression gates"
    as standing in every TF-122..TF-128 update
    paragraph (seven places, grepped), and the
    number **D-BENCH-14** was verified unused
    anywhere in benchmarks / docs / tests / src
    before this slice filed it (the
    D-BENCH-9..13 pattern: a gap assigned by the
    roadmap without ever being numbered).

What this module files, as verifiable data:

  * ``NOISE_ROWS`` — the baseline noise,
    measured: every gated baseline re-measured
    ``NOISE_REPEATS`` (5) times in one process
    through the harness's own measurers (no
    corpus or scorer copied), each row filing
    its repeats, its metric count, its maximum
    absolute spread and any metric whose spread
    was nonzero, named. The repeat count is
    filed from measured cost: one full
    re-measurement of all nine baselines took
    ~1.6 s warm in-process on this VM at this
    slice's probes (five repeats ~8.1 s) — five
    repeats is cheap enough for a checker a
    slice actually runs, and enough samples to
    see a spread if one existed. **The measured
    truth on this tree: every spread is 0.0** —
    no metric of any baseline moved at all
    across five repeats, and the one dict metric
    (``quality_tiers``) never changed. That is
    TF-036's determinism holding, measured here
    rather than assumed; any future nonzero
    spread is filed by name, never smoothed.
  * ``TOLERANCES`` — the gate tolerances,
    derived from that noise by a filed rule, per
    metric: ``max(measured noise spread, 1e-9)``
    for float metrics, where 1e-9 is the
    harness's already-filed baseline tolerance
    floor (TF-121). Noise can only widen a
    tolerance to what was actually observed —
    never to an invented number. Integer / count
    metrics and the dict metric are **exact**:
    any change at all is beyond tolerance. On
    this tree every float tolerance therefore
    files the floor, 1e-9, because the measured
    spreads are all 0.0 — the derivation is
    recomputed live by the checker from
    recomputed noise, not merely asserted here.
  * **The gate itself** — ``evaluate_gate()``
    is a pure evaluation over a
    measured-by-baseline mapping: per baseline
    and per metric it files the pinned value,
    the measured value, the delta, the derived
    tolerance and a classification —
    ``within_tolerance``, ``regression``,
    ``improvement``, ``count_change`` or
    ``unmeasurable`` — from the filed direction
    table (score-like metrics higher-is-better;
    ``*_loss`` metrics lower-is-better; count /
    total metrics, descriptive hit-counts and
    the dict metric classify any beyond-tolerance
    move as a count change). **Any**
    beyond-tolerance move is a gate failure,
    improvements included: a moved baseline must
    be re-pinned deliberately by a slice, never
    passed silently — ``baseline_problems()``'
    filed precedent, now with verdict semantics.
    ``run_regression_gate()`` re-measures live
    through the harness measurers and evaluates:
    PASS only when a FULL selection (all nine
    gated baselines) re-measures and every
    metric is within tolerance; FAIL when any
    metric is beyond tolerance or any baseline
    fails to re-measure; a partial selection
    files REVIEW however it scores; an empty
    selection files 0 baselines / REVIEW — no
    vacuous pass.
  * ``GATE_PROBES`` — proof the gate can fail,
    filed as data and run by the checker against
    the pure evaluator with synthetic perturbed
    measurements (real baselines are never
    perturbed): a score dropped by 0.1 fails as
    a regression, a score raised by 0.1 fails as
    an improvement, a count changed fails as a
    count change, a loss dropped fails as an
    improvement on the lower-is-better leg, the
    dict metric changed fails as a count change,
    a move of 5e-10 inside tolerance evaluates
    PASS-shaped, and a missing baseline fails as
    unmeasurable.

D-BENCH-14 is FIXED by this slice: the pinned
baselines now have a gate that consumes them —
noise-derived tolerances, classified deltas and
a verdict a slice gate fails on. What is NOT
fixed, and is stated in the filed limits: the
gate judges the harness's pinned fixture
baselines only (no live model is gated); the
noise was measured in one process on this VM;
the gate is a checker a slice runs — wiring it
into any external CI is NOT done and NOT
claimed; and generation-time enforcement
blocking (the share D-BENCH-9's docket note
left for the Part V / TF-129 line to take or
refuse openly) is **refused here, openly**:
citation-truth blocking at generation time is
not this slice, and nothing in this module
blocks a generation.

Filed limits, stated plainly:

  * The gate judges the harness's pinned
    fixture baselines only — synthetic fixtures
    through the real scorers (TF-017). No live
    model is gated; live behaviour is NOT
    PROVEN (TF-173).
  * Noise was measured in one process on this
    VM (five repeats, in-process). Cross-process
    and cross-machine noise are not measured
    here — TF-135's flake discipline owns that
    share, and nothing here claims it.
  * The gate is a checker a slice runs
    (``regression_gate_problems()`` /
    ``run_regression_gate()`` and their tests).
    Wiring it into any external CI is NOT done
    and NOT claimed — the house runs manual-only
    workflows and in-house verification.
  * TF-129 does NOT wire generation-time
    enforcement blocking: the gate fails a
    *slice* whose benchmarks moved; it does not
    sit in any generation path.

The human-readable write-up is
``docs/architecture/REGRESSION_GATES.md``.
Proof grade: FIXTURE — the harness's pinned
synthetic-fixture baselines re-measured through
their real scorers on this VM; no live model,
never claimed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGRESSION_GATE_FORMAT = "thoughtforge-regression-gate/1"
REGRESSION_GATE_DOC = "docs/architecture/REGRESSION_GATES.md"

# The real machinery, reused — never copied: the
# harness's pinned baselines and its own
# measurers. This module imports the harness at
# top level; the harness binds this module only
# lazily (inside its measure function), so no
# import cycle exists.
from benchmarks import harness as _harness  # noqa: E402
from benchmarks.harness import BASELINES  # noqa: E402

#: The new baseline this slice registers in the
#: harness — the gate's own pin. It is NOT one of
#: the gated baselines (a gate does not gate its
#: own verdict row; its checker judges it).
GATE_BASELINE_ID = "regression_gate_synthetic_v1"

#: TF-130 update, in the open: TF-130 registered
#: one further baseline after this gate was
#: filed — the human-shaped spot-check set's own
#: pin. It is NOT one of the gated baselines
#: either, by the same shape as the gate's own
#: baseline: the spot-check set is a
#: human-judgement artefact judged by its own
#: module's checker (spot_checks_problems()),
#: and the gated set stays the nine
#: TF-121..TF-128 baselines exactly as TF-129
#: filed it. The gated-set check below excludes
#: the ungated pins by name (TF-131 added a
#: third, below).
SPOT_CHECKS_BASELINE_ID = "spot_checks_synthetic_v1"

#: TF-131 update, in the open: TF-131 registered
#: one further baseline after TF-130 — the
#: benchmark data hygiene inventory's own pin.
#: It is NOT one of the gated baselines either,
#: by the same shape: the inventory is a
#: provenance filing judged by its own module's
#: checker (data_hygiene_problems()), and the
#: gated set stays the nine TF-121..TF-128
#: baselines exactly as TF-129 filed it. The
#: gated-set check below excludes all three
#: ungated pins by name.
DATA_HYGIENE_BASELINE_ID = "data_hygiene_synthetic_v1"

#: TF-132 update, in the open: TF-132 registered
#: one further baseline after TF-131 — the
#: WYRD-shaped panel's own pin. It is NOT one of
#: the gated baselines either, by the same
#: shape: the panel is a world-state filing
#: judged by its own module's checker
#: (wyrd_panel_problems()), and the gated set
#: stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all four ungated pins
#: by name.
WYRD_PANEL_BASELINE_ID = "wyrd_panel_synthetic_v1"

#: TF-133 update, in the open: TF-133 registered
#: one further baseline after TF-132 — the
#: cost panel's own pin. It is NOT one of the
#: gated baselines either, by the same
#: shape: the panel is a cost filing judged
#: by its own module's checker
#: (cost_panel_problems()), and the gated set
#: stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all five ungated pins
#: by name.
COST_PANEL_BASELINE_ID = "cost_panel_synthetic_v1"

#: TF-134 update, in the open: TF-134 registered
#: one further baseline after TF-133 — the
#: baseline book's own pin. It is NOT one of
#: the gated baselines either, by the same
#: shape: the book is a regenerated filing
#: judged by its own module's checker
#: (baseline_book_problems()), and the gated
#: set stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all six ungated pins
#: by name.
BASELINE_BOOK_BASELINE_ID = "baseline_book_synthetic_v1"

#: TF-135 update, in the open: TF-135 registered
#: one further baseline after TF-134 — the
#: flake discipline's own pin. It is NOT one of
#: the gated baselines either, by the same
#: shape: the discipline is a census filing
#: judged by its own module's checker
#: (flake_discipline_problems()), and the gated
#: set stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all seven ungated pins
#: by name.
FLAKE_DISCIPLINE_BASELINE_ID = "flake_discipline_synthetic_v1"

#: TF-136 update, in the open: TF-136 registered
#: one further baseline after TF-135 — the
#: public-claims audit's own pin. It is NOT one of
#: the gated baselines either, by the same
#: shape: the audit is a claims filing judged
#: by its own module's checker
#: (public_claims_problems()), and the gated
#: set stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all eight ungated pins
#: by name.
PUBLIC_CLAIMS_BASELINE_ID = "public_claims_synthetic_v1"

#: TF-137 update, in the open: TF-137 registered
#: one further baseline after TF-136 — the
#: harness tool's own pin. It is NOT one of the
#: gated baselines either, by the same
#: shape: the tool is a panel runner judged
#: by its own module's checker
#: (harness_tool_problems()), and the gated
#: set stays the nine TF-121..TF-128 baselines
#: exactly as TF-129 filed it. The gated-set
#: check below excludes all nine ungated pins
#: by name.
HARNESS_TOOL_BASELINE_ID = "harness_tool_synthetic_v1"

#: The harness baselines that are not gated:
#: the gate's own verdict row, TF-130's
#: spot-check pin, TF-131's data-hygiene
#: pin, TF-132's wyrd-panel pin,
#: TF-133's cost-panel pin, TF-134's
#: baseline-book pin, TF-135's
#: flake-discipline pin, TF-136's
#: public-claims pin, and TF-137's
#: harness-tool pin, each judged by its
#: own module's checker.
UNGATED_BASELINE_IDS: tuple[str, ...] = (
    GATE_BASELINE_ID,
    SPOT_CHECKS_BASELINE_ID,
    DATA_HYGIENE_BASELINE_ID,
    WYRD_PANEL_BASELINE_ID,
    COST_PANEL_BASELINE_ID,
    BASELINE_BOOK_BASELINE_ID,
    FLAKE_DISCIPLINE_BASELINE_ID,
    PUBLIC_CLAIMS_BASELINE_ID,
    HARNESS_TOOL_BASELINE_ID,
)

#: The baselines the gate gates: the nine pinned
#: by TF-121..TF-128, in harness filing order.
GATED_BASELINE_IDS: tuple[str, ...] = (
    "persona_consistency_synthetic_v1",
    "persona_suite_synthetic_v1",
    "memory_suite_synthetic_v1",
    "depth_suite_synthetic_v1",
    "presence_suite_synthetic_v1",
    "honesty_panel_synthetic_v1",
    "quant_panel_synthetic_v1",
    "profile_panel_synthetic_v1",
    "profile_benchmark_empty_store_v1",
)

#: How many times the noise measurement re-runs
#: every gated baseline in one process. Filed
#: from measured cost (~1.6 s warm per full pass
#: on this VM at this slice's probes, ~8.1 s for
#: five): cheap enough for a checker a slice
#: actually runs, enough samples to see a spread.
NOISE_REPEATS = 5

#: The tolerance floor: the harness's
#: already-filed baseline tolerance (TF-121).
#: Noise may widen a tolerance to what was
#: observed; it may never narrow one below the
#: floor, and no wider number is ever invented.
TOLERANCE_FLOOR = 1e-9

#: Total pinned metrics across the gated
#: baselines, counted live from the harness.
TOTAL_PINNED_METRICS = sum(
    len(b["metrics"]) for b in BASELINES
    if b["id"] in GATED_BASELINE_IDS)


# ── Directions: which way is better, per metric ─────────────────────────────
#
# Filed as data, resolved by direction_for():
# *_loss metrics are lower-is-better; count /
# total metrics and descriptive hit-counts are
# counts (any move is a count change); the dict
# metric and the one descriptive float below are
# exact-shape measures, not scores; every other
# float metric in the gated baselines is a
# score-like rate / score / delta, higher better.

#: Float metrics that are descriptive measures,
#: not scores: a move in either direction is a
#: shape change, filed as a count change.
DESCRIPTIVE_FLOAT_METRICS: tuple[str, ...] = (
    "avg_response_words",
)

#: Suffixes that mark a count metric even when a
#: future pin files it as a float.
COUNT_SUFFIXES: tuple[str, ...] = (
    "_turns", "_cases", "_cells", "_hits", "_words",
)


def direction_for(metric: str, pinned: Any) -> str:
    """One metric's direction, from the filed
    table: "higher_is_better", "lower_is_better",
    "count" (any change is a count change) or
    "exact" (non-numeric; any change is a count
    change)."""
    if isinstance(pinned, dict) or pinned is None \
            or isinstance(pinned, (str, bool)):
        return "exact"
    if metric in DESCRIPTIVE_FLOAT_METRICS:
        return "count"
    if isinstance(pinned, int):
        return "count"
    if metric.endswith("_loss"):
        return "lower_is_better"
    if metric.endswith(COUNT_SUFFIXES):
        return "count"
    return "higher_is_better"


# ── The noise measurement ───────────────────────────────────────────────────

def _spread_for(values: list[Any]) -> tuple[float | None, bool]:
    """(max-min spread, changed) for one metric's
    repeated values: numeric metrics file their
    absolute spread; non-numeric metrics file
    None + whether any repeat differed."""
    first = values[0]
    if isinstance(first, (int, float)) \
            and not isinstance(first, bool):
        nums = [float(v) for v in values]
        return max(nums) - min(nums), max(nums) != min(nums)
    changed = any(v != first for v in values[1:])
    return None, changed


def compute_noise_rows(
    repeats: int = NOISE_REPEATS,
) -> tuple[dict[str, Any], ...]:
    """The noise, measured live: every gated
    baseline re-measured `repeats` times in this
    process through the harness's own measurers,
    per baseline the spread of every metric."""
    rows: list[dict[str, Any]] = []
    baselines = {b["id"]: b for b in BASELINES}
    for baseline_id in GATED_BASELINE_IDS:
        baseline = baselines[baseline_id]
        measurer = _harness._MEASURERS[baseline_id]
        runs = [measurer() for _ in range(repeats)]
        spreads: dict[str, float | None] = {}
        nonzero: list[str] = []
        numeric_spreads: list[float] = []
        for metric in baseline["metrics"]:
            spread, changed = _spread_for(
                [run[metric] for run in runs])
            spreads[metric] = spread
            if spread is not None:
                numeric_spreads.append(spread)
            if changed:
                nonzero.append(metric)
        rows.append({
            "baseline": baseline_id,
            "benchmark": baseline["benchmark"],
            "repeats": repeats,
            "metrics_measured": len(baseline["metrics"]),
            "max_abs_spread": max(numeric_spreads)
            if numeric_spreads else 0.0,
            "metrics_with_nonzero_spread": nonzero,
            "spreads": spreads,
        })
    return tuple(rows)


# ── The published noise (literal data, TF-067's CURVE precedent) ────────────
#
# Measured live on this VM at banking (2026-11-01):
# every gated baseline re-measured five times in
# one process through the harness's measurers —
# every spread 0.0, the dict metric never changed
# (TF-036's determinism, measured). The checker
# recomputes every row live and names any drift.

NOISE_ROWS: tuple[dict[str, Any], ...] = (
    {"baseline": "persona_consistency_synthetic_v1", "benchmark": "persona_consistency",
     "repeats": 5, "metrics_measured": 7,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"consistency_score": 0.0, "total_turns": 0.0, "flagged_turns": 0.0, "generic_phrase_hits": 0.0, "norse_tone_hits": 0.0, "citation_turns": 0.0, "skald_quality_hits": 0.0}},
    {"baseline": "persona_suite_synthetic_v1", "benchmark": "persona_suite",
     "repeats": 5, "metrics_measured": 5,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"identity_score": 0.0, "voice_score": 0.0, "values_score": 0.0, "overall_score": 0.0, "total_turns": 0.0}},
    {"baseline": "memory_suite_synthetic_v1", "benchmark": "memory_suite",
     "repeats": 5, "metrics_measured": 5,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"present_score": 0.0, "true_score": 0.0, "explain_score": 0.0, "overall_score": 0.0, "total_cases": 0.0}},
    {"baseline": "depth_suite_synthetic_v1", "benchmark": "depth_suite",
     "repeats": 5, "metrics_measured": 10,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"chain_off_score": 0.0, "chain_on_score": 0.0, "plan_off_score": 0.0, "plan_on_score": 0.0, "check_off_score": 0.0, "check_on_score": 0.0, "overall_off_score": 0.0, "overall_on_score": 0.0, "delta": 0.0, "total_cases": 0.0}},
    {"baseline": "presence_suite_synthetic_v1", "benchmark": "presence_suite",
     "repeats": 5, "metrics_measured": 5,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"coherence_score": 0.0, "steering_score": 0.0, "closure_score": 0.0, "overall_score": 0.0, "total_cases": 0.0}},
    {"baseline": "honesty_panel_synthetic_v1", "benchmark": "honesty_panel",
     "repeats": 5, "metrics_measured": 13,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"reference_off_score": 0.0, "reference_on_score": 0.0, "reference_delta": 0.0, "mid_off_score": 0.0, "mid_on_score": 0.0, "mid_delta": 0.0, "small_off_score": 0.0, "small_on_score": 0.0, "small_delta": 0.0, "tiny_off_score": 0.0, "tiny_on_score": 0.0, "tiny_delta": 0.0, "total_cells": 0.0}},
    {"baseline": "quant_panel_synthetic_v1", "benchmark": "quant_panel",
     "repeats": 5, "metrics_measured": 16,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"iq2_xs_score": 0.0, "iq2_xs_loss": 0.0, "q2_k_score": 0.0, "q2_k_loss": 0.0, "q4_k_m_score": 0.0, "q4_k_m_loss": 0.0, "q5_k_m_score": 0.0, "q5_k_m_loss": 0.0, "q8_0_score": 0.0, "q8_0_loss": 0.0, "fp16_score": 0.0, "fp16_loss": 0.0, "fact_mean_score": 0.0, "reasoning_mean_score": 0.0, "instruction_mean_score": 0.0, "total_cells": 0.0}},
    {"baseline": "profile_panel_synthetic_v1", "benchmark": "profile_panel",
     "repeats": 5, "metrics_measured": 13,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"desktop_cpu_suite_score": 0.0, "desktop_cpu_bench_avg_words": 0.0, "desktop_gpu_suite_score": 0.0, "desktop_gpu_bench_avg_words": 0.0, "phone_low_suite_score": 0.0, "phone_low_bench_avg_words": 0.0, "pi_5_suite_score": 0.0, "pi_5_bench_avg_words": 0.0, "pi_zero_suite_score": 0.0, "pi_zero_bench_avg_words": 0.0, "server_gpu_suite_score": 0.0, "server_gpu_bench_avg_words": 0.0, "total_cells": 0.0}},
    {"baseline": "profile_benchmark_empty_store_v1", "benchmark": "profile_benchmark",
     "repeats": 5, "metrics_measured": 9,
     "max_abs_spread": 0.0,
     "metrics_with_nonzero_spread": [],
     "spreads": {"total_turns": 0.0, "successful_turns": 0.0, "failed_turns": 0.0, "citation_accuracy": 0.0, "enforcement_pass_rate": 0.0, "avg_response_words": 0.0, "min_words_met_rate": 0.0, "avg_confidence": 0.0, "quality_tiers": None}},
)


# ── Tolerance derivation ────────────────────────────────────────────────────

def tolerances_for(
    noise_rows: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """The gate tolerances derived from noise
    rows by the filed rule: per metric,
    max(measured spread, the 1e-9 floor) for
    float metrics; exact (tolerance 0.0) for
    count and non-numeric metrics. Nothing wider
    than an observed spread is ever filed."""
    spreads = {r["baseline"]: r["spreads"] for r in noise_rows}
    derived: dict[str, dict[str, dict[str, Any]]] = {}
    for baseline in BASELINES:
        if baseline["id"] not in GATED_BASELINE_IDS:
            continue
        per_metric: dict[str, dict[str, Any]] = {}
        for metric, pinned in baseline["metrics"].items():
            if direction_for(metric, pinned) in ("count", "exact"):
                per_metric[metric] = {
                    "tolerance": 0.0, "exact": True}
            else:
                spread = spreads[baseline["id"]][metric]
                per_metric[metric] = {
                    "tolerance": max(
                        float(spread or 0.0), TOLERANCE_FLOOR),
                    "exact": False}
        derived[baseline["id"]] = per_metric
    return derived


#: The published tolerances: literal data derived
#: from NOISE_ROWS by the filed rule at banking.
#: The checker re-derives them from live noise
#: and names any drift.
TOLERANCES: dict[str, dict[str, dict[str, Any]]] = (
{
    "persona_consistency_synthetic_v1": {
        "consistency_score": {"tolerance": 1e-09, "exact": False},
        "total_turns": {"tolerance": 0.0, "exact": True},
        "flagged_turns": {"tolerance": 0.0, "exact": True},
        "generic_phrase_hits": {"tolerance": 0.0, "exact": True},
        "norse_tone_hits": {"tolerance": 0.0, "exact": True},
        "citation_turns": {"tolerance": 0.0, "exact": True},
        "skald_quality_hits": {"tolerance": 0.0, "exact": True},
    },
    "persona_suite_synthetic_v1": {
        "identity_score": {"tolerance": 1e-09, "exact": False},
        "voice_score": {"tolerance": 1e-09, "exact": False},
        "values_score": {"tolerance": 1e-09, "exact": False},
        "overall_score": {"tolerance": 1e-09, "exact": False},
        "total_turns": {"tolerance": 0.0, "exact": True},
    },
    "memory_suite_synthetic_v1": {
        "present_score": {"tolerance": 1e-09, "exact": False},
        "true_score": {"tolerance": 1e-09, "exact": False},
        "explain_score": {"tolerance": 1e-09, "exact": False},
        "overall_score": {"tolerance": 1e-09, "exact": False},
        "total_cases": {"tolerance": 0.0, "exact": True},
    },
    "depth_suite_synthetic_v1": {
        "chain_off_score": {"tolerance": 1e-09, "exact": False},
        "chain_on_score": {"tolerance": 1e-09, "exact": False},
        "plan_off_score": {"tolerance": 1e-09, "exact": False},
        "plan_on_score": {"tolerance": 1e-09, "exact": False},
        "check_off_score": {"tolerance": 1e-09, "exact": False},
        "check_on_score": {"tolerance": 1e-09, "exact": False},
        "overall_off_score": {"tolerance": 1e-09, "exact": False},
        "overall_on_score": {"tolerance": 1e-09, "exact": False},
        "delta": {"tolerance": 1e-09, "exact": False},
        "total_cases": {"tolerance": 0.0, "exact": True},
    },
    "presence_suite_synthetic_v1": {
        "coherence_score": {"tolerance": 1e-09, "exact": False},
        "steering_score": {"tolerance": 1e-09, "exact": False},
        "closure_score": {"tolerance": 1e-09, "exact": False},
        "overall_score": {"tolerance": 1e-09, "exact": False},
        "total_cases": {"tolerance": 0.0, "exact": True},
    },
    "honesty_panel_synthetic_v1": {
        "reference_off_score": {"tolerance": 1e-09, "exact": False},
        "reference_on_score": {"tolerance": 1e-09, "exact": False},
        "reference_delta": {"tolerance": 1e-09, "exact": False},
        "mid_off_score": {"tolerance": 1e-09, "exact": False},
        "mid_on_score": {"tolerance": 1e-09, "exact": False},
        "mid_delta": {"tolerance": 1e-09, "exact": False},
        "small_off_score": {"tolerance": 1e-09, "exact": False},
        "small_on_score": {"tolerance": 1e-09, "exact": False},
        "small_delta": {"tolerance": 1e-09, "exact": False},
        "tiny_off_score": {"tolerance": 1e-09, "exact": False},
        "tiny_on_score": {"tolerance": 1e-09, "exact": False},
        "tiny_delta": {"tolerance": 1e-09, "exact": False},
        "total_cells": {"tolerance": 0.0, "exact": True},
    },
    "quant_panel_synthetic_v1": {
        "iq2_xs_score": {"tolerance": 1e-09, "exact": False},
        "iq2_xs_loss": {"tolerance": 1e-09, "exact": False},
        "q2_k_score": {"tolerance": 1e-09, "exact": False},
        "q2_k_loss": {"tolerance": 1e-09, "exact": False},
        "q4_k_m_score": {"tolerance": 1e-09, "exact": False},
        "q4_k_m_loss": {"tolerance": 1e-09, "exact": False},
        "q5_k_m_score": {"tolerance": 1e-09, "exact": False},
        "q5_k_m_loss": {"tolerance": 1e-09, "exact": False},
        "q8_0_score": {"tolerance": 1e-09, "exact": False},
        "q8_0_loss": {"tolerance": 1e-09, "exact": False},
        "fp16_score": {"tolerance": 1e-09, "exact": False},
        "fp16_loss": {"tolerance": 1e-09, "exact": False},
        "fact_mean_score": {"tolerance": 1e-09, "exact": False},
        "reasoning_mean_score": {"tolerance": 1e-09, "exact": False},
        "instruction_mean_score": {"tolerance": 1e-09, "exact": False},
        "total_cells": {"tolerance": 0.0, "exact": True},
    },
    "profile_panel_synthetic_v1": {
        "desktop_cpu_suite_score": {"tolerance": 1e-09, "exact": False},
        "desktop_cpu_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "desktop_gpu_suite_score": {"tolerance": 1e-09, "exact": False},
        "desktop_gpu_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "phone_low_suite_score": {"tolerance": 1e-09, "exact": False},
        "phone_low_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "pi_5_suite_score": {"tolerance": 1e-09, "exact": False},
        "pi_5_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "pi_zero_suite_score": {"tolerance": 1e-09, "exact": False},
        "pi_zero_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "server_gpu_suite_score": {"tolerance": 1e-09, "exact": False},
        "server_gpu_bench_avg_words": {"tolerance": 0.0, "exact": True},
        "total_cells": {"tolerance": 0.0, "exact": True},
    },
    "profile_benchmark_empty_store_v1": {
        "total_turns": {"tolerance": 0.0, "exact": True},
        "successful_turns": {"tolerance": 0.0, "exact": True},
        "failed_turns": {"tolerance": 0.0, "exact": True},
        "citation_accuracy": {"tolerance": 1e-09, "exact": False},
        "enforcement_pass_rate": {"tolerance": 1e-09, "exact": False},
        "avg_response_words": {"tolerance": 0.0, "exact": True},
        "min_words_met_rate": {"tolerance": 1e-09, "exact": False},
        "avg_confidence": {"tolerance": 1e-09, "exact": False},
        "quality_tiers": {"tolerance": 0.0, "exact": True},
    },
}
)


# ── The gate: pure evaluation ───────────────────────────────────────────────

def evaluate_gate(
    measured_by_baseline_id: Any,
    tolerances: Any = None,
) -> dict[str, Any]:
    """Judge measured values against the pins,
    purely: per gated baseline + metric a delta
    row with its classification. No measurement
    happens here — this is the function the
    filed probes exercise with synthetic inputs.

    Returns {"rows": [...], "failures": int,
    "metrics_compared": int, "outcome":
    "PASS" | "FAIL"} — the outcome is the
    evaluation's shape only; run_regression_gate
    files the gate's verdict (a partial or empty
    selection is REVIEW there, never a PASS
    here dressed as one).
    """
    tolerances = TOLERANCES if tolerances is None else tolerances
    measured_by_baseline_id = measured_by_baseline_id or {}
    rows: list[dict[str, Any]] = []
    failures = 0
    for baseline in BASELINES:
        baseline_id = baseline["id"]
        if baseline_id not in GATED_BASELINE_IDS:
            continue
        measured = measured_by_baseline_id.get(baseline_id)
        for metric, pinned in baseline["metrics"].items():
            direction = direction_for(metric, pinned)
            tolerance_row = tolerances[baseline_id][metric]
            tolerance = float(tolerance_row["tolerance"])
            row: dict[str, Any] = {
                "baseline": baseline_id,
                "benchmark": baseline["benchmark"],
                "metric": metric,
                "pinned": pinned,
                "measured": None,
                "delta": None,
                "tolerance": tolerance,
                "direction": direction,
                "classification": "unmeasurable",
                "beyond_tolerance": True,
            }
            if measured is None or metric not in measured:
                failures += 1
                rows.append(row)
                continue
            got = measured[metric]
            row["measured"] = got
            if direction == "exact":
                if got == pinned:
                    row.update(classification="within_tolerance",
                               beyond_tolerance=False, delta=0.0)
                else:
                    row.update(classification="count_change")
                    failures += 1
                rows.append(row)
                continue
            if not isinstance(got, (int, float)) \
                    or isinstance(got, bool):
                failures += 1
                rows.append(row)
                continue
            delta = float(got) - float(pinned)
            row["delta"] = delta
            if tolerance_row["exact"]:
                beyond = delta != 0.0
            else:
                beyond = abs(delta) > tolerance
            if not beyond:
                row.update(classification="within_tolerance",
                           beyond_tolerance=False)
            else:
                failures += 1
                if direction == "count":
                    row["classification"] = "count_change"
                elif direction == "higher_is_better":
                    row["classification"] = (
                        "improvement" if delta > 0 else "regression")
                else:  # lower_is_better
                    row["classification"] = (
                        "improvement" if delta < 0 else "regression")
            rows.append(row)
    return {
        "rows": rows,
        "failures": failures,
        "metrics_compared": len(rows),
        "outcome": "FAIL" if failures else "PASS",
    }


# ── The gate probes (filed as data; synthetic perturbations only) ───────────
#
# Each probe perturbs ONE pinned value in a copy
# of the pins (or drops one baseline) and states
# the evaluation it must produce. Real baselines
# are never perturbed — the evaluator is pure, so
# the probes need nothing real to break.

GATE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "score_dropped_is_a_regression_failure",
     "baseline": "persona_suite_synthetic_v1",
     "metric": "overall_score", "delta_applied": -0.1,
     "drop_baseline": False,
     "expected_outcome": "FAIL",
     "expected_classification": "regression"},
    {"id": "score_raised_is_an_improvement_failure",
     "baseline": "persona_suite_synthetic_v1",
     "metric": "overall_score", "delta_applied": 0.1,
     "drop_baseline": False,
     "expected_outcome": "FAIL",
     "expected_classification": "improvement"},
    {"id": "count_changed_is_a_count_change_failure",
     "baseline": "persona_suite_synthetic_v1",
     "metric": "total_turns", "delta_applied": 1,
     "drop_baseline": False,
     "expected_outcome": "FAIL",
     "expected_classification": "count_change"},
    {"id": "loss_dropped_is_an_improvement_failure",
     "baseline": "quant_panel_synthetic_v1",
     "metric": "q4_k_m_loss", "delta_applied": -0.1,
     "drop_baseline": False,
     "expected_outcome": "FAIL",
     "expected_classification": "improvement"},
    {"id": "dict_metric_changed_is_a_count_change_failure",
     "baseline": "profile_benchmark_empty_store_v1",
     "metric": "quality_tiers", "delta_applied": None,
     "set_value": {"weak": 9, "moderate": 1},
     "drop_baseline": False,
     "expected_outcome": "FAIL",
     "expected_classification": "count_change"},
    {"id": "a_move_inside_tolerance_passes_the_evaluation",
     "baseline": "persona_consistency_synthetic_v1",
     "metric": "consistency_score", "delta_applied": 5e-10,
     "drop_baseline": False,
     "expected_outcome": "PASS",
     "expected_classification": "within_tolerance"},
    {"id": "a_missing_baseline_is_unmeasurable_and_fails",
     "baseline": "memory_suite_synthetic_v1",
     "metric": None, "delta_applied": None,
     "drop_baseline": True,
     "expected_outcome": "FAIL",
     "expected_classification": "unmeasurable"},
)


def _pinned_measurements() -> dict[str, dict[str, Any]]:
    """The pins themselves, deep-copied: the
    synthetic starting point every probe perturbs
    (feeding the pins back must evaluate clean)."""
    return {
        b["id"]: copy.deepcopy(dict(b["metrics"]))
        for b in BASELINES if b["id"] in GATED_BASELINE_IDS}


def run_gate_probes() -> list[dict[str, Any]]:
    """Run every filed probe against the pure
    evaluator and file whether it behaved."""
    results: list[dict[str, Any]] = []
    for probe in GATE_PROBES:
        measured = _pinned_measurements()
        if probe["drop_baseline"]:
            measured.pop(probe["baseline"], None)
        elif "set_value" in probe:
            measured[probe["baseline"]][probe["metric"]] = \
                copy.deepcopy(probe["set_value"])
        elif probe["delta_applied"] is not None:
            measured[probe["baseline"]][probe["metric"]] += \
                probe["delta_applied"]
        evaluation = evaluate_gate(measured)
        if probe["metric"] is not None:
            row = [r for r in evaluation["rows"]
                   if r["baseline"] == probe["baseline"]
                   and r["metric"] == probe["metric"]][0]
            classification = row["classification"]
        else:
            classification = evaluation["rows"][
                [r["baseline"] for r in evaluation["rows"]].index(
                    probe["baseline"])]["classification"]
        passed = bool(
            evaluation["outcome"] == probe["expected_outcome"]
            and classification == probe["expected_classification"])
        results.append({
            "id": probe["id"],
            "expected_outcome": probe["expected_outcome"],
            "outcome": evaluation["outcome"],
            "expected_classification":
                probe["expected_classification"],
            "classification": classification,
            "passed": passed,
        })
    return results


# ── The gate run ────────────────────────────────────────────────────────────

@dataclass
class RegressionGateResult:
    """One full gate run: rows, verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    baseline_ids: list[str] = field(default_factory=list)
    metrics_compared: int = 0
    failures: int = 0
    overall_score: float = 0.0

    @property
    def baselines_gated(self) -> int:
        return len(self.baseline_ids)

    @property
    def verdict(self) -> str:
        """PASS only on the full gated selection
        re-measured with every metric within
        tolerance; FAIL on a full selection with
        any failure; REVIEW for any partial or
        empty selection, however it scores."""
        if tuple(self.baseline_ids) != GATED_BASELINE_IDS:
            return "REVIEW"
        return "FAIL" if self.failures else "PASS"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_regression_gate(
    baseline_ids: Any = None,
) -> RegressionGateResult:
    """Re-measure the selection live through the
    harness measurers and evaluate it.

    Any explicit baseline selection may be given
    — including an empty one, which gates 0
    baselines and files REVIEW (no vacuous pass).
    A baseline whose measurer raises files
    unmeasurable rows, never a silent skip.
    """
    baseline_ids = GATED_BASELINE_IDS if baseline_ids is None \
        else tuple(baseline_ids)
    if not baseline_ids:
        return RegressionGateResult(
            rows=[], baseline_ids=[], metrics_compared=0,
            failures=0, overall_score=0.0)
    measured: dict[str, dict[str, Any]] = {}
    for baseline_id in baseline_ids:
        measurer = _harness._MEASURERS.get(baseline_id)
        if measurer is None:
            continue
        try:
            measured[baseline_id] = measurer()
        except Exception:
            continue  # unmeasurable: the evaluator files it
    # Evaluate over the selection only: restrict
    # the gated set for this call by judging the
    # selected baselines' rows out of the full
    # evaluation (the full evaluation of a partial
    # measurement would file the unselected
    # baselines unmeasurable, which is not what a
    # partial selection means — it files REVIEW
    # on its own rows, per the filed rule).
    full = evaluate_gate(measured)
    rows = [r for r in full["rows"]
            if r["baseline"] in baseline_ids]
    failures = sum(1 for r in rows if r["beyond_tolerance"])
    compared = len(rows)
    overall = (
        (compared - failures) / compared if compared else 0.0)
    return RegressionGateResult(
        rows=rows, baseline_ids=list(baseline_ids),
        metrics_compared=compared, failures=failures,
        overall_score=overall)


# ── The filing's rules ──────────────────────────────────────────────────────

GATE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "noise_is_measured_never_assumed",
     "rule": "the gate's tolerances rest on "
             "NOISE_ROWS: every gated baseline "
             "re-measured NOISE_REPEATS times in "
             "one process through the harness's "
             "own measurers, spreads filed per "
             "metric and recomputed live by the "
             "checker — a spread that appears is "
             "named, never smoothed"},
    {"id": "tolerances_derive_from_noise_over_a_filed_floor",
     "rule": "per float metric the gate tolerance "
             "is max(measured noise spread, 1e-9) "
             "— 1e-9 being TF-121's already-filed "
             "baseline tolerance floor; noise can "
             "only widen a tolerance to what was "
             "observed, never to an invented number"},
    {"id": "counts_and_dict_metrics_are_exact",
     "rule": "integer / count metrics and the "
             "dict metric (quality_tiers) are exact: "
             "any change at all is beyond tolerance "
             "and files a count change"},
    {"id": "directions_are_filed_as_data",
     "rule": "score-like metrics are "
             "higher-is-better, *_loss metrics are "
             "lower-is-better, count / total / "
             "hit-count metrics and the descriptive "
             "avg_response_words file count changes "
             "— the table is direction_for() over "
             "DESCRIPTIVE_FLOAT_METRICS + "
             "COUNT_SUFFIXES, judged by tests"},
    {"id": "any_beyond_tolerance_move_fails_the_gate",
     "rule": "a regression, an improvement and a "
             "count change all fail the gate alike: "
             "an improvement is a moved baseline "
             "that must be re-pinned deliberately "
             "by a slice, never passed silently "
             "(baseline_problems' filed precedent)"},
    {"id": "pass_needs_the_full_selection_within_tolerance",
     "rule": "the verdict is PASS only when all "
             "nine gated baselines re-measure and "
             "every one of their metrics is within "
             "its derived tolerance; any beyond-"
             "tolerance metric or any baseline "
             "failing to re-measure files FAIL"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "a partial selection files REVIEW "
             "however it scores and an empty "
             "selection gates 0 baselines / REVIEW "
             "— the gate never passes on less than "
             "the full pinned estate"},
    {"id": "probes_prove_the_gate_can_fail",
     "rule": "GATE_PROBES perturb synthetic "
             "copies of the pins against the pure "
             "evaluator — a dropped score, a raised "
             "score, a changed count, a dropped "
             "loss, a changed dict and a missing "
             "baseline each fail with their filed "
             "classification, and a 5e-10 move "
             "inside tolerance passes; real "
             "baselines are never perturbed"},
    {"id": "the_gate_judges_fixture_baselines_only",
     "rule": "the gated baselines are the "
             "harness's pinned synthetic-fixture "
             "baselines (TF-017) — no live model is "
             "gated and live behaviour is NOT "
             "PROVEN (TF-173); the gate's own "
             "baseline (regression_gate_synthetic_v1) "
             "is judged by this module's checker, "
             "not gated by the gate"},
    {"id": "a_checker_a_slice_runs_not_a_ci_wiring",
     "rule": "the gate is a checker a slice runs "
             "in-house (the house runs manual-only "
             "workflows); wiring it into any "
             "external CI is NOT done and NOT "
             "claimed, and generation-time "
             "enforcement blocking is refused "
             "openly — D-BENCH-9's generation-time "
             "share is not this slice"},
)


# ── The live checker ────────────────────────────────────────────────────────

def regression_gate_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: gated set, repeats, totals, rules.
    harness_ids = [b["id"] for b in BASELINES
                   if b["id"] not in UNGATED_BASELINE_IDS]
    if list(GATED_BASELINE_IDS) != harness_ids:
        problems.append(
            f"GATED_BASELINE_IDS {list(GATED_BASELINE_IDS)} != "
            f"the harness's pinned baselines {harness_ids}")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append(
            f"the gate gates {len(GATED_BASELINE_IDS)} baselines, "
            f"not 9")
    if NOISE_REPEATS != 5:
        problems.append(f"NOISE_REPEATS is {NOISE_REPEATS}, not 5")
    if TOLERANCE_FLOOR != 1e-9:
        problems.append(
            f"TOLERANCE_FLOOR is {TOLERANCE_FLOOR}, not TF-121's 1e-9")
    if TOTAL_PINNED_METRICS != sum(
            len(b['metrics']) for b in BASELINES
            if b['id'] in GATED_BASELINE_IDS):
        problems.append("TOTAL_PINNED_METRICS drifted from the pins")
    rule_ids = [r["id"] for r in GATE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate GATE_RULES ids")
    for needed in ("noise_is_measured_never_assumed",
                   "tolerances_derive_from_noise_over_a_filed_floor",
                   "counts_and_dict_metrics_are_exact",
                   "directions_are_filed_as_data",
                   "any_beyond_tolerance_move_fails_the_gate",
                   "pass_needs_the_full_selection_within_tolerance",
                   "no_vacuous_pass_anywhere",
                   "probes_prove_the_gate_can_fail",
                   "the_gate_judges_fixture_baselines_only",
                   "a_checker_a_slice_runs_not_a_ci_wiring"):
        if needed not in rule_ids:
            problems.append(f"GATE_RULES lost {needed}")

    # The noise: literal rows recomputed live.
    live_noise = compute_noise_rows()
    if [dict(r) for r in live_noise] != [dict(r) for r in NOISE_ROWS]:
        problems.append(
            "NOISE_ROWS drifted from the live recomputation")
    for row in NOISE_ROWS:
        if row["repeats"] != NOISE_REPEATS:
            problems.append(
                f"{row['baseline']}: filed repeats "
                f"{row['repeats']} != {NOISE_REPEATS}")
        if row["metrics_measured"] != len(row["spreads"]):
            problems.append(
                f"{row['baseline']}: filed metrics_measured "
                f"{row['metrics_measured']} != its spreads "
                f"{len(row['spreads'])}")
        for metric, spread in row["spreads"].items():
            if spread is not None and spread > 0 \
                    and metric not in row[
                        "metrics_with_nonzero_spread"]:
                problems.append(
                    f"{row['baseline']}: metric {metric} has a "
                    f"nonzero spread but is not named in "
                    f"metrics_with_nonzero_spread")
        numeric = [s for s in row["spreads"].values()
                   if s is not None]
        if numeric and row["max_abs_spread"] != max(numeric):
            problems.append(
                f"{row['baseline']}: filed max_abs_spread "
                f"{row['max_abs_spread']} != {max(numeric)}")

    # Tolerances derive from the noise, live.
    try:
        if tolerances_for(live_noise) != TOLERANCES:
            problems.append(
                "TOLERANCES drifted from the derivation over "
                "live noise")
    except Exception as exc:
        problems.append(
            f"TOLERANCES cannot be derived over live noise: {exc!r}")
    try:
        if tolerances_for(NOISE_ROWS) != TOLERANCES:
            problems.append(
                "TOLERANCES drifted from the derivation over "
                "NOISE_ROWS")
    except Exception as exc:
        problems.append(
            f"TOLERANCES cannot be derived over the filed "
            f"NOISE_ROWS: {exc!r}")
    for baseline_id, per_metric in TOLERANCES.items():
        for metric, row in per_metric.items():
            if not row["exact"] and row["tolerance"] < TOLERANCE_FLOOR:
                problems.append(
                    f"{baseline_id}.{metric}: tolerance "
                    f"{row['tolerance']} below the filed floor")

    # The real gate on this tree: full selection PASS.
    result = run_regression_gate()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live gate does not file PASS on this tree: "
            f"{result.verdict} ({result.failures} failures)")
    if result.baselines_gated != 9 \
            or result.metrics_compared != TOTAL_PINNED_METRICS:
        problems.append(
            f"the live gate files {result.baselines_gated} baselines "
            f"/ {result.metrics_compared} metrics, not 9 / "
            f"{TOTAL_PINNED_METRICS}")
    if result.failures != 0:
        problems.append(
            f"the live gate files {result.failures} failures on "
            f"this tree")

    # Partial + empty selections file REVIEW.
    empty = run_regression_gate(baseline_ids=())
    if empty.baselines_gated != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    subset = run_regression_gate(
        baseline_ids=("persona_consistency_synthetic_v1",))
    if subset.baselines_gated != 1 or subset.verdict != "REVIEW":
        problems.append(
            "a single-baseline selection does not gate its "
            "baseline / REVIEW mechanically")

    # The probes: the gate can fail, as filed.
    probes = run_gate_probes()
    if len(probes) != len(GATE_PROBES):
        problems.append("the probe run lost a filed probe")
    for probe_result in probes:
        if not probe_result["passed"]:
            problems.append(
                f"gate probe {probe_result['id']} does not behave "
                f"as filed: {probe_result}")

    # The pins fed back through the pure evaluator
    # evaluate clean (the probes' starting point).
    if evaluate_gate(_pinned_measurements())["outcome"] != "PASS":
        problems.append("the pins do not evaluate PASS-shaped")

    # Harness consistency as filed + every earlier
    # pinned baseline preserved exactly (the new
    # baseline included: baseline_problems judges
    # all ten).
    from benchmarks.harness import baseline_problems

    bench_ids = [b["id"] for b in _harness.BENCHMARKS]
    if "regression_gate" not in bench_ids:
        problems.append("BENCHMARKS lost the regression_gate entry")
    fixture_ids = [f["id"] for f in _harness.FIXTURES]
    if "regression_gate_corpus" not in fixture_ids:
        problems.append("FIXTURES lost the regression_gate_corpus entry")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the gate: {earlier}")

    doc = Path(__file__).resolve().parent.parent / REGRESSION_GATE_DOC
    if not doc.exists():
        problems.append(
            f"the write-up {REGRESSION_GATE_DOC} is missing")
    return problems


def regression_gate_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_regression_gate()
    probes = run_gate_probes()
    return {
        "format": REGRESSION_GATE_FORMAT,
        "doc": REGRESSION_GATE_DOC,
        "gated_baselines": list(GATED_BASELINE_IDS),
        "noise_repeats": NOISE_REPEATS,
        "tolerance_floor": TOLERANCE_FLOOR,
        "noise_rows": [dict(r) for r in NOISE_ROWS],
        "tolerances": {
            bid: {m: dict(t) for m, t in per.items()}
            for bid, per in TOLERANCES.items()},
        "probes": probes,
        "measured": {
            "baselines_gated": result.baselines_gated,
            "metrics_compared": result.metrics_compared,
            "gate_failures": result.failures,
            "max_noise_spread": max(
                r["max_abs_spread"] for r in NOISE_ROWS),
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in GATE_RULES],
    }


__all__ = [
    "COUNT_SUFFIXES",
    "DESCRIPTIVE_FLOAT_METRICS",
    "GATE_BASELINE_ID",
    "GATE_PROBES",
    "GATE_RULES",
    "GATED_BASELINE_IDS",
    "NOISE_REPEATS",
    "NOISE_ROWS",
    "REGRESSION_GATE_DOC",
    "REGRESSION_GATE_FORMAT",
    "SPOT_CHECKS_BASELINE_ID",
    "TOLERANCES",
    "TOLERANCE_FLOOR",
    "TOTAL_PINNED_METRICS",
    "UNGATED_BASELINE_IDS",
    "WYRD_PANEL_BASELINE_ID",
    "COST_PANEL_BASELINE_ID",
    "BASELINE_BOOK_BASELINE_ID",
    "RegressionGateResult",
    "compute_noise_rows",
    "direction_for",
    "evaluate_gate",
    "regression_gate_as_data",
    "regression_gate_problems",
    "run_gate_probes",
    "run_regression_gate",
    "tolerances_for",
]
