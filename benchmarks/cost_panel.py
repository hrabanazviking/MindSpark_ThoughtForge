"""
Cost panel, as data (TF-133).

The roadmap's rule: *cost panel: tokens + time
per suite per profile — sharpness per watt, the
edge-hardware point of the project.* TF-121's
harness deliberately pinned NO latency (walls
under contention are not baselines) and assigned
cost measurement to this slice; TF-121 also
filed **D-BENCH-7** narrowed with its remaining
share standing here: ``utils/perf.py``'s
``PerfTracker`` was wired into the core and
consumed by no benchmark. TF-128 ran the suites
per profile and filed their scores, but no
artefact in the estate ever filed what a suite
run *costs* — how much response-token budget a
profile spends per scored unit, how long the
scorer work takes, or what either means for the
small, edge profiles this project exists for.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and no cost panel
    existed.** ``harness_problems()`` returned
    [], ``baseline_problems()`` returned [] over
    the thirteen pinned baselines,
    ``regression_gate_problems()`` returned []
    with the live gate PASS (9 baselines / 83
    metrics / 0 failures),
    ``data_hygiene_problems()`` returned []
    (21 inputs, all clean),
    ``spot_checks_problems()`` returned [] and
    ``wyrd_panel_problems()`` returned [] with
    the live WYRD panel PASS (18 cases, overall
    0.5556) — and a grep for any cost / tokens /
    time-per-suite-per-profile panel across
    benchmarks / docs / tests / src found
    nothing but TF-121's assignment of exactly
    this share to TF-133.
  * **No shipped profile declares a power
    figure.** Grepped live this tick: no
    ``power`` / ``watt`` / ``tdp`` key or value
    anywhere in ``hardware_profiles/*.json`` or
    in the profile loaders. "Sharpness per watt"
    therefore cannot be measured or filed on
    this estate — see the watt rule below.
  * **The suite scorers are millisecond-scale
    scripted work on this VM.** Probed live with
    ``perf_counter`` (3 repeats each): persona
    ~7-59 ms, memory ~13-69 ms, depth ~1-2 ms,
    presence ~6-7 ms per run, first runs slowest
    (warm-up / contention shape, exactly the
    variance TF-121 filed). Those walls are
    filed here as repeat-measured live data —
    never pinned, never a baseline.
  * **The gap was never numbered.** The number
    **D-BENCH-18** was verified unused anywhere
    in benchmarks / docs / tests / src before
    this slice filed it (the D-BENCH-9..17
    pattern: a gap assigned by the roadmap
    without ever being numbered).

What this module files, as verifiable data:

  * ``PANEL_ROWS`` — the deterministic cost
    rows: one per suite x shipped profile (the
    four Part VII suites x the harness's
    ``KNOWN_PROFILES`` six — TF-128's estate:
    its 30 cells are these 24 suite cells plus
    its 6 profile-benchmark cells; the benchmark
    cell is not a suite and files no suite
    score, so it is not a cost row here). Each
    row files the suite's scored units (counted
    by its real runner: persona 108 turns,
    memory / depth / presence 36 cases each),
    the profile's filed budgets read live from
    its JSON through TF-128's ``filed_budgets``
    (``max_response_tokens`` +
    ``max_refinement_passes``), the suite's
    headline score through its real runner (the
    depth headline is the ON score where the
    profile budgets a refinement pass and the
    OFF score on pi_zero, whose 0-pass budget
    refuses the ON road — TF-128's filing,
    carried exactly), and the derived cost:
    ``token_budget`` and
    ``sharpness_per_1000_tokens``.
  * **Tokens, filed exactly for what they
    are.** ``token_budget`` = scored units x
    the profile's ``max_response_tokens``: the
    response-token budget one scored response
    per unit may spend under that profile,
    computed from the real runners' unit counts
    and the real profile budgets. It is a
    *budget exposure*, not a consumed-token
    count: the suites score scripted fixtures
    and no live model generated anything, so no
    consumed tokens exist to count, and none
    are claimed. ``sharpness_per_1000_tokens``
    = suite score / token_budget x 1000 — the
    score a cell files per thousand budgeted
    response tokens, derived only from the real
    suite scores and the filed budgets.
  * **Time, filed exactly for what it is.**
    ``run_cost_panel()`` times every cell's
    suite runner ``TIME_REPEATS`` (5) times
    with ``perf_counter`` and records every
    sample into TF-121 / D-BENCH-7's
    ``PerfTracker`` (``thoughtforge.utils.perf``
    — consumed here, in the open: the panel is
    the benchmark that consumes the tracker),
    filing per cell the tracker's own summary
    shape (repeats / mean / median / min / max
    ms). The timing is the *scorer's* wall time
    on this VM — scripted scorers, no model —
    measured under whatever contention the VM
    carries; it is NOT a portable baseline, NOT
    a per-profile hardware speed (the scorer
    work does not vary by profile; the profile
    changes budgets, not scorer code), and it
    is pinned nowhere: the harness baseline for
    this panel carries no time metric, by
    TF-121's rule, and the checker judges the
    timing's *shape* (present, repeated,
    positive, median inside [min, max]) never
    its values.
  * **Watts, filed exactly for what they
    are: NOT PROVEN / not measured.** No
    shipped profile declares a power figure
    (probed above; the checker re-scans the
    profile JSONs live), no power was measured
    on any hardware, and no watt is invented
    anywhere in this panel: every row files
    ``power_watts: None`` and
    ``power_status: "not_proven_not_measured"``,
    and the pure checker *fails* a row that
    claims a watt (probe-proven). If a future
    profile declares a power figure, a future
    slice may file it as *declared* — never as
    measured — by binding that JSON live. The
    roadmap's "sharpness per watt" therefore
    files here as sharpness per budgeted token
    plus an honest NOT PROVEN on watts — the
    edge-hardware point stands (the narrowest
    profile files the highest sharpness per
    budgeted token on every suite) without a
    fabricated denominator.
  * ``PANEL_PROBES`` — filed probes against the
    pure checker parts with perturbed copies of
    filed rows (real filings never perturbed):
    a wrong token budget is named, a wrong
    sharpness is named, an invented watt is
    named, a claimed measured power status is
    named, an out-of-range score is named, a
    bad timing shape is named — plus the clean
    row that must pass.
  * ``run_cost_panel()`` files the verdict:
    PASS only on the FULL filed matrix (all 24
    cells) with every deterministic row
    reproducing ``PANEL_ROWS`` and every cell
    carrying a well-shaped timing; a partial
    selection files REVIEW however it scores;
    an empty selection files 0 cells / REVIEW
    — no vacuous pass.

D-BENCH-18 is FIXED by this slice in the
measuring sense: tokens + time per suite per
profile now have a panel — budget-derived token
cost and repeat-measured scorer time, published
as data recomputed live, with a pinned harness
baseline (deterministic metrics only). And
**D-BENCH-7's consumption share is FIXED by
this slice**: the cost panel consumes
``PerfTracker`` for every timing it files, so
the tracker is no longer wired-but-unconsumed.
What is NOT fixed, and is stated in the filed
limits: watts / sharpness-per-watt are NOT
PROVEN (no declared power figure to bind, no
power measured); live-model token consumption
and live per-profile hardware time are NOT
PROVEN (TF-173). This slice's own harness
baseline is judged by this module's checker,
NOT gated by TF-129's regression gate (the
gate's gated set stays the nine TF-121..TF-128
baselines exactly as filed; the exclusion is
filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129..TF-132 filed for their own baselines).
TF-131 coupling, filed plainly: this panel
adds NO benchmark input — it consumes only
already-filed inputs (the four suites' filed
corpora through their real runners and the
shipped hardware-profile files, both already
carrying hygiene rows) — so the hygiene
inventory does NOT grow and its pinned
baseline is NOT re-pinned; the panel files no
new harness fixture for the same reason (its
benchmark entry binds the module in place,
fixture None).

Filed limits, stated plainly:

  * Token figures are budget exposures from
    filed budgets x scored units — not consumed
    tokens. No live model ran; live token
    consumption is NOT PROVEN (TF-173).
  * Time figures are this VM's scorer wall
    time, repeat-measured at run time and filed
    live-only: they vary run to run and under
    contention, they are pinned nowhere, and
    they measure scripted scorers, not model
    inference on any profile's hardware. Live
    per-profile hardware time is NOT PROVEN
    (TF-082 / TF-173).
  * Watts are NOT PROVEN and not measured: no
    profile declares a power figure and no
    power measurement exists in the estate, so
    sharpness per watt is NOT PROVEN — only
    sharpness per budgeted token is filed.
  * Suite scores do not vary by profile except
    through pi_zero's refused depth ON road
    (TF-128's measured shape), so sharpness
    differences across profiles are budget
    arithmetic over identical scores — filed as
    exactly that, never as hardware efficiency.

The human-readable write-up is
``docs/architecture/COST_PANEL.md``.
Proof grade: FIXTURE — scripted synthetic
suites through their real runners and the
shipped profiles' filed budgets on this VM;
no live model, no measured watt, never claimed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COST_PANEL_FORMAT = "thoughtforge-cost-panel/1"
COST_PANEL_DOC = "docs/architecture/COST_PANEL.md"

#: The new baseline this slice registers in the
#: harness — the panel's own pin. It is NOT one
#: of TF-129's gated baselines (the gate gates
#: the nine TF-121..TF-128 baselines, as filed;
#: this baseline is judged by this module's
#: checker, the shape TF-129 filed for its own
#: baseline).
COST_PANEL_BASELINE_ID = "cost_panel_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: How many times each cell's suite runner is
#: timed per panel run. Filed from the measured
#: shape: the scorers are millisecond-scale, so
#: five repeats per cell (120 runs) cost seconds
#: and expose the spread TF-121 filed (first
#: runs slowest) instead of hiding it in one
#: sample.
TIME_REPEATS = 5

#: The power status every row files on this
#: estate: no profile declares a power figure
#: and no power was measured, so no other
#: status may be filed (the checker enforces
#: it and a probe proves a claimed status
#: fails).
POWER_STATUS = "not_proven_not_measured"

# The real machinery, reused — never copied:
# the shipped profiles + TF-128's filed-budget
# reader, the harness's profile filing, the
# four suites' real runners, and TF-121 /
# D-BENCH-7's PerfTracker as the timing
# instrument.
from benchmarks.harness import KNOWN_PROFILES  # noqa: E402
from benchmarks.profile_panel import filed_budgets  # noqa: E402
from thoughtforge.utils.config import (  # noqa: E402
    load_hardware_profile,
)
from thoughtforge.utils.perf import PerfTracker  # noqa: E402

#: The panel's profiles: the shipped six, in the
#: harness's filed order (TF-128's estate).
PANEL_PROFILES: tuple[str, ...] = KNOWN_PROFILES

#: The panel's suites, in panel order (TF-128's
#: PANEL_SUITES: the four Part VII suites whose
#: runners accept corpora).
PANEL_SUITES: tuple[str, ...] = (
    "persona", "memory", "depth", "presence")

TOTAL_CELLS = len(PANEL_PROFILES) * len(PANEL_SUITES)


def _suite_runner(suite: str) -> Any:
    """One suite's real runner, bound in place."""
    if suite == "persona":
        from benchmarks.persona_suite import run_persona_suite

        return run_persona_suite
    if suite == "memory":
        from benchmarks.memory_suite import run_memory_suite

        return run_memory_suite
    if suite == "depth":
        from benchmarks.depth_suite import run_depth_suite

        return run_depth_suite
    if suite == "presence":
        from benchmarks.presence_suite import run_presence_suite

        return run_presence_suite
    raise ValueError(f"unknown suite {suite!r}")


def suite_result_facts(
    suite: str, result: Any = None,
) -> dict[str, Any]:
    """One suite's scored facts from its real
    runner: its unit count, its per-road scores,
    and its headline under no profile (the ON
    headline — profile refusals are applied by
    the cell builder, from the profile's filed
    refinement budget)."""
    result = _suite_runner(suite)() if result is None else result
    if suite == "depth":
        return {
            "units": result.total_cases,
            "off_score": result.overall_off,
            "on_score": result.overall_on,
        }
    units = result.total_turns if suite == "persona" \
        else result.total_cases
    return {"units": units, "score": result.overall_score}


# ── The pure cost parts (probes run here) ───────────────────────────────────

def sharpness_for(score: float, token_budget: int) -> float:
    """Score per 1000 budgeted response tokens,
    purely: 0.0 when there is no budget to
    divide by (never an invented infinity)."""
    if token_budget <= 0:
        return 0.0
    return score / token_budget * 1000


def build_cell(
    profile: str, suite: str, units: int,
    max_response_tokens: int, max_refinement_passes: int,
    score: float, status: str,
) -> dict[str, Any]:
    """One deterministic cost cell from its
    measured parts, purely: the token budget and
    the sharpness are computed, never filed by
    hand, and the power fields file this
    estate's only honest values."""
    token_budget = units * max_response_tokens
    return {
        "profile": profile,
        "suite": suite,
        "units": units,
        "max_response_tokens": max_response_tokens,
        "max_refinement_passes": max_refinement_passes,
        "token_budget": token_budget,
        "suite_score": score,
        "sharpness_per_1000_tokens": sharpness_for(
            score, token_budget),
        "status": status,
        "power_watts": None,
        "power_status": POWER_STATUS,
    }


def cell_problems(cell: dict[str, Any]) -> list[str]:
    """Judge ONE filed cell against its own
    parts, purely: its token budget must be its
    units x its filed per-response budget, its
    sharpness must be its score over that
    budget, its score must be a score, and its
    power fields must file this estate's honest
    values — a claimed watt or a claimed
    measured status is named. Pure."""
    problems: list[str] = []
    label = f"{cell.get('profile')}/{cell.get('suite')}"
    expected_budget = cell["units"] * cell["max_response_tokens"]
    if cell["token_budget"] != expected_budget:
        problems.append(
            f"{label}: filed token budget "
            f"{cell['token_budget']} != units x "
            f"max_response_tokens {expected_budget}")
    expected_sharpness = sharpness_for(
        cell["suite_score"], expected_budget)
    if abs(cell["sharpness_per_1000_tokens"]
           - expected_sharpness) > 1e-12:
        problems.append(
            f"{label}: filed sharpness "
            f"{cell['sharpness_per_1000_tokens']} != "
            f"score / token_budget x 1000 "
            f"{expected_sharpness}")
    if not 0.0 <= cell["suite_score"] <= 1.0:
        problems.append(
            f"{label}: suite score "
            f"{cell['suite_score']} is not a score "
            f"in [0, 1]")
    if cell["units"] <= 0:
        problems.append(
            f"{label}: units {cell['units']} — a "
            f"cell over no scored units files no cost")
    if cell.get("power_watts") is not None:
        problems.append(
            f"{label}: power_watts "
            f"{cell['power_watts']} claimed — no "
            f"profile declares a power figure and "
            f"no power was measured; watts are NOT "
            f"PROVEN on this estate, never filed")
    if cell.get("power_status") != POWER_STATUS:
        problems.append(
            f"{label}: power status "
            f"{cell.get('power_status')!r} != "
            f"{POWER_STATUS!r} — the only status "
            f"this estate may file")
    return problems


def timing_problems(timing: dict[str, Any]) -> list[str]:
    """Judge ONE cell's timing shape, purely —
    never its values: the repeats must be the
    filed count, every filed statistic positive,
    and the median inside [min, max]. A timing
    that fails shape proves nothing and is
    named. Pure."""
    problems: list[str] = []
    if timing.get("repeats") != TIME_REPEATS:
        problems.append(
            f"timing repeats {timing.get('repeats')} "
            f"!= the filed {TIME_REPEATS}")
    for key in ("mean_ms", "median_ms", "min_ms", "max_ms"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(
                f"timing {key} {value!r} is not a "
                f"positive measured wall time")
    if isinstance(timing.get("median_ms"), (int, float)) \
            and isinstance(timing.get("min_ms"), (int, float)) \
            and isinstance(timing.get("max_ms"), (int, float)) \
            and not timing["min_ms"] <= timing["median_ms"] \
            <= timing["max_ms"]:
        problems.append(
            "timing median is outside [min, max] — "
            "the summary does not describe its samples")
    return problems


# ── The panel measurement ───────────────────────────────────────────────────

def compute_cost_rows(
    profiles: Any = None, suites: Any = None,
) -> tuple[dict[str, Any], ...]:
    """The deterministic rows, measured live:
    each suite run once through its real runner,
    each profile's budgets read live from its
    JSON — the depth headline per profile is
    TF-128's rule (ON where a refinement pass is
    budgeted, the OFF score filed as
    refused_by_budget where none is)."""
    profiles = PANEL_PROFILES if profiles is None else profiles
    suites = PANEL_SUITES if suites is None else suites
    facts = {s: suite_result_facts(s) for s in suites}
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        budgets = filed_budgets(profile)
        for suite in suites:
            fact = facts[suite]
            if suite == "depth":
                if budgets["max_refinement_passes"] >= 1:
                    score, status = fact["on_score"], "measured"
                else:
                    score, status = fact["off_score"], \
                        "refused_by_budget"
            else:
                score, status = fact["score"], "measured"
            rows.append(build_cell(
                profile, suite, fact["units"],
                budgets["max_response_tokens"],
                budgets["max_refinement_passes"],
                score, status))
    return tuple(rows)


def measure_timing(
    profile: str, suite: str,
    tracker: PerfTracker | None = None,
) -> dict[str, Any]:
    """One cell's time, measured honestly: its
    suite runner timed TIME_REPEATS times with
    perf_counter on this VM, every sample
    recorded into the PerfTracker (D-BENCH-7's
    consumption), and the tracker's own summary
    filed. The values are this run's walls —
    live data, never pinned."""
    tracker = PerfTracker() if tracker is None else tracker
    runner = _suite_runner(suite)
    event = f"{profile}:{suite}"
    for _ in range(TIME_REPEATS):
        start = time.perf_counter()
        runner()
        tracker.record(
            event, duration_ms=(time.perf_counter() - start) * 1000)
    stats = tracker.summary(last_n=10000).events[event]
    return {
        "repeats": stats.count,
        "mean_ms": stats.mean,
        "median_ms": stats.p50,
        "min_ms": stats.min,
        "max_ms": stats.max,
    }


def finding_for(rows: Any) -> dict[str, Any]:
    """The panel's finding, computed from the
    rows: the sharpest and least sharp profiles
    (mean sharpness over their cells), whether
    scores vary beyond the refused cell, the
    refused cells, the panel's total budget, and
    the watt verdict — filed whatever it is."""
    def _mean_sharpness(profile: str) -> float | None:
        cells = [r for r in rows if r["profile"] == profile]
        if not cells:
            return None
        return sum(
            r["sharpness_per_1000_tokens"] for r in cells) / len(cells)

    per_profile = {
        p: _mean_sharpness(p) for p in PANEL_PROFILES
        if _mean_sharpness(p) is not None}
    refused = [
        [r["profile"], r["suite"],
         f"max_refinement_passes={r['max_refinement_passes']}"]
        for r in rows if r["status"] == "refused_by_budget"]
    return {
        "sharpest_profile": (
            max(per_profile, key=lambda p: per_profile[p])
            if per_profile else None),
        "least_sharp_profile": (
            min(per_profile, key=lambda p: per_profile[p])
            if per_profile else None),
        "suite_scores_vary_by_profile": bool(
            rows and any(
                len({r["suite_score"] for r in rows
                     if r["suite"] == s}) > 1
                for s in PANEL_SUITES)),
        "suite_scores_vary_without_the_refused_cell": bool(
            rows and any(
                len({r["suite_score"] for r in rows
                     if r["suite"] == s
                     and r["status"] == "measured"}) > 1
                for s in PANEL_SUITES)),
        "refused_cells": refused,
        "total_token_budget": sum(r["token_budget"] for r in rows),
        "total_units": sum(r["units"] for r in rows),
        "power_status_by_profile": {
            p: POWER_STATUS for p in PANEL_PROFILES},
        "declared_power_profiles": [],
        "sharpness_per_watt": "not_proven_not_measured",
    }


# ── The published panel (literal data, TF-067's CURVE precedent) ────────────
#
# Measured live on this VM at banking (2026-11-01):
# every suite through its real runner, every
# profile's budgets read from its JSON — twice,
# identical (the deterministic rows only; time
# is live-only and watts are NOT PROVEN, so
# neither is filed here). The checker recomputes
# every row live and names any drift.

PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {"profile": "desktop_cpu", "suite": "persona", "units": 108,
     "max_response_tokens": 400, "max_refinement_passes": 2,
     "token_budget": 43200, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.019568758573388204,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_cpu", "suite": "memory", "units": 36,
     "max_response_tokens": 400, "max_refinement_passes": 2,
     "token_budget": 14400, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.05594135802469137,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_cpu", "suite": "depth", "units": 36,
     "max_response_tokens": 400, "max_refinement_passes": 2,
     "token_budget": 14400, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.05594135802469137,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_cpu", "suite": "presence", "units": 36,
     "max_response_tokens": 400, "max_refinement_passes": 2,
     "token_budget": 14400, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.05401234567901235,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_gpu", "suite": "persona", "units": 108,
     "max_response_tokens": 600, "max_refinement_passes": 3,
     "token_budget": 64800, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.01304583904892547,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_gpu", "suite": "memory", "units": 36,
     "max_response_tokens": 600, "max_refinement_passes": 3,
     "token_budget": 21600, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.037294238683127576,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_gpu", "suite": "depth", "units": 36,
     "max_response_tokens": 600, "max_refinement_passes": 3,
     "token_budget": 21600, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.037294238683127576,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "desktop_gpu", "suite": "presence", "units": 36,
     "max_response_tokens": 600, "max_refinement_passes": 3,
     "token_budget": 21600, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.0360082304526749,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "phone_low", "suite": "persona", "units": 108,
     "max_response_tokens": 180, "max_refinement_passes": 1,
     "token_budget": 19440, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.0434861301630849,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "phone_low", "suite": "memory", "units": 36,
     "max_response_tokens": 180, "max_refinement_passes": 1,
     "token_budget": 6480, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.1243141289437586,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "phone_low", "suite": "depth", "units": 36,
     "max_response_tokens": 180, "max_refinement_passes": 1,
     "token_budget": 6480, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.1243141289437586,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "phone_low", "suite": "presence", "units": 36,
     "max_response_tokens": 180, "max_refinement_passes": 1,
     "token_budget": 6480, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.12002743484224966,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_5", "suite": "persona", "units": 108,
     "max_response_tokens": 250, "max_refinement_passes": 2,
     "token_budget": 27000, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.03131001371742113,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_5", "suite": "memory", "units": 36,
     "max_response_tokens": 250, "max_refinement_passes": 2,
     "token_budget": 9000, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.08950617283950618,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_5", "suite": "depth", "units": 36,
     "max_response_tokens": 250, "max_refinement_passes": 2,
     "token_budget": 9000, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.08950617283950618,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_5", "suite": "presence", "units": 36,
     "max_response_tokens": 250, "max_refinement_passes": 2,
     "token_budget": 9000, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.08641975308641975,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_zero", "suite": "persona", "units": 108,
     "max_response_tokens": 120, "max_refinement_passes": 0,
     "token_budget": 12960, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.06522919524462735,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_zero", "suite": "memory", "units": 36,
     "max_response_tokens": 120, "max_refinement_passes": 0,
     "token_budget": 4320, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.1864711934156379,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_zero", "suite": "depth", "units": 36,
     "max_response_tokens": 120, "max_refinement_passes": 0,
     "token_budget": 4320, "suite_score": 0.47222222222222227,
     "sharpness_per_1000_tokens": 0.10931069958847738,
     "status": "refused_by_budget", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "pi_zero", "suite": "presence", "units": 36,
     "max_response_tokens": 120, "max_refinement_passes": 0,
     "token_budget": 4320, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.1800411522633745,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "server_gpu", "suite": "persona", "units": 108,
     "max_response_tokens": 2000, "max_refinement_passes": 3,
     "token_budget": 216000, "suite_score": 0.8453703703703704,
     "sharpness_per_1000_tokens": 0.003913751714677641,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "server_gpu", "suite": "memory", "units": 36,
     "max_response_tokens": 2000, "max_refinement_passes": 3,
     "token_budget": 72000, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.011188271604938273,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "server_gpu", "suite": "depth", "units": 36,
     "max_response_tokens": 2000, "max_refinement_passes": 3,
     "token_budget": 72000, "suite_score": 0.8055555555555557,
     "sharpness_per_1000_tokens": 0.011188271604938273,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
    {"profile": "server_gpu", "suite": "presence", "units": 36,
     "max_response_tokens": 2000, "max_refinement_passes": 3,
     "token_budget": 72000, "suite_score": 0.7777777777777778,
     "sharpness_per_1000_tokens": 0.010802469135802469,
     "status": "measured", "power_watts": None,
     "power_status": "not_proven_not_measured"},
)

FINDING: dict[str, Any] = {
    "sharpest_profile": "pi_zero",
    "least_sharp_profile": "server_gpu",
    "suite_scores_vary_by_profile": True,
    "suite_scores_vary_without_the_refused_cell": False,
    "refused_cells": [
        ["pi_zero", "depth", "max_refinement_passes=0"]],
    "total_token_budget": 766800,
    "total_units": 1296,
    "power_status_by_profile": {
        "desktop_cpu": "not_proven_not_measured",
        "desktop_gpu": "not_proven_not_measured",
        "phone_low": "not_proven_not_measured",
        "pi_5": "not_proven_not_measured",
        "pi_zero": "not_proven_not_measured",
        "server_gpu": "not_proven_not_measured",
    },
    "declared_power_profiles": [],
    "sharpness_per_watt": "not_proven_not_measured",
}


# ── The filed probes ────────────────────────────────────────────────────────
#
# Perturbed copies of filed rows run against the
# pure checker parts. Real filings are never
# perturbed; a probe passes when the pure parts
# file exactly the expected shape.

def _probe_cell(**overrides: Any) -> dict[str, Any]:
    cell = dict(PANEL_ROWS[0]) if PANEL_ROWS else build_cell(
        "desktop_cpu", "persona", 108, 400, 2,
        0.8453703703703704, "measured")
    cell.update(overrides)
    return cell


PANEL_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_filed_cell_passes",
     "kind": "cell", "overrides": {},
     "expect_problem": False,
     "detail": "a filed cell whose budget, "
               "sharpness, score and power fields "
               "agree with its own parts files no "
               "problem"},
    {"id": "wrong_token_budget_is_named",
     "kind": "cell",
     "overrides": {"token_budget": 999999},
     "expect_problem": True,
     "detail": "a token budget that is not units "
               "x the profile's filed per-response "
               "budget is named"},
    {"id": "wrong_sharpness_is_named",
     "kind": "cell",
     "overrides": {"sharpness_per_1000_tokens": 9.99},
     "expect_problem": True,
     "detail": "a sharpness that is not the "
               "cell's score over its token budget "
               "is named"},
    {"id": "invented_watts_are_named",
     "kind": "cell", "overrides": {"power_watts": 5.0},
     "expect_problem": True,
     "detail": "a watt no profile declares and no "
               "measurement took is an invention, "
               "named — sharpness per watt stays "
               "NOT PROVEN here"},
    {"id": "claimed_measured_power_is_named",
     "kind": "cell", "overrides": {"power_status": "measured"},
     "expect_problem": True,
     "detail": "a power status other than "
               "not_proven_not_measured claims a "
               "measurement this estate never took"},
    {"id": "out_of_range_score_is_named",
     "kind": "cell", "overrides": {"suite_score": 1.5},
     "expect_problem": True,
     "detail": "a suite score outside [0, 1] is "
               "not a score and its sharpness "
               "cannot stand"},
    {"id": "bad_timing_shape_is_named",
     "kind": "timing",
     "timing": {"repeats": 1, "mean_ms": 1.0,
                "median_ms": 0.0, "min_ms": 1.0,
                "max_ms": 1.0},
     "expect_problem": True,
     "detail": "a timing with the wrong repeat "
               "count and a non-positive median "
               "files no measured wall time and "
               "is named by shape"},
)


def run_panel_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    checker parts; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in PANEL_PROBES:
        if probe["kind"] == "timing":
            problems = timing_problems(probe["timing"])
        else:
            problems = cell_problems(
                _probe_cell(**probe["overrides"]))
        passed = bool(problems) == probe["expect_problem"]
        results.append({
            "id": probe["id"], "passed": passed,
            "problems": problems,
            "detail": probe["detail"]})
    return results


# ── The panel run ───────────────────────────────────────────────────────────

@dataclass
class CostPanelResult:
    """One full panel run: the deterministic
    rows, each cell's live timing, the finding,
    and the verdict data. Timing values are this
    run's walls only — the verdict judges their
    shape, never their values."""

    cells: list[dict[str, Any]] = field(default_factory=list)
    finding: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def total_token_budget(self) -> int:
        return sum(c["token_budget"] for c in self.cells)

    @property
    def overall_score(self) -> float:
        if not self.cells:
            return 0.0
        return sum(c["suite_score"] for c in self.cells) \
            / len(self.cells)

    @property
    def mean_sharpness(self) -> float:
        if not self.cells:
            return 0.0
        return sum(
            c["sharpness_per_1000_tokens"] for c in self.cells) \
            / len(self.cells)

    @property
    def rows_reproduce(self) -> bool:
        filed = [dict(r) for r in PANEL_ROWS]
        live = [{k: c[k] for k in PANEL_ROWS[0]} for c in self.cells] \
            if self.cells and PANEL_ROWS else []
        return live == filed

    @property
    def timings_well_shaped(self) -> bool:
        return bool(self.cells) and all(
            not timing_problems(c["timing"]) for c in self.cells)

    @property
    def verdict(self) -> str:
        """PASS only on the full filed matrix
        reproducing the filed rows with every
        timing well-shaped; REVIEW for any
        partial or empty selection, however it
        scores."""
        if [f"{c['profile']}:{c['suite']}" for c in self.cells] != [
                f"{r['profile']}:{r['suite']}" for r in PANEL_ROWS]:
            return "REVIEW"
        return "PASS" if (
            self.rows_reproduce and self.timings_well_shaped
        ) else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_cost_panel(
    profiles: Any = None, suites: Any = None,
) -> CostPanelResult:
    """Run the whole panel (filed matrix default):
    deterministic rows recomputed live, every
    cell timed live through the PerfTracker.

    Any explicit profile / suite selection may
    be given — including an empty one, which
    files 0 cells / REVIEW (no vacuous pass).
    """
    profiles = PANEL_PROFILES if profiles is None else tuple(profiles)
    suites = PANEL_SUITES if suites is None else tuple(suites)
    if not profiles or not suites:
        return CostPanelResult(cells=[], finding=finding_for([]))
    rows = compute_cost_rows(profiles, suites)
    tracker = PerfTracker()
    cells: list[dict[str, Any]] = []
    for row in rows:
        timing = measure_timing(row["profile"], row["suite"], tracker)
        cells.append({**row, "timing": timing})
    return CostPanelResult(
        cells=cells, finding=finding_for(rows))


# ── The readable render ─────────────────────────────────────────────────────

def render_cost_panel_markdown(
    rows: Any = None,
) -> str:
    """Render the deterministic panel as readable
    markdown — the table filed in COST_PANEL.md:
    per profile, each suite's units, filed
    budget, token budget, score and sharpness.
    Time and watts are not rendered: time is
    live-only and watts are NOT PROVEN."""
    rows = PANEL_ROWS if rows is None else rows
    lines: list[str] = []
    lines.append("## The filed rows (deterministic: tokens + sharpness)")
    lines.append("")
    lines.append(
        f"{TOTAL_CELLS} cells: {len(PANEL_SUITES)} suites x "
        f"{len(PANEL_PROFILES)} shipped profiles. Token budget "
        f"= scored units x the profile's filed "
        f"max_response_tokens (a budget exposure, not "
        f"consumed tokens); sharpness = suite score / "
        f"token budget x 1000. Power files "
        f"{POWER_STATUS} on every row (no watt filed, "
        f"none invented).")
    lines.append("")
    lines.append(
        "| Profile | Suite | Units | Max response tokens | "
        "Token budget | Suite score | Sharpness / 1000 tokens | "
        "Status |")
    lines.append(
        "|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['suite']} | "
            f"{row['units']} | {row['max_response_tokens']} | "
            f"{row['token_budget']} | {row['suite_score']} | "
            f"{row['sharpness_per_1000_tokens']} | "
            f"{row['status']} |")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── The filing's rules ──────────────────────────────────────────────────────

COST_PANEL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "tokens_are_budgets_from_real_runners_and_profiles",
     "rule": "every token figure is scored units "
             "(counted by the suite's real runner) "
             "x the profile's max_response_tokens "
             "(read live from its JSON through "
             "TF-128's filed_budgets) — a budget "
             "exposure per scored response, never "
             "a consumed-token count: no live "
             "model ran and none is claimed"},
    {"id": "sharpness_derives_only_from_real_suite_scores",
     "rule": "sharpness_per_1000_tokens is the "
             "cell's real suite score (through "
             "its real runner, TF-128's depth "
             "refusal carried) over its token "
             "budget x 1000 — no other numerator "
             "or denominator is ever filed"},
    {"id": "time_is_repeat_measured_live_and_never_pinned",
     "rule": "each cell's suite runner is timed "
             "TIME_REPEATS (5) times with "
             "perf_counter and filed as the "
             "PerfTracker's summary (mean / "
             "median / min / max ms) at run time "
             "only: this VM's scorer walls, "
             "varying under contention, pinned in "
             "no baseline and filed in no row — "
             "the checker judges the timing's "
             "shape, never its values (TF-121's "
             "latency rule, carried)"},
    {"id": "the_perf_tracker_is_consumed_here",
     "rule": "every timing sample is recorded "
             "into thoughtforge.utils.perf's "
             "PerfTracker and read back from its "
             "summary — D-BENCH-7's consumption "
             "share, fixed here in the open: the "
             "tracker is no longer wired into "
             "the core and consumed by nothing"},
    {"id": "watts_are_not_proven_and_never_invented",
     "rule": "no shipped profile declares a "
             "power figure (the checker re-scans "
             "the profile JSONs live for power / "
             "watt / tdp keys) and no power was "
             "measured, so every row files "
             "power_watts None + "
             "not_proven_not_measured, the pure "
             "checker fails any claimed watt or "
             "measured status (probe-proven), and "
             "sharpness per watt files NOT "
             "PROVEN — a future declared figure "
             "may be bound live and labelled "
             "declared, never measured"},
    {"id": "panel_rows_are_literal_and_recomputed",
     "rule": "PANEL_ROWS / FINDING are filed as "
             "literal deterministic data measured "
             "at banking and recomputed live by "
             "the checker — a row that moves is "
             "a finding, never silently re-filed "
             "(TF-067's CURVE precedent); time "
             "and watts are excluded from the "
             "literal rows by the rules above"},
    {"id": "pass_needs_the_full_matrix_reproducing",
     "rule": "the verdict is PASS only when all "
             "24 filed cells (4 suites x 6 "
             "profiles) recompute, every "
             "deterministic row reproduces "
             "PANEL_ROWS and every cell carries "
             "a well-shaped timing; a partial "
             "selection files REVIEW however it "
             "scores and an empty selection "
             "files 0 cells / REVIEW"},
    {"id": "probes_prove_the_checker_can_fail",
     "rule": "PANEL_PROBES files perturbed "
             "copies of filed rows against the "
             "pure checker parts (a wrong token "
             "budget, a wrong sharpness, an "
             "invented watt, a claimed measured "
             "power status, an out-of-range "
             "score, a bad timing shape) plus "
             "the clean row — real filings are "
             "never perturbed"},
    {"id": "no_new_benchmark_input_was_added",
     "rule": "the panel consumes only "
             "already-filed inputs — the four "
             "suites' filed corpora through "
             "their real runners and the "
             "shipped hardware-profile files, "
             "both already carrying TF-131 "
             "hygiene rows — so it files no new "
             "harness fixture and no hygiene "
             "row, and the hygiene inventory "
             "and its pinned baseline do not "
             "move (filed in the open, TF-131 "
             "coupling)"},
    {"id": "the_cost_panel_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates "
             "the nine TF-121..TF-128 pinned "
             "baselines exactly as filed; this "
             "slice's own baseline "
             "(cost_panel_synthetic_v1) is "
             "judged by this module's checker, "
             "not gated by the gate — the shape "
             "TF-129..TF-132 filed for their "
             "own baselines, with the exclusion "
             "filed in the open in "
             "regression_gate.py"},
)


# ── The live checker ────────────────────────────────────────────────────────

def _profile_power_keys() -> list[str]:
    """Every power-shaped key in the shipped
    profile JSONs, found live: on this estate
    the list is empty, which is exactly why no
    watt may be filed."""
    found: list[str] = []

    def _walk(profile: str, node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if any(shape in key.lower()
                       for shape in ("power", "watt", "tdp")):
                    found.append(f"{profile}:{here}")
                _walk(profile, value, here)
        elif isinstance(node, list):
            for item in node:
                _walk(profile, item, path)

    for profile in PANEL_PROFILES:
        _walk(profile, load_hardware_profile(profile), "")
    return found


def cost_panel_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: profiles, suites, cells, rules.
    if tuple(PANEL_PROFILES) != tuple(KNOWN_PROFILES) \
            or len(PANEL_PROFILES) != 6:
        problems.append(
            f"PANEL_PROFILES drifted from the shipped six: "
            f"{PANEL_PROFILES}")
    if list(PANEL_SUITES) != [
            "persona", "memory", "depth", "presence"]:
        problems.append(f"PANEL_SUITES drifted: {list(PANEL_SUITES)}")
    if TOTAL_CELLS != 24:
        problems.append(f"TOTAL_CELLS is {TOTAL_CELLS}, not 24")
    if TIME_REPEATS != 5:
        problems.append(f"TIME_REPEATS is {TIME_REPEATS}, not 5")
    rule_ids = [r["id"] for r in COST_PANEL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate COST_PANEL_RULES ids")
    for needed in (
            "tokens_are_budgets_from_real_runners_and_profiles",
            "sharpness_derives_only_from_real_suite_scores",
            "time_is_repeat_measured_live_and_never_pinned",
            "the_perf_tracker_is_consumed_here",
            "watts_are_not_proven_and_never_invented",
            "panel_rows_are_literal_and_recomputed",
            "pass_needs_the_full_matrix_reproducing",
            "probes_prove_the_checker_can_fail",
            "no_new_benchmark_input_was_added",
            "the_cost_panel_baseline_is_not_gated_by_tf129"):
        if needed not in rule_ids:
            problems.append(f"COST_PANEL_RULES lost {needed}")

    # Watts: the estate declares none, live.
    power_keys = _profile_power_keys()
    if power_keys:
        problems.append(
            f"shipped profiles now declare power-shaped "
            f"keys {power_keys} — a watt may be filed "
            f"declared (never measured) only by a "
            f"deliberate re-filing, not silently")

    # The filed rows: shape, budgets from the
    # JSONs live, pure checker clean per cell.
    cell_ids = [f"{r['profile']}:{r['suite']}" for r in PANEL_ROWS]
    if len(cell_ids) != len(set(cell_ids)) \
            or len(PANEL_ROWS) != TOTAL_CELLS:
        problems.append(
            f"the filed rows file {len(PANEL_ROWS)} cells, "
            f"not {TOTAL_CELLS} unique cells")
    for row in PANEL_ROWS:
        problems.extend(cell_problems(row))
        budgets = filed_budgets(row["profile"])
        if row["max_response_tokens"] != budgets[
                "max_response_tokens"] \
                or row["max_refinement_passes"] != budgets[
                    "max_refinement_passes"]:
            problems.append(
                f"{row['profile']}/{row['suite']}: filed "
                f"budgets do not match the profile JSON's "
                f"live budgets")
    refused = [f"{r['profile']}/{r['suite']}" for r in PANEL_ROWS
               if r["status"] == "refused_by_budget"]
    if refused != ["pi_zero/depth"]:
        problems.append(
            f"refused-by-budget cells {refused} != "
            f"['pi_zero/depth'] (TF-128's refusal, carried)")

    # The published panel: literal rows recomputed
    # live, twice — filing, recomputation and
    # determinism in one pass.
    live_rows = compute_cost_rows()
    if [dict(r) for r in live_rows] != [dict(r) for r in PANEL_ROWS]:
        problems.append(
            "PANEL_ROWS drifted from the live recomputation")
    if finding_for(live_rows) != FINDING:
        problems.append("FINDING drifted from the filed rows")
    second_rows = compute_cost_rows()
    if second_rows != live_rows:
        problems.append(
            "the deterministic rows are not deterministic: "
            "two live computations differ")

    # A fresh full run files PASS with well-shaped
    # timings; partial + empty selections REVIEW.
    result = run_cost_panel()
    if result.total_cells != TOTAL_CELLS \
            or result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live panel run does not file {TOTAL_CELLS} "
            f"cells / PASS: {result.total_cells} / "
            f"{result.verdict}")
    if result.total_token_budget != FINDING.get(
            "total_token_budget"):
        problems.append(
            "the live run's total token budget disagrees "
            "with the filed finding")
    empty = run_cost_panel(profiles=())
    if empty.total_cells != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    subset = run_cost_panel(profiles=("desktop_cpu",))
    if subset.total_cells != len(PANEL_SUITES) \
            or subset.verdict != "REVIEW":
        problems.append(
            "a single-profile selection does not measure "
            "its cells / REVIEW mechanically")

    # The probes all behave.
    probes = run_panel_probes()
    if len(probes) != 7 or not all(p["passed"] for p in probes):
        problems.append(
            f"the panel probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the rendered deterministic panel.
    doc = _REPO_ROOT / COST_PANEL_DOC
    if not doc.exists():
        problems.append(f"the write-up {COST_PANEL_DOC} is missing")
    else:
        doc_text = doc.read_text(encoding="utf-8")
        if render_cost_panel_markdown().strip() not in doc_text:
            problems.append(
                f"the doc {COST_PANEL_DOC} does not file "
                f"the rendered panel verbatim")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included), and the gate's gated set untouched.
    from benchmarks.harness import (
        BASELINES,
        BENCHMARKS,
        baseline_problems,
    )
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "cost_panel" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the cost_panel entry")
    if COST_PANEL_BASELINE_ID not in [b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the cost-panel baseline")
    if COST_PANEL_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the cost-panel baseline entered TF-129's "
            "gated set (it is judged by this checker, "
            "not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the cost "
            f"panel: {earlier}")
    return problems


def cost_panel_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_cost_panel()
    probes = run_panel_probes()
    return {
        "format": COST_PANEL_FORMAT,
        "doc": COST_PANEL_DOC,
        "profiles": list(PANEL_PROFILES),
        "suites": list(PANEL_SUITES),
        "rows": [dict(r) for r in PANEL_ROWS],
        "finding": dict(FINDING),
        "measured": {
            "total_cells": result.total_cells,
            "total_token_budget": result.total_token_budget,
            "mean_sharpness_per_1000_tokens":
                result.mean_sharpness,
            "overall_score": result.overall_score,
            "timings": {
                f"{c['profile']}:{c['suite']}": c["timing"]
                for c in result.cells},
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
            "verdict": result.verdict,
        },
        "probes": probes,
        "rules": [dict(r) for r in COST_PANEL_RULES],
    }


__all__ = [
    "COST_PANEL_BASELINE_ID",
    "COST_PANEL_DOC",
    "COST_PANEL_FORMAT",
    "COST_PANEL_RULES",
    "FINDING",
    "PANEL_PROBES",
    "PANEL_PROFILES",
    "PANEL_ROWS",
    "PANEL_SUITES",
    "POWER_STATUS",
    "TIME_REPEATS",
    "TOTAL_CELLS",
    "CostPanelResult",
    "build_cell",
    "cell_problems",
    "compute_cost_rows",
    "cost_panel_as_data",
    "cost_panel_problems",
    "finding_for",
    "measure_timing",
    "render_cost_panel_markdown",
    "run_cost_panel",
    "run_panel_probes",
    "sharpness_for",
    "suite_result_facts",
    "timing_problems",
]
