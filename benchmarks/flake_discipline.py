"""
Flake discipline, as data (TF-135).

The roadmap's rule: *flake discipline:
nondeterministic suites quarantined or made
deterministic (TF-036) — no silent re-rolls.*
TF-129 measured the baseline noise of the nine
gated baselines (five in-process repeats, every
spread 0.0) and derived its gate tolerances
from that measurement — but the discipline
stopped at the gate's edge: no census ever
classified EVERY pinned suite deterministic or
not, no quarantine registry existed for a suite
that did move, no flake-report shape existed
that a re-roll had to file, and nothing owned
the estate's one documented re-roll: TF-126's
full gate filed 4 transient EPERM write-block
failures in its tf024 / tf030 legs that passed
on immediate re-run — recorded in campaign
prose only, settled by the green re-run alone.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and no flake
    discipline existed.** ``harness_problems()``
    returned [] and ``baseline_problems()``
    returned [] over the fifteen pinned
    baselines, ``regression_gate_problems()``
    returned [] with the live gate PASS
    (9 baselines / 83 metrics / 0 failures),
    ``baseline_book_problems()`` returned [],
    ``data_hygiene_problems()`` returned []
    (21 inputs, all clean) and
    ``cost_panel_problems()`` returned [] — and
    a grep across benchmarks / docs / tests /
    src found no flake census, no quarantine
    registry and no flake-report shape anywhere:
    TF-129's noise measurement over the gated
    nine was the whole story.
  * **TF-129's noise was gated-only by design.**
    Its own filed limits state that
    cross-process noise is not measured there
    and assign that share here: "TF-135's flake
    discipline owns that share". The six ungated
    pinned baselines (the gate's own row, spot
    checks, hygiene, the WYRD panel, the cost
    panel, the baseline book) had never been
    repeat-measured at all.
  * **The gap was never numbered.** The number
    **D-BENCH-20** was verified unused anywhere
    in benchmarks / docs / tests / src before
    this slice filed it (the D-BENCH-9..19
    pattern: a gap assigned by the roadmap
    without ever being numbered).

What this module files, as verifiable data:

  * ``FLAKE_RULES`` — the discipline itself,
    filed as data: every pinned suite is
    census-measured by repetition; a suite is
    deterministic only if all repeats reproduce
    its pinned metrics exactly (TF-036's
    determinism mode is the cited reason exact
    reproduction is claimable at all);
    nondeterminism is either FIXED (made
    deterministic, citing TF-036 where
    relevant) or QUARANTINED with a filed
    reason — never silently tolerated; **no
    silent re-rolls**: any re-run after a
    differing result must be filed as a flake
    report naming the suite, the metric / leg,
    the differing values and the disposition —
    a green re-run alone settles nothing; and
    timing / wall metrics stay unpinned per
    TF-121's rule and are classified
    excluded-by-rule, not flakes.
  * ``CENSUS_ROWS`` — the census, published as
    literal data measured live on this VM at
    banking (TF-067's CURVE precedent): every
    pinned baseline in the harness re-measured
    ``CENSUS_REPEATS`` (5, TF-129's filed
    ``NOISE_REPEATS``) times in one process
    through the harness's own measurers (no
    corpus or scorer copied), each row filing
    its repeats, its metric count, its repeat
    spreads, its differing metrics (any metric
    whose repeats did not all reproduce the pin
    exactly) and its classification. **The
    measured truth on this tree: every censused
    baseline is deterministic** — no repeat of
    any pinned metric differed from its pin, so
    ``QUARANTINE_REGISTRY`` files empty and no
    suite is quarantined. Any future differing
    baseline files ``nondeterministic_unfiled``
    in a live census until a slice either fixes
    it (TF-036's road) or files its quarantine
    reason here — the checker names it either
    way, never smooths it.
  * The census of this slice's own baseline:
    ``flake_discipline_synthetic_v1`` is
    censused like every other pin — its harness
    measurer is literal-based over this filing
    (the shape TF-129's gate measurer files for
    its noise summary), so censusing it cannot
    recurse into the census itself.
  * ``excluded_metric_rows()`` — the metrics
    TF-121's rule keeps unpinned (latency /
    wall / power shapes filed in baselines'
    ``excluded_metrics``), built live from the
    harness registry: each files
    excluded-by-rule with its filed reason —
    classified, never counted as flakes.
  * **The flake-report shape** —
    ``FLAKE_REPORTS`` + ``flake_report_problems()``:
    a report is valid only with an id, a suite,
    a metric / leg, its differing values and a
    disposition, each missing part named. One
    report is filed: TF-126's EPERM re-roll,
    filed retroactively by this slice so the
    re-roll is no longer silent (its differing
    values and its disposition — an environmental
    sandbox write-block, not a suite
    nondeterminism; no suite quarantined).
    ``RERUN_RECORDS`` + ``rerun_record_problems()``
    file the companion rule: a re-run recorded
    after a differing first outcome with no
    flake report is a silent re-roll and is
    named; TF-126's record links its report.
  * ``FLAKE_PROBES`` — filed probes against the
    pure validators with perturbed copies of
    filed rows / reports / records (real filings
    never perturbed): the clean census row
    passes; a census-perturbed suite (differing
    metrics filed as deterministic) is named; a
    quarantine without a reason is named; a
    flake report missing its differing values is
    named; a flake report missing its
    disposition is named; a silent re-roll
    record is named; the filed report + its
    linked re-run record pass.
  * ``run_flake_discipline()`` files the
    verdict: PASS only on the FULL census (every
    pinned baseline) with every suite classified
    deterministic or reasoned-quarantined and
    every filed report / record valid; a full
    census carrying an unfiled nondeterministic
    suite files FAIL; a partial selection files
    REVIEW however it scores; an empty selection
    files 0 baselines / REVIEW — no vacuous pass.
  * The readable filing IS the render:
    ``render_flake_discipline_markdown()``
    renders the census, the excluded rows, the
    quarantine registry, the filed reports and
    records and the rules deterministically from
    this module's data, and
    ``docs/architecture/FLAKE_DISCIPLINE.md``
    files that render verbatim — the checker
    requires the doc to reproduce the render
    exactly (the baseline-book shape), and
    ``python3 -m benchmarks.flake_discipline``
    regenerates it.

D-BENCH-20 is FIXED by this slice: the pinned
estate now has a flake discipline — a repetition
census classifying every pinned suite, a
quarantine registry with reasons, a validated
flake-report shape, and a no-silent-re-rolls
rule with TF-126's re-roll filed under it. What
is NOT fixed, and is stated in the filed limits:
the census measures this VM, in one process per
run, over the harness's pinned fixture baselines
only; cross-machine flakiness and live-model
flakiness are NOT PROVEN (TF-173), never claimed.
This slice's own harness baseline is judged by
this module's checker, NOT gated by TF-129's
regression gate (the gate's gated set stays the
nine TF-121..TF-128 baselines exactly as filed;
the exclusion is filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129..TF-134 filed for their own baselines).
TF-131 coupling, filed plainly: this discipline
adds NO benchmark input — it re-measures only
already-filed pins through their already-filed
measurers — so the hygiene inventory does NOT
grow and its pinned baseline is NOT re-pinned;
the discipline files no new harness fixture for
the same reason (its benchmark entry binds the
module in place, fixture None — the TF-133 /
TF-134 shape).

Filed limits, stated plainly:

  * The census judges the harness's pinned
    fixture baselines only — synthetic fixtures
    through the real scorers (TF-017). No live
    model is censused; live-model flakiness is
    NOT PROVEN (TF-173).
  * The census runs in one process on this VM
    (five repeats, in-process, through the
    harness measurers). Cross-process repetition
    is exercised only as separate checker / test
    runs of this same census, which must agree;
    cross-machine noise is not measured here.
  * A quarantined suite stays pinned and
    censused: quarantine files the reason and
    the differing metrics, it never deletes a
    baseline, loosens a tolerance or hides a
    re-roll — and no suite is quarantined on
    this tree, because none differed.
  * The discipline is a checker a slice runs
    in-house; wiring it into any external CI is
    NOT done and NOT claimed (the house runs
    manual-only workflows and in-house
    verification).

The human-readable write-up is
``docs/architecture/FLAKE_DISCIPLINE.md``,
filed verbatim as this module's render.
Proof grade: FIXTURE — the harness's pinned
synthetic-fixture baselines re-measured live
on this VM; no live model, never claimed.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FLAKE_DISCIPLINE_FORMAT = "thoughtforge-flake-discipline/1"
FLAKE_DISCIPLINE_DOC = "docs/architecture/FLAKE_DISCIPLINE.md"

#: The new baseline this slice registers in the
#: harness — the discipline's own pin. It is NOT
#: one of TF-129's gated baselines (the gate gates
#: the nine TF-121..TF-128 baselines, as filed;
#: this baseline is judged by this module's
#: checker, the shape TF-129 filed for its own
#: baseline).
FLAKE_DISCIPLINE_BASELINE_ID = "flake_discipline_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: How many times the census re-measures every
#: pinned baseline in one process: TF-129's filed
#: NOISE_REPEATS, bound live by the checker (the
#: census is TF-129's noise road widened to every
#: pin, at the same repetition count).
CENSUS_REPEATS = 5

#: The determinism this discipline cites: TF-036's
#: determinism mode (seeded generation, turn
#: fingerprints, reproducibility_problems) is the
#: reason exact reproduction may be claimed at
#: all — the harness files it as the
#: ``determinism`` neighbour entry, bound live by
#: the checker.
DETERMINISM_CITATION = (
    "TF-036 determinism mode "
    "(thoughtforge.cognition.determinism: seeded "
    "generation + turn fingerprints; the harness's "
    "determinism neighbour entry)")

# The real machinery, reused — never copied: the
# harness's pinned baselines and its own
# measurers, and TF-129's filed repeat count.
# This module imports the harness at top level;
# the harness binds this module only lazily
# (inside its measure function), so no import
# cycle exists.
from benchmarks import harness as _harness  # noqa: E402
from benchmarks.regression_gate import (  # noqa: E402
    NOISE_REPEATS as _GATE_NOISE_REPEATS,
)


# ── The discipline's rules ──────────────────────────────────────────────────

FLAKE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_pinned_suite_is_census_measured_by_repetition",
     "rule": "every pinned baseline in the harness "
             "is re-measured CENSUS_REPEATS (5 — "
             "TF-129's filed NOISE_REPEATS) times in "
             "one process through the harness's own "
             "measurers, and the census files per "
             "baseline its repeats, its metric count, "
             "its repeat spreads and its differing "
             "metrics — no suite's determinism is "
             "assumed from another suite's noise"},
    {"id": "deterministic_means_every_repeat_reproduces_the_pin_exactly",
     "rule": "a suite is deterministic only if "
             "every repeat reproduces its pinned "
             "metrics exactly — the standard "
             "TF-036's determinism mode makes "
             "claimable (seeded generation, turn "
             "fingerprints); any metric whose "
             "repeats differ from the pin or from "
             "each other is a differing metric, "
             "named in the census row"},
    {"id": "nondeterminism_is_fixed_or_quarantined_never_tolerated",
     "rule": "a differing suite is either FIXED "
             "(made deterministic, citing TF-036's "
             "determinism mode where relevant) or "
             "QUARANTINED in QUARANTINE_REGISTRY "
             "with a filed reason — never silently "
             "tolerated, never re-pinned to whatever "
             "it happened to print, and never hidden "
             "behind a re-run"},
    {"id": "no_silent_rerolls",
     "rule": "any re-run after a differing result "
             "must be filed as a flake report naming "
             "the suite, the metric / leg, the "
             "differing values and the disposition — "
             "a green re-run alone settles nothing; "
             "a re-run recorded with no report is a "
             "silent re-roll and the checker names "
             "it"},
    {"id": "timing_and_wall_metrics_are_excluded_by_rule_not_flakes",
     "rule": "timing / wall metrics (and the other "
             "shapes filed in a baseline's "
             "excluded_metrics) stay unpinned per "
             "TF-121's rule — walls under contention "
             "are not baselines — and the census "
             "classifies them excluded-by-rule with "
             "their filed reasons, never as flakes "
             "and never as determinism evidence"},
    {"id": "quarantine_needs_a_filed_reason_and_keeps_the_pin",
     "rule": "a quarantine entry names its baseline "
             "and its reason beside the census row's "
             "differing metrics; a quarantine without "
             "a reason is named by the checker, and "
             "quarantine never deletes a baseline, "
             "loosens a tolerance or stops the census "
             "from re-measuring the suite"},
    {"id": "flake_reports_are_validated_as_data",
     "rule": "FLAKE_REPORTS are judged by "
             "flake_report_problems(): a report is "
             "valid only with an id, a suite, a "
             "metric / leg, its differing values and "
             "a disposition, each missing part named; "
             "RERUN_RECORDS are judged by "
             "rerun_record_problems(): a record's "
             "report id must resolve to a filed "
             "report"},
    {"id": "probes_prove_the_checker_can_fail",
     "rule": "FLAKE_PROBES perturbs copies of filed "
             "rows / reports / records against the "
             "pure validators (a census-perturbed "
             "suite, a quarantine without a reason, "
             "a report missing its differing values, "
             "a report missing its disposition, a "
             "silent re-roll record) plus the clean "
             "shapes — real filings are never "
             "perturbed"},
    {"id": "pass_needs_the_full_census_classified",
     "rule": "the verdict is PASS only on the full "
             "census with every suite classified "
             "deterministic or reasoned-quarantined "
             "and every filed report / record valid; "
             "a full census with an unfiled "
             "nondeterministic suite files FAIL, a "
             "partial selection files REVIEW however "
             "it scores, and an empty selection "
             "files 0 baselines / REVIEW"},
    {"id": "the_flake_discipline_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates the "
             "nine TF-121..TF-128 pinned baselines "
             "exactly as filed; this slice's own "
             "baseline (flake_discipline_synthetic_v1) "
             "is judged by this module's checker, "
             "not gated by the gate — the shape "
             "TF-129..TF-134 filed for their own "
             "baselines, with the exclusion filed "
             "in the open in regression_gate.py"},
)


# ── The quarantine registry ─────────────────────────────────────────────────
#
# Filed as data: one entry per quarantined
# baseline, each with its reason. On this tree
# the registry is EMPTY — the census measured
# every pinned baseline deterministic, so there
# is nothing to quarantine. An entry here is the
# only road by which a differing baseline may
# file quarantined in the census.

QUARANTINE_REGISTRY: tuple[dict[str, Any], ...] = ()


def quarantine_reason_for(baseline_id: str) -> str | None:
    """One baseline's filed quarantine reason, or
    None when it is not quarantined."""
    for entry in QUARANTINE_REGISTRY:
        if entry.get("baseline") == baseline_id:
            return entry.get("reason")
    return None


# ── The filed flake reports + re-run records ────────────────────────────────
#
# TF-126's re-roll, filed retroactively by this
# slice so it is no longer silent: the full gate
# filed 4 transient EPERM write-block failures in
# its tf024 / tf030 legs, which passed on
# immediate re-run (the campaign record filed the
# shape in prose only). The disposition stands as
# measured then: an environmental sandbox
# write-block, not a suite nondeterminism — so no
# suite is quarantined and no baseline moved.

FLAKE_REPORTS: tuple[dict[str, Any], ...] = (
    {"id": "tf126_full_gate_eperm_reroll",
     "suite": "full_gate (tests/test_tf024 + "
              "tests/test_tf030 legs)",
     "leg": "full-suite gate run filed in "
            "CAMPAIGN_STATE at TF-126 banking",
     "differing_values": [
         "first full-gate run: 4 failures — "
         "transient EPERM write-block errors in "
         "the tf024 / tf030 legs",
         "immediate re-run: the same legs passed "
         "55/55 and the gate filed clean",
     ],
     "disposition": "environmental sandbox "
                    "write-block, not a suite "
                    "nondeterminism: no suite "
                    "quarantined, no baseline "
                    "re-pinned; filed retroactively "
                    "by TF-135 so the re-roll is "
                    "owned by this discipline and "
                    "never again settles by the "
                    "green re-run alone"},
)

RERUN_RECORDS: tuple[dict[str, Any], ...] = (
    {"suite": "full_gate (tests/test_tf024 + "
              "tests/test_tf030 legs)",
     "first_outcome": "4 failures (transient EPERM "
                      "write-block)",
     "rerun_outcome": "passed (55/55 on the "
                      "affected legs)",
     "flake_report_id": "tf126_full_gate_eperm_reroll"},
)


# ── The pure validators (probes run here) ───────────────────────────────────

def census_row_problems(
    row: dict[str, Any], registry: Any = None,
) -> list[str]:
    """Judge ONE census row against its own parts,
    purely: its repeats must be the filed count,
    its metric count must match its spreads, a
    deterministic row files no differing metric,
    and a quarantined row files differing metrics
    plus a reason that the registry carries. Any
    other classification is an unfiled
    nondeterminism, named. Pure — an empty or
    partial row is named, never crashed on."""
    registry = QUARANTINE_REGISTRY if registry is None \
        else registry
    problems: list[str] = []
    label = row.get("baseline", "<unnamed>")
    if row.get("repeats") != CENSUS_REPEATS:
        problems.append(
            f"{label}: census repeats "
            f"{row.get('repeats')} != the filed "
            f"{CENSUS_REPEATS}")
    spreads = row.get("spreads") or {}
    if row.get("metrics_measured") != len(spreads):
        problems.append(
            f"{label}: filed metrics_measured "
            f"{row.get('metrics_measured')} != its "
            f"spreads {len(spreads)}")
    differing = row.get("differing_metrics") or []
    classification = row.get("classification")
    if classification == "deterministic":
        if differing:
            problems.append(
                f"{label}: files differing metrics "
                f"{differing} but is classified "
                f"deterministic — a flake filed as "
                f"determinism is named, never smoothed")
        nonzero = [m for m, s in spreads.items()
                   if s is not None and s != 0.0]
        if nonzero:
            problems.append(
                f"{label}: classified deterministic "
                f"but metrics {nonzero} spread across "
                f"repeats")
    elif classification == "quarantined":
        if not differing:
            problems.append(
                f"{label}: quarantined with no "
                f"differing metric filed — a quarantine "
                f"quarantines a measured difference")
        reason = row.get("reason") or ""
        if not reason:
            problems.append(
                f"{label}: quarantined without a "
                f"filed reason — a quarantine without "
                f"a reason is named, never tolerated")
        else:
            filed = [e.get("reason") for e in registry
                     if e.get("baseline") == label]
            if reason not in filed:
                problems.append(
                    f"{label}: its quarantine reason is "
                    f"not filed in QUARANTINE_REGISTRY")
    else:
        problems.append(
            f"{label}: classification "
            f"{classification!r} is not a filed "
            f"classification — the suite is neither "
            f"deterministic nor reasoned-quarantined: "
            f"an unfiled nondeterminism, named")
    return problems


def excluded_row_problems(
    row: dict[str, Any],
) -> list[str]:
    """Judge ONE excluded-by-rule row, purely: it
    must name its baseline, its metric, the
    classification and its filed reason (the
    reason TF-121's rule filed in the baseline's
    excluded_metrics). Pure."""
    problems: list[str] = []
    label = f"{row.get('baseline')}/{row.get('metric')}"
    if row.get("classification") != "excluded_by_rule":
        problems.append(
            f"{label}: classification "
            f"{row.get('classification')!r} != "
            f"excluded_by_rule")
    if not row.get("reason"):
        problems.append(
            f"{label}: excluded without its filed "
            f"reason — an exclusion files its reason "
            f"(TF-121's rule), never a bare name")
    return problems


def flake_report_problems(
    report: dict[str, Any],
) -> list[str]:
    """Judge ONE flake report against the filed
    shape, purely: an id, a suite, a metric / leg,
    its differing values and a disposition — each
    missing part named. Pure."""
    problems: list[str] = []
    label = report.get("id", "<unnamed report>")
    if not report.get("id"):
        problems.append("a flake report files no id")
    if not report.get("suite"):
        problems.append(
            f"{label}: the report names no suite — "
            f"a report that cannot name its suite "
            f"settles nothing")
    if not report.get("leg"):
        problems.append(
            f"{label}: the report names no metric / "
            f"leg — the differing leg must be named")
    if not report.get("differing_values"):
        problems.append(
            f"{label}: the report files no differing "
            f"values — a re-roll without the values "
            f"that differed is a silent re-roll "
            f"dressed as a report")
    if not report.get("disposition"):
        problems.append(
            f"{label}: the report files no "
            f"disposition — what was done about the "
            f"difference must be filed, not implied "
            f"by a green re-run")
    return problems


def rerun_record_problems(
    record: dict[str, Any], reports: Any = None,
) -> list[str]:
    """Judge ONE re-run record, purely: a re-run
    recorded after a differing first outcome must
    link a filed flake report — a green re-run
    with no report is a silent re-roll and is
    named; a linked id must resolve. Pure."""
    reports = FLAKE_REPORTS if reports is None else reports
    problems: list[str] = []
    label = record.get("suite", "<unnamed suite>")
    if not record.get("suite"):
        problems.append("a re-run record names no suite")
    if not record.get("rerun_outcome"):
        problems.append(
            f"{label}: the record files no re-run "
            f"outcome")
    report_id = record.get("flake_report_id")
    if not report_id:
        problems.append(
            f"{label}: a re-run after a differing "
            f"result ({record.get('first_outcome')!r} "
            f"-> {record.get('rerun_outcome')!r}) "
            f"files no flake report — a silent "
            f"re-roll, named: a green re-run alone "
            f"settles nothing")
    elif report_id not in [r.get("id") for r in reports]:
        problems.append(
            f"{label}: its flake report id "
            f"{report_id!r} resolves to no filed "
            f"report")
    return problems


# ── The census measurement ──────────────────────────────────────────────────

def _spread_for(values: list[Any]) -> tuple[float | None, bool]:
    """(max-min spread, changed) for one metric's
    repeated values: numeric metrics file their
    absolute spread; non-numeric metrics file
    None + whether any repeat differed (TF-129's
    filed shape, carried)."""
    first = values[0]
    if isinstance(first, (int, float)) \
            and not isinstance(first, bool):
        nums = [float(v) for v in values]
        return max(nums) - min(nums), max(nums) != min(nums)
    changed = any(v != first for v in values[1:])
    return None, changed


def compute_census_rows(
    baseline_ids: Any = None,
    repeats: int = CENSUS_REPEATS,
) -> tuple[dict[str, Any], ...]:
    """The census, measured live: every selected
    pinned baseline re-measured `repeats` times
    in this process through the harness's own
    measurers, per baseline its spreads, its
    differing metrics (any metric whose repeats
    did not all reproduce the pin exactly) and
    its classification — deterministic when
    nothing differed, quarantined when a filed
    quarantine reason covers it, otherwise
    nondeterministic_unfiled (named by the
    checker, never smoothed). A measurer that
    raises files an unmeasurable row, never a
    silent skip."""
    baselines = {b["id"]: b for b in _harness.BASELINES}
    selected = [b["id"] for b in _harness.BASELINES] \
        if baseline_ids is None else list(baseline_ids)
    rows: list[dict[str, Any]] = []
    for baseline_id in selected:
        baseline = baselines[baseline_id]
        measurer = _harness._MEASURERS.get(baseline_id)
        try:
            if measurer is None:
                raise LookupError("no harness measurer")
            runs = [measurer() for _ in range(repeats)]
        except Exception as exc:
            rows.append({
                "baseline": baseline_id,
                "benchmark": baseline["benchmark"],
                "repeats": repeats,
                "metrics_measured": len(baseline["metrics"]),
                "max_abs_spread": 0.0,
                "differing_metrics": list(baseline["metrics"]),
                "spreads": {m: None
                            for m in baseline["metrics"]},
                "classification": "unmeasurable",
                "reason": f"re-measurement failed: {exc!r}",
            })
            continue
        spreads: dict[str, float | None] = {}
        differing: list[str] = []
        numeric_spreads: list[float] = []
        for metric, pinned in baseline["metrics"].items():
            values = [run.get(metric) for run in runs]
            spread, changed = _spread_for(values)
            spreads[metric] = spread
            if spread is not None:
                numeric_spreads.append(spread)
            if changed or any(v != pinned for v in values):
                differing.append(metric)
        reason = quarantine_reason_for(baseline_id)
        if not differing:
            classification = "deterministic"
            row_reason = (
                "all repeats reproduced the pinned "
                "metrics exactly")
        elif reason:
            classification = "quarantined"
            row_reason = reason
        else:
            classification = "nondeterministic_unfiled"
            row_reason = ""
        rows.append({
            "baseline": baseline_id,
            "benchmark": baseline["benchmark"],
            "repeats": repeats,
            "metrics_measured": len(baseline["metrics"]),
            "max_abs_spread": max(numeric_spreads)
            if numeric_spreads else 0.0,
            "differing_metrics": differing,
            "spreads": spreads,
            "classification": classification,
            "reason": row_reason,
        })
    return tuple(rows)


def excluded_metric_rows() -> tuple[dict[str, Any], ...]:
    """The excluded-by-rule rows, built live from
    the harness registry: one per metric filed in
    a baseline's excluded_metrics (TF-121's rule —
    latency / wall / power shapes, never pinned),
    each carrying its filed reason."""
    rows: list[dict[str, Any]] = []
    for baseline in _harness.BASELINES:
        for metric, reason in baseline[
                "excluded_metrics"].items():
            rows.append({
                "baseline": baseline["id"],
                "metric": metric,
                "classification": "excluded_by_rule",
                "reason": reason,
            })
    return tuple(rows)


# ── The published census (literal data, TF-067's CURVE precedent) ───────────
#
# Measured live on this VM at banking (2026-11-01):
# every pinned baseline re-measured five times in
# one process through the harness's measurers —
# every repeat reproduced its pin exactly (TF-036's
# determinism, measured over the whole estate, not
# only TF-129's gated nine). The checker recomputes
# every row live and names any drift.

CENSUS_ROWS: tuple[dict[str, Any], ...] = (
    {'baseline': 'persona_consistency_synthetic_v1', 'benchmark': 'persona_consistency', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'consistency_score': 0.0, 'total_turns': 0.0, 'flagged_turns': 0.0, 'generic_phrase_hits': 0.0, 'norse_tone_hits': 0.0, 'citation_turns': 0.0, 'skald_quality_hits': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'persona_suite_synthetic_v1', 'benchmark': 'persona_suite', 'repeats': 5, 'metrics_measured': 5, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'identity_score': 0.0, 'voice_score': 0.0, 'values_score': 0.0, 'overall_score': 0.0, 'total_turns': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'memory_suite_synthetic_v1', 'benchmark': 'memory_suite', 'repeats': 5, 'metrics_measured': 5, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'present_score': 0.0, 'true_score': 0.0, 'explain_score': 0.0, 'overall_score': 0.0, 'total_cases': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'depth_suite_synthetic_v1', 'benchmark': 'depth_suite', 'repeats': 5, 'metrics_measured': 10, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'chain_off_score': 0.0, 'chain_on_score': 0.0, 'plan_off_score': 0.0, 'plan_on_score': 0.0, 'check_off_score': 0.0, 'check_on_score': 0.0, 'overall_off_score': 0.0, 'overall_on_score': 0.0, 'delta': 0.0, 'total_cases': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'presence_suite_synthetic_v1', 'benchmark': 'presence_suite', 'repeats': 5, 'metrics_measured': 5, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'coherence_score': 0.0, 'steering_score': 0.0, 'closure_score': 0.0, 'overall_score': 0.0, 'total_cases': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'honesty_panel_synthetic_v1', 'benchmark': 'honesty_panel', 'repeats': 5, 'metrics_measured': 13, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'reference_off_score': 0.0, 'reference_on_score': 0.0, 'reference_delta': 0.0, 'mid_off_score': 0.0, 'mid_on_score': 0.0, 'mid_delta': 0.0, 'small_off_score': 0.0, 'small_on_score': 0.0, 'small_delta': 0.0, 'tiny_off_score': 0.0, 'tiny_on_score': 0.0, 'tiny_delta': 0.0, 'total_cells': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'quant_panel_synthetic_v1', 'benchmark': 'quant_panel', 'repeats': 5, 'metrics_measured': 16, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'iq2_xs_score': 0.0, 'iq2_xs_loss': 0.0, 'q2_k_score': 0.0, 'q2_k_loss': 0.0, 'q4_k_m_score': 0.0, 'q4_k_m_loss': 0.0, 'q5_k_m_score': 0.0, 'q5_k_m_loss': 0.0, 'q8_0_score': 0.0, 'q8_0_loss': 0.0, 'fp16_score': 0.0, 'fp16_loss': 0.0, 'fact_mean_score': 0.0, 'reasoning_mean_score': 0.0, 'instruction_mean_score': 0.0, 'total_cells': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'profile_panel_synthetic_v1', 'benchmark': 'profile_panel', 'repeats': 5, 'metrics_measured': 13, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'desktop_cpu_suite_score': 0.0, 'desktop_cpu_bench_avg_words': 0.0, 'desktop_gpu_suite_score': 0.0, 'desktop_gpu_bench_avg_words': 0.0, 'phone_low_suite_score': 0.0, 'phone_low_bench_avg_words': 0.0, 'pi_5_suite_score': 0.0, 'pi_5_bench_avg_words': 0.0, 'pi_zero_suite_score': 0.0, 'pi_zero_bench_avg_words': 0.0, 'server_gpu_suite_score': 0.0, 'server_gpu_bench_avg_words': 0.0, 'total_cells': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'regression_gate_synthetic_v1', 'benchmark': 'regression_gate', 'repeats': 5, 'metrics_measured': 6, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'baselines_gated': 0.0, 'metrics_compared': 0.0, 'max_noise_spread': 0.0, 'gate_failures': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'spot_checks_synthetic_v1', 'benchmark': 'spot_checks', 'repeats': 5, 'metrics_measured': 6, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_transcripts': 0.0, 'total_turns': 0.0, 'suites_covered': 0.0, 'holding_transcripts': 0.0, 'failing_transcripts': 0.0, 'mean_score': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'data_hygiene_synthetic_v1', 'benchmark': 'data_hygiene', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_inputs': 0.0, 'synthetic_inputs': 0.0, 'licensed_inputs': 0.0, 'derived_inputs': 0.0, 'clean_inputs': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'wyrd_panel_synthetic_v1', 'benchmark': 'wyrd_panel', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_cases': 0.0, 'consistent_cases': 0.0, 'contradicted_cases': 0.0, 'honest_unanswerable_cases': 0.0, 'overall_score': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'cost_panel_synthetic_v1', 'benchmark': 'cost_panel', 'repeats': 5, 'metrics_measured': 10, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_cells': 0.0, 'total_units': 0.0, 'total_token_budget': 0.0, 'pi_zero_token_budget': 0.0, 'server_gpu_token_budget': 0.0, 'mean_sharpness_per_1000_tokens': 0.0, 'pi_zero_mean_sharpness_per_1000_tokens': 0.0, 'server_gpu_mean_sharpness_per_1000_tokens': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'baseline_book_synthetic_v1', 'benchmark': 'baseline_book', 'repeats': 5, 'metrics_measured': 6, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_baselines': 0.0, 'total_pinned_metrics': 0.0, 'gated_baselines': 0.0, 'ungated_baselines': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'flake_discipline_synthetic_v1', 'benchmark': 'flake_discipline', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_census_baselines': 0.0, 'deterministic_baselines': 0.0, 'quarantined_baselines': 0.0, 'excluded_by_rule_metrics': 0.0, 'census_repeats': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'public_claims_synthetic_v1', 'benchmark': 'public_claims', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_claims': 0.0, 'matched_claims': 0.0, 'struck_claims': 0.0, 'distinct_sources': 0.0, 'live_verified_matched': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'harness_tool_synthetic_v1', 'benchmark': 'harness_tool', 'repeats': 5, 'metrics_measured': 7, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_panels': 0.0, 'full_panel_baselines': 0.0, 'gate_panel_baselines': 0.0, 'audit_panel_baselines': 0.0, 'panel_member_slots': 0.0, 'probes_passed': 0.0, 'probes_total': 0.0}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
    {'baseline': 'profile_benchmark_empty_store_v1', 'benchmark': 'profile_benchmark', 'repeats': 5, 'metrics_measured': 9, 'max_abs_spread': 0.0, 'differing_metrics': [], 'spreads': {'total_turns': 0.0, 'successful_turns': 0.0, 'failed_turns': 0.0, 'citation_accuracy': 0.0, 'enforcement_pass_rate': 0.0, 'avg_response_words': 0.0, 'min_words_met_rate': 0.0, 'avg_confidence': 0.0, 'quality_tiers': None}, 'classification': 'deterministic', 'reason': 'all repeats reproduced the pinned metrics exactly'},
)


# ── The filed probes ────────────────────────────────────────────────────────
#
# Perturbed copies of filed rows / reports /
# records run against the pure validators. Real
# filings are never perturbed; a probe passes when
# the pure parts file exactly the expected shape.

def _probe_row(**overrides: Any) -> dict[str, Any]:
    row = copy.deepcopy(dict(CENSUS_ROWS[0])) if CENSUS_ROWS \
        else {
            "baseline": "persona_consistency_synthetic_v1",
            "benchmark": "persona_consistency",
            "repeats": CENSUS_REPEATS,
            "metrics_measured": 1,
            "max_abs_spread": 0.0,
            "differing_metrics": [],
            "spreads": {"consistency_score": 0.0},
            "classification": "deterministic",
            "reason": "all repeats reproduced the "
                      "pinned metrics exactly"}
    row.update(overrides)
    return row


def _probe_report(**overrides: Any) -> dict[str, Any]:
    report = copy.deepcopy(dict(FLAKE_REPORTS[0])) \
        if FLAKE_REPORTS else {
            "id": "probe_report", "suite": "probe_suite",
            "leg": "overall_score",
            "differing_values": ["0.5", "0.6"],
            "disposition": "filed"}
    report.update(overrides)
    return report


FLAKE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_deterministic_census_row_passes",
     "kind": "census_row", "overrides": {},
     "expect_problem": False,
     "detail": "a filed deterministic census row "
               "whose repeats, spreads and differing "
               "metrics agree files no problem"},
    {"id": "census_perturbed_suite_is_named",
     "kind": "census_row",
     "overrides": {"differing_metrics": ["consistency_score"]},
     "expect_problem": True,
     "detail": "a suite whose census repeats "
               "differed, filed as deterministic, is "
               "a flake filed as determinism and is "
               "named by baseline"},
    {"id": "quarantine_without_reason_is_named",
     "kind": "census_row",
     "overrides": {"classification": "quarantined",
                   "differing_metrics": ["consistency_score"],
                   "reason": ""},
     "expect_problem": True,
     "detail": "a quarantine without a filed reason "
               "is named — nondeterminism is "
               "quarantined with a reason or not "
               "at all"},
    {"id": "flake_report_missing_differing_values_is_named",
     "kind": "report",
     "overrides": {"differing_values": []},
     "expect_problem": True,
     "detail": "a flake report filing no differing "
               "values is a silent re-roll dressed "
               "as a report and is named"},
    {"id": "flake_report_missing_disposition_is_named",
     "kind": "report",
     "overrides": {"disposition": ""},
     "expect_problem": True,
     "detail": "a flake report filing no "
               "disposition leaves the difference "
               "unsettled and is named"},
    {"id": "silent_reroll_record_is_named",
     "kind": "rerun",
     "record": {"suite": "full_gate (probe legs)",
                "first_outcome": "2 failures",
                "rerun_outcome": "passed",
                "flake_report_id": None},
     "expect_problem": True,
     "detail": "a re-run recorded after a differing "
               "result with no flake report is a "
               "silent re-roll and is named — a "
               "green re-run alone settles nothing"},
    {"id": "filed_report_and_rerun_pass",
     "kind": "filed_pair", "overrides": {},
     "expect_problem": False,
     "detail": "the filed TF-126 report validates "
               "and its re-run record resolves to "
               "it — the estate's one re-roll, "
               "owned in the open"},
)


def run_flake_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in FLAKE_PROBES:
        if probe["kind"] == "census_row":
            problems = census_row_problems(
                _probe_row(**probe["overrides"]))
        elif probe["kind"] == "report":
            problems = flake_report_problems(
                _probe_report(**probe["overrides"]))
        elif probe["kind"] == "rerun":
            problems = rerun_record_problems(probe["record"])
        else:  # filed_pair
            problems = (
                flake_report_problems(dict(FLAKE_REPORTS[0]))
                + rerun_record_problems(dict(RERUN_RECORDS[0])))
        passed = bool(problems) == probe["expect_problem"]
        results.append({
            "id": probe["id"], "passed": passed,
            "problems": problems,
            "detail": probe["detail"]})
    return results


# ── The discipline run ──────────────────────────────────────────────────────

@dataclass
class FlakeDisciplineResult:
    """One full census run: the rows it filed, the
    excluded rows, and the verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    excluded_rows: list[dict[str, Any]] = field(
        default_factory=list)

    @property
    def total_baselines(self) -> int:
        return len(self.rows)

    @property
    def deterministic_baselines(self) -> int:
        return sum(1 for r in self.rows
                   if r["classification"] == "deterministic")

    @property
    def quarantined_baselines(self) -> int:
        return sum(1 for r in self.rows
                   if r["classification"] == "quarantined")

    @property
    def excluded_metrics(self) -> int:
        return len(self.excluded_rows)

    @property
    def overall_score(self) -> float:
        if not self.rows:
            return 0.0
        return self.deterministic_baselines / len(self.rows)

    @property
    def verdict(self) -> str:
        """PASS only on the full census with every
        suite classified deterministic or
        reasoned-quarantined; FAIL on a full census
        carrying an unfiled nondeterminism; REVIEW
        for any partial or empty selection, however
        it scores."""
        if [r["baseline"] for r in self.rows] != [
                b["id"] for b in _harness.BASELINES]:
            return "REVIEW"
        clean = all(
            not census_row_problems(r) for r in self.rows)
        return "PASS" if clean else "FAIL"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_flake_discipline(
    baseline_ids: Any = None,
) -> FlakeDisciplineResult:
    """Run the census over the selection (the full
    pinned estate by default), measured live
    through the harness measurers.

    Any explicit baseline selection may be given —
    including an empty one, which files 0 baselines
    / REVIEW (no vacuous pass).
    """
    if baseline_ids is not None and not tuple(baseline_ids):
        return FlakeDisciplineResult(
            rows=[], excluded_rows=[
                dict(r) for r in excluded_metric_rows()])
    rows = compute_census_rows(baseline_ids)
    return FlakeDisciplineResult(
        rows=[dict(r) for r in rows],
        excluded_rows=[dict(r) for r in excluded_metric_rows()])


# ── The readable render (the doc, verbatim) ─────────────────────────────────

def render_flake_discipline_markdown(
    rows: Any = None, excluded: Any = None,
) -> str:
    """Render the discipline deterministically —
    the exact text filed as
    docs/architecture/FLAKE_DISCIPLINE.md: the
    census, the excluded-by-rule rows, the
    quarantine registry, the filed flake reports
    and re-run records, and the rules. Every
    number comes from the filed data; nothing is
    hand-written here but the frame."""
    rows = CENSUS_ROWS if rows is None else tuple(rows)
    excluded = excluded_metric_rows() if excluded is None \
        else tuple(excluded)
    deterministic = sum(
        1 for r in rows
        if r["classification"] == "deterministic")
    quarantined = sum(
        1 for r in rows
        if r["classification"] == "quarantined")
    lines: list[str] = []
    lines.append("# Flake Discipline — TF-135")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`benchmarks/flake_discipline.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m benchmarks.flake_discipline`. "
        "The checker (`flake_discipline_problems()`) "
        "recomputes the census live through the "
        "harness measurers and fails on any drift "
        "from the filed rows.")
    lines.append("")
    lines.append(f"Format: `{FLAKE_DISCIPLINE_FORMAT}`")
    lines.append(f"Census repeats: {CENSUS_REPEATS}")
    lines.append(f"Baselines censused: {len(rows)}")
    lines.append(f"Deterministic baselines: {deterministic}")
    lines.append(f"Quarantined baselines: {quarantined}")
    lines.append(
        f"Excluded-by-rule metrics: {len(excluded)}")
    lines.append(f"Flake reports filed: {len(FLAKE_REPORTS)}")
    lines.append(f"Re-run records filed: {len(RERUN_RECORDS)}")
    lines.append("")
    lines.append(
        "Every pinned suite is census-measured by "
        "repetition: each baseline is re-measured "
        f"{CENSUS_REPEATS} times in one process "
        "through the harness's own measurers, and a "
        "suite is deterministic only if all repeats "
        "reproduce its pinned metrics exactly — the "
        "standard cited to " + DETERMINISM_CITATION +
        ". Nondeterminism is fixed or quarantined "
        "with a filed reason, never silently "
        "tolerated, and no re-roll is silent: a "
        "re-run after a differing result files a "
        "flake report naming the suite, the metric "
        "/ leg, the differing values and the "
        "disposition.")
    lines.append("")
    lines.append("## The census (filed rows)")
    lines.append("")
    lines.append(
        "| Baseline | Benchmark | Repeats | Metrics | "
        "Classification | Differing metrics | Max spread |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        differing = ", ".join(row["differing_metrics"]) \
            if row["differing_metrics"] else "—"
        lines.append(
            f"| `{row['baseline']}` | {row['benchmark']} | "
            f"{row['repeats']} | {row['metrics_measured']} | "
            f"{row['classification']} | {differing} | "
            f"{row['max_abs_spread']} |")
    lines.append("")
    lines.append("## Excluded by rule (never flakes)")
    lines.append("")
    lines.append(
        "Timing / wall metrics and the other shapes "
        "filed in a baseline's excluded_metrics stay "
        "unpinned per TF-121's rule and are classified "
        "excluded-by-rule with their filed reasons:")
    lines.append("")
    for row in excluded:
        lines.append(
            f"- `{row['baseline']}` / `{row['metric']}`: "
            f"{row['reason']}")
    lines.append("")
    lines.append("## Quarantine registry")
    lines.append("")
    if QUARANTINE_REGISTRY:
        for entry in QUARANTINE_REGISTRY:
            lines.append(
                f"- `{entry['baseline']}`: {entry['reason']}")
    else:
        lines.append(
            "No suite is quarantined on this tree: the "
            "census measured every pinned baseline "
            "deterministic, so the registry files "
            "empty. A differing suite enters here "
            "only with its reason filed, beside its "
            "census row's differing metrics.")
    lines.append("")
    lines.append("## Filed flake reports")
    lines.append("")
    for report in FLAKE_REPORTS:
        lines.append(f"### Report `{report['id']}`")
        lines.append("")
        lines.append(f"- Suite: {report['suite']}")
        lines.append(f"- Metric / leg: {report['leg']}")
        lines.append("- Differing values:")
        for value in report["differing_values"]:
            lines.append(f"  - {value}")
        lines.append(f"- Disposition: {report['disposition']}")
        lines.append("")
    lines.append("## Filed re-run records")
    lines.append("")
    for record in RERUN_RECORDS:
        lines.append(
            f"- Suite: {record['suite']} — first: "
            f"{record['first_outcome']}; re-run: "
            f"{record['rerun_outcome']}; flake report: "
            f"`{record['flake_report_id']}`")
    lines.append("")
    lines.append("## The rules")
    lines.append("")
    for rule in FLAKE_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## Filed limits")
    lines.append("")
    lines.append(
        "- The census judges the harness's pinned "
        "fixture baselines only — synthetic fixtures "
        "through the real scorers (TF-017). No live "
        "model is censused; live-model flakiness is "
        "NOT PROVEN (TF-173), never claimed.")
    lines.append(
        "- The census runs in one process on this VM "
        "(five repeats, in-process). Cross-machine "
        "noise is not measured here.")
    lines.append(
        "- A quarantined suite stays pinned and "
        "censused: quarantine files the reason and "
        "the differing metrics; it never deletes a "
        "baseline, loosens a tolerance or hides a "
        "re-roll.")
    lines.append("")
    lines.append(
        "Proof grade: FIXTURE — the harness's pinned "
        "synthetic-fixture baselines re-measured live "
        "on this VM; no live model, never claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_flake_discipline_doc() -> Path:
    """Regenerate the filed doc by command: render
    from the filed data and write the doc, byte for
    byte. Returns the doc's path."""
    doc = _REPO_ROOT / FLAKE_DISCIPLINE_DOC
    doc.write_text(
        render_flake_discipline_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m benchmarks.flake_discipline`."""
    doc = write_flake_discipline_doc()
    print(
        f"flake discipline doc regenerated: "
        f"{len(CENSUS_ROWS)} censused baselines -> "
        f"{FLAKE_DISCIPLINE_DOC}")
    return 0


# ── The live checker ────────────────────────────────────────────────────────

def flake_discipline_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: format, repeats, rules.
    if FLAKE_DISCIPLINE_FORMAT != "thoughtforge-flake-discipline/1":
        problems.append(
            f"FLAKE_DISCIPLINE_FORMAT drifted: "
            f"{FLAKE_DISCIPLINE_FORMAT}")
    if CENSUS_REPEATS != 5 or CENSUS_REPEATS != _GATE_NOISE_REPEATS:
        problems.append(
            f"CENSUS_REPEATS is {CENSUS_REPEATS}, not "
            f"TF-129's filed NOISE_REPEATS "
            f"({_GATE_NOISE_REPEATS})")
    rule_ids = [r["id"] for r in FLAKE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate FLAKE_RULES ids")
    for needed in (
            "every_pinned_suite_is_census_measured_by_repetition",
            "deterministic_means_every_repeat_reproduces_the_pin_exactly",
            "nondeterminism_is_fixed_or_quarantined_never_tolerated",
            "no_silent_rerolls",
            "timing_and_wall_metrics_are_excluded_by_rule_not_flakes",
            "quarantine_needs_a_filed_reason_and_keeps_the_pin",
            "flake_reports_are_validated_as_data",
            "probes_prove_the_checker_can_fail",
            "pass_needs_the_full_census_classified",
            "the_flake_discipline_baseline_is_not_gated_by_tf129"):
        if needed not in rule_ids:
            problems.append(f"FLAKE_RULES lost {needed}")

    # The quarantine registry: every entry named
    # and reasoned, over a real pinned baseline.
    baseline_ids = [b["id"] for b in _harness.BASELINES]
    for entry in QUARANTINE_REGISTRY:
        if entry.get("baseline") not in baseline_ids:
            problems.append(
                f"quarantine entry names unpinned "
                f"baseline {entry.get('baseline')}")
        if not entry.get("reason"):
            problems.append(
                f"quarantine entry for "
                f"{entry.get('baseline')} files no reason")

    # The filed reports + records validate.
    report_ids = [r.get("id") for r in FLAKE_REPORTS]
    if len(report_ids) != len(set(report_ids)):
        problems.append("duplicate FLAKE_REPORTS ids")
    for report in FLAKE_REPORTS:
        problems.extend(flake_report_problems(report))
    for record in RERUN_RECORDS:
        problems.extend(rerun_record_problems(record))

    # The filed census: shape + pure validation.
    if [r["baseline"] for r in CENSUS_ROWS] != baseline_ids:
        problems.append(
            "the filed census does not file every "
            "pinned baseline in harness order")
    for row in CENSUS_ROWS:
        problems.extend(census_row_problems(row))

    # The excluded rows: built live, each reasoned.
    excluded = excluded_metric_rows()
    expected_excluded = sum(
        len(b["excluded_metrics"]) for b in _harness.BASELINES)
    if len(excluded) != expected_excluded:
        problems.append(
            f"excluded-by-rule rows {len(excluded)} != "
            f"the baselines' filed excluded metrics "
            f"{expected_excluded}")
    for row in excluded:
        problems.extend(excluded_row_problems(row))

    # The published census: literal rows recomputed
    # live, once — filing, recomputation and the
    # full verdict judged from the same census.
    live_rows = compute_census_rows()
    if [dict(r) for r in live_rows] != [
            dict(r) for r in CENSUS_ROWS]:
        problems.append(
            "CENSUS_ROWS drifted from the live "
            "recomputation (a suite's repeats no "
            "longer reproduce the filed census — "
            "re-file deliberately, never silently)")
    live_result = FlakeDisciplineResult(
        rows=[dict(r) for r in live_rows],
        excluded_rows=[dict(r) for r in excluded])
    if live_result.verdict != "PASS" or not live_result.passes:
        problems.append(
            f"the live census does not file PASS on "
            f"this tree: {live_result.verdict} "
            f"({live_result.deterministic_baselines} "
            f"deterministic / "
            f"{live_result.quarantined_baselines} "
            f"quarantined of {live_result.total_baselines})")

    # Partial + empty selections file REVIEW,
    # mechanically (cheap selections only — the
    # full live census above is the expensive leg
    # and it runs once).
    empty = run_flake_discipline(baseline_ids=())
    if empty.total_baselines != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    subset = run_flake_discipline(
        baseline_ids=("persona_consistency_synthetic_v1",))
    if subset.total_baselines != 1 or subset.verdict != "REVIEW":
        problems.append(
            "a single-baseline selection does not file "
            "its baseline / REVIEW mechanically")

    # Determinism of the render + the filed doc is
    # the render, verbatim.
    if render_flake_discipline_markdown() != \
            render_flake_discipline_markdown(CENSUS_ROWS, excluded):
        problems.append(
            "the render is not deterministic: two "
            "renders over the filed data differ")
    doc = _REPO_ROOT / FLAKE_DISCIPLINE_DOC
    if not doc.exists():
        problems.append(
            f"the write-up {FLAKE_DISCIPLINE_DOC} is "
            f"missing (regenerate with python3 -m "
            f"benchmarks.flake_discipline)")
    elif doc.read_text(encoding="utf-8") != \
            render_flake_discipline_markdown():
        problems.append(
            f"the filed doc {FLAKE_DISCIPLINE_DOC} is "
            f"not the render verbatim (regenerate with "
            f"python3 -m benchmarks.flake_discipline)")

    # The probes all behave.
    probes = run_flake_probes()
    if len(probes) != len(FLAKE_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the flake probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # TF-036 citation: the determinism neighbour
    # entry stands in the harness index and files
    # this slice as its consumer.
    determinism = [b for b in _harness.BENCHMARKS
                   if b["id"] == "determinism"]
    if not determinism:
        problems.append("BENCHMARKS lost the determinism entry")
    elif "TF-135" not in determinism[0]["suite_slice"]:
        problems.append(
            "the determinism entry does not file "
            "TF-135 as its consumer (TF-036's citation)")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included), and the gate's gated set untouched.
    from benchmarks.harness import (
        BASELINES,
        BENCHMARKS,
        baseline_problems,
    )
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "flake_discipline" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the flake_discipline entry")
    if FLAKE_DISCIPLINE_BASELINE_ID not in [
            b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the flake-discipline baseline")
    if FLAKE_DISCIPLINE_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the flake-discipline baseline entered "
            "TF-129's gated set (it is judged by this "
            "checker, not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the "
            f"flake discipline: {earlier}")
    return problems


def flake_discipline_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_flake_probes()
    deterministic = sum(
        1 for r in CENSUS_ROWS
        if r["classification"] == "deterministic")
    quarantined = sum(
        1 for r in CENSUS_ROWS
        if r["classification"] == "quarantined")
    return {
        "format": FLAKE_DISCIPLINE_FORMAT,
        "doc": FLAKE_DISCIPLINE_DOC,
        "census_repeats": CENSUS_REPEATS,
        "determinism_citation": DETERMINISM_CITATION,
        "census_rows": [dict(r) for r in CENSUS_ROWS],
        "excluded_rows": [dict(r) for r in excluded_metric_rows()],
        "quarantine_registry": [dict(e) for e in QUARANTINE_REGISTRY],
        "flake_reports": [dict(r) for r in FLAKE_REPORTS],
        "rerun_records": [dict(r) for r in RERUN_RECORDS],
        "measured": {
            "total_baselines": len(CENSUS_ROWS),
            "deterministic_baselines": deterministic,
            "quarantined_baselines": quarantined,
            "excluded_by_rule_metrics": len(excluded_metric_rows()),
            "flake_reports_filed": len(FLAKE_REPORTS),
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
        "rules": [dict(r) for r in FLAKE_RULES],
    }


__all__ = [
    "CENSUS_REPEATS",
    "CENSUS_ROWS",
    "DETERMINISM_CITATION",
    "FLAKE_DISCIPLINE_BASELINE_ID",
    "FLAKE_DISCIPLINE_DOC",
    "FLAKE_DISCIPLINE_FORMAT",
    "FLAKE_PROBES",
    "FLAKE_REPORTS",
    "FLAKE_RULES",
    "QUARANTINE_REGISTRY",
    "RERUN_RECORDS",
    "FlakeDisciplineResult",
    "census_row_problems",
    "compute_census_rows",
    "excluded_metric_rows",
    "excluded_row_problems",
    "flake_discipline_as_data",
    "flake_discipline_problems",
    "flake_report_problems",
    "main",
    "quarantine_reason_for",
    "render_flake_discipline_markdown",
    "rerun_record_problems",
    "run_flake_discipline",
    "run_flake_probes",
    "write_flake_discipline_doc",
]

if __name__ == "__main__":
    sys.exit(main())
