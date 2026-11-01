"""
Public-claims audit, as data (TF-136).

The roadmap's rule: *public-claims audit: every
performance claim in the repo's docs matched to a
suite result or struck in the audit doc.* Part VII
turned ThoughtForge's claims into numbers (TF-121..
TF-135) — but the repo's public-facing docs still
carried claims no suite had ever produced: MODEL_CARD's
target table read as results, the "100+ turns" persona
wording stood in three docs with no run behind the
scorer it named (D-BENCH-4's standing share), and
TASK_PHASE6 still described the locust file as an
HttpUser load test after TF-121 excluded it as a
no-HTTP manual script (D-BENCH-5's standing share).

The measured starting state, probed live this tick
on this tree before anything was built:

  * **The estate stood clean, and no public-claims
    audit existed.** ``harness_problems()`` returned
    [] and ``baseline_problems()`` returned [] over
    the sixteen pinned baselines,
    ``regression_gate_problems()`` returned [] with
    the live gate PASS (9 baselines / 83 metrics /
    0 failures), ``baseline_book_problems()``
    returned [] (book PASS 16 baselines / 132 pinned
    metrics), ``data_hygiene_problems()`` returned
    [] (21 inputs, all clean) and
    ``flake_discipline_problems()`` returned [] —
    and a grep across benchmarks / docs / tests /
    src found no public-claims audit anywhere.
  * **The gap was never numbered.** The number
    **D-BENCH-21** was verified unused anywhere in
    benchmarks / docs / tests / src before this
    slice filed it (the D-BENCH-9..20 pattern).

What this module files, as verifiable data:

  * ``CLAIM_ROWS`` — the claim census, published as
    literal data: every performance claim found by
    scanning the repo's docs (``docs/**/*.md`` plus
    the top-level ``*.md`` docs; README.md is read
    for claims but never edited — striking happens
    only here, in the audit doc) that purports to
    state a suite / harness result or a public
    performance target. Each row files its source
    doc, a marker that must stand verbatim in that
    doc (checked live), the claim itself, and its
    disposition: **matched** (a harness baseline id
    + metric + the filed value, recomputed live by
    the checker against the harness's own measurers
    so a drifted value is named) or **struck** (a
    filed reason: no suite result supports it, a
    live-model claim with no live run, or a stale
    number / description). Struck claims are filed
    as struck here — never silently deleted, and
    their source docs are never edited by this
    slice.
  * The measured truth on this tree: **21 claims —
    12 matched, 9 struck.** The matched claims are
    the Part VII suites' / panels' own headline
    numbers, each reproducing its pin exactly. The
    struck claims are named in the audit doc: the
    MODEL_CARD target table's citation / enforcement
    / avg-words rows (targets only; the only measured
    values are the empty-store baseline's 0.0 / 0.0 /
    9.0), its "100+ turns" persona wording and the
    same wording in BUILD_PLAN_v1 and
    TASK_PHASE6_RELEASE (D-BENCH-4), BUILD_PLAN's
    per-profile ">85%" citation claim, TASK_PHASE6's
    stale HttpUser description (D-BENCH-5), and
    MODEL_CARD's "minutes per response" edge-speed
    claim (a live-model claim; no live run exists on
    this VM, TF-173, and latency is never pinned,
    TF-121).
  * ``CLAIM_PROBES`` — filed probes against the pure
    validator with perturbed copies of filed rows
    (real filings never perturbed): the clean matched
    row passes; the clean struck row passes; a matched
    claim with a wrong value is named; a struck claim
    that actually matches a suite result is named; a
    claim missing its source is named; a matched claim
    missing its metric is named; an empty claim is
    named, never crashed on.
  * ``run_public_claims()`` files the verdict: PASS
    only on the FULL census with every claim
    classified matched-with-live-verified-value or
    reasoned-struck; a partial selection files REVIEW
    however it scores; an empty selection files
    0 claims / REVIEW — no vacuous pass.
  * The readable filing IS the render:
    ``render_public_claims_markdown()`` renders the
    census and the rules deterministically, and
    ``docs/architecture/PUBLIC_CLAIMS_AUDIT.md``
    files that render verbatim — the checker requires
    the doc to reproduce the render exactly, and
    ``python3 -m benchmarks.public_claims``
    regenerates it.

D-BENCH-21 is FIXED by this slice: the repo's docs'
performance claims now have an audit — a census in
which every claim is either matched to a live suite
result or struck with its reason filed. D-BENCH-4
and D-BENCH-5 are narrowed by it in the open (their
public-claims shares are discharged here; their
source wording itself is not edited by this slice,
per the audit rule above). This slice's own harness
baseline is judged by this module's checker, NOT
gated by TF-129's regression gate (the gate's gated
set stays the nine TF-121..TF-128 baselines exactly
as filed; the exclusion is filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129..TF-135 filed for their own baselines).
TF-131 coupling, filed plainly: this audit adds NO
benchmark input — it consumes only already-filed
docs and pins — so the hygiene inventory does NOT
grow, its pinned baseline is NOT re-pinned, and no
new harness fixture is filed (the TF-133 / TF-134 /
TF-135 shape).

Filed limits, stated plainly:

  * The census covers performance claims that
    purport to state a suite / harness result or a
    public performance target, in the docs scanned
    above. Per-slice device filings whose numbers
    are owned and checked by their own slice modules
    (e.g. PI_PROFILE's TF-079 / TF-081 on-device
    rows) are not re-judged here; vague capability
    prose with no number or assertive performance
    content files no claim.
  * A matched claim is matched to a pinned
    FIXTURE baseline (TF-017): scripted synthetic
    suites through the real scorers on this VM. No
    live model stands behind any matched value;
    live behaviour is NOT PROVEN (TF-173).
  * A struck claim is struck from the ledger of
    claims the repo's suites support — the audit
    files the reason; it does not edit the source
    doc, and it does not assert an upstream fact
    (e.g. a data source's own scale) is false.
  * Canon / persona contents are never a claim
    source here (TF-131 / TF-017 discipline): the
    census reads docs' performance wording only.

The human-readable write-up is
``docs/architecture/PUBLIC_CLAIMS_AUDIT.md``,
filed verbatim as this module's render.
Proof grade: FIXTURE — pinned synthetic-fixture
baselines re-measured live on this VM; no live
model, never claimed.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PUBLIC_CLAIMS_FORMAT = "thoughtforge-public-claims/1"
PUBLIC_CLAIMS_DOC = "docs/architecture/PUBLIC_CLAIMS_AUDIT.md"

#: The new baseline this slice registers in the
#: harness — the audit's own pin. It is NOT one of
#: TF-129's gated baselines (the gate gates the
#: nine TF-121..TF-128 baselines, as filed; this
#: baseline is judged by this module's checker).
PUBLIC_CLAIMS_BASELINE_ID = "public_claims_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The real machinery, reused — never copied: the
# harness's pinned baselines and its own measurers.
# This module imports the harness at top level; the
# harness binds this module only lazily (inside its
# measure function), so no import cycle exists.
from benchmarks import harness as _harness  # noqa: E402


# ── The audit's rules ───────────────────────────────────────────────────────

PUBLIC_CLAIMS_RULES: tuple[dict[str, Any], ...] = (
    {"id": "every_public_performance_claim_is_censused",
     "rule": "every performance claim in the repo's "
             "docs (docs/**/*.md plus the top-level "
             "*.md docs) that purports to state a "
             "suite / harness result or a public "
             "performance target files one CLAIM_ROWS "
             "row: its source doc, a marker standing "
             "verbatim in that doc, the claim, and "
             "its disposition"},
    {"id": "matched_means_a_live_verified_suite_value",
     "rule": "a claim is matched only with a harness "
             "baseline id, a metric, and the filed "
             "value — and the checker recomputes that "
             "metric live through the harness's own "
             "measurer, so a drifted value is named, "
             "never carried"},
    {"id": "struck_means_a_filed_reason_never_a_deletion",
     "rule": "a claim no suite result supports — a "
             "target filed as if a result, a live-"
             "model claim with no live run, a stale "
             "number or description — is struck WITH "
             "its reason filed in the audit doc; "
             "struck claims are never silently "
             "deleted and their source docs are never "
             "edited by this slice (README.md is "
             "read for claims, never edited at all)"},
    {"id": "targets_are_not_results",
     "rule": "a target a doc files as a target stays "
             "a target: it files matched only when a "
             "suite result actually meets it under "
             "the instrument the claim names — a "
             "target table is not evidence"},
    {"id": "probes_prove_the_validator_can_fail",
     "rule": "CLAIM_PROBES perturbs copies of filed "
             "rows against the pure validator (a "
             "matched claim with a wrong value, a "
             "struck claim that actually matches, a "
             "claim missing its source, a matched "
             "claim missing its metric) plus the clean "
             "shapes — real filings are never "
             "perturbed"},
    {"id": "pass_needs_the_full_census_classified",
     "rule": "the verdict is PASS only on the full "
             "census with every claim classified "
             "matched-with-live-verified-value or "
             "reasoned-struck; a partial selection "
             "files REVIEW however it scores, and an "
             "empty selection files 0 claims / REVIEW"},
    {"id": "the_public_claims_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates the "
             "nine TF-121..TF-128 pinned baselines "
             "exactly as filed; this slice's own "
             "baseline (public_claims_synthetic_v1) "
             "is judged by this module's checker, "
             "not gated by the gate — the shape "
             "TF-129..TF-135 filed for their own "
             "baselines, with the exclusion filed "
             "in the open in regression_gate.py"},
    {"id": "no_new_benchmark_input_was_added",
     "rule": "the audit consumes only already-filed "
             "docs and pins — it adds no benchmark "
             "input, so it files no new harness "
             "fixture and no hygiene row, and "
             "TF-131's inventory and its pinned "
             "baseline do not move (filed in the "
             "open, TF-133's coupling shape)"},
)


# ── The claim census (literal data) ─────────────────────────────────────────
#
# Scanned live from the repo's docs at banking
# (2026-11-01). A matched row's filed_value is the
# harness pin for its baseline + metric; the
# checker recomputes it live through the harness
# measurer and names any drift. A struck row files
# its reason; the claim stays in its source doc,
# struck only here.

CLAIM_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "persona_suite_overall_score",
     "source": "docs/architecture/PERSONA_CONSISTENCY_SUITE.md",
     "marker": "0.8454",
     "claim": "Persona suite: 'Overall 0.8454 over "
              "108 scored turns — PASS'",
     "disposition": "matched",
     "baseline": "persona_suite_synthetic_v1",
     "metric": "overall_score",
     "filed_value": 0.8453703703703704,
     "reason": ""},
    {"id": "persona_suite_turn_count",
     "source": "docs/architecture/PERSONA_CONSISTENCY_SUITE.md",
     "marker": "108 scored turns",
     "claim": "Persona suite: scored over 108 turns",
     "disposition": "matched",
     "baseline": "persona_suite_synthetic_v1",
     "metric": "total_turns",
     "filed_value": 108,
     "reason": ""},
    {"id": "memory_suite_overall_score",
     "source": "docs/architecture/MEMORY_FIDELITY_SUITE.md",
     "marker": "0.8056",
     "claim": "Memory suite: 'Overall 0.8056 over 36 "
              "scored cases — PASS'",
     "disposition": "matched",
     "baseline": "memory_suite_synthetic_v1",
     "metric": "overall_score",
     "filed_value": 0.8055555555555557,
     "reason": ""},
    {"id": "depth_suite_overall_on_score",
     "source": "docs/architecture/DEPTH_SUITE.md",
     "marker": "0.8056",
     "claim": "Depth suite: overall ON score 0.8056 "
              "(OFF 0.4722) — PASS",
     "disposition": "matched",
     "baseline": "depth_suite_synthetic_v1",
     "metric": "overall_on_score",
     "filed_value": 0.8055555555555557,
     "reason": ""},
    {"id": "depth_suite_delta",
     "source": "docs/architecture/DEPTH_SUITE.md",
     "marker": "+0.3333",
     "claim": "Depth suite: scaffold delta +0.3333",
     "disposition": "matched",
     "baseline": "depth_suite_synthetic_v1",
     "metric": "delta",
     "filed_value": 0.3333333333333334,
     "reason": ""},
    {"id": "presence_suite_overall_score",
     "source": "docs/architecture/PRESENCE_SUITE.md",
     "marker": "0.7777777777777778",
     "claim": "Presence suite: 'Overall "
              "0.7777777777777778 over 36 scored "
              "sagas — PASS'",
     "disposition": "matched",
     "baseline": "presence_suite_synthetic_v1",
     "metric": "overall_score",
     "filed_value": 0.7777777777777778,
     "reason": ""},
    {"id": "quant_panel_q4_k_m_score",
     "source": "docs/architecture/QUANT_PANEL.md",
     "marker": "0.8125",
     "claim": "Quant panel: q4_k_m scores 0.8125 "
              "overall",
     "disposition": "matched",
     "baseline": "quant_panel_synthetic_v1",
     "metric": "q4_k_m_score",
     "filed_value": 0.8125,
     "reason": ""},
    {"id": "quant_panel_q5_k_m_score",
     "source": "docs/architecture/QUANT_PANEL.md",
     "marker": "0.7083",
     "claim": "Quant panel: q5_k_m scores 0.7083 "
              "overall (the doc rounds the pin)",
     "disposition": "matched",
     "baseline": "quant_panel_synthetic_v1",
     "metric": "q5_k_m_score",
     "filed_value": 0.7083333333333334,
     "reason": ""},
    {"id": "wyrd_panel_overall_score",
     "source": "docs/architecture/WYRD_PANEL.md",
     "marker": "0.5555555555555556",
     "claim": "WYRD panel: overall 0.5555555555555556 "
              "(9 consistent / 8 contradicted / 1 "
              "honest-unanswerable of 18)",
     "disposition": "matched",
     "baseline": "wyrd_panel_synthetic_v1",
     "metric": "overall_score",
     "filed_value": 0.5555555555555556,
     "reason": ""},
    {"id": "cost_panel_total_token_budget",
     "source": "docs/architecture/COST_PANEL.md",
     "marker": "766,800",
     "claim": "Cost panel: total token budget "
              "766,800 over 24 cells / 1,296 units",
     "disposition": "matched",
     "baseline": "cost_panel_synthetic_v1",
     "metric": "total_token_budget",
     "filed_value": 766800,
     "reason": ""},
    {"id": "baseline_book_baseline_count",
     "source": "docs/architecture/BASELINE_BOOK.md",
     "marker": "Baselines filed: 17",
     "claim": "Baseline book: 17 baselines filed "
              "(the generated book's own header)",
     "disposition": "matched",
     "baseline": "baseline_book_synthetic_v1",
     "metric": "total_baselines",
     "filed_value": 17,
     "reason": ""},
    {"id": "flake_discipline_deterministic_count",
     "source": "docs/architecture/FLAKE_DISCIPLINE.md",
     "marker": "Deterministic baselines: 17",
     "claim": "Flake discipline: 17 deterministic "
              "baselines (the rendered filing's own "
              "header)",
     "disposition": "matched",
     "baseline": "flake_discipline_synthetic_v1",
     "metric": "deterministic_baselines",
     "filed_value": 17,
     "reason": ""},
    {"id": "model_card_citation_accuracy_target",
     "source": "MODEL_CARD.md",
     "marker": "Citation accuracy",
     "claim": "MODEL_CARD target table: 'Citation "
              "accuracy ≥ 85%'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "filed as a target only, never as a "
               "result: the only measured "
               "citation_accuracy in the estate is "
               "the empty-store baseline's 0.0 "
               "(profile_benchmark_empty_store_v1, "
               "target 0.85 filed unmet) — no suite "
               "result supports the claim as achieved"},
    {"id": "model_card_enforcement_pass_target",
     "source": "MODEL_CARD.md",
     "marker": "Enforcement pass rate",
     "claim": "MODEL_CARD target table: 'Enforcement "
              "pass rate ≥ 90%'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "filed as a target only: the only "
               "measured enforcement_pass_rate is "
               "the empty-store baseline's 0.0 "
               "(target 0.90 filed unmet) — no suite "
               "result supports it as achieved"},
    {"id": "model_card_avg_response_words_target",
     "source": "MODEL_CARD.md",
     "marker": "Avg response words",
     "claim": "MODEL_CARD target table: 'Avg "
              "response words ≥ 30'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "filed as a target only: the measured "
               "avg_response_words is 9.0 on the "
               "empty-store baseline (target 30.0 "
               "filed unmet, D-BENCH-2 / TF-121) — "
               "no suite result supports it as "
               "achieved"},
    {"id": "model_card_persona_100_plus_turns",
     "source": "MODEL_CARD.md",
     "marker": "100+ turns",
     "claim": "MODEL_CARD: 'Persona consistency "
              "≥ 0.75 — PersonaConsistencyScorer "
              "across 100+ turns'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "D-BENCH-4: no PersonaConsistencyScorer "
               "run over 100+ turns exists — the "
               "scorer's own pin is 6 turns at "
               "0.7167 (below 0.75); the 108-turn run "
               "that does exist is TF-122's persona "
               "SUITE, a different instrument, and "
               "this doc does not point at it"},
    {"id": "build_plan_persona_100_plus_turns",
     "source": "BUILD_PLAN_v1.md",
     "marker": "100+ turns",
     "claim": "BUILD_PLAN_v1: 'Personality stable "
              "across 100+ turns on any hardware "
              "tier' (and its 100+ turn test "
              "milestone)",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "D-BENCH-4: the wording points at no "
               "run; no per-hardware-tier persona run "
               "exists (live per-profile behaviour is "
               "NOT PROVEN, TF-173) — the suite's "
               "108-turn fixture run is not a "
               "hardware-tier result"},
    {"id": "build_plan_citation_accuracy_85",
     "source": "BUILD_PLAN_v1.md",
     "marker": ">85%",
     "claim": "BUILD_PLAN_v1: 'Benchmarks per "
              "hardware profile: citation accuracy "
              "(>85%)'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "no per-profile citation-accuracy "
               "result exists: TF-128's profile panel "
               "files suite scores and bench average "
               "words per profile, no citation "
               "accuracy — the claim has no suite "
               "result behind it"},
    {"id": "task_phase6_persona_100_plus_turns",
     "source": "TASK_PHASE6_RELEASE.md",
     "marker": "100+ turns",
     "claim": "TASK_PHASE6: PersonaConsistencyScorer "
              "'Designed to validate the Skald "
              "persona across 100+ turns'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "D-BENCH-4: the same standing wording "
               "— no such scorer run exists (see "
               "model_card_persona_100_plus_turns)"},
    {"id": "task_phase6_locust_httpuser",
     "source": "TASK_PHASE6_RELEASE.md",
     "marker": "HttpUser",
     "claim": "TASK_PHASE6: 'ThoughtForgeUser"
              "(HttpUser) — Locust load test user'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "D-BENCH-5: stale description — the "
               "locust file is no-HTTP by its own "
               "header, unrunnable without the dev "
               "extra, and TF-121 excluded it from "
               "the harness panel by verdict as a "
               "manual throughput script"},
    {"id": "model_card_edge_inference_speed",
     "source": "MODEL_CARD.md",
     "marker": "minutes per response",
     "claim": "MODEL_CARD: 'Pi Zero and phone_low "
              "profiles are slow (minutes per "
              "response without model optimization)'",
     "disposition": "struck",
     "baseline": None, "metric": None,
     "filed_value": None,
     "reason": "a live-model speed claim with no "
               "live run behind it: no live model "
               "runs on this VM (TF-173) and latency "
               "is never a pinned baseline (TF-121) "
               "— no suite result supports any "
               "per-response time"},
)


def _pins() -> dict[str, dict[str, Any]]:
    """The harness pins, by baseline id — the pure
    validator's default comparison set."""
    return {b["id"]: dict(b["metrics"])
            for b in _harness.BASELINES}


# ── The pure validator (probes run here) ────────────────────────────────────

def claim_problems(
    row: dict[str, Any], pins: Any = None,
) -> list[str]:
    """Judge ONE claim row against its own parts and
    the pins, purely: it must file an id, a source,
    a marker and the claim itself; a matched row
    must file a baseline, a metric and a value that
    the pin carries exactly; a struck row must file
    a reason and must not carry a value that a pin
    actually supports. Pure — an empty or partial
    row is named, never crashed on."""
    pins = _pins() if pins is None else pins
    problems: list[str] = []
    label = row.get("id") or "<unnamed claim>"
    if not row.get("id"):
        problems.append("a claim files no id")
    if not row.get("source"):
        problems.append(
            f"{label}: the claim files no source — "
            f"a claim that cannot name its doc is "
            f"not a census row")
    if not row.get("marker"):
        problems.append(
            f"{label}: the claim files no source "
            f"marker — its wording cannot be found "
            f"in its source doc and checked")
    if not row.get("claim"):
        problems.append(
            f"{label}: the claim files no claim text")
    disposition = row.get("disposition")
    if disposition == "matched":
        baseline = row.get("baseline")
        metric = row.get("metric")
        if not baseline:
            problems.append(
                f"{label}: a matched claim files no "
                f"baseline — matched means matched TO "
                f"a suite result")
        if not metric:
            problems.append(
                f"{label}: a matched claim files no "
                f"metric — the suite result it matches "
                f"must be named")
        if baseline and metric:
            pin = (pins.get(baseline) or {}).get(metric)
            if pin is None:
                problems.append(
                    f"{label}: its baseline / metric "
                    f"({baseline} / {metric}) is carried "
                    f"by no harness pin")
            elif row.get("filed_value") != pin:
                problems.append(
                    f"{label}: its filed value "
                    f"{row.get('filed_value')!r} != the "
                    f"pinned {baseline}.{metric} "
                    f"{pin!r} — a matched claim with a "
                    f"wrong value is named, never "
                    f"carried")
        if row.get("filed_value") is None:
            problems.append(
                f"{label}: a matched claim files no "
                f"value")
    elif disposition == "struck":
        if not row.get("reason"):
            problems.append(
                f"{label}: a struck claim files no "
                f"reason — striking files its reason, "
                f"never a bare deletion")
        baseline = row.get("baseline")
        metric = row.get("metric")
        if baseline and metric \
                and row.get("filed_value") is not None:
            pin = (pins.get(baseline) or {}).get(metric)
            if pin is not None \
                    and row.get("filed_value") == pin:
                problems.append(
                    f"{label}: filed struck but its "
                    f"value matches the pinned "
                    f"{baseline}.{metric} {pin!r} — a "
                    f"claim a suite result supports "
                    f"is matched, not struck")
    else:
        problems.append(
            f"{label}: disposition {disposition!r} is "
            f"not a filed disposition — the claim is "
            f"neither matched nor reasoned-struck: "
            f"an unclassified claim, named")
    return problems


# ── The filed probes ────────────────────────────────────────────────────────

def _probe_row(**overrides: Any) -> dict[str, Any]:
    row = copy.deepcopy(dict(CLAIM_ROWS[0])) if CLAIM_ROWS \
        else {
            "id": "probe_matched", "source": "MODEL_CARD.md",
            "marker": "x", "claim": "probe",
            "disposition": "matched",
            "baseline": "persona_suite_synthetic_v1",
            "metric": "overall_score",
            "filed_value": 0.8453703703703704,
            "reason": ""}
    row.update(overrides)
    return row


def _probe_struck(**overrides: Any) -> dict[str, Any]:
    struck = [r for r in CLAIM_ROWS
              if r["disposition"] == "struck"]
    row = copy.deepcopy(dict(struck[0])) if struck else {
        "id": "probe_struck", "source": "MODEL_CARD.md",
        "marker": "x", "claim": "probe",
        "disposition": "struck", "baseline": None,
        "metric": None, "filed_value": None,
        "reason": "no suite result supports it"}
    row.update(overrides)
    return row


CLAIM_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_matched_claim_passes",
     "kind": "matched", "overrides": {},
     "expect_problem": False,
     "detail": "a filed matched claim whose "
               "baseline, metric and value agree "
               "with its pin files no problem"},
    {"id": "clean_struck_claim_passes",
     "kind": "struck", "overrides": {},
     "expect_problem": False,
     "detail": "a filed struck claim with its "
               "reason filed files no problem"},
    {"id": "matched_claim_with_wrong_value_is_named",
     "kind": "matched",
     "overrides": {"filed_value": 0.5},
     "expect_problem": True,
     "detail": "a matched claim whose filed value "
               "differs from its pin is named — the "
               "audit never carries a drifted value"},
    {"id": "struck_claim_that_actually_matches_is_named",
     "kind": "struck",
     "overrides": {
         "baseline": "persona_suite_synthetic_v1",
         "metric": "overall_score",
         "filed_value": 0.8453703703703704},
     "expect_problem": True,
     "detail": "a claim filed struck while carrying "
               "a value a pin supports is named — "
               "supported claims are matched, not "
               "struck"},
    {"id": "claim_missing_source_is_named",
     "kind": "matched",
     "overrides": {"source": ""},
     "expect_problem": True,
     "detail": "a claim that cannot name its source "
               "doc is named"},
    {"id": "matched_claim_missing_metric_is_named",
     "kind": "matched",
     "overrides": {"metric": None},
     "expect_problem": True,
     "detail": "a matched claim naming no metric "
               "matches nothing and is named"},
    {"id": "empty_claim_is_named_never_crashed",
     "kind": "empty", "overrides": {},
     "expect_problem": True,
     "detail": "an empty claim row is named by the "
               "pure validator, never crashed on "
               "(the TF-130 / TF-132 class)"},
)


def run_claim_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validator; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in CLAIM_PROBES:
        if probe["kind"] == "matched":
            problems = claim_problems(
                _probe_row(**probe["overrides"]))
        elif probe["kind"] == "struck":
            problems = claim_problems(
                _probe_struck(**probe["overrides"]))
        else:  # empty
            problems = claim_problems({})
        passed = bool(problems) == probe["expect_problem"]
        results.append({
            "id": probe["id"], "passed": passed,
            "problems": problems,
            "detail": probe["detail"]})
    return results


# ── The audit run ───────────────────────────────────────────────────────────

@dataclass
class PublicClaimsResult:
    """One full audit run: the rows it filed and
    the verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.rows)

    @property
    def matched_claims(self) -> int:
        return sum(1 for r in self.rows
                   if r["disposition"] == "matched")

    @property
    def struck_claims(self) -> int:
        return sum(1 for r in self.rows
                   if r["disposition"] == "struck")

    @property
    def distinct_sources(self) -> int:
        return len({r["source"] for r in self.rows})

    @property
    def overall_score(self) -> float:
        if not self.rows:
            return 0.0
        return self.matched_claims / len(self.rows)

    @property
    def verdict(self) -> str:
        """PASS only on the full census with every
        claim validly classified; REVIEW for any
        partial or empty selection, however it
        scores."""
        if [r["id"] for r in self.rows] != [
                r["id"] for r in CLAIM_ROWS]:
            return "REVIEW"
        clean = all(not claim_problems(r) for r in self.rows)
        return "PASS" if clean else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_public_claims(
    claim_ids: Any = None,
) -> PublicClaimsResult:
    """File the audit over the selection (the full
    census by default).

    Any explicit claim selection may be given —
    including an empty one, which files 0 claims /
    REVIEW (no vacuous pass).
    """
    rows = CLAIM_ROWS
    if claim_ids is not None:
        wanted = tuple(claim_ids)
        rows = tuple(r for r in CLAIM_ROWS if r["id"] in wanted)
    return PublicClaimsResult(rows=[dict(r) for r in rows])


# ── The readable render (the doc, verbatim) ─────────────────────────────────

def render_public_claims_markdown(
    rows: Any = None,
) -> str:
    """Render the audit deterministically — the
    exact text filed as
    docs/architecture/PUBLIC_CLAIMS_AUDIT.md: the
    census (matched rows with their suite results,
    struck rows with their reasons) and the rules.
    Every number comes from the filed data; nothing
    is hand-written here but the frame."""
    rows = CLAIM_ROWS if rows is None else tuple(rows)
    matched = [r for r in rows if r["disposition"] == "matched"]
    struck = [r for r in rows if r["disposition"] == "struck"]
    lines: list[str] = []
    lines.append("# Public-Claims Audit — TF-136")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`benchmarks/public_claims.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m benchmarks.public_claims`. "
        "The checker (`public_claims_problems()`) "
        "recomputes every matched value live through "
        "the harness measurers and fails on any "
        "drift from the filed rows.")
    lines.append("")
    lines.append(f"Format: `{PUBLIC_CLAIMS_FORMAT}`")
    lines.append(f"Claims censused: {len(rows)}")
    lines.append(f"Matched claims: {len(matched)}")
    lines.append(f"Struck claims: {len(struck)}")
    lines.append(
        f"Distinct source docs: "
        f"{len({r['source'] for r in rows})}")
    lines.append("")
    lines.append(
        "Every performance claim in the repo's docs "
        "that purports to state a suite / harness "
        "result or a public performance target is "
        "censused here: matched to a suite result "
        "(baseline + metric + the value, recomputed "
        "live) or struck with its reason filed. "
        "Struck claims stay in their source docs — "
        "the audit strikes them in this ledger only, "
        "and never edits a source doc.")
    lines.append("")
    lines.append("## Matched claims (suite result behind each)")
    lines.append("")
    lines.append(
        "| Claim | Source | Baseline | Metric | Filed value |")
    lines.append("|---|---|---|---|---|")
    for row in matched:
        lines.append(
            f"| `{row['id']}` | `{row['source']}` | "
            f"`{row['baseline']}` | `{row['metric']}` | "
            f"{row['filed_value']!r} |")
    lines.append("")
    lines.append("## Struck claims (reason filed for each)")
    lines.append("")
    for row in struck:
        lines.append(f"### Struck: `{row['id']}`")
        lines.append("")
        lines.append(f"- Source: `{row['source']}`")
        lines.append(f"- Claim: {row['claim']}")
        lines.append(f"- Reason: {row['reason']}")
        lines.append("")
    lines.append("## The rules")
    lines.append("")
    for rule in PUBLIC_CLAIMS_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## Filed limits")
    lines.append("")
    lines.append(
        "- The census covers performance claims that "
        "purport to state a suite / harness result or "
        "a public performance target. Per-slice "
        "device filings owned and checked by their "
        "own slice modules are not re-judged here; "
        "vague capability prose files no claim.")
    lines.append(
        "- A matched claim is matched to a pinned "
        "FIXTURE baseline — scripted synthetic suites "
        "through the real scorers on this VM. No live "
        "model stands behind any matched value; live "
        "behaviour is NOT PROVEN (TF-173).")
    lines.append(
        "- A struck claim is struck from the ledger "
        "of claims the repo's suites support; the "
        "audit does not edit its source doc.")
    lines.append("")
    lines.append(
        "Proof grade: FIXTURE — pinned "
        "synthetic-fixture baselines re-measured live "
        "on this VM; no live model, never claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_public_claims_doc() -> Path:
    """Regenerate the filed doc by command: render
    from the filed data and write the doc, byte for
    byte. Returns the doc's path."""
    doc = _REPO_ROOT / PUBLIC_CLAIMS_DOC
    doc.write_text(
        render_public_claims_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m benchmarks.public_claims`."""
    doc = write_public_claims_doc()
    print(
        f"public-claims audit regenerated: "
        f"{len(CLAIM_ROWS)} claims -> {PUBLIC_CLAIMS_DOC}")
    return 0


# ── The live checker ────────────────────────────────────────────────────────

def public_claims_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: format, rules.
    if PUBLIC_CLAIMS_FORMAT != "thoughtforge-public-claims/1":
        problems.append(
            f"PUBLIC_CLAIMS_FORMAT drifted: "
            f"{PUBLIC_CLAIMS_FORMAT}")
    rule_ids = [r["id"] for r in PUBLIC_CLAIMS_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate PUBLIC_CLAIMS_RULES ids")
    for needed in (
            "every_public_performance_claim_is_censused",
            "matched_means_a_live_verified_suite_value",
            "struck_means_a_filed_reason_never_a_deletion",
            "targets_are_not_results",
            "probes_prove_the_validator_can_fail",
            "pass_needs_the_full_census_classified",
            "the_public_claims_baseline_is_not_gated_by_tf129",
            "no_new_benchmark_input_was_added"):
        if needed not in rule_ids:
            problems.append(f"PUBLIC_CLAIMS_RULES lost {needed}")

    # Census shape: unique ids, every row valid
    # against the pins (purely).
    claim_ids = [r["id"] for r in CLAIM_ROWS]
    if len(claim_ids) != len(set(claim_ids)):
        problems.append("duplicate CLAIM_ROWS ids")
    for row in CLAIM_ROWS:
        problems.extend(claim_problems(row))

    # Every source doc exists and carries its
    # marker verbatim; every matched value
    # recomputes live through the harness measurer.
    for row in CLAIM_ROWS:
        source = _REPO_ROOT / row["source"]
        if not source.exists():
            problems.append(
                f"{row['id']}: its source doc "
                f"{row['source']} is missing")
        elif row["marker"] not in source.read_text(
                encoding="utf-8"):
            problems.append(
                f"{row['id']}: its marker "
                f"{row['marker']!r} no longer stands "
                f"in {row['source']} — the claim's "
                f"wording moved; re-census deliberately")
        if row["disposition"] == "matched":
            measurer = _harness._MEASURERS.get(row["baseline"])
            if measurer is None:
                problems.append(
                    f"{row['id']}: baseline "
                    f"{row['baseline']} has no harness "
                    f"measurer")
                continue
            try:
                measured = measurer()
            except Exception as exc:
                problems.append(
                    f"{row['id']}: live re-measurement "
                    f"of {row['baseline']} failed: {exc!r}")
                continue
            got = measured.get(row["metric"])
            pinned = row["filed_value"]
            if isinstance(pinned, float) \
                    or isinstance(got, float):
                drift = got is None or abs(
                    float(got) - float(pinned)) > 1e-9
            else:
                drift = got != pinned
            if drift:
                problems.append(
                    f"{row['id']}: its filed value "
                    f"{pinned!r} drifted from the live "
                    f"recomputation of "
                    f"{row['baseline']}.{row['metric']}: "
                    f"{got!r}")

    # A fresh full run files PASS; partial + empty
    # selections REVIEW.
    result = run_public_claims()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live audit does not file PASS on "
            f"this tree: {result.verdict} "
            f"({result.matched_claims} matched / "
            f"{result.struck_claims} struck of "
            f"{result.total_claims})")
    empty = run_public_claims(claim_ids=())
    if empty.total_claims != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    subset = run_public_claims(
        claim_ids=("persona_suite_overall_score",))
    if subset.total_claims != 1 or subset.verdict != "REVIEW":
        problems.append(
            "a single-claim selection does not file "
            "its claim / REVIEW mechanically")

    # Determinism of the render + the filed doc is
    # the render, verbatim.
    if render_public_claims_markdown() != \
            render_public_claims_markdown(CLAIM_ROWS):
        problems.append(
            "the render is not deterministic: two "
            "renders over the filed data differ")
    doc = _REPO_ROOT / PUBLIC_CLAIMS_DOC
    if not doc.exists():
        problems.append(
            f"the write-up {PUBLIC_CLAIMS_DOC} is "
            f"missing (regenerate with python3 -m "
            f"benchmarks.public_claims)")
    elif doc.read_text(encoding="utf-8") != \
            render_public_claims_markdown():
        problems.append(
            f"the filed doc {PUBLIC_CLAIMS_DOC} is "
            f"not the render verbatim (regenerate "
            f"with python3 -m benchmarks.public_claims)")

    # The probes all behave.
    probes = run_claim_probes()
    if len(probes) != len(CLAIM_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the claim probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included), and the gate's gated set untouched.
    from benchmarks.harness import (
        BASELINES,
        BENCHMARKS,
        baseline_problems,
    )
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "public_claims" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the public_claims entry")
    if PUBLIC_CLAIMS_BASELINE_ID not in [
            b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the public-claims baseline")
    if PUBLIC_CLAIMS_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the public-claims baseline entered "
            "TF-129's gated set (it is judged by this "
            "checker, not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the "
            f"public-claims audit: {earlier}")
    return problems


def public_claims_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_claim_probes()
    result = run_public_claims()
    return {
        "format": PUBLIC_CLAIMS_FORMAT,
        "doc": PUBLIC_CLAIMS_DOC,
        "claim_rows": [dict(r) for r in CLAIM_ROWS],
        "measured": {
            "total_claims": result.total_claims,
            "matched_claims": result.matched_claims,
            "struck_claims": result.struck_claims,
            "distinct_sources": result.distinct_sources,
            "live_verified_matched": result.matched_claims,
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
            "verdict": result.verdict,
        },
        "probes": probes,
        "rules": [dict(r) for r in PUBLIC_CLAIMS_RULES],
    }


__all__ = [
    "CLAIM_PROBES",
    "CLAIM_ROWS",
    "PUBLIC_CLAIMS_BASELINE_ID",
    "PUBLIC_CLAIMS_DOC",
    "PUBLIC_CLAIMS_FORMAT",
    "PUBLIC_CLAIMS_RULES",
    "PublicClaimsResult",
    "claim_problems",
    "main",
    "public_claims_as_data",
    "public_claims_problems",
    "render_public_claims_markdown",
    "run_claim_probes",
    "run_public_claims",
    "write_public_claims_doc",
]

if __name__ == "__main__":
    sys.exit(main())
