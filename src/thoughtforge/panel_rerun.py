"""
TF-151 — panel re-run: TF-132 + persona suites
re-measured with the verdicts landed; deltas
published.

TF-151's roadmap text: *Panel re-run: TF-132 +
persona suites re-measured with the verdicts
landed; deltas published.*

The measured start, probed live on this tree
before anything was built (runners only):

  * TF-132's WYRD panel re-runs PASS: 18 cases /
    9 consistent / 8 contradicted / 1
    honest-unanswerable, overall 10/18 =
    0.5555555555555556, probes 7/7,
    ``wyrd_panel_problems()`` [] — every value
    identical to its pinned harness baseline
    ``wyrd_panel_synthetic_v1``.
  * TF-122's persona suite re-runs PASS: 108
    turns, identity 0.8333333333333334 / voice
    0.8694444444444445 / values
    0.8333333333333334 / overall
    0.8453703703703704, ``persona_suite_problems()``
    [] — identical to ``persona_suite_synthetic_v1``.
  * The persona scorer underneath TF-122's voice
    axis (the TF-121 harness baseline
    ``persona_consistency_synthetic_v1``)
    re-measures identically: consistency
    0.7166666666666667 over 6 turns (flagged 2,
    generic 7, norse 10, citation 1, skald 5).
  * ``baseline_problems()`` [] over all pinned
    baselines.
  * **Coupling, measured not assumed**: none of
    the three re-run measurers consumes any
    Part VIII adopted code. A source inspection
    of ``benchmarks/wyrd_panel.py``,
    ``benchmarks/persona_suite.py`` and
    ``benchmarks/persona_consistency.py`` finds
    no reference to the bond graph, the persona
    compiler, the bond bridge or the Runmal
    cards (no ``bond_graph`` / ``persona_compiler``
    / ``BondGraph`` / ``CompiledPersona`` /
    ``bond_bridge`` / ``runmal`` token).
    ``persona_suite`` binds only TF-030's
    pre-existing ``thoughtforge.cognition.persona``
    expectation and the persona scorer — both
    predate Part VIII. The zero deltas below are
    therefore the honest result: the verdicts
    landed in OTHER modules, and these panels do
    not read them. No improvement is fabricated.

**What this slice files** (literal data below):

  * ``RERUN_ROWS`` — the three re-run panels
    (3): TF-132's WYRD panel, TF-122's persona
    suite, and the persona-consistency scorer
    baseline TF-122's voice axis consumes.
  * ``DELTA_ROWS`` — every re-measured metric
    (19): its pinned value, its re-measured
    value, the computed delta, and its outcome
    (``unmoved`` for all 19 — proven live).
  * ``VERDICT_ROWS`` — the Part VIII verdicts
    in force at the re-run (7): TF-139
    ADOPT_AS_LAYER, TF-140 RETIRE, TF-141 MERGE,
    TF-142 RETIRE, TF-143 adoptions implemented,
    TF-146 bond bridge, TF-147 Runmal cards —
    each filing whether any re-run panel
    consumes it (none does, with the reason).
  * 7 filed probes through the pure validators.

**Harness coupling, in the open (the TF-138
gate / doc+checker precedent)**: this slice adds
NO new harness baseline and re-pins none. The
re-run's whole point is to compare live values
against the EXISTING pins; a new baseline would
move the baseline book, the flake census and
the public-claims values for no measurement
reason and would obscure the delta it exists to
publish. The filing is therefore module +
checker + doc + tests only, exactly TF-138's
shape for a re-measurement slice. The pinned
baselines stand exactly as filed and
``baseline_problems()`` stays clean.

**Docket**: none. TF-151 has no share of any
TF-016 entry and finds no new defect; it does
NOT edit the docket. TF-016 totals stand
unchanged (137 — 31 OPEN / 1 OPEN-ENVIRONMENT /
101 FIXED / 4 CLOSED-BY-VERDICT).

Proof grade: FIXTURE/STATIC — pinned
synthetic-fixture baselines re-measured live on
this VM; no live model was read — live
behaviour is NOT PROVEN (TF-173 class), never
claimed. No canon / persona content is filed:
only the estate's synthetic fixture scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PANEL_RERUN_FORMAT = "thoughtforge-panel-rerun/1"
PANEL_RERUN_DOC = "docs/architecture/PANEL_RERUN.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: Outcomes a delta row may file.
DELTA_OUTCOMES: tuple[str, ...] = ("unmoved", "moved")

#: Tokens naming Part VIII adopted code. None may
#: appear in a re-run measurer's source — that
#: absence, checked live, is the coupling proof.
ADOPTED_CODE_TOKENS: tuple[str, ...] = (
    "bond_graph", "persona_compiler", "BondGraph",
    "CompiledPersona", "bond_bridge", "runmal",
)


# ── The re-run panels, as literal data ───────────────────────────────────────

RERUN_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "wyrd_panel",
     "slice": "TF-132",
     "baseline_id": "wyrd_panel_synthetic_v1",
     "measurer": "benchmarks.harness.measure_wyrd_panel_baseline "
                 "/ benchmarks.wyrd_panel.run_wyrd_panel",
     "checker": "benchmarks.wyrd_panel.wyrd_panel_problems",
     "source_file": "benchmarks/wyrd_panel.py",
     "verdict_before": "PASS",
     "verdict_after": "PASS",
     "consumes_adopted_code": False},
    {"id": "persona_suite",
     "slice": "TF-122",
     "baseline_id": "persona_suite_synthetic_v1",
     "measurer": "benchmarks.harness.measure_persona_suite_baseline "
                 "/ benchmarks.persona_suite.run_persona_suite",
     "checker": "benchmarks.persona_suite.persona_suite_problems",
     "source_file": "benchmarks/persona_suite.py",
     "verdict_before": "PASS",
     "verdict_after": "PASS",
     "consumes_adopted_code": False},
    {"id": "persona_consistency",
     "slice": "TF-121",
     "baseline_id": "persona_consistency_synthetic_v1",
     "measurer": "benchmarks.harness.measure_persona_baseline",
     "checker": "benchmarks.harness.baseline_problems",
     "source_file": "benchmarks/persona_consistency.py",
     "verdict_before": "PASS",
     "verdict_after": "PASS",
     "consumes_adopted_code": False},
)


# ── The verdicts in force, as literal data ───────────────────────────────────

VERDICT_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "tf139_bond_graph", "slice": "TF-139",
     "verdict": "ADOPT_AS_LAYER",
     "adopted_code": "src/thoughtforge/knowledge/bond_graph.py",
     "consumed_by_rerun": False,
     "reason": "no re-run panel measures bonds; the WYRD panel "
               "judges world-state claims and the persona suites "
               "judge persona turns — neither imports the bond layer"},
    {"id": "tf140_micro_rag", "slice": "TF-140",
     "verdict": "RETIRE",
     "adopted_code": "",
     "consumed_by_rerun": False,
     "reason": "a retirement adopts no code; no re-run panel "
               "ever used the micro-RAG retriever"},
    {"id": "tf141_persona_compiler", "slice": "TF-141",
     "verdict": "MERGE",
     "adopted_code": "src/thoughtforge/cognition/persona_compiler.py",
     "consumed_by_rerun": False,
     "reason": "the persona suites bind TF-030's PersonaExpectation "
               "and the phrase scorer directly; neither imports "
               "the compiled expectation TF-143 landed"},
    {"id": "tf142_truth_calibrator", "slice": "TF-142",
     "verdict": "RETIRE",
     "adopted_code": "",
     "consumed_by_rerun": False,
     "reason": "a retirement adopts no code; no re-run panel "
               "ever used the truth calibrator"},
    {"id": "tf143_adoptions", "slice": "TF-143",
     "verdict": "ADOPTIONS_IMPLEMENTED",
     "adopted_code": "src/thoughtforge/knowledge/bond_graph.py + "
                     "src/thoughtforge/cognition/persona_compiler.py",
     "consumed_by_rerun": False,
     "reason": "TF-143 landed the two adoptions as production "
               "modules beside the panels; the panels' imports "
               "did not change, so their measurements cannot move"},
    {"id": "tf146_bond_bridge", "slice": "TF-146",
     "verdict": "BRIDGE_BUILT",
     "adopted_code": "src/thoughtforge/knowledge/bond_bridge.py",
     "consumed_by_rerun": False,
     "reason": "the bridge exports bond data for WYRD consumers; "
               "no re-run panel consumes bond exports"},
    {"id": "tf147_runmal_cards", "slice": "TF-147",
     "verdict": "CARDS_BUILT",
     "adopted_code": "src/thoughtforge/cognition/runmal_cards.py",
     "consumed_by_rerun": False,
     "reason": "the cards export compiled personae; the persona "
               "suites score scripted turns, not card exports"},
)


# ── The deltas, as literal data ──────────────────────────────────────────────
#
# pinned == the harness baseline's filed value;
# remeasured == the value the runner produced
# live on this tree with the verdicts landed;
# delta == remeasured - pinned, computed.

def _d(panel: str, metric: str, pinned: Any, remeasured: Any) -> dict[str, Any]:
    delta = remeasured - pinned
    return {"panel": panel, "metric": metric, "pinned": pinned,
            "remeasured": remeasured, "delta": delta,
            "outcome": "unmoved" if delta == 0 else "moved"}


DELTA_ROWS: tuple[dict[str, Any], ...] = (
    _d("wyrd_panel", "total_cases", 18, 18),
    _d("wyrd_panel", "consistent_cases", 9, 9),
    _d("wyrd_panel", "contradicted_cases", 8, 8),
    _d("wyrd_panel", "honest_unanswerable_cases", 1, 1),
    _d("wyrd_panel", "overall_score", 0.5555555555555556, 0.5555555555555556),
    _d("wyrd_panel", "probes_passed", 7, 7),
    _d("wyrd_panel", "probes_total", 7, 7),
    _d("persona_suite", "identity_score", 0.8333333333333334, 0.8333333333333334),
    _d("persona_suite", "voice_score", 0.8694444444444445, 0.8694444444444445),
    _d("persona_suite", "values_score", 0.8333333333333334, 0.8333333333333334),
    _d("persona_suite", "overall_score", 0.8453703703703704, 0.8453703703703704),
    _d("persona_suite", "total_turns", 108, 108),
    _d("persona_consistency", "consistency_score", 0.7166666666666667, 0.7166666666666667),
    _d("persona_consistency", "total_turns", 6, 6),
    _d("persona_consistency", "flagged_turns", 2, 2),
    _d("persona_consistency", "generic_phrase_hits", 7, 7),
    _d("persona_consistency", "norse_tone_hits", 10, 10),
    _d("persona_consistency", "citation_turns", 1, 1),
    _d("persona_consistency", "skald_quality_hits", 5, 5),
)


# ── Pure validators (the probes run against these) ───────────────────────────

def rerun_row_problems(row: Any) -> list[str]:
    """Judge one re-run panel filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["rerun: the row is empty"]
    problems: list[str] = []
    for key in ("id", "slice", "baseline_id", "measurer",
                "checker", "source_file"):
        if not row.get(key):
            problems.append(f"rerun {row.get('id')}: no {key} filed")
    if not isinstance(row.get("consumes_adopted_code"), bool):
        problems.append(
            f"rerun {row.get('id')}: consumption flag is not a boolean")
    if row.get("consumes_adopted_code"):
        problems.append(
            f"rerun {row.get('id')}: files that it consumes adopted "
            "code — measured live, no re-run panel does; a panel "
            "that did would need its delta explained, not filed here")
    if row.get("verdict_before") != "PASS" or row.get("verdict_after") != "PASS":
        problems.append(
            f"rerun {row.get('id')}: verdicts must file PASS/PASS — "
            "both panels passed at filing and at re-run")
    return problems


def verdict_row_problems(row: Any) -> list[str]:
    """Judge one verdict filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["verdict: the row is empty"]
    problems: list[str] = []
    for key in ("id", "slice", "verdict", "reason"):
        if not row.get(key):
            problems.append(f"verdict {row.get('id')}: no {key} filed")
    if row.get("consumed_by_rerun") is not False:
        problems.append(
            f"verdict {row.get('id')}: files consumption by a re-run "
            "panel — measured live, none consumes it")
    return problems


def delta_row_problems(row: Any) -> list[str]:
    """Judge one delta filing, purely: the delta
    must be the arithmetic difference and the
    outcome must agree with it. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["delta: the row is empty"]
    problems: list[str] = []
    if not row.get("panel") or not row.get("metric"):
        problems.append("delta: the row files no panel/metric")
    if row.get("outcome") not in DELTA_OUTCOMES:
        problems.append(
            f"delta {row.get('panel')}.{row.get('metric')}: outcome "
            f"is {row.get('outcome')!r}, not one of {DELTA_OUTCOMES}")
    try:
        expected_delta = row["remeasured"] - row["pinned"]
    except Exception:
        return problems + [
            f"delta {row.get('panel')}.{row.get('metric')}: pinned / "
            "remeasured are not comparable numbers"]
    if row.get("delta") != expected_delta:
        problems.append(
            f"delta {row.get('panel')}.{row.get('metric')}: filed "
            f"delta {row.get('delta')!r} is not remeasured - pinned "
            f"({expected_delta!r})")
    if expected_delta == 0 and row.get("outcome") != "unmoved":
        problems.append(
            f"delta {row.get('panel')}.{row.get('metric')}: a zero "
            "delta must file unmoved")
    if expected_delta != 0 and row.get("outcome") != "moved":
        problems.append(
            f"delta {row.get('panel')}.{row.get('metric')}: a "
            "non-zero delta must file moved")
    return problems


def rerun_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured re-run state against its
    expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["rerun state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["rerun state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"rerun state: {key} measures {state.get(key)!r}, "
                f"not the filed {value!r}")
    return problems


def doc_text_problems(text: Any, render: Any, label: Any) -> list[str]:
    """Judge a filed doc against its render,
    purely. Empty-safe."""
    if not isinstance(render, str) or not render:
        return ["doc: the render is empty"]
    if not isinstance(text, str) or not text:
        return [f"doc: the {label} text is empty"]
    if text != render:
        return [f"doc: the {label} has drifted from its render"]
    return []


# ── The live re-run state ────────────────────────────────────────────────────

RERUN_EXPECTED: dict[str, Any] = {
    "panels": 3,
    "delta_rows": 19,
    "moved_rows": 0,
    "unmoved_rows": 19,
    "verdict_rows": 7,
    "verdicts_consumed": 0,
    "wyrd_total_cases": 18,
    "wyrd_overall_score": 0.5555555555555556,
    "wyrd_verdict": "PASS",
    "persona_total_turns": 108,
    "persona_overall_score": 0.8453703703703704,
    "persona_verdict": "PASS",
    "persona_consistency_score": 0.7166666666666667,
}


def _live_measurements() -> dict[str, dict[str, Any]]:
    from benchmarks.harness import (
        measure_persona_baseline,
        measure_persona_suite_baseline,
        measure_wyrd_panel_baseline,
    )
    return {
        "wyrd_panel": measure_wyrd_panel_baseline(),
        "persona_suite": measure_persona_suite_baseline(),
        "persona_consistency": measure_persona_baseline(),
    }


def run_rerun_state() -> dict[str, Any]:
    """Measure the re-run live: the harness
    measurers for the three panels plus their
    verdicts. Deterministic values only."""
    from benchmarks.persona_suite import run_persona_suite
    from benchmarks.wyrd_panel import run_wyrd_panel

    return {
        "panels": len(RERUN_ROWS),
        "delta_rows": len(DELTA_ROWS),
        "moved_rows": sum(1 for r in DELTA_ROWS if r["outcome"] == "moved"),
        "unmoved_rows": sum(1 for r in DELTA_ROWS if r["outcome"] == "unmoved"),
        "verdict_rows": len(VERDICT_ROWS),
        "verdicts_consumed": sum(
            1 for r in VERDICT_ROWS if r["consumed_by_rerun"]),
        "wyrd_total_cases": run_wyrd_panel().total_cases,
        "wyrd_overall_score": run_wyrd_panel().overall_score,
        "wyrd_verdict": run_wyrd_panel().verdict,
        "persona_total_turns": run_persona_suite().total_turns,
        "persona_overall_score": run_persona_suite().overall_score,
        "persona_verdict": run_persona_suite().verdict,
        "persona_consistency_score": _live_measurements()[
            "persona_consistency"]["consistency_score"],
    }


# ── The filed probes ─────────────────────────────────────────────────────────

RERUN_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_rerun_passes", "kind": "clean",
     "detail": "the filed rerun + verdict + delta rows pass "
               "their validators"},
    {"id": "moved_delta_filed_unmoved_is_named", "kind": "moved_unmoved",
     "detail": "a delta row whose remeasured value moved but "
               "which files unmoved is named"},
    {"id": "wrong_delta_arithmetic_is_named", "kind": "wrong_delta",
     "detail": "a delta row whose filed delta is not "
               "remeasured - pinned is named"},
    {"id": "panel_claiming_consumption_is_named", "kind": "false_consumption",
     "detail": "a rerun row claiming it consumes adopted code "
               "is named — no re-run panel does"},
    {"id": "verdict_claiming_consumption_is_named", "kind": "verdict_consumption",
     "detail": "a verdict row claiming a re-run panel consumes "
               "it is named"},
    {"id": "empty_row_is_named", "kind": "empty",
     "detail": "empty rerun / verdict / delta rows are named, "
               "never crashed on"},
    {"id": "doc_drift_is_named", "kind": "doc_drift",
     "detail": "a doc text that drifts from the filed render "
               "is named"},
)


def run_rerun_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in RERUN_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in RERUN_ROWS:
                problems += rerun_row_problems(row)
            for row in VERDICT_ROWS:
                problems += verdict_row_problems(row)
            for row in DELTA_ROWS:
                problems += delta_row_problems(row)
            passed = problems == []
        elif kind == "moved_unmoved":
            row = dict(DELTA_ROWS[0], remeasured=19, delta=1,
                       outcome="unmoved")
            passed = bool(delta_row_problems(row))
        elif kind == "wrong_delta":
            row = dict(DELTA_ROWS[0], delta=99)
            passed = bool(delta_row_problems(row))
        elif kind == "false_consumption":
            row = dict(RERUN_ROWS[0], consumes_adopted_code=True)
            passed = bool(rerun_row_problems(row))
        elif kind == "verdict_consumption":
            row = dict(VERDICT_ROWS[0], consumed_by_rerun=True)
            passed = bool(verdict_row_problems(row))
        elif kind == "empty":
            passed = bool(rerun_row_problems(None)) and bool(
                verdict_row_problems(None)) and bool(
                delta_row_problems(None)) and bool(
                rerun_state_problems(None, None)) and bool(
                doc_text_problems(None, None, "x"))
        elif kind == "doc_drift":
            render = render_panel_rerun_markdown()
            passed = bool(doc_text_problems(
                render + "drift\n", render, "panel rerun doc"))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the re-run doc, verbatim) ───────────────────────────

def _delta_table_lines() -> list[str]:
    lines = [
        "| Panel | Metric | Pinned | Re-measured | Delta | Outcome |",
        "|---|---|---|---|---|---|",
    ]
    for row in DELTA_ROWS:
        lines.append(
            f"| {row['panel']} | {row['metric']} | {row['pinned']} | "
            f"{row['remeasured']} | {row['delta']} | {row['outcome']} |")
    return lines


def _limits_lines() -> list[str]:
    return [
        "- Fixture/static grade only: every value above is a "
        "pinned synthetic-fixture baseline re-measured live on "
        "this VM through the harness's own measurers. No live "
        "model was read; live persona or world-state behaviour "
        "is NOT PROVEN (TF-173 class), never claimed.",
        "- The zero deltas are the finding, not a failure: the "
        "three re-run measurers consume no Part VIII adopted "
        "code (checked live against their sources), so the "
        "verdicts could not move them. Any future panel that "
        "binds the bond layer or the compiled persona must be "
        "re-run as its own slice and file its own delta.",
        "- No harness baseline was added or re-pinned by this "
        "slice (the TF-138 gate precedent for a re-measurement "
        "slice): the comparison stands against the existing "
        "pins, and the baseline book / flake census / "
        "public-claims filings do not move.",
        "- No docket change: TF-151 has no TF-016 share and "
        "found no defect; TF-016 totals stand unchanged (137 "
        "— 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED / 4 "
        "CLOSED-BY-VERDICT). TF-152 owns the Part VIII gate.",
        "- No canon or persona content is filed: only the "
        "estate's synthetic fixture scores, counts and verdict "
        "names.",
    ]


def render_panel_rerun_markdown() -> str:
    """Render the re-run doc deterministically —
    the exact text filed as
    docs/architecture/PANEL_RERUN.md."""
    state = run_rerun_state()
    lines: list[str] = []
    lines.append("# Panel Re-run — TF-132 + Persona Suites "
                 "Re-measured With the Verdicts Landed (TF-151)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/panel_rerun.py` — the filed doc IS "
        "the render: regenerate with "
        "`python3 -m thoughtforge.panel_rerun`. The checker "
        "(`panel_rerun_problems()`) re-measures every value "
        "live through the harness and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{PANEL_RERUN_FORMAT}`")
    lines.append(
        f"Panels re-run: {state['panels']}; metrics "
        f"re-measured: {state['delta_rows']} "
        f"(unmoved {state['unmoved_rows']} / moved "
        f"{state['moved_rows']}); verdicts in force: "
        f"{state['verdict_rows']} (consumed by a re-run panel: "
        f"{state['verdicts_consumed']})")
    lines.append(f"Probes: {len(RERUN_PROBES)}/{len(RERUN_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built, "
        "from the runners only: TF-132's WYRD panel PASSes at "
        "18 cases / 9 consistent / 8 contradicted / 1 "
        "honest-unanswerable, overall 0.5555555555555556; "
        "TF-122's persona suite PASSes at 108 turns, overall "
        "0.8453703703703704 (identity 0.8333333333333334 / "
        "voice 0.8694444444444445 / values "
        "0.8333333333333334); the persona-consistency scorer "
        "underneath the voice axis measures 0.7166666666666667 "
        "over 6 turns. Every value equals its pinned harness "
        "baseline exactly. Source inspection proves why: none "
        "of the three measurers imports the bond graph, the "
        "persona compiler, the bond bridge or the Runmal "
        "cards — the Part VIII verdicts landed in other "
        "modules, and these panels do not read them. The "
        "deltas are therefore honestly zero, published below "
        "rather than assumed.")
    lines.append("")
    lines.append("## 2. The verdicts in force at the re-run")
    lines.append("")
    lines.append("| Slice | Verdict | Consumed by a re-run panel? | Why |")
    lines.append("|---|---|---|---|")
    for row in VERDICT_ROWS:
        lines.append(
            f"| {row['slice']} | {row['verdict']} | "
            f"{'yes' if row['consumed_by_rerun'] else 'no'} | "
            f"{row['reason']} |")
    lines.append("")
    lines.append("## 3. The deltas, published")
    lines.append("")
    lines.extend(_delta_table_lines())
    lines.append("")
    lines.append(
        "Headline: 19 metrics re-measured, 19 unmoved, 0 "
        "moved. Both panel verdicts stand PASS before and "
        "after. The verdicts landed changed no re-run "
        "measurement — proven live, not assumed — because no "
        "re-run measurer consumes the adopted code.")
    lines.append("")
    lines.append("## 4. Harness coupling")
    lines.append("")
    lines.append(
        "None, filed in the open on the TF-138 gate "
        "precedent: a re-measurement slice adds no harness "
        "baseline and re-pins none, so the baseline book "
        "(18 baselines / 146 pinned metrics), the flake "
        "census (18 deterministic) and the public-claims "
        "audit do not move, and `baseline_problems()` stays "
        "clean over the existing pins this re-run compares "
        "against.")
    lines.append("")
    lines.append("## 5. Docket")
    lines.append("")
    lines.append(
        "None. TF-151 has no share of any TF-016 entry and "
        "its re-runs found no defect. TF-016 totals stand "
        "unchanged: 137 entries — 31 OPEN / 1 "
        "OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT.")
    lines.append("")
    lines.append("## 6. The probes")
    lines.append("")
    for probe in RERUN_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    for line in _limits_lines():
        lines.append(line)
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_panel_rerun_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / PANEL_RERUN_DOC
    doc.write_text(render_panel_rerun_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.panel_rerun`."""
    doc = write_panel_rerun_doc()
    print(f"panel rerun doc regenerated: -> {PANEL_RERUN_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def panel_rerun_problems() -> list[str]:
    """Re-derive TF-151's claims live; name any
    that stopped being true."""
    problems: list[str] = []

    for row in RERUN_ROWS:
        problems.extend(rerun_row_problems(row))
    for row in VERDICT_ROWS:
        problems.extend(verdict_row_problems(row))
    for row in DELTA_ROWS:
        problems.extend(delta_row_problems(row))
    if len(RERUN_ROWS) != 3 or len(VERDICT_ROWS) != 7 \
            or len(DELTA_ROWS) != 19:
        problems.append(
            "the filing is not 3 panels / 7 verdicts / 19 deltas")

    # Live re-measurement vs pins vs filed deltas.
    from benchmarks.harness import BASELINES, baseline_problems

    pins = {b["id"]: b["metrics"] for b in BASELINES}
    live = _live_measurements()
    panel_by_baseline = {r["baseline_id"]: r["id"] for r in RERUN_ROWS}
    for row in DELTA_ROWS:
        baseline_id = next(
            r["baseline_id"] for r in RERUN_ROWS
            if r["id"] == row["panel"])
        pinned = pins.get(baseline_id, {}).get(row["metric"])
        if pinned != row["pinned"]:
            problems.append(
                f"delta {row['panel']}.{row['metric']}: filed pin "
                f"{row['pinned']!r} is not the harness pin {pinned!r}")
        measured = live[row["panel"]].get(row["metric"])
        if measured != row["remeasured"]:
            problems.append(
                f"delta {row['panel']}.{row['metric']}: live "
                f"re-measurement is {measured!r}, not the filed "
                f"{row['remeasured']!r}")
    if baseline_problems():
        problems.append(
            f"harness: baseline_problems() is not clean: "
            f"{baseline_problems()[:3]}")
    _ = panel_by_baseline

    # Coupling, live: no re-run source names adopted code.
    for row in RERUN_ROWS:
        text = (_REPO_ROOT / row["source_file"]).read_text(
            encoding="utf-8")
        for token in ADOPTED_CODE_TOKENS:
            if token in text:
                problems.append(
                    f"coupling: {row['source_file']} now names "
                    f"{token!r} — it consumes adopted code and "
                    "this no-delta filing must be re-measured")

    # The verdicts still stand as filed.
    from thoughtforge.cognition.persona_compiler_verdict import (
        VERDICT as _persona_verdict,
    )
    from thoughtforge.knowledge.bond_graph_verdict import (
        VERDICT as _bond_verdict,
    )
    from thoughtforge.knowledge.micro_rag_verdict import (
        VERDICT as _micro_verdict,
    )
    from thoughtforge.refinement.truth_calibrator_verdict import (
        VERDICT as _truth_verdict,
    )

    for filed, live_verdict in (
            ("tf139_bond_graph", _bond_verdict.get("verdict")),
            ("tf140_micro_rag", _micro_verdict.get("verdict")),
            ("tf141_persona_compiler", _persona_verdict.get("verdict")),
            ("tf142_truth_calibrator", _truth_verdict.get("verdict"))):
        row = next(r for r in VERDICT_ROWS if r["id"] == filed)
        if row["verdict"] != live_verdict:
            problems.append(
                f"verdict {filed}: files {row['verdict']!r}, "
                f"the verdict module files {live_verdict!r}")

    # Neighbouring checkers clean live.
    from benchmarks.persona_suite import persona_suite_problems
    from benchmarks.wyrd_panel import wyrd_panel_problems
    from thoughtforge.cognition.persona_compiler import (
        persona_compiler_problems,
    )
    from thoughtforge.knowledge.bond_graph import bond_graph_problems

    for name, checker in (
            ("wyrd_panel", wyrd_panel_problems),
            ("persona_suite", persona_suite_problems),
            ("bond_graph", bond_graph_problems),
            ("persona_compiler", persona_compiler_problems)):
        neighbour = checker()
        if neighbour:
            problems.append(
                f"neighbour: {name}_problems() is not clean: "
                f"{neighbour[:3]}")

    # The measured state matches its filing.
    problems.extend(rerun_state_problems(
        run_rerun_state(), RERUN_EXPECTED))

    # The probes all behave.
    probes = run_rerun_probes()
    if len(probes) != len(RERUN_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the rerun probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The docket stands unchanged.
    import re

    docket_text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", docket_text)
    if "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, 101 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals no longer file 137 "
            "entries — 31 OPEN / 1 OPEN-ENVIRONMENT / 101 "
            "FIXED; TF-151 must not move them")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / PANEL_RERUN_DOC
    if not doc.exists():
        problems.append(f"the rerun doc {PANEL_RERUN_DOC} is missing")
    else:
        problems.extend(
            f"doc: {p}" for p in doc_text_problems(
                doc.read_text(encoding="utf-8"),
                render_panel_rerun_markdown(), "panel rerun doc"))
    return problems


def panel_rerun_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_rerun_probes()
    return {
        "format": PANEL_RERUN_FORMAT,
        "doc": PANEL_RERUN_DOC,
        "panels": [dict(r) for r in RERUN_ROWS],
        "verdicts": [dict(r) for r in VERDICT_ROWS],
        "deltas": [dict(r) for r in DELTA_ROWS],
        "state": run_rerun_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "ADOPTED_CODE_TOKENS",
    "DELTA_OUTCOMES",
    "DELTA_ROWS",
    "PANEL_RERUN_DOC",
    "PANEL_RERUN_FORMAT",
    "RERUN_EXPECTED",
    "RERUN_PROBES",
    "RERUN_ROWS",
    "VERDICT_ROWS",
    "delta_row_problems",
    "doc_text_problems",
    "main",
    "panel_rerun_as_data",
    "panel_rerun_problems",
    "render_panel_rerun_markdown",
    "rerun_row_problems",
    "rerun_state_problems",
    "run_rerun_probes",
    "run_rerun_state",
    "verdict_row_problems",
    "write_panel_rerun_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
