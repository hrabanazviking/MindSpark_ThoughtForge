"""
The harness as a tool, as data (TF-137).

The roadmap's rule: *the harness as a tool: one
command runs the panel a slice needs; keepers cite
its output verbatim.* Part VII built the estate
piece by piece (TF-121..TF-136): suites, panels, a
regression gate, spot checks, a hygiene inventory,
a WYRD panel, a cost panel, a baseline book, a
flake discipline and a public-claims audit — each
with its own module, its own checker and its own
pinned baseline in ``benchmarks/harness.py``. But
running "the panel a slice needs" still meant
knowing which of those modules to import and which
of its functions to call, and a keeper citing the
outcome had no single deterministic line to cite:
every slice re-invented its own verification
invocation in campaign prose.

The measured starting state, probed live this tick
on this tree before anything was built:

  * **The estate stood clean, and no harness tool
    existed.** ``harness_problems()`` returned []
    and ``baseline_problems()`` returned [] over
    the seventeen pinned baselines,
    ``regression_gate_problems()`` returned []
    with the live gate PASS (9 baselines / 83
    metrics / 0 failures),
    ``baseline_book_problems()`` returned []
    (book PASS 17 baselines / 139 pinned metrics),
    ``data_hygiene_problems()`` returned []
    (21 inputs, all clean),
    ``flake_discipline_problems()`` returned []
    and ``public_claims_problems()`` returned [] —
    and a grep across benchmarks / docs / tests /
    src found no panel runner, no panel registry
    and no citable run output anywhere: the panels
    existed, the one command did not.
  * **The gap was never numbered.** The number
    **D-BENCH-22** was verified unused anywhere in
    benchmarks / docs / tests / src before this
    slice filed it (the D-BENCH-9..21 pattern).

What this module files, as verifiable data:

  * ``PANEL_ROWS`` — the panels, published as
    literal data: each panel files its id, its
    purpose (which slice need it serves) and its
    member baselines. Members are drawn from the
    REAL estate only — the harness's pinned
    baselines, whose runners are the real
    ``run_*`` functions the harness's BENCHMARKS
    index binds for the persona / memory / depth /
    presence suites, the honesty / quant / profile
    panels, the regression gate, the spot checks,
    the data hygiene inventory, the WYRD panel,
    the cost panel, the baseline book, the flake
    discipline and the public-claims audit. No
    scorer or runner is copied here: a member is
    run by re-measuring its baseline through the
    harness's own measurer and comparing it with
    its pin, exactly as ``baseline_problems()``
    judges it. Three panels are filed: the
    **full panel** (every pinned baseline, this
    slice's included), the **gate panel**
    (TF-129's nine gated baselines) and the
    **audit panel** (the ungated, checker-judged
    baselines).
  * **One command.** ``python3 -m
    benchmarks.harness_tool`` regenerates the
    filed doc; with ``--list-panels`` it lists the
    panels, with ``--panel <id>`` it runs one
    panel, and with ``--baseline <id>`` (repeatable)
    it runs an individual selection. Every run
    prints deterministic, citable lines — a
    citation line naming panel, members run,
    members passed and verdict, one member line
    per baseline, and one line per problem — with
    NO wall times anywhere in the output (timing
    is excluded-by-rule, TF-121: walls under
    contention are not baselines, and a cited line
    must reproduce verbatim).
  * The verdict: PASS only on a FULL panel
    selection (the full panel's member list,
    exactly) with every member running and
    passing; a partial selection — a smaller
    panel, an individual selection, however it
    scores — files REVIEW; an empty or unknown
    selection is named and files REVIEW, never a
    vacuous pass and never a crash.
  * ``TOOL_PROBES`` — filed probes against the
    pure validators with perturbed copies of
    filed rows (real filings never perturbed):
    the clean panel passes; an unknown member is
    named; a duplicate member is named; an empty
    panel is named; a panel missing its purpose
    is named; a member reported passing that did
    not run is named; a citation line that does
    not match the run is named.
  * The readable filing IS the render:
    ``render_harness_tool_markdown()`` renders
    the panels, the commands and the rules
    deterministically, and
    ``docs/architecture/HARNESS_TOOL.md`` files
    that render verbatim — the checker requires
    the doc to reproduce the render exactly, and
    ``python3 -m benchmarks.harness_tool``
    regenerates it.

D-BENCH-22 is FIXED by this slice: the harness is
now a tool — one command runs the panel a slice
needs, and keepers cite its deterministic output
verbatim. This slice's own harness baseline is
judged by this module's checker, NOT gated by
TF-129's regression gate (the gate's gated set
stays the nine TF-121..TF-128 baselines exactly
as filed; the exclusion is filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129..TF-136 filed for their own baselines).
TF-131 coupling, filed plainly: this tool adds
NO benchmark input — it runs only already-filed
baselines through their already-filed measurers —
so the hygiene inventory does NOT grow, its
pinned baseline is NOT re-pinned, and no new
harness fixture is filed (the TF-133 / TF-134 /
TF-135 / TF-136 shape).

Filed limits, stated plainly:

  * The tool runs the harness's pinned fixture
    baselines only — synthetic fixtures through
    the real scorers (TF-017). No live model is
    run; live behaviour is NOT PROVEN (TF-173).
  * A member passes here when its re-measurement
    reproduces its pin within its filed tolerance
    (``baseline_problems()``'s judgement, per
    member). The audit panel's members are each
    ALSO judged by their own module's checker in
    their own slices; this tool does not re-run
    those checkers (several recompute the whole
    estate — a panel run stays one measurement
    pass, and the checkers stand as filed).
  * Run output files no timing: a run's wall time
    varies under contention and is never part of
    a cited line (TF-121's rule, carried).

The human-readable write-up is
``docs/architecture/HARNESS_TOOL.md``, filed
verbatim as this module's render.
Proof grade: FIXTURE — pinned synthetic-fixture
baselines re-measured live on this VM; no live
model, never claimed.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HARNESS_TOOL_FORMAT = "thoughtforge-harness-tool/1"
HARNESS_TOOL_DOC = "docs/architecture/HARNESS_TOOL.md"

#: The new baseline this slice registers in the
#: harness — the tool's own pin. It is NOT one of
#: TF-129's gated baselines (the gate gates the
#: nine TF-121..TF-128 baselines, as filed; this
#: baseline is judged by this module's checker).
HARNESS_TOOL_BASELINE_ID = "harness_tool_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The real machinery, reused — never copied: the
# harness's pinned baselines and its own
# measurers, and TF-129's gate registries. This
# module imports them at top level; the harness
# binds this module only lazily (inside its
# measure function), so no import cycle exists.
from benchmarks import harness as _harness  # noqa: E402
from benchmarks import regression_gate as _gate  # noqa: E402


# ── The tool's rules ────────────────────────────────────────────────────────

HARNESS_TOOL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "panels_are_literal_data_bound_to_the_real_estate",
     "rule": "PANEL_ROWS files each panel as literal "
             "data — its id, its purpose (which slice "
             "need it serves) and its member baselines "
             "— and every member is a pinned baseline "
             "in benchmarks/harness.py whose runner is "
             "the real run_* function the BENCHMARKS "
             "index binds; no scorer or runner is "
             "copied into this module"},
    {"id": "one_command_runs_the_panel_a_slice_needs",
     "rule": "python3 -m benchmarks.harness_tool is "
             "the one command: --list-panels lists the "
             "panels, --panel <id> runs one panel, and "
             "--baseline <id> (repeatable) runs an "
             "individual selection; with no selection "
             "argument it regenerates the filed doc"},
    {"id": "citation_lines_are_deterministic_and_verbatim",
     "rule": "every run prints deterministic lines a "
             "keeper cites verbatim — one citation "
             "line (panel, members run, members "
             "passed, verdict), one member line per "
             "baseline, one line per problem — and no "
             "wall time ever stands in the output: "
             "timing is excluded-by-rule (TF-121), so "
             "a cited line reproduces exactly"},
    {"id": "a_member_passes_by_reproducing_its_pin",
     "rule": "a member runs by re-measuring its "
             "baseline through the harness's own "
             "measurer and comparing every metric "
             "with its pin within its filed tolerance "
             "— baseline_problems()'s judgement, per "
             "member; a member that cannot re-measure "
             "files ran=False with its reason, never "
             "a silent skip"},
    {"id": "pass_needs_the_full_panel_all_passing",
     "rule": "the verdict is PASS only on a full "
             "panel selection (the full panel's "
             "member list, exactly) with every member "
             "running and passing; a partial "
             "selection — a smaller panel or an "
             "individual selection — files REVIEW "
             "however it scores"},
    {"id": "empty_and_unknown_selections_are_named_review",
     "rule": "an empty selection, an unknown panel "
             "and an unknown baseline are each named "
             "and file REVIEW — never a vacuous pass, "
             "never a crash"},
    {"id": "probes_prove_the_validator_can_fail",
     "rule": "TOOL_PROBES perturbs copies of filed "
             "rows against the pure validators (an "
             "unknown member, a duplicate member, an "
             "empty panel, a panel missing its "
             "purpose, a member reported passing that "
             "did not run, a citation line that does "
             "not match the run) plus the clean "
             "shape — real filings are never "
             "perturbed"},
    {"id": "the_harness_tool_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates the "
             "nine TF-121..TF-128 pinned baselines "
             "exactly as filed; this slice's own "
             "baseline (harness_tool_synthetic_v1) is "
             "judged by this module's checker, not "
             "gated by the gate — the shape "
             "TF-129..TF-136 filed for their own "
             "baselines, with the exclusion filed in "
             "the open in regression_gate.py"},
    {"id": "no_new_benchmark_input_was_added",
     "rule": "the tool runs only already-filed "
             "baselines through their already-filed "
             "measurers — it adds no benchmark input, "
             "so it files no new harness fixture and "
             "no hygiene row, and TF-131's inventory "
             "and its pinned baseline do not move "
             "(filed in the open, TF-133's coupling "
             "shape)"},
)


# ── The panels (literal data) ───────────────────────────────────────────────
#
# Filed in the harness's baseline order. The
# checker binds every panel live: the full panel
# must equal the harness's pinned baselines, the
# gate panel TF-129's GATED_BASELINE_IDS and the
# audit panel TF-129's UNGATED_BASELINE_IDS, so a
# registry move names the drift here.

PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "full_panel",
     "purpose": "the whole estate: the panel "
                "TF-138's Part VII gate runs, and "
                "any slice whose change could move "
                "any pinned baseline — every pinned "
                "baseline in the harness, this "
                "tool's own included; PASS lives on "
                "this selection only",
     "members": (
         "persona_consistency_synthetic_v1",
         "persona_suite_synthetic_v1",
         "memory_suite_synthetic_v1",
         "depth_suite_synthetic_v1",
         "presence_suite_synthetic_v1",
         "honesty_panel_synthetic_v1",
         "quant_panel_synthetic_v1",
         "profile_panel_synthetic_v1",
         "regression_gate_synthetic_v1",
         "spot_checks_synthetic_v1",
         "data_hygiene_synthetic_v1",
         "wyrd_panel_synthetic_v1",
         "cost_panel_synthetic_v1",
         "baseline_book_synthetic_v1",
         "flake_discipline_synthetic_v1",
         "public_claims_synthetic_v1",
         "harness_tool_synthetic_v1",
         "profile_benchmark_empty_store_v1",
     )},
    {"id": "gate_panel",
     "purpose": "a slice that touched a gated suite "
                "or panel (TF-121..TF-128): the nine "
                "baselines TF-129's regression gate "
                "gates, run here per member against "
                "their pins (the gate itself, with "
                "its noise-derived tolerances, stays "
                "benchmarks/regression_gate.py)",
     "members": (
         "persona_consistency_synthetic_v1",
         "persona_suite_synthetic_v1",
         "memory_suite_synthetic_v1",
         "depth_suite_synthetic_v1",
         "presence_suite_synthetic_v1",
         "honesty_panel_synthetic_v1",
         "quant_panel_synthetic_v1",
         "profile_panel_synthetic_v1",
         "profile_benchmark_empty_store_v1",
     )},
    {"id": "audit_panel",
     "purpose": "a slice that touched a "
                "checker-judged filing — the gate's "
                "verdict row, the spot checks, the "
                "hygiene inventory, the WYRD panel, "
                "the cost panel, the baseline book, "
                "the flake discipline, the "
                "public-claims audit or this tool: "
                "the ungated baselines, each judged "
                "by its own module's checker in its "
                "own slice and re-measured against "
                "its pin here",
     "members": (
         "regression_gate_synthetic_v1",
         "spot_checks_synthetic_v1",
         "data_hygiene_synthetic_v1",
         "wyrd_panel_synthetic_v1",
         "cost_panel_synthetic_v1",
         "baseline_book_synthetic_v1",
         "flake_discipline_synthetic_v1",
         "public_claims_synthetic_v1",
         "harness_tool_synthetic_v1",
     )},
)


def panel_by_id(panel_id: str) -> dict[str, Any] | None:
    """One filed panel by id, or None."""
    for panel in PANEL_ROWS:
        if panel["id"] == panel_id:
            return panel
    return None


def _full_panel_members() -> list[str]:
    panel = panel_by_id("full_panel")
    return list(panel["members"]) if panel else []


# ── The pure validators (probes run here) ───────────────────────────────────

def panel_problems(
    row: dict[str, Any], known_ids: Any = None,
) -> list[str]:
    """Judge ONE panel row against its own parts,
    purely: it must file an id, a purpose and a
    non-empty member list with no unknown and no
    duplicate member. Pure — an empty or partial
    row is named, never crashed on."""
    known = [b["id"] for b in _harness.BASELINES] \
        if known_ids is None else list(known_ids)
    problems: list[str] = []
    label = row.get("id") or "<unnamed panel>"
    if not row.get("id"):
        problems.append("a panel files no id")
    if not row.get("purpose"):
        problems.append(
            f"{label}: the panel files no purpose — "
            f"a panel that cannot say which slice "
            f"need it serves is not a panel a slice "
            f"can choose")
    members = row.get("members") or []
    if not members:
        problems.append(
            f"{label}: the panel files no members — "
            f"an empty panel runs nothing and passes "
            f"nothing")
    seen: set[str] = set()
    for member in members:
        if member in seen:
            problems.append(
                f"{label}: member {member} is filed "
                f"twice — a duplicate member runs "
                f"twice and counts twice, named")
        seen.add(member)
        if member not in known:
            problems.append(
                f"{label}: member {member} is no "
                f"pinned baseline in the harness — an "
                f"unknown member is named, never run")
    return problems


def member_result_problems(
    row: dict[str, Any],
) -> list[str]:
    """Judge ONE member-run row, purely: it must
    file its baseline and whether it ran; a member
    reported passing must have run and must file
    no problems; a member that did not run must
    file why. Pure."""
    problems: list[str] = []
    label = row.get("baseline") or "<unnamed member>"
    if not row.get("baseline"):
        problems.append("a member result files no baseline")
    ran = row.get("ran")
    passed = row.get("passed")
    if ran is None:
        problems.append(
            f"{label}: the member result files no "
            f"ran state — a result that cannot say "
            f"whether its member ran settles nothing")
    if passed is None:
        problems.append(
            f"{label}: the member result files no "
            f"passed state")
    if passed and not ran:
        problems.append(
            f"{label}: reported passing but did not "
            f"run — a pass is a measurement, never a "
            f"report about a run that never happened")
    if passed and row.get("problems"):
        problems.append(
            f"{label}: reported passing but files "
            f"problems {row.get('problems')} — a "
            f"passing member files no problems")
    if ran is False and not row.get("problems"):
        problems.append(
            f"{label}: did not run and files no "
            f"reason — a member that does not run "
            f"must file why, never a silent skip")
    return problems


def citation_line_for(run_data: dict[str, Any]) -> str:
    """The deterministic citation line for one run,
    from its data: panel, members run, members
    passed, verdict. No timing ever stands in it."""
    return (
        f"harness-tool run: panel={run_data.get('panel')} "
        f"members={run_data.get('members_run')} "
        f"passed={run_data.get('members_passed')} "
        f"verdict={run_data.get('verdict')}")


def citation_problems(
    line: str, run_data: dict[str, Any],
) -> list[str]:
    """Judge ONE citation line against the run it
    claims to cite, purely: the line must be the
    run's citation line exactly — a line that does
    not match the run is named, never carried."""
    expected = citation_line_for(run_data)
    if line != expected:
        return [
            f"citation line {line!r} does not match "
            f"the run it cites (expected {expected!r}) "
            f"— a keeper cites a run's output "
            f"verbatim or not at all"]
    return []


# ── The filed probes ────────────────────────────────────────────────────────

def _probe_panel(**overrides: Any) -> dict[str, Any]:
    gate = panel_by_id("gate_panel")
    row = copy.deepcopy(dict(gate)) if gate else {
        "id": "probe_panel", "purpose": "probe",
        "members": ("persona_suite_synthetic_v1",)}
    row.update(overrides)
    return row


TOOL_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_panel_passes",
     "kind": "panel", "overrides": {},
     "expect_problem": False,
     "detail": "a filed panel whose id, purpose "
               "and members agree with the estate "
               "files no problem"},
    {"id": "unknown_member_is_named",
     "kind": "panel",
     "overrides": {"members": (
         "persona_suite_synthetic_v1",
         "totally_bogus_baseline_v1")},
     "expect_problem": True,
     "detail": "a panel filing a member that is no "
               "pinned baseline is named by member"},
    {"id": "duplicate_member_is_named",
     "kind": "panel",
     "overrides": {"members": (
         "persona_suite_synthetic_v1",
         "persona_suite_synthetic_v1")},
     "expect_problem": True,
     "detail": "a panel filing the same member "
               "twice is named — a duplicate runs "
               "and counts twice if carried"},
    {"id": "empty_panel_is_named",
     "kind": "panel",
     "overrides": {"members": ()},
     "expect_problem": True,
     "detail": "a panel filing no members is named "
               "— an empty panel is never a vacuous "
               "pass"},
    {"id": "panel_missing_its_purpose_is_named",
     "kind": "panel",
     "overrides": {"purpose": ""},
     "expect_problem": True,
     "detail": "a panel that cannot say which "
               "slice need it serves is named"},
    {"id": "member_passing_that_did_not_run_is_named",
     "kind": "member_result",
     "row": {"baseline": "persona_suite_synthetic_v1",
             "benchmark": "persona_suite",
             "ran": False, "passed": True,
             "problems": ["re-measurement failed"]},
     "expect_problem": True,
     "detail": "a member reported passing that did "
               "not run is named — a pass is a "
               "measurement, never a report"},
    {"id": "citation_line_that_does_not_match_is_named",
     "kind": "citation",
     "line": "harness-tool run: panel=full_panel "
             "members=18 passed=17 verdict=PASS",
     "run_data": {"panel": "full_panel",
                  "members_run": 18,
                  "members_passed": 18,
                  "verdict": "PASS"},
     "expect_problem": True,
     "detail": "a citation line whose counts differ "
               "from the run it cites is named — "
               "keepers cite output verbatim"},
)


def run_tool_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in TOOL_PROBES:
        if probe["kind"] == "panel":
            problems = panel_problems(
                _probe_panel(**probe["overrides"]))
        elif probe["kind"] == "member_result":
            problems = member_result_problems(
                copy.deepcopy(probe["row"]))
        else:  # citation
            problems = citation_problems(
                probe["line"], probe["run_data"])
        passed = bool(problems) == probe["expect_problem"]
        results.append({
            "id": probe["id"], "passed": passed,
            "problems": problems,
            "detail": probe["detail"]})
    return results


# ── The tool run ────────────────────────────────────────────────────────────

def _run_member(baseline_id: str) -> dict[str, Any]:
    """Run ONE member: re-measure its baseline
    through the harness's own measurer and compare
    every metric with its pin within its filed
    tolerance (baseline_problems()'s judgement,
    per member). An unknown baseline and a failed
    re-measurement each file ran=False with their
    reason, never a silent skip."""
    baselines = {b["id"]: b for b in _harness.BASELINES}
    baseline = baselines.get(baseline_id)
    if baseline is None:
        return {
            "baseline": baseline_id, "benchmark": None,
            "ran": False, "passed": False,
            "problems": [
                f"unknown baseline {baseline_id} — "
                f"named, not run"]}
    measurer = _harness._MEASURERS.get(baseline_id)
    if measurer is None:
        return {
            "baseline": baseline_id,
            "benchmark": baseline["benchmark"],
            "ran": False, "passed": False,
            "problems": [
                f"baseline {baseline_id} has no "
                f"harness measurer"]}
    try:
        measured = measurer()
    except Exception as exc:
        return {
            "baseline": baseline_id,
            "benchmark": baseline["benchmark"],
            "ran": False, "passed": False,
            "problems": [
                f"re-measurement failed: {exc!r}"]}
    tolerance = float(baseline["tolerance"])
    problems: list[str] = []
    for metric, pinned in baseline["metrics"].items():
        if metric not in measured:
            problems.append(
                f"metric {metric} no longer measured")
            continue
        got = measured[metric]
        if isinstance(pinned, float) or isinstance(got, float):
            if abs(float(got) - float(pinned)) > tolerance:
                problems.append(
                    f"{metric} moved {pinned} -> {got} "
                    f"(tolerance {tolerance})")
        elif got != pinned:
            problems.append(
                f"{metric} moved {pinned} -> {got}")
    return {
        "baseline": baseline_id,
        "benchmark": baseline["benchmark"],
        "ran": True, "passed": not problems,
        "problems": problems}


@dataclass
class HarnessToolResult:
    """One tool run: the selection it filed, the
    member rows it ran, and the verdict data."""

    panel: str = "full_panel"
    requested: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    selection_problems: list[str] = field(default_factory=list)

    @property
    def members_run(self) -> int:
        return sum(1 for r in self.rows if r["ran"])

    @property
    def members_passed(self) -> int:
        return sum(1 for r in self.rows if r["passed"])

    @property
    def members_failed(self) -> int:
        return len(self.rows) - self.members_passed

    @property
    def overall_score(self) -> float:
        if not self.rows:
            return 0.0
        return self.members_passed / len(self.rows)

    @property
    def verdict(self) -> str:
        """PASS only on the full panel selection
        with every member running and passing;
        REVIEW for any partial, empty, unknown or
        imperfect selection, however it scores."""
        if self.requested != _full_panel_members():
            return "REVIEW"
        if self.selection_problems:
            return "REVIEW"
        clean = bool(self.rows) and all(
            r["ran"] and r["passed"] for r in self.rows)
        return "PASS" if clean else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"

    @property
    def run_data(self) -> dict[str, Any]:
        return {
            "panel": self.panel,
            "members_run": self.members_run,
            "members_passed": self.members_passed,
            "verdict": self.verdict}

    @property
    def citation_line(self) -> str:
        return citation_line_for(self.run_data)


def citation_lines(result: HarnessToolResult) -> list[str]:
    """The run's citable output, deterministically:
    the citation line, one member line per row in
    selection order, then one line per problem
    (selection problems first, then member
    problems in row order). No timing anywhere."""
    lines = [result.citation_line]
    for row in result.rows:
        status = "passed" if row["passed"] else "failed"
        lines.append(
            f"harness-tool member baseline={row['baseline']} "
            f"status={status}")
    for problem in result.selection_problems:
        lines.append(f"harness-tool problem: {problem}")
    for row in result.rows:
        for problem in row["problems"]:
            lines.append(
                f"harness-tool problem: "
                f"{row['baseline']}: {problem}")
    return lines


def run_harness_tool(
    panel_id: Any = None,
    baseline_ids: Any = None,
) -> HarnessToolResult:
    """Run the panel a slice needs.

    With ``panel_id``, run that filed panel; with
    ``baseline_ids``, run that individual
    selection; with neither, run the full panel.
    An unknown panel, an unknown baseline, a
    duplicate or an empty selection is named in
    the result and files REVIEW — never a vacuous
    pass, never a crash.
    """
    if panel_id is not None:
        panel = panel_by_id(str(panel_id))
        if panel is None:
            known = ", ".join(p["id"] for p in PANEL_ROWS)
            return HarnessToolResult(
                panel=str(panel_id), requested=[], rows=[],
                selection_problems=[
                    f"unknown panel {panel_id!r} — named, "
                    f"not run (known panels: {known})"])
        requested = list(panel["members"])
        return HarnessToolResult(
            panel=str(panel_id), requested=requested,
            rows=[_run_member(b) for b in requested],
            selection_problems=[])
    if baseline_ids is not None:
        requested = [str(b) for b in baseline_ids]
        problems: list[str] = []
        if not requested:
            problems.append(
                "empty selection — no baseline was "
                "named; an empty run passes nothing")
        seen: set[str] = set()
        for baseline_id in requested:
            if baseline_id in seen:
                problems.append(
                    f"duplicate member {baseline_id} in "
                    f"the selection — named; it runs "
                    f"once below")
            seen.add(baseline_id)
        unique = list(dict.fromkeys(requested))
        return HarnessToolResult(
            panel="custom", requested=requested,
            rows=[_run_member(b) for b in unique],
            selection_problems=problems)
    panel = panel_by_id("full_panel")
    requested = list(panel["members"]) if panel else []
    return HarnessToolResult(
        panel="full_panel", requested=requested,
        rows=[_run_member(b) for b in requested],
        selection_problems=[])


# ── The readable render (the doc, verbatim) ─────────────────────────────────

def render_harness_tool_markdown(
    rows: Any = None,
) -> str:
    """Render the tool deterministically — the
    exact text filed as
    docs/architecture/HARNESS_TOOL.md: the panels
    with their purposes and members, the commands,
    the citation shape and the rules. Every number
    comes from the filed data; nothing is
    hand-written here but the frame."""
    rows = PANEL_ROWS if rows is None else tuple(rows)
    lines: list[str] = []
    lines.append("# The Harness as a Tool — TF-137")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`benchmarks/harness_tool.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m benchmarks.harness_tool`. The "
        "checker (`harness_tool_problems()`) binds "
        "every panel against the live registries "
        "and fails on any drift from the filed "
        "rows.")
    lines.append("")
    lines.append(f"Format: `{HARNESS_TOOL_FORMAT}`")
    lines.append(f"Panels filed: {len(rows)}")
    for row in rows:
        lines.append(
            f"Panel `{row['id']}` members: "
            f"{len(row['members'])}")
    lines.append("")
    lines.append(
        "One command runs the panel a slice needs, "
        "and keepers cite its output verbatim. A "
        "member runs by re-measuring its pinned "
        "baseline through the harness's own "
        "measurer and comparing it with its pin — "
        "no scorer or runner is copied here.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append(
        "- List the panels: `python3 -m "
        "benchmarks.harness_tool --list-panels`")
    lines.append(
        "- Run one panel: `python3 -m "
        "benchmarks.harness_tool --panel gate_panel`")
    lines.append(
        "- Run individual baselines: `python3 -m "
        "benchmarks.harness_tool --baseline "
        "persona_suite_synthetic_v1` (repeat "
        "`--baseline` to select more)")
    lines.append(
        "- Regenerate this doc: `python3 -m "
        "benchmarks.harness_tool`")
    lines.append("")
    lines.append(
        "Every run prints its citation line first — "
        "`harness-tool run: panel=<panel> "
        "members=<run> passed=<passed> "
        "verdict=<verdict>` — then one member line "
        "per baseline and one line per problem. No "
        "wall time ever stands in the output, so a "
        "cited line reproduces verbatim.")
    lines.append("")
    lines.append("## The panels")
    lines.append("")
    for row in rows:
        lines.append(f"### Panel `{row['id']}`")
        lines.append("")
        lines.append(f"- Purpose: {row['purpose']}")
        lines.append(f"- Members: {len(row['members'])}")
        for member in row["members"]:
            lines.append(f"  - `{member}`")
        lines.append("")
    lines.append("## The rules")
    lines.append("")
    for rule in HARNESS_TOOL_RULES:
        lines.append(f"- `{rule['id']}`: {rule['rule']}")
    lines.append("")
    lines.append("## Filed limits")
    lines.append("")
    lines.append(
        "- The tool runs the harness's pinned "
        "FIXTURE baselines only — scripted synthetic "
        "suites through the real scorers on this VM. "
        "No live model is run; live behaviour is NOT "
        "PROVEN (TF-173).")
    lines.append(
        "- The audit panel's members are each also "
        "judged by their own module's checker in "
        "their own slices; this tool re-measures "
        "them against their pins and does not "
        "re-run those checkers.")
    lines.append(
        "- Run output files no timing: walls under "
        "contention are not baselines (TF-121).")
    lines.append("")
    lines.append(
        "Proof grade: FIXTURE — pinned "
        "synthetic-fixture baselines re-measured live "
        "on this VM; no live model, never claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_harness_tool_doc() -> Path:
    """Regenerate the filed doc by command: render
    from the filed data and write the doc, byte for
    byte. Returns the doc's path."""
    doc = _REPO_ROOT / HARNESS_TOOL_DOC
    doc.write_text(
        render_harness_tool_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The one command's entry point:
    `python3 -m benchmarks.harness_tool`.

    No selection argument regenerates the filed
    doc (the sibling shape); --list-panels lists
    the panels; --panel / --baseline run a
    selection and print its citable lines.
    """
    parser = argparse.ArgumentParser(
        prog="benchmarks.harness_tool",
        description="Run the harness panel a slice needs.")
    parser.add_argument("--list-panels", action="store_true")
    parser.add_argument("--panel", default=None)
    parser.add_argument(
        "--baseline", action="append", default=None)
    args = parser.parse_args(argv)
    if args.list_panels:
        print(f"harness-tool panels: {len(PANEL_ROWS)}")
        for row in PANEL_ROWS:
            print(
                f"harness-tool panel id={row['id']} "
                f"members={len(row['members'])}")
        return 0
    if args.panel is not None or args.baseline is not None:
        result = run_harness_tool(
            panel_id=args.panel, baseline_ids=args.baseline)
        for line in citation_lines(result):
            print(line)
        return 0
    doc = write_harness_tool_doc()
    print(
        f"harness tool regenerated: "
        f"{len(PANEL_ROWS)} panels -> {HARNESS_TOOL_DOC}")
    return 0


# ── The live checker ────────────────────────────────────────────────────────

def harness_tool_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: format, rules.
    if HARNESS_TOOL_FORMAT != "thoughtforge-harness-tool/1":
        problems.append(
            f"HARNESS_TOOL_FORMAT drifted: "
            f"{HARNESS_TOOL_FORMAT}")
    rule_ids = [r["id"] for r in HARNESS_TOOL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate HARNESS_TOOL_RULES ids")
    for needed in (
            "panels_are_literal_data_bound_to_the_real_estate",
            "one_command_runs_the_panel_a_slice_needs",
            "citation_lines_are_deterministic_and_verbatim",
            "a_member_passes_by_reproducing_its_pin",
            "pass_needs_the_full_panel_all_passing",
            "empty_and_unknown_selections_are_named_review",
            "probes_prove_the_validator_can_fail",
            "the_harness_tool_baseline_is_not_gated_by_tf129",
            "no_new_benchmark_input_was_added"):
        if needed not in rule_ids:
            problems.append(f"HARNESS_TOOL_RULES lost {needed}")

    # Panel shape: unique ids, every row valid
    # against the estate (purely).
    panel_ids = [r["id"] for r in PANEL_ROWS]
    if len(panel_ids) != len(set(panel_ids)):
        problems.append("duplicate PANEL_ROWS ids")
    for row in PANEL_ROWS:
        problems.extend(panel_problems(row))

    # The three filed panels bind the live
    # registries exactly.
    full = panel_by_id("full_panel")
    if full is None or list(full["members"]) != [
            b["id"] for b in _harness.BASELINES]:
        problems.append(
            "the full panel's members drifted from "
            "the harness's pinned baselines")
    gate = panel_by_id("gate_panel")
    if gate is None or list(gate["members"]) != list(
            _gate.GATED_BASELINE_IDS):
        problems.append(
            "the gate panel's members drifted from "
            "TF-129's gated baselines")
    audit = panel_by_id("audit_panel")
    if audit is None or list(audit["members"]) != list(
            _gate.UNGATED_BASELINE_IDS):
        problems.append(
            "the audit panel's members drifted from "
            "TF-129's ungated baselines")

    # A fresh full run files PASS, and its citation
    # line validates against its own data.
    result = run_harness_tool()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live full-panel run does not file "
            f"PASS on this tree: {result.verdict} "
            f"({result.members_passed} passed of "
            f"{len(result.rows)} members)")
    for row in result.rows:
        problems.extend(
            f"full run: {p}"
            for p in member_result_problems(row))
    problems.extend(citation_problems(
        result.citation_line, result.run_data))

    # Partial, empty and unknown selections REVIEW,
    # named, never a crash.
    subset = run_harness_tool(
        baseline_ids=("persona_consistency_synthetic_v1",))
    if subset.members_run != 1 or subset.verdict != "REVIEW" \
            or subset.passes:
        problems.append(
            "a single-baseline selection does not "
            "file its member / REVIEW mechanically")
    gate_run = run_harness_tool(panel_id="gate_panel")
    if gate_run.members_run != len(_gate.GATED_BASELINE_IDS) \
            or gate_run.verdict != "REVIEW":
        problems.append(
            "the gate panel does not file its nine "
            "members / REVIEW (PASS lives on the "
            "full panel only)")
    empty = run_harness_tool(baseline_ids=())
    if empty.rows or empty.verdict != "REVIEW" \
            or empty.passes or not empty.selection_problems:
        problems.append(
            "an empty selection does not file 0 / "
            "REVIEW with its naming")
    unknown_panel = run_harness_tool(
        panel_id="totally_bogus_panel")
    if unknown_panel.rows \
            or unknown_panel.verdict != "REVIEW" \
            or not unknown_panel.selection_problems:
        problems.append(
            "an unknown panel is not named / REVIEW")
    unknown_member = run_harness_tool(
        baseline_ids=("totally_bogus_baseline_v1",))
    if len(unknown_member.rows) != 1 \
            or unknown_member.rows[0]["ran"] \
            or unknown_member.verdict != "REVIEW":
        problems.append(
            "an unknown baseline is not named / "
            "REVIEW in its member row")

    # Determinism of the render + the filed doc is
    # the render, verbatim.
    if render_harness_tool_markdown() != \
            render_harness_tool_markdown(PANEL_ROWS):
        problems.append(
            "the render is not deterministic: two "
            "renders over the filed data differ")
    doc = _REPO_ROOT / HARNESS_TOOL_DOC
    if not doc.exists():
        problems.append(
            f"the write-up {HARNESS_TOOL_DOC} is "
            f"missing (regenerate with python3 -m "
            f"benchmarks.harness_tool)")
    elif doc.read_text(encoding="utf-8") != \
            render_harness_tool_markdown():
        problems.append(
            f"the filed doc {HARNESS_TOOL_DOC} is "
            f"not the render verbatim (regenerate "
            f"with python3 -m benchmarks.harness_tool)")

    # The probes all behave.
    probes = run_tool_probes()
    if len(probes) != len(TOOL_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the tool probes do not all pass: "
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

    if "harness_tool" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the harness_tool entry")
    if HARNESS_TOOL_BASELINE_ID not in [
            b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the harness-tool baseline")
    if HARNESS_TOOL_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the harness-tool baseline entered "
            "TF-129's gated set (it is judged by this "
            "checker, not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the "
            f"harness tool: {earlier}")
    return problems


def harness_tool_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_tool_probes()
    full = panel_by_id("full_panel")
    gate = panel_by_id("gate_panel")
    audit = panel_by_id("audit_panel")
    return {
        "format": HARNESS_TOOL_FORMAT,
        "doc": HARNESS_TOOL_DOC,
        "panel_rows": [
            {**dict(r), "members": list(r["members"])}
            for r in PANEL_ROWS],
        "measured": {
            "total_panels": len(PANEL_ROWS),
            "full_panel_baselines": len(full["members"])
            if full else 0,
            "gate_panel_baselines": len(gate["members"])
            if gate else 0,
            "audit_panel_baselines": len(audit["members"])
            if audit else 0,
            "panel_member_slots": sum(
                len(r["members"]) for r in PANEL_ROWS),
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
        "rules": [dict(r) for r in HARNESS_TOOL_RULES],
    }


__all__ = [
    "HARNESS_TOOL_BASELINE_ID",
    "HARNESS_TOOL_DOC",
    "HARNESS_TOOL_FORMAT",
    "HARNESS_TOOL_RULES",
    "PANEL_ROWS",
    "TOOL_PROBES",
    "HarnessToolResult",
    "citation_line_for",
    "citation_lines",
    "citation_problems",
    "harness_tool_as_data",
    "harness_tool_problems",
    "main",
    "member_result_problems",
    "panel_by_id",
    "panel_problems",
    "render_harness_tool_markdown",
    "run_harness_tool",
    "run_tool_probes",
    "write_harness_tool_doc",
]

if __name__ == "__main__":
    sys.exit(main())
