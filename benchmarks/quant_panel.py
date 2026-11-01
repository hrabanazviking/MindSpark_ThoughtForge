"""
Quant-vs-quality panel, as data (TF-127).

The roadmap's rule: *quant-vs-quality panel
(consumes TF-067 mechanically).* TF-067 built the
measuring estate in ``src/`` — one prompt set
(``QUANT_PROMPTS``, 8 synthetic prompts), every
quant level the profiles + catalogue span
(``QUANT_LEVELS``), one scorer (``score_answer``),
one runner (``run_quant_panel``) and a published
overall curve (``CURVE``) computed from its
fixture corpus (``FIXTURE_RESPONSES``, 48 cells).
Its doc assigned the Part VII panel to TF-127:
"TF-127's quant-vs-quality panel consumes this
runner mechanically with live models; TF-068
reads the curve." TF-068's quant policy in turn
filed review triggers that "TF-127's live panel
may re-rank" the neighbouring levels and
"calibrate" the q2_k exceptions.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **TF-067's estate re-measured clean.**
    ``quant_quality_problems()`` returned [] and
    ``fixture_panel()`` reproduced ``CURVE``
    exactly: iq2_xs 0.0 (loss 1.0, 0 perfect, 7
    contradiction prompts), q2_k 0.4375 (loss
    0.5625, 2, 2), q4_k_m 0.8125 (loss 0.1875, 6,
    1), q5_k_m 0.7083 (loss 0.2917, 4, 1), q8_0
    1.0 (loss 0.0, 8, 0), fp16 1.0 (loss 0.0, 8,
    0) — with the one filed non-monotone step
    (q4_k_m out-scores q5_k_m) standing.
  * **No per-kind row existed anywhere.** The
    prompts carry kinds (fact 5, reasoning 2,
    instruction 1 — counted live from
    ``QUANT_PROMPTS``), TF-067's runner files
    per-prompt scores, and nothing in the estate
    had ever broken the curve down by kind, and
    no harness baseline pinned the curve: TF-067
    stood in the harness as a bound neighbour
    whose suite_slice note read "TF-127 consumes
    it mechanically", unbuilt. Filed as
    **D-BENCH-13** and fixed by this slice in the
    measuring sense (see below).
  * **No live model can run on this VM** — no
    GGUF in the tree and no ``llama_cpp``
    (TF-067 / TF-078's filings, re-probed this
    tick). The live panel TF-067's doc imagined
    therefore cannot run here, and this slice
    does not pretend otherwise: TF-068's filed
    re-rank / q2_k calibration shares are NOT
    discharged — no live ranking and no live
    calibration is claimed anywhere in this
    panel. What runs is TF-067's fixture panel,
    consumed mechanically, broken down by kind.
    Live quant-vs-quality of real models stays
    NOT PROVEN; TF-173 owns the real-hardware
    road.

What this module files, as verifiable data:

  * **Mechanical consumption, stated exactly:**
    the panel imports TF-067's ``run_quant_panel``,
    ``fixture_generate``, ``fixture_panel``,
    ``QUANT_LEVELS``, ``QUANT_PROMPTS``,
    ``REFERENCE_QUANT``, ``CURVE``,
    ``NON_MONOTONE_STEPS``, ``score_answer`` and
    ``quant_quality_problems`` and reuses every
    one of them. It copies NO corpus, NO scorer
    and NO curve: edit TF-067's corpus and this
    panel's recomputation drifts with it, by
    design (the checker names that drift).
  * ``PANEL_ROWS`` — TF-067's overall curve as
    this panel's literal rows (per level: prompts,
    mean score, loss vs the panel's own fp16 row,
    perfect, contradiction prompts), recomputed
    live by the checker through TF-067's runner.
  * ``KIND_ROWS`` — the panel's own value beyond
    TF-067: the same live panel run broken down
    per kind (18 rows: 6 levels x fact /
    reasoning / instruction), each with its mean,
    its loss vs that kind's own fp16 mean, its
    perfect count and its contradiction count —
    all derived from the runner's per-prompt
    scores, filed as literal data and recomputed
    live by the checker.
  * ``FINDING`` — computed from the kind rows,
    filed whatever the numbers are (see the doc):
    reasoning carries the largest per-kind loss
    at both mid levels (0.5 at q4_k_m and at
    q5_k_m, against fact's 0.1 / 0.2 and
    instruction's 0.0 / 0.3333) and plateaus at
    0.5 from q2_k through q5_k_m, recovering only
    at q8_0; instruction collapses to 0.0 at
    q2_k (its single prompt contradicted) while
    fact and reasoning still hold 0.5 there.

The panel verdict is filed plainly: PASS only
when all 48 cells are measured, the overall rows
reproduce TF-067's published ``CURVE`` exactly
(rounded to 4 places, exactly as TF-067 files
them), and the kind rows reproduce ``KIND_ROWS``.
An empty panel or selection scores 0.0 / REVIEW
— no vacuous pass. A selection without the fp16
reference raises TF-067's own refusal, because a
loss curve with no measured reference is a
claim, not a measurement; partial prompt subsets
run mechanically through the same runner and
file REVIEW (their cells are measured, their
verdict is not the published panel's).

D-BENCH-13 is FIXED by this slice in the
measuring sense: quant-vs-quality across levels
now stands as a Part VII panel — per-kind
breakdown, published literal rows recomputed
live, and a pinned harness baseline. What is
NOT fixed, and is stated in the filed limits:
live quant-vs-quality of real models is NOT
PROVEN (no GGUF / llama_cpp on this VM, TF-078);
TF-068's live re-rank / calibration triggers
stand open on that road (TF-173).

Filed limits, stated plainly:

  * The panel measures TF-067's fixture corpus
    only — 8 synthetic prompts (TF-017; ``data/``
    never opened) through TF-067's real scorer.
    No row is a measurement of any real model at
    any quant level.
  * The per-kind counts are small and uneven by
    TF-067's filing — fact 5 prompts, reasoning
    2, instruction 1 — so a kind mean is a
    fixture shape, not a ranking instrument: the
    instruction rows rest on a single prompt, and
    the filed non-monotone q4_k_m / q5_k_m step
    is carried, never smoothed. Ranking
    neighbouring levels is NOT supported by this
    panel (TF-067's own filing, carried here).
  * TF-068's policy shares stand exactly as
    filed: this panel does not re-rank any level
    and does not calibrate q2_k — those triggers
    name a LIVE panel, and none ran.

The human-readable write-up is
``docs/architecture/QUANT_PANEL.md``.
Proof grade: FIXTURE — TF-067's synthetic fixture
corpus through its real scorer and runner on
this VM; no live model, never claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

QUANT_PANEL_FORMAT = "thoughtforge-quant-panel/1"
QUANT_PANEL_DOC = "docs/architecture/QUANT_PANEL.md"

# The real machinery, reused — never copied:
# TF-067's levels, prompts, scorer, runner,
# fixture generation, published curve and
# checker, bound in place.
from thoughtforge.inference.quant_quality import (  # noqa: E402
    CURVE,
    NON_MONOTONE_STEPS,
    QUANT_LEVELS,
    QUANT_PROMPTS,
    REFERENCE_QUANT,
    QuantPrompt,
    fixture_generate,
    fixture_panel,
    quant_quality_problems,
)
from thoughtforge.inference.quant_quality import (  # noqa: E402
    run_quant_panel as _tf067_run_quant_panel,
)

#: The prompt kinds, in first-appearance order in
#: TF-067's filed prompt set, with their counts
#: counted live from QUANT_PROMPTS by the checker
#: (fact 5, reasoning 2, instruction 1).
PANEL_KINDS: tuple[str, ...] = tuple(
    dict.fromkeys(p.kind for p in QUANT_PROMPTS))

KIND_PROMPT_COUNTS: dict[str, int] = {
    kind: sum(1 for p in QUANT_PROMPTS if p.kind == kind)
    for kind in PANEL_KINDS}

#: The published panel's cell count: every level
#: x every prompt through TF-067's runner.
TOTAL_CELLS = len(QUANT_LEVELS) * len(QUANT_PROMPTS)


# ── The panel measurement (TF-067's runner, mechanically) ───────────────────

def _overall_rows(panel: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"quant": r.quant, "prompts": r.prompts,
         "mean_score": r.mean_score,
         "loss_vs_reference": r.loss_vs_reference,
         "perfect": r.perfect,
         "contradiction_prompts": r.contradiction_prompts}
        for r in panel.rows)


def kind_rows_for(
    panel: Any, prompts: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Break one live TF-067 panel run down per
    kind, from the runner's own per-prompt scores:
    per (level, kind) the mean, the loss vs that
    kind's own fp16 mean, perfect and
    contradiction counts."""
    prompts = QUANT_PROMPTS if prompts is None else prompts
    kinds = tuple(dict.fromkeys(p.kind for p in prompts))
    by_quant = {r.quant: r for r in panel.rows}
    reference = by_quant[REFERENCE_QUANT]
    ref_means: dict[str, float] = {}
    for kind in kinds:
        scores = [s.score for s, p in zip(
            reference.per_prompt, prompts) if p.kind == kind]
        ref_means[kind] = sum(scores) / len(scores)
    rows: list[dict[str, Any]] = []
    for quant_row in panel.rows:
        for kind in kinds:
            scores = [s for s, p in zip(
                quant_row.per_prompt, prompts) if p.kind == kind]
            mean = sum(s.score for s in scores) / len(scores)
            rows.append({
                "quant": quant_row.quant, "kind": kind,
                "prompts": len(scores), "mean_score": mean,
                "loss_vs_fp16": ref_means[kind] - mean,
                "perfect": sum(1 for s in scores if s.score == 1.0),
                "contradiction_prompts": sum(
                    1 for s in scores if s.contradictions),
            })
    return tuple(rows)


def compute_quant_panel(
    levels: Any = None, prompts: Any = None,
    generate: Callable | None = None,
) -> tuple[tuple[dict[str, Any], ...],
           tuple[dict[str, Any], ...]]:
    """The whole panel, measured live through
    TF-067's real runner (overall rows, kind
    rows). A selection without the fp16 reference
    raises TF-067's own refusal, mechanically."""
    levels = QUANT_LEVELS if levels is None else levels
    prompts = QUANT_PROMPTS if prompts is None else prompts
    generate = fixture_generate if generate is None else generate
    panel = _tf067_run_quant_panel(generate, levels, prompts)
    return _overall_rows(panel), kind_rows_for(panel, prompts)


def finding_for(
    overall_rows: Any, kind_rows: Any,
) -> dict[str, Any]:
    """The panel's finding, computed from the
    rows: which kind carries the largest loss at
    each mid level, reasoning's plateau across
    q2_k..q5_k_m, instruction's q2_k score, and
    the non-monotone steps in the overall means."""
    def _kind_row(quant: str, kind: str) -> dict[str, Any] | None:
        cells = [r for r in kind_rows
                 if r["quant"] == quant and r["kind"] == kind]
        return cells[0] if cells else None

    def _largest_loss_kind(quant: str) -> str | None:
        cells = [r for r in kind_rows if r["quant"] == quant]
        if not cells:
            return None
        return max(cells, key=lambda r: r["loss_vs_fp16"])["kind"]

    plateau_levels = ("q2_k", "q4_k_m", "q5_k_m")
    reasoning_rows = [_kind_row(q, "reasoning") for q in plateau_levels]
    reasoning = ([r["mean_score"] for r in reasoning_rows]
                 if all(r is not None for r in reasoning_rows)
                 else [])
    instruction_q2 = _kind_row("q2_k", "instruction")
    means = {r["quant"]: r["mean_score"] for r in overall_rows}
    steps = [[a, b] for a, b in zip(QUANT_LEVELS, QUANT_LEVELS[1:])
             if a in means and b in means and means[a] > means[b]]
    return {
        "largest_loss_kind_at_q4_k_m": _largest_loss_kind("q4_k_m"),
        "largest_loss_kind_at_q5_k_m": _largest_loss_kind("q5_k_m"),
        "reasoning_plateau_levels": list(plateau_levels),
        "reasoning_plateau_score": (
            reasoning[0] if reasoning
            and all(v == reasoning[0] for v in reasoning) else None),
        "instruction_score_at_q2_k": (
            instruction_q2["mean_score"]
            if instruction_q2 is not None else None),
        "non_monotone_steps": steps,
    }


# ── The published panel (literal data, TF-067's CURVE precedent) ────────────
#
# Measured live on this VM at banking (2026-11-01)
# by running TF-067's fixture corpus through its
# real runner — twice, identical. The checker
# recomputes every row live and names any drift.
# PANEL_ROWS is TF-067's CURVE in row form (the
# checker also pins the rounded-to-4 equality
# against CURVE itself); KIND_ROWS is this
# panel's own breakdown. Filed whatever the
# numbers are: see the doc for the honest shapes
# (reasoning's 0.5 plateau, instruction's q2_k
# collapse, the non-monotone q4/q5 step).

PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {"quant": "iq2_xs", "prompts": 8, "mean_score": 0.0,
     "loss_vs_reference": 1.0, "perfect": 0,
     "contradiction_prompts": 7},
    {"quant": "q2_k", "prompts": 8, "mean_score": 0.4375,
     "loss_vs_reference": 0.5625, "perfect": 2,
     "contradiction_prompts": 2},
    {"quant": "q4_k_m", "prompts": 8, "mean_score": 0.8125,
     "loss_vs_reference": 0.1875, "perfect": 6,
     "contradiction_prompts": 1},
    {"quant": "q5_k_m", "prompts": 8,
     "mean_score": 0.7083333333333334,
     "loss_vs_reference": 0.29166666666666663, "perfect": 4,
     "contradiction_prompts": 1},
    {"quant": "q8_0", "prompts": 8, "mean_score": 1.0,
     "loss_vs_reference": 0.0, "perfect": 8,
     "contradiction_prompts": 0},
    {"quant": "fp16", "prompts": 8, "mean_score": 1.0,
     "loss_vs_reference": 0.0, "perfect": 8,
     "contradiction_prompts": 0},
)

KIND_ROWS: tuple[dict[str, Any], ...] = (
    {"quant": "iq2_xs", "kind": "fact", "prompts": 5,
     "mean_score": 0.0, "loss_vs_fp16": 1.0, "perfect": 0,
     "contradiction_prompts": 5},
    {"quant": "iq2_xs", "kind": "reasoning", "prompts": 2,
     "mean_score": 0.0, "loss_vs_fp16": 1.0, "perfect": 0,
     "contradiction_prompts": 2},
    {"quant": "iq2_xs", "kind": "instruction", "prompts": 1,
     "mean_score": 0.0, "loss_vs_fp16": 1.0, "perfect": 0,
     "contradiction_prompts": 0},
    {"quant": "q2_k", "kind": "fact", "prompts": 5,
     "mean_score": 0.5, "loss_vs_fp16": 0.5, "perfect": 1,
     "contradiction_prompts": 1},
    {"quant": "q2_k", "kind": "reasoning", "prompts": 2,
     "mean_score": 0.5, "loss_vs_fp16": 0.5, "perfect": 1,
     "contradiction_prompts": 0},
    {"quant": "q2_k", "kind": "instruction", "prompts": 1,
     "mean_score": 0.0, "loss_vs_fp16": 1.0, "perfect": 0,
     "contradiction_prompts": 1},
    {"quant": "q4_k_m", "kind": "fact", "prompts": 5,
     "mean_score": 0.9, "loss_vs_fp16": 0.09999999999999998,
     "perfect": 4, "contradiction_prompts": 0},
    {"quant": "q4_k_m", "kind": "reasoning", "prompts": 2,
     "mean_score": 0.5, "loss_vs_fp16": 0.5, "perfect": 1,
     "contradiction_prompts": 1},
    {"quant": "q4_k_m", "kind": "instruction", "prompts": 1,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 1,
     "contradiction_prompts": 0},
    {"quant": "q5_k_m", "kind": "fact", "prompts": 5,
     "mean_score": 0.8, "loss_vs_fp16": 0.19999999999999996,
     "perfect": 3, "contradiction_prompts": 0},
    {"quant": "q5_k_m", "kind": "reasoning", "prompts": 2,
     "mean_score": 0.5, "loss_vs_fp16": 0.5, "perfect": 1,
     "contradiction_prompts": 1},
    {"quant": "q5_k_m", "kind": "instruction", "prompts": 1,
     "mean_score": 0.6666666666666666,
     "loss_vs_fp16": 0.33333333333333337, "perfect": 0,
     "contradiction_prompts": 0},
    {"quant": "q8_0", "kind": "fact", "prompts": 5,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 5,
     "contradiction_prompts": 0},
    {"quant": "q8_0", "kind": "reasoning", "prompts": 2,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 2,
     "contradiction_prompts": 0},
    {"quant": "q8_0", "kind": "instruction", "prompts": 1,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 1,
     "contradiction_prompts": 0},
    {"quant": "fp16", "kind": "fact", "prompts": 5,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 5,
     "contradiction_prompts": 0},
    {"quant": "fp16", "kind": "reasoning", "prompts": 2,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 2,
     "contradiction_prompts": 0},
    {"quant": "fp16", "kind": "instruction", "prompts": 1,
     "mean_score": 1.0, "loss_vs_fp16": 0.0, "perfect": 1,
     "contradiction_prompts": 0},
)

FINDING: dict[str, Any] = {
    "largest_loss_kind_at_q4_k_m": "reasoning",
    "largest_loss_kind_at_q5_k_m": "reasoning",
    "reasoning_plateau_levels": ["q2_k", "q4_k_m", "q5_k_m"],
    "reasoning_plateau_score": 0.5,
    "instruction_score_at_q2_k": 0.0,
    "non_monotone_steps": [["q4_k_m", "q5_k_m"]],
}

PANEL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "consumes_tf067_mechanically",
     "rule": "the panel runs TF-067's own "
             "run_quant_panel over its own "
             "fixture_generate / QUANT_LEVELS / "
             "QUANT_PROMPTS and derives every row "
             "from that run's per-prompt scores — "
             "it copies no corpus, no scorer and "
             "no curve from src/"},
    {"id": "panel_rows_are_literal_and_recomputed",
     "rule": "PANEL_ROWS / KIND_ROWS / FINDING are "
             "filed as literal data measured at "
             "banking and recomputed live by the "
             "checker — a row that moves is a "
             "finding, never silently re-filed "
             "(TF-067's CURVE precedent)"},
    {"id": "verdict_passes_only_on_exact_curve_reproduction",
     "rule": "PASS requires all 48 cells measured "
             "and the overall rows reproducing "
             "TF-067's published CURVE exactly, "
             "rounded to 4 places as TF-067 files "
             "them, and the kind rows reproducing "
             "KIND_ROWS exactly"},
    {"id": "per_kind_breakdown_is_the_panels_own_value",
     "rule": "KIND_ROWS break the one live panel "
             "run down per prompt kind (fact / "
             "reasoning / instruction) from the "
             "runner's per-prompt scores, each "
             "kind's loss measured against that "
             "kind's own fp16 mean — the breakdown "
             "TF-067's overall curve never filed"},
    {"id": "loss_is_against_the_panels_own_fp16",
     "rule": "every loss is a measured fp16 row "
             "minus a measured level row (overall, "
             "and per kind against the kind's fp16 "
             "mean) — never against an assumed 1.0; "
             "a selection without fp16 raises "
             "TF-067's own refusal"},
    {"id": "non_monotone_step_carried_not_smoothed",
     "rule": "the measured q4_k_m > q5_k_m step "
             "stands in the rows and in the finding "
             "exactly as TF-067 filed it in "
             "NON_MONOTONE_STEPS — recomputed live, "
             "never edited away"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty panel or selection scores "
             "0.0 and files REVIEW; a partial "
             "selection files REVIEW however its "
             "measured cells score"},
    {"id": "neighbouring_levels_are_not_ranked",
     "rule": "8 prompts cannot rank neighbouring "
             "levels (TF-067's filing): the per-kind "
             "counts are fact 5 / reasoning 2 / "
             "instruction 1, the instruction rows "
             "rest on a single prompt, and no row "
             "here re-ranks any level or calibrates "
             "TF-068's q2_k exceptions"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "the panel measures TF-067's "
             "synthetic fixture corpus (TF-017): "
             "canon is never a benchmark input and "
             "data/ was never opened"},
    {"id": "live_quant_quality_is_not_proven",
     "rule": "no live model ran for any row — no "
             "GGUF in the tree, no llama_cpp on "
             "this VM (TF-078): live quant-vs-"
             "quality of real models is NOT PROVEN "
             "and TF-068's live re-rank / calibration "
             "triggers stand open on TF-173's "
             "real-hardware road"},
)


# ── The panel run ───────────────────────────────────────────────────────────

@dataclass
class QuantPanelResult:
    """One full panel run: overall rows, kind
    rows, the finding, and the verdict data."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    kind_rows: list[dict[str, Any]] = field(default_factory=list)
    finding: dict[str, Any] = field(default_factory=dict)
    total_cells: int = 0
    overall_score: float = 0.0

    @property
    def curve_reproduces(self) -> bool:
        """The overall rows reproduce TF-067's
        published CURVE exactly (rounded to 4
        places, as TF-067 files them)."""
        published = {
            (q, round(mean, 4), round(loss, 4), perfect, contra)
            for q, mean, loss, perfect, contra in CURVE}
        recomputed = {
            (r["quant"], round(r["mean_score"], 4),
             round(r["loss_vs_reference"], 4), r["perfect"],
             r["contradiction_prompts"]) for r in self.rows}
        return published == recomputed

    @property
    def kind_rows_reproduce(self) -> bool:
        return [dict(r) for r in self.kind_rows] == \
            [dict(r) for r in KIND_ROWS]

    @property
    def verdict(self) -> str:
        """PASS only on the full 48-cell panel
        reproducing CURVE and KIND_ROWS exactly."""
        if self.total_cells != TOTAL_CELLS:
            return "REVIEW"
        return "PASS" if (self.curve_reproduces
                          and self.kind_rows_reproduce) else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_quant_panel(
    levels: Any = None, prompts: Any = None,
    generate: Callable | None = None,
) -> QuantPanelResult:
    """Run the whole panel through TF-067's
    runner (filed levels/prompts/generation
    default).

    Any explicit level/prompt selection may be
    given — including an empty one, which scores
    0.0 and keeps the panel at REVIEW (no vacuous
    pass). A non-empty selection without the fp16
    reference raises TF-067's own ValueError.
    """
    levels = QUANT_LEVELS if levels is None else levels
    prompts = QUANT_PROMPTS if prompts is None else prompts
    if not levels or not prompts:
        return QuantPanelResult(
            rows=[], kind_rows=[], finding=finding_for([], []),
            total_cells=0, overall_score=0.0)
    overall_rows, kind_rows = compute_quant_panel(
        levels, prompts, generate)
    total = sum(r["prompts"] for r in overall_rows)
    overall = (
        sum(r["mean_score"] for r in overall_rows) / len(overall_rows)
        if overall_rows else 0.0)
    return QuantPanelResult(
        rows=[dict(r) for r in overall_rows],
        kind_rows=[dict(r) for r in kind_rows],
        finding=finding_for(overall_rows, kind_rows),
        total_cells=total, overall_score=overall)


# ── The live checker ────────────────────────────────────────────────────────

def quant_panel_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # TF-067's own estate, re-judged live: the
    # panel stands on it mechanically.
    upstream = quant_quality_problems()
    if upstream:
        problems.append(f"TF-067's estate is not clean: {upstream}")

    # Levels / kinds shape, counted live.
    if [r["quant"] for r in PANEL_ROWS] != list(QUANT_LEVELS):
        problems.append("PANEL_ROWS are not in QUANT_LEVELS order")
    counted = {kind: sum(1 for p in QUANT_PROMPTS if p.kind == kind)
               for kind in PANEL_KINDS}
    if counted != KIND_PROMPT_COUNTS or counted != {
            "fact": 5, "reasoning": 2, "instruction": 1}:
        problems.append(
            f"kind prompt counts drifted: {counted} vs filed "
            f"{KIND_PROMPT_COUNTS}")
    if TOTAL_CELLS != 48:
        problems.append(f"TOTAL_CELLS is {TOTAL_CELLS}, not 48")
    rule_ids = [r["id"] for r in PANEL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate PANEL_RULES ids")
    for needed in ("consumes_tf067_mechanically",
                   "panel_rows_are_literal_and_recomputed",
                   "verdict_passes_only_on_exact_curve_reproduction",
                   "per_kind_breakdown_is_the_panels_own_value",
                   "loss_is_against_the_panels_own_fp16",
                   "non_monotone_step_carried_not_smoothed",
                   "no_vacuous_pass_anywhere",
                   "neighbouring_levels_are_not_ranked",
                   "fixtures_are_synthetic_only",
                   "live_quant_quality_is_not_proven"):
        if needed not in rule_ids:
            problems.append(f"PANEL_RULES lost {needed}")

    # TF-067's fixture panel, consumed directly,
    # agrees with the panel's own computation.
    direct = _overall_rows(fixture_panel())
    if [dict(r) for r in direct] != [dict(r) for r in PANEL_ROWS]:
        problems.append(
            "TF-067's fixture_panel() disagrees with PANEL_ROWS")

    # The published panel: literal rows recomputed
    # live, twice — filing, recomputation and
    # determinism in one pass.
    live_rows, live_kinds = compute_quant_panel()
    if [dict(r) for r in live_rows] != [dict(r) for r in PANEL_ROWS]:
        problems.append(
            "PANEL_ROWS drifted from the live recomputation")
    if [dict(r) for r in live_kinds] != [dict(r) for r in KIND_ROWS]:
        problems.append(
            "KIND_ROWS drifted from the live recomputation")
    if finding_for(live_rows, live_kinds) != FINDING:
        problems.append("FINDING drifted from the filed rows")
    second_rows, second_kinds = compute_quant_panel()
    if (second_rows, second_kinds) != (live_rows, live_kinds):
        problems.append("the panel is not deterministic: two "
                        "live runs differ")

    # The non-monotone filing, recomputed.
    if [list(s) for s in NON_MONOTONE_STEPS] != \
            FINDING["non_monotone_steps"]:
        problems.append(
            "the finding's non-monotone steps disagree with "
            "TF-067's NON_MONOTONE_STEPS")

    # A fresh full run files PASS; an empty panel
    # files 0.0 / REVIEW; a partial prompt subset
    # measures its cells and files REVIEW.
    result = run_quant_panel()
    if result.total_cells != TOTAL_CELLS or result.verdict != "PASS" \
            or not result.passes:
        problems.append(
            f"the live panel run does not file {TOTAL_CELLS} cells "
            f"/ PASS: {result.total_cells} / {result.verdict}")
    empty = run_quant_panel(levels=())
    if empty.total_cells != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty panel does not file 0.0 / REVIEW")
    fact_only = tuple(p for p in QUANT_PROMPTS if p.kind == "fact")
    subset = run_quant_panel(prompts=fact_only)
    if subset.total_cells != len(QUANT_LEVELS) * len(fact_only) \
            or subset.verdict != "REVIEW":
        problems.append(
            "a fact-only prompt subset does not measure its "
            "cells / REVIEW mechanically")

    # Every earlier pinned baseline preserved exactly.
    from benchmarks.harness import baseline_problems

    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"earlier harness baselines moved: {earlier}")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / QUANT_PANEL_DOC
    if not doc.exists():
        problems.append(f"the write-up {QUANT_PANEL_DOC} is missing")
    return problems


def quant_panel_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_quant_panel()
    return {
        "format": QUANT_PANEL_FORMAT,
        "doc": QUANT_PANEL_DOC,
        "levels": list(QUANT_LEVELS),
        "reference_quant": REFERENCE_QUANT,
        "kinds": list(PANEL_KINDS),
        "kind_prompt_counts": dict(KIND_PROMPT_COUNTS),
        "curve": [list(row) for row in CURVE],
        "non_monotone_steps": [list(s) for s in NON_MONOTONE_STEPS],
        "rows": [dict(r) for r in PANEL_ROWS],
        "kind_rows": [dict(r) for r in KIND_ROWS],
        "finding": dict(FINDING),
        "measured": {
            "total_cells": result.total_cells,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in PANEL_RULES],
    }


__all__ = [
    "FINDING",
    "KIND_PROMPT_COUNTS",
    "KIND_ROWS",
    "PANEL_KINDS",
    "PANEL_ROWS",
    "PANEL_RULES",
    "QUANT_PANEL_DOC",
    "QUANT_PANEL_FORMAT",
    "TOTAL_CELLS",
    "QuantPanelResult",
    "QuantPrompt",
    "compute_quant_panel",
    "finding_for",
    "kind_rows_for",
    "quant_panel_as_data",
    "quant_panel_problems",
    "run_quant_panel",
]
