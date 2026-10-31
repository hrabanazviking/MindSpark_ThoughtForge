"""TF-067 — Quant quality fixtures: one prompt set, every quant
level, one scorer, and the first loss curve in the tree that is
*computed* rather than copied.

TF-066 filed every loss claim in the shipped tree and found all
of them unmeasured: catalogue copy ("Q8 — near full-precision
quality", "Q5 — higher fidelity", "IQ2 — extreme
compression"), an inventory line, a range statement, and a
research expectation. This slice builds the measuring estate
those claims always needed:

- **The levels** — ``QUANT_LEVELS``: exactly the quant levels
  the shipped system actually spans, read against TF-066's
  live surfaces: the hardware profiles recommend ``q2_k`` /
  ``q4_k_m`` / ``q8_0`` / ``fp16`` and the download catalogue
  declares ``Q4_K_M`` / ``Q5_K_M`` / ``Q8_0`` / ``IQ2_XS``.
  Ordered lowest-compression-fidelity first (``iq2_xs``) to the
  unquantised reference (``fp16``). No level is in the panel
  that no profile or catalogue entry names, and none they name
  is missing — the checker re-derives both surfaces live.
- **The prompts** — ``QUANT_PROMPTS``: eight synthetic prompts
  (TF-017's Lantern Hollow world plus two arithmetic reasoning
  prompts; no canon anywhere), each with required phrase
  *groups* and *forbidden* phrases in TF-028's anchor shape,
  and a reference answer that scores a clean 1.0. The SAME
  prompts run at every level — that sameness is the fixture's
  whole point.
- **The scorer** — :func:`score_answer` is a pure function in
  the Part VII harness shape TF-028/TF-063 established (and
  TF-121 will unify): retention = satisfied required groups /
  required groups; a forbidden phrase — the answer
  contradicting the fact it was asked about — zeroes the
  prompt, because a confident wrong answer is worse than a
  partial one. An empty panel is an error, never a vacuous
  pass (D-BENCH-4's lesson).
- **The runner** — :func:`run_quant_panel` takes any
  ``generate(quant, prompt) -> str`` callable, runs the
  identical prompt set at each level, and returns the curve:
  per level, the mean score, the perfect-prompt count, the
  contradiction count, and **loss = reference mean − level
  mean**, where the reference is the panel's own ``fp16`` row.
  TF-127's quant-vs-quality panel consumes this runner
  mechanically with live models; TF-068 reads the curve.
- **The published curve** — ``CURVE`` is filed as literal data
  and the checker recomputes it from the corpus on every run:
  edit the corpus without re-publishing and the checker
  fails. The corpus it is computed from is
  ``FIXTURE_RESPONSES``: one recorded-style response per
  (prompt, level), authored as fixtures in the failure shapes
  quantised small models are known to take — dropped anchors,
  terse partials, confident contradictions, empty hedges —
  deliberately *not* uniform: q2_k answers two prompts
  perfectly that q5_k_m fumbles, so the measured curve is not
  monotone and no reader may pretend a fixture corpus proves
  more than it does.

**Proof grade, stated plainly: FIXTURE.** No GGUF exists in
this tree, ``llama_cpp`` is not installed on this machine, and
no live model was run. The curve below is a real measurement
*of the fixture corpus through the real scorer* — it proves
the harness measures, discriminates, and catches
contradictions, and it gives TF-068 a curve *shape* to reason
from. It is NOT a measurement of any real model's quantisation
loss: the live curve is TF-127's (panel) and TF-126's
(honesty panel) road, and TF-066's ``LOSS_CLAIMS`` stay
``measured_in_tree == False`` for exactly that reason — this
slice measures fixtures, not the catalogue's models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.turboquant_map import KNOWN_QUANTS, profile_quants

PANEL_FORMAT = "thoughtforge-quant-quality-panel/1"

#: The reference level: loss is measured against the panel's
#: own fp16 row, never against an assumed 1.0.
REFERENCE_QUANT = "fp16"

# ── The levels ──────────────────────────────────────────────────────────────
# Ordered by compression (most compressed first), as the
# profiles and catalogue span them. Bits-per-weight figures are
# NOT stated: TF-066 ruled the tree states none and invents none.

QUANT_LEVELS: tuple[str, ...] = (
    "iq2_xs",
    "q2_k",
    "q4_k_m",
    "q5_k_m",
    "q8_0",
    "fp16",
)

# ── The prompts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuantPrompt:
    """One fixture prompt: the question, the groups a correct
    answer must satisfy, the phrases that contradict it, and
    the reference answer (scores 1.0 by construction, pinned)."""

    prompt_id: str
    kind: str  # fact | reasoning | instruction
    text: str
    required: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    reference: str


QUANT_PROMPTS: tuple[QuantPrompt, ...] = (
    QuantPrompt(
        "ferry_minutes", "fact",
        "How long does the ferry crossing at Lantern Hollow take in calm water?",
        required=(("eleven minutes", "11 minutes"),),
        forbidden=("twelve minutes", "twenty minutes", "five minutes", "ten minutes"),
        reference="The ferry crossing at Lantern Hollow takes eleven minutes in calm water.",
    ),
    QuantPrompt(
        "red_lantern", "fact",
        "What sign shows the crossing is closed when the River Sarn runs high?",
        required=(("red lantern",), ("runs high", "river is high", "high water")),
        forbidden=("blue lantern", "green lantern", "white lantern"),
        reference="When the River Sarn runs high, a red lantern hangs at the ferry post and the crossing is closed.",
    ),
    QuantPrompt(
        "river_record", "fact",
        "How does Wrenna record the river level?",
        required=(("chalk",), ("mill door",)),
        forbidden=("in a diary", "in a ledger", "on her tablet"),
        reference="Wrenna records the river level each morning in chalk on the mill door.",
    ),
    QuantPrompt(
        "fare_maths", "reasoning",
        "A ferryman carries 4 passengers per crossing and makes 3 crossings "
        "before noon. The fare is 2 coins per passenger. How many coins "
        "does he collect before noon?",
        required=(("24",), ("coins",)),
        forbidden=("12 coins", "18 coins", "8 coins"),
        reference="He carries 4 x 3 = 12 passengers, and at 2 coins each that is 24 coins.",
    ),
    QuantPrompt(
        "oil_maths", "reasoning",
        "Wrenna lights 6 lanterns. Each lantern burns 3 measures of oil "
        "per night. She has 40 measures. After two nights, how many "
        "measures remain?",
        required=(("4 measures", "four measures"),),
        forbidden=("22 measures", "2 measures remain", "10 measures"),
        reference="Each night burns 18 measures; two nights burn 36, so 4 measures remain.",
    ),
    QuantPrompt(
        "crossing_rules", "instruction",
        "List the three crossing rules for a traveller, numbered, one per line.",
        required=(
            ("eleven minutes", "11 minutes"),
            ("second tide bell",),
            ("red lantern",),
        ),
        forbidden=("white lantern", "runs all night"),
        reference=(
            "1. The crossing takes eleven minutes in calm water.\n"
            "2. No ferry runs after the second tide bell.\n"
            "3. A red lantern means the river runs high and the crossing is closed."
        ),
    ),
    QuantPrompt(
        "keeper_dusk", "fact",
        "Who keeps the lanterns of Lantern Hollow, and when does she light them?",
        required=(("wrenna",), ("dusk",)),
        forbidden=("tomas keeps the lanterns", "at dawn"),
        reference="Wrenna keeps the lanterns of Lantern Hollow and lights them at dusk.",
    ),
    QuantPrompt(
        "tide_bell", "fact",
        "How many times does the tide bell ring, and when?",
        required=(("twice", "two times", "rings two"), ("dusk",)),
        forbidden=("rings once", "three times", "at dawn"),
        reference="The tide bell rings twice at dusk.",
    ),
)

PROMPTS_BY_ID: dict[str, QuantPrompt] = {p.prompt_id: p for p in QUANT_PROMPTS}

# ── The fixture corpus ──────────────────────────────────────────────────────
# One recorded-style response per (prompt, level). Failure
# shapes are authored per cell (see module docstring): dropped
# anchors, terse partials, confident contradictions, hedges.
# The scorer — never the authorship — decides the curve.

FIXTURE_RESPONSES: dict[tuple[str, str], str] = {
    # ferry_minutes
    ("ferry_minutes", "fp16"): "The ferry crossing at Lantern Hollow takes eleven minutes in calm water.",
    ("ferry_minutes", "q8_0"): "In calm water the crossing at Lantern Hollow takes eleven minutes.",
    ("ferry_minutes", "q5_k_m"): "The ferry crossing takes eleven minutes in calm water.",
    ("ferry_minutes", "q4_k_m"): "Eleven minutes, in calm water, between the willow narrows and the mill landing.",
    ("ferry_minutes", "q2_k"): "The ferry crossing takes twelve minutes in calm water.",
    ("ferry_minutes", "iq2_xs"): "Ferry crossing, calm water, ten minutes, maybe more.",
    # red_lantern
    ("red_lantern", "fp16"): "When the River Sarn runs high, a red lantern hangs at the ferry post and the crossing is closed.",
    ("red_lantern", "q8_0"): "A red lantern at the ferry post: the River Sarn runs high and the crossing is closed.",
    ("red_lantern", "q5_k_m"): "A red lantern at the ferry post means the crossing is closed.",
    ("red_lantern", "q4_k_m"): "When the river is high, Wrenna hangs a red lantern at the ferry post to show the crossing is closed.",
    ("red_lantern", "q2_k"): "A red lantern hangs when the river is high.",
    ("red_lantern", "iq2_xs"): "A blue lantern hangs at the post when the water rises.",
    # river_record
    ("river_record", "fp16"): "Wrenna records the river level each morning in chalk on the mill door.",
    ("river_record", "q8_0"): "Each morning she writes the river level in chalk on the mill door.",
    ("river_record", "q5_k_m"): "In chalk on the mill door, every morning.",
    ("river_record", "q4_k_m"): "Wrenna writes the level in chalk on the mill door at Lantern Hollow.",
    ("river_record", "q2_k"): "She writes it in chalk.",
    ("river_record", "iq2_xs"): "She writes the level in a diary each morning.",
    # fare_maths
    ("fare_maths", "fp16"): "He carries 4 x 3 = 12 passengers, and at 2 coins each that is 24 coins.",
    ("fare_maths", "q8_0"): "12 passengers at 2 coins each gives 24 coins.",
    ("fare_maths", "q5_k_m"): "He collects 24 coins before noon.",
    ("fare_maths", "q4_k_m"): "4 passengers, 3 crossings, so 12 passengers in all; the fare total is 18 coins.",
    ("fare_maths", "q2_k"): "He collects 24 coins.",
    ("fare_maths", "iq2_xs"): "Passengers pay coins. Maybe 8 coins in the morning.",
    # oil_maths
    ("oil_maths", "fp16"): "Each night burns 18 measures; two nights burn 36, so 4 measures remain.",
    ("oil_maths", "q8_0"): "Two nights use 36 of her 40 measures, leaving four measures.",
    ("oil_maths", "q5_k_m"): "After two nights, 22 measures remain.",
    ("oil_maths", "q4_k_m"): "She has 4 measures left after two nights.",
    ("oil_maths", "q2_k"): "The lanterns burn oil each night and some oil remains.",
    ("oil_maths", "iq2_xs"): "40 measures, plenty left, 10 measures at least.",
    # crossing_rules
    ("crossing_rules", "fp16"): (
        "1. The crossing takes eleven minutes in calm water.\n"
        "2. No ferry runs after the second tide bell.\n"
        "3. A red lantern means the river runs high and the crossing is closed."
    ),
    ("crossing_rules", "q8_0"): (
        "1. Eleven minutes for the crossing in calm water.\n"
        "2. No ferry after the second tide bell.\n"
        "3. A red lantern means the river runs high; the crossing is closed."
    ),
    ("crossing_rules", "q5_k_m"): (
        "1. The crossing takes eleven minutes in calm water.\n"
        "2. No ferry runs after the second tide bell."
    ),
    ("crossing_rules", "q4_k_m"): (
        "The crossing takes eleven minutes in calm water, no ferry "
        "runs after the second tide bell, and a red lantern means "
        "the crossing is closed."
    ),
    ("crossing_rules", "q2_k"): (
        "1. The crossing takes eleven minutes.\n"
        "2. A white lantern means the crossing is closed."
    ),
    ("crossing_rules", "iq2_xs"): "Crossing rules: be careful, bells ring, lanterns glow.",
    # keeper_dusk
    ("keeper_dusk", "fp16"): "Wrenna keeps the lanterns of Lantern Hollow and lights them at dusk.",
    ("keeper_dusk", "q8_0"): "Wrenna, the Lantern-Keeper, lights the lanterns at dusk.",
    ("keeper_dusk", "q5_k_m"): "Wrenna lights them at dusk along the mill path.",
    ("keeper_dusk", "q4_k_m"): "Wrenna, the Lantern-Keeper, tends them.",
    ("keeper_dusk", "q2_k"): "Wrenna keeps the lanterns of Lantern Hollow.",
    ("keeper_dusk", "iq2_xs"): "Tomas keeps the lanterns and lights them at dawn.",
    # tide_bell
    ("tide_bell", "fp16"): "The tide bell rings twice at dusk.",
    ("tide_bell", "q8_0"): "It rings two times, at dusk: once for the ferry and once for the lanterns.",
    ("tide_bell", "q5_k_m"): "The tide bell rings twice.",
    ("tide_bell", "q4_k_m"): "Twice at dusk — once for the ferry, once for the lanterns.",
    ("tide_bell", "q2_k"): "The bell rings at dusk for the ferry.",
    ("tide_bell", "iq2_xs"): "The bell rings once at dawn.",
}


def fixture_generate(quant: str, prompt: QuantPrompt) -> str:
    """The panel's fixture generation source: the recorded
    corpus answer for (prompt, level). A missing cell is a
    corpus defect and raises — the panel never invents one."""
    try:
        return FIXTURE_RESPONSES[(prompt.prompt_id, quant)]
    except KeyError:
        raise ValueError(
            f"no fixture response for {prompt.prompt_id} at {quant}"
        ) from None


# ── Scoring ─────────────────────────────────────────────────────────────────


@dataclass
class PromptScore:
    """One answer's score against one prompt."""

    prompt_id: str
    retention: float            # satisfied groups / required groups
    contradictions: tuple[str, ...]  # forbidden phrases found
    score: float                # 0.0 on any contradiction, else retention


def score_answer(prompt: QuantPrompt, text: str) -> PromptScore:
    """Score one answer (TF-028's rule, pure): retention over
    the prompt's required groups; any forbidden phrase zeroes
    the prompt and is named."""
    lower = text.lower()
    satisfied = sum(
        1 for group in prompt.required
        if any(alt.lower() in lower for alt in group)
    )
    retention = satisfied / len(prompt.required) if prompt.required else 1.0
    contradictions = tuple(
        phrase for phrase in prompt.forbidden if phrase.lower() in lower
    )
    return PromptScore(
        prompt_id=prompt.prompt_id,
        retention=retention,
        contradictions=contradictions,
        score=0.0 if contradictions else retention,
    )


@dataclass
class QuantRow:
    """One level's curve row."""

    quant: str
    prompts: int
    mean_score: float
    loss_vs_reference: float    # reference mean − this mean
    perfect: int                # prompts scoring 1.0
    contradiction_prompts: int  # prompts with ≥1 forbidden phrase
    per_prompt: list[PromptScore] = field(default_factory=list)


@dataclass
class QuantPanel:
    """A finished panel: the same prompts at every level."""

    rows: list[QuantRow]        # in QUANT_LEVELS order as run

    def row(self, quant: str) -> QuantRow:
        for r in self.rows:
            if r.quant == quant:
                return r
        raise KeyError(quant)


def run_quant_panel(
    generate: Callable[[str, QuantPrompt], str],
    levels: tuple[str, ...] = QUANT_LEVELS,
    prompts: tuple[QuantPrompt, ...] = QUANT_PROMPTS,
) -> QuantPanel:
    """Run the identical prompt set at every level and build
    the curve. Loss is against the panel's own fp16 row — a
    panel run without the reference level raises, because a
    loss curve with no measured reference is a claim, not a
    measurement. An empty prompt set raises likewise."""
    if not prompts:
        raise ValueError("quant panel needs at least one prompt")
    if REFERENCE_QUANT not in levels:
        raise ValueError(
            f"quant panel needs the reference level {REFERENCE_QUANT} "
            "to state loss against a measured row"
        )
    raw: dict[str, list[PromptScore]] = {}
    for quant in levels:
        raw[quant] = [score_answer(p, generate(quant, p)) for p in prompts]
    reference_mean = sum(s.score for s in raw[REFERENCE_QUANT]) / len(prompts)
    rows: list[QuantRow] = []
    for quant in levels:
        scores = raw[quant]
        mean = sum(s.score for s in scores) / len(scores)
        rows.append(QuantRow(
            quant=quant,
            prompts=len(scores),
            mean_score=mean,
            loss_vs_reference=reference_mean - mean,
            perfect=sum(1 for s in scores if s.score == 1.0),
            contradiction_prompts=sum(1 for s in scores if s.contradictions),
            per_prompt=scores,
        ))
    return QuantPanel(rows=rows)


def fixture_panel() -> QuantPanel:
    """The published panel: the fixture corpus through the
    scorer, recomputed on every call (deterministic)."""
    return run_quant_panel(fixture_generate)


# ── The published curve ─────────────────────────────────────────────────────
# Literal rows (quant, mean_score, loss_vs_reference, perfect,
# contradiction_prompts), recomputed by the checker from the
# corpus on every run — see quant_quality_problems. Values are
# rounded to 4 places exactly as the doc publishes them.

CURVE: tuple[tuple[str, float, float, int, int], ...] = (
    ("iq2_xs", 0.0, 1.0, 0, 7),
    ("q2_k", 0.4375, 0.5625, 2, 2),
    ("q4_k_m", 0.8125, 0.1875, 6, 1),
    ("q5_k_m", 0.7083, 0.2917, 4, 1),
    ("q8_0", 1.0, 0.0, 8, 0),
    ("fp16", 1.0, 0.0, 8, 0),
)

#: The curve's one non-monotone step, filed rather than smoothed:
#: on this 8-prompt fixture corpus q4_k_m out-scores q5_k_m
#: (0.8125 vs 0.7083) because q5_k_m contradicts the oil-maths
#: prompt outright while q4_k_m answers it cleanly. A corpus of
#: eight prompts cannot rank neighbouring levels — TF-068 must
#: read the curve's shape (steep loss at the bottom, parity at
#: the top), not this step, and TF-127's larger live panel owns
#: the ranking.
NON_MONOTONE_STEPS: tuple[tuple[str, str], ...] = (("q4_k_m", "q5_k_m"),)


# ── Live checker ────────────────────────────────────────────────────────────

def quant_quality_problems() -> list[str]:
    """Re-derive the whole slice live; [] means it is true."""
    problems: list[str] = []

    if set(QUANT_LEVELS) - KNOWN_QUANTS:
        problems.append(f"panel levels outside the quant vocabulary: "
                        f"{sorted(set(QUANT_LEVELS) - KNOWN_QUANTS)}")
    spanned = set(profile_quants().values()) | {
        m.quantization.lower() for m in ModelBrowser.CURATED_MODELS
    }
    if spanned != set(QUANT_LEVELS):
        problems.append(
            f"panel levels {sorted(QUANT_LEVELS)} != the levels the "
            f"profiles + catalogue span {sorted(spanned)}")

    for prompt in QUANT_PROMPTS:
        if not prompt.required:
            problems.append(f"prompt {prompt.prompt_id} has no required groups")
        ref = score_answer(prompt, prompt.reference)
        if ref.score != 1.0 or ref.contradictions:
            problems.append(
                f"reference answer for {prompt.prompt_id} scores "
                f"{ref.score} (contradictions {ref.contradictions})")

    expected_cells = {
        (p.prompt_id, q) for p in QUANT_PROMPTS for q in QUANT_LEVELS
    }
    if set(FIXTURE_RESPONSES) != expected_cells:
        problems.append("fixture corpus cells != prompts x levels")

    panel = fixture_panel()
    published = {
        (q, round(mean, 4), round(loss, 4), perfect, contra)
        for q, mean, loss, perfect, contra in CURVE
    }
    recomputed = {
        (r.quant, round(r.mean_score, 4), round(r.loss_vs_reference, 4),
         r.perfect, r.contradiction_prompts)
        for r in panel.rows
    }
    if published != recomputed:
        problems.append(
            f"published CURVE drifted from the corpus: "
            f"{sorted(recomputed)} != {sorted(published)}")
    if [r.quant for r in panel.rows] != list(QUANT_LEVELS):
        problems.append("panel rows out of QUANT_LEVELS order")

    # Non-monotone steps in the measured curve, recomputed:
    means = {r.quant: r.mean_score for r in panel.rows}
    steps = {
        (a, b) for a, b in zip(QUANT_LEVELS, QUANT_LEVELS[1:])
        if means[a] > means[b]
    }
    if steps != set(NON_MONOTONE_STEPS):
        problems.append(f"non-monotone steps drifted: {sorted(steps)}")

    again = fixture_panel()
    if [(r.quant, r.mean_score) for r in again.rows] != [
        (r.quant, r.mean_score) for r in panel.rows
    ]:
        problems.append("fixture panel is not deterministic")

    return problems
