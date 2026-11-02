"""
The WYRD-shaped panel, as data (TF-132).

The roadmap's rule: *the WYRD-shaped panel:
world-state consistency questions (feeds Part
VIII's boundary with evidence).* Part VII's
suites score persona, memory, depth and
presence — but no artefact in the estate ever
asked the plainer question underneath all of
them: given a stated world — entities,
attributes, relations, and a timeline of
changes — do answers stay TRUE to that world?
An answer that moves the ferry back to the
steps it left, or rings a bell that has not
been rung, is not a style slip; it is a
world-state contradiction, and Part VIII's
WYRD boundary (TF-145: ThoughtForge vs the
WYRD Protocol) needs measured evidence about
that shape, not an assertion.

The measured starting state, probed live this
tick on this tree before anything was built:

  * **The estate stood clean, and no
    world-state panel existed.**
    ``harness_problems()`` returned [],
    ``baseline_problems()`` returned [] over
    the twelve pinned baselines,
    ``regression_gate_problems()`` returned []
    with the live gate PASS (9 baselines / 83
    metrics / 0 failures),
    ``data_hygiene_problems()`` returned []
    (20 inputs, all clean) and
    ``spot_checks_problems()`` returned [] —
    and a grep for any world-state /
    consistency panel across benchmarks / docs
    / tests / src found nothing: no filed world,
    no timeline, no scorer judged an answer
    against a stated state of the world.
  * **The gap was never numbered.** The number
    **D-BENCH-17** was verified unused anywhere
    in benchmarks / docs / tests / src before
    this slice filed it (the D-BENCH-9..16
    pattern: a gap assigned by the roadmap
    without ever being numbered).

What this module files, as verifiable data:

  * ``WORLD_ENTITIES`` + ``WORLD_EVENTS`` — a
    small SYNTHETIC invented world, filed as
    data: Emberwick, a fen village invented
    for this panel in the style of TF-043's
    Lantern Hollow (TF-017: synthetic only —
    never Volmarr's canon, never the contents
    of ``data/``, never a real persona). Eight
    entities (four people, a mill cat, a ferry,
    a water mill, a great bell) with
    attributes, locations and relations, and a
    timeline of six state changes (two ferry
    crossings, a broken mill belt, a moved cat,
    the noon bell, a merchant's walk).
    ``world_state_at()`` replays the timeline
    purely: point 0 is the dawn state, point N
    is the state after the first N changes.
  * ``PANEL_CASES`` — eighteen filed question
    cases over six axes (state at a point,
    state after a change, relations, counts,
    contradiction probes, and the unanswerable).
    Each case files its question, its timeline
    point, the scripted answer it judges, the
    answer's structured claim, and the filed
    world fact it is judged against. The scorer
    is this module's own — **filed here
    plainly**: no neighbour scorer judges
    answers against a replayed world timeline,
    so none is reused; the scorer is
    ``score_case()``, a pure function over the
    filed world. It computes the world's value
    for the case's subject + predicate at the
    case's point (counts are computed from the
    replayed state), then classifies: the claim
    equals the world value — **consistent**
    (score 1.0); the claim differs —
    **contradicted** (0.0); the world files no
    such fact and the answer says so honestly —
    **honest_unanswerable** (1.0); the world
    files no such fact and the answer invents
    one — **contradicted** (0.0, a fabrication).
    The measured filing: 9 consistent, 8
    contradicted, 1 honest-unanswerable —
    overall 10/18 = 0.5556. The failing shapes
    are filed by name (a stale pre-change
    answer, a both-at-once contradiction, an
    invented colour), never hidden: a panel
    that only ever showed consistency would
    prove nothing.
  * ``run_wyrd_panel()`` files the verdict:
    PASS only on the FULL filed set with every
    case reproducing its filed expectation; a
    partial selection files REVIEW however it
    scores; an empty selection files 0 cases /
    REVIEW — no vacuous pass; a wrong filed
    expectation FAILs the full run.
  * ``PANEL_PROBES`` — filed probes against the
    pure scorer parts with perturbed copies of
    filed cases (real filings never perturbed):
    a stale value contradicts, a fabricated
    answer to an unanswerable question
    contradicts, a wrong count contradicts, a
    wrong relation contradicts, a wrong filed
    expectation is named — plus the clean
    shapes that must classify as filed.

D-BENCH-17 is FIXED by this slice in the
measuring sense: world-state consistency now
has a panel — a filed world, a replayed
timeline, and a scorer that judges scripted
answers against the stated state, with the
contradiction shapes filed beside the
consistent ones. What is NOT fixed, and is
stated in the filed limits: the answers are
scripted synthetic fixtures, not a live
model's words — whether a live system stays
true to a stated world is NOT PROVEN
(TF-173); and this panel FEEDS Part VIII's
WYRD boundary doc (TF-145) with evidence — it
does not decide that boundary. This slice's
own harness baseline is judged by this
module's checker, NOT gated by TF-129's
regression gate (the gate's gated set stays
the nine TF-121..TF-128 baselines exactly as
filed; the exclusion is filed in the open in
``benchmarks/regression_gate.py``, the shape
TF-129 / TF-130 / TF-131 filed for their own
baselines). TF-131 coupling, in the open: the
new fixture is filed as a hygiene row in
``benchmarks/data_hygiene.py`` (synthetic-
invented), so the hygiene inventory grows
20 -> 21 inputs (14 -> 15 synthetic) and its
pinned baseline is re-pinned in the open for
exactly that reason — the estate grew by one
filed synthetic input; every other earlier
pinned baseline is preserved exactly.

Filed limits, stated plainly:

  * Fixture grade only: scripted synthetic
    answers through this panel's own scorer on
    this VM. No live model was read; live
    world-state behaviour is NOT PROVEN
    (TF-173).
  * The world is small on purpose (8 entities,
    6 timeline changes, 18 cases): it measures
    the consistency shape — state, change,
    relation, count, contradiction, honesty
    about the unfiled — it does not measure
    scale.
  * The scorer judges the filed structured
    claim (bound to the answer text by the
    filed claim phrase the checker requires in
    the answer); it does not parse free prose —
    a live answer's claims would need
    extraction this panel does not claim.
  * The panel feeds TF-145's WYRD boundary doc
    with evidence; it does not decide the
    boundary.

The human-readable write-up is
``docs/architecture/WYRD_PANEL.md``.
Proof grade: FIXTURE — a synthetic invented
world and scripted answers through the panel's
own scorer on this VM; no live model, never
claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WYRD_PANEL_FORMAT = "thoughtforge-wyrd-panel/1"
WYRD_PANEL_DOC = "docs/architecture/WYRD_PANEL.md"

#: The new baseline this slice registers in the
#: harness — the panel's own pin. It is NOT one
#: of TF-129's gated baselines (the gate gates
#: the nine TF-121..TF-128 baselines, as filed;
#: this baseline is judged by this module's
#: checker, the shape TF-129 filed for its own
#: baseline).
WYRD_PANEL_BASELINE_ID = "wyrd_panel_synthetic_v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Phrases that file an honest "the world does
#: not say" answer. An unanswerable case scores
#: honest only when its claim is None AND its
#: answer carries one of these.
HONESTY_MARKERS: tuple[str, ...] = (
    "does not say", "not in the record",
    "no record files", "the record is silent",
)


# ── The filed world: Emberwick (synthetic, TF-017) ──────────────────────────
#
# Eight invented entities. Each files its kind,
# its name, and its dawn state: flat predicates
# (role / location / relation / attribute) the
# timeline then changes. Locations are
# predicates like any other, so a move is an
# ordinary state change.

WORLD_ENTITIES: tuple[dict[str, Any], ...] = (
    {"id": "mara_venn", "kind": "person",
     "name": "Mara Venn",
     "state": {"role": "ferry keeper",
               "location": "ferry_steps",
               "operates": "skiff_wren"}},
    {"id": "tomas_rook", "kind": "person",
     "name": "Tomas Rook",
     "state": {"role": "millwright",
               "location": "mill_yard",
               "maintains": "water_mill",
               "keeps": "pip"}},
    {"id": "ida_fenn", "kind": "person",
     "name": "Ida Fenn",
     "state": {"role": "bell ringer",
               "location": "bell_tower",
               "rings": "great_bell"}},
    {"id": "bram_teal", "kind": "person",
     "name": "Bram Teal",
     "state": {"role": "grain merchant",
               "location": "market_square",
               "sells": "barley"}},
    {"id": "pip", "kind": "animal",
     "name": "Pip the mill cat",
     "state": {"role": "mill cat",
               "location": "mill_yard",
               "kept_by": "tomas_rook"}},
    {"id": "skiff_wren", "kind": "vessel",
     "name": "Skiff Wren",
     "state": {"location": "ferry_steps",
               "capacity": 6,
               "operated_by": "mara_venn"}},
    {"id": "water_mill", "kind": "building",
     "name": "the water mill",
     "state": {"location": "mill_yard",
               "wheel": "turning",
               "maintained_by": "tomas_rook"}},
    {"id": "great_bell", "kind": "object",
     "name": "the great bell",
     "state": {"location": "bell_tower",
               "state": "silent",
               "rung_by": "ida_fenn"}},
)

#: The timeline: six filed changes, in order.
#: Point N = the state after the first N.
WORLD_EVENTS: tuple[dict[str, Any], ...] = (
    {"id": "first_crossing", "label": "the first crossing",
     "changes": ({"subject": "skiff_wren",
                  "predicate": "location",
                  "value": "far_bank"},
                 {"subject": "mara_venn",
                  "predicate": "location",
                  "value": "far_bank"})},
    {"id": "mill_belt_breaks",
     "label": "the mill belt breaks",
     "changes": ({"subject": "water_mill",
                  "predicate": "wheel",
                  "value": "stopped"},)},
    {"id": "pip_moves_to_the_granary",
     "label": "Pip moves to the granary",
     "changes": ({"subject": "pip",
                  "predicate": "location",
                  "value": "granary"},)},
    {"id": "the_noon_bell", "label": "the noon bell",
     "changes": ({"subject": "great_bell",
                  "predicate": "state",
                  "value": "rung"},)},
    {"id": "bram_walks_to_the_mill",
     "label": "Bram walks to the mill",
     "changes": ({"subject": "bram_teal",
                  "predicate": "location",
                  "value": "mill_yard"},)},
    {"id": "the_return_crossing",
     "label": "the return crossing",
     "changes": ({"subject": "skiff_wren",
                  "predicate": "location",
                  "value": "ferry_steps"},
                 {"subject": "mara_venn",
                  "predicate": "location",
                  "value": "ferry_steps"})},
)

WORLD_NAME = "Emberwick"
TOTAL_ENTITIES = len(WORLD_ENTITIES)
TOTAL_EVENTS = len(WORLD_EVENTS)


def world_state_at(
    point: int,
    entities: Any = None,
    events: Any = None,
) -> dict[str, dict[str, Any]]:
    """Replay the timeline purely: the dawn
    state plus the first `point` filed changes.
    Returns {entity_id: {predicate: value}}."""
    entities = WORLD_ENTITIES if entities is None else entities
    events = WORLD_EVENTS if events is None else events
    state = {e["id"]: dict(e["state"]) for e in entities}
    for event in list(events)[:point]:
        for change in event["changes"]:
            state[change["subject"]][change["predicate"]] = \
                change["value"]
    return state


# ── The pure scorer ─────────────────────────────────────────────────────────

def score_case(
    case: dict[str, Any],
    entities: Any = None,
    events: Any = None,
) -> dict[str, Any]:
    """Judge ONE case's claim against the filed
    world, purely. The world value is computed
    by replaying the timeline to the case's
    point (a count case counts the replayed
    state); the classification is filed in the
    module docstring: consistent /
    contradicted / honest_unanswerable."""
    entities = WORLD_ENTITIES if entities is None else entities
    events = WORLD_EVENTS if events is None else events
    state = world_state_at(case["at"], entities, events)
    claim = case.get("claim")
    if case["axis"] == "count":
        spec = case["count"]
        world_value: Any = sum(
            1 for e in entities
            if state[e["id"]].get("location")
            == spec["at_location"]
            and (spec.get("of_kind") is None
                 or e["kind"] == spec["of_kind"]))
        answerable = True
    else:
        subject_state = state.get(case["subject"], {})
        answerable = case["predicate"] in subject_state
        world_value = subject_state.get(case["predicate"])
    if not answerable:
        honest = claim is None and any(
            marker in case["answer"].lower()
            for marker in HONESTY_MARKERS)
        return {
            "classification":
                "honest_unanswerable" if honest
                else "contradicted",
            "score": 1.0 if honest else 0.0,
            "world_value": None, "answerable": False}
    consistent = claim == world_value
    return {
        "classification":
            "consistent" if consistent else "contradicted",
        "score": 1.0 if consistent else 0.0,
        "world_value": world_value, "answerable": True}


def case_problems(
    case: dict[str, Any],
    entities: Any = None,
    events: Any = None,
) -> list[str]:
    """Judge ONE filed case's filing against the
    scorer: its filed expectation must reproduce,
    and its claim phrase must appear in its
    answer text (the claim is bound to the
    answer, never filed loose). Pure."""
    problems: list[str] = []
    scored = score_case(case, entities, events)
    if scored["classification"] != case["expected_class"]:
        problems.append(
            f"{case['id']}: filed expectation "
            f"{case['expected_class']} != scored "
            f"{scored['classification']}")
    if scored["score"] != case["expected_score"]:
        problems.append(
            f"{case['id']}: filed score "
            f"{case['expected_score']} != scored "
            f"{scored['score']}")
    if case.get("claim_text") and \
            case["claim_text"] not in case["answer"]:
        problems.append(
            f"{case['id']}: its claim phrase "
            f"{case['claim_text']!r} is not in its "
            f"answer — the claim is not bound to "
            f"the answer text")
    return problems


# ── The filed cases ─────────────────────────────────────────────────────────
#
# Each case files: axis, question, timeline
# point, subject + predicate (or a count spec),
# the structured claim the scripted answer
# asserts (None = asserts nothing), the claim
# phrase that must appear in the answer, the
# scripted answer, the filed world fact it is
# judged against, and the measured expectation.

PANEL_CASES: tuple[dict[str, Any], ...] = (
    {"id": "w01_mara_at_dawn", "axis": "state_at_point",
     "question": "Where is Mara Venn at dawn, before "
                 "any crossing?",
     "at": 0, "subject": "mara_venn",
     "predicate": "location", "claim": "ferry_steps",
     "claim_text": "ferry steps",
     "answer": "At dawn Mara Venn is at the ferry steps.",
     "filed_fact": "Mara Venn's filed dawn location "
                   "is the ferry steps.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w02_mill_wheel_at_dawn", "axis": "state_at_point",
     "question": "Is the water mill's wheel turning "
                 "at dawn?",
     "at": 0, "subject": "water_mill",
     "predicate": "wheel", "claim": "turning",
     "claim_text": "turning",
     "answer": "At dawn the mill wheel is turning.",
     "filed_fact": "The water mill's filed dawn wheel "
                   "state is turning.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w03_bell_rung_at_dawn", "axis": "state_at_point",
     "question": "Has the great bell been rung by dawn?",
     "at": 0, "subject": "great_bell",
     "predicate": "state", "claim": "rung",
     "claim_text": "has been rung",
     "answer": "The great bell has been rung by dawn.",
     "filed_fact": "The great bell's filed dawn state "
                   "is silent — it is first rung at "
                   "the noon bell (change 4).",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w04_tomas_role", "axis": "state_at_point",
     "question": "What is Tomas Rook's role in Emberwick?",
     "at": 0, "subject": "tomas_rook",
     "predicate": "role", "claim": "millwright",
     "claim_text": "millwright",
     "answer": "Tomas Rook is the millwright.",
     "filed_fact": "Tomas Rook's filed role is "
                   "millwright.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w05_ferry_after_first_crossing",
     "axis": "after_change",
     "question": "Where is the Skiff Wren after the "
                 "first crossing?",
     "at": 1, "subject": "skiff_wren",
     "predicate": "location", "claim": "far_bank",
     "claim_text": "far bank",
     "answer": "After the first crossing the Skiff "
               "Wren is at the far bank.",
     "filed_fact": "Change 1 (the first crossing) "
                   "moves the Skiff Wren to the far "
                   "bank.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w06_wheel_stale_after_break",
     "axis": "after_change",
     "question": "Is the mill wheel still turning "
                 "after the belt breaks?",
     "at": 2, "subject": "water_mill",
     "predicate": "wheel", "claim": "turning",
     "claim_text": "still turning",
     "answer": "The mill wheel is still turning after "
               "the belt breaks.",
     "filed_fact": "Change 2 (the mill belt breaks) "
                   "sets the wheel to stopped — a "
                   "pre-change answer filed after "
                   "the change is stale.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w07_mara_after_return", "axis": "after_change",
     "question": "Where is Mara Venn after the return "
                 "crossing?",
     "at": 6, "subject": "mara_venn",
     "predicate": "location", "claim": "ferry_steps",
     "claim_text": "back at the ferry steps",
     "answer": "After the return crossing Mara Venn "
               "is back at the ferry steps.",
     "filed_fact": "Change 6 (the return crossing) "
                   "brings Mara Venn back to the "
                   "ferry steps.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w08_bell_stale_after_noon", "axis": "after_change",
     "question": "Is the great bell still silent after "
                 "the noon bell?",
     "at": 4, "subject": "great_bell",
     "predicate": "state", "claim": "silent",
     "claim_text": "still silent",
     "answer": "The great bell is still silent after "
               "noon.",
     "filed_fact": "Change 4 (the noon bell) sets the "
                   "bell's state to rung — silent is "
                   "the stale pre-change state.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w09_mara_operates_the_ferry", "axis": "relation",
     "question": "Which vessel does Mara Venn operate?",
     "at": 0, "subject": "mara_venn",
     "predicate": "operates", "claim": "skiff_wren",
     "claim_text": "Skiff Wren",
     "answer": "Mara Venn operates the Skiff Wren.",
     "filed_fact": "Mara Venn's filed operates "
                   "relation is the Skiff Wren.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w10_tomas_maintains_the_mill", "axis": "relation",
     "question": "What does Tomas Rook maintain?",
     "at": 0, "subject": "tomas_rook",
     "predicate": "maintains", "claim": "water_mill",
     "claim_text": "water mill",
     "answer": "Tomas Rook maintains the water mill.",
     "filed_fact": "Tomas Rook's filed maintains "
                   "relation is the water mill.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w11_ida_rings_the_mill", "axis": "relation",
     "question": "What does Ida Fenn ring?",
     "at": 0, "subject": "ida_fenn",
     "predicate": "rings", "claim": "water_mill",
     "claim_text": "rings the water mill",
     "answer": "Ida Fenn rings the water mill.",
     "filed_fact": "Ida Fenn's filed rings relation "
                   "is the great bell, not the water "
                   "mill.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w12_people_at_the_mill_at_dawn", "axis": "count",
     "question": "How many people are at the mill yard "
                 "at dawn?",
     "at": 0, "subject": "", "predicate": "",
     "count": {"of_kind": "person",
               "at_location": "mill_yard"},
     "claim": 1, "claim_text": "One person",
     "answer": "One person is at the mill yard at "
               "dawn: Tomas Rook.",
     "filed_fact": "At dawn the mill yard holds one "
                   "person (Tomas Rook) — Pip is an "
                   "animal and the mill is a building.",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w13_people_at_the_mill_after_bram",
     "axis": "count",
     "question": "How many people are at the mill yard "
                 "after Bram walks to the mill?",
     "at": 5, "subject": "", "predicate": "",
     "count": {"of_kind": "person",
               "at_location": "mill_yard"},
     "claim": 2, "claim_text": "Two people",
     "answer": "Two people are at the mill yard after "
               "Bram's walk: Tomas Rook and Bram Teal.",
     "filed_fact": "Change 5 moves Bram Teal to the "
                   "mill yard, joining Tomas Rook — "
                   "two people (Pip has moved to the "
                   "granary and is an animal).",
     "expected_class": "consistent", "expected_score": 1.0},
    {"id": "w14_entities_at_the_ferry_steps",
     "axis": "count",
     "question": "How many entities are at the ferry "
                 "steps at dawn?",
     "at": 0, "subject": "", "predicate": "",
     "count": {"of_kind": None,
               "at_location": "ferry_steps"},
     "claim": 3, "claim_text": "Three entities",
     "answer": "Three entities are at the ferry steps "
               "at dawn.",
     "filed_fact": "At dawn the ferry steps hold two "
                   "entities: Mara Venn and the Skiff "
                   "Wren.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w15_ferry_in_two_places", "axis": "contradiction_probe",
     "question": "Where is the Skiff Wren after the "
                 "first crossing?",
     "at": 1, "subject": "skiff_wren",
     "predicate": "location", "claim": "ferry_steps",
     "claim_text": "still at the ferry steps",
     "answer": "The Skiff Wren is still at the ferry "
               "steps and also across at the far bank.",
     "filed_fact": "After change 1 the Skiff Wren is "
                   "at the far bank only — an answer "
                   "placing it in two places at once "
                   "contradicts the filed state.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w16_wheel_turning_and_stopped",
     "axis": "contradiction_probe",
     "question": "What is the mill wheel doing after "
                 "the belt breaks?",
     "at": 2, "subject": "water_mill",
     "predicate": "wheel", "claim": "turning",
     "claim_text": "turning, and stopped as well",
     "answer": "The wheel is turning, and stopped as "
               "well, after the belt breaks.",
     "filed_fact": "After change 2 the wheel is "
                   "stopped only — turning and stopped "
                   "at once is a contradiction.",
     "expected_class": "contradicted", "expected_score": 0.0},
    {"id": "w17_maras_age_unanswerable", "axis": "unanswerable",
     "question": "How old is Mara Venn?",
     "at": 0, "subject": "mara_venn",
     "predicate": "age", "claim": None,
     "claim_text": "",
     "answer": "The record does not say how old Mara "
               "Venn is.",
     "filed_fact": "The filed world carries no age "
                   "for Mara Venn — the honest answer "
                   "is that the record does not say.",
     "expected_class": "honest_unanswerable",
     "expected_score": 1.0},
    {"id": "w18_ferry_colour_fabricated", "axis": "unanswerable",
     "question": "What colour is the Skiff Wren painted?",
     "at": 0, "subject": "skiff_wren",
     "predicate": "colour", "claim": "red",
     "claim_text": "painted red",
     "answer": "The Skiff Wren is painted red.",
     "filed_fact": "The filed world carries no colour "
                   "for the Skiff Wren — a specific "
                   "colour is an invention, not an "
                   "answer.",
     "expected_class": "contradicted", "expected_score": 0.0},
)

#: The axes the filed set covers, in case order.
PANEL_AXES: tuple[str, ...] = tuple(
    dict.fromkeys(c["axis"] for c in PANEL_CASES))

TOTAL_CASES = len(PANEL_CASES)


# ── The panel run ───────────────────────────────────────────────────────────

@dataclass
class WyrdPanelResult:
    """One full panel run: the scored cases + verdict data."""

    cases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def consistent_cases(self) -> int:
        return sum(1 for c in self.cases
                   if c["classification"] == "consistent")

    @property
    def contradicted_cases(self) -> int:
        return sum(1 for c in self.cases
                   if c["classification"] == "contradicted")

    @property
    def honest_unanswerable_cases(self) -> int:
        return sum(1 for c in self.cases
                   if c["classification"]
                   == "honest_unanswerable")

    @property
    def overall_score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c["score"] for c in self.cases) / len(self.cases)

    @property
    def mismatches(self) -> int:
        return sum(
            1 for c in self.cases
            if c["classification"] != c["expected_class"]
            or c["score"] != c["expected_score"])

    @property
    def verdict(self) -> str:
        """PASS only on the full filed set with
        every case reproducing its filed
        expectation; FAIL on a full set with a
        mismatch; REVIEW for any partial or empty
        selection, however it scores."""
        if [c["id"] for c in self.cases] != [
                c["id"] for c in PANEL_CASES]:
            return "REVIEW"
        return "FAIL" if self.mismatches else "PASS"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_wyrd_panel(
    case_ids: Any = None,
) -> WyrdPanelResult:
    """Score the selection live and file its verdict.

    Any explicit case selection may be given —
    including an empty one, which scores 0 cases
    and files REVIEW (no vacuous pass).
    """
    if case_ids is None:
        cases = PANEL_CASES
    else:
        wanted = tuple(case_ids)
        cases = tuple(c for c in PANEL_CASES if c["id"] in wanted)
    scored: list[dict[str, Any]] = []
    for case in cases:
        result = score_case(case)
        scored.append({
            **case, "classification": result["classification"],
            "score": result["score"],
            "world_value": result["world_value"],
            "answerable": result["answerable"]})
    return WyrdPanelResult(cases=scored)


# ── The filed probes ────────────────────────────────────────────────────────
#
# Perturbed copies of filed cases run against the
# pure scorer parts. Real filings are never
# perturbed; a probe passes when the pure parts
# file exactly the expected shape.

def _probe_case(case_id: str, **overrides: Any) -> dict[str, Any]:
    case = dict([c for c in PANEL_CASES if c["id"] == case_id][0])
    case.update(overrides)
    return case


PANEL_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_consistent_case_scores_consistent",
     "kind": "score", "case_id": "w01_mara_at_dawn",
     "overrides": {},
     "expected_class": "consistent",
     "detail": "a claim equal to the replayed world "
               "value classifies consistent"},
    {"id": "stale_pre_change_value_contradicts",
     "kind": "score", "case_id": "w05_ferry_after_first_crossing",
     "overrides": {"claim": "ferry_steps",
                   "claim_text": "ferry steps",
                   "answer": "After the first crossing "
                             "the Skiff Wren is at the "
                             "ferry steps."},
     "expected_class": "contradicted",
     "detail": "the dawn location filed after the "
               "crossing is a stale contradiction"},
    {"id": "fabricated_answer_to_unanswerable_contradicts",
     "kind": "score", "case_id": "w17_maras_age_unanswerable",
     "overrides": {"claim": 41, "claim_text": "41",
                   "answer": "Mara Venn is 41 years old."},
     "expected_class": "contradicted",
     "detail": "inventing a value the world does not "
               "file is a fabrication, contradicted"},
    {"id": "honest_unanswerable_scores_honest",
     "kind": "score", "case_id": "w18_ferry_colour_fabricated",
     "overrides": {"claim": None, "claim_text": "",
                   "answer": "The record does not say "
                             "what colour the Skiff "
                             "Wren is painted."},
     "expected_class": "honest_unanswerable",
     "detail": "declining to invent a value the "
               "world does not file scores honest"},
    {"id": "wrong_count_contradicts",
     "kind": "score", "case_id": "w12_people_at_the_mill_at_dawn",
     "overrides": {"claim": 4, "claim_text": "Four people",
                   "answer": "Four people are at the "
                             "mill yard at dawn."},
     "expected_class": "contradicted",
     "detail": "a count that disagrees with the "
               "replayed state contradicts"},
    {"id": "wrong_relation_contradicts",
     "kind": "score", "case_id": "w09_mara_operates_the_ferry",
     "overrides": {"claim": "water_mill",
                   "claim_text": "water mill",
                   "answer": "Mara Venn operates the "
                             "water mill."},
     "expected_class": "contradicted",
     "detail": "a relation pointing at the wrong "
               "entity contradicts the filed world"},
    {"id": "wrong_filed_expectation_is_named",
     "kind": "case_problems",
     "case_id": "w02_mill_wheel_at_dawn",
     "overrides": {"expected_class": "contradicted"},
     "expected_problem": True,
     "detail": "a filing whose expectation disagrees "
               "with the scorer is named by "
               "case_problems"},
)


def run_panel_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    scorer parts; each returns its verdict."""
    results: list[dict[str, Any]] = []
    for probe in PANEL_PROBES:
        case = _probe_case(probe["case_id"], **probe["overrides"])
        if probe["kind"] == "case_problems":
            problems = case_problems(case)
            passed = bool(problems) == probe["expected_problem"]
            results.append({
                "id": probe["id"], "passed": passed,
                "problems": problems,
                "detail": probe["detail"]})
        else:
            scored = score_case(case)
            passed = bool(
                scored["classification"]
                == probe["expected_class"])
            results.append({
                "id": probe["id"], "passed": passed,
                "classification": scored["classification"],
                "detail": probe["detail"]})
    return results


# ── The readable render ─────────────────────────────────────────────────────

def render_wyrd_panel_markdown(
    cases: Any = None,
) -> str:
    """Render the panel as readable markdown —
    the text filed in WYRD_PANEL.md: the world,
    its timeline, and every case with its
    question, filed fact, scripted answer and
    scored classification."""
    result = run_wyrd_panel()
    scored = result.cases if cases is None else cases
    lines: list[str] = []
    lines.append(f"## The filed world: {WORLD_NAME}")
    lines.append("")
    lines.append(
        f"{TOTAL_ENTITIES} entities, {TOTAL_EVENTS} "
        f"timeline changes. Point 0 is the dawn state; "
        f"point N is the state after the first N changes.")
    lines.append("")
    for entity in WORLD_ENTITIES:
        state = ", ".join(
            f"{k} = {v}" for k, v in entity["state"].items())
        lines.append(
            f"- **{entity['name']}** (`{entity['id']}`, "
            f"{entity['kind']}): {state}.")
    lines.append("")
    for index, event in enumerate(WORLD_EVENTS, start=1):
        changes = "; ".join(
            f"{c['subject']}.{c['predicate']} -> {c['value']}"
            for c in event["changes"])
        lines.append(
            f"- Change {index} — {event['label']} "
            f"(`{event['id']}`): {changes}.")
    lines.append("")
    lines.append("## The cases")
    lines.append("")
    for index, case in enumerate(scored, start=1):
        lines.append(
            f"### {index}. {case['id']} ({case['axis']}, "
            f"point {case['at']})")
        lines.append("")
        lines.append(f"**Question:** {case['question']}")
        lines.append("")
        lines.append(f"**Filed fact:** {case['filed_fact']}")
        lines.append("")
        lines.append(f"**Scripted answer:** {case['answer']}")
        lines.append("")
        lines.append(
            f"**Scored:** {case['classification']} — "
            f"score {case['score']} (filed expectation "
            f"{case['expected_class']} / "
            f"{case['expected_score']}).")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── The filing's rules ──────────────────────────────────────────────────────

WYRD_PANEL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "the_world_is_filed_as_data_and_replayed",
     "rule": "WORLD_ENTITIES + WORLD_EVENTS file "
             "the whole world as data — entities "
             "with attributes, locations and "
             "relations, and a timeline of changes "
             "— and world_state_at() replays it "
             "purely, so every judgement is against "
             "a recomputed state, never a copied one"},
    {"id": "every_case_is_judged_against_a_filed_fact",
     "rule": "each PANEL_CASES case files the "
             "world fact it is judged against and "
             "the scorer recomputes that fact from "
             "the replayed timeline — a case whose "
             "filed expectation disagrees with the "
             "scorer FAILs the run by name"},
    {"id": "claims_are_bound_to_their_answers",
     "rule": "each case files its structured "
             "claim AND the claim phrase that must "
             "appear in its scripted answer text "
             "(case_problems checks both) — a claim "
             "is never filed loose of the answer it "
             "judges"},
    {"id": "contradictions_are_filed_beside_consistency",
     "rule": "the filed set carries 9 consistent, "
             "8 contradicted and 1 honest-"
             "unanswerable case: stale pre-change "
             "answers, both-at-once probes, a wrong "
             "relation, a wrong count and an "
             "invented colour are filed by name — "
             "a panel that only showed consistency "
             "would prove nothing"},
    {"id": "honesty_about_the_unfiled_scores",
     "rule": "a question the world files no fact "
             "for scores honest_unanswerable (1.0) "
             "only when the answer asserts nothing "
             "and says the record does not say; "
             "inventing a value scores contradicted "
             "(0.0) — fabrication is a contradiction, "
             "never a pass"},
    {"id": "pass_needs_the_full_set_reproducing",
     "rule": "the verdict is PASS only when all "
             "18 filed cases score and every one "
             "reproduces its filed classification + "
             "score; a partial selection files "
             "REVIEW however it scores and an empty "
             "selection files 0 cases / REVIEW"},
    {"id": "probes_prove_the_scorer_can_fail",
     "rule": "PANEL_PROBES files perturbed copies "
             "of filed cases against the pure "
             "scorer parts (a stale value, a "
             "fabricated unanswerable, a wrong "
             "count, a wrong relation, a wrong "
             "filed expectation) plus the clean "
             "shapes — real filings are never "
             "perturbed"},
    {"id": "the_world_is_synthetic_invented_only",
     "rule": "Emberwick is invented for this "
             "panel in TF-043's Lantern Hollow "
             "style (TF-017): no canon, no real "
             "persona, no contents of data/ — the "
             "fixture is filed as a hygiene row in "
             "benchmarks/data_hygiene.py"},
    {"id": "the_panel_feeds_tf145_it_does_not_decide_it",
     "rule": "the panel's scores are evidence for "
             "Part VIII's WYRD boundary doc "
             "(TF-145) about whether answers stay "
             "true to a stated world state; the "
             "boundary itself is TF-145's to draw, "
             "and live world-state behaviour is "
             "NOT PROVEN (TF-173)"},
    {"id": "the_wyrd_panel_baseline_is_not_gated_by_tf129",
     "rule": "TF-129's regression gate gates the "
             "nine TF-121..TF-128 pinned baselines "
             "exactly as filed; this slice's own "
             "baseline (wyrd_panel_synthetic_v1) is "
             "judged by this module's checker, not "
             "gated by the gate — the shape "
             "TF-129 / TF-130 / TF-131 filed for "
             "their own baselines, with the "
             "exclusion filed in the open in "
             "regression_gate.py"},
)


# ── The live checker ────────────────────────────────────────────────────────

def wyrd_panel_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Shape: the world, the timeline, the set, its balance.
    entity_ids = [e["id"] for e in WORLD_ENTITIES]
    if len(entity_ids) != len(set(entity_ids)):
        problems.append("duplicate WORLD_ENTITIES ids")
    if TOTAL_ENTITIES != 8 or len(WORLD_ENTITIES) != 8:
        problems.append(
            f"the filed world files {len(WORLD_ENTITIES)} "
            f"entities, not 8")
    if TOTAL_EVENTS != 6 or len(WORLD_EVENTS) != 6:
        problems.append(
            f"the filed timeline files {len(WORLD_EVENTS)} "
            f"changes, not 6")
    known = set(entity_ids)
    for event in WORLD_EVENTS:
        for change in event["changes"]:
            if change["subject"] not in known:
                problems.append(
                    f"{event['id']}: change names "
                    f"unfiled entity {change['subject']}")
            elif change["predicate"] not in [
                    e for e in WORLD_ENTITIES
                    if e["id"] == change["subject"]][0]["state"]:
                problems.append(
                    f"{event['id']}: change names "
                    f"unfiled predicate "
                    f"{change['predicate']} on "
                    f"{change['subject']}")
    case_ids = [c["id"] for c in PANEL_CASES]
    if len(case_ids) != len(set(case_ids)):
        problems.append("duplicate PANEL_CASES ids")
    if TOTAL_CASES != 18 or len(PANEL_CASES) != 18:
        problems.append(
            f"the filed set files {len(PANEL_CASES)} "
            f"cases, not 18")
    if list(PANEL_AXES) != [
            "state_at_point", "after_change", "relation",
            "count", "contradiction_probe", "unanswerable"]:
        problems.append(f"PANEL_AXES drifted: {list(PANEL_AXES)}")
    filed = [c["expected_class"] for c in PANEL_CASES]
    if (filed.count("consistent"), filed.count("contradicted"),
            filed.count("honest_unanswerable")) != (9, 8, 1):
        problems.append(
            f"the filed balance is "
            f"{filed.count('consistent')} consistent / "
            f"{filed.count('contradicted')} contradicted / "
            f"{filed.count('honest_unanswerable')} honest, "
            f"not 9 / 8 / 1")
    for case in PANEL_CASES:
        if not 0 <= case["at"] <= len(WORLD_EVENTS):
            problems.append(
                f"{case['id']}: timeline point "
                f"{case['at']} outside the filed timeline")
        if case["axis"] != "count" \
                and case["subject"] not in known:
            problems.append(
                f"{case['id']}: subject "
                f"{case['subject']} is not a filed entity")
        problems.extend(case_problems(case))
    rule_ids = [r["id"] for r in WYRD_PANEL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate WYRD_PANEL_RULES ids")
    for needed in ("the_world_is_filed_as_data_and_replayed",
                   "every_case_is_judged_against_a_filed_fact",
                   "claims_are_bound_to_their_answers",
                   "contradictions_are_filed_beside_consistency",
                   "honesty_about_the_unfiled_scores",
                   "pass_needs_the_full_set_reproducing",
                   "probes_prove_the_scorer_can_fail",
                   "the_world_is_synthetic_invented_only",
                   "the_panel_feeds_tf145_it_does_not_decide_it",
                   "the_wyrd_panel_baseline_is_not_gated_by_tf129"):
        if needed not in rule_ids:
            problems.append(f"WYRD_PANEL_RULES lost {needed}")

    # The live run: full PASS, measured totals.
    result = run_wyrd_panel()
    if result.verdict != "PASS" or not result.passes:
        problems.append(
            f"the live panel does not file PASS: "
            f"{result.verdict} ({result.mismatches} "
            f"mismatches)")
    if result.total_cases != 18 \
            or result.consistent_cases != 9 \
            or result.contradicted_cases != 8 \
            or result.honest_unanswerable_cases != 1:
        problems.append(
            f"the live panel files {result.total_cases} "
            f"cases / {result.consistent_cases} "
            f"consistent / {result.contradicted_cases} "
            f"contradicted / "
            f"{result.honest_unanswerable_cases} honest, "
            f"not 18 / 9 / 8 / 1")
    if result.overall_score != 10 / 18:
        problems.append(
            f"the live panel overall is "
            f"{result.overall_score}, not 10/18")

    # Partial + empty selections file REVIEW.
    empty = run_wyrd_panel(case_ids=())
    if empty.total_cases != 0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty selection does not file 0 / REVIEW")
    if case_ids:
        subset = run_wyrd_panel(case_ids=(case_ids[0],))
        if subset.total_cases != 1 or subset.verdict != "REVIEW":
            problems.append(
                "a single-case selection does not score "
                "its case / REVIEW mechanically")

    # The probes all behave (they build on filed
    # cases, so an emptied filing is named here
    # instead of crashing the probe leg).
    if PANEL_CASES:
        probes = run_panel_probes()
        if len(probes) != 7 or not all(p["passed"] for p in probes):
            problems.append(
                f"the panel probes do not all pass: "
                f"{[p['id'] for p in probes if not p['passed']]}")
    else:
        problems.append(
            "the filed set is empty — the probes "
            "cannot run")

    # The doc files the world + every case.
    doc = _REPO_ROOT / WYRD_PANEL_DOC
    if not doc.exists():
        problems.append(f"the write-up {WYRD_PANEL_DOC} is missing")
    else:
        doc_text = doc.read_text(encoding="utf-8")
        rendered = render_wyrd_panel_markdown(result.cases)
        if rendered.strip() not in doc_text:
            problems.append(
                f"the doc {WYRD_PANEL_DOC} does not file "
                f"the rendered panel verbatim")
        for case in result.cases:
            for text in (case["id"], case["question"],
                         case["answer"]):
                if text not in doc_text:
                    problems.append(
                        f"the doc {WYRD_PANEL_DOC} lost "
                        f"{text!r} of {case['id']}")

    # Harness consistency as filed + every pinned
    # baseline preserved exactly (this slice's
    # included), and the gate's gated set untouched.
    from benchmarks.harness import (
        BASELINES,
        BENCHMARKS,
        FIXTURES,
        baseline_problems,
    )
    from benchmarks.regression_gate import GATED_BASELINE_IDS

    if "wyrd_panel" not in [b["id"] for b in BENCHMARKS]:
        problems.append("BENCHMARKS lost the wyrd_panel entry")
    if "wyrd_panel_corpus" not in [f["id"] for f in FIXTURES]:
        problems.append("FIXTURES lost the wyrd_panel_corpus entry")
    if WYRD_PANEL_BASELINE_ID not in [b["id"] for b in BASELINES]:
        problems.append("BASELINES lost the wyrd-panel baseline")
    if WYRD_PANEL_BASELINE_ID in GATED_BASELINE_IDS:
        problems.append(
            "the wyrd-panel baseline entered TF-129's "
            "gated set (it is judged by this checker, "
            "not gated)")
    if len(GATED_BASELINE_IDS) != 9:
        problems.append("TF-129's gated set changed shape")
    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"harness baselines moved under the wyrd "
            f"panel: {earlier}")
    return problems


def wyrd_panel_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_wyrd_panel()
    probes = run_panel_probes()
    return {
        "format": WYRD_PANEL_FORMAT,
        "doc": WYRD_PANEL_DOC,
        "world": {
            "name": WORLD_NAME,
            "entities": [dict(e) for e in WORLD_ENTITIES],
            "events": [dict(e) for e in WORLD_EVENTS],
        },
        "axes": list(PANEL_AXES),
        "cases": result.cases,
        "measured": {
            "total_cases": result.total_cases,
            "consistent_cases": result.consistent_cases,
            "contradicted_cases": result.contradicted_cases,
            "honest_unanswerable_cases":
                result.honest_unanswerable_cases,
            "overall_score": result.overall_score,
            "probes_passed": sum(
                1 for p in probes if p["passed"]),
            "probes_total": len(probes),
            "verdict": result.verdict,
        },
        "probes": probes,
        "rules": [dict(r) for r in WYRD_PANEL_RULES],
    }


__all__ = [
    "HONESTY_MARKERS",
    "PANEL_AXES",
    "PANEL_CASES",
    "PANEL_PROBES",
    "TOTAL_CASES",
    "TOTAL_ENTITIES",
    "TOTAL_EVENTS",
    "WORLD_ENTITIES",
    "WORLD_EVENTS",
    "WORLD_NAME",
    "WYRD_PANEL_BASELINE_ID",
    "WYRD_PANEL_DOC",
    "WYRD_PANEL_FORMAT",
    "WYRD_PANEL_RULES",
    "WyrdPanelResult",
    "case_problems",
    "render_wyrd_panel_markdown",
    "run_panel_probes",
    "run_wyrd_panel",
    "score_case",
    "world_state_at",
    "wyrd_panel_as_data",
    "wyrd_panel_problems",
]
