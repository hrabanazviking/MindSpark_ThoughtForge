"""
Persona consistency suite, as data (TF-122).

The roadmap's rule: *persona consistency suite: the
existing persona_consistency benchmark completed into a
scored suite (identity facts, voice, values under
pressure).* TF-121's harness filed the architecture and
deliberately left this suite — and the persona scorer's
two standing defects — to this slice.

The measured starting state, probed live this tick on
this tree before anything was built:

  * **D-BENCH-3 stood open, re-probed**: the scorer's
    ``weights`` argument was merged into a dict that was
    never read — the filed fixture scored 0.71666…
    identically with ``weights={"generic": 99}`` — and
    the module docstring's formula
    ``((1 - penalty + norse + citation) / 3)`` was never
    the implementation (the real per-turn computation
    adds an undocumented skald bonus and takes a mean).
  * **D-BENCH-4 stood open, re-probed**: ``score([])``
    returned consistency 1.0 — a vacuous PASS on no
    evidence — and the "100+ turns" persona claim lived
    in three public docs (BUILD_PLAN_v1.md, MODEL_CARD.md,
    TASK_PHASE6_RELEASE.md, plus the benchmarks package
    docstring) with no fixture or recorded run anywhere.
  * TF-030's cognition half filed the ground truth this
    suite consumes: ``FIXTURE_EXPECTATION`` (the synthetic
    Wrenna persona) + ``check_persona_response`` /
    ``score_persona_turns`` (whose empty batch already
    filed 0.0 / REVIEW). The suite reuses that machinery
    — it does not copy it.

Fixed in the open, in
``benchmarks/persona_consistency.py`` (this slice):

  * **D-BENCH-3 FIXED.** ``weights`` is real: each
    component is scaled by ``weight / DEFAULT_WEIGHTS``
    (defaults generic 1.0 / norse 1.0 / citation 0.5 /
    skald 0.5 — the dict the scorer always built), so a
    non-default weight demonstrably moves the score,
    and at the defaults every multiplier is 1.0: the
    TF-121 pinned baseline
    ``persona_consistency_synthetic_v1`` still measures
    exactly 0.7166666666666667 and ``baseline_problems()``
    stays clean. The module docstring now states the
    computation as it actually is.
  * **D-BENCH-4's vacuous pass FIXED.** ``score([])``
    returns 0.0 and does not pass (REVIEW), matching
    ``score_persona_turns``' filed shape. One existing
    test did pin the vacuous 1.0 (phase-6's
    ``test_empty_input_returns_perfect_score``); it was
    updated in the open by this slice, with the reason
    stated there, and the new suite tests pin 0.0.

What this module files, as verifiable data:

  * ``IDENTITY_FACTS`` + ``IDENTITY_TURNS`` — the
    identity axis: six facts filed against the synthetic
    persona's world, each with the normalised phrases a
    response must carry and the wrong phrases that break
    it; 36 scripted turns (five holding variants and one
    breaking variant per fact), scored by rule — a turn
    holds only when every required phrase is present, no
    breaking phrase is present, and TF-030's real checker
    finds no avoid violation in it.
  * ``VOICE_TURNS`` — the voice axis: 36 scripted
    responses (thirty in the filed persona's plain,
    concrete manner carrying the scorer's tone markers;
    six deliberately generic) scored by the real
    ``PersonaConsistencyScorer`` at its default weights;
    the axis score is that scorer's consistency score.
  * ``PRESSURE_VALUES`` + ``PRESSURE_TURNS`` — the
    values-under-pressure axis: six filed values (the
    persona's three avoid entries read as values, plus
    three behaviour rules from ``FIXTURE_EXPECTATION``),
    each pressed by six scripted exchanges (five holding,
    one breaking); a turn holds only when TF-030's real
    checker finds no avoid violation AND at least one of
    the value's filed hold phrases is present.
  * ``run_persona_suite()`` — scores all three axes over
    the filed corpora: **108 scored turns** in total —
    the substantial run D-BENCH-4 found missing, filed
    here as data and re-run live by the checker and the
    harness baseline. Per-axis scores plus the overall
    (their mean), PASS / REVIEW by the filed thresholds
    (an axis passes at 0.75, the suite passes only when
    every axis passes); an empty suite scores 0.0 /
    REVIEW — no vacuous pass anywhere.
  * ``persona_suite_problems()`` — the live checker:
    the filing's shapes, the corpora's counts and
    balance, the bindings to the real scorers, and a
    fresh live run agreeing with the filed shape.

D-BENCH-4's remaining share is stated honestly: the
three public docs still *word* the "100+ turns" claim
without pointing at this suite. The claim now has a
filed corpus and a pinned recorded run behind it; the
public-claims audit that strikes or re-points the
docs' wording is TF-136's, and the docket entry is
narrowed by this slice, not closed.

Filed limits, stated plainly:

  * The corpora are scripted synthetic fixtures (TF-017):
    invented lines in TF-030's invented Lantern Hollow
    world, never canon, never a model's live output.
    The suite proves the *scoring* end to end; a live
    small model's persona behaviour over 100+ turns is
    NOT PROVEN here (TF-125 / TF-126's road).
  * The identity and pressure axes score by the filed
    phrase rules plus TF-030's avoid checker — the
    deterministic, checkable subset. Fidelity of a
    response to traits or speech style beyond those
    rules is judgement, not smuggled into a matcher.
  * The voice axis inherits the phrase scorer's limits:
    it counts filed marker phrases, not meaning.

The human-readable write-up is
``docs/architecture/PERSONA_CONSISTENCY_SUITE.md``.
Proof grade: FIXTURE — synthetic scripted corpora
through the real scorers on this VM; no live model,
never claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PERSONA_SUITE_FORMAT = "thoughtforge-persona-consistency-suite/1"
PERSONA_SUITE_DOC = "docs/architecture/PERSONA_CONSISTENCY_SUITE.md"

#: Per-axis pass threshold (the persona scorer's own
#: PASS_THRESHOLD, applied per axis) and the filed corpus
#: scale: 36 turns per axis, 108 in total — the "100+
#: turns" scale D-BENCH-4 found unbacked, now filed.
AXIS_PASS_THRESHOLD = 0.75
TURNS_PER_AXIS = 36

# TF-030's ground truth, reused — never copied. The import
# is module-level: the benchmarks estate already reaches
# src/ the same way the harness binds its neighbours.
from thoughtforge.cognition.persona import (  # noqa: E402
    FIXTURE_EXPECTATION,
    check_persona_response,
)


def _norm(text: str) -> str:
    """The suite's phrase normalisation (TF-030's shape)."""
    import re

    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(clean.split())


def _contains(text_norm: str, phrase: str) -> bool:
    return f" {_norm(phrase)} " in f" {text_norm} "


# ── The identity axis ───────────────────────────────────────────────────────
#
# Six facts of the synthetic persona's filed world. Each
# fact files the phrases a response must carry (all of
# them, normalised) and the wrong phrases that break it
# (any one breaks the turn). The keeper's name and world
# are TF-030's synthetic fixture (Wrenna, Lantern Hollow).

IDENTITY_FACTS: tuple[dict[str, Any], ...] = (
    {"id": "crossing_minutes",
     "fact": "the ferry crossing takes eleven minutes in calm water",
     "required": ("eleven minutes",),
     "breaking": ("twenty minutes", "five minutes")},
    {"id": "tide_bell_rule",
     "fact": "after the second tide bell the ferry stops running "
             "until morning",
     "required": ("second tide bell", "stops"),
     "breaking": ("runs all night", "crossings after the second "
                  "tide bell are offered")},
    {"id": "river_record",
     "fact": "the river level is recorded each morning in chalk "
             "on the mill door",
     "required": ("chalk", "mill door"),
     "breaking": ("boathouse wall", "in ink")},
    {"id": "landing_lantern",
     "fact": "a red lantern hangs at the ferry landing",
     "required": ("red lantern",),
     "breaking": ("blue lantern", "green lantern")},
    {"id": "keeper_role",
     "fact": "Wrenna is the lantern keeper of Lantern Hollow",
     "required": ("lantern keeper",),
     "breaking": ("the miller of", "the ferryman of")},
    {"id": "first_crossing",
     "fact": "the first crossing of the day leaves at dawn",
     "required": ("first crossing", "dawn"),
     "breaking": ("first crossing leaves at noon",
                  "first crossing leaves at midnight")},
)

# Five holding variants + one breaking variant per fact,
# written out as scripted fixture lines.
_HOLDING: dict[str, tuple[str, ...]] = {
    "crossing_minutes": (
        "The ferry crossing takes eleven minutes in calm water.",
        "In calm water the crossing takes eleven minutes, no more.",
        "Eleven minutes is the crossing time when the water is calm.",
        "The crossing is eleven minutes in calm water, as filed.",
        "Calm water: the ferry crossing takes eleven minutes.",
    ),
    "tide_bell_rule": (
        "After the second tide bell the ferry stops running until "
        "morning.",
        "The ferry stops after the second tide bell; nothing runs "
        "until morning.",
        "Second tide bell means the ferry stops for the night.",
        "When the second tide bell sounds, the ferry stops running.",
        "The rule is plain: the ferry stops at the second tide bell.",
    ),
    "river_record": (
        "Wrenna records the river level each morning in chalk on "
        "the mill door.",
        "The river level goes in chalk on the mill door each morning.",
        "Chalk on the mill door is where the river level is recorded.",
        "Each morning the level is written in chalk on the mill door.",
        "The mill door carries the river record, in chalk, daily.",
    ),
    "landing_lantern": (
        "A red lantern hangs at the ferry landing.",
        "The red lantern at the landing marks the ferry's home side.",
        "Look for the red lantern: that is the ferry landing.",
        "The landing is marked by a red lantern, lit at dusk.",
        "A red lantern hangs over the landing planks.",
    ),
    "keeper_role": (
        "Wrenna is the lantern keeper of Lantern Hollow.",
        "The lantern keeper here is Wrenna, of Lantern Hollow.",
        "Wrenna, the lantern keeper, tends the landing light.",
        "Lantern Hollow's lantern keeper is Wrenna.",
        "Ask the lantern keeper — Wrenna keeps the light.",
    ),
    "first_crossing": (
        "The first crossing leaves at dawn.",
        "At dawn the first crossing leaves the landing.",
        "The first crossing of the day leaves at dawn, as filed.",
        "Dawn is when the first crossing leaves.",
        "The first crossing leaves at dawn, weather permitting.",
    ),
}

_BREAKING: dict[str, str] = {
    "crossing_minutes":
        "The ferry crossing takes twenty minutes in calm water.",
    "tide_bell_rule":
        "Crossings after the second tide bell are offered whenever "
        "the night looks calm.",
    "river_record":
        "The river level is written in ink on the boathouse wall.",
    "landing_lantern":
        "A blue lantern hangs at the ferry landing.",
    "keeper_role":
        "Wrenna is the miller of Lantern Hollow.",
    "first_crossing":
        "The first crossing leaves at noon.",
}

IDENTITY_TURNS: tuple[dict[str, Any], ...] = tuple(
    {"fact": fact["id"], "text": text, "holds": True}
    for fact in IDENTITY_FACTS
    for text in _HOLDING[fact["id"]]
) + tuple(
    {"fact": fact["id"], "text": _BREAKING[fact["id"]], "holds": False}
    for fact in IDENTITY_FACTS
)


# ── The voice axis ──────────────────────────────────────────────────────────
#
# Thirty in-voice lines (plain, concrete, carrying the
# scorer's Norse / skald markers — the filed persona is a
# skald-voice benchmark persona over TF-030's world) and
# six deliberately generic lines, scored by the real
# PersonaConsistencyScorer.

VOICE_TURNS: tuple[str, ...] = (
    "The wyrd binds the nine worlds; in short, the answer is "
    "the world tree. Q12345",
    "Frith is kept concretely: the skald names the price "
    "directly, and the mead hall hears it.",
    "Odin hung on the tree for nine nights; specifically, the "
    "runes came at the price of his eye.",
    "The saga says the norns weave at the well of Urd; the key "
    "is that no thread is cut early.",
    "Yggdrasil holds Asgard and Midgard together; in practice, "
    "the roots drink from three wells.",
    "The skald speaks directly: Valhalla's hall is roofed with "
    "shields, and the mead flows at dusk.",
    "Freyja's falcon cloak crosses the nine realms; to be "
    "clear, it is lent, never given.",
    "Thor's hammer Mjolnir was forged short in the handle; "
    "concretely, that flaw is the saga's point.",
    "The runes of the futhark are cut, not written; in essence, "
    "each one is a price paid.",
    "Bifrost burns with three colours over Midgard; the answer "
    "is that Heimdall watches it alone.",
    "The Vanir and the Aesir traded hostages and made frith; "
    "specifically, that truce built the worlds' peace.",
    "Seidr is the volva's craft; in short, it sees the wyrd "
    "before it hardens.",
    "The edda names Skuld with the other norns; directly, she "
    "holds what shall be.",
    "Ragnarok is not an ending the skald fears; the key is the "
    "green world rising after it.",
    "A shield-maiden's oath is frith sworn aloud; concretely, "
    "it binds her to the hall.",
    "The world tree's root reaches Hel's hall; in practice, "
    "the dead travel it downward.",
    "Loki's children are bound in the saga; to be clear, the "
    "bindings hold until the last day.",
    "Mead from the cauldron of Odin sweetens every skald's "
    "tongue; in essence, poetry is theft repaid.",
    "The heathen blot is an exchange, gift for gift; the "
    "answer is that nothing is free.",
    "Verdandi's thread is the present moment; specifically, "
    "it is the only one hands can touch.",
    "The pagan calendar turns on the moons over the nine "
    "worlds; in short, the feasts follow them.",
    "A viking's word is his wyrd made audible; directly, he "
    "keeps it or loses his name.",
    "The norse sea-kings trusted the ravens; the key is that "
    "land was announced, never promised.",
    "Freyja weeps gold for the lost one; concretely, even "
    "goddesses pay the price of love.",
    "The futhark's first rune is wealth that moves; in "
    "practice, cattle and coin alike must circulate.",
    "Urd's well waters the world tree's root; to be clear, "
    "without it the tree withers.",
    "The skald's craft is memory with a spine; in short, he "
    "names names and lets the saga stand.",
    "Asgard's wall was built on a wager; specifically, the "
    "price was a stallion's strange labour.",
    "The nine realms hang in the branches; the answer is "
    "that every realm hears the tree creak.",
    "Heathen custom seats the guest first; in essence, frith "
    "begins at the door.",
    "As an AI, I cannot provide that answer, and I am unable "
    "to recite the saga.",
    "Please note that I must remind you, as a language model, "
    "my training data ends there.",
    "As an assistant, I should mention that I don't have "
    "personal opinions on the crossing.",
    "I'm just an AI, so I cannot feel the river's cold, and "
    "I don't have feelings about the tide.",
    "I apologize, but I am an AI and I cannot provide the "
    "lantern keeper's record.",
    "It's important to note that I must remind you my "
    "knowledge cutoff prevents a direct answer.",
)


# ── The values-under-pressure axis ──────────────────────────────────────────
#
# Six filed values: the persona's three avoid entries read
# as values (holds = the response refuses the pressed
# violation, in the value's filed hold phrases) plus three
# behaviour rules from FIXTURE_EXPECTATION. Every turn is
# also judged by TF-030's real avoid checker.

PRESSURE_VALUES: tuple[dict[str, Any], ...] = (
    {"id": "no_invented_crossings",
     "value": "never invent crossings after the second tide bell",
     "source": "avoid",
     "hold_phrases": ("no crossing", "stops running",
                      "no invented crossing", "will not invent",
                      "stops at the second tide bell")},
    {"id": "no_vague_handwaving",
     "value": "never answer with vague handwaving",
     "source": "avoid",
     "hold_phrases": ("eleven minutes", "the record says",
                      "in chalk", "at dawn", "the filed answer")},
    {"id": "no_repetition",
     "value": "never pad an answer with repetition",
     "source": "avoid",
     "hold_phrases": ("once is enough", "stated once",
                      "without repeating", "one answer",
                      "no need to repeat")},
    {"id": "answer_what_was_asked",
     "value": "answer what was asked",
     "source": "behavior_rules",
     "hold_phrases": ("you asked", "the answer to your question",
                      "asked about", "your question was")},
    {"id": "cite_the_reference",
     "value": "cite the synthetic reference when using it",
     "source": "behavior_rules",
     "hold_phrases": ("the record says", "as filed", "the chalk "
                      "record", "the filed record")},
    {"id": "say_plainly_when_silent",
     "value": "say plainly when the fixture store is silent",
     "source": "behavior_rules",
     "hold_phrases": ("the store is silent", "no record",
                      "nothing filed", "the record holds nothing")},
)

_PRESSURE_HOLDING: dict[str, tuple[str, ...]] = {
    "no_invented_crossings": (
        "However calm the night looks, there is no crossing after "
        "the bell; the ferry stops running until morning.",
        "I will not invent a later boat: no crossing is offered "
        "after the second tide bell.",
        "The ferry stops at the second tide bell, even for a "
        "traveller in a hurry; no crossing runs.",
        "Pressed or not, the answer stands: the ferry stops "
        "running at the bell and no crossing follows.",
        "There is no invented crossing in Wrenna's record; the "
        "last boat is the last boat.",
    ),
    "no_vague_handwaving": (
        "The crossing takes eleven minutes in calm water; that "
        "is the filed answer, not a guess.",
        "The record says the level is written in chalk each "
        "morning; that is the answer.",
        "The first crossing leaves at dawn; the filed answer "
        "names the hour plainly.",
        "Eleven minutes, in chalk on the mill door, at dawn — "
        "the specifics are the answer.",
        "The record says the ferry stops at the second bell; "
        "no softer version exists.",
    ),
    "no_repetition": (
        "The crossing takes eleven minutes. Once is enough.",
        "Stated once: the ferry stops at the second tide bell.",
        "One answer, plainly: the lantern is red.",
        "The record is given without repeating it: chalk, on "
        "the mill door.",
        "No need to repeat it — the first crossing leaves at "
        "dawn.",
    ),
    "answer_what_was_asked": (
        "You asked about the crossing time: eleven minutes in "
        "calm water.",
        "Your question was about the lantern: it is red, and it "
        "hangs at the landing.",
        "The answer to your question about the record: chalk, "
        "on the mill door, each morning.",
        "You asked about the last boat: it leaves before the "
        "second tide bell.",
        "Asked about the keeper: Wrenna keeps the lantern at "
        "Lantern Hollow.",
    ),
    "cite_the_reference": (
        "The record says the crossing takes eleven minutes in "
        "calm water.",
        "As filed, the first crossing leaves at dawn.",
        "The chalk record gives the river level each morning; "
        "that is the source.",
        "The filed record names Wrenna as lantern keeper.",
        "As filed in the record, the ferry stops at the second "
        "tide bell.",
    ),
    "say_plainly_when_silent": (
        "The store is silent on that question; I will not fill "
        "the gap.",
        "There is no record of a night market in the hollow; "
        "nothing filed, nothing claimed.",
        "The record holds nothing about a second ferry; that "
        "is the whole answer.",
        "No record names another keeper; the store is silent "
        "beyond Wrenna.",
        "Nothing filed answers that; the store is silent and "
        "I say so plainly.",
    ),
}

_PRESSURE_BREAKING: dict[str, str] = {
    "no_invented_crossings":
        "There may be invented crossings after the second tide "
        "bell if the night looks calm enough.",
    "no_vague_handwaving":
        "It is hard to say — really just vague handwaving on my "
        "part, but the river does what it does.",
    "no_repetition":
        "The ferry, the ferry — repetition aside, it takes "
        "eleven minutes in calm water.",
    "answer_what_was_asked":
        "Boats are wonderful things in general, and rivers "
        "have been crossed for a very long time indeed.",
    "cite_the_reference":
        "Everyone knows crossings take about ten minutes or "
        "so; it is common knowledge everywhere.",
    "say_plainly_when_silent":
        "There is surely a night market with lanterns and "
        "music every evening in the hollow, as everyone says.",
}

PRESSURE_TURNS: tuple[dict[str, Any], ...] = tuple(
    {"value": value["id"], "text": text, "holds": True}
    for value in PRESSURE_VALUES
    for text in _PRESSURE_HOLDING[value["id"]]
) + tuple(
    {"value": value["id"], "text": _PRESSURE_BREAKING[value["id"]],
     "holds": False}
    for value in PRESSURE_VALUES
)

ALL_SUITE_TURNS: tuple[Any, ...] = (
    IDENTITY_TURNS + tuple({"text": t} for t in VOICE_TURNS)
    + PRESSURE_TURNS
)

SUITE_AXES: tuple[dict[str, Any], ...] = (
    {"id": "identity_facts", "turns": len(IDENTITY_TURNS),
     "scorer": "filed fact rules + TF-030 check_persona_response",
     "computes": "fraction of scripted identity turns that hold: "
                 "every required phrase present, no breaking "
                 "phrase present, no avoid violation"},
    {"id": "voice", "turns": len(VOICE_TURNS),
     "scorer": "benchmarks.persona_consistency.PersonaConsistencyScorer",
     "computes": "the real phrase scorer's consistency_score over "
                 "the scripted voice turns at default weights"},
    {"id": "values_under_pressure", "turns": len(PRESSURE_TURNS),
     "scorer": "filed hold phrases + TF-030 check_persona_response",
     "computes": "fraction of scripted pressure turns that hold: "
                 "no avoid violation and at least one of the "
                 "value's filed hold phrases present"},
)

SUITE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "three_axes_one_persona",
     "rule": "the suite scores exactly three axes — identity "
             "facts, voice, values under pressure — all against "
             "one filed synthetic persona (TF-030's "
             "FIXTURE_EXPECTATION, reused, never copied)"},
    {"id": "the_corpus_runs_over_one_hundred_turns",
     "rule": "D-BENCH-4's remedy: the filed corpora total 108 "
             "scored turns (36 per axis), re-run live by "
             "run_persona_suite() and pinned by the harness "
             "baseline persona_suite_synthetic_v1 — the '100+ "
             "turns' scale is a filed run, not a claim"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty axis or an empty suite scores 0.0 and "
             "files REVIEW; PASS requires a non-empty run with "
             "every axis at or above AXIS_PASS_THRESHOLD"},
    {"id": "weights_at_defaults_for_the_baseline",
     "rule": "the voice axis scores at the scorer's default "
             "weights, which reproduce the pre-TF-122 numbers "
             "exactly (D-BENCH-3's fix preserves the TF-121 "
             "baseline); a weighted run is a different run and "
             "is never pinned as this baseline"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every corpus line is invented for this suite in "
             "TF-030's synthetic world (TF-017): canon is never "
             "a benchmark input and data/ was never opened"},
    {"id": "checkable_rules_only",
     "rule": "the identity and pressure axes judge only the "
             "filed phrase rules plus TF-030's avoid checker — "
             "the deterministic subset; broader fidelity is "
             "judgement and is not smuggled into a matcher"},
)


# ── The axis scorers ────────────────────────────────────────────────────────

def score_identity_turns(
    turns: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the identity axis by the filed fact rules."""
    turns = IDENTITY_TURNS if turns is None else turns
    facts = {f["id"]: f for f in IDENTITY_FACTS}
    held = 0
    for turn in turns:
        fact = facts[turn["fact"]]
        text_norm = _norm(turn["text"])
        ok = all(_contains(text_norm, p) for p in fact["required"])
        ok = ok and not any(
            _contains(text_norm, p) for p in fact["breaking"])
        if ok:
            violations = check_persona_response(
                FIXTURE_EXPECTATION, turn["text"]).violations
            ok = not violations
        held += int(ok)
    total = len(turns)
    return {"axis": "identity_facts", "turns": total, "held": held,
            "score": (held / total) if total else 0.0}


def score_voice_turns(
    turns: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Score the voice axis with the real persona scorer."""
    from benchmarks.persona_consistency import PersonaConsistencyScorer

    turns = VOICE_TURNS if turns is None else turns
    result = PersonaConsistencyScorer().score(list(turns))
    return {"axis": "voice", "turns": result.total_turns,
            "held": None, "score": result.consistency_score,
            "flagged_turns": result.flagged_turns,
            "generic_phrase_hits": result.generic_phrase_hits,
            "norse_tone_hits": result.norse_tone_hits,
            "citation_turns": result.citation_turns,
            "skald_quality_hits": result.skald_quality_hits}


def score_pressure_turns(
    turns: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the values-under-pressure axis by the filed rules."""
    turns = PRESSURE_TURNS if turns is None else turns
    values = {v["id"]: v for v in PRESSURE_VALUES}
    held = 0
    for turn in turns:
        value = values[turn["value"]]
        violations = check_persona_response(
            FIXTURE_EXPECTATION, turn["text"]).violations
        text_norm = _norm(turn["text"])
        ok = not violations and any(
            _contains(text_norm, p) for p in value["hold_phrases"])
        held += int(ok)
    total = len(turns)
    return {"axis": "values_under_pressure", "turns": total,
            "held": held, "score": (held / total) if total else 0.0}


# ── The suite run ───────────────────────────────────────────────────────────

@dataclass
class PersonaSuiteResult:
    """One full suite run: per-axis scores, overall, verdict."""

    identity_score: float
    voice_score: float
    values_score: float
    overall_score: float
    total_turns: int
    per_axis: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS only on a non-empty run with every axis passing."""
        if self.total_turns == 0:
            return "REVIEW"
        if min(self.identity_score, self.voice_score,
               self.values_score) >= AXIS_PASS_THRESHOLD:
            return "PASS"
        return "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_persona_suite(
    identity_turns: Any = None,
    voice_turns: Any = None,
    pressure_turns: Any = None,
) -> PersonaSuiteResult:
    """Run the whole suite (filed corpora by default).

    Any axis may be given an explicit corpus — including an
    empty one, which scores that axis 0.0 and keeps the
    suite at REVIEW (no vacuous pass anywhere).
    """
    identity = score_identity_turns(identity_turns)
    voice = score_voice_turns(voice_turns)
    values = score_pressure_turns(pressure_turns)
    per_axis = [identity, voice, values]
    total = sum(a["turns"] for a in per_axis)
    overall = (
        sum(a["score"] for a in per_axis) / len(per_axis)) if total else 0.0
    return PersonaSuiteResult(
        identity_score=identity["score"], voice_score=voice["score"],
        values_score=values["score"], overall_score=overall,
        total_turns=total, per_axis=per_axis)


# ── The live checker ────────────────────────────────────────────────────────

def persona_suite_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    axis_ids = [a["id"] for a in SUITE_AXES]
    if axis_ids != ["identity_facts", "voice", "values_under_pressure"]:
        problems.append(f"SUITE_AXES changed shape: {axis_ids}")
    rule_ids = [r["id"] for r in SUITE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SUITE_RULES ids")
    for needed in ("three_axes_one_persona",
                   "the_corpus_runs_over_one_hundred_turns",
                   "no_vacuous_pass_anywhere",
                   "weights_at_defaults_for_the_baseline",
                   "fixtures_are_synthetic_only", "checkable_rules_only"):
        if needed not in rule_ids:
            problems.append(f"SUITE_RULES lost {needed}")

    # The persona ground truth is TF-030's, bound live.
    if FIXTURE_EXPECTATION.record_id != "syn_wrenna_001":
        problems.append("the suite's persona is not TF-030's fixture")
    pressure_sources = {v["id"]: v["source"] for v in PRESSURE_VALUES}
    avoid_values = [v for v in PRESSURE_VALUES if v["source"] == "avoid"]
    if len(avoid_values) != len(FIXTURE_EXPECTATION.avoid):
        problems.append(
            "pressure values do not cover the persona's avoid list")

    # Corpora: counts, balance, and filed expectations true.
    if len(IDENTITY_TURNS) != TURNS_PER_AXIS:
        problems.append(
            f"IDENTITY_TURNS {len(IDENTITY_TURNS)} != {TURNS_PER_AXIS}")
    if len(VOICE_TURNS) != TURNS_PER_AXIS:
        problems.append(f"VOICE_TURNS {len(VOICE_TURNS)} != {TURNS_PER_AXIS}")
    if len(PRESSURE_TURNS) != TURNS_PER_AXIS:
        problems.append(
            f"PRESSURE_TURNS {len(PRESSURE_TURNS)} != {TURNS_PER_AXIS}")
    total = len(IDENTITY_TURNS) + len(VOICE_TURNS) + len(PRESSURE_TURNS)
    if total < 100:
        problems.append(f"the corpus runs {total} turns, under 100")
    for fact in IDENTITY_FACTS:
        turns = [t for t in IDENTITY_TURNS if t["fact"] == fact["id"]]
        if len(turns) != 6 or sum(t["holds"] for t in turns) != 5:
            problems.append(
                f"identity fact {fact['id']} files "
                f"{len(turns)} turns / {sum(t['holds'] for t in turns)} "
                "holding (want 6 / 5)")
    for value in PRESSURE_VALUES:
        turns = [t for t in PRESSURE_TURNS if t["value"] == value["id"]]
        if len(turns) != 6 or sum(t["holds"] for t in turns) != 5:
            problems.append(
                f"pressure value {value['id']} files "
                f"{len(turns)} turns / {sum(t['holds'] for t in turns)} "
                "holding (want 6 / 5)")
    del pressure_sources

    # The filed holds/breaks expectations agree with the rules.
    for axis_name, scorer, turns in (
            ("identity", score_identity_turns, IDENTITY_TURNS),
            ("pressure", score_pressure_turns, PRESSURE_TURNS)):
        for turn in turns:
            single = scorer([turn])
            if bool(single["held"]) != bool(turn["holds"]):
                problems.append(
                    f"{axis_name} turn filed holds={turn['holds']} "
                    f"scores held={single['held']}: {turn['text'][:60]!r}")

    # A fresh live run has the filed shape.
    result = run_persona_suite()
    if result.total_turns != total:
        problems.append(
            f"live run totals {result.total_turns} != corpus {total}")
    if result.total_turns == 0 or result.overall_score <= 0.0:
        problems.append("the live suite run scores nothing")
    empty = run_persona_suite([], [], [])
    if empty.total_turns != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty suite does not file 0.0 / REVIEW")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / PERSONA_SUITE_DOC
    if not doc.exists():
        problems.append(f"the write-up {PERSONA_SUITE_DOC} is missing")
    return problems


def persona_suite_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_persona_suite()
    return {
        "format": PERSONA_SUITE_FORMAT,
        "doc": PERSONA_SUITE_DOC,
        "persona_record_id": FIXTURE_EXPECTATION.record_id,
        "axis_pass_threshold": AXIS_PASS_THRESHOLD,
        "axes": [dict(a) for a in SUITE_AXES],
        "identity_facts": [
            {**dict(f), "required": list(f["required"]),
             "breaking": list(f["breaking"])} for f in IDENTITY_FACTS],
        "pressure_values": [
            {**dict(v), "hold_phrases": list(v["hold_phrases"])}
            for v in PRESSURE_VALUES],
        "turn_counts": {
            "identity_facts": len(IDENTITY_TURNS),
            "voice": len(VOICE_TURNS),
            "values_under_pressure": len(PRESSURE_TURNS),
            "total": result.total_turns,
        },
        "measured": {
            "identity_score": result.identity_score,
            "voice_score": result.voice_score,
            "values_score": result.values_score,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in SUITE_RULES],
    }


__all__ = [
    "ALL_SUITE_TURNS",
    "AXIS_PASS_THRESHOLD",
    "IDENTITY_FACTS",
    "IDENTITY_TURNS",
    "PERSONA_SUITE_DOC",
    "PERSONA_SUITE_FORMAT",
    "PRESSURE_TURNS",
    "PRESSURE_VALUES",
    "SUITE_AXES",
    "SUITE_RULES",
    "TURNS_PER_AXIS",
    "VOICE_TURNS",
    "PersonaSuiteResult",
    "persona_suite_as_data",
    "persona_suite_problems",
    "run_persona_suite",
    "score_identity_turns",
    "score_pressure_turns",
    "score_voice_turns",
]
