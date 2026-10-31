"""Persona consistency hooks — the cognition layer's half (TF-030).

The persona_consistency benchmark (``benchmarks/persona_consistency.py``)
scores *responses* against a hard-coded Skald phrase list. Before
TF-030 the cognition layer gave that benchmark — and any other
consumer — nothing to score against except its own hard-coding,
because the persona the forge actually loaded barely steered
anything:

* ``PersonalityCoreRecord`` carries ``traits``, ``speech_style``,
  ``behavior_rules``, ``avoid`` and five tuning ``weights``
  (knowledge/models.py), but TF-009 measured the whole expressive
  surface DORMANT (D-CFG-2): the scaffold consumed only
  ``avoid[:3]`` plus ``record_id`` for a log line. Traits, style
  and behaviour rules were parsed, carried, and silently dropped —
  a persona whose voice never reaches the prompt cannot drift,
  because it was never present.
* The core's fragment/final scorers compute a ``personality_fit``
  component, but it is derived from a genericness penalty on the
  text alone — it never looks at the loaded persona either.

This module is the cognition layer's half of persona consistency —
the hooks the benchmark half (TF-122's suite; the roadmap's TF-030
note names TF-128) binds to:

* **The expectation is data.** :class:`PersonaExpectation` is a
  snapshot of the persona *in force* for a core instance, derived
  from the loaded ``PersonalityCoreRecord`` — never from a
  hard-coded phrase list. It is the one object a scorer needs:
  what the persona is, in the persona's own recorded terms.
* **The persona now steers.** :func:`inject_persona` prepends the
  expectation's steering block — ``Persona traits:``,
  ``Persona speech style:`` and ``Persona behavior:`` lines, each
  list capped at ``MAX_STEERING_ENTRIES`` — to the scaffold's
  ``fact_block``, the same injection channel chat history
  (TF-019/TF-024) and the will layer (TF-029) use. Traits, speech
  style and behavior rules therefore reach every generation prompt
  (they ride the fact block's knowledge-context section, TF-023).
  The ``avoid`` list is *not* repeated in the block: it already
  steers through ``scaffold.avoid`` (universal + intent extras +
  personality ``avoid[:3]``, rendered as the prompt's own Avoid
  line, TF-023). The five tuning ``weights`` are carried on the
  expectation for scorers and for TF-141's persona-compiler
  verdict, and are given **no** prompt effect here — numbers are
  not prose, and inventing prose from them would be a compiler,
  which is exactly the question TF-141 exists to decide. That is
  the honest remainder of D-CFG-2, recorded in TF-016's docket.
* **The checkable subset is checked.** :func:`check_persona_response`
  is a pure function of (expectation, response text): it reports
  the persona's ``avoid`` entries that literally appear in the
  response (whitespace/punctuation-normalised; multi-word entries
  match as phrases, single-word entries as whole words). That is
  the part of consistency a deterministic hook can prove. Traits,
  speech style and behavior rules are *steering*, verified by
  their presence in the prompt — scoring a response's fidelity to
  them is judgement, owned by TF-122's suite, not smuggled into a
  phrase matcher here.
* **No vacuous pass.** :func:`score_persona_turns` aggregates
  per-turn checks into a :class:`PersonaConsistencyReport` whose
  verdict is ``PASS`` only when at least one turn was checked and
  every turn was consistent; an empty response list scores 0.0 /
  ``REVIEW`` — D-BENCH-4's lesson (TF-010 caught the benchmark's
  empty list scoring a vacuous 1.0 PASS), applied in the cognition
  half from the start, exactly as TF-028 applied it. Fixing the
  benchmark module itself stays TF-122's.
* **Fixtures ship with the hooks.** ``FIXTURE_EXPECTATION`` +
  ``PERSONA_FIXTURE_TURNS`` are synthetic ground truth in TF-017's
  invented Lantern Hollow world (Wrenna the Lantern-Keeper; the
  expectation mirrors ``tests/fixtures/synthetic_personality_core.yaml``
  and is synthetic in exactly the same way — no canon anywhere):
  in-voice responses that violate nothing, and out-of-voice
  responses that each trip exactly one avoid entry. TF-096
  (persona-voice preservation under salvage) and TF-122 consume
  these instead of inventing their own.

Core wiring (TF-030, minimal): ``ThoughtForgeCore`` derives one
expectation from its loaded personality at construction, exposes
it as ``core.persona_expectation``, and injects the steering block
into each turn's scaffold after history injection and before the
will layer observes the turn — so the fact_block order is: will
line, persona block, history, knowledge facts. A core with no
personality loaded has no expectation: injection is a no-op and
every pre-TF-030 prompt is byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.models import (
    CognitionScaffold,
    PersonalityCoreRecord,
)

# ── The tables ─────────────────────────────────────────────────────────────────

#: Steering entries carried per persona list (traits / speech
#: style / behavior rules). Bounded at injection, like every other
#: scaffold list (TF-021/TF-023 discipline): the whole bounded list
#: is the contract, and the bound lives here, once.
MAX_STEERING_ENTRIES = 6

_STEERING_LABELS: tuple[tuple[str, str], ...] = (
    ("traits", "Persona traits"),
    ("speech_style", "Persona speech style"),
    ("behavior_rules", "Persona behavior"),
)


def _normalise(text: str) -> str:
    """Lower-case, punctuation-stripped, whitespace-collapsed text.

    The one normalisation both sides of an avoid match go through,
    so "Vague  handwaving!" and "vague handwaving" are the same
    phrase to the checker.
    """
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(clean.split())


def _appears(entry_norm: str, text_norm: str) -> bool:
    """True when a normalised avoid entry appears in normalised text.

    Multi-word entries match as phrases; single-word entries match
    as whole words only (an avoid of "repetition" must not fire on
    "repetitions" inside another word's clothing — token equality,
    not substring).
    """
    if not entry_norm:
        return False
    if " " in entry_norm:
        return f" {entry_norm} " in f" {text_norm} "
    return entry_norm in text_norm.split()


# ── The expectation ────────────────────────────────────────────────────────────


@dataclass
class PersonaExpectation:
    """The persona in force, as data — what consistency is judged against."""

    record_id: str
    traits: list[str] = field(default_factory=list)
    speech_style: list[str] = field(default_factory=list)
    behavior_rules: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_record(
        cls, record: PersonalityCoreRecord | None
    ) -> "PersonaExpectation | None":
        """Snapshot the loaded record; None when no persona is loaded.

        Lists and weights are copied — the expectation is the
        construction-time snapshot, so later edits to the record
        object do not quietly re-persona a live conversation.
        """
        if record is None:
            return None
        return cls(
            record_id=record.record_id,
            traits=list(record.traits),
            speech_style=list(record.speech_style),
            behavior_rules=list(record.behavior_rules),
            avoid=list(record.avoid),
            weights=dict(record.weights),
        )

    # ── Steering ───────────────────────────────────────────────────────────

    def steering_lines(self) -> list[str]:
        """The persona block's lines, in fixed label order.

        One line per non-empty expressive list, entries joined with
        "; " and each list capped at MAX_STEERING_ENTRIES. The
        avoid list and the weights deliberately have no line here:
        avoid already steers via ``scaffold.avoid`` (TF-023), and
        weights are scorer data, not prompt prose (module docstring).
        """
        lines: list[str] = []
        for field_name, label in _STEERING_LABELS:
            entries = getattr(self, field_name)[:MAX_STEERING_ENTRIES]
            if entries:
                lines.append(f"{label}: {'; '.join(entries)}")
        return lines

    def steering_block(self) -> str:
        """The lines as one block ("" when the persona says nothing)."""
        return "\n".join(self.steering_lines())

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "traits": list(self.traits),
            "speech_style": list(self.speech_style),
            "behavior_rules": list(self.behavior_rules),
            "avoid": list(self.avoid),
            "weights": dict(self.weights),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonaExpectation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Scaffold injection ─────────────────────────────────────────────────────────


def inject_persona(
    scaffold: CognitionScaffold, expectation: PersonaExpectation | None
) -> bool:
    """Prepend the persona steering block to ``scaffold.fact_block``.

    Returns True when the block was added. Adds nothing when no
    persona is in force or the persona carries no expressive lines,
    and never duplicates its own block (the first steering line is
    the duplicate guard — the TF-029 will-line rule, generalised).
    The scaffold's per-turn fields are never touched: persona
    steers the conversation's voice, the turn's intent steers the
    turn.
    """
    if expectation is None:
        return False
    block = expectation.steering_block()
    if not block:
        return False
    first_line = block.split("\n", 1)[0]
    if first_line in scaffold.fact_block:
        return False
    scaffold.fact_block = (
        f"{block}\n\n{scaffold.fact_block}" if scaffold.fact_block else block
    )
    return True


# ── The per-turn check ─────────────────────────────────────────────────────────


@dataclass
class PersonaTurnCheck:
    """One response checked against the persona in force."""

    turn_index: int
    violations: list[str] = field(default_factory=list)  # avoid entries found

    @property
    def consistent(self) -> bool:
        return not self.violations


def check_persona_response(
    expectation: PersonaExpectation | None,
    text: str,
    turn_index: int = 0,
) -> PersonaTurnCheck:
    """Check one response against the persona's avoid list.

    Pure and deterministic. Raises ValueError when no persona is
    in force — consistency with nothing is not a passing grade,
    it is a category error, and the hook says so instead of
    returning a clean sheet.
    """
    if expectation is None:
        raise ValueError("no persona in force: nothing to check against")
    text_norm = _normalise(text)
    violations = [
        entry
        for entry in expectation.avoid
        if _appears(_normalise(entry), text_norm)
    ]
    return PersonaTurnCheck(turn_index=turn_index, violations=violations)


# ── The aggregate report ───────────────────────────────────────────────────────


@dataclass
class PersonaConsistencyReport:
    """Persona consistency across a batch of responses (cognition half).

    The benchmark half (TF-122) owns the full suite — identity
    facts, voice, values under pressure. This report is the
    mechanical floor the cognition layer can prove on its own:
    of the turns checked, how many kept the persona's avoid list.
    """

    total_turns: int = 0
    consistent_turns: int = 0
    violation_count: int = 0
    per_turn: list[PersonaTurnCheck] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Consistent turns / turns checked; 0.0 when nothing was checked."""
        if self.total_turns == 0:
            return 0.0
        return self.consistent_turns / self.total_turns

    @property
    def verdict(self) -> str:
        """PASS only on a non-empty, fully consistent batch."""
        if self.total_turns > 0 and self.consistent_turns == self.total_turns:
            return "PASS"
        return "REVIEW"


def score_persona_turns(
    expectation: PersonaExpectation | None, responses: list[str]
) -> PersonaConsistencyReport:
    """Aggregate check over a batch of final responses.

    An empty batch is 0 turns / score 0.0 / REVIEW — never the
    vacuous PASS D-BENCH-4 caught the benchmark awarding (TF-010).
    Raises ValueError when no persona is in force.
    """
    if expectation is None:
        raise ValueError("no persona in force: nothing to score against")
    per_turn = [
        check_persona_response(expectation, text, turn_index=i)
        for i, text in enumerate(responses)
    ]
    return PersonaConsistencyReport(
        total_turns=len(per_turn),
        consistent_turns=sum(1 for c in per_turn if c.consistent),
        violation_count=sum(len(c.violations) for c in per_turn),
        per_turn=per_turn,
    )


# ── Synthetic fixtures (TF-017's world; for TF-096 / TF-122) ──────────────────

#: SYNTHETIC fixture expectation — Wrenna the Lantern-Keeper,
#: mirroring tests/fixtures/synthetic_personality_core.yaml field
#: for field. Invented; no canon. Defined here so production-side
#: consumers (TF-096's salvage fixtures) import ground truth from
#: the cognition layer, not from the test tree.
FIXTURE_EXPECTATION = PersonaExpectation(
    record_id="syn_wrenna_001",
    traits=["steady", "observant", "plain-spoken"],
    speech_style=[
        "short, concrete sentences",
        "names the river level before giving advice",
        "no filler",
    ],
    behavior_rules=[
        "answer what was asked",
        "cite the synthetic reference when using it",
        "say plainly when the fixture store is silent",
    ],
    avoid=[
        "invented crossings after the second tide bell",
        "vague handwaving",
        "repetition",
    ],
    weights={
        "directness": 0.8,
        "warmth": 0.5,
        "depth": 0.6,
        "wit": 0.2,
        "formality": 0.3,
    },
)

#: (response text, expected consistent) ground-truth pairs against
#: FIXTURE_EXPECTATION. The three out-of-voice turns each trip
#: exactly one avoid entry, in fixture order.
PERSONA_FIXTURE_TURNS: tuple[tuple[str, bool], ...] = (
    (
        "The ferry crossing takes eleven minutes in calm water. "
        "Wrenna records the river level each morning in chalk on "
        "the mill door.",
        True,
    ),
    (
        "After the second tide bell the ferry stops running. "
        "No crossing is offered until morning, whatever the weather.",
        True,
    ),
    (
        "There may be invented crossings after the second tide bell "
        "if the night looks calm enough.",
        False,
    ),
    (
        "It is hard to say — really just vague handwaving on my "
        "part, but the river does what it does.",
        False,
    ),
    (
        "The ferry, the ferry — repetition aside, it takes eleven "
        "minutes in calm water.",
        False,
    ),
)
