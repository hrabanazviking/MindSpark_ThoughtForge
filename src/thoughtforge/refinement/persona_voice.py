"""TF-096 — Persona-voice preservation under
salvage: refinement must not sand a character's
voice flat, specified as data and enforced in
``refinement/salvage.py``'s selection and refine
adoption rules.

Roadmap: "Persona-voice preservation under
salvage: refinement must not sand a character's
voice flat (fixtures from TF-030)."

Measured start (probed live this tick, every leg
run through the real ``FragmentSalvage`` with a
stub engine and the real ``PromptBuilder``):

- **The no-engine paths cannot rewrite.** Selection
  returns a candidate verbatim and extraction
  returns verbatim source sentences (TF-087) — but
  *which* candidate selection returns is a voice
  question the score cannot see: the composite
  (0.45*length + 0.55*citation, TF-087) rewards
  length, and Wrenna's filed voice (TF-030's
  ``FIXTURE_EXPECTATION``) is *short, concrete
  sentences*. Probed: handed the in-voice cited
  draft (144 chars, no violations) beside a longer
  cited rival that trips ``vague handwaving``
  (337 chars), selection returned the rival —
  the in-voice draft was at hand and salvage
  chose the flattened one.
- **The refine path could flatten, and did.** Its
  adoption rule was score + TF-088's fabrication
  guard, and neither can see voice: three
  stub-engine outputs built on TF-030's three
  out-of-voice fixture turns — each tripping
  exactly one of the persona's ``avoid`` entries
  (``invented crossings after the second tide
  bell``, ``vague handwaving``, ``repetition``),
  each padded longer than the in-voice draft —
  were ALL adopted as ``refine_pass_1`` over the
  in-voice cited draft. A benign in-voice refine
  output adopted too (the control leg). Filed as
  D-SALV-3 and FIXED here.
- **The refine prompt is a contributing shape,
  filed not fixed here.** TF-030's persona block
  (traits / speech style / behavior) rides the
  scaffold's ``fact_block``, and
  ``PromptBuilder.build_refine_prompt`` never
  reads ``fact_block`` — the refine generation is
  steered by tone / goal / ``scaffold.avoid``
  (which in core flow does carry the persona's
  avoids, TF-023/TF-030) but not by the persona
  block itself. The prompt's assembly is TF-023's
  frozen contract; this slice enforces at the
  adoption gate, where a flattened output can be
  *refused*, whatever the prompt asked for.
- **The checkable subset is TF-030's.** Traits,
  speech style and behavior rules are steering,
  not scoring (TF-030's ruling: scoring a
  response's fidelity to "plain-spoken" is
  judgement, owned by TF-122's suite). What a
  deterministic guard can prove is the persona's
  ``avoid`` list, checked by TF-030's own
  ``check_persona_response`` — so voice breakage
  here means exactly: *salvage introduced an
  avoid violation the source did not carry.*
  A flattening that trips no avoid entry (a
  longer, blander rephrase in common words) is
  the filed residual, kin to TF-088's: not
  mechanically decidable, measured by TF-122's
  suite and TF-104's gate, never claimed here.

What this slice builds:

- ``persona_violations`` / ``voice_problems`` /
  ``check_voice`` in this module — the relative
  judgement (TF-088's pool-relative shape): a
  violation already in the source is the source
  draft's own (generation's judgement), never
  counted as introduced by salvage for passing
  it through or rephrasing within it.
- **The guard, wired into salvage.** When the
  caller supplies a ``PersonaExpectation``
  (``FragmentSalvage.forge(...,
  persona_expectation=...)``; the core passes
  its own TF-030 expectation in ``_compose_final``):
  selection ranks only persona-consistent drafts
  when at least one is at hand (score order kept
  within them; when none is consistent, score
  order over all stands — salvage never invents
  an in-voice draft), and a refine pass is
  adopted only when it introduces no new avoid
  violation against the current best text (the
  attempt still counts, TF-087 naming; a later
  clean pass may still adopt). With no persona
  supplied, every leg is byte-identical to
  before this slice — the guard is inert, and
  the filed fixtures prove both directions.

Proof grade FIXTURE — stub-engine outputs and
candidate sets run through the real
FragmentSalvage on this VM, judged by TF-030's
real checker against TF-030's filed expectation.
A live model's voice drift is NOT PROVEN (no
model in tree); TF-122's persona suite owns the
full judgement (identity facts, voice, values).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Imported lazily at call time below: a module-level
    # import would cycle (refinement → cognition.persona →
    # cognition/__init__ → core → refinement.salvage).
    from thoughtforge.cognition.persona import PersonaExpectation

VOICE_FORMAT = "thoughtforge-persona-voice/1"

VOICE_DOC = "docs/architecture/PERSONA_VOICE_PRESERVATION.md"

# ── The rules ───────────────────────────────────────────────────────────────

VOICE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "voice_is_judged_by_tf030s_checkable_subset",
     "rule": "voice breakage is judged only by the persona's "
             "avoid list, through TF-030's check_persona_response "
             "— traits, speech style and behavior rules are "
             "steering, not scoring (TF-030's ruling; TF-122 "
             "owns scoring fidelity to them)"},
    {"id": "the_judgement_is_relative_to_the_source",
     "rule": "a violation is salvage's only when the output "
             "carries it and the source text it came from does "
             "not — a violation already in a draft is the "
             "draft's own (generation's judgement), never "
             "counted against salvage for passing it through"},
    {"id": "selection_prefers_an_in_voice_draft_at_hand",
     "rule": "when a persona is supplied and at least one "
             "candidate is persona-consistent, selection ranks "
             "only the consistent drafts (score order kept "
             "within them) — salvage must not choose a "
             "flattened rival over an in-voice draft at hand"},
    {"id": "selection_never_invents_a_voice",
     "rule": "when no candidate is persona-consistent, score "
             "order over all candidates stands — the guard "
             "restricts choice among what generation offered; "
             "it writes no in-voice text of its own"},
    {"id": "a_refine_pass_introducing_a_violation_is_rejected",
     "rule": "a refine pass whose output introduces a new "
             "avoid violation against the current best text is "
             "never adopted, however it scores; the attempt "
             "still counts in passes_used / salvage_path "
             "(TF-087 naming) and a later clean pass may "
             "still adopt"},
    {"id": "no_persona_no_guard",
     "rule": "with no persona supplied, selection and refine "
             "adoption are byte-identical to before this "
             "slice — the guard judges against a persona in "
             "force, and with none in force there is nothing "
             "to preserve (TF-030: consistency with nothing "
             "is a category error, so the check functions "
             "raise ValueError rather than pass silently)"},
    {"id": "verbatim_is_preservation",
     "rule": "an output byte-identical to its source is "
             "preserved by construction — the no-engine "
             "paths rewrite nothing, so their only voice "
             "question is selection, which the rules above "
             "govern"},
    {"id": "flattening_without_a_violation_is_the_residual",
     "rule": "a longer, blander rephrase that trips no avoid "
             "entry is NOT mechanically decidable against "
             "legitimate rephrasing; this guard does not claim "
             "it — TF-122's persona suite scores voice "
             "fidelity and TF-104's gate owns the estate"},
    {"id": "the_surfaces_are_unchanged",
     "rule": "TF-030's checker, TF-088's fabrication guard, "
             "every Part V gate (TF-090–TF-095) and the "
             "scoring formula are unchanged; this slice adds "
             "the persona guard beside TF-088's in the "
             "adoption rule and the persona filter in "
             "selection, both active only when a persona "
             "is supplied"},
)

# ── The judgement ───────────────────────────────────────────────────────────


def persona_violations(
    expectation: "PersonaExpectation | None", text: str
) -> list[str]:
    """The persona's avoid entries present in ``text``,
    by TF-030's own checker. Raises ``ValueError``
    when no persona is in force."""
    from thoughtforge.cognition.persona import check_persona_response

    return list(check_persona_response(expectation, text).violations)


def voice_problems(
    expectation: "PersonaExpectation | None",
    source_text: str,
    output_text: str,
) -> list[str]:
    """The avoid violations ``output_text`` carries
    that ``source_text`` does not — the violations
    the step from source to output *introduced*.
    An empty list means the step broke no voice
    TF-030 can check. Raises ``ValueError`` when
    no persona is in force."""
    if expectation is None:
        raise ValueError("no persona in force: nothing to check against")
    source = set(persona_violations(expectation, source_text))
    return [
        entry for entry in persona_violations(expectation, output_text)
        if entry not in source
    ]


@dataclass
class VoiceCheck:
    """One source → output step, judged for voice."""
    verdict: str                    # preserved / verbatim_pass_through /
                                    # already_out_of_voice / voice_broken
    source_violations: tuple[str, ...]
    output_violations: tuple[str, ...]
    new_violations: tuple[str, ...]


def check_voice(
    expectation: "PersonaExpectation | None",
    source_text: str,
    output_text: str,
) -> VoiceCheck:
    """Judge one step: did it sand the voice flat,
    in the subset TF-030 can check?"""
    if expectation is None:
        raise ValueError("no persona in force: nothing to check against")
    source_v = tuple(persona_violations(expectation, source_text))
    output_v = tuple(persona_violations(expectation, output_text))
    new_v = tuple(
        entry for entry in output_v if entry not in set(source_v))
    if output_text == source_text:
        verdict = "verbatim_pass_through"
    elif new_v:
        verdict = "voice_broken"
    elif output_v:
        verdict = "already_out_of_voice"
    else:
        verdict = "preserved"
    return VoiceCheck(
        verdict=verdict, source_violations=source_v,
        output_violations=output_v, new_violations=new_v)


# ── Fixtures ────────────────────────────────────────────────────────────────
# Synthetic throughout: TF-030's filed expectation
# (Wrenna the Lantern-Keeper, TF-017's world, never
# canon) and texts built on TF-030's fixture turns —
# the in-voice turn as the candidate (with its Q123
# citation, so fragments clear salvage's keep
# threshold and the refine legs actually run), and
# each out-of-voice turn padded longer as a stub
# engine's flattened refine output (the padding is
# what the length-weighted score rewards).

_IN_VOICE = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water. Wrenna records the river level "
    "each morning in chalk on the mill door.")
_FLAT_HANDWAVING = (
    "According to Q123, it is hard to say — really just "
    "vague handwaving on my part, but the river does what "
    "it does, and the ferry crossing takes eleven minutes "
    "in calm water while Wrenna records the river level "
    "each morning in chalk on the mill door, and travellers "
    "should simply wait and see what the morning brings "
    "for everyone who waits.")
_FLAT_REPETITION = (
    "According to Q123, the ferry, the ferry — repetition "
    "aside, it takes eleven minutes in calm water, and "
    "Wrenna records the river level each morning in chalk "
    "on the mill door, and the crossing is a pleasant "
    "crossing in calm water for every traveller who comes "
    "down to the river in the morning light today at the "
    "landing.")
_FLAT_CROSSINGS = (
    "According to Q123, there may be invented crossings "
    "after the second tide bell if the night looks calm "
    "enough, and the ferry crossing takes eleven minutes "
    "in calm water while Wrenna records the river level "
    "each morning in chalk on the mill door for every "
    "traveller who waits by the river in the soft "
    "morning light.")
_BENIGN = (
    "According to Q123, the ferry crossing takes eleven "
    "minutes in calm water. Each morning Wrenna records "
    "the river level in chalk on the mill door, and after "
    "the second tide bell the ferry stops running until "
    "morning, whatever the weather.")
_SAME_VIOLATION_REFINE = (
    "According to Q123, the ferry, the ferry — repetition "
    "aside, it takes eleven minutes in calm water, and "
    "Wrenna records the river level each morning in chalk "
    "on the mill door, and the crossing is a pleasant "
    "crossing in calm water for every traveller who comes "
    "down to the river in the morning light today at the "
    "landing, where the boats wait in a row by the "
    "landing steps.")

VOICE_FIXTURES: tuple[dict[str, Any], ...] = (
    {"id": "flattened_handwaving_rejected",
     "kind": "refine",
     "candidates": (_IN_VOICE,),
     "engine_text": _FLAT_HANDWAVING,
     "with_expectation": True,
     "expected_text": _IN_VOICE,
     "expected_path": "refine_pass_1",
     "expected_new_violations": ("vague handwaving",)},
    {"id": "flattened_repetition_rejected",
     "kind": "refine",
     "candidates": (_IN_VOICE,),
     "engine_text": _FLAT_REPETITION,
     "with_expectation": True,
     "expected_text": _IN_VOICE,
     "expected_path": "refine_pass_1",
     "expected_new_violations": ("repetition",)},
    {"id": "flattened_crossings_rejected",
     "kind": "refine",
     "candidates": (_IN_VOICE,),
     "engine_text": _FLAT_CROSSINGS,
     "with_expectation": True,
     "expected_text": _IN_VOICE,
     "expected_path": "refine_pass_1",
     "expected_new_violations": (
         "invented crossings after the second tide bell",)},
    {"id": "benign_in_voice_refine_adopted",
     "kind": "refine",
     "candidates": (_IN_VOICE,),
     "engine_text": _BENIGN,
     "with_expectation": True,
     "expected_text": _BENIGN,
     "expected_path": "refine_pass_1",
     "expected_new_violations": ()},
    {"id": "same_violation_refine_adopted",
     "kind": "refine",
     "candidates": (_FLAT_REPETITION,),
     "engine_text": _SAME_VIOLATION_REFINE,
     "with_expectation": True,
     "expected_text": _SAME_VIOLATION_REFINE,
     "expected_path": "refine_pass_1",
     "expected_new_violations": (),
     "note": "the candidate already trips repetition; the "
             "refine output trips the same entry and no "
             "other — salvage introduced nothing, so the "
             "score decides (the relative judgement)"},
    {"id": "flattened_adopted_without_a_persona",
     "kind": "refine",
     "candidates": (_IN_VOICE,),
     "engine_text": _FLAT_HANDWAVING,
     "with_expectation": False,
     "expected_text": _FLAT_HANDWAVING,
     "expected_path": "refine_pass_1",
     "expected_new_violations": ("vague handwaving",),
     "note": "the contrast leg: with no persona supplied "
             "the guard is inert and the pre-TF-096 "
             "adoption stands, byte-identical"},
    {"id": "selection_prefers_the_in_voice_draft",
     "kind": "selection",
     "candidates": (_FLAT_HANDWAVING, _IN_VOICE),
     "engine_text": None,
     "with_expectation": True,
     "expected_text": _IN_VOICE,
     "expected_path": "best_draft",
     "expected_new_violations": ()},
    {"id": "selection_without_a_persona_picks_the_longer",
     "kind": "selection",
     "candidates": (_FLAT_HANDWAVING, _IN_VOICE),
     "engine_text": None,
     "with_expectation": False,
     "expected_text": _FLAT_HANDWAVING,
     "expected_path": "best_draft",
     "expected_new_violations": (),
     "note": "the contrast leg: score-only selection "
             "returns the flattened rival, as probed"},
    {"id": "selection_all_out_of_voice_falls_back",
     "kind": "selection",
     "candidates": (_FLAT_REPETITION, _FLAT_HANDWAVING),
     "engine_text": None,
     "with_expectation": True,
     "expected_text": _FLAT_HANDWAVING,
     "expected_path": "best_draft",
     "expected_new_violations": (),
     "note": "no consistent candidate at hand: score "
             "order over all stands (the longer cited "
             "draft) — the guard invents no voice"},
    {"id": "out_of_voice_pass_through_verbatim",
     "kind": "selection",
     "candidates": (_FLAT_HANDWAVING,),
     "engine_text": None,
     "with_expectation": True,
     "expected_text": _FLAT_HANDWAVING,
     "expected_path": "best_draft",
     "expected_new_violations": (),
     "note": "a lone out-of-voice draft passes through "
             "verbatim — its violation is generation's, "
             "not salvage's"},
)

_FIXTURE_BY_ID = {f["id"]: f for f in VOICE_FIXTURES}


class _StubEngine:
    def __init__(self, text: str):
        self._text = text

    def generate(self, prompt, params=None):
        from types import SimpleNamespace
        return SimpleNamespace(text=self._text)


def run_voice_fixture(fixture_id: str):
    """Run one filed fixture through the real
    ``FragmentSalvage`` (stub engine + the real
    ``PromptBuilder`` for refine fixtures; no engine
    for selection fixtures) and return its
    SalvageResult."""
    from thoughtforge.cognition.persona import FIXTURE_EXPECTATION
    from thoughtforge.cognition.prompt_builder import PromptBuilder
    from thoughtforge.knowledge.models import (
        ActivatedRecord,
        CandidateRecord,
        CandidateScores,
        CognitionScaffold,
        InputSketch,
        MemoryActivationBundle,
    )
    from thoughtforge.refinement.salvage import FragmentSalvage

    fixture = _FIXTURE_BY_ID[fixture_id]
    candidates = [
        CandidateRecord(
            candidate_id=f"cand_{i}", mode="practical", text=text,
            token_estimate=len(text.split()), scores=CandidateScores())
        for i, text in enumerate(fixture["candidates"])
    ]
    bundle = MemoryActivationBundle(
        activated_records=[ActivatedRecord(
            "Q123", "entity", 0.8, "cue for Q123",
            raw={"qid": "Q123"})],
        total_retrieved=1, retrieval_confidence=0.8)
    expectation = FIXTURE_EXPECTATION if fixture[
        "with_expectation"] else None
    if fixture["kind"] == "selection":
        return FragmentSalvage().forge(
            candidates, bundle, persona_expectation=expectation)
    sketch = InputSketch(
        raw_text="when does the ferry leave?",
        intent="factual_query", topic="ferry",
        response_mode="factual_direct")
    scaffold = CognitionScaffold(
        goal="give a clear direct answer",
        tone=["direct", "factual"], avoid=["filler"])
    return FragmentSalvage().forge(
        candidates, bundle, engine=_StubEngine(
            fixture["engine_text"]),
        prompt_builder=PromptBuilder(), sketch=sketch,
        scaffold=scaffold, max_passes=1,
        persona_expectation=expectation)


# ── Checkers ────────────────────────────────────────────────────────────────


def persona_voice_problems() -> list[str]:
    """Data-integrity checker: rule ids unique,
    fixtures well-formed with known violation
    entries, and the guard actually wired into
    salvage.py (selection + the adoption rule,
    source anchors)."""
    import inspect

    problems: list[str] = []
    rule_ids = [r["id"] for r in VOICE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate VOICE_RULES ids")
    fixture_ids = [f["id"] for f in VOICE_FIXTURES]
    if len(fixture_ids) != len(set(fixture_ids)):
        problems.append("duplicate VOICE_FIXTURES ids")
    from thoughtforge.cognition.persona import FIXTURE_EXPECTATION

    for fixture in VOICE_FIXTURES:
        if fixture["kind"] not in ("refine", "selection"):
            problems.append(
                f"fixture {fixture['id']}: unknown kind "
                f"{fixture['kind']!r}")
        if (fixture["kind"] == "refine") != (
                fixture["engine_text"] is not None):
            problems.append(
                f"fixture {fixture['id']}: kind/engine_text "
                "disagree")
        for entry in fixture["expected_new_violations"]:
            if entry not in FIXTURE_EXPECTATION.avoid:
                problems.append(
                    f"fixture {fixture['id']}: expected "
                    f"violation {entry!r} is not a filed "
                    "avoid entry")
    from thoughtforge.refinement import salvage as salvage_module
    source = inspect.getsource(salvage_module)
    if "persona_expectation" not in source:
        problems.append(
            "salvage.py takes no persona_expectation — "
            "the guard is not wired")
    if "voice_problems" not in source:
        problems.append(
            "salvage.py does not call voice_problems — "
            "the adoption guard is not wired")
    return problems


def persona_voice_live_problems() -> list[str]:
    """Live checker: every filed fixture is run
    through the real FragmentSalvage and must land
    exactly as filed — flattened refine outputs
    rejected with the candidate standing (the
    attempt still counted), the benign and
    same-violation legs adopted, selection
    preferring the in-voice draft only when a
    persona is supplied, the fallbacks verbatim —
    and each refine fixture's new violations,
    judged against its best candidate, must match
    the filing."""
    from thoughtforge.cognition.persona import FIXTURE_EXPECTATION

    problems: list[str] = []
    for fixture in VOICE_FIXTURES:
        result = run_voice_fixture(fixture["id"])
        if result.text != fixture["expected_text"]:
            problems.append(
                f"fixture {fixture['id']}: text differs "
                "from filed")
        if result.salvage_path != fixture["expected_path"]:
            problems.append(
                f"fixture {fixture['id']}: path "
                f"{result.salvage_path} != "
                f"{fixture['expected_path']}")
        if fixture["kind"] == "refine":
            new = voice_problems(
                FIXTURE_EXPECTATION, fixture["candidates"][0],
                fixture["engine_text"])
            if tuple(new) != tuple(
                    fixture["expected_new_violations"]):
                problems.append(
                    f"fixture {fixture['id']}: new "
                    f"violations {new} != filed "
                    f"{fixture['expected_new_violations']}")
            if fixture["with_expectation"] and new and \
                    result.text == fixture["engine_text"]:
                problems.append(
                    f"fixture {fixture['id']}: a flattened "
                    "refine output was adopted under a persona")
    return problems


def persona_voice_as_data() -> dict[str, Any]:
    return {
        "format": VOICE_FORMAT,
        "voice_doc": VOICE_DOC,
        "rules": [dict(r) for r in VOICE_RULES],
        "fixtures": [
            {k: v for k, v in f.items()} for f in VOICE_FIXTURES],
        "proof": {
            "grade": "fixture",
            "live_model_voice_drift": "not_proven",
            "judged_subset": "persona_avoid_list_via_tf030",
            "flattening_without_a_violation": "residual_not_judged",
            "fragment_salvage": "guarded_when_persona_supplied",
        },
    }
