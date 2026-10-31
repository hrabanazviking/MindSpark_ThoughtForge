"""
ScaffoldRegistry — the one registry of ThoughtForge's existing scaffolds
(TF-021, Scaffold Library I).

Before this module the scaffold estate lived scattered across four
modules: the intent tables in ``scaffold.py``, the response-mode table
in ``router.py``, the opener / output-shape tables in
``prompt_builder.py``, and the mode temperatures in ``core.py``.
This registry assembles those live tables — it copies none of their
values — into one catalogue:

  * ``scaffold_shape()``       — the CognitionScaffold object's fields.
  * ``scaffold_registry()``    — one entry per router intent (plus the
    unknown-intent fallback): goal, tone, focus, avoid, depth,
    candidate modes, response mode, output shape, length hint.
  * ``candidate_mode_registry()`` — one entry per candidate mode:
    opener, temperature, purpose, and the intents that use it.
  * ``measure_prompt_costs()`` — each entry's candidate-prompt cost,
    measured by actually building the prompt for a canonical sketch
    (fixed synthetic request, empty memory bundle, no cues) and
    counting words — the tree's own token convention
    (``core`` counts a final's tokens as ``len(text.split())``).

The human-readable catalogue rendered from this registry is
``docs/architecture/SCAFFOLD_REGISTRY.md``. TF-022 brings new
scaffolds (chain-of-thought, plan-then-answer, self-check) to the
interface this registry describes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from thoughtforge.cognition.core import _mode_temperature
from thoughtforge.cognition.prompt_builder import (
    _CANDIDATE_OPENERS,
    _DEPTH_TO_LENGTH_HINT,
    _MODE_TO_OUTPUT_SHAPE,
    PromptBuilder,
)
from thoughtforge.cognition.router import _INTENT_TO_MODE
from thoughtforge.cognition.scaffold import (
    _EXPERT_INTENTS,
    _LIGHT_INTENTS,
    ScaffoldBuilder,
)
from thoughtforge.knowledge.models import (
    CognitionScaffold,
    InputSketch,
    MemoryActivationBundle,
)

# ── The canonical measurement sketch ──────────────────────────────────────────
# Cost is only comparable across scaffolds when the request is held
# fixed: a two-word synthetic request, neutral tone, mid urgency, an
# empty memory bundle and no memory cues. Anything a real turn adds
# (the user's own words, up to 3 cues, the fact block) rides on top of
# the base cost measured here.

CANONICAL_RAW_TEXT = "test query"
CANONICAL_TOPIC = "test topic"
CANONICAL_URGENCY = 0.4

FALLBACK_INTENT = "fallback"


def canonical_sketch(intent: str) -> InputSketch:
    """The fixed synthetic sketch a registry cost is measured on."""
    return InputSketch(
        raw_text=CANONICAL_RAW_TEXT,
        intent=intent,
        topic=CANONICAL_TOPIC,
        tone_in="neutral",
        response_mode=_INTENT_TO_MODE.get(intent, "light_natural_chat"),
        urgency=CANONICAL_URGENCY,
    )


# ── Candidate-mode purposes (catalogue metadata) ─────────────────────────────
# The one thing the scattered tables never wrote down: what each mode
# is *for*. Grounded in each mode's opener, its temperature, and the
# intents that call it — not new behaviour.

_MODE_PURPOSES: dict[str, str] = {
    "strict_spec": "Precision pass: one concise, structured, implementation-ready answer; the coldest temperature in the registry.",
    "implementation_friendly": "Builder's pass: a clear response a developer can act on directly; near-coldest, slightly warmer than strict_spec.",
    "editorial_sharp": "Editor's pass: rewrite the user's text cleanly and precisely, preserving meaning.",
    "brainstorm_distinct": "Divergence pass: one strong, distinctive option; the hottest temperature in the registry, built for originality over safety.",
    "practical": "Grounding pass: one direct, practical response — the counterweight paired with divergent or formal first passes.",
    "empathic": "Support pass: one short, warm, grounded reply that validates before it advises.",
    "reflective": "Depth pass: one thoughtful, perceptive reply — emotional support's second, slower angle.",
    "playful": "Lightness pass: one light, natural reply with warmth — light conversation's second angle.",
}


# ── Registry entries ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandidateModeEntry:
    """One candidate mode: its steering text, heat, purpose, users."""

    name: str
    opener: str
    temperature: float
    purpose: str
    used_by: tuple[str, ...]


@dataclass(frozen=True)
class ScaffoldEntry:
    """One intent's scaffold, as the builder actually builds it."""

    intent: str
    response_mode: str
    goal: str
    tone: tuple[str, ...]
    focus: tuple[str, ...]
    avoid: tuple[str, ...]
    depth: str
    depth_rule: str
    candidate_modes: tuple[str, ...]
    output_shape: str
    length_hint: str


@dataclass(frozen=True)
class PromptCost:
    """One candidate prompt's measured cost on the canonical sketch."""

    intent: str
    mode: str
    chars: int
    word_tokens: int


# ── Shape ─────────────────────────────────────────────────────────────────────

def scaffold_shape() -> tuple[str, ...]:
    """The CognitionScaffold field names, in declaration order."""
    return tuple(f.name for f in dataclasses.fields(CognitionScaffold))


# ── Candidate modes ───────────────────────────────────────────────────────────

def candidate_mode_registry() -> dict[str, CandidateModeEntry]:
    """Every candidate mode in the tree, keyed by mode name."""
    scaffolds = scaffold_registry()
    used_by: dict[str, list[str]] = {name: [] for name in _CANDIDATE_OPENERS}
    for entry in scaffolds.values():
        for mode in entry.candidate_modes:
            used_by.setdefault(mode, []).append(entry.intent)
    return {
        name: CandidateModeEntry(
            name=name,
            opener=opener,
            temperature=_mode_temperature(name),
            purpose=_MODE_PURPOSES.get(name, ""),
            used_by=tuple(used_by.get(name, [])),
        )
        for name, opener in _CANDIDATE_OPENERS.items()
    }


# ── Scaffolds per intent ──────────────────────────────────────────────────────

def _depth_rule(intent: str) -> str:
    if intent in _EXPERT_INTENTS:
        return "expert below urgency 0.8; medium at or above it"
    if intent in _LIGHT_INTENTS:
        return "light at any urgency"
    return "medium at any urgency"


def scaffold_registry() -> dict[str, ScaffoldEntry]:
    """
    One entry per router intent, plus the unknown-intent fallback.

    Every value is read from a scaffold the real ScaffoldBuilder
    builds for the canonical sketch — the registry cannot drift from
    the builder without its tests noticing.
    """
    builder = ScaffoldBuilder()
    entries: dict[str, ScaffoldEntry] = {}
    for intent in [*_INTENT_TO_MODE, FALLBACK_INTENT]:
        sketch = canonical_sketch(intent)
        scaffold = builder.build(sketch, MemoryActivationBundle())
        entries[intent] = ScaffoldEntry(
            intent=intent,
            response_mode=sketch.response_mode,
            goal=scaffold.goal,
            tone=tuple(scaffold.tone),
            focus=tuple(scaffold.focus),
            avoid=tuple(scaffold.avoid),
            depth=scaffold.depth,
            depth_rule=_depth_rule(intent),
            candidate_modes=tuple(scaffold.candidate_modes),
            output_shape=_MODE_TO_OUTPUT_SHAPE.get(
                sketch.response_mode, "2-4 sentences"
            ),
            length_hint=_DEPTH_TO_LENGTH_HINT.get(scaffold.depth, ""),
        )
    return entries


# ── Measured cost ─────────────────────────────────────────────────────────────

def measure_prompt_costs() -> list[PromptCost]:
    """
    Build every registry entry's candidate prompts on the canonical
    sketch (empty bundle, no cues) and count their cost: characters,
    and word-tokens by the tree's own ``len(text.split())`` rule.
    """
    builder = ScaffoldBuilder()
    prompts = PromptBuilder()
    costs: list[PromptCost] = []
    for intent in scaffold_registry():
        sketch = canonical_sketch(intent)
        scaffold = builder.build(sketch, MemoryActivationBundle())
        for mode in scaffold.candidate_modes:
            prompt = prompts.build_candidate_prompt(sketch, scaffold, mode, [])
            costs.append(
                PromptCost(
                    intent=intent,
                    mode=mode,
                    chars=len(prompt),
                    word_tokens=len(prompt.split()),
                )
            )
    return costs


__all__ = [
    "CANONICAL_RAW_TEXT",
    "CANONICAL_TOPIC",
    "CANONICAL_URGENCY",
    "FALLBACK_INTENT",
    "CandidateModeEntry",
    "PromptCost",
    "ScaffoldEntry",
    "candidate_mode_registry",
    "canonical_sketch",
    "measure_prompt_costs",
    "scaffold_registry",
    "scaffold_shape",
]
