"""
PromptContract — the PromptBuilder assembly contract as data (TF-023).

``prompt_builder.py`` builds six kinds of prompt. Before this slice
their assembly order lived only inside the f-strings: nobody could
say, from outside the builder, what order the sections run in, which
sections are optional, or what each cap is. This module states the
contract as data, and the golden tests
(``tests/test_tf023_prompt_builder_contract.py``) hold the builder
to it:

  * ``PROMPT_SECTIONS`` — per prompt kind, the sections in assembly
    order. The roadmap's TF-023 shorthand names the ingredients
    (persona, memory, scaffold, context); §2 of the contract doc
    (``docs/architecture/PROMPT_BUILDER_CONTRACT.md``) maps those
    ingredients onto these sections — persona reaches a prompt only
    pre-digested into the scaffold (tone overlay + avoid entries),
    memory rides as cues + the knowledge-context fact block.
  * ``OPTIONAL_SECTIONS`` — sections a prompt of that kind may omit
    (knowledge context exists only when the scaffold carries a fact
    block; working-so-far only on prior-consuming strategy steps).
  * ``SECTION_MARKERS`` — the literal header each shared section
    opens with, so ``check_prompt_order`` can verify a built prompt
    section by section.
  * ``check_prompt_order(prompt, kind)`` — returns the list of
    contract violations in a built prompt (missing required section,
    or sections out of order); an empty list means the prompt
    honours the contract's order. The opening instruction/opener
    section carries no shared marker (its text is per-mode), so
    order checking starts at the first marked section — the opener
    itself is pinned by the golden tests.
"""

from __future__ import annotations

# ── Section order per prompt kind ─────────────────────────────────────────────

PROMPT_SECTIONS: dict[str, tuple[str, ...]] = {
    "candidate": (
        "opener", "user_request", "intent", "topic", "tone", "goal",
        "focus", "avoid", "memory_cues", "knowledge_context", "output",
    ),
    "strategy": (
        "instruction", "user_request", "intent", "topic", "tone",
        "goal", "focus", "avoid", "memory_cues", "knowledge_context",
        "working_so_far", "output",
    ),
    "refine": (
        "fragments", "instruction", "intent", "tone", "goal", "avoid",
        "output",
    ),
    "repair": (
        "opener", "user_request", "intent", "tone", "goal",
        "memory_cues", "directive", "output",
    ),
    "episode_summary": (
        "instruction", "user_request", "final_response", "output",
    ),
    "preference_inference": (
        "instruction", "user_request", "context", "output",
    ),
}

OPTIONAL_SECTIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"knowledge_context"}),
    "strategy": frozenset({"knowledge_context", "working_so_far"}),
    "refine": frozenset(),
    "repair": frozenset(),
    "episode_summary": frozenset(),
    "preference_inference": frozenset(),
}

# ── Section markers ───────────────────────────────────────────────────────────
# The literal text a section opens with in a built prompt. "opener" /
# "instruction" for the candidate/strategy kinds are per-mode /
# per-step text, so they have no single marker; every other section
# does. Refine's fragments and repair's opener do have fixed text.

SECTION_MARKERS: dict[str, str] = {
    "user_request": "User request:",
    "intent": "Intent:",
    "topic": "Topic:",
    "tone": "Tone:",
    "goal": "Goal:",
    "focus": "Focus:",
    "avoid": "Avoid:",
    "memory_cues": "Memory cues:",
    "knowledge_context": "Knowledge context:",
    "working_so_far": "Working so far:",
    "output": "Output:",
    "final_response": "Final response:",
    "context": "Context:",
}

_KIND_MARKERS: dict[str, dict[str, str]] = {
    "refine": {"fragments": "Using these useful fragments:"},
    "repair": {"directive": "Avoid generic wording."},
}


def section_markers(kind: str) -> dict[str, str]:
    """The marker for each checkable section of ``kind``, in order."""
    markers = {
        section: SECTION_MARKERS[section]
        for section in PROMPT_SECTIONS[kind]
        if section in SECTION_MARKERS
    }
    markers.update(_KIND_MARKERS.get(kind, {}))
    # Re-order to the kind's section order (updates may append).
    return {
        section: markers[section]
        for section in PROMPT_SECTIONS[kind]
        if section in markers
    }


def check_prompt_order(prompt: str, kind: str) -> list[str]:
    """
    Verify a built prompt against the contract's section order.

    Returns a list of violations: a required section whose marker is
    absent, or a section whose marker appears before the previous
    section's. Optional sections are checked only when present.
    """
    violations: list[str] = []
    optional = OPTIONAL_SECTIONS[kind]
    position = -1
    for section, marker in section_markers(kind).items():
        found = prompt.find(marker)
        if found == -1:
            if section not in optional:
                violations.append(f"{kind}: missing required section {section!r}")
            continue
        if found < position:
            violations.append(f"{kind}: section {section!r} is out of order")
        position = max(position, found)
    return violations


__all__ = [
    "OPTIONAL_SECTIONS",
    "PROMPT_SECTIONS",
    "SECTION_MARKERS",
    "check_prompt_order",
    "section_markers",
]
