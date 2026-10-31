"""
TF-021 — Scaffold Library I: the registry of existing scaffolds.

The catalogue itself is docs/architecture/SCAFFOLD_REGISTRY.md; the
machine-readable registry lives in
``src/thoughtforge/cognition/scaffold_registry.py`` and assembles the
live tables scattered across scaffold.py / router.py /
prompt_builder.py / core.py. These tests pin:

  * the shape (CognitionScaffold's fields, in order),
  * one registry entry per router intent plus the fallback, with every
    value matching a scaffold the real ScaffoldBuilder builds,
  * the TF-021 ruling on CORE_LOOP §8-F8 — no registry row repeats a
    candidate mode; brainstorming pairs brainstorm_distinct with
    practical; the unknown-intent fallback is a single practical pass,
  * the candidate-mode registry (all 8 modes: opener and temperature
    read from the live tables, used_by reverse index exact),
  * the measured prompt costs — recomputed independently through
    PromptBuilder on the canonical sketch, word-tokens counted by the
    tree's own len(text.split()) rule,
  * the catalogue doc naming every intent and every mode.
"""

from pathlib import Path

import pytest

from thoughtforge.cognition.core import _mode_temperature
from thoughtforge.cognition.prompt_builder import (
    _CANDIDATE_OPENERS,
    PromptBuilder,
)
from thoughtforge.cognition.router import _INTENT_TO_MODE
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.scaffold_registry import (
    FALLBACK_INTENT,
    candidate_mode_registry,
    canonical_sketch,
    measure_prompt_costs,
    scaffold_registry,
    scaffold_shape,
)
from thoughtforge.knowledge.models import (
    CognitionScaffold,
    MemoryActivationBundle,
)

import dataclasses


@pytest.fixture()
def registry():
    return scaffold_registry()


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_shape_is_cognition_scaffold_fields_in_order():
    expected = tuple(f.name for f in dataclasses.fields(CognitionScaffold))
    assert scaffold_shape() == expected
    assert scaffold_shape() == (
        "goal", "tone", "focus", "avoid", "depth",
        "candidate_modes", "fact_block",
    )
    # TF-023 removed "prompt_text" (CORE_LOOP §8-F7): a turn builds
    # many prompts and the builder returns each to its caller; no
    # single scaffold field can honestly hold "the" prompt.


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_registry_covers_every_router_intent_plus_fallback(registry):
    assert set(registry) == set(_INTENT_TO_MODE) | {FALLBACK_INTENT}
    assert len(registry) == 10


def test_every_entry_matches_the_live_builder(registry):
    builder = ScaffoldBuilder()
    for intent, entry in registry.items():
        sketch = canonical_sketch(intent)
        built = builder.build(sketch, MemoryActivationBundle())
        assert entry.response_mode == sketch.response_mode
        assert entry.goal == built.goal
        assert entry.tone == tuple(built.tone)
        assert entry.focus == tuple(built.focus)
        assert entry.avoid == tuple(built.avoid)
        assert entry.depth == built.depth
        assert entry.candidate_modes == tuple(built.candidate_modes)


def test_fallback_entry_is_the_builders_unknown_intent_shape(registry):
    entry = registry[FALLBACK_INTENT]
    assert entry.goal == "produce a useful, direct response"
    assert entry.focus == ("clarity", "usefulness")
    # Only the universal avoids — no intent extras exist to add.
    assert entry.avoid == (
        "filler", "padding", "repetition of the user's question",
    )


def test_depth_rules_match_builder_at_both_urgencies(registry):
    builder = ScaffoldBuilder()
    for intent, entry in registry.items():
        low = canonical_sketch(intent)
        high = canonical_sketch(intent)
        high.urgency = 0.9
        if "expert below urgency 0.8" in entry.depth_rule:
            assert builder.build(low, MemoryActivationBundle()).depth == "expert"
            assert builder.build(high, MemoryActivationBundle()).depth == "medium"
        elif entry.depth_rule.startswith("light"):
            assert builder.build(high, MemoryActivationBundle()).depth == "light"
        else:
            assert builder.build(high, MemoryActivationBundle()).depth == "medium"


def test_personality_avoids_are_the_only_builder_addition_not_in_registry():
    # The registry catalogues the no-personality shape; CORE_LOOP §4:
    # a personality core appends at most its first 3 avoid entries.
    from thoughtforge.knowledge.models import PersonalityCoreRecord

    builder = ScaffoldBuilder()
    sketch = canonical_sketch("factual_query")
    persona = PersonalityCoreRecord(
        record_id="synthetic", avoid=["taboo-a", "taboo-b", "taboo-c", "taboo-d"],
    )
    built = builder.build(sketch, MemoryActivationBundle(), personality=persona)
    base = list(scaffold_registry()["factual_query"].avoid)
    assert built.avoid == base + ["taboo-a", "taboo-b", "taboo-c"]


# ── The F8 ruling ─────────────────────────────────────────────────────────────

def test_no_registry_row_repeats_a_candidate_mode(registry):
    for entry in registry.values():
        assert len(set(entry.candidate_modes)) == len(entry.candidate_modes), (
            f"{entry.intent} repeats a candidate mode: {entry.candidate_modes}"
        )


def test_brainstorming_pairs_distinct_with_practical(registry):
    assert registry["brainstorming"].candidate_modes == (
        "brainstorm_distinct", "practical",
    )


def test_brainstorming_two_passes_now_build_different_prompts():
    builder = ScaffoldBuilder()
    prompts = PromptBuilder()
    sketch = canonical_sketch("brainstorming")
    scaffold = builder.build(sketch, MemoryActivationBundle())
    built = [
        prompts.build_candidate_prompt(sketch, scaffold, mode, [])
        for mode in scaffold.candidate_modes
    ]
    assert len(built) == 2
    assert built[0] != built[1]
    assert built[0].startswith(_CANDIDATE_OPENERS["brainstorm_distinct"])
    assert built[1].startswith(_CANDIDATE_OPENERS["practical"])


def test_unified_backend_first_mode_unchanged_by_f8_fix(registry):
    # core's unified branch uses candidate_modes[0] only — the ruling
    # changed the second (local) pass, never the first.
    assert registry["brainstorming"].candidate_modes[0] == "brainstorm_distinct"


# ── Candidate modes ───────────────────────────────────────────────────────────

def test_mode_registry_covers_exactly_the_openers_table():
    modes = candidate_mode_registry()
    assert set(modes) == set(_CANDIDATE_OPENERS)
    assert len(modes) == 8


def test_mode_entries_read_opener_and_temperature_from_live_code():
    for name, entry in candidate_mode_registry().items():
        assert entry.opener == _CANDIDATE_OPENERS[name]
        assert entry.temperature == _mode_temperature(name)
        assert entry.purpose != ""


def test_mode_used_by_reverse_index_is_exact(registry):
    modes = candidate_mode_registry()
    for name, entry in modes.items():
        expected = tuple(
            intent for intent, e in registry.items() if name in e.candidate_modes
        )
        assert entry.used_by == expected
    # Every mode is used by at least one scaffold — no orphan modes.
    assert all(entry.used_by for entry in modes.values())


def test_every_scaffold_mode_exists_in_mode_registry(registry):
    modes = candidate_mode_registry()
    for entry in registry.values():
        for mode in entry.candidate_modes:
            assert mode in modes


# ── Measured cost ─────────────────────────────────────────────────────────────

def test_prompt_costs_recomputed_independently():
    builder = ScaffoldBuilder()
    prompts = PromptBuilder()
    measured = {(c.intent, c.mode): c for c in measure_prompt_costs()}
    registry = scaffold_registry()
    # One cost row per (intent, mode) pair the registry carries.
    assert len(measured) == sum(
        len(e.candidate_modes) for e in registry.values()
    )
    for intent, entry in registry.items():
        sketch = canonical_sketch(intent)
        scaffold = builder.build(sketch, MemoryActivationBundle())
        for mode in entry.candidate_modes:
            prompt = prompts.build_candidate_prompt(sketch, scaffold, mode, [])
            cost = measured[(intent, mode)]
            assert cost.chars == len(prompt)
            assert cost.word_tokens == len(prompt.split())
            assert cost.word_tokens > 0


def test_base_costs_stay_inside_the_small_model_band():
    # The catalogue's measured claim: on the canonical sketch every
    # base candidate prompt is a lean 45–60 word-tokens. A scaffold
    # growing past that band must update the catalogue deliberately.
    # TF-023 widened the ceiling 65 → 70 deliberately: candidate
    # prompts now carry the scaffold's FULL avoid list (intent extras
    # included) instead of its first 3 entries — measured range at
    # TF-023 is 45–66 words (see PROMPT_BUILDER_CONTRACT §6).
    for cost in measure_prompt_costs():
        assert 40 <= cost.word_tokens <= 70


# ── The catalogue doc ─────────────────────────────────────────────────────────

def test_catalogue_doc_names_every_intent_and_mode(registry):
    doc = (
        Path(__file__).resolve().parent.parent
        / "docs" / "architecture" / "SCAFFOLD_REGISTRY.md"
    )
    text = doc.read_text(encoding="utf-8")
    for intent in registry:
        assert intent in text
    for mode in candidate_mode_registry():
        assert mode in text
