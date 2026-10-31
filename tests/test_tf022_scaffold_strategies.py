"""
TF-022 — Scaffold Library II: chain-of-thought, plan-then-answer,
and self-check brought to one interface.

The catalogue is docs/architecture/SCAFFOLD_STRATEGIES.md; the
registry lives in ``src/thoughtforge/cognition/scaffold_strategies.py``
and every step prompt is built by the one builder method,
``PromptBuilder.build_strategy_prompt``. These tests pin:

  * exactly the three roadmap strategies, on one step interface
    (same fields, unique step names, temperatures in range),
  * the interface invariants: exactly one final_answer step per
    strategy and it is last; only the first step runs without prior
    output; the builder enforces the prior-output contract both ways,
  * the selection table derived from the registry: six reasoning
    intents mapped, light intents and unknowns get no strategy,
  * the prompt contract: instruction opens the prompt, the shared
    candidate-prompt fields follow, Working-so-far appears only for
    prior-consuming steps, the output line names what a step produces,
  * measured costs recomputed independently, inside the pinned band,
    with self_check the dearest strategy,
  * the F6 ruling (CORE_LOOP §8): the core and salvage fragment
    thresholds are different instruments — both values pinned, and
    the catalogue's measured sentence is kept by one, refused by the
    other,
  * the catalogue doc naming every strategy and every step.
"""

from pathlib import Path

import pytest

from thoughtforge.cognition import core as core_module
from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.router import _INTENT_TO_MODE
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.scaffold_registry import (
    FALLBACK_INTENT,
    canonical_sketch,
)
from thoughtforge.cognition.scaffold_strategies import (
    CANONICAL_PRIOR_TEXT,
    measure_strategy_costs,
    select_strategy,
    strategy_registry,
)
from thoughtforge.knowledge.models import MemoryActivationBundle
from thoughtforge.refinement import salvage as salvage_module
from thoughtforge.refinement.salvage import FragmentSalvage


@pytest.fixture()
def registry():
    return strategy_registry()


@pytest.fixture()
def prompts():
    return PromptBuilder()


def _build(prompts, strategy, step, prior=""):
    sketch = canonical_sketch(strategy.applies_to[0])
    scaffold = ScaffoldBuilder().build(sketch, MemoryActivationBundle())
    return prompts.build_strategy_prompt(sketch, scaffold, step, [], prior_text=prior)


# ── The three strategies, one interface ───────────────────────────────────────

def test_registry_is_exactly_the_three_roadmap_strategies(registry):
    assert set(registry) == {"chain_of_thought", "plan_then_answer", "self_check"}


def test_every_step_uses_the_one_interface(registry):
    for strategy in registry.values():
        assert strategy.summary != ""
        assert len(strategy.steps) >= 2
        for step in strategy.steps:
            assert step.name != ""
            assert step.instruction != ""
            assert step.purpose != ""
            assert 0.0 < step.temperature <= 1.0
            assert step.produces in {
                "reasoning", "plan", "draft", "check", "final_answer",
            }


def test_step_names_are_unique_across_all_strategies(registry):
    names = [s.name for strat in registry.values() for s in strat.steps]
    assert len(names) == len(set(names))


def test_check_answer_is_the_coldest_step_in_the_module(registry):
    steps = {s.name: s for strat in registry.values() for s in strat.steps}
    assert steps["check_answer"].temperature == min(
        s.temperature for s in steps.values()
    )


# ── Interface invariants ──────────────────────────────────────────────────────

def test_exactly_one_final_answer_step_and_it_is_last(registry):
    for strategy in registry.values():
        finals = [s for s in strategy.steps if s.produces == "final_answer"]
        assert len(finals) == 1
        assert strategy.steps[-1].produces == "final_answer"


def test_only_the_first_step_runs_without_prior_output(registry):
    for strategy in registry.values():
        assert strategy.steps[0].requires_prior is False
        assert all(s.requires_prior for s in strategy.steps[1:])


def test_builder_rejects_a_later_step_run_cold(prompts, registry):
    strategy = registry["chain_of_thought"]
    with pytest.raises(ValueError):
        _build(prompts, strategy, strategy.steps[1], prior="")


def test_builder_rejects_prior_output_fed_to_a_first_step(prompts, registry):
    strategy = registry["plan_then_answer"]
    with pytest.raises(ValueError):
        _build(prompts, strategy, strategy.steps[0], prior=CANONICAL_PRIOR_TEXT)


# ── Selection ─────────────────────────────────────────────────────────────────

def test_selection_table_is_exact(registry):
    expected = {
        "technical_debugging": "chain_of_thought",
        "technical_spec_request": "chain_of_thought",
        "planning_request": "plan_then_answer",
        "brainstorming": "plan_then_answer",
        "factual_query": "self_check",
        "editing_refinement": "self_check",
    }
    for intent, name in expected.items():
        assert select_strategy(intent) is registry[name]


def test_no_intent_is_claimed_by_two_strategies(registry):
    claimed = [i for s in registry.values() for i in s.applies_to]
    assert len(claimed) == len(set(claimed))


def test_selection_covers_the_registry_applies_to_exactly(registry):
    for strategy in registry.values():
        for intent in strategy.applies_to:
            assert select_strategy(intent) is strategy


def test_light_and_unknown_intents_get_no_strategy():
    for intent in (
        "emotional_support", "light_conversation", "naming_or_branding",
        FALLBACK_INTENT, "no_such_intent",
    ):
        assert select_strategy(intent) is None


def test_every_selected_intent_is_a_real_router_intent(registry):
    for strategy in registry.values():
        for intent in strategy.applies_to:
            assert intent in _INTENT_TO_MODE


# ── The prompt contract ───────────────────────────────────────────────────────

def test_every_step_prompt_starts_with_its_instruction(prompts, registry):
    for strategy in registry.values():
        for step in strategy.steps:
            prior = CANONICAL_PRIOR_TEXT if step.requires_prior else ""
            prompt = _build(prompts, strategy, step, prior=prior)
            assert prompt.startswith(step.instruction)
            assert "User request: test query" in prompt
            assert "Goal:" in prompt


def test_working_so_far_appears_only_for_prior_consuming_steps(prompts, registry):
    for strategy in registry.values():
        first = _build(prompts, strategy, strategy.steps[0])
        assert "Working so far" not in first
        for step in strategy.steps[1:]:
            prompt = _build(prompts, strategy, step, prior=CANONICAL_PRIOR_TEXT)
            assert "Working so far" in prompt
            assert CANONICAL_PRIOR_TEXT in prompt


def test_output_line_names_what_the_step_produces(prompts, registry):
    for strategy in registry.values():
        for step in strategy.steps:
            prior = CANONICAL_PRIOR_TEXT if step.requires_prior else ""
            prompt = _build(prompts, strategy, step, prior=prior)
            if step.produces == "final_answer":
                assert "final_answer only" not in prompt
                assert prompt.rstrip().endswith((
                    "Keep it short.", "Keep it focused.",
                    "Be thorough but not verbose.",
                ))
            else:
                assert prompt.rstrip().endswith(f"Output: {step.produces} only")


def test_fact_block_rides_into_strategy_prompts(prompts, registry):
    strategy = registry["self_check"]
    sketch = canonical_sketch(strategy.applies_to[0])
    scaffold = ScaffoldBuilder().build(sketch, MemoryActivationBundle())
    scaffold.fact_block = "[Q1] synthetic fact line"
    prompt = prompts.build_strategy_prompt(
        sketch, scaffold, strategy.steps[0], [],
    )
    assert "Knowledge context:" in prompt
    assert "synthetic fact line" in prompt


# ── Measured cost ─────────────────────────────────────────────────────────────

def test_strategy_costs_recomputed_independently(prompts, registry):
    measured = {(c.strategy, c.step): c for c in measure_strategy_costs()}
    total_steps = sum(len(s.steps) for s in registry.values())
    assert len(measured) == total_steps
    builder = ScaffoldBuilder()
    for name, strategy in registry.items():
        sketch = canonical_sketch(strategy.applies_to[0])
        scaffold = builder.build(sketch, MemoryActivationBundle())
        for step in strategy.steps:
            prior = CANONICAL_PRIOR_TEXT if step.requires_prior else ""
            prompt = prompts.build_strategy_prompt(
                sketch, scaffold, step, [], prior_text=prior,
            )
            cost = measured[(name, step.name)]
            assert cost.chars == len(prompt)
            assert cost.word_tokens == len(prompt.split())


def test_step_costs_stay_inside_the_pinned_band():
    for cost in measure_strategy_costs():
        assert 50 <= cost.word_tokens <= 85


def test_self_check_is_the_dearest_strategy():
    totals: dict[str, int] = {}
    for cost in measure_strategy_costs():
        totals[cost.strategy] = totals.get(cost.strategy, 0) + cost.word_tokens
    assert totals["self_check"] > totals["chain_of_thought"]
    assert totals["self_check"] > totals["plan_then_answer"]
    # A strategy run costs more than TF-021's dearest two-pass turn (119).
    assert all(total > 119 for total in totals.values())


# ── The F6 ruling ─────────────────────────────────────────────────────────────

def test_f6_thresholds_are_pinned_and_different():
    assert core_module._MIN_FRAGMENT_SCORE == 0.25
    assert salvage_module._MIN_FRAGMENT_SCORE == 0.54


def test_f6_measured_sentence_is_kept_by_core_and_refused_by_salvage():
    # The catalogue's proof sentence (SCAFFOLD_STRATEGIES §5): the two
    # formulas are different instruments, so their verdicts differ —
    # and both are right for their job.
    sentence = (
        "The lantern keeper trims the wick at dusk every single "
        "evening without fail."
    )
    sketch = canonical_sketch("factual_query")
    scaffold = ScaffoldBuilder().build(sketch, MemoryActivationBundle())
    core_composite = core_module._score_fragment(sentence, sketch, scaffold).composite
    salvage_score = FragmentSalvage()._score_fragment(sentence, set())
    assert core_composite == pytest.approx(0.285, abs=0.002)
    assert salvage_score == pytest.approx(0.446, abs=0.002)
    assert core_composite >= core_module._MIN_FRAGMENT_SCORE
    assert salvage_score < salvage_module._MIN_FRAGMENT_SCORE


# ── The catalogue doc ─────────────────────────────────────────────────────────

def test_catalogue_doc_names_every_strategy_and_step(registry):
    doc = (
        Path(__file__).resolve().parent.parent
        / "docs" / "architecture" / "SCAFFOLD_STRATEGIES.md"
    )
    text = doc.read_text(encoding="utf-8")
    for name, strategy in registry.items():
        assert name in text
        for step in strategy.steps:
            assert step.name in text
        for intent in strategy.applies_to:
            assert intent in text
