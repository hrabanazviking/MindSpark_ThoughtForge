"""
ScaffoldStrategies — the multi-step reasoning scaffolds (TF-022,
Scaffold Library II).

TF-021 catalogued the scaffolds the tree already had: one
``CognitionScaffold`` steering object per intent, plus the single-pass
candidate modes it names. This module adds the three multi-step
reasoning scaffolds the roadmap names — chain-of-thought,
plan-then-answer, and self-check — and brings them to **one
interface**, the same interface discipline TF-021 set:

  * A strategy is an ordered tuple of steps (``StrategyStep``).
  * Every step carries the same fields: a name, an instruction (the
    opener line its prompt starts with), a purpose, a temperature,
    whether it consumes the previous steps' output, and what it
    produces. No strategy has its own private calling convention.
  * Every step's prompt is built by the one builder method,
    ``PromptBuilder.build_strategy_prompt`` — the same prompt
    assembly order as a candidate prompt, plus a "Working so far"
    section for the steps that consume prior output.
  * A strategy ends with exactly one step whose ``produces`` is
    ``"final_answer"``, and it is the last step. Every earlier step
    produces working material (reasoning / a plan / a draft / a
    check), never the answer the user sees.
  * Only the first step may run without prior output; every later
    step *requires* it. The builder enforces both directions —
    running a later step cold, or feeding prior output to a first
    step, is a caller error, not a silent no-op.
  * ``select_strategy`` is the one selection table: a deterministic
    intent → strategy map, written down here. Intents it does not
    list get no strategy (see the registry doc for why: light-depth
    turns would pay multi-pass cost for nothing and risk their tone).

Strategies are *not* candidate modes: they do not enter
``scaffold_registry``'s candidate-mode tables (TF-021's tests pin
those at the eight single-pass modes), because a strategy is a
sequence *over* passes, not one pass. The human-readable catalogue
is ``docs/architecture/SCAFFOLD_STRATEGIES.md``, including this
slice's ruling on CORE_LOOP §8-F6 (the two fragment thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass

from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.scaffold_registry import canonical_sketch
from thoughtforge.knowledge.models import MemoryActivationBundle

# ── The canonical measurement sketch's prior output ──────────────────────────
# Cost comparisons hold the working material fixed, exactly as TF-021
# held the request fixed: a later step is measured carrying this
# three-word synthetic prior output, nothing more.

CANONICAL_PRIOR_TEXT = "First step output."


# ── The one step interface ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyStep:
    """One step of a strategy: the same fields for every strategy."""

    name: str
    instruction: str
    purpose: str
    temperature: float
    requires_prior: bool
    produces: str  # "reasoning" | "plan" | "draft" | "check" | "final_answer"


@dataclass(frozen=True)
class ScaffoldStrategy:
    """One multi-step scaffold: ordered steps + the intents it serves."""

    name: str
    steps: tuple[StrategyStep, ...]
    applies_to: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class StrategyPromptCost:
    """One step prompt's measured cost on the canonical sketch."""

    strategy: str
    step: str
    chars: int
    word_tokens: int


# ── The three strategies ──────────────────────────────────────────────────────

_CHAIN_OF_THOUGHT = ScaffoldStrategy(
    name="chain_of_thought",
    summary=(
        "Reason in short numbered steps first, then answer from the "
        "reasoning — the intermediate steps a small model would "
        "otherwise skip are made explicit, and kept out of the answer."
    ),
    applies_to=("technical_debugging", "technical_spec_request"),
    steps=(
        StrategyStep(
            name="reason_stepwise",
            instruction=(
                "Think step by step in short numbered steps. "
                "Reason only — do not give the final answer yet."
            ),
            purpose=(
                "Reasoning pass: externalises the causal chain "
                "(symptom → cause → fix, or requirement → design) so "
                "the answer step can build on it instead of guessing."
            ),
            temperature=0.60,
            requires_prior=False,
            produces="reasoning",
        ),
        StrategyStep(
            name="answer_from_reasoning",
            instruction=(
                "Using the reasoning above, write the final answer. "
                "Do not repeat the reasoning steps."
            ),
            purpose=(
                "Answer pass: converts the reasoning into the clean "
                "final response at the registry's coldest band, with "
                "the working kept out of what the user reads."
            ),
            temperature=0.50,
            requires_prior=True,
            produces="final_answer",
        ),
    ),
)

_PLAN_THEN_ANSWER = ScaffoldStrategy(
    name="plan_then_answer",
    summary=(
        "Decide the shape before generating: a short ordered plan "
        "first, then the answer executed against that plan."
    ),
    applies_to=("planning_request", "brainstorming"),
    steps=(
        StrategyStep(
            name="outline_plan",
            instruction=(
                "Write a short plan only: the ordered steps needed "
                "to answer this well. Do not answer yet."
            ),
            purpose=(
                "Planning pass: fixes order, coverage, and — for "
                "brainstorming — the distinct angles, before any "
                "answer text exists to anchor on."
            ),
            temperature=0.55,
            requires_prior=False,
            produces="plan",
        ),
        StrategyStep(
            name="execute_plan",
            instruction="Follow the plan above in order and write the final answer.",
            purpose=(
                "Execution pass: writes the answer against the plan, "
                "so order and completeness come from the plan step "
                "rather than from improvisation."
            ),
            temperature=0.55,
            requires_prior=True,
            produces="final_answer",
        ),
    ),
)

_SELF_CHECK = ScaffoldStrategy(
    name="self_check",
    summary=(
        "Draft, then check the draft against the request, the goal, "
        "and the facts given, then rewrite fixing what the check "
        "found — for intents whose answers can actually be checked."
    ),
    applies_to=("factual_query", "editing_refinement"),
    steps=(
        StrategyStep(
            name="draft_answer",
            instruction=(
                "Write a first draft answer. It will be checked, "
                "so be direct and specific."
            ),
            purpose=(
                "Draft pass: a complete first attempt, written to be "
                "checked — specificity is what makes checking possible."
            ),
            temperature=0.60,
            requires_prior=False,
            produces="draft",
        ),
        StrategyStep(
            name="check_answer",
            instruction=(
                "Check the draft above against the request, the goal, "
                "and the facts given. List each problem you find: "
                "wrong, missing, or unsupported. If the draft is "
                "clean, say so in one line."
            ),
            purpose=(
                "Check pass: the coldest step in this module — "
                "verification, not generation. Its output is a "
                "problem list, never a new answer."
            ),
            temperature=0.40,
            requires_prior=True,
            produces="check",
        ),
        StrategyStep(
            name="revise_from_check",
            instruction=(
                "Rewrite the draft, fixing every problem the check "
                "listed. Write the final answer only."
            ),
            purpose=(
                "Revision pass: the only step whose text the user "
                "sees. Its prior input is the draft and the check "
                "together, supplied by the caller as one working text."
            ),
            temperature=0.50,
            requires_prior=True,
            produces="final_answer",
        ),
    ),
)

_STRATEGIES: tuple[ScaffoldStrategy, ...] = (
    _CHAIN_OF_THOUGHT,
    _PLAN_THEN_ANSWER,
    _SELF_CHECK,
)


# ── The one selection table ───────────────────────────────────────────────────
# Deterministic, intent-only. Every router intent appears in at most
# one strategy's applies_to; select_strategy is derived from the
# registry (never a second hand-kept table), so the two cannot drift.

def strategy_registry() -> dict[str, ScaffoldStrategy]:
    """Every strategy in this module, keyed by strategy name."""
    return {s.name: s for s in _STRATEGIES}


def select_strategy(intent: str) -> ScaffoldStrategy | None:
    """
    The strategy for an intent, or None.

    None is a real answer, not a gap: emotional_support,
    light_conversation, naming_or_branding, the unknown-intent
    fallback, and any unknown intent get no multi-step scaffold —
    they are light-depth turns where extra passes cost more than
    they add and a reasoning voice would damage the tone the
    scaffold exists to keep.
    """
    for strategy in _STRATEGIES:
        if intent in strategy.applies_to:
            return strategy
    return None


# ── Measured cost ─────────────────────────────────────────────────────────────

def measure_strategy_costs() -> list[StrategyPromptCost]:
    """
    Build every step's prompt on the canonical sketch of its
    strategy's first intent (TF-021's fixed synthetic request, empty
    bundle, no cues; later steps carry CANONICAL_PRIOR_TEXT) and
    count the cost: characters, and word-tokens by the tree's own
    ``len(text.split())`` rule.
    """
    builder = ScaffoldBuilder()
    prompts = PromptBuilder()
    costs: list[StrategyPromptCost] = []
    for strategy in _STRATEGIES:
        sketch = canonical_sketch(strategy.applies_to[0])
        scaffold = builder.build(sketch, MemoryActivationBundle())
        for step in strategy.steps:
            prior = CANONICAL_PRIOR_TEXT if step.requires_prior else ""
            prompt = prompts.build_strategy_prompt(
                sketch, scaffold, step, [], prior_text=prior,
            )
            costs.append(
                StrategyPromptCost(
                    strategy=strategy.name,
                    step=step.name,
                    chars=len(prompt),
                    word_tokens=len(prompt.split()),
                )
            )
    return costs


__all__ = [
    "CANONICAL_PRIOR_TEXT",
    "ScaffoldStrategy",
    "StrategyPromptCost",
    "StrategyStep",
    "measure_strategy_costs",
    "select_strategy",
    "strategy_registry",
]
