"""
Depth suite, as data (TF-124).

The roadmap's rule: *depth suite: multi-step reasoning
with scaffolds on/off — the scaffold value measured, not
assumed.* TF-121's harness filed the architecture,
TF-122 built the persona suite, TF-123 the memory
suite. This slice is the depth consumer: it scores
TF-022's three multi-step scaffolds against the
single-pass road they are meant to beat.

The measured starting state, probed live this tick on
this tree before anything was built:

  * **No depth / reasoning scorer existed anywhere.**
    The benchmarks estate held the persona scorer +
    suite, the memory suite and the profile runner;
    no module, fixture or harness entry scored a
    multi-step reasoning outcome, with or without a
    scaffold. The scaffold value was asserted in
    TF-022's purposes ("so order and completeness come
    from the plan step rather than from
    improvisation") and measured by nothing.
  * **The scaffold estate itself is real and was
    probed live**: TF-021's registry files 10 intent
    scaffolds / 8 candidate modes; TF-022's
    ``strategy_registry()`` files exactly three
    strategies (chain_of_thought, plan_then_answer,
    self_check) over six intents, ``select_strategy``
    returning None for the light intents by design;
    ``PromptBuilder.build_strategy_prompt`` builds
    every step prompt and enforces the prior-output
    contract both ways (a later step run cold raises
    ValueError — probed); a single-pass candidate
    prompt (the OFF road) carries no "Working so far"
    section, a strategy step prompt consuming prior
    work carries one (both probed).
  * **The filed limit the suite must state, not
    hide**: the core loop does not execute strategies
    automatically (TF-022's doc, §1: wiring execution
    into ``ThoughtForgeCore.think`` belongs to the
    slices that own those surfaces; TF-032 landed
    per-conversation *selectability* only —
    ``core.selected_strategy`` resolves a strategy,
    and no core code path calls
    ``build_strategy_prompt``; grepped live this
    tick). TF-022's doc assigned that wiring without
    the gap ever being numbered in the TF-016
    docket. Filed as D-BENCH-10 and fixed by this
    slice in the measuring sense only (see below):
    the suite measures the scaffold machinery's
    value at the prompt + scripted-fixture level
    through its real code paths; live execution is
    NOT PROVEN and never claimed.

What this module files, as verifiable data:

  * ``CHAIN_CASES`` / ``PLAN_CASES`` / ``CHECK_CASES``
    — 36 scripted synthetic multi-step problems,
    12 per axis, one axis per TF-022 strategy. Every
    case files a question with a known correct
    answer, the strategy + intent TF-022's real
    selection table must return for it, a scripted
    direct (scaffold OFF) answer attempt, the
    scaffold's scripted working material (reasoning
    steps / an ordered plan / a draft check), the
    filed intermediate values that working material
    must carry, a scripted scaffolded (ON) answer,
    and the filed on/off expectations.
  * The scorers judge mechanically. OFF holds exactly
    when the direct answer equals the filed correct
    answer. ON holds exactly when (a) the working
    material carries every filed intermediate (for
    the plan axis, in the filed order), and (b) the
    scaffolded answer equals the correct answer.
    Correctness is exact normalised equality with
    the known answer — no phrase matcher, no
    judgement smuggled in.
  * ``scaffold_prompts_for()`` — the real-code leg:
    for a case it builds the OFF prompt through
    TF-021's real ``ScaffoldBuilder`` +
    ``PromptBuilder.build_candidate_prompt`` (the
    single-pass road) and the ON prompts through
    ``build_strategy_prompt`` for every step of the
    strategy TF-022's ``select_strategy`` really
    returns, chaining the case's working material as
    prior output. The checker runs this leg for
    every filed case, so the suite's on/off contrast
    is judged against prompts the real builders
    actually produce (the OFF prompt carries no
    "Working so far"; the final ON prompt does).
  * ``run_depth_suite()`` — per-axis OFF and ON
    scores, their deltas, and the overall OFF / ON /
    delta (the means over the three axes). The
    verdict is PASS only on a non-empty run where
    every axis's ON score is at or above the filed
    0.75 threshold AND the overall delta is
    positive — a scaffold that does not measurably
    help cannot PASS this suite. An empty axis or
    empty suite scores 0.0 / REVIEW (the TF-122 /
    TF-123 precedent: no vacuous pass anywhere).

The measured shape is filed honestly, including
where the scaffold does not help: the corpora file
no-gain cases (the scaffolded answer fails where the
direct one also fails, and cases both roads pass)
and scaffold-loss cases (the direct answer holds and
the scaffolded one fails — a reasoning slip, a plan
that locks in a wrong order, a check that invents a
problem and breaks a correct draft). The delta is
the measured difference of the two scored roads,
never an asserted one.

D-BENCH-10 is FIXED by this slice in the measuring
sense: scaffold value now has a mechanical scorer
with a pinned harness baseline. What is NOT fixed,
and is stated in the docket row and the filed
limits: strategy execution in the core loop — the
suite runs scripted working material through the
real prompt builders; it does not make
``ThoughtForgeCore.think`` run strategies, and a
live model's reasoning under these scaffolds is
NOT PROVEN (TF-126's road).

Filed limits, stated plainly:

  * The cases are scripted synthetic fixtures
    (TF-017): invented problems with known answers,
    scripted direct / working / scaffolded texts,
    never canon, never a model's live output, and
    ``data/`` was never opened. The suite proves the
    *scoring* end to end and measures the scripted
    delta; a live small model's depth behaviour is
    NOT PROVEN here (TF-126's road).
  * ON correctness requires the filed intermediates
    in the working material plus an exactly correct
    final answer; working material that reaches the
    right answer without the filed steps does not
    hold — the suite scores scaffolded reasoning,
    not lucky guesses.
  * The OFF road is the single-pass candidate prompt
    (TF-021's first candidate mode for the intent);
    the core's candidate *selection* among modes is
    not scored here.

The human-readable write-up is
``docs/architecture/DEPTH_SUITE.md``.
Proof grade: FIXTURE — synthetic scripted cases
through the real TF-020 / TF-021 / TF-022 machinery
on this VM; no live model, never claimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DEPTH_SUITE_FORMAT = "thoughtforge-depth-suite/1"
DEPTH_SUITE_DOC = "docs/architecture/DEPTH_SUITE.md"

#: Per-axis ON pass threshold (the estate's filed
#: 0.75, TF-122 / TF-123's shape) and the filed
#: corpus scale: 12 cases per axis, 36 in total.
AXIS_PASS_THRESHOLD = 0.75
CASES_PER_AXIS = 12

# The real machinery, reused — never copied: TF-020's
# intent -> response-mode table, TF-021's scaffold
# builder, TF-022's selection table + strategies, and
# the one prompt builder both roads run through.
from thoughtforge.cognition.prompt_builder import PromptBuilder  # noqa: E402
from thoughtforge.cognition.router import _INTENT_TO_MODE  # noqa: E402
from thoughtforge.cognition.scaffold import ScaffoldBuilder  # noqa: E402
from thoughtforge.cognition.scaffold_strategies import (  # noqa: E402
    select_strategy,
    strategy_registry,
)
from thoughtforge.knowledge.models import (  # noqa: E402
    InputSketch,
    MemoryActivationBundle,
)


def _norm(text: str) -> str:
    """The suite's answer normalisation (TF-122's shape)."""
    clean = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(clean.split())


def _contains(text_norm: str, phrase: str) -> bool:
    return f" {_norm(phrase)} " in f" {text_norm} "


def _answer_correct(answer: str, expected: str) -> bool:
    return _norm(answer) == _norm(expected)


# ── The chain-of-thought axis ───────────────────────────────────────────────
#
# Arithmetic chains and multi-hop logic chains under
# TF-022's chain_of_thought (technical_debugging /
# technical_spec_request). Filed shape: 5 direct holds,
# 10 scaffolded holds — two no-gain cases (one both
# pass, counted in both; one both fail) and one
# scaffold-loss case (reasoning slip) among them, by
# design; see the per-case expectations.

CHAIN_CASES: tuple[dict[str, Any], ...] = (
    {"id": "c01_precedence_chain", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "Compute 3 * 4 + 5 * 2.",
     "expected": "22",
     "off_answer": "34",
     "working": "1. 3 * 4 = 12. 2. 5 * 2 = 10. 3. 12 + 10 = 22.",
     "intermediates": ("12", "10", "22"),
     "on_answer": "22", "off_holds": False, "on_holds": True},
    {"id": "c02_divide_then_scale", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "Compute (48 / 6) + 7 * 3.",
     "expected": "29",
     "off_answer": "45",
     "working": "1. 48 / 6 = 8. 2. 7 * 3 = 21. 3. 8 + 21 = 29.",
     "intermediates": ("8", "21", "29"),
     "on_answer": "29", "off_holds": False, "on_holds": True},
    {"id": "c03_percent_then_add", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "Take 15 percent of 240, then add 36.",
     "expected": "72",
     "off_answer": "72",
     "working": "1. 15 percent of 240 = 36. 2. 36 + 36 = 72.",
     "intermediates": ("36", "72"),
     "on_answer": "72", "off_holds": True, "on_holds": True},
    {"id": "c04_successive_discounts", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "A price of 80 is cut by 25 percent, then the "
                 "result is cut by 10 percent. Final price?",
     "expected": "54",
     "off_answer": "52",
     "working": "1. 80 cut by 25 percent = 60. 2. 60 cut by "
                "10 percent = 54.",
     "intermediates": ("60", "54"),
     "on_answer": "54", "off_holds": False, "on_holds": True},
    {"id": "c05_logic_all_chain", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "All A are B. All B are C. X is an A. Is X a C?",
     "expected": "yes",
     "off_answer": "yes",
     "working": "1. X is an A. 2. All A are B, so X is a B. "
                "3. All B are C, so X is a C.",
     "intermediates": ("x is a b", "x is a c"),
     "on_answer": "yes", "off_holds": True, "on_holds": True},
    {"id": "c06_two_products", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "Compute 7 * 8 - 9 * 4.",
     "expected": "20",
     "off_answer": "22",
     "working": "1. 7 * 8 = 56. 2. 9 * 4 = 36. 3. 56 - 36 = 20.",
     "intermediates": ("56", "36", "20"),
     "on_answer": "20", "off_holds": False, "on_holds": True},
    {"id": "c07_ratio_share", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "Split 64 in the ratio 3 to 5. What is the "
                 "larger share?",
     "expected": "40",
     "off_answer": "38",
     "working": "1. 3 + 5 = 8 parts. 2. 64 / 8 = 8 per part. "
                "3. Larger share = 5 * 8 = 40.",
     "intermediates": ("8 parts", "40"),
     "on_answer": "40", "off_holds": False, "on_holds": True},
    {"id": "c08_journey_with_stop", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "A trip covers 150 km at 50 km per hour, plus "
                 "a 45 minute stop. Total time?",
     "expected": "3 hours 45 minutes",
     "off_answer": "3 hours",
     "working": "1. 150 / 50 = 3 hours driving. 2. Add the "
                "45 minute stop: 3 hours 45 minutes.",
     "intermediates": ("3 hours", "45 minute"),
     "on_answer": "3 hours 45 minutes",
     "off_holds": False, "on_holds": True},
    {"id": "c09_carry_chain", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "Compute (999 + 1) * 3.",
     "expected": "3000",
     "off_answer": "3000",
     "working": "1. 999 + 1 = 1000. 2. 1000 * 3 = 3000.",
     "intermediates": ("1000", "3000"),
     "on_answer": "3000", "off_holds": True, "on_holds": True},
    {"id": "c10_ordering_chain", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "A comes before B. B comes before C. C comes "
                 "before D. Which comes first?",
     "expected": "a",
     "off_answer": "a",
     "working": "1. A is before B. 2. B is before C and C "
                "before D, so the order starts with A.",
     "intermediates": ("before b", "before c"),
     "on_answer": "a", "off_holds": True, "on_holds": True},
    {"id": "c11_scaffold_loss_slip", "strategy": "chain_of_thought",
     "intent": "technical_debugging",
     "question": "Compute 12 * 3 + 4.",
     "expected": "40",
     "off_answer": "40",
     "working": "1. 12 * 3 = 36. 2. 36 + 4 = 44.",
     "intermediates": ("36", "44"),
     "on_answer": "44", "off_holds": True, "on_holds": False},
    {"id": "c12_no_gain_transcription", "strategy": "chain_of_thought",
     "intent": "technical_spec_request",
     "question": "Start with 200. Add 10 percent, then add "
                 "10 percent to the result.",
     "expected": "242",
     "off_answer": "240",
     "working": "1. 200 plus 10 percent = 220. 2. 220 plus "
                "10 percent = 242.",
     "intermediates": ("220", "242"),
     "on_answer": "224", "off_holds": False, "on_holds": False},
)


# ── The plan-then-answer axis ───────────────────────────────────────────────
#
# Multi-step ordering / coverage problems under TF-022's
# plan_then_answer (planning_request / brainstorming).
# The expected answer is an ordered sequence; the plan
# axis additionally requires the filed intermediates
# to appear in the working plan IN ORDER. Filed shape:
# 6 direct holds, 9 scaffolded holds — one scaffold-
# loss case (a plan locking in a wrong order) and
# no-gain cases among them, by design.

PLAN_CASES: tuple[dict[str, Any], ...] = (
    {"id": "p01_fete_order", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order a village fete: the hall must be booked "
                 "before invites go out, invites before the food "
                 "order (numbers needed), and setup after the "
                 "food order is placed.",
     "expected": "book hall, send invites, order food, set up hall",
     "off_answer": "send invites, book hall, order food, set up hall",
     "working": "Plan: 1. book hall. 2. send invites. 3. order "
                "food. 4. set up hall.",
     "intermediates": ("book hall", "send invites", "order food",
                       "set up hall"),
     "on_answer": "book hall, send invites, order food, set up hall",
     "off_holds": False, "on_holds": True},
    {"id": "p02_build_order", "strategy": "plan_then_answer",
     "intent": "brainstorming",
     "question": "Order a small build: design before code, code "
                 "before tests, tests before release.",
     "expected": "design, code, tests, release",
     "off_answer": "design, tests, code, release",
     "working": "Plan: 1. design. 2. code. 3. tests. 4. release.",
     "intermediates": ("design", "code", "tests", "release"),
     "on_answer": "design, code, tests, release",
     "off_holds": False, "on_holds": True},
    {"id": "p03_tea_order", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order making tea: boil water before pouring, "
                 "pour before steeping, steep before serving.",
     "expected": "boil water, pour, steep, serve",
     "off_answer": "boil water, pour, steep, serve",
     "working": "Plan: 1. boil water. 2. pour. 3. steep. "
                "4. serve.",
     "intermediates": ("boil water", "pour", "steep", "serve"),
     "on_answer": "boil water, pour, steep, serve",
     "off_holds": True, "on_holds": True},
    {"id": "p04_crossing_prep", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order the ferry prep: check the tide before "
                 "casting off, cast off before crossing, cross "
                 "before docking.",
     "expected": "check tide, cast off, cross, dock",
     "off_answer": "cast off, check tide, cross, dock",
     "working": "Plan: 1. check tide. 2. cast off. 3. cross. "
                "4. dock.",
     "intermediates": ("check tide", "cast off", "cross", "dock"),
     "on_answer": "check tide, cast off, cross, dock",
     "off_holds": False, "on_holds": True},
    {"id": "p05_letter_order", "strategy": "plan_then_answer",
     "intent": "brainstorming",
     "question": "Order sending a letter: write before folding, "
                 "fold before sealing, seal before posting.",
     "expected": "write, fold, seal, post",
     "off_answer": "write, fold, seal, post",
     "working": "Plan: 1. write. 2. fold. 3. seal. 4. post.",
     "intermediates": ("write", "fold", "seal", "post"),
     "on_answer": "write, fold, seal, post",
     "off_holds": True, "on_holds": True},
    {"id": "p06_garden_order", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order the garden work: clear the bed before "
                 "digging, dig before sowing, sow before "
                 "watering.",
     "expected": "clear bed, dig, sow, water",
     "off_answer": "dig, clear bed, sow, water",
     "working": "Plan: 1. clear bed. 2. dig. 3. sow. 4. water.",
     "intermediates": ("clear bed", "dig", "sow", "water"),
     "on_answer": "clear bed, dig, sow, water",
     "off_holds": False, "on_holds": True},
    {"id": "p07_bread_order", "strategy": "plan_then_answer",
     "intent": "brainstorming",
     "question": "Order baking bread: mix before kneading, "
                 "knead before proving, prove before baking.",
     "expected": "mix, knead, prove, bake",
     "off_answer": "mix, knead, prove, bake",
     "working": "Plan: 1. mix. 2. knead. 3. prove. 4. bake.",
     "intermediates": ("mix", "knead", "prove", "bake"),
     "on_answer": "mix, knead, prove, bake",
     "off_holds": True, "on_holds": True},
    {"id": "p08_plan_misses_step", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order a survey: calibrate the instrument "
                 "before measuring, measure before recording, "
                 "record before reporting.",
     "expected": "calibrate, measure, record, report",
     "off_answer": "measure, calibrate, record, report",
     "working": "Plan: 1. measure. 2. record. 3. report.",
     "intermediates": ("measure", "record", "report"),
     "on_answer": "measure, record, report",
     "off_holds": False, "on_holds": False},
    {"id": "p09_lantern_order", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order the lantern work: trim the wick before "
                 "filling, fill before lighting, light before "
                 "hanging.",
     "expected": "trim wick, fill, light, hang",
     "off_answer": "trim wick, fill, light, hang",
     "working": "Plan: 1. trim wick. 2. fill. 3. light. "
                "4. hang.",
     "intermediates": ("trim wick", "fill", "light", "hang"),
     "on_answer": "trim wick, fill, light, hang",
     "off_holds": True, "on_holds": True},
    {"id": "p10_voyage_order", "strategy": "plan_then_answer",
     "intent": "brainstorming",
     "question": "Order a voyage: provision before loading, "
                 "load before departing, depart before "
                 "navigating.",
     "expected": "provision, load, depart, navigate",
     "off_answer": "load, provision, depart, navigate",
     "working": "Plan: 1. provision. 2. load. 3. depart. "
                "4. navigate.",
     "intermediates": ("provision", "load", "depart", "navigate"),
     "on_answer": "provision, load, depart, navigate",
     "off_holds": False, "on_holds": True},
    {"id": "p11_scaffold_loss_order", "strategy": "plan_then_answer",
     "intent": "planning_request",
     "question": "Order dressing a wound: clean before "
                 "dressing, dress before bandaging.",
     "expected": "clean, dress, bandage",
     "off_answer": "clean, dress, bandage",
     "working": "Plan: 1. dress. 2. clean. 3. bandage.",
     "intermediates": ("dress", "clean", "bandage"),
     "on_answer": "dress, clean, bandage",
     "off_holds": True, "on_holds": False},
    {"id": "p12_no_gain_coverage", "strategy": "plan_then_answer",
     "intent": "brainstorming",
     "question": "Order a harvest day: sharpen tools before "
                 "cutting, cut before gathering, gather before "
                 "storing.",
     "expected": "sharpen, cut, gather, store",
     "off_answer": "sharpen, cut, gather, store",
     "working": "Plan: 1. sharpen. 2. gather. 3. store.",
     "intermediates": ("sharpen", "gather", "store"),
     "on_answer": "sharpen, gather, store",
     "off_holds": True, "on_holds": False},
)


# ── The self-check axis ─────────────────────────────────────────────────────
#
# Draft -> check -> revise under TF-022's self_check
# (factual_query / editing_refinement). The OFF answer
# is the draft; the working material is the check
# step's output, which must name the filed error token
# (for a correct draft, the filed token is the check's
# clean verdict). Filed shape: 6 direct (draft) holds,
# 10 scaffolded (revised) holds — one scaffold-loss
# case (an invented problem breaks a correct draft)
# and one no-gain case (the check misses the error),
# by design.

CHECK_CASES: tuple[dict[str, Any], ...] = (
    {"id": "s01_capital_draft", "strategy": "self_check",
     "intent": "factual_query",
     "question": "What is the capital of Norway?",
     "expected": "oslo",
     "off_answer": "bergen",
     "working": "Check: the draft says bergen; the capital of "
                "norway is oslo — wrong capital.",
     "intermediates": ("wrong capital",),
     "on_answer": "oslo", "off_holds": False, "on_holds": True},
    {"id": "s02_sum_draft", "strategy": "self_check",
     "intent": "editing_refinement",
     "question": "State the sum: 17 + 25.",
     "expected": "42",
     "off_answer": "41",
     "working": "Check: 17 + 25 = 42, not 41 — arithmetic error.",
     "intermediates": ("arithmetic error",),
     "on_answer": "42", "off_holds": False, "on_holds": True},
    {"id": "s03_clean_draft", "strategy": "self_check",
     "intent": "factual_query",
     "question": "What is the capital of Sweden?",
     "expected": "stockholm",
     "off_answer": "stockholm",
     "working": "Check: the draft is clean, no problems found.",
     "intermediates": ("no problems",),
     "on_answer": "stockholm", "off_holds": True, "on_holds": True},
    {"id": "s04_days_draft", "strategy": "self_check",
     "intent": "factual_query",
     "question": "How many days are in a leap year?",
     "expected": "366",
     "off_answer": "365",
     "working": "Check: a leap year has 366 days — wrong count.",
     "intermediates": ("wrong count",),
     "on_answer": "366", "off_holds": False, "on_holds": True},
    {"id": "s05_clean_product", "strategy": "self_check",
     "intent": "editing_refinement",
     "question": "State the product: 6 * 7.",
     "expected": "42",
     "off_answer": "42",
     "working": "Check: the draft is clean, no problems found.",
     "intermediates": ("no problems",),
     "on_answer": "42", "off_holds": True, "on_holds": True},
    {"id": "s06_check_misses_error", "strategy": "self_check",
     "intent": "factual_query",
     "question": "What is the largest planet in the solar system?",
     "expected": "jupiter",
     "off_answer": "saturn",
     "working": "Check: the draft names a large planet; no "
                "problems found.",
     "intermediates": ("no problems",),
     "on_answer": "saturn", "off_holds": False, "on_holds": False},
    {"id": "s07_clean_river", "strategy": "self_check",
     "intent": "factual_query",
     "question": "Which river flows through Egypt?",
     "expected": "the nile",
     "off_answer": "the nile",
     "working": "Check: the draft is clean, no problems found.",
     "intermediates": ("no problems",),
     "on_answer": "the nile", "off_holds": True, "on_holds": True},
    {"id": "s08_clean_boiling", "strategy": "self_check",
     "intent": "editing_refinement",
     "question": "At sea level, water boils at how many "
                 "degrees celsius?",
     "expected": "100",
     "off_answer": "100",
     "working": "Check: the draft is clean, no problems found.",
     "intermediates": ("no problems",),
     "on_answer": "100", "off_holds": True, "on_holds": True},
    {"id": "s09_scaffold_loss_invented", "strategy": "self_check",
     "intent": "factual_query",
     "question": "What is the capital of Finland?",
     "expected": "helsinki",
     "off_answer": "helsinki",
     "working": "Check: the draft says helsinki; the check "
                "claims the capital moved — unsupported claim.",
     "intermediates": ("unsupported claim",),
     "on_answer": "turku", "off_holds": True, "on_holds": False},
    {"id": "s10_sides_draft", "strategy": "self_check",
     "intent": "factual_query",
     "question": "How many sides does a hexagon have?",
     "expected": "6",
     "off_answer": "5",
     "working": "Check: a hexagon has 6 sides — wrong count.",
     "intermediates": ("wrong count",),
     "on_answer": "6", "off_holds": False, "on_holds": True},
    {"id": "s11_clean_author", "strategy": "self_check",
     "intent": "editing_refinement",
     "question": "Who wrote the edda compilation attributed "
                 "to snorri sturluson — snorri or saxo?",
     "expected": "snorri",
     "off_answer": "snorri",
     "working": "Check: the draft is clean, no problems found.",
     "intermediates": ("no problems",),
     "on_answer": "snorri", "off_holds": True, "on_holds": True},
    {"id": "s12_continent_draft", "strategy": "self_check",
     "intent": "factual_query",
     "question": "On which continent is the sahara?",
     "expected": "africa",
     "off_answer": "asia",
     "working": "Check: the sahara is in africa — wrong "
                "continent.",
     "intermediates": ("wrong continent",),
     "on_answer": "africa", "off_holds": False, "on_holds": True},
)

ALL_SUITE_CASES: tuple[dict[str, Any], ...] = (
    CHAIN_CASES + PLAN_CASES + CHECK_CASES
)

SUITE_AXES: tuple[dict[str, Any], ...] = (
    {"id": "chain_of_thought", "cases": len(CHAIN_CASES),
     "strategy": "chain_of_thought",
     "scorer": "filed intermediates in the scripted reasoning "
               "+ exact answer equality; strategy + prompts by "
               "TF-022 select_strategy / PromptBuilder (reused)",
     "computes": "OFF: fraction of scripted direct answers "
                 "equal to the known answer. ON: fraction "
                 "where the reasoning carries every filed "
                 "intermediate and the final answer is exact"},
    {"id": "plan_then_answer", "cases": len(PLAN_CASES),
     "strategy": "plan_then_answer",
     "scorer": "filed intermediates in the scripted plan, in "
               "order, + exact sequence equality; strategy + "
               "prompts by TF-022 (reused)",
     "computes": "OFF: fraction of scripted direct sequences "
                 "equal to the known order. ON: fraction where "
                 "the plan carries the filed steps in order "
                 "and the executed answer is exact"},
    {"id": "self_check", "cases": len(CHECK_CASES),
     "strategy": "self_check",
     "scorer": "the scripted check naming the filed error "
               "token (or its clean verdict) + exact answer "
               "equality; strategy + prompts by TF-022 "
               "(reused)",
     "computes": "OFF: fraction of scripted drafts equal to "
                 "the known answer. ON: fraction where the "
                 "check names the filed token and the revised "
                 "answer is exact"},
)

SUITE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "three_axes_three_strategies",
     "rule": "the suite scores exactly three axes, one per "
             "TF-022 strategy — chain_of_thought, "
             "plan_then_answer, self_check — each over the "
             "intents TF-022's real selection table assigns "
             "to that strategy"},
    {"id": "delta_is_measured_never_asserted",
     "rule": "both roads are scored over the same filed "
             "cases by the same exact-answer rule, and the "
             "delta filed is on minus off as measured; "
             "no-gain cases (both roads equal) and "
             "scaffold-loss cases (off holds, on fails) "
             "are filed by name in every axis — a suite "
             "that hid them would be asserting, not "
             "measuring"},
    {"id": "on_requires_the_working",
     "rule": "an ON case holds only when the scripted "
             "working material carries the filed "
             "intermediates (in order, on the plan axis) "
             "AND the scaffolded answer is exactly right: "
             "the right answer without the scaffold's "
             "work does not score as scaffold value"},
    {"id": "machinery_reused_never_copied",
     "rule": "the strategy for every case is TF-022's "
             "select_strategy, the scaffold is TF-021's "
             "ScaffoldBuilder, and both roads' prompts are "
             "PromptBuilder's real builders — the checker "
             "runs scaffold_prompts_for() over every case; "
             "this module files cases and scorers, not a "
             "second scaffold"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty axis or an empty suite scores "
             "0.0 and files REVIEW; PASS requires a "
             "non-empty run with every axis's ON score at "
             "or above AXIS_PASS_THRESHOLD and a positive "
             "overall delta (the TF-122 / TF-123 "
             "precedent, with the delta leg added — this "
             "suite's whole claim is the delta)"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every case is invented for this suite "
             "(TF-017): known-answer schoolbook problems, "
             "never canon, and data/ was never opened"},
    {"id": "failing_shapes_are_filed_not_hidden",
     "rule": "the corpora file the failing shapes by "
             "name (direct slips, reasoning slips, plans "
             "missing a step, a plan locking a wrong "
             "order, a check missing an error, a check "
             "inventing a problem, a transcription slip) "
             "with their on/off expectations — the suite "
             "scores failures on both roads, not just "
             "passes"},
)


# ── The real-code prompt leg ────────────────────────────────────────────────

def scaffold_prompts_for(case: dict[str, Any]) -> dict[str, Any]:
    """Build a case's OFF and ON prompts through the real
    builders: the OFF prompt is TF-021's single-pass
    candidate prompt for the intent's first candidate
    mode; the ON prompts are TF-022's strategy step
    prompts in order, chaining the case's working
    material as prior output for the steps that
    require it."""
    sketch = InputSketch(
        raw_text=case["question"], intent=case["intent"],
        topic="depth suite fixture", tone_in="neutral",
        response_mode=_INTENT_TO_MODE[case["intent"]], urgency=0.4)
    scaffold = ScaffoldBuilder().build(sketch, MemoryActivationBundle())
    prompts = PromptBuilder()
    off_prompt = prompts.build_candidate_prompt(
        sketch, scaffold, scaffold.candidate_modes[0], [])
    strategy = select_strategy(case["intent"])
    step_prompts: list[str] = []
    if strategy is not None:
        for step in strategy.steps:
            prior = case["working"] if step.requires_prior else ""
            step_prompts.append(prompts.build_strategy_prompt(
                sketch, scaffold, step, [], prior_text=prior))
    return {"off_prompt": off_prompt, "strategy": strategy,
            "step_prompts": step_prompts,
            "candidate_mode": scaffold.candidate_modes[0]}


# ── The scorers ─────────────────────────────────────────────────────────────

def _off_holds(case: dict[str, Any]) -> bool:
    return _answer_correct(case["off_answer"], case["expected"])


def _on_holds(case: dict[str, Any], ordered: bool = False) -> bool:
    strategy = select_strategy(case["intent"])
    if strategy is None or strategy.name != case["strategy"]:
        return False
    working_norm = _norm(case["working"])
    if ordered:
        pos = -1
        for phrase in case["intermediates"]:
            idx = f" {working_norm} ".find(f" {_norm(phrase)} ", pos + 1)
            if idx <= pos:
                return False
            pos = idx
    elif not all(_contains(working_norm, p)
                 for p in case["intermediates"]):
        return False
    return _answer_correct(case["on_answer"], case["expected"])


def _score_axis(cases, ordered: bool = False) -> dict[str, Any]:
    cases = list(cases or [])
    off = sum(int(_off_holds(c)) for c in cases)
    on = sum(int(_on_holds(c, ordered=ordered)) for c in cases)
    total = len(cases)
    off_score = (off / total) if total else 0.0
    on_score = (on / total) if total else 0.0
    return {"cases": total, "off_held": off, "on_held": on,
            "off_score": off_score, "on_score": on_score,
            "delta": on_score - off_score}


def score_chain_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the chain-of-thought axis, both roads."""
    result = _score_axis(CHAIN_CASES if cases is None else cases)
    return {"axis": "chain_of_thought", **result}


def score_plan_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the plan-then-answer axis, both roads."""
    result = _score_axis(
        PLAN_CASES if cases is None else cases, ordered=True)
    return {"axis": "plan_then_answer", **result}


def score_check_cases(
    cases: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the self-check axis, both roads."""
    result = _score_axis(CHECK_CASES if cases is None else cases)
    return {"axis": "self_check", **result}


# ── The suite run ───────────────────────────────────────────────────────────

@dataclass
class DepthSuiteResult:
    """One full suite run: per-axis OFF / ON / delta."""

    chain_off: float
    chain_on: float
    plan_off: float
    plan_on: float
    check_off: float
    check_on: float
    overall_off: float
    overall_on: float
    delta: float
    total_cases: int
    per_axis: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS only on a non-empty run with every axis's
        ON score passing and a positive measured delta."""
        if self.total_cases == 0:
            return "REVIEW"
        if min(self.chain_on, self.plan_on, self.check_on) \
                >= AXIS_PASS_THRESHOLD and self.delta > 0.0:
            return "PASS"
        return "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_depth_suite(
    chain_cases: Any = None,
    plan_cases: Any = None,
    check_cases: Any = None,
) -> DepthSuiteResult:
    """Run the whole suite (filed corpora by default).

    Any axis may be given an explicit corpus — including
    an empty one, which scores that axis 0.0 and keeps
    the suite at REVIEW (no vacuous pass anywhere).
    """
    chain = score_chain_cases(chain_cases)
    plan = score_plan_cases(plan_cases)
    check = score_check_cases(check_cases)
    per_axis = [chain, plan, check]
    total = sum(a["cases"] for a in per_axis)
    overall_off = (
        sum(a["off_score"] for a in per_axis) / len(per_axis)
    ) if total else 0.0
    overall_on = (
        sum(a["on_score"] for a in per_axis) / len(per_axis)
    ) if total else 0.0
    return DepthSuiteResult(
        chain_off=chain["off_score"], chain_on=chain["on_score"],
        plan_off=plan["off_score"], plan_on=plan["on_score"],
        check_off=check["off_score"], check_on=check["on_score"],
        overall_off=overall_off, overall_on=overall_on,
        delta=overall_on - overall_off, total_cases=total,
        per_axis=per_axis)


# ── The live checker ────────────────────────────────────────────────────────

def depth_suite_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    axis_ids = [a["id"] for a in SUITE_AXES]
    if axis_ids != ["chain_of_thought", "plan_then_answer", "self_check"]:
        problems.append(f"SUITE_AXES changed shape: {axis_ids}")
    rule_ids = [r["id"] for r in SUITE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SUITE_RULES ids")
    for needed in ("three_axes_three_strategies",
                   "delta_is_measured_never_asserted",
                   "on_requires_the_working",
                   "machinery_reused_never_copied",
                   "no_vacuous_pass_anywhere",
                   "fixtures_are_synthetic_only",
                   "failing_shapes_are_filed_not_hidden"):
        if needed not in rule_ids:
            problems.append(f"SUITE_RULES lost {needed}")

    # TF-022's registry, bound live: exactly the three
    # filed strategies, each ending in one final step.
    registry = strategy_registry()
    if set(registry) != {"chain_of_thought", "plan_then_answer",
                         "self_check"}:
        problems.append(f"strategy registry drifted: {sorted(registry)}")
    for name, strategy in registry.items():
        finals = [s for s in strategy.steps
                  if s.produces == "final_answer"]
        if len(finals) != 1 or strategy.steps[-1].produces \
                != "final_answer":
            problems.append(f"strategy {name} lost its one final step")
        if strategy.steps[0].requires_prior or not all(
                s.requires_prior for s in strategy.steps[1:]):
            problems.append(f"strategy {name} lost its prior contract")

    # Corpora: counts, balance, and filed expectations true.
    for name, cases, off_want, on_want, ordered in (
            ("CHAIN_CASES", CHAIN_CASES, 5, 10, False),
            ("PLAN_CASES", PLAN_CASES, 6, 9, True),
            ("CHECK_CASES", CHECK_CASES, 6, 10, False)):
        if len(cases) != CASES_PER_AXIS:
            problems.append(f"{name} {len(cases)} != {CASES_PER_AXIS}")
        if sum(c["off_holds"] for c in cases) != off_want:
            problems.append(
                f"{name} files {sum(c['off_holds'] for c in cases)} "
                f"off-holding (want {off_want})")
        if sum(c["on_holds"] for c in cases) != on_want:
            problems.append(
                f"{name} files {sum(c['on_holds'] for c in cases)} "
                f"on-holding (want {on_want})")
        for case in cases:
            if _off_holds(case) != bool(case["off_holds"]):
                problems.append(
                    f"off expectation disagrees with the scorer: "
                    f"{case['id']}")
            if _on_holds(case, ordered=ordered) != bool(case["on_holds"]):
                problems.append(
                    f"on expectation disagrees with the scorer: "
                    f"{case['id']}")
    case_ids = [c["id"] for c in ALL_SUITE_CASES]
    if len(case_ids) != len(set(case_ids)):
        problems.append("duplicate case ids across the corpora")

    # The real-code leg over every case: selection,
    # OFF prompt shape, ON step prompts in order.
    for case in ALL_SUITE_CASES:
        try:
            built = scaffold_prompts_for(case)
        except Exception as exc:  # a leg that cannot run is a finding
            problems.append(
                f"scaffold prompts failed for {case['id']}: {exc!r}")
            continue
        strategy = built["strategy"]
        if strategy is None or strategy.name != case["strategy"]:
            problems.append(
                f"select_strategy({case['intent']}) != filed "
                f"{case['strategy']} for {case['id']}")
            continue
        if "Working so far" in built["off_prompt"]:
            problems.append(
                f"the OFF prompt carries working material: {case['id']}")
        if len(built["step_prompts"]) != len(strategy.steps):
            problems.append(
                f"step prompts != strategy steps: {case['id']}")
        for step, prompt in zip(strategy.steps, built["step_prompts"]):
            if not prompt.startswith(step.instruction):
                problems.append(
                    f"step prompt does not open with its "
                    f"instruction: {case['id']}/{step.name}")
            if step.requires_prior and "Working so far" not in prompt:
                problems.append(
                    f"a prior-consuming step prompt carries no "
                    f"working section: {case['id']}/{step.name}")

    # The D-BENCH-10 probe, pinned: the filed loss and
    # no-gain cases score exactly as filed (the delta
    # is measured over honest failures, not a sweep).
    for case_id, off_want, on_want in (
            ("c11_scaffold_loss_slip", True, False),
            ("c12_no_gain_transcription", False, False),
            ("p11_scaffold_loss_order", True, False),
            ("s09_scaffold_loss_invented", True, False),
            ("s06_check_misses_error", False, False)):
        case = [c for c in ALL_SUITE_CASES if c["id"] == case_id]
        if not case:
            problems.append(f"the filed failure case {case_id} is gone")
            continue
        ordered = case[0] in PLAN_CASES
        if _off_holds(case[0]) != off_want or \
                _on_holds(case[0], ordered=ordered) != on_want:
            problems.append(
                f"failure case {case_id} no longer scores as filed")

    # A fresh live run has the filed shape; empty is 0.0 / REVIEW.
    result = run_depth_suite()
    if result.total_cases != len(ALL_SUITE_CASES):
        problems.append(
            f"live run totals {result.total_cases} != corpus "
            f"{len(ALL_SUITE_CASES)}")
    if result.total_cases == 0 or result.overall_on <= 0.0:
        problems.append("the live suite run scores nothing")
    if result.delta <= 0.0:
        problems.append("the live measured delta is not positive")
    empty = run_depth_suite([], [], [])
    if empty.total_cases != 0 or empty.overall_on != 0.0 \
            or empty.overall_off != 0.0 or empty.verdict != "REVIEW" \
            or empty.passes:
        problems.append("an empty suite does not file 0.0 / REVIEW")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / DEPTH_SUITE_DOC
    if not doc.exists():
        problems.append(f"the write-up {DEPTH_SUITE_DOC} is missing")
    return problems


def depth_suite_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_depth_suite()
    return {
        "format": DEPTH_SUITE_FORMAT,
        "doc": DEPTH_SUITE_DOC,
        "axis_pass_threshold": AXIS_PASS_THRESHOLD,
        "axes": [dict(a) for a in SUITE_AXES],
        "case_counts": {
            "chain_of_thought": len(CHAIN_CASES),
            "plan_then_answer": len(PLAN_CASES),
            "self_check": len(CHECK_CASES),
            "total": result.total_cases,
        },
        "measured": {
            "chain_off": result.chain_off, "chain_on": result.chain_on,
            "plan_off": result.plan_off, "plan_on": result.plan_on,
            "check_off": result.check_off, "check_on": result.check_on,
            "overall_off": result.overall_off,
            "overall_on": result.overall_on,
            "delta": result.delta,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in SUITE_RULES],
    }


__all__ = [
    "ALL_SUITE_CASES",
    "AXIS_PASS_THRESHOLD",
    "CASES_PER_AXIS",
    "CHAIN_CASES",
    "CHECK_CASES",
    "DEPTH_SUITE_DOC",
    "DEPTH_SUITE_FORMAT",
    "PLAN_CASES",
    "SUITE_AXES",
    "SUITE_RULES",
    "DepthSuiteResult",
    "depth_suite_as_data",
    "depth_suite_problems",
    "run_depth_suite",
    "scaffold_prompts_for",
    "score_chain_cases",
    "score_check_cases",
    "score_plan_cases",
]
