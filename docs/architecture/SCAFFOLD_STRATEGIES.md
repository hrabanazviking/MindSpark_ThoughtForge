# ThoughtForge Architecture — The Scaffold Strategies (TF-022)

Scaffold Library II: the three multi-step reasoning scaffolds —
chain-of-thought, plan-then-answer, and self-check — brought to one
interface. Built 2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The machine-readable registry is
`src/thoughtforge/cognition/scaffold_strategies.py`; every step's
prompt is built by the one new builder method,
`PromptBuilder.build_strategy_prompt`. TF-021 catalogued the
scaffolds the tree already had (one steering object per intent plus
its single-pass candidate modes); this slice adds the strategies
that are **sequences over passes**, which is why they are not
candidate modes and do not enter TF-021's mode tables: a mode is one
pass, a strategy is an ordered run of passes ending in the answer.

## 1. The one interface

Every strategy is an ordered tuple of steps, and every step — in
every strategy — carries the same six fields:

| Field | Meaning |
|---|---|
| `name` | Unique across all strategies |
| `instruction` | The opener line the step's prompt starts with |
| `purpose` | What the step is for, written down (TF-021's lesson) |
| `temperature` | The heat that step generates at |
| `requires_prior` | Whether it consumes the previous steps' output |
| `produces` | `reasoning` / `plan` / `draft` / `check` / `final_answer` |

The invariants, all pinned by tests:

- **Exactly one final step.** Each strategy ends with a single step
  whose `produces` is `final_answer`, and it is last. Earlier steps
  produce working material only — reasoning, a plan, a draft, a
  check — never the text the user sees.
- **Prior-output contract, enforced.** Only the first step runs
  without prior output; every later step requires it.
  `build_strategy_prompt` raises `ValueError` both ways (a later
  step run cold; a first step handed prior output) — a strategy run
  out of order is a caller error and surfaces as one.
- **One prompt shape.** A step prompt assembles in the candidate
  prompt's order (instruction, request, intent, topic, tone, goal,
  focus, avoid, memory cues, knowledge context), plus a *Working so
  far* section (prior output, capped at 1,000 chars) when the step
  consumes it, and a closing output line: the response mode's output
  shape for the final step, "`<produces>` only" for working steps.
- **Temperatures tell the job.** Checking is the coldest step in the
  module (0.40 — verification, not generation); reasoning and
  drafting run warmest (0.60); final answers sit at 0.50–0.55, the
  candidate registry's cold band.

What this interface does *not* yet do: the core loop does not run
strategies automatically. Wiring strategy execution into
`ThoughtForgeCore.think` — pass budgets, streaming, traces — belongs
to the slices that own those surfaces (TF-026/TF-033/TF-038), and
TF-032 owns making strategies selectable per conversation through
config. This slice delivers the strategies as a tested, selectable
library on one interface, which is what the roadmap's TF-022 asks
for; claiming the loop runs them today would be false.

## 2. The three strategies

### chain_of_thought — for technical_debugging, technical_spec_request

*Reason in short numbered steps first, then answer from the
reasoning — the intermediate steps a small model would otherwise
skip are made explicit, and kept out of the answer.*

| Step | Temp | Produces | Purpose |
|---|---|---|---|
| reason_stepwise | 0.60 | reasoning | Externalises the causal chain (symptom → cause → fix, or requirement → design) so the answer step builds on it instead of guessing |
| answer_from_reasoning | 0.50 | final_answer | Converts the reasoning into the clean final response; the working stays out of what the user reads |

### plan_then_answer — for planning_request, brainstorming

*Decide the shape before generating: a short ordered plan first,
then the answer executed against that plan.*

| Step | Temp | Produces | Purpose |
|---|---|---|---|
| outline_plan | 0.55 | plan | Fixes order, coverage, and — for brainstorming — the distinct angles, before any answer text exists to anchor on |
| execute_plan | 0.55 | final_answer | Writes the answer against the plan, so order and completeness come from the plan step rather than improvisation |

### self_check — for factual_query, editing_refinement

*Draft, then check the draft against the request, the goal, and the
facts given, then rewrite fixing what the check found — for intents
whose answers can actually be checked.*

| Step | Temp | Produces | Purpose |
|---|---|---|---|
| draft_answer | 0.60 | draft | A complete first attempt, written to be checked — specificity is what makes checking possible |
| check_answer | 0.40 | check | Verification, not generation: a problem list (wrong / missing / unsupported), never a new answer |
| revise_from_check | 0.50 | final_answer | The only step whose text the user sees; its prior input is the draft and the check together, supplied by the caller as one working text |

## 3. Selection — one table, derived, never duplicated

`select_strategy(intent)` is derived from the registry's
`applies_to` fields (not a second hand-kept table, so the two cannot
drift). Each router intent appears in at most one strategy:

| Intent | Strategy | Why |
|---|---|---|
| technical_debugging | chain_of_thought | Causal reasoning is the work |
| technical_spec_request | chain_of_thought | Requirement → design is a chain |
| planning_request | plan_then_answer | The answer *is* an ordered plan |
| brainstorming | plan_then_answer | Angles decided first stay distinct |
| factual_query | self_check | Claims can be checked against facts |
| editing_refinement | self_check | A rewrite can be checked against its source |
| emotional_support | — none — | Light depth; a reasoning voice damages the tone |
| light_conversation | — none — | Light depth; extra passes cost more than they add |
| naming_or_branding | — none — | Light depth; single-pass divergence already serves it |
| fallback / unknown | — none — | An unclassified turn gets one honest pass (TF-021's rule) |

## 4. Cost in tokens — measured, not estimated

**Method.** As TF-021: word-tokens (`len(text.split())`), prompts
built by the real builder on the canonical sketch of each strategy's
first intent (fixed synthetic request, empty bundle, no cues); later
steps carry the fixed three-word synthetic prior output
(`CANONICAL_PRIOR_TEXT`). Measured this slice:

| Strategy | Steps (chars / words) | Total prompt cost |
|---|---|---|
| chain_of_thought | reason_stepwise 406 / 64 → answer_from_reasoning 495 / 74 | 901 chars / 138 words |
| plan_then_answer | outline_plan 372 / 57 → execute_plan 441 / 65 | 813 chars / 122 words |
| self_check | draft_answer 359 / 56 → check_answer 497 / 80 → revise_from_check 448 / 68 | 1,304 chars / 204 words |

Findings:

- **Strategies cost more than a plain turn — honestly.** A normal
  two-pass candidate turn's prompt overhead is 99–119 words
  (TF-021 §4). A strategy run costs 122–204 words of prompt alone,
  and self_check's three passes are the dearest thing in the
  cognition layer. That is the price of depth on a small model, and
  it is why selection (§3) refuses strategies to light intents.
- **Later steps cost more than first steps** in every strategy:
  they carry the Working-so-far section and (for check_answer) the
  longest instruction in the module. Real turns add the real prior
  output on top — the 1,000-char cap in the builder is the bound.
- **Generation cost is on top of this**, as in TF-021: each step is
  a generation. A self_check turn is three model calls before
  salvage; TF-033's latency budgets must price strategies per
  profile, and phone_low may prove too small for self_check — that
  verdict belongs to TF-037's floor test, not to this slice.

## 5. The ruling — CORE_LOOP §8-F6 closed

**Finding.** Two fragment systems run per turn with two thresholds:
the core keeps fragments at composite ≥ 0.25, salvage at ≥ 0.54.
Read as one metric, that looks like drift. It is not one metric:

- The core's 0.25 gates `FragmentScores.composite` — relevance
  .30 + clarity .18 + specificity .18 + usefulness .20 +
  personality .14 − genericness .10 (`knowledge/models.py`) — and
  decides only which fragments are **kept on the turn record** as
  provenance for the final answer.
- Salvage's 0.54 gates a different formula — length .45 + citation
  .55, from the Retrieval_and_Scoring_Spec §30 — and decides which
  fragments are **worth spending a refine generation on**.

**Measured proof they are different instruments.** One synthetic
sentence ("The lantern keeper trims the wick at dusk every single
evening without fail."), scored both ways this slice: core
composite **0.285** (kept for provenance at 0.25), salvage score
**0.446** (refused for refinement at 0.54). Both verdicts are
correct for their job: the sentence is relevant enough to record,
and too short / citation-free to justify a refine pass.

**Ruling.** Both thresholds stay. "Unifying" the numbers would
either flood the turn record or starve the refine pass, because it
would be tuning one instrument with the other's scale. What is
unified is the *documentation*: both constants now carry a comment
naming the other system, its formula, and its job (core.py,
salvage.py), and tests in this slice pin both values, both
formulas' verdicts on the measured sentence, and the fact that the
thresholds differ. F6 is closed as *not a defect — two instruments,
now labelled as such*.

## 6. Verification for this slice

- The three strategies, the interface invariants, the selection
  table, the prompt contract (including both `ValueError` cases),
  independently recomputed costs, the step band, and the F6 pins:
  `tests/test_tf022_scaffold_strategies.py`.
- Full suite re-run at banking; counts recorded in
  `CAMPAIGN_STATE.md` (TF-014 zero line held or the delta named).
