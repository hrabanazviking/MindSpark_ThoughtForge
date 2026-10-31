# ThoughtForge Architecture — The Scaffold Registry (TF-021)

Scaffold Library I: every scaffold the tree already has, catalogued
in one place — its shape, its purpose, and its cost in tokens.
Catalogued 2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The machine-readable registry is
`src/thoughtforge/cognition/scaffold_registry.py`. It **assembles
the live tables** — the intent tables in `cognition/scaffold.py`,
the response-mode table in `cognition/router.py`, the opener and
output-shape tables in `cognition/prompt_builder.py`, and the mode
temperatures in `cognition/core.py` — and copies none of their
values, so it cannot silently drift from the builder: its tests
(`tests/test_tf021_scaffold_registry.py`) rebuild every entry
through the real `ScaffoldBuilder` and compare. This document is
the human-readable rendering of that registry, plus the slice's
one ruling (§5). TF-022 brings the new scaffolds (chain-of-thought,
plan-then-answer, self-check) to the interface described here;
TF-023 rules on the prompt builder's contract over it.

**What a "scaffold" is in this tree.** There are not separate
scaffold classes per thinking style. A scaffold is one
`CognitionScaffold` steering object, built deterministically per
turn by `ScaffoldBuilder` from the router's intent, plus the
**candidate modes** it names: the generation passes the core runs
inside that steering. The catalogue therefore has three layers:
the shape (§1), the ten intent rows (§2), and the eight candidate
modes (§3). Costs are in §4.

## 1. The shape — `CognitionScaffold` (`knowledge/models.py`)

| Field | Type | Filled by | Content |
|---|---|---|---|
| `goal` | str | ScaffoldBuilder | One fixed sentence per intent (§2) |
| `tone` | list[str] | ScaffoldBuilder | Response-mode base (≤3) + user-tone overlay, max 4 |
| `focus` | list[str] | ScaffoldBuilder | 2–3 fixed steering words per intent |
| `avoid` | list[str] | ScaffoldBuilder | 3 universal + intent extras + personality `avoid[:3]` |
| `depth` | str | ScaffoldBuilder | `light` / `medium` / `expert` (rule per row, §2) |
| `candidate_modes` | list[str] | ScaffoldBuilder | The generation passes, in order (§3) |
| `fact_block` | str | ScaffoldBuilder (+ core) | Top-8 retrieved records, 1 line each; core may prepend chat history (CORE_LOOP §4) |
| `prompt_text` | str | **nobody** | Always `""` — CORE_LOOP §8-F7; TF-023 owns the ruling |

Two modifiers ride on top of every row below and are not part of
the row itself: the **user-tone overlay** (frustrated → +patient,
steady; tired → +gentle, clear; urgent → +focused, fast; curious →
+curious, engaged; playful → +light, playful — merged to a max of
4 tone entries) and the **personality core**, whose only scaffold
effect is appending up to its first 3 `avoid` entries (TF-009
D-CFG-2). The §2 rows are the neutral-tone, no-personality base
shapes.

## 2. The intent rows

One row per router intent, plus the builder's unknown-intent
fallback. Tone/focus/avoid are the base tables; `avoid` always
starts with the universal three — *filler, padding, repetition of
the user's question* — shown once here, not repeated per row.

| Intent (response mode) | Goal | Tone base | Focus | Avoid extras | Depth rule |
|---|---|---|---|---|---|
| technical_spec_request (structured_technical) | produce a clear, structured, implementation-ready response | direct, structured, precise | clarity, specificity, completeness | vagueness, handwaving, pseudo-code | expert < urgency 0.8, else medium |
| technical_debugging (structured_technical) | identify the likely cause and provide a concrete fix | direct, structured, precise | root cause, concrete fix, minimal change | speculation without evidence, multiple unrelated fixes | expert < urgency 0.8, else medium |
| editing_refinement (concise_editorial) | preserve meaning while improving clarity and force | sharp, natural, economical | precision, rhythm, no redundancy | meaning distortion, over-editing | medium |
| brainstorming (creative_brainstorm) | generate strong, distinct options with low repetition | energetic, open, inventive | originality, relevance, distinctiveness | generic ideas, overexplaining | medium |
| naming_or_branding (creative_brainstorm) | produce memorable, distinctive name candidates | energetic, open, inventive | memorability, fit, brevity | generic options, overly long names | light |
| planning_request (structured_technical) | produce a clear, ordered, actionable plan | direct, structured, precise | order, clarity, feasibility | vague milestones, over-engineering | expert < urgency 0.8, else medium |
| emotional_support (calm_supportive) | validate the emotion and offer one useful stabilizing thought | calm, warm, grounded | recognition, one useful thought | platitudes, preachiness, robotic tone | light |
| factual_query (factual_direct) | give an accurate, direct answer with cited sources | direct, factual, clear | accuracy, brevity, citation | guessing, unverified claims | medium |
| light_conversation (light_natural_chat) | keep the exchange natural, warm, and engaging | warm, natural, easy | naturalness, engagement | lecture mode, unsolicited advice | light |
| fallback (light_natural_chat)* | produce a useful, direct response | warm, natural, easy | clarity, usefulness | — (universal only) | medium |

\* The fallback never arrives from the router (it emits only the
nine intents); it is the builder's defensive shape for an unknown
intent string, catalogued so its behaviour is written down, not
implied.

Each row also carries an **output shape** (from the response mode,
`prompt_builder._MODE_TO_OUTPUT_SHAPE`) and a **length hint** (from
the depth): structured_technical → "markdown-ready section with
concise headings"; concise_editorial → "one refined version only";
creative_brainstorm → "3 compact distinct options"; calm_supportive
→ "3-5 sentences"; factual_direct → "1-3 sentences with source
citation"; light_natural_chat → "2-4 sentences". Length hints:
light → "Keep it short.", medium → "Keep it focused.", expert →
"Be thorough but not verbose."

## 3. The candidate modes

Eight modes exist. A mode is: an **opener** line that starts the
candidate prompt, a **temperature** the core generates at, and —
written down here for the first time — a **purpose**. The core's
local branch runs the scaffold's modes in order, at most 2 passes
(`candidate_modes[:2]`); the unified-backend branch runs only the
first mode, at a hard-coded `max_tokens=512` (CORE_LOOP §5).

| Mode | Temp | Opener | Purpose | Used by |
|---|---|---|---|---|
| strict_spec | 0.50 | Write one concise, structured technical response. | Precision pass: structured, implementation-ready; coldest in the registry | technical_spec_request, technical_debugging, planning_request, factual_query |
| implementation_friendly | 0.55 | Write one clear, implementation-ready response. | Builder's pass: a response a developer can act on directly | technical_spec_request, technical_debugging, editing_refinement, factual_query |
| editorial_sharp | 0.50 | Rewrite the text cleanly and precisely. | Editor's pass: clean rewrite, meaning preserved | editing_refinement |
| brainstorm_distinct | 0.80 | Write one strong, distinctive option. | Divergence pass: originality over safety; hottest in the registry | brainstorming, naming_or_branding |
| practical | 0.55 | Write one direct, practical response. | Grounding pass: the counterweight paired with divergent or formal first passes | brainstorming, naming_or_branding, planning_request, light_conversation, fallback |
| empathic | 0.65 | Write one short, warm, grounded supportive reply. | Support pass: validates before it advises | emotional_support |
| reflective | 0.70 | Write one thoughtful, perceptive reply. | Depth pass: emotional support's second, slower angle | emotional_support |
| playful | 0.75 | Write one light, natural reply with some warmth. | Lightness pass: light conversation's second angle | light_conversation |

No mode is an orphan (every mode is used by at least one row) and
no row names a mode the registry lacks — both pinned by tests.

## 4. Cost in tokens — measured, not estimated

**Method.** A token here is a word — the tree's own convention
(`core` sets a final's `token_count = len(text.split())`). Costs
were measured by `scaffold_registry.measure_prompt_costs()`:
every row's candidate prompts built by the real `PromptBuilder`
on a canonical sketch (fixed synthetic request "test query",
topic "test topic", neutral tone, urgency 0.4, **empty memory
bundle, no cues**), then counted. Holding the request fixed makes
the rows comparable: the differences are the scaffold's own cost.
A real turn adds the user's words in place of the 2-word request,
up to 3 memory cues, and the fact block (below).

| Intent | Pass 1 (mode: chars / words) | Pass 2 (mode: chars / words) | Both passes |
|---|---|---|---|
| technical_spec_request | strict_spec: 426 / 55 | implementation_friendly: 424 / 54 | 850 chars / 109 words |
| technical_debugging | implementation_friendly: 421 / 59 | strict_spec: 423 / 60 | 844 chars / 119 words |
| editing_refinement | editorial_sharp: 369 / 53 | implementation_friendly: 377 / 52 | 746 chars / 105 words |
| brainstorming | brainstorm_distinct: 374 / 50 | practical: 374 / 50 | 748 chars / 100 words |
| naming_or_branding | brainstorm_distinct: 357 / 50 | practical: 357 / 50 | 714 chars / 100 words |
| planning_request | strict_spec: 396 / 54 | practical: 384 / 53 | 780 chars / 107 words |
| emotional_support | empathic: 369 / 54 | reflective: 359 / 52 | 728 chars / 106 words |
| factual_query | implementation_friendly: 373 / 53 | strict_spec: 375 / 54 | 748 chars / 107 words |
| light_conversation | practical: 333 / 48 | playful: 344 / 51 | 677 chars / 99 words |
| fallback | practical: 309 / 45 | — (single pass) | 309 chars / 45 words |

Findings from the measurement:

- **The scaffold is lean and flat.** Every base candidate prompt
  costs 45–60 words; a full two-pass turn's prompt overhead is
  99–119 words. No intent is secretly expensive; the dearest row
  (technical_debugging, 119) costs only ~20% more than the
  cheapest two-pass row (light_conversation, 99).
- **Retrieval is the variable cost, not the scaffold.** Measured
  on the canonical factual row with synthetic records: 1 record
  (+ its cue) takes the first prompt from 53 → 67 words; a full
  8-record fact block (+ 3 cues) takes it to 150 words — nearly
  triple the base. Prompt budgets (TF-024/TF-033) should be set
  against the fact block, not the scaffold tables.
- **Generation cost is not in this table** — it is the model's
  output, capped only on the unified branch (512, above); the
  local branch passes no token cap at all in `GenerationParams`
  (a TF-022/TF-033 measurement point, recorded here, not fixed —
  it is a generation-budget question, not a scaffold defect).

## 5. The ruling — CORE_LOOP §8-F8 closed

**Finding.** `brainstorming` paired `brainstorm_distinct` with
itself: two local passes with identical opener and identical
temperature (0.80). Measured before the fix, the two prompts were
byte-identical (422 chars / 58 words on a live router probe) —
the second pass bought a second *sample*, not a second
*scaffold*, in the one intent whose goal is "strong, **distinct**
options with low repetition" and whose focus demands
distinctiveness. Every other row pairs two different angles.

**Ruling.** Brainstorming's second pass is now **`practical`**
(0.55, "Write one direct, practical response."): the divergent
pass proposes, the grounding pass answers with the feasible
option — originality *and* relevance, the row's own focus words.
The precedent was already in the tree: naming_or_branding, its
nearest kin, pairs exactly these two modes in this order.
Behaviour preserved where it must be: the first mode is
unchanged, so the unified-backend branch (first mode only) is
byte-for-byte unaffected; only the local second pass changes
steering. The two prompts now differ (opener + temperature),
pinned by test — their canonical lengths happen to stay equal
(374 / 50 each).

**Same defect, second seat — fixed with it.** The builder's
unknown-intent fallback was `["practical", "practical"]`, the
identical duplicate in miniature. It is now a single practical
pass: an intent nobody classified gets one honest pass, not two
copies of it. The registry-wide rule, pinned by a guard test:
**no scaffold row may repeat a candidate mode.**

Docket note: F8 was a CORE_LOOP finding with TF-021 as its
proposed owner; it is not a TF-016 `D-*` entry, so no docket row
flips — the closure is recorded here and in CORE_LOOP's terms.

## 6. Verification for this slice

- Registry, ruling, and costs all measured on the tree this
  slice: §4's numbers come from `measure_prompt_costs()` run at
  banking; the pre-fix byte-identical brainstorming prompts were
  measured by the same route before the edit (422 chars / 58
  words, router probe "I need some ideas and alternatives").
- 17 new tests in `tests/test_tf021_scaffold_registry.py`:
  shape, full builder-parity per row, depth rules at both
  urgencies, the personality-avoid boundary, the no-duplicate
  guard, the brainstorming pair + prompt-difference + first-mode
  preservation, mode-registry completeness and reverse index,
  independently recomputed costs, the 40–65-word base band, and
  this doc naming every intent and mode.
- Full suite re-run at banking; counts recorded in
  `CAMPAIGN_STATE.md` (TF-014 zero line held or the delta named).
