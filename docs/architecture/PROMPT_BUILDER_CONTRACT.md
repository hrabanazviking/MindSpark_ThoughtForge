# ThoughtForge Architecture — The Prompt Builder Contract (TF-023)

Prompt builder contract: the assembly order of every prompt
`PromptBuilder` builds, specified, capped, and golden-tested.
Built 2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The machine-readable contract is
`src/thoughtforge/cognition/prompt_contract.py` (section order per
kind, optional sections, section markers, and the
`check_prompt_order` verifier); the golden tests are
`tests/test_tf023_prompt_builder_contract.py`. Before this slice,
the order lived only inside the builder's f-strings — and two of
its silent habits were defects (§7, §8 below).

## 1. The six prompt kinds and who calls them

| Kind | Builder method | Caller(s) today |
|---|---|---|
| candidate | `build_candidate_prompt` | core's local-engine passes (≤2 modes) and the unified-backend pass (1 mode) |
| strategy | `build_strategy_prompt` | the TF-022 strategy library; not yet wired into the core loop (TF-026/032/033 own that) |
| refine | `build_refine_prompt` | `FragmentSalvage.forge` refine passes (local engine only, ≤2) |
| repair | `build_repair_prompt` | core's quality-gate repair pass (local engine only, ≤2 cues) |
| episode_summary | `build_episode_summary_prompt` | writeback machinery — built, invoked by nothing yet (TF-047 owns the wiring; CORE_LOOP §7) |
| preference_inference | `build_preference_inference_prompt` | writeback machinery — same status as episode_summary |

Plus the one support method: `extract_memory_cues(bundle,
max_cues=3)` turns a `MemoryActivationBundle` into the cue strings
the candidate/strategy/repair prompts carry (§5).

## 2. The ingredients — where the roadmap's four actually enter

The roadmap's TF-023 shorthand names the ingredients in the order
"(persona, memory, scaffold, context)". Read as a literal sequence
that is not what a prompt is, and this contract does not pretend
otherwise. What each ingredient *is* in a built prompt:

- **Persona** has **no section of its own** and is never pasted in
  raw. It reaches a prompt pre-digested, through the scaffold:
  `ScaffoldBuilder` folds the personality core's first 3 `avoid`
  entries into `scaffold.avoid` (TF-009 D-CFG-2 — the persona's
  expressive fields stay dormant in the scaffold), and the router's
  user-tone signal overlays `scaffold.tone`. If those entries do
  not survive into the prompt text, the persona does not exist for
  the model — which is exactly the defect §7 fixes.
- **Memory** rides in two sections: the *Memory cues* section
  (≤3 compressed cues from `extract_memory_cues`) and the
  *Knowledge context* section (`scaffold.fact_block`). The fact
  block itself is assembled upstream: top-8 retrieved records from
  the scaffold builder, with chat history prepended by the core
  when a history is passed (labelled inside the block, CORE_LOOP
  §4) — so "context" from earlier turns enters *inside* the memory
  ingredient, in that order: history first, then knowledge.
- **Scaffold** is the steering block: Tone / Goal / Focus / Avoid
  lines, built by `ScaffoldBuilder` from the intent/mode tables.
- **Context** (the turn itself) is the User request / Intent /
  Topic block, taken from the `InputSketch`.

## 3. Assembly order — the candidate prompt

Every generation prompt opens with its instruction, because a
small model anchors on its first line; the request follows, so the
steering is read against the ask; the output line closes, closest
to where generation begins.

| # | Section | Source | Present |
|---|---|---|---|
| 1 | opener | mode's opener (8 modes; unknown mode → "Write one short reply.") | always |
| 2 | User request | `sketch.raw_text` | always |
| 3 | Intent | `sketch.intent`, underscores → spaces | always |
| 4 | Topic | `sketch.topic` (may be empty) | always |
| 5 | Tone | **all** of `scaffold.tone` (≤4 by construction); empty → "direct, natural" | always |
| 6 | Goal | `scaffold.goal` | always |
| 7 | Focus | **all** of `scaffold.focus` (≤3 by construction); empty → "clarity" | always |
| 8 | Avoid | **all** of `scaffold.avoid`; empty → "filler" | always |
| 9 | Memory cues | first 3 cues as `- ` lines; none → "- none" | always |
| 10 | Knowledge context | `scaffold.fact_block` under a "Knowledge context:" header | only when the fact block is non-empty |
| 11 | Output | response mode's output shape + depth's length hint; unknown response mode → "2-4 sentences", unknown depth → no hint | always |

## 4. Assembly order — the other kinds

**strategy** — the candidate's order with the step's instruction as
its opener, plus one optional section: sections 1–10 as above
(instruction, request, intent, topic, tone, goal, focus, avoid,
memory cues, knowledge context), then *Working so far* (prior
steps' output, capped at 1,000 chars — present exactly when the
step consumes prior output; the builder enforces that contract both
ways, TF-022), then the Output line: the response mode's shape for
the final step, "`<produces>` only" for working steps.

**refine** — fragments first (that is the material being refined),
then a reduced steering block — no request/topic/focus/cues/fact
sections, because the fragments already carry the turn's content:

| # | Section | Notes |
|---|---|---|
| 1 | fragments | "Using these useful fragments:" + `- ` lines (§5) |
| 2 | instruction | fixed: "Write one clean response." |
| 3 | Intent | as candidate |
| 4 | Tone | **all** of `scaffold.tone`; empty → "direct, natural" |
| 5 | Goal | `scaffold.goal` |
| 6 | Avoid | **all** of `scaffold.avoid`; empty → "filler, repetition" |
| 7 | Output | response mode's shape — **no length hint** (refine has none, by its shape) |

**repair** — the deliberately narrowed last-resort prompt; see §5
for its specified narrowing. Order: opener ("Write one direct
response."), User request, Intent, Tone (first 2), Goal, Memory
cues (≤2), the fixed directive ("Avoid generic wording. Keep it
specific and useful."), Output (shape only).

**episode_summary** — instruction, User request, Final response
(truncated at 300 chars), Output ("one compact summary under 20
words"). **preference_inference** — instruction, User request,
Context, Output (the fixed "- preference summary / or / - none"
shape).

## 5. Caps, fallbacks, and the one specified narrowing

| Cap / fallback | candidate | strategy | refine | repair |
|---|---|---|---|---|
| tone entries | all (≤4) | all (≤4) | all (≤4) | **first 2, by specification** |
| focus entries | all (≤3) | all (≤3) | — no section | — no section |
| avoid entries | all | all | all | — no section; fixed directive instead |
| memory cues | ≤3 | ≤3 | — no section | **≤2, by specification** |
| knowledge context | fact block, uncapped by the builder (bounded upstream: 8 records, history ≤1,500 chars) | same | — no section | — no section |
| prior output | — | ≤1,000 chars | — | — |

- **Repair's narrowing is a specification, not a bug.** The repair
  prompt is the tight fallback for a turn that already failed the
  quality gate once: fewer cues, two tone words, no fact block to
  re-drown it in, and a single blunt directive. TF-023 *states*
  this narrowing and pins it in tests — the opposite of the silent
  truncation §7 removed elsewhere.
- **Refine fragments:** kept fragments only, at most 5, in caller
  order, blank texts skipped in the lines. If *no* fragment is
  flagged `keep`, the builder falls back to the top 3 fragments by
  composite score — rescue never refines from nothing when material
  exists.
- **`extract_memory_cues`:** records sort by type priority
  (user_preference < user_fact < response_pattern <
  active_thread_state < episode < anything else), score descending
  inside a type — **priority beats score**, so a weak preference
  outranks a strong entity record. `max_cues` is applied *before*
  empty cues are dropped: a cue-less record inside the top
  `max_cues` consumes its slot and yields nothing. That quirk is
  now specified and pinned, not accidental.
- **Unknown-value fallbacks** (all golden-tested): unknown candidate
  mode → "Write one short reply."; unknown response mode → output
  shape "2-4 sentences"; unknown depth → no length hint; empty
  tone/focus/avoid lists → the defaults named in §3/§4.

## 6. Cost in tokens — measured at this contract

**Method.** As TF-021/TF-022: word-tokens (`len(text.split())`),
prompts built by the real builder on the canonical sketches (fixed
synthetic request, empty bundle, no cues).

- **Candidate prompts:** 45–66 words per pass (TF-021 measured
  45–60; the full avoid lists of §7 cost up to +6 words on the
  dearest intent, technical_debugging). Two-pass turns therefore
  run 106–131 words of prompt overhead.
- **Strategy steps:** 59–83 words per step (TF-022's 56–80 band
  still holds); strategy totals are now chain_of_thought **150**,
  plan_then_answer **128**, self_check **213** words (TF-022:
  138 / 122 / 204) — self_check remains the dearest thing in the
  cognition layer, and the ordering of strategies by cost is
  unchanged.
- The golden fixtures of this slice (a factual turn with 3 cues
  and a one-line fact block) are *not* the canonical cost basis —
  real turns add the user's words, cues, and fact block on top,
  exactly as TF-021 §4 stated.

## 7. The defect fixed in this slice — steering that never arrived

**Finding, proven live before the fix.** `build_candidate_prompt`,
`build_strategy_prompt`, and `build_refine_prompt` joined
`scaffold.tone[:3]` / `scaffold.focus[:3]` / `scaffold.avoid[:3]`
(refine: tone `[:2]`). The scaffold's `avoid` list is built as
3 universal entries + intent extras + personality avoids, so the
`[:3]` cap passed *exactly the universal trio and nothing else*:
the intent's own avoid extras (e.g. factual_query's "guessing,
unverified claims") and the personality core's avoids — the
persona's only channel into a prompt (§2) — **never reached the
model in any generation or refine prompt**. Tone's cap likewise
dropped the user-tone overlay, which is always the 4th entry.

**Fix.** Candidate, strategy, and refine prompts now carry the
scaffold's full tone, focus, and avoid lists. The lists are bounded
where they are built (tone ≤4 by the scaffold builder's merge cap,
focus ≤3 by its intent tables, avoid = 3 universal + ≤3 intent
extras + ≤3 persona entries), so "full" is a bounded contract, not
an open door. Repair keeps its §5 narrowing, now specified.
Consequences recorded: TF-021's candidate cost band pin widened
65 → 70 (measured max is 66), and the measured costs of §6
supersede TF-021 §4 / TF-022 §4 numbers where they differ. The
fragments of proof are pinned in this slice's tests: a synthetic
personality's avoids and the "curious" tone overlay are asserted
present in built prompts, end-to-end through `ScaffoldBuilder`.

## 8. The ruling — CORE_LOOP §8-F7 closed

**Finding.** `CognitionScaffold.prompt_text` was documented as
"fully assembled prompt", set to `""` by the scaffold builder with
the comment "filled in by PromptBuilder", and assigned by nothing —
builders return prompt strings directly to their callers.

**Ruling.** The field is **removed**, not filled. A single turn
builds many prompts — up to 2 candidate passes, up to 2 refine
passes, strategy steps, a possible repair — and there is no honest
single value the field could hold; the obvious candidates (first
prompt? last prompt?) would each mislead a different reader.
Prompts belong to the callers that build them, in the order they
build them; if a consumer needs a turn's prompts exported, that is
TF-038's trace surface, designed per-pass. The removal is pinned:
`scaffold_shape()` no longer lists the field, `CognitionScaffold`
has no such attribute, and the builders are asserted not to mutate
the scaffold at all. F7 is closed as *fixed by removal*.

## 9. Verification for this slice

- Golden prompts (byte-exact) for all six kinds, section-order
  verification of every built prompt, the checker proving it can
  fail, the §7 fix proven end-to-end with a synthetic persona, the
  §5 caps and narrowing, the F7 pins, and costs recomputed
  independently: `tests/test_tf023_prompt_builder_contract.py`.
- Full suite re-run at banking; counts recorded in
  `CAMPAIGN_STATE.md` (TF-014 zero line held or the delta named).
