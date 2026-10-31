# ThoughtForge Router — The Decision Table (TF-020)

The cognition router's decision table, made explicit: every rule
`InputRouter.route` applies, every row tested. Drawn from — and now
partly written back into — `src/thoughtforge/cognition/router.py`.
Mapped and hardened 2026-10-30 on branch
`yrsa/thoughtforge-road-build`. The tests are
`tests/test_tf020_router_decision_table.py`; every example below was
measured live on this tree, before and after the hardening.

`route(raw_text, thread_state=None)` returns an `InputSketch` with
seven derived fields: `intent`, `tone_in`, `response_mode`, `topic`,
`memory_triggers`, `urgency`, `retrieval_path`. The router is
heuristic-only: regex signal tables, no model call, deterministic —
the same input always yields the same sketch (pinned by test).

## 1. The decision procedure

1. **Normalise.** `raw_text.strip().lower()` — all matching happens on
   lower-cased text. (Consequence: every signal pattern must be a
   lower-case literal. See §7, the bug this slice fixed.)
2. **Score intent and tone independently.** For each row of the
   signal table, count how many of its patterns match. A pattern
   counts **once** no matter how often it fires in the text —
   repetitions never raise a score.
3. **Pick the winner.** Highest distinct-pattern count wins. **Ties
   break by precedence**: the row that appears earlier in its signal
   table. Table order is now declared in code as `_INTENT_PRECEDENCE`
   / `_TONE_PRECEDENCE` (derived from the tables, so they cannot
   drift apart).
4. **Fallbacks.** No intent signal at all → `light_conversation`.
   No tone signal at all → `neutral`.
5. **Derive the rest** from the winning intent/tone by the fixed
   tables in §3–§5; extract `topic` and `memory_triggers` by §6.

## 2. The intent table

Precedence is top-to-bottom. Signals shown are examples, not the
whole row — the code's `_INTENT_SIGNALS` is the exhaustive list.

| # | Intent | Signals include | Mode | Retrieval | Urgency bonus |
|---|---|---|---|---|---|
| 1 | `technical_debugging` | error, exception, crash, traceback, bug, broken, not working | `structured_technical` | hybrid | +0.15 |
| 2 | `technical_spec_request` | spec, design, schema, api, module, class, build, refactor | `structured_technical` | sql | — |
| 3 | `editing_refinement` | rewrite, edit, polish, shorter, more concise, tighten | `concise_editorial` | hybrid | — |
| 4 | `brainstorming` | ideas, brainstorm, options, alternatives, what if, creative | `creative_brainstorm` | vector | — |
| 5 | `naming_or_branding` | name … for, names, branding, slogan, what … call | `creative_brainstorm` | hybrid | — |
| 6 | `planning_request` | plan, roadmap, steps, phases, how do i/we, schedule | `structured_technical` | sql | — |
| 7 | `emotional_support` | tired, exhausted, stressed, anxious, overwhelmed, i don't know what/how | `calm_supportive` | vector | +0.10 |
| 8 | `factual_query` | what is, who is, when did/was/is, how does, define, explain | `factual_direct` | sql | — |
| — | `light_conversation` | *(fallback: no signal matched)* | `light_natural_chat` | vector | — |

Worked rows (measured):

- "My function throws an error: AttributeError on line 12" →
  `technical_debugging` (error is one distinct pattern; nothing else
  scores).
- "error error error, schema design architecture" →
  `technical_spec_request`, 3 distinct patterns to debugging's 1 —
  **repetition does not score**; precedence only breaks ties.
- "error in the schema" → `technical_debugging`: 1–1 tie, row 1
  beats row 2. (The pre-TF-020 code comment claimed debugging wins
  "if error signals dominate"; the truth, now documented in code,
  is that it also wins every tie.)
- "what is the plan" → `planning_request`: 1–1 tie (plan / what is),
  row 6 beats row 8.
- "suggest names for a coffee shop" → `brainstorming`, not naming:
  1–1 tie (suggests / names), row 4 beats row 5.
- "" (empty or whitespace) → `light_conversation`, tone `neutral`,
  topic falls back to "light conversation", triggers `[]`.

## 3. The tone table

| # | Tone | Signals include | Urgency bonus |
|---|---|---|---|
| 1 | `frustrated` | ugh, why won't/can't, doesn't work, fed up, failing … again | +0.20 |
| 2 | `tired` | tired, exhausted, drained, burnt out, can't focus | +0.10 |
| 3 | `urgent` | asap, urgent, quickly, right now, deadline, today | +0.40 |
| 4 | `curious` | wonder, curious, interesting, i've been thinking, what if | — |
| 5 | `playful` | lol, haha, funny, joke, lmao, xd | — |
| 6 | `focused` | need to, must, let's start, focus, specifically, exactly | — |
| — | `neutral` | *(fallback: no signal matched)* | — |

Tone ties bite harder than they look, because the winning tone also
sets the urgency bonus: "I am tired, need this asap" scores tired 1,
urgent 1 → **tired** wins (row 2 beats row 3), so urgency takes the
+0.10 tired bonus, not the +0.40 urgent bonus. Measured: that input
routes `emotional_support / tired / 0.50`, not 0.80.

## 4. Response mode and retrieval path

Both are pure functions of the intent (columns in §2's table):

- **Mode** — the `_INTENT_TO_MODE` lookup; all 9 intents are keys
  (completeness-tested). Three intents share `structured_technical`;
  brainstorming and naming share `creative_brainstorm`.
- **Retrieval path** — set membership: {factual, spec, planning} →
  `sql`; {emotional, brainstorming, light conversation} → `vector`;
  the remaining three {debugging, editing, naming} → `hybrid`. The
  two sets are disjoint and, with the hybrid remainder, partition
  the taxonomy (tested). A caller-supplied `retrieval_path` argument
  to `ThoughtForgeCore.think()` still overrides the sketch after
  routing (CORE_LOOP §3).

## 5. Urgency arithmetic

`urgency = 0.30 baseline + tone bonus + intent bonus`, capped at
1.0 — now the `_URGENCY_BASELINE` / `_TONE_URGENCY_BONUS` /
`_INTENT_URGENCY_BONUS` / `_URGENCY_CAP` tables in code. Only three
tones and two intents carry a bonus (§2–§3); every other row adds 0.

Measured rows: baseline factual 0.30 · urgent tone alone 0.70 ·
frustrated alone 0.50 · tired alone 0.40 · debugging alone 0.45 ·
emotional alone 0.40 · frustrated+debugging 0.65 · tired+emotional
0.50 · urgent+emotional 0.80 · urgent+debugging **0.85 — the highest
reachable value**. The 1.0 cap can never bind today (0.30 + 0.40 +
0.15 = 0.85); it is a defensive bound, recorded as such in code.
Downstream note: 0.80 is the scaffold depth threshold (CORE_LOOP
§4), so exactly two urgency rows — urgent+emotional and
urgent+debugging — can push a turn over it.

## 6. Topic and memory triggers

**Topic.** Strip one leading prefix from a fixed list (`what is`,
`who is`, `how does`, `tell me about`, `explain`, `define`, `write`,
`build`, `create`, `fix`, `help with`), take the first 6 words of
what remains; if nothing remains, the topic is the intent name with
underscores spaced ("light conversation"). Measured: "What is a
ferry?" → `a ferry?`; a 8-word opener truncates at 6 words.

**Memory triggers.** From the raw text: lower-case, strip
punctuation, drop a fixed stopword list and every word of length
≤ 2, keep order. Then the pinned quirks:

- The **cap of 8 applies before de-duplication** — a repeated term
  consumes a slot. Measured: "lantern hollow lantern keeper wrenna
  sails harbor mist fjord rune stone" yields 7 unique message
  triggers, not 8 (`fjord` is cut by the pre-dedup cap).
- Up to **3 thread-state tags** append after the message triggers,
  skipping any already present; the whole list is then **capped at
  10**.

**Ruling on CORE_LOOP §8-F5** (triggers are computed but nothing in
the loop consumes them — retrieval queries with `raw_text`): the
router's contract is extraction only, and that is correct for this
layer — the sketch is the router's published output and other
consumers (scaffold cues, future memory work) read it from there.
Whether *retrieval* should consume the triggers is a knowledge-layer
question and stays owned by **TF-041**, which now inherits this
table as the extraction contract it would consume.

## 7. Bug found and fixed in this slice: dead patterns

`route()` lower-cases the input before matching — so any signal
pattern containing an upper-case letter could **never** match. Five
such patterns sat in the tables:

| Pattern (before) | Row | State before TF-020 |
|---|---|---|
| `\bAPI\b` | spec intent | dead — "api" was not a spec signal at all |
| `\bhow (do\|should\|can) (I\|we)\b` | planning intent | half-dead — only the "we" branch could match |
| `\bi don'?t know…` written with capital I | emotional intent | dead |
| `\bI'?ve been thinking\b` | curious tone | dead |
| `\bTraceback\b` | debugging intent | dead duplicate of its own `\btraceback\b` twin |

Fix: the four live signals lower-cased; the dead duplicate removed
(behaviour identical — its twin already matched everything the
router can see). Measured behaviour changes, all pinned by tests:
"tell me about the api" now routes `technical_spec_request` (was
factual); "how do i plan the phases" routes `planning_request` on
the strength of that signal too; "i don't know what to do anymore"
routes `emotional_support`; "i've been thinking about the old hall"
reads tone `curious`. A regression guard test now asserts **every**
pattern in both signal tables is lower-case, so this class cannot
return silently.

Also made explicit in code (behaviour preserved, verified by the
pre-existing phase-3 router tests passing unchanged): the
precedence-based tie-break in `_classify_intent` / `_classify_tone`
(replacing bare `max()` on dict order) and the urgency tables
(replacing inline conditionals in `_estimate_urgency`).

**Observation, not fixed here:** `InputSketch`'s field comments in
`knowledge/models.py` describe intents and modes the router never
emits ("question", "conversational"). That is model-doc drift for
TF-034 (core API freeze) / TF-039 (core docs) to reconcile; the
router's true taxonomy is this table.

## 8. Verification for this slice

- `tests/test_tf020_router_decision_table.py`: 92 tests — every
  intent row, tone row, mode row, and retrieval row; scoring and
  tie-break rows for both classifiers; every urgency row incl. the
  0.85 maximum; topic and trigger rows incl. the cap-before-dedup
  pin; the dead-pattern regression guard; taxonomy completeness.
- All probes in §§2–6 measured live against the tree (router alone),
  before the code changes for the baseline and after for the table.
- Full suite re-run at banking; counts recorded in
  `CAMPAIGN_STATE.md`.
