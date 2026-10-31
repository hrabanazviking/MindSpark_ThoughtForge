# TF-029 — The Will Layer: Goal/Intent Persistence Across Turns

*Part II, slice TF-029. Code: `src/thoughtforge/cognition/will.py`
(`StandingGoal`, `GoalLedger`, `inject_will`), wired minimally into
`ThoughtForgeCore._run_turn`. Tests:
`tests/test_tf029_will_layer.py` (26 tests).*

## 1. The gap, as found

The roadmap promises small models "depth, presence, and **will**".
Measured against the tree before this slice, will did not exist:

- The scaffold's goal is re-derived from **this turn's intent
  alone** every turn (`scaffold._INTENT_TO_GOAL`). A planning turn
  followed by a small factual question inside the same project
  loses the project's goal entirely — the forge answers the small
  question as if the conversation had no purpose.
- The one record shaped for persistence, `ActiveThreadStateRecord`
  (topic, tags, open loops, priority, `expires_after_turns=10`,
  `cool_down()`), is loaded by the core every turn and handed to
  the router, which reads only its first 3 tags as memory
  triggers. Nothing updates it, nothing carries a goal in it, and
  spec §17's thread update rules were implemented nowhere
  (TF-019/CORE_LOOP).
- `FragmentScores`/`FinalResponseScores` carry a `goal_fit`
  component — but it scores a candidate against the *per-turn*
  scaffold goal only. There was no conversation-level goal for
  anything to fit.

## 2. The specification (from the hints + the plan)

A **standing goal** is the conversation-level counterpart of the
per-turn scaffold goal:

| Hint in the tree | What TF-029 made of it |
|---|---|
| Scaffold goal per intent (scaffold.py) | The standing goal records the response goal in force when it was set, and re-offers it on later turns whose own goal differs. |
| `planning_request` / `technical_spec_request` are endeavours; brainstorming/naming/debugging are one-shot or reactive | Only the two endeavour intents are **goal-forming**. |
| Users also *say* their goals | Explicit declarations set a goal in any intent: "my goal is …", "the goal is …", "our goal is …", "I'm trying to …", "we're trying to …", "I want to …" — captured tail, sentence-cut, capped at 120 chars. |
| Active-thread aggressive expiry (record default 10 turns; lifecycle spec §5.6/§18) | An untouched standing goal cools one count per turn and goes **stale past 10 untouched turns** (`MAX_UNTOUCHED_TURNS`). Topic overlap touches it (content tokens, stopword-filtered like the router, with a crude singular fold). |
| Spec §17: "remove resolved loop", thread collapse on shift | Closing words end a goal: completion phrases → `completed`; abandonment phrases → `abandoned`. Closing is checked **before** goal-forming, and a closing turn forms no goal even when none stands. A goal-forming turn on a new topic **supersedes** the old goal. Every ending is recorded on the ledger's `closed` list with its status — nothing vanishes silently. |

At most **one** goal stands at a time. The per-turn scaffold goal
is **never overwritten**: the turn's intent steers the turn; the
standing goal steers the conversation.

## 3. How it steers

`inject_will` prepends exactly one line to the scaffold's
`fact_block` — the same injection channel the core already uses
for chat history (TF-019/TF-024):

```
Standing goal: help me plan the lantern festival (pursue it as: produce a clear, ordered, actionable plan)
```

- The `(pursue it as: …)` clause appears only when the standing
  response goal differs from the current turn's own goal.
- **No line on the setting turn** — that turn's scaffold already
  carries its response goal; persistence is meaningful only from
  the next turn on. Injection never duplicates its line.
- No new scaffold field: TF-021's pinned scaffold shape is
  untouched, and prompt assembly (TF-023) is unchanged — the line
  rides the existing knowledge-context section.

## 4. Persistence

The ledger is conversation state on the core (one per
`ThoughtForgeCore` instance), observed after each turn's scaffold
is built and before the TF-027 interrupt cut points — a cut turn
still stated its goal; `resume_turn` does not re-observe (it
finishes the same turn).

For persistence across instances, the ledger serialises
(`to_dict`/`from_dict`) and round-trips through the existing
`ActiveThreadStateRecord` / `MemoryStore` thread-state file:
statement ↔ `summary`, topic ↔ `topic`, `open_loops` carried
through untouched (the will layer preserves loops; it does not
invent them — loop lifecycle is TF-045). One honest loss in that
bridge: the thread record has no response-goal field, so a goal
restored from it steers by statement only (`response_goal == ""`
after restore, pinned by test).

## 5. Measured (this slice, synthetic Lantern Hollow only)

Through the real core, knowledge-only, no model:

- Turn 1, "Help me plan the lantern festival route for Lantern
  Hollow" → standing goal set at turn 1, statement
  `help me plan the lantern festival`, response goal
  `produce a clear, ordered, actionable plan`; turn 1's own
  fact_block carries **no** standing-goal line.
- Turn 2, "What time does the ferry leave?" (routes
  light_conversation; its own scaffold goal is
  `keep the exchange natural, warm, and engaging`) → fact_block is
  exactly the standing-goal line above; the turn still answers.
- Turn 3, "Never mind the festival plan" → goal closed as
  `abandoned`, ledger active = None, closed list holds the record.
- Cooling arithmetic pinned by test: 3 unrelated turns →
  `turns_since_touch == 3`; one lantern turn → 0; exactly 10
  untouched turns leave the goal alive, the 11th stales it.

Two defects found by this slice's own tests and fixed in scope
before banking: (1) exact-token overlap missed ordinary plurals
("lanterns" never touched "lantern") → the singular fold;
(2) closing words were only consulted when a goal stood, so a
first-turn "Never mind the festival plan" *set* a planning goal →
closing is now checked unconditionally, before forming.

## 6. What this slice does not claim

- No model has been asked to *honour* the standing-goal line —
  that is a live-model presence question for Part VII (TF-125's
  suite, whose follow-through scoring TF-029's ledger feeds).
- The core does not yet write the ledger into the store's
  thread-state file on its own; the bridge exists and is tested,
  and the store write path belongs with TF-047's
  conversation→memory slice, which owns all turn write-back.
- Goal *extraction* beyond the six declaration patterns and the
  two endeavour intents is deliberately not attempted: a missed
  goal costs one re-statement; an invented goal steers a whole
  conversation wrongly.
