# TF-047 — Conversation → Memory: the Writeback Leg

*Part III, slice TF-047. Code:
`src/thoughtforge/knowledge/writeback.py` (the leg, its gates,
and `writeback_problems()` re-deriving every claim below from
the live tree). Tests: `tests/test_tf047_writeback.py`.
Builds on TF-042's ingestion contract, TF-044's writeback
scoring, TF-045's lifecycle wiring discipline, and TF-046's
tier verdicts.*

## 1. The question and the measured starting state

The roadmap asks: what does a chat write back to the store,
gated by the enforcement rules? Measured before this slice,
the honest answer was **nothing** — every part of writeback
existed except the leg itself:

- `WritebackRecord` ("transient — drives persistence") was
  never constructed for a real turn and never applied;
  `RuntimeTurnState.writeback` was a slot no producer filled
  and no consumer read (TF-038 could only export a writeback a
  test had set by hand).
- The writeback scoring formulas and thresholds (preference
  ≥ 0.72, episode ≥ 0.58, pattern-learning ≥ 0.78, named by
  TF-044) had no consumer.
- TF-042 wrote `writeback_fact_problems()` for exactly this
  leg and pinned the provenance gap — no memory record could
  carry a source stamp — with "TF-047 owns stamping it".
- The store's contract-gated append methods had no caller in
  any turn path; the core read identity and thread state and
  the knowledge DB, and wrote nothing at all.

## 2. The leg and its gates

`apply_writeback(store, writeback, *, enforcement_passed)`
applies every leg of a `WritebackRecord` — new preferences,
new facts, new episodes, new patterns, thread updates, and
the reinforce / archive / delete lists — and returns a
`WritebackOutcome` in which **every requested item appears
exactly once**, applied or refused with reasons. Silence is
not an outcome this leg produces.

The gates, in the order they fire:

1. **Enforcement (the Part V rule).** The caller passes the
   turn's final-response verdict from the EnforcementGate.
   `False` closes the leg entirely: nothing is written — no
   records, no thread updates, no mutations — and every
   requested item is refused with the gate reason. A failed
   answer teaches the store nothing.
2. **Contract (TF-042).** Facts pass through
   `writeback_fact_problems` (content plus a provenance stamp
   in `MEMORY_PROVENANCE` required); the other kinds are
   validated via `record_problems` on the constructed record
   plus the same stamp requirement. A source outside the
   vocabulary is a refusal, never a silent re-stamp.
3. **Score (TF-044).** Preferences, episodes, and patterns
   must clear their named thresholds under the named
   formulas (imported, never copied — `writeback_problems()`
   re-checks identity with scoring.py's constants). Score
   components come from the entry's `components` dict, fall
   back to the entry's own matching field (confidence /
   importance / quality), then to the stated neutral 0.5 —
   the value TF-045 measured as the lifecycle builder's
   neutral share. Facts carry no writeback formula in the
   tree, so facts are contract-gated only; no fact score was
   invented for symmetry.
4. **Store (TF-042).** The append re-validates through
   `MemoryStore`; a refusal there is caught and recorded,
   never raised into a turn.

## 3. Provenance — the TF-042 gap, closed

The four writeback record kinds (`UserFactRecord`,
`UserPreferenceRecord`, `EpisodicMemoryRecord`,
`ResponsePatternRecord`) now carry a `source` field; legacy
records load with `""`. Every record the leg writes is
stamped: the entry's own valid source, else
`conversation_writeback`. Identity and thread state carry no
stamp by ruling — they are not ingested claims — and
TF-042's `contract_problems()` pin now checks the split both
ways (a promised stamp must exist; an unpromised one must
not appear).

## 4. Mutations and thread updates

Reinforce / archive / delete resolve an id across the profile,
episodic, and pattern stores. Reinforce increments
`times_reinforced` and sets status `reinforced`; archive and
delete set the status and rewrite the store, which drops the
record from its file (TF-041's own deletion shape, reused).
An unknown id is a recorded refusal. The personality core is
deliberately unreachable from these lists: writeback cannot
archive or delete identity.

Thread updates replace the named fields (topic / summary /
tags / open_loops / priority) of the standing working-tier
record, reset its turn counter, and save through the store's
contract gate. With no standing record, an update creates one
only if it names a topic.

## 5. Core wiring

`ThoughtForgeCore._finish_turn` applies `turn.writeback`
through `core.apply_writeback(...)` with
`enforcement_passed=final.enforcement_passed`, after TF-045's
thread cooling, under the same discipline: a writeback failure
is logged at warning and the completed final stands. The
outcome is retained on `core.last_writeback_outcome` (None
until a writeback is applied). Both core additions and the
new module are recorded in TF-039's `POST_FREEZE_ADDITIONS`;
the frozen v1 tables are untouched (additions are minor under
TF-034's rules).

## 6. Honesty — what is NOT built

The **producer** is not built here: no extractor yet fills a
`WritebackRecord` from a live conversation. The prompt
builder's episode-summary and preference-inference prompts
(TF-023) exist, but no code calls them and no model output is
parsed into a writeback. So: what a chat writes back *when a
writeback is produced* is now fully specified, gated, and
proven on fixtures; claiming conversations already learn on
their own would be false. TF-063's long-conversation proof
owns the producer question. Proof grade: FIXTURE (synthetic
stores, a knowledge-only core, this VM) — no live store, no
live model.
