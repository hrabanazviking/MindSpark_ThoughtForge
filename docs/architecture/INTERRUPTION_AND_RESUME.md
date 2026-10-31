# ThoughtForge Architecture — Interruption & Resume (TF-027)

Interruption & resume: a turn can be cut and the state remains
coherent — the Rúnmál voice lesson, applied to text. Built
2026-10-31 on branch `yrsa/thoughtforge-road-build`.

The contract lives in `src/thoughtforge/cognition/interruption.py`
(`InterruptToken`, `TurnInterrupted`, `check_interrupt`,
`mark_interrupted`); the cut points and `resume_turn()` live in
`ThoughtForgeCore` (`cognition/core.py`); the cut record lives on
TF-025's `RuntimeTurnState` (`knowledge/models.py` §14); the
streamed terminal event is TF-026's `StreamEvent` kind
`interrupted`. Tests: `tests/test_tf027_interruption_resume.py`
(21 tests). No TF-016 docket entry owned interruption; nothing is
flipped by this slice.

## 1. What was true before

- A turn had exactly two fates: complete inside `think()` /
  `think_stream()`, or vanish with its caller. There was no way
  to stop a turn in progress — a user who changed their mind, or
  a voice front-end applying barge-in, could only abandon the
  call and lose whatever the turn had already truthfully built.
- Nothing recorded a cut. A caller who stopped consuming a
  `think_stream()` generator left the retained state claiming
  whatever phase it had reached, indistinguishable from a turn
  still running — and the partial candidate it had been streaming
  simply evaporated, unrecorded either way.
- The phased, retained state (TF-025) was built for exactly this
  slice (TURN_STATE §5/§7), but no transition used it.

## 2. The coherence contract

One sentence: **a cut is a recorded state transition at a named
cut point — never a tear, never a fabricated completion, and the
same turn can be finished later from what it truthfully holds.**

- **Cooperative, stage-atomic.** The caller hands the turn an
  `InterruptToken` (thread-safe `threading.Event` + first reason
  wins; `interrupt()` is idempotent and callable from any thread,
  or by a stream consumer between events). The core checks it at
  named cut points only:
  1. after the scaffold is built, before generation;
  2. before each candidate, in both generation paths;
  3. before each chunk is delivered, in both streaming paths
     (native engine, simulated re-chunking, unified backend);
  4. after generation, before compose.
  Between cut points a stage is atomic: a token fired during
  routing/retrieval is honoured at cut point 1; a token fired
  during the final compose is *not* retroactive — that turn
  completes, and firing a token after completion rewrites
  nothing (pinned by test).
- **The cut keeps exactly what was true.** `turn.interrupted` is
  set with `interrupt_reason` and `interrupted_phase` (the phase
  at the cut — always the last *finished* stage, because phases
  record completed milestones). `final_response` stays None,
  `completed_at` stays empty, `salvage_ms` stays 0 when compose
  never ran, and `generation_ms` is measured for the partial leg
  too (assigned in a `finally`, so a cut leg is real time, not a
  default). Completed candidates stay on `turn.candidates` — the
  generation helpers now append into that list (the *sink*) as
  each candidate finishes, in every path.
- **The in-flight candidate is discarded whole.** TF-026's
  mid-stream failure rule, generalised to interruption: chunks
  already delivered to a stream consumer were delivery, not a
  candidate; the candidate record is appended only when its full
  text exists. A cut turn therefore never holds a partial text
  that a later compose could mistake for a finished draft.
- **The cut is visible, never silent.** A `think()` cut raises
  `TurnInterrupted` (carrying the turn and the reason) — a cut
  is not a content failure, so it is never swallowed into a
  fabricated answer; TF-031 owns what degraded *answers* look
  like, and this is not one. A streamed cut ends the stream with
  exactly one terminal `interrupted` event (reason in `text`,
  `final=None`) instead of `final` — TF-026's event-order law is
  unchanged apart from the one added kind (the TF-026 pin was
  updated with the reason recorded in its test).
- **Closing a stream is a cut.** A consumer who calls `.close()`
  on the `think_stream()` generator mid-turn cuts it:
  `GeneratorExit` is caught only long enough to record
  `consumer closed the stream` on the retained state, then
  re-raised untouched (a closing generator never yields again).
  Closing *after* the `final` event records nothing — that turn
  already completed.

## 3. Resume: the same turn, finished

`ThoughtForgeCore.resume_turn(interrupt=None)`:

- Requires the retained turn to be interrupted, and its phase to
  be at least `scaffolded`; anything else raises `ValueError`
  (no turn, a completed turn, or the one non-resumable cut —
  §4). A failed resume changes nothing on the turn.
- Reuses the retained sketch / bundle / scaffold — no
  re-routing, no re-retrieval, no history re-fold (history was
  folded into the scaffold in the original leg).
- If the phase is before `generated`, generation runs **only
  for the scaffold modes with no candidate on the turn**
  (`only_modes` in `_generate_candidates`; the unified-backend
  path likewise skips its one mode if that candidate exists),
  then fragments are scored over the full candidate set.
  A candidate completed before the cut is the same object after
  resume (identity-pinned by test), never regenerated.
- Compose, the quality gate and completion run through the same
  `_finish_turn` the live pipeline uses (extracted from
  `_run_turn` in this slice; the live path's behaviour is
  unchanged — TF-025/TF-026 tests green untouched apart from the
  one event-kind pin).
- The turn keeps its `turn_id`; the returned
  `FinalResponseRecord` is the turn's own. `resume_count`
  increments when a resume begins; the cut fields
  (`interrupted`, `interrupt_reason`, `interrupted_phase`) are
  cleared at that moment, so they always describe the *current*
  cut. A resume can itself be cut (pass a fresh token) and
  resumed again — pinned: cut → resume cut (count 1) → resume
  completes (count 2). Resume completes through the
  completed-text generation path even when the original leg was
  streamed: resume is about finishing the *turn*, and the final
  record is the same kind either way.
- A resumed turn records a `think.resume` perf sample (the
  resume leg's wall time); the original leg's `think.*` samples
  were already recorded only on completion, so a cut turn leaves
  no half-samples behind.

## 4. The one non-resumable cut

A stream consumer who closes the generator immediately after
`turn_started` — before routing has run — leaves a coherent cut
at phase `created`: no sketch, no scaffold, nothing invented.
`resume_turn()` refuses it (`ValueError`) because there is no
front half to resume *from*; the caller starts a new turn.
Every token-driven cut is resumable by construction, because
the first token cut point is the scaffold checkpoint.

## 5. History coherence

The core never appends to a caller's `ChatHistory` itself, and
callers append assistant entries from a turn's
`FinalResponseRecord`. A cut turn has none — so no partial
utterance can enter the history a later turn is scaffolded from.
Pinned by test: a history passed into a cut `think()` is
byte-for-byte the messages it held before the call.

## 6. Measured (synthetic stubs, this slice's tests)

- Non-stream cut, stub engine firing the token inside its first
  `generate()`: turn cut at `scaffolded` with exactly 1 candidate
  (the second mode never started); resume made exactly 1 further
  `generate()` call; final composite 0.579 / usable on the stub
  text; same `turn_id` throughout.
- Streamed cut after 3 delivered chunks of candidate 1: stream
  ends `interrupted`, no `candidate_end`, turn holds 0
  candidates; resume completes it with 2 candidates via the
  completed-text path.
- Streamed cut after candidate 1's `candidate_end`: turn holds
  exactly that candidate; resume generates only mode 2 (1
  `generate()` call).
- Consumer close after `candidate_start`: cut recorded at
  `scaffolded`, resume completes the turn; close after
  `turn_started` only: cut at `created`, resume refuses.

## 7. Deliberately not here

- **Resuming a cut generation mid-candidate** (token-level
  continuation of a half-generated draft): candidates are
  completed texts or they are not candidates (TF-026's rule);
  resume regenerates the missing *modes*, never a half draft.
- **Persisting a cut turn across processes**: the retained state
  is in-memory on the core, like TF-025's retention; trace/export
  machinery is TF-038's road.
- **Interrupting the HTTP backends inside a single blocking
  `generate()` call**: a completed-text call is atomic between
  cut points by the §2 ruling; the token is honoured at the next
  candidate boundary.
