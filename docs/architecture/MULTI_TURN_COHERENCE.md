# Multi-Turn Coherence Fixtures (TF-028)

*Part II, TF-028. The discipline lives in
`src/thoughtforge/cognition/coherence.py`; its tests are
`tests/test_tf028_multi_turn_coherence.py` (22 tests). Everything
below was measured on the campaign tree at this slice, through
`ThoughtForgeCore` itself — not quoted from any plan.*

## 1. The question

Every Part II slice before this one proved a single turn. This
slice asks the multi-turn question: **across 20 turns of one
conversation, does the forge keep its facts straight, or does
it drift?** The roadmap's wording: *"20-turn conversations on a
small model fixture, scored for drift (with Part VII's harness)."*
Part VII's harness (TF-121) does not exist yet, so this slice
builds the fixture, the scorer and the runner in the shape that
harness will consume — the same relationship TF-017's fixtures
bear to the tests that use them.

## 2. The script and its anchors

`COHERENCE_SCRIPT` is exactly 20 turns, all in TF-017's synthetic
world (Lantern Hollow on the River Sarn; Wrenna the
Lantern-Keeper). No canon appears anywhere in the script, the
anchors, or the tests — a guard test scans for canon markers.

- **Turns 1–6 establish** six facts. **Turns 7–14 use** them
  (planning, arriving, describing). **Turns 15–20 probe** them
  again in new wording; turn 20 is a combined probe checking
  three anchors in one answer.

The facts are **anchors**, specified as data (`ANCHORS`): each
carries required phrase *groups* (a group is satisfied when any
alternative appears) and *forbidden* phrases that contradict it.

| Anchor | Fact | Probed on turns |
|---|---|---|
| `ferry_minutes` | crossing takes eleven minutes in calm water | 1, 7, 13, 16, 20 |
| `ferry_closes` | no ferry runs after the second tide bell | 2, 8, 15, 20 |
| `red_lantern` | river runs high → a red lantern at the ferry post | 6, 12, 17, 20 |
| `river_record` | river level recorded in chalk on the mill door | 4, 10, 14, 18 |
| `keeper` | Wrenna is the Lantern-Keeper | 3, 4, 6, 9, 12, 14, 17, 18 |
| `tide_bell` | the tide bell rings twice at dusk | 5, 11, 19 |
| `place` | the conversation's ground is Lantern Hollow | all 20 |

Every turn probes at least one anchor; every anchor is probed at
least three times. Each script turn carries **two** answers: the
coherent one and a drifted one (wrong minutes, a ferry that runs
all night, a blue lantern, Tomas keeping the lanterns, a level
written in a diary, a bell ringing three times at dawn) — the
ground truth the scorer must catch.

## 3. The scorer

`score_conversation(transcript, script)` is a pure function. The
transcript is paired with the script by position (a full run or
a prefix); a user turn that does not match the script raises —
that is a harness error, not drift.

- **Per turn**, only the anchors that turn's question probes are
  scored; an unprobed anchor is neither credited nor blamed
  (pinned by test: a turn that never asks about the crossing is
  not failed for a wrong crossing time mentioned in passing —
  that failure belongs to the turn that *does* probe it).
- **Retention** = satisfied required groups / required groups
  over the probed anchors. Partial satisfaction scores
  fractionally (measured pin: chalk without the mill door on the
  river-record turn = 3 of 4 groups = 0.75).
- **A contradiction zeroes the turn.** If a forbidden phrase for
  a probed anchor appears, the turn scores 0.0 even when the
  correct fact is also stated (pinned by test). An answer that
  says both "eleven minutes" and "twelve minutes" has not kept
  the fact; it has lost it.
- **Windows and drift.** Early = turns 1–5, late = turns 16–20,
  **drift = early_mean − late_mean**. The report also carries
  the overall mean, the contradiction-turn count, and per-anchor
  retention over that anchor's probe turns.
- **Verdict.** `PASS` only when late_mean ≥ 0.80, drift ≤ 0.20,
  and contradiction turns = 0; otherwise `REVIEW`. These are
  this fixture's gate values, stated here for Part VII to
  revisit with baselines (TF-129), not universal constants.
- **An empty transcript scores 0.0 / REVIEW** — never the
  vacuous 1.0 PASS TF-010 caught the persona scorer awarding an
  empty list (D-BENCH-4's lesson, applied from the start).

## 4. The small-model fixture

`CoherenceFixtureEngine` is a deterministic completed-text
engine in the shape the core drives (`generate(prompt, params)`
→ an object with `text` / `tokens_generated`). With
`drift_from_turn` set (1-based), turns before it answer the
coherent text and turns from it onward answer the drifted text;
`None` never drifts. Two disciplines keep it honest:

- **It routes on the `User request:` line only.** Every prompt
  kind carries that line (TF-023's contract), and history rides
  *inside* candidate prompts below it. A probe while building
  this slice showed whole-prompt keyword matching misrouting
  turn 4 ("Remind me again, how long…") to the red-lantern
  answer, because the history section quoted turn 3 — drift
  manufactured by the fixture, not the model. The engine
  therefore extracts the first `User request:` line
  (`extract_request`) and matches the script on it alone; a
  regression test pins the contaminated-prompt case.
- **It ignores generation params**, temperature included, so a
  measured difference is the scripted degradation, never
  sampling luck. Both candidate modes return the same text, as
  TF-026/027's stubs did, and salvage/refine prompts (which
  carry the same request line) route identically. A request no
  script turn owns gets a grounded fallback sentence, never a
  fabricated fact.

## 5. The runner and the history it threads

`run_conversation(core, script)` threads one `ChatHistory`
through `core.think`, following the phase-7 chat pattern:
`add_user` *before* the turn (the current question rides in
that turn's history render, as a real caller's would),
`add_assistant(final.text, turn_id=final.turn_id)` after it.
The transcript scored is the **final** answers — what a user
actually reads, salvage and enforcement included.

Measured on the coherent run: the stored history ends at
exactly **40 messages with dropped_count = 0** — the 20-turn
script sits exactly at TF-024's default window (20 turns /
1,000 word-tokens; the script's stored content is ≈580 words),
so nothing is silently lost. The *render* is a different
matter, also measured: turn 2's candidate prompt contains turn
1's question and answer in full, while by the probe phase the
core's 1,500-char history render (TF-024's
`CORE_HISTORY_MAX_CHARS`) no longer reaches back to turn 1.
Storage is whole; the prompt window is budgeted. Both facts
are the discipline working as specified, and both are recorded
here because a future live-model run's late-turn behaviour
must be read against them.

## 6. Measured results (this tree, this slice)

Both runs went through the full core (route → retrieve →
scaffold → generate → salvage → enforce) with the fixture as
the engine, on an empty knowledge store.

**Coherent fixture** (never drifts):

| Metric | Value |
|---|---|
| Turns | 20 |
| Mean / early / late | 1.000 / 1.000 / 1.000 |
| Drift | 0.000 |
| Contradiction turns | 0 |
| Per-anchor retention | all seven anchors 1.000 |
| Verdict | **PASS** |
| Engine calls | 54 = 40 candidate (2 modes × 20 turns) + 14 salvage-refine |
| History | 40 messages, dropped 0, 20 unique turn_ids |

**Drifting fixture** (`drift_from_turn=15` — the probe phase):

| Metric | Value |
|---|---|
| Turns scoring 1.0 / 0.0 | turns 1–14 / turns 15–20, no partial turns |
| Mean / early / late | 0.700 / 1.000 / 0.000 |
| Drift | 1.000 |
| Contradiction turns | 6 (turn 18 contradicts two anchors; turn 20, three) |
| Per-anchor retention | ferry_minutes 0.600, ferry_closes 0.500, red_lantern 0.500, river_record 0.750, keeper 0.750, tide_bell 0.667, place 0.800 |
| Verdict | **REVIEW** |

The per-anchor numbers are hand-checkable against the script:
`ferry_minutes`, for example, is probed on five turns, kept on
three and contradicted on two → 0.600. Two coherent runs
produce byte-identical transcripts (pinned by test).

## 7. Proof grade and what consumes this

**Fixture-proven only.** The script, scorer, runner and their
ground truth are proven; the coherent fixture's perfect score
proves the *core pipeline preserves* scripted answers across
20 threaded turns, and the drifting fixture proves the scorer
*catches* scripted degradation with the right arithmetic. No
real model has been measured here and no drift claim about any
real model is made — that is TF-125's presence suite (Part VII)
re-running this script against a live small model, with TF-063
extending the same discipline to 100 turns. TF-121's harness
inherits the scorer's verdict shape and the empty-transcript
rule.
