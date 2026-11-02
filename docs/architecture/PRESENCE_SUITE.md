# ThoughtForge Architecture — Presence Suite (TF-125)

Presence suite: long-conversation coherence + the
will layer's follow-through (TF-029) scored over
scripted sagas. Part VII's fifth slice, built on
TF-121's harness and TF-122 / TF-123 / TF-124's
suite pattern. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/presence_suite.py` (`COHERENCE_SAGAS`,
`STEERING_SAGAS`, `CLOSURE_SAGAS`, `SUITE_AXES`,
`SUITE_RULES`, checked by
`presence_suite_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline. The
tests are `tests/test_tf125_presence_suite.py`.
This slice **adds and fixes D-BENCH-11** in the
measuring sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **No presence / coherence / will scorer existed
  in `benchmarks/`.** The estate stood at 12
  benchmarks / 5 fixtures / 5 baselines / 36 metric
  definitions, all checkers clean; nothing scored
  whether a conversation keeps its facts over many
  turns, and nothing scored whether a standing goal
  actually steers later turns or stops steering
  when it closes. TF-121's harness had filed the
  gap as a rule — `the_will_axis_has_no_scorer_yet`,
  "TF-125 owns the will suite" — in the harness,
  not in the TF-016 docket (see §5).
- **The coherence machinery is real — re-measured
  live.** TF-028's `score_conversation` over its
  filed 20-turn script (7 anchors) scores the
  coherent transcript mean 1.000 / verdict PASS,
  and the scripted drift-from-turn-15 transcript
  mean 0.700 / REVIEW with 6 contradiction turns —
  exactly TF-028's filed numbers. TF-063's
  100-turn proof is a different leg: it carries no
  chat history and measures the store channel, not
  presence or will.
- **The will machinery is real — probed.** TF-029's
  `GoalLedger` sets a goal from a declaration in
  any intent or from the two goal-forming intents
  (`planning_request`, `technical_spec_request`),
  touches it on topic overlap, stales it after 10
  untouched turns, and closes it completed /
  abandoned on the filed closing words, recording
  superseded goals rather than dropping them. The
  core holds one ledger per instance and injects
  the steering line before generation
  (`cognition/core.py`, grepped live). What nothing
  had ever done is run a scripted conversation's
  *answers* against the ledger and score whether
  they followed through.

## 2. The suite

`benchmarks/presence_suite.py` scores three axes
over scripted synthetic sagas (TF-017; `data/`
never opened):

| Axis | Corpus | Judge | Held | Score |
|---|---|---|---|---|
| long_conversation_coherence | 12 sagas over TF-028's 20-turn script | TF-028's real `score_conversation`; a saga holds at whole-saga mean ≥ 0.80 (TF-028's `PASS_LATE_MIN`, reused) | 9/12 | **0.75** |
| will_steering | 12 sagas of filed user turns + scripted answers | TF-029's real `GoalLedger` replayed; every steered turn's answer must carry the standing statement; no closed statement may steer; end state must match the filing | 10/12 | **0.8333** |
| will_closure | 12 sagas, same shape | the same replay, over completion / abandonment / supersession / staleness shapes | 9/12 | **0.75** |

**Overall 0.7777777777777778 over 36 scored sagas
— PASS** (every axis at or above the filed 0.75
threshold; an empty axis or empty suite scores
0.0 / REVIEW — no vacuous pass anywhere). The
coherence and closure axes sit exactly at the
threshold — the measured shape, not a tuned one.

**The coherence axis in detail.** Each saga files
a drift shape over TF-028's script answers: the
coherent run (mean 1.000), drift onsets at turns
20 / 19 / 18 / 17 (means 0.95 / 0.90 / 0.85 /
0.80 — the last exactly at the hold floor),
isolated slips (turn 10 alone, turns 7 + 10, turn
2 alone — means 0.95 / 0.90 / 0.95 over sagas
whose late probes all hold), and a truncated
coherent run of 14 turns (1.000). The failing
shapes are §3's. TF-028's stricter verdict is
carried per saga: six holding sagas (the onsets
19–17, onset 20, the use-window slips, the early
slip) are REVIEW under it — their late window or
their contradiction leg fails TF-028's gate while
their whole-saga retention holds. The suite files
both numbers and claims only the mean leg (§6).

**The will axes in detail.** Steering holds
across every way a goal comes to stand — a
declaration in a light or factual turn ("my goal
is…", "I want to…", "we're trying to…"), a
goal-forming planning or spec turn — and across
the ledger's own persistence discipline: intent
changes, cooling turns that still steer, a
topic-overlap touch that resets the cooling, an
identical re-declaration that touches instead of
replacing, and a same-topic planning turn that
touches instead of superseding. Closure holds
across "that's done" and "goal achieved"
completions, "never mind" and "forget it"
abandonments, supersession by a new declaration
and by a new-topic plan, staleness after exactly
10 untouched steering turns (the goal goes stale
on the 11th, TF-029's filed count), closing words
spoken with no goal standing, and a completion
followed by a new goal that then steers.

**The real-code legs.** The checker runs
TF-028's `CoherenceFixtureEngine` over every
non-selective coherence saga (it must reproduce
the saga's transcript from the prompt's request
line alone), and TF-029's `inject_will` over a
ledger replayed from a filed steering saga (it
must prepend the filed standing-goal line to a
real scaffold exactly once, and the replayed
ledger must survive its dict round-trip). A suite
whose machinery were fictional could not pass its
own checker.

## 3. Honesty shapes, filed by name

The corpora score failures, not just passes:

- **Coherence breaks** — drift from turn 15
  (TF-028's canonical drifting run: mean 0.700, 6
  contradiction turns), drift from turn 16 (mean
  0.75, below the 0.80 floor), and a 14-turn saga
  drifting from turn 12 (mean 0.7857) all score
  held=False.
- **A goal dropped** (w11): the answer to a later
  question simply omits the standing goal the
  ledger is steering by — held=False.
- **A goal contradicted** (w12): the answer tells
  the user to forget the timetable and design
  posters instead — held=False.
- **Still-steered closures** — a goal abandoned
  but still steered (c10), a goal completed but
  still steered (c11), and a superseded goal
  still steered in place of its successor (c12)
  all score held=False: the judge flags the
  closed goal's statement in the answer on the
  very turn it appears, by name and status.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `presence_suite` benchmark entry (kind
  `suite`, claims axes presence + will) bound live
  to `benchmarks.presence_suite.run_presence_suite`,
  with a new `presence_suite_corpus` fixture — the
  estate now files **13 benchmarks / 6 fixtures /
  6 baselines / 41 metric definitions**;
- `METRIC_DEFINITIONS` extended with the suite's
  five metrics (`coherence_score`, `steering_score`,
  `closure_score`, `overall_score`, `total_cases`
  for the `presence_suite` benchmark), each stating
  what it actually computes;
- the harness's will-gap rule is discharged in the
  open: `the_will_axis_has_no_scorer_yet` is
  replaced by `the_will_axis_is_scored_by_tf125`,
  and TF-121's / TF-123's / TF-124's test pins were
  updated in the open for the new counts;
- a new pinned baseline, measured live on this VM
  twice, identical, latency pinned nowhere:

**`presence_suite_synthetic_v1`** — coherence
0.75, steering 0.8333, closure 0.75, overall
0.7778, 36 sagas; target overall ≥ 0.75 **met**.
(**Every earlier pinned baseline is preserved
exactly** — persona 0.7166666666666667, persona
suite 0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557, the
empty-store profile baseline — never re-pinned;
`baseline_problems()` is clean over all six.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-11 — added and FIXED by this slice,
  in the measuring sense.** The defect, as probed
  in §1: presence and will follow-through were
  scored by nothing in the benchmarks estate —
  TF-029's steering existed and the core injected
  it, but no scorer ever judged a conversation's
  answers against the ledger, and no scorer
  judged long-conversation coherence as a suite
  (TF-028's scorer stood as a Part II fixture, its
  Part VII consumption assigned to TF-125 in its
  own doc). The gap was filed as a harness rule by
  TF-121 and assigned in TF-028's doc, but was
  never numbered in this docket — the D-BENCH-9 /
  D-BENCH-10 pattern. This slice numbers it and
  fixes the measuring share the roadmap assigns
  to TF-125: both claims now have a mechanical
  scorer (`benchmarks/presence_suite.py`) with a
  pinned harness baseline.
- No other entry moves. TF-016 totals are
  therefore 126 entries — **35 OPEN, 1
  OPEN-ENVIRONMENT, 87 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-11 added + FIXED in
  the measuring sense). Counting note, in TF-122 /
  TF-123 / TF-124's manner: the docket's
  header-vs-row counting mismatches (the header
  counts D-LIC-2 FIXED while its row reads
  OPEN-narrowed, and the primary-owner part counts
  summed 123 against 125 before this slice) are
  pre-existing, are preserved unchanged in shape
  (parts now sum 124 against 126, D-BENCH-11's
  owner part VII counted), and are noted here
  rather than silently "repaired".

## 6. Filed limits

- The sagas are **scripted synthetic fixtures**
  (TF-017): TF-028's invented Lantern Hollow world
  and invented ferry / harbour talk, with scripted
  answers, never canon, never a model's live
  output; `data/` was never opened. The suite
  proves the scoring end to end through the real
  TF-028 / TF-029 machinery; **a live small
  model's presence over real long conversations —
  and its own answers' follow-through against the
  ledger the core keeps for it — is NOT PROVEN**
  here (TF-126's road).
- The coherence hold rule is the whole-saga mean
  at TF-028's 0.80 floor. TF-028's own verdict is
  stricter (late window ≥ 0.80, drift ≤ 0.20,
  zero contradictions) and disagrees with the
  suite's hold on six filed sagas, named in §2 —
  the suite measures retention over the saga and
  does not claim TF-028's verdict for them.
- The will axes replay the filed user turns with
  their filed intents / topics; the router is not
  re-run (TF-020 / TF-029 own routing, and TF-029's
  tests pin it). Steering is judged by the
  standing statement's presence in the answer —
  the deterministic shape of the core's injection
  line — not by semantic fidelity to the goal.
- The suite baseline pins the scripted run only.

Proof grade: FIXTURE — synthetic scripted sagas
through the real TF-028 / TF-029 machinery on this
VM. No live model was run; that is NOT PROVEN,
never claimed.
