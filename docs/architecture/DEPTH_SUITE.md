# ThoughtForge Architecture — Depth Suite (TF-124)

Depth suite: multi-step reasoning with scaffolds
on/off — the scaffold value measured, not assumed.
Part VII's fourth slice, built on TF-121's harness
and TF-122 / TF-123's suite pattern. Built
2026-11-01 on branch `yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/depth_suite.py` (`CHAIN_CASES`,
`PLAN_CASES`, `CHECK_CASES`, `SUITE_AXES`,
`SUITE_RULES`, checked by `depth_suite_problems()`),
registered in `benchmarks/harness.py` with a pinned
baseline. The tests are
`tests/test_tf124_depth_suite.py`. This slice
**adds and fixes D-BENCH-10** in the measuring
sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **No depth / reasoning scorer existed anywhere.**
  The harness stood at 11 benchmarks / 4 fixtures /
  4 baselines / 26 metric definitions, all checkers
  clean; no module, fixture or harness entry scored
  a multi-step reasoning outcome on either road.
  TF-022's strategy purposes asserted the value
  ("so order and completeness come from the plan
  step rather than from improvisation"); nothing
  measured it.
- **The scaffold estate is real — probed.** TF-021's
  registry files 10 intent scaffolds / 8 candidate
  modes. TF-022's `strategy_registry()` files
  exactly three strategies over six intents
  (chain_of_thought: technical_debugging,
  technical_spec_request; plan_then_answer:
  planning_request, brainstorming; self_check:
  factual_query, editing_refinement), with
  `select_strategy` returning None for the light
  intents by design. `PromptBuilder` builds both
  roads: a single-pass candidate prompt carries no
  "Working so far" section; a strategy step prompt
  consuming prior work carries one; a later strategy
  step run cold raises `ValueError` (all probed
  live this tick).
- **The filed limit, stated not hidden.** The core
  loop does not execute strategies automatically:
  TF-022's doc (§1) assigned that wiring to the
  slices owning those surfaces, and TF-032 landed
  per-conversation *selectability* only
  (`core.selected_strategy` resolves a strategy;
  no core code path calls `build_strategy_prompt` —
  grepped live this tick). That assignment was
  never numbered in the TF-016 docket — the
  D-BENCH-9 pattern. This slice numbers it
  (D-BENCH-10) and fixes the measuring share; the
  execution share stands exactly as TF-022 filed
  it, unclaimed here.

## 2. The suite

`benchmarks/depth_suite.py` scores three axes over
scripted synthetic problems, one axis per TF-022
strategy, every case run on **both roads** over the
same filed problem:

| Axis (strategy) | Corpus | OFF scorer | ON scorer | OFF | ON | Delta |
|---|---|---|---|---|---|---|
| chain_of_thought | 12 arithmetic / logic chains | direct answer exact vs known | filed intermediates in the reasoning + exact final | 5/12 → **0.4167** | 10/12 → **0.8333** | **+0.4167** |
| plan_then_answer | 12 ordering problems | direct sequence exact vs known order | filed steps in the plan, in order, + exact sequence | 6/12 → **0.5** | 9/12 → **0.75** | **+0.25** |
| self_check | 12 draft problems | draft exact vs known | check names the filed error token (or clean verdict) + revised exact | 6/12 → **0.5** | 10/12 → **0.8333** | **+0.3333** |

**Overall OFF 0.47222222222222227, ON
0.8055555555555557, measured delta
+0.3333333333333334 over 36 scored problems —
PASS** (every axis's ON score at or above the
filed 0.75 threshold *and* a positive overall
delta; the suite passes only when both hold). The
plan axis's ON score is the lowest of the three,
exactly at the threshold — the measured shape,
not a tuned one.

**The real-code leg.** Correctness is judged
against known answers, but the on/off contrast is
judged against the scaffold's real output:
`depth_suite_problems()` runs
`scaffold_prompts_for()` over **every** filed case —
TF-022's `select_strategy` must return the filed
strategy, TF-021's `ScaffoldBuilder` +
`PromptBuilder.build_candidate_prompt` produce the
OFF prompt (no "Working so far", no strategy
instruction), and `build_strategy_prompt` produces
every ON step prompt in order, chaining the case's
working material as prior output (each
prior-consuming step's prompt opens with its real
instruction and carries the "Working so far"
section). A suite whose scaffolds were fictional
could not pass its own checker.

**Honesty shapes, filed by name.** The delta is
not a sweep, and the corpora say so: no-gain cases
where both roads pass (c03, c05, c09, p03, s03…)
and both fail (c12 a correct reasoning chain whose
final answer is a transcription slip, p08 a plan
missing its first step, s06 a check that misses
the error), and **scaffold-loss cases** where the
direct road holds and the scaffolded road fails —
c11 (a reasoning slip: 36 + 4 filed as 44), p11 (a
plan locking in a wrong order), p12 (a plan
dropping the cutting step its direct answer had),
s09 (a check inventing an unsupported problem and
breaking a correct draft). An ON case holds only
when the working material carries the filed
intermediates *and* the final answer is exact —
the right answer without the scaffold's work does
not score as scaffold value.

An empty axis scores 0.0; an empty suite scores
0.0 / REVIEW. No vacuous pass anywhere (the
TF-122 / TF-123 precedent), and this suite adds
the delta leg: PASS additionally requires the
measured delta to be positive.

## 3. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `depth_suite` benchmark entry (kind
  `suite`) bound live to
  `benchmarks.depth_suite.run_depth_suite`, with a
  new `depth_suite_corpus` fixture — the estate now
  files **12 benchmarks / 5 fixtures / 5 baselines
  / 36 metric definitions**;
- `METRIC_DEFINITIONS` extended with the suite's
  ten metrics (`chain_off_score`, `chain_on_score`,
  `plan_off_score`, `plan_on_score`,
  `check_off_score`, `check_on_score`,
  `overall_off_score`, `overall_on_score`, `delta`,
  `total_cases` for the `depth_suite` benchmark),
  each stating what it actually computes;
- a new pinned baseline, measured live on this VM
  twice, identical, latency pinned nowhere:

**`depth_suite_synthetic_v1`** — chain OFF 0.4167 /
ON 0.8333, plan OFF 0.5 / ON 0.75, check OFF 0.5 /
ON 0.8333, overall OFF 0.4722, overall ON 0.8056,
delta +0.3333, 36 cases; target overall ON ≥ 0.75
**met**. (TF-121's test pins for the index /
fixture counts were updated in the open: 11 → 12
benchmarks, 4 → 5 fixtures. **Every earlier pinned
baseline is preserved exactly** — persona
0.7166666666666667, persona suite
0.8453703703703704, memory suite overall
0.8055555555555557, the empty-store profile
baseline — never re-pinned; `baseline_problems()`
is clean over all five.)

## 4. Docket dispositions (TF-016)

- **D-BENCH-10 — added and FIXED by this slice, in
  the measuring sense.** The defect, as probed in
  §1: scaffold value was measured by nothing — no
  scorer anywhere compared a scaffolded reasoning
  run against the single-pass road — and TF-022's
  doc had assigned core-loop strategy execution to
  other slices without the gap ever being numbered.
  This slice numbers it per the docket's rule and
  fixes the measuring share assigned to TF-124 by
  the roadmap: both roads now have a mechanical
  scorer (`benchmarks/depth_suite.py`) with a
  pinned harness baseline. The execution share is
  unchanged and stated plainly in the docket row:
  the core loop still does not run strategies
  automatically; wiring it is not claimed here and
  stands with the surfaces TF-022 named.
- No other entry moves. TF-016 totals are
  therefore 125 entries — **35 OPEN, 1
  OPEN-ENVIRONMENT, 86 FIXED, 3 CLOSED-BY-VERDICT**
  (D-BENCH-10 added + FIXED in the measuring
  sense). Counting note, in TF-122 / TF-123's
  manner: the docket's header totals and a
  row-level recount have disagreed before this
  slice (the header counts D-LIC-2 FIXED while its
  row reads OPEN-narrowed, and the primary-owner
  part counts summed 122 against 124 before this
  slice) — both mismatches are pre-existing, are
  preserved unchanged in shape (parts now sum 123
  against 125, D-BENCH-10's owner part VII
  counted), and are noted here rather than
  silently "repaired".

## 5. Filed limits

- The cases are **scripted synthetic fixtures**
  (TF-017): invented known-answer problems with
  scripted direct answers, working material and
  scaffolded answers, never canon, never a model's
  live output; `data/` was never opened. The suite
  proves the scoring end to end and measures the
  scripted delta through the real prompt builders;
  **a live small model's depth behaviour under
  these scaffolds is NOT PROVEN** here (TF-126's
  road).
- The core loop does not execute strategies
  automatically (TF-022's filed limit, §1): the
  suite exercises the prompt-level machinery, not
  a live multi-pass generation run.
- Correctness is exact normalised equality with
  the filed known answer — the deterministic,
  checkable rule for these problem shapes. Free-form
  reasoning quality beyond carrying the filed
  intermediates and landing the exact answer is
  judgement, not smuggled into a scorer.
- The OFF road is the intent's first candidate mode
  (TF-021's single-pass road); candidate *selection*
  among modes is not scored here.
- The suite baseline pins the scripted run only.

Proof grade: FIXTURE — synthetic scripted cases
through the real TF-020 / TF-021 / TF-022 machinery
on this VM. No live model was run; that is NOT
PROVEN, never claimed.
