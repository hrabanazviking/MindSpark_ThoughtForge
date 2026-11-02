# ThoughtForge Architecture — Persona Consistency Suite (TF-122)

Persona consistency suite: the existing
`persona_consistency` benchmark completed into a scored
suite (identity facts, voice, values under pressure).
Part VII's second slice, built on TF-121's harness.
Built 2026-11-01 on branch `yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/persona_suite.py` (`IDENTITY_FACTS`,
`IDENTITY_TURNS`, `VOICE_TURNS`, `PRESSURE_VALUES`,
`PRESSURE_TURNS`, `SUITE_AXES`, `SUITE_RULES`, checked
by `persona_suite_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline; the
scorer's two standing defects were fixed in the open in
`benchmarks/persona_consistency.py`. The tests are
`tests/test_tf122_persona_suite.py`. This slice
**fixes D-BENCH-3** and **narrows D-BENCH-4** (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **D-BENCH-3 stood open**: `PersonaConsistencyScorer`'s
  `weights` argument was merged into a dict that was
  never read — the TF-121 fixture scored
  0.7166666666666667 identically with
  `weights={"generic": 99}` — and the module docstring's
  formula `((1 − penalty + norse + citation) / 3)` was
  never the implementation (the real computation adds
  an undocumented skald bonus per turn and averages).
- **D-BENCH-4 stood open**: `score([])` returned
  consistency 1.0 — a vacuous PASS on no evidence — and
  the "100+ turns" claim lived in BUILD_PLAN_v1.md,
  MODEL_CARD.md and TASK_PHASE6_RELEASE.md (plus the
  `benchmarks/` package docstring) with no fixture or
  recorded run anywhere in the tree.
- TF-121 had pinned the scorer's behaviour exactly as
  it stood: `persona_consistency_synthetic_v1` =
  0.7166666666666667 (flagged 2, generic 7, norse 10,
  citation turns 1, skald 5), honestly below the 0.75
  pass threshold — and filed that any scorer fix must
  move it through `baseline_problems()`, never silently.
- TF-030's cognition half filed the ground truth:
  `FIXTURE_EXPECTATION` (synthetic Wrenna) with
  `check_persona_response` / `score_persona_turns`,
  whose empty batch already scored 0.0 / REVIEW.

## 2. The scorer fixes (in the open)

In `benchmarks/persona_consistency.py`:

- **Weights are real (D-BENCH-3).** Each per-turn
  component is scaled by `weight / DEFAULT_WEIGHTS`
  (defaults: generic 1.0, norse 1.0, citation 0.5,
  skald 0.5 — the dict the scorer always built). A
  non-default weight demonstrably moves the score
  (the fixture at `{"generic": 2.0}` scores 0.6667, at
  `{"norse": 3.0}` 0.7333, at `{"generic": 0.0}` 1.0),
  and at the defaults every multiplier is 1.0, so the
  TF-121 pinned baseline re-measures **exactly**
  0.7166666666666667 and `baseline_problems()` stays
  clean — the pin was preserved, not re-pinned. The
  module docstring now states the computation as it
  actually is (per-turn penalty / bonuses with caps,
  weighted, clamped, then the mean). The public API —
  `score()` signature and every result field — is
  unchanged.
- **No vacuous pass (D-BENCH-4).** `score([])` returns
  consistency 0.0 and does not pass (REVIEW), matching
  `score_persona_turns`' filed shape. One existing
  test did pin the vacuous 1.0 — phase-6's
  `test_empty_input_returns_perfect_score` — and it
  was updated in the open by this slice (renamed
  `test_empty_input_scores_zero_not_vacuous_pass`,
  reason stated in its comment); the new tests pin
  0.0 as well.

## 3. The suite

`benchmarks/persona_suite.py` scores three axes over
scripted synthetic corpora, all against the one filed
persona — TF-030's `FIXTURE_EXPECTATION`, imported and
reused, never copied:

| Axis | Corpus | Scorer | Filed result |
|---|---|---|---|
| identity facts | 36 turns: 6 filed facts × (5 holding + 1 breaking) | filed required/breaking phrase rules + TF-030's `check_persona_response` | 30/36 held → **0.8333** |
| voice | 36 turns: 30 in-voice + 6 deliberately generic | the real `PersonaConsistencyScorer` at default weights | **0.8694** (flagged 6, generic hits 19, norse hits 65, citation turns 1, skald hits 32) |
| values under pressure | 36 turns: 6 filed values × (5 holding + 1 breaking) | filed hold phrases + TF-030's `check_persona_response` | 30/36 held → **0.8333** |

The six identity facts are the synthetic persona's
filed world (crossing minutes, the tide-bell rule, the
chalk river record, the red lantern, the keeper's
role, the first crossing at dawn). The six pressure
values are the persona's three `avoid` entries read as
values plus three `behavior_rules` (answer what was
asked, cite the reference, say plainly when the store
is silent).

**Overall 0.8454 over 108 scored turns — PASS** (every
axis at or above the filed 0.75 threshold; the suite
passes only when every axis passes). That is the
substantial run D-BENCH-4 found missing: the "100+
turns" scale is now a filed corpus, a live checker and
a pinned baseline — not a claim in a doc.

An empty axis scores 0.0; an empty suite scores 0.0 /
REVIEW. No vacuous pass anywhere.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- the `persona_consistency` entry's suite-slice note
  now files TF-122 done, with its docket lines updated
  (D-BENCH-3 FIXED; D-BENCH-4 narrowed);
- a new `persona_suite` benchmark entry (kind `suite`)
  bound live to `benchmarks.persona_suite.run_persona_suite`,
  with a new `persona_suite_corpus` fixture;
- `METRIC_DEFINITIONS` extended with the suite's axis
  metrics (`identity_score`, `voice_score`,
  `values_score`, `overall_score`, `total_turns` for
  the `persona_suite` benchmark), each stating what it
  actually computes;
- a new pinned baseline, measured live on this VM
  twice, identical, latency pinned nowhere:

**`persona_suite_synthetic_v1`** — identity 0.8333,
voice 0.8694, values 0.8333, overall 0.8454, 108
turns; target overall ≥ 0.75 **met**. (TF-121's test
pins for the index / fixture counts were updated in
the open: 9 → 10 benchmarks, 2 → 3 fixtures.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-3 — FIXED.** The `weights` argument is
  wired into the per-turn scoring (§2), the docstring
  formula states the real computation, the public API
  is intact, and the TF-121 persona baseline passes
  `baseline_problems()` unchanged — exact preservation
  proven, not assumed.
- **D-BENCH-4 — narrowed.** Both filed defects are
  remedied in substance: the empty list no longer
  scores a vacuous PASS (0.0 / REVIEW), and the
  missing 100+ turn run now exists as the suite's
  filed 108-turn corpus with a pinned recorded
  baseline. What stands is the share TF-010 itself
  filed for TF-136: the three public docs
  (BUILD_PLAN_v1.md, MODEL_CARD.md,
  TASK_PHASE6_RELEASE.md) still *word* the claim
  without pointing at the suite; the public-claims
  audit strikes or re-points that wording. The entry
  stays OPEN on that share.
- No new docket entry: every gap probed this tick was
  already numbered. TF-016 totals are therefore 123
  entries — **35 OPEN, 1 OPEN-ENVIRONMENT, 84 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-3 moves OPEN → FIXED).

## 6. Filed limits

- The corpora are **scripted synthetic fixtures**
  (TF-017): invented lines in TF-030's invented
  Lantern Hollow world, never canon, never a model's
  live output; `data/` was never opened. The suite
  proves the scoring end to end; **a live small
  model's persona behaviour over 100+ turns is NOT
  PROVEN** here (TF-125 / TF-126's road).
- The identity and pressure axes judge only the filed
  phrase rules plus TF-030's avoid checker — the
  deterministic, checkable subset. Broader fidelity
  (to traits or speech style beyond those rules) is
  judgement, not smuggled into a matcher.
- The voice axis inherits the phrase scorer's limits:
  it counts filed marker phrases, not meaning; the
  voice corpus's single citation turn is a property
  of the scripted lines, not a citation-quality claim
  (TF-123 owns citation truth).
- The suite baseline pins the scripted run at the
  scorer's default weights; a weighted run is a
  different run and is never pinned as this baseline.

Proof grade: FIXTURE — synthetic scripted corpora
through the real scorers on this VM. No live model was
run; that is NOT PROVEN, never claimed.
