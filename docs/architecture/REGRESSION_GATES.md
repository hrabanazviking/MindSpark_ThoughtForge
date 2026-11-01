# ThoughtForge Architecture — Regression Gates (TF-129)

Regression gates: benchmark deltas beyond
tolerance fail the campaign's slice gate
(thresholds from baseline noise, measured).
Part VII's ninth slice, built on TF-121's
harness (the pinned baselines and
`baseline_problems()`), TF-036's determinism,
and TF-122..TF-128's pinned suite / panel
baselines. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/regression_gate.py` (`NOISE_ROWS`,
`TOLERANCES`, `GATE_PROBES`, `GATE_RULES`,
checked by `regression_gate_problems()`),
registered in `benchmarks/harness.py` with a
pinned baseline. The tests are
`tests/test_tf129_regression_gate.py`. This
slice **fixes D-BENCH-14** (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The estate stood clean, and nothing gated
  it.** `baseline_problems()` and
  `harness_problems()` both returned [] over
  the nine pinned baselines (83 pinned metrics
  in total, counted live). But naming a moved
  metric was the whole story: no verdict
  object, no PASS / FAIL semantics a slice gate
  could consume, and no classification of a
  move (regression vs improvement vs count
  change) existed anywhere in the estate. A
  slice could drift every baseline and no gate
  would fail.
- **The tolerances were uniform and asserted,
  never measured.** Every pinned baseline
  carries `tolerance: 1e-9` (TF-121's filing).
  No slice had ever re-measured a baseline
  repeatedly and looked at the spread, so
  nothing said whether 1e-9 was the estate's
  noise, far above it, or below it.
- **The gap stood filed as future work** in
  `benchmarks/harness.py`'s own filed-limits
  passages — every TF-122..TF-128 update
  paragraph listed "TF-129's regression gates"
  as standing (seven places, grepped) — and
  the docket number **D-BENCH-14** was verified
  unused anywhere in benchmarks / docs / tests
  / src before this slice filed it.

## 2. The noise, and the tolerances derived from it

`benchmarks/regression_gate.py` re-measures
every gated baseline **5 times in one process**
(`NOISE_REPEATS = 5`) through the harness's own
measurers — no corpus or scorer is copied.
The repeat count is filed from measured cost:
one full re-measurement of all nine baselines
took ~1.6 s warm in-process on this VM at this
slice's probes (~7 s cold; five repeats
~8.1 s) — cheap enough for a checker a slice
actually runs, and enough samples to see a
spread if one existed.

**The measured noise** (exact rows are the
literal `NOISE_ROWS`, recomputed live by the
checker):

| Baseline | Metrics | Max abs spread | Nonzero spreads |
|---|---|---|---|
| persona_consistency_synthetic_v1 | 7 | 0.0 | — |
| persona_suite_synthetic_v1 | 5 | 0.0 | — |
| memory_suite_synthetic_v1 | 5 | 0.0 | — |
| depth_suite_synthetic_v1 | 10 | 0.0 | — |
| presence_suite_synthetic_v1 | 5 | 0.0 | — |
| honesty_panel_synthetic_v1 | 13 | 0.0 | — |
| quant_panel_synthetic_v1 | 16 | 0.0 | — |
| profile_panel_synthetic_v1 | 13 | 0.0 | — |
| profile_benchmark_empty_store_v1 | 9 | 0.0 | — |

**Every spread is 0.0.** No metric of any
baseline moved at all across five repeats, and
the one dict metric (`quality_tiers`) never
changed. That is TF-036's determinism holding,
measured here rather than assumed; the rows
file per-metric spreads, so any future nonzero
spread is named in its row, never smoothed.

**Tolerance derivation, filed as a rule.** Per
metric, the gate tolerance is
`max(measured noise spread, 1e-9)` for float
metrics, where 1e-9 is the harness's
already-filed baseline tolerance floor
(TF-121). Noise can only widen a tolerance to
what was actually observed — never to an
invented number. Integer / count metrics and
the dict metric are **exact**: any change at
all is beyond tolerance. On this tree, since
every measured spread is 0.0, every float
tolerance files the floor 1e-9. `TOLERANCES`
is published as data and re-derived from
recomputed noise by the checker — a tolerance
that drifts from the derivation is named.

## 3. The gate

`evaluate_gate(measured_by_baseline_id)` is a
pure evaluation: per gated baseline and per
metric it files the pinned value, the measured
value, the delta, the derived tolerance and a
classification from the filed direction table:

- score-like metrics are **higher-is-better**
  (a beyond-tolerance drop files `regression`,
  a rise files `improvement`);
- `*_loss` metrics are **lower-is-better** (a
  beyond-tolerance rise files `regression`, a
  drop files `improvement`);
- count / total metrics, descriptive
  hit-counts, the descriptive float
  `avg_response_words` and the dict metric
  file **`count_change`** for any
  beyond-tolerance move;
- a baseline or metric that cannot be
  re-measured files **`unmeasurable`**;
- anything inside its tolerance files
  `within_tolerance`.

**Any beyond-tolerance move is a gate
failure** — regressions, improvements and
count changes alike. An improvement is a moved
baseline that must be re-pinned deliberately
by a slice, never passed silently:
`baseline_problems()`' filed precedent, now
with verdict semantics.

`run_regression_gate(baseline_ids=None)`
re-measures live through the harness measurers
and evaluates. The verdict rule:

- **PASS** only when a FULL selection (all
  nine gated baselines) re-measures and every
  metric is within tolerance;
- **FAIL** when any metric is beyond tolerance
  or any baseline fails to re-measure, on a
  full selection;
- a **partial** selection files **REVIEW**
  however it scores, and an **empty** selection
  files 0 baselines / REVIEW — no vacuous pass.

**On this tree the gate files PASS**: 9
baselines gated, 83 metrics compared, 0
failures.

**The probes** (`GATE_PROBES`, filed as data,
run by the checker against the pure evaluator
with synthetic perturbed measurements — real
baselines are never perturbed) prove the gate
can fail:

| Probe | Perturbation | Outcome |
|---|---|---|
| score dropped | persona suite overall_score −0.1 | FAIL, regression |
| score raised | persona suite overall_score +0.1 | FAIL, improvement |
| count changed | persona suite total_turns +1 | FAIL, count_change |
| loss dropped | quant panel q4_k_m_loss −0.1 | FAIL, improvement |
| dict changed | profile baseline quality_tiers → {weak 9, moderate 1} | FAIL, count_change |
| inside tolerance | persona consistency_score +5e-10 | PASS-shaped, within_tolerance |
| baseline missing | memory suite dropped | FAIL, unmeasurable |

All 7 probes pass as filed.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `regression_gate` benchmark entry
  (kind `gate`, claims axes depth / presence /
  will) bound live to
  `benchmarks.regression_gate.run_regression_gate`,
  with a new `regression_gate_corpus` fixture
  binding `NOISE_ROWS` — the estate now files
  **17 benchmarks / 10 fixtures / 10 baselines
  / 89 metric definitions**;
- `METRIC_DEFINITIONS` extended with the gate's
  six metrics (`baselines_gated`,
  `metrics_compared`, `max_noise_spread`,
  `gate_failures`, `probes_passed`,
  `probes_total`), each stating what it
  computes;
- a new `HARNESS_RULES` entry,
  `regression_gates_fail_on_moved_baselines`
  (the rules now file 10), pinned in
  `harness_problems()`' needed list in the
  open;
- the filed-limits passages that listed
  TF-129's regression gates as standing future
  work now file them DONE (all seven places),
  with TF-134's baseline book and TF-137's
  one-command tool still standing as filed;
- TF-121 + TF-123 + TF-124 + TF-125 + TF-126 +
  TF-127 + TF-128's test pins updated in the
  open for the new counts;
- a new pinned baseline, measured live on this
  VM twice, identical, latency pinned nowhere:

**`regression_gate_synthetic_v1`** — baselines
gated 9, metrics compared 83, max noise spread
0.0, gate failures 0, probes 7/7 passed; its
filed target is the gate truth (gate failures
at 0, met). (**Every earlier pinned baseline
is preserved exactly** — persona
0.7166666666666667, persona suite
0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557,
presence suite 0.7777777777777778, honesty
panel per-tier means, quant panel per-level
scores, profile panel per-profile scores, the
empty-store profile baseline — never
re-pinned; `baseline_problems()` is clean over
all ten.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-14 — FIXED by this slice.** The
  gap as filed by this slice's probes: pinned
  baselines existed and `baseline_problems()`
  named moves, but no gate consumed them —
  nothing classified deltas, nothing could
  fail a slice, and the tolerances were
  uniform and asserted rather than derived
  from measured baseline noise. The gap was
  filed as future work in `harness.py`'s
  limits passages and assigned to TF-129 by
  the roadmap without ever being numbered —
  the D-BENCH-9..13 pattern. It is fixed here:
  noise measured and filed as literal rows,
  tolerances derived by a filed rule, deltas
  classified, a verdict a slice gate fails on,
  and seven probes proving the gate can fail —
  registered in the harness in the open with a
  pinned baseline.
- The generation-time share named in
  D-BENCH-9's docket note (the Part V / TF-129
  line was to take or refuse it openly) is
  **refused, openly**: this gate fails a slice
  whose benchmarks moved; it does not block
  generation, and citation-truth blocking at
  generation time is not this slice.
- No other entry moves. TF-016 totals are
  therefore 129 entries — **34 OPEN, 1
  OPEN-ENVIRONMENT, 91 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-14 added + FIXED;
  its owner part VII counted). Counting note,
  in TF-128's manner: the docket's
  header-vs-row counting mismatches are
  pre-existing and are preserved unchanged in
  shape (the primary-owner part counts now sum
  127 against 129, D-BENCH-14's owner part VII
  counted), noted here rather than silently
  "repaired".

## 6. Filed limits

- The gate judges the **harness's pinned
  fixture baselines only** — synthetic fixtures
  through the real scorers (TF-017). **No live
  model is gated**; live behaviour is NOT
  PROVEN (TF-173).
- Noise was measured **in one process on this
  VM** (five repeats, in-process).
  Cross-process and cross-machine noise are
  not measured here — TF-135's flake
  discipline owns that share, and nothing here
  claims it.
- The gate is **a checker a slice runs**
  (`regression_gate_problems()` /
  `run_regression_gate()` and their tests).
  Wiring it into any external CI is NOT done
  and NOT claimed — the house runs manual-only
  workflows and in-house verification.
- TF-129 does **not** wire generation-time
  enforcement blocking (see §5): the gate sits
  in no generation path.
- The gate baseline pins the filed gate only:
  the gate's own baseline is judged by this
  module's checker, not gated by the gate (a
  gate does not gate its own verdict row).

Proof grade: FIXTURE — the harness's pinned
synthetic-fixture baselines re-measured
through their real scorers on this VM. No live
model was run or gated; live behaviour is NOT
PROVEN, never claimed.
