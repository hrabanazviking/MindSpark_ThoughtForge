# ThoughtForge Architecture — Quant-vs-Quality Panel (TF-127)

Quant-vs-quality panel (consumes TF-067
mechanically). Part VII's seventh slice, built on
TF-067's quant quality fixtures and TF-121's
harness. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/quant_panel.py` (`PANEL_ROWS`,
`KIND_ROWS`, `FINDING`, `PANEL_RULES`, checked by
`quant_panel_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline.
The tests are `tests/test_tf127_quant_panel.py`.
This slice **adds and fixes D-BENCH-13** in the
measuring sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **TF-067's estate re-measured clean.**
  `quant_quality_problems()` returned [] and
  `fixture_panel()` reproduced its published
  `CURVE` exactly: iq2_xs 0.0 (loss 1.0, 0
  perfect, 7 contradiction prompts), q2_k 0.4375
  (loss 0.5625, 2, 2), q4_k_m 0.8125 (loss
  0.1875, 6, 1), q5_k_m 0.7083 (loss 0.2917, 4,
  1), q8_0 1.0 (loss 0.0, 8, 0), fp16 1.0 (loss
  0.0, 8, 0) — with the one filed non-monotone
  step (q4_k_m out-scores q5_k_m) standing.
- **No per-kind row existed anywhere.** The
  prompts carry kinds — fact 5, reasoning 2,
  instruction 1, counted live from
  `QUANT_PROMPTS` — and TF-067's runner files
  per-prompt scores, but nothing had ever broken
  the curve down by kind, and no harness baseline
  pinned the curve: TF-067 stood in the harness
  as a bound neighbour whose suite-slice note
  read "TF-127 consumes it mechanically",
  unbuilt (see §5).
- **No live model can run on this VM.** No GGUF
  in the tree, no `llama_cpp` (TF-067 / TF-078's
  filings, re-probed). TF-067's doc imagined
  TF-127 consuming the runner "with live models",
  and TF-068's quant policy filed review triggers
  that TF-127's *live* panel may re-rank the
  neighbouring levels and calibrate the q2_k
  exceptions. That live panel cannot run here,
  and this slice does not pretend otherwise:
  **no live ranking and no live calibration is
  claimed** — TF-068's shares stand open on the
  live road (§6). What runs is TF-067's fixture
  panel, consumed mechanically, broken down by
  kind. Live quant-vs-quality stays NOT PROVEN.

## 2. The panel

`benchmarks/quant_panel.py` consumes TF-067
mechanically: it imports `run_quant_panel`,
`fixture_generate`, `fixture_panel`,
`QUANT_LEVELS`, `QUANT_PROMPTS`,
`REFERENCE_QUANT`, `CURVE`, `NON_MONOTONE_STEPS`,
`score_answer`'s estate and
`quant_quality_problems`, and reuses every one.
It copies **no corpus, no scorer and no curve** —
edit TF-067's corpus and this panel's live
recomputation drifts with it, and the checker
names that drift.

**Overall rows** (TF-067's curve in row form;
exact values are the literal `PANEL_ROWS`,
recomputed live by the checker):

| Level | Mean | Loss vs fp16 | Perfect | Contradictions |
|---|---|---|---|---|
| iq2_xs | 0.0000 | 1.0000 | 0 | 7 |
| q2_k | 0.4375 | 0.5625 | 2 | 2 |
| q4_k_m | 0.8125 | 0.1875 | 6 | 1 |
| q5_k_m | 0.7083 | 0.2917 | 4 | 1 |
| q8_0 | 1.0000 | 0.0000 | 8 | 0 |
| fp16 | 1.0000 | 0.0000 | 8 | 0 |

**Per-kind rows** (the panel's own value beyond
TF-067: the same live run broken down per kind,
each kind's loss against that kind's own fp16
mean; exact values are the literal `KIND_ROWS`):

| Level | Fact (5) | Reasoning (2) | Instruction (1) |
|---|---|---|---|
| iq2_xs | 0.0000 (loss 1.0000, 0 perfect, 5 contra) | 0.0000 (1.0000, 0, 2) | 0.0000 (1.0000, 0, 0) |
| q2_k | 0.5000 (0.5000, 1, 1) | 0.5000 (0.5000, 1, 0) | 0.0000 (1.0000, 0, 1) |
| q4_k_m | 0.9000 (0.1000, 4, 0) | 0.5000 (0.5000, 1, 1) | 1.0000 (0.0000, 1, 0) |
| q5_k_m | 0.8000 (0.2000, 3, 0) | 0.5000 (0.5000, 1, 1) | 0.6667 (0.3333, 0, 0) |
| q8_0 | 1.0000 (0.0000, 5, 0) | 1.0000 (0.0000, 2, 0) | 1.0000 (0.0000, 1, 0) |
| fp16 | 1.0000 (0.0000, 5, 0) | 1.0000 (0.0000, 2, 0) | 1.0000 (0.0000, 1, 0) |

**Verdict PASS** — on the filed rule only: all
48 cells (6 levels x 8 prompts) are measured,
the overall rows reproduce TF-067's published
`CURVE` **exactly** (rounded to 4 places, exactly
as TF-067 files them), and the kind rows
reproduce `KIND_ROWS`. An empty panel or
selection scores 0.0 / REVIEW — no vacuous pass.
A selection without the fp16 reference raises
TF-067's own refusal (a loss curve with no
measured reference is a claim, not a
measurement); a partial prompt subset runs
mechanically through the same runner, measures
its cells, and files REVIEW.

## 3. The finding, and the honest shapes

**Reasoning is where quantisation loss lives
longest.** At both mid levels reasoning carries
the largest per-kind loss — 0.5 at q4_k_m and
0.5 at q5_k_m, against fact's 0.1 / 0.2 and
instruction's 0.0 / 0.3333 — and reasoning's mean
**plateaus at 0.5 from q2_k through q5_k_m**,
recovering only at q8_0 (1.0). On this corpus
one of the two reasoning prompts (the oil-maths
prompt) is contradicted at every level from
q2_k to q5_k_m; the fare prompt is answered
perfectly at q2_k and not at q4_k_m. Fact, by
contrast, recovers early (0.9 already at
q4_k_m).

- **Instruction collapses at q2_k (0.0)** while
  fact and reasoning still hold 0.5 there: its
  single prompt is contradicted outright (a
  white lantern named for the closed crossing).
  That row rests on **one prompt** — it is a
  fixture shape, filed by name, not a ranking
  instrument (§6).
- **The non-monotone step stands**: q4_k_m
  (0.8125) out-scores q5_k_m (0.7083) overall,
  exactly as TF-067 filed it in
  `NON_MONOTONE_STEPS` — carried in the rows and
  the finding, recomputed live, never smoothed.
  Per kind, the step is fact's (0.9 vs 0.8) and
  instruction's (1.0 vs 0.6667); reasoning is
  flat across it.
- **iq2_xs scores 0.0 on every kind**, with 7 of
  8 prompts contradicted — the corpus's bottom
  level fails by contradiction, not by partial
  credit, on fact and reasoning alike (the one
  uncontradicted cell, instruction, still scores
  0.0 on retention).
- **q8_0 is at parity with fp16 on every kind**
  (1.0 / loss 0.0 across the board) — on this
  8-prompt fixture corpus, stated as exactly
  that and no more.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `quant_panel` benchmark entry (kind
  `panel`, claims axis depth) bound live to
  `benchmarks.quant_panel.run_quant_panel`, with
  a new `quant_panel_corpus` fixture — the estate
  now files **15 benchmarks / 8 fixtures / 8
  baselines / 70 metric definitions**;
- the `quant_quality` neighbour entry's
  suite-slice note now files TF-127 DONE in the
  measuring sense (its doc's live-panel share
  stands open, §6);
- `METRIC_DEFINITIONS` extended with the panel's
  sixteen metrics (per level `<level>_score` /
  `<level>_loss` for the six levels, the three
  per-kind means across levels
  `fact_mean_score` / `reasoning_mean_score` /
  `instruction_mean_score`, plus `total_cells`),
  each stating what it actually computes;
- TF-121 + TF-123 + TF-124 + TF-125 + TF-126's
  test pins updated in the open for the new
  counts;
- a new pinned baseline, measured live on this
  VM twice, identical, latency pinned nowhere:

**`quant_panel_synthetic_v1`** — per-level scores
iq2_xs 0.0, q2_k 0.4375, q4_k_m 0.8125, q5_k_m
0.7083, q8_0 1.0, fp16 1.0 with their losses,
per-kind means fact 0.7000 / reasoning 0.5833 /
instruction 0.6111, 48 cells; its filed target
is the curve-reproduction truth (the fp16
reference row is pinned at its measured 1.0,
met). (**Every earlier pinned baseline is
preserved exactly** — persona 0.7166666666666667,
persona suite 0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557,
presence suite 0.7777777777777778, honesty panel
per-tier means, the empty-store profile baseline
— never re-pinned; `baseline_problems()` is
clean over all eight.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-13 — added and FIXED by this slice,
  in the measuring sense.** The defect, as probed
  in §1: quant-vs-quality across levels was
  measured only as TF-067's single overall curve
  in `src/` — no Part VII panel had ever consumed
  that runner, no per-kind breakdown existed
  (the prompts' kinds were filed and the runner
  scored per prompt, and nothing aggregated
  either), and no harness baseline pinned the
  curve. TF-067's own doc assigned the panel to
  TF-127 without the gap ever being numbered in
  this docket — the D-BENCH-9 / D-BENCH-10 /
  D-BENCH-11 / D-BENCH-12 pattern. This slice
  numbers it and fixes the measuring share the
  roadmap assigns to TF-127:
  `benchmarks/quant_panel.py` consumes TF-067's
  runner mechanically, publishes the overall
  rows + the per-kind breakdown as literal data
  recomputed live, and registers a pinned harness
  baseline.
- No other entry moves. TF-016 totals are
  therefore 128 entries — **35 OPEN, 1
  OPEN-ENVIRONMENT, 89 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-13 added + FIXED in
  the measuring sense). Counting note, in
  TF-126's manner: the docket's header-vs-row
  counting mismatches are pre-existing, are
  preserved unchanged in shape (parts now sum
  126 against 128, D-BENCH-13's owner part VII
  counted), and are noted here rather than
  silently "repaired".

## 6. Filed limits

- The panel measures **TF-067's fixture corpus
  only** — 8 synthetic prompts (TF-017), 48
  cells, through TF-067's real scorer and
  runner; `data/` was never opened. **Live
  quant-vs-quality of real models is NOT
  PROVEN** — no GGUF in the tree, no `llama_cpp`
  on this VM (TF-078); TF-173 owns the
  real-hardware road.
- **TF-068's live shares stand open**: its
  review triggers name TF-127's *live* panel as
  what may re-rank the neighbouring levels and
  calibrate the q2_k profile exceptions
  (D-PROF-5's line). No live panel ran, so
  nothing is re-ranked and q2_k is not
  calibrated by this slice — the policy stands
  exactly as TF-068 filed it.
- The per-kind counts are small and uneven by
  TF-067's filing (fact 5 / reasoning 2 /
  instruction 1): a kind mean is a fixture
  shape, not a ranking instrument, the
  instruction rows rest on a single prompt, and
  **ranking neighbouring levels is not supported
  by 8 prompts** (TF-067's own filing, carried
  here) — the non-monotone q4_k_m / q5_k_m step
  is the standing proof of that limit.
- The panel baseline pins the filed panel only.

Proof grade: FIXTURE — TF-067's synthetic fixture
corpus through its real scorer and runner on
this VM. No live model was run; that is NOT
PROVEN, never claimed.
