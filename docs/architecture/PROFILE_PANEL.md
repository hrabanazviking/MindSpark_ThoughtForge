# ThoughtForge Architecture — Profile Panels (TF-128)

Profile panels: suites per hardware profile with
the profile's budgets enforced during the run.
Part VII's eighth slice, built on TF-032 /
TF-037 / TF-056's profile application in the
core, TF-069's profile runner, and TF-121's
harness. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/profile_panel.py` (`PANEL_ROWS`,
`FINDING`, `PANEL_RULES`, checked by
`profile_panel_problems()`), registered in
`benchmarks/harness.py` with a pinned baseline.
The tests are `tests/test_tf128_profile_panel.py`.
This slice **fixes D-BENCH-1** in the measuring
sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The core applies a selected profile,
  exactly as filed.** For every shipped profile,
  a core built with
  `config={"conversation": {"profile_id": …}}`
  resolved `max_response_tokens` / refinement
  passes to the profile JSON's own generation
  values — pi_zero 120 / 0, phone_low 180 / 1,
  pi_5 250 / 2, desktop_cpu 400 / 2, desktop_gpu
  600 / 3, server_gpu 2000 / 3 — and its
  lifecycle caps equalled TF-056's
  `budget_for_profile` caps for that profile.
  All six, measured live (TF-032 / TF-037 /
  TF-056's D-PROF-3 narrowings, standing in the
  core since those slices).
- **The benchmark still did not apply one.**
  `ProfileBenchmark`'s ephemeral core is built
  with no profile config at all
  (`benchmark_profiles`' own filing since
  TF-121), so per-profile runs measured the same
  core whatever id they were handed — the
  application share of D-BENCH-1, standing
  exactly as TF-121 filed it for TF-128.
- **TF-069's ceiling rows stand**, re-measured
  live through `check_model_fit`: desktop_cpu
  (7B q4_k_m accounts 4.6667 GB ≤ 8 GB), pi_5
  (2.0 ≤ 4) and desktop_gpu (13B q8_0, 14.3 ≤ 28)
  caps fit; server_gpu's cap **exceeds its own
  ceiling** (70B at its fp16 default accounts
  140.0 GB against 64 + 24 = 88 GB declared);
  phone_low / pi_zero caps are **unprovable**
  in-tree (their q2_k default has no catalogue
  calibration — TF-069's filing, D-PROF-5).

## 2. The panel

`benchmarks/profile_panel.py` runs, per shipped
profile (the harness's `KNOWN_PROFILES` six),
five filed cells — the profile benchmark plus
the four Part VII suites — with the profile's
budgets enforced during the run by real
machinery only:

- **The benchmark cell** runs
  `ProfileBenchmark` over its filed default
  queries on a `ThoughtForgeCore` built with the
  profile selected, supplied through the
  runner's existing `core=` parameter. The cell
  counts only when that core's resolved
  `max_response_tokens` + refinement passes and
  its lifecycle caps equal the profile's filed
  budgets (`budgets_applied`, measured per
  profile), and TF-069's `run_profile_boot` of
  the same profile is re-run beside it — its
  verdict must reproduce TF-069's filed
  `BOOT_VERDICTS` (only desktop_gpu files a
  gap-free `boots`; the other five file
  `boots_with_ceiling_gaps`, for TF-069's named
  ceiling / recommendation reasons).
- **The RAM ceiling** is TF-069's
  `check_model_fit` cap row per profile (its own
  `max_params_b` at its default quant against
  ram + vram), filed in the row and reproduced
  against `RAM_ROWS`: fit × 3, exceeds_ceiling
  × 1 (server_gpu), unprovable × 2 (phone_low,
  pi_zero). A gap is named in the row, never
  passed silently.
- **The suite cells** run persona / memory /
  depth / presence through their real
  `run_*_suite` runners per profile. One budget
  bites inside the suites: the depth suite's
  scaffold (ON) road runs only where the profile
  budgets at least one refinement pass.

**Per-profile rows** (exact values are the
literal `PANEL_ROWS`, recomputed live by the
checker; budgets = tokens / passes / bundle):

| Profile | Budgets | Cap (TF-069) | Boot | Suite score |
|---|---|---|---|---|
| desktop_cpu | 400 / 2 / 15 | fit (4.6667 ≤ 8 GB) | boots_with_ceiling_gaps | 0.8086 |
| desktop_gpu | 600 / 3 / 15 | fit (14.3 ≤ 28 GB) | boots | 0.8086 |
| phone_low | 180 / 1 / 5 | unprovable (q2_k) | boots_with_ceiling_gaps | 0.8086 |
| pi_5 | 250 / 2 / 10 | fit (2.0 ≤ 4 GB) | boots_with_ceiling_gaps | 0.8086 |
| pi_zero | 120 / 0 / 3 | unprovable (q2_k) | boots_with_ceiling_gaps | 0.7252 |
| server_gpu | 2000 / 3 / 30 | exceeds (140.0 > 88 GB) | boots_with_ceiling_gaps | 0.8086 |

Every profile's benchmark cell files the
empty-store shape: 10/10 successful, citation
0.0, enforcement 0.0, 9 avg words, confidence
0.408. Every measured suite cell files the
suites' pinned scores (persona 0.8454, memory
0.8056, depth OFF 0.4722 / ON 0.8056, presence
0.7778).

**Verdict PASS** — on the filed rule only: all
30 cells (6 profiles × benchmark + 4 suites)
are filed, every applied budget matches its
profile's filing exactly, TF-069's boot + cap
rows reproduce, and the rows reproduce
`PANEL_ROWS`. One cell is **refused by budget**
(pi_zero's depth ON road, §3). An empty panel
or selection scores 0.0 / REVIEW — no vacuous
pass; a partial selection measures its rows and
files REVIEW.

## 3. The finding, and the honest shapes

**The budgets are real, and on this estate they
change limits, not scores.** Applying each
profile moves the core's filed limits exactly
as the JSONs declare — response tokens from 120
(pi_zero) to 2000 (server_gpu), refinement
passes 0–3, bundle facts 3–30, retrieval caps
with them — proven live in the applied core on
all six profiles. But:

- **No score varies by profile except through
  the refusal.** The benchmark cell is an
  empty-store run: the refusal text is the same
  9 words whatever the budgets, because an empty
  store gives the budgets nothing to bite on.
  The suite scorers take no profile input, so
  their per-profile scores are identical by
  construction of the estate. The panel files
  that identity plainly (`bench_scores_vary_by_profile`
  false; suite scores vary only without-the-refused-cell
  false) — it is a measurement of where the
  estate stands, not a per-profile quality claim.
- **pi_zero's depth ON road is refused by
  budget.** pi_zero files
  `max_refinement_passes: 0` — a scaffolded run
  is exactly what its budget forbids (TF-037's
  floor: the core hands that 0 to salvage). The
  cell files `refused_by_budget` with the budget
  named, `depth_on_score` is None, and pi_zero's
  depth score is its single-pass OFF 0.4722 —
  which is why its suite score (0.7252) is the
  panel's only different row. A refusal, never a
  silent pass and never a borrowed ON score.
- **The ceiling gaps stand in the rows**:
  server_gpu promises a 70B fp16 cap its own
  declared memory cannot hold (140.0 > 88 GB —
  TF-069 / D-PROF-5's finding, carried), and the
  two q2_k profiles' caps cannot be accounted
  in-tree at all. The stacks still boot (in
  simulation) — the gaps are filed as gaps, in
  TF-069's own verdict language.

## 4. Harness registration

In `benchmarks/harness.py`, in the open:

- a new `profile_panel` benchmark entry (kind
  `panel`, claims axes depth / presence / will)
  bound live to
  `benchmarks.profile_panel.run_profile_panel`,
  with a new `profile_panel_corpus` fixture —
  the estate now files **16 benchmarks / 9
  fixtures / 9 baselines / 83 metric
  definitions**;
- the `profile_benchmark` entry's suite-slice
  note now files TF-128 DONE, and its docket
  line files D-BENCH-1 FIXED by TF-128 in the
  measuring sense; the harness rule
  `an_unknown_profile_is_named_never_run` now
  files the application share discharged here;
- `METRIC_DEFINITIONS` extended with the panel's
  thirteen metrics (per profile
  `<profile>_suite_score` — the mean of its four
  suite headline scores, depth's headline being
  its ON score where measured and its OFF score
  where the ON road is refused — and
  `<profile>_bench_avg_words`, plus
  `total_cells`), each stating what it computes;
- TF-121 + TF-123 + TF-124 + TF-125 + TF-126 +
  TF-127's test pins updated in the open for the
  new counts;
- a new pinned baseline, measured live on this
  VM twice, identical, latency pinned nowhere:

**`profile_panel_synthetic_v1`** — per-profile
suite scores (five profiles at 0.8086, pi_zero
at 0.7252 through its refused depth ON road),
per-profile benchmark avg words (9 on every
profile — the empty-store refusal), 30 cells;
its filed target is the reproduction truth (the
desktop_cpu suite score pinned at its measured
0.8086, met). (**Every earlier pinned baseline
is preserved exactly** — persona 0.7166666666666667,
persona suite 0.8453703703703704, memory suite
0.8055555555555557, depth suite OFF
0.47222222222222227 → ON 0.8055555555555557,
presence suite 0.7777777777777778, honesty panel
per-tier means, quant panel per-level scores,
the empty-store profile baseline — never
re-pinned; `baseline_problems()` is clean over
all nine.)

## 5. Docket dispositions (TF-016)

- **D-BENCH-1 — FIXED by this slice, in the
  measuring sense.** The defect as filed: the
  profile id was never validated (TF-121 fixed
  that share) and never applied — every profile
  measured the same ephemeral empty-store core.
  The validation share was already fixed; the
  application share is discharged here in the
  measuring sense the roadmap assigns to
  TF-128: the Part VII suites and the profile
  benchmark now run per shipped profile with
  the profile applied to a real core and its
  budgets enforced during the run (applied-core
  equality asserted live, TF-069's boot + RAM
  ceiling re-run per profile, the one
  budget-refused cell filed by name), published
  as literal rows recomputed live with a pinned
  harness baseline. Stated limits stand in §6:
  `benchmark_profiles`' own default ephemeral
  core is unchanged (the panel supplies the
  applied core through its existing `core=`
  parameter), and no new D-BENCH entry was
  needed — every gap probed was already
  numbered (D-BENCH-1 itself, D-PROF-3's
  remainders, D-PROF-5's ceiling rows).
- No other entry moves. TF-016 totals are
  therefore still 128 entries — **34 OPEN, 1
  OPEN-ENVIRONMENT, 90 FIXED, 3
  CLOSED-BY-VERDICT** (D-BENCH-1 OPEN → FIXED in
  the measuring sense). Counting note, in
  TF-127's manner: the docket's header-vs-row
  counting mismatches are pre-existing and are
  preserved unchanged in shape (the primary-owner
  part counts still sum 126 against 128 — a
  status move inside Part VII changes no part
  count), noted here rather than silently
  "repaired".

## 6. Filed limits

- The benchmark cell measures an **ephemeral
  core over an empty store** (`model_path=None`)
  on every profile — the empty-store refusal,
  never inference, never a loaded model. The
  budgets are proven live in the core; their
  effect on **seeded-store** quality per profile
  is NOT PROVEN here (TF-134's baseline book
  owns the seeded road).
- **Live per-profile hardware behaviour is NOT
  PROVEN**: every row ran on this VM; the pi /
  phone profiles ran in TF-069's simulation
  sense (hardware declared by the profile,
  backends the host's truth). No phone, Pi or
  server GPU ran — TF-082 / TF-173 own the
  real-hardware road.
- The suite scorers take no profile input: apart
  from the budget-refused depth ON road, the
  panel proves the suites *run* per profile and
  files their (identical) scores — it does not
  claim a profile changes a scripted suite's
  outcome.
- `benchmark_profiles`' default ephemeral core
  still applies no profile when a caller passes
  no core — unchanged by this slice, stated
  openly; the panel's road (an applied core
  through the existing `core=` parameter) is
  the estate's measured application path.
- The panel baseline pins the filed panel only.

Proof grade: FIXTURE/SIMULATION — synthetic
scripted corpora and an empty-store core through
the real machinery on this VM, profiles applied
and budgets enforced as filed. No live model was
run; real per-profile hardware is NOT PROVEN,
never claimed.
