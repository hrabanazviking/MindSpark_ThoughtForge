# The Cost Panel — TF-133

The proof lives as verifiable data in
`benchmarks/cost_panel.py` (`PANEL_ROWS`,
`FINDING`, `PANEL_PROBES`, `COST_PANEL_RULES`,
checked by `cost_panel_problems()`), registered
in `benchmarks/harness.py` with the pinned
baseline `cost_panel_synthetic_v1`. The tests
are `tests/test_tf133_cost_panel.py`. This
slice **adds and fixes D-BENCH-18** in the
measuring sense, and **fixes D-BENCH-7's
consumption share** (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The estate stood clean, and no cost panel
  existed.** `harness_problems()` returned []
  and `baseline_problems()` returned [] over
  the thirteen pinned baselines;
  `regression_gate_problems()` returned [] with
  the live gate PASS (9 baselines / 83 metrics /
  0 failures); `data_hygiene_problems()` returned
  [] (21 inputs, all clean);
  `spot_checks_problems()` returned [] and
  `wyrd_panel_problems()` returned [] with the
  live WYRD panel PASS (18 cases, overall
  0.5556). And a grep for any cost / tokens /
  time-per-suite-per-profile panel across
  benchmarks / docs / tests / src found nothing
  but TF-121's assignment of exactly this share
  to TF-133 (latency deliberately NOT pinned
  there — walls under contention are not
  baselines; TF-133 owns cost measurement).
- **No shipped profile declares a power
  figure.** Grepped live: no `power` / `watt` /
  `tdp` key or value anywhere in
  `hardware_profiles/*.json` or the profile
  loaders. "Sharpness per watt" therefore has
  no honest denominator on this estate (§3).
- **The suite scorers are millisecond-scale
  scripted work on this VM**, probed live with
  `perf_counter` (3 repeats each): persona
  ~7–59 ms, memory ~13–69 ms, depth ~1–2 ms,
  presence ~6–7 ms per run, first runs slowest
  — the warm-up / contention variance TF-121
  filed, in the flesh.
- **D-BENCH-18 was verified unused** anywhere
  in benchmarks / docs / tests / src before
  filing (the D-BENCH-9..17 pattern: a gap
  assigned by the roadmap without ever being
  numbered).

## 2. What the panel measures

The panel runs over TF-128's estate: the four
Part VII suites (persona, memory, depth,
presence) through their **real runners** × the
six shipped hardware profiles, budgets read
**live from the profile JSONs** through
TF-128's `filed_budgets` — bind, never copy.
TF-128's 30 cells are these 24 suite cells plus
its 6 profile-benchmark cells; the benchmark
cell is not a suite and files no suite score,
so it is not a cost row here. Per cell:

- **Tokens — a budget exposure, filed as
  exactly that.** `token_budget` = scored units
  (counted by the suite's real runner: persona
  108 turns; memory / depth / presence 36 cases
  each) × the profile's filed
  `max_response_tokens`. It is the response-token
  budget one scored response per unit may spend
  under that profile — **not** a consumed-token
  count: the suites score scripted fixtures, no
  live model generated anything, and no consumed
  tokens exist to count.
- **Sharpness — derived only from real suite
  scores.** `sharpness_per_1000_tokens` = suite
  score / token budget × 1000, the headline
  score being TF-128's (the depth ON score
  where the profile budgets a refinement pass;
  pi_zero's 0-pass budget refuses that road, so
  its depth headline is the OFF score, filed
  `refused_by_budget`).
- **Time — repeat-measured live, pinned
  nowhere.** `run_cost_panel()` times every
  cell's suite runner 5 times with
  `perf_counter` and records every sample into
  TF-121 / D-BENCH-7's `PerfTracker`
  (`thoughtforge.utils.perf` — consumed here,
  in the open), filing the tracker's summary
  (repeats / mean / median / min / max ms) at
  run time only. It is this VM's *scorer* wall
  time under whatever contention the VM
  carries: not a portable baseline, not a
  per-profile hardware speed (the profile
  changes budgets, not scorer code), and the
  checker judges its **shape** (present,
  repeated, positive, median inside
  [min, max]), never its values.

**The measured filing (deterministic rows):
24 cells, 1,296 scored units, total token
budget 766,800 — verdict PASS** (the full
matrix reproduces every filed row and every
timing is well-shaped). pi_zero files the
highest mean sharpness per budgeted token
(0.1353) and server_gpu the lowest (0.00927) —
budget arithmetic over identical scores, filed
as exactly that in §3, never as hardware
efficiency.

## 3. The watt honesty call, and the finding

The roadmap names this slice "sharpness per
watt, the edge-hardware point of the project".
The honest filing is:

- **Watts are NOT PROVEN / not measured.**
  No shipped profile declares a power figure
  (the checker re-scans the profile JSONs live
  for power / watt / tdp keys), no power was
  measured on any hardware, and **no watt is
  invented anywhere in this panel**: every row
  files `power_watts: None` and
  `power_status: "not_proven_not_measured"`,
  and the pure checker *fails* a row that
  claims a watt or a measured status
  (probe-proven, §4). If a future profile
  declares a power figure, a future slice may
  file it as **declared, not measured**, by
  binding that JSON live.
- **What stands instead is sharpness per
  budgeted token** — and the edge-hardware
  point survives honestly: the narrowest
  profile (pi_zero, 120 tokens/response,
  total budget 25,920) files the highest
  sharpness per budgeted token on every suite,
  and the widest (server_gpu, 2,000
  tokens/response, total budget 432,000) the
  lowest, because the suite scores do not vary
  by profile except through pi_zero's refused
  depth ON road (TF-128's measured shape:
  `suite_scores_vary_without_the_refused_cell`
  is false). That is a statement about filed
  budgets, not about silicon.
- **Time at banking, filed as colour only.**
  One full live run at banking filed per-cell
  median scorer walls whose per-suite ranges
  across the six profiles were: persona
  4.036–32.533 ms, memory 11.227–84.688 ms,
  depth 0.517–1.252 ms, presence
  4.021–89.667 ms (5 repeats per cell, medians
  of the PerfTracker summaries). The spread
  *within* a suite across identical-work cells
  is the contention noise itself — the reason
  no time figure is pinned, in numbers.

## 4. The panel, filed in full

The render below is generated from
`benchmarks/cost_panel.py` and filed verbatim —
`cost_panel_problems()` requires it in this
doc exactly as rendered. Time is not rendered
(live-only, §2) and watts are not rendered
(NOT PROVEN, §3).

## The filed rows (deterministic: tokens + sharpness)

24 cells: 4 suites x 6 shipped profiles. Token budget = scored units x the profile's filed max_response_tokens (a budget exposure, not consumed tokens); sharpness = suite score / token budget x 1000. Power files not_proven_not_measured on every row (no watt filed, none invented).

| Profile | Suite | Units | Max response tokens | Token budget | Suite score | Sharpness / 1000 tokens | Status |
|---|---|---|---|---|---|---|---|
| desktop_cpu | persona | 108 | 400 | 43200 | 0.8453703703703704 | 0.019568758573388204 | measured |
| desktop_cpu | memory | 36 | 400 | 14400 | 0.8055555555555557 | 0.05594135802469137 | measured |
| desktop_cpu | depth | 36 | 400 | 14400 | 0.8055555555555557 | 0.05594135802469137 | measured |
| desktop_cpu | presence | 36 | 400 | 14400 | 0.7777777777777778 | 0.05401234567901235 | measured |
| desktop_gpu | persona | 108 | 600 | 64800 | 0.8453703703703704 | 0.01304583904892547 | measured |
| desktop_gpu | memory | 36 | 600 | 21600 | 0.8055555555555557 | 0.037294238683127576 | measured |
| desktop_gpu | depth | 36 | 600 | 21600 | 0.8055555555555557 | 0.037294238683127576 | measured |
| desktop_gpu | presence | 36 | 600 | 21600 | 0.7777777777777778 | 0.0360082304526749 | measured |
| phone_low | persona | 108 | 180 | 19440 | 0.8453703703703704 | 0.0434861301630849 | measured |
| phone_low | memory | 36 | 180 | 6480 | 0.8055555555555557 | 0.1243141289437586 | measured |
| phone_low | depth | 36 | 180 | 6480 | 0.8055555555555557 | 0.1243141289437586 | measured |
| phone_low | presence | 36 | 180 | 6480 | 0.7777777777777778 | 0.12002743484224966 | measured |
| pi_5 | persona | 108 | 250 | 27000 | 0.8453703703703704 | 0.03131001371742113 | measured |
| pi_5 | memory | 36 | 250 | 9000 | 0.8055555555555557 | 0.08950617283950618 | measured |
| pi_5 | depth | 36 | 250 | 9000 | 0.8055555555555557 | 0.08950617283950618 | measured |
| pi_5 | presence | 36 | 250 | 9000 | 0.7777777777777778 | 0.08641975308641975 | measured |
| pi_zero | persona | 108 | 120 | 12960 | 0.8453703703703704 | 0.06522919524462735 | measured |
| pi_zero | memory | 36 | 120 | 4320 | 0.8055555555555557 | 0.1864711934156379 | measured |
| pi_zero | depth | 36 | 120 | 4320 | 0.47222222222222227 | 0.10931069958847738 | refused_by_budget |
| pi_zero | presence | 36 | 120 | 4320 | 0.7777777777777778 | 0.1800411522633745 | measured |
| server_gpu | persona | 108 | 2000 | 216000 | 0.8453703703703704 | 0.003913751714677641 | measured |
| server_gpu | memory | 36 | 2000 | 72000 | 0.8055555555555557 | 0.011188271604938273 | measured |
| server_gpu | depth | 36 | 2000 | 72000 | 0.8055555555555557 | 0.011188271604938273 | measured |
| server_gpu | presence | 36 | 2000 | 72000 | 0.7777777777777778 | 0.010802469135802469 | measured |

## 5. The probes

`PANEL_PROBES` files seven perturbed copies of
filed rows against the pure checker parts
(`cell_problems`, `timing_problems`): a clean
filed cell passes; a wrong token budget is
named; a wrong sharpness is named; an invented
watt is named; a claimed measured power status
is named; an out-of-range score is named; and
a bad timing shape (wrong repeat count, a
non-positive median) is named. Real filings
are never perturbed. Measured: 7/7.

## 6. D-BENCH-18, and D-BENCH-7's share

No benchmark in the estate measured cost:
TF-128 filed what the suites *score* per
profile, but nothing filed what a suite run
*costs* under a profile — no token figure per
suite per profile, no repeat-measured time,
and the tracker TF-121 filed stood consumed by
nothing. TF-133 files the panel above and
fixes the gap in the measuring sense: budget-
derived token cost and repeat-measured scorer
time over the suites × profiles matrix,
registered in the harness in the open with the
pinned baseline `cost_panel_synthetic_v1`
(24 cells / 1,296 units / total budget 766,800 /
pi_zero mean sharpness 0.13526306012802927,
target met; **time and watts are excluded
metrics, filed in the open** — time because
walls are not baselines (TF-121's rule), watts
because they are NOT PROVEN). The baseline is
NOT gated by TF-129's regression gate — the
exclusion is filed in the open in
`benchmarks/regression_gate.py` as
`UNGATED_BASELINE_IDS`, the shape TF-129..
TF-132 filed for their own baselines; the gated
set stays the nine TF-121..TF-128 baselines and
the gate still PASSes (9 / 83 / 0). **D-BENCH-7
is FIXED by this slice in its consumption
share**: the cost panel records every timing
sample into `PerfTracker` and files its
summary, so the core's tracker is no longer
wired-but-unconsumed; TF-121's filing share
(the two systems indexed with their roles)
stands as filed. TF-131 coupling, in the open:
the panel adds **no** benchmark input — it
consumes only already-filed inputs (the four
suites' filed corpora through their real
runners and the shipped hardware-profile
files, both already carrying hygiene rows) —
so the hygiene inventory stays 21 inputs
(15 synthetic / 0 licensed / 6 derived, all
clean), its pinned baseline is NOT re-pinned,
and the panel files no new harness fixture.
Every other earlier pinned baseline is
preserved exactly.

## 7. Filed limits

- **Tokens are budget exposures**, not
  consumed tokens: units × filed
  `max_response_tokens`, one scored response
  per unit by filed convention (the depth
  suite scores each case on both roads over
  the same units). No live model ran; live
  token consumption is NOT PROVEN (TF-173).
- **Time is this VM's scorer wall time**,
  repeat-measured at run time and filed
  live-only: it varies run to run and under
  contention, is pinned nowhere, and measures
  scripted scorers, not model inference on
  any profile's hardware. Live per-profile
  hardware time is NOT PROVEN (TF-082 /
  TF-173).
- **Watts are NOT PROVEN and not measured**:
  no profile declares a power figure and no
  power measurement exists in the estate, so
  sharpness per watt is NOT PROVEN — only
  sharpness per budgeted token is filed.
- **Sharpness differences across profiles are
  budget arithmetic** over scores that do not
  vary except through the refused cell —
  filed as that, never as hardware efficiency.

Proof grade: FIXTURE — scripted synthetic
suites through their real runners and the
shipped profiles' filed budgets on this VM;
no live model, no measured watt, never
claimed.
