# Flake Discipline — TF-135

GENERATED from the filed data in `benchmarks/flake_discipline.py` — the filed doc IS the render: regenerate with `python3 -m benchmarks.flake_discipline`. The checker (`flake_discipline_problems()`) recomputes the census live through the harness measurers and fails on any drift from the filed rows.

Format: `thoughtforge-flake-discipline/1`
Census repeats: 5
Baselines censused: 16
Deterministic baselines: 16
Quarantined baselines: 0
Excluded-by-rule metrics: 6
Flake reports filed: 1
Re-run records filed: 1

Every pinned suite is census-measured by repetition: each baseline is re-measured 5 times in one process through the harness's own measurers, and a suite is deterministic only if all repeats reproduce its pinned metrics exactly — the standard cited to TF-036 determinism mode (thoughtforge.cognition.determinism: seeded generation + turn fingerprints; the harness's determinism neighbour entry). Nondeterminism is fixed or quarantined with a filed reason, never silently tolerated, and no re-roll is silent: a re-run after a differing result files a flake report naming the suite, the metric / leg, the differing values and the disposition.

## The census (filed rows)

| Baseline | Benchmark | Repeats | Metrics | Classification | Differing metrics | Max spread |
|---|---|---|---|---|---|---|
| `persona_consistency_synthetic_v1` | persona_consistency | 5 | 7 | deterministic | — | 0.0 |
| `persona_suite_synthetic_v1` | persona_suite | 5 | 5 | deterministic | — | 0.0 |
| `memory_suite_synthetic_v1` | memory_suite | 5 | 5 | deterministic | — | 0.0 |
| `depth_suite_synthetic_v1` | depth_suite | 5 | 10 | deterministic | — | 0.0 |
| `presence_suite_synthetic_v1` | presence_suite | 5 | 5 | deterministic | — | 0.0 |
| `honesty_panel_synthetic_v1` | honesty_panel | 5 | 13 | deterministic | — | 0.0 |
| `quant_panel_synthetic_v1` | quant_panel | 5 | 16 | deterministic | — | 0.0 |
| `profile_panel_synthetic_v1` | profile_panel | 5 | 13 | deterministic | — | 0.0 |
| `regression_gate_synthetic_v1` | regression_gate | 5 | 6 | deterministic | — | 0.0 |
| `spot_checks_synthetic_v1` | spot_checks | 5 | 6 | deterministic | — | 0.0 |
| `data_hygiene_synthetic_v1` | data_hygiene | 5 | 7 | deterministic | — | 0.0 |
| `wyrd_panel_synthetic_v1` | wyrd_panel | 5 | 7 | deterministic | — | 0.0 |
| `cost_panel_synthetic_v1` | cost_panel | 5 | 10 | deterministic | — | 0.0 |
| `baseline_book_synthetic_v1` | baseline_book | 5 | 6 | deterministic | — | 0.0 |
| `flake_discipline_synthetic_v1` | flake_discipline | 5 | 7 | deterministic | — | 0.0 |
| `profile_benchmark_empty_store_v1` | profile_benchmark | 5 | 9 | deterministic | — | 0.0 |

## Excluded by rule (never flakes)

Timing / wall metrics and the other shapes filed in a baseline's excluded_metrics stay unpinned per TF-121's rule and are classified excluded-by-rule with their filed reasons:

- `cost_panel_synthetic_v1` / `cell_wall_time_ms`: timing varies run to run and under contention; walls are not baselines (TF-121's rule) — the panel files repeat-measured timing live only, never pinned
- `cost_panel_synthetic_v1` / `power_watts`: no shipped profile declares a power figure and no power was measured; watts are NOT PROVEN, never pinned, never invented
- `flake_discipline_synthetic_v1` / `census_wall_time_s`: timing varies run to run and under contention; walls are not baselines (TF-121's rule) — the census files its repeat-measured classifications only, never its own wall
- `profile_benchmark_empty_store_v1` / `avg_latency_ms`: timing varies between probes (avg 7ms / 8ms on this VM) and under contention; walls are not baselines — TF-133 owns cost measurement
- `profile_benchmark_empty_store_v1` / `median_latency_ms`: same rule as avg_latency_ms
- `profile_benchmark_empty_store_v1` / `p95_latency_ms`: same rule as avg_latency_ms

## Quarantine registry

No suite is quarantined on this tree: the census measured every pinned baseline deterministic, so the registry files empty. A differing suite enters here only with its reason filed, beside its census row's differing metrics.

## Filed flake reports

### Report `tf126_full_gate_eperm_reroll`

- Suite: full_gate (tests/test_tf024 + tests/test_tf030 legs)
- Metric / leg: full-suite gate run filed in CAMPAIGN_STATE at TF-126 banking
- Differing values:
  - first full-gate run: 4 failures — transient EPERM write-block errors in the tf024 / tf030 legs
  - immediate re-run: the same legs passed 55/55 and the gate filed clean
- Disposition: environmental sandbox write-block, not a suite nondeterminism: no suite quarantined, no baseline re-pinned; filed retroactively by TF-135 so the re-roll is owned by this discipline and never again settles by the green re-run alone

## Filed re-run records

- Suite: full_gate (tests/test_tf024 + tests/test_tf030 legs) — first: 4 failures (transient EPERM write-block); re-run: passed (55/55 on the affected legs); flake report: `tf126_full_gate_eperm_reroll`

## The rules

- `every_pinned_suite_is_census_measured_by_repetition`: every pinned baseline in the harness is re-measured CENSUS_REPEATS (5 — TF-129's filed NOISE_REPEATS) times in one process through the harness's own measurers, and the census files per baseline its repeats, its metric count, its repeat spreads and its differing metrics — no suite's determinism is assumed from another suite's noise
- `deterministic_means_every_repeat_reproduces_the_pin_exactly`: a suite is deterministic only if every repeat reproduces its pinned metrics exactly — the standard TF-036's determinism mode makes claimable (seeded generation, turn fingerprints); any metric whose repeats differ from the pin or from each other is a differing metric, named in the census row
- `nondeterminism_is_fixed_or_quarantined_never_tolerated`: a differing suite is either FIXED (made deterministic, citing TF-036's determinism mode where relevant) or QUARANTINED in QUARANTINE_REGISTRY with a filed reason — never silently tolerated, never re-pinned to whatever it happened to print, and never hidden behind a re-run
- `no_silent_rerolls`: any re-run after a differing result must be filed as a flake report naming the suite, the metric / leg, the differing values and the disposition — a green re-run alone settles nothing; a re-run recorded with no report is a silent re-roll and the checker names it
- `timing_and_wall_metrics_are_excluded_by_rule_not_flakes`: timing / wall metrics (and the other shapes filed in a baseline's excluded_metrics) stay unpinned per TF-121's rule — walls under contention are not baselines — and the census classifies them excluded-by-rule with their filed reasons, never as flakes and never as determinism evidence
- `quarantine_needs_a_filed_reason_and_keeps_the_pin`: a quarantine entry names its baseline and its reason beside the census row's differing metrics; a quarantine without a reason is named by the checker, and quarantine never deletes a baseline, loosens a tolerance or stops the census from re-measuring the suite
- `flake_reports_are_validated_as_data`: FLAKE_REPORTS are judged by flake_report_problems(): a report is valid only with an id, a suite, a metric / leg, its differing values and a disposition, each missing part named; RERUN_RECORDS are judged by rerun_record_problems(): a record's report id must resolve to a filed report
- `probes_prove_the_checker_can_fail`: FLAKE_PROBES perturbs copies of filed rows / reports / records against the pure validators (a census-perturbed suite, a quarantine without a reason, a report missing its differing values, a report missing its disposition, a silent re-roll record) plus the clean shapes — real filings are never perturbed
- `pass_needs_the_full_census_classified`: the verdict is PASS only on the full census with every suite classified deterministic or reasoned-quarantined and every filed report / record valid; a full census with an unfiled nondeterministic suite files FAIL, a partial selection files REVIEW however it scores, and an empty selection files 0 baselines / REVIEW
- `the_flake_discipline_baseline_is_not_gated_by_tf129`: TF-129's regression gate gates the nine TF-121..TF-128 pinned baselines exactly as filed; this slice's own baseline (flake_discipline_synthetic_v1) is judged by this module's checker, not gated by the gate — the shape TF-129..TF-134 filed for their own baselines, with the exclusion filed in the open in regression_gate.py

## Filed limits

- The census judges the harness's pinned fixture baselines only — synthetic fixtures through the real scorers (TF-017). No live model is censused; live-model flakiness is NOT PROVEN (TF-173), never claimed.
- The census runs in one process on this VM (five repeats, in-process). Cross-machine noise is not measured here.
- A quarantined suite stays pinned and censused: quarantine files the reason and the differing metrics; it never deletes a baseline, loosens a tolerance or hides a re-roll.

Proof grade: FIXTURE — the harness's pinned synthetic-fixture baselines re-measured live on this VM; no live model, never claimed.
