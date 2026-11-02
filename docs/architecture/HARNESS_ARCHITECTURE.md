# ThoughtForge Architecture — Harness Architecture: Benchmarks as Code (TF-121)

Harness architecture: benchmarks as code with fixtures,
scorers, and pinned baselines — the `benchmarks/` estate
TF-010 censused, unified under one index with its first
pinned baselines. Part VII's first slice. Built
2026-11-01 on branch `yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`benchmarks/harness.py` (`BENCHMARKS`, `FIXTURES`,
`PERSONA_FIXTURE_RESPONSES`, `METRIC_DEFINITIONS`,
`BASELINES`, `HARNESS_RULES`, checked by
`harness_problems()` and `baseline_problems()`); two
standing defects were fixed in the estate's own modules
in the open (`benchmarks/benchmark_profiles.py`,
`benchmarks/persona_consistency.py`). The tests are
`tests/test_tf121_harness.py`. This slice **fixes
D-BENCH-2 and D-BENCH-6**, and **narrows D-BENCH-1,
D-BENCH-5 and D-BENCH-7** (see §5).

Where the harness lives, and why: in the estate's own
package, `benchmarks/`. The estate is a top-level
package imported as `benchmarks.*` by its tests; the
shipping library under `src/thoughtforge/` imports
nothing from it, and a harness filed under `src/` would
reach back across the root package to bind the very
modules it unifies. The neighbour scorers from earlier
parts (TF-043 / TF-044 / TF-067 / TF-102 / TF-036) are
bound by module + attribute and judged live — bound,
not copied.

## 1. Measured start

Probed live this tick, before any change:

- **TF-010's census stood exactly as filed**: the whole
  estate is 2 modules, 0 suites — no fixture corpus, no
  seeded store, no results directory, no persisted
  output, and **no pinned baseline anywhere in the
  tree** to regress against.
- **D-BENCH-1 stood open**:
  `python3 -m benchmarks.benchmark_profiles
  totally_bogus_profile` ran to exit 0 with
  identical-shaped output. The profile is never
  validated and never applied.
- **D-BENCH-2 stood open**: `token_efficiency` is
  documented as "useful words / total words" but
  computes the fraction of turns meeting
  `min_response_words`; `citation_accuracy`'s comment
  says "when QIDs retrieved" but its denominator is
  all successful turns; MODEL_CARD's "avg response
  words ≥ 30" target was checked nowhere.
- **D-BENCH-6 stood open, re-probed live**: a fresh
  `ProfileBenchmark` run filed `quality_tiers == {}`
  (declared, never populated); warm-up turns were
  drawn from the measured set; the p95 index picked
  ~p90 at n=10 — this tick's CLI run printed avg 8ms
  with "p95: 2ms", below the mean; nothing persisted;
  the persona CLI could not reach `generate_report()`.
- **D-BENCH-5 / D-BENCH-7 stood open**: `locust` is
  not installed on this VM (a dev extra; import probed
  and failed), and the core's `utils/perf.py` tracker
  is consumed by no benchmark.
- The two measured baselines (twice each, identical):
  the persona scorer over this slice's synthetic
  fixture scored **0.7167** with 2 flagged turns; the
  profile benchmark over its ephemeral empty-store
  core filed citation 0.0, enforcement 0.0, avg words
  9.0, min-words met 0.0, avg confidence **0.408** —
  while its latency varied between probes (avg 7ms /
  8ms).

## 2. The unified index

`BENCHMARKS` in `benchmarks/harness.py` files nine
entries — every benchmark, neighbour scorer and
instrument in the estate — each bound to the real
thing it lives in (import + getattr for package
modules; file + markers for the locust script, which
is not importable without its dev extra):

| Entry | Kind | Bound to | Axes | Suite slice |
|---|---|---|---|---|
| `persona_consistency` | scorer | `benchmarks.persona_consistency.PersonaConsistencyScorer` | presence | TF-122 completes it |
| `profile_benchmark` | runner | `benchmarks.benchmark_profiles.ProfileBenchmark` | depth | TF-128 applies the profile |
| `locust_load_test` | load instrument | `locustfile.py` (file + markers) | — | none — excluded (§5) |
| `retrieval_precision` | neighbour scorer | `thoughtforge.knowledge.retrieval_precision.evaluate_retrieval` (TF-043) | depth | TF-123 consumes |
| `scoring_honesty` | neighbour scorer | `thoughtforge.knowledge.scoring_honesty.scoring_problems` (TF-044) | depth, presence | TF-122..125 build on its census |
| `quant_quality` | neighbour scorer | `thoughtforge.inference.quant_quality.quant_quality_problems` (TF-067) | depth | TF-127 consumes |
| `quality_marks` | neighbour scorer | `thoughtforge.refinement.quality_marks.quality_marks_problems` (TF-102) | depth, presence | TF-122..125 file marks through it |
| `determinism` | neighbour scorer | `thoughtforge.cognition.determinism.reproducibility_problems` (TF-036) | all three | TF-135 consumes |
| `perf_tracker` | instrument | `thoughtforge.utils.perf.PerfTracker` | — | TF-133 owns consumption |

`KNOWN_PROFILES` files the six shipped profile ids
(`desktop_cpu`, `desktop_gpu`, `phone_low`, `pi_5`,
`pi_zero`, `server_gpu`), judged live against
`hardware_profiles/*.json` and against the benchmark
module's own `known_profile_ids()` reader.

## 3. Fixtures and metric definitions

`FIXTURES` files the estate's first fixture corpus:

- `persona_fixture_responses` — six synthetic response
  lines written for this slice (`PERSONA_FIXTURE_RESPONSES`:
  three in-voice, one plain factual, two deliberately
  generic AI-flavoured lines, so the baseline exercises
  the scorer's penalty leg as well as its bonuses).
- `profile_default_queries` — the profile benchmark's
  own 10 standard queries, bound in place
  (`_DEFAULT_QUERIES`).

All fixtures are synthetic (TF-017 / TF-131): canon is
never a benchmark input, and `data/` was never opened.

`METRIC_DEFINITIONS` files every metric a baseline
carries with what it **actually computes** — including
the canonical `min_words_met_rate` beside its legacy
field name `token_efficiency`, and `citation_accuracy`
stated as a citation-carrying rate over successful
turns, not a retrieved-QID accuracy.

## 4. The pinned baselines

`BASELINES` files the estate's first pinned baselines,
measured live on this VM at banking by running the
filed fixtures through the real scorers.
`baseline_problems()` re-measures both live and names
any metric that moved beyond tolerance (1e-9 — both
scorers are deterministic on these fixtures, TF-036);
a moved baseline is a finding, never silently
re-pinned.

**`persona_consistency_synthetic_v1`** — the filed
fixture through the real `PersonaConsistencyScorer`:

| Metric | Pinned |
|---|---|
| consistency_score | 0.7167 |
| total / flagged turns | 6 / 2 |
| generic phrase hits | 7 |
| Norse tone hits | 10 |
| citation turns | 1 |
| Skald quality hits | 5 |
| target: score ≥ 0.75 | **unmet** |

The baseline sits **below** the pass threshold,
honestly: the fixture's two generic turns exercise the
penalty leg. TF-122 completing the scorer (D-BENCH-3,
the dead `weights`) will move this baseline, and the
checker will name the move.

**`profile_benchmark_empty_store_v1`** —
`ProfileBenchmark` over its ephemeral core (empty
store, `model_path=None`), profile `desktop_cpu`:

| Metric | Pinned |
|---|---|
| turns (successful / failed) | 10 (10 / 0) |
| citation_accuracy | 0.0 |
| enforcement_pass_rate | 0.0 |
| avg_response_words | 9.0 |
| min_words_met_rate (`token_efficiency`) | 0.0 |
| avg_confidence | 0.408 |
| quality_tiers | `{"weak": 10}` |
| targets: citation ≥ 0.85, enforcement ≥ 0.90, avg words ≥ 30 | **all unmet** |

Every target is unmet, honestly: the store is empty,
so every turn is the 9-word empty-store refusal. This
baseline pins the empty-store behaviour a seeded run
must beat; it is not a quality claim.

**Latency is pinned nowhere.** It varied between this
tick's own probes and varies under contention — walls
are colour, never baselines (filed per baseline in
`excluded_metrics`). TF-133 owns cost measurement.

## 5. Docket dispositions (TF-016)

- **D-BENCH-2 — FIXED.** In
  `benchmark_profiles.py`: the module docstring and
  the field comments now state what each metric
  computes (`token_efficiency` is the min-words met
  rate; `citation_accuracy`'s denominator is all
  successful turns); MODEL_CARD's avg-words target is
  filed as `BenchmarkResult.AVG_WORDS_TARGET = 30.0`,
  judged by `passes_avg_words_target()` and printed
  in the summary beside the other two targets. The
  field name `token_efficiency` is kept (the phase-6
  tests and the dataclass API pin it) and filed under
  its canonical name in `METRIC_DEFINITIONS`.
- **D-BENCH-6 — FIXED, all five bundle items.**
  `quality_tiers` is populated from each successful
  turn's score tier (the baseline files
  `{"weak": 10}`); warm-up runs the dedicated
  `WARM_UP_QUERY`, never a measured query; p95 is
  nearest-rank (`ceil(0.95·n) − 1`); the pinned
  baselines in `benchmarks/harness.py` are the
  persistence the estate lacked (TF-134 owns the
  baseline book's regenerated form); and the persona
  CLI reaches `generate_report()` via `--report`
  (proven live: the markdown report prints).
- **D-BENCH-1 — narrowed.** `ProfileBenchmark.run`
  validates `profile_id` against the shipped profiles
  and raises `ValueError` naming the bogus id and the
  known set; the CLI exits 2 with that message (proven
  live with `totally_bogus_profile`). The profile is
  still not **applied** — `ThoughtForgeCore` has no
  profile parameter — and that share stays TF-128's;
  the entry stays OPEN on it.
- **D-BENCH-5 — narrowed (fate decided).** The
  harness files the locust script's fate:
  **excluded from the panel** — a manual throughput
  script, unrunnable without the dev extra, no-HTTP
  by its own header, measuring no claim axis. It stays
  a script a human runs on purpose. TF-136 owns the
  stale TASK_PHASE6 description; the entry stays OPEN
  on that share.
- **D-BENCH-7 — narrowed.** Both latency systems are
  filed in one index with roles stated: the profile
  benchmark's per-turn `perf_counter` is the harness's
  turn timing; `PerfTracker` is core instrumentation,
  filed as an instrument, not a scorer. Consuming the
  tracker is TF-133's share; the entry stays OPEN on
  it.
- **D-BENCH-3 / D-BENCH-4 — untouched, TF-122's.**
  The persona baseline pins the scorer exactly as it
  behaves today, dead `weights` and vacuous-empty
  shape included; the suite slice owns those fixes.
- No new docket entry: every gap probed this tick was
  already numbered by TF-010. TF-016 totals are
  therefore 123 entries — **36 OPEN, 1
  OPEN-ENVIRONMENT, 83 FIXED, 3 CLOSED-BY-VERDICT**
  (D-BENCH-2 and D-BENCH-6 move OPEN → FIXED).

## 6. Filed limits

- The profile baseline measures an ephemeral core over
  an **empty store** with `model_path=None` — the
  empty-store refusal, never inference, never a loaded
  model. A seeded store and a loaded model are NOT
  PROVEN here.
- **The will axis has no scorer anywhere in the
  estate yet**: the index covers depth and presence
  only. TF-125 owns the will suite; the gap is filed
  as the rule `the_will_axis_has_no_scorer_yet`, not
  dressed as coverage.
- No Part VII suite exists yet: TF-122..TF-125 build
  the suites on this architecture, TF-129 owns
  regression gates, TF-134 the baseline book, TF-137
  the one-command tool. This slice files the
  architecture, the bindings and the first baselines
  — nothing more is claimed.
- The persona baseline's exactness rests on the
  scorer's determinism (pure text scoring); any
  scorer change moves it and is named by
  `baseline_problems()`, by design.

Proof grade: FIXTURE — synthetic fixtures through the
real scorers and the real profile runner on this VM.
No loaded model and no seeded store were measured;
that is NOT PROVEN, never claimed.
