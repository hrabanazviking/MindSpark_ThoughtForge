# TF-010 — The Benchmark Estate (Part I Audit)

Measured 2026-10-30. Everything under `benchmarks/` inventoried,
classified (library / script / suite), and actually run today on
this VM under the TF-004 method (`PYTHONPATH=src`, package not
pip-installed). Benchmark-adjacent instruments outside
`benchmarks/` (locustfile, the in-core perf tracker) are censused
with it because TASK_PHASE6_RELEASE counts them as the same
deliverable family.

## 1. The estate, whole

| File | Bytes | Kind | Runs today? |
|---|---|---|---|
| `benchmarks/__init__.py` | 244 | Doc-only package marker | n/a |
| `benchmarks/benchmark_profiles.py` | 12,427 | Library + CLI script (`ProfileBenchmark`) | YES, exit 0 — against an empty ephemeral store only (see §3) |
| `benchmarks/persona_consistency.py` | 11,821 | Scorer library + stdin CLI | YES, exit 0 (see §2) |
| `locustfile.py` (repo root) | 4,910 | Load-test script (Phase-6 deliverable) | NO — `locust` is a `[dev]` extra, not installed on this VM (see §4) |

That is the entire estate: **2 real modules, 0 suites.** No
fixture corpus, no seeded benchmark store, no results directory,
no persisted output, no pinned baselines anywhere in the tree.
Both modules arrived in a single commit (8aa8c06, 2026-03-31,
"feat(phase6)") and have not been touched since — roughly seven
months stale against a tree that kept moving.

Test coverage lives in `tests/test_phase6_release.py`: of its 54
tests, 37 touch the benchmark family (TestBenchmarkResultDataclass
10, TestConsistencyResultDataclass 6, TestPersonaConsistencyScorer
9, TestProfileBenchmarkInterface 9, TestLocustfile 3). All 54 pass
under the TF-004 method (verified this slice). The run-tests among
them exercise `ProfileBenchmark.run()` on 2-query sets against the
same ephemeral core the CLI uses.

## 2. `persona_consistency.py` — a real scorer, honestly runnable

Pure-text phrase scoring (21 generic AI phrases penalised, 42
Norse tone markers + 9 Skald quality markers rewarded, QID-pattern
citation bonus). No model, no store, no network.

Verified today: a 3-line synthetic sample piped to
`python3 -m benchmarks.persona_consistency` scored **0.883 PASS**,
exit 0, correctly flagging the one generic-phrase line
("as an ai", "i cannot provide"). This is the healthiest artefact
in the estate — and still carries three defects:

- **D-BENCH-3 (dead parameter + lying formula):** `score()`'s
  `weights` argument is built into a dict `w`, updated, and never
  read again — callers who pass weights are silently ignored. The
  module docstring's formula `((1.0 − generic_penalty) + norse +
  citation) / 3.0` is not the implementation, which averages
  per-turn scores of `clamp(1.0 − penalty + norse_bonus +
  citation_bonus + skald_bonus)` (the skald bonus is undocumented
  in the docstring).
- **D-BENCH-4 ("100+ turns" is a claim, not a fact):**
  `benchmarks/__init__.py`, TASK_PHASE6_RELEASE.md, and
  MODEL_CARD.md all describe validation "across 100+ turns". No
  100-turn fixture, dataset, or recorded run exists anywhere in
  the tree. The scorer accepts any list — and an **empty list
  scores a vacuous 1.0 PASS** ("vacuously consistent"), so the
  headline target (≥ 0.75) can be "met" by measuring nothing.
- The CLI reads stdin lines only; the module's own
  `generate_report()` markdown writer is unreachable from the CLI.

## 3. `benchmark_profiles.py` — runs, but measures almost nothing

Verified today: `python3 -m benchmarks.benchmark_profiles
desktop_cpu` completed in ~2.5 s, exit 0:

```
Turns 10/10 successful · Citation acc 0.0% (target ≥85%) ✗ ·
Enforcement 0.0% (target ≥90%) ✗ · Avg words 9.0 ·
Avg latency 7ms (p95: 5ms) · Avg confidence 0.408 · Token eff. 0.0%
```

Every "successful" turn is the honest empty-store refusal ("The
forge has no knowledge relevant to this query." — 9 words), because
the benchmark builds an **ephemeral core over an empty store** and
no seeded corpus ships with it. Out of the box, its headline
targets (citation ≥ 85%, enforcement ≥ 90%, from BUILD_PLAN_v1 via
MODEL_CARD) are unmeasurable, not merely unmet. Defects:

- **D-BENCH-1 (the profile is a label, nothing more):**
  `profile_id` is never validated and never applied. Verified:
  `totally_bogus_profile` runs to exit 0 with identical-shaped
  output. It cannot be applied — `ThoughtForgeCore.__init__` has
  **no profile parameter at all** — and the ephemeral core is
  identical for every profile, with `model_path=None`, so the
  latency figures are retrieval+scaffold overhead over an empty
  store, never inference, never per-profile. The six real profiles
  in `hardware_profiles/*.json` are never consulted.
- **D-BENCH-2 (metric name drift):** the `token_efficiency` field
  is documented as "useful words / total words"; the code computes
  the fraction of turns meeting each query's `min_response_words`.
  `citation_accuracy` is commented "when QIDs retrieved" but its
  denominator is all successful turns and any citation string
  counts. MODEL_CARD's separate "Avg response words ≥ 30" target
  is checked nowhere (per-query minimums are 10–20).
- `BenchmarkResult.quality_tiers` is declared and never
  populated — always `{}`.
- Warm-up turns are drawn from the measured set itself
  (query[0] is warmed up, then measured again).
- The p95 index (`int(n·0.95) − 1`) is a 90th-percentile pick at
  n=10; today's run printed p95 (5ms) below the mean (7ms).
- Nothing is persisted: `summary()` goes to stdout and is gone.
  There is no baseline to regress against — Part VII's harness
  (TF-121) starts from zero, not from this module's output.

## 4. The adjacent instruments

- **`locustfile.py`** — a no-HTTP Locust user that shares one
  ephemeral core per worker. Cannot run on this VM today: `locust`
  sits in pyproject's `[dev]` extras and is not installed; the
  file guards the import (`_LOCUST_AVAILABLE`) and its 3 tests are
  structural only. TASK_PHASE6_RELEASE.md's description of it
  ("ThoughtForgeUser(HttpUser)… via HTTP interface") is stale —
  the file's own header says the opposite (no-HTTP mode).
  → D-BENCH-5 for TF-016.
- **`src/thoughtforge/utils/perf.py`** — a ring-buffer latency
  tracker (p50/p95/p99, bottleneck report) wired into
  `ThoughtForgeCore` as `self._perf`, covered by 17 tests in
  `tests/test_phase8_perf.py`. It is instrumentation, not a
  benchmark — and no benchmark consumes it: `ProfileBenchmark`
  times turns with its own `perf_counter` calls. Two latency
  systems, unconnected.

## 5. Verdict for the road

- Scripts vs suites: both modules are **library+script hybrids**;
  neither is a suite (no fixtures, no baselines, no persistence,
  no pass/fail gate). TF-010 → TF-121 (harness architecture) and
  TF-122 (persona suite) inherit exactly this gap, as written.
- Stale claims to strike or prove in Part VII / TF-136: the
  "100+ turns" persona claim, MODEL_CARD's target table (targets
  real, measurements absent), and the per-profile framing of
  `benchmark_profiles`.
- Docket for TF-016: D-BENCH-1 (profile label never applied or
  validated), D-BENCH-2 (metric name drift), D-BENCH-3 (dead
  `weights` + docstring formula mismatch), D-BENCH-4 (vacuous
  empty-list PASS + unproven 100+ turns claim), D-BENCH-5
  (locustfile unrunnable as shipped on a dev-extra + stale
  TASK_PHASE6 description).
- One method note for every later keeper: without
  `PYTHONPATH=src` the benchmark's run-tests fail with
  `ModuleNotFoundError: thoughtforge` (verified — 3 failures
  standalone, 54/54 pass under the TF-004 method). The usage docs
  (`docs/api.md`, module docstrings) never state this requirement.

Suite at banking: 618 passed, 1 skipped, 1 known env failure
(D-ENV-1) — baseline held, no new failures.
