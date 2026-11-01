# Benchmark Data Hygiene — TF-131

The proof lives as verifiable data in
`benchmarks/data_hygiene.py` (`HYGIENE_ROWS`,
`HYGIENE_PROBES`, `HYGIENE_RULES`, checked by
`data_hygiene_problems()`), registered in
`benchmarks/harness.py` with the pinned
baseline `data_hygiene_synthetic_v1`. The tests
are `tests/test_tf131_data_hygiene.py`. This
slice **adds and fixes D-BENCH-16** in the
measuring sense (see §5).

## 1. Measured start

Probed live this tick, before any change:

- **The estate stood clean, and unfiled.**
  `harness_problems()` returned [] and
  `baseline_problems()` returned [] over the
  eleven pinned baselines;
  `regression_gate_problems()` returned [] with
  the live gate PASS (9 baselines / 83 metrics /
  0 failures); `spot_checks_problems()` returned
  []. And a grep for any hygiene / provenance /
  licence registry across the benchmark estate
  found nothing: TF-017's guards scan `tests/`
  only, and no filing anywhere said, per
  benchmark input, where it came from or proved
  it was not canon.
- **The canon scan was clean but unguarded.** A
  grep of `benchmarks/*.py` for the canon-access
  shapes TF-017 forbids found no hits — true
  that day, proven by nothing that runs.
- **D-BENCH-16 was verified unused** anywhere in
  benchmarks / docs / tests / src before filing
  (the D-BENCH-9..15 pattern: a gap assigned by
  the roadmap without ever being numbered).

## 2. The inventory

`HYGIENE_ROWS` files every benchmark input in
the estate as data — where it lives, its origin
class, and its proof. No corpus is copied: rows
bind the real modules / symbols / files and the
checker inspects them live.

| Row | Where it lives | Origin | Proof |
|---|---|---|---|
| persona_fixture_responses | benchmarks.harness `PERSONA_FIXTURE_RESPONSES` | synthetic-invented | harness fixture, filed synthetic |
| profile_default_queries | benchmarks.benchmark_profiles `_DEFAULT_QUERIES` (+ `WARM_UP_QUERY`) | synthetic-invented | harness fixture, filed synthetic |
| persona_suite_corpus | benchmarks.persona_suite `ALL_SUITE_TURNS` | synthetic-invented | harness fixture, filed synthetic |
| memory_suite_corpus | benchmarks.memory_suite `ALL_SUITE_CASES` | synthetic-invented | harness fixture, filed synthetic |
| depth_suite_corpus | benchmarks.depth_suite `ALL_SUITE_CASES` | synthetic-invented | harness fixture, filed synthetic |
| presence_suite_corpus | benchmarks.presence_suite `ALL_SUITE_CASES` | synthetic-invented | harness fixture, filed synthetic |
| honesty_panel_corpus | benchmarks.honesty_panel `MODEL_TIERS` | derived-live | derives from the four suite rows, clean in the same run |
| quant_panel_corpus | benchmarks.quant_panel `PANEL_ROWS` | derived-live | derives from tf067_quant_fixture_corpus |
| profile_panel_corpus | benchmarks.profile_panel `PANEL_ROWS` | derived-live | derives from the query + suite + hardware-profile rows |
| regression_gate_corpus | benchmarks.regression_gate `NOISE_ROWS` | derived-live | derives from the nine gated baselines' rows |
| spot_checks_corpus | benchmarks.spot_checks `TRANSCRIPT_SPECS` | derived-live | derives from the suite + TF-067 rows |
| wyrd_panel_corpus | benchmarks.wyrd_panel `PANEL_CASES` (+ `WORLD_ENTITIES` / `WORLD_EVENTS` / `PANEL_PROBES`) | synthetic-invented | harness fixture, filed synthetic — TF-132's invented Emberwick world (added in the open by TF-132; see §5) |
| data_hygiene_corpus | benchmarks.data_hygiene `HYGIENE_ROWS` | derived-live | this registry itself, derived from every other row |
| tf043_retrieval_fixture_corpus | thoughtforge.knowledge.retrieval_precision `FIXTURE_ENTITIES` (+ chunks / edges / place rows / queries) | synthetic-invented | the module's own filing: a synthetic corpus in TF-017's invented Lantern Hollow world (the ConceptNet / GeoNames shapes are record shapes; every row is invented, no external dataset is read) |
| tf030_persona_fixture | thoughtforge.cognition.persona `FIXTURE_EXPECTATION` (+ `PERSONA_FIXTURE_TURNS`) | synthetic-invented | the module's own synthetic filing |
| tf067_quant_fixture_corpus | thoughtforge.inference.quant_quality `QUANT_PROMPTS` (+ `FIXTURE_RESPONSES`) | synthetic-invented | the module's own synthetic filing |
| hardware_profile_files | `hardware_profiles/*.json` (6 files) | synthetic-invented | the shipped files themselves; ids checked live against KNOWN_PROFILES |
| tests_fixtures_synthetic_persona_core | `tests/fixtures/synthetic_personality_core.yaml` | synthetic-invented | TF-017 marker in the file |
| tests_fixtures_synthetic_reference | `tests/fixtures/synthetic_reference/*` (3 files) | synthetic-invented | TF-017 marker or structured synthetic flag per file |
| tests_fixtures_synthetic_module | `tests/fixtures/synthetic.py` | synthetic-invented | TF-017 marker constant filed in the module |
| tests_fixtures_chaos_corpus | `tests/fixtures/chaos_corpus.json` | synthetic-invented | the corpus's own SYNTHETIC_FIXTURE world line |

The `tests/fixtures/` rows are filed for
completeness of the census: no benchmark module
imports that estate — the suites carry their own
corpora in TF-017's same invented world — and
the chaos corpus is consumed by the chaos tests,
not by any benchmark. That is filed on the rows
themselves, not smoothed over.

**The classification, measured: 21 inputs — 15
synthetic-invented, 0 licensed, 6 derived-live.**
(TF-132 update, in the open: the inventory
stood 20 inputs — 14 synthetic — at TF-131's
banking; TF-132 filed one new synthetic input,
the WYRD panel's corpus, and the counts moved
for exactly that reason.)
No benchmark input in the estate is licensed or
public-domain material, so the licensed class
files no rows; its rule (a licensed row must
name its licence or source) is still enforced,
by probe (§4).

## 3. The checker and the verdict

`run_data_hygiene()` evaluates the inventory
live: every module binding resolves (with its
sibling symbols), every files row's paths exist
and carry their proof (TF-017 markers where the
policy files them; the hardware profiles' ids
equal KNOWN_PROFILES), every bound module's
source is scanned with TF-017's guard shapes,
and derived rows inherit the failures of the
rows they derive from in the same run.

The canon scan reuses TF-017's guard shapes —
the repo-anchored canon path, the canon path
helpers, the reference-library and picture
markers, and the canon persona names. TF-017's
lists live in a test module the benchmark
estate does not import, so the shapes are
re-filed in `benchmarks/data_hygiene.py`
**openly, as a duplication**, with the pattern
strings assembled from parts so the hygiene
module's own source — which the scan also
covers — stays clean under its own scan. The
scan also covers every `benchmarks/*.py` file
as a whole, this slice's included.

The verdict: PASS only on the FULL filed
inventory with every input classified and
clean; a partial selection files REVIEW however
it scores; an empty selection files 0 inputs /
REVIEW — no vacuous pass. Canon contamination,
an unregistered fixture, or a classification
that cannot be proven FAILs by name.

## 4. The probes

`HYGIENE_PROBES` files seven perturbed filings
against the pure checker parts (`row_problems`,
`coverage_problems`, `scan_source_text`): a
canon-origin row fails; a licensed row naming
no licence fails; a synthetic row with no proof
fails; an unresolvable binding fails; a source
carrying a canon-access shape fails; an
unregistered harness fixture fails coverage by
name; and one clean synthetic row passes. Real
filings are never perturbed.

## 5. D-BENCH-16

The benchmark estate carried no per-fixture
provenance filing at all: TF-017's guards scan
`tests/` only, so a benchmark corpus could have
drifted to real material and nothing running
would have named it. TF-131 files the inventory
above and fixes the gap in the measuring sense:
every benchmark input now carries a hygiene row
judged live by `data_hygiene_problems()`,
registered in the harness in the open with the
pinned baseline `data_hygiene_synthetic_v1`
(21 / 15 / 0 / 6 / 21 clean, probes 7/7; target
is the hygiene truth on clean_inputs at 21,
met — re-pinned in the open by TF-132 from
20 / 14 / 0 / 6 / 20, target 20, for exactly
one reason: TF-132's new synthetic row grew
the inventory by one input). The baseline is NOT gated by TF-129's
regression gate — the exclusion is filed in the
open in `benchmarks/regression_gate.py` as
`UNGATED_BASELINE_IDS`, the shape TF-129 /
TF-130 filed for their own baselines; the gated
set stays the nine TF-121..TF-128 baselines and
the gate still PASSes (9 / 83 / 0).

## 6. Filed limits

- **Proof is marker / path / provenance-based**
  over the estate's own filings: bindings
  resolve live, TF-017 markers are checked
  where the policy files them, module filings
  are checked for their synthetic claims, and
  benchmark code is scanned for canon-access
  shapes. Canon content itself is never
  opened, copied, or quoted (TF-017 / TF-007).
  A synthetic corpus whose text quietly
  paraphrased canon without using any forbidden
  shape would not be caught by this scan, and
  no such claim is made.
- **The scan covers benchmark code**
  (`benchmarks/*.py` and the bound neighbour
  modules in `src/`), not fixture data files
  and not the whole `src/` tree — TF-017's
  guards cover `tests/`; other `src/` modules
  are other slices' filings.
- **The licensed class files 0 rows** on this
  tree: it is enforced by rule + probe, not
  exercised by a real row.
- **Live-model benchmark data is NOT PROVEN**
  (TF-173): every input filed here is a fixture
  for the harness's scorers, never a live
  model's output or a real conversation.

Proof grade: FIXTURE — the estate's own filed
fixtures inspected live on this VM; no canon
opened, no live model, never claimed.
