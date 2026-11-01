# ThoughtForge Architecture — Ingestion Metrics (TF-115)

Ingestion metrics: records/sec, refusal rate, store
growth per source — reported per run. Part VI's
eleventh slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The metrics live as verifiable data in
`src/thoughtforge/etl/ingestion_metrics.py`
(`SOURCE_METRIC_SHAPES`, `METRICS_RULES`,
`store_snapshot()`, `measure_ingestion()`,
`summarise_runs()`, checked by `metrics_problems()`
and `metrics_live_problems()`); TF-105's living map
gained its `metrics` orchestration entry in the open.
The tests are `tests/test_tf115_ingestion_metrics.py`.
This slice **adds and fixes D-MET-1, D-MET-2 and
D-MET-3**.

## 1. Measured start

Probed live this tick, before any change:

- **No metrics existed anywhere (D-MET-1).** No
  module, CLI command or test named ingestion
  metrics. Every `forge_memory.py` command prints a
  builder's stats dict and discards the rest — the
  TF-108 gate ledger, TF-109 merge ledger and TF-114
  update ledger live in memory on the builder and are
  thrown away by every caller except TF-112's scratch
  dry-run, which reports gate + merge for one source
  but no timing, no refusal *rate*, no update ledger
  and no growth. TF-108 ("TF-115 owns reporting the
  rate per run"), TF-109 ("TF-115 owns reporting the
  difference per run") and TF-114 ("TF-115 owns
  reporting it per run") all pointed here.
- **No build records its own elapsed time
  (D-MET-2).** The key sets of a finished GeoNames
  build's stats, gate ledger and merge ledger were
  listed live: none carries a time, elapsed or
  seconds field. The only rate in the tree was the
  wikidata builder's checkpoint *log line* (entities
  per second, logged, never returned). A records/sec
  figure cannot be computed after the fact from a
  build's outputs — it must be measured around the
  build, or not claimed.
- **Accepted is not landed, and nothing named the
  difference (D-MET-3).** Probed live: an identical
  GeoNames re-run reported `{'inserted': 1,
  'skipped': 1}` a second time while the store stayed
  at 1 row — only the discarded merge ledger knew the
  record was a duplicate. Store growth (rows and
  bytes actually landed) is derivable only by
  measuring the store before and after a run, which
  no caller did (TF-112 measures a target's untouched
  state and a scratch's counts — never a real run's
  growth). A one-row store file also measured
  192,512 bytes (schema + indexes), so growth must
  be a *delta*, never an absolute size.

## 2. The per-run report

`measure_ingestion(source, input, db, clock=...)`
runs ONE source's real builder once into the named
store and returns the report (format
`thoughtforge-ingestion-metrics/1`):

- **Records/sec, measured.** The clock
  (`time.perf_counter` by default; a fake clock in
  tests and the live checker) is read around the
  build call only — the after snapshot sits outside
  the timed span. The rate is records *presented*
  (accepted + skipped, in TF-105's filed stats
  shapes) over that elapsed. A zero or unmeasured
  elapsed files the rate as `None` — never a division
  dressed as a fact.
- **Refusal rate, the gate's own fraction.** TF-108
  `refused / seen` for the run, with the by-reason
  and by-source breakdowns carried whole. Stated
  filter skips are not refusals: `other_skips`
  (stats skipped − gate refused) files them apart.
  The ungated reference builder files `gate_present:
  False` and a `None` rate with the TF-108 reason —
  never a fabricated zero.
- **Store growth per source, a delta of snapshots.**
  `store_snapshot()` measures a store without
  creating it (a missing store files `exists: False`,
  zero bytes, `counts: None` — a plain sqlite connect
  would conjure an empty file). Growth is after −
  before: per-table row deltas over TF-105's counted
  tables, total rows, and the byte delta (db + WAL +
  SHM, named apart in each snapshot). Growth may be
  negative — a TF-114 update run retires rows — and
  is filed as measured, never clamped.
- **Accepted vs landed, filed.** The report carries
  the stats verbatim, the output rows they claim,
  the primary table's growth (`entities` /
  `conceptnet_edges` / `geonames_places` /
  `dbpedia_entities` / `reference_chunks`), and
  `accepted_vs_primary_growth` — the primary count
  (entities / inserted / chunks per source) minus
  the primary table's growth. Duplicates, merged
  conflicts and retirements live in that difference;
  the merge and update ledgers, carried whole as
  data, say which. A builder with no update ledger
  files `None` with the TF-114 note that its updates
  ride its merge ledger.
- **Per source, and only aggregated honestly.**
  `summarise_runs()` aggregates per-run reports per
  source (the multi-source view a `forge_memory all`
  run needs): runs, presented, gate totals with the
  refusal rate recomputed from the sums, summed
  elapsed with the rate recomputed from the sums —
  never an average of rates — and summed growth.
  Runs with an unmeasured elapsed contribute their
  records but no time, counted in `unmeasured_runs`.

## 3. Filed limits

- **A failed build files no report.** The builder's
  exception propagates unchanged; partial measurements
  are discarded. For wikidata, TF-107's checkpoint is
  the failure's resume truth.
- **Wikidata resumed stats are cumulative.** On a
  resumed or completed-checkpoint wikidata run the
  stats are TF-107's cumulative filed stats for the
  whole dump ask, so presented and the rate describe
  the ask, not only this call's streaming. The report
  files stats verbatim and never adjusts them; the
  rule is filed in `METRICS_RULES`, not buried here.
- **No report file, no schema change, no CLI
  command.** The report is data returned to the
  caller; measuring writes nothing of its own, adds
  no schema table (TF-105's facts stand), and
  `forge_memory.py`'s stats-only printing stands as
  filed — TF-119 owns the recipe / wiring share.
- **Rates at scale are NOT PROVEN.** Every rate here
  is proven on synthetic fixtures under a fake clock
  (deterministic) or a real clock on tiny inputs (the
  figure is whatever the VM did that run). No real
  dump has been measured, and no at-scale
  records/sec is claimed.

## 4. Docket

- **D-MET-1 — added + FIXED.** No per-run metrics
  report existed (§1): `etl/ingestion_metrics.py`
  now reports records/sec, the refusal rate and
  store growth per source per run, with the gate,
  merge and update ledgers carried whole.
- **D-MET-2 — added + FIXED.** No build recorded its
  elapsed (§1): the metrics wrapper measures the
  build with a caller-supplied clock and files the
  rate only from a positive measured elapsed.
- **D-MET-3 — added + FIXED.** Accepted counted
  records the store never landed, unnamed (§1):
  growth is snapshotted before/after and
  `accepted_vs_primary_growth` files the difference
  per run (the probed identical re-run now reports
  accepted 2 / growth 1 on its first measured run's
  fixture shape and zero growth on a re-run).

TF-016 totals after this slice: 115 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 69 FIXED /
3 CLOSED-BY-VERDICT.

**Proof grade: FIXTURE** — synthetic sources through
the real builders on this VM, rates deterministic
under a fake clock. A real dump's ingestion rate,
refusal rate and growth are NOT PROVEN, never
claimed. TF-116 owns poisoned-source fixtures;
TF-117 owns offline ingestion; TF-118 owns the
reference library's ingestion side; TF-119 owns the
ETL docs + new-source recipe (including any CLI
wiring of these reports).
