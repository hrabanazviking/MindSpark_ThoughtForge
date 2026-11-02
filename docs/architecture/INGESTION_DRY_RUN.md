# ThoughtForge Architecture — Ingestion Dry-Run (TF-112)

Ingestion dry-run: a source can be test-ingested into a
scratch store with a full report before touching a real
store. Part VI's eighth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The driver lives as verifiable data in
`src/thoughtforge/etl/ingestion_dry_run.py`
(`BUILDER_SOURCES`, `DRY_RUN_RULES`,
`dry_run_ingestion()`, `report_as_data()`, checked by
`dry_run_problems()` and `dry_run_live_problems()`);
TF-105's living map was updated in the open (the
orchestration entry now names this single-source scratch
runner — it is not the `all` pipeline, whose filed shape
stands). The tests are
`tests/test_tf112_ingestion_dry_run.py`. This slice
**fixes D-COV-2** and **discharges D-ETL-2's TF-112
share**.

## 1. Measured start

Probed live before this slice changed anything:

- **No dry-run existed anywhere.** No module, no
  `forge_memory.py` command and no test named one —
  the only hits in the tree were the filed promises
  (TF-106/107/108/109/111) that TF-112 would be a
  consumer. To learn what a source file would do, the
  only road was building it straight into a real store.
- **The registry was consulted nowhere at build
  time** (D-ETL-2): neither builder module nor the
  root instrument mentions the registry or its gate.
- **A build's full story was thrown away.** The CLI
  prints only a builder's stats; the TF-108 gate
  ledger and TF-109 merge ledger live in memory on
  the builder and are discarded by every command.
  Live probe (GeoNames, one valid row + one row with
  latitude 999.0): the build reported
  `{'inserted': 1, 'skipped': 1}` — the refusal's
  reason existed only in the discarded ledger.

## 2. The road

`dry_run_ingestion(source, input_path, ...)` — one
call, in this order:

1. **Registry first.** The source's primary TF-106
   entry is judged by `check_source_use(...,
   ingest_private)` before any scratch file exists
   and before any builder runs. An unregistered
   source raises by name with nothing created.
2. **Input checked.** A missing file/directory
   raises `FileNotFoundError` naming it, still
   before any scratch is created.
3. **Scratch, never the real store.** The real
   builder is constructed with a scratch database
   (`dry-run-scratch.db` in a fresh temporary
   directory, or in the caller's `scratch_dir`) and
   nothing else. A scratch path resolving to the
   named target raises `DryRunTargetError`.
4. **The real builder runs** on the scratch with the
   caller's builder arguments (the wikidata leg
   defaults to `resume=False` — a scratch is a fresh
   ask).
5. **The full report is filed** (§3), the named
   target is re-measured, and an unkept scratch is
   removed — database, WAL/SHM sidecars and builder
   checkpoint sidecars — including after a failure,
   which re-raises unchanged.

## 3. The full report

Format `thoughtforge-ingestion-dry-run/1`:

| Field | What it carries |
|---|---|
| `registry` | The TF-106 entry as consulted: provenance, licence, status, obligations, the use checked. A quarantined source (the reference library) passes private ingestion with `licence: None` and its quarantine stated — no licence is invented for it. |
| `stats` | The builder's stats, in TF-105's filed shapes, unchanged. |
| `store_counts` | `get_entity_count` over the scratch estate after the build (re-measured after the embedding leg when one runs). |
| `gate` | The builder's TF-108 `SchemaGate.as_data()`: refusals with reasons, broken down by source and reason code — this report is the ledger's first reporting consumer. The reference builder is not gated (TF-108's filed limit): its report files `gate: None` plus a `gate_note`, never a fabricated empty ledger. |
| `merge` | The builder's TF-109 `MergeLedger.as_data()`: collisions by outcome and by table — the ledger's first reporting consumer. |
| `overlaps` | TF-109's `find_overlaps` over the scratch estate, as data. |
| `embeddings` | `not_run` with its reason, or — when the caller injects an encoder — TF-111's driver result over the scratch (counts + estate digest). This is the injected encoder's filed first consumer; a dry run loads no model on its own word. |
| `scratch_*` | The scratch path, its measured byte size, whether it was kept, and whether this run appended to a kept scratch. |
| `target` / `target_untouched` | When a real store is named: its before/after existence, byte size, SHA-256 and row counts, and whether they agree exactly. When none is named, `target_untouched` is `None` and the report says the only database written was the scratch. |

## 4. Composition, filed plainly

A dry run judges one source per call. Sequential
calls into one kept scratch compose an estate only
when the caller passes `append=True` explicitly —
a non-append run always starts from no scratch
file. Composition is what lets a cross-source
overlap be named: the live checker dry-runs a
synthetic wikidata town and a synthetic GeoNames
town of the same name into one kept scratch, and
the second report's overlaps name the
`lantern town` group across both tables. Each
report still files only its own build's gate and
merge ledgers — they are per-build ledgers by
TF-108/TF-109's own filing.

## 5. Docket

- **D-COV-2 — FIXED by TF-112.** Its remaining share
  was this dry-run's depth: the live checker and the
  tests now exercise all five builders and both
  embedding legs (through TF-111's injected encoder)
  directly, in depth, on synthetic inputs — the ETL
  modules TF-002 found unimported are the dry-run's
  whole road.
- **D-ETL-2 — narrowed (TF-112 share FIXED).** The
  dry run consults the registry first, as filed.
  The entry stays OPEN for TF-119: the builders
  themselves still consult nothing at build time
  (TF-106's static checker keeps pinning exactly
  that shape), and the new-source recipe must file
  a source in the registry before documenting it.

TF-016 totals after this slice: 108 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 62 FIXED /
3 CLOSED-BY-VERDICT.

## 6. Filed limits

- One source per call; cross-source composition is
  the caller's explicit `append=True` sequence.
- Gate/merge ledgers are per-build even on an
  appended scratch.
- No report file is written anywhere by the module
  and no schema table is added — the report is the
  returned data.

## 7. What is NOT proven

Any real dump dry-run at scale, and any real
model's embedding leg (the leg here runs through
an injected fake encoder, as TF-111 filed).
Proof grade: FIXTURE — synthetic sources through
the real builders into scratch stores on this VM.
TF-115 owns per-run ingestion metrics beyond this
report; TF-120's Part VI gate owns the full
source → store → retrieval round trip.
