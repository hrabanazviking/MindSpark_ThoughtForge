# ThoughtForge Architecture — ETL Docs + a New-Source Recipe (TF-119)

ETL docs + a new-source recipe: the Part VI
documentation index, the ordered recipe a new source
follows, and two standing defects fixed in scope.
Part VI's fifteenth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`src/thoughtforge/etl/etl_docs.py` (`DOC_INDEX`,
`RECIPE_STEPS`, `RECIPE_RULES`,
`recipe_application()`, `example_recipe_application()`,
checked by `etl_docs_problems()`); `forge_memory.py`'s
`build_all` gained its DBpedia step, TF-105's map was
updated in the open, and `etl/sources.py`'s stale
reference count was corrected in the open. The tests
are `tests/test_tf119_etl_docs.py`. This slice
**fixes D-ETL-1, D-DATA-1 and D-ETL-2** (both of
D-ETL-2's shares now discharged; see §5).

## 1. Measured start

Probed live this tick, before any change:

- **D-ETL-1 stood open exactly as TF-105 filed it.**
  `forge_memory.py all` accepted `--dbpedia-json` and
  never ingested it: `build_all` echoed steps /6 —
  schema, reference, wikidata, conceptnet, geonames,
  embeddings — and `DBpediaETL` appeared nowhere in
  its body. Passing a DBpedia file told the user
  nothing and landed no rows.
- **D-DATA-1 stood open exactly as TF-007 filed it.**
  `etl/sources.py` line 8 said "(40 files)" and the
  `ReferenceDataETL` docstring said "the 40 built-in
  knowledge reference files" — while TF-007's measured
  census of the library is 215 root files + 17 SRD
  chapters.
- **No documentation index and no new-source recipe
  existed — D-ETL-2's recipe share.** Part VI's
  fourteen architecture docs (TF-105..TF-118) were
  filed nowhere as a set, and the filings a new
  source needs — registry, schema gate, dedup
  identity, update behaviour, poison screening,
  builder contract, CLI wiring, dry-run proof,
  metrics, offline boundary, its own doc — lived only
  as fourteen slices' separate habits, in no order,
  with nothing requiring a source to be filed in
  TF-106's registry before it was documented or built.

## 2. The documentation index

`DOC_INDEX` in `etl/etl_docs.py` files every Part VI
architecture doc, in slice order, each with its slice
id, file path and one-line subject:

| Slice | Doc | Subject |
|---|---|---|
| TF-105 | `docs/architecture/ETL_MAPPED.md` | the pipeline mapped as data |
| TF-106 | `docs/architecture/SOURCE_REGISTRY.md` | provenance + licence per source; quarantine by rule |
| TF-107 | `docs/architecture/WIKIDATA_PIPELINE.md` | wikidata hardened: resumable, checkpointed |
| TF-108 | `docs/architecture/SCHEMA_VALIDATION_GATE.md` | malformed records refused with reasons |
| TF-109 | `docs/architecture/DEDUP_AND_MERGE.md` | dedup & merge per filed table identity |
| TF-110 | `docs/architecture/SUBSET_CRAFT.md` | profile-sized subsets, measured |
| TF-111 | `docs/architecture/EMBEDDING_AT_INGESTION.md` | embeddings per profile, resumable, checksummed |
| TF-112 | `docs/architecture/INGESTION_DRY_RUN.md` | test-ingest into a scratch store, full report |
| TF-113 | `docs/architecture/CANON_INGESTION.md` | the owner's materials, private stores only |
| TF-114 | `docs/architecture/UPDATE_INGESTION.md` | re-runs update by the lifecycle rules |
| TF-115 | `docs/architecture/INGESTION_METRICS.md` | records/sec, refusal rate, store growth per run |
| TF-116 | `docs/architecture/POISONED_SOURCE_FIXTURES.md` | adversarial content refused / quarantined |
| TF-117 | `docs/architecture/OFFLINE_INGESTION.md` | everything except the fetch, proven under a cut |
| TF-118 | `docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md` | the reference library, reproducible + checksummed |
| TF-119 | `docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md` | this index + the new-source recipe |

The static checker verifies every indexed path exists
on disk, the slices are exactly TF-105..TF-119 in
order, and every Part VI module's filed doc constant
appears in the index.

## 3. The new-source recipe

`RECIPE_STEPS` files the ordered new-source recipe —
each step bound to the real thing it files into,
re-checked live (import + getattr for package modules;
the root instrument by its file, since importing it
configures logging, TF-057's filed side effect):

1. **File the source in TF-106's registry first** —
   `etl/source_registry.py` (`SOURCES`,
   `check_source_use` gate): provenance + a verified
   licence, BEFORE documenting or building. This
   discharges D-ETL-2's TF-119 share.
2. **File the record schema in TF-108's gate** —
   `etl/schema_gate.py` (`RECORD_SCHEMAS`).
3. **Name the target table + filed identity in
   TF-109's filing** — `etl/dedup_merge.py`
   (`TABLE_IDENTITIES`).
4. **State update behaviour per TF-114's rules** —
   `etl/update_ingestion.py` (`UPDATE_RULES`).
5. **Name the landed text fields for TF-116's
   screening** — `etl/poisoned_source.py`
   (`SOURCE_TEXT_FIELDS`).
6. **Build the builder to TF-105's contract** —
   `etl/etl_map.py` (`SOURCE_BUILDERS`): initialises
   the schema itself, `FileNotFoundError` on a missing
   input, returns a stats dict, fresh gate / merge /
   update / poison ledgers per build; file the builder
   in TF-105's map in the open.
7. **Wire the CLI** — a `forge_memory` per-source
   command, plus an `all` step in `build_all` if the
   source belongs in full builds (numbered, with a
   skipped message when its file is not passed).
8. **Prove it by TF-112's dry-run** —
   `etl/ingestion_dry_run.py` (`dry_run_ingestion`)
   into a scratch store, never the target.
9. **Report it by TF-115's metrics** —
   `etl/ingestion_metrics.py` (`measure_ingestion`).
10. **File its offline / fetch boundary per TF-117**
    — `etl/offline_ingestion.py` (`FETCH_STEPS`,
    `OFFLINE_STEPS`).
11. **Write its doc + register it in `DOC_INDEX`** —
    `etl/etl_docs.py` (this module), doc in the
    sibling style, with slice id, path and subject.
12. **Synthetic fixtures only** — TF-017, as
    TF-105's `run_synthetic_pipeline` does: canon is
    never a fixture.

The worked example is a tiny fictional source, the
"lantern gazetteer" (`EXAMPLE_SOURCE`).
`example_recipe_application()` judges it against the
live registries: it is filed nowhere, so its
application is incomplete and its problems name the
missing registry filing first, then the missing
schema, identity and poison filings and the missing
index registration. No real builder and no real
registry entry are added for it — the example proves
the recipe's checklist catches an unfiled source.

## 4. D-ETL-1 and D-DATA-1 — FIXED

- **D-ETL-1.** `build_all` now runs **seven** steps:
  1 schema, 2 reference, 3 wikidata, 4 conceptnet,
  5 geonames, **6 DBpedia — only when `--dbpedia-json`
  is passed, otherwise a skipped message naming the
  missing option**, 7 embeddings + FTS rebuild unless
  `--skip-embeddings`. The `--dbpedia-json` click
  option and the per-source `dbpedia` command are
  unchanged. TF-105's map was updated in the open
  (`etl/etl_map.py` + `ETL_MAPPED.md` §6): the sources
  stage sits at steps 2–6, embeddings at step 7, the
  DBpedia builder's "NOT wired" note is struck, the
  orchestration steps/gaps lose the DBpedia gap, the
  rule `build_all_never_ingests_dbpedia` is replaced
  by `build_all_ingests_dbpedia_when_passed`, and
  TF-105's static checker now **requires** `DBpediaETL`
  in `build_all` between `GeoNamesETL` and
  `build_entity_embeddings`. TF-119's tests prove the
  wiring live: a synthetic DBpedia file through the
  real `all` command (reference directory filed to a
  synthetic library by monkeypatch, embeddings
  skipped) lands its rows in `dbpedia_entities`.
- **D-DATA-1.** Both stale counts in
  `etl/sources.py` now state TF-007's measured truth:
  the built-in knowledge reference files — **215 root
  files + 17 SRD chapters**. The static checker pins
  that no "40 built-in" / "40 files" claim remains in
  that module and that the measured census is stated.

## 5. D-ETL-2 — disposition

**FIXED, with its standing shape stated plainly.**
D-ETL-2 had two shares: TF-112's (the dry-run
consults the registry first — discharged by TF-112)
and TF-119's (the new-source recipe files a source in
the registry before documenting it — discharged here,
as recipe step 1 and the rule
`registry_before_documentation_or_building`).
The builders themselves still consult nothing at
build time — TF-106's filed limit, pinned by TF-106's
own checker, which still passes unchanged. That
standing shape is stated here, **not** claimed fixed:
what TF-119 fixes is the filing order a new source
must follow, not the builders' runtime behaviour.

## 6. Filed limits

- **No new CLI commands are added by this slice for
  the TF-115 / TF-116 / TF-117 / TF-118 drivers** —
  nor for this module. The recipe is the wiring
  contract a future source follows when *its* slice
  adds wiring; TF-119 states that plainly rather than
  claiming driver CLIs that do not exist.
- The index files Part VI's docs only; earlier parts'
  docs are their own filings.
- The example is a checklist judgement over live
  registries as data: it builds nothing, writes
  nothing, and files no real entry anywhere.
- The DBpedia step is proven on a tiny synthetic
  NDJSON file; a real DBpedia export at scale is not
  ingested here.

Proof grade: FIXTURE — synthetic sources through the
real builders and the real `all` command on this VM.
No real dump was ingested at scale; that is NOT
PROVEN, never claimed.
