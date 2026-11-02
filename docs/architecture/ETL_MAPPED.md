# ThoughtForge Architecture — ETL Mapped (TF-105)

ETL mapped: sources → schema → embeddings → subset → store,
drawn from code; the pipeline's contracts per stage.
Part VI's first slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The map lives as verifiable data in
`src/thoughtforge/etl/etl_map.py` (`PIPELINE_STAGES`,
`SOURCE_BUILDERS`, `SCHEMA_FACTS`, `EMBEDDING_FACTS`,
`SUBSET_FACTS`, `ORCHESTRATION`, `ETL_RULES`, checked by
`etl_map_problems()` and `etl_map_live_problems()`); the
tests are `tests/test_tf105_etl_map.py`. This slice also
**fixes D-SUBSET-1** (§5) and **files D-ETL-1** (§6;
**FIXED by TF-119**, which updated this map in the open).

## 1. Measured start

Probed live this tick before the module existed: the ETL
wing is five source builders, one schema module, one
embeddings module (plus TF-053's discipline and TF-054's
vector-store shape beside it), and one subset builder —
with **no orchestrator inside the package**: nothing in
`src/thoughtforge/etl/` calls another stage. The only
pipeline runner in the tree is the root instrument
`forge_memory.py`. Two live probes shaped the slice:

- A pi_zero subset build of a store holding one reference
  chunk copied **0 chunks and reported success**, its only
  trace a note: `reference_chunks skipped: no such column:
  retrieval_score` (§5).
- The orchestrator's `all` command accepts `--dbpedia-json`
  and then never ingests it (§6).

## 2. The five stages

| Stage | Modules | Consumes → produces |
|---|---|---|
| sources | `etl/wikidata.py`, `etl/sources.py` | source files → rows in the target tables |
| schema | `etl/schema.py` | a DB path → the 10-table + 3-FTS shape, idempotently |
| embeddings | `etl/embeddings.py` (+ discipline / vector-store) | stored entities + chunks → validated BLOBs |
| subset | `etl/subset.py` | a populated store + profile → a smaller store file |
| store | `etl/schema.py`, `knowledge/forge.py` | everything written → the database retrieval reads (TF-041) |

The roadmap's order is the data flow. The one
orchestrator runs it differently: schema **first** (step
1), then sources (steps 2–6), embeddings last (step 7),
and the subset not at all. Both orders are filed in the
map; the difference is stated, never smoothed.
*Updated in the open by TF-119:* sources were steps 2–5
and embeddings step 6 of 6 until TF-119 wired DBpedia
in as step 6 of 7 (§6).

## 3. The sources stage — five builders, five contracts

| Builder | Input | Table(s) | Write mode | Batch | Logs run? | Synchronous |
|---|---|---|---|---|---|---|
| WikidataETL | JSON dump (.json/.gz), ijson-streamed | entities, statements, labels | REPLACE / append / REPLACE | 2000 | **yes** | OFF |
| ConceptNetETL | assertions CSV, tab-delimited | conceptnet_edges | INSERT OR IGNORE | 5000 | no | OFF |
| GeoNamesETL | tab-delimited text, 19 cols | geonames_places | INSERT OR REPLACE | 5000 | no | OFF |
| DBpediaETL | NDJSON abstracts | dbpedia_entities | INSERT OR REPLACE | 2000 | no | OFF |
| ReferenceDataETL | reference dir (.md/.json/.jsonl/.yaml/.yml/.txt) | reference_chunks | INSERT OR REPLACE | 1000 chars | no | NORMAL |

Shared contract: every builder calls `initialize_schema`
itself, raises `FileNotFoundError` on a missing input
before writing, and returns its own stats dict. Filed
differences: only wikidata writes the `ingestion_log`
(start + finish, complete or failed); only wikidata
commits on a schedule (every 50,000 entities) and guards
host memory (commit early above 85%, psutil optional);
the reference builder passes unsupported suffixes over
**uncounted** — only empty-chunk files and read errors
count as skipped (TF-118 owns that share of D-DATA-4) —
and ends by rebuilding the FTS indexes (TF-058's
D-REF-1 fix, so an ingested library is a searchable one).
*Updated in the open by TF-118:* that share is now fixed —
unsupported suffixes are skipped by rule (classified,
counted in `skipped`, named in the builder's
`library_gate` ledger), byte-duplicate readable files
are chunked once under their sorted-order keeper, and
the checksummed ingestion filing lives in
`etl/reference_ingestion.py` +
`docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md`.

## 4. Schema, embeddings, subset, store — the filed facts

- **Schema:** 10 base tables, 3 external-content FTS5
  indexes, 15 named indexes, WAL throughout;
  `get_entity_count` counts 8 of the 10 tables (never
  reference_embeddings or ingestion_log).
- **Embeddings:** default model `all-MiniLM-L6-v2`
  (canonicalised by TF-053), 384 dimensions; entity text
  is `label: description`, reference text is
  `title: content[:500]`; both builders select only rows
  missing an embedding (a re-run builds nothing) and
  write validated BLOBs at `synchronous = FULL`
  (TF-054). Searches are brute-force cosine — entities
  top-10 / reference top-5, floor 0.3. With
  sentence-transformers absent the loader raises a
  RuntimeError naming the install (D-ENV-1 on this VM);
  this slice proves the builders through a fake model in
  the real cache, and says so.
- **Subset:** per-profile entity limits (pi_zero 50k,
  phone_low 80k, pi_5 200k, desktop_cpu 1M, desktop_gpu
  2M, server_gpu unlimited) — a profile JSON may override
  via `subset_entity_limit`, but no shipped profile
  carries the key, so the table is what runs. A subset
  copies **three tables' rows** (entities top-N by
  popularity, their statements, reference chunks capped
  at `max(1000, limit // 10)` when limited); the other
  seven tables are schema-cloned and row-empty — a subset
  has no labels and no vector search until embeddings are
  rebuilt against it. The schema clone copies table DDL
  only: the 15 named indexes are **not** recreated.
- **Store:** one SQLite file; its readers and retrieval
  flow are TF-041's map. This slice's live pipeline ends
  with `get_entity_count` reading the built store back.

## 5. D-SUBSET-1 — found by this slice, FIXED in scope

`etl/subset.py`'s reference-chunk copy ordered by
`retrieval_score DESC` — a column that exists nowhere in
the schema (grep: the only occurrence in `src/` was that
query). For every limited build the query raised
`OperationalError`, the copy's handler filed it as a
note, and the build returned **success with zero
reference chunks** — every edge profile's subset silently
lost the whole reference library, the knowledge an edge
device most needs offline. The phase-5 tests never
asserted the chunk count of a limited build, so nothing
caught it. The fix orders by `chunk_id` (insertion order;
the schema carries no per-chunk value score to order
by), and the live checker now proves a pi_zero subset of
the synthetic store keeps its chunk. The static checker
pins the fix (the DESC ordering returning turns it red).

## 6. D-ETL-1 — FIXED by TF-119

`forge_memory.py all` declared `--dbpedia-json`, echoed
six steps, and ran five kinds of work: schema, reference,
wikidata, conceptnet, geonames, embeddings. DBpedia had
its own per-source command but **no step in `all`** — a
user passing the file got no error and no rows. TF-105
mapped and pinned the gap (its static checker failed if
the option vanished or a DBpedia step appeared, so the
fix had to update the map in the open).

TF-119 wired the step: `all` now echoes seven steps —
1 schema, 2 reference, 3 wikidata, 4 conceptnet,
5 geonames, **6 DBpedia (only when `--dbpedia-json` is
passed; otherwise a skipped message naming the missing
option)**, 7 embeddings + FTS rebuild (unless
`--skip-embeddings`). The per-source `dbpedia` command
and the `--dbpedia-json` option are unchanged. This
map was updated in the open with the fix: the sources
stage's `position_in_build_all` is steps 2–6, the
embeddings stage's is step 7, the DBpedia builder's
notes no longer claim it is unwired, the
`ORCHESTRATION` steps list seven steps with the
DBpedia gap removed from its gaps, the rule
`build_all_never_ingests_dbpedia` is replaced by
`build_all_ingests_dbpedia_when_passed`, and the
static checker now **requires** `DBpediaETL` in
`build_all`, between `GeoNamesETL` and
`build_entity_embeddings`, in the filed order.

## 7. What this slice does not do

No builder's behaviour changes except the D-SUBSET-1
ordering. Provenance and licensing per source are
TF-106; wikidata hardening (resumable, checkpointed) is
TF-107; schema validation at the gate is TF-108; dedup
and update ingestion are TF-109/TF-114; subset craft
proper (profile-sized subsets measured) is TF-110;
embedding-at-ingestion discipline is TF-111. D-COV-2
stands OPEN for TF-107/TF-112, narrowed: this slice's
tests are the first to import and run all five builders,
both embedding builders, and the subset builder directly.

**Proof grade: FIXTURE** — synthetic inputs through the
real builders, a fake embedding model in the real cache,
a real limited subset build, on this VM. An at-scale
ingestion of any real dump is NOT PROVEN, never claimed.
