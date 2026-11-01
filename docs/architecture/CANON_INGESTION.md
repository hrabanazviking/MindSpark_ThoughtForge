# ThoughtForge Architecture — Canon Ingestion Path (TF-113)

Canon ingestion path: Volmarr's own materials ingested by
the same pipeline — private stores only, fixtures
synthetic. Part VI's ninth slice. Built 2026-11-01 on
branch `yrsa/thoughtforge-road-build`.

The driver lives as verifiable data in
`src/thoughtforge/etl/canon_ingestion.py`
(`CANON_PROVENANCE`, `CANON_RULES`, `check_canon_use()`,
`ingest_canon()`, checked by `canon_problems()` and
`canon_live_problems()`); TF-105's living map was updated
in the open (the orchestration entry now names this
canon runner beside `all` — it is not a step of `all`,
whose filed shape stands). The tests are
`tests/test_tf113_canon_ingestion.py`. This slice
**adds and fixes D-CANON-1**.

## 1. Measured start

Probed live before this slice changed anything:

- **No canon path existed anywhere.** No module, no
  `forge_memory.py` command and no test named one. The
  owner's own materials could enter a store only
  through the generic `ReferenceDataETL` — the built-in
  library's builder — indistinguishably from that
  library, with no record that the material was the
  owner's canon at all.
- **Nothing guarded the store.** The generic builder
  writes into any `db_path` it is handed: no affirmation
  that the target is a private local store, nothing
  forbidding the store inside the canon directory, and
  nothing in or beside the resulting store saying the
  chunks were private canon (probe: a synthetic canon
  directory built into an arbitrary DB — rows carry the
  schema-default `source` of `reference`, and no
  manifest exists).
- **The TF-106 registry has no canon entry** —
  `check_source_use("owner_canon", ...)` raises
  unregistered-by-name. Correct for a registry of
  *external* sources, but it left the owner's own
  materials with no gate and no provenance record
  anywhere.
- **Skips and duplicates were silent on this road:**
  the probe's stats were
  `{'files': 2, 'chunks': 2, 'skipped': 0}` for one
  unique readable file, one byte-identical copy, one
  .jpg and one .csv — the image and CSV passed over
  uncounted and unnamed; the duplicate chunked twice
  with nothing recording it.

## 2. The road

`ingest_canon(canon_dir, db_path, private_store=True)` —
one call, in this order:

1. **The gate and the affirmation, first.**
   `check_canon_use(ingest_private)` returns the filed
   provenance; without an explicit `private_store=True`
   the call raises `CanonPrivacyError` before anything
   is read or written — no store is created.
2. **The directory is checked.** A missing canon
   directory raises `FileNotFoundError` naming it,
   still before any store exists. A `db_path` resolving
   inside the canon directory raises
   `CanonTargetError` — the pipeline never writes into
   the material it reads.
3. **Inventory before.** A TF-058 `ReferenceLibrary`
   inventory (metadata only: path, size, SHA-256,
   class, duplicate keeper — never content).
4. **The same pipeline runs.** The real
   `ReferenceDataETL.build()` — its chunking, its
   TF-109 merge ledger, its TF-058 FTS rebuild. This
   module builds no second chunker.
5. **Embeddings only when an encoder is injected,**
   through TF-111's driver; otherwise filed not-run.
6. **Inventory after, and the manifest.** The two
   inventories must agree exactly for the report to
   carry `source_untouched=True`. The report and the
   content-hashed sidecar manifest
   (`<db>.canon-manifest.json`, written atomically)
   file the whole story (§3).

`check_canon_use` is the standing gate for consumers:
`ingest_private` and `retrieve_private` pass with the
provenance record; `redistribute` and `publish` raise
`CanonPrivacyError`; an unknown use raises by name.

## 3. The report and the manifest

Format `thoughtforge-canon-ingestion/1`:

| Field | What it carries |
|---|---|
| `provenance` | The filed record: owner-authored, `licence: None` with the note that no licence is stated or claimed by the campaign for the owner's own materials, `privacy: private_local_only`, obligations `no_redistribution` / `no_publication` / `private_stores_only`. |
| `stats` | The real builder's stats, in TF-105's filed shape, unchanged. |
| `store_counts` | `get_entity_count` over the canon store after the build (re-measured after the embedding leg when one runs). |
| `merge` | The builder's TF-109 `MergeLedger.as_data()`. |
| `inventory` | The TF-058 metadata-only inventory: totals, per-class counts and bytes, and every file's path/size/SHA-256/class/keeper. No content. |
| `skipped_by_rule` | Every non-readable file, named, with its TF-058 class (image / other). |
| `duplicates` | Every byte-duplicate readable file, named with its keeper and hash. |
| `embeddings` | `not_run` with its reason, or TF-111's driver result when an encoder was injected. |
| `source_untouched` | Whether the before/after inventories agree exactly. |
| `manifest_sha256` | SHA-256 over the manifest's canonical JSON; the manifest file carries the same digest, so an edited manifest does not verify. |

## 4. Docket

- **D-CANON-1 — added + FIXED by TF-113.** The owner's
  own materials had no ingestion path carrying a
  private-store guard or a provenance record: they
  entered stores, if at all, through the built-in
  library's generic builder, indistinguishable from
  it, with canon provenance existing only in campaign
  prose. The path, the gate, the affirmation and the
  manifest now exist here and are proven on synthetic
  fixtures.

TF-016 totals after this slice: 109 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 63 FIXED /
3 CLOSED-BY-VERDICT.

## 5. Filed limits

- Canon is not a TF-106 registry entry: the registry
  records external sources' upstream terms, and the
  owner's materials have no upstream. TF-119's recipe
  and TF-190's final pass inherit this filing.
- The privacy affirmation is an explicit caller
  declaration, not a filesystem proof: the module
  cannot prove a path is private, so it requires the
  declaration, refuses the store-in-canon shape, and
  files the obligations in the manifest for any
  consumer of the store to read.
- Duplicates are named, not deduped, on ingestion:
  the builder's identity is per-file
  (source_file, chunk_index); ingestion-side dedup is
  TF-118's filed share of D-DATA-4 for the reference
  road generally.
- One canon directory per call; a second call into
  the same store appends through the builder's merge
  rules and rewrites the manifest for that call only.

## 6. What is NOT proven

The real canon tree: it is not in this sparse
checkout and no canon file was opened by this slice.
Proof grade: FIXTURE — synthetic canon stand-ins
through the real builder into private stores on this
VM. TF-118 owns the reference library's reproducible,
checksummed ingestion; TF-120's Part VI gate owns the
full source → store → retrieval round trip.
