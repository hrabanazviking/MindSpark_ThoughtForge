# ThoughtForge Architecture — The Reference Library as a Source (TF-118)

The reference library as a source (TF-058's write side):
its ingestion reproducible + checksummed. Part VI's
fourteenth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The proof lives as verifiable data in
`src/thoughtforge/etl/reference_ingestion.py`
(`REFERENCE_RULES`, `LibraryLedger`,
`ingest_reference_library()`, `store_reference_digest()`,
`verify_reference_store()`, `check_reproducibility()`,
checked by `reference_ingestion_problems()` and
`reference_ingestion_live_problems()`); the builder in
`etl/sources.py` gained its `library_gate` ledger,
skip-by-rule and dedup-by-hash in the open, TF-105's
living map gained its `reference_library` orchestration
entry in the open, and TF-113's canon filing was
updated in the open to the new builder behaviour. The
tests are `tests/test_tf118_reference_library_as_source.py`.
This slice **fixes D-DATA-4** (its TF-058 read share
already stood fixed) and **adds and fixes D-REF-2 and
D-REF-3**.

## 1. Measured start

Probed live this tick, before any change:

- **The ingestion side of D-DATA-4 stood open exactly
  as TF-058 filed it.** A probe library holding one
  unique readable file, a byte-identical copy of it, a
  .jpg, a .csv and an empty .md built to
  `{'files': 2, 'chunks': 2, 'skipped': 1}`: the
  duplicate was chunked a second time under its own
  name — the store held the same content twice — and
  the .jpg/.csv were passed over by a bare `continue`,
  uncounted (the one skip was the empty file) and
  named nowhere.
- **No checksum existed anywhere on this road
  (D-REF-2).** `etl/sources.py` carried no `sha256` at
  all; no per-file hash, per-chunk checksum, store
  digest or manifest was written beside a store. A
  stored chunk edited in place (probed with one
  `UPDATE` setting its content to `TAMPERED`) was
  detectable by no tool in the tree, and a plain
  re-run silently overwrote it through the merge
  rules, leaving no record the store had drifted.
- **Reproducibility held in fact but was proven
  nowhere (D-REF-3).** Two fresh builds of the probe
  library produced equal chunk rows — but no test,
  checker, digest or manifest had ever compared two
  builds, so the claim rested on reading the builder's
  sort order rather than on checkable evidence.

## 2. What the slice builds

- **Skip by rule, chunk once — in the builder.**
  `ReferenceDataETL` now holds a fresh `library_gate`
  (`LibraryLedger`) per build. Every unsupported file
  is classified (TF-058's image / other classes),
  counted in `skipped` and named in the ledger. Every
  readable file is SHA-256'd before chunking; a
  byte-duplicate of an earlier sorted-order file is
  named in the ledger with its keeper, counted in
  `skipped`, never chunked — and its stale stored
  chunks from a pre-fix build are retired in the same
  run through TF-114's retirement, ledgered in the
  builder's `update_ledger`. The probe library now
  builds to `{'files': 1, 'chunks': 1, 'skipped': 4}`
  in its TF-118 fixture shape (two unique files build
  2 / 2 / 4 — see the live checker).
- **The checksummed manifest.**
  `ingest_reference_library()` runs the real builder
  and files `<db>.reference-manifest.json` beside the
  store, atomically and content-hashed (the TF-113
  posture): a TF-058 metadata-only inventory, per-file
  SHA-256 + status (`ingested` / `duplicate` /
  `skipped_by_rule` / `no_chunks`), per-chunk
  checksums over the filed chunk fields, a library
  digest over the filed file hashes, a store digest
  over the landed chunk rows, the build stats, store
  counts, and the builder's library / merge / update /
  poison ledgers whole. The manifest carries hashes
  and counts only — no file or chunk content. A
  TF-058 inventory before and after proves the source
  untouched, or the report files
  `source_untouched=False` in the open.
- **Verification that names drift.**
  `verify_reference_store()` re-derives the manifest's
  own hash, every chunk checksum and the store digest,
  and the source files' current hashes. It returns a
  report naming each tampered (checksum_mismatch),
  missing or extra chunk and each changed, missing or
  added source file — read-only, repairing nothing. A
  clean store verifies `ok: True`.
- **Reproducibility as three agreements.**
  `check_reproducibility()` builds one library into
  two independent stores and calls it reproducible
  only when stats, every chunk checksum and the store
  digest all agree. A re-run into the same store
  leaves the store digest unchanged (proven in the
  live checker).

## 3. Filed limits

- The store digest covers the `reference_chunks` rows
  in (source_file, chunk_index) order — the ingestion
  claim of this source. Embeddings and the ingestion
  log are other stages and are filed as counts and
  leg states, not digested here.
- A plain-builder re-run still repairs drift silently
  (TF-109/TF-114 own re-run semantics). This slice's
  addition is that drift is detectable and nameable
  *before* a re-run, through the manifest + verify
  road.
- Embeddings run only through TF-111's driver with an
  injected encoder; with none, the leg is filed
  not-run and no model is loaded.
- No CLI wiring is added; TF-119 owns ETL docs + CLI.
- Neighbour filings updated in the open: TF-105's map
  (reference builder notes, orchestration entry, the
  synthetic pipeline's expected reference stats —
  the pipeline .jpg is now a counted by-rule skip) and
  TF-113's canon road (the builder now chunks canon
  duplicates once: canon stats, store counts, the
  retrieval keeper name and the embedding leg counts
  were re-pinned to the deduped values).

Proof grade: FIXTURE — synthetic libraries through
the real builder on this VM. The real built-in
library is not in this sparse checkout and was never
opened; ingesting it at scale is NOT PROVEN, never
claimed.
