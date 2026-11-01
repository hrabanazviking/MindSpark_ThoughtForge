# ThoughtForge Architecture — Embedding at Ingestion (TF-111)

Embedding at ingestion: model per profile (TF-053),
resumable, checksummed. Part VI's seventh slice.
Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`. (The slice's module
was drafted by the preceding keeper tick and left
unbanked in the tree; this tick verified it against
its own live checker, then completed the slice —
tests, this write-up, the docket — in the open.)

The driver lives as verifiable data in
`src/thoughtforge/etl/embedding_ingestion.py`
(`INGESTION_RULES`, `embedding_plan()`,
`build_embeddings_at_ingestion()`, `row_checksums()`,
`estate_digest()`, checked by
`embedding_ingestion_problems()` and
`embedding_ingestion_live_problems()`); the D-EMB-3
fix is enforced in `src/thoughtforge/etl/embeddings.py`
(both builders); TF-105's living map was updated in
the open (the embeddings stage contract, the
only-missing scope, and the ingestion-driver fact in
`EMBEDDING_FACTS`); the tests are
`tests/test_tf111_embedding_ingestion.py`. This slice
**adds and fixes D-EMB-3**.

## 1. Measured start

Probed live before this slice changed anything:

- **TF-053's assignment was wired nowhere.** Neither
  builder takes a profile — a direct build on
  pi_zero's behalf would embed at the default batch
  128 through the sentence-transformers stack TF-053
  ruled it cannot carry, and phone_low would batch at
  128/64 instead of its assigned 16.
- **D-EMB-3 — building under a second model silently
  built nothing.** Both builders selected "missing"
  with a bare `LEFT JOIN ... ON qid` / `ON chunk_id`
  — no `model_name` in the join — so any row embedded
  under one model blocked building it under another,
  while the searches (which do filter by model) found
  nothing for the second model. A profile switch
  changed which model searches use without changing
  which rows exist.
- **No checksum existed anywhere in the path.**
  TF-054 gates a vector's dimension and finiteness
  before the write and TF-053's stored check judges a
  blob's LENGTH — a same-length corruption of a
  stored blob was undetectable by any surface.
- **No checkpoint existed.** The builders commit per
  batch, so an interruption leaves a partial,
  resumable index — but nothing recorded which model,
  which profile, or how far a build got, and nothing
  distinguished a finished build from an interrupted
  one.

## 2. The plan

`embedding_plan()` resolves a run's ask once, from
TF-053's live assignment — never a copied table:

| Ask | Model | Batches (entity / reference) | Source |
|---|---|---|---|
| profile named | TF-053's assigned model (canonical) | TF-053's assigned batch, both legs | `profile:<id> (TF-053 assignment)` |
| pi_zero | None — no build | 0 | TF-053 assignment (runtime none) |
| no profile | the builders' default model | 128 / 64 (the builders' filed defaults, TF-105) | builders' defaults |
| explicit `model_name` / `batch_size` | the override wins | the override wins | recorded as `explicit:...` |

An unknown profile raises by name before any store
is opened. Building a not-assigned plan raises
`EmbeddingNotAssignedError` — nothing is written and
no checkpoint is filed; pi_zero's retrieval is
SQL-only by TF-053's rule, not by a failed build.

## 3. The driver

`build_embeddings_at_ingestion(db, ...)` fills both
tables in the builders' own text shapes (TF-105),
batched at the plan's sizes, selecting only rows
missing an embedding **under the run's model**
(D-EMB-3's rule, now also the builders' own join):

1. **Checksummed.** Each row's validated blob
   (TF-054's `validated_blob`, on a FULL-hardened
   connection) is SHA-256'd before the write; after
   the batch commits, the stored blobs are read back
   and every checksum must match, or the run stops
   with `ChecksumMismatchError` naming table + id.
2. **Checkpointed.** A sidecar beside the DB
   (`<db>.embedding-checkpoint.json`), written
   atomically (tmp + `os.replace`) only after the
   commit + read-back it describes. It carries the
   run fingerprint (resolved DB path, canonical
   model, profile, both batch sizes), every verified
   batch's per-row checksums, the counts as of the
   last verified batch, and its status.
3. **Resumable by fingerprint.** Resume continues
   only on an exact fingerprint match — any
   difference raises `CheckpointMismatchError` naming
   each differing field; an unreadable or malformed
   record raises `CheckpointCorruptError`, never a
   silent restart; `resume=False` discards the record
   explicitly. Every resume first re-verifies the
   recorded rows' checksums against the store — a
   tampered or rotted recorded row raises by name. A
   completed checkpoint, re-verified, IS the result:
   it returns without encoding anything.
4. **Proven at the end.** At completion the driver
   runs TF-053's `stored_embeddings_problems` for
   the run's model; any problem raises instead of
   filing a complete checkpoint over a dirty estate.
   The result carries the estate digest — SHA-256
   over every stored row checksum for the model,
   both tables (`row_checksums()` / `estate_digest()`
   expose the same sums as data).

An `encoder` may be injected (tests, TF-112's
dry-run): it receives (texts, model_name) and returns
the vector matrix, and the checksum, checkpoint and
estate machinery judges its output exactly as it
judges the real encoder's. By default the driver
encodes through `etl/embeddings.encode_texts`, so
TF-053's runtime story is unchanged — the ONNX
profile's export path is TF-070's, and no ONNX
encoder is wired here.

## 4. Filed limits

- The checkpoint is a sidecar, not a schema table —
  TF-105's filed schema facts (10 tables) stand.
- **One model per row in storage (D-EMB-4, OPEN).**
  Found by this slice's own regression test: the
  embeddings tables' primary key is the row id alone
  (TF-041's filed key), so building under a second
  model REPLACES the first model's rows — the two
  models' estates cannot coexist in one store.
  Per-model selection (D-EMB-3) makes each build
  honest about what it builds; a composite
  (row id, model) primary key is a schema migration,
  owned by TF-177's upgrade path, and is not smuggled
  into this slice.
- Checksums prove the stored bytes are the bytes the
  encoder produced and the read-back saw — they do
  not prove the vectors are GOOD (retrieval quality
  remains NOT PROVEN, TF-053's standing caveat).
- A phone_low run on this VM reaches the same honest
  package-missing wall as any other profile
  (D-ENV-1), with its plan (model + batch 16)
  resolved and recorded before the wall.

## 5. What is NOT proven

A real-model ingestion at scale (sentence-transformers
is absent on this VM, D-ENV-1 — the real encoder's
leg here is the filed RuntimeError, never a run),
a real kill -9 / power loss against a mid-batch
write, and retrieval quality of the embedded estate.
Proof grade: FIXTURE — synthetic stores through the
real driver with a fake encoder on this VM. TF-112's
dry-run is the filed first consumer of the injected
encoder; TF-120's Part VI gate owns the full round
trip.
