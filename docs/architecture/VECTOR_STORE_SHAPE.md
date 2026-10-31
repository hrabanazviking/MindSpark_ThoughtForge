# TF-054 — Vector Store Shape: Restarts, Atomic Writes, Corruption Fixtures

*Part III, slice TF-054. Code:
`src/thoughtforge/etl/vector_store.py` (the shape as data,
blob classification, the searchable-row loader, the builders'
gate, write hardening, inspection, quarantine, and
`vector_store_problems()` re-deriving every claim below from
the live tree), wired into `src/thoughtforge/etl/embeddings.py`
(builders gate + harden; both searches load through the
loader). Tests: `tests/test_tf054_vector_store.py`.*

## 1. The question and the measured starting state

The roadmap asks: does the store's vector index survive
restarts, and what happens when it is corrupted? Measured
before this slice, by direct probe on synthetic stores:

- **There is no separate vector index.** The "index" is two
  tables in the same SQLite knowledge DB as everything else —
  `embeddings` (keyed by QID) and `reference_embeddings`
  (keyed by chunk id). Every vector search loads every blob
  for its model and brute-force cosine-scores the whole
  matrix in memory. No ANN structure exists; none was
  documented, and a reader could easily assume one.
- **Restart survival was true but unproven.** A committed
  row survives a close/reopen; a row left uncommitted rolls
  back cleanly. Nothing stated or pinned that contract.
- **One corrupt row poisoned the whole search (D-VEC-1).**
  Both searches stacked every blob with `np.stack` before
  scoring. One row with a NULL, empty, truncated (40-byte),
  or non-multiple-of-4 blob raised `TypeError`/`ValueError`
  and the search returned nothing — every good row lost
  with the bad one (the forge's TF-053 degrade then saved
  the SQL leg, at the price of the entire vector leg).
  A right-length blob holding non-finite floats (`0xFF`
  bytes decode to NaNs) scored NaN — and NaN sorted
  *ahead of* a perfect 1.0 match. Garbage outranked truth.
- **The builders manufactured poison rows blindly
  (D-VEC-2).** With a fake model returning
  10-dimensional vectors, `build_entity_embeddings`
  stored 40-byte blobs without a check — rows the next
  search would choke on. And embedding writes ran at the
  schema's `synchronous = NORMAL`: under WAL that cannot
  corrupt, but a power failure may lose the last committed
  batches. The Samtal lesson — WAL +
  `synchronous = FULL`, commit before the answer is
  trusted — was applied nowhere in the embedding path.

## 2. The shape, as data

`VECTOR_STORE_SHAPE` in `etl/vector_store.py`:

| aspect | value |
|---|---|
| store file | the SQLite knowledge DB (`get_knowledge_db_path`) — no separate index file |
| tables | `embeddings` (key `qid`), `reference_embeddings` (key `chunk_id`) |
| blob format | little-endian float32, exactly 384 × 4 = 1536 bytes (dimension from TF-053's registry, pinned == `EMBEDDING_DIM`) |
| model filter | exact match on `model_name`, canonical spelling only (TF-053) |
| index structure | none — brute-force cosine scan over every stored blob, in memory, per search |
| journal | WAL (schema) |
| write durability | `synchronous = FULL` on embedding write connections (this slice) |
| read durability | `synchronous = NORMAL` (schema default, unchanged) |
| batch atomicity | one transaction per builder batch — a batch lands whole or not at all |
| restart rule | committed rows survive close/reopen; uncommitted rows roll back |

## 3. Corruption, classified and isolated

`classify_blob` judges every stored blob by one rule:
`ok` / `null_blob` / `empty_blob` / `wrong_byte_count` /
`non_finite`. A zero vector is finite and scores 0.0 —
`ok`, not corruption.

Both searches now load through `load_searchable_rows`:
a corrupt row is skipped *by row*, logged with its table,
id, and class, and the good rows' matrix is what gets
scored. A store whose rows are all corrupt searches empty
instead of raising. Search never deletes anything.

**The builders' gate.** `validated_blob` is applied to
every encoded vector before storage: exactly the
registered dimension and all-finite, or the batch raises
`ValueError` naming the defect *before any row of that
batch is written*. Earlier committed batches stay — that
is a partial index, not a corrupt one, and the builders'
existing LEFT-JOIN resume re-offers exactly the missing
rows (pinned: a mid-build crash at batch 2 of 3 leaves 1
row; a healthy resume builds the other 2).

**Atomic writes, the Samtal lesson.** Both builders open
their write connection through `harden_write_connection`
(`PRAGMA synchronous = FULL`). Combined with WAL and the
per-batch transaction, a committed batch is durable and a
batch is never half-landed. The schema's general/read
connections keep NORMAL — the hardening is scoped to the
embedding write path and stated here, not silently global.

## 4. Inspection and the repair

`inspect_vector_store(db_path)` returns a
`VectorStoreReport` as data — presence, readability,
`PRAGMA integrity_check`, tables present/missing, per-table
row counts split ok/corrupt, and every corrupt row named
by id and class. It never raises: a missing file, a
missing table, and a file that is not a database are all
verdicts in the report.

`quarantine_corrupt_rows(db_path)` is the repair: it
deletes exactly the rows `classify_blob` judges corrupt,
in one transaction on a hardened connection, and reports
what it removed by id and class. Good rows are
byte-untouched (pinned). It is an explicit operator act —
neither search nor inspection calls it, because a search
that silently deletes is a search that hides its losses.

## 5. Boundaries stated

- **A corrupt file is not row corruption.** A bad header
  or truncated DB is named `unreadable` by inspection; the
  searches still raise the SQLite error and the forge's
  TF-053 degrade keeps hybrid retrieval on its SQL leg.
  Rebuilding a destroyed file is `etl/db_integrity`'s
  repair road.
- **A missing embeddings table** is the same class: named
  by inspection, raised by search, degraded by the forge.
- **Skipped is not fixed.** Search-time skips are logged
  per search and named by inspection; only quarantine
  removes a corrupt row, and only when called.
- TF-053's `stored_embeddings_problems` still judges the
  stored estate against the registry (alias spellings,
  byte counts). This slice's classification extends the
  same byte rule to NULL and non-finite blobs and to the
  search path itself; the two checks agree on byte count
  by construction (both read the registry dimension).

## 6. Docket

- **D-VEC-1 — found and FIXED by this slice.** One corrupt
  blob poisoned the whole vector search (raise on stack,
  or NaN outranking a perfect match). Both searches now
  isolate corrupt rows by row.
- **D-VEC-2 — found and FIXED by this slice.** The builders
  stored wrong-dimension vectors blindly and wrote at
  `synchronous = NORMAL`. They now gate through
  `validated_blob` and write on FULL-synchronous
  connections.

## 7. Proof grade

FIXTURE — synthetic stores on this VM, a fake embedding
backend where a model is needed, and a fresh-process
restart proof (a subprocess reopens the store file and
inspects it: 1 row, integrity ok). No real lived-in vector
store exists on this machine (TF-041), and vector
retrieval *quality* remains NOT PROVEN (D-ENV-1) — this
slice proves the store's shape, durability, and corruption
behaviour, never the goodness of its answers.
