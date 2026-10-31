# TF-058 — Reference-Library Binding: the Library as a Store Source

*Part III, slice TF-058. Code:
`src/thoughtforge/knowledge/reference_library.py` (the file
classes and binding rules as data, `ReferenceLibrary` —
inventory / chunks / search — and
`reference_library_problems()` re-deriving every claim below
from the live tree), plus one write-side fix in
`src/thoughtforge/etl/sources.py` (D-REF-1). Tests:
`tests/test_tf058_reference_library.py`.*

## 1. The question and the measured starting state

The roadmap asks: can the `knowledge_reference` library be
read as a store source — read-only, canon-respecting?
Measured this tick by direct probe on the TF-017 synthetic
library:

- **The library was readable only through the knowledge
  DB, after write steps.** No module could read, list, or
  search the files directly. The one reader was
  `ReferenceDataETL.build()` — a *write* pipeline into the
  `reference_chunks` table — and `KnowledgeForge` reads
  that table, never the files. With the synthetic library
  on disk and an empty (schema-only) DB, the forge
  returned **0 records**: the library, present, was
  invisible.
- **The DB road was itself broken (D-REF-1).**
  `reference_chunks` is searched through an
  external-content FTS5 table, which a plain INSERT does
  not populate — only a `rebuild` does — and neither
  `ReferenceDataETL.build()` nor the `forge_memory
  reference` command ever rebuilt it (only the
  *embeddings* step did). After a plain ETL build of the
  synthetic library (3 files, 6 chunks),
  `KnowledgeForge.sql_retrieve` returned **0 rows for
  every query tried**.
- **Skipping was by accident, not by rule (D-DATA-4).**
  The ETL's supported set {.md, .json, .jsonl, .yaml,
  .yml, .txt} silently `continue`d past everything else:
  a probed .jpg and .csv changed no counter at all
  (`skipped` stayed 0) and were named nowhere. TF-007
  counted the real stakes: 225 persona images / 152.9 MB
  live inside the real library, skipped by that silence.
- **Duplicates were double-counted.** A byte-identical
  copy of the synthetic .md under a second name ingested
  as a second full chunk set (4 files / 10 chunks for
  3 files / 6 chunks of content): DB uniqueness is
  (source_file, chunk_index), so identical content in
  two files is two records. TF-007 finding 3 named the
  rule for this slice and TF-118: dedup by blob hash
  before chunking.

## 2. The binding, as data

`FILE_CLASSES` — every library file is one of:

| Class | Extensions | Rule |
|---|---|---|
| readable | .json .jsonl .md .txt .yaml .yml | chunked and searchable through the binding |
| image | .bmp .gif .jpeg .jpg .png .tif .tiff .webp | skipped by rule — imagery is not knowledge text (TF-007) |
| other | everything else | skipped by rule — outside the ETL's supported text set |

The readable set is *exactly* the ETL's supported set;
`reference_library_problems()` pins the two equal by
reading the ETL's own build method, so the binding can
never drift from what ingestion would read.

`BINDING_RULES` states the six rules as data: read-only;
canon-respecting (metadata-only inventory, content only
as cited hits); skip-by-rule; dedup-by-hash; chunking is
the ETL's; a missing library is a verdict, not an error.

## 3. Inventory — metadata only

`ReferenceLibrary.inventory()` returns one JSON-safe
report: root, existence, total files/bytes, counts and
bytes per class, the duplicate list (file, keeper,
SHA-256), and per-file rows of *metadata only* —
relative path, extension, size, SHA-256, class,
duplicate keeper. No file content is carried in the
report (hashing reads bytes; nothing is copied or
quoted): TF-007's census discipline as a live API.

A missing directory reports `exists: False` with empty
lists and searches empty — it never raises. That is
this VM's own state: the campaign's sparse checkout
carries no library, and the binding reads that absence
as absence.

## 4. Chunks — the ETL's own chunking

`chunks()` chunks every kept readable file through
`ReferenceDataETL._load_and_chunk` at the ETL's own
`CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` (the ETL
class is instantiated only to borrow that pure method;
its DB path is never opened and `build()` is never
called). A binding chunk is therefore byte-for-byte the
chunk ingestion would land. On the synthetic library:
3 files → 6 chunks, identical per file to the ETL's.

Rules around the chunks, all named in `chunk_report()`:

- **Duplicates:** a readable file whose SHA-256 equals
  an earlier file's (sorted relative-path order) is a
  duplicate — inventoried with its keeper named, and
  never chunked twice.
- **No-chunk files:** a readable, kept file that yields
  no chunks (empty, unreadable) is listed in
  `files_no_chunks`, not silently dropped.
- **Identity:** a chunk's id is deterministic —
  `<source_file>#<chunk_index>` — so `get_chunk(id)`
  and citations are stable across runs and machines.

## 5. Search — a cited bundle, by a stated rule

`search(query, top_k=10)` returns a
`MemoryActivationBundle` in the forge's own reference
shape. Each hit is an `ActivatedRecord` of type
`reference_chunk` whose raw dict carries exactly the
forge SQL pass's fields (`chunk_id`, `source_file`,
`title`, `content` truncated to 500 chars as the forge
truncates, `tags`, `source`, `record_type`), so TF-048
derives its citation payload by construction: marker
`ref:<source_file>#<chunk_index>`, locator the source
file — TF-042's ruling that the file *is* a reference
chunk's provenance.

The scoring rule is stated, not hidden: the share of
the query's distinct key terms (the forge's own
`_extract_key_terms`) present whole-word in the chunk's
title + content + tags. A chunk matching both terms of
a two-term query scores 1.0; one of two, 0.5 (both
pinned). Only chunks matching at least one term are
returned; ties break by (source_file, chunk_index), so
a search is reproducible run to run (pinned). No match,
an empty query, and a stopword-only query all return an
empty bundle — never a fabricated hit. Bundle
bookkeeping is the forge's (total == records,
confidence == mean score; checked with TF-041's
`bundle_problems`).

## 6. D-REF-1 — fixed in scope (write side)

`ReferenceDataETL.build()` now calls
`rebuild_fts_indexes(self.db_path)` before returning.
Proven behaviourally: after a plain ETL build of the
synthetic library, `KnowledgeForge.sql_retrieve("ferry")`
returns the reference rows with no manual rebuild, and
`retrieve(..., min_score=0)` bundles them. The build's
own stats are unchanged ({files: 3, chunks: 6,
skipped: 0}). The rebuild touches all three FTS tables,
as the schema's helper does — harmless for the other
sources, and the same call the embeddings step already
made.

## 7. Docket

- **D-REF-1 added + FIXED by this slice** (see §1/§6).
  Totals: 54 entries — 40 OPEN / 1 OPEN-ENVIRONMENT /
  11 FIXED / 2 CLOSED-BY-VERDICT.
- **D-DATA-4 narrowed:** on the read side, skipping is
  now by classified, counted rule and duplicates are
  deduped by hash before chunking. The ingestion side
  (the ETL's own silent `continue` and double-chunking)
  remains TF-118's share, unchanged.

## 8. Boundaries, stated

- This is the READ side. Writing the library into the
  knowledge DB remains the ETL's — TF-118 owns its
  reproducible, checksummed ingestion (and the
  ingestion-side halves of D-DATA-2 / D-DATA-4).
  Profile-sized subsets of the library are TF-110's.
- Search here is a stated term-overlap rule, not the
  forge's bm25/vector ranking. Its fixtures prove
  identity (the right chunk, cited, reproducible), not
  ranking quality — Part VII owns ranking research.
- Proof grade is FIXTURE (the TF-017 synthetic library
  and synthetic temp libraries, on this VM). The binding
  over the real canon library is **NOT PROVEN**: the
  real library is not in this sparse checkout and no
  canon file was opened by this slice — never claimed.
