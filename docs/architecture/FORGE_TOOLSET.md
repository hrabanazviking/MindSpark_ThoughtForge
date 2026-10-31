# TF-057 — Knowledge Forge UI-Shape: the forge_memory Toolset

*Part III, slice TF-057. Code:
`src/thoughtforge/knowledge/forge_toolset.py` (the operations
as data, the four legs, and `forge_toolset_problems()`
re-deriving every claim below from the live tree), wiring in
`forge_memory.py` (a `--memory-dir` group option and the
`add` / `inspect` / `prune` / `export` commands — thin shells
printing the toolset's JSON reports). Tests:
`tests/test_tf057_forge_toolset.py`.*

## 1. The question and the measured starting state

The roadmap asks: are forge_memory's operations
(add / inspect / prune / export) a coherent toolset?

Measured this tick, by direct probe of the CLI:

- **forge_memory served only the knowledge database.** Its
  nine commands (init, wikidata, conceptnet, geonames,
  dbpedia, reference, embeddings, status, all) all act on
  the knowledge DB — the ETL estate. Not one command, and
  no library module, operated on the *memory store*: the
  store a user actually lives in had no operator toolset.
  Even `status` reports the DB's tables, not the store.
- **The store's operations were scattered, one per slice.**
  Adding was the store's raw `append_*` gates (TF-042),
  pruning was `MemoryLifecycle.prune` (TF-045) or the
  core's `prune_memory`, exporting was TF-052's bundle,
  and inspecting a memory store did not exist at all.
- **Two slices had already assigned the missing legs
  here.** TF-042: "TF-057 stamps manual entries in the
  forge_memory toolset" — `MEMORY_PROVENANCE` carried
  `manual_entry`, unused by any code. TF-045 named "a
  forge_memory.py prune command" as TF-057's. Both
  assignments land in this slice.

## 2. The toolset, as data

`OPERATIONS` — exactly four operations, in order:

| Operation | Writes the store? | What it does |
|---|---|---|
| add | yes | one record by hand, stamped `manual_entry`, through the TF-042 gate |
| inspect | no | whole-store report, filtered record list, or one record by id |
| prune | yes | one lifecycle pass in a named mode (TF-045), optional profile caps (TF-056) |
| export | no (writes a file outside the store) | the whole store as one verified TF-052 bundle |

The knowledge-DB commands are not in this table and are
untouched: the two estates keep their own tools, stated,
not merged. `--db` still governs only the DB commands;
`--memory-dir` governs only the four toolset commands.

## 3. Add: manual entries, stamped

`add_record(store, kind, ...)` covers exactly TF-042's
four stamped memory kinds (`ADD_KINDS`: user_preference,
user_fact, episode, response_pattern — pinned against the
contract's stamped set by the checker). Rules:

- **The stamp is by construction.** Every landed record
  carries `source="manual_entry"`; there is no source
  parameter to override it. TF-042's assignment, landed.
- **The store's gate decides validity.** Records land
  through `MemoryStore`'s append methods, so TF-042's
  ingestion contract validates first: an invalid record
  (empty fact, over-cap summary, out-of-range number)
  is refused by the gate's ValueError naming every
  problem, and nothing is written.
- **No silent duplicates.** An explicit `record_id` that
  already lives anywhere inspect can see (any append
  store, the thread state, the personality core) is
  refused by name before anything is written. Without an
  explicit id, the store's own factories mint one.
- **A landed add is findable.** The tag index is rebuilt
  and the id pushed onto the recent queue — the same
  derived surfaces a writeback record reaches.
- Kind defaults mirror the record dataclasses (preference
  importance/confidence 0.5/0.5, fact 0.5/0.6, episode
  0.4/0.5 with quality 0.5, pattern quality 0.6).

## 4. Inspect: read-only, proven

`inspect_store` returns one JSON-safe report: live counts
per kind and bytes per file (TF-056's `measure_store` —
per-kind bytes are *not* claimed; facts and preferences
share one file), counts by status, personality-core and
thread-state presence, recent-queue length, tag-index
size, and — only when a profile is named — TF-056's
budget report for the same store. `inspect_records`
lists record summaries (id, kind, summary, tags, status,
source) filtered by kind / tag / status / text, and
refuses an unknown kind filter by name rather than
reading a typo as an empty store. `find_record` resolves
one id across every surface, a miss being data
(`found: False`), not an exception. Inspect writes
nothing — pinned by a byte-for-byte directory comparison
across all three legs.

## 5. Prune: the lifecycle's act, reported

`prune_store` runs one `MemoryLifecycle` pass in the
named mode, with an optional TF-056 `profile_id` swapping
in that profile's caps, and reports before-counts, the
lifecycle's own counts, and after-counts. An unknown mode
is refused by the lifecycle's up-front ValueError
(TF-045) before any store is rewritten. Boundaries,
stated: prune is score-driven retirement *by rule* —
protected records are kept — it is not user-ordered
forgetting (TF-051 owns that, unchanged), and a profile
here only changes the caps the same prune enforces
(TF-056's construction), it is not a second eviction
mechanism.

## 6. Export: TF-052's bundle, verified after writing

`export_store_bundle` writes the whole store with
TF-052's `write_bundle` to the named path, then
re-verifies the bundle with `verify_bundle`. The report
names the path, format, version, per-section counts, and
`verified` — a bundle that failed verification would be
reported with its problems named, never as a clean
export. The store itself is only read.

## 7. Boundaries

- One store at a time (one directory = one persona,
  TF-055). No knowledge-DB operation, no cross-store
  operation: moving a whole store remains TF-052
  import's explicit operator act, not a toolset verb.
- Personality core and thread state are inspect-only:
  identity is versioned manually (TF-042) and the
  working record is the conversation's, so `add` covers
  exactly the four append kinds.
- Proof grade is FIXTURE (synthetic stores + the CLI
  runner on this VM). The toolset over a real, lived-in
  store is NOT PROVEN — no real store exists on this
  machine (TF-041) — and is never claimed.
- Test-isolation note (found + fixed in this slice):
  `forge_memory.py` configures logging at its top level,
  as a CLI entry point does; a test that *imports* it at
  collection time leaves a RotatingFileHandler on the
  root logger and breaks TF-038's handler-count test.
  The TF-057 tests therefore load it lazily, at execution
  time. The CLI's own behaviour is unchanged and correct;
  the defect was in this slice's first test draft.
