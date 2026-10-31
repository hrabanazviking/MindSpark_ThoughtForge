# TF-052 — Store Portability: A Whole Store as a Documented Bundle

*Part III, slice TF-052. Code:
`src/thoughtforge/knowledge/portability.py` (the bundle
shape as data, `export_store()` / `write_bundle()` /
`read_bundle()`, `verify_bundle()`, `import_store()`, and
`portability_problems()` re-deriving every claim below
from the live tree, including a live round trip). Tests:
`tests/test_tf052_portability.py`.*

## 1. The question and the measured starting state

The roadmap asks: can a whole memory store move — to
another machine, another runtime (the Hermes-portability
law applied to memory), or into safekeeping — as one
documented, verifiable bundle, instead of as a folder
someone copied and hoped about? Measured before this
slice, by direct probe:

- **No portability API existed.** There was no
  portability module in `thoughtforge.knowledge`, and
  `MemoryStore` carried no export, import, or bundle
  method at all.
- **The only move was copying the directory by hand** —
  undocumented (which files are the store? which are
  derived from it?), unvalidated (a truncated or edited
  copy lands silently), and unverifiable (no counts, no
  checksums — nothing could prove the copy *is* the
  store).

## 2. The bundle

One JSON document, `format: "thoughtforge-store-bundle"`,
`version: 1`:

| section | content |
|---|---|
| `personality_core` | the identity record's `to_dict()`, or `null` |
| `profile_records` | every preference + fact dict, verbatim |
| `episodes` | every episode dict, verbatim |
| `patterns` | every response-pattern dict, verbatim |
| `thread_state` | the working record's dict, or `null` |
| `recent_queue` | the recency ordering (record ids) |

Plus a **manifest**: per-section record counts, a
SHA-256 checksum per section over its canonical JSON
(sorted keys, compact separators), and `total_records`.
`exported_at` is informational — nothing in verification
depends on it.

Not carried, by named ruling in `EXCLUDED_SURFACES`:

- **Tag index** (`tag_index.json`) — derived data
  (TF-051). It is rebuilt from the landed records on
  import, never carried as authority; a bundle whose
  index disagreed with its records would have nothing
  to disagree about.
- **Knowledge DB** (SQLite) — the read-only reference
  estate (Wikidata / ConceptNet / reference chunks). It
  holds no memory-store records and is rebuilt by the
  ETL wing, not carried between stores.

## 3. The gate: `verify_bundle(bundle)`

Verification re-derives everything the bundle claims,
before anything is written anywhere:

- the format string and version are exactly this
  bundle's;
- the section set is exactly `BUNDLE_SECTIONS`;
- every manifest count and checksum is recomputed from
  the sections and must match — a single edited byte in
  a record fails the checksum and names its section;
- every record passes TF-042's ingestion contract
  (`record_problems`) — an invalid record is named, with
  its contract problems;
- no record id appears twice anywhere in the bundle —
  one store may not hold two records under one identity.

`verify_bundle` is pure: it reads nothing and writes
nothing. `read_bundle` deliberately returns a malformed
file's JSON as data for this gate to refuse, rather than
crashing on it.

## 4. The landing: `import_store(store, bundle, mode=...)`

The mode is **required** — there is no default, because
the two modes differ in what may be destroyed:

- **`replace`** — the bundle becomes the whole store.
  Existing records are rewritten out; the bundle's
  records land through the store's own gated append
  paths (after the file is emptied, so even a bundled
  record carrying an `archived` status lands exactly as
  bundled). A bundle with no personality core or no
  thread state removes the target's.
- **`merge`** — the bundle's records are added to what
  the target already holds, and **nothing is silently
  overwritten**: a record whose id already lives in the
  target is skipped and named on the report as a
  collision (`skipped_collisions`).

Either way, **validation is whole-bundle first**: any
problem in `verify_bundle` refuses the import entire —
status `refused`, every problem named, and the target
store left byte-for-byte untouched. There are no partial
imports.

**Provenance.** Landed preferences, facts, episodes, and
patterns are stamped `source="import"` — TF-042's
vocabulary word for a record carried in from a store
bundle, and the stamp TF-050's contradiction authority
table already prices at 0.5. Every other field lands
verbatim. The personality core and the thread state
carry no stamp, per TF-042's ruling that identity and
working state are not ingested claims. The round-trip
rule, stated exactly: export → import → export is
stable from the second export on; the first import's
only change to any record is that stamp.

**The queue and the index after landing.** The recent
queue is carried as data because an ordering cannot be
re-derived — on import it is filtered to ids that
actually landed (in `merge`, ids already present count),
and entries naming nothing landed are dropped and
counted on the report (`queue_dropped`), never left
dangling. The tag index is then rebuilt from the landed
store.

## 5. Stated boundaries

- **A bundle is plain JSON and exactly as private as
  the store it came from.** It carries the store's
  contents whole — persona/canon data included. This
  leg never publishes a bundle anywhere; where a bundle
  file may live is its owner's decision, the same as
  the store's.
- **`replace` is destructive by its stated meaning** —
  that is why it must be named explicitly, and why the
  refused-import guarantee (target untouched) matters
  most on that path.
- **Backups made outside this leg are not bundles** —
  a copied directory gains counts and checksums only
  by being exported through here.

## 6. Proof

Fixture grade: synthetic stores through export, the
gate, and both import modes (full round trip verbatim
but for the stamp; second round trip exactly stable;
checksum tampering caught by section; a contract-invalid
record refused with the target byte-untouched; duplicate
ids refused; wrong format/version refused; replace
removing what the bundle lacks; merge collisions skipped
and named, never overwritten; the queue filtered and
its drops counted; the tag index rebuilt, not carried;
an empty store round-tripping; bundle files through
`write_bundle`/`read_bundle`). `portability_problems()
== []` re-derives the shape from the live tree and runs
a live round trip. Portability over a real, lived-in
store is NOT PROVEN — no real store exists on this
machine (TF-041).
