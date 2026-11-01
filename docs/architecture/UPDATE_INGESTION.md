# ThoughtForge Architecture — Update Ingestion (TF-114)

Update ingestion: re-running a source updates facts by
the lifecycle rules instead of duplicating them. Part
VI's tenth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The rules live as verifiable data in
`src/thoughtforge/etl/update_ingestion.py`
(`UPDATE_RULES`, `UpdateLedger`,
`retire_unpresented_statements()`,
`retire_unpresented_chunks()`, checked by
`update_problems()` and `update_live_problems()`);
the enforcement is in `src/thoughtforge/etl/wikidata.py`
and `src/thoughtforge/etl/sources.py`; TF-109's
`dedup_merge.py` was updated in the open (its
within-table conflict rule now states the lifecycle
supersede instead of the placeholder that named this
slice), as was TF-107's `wikidata_pipeline.py` rule
text and TF-105's living map. The tests are
`tests/test_tf114_update_ingestion.py`. This slice
**adds and fixes D-UPDATE-1, D-UPDATE-2 and
D-UPDATE-3**.

## 1. Measured start

Probed live this tick, before any change, with
synthetic fixtures through the real builders:

- **Changed wikidata statements appended beside the
  old fact (D-UPDATE-1).** Q1 built with statements
  P31 → Q111 and P17 → Q900, then re-built
  (`resume=False`) with P31 → Q999 and the P17 claim
  removed: the store held Q111 *and* Q999 live for the
  same subject and property — an unflagged
  contradiction in TF-050's sense — and the removed
  P17 statement stayed. Statements only ever grew;
  TF-107's idempotence covered exact duplicates only.
- **Disappeared chunks stayed stale (D-UPDATE-2).** A
  reference file built with two chunks, then edited
  to one changed chunk, kept the vanished second
  chunk in the store, still searchable through the
  rebuilt FTS index. A file deleted outright kept
  every one of its chunks.
- **No lifecycle statement, no update record
  (D-UPDATE-3).** Row tables resolved a changed fact
  by TF-109's filed placeholder ("the incoming row
  wins — TF-114 owns replacing this default"), and
  nothing anywhere stated what a re-run should do
  with an unchanged fact, a changed fact, or a fact
  that disappears; no ledger named a supersede or a
  retirement as such. Measured in the same probes:
  the GeoNames place "Beta", DBpedia's B, wikidata's
  Q2 and the deleted file's chunks all survived their
  re-runs — by accident, not by rule.

## 2. The lifecycle rules

Justified from the repo's own precedents: TF-050's
contradiction rule (*surfaced, scored, resolved by
the stated rule, never silently overwritten*; a
decisive new claim supersedes the stored one, which
is archived out of the live store; a duplicate is
reinforced, not copied) and TF-045's lifecycle
(*retirement is physical; fact retirement is an
operator's act, never a side effect*).

- **Unchanged — duplicate, not rewritten.** TF-109's
  rule stands unchanged.
- **Changed within the same source — supersede.** The
  incoming value stands in place of the stored one;
  the stored value is preserved in the ledgers
  (TF-109's per-field conflict detail, truncated to
  120 chars; this slice's `UpdateLedger` where the
  update is structural) *before* the write. The ETL
  tables carry no status column and no archive table
  exists (TF-105's schema facts stand — no schema
  table is added), so the superseded value leaves the
  live row and its record is the ledger: TF-050's
  supersede in the form this store can honour.
  A thinner re-run still merges complementarily —
  an empty incoming field never erases (TF-109).
- **Statements supersede by presented set.** A
  wikidata entity the re-run presents is that
  source's newest complete statement of the entity.
  Before the build's final commit, stored statements
  (source `wikidata`) for presented subjects that the
  run did not re-present are deleted and ledgered:
  `statement_superseded` when another tuple for the
  same property was presented (the fact changed —
  the ledger names the superseding value),
  `statement_retired` when none was (the fact
  disappeared from a presented entity). Every
  presented tuple stands, so multi-valued facts are
  never collapsed. The judgement runs in SQL against
  connection-local TEMP presented-set tables the
  builder fills as it streams — no dump is held in
  memory and no schema changes.
- **Only presented subjects are judged.** A subject
  the run does not stream — beyond a checkpoint
  resume point (TF-107), excluded by a filter, or
  past a limit — is not judged at all. A
  completed-checkpoint return presents nothing and
  retires nothing. Retirement runs inside the build
  before the final commit, so a failed build rolls
  it back with the rest of the run's work.
- **A disappeared chunk of a presented file is
  retired.** For each reference file the builder
  successfully reads (including one that now produces
  no chunks at all), stored chunks whose index the
  new presentation does not produce are deleted in
  the same per-file commit — with any
  `reference_embeddings` row keyed on them — and
  ledgered `chunk_retired`; the build's closing FTS
  rebuild (TF-058) drops them from search in the
  same build.
- **A unit absent from the re-run is KEPT.** An
  entity, place, uri or whole file the re-run does
  not present is left untouched, however long it has
  been absent. The builders accept partial inputs by
  design (a limit, a filter, a country subset, one
  file of a dump), so absence cannot be told from
  partiality; deleting on absence would let a
  partial build destroy the store. Whole-unit
  retirement stays an operator's act (TF-045).
- **Cross-source rows are never updated by a
  re-run.** TF-109 stands: overlap across tables is
  a finding, never a merge, and one source's re-run
  touches only its own rows (statements judged under
  `source = 'wikidata'`, chunks under their own
  `source_file`).

## 3. The update ledger

Each wikidata and reference build holds a fresh
`UpdateLedger` beside its TF-108 gate and TF-109
merge ledgers, recording every supersede and
retirement the build performed — table, key (the
full stored tuple for a statement, truncated in its
detail), outcome and resolution — as data
(`as_data()`, format `thoughtforge-update-ingestion/1`).
Builder stats shapes do not change (TF-105);
TF-115 owns reporting the ledger per run.

## 4. Filed limits

- **Supersede is judged per presented unit, not per
  dump.** A dump that presents an entity with a
  property's values *reduced* retires the unpresented
  ones by the stated rule — that is the rule working,
  and it is why only complete presentations of an
  entity should be re-run against a live store.
- **No archive table.** Superseded/retired tuples
  leave the live store physically; their record is
  the in-memory per-build ledger (and, for fields,
  TF-109's merge ledger). A durable update history
  table would change TF-105's schema facts and is
  not smuggled in here; TF-115 owns per-run reporting
  of the ledger as it stands.
- **GeoNames / DBpedia / ConceptNet carry no
  update ledger of their own yet:** their updates
  ride TF-109's merge ledger (a changed row is a
  ledgered supersede; ConceptNet resolves by
  strength, unchanged), and their statement/chunk
  shapes have no presented-set retirement to perform
  — a GeoNames row and a DBpedia row are whole-unit
  records, and whole units are kept when absent by
  the stated rule.

## 5. Docket

- **D-UPDATE-1 — added + FIXED.** Changed wikidata
  statements appended beside the old fact and removed
  claims never left (§1): statements now supersede
  by presented set; the probed re-run leaves exactly
  the presented set live, with the changed tuple
  ledgered `statement_superseded` and the removed
  one `statement_retired`.
- **D-UPDATE-2 — added + FIXED.** Disappeared chunks
  of a re-presented reference file stayed stale and
  searchable (§1): they are now retired in the same
  commit, with their embedding rows, and the FTS
  rebuild drops them from search; the live checker
  and tests prove the shrunk file's store holds only
  the presented chunk.
- **D-UPDATE-3 — added + FIXED.** No lifecycle
  statement and no update record (§1): the rules are
  filed as data in `etl/update_ingestion.py`,
  TF-109's placeholder rule was replaced in the open,
  and the builders' `update_ledger` names every
  supersede and retirement a re-run performs.

TF-016 totals after this slice: 112 entries —
42 OPEN / 1 OPEN-ENVIRONMENT / 66 FIXED /
3 CLOSED-BY-VERDICT.

**Proof grade: FIXTURE** — synthetic re-runs through
the real builders on this VM. No real dump has been
re-run, and no at-scale update rate is claimed.
TF-115 owns ingestion metrics (including reporting
this slice's update ledger per run).
