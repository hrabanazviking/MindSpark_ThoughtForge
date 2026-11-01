# ThoughtForge Architecture — Dedup & Merge (TF-109)

Dedup & merge rules for overlapping sources (consumes
TF-050's contradiction rule). Part VI's fifth slice.
Built 2026-11-01 on branch `yrsa/thoughtforge-road-build`.

The rules live as verifiable data in
`src/thoughtforge/etl/dedup_merge.py` (`TABLE_IDENTITIES`,
`MERGE_RULES`, `decide_merge()`, `MergeLedger`,
`apply_row_merge()`, `merge_conceptnet_edge()`,
`find_overlaps()`, checked by `merge_problems()` and
`merge_live_problems()`); the enforcement is in
`src/thoughtforge/etl/wikidata.py` and
`src/thoughtforge/etl/sources.py`; the tests are
`tests/test_tf109_dedup_merge.py`. This slice **adds and
fixes D-MERGE-1, D-MERGE-2, D-MERGE-3 and D-MERGE-4** and
**narrows D-DATA-2**. TF-105's living map was updated in
the open (all five builders' write modes + notes).

## 1. Measured start

Probed live this tick, before any change, with synthetic
overlapping records through the real builders:

- **A thinner re-run erased fields (D-MERGE-1).** A
  wikidata entity built with a description, aliases and
  sitelink popularity, then re-built (`resume=False`)
  with a changed label and no description, came back
  with `description_en` NULL, `aliases_en` NULL and
  `popularity_score` 0.0 — `INSERT OR REPLACE` replaced
  the whole row with the thinner one, and nothing
  recorded the loss.
- **ConceptNet had no semantic identity (D-MERGE-2).**
  The same edge (lantern / IsA / light) asserted under
  `/a/1` (weight 2.0) and `/a/2` (weight 5.0) landed as
  two rows — the uri was the only identity. The same uri
  re-asserted at weight 9.0 was dropped by
  `INSERT OR IGNORE`, keeping 2.0. The stats counted
  all three lines "inserted" (3 counted, 2 rows, the
  strongest assertion lost).
- **Every other collision was a silent overwrite
  (D-MERGE-3).** A GeoNames place re-built with a new
  name and population, and a DBpedia entity re-built
  with a new abstract, replaced their rows whole with
  no record of old value, new value, or the collision
  itself. Labels and reference chunks replaced the same
  silent way. TF-050's rule — *surfaced, scored,
  resolved by the stated rule, never silently
  overwritten* — governed the profile store and had no
  counterpart on the ingestion side.
- **Cross-source overlap was invisible (D-MERGE-4).**
  The same synthetic town from wikidata (Q1,
  "Lantern Town"), GeoNames (109001, "Lantern Town")
  and DBpedia (…/Lantern_Town, "Lantern Town") sat in
  three tables unlinked, and no function in the tree
  could name the overlap.

## 2. Identities

| Table | Key | Judged value fields |
|---|---|---|
| entities | `qid` | label, description, aliases, instance_of, popularity |
| labels | `(qid, language)` | label |
| statements | the whole stored tuple | — (identity is the tuple, TF-107) |
| conceptnet_edges | semantic: `(concept1, relation, concept2, language)`; uri is provenance | weight |
| geonames_places | `geonames_id` | name, ascii name, coordinates, feature class/code, country, population, elevation, timezone, admin codes |
| dbpedia_entities | `uri` | label, abstract, types |
| reference_chunks | `(source_file, chunk_index)` | title, content, tags, word count |

## 3. The merge rules

- **Exact duplicate — not rewritten.** A collision in
  which no value field differs is recorded in the
  builder's `MergeLedger` as `duplicate` and the stored
  row is left untouched.
- **Complementary — union, never erase.** Differences
  only where one side is empty (None or "") merge to
  the union: a stored gap is filled, and an empty
  incoming field never erases a stored value
  (`merged_complementary`). Zero is a value, not a gap.
- **Conflict — surfaced, then resolved by the stated
  rule.** A field with two different non-empty values
  is a conflict in TF-050's sense. It is recorded in
  the ledger with the field and both values (truncated
  to 120 chars, so the ledger never becomes a second
  copy of the store) *before* anything writes.
  Within one source's own table the incoming row wins
  (`conflict_overwrite`) — it is that source's newest
  statement of itself. TF-114 owns replacing this
  default with lifecycle update rules, in the open.
- **ConceptNet — strength wins.** Where two provenances
  assert the same edge (the assertions file and
  `build_from_reference`), the edge's own weight is its
  strength (TF-050's scoring in miniature): the
  stronger assertion stands with its uri and weight
  (`conflict_stronger_wins` / `conflict_first_stands`),
  and an exact tie keeps the first row.
- **The ledger.** Each builder holds a fresh
  `MergeLedger` per build beside its TF-108
  `SchemaGate`, recording every collision with table,
  key, outcome and per-field detail, broken down by
  outcome and by table. New rows are not collisions —
  they are the stats' job, as the gate ledger records
  only refusals. Builder stats shapes do not change
  (TF-105); where an accepted record's write was a
  duplicate or a lost conflict, the store's rows and
  the ledger are the truth and the stats the count of
  records accepted — TF-115 owns reporting that
  difference per run.

## 4. Cross-source overlap

`find_overlaps(db_path)` groups the store's
name-carrying rows — entity labels, DBpedia labels,
GeoNames names, ConceptNet concept texts — by a derived
normalised name (casefolded; underscores/hyphens to
spaces; punctuation stripped; whitespace collapsed —
stored texts are never altered) and returns every group
with members from at least two tables, as data: the
normalised name plus each member's table, key, source
and label. It is a **finding, never a merge**: dedup
does not merge, rewrite or delete across tables,
because the sources keep their provenance (TF-106) and
settling a cross-source disagreement is a reader's
judgement, not ingestion's. TF-110 (subset craft) and
TF-112 (dry-run reports) are the filed first consumers.

## 5. Filed limits

- **Same-source changed facts still land.** The
  within-table conflict default (incoming wins, now
  ledgered) is a dedup rule, not an update policy:
  TF-114 owns re-run update semantics (lifecycle rules
  instead of replacement) and must update
  `dedup_merge.py` in the open when it changes the
  default.
- **Overlap matching is name-shaped only.** Two sources
  naming one thing differently (a QID's label vs a
  place's local name) form no group, and two different
  things sharing a name form a group a reader must
  split: the finder reports candidates, never verdicts.
  No schema table was added for groups — the finding
  is recomputed from the store, so TF-105's filed
  schema facts stand.
- **Embeddings are not merged here.** They are
  regenerated per key by TF-111's ingestion discipline;
  the subset builder copies by identity and is
  TF-110's craft.
- **D-DATA-2's remainder.** The duplicate *files*
  inside `data/` (37 duplicate blob hashes in TF-007's
  census) are Volmarr's canon: nothing is moved or
  deleted by this campaign, and that share stands OPEN
  for his word. This slice fixes the store-level share
  — the same fact duplicated inside the knowledge
  store.

## 6. Docket

- **D-MERGE-1 — added + FIXED.** A thinner re-run
  erased stored fields (§1): complementary merges now
  union the rows, and the live checker + tests prove
  the re-run entity keeps its description and aliases.
- **D-MERGE-2 — added + FIXED.** ConceptNet semantic
  duplicates and IGNOREd stronger assertions (§1):
  edges dedup by semantic key and resolve by weight;
  the probed three lines now land as one row at the
  strongest weight.
- **D-MERGE-3 — added + FIXED.** Silent overwrites
  (§1): every collision on every merged table is
  ledgered with both values before resolution.
- **D-MERGE-4 — added + FIXED.** Cross-source overlap
  invisible (§1): `find_overlaps` names the group; the
  live checker proves the three-source town is one
  group of three tables, and that finding it changes
  no stored row.
- **D-DATA-2 — narrowed.** Store-level duplication is
  fixed here; the duplicate canon files stand OPEN
  for Volmarr's word (nothing moved or deleted).

**Proof grade: FIXTURE** — synthetic overlapping
records through the real builders on this VM. No real
dump's overlap or duplication rate has been measured,
and none is claimed. TF-110 owns subset craft;
TF-112's dry-run is the merge ledger's first reporting
consumer; TF-114 owns update ingestion; TF-115 owns
per-run merge reporting.
