# ThoughtForge Architecture — Ingestion Contract: What a "Fact" Is (TF-042)

Ingestion contract: what a "fact" is in ThoughtForge (shape,
provenance, confidence) — one schema, validated. Part III's
second slice. Built 2026-10-31 on branch
`yrsa/thoughtforge-road-build`.

The contract lives as verifiable data in
`src/thoughtforge/knowledge/ingestion_contract.py`
(`SCHEMA_PARTS`, `FACT_KINDS`, `DB_PROVENANCE`,
`MEMORY_PROVENANCE`, `record_problems()`,
`db_row_problems()`, `writeback_fact_problems()`,
`contract_problems()`); the tests are
`tests/test_tf042_ingestion_contract.py`. This slice also
**wires validation into the memory store's ingestion gates**
(§5).

## 1. One schema, six parts

Before this slice the tree had shapes but no contract. Six
persisted memory record types and six knowledge-DB tables hold
claims, and nothing validated any of them at the point of
ingestion: a record with `confidence=999`, an empty summary, a
summary ten times its own stated cap, or a status outside the
closed vocabulary was appended verbatim — and the scoring maths
that later consumed it silently assumed values in [0, 1].

A **fact** in ThoughtForge, in either store, is a persisted
claim carrying six parts:

| Part | Memory store | Knowledge DB |
|---|---|---|
| identity | `record_id` | `qid` / triple keys / `geonames_id` / `uri` / (`source_file`, `chunk_index`) |
| kind | `record_type` (5 typed + personality core) | the table itself |
| content | summary (+ `fact_value` / `category` / `pattern_shape`) | label / triple / abstract / chunk text |
| provenance | **absent — see §3** | `source` column (+ `refs` on statements) |
| confidence | `confidence` in [0, 1], per-kind default | **no true confidence — see §4** |
| lifecycle | status / retention / stale horizon / timestamps | `ingestion_log` at run level |

## 2. The twelve kinds

**Memory kinds** (dataclass, construction confidence default):
personality_core (none — identity is versioned manually),
user_preference (0.5), user_fact (0.6), episode (0.5),
response_pattern (0.6), active_thread_state (none — its
`priority`, default 0.8, is urgency, not trust).

**DB kinds** (table): entities, statements, conceptnet_edges,
geonames_places, dbpedia_entities, reference_chunks.

Content rules are per kind and match how the estate actually
constructs records: a user_fact needs a non-empty summary *or*
a non-empty `fact_value`; a preference needs category *and*
summary; an episode needs a summary; a pattern needs a summary
*or* a pattern shape (the estate's own round-trip records carry
shape alone); personality core and thread state are
identity-only by ruling — a blank persona is a valid identity,
and a thread snapshot may legitimately stand empty between
topics. Summary caps are the caps `models.py` itself states in
prose: preference 160, fact 180, episode 240 chars.

## 3. Provenance: present in the DB, absent in memory

The knowledge DB's provenance vocabulary is closed and
*measured*: `DB_PROVENANCE` = {wikidata, conceptnet, geonames,
dbpedia, reference} is exactly the set of `source` column
defaults declared in `etl/schema.py`, and exactly the forge's
SQL source names plus dbpedia. `db_row_problems()` enforces it
per row.

**The headline finding: no persisted memory record can say
where it came from.** None of the six record types has a
`source` or `provenance` field; a user_fact the user stated
verbatim and one a model inferred are indistinguishable on
disk. This slice does not paper over that: it defines the
vocabulary the stamp will use — `MEMORY_PROVENANCE` =
{user_stated, conversation_writeback, manual_entry, import,
synthetic_fixture} — pins the gap in `contract_problems()` (the
check fires the day a record gains the field, demanding the
contract be updated with evidence), and assigns the stamping:
**TF-047** stamps facts when it wires conversation → memory
writeback, **TF-057** stamps manual entries in the forge_memory
toolset, **TF-052** stamps imports. Until then, provenance for
memory is honestly NOT CARRIED, never implied.

**Landed since (all three assignments now stand):** TF-047's
writeback stamps `conversation_writeback` (and honours
`user_stated` where the entry carries it), TF-052's import
stamps `import`, and TF-057's forge_memory toolset stamps
every manual add `manual_entry` by construction — see
`FORGE_TOOLSET.md`. The four stamped kinds carry the field;
personality core and thread state still carry none, by the
rulings above.

## 4. Confidence: what the numbers actually mean

A confidence is a probability-like rating in [0, 1] — 0 = no
trust, 1 = certain. The same range binds its kin on memory
records: importance, quality, priority, recency_weight. The
validator rejects out-of-range, non-numeric, and non-finite
(NaN/inf) values by field name, and rejects negative counters,
negative stale horizons, statuses/retentions outside the
closed vocabularies (TF-041 K8), non-list-of-string tag fields,
and unparseable timestamps.

On the DB side **no kind carries a true confidence**, and the
contract refuses to pretend otherwise: entities carry
`popularity_score` — a sitelink-count proxy its estimator caps
at 1.0 — statements carry `refs` (provenance evidence as JSON
text, no number), GeoNames and DBpedia rows carry nothing, and
ConceptNet `weight` is an **unbounded strength, not a
probability**: a weight of 12.0 validates, a negative weight
does not, and the forge caps weight's contribution at fusion
time (TF-041 §5). Reference chunks carry no confidence either;
their `source_file` is both identity and provenance.

## 5. Validation, wired at the ingestion gates

`MemoryStore`'s five ingestion gates — `append_profile_record`,
`append_episode`, `append_pattern`, `save_personality_core`,
`save_thread_state` — now validate through the store's one
`_validate_record` helper (which calls `record_problems()`)
and raise `ValueError` naming every problem **before anything
is written**; a refused append leaves no file behind (pinned in
the tests). The whole pre-existing estate — phase suite,
lifecycle, TF-017 fixtures, TF-041 stores — passes the gates
unchanged, which is itself measured in the full suite.

Two deliberate non-gates, ruled and recorded:

- **Load paths do not validate.** A record written before the
  contract (or by hand) still loads — TF-041's K13 tolerance
  stands — and `record_problems()` flags it by name when
  audited. A stored invalid record is evidence, not an excuse
  to silently drop memory.
- **`rewrite_*` methods do not validate.** They are the
  prune/deletion mechanism and must be able to carry legacy
  records out of the store; gating them would strand exactly
  the records pruning exists to remove.

Unknown keys are not problems anywhere in the contract:
`from_dict` ignores them by design (forward compatibility,
TF-041 K12), so the validator does too.

## 6. The writeback shape (TF-047's raw material)

`WritebackRecord.new_facts` is a list of untyped dicts
(TF-025 §7 left them dict-typed for this slice and TF-047).
`writeback_fact_problems()` states what one such dict must
satisfy before TF-047 may turn it into a `UserFactRecord`:
content (summary ≤180 or a non-empty `fact_value`), tags as a
list of strings, confidence/importance in [0, 1] when present,
and — stricter than stored records, because writeback is *new*
ingestion and the extractor still knows the answer — a
**required provenance stamp**: key `source`, a member of
`MEMORY_PROVENANCE`. Measured consequence, stated plainly:
TF-038's trace fixture dict `{"fact": "ferry"}` is **not** a
valid writeback fact under this contract (no contract content,
no stamp) — it is a trace shape and was never an ingestion
candidate; TF-047's extractor must emit the contract shape.

## 7. What this slice does not do

- It does not add the provenance field to memory records —
  that is a stored-shape change owned by TF-047's wiring,
  which will flip the pinned gap check with its evidence.
- It does not validate ETL inserts in the pipeline — TF-108
  owns schema validation at the ETL gate; `db_row_problems()`
  is the schema that gate will enforce, exercised here on
  synthetic rows.
- It does not retune any confidence default or scoring weight
  (TF-044) and does not judge retrieval quality (TF-043).

Proof grade: FIXTURE — synthetic records, DB rows, and stores
on this VM, validated through the real store and the real
dataclasses. Live-corpus conformance (how much of an existing
real store would pass) is NOT PROVEN — no real store exists on
this machine to audit (TF-041 found none committed).
