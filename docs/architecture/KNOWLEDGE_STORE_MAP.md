# ThoughtForge Architecture — Knowledge Store Mapped (TF-041)

Knowledge store mapped: models/lifecycle/scoring/store
documented from code; the store's invariants listed + tested.
Part III's first slice. Built 2026-10-31 on branch
`yrsa/thoughtforge-road-build`.

The map lives as verifiable data in
`src/thoughtforge/knowledge/store_map.py`
(`KNOWLEDGE_DB_TABLES`, `KNOWLEDGE_DB_FTS`, `MEMORY_FILES`,
`PERSISTED_RECORDS`, `TRANSIENT_RECORDS`, `RETRIEVAL_MAP`,
`INVARIANTS`, `store_map_problems()`, `bundle_problems()`);
the tests are `tests/test_tf041_knowledge_store_map.py`.
This slice also **fixes CORE_LOOP §8-F2** (§6 below).

## 1. There are two stores, not one

"Knowledge store" in this tree is two different machines:

- **The SQLite knowledge database** — the sovereign RAG corpus
  (Wikidata / ConceptNet / GeoNames / DBpedia / reference
  documents), owned by the ETL wing (`etl/schema.py`,
  `etl/embeddings.py`), read at turn time by `KnowledgeForge`
  (`knowledge/forge.py`). One file: `knowledge.db`.
- **The file-based memory store** — the conversation's own
  memory (personality, preferences, facts, episodes, patterns,
  thread state), owned by `MemoryStore`
  (`knowledge/store.py`), one directory of small files.

They share the record vocabulary (`knowledge/models.py`) and
the scoring formulas (`knowledge/scoring.py`), and nothing
else: different files, different writers, different readers.
Part III's later slices each name which store they mean —
TF-042's ingestion contract is about what may enter either;
TF-051's forgetting must erase from both.

## 2. The knowledge database (schema from `etl/schema.py`)

10 base tables, created idempotently (`IF NOT EXISTS`):

| Table | Key | Holds | Read by retrieval today |
|---|---|---|---|
| entities | qid | Wikidata entities (label, description, aliases, popularity) | SQL entity pass + vector search |
| statements | statement_id | subject–property–object facts | — (ETL-side) |
| labels | (qid, language) | multi-language labels | — |
| embeddings | qid | entity embedding BLOBs | vector search (etl/embeddings) |
| conceptnet_edges | edge_id | commonsense edges + weight | SQL conceptnet pass |
| geonames_places | geonames_id | places | SQL geonames pass |
| dbpedia_entities | uri | DBpedia label/abstract/types | — |
| reference_chunks | chunk_id | chunked reference docs | SQL reference pass + vector search |
| reference_embeddings | chunk_id | reference embedding BLOBs | vector search |
| ingestion_log | log_id | ETL run log | — |

3 FTS5 indexes: `entities_fts` (over entities),
`reference_fts` (over reference_chunks), `conceptnet_fts`
(over conceptnet_edges — built by the schema but read by no
pass today; the conceptnet SQL pass queries the base table
directly). 15 secondary indexes; connections run WAL with
the schema module's read/write pragma sets.

## 3. The memory store (files from `knowledge/store.py`)

`MemoryStore` manages exactly 7 files in the memory dir:

| File | Format | Holds |
|---|---|---|
| personality_core.yaml | YAML, single record | PersonalityCoreRecord |
| user_profile_store.jsonl | JSONL | UserPreferenceRecord + UserFactRecord (told apart by `record_type`) |
| episodic_store.jsonl | JSONL | EpisodicMemoryRecord |
| response_patterns.jsonl | JSONL | ResponsePatternRecord |
| active_thread_state.json | JSON, single record | ActiveThreadStateRecord |
| tag_index.json | JSON, derived | tag → [record_id] over profile+episodes+patterns |
| recent_queue.json | JSON, derived | record ids, most-recent-first, cap 64 |

## 4. The record census (`knowledge/models.py`)

15 named record dataclasses: **6 persisted** (above) and
**9 transient** — InputSketch, ActivatedRecord,
MemoryActivationBundle, CognitionScaffold, CandidateRecord,
FragmentRecord, FinalResponseRecord, WritebackRecord,
RuntimeTurnState — which exist only inside a turn and are
never written to either store. (The models header says "14
core types": its numbered list counts the bundle but not its
ActivatedRecord member.)

Persisted defaults, as constructed:

| Record | status | retention | stale after | record_type |
|---|---|---|---|---|
| PersonalityCoreRecord | active | protected | — (never auto-pruned) | — (no such field) |
| UserPreferenceRecord | created | standard | 200 turns | user_preference |
| UserFactRecord | created | standard | 500 turns | user_fact |
| EpisodicMemoryRecord | created | ephemeral | 30 turns | episode |
| ResponsePatternRecord | created | standard | 100 turns | response_pattern |
| ActiveThreadStateRecord | active | ephemeral | expires_after_turns 10 | active_thread_state |

The vocabularies are closed: `RecordStatus` has exactly 7
values (created, reinforced, active, cooling, stale,
archived, deleted) and `RetentionClass` exactly 5 (protected,
durable, standard, ephemeral, discardable). `from_dict` on
every type ignores unknown keys — older/newer files load
without crashing, and silently drop what they don't know.

## 5. Retrieval, scoring, lifecycle — as they actually run

**Retrieval** (`KnowledgeForge.retrieve`, paths sql / vector
/ hybrid; the core calls it flat at `top_k=10`, path from the
router's sketch). SQL runs 4 passes — entities via FTS
(asks `top_k*2`), reference chunks via FTS (`top_k//2`),
ConceptNet by key-term WHERE (`top_k//3`), GeoNames by
first-key-term LIKE (`top_k//5`). Hybrid dedupes vector hits
against SQL hits **by qid only**. Candidates are fused,
filtered at `min_score` (default `MIN_MEMORY_SCORE` 0.45),
sorted, capped at `top_k`, and assembled into a bundle whose
cue is label + first 80 chars of description/content,
trimmed to ≤18 words.

**The fusion is a proxy, recorded plainly** (TF-044 owns
making it honest/tunable): every record is scored with the
**user_fact** weight row whatever its real type;
semantic similarity is the vector score or
`min(fts/20, 1)`; tone and preference relevance are the
neutral constant 0.5; recency is a per-source constant
(reference .9, conceptnet .8, geonames .75, wikidata .7,
dbpedia .65); importance is
`min(popularity + weight (default 0.5), 1)`.

**Scoring module** (`knowledge/scoring.py`): retrieval
weights per record type (5 rows) + prune/survival weight
rows + 3 writeback formulas — every row sums to exactly 1.0
(K1). Thresholds: memory 0.45, fragment 0.54, final 0.68,
preference-writeback 0.72, episode-writeback 0.58,
pattern-learning 0.78 (K2). Recency is a true half-life
decay per type (K3): thread ~0.07 d < episode 7 d < pattern
30 d < preference 90 d < fact 180 d < personality 9999 d.

**Lifecycle** (`knowledge/lifecycle.py`): 4 prune modes
(light/routine/heavy/emergency) with archive+delete
thresholds per record kind, hard store caps enforced by
lowest survival score, dedupe of profile records by
category+tags keeping the higher confidence, reinforcement
(+0.05 confidence / +0.03 importance / +0.02 quality per
unit signal, capped at 1.0), and thread cooling. **TF-045
note:** it was mapped-not-wired when this map was written
(D-ORPHAN-1); TF-045 has since wired it (per-turn thread
cooling in the core + `prune_memory`, K10 below) and stated
its rules in `docs/architecture/LIFECYCLE.md`. The map's
K10 checker now pins the wiring's presence instead.

## 6. F2 FIXED: the retrieval gate reads the configured path

Before this slice, `core._retrieve` tested
`get_knowledge_db_path().exists()` — the *default* DB — while
retrieval itself used the core's configured
`self._knowledge.db_path`. A core built for a custom DB was
gated by a different file entirely. Now the gate reads
`self._knowledge.db_path`, and the fix is proven both ways
in the tests: a populated custom DB **is** retrieved while
the default path is absent, and a missing custom DB is
**not even created** while a populated default exists (the
old gate passed and retrieval then touched the custom path).
`RECONCILED_FINDINGS` in `cognition/core_docs.py` flips F2
to CLOSED with this evidence, exactly as TF-039's checker
instructed; both live checkers (TF-039's and this slice's)
now AST-check the gate so the fixing docstring's prose
cannot be misread as a reopening, and a real revert is
caught by name.

## 7. The invariants (K1–K16)

Stated as data in `store_map.INVARIANTS`; K1–K10 are
re-checked against the live tree by `store_map_problems()`,
K11–K16 are exercised behaviourally in the tests:

- **K1** every weight row (retrieval ×6, prune, survival,
  writeback ×3) sums to 1.0.
- **K2** the six thresholds stand at their mapped values.
- **K3** recency: 1.0 at age 0, 0.5 at one half-life,
  strictly falling; half-life ordering as §5.
- **K4** prune thresholds monotone per kind
  (light ≥ routine ≥ heavy ≥ emergency), delete > archive
  in every cell.
- **K5** every store cap has hard ≥ soft.
- **K6** the schema creates exactly the mapped 10+3 tables,
  idempotently.
- **K7** MemoryStore manages exactly the 7 mapped files;
  `load_all` returns exactly the 5 mapped sections.
- **K8** persisted defaults + the closed 7/5 vocabularies,
  as §4.
- **K9** the retrieval gate reads the configured path (§6).
- **K10** lifecycle wired by TF-045 (D-ORPHAN-1 fixed):
  per-turn thread cooling + the explicit `prune_memory`
  surface; fact pruning never automatic per turn.
- **K11** bundle contract: total == len(records),
  confidence == mean(score), scores descending, all ≥
  min_score (rounding aside), ≤ top_k, cues ≤ 18 words.
- **K12** round-trips faithful for every persisted type.
- **K13** rewrites are the deletion mechanism: archived /
  deleted records leave the file at the next rewrite;
  malformed JSONL lines are skipped, never fatal; missing
  files load as empty/None.
- **K14** tag index covers profile+episodes+patterns only,
  ids deduped per tag; recent queue most-recent-first,
  deduped, capped at 64.
- **K15** pruning never touches a protected record in any
  mode; an ancient unprotected record falls to the
  emergency pass and is then gone from its file.
- **K16** the forge's composite equals the mapped proxy
  formula exactly (hand-checked: reference record, fts 10 →
  0.49).

## 8. Measured (this VM, synthetic only)

- **Populated corpus** (1 Lantern Hollow entity + 50 filler
  entities, FTS rebuilt): SQL retrieval for "Lantern Hollow
  ferry" returns the known entity first, bm25 14.92 → fused
  score **0.6685**, bundle contract clean (K11).
- **Tiny corpus** (the same entity alone): bm25 ≈ 0, so the
  mapped proxies score the real hit at **0.37 — below the
  0.45 default threshold**: the default bundle is empty
  while `min_score=0` returns it. Retrieval *correctness* on
  small corpora is therefore a scoring question, pinned
  here as arithmetic for TF-043 (precision fixtures) and
  TF-044 (scoring honesty) — not retuned in this slice.
- **Prune**: a 1000-day-old episode under emergency prune is
  deleted (count 1) and absent from its file afterwards; a
  1000-day-old **protected** preference survives all four
  modes with its status untouched (`created`).

Proof grade: FIXTURE — synthetic DB and synthetic memory
stores on this VM. The **vector path is not exercised**:
sentence-transformers is absent here (D-ENV-1), so vector /
hybrid retrieval quality is NOT PROVEN (TF-053 owns
per-profile embedding availability; TF-043 will pin
precision for the paths that can run).

## 9. What this slice did not do

- **No retuning.** The proxy fusion, the 0.45 threshold, and
  the flat `top_k=10` core call are mapped, not changed —
  TF-043 measures precision, TF-044 makes weights tunable,
  TF-098 keeps the core's flat retrieval cost.
- **No ingestion contract** — what a "fact" is (shape,
  provenance, confidence) is TF-042, next in roadmap order.
- **No lifecycle wiring** — TF-045 owns connecting
  MemoryLifecycle to the engine (D-ORPHAN-1 stays open).
  *(TF-045 update: landed — see the §5 note and K10 above;
  D-ORPHAN-1 is FIXED in TF-016's docket.)*
- **No new core surface.** The F2 fix changes one gate
  expression inside `_retrieve`; the TF-034 frozen API is
  untouched (`verify_api_surface() == []` stands).
