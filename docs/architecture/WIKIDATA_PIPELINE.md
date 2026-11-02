# ThoughtForge Architecture — Wikidata Pipeline Hardened (TF-107)

Wikidata pipeline hardened: rate-respecting, resumable,
checkpointed (long ingestions survive interruption).
Part VI's third slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The rules live as verifiable data in
`src/thoughtforge/etl/wikidata_pipeline.py`
(`PIPELINE_RULES`, the checkpoint record IO, `RatePacer`,
checked by `pipeline_problems()` and
`pipeline_live_problems()`); the enforcement is in
`src/thoughtforge/etl/wikidata.py`; the tests are
`tests/test_tf107_wikidata_pipeline.py`. This slice
**adds and fixes D-WIKI-1, D-WIKI-2 and D-WIKI-3** and
**discharges D-COV-2's wikidata-depth share** (§5).
TF-105's living map was updated in the open (the
wikidata row's write mode, durability posture and notes).

## 1. Measured start

Probed live this tick, before any change:

- **No checkpoint, no resume, no pacing.** Grep over the
  builder found no checkpoint, resume or throttle of any
  kind. The only commits were every 50,000 entities and
  at the end, under `PRAGMA synchronous = OFF` — so an
  interruption discarded everything since the last 50k
  boundary, and OFF is the posture SQLite itself warns
  can corrupt a database on power loss.
- **Re-running duplicated statements (D-WIKI-1).** A
  7-entity synthetic dump built twice moved the store
  from 7 statements to **14**, while entities and labels
  stayed at 7: entities/labels flush `INSERT OR REPLACE`,
  statements flushed a plain `INSERT` into an
  AUTOINCREMENT table. The only "resume" the old shape
  offered — run it again — corrupted the store it was
  meant to repair.
- **Failure masked itself (D-WIKI-3).** A stream raising
  mid-dump (4 entities processed, batches flushed) did
  not surface its own error: the build's connection was
  never closed on the exception path, so `_finish_log`
  on a fresh connection raised
  `sqlite3.OperationalError: database is locked` — the
  original failure was replaced by that error, and no
  failure row reached `ingestion_log` at all.

## 2. Checkpoints and resume

A checkpoint is a sidecar JSON record beside the DB —
`<db>.wikidata-checkpoint.json` — in format
`thoughtforge-wikidata-checkpoint/1`. The schema gains
**no** table for it (TF-105's filed schema facts stand).

- **Cadence.** Every `checkpoint_every` stream items
  (default `DEFAULT_CHECKPOINT_EVERY = 1000`, counted
  over accepted *and* skipped items) the build flushes
  all batches, commits, and only then writes the
  checkpoint — atomically (tmp file + `os.replace`).
  A checkpoint therefore always describes a commit that
  actually landed; an interruption loses at most one
  cadence interval of uncommitted work, rolled back on
  the failure path. The old bound was 50,000 entities.
- **The record** carries the dump fingerprint (resolved
  path + size + mtime_ns), the config fingerprint
  (languages, `instance_of` filter, limit), the stream
  position (`items_seen`), the stats as of that commit,
  the last QID seen, and its status (`in_progress` /
  `complete`).
- **Resume** (default on): a matching in-progress
  checkpoint continues the stream at `items_seen`, with
  its stats as the starting counts; a matching
  **complete** checkpoint returns its filed stats
  without streaming at all (pinned by a test whose
  stream raises if touched). Any fingerprint difference
  raises `CheckpointMismatchError` naming each differing
  field — a changed ask is a new build (`resume=False`
  discards the checkpoint explicitly), never a silent
  continuation over the wrong dump. An unreadable or
  malformed checkpoint raises `CheckpointCorruptError`;
  the builder never restarts silently over a checkpoint
  it cannot read.
- **Durability.** The build now runs WAL with
  `synchronous = FULL` — the TF-054 / Samtal posture —
  so a landed checkpoint commit survives power loss.
  TF-105's map is updated in the open: wikidata leaves
  the OFF leg; the other builders' postures are
  unchanged (TF-111 owns any further harmonising).

## 3. Idempotent rows

Re-processing an item never duplicates a row. Entities
and labels already replaced by key. Statements now
flush through a guard: the batch's subjects' stored
tuples are read back, and any statement whose full
stored tuple (subject, property, object value, object
type, qualifiers, refs, source) is **already present**
is not inserted again (duplicates inside one batch are
likewise collapsed). The schema stores no claim id, so
exact-tuple identity is the fidelity the table itself
offers — stated plainly: two genuinely distinct claims
that serialise to the identical tuple were already
indistinguishable in this schema before this slice.
A *changed* fact (same subject + property, new object)
is not a duplicate and still appends; resolving it is
TF-114's update-ingestion road, not this slice's.

Measured after the fix, on the §1 probe's shape: fresh
run 7/7/7 (entities/statements/labels), a completed
re-run still 7/7/7, a `resume=False` re-run still
7/7/7, and an interrupted-then-resumed run ending at
exactly 7/7/7 with the resumed stats equal to a fresh
run's.

## 4. The rate cap — and its filed limit

`max_items_per_second` (default: unset) is enforced by
`RatePacer`: item N of a run may not finish processing
before N / cap seconds have elapsed on the run's clock.
The pacer's clock and sleeper are injectable, and its
arithmetic is proven with a fake clock (5 items at
10/s sleep exactly 0.5s; past the deadline, 0). The
builder wiring is pinned by a recording fake pacer
(one wait per processed item, counts 1..7). A
non-positive cap raises by name — a cap that cannot
bind is a configuration error, not a no-op.

Filed limit, stated plainly: **the cap has no default.**
This builder reads a *local* dump file — there is no
upstream service, no API quota, nothing external whose
limits a default would respect, and a default cap set
above any real parse rate would be theatre. "Rate-
respecting" for this pipeline therefore means: the
ceiling exists, is hard when set, and the checkpoint
cadence makes the run's rate observable in the log
(entities/sec at every checkpoint). A future network
fetch leg (none exists in the tree) would arrive with
its service's limits and set the cap from them.

## 5. Docket

- **D-WIKI-1 — added + FIXED.** Statement duplication
  on re-run (§1, §3): the flush guards on existing
  stored tuples; live checker and tests pin a second
  run and a resume=False re-run at 7 statements, not 14.
- **D-WIKI-2 — added + FIXED.** No checkpoint/resume
  and the synchronous-OFF posture (§1, §2):
  checkpointed commits every 1000 stream items under
  FULL, resume by fingerprint, mismatch and corruption
  refusing by name.
- **D-WIKI-3 — added + FIXED.** The failure path
  masked its own error behind a locked log write
  (§1): the build connection now rolls back and closes
  *before* the failure is recorded, the original
  exception re-raises unchanged, and the interrupted-run
  test finds the `failed:` row in `ingestion_log`.
- **D-COV-2 — narrowed (wikidata share discharged).**
  `tests/test_tf107_wikidata_pipeline.py` imports and
  exercises the wikidata builder directly and in depth
  (21 tests: checkpoint IO, pacing, interruption,
  resume, idempotence, mismatch). The entry stays OPEN
  for TF-112's dry-run depth, its remaining owner.

**Proof grade: FIXTURE** — synthetic dumps through the
real builder on this VM; interruption simulated by a
stream that raises mid-dump (the same shape a kill
presents to the code: an exception between checkpoints,
an uncommitted tail, a checkpoint on disk). A real
multi-GB dump ingestion, a real kill -9 / power loss
against the FULL posture, and any at-scale rate
measurement are **NOT PROVEN**, never claimed.
TF-108 owns schema validation at the gate; TF-111
embedding-at-ingestion resumability (this slice's
checkpoint shape is its precedent); TF-114 changed-fact
updates.
