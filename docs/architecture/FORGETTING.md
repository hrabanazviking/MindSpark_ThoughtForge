# TF-051 — Forgetting: User-Ordered Deletion, Total and Provable

*Part III, slice TF-051. Code:
`src/thoughtforge/knowledge/forgetting.py` (the forgettable
surfaces as data, `forget_record()`, `verify_forgotten()`,
and `forgetting_problems()` re-deriving every claim below
from the live tree), wired into TF-047's writeback delete
leg in `src/thoughtforge/knowledge/writeback.py`. Tests:
`tests/test_tf051_forgetting.py`.*

## 1. The question and the measured starting state

The roadmap asks: when the user orders a record deleted,
is the deletion *total* — store, indices, and caches —
and *provable*, or does the record linger in derived
surfaces after its file entry is gone? Measured before
this slice, by direct probe on a synthetic store:

- **No forgetting API existed.** Nothing in the tree took
  a user's deletion order for a record and carried it
  out. The only deletion-shaped mechanisms were the
  lifecycle prune (TF-045 — the *system* retiring records
  by score) and TF-047's writeback `records_to_delete`
  leg (mark + rewrite).
- **The writeback deletion was not total.** Seed a fact,
  an episode whose `linked_facts` names it, a rebuilt tag
  index, and the fact's id in the recent queue; delete the
  fact through the writeback leg. The fact left
  `user_profile_store.jsonl` — and its id stayed in
  `tag_index.json` (`ferry -> [fct_target, ep_link]`),
  stayed in `recent_queue.json`, and stayed dangling in
  the episode's `linked_facts`. The "forgotten" record's
  identity remained readable on three surfaces.
- **Nothing could prove a deletion.** No function scanned
  the store, indices, and caches for a deleted id's
  residue; "deleted" meant "one rewrite dropped it from
  one file" and the rest was faith.

## 2. The forgettable surfaces

`FORGET_SURFACES`, in search order:

| surface | file | treatment |
|---|---|---|
| `profile_store` | `user_profile_store.jsonl` | home: mark `deleted` + rewrite drops the record |
| `episodic_store` | `episodic_store.jsonl` | home, same shape |
| `patterns_store` | `response_patterns.jsonl` | home, same shape |
| `thread_state` | `active_thread_state.json` | cleared (direct API only — see §5) |
| `tag_index` | `tag_index.json` | derived index, rebuilt from live records only |
| `recent_queue` | `recent_queue.json` | recency cache, the id purged |
| `episode_links` | (inside `episodic_store.jsonl`) | `linked_preferences` / `linked_facts` scrubbed of the id |

Excluded, by named ruling in `EXCLUDED_SURFACES`:

- **Personality core** (`personality_core.yaml`) — the
  persona's identity record, not user memory (TF-046:
  who is remembering, not a memory). It carries no user
  fact and is removed only by manual versioning.
- **Knowledge DB** (SQLite) — a read-only reference
  estate (Wikidata / ConceptNet / reference chunks). It
  holds no memory-store record ids; there is no user
  memory in it to forget.

## 3. The leg: `forget_record(store, record_id)`

1. **The one refusal.** If the id is the personality
   core's, the report is `refused_protected`, the reason
   is stated, the core is left byte-intact, and the proof
   (which honestly shows the record still present) rides
   the report.
2. **Home removal.** The id is resolved across profile,
   episodes, patterns (in that order), then the thread
   state. Removal uses the store's own deletion shape —
   status `deleted` + rewrite of that store — so a
   forgotten record leaves its file exactly the way
   TF-045/TF-047 removals do; the thread record's file
   is cleared.
3. **Link scrub.** Every episode's
   `linked_preferences` / `linked_facts` is filtered of
   the id (counted on the report as `links_scrubbed`);
   the linking episodes themselves survive — a moment
   that referenced a fact is not the fact.
4. **Cache purge.** The id is removed from the recent
   queue.
5. **Index rebuild.** The tag index is rebuilt from live
   records only — a forgotten id has no live record left
   to be indexed by.
6. **Proof.** `verify_forgotten` runs (§4); the report
   claims `forgotten` only when the proof passes.

**A user's order outranks retention.** The lifecycle's
`protected` retention class guards a record against the
*system's* prune; it does not guard it against its owner
ordering it forgotten. A protected fact is forgotten by
this leg exactly as any other (pinned in tests).

**Residues count.** Steps 3–5 run even when no home
record exists: a stale id sitting only in the queue and
the index is still the user's data, and forgetting it
reports `forgotten` with the cleaned surfaces named. A
second forget of the same id, with nothing left
anywhere, reports `not_found` — never a silent success.

## 4. The proof: `verify_forgotten(store, record_id)`

A `ForgetProof` whose checks (`PROOF_CHECKS`) each
re-read their surface fresh:

- absent from the profile, episodic, and patterns
  loaders; absent from the thread state; absent from
  the personality core;
- absent from every tag-index list; absent from the
  recent queue; no episode links to it;
- **raw scan:** the id string appears in *no file* in
  the memory directory (files scanned are named on the
  proof) — the loaders could all agree and an unlisted
  derived file would still betray a residue; the bytes
  are the ground truth.

`proof.forgotten` is the AND of all checks. A forget
whose proof fails reports status `not_proven` with the
failed checks named — success is never claimed on a
failed proof. The proof is by **identity** (the record
id): content a second, live record carries in its own
right (the same summary on a different record) is that
record's content, not a residue of the forgotten one.

## 5. Wiring

TF-047's writeback `records_to_delete` leg routes
through `forget_record` with `include_thread=False` —
TF-047's ruling that writeback cannot touch the
standing working record stands, so the direct
`forget_record` API is the only path that can forget
the thread state. A writeback deletion is now applied
only when the report says `forgotten`; otherwise the
leg refuses with the report's reason — an unknown id
keeps TF-047's "not found" refusal, and the personality
core keeps its refusal, now with the reason named
("identity, not user memory"). The reinforce and
archive legs are unchanged: archive is retirement by
rule, not forgetting, and keeps its own shape and its
own (weaker) claim.

## 6. Stated boundaries

- **The lifecycle prune is not user-ordered forgetting**
  and is unchanged. Its tag-index cadence (rebuilt in
  routine/heavy/emergency modes, not light) is TF-045's
  pinned behaviour; prune retirement is the system's
  housekeeping, and TF-051's totality claim covers the
  deletion legs a user or a conversation orders.
- **In-memory turn state is out of scope.** Activation
  bundles and scaffolds are transient per-turn data
  (TF-046), not a store, index, or cache. A record
  forgotten between turns cannot be retrieved by a
  later turn, because retrieval re-reads the store; a
  bundle already built inside a turn already in flight
  is that turn's working copy, and no claim is made
  about it.
- **Proof is by record id, on disk.** Backups of the
  memory directory made outside the store (the Samtal
  lesson: copies exist) are outside this leg's reach —
  forgetting proves the live store clean, not every
  copy anyone ever made of it.

## 7. Proof

Fixture grade: synthetic stores through the leg and the
proof (each home store, the thread both ways, protected
retention yielding to a user order, the personality
refusal, residue-only cleanup, idempotence, the proof
failing before deletion and catching a re-inserted
queue residue, the raw scan covering every file), and
real writeback turns through TF-047's leg (total
deletion with residues seeded, unknown-id refusal,
identity refusal). `forgetting_problems() == []`
re-derives the leg from the live tree, and
`writeback_problems()` now includes it. Forgetting over
a real, lived-in store is NOT PROVEN — no real store
exists on this machine (TF-041).
