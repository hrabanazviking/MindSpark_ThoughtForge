# ThoughtForge Architecture — Subset Craft (TF-110)

Subset craft: profile-sized subsets (phone_low's
knowledge must fit phone_low) built by rule + measured.
Part VI's sixth slice. Built 2026-11-01 on branch
`yrsa/thoughtforge-road-build`.

The craft lives as verifiable data in
`src/thoughtforge/etl/subset_craft.py` (`SUBSET_RULES`,
`budget_for_profile()`, `measure_subset()`,
`build_profile_subset()`, checked by
`subset_craft_problems()` and
`subset_craft_live_problems()`); the fixes are enforced
in `src/thoughtforge/etl/subset.py`; TF-105's living
map was updated in the open (`SUBSET_FACTS` gains the
byte-budget + fresh-build facts, the chunk-cap line
loses its floor); the tests are
`tests/test_tf110_subset_craft.py`. This slice **adds
and fixes D-SUBSET-2, D-SUBSET-3 and D-SUBSET-4** and
**discharges D-PROF-1's deployment/subset share**.

## 1. Measured start

Probed live this tick, before any change, with a
synthetic store (30 entities × 20 KB descriptions,
50 reference chunks × 20 KB) through the real builder:

- **No byte budget existed (D-SUBSET-2).** The only
  sizing rule was an entity count (phone_low 80,000).
  Only pi_zero's profile declares any knowledge size
  (`knowledge_subset_max_gb` 0.5 — its entire RAM,
  filed as declared, not endorsed); the other five
  profiles declare nothing, no code compared a build's
  size to anything, and the builder's docstring claimed
  output sizes (~50MB / ~80MB / ~200MB) measured
  nowhere. The reference-chunk cap floored at 1,000
  (`max(1000, limit // 10)`): a phone_low build limited
  to **5 entities copied all 50 chunks — 1.24 MB** of
  chunk text — and reported success.
- **A re-build was not the subset it claimed
  (D-SUBSET-3).** Re-building at limit 2 onto the
  limit-5 output reported 2 entities; the file still
  held **5 entities and all 50 chunks** — the builder
  connected to the existing file and INSERT OR IGNOREd
  into it.
- **Selection order was undefined on ties
  (D-SUBSET-4).** `ORDER BY popularity_score DESC`
  alone leaves tied rows to SQLite's unspecified scan
  order (code-read finding).

## 2. The budgets

| Profile | Entity limit | Byte budget | Source |
|---|---|---|---|
| pi_zero | 50,000 | 0.5 GB | declared: `deployment.knowledge_subset_max_gb` |
| phone_low | 80,000 | 0.5 GB | derived: 0.25 × 2 GB RAM (campaign rule) |
| pi_5 | 200,000 | 1 GB | derived: 0.25 × 4 GB RAM |
| desktop_cpu | 1,000,000 | 2 GB | derived: 0.25 × 8 GB RAM |
| desktop_gpu | 2,000,000 | 4 GB | derived: 0.25 × 16 GB RAM |
| server_gpu | 0 (unlimited) | 16 GB | derived: 0.25 × 64 GB RAM — measured, never trimmed |

Budgets resolve **live** from the shipped profile
JSONs (`budget_for_profile`); the declared value wins
where present, and `budget_source` names which road
produced every budget. The 0.25 derivation is a
campaign rule this slice sets — never presented as a
profile fact.

## 3. The craft loop

`build_profile_subset(source, output, profile)`:

1. Build fresh at the profile's entity limit
   (selection: popularity DESC, qid ASC on ties;
   statements ride kept subjects; chunks ride at
   `entity_limit // 10`, no floor).
2. **Measure** the built file: stat bytes (db + -wal +
   -shm named apart, after the builder's VACUUM),
   per-table row counts, per-table bytes from SQLite
   `dbstat` (None where unavailable — never estimated).
3. While measured > budget: cut the limit to
   `floor(copied × budget / measured)` (at least one
   below the count the last build copied) and rebuild
   fresh. Every attempt is recorded.
4. End states, exactly two: a subset whose **measured**
   size fits, or `success=False` naming the smallest
   build's measured size (schema + FTS alone can exceed
   a tiny budget; no entity trim crafts that, and the
   result says so). An unlimited profile builds once,
   is measured, and its fit is reported — never trimmed.

## 4. What is NOT proven

A real dump's subset size (all fixtures synthetic), a
real phone's storage behaviour, and whether pi_zero's
declared 0.5 GB — its entire RAM — is a sane device
budget: it is filed as the profile's declaration,
read live, not endorsed. TF-111 owns embedding at
ingestion against these subsets; TF-120's Part VI
gate owns the full round trip.
