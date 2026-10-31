# ThoughtForge Architecture — Retrieval Precision Fixtures (TF-043)

Retrieval precision fixtures: a synthetic corpus with known
right answers; recall/precision measured, baseline pinned.
Part III's second measurement slice. Built 2026-10-31 on
branch `yrsa/thoughtforge-road-build`.

The fixture lives as verifiable data in
`src/thoughtforge/knowledge/retrieval_precision.py`
(`FIXTURE_ENTITIES`, `FIXTURE_REFERENCE_CHUNKS`,
`FIXTURE_CONCEPTNET_EDGES`, `FIXTURE_GEONAMES`,
`FIXTURE_QUERIES`, `build_fixture_db()`,
`evaluate_retrieval()`, `BASELINE`, `baseline_problems()`);
the tests are `tests/test_tf043_retrieval_precision.py`.

## 1. The corpus

A synthetic corpus in TF-017's invented Lantern Hollow world,
built into a fresh SQLite DB through the tree's own schema:

- **5 target entities** (ferry, mill, observatory,
  lighthouse, bridge) and **4 reference chunks** answering
  alongside three of them, **2 ConceptNet edges**, **2
  GeoNames places**.
- **3 distractor entities** that share query terms without
  answering them (Ferry House Inn, Mill Cottage, Star Inn) —
  and one query makes a distractor the target, so the
  fixture cannot pass by ignoring shared-term records.
- **Filler**: 60 entities + 30 reference chunks of unrelated
  text, so FTS bm25 lands in a realistic range (TF-041
  measured a one-row corpus scoring near zero).

## 2. The queries and the metric rules

Nine queries with known right answers by construction:
entity+reference answers (Q1–Q3), a single-entity answer
(Q4), a shared-term query with two right answers (Q5,
"river crossing"), a negative query whose right answer is
the empty bundle (Q6), a ConceptNet-isolating query (Q7,
"ferry crosses river" — its terms match no FTS record), a
distractor-as-target query (Q8, "ferry house inn"), and a
GeoNames query (Q9, "Lantern Hollow").

Relevant sets name bundle record ids exactly as the forge
emits them (entity qids, chunk ids as strings, a ConceptNet
concept1, a GeoNames name). Rules, stated once in the module:
precision counts returned records in the answer set; an
empty return against a non-empty answer set scores 0/0; an
empty return against the labelled negative query scores
1/1 — correctly returning nothing is a success, not a
vacuous pass. Reciprocal rank is computed per answered
query; the negative query is excluded from the MRR mean.

## 3. The baseline (measured on this VM at banking)

| Setting | Precision | Recall | F1 | MRR |
|---|---|---|---|---|
| Default (sql, top_k 10, min_score 0.45) | **0.8889** | **0.9630** | 0.9111 | 1.000 |
| Open (sql, top_k 10, min_score 0.0) | 0.8148 | 1.0000 | 0.8815 | 1.000 |

At the defaults: every answered query **opens with a right
answer** (MRR 1.0), recall is perfect on 8 of 9 queries,
and the negative query returns exactly nothing. The open
setting recovers the last of the recall and pays for it in
precision — the threshold's trade, measured both ways.
`baseline_problems()` names any future fall below these
numbers (tolerance 0.01); TF-064's gate consumes it, and
improvement is never a problem.

## 4. Finding 1 — GeoNames is threshold-dead at the defaults

A GeoNames record's `fts_score` is the fixed constant 0.5
in the forge's geonames pass, so through the mapped proxies
its composite is **always exactly 0.285** — it can never
pass the 0.45 default threshold, however well a place
matches. Measured live: the "Lantern Hollow" place record
is present at min_score 0 (Q9 recall 1.000) and absent at
the defaults (Q9 recall 0.667 — the slice's only default
recall loss). The geonames pass retrieves; the threshold
silently discards everything it retrieves. Pinned as
`GEONAMES_DEFAULT_SCORE`; TF-044 owns the scoring question.

## 5. Finding 2 — ConceptNet fires on any single shared term

The ConceptNet pass matches an edge when **any one** query
term names either concept. The ferry→river edge (weight
4.0, composite exactly 0.46 — just over the threshold)
therefore rides every query containing "ferry" or "river":
at the defaults it is a non-answer in Q1, Q5 and Q8, and
each of those queries' precision is exactly **2/3** for
that reason alone. The same pass is Q7's only answer —
the fixture holds both facts at once: the pass is genuinely
useful and genuinely imprecise, and now both are measured.
Pinned as `CONCEPTNET_FERRY_EDGE_SCORE`; TF-044 owns
whether single-term firing should stay.

## 6. What this slice did not do

- **No retuning.** The 0.45 threshold, the proxy fusion,
  the fixed geonames score and the ConceptNet pass are
  measured, not changed — TF-044 makes the weights explicit
  and tunable; TF-064 judges improvement against §3.
- **No vector/hybrid measurement.** sentence-transformers
  is absent on this VM (D-ENV-1), so only the SQL path is
  measured; TF-053 owns per-profile embedding availability.
- **No live corpus.** Proof grade FIXTURE — synthetic
  corpus on this VM. Real-corpus precision is a different
  measurement, owned by no slice yet and never claimed here.
