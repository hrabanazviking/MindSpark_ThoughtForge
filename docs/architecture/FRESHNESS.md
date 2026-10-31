# TF-059 — Freshness: Time-Aware Retrieval

*Part III, slice TF-059. Code:
`src/thoughtforge/knowledge/freshness.py` (the domains,
weights, and rules as data; `assess_freshness`,
`rank_records`, `search_store`, `freshness_report`, and
`freshness_problems()` re-deriving every claim below from
the live tree), plus one wiring change in
`src/thoughtforge/knowledge/forge.py` (D-FRESH-1). Tests:
`tests/test_tf059_freshness.py`.*

## 1. The question and the measured starting state

The roadmap asks: can retrieval be time-aware — recent
facts weighted per a stated rule, for time-sensitive
domains? Measured this tick by direct probe:

- **Retrieval recency was time-blind.** The forge's
  `_score_records` set the recency dimension from
  `SOURCE_RECENCY_PROXIES` — a constant per source
  (reference 0.9, wikidata 0.7, conceptnet 0.8,
  geonames 0.75, dbpedia 0.65). Two records identical
  except a `created_at` of 2020 vs 2026 scored
  *exactly* the same composite (**0.57**). The
  knowledge DB's tables carry no timestamp column at
  all, so for DB records the proxy was the only
  recency available — and nothing said so.
- **Personal records carried time nobody read.** Every
  persisted record carries `created_at` / `updated_at`
  / `last_used_at` and a `recency_weight` field that
  *no code reads* (only the model definitions and
  TF-042's field census mention it). The half-life
  decay (`compute_recency_score`, TF-041 K3) was
  consumed only by the lifecycle: time decided what
  got **retired**, never what got **retrieved first**.
- **No domain was time-sensitive.** A ferry departure
  time and a river's name would age identically — at
  the user_fact half-life of 180 days, if at all —
  and a stale schedule fact carried no mark.

## 2. The domains, as data

`TIME_SENSITIVE_DOMAINS` — a record is in a domain
when a folded marker (lowercased, `-`/`_`/space
removed) matches one of its tags, its preference
category, or its fact_value keys:

| Domain | Example markers | Half-life | Stale after |
|---|---|---|---|
| schedule | schedule, departure, appointment, deadline, openinghours | 2 days | 7 days |
| location | location, whereabouts, address, travel | 7 days | 30 days |
| status | status, currentproject, currentrole, availability | 7 days | 30 days |
| price | price, cost, fee, rate, fare | 30 days | 90 days |

Rules (`FRESHNESS_RULES`, as data): the domain governs
(a sensitive record ages at its domain's half-life,
not its type's; several domains → the most perishable
governs); timeless records are never stale; freshness
is decayed, never binary (Retrieval spec §10.4); a
stale record is never silent; the clock is the write;
every leg is read-only.

## 3. The clock and the score

The freshness clock is `updated_at` (when the stored
claim last changed or was re-stamped), falling back
to `created_at`, then `last_used_at`. A missing or
unparseable clock reads as 30 days old —
`FRESHNESS_AGE_DEFAULT_DAYS`, pinned equal to the
lifecycle's own parse default: an unreadable timestamp
is neither immortal nor instantly ancient.

The score is the decayed recency `0.5 ** (age /
half_life)`: the domain's half-life for sensitive
records, the record type's `HALF_LIFE_DAYS` (through
scoring's own `compute_recency_score`) for timeless
ones. Pin: a schedule fact exactly 2 days old scores
exactly 0.5.

Verdicts: `fresh` (score ≥ 0.5), `ageing`
(0.2 ≤ score < 0.5), `stale` (**sensitive only**:
score < 0.2 or age beyond the domain's stale
horizon), `timeless` (no domain — freshness does not
judge this record; a 10-year-old timeless fact is
still `timeless`, pinned).

## 4. The weighting rule

`rank_records` scores with the tree's own
`score_retrieval`: semantic = the caller's relevance,
recency = the freshness score, importance = the
record's own. The weight row is the rule's teeth:

- **Sensitive records** use `TIME_SENSITIVE_WEIGHTS`
  — recency carries **0.40**. Two otherwise identical
  schedule facts, one fresh and one 30 days old,
  separate by ~0.40 of composite (pinned > 0.3), not
  by the fact row's 0.10.
- **Timeless records** keep their record type's
  shipped row, weights byte-identical.

Ties break by record_id — ranking is reproducible.

## 5. Search over the personal store, stale never silent

`search_store(store, query)` ranks a MemoryStore's
profile records, episodes, and patterns by term-overlap
relevance (TF-058's stated rule: the share of the
forge's key terms present in summary + tags) composed
with freshness as in §4, and returns a
`MemoryActivationBundle` whose raw dicts carry the
full assessment (domain, age, score, verdict) — a
ranking a consumer can explain, and a freshness a
consumer can show.

**Stale sensitive records are excluded by default**
and named in the returned report's `stale_excluded`.
A caller that wants them passes `include_stale=True`
and receives them flagged `stale` in their raw. The
store is only read — byte-pinned across search and
report.

`freshness_report(records_or_store)` gives the
inspector's view: counts by verdict and domain, every
sensitive record named with age/score/verdict, and
the stale ids listed.

## 6. D-FRESH-1 — fixed where a clock exists, stated where none does

`KnowledgeForge._score_records` now consults a
record's own timestamp first (`updated_at` /
`created_at` / `last_used_at` / `timestamp`, via
`freshness.recency_from_timestamps`): a record
carrying one gets decayed recency of its real age at
the default 30-day half-life instead of the source
proxy. Pinned both ways:

- a timestamped pair (2020 vs now, otherwise
  identical) now separates — the new record outscores
  the old;
- a timestamp-**less** pair scores exactly as before
  (proxy equality) — which is every record the
  shipped DB tables can produce today, and TF-043's
  whole fixture corpus, so the pinned retrieval
  baseline does not move.

The remaining share is stated, not hidden: the ETL
schema carries no date column, so DB recency stays a
proxy until an ingestion slice gives it a date to
carry (TF-106's source registry / TF-111's ingestion
own that road).

## 7. Docket

- **D-FRESH-1 added + FIXED by this slice** for
  timestamp-carrying records (see §1/§6); its
  timestamp-less DB share is the stated boundary
  above. Totals: 55 entries — 40 OPEN /
  1 OPEN-ENVIRONMENT / 12 FIXED / 2
  CLOSED-BY-VERDICT.
- No other entry moves: TF-016 totals in the state
  file are updated to match.

## 8. Boundaries, stated

- Freshness judges *when a claim was last stamped*,
  not whether it is true. A fresh falsehood scores
  fresh; contradiction (TF-050) judges claims against
  each other, freshness against the clock.
- `search_store`'s relevance is a stated term-overlap
  rule, not bm25/vector ranking; its fixtures prove
  the freshness weighting and the stale rule, not
  ranking quality — Part VII owns ranking research.
- Nothing here deletes, archives, or rewrites a
  record: retirement is the lifecycle's (TF-045),
  forgetting is TF-051's. A stale verdict is a
  retrieval/report fact, not a store mutation.
- Proof grade is FIXTURE (synthetic stores and
  records with explicit clocks, on this VM).
  Freshness over a real lived-in store is **NOT
  PROVEN** (no real store on this machine, TF-041),
  and live-model use of the flagged bundle is NOT
  PROVEN — never claimed.
