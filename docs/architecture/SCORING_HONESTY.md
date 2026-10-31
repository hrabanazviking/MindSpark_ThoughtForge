# ThoughtForge Architecture — Scoring Honesty (TF-044)

Scoring honesty: the scoring module's weights made explicit +
tunable per profile; no magic constants unlisted. Part III's
third slice. Built 2026-10-31 on branch
`yrsa/thoughtforge-road-build`.

The census + checker live as verifiable data in
`src/thoughtforge/knowledge/scoring_honesty.py`
(`SCORING_CENSUS`, `EXTERNAL_SCORING`, `STRUCTURAL_LITERALS`,
`FORGE_FUSION_NAMES`, `CONFIG_DRIFT_KEYS`,
`PROFILE_SEMANTIC_TO_IMPORTANCE_SHIFT`,
`PROFILE_BASELINE_IMPACT`, `census()`, `weights_for_profile()`,
`weight_table_problems()`, `scoring_problems()`,
`geonames_composite()`); the tests are
`tests/test_tf044_scoring_honesty.py`.

## 1. What was dishonest

The scoring module's docstring claimed "no magic numbers —
all weights are named constants". Measured against the tree,
that was half true:

- The three **writeback formulas** carried their weights as
  bare literals inside the function bodies (0.30/0.20/0.25/…),
  the **genericness penalty** carried 0.15 / 0.6 / 0.4 inline,
  and the recency fallback half-life was a bare 30.0.
- The **forge's fusion arithmetic** — the scoring that runs on
  every retrieval — carried its constants inline in
  `forge.py`: the fts/20 semantic proxy, per-source recency
  proxies, two 0.5 neutral dimensions, the importance default
  0.5, GeoNames' fixed fts score 0.5, the vector floor 0.3.
  Worse, `_score_records`' docstring quoted the BASE weight
  row (0.35/0.20/0.20/0.10/0.15) while the code scored with
  the `user_fact` row — TF-041 had measured the true
  arithmetic; the docstring still told the old story.
- `configs/default.yaml` carried a **shadow copy** of the
  retrieval weights with two name-drifted keys (`tone_match`,
  `user_preference` — D-CFG-1/TF-009) that no code read: the
  one place a user would tune the weights tuned nothing.
- **No profile could tune anything**: all six hardware
  profiles scored identically (the retrieval share of
  D-PROF-3).

## 2. What is named now (values unchanged, every one)

In `knowledge/scoring.py`: `PREFERENCE_WRITEBACK_WEIGHTS`,
`EPISODE_WRITEBACK_WEIGHTS`, `PATTERN_LEARNING_WEIGHTS`,
`GENERICNESS_PHRASE_PENALTY` (0.15),
`GENERICNESS_PHRASE_PENALTY_CAP` (0.6),
`GENERICNESS_REPETITION_WEIGHT` (0.4),
`DEFAULT_HALF_LIFE_DAYS` (30.0), and the fusion constants the
forge now imports: `FTS_SCORE_DIVISOR` (20.0),
`SOURCE_RECENCY_PROXIES`, `DEFAULT_SOURCE_RECENCY` (0.7),
`NEUTRAL_TONE_SIMILARITY` (0.5),
`NEUTRAL_PREFERENCE_RELEVANCE` (0.5),
`IMPORTANCE_WEIGHT_DEFAULT` (0.5),
`GEONAMES_FIXED_FTS_SCORE` (0.5), `VECTOR_MIN_SCORE` (0.3).
The forge's docstring now states the `user_fact` row it
actually scores with. Every scoring function takes an
optional `weights` table; the default is the shipped table,
byte-identical.

Scoring-shaped constants outside the knowledge layer are
censused in `EXTERNAL_SCORING` with their owners, not moved:
the three `models.py` composite properties (weights checked
live to sum to 1.0 with a −0.10 genericness share), salvage's
already-named `_LENGTH_WEIGHT`/`_CITATION_WEIGHT` (0.45/0.55,
matching `default.yaml` exactly), the core's own
0.2-per-phrase genericness (a different formula — TF-022's
recorded duality, Part II's to keep), and lifecycle's
threshold/cap tables (TF-041 mapped, TF-045 rules).

## 3. The checker (`scoring_problems() == []`)

- Every censused name exists; every weight table sums to 1.0;
  the six thresholds stand at TF-041's pinned values; the
  genericness constants stand at their shipped values.
- **AST scan**: no numeric literal appears in any
  `scoring.py` function body outside `STRUCTURAL_LITERALS`
  (0 / 0.0 / the 0.5 decay base — the half-life formula's own
  definition / 1 / 1.0 unit caps). An unnamed weight cannot
  be reintroduced silently.
- **AST wiring**: the forge's `_score_records`,
  `sql_retrieve`, and `vector_retrieve` reference the named
  fusion constants.
- **Config drift pinned**: `default.yaml`'s pruning/survival
  tables equal the code's; its fragment weights equal
  salvage's; its retrieval row differs by exactly the two
  recorded drift keys, values equal under the mapping. The
  file stays a shadow copy (TF-032's live-keys ruling) — but
  its drift is now a checked fact, not an audit discovery.
- Every shipped profile resolves through
  `weights_for_profile` and validates (exact dimension keys,
  finite, non-negative, sums to 1.0).

## 4. Per-profile tuning — and its measured verdict

`weights_for_profile(profile_id[, record_type])` returns the
shipped tables with the profile's stated shift applied
(weight moved semantic_similarity → importance):
server_gpu / desktop_gpu / desktop_cpu / pi_5 = 0.0,
phone_low = 0.05, pi_zero = 0.10. A `KnowledgeForge` built
with `retrieval_weights=weights_for_profile(pid, "user_fact")`
scores with that profile's table; a default forge scores
exactly as before. Unknown profiles/record types raise
naming the valid options. `ThoughtForgeCore` does not select
a scoring profile yet — the surface is wired at the forge,
the layer that owns the arithmetic; the core's flat
retrieval (top_k included) stays TF-098's remainder of
D-PROF-3.

Measured on TF-043's known-answer fixtures (SQL path,
top_k 10, min_score 0.45, this VM), `PROFILE_BASELINE_IMPACT`:

| Profile | Precision | Recall | F1 | MRR |
|---|---|---|---|---|
| desktop_cpu / desktop_gpu / pi_5 / server_gpu (shift 0) | 0.8889 | 0.9630 | 0.9111 | 1.000 |
| phone_low (shift 0.05) | 0.8519 | 0.9630 | 0.8889 | 1.000 |
| pi_zero (shift 0.10) | 0.8519 | 0.9630 | 0.8889 | 1.000 |

Two facts, stated plainly. (1) The zero-shift rows reproduce
TF-043's pinned baseline **exactly** — the shipped scoring
did not move under this slice. (2) The shift hypothesis
(lean on the corpus's importance signal where the semantic
dimension is only an FTS proxy) **fails on the only corpus
with known answers**: it costs 0.037 precision and buys no
recall. The shifts remain recorded — labelled with their
measured cost, as the tuning surface's first experiment —
and TF-064's gate owns judging whether any nonzero shift
earns its place against the full Part III evidence.

## 5. TF-043's findings, as arithmetic

With every constant named, TF-043's GeoNames finding is a
derivation, not a mystery: `geonames_composite()` recomputes
a GeoNames record's composite from the named constants —
fixed fts 0.5 → semantic proxy 0.5/20 = 0.025, neutral tone
and preference 0.5, geonames recency proxy 0.75, importance
0.5 — under the `user_fact` row: **exactly 0.285**, under
`MIN_MEMORY_SCORE` (0.45) by construction. Naming it does not
fix it, and this slice does not retune it: whether GeoNames
should score by a different rule is a retrieval-design
decision for TF-064's gate (with TF-059). The ConceptNet
single-term firing stands likewise — corpus weight through
the same now-named arithmetic (the ferry edge's composite
0.46, pinned by TF-043's tests, still passes).

## 6. Docket

- **D-CFG-1 scoring share CLOSED** (TF-016 updated): the
  weights are explicit, censused, checker-pinned, and tunable
  through code. The entry itself stays open only for the
  unwired/unstruck shadow config keys.
- **D-PROF-3 narrowed again**: retrieval *scoring* is now
  per-profile tunable at the forge; retrieval top_k, history
  budgets, and mode temperatures stay flat for TF-098.

## 7. Proof grade

FIXTURE (TF-043's synthetic corpus, SQL path, this VM) for
the impact table; live-tree proofs for the census, checker,
and wiring. Live-corpus tuning NOT PROVEN; vector/hybrid
scoring NOT PROVEN (sentence-transformers absent, D-ENV-1;
TF-053 owns). No shipped scoring value was retuned.
