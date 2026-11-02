# Micro-RAG Verdict — TF-140

GENERATED from the filed data in `src/thoughtforge/knowledge/micro_rag_verdict.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.knowledge.micro_rag_verdict`. The checker (`micro_rag_verdict_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-micro-rag-verdict/1`
Verdict: **RETIRE**
Comparison rows: 9
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: the wing cannot import — `research_data/src/wyrdforge/models` does not exist and `micro_rag_pipeline.py` imports its types from that missing package (TF-006 / D-WING-1). The pipeline itself is a reranker + assembler, not a retriever: `score()` (a nine-term weighted sum over a substring similarity; recency is the constant 0.5) and `assemble()` (per-family top 5, token budget 900, typed packet sections, NO score threshold, NO corpus, NO index, NO chunking, NO candidate gathering in code). TF-006's census files micro-RAG OVERLAP in purpose vs the shipping engine, different shape. TF-043's fixtures are the measuring ground both mechanisms run over.

## 2. The comparison

| Dimension | Micro-RAG | Shipping engine | Finding |
|---|---|---|---|
| `candidate_gathering_and_indexing` | none in code: no corpus, no index, no chunking — assemble() receives candidates_by_family pre-gathered by an unnamed caller, items arriving whole at whatever scale the caller filed them; the spec's gather step and its seven index families (atom / episode / canonical / bond / symbolic / code / graph) are prose only | the forge gathers itself over its own corpus, which is the index: entities, pre-chunked reference_chunks, conceptnet edges and geonames places in FTS5-indexed SQLite tables, searched by bm25 plus a ConceptNet pass and a GeoNames pass | **different** |
| `ranking_signal` | a nine-term weighted sum (filed in SCORE_WEIGHTS) over a substring similarity — the fraction of query terms appearing anywhere in text + lexical terms; term frequency and rarity count for nothing, and recency is the constant 0.5 for every item | bm25 term statistics from FTS5 fused per source through scoring.py's named weights (fts proxy, source recency proxies, popularity / weight as importance), so rare and repeated terms both move the score | **different** |
| `score_threshold_and_negative_queries` | no score floor exists in the pipeline: assemble() admits each family's top 5 however weak the match — on TF-043's negative query (Q6, similarity 0 for every item) it still returns the five highest-confidence entities, scoring precision 0 / recall 0 under TF-043's convention | MIN_MEMORY_SCORE 0.45 at the defaults: the negative query returns nothing and scores precision 1 / recall 1 — correctly returning nothing is a success (TF-043's convention) | **shipping-wins** |
| `measured_precision_and_f1` | mean precision 0.3333 / mean F1 0.4749 on TF-043's fixtures (run_micro_spike, filed below): the per-family cap fills every packet with exactly five items — 45 returned across the nine queries, only 15 of them known answers | mean precision 0.8889 / mean F1 0.9111 at the defaults (TF-043's pinned baseline, re-measured live by this module), returning 17 records across the nine queries, 14 of them known answers; even the open setting (min_score 0) files 0.8148 / 0.8815 | **shipping-wins** |
| `measured_recall` | mean recall 0.8889: every answered query's known answers are found (8 perfect-recall queries), but the negative query's junk return scores recall 0 and drags the mean below the shipping engine's | mean recall 0.9630 at the defaults: eight queries at recall 1.0 and the geonames query at 0.6667 (the filed threshold defect, next row); the open setting files recall 1.0 | **shipping-wins** |
| `place_records_geonames` | the geonames query (Q9) files recall 1.0 / precision 0.6: with no threshold, the GeoNames record enters the packet on its substring similarity like any other item | Q9 files recall 0.6667 at the defaults: a GeoNames record's composite is the fixed 0.285 (TF-043's filed finding), which can never pass the 0.45 threshold — a defect of the shipping threshold, already filed by TF-043 and owned by the scoring line, not evidence the micro mechanism ranks better | **micro-wins** |
| `packet_assembly` | a typed MicroContextPacket: mode + goal + a TruthPacket header (must_be_true / open_unknowns / forbidden_assumptions) + six typed sections (canonical / recent / bond / symbolic / code / contradiction), assembled in MODE_TARGETS family order — a presentation shape the shipping bundle does not have, unmeasured by these fixtures | a flat MemoryActivationBundle: the ranked activated records with their scores and citations, no mode, no truth header, no typed sections | **different** |
| `token_budgeting` | an explicit packet budget (default 900) charged per item's filed token_cost (schema default 80): on the fixtures every packet uses exactly 400 (5 x 80) — the budget never binds before the per-family cap does, and nothing trims a weak item | no token budget: a global top_k 10 with per-source sub-caps (reference top_k//2, conceptnet top_k//3, geonames top_k//5 in sql_retrieve) plus the score threshold doing the trimming — 17 records returned across the nine fixture queries | **different** |
| `citation_handling` | each item carries item_id + a free-text source_ref string (e.g. 'bond_graph' in the wing's test); no citation is built, checked or attached to the packet beyond those two fields | each record carries its record_id and the forge's citation payload (the ConceptNet triple fields ride along, TF-048), under the estate's citation discipline — both sides carry provenance ids, only the shipping side builds citations from them | **overlap** |

## 3. The verdict: RETIRE

### Reasons

- measured loss on every aggregate but one: on TF-043's fixtures the shipping engine at its defaults files precision 0.8889 / recall 0.9630 / F1 0.9111 against the micro-RAG spike's 0.3333 / 0.8889 / 0.4749, and mean reciprocal rank ties at 1.0 — the micro mechanism wins no aggregate, and even the shipping engine's open setting (threshold removed: 0.8148 / 1.0 / 0.8815) beats it on all three, so the loss is in its ranking and filling, not in the threshold alone
- the mechanism's defects are structural, not tuning: no score floor (the negative query returns five junk items and scores 0/0 where the shipping engine returns nothing and scores 1/1), substring similarity with no term statistics (frequency and rarity count for nothing), recency filed as a constant, and a per-family cap that always fills — 45 items returned across nine queries against the shipping engine's 17
- not adopt: TF-006 files micro-RAG OVERLAP in purpose — the estate already has a retriever, and the measured one is the shipping engine; adopting a second, worse retriever beside it buys a competing ranking law, not a layer (unlike TF-139's bond graph, there is no unique retrieval concept here to house)
- not a merge: folding the micro scoring sum into the shipping fusion would replace bm25 term statistics with substring matching and repeal the threshold — it would regress TF-043's pinned baseline by construction; the two ranking laws cannot be averaged into a better one on this evidence
- its one measured win is not its merit: Q9's recall (1.0 vs 0.6667) comes from having no threshold at all — the same absence that fails Q6 — while the shipping side's miss is TF-043's already-filed GeoNames threshold defect, owned by the shipping scoring line; fixing that defect does not need this mechanism

### Conditions

- RETIRE the micro-RAG retrieval mechanism: the shipping KnowledgeForge remains the estate's sole retriever; no micro-RAG scoring or assembly code is adopted or merged into the shipping tree
- retirement is NOT executed in TF-140: TF-144 owns executing each RETIRE verdict (code removed or archived-in-place, lessons written into the wing's closing doc); TF-143 owns adoptions — TF-140 files no adoption for it to land; TF-149 owns the wing's tests
- the harvest, filed for TF-144's closing doc per TF-006 (a retire verdict must still harvest): the typed packet sections, the explicit token budget and contradiction injection are presentation lessons recorded here — they are NOT adopted by this verdict, and any future packet presentation over shipping bundles needs its own roadmap word
- TF-140 changes NO shipping behaviour and NO research-wing code: the spike lives in this verdict module only, namespaced Spike*, and is not the missing wyrdforge.models package
- the fixture adaptation is filed in this module's docstring (schema defaults throughout, all rows as the canonical family, factual_lookup mode): the verdict stands on those filed terms, and a different candidate gathering — which the wing never implemented — is a different mechanism than the one measured

## 4. The working spike + the measurement

Spike dataclasses rebuilt from the two schemas (RetrievalItem; RankedCandidate / TruthPacket / MicroContextPacket + the QueryMode enum), namespaced `Spike*` in the verdict module — NOT the missing `wyrdforge.models` package and NOT production. `SpikeMicroRAG` re-implements the pipeline faithfully: the exact MODE_TARGETS per mode, the nine filed score weights (similarity 0.24, task_relevance 0.24, support_quality 0.14, scope_match 0.12, recency 0.08, contradiction_penalty -0.10, token_cost_penalty -0.04, novelty_bonus 0.04, bond_fit 0.08), per-family cap 5, budget 900, the section mapping. The fixture adaptation is filed in the module docstring: TF-043's corpus verbatim as 106 items (item_id = TF-043's bundle id; confidence = entity popularity, else the schema default 0.5; lexical_terms and facets empty, as the wing's own test items file them; token_cost the schema default 80), all as the canonical family, mode factual_lookup. The shipping engine runs through TF-043's own `evaluate_retrieval`; the spike's returns are scored by TF-043's own `score_query`.

Aggregates (TF-043's nine queries):

- shipping, defaults (min_score 0.45): precision 0.8889 / recall 0.963 / F1 0.9111 / MRR 1.0 / perfect-recall queries 8 / total returned 17
- shipping, open (min_score 0): precision 0.8148 / recall 1.0 / F1 0.8815 / MRR 1.0 / perfect-recall queries 9 / total returned 21
- micro-RAG spike: precision 0.3333 / recall 0.8889 / F1 0.4749 / MRR 1.0 / perfect-recall queries 8 / total returned 45

Per query (shipping defaults vs micro-RAG spike):

| Query | Shipping returned | P / R | Micro returned | P / R |
|---|---|---|---|---|
| `Q1` Lantern Hollow ferry | QF4301, 4301, ferry | 0.6667 / 1.0 | QF4301, 4301, Lantern Hollow, QF4311, 4311 | 0.4 / 1.0 |
| `Q2` Bracken Mill grain flour | QF4302, 4302 | 1.0 / 1.0 | QF4302, 4302, mill, QF4312, Bracken Ford | 0.4 / 1.0 |
| `Q3` Alder Observatory telescope stars | QF4303, 4303 | 1.0 / 1.0 | QF4303, 4303, QF4313, QF4301, QF4302 | 0.4 / 1.0 |
| `Q4` Kestrel lighthouse beacon | QF4304 | 1.0 / 1.0 | QF4304, QF4301, QF4302, QF4303, QF4305 | 0.2 / 1.0 |
| `Q5` river crossing | QF4305, QF4301, ferry | 0.6667 / 1.0 | QF4301, QF4305, 4301, ferry, QF4302 | 0.4 / 1.0 |
| `Q6` volcano eruption crater | (nothing) | 1.0 / 1.0 | QF4301, QF4302, QF4303, QF4304, QF4305 | 0.0 / 0.0 |
| `Q7` ferry crosses river | ferry | 1.0 / 1.0 | ferry, QF4301, QF4305, QF4311, 4301 | 0.2 / 1.0 |
| `Q8` ferry house inn | QF4311, 4311, ferry | 0.6667 / 1.0 | QF4311, 4311, QF4301, QF4304, QF4313 | 0.4 / 1.0 |
| `Q9` Lantern Hollow | QF4301, 4301 | 1.0 / 0.6667 | QF4301, 4301, Lantern Hollow, QF4302, QF4303 | 0.6 / 1.0 |

The two decisive queries: Q6 (negative — 'volcano eruption crater') — the shipping engine returns nothing and scores 1/1; the spike returns the five highest-confidence entities at similarity 0 and scores 0/0. Q9 (geonames — 'Lantern Hollow') — the spike files recall 1.0 against the shipping defaults' 0.6667: the filed GeoNames threshold defect (TF-043), the spike's one win, bought by the same missing threshold that fails Q6. Every micro packet used exactly 400 of its 900 budget (5 items x the default token_cost 80).

## 5. The probes

- `clean_verdict_and_rows_pass`: the filed verdict + rows pass their validators
- `verdict_without_a_reason_is_named`: a verdict filing with its reasons emptied is named
- `comparison_row_missing_a_side_is_named`: a comparison row with its shipping side emptied is named
- `spike_score_weight_wrong_is_named`: a weight map with the similarity weight perturbed (0.24 -> 0.20) is named against the filed weights
- `measured_metric_wrong_is_named`: an expected filing with the micro mean precision perturbed is named against the measured aggregate
- `schema_field_missing_from_spike_is_named`: a field map with token_cost dropped from RetrievalItem is named against the schemas
- `unknown_verdict_value_is_named`: a verdict value outside the roadmap's three is named

## 6. Docket

TF-140 discharges the micro-RAG share of D-WING-1 (verdict decided by measurement on TF-043's fixtures, narrowed in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 itself stays OPEN for the TF-141..TF-142 verdicts and the TF-143 rebuild / TF-144 retirements / TF-149 tests it assigns. The retirement this verdict files is executed by TF-144, not by TF-140.

## 7. Filed limits

- Spike grade only: the spike is a faithful rebuild for measurement, NOT production and NOT an adoption — TF-144 owns executing the retirement; TF-149 owns the wing's tests.
- The verdict covers the retrieval mechanism. The packet's typed sections, token budget and contradiction injection are filed as harvested lessons for TF-144's closing doc — unmeasured by these fixtures and NOT adopted here.
- Fixture grade only: TF-043's corpus is synthetic; no live model was read and live retrieval quality is NOT PROVEN (TF-173 class).
- The candidate gathering the spec proposes was never implemented in the wing; a different gathering would be a different mechanism than the one measured.
