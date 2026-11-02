# The Wyrdforge Wing — Closing Doc (TF-144 retirements)

GENERATED from the filed data in `src/thoughtforge/wing_retirements.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.wing_retirements`. The checker (`wing_retirements_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-wing-retirements/1`
Retirements executed: 2 (TF-140 micro-RAG, TF-142 truth calibrator)
Disposition: **archived-in-place** (both)
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: the wing still cannot import — `research_data/src/wyrdforge/models` does not exist and both retired service shells import their types from that missing package (TF-006 / D-WING-1). No retirement had been executed: neither retired service file nor the wing's shared micro/truth test file carried any archived / retired marking, and the three files stood at exactly TF-006's censused sizes (micro_rag_pipeline.py 4,198 B / truth_calibrator.py 3,157 B / test_micro_rag_and_truth.py 2,173 B). The shipping tree holds no retired mechanism — no file under `src/thoughtforge/` outside the two verdict modules names MicroRAGPipeline or TruthCalibrator — so the retired code is the wing's, and only the wing's. The four neighbouring checkers (both verdicts, both TF-143 production landings) stood clean live.

## 2. The retirements, executed

| Mechanism | Verdict | Disposition | Successor |
|---|---|---|---|
| wyrdforge micro-RAG retrieval (MicroRAGPipeline: score() + assemble()) | TF-140 RETIRE | archived-in-place | the shipping KnowledgeForge (Part III) — the estate's sole retriever; no micro-RAG scoring or assembly code adopted or merged |
| wyrdforge truth calibrator (TruthCalibrator.evaluate()) | TF-142 RETIRE | archived-in-place | Part V's shipping enforcement (TF-093's surface, TF-049's per-claim law, TF-094's dispositions, TF-095's telling) — the estate's sole runtime truth law; no calibrator scoring adopted or merged |

### micro_rag (TF-140)

Measured ground: shipping defaults precision 0.8889 / recall 0.9630 / F1 0.9111 vs the micro-RAG spike's 0.3333 / 0.8889 / 0.4749 on TF-043's fixtures (MRR tied 1.0); the spike returns 45 items across nine queries against the shipping engine's 17, and five junk items on the negative query where the shipping engine returns nothing

Archived files:

- `research_data/src/wyrdforge/services/micro_rag_pipeline.py`
- `research_data/tests/test_micro_rag_and_truth.py`

Schemas kept unmarked as verdict evidence: `micro_context_packet.schema.json` / `retrieval_item.schema.json`

### truth_calibrator (TF-142)

Measured ground: Part V enforcement 29/38 as demanded vs the calibrator spike's 5/38 on TF-100's chaos corpus (clean controls only; rescued 0/12, refused-side 0/21, broken outputs detected 0/33, mean average 3.0 — it passes all 38); the records-fed packet run is row-for-row identical to the naked run

Archived files:

- `research_data/src/wyrdforge/services/truth_calibrator.py`
- `research_data/tests/test_micro_rag_and_truth.py`

Schemas kept unmarked as verdict evidence: `eval_case.schema.json` / `eval_result.schema.json`

Each archived file opens with the filed retirement banner verbatim (verdict, successor, do-not-revive terms); its original body stands byte-for-byte below the banner — every code marker the verdict checkers read is retained, re-proven live: 3/3 files bannered, code markers retained True, wing test markers retained True.

## 3. Why archived-in-place, not removed

- TF-140's and TF-142's checkers re-read the retired wing files live (weights / mode targets by AST, markers / threshold / the wing test's relational catch as text): deleting the files would falsify the verdicts' own proof, so removal is not an execution this estate can file honestly
- TF-142 files the test disposition in so many words: the wing's truth test 'is archived with the calibrator, not silently dropped'; TF-149 files the same rule for all retired code's tests
- the retired mechanisms' schemas are verdict evidence (the TF-140/TF-142 spikes are checked against them live) and TF-006 files the 14 schemas as the wing's most complete artefact — they stay unmarked, as shapes that execute nothing
- TF-150 owns the wing's provenance: archived code keeps its history readable in place; deleted code does not

## 4. Lessons harvested (the wing's closing lessons)

Per TF-006's rule that a retire verdict must still harvest. Every lesson is filed **harvested-not-adopted**: TF-144 adopts nothing.

- `typed_packet_sections` (TF-140): the typed packet sections: a context packet filed as mode + goal + a TruthPacket header (must_be_true / open_unknowns / forbidden_assumptions) + six typed sections (canonical / recent / bond / symbolic / code / contradiction), assembled in per-mode family order — a presentation shape over retrieved records that the shipping flat bundle does not have — Future: any future packet presentation over shipping bundles needs its own roadmap word.
- `explicit_token_budget` (TF-140): the explicit token budget: a packet carries a stated budget (default 900) charged per item's filed token_cost — budgeting as a filed, checkable number rather than an implicit top-k (on TF-043's fixtures the budget never bound before the per-family cap did: every packet used exactly 400) — Future: any future packet presentation over shipping bundles needs its own roadmap word.
- `contradiction_injection` (TF-140): contradiction injection: contradiction items are a first-class packet family, admitted by mode (factual_lookup / world_state / coding_task / repair_or_boundary) with a scoring penalty — contradictions presented to the model as filed items, not hidden from it — Future: any future packet presentation over shipping bundles needs its own roadmap word.
- `four_dimension_rubric` (TF-142): the four-dimension rubric as SCORED eval dimensions — factual_integrity / uncertainty_honesty / relational_safety / exploit_resistance, each 0..3 with a rationale — an offline eval-scorecard design; uncertainty_honesty in particular names a leg Part V's runtime surface does not judge — Future: any future eval scoring (a Part VII-style panel is the shape the estate already knows) needs its own roadmap word and must be built defect-free, not by reviving this scorer.
- `truth_packet_shape` (TF-142): the TruthPacket shape: must_be_true / open_unknowns / forbidden_assumptions filed per case, before the answer exists — the expectations an eval grades against are data authored in advance, not impressions after the fact — Future: any future eval scoring needs its own roadmap word and must be built defect-free, not by reviving this scorer.
- `eval_set_categories` (TF-142): the eval-set categories + modes as a design: a filed eval set with named categories and strict (higher threshold + zero-tolerance) vs balanced modes — filed here as a DESIGN lesson only; the retired code implemented only the balanced threshold, which is one of its nine measured defects, not part of the lesson — Future: any future eval scoring needs its own roadmap word and must be built defect-free, not by reviving this scorer.

## 5. What was NOT retired, and NOT touched

- TF-139's bond graph (ADOPT_AS_LAYER) and TF-141's persona compiler (MERGE) were adopted in TF-143 and are NOT retired: their wing files (`bond_graph_service.py`, `persona_compiler.py`, and the shared `memory_store.py`) carry no retirement banner — pinned live by this doc's checker (adopted files bannered: 0).
- All 14 wing schemas stand and parse; the specs and design docs stand as provenance (TF-150); the verdict modules' Spike* rebuilds stand as the verdicts' measurement evidence — none of them is the wing's retired code.
- No shipping behaviour changes; no README.md anywhere was touched; the wing's models package was not resurrected.

## 6. Docket

TF-144 discharges the retirements share of D-WING-1 (narrowed in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 itself stays OPEN for the TF-148 research hygiene and TF-149 wing-tests work it assigns — TF-144 is not the last Part VIII slice on the entry, and the TF-143 pattern holds: no new docket entry, totals do not move.

## 7. The probes

- `clean_retirement_filing_passes`: the filed retirements + lessons + archived set pass their validators
- `retirement_without_a_successor_is_named`: a retirement row with its successor emptied is named
- `lesson_without_a_source_is_named`: a lesson row with its source verdict emptied is named
- `banner_missing_from_file_is_named`: a file text with the banner stripped is named against the filed banner
- `adopted_file_in_archived_set_is_named`: an archived set carrying the bond graph service (an ADOPT verdict's file) is named
- `unknown_disposition_is_named`: a retirement row filing a disposition outside the filed vocabulary is named
- `duplicate_archived_path_is_named`: an archived set filing one path twice is named

## 8. Filed limits

- Archive grade only: archiving-in-place fences the retired code with its verdict and successor filed at its head; it does not make the wing importable (nothing here resurrects `wyrdforge.models`) and the archived tests still cannot run (TF-149 owns the wing's test disposition).
- The lessons are harvested, NOT adopted: any future packet presentation or eval scoring needs its own roadmap word, defect-free (TF-142's nine defects stay filed against the calibrator).
- TF-148 owns the `research_data/` fencing and TF-150 the wing's full provenance in this closing doc's line; this doc closes the RETIREMENTS, not the wing's whole Part VIII account (TF-145..TF-152 stand).
- No live model was read; live behaviour is NOT PROVEN (TF-173 class), never claimed.
