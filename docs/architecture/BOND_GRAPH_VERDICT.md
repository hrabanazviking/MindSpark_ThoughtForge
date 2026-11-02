# Bond Graph Verdict — TF-139

GENERATED from the filed data in `src/thoughtforge/knowledge/bond_graph_verdict.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.knowledge.bond_graph_verdict`. The checker (`bond_graph_verdict_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-bond-graph-verdict/1`
Verdict: **ADOPT_AS_LAYER**
Comparison rows: 9
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: a grep for `bond|relationship` over `src/thoughtforge/knowledge/*.py` + `src/thoughtforge/cognition/*.py` returns nothing (this verdict module excluded) — the shipping store has no bond or relationship graph, and its only `edges` are `conceptnet_edges` world-knowledge rows. `research_data/src/wyrdforge/models` does not exist (the wing cannot import, TF-006 / D-WING-1); the wing's 14 JSON schemas all parse, and its `BondGraphService` shell files complete logic over the missing types. TF-006's census files the bond graph UNIQUE — no shipping equivalent.

## 2. The comparison

| Dimension | Bond graph | Shipping store | Finding |
|---|---|---|---|
| `represented_subject` | a directional EDGE between two entities (entity_a + entity_b + domain): the relationship itself is the stored object | RECORDS about one side — user facts, user preferences, episodes, response patterns, the personality core; no record type names a pair | **unique** |
| `vector_state` | a ten-dimension bond vector per edge (trust, warmth, familiarity, devotion, attraction_affinity, safety, sacred_resonance, playfulness, vulnerability, initiative_balance -1..1), clamped 0..1 | per-record lifecycle scores (recency / half-life in TF-041's ladder, pattern-learning scores) — no per-relationship state vector exists | **unique** |
| `vows` | first-class Vow objects (text, kind, strength, pending|kept|strained|broken|released, witnesses, kept/broken event links) anchored to a bond | no vow, promise or commitment object in any record type — a promise can only be prose inside a fact or episode summary | **unique** |
| `hurts_scars_repair` | first-class Hurt objects plus edge scars: repair_debt, vow_strain, unresolved_hurts; a hurt moves the edge to REPAIRING by filed arithmetic | no hurt, scar or repair state — contradictions are handled as knowledge conflicts (knowledge/contradictions.py), never as relational damage with a repair debt | **unique** |
| `evidence_links` | BondEvidence.supporting_record_ids / contradiction_record_ids name record ids — by shape, the SHIPPING store's record ids: bond claims must be backed by stored records, and apply_event appends the source record id | the record store those ids live in: profile / episodic / pattern records with stable record_ids, plus KnowledgeForge citations | **layer-link** |
| `storage` | the service shell keeps dicts in memory; BondGraphSpec.md SUGGESTS a bond_edges SQL table (pair+domain unique) — no storage is committed | MemoryStore JSONL/YAML files per record kind + the SQLite knowledge DB, both committed and lifecycle-managed (TF-041/TF-045) | **different** |
| `runtime_excerpt` | excerpt(): seven filed lines (domain, status, the three derived indices, relational mode, emotional weather) + unresolved hurts + vows — sized for a small model to honour, per the spec's small-model tactics | retrieval returns records / knowledge chunks for the turn; no compact per-relationship excerpt exists for the prompt | **unique** |
| `update_law` | event-derived bounded updates ONLY: clamped integrator deltas from events, hurt severity shifts by filed table, status transitions at filed thresholds — no arbitrary mood swings (the spec's update law) | ingestion-contract appends (TF-042) + lifecycle ageing/pruning (TF-045): records change by being written or aged, never by relational events | **different** |
| `governance` | per-edge governance: review_required_for_large_shift + synthetic_claim_guard, plus the spec's overclaim rules (no asserted shared memory without a supporting record, no claimed vow without a vow object, open hurts never silently skipped) | estate-wide governance: the TF-042 ingestion contract, protected identity records, citation discipline — real governance, but over RECORDS, not over relational claims | **overlap** |

## 3. The verdict: ADOPT_AS_LAYER

### Reasons

- unique concept: five of the nine comparison dimensions are filed unique and none is filed redundant — the shipping store has no relationship edge, no bond vector, no vow, no hurt / repair state and no relational excerpt, so there is nothing to merge INTO and nothing whose loss retirement could justify
- not a merge: the shipping MemoryStore / KnowledgeForge answer 'what is known' (records and world knowledge); the bond layer answers 'how entities stand with each other' (edges and their state) — folding edge state into record types, or records into edges, would corrupt both shapes
- not retire: the concept is complete as a design — spec, three schemas and a full service shell — and its absence is the estate's known gap for long-term relational continuity (TF-006: a RETIRE verdict must still harvest the concept; adopting it as a layer IS the harvest, with a home)
- the layer link already exists in the shapes: bond evidence names shipping-store record ids, so the layer can stand BESIDE the store referencing its records as evidence without duplicating a single fact
- the spike proves the shapes run: dataclasses rebuilt from the three schemas carry the service shell's operations faithfully (exact severity shifts, clamps, thresholds and excerpt lines) on synthetic fixtures, deterministically

### Conditions

- adopt AS A LAYER beside the shipping store, referencing store record ids as evidence — never duplicating facts, never a second record store
- adoption is NOT done in TF-139: TF-143 owns rebuilding wyrdforge.models from the 14 schemas and landing production code under the shipping tree's discipline; TF-146 owns the WYRD-consumable bond bridge (live work — the verdict is adopt); TF-149 owns the wing's tests
- TF-139 changes NO shipping behaviour and NO research-wing code: the spike lives in this verdict module only, namespaced Spike*, and is not the missing wyrdforge.models package
- the derived indices' weights are UNSPECIFIED in BondGraphSpec.md — the spike's equal weights are a spike assumption; TF-143 must file a weighting decision with evidence before any production use
- the schemas win over the spec where they drift (hurt carries notes + a mythic severity in the schema, not the spec's repair_state / residual_sensitivity) — TF-143 rebuilds from the schemas, per TF-006

## 4. The working spike

Spike dataclasses rebuilt from the three schemas (BondEdge / Vow / Hurt + BondVector, BondScars, BondChronology, BondEvidence, BondConstraints, BondActiveModes, BondGovernance; enums BondDomain / BondStatus / VowState), namespaced `Spike*` in the verdict module — NOT the missing `wyrdforge.models` package and NOT production adoption. `SpikeBondGraph` re-implements the service shell faithfully: the exact severity shifts {low: 0.1, medium: 0.2, high: 0.35, mythic: 0.5}, repair_debt = min(1, debt + shift), safety = max(0, safety - shift/2), status -> REPAIRING on a hurt, vector clamps 0..1, status -> ACTIVE at trust >= 0.55 and warmth >= 0.55, and the service's excerpt line shapes. The derived indices use EQUAL weights — BondGraphSpec.md files the index shapes but not the weights; equal weights are a filed spike assumption. Fixtures are synthetic invented entities (TF-017): `user:emberwick-keeper` / `persona:emberwick-lantern`.

Spike lifecycle (synthetic): edge formed -> vow filed -> care event -> high hurt -> excerpt.

- status after event: active
- status final: repairing
- trust 0.6 / warmth 0.6
- repair_debt 0.35 / safety 0.725 (high hurt: +0.35 debt, -0.175 safety)
- closeness 0.4375 / sacred 0.5333 / rupture 0.2083
- evidence records 1 / unresolved hurts 1

The excerpt, verbatim:

```
domain=companion
status=repairing
closeness_index=0.44
sacred_bond_index=0.53
rupture_index=0.21
relational_mode=builder
emotional_weather=calm
unresolved_hurts=hurt:emberwick-001
vows=promise:pending:keep the lantern lit through winter
```

## 5. The probes

- `clean_verdict_and_rows_pass`: the filed verdict + rows pass their validators
- `verdict_without_a_reason_is_named`: a verdict filing with its reasons emptied is named
- `comparison_row_missing_a_side_is_named`: a comparison row with its shipping side emptied is named
- `spike_hurt_shift_wrong_is_named`: a severity map with the high shift perturbed (0.35 -> 0.30) is named against the filed shifts
- `spike_index_wrong_is_named`: an expected filing with the spike's closeness perturbed is named against the computed indices
- `schema_field_missing_from_spike_is_named`: a field map with vow_strain dropped from BondScars is named against the schemas
- `unknown_verdict_value_is_named`: a verdict value outside the roadmap's three is named

## 6. Docket

TF-139 discharges the bond-graph share of D-WING-1 (verdict decided on TF-006's census + this working spike, narrowed in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 itself stays OPEN for the TF-140..TF-142 verdicts and the TF-143 rebuild / TF-149 tests it assigns.

## 7. Filed limits

- Spike grade only: production adoption is NOT done and NOT PROVEN here — TF-143 owns rebuilding `wyrdforge.models` from the 14 schemas and landing production code; TF-146 owns the WYRD-consumable bridge; TF-149 owns the wing's tests.
- The derived indices' weights are unspecified in BondGraphSpec.md; the spike's equal weights are an assumption TF-143 must replace with a filed decision.
- No live model was read; whether a live system honours a bond excerpt is NOT PROVEN (TF-173 class).
- Where the spec and the schemas drift (the hurt shape), the schemas win — TF-006 ruled them the wing's most complete artefact.
