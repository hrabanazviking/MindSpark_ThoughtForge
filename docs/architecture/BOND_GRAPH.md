# Bond Graph — TF-143 (production layer)

GENERATED from the filed data in `src/thoughtforge/knowledge/bond_graph.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.knowledge.bond_graph`. The checker (`bond_graph_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-bond-graph/1`
Verdict implemented: **ADOPT_AS_LAYER** (TF-139)
Probes: 7/7

## 1. What landed

Production dataclasses rebuilt from the bond_edge / vow / hurt schemas (BondEdge / Vow / Hurt + BondVector, BondScars, BondChronology, BondEvidence, BondConstraints, BondActiveModes, BondGovernance; enums BondDomain / BondStatus / VowState) and a production `BondGraph` service carrying the wing shell's operations faithfully (add_edge / add_vow / add_hurt / apply_event / excerpt). The layer stands BESIDE the shipping store: evidence references shipping-store record ids, no facts are duplicated, and it is not a second record store. The research wing's missing `wyrdforge.models` package is NOT resurrected — TF-143 lands production code in the shipping tree only; the wing stays as-is for TF-148/TF-149.

## 2. The weighting decision (TF-139's condition)

Decision: **equal_weights**

Evidence:

- BondGraphSpec.md files each index only as a weighted_mean over its terms — re-read for TF-143, the spec carries no numeric weight for any index term (its weight tables are event_weights for event ingestion, a different mechanism the service shell does not use for indices)
- the wing's BondGraphService computes no index itself — it calls edge.closeness_index() etc. on the missing models, so no implementation weighting exists to inherit
- TF-139's spike used equal weights as a filed assumption and filed deterministic metrics under them (closeness 0.4375 / sacred 0.5333 / rupture 0.2083 on the Emberwick fixtures); equal weights reproduce those metrics exactly, keeping the verdict's measurement as the production baseline instead of silently re-basing it

Limits:

- equal weights are a default under spec silence, NOT a measured optimum — no evidence in the estate ranks one index term above another
- a future slice with real evidence (a measured corpus, a spec amendment) may re-rule the weights; until then any unequal weighting would be invented tuning

## 3. Parity with TF-139's spike

The production service run over TF-139's synthetic Emberwick fixtures reproduces the spike's filed metrics exactly (equal weights — the filed decision — so no index moves):

- status after event: active
- status final: repairing
- trust 0.6 / warmth 0.6
- repair_debt 0.35 / safety 0.725 (high hurt: +0.35 debt, -0.175 safety)
- closeness 0.4375 / sacred 0.5333 / rupture 0.2083
- excerpt lines: 9

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

## 4. The probes

- `clean_production_and_weighting_pass`: the production run + weighting filing pass their validators
- `severity_shift_wrong_is_named`: a severity map with the high shift perturbed (0.35 -> 0.30) is named against the filed shifts
- `index_wrong_is_named`: an expected filing with the parity closeness perturbed is named against the computed indices
- `schema_field_missing_is_named`: a field map with vow_strain dropped from BondScars is named against the schemas
- `weighting_without_evidence_is_named`: a weighting filing with its evidence emptied is named
- `edge_vector_out_of_range_is_named`: an edge whose trust files 1.4 is named by the edge validator
- `hurt_unknown_severity_is_named`: a hurt filing an unknown severity is named

## 5. Filed limits

- Fixture grade only: parity is proven on synthetic Emberwick fixtures (TF-017); no live model was read — whether a live system honours a bond excerpt is NOT PROVEN (TF-173 class).
- Equal weights are a filed default under spec silence, not a measured optimum (see §2).
- TF-146 owns the WYRD-consumable bond bridge; TF-149 owns the wing's tests. TF-140/TF-142's RETIRE verdicts land nothing here — TF-144 executes them.
