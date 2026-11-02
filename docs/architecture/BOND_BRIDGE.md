# The Bond-Graph Bridge — bonds as WYRD-consumable data (TF-146)

GENERATED from the filed data in `src/thoughtforge/knowledge/bond_bridge.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.knowledge.bond_bridge`. The checker (`bond_bridge_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-bond-bridge/1`
Source format: `thoughtforge-bond-graph/1` (TF-143)
Antecedent: TF-139 **ADOPT_AS_LAYER** — the bridge is live work
Exported sections: 9
Reservations: 4
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: **no bridge existed** — no bridge module under `src/thoughtforge/knowledge/`, no bridge doc, and the production `BondGraph`'s only outward shape was `excerpt()`'s human-readable lines: nothing a WYRD-side consumer could consume as data (no ids as fields, no vector, no scars, no vow/hurt records, no evidence links, no versioned format). The antecedent stood clean live: TF-139's verdict ADOPT_AS_LAYER (`bond_graph_verdict_problems()` []), TF-143's production graph clean (parity excerpt 9 lines), TF-145's boundary clean with the bridge row filing `not-built-tf146`, and TF-132's panel PASS (18 cases / 9 consistent / 8 contradicted / 1 honest-unanswerable).

## 2. The export shape

`export_bond(graph, bond_id)` builds the export FROM the production `BondGraph` (TF-143): every value is read from the graph — the edge's fields, the edge's own index methods, the graph's vow/hurt stores, and `BondGraph.excerpt()` itself for the excerpt lines. No bond logic is duplicated in the bridge. The export is plain data (dicts / lists / strings / numbers / null), JSON-serialisable as it stands, under the versioned format above.

| Section | Source in the production graph | Carries |
|---|---|---|
| bond_id / entity_a / entity_b / domain / status | BondEdge's own fields (bond_id, entity_a, entity_b, domain, status) | which bond this is, the two entities it stands between (ThoughtForge-side ids, filed as references — entity identity in the world stays WYRD-side, TF-145), its domain and its lifecycle status |
| vector | BondVector, all ten dimensions | the full relational vector exactly as the production edge holds it — no dimension dropped, none re-scaled |
| indices (closeness / sacred / rupture) | BondEdge.closeness_index() / sacred_bond_index() / rupture_index() — the edge's own methods, under TF-143's filed equal weights | the three derived indices a consumer reads without re-deriving them; the pure validator recomputes them from the exported vector + scars so the filing is checkable data-side |
| scars (repair_debt / vow_strain / unresolved_hurts) | BondScars | the repair state: how much repair debt the bond carries, its vow strain, and the ids of its unresolved hurts |
| vows | the graph's Vow records for this bond, in the graph's filing order | every vow on the bond as data — id, kind, state, text, strength, the store record it was created from, witnesses and its kept / broken event links |
| hurts | the graph's Hurt records for this bond, in the graph's filing order | every hurt on the bond as data — id, kind, severity, notes, and the source event id it answers to |
| evidence (supporting_record_ids / contradiction_record_ids) | BondEvidence | the layer link itself: the shipping-store record IDS the bond's state rests on — ids only, never the records (the facts stay in the store, TF-139's layer link) |
| excerpt | BondGraph.excerpt(bond_id), called on the production graph — never re-rendered here | TF-145's filed source shape, carried verbatim as data lines beside the structured sections, so a consumer that only reads lines and a consumer that reads fields see the same bond |
| constraints / active_modes / governance / chronology | BondConstraints / BondActiveModes / BondGovernance / BondChronology (timestamps as ISO strings or null) | the bond's filed constraints, its active relational mode + emotional weather, its governance flags, and its chronology — the context a consumer needs to honour the bond, with no wall-clock value ever filed as a metric |

## 3. Round-trip proof (synthetic fixtures, TF-017)

A production `BondGraph` built over TF-139/TF-143's Emberwick parity fixtures (edge → vow → care event → high hurt, exactly `run_bond_parity`'s lifecycle) is exported, and the export is proven to carry the filing's claims:

- bond: bond:emberwick-keeper-companion (user:emberwick-keeper ↔ persona:emberwick-lantern)
- domain companion / status repairing
- vector: 10 dims (trust 0.6 / warmth 0.6 / safety 0.725)
- indices: closeness 0.4375 / sacred 0.5333 / rupture 0.2083 — TF-143's parity values, unmoved
- scars: repair_debt 0.35 / unresolved hurts 1
- vows 1 / hurts 1 as full records bound to this bond
- evidence links: 1 supporting / 0 contradiction record ids — ids only, never records
- excerpt lines: 9, verbatim:

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

The export JSON-round-trips unchanged (`json.loads(json.dumps(export)) == export`, pinned by the clean probe and the tests), and a second build exports identically except its chronology stamps — timestamps are values a consumer may read, never metrics this filing pins.

## 4. What the bridge does NOT carry

- **no_world_state** (reserved to wyrd): no canonical world state — no entity attributes, locations or relations of the lived world; the export's entity ids are ThoughtForge-side references only (TF-145: canonical_world_state + entity_identity_in_world are WYRD's)
- **no_canon_data** (reserved to neither): no canon data — nothing from data/, no persona material, no dataset content (TF-145: canon_data is neither side's; TF-017: fixtures synthetic only)
- **no_store_facts** (reserved to thoughtforge-store): no store facts — evidence carries the record ids only; the records themselves stay in the shipping store and are never duplicated into the export (TF-139's layer link, TF-143's layer discipline)
- **no_live_claim** (reserved to neither): no claim of live consumption — the bridge files a consumable SHAPE; whether a live WYRD-side system consumes it is NOT PROVEN (TF-173 class), never claimed

The reservation is enforced, not just filed: the pure validator names any export carrying a forbidden key (`world_state` / `world` / `canon` / `canon_data` / `facts` / `store_records`) at any depth — probe `forbidden_world_state_is_named`.

## 5. TF-145's filed status, handled in the open

TF-145's boundary doc files this bridge as its one not-built row (`bond_excerpt_bridge`, status `not-built-tf146`) — and **it stands as filed**. The estate's precedent is uniform: TF-143 did not rewrite TF-139's verdict module (which still files TF-143 as the rebuild's owner, after the rebuild landed) and TF-144 did not rewrite TF-140/TF-142's verdict modules; a filed slice's doc records what was true and measured when it was filed, and the slice that changes the estate files the change in its own module — this one. (TF-137's in-the-open updates were to `benchmarks/harness.py`, a living registry of the current estate, not a filed slice doc.) TF-151 / TF-152 re-measure the estate with the verdicts landed. This module's checker pins TF-145's row still naming TF-146 as the bridge's owner — the antecedent the bridge stands on — and the bridge was shaped to leave TF-145's live measurements true (no WYRD-named file, no live-client token, no consumer name under `src/`).

## 6. Docket

TF-146 has **no share** of TF-016's D-WING-1: the entry's current text (checked live for this slice) assigns the verdicts (TF-139..TF-142), the adoptions (TF-143), the retirements (TF-144), research hygiene (TF-148) and the wing's tests (TF-149) — the bridge's ownership is filed in TF-139's verdict conditions, TF-143's filing and TF-145's boundary row, not in D-WING-1. Per the TF-145 pattern's rule (a slice touches the docket only for its own filed share), this slice does not edit the docket and adds no entry; TF-016 totals stand unchanged (137 entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 CLOSED-BY-VERDICT), filed in the open here.

## 7. The probes

- `clean_export_passes`: the fixture export + the filed bridge / reservation rows pass their validators, and the export JSON-round-trips unchanged
- `missing_bond_id_is_named`: an export with its bond_id emptied is named
- `index_mismatch_is_named`: an export whose closeness is perturbed is named against the value recomputed from its own vector + scars
- `excerpt_drift_is_named`: an export whose excerpt status line is perturbed is named against its structured sections
- `forbidden_world_state_is_named`: an export carrying a world_state key is named — the world state is WYRD-side, never the bridge's to file
- `vow_bound_to_another_bond_is_named`: an export whose vow is bound to a different bond is named
- `vector_out_of_range_is_named`: an export whose trust files 1.4 is named

## 8. Filed limits

- Fixture grade only: the round-trip proof runs on synthetic Emberwick fixtures (TF-017); no live WYRD consumer was read — a live system consuming the bridge is NOT PROVEN (TF-173 class), never claimed.
- The export is a shape, not a transport: no client, no feed, no protocol session is built here (TF-145 files transport as the consumer side's; Part IX owns bindings).
- The indices in an export are the production edge's own, under TF-143's filed equal-weights decision — the bridge files no weighting of its own.
- No canon data was touched; no README.md anywhere was touched.
