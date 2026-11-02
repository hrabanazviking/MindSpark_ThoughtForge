# The WYRD Boundary — ThoughtForge vs the WYRD Protocol (TF-145)

GENERATED from the filed data in `src/thoughtforge/wyrd_boundary.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.wyrd_boundary`. The checker (`wyrd_boundary_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-wyrd-boundary/1`
Ownership rows: 12 (ThoughtForge 6 / WYRD 3 / neither 3)
Interface rows: 5 (in production 2 / fixture grade 2 / not built 1)
Consumers named: 2
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: **no WYRD interface code exists in the shipping tree** — a grep for WYRD across `src/thoughtforge/` finds the word only in filings about this boundary (TF-139's verdict module and TF-143's production bond graph, both naming TF-146 as the bridge's owner). There is no WYRD client, no protocol implementation, no live feed, and no file under `src/` named for WYRD outside those filings (measured live by this doc's checker: 0 interface files, 0 live-client tokens). **WYRD itself lives outside this repo**, and this doc claims no live WYRD integration anywhere — live behaviour is NOT PROVEN (TF-173 class), never claimed. The evidence the boundary rests on stood clean live: TF-132's panel re-runs here at 18 cases / 9 consistent / 8 contradicted / 1 honest-unanswerable, verdict PASS; TF-143's bond parity excerpt files 9 lines.

## 2. What each side owns

| Subject | Owner | Evidence in this repo |
|---|---|---|
| the cognition turn — persona, memory activation, scaffolding, candidate scoring and the citation gate that turn a question into an answer | **thoughtforge** | src/thoughtforge/cognition/core.py (ThoughtForgeCore) + docs/architecture/CORE_LOOP.md (Part II) |
| the knowledge store — what is known: the filed record types and their retrieval | **thoughtforge** | src/thoughtforge/knowledge/store.py + models.py + forge.py (Part III) + TF-139's comparison rows + TF-144's successor filing |
| the bond layer — how entities stand with each other: edges, vows, hurts, repair state and the relational excerpt | **thoughtforge** | src/thoughtforge/knowledge/bond_graph.py (TF-143) + bond_graph_verdict.py (TF-139 ADOPT_AS_LAYER) |
| persona compilation — canonical records compiled into the expectation a turn is steered by | **thoughtforge** | src/thoughtforge/cognition/persona_compiler.py (TF-143) + persona_compiler_verdict.py (TF-141) |
| runtime truth enforcement — the per-claim law an answer is judged under as it is produced | **thoughtforge** | TF-142's verdict filing + TF-144's successor filing in src/thoughtforge/wing_retirements.py |
| the consistency MEASUREMENT — the instrument that scores answers against a stated world state | **thoughtforge** | benchmarks/wyrd_panel.py (TF-132: 18 cases, verdict PASS, its rule the_panel_feeds_tf145_it_does_not_decide_it) |
| the canonical lived world state — the entities, attributes, locations and relations of the world as it actually stands | **wyrd** | the roadmap's law (docs/MINDSPARK_THOUGHTFORGE_ROADMAP_2026-10-30.md: WYRD is the world-model neighbour) + TF-132's filed world standing in for it as a synthetic fixture |
| the world's change log — the timeline of changes by which the canonical state moves | **wyrd** | TF-132's WORLD_EVENTS + world_state_at() replay shape in benchmarks/wyrd_panel.py, standing in for the lived log this repo does not hold |
| entity identity in the world — which entity in the lived world a name or id denotes | **wyrd** | TF-139/TF-143's evidence-link shape (BondEvidence names record ids) + TF-132's panel-local entity ids, filed as fixture ids only |
| embodiment and transport — logging in, moving, perceiving and messaging inside a lived world | **neither** | the roadmap's Part IX filing (bindings are cognition only) + no embodiment code anywhere under src/ |
| Volmarr's canon data — the datasets and persona material under data/ | **neither** | the roadmap's canon law + TF-017's synthetic-fixtures rule, filed across the estate |
| proof of live behaviour — whether a live system actually stays true to the world in use | **neither** | TF-132's filed limits (no live model was read; live world-state behaviour NOT PROVEN, TF-173) |

### cognition_turn — thoughtforge

ThoughtForgeCore.think() / think_stream() (src/thoughtforge/cognition/core.py) is the estate's cognition surface; Part II files its loop (docs/architecture/CORE_LOOP.md)

### knowledge_store — thoughtforge

MemoryStore / KnowledgeForge (Part III) answer 'what is known'; TF-140's verdict leaves KnowledgeForge the estate's sole retriever (TF-144)

### bond_layer — thoughtforge

TF-139 adopted wyrdforge's bond graph AS A LAYER beside the store and TF-143 landed it as production BondGraph; the layer references store record ids as evidence and duplicates no facts

### persona_compilation — thoughtforge

TF-141's MERGE landed in TF-143 beside TF-030: CompiledPersonaExpectation extends PersonaExpectation, approval checked first, defects not imported

### truth_enforcement — thoughtforge

Part V's shipping enforcement is the estate's sole runtime truth law; TF-142 retired the wing's calibrator against it and TF-144 executed that retirement

### consistency_measurement — thoughtforge

TF-132's WYRD-shaped panel is a ThoughtForge-side measuring instrument over a filed synthetic world; measuring consistency is cognition-side work — the panel is not a world model and files that it feeds this boundary, it does not decide it

### canonical_world_state — wyrd

the world-model side owns what the world IS; ThoughtForge owns only what it knows and infers — a cognition answer that contradicts the canonical state is wrong however fluent it is (TF-132's contradicted cases are exactly that shape, in miniature)

### world_change_log — wyrd

state is a replay of changes, not a snapshot a cognition layer may freeze: TF-132's stale-answer cases (the wheel still turning after the belt breaks) are failures to honour the change log

### entity_identity_in_world — wyrd

ThoughtForge-side ids (store record ids, bond ids, panel entity ids) are references; the world's own identity for an entity is the world side's to file — the bond layer's evidence link already works this way, referencing ids it does not own

### embodiment_and_transport — neither

neither the cognition layer nor the world-model protocol is a body: the consumer side (the Second Life / Heimdall body) owns embodiment, as Part IX files bindings as cognition only (TF-157's Samtal shape)

### canon_data — neither

canon is Volmarr's, private, never published and never either system's property; both sides consume only what he files for them, and every fixture in this estate is synthetic (TF-017)

### live_behaviour_proof — neither

unclaimed because unproven: every instrument in this repo is fixture grade, so neither side may file live truthfulness as owned or done — it is NOT PROVEN (TF-173 class) until a live run proves it

## 3. The interface between them

Five filed shapes, each with its honest status. There is deliberately no live status in the vocabulary: a shape is in production, fixture grade, or not built — and the one bridge that would carry ThoughtForge data to WYRD consumers is the not-built one, TF-146's.

### world_state_replay — world state, replayed to a point [fixture-stand-in-only]

Direction: wyrd -> thoughtforge

Shape: a filed world (entities with attributes / locations / relations) + an ordered change log; the state at a point is the dawn state plus the first N changes, recomputed purely — never a copied snapshot

Evidence: benchmarks/wyrd_panel.py: WORLD_ENTITIES (8) + WORLD_EVENTS (6) + world_state_at() over the synthetic Emberwick; no live WYRD feed or client stands in this tree

### consistency_judgement — an answer judged against the world [exists-as-panel-fixture]

Direction: thoughtforge measurement over wyrd-shaped state

Shape: the answer's structured claim (bound to its text by its claim phrase) vs the replayed world value -> consistent / contradicted / honest_unanswerable, with inventing an unfiled value scored contradicted

Evidence: benchmarks/wyrd_panel.py: score_case() over 18 filed cases (9 consistent / 8 contradicted / 1 honest-unanswerable, PASS) — scripted answers only, no live model

### bond_evidence_link — bond evidence naming store records [exists-in-production]

Direction: thoughtforge layer -> thoughtforge store (the layer link the bridge builds on)

Shape: BondEvidence.supporting_record_ids / contradiction_record_ids reference shipping-store record ids, and apply_event(source_record_id=...) appends the source record id — the layer holds relational state only and duplicates no fact

Evidence: src/thoughtforge/knowledge/bond_graph.py (TF-143, parity exact with TF-139's spike)

### bond_excerpt_bridge — bonds as WYRD-consumable data [not-built-tf146]

Direction: thoughtforge -> wyrd

Shape: the source shape exists: BondGraph.excerpt(bond_id) files deterministic lines (domain / status / the three indices / relational mode / emotional weather / unresolved hurts / vows — 9 lines on the parity fixtures); the bridge that files those lines as WYRD-consumable data is TF-146's work and is NOT built here

Evidence: TF-139's conditions + TF-143's filing, both naming TF-146 as the bridge's owner; no bridge code stands in this tree

### cognition_turn — the cognition turn a consumer calls [exists-in-production]

Direction: consumer -> thoughtforge

Shape: turn-in / turn-out through ThoughtForgeCore.think() / think_stream(), with the turn's traces and citation check filed on the core; service-shaped bindings over this surface are Part IX's work (TF-153..TF-158), not this boundary's

Evidence: src/thoughtforge/cognition/core.py (ThoughtForgeCore.think) + the roadmap's Part IX filing

## 4. Written for the consumers

### Second Life / Heimdall [named-consumer-not-integrated]

Consumes: the cognition turn and, once TF-146 builds it, the bond bridge's WYRD-consumable bond data — a body in a lived world thinking with ThoughtForge while the world's state stays WYRD-side

Evidence: TF-145's roadmap text names the Second Life consumers; no Heimdall or Second Life code stands under src/ in this tree

### Hermes [named-consumer-not-integrated]

Consumes: ThoughtForge as a cognition provider under the volmarr-core shaped contract (TF-158), with the campaign's contracts handed over in the portable shapes (TF-193); memory portability is already filed on that law

Evidence: the roadmap's TF-158 / TF-193 filings + src/thoughtforge/knowledge/portability.py (Hermes-portability law applied to memory); no Hermes binding stands in this tree

For a Second Life / Heimdall reader: your body owns embodiment and transport (neither row `embodiment_and_transport`); the world you stand in is WYRD-side; the thinking you do with it is ThoughtForge-side, through the cognition turn — and the bond data becomes WYRD-consumable only when TF-146 builds its bridge. For a Hermes reader: ThoughtForge is a cognition provider under TF-158's contract; nothing in this boundary makes WYRD a Hermes component or Hermes a world model.

## 5. Docket

TF-145 has **no share** of TF-016's D-WING-1: the entry assigns the verdicts (TF-139..TF-142), the adoptions (TF-143), the retirements (TF-144), research hygiene (TF-148) and the wing's tests (TF-149) — a boundary doc over the estate those slices built is none of those shares. This slice therefore does not edit the docket and adds no entry; TF-016 totals stand unchanged (137 entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 CLOSED-BY-VERDICT), filed in the open here, per the TF-144 pattern's rule that a slice touches the docket only for its own filed share.

## 6. The probes

- `clean_boundary_filing_passes`: the filed ownership + interface + consumer rows pass their validators
- `ownership_without_evidence_is_named`: an ownership row with its evidence emptied is named
- `unknown_owner_is_named`: an ownership row filing an owner outside the filed vocabulary is named
- `wyrd_row_citing_shipping_code_is_named`: a WYRD row citing a shipping code path as its evidence is named — no WYRD code stands in this repo
- `live_interface_claim_is_named`: an interface row claiming a live status outside the filed vocabulary is named
- `interface_without_a_shape_is_named`: an interface row with its shape emptied is named
- `consumer_claiming_integration_is_named`: a consumer row filing any status but not-integrated is named

## 7. Filed limits

- Boundary grade only: this doc files who owns what and the shapes between them, grounded in this repo's evidence. It builds no bridge (TF-146), no binding (Part IX) and no live integration of any kind.
- The WYRD side is filed from the roadmap's own words plus the shapes TF-132's panel stands in for: WYRD's implementation lives outside this repo and was not read here, so no WYRD internal is described — only the boundary surface this estate can evidence.
- TF-132's world is synthetic and small on purpose (8 entities / 6 changes / 18 cases): it evidences the consistency shape, not scale, and its answers are scripted — a live system staying true to a lived world is NOT PROVEN (TF-173).
- No live model was read; no canon data was touched (TF-017); no README.md anywhere was touched.
