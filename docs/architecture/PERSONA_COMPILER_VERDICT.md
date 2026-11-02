# Persona Compiler Verdict — TF-141

GENERATED from the filed data in `src/thoughtforge/cognition/persona_compiler_verdict.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.cognition.persona_compiler_verdict`. The checker (`persona_compiler_verdict_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-persona-compiler-verdict/1`
Verdict: **MERGE**
Comparison rows: 12
Probes: 7/7

## 1. Measured start

Probed live on this tree before anything was built: the wing cannot import — `research_data/src/wyrdforge/models` does not exist and `persona_compiler.py` imports its types from that missing package (TF-006 / D-WING-1). The compiler turns memory records (+ an optional BondEdge) into a PersonaPacket (identity/anchor split by domain, confidence-weighted traits, symbolic lines, uncertainty points, mode additions, bond guidance, source items with confidences). The shipping path (TF-009 + TF-030) loads an authored PersonalityCoreRecord into a PersonaExpectation whose steering is injected into every prompt and whose avoid list is checked deterministically — and whose five weights steer nothing, a deferral TF-030 filed to this slice by name. Both paths run here over the same synthetic ground truth (TF-017 / TF-030's Wrenna fixtures; no canon).

## 2. The comparison

| Dimension | Compiler | Shipping path | Finding |
|---|---|---|---|
| `expressive_coverage` | a compiled packet of typed sections built from records: identity_core lines, evidence-weighted active_traits, a tone_contract, symbolic_context, response_guidance — traits/style/behaviour reach the packet only as facts about the persona (identity/style/mission domains, trait keys), never as verbatim authored lists | the authored core's three expressive lists verbatim (traits / speech_style / behavior_rules, 3 + 3 + 3 entries on the fixture) delivered as three steering lines — narrower (no symbolic, no relationship, no uncertainty sections) but the author's exact words | **different** |
| `provenance` | every input record becomes a source_item (record_id + item_type + summary text + the record's truth.confidence, cap 20 — 14/14 carried on the spike run), and every active trait carries its supporting_record_ids | the expectation carries only the core's own record_id — no per-fact provenance, no confidences, nothing a scorer can trace a steering line back to | **compiler-wins** |
| `trait_weighting` | traits are DERIVED from evidence: records whose fact_key is temperament/tone/value/mood add their truth.confidence to the bucket named by fact_value; the spike's 'steady' sums 0.6 + 0.5 = 1.1 and files at the min(1.0, sum) cap with both supporting ids; traits sort by weight, cap 8 | traits are a static authored list, unweighted in the prompt; the core's five tuning weights are carried on the expectation with NO prompt effect at all (TF-030 filed the deferral to TF-141 by name) — the numbers steer nothing | **compiler-wins** |
| `uncertainty_surfacing` | unapproved records surface as uncertainty_points (cap 8): the spike's pending + quarantined generic records file 2 points. Filed defect: the elif chain checks canonical/symbolic BEFORE approval, so the spike's PENDING canonical fact never surfaces — it lands in truth_anchor_points as truth (see identity_vs_truth_anchor_separation) | no uncertainty surface exists: the authored core is taken as in force, and the store's approval states never reach the expectation | **compiler-wins** |
| `identity_vs_truth_anchor_separation` | canonical facts split by domain: identity/style/mission -> identity_core (4 on the spike run), everything else -> truth_anchor_points (5) — the persona's self vs the world's facts are separate sections. The split is only as honest as the approval blind spot above: the pending fact is one of the 5 anchors | no split: one authored core record IS the identity, and world facts live in the knowledge store on a different path entirely — the persona output never files them side by side, separated or otherwise | **compiler-wins** |
| `mode_adaptation` | three modes adapt the packet: CODING_GUIDE, WORLD_SEER and COMPANION each add one tone_contract line + one response_guidance line to the 3 base contract lines (4 / 1 measured per mode, no bond); the schema files two further modes (ritual, debrief) the compiler treats as base-only | no modes: one expectation serves every conversation and intent; adaptation is the scaffold's per-turn business, never the persona's | **compiler-wins** |
| `relationship_context` | an optional BondEdge compiles to three guidance lines (relationship_mode / relationship_weather / closeness_index at 2dp — builder / calm / 0.44 on the spike's TF-139 edge). Filed defect: relationship_excerpt is response_guidance[:6] verbatim — on the spike run it duplicates all 4 guidance lines including the mode line; it is not a bond-only excerpt | no relationship input exists: the expectation is built from the personality core alone (TF-139's bond layer, adopted as a layer, has no consumer in the persona path) | **compiler-wins** |
| `symbolic_context` | symbolic traces compile to formatted lines (symbol=<type>;runes=<csv>;charge=<2dp>, cap 8 — 2 on the spike run) in their own packet section | no symbolic input or section: a symbolic trace can only be prose inside a fact, indistinguishable from any other steering | **compiler-wins** |
| `prompt_steering_delivered` | the packet is inert data: no code in the shipping tree consumes a PersonaPacket, injects it into a scaffold, or renders it into a prompt — compiled persona reaches no generation | inject_persona prepends the steering block to the scaffold's fact_block and ThoughtForgeCore runs it every turn (TF-030's wiring): on the spike run the block lands verbatim above the facts, first line as the duplicate guard | **shipping-wins** |
| `avoid_and_consistency_checking` | no avoid concept and no response checker: the tone_contract is prose a model may ignore, and nothing in the wing scores a response against the packet | avoid is checked deterministically: check_persona_response matches the persona's avoid entries (normalised, phrases / whole words) and score_persona_turns aggregates — on TF-030's fixture turns: 5 checked, 2 consistent, 3 violations, score 0.4, verdict REVIEW (never a vacuous pass) | **shipping-wins** |
| `bounds_caps_and_determinism` | every section capped (identity 10, anchors 16, traits 8, symbolic 8, uncertainty 8, source_items 20, relationship_excerpt 6, response_guidance 10 — all measured binding on the caps run: 57 records in, exactly the caps out) and the compile is deterministic; but token_budget_hint (800) is carried, never enforced — no trimming honours it | steering capped at MAX_STEERING_ENTRIES=6 per list (a 9-trait expectation files exactly 6 in its line), avoid uncapped in the expectation but bounded by the scaffold's own discipline, and the hooks are deterministic pure functions — both paths bound and reproduce; they bound different things | **different** |
| `source_of_truth_and_authoring` | the persona is COMPILED from the memory record store: change a canonical fact's confidence or approval and the packet changes — persona as a view over evidence, with the store as the source of truth | the persona is DECLARED in one authored config record (TF-009's census: the personality core YAML), versioned by hand, protected from auto-pruning — persona as a document, with the author as the source of truth | **different** |

## 3. The verdict: MERGE

### Reasons

- the compiler wins seven measured facets the shipping path does not have at all: per-record provenance with confidences (14/14 source items), evidence-weighted traits (steady 1.1 filed at the 1.0 cap with both supporting ids — the answer to TF-030's weights deferral), uncertainty surfacing (2 points), the identity/truth-anchor split (4 / 5), mode adaptation (three modes, 4-line contracts) and symbolic + relationship sections (2 symbolic lines, 3 bond guidance lines) — retiring or ignoring that shape throws away the estate's only compiled-persona design
- the shipping path wins the two facets that make a persona real: its steering is DELIVERED (inject_persona puts the block into every prompt; a PersonaPacket has no consumer anywhere in the shipping tree) and its avoid list is CHECKED deterministically (score_persona_turns over the fixture turns: 2/5 consistent, score 0.4) — the compiler's tone_contract is unchecked prose, so adopting the compiler INSTEAD of the shipping path would leave the persona uninjected and unjudged
- not adopt wholesale: the compiler imports two measured defects — its elif chain promotes unapproved canonical facts into truth_anchor_points (the spike's pending fact files as an anchor, never as uncertainty), relationship_excerpt duplicates response_guidance[:6] instead of filing bond-only lines, and its token_budget_hint is carried but never enforced; a merge adopts the facets and files the defects as TF-143 work, an adoption-as-is would import them
- not retire, not keep-shipping-alone: the shipping path's traits are static unweighted lists and its five tuning weights steer nothing (carried since TF-030 with the question expressly deferred here) — the compiler's TraitSignal arithmetic is the measured answer, and the packet's typed sections give TF-147's Rúnmál cards a compiled shape to export
- both paths are bounded and deterministic on the same fixtures (compiler caps bind exactly on the 57-record caps run; shipping steering caps at 6 and its checks are pure) — the merge keeps one persona path whose inputs may be compiled, whose delivery and checking are TF-030's

### Conditions — what TF-143 must implement

- MERGE the compiler's proven facets into the shipping persona path: TF-143 extends the shipping expectation (or lands a compiled-expectation builder beside it) carrying evidence-weighted TraitSignals with supporting record ids, source_items provenance with confidences, uncertainty_points, the identity_core / truth_anchor_points split, mode adaptation, and symbolic + relationship sections — compiled from canonical records where they exist, with TF-009's authored PersonalityCoreRecord config path remaining a supported source, never silently replaced
- TF-143 keeps TF-030's runtime as the merged path's delivery and judgement: inject_persona steering into the prompt and check_persona_response / score_persona_turns avoid checking are NOT replaced by a packet; how avoid is sourced in the merged shape (declared, compiled, or both) is TF-143's filed decision
- TF-143 must NOT import the compiler's defects: unapproved canonical / symbolic records surface as uncertainty, never silently as truth anchors (the verdict's recommendation, decided in the open there); relationship_excerpt files bond-only lines, not response_guidance[:6] duplicated; token_budget_hint is either enforced by real trimming or dropped — a hint nothing honours is not a bound
- TF-147 (persona compiler -> Rúnmál cards) consumes the MERGED compiled output: this verdict answers its 'if adopted' with adopt-in-part, so TF-147 is live work on the merged shape, not on the wing's packet as-is
- TF-141 changes NO shipping behaviour and NO research-wing code: the spike lives in this verdict module only, namespaced Spike*, and is not the missing wyrdforge.models package; TF-149 owns the wing's tests

## 4. The working spike + the measurement

Spike dataclasses rebuilt from the schemas inside the verdict module (the packet shapes in full from persona_packet.schema.json; CanonicalFactPayload / CanonicalFactContent, SymbolicTracePayload / SymbolicTraceContent and TruthMeta from the two record schemas), namespaced `Spike*` — NOT the missing `wyrdforge.models` package and NOT production. The input records are reduced to what compile() actually reads; the BondEdge input is TF-139's spike edge, reused. `SpikePersonaCompiler` re-implements compile() faithfully, including its elif approval blind spot and its relationship_excerpt duplication, so both are measured, not asserted.

Main run (COMPANION + bond edge, 14 records): identity_core 4 / truth_anchor_points 5 / active_traits 3 / symbolic_context 2 / uncertainty_points 2 / source_items 14 / tone_contract 4 / response_guidance 4.

Active traits (evidence-weighted):

- steady: weight 1.0 from rec:synthetic-wrenna-001, rec:synthetic-wrenna-005
- plain-spoken: weight 0.8 from rec:synthetic-wrenna-003
- honesty: weight 0.6 from rec:synthetic-wrenna-008

The pending canonical fact (bell_count, confidence 0.3, approval pending) files as the truth anchor `world:lantern-hollow.bell_count=two` and NOT as an uncertainty point — the compiler's elif blind spot, measured. relationship_excerpt duplicates all 4 response_guidance lines verbatim (bond lines + the mode line).

Caps run (57 generated records): identity_core 10 / truth_anchor_points 16 / active_traits 8 / symbolic_context 8 / uncertainty_points 8 / source_items 20 — every cap binds exactly; token_budget_hint (800) is carried and never enforced.

Shipping path (TF-030 fixture expectation): 3 steering lines / 9 entries delivered by inject_persona (injected=True); avoid 3 entries, 0 in the steering block; weights carried 5, in steering 0; fixture turns checked 5, consistent 2, violations 3, score 0.4, verdict REVIEW; a 9-trait expectation files 6 steering entries (MAX_STEERING_ENTRIES=6).

## 5. The probes

- `clean_verdict_and_rows_pass`: the filed verdict + rows pass their validators
- `verdict_without_a_reason_is_named`: a verdict filing with its reasons emptied is named
- `comparison_row_missing_a_side_is_named`: a comparison row with its shipping side emptied is named
- `spike_trait_weight_wrong_is_named`: an expected filing with the steady trait's weight perturbed (1.0 -> 0.9) is named against the measured traits
- `caps_count_wrong_is_named`: an expected filing with the caps run's identity count perturbed (10 -> 9) is named against the measured caps
- `schema_field_missing_from_spike_is_named`: a field map with token_budget_hint dropped from PersonaPacket is named against the schemas
- `unknown_verdict_value_is_named`: a verdict value outside the roadmap's three is named

## 6. Docket

TF-141 discharges the persona-compiler share of D-WING-1 (verdict decided on the same synthetic ground truth through both paths, narrowed in the open in `docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 itself stays OPEN for the TF-142 verdict and the TF-143 rebuild / TF-144 retirements / TF-149 tests it assigns. The merge this verdict files is implemented by TF-143, not by TF-141; TF-147 consumes the merged output.

## 7. Filed limits

- Spike grade only: production merging is NOT done and NOT PROVEN here — TF-143 owns implementing the adopted facets under the conditions in §3; TF-149 owns the wing's tests.
- The compiler's two measured defects (the elif approval blind spot promoting pending canonical facts to truth anchors; relationship_excerpt duplicating response_guidance) are filed for TF-143 to NOT import — the verdict adopts facets, not bugs.
- Fixture grade only: the ground truth is synthetic (Wrenna / Lantern Hollow); no live model was read and live persona quality is NOT PROVEN (TF-173 class).
