# Persona Compiler — TF-143 (production merge)

GENERATED from the filed data in `src/thoughtforge/cognition/persona_compiler.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.cognition.persona_compiler`. The checker (`persona_compiler_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-persona-compiler/1`
Verdict implemented: **MERGE** (TF-141, adopt-in-part)
Probes: 7/7

## 1. What landed

A compiled-expectation builder beside `persona.py`: `PersonaCompiler` compiles canonical records into a `CompiledPersona` carrying every adopted facet (evidence-weighted TraitSignals, source_items provenance with confidences, uncertainty_points, the identity_core / truth_anchor_points split, mode adaptation, symbolic_context, and a relationship section compiled from the PRODUCTION `BondEdge` of TF-143's bond graph layer), and a `CompiledPersonaExpectation` — a `PersonaExpectation` subclass — delivers it through TF-030's runtime: `inject_persona` delivers base + compiled steering and `check_persona_response` / `score_persona_turns` judge responses against it. persona.py is not modified; TF-009's authored PersonalityCoreRecord path remains a supported source (an optional base expectation whose expressive lists and avoid ride verbatim), never silently replaced.

## 2. Filed decisions

Avoid sourcing: **declared** — the merged expectation's avoid list is sourced solely from the authored PersonalityCoreRecord (via the base PersonaExpectation); the compiler contributes no avoid entries.

- no record shape the compiler reads (canonical fact, symbolic trace, generic record) carries an avoid entry — a compiled avoid would be invented prose wearing evidence's clothing
- avoid is a prohibition the author states, not a fact evidence sums to: TF-030 checks it deterministically precisely because it is declared verbatim
- the authored path stays a supported source (TF-141's condition), so a declared avoid always has a home in the merged shape; compiling without a base files an empty avoid, honestly, never a fabricated one

Token budget hint: **dropped** — the production compiled persona carries no token_budget_hint field at all.

- TF-141 measured the spike's hint (default 800) as carried but never enforced — a hint nothing honours is not a bound, and TF-141's condition allows exactly two honest outcomes: real trimming or dropping it
- the merged shape's real bounds already exist and are enforced: the compiler's filed caps on every section and TF-030's MAX_STEERING_ENTRIES at delivery — a second, unenforced number would only pretend to bound what those already bound
- no trimming order is filed because nothing is trimmed by budget: if a future slice needs a token bound it must file its own trim order against the tree's word-token convention then, with evidence

## 3. Defects not imported (measured differences)

- `approval_checked_first` — spike: the elif chain checks canonical/symbolic BEFORE approval: the pending canonical fact (bell_count, confidence 0.3) files as the truth anchor world:lantern-hollow.bell_count=two; spike anchors 5 / uncertainty 2; production: approval is checked FIRST: any unapproved record (canonical, symbolic or generic) surfaces as an uncertainty_point and never as an anchor, identity line, trait or symbolic line; production anchors 4 / uncertainty 3 on the same fixtures
- `relationship_excerpt_bond_only` — spike: relationship_excerpt is response_guidance[:6] verbatim, duplicating all 4 guidance lines (bond lines + the mode line); production: relationship_excerpt files the bond-only lines (3 on the fixture run) and shares no line with the mode guidance
- `token_budget_hint_dropped` — spike: token_budget_hint (default 800) is carried onto the packet and never enforced; production: no token_budget_hint field exists on the compiled persona (TOKEN_BUDGET_DECISION)

## 4. The production compile (Wrenna fixtures)

- identity_core 4 / truth_anchor_points 4 (spike: 4 / 5 — the pending bell_count fact moves to uncertainty under approval-first)
- active_traits 3 (steady 1.0 at the cap from 0.6 + 0.5 with both supporting ids; plain-spoken 0.8; honesty 0.6)
- symbolic_context 2 / uncertainty_points 3 (spike: 2) / source_items 14
- tone_contract 4 / response_guidance 4 / relationship_excerpt 3 (bond-only lines)
- caps run (57 records): every cap binds exactly — identity 10 / anchors 16 / traits 8 / symbolic 8 / uncertainty 8 / source_items 20
- merged delivery: 11 steering lines injected into a real scaffold (base lines preserved: True); fixture turns judged 2/5 consistent, score 0.4, verdict REVIEW — TF-030's avoid checking unchanged on the merged shape

## 5. The probes

- `clean_compile_passes`: the production compile + its merged delivery pass their validators
- `trait_weight_wrong_is_named`: an expected filing with steady's weight perturbed is named against the compiled traits
- `caps_breach_is_named`: a counts filing with identity_core over its cap is named
- `schema_field_missing_is_named`: a field map with supporting_record_ids dropped from TraitSignal is named against the schemas
- `unapproved_anchor_is_named`: a compile filing the pending bell_count line as a truth anchor is named by the approval validator
- `relationship_duplicating_guidance_is_named`: an excerpt copied from response_guidance is named by the relationship validator
- `unknown_mode_is_named`: a mode value outside the schema's PersonaMode enum is named

## 6. Filed limits

- Fixture grade only: the ground truth is TF-017 / TF-030's synthetic Wrenna world; no live model was read — live persona quality is NOT PROVEN (TF-173 class).
- TF-147 consumes this merged output for Rúnmál cards; TF-149 owns the wing's tests. TF-140/TF-142's RETIRE verdicts land nothing here — TF-144 executes them.
