# Persona Compiler -> Rúnmál Cards — compiled personae in Rúnmál's card shape (TF-147)

GENERATED from the filed data in `src/thoughtforge/cognition/runmal_cards.py` — the filed doc IS the render: regenerate with `python3 -m thoughtforge.cognition.runmal_cards`. The checker (`runmal_cards_problems()`) re-derives every claim live from the tree and fails on any drift.

Format: `thoughtforge-runmal-cards/1`
Source format: `thoughtforge-persona-compiler/1` (TF-143)
Card envelope: `chara_card_v2` / spec_version `2.0`
Antecedent: TF-141 **MERGE** (adopt-in-part) — TF-147 is live work on TF-143's merged compiled shape
Card fields filed: 14
Probes: 8/8

## 1. Measured start

Probed live on this tree before anything was built: **no card export existed** — no card module under `src/thoughtforge/cognition/`, no card doc, and the production `CompiledPersona`'s only outward shapes were its dataclass fields and the merged expectation's `steering_lines()`: nothing in the public character-card shape a Rúnmál-side consumer imports and exports. The antecedent stood clean live: TF-141's verdict MERGE (`persona_compiler_verdict_problems()` []), TF-143's production compiler clean (main run identity 4 / anchors 4 / traits 3 / source items 14), and TF-146's bridge clean beside it. TF-141's verdict doc files the condition verbatim: TF-147 consumes the MERGED compiled output — live work on the merged shape, not on the wing's packet as-is. This module exports TF-143's `CompiledPersona`; no Rúnmál code is copied or read as code — the card shape's field names are the public character-card v2 spec's.

## 2. The card shape — every field filed

`export_card(compiled)` builds the card FROM a real `CompiledPersona` compiled by TF-143's production `PersonaCompiler`. The envelope is `{"spec": "chara_card_v2", "spec_version": "2.0", "data": {...}}`; every data field files its source and disposition below — none silently dropped, none invented.

| Field | Source in the compiled persona | Disposition | Mapping rule |
|---|---|---|---|
| name | CompiledPersona.identity_core — the "<subject>.name=<value>" identity line when present, else persona_id | mapped | the value after ".name=" in the identity line, verbatim; the fallback persona_id is filed, never an invented name |
| description | CompiledPersona.identity_core (all lines) | mapped | the identity lines joined with "; " — the compiled identity, verbatim, nothing added |
| personality | CompiledPersona.active_traits (TraitSignal: name / weight / supporting record ids) + CompiledPersona.tone_contract | mapped | one "<trait> (weight <w:.2f>; evidence: <ids>)" per trait, joined with "; ", then a "Tone: " line filing the tone contract verbatim — the weights and their evidence ride in the text, not just the trait names |
| scenario | filed empty — no compiled source | filed-empty | the empty string: no compiled facet files a scenario, and scenario prose is never fabricated to fill a card |
| first_mes | filed empty — no compiled source | filed-empty | the empty string: the compiler files no greeting or dialogue, and a first message is never fabricated |
| mes_example | filed empty — no compiled source | filed-empty | the empty string: no compiled facet files example dialogue, and example dialogue is never fabricated |
| creator_notes | the export itself: RUNMAL_CARDS_FORMAT + the compiled persona's ids / mode + every source_items record id (provenance) | mapped | one deterministic note filing the format, the persona / user / mode it was compiled from, the full provenance record-id list, the filed-empty honesty rule, and the filed limit that a live Rúnmál consumer importing the card is NOT PROVEN (TF-156) |
| system_prompt | CompiledPersonaExpectation.steering_lines() — the merged expectation TF-030's inject_persona delivers | mapped | the steering lines joined with newlines, reused from the expectation itself — the real delivered steering, never re-derived or paraphrased here |
| post_history_instructions | filed empty — no compiled source | filed-empty | the empty string: the compiled shape files no post-history instruction, and none is invented |
| alternate_greetings | filed empty — no compiled source | filed-empty | the empty list: the compiler files no greetings, and greetings are never fabricated |
| tags | exporter-derived — the export's format family + the compiled persona's mode | derived | ["thoughtforge", "compiled-persona", "mode:<mode>"] — provenance tags only, no compiled facet is summarised into a tag |
| creator | exporter constant — the producing pipeline, named | exporter-constant | "ThoughtForge persona compiler (TF-143), exported by TF-147" — the pipeline that produced the card, not a persona claim |
| character_version | exporter constant — RUNMAL_CARDS_FORMAT | exporter-constant | the export format string itself, so the card files the version of the shape that produced it |
| extensions | CompiledPersona facets with no card home, under the namespaced key thoughtforge_compiled_persona | extension | the structured facets verbatim: identity / tone / guidance beside the six EXTENSION_FACETS (truth anchors, uncertainty, symbolic context, the relationship excerpt, source items with confidences, structured trait weights) + the merged expectation's declared side — nothing the compiler filed is lost |

Filed counts: 14 fields — 5 mapped, 5 filed empty, 1 extension, 1 derived, 2 exporter constants.

## 3. Export proof (synthetic fixtures, TF-017)

TF-143's main fixture compile (the Wrenna records + the production bond edge + the authored base expectation, exactly `run_persona_compile`'s main run) is exported, and the card is proven to carry the filing's claims:

- name: Wrenna the Lantern-Keeper (from the compiled identity's name fact)
- provenance: 14 source records, every record id filed in creator_notes
- system_prompt: the merged expectation's 11 steering lines verbatim — the real delivered steering, reused
- personality: 3 evidence-weighted traits with their weights + supporting ids, plus the tone contract
- extension facets (6): truth anchors 4 / uncertainty 3 / symbolic 2 / relationship lines 3 / source items + structured trait weights — nothing the compiler filed is lost

The exported card, verbatim:

```json
{
  "data": {
    "alternate_greetings": [],
    "character_version": "thoughtforge-runmal-cards/1",
    "creator": "ThoughtForge persona compiler (TF-143), exported by TF-147",
    "creator_notes": "Exported from a ThoughtForge compiled persona (TF-143) by TF-147 under thoughtforge-runmal-cards/1; persona_id=persona:wrenna; user_id=user:lantern-keeper; mode=companion. Provenance \u2014 compiled from source records: rec:synthetic-wrenna-001, rec:synthetic-wrenna-002, rec:synthetic-wrenna-003, rec:synthetic-wrenna-004, rec:synthetic-wrenna-005, rec:synthetic-wrenna-006, rec:synthetic-wrenna-007, rec:synthetic-wrenna-008, rec:synthetic-wrenna-009, rec:synthetic-wrenna-010, rec:synthetic-wrenna-011, rec:synthetic-wrenna-012, rec:synthetic-wrenna-013, rec:synthetic-wrenna-014. Filed limits: fields with no compiled source are filed empty, never fabricated (see CARD_FIELD_ROWS in thoughtforge/cognition/runmal_cards.py); compiled facets with no card home ride in extensions.thoughtforge_compiled_persona. Fixture/static grade \u2014 a live R\u00fanm\u00e1l consumer importing this card is NOT PROVEN (TF-156 owns the live binding).",
    "description": "persona:wrenna.temperament=steady; persona:wrenna.name=Wrenna the Lantern-Keeper; persona:wrenna.tone=plain-spoken; persona:wrenna.duty=keep the ferry lantern lit",
    "extensions": {
      "thoughtforge_compiled_persona": {
        "active_traits": [
          {
            "supporting_record_ids": [
              "rec:synthetic-wrenna-001",
              "rec:synthetic-wrenna-005"
            ],
            "trait_name": "steady",
            "weight": 1.0
          },
          {
            "supporting_record_ids": [
              "rec:synthetic-wrenna-003"
            ],
            "trait_name": "plain-spoken",
            "weight": 0.8
          },
          {
            "supporting_record_ids": [
              "rec:synthetic-wrenna-008"
            ],
            "trait_name": "honesty",
            "weight": 0.6
          }
        ],
        "expectation": {
          "avoid": [
            "invented crossings after the second tide bell",
            "vague handwaving",
            "repetition"
          ],
          "base_record_id": "syn_wrenna_001",
          "behavior_rules": [
            "answer what was asked",
            "cite the synthetic reference when using it",
            "say plainly when the fixture store is silent"
          ],
          "record_id": "syn_wrenna_001",
          "speech_style": [
            "short, concrete sentences",
            "names the river level before giving advice",
            "no filler"
          ],
          "traits": [
            "steady",
            "observant",
            "plain-spoken"
          ],
          "weights": {
            "depth": 0.6,
            "directness": 0.8,
            "formality": 0.3,
            "warmth": 0.5,
            "wit": 0.2
          }
        },
        "format": "thoughtforge-persona-compiler/1",
        "identity_core": [
          "persona:wrenna.temperament=steady",
          "persona:wrenna.name=Wrenna the Lantern-Keeper",
          "persona:wrenna.tone=plain-spoken",
          "persona:wrenna.duty=keep the ferry lantern lit"
        ],
        "mode": "companion",
        "persona_id": "persona:wrenna",
        "relationship_excerpt": [
          "relationship_mode=builder",
          "relationship_weather=calm",
          "closeness_index=0.44"
        ],
        "response_guidance": [
          "relationship_mode=builder",
          "relationship_weather=calm",
          "closeness_index=0.44",
          "never claim vows or memories that are not supported"
        ],
        "source_items": [
          {
            "confidence": 0.6,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-001",
            "text": "Wrenna is steady"
          },
          {
            "confidence": 0.95,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-002",
            "text": "the keeper's name"
          },
          {
            "confidence": 0.8,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-003",
            "text": "Wrenna is plain-spoken"
          },
          {
            "confidence": 0.85,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-004",
            "text": "the keeper's duty"
          },
          {
            "confidence": 0.5,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-005",
            "text": "a steady mood, observed"
          },
          {
            "confidence": 0.9,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-006",
            "text": "the crossing takes eleven minutes"
          },
          {
            "confidence": 0.7,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-007",
            "text": "the river's name"
          },
          {
            "confidence": 0.6,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-008",
            "text": "honesty about the river"
          },
          {
            "confidence": 0.3,
            "item_type": "canonical_fact",
            "record_id": "rec:synthetic-wrenna-009",
            "text": "an unapproved bell count"
          },
          {
            "confidence": 0.65,
            "item_type": "symbolic_trace",
            "record_id": "rec:synthetic-wrenna-010",
            "text": "a lantern trace"
          },
          {
            "confidence": 0.55,
            "item_type": "symbolic_trace",
            "record_id": "rec:synthetic-wrenna-011",
            "text": "a river-stone trace"
          },
          {
            "confidence": 0.7,
            "item_type": "episode_summary",
            "record_id": "rec:synthetic-wrenna-012",
            "text": "Wrenna logged the river level at dawn"
          },
          {
            "confidence": 0.4,
            "item_type": "episode_summary",
            "record_id": "rec:synthetic-wrenna-013",
            "text": "an unverified tale of a third tide bell"
          },
          {
            "confidence": 0.35,
            "item_type": "episode_summary",
            "record_id": "rec:synthetic-wrenna-014",
            "text": "a disputed tale of a night crossing"
          }
        ],
        "symbolic_context": [
          "symbol=lantern;runes=ansuz,raidho;charge=0.75",
          "symbol=river-stone;runes=laguz;charge=0.40"
        ],
        "tone_contract": [
          "preserve continuity without inventing unsupported memory",
          "prefer grounded specifics over vague mythic filler",
          "mark uncertainty cleanly when canon or memory is missing",
          "maintain warmth without synthetic intimacy inflation"
        ],
        "truth_anchor_points": [
          "persona:wrenna.mood=steady",
          "world:lantern-hollow.ferry_minutes=11",
          "world:lantern-hollow.river=the Bracken",
          "persona:wrenna.value=honesty"
        ],
        "uncertainty_points": [
          "an unapproved bell count",
          "an unverified tale of a third tide bell",
          "a disputed tale of a night crossing"
        ],
        "user_id": "user:lantern-keeper"
      }
    },
    "first_mes": "",
    "mes_example": "",
    "name": "Wrenna the Lantern-Keeper",
    "personality": "steady (weight 1.00; evidence: rec:synthetic-wrenna-001, rec:synthetic-wrenna-005); plain-spoken (weight 0.80; evidence: rec:synthetic-wrenna-003); honesty (weight 0.60; evidence: rec:synthetic-wrenna-008)\nTone: preserve continuity without inventing unsupported memory; prefer grounded specifics over vague mythic filler; mark uncertainty cleanly when canon or memory is missing; maintain warmth without synthetic intimacy inflation",
    "post_history_instructions": "",
    "scenario": "",
    "system_prompt": "Persona traits: steady; observant; plain-spoken\nPersona speech style: short, concrete sentences; names the river level before giving advice; no filler\nPersona behavior: answer what was asked; cite the synthetic reference when using it; say plainly when the fixture store is silent\nPersona identity: persona:wrenna.temperament=steady; persona:wrenna.name=Wrenna the Lantern-Keeper; persona:wrenna.tone=plain-spoken; persona:wrenna.duty=keep the ferry lantern lit\nPersona truth anchors: persona:wrenna.mood=steady; world:lantern-hollow.ferry_minutes=11; world:lantern-hollow.river=the Bracken; persona:wrenna.value=honesty\nPersona tone: preserve continuity without inventing unsupported memory; prefer grounded specifics over vague mythic filler; mark uncertainty cleanly when canon or memory is missing; maintain warmth without synthetic intimacy inflation\nPersona guidance: relationship_mode=builder; relationship_weather=calm; closeness_index=0.44; never claim vows or memories that are not supported\nPersona relationship: relationship_mode=builder; relationship_weather=calm; closeness_index=0.44\nPersona symbolic: symbol=lantern;runes=ansuz,raidho;charge=0.75; symbol=river-stone;runes=laguz;charge=0.40\nPersona uncertainty: an unapproved bell count; an unverified tale of a third tide bell; a disputed tale of a night crossing\nPersona trait weights: steady=1.00; plain-spoken=0.80; honesty=0.60",
    "tags": [
      "thoughtforge",
      "compiled-persona",
      "mode:companion"
    ]
  },
  "spec": "chara_card_v2",
  "spec_version": "2.0"
}
```

The card JSON-round-trips byte-stable (`json.dumps(card, sort_keys=True)` re-serialised identically after a parse), and a second compile + export produces the identical card — the export carries no wall-clock value anywhere.

## 4. Filed empty — never fabricated

Five fields have no honest compiled source — `scenario`, `first_mes`, `mes_example`, `post_history_instructions`, `alternate_greetings` — and are filed empty (empty string / empty list) with the reason in the table above. The compiler files no scenario, no greeting and no dialogue, so the card invents none: the pure validator names any export that fills one of them (probe `fabricated_scenario_is_named`). A consumer that needs those fields authors them itself, outside this export.

## 5. Neighbour coupling, in the open

TF-139's verdict checker greps the shipping knowledge + cognition sources for bond / relationship vocabulary (its uniqueness premise: the persisted store files no bond record type). This module legitimately files the compiled persona's relationship excerpt and compiles from TF-143's production bond edge, so TF-139's verdict-module exclusion is extended to `runmal_cards.py` by name in `bond_graph_verdict.py` + its tests, exactly as TF-143 extended it for the production modules and TF-146 for the bridge. The premise guarded is unchanged: the persisted store still files no bond record type, checked there still.

## 6. Docket

TF-147 has **no share** of TF-016's D-WING-1: the entry's current text (checked live for this slice) assigns the verdicts (TF-139..TF-142), the adoptions (TF-143), the retirements (TF-144), research hygiene (TF-148) and the wing's tests (TF-149) — the card export's ownership is filed in TF-141's verdict conditions, not in D-WING-1. Per the TF-145/TF-146 pattern's rule (a slice touches the docket only for its own filed share), this slice does not edit the docket and adds no entry; TF-016 totals stand unchanged (137 entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 CLOSED-BY-VERDICT), filed in the open here.

## 7. The probes

- `clean_export_passes`: the fixture card + the filed field rows pass their validators against the compiled persona, and the card JSON-round-trips byte-stable
- `missing_required_field_is_named`: a card with its name field removed is named
- `unknown_field_is_named`: a card carrying an unknown data field is named — no field rides unfiled
- `field_row_missing_disposition_is_named`: a field row with its disposition emptied is named — every field must file its disposition
- `extension_disagreement_is_named`: a card whose extension truth anchors are perturbed is named against the compiled persona
- `name_mismatch_is_named`: a card whose name does not match the compiled identity is named
- `provenance_missing_is_named`: a card whose creator_notes drops a source record id is named — provenance is filed, never assumed
- `fabricated_scenario_is_named`: a card whose scenario is filled with prose is named — a field with no compiled source stays empty

## 8. Filed limits

- Fixture grade only: the export proof runs on synthetic Wrenna fixtures (TF-017); no live Rúnmál consumer was read — a live consumer importing the card is NOT PROVEN (TF-156 owns the live binding), never claimed.
- The export is a shape, not a binding: no Rúnmál call is routed anywhere here (TF-156 owns the binding, behind its backend table, off by default).
- The card's system_prompt is TF-030/TF-143's delivered steering reused verbatim — this module files no steering of its own.
- No canon data was touched; no README.md anywhere was touched.
