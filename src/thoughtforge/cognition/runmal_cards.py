"""
TF-147 — persona compiler -> Rúnmál cards (if
adopted at TF-141): compiled personae exportable
in Rúnmál's card shape.

TF-147's roadmap text: *Persona compiler ->
Rúnmál cards (if adopted): compiled personae
exportable in Rúnmál's card shape.* The
antecedent is TRUE in part, exactly as TF-141
filed it: the verdict was **MERGE**
(adopt-in-part), and its conditions state
verbatim that TF-147 ``consumes the MERGED
compiled output: this verdict answers its 'if
adopted' with adopt-in-part, so TF-147 is live
work on the merged shape, not on the wing's
packet as-is`` (``docs/architecture/
PERSONA_COMPILER_VERDICT.md`` §conditions).
TF-143 landed that merged shape as production
code (``src/thoughtforge/cognition/
persona_compiler.py``: ``PersonaCompiler`` /
``CompiledPersona`` /
``CompiledPersonaExpectation``). This module
exports THAT shape — never the wing's packet.

The measured start, probed live on this tree
before anything was built:

  * **No card export existed.** No file under
    ``src/thoughtforge/cognition/`` named for
    cards, no ``docs/architecture/RUNMAL_CARDS``
    doc, and the production ``CompiledPersona``'s
    only outward shapes were its dataclass fields
    and the merged expectation's
    ``steering_lines()`` — nothing in the public
    character-card shape a Rúnmál-side consumer
    imports and exports.
  * **The antecedent and the evidence stood
    clean live**: TF-141's verdict MERGE
    (``persona_compiler_verdict_problems()``
    []), TF-143's production compiler clean
    (``persona_compiler_problems()`` [], main
    run identity 4 / anchors 4 / traits 3 /
    source items 14), and TF-146's bridge clean
    beside it.

What the export is (literal data below):

  * The Rúnmál card shape filed as literal
    data: the public character-card v2 envelope
    ``{"spec": "chara_card_v2", "spec_version":
    "2.0", "data": {...}}`` whose data fields
    are the fourteen ``CARD_FIELDS``. This is
    the public card spec shape Rúnmál's
    importer/exporter works with; no Rúnmál
    code is copied or read as code here — the
    shape's field names are the public spec's.
  * ``CARD_FIELD_ROWS`` — every card data
    field as literal data: the field, its
    source in TF-143's ``CompiledPersona`` (or
    its filed reason for having none), its
    disposition, and the mapping rule. Every
    field has a filed disposition — none is
    silently dropped, none is invented.
  * :func:`export_card` — a card built FROM a
    real ``CompiledPersona`` (the TF-143
    synthetic Wrenna fixtures, TF-017 data
    only): name from the compiled identity,
    description from ``identity_core``,
    personality from the evidence-weighted
    ``active_traits`` + the tone contract,
    ``system_prompt`` from the merged
    expectation's ``steering_lines()`` itself —
    the real delivered steering, reused, never
    re-derived — and ``creator_notes`` carrying
    the filed limits + the provenance record
    ids.
  * Fields with NO honest compiled source
    (``scenario``, ``first_mes``,
    ``mes_example``,
    ``post_history_instructions``,
    ``alternate_greetings``) are filed EMPTY
    with the reason in ``CARD_FIELD_ROWS`` —
    dialogue and scenario prose are never
    fabricated to fill a card.
  * Compiled facets with no card home
    (``truth_anchor_points``,
    ``uncertainty_points``, ``symbolic_context``,
    ``relationship_excerpt``, ``source_items``,
    and the structured trait weights) ride in
    ``extensions`` under the namespaced key
    ``thoughtforge_compiled_persona``, so the
    export loses nothing the compiler filed.
  * The round-trip proof, on synthetic fixtures
    only (TF-017): the fixture compiled persona
    is exported, the card is JSON round-tripped
    byte-stable under ``sort_keys``, and a
    second compile + export produces the
    identical card — the export carries no
    wall-clock value anywhere.

**Docket**: TF-147 has NO share of TF-016's
D-WING-1 — the entry's current text (checked
live for this slice) assigns the verdicts
(TF-139..TF-142), the adoptions (TF-143), the
retirements (TF-144), research hygiene
(TF-148) and the wing's tests (TF-149); the
card export's ownership is filed in TF-141's
verdict conditions, not in D-WING-1. Per the
TF-145/TF-146 pattern's rule (a slice touches
the docket only for its own filed share), this
slice does NOT edit the docket and adds no
entry: TF-016 totals stand unchanged (137
entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100
FIXED / 3 CLOSED-BY-VERDICT), filed in the
open here and in the render.

**Neighbour coupling, in the open**: TF-139's
verdict checker greps the shipping knowledge +
cognition sources for bond / relationship
vocabulary (its uniqueness premise: the
persisted STORE files no bond record type).
This module legitimately files the compiled
persona's relationship excerpt and compiles
from TF-143's production bond edge, so TF-139's
verdict-module exclusion is extended to
``runmal_cards.py`` by name in
``bond_graph_verdict.py`` + its tests, exactly
as TF-143 extended it for the production
modules and TF-146 for the bridge — the
premise guarded is unchanged.

The human-readable write-up is
``docs/architecture/RUNMAL_CARDS.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.cognition.
runmal_cards``).
Proof grade: FIXTURE/STATIC — the production
compiler + synthetic fixtures re-measured live
on this VM; no live Rúnmál consumer imported
the card — a live consumer importing it is
NOT PROVEN (TF-156 owns the live binding),
never claimed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from thoughtforge.cognition.persona import FIXTURE_EXPECTATION
from thoughtforge.cognition.persona_compiler import (
    CompiledPersona,
    PersonaCompiler,
    PersonaMode,
    fixture_bond_edge,
    fixture_records,
)

RUNMAL_CARDS_FORMAT = "thoughtforge-runmal-cards/1"
RUNMAL_CARDS_DOC = "docs/architecture/RUNMAL_CARDS.md"
SOURCE_FORMAT = "thoughtforge-persona-compiler/1"
CARD_SPEC = "chara_card_v2"
CARD_SPEC_VERSION = "2.0"
EXTENSION_KEY = "thoughtforge_compiled_persona"

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The card envelope's top-level keys, exactly.
CARD_ENVELOPE_KEYS: tuple[str, ...] = ("spec", "spec_version", "data")

#: The public character-card v2 data fields, in
#: the spec's filing order — every one of them
#: has a row in CARD_FIELD_ROWS.
CARD_FIELDS: tuple[str, ...] = (
    "name", "description", "personality", "scenario",
    "first_mes", "mes_example", "creator_notes",
    "system_prompt", "post_history_instructions",
    "alternate_greetings", "tags", "creator",
    "character_version", "extensions",
)

#: The data fields filed empty for want of an
#: honest compiled source — never fabricated.
FILED_EMPTY_FIELDS: tuple[str, ...] = (
    "scenario", "first_mes", "mes_example",
    "post_history_instructions", "alternate_greetings",
)

#: The compiled facets that have no card home
#: and ride in the extension, structured.
EXTENSION_FACETS: tuple[str, ...] = (
    "truth_anchor_points", "uncertainty_points",
    "symbolic_context", "relationship_excerpt",
    "source_items", "active_traits",
)


# ── Every card field, its source + disposition, as literal data ──────────────

CARD_FIELD_ROWS: tuple[dict[str, Any], ...] = (
    {"field": "name",
     "source": "CompiledPersona.identity_core — the "
               "\"<subject>.name=<value>\" identity line "
               "when present, else persona_id",
     "disposition": "mapped",
     "rule": "the value after \".name=\" in the identity "
             "line, verbatim; the fallback persona_id is "
             "filed, never an invented name"},
    {"field": "description",
     "source": "CompiledPersona.identity_core (all lines)",
     "disposition": "mapped",
     "rule": "the identity lines joined with \"; \" — the "
             "compiled identity, verbatim, nothing added"},
    {"field": "personality",
     "source": "CompiledPersona.active_traits (TraitSignal: "
               "name / weight / supporting record ids) + "
               "CompiledPersona.tone_contract",
     "disposition": "mapped",
     "rule": "one \"<trait> (weight <w:.2f>; evidence: "
             "<ids>)\" per trait, joined with \"; \", then "
             "a \"Tone: \" line filing the tone contract "
             "verbatim — the weights and their evidence "
             "ride in the text, not just the trait names"},
    {"field": "scenario",
     "source": "filed empty — no compiled source",
     "disposition": "filed-empty",
     "rule": "the empty string: no compiled facet files "
             "a scenario, and scenario prose is never "
             "fabricated to fill a card"},
    {"field": "first_mes",
     "source": "filed empty — no compiled source",
     "disposition": "filed-empty",
     "rule": "the empty string: the compiler files no "
             "greeting or dialogue, and a first message "
             "is never fabricated"},
    {"field": "mes_example",
     "source": "filed empty — no compiled source",
     "disposition": "filed-empty",
     "rule": "the empty string: no compiled facet files "
             "example dialogue, and example dialogue is "
             "never fabricated"},
    {"field": "creator_notes",
     "source": "the export itself: RUNMAL_CARDS_FORMAT + "
               "the compiled persona's ids / mode + every "
               "source_items record id (provenance)",
     "disposition": "mapped",
     "rule": "one deterministic note filing the format, "
             "the persona / user / mode it was compiled "
             "from, the full provenance record-id list, "
             "the filed-empty honesty rule, and the "
             "filed limit that a live Rúnmál consumer "
             "importing the card is NOT PROVEN (TF-156)"},
    {"field": "system_prompt",
     "source": "CompiledPersonaExpectation.steering_lines() "
               "— the merged expectation TF-030's "
               "inject_persona delivers",
     "disposition": "mapped",
     "rule": "the steering lines joined with newlines, "
             "reused from the expectation itself — the "
             "real delivered steering, never re-derived "
             "or paraphrased here"},
    {"field": "post_history_instructions",
     "source": "filed empty — no compiled source",
     "disposition": "filed-empty",
     "rule": "the empty string: the compiled shape files "
             "no post-history instruction, and none is "
             "invented"},
    {"field": "alternate_greetings",
     "source": "filed empty — no compiled source",
     "disposition": "filed-empty",
     "rule": "the empty list: the compiler files no "
             "greetings, and greetings are never "
             "fabricated"},
    {"field": "tags",
     "source": "exporter-derived — the export's format "
               "family + the compiled persona's mode",
     "disposition": "derived",
     "rule": "[\"thoughtforge\", \"compiled-persona\", "
             "\"mode:<mode>\"] — provenance tags only, no "
             "compiled facet is summarised into a tag"},
    {"field": "creator",
     "source": "exporter constant — the producing "
               "pipeline, named",
     "disposition": "exporter-constant",
     "rule": "\"ThoughtForge persona compiler (TF-143), "
             "exported by TF-147\" — the pipeline that "
             "produced the card, not a persona claim"},
    {"field": "character_version",
     "source": "exporter constant — RUNMAL_CARDS_FORMAT",
     "disposition": "exporter-constant",
     "rule": "the export format string itself, so the "
             "card files the version of the shape that "
             "produced it"},
    {"field": "extensions",
     "source": "CompiledPersona facets with no card home, "
               "under the namespaced key "
               "thoughtforge_compiled_persona",
     "disposition": "extension",
     "rule": "the structured facets verbatim: identity / "
             "tone / guidance beside the six "
             "EXTENSION_FACETS (truth anchors, uncertainty, "
             "symbolic context, the relationship excerpt, "
             "source items with confidences, structured "
             "trait weights) + the merged expectation's "
             "declared side — nothing the compiler filed "
             "is lost"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def field_row_problems(row: Any) -> list[str]:
    """Judge one card-field filing, purely: a
    field, its source, its disposition and its
    rule must all be filed. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["field row: the row is empty"]
    problems: list[str] = []
    for key in ("field", "source", "disposition", "rule"):
        if not row.get(key):
            problems.append(
                f"field row {row.get('field')}: no {key} filed")
    return problems


def _expected_name(compiled: CompiledPersona) -> str:
    for line in compiled.identity_core:
        if ".name=" in line:
            return line.split(".name=", 1)[1]
    return compiled.persona_id


def _expected_personality(compiled: CompiledPersona) -> str:
    traits = "; ".join(
        f"{t.trait_name} (weight {t.weight:.2f}; evidence: "
        f"{', '.join(t.supporting_record_ids)})"
        for t in compiled.active_traits)
    return traits + "\nTone: " + "; ".join(compiled.tone_contract)


def _expected_extension(compiled: CompiledPersona) -> dict[str, Any]:
    expectation = compiled.to_expectation()
    return {
        "format": SOURCE_FORMAT,
        "persona_id": compiled.persona_id,
        "user_id": compiled.user_id,
        "mode": compiled.mode.value,
        "identity_core": list(compiled.identity_core),
        "tone_contract": list(compiled.tone_contract),
        "response_guidance": list(compiled.response_guidance),
        "truth_anchor_points": list(compiled.truth_anchor_points),
        "uncertainty_points": list(compiled.uncertainty_points),
        "symbolic_context": list(compiled.symbolic_context),
        "relationship_excerpt": list(compiled.relationship_excerpt),
        "active_traits": [
            {"trait_name": t.trait_name,
             "weight": round(t.weight, 4),
             "supporting_record_ids": list(t.supporting_record_ids)}
            for t in compiled.active_traits],
        "source_items": [
            {"record_id": s.record_id, "item_type": s.item_type,
             "text": s.text, "confidence": s.confidence}
            for s in compiled.source_items],
        "expectation": {
            "record_id": expectation.record_id,
            "base_record_id": expectation.base_record_id,
            "traits": list(expectation.traits),
            "speech_style": list(expectation.speech_style),
            "behavior_rules": list(expectation.behavior_rules),
            "avoid": list(expectation.avoid),
            "weights": dict(expectation.weights),
        },
    }


def card_problems(card: Any) -> list[str]:
    """Judge one card's SHAPE, purely: the v2
    envelope exactly, every data field present,
    no unknown field anywhere, the string / list
    / dict types the spec files, and the
    filed-empty fields actually empty. Empty-safe."""
    if not isinstance(card, dict) or not card:
        return ["card: the card is empty"]
    problems: list[str] = []
    for key in card:
        if key not in CARD_ENVELOPE_KEYS:
            problems.append(f"card: unknown top-level field {key!r}")
    if card.get("spec") != CARD_SPEC:
        problems.append(
            f"card: spec is {card.get('spec')!r}, not {CARD_SPEC!r}")
    if card.get("spec_version") != CARD_SPEC_VERSION:
        problems.append(
            f"card: spec_version is {card.get('spec_version')!r}, "
            f"not {CARD_SPEC_VERSION!r}")
    data = card.get("data")
    if not isinstance(data, dict):
        return problems + ["card: data is not an object"]
    for field_name in CARD_FIELDS:
        if field_name not in data:
            problems.append(f"card: required field {field_name} is missing")
    for key in data:
        if key not in CARD_FIELDS:
            problems.append(f"card: unknown data field {key!r}")
    for field_name in ("name", "description", "personality",
                       "scenario", "first_mes", "mes_example",
                       "creator_notes", "system_prompt",
                       "post_history_instructions", "creator",
                       "character_version"):
        if field_name in data and not isinstance(data[field_name], str):
            problems.append(f"card: {field_name} is not a string")
    for field_name in ("alternate_greetings", "tags"):
        if field_name in data and not isinstance(data[field_name], list):
            problems.append(f"card: {field_name} is not a list")
    if "extensions" in data and not isinstance(data["extensions"], dict):
        problems.append("card: extensions is not an object")
    for field_name in FILED_EMPTY_FIELDS:
        value = data.get(field_name)
        if value not in ("", []):
            problems.append(
                f"card: {field_name} files {value!r} — it has no "
                "compiled source and must be filed empty, never "
                "fabricated")
    return problems


def export_problems(card: Any, compiled: Any = None) -> list[str]:
    """Judge one export, purely: the card shape,
    and — when the compiled persona it was built
    from is supplied — that the card AGREES with
    it: the name against the compiled identity,
    description / personality / system_prompt
    recomputed from the compiled facets, the
    extension facet for facet, and the provenance
    record ids all filed in creator_notes.
    Empty-safe."""
    problems = card_problems(card)
    if compiled is None or not isinstance(card, dict):
        return problems
    data = card.get("data") or {}
    if data.get("name") != _expected_name(compiled):
        problems.append(
            f"export: name files {data.get('name')!r}, not the "
            f"compiled identity's {_expected_name(compiled)!r}")
    if data.get("description") != "; ".join(compiled.identity_core):
        problems.append(
            "export: description does not file the compiled "
            "identity_core verbatim")
    if data.get("personality") != _expected_personality(compiled):
        problems.append(
            "export: personality does not file the compiled "
            "traits + tone contract")
    steering = "\n".join(compiled.to_expectation().steering_lines())
    if data.get("system_prompt") != steering:
        problems.append(
            "export: system_prompt is not the merged "
            "expectation's steering_lines() verbatim")
    extensions = data.get("extensions") or {}
    filed = extensions.get(EXTENSION_KEY)
    expected = _expected_extension(compiled)
    if not isinstance(filed, dict):
        problems.append(
            f"export: extensions carries no {EXTENSION_KEY} object")
    else:
        for facet, value in expected.items():
            if filed.get(facet) != value:
                problems.append(
                    f"export: extension facet {facet} disagrees "
                    "with the compiled persona")
        for facet in filed:
            if facet not in expected:
                problems.append(
                    f"export: extension facet {facet} is unknown")
    notes = data.get("creator_notes") or ""
    for item in compiled.source_items:
        if item.record_id not in notes:
            problems.append(
                f"export: provenance missing — source record "
                f"{item.record_id} is not filed in creator_notes")
    if "NOT PROVEN" not in notes:
        problems.append(
            "export: creator_notes files no NOT PROVEN limit")
    return problems


# ── The export itself: a card built FROM a compiled persona ──────────────────

def _creator_notes(compiled: CompiledPersona) -> str:
    ids = ", ".join(s.record_id for s in compiled.source_items)
    return (
        f"Exported from a ThoughtForge compiled persona "
        f"(TF-143) by TF-147 under {RUNMAL_CARDS_FORMAT}; "
        f"persona_id={compiled.persona_id}; "
        f"user_id={compiled.user_id}; "
        f"mode={compiled.mode.value}. Provenance — compiled "
        f"from source records: {ids}. Filed limits: fields "
        f"with no compiled source are filed empty, never "
        f"fabricated (see CARD_FIELD_ROWS in "
        f"thoughtforge/cognition/runmal_cards.py); compiled "
        f"facets with no card home ride in "
        f"extensions.{EXTENSION_KEY}. Fixture/static grade "
        f"— a live Rúnmál consumer importing this card is "
        f"NOT PROVEN (TF-156 owns the live binding).")


def export_card(compiled: CompiledPersona) -> dict[str, Any]:
    """Export one compiled persona as a Rúnmál
    card. Every value is read from the compiled
    persona or its merged expectation: the name
    from the compiled identity, the description
    from identity_core, the personality from the
    weighted traits + tone, the system_prompt
    from steering_lines() itself. Nothing is
    fabricated: the five fields with no compiled
    source are filed empty. The result is plain
    data — JSON-serialisable as it stands, with
    no wall-clock value anywhere."""
    return {
        "spec": CARD_SPEC,
        "spec_version": CARD_SPEC_VERSION,
        "data": {
            "name": _expected_name(compiled),
            "description": "; ".join(compiled.identity_core),
            "personality": _expected_personality(compiled),
            "scenario": "",
            "first_mes": "",
            "mes_example": "",
            "creator_notes": _creator_notes(compiled),
            "system_prompt": "\n".join(
                compiled.to_expectation().steering_lines()),
            "post_history_instructions": "",
            "alternate_greetings": [],
            "tags": ["thoughtforge", "compiled-persona",
                     f"mode:{compiled.mode.value}"],
            "creator": "ThoughtForge persona compiler (TF-143), "
                       "exported by TF-147",
            "character_version": RUNMAL_CARDS_FORMAT,
            "extensions": {EXTENSION_KEY: _expected_extension(compiled)},
        },
    }


# ── The export proof on the fixture compiled persona ─────────────────────────

def fixture_compiled() -> CompiledPersona:
    """TF-143's main fixture compile, rebuilt
    through the PRODUCTION compiler exactly as
    ``run_persona_compile`` runs it (the Wrenna
    records + the production bond edge + the
    authored base expectation; synthetic only,
    TF-017)."""
    return PersonaCompiler().compile(
        persona_id="persona:wrenna",
        user_id="user:lantern-keeper",
        mode=PersonaMode.COMPANION,
        records=fixture_records(),
        bond_edge=fixture_bond_edge(),
        base_expectation=FIXTURE_EXPECTATION)


#: The filed expectation for the fixture export —
#: deterministic values only (counts, the name,
#: the facet sizes); no wall-clock value exists
#: anywhere in the export to file.
CARD_EXPECTED: dict[str, Any] = {
    "fields_filed": 14,
    "fields_mapped": 5,
    "fields_filed_empty": 5,
    "fields_derived": 1,
    "fields_exporter_constant": 2,
    "fields_extension": 1,
    "extension_facets": 6,
    "name": "Wrenna the Lantern-Keeper",
    "source_records": 14,
    "traits": 3,
    "steering_lines": 11,
    "truth_anchor_points": 4,
    "uncertainty_points": 3,
    "symbolic_context": 2,
    "relationship_excerpt": 3,
}


def run_runmal_cards() -> dict[str, Any]:
    """Compile the fixture persona, export its
    card, and file the deterministic metrics of
    the export."""
    compiled = fixture_compiled()
    card = export_card(compiled)
    data = card["data"]
    extension = data["extensions"][EXTENSION_KEY]
    by_disposition: dict[str, int] = {}
    for row in CARD_FIELD_ROWS:
        by_disposition[row["disposition"]] = (
            by_disposition.get(row["disposition"], 0) + 1)
    metrics = {
        "fields_filed": len(CARD_FIELD_ROWS),
        "fields_mapped": by_disposition.get("mapped", 0),
        "fields_filed_empty": by_disposition.get("filed-empty", 0),
        "fields_derived": by_disposition.get("derived", 0),
        "fields_exporter_constant":
            by_disposition.get("exporter-constant", 0),
        "fields_extension": by_disposition.get("extension", 0),
        "extension_facets": len(EXTENSION_FACETS),
        "name": data["name"],
        "source_records": len(compiled.source_items),
        "traits": len(compiled.active_traits),
        "steering_lines": len(
            compiled.to_expectation().steering_lines()),
        "truth_anchor_points": len(extension["truth_anchor_points"]),
        "uncertainty_points": len(extension["uncertainty_points"]),
        "symbolic_context": len(extension["symbolic_context"]),
        "relationship_excerpt": len(extension["relationship_excerpt"]),
    }
    return {"metrics": metrics, "card": card}


# ── The filed probes ─────────────────────────────────────────────────────────

RUNMAL_CARDS_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_export_passes",
     "kind": "clean",
     "detail": "the fixture card + the filed field rows pass "
               "their validators against the compiled persona, "
               "and the card JSON-round-trips byte-stable"},
    {"id": "missing_required_field_is_named",
     "kind": "missing_field",
     "detail": "a card with its name field removed is named"},
    {"id": "unknown_field_is_named",
     "kind": "unknown_field",
     "detail": "a card carrying an unknown data field is "
               "named — no field rides unfiled"},
    {"id": "field_row_missing_disposition_is_named",
     "kind": "row_no_disposition",
     "detail": "a field row with its disposition emptied is "
               "named — every field must file its disposition"},
    {"id": "extension_disagreement_is_named",
     "kind": "extension_drift",
     "detail": "a card whose extension truth anchors are "
               "perturbed is named against the compiled "
               "persona"},
    {"id": "name_mismatch_is_named",
     "kind": "name_wrong",
     "detail": "a card whose name does not match the "
               "compiled identity is named"},
    {"id": "provenance_missing_is_named",
     "kind": "provenance_missing",
     "detail": "a card whose creator_notes drops a source "
               "record id is named — provenance is filed, "
               "never assumed"},
    {"id": "fabricated_scenario_is_named",
     "kind": "fabricated_empty",
     "detail": "a card whose scenario is filled with prose "
               "is named — a field with no compiled source "
               "stays empty"},
)


def run_runmal_cards_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    deep copies."""
    compiled = fixture_compiled()
    fixture = export_card(compiled)
    results: list[dict[str, Any]] = []
    for probe in RUNMAL_CARDS_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems = export_problems(fixture, compiled)
            for row in CARD_FIELD_ROWS:
                problems += field_row_problems(row)
            serialised = json.dumps(fixture, sort_keys=True)
            round_tripped = json.loads(serialised)
            passed = problems == [] \
                and json.dumps(round_tripped, sort_keys=True) \
                == serialised
        elif kind == "missing_field":
            data = copy.deepcopy(fixture)
            del data["data"]["name"]
            passed = bool(card_problems(data))
        elif kind == "unknown_field":
            data = copy.deepcopy(fixture)
            data["data"]["invented_field"] = "x"
            passed = bool(card_problems(data))
        elif kind == "row_no_disposition":
            row = dict(CARD_FIELD_ROWS[0], disposition="")
            passed = bool(field_row_problems(row))
        elif kind == "extension_drift":
            data = copy.deepcopy(fixture)
            data["data"]["extensions"][EXTENSION_KEY][
                "truth_anchor_points"] = ["invented anchor"]
            passed = bool(export_problems(data, compiled))
        elif kind == "name_wrong":
            data = copy.deepcopy(fixture)
            data["data"]["name"] = "Somebody Else"
            passed = bool(export_problems(data, compiled))
        elif kind == "provenance_missing":
            data = copy.deepcopy(fixture)
            data["data"]["creator_notes"] = "no provenance filed"
            passed = bool(export_problems(data, compiled))
        elif kind == "fabricated_empty":
            data = copy.deepcopy(fixture)
            data["data"]["scenario"] = "A tavern, probably."
            passed = bool(card_problems(data))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_runmal_cards_markdown() -> str:
    """Render the cards filing deterministically —
    the exact text filed as
    docs/architecture/RUNMAL_CARDS.md."""
    run = run_runmal_cards()
    metrics = run["metrics"]
    card = run["card"]
    lines: list[str] = []
    lines.append("# Persona Compiler -> Rúnmál Cards — compiled personae in Rúnmál's card shape (TF-147)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/cognition/runmal_cards.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.cognition.runmal_cards`. The "
        "checker (`runmal_cards_problems()`) re-derives every "
        "claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{RUNMAL_CARDS_FORMAT}`")
    lines.append(f"Source format: `{SOURCE_FORMAT}` (TF-143)")
    lines.append(f"Card envelope: `{CARD_SPEC}` / spec_version `{CARD_SPEC_VERSION}`")
    lines.append("Antecedent: TF-141 **MERGE** (adopt-in-part) — TF-147 is live work on TF-143's merged compiled shape")
    lines.append(f"Card fields filed: {len(CARD_FIELD_ROWS)}")
    lines.append(f"Probes: {len(RUNMAL_CARDS_PROBES)}/{len(RUNMAL_CARDS_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: "
        "**no card export existed** — no card module under "
        "`src/thoughtforge/cognition/`, no card doc, and the "
        "production `CompiledPersona`'s only outward shapes "
        "were its dataclass fields and the merged "
        "expectation's `steering_lines()`: nothing in the "
        "public character-card shape a Rúnmál-side consumer "
        "imports and exports. The antecedent stood clean "
        "live: TF-141's verdict MERGE "
        "(`persona_compiler_verdict_problems()` []), TF-143's "
        "production compiler clean (main run identity 4 / "
        "anchors 4 / traits 3 / source items 14), and "
        "TF-146's bridge clean beside it. TF-141's verdict "
        "doc files the condition verbatim: TF-147 consumes "
        "the MERGED compiled output — live work on the "
        "merged shape, not on the wing's packet as-is. This "
        "module exports TF-143's `CompiledPersona`; no "
        "Rúnmál code is copied or read as code — the card "
        "shape's field names are the public character-card "
        "v2 spec's.")
    lines.append("")
    lines.append("## 2. The card shape — every field filed")
    lines.append("")
    lines.append(
        "`export_card(compiled)` builds the card FROM a real "
        "`CompiledPersona` compiled by TF-143's production "
        "`PersonaCompiler`. The envelope is "
        "`{\"spec\": \"chara_card_v2\", \"spec_version\": "
        "\"2.0\", \"data\": {...}}`; every data field files "
        "its source and disposition below — none silently "
        "dropped, none invented.")
    lines.append("")
    lines.append("| Field | Source in the compiled persona | Disposition | Mapping rule |")
    lines.append("|---|---|---|---|")
    for row in CARD_FIELD_ROWS:
        lines.append(
            f"| {row['field']} | {row['source']} | "
            f"{row['disposition']} | {row['rule']} |")
    lines.append("")
    lines.append(
        f"Filed counts: {metrics['fields_filed']} fields — "
        f"{metrics['fields_mapped']} mapped, "
        f"{metrics['fields_filed_empty']} filed empty, "
        f"{metrics['fields_extension']} extension, "
        f"{metrics['fields_derived']} derived, "
        f"{metrics['fields_exporter_constant']} exporter "
        "constants.")
    lines.append("")
    lines.append("## 3. Export proof (synthetic fixtures, TF-017)")
    lines.append("")
    lines.append(
        "TF-143's main fixture compile (the Wrenna records + "
        "the production bond edge + the authored base "
        "expectation, exactly `run_persona_compile`'s main "
        "run) is exported, and the card is proven to carry "
        "the filing's claims:")
    lines.append("")
    lines.append(f"- name: {metrics['name']} (from the compiled identity's name fact)")
    lines.append(
        f"- provenance: {metrics['source_records']} source "
        "records, every record id filed in creator_notes")
    lines.append(
        f"- system_prompt: the merged expectation's "
        f"{metrics['steering_lines']} steering lines verbatim "
        "— the real delivered steering, reused")
    lines.append(
        f"- personality: {metrics['traits']} evidence-weighted "
        "traits with their weights + supporting ids, plus "
        "the tone contract")
    lines.append(
        f"- extension facets ({metrics['extension_facets']}): "
        f"truth anchors {metrics['truth_anchor_points']} / "
        f"uncertainty {metrics['uncertainty_points']} / "
        f"symbolic {metrics['symbolic_context']} / "
        f"relationship lines {metrics['relationship_excerpt']} "
        "/ source items + structured trait weights — nothing "
        "the compiler filed is lost")
    lines.append("")
    lines.append("The exported card, verbatim:")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(card, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append(
        "The card JSON-round-trips byte-stable "
        "(`json.dumps(card, sort_keys=True)` re-serialised "
        "identically after a parse), and a second compile + "
        "export produces the identical card — the export "
        "carries no wall-clock value anywhere.")
    lines.append("")
    lines.append("## 4. Filed empty — never fabricated")
    lines.append("")
    lines.append(
        "Five fields have no honest compiled source — "
        "`scenario`, `first_mes`, `mes_example`, "
        "`post_history_instructions`, "
        "`alternate_greetings` — and are filed empty "
        "(empty string / empty list) with the reason in the "
        "table above. The compiler files no scenario, no "
        "greeting and no dialogue, so the card invents "
        "none: the pure validator names any export that "
        "fills one of them (probe "
        "`fabricated_scenario_is_named`). A consumer that "
        "needs those fields authors them itself, outside "
        "this export.")
    lines.append("")
    lines.append("## 5. Neighbour coupling, in the open")
    lines.append("")
    lines.append(
        "TF-139's verdict checker greps the shipping "
        "knowledge + cognition sources for bond / "
        "relationship vocabulary (its uniqueness premise: "
        "the persisted store files no bond record type). "
        "This module legitimately files the compiled "
        "persona's relationship excerpt and compiles from "
        "TF-143's production bond edge, so TF-139's "
        "verdict-module exclusion is extended to "
        "`runmal_cards.py` by name in "
        "`bond_graph_verdict.py` + its tests, exactly as "
        "TF-143 extended it for the production modules and "
        "TF-146 for the bridge. The premise guarded is "
        "unchanged: the persisted store still files no "
        "bond record type, checked there still.")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-147 has **no share** of TF-016's D-WING-1: the "
        "entry's current text (checked live for this slice) "
        "assigns the verdicts (TF-139..TF-142), the "
        "adoptions (TF-143), the retirements (TF-144), "
        "research hygiene (TF-148) and the wing's tests "
        "(TF-149) — the card export's ownership is filed in "
        "TF-141's verdict conditions, not in D-WING-1. Per "
        "the TF-145/TF-146 pattern's rule (a slice touches "
        "the docket only for its own filed share), this "
        "slice does not edit the docket and adds no entry; "
        "TF-016 totals stand unchanged (137 entries — 33 "
        "OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 "
        "CLOSED-BY-VERDICT), filed in the open here.")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in RUNMAL_CARDS_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Fixture grade only: the export proof runs on "
        "synthetic Wrenna fixtures (TF-017); no live Rúnmál "
        "consumer was read — a live consumer importing the "
        "card is NOT PROVEN (TF-156 owns the live binding), "
        "never claimed.")
    lines.append(
        "- The export is a shape, not a binding: no Rúnmál "
        "call is routed anywhere here (TF-156 owns the "
        "binding, behind its backend table, off by "
        "default).")
    lines.append(
        "- The card's system_prompt is TF-030/TF-143's "
        "delivered steering reused verbatim — this module "
        "files no steering of its own.")
    lines.append(
        "- No canon data was touched; no README.md anywhere "
        "was touched.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_runmal_cards_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / RUNMAL_CARDS_DOC
    doc.write_text(render_runmal_cards_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.cognition.runmal_cards`."""
    doc = write_runmal_cards_doc()
    print(f"runmal cards regenerated: -> {RUNMAL_CARDS_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def runmal_cards_problems() -> list[str]:
    """Re-derive TF-147's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    row_fields = [r["field"] for r in CARD_FIELD_ROWS]
    if len(row_fields) != len(set(row_fields)) \
            or tuple(row_fields) != CARD_FIELDS:
        problems.append(
            "the CARD_FIELD_ROWS filing is not the fourteen "
            "card fields, unique, in spec order")
    for row in CARD_FIELD_ROWS:
        problems.extend(field_row_problems(row))
    dispositions = {r["field"]: r["disposition"]
                    for r in CARD_FIELD_ROWS}
    for field_name in FILED_EMPTY_FIELDS:
        if dispositions.get(field_name) != "filed-empty":
            problems.append(
                f"filing: {field_name} is not filed filed-empty")

    # The antecedent: TF-141's verdict was MERGE —
    # without it this export is not live work on
    # the merged shape.
    from thoughtforge.cognition.persona_compiler_verdict import (
        VERDICT as PERSONA_VERDICT,
    )

    if PERSONA_VERDICT.get("verdict") != "MERGE":
        problems.append(
            "antecedent: TF-141 no longer files MERGE — the "
            "card export's condition is gone")

    # The fixture export carries the filing exactly:
    # valid against its own compiled persona,
    # byte-stable, deterministic.
    compiled = fixture_compiled()
    run = run_runmal_cards()
    card = run["card"]
    problems.extend(export_problems(card, compiled))
    if run["metrics"] != CARD_EXPECTED:
        problems.append(
            f"fixture: the export files {run['metrics']}, not "
            f"the filed {CARD_EXPECTED}")
    serialised = json.dumps(card, sort_keys=True)
    if json.dumps(json.loads(serialised), sort_keys=True) \
            != serialised:
        problems.append(
            "fixture: the card does not JSON-round-trip "
            "byte-stable under sort_keys")
    if export_card(fixture_compiled()) != card:
        problems.append(
            "fixture: a second compile + export produces a "
            "different card — the export must be deterministic")

    # The probes all behave.
    probes = run_runmal_cards_probes()
    if len(probes) != len(RUNMAL_CARDS_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the runmal-cards probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / RUNMAL_CARDS_DOC
    if not doc.exists():
        problems.append(f"the write-up {RUNMAL_CARDS_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != \
            render_runmal_cards_markdown():
        problems.append(
            f"the filed doc {RUNMAL_CARDS_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.cognition.runmal_cards)")
    return problems


def runmal_cards_as_data() -> dict[str, Any]:
    """The whole cards filing as plain data, for
    consumers."""
    run = run_runmal_cards()
    probes = run_runmal_cards_probes()
    return {
        "format": RUNMAL_CARDS_FORMAT,
        "doc": RUNMAL_CARDS_DOC,
        "source_format": SOURCE_FORMAT,
        "card_spec": CARD_SPEC,
        "card_spec_version": CARD_SPEC_VERSION,
        "fields": [dict(r) for r in CARD_FIELD_ROWS],
        "fixture_metrics": run["metrics"],
        "fixture_card": run["card"],
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "CARD_ENVELOPE_KEYS",
    "CARD_EXPECTED",
    "CARD_FIELDS",
    "CARD_FIELD_ROWS",
    "CARD_SPEC",
    "CARD_SPEC_VERSION",
    "EXTENSION_FACETS",
    "EXTENSION_KEY",
    "FILED_EMPTY_FIELDS",
    "RUNMAL_CARDS_DOC",
    "RUNMAL_CARDS_FORMAT",
    "RUNMAL_CARDS_PROBES",
    "SOURCE_FORMAT",
    "card_problems",
    "export_card",
    "export_problems",
    "field_row_problems",
    "fixture_compiled",
    "main",
    "render_runmal_cards_markdown",
    "run_runmal_cards",
    "run_runmal_cards_probes",
    "runmal_cards_as_data",
    "runmal_cards_problems",
    "write_runmal_cards_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
