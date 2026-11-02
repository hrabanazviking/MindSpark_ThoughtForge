"""
TF-147 — Persona compiler -> Rúnmál cards (if
adopted at TF-141): compiled personae exportable
in Rúnmál's card shape.

The antecedent is TRUE in part, exactly as
TF-141 filed it: the verdict was MERGE
(adopt-in-part), its conditions file TF-147 as
live work on the MERGED compiled shape, and
TF-143 landed that shape as production code.
The proof lives as verifiable data in
``src/thoughtforge/cognition/runmal_cards.py``
(``CARD_FIELD_ROWS``, ``CARD_EXPECTED``,
``RUNMAL_CARDS_PROBES``, checked by
``runmal_cards_problems()``); the write-up
``docs/architecture/RUNMAL_CARDS.md`` is filed
verbatim as the module's render.

Pinned here:

  * the antecedent + the source: TF-141 files
    MERGE, and the export is built FROM TF-143's
    production CompiledPersona — its values are
    the production compile's, unmoved,
  * the card shape: the public character-card
    v2 envelope, all fourteen data fields filed
    with a source + disposition + rule — none
    dropped, none invented,
  * the mapping: name from the compiled
    identity, description from identity_core,
    personality from the weighted traits + tone,
    system_prompt from the merged expectation's
    steering_lines() verbatim, creator_notes
    carrying the provenance record ids,
  * the filed-empty honesty: scenario /
    first_mes / mes_example /
    post_history_instructions /
    alternate_greetings empty — a filled one is
    named, never fabricated,
  * the extension: the six facets with no card
    home ride structured under the
    thoughtforge namespaced key, agreeing with
    the compiled persona facet for facet,
  * the round trip: the card JSON-round-trips
    byte-stable under sort_keys, and a second
    compile + export is identical,
  * the coupling in the open: TF-139's
    verdict-module exclusion extended to
    runmal_cards.py (the TF-143/TF-146 pattern),
    its checker still clean,
  * the docket: TF-147 files NO D-WING-1 share
    — the entry's text assigns no card share,
    so the docket file is untouched by this
    slice,
  * the probes: 8/8 against the pure validators,
  * the doc matches the render verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live Rúnmál
consumer, never claimed (TF-156 owns the live
binding).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from thoughtforge.cognition.persona_compiler import run_persona_compile
from thoughtforge.cognition.runmal_cards import (
    CARD_EXPECTED,
    CARD_FIELDS,
    CARD_FIELD_ROWS,
    CARD_SPEC,
    CARD_SPEC_VERSION,
    EXTENSION_FACETS,
    EXTENSION_KEY,
    FILED_EMPTY_FIELDS,
    RUNMAL_CARDS_DOC,
    RUNMAL_CARDS_FORMAT,
    RUNMAL_CARDS_PROBES,
    SOURCE_FORMAT,
    card_problems,
    export_card,
    export_problems,
    field_row_problems,
    fixture_compiled,
    render_runmal_cards_markdown,
    run_runmal_cards,
    run_runmal_cards_probes,
    runmal_cards_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _card() -> dict:
    return run_runmal_cards()["card"]


# ── The antecedent + the export, filed ───────────────────────────────────────

class TestCardsFiled:
    def test_problems_empty(self) -> None:
        assert runmal_cards_problems() == []

    def test_antecedent_is_merge(self) -> None:
        from thoughtforge.cognition.persona_compiler_verdict import (
            VERDICT,
        )

        assert VERDICT.get("verdict") == "MERGE"

    def test_every_field_files_a_disposition(self) -> None:
        assert [r["field"] for r in CARD_FIELD_ROWS] == \
            list(CARD_FIELDS)
        assert len(CARD_FIELDS) == 14
        for row in CARD_FIELD_ROWS:
            assert field_row_problems(row) == []
        by_disp: dict[str, int] = {}
        for row in CARD_FIELD_ROWS:
            by_disp[row["disposition"]] = \
                by_disp.get(row["disposition"], 0) + 1
        assert by_disp == {
            "mapped": 5, "filed-empty": 5, "derived": 1,
            "exporter-constant": 2, "extension": 1}

    def test_format_and_envelope_are_versioned(self) -> None:
        card = _card()
        assert RUNMAL_CARDS_FORMAT == "thoughtforge-runmal-cards/1"
        assert SOURCE_FORMAT == "thoughtforge-persona-compiler/1"
        assert card["spec"] == CARD_SPEC == "chara_card_v2"
        assert card["spec_version"] == CARD_SPEC_VERSION == "2.0"
        assert set(card) == {"spec", "spec_version", "data"}
        assert set(card["data"]) == set(CARD_FIELDS)


# ── The mapping carries the compiled persona ─────────────────────────────────

class TestMapping:
    def test_fixture_metrics_match_the_filing(self) -> None:
        run = run_runmal_cards()
        assert run["metrics"] == CARD_EXPECTED
        assert export_problems(
            run["card"], fixture_compiled()) == []

    def test_name_comes_from_the_compiled_identity(self) -> None:
        card = _card()
        compiled = fixture_compiled()
        assert card["data"]["name"] == \
            "Wrenna the Lantern-Keeper"
        assert "persona:wrenna.name=Wrenna the Lantern-Keeper" \
            in compiled.identity_core

    def test_export_comes_from_the_live_compile(self) -> None:
        # A compile without the name fact falls back
        # to the persona_id — the export reads the
        # compiled persona, it does not snapshot a
        # filing.
        from thoughtforge.cognition.persona_compiler import (
            PersonaCompiler,
            PersonaMode,
            fixture_records,
        )

        records = [r for r in fixture_records()
                   if r.record_id != "rec:synthetic-wrenna-002"]
        compiled = PersonaCompiler().compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper",
            mode=PersonaMode.COMPANION, records=records)
        card = export_card(compiled)
        assert card["data"]["name"] == "persona:wrenna"
        assert export_problems(card, compiled) == []

    def test_description_and_personality_are_the_facets(self) -> None:
        card = _card()
        compiled = fixture_compiled()
        assert card["data"]["description"] == \
            "; ".join(compiled.identity_core)
        personality = card["data"]["personality"]
        for trait in compiled.active_traits:
            assert trait.trait_name in personality
            assert f"{trait.weight:.2f}" in personality
            for record_id in trait.supporting_record_ids:
                assert record_id in personality
        assert "Tone: " in personality

    def test_system_prompt_is_the_delivered_steering(self) -> None:
        card = _card()
        compiled = fixture_compiled()
        steering = compiled.to_expectation().steering_lines()
        assert card["data"]["system_prompt"] == "\n".join(steering)
        assert len(steering) == 11

    def test_provenance_is_filed_in_creator_notes(self) -> None:
        card = _card()
        compiled = fixture_compiled()
        notes = card["data"]["creator_notes"]
        assert len(compiled.source_items) == 14
        for item in compiled.source_items:
            assert item.record_id in notes
        assert "NOT PROVEN" in notes
        assert "TF-156" in notes

    def test_compiled_run_is_tf143s_unmoved(self) -> None:
        run = run_persona_compile()
        assert run["main_counts"]["identity_core"] == 4
        assert run["main_counts"]["truth_anchor_points"] == 4
        assert run["main_counts"]["source_items"] == 14


# ── Filed empty + the extension ──────────────────────────────────────────────

class TestFiledEmptyAndExtension:
    def test_no_source_fields_are_filed_empty(self) -> None:
        data = _card()["data"]
        for field_name in FILED_EMPTY_FIELDS:
            assert data[field_name] in ("", [])
        assert [r["field"] for r in CARD_FIELD_ROWS
                if r["disposition"] == "filed-empty"] == \
            list(FILED_EMPTY_FIELDS)

    def test_a_filled_empty_field_is_named(self) -> None:
        card = _card()
        card["data"]["first_mes"] = "Well met, traveller."
        assert any("first_mes" in p for p in card_problems(card))

    def test_extension_carries_the_homeless_facets(self) -> None:
        compiled = fixture_compiled()
        extension = _card()["data"]["extensions"][EXTENSION_KEY]
        assert tuple(f for f in EXTENSION_FACETS
                     if f in extension) == EXTENSION_FACETS
        assert extension["truth_anchor_points"] == \
            compiled.truth_anchor_points
        assert extension["uncertainty_points"] == \
            compiled.uncertainty_points
        assert extension["symbolic_context"] == \
            compiled.symbolic_context
        assert extension["relationship_excerpt"] == \
            compiled.relationship_excerpt
        assert [s["record_id"] for s in extension["source_items"]] == \
            [s.record_id for s in compiled.source_items]
        assert extension["active_traits"][0] == {
            "trait_name": "steady", "weight": 1.0,
            "supporting_record_ids": [
                "rec:synthetic-wrenna-001",
                "rec:synthetic-wrenna-005"]}

    def test_extension_drift_is_named(self) -> None:
        card = _card()
        card["data"]["extensions"][EXTENSION_KEY][
            "uncertainty_points"] = []
        assert any("uncertainty_points" in p
                   for p in export_problems(card, fixture_compiled()))


# ── The round trip ───────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_card_json_round_trips_byte_stable(self) -> None:
        serialised = json.dumps(_card(), sort_keys=True)
        assert json.dumps(
            json.loads(serialised), sort_keys=True) == serialised

    def test_second_compile_exports_the_identical_card(self) -> None:
        assert export_card(fixture_compiled()) == _card()
        assert json.dumps(export_card(fixture_compiled()),
                          sort_keys=True) == \
            json.dumps(_card(), sort_keys=True)


# ── Coupling + the docket, handled in the open ───────────────────────────────

class TestCouplingAndDocket:
    def test_tf139_exclusion_names_this_module(self) -> None:
        verdict = _read(
            "src/thoughtforge/knowledge/bond_graph_verdict.py")
        tests = _read("tests/test_tf139_bond_graph_verdict.py")
        assert '"runmal_cards.py"' in verdict
        assert '"runmal_cards.py"' in tests

    def test_tf139_checker_still_clean(self) -> None:
        from thoughtforge.knowledge.bond_graph_verdict import (
            bond_graph_verdict_problems,
        )

        assert bond_graph_verdict_problems() == []

    def test_tf143_checker_still_clean(self) -> None:
        from thoughtforge.cognition.persona_compiler import (
            persona_compiler_problems,
        )

        assert persona_compiler_problems() == []

    def test_docket_file_untouched_by_this_slice(self) -> None:
        # D-WING-1's text assigns no card share
        # (verdicts / TF-143 / TF-144 / TF-148 /
        # TF-149 only), so TF-147 adds no narrowing
        # — the last narrowing stands TF-144's.
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        assert "Narrowed by TF-144" in docket
        assert "Narrowed by TF-147" not in docket
        assert "Narrowed by TF-146" not in docket


# ── Probes + validators + doc ────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_runmal_cards_probes()
        assert len(probes) == len(RUNMAL_CARDS_PROBES) == 8
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert card_problems(None)
        assert card_problems({})
        assert export_problems(None)
        assert field_row_problems(None)

    def test_each_probe_shape_is_named_by_the_validator(self) -> None:
        card = _card()
        broken = copy.deepcopy(card)
        del broken["data"]["system_prompt"]
        assert card_problems(broken)
        broken = copy.deepcopy(card)
        broken["data"]["bogus"] = 1
        assert card_problems(broken)
        broken = copy.deepcopy(card)
        broken["data"]["name"] = "Not Wrenna"
        assert export_problems(broken, fixture_compiled())

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / RUNMAL_CARDS_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_runmal_cards_markdown()

    def test_doc_files_the_honest_limits(self) -> None:
        doc = _read(RUNMAL_CARDS_DOC)
        assert "MERGE" in doc
        assert "NOT PROVEN" in doc
        assert "TF-156" in doc
        assert "no share" in doc
        assert "TF-017" in doc
        assert "never fabricated" in doc
