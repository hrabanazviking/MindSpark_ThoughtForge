"""
TF-143 — Adoptions implemented: each ADOPT verdict
lands as production code under the shipping tree's
discipline + tests.

The proof lives as verifiable data in
``src/thoughtforge/knowledge/bond_graph.py``
(``WEIGHTING_DECISION``, the production dataclasses +
``BondGraph``, ``BOND_GRAPH_PROBES``, checked by
``bond_graph_problems()``; write-up
``docs/architecture/BOND_GRAPH.md`` filed verbatim as
the module's render) and
``src/thoughtforge/cognition/persona_compiler.py``
(``AVOID_DECISION`` / ``TOKEN_BUDGET_DECISION`` /
``DEFECT_DIFFERENCES``, ``PersonaCompiler`` +
``CompiledPersona`` / ``CompiledPersonaExpectation``,
``PERSONA_COMPILER_PROBES``, checked by
``persona_compiler_problems()``; write-up
``docs/architecture/PERSONA_COMPILER.md``).

Pinned here:

  * the bond layer: production field sets + enums
    match the schemas live, the severity arithmetic /
    clamps / ACTIVE threshold / excerpt shapes are the
    wing shell's, parity with TF-139's spike is exact
    (status forming->active->repairing, repair_debt
    0.35 / safety 0.725, closeness 0.4375 / sacred
    0.5333 / rupture 0.2083, excerpt 9 lines verbatim),
    the weighting decision is equal weights filed
    WITH evidence + limits, the layer is not a
    persisted record type and imports no wing code,
  * the persona merge: the production compile carries
    every adopted facet with the compiler's caps +
    trait arithmetic (steady 1.0 at the cap, both
    supporting ids), the three defects are NOT
    imported — approval first (anchors 4 /
    uncertainty 3 against the spike's 5 / 2, the
    pending bell_count fact an uncertainty point,
    never an anchor; an unapproved symbolic trace
    likewise), relationship_excerpt bond-only from
    the PRODUCTION BondEdge, token_budget_hint
    dropped (no such field), avoid declared-only,
  * the merge delivery: the compiled expectation IS
    a PersonaExpectation, inject_persona delivers
    base + compiled steering into a real scaffold,
    score_persona_turns judges the fixture turns
    exactly as TF-030 filed (5 / 0.4 / REVIEW), the
    authored path stands unmodified, compiling
    without a base files an empty avoid,
  * both checkers clean, both probe sets 7/7, both
    docs match their renders verbatim, the wing's
    models package still does NOT exist (TF-143
    resurrects nothing), all fixtures synthetic.

Proof grade: FIXTURE/STATIC — no live model was
read, never claimed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from thoughtforge.cognition.persona import (
    FIXTURE_EXPECTATION,
    PersonaExpectation,
    check_persona_response,
    inject_persona,
    score_persona_turns,
)
from thoughtforge.cognition.persona_compiler import (
    AVOID_DECISION,
    CAPS,
    COMPILE_EXPECTED,
    DEFECT_DIFFERENCES,
    PERSONA_COMPILER_DOC,
    PERSONA_COMPILER_PROBES,
    TOKEN_BUDGET_DECISION,
    ApprovalState,
    CanonicalFactRecord,
    CompiledPersona,
    CompiledPersonaExpectation,
    PersonaCompiler,
    PersonaMode,
    SymbolicTraceRecord,
    caps_records,
    fixture_bond_edge,
    fixture_records,
    persona_compiler_problems,
    production_field_map,
    render_persona_compiler_markdown,
    run_persona_compile,
    run_persona_compiler_probes,
)
from thoughtforge.knowledge.bond_graph import (
    BOND_GRAPH_DOC,
    BOND_GRAPH_PROBES,
    PARITY_EXPECTED,
    PARITY_EXPECTED_EXCERPT,
    SEVERITY_SHIFTS,
    WEIGHTING_DECISION,
    BondDomain,
    BondEdge,
    BondGraph,
    BondStatus,
    BondVector,
    Hurt,
    Vow,
    VowState,
    bond_graph_problems,
    parity_edge,
    production_field_map as bond_field_map,
    render_bond_graph_markdown,
    run_bond_graph_probes,
    run_bond_parity,
)
from thoughtforge.knowledge.bond_graph_verdict import (
    SPIKE_EXPECTED,
    SPIKE_EXPECTED_EXCERPT,
)
from thoughtforge.knowledge.models import CognitionScaffold

REPO_ROOT = Path(__file__).resolve().parents[1]
WING = REPO_ROOT / "research_data" / "src" / "wyrdforge"
SCHEMAS = WING / "schemas"


# ── The bond graph layer ─────────────────────────────────────────────────────

class TestBondGraphLanding:
    def test_problems_empty(self) -> None:
        assert bond_graph_problems() == []

    def test_production_names_are_not_spike_names(self) -> None:
        for cls in (BondEdge, Vow, Hurt):
            assert not cls.__name__.startswith("Spike")

    def test_field_sets_match_schemas(self) -> None:
        edge = json.loads(
            (SCHEMAS / "bond_edge.schema.json").read_text(
                encoding="utf-8"))
        expected = {"BondEdge": tuple(edge["properties"])}
        for name, shape in edge["$defs"].items():
            if "properties" in shape:
                expected[name] = tuple(shape["properties"])
        expected["Vow"] = tuple(json.loads(
            (SCHEMAS / "vow.schema.json").read_text(
                encoding="utf-8"))["properties"])
        expected["Hurt"] = tuple(json.loads(
            (SCHEMAS / "hurt.schema.json").read_text(
                encoding="utf-8"))["properties"])
        filed = bond_field_map()
        assert set(filed) == set(expected)
        for shape, fields in expected.items():
            # Set equality, the checker's own judgement —
            # the schema's property order is not the
            # dataclass's declaration order (vow files
            # created_from_record_id after its witnesses).
            assert set(filed[shape]) == set(fields)
            assert len(filed[shape]) == len(fields)

    def test_enums_match_schemas(self) -> None:
        edge = json.loads(
            (SCHEMAS / "bond_edge.schema.json").read_text(
                encoding="utf-8"))
        assert [e.value for e in BondDomain] == \
            edge["$defs"]["BondDomain"]["enum"]
        assert [e.value for e in BondStatus] == \
            edge["$defs"]["BondStatus"]["enum"]
        vow = json.loads(
            (SCHEMAS / "vow.schema.json").read_text(encoding="utf-8"))
        assert [e.value for e in VowState] == \
            vow["$defs"]["VowState"]["enum"]

    def test_weighting_decision_is_equal_with_evidence(self) -> None:
        assert WEIGHTING_DECISION["decision"] == "equal_weights"
        assert WEIGHTING_DECISION["weights"] == {
            "closeness": 0.25, "sacred": 1 / 3, "rupture": 1 / 3}
        assert len(WEIGHTING_DECISION["evidence"]) >= 3
        assert len(WEIGHTING_DECISION["limits"]) >= 1
        spec = (REPO_ROOT / "research_data"
                / "BondGraphSpec.md").read_text(encoding="utf-8")
        assert "weighted_mean(trust, warmth, familiarity, " \
               "vulnerability)" in spec

    def test_parity_matches_tf139_spike_exactly(self) -> None:
        assert PARITY_EXPECTED == SPIKE_EXPECTED
        assert PARITY_EXPECTED_EXCERPT == SPIKE_EXPECTED_EXCERPT
        parity = run_bond_parity()
        assert parity["metrics"] == SPIKE_EXPECTED
        assert tuple(parity["excerpt"]) == SPIKE_EXPECTED_EXCERPT

    def test_parity_run_is_deterministic(self) -> None:
        assert run_bond_parity() == run_bond_parity()

    def test_severity_arithmetic_and_thresholds(self) -> None:
        graph = BondGraph()
        edge = parity_edge()
        graph.add_edge(edge)
        assert edge.status is BondStatus.FORMING
        graph.apply_event(
            edge.bond_id, warmth_delta=0.2, trust_delta=0.2,
            source_record_id="rec:test-1")
        assert edge.status is BondStatus.ACTIVE
        assert edge.evidence.supporting_record_ids == ["rec:test-1"]
        graph.add_hurt(Hurt(
            hurt_id="hurt:test", bond_id=edge.bond_id,
            source_event_id="evt:test", hurt_kind="neglect",
            severity="low"))
        assert edge.scars.repair_debt == SEVERITY_SHIFTS["low"]
        assert edge.vector.safety == pytest.approx(0.9 - 0.05)
        assert edge.status is BondStatus.REPAIRING

    def test_vector_clamps(self) -> None:
        graph = BondGraph()
        edge = parity_edge()
        graph.add_edge(edge)
        graph.apply_event(
            edge.bond_id, warmth_delta=5.0, trust_delta=-5.0)
        assert edge.vector.warmth == 1.0
        assert edge.vector.trust == 0.0

    def test_layer_is_not_a_persisted_record_type(self) -> None:
        from thoughtforge.knowledge.store_map import PERSISTED_RECORDS

        names = {row["name"].lower() for row in PERSISTED_RECORDS}
        assert not any(
            marker in name for name in names
            for marker in ("bond", "vow", "hurt"))

    def test_probes_all_pass(self) -> None:
        probes = run_bond_graph_probes()
        assert len(probes) == len(BOND_GRAPH_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        from thoughtforge.knowledge.bond_graph import (
            edge_problems,
            hurt_problems,
            index_problems,
            schema_field_problems,
            severity_map_problems,
            weighting_problems,
        )

        assert weighting_problems(None)
        assert severity_map_problems(None)
        assert index_problems(0.0, 0.0, 0.0, None)
        assert schema_field_problems(None)
        assert edge_problems(None)
        assert hurt_problems(None)

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / BOND_GRAPH_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_bond_graph_markdown()


# ── The persona compiler merge ───────────────────────────────────────────────

class TestPersonaCompilerLanding:
    def test_problems_empty(self) -> None:
        assert persona_compiler_problems() == []

    def test_main_counts_move_exactly_as_filed(self) -> None:
        run = run_persona_compile()
        assert run["main_counts"] == COMPILE_EXPECTED["main_counts"]
        # Against TF-141's spike filing (anchors 5 / uncertainty 2):
        assert run["main_counts"]["truth_anchor_points"] == 4
        assert run["main_counts"]["uncertainty_points"] == 3

    def test_run_is_deterministic(self) -> None:
        assert run_persona_compile() == run_persona_compile()

    def test_field_sets_match_schemas(self) -> None:
        packet = json.loads(
            (SCHEMAS / "persona_packet.schema.json").read_text(
                encoding="utf-8"))
        canonical = json.loads(
            (SCHEMAS / "canonical_fact_record.schema.json").read_text(
                encoding="utf-8"))
        symbolic = json.loads(
            (SCHEMAS / "symbolic_trace_record.schema.json").read_text(
                encoding="utf-8"))
        expected = {
            "PersonaSourceItem": tuple(
                packet["$defs"]["PersonaSourceItem"]["properties"]),
            "TraitSignal": tuple(
                packet["$defs"]["TraitSignal"]["properties"]),
            "CanonicalFactPayload": tuple(canonical["$defs"]
                ["CanonicalFactPayload"]["properties"]),
            "CanonicalFactContent": tuple(canonical["$defs"]
                ["CanonicalFactContent"]["properties"]),
            "SymbolicTracePayload": tuple(symbolic["$defs"]
                ["SymbolicTracePayload"]["properties"]),
            "SymbolicTraceContent": tuple(symbolic["$defs"]
                ["SymbolicTraceContent"]["properties"]),
            "TruthMeta": tuple(
                canonical["$defs"]["TruthMeta"]["properties"]),
        }
        assert production_field_map() == expected

    def test_enums_match_schemas(self) -> None:
        packet = json.loads(
            (SCHEMAS / "persona_packet.schema.json").read_text(
                encoding="utf-8"))
        assert [m.value for m in PersonaMode] == \
            packet["$defs"]["PersonaMode"]["enum"]
        canonical = json.loads(
            (SCHEMAS / "canonical_fact_record.schema.json").read_text(
                encoding="utf-8"))
        assert [a.value for a in ApprovalState] == \
            canonical["$defs"]["ApprovalState"]["enum"]

    def test_trait_weighting_sums_and_caps_at_one(self) -> None:
        run = run_persona_compile()
        assert run["main"]["active_traits"] == COMPILE_EXPECTED["traits"]
        steady = run["main"]["active_traits"][0]
        assert steady["weight"] == 1.0  # 0.6 + 0.5 = 1.1, capped
        assert steady["supporting_record_ids"] == [
            "rec:synthetic-wrenna-001", "rec:synthetic-wrenna-005"]

    def test_approval_first_pending_fact_is_uncertainty(self) -> None:
        run = run_persona_compile()
        assert "world:lantern-hollow.bell_count=two" not in \
            run["main"]["truth_anchor_points"]
        assert run["main"]["uncertainty_points"] == \
            COMPILE_EXPECTED["uncertainty_points"]
        assert run["main"]["truth_anchor_points"] == \
            COMPILE_EXPECTED["truth_anchor_points"]

    def test_approval_first_unapproved_symbolic_too(self) -> None:
        records = fixture_records()
        pending_symbolic = SymbolicTraceRecord(
            record_id="rec:synthetic-wrenna-015",
            record_type="symbolic_trace",
            content=records[9].content.__class__(
                title="bell", summary="an unapproved bell trace",
                structured_payload=records[9].content
                .structured_payload),
            truth=records[9].truth.__class__(
                confidence=0.5,
                approval_state=ApprovalState.PENDING))
        compiled = PersonaCompiler().compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper",
            mode=PersonaMode.COMPANION,
            records=[pending_symbolic])
        assert compiled.symbolic_context == []
        assert compiled.uncertainty_points == [
            "an unapproved bell trace"]

    def test_unapproved_canonical_feeds_no_trait(self) -> None:
        record = CanonicalFactRecord(
            record_id="rec:synthetic-pending-trait",
            record_type="canonical_fact",
            content=fixture_records()[0].content.__class__(
                title="t", summary="a pending steadiness",
                structured_payload=fixture_records()[0].content
                .structured_payload),
            truth=fixture_records()[0].truth.__class__(
                confidence=0.9,
                approval_state=ApprovalState.PENDING))
        compiled = PersonaCompiler().compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper",
            mode=PersonaMode.COMPANION, records=[record])
        assert compiled.active_traits == []
        assert compiled.identity_core == []
        assert compiled.uncertainty_points == ["a pending steadiness"]

    def test_relationship_excerpt_is_bond_only(self) -> None:
        run = run_persona_compile()
        main = run["main"]
        assert main["relationship_excerpt"] == \
            COMPILE_EXPECTED["relationship_excerpt"]
        assert main["relationship_excerpt"] != main["response_guidance"]
        mode_line = "never claim vows or memories that are not supported"
        assert mode_line in main["response_guidance"]
        assert mode_line not in main["relationship_excerpt"]

    def test_relationship_compiles_from_production_edge(self) -> None:
        edge = fixture_bond_edge()
        assert isinstance(edge, BondEdge)
        assert not type(edge).__name__.startswith("Spike")
        assert edge.closeness_index() == pytest.approx(0.4375)

    def test_token_budget_hint_is_dropped(self) -> None:
        assert TOKEN_BUDGET_DECISION["decision"] == "dropped"
        assert "token_budget_hint" not in \
            CompiledPersona.__dataclass_fields__
        run = run_persona_compile()
        assert run["main"]["has_token_budget_hint"] is False

    def test_avoid_is_declared_only(self) -> None:
        assert AVOID_DECISION["decision"] == "declared"
        run = run_persona_compile()
        assert run["merged"]["avoid"] == list(FIXTURE_EXPECTATION.avoid)
        assert run["merged"]["avoid_matches_base"] is True

    def test_defect_differences_are_three_and_filed(self) -> None:
        assert [row["id"] for row in DEFECT_DIFFERENCES] == [
            "approval_checked_first",
            "relationship_excerpt_bond_only",
            "token_budget_hint_dropped"]
        assert all(row["spike"] and row["production"]
                   for row in DEFECT_DIFFERENCES)

    def test_caps_run_binds_every_cap(self) -> None:
        run = run_persona_compile()
        assert run["caps"] == COMPILE_EXPECTED["caps"]
        assert run["caps"]["identity_core"] == CAPS["identity_core"]
        assert run["caps"]["source_items"] == CAPS["source_items"]
        assert len(caps_records()) == 57

    def test_source_items_carry_every_record_with_confidence(self) -> None:
        run = run_persona_compile()
        items = run["main"]["source_items"]
        assert len(items) == 14
        assert items[8] == {
            "record_id": "rec:synthetic-wrenna-009",
            "item_type": "canonical_fact", "confidence": 0.3}

    def test_modes_adapt_contract_and_guidance(self) -> None:
        run = run_persona_compile()
        for mode in ("companion", "coding_guide", "world_seer"):
            assert len(run["modes"][mode]["tone_contract"]) == 4
            assert len(run["modes"][mode]["response_guidance"]) == 1


# ── The merge with TF-030's runtime ──────────────────────────────────────────

class TestMergedDelivery:
    def _compiled(self) -> CompiledPersona:
        return PersonaCompiler().compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper",
            mode=PersonaMode.COMPANION,
            records=fixture_records(),
            bond_edge=fixture_bond_edge(),
            base_expectation=FIXTURE_EXPECTATION)

    def test_expectation_extends_persona_expectation(self) -> None:
        expectation = self._compiled().to_expectation()
        assert isinstance(expectation, PersonaExpectation)
        assert isinstance(expectation, CompiledPersonaExpectation)
        assert expectation.base_record_id == FIXTURE_EXPECTATION.record_id

    def test_inject_persona_delivers_base_and_compiled(self) -> None:
        expectation = self._compiled().to_expectation()
        steering = expectation.steering_lines()
        for line in FIXTURE_EXPECTATION.steering_lines():
            assert line in steering
        assert any(line.startswith("Persona identity:")
                   for line in steering)
        assert any(line.startswith("Persona relationship:")
                   for line in steering)
        scaffold = CognitionScaffold(
            goal="turn goal", fact_block="Knowledge:\nx")
        assert inject_persona(scaffold, expectation) is True
        assert scaffold.fact_block.startswith(steering[0])
        assert "Persona truth anchors:" in scaffold.fact_block
        # The duplicate guard still holds on a second call.
        assert inject_persona(scaffold, expectation) is False

    def test_avoid_checking_judges_the_merged_shape(self) -> None:
        expectation = self._compiled().to_expectation()
        bad = check_persona_response(
            expectation,
            "There may be invented crossings after the second "
            "tide bell if the night looks calm enough.")
        assert bad.violations == [
            "invented crossings after the second tide bell"]
        from thoughtforge.cognition.persona import PERSONA_FIXTURE_TURNS

        report = score_persona_turns(
            expectation, [text for text, _ in PERSONA_FIXTURE_TURNS])
        assert (report.total_turns, report.consistent_turns,
                report.score, report.verdict) == (5, 2, 0.4, "REVIEW")

    def test_compile_without_base_files_empty_avoid(self) -> None:
        compiled = PersonaCompiler().compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper",
            mode=PersonaMode.COMPANION, records=fixture_records())
        expectation = compiled.to_expectation()
        assert expectation.avoid == []
        assert expectation.traits == [
            "steady", "plain-spoken", "honesty"]
        # An empty batch on the merged shape is still the
        # vacuous case TF-030 refuses: 0.0 / REVIEW.
        report = score_persona_turns(expectation, [])
        assert report.verdict == "REVIEW" and report.score == 0.0

    def test_authored_path_stands_unmodified(self) -> None:
        assert FIXTURE_EXPECTATION.steering_lines() == [
            "Persona traits: steady; observant; plain-spoken",
            "Persona speech style: short, concrete sentences; "
            "names the river level before giving advice; no filler",
            "Persona behavior: answer what was asked; cite the "
            "synthetic reference when using it; say plainly when "
            "the fixture store is silent",
        ]
        data = FIXTURE_EXPECTATION.to_dict()
        assert set(data) == {
            "record_id", "traits", "speech_style",
            "behavior_rules", "avoid", "weights"}
        assert dataclasses.fields(PersonaExpectation)


# ── Cross-cutting ────────────────────────────────────────────────────────────

class TestCrossCutting:
    def test_both_probe_sets_pass(self) -> None:
        for probes in (run_bond_graph_probes(),
                       run_persona_compiler_probes()):
            assert len(probes) == 7
            assert all(p["passed"] for p in probes)

    def test_persona_probes_filed(self) -> None:
        assert len(PERSONA_COMPILER_PROBES) == 7
        assert len(BOND_GRAPH_PROBES) == 7

    def test_validators_are_empty_safe(self) -> None:
        from thoughtforge.cognition.persona_compiler import (
            approval_problems,
            caps_problems,
            measured_problems,
            mode_problems,
            relationship_problems,
            schema_field_problems as persona_schema_problems,
            trait_problems,
        )

        assert caps_problems(None)
        assert trait_problems(None, None)
        assert approval_problems([], [], [], [], None)
        assert relationship_problems([], [], None)
        assert mode_problems(None)
        assert measured_problems(None, None)
        assert persona_schema_problems(None)

    def test_wing_models_package_still_does_not_exist(self) -> None:
        # TF-143 lands production code in the SHIPPING tree;
        # resurrecting wyrdforge.models is not this slice.
        assert not (WING / "models").exists()

    def test_fixtures_are_synthetic_only(self) -> None:
        for record in fixture_records():
            assert "synthetic" in record.record_id
        assert parity_edge().entity_a == "user:emberwick-keeper"

    def test_persona_doc_matches_render(self) -> None:
        doc = REPO_ROOT / PERSONA_COMPILER_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_persona_compiler_markdown()
