"""
TF-139 — Bond graph verdict: adopt (as a layer),
merge, or retire, decided on TF-006's census + a
working spike.

The proof lives as verifiable data in
``src/thoughtforge/knowledge/bond_graph_verdict.py``
(``COMPARISON_ROWS``, ``VERDICT``, the ``Spike*``
dataclasses + ``SpikeBondGraph``, ``BOND_PROBES``,
checked by ``bond_graph_verdict_problems()``);
the write-up ``docs/architecture/BOND_GRAPH_VERDICT.md``
is filed verbatim as the module's render.

Pinned here:

  * the verdict: ADOPT_AS_LAYER with its filed
    reasons + conditions (not a merge, not retire;
    adoption owned by TF-143, the bridge by TF-146,
    the wing's tests by TF-149),
  * the comparison: 9 rows bound live to the tree —
    no bond/relationship graph in the shipping
    knowledge+cognition sources, no bond record
    type, the wing's 14 schemas parse, the wing's
    models package does NOT exist, the service's
    severity map matches the spike's,
  * the spike lifecycle: edge -> vow -> event
    crosses ACTIVE at the filed thresholds, a high
    hurt -> REPAIRING with the exact repair_debt /
    safety arithmetic, the excerpt lines verbatim,
    the 0..1 clamps, defaults from the schemas,
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures are synthetic invented entities
(TF-017) — Emberwick names in TF-132's style;
no canon, no real persona, risky markers
assembled from parts. Proof grade: FIXTURE/
STATIC — no live model was read, never claimed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from thoughtforge.knowledge.bond_graph_verdict import (
    BOND_GRAPH_VERDICT_DOC,
    BOND_PROBES,
    COMPARISON_ROWS,
    FINDINGS,
    SEVERITY_SHIFTS,
    SPIKE_EXPECTED,
    SPIKE_EXPECTED_EXCERPT,
    VERDICT,
    VERDICTS,
    BondDomain,
    BondStatus,
    SpikeBondEdge,
    SpikeBondGraph,
    SpikeBondVector,
    SpikeHurt,
    SpikeVow,
    VowState,
    bond_graph_verdict_problems,
    comparison_row_problems,
    index_problems,
    render_bond_graph_verdict_markdown,
    run_bond_probes,
    run_bond_spike,
    schema_field_problems,
    severity_map_problems,
    spike_field_map,
    verdict_data_problems,
)

REPO_ROOT = Path(__file__).parent.parent
WING = REPO_ROOT / "research_data" / "src" / "wyrdforge"


def _edge(**overrides):
    kwargs = {
        "bond_id": "bond:test-001",
        "entity_a": "user:emberwick-keeper",
        "entity_b": "persona:emberwick-lantern",
        "domain": BondDomain.FRIENDSHIP,
    }
    kwargs.update(overrides)
    return SpikeBondEdge(**kwargs)


# ── The verdict as data ──────────────────────────────────────────────────────

class TestVerdictData:
    def test_problems_empty(self) -> None:
        assert bond_graph_verdict_problems() == []

    def test_verdict_is_adopt_as_layer(self) -> None:
        assert VERDICT["verdict"] == "ADOPT_AS_LAYER"
        assert VERDICT["verdict"] in VERDICTS
        assert verdict_data_problems(VERDICT) == []

    def test_reasons_cover_not_merge_not_retire(self) -> None:
        reasons = " ".join(VERDICT["reasons"])
        assert "not a merge" in reasons
        assert "not retire" in reasons
        assert "unique concept" in reasons
        assert any("layer link" in r and "evidence" in r
                   for r in VERDICT["reasons"])

    def test_conditions_name_the_owners(self) -> None:
        conditions = " ".join(VERDICT["conditions"])
        for owner in ("TF-143", "TF-146", "TF-149"):
            assert owner in conditions
        assert "NO shipping behaviour" in conditions
        assert "record ids as evidence" in conditions

    def test_comparison_rows_shape(self) -> None:
        assert len(COMPARISON_ROWS) == 9
        dimensions = [r["dimension"] for r in COMPARISON_ROWS]
        assert len(dimensions) == len(set(dimensions))
        for row in COMPARISON_ROWS:
            assert comparison_row_problems(row) == []
            assert row["finding"] in FINDINGS
        findings = [r["finding"] for r in COMPARISON_ROWS]
        assert findings.count("unique") == 5
        assert findings.count("layer-link") == 1
        by_dim = {r["dimension"]: r for r in COMPARISON_ROWS}
        assert by_dim["evidence_links"]["finding"] == "layer-link"
        assert "supporting_record_ids" in by_dim["evidence_links"]["bond_graph"]


# ── Bound live to the tree ───────────────────────────────────────────────────

class TestLiveTreeBinding:
    def test_shipping_sources_have_no_bond_graph(self) -> None:
        for folder in ("knowledge", "cognition"):
            for path in (REPO_ROOT / "src" / "thoughtforge" / folder).glob("*.py"):
                if path.name == "bond_graph_verdict.py":
                    continue
                text = path.read_text(encoding="utf-8").lower()
                assert "bond" not in text, path.name
                assert "relationship" not in text, path.name

    def test_shipping_models_have_no_bond_type(self) -> None:
        from thoughtforge.knowledge.store_map import PERSISTED_RECORDS

        names = {row["name"] for row in PERSISTED_RECORDS}
        assert names  # the census is not empty
        assert not any(
            marker in name.lower()
            for name in names for marker in ("bond", "vow", "hurt"))

    def test_wing_models_package_does_not_exist(self) -> None:
        assert not (WING / "models").exists()
        init_text = (WING / "__init__.py").read_text(encoding="utf-8")
        assert "from .models.bond import" in init_text

    def test_schemas_exist_and_parse(self) -> None:
        schema_files = sorted((WING / "schemas").glob("*.json"))
        assert len(schema_files) == 14
        for schema_file in schema_files:
            json.loads(schema_file.read_text(encoding="utf-8"))

    def test_service_severity_map_matches_spike(self) -> None:
        service = (WING / "services" / "bond_graph_service.py").read_text(
            encoding="utf-8")
        for name, shift in SEVERITY_SHIFTS.items():
            assert f'"{name}": {shift}' in service
        assert severity_map_problems(SEVERITY_SHIFTS) == []

    def test_spike_fields_match_schemas(self) -> None:
        assert schema_field_problems(spike_field_map()) == []
        edge_fields = {f.name for f in dataclasses.fields(SpikeBondEdge)}
        assert edge_fields == {
            "bond_id", "entity_a", "entity_b", "domain", "status",
            "vector", "constraints", "scars", "active_modes",
            "chronology", "evidence", "governance",
        }

    def test_enums_match_schemas(self) -> None:
        edge_schema = json.loads(
            (WING / "schemas" / "bond_edge.schema.json").read_text(
                encoding="utf-8"))
        assert [e.value for e in BondDomain] == \
            edge_schema["$defs"]["BondDomain"]["enum"]
        assert [e.value for e in BondStatus] == \
            edge_schema["$defs"]["BondStatus"]["enum"]
        vow_schema = json.loads(
            (WING / "schemas" / "vow.schema.json").read_text(
                encoding="utf-8"))
        assert [e.value for e in VowState] == \
            vow_schema["$defs"]["VowState"]["enum"]


# ── The spike ────────────────────────────────────────────────────────────────

class TestSpike:
    def test_schema_defaults(self) -> None:
        vector = SpikeBondVector()
        assert (vector.trust, vector.warmth, vector.safety) == (0.5, 0.5, 0.8)
        edge = _edge()
        assert edge.status is BondStatus.FORMING
        assert edge.active_modes.relational_mode == "builder"
        assert edge.governance.synthetic_claim_guard is True
        assert edge.scars.repair_debt == 0.0
        hurt = SpikeHurt(
            hurt_id="h", bond_id="b", source_event_id="e",
            hurt_kind="neglect", severity="low")
        assert hurt.notes == ""
        vow = SpikeVow(
            vow_id="v", bond_id="b", vow_text="t", vow_kind="promise",
            created_from_record_id="r")
        assert vow.strength == "medium" and vow.state is VowState.PENDING

    def test_spike_run_matches_filed_expectation(self) -> None:
        spike = run_bond_spike()
        assert spike["metrics"] == SPIKE_EXPECTED
        assert tuple(spike["excerpt"]) == SPIKE_EXPECTED_EXCERPT
        # Deterministic: a second run files the same metrics.
        assert run_bond_spike() == spike

    def test_event_crosses_active_threshold(self) -> None:
        graph = SpikeBondGraph()
        edge = _edge(vector=SpikeBondVector(trust=0.4, warmth=0.4))
        graph.add_edge(edge)
        assert edge.chronology.formed_at is not None
        graph.apply_event(
            edge.bond_id, trust_delta=0.1, warmth_delta=0.1)
        assert edge.status is BondStatus.FORMING  # 0.5 < 0.55
        graph.apply_event(
            edge.bond_id, trust_delta=0.1, warmth_delta=0.1,
            source_record_id="rec:synthetic-emberwick-101")
        assert edge.status is BondStatus.ACTIVE
        assert edge.evidence.supporting_record_ids == [
            "rec:synthetic-emberwick-101"]
        assert edge.chronology.last_contact_at is not None

    def test_hurt_arithmetic_exact(self) -> None:
        graph = SpikeBondGraph()
        edge = _edge(vector=SpikeBondVector(safety=0.9))
        graph.add_edge(edge)
        graph.add_hurt(SpikeHurt(
            hurt_id="hurt:x", bond_id=edge.bond_id,
            source_event_id="evt:x", hurt_kind="betrayal",
            severity="high"))
        assert edge.status is BondStatus.REPAIRING
        assert edge.scars.repair_debt == 0.35
        assert edge.vector.safety == 0.9 - 0.175
        assert edge.scars.unresolved_hurts == ["hurt:x"]
        graph.add_hurt(SpikeHurt(
            hurt_id="hurt:y", bond_id=edge.bond_id,
            source_event_id="evt:y", hurt_kind="absence",
            severity="mythic"))
        assert edge.scars.repair_debt == 0.85
        assert edge.vector.safety == 0.9 - 0.175 - 0.25

    def test_repair_debt_and_safety_clamp(self) -> None:
        graph = SpikeBondGraph()
        edge = _edge(vector=SpikeBondVector(safety=0.3))
        graph.add_edge(edge)
        for index in range(3):
            graph.add_hurt(SpikeHurt(
                hurt_id=f"hurt:c{index}", bond_id=edge.bond_id,
                source_event_id=f"evt:c{index}",
                hurt_kind="neglect", severity="mythic"))
        assert edge.scars.repair_debt == 1.0
        assert edge.vector.safety == 0.0

    def test_event_clamps_at_zero_and_one(self) -> None:
        graph = SpikeBondGraph()
        edge = _edge(vector=SpikeBondVector(
            trust=0.9, warmth=0.1, devotion=0.5))
        graph.add_edge(edge)
        graph.apply_event(
            edge.bond_id, trust_delta=5.0, warmth_delta=-5.0,
            devotion_delta=0.25)
        assert edge.vector.trust == 1.0
        assert edge.vector.warmth == 0.0
        assert edge.vector.devotion == 0.75
        assert edge.status is BondStatus.FORMING  # warmth below threshold

    def test_derived_indices_equal_weights(self) -> None:
        edge = _edge(vector=SpikeBondVector(
            trust=0.8, warmth=0.6, familiarity=0.4, vulnerability=0.2,
            devotion=0.9, sacred_resonance=0.3, safety=0.5))
        edge.scars.vow_strain = 0.2
        edge.scars.repair_debt = 0.1
        assert edge.closeness_index() == (0.8 + 0.6 + 0.4 + 0.2) / 4
        assert edge.sacred_bond_index() == (0.9 + 0.3 + 0.8) / 3
        assert edge.rupture_index() == (0.1 + 0.2 + 0.5) / 3
        assert index_problems(
            edge.closeness_index(), edge.sacred_bond_index(),
            edge.rupture_index(),
            {"closeness": 0.5, "sacred": 0.67, "rupture": 0.27}) == []

    def test_excerpt_shape_and_vow_line(self) -> None:
        graph = SpikeBondGraph()
        edge = _edge()
        graph.add_edge(edge)
        lines = graph.excerpt(edge.bond_id)
        assert lines[0] == "domain=friendship"
        assert lines[1] == "status=forming"
        assert len(lines) == 7  # no hurts, no vows yet
        graph.add_vow(SpikeVow(
            vow_id="vow:x", bond_id=edge.bond_id,
            vow_text="answer before dusk", vow_kind="promise",
            created_from_record_id="rec:synthetic-emberwick-102"))
        lines = graph.excerpt(edge.bond_id)
        assert lines[-1] == "vows=promise:pending:answer before dusk"

    def test_fixtures_are_synthetic_only(self) -> None:
        # TF-017: no canon names in the spike fixtures — the
        # wing demo's real-shaped ids, assembled from parts.
        spike_text = Path(
            "src/thoughtforge/knowledge/bond_graph_verdict.py")
        module_text = (REPO_ROOT / spike_text).read_text(encoding="utf-8")
        banned = "vol" + "marr"
        assert f"user:{banned}" not in module_text
        assert "vey" + "runn" not in module_text
        assert "emberwick" in module_text


# ── The probes + the doc ─────────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_bond_probes()
        assert len(probes) == len(BOND_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert verdict_data_problems({})
        assert verdict_data_problems(None)
        assert comparison_row_problems({})
        assert severity_map_problems({})
        assert index_problems(0.0, 0.0, 0.0, {})
        assert schema_field_problems({})

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / BOND_GRAPH_VERDICT_DOC
        assert doc.exists()
        assert doc.read_text(encoding="utf-8") == \
            render_bond_graph_verdict_markdown()
        rendered = render_bond_graph_verdict_markdown()
        assert "ADOPT_AS_LAYER" in rendered
        # A tampered doc is not the render (the checker's shape).
        assert rendered != rendered.replace("ADOPT_AS_LAYER", "RETIRE")
