"""
TF-141 — Persona compiler verdict: its persona
compilation vs the shipping config/persona path
(TF-009/TF-030) — adopt the compiler's shape where
it proves better.

The proof lives as verifiable data in
``src/thoughtforge/cognition/persona_compiler_verdict.py``
(``COMPARISON_ROWS``, ``VERDICT``, the ``Spike*``
dataclasses + ``SpikePersonaCompiler``,
``run_persona_spike()``, ``PERSONA_PROBES``, checked
by ``persona_compiler_verdict_problems()``); the
write-up ``docs/architecture/PERSONA_COMPILER_VERDICT.md``
is filed verbatim as the module's render. (The
verdict lives in cognition/ because its subject —
TF-030's persona path — does; TF-139/TF-140's live
in knowledge/ because theirs do.)

Pinned here:

  * the verdict: MERGE (adopt-in-part) — the
    compiler wins seven measured facets, the
    shipping path wins delivery + checking;
    TF-143 owns implementing the adopted facets
    under the filed conditions (including NOT
    importing the compiler's two measured
    defects), TF-147 consumes the merged output,
  * the comparison: 12 rows bound live to the
    tree — the wing's models package does NOT
    exist, its 14 schemas parse, the compiler's
    caps / domain / trait-key markers match the
    spike's, TF-030's hooks stand as filed,
  * the spike measurement: the same synthetic
    ground truth through both paths — the main
    packet verbatim (trait arithmetic 0.6 + 0.5
    filed at the 1.0 cap, the pending canonical
    fact filed as a truth anchor — the elif blind
    spot, relationship_excerpt duplicating the
    guidance), the three modes, the 57-record
    caps run, and the shipping path's steering /
    injection / checking numbers,
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures are TF-017 / TF-030's synthetic
Wrenna / Lantern Hollow world; no canon, no real
persona, risky markers assembled from parts.
Proof grade: FIXTURE/STATIC — no live model was
read, never claimed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from thoughtforge.cognition.persona import (
    FIXTURE_EXPECTATION,
    PersonaExpectation,
    check_persona_response,
)
from thoughtforge.cognition.persona_compiler_verdict import (
    COMPARISON_ROWS,
    FINDINGS,
    PERSONA_COMPILER_VERDICT_DOC,
    PERSONA_PROBES,
    SPIKE_EXPECTED,
    VERDICT,
    VERDICTS,
    ApprovalState,
    PersonaMode,
    SpikePersonaCompiler,
    SpikePersonaSourceItem,
    SpikeTraitSignal,
    caps_records,
    comparison_row_problems,
    fixture_bond_edge,
    fixture_records,
    measured_problems,
    persona_compiler_verdict_problems,
    render_persona_compiler_verdict_markdown,
    run_persona_probes,
    run_persona_spike,
    schema_field_problems,
    spike_field_map,
    verdict_data_problems,
)

REPO_ROOT = Path(__file__).parent.parent
WING = REPO_ROOT / "research_data" / "src" / "wyrdforge"

_SPIKE: dict | None = None


def spike() -> dict:
    global _SPIKE
    if _SPIKE is None:
        _SPIKE = run_persona_spike()
    return _SPIKE


# ── The verdict as data ──────────────────────────────────────────────────────

class TestVerdictData:
    def test_problems_empty(self) -> None:
        assert persona_compiler_verdict_problems() == []

    def test_verdict_is_merge(self) -> None:
        assert VERDICT["verdict"] == "MERGE"
        assert VERDICT["verdict"] in VERDICTS
        assert verdict_data_problems(VERDICT) == []

    def test_reasons_cover_both_wins_and_not_wholesale(self) -> None:
        reasons = " ".join(VERDICT["reasons"])
        assert "seven measured facets" in reasons
        assert "DELIVERED" in reasons
        assert "not adopt wholesale" in reasons
        assert "not retire" in reasons
        assert "1.0 cap" in reasons

    def test_conditions_name_the_owners_and_the_defects(self) -> None:
        conditions = " ".join(VERDICT["conditions"])
        for owner in ("TF-143", "TF-147", "TF-149"):
            assert owner in conditions
        assert "NOT import the compiler's defects" in conditions
        assert "NO shipping behaviour" in conditions
        assert "inject_persona" in conditions

    def test_comparison_rows_shape(self) -> None:
        assert len(COMPARISON_ROWS) == 12
        dimensions = [r["dimension"] for r in COMPARISON_ROWS]
        assert len(dimensions) == len(set(dimensions))
        for row in COMPARISON_ROWS:
            assert comparison_row_problems(row) == []
            assert row["finding"] in FINDINGS
        findings = [r["finding"] for r in COMPARISON_ROWS]
        assert findings.count("compiler-wins") == 7
        assert findings.count("shipping-wins") == 2
        assert findings.count("different") == 3
        by_dim = {r["dimension"]: r for r in COMPARISON_ROWS}
        assert by_dim["prompt_steering_delivered"]["finding"] == \
            "shipping-wins"
        assert by_dim["avoid_and_consistency_checking"]["finding"] == \
            "shipping-wins"
        assert by_dim["trait_weighting"]["finding"] == "compiler-wins"


# ── Bound live to the tree ───────────────────────────────────────────────────

class TestLiveTreeBinding:
    def test_wing_models_package_does_not_exist(self) -> None:
        assert not (WING / "models").exists()
        compiler = (WING / "services" / "persona_compiler.py").read_text(
            encoding="utf-8")
        assert "from wyrdforge.models.persona import" in compiler
        assert "from wyrdforge.models.memory import" in compiler

    def test_schemas_exist_and_parse(self) -> None:
        schema_files = sorted((WING / "schemas").glob("*.json"))
        assert len(schema_files) == 14
        for schema_file in schema_files:
            json.loads(schema_file.read_text(encoding="utf-8"))

    def test_compiler_markers_match_spike(self) -> None:
        compiler = (WING / "services" / "persona_compiler.py").read_text(
            encoding="utf-8")
        for marker in ('{"identity", "style", "mission"}',
                       '{"temperament", "tone", "value", "mood"}',
                       "min(1.0, weight)", "identity_core[:10]",
                       "truth_anchor_points[:16]",
                       "source_items[:20]", "response_guidance[:6]",
                       "token_budget_hint: int = 800"):
            assert marker in compiler, marker

    def test_spike_fields_match_schemas(self) -> None:
        assert schema_field_problems(spike_field_map()) == []
        packet_fields = {
            f.name for f in dataclasses.fields(
                __import__(
                    "thoughtforge.cognition.persona_compiler_verdict",
                    fromlist=["SpikePersonaPacket"]).SpikePersonaPacket)}
        assert packet_fields == {
            "persona_id", "user_id", "mode", "tone_contract",
            "identity_core", "active_traits", "relationship_excerpt",
            "truth_anchor_points", "uncertainty_points",
            "symbolic_context", "response_guidance", "source_items",
            "token_budget_hint",
        }
        assert {f.name for f in dataclasses.fields(SpikeTraitSignal)} == {
            "trait_name", "weight", "supporting_record_ids"}
        assert {f.name for f in dataclasses.fields(
            SpikePersonaSourceItem)} == {
            "record_id", "item_type", "text", "confidence"}

    def test_enums_match_schemas(self) -> None:
        packet_schema = json.loads(
            (WING / "schemas" / "persona_packet.schema.json")
            .read_text(encoding="utf-8"))
        assert [m.value for m in PersonaMode] == \
            packet_schema["$defs"]["PersonaMode"]["enum"]
        canonical_schema = json.loads(
            (WING / "schemas" / "canonical_fact_record.schema.json")
            .read_text(encoding="utf-8"))
        assert [a.value for a in ApprovalState] == \
            canonical_schema["$defs"]["ApprovalState"]["enum"]
        symbolic_schema = json.loads(
            (WING / "schemas" / "symbolic_trace_record.schema.json")
            .read_text(encoding="utf-8"))
        assert canonical_schema["$defs"]["TruthMeta"] == \
            symbolic_schema["$defs"]["TruthMeta"]

    def test_wing_persona_test_intent_stands(self) -> None:
        wing_test = (
            REPO_ROOT / "research_data" / "tests"
            / "test_memory_and_persona.py").read_text(encoding="utf-8")
        assert "test_persona_compiler_builds_packet" in wing_test
        assert "packet.identity_core" in wing_test
        assert "packet.response_guidance" in wing_test

    def test_shipping_hooks_stand_as_filed(self) -> None:
        persona_text = (
            REPO_ROOT / "src" / "thoughtforge" / "cognition"
            / "persona.py").read_text(encoding="utf-8")
        assert "MAX_STEERING_ENTRIES = 6" in persona_text
        assert "TF-141" in persona_text  # TF-030's filed deferral
        assert (REPO_ROOT / "docs" / "audit"
                / "TF-009_CONFIG_CENSUS.md").exists()
        assert FIXTURE_EXPECTATION.record_id == "syn_wrenna_001"


# ── The compiler spike ───────────────────────────────────────────────────────

class TestCompilerSpike:
    def test_main_run_matches_filed_expectation(self) -> None:
        result = spike()
        assert result["compiler_main_counts"] == \
            SPIKE_EXPECTED["compiler_main_counts"]
        assert result["compiler_main"]["active_traits"] == \
            SPIKE_EXPECTED["compiler_traits"]
        assert result["compiler_main"]["identity_core"] == \
            SPIKE_EXPECTED["compiler_identity_core"]
        assert result["compiler_main"]["truth_anchor_points"] == \
            SPIKE_EXPECTED["compiler_truth_anchors"]
        assert measured_problems(
            result["compiler_main_counts"],
            SPIKE_EXPECTED["compiler_main_counts"]) == []

    def test_run_is_deterministic(self) -> None:
        assert run_persona_spike() == spike()

    def test_trait_weighting_sums_and_caps_at_one(self) -> None:
        traits = {
            t["trait_name"]: t
            for t in spike()["compiler_main"]["active_traits"]}
        # steady: 0.6 (temperament) + 0.5 (mood) = 1.1 -> 1.0.
        assert traits["steady"]["weight"] == 1.0
        assert traits["steady"]["supporting_record_ids"] == [
            "rec:synthetic-wrenna-001", "rec:synthetic-wrenna-005"]
        assert traits["plain-spoken"]["weight"] == 0.8
        assert traits["honesty"]["weight"] == 0.6
        names = [t["trait_name"]
                 for t in spike()["compiler_main"]["active_traits"]]
        assert names == ["steady", "plain-spoken", "honesty"]

    def test_pending_canonical_fact_is_an_anchor_not_uncertainty(
            self) -> None:
        main = spike()["compiler_main"]
        # The elif blind spot, measured: bell_count (pending)
        # files as truth, never as an uncertainty point.
        assert "world:lantern-hollow.bell_count=two" in \
            main["truth_anchor_points"]
        assert main["uncertainty_points"] == \
            SPIKE_EXPECTED["compiler_uncertainty"]
        assert not any(
            "bell_count" in p for p in main["uncertainty_points"])

    def test_symbolic_lines_verbatim(self) -> None:
        assert spike()["compiler_main"]["symbolic_context"] == \
            SPIKE_EXPECTED["compiler_symbolic"]

    def test_relationship_excerpt_duplicates_guidance(self) -> None:
        main = spike()["compiler_main"]
        assert main["response_guidance"] == \
            SPIKE_EXPECTED["compiler_guidance"]
        assert main["relationship_excerpt"] == \
            main["response_guidance"]  # the filed defect
        assert "closeness_index=0.44" in main["response_guidance"]

    def test_source_items_carry_every_record_with_confidence(
            self) -> None:
        items = spike()["compiler_main"]["source_items"]
        assert len(items) == 14
        assert items[0] == {
            "record_id": "rec:synthetic-wrenna-001",
            "item_type": "canonical_fact", "confidence": 0.6}
        assert items[8]["confidence"] == 0.3  # the pending fact
        assert spike()["compiler_main"]["token_budget_hint"] == 800

    def test_modes_adapt_contract_and_guidance(self) -> None:
        modes = spike()["compiler_modes"]
        assert set(modes) == {"companion", "coding_guide", "world_seer"}
        for filing in modes.values():
            assert len(filing["tone_contract"]) == 4
            assert filing["tone_contract"][:3] == \
                modes["companion"]["tone_contract"][:3]
            assert len(filing["response_guidance"]) == 1
        assert "terse correctness" in \
            modes["coding_guide"]["tone_contract"][3]
        assert "mythic atmosphere" in \
            modes["world_seer"]["tone_contract"][3]
        assert "warmth" in modes["companion"]["tone_contract"][3]
        assert modes["coding_guide"]["response_guidance"] == [
            "surface missing evidence before offering "
            "code-level certainty"]

    def test_caps_run_binds_every_cap(self) -> None:
        assert spike()["compiler_caps"] == SPIKE_EXPECTED["compiler_caps"]
        assert len(caps_records()) == 57

    def test_schema_defaults(self) -> None:
        item = SpikePersonaSourceItem(
            record_id="r", item_type="t", text="x")
        assert item.confidence == 0.5
        signal = SpikeTraitSignal(trait_name="steady")
        assert signal.weight == 0.5
        assert signal.supporting_record_ids == []
        packet = SpikePersonaCompiler().compile(
            persona_id="p", user_id="u", mode=PersonaMode.RITUAL,
            records=[])
        assert packet.token_budget_hint == 800
        assert len(packet.tone_contract) == 3  # ritual: base only
        assert packet.response_guidance == []

    def test_bond_edge_is_tf139s_spike_shape(self) -> None:
        edge = fixture_bond_edge()
        assert round(edge.closeness_index(), 4) == 0.4375
        assert edge.active_modes.relational_mode == "builder"


# ── The shipping path, measured ──────────────────────────────────────────────

class TestShippingPath:
    def test_shipping_run_matches_filed_expectation(self) -> None:
        assert spike()["shipping"] == SPIKE_EXPECTED["shipping"]
        assert measured_problems(
            spike()["shipping"], SPIKE_EXPECTED["shipping"]) == []

    def test_steering_carries_no_avoid_and_no_weights(self) -> None:
        ship = spike()["shipping"]
        assert ship["avoid_in_steering"] == 0
        assert ship["weights_carried"] == 5
        assert ship["weights_in_steering"] == 0
        assert ship["steering_entries"] == 9

    def test_avoid_checking_is_the_shippings_win(self) -> None:
        ship = spike()["shipping"]
        assert (ship["turns_checked"], ship["turns_consistent"],
                ship["violations"]) == (5, 2, 3)
        assert ship["score"] == 0.4 and ship["verdict"] == "REVIEW"
        check = check_persona_response(
            FIXTURE_EXPECTATION,
            "Really just vague handwaving on my part.")
        assert check.violations == ["vague handwaving"]

    def test_steering_cap_is_six(self) -> None:
        assert spike()["shipping"]["steering_cap_entries"] == 6
        exp = PersonaExpectation(
            record_id="cap", traits=[f"t{i}" for i in range(9)])
        line = exp.steering_lines()[0]
        assert "t5" in line and "t6" not in line


# ── The probes + the doc ─────────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_persona_probes()
        assert len(probes) == len(PERSONA_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert verdict_data_problems({})
        assert verdict_data_problems(None)
        assert comparison_row_problems({})
        assert measured_problems({}, {"identity_core": 10})
        assert measured_problems({"identity_core": 10}, {})
        assert schema_field_problems({})

    def test_fixtures_are_synthetic_only(self) -> None:
        module_text = (
            REPO_ROOT / "src" / "thoughtforge" / "cognition"
            / "persona_compiler_verdict.py").read_text(
                encoding="utf-8")
        banned = "vol" + "marr"
        assert f"user:{banned}" not in module_text
        assert "vey" + "runn" not in module_text
        assert "wrenna" in module_text.lower()
        assert len(fixture_records()) == 14

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / PERSONA_COMPILER_VERDICT_DOC
        assert doc.exists()
        assert doc.read_text(encoding="utf-8") == \
            render_persona_compiler_verdict_markdown()
        rendered = render_persona_compiler_verdict_markdown()
        assert "MERGE" in rendered
        assert "bell_count=two" in rendered
        # A tampered doc is not the render (the checker's shape).
        assert rendered != rendered.replace("MERGE", "RETIRE", 1)
