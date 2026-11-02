"""
TF-140 — Micro-RAG verdict: its retrieval vs Part
III's engine on TF-043's fixtures — the better
mechanism wins by measurement.

The proof lives as verifiable data in
``src/thoughtforge/knowledge/micro_rag_verdict.py``
(``COMPARISON_ROWS``, ``VERDICT``, the ``Spike*``
dataclasses + ``SpikeMicroRAG``, ``run_micro_spike()``,
``MICRO_PROBES``, checked by
``micro_rag_verdict_problems()``); the write-up
``docs/architecture/MICRO_RAG_VERDICT.md`` is filed
verbatim as the module's render.

Pinned here:

  * the verdict: RETIRE (as a retrieval mechanism),
    decided by measurement — the shipping engine wins
    precision / recall / F1 at its defaults and at the
    open setting, MRR ties; retirement owned by
    TF-144, no adoption for TF-143, the wing's tests
    by TF-149,
  * the comparison: 9 rows bound live to the tree —
    the wing's models package does NOT exist, its 14
    schemas parse, the pipeline's nine score weights
    and MODE_TARGETS match the spike's,
  * the measurement: both mechanisms over TF-043's
    corpus verbatim (106 spike items under the filed
    adaptation), the filed aggregates + the micro
    returned ids per query reproduced exactly, the
    decisive queries pinned (Q6 negative: micro
    returns five junk items and scores 0/0, shipping
    returns nothing and scores 1/1; Q9 geonames: the
    micro spike's one win, recall 1.0 vs 0.6667),
  * the spike mechanics: the exact score arithmetic
    on a known item, the per-family cap of 5, the
    budget law, the typed section mapping,
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures are TF-043's synthetic Lantern Hollow
corpus (TF-017); no canon, no real persona, risky
markers assembled from parts. Proof grade: FIXTURE
— no live model was read, never claimed.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

from thoughtforge.knowledge.micro_rag_verdict import (
    COMPARISON_ROWS,
    FINDINGS,
    MICRO_PROBES,
    MICRO_RAG_VERDICT_DOC,
    SCORE_WEIGHTS,
    SPIKE_EXPECTED,
    VERDICT,
    VERDICTS,
    QueryMode,
    SpikeMicroRAG,
    SpikeRetrievalItem,
    SpikeTruthPacket,
    comparison_row_problems,
    fixture_items,
    measured_problems,
    micro_rag_verdict_problems,
    render_micro_rag_verdict_markdown,
    run_micro_probes,
    run_micro_spike,
    schema_field_problems,
    score_weights_problems,
    spike_field_map,
    verdict_data_problems,
)
from thoughtforge.knowledge.retrieval_precision import (
    BASELINE,
    FIXTURE_ENTITIES,
    FIXTURE_QUERIES,
)

REPO_ROOT = Path(__file__).parent.parent
WING = REPO_ROOT / "research_data" / "src" / "wyrdforge"

_SPIKE: dict | None = None


def spike() -> dict:
    global _SPIKE
    if _SPIKE is None:
        _SPIKE = run_micro_spike()
    return _SPIKE


def by_id(section: dict) -> dict[str, dict]:
    return {q["id"]: q for q in section["per_query"]}


# ── The verdict as data ──────────────────────────────────────────────────────

class TestVerdictData:
    def test_problems_empty(self) -> None:
        assert micro_rag_verdict_problems() == []

    def test_verdict_is_retire(self) -> None:
        assert VERDICT["verdict"] == "RETIRE"
        assert VERDICT["verdict"] in VERDICTS
        assert verdict_data_problems(VERDICT) == []

    def test_reasons_cover_measured_loss_not_adopt_not_merge(self) -> None:
        reasons = " ".join(VERDICT["reasons"])
        assert "0.8889" in reasons and "0.3333" in reasons
        assert "not adopt" in reasons
        assert "not a merge" in reasons
        assert "no score floor" in reasons

    def test_conditions_name_the_owners(self) -> None:
        conditions = " ".join(VERDICT["conditions"])
        for owner in ("TF-144", "TF-143", "TF-149"):
            assert owner in conditions
        assert "NO shipping behaviour" in conditions
        assert "sole retriever" in conditions

    def test_comparison_rows_shape(self) -> None:
        assert len(COMPARISON_ROWS) == 9
        dimensions = [r["dimension"] for r in COMPARISON_ROWS]
        assert len(dimensions) == len(set(dimensions))
        for row in COMPARISON_ROWS:
            assert comparison_row_problems(row) == []
            assert row["finding"] in FINDINGS
        findings = [r["finding"] for r in COMPARISON_ROWS]
        assert findings.count("shipping-wins") == 3
        assert findings.count("micro-wins") == 1
        assert findings.count("different") == 4
        assert findings.count("overlap") == 1
        by_dim = {r["dimension"]: r for r in COMPARISON_ROWS}
        assert by_dim["place_records_geonames"]["finding"] == "micro-wins"
        assert by_dim["citation_handling"]["finding"] == "overlap"


# ── Bound live to the tree ───────────────────────────────────────────────────

class TestLiveTreeBinding:
    def test_wing_models_package_does_not_exist(self) -> None:
        assert not (WING / "models").exists()
        pipeline = (WING / "services" / "micro_rag_pipeline.py").read_text(
            encoding="utf-8")
        assert "from wyrdforge.models.micro_rag import" in pipeline

    def test_schemas_exist_and_parse(self) -> None:
        schema_files = sorted((WING / "schemas").glob("*.json"))
        assert len(schema_files) == 14
        for schema_file in schema_files:
            json.loads(schema_file.read_text(encoding="utf-8"))

    def test_pipeline_weights_match_spike(self) -> None:
        pipeline = (WING / "services" / "micro_rag_pipeline.py").read_text(
            encoding="utf-8")
        for name, weight in SCORE_WEIGHTS.items():
            assert f"{name} * {weight}" in pipeline or \
                f"{name} * {weight:.2f}" in pipeline, name
        assert score_weights_problems(SCORE_WEIGHTS) == []

    def test_mode_targets_match_pipeline(self) -> None:
        tree = ast.parse(
            (WING / "services" / "micro_rag_pipeline.py").read_text(
                encoding="utf-8"))
        wing_targets: dict[str, tuple[str, ...]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "MODE_TARGETS"
                    for t in node.targets):
                for key, value in zip(node.value.keys, node.value.values):
                    wing_targets[key.attr] = tuple(
                        elt.value for elt in value.elts)
        spike_targets = {
            mode.name: families
            for mode, families in SpikeMicroRAG.MODE_TARGETS.items()}
        assert wing_targets == spike_targets
        assert spike_targets["FACTUAL_LOOKUP"] == (
            "canonical", "recent", "contradiction")

    def test_spike_fields_match_schemas(self) -> None:
        assert schema_field_problems(spike_field_map()) == []
        item_fields = {
            f.name for f in dataclasses.fields(SpikeRetrievalItem)}
        assert item_fields == {
            "item_id", "item_type", "text", "source_ref",
            "support_class", "confidence", "lexical_terms", "facets",
            "token_cost",
        }

    def test_enum_matches_schema(self) -> None:
        packet_schema = json.loads(
            (WING / "schemas" / "micro_context_packet.schema.json")
            .read_text(encoding="utf-8"))
        assert [m.value for m in QueryMode] == \
            packet_schema["$defs"]["QueryMode"]["enum"]


# ── The fixture adaptation ───────────────────────────────────────────────────

class TestFixtureAdaptation:
    def test_corpus_shape(self) -> None:
        items = fixture_items()
        assert len(items) == 106
        assert sum(1 for i in items if i.source_ref == "wikidata") == 68
        assert sum(1 for i in items if i.source_ref == "reference") == 34
        assert sum(1 for i in items if i.source_ref == "conceptnet") == 2
        assert sum(1 for i in items if i.source_ref == "geonames") == 2

    def test_relevant_ids_are_covered(self) -> None:
        item_ids = {i.item_id for i in fixture_items()}
        for row in FIXTURE_QUERIES:
            assert set(row["relevant"]) <= item_ids, row["id"]

    def test_defaults_are_the_schema_defaults(self) -> None:
        popularity = {row[0]: row[3] for row in FIXTURE_ENTITIES}
        for item in fixture_items():
            assert item.token_cost == 80
            assert item.lexical_terms == [] and item.facets == {}
            if item.item_id in popularity:
                # Entity confidence IS the fixture's filed popularity.
                assert item.confidence == popularity[item.item_id]
            elif item.item_type == "entity":
                assert item.confidence == 0.1  # filler, as TF-043 files
            else:
                assert item.confidence == 0.5  # the schema default

    def test_fixtures_are_synthetic_only(self) -> None:
        module_text = (
            REPO_ROOT / "src" / "thoughtforge" / "knowledge"
            / "micro_rag_verdict.py").read_text(encoding="utf-8")
        banned = "vol" + "marr"
        assert f"user:{banned}" not in module_text
        assert "vey" + "runn" not in module_text
        assert "Lantern Hollow" in module_text  # TF-043's world


# ── The spike mechanics ──────────────────────────────────────────────────────

class TestSpikeMechanics:
    def test_score_arithmetic_on_a_known_item(self) -> None:
        pipeline = SpikeMicroRAG()
        item = SpikeRetrievalItem(
            item_id="ferry", item_type="concept_edge",
            text="ferry crosses river", source_ref="conceptnet")
        scored = pipeline.score(
            "ferry crosses river", item, QueryMode.FACTUAL_LOOKUP)
        assert scored.similarity == 1.0
        assert scored.task_relevance == 0.5
        assert scored.recency == 0.5
        # 0.24 + 0.12 + 0.07 + 0.12 + 0.04 - 0.008, rounded to 4dp.
        assert scored.final_score == 0.582

    def test_assemble_caps_each_family_at_five(self) -> None:
        pipeline = SpikeMicroRAG()
        items = [
            SpikeRetrievalItem(
                item_id=f"i{n}", item_type="entity",
                text="lantern hollow ferry", source_ref="wikidata",
                token_cost=500)
            for n in range(7)
        ]
        packet = pipeline.assemble(
            query="lantern hollow ferry", mode=QueryMode.FACTUAL_LOOKUP,
            candidates_by_family={"canonical": items},
            packet_budget=100000)
        assert len(packet.canonical_facts) == 5
        assert packet.packet_budget_used == 2500

    def test_assemble_budget_skips_what_does_not_fit(self) -> None:
        pipeline = SpikeMicroRAG()
        items = [
            SpikeRetrievalItem(
                item_id=f"i{n}", item_type="entity",
                text="lantern hollow ferry", source_ref="wikidata",
                token_cost=500)
            for n in range(7)
        ]
        packet = pipeline.assemble(
            query="lantern hollow ferry", mode=QueryMode.FACTUAL_LOOKUP,
            candidates_by_family={"canonical": items})
        assert len(packet.canonical_facts) == 1  # 500 fits, +500 > 900
        assert packet.packet_budget_used == 500

    def test_assemble_files_typed_sections_and_truth_packet(self) -> None:
        pipeline = SpikeMicroRAG()
        truth = SpikeTruthPacket(must_be_true=["no vow exists"])
        packet = pipeline.assemble(
            query="companion packet", mode=QueryMode.COMPANION_CONTINUITY,
            truth_packet=truth,
            candidates_by_family={
                "bond": [SpikeRetrievalItem(
                    item_id="b1", item_type="bond", text="trust high",
                    source_ref="bond_graph", token_cost=60)],
                "canonical": [SpikeRetrievalItem(
                    item_id="c1", item_type="canonical_fact",
                    text="companion continuity", source_ref="canonical",
                    token_cost=70)],
            })
        assert [c.item_id for c in packet.bond_excerpt] == ["b1"]
        assert [c.item_id for c in packet.canonical_facts] == ["c1"]
        assert packet.truth_packet.must_be_true == ["no vow exists"]
        assert packet.packet_budget_used == 130


# ── The measurement ──────────────────────────────────────────────────────────

class TestMeasurement:
    def test_run_matches_filed_expectation(self) -> None:
        result = spike()
        assert result["micro"]["aggregate"] == \
            SPIKE_EXPECTED["micro_aggregate"]
        assert result["shipping_default"]["aggregate"] == \
            SPIKE_EXPECTED["shipping_default_aggregate"]
        assert result["shipping_open"]["aggregate"] == \
            SPIKE_EXPECTED["shipping_open_aggregate"]
        returned = {
            q["id"]: q["returned"] for q in result["micro"]["per_query"]}
        assert returned == SPIKE_EXPECTED["micro_returned"]
        assert measured_problems(
            result["micro"]["aggregate"],
            SPIKE_EXPECTED["micro_aggregate"]) == []

    def test_run_is_deterministic(self) -> None:
        assert run_micro_spike() == spike()

    def test_shipping_aggregates_match_tf043_baseline(self) -> None:
        result = spike()
        for setting, key in (("default", "shipping_default"),
                             ("open", "shipping_open")):
            for metric, pinned in BASELINE[setting].items():
                assert result[key]["aggregate"][metric] == pinned, metric

    def test_shipping_wins_the_aggregates_mrr_ties(self) -> None:
        result = spike()
        micro = result["micro"]["aggregate"]
        ship = result["shipping_default"]["aggregate"]
        assert ship["mean_precision"] > micro["mean_precision"]
        assert ship["mean_recall"] > micro["mean_recall"]
        assert ship["mean_f1"] > micro["mean_f1"]
        assert ship["mean_reciprocal_rank"] == \
            micro["mean_reciprocal_rank"] == 1.0
        # Even with its threshold removed the shipping engine wins.
        open_agg = result["shipping_open"]["aggregate"]
        assert open_agg["mean_precision"] > micro["mean_precision"]
        assert open_agg["mean_f1"] > micro["mean_f1"]

    def test_negative_query_is_decisive(self) -> None:
        result = spike()
        micro_q6 = by_id(result["micro"])["Q6"]
        assert micro_q6["returned"] == [
            "QF4301", "QF4302", "QF4303", "QF4304", "QF4305"]
        assert micro_q6["precision"] == 0.0
        assert micro_q6["recall"] == 0.0
        ship_q6 = by_id(result["shipping_default"])["Q6"]
        assert ship_q6["returned"] == []
        assert ship_q6["precision"] == 1.0
        assert ship_q6["recall"] == 1.0

    def test_geonames_query_is_the_micro_win(self) -> None:
        result = spike()
        micro_q9 = by_id(result["micro"])["Q9"]
        assert micro_q9["recall"] == 1.0
        assert "Lantern Hollow" in micro_q9["returned"]
        ship_q9 = by_id(result["shipping_default"])["Q9"]
        assert ship_q9["recall"] == 2 / 3
        assert "Lantern Hollow" not in ship_q9["returned"]

    def test_every_micro_packet_uses_exactly_400_budget(self) -> None:
        result = spike()
        assert {
            q["packet_budget_used"]
            for q in result["micro"]["per_query"]} == {400}
        assert result["micro"]["aggregate"]["total_returned"] == 45
        assert result["shipping_default"]["aggregate"][
            "total_returned"] == 17


# ── The probes + the doc ─────────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_micro_probes()
        assert len(probes) == len(MICRO_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert verdict_data_problems({})
        assert verdict_data_problems(None)
        assert comparison_row_problems({})
        assert score_weights_problems({})
        assert measured_problems({}, {"mean_precision": 0.5})
        assert measured_problems({"mean_precision": 0.5}, {})
        assert schema_field_problems({})

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / MICRO_RAG_VERDICT_DOC
        assert doc.exists()
        assert doc.read_text(encoding="utf-8") == \
            render_micro_rag_verdict_markdown()
        rendered = render_micro_rag_verdict_markdown()
        assert "RETIRE" in rendered
        assert "0.3333" in rendered and "0.8889" in rendered
        # A tampered doc is not the render (the checker's shape).
        assert rendered != rendered.replace("RETIRE", "MERGE", 1)
