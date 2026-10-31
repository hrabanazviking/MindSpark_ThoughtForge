"""TF-048 — cite-or-explain I (retrieval side): every retrieved
item carries its citation payload into the prompt by construction.

Pins knowledge/citations.py + the CitationPayload on
ActivatedRecord against the live tree: derivation per kind
(TF-042 identity fields), totality (the marker is never
empty), construction-time attachment, the forge bundle
builder on every source shape (SQL + vector shapes), the
ConceptNet triple and vector-reference typing fixes, the
scaffold fact block's every-line-cited rendering, and the
end-to-end prompt — measured on the TF-043 fixture corpus
and through a real knowledge-only core turn.

Proof grade: FIXTURE (synthetic corpus, SQL path, VM).
Vector/hybrid retrieval itself NOT PROVEN (D-ENV-1); the
vector *shapes* are proven at the bundle builder, where the
payload attaches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.prompt_builder import PromptBuilder
from thoughtforge.cognition.scaffold import ScaffoldBuilder, _build_fact_block
from thoughtforge.knowledge.citations import (
    CITATION_FORMATS,
    citation_for_record,
    citation_problems,
    markers_for_bundle,
    payloads_for_bundle,
)
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.models import (
    ActivatedRecord,
    CitationPayload,
    InputSketch,
    MemoryActivationBundle,
    derive_citation,
)
from thoughtforge.knowledge.retrieval_precision import build_fixture_db
from thoughtforge.refinement.enforcement import _QID_RE

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "architecture" / "CITATIONS.md"


def _rec(record_id: str, record_type: str, raw: dict, score: float = 0.5,
         cue: str = "a cue") -> ActivatedRecord:
    return ActivatedRecord(record_id, record_type, score, cue, raw=raw)


# ── Self-check and formats ───────────────────────────────────────────────────


class TestSelfCheck:
    def test_citation_problems_empty(self) -> None:
        assert citation_problems() == []

    def test_doc_exists(self) -> None:
        assert DOC.exists()

    def test_formats_table(self) -> None:
        kinds = [row["kind"] for row in CITATION_FORMATS]
        assert kinds == [
            "entity", "reference_chunk", "geonames_place",
            "conceptnet_edge", "fallback",
        ]
        for row in CITATION_FORMATS:
            assert row["identity"] and row["marker_shape"] and row["example"]


# ── Derivation per kind ──────────────────────────────────────────────────────


class TestDerivation:
    def test_entity_marker_is_the_bare_qid(self) -> None:
        p = derive_citation("Q123", "entity", {
            "qid": "Q123", "label_en": "Odin", "source": "wikidata",
        })
        assert p.marker == "Q123"
        assert p.locator == "Q123"
        assert p.source == "wikidata"
        assert p.label == "Odin"
        # The prompt token and TF-049's check speak the same shape.
        assert _QID_RE.fullmatch(p.marker)

    def test_reference_chunk_marker_and_locator(self) -> None:
        p = derive_citation("chunk_7", "reference_chunk", {
            "chunk_id": "chunk_7", "source_file": "hollow.md",
            "title": "The Hollow", "source": "reference",
        })
        assert p.marker == "ref:chunk_7"
        assert p.locator == "hollow.md"  # TF-042: the file IS the provenance

    def test_geonames_marker(self) -> None:
        p = derive_citation("Hollow Creek", "place", {
            "geonames_id": 5128581, "label_en": "Hollow Creek",
            "source": "geonames",
        })
        assert p.marker == "geonames:5128581"
        assert p.locator == "5128581"

    def test_concept_edge_marker_is_the_triple(self) -> None:
        p = derive_citation("ferry", "concept_edge", {
            "concept1": "ferry", "relation": "UsedFor",
            "concept2": "crossing", "source": "conceptnet",
            "content": "ferry UsedFor crossing",
        })
        assert p.marker == "conceptnet:ferry|UsedFor|crossing"
        assert p.locator == "ferry UsedFor crossing"

    def test_fallback_marker_from_source_and_id(self) -> None:
        p = derive_citation("fct_1", "user_fact", {})
        assert p.marker == "user_fact:fct_1"
        assert p.locator == "fct_1"

    def test_derivation_is_total(self) -> None:
        # No raw at all, and even an empty id, still yield a marker.
        assert derive_citation("x", "entity", None).marker == "entity:x"
        assert derive_citation("", "entity", {}).marker == "entity:unidentified"
        assert derive_citation("", "", {}).marker == "unknown:unidentified"

    def test_derivation_is_deterministic(self) -> None:
        raw = {"qid": "Q5", "label_en": "Five", "source": "wikidata"}
        assert derive_citation("Q5", "entity", raw) == derive_citation("Q5", "entity", raw)


# ── Construction carries the payload ─────────────────────────────────────────


class TestConstruction:
    def test_bare_record_derives_its_payload(self) -> None:
        rec = _rec("Q9", "entity", {"qid": "Q9"})
        assert isinstance(rec.citation, CitationPayload)
        assert rec.citation.marker == "Q9"

    def test_supplied_payload_is_respected(self) -> None:
        supplied = CitationPayload(
            marker="custom:1", source="custom", record_id="r1",
            record_type="entity", locator="r1",
        )
        rec = ActivatedRecord("r1", "entity", 0.5, "cue", raw={}, citation=supplied)
        assert rec.citation is supplied

    def test_citation_for_record_matches_the_records_own(self) -> None:
        rec = _rec("chunk_1", "reference_chunk",
                   {"chunk_id": "chunk_1", "source_file": "a.md"})
        assert citation_for_record(rec) is rec.citation


# ── The forge bundle: every source shape ─────────────────────────────────────


class TestForgeBundle:
    def test_bundle_records_all_carry_payloads(self) -> None:
        forge = KnowledgeForge.__new__(KnowledgeForge)
        scored = [
            {"qid": "Q1", "label_en": "One", "description_en": "first",
             "source": "wikidata", "record_type": "entity", "_composite_score": 0.9},
            {"chunk_id": "c1", "source_file": "s.md", "title": "T",
             "content": "body", "source": "reference_chunk",
             "_composite_score": 0.8},  # vector shape: no record_type
            {"geonames_id": 7, "label_en": "Place", "description_en": "d",
             "source": "geonames", "record_type": "place", "_composite_score": 0.7},
        ]
        bundle = forge._build_bundle([dict(r) for r in scored])
        assert markers_for_bundle(bundle) == ["Q1", "ref:c1", "geonames:7"]
        assert all(p.marker for p in payloads_for_bundle(bundle))
        # The vector reference shape is a chunk, not an entity.
        chunk = bundle.activated_records[1]
        assert chunk.record_type == "reference_chunk"
        assert chunk.citation is not None and chunk.citation.locator == "s.md"


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_db(tmp_path_factory.mktemp("tf048") / "fixture.db")


class TestRealRetrieval:
    def test_every_retrieved_item_carries_a_payload(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("ferry crosses river", path="sql", min_score=0.0)
        assert bundle.activated_records
        for rec in bundle.activated_records:
            assert rec.citation is not None
            assert rec.citation.marker
            assert rec.citation.record_id == rec.record_id

    def test_concept_edge_triple_rides_the_raw(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("river crossing", path="sql", min_score=0.0)
        edges = [r for r in bundle.activated_records if r.record_type == "concept_edge"]
        assert edges, "the ferry/river ConceptNet edge must ride this query (TF-043)"
        for edge in edges:
            # The defect this slice fixed: the triple was selected
            # by the query and dropped before the raw dict.
            assert edge.raw["concept1"] and edge.raw["relation"] and edge.raw["concept2"]
            assert edge.citation is not None
            assert edge.citation.marker.startswith("conceptnet:")

    def test_entity_markers_are_qids(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("Lantern Hollow ferry", path="sql", min_score=0.0)
        entities = [r for r in bundle.activated_records if r.record_type == "entity"]
        assert entities
        for rec in entities:
            assert rec.citation is not None
            # The marker IS the record's QID. (The fixture's QIDs are
            # synthetic — "QF4301" — and do not match the real
            # Wikidata Q<digits> shape _QID_RE enforces; the regex
            # shape is pinned on a real-shaped QID in TestDerivation.)
            assert rec.citation.marker == rec.raw["qid"] == rec.record_id


# ── The fact block: every line cited ─────────────────────────────────────────


class TestFactBlock:
    def test_empty_bundle_empty_block(self) -> None:
        assert _build_fact_block([]) == ""

    def test_entity_line_is_byte_identical_to_the_legacy_shape(self) -> None:
        rec = _rec("Q123", "entity", {
            "qid": "Q123", "label_en": "Odin",
            "description_en": "chief god of Norse mythology",
            "source": "wikidata",
        }, score=0.9)
        # The legacy renderer's desc[:120].rsplit truncation drops
        # the final word even under 120 chars — pre-existing
        # behaviour this slice did not change; the line is pinned
        # as it actually renders, legacy and now alike.
        assert _build_fact_block([rec]) == "[Q123] Odin — chief god of Norse"

    def test_labelless_record_no_longer_falls_through_uncited(self) -> None:
        # Pre-TF-048 a record with no qid, no label, and no id
        # entered the prompt as its bare cue — no citation at all.
        rec = _rec("", "entity", {}, score=0.9, cue="orphan cue with no label")
        line = _build_fact_block([rec])
        assert line == "[entity:unidentified] orphan cue with no label"

    def test_every_line_opens_with_its_records_marker(self) -> None:
        records = [
            _rec("Q1", "entity", {"qid": "Q1", "label_en": "One"}, score=0.9),
            _rec("c1", "reference_chunk",
                 {"chunk_id": "c1", "source_file": "s.md", "title": "T"}, score=0.8),
            _rec("p1", "user_preference", {}, score=0.7, cue="prefers plain words"),
        ]
        lines = _build_fact_block(records).splitlines()
        assert len(lines) == 3
        for line, rec in zip(lines, records):
            assert rec.citation is not None
            assert line.startswith(f"[{rec.citation.marker}]")

    def test_budget_unchanged_top_eight_of_ten(self) -> None:
        records = [
            _rec(f"Q{i}", "entity", {"qid": f"Q{i}", "label_en": f"E{i}"},
                 score=i / 100.0)
            for i in range(100, 110)
        ]
        lines = _build_fact_block(records).splitlines()
        assert len(lines) == 8
        assert lines[0].startswith("[Q109]")  # top-scored first
        # Records past the budget still carry their payloads.
        assert all(r.citation is not None and r.citation.marker for r in records)


# ── Into the prompt, end to end ──────────────────────────────────────────────


class TestPromptEndToEnd:
    def _bundle(self) -> MemoryActivationBundle:
        records = [
            _rec("Q123", "entity", {
                "qid": "Q123", "label_en": "Odin",
                "description_en": "chief god", "source": "wikidata",
            }, score=0.9),
            _rec("chunk_7", "reference_chunk", {
                "chunk_id": "chunk_7", "source_file": "hollow.md",
                "title": "The Hollow", "content": "lantern oil",
                "source": "reference",
            }, score=0.8),
        ]
        return MemoryActivationBundle(
            activated_records=records, total_retrieved=2, retrieval_confidence=0.85,
        )

    def test_candidate_prompt_carries_every_marker(self) -> None:
        bundle = self._bundle()
        sketch = InputSketch(
            raw_text="who keeps the lantern oil?", intent="factual_query",
            topic="hollow", response_mode="factual_direct",
        )
        scaffold = ScaffoldBuilder().build(sketch, bundle)
        prompt = PromptBuilder().build_candidate_prompt(
            sketch, scaffold, "practical", [],
        )
        assert "[Q123]" in prompt
        assert "[ref:chunk_7]" in prompt
        # And they arrive inside the Knowledge context section.
        assert "Knowledge context:" in prompt

    def test_core_turn_fact_block_is_fully_cited(self, fixture_db: Path,
                                                  tmp_path: Path) -> None:
        core = ThoughtForgeCore(
            config={}, memory_dir=tmp_path / "mem", db_path=fixture_db,
        )
        # Path "sql": the core's default hybrid path needs
        # sentence-transformers, absent on this VM (D-ENV-1) —
        # the same reason TF-043 measures the SQL path.
        final = core.think("ferry crosses river", retrieval_path="sql")
        turn = core.last_turn_state
        assert turn is not None and turn.memory_bundle is not None
        bundle = turn.memory_bundle
        assert bundle.activated_records
        assert all(r.citation is not None and r.citation.marker
                   for r in bundle.activated_records)
        assert turn.scaffold is not None
        # The turn's fact_block is composite: TF-030 persona lines
        # ride in front of the retrieved lines. The retrieved lines
        # are exactly the builder's fact block for this bundle —
        # every one of them must be present and cited.
        expected = _build_fact_block(bundle.activated_records).splitlines()
        assert expected
        turn_lines = turn.scaffold.fact_block.splitlines()
        markers = set(markers_for_bundle(bundle))
        for line in expected:
            assert line in turn_lines
            match = re.match(r"\[([^\]]+)\]", line)
            assert match is not None, f"uncited fact line: {line!r}"
            assert match.group(1) in markers
        # The knowledge-only final cites bundle record ids, as before.
        assert set(final.citations) <= {r.record_id for r in bundle.activated_records}
