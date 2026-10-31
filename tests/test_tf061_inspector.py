"""TF-061 — memory inspector data: what the store holds
+ why a retrieval fired, exportable per turn (consumes
TF-038).

Pins knowledge/inspector.py against the live tree: the
forge's retained dimension explanations (D-INSP-1) and
the mirror that recomputes them, the forge's retained
retrieval report (fired + rejected-with-reasons), the
TF-038 trace's citation payloads (TF-048's assignment),
the knowledge-DB holdings report, the per-turn export
(store + DB + turn + retrieval + citations) and its
pure checker, and the core's last_turn_inspector()
join — including the stale-report guard. Fixtures are
synthetic (TF-043's fixture DB, TF-017's Lantern
Hollow shapes); no canon is read.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.traces import export_turn_trace
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.inspector import (
    CARRIED_EVIDENCE_KEYS,
    DIMENSION_RULES,
    INSPECTOR_FORMAT,
    INSPECTOR_SECTIONS,
    build_inspector_report,
    dimension_explanation,
    explain_record,
    explain_retrieval,
    inspector_live_problems,
    inspector_problems,
    inspector_to_json,
    knowledge_db_report,
    write_inspector_report,
)
from thoughtforge.knowledge.models import (
    MemoryActivationBundle,
    RuntimeTurnState,
    UserFactRecord,
)
from thoughtforge.knowledge.multilingual import search_store as ml_search
from thoughtforge.knowledge.retrieval_precision import build_fixture_db
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.refinement.citation_enforcement import CitationCheck

TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)

FTS_RECORD = {
    "qid": "Q1", "label_en": "Ferry",
    "description_en": "crosses the hollow",
    "popularity_score": 0.9, "fts_score": 10.0,
    "source": "wikidata", "record_type": "entity",
}
VECTOR_RECORD = {
    "chunk_id": "c1", "title": "T", "content": "lantern oil",
    "similarity": 0.8, "source": "reference_chunk",
    "record_type": "reference_chunk",
}
OLD_RECORD = {
    "qid": "Q2", "label_en": "Old Ferry",
    "description_en": "crosses the hollow",
    "popularity_score": 0.5, "fts_score": 4.0,
    "source": "wikidata", "record_type": "entity",
    "updated_at": "2020-01-01T00:00:00+00:00",
}


class StubBackend(UnifiedBackend):
    """One completed-text candidate per turn (the unified path)."""

    def generate(self, request):
        return GenerationResponse(
            text=TEXT, tokens_generated=len(TEXT.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "tf061-stub"


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_db(tmp_path_factory.mktemp("tf061") / "fixture.db")


def _forge() -> KnowledgeForge:
    forge = KnowledgeForge.__new__(KnowledgeForge)
    forge.retrieval_weights = None
    forge._last_retrieval_report = None
    return forge


def _core(db_path: Path, **kwargs) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=db_path,
        model_path=None,
        **kwargs,
    )


# ── Format & rules as data ───────────────────────────────────────────────────


class TestFormatAndRules:
    def test_format_and_sections(self) -> None:
        assert INSPECTOR_FORMAT == "thoughtforge-inspector/1"
        assert INSPECTOR_SECTIONS == (
            "store", "knowledge_db", "turn", "retrieval", "citations",
        )

    def test_dimension_rules_cover_every_dimension(self) -> None:
        named = " ".join(row["dimension"] for row in DIMENSION_RULES)
        for dimension in (
            "semantic_similarity", "tone_similarity",
            "preference_relevance", "recency", "importance",
        ):
            assert dimension in named

    def test_carried_evidence_keys_are_the_searches_keys(self) -> None:
        assert "matched_terms" in CARRIED_EVIDENCE_KEYS
        assert "freshness" in CARRIED_EVIDENCE_KEYS
        assert "language" in CARRIED_EVIDENCE_KEYS


# ── The dimension explanation ────────────────────────────────────────────────


class TestDimensionExplanation:
    def test_fts_proxy_shape_reproduces_the_measured_composite(self) -> None:
        # The probe-P1 record: TF-059's pinned 0.57 composite.
        explanation = dimension_explanation(dict(FTS_RECORD))
        assert explanation["weights_row"] == "user_fact"
        assert explanation["semantic_source"] == "fts_proxy"
        assert explanation["recency_source"] == "source_proxy"
        assert explanation["dimensions"]["semantic_similarity"] == 0.5
        assert explanation["dimensions"]["recency"] == 0.7
        assert explanation["dimensions"]["importance"] == 1.0
        assert explanation["composite"] == pytest.approx(0.57)

    def test_contributions_sum_to_the_composite(self) -> None:
        for raw in (FTS_RECORD, VECTOR_RECORD, OLD_RECORD):
            explanation = dimension_explanation(dict(raw))
            assert sum(explanation["contributions"].values()) == (
                pytest.approx(explanation["composite"])
            )

    def test_vector_shape_prefers_similarity(self) -> None:
        explanation = dimension_explanation(dict(VECTOR_RECORD))
        assert explanation["semantic_source"] == "vector_similarity"
        assert explanation["dimensions"]["semantic_similarity"] == 0.8

    def test_timestamped_record_ages_for_real(self) -> None:
        explanation = dimension_explanation(dict(OLD_RECORD))
        assert explanation["recency_source"] == "record_timestamp"
        assert explanation["dimensions"]["recency"] < 0.01
        fresh = dimension_explanation(
            dict(OLD_RECORD, updated_at="2999-01-01T00:00:00+00:00")
        )
        assert fresh["dimensions"]["recency"] == pytest.approx(1.0)

    def test_no_signal_shape_is_stated_none(self) -> None:
        explanation = dimension_explanation(
            {"qid": "Q9", "label_en": "Mute", "source": "wikidata",
             "record_type": "entity"}
        )
        assert explanation["semantic_source"] == "none"
        assert explanation["dimensions"]["semantic_similarity"] == 0.0

    def test_custom_weights_name_themselves(self) -> None:
        row = {"semantic_similarity": 1.0}
        explanation = dimension_explanation(dict(FTS_RECORD), weights=row)
        assert explanation["weights_row"] == "custom"
        assert explanation["weights"] == row
        assert explanation["composite"] == pytest.approx(0.5)


# ── D-INSP-1: the forge retains its dimensions ───────────────────────────────


class TestForgeRetention:
    def test_score_records_retains_the_explanation(self) -> None:
        scored = _forge()._score_records([dict(FTS_RECORD)], "ferry")
        explanation = scored[0]["_explanation"]
        assert explanation["composite"] == scored[0]["_composite_score"]
        assert explanation["semantic_source"] == "fts_proxy"

    def test_mirror_agrees_with_retained_on_every_shape(self) -> None:
        forge = _forge()
        for raw in (FTS_RECORD, VECTOR_RECORD):
            scored = forge._score_records([dict(raw)], "q")
            mirrored = dimension_explanation(dict(raw))
            assert mirrored == scored[0]["_explanation"]

    def test_mirror_agrees_with_retained_timestamped_within_tolerance(
        self,
    ) -> None:
        scored = _forge()._score_records([dict(OLD_RECORD)], "q")
        mirrored = dimension_explanation(dict(OLD_RECORD))
        assert mirrored["composite"] == pytest.approx(
            scored[0]["_explanation"]["composite"], abs=1e-4
        )
        assert mirrored["recency_source"] == "record_timestamp"


# ── The retrieval report: fired + rejected, with reasons ─────────────────────


class TestRetrievalReport:
    def test_report_is_none_before_the_first_retrieve(
        self, fixture_db: Path
    ) -> None:
        assert KnowledgeForge(db_path=fixture_db).last_retrieval_report is None

    def test_fired_ids_are_the_bundle_ids(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("ferry crossing", path="sql", top_k=3)
        report = forge.last_retrieval_report
        assert report is not None
        assert report["fired"] == [r.record_id for r in bundle.activated_records]
        assert report["considered"] >= len(report["fired"])

    def test_beyond_top_k_rejections_are_named(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        forge.retrieve("Lantern Hollow ferry", path="sql", top_k=1)
        report = forge.last_retrieval_report
        assert report is not None
        reasons = {r["reason"] for r in report["rejected"]}
        assert "beyond_top_k" in reasons
        for rejected in report["rejected"]:
            assert rejected["record_id"]
            assert "composite_score" in rejected

    def test_below_min_score_rejections_are_named(
        self, fixture_db: Path
    ) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve(
            "ferry crossing", path="sql", top_k=3, min_score=0.99
        )
        assert bundle.activated_records == []
        report = forge.last_retrieval_report
        assert report is not None
        assert report["rejected"]
        assert {r["reason"] for r in report["rejected"]} == {"below_min_score"}

    def test_the_report_is_detached(self, fixture_db: Path) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        forge.retrieve("ferry crossing", path="sql", top_k=3)
        report = forge.last_retrieval_report
        assert report is not None
        report["fired"].append("tampered")
        again = forge.last_retrieval_report
        assert again is not None
        assert "tampered" not in again["fired"]


# ── One record / one bundle, explained ───────────────────────────────────────


class TestExplain:
    def test_explain_record_prefers_the_retained_explanation(
        self, fixture_db: Path
    ) -> None:
        forge = KnowledgeForge(db_path=fixture_db)
        bundle = forge.retrieve("ferry crossing", path="sql", top_k=3)
        entry = explain_record(bundle.activated_records[0])
        assert entry["explanation_source"] == "retained_at_retrieval"
        assert entry["citation"]["marker"]
        assert entry["dimensions"]["composite"] == pytest.approx(
            bundle.activated_records[0].score, abs=1e-3
        )

    def test_explain_record_recomputes_a_pre_retention_record(self) -> None:
        forge = _forge()
        bundle = forge._build_bundle(
            forge._score_records([dict(FTS_RECORD)], "ferry")
        )
        record = bundle.activated_records[0]
        del record.raw["_explanation"]
        entry = explain_record(record)
        assert entry["explanation_source"] == "recomputed_at_inspection"
        assert entry["dimensions"]["composite"] == pytest.approx(0.57)

    def test_explain_record_surfaces_carried_evidence(
        self, tmp_path: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.append_profile_record(UserFactRecord(
            record_id="insp_es", tags=["faro", "travesía"],
            summary="el faro rojo guía la travesía nocturna",
            fact_value={"lugar": "biblioteca del faro"},
            source="synthetic_fixture"))
        bundle, _report = ml_search(store, "faro travesía")
        assert bundle.activated_records
        entry = explain_record(bundle.activated_records[0])
        assert entry["explanation_source"] == "carried_evidence"
        assert entry["carried_evidence"]["matched_terms"]
        assert "language" in entry["carried_evidence"]

    def test_explain_retrieval_without_a_report_says_unknown(self) -> None:
        section = explain_retrieval(MemoryActivationBundle())
        assert section["rejections_known"] is False
        assert section["rejected"] is None
        assert "unknown" in section["note"]


# ── The trace carries citation payloads (TF-048's assignment) ────────────────


class TestTraceCitations:
    def test_trace_retrieval_entries_carry_the_payload(self) -> None:
        forge = _forge()
        bundle = forge._build_bundle(
            forge._score_records([dict(FTS_RECORD)], "ferry")
        )
        turn = RuntimeTurnState(turn_id="tf061_trace")
        turn.memory_bundle = bundle
        trace = export_turn_trace(turn)
        entry = trace["retrieval"]["activated_records"][0]
        assert entry["citation"]["marker"] == "Q1"
        assert entry["citation"]["record_id"] == "Q1"

    def test_real_turn_trace_carries_payloads(self, fixture_db: Path) -> None:
        core = _core(fixture_db)
        core.think("Lantern Hollow ferry", retrieval_path="sql")
        trace = core.last_turn_trace()
        assert trace is not None
        entries = trace["retrieval"]["activated_records"]
        assert entries
        for entry in entries:
            assert entry["citation"]["marker"]


# ── What the estates hold ────────────────────────────────────────────────────


class TestKnowledgeDbReport:
    def test_missing_db_is_a_verdict(self, tmp_path: Path) -> None:
        report = knowledge_db_report(tmp_path / "nope.db")
        assert report["present"] is False

    def test_no_path_is_a_named_verdict(self) -> None:
        report = knowledge_db_report(None)
        assert report["present"] is False
        assert "reason" in report

    def test_fixture_db_tables_are_counted(self, fixture_db: Path) -> None:
        report = knowledge_db_report(fixture_db)
        assert report["present"] is True
        assert report["readable"] is True
        assert report["tables"]["entities"] > 0

    def test_unreadable_db_is_a_verdict(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"this is not a database")
        report = knowledge_db_report(bad)
        assert report["present"] is True
        assert report["readable"] is False
        assert "error" in report


# ── The per-turn export ──────────────────────────────────────────────────────


def _full_report(tmp_path: Path, fixture_db: Path) -> dict:
    store = MemoryStore(memory_dir=tmp_path / "mem")
    store.append_profile_record(UserFactRecord(
        record_id="insp_fact", tags=["ferry"],
        summary="the ferry crosses at dawn",
        source="synthetic_fixture"))
    forge = KnowledgeForge(db_path=fixture_db)
    bundle = forge.retrieve("ferry crossing", path="sql", top_k=3)
    turn = RuntimeTurnState(turn_id="tf061_full")
    turn.memory_bundle = bundle
    return build_inspector_report(
        store=store,
        db_path=fixture_db,
        trace=export_turn_trace(turn),
        bundle=bundle,
        retrieval_report=forge.last_retrieval_report,
    )


class TestBuildReport:
    def test_full_report_is_clean(self, tmp_path: Path, fixture_db: Path) -> None:
        report = _full_report(tmp_path, fixture_db)
        assert report["format"] == INSPECTOR_FORMAT
        assert inspector_problems(report) == []
        assert report["store"]["present"] is True
        assert report["store"]["total_records"] >= 1
        assert report["knowledge_db"]["present"] is True
        assert report["turn"]["turn_id"] == "tf061_full"
        assert report["retrieval"]["rejections_known"] is True
        assert report["citations"]["present"] is False

    def test_report_json_round_trips(self, tmp_path: Path, fixture_db: Path) -> None:
        report = _full_report(tmp_path, fixture_db)
        assert json.loads(inspector_to_json(report))["format"] == INSPECTOR_FORMAT
        path = write_inspector_report(report, tmp_path / "out" / "insp.json")
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
            inspector_to_json(report)
        )

    def test_building_a_report_never_writes_the_store(
        self, tmp_path: Path, fixture_db: Path
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.append_profile_record(UserFactRecord(
            record_id="insp_fact", tags=["ferry"],
            summary="the ferry crosses at dawn",
            source="synthetic_fixture"))
        before = {
            p.name: p.read_bytes()
            for p in sorted(store.dir.iterdir()) if p.is_file()
        }
        build_inspector_report(store=store, db_path=fixture_db)
        after = {
            p.name: p.read_bytes()
            for p in sorted(store.dir.iterdir()) if p.is_file()
        }
        assert before == after

    def test_empty_inputs_are_stated_verdicts(self) -> None:
        report = build_inspector_report()
        assert report["store"] == {"present": False}
        assert report["knowledge_db"]["present"] is False
        assert report["turn"] == {"present": False}
        assert report["retrieval"] == {"present": False}
        assert inspector_problems(report) == []

    def test_citation_check_rides_the_export(self, fixture_db: Path) -> None:
        check = CitationCheck(foreign_markers=["Q999"])
        report = build_inspector_report(
            db_path=fixture_db, citation_check=check
        )
        assert report["citations"]["present"] is True
        assert report["citations"]["passed"] is False
        assert report["citations"]["foreign_markers"] == ["Q999"]


# ── The pure checker catches tampering ───────────────────────────────────────


class TestInspectorProblems:
    def test_wrong_format_and_missing_section(self) -> None:
        problems = inspector_problems({"format": "nope"})
        assert any("format" in p for p in problems)
        assert any("missing section" in p for p in problems)

    def test_record_without_citation_is_named(
        self, tmp_path: Path, fixture_db: Path
    ) -> None:
        report = _full_report(tmp_path, fixture_db)
        del report["retrieval"]["records"][0]["citation"]
        problems = inspector_problems(report)
        assert any("citation" in p for p in problems)

    def test_contributions_must_sum_to_the_composite(
        self, tmp_path: Path, fixture_db: Path
    ) -> None:
        report = _full_report(tmp_path, fixture_db)
        dims = report["retrieval"]["records"][0]["dimensions"]
        dims["contributions"]["importance"] = 99.0
        problems = inspector_problems(report)
        assert any("contributions" in p for p in problems)

    def test_unknown_rejection_reason_is_named(
        self, tmp_path: Path, fixture_db: Path
    ) -> None:
        report = _full_report(tmp_path, fixture_db)
        report["retrieval"]["rejected"] = [
            {"record_id": "Qx", "reason": "vibes"}
        ]
        problems = inspector_problems(report)
        assert any("unknown reason" in p for p in problems)


# ── The core surface ─────────────────────────────────────────────────────────


class TestCoreInspector:
    def test_none_before_the_first_turn(self, fixture_db: Path) -> None:
        assert _core(fixture_db).last_turn_inspector() is None

    def test_knowledge_only_turn_inspects_clean(
        self, fixture_db: Path
    ) -> None:
        core = _core(fixture_db)
        core.think("Lantern Hollow ferry", retrieval_path="sql")
        report = core.last_turn_inspector()
        assert report is not None
        assert inspector_problems(report) == []
        assert report["turn"]["present"] is True
        records = report["retrieval"]["records"]
        assert records
        for entry in records:
            assert entry["explanation_source"] == "retained_at_retrieval"
            assert entry["citation"]["marker"]
        assert report["retrieval"]["rejections_known"] is True

    def test_no_db_turn_reports_rejections_unknown(
        self, tmp_path: Path
    ) -> None:
        core = _core(tmp_path / "missing.db")
        core.think("What time does the ferry leave?")
        report = core.last_turn_inspector()
        assert report is not None
        assert report["knowledge_db"]["present"] is False
        assert report["retrieval"]["rejections_known"] is False

    def test_composed_turn_carries_the_citation_check(
        self, fixture_db: Path
    ) -> None:
        core = _core(fixture_db, backend=StubBackend())
        core.think("Lantern Hollow ferry", retrieval_path="sql")
        report = core.last_turn_inspector()
        assert report is not None
        assert report["citations"]["present"] is True


# ── The live checker ─────────────────────────────────────────────────────────


def test_inspector_live_problems() -> None:
    assert inspector_live_problems() == []
