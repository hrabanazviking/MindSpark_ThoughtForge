"""
TF-042 — Ingestion contract: what a "fact" is in ThoughtForge
(shape, provenance, confidence) — one schema, validated.

The contract lives as verifiable data in
``src/thoughtforge/knowledge/ingestion_contract.py``
(``SCHEMA_PARTS`` / ``FACT_KINDS`` / ``DB_PROVENANCE`` /
``MEMORY_PROVENANCE``, checked by ``contract_problems()`` and
enforced by ``record_problems()`` / ``db_row_problems()`` /
``writeback_fact_problems()``); the write-up is
``docs/architecture/INGESTION_CONTRACT.md``.

Pinned here:

  * the schema covers the whole estate exactly — 6 parts, 12
    kinds (6 memory records + 6 knowledge-DB tables), and the
    contract re-checks itself against the live tree with no
    drift (defaults, summary caps, schema provenance),
  * the memory provenance gap is pinned, not hidden: no
    persisted memory record has a field to carry a stamp,
  * a memory record is validated on shape, content,
    confidence-range, caps, closed vocabularies, counters and
    timestamps — each violation named by its field,
  * MemoryStore's ingestion gates (append_*/save_*) refuse an
    invalid record before anything is written and accept every
    valid shape the estate already uses, while load paths keep
    TF-041's legacy tolerance (an invalid stored record loads
    and is flagged by the validator, never dropped),
  * DB rows are validated per kind — provenance from the closed
    vocabulary, entity popularity as a capped proxy, ConceptNet
    weight as unbounded strength (a weight of 2.5 is valid; a
    confidence of 2.5 never is),
  * writeback fact dicts (TF-047's raw material) require
    content AND a provenance stamp from MEMORY_PROVENANCE.

Proof grade: FIXTURE (synthetic records, rows and stores on
this VM). No live ETL run is re-validated here — TF-108 owns
schema validation at the ETL gate; this slice states the schema
that gate will enforce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.knowledge.ingestion_contract import (
    DB_PROVENANCE,
    FACT_KINDS,
    MEMORY_PROVENANCE,
    SCHEMA_PARTS,
    contract_problems,
    db_row_problems,
    record_problems,
    writeback_fact_problems,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DOC = REPO_ROOT / "docs" / "architecture" / "INGESTION_CONTRACT.md"


def valid_fact(**overrides: object) -> UserFactRecord:
    base: dict[str, object] = {
        "record_id": "fact_1",
        "tags": ["ferry"],
        "summary": "the ferry takes eleven minutes",
        "fact_value": {"crossing_minutes": 11},
    }
    base.update(overrides)
    return UserFactRecord(**base)  # type: ignore[arg-type]


# ── The contract itself ──────────────────────────────────────────────────────


class TestContractCompleteness:
    def test_schema_parts(self) -> None:
        assert SCHEMA_PARTS == (
            "identity", "kind", "content", "provenance",
            "confidence", "lifecycle",
        )

    def test_twelve_kinds_six_and_six(self) -> None:
        memory = [r for r in FACT_KINDS if r["store"] == "memory"]
        db = [r for r in FACT_KINDS if r["store"] == "knowledge_db"]
        assert len(FACT_KINDS) == 12
        assert len(memory) == 6
        assert len(db) == 6
        assert {r["kind"] for r in memory} == {
            "personality_core", "user_preference", "user_fact",
            "episode", "response_pattern", "active_thread_state",
        }
        assert {r["record"] for r in db} == {
            "entities", "statements", "conceptnet_edges",
            "geonames_places", "dbpedia_entities", "reference_chunks",
        }

    def test_contract_matches_live_tree(self) -> None:
        assert contract_problems() == []

    def test_provenance_vocabularies(self) -> None:
        assert set(DB_PROVENANCE) == {
            "wikidata", "conceptnet", "geonames", "dbpedia", "reference",
        }
        assert "user_stated" in MEMORY_PROVENANCE
        assert "synthetic_fixture" in MEMORY_PROVENANCE
        assert not set(MEMORY_PROVENANCE) & set(DB_PROVENANCE)

    def test_confidence_defaults_as_constructed(self) -> None:
        by_kind = {r["kind"]: r for r in FACT_KINDS}
        assert by_kind["user_fact"]["confidence_field"] == ("confidence", 0.6)
        assert by_kind["user_preference"]["confidence_field"] == ("confidence", 0.5)
        assert by_kind["episode"]["confidence_field"] == ("confidence", 0.5)
        assert by_kind["response_pattern"]["confidence_field"] == ("confidence", 0.6)
        assert valid_fact().confidence == 0.6

    def test_doc_states_the_contract(self) -> None:
        text = CONTRACT_DOC.read_text(encoding="utf-8")
        assert "TF-042" in text
        assert "provenance" in text.lower()


# ── Memory record validation ─────────────────────────────────────────────────


class TestRecordProblems:
    def test_valid_records_of_every_memory_kind_pass(self) -> None:
        assert record_problems(valid_fact()) == []
        assert record_problems(UserPreferenceRecord(
            record_id="p1", category="tone", summary="prefers calm answers",
        )) == []
        assert record_problems(EpisodicMemoryRecord(
            record_id="e1", summary="the river ran high",
        )) == []
        assert record_problems(ResponsePatternRecord(
            record_id="pa1", pattern_shape=["acknowledge", "explain"],
        )) == []
        assert record_problems(PersonalityCoreRecord(
            record_id="pers_1", traits=["plain-spoken"],
        )) == []
        assert record_problems(ActiveThreadStateRecord(
            record_id="t1", topic="ferry times",
        )) == []

    def test_identity_required(self) -> None:
        problems = record_problems(valid_fact(record_id=""))
        assert any("record_id" in p for p in problems)
        problems = record_problems({"record_type": "user_fact", "summary": "x"})
        assert any("record_id" in p for p in problems)

    def test_fact_needs_content(self) -> None:
        problems = record_problems(valid_fact(summary="", fact_value={}))
        assert any("no content" in p for p in problems)
        # Either half alone is enough content.
        assert record_problems(valid_fact(fact_value={})) == []
        assert record_problems(valid_fact(summary="")) == []

    def test_preference_needs_category_and_summary(self) -> None:
        problems = record_problems(UserPreferenceRecord(
            record_id="p1", category="", summary="",
        ))
        assert any("category" in p for p in problems)
        assert any("summary" in p for p in problems)

    def test_pattern_needs_summary_or_shape(self) -> None:
        problems = record_problems(ResponsePatternRecord(record_id="pa1"))
        assert any("no content" in p for p in problems)
        assert record_problems(ResponsePatternRecord(
            record_id="pa1", summary="answer, then example",
        )) == []

    def test_confidence_range_enforced(self) -> None:
        for bad in (999.0, -0.1, 1.5):
            problems = record_problems(valid_fact(confidence=bad))
            assert any("confidence" in p and "[0, 1]" in p for p in problems)
        problems = record_problems(valid_fact(confidence=float("nan")))
        assert any("confidence" in p and "finite" in p for p in problems)
        problems = record_problems(valid_fact(confidence="high"))  # type: ignore[arg-type]
        assert any("confidence" in p and "not a number" in p for p in problems)
        problems = record_problems(valid_fact(importance=2.0))
        assert any("importance" in p for p in problems)
        # Boundary values are valid confidence.
        assert record_problems(valid_fact(confidence=0.0)) == []
        assert record_problems(valid_fact(confidence=1.0)) == []

    def test_summary_caps_enforced(self) -> None:
        problems = record_problems(UserPreferenceRecord(
            record_id="p1", category="tone", summary="y" * 161,
        ))
        assert any("160" in p for p in problems)
        problems = record_problems(valid_fact(summary="y" * 181))
        assert any("180" in p for p in problems)
        problems = record_problems(EpisodicMemoryRecord(
            record_id="e1", summary="y" * 241,
        ))
        assert any("240" in p for p in problems)
        assert record_problems(valid_fact(summary="y" * 180)) == []

    def test_record_type_must_match_kind(self) -> None:
        problems = record_problems(
            {"record_id": "f1", "record_type": "episode", "summary": "x"},
            kind="user_fact",
        )
        assert any("record_type" in p for p in problems)

    def test_closed_vocabularies_enforced(self) -> None:
        problems = record_problems(valid_fact(status="bogus"))  # type: ignore[arg-type]
        assert any("status" in p for p in problems)
        problems = record_problems(valid_fact(retention_class="forever"))  # type: ignore[arg-type]
        assert any("retention_class" in p for p in problems)

    def test_counters_and_timestamps(self) -> None:
        problems = record_problems(valid_fact(times_retrieved=-1))
        assert any("times_retrieved" in p for p in problems)
        problems = record_problems(valid_fact(created_at="not a time"))
        assert any("created_at" in p for p in problems)
        problems = record_problems(valid_fact(tags="ferry"))  # type: ignore[arg-type]
        assert any("tags" in p for p in problems)

    def test_dict_and_dataclass_forms_agree(self) -> None:
        record = valid_fact(confidence=1.5)
        from_dict_form = record_problems(record.to_dict())
        from_dataclass = record_problems(record)
        assert from_dict_form == from_dataclass
        assert from_dataclass != []

    def test_unknown_keys_are_not_problems(self) -> None:
        data = valid_fact().to_dict()
        data["future_field"] = {"anything": 1}
        assert record_problems(data) == []


# ── The store's ingestion gates ──────────────────────────────────────────────


class TestStoreGates:
    def test_invalid_append_refused_and_nothing_written(
        self, tmp_path: Path,
    ) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        with pytest.raises(ValueError, match="ingestion contract"):
            store.append_profile_record(valid_fact(confidence=999.0))
        assert store.load_profile_records() == []
        assert not store._profile_path.exists()

    def test_invalid_episode_and_pattern_refused(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        with pytest.raises(ValueError, match="summary"):
            store.append_episode(EpisodicMemoryRecord(record_id="e1", summary=""))
        with pytest.raises(ValueError, match="content"):
            store.append_pattern(ResponsePatternRecord(record_id="pa1"))
        assert store.load_episodes() == []
        assert store.load_patterns() == []

    def test_invalid_saves_refused(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        with pytest.raises(ValueError, match="priority"):
            store.save_thread_state(ActiveThreadStateRecord(
                record_id="t1", priority=1.5,
            ))
        with pytest.raises(ValueError, match="record_id"):
            store.save_personality_core(PersonalityCoreRecord(record_id=""))
        assert store.load_thread_state() is None
        assert store.load_personality_core() is None

    def test_valid_appends_still_work(self, tmp_path: Path) -> None:
        store = MemoryStore(memory_dir=tmp_path / "mem")
        store.append_profile_record(valid_fact())
        store.append_episode(EpisodicMemoryRecord(
            record_id="e1", summary="the river ran high",
        ))
        store.append_pattern(ResponsePatternRecord(
            record_id="pa1", pattern_shape=["answer"],
        ))
        assert [r.record_id for r in store.load_profile_records()] == ["fact_1"]
        assert [r.record_id for r in store.load_episodes()] == ["e1"]
        assert [r.record_id for r in store.load_patterns()] == ["pa1"]

    def test_legacy_invalid_record_loads_and_is_flagged(
        self, tmp_path: Path,
    ) -> None:
        # Load paths do NOT validate (TF-041 K13 tolerance): a
        # record written before the contract still loads, and the
        # validator flags it by name for audit.
        store = MemoryStore(memory_dir=tmp_path / "mem")
        legacy = valid_fact(record_id="legacy_1", confidence=999.0)
        with store._profile_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(legacy.to_dict()) + "\n")
        loaded = store.load_profile_records()
        assert [r.record_id for r in loaded] == ["legacy_1"]
        assert any("confidence" in p for p in record_problems(loaded[0]))


# ── Knowledge-DB rows ────────────────────────────────────────────────────────


class TestDbRowProblems:
    def test_valid_rows_pass(self) -> None:
        assert db_row_problems(
            {"qid": "Q1", "label_en": "Lantern Hollow", "source": "wikidata",
             "popularity_score": 0.9}, kind="entity",
        ) == []
        assert db_row_problems(
            {"subject_qid": "Q1", "property_pid": "P31",
             "object_value": "Q5", "source": "wikidata",
             "refs": '[{"stated_in": "Q1"}]'}, kind="statement",
        ) == []
        assert db_row_problems(
            {"concept1": "ferry", "relation": "UsedFor", "concept2": "crossing",
             "weight": 2.5, "source": "conceptnet"}, kind="conceptnet_edge",
        ) == []
        assert db_row_problems(
            {"geonames_id": 1, "name": "Lantern Hollow", "population": 40,
             "source": "geonames"}, kind="geonames_place",
        ) == []
        assert db_row_problems(
            {"uri": "http://dbpedia.org/resource/X", "label_en": "X",
             "source": "dbpedia"}, kind="dbpedia_entity",
        ) == []
        assert db_row_problems(
            {"source_file": "hollow.md", "chunk_index": 0,
             "content": "the ferry takes eleven minutes",
             "source": "reference"}, kind="reference_chunk",
        ) == []

    def test_provenance_vocabulary_enforced(self) -> None:
        problems = db_row_problems(
            {"qid": "Q1", "label_en": "X", "source": "mars",
             "popularity_score": 0.5}, kind="entity",
        )
        assert any("source" in p and "provenance" in p for p in problems)

    def test_weight_is_strength_not_confidence(self) -> None:
        # A ConceptNet weight above 1 is legitimate strength…
        assert db_row_problems(
            {"concept1": "a", "relation": "IsA", "concept2": "b",
             "weight": 12.0, "source": "conceptnet"}, kind="conceptnet_edge",
        ) == []
        # …but a negative weight is not a weight at all.
        problems = db_row_problems(
            {"concept1": "a", "relation": "IsA", "concept2": "b",
             "weight": -1.0, "source": "conceptnet"}, kind="conceptnet_edge",
        )
        assert any("weight" in p for p in problems)

    def test_identity_and_content_required(self) -> None:
        problems = db_row_problems(
            {"label_en": "X", "source": "wikidata"}, kind="entity",
        )
        assert any("qid" in p for p in problems)
        problems = db_row_problems(
            {"source_file": "", "chunk_index": 0, "content": "x"},
            kind="reference_chunk",
        )
        assert any("source_file" in p for p in problems)
        problems = db_row_problems(
            {"subject_qid": "Q1", "property_pid": "", "object_value": "Q5"},
            kind="statement",
        )
        assert any("property_pid" in p for p in problems)


# ── Writeback fact dicts ─────────────────────────────────────────────────────


class TestWritebackFactProblems:
    def test_valid_writeback_fact_passes(self) -> None:
        assert writeback_fact_problems({
            "summary": "the user lives by the River Sarn",
            "tags": ["river"],
            "confidence": 0.9,
            "source": "user_stated",
        }) == []
        assert writeback_fact_problems({
            "fact_value": {"crossing_minutes": 11},
            "source": "conversation_writeback",
        }) == []

    def test_provenance_stamp_required(self) -> None:
        problems = writeback_fact_problems({"summary": "unstamped fact"})
        assert any("provenance missing" in p for p in problems)
        problems = writeback_fact_problems({
            "summary": "badly stamped fact", "source": "a dream",
        })
        assert any("source" in p and "vocabulary" in p for p in problems)

    def test_trace_fixture_dict_is_not_a_fact(self) -> None:
        # TF-038's writeback trace fixture ({"fact": "ferry"}) is a
        # trace shape, not an ingestion candidate — the contract
        # says so by name instead of letting it pass silently.
        problems = writeback_fact_problems({"fact": "ferry"})
        assert any("no content" in p for p in problems)
        assert any("provenance missing" in p for p in problems)

    def test_writeback_confidence_range(self) -> None:
        problems = writeback_fact_problems({
            "summary": "overconfident", "source": "user_stated",
            "confidence": 7.0,
        })
        assert any("confidence" in p for p in problems)
