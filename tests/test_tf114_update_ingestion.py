"""
TF-114 — Update ingestion: re-running a source updates
facts by the lifecycle rules instead of duplicating
them.

The rules live as verifiable data in
``src/thoughtforge/etl/update_ingestion.py``
(``UPDATE_RULES``, ``UpdateLedger``,
``retire_unpresented_statements``,
``retire_unpresented_chunks``, checked by
``update_problems()`` and ``update_live_problems()``)
and are enforced in ``etl/wikidata.py`` (statement
supersede by presented set) and ``etl/sources.py``
(reference chunk retirement); TF-109's
``dedup_merge.py`` was updated in the open — its
within-table rule now states the lifecycle supersede.
D-UPDATE-1 (changed statements appended beside the
old), D-UPDATE-2 (disappeared chunks staying stale)
and D-UPDATE-3 (no lifecycle statement, no update
record) were found by this slice's probes and are
FIXED here.

Pinned here:

  * the rules + outcomes as data, TF-109's replaced
    placeholder, and the ledger (unknown outcome
    refused, counts by outcome + table),
  * through the REAL wikidata builder: a changed
    statement supersedes (the old tuple leaves, the
    ledger names the superseding value), a removed
    claim retires, a multi-valued fact presented whole
    survives whole, an identical re-run retires
    nothing, an entity presented with no claims loses
    its statements, an absent entity is kept with its
    statements, and a completed-checkpoint re-run
    retires nothing,
  * through the REAL reference builder: a shrunk file
    retires its disappeared chunk (and the chunk's
    embedding row, and its FTS presence), a file
    emptied retires all its chunks, a deleted file
    keeps its chunks by the stated rule,
  * the row-table road under the lifecycle statement:
    a changed GeoNames/DBpedia fact supersedes in
    place ledgered, and an absent unit is kept,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented lantern
records in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic re-runs through the
real builders on this VM). A real dump re-run is
NOT PROVEN.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.dedup_merge import MERGE_RULES
from thoughtforge.etl.sources import (
    DBpediaETL,
    GeoNamesETL,
    ReferenceDataETL,
)
from thoughtforge.etl.update_ingestion import (
    UPDATE_DOC,
    UPDATE_FORMAT,
    UPDATE_OUTCOMES,
    UPDATE_RULES,
    UpdateLedger,
    update_as_data,
    update_live_problems,
    update_problems,
)
from thoughtforge.etl.wikidata import WikidataETL


def _entity(qid: str, label: str = "Lantern Town",
            claims: dict | None = None) -> dict:
    if claims is None:
        claims = {"P31": ["Q111"], "P17": ["Q900"]}
    return {
        "id": qid,
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": "A synthetic town"}},
        "aliases": {"en": [{"value": "the lit town"}]},
        "sitelinks": {"enwiki": {}, "dewiki": {}},
        "claims": {pid: [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": obj}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        } for obj in objs] for pid, objs in claims.items()},
    }


def _rows(db: Path, sql: str) -> list:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _statements(db: Path) -> list:
    return _rows(db, "SELECT subject_qid, property_pid, object_value "
                     "FROM statements ORDER BY 1, 2, 3")


def _geonames_row(gid: str = "109001", name: str = "Lantern Town",
                  population: str = "100") -> str:
    parts = [gid, name, name, "", "1.5", "2.5", "P", "PPL", "XX", "",
             "A1", "A2", "", "", population, "10", "10", "UTC",
             "2026-01-01"]
    return "\t".join(parts)


class TestUpdateData:
    def test_format_and_doc(self) -> None:
        assert UPDATE_FORMAT == "thoughtforge-update-ingestion/1"
        assert UPDATE_DOC == "docs/architecture/UPDATE_INGESTION.md"
        assert Path(UPDATE_DOC).exists()

    def test_rules_unique_and_complete(self) -> None:
        ids = [r["id"] for r in UPDATE_RULES]
        assert len(ids) == len(set(ids)) == 10
        for needed in ("unchanged_fact_is_a_duplicate_not_rewritten",
                       "changed_fact_within_a_source_supersedes",
                       "statements_supersede_by_presented_set",
                       "a_unit_absent_from_the_rerun_is_kept"):
            assert needed in ids

    def test_outcomes_vocabulary(self) -> None:
        assert UPDATE_OUTCOMES == (
            "statement_superseded", "statement_retired", "chunk_retired")

    def test_tf109_placeholder_replaced_in_the_open(self) -> None:
        rule = [r for r in MERGE_RULES
                if r["id"] == "within_a_table_the_incoming_row_wins"]
        assert len(rule) == 1
        assert "TF-114" in rule[0]["rule"]
        assert "supersede" in rule[0]["rule"]
        assert "owns replacing this default" not in rule[0]["rule"]

    def test_as_data_shape(self) -> None:
        data = update_as_data()
        assert data["format"] == UPDATE_FORMAT
        assert len(data["rules"]) == 10
        assert data["outcomes"] == list(UPDATE_OUTCOMES)


class TestUpdateLedger:
    def test_counts_by_outcome_and_table(self) -> None:
        ledger = UpdateLedger()
        ledger.record("statements", {"subject_qid": "Q1"},
                      "statement_superseded", [])
        ledger.record("statements", {"subject_qid": "Q1"},
                      "statement_retired", [])
        ledger.record("reference_chunks", {"source_file": "a.md"},
                      "chunk_retired", [])
        data = ledger.as_data()
        assert data["updates"] == 3
        assert data["by_outcome"] == {
            "statement_superseded": 1, "statement_retired": 1,
            "chunk_retired": 1}
        assert data["by_table"] == {"statements": 2,
                                    "reference_chunks": 1}

    def test_unknown_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown update outcome"):
            UpdateLedger().record("statements", {}, " vanished", [])


class TestWikidataUpdate:
    def _build(self, tmp_path: Path, entities: list, name: str = "d.json"):
        dump = tmp_path / name
        dump.write_text(json.dumps(entities), encoding="utf-8")
        etl = WikidataETL(tmp_path / "w.db")
        etl.build(dump_path=dump, resume=False)
        return etl

    def test_changed_statement_supersedes_removed_retires(
            self, tmp_path: Path) -> None:
        self._build(tmp_path, [_entity("Q1")])
        etl = self._build(tmp_path, [
            _entity("Q1", claims={"P31": ["Q999"]})])
        assert _statements(tmp_path / "w.db") == [("Q1", "P31", "Q999")]
        assert etl.update_ledger.by_outcome() == {
            "statement_superseded": 1, "statement_retired": 1}
        decision = [d for d in etl.update_ledger.decisions
                    if d["outcome"] == "statement_superseded"][0]
        assert decision["key"]["object_value"] == "Q111"
        assert decision["detail"][0]["incoming"] == "Q999"

    def test_identical_rerun_retires_nothing(
            self, tmp_path: Path) -> None:
        self._build(tmp_path, [_entity("Q1")])
        etl = self._build(tmp_path, [_entity("Q1")])
        assert _statements(tmp_path / "w.db") == [
            ("Q1", "P17", "Q900"), ("Q1", "P31", "Q111")]
        assert etl.update_ledger.updates == 0

    def test_multi_valued_fact_presented_whole_survives(
            self, tmp_path: Path) -> None:
        both = {"P31": ["Q111", "Q112"]}
        self._build(tmp_path, [_entity("Q1", claims=both)])
        etl = self._build(tmp_path, [_entity("Q1", claims=both)])
        assert _statements(tmp_path / "w.db") == [
            ("Q1", "P31", "Q111"), ("Q1", "P31", "Q112")]
        assert etl.update_ledger.updates == 0

    def test_entity_presented_with_no_claims_loses_statements(
            self, tmp_path: Path) -> None:
        self._build(tmp_path, [_entity("Q1")])
        etl = self._build(tmp_path, [_entity("Q1", claims={})])
        assert _statements(tmp_path / "w.db") == []
        assert etl.update_ledger.by_outcome() == {"statement_retired": 2}

    def test_absent_entity_is_kept_with_its_statements(
            self, tmp_path: Path) -> None:
        self._build(tmp_path, [_entity("Q1"),
                               _entity("Q2", label="Other Town",
                                       claims={"P31": ["Q222"]})])
        etl = self._build(tmp_path, [
            _entity("Q1", claims={"P31": ["Q999"]})])
        assert _statements(tmp_path / "w.db") == [
            ("Q1", "P31", "Q999"), ("Q2", "P31", "Q222")]
        assert _rows(tmp_path / "w.db",
                     "SELECT qid FROM entities ORDER BY 1") == [
            ("Q1",), ("Q2",)]

    def test_completed_checkpoint_rerun_retires_nothing(
            self, tmp_path: Path) -> None:
        dump = tmp_path / "d.json"
        dump.write_text(json.dumps([_entity("Q1")]), encoding="utf-8")
        etl = WikidataETL(tmp_path / "w.db")
        etl.build(dump_path=dump)  # resume defaults on
        etl.build(dump_path=dump)  # completed checkpoint returns
        assert _statements(tmp_path / "w.db") == [
            ("Q1", "P17", "Q900"), ("Q1", "P31", "Q111")]
        assert etl.update_ledger.updates == 0


class TestReferenceUpdate:
    def _ref(self, tmp_path: Path) -> Path:
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir(exist_ok=True)
        return ref_dir

    def test_shrunk_file_retires_disappeared_chunk(
            self, tmp_path: Path) -> None:
        ref_dir = self._ref(tmp_path)
        doc = ref_dir / "doc.md"
        doc.write_text("# Part One\n\nAlpha text here.\n\n"
                       "# Part Two\n\nBeta text here.\n",
                       encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        doc.write_text("# Part One\n\nAlpha text changed.\n",
                       encoding="utf-8")
        etl = ReferenceDataETL(db)
        etl.build(ref_dir=ref_dir)
        assert _rows(db, "SELECT source_file, chunk_index, content "
                         "FROM reference_chunks") == [
            ("doc.md", 0, "Alpha text changed.")]
        assert etl.update_ledger.by_outcome() == {"chunk_retired": 1}
        # The retired chunk is gone from FTS search too.
        assert _rows(db, "SELECT chunk_id FROM reference_fts "
                         "WHERE reference_fts MATCH 'Beta'") == []

    def test_retired_chunk_loses_its_embedding_row(
            self, tmp_path: Path) -> None:
        ref_dir = self._ref(tmp_path)
        doc = ref_dir / "doc.md"
        doc.write_text("# Part One\n\nAlpha text here.\n\n"
                       "# Part Two\n\nBeta text here.\n",
                       encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        conn = sqlite3.connect(str(db))
        try:
            ids = [r[0] for r in conn.execute(
                "SELECT chunk_id, chunk_index FROM reference_chunks "
                "ORDER BY 2")]
            for chunk_id in ids:
                conn.execute(
                    "INSERT INTO reference_embeddings (chunk_id, "
                    "embedding, model_name) VALUES (?, ?, ?)",
                    (chunk_id, b"x", "fake"))
            conn.commit()
        finally:
            conn.close()
        doc.write_text("# Part One\n\nAlpha text changed.\n",
                       encoding="utf-8")
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        remaining = _rows(db, "SELECT chunk_id FROM reference_embeddings")
        assert remaining == [(ids[0],)]

    def test_emptied_file_retires_all_its_chunks(
            self, tmp_path: Path) -> None:
        ref_dir = self._ref(tmp_path)
        doc = ref_dir / "doc.md"
        doc.write_text("# Part One\n\nAlpha text here.\n",
                       encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        doc.write_text("", encoding="utf-8")
        etl = ReferenceDataETL(db)
        etl.build(ref_dir=ref_dir)
        assert _rows(db, "SELECT COUNT(*) FROM reference_chunks") == [(0,)]
        assert etl.update_ledger.by_outcome() == {"chunk_retired": 1}

    def test_deleted_file_keeps_its_chunks_by_rule(
            self, tmp_path: Path) -> None:
        ref_dir = self._ref(tmp_path)
        doc = ref_dir / "doc.md"
        doc.write_text("# Part One\n\nAlpha text here.\n",
                       encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        doc.unlink()
        (ref_dir / "new.md").write_text("# New\n\nNew text.\n",
                                        encoding="utf-8")
        ReferenceDataETL(db).build(ref_dir=ref_dir)
        assert _rows(db, "SELECT source_file FROM reference_chunks "
                         "ORDER BY 1") == [("doc.md",), ("new.md",)]


class TestRowTableLifecycle:
    def test_geonames_change_supersedes_absent_kept(
            self, tmp_path: Path) -> None:
        txt_path = tmp_path / "p.txt"
        txt_path.write_text(_geonames_row() + "\n" + _geonames_row(
            gid="109002", name="Beta Town") + "\n", encoding="utf-8")
        db = tmp_path / "g.db"
        GeoNamesETL(db).build(txt_path=txt_path)
        txt_path.write_text(_geonames_row(population="999") + "\n",
                            encoding="utf-8")
        etl = GeoNamesETL(db)
        etl.build(txt_path=txt_path)
        assert _rows(db, "SELECT geonames_id, population FROM "
                         "geonames_places ORDER BY 1") == [
            (109001, 999), (109002, 100)]
        assert etl.merge_ledger.by_outcome() == {"conflict_overwrite": 1}

    def test_dbpedia_change_supersedes_absent_kept(
            self, tmp_path: Path) -> None:
        json_path = tmp_path / "a.json"
        a = {"uri": "http://example.org/A", "label": "A",
             "abstract": "First.", "type": "T"}
        b = {"uri": "http://example.org/B", "label": "B",
             "abstract": "Bee.", "type": "T"}
        json_path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n",
                             encoding="utf-8")
        db = tmp_path / "d.db"
        DBpediaETL(db).build(json_path=json_path)
        a["abstract"] = "Second."
        json_path.write_text(json.dumps(a) + "\n", encoding="utf-8")
        etl = DBpediaETL(db)
        etl.build(json_path=json_path)
        assert _rows(db, "SELECT uri, abstract_en FROM dbpedia_entities "
                         "ORDER BY 1") == [
            ("http://example.org/A", "Second."),
            ("http://example.org/B", "Bee.")]
        assert etl.merge_ledger.by_outcome() == {"conflict_overwrite": 1}


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert update_problems() == []

    def test_live_checker_clean(self) -> None:
        assert update_live_problems() == []
