"""
TF-109 — Dedup & merge rules for overlapping sources
(consumes TF-050's contradiction rule).

The rules live as verifiable data in
``src/thoughtforge/etl/dedup_merge.py``
(``TABLE_IDENTITIES``, ``MERGE_RULES``, ``decide_merge``,
``MergeLedger``, ``apply_row_merge``,
``merge_conceptnet_edge``, ``find_overlaps``, checked by
``merge_problems()`` and ``merge_live_problems()``) and
are enforced in ``etl/wikidata.py`` and
``etl/sources.py``; the write-up is
``docs/architecture/DEDUP_AND_MERGE.md``. D-MERGE-1
(a thinner re-run erasing fields), D-MERGE-2 (ConceptNet
semantic duplicates + stronger assertions IGNOREd away),
D-MERGE-3 (silent overwrites) and D-MERGE-4 (cross-source
overlap invisible) were found by this slice's probes and
are FIXED here; D-DATA-2 is narrowed (store-level share).

Pinned here:

  * the filed identities + rules as data, and the pure
    judgement (duplicate / complementary union, with an
    empty incoming field never erasing / conflict with
    both values named),
  * the ledger (collisions only, by outcome + by table,
    an unknown outcome refused),
  * through the REAL builders: a wikidata entity re-run
    thinner keeps its description + aliases while its
    changed label lands ledgered; an identical re-run is
    a ledgered duplicate and rewrites nothing; ConceptNet
    semantic duplicates collapse to one row and the
    stronger weight stands; a GeoNames re-run with gaps
    fills instead of erasing; a DBpedia identical re-run
    duplicates and a changed abstract lands ledgered;
    reference chunks merge the same way,
  * cross-source overlap: the same synthetic town from
    three sources is one group naming all three tables,
    single-source names form no group, and finding an
    overlap changes no stored row,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented lantern
records in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic overlapping records
through the real builders on this VM). A real dump's
overlap rate is NOT PROVEN.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.dedup_merge import (
    MERGE_DOC,
    MERGE_FORMAT,
    MERGE_OUTCOMES,
    MERGE_RULES,
    TABLE_IDENTITIES,
    MergeLedger,
    decide_merge,
    find_overlaps,
    merge_as_data,
    merge_live_problems,
    merge_problems,
    normalise_name,
    overlap_as_data,
)
from thoughtforge.etl.sources import (
    ConceptNetETL,
    DBpediaETL,
    GeoNamesETL,
    ReferenceDataETL,
)
from thoughtforge.etl.wikidata import WikidataETL


def _entity(qid: str, label: str = "Lantern Town",
            desc: str | None = "A synthetic town") -> dict:
    return {
        "id": qid,
        "labels": {"en": {"value": label}},
        "descriptions": {"en": {"value": desc}} if desc else {},
        "aliases": {"en": [{"value": "the lit town"}]} if desc else
        {"en": []},
        "sitelinks": {"enwiki": {}, "dewiki": {}},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q109900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }


def _geonames_row(gid: str = "109001", name: str = "Lantern Town",
                  population: str = "100") -> str:
    parts = [gid, name, name, "", "1.5", "2.5", "P", "PPL", "XX", "",
             "A1", "A2", "", "", population, "10", "10", "UTC",
             "2026-01-01"]
    return "\t".join(parts)


def _rows(db: Path, sql: str) -> list:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


class TestMergeData:
    def test_format_and_doc(self) -> None:
        assert MERGE_FORMAT == "thoughtforge-dedup-merge/1"
        assert MERGE_DOC == "docs/architecture/DEDUP_AND_MERGE.md"
        assert Path(MERGE_DOC).exists()

    def test_rules_unique(self) -> None:
        ids = [r["id"] for r in MERGE_RULES]
        assert len(ids) == len(set(ids)) == 10

    def test_identities_cover_the_merged_tables(self) -> None:
        assert [e["table"] for e in TABLE_IDENTITIES] == [
            "entities", "labels", "statements", "conceptnet_edges",
            "geonames_places", "dbpedia_entities", "reference_chunks"]
        conceptnet = TABLE_IDENTITIES[3]
        assert conceptnet["semantic_key"] == (
            "concept1", "relation", "concept2", "language")
        statements = TABLE_IDENTITIES[2]
        assert statements["value_fields"] == ()

    def test_outcomes_vocabulary(self) -> None:
        assert MERGE_OUTCOMES == (
            "duplicate", "merged_complementary", "conflict_overwrite",
            "conflict_stronger_wins", "conflict_first_stands")

    def test_as_data_shape(self) -> None:
        data = merge_as_data()
        assert data["format"] == MERGE_FORMAT
        assert len(data["rules"]) == 10
        assert len(data["identities"]) == 7


class TestDecideMerge:
    def test_identical_is_duplicate(self) -> None:
        row = {"label": "Lantern", "desc": "A light"}
        outcome, merged, detail = decide_merge(
            row, dict(row), ("label", "desc"))
        assert outcome == "duplicate" and merged == row and detail == []

    def test_complementary_union_never_erases(self) -> None:
        outcome, merged, detail = decide_merge(
            {"label": "Lantern", "desc": "A light", "aka": None},
            {"label": "Lantern", "desc": None, "aka": "lamp"},
            ("label", "desc", "aka"))
        assert outcome == "merged_complementary"
        assert merged == {"label": "Lantern", "desc": "A light",
                          "aka": "lamp"}
        resolutions = {d["field"]: d["resolution"] for d in detail}
        assert resolutions == {"desc": "kept_stored_empty_incoming",
                               "aka": "filled"}

    def test_conflict_names_both_values(self) -> None:
        outcome, merged, detail = decide_merge(
            {"label": "Lantern Town"}, {"label": "Lantern City"},
            ("label",))
        assert outcome == "conflict_overwrite"
        assert merged["label"] == "Lantern City"
        assert detail == [{"field": "label", "stored": "Lantern Town",
                           "incoming": "Lantern City",
                           "resolution": "incoming_wins"}]

    def test_zero_is_a_value_not_a_gap(self) -> None:
        outcome, _, _ = decide_merge(
            {"population": 100}, {"population": 0}, ("population",))
        assert outcome == "conflict_overwrite"


class TestLedger:
    def test_counts_by_outcome_and_table(self) -> None:
        ledger = MergeLedger()
        ledger.record("entities", {"qid": "Q1"}, "duplicate", [])
        ledger.record("entities", {"qid": "Q2"}, "conflict_overwrite",
                      [{"field": "label_en", "stored": "a",
                        "incoming": "b", "resolution": "incoming_wins"}])
        data = ledger.as_data()
        assert data["collisions"] == 2
        assert data["by_outcome"] == {"duplicate": 1,
                                      "conflict_overwrite": 1}
        assert data["by_table"] == {"entities": 2}

    def test_unknown_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown merge outcome"):
            MergeLedger().record("entities", {}, " vanished", [])


class TestWikidataMerge:
    def test_thinner_rerun_keeps_fields_and_ledgers_conflict(
            self, tmp_path: Path) -> None:
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps([_entity("Q1")]), encoding="utf-8")
        etl = WikidataETL(tmp_path / "w.db")
        etl.build(dump_path=dump, resume=False)
        dump.write_text(json.dumps([
            _entity("Q1", label="Lantern Town Renamed", desc=None)]),
            encoding="utf-8")
        etl.build(dump_path=dump, resume=False)
        assert _rows(tmp_path / "w.db",
                     "SELECT label_en, description_en, aliases_en "
                     "FROM entities") == [
            ("Lantern Town Renamed", "A synthetic town", "the lit town")]
        outcomes = etl.merge_ledger.by_outcome()
        assert outcomes.get("conflict_overwrite", 0) >= 1
        conflict = [d for d in etl.merge_ledger.decisions
                    if d["outcome"] == "conflict_overwrite"
                    and d["table"] == "entities"]
        fields = {x["field"] for x in conflict[0]["detail"]}
        assert "label_en" in fields

    def test_identical_rerun_is_a_ledgered_duplicate(
            self, tmp_path: Path) -> None:
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps([_entity("Q1")]), encoding="utf-8")
        etl = WikidataETL(tmp_path / "w.db")
        etl.build(dump_path=dump, resume=False)
        etl.build(dump_path=dump, resume=False)
        assert _rows(tmp_path / "w.db",
                     "SELECT COUNT(*) FROM entities") == [(1,)]
        assert _rows(tmp_path / "w.db",
                     "SELECT COUNT(*) FROM statements") == [(1,)]
        outcomes = etl.merge_ledger.by_outcome()
        assert outcomes.get("duplicate", 0) >= 2  # entity + statement


class TestConceptNetMerge:
    def test_semantic_duplicates_collapse_stronger_stands(
            self, tmp_path: Path) -> None:
        csv_path = tmp_path / "a.csv"
        csv_path.write_text("\n".join([
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}",
            "/a/2\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 5.0}",
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 9.0}",
            "/a/3\t/r/UsedFor\t/c/en/lantern\t/c/en/light\t"
            "{\"weight\": 1.0}",
        ]) + "\n", encoding="utf-8")
        etl = ConceptNetETL(tmp_path / "c.db")
        etl.build(csv_path=csv_path)
        assert _rows(tmp_path / "c.db",
                     "SELECT uri, relation, weight FROM "
                     "conceptnet_edges ORDER BY relation") == [
            ("/a/1", "IsA", 9.0), ("/a/3", "UsedFor", 1.0)]
        outcomes = etl.merge_ledger.by_outcome()
        assert outcomes == {"conflict_stronger_wins": 2}

    def test_weaker_reassertion_stands_first_and_exact_dup(
            self, tmp_path: Path) -> None:
        csv_path = tmp_path / "a.csv"
        csv_path.write_text("\n".join([
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 5.0}",
            "/a/2\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 2.0}",
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t{\"weight\": 5.0}",
        ]) + "\n", encoding="utf-8")
        etl = ConceptNetETL(tmp_path / "c.db")
        etl.build(csv_path=csv_path)
        assert _rows(tmp_path / "c.db",
                     "SELECT uri, weight FROM conceptnet_edges") == [
            ("/a/1", 5.0)]
        outcomes = etl.merge_ledger.by_outcome()
        assert outcomes == {"conflict_first_stands": 1, "duplicate": 1}


class TestGeoNamesAndDbpediaMerge:
    def test_geonames_conflict_lands_and_is_ledgered(
            self, tmp_path: Path) -> None:
        txt_path = tmp_path / "p.txt"
        txt_path.write_text(_geonames_row() + "\n", encoding="utf-8")
        etl = GeoNamesETL(tmp_path / "g.db")
        etl.build(txt_path=txt_path)
        txt_path.write_text(
            _geonames_row(name="Lantern Town Renamed",
                          population="999") + "\n", encoding="utf-8")
        etl.build(txt_path=txt_path)
        assert _rows(tmp_path / "g.db",
                     "SELECT name, population FROM geonames_places") == [
            ("Lantern Town Renamed", 999)]
        assert etl.merge_ledger.by_outcome() == {"conflict_overwrite": 1}

    def test_dbpedia_duplicate_then_conflict(
            self, tmp_path: Path) -> None:
        json_path = tmp_path / "a.json"
        record = {"uri": "http://example.org/Lantern",
                  "label": "Lantern", "abstract": "First.",
                  "type": "Device"}
        json_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        etl = DBpediaETL(tmp_path / "d.db")
        etl.build(json_path=json_path)
        etl.build(json_path=json_path)
        assert etl.merge_ledger.by_outcome() == {"duplicate": 1}
        record["abstract"] = "Second."
        json_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        etl.build(json_path=json_path)
        assert _rows(tmp_path / "d.db",
                     "SELECT abstract_en FROM dbpedia_entities") == [
            ("Second.",)]
        assert etl.merge_ledger.by_outcome() == {"conflict_overwrite": 1}


class TestReferenceMerge:
    def test_reference_chunks_merge_by_identity(
            self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        (ref_dir / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        etl = ReferenceDataETL(tmp_path / "r.db")
        etl.build(ref_dir=ref_dir)
        etl.build(ref_dir=ref_dir)
        assert etl.merge_ledger.by_outcome().get("duplicate", 0) >= 1
        assert _rows(tmp_path / "r.db",
                     "SELECT COUNT(*) FROM reference_chunks") == [(1,)]


class TestFindOverlaps:
    def _three_source_store(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "x.db"
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps([_entity("Q1")]), encoding="utf-8")
        WikidataETL(db_path).build(dump_path=dump, resume=False)
        txt_path = tmp_path / "p.txt"
        txt_path.write_text(_geonames_row() + "\n", encoding="utf-8")
        GeoNamesETL(db_path).build(txt_path=txt_path)
        json_path = tmp_path / "a.json"
        json_path.write_text(json.dumps({
            "uri": "http://example.org/Lantern_Town",
            "label": "Lantern Town", "abstract": "A synthetic town.",
            "type": "Place"}) + "\n", encoding="utf-8")
        DBpediaETL(db_path).build(json_path=json_path)
        return db_path

    def test_normalise_name(self) -> None:
        assert normalise_name("Lantern_Town") == "lantern town"
        assert normalise_name("  Lantern—Town! ") == "lantern town"
        assert normalise_name(None) == ""

    def test_three_sources_one_group(self, tmp_path: Path) -> None:
        db_path = self._three_source_store(tmp_path)
        groups = find_overlaps(db_path)
        assert len(groups) == 1
        group = groups[0]
        assert group["name"] == "lantern town"
        assert {m["table"] for m in group["members"]} == {
            "entities", "geonames_places", "dbpedia_entities"}
        data = overlap_as_data(db_path)
        assert data["group_count"] == 1 and data["member_count"] == 3

    def test_single_source_names_form_no_group(
            self, tmp_path: Path) -> None:
        db_path = tmp_path / "g.db"
        txt_path = tmp_path / "p.txt"
        txt_path.write_text(_geonames_row() + "\n", encoding="utf-8")
        GeoNamesETL(db_path).build(txt_path=txt_path)
        assert find_overlaps(db_path) == []

    def test_finding_an_overlap_changes_no_row(
            self, tmp_path: Path) -> None:
        db_path = self._three_source_store(tmp_path)
        before = {t: _rows(db_path, f"SELECT COUNT(*) FROM {t}")
                  for t in ("entities", "geonames_places",
                            "dbpedia_entities")}
        find_overlaps(db_path)
        after = {t: _rows(db_path, f"SELECT COUNT(*) FROM {t}")
                 for t in before}
        assert before == after


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert merge_problems() == []

    def test_live_checker_clean(self) -> None:
        assert merge_live_problems() == []
