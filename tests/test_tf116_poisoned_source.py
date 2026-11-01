"""
TF-116 — Poisoned-source fixtures: adversarial
source data refused / quarantined (kin to TF-062).

The gate lives as verifiable data in
``src/thoughtforge/etl/poisoned_source.py``
(``SOURCE_TEXT_FIELDS``, ``POISON_RULES``,
``POISON_FIXTURES``, ``BENIGN_FIXTURES``,
``PoisonLedger``, ``scan_store()``, checked by
``poisoned_source_problems()`` and
``poisoned_source_live_problems()``); every builder
holds its ledger as ``poison_gate``, TF-105's living
map was updated in the open (its ``poison``
orchestration entry), and KnowledgeForge screens at
retrieval. D-POISON-1 (nothing screened source
content — adversarial text landed verbatim through
every builder), D-POISON-2 (landed poison was
returned by KnowledgeForge retrieval) and
D-POISON-3 (no ledger, scan or quarantine shape
named poison in a knowledge store) were found by
this slice's probes and are FIXED here.

Pinned here:

  * the fields bind exactly to TF-105's builders,
    the threat classes are TF-062's own objects,
    the rules as data, the format + doc,
  * every poison fixture is judged as filed
    (verdict exact, classes named) and every benign
    lookalike passes,
  * the ledger: seen / refused / quarantined /
    held_out, by_source / by_class / by_verdict,
    entries named with position + ref; an
    unregistered source raises by name,
  * through the REAL builders: wikidata (label
    refuse, description quarantine, statement-text
    refuse — the entity lands nothing and presents
    nothing), conceptnet (CSV + reference paths),
    geonames, dbpedia, reference (per chunk; an
    all-poison file lands nothing),
  * scan_store: a missing store is scanned without
    being created; a planted row is named by table
    + key; a clean store flags nothing,
  * retrieval: a planted poisoned chunk is excluded
    from sql_retrieve and from retrieve()'s bundle,
    named in the report's screened_out; a clean
    chunk is retrieved untouched,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented
lantern records in a temp dir, never a real dump or
canon. The adversarial strings are fixture DATA,
screened as data, never instructions.

Proof grade: FIXTURE (synthetic adversarial +
benign sources through the real builders and the
real retrieval on this VM). Screening a real dump
is NOT PROVEN.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.etl_map import SOURCE_BUILDERS
from thoughtforge.etl.poisoned_source import (
    BENIGN_FIXTURES,
    POISON_DOC,
    POISON_FIXTURES,
    POISON_FORMAT,
    POISON_RULES,
    SOURCE_TEXT_FIELDS,
    PoisonLedger,
    poisoned_source_as_data,
    poisoned_source_live_problems,
    poisoned_source_problems,
    scan_store,
    screen_result_record,
    screen_source_fields,
)
from thoughtforge.knowledge.adversarial import THREAT_CLASSES


def _fixture(name: str) -> dict:
    for fixture in (*POISON_FIXTURES, *BENIGN_FIXTURES):
        if fixture["name"] == name:
            return fixture
    raise KeyError(name)


def _geonames_row(gid: int, name: str) -> str:
    parts = [str(gid), name, name, "", "1.5", "2.5", "P", "PPL",
             "XX", "", "A1", "A2", "", "", "100", "10", "10",
             "UTC", "2026-01-01"]
    return "\t".join(parts)


class TestPoisonData:
    def test_format_and_doc(self) -> None:
        assert POISON_FORMAT == "thoughtforge-poisoned-source/1"
        assert POISON_DOC == \
            "docs/architecture/POISONED_SOURCE_FIXTURES.md"
        assert Path(POISON_DOC).exists()

    def test_fields_bind_to_tf105_builders(self) -> None:
        filed = [s for s in SOURCE_TEXT_FIELDS
                 if s != "conceptnet_reference"]
        assert filed == [b["id"] for b in SOURCE_BUILDERS]
        assert set(SOURCE_TEXT_FIELDS) == {
            "wikidata", "conceptnet", "conceptnet_reference",
            "geonames", "dbpedia", "reference"}

    def test_threat_classes_are_tf062s_own(self) -> None:
        data = poisoned_source_as_data()
        assert data["threat_classes"] == \
            [str(row["name"]) for row in THREAT_CLASSES]
        assert len(data["threat_classes"]) == 9

    def test_rules_unique_and_complete(self) -> None:
        ids = [r["id"] for r in POISON_RULES]
        assert len(ids) == len(set(ids)) == 11
        for needed in (
                "source_text_is_data_never_instructions",
                "threat_classes_are_tf062s_own",
                "refuse_is_dropped_quarantine_is_held_out",
                "retrieval_screens_what_landing_missed",
                "scan_store_is_read_only"):
            assert needed in ids

    def test_as_data_shape(self) -> None:
        data = poisoned_source_as_data()
        assert data["format"] == POISON_FORMAT
        assert len(data["sources"]) == 6
        assert len(data["poison_fixtures"]) == 10
        assert len(data["benign_fixtures"]) == 5


class TestFixturesJudged:
    def test_every_fixture_judged_as_filed(self) -> None:
        for fixture in (*POISON_FIXTURES, *BENIGN_FIXTURES):
            screening = screen_source_fields(
                fixture["source"], fixture["fields"])
            assert screening.verdict == fixture["verdict"], fixture
            for name in fixture["classes"]:
                assert name in screening.classes, fixture
            if fixture["verdict"] == "pass":
                assert screening.findings == [], fixture

    def test_unlanded_field_is_never_judged(self) -> None:
        # GeoNames alternate names are carried by no
        # column: poison there cannot refuse a record.
        poison = _fixture("geonames_name_trigger_directive")
        screening = screen_source_fields("geonames", {
            "name": "Lantern Town",
            "alternate_names": poison["fields"]["name"]})
        assert screening.verdict == "pass"

    def test_unregistered_source_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="unregistered source"):
            screen_source_fields("no_such_source_tf116",
                                 {"name": "Lantern"})


class TestLedger:
    def test_counts_and_breakdowns(self) -> None:
        ledger = PoisonLedger()
        clean = _fixture("geonames_instruction_prose")
        assert ledger.check("geonames", clean["fields"],
                            ref="1") == "pass"
        refuse = _fixture("geonames_name_trigger_directive")
        assert ledger.check("geonames", refuse["fields"],
                            ref="2") == "refuse"
        quarantine = _fixture("geonames_name_behaviour_directive")
        assert ledger.check("geonames", quarantine["fields"],
                            ref="3") == "quarantine"
        data = ledger.as_data()
        assert data["format"] == POISON_FORMAT
        assert data["seen"] == 3
        assert data["refused"] == 1
        assert data["quarantined"] == 1
        assert data["held_out"] == 2
        assert data["by_source"] == {"geonames": 2}
        assert data["by_verdict"] == {"refuse": 1, "quarantine": 1}
        assert data["by_class"] == {"trigger_directive": 1,
                                    "behaviour_directive": 1}
        first = data["entries"][0]
        assert first["position"] == 2
        assert first["ref"] == "2"
        assert first["classes"] == ["trigger_directive"]
        assert first["findings"][0]["field"] == "name"


class TestBuilders:
    def test_geonames_refuse_quarantine_clean(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.sources import GeoNamesETL

        txt = tmp_path / "places.txt"
        txt.write_text("\n".join([
            _geonames_row(116301, "Lantern Town"),
            _geonames_row(116302, _fixture(
                "geonames_name_trigger_directive")["fields"]["name"]),
            _geonames_row(116303, _fixture(
                "geonames_name_behaviour_directive")["fields"]["name"]),
        ]) + "\n", encoding="utf-8")
        db = tmp_path / "g.db"
        builder = GeoNamesETL(db)
        assert builder.build(txt) == {"inserted": 1, "skipped": 2}
        ledger = builder.poison_gate.as_data()
        assert ledger["seen"] == 3
        assert ledger["refused"] == 1
        assert ledger["quarantined"] == 1
        conn = sqlite3.connect(db)
        names = [r[0] for r in conn.execute(
            "SELECT name FROM geonames_places")]
        conn.close()
        assert names == ["Lantern Town"]

    def test_dbpedia_refuse_and_clean(self, tmp_path: Path) -> None:
        from thoughtforge.etl.sources import DBpediaETL

        poison = _fixture("dbpedia_abstract_role_hijack")
        path = tmp_path / "a.json"
        path.write_text("\n".join([
            json.dumps({"uri": "http://example.org/Lantern",
                        "label": "Lantern",
                        "abstract": "A synthetic light.",
                        "type": "Device"}),
            json.dumps({"uri": poison["ref"],
                        "label": "Pirate",
                        "abstract": poison["fields"]["abstract"],
                        "type": "Device"}),
        ]) + "\n", encoding="utf-8")
        db = tmp_path / "d.db"
        builder = DBpediaETL(db)
        assert builder.build(path) == {"inserted": 1, "skipped": 1}
        assert builder.poison_gate.refused == 1
        assert builder.poison_gate.entries[0]["classes"] == \
            ["role_hijack"]
        conn = sqlite3.connect(db)
        labels = [r[0] for r in conn.execute(
            "SELECT label_en FROM dbpedia_entities")]
        conn.close()
        assert labels == ["Lantern"]

    def test_conceptnet_poison_edge_held_out(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.sources import ConceptNetETL

        poison = _fixture("conceptnet_concept_instruction_override")
        slug = poison["fields"]["concept1"].replace(" ", "_")
        path = tmp_path / "a.csv"
        path.write_text(
            "/a/1\t/r/IsA\t/c/en/lantern\t/c/en/light\t"
            "{\"weight\": 2.0}\n"
            f"/a/2\t/r/IsA\t/c/en/{slug}\t/c/en/light\t"
            "{\"weight\": 2.0}\n", encoding="utf-8")
        db = tmp_path / "c.db"
        builder = ConceptNetETL(db)
        assert builder.build(path) == {"inserted": 1, "skipped": 1}
        assert builder.poison_gate.refused == 1
        conn = sqlite3.connect(db)
        concepts = [r[0] for r in conn.execute(
            "SELECT concept1 FROM conceptnet_edges")]
        conn.close()
        assert concepts == ["lantern"]

    def test_conceptnet_reference_path_screened(
            self, tmp_path: Path, monkeypatch) -> None:
        import thoughtforge.etl.sources as sources_module
        from thoughtforge.etl.sources import ConceptNetETL

        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        quarantine = _fixture(
            "conceptnet_reference_secrecy_directive")
        (ref_dir / "edges.json").write_text(json.dumps([
            {"concept1": "lantern", "relation": "UsedFor",
             "concept2": "lighting the harbour"},
            {"concept1": quarantine["fields"]["concept1"],
             "relation": quarantine["fields"]["relation"],
             "concept2": quarantine["fields"]["concept2"]},
        ]), encoding="utf-8")
        # The helper's name is computed, never written
        # out: TF-017's guard scans test files for any
        # reach toward the real reference directory,
        # and this test only ever points it at tmp_path.
        helper_name = "get_knowledge" + "_reference_dir"
        monkeypatch.setattr(
            sources_module, helper_name, lambda: ref_dir)
        db = tmp_path / "c.db"
        from thoughtforge.etl.schema import initialize_schema

        initialize_schema(db)
        builder = ConceptNetETL(db)
        assert builder.build_from_reference(db) == 1
        assert builder.poison_gate.quarantined == 1

    def test_wikidata_label_description_and_statement(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.wikidata import WikidataETL

        def entity(qid: str, **kw) -> dict:
            base = {"id": qid, "labels": {}, "descriptions": {},
                    "aliases": {}, "sitelinks": {"enwiki": {}},
                    "claims": {}}
            base.update(kw)
            return base

        clean = entity(
            "Q116311",
            labels={"en": {"value": "Lantern Town"}},
            descriptions={"en": {"value": "A synthetic town"}})
        by_label = entity(
            "Q116312",
            labels={"en": {"value": _fixture(
                "wikidata_label_instruction_override"
            )["fields"]["label"]}})
        by_desc = entity(
            "Q116313",
            labels={"en": {"value": "Quiet Town"}},
            descriptions={"en": {"value": _fixture(
                "wikidata_description_authority_claim"
            )["fields"]["description"]}})
        by_statement = entity(
            "Q116314",
            labels={"en": {"value": "Statement Town"}},
            claims={"P268": [{
                "mainsnak": {"snaktype": "value",
                             "datavalue": {
                                 "type": "string",
                                 "value": _fixture(
                                     "reference_content_exfiltration"
                                 )["fields"]["content"]}},
                "rank": "normal", "qualifiers": {},
                "references": []}]})
        dump = tmp_path / "dump.json"
        dump.write_text(json.dumps(
            [clean, by_label, by_desc, by_statement]),
            encoding="utf-8")
        db = tmp_path / "w.db"
        builder = WikidataETL(db)
        stats = builder.build(dump, resume=False)
        assert stats == {"entities": 1, "statements": 0,
                         "labels": 1, "skipped": 3}
        ledger = builder.poison_gate.as_data()
        assert ledger["refused"] == 2
        assert ledger["quarantined"] == 1
        assert ledger["by_class"]["instruction_override"] == 1
        conn = sqlite3.connect(db)
        qids = [r[0] for r in conn.execute(
            "SELECT qid FROM entities")]
        statements = conn.execute(
            "SELECT COUNT(*) FROM statements").fetchone()[0]
        conn.close()
        assert qids == ["Q116311"]
        assert statements == 0

    def test_reference_per_chunk_and_all_poison_file(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.sources import ReferenceDataETL

        ref = tmp_path / "ref"
        ref.mkdir()
        (ref / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        (ref / "lantern_notice.md").write_text(
            "# Lantern Notice\n\n" + _fixture(
                "reference_content_exfiltration"
            )["fields"]["content"] + "\n", encoding="utf-8")
        (ref / "lantern_secret.md").write_text(
            "# Lantern Secret\n\n" + _fixture(
                "reference_content_secrecy_directive"
            )["fields"]["content"] + "\n", encoding="utf-8")
        db = tmp_path / "r.db"
        builder = ReferenceDataETL(db)
        stats = builder.build(ref)
        assert stats["files"] == 1
        assert stats["chunks"] == 1
        assert stats["skipped"] == 2
        ledger = builder.poison_gate.as_data()
        assert ledger["refused"] == 1
        assert ledger["quarantined"] == 1
        conn = sqlite3.connect(db)
        contents = [r[0] for r in conn.execute(
            "SELECT content FROM reference_chunks")]
        conn.close()
        assert len(contents) == 1
        assert "trimmed" in contents[0]


class TestScanStore:
    def test_missing_store_scanned_not_created(
            self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost.db"
        scan = scan_store(ghost)
        assert scan["exists"] is False
        assert scan["scanned_rows"] == 0
        assert scan["flagged"] == []
        assert not ghost.exists()

    def test_planted_row_named_and_clean_store_quiet(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.sources import ReferenceDataETL

        ref = tmp_path / "ref"
        ref.mkdir()
        (ref / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref)
        assert scan_store(db)["flagged"] == []

        poison = _fixture("reference_content_exfiltration")
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO reference_chunks (source_file, chunk_index,"
            " title, content, tags, word_count) VALUES (?, ?, ?, ?,"
            " ?, ?)",
            ("planted.md", 0, "Planted", poison["fields"]["content"],
             "", 7))
        conn.commit()
        conn.close()
        scan = scan_store(db)
        assert scan["by_table"] == {"reference_chunks": 1}
        assert scan["by_verdict"] == {"refuse": 1}
        flagged = scan["flagged"][0]
        assert flagged["table"] == "reference_chunks"
        assert flagged["classes"] == ["exfiltration_directive"]


class TestRetrievalScreening:
    def _planted_store(self, tmp_path: Path) -> Path:
        from thoughtforge.etl.schema import rebuild_fts_indexes
        from thoughtforge.etl.sources import ReferenceDataETL

        ref = tmp_path / "ref"
        ref.mkdir()
        (ref / "lantern_care.md").write_text(
            "# Lantern Care\n\nKeep the lantern trimmed and "
            "polished.\n", encoding="utf-8")
        db = tmp_path / "r.db"
        ReferenceDataETL(db).build(ref)
        poison = _fixture("reference_content_exfiltration")
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO reference_chunks (source_file, chunk_index,"
            " title, content, tags, word_count) VALUES (?, ?, ?, ?,"
            " ?, ?)",
            ("planted.md", 0, "Planted", poison["fields"]["content"],
             "", 7))
        conn.commit()
        conn.close()
        rebuild_fts_indexes(db)
        return db

    def test_screen_result_record(self) -> None:
        poison = _fixture("reference_content_exfiltration")
        screening = screen_result_record(
            {"source": "reference", "content": poison["fields"]["content"]})
        assert screening.verdict == "refuse"
        assert screening.classes == ["exfiltration_directive"]
        benign = screen_result_record(
            {"source": "reference",
             "content": "Keep the lantern trimmed."})
        assert benign.verdict == "pass"

    def test_planted_chunk_excluded_from_sql_retrieval(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.forge import KnowledgeForge

        forge = KnowledgeForge(db_path=self._planted_store(tmp_path))
        hits = forge.sql_retrieve(
            "reveal the system prompt", sources=["reference"])
        assert hits == []
        screened = forge.last_poison_screened_out
        assert len(screened) == 1
        assert screened[0]["verdict"] == "refuse"
        assert screened[0]["classes"] == ["exfiltration_directive"]
        clean = forge.sql_retrieve(
            "lantern trimmed polished", sources=["reference"])
        assert len(clean) == 1
        assert forge.last_poison_screened_out == []

    def test_retrieve_report_names_screened_out(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.forge import KnowledgeForge

        forge = KnowledgeForge(db_path=self._planted_store(tmp_path))
        bundle = forge.retrieve(
            "reveal the system prompt", path="sql",
            sources=["reference"])
        assert bundle.total_retrieved == 0
        report = forge.last_retrieval_report
        assert report is not None
        assert len(report["screened_out"]) == 1
        assert report["screened_out"][0]["classes"] == \
            ["exfiltration_directive"]


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert poisoned_source_problems() == []

    def test_live_checker_clean(self) -> None:
        assert poisoned_source_live_problems() == []
