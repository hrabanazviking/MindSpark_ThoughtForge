"""
TF-105 — ETL mapped: sources -> schema -> embeddings ->
subset -> store, drawn from code; the pipeline's contracts
per stage.

Part VI's first slice. The map lives as verifiable data in
``src/thoughtforge/etl/etl_map.py`` (``PIPELINE_STAGES`` /
``SOURCE_BUILDERS`` / ``SCHEMA_FACTS`` / ``EMBEDDING_FACTS``
/ ``SUBSET_FACTS`` / ``ORCHESTRATION`` / ``ETL_RULES``,
checked by ``etl_map_problems()`` and
``etl_map_live_problems()``), the write-up is
``docs/architecture/ETL_MAPPED.md``, and D-SUBSET-1 is
FIXED here: the subset's reference-chunk copy ordered by a
column the schema has never carried, so every limited
build copied zero chunks and still reported success.

Pinned here:

  * the map is complete and true: 5 stages in roadmap
    order, 5 builders with their real batch defaults,
    write modes, stats shapes, logging and durability
    postures, 10 tables + 3 FTS + 15 named indexes, and
    the live checker finds no drift,
  * a whole synthetic pipeline runs end to end through
    the REAL builders: one wikidata entity (with its
    statement, label, and the pipeline's only ingestion
    log row), one conceptnet edge (the French edge skipped
    by the language filter), one geonames place, one
    dbpedia entry (the uri-less line skipped), one
    reference chunk (the .jpg skipped by rule and
    counted since TF-118),
  * embeddings are built only for rows missing one —
    through a fake model in the real cache, 1 + 1 built
    and a re-run builds 0,
  * a LIMITED (pi_zero) subset keeps its reference chunk
    (the D-SUBSET-1 regression), copies no embedding or
    label rows, and recreates none of the named indexes,
  * every builder raises FileNotFoundError on a missing
    input, and D-ETL-1's fix is pinned (TF-119 updated
    this map in the open: the orchestrator's all command
    now ingests a dbpedia file when one is passed, as
    step 6 of 7).

All fixtures are synthetic (TF-017): tiny stand-in inputs
in a temporary directory, never the real canon tree.

Proof grade: FIXTURE (synthetic inputs through the real
builders on this VM; the embedding model is a fake —
sentence-transformers is absent here, D-ENV-1). An
at-scale ingestion of any real dump is NOT PROVEN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.etl.etl_map import (
    EMBEDDING_FACTS,
    ETL_DOC,
    ETL_FORMAT,
    ETL_RULES,
    ORCHESTRATION,
    PIPELINE_STAGES,
    SCHEMA_FACTS,
    SOURCE_BUILDERS,
    SUBSET_FACTS,
    etl_as_data,
    etl_map_live_problems,
    etl_map_problems,
    run_synthetic_pipeline,
)
from thoughtforge.etl.sources import (
    ConceptNetETL,
    DBpediaETL,
    GeoNamesETL,
    ReferenceDataETL,
)
from thoughtforge.etl.wikidata import WikidataETL


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("tf105-pipeline")
    return run_synthetic_pipeline(root)


class TestMapData:
    def test_format_and_doc(self) -> None:
        assert ETL_FORMAT == "thoughtforge-etl-map/1"
        assert ETL_DOC == "docs/architecture/ETL_MAPPED.md"
        assert Path(ETL_DOC).exists()

    def test_stages_in_roadmap_order(self) -> None:
        assert [s["id"] for s in PIPELINE_STAGES] == [
            "sources", "schema", "embeddings", "subset", "store"]
        for stage in PIPELINE_STAGES:
            assert stage["modules"] and stage["consumes"]
            assert stage["produces"] and stage["contract"]

    def test_five_builders_complete(self) -> None:
        assert [b["id"] for b in SOURCE_BUILDERS] == [
            "wikidata", "conceptnet", "geonames", "dbpedia", "reference"]
        for builder in SOURCE_BUILDERS:
            assert builder["class"] and builder["module"]
            assert builder["tables"] and builder["stats_keys"]
            assert builder["write_mode"] and builder["synchronous"]

    def test_only_wikidata_logs(self) -> None:
        assert [b["id"] for b in SOURCE_BUILDERS if b["logs_ingestion"]] \
            == ["wikidata"]

    def test_schema_facts(self) -> None:
        assert len(SCHEMA_FACTS["tables"]) == 10
        assert len(SCHEMA_FACTS["fts_tables"]) == 3
        assert SCHEMA_FACTS["named_indexes"] == 15
        assert len(SCHEMA_FACTS["counted_by_get_entity_count"]) == 8

    def test_embedding_facts(self) -> None:
        assert EMBEDDING_FACTS["default_model"] == "all-MiniLM-L6-v2"
        assert EMBEDDING_FACTS["dimension"] == 384
        assert EMBEDDING_FACTS["only_missing"] is True

    def test_subset_facts(self) -> None:
        assert SUBSET_FACTS["limits"]["pi_zero"] == 50000
        assert SUBSET_FACTS["limits"]["server_gpu"] == 0
        assert SUBSET_FACTS["copies_rows"] == (
            "entities", "statements", "reference_chunks")
        assert len(SUBSET_FACTS["does_not_copy_rows"]) == 7

    def test_orchestration_filed(self) -> None:
        assert ORCHESTRATION["in_package_orchestrator"] is False
        assert len(ORCHESTRATION["steps"]) == 7
        assert len(ORCHESTRATION["gaps"]) == 2

    def test_rules_unique(self) -> None:
        ids = [r["id"] for r in ETL_RULES]
        assert len(ids) == len(set(ids)) == 9

    def test_as_data_shape(self) -> None:
        data = etl_as_data()
        assert data["format"] == ETL_FORMAT
        assert len(data["stages"]) == 5
        assert len(data["builders"]) == 5
        assert len(data["rules"]) == 9


class TestStaticChecker:
    def test_no_problems(self) -> None:
        assert etl_map_problems() == []


class TestMissingInputs:
    def test_wikidata_missing_dump(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            WikidataETL(tmp_path / "db.sqlite").build(
                dump_path=tmp_path / "absent.json")

    def test_conceptnet_missing_csv(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ConceptNetETL(tmp_path / "db.sqlite").build(
                csv_path=tmp_path / "absent.csv")

    def test_geonames_missing_txt(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            GeoNamesETL(tmp_path / "db.sqlite").build(
                txt_path=tmp_path / "absent.txt")

    def test_dbpedia_missing_json(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DBpediaETL(tmp_path / "db.sqlite").build(
                json_path=tmp_path / "absent.json")

    def test_reference_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ReferenceDataETL(tmp_path / "db.sqlite").build(
                ref_dir=tmp_path / "absent-dir")


class TestLivePipeline:
    def test_live_checker_clean(self) -> None:
        assert etl_map_live_problems() == []

    def test_wikidata_leg(self, pipeline: dict) -> None:
        assert pipeline["wikidata"] == {
            "entities": 1, "statements": 1, "labels": 1, "skipped": 0}

    def test_conceptnet_language_filter(self, pipeline: dict) -> None:
        assert pipeline["conceptnet"] == {"inserted": 1, "skipped": 1}

    def test_geonames_leg(self, pipeline: dict) -> None:
        assert pipeline["geonames"] == {"inserted": 1, "skipped": 0}

    def test_dbpedia_leg(self, pipeline: dict) -> None:
        assert pipeline["dbpedia"] == {"inserted": 1, "skipped": 1}

    def test_reference_leg_unsupported_counted_by_rule(
            self, pipeline: dict) -> None:
        # TF-118 re-pin: the .jpg in the synthetic dir is
        # skipped BY RULE and counted in skipped (it was
        # passed over uncounted before TF-118 fixed
        # D-DATA-4's ingestion share): files 1, skipped 1.
        assert pipeline["reference"] == {
            "files": 1, "chunks": 1, "skipped": 1}

    def test_store_counts(self, pipeline: dict) -> None:
        counts = pipeline["store_counts"]
        for table in ("entities", "statements", "labels", "embeddings",
                      "conceptnet_edges", "geonames_places",
                      "dbpedia_entities", "reference_chunks"):
            assert counts[table] == 1, table

    def test_only_wikidata_logged(self, pipeline: dict) -> None:
        assert pipeline["ingestion_log"] == [("wikidata", "complete")]

    def test_embeddings_only_missing(self, pipeline: dict) -> None:
        assert pipeline["entity_embeddings"] == 1
        assert pipeline["reference_embeddings"] == 1
        assert pipeline["entity_embeddings_rerun"] == 0


class TestSubsetRegression:
    def test_limited_subset_keeps_reference_chunks(
            self, pipeline: dict) -> None:
        # D-SUBSET-1: before the fix this was 0 with success.
        assert pipeline["subset"]["reference_chunks"] == 1
        assert pipeline["subset"]["success"] is True
        assert not any("reference_chunks skipped" in note
                       for note in pipeline["subset"]["notes"])

    def test_subset_copies_entities_and_statements(
            self, pipeline: dict) -> None:
        assert pipeline["subset"]["entities"] == 1
        assert pipeline["subset"]["statements"] == 1

    def test_subset_copies_no_embeddings_or_labels(
            self, pipeline: dict) -> None:
        assert pipeline["subset_counts"]["embeddings"] == 0
        assert pipeline["subset_counts"]["labels"] == 0

    def test_subset_recreates_no_named_indexes(
            self, pipeline: dict) -> None:
        assert pipeline["subset_named_indexes"] == []
