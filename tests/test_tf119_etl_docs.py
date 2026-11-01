"""
TF-119 — ETL docs + a new-source recipe.

The proof lives as verifiable data in
``src/thoughtforge/etl/etl_docs.py`` (``DOC_INDEX``,
``RECIPE_STEPS``, ``RECIPE_RULES``,
``recipe_application()``, ``example_recipe_application()``,
checked by ``etl_docs_problems()``);
``forge_memory.py``'s ``build_all`` gained its DBpedia
step in the open (D-ETL-1 FIXED), TF-105's map was
updated in the open with it, and ``etl/sources.py``'s
stale "40 built-in files" count was replaced by
TF-007's measured census (D-DATA-1 FIXED). The recipe's
step 1 — file a source in TF-106's registry before
documenting or building it — discharges D-ETL-2's
TF-119 share (D-ETL-2 FIXED, both shares discharged;
the builders themselves still consult nothing, a
standing shape stated, not claimed fixed).

Pinned here:

  * the filing as data: format, doc, rules, index,
  * DOC_INDEX completeness: exactly the Part VI docs
    TF-105..TF-119, every path on disk,
  * the recipe: 12 steps in filed order, step 1 the
    registry, every step bound to a real module /
    attribute / file that exists,
  * the worked example: the fictional lantern
    gazetteer is caught unfiled, registry step first,
    and files no real entry anywhere; a fully filed
    real source judges complete,
  * D-ETL-1 live: ``build_all`` through the real CLI
    lands a synthetic DBpedia file's rows (step 6/7),
    and reports the step skipped without the file,
  * D-DATA-1: no stale count in sources.py, the
    measured census stated,
  * the checkers are clean, TF-105's included.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no real dump at scale, never claimed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from thoughtforge.etl.sources import ReferenceDataETL
from thoughtforge.etl.etl_docs import (
    DOC_INDEX,
    ETL_DOCS_DOC,
    ETL_DOCS_FORMAT,
    EXAMPLE_SOURCE,
    RECIPE_RULES,
    RECIPE_STEPS,
    etl_docs_as_data,
    etl_docs_problems,
    example_recipe_application,
    recipe_application,
)
from thoughtforge.etl.etl_map import (
    ETL_RULES,
    ORCHESTRATION,
    SOURCE_BUILDERS,
    etl_map_problems,
)
from thoughtforge.etl.source_registry import SOURCES

REPO_ROOT = Path(__file__).parent.parent

PART_VI_DOCS = (
    "ETL_MAPPED.md", "SOURCE_REGISTRY.md", "WIKIDATA_PIPELINE.md",
    "SCHEMA_VALIDATION_GATE.md", "DEDUP_AND_MERGE.md",
    "SUBSET_CRAFT.md", "EMBEDDING_AT_INGESTION.md",
    "INGESTION_DRY_RUN.md", "CANON_INGESTION.md",
    "UPDATE_INGESTION.md", "INGESTION_METRICS.md",
    "POISONED_SOURCE_FIXTURES.md", "OFFLINE_INGESTION.md",
    "REFERENCE_LIBRARY_AS_SOURCE.md",
    "ETL_DOCS_AND_NEW_SOURCE_RECIPE.md",
)


_forge_memory_mod = None


def _load_forge_memory():
    """Load forge_memory LAZILY (TF-057's pattern):
    importing it configures logging, so it must never
    load during collection."""
    global _forge_memory_mod
    if _forge_memory_mod is None:
        spec = importlib.util.spec_from_file_location(
            "forge_memory_tf119", REPO_ROOT / "forge_memory.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _forge_memory_mod = mod
    return _forge_memory_mod


def _ref_lib(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "lantern_care.md").write_text(
        "# Lantern Care\n\nKeep the lantern trimmed and filled.\n",
        encoding="utf-8")
    return root


def _dbpedia_file(root: Path) -> Path:
    path = root / "abstracts.json"
    path.write_text("\n".join((
        json.dumps({"uri": "http://example.org/Lantern",
                    "label": "Lantern",
                    "abstract": "A portable synthetic light.",
                    "type": "Device"}),
        json.dumps({"uri": "http://example.org/Tide_Bell",
                    "label": "Tide Bell",
                    "abstract": "A synthetic bell rung at dusk.",
                    "type": "Device"}),
    )) + "\n", encoding="utf-8")
    return path


def _run_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
             db: Path, *extra: str):
    """Run the real ``all`` command with the reference
    step filed to a synthetic tmp library: build_all
    calls the builder with no directory, so a subclass
    whose default is the synthetic library is patched
    into the loaded instrument in its place (TF-017:
    the real built-in library is never a test input)."""
    lib = _ref_lib(tmp_path / "ref")

    class _SyntheticRefETL(ReferenceDataETL):
        def build(self, ref_dir=None, **kwargs):  # type: ignore[override]
            return super().build(ref_dir=ref_dir or lib, **kwargs)

    mod = _load_forge_memory()
    monkeypatch.setattr(mod, "ReferenceDataETL", _SyntheticRefETL)
    return CliRunner().invoke(mod.cli, [
        "--db", str(db), "all", "--skip-embeddings", *extra])


def _count(db: Path, table: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert ETL_DOCS_FORMAT == "thoughtforge-etl-docs/1"
        assert ETL_DOCS_DOC == \
            "docs/architecture/ETL_DOCS_AND_NEW_SOURCE_RECIPE.md"
        assert Path(ETL_DOCS_DOC).exists()
        ids = [r["id"] for r in RECIPE_RULES]
        assert len(ids) == len(set(ids)) == 9
        data = etl_docs_as_data()
        assert data["format"] == ETL_DOCS_FORMAT
        assert data["doc"] == ETL_DOCS_DOC
        assert [r["id"] for r in data["rules"]] == ids
        assert len(data["doc_index"]) == len(DOC_INDEX)
        assert len(data["recipe_steps"]) == len(RECIPE_STEPS)


# ── DOC_INDEX completeness ──────────────────────────────────────────────────


class TestDocIndex:
    def test_index_is_exactly_part_vi_in_order(self) -> None:
        assert [e["slice"] for e in DOC_INDEX] == [
            f"TF-{n}" for n in range(105, 120)]
        assert [Path(e["file"]).name for e in DOC_INDEX] == \
            list(PART_VI_DOCS)
        for entry in DOC_INDEX:
            assert entry["subject"]
            assert (REPO_ROOT / entry["file"]).exists(), entry["file"]

    def test_every_part_vi_module_doc_is_indexed(self) -> None:
        indexed = {e["file"] for e in DOC_INDEX}
        for module_name in (
                "etl_map", "source_registry", "wikidata_pipeline",
                "schema_gate", "dedup_merge", "embedding_ingestion",
                "ingestion_dry_run", "canon_ingestion",
                "update_ingestion", "ingestion_metrics",
                "poisoned_source", "offline_ingestion",
                "reference_ingestion"):
            module = importlib.import_module(
                f"thoughtforge.etl.{module_name}")
            docs = [v for k, v in vars(module).items()
                    if k.endswith("_DOC") and isinstance(v, str)
                    and v.startswith("docs/architecture/")]
            for doc in docs:
                assert doc in indexed, (module_name, doc)


# ── The recipe ──────────────────────────────────────────────────────────────


class TestRecipe:
    def test_twelve_steps_in_filed_order(self) -> None:
        assert len(RECIPE_STEPS) == 12
        assert [s["order"] for s in RECIPE_STEPS] == list(range(1, 13))
        ids = [s["id"] for s in RECIPE_STEPS]
        assert len(ids) == len(set(ids))
        assert ids[0] == "file_in_registry_first"
        assert ids[-1] == "synthetic_fixtures_only"

    def test_every_step_binds_a_real_thing(self) -> None:
        for step in RECIPE_STEPS:
            module = importlib.import_module(step["bind_module"])
            assert hasattr(module, step["bind_attribute"]), step["id"]
            assert (REPO_ROOT / step["bind_file"]).exists(), step["id"]
            text = (REPO_ROOT / step["bind_file"]).read_text(
                encoding="utf-8")
            for marker in step["bind_file_markers"]:
                assert marker in text, (step["id"], marker)

    def test_step_one_is_the_registry_gate(self) -> None:
        first = RECIPE_STEPS[0]
        assert first["bind_module"] == \
            "thoughtforge.etl.source_registry"
        assert first["bind_attribute"] == "check_source_use"
        assert "BEFORE" in first["step"]


# ── The worked example ──────────────────────────────────────────────────────


class TestExample:
    def test_unfiled_source_is_caught_registry_first(self) -> None:
        app = example_recipe_application()
        assert app["complete"] is False
        assert app["source"] == "lantern_gazetteer"
        assert app["missing"][0] == "file_in_registry_first"
        assert set(app["missing"]) == {
            "file_in_registry_first",
            "file_record_schema_in_gate",
            "name_target_table_and_identity",
            "name_landed_text_fields_for_poison",
            "write_doc_and_register_in_index"}
        assert any("registry" in p for p in app["problems"])
        assert len(app["problems"]) == len(app["missing"])

    def test_example_files_no_real_entry(self) -> None:
        assert EXAMPLE_SOURCE["id"] not in {
            e["id"] for e in SOURCES}
        assert EXAMPLE_SOURCE["id"] not in {
            b["id"] for b in SOURCE_BUILDERS}
        assert EXAMPLE_SOURCE["doc_file"] not in {
            e["file"] for e in DOC_INDEX}

    def test_a_fully_filed_source_judges_complete(self) -> None:
        app = recipe_application(
            "dbpedia", "dbpedia_entities",
            "docs/architecture/ETL_MAPPED.md")
        assert app["complete"] is True
        assert app["missing"] == []
        assert app["problems"] == []


# ── D-ETL-1: build_all wires DBpedia ────────────────────────────────────────


class TestBuildAllDbpedia:
    def test_build_all_source_order(self) -> None:
        text = (REPO_ROOT / "forge_memory.py").read_text(encoding="utf-8")
        body = text.split("def build_all", 1)[1].split("\ndef ", 1)[0]
        marks = ["initialize_schema", "ReferenceDataETL", "WikidataETL",
                 "ConceptNetETL", "GeoNamesETL", "DBpediaETL",
                 "build_entity_embeddings"]
        positions = [body.find(m) for m in marks]
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)
        assert "--dbpedia-json" in text
        assert "Step 6/7" in body and "Step 7/7" in body

    def test_all_lands_dbpedia_rows_live(
            self, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        db = tmp_path / "store.db"
        json_path = _dbpedia_file(tmp_path)
        result = _run_all(tmp_path, monkeypatch, db,
                           "--dbpedia-json", str(json_path))
        assert result.exit_code == 0, result.output
        assert "Step 6/7: Ingesting DBpedia" in result.output
        assert "Step 7/7: Embeddings — skipped" in result.output
        assert _count(db, "dbpedia_entities") == 2
        assert _count(db, "reference_chunks") >= 1

    def test_all_reports_dbpedia_skipped_without_file(
            self, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        db = tmp_path / "store.db"
        result = _run_all(tmp_path, monkeypatch, db)
        assert result.exit_code == 0, result.output
        assert "Step 6/7: DBpedia — skipped" in result.output
        assert "--dbpedia-json" in result.output
        assert _count(db, "dbpedia_entities") == 0


# ── D-DATA-1: the measured census ───────────────────────────────────────────


class TestStaleCount:
    def test_no_stale_count_and_census_stated(self) -> None:
        text = (REPO_ROOT / "src/thoughtforge/etl/sources.py").read_text(
            encoding="utf-8")
        assert "40 built-in" not in text
        assert "40 files" not in text
        assert text.count("215 root files + 17 SRD chapters") >= 2


# ── Neighbours + checkers ───────────────────────────────────────────────────


class TestCheckers:
    def test_etl_docs_checker_clean(self) -> None:
        assert etl_docs_problems() == []

    def test_tf105_map_checker_still_clean(self) -> None:
        assert etl_map_problems() == []
        assert ORCHESTRATION["steps"][-2] == \
            "6 dbpedia (only with --dbpedia-json)"
        assert len(ORCHESTRATION["steps"]) == 7
        assert not any("dbpedia" in gap.lower()
                       and "never" in gap.lower()
                       for gap in ORCHESTRATION["gaps"])
        assert "build_all_ingests_dbpedia_when_passed" in {
            r["id"] for r in ETL_RULES}
        assert "build_all_never_ingests_dbpedia" not in {
            r["id"] for r in ETL_RULES}
        dbpedia = [b for b in SOURCE_BUILDERS
                   if b["id"] == "dbpedia"][0]
        assert "NOT wired" not in dbpedia["notes"]
        assert "TF-119" in dbpedia["notes"]
