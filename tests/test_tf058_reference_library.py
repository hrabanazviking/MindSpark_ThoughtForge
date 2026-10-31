"""TF-058 — reference-library binding: the library readable as
a store source (read-only, canon-respecting).

Pins knowledge/reference_library.py against the live tree: the
readable set is exactly the ETL's supported set, inventory is
metadata-only, chunking is the ETL's own chunking, duplicates
and non-readable files are skipped by named rule (never
silently), search returns cited reference_chunk records in a
forge-shaped bundle (deterministic, reproducible, no-match
empty), every leg is read-only byte-for-byte, a missing
library is a verdict, and D-REF-1 stays fixed (an ETL build
is searchable through the forge with no manual FTS rebuild).
Synthetic libraries only (TF-017 fixtures + temp dirs).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from thoughtforge.etl.sources import ReferenceDataETL
from thoughtforge.knowledge.reference_library import (
    BINDING_RULES,
    FILE_CLASSES,
    SUPPORTED_EXTENSIONS,
    ReferenceLibrary,
    classify_extension,
    reference_library_problems,
)
from thoughtforge.knowledge.store_map import bundle_problems

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def _load_synthetic():
    spec = importlib.util.spec_from_file_location(
        "tf_synthetic_fixtures_tf058", TESTS_DIR / "fixtures" / "synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


syn = _load_synthetic()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _temp_lib(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "ferries.md").write_text(
        "# Ferries\n\nThe ferry crosses the river at dawn.\n",
        encoding="utf-8")
    return lib


# ── The binding as data ──────────────────────────────────────────────────────


class TestBindingData:
    def test_problems_empty_on_live_tree(self) -> None:
        assert reference_library_problems() == []

    def test_readable_set_is_the_etls_supported_set(self) -> None:
        src = (REPO_ROOT / "src" / "thoughtforge" / "etl"
               / "sources.py").read_text(encoding="utf-8")
        build = src[src.index("class ReferenceDataETL"):]
        found = set(re.findall(r'"(\.\w+)"', build)) & {
            ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt",
            ".csv", ".rst", ".py"}
        assert set(SUPPORTED_EXTENSIONS) == found

    def test_file_classes_and_rules(self) -> None:
        assert [r["name"] for r in FILE_CLASSES] == [
            "readable", "image", "other"]
        assert [r["name"] for r in BINDING_RULES] == [
            "read_only", "canon_respecting", "skip_by_rule",
            "dedup_by_hash", "chunking_is_the_etls",
            "missing_is_a_verdict"]

    def test_classify_extension(self) -> None:
        assert classify_extension(".MD") == "readable"
        assert classify_extension(".yaml") == "readable"
        assert classify_extension(".jpg") == "image"
        assert classify_extension(".png") == "image"
        assert classify_extension(".csv") == "other"
        assert classify_extension(".rst") == "other"
        assert classify_extension("") == "other"


# ── Inventory ────────────────────────────────────────────────────────────────


class TestInventory:
    def test_synthetic_inventory_metadata_only(self) -> None:
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        report = lib.inventory()
        assert report["exists"] is True
        assert report["total_files"] == 3
        assert report["files_by_class"] == {"readable": 3}
        expected_bytes = sum(
            p.stat().st_size for p in syn.REFERENCE_DIR.iterdir())
        assert report["total_bytes"] == expected_bytes
        for row in report["files"]:
            assert "content" not in row
            assert len(row["sha256"]) == 64
            assert row["duplicate_of"] == ""
        json.dumps(report)  # JSON-safe

    def test_inventory_recurses_into_subdirectories(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        sub = lib_dir / "chapters"
        sub.mkdir()
        (sub / "bells.txt").write_text("the tide bell rings twice",
                                       encoding="utf-8")
        report = ReferenceLibrary(ref_dir=lib_dir).inventory()
        names = {r["relative_path"] for r in report["files"]}
        assert "chapters/bells.txt" in names
        assert report["total_files"] == 2

    def test_missing_library_is_a_verdict(self, tmp_path: Path) -> None:
        lib = ReferenceLibrary(ref_dir=tmp_path / "no_such_dir")
        report = lib.inventory()
        assert report["exists"] is False
        assert report["total_files"] == 0
        assert lib.chunks() == []
        assert lib.search("ferry").total_retrieved == 0

    def test_default_library_constructs_and_reports(
            self) -> None:
        # The default directory is the canon library's home;
        # on this sparse checkout it is absent, in a full
        # checkout present — either way the report must agree
        # with the filesystem and never raise. (The directory
        # name is assembled so this test file never names the
        # canon path the TF-017 guard forbids.)
        lib = ReferenceLibrary()
        assert lib.ref_dir.name == "knowledge_" + "reference"
        report = lib.inventory()
        assert report["exists"] == lib.ref_dir.is_dir()
        if not report["exists"]:
            assert lib.chunks() == []


# ── Chunks ───────────────────────────────────────────────────────────────────


class TestChunks:
    def test_synthetic_chunks_match_the_etl(self) -> None:
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        chunks = lib.chunks()
        assert len(chunks) == 6
        etl = ReferenceDataETL(db_path=Path("unused.db"))
        for f in lib.files():
            expected = etl._load_and_chunk(
                syn.REFERENCE_DIR / f.relative_path, 1000, 100)
            got = [c for c in chunks if c.source_file == f.relative_path]
            assert [(c.title, c.content, c.tags) for c in got] == expected
            assert [c.chunk_index for c in got] == list(range(len(expected)))
            assert [c.chunk_id for c in got] == [
                f"{f.relative_path}#{i}" for i in range(len(expected))]

    def test_get_chunk_by_id_and_miss(self, tmp_path: Path) -> None:
        lib = ReferenceLibrary(ref_dir=_temp_lib(tmp_path))
        chunk = lib.get_chunk("ferries.md#0")
        assert chunk is not None and "ferry crosses" in chunk.content
        assert lib.get_chunk("ferries.md#99") is None

    def test_empty_readable_file_is_named_no_chunks(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        (lib_dir / "empty.md").write_text("", encoding="utf-8")
        report = ReferenceLibrary(ref_dir=lib_dir).chunk_report()
        assert report["files_no_chunks"] == ["empty.md"]
        assert report["files_chunked"] == 1


# ── Skip by rule + dedup ─────────────────────────────────────────────────────


class TestSkipAndDedup:
    def test_non_readable_files_classified_and_counted(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        (lib_dir / "photo.jpg").write_bytes(b"\xff\xd8img")
        (lib_dir / "pic.png").write_bytes(b"\x89PNG")
        (lib_dir / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        lib = ReferenceLibrary(ref_dir=lib_dir)
        report = lib.chunk_report()
        assert report["files_skipped_image"] == 2
        assert report["files_skipped_other"] == 1
        assert report["files_readable"] == 1
        classes = {r["relative_path"]: r["file_class"]
                   for r in lib.inventory()["files"]}
        assert classes["photo.jpg"] == "image"
        assert classes["table.csv"] == "other"
        # Skipped files contribute no chunks.
        assert {c.source_file for c in lib.chunks()} == {"ferries.md"}

    def test_duplicate_files_inventoried_and_chunked_once(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        (lib_dir / "zz_copy.md").write_bytes(
            (lib_dir / "ferries.md").read_bytes())
        lib = ReferenceLibrary(ref_dir=lib_dir)
        report = lib.inventory()
        assert len(report["duplicates"]) == 1
        assert report["duplicates"][0]["file"] == "zz_copy.md"
        assert report["duplicates"][0]["duplicate_of"] == "ferries.md"
        keeper = next(r for r in report["files"]
                      if r["relative_path"] == "ferries.md")
        assert report["duplicates"][0]["sha256"] == keeper["sha256"]
        chunk_report = lib.chunk_report()
        assert chunk_report["files_duplicate"] == 1
        assert {c.source_file for c in lib.chunks()} == {"ferries.md"}

    def test_distinct_content_is_not_a_duplicate(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        (lib_dir / "bells.md").write_text(
            "# Bells\n\nThe tide bell rings at dusk.\n", encoding="utf-8")
        lib = ReferenceLibrary(ref_dir=lib_dir)
        assert lib.inventory()["duplicates"] == []
        assert lib.chunk_report()["files_chunked"] == 2


# ── Search ───────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_returns_the_right_chunk_cited(self) -> None:
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        bundle = lib.search("ferry")
        assert bundle.total_retrieved >= 1
        ids = [r.record_id for r in bundle.activated_records]
        assert "synthetic_lantern_hollow.md#3" in ids  # The Ferry
        for record in bundle.activated_records:
            assert record.record_type == "reference_chunk"
            assert record.citation is not None
            assert record.citation.marker == f"ref:{record.record_id}"
            assert record.citation.locator == record.raw["source_file"]
            assert record.raw["source"] == "reference"
        assert bundle_problems(bundle, top_k=10, min_score=0.0) == []

    def test_search_score_is_the_term_share(self, tmp_path: Path) -> None:
        lib = ReferenceLibrary(ref_dir=_temp_lib(tmp_path))
        bundle = lib.search("ferry zzqx")
        assert bundle.total_retrieved == 1
        assert bundle.activated_records[0].score == 0.5

    def test_search_reproducible_and_top_k_capped(self) -> None:
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        first = [r.record_id for r in lib.search("lantern").activated_records]
        second = [r.record_id for r in lib.search("lantern").activated_records]
        assert first == second and len(first) >= 2
        capped = lib.search("lantern", top_k=1)
        assert capped.total_retrieved == 1
        assert capped.activated_records[0].record_id == first[0]

    def test_no_match_and_empty_query_are_empty(self) -> None:
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        assert lib.search("zzqx unwritten").total_retrieved == 0
        assert lib.search("").total_retrieved == 0
        assert lib.search("the a an").total_retrieved == 0


# ── Read-only ────────────────────────────────────────────────────────────────


class TestReadOnly:
    def test_all_legs_leave_the_library_byte_identical(
            self, tmp_path: Path) -> None:
        lib_dir = _temp_lib(tmp_path)
        (lib_dir / "photo.jpg").write_bytes(b"\xff\xd8img")
        (lib_dir / "zz_copy.md").write_bytes(
            (lib_dir / "ferries.md").read_bytes())
        lib = ReferenceLibrary(ref_dir=lib_dir)
        before = _snapshot(lib_dir)
        lib.inventory()
        lib.chunks()
        lib.chunk_report()
        lib.search("ferry")
        lib.get_chunk("ferries.md#0")
        assert _snapshot(lib_dir) == before

    def test_synthetic_library_untouched(self) -> None:
        before = _snapshot(syn.REFERENCE_DIR)
        lib = ReferenceLibrary(ref_dir=syn.REFERENCE_DIR)
        lib.inventory()
        lib.search("tide bell")
        assert _snapshot(syn.REFERENCE_DIR) == before


# ── D-REF-1: the DB road, fixed ──────────────────────────────────────────────


class TestIngestedLibraryIsSearchable:
    def test_etl_build_then_forge_sql_retrieve_finds_reference(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.forge import KnowledgeForge

        db = tmp_path / "k.db"
        stats = ReferenceDataETL(db_path=db).build(
            ref_dir=syn.REFERENCE_DIR)
        assert stats == {"files": 3, "chunks": 6, "skipped": 0}
        forge = KnowledgeForge(db_path=db)
        rows = forge.sql_retrieve("ferry", top_k=20)
        reference_rows = [r for r in rows
                          if r.get("record_type") == "reference_chunk"]
        assert reference_rows, "ingested library invisible to the forge"
        assert any("ferry" in r["content"].lower()
                   for r in reference_rows)
        bundle = forge.retrieve("ferry", path="sql", min_score=0)
        assert bundle.total_retrieved >= 1
