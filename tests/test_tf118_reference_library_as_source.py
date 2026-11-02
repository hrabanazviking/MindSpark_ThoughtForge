"""
TF-118 — The reference library as a source: its
ingestion reproducible + checksummed.

The proof lives as verifiable data in
``src/thoughtforge/etl/reference_ingestion.py``
(``REFERENCE_RULES``, ``LibraryLedger``,
``ingest_reference_library()``, ``store_reference_digest()``,
``verify_reference_store()``, ``check_reproducibility()``,
checked by ``reference_ingestion_problems()`` and
``reference_ingestion_live_problems()``); the builder in
``etl/sources.py`` gained its ``library_gate`` ledger,
skip-by-rule and dedup-by-hash in the open, TF-105's
map gained its ``reference_library`` entry in the
open, and TF-113's canon filing was re-pinned in the
open. D-DATA-4 (the ETL's silent skip + double
chunking) is FIXED here, and D-REF-2 (no checksum
anywhere on the reference road) and D-REF-3
(reproducibility unproven) were found by this slice's
probes and are FIXED here.

Pinned here:

  * the filing as data: format, doc, rules, statuses,
  * the builder alone: unsupported files classified,
    counted and named; a byte-duplicate chunked once
    under its sorted keeper; a duplicate re-run
    retiring stale chunks,
  * the driver: per-file statuses + hashes, per-chunk
    checksums, library + store digests, the manifest
    (content-free, hash true), the source untouched,
  * verification: a clean store verifies; a tampered
    chunk, a missing chunk and a changed source file
    are each named exactly,
  * reproducibility: two independent builds agree on
    stats, chunk checksums and store digest; a re-run
    leaves the digest,
  * a missing library raises before any store; the
    embedding leg runs only with an injected encoder,
  * both checkers are clean.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — the real built-in library is NOT PROVEN.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.reference_ingestion import (
    CHUNK_FIELDS,
    FILE_STATUSES,
    MANIFEST_SUFFIX,
    REFERENCE_DOC,
    REFERENCE_FORMAT,
    REFERENCE_RULES,
    LibraryLedger,
    check_reproducibility,
    chunk_checksum,
    classify_reference_extension,
    file_sha256,
    ingest_reference_library,
    reference_ingestion_as_data,
    reference_ingestion_live_problems,
    reference_ingestion_problems,
    report_as_data,
    store_reference_digest,
    verify_reference_store,
)
from thoughtforge.etl.sources import ReferenceDataETL


def _lib(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a_ferries.md").write_text(
        "# Ferries\n\nThe ferry crosses the river at dawn.\n",
        encoding="utf-8")
    (root / "bells.txt").write_text(
        "The tide bell rings twice at dusk.\n", encoding="utf-8")
    (root / "zz_copy.md").write_bytes(
        (root / "a_ferries.md").read_bytes())
    (root / "photo.jpg").write_bytes(b"\xff\xd8x")
    (root / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "empty.md").write_text("", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ── The filing as data ──────────────────────────────────────────────────────


class TestFilingData:
    def test_format_doc_and_rules(self) -> None:
        assert REFERENCE_FORMAT == "thoughtforge-reference-ingestion/1"
        assert REFERENCE_DOC == \
            "docs/architecture/REFERENCE_LIBRARY_AS_SOURCE.md"
        assert Path(REFERENCE_DOC).exists()
        ids = [r["id"] for r in REFERENCE_RULES]
        assert len(ids) == len(set(ids)) == 11
        assert FILE_STATUSES == (
            "ingested", "duplicate", "skipped_by_rule", "no_chunks")
        data = reference_ingestion_as_data()
        assert data["format"] == REFERENCE_FORMAT
        assert [r["id"] for r in data["rules"]] == ids
        assert list(CHUNK_FIELDS) == [
            "source_file", "chunk_index", "title", "content",
            "tags", "word_count"]

    def test_classify_matches_tf058_classes(self) -> None:
        assert classify_reference_extension(".JPG") == "image"
        assert classify_reference_extension(".png") == "image"
        assert classify_reference_extension(".csv") == "other"
        assert classify_reference_extension(".rst") == "other"

    def test_ledger_as_data(self) -> None:
        ledger = LibraryLedger()
        ledger.note_skip("photo.jpg", ".jpg")
        ledger.note_duplicate("zz.md", "a.md", "ab" * 32)
        data = ledger.as_data()
        assert data["skipped_by_rule"] == [
            {"file": "photo.jpg", "extension": ".jpg",
             "file_class": "image"}]
        assert data["duplicates"] == [
            {"file": "zz.md", "duplicate_of": "a.md",
             "sha256": "ab" * 32}]
        assert data["skipped_files"] == ledger.skipped == 2

    def test_checksums_are_real_sha256(self, tmp_path: Path) -> None:
        path = tmp_path / "x.md"
        path.write_bytes(b"lantern")
        assert file_sha256(path) == \
            hashlib.sha256(b"lantern").hexdigest()
        row = {"source_file": "x.md", "chunk_index": 0,
               "title": "X", "content": "lantern", "tags": "",
               "word_count": 1}
        assert len(chunk_checksum(row)) == 64
        changed = dict(row, content="lanterns")
        assert chunk_checksum(changed) != chunk_checksum(row)


# ── The builder alone (D-DATA-4) ────────────────────────────────────────────


class TestBuilderAlone:
    def test_skip_by_rule_and_dedup(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        builder = ReferenceDataETL(tmp_path / "s.db")
        stats = builder.build(ref_dir=lib)
        assert stats == {"files": 2, "chunks": 2, "skipped": 4}
        gate = builder.library_gate.as_data()
        assert gate["skipped_by_rule"] == [
            {"file": "photo.jpg", "extension": ".jpg",
             "file_class": "image"},
            {"file": "table.csv", "extension": ".csv",
             "file_class": "other"}]
        assert gate["duplicates"] == [
            {"file": "zz_copy.md", "duplicate_of": "a_ferries.md",
             "sha256": file_sha256(lib / "zz_copy.md")}]
        conn = sqlite3.connect(str(tmp_path / "s.db"))
        try:
            files = [r[0] for r in conn.execute(
                "SELECT DISTINCT source_file FROM reference_chunks")]
        finally:
            conn.close()
        assert sorted(files) == ["a_ferries.md", "bells.txt"]

    def test_duplicate_rerun_retires_stale_chunks(
            self, tmp_path: Path) -> None:
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "a.md").write_text("# A\n\nalpha text here\n",
                                  encoding="utf-8")
        (lib / "b.md").write_text("# B\n\nbravo text here\n",
                                  encoding="utf-8")
        db = tmp_path / "s.db"
        ReferenceDataETL(db).build(ref_dir=lib)
        # b.md becomes a byte-duplicate of a.md.
        (lib / "b.md").write_bytes((lib / "a.md").read_bytes())
        builder = ReferenceDataETL(db)
        stats = builder.build(ref_dir=lib)
        assert stats == {"files": 1, "chunks": 1, "skipped": 1}
        conn = sqlite3.connect(str(db))
        try:
            files = [r[0] for r in conn.execute(
                "SELECT source_file FROM reference_chunks")]
        finally:
            conn.close()
        assert files == ["a.md"]
        assert builder.update_ledger.as_data()["by_outcome"] == \
            {"chunk_retired": 1}


# ── The driver + manifest ───────────────────────────────────────────────────


class TestDriver:
    def test_full_story_in_one_report(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        before = _snapshot(lib)
        db = tmp_path / "store.db"
        report = ingest_reference_library(lib, db)
        data = report_as_data(report)
        assert data["format"] == REFERENCE_FORMAT
        assert report.stats == {"files": 2, "chunks": 2, "skipped": 4}
        assert report.store_counts["reference_chunks"] == 2
        assert report.source_untouched is True
        assert _snapshot(lib) == before
        assert report.embeddings["status"] == "not_run"
        statuses = {f["relative_path"]: f["status"]
                    for f in report.files}
        assert statuses == {
            "a_ferries.md": "ingested", "bells.txt": "ingested",
            "zz_copy.md": "duplicate",
            "photo.jpg": "skipped_by_rule",
            "table.csv": "skipped_by_rule",
            "empty.md": "no_chunks"}
        assert {c["chunk_id"] for c in report.chunks} == \
            {"a_ferries.md#0", "bells.txt#0"}
        assert report.store_digest == store_reference_digest(db)
        assert len(report.library_digest) == 64

    def test_manifest_verifies_and_carries_no_content(
            self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        report = ingest_reference_library(lib, db)
        manifest_path = Path(report.manifest_path)
        assert manifest_path.name == "store.db" + MANIFEST_SUFFIX
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["manifest_sha256"] == report.manifest_sha256
        assert all("content" not in chunk
                   for chunk in payload["chunks"])
        assert all("content" not in row for row in payload["files"])
        payload["stats"] = {"files": 99, "chunks": 99, "skipped": 0}
        digest = payload.pop("manifest_sha256")
        canonical = json.dumps(payload, sort_keys=True,
                               separators=(",", ":"),
                               ensure_ascii=False)
        assert hashlib.sha256(
            canonical.encode("utf-8")).hexdigest() != digest

    def test_missing_library_raises_before_store(
            self, tmp_path: Path) -> None:
        db = tmp_path / "store.db"
        with pytest.raises(FileNotFoundError, match="absent-lib"):
            ingest_reference_library(tmp_path / "absent-lib", db)
        assert not db.exists()
        assert not Path(str(db) + MANIFEST_SUFFIX).exists()

    def test_injected_encoder_runs_the_embedding_leg(
            self, tmp_path: Path) -> None:
        import numpy as np

        def fake_encoder(texts, model_name):
            vectors = []
            for text in texts:
                seed = int(hashlib.sha256(
                    text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                vectors.append(rng.random(384, dtype=np.float32))
            return np.array(vectors, dtype=np.float32)

        lib = _lib(tmp_path / "lib")
        report = ingest_reference_library(
            lib, tmp_path / "store.db", encoder=fake_encoder)
        assert report.embeddings["status"] == "run"
        assert report.embeddings["reference_built"] == 2


# ── Verification (D-REF-2) ──────────────────────────────────────────────────


class TestVerification:
    def test_clean_store_verifies(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        ingest_reference_library(lib, db)
        verdict = verify_reference_store(db)
        assert verdict["ok"] is True
        assert verdict["manifest_ok"] is True
        assert verdict["store_digest_matches"] is True
        assert verdict["chunks_checked"] == 2
        assert verdict["chunk_problems"] == []
        assert verdict["source"]["problems"] == []

    def test_tampered_chunk_is_named(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        ingest_reference_library(lib, db)
        conn = sqlite3.connect(str(db))
        try:
            conn.execute(
                "UPDATE reference_chunks SET content='TAMPERED'"
                " WHERE source_file='bells.txt'")
            conn.commit()
        finally:
            conn.close()
        verdict = verify_reference_store(db)
        assert verdict["ok"] is False
        assert verdict["chunk_problems"] == [
            {"chunk_id": "bells.txt#0", "drift": "checksum_mismatch"}]

    def test_missing_and_extra_chunks_are_named(
            self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        ingest_reference_library(lib, db)
        conn = sqlite3.connect(str(db))
        try:
            conn.execute(
                "DELETE FROM reference_chunks"
                " WHERE source_file='bells.txt'")
            conn.execute(
                "INSERT INTO reference_chunks"
                " (source_file, chunk_index, title, content, tags,"
                " word_count) VALUES ('ghost.md', 0, 'G', 'boo', '', 1)")
            conn.commit()
        finally:
            conn.close()
        verdict = verify_reference_store(db)
        assert verdict["ok"] is False
        assert verdict["chunk_problems"] == [
            {"chunk_id": "bells.txt#0", "drift": "missing"},
            {"chunk_id": "ghost.md#0", "drift": "extra"}]

    def test_changed_source_file_is_named(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        ingest_reference_library(lib, db)
        (lib / "bells.txt").write_text(
            "The tide bell rings three times.\n", encoding="utf-8")
        verdict = verify_reference_store(db)
        assert verdict["ok"] is False
        assert verdict["store_digest_matches"] is True
        assert verdict["source"]["problems"] == [
            {"file": "bells.txt", "drift": "changed"}]

    def test_missing_manifest_is_a_verdict(self, tmp_path: Path) -> None:
        verdict = verify_reference_store(tmp_path / "never.db")
        assert verdict["ok"] is False
        assert "manifest not found" in verdict["problems"][0]


# ── Reproducibility (D-REF-3) ───────────────────────────────────────────────


class TestReproducibility:
    def test_two_builds_agree_on_all_three(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        repro = check_reproducibility(
            lib, tmp_path / "r1.db", tmp_path / "r2.db")
        assert repro["reproducible"] is True
        assert repro["stats_agree"] is True
        assert repro["chunks_agree"] is True
        assert repro["digests_agree"] is True
        assert repro["chunks"] == 2

    def test_rerun_leaves_the_digest(self, tmp_path: Path) -> None:
        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        first = ingest_reference_library(lib, db)
        second = ingest_reference_library(lib, db)
        assert second.store_digest == first.store_digest
        assert second.stats == first.stats

    def test_ingested_library_is_searchable_under_the_keeper(
            self, tmp_path: Path) -> None:
        from thoughtforge.knowledge.forge import KnowledgeForge

        lib = _lib(tmp_path / "lib")
        db = tmp_path / "store.db"
        ingest_reference_library(lib, db)
        rows = KnowledgeForge(db_path=db).sql_retrieve("ferry")
        assert rows
        assert rows[0]["source_file"] == "a_ferries.md"
        assert all(r["source_file"] != "zz_copy.md" for r in rows)


# ── Checkers ────────────────────────────────────────────────────────────────


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert reference_ingestion_problems() == []

    def test_live_checker_clean(self) -> None:
        assert reference_ingestion_live_problems() == []
