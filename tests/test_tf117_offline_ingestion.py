"""
TF-117 — Offline ingestion: everything except the
fetch works with the network cut.

The proof lives as verifiable data in
``src/thoughtforge/etl/offline_ingestion.py``
(``FETCH_STEPS``, ``OFFLINE_STEPS``, ``OFFLINE_RULES``,
``NetworkCut``, ``run_offline_pipeline()``, checked by
``offline_problems()`` and ``offline_live_problems()``);
the embedding loader in ``etl/embeddings.py`` gained
its local-files-only guard in the open and TF-105's
map gained its ``offline`` entry in the open.
D-OFFLINE-1 (offline assumed, never proven),
D-OFFLINE-2 (the embedding loader was an unguarded
fetch) and D-OFFLINE-3 (the fetch boundary was
undefined) were found by this slice's probes and are
FIXED here.

Pinned here:

  * the boundary as data: exactly three fetch steps,
    none performed by ingestion; the offline steps
    bind exactly to TF-105's builders; the rules,
    the format + doc,
  * the cut: socket connect, create_connection,
    getaddrinfo and urlopen all blocked + recorded;
    restoration exact, on exception and nested;
    socket creation alone is not an attempt,
  * the whole synthetic pipeline under the cut:
    filed counts, embeddings, subset, scan, zero
    attempts, offline True,
  * a dry-run and a TF-111 injected-encoder embedding
    build under the cut,
  * the loader: local_files_only reaches the real
    constructor shape; a constructor rejecting it
    (the filed fake shape) falls back plainly,
  * both checkers are clean.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — a real dump with the machine's network
physically cut is NOT PROVEN.
"""

from __future__ import annotations

import socket
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from thoughtforge.etl import embeddings
from thoughtforge.etl.etl_map import SOURCE_BUILDERS
from thoughtforge.etl.offline_ingestion import (
    FETCH_STEPS,
    OFFLINE_DOC,
    OFFLINE_FORMAT,
    OFFLINE_RULES,
    OFFLINE_STEPS,
    NetworkCut,
    OfflineAttemptError,
    offline_as_data,
    offline_live_problems,
    offline_problems,
    run_offline_pipeline,
)


class TestBoundaryData:
    def test_format_and_doc(self) -> None:
        assert OFFLINE_FORMAT == "thoughtforge-offline-ingestion/1"
        assert OFFLINE_DOC == "docs/architecture/OFFLINE_INGESTION.md"
        assert Path(OFFLINE_DOC).exists()

    def test_fetch_steps_are_exactly_three_and_external(self) -> None:
        assert [s["id"] for s in FETCH_STEPS] == [
            "source_file_download",
            "embedding_model_download",
            "package_install",
        ]
        assert all(
            s["ingestion_code_performs_it"] is False for s in FETCH_STEPS)

    def test_offline_steps_bind_to_tf105_builders(self) -> None:
        bound = [s["builder"] for s in OFFLINE_STEPS if s["builder"]]
        assert bound == [b["id"] for b in SOURCE_BUILDERS]

    def test_rules_unique_and_as_data(self) -> None:
        ids = [r["id"] for r in OFFLINE_RULES]
        assert len(ids) == len(set(ids)) == 11
        data = offline_as_data()
        assert data["format"] == OFFLINE_FORMAT
        assert [r["id"] for r in data["rules"]] == ids
        assert len(data["fetch_steps"]) == 3


class TestNetworkCut:
    def test_create_connection_blocked_and_recorded(self) -> None:
        cut = NetworkCut()
        with cut:
            with pytest.raises(OfflineAttemptError):
                socket.create_connection(("example.invalid", 443))
        assert cut.attempts == [
            {"kind": "socket.create_connection",
             "target": "('example.invalid', 443)"}]

    def test_urlopen_and_dns_blocked(self) -> None:
        cut = NetworkCut()
        with cut:
            with pytest.raises(OfflineAttemptError):
                urllib.request.urlopen("https://example.invalid/x")
            with pytest.raises(OfflineAttemptError):
                socket.getaddrinfo("example.invalid", 443)
        assert [a["kind"] for a in cut.attempts] == [
            "urllib.request.urlopen", "socket.getaddrinfo"]

    def test_socket_creation_is_not_an_attempt_but_connect_is(
            self) -> None:
        cut = NetworkCut()
        with cut:
            sock = socket.socket()
            sock.close()
            assert cut.attempts == []
            sock = socket.socket()
            with pytest.raises(OfflineAttemptError):
                sock.connect(("192.0.2.1", 80))
            sock.close()
        assert [a["kind"] for a in cut.attempts] == ["socket.connect"]

    def test_restores_exactly_on_exit_and_on_exception(self) -> None:
        real_gai, real_urlopen = socket.getaddrinfo, urllib.request.urlopen
        with NetworkCut():
            assert socket.getaddrinfo is not real_gai
        assert socket.getaddrinfo is real_gai
        with pytest.raises(RuntimeError, match="boom"):
            with NetworkCut():
                raise RuntimeError("boom")
        assert socket.getaddrinfo is real_gai
        assert urllib.request.urlopen is real_urlopen

    def test_nested_cut_restores_only_at_the_outer_exit(self) -> None:
        real_gai = socket.getaddrinfo
        outer, inner = NetworkCut(), NetworkCut()
        with outer:
            with inner:
                pass
            assert socket.getaddrinfo is not real_gai
        assert socket.getaddrinfo is real_gai


class TestOfflinePipeline:
    def test_whole_pipeline_under_the_cut(self, tmp_path: Path) -> None:
        report = run_offline_pipeline(tmp_path / "run")
        assert report["offline"] is True
        assert report["network_attempts"] == []
        counts = report["pipeline"]["store_counts"]
        assert counts == {
            "entities": 1, "statements": 1, "labels": 1,
            "embeddings": 1, "conceptnet_edges": 1,
            "geonames_places": 1, "dbpedia_entities": 1,
            "reference_chunks": 1,
        }
        assert report["pipeline"]["entity_embeddings"] == 1
        assert report["pipeline"]["reference_embeddings"] == 1
        assert report["pipeline"]["subset"]["success"] is True
        assert report["scan"]["scanned_rows"] >= 1
        assert report["scan"]["flagged"] == 0

    def test_dry_run_under_the_cut(self, tmp_path: Path) -> None:
        from thoughtforge.etl.ingestion_dry_run import dry_run_ingestion

        txt = tmp_path / "places.txt"
        row = "\t".join([
            "117001", "Lantern Town", "Lantern Town", "", "1.5",
            "2.5", "P", "PPL", "XX", "", "A1", "A2", "", "",
            "100", "10", "10", "UTC", "2026-01-01"])
        txt.write_text(row + "\n", encoding="utf-8")
        with NetworkCut() as cut:
            report = dry_run_ingestion("geonames", txt)
        assert cut.attempts == []
        assert report.stats == {"inserted": 1, "skipped": 0}

    def test_tf111_injected_encoder_under_the_cut(
            self, tmp_path: Path) -> None:
        from thoughtforge.etl.embedding_ingestion import (
            _fake_encoder,
            build_embeddings_at_ingestion,
        )
        from thoughtforge.etl.sources import ReferenceDataETL

        ref = tmp_path / "ref"
        ref.mkdir()
        (ref / "lantern.md").write_text(
            "# Lantern\n\nKeep the lantern trimmed.\n",
            encoding="utf-8")
        db = tmp_path / "k.db"
        ReferenceDataETL(db).build(ref)
        with NetworkCut() as cut:
            result = build_embeddings_at_ingestion(
                db, encoder=_fake_encoder())
        assert cut.attempts == []
        assert result.reference_built >= 1


class TestLoaderGuard:
    def test_local_files_only_reaches_the_constructor(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict] = []

        class RecordingST:
            def __init__(self, name: str, **kwargs) -> None:
                calls.append({"name": name, **kwargs})
                self.name = name

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = RecordingST  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embeddings._MODEL_CACHE.clear()
        try:
            model = embeddings._get_model("all-MiniLM-L6-v2")
            assert model.name == "all-MiniLM-L6-v2"
            assert calls == [{
                "name": "all-MiniLM-L6-v2", "local_files_only": True}]
        finally:
            embeddings._MODEL_CACHE.clear()

    def test_constructor_rejecting_the_argument_falls_back(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class PlainST:
            def __init__(self, name: str) -> None:
                self.name = name

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = PlainST  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embeddings._MODEL_CACHE.clear()
        try:
            model = embeddings._get_model("all-MiniLM-L6-v2")
            assert model.name == "all-MiniLM-L6-v2"
        finally:
            embeddings._MODEL_CACHE.clear()

    def test_missing_package_still_raises_the_named_error(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        embeddings._MODEL_CACHE.clear()
        with pytest.raises(RuntimeError, match="not installed"):
            embeddings._get_model("all-MiniLM-L6-v2")
        embeddings._MODEL_CACHE.clear()


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert offline_problems() == []

    def test_live_checker_clean(self) -> None:
        assert offline_live_problems() == []
