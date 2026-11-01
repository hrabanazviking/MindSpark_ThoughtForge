"""
TF-113 — Canon ingestion path: Volmarr's own
materials ingested by the same pipeline (private
stores only; fixtures synthetic).

The driver lives as verifiable data in
``src/thoughtforge/etl/canon_ingestion.py``
(``CANON_PROVENANCE``, ``CANON_RULES``,
``check_canon_use``, ``ingest_canon``, checked by
``canon_problems()`` and ``canon_live_problems()``).
D-CANON-1 (owner's materials had no ingestion path
with a private-store guard or a provenance record)
is added and FIXED here.

Pinned here:

  * the rules are unique data and the provenance
    states no licence and private_local_only privacy,
  * the use gate passes private ingestion/retrieval
    with the provenance record and raises for
    redistribution, publication and unknown uses,
  * without private_store=True nothing is created;
    a store inside the canon directory raises; a
    missing canon directory raises by name before
    any store exists,
  * the happy road runs the real reference builder:
    stats in its filed shape, store counts, the
    merge ledger, and retrieval through the forge
    finds the canon chunk in its private store,
  * skips are named by rule (image / other) and the
    byte-duplicate is named with its keeper,
  * the canon source is proven untouched
    byte-for-byte across the build,
  * the manifest sits beside the store, carries no
    file content, and its filed hash verifies its
    contents (an edited manifest does not verify),
  * the embedding leg is not run without an encoder
    and, with an injected encoder, lands through
    TF-111's driver,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented
materials in a temp dir, never real canon.

Proof grade: FIXTURE (synthetic canon stand-ins
through the real builder on this VM). The real
canon tree is NOT PROVEN — never opened here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl.canon_ingestion import (
    CANON_FORMAT,
    CANON_PROVENANCE,
    CANON_RULES,
    CANON_SOURCE_ID,
    USE_PUBLISH,
    USE_REDISTRIBUTE,
    USE_RETRIEVE_PRIVATE,
    CanonPrivacyError,
    CanonTargetError,
    canon_as_data,
    canon_live_problems,
    canon_problems,
    check_canon_use,
    ingest_canon,
    report_as_data,
)
from thoughtforge.etl.schema import get_connection
from thoughtforge.knowledge.forge import KnowledgeForge


def _canon(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "lantern_lore.md").write_text(
        "# Lantern Lore\n\nThe keeper trims the lantern at dusk.\n",
        encoding="utf-8")
    (root / "copy.md").write_bytes(
        (root / "lantern_lore.md").read_bytes())
    (root / "empty.md").write_text("", encoding="utf-8")
    (root / "portrait.jpg").write_bytes(b"\xff\xd8x")
    (root / "notes.csv").write_text("a,b\n", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _fake_encoder(texts, model_name):
    vectors = []
    for text in texts:
        seed = int(hashlib.sha256(
            text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vectors.append(rng.random(384, dtype=np.float32))
    return np.array(vectors, dtype=np.float32)


class TestRulesAndProvenance:
    def test_rules_are_unique_data(self) -> None:
        ids = [r["id"] for r in CANON_RULES]
        assert len(ids) == len(set(ids))
        assert "the_same_pipeline" in ids
        assert "private_stores_only_affirmed" in ids
        assert "the_provenance_manifest_travels_with_the_store" in ids

    def test_provenance_states_no_licence(self) -> None:
        assert CANON_PROVENANCE["id"] == CANON_SOURCE_ID
        assert CANON_PROVENANCE["licence"] is None
        assert CANON_PROVENANCE["privacy"] == "private_local_only"
        assert "no_redistribution" in CANON_PROVENANCE["obligations"]

    def test_as_data_shape(self) -> None:
        data = canon_as_data()
        assert data["format"] == CANON_FORMAT
        assert data["provenance"]["id"] == CANON_SOURCE_ID
        assert len(data["rules"]) == len(CANON_RULES)


class TestUseGate:
    def test_private_uses_pass_with_provenance(self) -> None:
        for use in ("ingest_private", USE_RETRIEVE_PRIVATE):
            assert check_canon_use(use)["id"] == CANON_SOURCE_ID

    @pytest.mark.parametrize("use", [USE_REDISTRIBUTE, USE_PUBLISH])
    def test_public_uses_raise(self, use: str) -> None:
        with pytest.raises(CanonPrivacyError):
            check_canon_use(use)

    def test_unknown_use_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="publish_everywhere"):
            check_canon_use("publish_everywhere")


class TestPrivateStoreGuard:
    def test_no_affirmation_no_store(self, tmp_path: Path) -> None:
        canon = _canon(tmp_path / "canon")
        db = tmp_path / "private.db"
        with pytest.raises(CanonPrivacyError):
            ingest_canon(canon, db)
        assert not db.exists()
        assert not Path(str(db) + ".canon-manifest.json").exists()

    def test_store_inside_canon_raises(self, tmp_path: Path) -> None:
        canon = _canon(tmp_path / "canon")
        with pytest.raises(CanonTargetError):
            ingest_canon(canon, canon / "store.db", private_store=True)
        assert not (canon / "store.db").exists()

    def test_missing_canon_raises_before_store(
        self, tmp_path: Path,
    ) -> None:
        db = tmp_path / "private.db"
        with pytest.raises(FileNotFoundError, match="absent-canon"):
            ingest_canon(tmp_path / "absent-canon", db,
                         private_store=True)
        assert not db.exists()


class TestCanonRoad:
    def test_full_story_in_one_report(self, tmp_path: Path) -> None:
        canon = _canon(tmp_path / "canon")
        before = _snapshot(canon)
        db = tmp_path / "private.db"
        report = ingest_canon(canon, db, private_store=True)
        data = report_as_data(report)
        assert data["format"] == CANON_FORMAT
        assert report.stats == {"files": 2, "chunks": 2, "skipped": 1}
        assert report.store_counts["reference_chunks"] == 2
        assert report.merge["format"] == "thoughtforge-dedup-merge/1"
        assert report.source_untouched is True
        assert _snapshot(canon) == before
        assert report.embeddings["status"] == "not_run"
        assert report.provenance["privacy"] == "private_local_only"

    def test_skips_and_duplicates_are_named(
        self, tmp_path: Path,
    ) -> None:
        canon = _canon(tmp_path / "canon")
        report = ingest_canon(canon, tmp_path / "p.db",
                              private_store=True)
        skipped = {s["file"]: s["file_class"]
                   for s in report.skipped_by_rule}
        assert skipped == {"portrait.jpg": "image",
                           "notes.csv": "other"}
        assert len(report.duplicates) == 1
        # Sorted order keeps copy.md; the lore file is its duplicate.
        assert report.duplicates[0]["file"] == "lantern_lore.md"
        assert report.duplicates[0]["duplicate_of"] == "copy.md"
        assert report.inventory["files_by_class"] == {
            "readable": 3, "image": 1, "other": 1}
        assert all("content" not in row
                   for row in report.inventory["files"])

    def test_ingested_canon_is_searchable_in_its_store(
        self, tmp_path: Path,
    ) -> None:
        canon = _canon(tmp_path / "canon")
        db = tmp_path / "private.db"
        ingest_canon(canon, db, private_store=True)
        rows = KnowledgeForge(db_path=db).sql_retrieve("lantern keeper")
        assert rows
        assert rows[0]["source_file"] == "lantern_lore.md"


class TestManifest:
    def test_manifest_verifies_and_carries_no_content(
        self, tmp_path: Path,
    ) -> None:
        canon = _canon(tmp_path / "canon")
        db = tmp_path / "private.db"
        report = ingest_canon(canon, db, private_store=True)
        manifest_path = Path(report.manifest_path)
        assert manifest_path.is_file()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["manifest_sha256"] == report.manifest_sha256
        assert payload["provenance"]["privacy"] == "private_local_only"
        assert all("content" not in row
                   for row in payload["inventory"]["files"])
        # An edited manifest no longer verifies against its hash.
        payload["stats"] = {"files": 99, "chunks": 99, "skipped": 0}
        digest = payload.pop("manifest_sha256")
        canonical = json.dumps(payload, sort_keys=True,
                               separators=(",", ":"),
                               ensure_ascii=False)
        assert hashlib.sha256(
            canonical.encode("utf-8")).hexdigest() != digest


class TestEmbeddingLeg:
    def test_no_encoder_no_leg(self, tmp_path: Path) -> None:
        canon = _canon(tmp_path / "canon")
        report = ingest_canon(canon, tmp_path / "p.db",
                              private_store=True)
        assert report.embeddings["status"] == "not_run"

    def test_injected_encoder_lands_reference_embeddings(
        self, tmp_path: Path,
    ) -> None:
        canon = _canon(tmp_path / "canon")
        db = tmp_path / "private.db"
        report = ingest_canon(canon, db, private_store=True,
                              encoder=_fake_encoder)
        assert report.embeddings["status"] == "run"
        assert report.embeddings["reference_built"] == 2
        conn = get_connection(db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM reference_embeddings"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 2


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert canon_problems() == []

    def test_live_checker_clean(self) -> None:
        assert canon_live_problems() == []
