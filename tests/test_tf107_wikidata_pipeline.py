"""
TF-107 — Wikidata pipeline hardened: rate-respecting,
resumable, checkpointed (long ingestions survive
interruption).

The hardening lives as verifiable data in
``src/thoughtforge/etl/wikidata_pipeline.py``
(``PIPELINE_RULES``, the checkpoint record IO, and
``RatePacer``, checked by ``pipeline_problems()`` and
``pipeline_live_problems()``) and is enforced in
``src/thoughtforge/etl/wikidata.py``; the write-up is
``docs/architecture/WIKIDATA_PIPELINE.md``. D-WIKI-1
(statement duplication), D-WIKI-2 (no checkpoint /
resume, synchronous OFF) and D-WIKI-3 (failure path
masking its own error) were found by this slice's probes
and are FIXED here; D-COV-2's wikidata-depth share is
discharged.

Pinned here:

  * a checkpoint is filed only after its commit, every
    ``checkpoint_every`` stream items and at completion,
    atomically, beside the DB — no schema table added,
  * resume continues from the checkpoint's stream
    position under the same dump + config fingerprint,
    a completed checkpoint returns its filed stats with
    no streaming at all, and a changed ask (limit /
    languages / dump) raises CheckpointMismatchError,
  * re-processing never duplicates a row: a second full
    run and a resume=False re-run leave a 7-statement
    store at 7 statements (the pre-fix probe reached 14),
  * an interruption mid-dump leaves the last checkpoint
    landed in the store, the failure in ingestion_log,
    and the ORIGINAL exception in the caller's hand,
  * a set rate cap is a hard ceiling (fake-clock pacer
    arithmetic + the builder wiring), a corrupt
    checkpoint refuses by name, and both checkers are
    clean.

All fixtures are synthetic (TF-017): invented lantern
entities in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic dumps through the real
builder on this VM; interruption simulated by a stream
raising mid-dump). A real multi-GB ingestion, a real
kill / power loss, and at-scale rates are NOT PROVEN.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from thoughtforge.etl.wikidata import WikidataETL
from thoughtforge.etl.wikidata_pipeline import (
    CHECKPOINT_FORMAT,
    DEFAULT_CHECKPOINT_EVERY,
    PIPELINE_DOC,
    PIPELINE_FORMAT,
    PIPELINE_RULES,
    CheckpointCorruptError,
    CheckpointMismatchError,
    RatePacer,
    checkpoint_mismatches,
    checkpoint_path_for,
    config_fingerprint,
    discard_checkpoint,
    dump_fingerprint,
    pipeline_as_data,
    pipeline_live_problems,
    pipeline_problems,
    read_checkpoint,
    write_checkpoint,
)


def _entity(qid: str) -> dict:
    return {
        "id": qid,
        "labels": {"en": {"value": f"Lantern {qid}"}},
        "descriptions": {"en": {"value": "a synthetic entity"}},
        "aliases": {"en": []},
        "sitelinks": {},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q107900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }


@pytest.fixture()
def dump(tmp_path: Path) -> Path:
    path = tmp_path / "dump.json"
    path.write_text(json.dumps([_entity(f"Q{i}") for i in range(1, 8)]),
                    encoding="utf-8")
    return path


def _counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {t: conn.execute(
            f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("entities", "statements", "labels")}
    finally:
        conn.close()


class TestRulesData:
    def test_format_and_doc(self) -> None:
        assert PIPELINE_FORMAT == "thoughtforge-wikidata-pipeline/1"
        assert PIPELINE_DOC == "docs/architecture/WIKIDATA_PIPELINE.md"
        assert Path(PIPELINE_DOC).exists()

    def test_rules_unique(self) -> None:
        ids = [r["id"] for r in PIPELINE_RULES]
        assert len(ids) == len(set(ids)) == 9

    def test_as_data_shape(self) -> None:
        data = pipeline_as_data()
        assert data["format"] == PIPELINE_FORMAT
        assert data["checkpoint_format"] == CHECKPOINT_FORMAT
        assert data["checkpoint_every_default"] == \
            DEFAULT_CHECKPOINT_EVERY == 1000
        assert len(data["rules"]) == 9


class TestCheckpointIO:
    def test_path_is_sidecar(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        assert checkpoint_path_for(db) == \
            tmp_path / "k.db.wikidata-checkpoint.json"

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        record = {"status": "in_progress", "items_seen": 3,
                  "stats": {"entities": 3}, "dump": {"path": "x"},
                  "config": {"languages": ["en"]}}
        write_checkpoint(db, record)
        got = read_checkpoint(db)
        assert got is not None
        assert got["format"] == CHECKPOINT_FORMAT
        assert got["items_seen"] == 3
        # Atomic write leaves no tmp file behind.
        assert not checkpoint_path_for(db).with_suffix(
            ".json.tmp").exists()

    def test_absent_checkpoint_is_none(self, tmp_path: Path) -> None:
        assert read_checkpoint(tmp_path / "none.db") is None

    def test_discard(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        write_checkpoint(db, {"status": "complete", "items_seen": 1,
                              "stats": {}, "dump": {}, "config": {}})
        assert discard_checkpoint(db) is True
        assert read_checkpoint(db) is None
        assert discard_checkpoint(db) is False

    def test_corrupt_checkpoint_refuses(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        checkpoint_path_for(db).write_text("{not json",
                                           encoding="utf-8")
        with pytest.raises(CheckpointCorruptError):
            read_checkpoint(db)

    def test_malformed_checkpoint_refuses(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        checkpoint_path_for(db).write_text(
            json.dumps({"format": CHECKPOINT_FORMAT}),
            encoding="utf-8")
        with pytest.raises(CheckpointCorruptError):
            read_checkpoint(db)

    def test_fingerprints_and_mismatch_names(self, dump: Path) -> None:
        fp = dump_fingerprint(dump)
        assert fp["size"] == dump.stat().st_size
        cfg = config_fingerprint(["en"], None, None)
        assert cfg == {"languages": ["en"],
                       "instance_of_filter": None, "limit": None}
        record = {"dump": fp, "config": cfg}
        assert checkpoint_mismatches(record, fp, cfg) == []
        changed = config_fingerprint(["en"], None, 2)
        problems = checkpoint_mismatches(record, fp, changed)
        assert len(problems) == 1 and "limit" in problems[0]


class TestRatePacer:
    def test_fake_clock_ceiling(self) -> None:
        now = [50.0]
        slept: list[float] = []
        pacer = RatePacer(10.0, clock=lambda: now[0],
                          sleep=lambda s: slept.append(s))
        assert pacer.wait_for(5) == 0.5
        assert slept == [0.5]
        now[0] += 1.0
        assert pacer.wait_for(10) == 0.0

    def test_zero_cap_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            RatePacer(0)


class TestHardenedBuild:
    def test_fresh_run_and_complete_checkpoint(
            self, tmp_path: Path, dump: Path) -> None:
        db = tmp_path / "k.db"
        stats = WikidataETL(db).build(dump_path=dump, checkpoint_every=3)
        assert stats == {"entities": 7, "statements": 7,
                         "labels": 7, "skipped": 0}
        assert _counts(db) == {"entities": 7, "statements": 7,
                               "labels": 7}
        record = read_checkpoint(db)
        assert record is not None
        assert record["status"] == "complete"
        assert record["items_seen"] == 7
        assert record["last_qid"] == "Q7"
        assert record["stats"] == stats

    def test_completed_resume_never_streams(
            self, tmp_path: Path, dump: Path) -> None:
        db = tmp_path / "k.db"
        WikidataETL(db).build(dump_path=dump, checkpoint_every=3)
        etl = WikidataETL(db)

        def boom(path: Path):  # pragma: no cover - must not run
            raise AssertionError("a completed build re-streamed")
            yield

        etl._stream_entities = boom  # type: ignore[method-assign]
        stats = etl.build(dump_path=dump, checkpoint_every=3)
        assert stats == {"entities": 7, "statements": 7,
                         "labels": 7, "skipped": 0}

    def test_rerun_duplicates_nothing(
            self, tmp_path: Path, dump: Path) -> None:
        db = tmp_path / "k.db"
        WikidataETL(db).build(dump_path=dump, checkpoint_every=3)
        WikidataETL(db).build(dump_path=dump, checkpoint_every=3,
                              resume=False)
        # Pre-fix probe: the second run moved statements 7 -> 14.
        assert _counts(db) == {"entities": 7, "statements": 7,
                               "labels": 7}

    def test_interruption_then_resume(
            self, tmp_path: Path, dump: Path) -> None:
        db = tmp_path / "k.db"
        etl = WikidataETL(db)
        stream = etl._stream_entities

        def flaky(path: Path):
            for i, item in enumerate(stream(path)):
                if i == 4:
                    raise RuntimeError("tf107 simulated interruption")
                yield item

        etl._stream_entities = flaky  # type: ignore[method-assign]
        with pytest.raises(RuntimeError,
                           match="simulated interruption"):
            etl.build(dump_path=dump, checkpoint_every=3)

        # The last checkpoint (item 3) is what landed; the
        # failure is logged; the original error reached us.
        assert _counts(db)["entities"] == 3
        record = read_checkpoint(db)
        assert record is not None
        assert record["status"] == "in_progress"
        assert record["items_seen"] == 3
        conn = sqlite3.connect(str(db))
        try:
            log = conn.execute(
                "SELECT status FROM ingestion_log").fetchall()
        finally:
            conn.close()
        assert any(str(row[0]).startswith("failed") for row in log)

        stats = WikidataETL(db).build(dump_path=dump,
                                      checkpoint_every=3)
        assert stats == {"entities": 7, "statements": 7,
                         "labels": 7, "skipped": 0}
        assert _counts(db) == {"entities": 7, "statements": 7,
                               "labels": 7}

    def test_changed_ask_refuses_resume(
            self, tmp_path: Path, dump: Path) -> None:
        db = tmp_path / "k.db"
        WikidataETL(db).build(dump_path=dump, checkpoint_every=3)
        with pytest.raises(CheckpointMismatchError, match="limit"):
            WikidataETL(db).build(dump_path=dump, checkpoint_every=3,
                                  limit=2)
        with pytest.raises(CheckpointMismatchError,
                           match="languages"):
            WikidataETL(db).build(dump_path=dump, checkpoint_every=3,
                                  languages=["en", "de"])

    def test_rate_cap_is_wired(
            self, tmp_path: Path, dump: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        import thoughtforge.etl.wikidata as wikidata_module

        calls: list[int] = []

        class FakePacer:
            def __init__(self, cap: float) -> None:
                assert cap == 123.0
                calls.append(0)

            def wait_for(self, n: int) -> float:
                calls.append(n)
                return 0.0

        monkeypatch.setattr(wikidata_module, "RatePacer", FakePacer)
        db = tmp_path / "k.db"
        WikidataETL(db).build(dump_path=dump, checkpoint_every=3,
                              max_items_per_second=123.0)
        # One wait per processed item, counts rising 1..7.
        assert calls == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_invalid_checkpoint_every_refused(
            self, tmp_path: Path, dump: Path) -> None:
        with pytest.raises(ValueError, match="checkpoint_every"):
            WikidataETL(tmp_path / "k.db").build(
                dump_path=dump, checkpoint_every=0)


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert pipeline_problems() == []

    def test_live_checker_clean(self) -> None:
        assert pipeline_live_problems() == []
