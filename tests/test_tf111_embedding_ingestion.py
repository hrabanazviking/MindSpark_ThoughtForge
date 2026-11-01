"""
TF-111 — Embedding at ingestion: model per profile
(TF-053), resumable, checksummed.

The driver lives as verifiable data in
``src/thoughtforge/etl/embedding_ingestion.py``
(``INGESTION_RULES``, ``embedding_plan``,
``build_embeddings_at_ingestion``, ``row_checksums``,
``estate_digest``, checked by
``embedding_ingestion_problems()`` and
``embedding_ingestion_live_problems()``); the
D-EMB-3 fix is enforced in
``src/thoughtforge/etl/embeddings.py`` (both
builders' only-missing joins are per model); the
write-up is
``docs/architecture/EMBEDDING_AT_INGESTION.md``.
D-EMB-3 (a row embedded under one model silently
blocked building it under another, while the
searches filter by model) was found by this slice's
probes and is FIXED here.

Pinned here:

  * plans resolve live from TF-053's assignment for
    every shipped profile (pi_zero assigned no model),
    no profile keeps the builders' defaults, explicit
    overrides win and are recorded as overrides,
    unknown profiles and bad batch sizes raise,
  * a full run through the real driver builds every
    row of both tables, checksummed, with a complete
    checkpoint whose estate digest re-derives,
  * a completed checkpoint IS the result (an encoder
    that raises if touched proves nothing encoded),
  * a changed ask refuses by name, a corrupt
    checkpoint refuses by name, resume=False discards
    explicitly and still builds nothing twice,
  * interruption stops at a verified batch edge and
    resume finishes exactly the remainder,
  * tampering with a recorded row is caught on
    resume by checksum,
  * pi_zero refuses by rule and writes nothing,
  * the builders themselves (fake
    sentence-transformers): a second model builds the
    rows the first model already embedded — D-EMB-3's
    regression pin,
  * both checkers are clean.

All fixtures are synthetic (TF-017): invented records
in a temp dir, never a real dump or canon.

Proof grade: FIXTURE (synthetic stores through the
real driver with a fake encoder on this VM). A
real-model ingestion at scale is NOT PROVEN.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl import embeddings
from thoughtforge.etl.embedding_discipline import (
    PROFILE_EMBEDDINGS,
    canonical_model_name,
)
from thoughtforge.etl.embedding_ingestion import (
    INGESTION_RULES,
    CheckpointCorruptError,
    CheckpointMismatchError,
    ChecksumMismatchError,
    EmbeddingNotAssignedError,
    blob_checksum,
    build_embeddings_at_ingestion,
    default_checkpoint_path,
    embedding_ingestion_live_problems,
    embedding_ingestion_problems,
    embedding_plan,
    estate_digest,
    row_checksums,
)
from thoughtforge.etl.schema import get_connection, initialize_schema


def _fixture_store(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    db = base / "fixture.db"
    initialize_schema(db)
    conn = get_connection(db)
    try:
        for i in range(1, 6):
            conn.execute(
                "INSERT INTO entities (qid, label_en, description_en) "
                "VALUES (?, ?, ?)",
                (f"Q{i}", f"lantern_{i}", f"a synthetic light {i}"),
            )
        for j in range(1, 4):
            conn.execute(
                "INSERT INTO reference_chunks (source_file, chunk_index, "
                "title, content) VALUES (?, ?, ?, ?)",
                ("fixture.md", j, f"chunk_{j}", f"synthetic body {j}"),
            )
        conn.commit()
    finally:
        conn.close()
    return db


def _fake_encoder(fail_on_call: int | None = None):
    import hashlib

    calls = {"n": 0}

    def encode(texts, model_name):
        calls["n"] += 1
        if fail_on_call is not None and calls["n"] == fail_on_call:
            raise RuntimeError("encoder interrupted (fixture)")
        vectors = []
        for text in texts:
            seed = int(
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(384, dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

    return encode


class TestPlan:
    def test_rules_are_unique_data(self) -> None:
        ids = [r["id"] for r in INGESTION_RULES]
        assert len(ids) == len(set(ids))
        assert "not_assigned_means_no_build" in ids
        assert "every_batch_is_checksummed_and_read_back" in ids
        assert "resume_is_by_fingerprint" in ids

    def test_every_profile_plan_matches_tf053(self) -> None:
        for pid, assignment in PROFILE_EMBEDDINGS.items():
            plan = embedding_plan(pid)
            if assignment["model"] is None:
                assert plan.model is None
            else:
                assert plan.model == canonical_model_name(
                    assignment["model"])
                assert plan.entity_batch_size == assignment["batch_size"]
                assert plan.reference_batch_size == assignment["batch_size"]
            assert plan.batch_source.startswith(f"profile:{pid}")

    def test_phone_low_plan_is_its_assignment(self) -> None:
        plan = embedding_plan("phone_low")
        assert plan.model == "all-MiniLM-L6-v2"
        assert plan.entity_batch_size == 16
        assert plan.reference_batch_size == 16

    def test_no_profile_keeps_the_builders_defaults(self) -> None:
        plan = embedding_plan()
        assert plan.model == canonical_model_name(embeddings.DEFAULT_MODEL)
        assert (plan.entity_batch_size, plan.reference_batch_size) == (128, 64)
        assert plan.batch_source.startswith("default:")

    def test_explicit_overrides_win_and_are_recorded(self) -> None:
        plan = embedding_plan("phone_low", batch_size=4)
        assert plan.entity_batch_size == 4
        assert plan.batch_source == "explicit:batch_size"
        named = embedding_plan(model_name="sentence-transformers/"
                                         "all-MiniLM-L6-v2")
        assert named.model == "all-MiniLM-L6-v2"
        assert named.model_source == "explicit:model_name"

    def test_unknown_profile_raises_by_name(self) -> None:
        with pytest.raises(ValueError, match="neptune_ufo"):
            embedding_plan("neptune_ufo")

    def test_bad_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            embedding_plan(batch_size=0)


class TestDriver:
    def test_full_run_builds_everything_checksummed(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        result = build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        assert (result.entities_built, result.reference_built) == (5, 3)
        assert result.resumed is False
        assert result.from_completed_checkpoint is False
        assert result.stored_problems == []
        sums = row_checksums(db, result.plan["model"])
        assert len(sums["embeddings"]) == 5
        assert len(sums["reference_embeddings"]) == 3
        assert estate_digest(sums) == result.estate_digest
        record = json.loads(
            default_checkpoint_path(db).read_text(encoding="utf-8"))
        assert record["status"] == "complete"
        assert record["estate_digest"] == result.estate_digest

    def test_profile_run_uses_the_assignment_batch(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        result = build_embeddings_at_ingestion(
            db, profile_id="pi_5", encoder=_fake_encoder())
        assert result.plan["entity_batch_size"] == 32
        # 5 entities + 3 chunks at batch 32: one batch per leg.
        assert result.batches == 2
        assert (result.entities_built, result.reference_built) == (5, 3)

    def test_completed_checkpoint_is_the_result(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        again = build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder(fail_on_call=1))
        assert again.from_completed_checkpoint is True
        assert again.resumed is True
        assert (again.entities_built, again.reference_built) == (5, 3)

    def test_changed_ask_refuses_by_name(self, tmp_path: Path) -> None:
        db = _fixture_store(tmp_path)
        build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        with pytest.raises(CheckpointMismatchError,
                           match="entity_batch_size"):
            build_embeddings_at_ingestion(
                db, batch_size=4, encoder=_fake_encoder())

    def test_corrupt_checkpoint_refuses_by_name(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        default_checkpoint_path(db).write_text("{ not json",
                                               encoding="utf-8")
        with pytest.raises(CheckpointCorruptError, match="unreadable"):
            build_embeddings_at_ingestion(
                db, batch_size=2, encoder=_fake_encoder())

    def test_resume_false_discards_and_rebuilds_nothing_twice(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        result = build_embeddings_at_ingestion(
            db, batch_size=4, resume=False, encoder=_fake_encoder())
        assert result.resumed is False
        # Per-model only-missing: the rows already stand.
        assert (result.entities_built, result.reference_built) == (0, 0)

    def test_interruption_stops_at_a_batch_edge_and_resume_finishes(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        with pytest.raises(RuntimeError, match="interrupted"):
            build_embeddings_at_ingestion(
                db, batch_size=2, encoder=_fake_encoder(fail_on_call=2))
        partial = row_checksums(db, embeddings.DEFAULT_MODEL)
        assert len(partial["embeddings"]) == 2
        record = json.loads(
            default_checkpoint_path(db).read_text(encoding="utf-8"))
        assert record["status"] == "in_progress"
        finished = build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        assert finished.resumed is True
        assert (finished.entities_built, finished.reference_built) == (5, 3)

    def test_tampered_recorded_row_is_caught_on_resume(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        conn = get_connection(db)
        try:
            blob = conn.execute(
                "SELECT embedding FROM embeddings WHERE qid = 'Q1'"
            ).fetchone()[0]
            conn.execute(
                "UPDATE embeddings SET embedding = ? WHERE qid = 'Q1'",
                (bytes([blob[0] ^ 0xFF]) + blob[1:],),
            )
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(ChecksumMismatchError, match="Q1"):
            build_embeddings_at_ingestion(
                db, batch_size=2, encoder=_fake_encoder())

    def test_pi_zero_refuses_and_writes_nothing(
        self, tmp_path: Path,
    ) -> None:
        db = _fixture_store(tmp_path)
        with pytest.raises(EmbeddingNotAssignedError, match="pi_zero"):
            build_embeddings_at_ingestion(
                db, profile_id="pi_zero", encoder=_fake_encoder())
        assert not default_checkpoint_path(db).exists()
        sums = row_checksums(db, embeddings.DEFAULT_MODEL)
        assert sums == {"embeddings": {}, "reference_embeddings": {}}

    def test_blob_checksum_is_sha256(self) -> None:
        import hashlib

        assert blob_checksum(b"abc") == hashlib.sha256(b"abc").hexdigest()


class TestBuildersPerModel:
    """D-EMB-3's regression pin, through the real builders."""

    @pytest.fixture()
    def fake_st(self, monkeypatch):
        def unit(texts):
            return np.stack([
                np.eye(384, dtype=np.float32)[i % 384] for i, _ in
                enumerate(texts)
            ]) if texts else np.zeros((0, 384), dtype=np.float32)

        class FakeSentenceTransformer:
            def __init__(self, name: str) -> None:
                self.name = name

            def encode(self, texts, **kwargs):
                vectors = []
                for text in texts:
                    vec = np.zeros(384, dtype=np.float32)
                    vec[len(text) % 384] = 1.0
                    vectors.append(vec)
                return np.stack(vectors)

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        embeddings._MODEL_CACHE.clear()
        yield
        embeddings._MODEL_CACHE.clear()

    def test_second_model_builds_what_the_first_built(
        self, tmp_path: Path, fake_st,
    ) -> None:
        db = _fixture_store(tmp_path / "builders")
        first = embeddings.build_entity_embeddings(db_path=db)
        assert first == 5
        # Before TF-111 this returned 0: the bare join saw the
        # rows as already embedded, under ANY model.
        second = embeddings.build_entity_embeddings(
            db_path=db, model_name="some-other-model")
        assert second == 5
        # A re-run under the SAME model builds nothing.
        assert embeddings.build_entity_embeddings(
            db_path=db, model_name="some-other-model") == 0
        # D-EMB-4 (filed, OPEN -> TF-177): storage holds one
        # model per row (the PK is the qid alone), so the
        # second model's build REPLACED the first model's
        # rows — building under the first model again
        # rebuilds them. Coexistence needs the composite-
        # key migration; it is not claimed here.
        assert embeddings.build_entity_embeddings(db_path=db) == 5

    def test_reference_builder_is_per_model_too(
        self, tmp_path: Path, fake_st,
    ) -> None:
        db = _fixture_store(tmp_path / "builders_ref")
        assert embeddings.build_reference_embeddings(db_path=db) == 3
        assert embeddings.build_reference_embeddings(
            db_path=db, model_name="some-other-model") == 3
        assert embeddings.build_reference_embeddings(
            db_path=db, model_name="some-other-model") == 0
        # D-EMB-4, as above: the first model's rows were
        # replaced, so it rebuilds.
        assert embeddings.build_reference_embeddings(db_path=db) == 3


class TestCheckers:
    def test_static_checker_clean(self) -> None:
        assert embedding_ingestion_problems() == []

    def test_live_checker_clean(self, tmp_path: Path) -> None:
        assert embedding_ingestion_live_problems(tmp_path / "live") == []
