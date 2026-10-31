"""TF-054 — vector store shape: restarts, atomic writes, corruption.

Pins etl/vector_store.py against the live tree: the shape as
data (two tables in the knowledge DB, brute-force scan, blob
format tied to the TF-053 registry), blob classification,
restart survival (committed survives a fresh process,
uncommitted rolls back), the builders' validated gate and
synchronous=FULL hardening, per-batch atomicity, search-time
isolation of every corrupt-row class, inspection as a
never-raising report, and quarantine as the explicit repair.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl import embeddings
from thoughtforge.etl.schema import get_connection, initialize_schema
from thoughtforge.etl.vector_store import (
    BLOB_CLASSES,
    CLASS_EMPTY,
    CLASS_NON_FINITE,
    CLASS_NULL,
    CLASS_OK,
    CLASS_WRONG_BYTES,
    EXPECTED_BLOB_BYTES,
    EXPECTED_DIMENSION,
    VECTOR_STORE_SHAPE,
    VECTOR_TABLES,
    classify_blob,
    harden_write_connection,
    inspect_vector_store,
    load_searchable_rows,
    quarantine_corrupt_rows,
    validated_blob,
    vector_store_problems,
)

CANONICAL = "all-MiniLM-L6-v2"


def unit(dim: int = EXPECTED_DIMENSION) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0
    return vec


def make_db(tmp_path: Path) -> Path:
    db = tmp_path / "k.db"
    initialize_schema(db)
    return db


def insert_entity(db: Path, qid: str, blob) -> None:
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en) "
            "VALUES (?, ?, ?)", (qid, qid, "desc"))
        conn.execute(
            "INSERT INTO embeddings (qid, text_for_embedding, "
            "embedding, model_name) VALUES (?, ?, ?, ?)",
            (qid, qid, blob, CANONICAL))
        conn.commit()
    finally:
        conn.close()


class FakeST:
    def __init__(self, dim: int = EXPECTED_DIMENSION, fail_on: int = -1):
        self.dim = dim
        self.calls = 0
        self.fail_on = fail_on

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("boom mid-build")
        return np.stack([unit(self.dim) for _ in texts])


@pytest.fixture()
def fake_model():
    holder = {}

    def install(model):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = lambda name: model
        sys.modules["sentence_transformers"] = module
        embeddings._MODEL_CACHE.clear()
        holder["model"] = model
        return model

    yield install
    embeddings._MODEL_CACHE.clear()
    sys.modules.pop("sentence_transformers", None)


def search_entities(db: Path):
    return embeddings.vector_search_entities(
        "ferry", db_path=db, min_score=0.0)


# ── The shape, as data ────────────────────────────────────────────────────────

def test_shape_names_exactly_two_tables():
    assert [(t["table"], t["key_column"]) for t in VECTOR_TABLES] == [
        ("embeddings", "qid"), ("reference_embeddings", "chunk_id")]


def test_shape_blob_format_tied_to_registry():
    assert EXPECTED_DIMENSION == embeddings.EMBEDDING_DIM == 384
    assert EXPECTED_BLOB_BYTES == 1536
    assert VECTOR_STORE_SHAPE["dimension"] == 384
    assert "brute-force" in VECTOR_STORE_SHAPE["index_structure"]
    assert VECTOR_STORE_SHAPE["write_synchronous"].startswith("FULL")


# ── Blob classification ───────────────────────────────────────────────────────

def test_classify_ok_and_zero_vector():
    assert classify_blob(unit().tobytes()) == CLASS_OK
    assert classify_blob(np.zeros(384, dtype=np.float32).tobytes()) == CLASS_OK


@pytest.mark.parametrize("blob,expected", [
    (None, CLASS_NULL),
    (b"", CLASS_EMPTY),
    (np.zeros(10, dtype=np.float32).tobytes(), CLASS_WRONG_BYTES),
    (b"x" * 1537, CLASS_WRONG_BYTES),
    (b"\xff" * 1536, CLASS_NON_FINITE),
])
def test_classify_corrupt_classes(blob, expected):
    assert classify_blob(blob) == expected
    assert expected in BLOB_CLASSES


def test_classify_nan_vector_is_non_finite():
    bad = unit()
    bad[3] = float("nan")
    assert classify_blob(bad.tobytes()) == CLASS_NON_FINITE
    bad[3] = float("inf")
    assert classify_blob(bad.tobytes()) == CLASS_NON_FINITE


# ── The builders' gate ────────────────────────────────────────────────────────

def test_validated_blob_round_trip():
    assert validated_blob(unit()) == unit().tobytes()


def test_validated_blob_refuses_wrong_dimension():
    with pytest.raises(ValueError, match="shape"):
        validated_blob(np.zeros(10, dtype=np.float32))


def test_validated_blob_refuses_non_finite():
    bad = unit()
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validated_blob(bad)


def test_builder_refuses_wrong_dimension_model(tmp_path, fake_model):
    db = make_db(tmp_path)
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en) "
            "VALUES ('Q9', 'X', 'y')")
        conn.commit()
    finally:
        conn.close()
    fake_model(FakeST(dim=10))
    with pytest.raises(ValueError):
        embeddings.build_entity_embeddings(db_path=db)
    conn = get_connection(db, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    finally:
        conn.close()


def test_builder_writes_on_a_full_synchronous_connection(tmp_path, fake_model):
    db = make_db(tmp_path)
    conn = get_connection(db)
    harden_write_connection(conn)
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    conn.close()
    fake_model(FakeST())
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en) "
            "VALUES ('Q1', 'Ferry', 'boat')")
        conn.commit()
    finally:
        conn.close()
    assert embeddings.build_entity_embeddings(db_path=db) == 1


def test_mid_build_crash_leaves_earlier_batches_and_resumes(tmp_path, fake_model):
    db = make_db(tmp_path)
    conn = get_connection(db)
    try:
        for i in range(3):
            conn.execute(
                "INSERT INTO entities (qid, label_en, description_en) "
                "VALUES (?, ?, ?)", (f"Q{i}", f"E{i}", "d"))
        conn.commit()
    finally:
        conn.close()
    fake_model(FakeST(fail_on=2))
    with pytest.raises(RuntimeError):
        embeddings.build_entity_embeddings(db_path=db, batch_size=1)
    conn = get_connection(db, read_only=True)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    finally:
        conn.close()
    # Resume with a healthy model: only the missing rows are built.
    fake_model(FakeST())
    assert embeddings.build_entity_embeddings(db_path=db, batch_size=1) == 2


# ── Restart survival ──────────────────────────────────────────────────────────

def test_committed_survives_reopen_uncommitted_rolls_back(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en) VALUES ('Q2', 'Ghost')")
        conn.execute(
            "INSERT INTO embeddings (qid, embedding, model_name) "
            "VALUES ('Q2', ?, ?)", (unit().tobytes(), CANONICAL))
        # no commit — a crash here must roll this back
    finally:
        conn.close()
    report = inspect_vector_store(db)
    assert report.counts["embeddings"]["total"] == 1


def test_store_survives_a_fresh_process(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    src = Path(embeddings.__file__).resolve().parents[2]
    code = (
        "from pathlib import Path;"
        "from thoughtforge.etl.vector_store import inspect_vector_store;"
        f"r = inspect_vector_store(Path({str(db)!r}));"
        "print(r.counts['embeddings']['total'], r.integrity, r.ok)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1 ok True"


# ── Search-time isolation of corrupt rows ─────────────────────────────────────

CORRUPT_BLOBS = [
    np.zeros(10, dtype=np.float32).tobytes(),
    None,
    b"",
    b"x" * 1537,
    b"\xff" * 1536,
]


@pytest.mark.parametrize("blob", CORRUPT_BLOBS)
def test_one_corrupt_row_never_poisons_the_search(tmp_path, fake_model, blob):
    db = make_db(tmp_path)
    insert_entity(db, "QGOOD", unit().tobytes())
    insert_entity(db, "QBAD", blob)
    fake_model(FakeST())
    results = search_entities(db)
    assert [r["qid"] for r in results] == ["QGOOD"]
    assert results[0]["similarity"] == 1.0


def test_non_finite_row_never_outranks_a_perfect_match(tmp_path, fake_model):
    db = make_db(tmp_path)
    insert_entity(db, "QGOOD", unit().tobytes())
    insert_entity(db, "QNAN", b"\xff" * 1536)
    fake_model(FakeST())
    results = search_entities(db)
    assert results and results[0]["qid"] == "QGOOD"


def test_all_rows_corrupt_searches_empty_not_raising(tmp_path, fake_model):
    db = make_db(tmp_path)
    insert_entity(db, "QBAD", None)
    fake_model(FakeST())
    assert search_entities(db) == []


def test_reference_search_isolates_corrupt_rows(tmp_path, fake_model):
    db = make_db(tmp_path)
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO reference_chunks (chunk_id, source_file, "
            "chunk_index, title, content) VALUES (1, 'f.md', 0, 'T', 'body')")
        conn.execute(
            "INSERT INTO reference_chunks (chunk_id, source_file, "
            "chunk_index, title, content) VALUES (2, 'f.md', 1, 'T2', 'body')")
        conn.execute(
            "INSERT INTO reference_embeddings (chunk_id, embedding, "
            "model_name) VALUES (1, ?, ?)", (unit().tobytes(), CANONICAL))
        conn.execute(
            "INSERT INTO reference_embeddings (chunk_id, embedding, "
            "model_name) VALUES (2, ?, ?)", (b"", CANONICAL))
        conn.commit()
    finally:
        conn.close()
    fake_model(FakeST())
    results = embeddings.vector_search_reference(
        "ferry", db_path=db, min_score=0.0)
    assert [r["chunk_id"] for r in results] == [1]


def test_loader_skips_are_named():
    rows = [("a", unit().tobytes()), ("b", None), ("c", unit().tobytes())]
    kept, matrix, skipped = load_searchable_rows(
        rows, lambda r: r[0], lambda r: r[1], table="embeddings")
    assert [r[0] for r in kept] == ["a", "c"]
    assert matrix.shape == (2, 384)
    assert skipped == [{"table": "embeddings", "id": "b", "class": CLASS_NULL}]


# ── Inspection ────────────────────────────────────────────────────────────────

def test_inspect_clean_store(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    report = inspect_vector_store(db)
    assert report.ok is True
    assert report.integrity == "ok"
    assert report.counts["embeddings"] == {"total": 1, "ok": 1, "corrupt": 0}
    assert report.to_dict()["ok"] is True


def test_inspect_names_corrupt_rows_by_id_and_class(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    insert_entity(db, "Q2", b"\xff" * 1536)
    insert_entity(db, "Q3", None)
    report = inspect_vector_store(db)
    assert report.ok is False
    named = {(r["id"], r["class"]) for r in report.corrupt_rows}
    assert named == {("Q2", CLASS_NON_FINITE), ("Q3", CLASS_NULL)}
    assert report.counts["embeddings"] == {"total": 3, "ok": 1, "corrupt": 2}


def test_inspect_missing_file_is_a_verdict_not_a_raise(tmp_path):
    report = inspect_vector_store(tmp_path / "nope.db")
    assert report.exists is False
    assert report.readable is False
    assert report.problems


def test_inspect_missing_table_is_named(tmp_path):
    db = make_db(tmp_path)
    conn = get_connection(db)
    try:
        conn.execute("DROP TABLE reference_embeddings")
        conn.commit()
    finally:
        conn.close()
    report = inspect_vector_store(db)
    assert report.tables_missing == ["reference_embeddings"]
    assert any("reference_embeddings" in p for p in report.problems)


def test_inspect_corrupt_file_is_unreadable_not_a_raise(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    data = db.read_bytes()
    db.write_bytes(b"NOTSQLITE" + data[9:])
    report = inspect_vector_store(db)
    assert report.exists is True
    assert report.readable is False
    assert report.integrity == "unreadable"
    assert report.problems


# ── Quarantine, the explicit repair ───────────────────────────────────────────

def test_quarantine_removes_only_corrupt_rows(tmp_path):
    db = make_db(tmp_path)
    good = unit().tobytes()
    insert_entity(db, "Q1", good)
    insert_entity(db, "Q2", b"\xff" * 1536)
    insert_entity(db, "Q3", np.zeros(10, dtype=np.float32).tobytes())
    result = quarantine_corrupt_rows(db)
    assert result.removed_count == 2
    assert {(r["id"], r["class"]) for r in result.removed} == {
        ("Q2", CLASS_NON_FINITE), ("Q3", CLASS_WRONG_BYTES)}
    report = inspect_vector_store(db)
    assert report.ok is True
    conn = get_connection(db, read_only=True)
    try:
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE qid='Q1'").fetchone()
        assert row[0] == good  # the good row is byte-untouched
    finally:
        conn.close()


def test_quarantine_on_clean_store_removes_nothing(tmp_path):
    db = make_db(tmp_path)
    insert_entity(db, "Q1", unit().tobytes())
    assert quarantine_corrupt_rows(db).removed_count == 0
    assert inspect_vector_store(db).ok is True


# ── The live checker ──────────────────────────────────────────────────────────

def test_vector_store_problems_live_checker():
    assert vector_store_problems() == []
