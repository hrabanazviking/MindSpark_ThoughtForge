"""TF-053 — embedding discipline: models per profile, offline proof.

Pins etl/embedding_discipline.py against the live tree: the
model registry (dimension tied to EMBEDDING_DIM, licence
recorded), the canonical name fold and its wiring into the
ETL loader / builders / searches and the forge (the split
spelling can no longer split the cache or the DB), the
per-profile assignment (ONNX for phone_low, none for
pi_zero), the offline availability report's verdicts under
fabricated machine facts, the forge's degrade-to-SQL on a
dead vector leg, and the stored-estate dimension check.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl import embeddings
from thoughtforge.etl.embedding_discipline import (
    AVAILABILITY_STATUSES,
    EMBEDDING_MODELS,
    PROFILE_EMBEDDINGS,
    RUNTIME_NONE,
    RUNTIME_ONNX,
    RUNTIME_SENTENCE_TRANSFORMERS,
    STATUS_AVAILABLE,
    STATUS_NOT_ASSIGNED,
    STATUS_NOT_CACHED,
    STATUS_NO_ONNX,
    STATUS_PACKAGE_MISSING,
    availability_report,
    canonical_model_name,
    embedding_assignment,
    embedding_problems,
    model_cache_status,
    onnx_artefact_status,
    package_installed,
    stored_embeddings_problems,
)
from thoughtforge.etl.schema import get_connection, initialize_schema
from thoughtforge.inference import onnx_export
from thoughtforge.knowledge import forge as forge_module
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.retrieval_precision import build_fixture_db
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)
from thoughtforge.utils.paths import get_embeddings_dir

CANONICAL = "all-MiniLM-L6-v2"
HUB_ID = "sentence-transformers/all-MiniLM-L6-v2"


def unit_vector() -> np.ndarray:
    vec = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
    vec[0] = 1.0
    return vec


class FakeSentenceTransformer:
    constructed: list[str] = []

    def __init__(self, name: str) -> None:
        FakeSentenceTransformer.constructed.append(name)
        self.name = name

    def encode(self, texts, **kwargs):
        return np.stack([unit_vector() for _ in texts])


@pytest.fixture()
def fake_st(monkeypatch):
    FakeSentenceTransformer.constructed = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embeddings._MODEL_CACHE.clear()
    yield FakeSentenceTransformer
    embeddings._MODEL_CACHE.clear()


def make_db(tmp_path: Path) -> Path:
    db = tmp_path / "k.db"
    initialize_schema(db)
    return db


def insert_entity_with_embedding(db: Path, model_name: str,
                                 blob: bytes | None = None) -> None:
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en) "
            "VALUES (?, ?, ?)",
            ("Q53", "Ferry", "a boat that crosses the river"),
        )
        conn.execute(
            "INSERT INTO embeddings (qid, text_for_embedding, "
            "embedding, model_name) VALUES (?, ?, ?, ?)",
            ("Q53", "Ferry: a boat",
             unit_vector().tobytes() if blob is None else blob,
             model_name),
        )
        conn.commit()
    finally:
        conn.close()


# ── The registry and the canonical fold ───────────────────────────────────────

def test_registry_holds_exactly_the_trees_model():
    assert sorted(EMBEDDING_MODELS) == [CANONICAL]
    row = EMBEDDING_MODELS[CANONICAL]
    assert row["hub_id"] == HUB_ID
    assert row["dimension"] == embeddings.EMBEDDING_DIM == 384


def test_registry_records_the_models_licence():
    row = EMBEDDING_MODELS[CANONICAL]
    assert row["licence"] == "Apache-2.0"
    assert "model card" in row["licence_source"]


def test_canonical_fold_maps_every_spelling():
    assert canonical_model_name(HUB_ID) == CANONICAL
    assert canonical_model_name(CANONICAL) == CANONICAL
    assert canonical_model_name("some-org/all-MiniLM-L6-v2") == CANONICAL


def test_canonical_fold_passes_unknowns_and_paths_through(tmp_path):
    assert canonical_model_name("some-other-model") == "some-other-model"
    local = tmp_path / "all-MiniLM-L6-v2"
    local.mkdir()
    assert canonical_model_name(str(local)) == str(local)


# ── The per-profile assignment ────────────────────────────────────────────────

def test_assignment_covers_exactly_the_shipped_profiles():
    assert sorted(PROFILE_EMBEDDINGS) == sorted(list_hardware_profiles())


def test_assignment_models_and_runtimes():
    for pid in ("server_gpu", "desktop_gpu", "desktop_cpu", "pi_5"):
        assignment = embedding_assignment(pid)
        assert assignment["model"] == CANONICAL
        assert assignment["runtime"] == RUNTIME_SENTENCE_TRANSFORMERS
    phone = embedding_assignment("phone_low")
    assert phone["model"] == CANONICAL
    assert phone["runtime"] == RUNTIME_ONNX
    zero = embedding_assignment("pi_zero")
    assert zero["model"] is None
    assert zero["runtime"] == RUNTIME_NONE
    assert zero["batch_size"] == 0


def test_assignment_batch_sizes_step_down_with_ram():
    sizes = {pid: embedding_assignment(pid)["batch_size"]
             for pid in PROFILE_EMBEDDINGS}
    assert sizes == {
        "server_gpu": 128, "desktop_gpu": 128, "desktop_cpu": 64,
        "pi_5": 32, "phone_low": 16, "pi_zero": 0,
    }
    assert all(embedding_assignment(pid)["fallback"] == "sql"
               for pid in PROFILE_EMBEDDINGS)


def test_phone_low_assignment_matches_its_own_profile_file():
    profile = load_hardware_profile("phone_low")
    assert profile["deployment"]["onnx_export"] is True


def test_unknown_profile_raises_naming_the_assigned():
    with pytest.raises(ValueError, match="pi_5"):
        embedding_assignment("toaster_9000")


# ── Loader / builder / search wiring (fake sentence-transformers) ────────────

def test_loader_caches_under_the_canonical_name(fake_st):
    model = embeddings._get_model(HUB_ID)
    assert fake_st.constructed == [CANONICAL]
    again = embeddings._get_model(CANONICAL)
    assert again is model
    assert fake_st.constructed == [CANONICAL]  # one construction only


def test_vector_search_finds_rows_across_spellings(tmp_path, fake_st):
    db = make_db(tmp_path)
    insert_entity_with_embedding(db, CANONICAL)
    results = embeddings.vector_search_entities(
        "ferry", db_path=db, model_name=HUB_ID, min_score=0.3)
    assert [r["qid"] for r in results] == ["Q53"]


def test_builder_stores_the_canonical_name(tmp_path, fake_st):
    db = make_db(tmp_path)
    conn = get_connection(db)
    try:
        conn.execute(
            "INSERT INTO entities (qid, label_en, description_en) "
            "VALUES (?, ?, ?)", ("Q54", "Harbour", "where boats rest"))
        conn.commit()
    finally:
        conn.close()
    built = embeddings.build_entity_embeddings(db_path=db, model_name=HUB_ID)
    assert built == 1
    conn = get_connection(db, read_only=True)
    try:
        rows = conn.execute("SELECT model_name FROM embeddings").fetchall()
    finally:
        conn.close()
    assert [r["model_name"] for r in rows] == [CANONICAL]


def test_forge_normalises_its_model_name():
    forge = KnowledgeForge(model_name=HUB_ID)
    assert forge.model_name == CANONICAL


# ── Offline availability, proven per profile ─────────────────────────────────

def fabricated_hub_cache(tmp_path: Path) -> Path:
    snapshot = (tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
                / "snapshots" / "rev1")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return tmp_path


def test_availability_package_missing():
    report = availability_report("desktop_cpu", package=False,
                                 roots=[Path("/nonexistent-tf053")])
    assert report.status == STATUS_PACKAGE_MISSING
    assert report.available is False
    assert report.package_installed is False


def test_availability_installed_but_not_cached(tmp_path):
    report = availability_report("desktop_cpu", package=True,
                                 roots=[tmp_path])
    assert report.status == STATUS_NOT_CACHED
    assert report.model_cached is False


def test_availability_installed_and_cached(tmp_path):
    root = fabricated_hub_cache(tmp_path)
    report = availability_report("pi_5", package=True, roots=[root])
    assert report.status == STATUS_AVAILABLE
    assert report.available is True
    assert report.cache_location is not None
    assert "rev1" in report.cache_location


def test_model_cache_status_recognises_the_legacy_layout(tmp_path):
    folder = tmp_path / "sentence-transformers_all-MiniLM-L6-v2"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "pytorch_model.bin").write_bytes(b"weights")
    status = model_cache_status(CANONICAL, roots=[tmp_path])
    assert status["cached"] is True
    empty = model_cache_status(CANONICAL, roots=[tmp_path / "nope"])
    assert empty["cached"] is False


def test_availability_phone_low_needs_the_onnx_artefact(tmp_path):
    missing = availability_report("phone_low", package=True,
                                  onnx_dir=tmp_path)
    assert missing.status == STATUS_NO_ONNX
    assert missing.onnx_present is False
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    present = availability_report("phone_low", package=True,
                                  onnx_dir=tmp_path)
    assert present.status == STATUS_AVAILABLE
    assert present.onnx_present is True


def test_availability_pi_zero_is_never_assigned(tmp_path):
    root = fabricated_hub_cache(tmp_path)
    report = availability_report("pi_zero", package=True, roots=[root])
    assert report.status == STATUS_NOT_ASSIGNED
    assert report.available is False
    assert report.model is None


def test_availability_report_dict_shape():
    report = availability_report("server_gpu", package=False)
    as_dict = report.to_dict()
    assert as_dict["status"] in AVAILABILITY_STATUSES
    assert as_dict["available"] is False
    assert as_dict["profile_id"] == "server_gpu"
    assert as_dict["reasons"]


def test_onnx_home_is_the_paths_helpers_embeddings_dir(tmp_path):
    status = onnx_artefact_status()
    assert status["dir"] == str(get_embeddings_dir())
    local = onnx_artefact_status(tmp_path)
    assert local["present"] is False
    (tmp_path / "model_quantized.onnx").write_bytes(b"q")
    assert onnx_artefact_status(tmp_path)["file"] == "model_quantized.onnx"


# ── The forge degrades; the SQL leg stands ────────────────────────────────────

def test_hybrid_retrieval_degrades_when_the_vector_leg_dies(
        tmp_path, monkeypatch):
    db = build_fixture_db(tmp_path / "fixture.db")
    forge = KnowledgeForge(db_path=db)

    def _raising(*args, **kwargs):
        raise RuntimeError("embedding backend unavailable (test)")

    monkeypatch.setattr(forge_module, "vector_search_entities", _raising)
    bundle = forge.retrieve("ferry", path="hybrid")
    assert bundle.total_retrieved >= 1


@pytest.mark.skipif(
    package_installed("sentence_transformers"),
    reason="needs the package absent to prove the live degrade (D-ENV-1)",
)
def test_live_vector_leg_degrades_on_this_vm(tmp_path):
    db = make_db(tmp_path)
    forge = KnowledgeForge(db_path=db)
    assert forge.vector_retrieve("ferry") == []
    bundle = forge.retrieve("ferry", path="hybrid")
    assert bundle.total_retrieved == 0  # empty DB, but no raise


# ── The stored estate, checked against the registry ───────────────────────────

def test_stored_problems_clean_db(tmp_path):
    db = make_db(tmp_path)
    insert_entity_with_embedding(db, CANONICAL)
    assert stored_embeddings_problems(db) == []


def test_stored_problems_name_alias_filed_rows(tmp_path):
    db = make_db(tmp_path)
    insert_entity_with_embedding(db, HUB_ID)
    problems = stored_embeddings_problems(db)
    assert len(problems) == 1
    assert "alias" in problems[0] and HUB_ID in problems[0]


def test_stored_problems_name_wrong_dimension_blobs(tmp_path):
    db = make_db(tmp_path)
    insert_entity_with_embedding(
        db, CANONICAL, blob=np.zeros(10, dtype=np.float32).tobytes())
    problems = stored_embeddings_problems(db)
    assert len(problems) == 1
    assert "40 bytes" in problems[0] and "1536" in problems[0]


# ── Docket resolutions and the live checker ───────────────────────────────────

def test_dead_scalar_cosine_retired_batch_sibling_live():
    assert not hasattr(embeddings, "cosine_similarity")
    unit = np.array([1.0, 0.0], dtype=np.float32)
    scores = embeddings.batch_cosine_similarity(unit, unit)
    assert float(scores[0]) == 1.0


def test_onnx_exports_default_is_the_same_model():
    assert onnx_export._DEFAULT_MODEL == HUB_ID
    assert canonical_model_name(onnx_export._DEFAULT_MODEL) == CANONICAL


def test_embedding_problems_live_checker():
    assert embedding_problems() == []
