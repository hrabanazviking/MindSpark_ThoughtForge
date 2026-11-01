"""
Embedding generation pipeline for ThoughtForge.

Generates sentence-transformer embeddings for all knowledge sources
and stores them as BLOBs in the SQLite knowledge database.
Uses all-MiniLM-L6-v2 by default (fast, <100MB, CPU-friendly).

Vector similarity is computed via cosine similarity on numpy float32 arrays.
Optional: if sqlite-vss is installed, uses it for ANN search; otherwise
falls back to brute-force cosine similarity.

TF-053 (embedding discipline): model names are normalised through
``etl/embedding_discipline.canonical_model_name`` at every entry
point below — the registry there holds the per-profile assignment,
the recorded licence, and the offline-availability proof. The
scalar ``cosine_similarity`` that used to live here was removed by
TF-053 (docket D-DEAD-2): it had no caller anywhere, while
``batch_cosine_similarity`` is the live path.

TF-054 (vector store shape): the builders below gate every
vector through ``etl/vector_store.validated_blob`` and write on
connections hardened to ``synchronous = FULL``
(``harden_write_connection`` — the Samtal lesson), and both
searches load their rows through ``load_searchable_rows``, so
one corrupt stored blob is skipped by row instead of poisoning
the whole search (docket D-VEC-1 / D-VEC-2).

TF-111 (embedding at ingestion): both builders' only-missing
selection joins on ``model_name`` as well as the row id, so
"missing" means missing *under the model being built* — before
this fix, a row embedded under one model silently blocked
building it under another, while the searches (which do filter
by model) found nothing for the second model (docket D-EMB-3).
The per-profile, resumable, checksummed ingestion driver that
consumes these builders' contract lives in
``etl/embedding_ingestion.py``.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np

from thoughtforge.etl.embedding_discipline import canonical_model_name
from thoughtforge.etl.schema import get_connection
from thoughtforge.etl.vector_store import (
    harden_write_connection,
    load_searchable_rows,
    validated_blob,
)
from thoughtforge.utils.paths import get_knowledge_db_path

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dimension


def _get_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Load and cache a SentenceTransformer model."""
    model_name = canonical_model_name(model_name)
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", model_name)
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info("Model loaded: %s", model_name)
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from e
    return _MODEL_CACHE[model_name]


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized float32 embeddings.
    Shape: (len(texts), EMBEDDING_DIM)
    """
    model = _get_model(model_name)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_single(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode a single text. Returns shape (EMBEDDING_DIM,)."""
    return encode_texts([text], model_name)[0]


def to_blob(embedding: np.ndarray) -> bytes:
    """Convert float32 numpy array to SQLite BLOB (little-endian bytes)."""
    return embedding.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB back to float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix of vectors.
    query: (D,), matrix: (N, D)
    Returns: (N,) similarity scores.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    q_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = q_norm * row_norms
    denom = np.where(denom == 0, 1e-9, denom)
    return (matrix @ query) / denom


# ── Entity embedding generation ───────────────────────────────────────────────

def build_entity_embeddings(
    db_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    limit: int | None = None,
) -> int:
    """
    Generate embeddings for all entities that don't have one yet.
    Text used: "label_en: description_en"
    Returns count of embeddings generated.
    """
    model_name = canonical_model_name(model_name)
    db_path = db_path or get_knowledge_db_path()
    conn = get_connection(db_path)
    harden_write_connection(conn)
    total = 0

    try:
        query = """
            SELECT e.qid, e.label_en, e.description_en
            FROM entities e
            LEFT JOIN embeddings em ON e.qid = em.qid
                AND em.model_name = ?
            WHERE em.qid IS NULL
            AND e.label_en IS NOT NULL
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, (model_name,))
        rows = cursor.fetchall()

        if not rows:
            logger.info("No entities without embeddings found")
            return 0

        logger.info("Generating embeddings for %d entities", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = []
            qids = []
            for row in batch:
                label = row["label_en"] or ""
                desc = row["description_en"] or ""
                texts.append(f"{label}: {desc}" if desc else label)
                qids.append(row["qid"])

            embeddings = encode_texts(texts, model_name)

            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (qid, text_for_embedding, embedding, model_name) "
                "VALUES (?, ?, ?, ?)",
                [
                    (qids[j], texts[j], validated_blob(embeddings[j]), model_name)
                    for j in range(len(batch))
                ],
            )
            conn.commit()
            total += len(batch)
            if total % 10000 == 0:
                logger.info("  %d entity embeddings committed", total)

    finally:
        conn.close()

    logger.info("Entity embedding generation complete: %d embeddings", total)
    return total


def build_reference_embeddings(
    db_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> int:
    """Generate embeddings for all reference_chunks without one."""
    model_name = canonical_model_name(model_name)
    db_path = db_path or get_knowledge_db_path()
    conn = get_connection(db_path)
    harden_write_connection(conn)
    total = 0

    try:
        rows = conn.execute("""
            SELECT rc.chunk_id, rc.title, rc.content
            FROM reference_chunks rc
            LEFT JOIN reference_embeddings re ON rc.chunk_id = re.chunk_id
                AND re.model_name = ?
            WHERE re.chunk_id IS NULL
        """, (model_name,)).fetchall()

        if not rows:
            logger.info("No reference chunks without embeddings found")
            return 0

        logger.info("Generating embeddings for %d reference chunks", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [
                f"{r['title']}: {r['content'][:500]}" if r["title"] else r["content"][:500]
                for r in batch
            ]
            chunk_ids = [r["chunk_id"] for r in batch]
            embeddings = encode_texts(texts, model_name)

            conn.executemany(
                "INSERT OR REPLACE INTO reference_embeddings (chunk_id, embedding, model_name) "
                "VALUES (?, ?, ?)",
                [(chunk_ids[j], validated_blob(embeddings[j]), model_name) for j in range(len(batch))],
            )
            conn.commit()
            total += len(batch)

    finally:
        conn.close()

    logger.info("Reference embedding generation complete: %d embeddings", total)
    return total


# ── Vector search ──────────────────────────────────────────────────────────────

def vector_search_entities(
    query_text: str,
    db_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Brute-force cosine similarity search over entity embeddings.
    Returns top_k entities sorted by similarity score descending.
    """
    model_name = canonical_model_name(model_name)
    db_path = db_path or get_knowledge_db_path()
    query_vec = encode_single(query_text, model_name)

    conn = get_connection(db_path, read_only=True)
    results: list[dict[str, Any]] = []

    try:
        rows = conn.execute(
            "SELECT em.qid, em.embedding, e.label_en, e.description_en "
            "FROM embeddings em JOIN entities e ON em.qid = e.qid "
            "WHERE em.model_name = ?",
            (model_name,),
        ).fetchall()

        if not rows:
            return []

        # TF-054: corrupt rows are skipped by row (logged, named);
        # the good rows are the matrix that gets scored.
        kept, matrix, _skipped = load_searchable_rows(
            rows, lambda r: r["qid"], lambda r: r["embedding"],
            table="embeddings",
        )
        if matrix is None:
            return []

        qids = [r["qid"] for r in kept]
        labels = [r["label_en"] for r in kept]
        descs = [r["description_en"] for r in kept]

        scores = batch_cosine_similarity(query_vec, matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append({
                "qid": qids[idx],
                "label_en": labels[idx],
                "description_en": descs[idx],
                "similarity": round(score, 4),
                "source": "entity_embedding",
            })

    finally:
        conn.close()

    return results


def vector_search_reference(
    query_text: str,
    db_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Cosine similarity search over built-in reference chunk embeddings."""
    model_name = canonical_model_name(model_name)
    db_path = db_path or get_knowledge_db_path()
    query_vec = encode_single(query_text, model_name)

    conn = get_connection(db_path, read_only=True)
    results: list[dict[str, Any]] = []

    try:
        rows = conn.execute(
            "SELECT re.chunk_id, re.embedding, rc.source_file, rc.title, rc.content "
            "FROM reference_embeddings re JOIN reference_chunks rc ON re.chunk_id = rc.chunk_id "
            "WHERE re.model_name = ?",
            (model_name,),
        ).fetchall()

        if not rows:
            return []

        # TF-054: same loader as the entity search — corrupt
        # reference rows are skipped by row, never stacked.
        kept, matrix, _skipped = load_searchable_rows(
            rows, lambda r: r["chunk_id"], lambda r: r["embedding"],
            table="reference_embeddings",
        )
        if matrix is None:
            return []

        chunk_ids = [r["chunk_id"] for r in kept]
        files = [r["source_file"] for r in kept]
        titles = [r["title"] for r in kept]
        contents = [r["content"] for r in kept]

        scores = batch_cosine_similarity(query_vec, matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append({
                "chunk_id": chunk_ids[idx],
                "source_file": files[idx],
                "title": titles[idx],
                "content": contents[idx][:500],
                "similarity": round(score, 4),
                "source": "reference_chunk",
            })

    finally:
        conn.close()

    return results
