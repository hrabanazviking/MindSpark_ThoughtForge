"""
Vector store shape — restarts, atomic writes, corruption (TF-054).

The roadmap's rule: *the store's index survives restarts
+ corruption fixtures (atomic writes, the Samtal lesson).*

The measured starting state, before this slice (probed
this tick, keeper 07:32, on synthetic stores):

  * **There is no separate vector index at all.** The
    "index" is two tables in the same SQLite knowledge DB
    as everything else — ``embeddings`` (keyed by QID)
    and ``reference_embeddings`` (keyed by chunk id).
    Every vector search loads *every* blob for its model
    out of the DB and brute-force cosine-scores the whole
    matrix in memory. That shape is stated here as data
    (``VECTOR_STORE_SHAPE``); no ANN structure exists and
    none is claimed.
  * **Restart survival was true but unproven and
    unstated.** A committed embedding row survives a
    close/reopen (proven: Q1 committed, Q2 left
    uncommitted — after reopen only Q1 exists; an
    uncommitted row is rolled back, never half-landed).
    Nothing in the tree stated or pinned that contract.
  * **One corrupt row poisoned the whole search**
    (D-VEC-1, found by this slice). ``vector_search_*``
    stacked every blob with ``np.stack`` before scoring,
    so a single row whose blob was NULL, empty, truncated
    (40 bytes), or not a multiple of 4 raised ``TypeError``
    / ``ValueError`` and the search returned *nothing* —
    every good row lost with the bad one. A blob of the
    right length holding non-finite floats (``0xFF`` bytes
    decode to NaNs) scored NaN, and NaN sorted *ahead of*
    a perfect 1.0 match — garbage outranked truth.
  * **The builders manufactured poison rows blindly**
    (D-VEC-2, found by this slice). With a fake model
    returning 10-dimensional vectors, ``build_entity_
    embeddings`` stored 40-byte blobs without blinking —
    rows the first search over them would choke on. And
    embedding writes ran at the schema's
    ``synchronous = NORMAL``: under WAL that is safe from
    corruption but may lose the last committed batches on
    a power failure. The Samtal lesson (WAL +
    ``synchronous = FULL``, commit before the answer is
    trusted) was applied nowhere in the embedding path.

What this slice builds:

  * ``VECTOR_STORE_SHAPE`` — the shape as data: the two
    tables, their key columns and blob columns, the blob
    format (little-endian float32, the TF-053 registry's
    dimension × 4 bytes), the canonical ``model_name``
    filter, the brute-force scan, and the write discipline.
  * ``classify_blob`` — every stored blob judged by one
    rule, as data (``BLOB_CLASSES``): ``ok`` /
    ``null_blob`` / ``empty_blob`` / ``wrong_byte_count``
    / ``non_finite``. Zero vectors are finite and score
    0.0 — ``ok``.
  * ``load_searchable_rows`` — the one loader both vector
    searches now use: corrupt rows are skipped *by row*,
    each skip logged with its id and class, and the good
    rows' matrix is what gets scored. Search never
    deletes; quarantine is the repair tool's act, below.
  * ``validated_blob`` — the builders' gate: an encoded
    vector is stored only if it is exactly the registered
    dimension and all-finite, else the batch raises
    ``ValueError`` naming the defect *before any row of
    that batch is written* (a batch is one transaction —
    it lands whole or not at all; earlier batches stay,
    and the builders' LEFT-JOIN resume re-offers the rest).
  * ``harden_write_connection`` — embedding write
    connections run ``synchronous = FULL`` (the Samtal
    lesson) on top of the schema's WAL. General/read
    connections keep the schema's NORMAL — stated, not
    silently global.
  * ``inspect_vector_store(db_path)`` — the store read
    as a report (``VectorStoreReport``): presence, table
    presence, per-table row counts split ok/corrupt with
    corrupt ids named by class, and ``PRAGMA
    integrity_check`` — returned as data even when the
    file is missing, a table is missing, or the file is
    not a database at all (never a raise).
  * ``quarantine_corrupt_rows(db_path)`` — the repair:
    deletes exactly the rows ``classify_blob`` judges
    corrupt, in one transaction on a hardened connection,
    and reports what it removed by id and class. Good
    rows are byte-untouched (pinned). It is an explicit
    operator act — search and inspection never call it.
  * ``vector_store_problems()`` — the live checker:
    re-derives every claim above from the live tree
    (schema DDL, builder/search wiring by AST, synthetic
    clean + corrupt stores, a restart cycle).

Stated boundaries:

  * A corrupt *file* (bad header, truncated DB) is not
    row corruption: inspection names it ``unreadable``,
    the searches still raise the SQLite error, and the
    forge's TF-053 degrade keeps hybrid retrieval on its
    SQL leg. Rebuilding a destroyed file is
    ``etl/db_integrity``'s repair road, not this slice's.
  * A missing embeddings table is the same class: named
    by inspection, raised by search, degraded by the forge.
  * Search-time skipping is not silent loss made good:
    skipped ids are logged per search and named by
    inspection; only ``quarantine_corrupt_rows`` removes
    them, and only when an operator calls it.
  * Proof grade is FIXTURE (synthetic stores on this VM,
    a fake backend where a model is needed): no real
    lived-in vector store exists on this machine
    (TF-041), and vector retrieval *quality* stays NOT
    PROVEN (D-ENV-1).
"""

from __future__ import annotations

import ast
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from thoughtforge.etl.embedding_discipline import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    canonical_model_name,
)

logger = logging.getLogger(__name__)

# ── The shape, as data ────────────────────────────────────────────────────────

EXPECTED_DIMENSION = EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]["dimension"]
EXPECTED_BLOB_BYTES = EXPECTED_DIMENSION * 4  # float32, little-endian

VECTOR_STORE_SHAPE: dict[str, Any] = {
    "store_file": "the SQLite knowledge DB (utils.paths.get_knowledge_db_path)",
    "index_structure": "none — brute-force cosine scan over every stored blob, in memory, per search",
    "blob_format": "little-endian float32, exactly dimension x 4 bytes",
    "dimension": EXPECTED_DIMENSION,
    "blob_bytes": EXPECTED_BLOB_BYTES,
    "model_filter": "exact match on model_name, canonical spelling only (TF-053)",
    "journal_mode": "WAL",
    "write_synchronous": "FULL (embedding write connections, TF-054)",
    "read_synchronous": "NORMAL (schema default, unchanged)",
    "batch_atomicity": "one transaction per builder batch — a batch lands whole or not at all",
    "restart_rule": "committed rows survive close/reopen; uncommitted rows roll back",
    "tables": (
        {"table": "embeddings", "key_column": "qid",
         "blob_column": "embedding"},
        {"table": "reference_embeddings", "key_column": "chunk_id",
         "blob_column": "embedding"},
    ),
}

VECTOR_TABLES: tuple[dict[str, str], ...] = tuple(VECTOR_STORE_SHAPE["tables"])

# ── Blob classification ───────────────────────────────────────────────────────

CLASS_OK = "ok"
CLASS_NULL = "null_blob"
CLASS_EMPTY = "empty_blob"
CLASS_WRONG_BYTES = "wrong_byte_count"
CLASS_NON_FINITE = "non_finite"

BLOB_CLASSES = (
    CLASS_OK, CLASS_NULL, CLASS_EMPTY, CLASS_WRONG_BYTES, CLASS_NON_FINITE,
)
CORRUPT_CLASSES = tuple(c for c in BLOB_CLASSES if c != CLASS_OK)


def classify_blob(blob: Any, dimension: int = EXPECTED_DIMENSION) -> str:
    """Judge one stored blob by the one rule; returns a BLOB_CLASSES word."""
    if blob is None:
        return CLASS_NULL
    if len(blob) == 0:
        return CLASS_EMPTY
    if len(blob) != dimension * 4:
        return CLASS_WRONG_BYTES
    values = np.frombuffer(blob, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return CLASS_NON_FINITE
    return CLASS_OK


def validated_blob(vector: Any, dimension: int = EXPECTED_DIMENSION) -> bytes:
    """The builders' gate: a vector is stored only if it is exactly
    *dimension* finite float32 values. Raises ValueError naming the
    defect — the caller raises before writing any row of its batch."""
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (dimension,):
        raise ValueError(
            f"embedding vector has shape {array.shape}, "
            f"not the registered ({dimension},)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding vector holds non-finite values")
    return array.tobytes()


def harden_write_connection(conn: sqlite3.Connection) -> None:
    """The Samtal lesson for embedding writes: WAL is the schema's;
    durability of a commit is FULL synchronous on this connection."""
    conn.execute("PRAGMA synchronous = FULL")


def load_searchable_rows(
    rows: list[Any],
    key_of: Any,
    blob_of: Any,
    *,
    table: str,
) -> tuple[list[Any], np.ndarray | None, list[dict[str, str]]]:
    """Split fetched rows into (kept rows, matrix, skipped).

    Corrupt rows are skipped by row — logged with id and class —
    never stacked, never scored. ``matrix`` is None when no row
    is searchable.
    """
    kept: list[Any] = []
    vectors: list[np.ndarray] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        blob = blob_of(row)
        verdict = classify_blob(blob)
        if verdict != CLASS_OK:
            key = str(key_of(row))
            skipped.append({"table": table, "id": key, "class": verdict})
            logger.warning(
                "Vector store: skipping corrupt %s row %s (%s) — "
                "search continues over the good rows; "
                "quarantine_corrupt_rows removes it",
                table, key, verdict,
            )
            continue
        kept.append(row)
        vectors.append(np.frombuffer(blob, dtype=np.float32))
    matrix = np.stack(vectors) if vectors else None
    return kept, matrix, skipped


# ── Inspection ────────────────────────────────────────────────────────────────

@dataclass
class VectorStoreReport:
    exists: bool
    readable: bool
    integrity: str  # "ok" | a PRAGMA message | "unreadable" | "missing"
    tables_present: list[str] = field(default_factory=list)
    tables_missing: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    corrupt_rows: list[dict[str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.readable and not self.problems and not self.corrupt_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "readable": self.readable,
            "ok": self.ok,
            "integrity": self.integrity,
            "tables_present": list(self.tables_present),
            "tables_missing": list(self.tables_missing),
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "corrupt_rows": [dict(r) for r in self.corrupt_rows],
            "problems": list(self.problems),
        }


def inspect_vector_store(db_path: Path) -> VectorStoreReport:
    """Read a vector store as a report. Never raises: a missing
    file, a missing table, and a file that is not a database are
    all verdicts in the report, named."""
    db_path = Path(db_path)
    if not db_path.exists():
        return VectorStoreReport(
            exists=False, readable=False, integrity="missing",
            tables_missing=[t["table"] for t in VECTOR_TABLES],
            problems=[f"vector store file does not exist: {db_path}"],
        )
    report = VectorStoreReport(exists=True, readable=False, integrity="unreadable")
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            integrity_rows = conn.execute("PRAGMA integrity_check").fetchall()
            report.integrity = (
                "ok" if len(integrity_rows) == 1 and integrity_rows[0][0] == "ok"
                else "; ".join(str(r[0]) for r in integrity_rows)
            )
            report.readable = True
            existing = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for spec in VECTOR_TABLES:
                table = spec["table"]
                if table not in existing:
                    report.tables_missing.append(table)
                    report.problems.append(f"vector table '{table}' is missing")
                    continue
                report.tables_present.append(table)
                rows = conn.execute(
                    f"SELECT {spec['key_column']} AS k, "
                    f"{spec['blob_column']} AS b FROM {table}"
                ).fetchall()
                ok_count = 0
                for row in rows:
                    verdict = classify_blob(row["b"])
                    if verdict == CLASS_OK:
                        ok_count += 1
                    else:
                        report.corrupt_rows.append(
                            {"table": table, "id": str(row["k"]), "class": verdict}
                        )
                report.counts[table] = {
                    "total": len(rows), "ok": ok_count,
                    "corrupt": len(rows) - ok_count,
                }
            if report.integrity != "ok":
                report.problems.append(
                    f"integrity_check: {report.integrity}"
                )
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        report.problems.append(f"vector store unreadable: {exc}")
    return report


# ── Repair ────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineReport:
    removed: list[dict[str, str]] = field(default_factory=list)
    tables_missing: list[str] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed": [dict(r) for r in self.removed],
            "removed_count": self.removed_count,
            "tables_missing": list(self.tables_missing),
        }


def quarantine_corrupt_rows(db_path: Path) -> QuarantineReport:
    """Delete exactly the rows classify_blob judges corrupt, in one
    transaction on a hardened (synchronous=FULL) connection. An
    explicit operator act — nothing calls this automatically."""
    from thoughtforge.etl.schema import get_connection

    report = QuarantineReport()
    conn = get_connection(Path(db_path))
    try:
        harden_write_connection(conn)
        existing = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        with conn:  # one transaction: the whole quarantine lands or none does
            for spec in VECTOR_TABLES:
                table = spec["table"]
                if table not in existing:
                    report.tables_missing.append(table)
                    continue
                rows = conn.execute(
                    f"SELECT {spec['key_column']} AS k, "
                    f"{spec['blob_column']} AS b FROM {table}"
                ).fetchall()
                for row in rows:
                    verdict = classify_blob(row["b"])
                    if verdict == CLASS_OK:
                        continue
                    conn.execute(
                        f"DELETE FROM {table} WHERE {spec['key_column']} = ?",
                        (row["k"],),
                    )
                    report.removed.append(
                        {"table": table, "id": str(row["k"]), "class": verdict}
                    )
    finally:
        conn.close()
    return report


# ── The live checker ──────────────────────────────────────────────────────────

def _functions_calling(source: str, callee: str) -> set[str]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == callee):
                    found.add(node.name)
    return found


def vector_store_problems() -> list[str]:
    """Re-derive TF-054's claims from the live tree; [] = all hold."""
    import tempfile

    from thoughtforge.etl import embeddings
    from thoughtforge.etl.schema import get_connection, initialize_schema

    problems: list[str] = []

    # The shape's dimension is the registry's (TF-053) and EMBEDDING_DIM's.
    if VECTOR_STORE_SHAPE["dimension"] != embeddings.EMBEDDING_DIM:
        problems.append("shape dimension != EMBEDDING_DIM")
    if EXPECTED_BLOB_BYTES != embeddings.EMBEDDING_DIM * 4:
        problems.append("shape blob bytes != EMBEDDING_DIM x 4")

    # The schema DDL holds exactly the shape's tables/keys.
    from thoughtforge.etl import schema as schema_module

    ddl = schema_module._TABLES
    for spec in VECTOR_TABLES:
        if f"CREATE TABLE IF NOT EXISTS {spec['table']}" not in ddl:
            problems.append(f"schema DDL lacks table {spec['table']}")

    # Wiring: builders gate + harden; searches load through the loader.
    etl_source = Path(embeddings.__file__).read_text(encoding="utf-8")
    for function in ("build_entity_embeddings", "build_reference_embeddings"):
        if function not in _functions_calling(etl_source, "validated_blob"):
            problems.append(f"{function} does not gate through validated_blob")
        if function not in _functions_calling(etl_source, "harden_write_connection"):
            problems.append(f"{function} does not harden its write connection")
    for function in ("vector_search_entities", "vector_search_reference"):
        if function not in _functions_calling(etl_source, "load_searchable_rows"):
            problems.append(f"{function} does not load through load_searchable_rows")

    # Synthetic clean + corrupt stores behave as the doc states.
    unit = np.zeros(EXPECTED_DIMENSION, dtype=np.float32)
    unit[0] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "k.db"
        initialize_schema(db)
        conn = get_connection(db)
        try:
            harden_write_connection(conn)
            if conn.execute("PRAGMA synchronous").fetchone()[0] != 2:
                problems.append("hardened connection is not synchronous=FULL")
            conn.execute(
                "INSERT INTO entities (qid, label_en) VALUES ('Q1', 'Ferry')")
            conn.execute(
                "INSERT INTO embeddings (qid, embedding, model_name) "
                "VALUES ('Q1', ?, ?)",
                (unit.tobytes(), DEFAULT_EMBEDDING_MODEL))
            conn.execute(
                "INSERT INTO entities (qid, label_en) VALUES ('Q2', 'Bad')")
            conn.execute(
                "INSERT INTO embeddings (qid, embedding, model_name) "
                "VALUES ('Q2', ?, ?)",
                (b"\xff" * EXPECTED_BLOB_BYTES, DEFAULT_EMBEDDING_MODEL))
            conn.commit()
        finally:
            conn.close()

        report = inspect_vector_store(db)
        if not report.readable or report.integrity != "ok":
            problems.append(f"clean-ish store not readable/ok: {report.to_dict()}")
        if [r["id"] for r in report.corrupt_rows] != ["Q2"]:
            problems.append(f"inspection named {report.corrupt_rows}, not Q2")
        # Restart: a fresh inspection still sees both rows.
        again = inspect_vector_store(db)
        if again.counts.get("embeddings", {}).get("total") != 2:
            problems.append("committed rows did not survive a reopen")
        removed = quarantine_corrupt_rows(db)
        if removed.removed_count != 1:
            problems.append(f"quarantine removed {removed.removed_count}, not 1")
        after = inspect_vector_store(db)
        if not after.ok or after.counts["embeddings"]["total"] != 1:
            problems.append("store not clean after quarantine")

    return problems
