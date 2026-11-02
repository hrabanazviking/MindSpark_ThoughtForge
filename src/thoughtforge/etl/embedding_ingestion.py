"""
Embedding at ingestion, as data (TF-111).

The roadmap's rule: *embedding at ingestion — model per
profile (TF-053), resumable, checksummed.*

TF-053 filed the discipline — the model registry, the
per-profile assignment (``PROFILE_EMBEDDINGS``: model,
runtime, batch size per shipped profile; pi_zero assigned
none), and the stored-estate check — and stated its own
boundary plainly: "the builders still take their batch_size
arguments from their callers (TF-111 wires ingestion to the
assignment)" and "TF-111 (embedding at ingestion) consumes
``stored_embeddings_problems``". TF-111's probes measured
the estate before this slice changed it:

  * **The assignment was wired nowhere.** Neither builder
    takes a profile (probed signatures:
    ``build_entity_embeddings(db_path, model_name,
    batch_size, limit)`` / ``build_reference_embeddings
    (db_path, model_name, batch_size)``) — a direct build
    on pi_zero's behalf would embed at the default batch
    128 through the sentence-transformers stack TF-053
    ruled it cannot carry, and phone_low would batch at
    128/64 instead of its assigned 16.
  * **D-EMB-3 — building under a second model silently
    built nothing.** Both builders selected "missing" with
    a bare ``LEFT JOIN ... ON qid`` / ``ON chunk_id`` —
    no ``model_name`` in the join — so any row embedded
    under one model blocked building the same row under
    another (live probe: 2 entities embedded under the
    default model; a build under a second model name
    returned 0, and that model's searches, which DO
    filter by ``model_name``, find nothing). A profile
    switch changed which model searches use without
    changing which rows exist.
  * **No checksum existed anywhere in the path** (grep:
    hashlib / sha256 / checksum all absent from
    ``etl/embeddings.py`` and
    ``etl/embedding_discipline.py``). TF-054 gates a
    vector's dimension and finiteness before the write
    and TF-053's stored check judges a blob's LENGTH —
    a same-length corruption of a stored blob (bit rot,
    a torn write that still commits) was undetectable
    by any surface in the tree.
  * **No checkpoint existed.** The builders commit per
    batch (TF-054), so an interruption leaves a partial,
    resumable index — but no record anywhere says which
    model, which profile, or how far a build got, and
    nothing distinguishes a finished build from an
    interrupted one.

What this slice builds, filed as data in
``INGESTION_RULES`` and enforced here:

  * ``embedding_plan()`` — the run's ask resolved ONCE:
    the model and batch sizes come from TF-053's live
    assignment when a profile is named (an explicit
    ``model_name`` / ``batch_size`` override wins and is
    recorded as an override); no profile keeps the
    builders' filed defaults (128 / 64, TF-105). A
    profile TF-053 assigns no model (pi_zero) yields a
    plan whose model is None, and building it raises
    ``EmbeddingNotAssignedError`` — nothing is written
    and no checkpoint is filed.
  * ``build_embeddings_at_ingestion()`` — the driver:
    both tables, in the builders' own text shapes
    (TF-105's filed entity/reference texts), batched at
    the plan's sizes, per-model only-missing selection,
    every vector gated by TF-054's ``validated_blob``
    on a FULL-hardened connection (the builders' own
    posture), committing per batch.
  * **Checksummed.** Every row's validated blob is
    SHA-256'd before the write; after the batch commits,
    the stored blobs are read back and their checksums
    must equal the written ones, or the run stops with
    ``ChecksumMismatchError`` naming table + id (the
    checkpoint stays at the last verified batch). The
    checkpoint records every verified batch's per-row
    checksums, and every resume — in-progress or
    complete — re-verifies the recorded rows against
    the store before it trusts them: a tampered or
    rotted recorded row raises by name instead of being
    silently carried. ``row_checksums()`` /
    ``estate_digest()`` expose the same sums as data.
  * **Resumable.** A sidecar checkpoint beside the DB
    (``<db>.embedding-checkpoint.json``), written
    atomically (tmp + ``os.replace``) ONLY after the
    commit + read-back it describes has landed. It
    carries the run fingerprint (resolved DB path,
    canonical model, profile, both batch sizes), the
    verified batches, the counts as of the last verified
    batch, and its status. Resume continues only on an
    exact fingerprint match — any difference raises
    ``CheckpointMismatchError`` naming each differing
    field; an unreadable/malformed record raises
    ``CheckpointCorruptError`` (never a silent restart);
    ``resume=False`` discards the record explicitly. A
    completed checkpoint, its estate re-verified, IS
    the result: it returns without encoding anything.
  * **The builders' fix (D-EMB-3).** Both builders in
    ``etl/embeddings.py`` now join on ``model_name`` as
    well as the row id, so "missing" means missing
    *under the model being built* everywhere, not only
    in this driver; TF-105's map is updated in the open.
  * **The run ends proven.** At completion the driver
    runs TF-053's ``stored_embeddings_problems`` for the
    run's model over the store; any problem raises
    instead of filing a complete checkpoint over a
    dirty estate. The result carries the estate digest
    (SHA-256 over every stored row checksum for the
    model, both tables).

Filed limits, stated plainly:

  * The checkpoint is a sidecar, not a schema table —
    TF-105's filed schema facts (10 tables) stand.
  * **One model per row in storage (D-EMB-4).** The
    embeddings tables' primary key is the row id alone
    (TF-041's filed key), so two models' embeddings for
    the same row cannot coexist: building under a new
    model replaces the previous model's rows for the
    rows it builds. Per-model selection (D-EMB-3) makes
    the build honest; a composite (row id, model)
    primary key is a schema migration, owned by
    TF-177's upgrade path — not smuggled into this
    slice.
  * The driver encodes through
    ``etl/embeddings.encode_texts`` by default, so the
    TF-053 runtime story is unchanged: sentence-
    transformers profiles encode when the package and
    cache allow (TF-053's availability report owns that
    verdict); the ONNX profile's export path is
    TF-070's, and no ONNX encoder is wired here — a
    phone_low run on this VM therefore reaches the same
    honest package-missing wall as any other profile
    (D-ENV-1), with its plan (model + batch 16) resolved
    and recorded before the wall.
  * An ``encoder`` may be injected (tests, TF-112's
    dry-run): it receives (texts, model_name) and
    returns the vector matrix; the checksum, checkpoint
    and estate machinery judges its output exactly as
    it judges the real encoder's.
  * Checksums prove the stored bytes are the bytes the
    encoder produced and the read-back saw — they do
    not prove the vectors are GOOD (retrieval quality
    remains NOT PROVEN, TF-053's standing caveat).

Docket (TF-016): D-EMB-3 is added and FIXED by this
slice.

The human-readable write-up is
``docs/architecture/EMBEDDING_AT_INGESTION.md``. Proof
grade: FIXTURE — synthetic stores through the real
driver with a fake encoder on this VM (sentence-
transformers is absent here, D-ENV-1, so the real
encoder's leg is the filed RuntimeError, never a run);
a real-model ingestion at scale is NOT PROVEN, never
claimed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from thoughtforge.etl import embeddings as embeddings_module
from thoughtforge.etl.embedding_discipline import (
    canonical_model_name,
    embedding_assignment,
    stored_embeddings_problems,
)
from thoughtforge.etl.schema import get_connection
from thoughtforge.etl.vector_store import harden_write_connection, validated_blob

INGESTION_FORMAT = "thoughtforge-embedding-ingestion/1"
INGESTION_DOC = "docs/architecture/EMBEDDING_AT_INGESTION.md"
CHECKPOINT_FORMAT = "thoughtforge-embedding-checkpoint/1"

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"

#: The tables this driver fills, in build order: (table,
#: id column, source of rows). The id shapes are the
#: schema's own (TF-105): entities by qid, chunks by id.
TABLE_ENTITIES = "embeddings"
TABLE_REFERENCE = "reference_embeddings"
INGESTION_TABLES = (TABLE_ENTITIES, TABLE_REFERENCE)


class EmbeddingNotAssignedError(ValueError):
    """The named profile is assigned no embedding model (TF-053)."""


class CheckpointMismatchError(ValueError):
    """A stored checkpoint does not describe this run's ask."""


class CheckpointCorruptError(ValueError):
    """A stored checkpoint cannot be read or is malformed."""


class ChecksumMismatchError(ValueError):
    """A stored blob's checksum is not the checksum written for it."""


# ── The rules ───────────────────────────────────────────────────────────────

INGESTION_RULES: tuple[dict[str, Any], ...] = (
    {"id": "model_and_batch_come_from_the_profile_assignment",
     "rule": "when a profile is named, the run's model and batch sizes are TF-053's PROFILE_EMBEDDINGS assignment read live — never a copied table; an explicit model_name / batch_size override wins over the assignment and is recorded as an override in the plan and the checkpoint"},
    {"id": "no_profile_keeps_the_builders_defaults",
     "rule": "with no profile named, the plan is the builders' filed defaults (TF-105: the default model, entity batch 128, reference batch 64) — no cap or model is invented"},
    {"id": "not_assigned_means_no_build",
     "rule": "a profile TF-053 assigns no model (pi_zero, runtime none) raises EmbeddingNotAssignedError before anything is written or checkpointed; its retrieval is SQL-only by TF-053's rule, not by a failed build"},
    {"id": "unknown_profile_raises_by_name",
     "rule": "an unknown profile id raises ValueError naming the assigned profiles (TF-053's embedding_assignment), before any store is opened"},
    {"id": "selection_is_missing_under_this_model",
     "rule": "a row is built only when it has no embedding stored under the run's canonical model — a row stored under another model no longer blocks the build (D-EMB-3, fixed in this driver and in both builders); the tables hold ONE model's row per entity/chunk (the schema's primary key is the row id alone), so building under a new model replaces the previous model's rows for the rows it builds — coexistence is D-EMB-4, owned by TF-177's upgrade path"},
    {"id": "every_batch_is_checksummed_and_read_back",
     "rule": "each row's validated blob is SHA-256'd before its batch writes; after the batch commits, the stored blobs are read back and every checksum must match, or the run stops with ChecksumMismatchError naming table + id and the checkpoint stays at the last verified batch"},
    {"id": "checkpoint_after_verified_commit_only",
     "rule": "the checkpoint sidecar is written atomically (tmp + os.replace) only after the commit AND read-back verification it describes have landed; the checkpoint on disk is therefore always a truthful, verified resume point"},
    {"id": "resume_is_by_fingerprint",
     "rule": "resume continues only when the checkpoint's fingerprint (resolved DB path, canonical model, profile, entity + reference batch sizes) matches this run's plan exactly; any difference raises CheckpointMismatchError naming each differing field — a changed ask is a new build (resume=False), never a silent continuation"},
    {"id": "a_checkpoint_is_verified_before_it_is_trusted",
     "rule": "every resume first re-computes the checksums of the rows the checkpoint records and compares them to the recorded sums; a missing or altered recorded row raises ChecksumMismatchError by name instead of being carried silently"},
    {"id": "a_completed_checkpoint_is_the_result",
     "rule": "resuming a run whose checkpoint is complete (and re-verified) returns the checkpoint's filed result without encoding anything"},
    {"id": "resume_false_discards_the_checkpoint",
     "rule": "resume=False deletes the checkpoint record explicitly and reconsiders the whole ask; rows already stored under the run's model still stand (per-model only-missing is the store's law, not the checkpoint's)"},
    {"id": "the_run_ends_proven_against_the_registry",
     "rule": "at completion, TF-053's stored_embeddings_problems for the run's model must be empty over the store, or the run raises instead of filing a complete checkpoint over a dirty estate; the filed result carries the estate digest over every stored row checksum for the model"},
)


# ── The plan ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingPlan:
    profile_id: str | None
    model: str | None            # canonical; None = not assigned
    entity_batch_size: int
    reference_batch_size: int
    model_source: str
    batch_source: str

    def fingerprint(self, db_path: Path) -> dict[str, Any]:
        return {
            "db_path": str(Path(db_path).resolve()),
            "model": self.model,
            "profile_id": self.profile_id,
            "entity_batch_size": self.entity_batch_size,
            "reference_batch_size": self.reference_batch_size,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "entity_batch_size": self.entity_batch_size,
            "reference_batch_size": self.reference_batch_size,
            "model_source": self.model_source,
            "batch_source": self.batch_source,
        }


def embedding_plan(
    profile_id: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> EmbeddingPlan:
    """Resolve one ingestion run's ask, once, from TF-053 live.

    Raises ValueError for an unknown profile (TF-053's own
    error). A not-assigned profile resolves to a plan with
    ``model is None`` — the refusal is the build's, so a
    caller may file the plan without building.
    """
    assignment: dict[str, Any] | None = None
    if profile_id is not None:
        assignment = embedding_assignment(profile_id)

    if model_name is not None:
        model: str | None = canonical_model_name(model_name)
        model_source = "explicit:model_name"
    elif assignment is not None:
        model = assignment["model"]
        if model is not None:
            model = canonical_model_name(model)
        model_source = f"profile:{profile_id} (TF-053 assignment)"
    else:
        model = canonical_model_name(embeddings_module.DEFAULT_MODEL)
        model_source = "default:etl/embeddings.DEFAULT_MODEL"

    # The builders' filed defaults (TF-105 pins 128 / 64);
    # reading them off the signatures keeps one source of
    # truth if they ever move in the open.
    import inspect

    entity_default = inspect.signature(
        embeddings_module.build_entity_embeddings
    ).parameters["batch_size"].default
    reference_default = inspect.signature(
        embeddings_module.build_reference_embeddings
    ).parameters["batch_size"].default

    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}"
            )
        entity_batch = reference_batch = batch_size
        batch_source = "explicit:batch_size"
    elif assignment is not None:
        assigned = int(assignment["batch_size"])
        entity_batch = reference_batch = assigned
        batch_source = f"profile:{profile_id} (TF-053 assignment)"
    else:
        entity_batch, reference_batch = entity_default, reference_default
        batch_source = "default:builders' filed defaults (TF-105)"

    return EmbeddingPlan(
        profile_id=profile_id,
        model=model,
        entity_batch_size=entity_batch,
        reference_batch_size=reference_batch,
        model_source=model_source,
        batch_source=batch_source,
    )


# ── Checksums ───────────────────────────────────────────────────────────────

def blob_checksum(blob: bytes) -> str:
    """SHA-256 of one stored blob's exact bytes."""
    return hashlib.sha256(blob).hexdigest()


def row_checksums(
    db_path: Path, model: str,
) -> dict[str, dict[str, str]]:
    """Every stored row checksum for *model*, both tables.

    Keys are the row ids as strings (qid / chunk_id). Rows
    filed under other models are not this model's estate.
    """
    canonical = canonical_model_name(model)
    sums: dict[str, dict[str, str]] = {t: {} for t in INGESTION_TABLES}
    conn = get_connection(Path(db_path), read_only=True)
    try:
        for table, id_column in ((TABLE_ENTITIES, "qid"),
                                 (TABLE_REFERENCE, "chunk_id")):
            rows = conn.execute(
                f"SELECT {id_column}, embedding FROM {table} "
                f"WHERE model_name = ?",
                (canonical,),
            ).fetchall()
            for row_id, blob in rows:
                sums[table][str(row_id)] = blob_checksum(blob)
    finally:
        conn.close()
    return sums


def estate_digest(checksums: dict[str, dict[str, str]]) -> str:
    """One digest over an estate's checksums (order-free)."""
    lines = [
        f"{table}:{row_id}:{checksums[table][row_id]}"
        for table in sorted(checksums)
        for row_id in sorted(checksums[table])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ── The checkpoint ──────────────────────────────────────────────────────────

def default_checkpoint_path(db_path: Path) -> Path:
    return Path(str(db_path) + ".embedding-checkpoint.json")


def _write_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(
            f"embedding checkpoint at {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(record, dict) or record.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointCorruptError(
            f"embedding checkpoint at {path} is malformed: not a "
            f"{CHECKPOINT_FORMAT} record"
        )
    for key in ("fingerprint", "status", "batches", "counts"):
        if key not in record:
            raise CheckpointCorruptError(
                f"embedding checkpoint at {path} is malformed: "
                f"missing {key!r}"
            )
    return record


def _fingerprint_differences(
    filed: dict[str, Any], asked: dict[str, Any],
) -> list[str]:
    differences = []
    for key in sorted(set(filed) | set(asked)):
        if filed.get(key) != asked.get(key):
            differences.append(
                f"{key}: checkpoint has {filed.get(key)!r}, "
                f"this run asks {asked.get(key)!r}"
            )
    return differences


def _verify_recorded_rows(
    db_path: Path, record: dict[str, Any],
) -> None:
    """Re-check every row the checkpoint records, by checksum."""
    stored = row_checksums(db_path, record["fingerprint"]["model"])
    for batch in record["batches"]:
        table = batch["table"]
        for row_id, filed_sum in batch["checksums"].items():
            actual = stored[table].get(str(row_id))
            if actual is None:
                raise ChecksumMismatchError(
                    f"{table} row {row_id}: recorded in the checkpoint "
                    f"but missing from the store"
                )
            if actual != filed_sum:
                raise ChecksumMismatchError(
                    f"{table} row {row_id}: stored checksum {actual} "
                    f"!= checkpoint checksum {filed_sum} — the stored "
                    f"bytes changed since they were verified"
                )


# ── The driver ──────────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    plan: dict[str, Any]
    entities_built: int
    reference_built: int
    batches: int
    resumed: bool
    from_completed_checkpoint: bool
    estate_digest: str
    stored_problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "entities_built": self.entities_built,
            "reference_built": self.reference_built,
            "batches": self.batches,
            "resumed": self.resumed,
            "from_completed_checkpoint": self.from_completed_checkpoint,
            "estate_digest": self.estate_digest,
            "stored_problems": list(self.stored_problems),
        }


Encoder = Callable[[list[str], str], np.ndarray]


def _default_encoder(texts: list[str], model_name: str) -> np.ndarray:
    return embeddings_module.encode_texts(texts, model_name)


def _missing_entities(conn: Any, model: str) -> list[Any]:
    return conn.execute(
        "SELECT e.qid, e.label_en, e.description_en FROM entities e "
        "LEFT JOIN embeddings em ON e.qid = em.qid AND em.model_name = ? "
        "WHERE em.qid IS NULL AND e.label_en IS NOT NULL",
        (model,),
    ).fetchall()


def _missing_reference(conn: Any, model: str) -> list[Any]:
    return conn.execute(
        "SELECT rc.chunk_id, rc.title, rc.content FROM reference_chunks rc "
        "LEFT JOIN reference_embeddings re ON rc.chunk_id = re.chunk_id "
        "AND re.model_name = ? WHERE re.chunk_id IS NULL",
        (model,),
    ).fetchall()


def build_embeddings_at_ingestion(
    db_path: Path,
    profile_id: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    resume: bool = True,
    checkpoint_path: Path | None = None,
    encoder: Encoder | None = None,
) -> IngestionResult:
    """Embed a store's unembedded rows, per profile, resumably.

    See the module docstring for the filed rules. ``encoder``
    defaults to ``etl/embeddings.encode_texts``; an injected
    encoder is judged by exactly the same machinery.
    """
    db_path = Path(db_path)
    plan = embedding_plan(profile_id, model_name, batch_size)
    if plan.model is None:
        raise EmbeddingNotAssignedError(
            f"profile {profile_id!r} is assigned no embedding model "
            f"by TF-053 (runtime none) — embedding at ingestion is "
            f"not run for it; retrieval is SQL-only by rule"
        )
    encode = encoder if encoder is not None else _default_encoder
    checkpoint = (Path(checkpoint_path) if checkpoint_path is not None
                  else default_checkpoint_path(db_path))
    fingerprint = plan.fingerprint(db_path)

    record: dict[str, Any] | None = None
    resumed = False
    if checkpoint.exists():
        if not resume:
            checkpoint.unlink()
        else:
            record = _read_checkpoint(checkpoint)
            differences = _fingerprint_differences(
                record["fingerprint"], fingerprint)
            if differences:
                raise CheckpointMismatchError(
                    "embedding checkpoint does not describe this run: "
                    + "; ".join(differences)
                )
            _verify_recorded_rows(db_path, record)
            resumed = True
            if record["status"] == STATUS_COMPLETE:
                return IngestionResult(
                    plan=plan.to_dict(),
                    entities_built=record["counts"]["entities_built"],
                    reference_built=record["counts"]["reference_built"],
                    batches=len(record["batches"]),
                    resumed=True,
                    from_completed_checkpoint=True,
                    estate_digest=record["estate_digest"],
                    stored_problems=[],
                )
    if record is None:
        record = {
            "format": CHECKPOINT_FORMAT,
            "fingerprint": fingerprint,
            "plan": plan.to_dict(),
            "status": STATUS_IN_PROGRESS,
            "batches": [],
            "counts": {"entities_built": 0, "reference_built": 0},
            "estate_digest": None,
        }

    conn = get_connection(db_path)
    harden_write_connection(conn)
    try:
        legs = (
            (TABLE_ENTITIES, "entities_built",
             _missing_entities(conn, plan.model),
             lambda row: (str(row["qid"]),
                          f"{row['label_en']}: {row['description_en']}"
                          if row["description_en"] else row["label_en"]),
             plan.entity_batch_size,
             "INSERT OR REPLACE INTO embeddings "
             "(qid, text_for_embedding, embedding, model_name) "
             "VALUES (?, ?, ?, ?)"),
            (TABLE_REFERENCE, "reference_built",
             _missing_reference(conn, plan.model),
             lambda row: (str(row["chunk_id"]),
                          f"{row['title']}: {row['content'][:500]}"
                          if row["title"] else row["content"][:500]),
             plan.reference_batch_size,
             "INSERT OR REPLACE INTO reference_embeddings "
             "(chunk_id, embedding, model_name) VALUES (?, ?, ?)"),
        )
        for table, count_key, rows, shape, size, insert_sql in legs:
            for start in range(0, len(rows), size):
                batch_rows = rows[start:start + size]
                shaped = [shape(row) for row in batch_rows]
                texts = [text for _row_id, text in shaped]
                vectors = encode(texts, plan.model)
                blobs = [validated_blob(vectors[j])
                         for j in range(len(batch_rows))]
                written = {row_id: blob_checksum(blobs[j])
                           for j, (row_id, _t) in enumerate(shaped)}
                if table == TABLE_ENTITIES:
                    conn.executemany(insert_sql, [
                        (batch_rows[j]["qid"], texts[j], blobs[j], plan.model)
                        for j in range(len(batch_rows))
                    ])
                else:
                    conn.executemany(insert_sql, [
                        (batch_rows[j]["chunk_id"], blobs[j], plan.model)
                        for j in range(len(batch_rows))
                    ])
                conn.commit()
                # Read back what the store actually holds.
                id_column = ("qid" if table == TABLE_ENTITIES
                             else "chunk_id")
                for row_id, expected in written.items():
                    stored_row = conn.execute(
                        f"SELECT embedding FROM {table} "
                        f"WHERE {id_column} = ? AND model_name = ?",
                        (_stored_key(batch_rows, shaped, row_id, table),
                         plan.model),
                    ).fetchone()
                    if stored_row is None:
                        raise ChecksumMismatchError(
                            f"{table} row {row_id}: written but not "
                            f"found on read-back"
                        )
                    actual = blob_checksum(stored_row[0])
                    if actual != expected:
                        raise ChecksumMismatchError(
                            f"{table} row {row_id}: read-back checksum "
                            f"{actual} != written checksum {expected}"
                        )
                record["batches"].append({
                    "table": table,
                    "rows": len(batch_rows),
                    "checksums": written,
                })
                record["counts"][count_key] += len(batch_rows)
                _write_checkpoint(checkpoint, record)
    except BaseException:
        # The checkpoint on disk stays at the last verified
        # batch (it is only ever written after one); the
        # original error re-raises unchanged.
        raise
    finally:
        conn.close()

    problems = stored_embeddings_problems(db_path, plan.model)
    if problems:
        raise RuntimeError(
            "embedding ingestion completed its writes but the stored "
            "estate fails TF-053's registry check: " + "; ".join(problems)
        )
    digest = estate_digest(row_checksums(db_path, plan.model))
    record["status"] = STATUS_COMPLETE
    record["estate_digest"] = digest
    _write_checkpoint(checkpoint, record)
    return IngestionResult(
        plan=plan.to_dict(),
        entities_built=record["counts"]["entities_built"],
        reference_built=record["counts"]["reference_built"],
        batches=len(record["batches"]),
        resumed=resumed,
        from_completed_checkpoint=False,
        estate_digest=digest,
        stored_problems=[],
    )


def _stored_key(batch_rows: list[Any], shaped: list[tuple[str, str]],
                row_id: str, table: str) -> Any:
    """The store's own key type for one shaped row id."""
    for row, (shaped_id, _text) in zip(batch_rows, shaped):
        if shaped_id == row_id:
            return row["qid"] if table == TABLE_ENTITIES else row["chunk_id"]
    raise KeyError(row_id)  # unreachable: ids come from shaped itself


# ── The checkers ────────────────────────────────────────────────────────────

def embedding_ingestion_problems() -> list[str]:
    """Static truth: the filed rules against the live tree."""
    problems: list[str] = []

    rule_ids = [r["id"] for r in INGESTION_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate INGESTION_RULES ids")
    for needed in ("model_and_batch_come_from_the_profile_assignment",
                   "not_assigned_means_no_build",
                   "selection_is_missing_under_this_model",
                   "every_batch_is_checksummed_and_read_back",
                   "resume_is_by_fingerprint",
                   "the_run_ends_proven_against_the_registry"):
        if needed not in rule_ids:
            problems.append(f"INGESTION_RULES lost {needed}")

    # TF-053's assignment is the plan's source, read live.
    from thoughtforge.etl.embedding_discipline import PROFILE_EMBEDDINGS

    for profile_id, assignment in PROFILE_EMBEDDINGS.items():
        plan = embedding_plan(profile_id)
        if assignment["model"] is None:
            if plan.model is not None:
                problems.append(f"{profile_id}: plan assigns a model "
                                "TF-053 does not")
        else:
            if plan.model != canonical_model_name(assignment["model"]):
                problems.append(f"{profile_id}: plan model drifted from "
                                "the TF-053 assignment")
            if plan.entity_batch_size != assignment["batch_size"]:
                problems.append(f"{profile_id}: plan batch drifted from "
                                "the TF-053 assignment")
    plain = embedding_plan()
    if plain.model != canonical_model_name(embeddings_module.DEFAULT_MODEL):
        problems.append("the no-profile plan lost the default model")
    if (plain.entity_batch_size, plain.reference_batch_size) != (128, 64):
        problems.append("the no-profile plan lost the builders' defaults")

    # D-EMB-3's fix is present in both builders, not only here.
    import inspect

    for fn_name, needle in (
            ("build_entity_embeddings", "AND em.model_name"),
            ("build_reference_embeddings", "AND re.model_name")):
        source = inspect.getsource(getattr(embeddings_module, fn_name))
        if needle not in source:
            problems.append(f"{fn_name}: the only-missing join does not "
                            "filter by model_name (D-EMB-3 reopened)")

    doc = Path(__file__).resolve().parents[3] / INGESTION_DOC
    if not doc.exists():
        problems.append(f"the write-up {INGESTION_DOC} is missing")
    return problems


def _fixture_store(base: Path) -> Path:
    from thoughtforge.etl.schema import initialize_schema

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


def _fake_encoder(fail_on_call: int | None = None) -> Encoder:
    calls = {"n": 0}

    def encode(texts: list[str], model_name: str) -> np.ndarray:
        calls["n"] += 1
        if fail_on_call is not None and calls["n"] == fail_on_call:
            raise RuntimeError("encoder interrupted (fixture)")
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(384, dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

    return encode


def embedding_ingestion_live_problems(base: Path | None = None) -> list[str]:
    """Behavioural truth: the whole road on a synthetic store."""
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(base) if base is not None else Path(tmp) / "live"
        db = _fixture_store(root)

        # A full run at a small explicit batch: every row lands,
        # checksummed, checkpoint complete.
        result = build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder())
        if (result.entities_built, result.reference_built) != (5, 3):
            problems.append(f"full run built {result.entities_built}/"
                            f"{result.reference_built}, expected 5/3")
        if result.from_completed_checkpoint:
            problems.append("a fresh run claims the checkpoint's result")
        sums = row_checksums(db, result.plan["model"])
        if len(sums[TABLE_ENTITIES]) != 5 or len(sums[TABLE_REFERENCE]) != 3:
            problems.append("row_checksums does not cover the estate")
        if estate_digest(sums) != result.estate_digest:
            problems.append("the filed estate digest does not re-derive")

        # A completed checkpoint IS the result (an encoder that
        # raises if touched proves nothing was encoded).
        again = build_embeddings_at_ingestion(
            db, batch_size=2, encoder=_fake_encoder(fail_on_call=1))
        if not again.from_completed_checkpoint:
            problems.append("a completed checkpoint was not the result")
        if again.entities_built != 5 or again.reference_built != 3:
            problems.append("the completed checkpoint's counts drifted")

        # A changed ask refuses by name.
        try:
            build_embeddings_at_ingestion(
                db, batch_size=4, encoder=_fake_encoder())
            problems.append("a changed batch size resumed silently")
        except CheckpointMismatchError as exc:
            if "entity_batch_size" not in str(exc):
                problems.append(f"mismatch did not name the field: {exc}")

        # Tampering with a recorded row is caught on resume.
        conn = get_connection(db)
        try:
            blob = conn.execute(
                "SELECT embedding FROM embeddings WHERE qid = 'Q1'"
            ).fetchone()[0]
            tampered = bytes([blob[0] ^ 0xFF]) + blob[1:]
            conn.execute(
                "UPDATE embeddings SET embedding = ? WHERE qid = 'Q1'",
                (tampered,),
            )
            conn.commit()
        finally:
            conn.close()
        try:
            build_embeddings_at_ingestion(
                db, batch_size=2, encoder=_fake_encoder())
            problems.append("a tampered recorded row resumed silently")
        except ChecksumMismatchError:
            pass

        # Interruption + resume on a fresh store: the first run
        # dies mid-entities, the resume finishes exactly the
        # remainder, and no row is built twice.
        db2 = _fixture_store(root / "second")
        try:
            build_embeddings_at_ingestion(
                db2, batch_size=2,
                encoder=_fake_encoder(fail_on_call=2))
            problems.append("the interrupting encoder did not raise")
        except RuntimeError:
            pass
        partial = row_checksums(db2, embeddings_module.DEFAULT_MODEL)
        if len(partial[TABLE_ENTITIES]) != 2:
            problems.append("interruption did not stop at a batch edge: "
                            f"{len(partial[TABLE_ENTITIES])} entity rows")
        finished = build_embeddings_at_ingestion(
            db2, batch_size=2, encoder=_fake_encoder())
        if not finished.resumed:
            problems.append("the second run does not know it resumed")
        if (finished.entities_built, finished.reference_built) != (5, 3):
            problems.append("resume counts do not total the whole store")

        # pi_zero refuses by rule and writes nothing.
        db3 = _fixture_store(root / "third")
        try:
            build_embeddings_at_ingestion(
                db3, profile_id="pi_zero", encoder=_fake_encoder())
            problems.append("pi_zero built embeddings it is not assigned")
        except EmbeddingNotAssignedError:
            pass
        if default_checkpoint_path(db3).exists():
            problems.append("pi_zero left a checkpoint behind")
    return problems
