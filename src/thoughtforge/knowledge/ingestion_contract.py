"""
Ingestion contract — what a "fact" is in ThoughtForge (TF-042).

Before this slice the tree had *shapes* but no contract: six
persisted memory record types and six knowledge-DB tables that
hold claims, each constructed and written with no validation at
all. A record with ``confidence=999``, an empty summary, a
summary ten times its stated cap, or a status outside the closed
vocabulary was appended verbatim, and the scoring maths that
later consumed it silently assumed values in [0, 1]. Provenance
existed on the DB side only as an unvalidated free-text
``source`` column, and on the memory side not at all — no
persisted memory record carries any field saying where it came
from.

This module states the one schema as *verifiable data* (the
TF-039/TF-041 pattern) and validates against it:

  * ``SCHEMA_PARTS`` — the six parts every fact carries, in
    either store: identity, kind, content, provenance,
    confidence, lifecycle.
  * ``FACT_KINDS`` — every ingestible kind: the 6 memory kinds
    (``knowledge/models.py`` records, written through
    ``MemoryStore``) and the 6 knowledge-DB kinds
    (``etl/schema.py`` tables, written by the ETL wing), each
    with its identity/content fields, its provenance field and
    vocabulary, and its confidence-like field with the default
    the code actually constructs.
  * ``DB_PROVENANCE`` — the closed provenance vocabulary of the
    knowledge DB, proven live from the schema's own ``source``
    defaults. ``MEMORY_PROVENANCE`` — the contract's vocabulary
    for the provenance stamp on memory records. At TF-042 no
    record could carry one (the gap was pinned by
    ``contract_problems()``); TF-047 closed it for the four
    writeback kinds — ``user_preference``, ``user_fact``,
    ``episode``, ``response_pattern`` now carry a ``source``
    field and the writeback leg stamps every record it writes.
    Identity and thread state carry no stamp by ruling (they are
    not ingested claims), and the pin now checks exactly that
    split.
  * ``record_problems()`` — the validator for memory records
    (dataclass instance or plain dict). ``MemoryStore``'s
    ingestion gates (append_* / save_*) now call it and refuse
    an invalid record with a ``ValueError`` naming every
    problem, before anything is written. Load paths deliberately
    do NOT validate: a legacy invalid record on disk still loads
    (TF-041's K13 tolerance stands) and is flagged by this
    validator when audited, never silently dropped.
    ``rewrite_*`` methods do not validate either — they are the
    prune/deletion mechanism and must be able to carry legacy
    records out of the store.
  * ``db_row_problems()`` — the same contract for knowledge-DB
    rows (dicts as the ETL wing builds them / ``sqlite3.Row``
    shapes), per kind.
  * ``writeback_fact_problems()`` — the contract for the dicts a
    ``WritebackRecord.new_facts`` entry must satisfy before
    TF-047 may turn it into a ``UserFactRecord``: content plus a
    provenance stamp from ``MEMORY_PROVENANCE`` are required
    there, because writeback is new ingestion, not legacy load.
  * ``contract_problems()`` — re-checks the contract itself
    against the live tree (dataclass defaults, summary caps as
    written in ``models.py``, schema provenance defaults, the
    store's wiring, the provenance gap), so drift is caught by
    name.

Confidence semantics, stated once: a confidence (and its kin
importance / quality / priority / recency_weight) is a
probability-like rating in [0, 1] — 0 = no trust, 1 = certain.
On the DB side *no kind carries a true confidence*: entities
carry ``popularity_score`` (a sitelink proxy, capped at 1.0 by
its estimator in ``etl/wikidata.py``) and ConceptNet edges carry
``weight`` (an unbounded strength, *not* a probability — the
forge caps its contribution at fusion time, TF-041 §5);
statements carry ``refs`` — provenance evidence, not a number.
The contract records those honestly instead of renaming them.

The human-readable write-up is
``docs/architecture/INGESTION_CONTRACT.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from datetime import datetime
from typing import Any, Mapping

from thoughtforge.knowledge import models as models_module

# ── The schema: six parts every fact carries ─────────────────────────────────

SCHEMA_PARTS: tuple[str, ...] = (
    "identity",     # a stable id naming this claim (record_id / qid / uri / …)
    "kind",         # which of FACT_KINDS it is (record_type / table)
    "content",      # the claim itself (summary+fact_value / label / triple / …)
    "provenance",   # where it came from (DB: source column; memory: see gap)
    "confidence",   # how far to trust it, in [0, 1] where the kind has one
    "lifecycle",    # status / retention / timestamps governing its ageing
)

# ── Provenance vocabularies ──────────────────────────────────────────────────
#
# DB_PROVENANCE is *measured*: it is exactly the set of `source`
# column defaults declared in etl/schema.py's _TABLES, and the
# forge's SQL source names plus dbpedia (TF-041 RETRIEVAL_MAP).
# contract_problems() re-proves both.

DB_PROVENANCE: tuple[str, ...] = (
    "wikidata", "conceptnet", "geonames", "dbpedia", "reference",
)

# MEMORY_PROVENANCE is *contract-defined*: the vocabulary for the
# provenance stamp on memory-side ingestion. At TF-042 no
# persisted memory record had a field to carry it — that absence
# was the slice's headline finding, pinned in
# contract_problems(). TF-047 closed it for the four writeback
# kinds: they carry a `source` field and the writeback leg
# (knowledge/writeback.py) stamps every record it writes.
# TF-057 (forge_memory toolset) stamps manual entries.

MEMORY_PROVENANCE: tuple[str, ...] = (
    "user_stated",            # the user said it, in so many words
    "conversation_writeback",  # extracted from a conversation by TF-047's leg
    "manual_entry",           # entered by hand through the memory toolset
    "import",                 # imported from a store bundle (TF-052)
    "synthetic_fixture",      # TF-017 synthetic stand-ins, tests only
)

# ── The kinds ────────────────────────────────────────────────────────────────
#
# One row per ingestible kind. Memory kinds name their dataclass
# and record_type; DB kinds name their table. "confidence_field"
# is (field, construction default) or None where the kind has no
# such field — with "confidence_note" saying what it has instead.

FACT_KINDS: tuple[dict[str, Any], ...] = (
    # — the file memory store (knowledge/store.py) —
    {"kind": "personality_core", "store": "memory",
     "record": "PersonalityCoreRecord", "record_type": "",
     "identity_fields": ("record_id",),
     "content_fields": ("traits", "speech_style", "behavior_rules"),
     "content_rule": "identity_only",
     "provenance_field": None,
     "confidence_field": None,
     "confidence_note": "no confidence: identity is versioned manually, never scored for trust",
     "summary_cap": None},
    {"kind": "user_preference", "store": "memory",
     "record": "UserPreferenceRecord", "record_type": "user_preference",
     "identity_fields": ("record_id",),
     "content_fields": ("category", "summary"),
     "content_rule": "category_and_summary",
     "provenance_field": "source",
     "confidence_field": ("confidence", 0.5),
     "confidence_note": "",
     "summary_cap": 160},
    {"kind": "user_fact", "store": "memory",
     "record": "UserFactRecord", "record_type": "user_fact",
     "identity_fields": ("record_id",),
     "content_fields": ("summary", "fact_value"),
     "content_rule": "summary_or_fact_value",
     "provenance_field": "source",
     "confidence_field": ("confidence", 0.6),
     "confidence_note": "",
     "summary_cap": 180},
    {"kind": "episode", "store": "memory",
     "record": "EpisodicMemoryRecord", "record_type": "episode",
     "identity_fields": ("record_id",),
     "content_fields": ("summary",),
     "content_rule": "summary",
     "provenance_field": "source",
     "confidence_field": ("confidence", 0.5),
     "confidence_note": "",
     "summary_cap": 240},
    {"kind": "response_pattern", "store": "memory",
     "record": "ResponsePatternRecord", "record_type": "response_pattern",
     "identity_fields": ("record_id",),
     "content_fields": ("summary", "pattern_shape"),
     "content_rule": "summary_or_pattern_shape",
     "provenance_field": "source",
     "confidence_field": ("confidence", 0.6),
     "confidence_note": "",
     "summary_cap": None},
    {"kind": "active_thread_state", "store": "memory",
     "record": "ActiveThreadStateRecord", "record_type": "active_thread_state",
     "identity_fields": ("record_id",),
     "content_fields": ("topic", "summary", "open_loops"),
     "content_rule": "identity_only",
     "provenance_field": None,
     "confidence_field": None,
     "confidence_note": "no confidence: priority (default 0.8, in [0, 1]) is urgency, not trust",
     "summary_cap": None},
    # — the SQLite knowledge database (etl/schema.py) —
    {"kind": "entity", "store": "knowledge_db",
     "record": "entities", "record_type": "",
     "identity_fields": ("qid",),
     "content_fields": ("label_en",),
     "content_rule": "label",
     "provenance_field": "source",
     "confidence_field": ("popularity_score", 0.0),
     "confidence_note": "popularity is a sitelink-count proxy capped at 1.0, not a trust rating",
     "summary_cap": None},
    {"kind": "statement", "store": "knowledge_db",
     "record": "statements", "record_type": "",
     "identity_fields": ("subject_qid", "property_pid"),
     "content_fields": ("object_value",),
     "content_rule": "triple",
     "provenance_field": "source",
     "confidence_field": None,
     "confidence_note": "no numeric confidence: refs carries provenance evidence (JSON text)",
     "summary_cap": None},
    {"kind": "conceptnet_edge", "store": "knowledge_db",
     "record": "conceptnet_edges", "record_type": "",
     "identity_fields": ("concept1", "relation", "concept2"),
     "content_fields": ("concept1", "relation", "concept2"),
     "content_rule": "triple",
     "provenance_field": "source",
     "confidence_field": None,
     "confidence_note": "weight (default 1.0) is an unbounded strength, NOT a probability — never range-checked as confidence",
     "summary_cap": None},
    {"kind": "geonames_place", "store": "knowledge_db",
     "record": "geonames_places", "record_type": "",
     "identity_fields": ("geonames_id",),
     "content_fields": ("name",),
     "content_rule": "label",
     "provenance_field": "source",
     "confidence_field": None,
     "confidence_note": "no confidence field at all",
     "summary_cap": None},
    {"kind": "dbpedia_entity", "store": "knowledge_db",
     "record": "dbpedia_entities", "record_type": "",
     "identity_fields": ("uri",),
     "content_fields": ("label_en", "abstract_en"),
     "content_rule": "label_or_abstract",
     "provenance_field": "source",
     "confidence_field": None,
     "confidence_note": "no confidence field at all",
     "summary_cap": None},
    {"kind": "reference_chunk", "store": "knowledge_db",
     "record": "reference_chunks", "record_type": "",
     "identity_fields": ("source_file", "chunk_index"),
     "content_fields": ("content",),
     "content_rule": "content",
     "provenance_field": "source_file",
     "confidence_field": None,
     "confidence_note": "no confidence: source_file is both identity and provenance",
     "summary_cap": None},
)

_KIND_BY_NAME: dict[str, dict[str, Any]] = {row["kind"]: row for row in FACT_KINDS}
_MEMORY_KIND_BY_RECORD_TYPE: dict[str, str] = {
    row["record_type"]: row["kind"]
    for row in FACT_KINDS
    if row["store"] == "memory" and row["record_type"]
}
_MEMORY_KIND_BY_CLASS: dict[str, str] = {
    row["record"]: row["kind"] for row in FACT_KINDS if row["store"] == "memory"
}

# Unit-interval fields on memory records, wherever the kind has them.
_UNIT_FIELDS: tuple[str, ...] = (
    "confidence", "importance", "quality", "priority", "recency_weight",
)
_COUNTER_FIELDS: tuple[str, ...] = (
    "times_reinforced", "times_retrieved", "times_successful",
    "turns_since_touch",
)
_HORIZON_FIELDS: tuple[str, ...] = ("stale_after_turns", "expires_after_turns")
_LIST_FIELDS: tuple[str, ...] = (
    "tags", "traits", "speech_style", "behavior_rules", "avoid",
    "linked_preferences", "linked_facts", "pattern_shape", "open_loops",
)
_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "created_at", "updated_at", "last_used_at", "last_confirmed_at",
)


def _as_dict(record: Any) -> dict[str, Any] | None:
    if isinstance(record, Mapping):
        return dict(record)
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return dataclasses.asdict(record)
    return None


def _infer_memory_kind(record: Any, data: Mapping[str, Any]) -> str | None:
    cls_name = type(record).__name__
    if cls_name in _MEMORY_KIND_BY_CLASS:
        return _MEMORY_KIND_BY_CLASS[cls_name]
    record_type = data.get("record_type", "")
    if isinstance(record_type, str) and record_type in _MEMORY_KIND_BY_RECORD_TYPE:
        return _MEMORY_KIND_BY_RECORD_TYPE[record_type]
    if "traits" in data or "speech_style" in data:
        return "personality_core"
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_unit(data: Mapping[str, Any], field_name: str, problems: list[str]) -> None:
    if field_name not in data or data[field_name] is None:
        return
    value = data[field_name]
    if not _is_number(value):
        problems.append(f"{field_name} is not a number: {value!r}")
    elif not math.isfinite(float(value)):
        problems.append(f"{field_name} is not finite: {value!r}")
    elif not 0.0 <= float(value) <= 1.0:
        problems.append(f"{field_name} {value} outside [0, 1]")


def record_problems(record: Any, *, kind: str | None = None) -> list[str]:
    """
    Validate one memory record (dataclass instance or dict, as
    ``from_dict`` would receive it) against the contract.
    Returns problem strings naming the field at fault; ``[]``
    means the record may be ingested. Pure — writes nothing.
    Unknown keys are NOT problems (``from_dict`` ignores them by
    design, TF-041 K12); a missing provenance stamp is not a
    problem *here* either — legacy records predate the stamp
    (TF-047 added the ``source`` field to the four writeback
    kinds with a ``""`` default). The stamp is contract-level,
    pinned by ``contract_problems()``, and becomes an
    ingestion-time requirement only on the writeback path
    (``writeback_fact_problems``).
    """
    data = _as_dict(record)
    if data is None:
        return [f"not a record or mapping: {type(record).__name__}"]
    resolved = kind or _infer_memory_kind(record, data)
    if resolved is None:
        return ["cannot determine the record kind (no known class or record_type)"]
    row = _KIND_BY_NAME.get(resolved)
    if row is None or row["store"] != "memory":
        return [f"kind {resolved!r} is not a memory ingestion kind"]
    problems: list[str] = []

    # identity
    record_id = data.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        problems.append("record_id missing or empty")

    # kind agreement: a fact may not ride in under another type's name
    expected_type = row["record_type"]
    if expected_type and data.get("record_type", expected_type) != expected_type:
        problems.append(
            f"record_type {data.get('record_type')!r} != kind {resolved} "
            f"({expected_type!r})"
        )

    # content, per the kind's rule
    rule = row["content_rule"]
    summary = data.get("summary", "")
    if rule == "summary":
        if not isinstance(summary, str) or not summary.strip():
            problems.append("summary missing or empty")
    elif rule == "category_and_summary":
        category = data.get("category", "")
        if not isinstance(category, str) or not category.strip():
            problems.append("category missing or empty")
        if not isinstance(summary, str) or not summary.strip():
            problems.append("summary missing or empty")
    elif rule == "summary_or_fact_value":
        fact_value = data.get("fact_value", {})
        has_summary = isinstance(summary, str) and bool(summary.strip())
        has_value = isinstance(fact_value, dict) and bool(fact_value)
        if not has_summary and not has_value:
            problems.append("fact carries no content: summary empty and fact_value empty")
        if "fact_value" in data and not isinstance(fact_value, dict):
            problems.append(f"fact_value is not a dict: {type(fact_value).__name__}")
    elif rule == "summary_or_pattern_shape":
        shape = data.get("pattern_shape", [])
        has_summary = isinstance(summary, str) and bool(summary.strip())
        has_shape = isinstance(shape, list) and bool(shape)
        if not has_summary and not has_shape:
            problems.append("pattern carries no content: summary empty and pattern_shape empty")
    # "identity_only" kinds (personality core, thread state) need
    # nothing beyond identity — ruled, not an oversight: a blank
    # persona is a valid (if empty) identity, and a thread-state
    # snapshot may legitimately stand empty between topics.

    cap = row["summary_cap"]
    if cap is not None and isinstance(summary, str) and len(summary) > cap:
        problems.append(f"summary {len(summary)} chars exceeds the {cap}-char cap")

    # confidence & kin: probability-like, in [0, 1]
    for field_name in _UNIT_FIELDS:
        _check_unit(data, field_name, problems)
    weights = data.get("weights")
    if isinstance(weights, dict):
        for key, value in weights.items():
            if not _is_number(value) or not math.isfinite(float(value)):
                problems.append(f"weights[{key!r}] is not a finite number: {value!r}")
    elif "weights" in data and weights is not None:
        problems.append(f"weights is not a dict: {type(weights).__name__}")

    # counters & horizons: non-negative whole numbers
    for field_name in _COUNTER_FIELDS + _HORIZON_FIELDS:
        if field_name not in data or data[field_name] is None:
            continue
        value = data[field_name]
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"{field_name} is not an int: {value!r}")
        elif value < 0:
            problems.append(f"{field_name} is negative: {value}")
    version = data.get("version")
    if version is not None and (
        not isinstance(version, int) or isinstance(version, bool) or version < 1
    ):
        problems.append(f"version is not a positive int: {version!r}")

    # lifecycle vocabularies are closed (TF-041 K8)
    from typing import get_args

    if "status" in data and data["status"] not in get_args(models_module.RecordStatus):
        problems.append(f"status {data['status']!r} outside the RecordStatus vocabulary")
    if "retention_class" in data and data["retention_class"] not in get_args(
        models_module.RetentionClass
    ):
        problems.append(
            f"retention_class {data['retention_class']!r} outside the RetentionClass vocabulary"
        )

    # list fields hold lists of strings
    for field_name in _LIST_FIELDS:
        if field_name not in data or data[field_name] is None:
            continue
        value = data[field_name]
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            problems.append(f"{field_name} is not a list of strings")

    # timestamps, when present and non-empty, parse as ISO
    for field_name in _TIMESTAMP_FIELDS:
        value = data.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"{field_name} is not ISO time: {value!r}")

    return problems


def db_row_problems(row: Mapping[str, Any], *, kind: str) -> list[str]:
    """
    Validate one knowledge-DB row (a mapping, as the ETL wing
    builds it or a ``sqlite3.Row`` converts) against the
    contract for its kind. Provenance IS required here: every
    DB kind's ``source`` (or ``source_file``) must name a member
    of ``DB_PROVENANCE`` — an absent source falls back to the
    schema default at INSERT time, which is itself in the
    vocabulary, so absence is accepted only where the schema
    declares a default (all but ``reference_chunks.source_file``
    and ``ingestion_log``-style NOT NULL columns).
    """
    spec = _KIND_BY_NAME.get(kind)
    if spec is None or spec["store"] != "knowledge_db":
        return [f"kind {kind!r} is not a knowledge-DB ingestion kind"]
    problems: list[str] = []
    data = dict(row)

    for field_name in spec["identity_fields"]:
        value = data.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"{field_name} missing or empty (identity)")

    rule = spec["content_rule"]
    if rule == "label":
        field_name = spec["content_fields"][0]
        value = data.get(field_name)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field_name} missing or empty (content)")
    elif rule == "label_or_abstract":
        if not any(
            isinstance(data.get(f), str) and data[f].strip()
            for f in spec["content_fields"]
        ):
            problems.append("neither label_en nor abstract_en carries content")
    elif rule == "triple":
        for field_name in spec["content_fields"]:
            value = data.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                problems.append(f"{field_name} missing or empty (content)")
    elif rule == "content":
        value = data.get("content")
        if not isinstance(value, str) or not value.strip():
            problems.append("content missing or empty")

    # provenance
    prov_field = spec["provenance_field"]
    if prov_field == "source_file":
        # Already checked as identity: the file IS the provenance,
        # and it must be present (schema: NOT NULL, no default).
        pass
    else:
        source = data.get("source")
        if source is not None and source not in DB_PROVENANCE:
            problems.append(
                f"source {source!r} outside the provenance vocabulary {DB_PROVENANCE}"
            )

    # confidence-like fields, per kind — never a blanket rule:
    # popularity is a capped proxy, weight is unbounded strength.
    if kind == "entity":
        _check_unit(data, "popularity_score", problems)
    if kind == "conceptnet_edge":
        weight = data.get("weight")
        if weight is not None:
            if not _is_number(weight) or not math.isfinite(float(weight)):
                problems.append(f"weight is not a finite number: {weight!r}")
            elif float(weight) < 0:
                problems.append(f"weight is negative: {weight}")
    if kind == "geonames_place":
        population = data.get("population")
        if population is not None and (
            not isinstance(population, int) or isinstance(population, bool)
            or population < 0
        ):
            problems.append(f"population is not a non-negative int: {population!r}")
    if kind == "reference_chunk":
        index = data.get("chunk_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            problems.append(f"chunk_index is not a non-negative int: {index!r}")

    return problems


def writeback_fact_problems(entry: Any) -> list[str]:
    """
    Validate one ``WritebackRecord.new_facts`` entry — a plain
    dict that TF-047's wiring would turn into a
    ``UserFactRecord``. Stricter than ``record_problems`` in
    exactly one place: provenance is REQUIRED (key ``source``,
    a member of ``MEMORY_PROVENANCE``), because writeback is new
    ingestion and the stamp must be applied at the gate, while
    the extractor still knows where the fact came from.
    """
    if not isinstance(entry, Mapping):
        return [f"writeback fact is not a dict: {type(entry).__name__}"]
    data = dict(entry)
    problems: list[str] = []

    summary = data.get("summary", "")
    fact_value = data.get("fact_value", {})
    has_summary = isinstance(summary, str) and bool(summary.strip())
    has_value = isinstance(fact_value, dict) and bool(fact_value)
    if not has_summary and not has_value:
        problems.append("fact carries no content: summary empty and fact_value empty")
    if isinstance(summary, str) and len(summary) > 180:
        problems.append(f"summary {len(summary)} chars exceeds the 180-char cap")
    if "fact_value" in data and not isinstance(fact_value, dict):
        problems.append(f"fact_value is not a dict: {type(fact_value).__name__}")

    source = data.get("source")
    if source is None:
        problems.append("provenance missing: writeback facts must carry a source stamp")
    elif source not in MEMORY_PROVENANCE:
        problems.append(
            f"source {source!r} outside the memory provenance vocabulary {MEMORY_PROVENANCE}"
        )

    _check_unit(data, "confidence", problems)
    _check_unit(data, "importance", problems)
    tags = data.get("tags")
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
    ):
        problems.append("tags is not a list of strings")
    return problems


def contract_problems() -> list[str]:
    """
    Re-check the contract against the live tree. Returns problems
    by name; ``[]`` when the contract still matches the code —
    including the pinned honesty records (the memory provenance
    gap, the store's validation wiring), so a future slice that
    changes either must update this contract in the same commit.
    """
    from thoughtforge.etl import schema as schema_module
    from thoughtforge.knowledge.store import MemoryStore
    from thoughtforge.knowledge.store_map import (
        KNOWLEDGE_DB_TABLES,
        PERSISTED_RECORDS,
        RETRIEVAL_MAP,
    )

    problems: list[str] = []

    # The kinds cover the mapped estate exactly: 6 persisted
    # memory records (TF-041) and the 6 fact-holding DB tables.
    memory_kinds = {r["kind"]: r for r in FACT_KINDS if r["store"] == "memory"}
    if {r["record"] for r in memory_kinds.values()} != {
        r["name"] for r in PERSISTED_RECORDS
    }:
        problems.append("memory kinds != TF-041's persisted records")
    db_kinds = {r["record"] for r in FACT_KINDS if r["store"] == "knowledge_db"}
    if not db_kinds <= {r["name"] for r in KNOWLEDGE_DB_TABLES}:
        problems.append("DB kinds name tables outside TF-041's map")
    if len(FACT_KINDS) != 12 or len(memory_kinds) != 6:
        problems.append(f"FACT_KINDS holds {len(FACT_KINDS)} rows, not 12 (6 + 6)")

    # Construction defaults match the live dataclasses.
    for row in FACT_KINDS:
        if row["store"] != "memory":
            continue
        cls = getattr(models_module, row["record"])
        required = {
            f.name: "contract_probe"
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        }
        record = cls(**required)
        if row["confidence_field"] is not None:
            field_name, default = row["confidence_field"]
            if getattr(record, field_name) != default:
                problems.append(
                    f"{row['kind']}: {field_name} default "
                    f"{getattr(record, field_name)} != contract {default}"
                )
        if getattr(record, "record_type", "") != row["record_type"] and (
            row["kind"] != "personality_core"
        ):
            problems.append(f"{row['kind']}: record_type drifted from the contract")

    # Summary caps are the caps models.py itself states in prose.
    models_src = inspect.getsource(models_module)
    for cap in (160, 180, 240):
        if f"max {cap} chars" not in models_src:
            problems.append(f"summary cap {cap} no longer stated in models.py")

    # DB provenance: the vocabulary IS the schema's source defaults
    # and the forge's source names (+ dbpedia, ETL-side today).
    schema_sources = set(re.findall(r"source\s+TEXT DEFAULT '(\w+)'", schema_module._TABLES))
    if schema_sources != set(DB_PROVENANCE):
        problems.append(
            f"schema source defaults {sorted(schema_sources)} != DB_PROVENANCE"
        )
    forge_sources = set(RETRIEVAL_MAP["sql_sources"]) | {"dbpedia"}
    if forge_sources != set(DB_PROVENANCE):
        problems.append("forge SQL sources + dbpedia != DB_PROVENANCE")
    if len(set(DB_PROVENANCE)) != len(DB_PROVENANCE):
        problems.append("DB_PROVENANCE holds duplicates")
    if len(set(MEMORY_PROVENANCE)) != len(MEMORY_PROVENANCE):
        problems.append("MEMORY_PROVENANCE holds duplicates")
    if set(MEMORY_PROVENANCE) & set(DB_PROVENANCE):
        problems.append("provenance vocabularies overlap — the two stores' stamps must stay distinguishable")

    # THE PROVENANCE SPLIT, pinned (was TF-042's gap pin; TF-047
    # closed the gap for the writeback kinds and this pin now
    # checks the split both ways): a kind whose FACT_KINDS row
    # promises a `source` stamp must carry the field, and a kind
    # whose row promises none (identity, thread state) must not
    # grow one silently.
    for row in FACT_KINDS:
        if row["store"] != "memory":
            continue
        cls = getattr(models_module, row["record"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        if row["provenance_field"] == "source":
            if "source" not in field_names:
                problems.append(
                    f"provenance: {row['record']} promises a source stamp "
                    "in FACT_KINDS but carries no source field"
                )
        elif "source" in field_names or "provenance" in field_names:
            problems.append(
                f"provenance: {row['record']} gained a provenance field "
                "but its FACT_KINDS row promises none — update the row "
                "and this pin with the wiring evidence"
            )

    # The store's ingestion gates are wired to the validator:
    # each gate calls the store's one _validate_record helper,
    # and that helper calls record_problems.
    from thoughtforge.knowledge import store as store_module

    helper_src = inspect.getsource(store_module._validate_record)
    if "record_problems" not in helper_src:
        problems.append("MemoryStore._validate_record no longer calls record_problems")
    for method_name in (
        "append_profile_record", "append_episode", "append_pattern",
        "save_personality_core", "save_thread_state",
    ):
        src = inspect.getsource(getattr(MemoryStore, method_name))
        if "_validate_record" not in src:
            problems.append(
                f"MemoryStore.{method_name} no longer validates at ingestion"
            )

    return problems


__all__ = [
    "DB_PROVENANCE",
    "FACT_KINDS",
    "MEMORY_PROVENANCE",
    "SCHEMA_PARTS",
    "contract_problems",
    "db_row_problems",
    "record_problems",
    "writeback_fact_problems",
]
