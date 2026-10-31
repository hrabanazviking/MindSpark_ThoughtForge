"""
Knowledge store map, as data (TF-041).

Part III opens by mapping the sovereign memory exactly as the
code stands — the first Part I/II-style census of the knowledge
layer, kept as *verifiable data* rather than prose alone (the
pattern TF-039 set for the core docs):

  * ``KNOWLEDGE_DB_TABLES`` / ``KNOWLEDGE_DB_FTS`` — the SQLite
    knowledge database behind ``KnowledgeForge``
    (``etl/schema.py``): 10 base tables + 3 FTS5 indexes, who
    writes them (the ETL wing) and who reads them (the forge).
  * ``MEMORY_FILES`` — the file-based ``MemoryStore`` in the
    memory dir: 7 files, their formats, and the record types
    each one holds.
  * ``PERSISTED_RECORDS`` — the 6 record types that persist,
    with the storage each lands in and its construction
    defaults (status / retention / stale horizon), checked
    against the live dataclasses.
  * ``TRANSIENT_RECORDS`` — the 9 record types that never
    persist (turn-time only), named so nobody goes looking for
    them on disk.
  * ``RETRIEVAL_MAP`` — how ``KnowledgeForge.retrieve`` actually
    flows: the three paths, the four SQL sources, the fusion
    proxies it really uses, and the threshold that filters.
  * ``INVARIANTS`` — the store's invariants, stated as data.
    ``store_map_problems()`` re-checks every machine-checkable
    one against the LIVE tree, and the TF-041 tests exercise
    the behavioural ones against synthetic stores, so a future
    change that silently breaks an invariant (a weight row
    that stops summing to 1, the retrieval gate drifting back
    to the default DB path, a prune threshold inverting) is
    caught by name.

Two honesty records this map carries, not hides:

  * **Lifecycle wiring (TF-045).** ``MemoryLifecycle`` was
    mapped-but-unwired at TF-041 (TF-011's D-ORPHAN-1); TF-045
    wired it into the core: the active thread state is cooled
    once per completed turn (``_finish_turn``) and
    ``ThoughtForgeCore.prune_memory`` runs an explicit prune
    pass over the core's store. Fact pruning is never automatic
    per turn. ``store_map_problems`` (K10) now pins the wiring's
    presence — if it is removed, the checker names the loss.
    The lifecycle's own rules and edges are stated in
    ``knowledge/lifecycle_rules.py`` (TF-045).
  * **Forge scoring is a proxy, recorded as such.** The forge
    scores every candidate with the ``user_fact`` weight row
    whatever its real type, tone/preference as neutral 0.5,
    recency as a per-source constant, importance as
    popularity+weight capped at 1, and semantic similarity as
    ``min(fts/20, 1)`` when no vector score exists. With a
    one-row FTS corpus those proxies score real hits *below*
    the 0.45 bundle threshold (measured in the TF-041 doc);
    TF-043 pins retrieval precision and TF-044 owns making
    the weights honest and tunable. This slice maps; it does
    not retune.

The human-readable write-up is
``docs/architecture/KNOWLEDGE_STORE_MAP.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import sqlite3
import tempfile
import textwrap
from pathlib import Path
from typing import Any, get_args

# ── The SQLite knowledge database (etl/schema.py) ────────────────────────────
#
# kind: "table" rows are the base tables of _TABLES; the FTS
# rows are the virtual tables of _FTS5. "read_by" names the
# KnowledgeForge SQL pass that reads the table today, or ""
# where no retrieval pass reads it (statements/labels/
# embeddings are ETL-side today — embeddings are read through
# etl/embeddings.py's vector search, not the SQL passes).

KNOWLEDGE_DB_TABLES: tuple[dict[str, Any], ...] = (
    {"name": "entities", "key": "qid TEXT PRIMARY KEY",
     "holds": "Wikidata entities: label/description/aliases/instance_of/popularity",
     "read_by": "sql entity pass (via entities_fts) + vector entity search"},
    {"name": "statements", "key": "statement_id AUTOINCREMENT",
     "holds": "subject–property–object facts about entities",
     "read_by": ""},
    {"name": "labels", "key": "(qid, language)",
     "holds": "multi-language labels per entity", "read_by": ""},
    {"name": "embeddings", "key": "qid TEXT PRIMARY KEY",
     "holds": "pre-computed entity embedding BLOBs (float32)",
     "read_by": "vector entity search (etl/embeddings.py)"},
    {"name": "conceptnet_edges", "key": "edge_id AUTOINCREMENT",
     "holds": "commonsense edges (concept1, relation, concept2, weight)",
     "read_by": "sql conceptnet pass (direct WHERE, not FTS)"},
    {"name": "geonames_places", "key": "geonames_id INTEGER PRIMARY KEY",
     "holds": "places: name/coords/feature/population",
     "read_by": "sql geonames pass (name LIKE, first key term only)"},
    {"name": "dbpedia_entities", "key": "uri TEXT PRIMARY KEY",
     "holds": "DBpedia entries: label/abstract/types", "read_by": ""},
    {"name": "reference_chunks", "key": "chunk_id AUTOINCREMENT",
     "holds": "chunked reference documents (source_file, chunk_index unique)",
     "read_by": "sql reference pass (via reference_fts) + vector reference search"},
    {"name": "reference_embeddings", "key": "chunk_id PRIMARY KEY",
     "holds": "pre-computed reference-chunk embedding BLOBs",
     "read_by": "vector reference search (etl/embeddings.py)"},
    {"name": "ingestion_log", "key": "log_id AUTOINCREMENT",
     "holds": "ETL run log (source, times, counts, status)", "read_by": ""},
)

KNOWLEDGE_DB_FTS: tuple[dict[str, Any], ...] = (
    {"name": "entities_fts", "over": "entities",
     "columns": ("qid", "label_en", "description_en", "aliases_en")},
    {"name": "reference_fts", "over": "reference_chunks",
     "columns": ("chunk_id", "title", "content", "tags")},
    {"name": "conceptnet_fts", "over": "conceptnet_edges",
     "columns": ("edge_id", "concept1", "relation", "concept2")},
)

# ── The file-based memory store (knowledge/store.py) ─────────────────────────
#
# One row per file MemoryStore manages in the memory dir.
# "holds" names the record types stored there; the derived
# files (tag index, recent queue) hold ids, not records.

MEMORY_FILES: tuple[dict[str, Any], ...] = (
    {"file": "personality_core.yaml", "format": "yaml-single",
     "holds": ("PersonalityCoreRecord",), "attr": "_personality_path"},
    {"file": "user_profile_store.jsonl", "format": "jsonl",
     "holds": ("UserPreferenceRecord", "UserFactRecord"),
     "attr": "_profile_path"},
    {"file": "episodic_store.jsonl", "format": "jsonl",
     "holds": ("EpisodicMemoryRecord",), "attr": "_episodic_path"},
    {"file": "response_patterns.jsonl", "format": "jsonl",
     "holds": ("ResponsePatternRecord",), "attr": "_patterns_path"},
    {"file": "active_thread_state.json", "format": "json-single",
     "holds": ("ActiveThreadStateRecord",), "attr": "_thread_path"},
    {"file": "tag_index.json", "format": "json-derived",
     "holds": (), "attr": "_tag_index_path"},
    {"file": "recent_queue.json", "format": "json-derived",
     "holds": (), "attr": "_recent_queue_path"},
)

# ── The persisted record types (knowledge/models.py) ─────────────────────────
#
# Defaults are the dataclass construction defaults, checked
# live by store_map_problems(): status / retention_class /
# record_type, plus "stale_after_turns" where the type carries
# one (the thread state carries expires_after_turns instead).

PERSISTED_RECORDS: tuple[dict[str, Any], ...] = (
    {"name": "PersonalityCoreRecord", "record_type": "",
     "file": "personality_core.yaml",
     "status": "active", "retention_class": "protected",
     "stale_after_turns": None,
     "note": "stable identity; never auto-pruned; versioned manually"},
    {"name": "UserPreferenceRecord", "record_type": "user_preference",
     "file": "user_profile_store.jsonl",
     "status": "created", "retention_class": "standard",
     "stale_after_turns": 200,
     "note": "tone/style preference; soft-decays if unconfirmed"},
    {"name": "UserFactRecord", "record_type": "user_fact",
     "file": "user_profile_store.jsonl",
     "status": "created", "retention_class": "standard",
     "stale_after_turns": 500,
     "note": "durable contextual fact (fact_value is a dict)"},
    {"name": "EpisodicMemoryRecord", "record_type": "episode",
     "file": "episodic_store.jsonl",
     "status": "created", "retention_class": "ephemeral",
     "stale_after_turns": 30,
     "note": "compact summary of a notable conversation moment"},
    {"name": "ResponsePatternRecord", "record_type": "response_pattern",
     "file": "response_patterns.jsonl",
     "status": "created", "retention_class": "standard",
     "stale_after_turns": 100,
     "note": "reusable successful response structure"},
    {"name": "ActiveThreadStateRecord", "record_type": "active_thread_state",
     "file": "active_thread_state.json",
     "status": "active", "retention_class": "ephemeral",
     "stale_after_turns": None,
     "note": "hot-path thread state; expires_after_turns=10, cool_down() per turn"},
)

TRANSIENT_RECORDS: tuple[str, ...] = (
    "InputSketch", "ActivatedRecord", "MemoryActivationBundle",
    "CognitionScaffold", "CandidateRecord", "FragmentRecord",
    "FinalResponseRecord", "WritebackRecord", "RuntimeTurnState",
)

# ── Retrieval, as it actually flows (knowledge/forge.py) ─────────────────────

RETRIEVAL_MAP: dict[str, Any] = {
    "paths": ("sql", "vector", "hybrid"),
    "default_path": "hybrid",
    "default_top_k": 10,
    "core_top_k": 10,  # core._retrieve's flat call — TF-098 keeps it
    "sql_sources": ("wikidata", "reference", "conceptnet", "geonames"),
    "sql_passes": (
        "entities via entities_fts (bm25, top_k*2 asked)",
        "reference chunks via reference_fts (bm25, top_k//2 asked)",
        "conceptnet edges by key-term WHERE (top_k//3 asked)",
        "geonames places by first-key-term LIKE (top_k//5 asked)",
    ),
    "hybrid_dedupe": "vector hits dropped when their qid already came from SQL",
    "scoring": {
        "weights_row_for_every_record": "user_fact",
        "semantic_similarity": "vector similarity, else min(fts_score/20, 1)",
        "tone_similarity": 0.5,
        "preference_relevance": 0.5,
        "recency_by_source": {
            "reference": 0.9, "wikidata": 0.7, "conceptnet": 0.8,
            "geonames": 0.75, "dbpedia": 0.65, "default": 0.7,
        },
        "importance": "min(popularity_score + weight (default 0.5), 1)",
    },
    "cue": "label + first 80 chars of description/content, capped at 18 words",
    "bundle": "total_retrieved == len(activated); confidence == mean(score)",
}

# ── The invariants ────────────────────────────────────────────────────────────
#
# Each invariant is stated once here; store_map_problems()
# checks the machine-checkable ones against the live tree and
# the TF-041 tests exercise the behavioural ones (marked
# "tested") against synthetic stores and a synthetic DB.

INVARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "K1", "check": "problems",
     "statement": "every retrieval/prune/survival weight row sums to 1.0, and each writeback formula's weights sum to 1.0 (probed with unit components)"},
    {"id": "K2", "check": "problems",
     "statement": "the six score thresholds stand at 0.45/0.54/0.68/0.72/0.58/0.78 (memory/fragment/final/preference-writeback/episode-writeback/pattern-learning)"},
    {"id": "K3", "check": "problems",
     "statement": "recency is a half-life decay: score(0 days) == 1, score(one half-life) == 0.5, strictly falling; half-lives order thread < episode < pattern < preference < fact < personality"},
    {"id": "K4", "check": "problems",
     "statement": "prune thresholds are monotone: per kind, archive and delete fall light >= routine >= heavy >= emergency, and delete > archive in every cell"},
    {"id": "K5", "check": "problems",
     "statement": "every store cap's hard limit is >= its soft limit"},
    {"id": "K6", "check": "problems",
     "statement": "the schema creates exactly the mapped 10 base tables + 3 FTS indexes, idempotently"},
    {"id": "K7", "check": "problems",
     "statement": "MemoryStore manages exactly the 7 mapped files in the memory dir, and load_all returns exactly the 5 mapped sections"},
    {"id": "K8", "check": "problems",
     "statement": "each persisted record type constructs with the mapped status/retention/record_type/stale defaults, and the status and retention vocabularies are the closed 7- and 5-value sets"},
    {"id": "K9", "check": "problems",
     "statement": "the core's retrieval gate reads the CONFIGURED db path (self._knowledge.db_path), never the default path (CORE_LOOP F2, fixed by TF-041)"},
    {"id": "K10", "check": "problems",
     "statement": "MemoryLifecycle IS wired into the core by TF-045 (D-ORPHAN-1 fixed): _finish_turn cools the thread state once per completed turn and prune_memory runs an explicit prune pass; fact pruning is never automatic per turn"},
    {"id": "K11", "check": "tested",
     "statement": "a bundle honours its contract: total == len(records), confidence == mean(score), scores sorted descending, every score >= min_score (rounding aside), at most top_k records, every cue <= 18 words"},
    {"id": "K12", "check": "tested",
     "statement": "store round-trips are faithful: every persisted type survives save/load with its fields intact, and from_dict ignores unknown keys"},
    {"id": "K13", "check": "tested",
     "statement": "rewrites are the deletion mechanism: a record marked archived/deleted is absent from the file after a rewrite; malformed JSONL lines are skipped on load, never fatal; a missing file loads as empty/None"},
    {"id": "K14", "check": "tested",
     "statement": "the tag index covers profile+episode+pattern records (never personality/thread), dedupes ids per tag; the recent queue is most-recent-first, deduped, capped at 64 by default"},
    {"id": "K15", "check": "tested",
     "statement": "pruning never touches a protected record, in any mode; an ancient unprotected record is archived or deleted by the emergency pass and is then gone from its file (K13)"},
    {"id": "K16", "check": "tested",
     "statement": "the forge's fusion arithmetic is the mapped proxy: a record's composite equals score_retrieval(user_fact weights, mapped proxy dims) exactly"},
)


def bundle_problems(bundle: Any, *, top_k: int, min_score: float) -> list[str]:
    """
    Pure checker for invariant K11 on one
    ``MemoryActivationBundle``: the bundle's own bookkeeping
    (total, confidence, ordering, threshold, cap, cue length).
    Scores are rounded to 4 decimals in the bundle, so the
    threshold is judged with that rounding slack.
    """
    problems: list[str] = []
    records = list(bundle.activated_records)
    if bundle.total_retrieved != len(records):
        problems.append(
            f"total_retrieved {bundle.total_retrieved} != {len(records)} records"
        )
    if len(records) > top_k:
        problems.append(f"bundle holds {len(records)} records > top_k {top_k}")
    scores = [r.score for r in records]
    if scores != sorted(scores, reverse=True):
        problems.append("bundle scores are not sorted descending")
    for r in records:
        if r.score < min_score - 0.0001:
            problems.append(
                f"record {r.record_id} score {r.score} below min_score {min_score}"
            )
        if len(r.cue.split()) > 18:
            problems.append(f"record {r.record_id} cue exceeds 18 words")
    expected = round(sum(scores) / len(scores), 4) if scores else 0.0
    if bundle.retrieval_confidence != expected:
        problems.append(
            f"retrieval_confidence {bundle.retrieval_confidence} != mean {expected}"
        )
    return problems


def store_map_problems() -> list[str]:
    """
    Re-check the map's machine-checkable invariants (K1–K10)
    against the live tree. Returns problems by invariant id —
    ``[]`` when the map still matches the code exactly.
    """
    from thoughtforge.cognition import core as core_module
    from thoughtforge.etl import schema as schema_module
    from thoughtforge.knowledge import lifecycle as lifecycle_module
    from thoughtforge.knowledge import models as models_module
    from thoughtforge.knowledge import scoring as scoring_module
    from thoughtforge.knowledge.forge import KnowledgeForge
    from thoughtforge.knowledge.store import MemoryStore

    problems: list[str] = []

    # K1 — every weight row sums to 1.
    def _sum_ok(label: str, total: float) -> None:
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            problems.append(f"K1: {label} weights sum to {total}, not 1.0")

    _sum_ok("RETRIEVAL_WEIGHTS_BASE", sum(scoring_module.RETRIEVAL_WEIGHTS_BASE.values()))
    for rtype, row in scoring_module.RETRIEVAL_WEIGHTS_BY_TYPE.items():
        _sum_ok(f"RETRIEVAL_WEIGHTS_BY_TYPE[{rtype}]", sum(row.values()))
    _sum_ok("PRUNE_SCORE_WEIGHTS", sum(scoring_module.PRUNE_SCORE_WEIGHTS.values()))
    _sum_ok("SURVIVAL_SCORE_WEIGHTS", sum(scoring_module.SURVIVAL_SCORE_WEIGHTS.values()))
    wb = scoring_module.WritebackComponents
    for fn, fields in (
        (scoring_module.compute_preference_writeback_score,
         ("durability", "novelty", "confidence", "importance", "future_relevance")),
        (scoring_module.compute_episode_writeback_score,
         ("novelty", "confidence", "importance", "future_relevance", "thread_relevance")),
        (scoring_module.compute_pattern_learning_score,
         ("final_response_quality", "pattern_reusability", "future_relevance")),
    ):
        _sum_ok(fn.__name__, sum(fn(wb(**{f: 1.0})) for f in fields))

    # K2 — the thresholds stand where the map puts them.
    expected_thresholds = {
        "MIN_MEMORY_SCORE": 0.45, "MIN_FRAGMENT_SCORE": 0.54,
        "MIN_FINAL_RESPONSE_SCORE": 0.68,
        "MIN_PREFERENCE_WRITEBACK_SCORE": 0.72,
        "MIN_EPISODE_WRITEBACK_SCORE": 0.58,
        "MIN_PATTERN_LEARNING_SCORE": 0.78,
    }
    for name, value in expected_thresholds.items():
        if getattr(scoring_module, name) != value:
            problems.append(f"K2: {name} moved from {value}")

    # K3 — recency half-lives: values, ordering, decay shape.
    half = scoring_module.HALF_LIFE_DAYS
    order = ["active_thread_state", "episode", "response_pattern",
             "user_preference", "user_fact", "personality_core"]
    if set(half) != set(order):
        problems.append(f"K3: HALF_LIFE_DAYS keys {sorted(half)} != the mapped 6")
    elif [half[k] for k in order] != sorted(half[k] for k in order):
        problems.append("K3: half-life ordering changed")
    for rtype in order:
        if scoring_module.compute_recency_score(0.0, rtype) != 1.0:
            problems.append(f"K3: recency(0, {rtype}) != 1.0")
        if not math.isclose(
            scoring_module.compute_recency_score(half[rtype], rtype), 0.5
        ):
            problems.append(f"K3: recency(half-life, {rtype}) != 0.5")

    # K4/K5 — lifecycle thresholds monotone; caps sane.
    modes = ["light", "routine", "heavy", "emergency"]
    kinds = ["preference", "fact", "episode", "pattern"]
    for table_name in ("ARCHIVE_THRESHOLDS", "DELETE_THRESHOLDS"):
        table = getattr(lifecycle_module, table_name)
        if set(table) != set(modes):
            problems.append(f"K4: {table_name} modes != the mapped 4")
            continue
        for kind in kinds:
            column = [table[m][kind] for m in modes]
            if column != sorted(column, reverse=True):
                problems.append(f"K4: {table_name}[{kind}] not monotone: {column}")
    for mode in modes:
        for kind in kinds:
            if not (
                lifecycle_module.DELETE_THRESHOLDS[mode][kind]
                > lifecycle_module.ARCHIVE_THRESHOLDS[mode][kind]
            ):
                problems.append(f"K4: delete <= archive at {mode}/{kind}")
    for name, caps in lifecycle_module.STORE_CAPS.items():
        if caps["hard"] < caps["soft"]:
            problems.append(f"K5: STORE_CAPS[{name}] hard < soft")

    # K6 — the schema creates exactly the mapped tables, twice-safe.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "map_probe.db"
        schema_module.initialize_schema(db)
        schema_module.initialize_schema(db)  # idempotent
        conn = sqlite3.connect(str(db))
        try:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
        finally:
            conn.close()
    expected_tables = {r["name"] for r in KNOWLEDGE_DB_TABLES} | {
        r["name"] for r in KNOWLEDGE_DB_FTS
    }
    fts_shadows = {
        n for n in names
        if any(n.startswith(f"{fts}_") for fts in
               (r["name"] for r in KNOWLEDGE_DB_FTS))
    }
    extra = names - expected_tables - {"sqlite_sequence"} - fts_shadows
    missing = expected_tables - names
    if extra or missing:
        problems.append(f"K6: schema tables differ — extra {sorted(extra)}, missing {sorted(missing)}")
    for row in KNOWLEDGE_DB_TABLES:
        if f"CREATE TABLE IF NOT EXISTS {row['name']}" not in schema_module._TABLES:
            problems.append(f"K6: {row['name']} not declared in schema _TABLES")
    for row in KNOWLEDGE_DB_FTS:
        if row["name"] not in schema_module._FTS5:
            problems.append(f"K6: {row['name']} not declared in schema _FTS5")

    # K7 — MemoryStore manages exactly the mapped files.
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(memory_dir=Path(tmp) / "mem")
        for row in MEMORY_FILES:
            attr_path = getattr(store, row["attr"], None)
            if attr_path is None or attr_path.name != row["file"]:
                problems.append(f"K7: MemoryStore {row['attr']} is not {row['file']}")
        if set(store.load_all()) != {
            "personality_core", "profile_records", "episodes",
            "patterns", "thread_state",
        }:
            problems.append("K7: load_all sections != the mapped 5")

    # K8 — persisted defaults + closed vocabularies.
    if get_args(models_module.RecordStatus) != (
        "created", "reinforced", "active", "cooling", "stale",
        "archived", "deleted",
    ):
        problems.append("K8: RecordStatus vocabulary changed")
    if get_args(models_module.RetentionClass) != (
        "protected", "durable", "standard", "ephemeral", "discardable",
    ):
        problems.append("K8: RetentionClass vocabulary changed")
    for row in PERSISTED_RECORDS:
        cls = getattr(models_module, row["name"])
        required = {
            f.name: "map_probe"
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        }
        record = cls(**required)
        for field_name in ("status", "retention_class"):
            if getattr(record, field_name) != row[field_name]:
                problems.append(
                    f"K8: {row['name']}.{field_name} default "
                    f"{getattr(record, field_name)} != mapped {row[field_name]}"
                )
        if getattr(record, "record_type", "") != row["record_type"] and row["name"] != "PersonalityCoreRecord":
            problems.append(f"K8: {row['name']}.record_type != {row['record_type']}")
        if row["stale_after_turns"] is not None and (
            getattr(record, "stale_after_turns", None) != row["stale_after_turns"]
        ):
            problems.append(f"K8: {row['name']}.stale_after_turns moved")
        fields = {f.name for f in dataclasses.fields(cls)}
        if row["name"] == "ActiveThreadStateRecord" and record.expires_after_turns != 10:
            problems.append("K8: ActiveThreadStateRecord.expires_after_turns != 10")
        if "record_id" not in fields:
            problems.append(f"K8: {row['name']} lost record_id")
    all_records = set(PERSISTED_RECORDS[i]["name"] for i in range(len(PERSISTED_RECORDS)))
    if all_records | set(TRANSIENT_RECORDS) != {
        "PersonalityCoreRecord", "UserPreferenceRecord", "UserFactRecord",
        "EpisodicMemoryRecord", "ResponsePatternRecord",
        "ActiveThreadStateRecord", "InputSketch", "ActivatedRecord",
        "MemoryActivationBundle", "CognitionScaffold", "CandidateRecord",
        "FragmentRecord", "FinalResponseRecord", "WritebackRecord",
        "RuntimeTurnState",
    }:
        problems.append("K8: the 15 record types (6 persisted + 9 transient) changed")

    # K9 — the retrieval gate reads the configured path (F2 fixed).
    # AST-checked (calls/names), not substring-checked: the fixed
    # method's own docstring names the old call while explaining
    # the fix, and prose must not read as a reopening.
    import ast

    retrieve_src = inspect.getsource(core_module.ThoughtForgeCore._retrieve)
    called = {
        node.func.id
        for node in ast.walk(ast.parse(textwrap.dedent(retrieve_src)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if "get_knowledge_db_path" in called:
        problems.append("K9: _retrieve again calls the default DB path (F2 reopened)")
    if "self._knowledge.db_path" not in retrieve_src:
        problems.append("K9: _retrieve does not gate on self._knowledge.db_path")

    # K10 — lifecycle wiring, as TF-045 landed it. The core must
    # hold a MemoryLifecycle over its store, cool the thread in
    # _finish_turn, and expose prune_memory; losing any leg is a
    # named regression of the D-ORPHAN-1 fix.
    core_src = inspect.getsource(core_module)
    if "MemoryLifecycle" not in core_src:
        problems.append(
            "K10: the core no longer references MemoryLifecycle — "
            "the TF-045 wiring (D-ORPHAN-1 fix) has regressed"
        )
    finish_src = inspect.getsource(core_module.ThoughtForgeCore._finish_turn)
    if "cool_thread" not in finish_src:
        problems.append(
            "K10: _finish_turn no longer cools the thread state — "
            "the TF-045 per-turn aging has regressed"
        )
    if not callable(getattr(core_module.ThoughtForgeCore, "prune_memory", None)):
        problems.append(
            "K10: ThoughtForgeCore.prune_memory is gone — the "
            "TF-045 explicit prune surface has regressed"
        )

    # Retrieval map defaults still match the forge's signature.
    sig = inspect.signature(KnowledgeForge.retrieve)
    if sig.parameters["top_k"].default != RETRIEVAL_MAP["default_top_k"]:
        problems.append("retrieval map: forge default top_k moved")
    if sig.parameters["path"].default != RETRIEVAL_MAP["default_path"]:
        problems.append("retrieval map: forge default path moved")
    if sig.parameters["min_score"].default != scoring_module.MIN_MEMORY_SCORE:
        problems.append("retrieval map: forge default min_score != MIN_MEMORY_SCORE")

    return problems


__all__ = [
    "INVARIANTS",
    "KNOWLEDGE_DB_FTS",
    "KNOWLEDGE_DB_TABLES",
    "MEMORY_FILES",
    "PERSISTED_RECORDS",
    "RETRIEVAL_MAP",
    "TRANSIENT_RECORDS",
    "bundle_problems",
    "store_map_problems",
]
