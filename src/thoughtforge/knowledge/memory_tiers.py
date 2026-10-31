"""
TF-046 — memory tiers, stated as verifiable data with verdicts.

The roadmap asks for working / episodic / semantic separation
"where the code hints it; unified where it doesn't". The code's
hints, measured — never asserted — are these:

  * **Working — SEPARATE, and the code hints it twice.** No type
    is named "working memory", but ``ActiveThreadStateRecord``
    is built exactly like one: a SINGLE record in its own file,
    retention ``ephemeral``, ``expires_after_turns=10``, the
    shortest half-life in the ladder (0.07 days, ~2 turns),
    cooled once per completed turn (TF-045), and the one memory
    record the core reads on EVERY turn (routing reads it in
    Step 1). Around it sit the transient working surfaces — the
    turn state, the sketch, the scaffold, the activation bundle,
    and the caller-owned ``ChatHistory`` window (TF-024 budgets:
    20 turns / 1,000 word-tokens / 2,000 chars).
  * **Episodic — SEPARATE, and the code names it.** ``episode``
    records live alone in ``episodic_store.jsonl``, retention
    ``ephemeral`` yet persisted, ``stale_after_turns=30`` (the
    shortest stale horizon), half-life 7 days. An episode is the
    only record type that LINKS across tiers: its
    ``linked_preferences`` / ``linked_facts`` point into the
    semantic tier. Nothing in the live core loads episodes
    today — the tier exists as storage and lifecycle, not as a
    retrieval source (pinned below, not hidden).
  * **Semantic — UNIFIED in the personal store, SPLIT across
    stores overall.** Personal semantic memory is
    ``user_fact`` + ``user_preference``: two record types
    interleaved in ONE file (``user_profile_store.jsonl``),
    separated only by their ``record_type`` field and one
    unified loader (``load_profile_records``) — the code does
    not give facts and preferences separate homes, so this
    slice does not either. World semantic memory is a different
    store entirely: the SQLite knowledge DB (entities,
    statements, reference chunks, edges, places), owned by the
    ETL wing and read by ``KnowledgeForge``. The two semantic
    halves are unified NOWHERE in retrieval: the core's
    ``_retrieve`` reads the knowledge DB only, and no code
    outside store/lifecycle ever loads the personal tier.

Two persisted types sit OUTSIDE the three tiers, by ruling:

  * **Identity** — ``PersonalityCoreRecord``: protected, never
    auto-pruned, half-life 9,999 days, loaded at core
    construction and steering every turn (TF-030). It is not a
    tier of memory; it is who is remembering.
  * **Procedural** — ``ResponsePatternRecord``: reusable
    response structures with their own file, half-life, stale
    horizon, and learning scores. That is a genuine fourth-tier
    hint; TF-046 records it as procedural, outside the named
    three, and builds no fourth tier on a hint alone.

What IS unified, across every tier: one ``MemoryStore`` and
one ``load_all``, one ingestion contract (TF-042), one tag
index (rebuilt over profile + episodes + patterns), one
recent queue, one lifecycle prune pass (TF-045), and one
half-life ladder in scoring (TF-041 K3) whose ordering IS the
tier ordering: working < episodic < procedural < preference
< fact < identity.

``tier_problems()`` re-derives every claim above from the live
tree and names any that stops being true — including the
retrieval gap, so the slice that wires personal-tier
retrieval (TF-047's writeback is the inbound leg) must flip
the pin with evidence.

The human-readable write-up is
``docs/architecture/MEMORY_TIERS.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from thoughtforge.knowledge.models import EpisodicMemoryRecord
from thoughtforge.knowledge.scoring import HALF_LIFE_DAYS
from thoughtforge.knowledge.store_map import (
    MEMORY_FILES,
    PERSISTED_RECORDS,
    TRANSIENT_RECORDS,
)

# ── The tiers, as data ───────────────────────────────────────────────────────

#: The three tiers the roadmap names, in half-life order.
TIER_NAMES: tuple[str, ...] = ("working", "episodic", "semantic")

#: Record type -> tier (or outside-tier) for every persisted
#: record type. PersonalityCoreRecord carries no record_type
#: field; its key here is the TF-041 map's "" row, named.
TIER_BY_RECORD_TYPE: dict[str, str] = {
    "active_thread_state": "working",
    "episode": "episodic",
    "user_fact": "semantic",
    "user_preference": "semantic",
    "personality_core": "identity",
    "response_pattern": "procedural",
}

TIERS: tuple[dict[str, Any], ...] = (
    {"tier": "working", "verdict": "separate",
     "record_types": ("active_thread_state",),
     "storage": "active_thread_state.json — one single record",
     "retention_class": "ephemeral",
     "horizon": "expires_after_turns=10; cooled per completed turn",
     "consumed_by": "the core, every turn (routing, Step 1) + lifecycle cool_thread",
     "evidence": "docstring 'Hot-path current conversation thread state. Aggressive expiry.'"},
    {"tier": "episodic", "verdict": "separate",
     "record_types": ("episode",),
     "storage": "episodic_store.jsonl — episodes alone",
     "retention_class": "ephemeral",
     "horizon": "stale_after_turns=30 (shortest); half-life 7 days",
     "consumed_by": "nothing in the live core — store + lifecycle only (pinned)",
     "evidence": "named by the code itself: EpisodicMemoryRecord / episodic_store.jsonl"},
    {"tier": "semantic", "verdict": "unified-personal / split-world",
     "record_types": ("user_fact", "user_preference"),
     "storage": "user_profile_store.jsonl — facts + preferences interleaved in ONE file",
     "retention_class": "standard",
     "horizon": "stale_after_turns 500 (fact) / 200 (preference); half-lives 180 / 90 days",
     "consumed_by": "nothing in the live core — store + lifecycle only (pinned); "
                    "the world half is consumed via KnowledgeForge (see WORLD_SEMANTIC)",
     "evidence": "one loader, load_profile_records, returns both types from one file"},
)

#: The rows outside the named three tiers, with their rulings.
OUTSIDE_TIERS: tuple[dict[str, Any], ...] = (
    {"tier": "identity", "record_types": ("personality_core",),
     "record": "PersonalityCoreRecord",
     "ruling": "not a memory tier — who is remembering; protected, never "
               "auto-pruned, half-life 9999 days, loaded at core "
               "construction, steers every turn (TF-030)"},
    {"tier": "procedural", "record_types": ("response_pattern",),
     "record": "ResponsePatternRecord",
     "ruling": "a genuine fourth-tier hint (own file, half-life 30 days, "
               "stale 100 turns, pattern-learning scores) — recorded as "
               "procedural, outside the named three; no fourth tier is "
               "built on a hint alone"},
)

#: The transient working surfaces (turn-time only, TF-041's
#: transient census) plus the caller-owned history window.
TRANSIENT_WORKING: tuple[str, ...] = (
    "InputSketch", "RuntimeTurnState", "CognitionScaffold",
    "MemoryActivationBundle",
)
HISTORY_WINDOW: dict[str, Any] = {
    "surface": "ChatHistory (caller-owned, cognition/chat_history.py)",
    "max_turns": 20, "max_tokens": 1000, "max_chars": 2000,
    "owner": "TF-024 budgets, enforced on append",
}

#: The world half of the semantic tier: the knowledge DB's
#: content tables (TF-041 names all 10; these carry knowledge,
#: the rest carry embeddings / the ETL log).
WORLD_SEMANTIC_TABLES: tuple[str, ...] = (
    "entities", "statements", "labels", "conceptnet_edges",
    "geonames_places", "dbpedia_entities", "reference_chunks",
)

#: Surfaces unified across tiers — one implementation each,
#: checked live by tier_problems().
UNIFIED_SURFACES: tuple[dict[str, Any], ...] = (
    {"surface": "store", "what": "one MemoryStore, one load_all (5 sections) over all tiers"},
    {"surface": "personal-semantic file", "what": "facts + preferences share user_profile_store.jsonl and one loader"},
    {"surface": "tag index", "what": "rebuild_tag_index covers profile + episodes + patterns across tiers"},
    {"surface": "recent queue", "what": "one queue of record ids, tier-blind, most-recent-first, cap 64"},
    {"surface": "lifecycle", "what": "one prune pass loads and rewrites profile + episodes + patterns (TF-045)"},
    {"surface": "ingestion", "what": "one TF-042 contract gates every tier's appends"},
    {"surface": "half-life ladder", "what": "one HALF_LIFE_DAYS table; its ordering is the tier ordering"},
)

#: Cross-tier links the code itself carries.
CROSS_TIER_LINKS: tuple[dict[str, Any], ...] = (
    {"from": "episodic", "to": "semantic",
     "fields": ("linked_preferences", "linked_facts"),
     "what": "an episode names the preferences/facts it belongs with"},
    {"from": "working", "to": "all",
     "fields": ("record_id",),
     "what": "the thread state's topic steers routing for every retrieval"},
)


def tier_of_record_type(record_type: str) -> str | None:
    """The tier (or outside-tier) a record_type belongs to;
    None for a type the tiers do not know."""
    return TIER_BY_RECORD_TYPE.get(record_type)


def tier_of_record(record: Any) -> str | None:
    """The tier of one record instance. PersonalityCoreRecord
    carries no record_type field and is read by class name."""
    record_type = getattr(record, "record_type", None)
    if record_type:
        return tier_of_record_type(record_type)
    if type(record).__name__ == "PersonalityCoreRecord":
        return "identity"
    return None


def classify_records(records: Any) -> dict[str, int]:
    """Count records per tier; unknown types count as
    ``unknown`` rather than being silently dropped."""
    counts: dict[str, int] = {}
    for record in records:
        tier = tier_of_record(record) or "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def tier_census(store: Any) -> dict[str, int]:
    """Per-tier record counts of one MemoryStore, via its own
    load_all: working/identity count 0 or 1 (singletons),
    episodic/semantic/procedural count their files' records."""
    all_records = store.load_all()
    census = {
        "working": 1 if all_records["thread_state"] is not None else 0,
        "identity": 1 if all_records["personality_core"] is not None else 0,
        "episodic": len(all_records["episodes"]),
        "procedural": len(all_records["patterns"]),
        "semantic": len(all_records["profile_records"]),
    }
    return census


def tier_problems() -> list[str]:
    """Re-derive TF-046's claims from the live tree; name any
    that stopped being true."""
    problems: list[str] = []
    from thoughtforge.cognition import core as core_module
    from thoughtforge.knowledge import lifecycle as lifecycle_module
    from thoughtforge.knowledge import store as store_module

    # The tier map covers exactly the persisted record types.
    persisted_types = {
        row["record_type"] or "personality_core" for row in PERSISTED_RECORDS
    }
    if set(TIER_BY_RECORD_TYPE) != persisted_types:
        problems.append(
            f"map: tier keys {sorted(TIER_BY_RECORD_TYPE)} != persisted "
            f"record types {sorted(persisted_types)}"
        )
    for row in TIERS:
        for rtype in row["record_types"]:
            if TIER_BY_RECORD_TYPE.get(rtype) != row["tier"]:
                problems.append(f"map: {rtype} is not in tier {row['tier']}")

    # Storage separation / unification, from the TF-041 file map.
    holds = {row["file"]: tuple(row["holds"]) for row in MEMORY_FILES}
    if holds.get("episodic_store.jsonl") != ("EpisodicMemoryRecord",):
        problems.append("storage: episodes no longer alone in their file")
    if holds.get("active_thread_state.json") != ("ActiveThreadStateRecord",):
        problems.append("storage: the thread state no longer has its own file")
    if set(holds.get("user_profile_store.jsonl", ())) != {
        "UserPreferenceRecord", "UserFactRecord",
    }:
        problems.append(
            "storage: personal semantic memory is no longer the unified "
            "fact+preference file — the semantic verdict must be re-ruled"
        )

    # The half-life ladder is the tier ordering (TF-041 K3's order).
    ladder = [
        HALF_LIFE_DAYS[k] for k in (
            "active_thread_state", "episode", "response_pattern",
            "user_preference", "user_fact", "personality_core",
        )
    ]
    if ladder != sorted(ladder):
        problems.append(f"ladder: half-life ordering is no longer the tier ordering: {ladder}")
    if HALF_LIFE_DAYS["active_thread_state"] != 0.07:
        problems.append("ladder: working half-life moved from 0.07 days")

    # Horizons + retention from the live dataclass defaults
    # (TF-041's persisted census, re-read here by tier).
    by_name = {row["name"]: row for row in PERSISTED_RECORDS}
    if by_name["ActiveThreadStateRecord"]["retention_class"] != "ephemeral":
        problems.append("working: thread retention is no longer ephemeral")
    if by_name["EpisodicMemoryRecord"]["stale_after_turns"] != 30:
        problems.append("episodic: stale horizon moved from 30 turns")
    stale_order = [
        by_name[n]["stale_after_turns"] for n in (
            "EpisodicMemoryRecord", "ResponsePatternRecord",
            "UserPreferenceRecord", "UserFactRecord",
        )
    ]
    if stale_order != sorted(stale_order):
        problems.append(f"horizons: stale ordering is no longer tier order: {stale_order}")

    # Cross-tier links exist on the episode type itself.
    episode_fields = {f.name for f in dataclasses.fields(EpisodicMemoryRecord)}
    for field_name in ("linked_preferences", "linked_facts"):
        if field_name not in episode_fields:
            problems.append(f"links: EpisodicMemoryRecord lost {field_name}")

    # Transient working surfaces are all in the transient census.
    for name in TRANSIENT_WORKING:
        if name not in TRANSIENT_RECORDS:
            problems.append(f"working: transient surface {name} left the census")
    from thoughtforge.cognition import history_discipline

    if (
        history_discipline.DEFAULT_MAX_TURNS != HISTORY_WINDOW["max_turns"]
        or history_discipline.DEFAULT_MAX_TOKENS != HISTORY_WINDOW["max_tokens"]
        or history_discipline.DEFAULT_MAX_CHARS != HISTORY_WINDOW["max_chars"]
    ):
        problems.append("working: ChatHistory budgets moved from TF-024's 20/1000/2000")

    # Core consumption pins — the tier verdict's honest half.
    core_src = inspect.getsource(core_module)
    run_src = inspect.getsource(core_module.ThoughtForgeCore._run_turn)
    if "load_thread_state" not in run_src:
        problems.append("consumption: the core no longer reads the working tier per turn")
    if "load_personality_core" not in core_src:
        problems.append("consumption: the core no longer loads identity at construction")
    retrieve_src = inspect.getsource(core_module.ThoughtForgeCore._retrieve)
    if "self._knowledge.retrieve" not in retrieve_src:
        problems.append("consumption: _retrieve no longer reads the knowledge DB")
    for loader in ("load_episodes", "load_patterns", "load_profile_records"):
        if loader in core_src:
            problems.append(
                f"consumption: the core now calls {loader} — personal-tier "
                "retrieval has been wired; TF-046's pinned gap must be "
                "flipped with evidence by the wiring slice"
            )

    # Unified surfaces stand.
    store_src = inspect.getsource(store_module.MemoryStore.rebuild_tag_index)
    for loader in ("load_profile_records", "load_episodes", "load_patterns"):
        if loader not in store_src:
            problems.append(f"unified: rebuild_tag_index no longer covers {loader}")
    prune_src = inspect.getsource(lifecycle_module.MemoryLifecycle)
    for loader in ("load_profile_records", "load_episodes", "load_patterns"):
        if loader not in prune_src:
            problems.append(f"unified: the lifecycle prune no longer covers {loader}")

    # The world semantic half is inside the mapped knowledge DB.
    from thoughtforge.knowledge.store_map import KNOWLEDGE_DB_TABLES

    db_names = {row["name"] for row in KNOWLEDGE_DB_TABLES}
    for name in WORLD_SEMANTIC_TABLES:
        if name not in db_names:
            problems.append(f"world semantic: table {name} left the knowledge DB map")
    return problems


__all__ = [
    "CROSS_TIER_LINKS",
    "HISTORY_WINDOW",
    "OUTSIDE_TIERS",
    "TIERS",
    "TIER_BY_RECORD_TYPE",
    "TIER_NAMES",
    "TRANSIENT_WORKING",
    "UNIFIED_SURFACES",
    "WORLD_SEMANTIC_TABLES",
    "classify_records",
    "tier_census",
    "tier_of_record",
    "tier_of_record_type",
    "tier_problems",
]
