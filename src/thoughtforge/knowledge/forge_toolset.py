"""
Knowledge forge UI-shape — the forge_memory toolset (TF-057).

The roadmap's rule: *forge_memory's operations
(add/inspect/prune/export) completed to a coherent
toolset.*

The measured starting state, before this slice (probed
this tick, keeper 08:12):

  * ``forge_memory.py`` carried nine commands — init,
    wikidata, conceptnet, geonames, dbpedia, reference,
    embeddings, status, all — and every one of them acts
    on the knowledge DATABASE (the ETL estate). Not one
    command, and no library module, operated on the
    *memory store* (``MemoryStore``): the store a user
    actually lives in had no operator toolset at all.
  * The store's operations existed only scattered, one
    per earlier slice, with no common shape: adding was
    the store's raw ``append_*`` gates (TF-042), pruning
    was ``MemoryLifecycle.prune`` (TF-045) or the core's
    ``prune_memory``, exporting was TF-052's bundle, and
    inspecting a memory store did not exist at all (the
    CLI's ``status`` reports the knowledge DB's tables).
  * Two earlier slices named this one as the owner of
    the missing legs: TF-042 assigned manual-entry
    provenance stamping to TF-057 ("TF-057 stamps manual
    entries in the forge_memory toolset" —
    ``MEMORY_PROVENANCE`` already carries
    ``manual_entry``, unused), and TF-045 named "a
    forge_memory.py prune command" as TF-057's.

What this slice builds:

  * ``OPERATIONS`` — the toolset as data: exactly four
    operations (add / inspect / prune / export), each
    with its CLI command name in ``forge_memory.py``,
    what it acts on, and whether it writes the store.
    The knowledge-DB commands are untouched and are not
    part of this table — the two estates keep their own
    tools, stated, not merged.
  * ``add_record`` — the manual add. It builds one
    record of a kind in ``ADD_KINDS`` (user_preference /
    user_fact / episode / response_pattern — TF-042's
    four stamped memory kinds), stamps it
    ``source="manual_entry"`` *by construction* (there
    is no source parameter to override it — TF-042's
    assignment, landed), lands it through the store's
    own TF-042 ingestion gate (an invalid record is
    refused by the gate's ValueError naming every
    problem, and nothing is written), rebuilds the tag
    index, and pushes the id onto the recent queue — a
    manual add is immediately findable the same ways a
    writeback record is. An explicit ``record_id`` that
    already lives anywhere in the store (any append
    store, the thread state, or the personality core)
    is refused by name before anything is written:
    adding never silently duplicates or overwrites.
    Kind defaults mirror the record dataclasses (and
    TF-047's filled defaults) exactly.
  * ``inspect_store`` / ``inspect_records`` /
    ``find_record`` — the inspect leg, read-only.
    ``inspect_store`` returns one JSON-safe report:
    live counts per kind (TF-056's ``measure_store``),
    bytes per file, counts by status, the personality
    core's and thread state's presence, the recent
    queue's length, the tag index's size, and — when a
    hardware profile is named — TF-056's budget report
    for the same store. ``inspect_records`` lists
    record summaries filtered by kind / tag / status /
    text; ``find_record`` resolves one id across every
    surface inspect can see. Inspect writes nothing:
    pinned by a byte-for-byte directory comparison.
  * ``prune_store`` — the prune leg TF-045 named. One
    ``MemoryLifecycle`` pass in a named mode (with an
    optional TF-056 profile for its caps), reported as
    before-counts / the lifecycle's own counts /
    after-counts. An unknown mode is refused by the
    lifecycle's up-front ValueError (TF-045) before any
    store is rewritten. Prune is the lifecycle's act —
    score-driven retirement by rule, protected records
    kept — *not* user-ordered forgetting (TF-051 owns
    that, unchanged) and not budget eviction as such
    (TF-056 owns that; a profile here only swaps the
    caps the same prune enforces, as TF-056 built it).
  * ``export_store_bundle`` — the export leg: TF-052's
    ``write_bundle`` to a named path, re-verified with
    ``verify_bundle`` after the write; the report names
    the path, format, version, per-section counts, and
    whether verification passed. A bundle that failed
    verification is reported ``verified: False`` with
    the problems named, never as a clean export.
  * **CLI wiring.** ``forge_memory.py`` gains a
    ``--memory-dir`` group option (default: the
    standing ``get_memory_dir()``) and four commands —
    ``add``, ``inspect``, ``prune``, ``export`` — that
    print the toolset's JSON reports. A toolset refusal
    (ValueError) surfaces as a click error naming the
    problem, exit 1, with nothing written. The nine
    knowledge-DB commands keep their exact behaviour;
    ``--db`` still governs only them.

Stated boundaries:

  * The toolset operates on ONE memory store (one
    directory = one persona, TF-055). It has no
    knowledge-DB operation and no cross-store operation:
    moving a whole store is TF-052 import's operator
    act, not a toolset verb.
  * Personality core and thread state are inspect-only
    here: identity is versioned manually (TF-042) and
    the working record is the conversation's, so `add`
    covers exactly the four append kinds.
  * Proof grade is FIXTURE (synthetic stores + CLI
    runner on this VM); the toolset over a real,
    lived-in store is NOT PROVEN (no real store exists
    on this machine, TF-041), never claimed.

The human-readable write-up is
``docs/architecture/FORGE_TOOLSET.md``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.ingestion_contract import MEMORY_PROVENANCE
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore

#: The provenance stamp every manual add carries (TF-042).
MANUAL_SOURCE = "manual_entry"

# ── The toolset, as data ─────────────────────────────────────────────────────

#: The four operations, in toolset order. ``writes_store``
#: is the honesty column: inspect never writes, export
#: writes a file *outside* the store and leaves the store
#: byte-identical, add and prune change the store.
OPERATIONS: tuple[dict[str, Any], ...] = (
    {"name": "add", "cli": "add", "writes_store": True,
     "what": "add one memory record by hand, stamped "
             "source='manual_entry' (TF-042), through the "
             "store's ingestion gate"},
    {"name": "inspect", "cli": "inspect", "writes_store": False,
     "what": "report one store (counts, bytes, statuses, "
             "budget) or list/find its records — read-only"},
    {"name": "prune", "cli": "prune", "writes_store": True,
     "what": "one MemoryLifecycle prune pass in a named "
             "mode (TF-045), optionally under a profile's "
             "caps (TF-056), reported before/after"},
    {"name": "export", "cli": "export", "writes_store": False,
     "what": "write the whole store as one verified "
             "TF-052 bundle to a named path"},
)

#: The kinds `add` can land — TF-042's four stamped memory
#: kinds. ``home`` names the store leg that appends them.
ADD_KINDS: tuple[dict[str, Any], ...] = (
    {"kind": "user_preference", "home": "profile",
     "record": "UserPreferenceRecord",
     "id_factory": "new_preference_id"},
    {"kind": "user_fact", "home": "profile",
     "record": "UserFactRecord",
     "id_factory": "new_fact_id"},
    {"kind": "episode", "home": "episodes",
     "record": "EpisodicMemoryRecord",
     "id_factory": "new_episode_id"},
    {"kind": "response_pattern", "home": "patterns",
     "record": "ResponsePatternRecord",
     "id_factory": "new_pattern_id"},
)

ADD_KIND_NAMES: tuple[str, ...] = tuple(r["kind"] for r in ADD_KINDS)


# ── Record lookup (inspect's eyes) ───────────────────────────────────────────


def _all_records(store: MemoryStore) -> list[tuple[str, Any]]:
    """Every list-home record as (kind, record), in file
    order: profile (preferences + facts), episodes,
    patterns. Kind is the record's own record_type."""
    out: list[tuple[str, Any]] = []
    for r in store.load_profile_records():
        out.append((r.record_type, r))
    for r in store.load_episodes():
        out.append((r.record_type, r))
    for r in store.load_patterns():
        out.append((r.record_type, r))
    return out


def find_record(store: MemoryStore, record_id: str) -> dict[str, Any]:
    """Resolve one id across every surface inspect can
    see: the append stores, the thread state, and the
    personality core. JSON-safe either way — a miss is
    data (``found: False``), not an exception."""
    for kind, record in _all_records(store):
        if record.record_id == record_id:
            home = {"user_preference": "profile", "user_fact": "profile",
                    "episode": "episodes",
                    "response_pattern": "patterns"}[kind]
            return {"found": True, "record_id": record_id, "kind": kind,
                    "home": home, "record": record.to_dict()}
    thread = store.load_thread_state()
    if thread is not None and thread.record_id == record_id:
        return {"found": True, "record_id": record_id,
                "kind": "active_thread_state", "home": "thread_state",
                "record": thread.to_dict()}
    core = store.load_personality_core()
    if core is not None and core.record_id == record_id:
        return {"found": True, "record_id": record_id,
                "kind": "personality_core", "home": "personality_core",
                "record": core.to_dict()}
    return {"found": False, "record_id": record_id}


# ── Add ──────────────────────────────────────────────────────────────────────


def add_record(
    store: MemoryStore,
    kind: str,
    *,
    summary: str = "",
    tags: list[str] | None = None,
    category: str = "",
    value: str = "",
    fact_value: dict[str, Any] | None = None,
    pattern_shape: list[str] | None = None,
    tone: str = "",
    importance: float | None = None,
    confidence: float | None = None,
    quality: float | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Add one record by hand, stamped ``manual_entry``.

    Refusals are ValueError, raised *before* anything is
    written: an unknown kind (naming the valid kinds), an
    explicit id that already lives in the store (naming
    it), or the store gate's own contract verdict on an
    invalid record (TF-042 names every problem). On
    success the record is landed, the tag index is
    rebuilt, the id is pushed to the recent queue, and
    the JSON-safe report names what landed."""
    if kind not in ADD_KIND_NAMES:
        raise ValueError(
            f"unknown add kind {kind!r}: valid kinds are "
            f"{list(ADD_KIND_NAMES)}"
        )
    row = next(r for r in ADD_KINDS if r["kind"] == kind)
    rid = record_id or getattr(MemoryStore, row["id_factory"])()
    if find_record(store, rid)["found"]:
        raise ValueError(
            f"record id {rid!r} already exists in this store — "
            f"add never duplicates or overwrites; choose another id"
        )
    tags = list(tags or [])
    if kind == "user_preference":
        record: Any = UserPreferenceRecord(
            record_id=rid, category=category, tags=tags,
            summary=summary, value=value,
            importance=0.5 if importance is None else importance,
            confidence=0.5 if confidence is None else confidence,
            source=MANUAL_SOURCE,
        )
        store.append_profile_record(record)
    elif kind == "user_fact":
        record = UserFactRecord(
            record_id=rid, tags=tags, summary=summary,
            fact_value=dict(fact_value or {}),
            importance=0.5 if importance is None else importance,
            confidence=0.6 if confidence is None else confidence,
            source=MANUAL_SOURCE,
        )
        store.append_profile_record(record)
    elif kind == "episode":
        record = EpisodicMemoryRecord(
            record_id=rid, tags=tags, summary=summary, tone=tone,
            importance=0.4 if importance is None else importance,
            confidence=0.5 if confidence is None else confidence,
            quality=0.5 if quality is None else quality,
            source=MANUAL_SOURCE,
        )
        store.append_episode(record)
    else:  # response_pattern
        record = ResponsePatternRecord(
            record_id=rid, tags=tags, summary=summary,
            pattern_shape=list(pattern_shape or []),
            quality=0.6 if quality is None else quality,
            importance=0.5 if importance is None else importance,
            confidence=0.6 if confidence is None else confidence,
            source=MANUAL_SOURCE,
        )
        store.append_pattern(record)
    store.rebuild_tag_index()
    store.push_to_recent_queue(rid)
    return {"status": "added", "kind": kind, "record_id": rid,
            "source": MANUAL_SOURCE, "record": record.to_dict()}


# ── Inspect ──────────────────────────────────────────────────────────────────


def inspect_store(
    store: MemoryStore, *, profile_id: str | None = None
) -> dict[str, Any]:
    """One JSON-safe report on a whole store. Read-only.

    Counts and bytes are TF-056's ``measure_store`` (live
    counts per kind; bytes per *file* — per-kind bytes are
    not claimed, facts and preferences share one file).
    The budget section appears only when a profile is
    named, and is TF-056's own report for the same store."""
    from thoughtforge.knowledge.memory_budgets import (
        budget_report,
        measure_store,
    )

    size = measure_store(store)
    by_status: dict[str, int] = {}
    for _, record in _all_records(store):
        by_status[record.status] = by_status.get(record.status, 0) + 1
    core = store.load_personality_core()
    thread = store.load_thread_state()
    report: dict[str, Any] = {
        "store_dir": str(store.dir),
        "counts": dict(size.counts),
        "total_records": size.total_records,
        "file_bytes": dict(size.file_bytes),
        "total_bytes": size.total_bytes,
        "counts_by_status": by_status,
        "personality_core": (
            {"present": True, "record_id": core.record_id}
            if core is not None else {"present": False}
        ),
        "thread_state": (
            {"present": True, "record_id": thread.record_id,
             "topic": thread.topic}
            if thread is not None else {"present": False}
        ),
        "recent_queue_length": len(store.load_recent_queue()),
        "tag_count": len(store.load_tag_index()),
    }
    if profile_id is not None:
        report["budget"] = budget_report(store, profile_id).to_dict()
    return report


def inspect_records(
    store: MemoryStore,
    *,
    kind: str | None = None,
    tag: str | None = None,
    status: str | None = None,
    text: str | None = None,
) -> list[dict[str, Any]]:
    """Record summaries (never full records), filtered by
    any of kind / tag / status / a case-insensitive text
    needle over the summary. An unknown kind filter is a
    ValueError naming the kinds inspect can list — a
    typo'd filter must not read as an empty store."""
    listable = ADD_KIND_NAMES + ("active_thread_state", "personality_core")
    if kind is not None and kind not in listable:
        raise ValueError(
            f"unknown inspect kind {kind!r}: valid kinds are "
            f"{list(listable)}"
        )
    rows: list[tuple[str, Any]] = list(_all_records(store))
    thread = store.load_thread_state()
    if thread is not None:
        rows.append(("active_thread_state", thread))
    core = store.load_personality_core()
    if core is not None:
        rows.append(("personality_core", core))
    out: list[dict[str, Any]] = []
    for rkind, record in rows:
        if kind is not None and rkind != kind:
            continue
        if tag is not None and tag not in getattr(record, "tags", []):
            continue
        if status is not None and record.status != status:
            continue
        summary = getattr(record, "summary", "")
        if text is not None and text.lower() not in summary.lower():
            continue
        out.append({
            "record_id": record.record_id, "kind": rkind,
            "summary": summary, "tags": list(getattr(record, "tags", [])),
            "status": record.status,
            "source": getattr(record, "source", ""),
        })
    return out


# ── Prune ────────────────────────────────────────────────────────────────────


def prune_store(
    store: MemoryStore,
    *,
    mode: str = "routine",
    profile_id: str | None = None,
) -> dict[str, Any]:
    """One lifecycle prune pass, reported before/after.

    The lifecycle refuses an unknown mode up front
    (TF-045), before any rewrite — that ValueError is
    this leg's refusal too. ``profile_id`` swaps in
    TF-056's per-profile caps for the same pass."""
    from thoughtforge.knowledge.lifecycle import MemoryLifecycle
    from thoughtforge.knowledge.memory_budgets import measure_store

    before = measure_store(store).counts
    lifecycle = MemoryLifecycle(store=store, profile_id=profile_id)
    counts = lifecycle.prune(mode)  # type: ignore[arg-type]
    after = measure_store(store).counts
    return {"status": "pruned", "mode": mode, "profile_id": profile_id,
            "counts": dict(counts), "before": dict(before),
            "after": dict(after)}


# ── Export ───────────────────────────────────────────────────────────────────


def export_store_bundle(
    store: MemoryStore, destination: str | Path
) -> dict[str, Any]:
    """Write the whole store as one TF-052 bundle and
    re-verify what was written. The store itself is only
    read. A verification failure is reported, not hidden:
    ``verified`` False with every problem named."""
    from thoughtforge.knowledge.portability import (
        verify_bundle,
        write_bundle,
    )

    path = Path(destination)
    bundle = write_bundle(store, path)
    problems = verify_bundle(bundle)
    return {"status": "exported", "path": str(path),
            "format": bundle["format"], "version": bundle["version"],
            "counts": dict(bundle["manifest"]["counts"]),
            "total_records": bundle["manifest"]["total_records"],
            "verified": problems == [], "problems": problems}


# ── The live checker ─────────────────────────────────────────────────────────


def forge_toolset_problems() -> list[str]:
    """Re-derive TF-057's claims from the live tree; [] = holds.

    Pins: the operations table is exactly add/inspect/
    prune/export with CLI names matching forge_memory's
    real commands (and its nine DB commands intact); the
    add kinds are exactly TF-042's stamped memory kinds;
    the manual stamp is in MEMORY_PROVENANCE; a fixture
    store proves the round trip — add stamps + lands +
    indexes, a duplicate id and an invalid record are
    refused with nothing written, inspect is byte-clean
    read-only, an unknown prune mode is refused, and an
    exported bundle verifies."""
    problems: list[str] = []

    if [r["name"] for r in OPERATIONS] != ["add", "inspect", "prune", "export"]:
        problems.append("operations table drifted from add/inspect/prune/export")
    if MANUAL_SOURCE not in MEMORY_PROVENANCE:
        problems.append(f"{MANUAL_SOURCE!r} not in MEMORY_PROVENANCE")
    if MANUAL_SOURCE != "manual_entry":
        problems.append("MANUAL_SOURCE moved from 'manual_entry'")

    from thoughtforge.knowledge.ingestion_contract import FACT_KINDS

    stamped = {
        r["record_type"] for r in FACT_KINDS
        if r["store"] == "memory" and r["provenance_field"] == "source"
    }
    if set(ADD_KIND_NAMES) != stamped:
        problems.append(
            f"add kinds {sorted(ADD_KIND_NAMES)} != TF-042 stamped "
            f"kinds {sorted(stamped)}"
        )

    # The CLI wiring stands, and the DB estate is intact.
    forge_memory_path = Path(__file__).parents[3] / "forge_memory.py"
    if not forge_memory_path.exists():
        problems.append("wiring: forge_memory.py not found at the repo root")
    else:
        text = forge_memory_path.read_text(encoding="utf-8")
        for cli_name in ("add", "inspect", "prune", "export"):
            if f'"{cli_name}"' not in text and f"'{cli_name}'" not in text \
                    and f"def {cli_name}" not in text:
                problems.append(f"wiring: forge_memory lacks the {cli_name} command")
        for db_name in ("wikidata", "conceptnet", "geonames", "dbpedia",
                        "reference", "embeddings", "status"):
            if db_name not in text:
                problems.append(f"wiring: forge_memory lost the {db_name} DB command")

    # Fixture round trip.
    with tempfile.TemporaryDirectory(prefix="tf057-") as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        report = add_record(store, "user_fact", summary="ferry at 07:30",
                            tags=["ferry"], record_id="fct_tf057")
        if report["source"] != MANUAL_SOURCE:
            problems.append("fixture: add did not stamp manual_entry")
        if not find_record(store, "fct_tf057")["found"]:
            problems.append("fixture: added record not findable")
        if store.load_tag_index().get("ferry") != ["fct_tf057"]:
            problems.append("fixture: tag index not rebuilt on add")
        if "fct_tf057" not in store.load_recent_queue():
            problems.append("fixture: added id not on the recent queue")
        try:
            add_record(store, "user_fact", summary="again",
                       record_id="fct_tf057")
            problems.append("fixture: duplicate id was not refused")
        except ValueError:
            pass
        try:
            add_record(store, "user_fact", summary="")
            problems.append("fixture: empty fact was not refused by the gate")
        except ValueError:
            pass
        if len(store.load_profile_records()) != 1:
            problems.append("fixture: a refused add wrote a record")
        before = {p.name: p.read_bytes() for p in store.dir.iterdir()}
        inspect_store(store)
        inspect_records(store)
        after = {p.name: p.read_bytes() for p in store.dir.iterdir()}
        if before != after:
            problems.append("fixture: inspect wrote to the store")
        try:
            prune_store(store, mode="bogus_mode")
            problems.append("fixture: unknown prune mode was not refused")
        except ValueError:
            pass
        out = Path(tmp) / "bundle.json"
        exported = export_store_bundle(store, out)
        if not exported["verified"] or not out.exists():
            problems.append(f"fixture: export not verified {exported}")
    return problems
