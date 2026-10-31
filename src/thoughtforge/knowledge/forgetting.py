"""
Forgetting — user-ordered deletion, total and provable (TF-051).

The roadmap's rule: *user-ordered deletion is total and
provable (store, indices, caches) — the right to be
forgotten, locally.*

The measured starting state, before this slice (probed on a
synthetic store, keeper tick 07:02):

  * No forgetting API existed anywhere: no function took a
    user's order to delete a record and carried it out. The
    two deletion-shaped mechanisms in the tree were the
    lifecycle's rule-driven prune (TF-045 — the *system*
    retiring records by score, not the user ordering
    anything) and TF-047's writeback ``records_to_delete``
    leg (a conversation's request, applied as mark + rewrite).
  * Neither was total. The probe: seed a fact, an episode
    linking to it (``linked_facts``), a rebuilt tag index,
    and the fact's id in the recent queue; delete the fact
    through the writeback leg. The fact left its file — and
    its id stayed in ``tag_index.json`` (``ferry ->
    [fct_target, ep_link]``), stayed in
    ``recent_queue.json``, and stayed dangling in the
    episode's ``linked_facts``. A reader of the indices or
    the episode would still meet the "forgotten" record's
    identity after its deletion.
  * Nothing could *prove* a deletion: no function scanned
    the store, the indices, and the caches for a deleted
    id's residue. "Deleted" meant "the rewrite dropped it
    from one file", and every other surface was taken on
    faith.

What this slice builds:

  * ``forget_record(store, record_id)`` — the user-ordered
    leg. It resolves the id across the forgettable surfaces
    in ``FORGET_SURFACES`` (profile store, episodic store,
    patterns store, and — for the direct API — the active
    thread state), removes the record through the store's
    own rewrite shape (mark ``deleted`` + rewrite, exactly
    the removal TF-045/TF-047 use), then cleans every
    derived surface: the tag index is rebuilt from live
    records only, the recent queue is purged of the id,
    and episode ``linked_preferences`` / ``linked_facts``
    references to the id are scrubbed (the linking episode
    itself survives — a moment that referenced a fact is
    not the fact). Residues are cleaned even when the
    record itself is already gone: a stale id in the queue
    or index is still the user's data to forget, so the
    operation is idempotent and total, never a silent
    no-op — every call returns a ``ForgetReport`` saying
    what was found, what was removed, and what the proof
    found afterwards.
  * **A user's order outranks retention.** Lifecycle
    protections (``retention_class == "protected"``) guard
    a record against the *system's* prune deciding to
    retire it; they do not guard it against the user who
    owns it ordering it forgotten. ``forget_record``
    deletes a protected profile/episode/pattern record
    exactly as any other. The one refusal is the
    personality core: it is the persona's identity record,
    not user memory (TF-046's ruling — who is remembering,
    not a memory), it carries no user fact, and it is
    removed only by manual versioning. An order naming its
    id is refused by name (``refused_protected``) and the
    core is left byte-intact.
  * ``verify_forgotten(store, record_id)`` — the proof.
    A ``ForgetProof`` whose checks (``PROOF_CHECKS``) each
    re-read a surface fresh: the id is absent from every
    store loader, from the thread state, from the
    personality core, from the tag index, and from the
    recent queue; no episode links to it; and a raw scan
    of every file in the memory directory finds the id
    string nowhere. ``proof.forgotten`` is the AND of all
    checks — the report claims success only when the
    proof passes; a failed proof surfaces as status
    ``not_proven`` with the failed checks named, never as
    a quiet success. Proof is by identity (the record id):
    content another live record carries in its own right
    (a second record with the same summary) is that
    record's content, not a residue of the forgotten one.
  * **Wiring.** TF-047's writeback ``records_to_delete``
    leg now routes through ``forget_record`` (with
    ``include_thread=False`` — TF-047's ruling that
    writeback cannot touch the working record stands), so
    a conversation-ordered deletion is total the same way:
    applied only when the report says forgotten, refused
    with the report's reason otherwise (unknown id keeps
    TF-047's "not found" refusal; the personality core
    keeps its refusal, now with the reason named).
    Reinforce and archive legs are unchanged — archive is
    retirement, not forgetting, and keeps its own shape.

Stated boundaries:

  * The knowledge DB (Wikidata / ConceptNet / reference
    chunks in SQLite) is NOT a forgettable surface: it is
    a read-only reference estate, not the user's memory,
    and it holds no record ids from the memory store.
  * The lifecycle prune is not user-ordered forgetting
    and is unchanged: its tag-index rebuild cadence
    (rebuilt in routine/heavy/emergency, not light) is
    TF-045's pinned behaviour, and prune retirement
    (archive) deliberately keeps records reachable by
    rule. Only the deletion legs ordered by a user or a
    conversation run through here.
  * In-memory turn state (activation bundles, scaffolds)
    is transient per-turn data (TF-046), not a store,
    index, or cache on disk; a record forgotten between
    turns cannot be retrieved by a later turn because
    retrieval re-reads the store. A bundle already built
    inside a running turn is that turn's working copy —
    forgetting lands at the store, and no claim is made
    about a turn already in flight.
  * Proof grade is FIXTURE (synthetic stores on this VM);
    forgetting over a real, lived-in store is NOT PROVEN
    (no real store exists on this machine, TF-041).

The human-readable write-up is
``docs/architecture/FORGETTING.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.common import now_iso as _now_iso

# ── The forgettable surfaces, as data ────────────────────────────────────────

#: Every surface a user-ordered deletion must clean, in the
#: order they are searched. ``file`` names the on-disk file
#: MemoryStore owns for the surface.
FORGET_SURFACES: tuple[dict[str, Any], ...] = (
    {"name": "profile_store", "file": "user_profile_store.jsonl",
     "what": "preferences + facts (the record's home file)"},
    {"name": "episodic_store", "file": "episodic_store.jsonl",
     "what": "episodes (a home file, and the link-scrub surface)"},
    {"name": "patterns_store", "file": "response_patterns.jsonl",
     "what": "response patterns (a home file)"},
    {"name": "thread_state", "file": "active_thread_state.json",
     "what": "the standing working record (direct API only)"},
    {"name": "tag_index", "file": "tag_index.json",
     "what": "derived index — rebuilt from live records only"},
    {"name": "recent_queue", "file": "recent_queue.json",
     "what": "recency cache — the id is purged"},
    {"name": "episode_links", "file": "episodic_store.jsonl",
     "what": "episodes' linked_preferences / linked_facts — scrubbed"},
)

#: Surfaces deliberately NOT forgettable through this leg,
#: with the reason each is out of reach.
EXCLUDED_SURFACES: tuple[dict[str, Any], ...] = (
    {"name": "personality_core", "file": "personality_core.yaml",
     "reason": "identity, not user memory (TF-046) — removed only "
               "by manual versioning; an order naming it is refused"},
    {"name": "knowledge_db", "file": "knowledge.db",
     "reason": "read-only reference estate (Wikidata/ConceptNet/"
               "reference chunks); holds no memory-store record ids"},
)

#: The checks verify_forgotten runs, in order. proof.forgotten
#: is the AND of exactly these.
PROOF_CHECKS: tuple[str, ...] = (
    "absent_from_profile_store",
    "absent_from_episodic_store",
    "absent_from_patterns_store",
    "absent_from_thread_state",
    "absent_from_personality_core",
    "absent_from_tag_index",
    "absent_from_recent_queue",
    "no_dangling_episode_links",
    "raw_files_free_of_id",
)

#: The statuses a ForgetReport can carry.
FORGET_STATUSES: tuple[str, ...] = (
    "forgotten", "not_found", "refused_protected", "not_proven",
)


# ── Reports ──────────────────────────────────────────────────────────────────


@dataclass
class ForgetProof:
    """The post-deletion scan, as data. ``forgotten`` is
    True only when every check in PROOF_CHECKS passed —
    each check re-read its surface fresh from the store,
    and the raw scan read the files' bytes."""

    record_id: str
    checks: dict[str, bool] = field(default_factory=dict)
    files_scanned: list[str] = field(default_factory=list)

    @property
    def forgotten(self) -> bool:
        return bool(self.checks) and all(
            self.checks.get(name, False) for name in PROOF_CHECKS
        )

    @property
    def failed_checks(self) -> list[str]:
        return [name for name in PROOF_CHECKS if not self.checks.get(name, False)]

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["forgotten"] = self.forgotten
        data["failed_checks"] = self.failed_checks
        return data


@dataclass
class ForgetReport:
    """What one forget_record call did, as data. Nothing is
    silent: a found record, a residue-only cleanup, a
    not-found order, and a protected refusal each carry
    their status and reason."""

    record_id: str
    status: str = "not_found"
    home: str | None = None
    removed_from: list[str] = field(default_factory=list)
    links_scrubbed: int = 0
    reason: str = ""
    proof: ForgetProof | None = None

    @property
    def forgotten(self) -> bool:
        return self.status == "forgotten"

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if self.proof is not None:
            data["proof"] = self.proof.to_dict()
        data["forgotten"] = self.forgotten
        return data


# ── The proof ────────────────────────────────────────────────────────────────


def verify_forgotten(store: MemoryStore, record_id: str) -> ForgetProof:
    """Scan every surface for ``record_id`` and return the
    proof. Pure read: this function removes nothing."""
    checks: dict[str, bool] = {}
    checks["absent_from_profile_store"] = all(
        r.record_id != record_id for r in store.load_profile_records()
    )
    episodes = store.load_episodes()
    checks["absent_from_episodic_store"] = all(
        r.record_id != record_id for r in episodes
    )
    checks["absent_from_patterns_store"] = all(
        r.record_id != record_id for r in store.load_patterns()
    )
    thread = store.load_thread_state()
    checks["absent_from_thread_state"] = (
        thread is None or thread.record_id != record_id
    )
    core = store.load_personality_core()
    checks["absent_from_personality_core"] = (
        core is None or core.record_id != record_id
    )
    checks["absent_from_tag_index"] = all(
        record_id not in ids for ids in store.load_tag_index().values()
    )
    checks["absent_from_recent_queue"] = (
        record_id not in store.load_recent_queue()
    )
    checks["no_dangling_episode_links"] = all(
        record_id not in ep.linked_preferences
        and record_id not in ep.linked_facts
        for ep in episodes
    )
    # Raw scan: the id string must appear in no file in the
    # memory directory — the loaders could all agree and a
    # derived file nobody remembered would still betray it.
    needle = record_id.encode("utf-8")
    files_scanned: list[str] = []
    raw_clean = True
    if store.dir.is_dir():
        for path in sorted(store.dir.iterdir()):
            if not path.is_file():
                continue
            files_scanned.append(path.name)
            try:
                if needle in path.read_bytes():
                    raw_clean = False
            except OSError:
                raw_clean = False
    checks["raw_files_free_of_id"] = raw_clean
    return ForgetProof(
        record_id=record_id, checks=checks, files_scanned=files_scanned
    )


# ── The leg ──────────────────────────────────────────────────────────────────


def _rewrite_home(store: MemoryStore, home: str, records: list[Any]) -> None:
    if home == "profile_store":
        store.rewrite_profile_store(records)
    elif home == "episodic_store":
        store.rewrite_episodic_store(records)
    else:
        store.rewrite_patterns_store(records)


def forget_record(
    store: MemoryStore,
    record_id: str,
    *,
    include_thread: bool = True,
) -> ForgetReport:
    """
    Carry out one user-ordered deletion, totally, and prove
    it. ``include_thread=False`` keeps the standing thread
    record out of reach (TF-047's writeback ruling); the
    thread's id then resolves as not found unless it also
    lives in an append store.

    Never raises on a missing or protected id — those are
    outcomes (``not_found`` / ``refused_protected``) carried
    on the report, with the proof still run and attached.
    """
    report = ForgetReport(record_id=record_id)

    # The one refusal: identity is not user memory.
    core = store.load_personality_core()
    if core is not None and core.record_id == record_id:
        report.status = "refused_protected"
        report.reason = (
            "the personality core is identity, not user memory — "
            "it is removed only by manual versioning, never by a "
            "forgetting order"
        )
        report.proof = verify_forgotten(store, record_id)
        return report

    # 1. The record's home: mark deleted + rewrite drops it
    #    from its file (the store's own deletion shape).
    homes = (
        ("profile_store", store.load_profile_records()),
        ("episodic_store", store.load_episodes()),
        ("patterns_store", store.load_patterns()),
    )
    for home, records in homes:
        for record in records:
            if record.record_id == record_id:
                record.status = "deleted"
                record.updated_at = _now_iso()
                _rewrite_home(store, home, records)
                report.home = home
                report.removed_from.append(home)
                break
        if report.home is not None:
            break
    if report.home is None and include_thread:
        thread = store.load_thread_state()
        if thread is not None and thread.record_id == record_id:
            store.clear_thread_state()
            report.home = "thread_state"
            report.removed_from.append("thread_state")

    # 2. Episode links: a forgotten id must not dangle from
    #    a surviving episode's link lists.
    episodes = store.load_episodes()
    scrubbed = 0
    changed = False
    for ep in episodes:
        for attr in ("linked_preferences", "linked_facts"):
            links = getattr(ep, attr)
            if record_id in links:
                scrubbed += links.count(record_id)
                setattr(ep, attr, [x for x in links if x != record_id])
                changed = True
    if changed:
        store.rewrite_episodic_store(episodes)
        report.links_scrubbed = scrubbed
        report.removed_from.append("episode_links")

    # 3. The recency cache: purge the id.
    queue = store.load_recent_queue()
    if record_id in queue:
        store.save_recent_queue([x for x in queue if x != record_id])
        report.removed_from.append("recent_queue")

    # 4. The derived index: rebuild from live records only —
    #    a forgotten id has no live record to be indexed by.
    had_index_entry = any(
        record_id in ids for ids in store.load_tag_index().values()
    )
    store.rebuild_tag_index()
    if had_index_entry:
        report.removed_from.append("tag_index")

    # 5. Prove it, and let the proof decide the claim.
    report.proof = verify_forgotten(store, record_id)
    if report.home is None and not (
        report.links_scrubbed or "recent_queue" in report.removed_from
        or "tag_index" in report.removed_from
    ):
        report.status = "not_found"
        report.reason = f"record {record_id} not found in any forgettable surface"
    elif not report.proof.forgotten:
        report.status = "not_proven"
        report.reason = (
            "deletion performed but the proof failed checks: "
            + ", ".join(report.proof.failed_checks)
        )
    else:
        report.status = "forgotten"
        report.reason = (
            f"record {record_id} removed from "
            f"{', '.join(report.removed_from) or 'no surface'}; "
            "proof: absent from every store, index, cache, and file"
        )
    return report


# ── Self-check ───────────────────────────────────────────────────────────────


def forgetting_problems() -> list[str]:
    """
    Re-check this slice's claims against the live tree:
    every named surface is a file the store owns, the proof
    covers exactly PROOF_CHECKS, and TF-047's delete leg is
    wired through here. Returns problems by name; ``[]``
    when forgetting still matches the code.
    """
    from thoughtforge.knowledge import store as store_module
    from thoughtforge.knowledge import writeback as writeback_module

    problems: list[str] = []

    store_src = inspect.getsource(store_module)
    for row in FORGET_SURFACES + EXCLUDED_SURFACES:
        if row["file"] == "knowledge.db":
            continue  # not a MemoryStore file — the ETL estate's
        if row["file"] not in store_src:
            problems.append(
                f"surface {row['name']}: file {row['file']} is not a "
                "MemoryStore file in store.py"
            )
    surface_names = [row["name"] for row in FORGET_SURFACES]
    if len(set(surface_names)) != len(surface_names):
        problems.append("FORGET_SURFACES carries a duplicate surface name")
    if set(FORGET_STATUSES) != {
        "forgotten", "not_found", "refused_protected", "not_proven",
    }:
        problems.append("FORGET_STATUSES drifted from the stated statuses")

    proof_src = inspect.getsource(verify_forgotten)
    for name in PROOF_CHECKS:
        if name not in proof_src:
            problems.append(f"verify_forgotten no longer runs check {name}")

    mutate_src = inspect.getsource(writeback_module._mutate_records)
    if "forget_record" not in mutate_src:
        problems.append(
            "writeback wiring: _mutate_records no longer routes its "
            "delete leg through forget_record"
        )
    return problems


__all__ = [
    "EXCLUDED_SURFACES",
    "FORGET_STATUSES",
    "FORGET_SURFACES",
    "PROOF_CHECKS",
    "ForgetProof",
    "ForgetReport",
    "forget_record",
    "forgetting_problems",
    "verify_forgotten",
]
