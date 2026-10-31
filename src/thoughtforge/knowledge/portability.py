"""
Store portability — a whole store as a documented bundle (TF-052).

The roadmap's rule: *export/import of a whole store as a
documented bundle (Hermes-portability law applied to
memory).*

The measured starting state, before this slice (probed
this tick, keeper 07:12):

  * No portability module existed in
    ``thoughtforge.knowledge`` and ``MemoryStore`` carried
    no export, import, or bundle method at all. The only
    way to move a store was to copy the memory directory
    by hand — undocumented (which files are the store?
    which are derived?), unvalidated (a truncated or
    edited copy imports silently), and unverifiable (no
    counts, no checksums, nothing to prove the copy is
    the store).

What this slice builds:

  * ``export_store(store)`` — the whole store as one
    JSON-safe bundle (``BUNDLE_FORMAT`` /
    ``BUNDLE_VERSION``): every record home verbatim via
    its own ``to_dict()`` (personality core, profile
    records, episodes, patterns, thread state), the
    recent queue carried as data, a manifest naming the
    per-section record counts, and a SHA-256 checksum
    per section over its canonical JSON. ``write_bundle``
    persists it as one documented JSON file.
  * ``verify_bundle(bundle)`` — the gate. It re-derives
    everything the manifest claims (format, version,
    section set, counts, checksums), validates every
    record against TF-042's ingestion contract
    (``record_problems``), and refuses duplicate record
    ids anywhere in the bundle. Problems are named, one
    per fault; ``[]`` means the bundle may be imported.
  * ``import_store(store, bundle, mode=...)`` — the
    landing leg, in two modes named as data
    (``IMPORT_MODES``). Validation runs *whole-bundle
    first*: a bundle with any problem is refused entire
    and the target store is left untouched — no partial
    imports. ``replace`` lands the bundle as the whole
    store (existing records are rewritten out through
    the store's own rewrite shape). ``merge`` adds what
    is new and *never silently overwrites*: an id that
    already lives in the target is skipped and named on
    the report as a collision.
  * **Provenance.** Records on the four stamped kinds
    (preferences, facts, episodes, patterns) land with
    ``source="import"`` — TF-042's vocabulary for exactly
    this, and the stamp TF-050's contradiction authority
    table already prices. Every other field lands
    verbatim. The personality core and thread state carry
    no stamp by TF-042's ruling (identity / working state
    are not ingested claims). Round-trip rule, stated:
    export → import → export is stable from the second
    export on (the first import's only change to any
    record is that stamp).
  * **Derived surfaces are not carried.** The tag index
    is derived data (TF-051): the bundle does not carry
    it as authority; import rebuilds it from the landed
    records. The recent queue *is* carried (it is an
    ordering the rebuild cannot re-derive), filtered on
    import to ids that actually landed — a queue entry
    naming no landed record is dropped and counted, not
    left dangling.
  * ``ImportReport`` names the outcome either way:
    ``imported`` with per-section landed counts, skipped
    collisions, and the queue entries dropped; or
    ``refused`` with every problem named.

Stated boundaries:

  * The knowledge DB (Wikidata / ConceptNet / reference
    chunks in SQLite) is NOT part of the bundle: it is a
    read-only reference estate (TF-051's ruling), not
    this store's memory, and it is rebuilt by the ETL
    wing, not carried between stores.
  * A bundle is a *file*, and files can be copied by
    anyone who can read them: the bundle carries the
    store's contents in plain JSON — canon/persona data
    in a bundle is exactly as private as the store it
    came from, and is never published by this leg.
  * ``replace`` is destructive to the target by its
    stated meaning; that is why it must be named
    explicitly — there is no default mode.
  * Proof grade is FIXTURE (synthetic stores on this
    VM); portability over a real, lived-in store is NOT
    PROVEN (no real store exists on this machine,
    TF-041).

The human-readable write-up is
``docs/architecture/STORE_PORTABILITY.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.ingestion_contract import record_problems
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    PersonalityCoreRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.common import now_iso as _now_iso

# ── The bundle shape, as data ────────────────────────────────────────────────

BUNDLE_FORMAT = "thoughtforge-store-bundle"
BUNDLE_VERSION = 1

#: The sections a bundle carries, in bundle order.
#: ``single`` sections hold one record dict or None;
#: ``list`` sections hold record dicts; ``recent_queue``
#: holds record ids (an ordering, not records).
BUNDLE_SECTIONS: tuple[dict[str, Any], ...] = (
    {"name": "personality_core", "kind": "single",
     "what": "the persona's identity record (personality_core.yaml)"},
    {"name": "profile_records", "kind": "list",
     "what": "preferences + facts (user_profile_store.jsonl)"},
    {"name": "episodes", "kind": "list",
     "what": "episodes (episodic_store.jsonl)"},
    {"name": "patterns", "kind": "list",
     "what": "response patterns (response_patterns.jsonl)"},
    {"name": "thread_state", "kind": "single",
     "what": "the standing working record (active_thread_state.json)"},
    {"name": "recent_queue", "kind": "queue",
     "what": "recency ordering of record ids (recent_queue.json)"},
)

#: Surfaces deliberately NOT carried, with the reason.
EXCLUDED_SURFACES: tuple[dict[str, Any], ...] = (
    {"name": "tag_index", "file": "tag_index.json",
     "reason": "derived index — rebuilt from the landed records "
               "on import, never carried as authority"},
    {"name": "knowledge_db", "file": "knowledge.db",
     "reason": "read-only reference estate (Wikidata/ConceptNet/"
               "reference chunks); rebuilt by the ETL wing, holds "
               "no memory-store records"},
)

#: The import modes. There is deliberately no default:
#: ``replace`` destroys the target's current records by
#: its stated meaning, ``merge`` never overwrites one.
IMPORT_MODES: tuple[str, ...] = ("replace", "merge")

#: The statuses an ImportReport can carry.
IMPORT_STATUSES: tuple[str, ...] = ("imported", "refused")

#: The provenance stamp landed records carry (TF-042).
IMPORT_SOURCE = "import"

_RECORD_SECTIONS = ("personality_core", "profile_records", "episodes",
                    "patterns", "thread_state")
_LIST_SECTIONS = ("profile_records", "episodes", "patterns")


def _canonical(section_data: Any) -> str:
    return json.dumps(section_data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _checksum(section_data: Any) -> str:
    return hashlib.sha256(_canonical(section_data).encode("utf-8")).hexdigest()


def _section_count(section_data: Any, kind: str) -> int:
    if kind == "single":
        return 1 if section_data is not None else 0
    return len(section_data or [])


# ── Export ───────────────────────────────────────────────────────────────────


def export_store(store: MemoryStore) -> dict[str, Any]:
    """The whole store as one documented, checksummed
    bundle. Pure read of the store — exports nothing to
    disk; see ``write_bundle``."""
    core = store.load_personality_core()
    thread = store.load_thread_state()
    sections: dict[str, Any] = {
        "personality_core": core.to_dict() if core is not None else None,
        "profile_records": [r.to_dict() for r in store.load_profile_records()],
        "episodes": [r.to_dict() for r in store.load_episodes()],
        "patterns": [r.to_dict() for r in store.load_patterns()],
        "thread_state": thread.to_dict() if thread is not None else None,
        "recent_queue": list(store.load_recent_queue()),
    }
    manifest = {
        "counts": {
            row["name"]: _section_count(sections[row["name"]], row["kind"])
            for row in BUNDLE_SECTIONS
        },
        "checksums": {
            row["name"]: _checksum(sections[row["name"]])
            for row in BUNDLE_SECTIONS
        },
    }
    manifest["total_records"] = sum(
        manifest["counts"][name] for name in _RECORD_SECTIONS
    )
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "exported_at": _now_iso(),
        "manifest": manifest,
        "sections": sections,
    }


def write_bundle(store: MemoryStore, path: Path) -> dict[str, Any]:
    """Export ``store`` and write the bundle to ``path``
    as one JSON file. Returns the bundle."""
    bundle = export_store(store)
    Path(path).write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def read_bundle(path: Path) -> Any:
    """Read a bundle file back. Returns whatever JSON the
    file holds — validation is ``verify_bundle``'s job,
    so a malformed bundle is data to refuse, not a crash."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Verification ─────────────────────────────────────────────────────────────


def _record_dicts(bundle: dict[str, Any]) -> list[tuple[str, Any]]:
    sections = bundle.get("sections") or {}
    out: list[tuple[str, Any]] = []
    for name in _RECORD_SECTIONS:
        data = sections.get(name)
        if name in _LIST_SECTIONS:
            for entry in data or []:
                out.append((name, entry))
        elif data is not None:
            out.append((name, data))
    return out


def verify_bundle(bundle: Any) -> list[str]:
    """
    Every problem that bars this bundle from import, by
    name; ``[]`` means it may be imported. Pure — reads
    nothing, writes nothing. Checks: the format and
    version, the exact section set, manifest counts and
    checksums re-derived from the sections, every record
    against TF-042's ingestion contract, and no record
    id appearing twice anywhere in the bundle.
    """
    problems: list[str] = []
    if not isinstance(bundle, dict):
        return [f"bundle is not a mapping: {type(bundle).__name__}"]
    if bundle.get("format") != BUNDLE_FORMAT:
        problems.append(
            f"format {bundle.get('format')!r} != {BUNDLE_FORMAT!r}"
        )
    if bundle.get("version") != BUNDLE_VERSION:
        problems.append(
            f"version {bundle.get('version')!r} != bundle version "
            f"{BUNDLE_VERSION}"
        )
    sections = bundle.get("sections")
    if not isinstance(sections, dict):
        problems.append("sections missing or not a mapping")
        return problems
    expected = [row["name"] for row in BUNDLE_SECTIONS]
    if set(sections) != set(expected):
        problems.append(
            f"sections {sorted(sections)} != bundle sections {expected}"
        )
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        problems.append("manifest missing or not a mapping")
        manifest = {}
    counts = manifest.get("counts") or {}
    checksums = manifest.get("checksums") or {}
    for row in BUNDLE_SECTIONS:
        name = row["name"]
        if name not in sections:
            continue
        data = sections[name]
        if row["kind"] in ("list", "queue") and not isinstance(data, list):
            problems.append(f"section {name} is not a list")
            continue
        if counts.get(name) != _section_count(data, row["kind"]):
            problems.append(
                f"manifest count for {name} is {counts.get(name)!r}, "
                f"section holds {_section_count(data, row['kind'])}"
            )
        if checksums.get(name) != _checksum(data):
            problems.append(f"manifest checksum mismatch for section {name}")
    if isinstance(manifest.get("total_records"), int):
        total = sum(
            _section_count(sections.get(name), "single" if name in
                           ("personality_core", "thread_state") else "list")
            for name in _RECORD_SECTIONS if name in sections
        )
        if manifest["total_records"] != total:
            problems.append(
                f"manifest total_records is {manifest['total_records']}, "
                f"sections hold {total}"
            )
    seen: dict[str, str] = {}
    for section_name, entry in _record_dicts(bundle):
        if not isinstance(entry, dict):
            problems.append(f"{section_name}: record is not a mapping")
            continue
        record_id = entry.get("record_id")
        for problem in record_problems(entry):
            problems.append(f"{section_name} {record_id or '?'}: {problem}")
        # TF-062: a refuse-verdict record (instruction-shaped
        # text) bars the bundle here, before anything lands —
        # the import stays refused-entire, never half-landed
        # on a poisoned record (the landing gate would raise
        # mid-import otherwise). Quarantine verdicts do not
        # bar a bundle; retrieval screening governs them.
        if section_name in ("profile_records", "episodes", "patterns"):
            from thoughtforge.knowledge.adversarial import (
                refusal_problems,
            )

            for problem in refusal_problems(entry):
                problems.append(
                    f"{section_name} {record_id or '?'}: {problem}")
        if isinstance(record_id, str) and record_id:
            if record_id in seen:
                problems.append(
                    f"duplicate record id {record_id} in {section_name} "
                    f"(already in {seen[record_id]})"
                )
            seen.setdefault(record_id, section_name)
    queue = sections.get("recent_queue")
    if isinstance(queue, list) and any(not isinstance(x, str) for x in queue):
        problems.append("recent_queue holds a non-string entry")
    return problems


# ── Import ───────────────────────────────────────────────────────────────────


@dataclass
class ImportReport:
    """What one import_store call did, as data. A refusal
    names every problem; an import names what landed,
    what collided (merge), and what queue entries were
    dropped as dangling."""

    status: str = "refused"
    mode: str = ""
    landed: dict[str, int] = field(default_factory=dict)
    skipped_collisions: list[str] = field(default_factory=list)
    queue_dropped: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def imported(self) -> bool:
        return self.status == "imported"

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["imported"] = self.imported
        return data


def _build_records(bundle: dict[str, Any]) -> dict[str, Any]:
    sections = bundle["sections"]
    built: dict[str, Any] = {}
    core = sections.get("personality_core")
    built["personality_core"] = (
        PersonalityCoreRecord.from_dict(core) if core is not None else None
    )
    profile: list[Any] = []
    for entry in sections.get("profile_records") or []:
        entry = dict(entry, source=IMPORT_SOURCE)
        if entry.get("record_type") == "user_preference":
            profile.append(UserPreferenceRecord.from_dict(entry))
        else:
            profile.append(UserFactRecord.from_dict(entry))
    built["profile_records"] = profile
    built["episodes"] = [
        EpisodicMemoryRecord.from_dict(dict(e, source=IMPORT_SOURCE))
        for e in sections.get("episodes") or []
    ]
    built["patterns"] = [
        ResponsePatternRecord.from_dict(dict(e, source=IMPORT_SOURCE))
        for e in sections.get("patterns") or []
    ]
    thread = sections.get("thread_state")
    built["thread_state"] = (
        ActiveThreadStateRecord.from_dict(thread) if thread is not None else None
    )
    return built


def import_store(store: MemoryStore, bundle: Any, *, mode: str) -> ImportReport:
    """
    Land a verified bundle in ``store``. ``mode`` is
    required and must name a mode in ``IMPORT_MODES``:
    ``replace`` makes the bundle the whole store;
    ``merge`` adds new records and skips — never
    overwrites — ids already present.

    The bundle is verified whole before anything is
    written: any problem refuses the import entire and
    the target is left untouched.
    """
    if mode not in IMPORT_MODES:
        raise ValueError(
            f"import mode {mode!r} not in {list(IMPORT_MODES)} — "
            "name 'replace' or 'merge' explicitly"
        )
    report = ImportReport(mode=mode)
    problems = verify_bundle(bundle)
    if problems:
        report.problems = problems
        return report
    built = _build_records(bundle)

    if mode == "replace":
        if built["personality_core"] is not None:
            store.save_personality_core(built["personality_core"])
        elif store.load_personality_core() is not None:
            store._personality_path.unlink()
        _land_list(store, "profile_records", built["profile_records"], replace=True)
        _land_list(store, "episodes", built["episodes"], replace=True)
        _land_list(store, "patterns", built["patterns"], replace=True)
        if built["thread_state"] is not None:
            store.save_thread_state(built["thread_state"])
        else:
            store.clear_thread_state()
        landed_ids = {
            r.record_id
            for name in ("profile_records", "episodes", "patterns")
            for r in built[name]
        }
        if built["personality_core"] is not None:
            landed_ids.add(built["personality_core"].record_id)
        if built["thread_state"] is not None:
            landed_ids.add(built["thread_state"].record_id)
    else:  # merge
        existing: set[str] = set()
        for record in store.load_profile_records():
            existing.add(record.record_id)
        for record in store.load_episodes():
            existing.add(record.record_id)
        for record in store.load_patterns():
            existing.add(record.record_id)
        core = store.load_personality_core()
        if core is not None:
            existing.add(core.record_id)
        thread = store.load_thread_state()
        if thread is not None:
            existing.add(thread.record_id)

        def _keep(records: list[Any]) -> list[Any]:
            kept = []
            for record in records:
                if record.record_id in existing:
                    report.skipped_collisions.append(record.record_id)
                else:
                    kept.append(record)
                    existing.add(record.record_id)
            return kept

        built["profile_records"] = _keep(built["profile_records"])
        built["episodes"] = _keep(built["episodes"])
        built["patterns"] = _keep(built["patterns"])
        for name in ("personality_core", "thread_state"):
            record = built[name]
            if record is not None and record.record_id in existing:
                report.skipped_collisions.append(record.record_id)
                built[name] = None
        if built["personality_core"] is not None:
            store.save_personality_core(built["personality_core"])
        _land_list(store, "profile_records", built["profile_records"], replace=False)
        _land_list(store, "episodes", built["episodes"], replace=False)
        _land_list(store, "patterns", built["patterns"], replace=False)
        if built["thread_state"] is not None:
            store.save_thread_state(built["thread_state"])
        landed_ids = {
            r.record_id
            for name in ("profile_records", "episodes", "patterns")
            for r in built[name]
        }
        if built["personality_core"] is not None:
            landed_ids.add(built["personality_core"].record_id)
        if built["thread_state"] is not None:
            landed_ids.add(built["thread_state"].record_id)

    report.landed = {
        "personality_core": 1 if built["personality_core"] is not None else 0,
        "profile_records": len(built["profile_records"]),
        "episodes": len(built["episodes"]),
        "patterns": len(built["patterns"]),
        "thread_state": 1 if built["thread_state"] is not None else 0,
    }
    # The queue: carried as data, filtered to landed (merge:
    # or already-present) ids — a dangling entry is dropped
    # and counted, never left pointing at nothing.
    if mode == "merge":
        landed_ids |= {
            r.record_id
            for loader in (store.load_profile_records, store.load_episodes,
                           store.load_patterns)
            for r in loader()
        }
    queue_in = list(bundle["sections"].get("recent_queue") or [])
    queue_out = [x for x in queue_in if x in landed_ids]
    report.queue_dropped = len(queue_in) - len(queue_out)
    store.save_recent_queue(queue_out)
    # The tag index is derived: rebuilt from what landed,
    # never carried as authority.
    store.rebuild_tag_index()
    report.status = "imported"
    return report


def _land_list(store: MemoryStore, section: str, records: list[Any],
               *, replace: bool) -> None:
    """Write one list section through the store's own
    shapes: replace rewrites the file with exactly the
    bundled records; merge appends through the gated
    append paths (TF-042 validation at the gate)."""
    if replace:
        # rewrite_* drops records whose status is archived /
        # deleted; a whole-store replace must land the file
        # empty first, then append every bundled record.
        if section == "profile_records":
            store.rewrite_profile_store([])
            for record in records:
                store.append_profile_record(record)
        elif section == "episodes":
            store.rewrite_episodic_store([])
            for record in records:
                store.append_episode(record)
        else:
            store.rewrite_patterns_store([])
            for record in records:
                store.append_pattern(record)
    else:
        for record in records:
            if section == "profile_records":
                store.append_profile_record(record)
            elif section == "episodes":
                store.append_episode(record)
            else:
                store.append_pattern(record)


# ── Self-check ───────────────────────────────────────────────────────────────


def portability_problems() -> list[str]:
    """
    Re-check this slice's claims against the live tree:
    the bundle sections name files the store owns, the
    excluded surfaces are the store's derived file and
    the ETL estate, the import stamp is TF-042's
    vocabulary word for it, and a synthetic round trip
    through the live code lands stable. Returns problems
    by name; ``[]`` when portability still matches.
    """
    import tempfile
    from pathlib import Path as _Path

    from thoughtforge.knowledge import store as store_module
    from thoughtforge.knowledge.ingestion_contract import MEMORY_PROVENANCE

    problems: list[str] = []

    store_src = inspect.getsource(store_module)
    for row in BUNDLE_SECTIONS:
        file_hint = {
            "personality_core": "personality_core.yaml",
            "profile_records": "user_profile_store.jsonl",
            "episodes": "episodic_store.jsonl",
            "patterns": "response_patterns.jsonl",
            "thread_state": "active_thread_state.json",
            "recent_queue": "recent_queue.json",
        }[row["name"]]
        if file_hint not in store_src:
            problems.append(
                f"section {row['name']}: file {file_hint} is not a "
                "MemoryStore file in store.py"
            )
    if "tag_index.json" not in store_src:
        problems.append("excluded tag_index is not a MemoryStore file")
    if IMPORT_SOURCE not in MEMORY_PROVENANCE:
        problems.append(
            f"import stamp {IMPORT_SOURCE!r} is not in TF-042's "
            "MEMORY_PROVENANCE vocabulary"
        )
    if set(IMPORT_STATUSES) != {"imported", "refused"}:
        problems.append("IMPORT_STATUSES drifted from the stated statuses")
    if tuple(IMPORT_MODES) != ("replace", "merge"):
        problems.append("IMPORT_MODES drifted from the stated modes")

    # The round trip, live: export a seeded store, replace
    # into a fresh one, re-export — sections must agree.
    with tempfile.TemporaryDirectory() as tmp:
        src = MemoryStore(memory_dir=_Path(tmp) / "src")
        src.append_profile_record(UserFactRecord(
            record_id="fct_probe", summary="probe fact",
            fact_value={"k": "v"}, source="user_stated",
        ))
        bundle = export_store(src)
        if verify_bundle(bundle):
            problems.append(
                f"self-check bundle does not verify: {verify_bundle(bundle)}"
            )
        dst = MemoryStore(memory_dir=_Path(tmp) / "dst")
        report = import_store(dst, bundle, mode="replace")
        if not report.imported:
            problems.append(f"self-check import refused: {report.problems}")
        else:
            again = export_store(dst)
            landed = again["sections"]["profile_records"]
            if len(landed) != 1 or landed[0]["record_id"] != "fct_probe":
                problems.append("self-check round trip lost the probe record")
            elif landed[0].get("source") != IMPORT_SOURCE:
                problems.append("self-check landed record lacks the import stamp")
    return problems


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SECTIONS",
    "BUNDLE_VERSION",
    "EXCLUDED_SURFACES",
    "IMPORT_MODES",
    "IMPORT_SOURCE",
    "IMPORT_STATUSES",
    "ImportReport",
    "export_store",
    "import_store",
    "portability_problems",
    "read_bundle",
    "verify_bundle",
    "write_bundle",
]
