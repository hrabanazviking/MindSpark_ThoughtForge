"""
Wikidata pipeline hardening, as data (TF-107).

TF-105 mapped the Wikidata builder honestly: it streamed a
dump into the store, but nothing about a long ingestion
survived interruption. TF-107's probes measured the estate
before this slice changed it:

  * no checkpoint, no resume, no pacing anywhere in
    ``etl/wikidata.py`` (grep: checkpoint / resume /
    throttle all absent); the only commits were every
    50,000 entities and at the end, under
    ``PRAGMA synchronous = OFF`` — an interruption (kill,
    crash, power loss) discarded every uncommitted entity
    since the last 50k boundary, and OFF is the posture
    SQLite itself warns can corrupt on power loss;
  * re-running a build — the only "resume" the old shape
    offered — DUPLICATED statements: entities and labels
    flush ``INSERT OR REPLACE`` but statements flush a
    plain ``INSERT`` into an AUTOINCREMENT table, so a
    live probe's second run moved a 7-statement store to
    14 while entities and labels stayed at 7;
  * the failure path masked its own error: the build's
    connection was never closed on exception, so the
    ``_finish_log`` call on a fresh connection raised
    ``sqlite3.OperationalError: database is locked`` and
    the original failure — and its log row — were lost.

The hardening, filed here as data and enforced in
``etl/wikidata.py``:

  * ``PIPELINE_RULES`` — the law: checkpointed commits,
    resume by fingerprint, idempotent rows, a stated rate
    cap, durable posture, honest failure.
  * Checkpoints — a sidecar JSON record beside the DB
    (``<db>.wikidata-checkpoint.json``), written atomically
    (tmp file + ``os.replace``) ONLY after the commit it
    describes has landed, at ``DEFAULT_CHECKPOINT_EVERY``
    stream items and at completion. The record carries the
    dump fingerprint (resolved path + size + mtime_ns), the
    config fingerprint (languages, instance_of filter,
    limit), the stream position (``items_seen``), the stats
    as of that commit, the last QID seen, and its status.
    The schema gains NO table for this (TF-105's filed
    schema facts stand untouched).
  * ``RatePacer`` — the rate cap as a measured ceiling:
    when ``max_items_per_second`` is set, processing item
    N may not finish before N / cap seconds have elapsed
    since the run started (clock + sleeper injectable, so
    the pacing is proven with a fake clock, never by
    watching a wall clock in a test). Unset (the default)
    means no cap: this builder reads a LOCAL dump file —
    there is no upstream service whose limits a default
    cap would respect, and a default that never binds
    would be theatre. The filed limit is stated in
    ``docs/architecture/WIKIDATA_PIPELINE.md`` §4.
  * The checkers: ``pipeline_problems()`` judges the filed
    rules against the live builder source (the hardening
    must be present in fact, not just described), and
    ``pipeline_live_problems()`` re-runs the whole road
    behaviourally on a synthetic dump in a temp dir —
    fresh run, interruption, resume, re-run, pacing,
    mismatch refusal.

Docket (TF-016): D-WIKI-1 (statement duplication),
D-WIKI-2 (no checkpoint/resume; OFF posture), and
D-WIKI-3 (failure path masks the error) are added and
FIXED by this slice; D-COV-2's wikidata-depth share is
discharged here (TF-112's dry-run depth remains).

The human-readable write-up is
``docs/architecture/WIKIDATA_PIPELINE.md``. Proof grade:
FIXTURE — synthetic dumps through the real builder on
this VM, interruption simulated by a stream that raises
mid-dump; a real multi-GB dump ingestion, a real kill
-9 / power loss, and any at-scale rate measurement are
NOT PROVEN, never claimed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

PIPELINE_FORMAT = "thoughtforge-wikidata-pipeline/1"
PIPELINE_DOC = "docs/architecture/WIKIDATA_PIPELINE.md"
CHECKPOINT_FORMAT = "thoughtforge-wikidata-checkpoint/1"

# Stream items between checkpointed commits in the builder.
DEFAULT_CHECKPOINT_EVERY = 1000

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"


class CheckpointMismatchError(ValueError):
    """A stored checkpoint does not describe this build's ask."""


class CheckpointCorruptError(ValueError):
    """A stored checkpoint cannot be read or is malformed."""


# ── The rules ───────────────────────────────────────────────────────────────

PIPELINE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "checkpoint_after_commit_only",
     "rule": "a checkpoint is written only after the commit it describes has landed; the checkpoint on disk is therefore always a truthful resume point, never an aspiration"},
    {"id": "checkpoint_cadence_bounds_loss",
     "rule": "commits + checkpoints land every DEFAULT_CHECKPOINT_EVERY stream items and at completion, so an interruption loses at most one cadence interval of uncommitted work (the pre-TF-107 bound was 50,000 entities under synchronous OFF)"},
    {"id": "resume_is_by_fingerprint",
     "rule": "resume continues only when the checkpoint's dump fingerprint (resolved path + size + mtime_ns) and config fingerprint (languages, instance_of filter, limit) match this build's ask exactly; any difference raises CheckpointMismatchError by name — a changed ask is a new build (resume=False), never a silent continuation"},
    {"id": "a_completed_checkpoint_is_the_result",
     "rule": "resuming a build whose checkpoint is complete returns the checkpoint's filed stats without re-streaming the dump"},
    {"id": "rows_are_idempotent",
     "rule": "re-processing an item never duplicates a row: entities and labels replace by key, and a statement whose full stored tuple (subject, property, object, type, qualifiers, refs, source) is already present is not inserted again — this slice owns only the duplicate line; changed-fact updates are TF-114's lifecycle supersede (etl/update_ingestion.py)"},
    {"id": "the_rate_cap_is_a_ceiling_when_set",
     "rule": "with max_items_per_second set, item N of a run may not finish processing before N / cap seconds have elapsed on the run's clock; unset means uncapped, stated plainly — the dump is a local file and no upstream limit exists to respect by default"},
    {"id": "checkpoints_are_durable",
     "rule": "the build runs under WAL with synchronous = FULL (the TF-054 / Samtal posture), so a landed checkpoint commit survives power loss; the checkpoint file itself is replaced atomically"},
    {"id": "failure_never_masks_failure",
     "rule": "on error the build connection is rolled back and closed BEFORE the failure is recorded in ingestion_log, the original exception is re-raised unchanged, and the last landed checkpoint stays on disk as the resume point"},
    {"id": "a_corrupt_checkpoint_refuses",
     "rule": "an unreadable or malformed checkpoint raises CheckpointCorruptError by name; the builder never silently restarts over a checkpoint it cannot read (resume=False discards it explicitly)"},
)


# ── Checkpoint records ──────────────────────────────────────────────────────

def checkpoint_path_for(db_path: Path | str) -> Path:
    """The sidecar checkpoint path for a knowledge DB."""
    return Path(str(db_path) + ".wikidata-checkpoint.json")


def dump_fingerprint(dump_path: Path | str) -> dict[str, Any]:
    """Identity of the dump file: resolved path + size + mtime_ns."""
    path = Path(dump_path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def config_fingerprint(languages: list[str] | tuple[str, ...],
                       instance_of_filter: set[str] | None,
                       limit: int | None) -> dict[str, Any]:
    """Identity of the build ask a checkpoint was made under."""
    return {
        "languages": sorted(languages),
        "instance_of_filter": (sorted(instance_of_filter)
                               if instance_of_filter else None),
        "limit": limit,
    }


def write_checkpoint(db_path: Path | str, record: dict[str, Any]) -> Path:
    """Write ``record`` atomically to the checkpoint sidecar."""
    target = checkpoint_path_for(db_path)
    payload = dict(record)
    payload["format"] = CHECKPOINT_FORMAT
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_checkpoint(db_path: Path | str) -> dict[str, Any] | None:
    """The stored checkpoint, or None when none exists.

    Raises ``CheckpointCorruptError`` when a checkpoint file
    exists but cannot be read or lacks the fields resume
    depends on — never a silent None over a present file.
    """
    target = checkpoint_path_for(db_path)
    if not target.exists():
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(
            f"checkpoint at {target} is unreadable: {exc}") from exc
    if not isinstance(record, dict) \
            or record.get("format") != CHECKPOINT_FORMAT \
            or not isinstance(record.get("items_seen"), int) \
            or not isinstance(record.get("stats"), dict) \
            or record.get("status") not in (STATUS_IN_PROGRESS,
                                            STATUS_COMPLETE) \
            or "dump" not in record or "config" not in record:
        raise CheckpointCorruptError(
            f"checkpoint at {target} is malformed (format, "
            "items_seen, stats, status, dump and config are "
            "required)")
    return record


def discard_checkpoint(db_path: Path | str) -> bool:
    """Remove the checkpoint sidecar; True when one was removed."""
    target = checkpoint_path_for(db_path)
    if target.exists():
        target.unlink()
        return True
    return False


def checkpoint_mismatches(record: dict[str, Any],
                          dump_fp: dict[str, Any],
                          config_fp: dict[str, Any]) -> list[str]:
    """Named differences between a checkpoint and this build's ask."""
    problems: list[str] = []
    filed_dump = record.get("dump") or {}
    for field in ("path", "size", "mtime_ns"):
        if filed_dump.get(field) != dump_fp.get(field):
            problems.append(
                f"dump {field}: checkpoint has "
                f"{filed_dump.get(field)!r}, this build asks "
                f"{dump_fp.get(field)!r}")
    filed_config = record.get("config") or {}
    for field in ("languages", "instance_of_filter", "limit"):
        if filed_config.get(field) != config_fp.get(field):
            problems.append(
                f"config {field}: checkpoint has "
                f"{filed_config.get(field)!r}, this build asks "
                f"{config_fp.get(field)!r}")
    return problems


# ── Rate pacing ─────────────────────────────────────────────────────────────

class RatePacer:
    """A hard items-per-second ceiling over a run's processing.

    ``wait_for(n)`` sleeps (via the injected sleeper) until
    at least ``n / max_per_second`` seconds have elapsed on
    the injected clock since construction. A non-positive
    cap is refused by name — a cap that cannot bind is a
    configuration error, not a no-op.
    """

    def __init__(self, max_items_per_second: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if max_items_per_second <= 0:
            raise ValueError(
                f"max_items_per_second must be positive, got "
                f"{max_items_per_second!r}")
        self.max_items_per_second = float(max_items_per_second)
        self._clock = clock
        self._sleep = sleep
        self._started = clock()
        self.sleeps: list[float] = []

    def wait_for(self, items_done: int) -> float:
        """Block until item ``items_done`` is allowed; seconds slept."""
        if items_done <= 0:
            return 0.0
        deadline = self._started + items_done / self.max_items_per_second
        remaining = deadline - self._clock()
        if remaining <= 0:
            return 0.0
        self._sleep(remaining)
        self.sleeps.append(remaining)
        return remaining


# ── Checkers ────────────────────────────────────────────────────────────────

def pipeline_problems() -> list[str]:
    """Static truth: the filed rules against the live builder."""
    from thoughtforge.etl import wikidata as wikidata_module

    problems: list[str] = []
    rule_ids = [r["id"] for r in PIPELINE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate PIPELINE_RULES ids")

    src = Path(wikidata_module.__file__).read_text(encoding="utf-8")
    if "from thoughtforge.etl.wikidata_pipeline import" not in src:
        problems.append("wikidata lost its wikidata_pipeline "
                        "import: the builder must use the "
                        "checkpoint/pacing surface filed here")
    build_src = src[src.find("def build("):]
    for marker, why in (
            ("read_checkpoint", "resume must read the checkpoint"),
            ("write_checkpoint", "the builder must write checkpoints"),
            ("checkpoint_mismatches", "resume must judge the "
             "fingerprint before continuing"),
            ("RatePacer", "the rate cap must be wired"),
            ('"PRAGMA synchronous = FULL"', "the durable posture "
             "must be FULL (D-WIKI-2)"),
            ):
        if marker not in build_src:
            problems.append(f"wikidata build lost {marker!r}: {why}")
    if "PRAGMA synchronous = OFF" in src:
        problems.append("D-WIKI-2 reopened: the wikidata load "
                        "runs synchronous OFF again")
    flush_src = src[src.find("def _flush_statements"):]
    flush_src = flush_src[:flush_src.find("\n    @staticmethod", 1)] \
        if "\n    @staticmethod" in flush_src[1:] else flush_src
    if "already" not in flush_src and "existing" not in flush_src:
        problems.append("D-WIKI-1 reopened: _flush_statements no "
                        "longer guards against existing rows")
    if wikidata_module.DEFAULT_CHECKPOINT_EVERY != DEFAULT_CHECKPOINT_EVERY:
        problems.append("the builder's checkpoint cadence drifted "
                        "from DEFAULT_CHECKPOINT_EVERY")
    return problems


def _synthetic_entity(qid: str) -> dict[str, Any]:
    return {
        "id": qid,
        "labels": {"en": {"value": f"Lantern {qid}"}},
        "descriptions": {"en": {"value": "a synthetic entity"}},
        "aliases": {"en": []},
        "sitelinks": {},
        "claims": {"P31": [{
            "mainsnak": {"snaktype": "value",
                         "datavalue": {"type": "wikibase-entityid",
                                       "value": {"id": "Q107900"}}},
            "rank": "normal", "qualifiers": {}, "references": [],
        }]},
    }


def pipeline_live_problems() -> list[str]:
    """Behavioural truth: the whole road, run on synthetic dumps."""
    from thoughtforge.etl.wikidata import WikidataETL

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "live.db"
        dump = root / "dump.json"
        entities = [_synthetic_entity(f"Q{i}") for i in range(1, 8)]
        dump.write_text(json.dumps(entities), encoding="utf-8")

        def counts() -> dict[str, int]:
            conn = sqlite3.connect(str(db_path))
            try:
                return {t: conn.execute(
                    f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("entities", "statements", "labels")}
            finally:
                conn.close()

        # Fresh run completes and files a complete checkpoint.
        stats = WikidataETL(db_path).build(
            dump_path=dump, checkpoint_every=3)
        if stats != {"entities": 7, "statements": 7, "labels": 7,
                     "skipped": 0}:
            problems.append(f"fresh run stats drifted: {stats}")
        record = read_checkpoint(db_path)
        if record is None or record["status"] != STATUS_COMPLETE \
                or record["items_seen"] != 7:
            problems.append(f"no complete checkpoint after run: "
                            f"{record and record.get('status')}")
        if counts() != {"entities": 7, "statements": 7, "labels": 7}:
            problems.append(f"store counts after fresh run: {counts()}")

        # Re-running the same build duplicates nothing (D-WIKI-1).
        WikidataETL(db_path).build(dump_path=dump, checkpoint_every=3)
        if counts()["statements"] != 7:
            problems.append("D-WIKI-1 reopened live: a re-run "
                            f"duplicated statements: {counts()}")

        # A fresh start over the same store still duplicates
        # nothing (statement idempotence at the row level).
        WikidataETL(db_path).build(
            dump_path=dump, checkpoint_every=3, resume=False)
        if counts() != {"entities": 7, "statements": 7, "labels": 7}:
            problems.append(f"resume=False re-run changed counts: "
                            f"{counts()}")

        # Interruption + resume on a second store.
        db2 = root / "live2.db"
        etl = WikidataETL(db2)
        stream = etl._stream_entities

        def flaky(path: Path, _stream: Any = stream) -> Any:
            for i, item in enumerate(_stream(path)):
                if i == 4:
                    raise RuntimeError("tf107 simulated interruption")
                yield item

        etl._stream_entities = flaky  # type: ignore[method-assign]
        try:
            etl.build(dump_path=dump, checkpoint_every=3)
            problems.append("the interrupted build did not raise")
        except RuntimeError as exc:
            if "simulated interruption" not in str(exc):
                problems.append(f"the interruption was masked: {exc!r}")
        record2 = read_checkpoint(db2)
        if record2 is None or record2["status"] != STATUS_IN_PROGRESS \
                or record2["items_seen"] != 3:
            problems.append(f"checkpoint after interruption: "
                            f"{record2 and record2.get('items_seen')}")
        conn = sqlite3.connect(str(db2))
        try:
            landed = conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0]
            log = conn.execute(
                "SELECT status FROM ingestion_log").fetchall()
        finally:
            conn.close()
        if landed != 3:
            problems.append(f"entities landed at interruption: "
                            f"{landed} (one cadence interval = 3)")
        if not any(str(row[0]).startswith("failed") for row in log):
            problems.append(f"the failure was not logged: {log}")

        resumed = WikidataETL(db2).build(
            dump_path=dump, checkpoint_every=3)
        if resumed != stats:
            problems.append(f"resumed stats {resumed} != {stats}")
        conn = sqlite3.connect(str(db2))
        try:
            final = {t: conn.execute(
                f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("entities", "statements", "labels")}
        finally:
            conn.close()
        if final != {"entities": 7, "statements": 7, "labels": 7}:
            problems.append(f"store after resume: {final}")

        # A changed ask refuses to resume (fingerprint rule).
        try:
            WikidataETL(db2).build(
                dump_path=dump, checkpoint_every=3, limit=2)
            problems.append("a changed-limit build resumed over a "
                            "mismatched checkpoint")
        except CheckpointMismatchError:
            pass

        # Pacing: a fake clock proves the ceiling's arithmetic.
        now = [100.0]
        slept: list[float] = []
        pacer = RatePacer(10.0, clock=lambda: now[0],
                          sleep=lambda s: slept.append(s))
        pacer.wait_for(5)
        if slept != [0.5]:
            problems.append(f"pacer slept {slept} for 5 items at "
                            "10/s (0.5 expected)")
        now[0] += 1.0
        if pacer.wait_for(10) != 0.0:
            problems.append("pacer slept past its deadline")
        try:
            RatePacer(0)
            problems.append("a zero rate cap was admitted")
        except ValueError:
            pass
    return problems


def pipeline_as_data() -> dict[str, Any]:
    """The hardening as plain data, for consumers and the doctor."""
    return {
        "format": PIPELINE_FORMAT,
        "doc": PIPELINE_DOC,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_every_default": DEFAULT_CHECKPOINT_EVERY,
        "rules": [dict(r) for r in PIPELINE_RULES],
    }
