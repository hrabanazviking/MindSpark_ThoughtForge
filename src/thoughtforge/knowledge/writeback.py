"""
Conversation → memory writeback — the gated persistence leg (TF-047).

Before this slice the estate had every part of writeback except
the leg itself:

  * ``WritebackRecord`` (knowledge/models.py) — the transient
    record that "drives persistence", with slots for new
    preferences / facts / episodes / patterns, thread updates,
    and reinforce / archive / delete lists. Nothing in the live
    tree ever constructed one for a real turn, and nothing ever
    applied one: ``RuntimeTurnState.writeback`` was a slot no
    producer filled and no consumer read.
  * The writeback *scoring* formulas and their thresholds
    (knowledge/scoring.py, named by TF-044): preference ≥ 0.72,
    episode ≥ 0.58, pattern-learning ≥ 0.78. No consumer existed.
  * The writeback *contract* (TF-042): ``writeback_fact_problems``
    was written for exactly this leg, and the provenance gap —
    no memory record could carry a source stamp — was pinned
    with "TF-047 owns stamping it".
  * The store's append gates (TF-042): append_* / save_* refuse
    a contract-invalid record. Nothing called them from a turn.

This module is the leg, and its gates, in the order they fire:

  1. **The enforcement gate (the Part V rule).** A turn whose
     final response did not pass the EnforcementGate writes
     NOTHING — no new records, no thread updates, no reinforce /
     archive / delete. The outcome says so
     (``enforcement_gated``) and lists every requested item as
     refused with that reason: a failed answer teaches the store
     nothing, and nothing it asked for is silently dropped.
  2. **The contract gate (TF-042).** Every new entry is
     validated before construction reaches a file: facts through
     ``writeback_fact_problems`` (content + a provenance stamp
     in ``MEMORY_PROVENANCE`` are required there), the other
     kinds through ``record_problems`` on the constructed record
     plus the same stamp requirement. An entry naming a source
     outside the vocabulary is refused, never re-stamped.
  3. **The score gates (TF-044).** Preferences, episodes, and
     patterns must clear their named thresholds under the named
     formulas. Components come from the entry's ``components``
     dict where given; a missing component falls back to the
     entry's own matching field (confidence, importance,
     quality) or to ``NEUTRAL_COMPONENT`` (0.5) — the neutral
     value TF-045 already measured in the lifecycle builder.
     Facts carry no writeback formula in the tree, so facts are
     contract-gated only; this module invents no fact score.
  4. **The store gate (TF-042).** The append itself re-validates
     through ``MemoryStore``; a ``ValueError`` there is caught
     and recorded as a refusal, never raised into a turn.

**Provenance (the TF-042 gap, closed).** The four writeback
record kinds now carry a ``source`` field (models.py; legacy
records load with ``""``). Every record this leg writes is
stamped: the entry's own ``source`` when it names a vocabulary
member, else ``WRITEBACK_SOURCE`` (``conversation_writeback``).
Identity and thread state carry no stamp by ruling — they are
not ingested claims — and TF-042's contract pin now checks the
split both ways.

**Reinforce / archive / delete** resolve an id across the three
append stores (profile, episodes, patterns) in that order.
Reinforce increments ``times_reinforced``, sets status
``reinforced``, and rewrites that store (the rewrite mechanisms
are TF-041's prune machinery, reused, not duplicated). Archive
and delete set the status and rewrite, which drops the record
from its file — the store's own deletion shape. An id found
nowhere is a recorded refusal (``record not found``), never a
silent no-op. The personality core is deliberately NOT
reachable from any of these lists: writeback cannot archive or
delete identity.

**Thread updates** apply to the one working-tier record: the
fields an update names (topic / summary / tags / open_loops /
priority) replace the standing ones, ``turns_since_touch``
resets to 0, and the record is saved through the store's
contract gate. With no standing record, an update creates one —
but only with a topic; a topicless creation is refused.

**Core wiring.** ``ThoughtForgeCore._finish_turn`` applies
``turn.writeback`` through ``core.apply_writeback(...)`` with
the final response's ``enforcement_passed`` — the same
never-kill-a-completed-turn discipline as TF-045's cooling (a
writeback failure is logged at warning and the final stands).
The outcome is retained on ``core.last_writeback_outcome``.

**Honesty.** The *producer* is not built here: no extractor yet
fills ``WritebackRecord`` from a live conversation (the
prompt builder's episode-summary / preference-inference prompts
exist, TF-023, but no code calls them and no model output is
parsed into a writeback). What a chat writes back, when a
writeback is produced, is now fully specified and enforced;
claiming conversations already learn on their own would be
false. TF-063's long-conversation proof owns the producer
question. Proof grade for this slice is FIXTURE (synthetic
stores and a knowledge-only core on this VM).

The human-readable write-up is
``docs/architecture/CONVERSATION_WRITEBACK.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.ingestion_contract import (
    FACT_KINDS,
    MEMORY_PROVENANCE,
    record_problems,
    writeback_fact_problems,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    EpisodicMemoryRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
    WritebackRecord,
)
from thoughtforge.knowledge.scoring import (
    MIN_EPISODE_WRITEBACK_SCORE,
    MIN_PATTERN_LEARNING_SCORE,
    MIN_PREFERENCE_WRITEBACK_SCORE,
    WritebackComponents,
    compute_episode_writeback_score,
    compute_pattern_learning_score,
    compute_preference_writeback_score,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.common import now_iso as _now_iso

#: The stamp applied when a writeback entry names no source of
#: its own — TF-042's vocabulary member for exactly this leg.
WRITEBACK_SOURCE: str = "conversation_writeback"

#: The value a missing score component falls back to when the
#: entry carries no matching field either. 0.5 is the neutral
#: share TF-045 measured for future_relevance in the lifecycle
#: builder's defaults; it is a stated policy value here, not a
#: measured property of any conversation.
NEUTRAL_COMPONENT: float = 0.5

#: Score gates, as data: kind → (formula, threshold). The
#: thresholds are scoring.py's named constants (TF-044), imported
#: — never copied — and writeback_problems() re-checks that.
SCORE_GATES: dict[str, tuple[Any, float]] = {
    "user_preference": (compute_preference_writeback_score, MIN_PREFERENCE_WRITEBACK_SCORE),
    "episode": (compute_episode_writeback_score, MIN_EPISODE_WRITEBACK_SCORE),
    "response_pattern": (compute_pattern_learning_score, MIN_PATTERN_LEARNING_SCORE),
}

#: Which entry fields feed which score components when the entry
#: carries no explicit ``components`` dict (per formula key).
_COMPONENT_FIELD_FALLBACKS: dict[str, str] = {
    "confidence": "confidence",
    "importance": "importance",
    "final_response_quality": "quality",
}

#: The WritebackRecord legs this module applies, in order.
WRITEBACK_LEGS: tuple[str, ...] = (
    "new_preferences",
    "new_facts",
    "new_episodes",
    "new_patterns",
    "thread_updates",
    "records_to_reinforce",
    "records_to_archive",
    "records_to_delete",
)

#: Thread-update keys an update may set on the standing record.
THREAD_UPDATE_KEYS: tuple[str, ...] = (
    "topic", "summary", "tags", "open_loops", "priority",
)


@dataclass
class WritebackOutcome:
    """What one apply_writeback call did, as data.

    ``applied`` rows: {"kind", "record_id", "score"} — score is
    the writeback score that cleared the gate, or None for the
    kinds with no formula (facts, thread state, reinforce /
    archive / delete). ``refused`` rows: {"kind", "label",
    "reasons"} — every item that was asked for and not written,
    with every reason. Nothing requested is absent from both
    lists: silence is not an outcome this leg produces.
    """

    turn_id: str
    enforcement_gated: bool = False
    gate_reason: str = ""
    applied: list[dict[str, Any]] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)

    @property
    def applied_ids(self) -> list[str]:
        return [row["record_id"] for row in self.applied]

    @property
    def written(self) -> bool:
        """True when anything at all reached the store."""
        return bool(self.applied)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ── Entry plumbing ───────────────────────────────────────────────────────────


def _entry_source(entry: dict[str, Any], problems: list[str]) -> str:
    """The stamp for one entry: its own source when valid, else
    the writeback default. An out-of-vocabulary source is a
    problem (refusal), never silently replaced."""
    source = entry.get("source")
    if source is None:
        return WRITEBACK_SOURCE
    if source not in MEMORY_PROVENANCE:
        problems.append(
            f"source {source!r} outside the memory provenance "
            f"vocabulary {MEMORY_PROVENANCE}"
        )
        return ""
    return str(source)


def _components_for(
    kind: str, entry: dict[str, Any], problems: list[str]
) -> WritebackComponents | None:
    """Resolve the score components for a gated kind.

    Explicit ``components`` win; a missing key falls back to the
    entry's matching field, then to NEUTRAL_COMPONENT. A
    component that is not a number in [0, 1] is a refusal."""
    # The formula's weight table names its components exactly.
    from thoughtforge.knowledge import scoring as scoring_module

    weights = {
        "user_preference": scoring_module.PREFERENCE_WRITEBACK_WEIGHTS,
        "episode": scoring_module.EPISODE_WRITEBACK_WEIGHTS,
        "response_pattern": scoring_module.PATTERN_LEARNING_WEIGHTS,
    }[kind]
    keys = list(weights)
    explicit = entry.get("components") or {}
    if not isinstance(explicit, dict):
        problems.append(f"components is not a dict: {type(explicit).__name__}")
        return None
    values: dict[str, float] = {}
    for key in keys:
        if key in explicit:
            raw = explicit[key]
        elif _COMPONENT_FIELD_FALLBACKS.get(key) and _COMPONENT_FIELD_FALLBACKS[key] in entry:
            raw = entry[_COMPONENT_FIELD_FALLBACKS[key]]
        else:
            raw = NEUTRAL_COMPONENT
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not 0.0 <= float(raw) <= 1.0:
            problems.append(f"component {key} is not a number in [0, 1]: {raw!r}")
            return None
        values[key] = float(raw)
    return WritebackComponents(**values)


def _score_gate(
    kind: str, entry: dict[str, Any], problems: list[str]
) -> float | None:
    """The entry's writeback score, or None with the refusal
    reason appended (bad components, or a score below the gate)."""
    formula, threshold = SCORE_GATES[kind]
    components = _components_for(kind, entry, problems)
    if components is None:
        return None
    score = float(formula(components))
    if score < threshold:
        problems.append(
            f"writeback score {score:.3f} below the {kind} gate {threshold}"
        )
        return None
    return score


def _refuse(
    outcome: WritebackOutcome, kind: str, label: str, reasons: list[str]
) -> None:
    outcome.refused.append({"kind": kind, "label": label, "reasons": list(reasons)})


def _apply(outcome: WritebackOutcome, kind: str, record_id: str, score: float | None) -> None:
    outcome.applied.append({"kind": kind, "record_id": record_id, "score": score})


# ── The new-record legs ──────────────────────────────────────────────────────


def _write_fact(
    store: MemoryStore, entry: Any, outcome: WritebackOutcome
) -> None:
    if not isinstance(entry, dict):
        _refuse(outcome, "user_fact", repr(entry)[:60], ["fact entry is not a dict"])
        return
    entry = dict(entry)
    entry["source"] = entry.get("source", WRITEBACK_SOURCE)
    problems = writeback_fact_problems(entry)
    if problems:
        _refuse(outcome, "user_fact", str(entry.get("summary", ""))[:60], problems)
        return
    record = UserFactRecord(
        record_id=store.new_fact_id(),
        tags=list(entry.get("tags") or []),
        summary=str(entry.get("summary", "")),
        fact_value=dict(entry.get("fact_value") or {}),
        importance=float(entry.get("importance", 0.5)),
        confidence=float(entry.get("confidence", 0.6)),
        source=str(entry["source"]),
    )
    try:
        store.append_profile_record(record)
    except ValueError as exc:
        _refuse(outcome, "user_fact", record.summary[:60], [str(exc)])
        return
    _apply(outcome, "user_fact", record.record_id, None)


def _write_preference(
    store: MemoryStore, entry: Any, outcome: WritebackOutcome
) -> None:
    if not isinstance(entry, dict):
        _refuse(outcome, "user_preference", repr(entry)[:60], ["preference entry is not a dict"])
        return
    entry = dict(entry)
    # The record's construction defaults, filled before scoring so
    # a score component falling back to an entry field sees the
    # same value the record will carry.
    entry.setdefault("confidence", 0.5)
    entry.setdefault("importance", 0.5)
    problems: list[str] = []
    source = _entry_source(entry, problems)
    score = _score_gate("user_preference", entry, problems) if not problems else None
    if problems:
        _refuse(outcome, "user_preference", str(entry.get("summary", ""))[:60], problems)
        return
    record = UserPreferenceRecord(
        record_id=store.new_preference_id(),
        category=str(entry.get("category", "")),
        tags=list(entry.get("tags") or []),
        summary=str(entry.get("summary", "")),
        value=str(entry.get("value", "")),
        importance=float(entry.get("importance", 0.5)),
        confidence=float(entry.get("confidence", 0.5)),
        source=source,
    )
    problems = record_problems(record)
    if problems:
        _refuse(outcome, "user_preference", record.summary[:60], problems)
        return
    try:
        store.append_profile_record(record)
    except ValueError as exc:
        _refuse(outcome, "user_preference", record.summary[:60], [str(exc)])
        return
    _apply(outcome, "user_preference", record.record_id, score)


def _write_episode(
    store: MemoryStore, entry: Any, outcome: WritebackOutcome
) -> None:
    if not isinstance(entry, dict):
        _refuse(outcome, "episode", repr(entry)[:60], ["episode entry is not a dict"])
        return
    entry = dict(entry)
    entry.setdefault("confidence", 0.5)
    entry.setdefault("importance", 0.4)  # EpisodicMemoryRecord's default
    entry.setdefault("quality", 0.5)
    problems: list[str] = []
    source = _entry_source(entry, problems)
    score = _score_gate("episode", entry, problems) if not problems else None
    if problems:
        _refuse(outcome, "episode", str(entry.get("summary", ""))[:60], problems)
        return
    record = EpisodicMemoryRecord(
        record_id=store.new_episode_id(),
        tags=list(entry.get("tags") or []),
        summary=str(entry.get("summary", "")),
        tone=str(entry.get("tone", "")),
        importance=float(entry.get("importance", 0.4)),
        confidence=float(entry.get("confidence", 0.5)),
        quality=float(entry.get("quality", 0.5)),
        linked_preferences=list(entry.get("linked_preferences") or []),
        linked_facts=list(entry.get("linked_facts") or []),
        source=source,
    )
    problems = record_problems(record)
    if problems:
        _refuse(outcome, "episode", record.summary[:60], problems)
        return
    try:
        store.append_episode(record)
    except ValueError as exc:
        _refuse(outcome, "episode", record.summary[:60], [str(exc)])
        return
    _apply(outcome, "episode", record.record_id, score)


def _write_pattern(
    store: MemoryStore, entry: Any, outcome: WritebackOutcome
) -> None:
    if not isinstance(entry, dict):
        _refuse(outcome, "response_pattern", repr(entry)[:60], ["pattern entry is not a dict"])
        return
    entry = dict(entry)
    entry.setdefault("quality", 0.6)  # ResponsePatternRecord's default
    problems: list[str] = []
    source = _entry_source(entry, problems)
    score = _score_gate("response_pattern", entry, problems) if not problems else None
    if problems:
        _refuse(outcome, "response_pattern", str(entry.get("summary", ""))[:60], problems)
        return
    record = ResponsePatternRecord(
        record_id=store.new_pattern_id(),
        tags=list(entry.get("tags") or []),
        summary=str(entry.get("summary", "")),
        pattern_shape=list(entry.get("pattern_shape") or []),
        quality=float(entry.get("quality", 0.6)),
        importance=float(entry.get("importance", 0.5)),
        confidence=float(entry.get("confidence", 0.6)),
        source=source,
    )
    problems = record_problems(record)
    if problems:
        _refuse(outcome, "response_pattern", record.summary[:60], problems)
        return
    try:
        store.append_pattern(record)
    except ValueError as exc:
        _refuse(outcome, "response_pattern", record.summary[:60], [str(exc)])
        return
    _apply(outcome, "response_pattern", record.record_id, score)


# ── Reinforce / archive / delete ─────────────────────────────────────────────


def _find_record(
    store: MemoryStore, record_id: str
) -> tuple[str, list[Any], Any] | None:
    """Resolve an id across the append stores, in file order:
    profile (facts + preferences), episodes, patterns. Returns
    (home, the loaded list the record rides in, the record) —
    the caller mutates the record and rewrites THAT list (a
    rewrite re-read from disk would carry the unmutated rows).
    The personality core and thread state are NOT searched —
    writeback cannot touch identity or the working record
    through these lists."""
    homes = (
        ("profile", store.load_profile_records()),
        ("episodes", store.load_episodes()),
        ("patterns", store.load_patterns()),
    )
    for home, records in homes:
        for record in records:
            if record.record_id == record_id:
                return (home, records, record)
    return None


def _rewrite_for(store: MemoryStore, home: str, records: list[Any]) -> None:
    if home == "profile":
        store.rewrite_profile_store(records)
    elif home == "episodes":
        store.rewrite_episodic_store(records)
    else:
        store.rewrite_patterns_store(records)


def _mutate_records(
    store: MemoryStore,
    outcome: WritebackOutcome,
    leg: str,
    record_ids: list[str],
    *,
    action: str,
) -> None:
    for record_id in record_ids:
        found = _find_record(store, str(record_id))
        if found is None:
            _refuse(
                outcome, leg, str(record_id),
                [f"record {record_id} not found in any writeback store"],
            )
            continue
        home, records, record = found
        if action == "reinforce":
            record.times_reinforced += 1
            record.status = "reinforced"
        else:
            record.status = "archived" if action == "archive" else "deleted"
        record.updated_at = _now_iso()
        _rewrite_for(store, home, records)
        _apply(outcome, leg, record.record_id, None)


# ── Thread updates ───────────────────────────────────────────────────────────


def _apply_thread_updates(
    store: MemoryStore, updates: dict[str, Any], outcome: WritebackOutcome
) -> None:
    if not updates:
        return
    unknown = [k for k in updates if k not in THREAD_UPDATE_KEYS]
    if unknown:
        _refuse(
            outcome, "thread_state", "thread_updates",
            [f"unknown thread update keys: {sorted(unknown)}"],
        )
        return
    existing = store.load_thread_state()
    if existing is None:
        if not str(updates.get("topic", "")).strip():
            _refuse(
                outcome, "thread_state", "thread_updates",
                ["no standing thread state and the update names no topic to create one"],
            )
            return
        record = ActiveThreadStateRecord(
            record_id=store.new_thread_id(),
            topic=str(updates.get("topic", "")),
            tags=list(updates.get("tags") or []),
            summary=str(updates.get("summary", "")),
            open_loops=list(updates.get("open_loops") or []),
        )
        if "priority" in updates:
            record.priority = float(updates["priority"])
    else:
        record = existing
        for key in THREAD_UPDATE_KEYS:
            if key in updates:
                value = updates[key]
                setattr(record, key, list(value) if key in ("tags", "open_loops") else value)
        record.turns_since_touch = 0
    try:
        store.save_thread_state(record)
    except ValueError as exc:
        _refuse(outcome, "thread_state", record.record_id, [str(exc)])
        return
    _apply(outcome, "thread_state", record.record_id, None)


# ── The leg ──────────────────────────────────────────────────────────────────


def _requested_items(writeback: WritebackRecord) -> list[tuple[str, str]]:
    """Every item a writeback asks for, as (leg, label) — used to
    refuse them all by name when the enforcement gate closes."""
    items: list[tuple[str, str]] = []
    for leg, entries in (
        ("new_preferences", writeback.new_preferences),
        ("new_facts", writeback.new_facts),
        ("new_episodes", writeback.new_episodes),
        ("new_patterns", writeback.new_patterns),
    ):
        for entry in entries:
            label = ""
            if isinstance(entry, dict):
                label = str(entry.get("summary", ""))[:60]
            items.append((leg, label or repr(entry)[:60]))
    if writeback.thread_updates:
        items.append(("thread_updates", "thread_updates"))
    for leg, ids in (
        ("records_to_reinforce", writeback.records_to_reinforce),
        ("records_to_archive", writeback.records_to_archive),
        ("records_to_delete", writeback.records_to_delete),
    ):
        for record_id in ids:
            items.append((leg, str(record_id)))
    return items


def apply_writeback(
    store: MemoryStore,
    writeback: WritebackRecord,
    *,
    enforcement_passed: bool,
) -> WritebackOutcome:
    """
    Apply one turn's writeback to the store, through every gate.

    ``enforcement_passed`` is the turn's final-response verdict
    from the EnforcementGate (Part V): False closes the leg
    entirely — nothing is written and every requested item is
    refused with the gate reason. True opens the per-entry
    contract and score gates. The outcome records both halves;
    this function raises only on a malformed writeback object
    itself, never on a bad entry.
    """
    outcome = WritebackOutcome(turn_id=writeback.turn_id)
    if not enforcement_passed:
        outcome.enforcement_gated = True
        outcome.gate_reason = (
            "the turn's final response did not pass enforcement — "
            "a failed answer writes nothing to memory"
        )
        for leg, label in _requested_items(writeback):
            _refuse(outcome, leg, label, [outcome.gate_reason])
        return outcome

    for entry in writeback.new_preferences:
        _write_preference(store, entry, outcome)
    for entry in writeback.new_facts:
        _write_fact(store, entry, outcome)
    for entry in writeback.new_episodes:
        _write_episode(store, entry, outcome)
    for entry in writeback.new_patterns:
        _write_pattern(store, entry, outcome)
    _apply_thread_updates(store, writeback.thread_updates, outcome)
    _mutate_records(
        store, outcome, "records_to_reinforce",
        list(writeback.records_to_reinforce), action="reinforce",
    )
    _mutate_records(
        store, outcome, "records_to_archive",
        list(writeback.records_to_archive), action="archive",
    )
    _mutate_records(
        store, outcome, "records_to_delete",
        list(writeback.records_to_delete), action="delete",
    )
    return outcome


# ── Self-check ───────────────────────────────────────────────────────────────


def writeback_problems() -> list[str]:
    """
    Re-check this slice's claims against the live tree. Returns
    problems by name; ``[]`` when the leg still matches the code —
    the gates are scoring.py's own constants and formulas, the
    contract's provenance split holds, the four writeback kinds
    carry the stamp, and the core wiring is in place.
    """
    from thoughtforge.cognition import core as core_module
    from thoughtforge.knowledge import scoring as scoring_module

    problems: list[str] = []

    # The gates are the named constants, not copies.
    expected = {
        "user_preference": (
            scoring_module.compute_preference_writeback_score,
            scoring_module.MIN_PREFERENCE_WRITEBACK_SCORE,
        ),
        "episode": (
            scoring_module.compute_episode_writeback_score,
            scoring_module.MIN_EPISODE_WRITEBACK_SCORE,
        ),
        "response_pattern": (
            scoring_module.compute_pattern_learning_score,
            scoring_module.MIN_PATTERN_LEARNING_SCORE,
        ),
    }
    if SCORE_GATES != expected:
        problems.append("score gates drifted from scoring.py's named formulas/thresholds")
    if WRITEBACK_SOURCE not in MEMORY_PROVENANCE:
        problems.append("WRITEBACK_SOURCE is outside MEMORY_PROVENANCE")

    # The stamp split: the four writeback kinds carry `source`;
    # identity and thread state do not.
    stamped = {"user_preference", "user_fact", "episode", "response_pattern"}
    from thoughtforge.knowledge import models as models_module

    for row in FACT_KINDS:
        if row["store"] != "memory":
            continue
        record_cls = getattr(models_module, row["record"])
        field_names = {f.name for f in dataclasses.fields(record_cls)}
        if row["kind"] in stamped and "source" not in field_names:
            problems.append(f"{row['record']} lost its TF-047 source stamp field")
        if row["kind"] not in stamped and "source" in field_names:
            problems.append(f"{row['record']} gained a source field outside the writeback kinds")

    # The legs cover WritebackRecord's slots exactly.
    wb_fields = {f.name for f in dataclasses.fields(WritebackRecord)} - {"turn_id"}
    if wb_fields != set(WRITEBACK_LEGS):
        problems.append(
            f"WritebackRecord slots {sorted(wb_fields)} != the legs this module applies {sorted(WRITEBACK_LEGS)}"
        )

    # Core wiring: _finish_turn applies a present writeback via
    # the core's apply_writeback, gated on the final's verdict.
    finish_src = inspect.getsource(core_module.ThoughtForgeCore._finish_turn)
    if "turn.writeback" not in finish_src or "apply_writeback" not in finish_src:
        problems.append("core wiring: _finish_turn no longer applies turn.writeback")
    if "enforcement_passed=final.enforcement_passed" not in finish_src:
        problems.append("core wiring: _finish_turn no longer gates writeback on the final's enforcement verdict")
    if not hasattr(core_module.ThoughtForgeCore, "apply_writeback"):
        problems.append("core wiring: ThoughtForgeCore.apply_writeback missing")
    if not isinstance(
        getattr(core_module.ThoughtForgeCore, "last_writeback_outcome", None), property
    ):
        problems.append("core wiring: ThoughtForgeCore.last_writeback_outcome is not a property")

    return problems


__all__ = [
    "NEUTRAL_COMPONENT",
    "SCORE_GATES",
    "THREAD_UPDATE_KEYS",
    "WRITEBACK_LEGS",
    "WRITEBACK_SOURCE",
    "WritebackOutcome",
    "apply_writeback",
    "writeback_problems",
]
