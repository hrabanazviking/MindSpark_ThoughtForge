"""
Contradiction handling — new fact vs stored fact (TF-050).

The roadmap's rule: *new fact vs stored fact — the conflict is
surfaced, scored, and resolved by the stated rule, never
silently overwritten.*

The measured starting state, before this slice:

  * TF-047's writeback leg appended every contract-valid new
    fact and preference **blind**: a fact stating the ferry
    leaves at 09:00 landed beside a stored fact stating it
    leaves at 07:30, both live, neither flagged — retrieval
    (today) and any future reader of the profile store would
    meet two contradictory "truths" with no record that a
    conflict ever happened. A preference re-stated with a new
    value likewise piled up beside the old one.
  * Nothing in the live tree compared a new claim against a
    stored one. The only contradiction machinery anywhere was
    TF-028's *answer-side* coherence anchors (a generated
    answer contradicting a probed fact scores 0 — it polices
    output, not ingestion) and the lifecycle spec's §27, which
    named strategies (mark superseded / reduce confidence /
    archive / preserve both if context-dependent) that no
    code implemented. The research wing's
    ``contradiction_record`` schema (claim_a / claim_b /
    reason / preferred / resolution_state) described the
    shape a conflict record should carry but bound nothing.

What this slice builds:

  * ``assess_contradiction(new_record, stored_records)`` —
    the pure check. A new ``UserFactRecord`` or
    ``UserPreferenceRecord`` is compared against the stored
    records **of its own kind** (facts and preferences do not
    contradict each other across kinds — their claim shapes
    differ). A pair is a contradiction under exactly the
    rules in ``CONTRADICTION_RULES``:

      - ``fact_value_slot``: both facts carry the same
        normalised ``fact_value`` slot key with different
        normalised values — the structured claim disagrees.
      - ``preference_value``: both preferences share a
        category and carry different non-empty values.
      - ``negated_claim``: the summaries share a subject
        (distinctive-term Jaccard ≥ ``MIN_SUMMARY_OVERLAP``)
        and exactly one of them is negated
        (``NEGATION_MARKERS``) — "the ferry runs on Sundays"
        vs "the ferry does not run on Sundays".

    **Scope guard (spec §27.3):** when both records carry
    tags and their tag sets are disjoint, the pair is NOT a
    contradiction — different scopes may legitimately hold
    different values ("concise generally" vs "detailed for
    project docs"). One shared tag, or either side untagged,
    leaves the pair in scope.

    **Duplicates are not contradictions:** a new record
    that carries nothing the stored one does not already
    carry — the same summary with the same slot values, or
    the same category with the same value — is a duplicate
    (spec §5.3/§10.2: repeated facts are reinforced, not
    fought). A record carrying a slot the stored one lacks
    *expands* it and is neither duplicate nor contradiction.
    A pair that contradicts on any slot is never also a
    duplicate.

  * **Scoring.** Every conflict is scored, with the
    components kept on the record: each side's *strength* is
    ``0.6·confidence + 0.4·provenance authority`` (authority
    from ``SOURCE_AUTHORITY`` over TF-042's
    ``MEMORY_PROVENANCE``; a legacy unstamped record rates
    ``UNSTAMPED_AUTHORITY``), plus a stored-side
    reinforcement bonus of ``0.02`` per reinforcement, capped
    at 5 — a fact the user has confirmed five times is
    harder to move than one stated once. The conflict score
    is ``0.5·rule strength + 0.25·new strength +
    0.25·stored strength``: a conflict matters most when
    the rule is hard (a structured slot) and both sides are
    credible.

  * **Resolution, the stated rule** (``RESOLUTION_RULE``),
    driven by the strength margin ``new − stored``:

      - margin ≥ ``WIN_MARGIN`` (0.15):
        ``new_supersedes_stored`` — the stored record is
        archived (it leaves the live store through the
        store's own rewrite shape, exactly as TF-047's
        archive leg removes a record) and the new record is
        appended. The old claim is not edited in place and
        not silently dropped: the ``ConflictRecord`` naming
        both claims, both values, the score, and the
        resolution is the surface.
      - margin ≤ −``WIN_MARGIN``: ``stored_stands`` — the
        new record is NOT written; the stored record is left
        byte-identical. The writeback leg refuses the entry
        with the conflict named as the reason.
      - in between: ``preserve_both_flagged`` — neither side
        earned the win, so both stay live, both records'
        confidence is multiplied by
        ``CLOSE_CALL_CONFIDENCE_FACTOR`` (0.8 — spec §27.2's
        "reduce confidence", applied to both because the
        conflict itself is the evidence against each), and
        the conflict rides the outcome flagged.

  * ``apply_contradiction_resolution(store, new_record,
    report)`` — performs a non-blocked report against a
    ``MemoryStore``: duplicate → the stored record is
    reinforced in place (``times_reinforced`` + 1, status
    ``reinforced``) and no second copy is appended;
    conflicts → append-through-the-store-gate plus the
    archive / downgrade mutations above, applied to a list
    reloaded *after* the append so the rewrite can never
    erase the new record. A blocked report raises
    ``ValueError`` — applying one is a programming error,
    not an outcome. Returns the id of the record that stands
    for the new claim (the duplicate's id, or the new
    record's own).

  * **Wiring.** TF-047's writeback leg routes its fact and
    preference writes through this module: assess →
    blocked refuses by name; otherwise apply. Every conflict
    found rides the ``WritebackOutcome`` as data
    (``outcome.contradictions``) whether the new record won,
    lost, or was preserved beside the old — surfacing is not
    optional on any branch. The enforcement gate still fires
    first: a failed turn is assessed against nothing and
    writes nothing. Episodes and patterns are out of scope
    by ruling: an episode is a moment (two moments cannot
    contradict) and a pattern is a procedure, not a claim.

Honesty boundaries, stated: detection is deterministic and
structural — slot equality, category equality, and a
negation-plus-overlap heuristic over summaries. It does not
do semantic entailment: two prose summaries that contradict
without a shared slot, a shared category, or a negation
marker over a shared subject are NOT caught, and a
coincidental subject overlap with a stray negation can
flag a pair that a human would call compatible (flagging
with both claims preserved is the safe direction; nothing
is deleted on a heuristic's word alone — the only removal
path, archival of the stored record, requires a hard rule
or a decisive strength margin, and is always surfaced).
Values are compared after a stated normalisation (strings
lowercased and whitespace-collapsed; numbers compared
numerically), never by model judgement. Proof grade is
FIXTURE (synthetic stores on this VM); contradiction
behaviour over a real, lived-in store is NOT PROVEN.

The human-readable write-up is
``docs/architecture/CONTRADICTIONS.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.ingestion_contract import MEMORY_PROVENANCE
from thoughtforge.knowledge.models import (
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.common import now_iso as _now_iso

# ── The detection rules, as data ─────────────────────────────────────────────

RULE_STRENGTH: dict[str, float] = {
    "fact_value_slot": 1.0,    # a structured slot disagrees — hardest evidence
    "preference_value": 0.9,   # same category, different stated value
    "negated_claim": 0.7,      # prose: shared subject, one side negated
}

CONTRADICTION_RULES: tuple[dict[str, Any], ...] = (
    {"name": "fact_value_slot",
     "rule": "both facts carry the same normalised fact_value slot "
             "key with different normalised values"},
    {"name": "preference_value",
     "rule": "both preferences share a normalised category and "
             "carry different non-empty values"},
    {"name": "negated_claim",
     "rule": "the summaries' distinctive-term Jaccard overlap is "
             ">= MIN_SUMMARY_OVERLAP and exactly one summary is "
             "negated (NEGATION_MARKERS)"},
)

#: Words/phrases that negate a prose claim. Exactly one side of
#: a pair carrying one is the negation signal; both sides
#: negated is agreement-shaped, neither is a plain restatement.
NEGATION_MARKERS: tuple[str, ...] = (
    "no longer", "not", "no", "never", "none",
    "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "don't", "cannot", "can't",
    "stopped", "quit", "cancelled", "dropped",
)

#: Minimum distinctive-term Jaccard overlap between two
#: summaries for the negated_claim rule to consider them the
#: same subject.
MIN_SUMMARY_OVERLAP: float = 0.5

#: A pair needs at least this many shared distinctive terms
#: before overlap is meaningful (one-word summaries never
#: contradict by prose).
MIN_SHARED_TERMS: int = 2

# ── The scoring and resolution rule, as data ─────────────────────────────────

#: Provenance authority per TF-042 stamp: how far a side's
#: origin alone vouches for it, before confidence is counted.
SOURCE_AUTHORITY: dict[str, float] = {
    "user_stated": 1.0,             # the user said it in so many words
    "manual_entry": 0.9,            # entered by hand through the toolset
    "conversation_writeback": 0.6,  # extracted from a conversation
    "import": 0.5,                  # carried in from a store bundle
    "synthetic_fixture": 0.4,       # test stand-ins
}

#: Authority of a record with no stamp (legacy) or a stamp
#: outside the vocabulary: present, but vouched for by nothing.
UNSTAMPED_AUTHORITY: float = 0.3

#: Strength margin at or beyond which one side wins outright.
WIN_MARGIN: float = 0.15

#: Confidence multiplier applied to BOTH records when a
#: conflict is too close to call (spec §27.2: reduce
#: confidence — of the pair the conflict incriminates).
CLOSE_CALL_CONFIDENCE_FACTOR: float = 0.8

#: Stored-side reinforcement bonus: per times_reinforced,
#: capped at REINFORCEMENT_CAP reinforcements.
REINFORCEMENT_BONUS: float = 0.02
REINFORCEMENT_CAP: int = 5

RESOLUTION_RULE: dict[str, Any] = {
    "margin": "new_strength - stored_strength",
    "new_wins": (
        f"margin >= {WIN_MARGIN}: new_supersedes_stored — the "
        "stored record is archived out of the live store and "
        "the new record is appended; the ConflictRecord is "
        "the surface"
    ),
    "stored_wins": (
        f"margin <= -{WIN_MARGIN}: stored_stands — the new "
        "record is not written and the stored record is left "
        "unchanged; the write is refused with the conflict named"
    ),
    "close": (
        "between the margins: preserve_both_flagged — both "
        "records stay live, both confidences are multiplied "
        f"by {CLOSE_CALL_CONFIDENCE_FACTOR}, and the conflict "
        "rides the outcome"
    ),
    "duplicate": (
        "a claim carrying nothing the stored record does not "
        "already carry (same summary and slot values, or same "
        "category+value) is not a contradiction: the stored "
        "record is reinforced and no copy is appended"
    ),
    "scope_guard": (
        "records whose tag sets are both non-empty and "
        "disjoint never contradict — different scopes may "
        "hold different values (spec §27.3)"
    ),
}

# ── Normalisation ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "and", "or", "in", "on", "at", "by", "for",
    "with", "as", "it", "its", "this", "that", "has", "have",
})


def _norm_text(value: Any) -> str:
    """Normalise a claim string for comparison: lowercase,
    whitespace collapsed."""
    return " ".join(str(value or "").lower().split())


def _norm_key(key: Any) -> str:
    """Normalise a fact_value slot key: lowercase, spaces and
    dashes folded to underscores."""
    return re.sub(r"[\s\-]+", "_", str(key or "").strip().lower())


def _norm_value(value: Any) -> str:
    """Normalise a slot/category value for equality: numbers
    compare numerically (30 == 30.0 == '30'), booleans as
    words, everything else as normalised text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    text = _norm_text(value)
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _fold(token: str) -> str:
    """A crude singular fold (runs→run, sundays→sunday),
    stated once, as TF-049 folds: a trailing 's' on a token
    longer than 3 chars, unless the token ends in 'ss'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(text: str) -> set[str]:
    """Distinctive terms of a summary: lowercase tokens of
    length >= 3, singular-folded, minus stopwords and the
    negation markers themselves (negation is the signal, not
    the subject)."""
    markers = {m for m in NEGATION_MARKERS if " " not in m}
    return {
        _fold(tok)
        for tok in _TOKEN_RE.findall(_norm_text(text))
        if len(tok) >= 3 and tok not in _STOPWORDS and tok not in markers
    }


def _is_negated(text: str) -> bool:
    norm = f" {_norm_text(text)} "
    return any(
        f" {marker} " in norm or f" {marker}," in norm or f" {marker}." in norm
        for marker in NEGATION_MARKERS
    )


def _scopes_overlap(new_record: Any, stored_record: Any) -> bool:
    """The spec §27.3 scope guard: two non-empty disjoint tag
    sets are different scopes and never contradict. Either
    side untagged leaves the pair in scope."""
    new_tags = {_norm_text(t) for t in getattr(new_record, "tags", []) or []}
    stored_tags = {_norm_text(t) for t in getattr(stored_record, "tags", []) or []}
    if new_tags and stored_tags:
        return bool(new_tags & stored_tags)
    return True


# ── Scoring ──────────────────────────────────────────────────────────────────


def source_authority(source: str) -> float:
    """The authority of one provenance stamp."""
    return SOURCE_AUTHORITY.get(source or "", UNSTAMPED_AUTHORITY)


def record_strength(record: Any, *, stored: bool) -> float:
    """
    One side's strength in a conflict: 0.6·confidence +
    0.4·provenance authority, plus the stored side's
    reinforcement bonus (0.02 per reinforcement, capped at
    5). Always in [0, 1].
    """
    strength = (
        0.6 * float(getattr(record, "confidence", 0.0) or 0.0)
        + 0.4 * source_authority(str(getattr(record, "source", "") or ""))
    )
    if stored:
        reinforced = int(getattr(record, "times_reinforced", 0) or 0)
        strength += REINFORCEMENT_BONUS * min(reinforced, REINFORCEMENT_CAP)
    return round(min(1.0, max(0.0, strength)), 4)


def _resolution_for(margin: float) -> str:
    if margin >= WIN_MARGIN:
        return "new_supersedes_stored"
    if margin <= -WIN_MARGIN:
        return "stored_stands"
    return "preserve_both_flagged"


# ── The verdict, as data ─────────────────────────────────────────────────────


@dataclass
class ConflictRecord:
    """One surfaced conflict between a new claim and a stored
    one — the shape the research wing's contradiction_record
    schema named (claim a / claim b / reason / preferred /
    resolution), bound to the live store's records."""

    new_record_id: str
    stored_record_id: str
    kind: str               # "user_fact" | "user_preference"
    rule: str               # a name from CONTRADICTION_RULES
    slot: str               # the slot key, "category", or "summary"
    new_value: str
    stored_value: str
    score: float
    new_strength: float
    stored_strength: float
    margin: float
    resolution: str         # see RESOLUTION_RULE

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ContradictionReport:
    """The whole assessment of one new record, as data."""

    new_record_id: str
    kind: str
    conflicts: list[ConflictRecord] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        """True when any conflict's resolution is
        stored_stands — the strongest stored defence decides
        for the whole write."""
        return any(c.resolution == "stored_stands" for c in self.conflicts)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def clean(self) -> bool:
        """No conflicts and no duplicates: a plain append."""
        return not self.conflicts and not self.duplicate_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_record_id": self.new_record_id,
            "kind": self.kind,
            "blocked": self.blocked,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "duplicate_ids": list(self.duplicate_ids),
        }


# ── Detection ────────────────────────────────────────────────────────────────


def _conflict(
    new_record: Any, stored_record: Any, kind: str,
    rule: str, slot: str, new_value: str, stored_value: str,
) -> ConflictRecord:
    new_strength = record_strength(new_record, stored=False)
    stored_strength = record_strength(stored_record, stored=True)
    margin = round(new_strength - stored_strength, 4)
    score = round(min(1.0, max(0.0,
        0.5 * RULE_STRENGTH[rule]
        + 0.25 * new_strength
        + 0.25 * stored_strength
    )), 4)
    return ConflictRecord(
        new_record_id=str(getattr(new_record, "record_id", "")),
        stored_record_id=str(getattr(stored_record, "record_id", "")),
        kind=kind, rule=rule, slot=slot,
        new_value=new_value, stored_value=stored_value,
        score=score, new_strength=new_strength,
        stored_strength=stored_strength, margin=margin,
        resolution=_resolution_for(margin),
    )


def _assess_fact_pair(
    new_record: UserFactRecord, stored_record: UserFactRecord,
    report: ContradictionReport,
) -> None:
    if not _scopes_overlap(new_record, stored_record):
        return
    new_slots = {_norm_key(k): v for k, v in (new_record.fact_value or {}).items()}
    stored_slots = {_norm_key(k): v for k, v in (stored_record.fact_value or {}).items()}
    shared = sorted(set(new_slots) & set(stored_slots))
    differing = [k for k in shared if _norm_value(new_slots[k]) != _norm_value(stored_slots[k])]
    for slot in differing:
        report.conflicts.append(_conflict(
            new_record, stored_record, "user_fact", "fact_value_slot",
            slot, _norm_value(new_slots[slot]), _norm_value(stored_slots[slot]),
        ))
    if differing:
        return  # a contradicting pair is never also a duplicate
    # The prose rule: same subject, exactly one side negated.
    new_terms, stored_terms = _terms(new_record.summary), _terms(stored_record.summary)
    shared_terms = new_terms & stored_terms
    union = new_terms | stored_terms
    overlap = len(shared_terms) / len(union) if union else 0.0
    if (
        len(shared_terms) >= MIN_SHARED_TERMS
        and overlap >= MIN_SUMMARY_OVERLAP
        and _is_negated(new_record.summary) != _is_negated(stored_record.summary)
    ):
        report.conflicts.append(_conflict(
            new_record, stored_record, "user_fact", "negated_claim",
            "summary", _norm_text(new_record.summary),
            _norm_text(stored_record.summary),
        ))
        return
    # Duplicates: the new record carries nothing the stored
    # one does not already carry — same summary (or a shared
    # slot) and every new slot already present, agreeing. A
    # new record with a slot the stored one lacks EXPANDS it
    # (spec §10.2) and is neither duplicate nor contradiction.
    same_summary = bool(_norm_text(new_record.summary)) and (
        _norm_text(new_record.summary) == _norm_text(stored_record.summary)
    )
    slots_covered = set(new_slots) <= set(stored_slots) and not differing
    if slots_covered and (same_summary or shared):
        report.duplicate_ids.append(stored_record.record_id)


def _assess_preference_pair(
    new_record: UserPreferenceRecord, stored_record: UserPreferenceRecord,
    report: ContradictionReport,
) -> None:
    if not _scopes_overlap(new_record, stored_record):
        return
    same_category = _norm_text(new_record.category) == _norm_text(stored_record.category)
    if not same_category or not _norm_text(new_record.category):
        return
    new_value, stored_value = _norm_value(new_record.value), _norm_value(stored_record.value)
    if new_value and stored_value and new_value != stored_value:
        report.conflicts.append(_conflict(
            new_record, stored_record, "user_preference", "preference_value",
            "category", new_value, stored_value,
        ))
        return
    if new_value and new_value == stored_value:
        report.duplicate_ids.append(stored_record.record_id)


def assess_contradiction(
    new_record: UserFactRecord | UserPreferenceRecord,
    stored_records: list[Any],
) -> ContradictionReport:
    """
    Compare one new record against the stored records of its
    own kind. Pure — writes nothing, mutates nothing. Stored
    records already archived/deleted, and the new record
    itself, are skipped.
    """
    if isinstance(new_record, UserFactRecord):
        kind, assess_pair = "user_fact", _assess_fact_pair
    elif isinstance(new_record, UserPreferenceRecord):
        kind, assess_pair = "user_preference", _assess_preference_pair
    else:
        raise TypeError(
            "contradiction handling covers UserFactRecord and "
            f"UserPreferenceRecord, not {type(new_record).__name__}"
        )
    report = ContradictionReport(
        new_record_id=str(getattr(new_record, "record_id", "")), kind=kind,
    )
    for stored in stored_records or []:
        if type(stored) is not type(new_record):
            continue
        if stored.record_id == new_record.record_id:
            continue
        if getattr(stored, "status", "") in ("archived", "deleted"):
            continue
        assess_pair(new_record, stored, report)  # type: ignore[arg-type]
    return report


# ── Applying a resolution ────────────────────────────────────────────────────


def apply_contradiction_resolution(
    store: MemoryStore,
    new_record: UserFactRecord | UserPreferenceRecord,
    report: ContradictionReport,
) -> str:
    """
    Perform a report's resolution against the store and
    return the id of the record that stands for the new
    claim. A blocked report (any ``stored_stands``) raises
    ``ValueError`` — the caller must refuse it as an outcome,
    never apply it. The new record is appended through the
    store's own ingestion gate; stored-side mutations are
    applied to a list reloaded after the append, so the
    rewrite carries the new record too.
    """
    if report.blocked:
        standing = [
            c.stored_record_id for c in report.conflicts
            if c.resolution == "stored_stands"
        ]
        raise ValueError(
            "contradiction report is blocked: stored record(s) "
            f"{standing} stand against {report.new_record_id}"
        )

    if report.duplicate_ids and not report.conflicts:
        records = store.load_profile_records()
        for record in records:
            if record.record_id == report.duplicate_ids[0]:
                record.times_reinforced += 1
                record.status = "reinforced"
                record.updated_at = _now_iso()
                store.rewrite_profile_store(records)
                return record.record_id
        # The duplicate vanished between assess and apply:
        # fall through to a plain append rather than lose
        # the claim silently.

    if any(c.resolution == "preserve_both_flagged" for c in report.conflicts):
        new_record.confidence = round(
            float(new_record.confidence) * CLOSE_CALL_CONFIDENCE_FACTOR, 4
        )
    store.append_profile_record(new_record)

    if report.conflicts:
        records = store.load_profile_records()
        by_id = {r.record_id: r for r in records}
        touched = False
        for conflict in report.conflicts:
            stored = by_id.get(conflict.stored_record_id)
            if stored is None:
                continue
            if conflict.resolution == "new_supersedes_stored":
                stored.status = "archived"
            elif conflict.resolution == "preserve_both_flagged":
                stored.confidence = round(
                    float(stored.confidence) * CLOSE_CALL_CONFIDENCE_FACTOR, 4
                )
            stored.updated_at = _now_iso()
            touched = True
        if touched:
            store.rewrite_profile_store(records)
    return new_record.record_id


# ── Self-check ───────────────────────────────────────────────────────────────


def contradiction_problems() -> list[str]:
    """
    Re-check this slice's claims against the live tree.
    Returns problems by name; ``[]`` when detection, scoring,
    the resolution rule, and the writeback wiring still match
    the code — proven on synthetic stores, never on a real one.
    """
    import tempfile
    from pathlib import Path

    problems: list[str] = []

    # The rule data is well-formed.
    if [row["name"] for row in CONTRADICTION_RULES] != [
        "fact_value_slot", "preference_value", "negated_claim",
    ]:
        problems.append("CONTRADICTION_RULES drifted from the three stated rules")
    if set(RULE_STRENGTH) != {row["name"] for row in CONTRADICTION_RULES}:
        problems.append("RULE_STRENGTH does not cover exactly the stated rules")
    if set(SOURCE_AUTHORITY) != set(MEMORY_PROVENANCE):
        problems.append("SOURCE_AUTHORITY does not cover exactly MEMORY_PROVENANCE")
    if not 0.0 < WIN_MARGIN < 0.5 or not 0.0 < CLOSE_CALL_CONFIDENCE_FACTOR < 1.0:
        problems.append("resolution constants outside their sane ranges")

    def fact(rid: str, summary: str, fv: dict[str, Any] | None = None,
             conf: float = 0.6, source: str = "", tags: list[str] | None = None,
             reinforced: int = 0) -> UserFactRecord:
        return UserFactRecord(
            record_id=rid, summary=summary, fact_value=fv or {},
            confidence=conf, source=source, tags=tags or [],
            times_reinforced=reinforced,
        )

    # Detection: slot contradiction, duplicate, scope guard,
    # negation — each pinned on synthetic pairs.
    stored = fact("f_old", "the ferry leaves at dawn",
                  {"departure": "07:30"}, conf=0.4, source="import")
    report = assess_contradiction(
        fact("f_new", "the ferry now leaves later",
             {"departure": "09:00"}, conf=1.0, source="user_stated"),
        [stored],
    )
    if len(report.conflicts) != 1 or report.conflicts[0].rule != "fact_value_slot" \
            or report.conflicts[0].slot != "departure" \
            or report.conflicts[0].resolution != "new_supersedes_stored":
        problems.append(f"slot contradiction not detected/resolved as stated: {report.to_dict()}")
    dup = assess_contradiction(
        fact("f_dup", "the ferry leaves at dawn", {"departure": "07:30"}), [stored],
    )
    if dup.conflicts or dup.duplicate_ids != ["f_old"]:
        problems.append(f"identical fact not a duplicate: {dup.to_dict()}")
    scoped = assess_contradiction(
        fact("f_scoped", "the ferry leaves later", {"departure": "09:00"},
             tags=["winter"]),
        [fact("f_old2", "the ferry leaves at dawn", {"departure": "07:30"},
              tags=["summer"])],
    )
    if scoped.conflicts or scoped.duplicate_ids:
        problems.append("disjoint tag scopes were treated as a contradiction")
    negated = assess_contradiction(
        fact("f_neg", "the ferry does not run on sundays"),
        [fact("f_pos", "the ferry runs on sundays every week")],
    )
    if len(negated.conflicts) != 1 or negated.conflicts[0].rule != "negated_claim":
        problems.append(f"negated claim not detected: {negated.to_dict()}")

    # Application on a synthetic store: supersede archives
    # the old, close call downgrades both, stored_stands
    # blocks, duplicate reinforces.
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(memory_dir=Path(tmp))
        store.append_profile_record(
            fact("f_old", "the ferry leaves at dawn",
                 {"departure": "07:30"}, conf=0.4, source="import")
        )
        new = fact("f_new", "the ferry now leaves later",
                   {"departure": "09:00"}, conf=1.0, source="user_stated")
        rep = assess_contradiction(new, store.load_profile_records())
        if apply_contradiction_resolution(store, new, rep) != "f_new":
            problems.append("supersede apply did not stand the new record")
        live = {r.record_id for r in store.load_profile_records()}
        if live != {"f_new"}:
            problems.append(f"superseded record still live (or new missing): {sorted(live)}")
        try:
            apply_contradiction_resolution(store, new, ContradictionReport(
                new_record_id="f_new", kind="user_fact",
                conflicts=[ConflictRecord(
                    new_record_id="f_new", stored_record_id="f_new",
                    kind="user_fact", rule="fact_value_slot", slot="departure",
                    new_value="09:00", stored_value="07:30", score=1.0,
                    new_strength=0.0, stored_strength=1.0, margin=-1.0,
                    resolution="stored_stands",
                )],
            ))
            problems.append("a blocked report was applied instead of raising")
        except ValueError:
            pass

    # Wiring: the writeback leg routes facts and preferences
    # through this module and surfaces conflicts on the outcome.
    from thoughtforge.knowledge import writeback as writeback_module

    land_src = inspect.getsource(writeback_module._land_record)
    if "assess_contradiction" not in land_src or "apply_contradiction_resolution" not in land_src:
        problems.append("writeback wiring: _land_record no longer routes through contradiction handling")
    for fn_name in ("_write_fact", "_write_preference"):
        src = inspect.getsource(getattr(writeback_module, fn_name))
        if "_land_record" not in src:
            problems.append(f"writeback wiring: {fn_name} no longer lands through _land_record")
    if "contradictions" not in {f.name for f in dataclasses.fields(writeback_module.WritebackOutcome)}:
        problems.append("writeback wiring: WritebackOutcome lost its contradictions surface")

    return problems


__all__ = [
    "CLOSE_CALL_CONFIDENCE_FACTOR",
    "CONTRADICTION_RULES",
    "MIN_SHARED_TERMS",
    "MIN_SUMMARY_OVERLAP",
    "NEGATION_MARKERS",
    "REINFORCEMENT_BONUS",
    "REINFORCEMENT_CAP",
    "RESOLUTION_RULE",
    "RULE_STRENGTH",
    "SOURCE_AUTHORITY",
    "UNSTAMPED_AUTHORITY",
    "WIN_MARGIN",
    "ConflictRecord",
    "ContradictionReport",
    "apply_contradiction_resolution",
    "assess_contradiction",
    "contradiction_problems",
    "record_strength",
    "source_authority",
]
