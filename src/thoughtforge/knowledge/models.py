"""
All 14 core data structure types for MindSpark: ThoughtForge.

Storage formats:
- PersonalityCoreRecord     → YAML  (personality_core.yaml)
- UserPreferenceRecord      → JSONL (user_profile_store.jsonl)
- UserFactRecord            → JSONL (user_profile_store.jsonl)
- EpisodicMemoryRecord      → JSONL (episodic_store.jsonl)
- ResponsePatternRecord     → JSONL (response_patterns.jsonl)
- ActiveThreadStateRecord   → JSON  (active_thread_state.json)
- InputSketch               → transient (not persisted)
- MemoryActivationBundle    → transient
- CognitionScaffold         → transient
- CandidateRecord           → transient
- FragmentRecord            → transient
- FinalResponseRecord       → transient (optionally logged)
- WritebackRecord           → transient (drives persistence)
- RuntimeTurnState          → transient (full execution trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# TF-025 (D-DUP-1): _now_iso's one implementation lives in
# utils/common.py; this name is its alias, kept for the field
# factories below and for existing importers.
from thoughtforge.utils.common import now_iso as _now_iso


RecordStatus = Literal["created", "reinforced", "active", "cooling", "stale", "archived", "deleted"]
RetentionClass = Literal["protected", "durable", "standard", "ephemeral", "discardable"]


# ── 1. PersonalityCoreRecord ───────────────────────────────────────────────────

@dataclass
class PersonalityCoreRecord:
    """Stable behavioral identity. Never auto-pruned. Versioned manually."""

    record_id: str                         # e.g. "pers_001"
    version: int = 1
    traits: list[str] = field(default_factory=list)
    speech_style: list[str] = field(default_factory=list)
    behavior_rules: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    status: RecordStatus = "active"
    retention_class: RetentionClass = "protected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "version": self.version,
            "traits": self.traits,
            "speech_style": self.speech_style,
            "behavior_rules": self.behavior_rules,
            "avoid": self.avoid,
            "weights": self.weights,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "retention_class": self.retention_class,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonalityCoreRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 2. UserPreferenceRecord ────────────────────────────────────────────────────

@dataclass
class UserPreferenceRecord:
    """User tone/style preference. Soft-decays if unconfirmed."""

    record_id: str
    category: str                          # e.g. "tone", "format", "depth"
    tags: list[str] = field(default_factory=list)
    summary: str = ""                      # max 160 chars
    value: str = ""
    importance: float = 0.5               # 0.0–1.0
    confidence: float = 0.5
    recency_weight: float = 1.0
    times_reinforced: int = 0
    times_retrieved: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used_at: str = field(default_factory=_now_iso)
    last_confirmed_at: str = field(default_factory=_now_iso)
    status: RecordStatus = "created"
    retention_class: RetentionClass = "standard"
    stale_after_turns: int = 200
    record_type: str = "user_preference"
    # TF-047: provenance stamp (TF-042's MEMORY_PROVENANCE
    # vocabulary). "" on legacy records written before the stamp
    # existed; the writeback leg always stamps new records.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserPreferenceRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 3. UserFactRecord ──────────────────────────────────────────────────────────

@dataclass
class UserFactRecord:
    """Durable contextual fact about the user or their work."""

    record_id: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""                      # max 180 chars
    fact_value: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 0.6
    recency_weight: float = 1.0
    times_reinforced: int = 0
    times_retrieved: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used_at: str = field(default_factory=_now_iso)
    status: RecordStatus = "created"
    retention_class: RetentionClass = "standard"
    stale_after_turns: int = 500
    record_type: str = "user_fact"
    # TF-047: provenance stamp — see UserPreferenceRecord.source.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserFactRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 4. EpisodicMemoryRecord ────────────────────────────────────────────────────

@dataclass
class EpisodicMemoryRecord:
    """Compact summary of a notable conversation moment."""

    record_id: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""                      # max 240 chars
    tone: str = ""
    importance: float = 0.4
    confidence: float = 0.5
    quality: float = 0.5
    recency_weight: float = 1.0
    linked_preferences: list[str] = field(default_factory=list)
    linked_facts: list[str] = field(default_factory=list)
    times_reinforced: int = 0
    times_retrieved: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used_at: str = field(default_factory=_now_iso)
    status: RecordStatus = "created"
    retention_class: RetentionClass = "ephemeral"
    stale_after_turns: int = 30
    record_type: str = "episode"
    # TF-047: provenance stamp — see UserPreferenceRecord.source.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodicMemoryRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 5. ResponsePatternRecord ───────────────────────────────────────────────────

@dataclass
class ResponsePatternRecord:
    """Reusable successful response structure."""

    record_id: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    pattern_shape: list[str] = field(default_factory=list)  # e.g. ["acknowledge", "explain", "example"]
    quality: float = 0.6
    times_successful: int = 0
    times_reinforced: int = 0
    times_retrieved: int = 0
    importance: float = 0.5
    confidence: float = 0.6
    recency_weight: float = 1.0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_used_at: str = field(default_factory=_now_iso)
    status: RecordStatus = "created"
    retention_class: RetentionClass = "standard"
    stale_after_turns: int = 100
    record_type: str = "response_pattern"
    # TF-047: provenance stamp — see UserPreferenceRecord.source.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResponsePatternRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 6. ActiveThreadStateRecord ─────────────────────────────────────────────────

@dataclass
class ActiveThreadStateRecord:
    """Hot-path current conversation thread state. Aggressive expiry."""

    record_id: str
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    priority: float = 0.8
    status: RecordStatus = "active"
    open_loops: list[str] = field(default_factory=list)   # max 6 soft / 8 hard
    expires_after_turns: int = 10
    turns_since_touch: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    retention_class: RetentionClass = "ephemeral"
    record_type: str = "active_thread_state"

    def cool_down(self) -> None:
        """Apply one turn of priority cooling."""
        self.turns_since_touch += 1
        self.priority = max(0.0, self.priority - 0.08 * self.turns_since_touch)
        self.expires_after_turns = max(0, self.expires_after_turns - self.turns_since_touch)
        if self.expires_after_turns <= 0 or self.priority < 0.20:
            self.status = "stale"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActiveThreadStateRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 7. InputSketch ─────────────────────────────────────────────────────────────

@dataclass
class InputSketch:
    """Transient representation of the user's current message. Not persisted."""

    raw_text: str
    intent: str = ""                       # the router's taxonomy (TF-020): technical_debugging | technical_spec_request | editing_refinement | brainstorming | naming_or_branding | planning_request | emotional_support | factual_query | light_conversation
    topic: str = ""
    tone_in: str = ""                      # detected tone of user's message: frustrated | tired | urgent | curious | playful | focused | neutral
    response_mode: str = "conversational"  # the router's modes (TF-020): structured_technical | concise_editorial | creative_brainstorm | calm_supportive | factual_direct | light_natural_chat (the default here is a construction fallback the router never emits)
    memory_triggers: list[str] = field(default_factory=list)
    urgency: float = 0.5                   # 0.0 = low, 1.0 = urgent
    retrieval_path: str = "hybrid"         # sql | vector | hybrid


# ── 8. MemoryActivationBundle ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CitationPayload:
    """TF-048: the citation an activated record carries by construction.

    One payload per retrieved item, derived deterministically from
    the record's own raw retrieval dict (see ``derive_citation``):
    ``marker`` is the exact bracket token the prompt renders
    (``[marker]`` — a bare QID for Wikidata entities, so the
    existing QID citation checks in refinement/ keep matching it;
    ``ref:<chunk_id>`` for reference chunks; ``geonames:<id>`` for
    places; ``conceptnet:<c1>|<relation>|<c2>`` for ConceptNet
    edges; ``<source>:<record_id>`` for everything else). ``source``
    is the retrieval source name, ``locator`` the durable pointer
    back to the stored claim (the QID, the reference chunk's
    source_file — TF-042: that file IS a reference chunk's
    provenance — the GeoNames id, the edge's triple text), and
    ``label`` the human name the prompt line shows.
    """

    marker: str
    source: str
    record_id: str
    record_type: str
    locator: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "source": self.source,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "locator": self.locator,
            "label": self.label,
        }


def derive_citation(
    record_id: str,
    record_type: str,
    raw: dict[str, Any] | None,
) -> CitationPayload:
    """Derive the citation payload for one retrieved item.

    Pure and total: any raw dict yields a payload, and the marker
    is never empty — an item whose raw carries no stronger identity
    falls back to ``<source>:<record_id>`` (or
    ``<source>:unidentified`` when even the id is empty), so the
    prompt-side rule "every line is cited" has no silent exception.
    Identity fields follow TF-042's contract per kind: entities by
    qid, reference chunks by chunk_id + source_file, GeoNames
    places by geonames_id, ConceptNet edges by their triple.
    """
    raw = raw or {}
    label = str(raw.get("label_en") or raw.get("title") or "")

    qid = raw.get("qid")
    if qid:
        return CitationPayload(
            marker=str(qid),
            source=str(raw.get("source") or "wikidata"),
            record_id=str(record_id),
            record_type=str(record_type),
            locator=str(qid),
            label=label,
        )

    chunk_id = raw.get("chunk_id")
    if chunk_id:
        return CitationPayload(
            marker=f"ref:{chunk_id}",
            source=str(raw.get("source") or "reference"),
            record_id=str(record_id),
            record_type=str(record_type),
            locator=str(raw.get("source_file") or chunk_id),
            label=label,
        )

    geonames_id = raw.get("geonames_id")
    if geonames_id:
        return CitationPayload(
            marker=f"geonames:{geonames_id}",
            source=str(raw.get("source") or "geonames"),
            record_id=str(record_id),
            record_type=str(record_type),
            locator=str(geonames_id),
            label=label,
        )

    concept1 = raw.get("concept1")
    concept2 = raw.get("concept2")
    if concept1 and concept2:
        relation = str(raw.get("relation") or "")
        return CitationPayload(
            marker=f"conceptnet:{concept1}|{relation}|{concept2}",
            source=str(raw.get("source") or "conceptnet"),
            record_id=str(record_id),
            record_type=str(record_type),
            locator=str(raw.get("content") or f"{concept1} {relation} {concept2}"),
            label=label,
        )

    source = str(raw.get("source") or record_type or "unknown")
    return CitationPayload(
        marker=f"{source}:{record_id}" if record_id else f"{source}:unidentified",
        source=source,
        record_id=str(record_id),
        record_type=str(record_type),
        locator=str(record_id or ""),
        label=label,
    )


@dataclass
class ActivatedRecord:
    """A single activated memory record with its retrieval score and compressed cue.

    TF-048: ``citation`` is the record's CitationPayload, derived
    in ``__post_init__`` whenever the caller did not supply one —
    a retrieved item cannot exist without its citation payload.
    """

    record_id: str
    record_type: str
    score: float
    cue: str                               # 5–18 word compressed cue
    raw: dict[str, Any] = field(default_factory=dict)
    citation: CitationPayload | None = None

    def __post_init__(self) -> None:
        if self.citation is None:
            self.citation = derive_citation(self.record_id, self.record_type, self.raw)


@dataclass
class MemoryActivationBundle:
    """Compact retrieved memory passed to the generation layer."""

    personality_core_id: str | None = None
    activated_records: list[ActivatedRecord] = field(default_factory=list)
    total_retrieved: int = 0
    retrieval_confidence: float = 0.0


# ── 9. CognitionScaffold ──────────────────────────────────────────────────────

@dataclass
class CognitionScaffold:
    """Tiny YAML-like steering object for deterministic generation guidance."""

    goal: str = ""
    tone: list[str] = field(default_factory=list)          # e.g. ["warm", "direct", "grounded"]
    focus: list[str] = field(default_factory=list)         # e.g. ["clarity", "brevity"]
    avoid: list[str] = field(default_factory=list)         # e.g. ["fluff", "handwaving"]
    depth: str = "medium"                                  # light | medium | expert
    candidate_modes: list[str] = field(default_factory=list)
    fact_block: str = ""                                   # injected retrieval context
    # TF-023 (CORE_LOOP §8-F7 closed): there is deliberately NO
    # prompt_text field. A turn builds many prompts (up to 2 candidate
    # passes, refine passes, a repair pass) and PromptBuilder returns
    # each prompt string to its caller; a single scaffold field could
    # only ever hold one of them and mislead. Prompt export, if a
    # consumer needs it, is TF-038's trace surface, not this object.

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ── 10. CandidateRecord ────────────────────────────────────────────────────────

@dataclass
class CandidateScores:
    relevance: float = 0.0
    clarity: float = 0.0
    coherence: float = 0.0
    personality_fit: float = 0.0
    specificity: float = 0.0
    genericness_penalty: float = 0.0
    goal_fit: float = 0.0

    @property
    def composite(self) -> float:
        return (
            self.relevance * 0.28
            + self.clarity * 0.18
            + self.coherence * 0.16
            + self.personality_fit * 0.16
            + self.specificity * 0.14
            + self.goal_fit * 0.08
            - self.genericness_penalty * 0.10
        )


@dataclass
class CandidateRecord:
    """First-pass generation output awaiting salvage scoring."""

    candidate_id: str
    mode: str = ""
    text: str = ""
    token_estimate: int = 0
    scores: CandidateScores = field(default_factory=CandidateScores)


# ── 11. FragmentRecord ────────────────────────────────────────────────────────

@dataclass
class FragmentScores:
    relevance: float = 0.0
    clarity: float = 0.0
    specificity: float = 0.0
    usefulness: float = 0.0
    personality_fit: float = 0.0
    genericness_penalty: float = 0.0

    @property
    def composite(self) -> float:
        return (
            self.relevance * 0.30
            + self.clarity * 0.18
            + self.specificity * 0.18
            + self.usefulness * 0.20
            + self.personality_fit * 0.14
            - self.genericness_penalty * 0.10
        )


@dataclass
class FragmentRecord:
    """A salvaged text fragment extracted from a candidate draft."""

    fragment_id: str
    source_candidate_id: str
    text: str = ""
    position: int = 0                      # character offset in source candidate
    scores: FragmentScores = field(default_factory=FragmentScores)
    keep: bool = False


# ── 12. FinalResponseRecord ───────────────────────────────────────────────────

@dataclass
class FinalResponseScores:
    relevance: float = 0.0
    clarity: float = 0.0
    coherence: float = 0.0
    personality_fit: float = 0.0
    usefulness: float = 0.0
    goal_fit: float = 0.0
    genericness_penalty: float = 0.0

    @property
    def composite(self) -> float:
        return (
            self.relevance * 0.24
            + self.clarity * 0.18
            + self.coherence * 0.18
            + self.personality_fit * 0.14
            + self.usefulness * 0.16
            + self.goal_fit * 0.10
            - self.genericness_penalty * 0.10
        )

    @property
    def quality_tier(self) -> str:
        s = self.composite
        if s >= 0.85:
            return "excellent"
        if s >= 0.70:
            return "strong"
        if s >= 0.55:
            return "usable"
        if s >= 0.40:
            return "weak"
        return "reject"


@dataclass
class FinalResponseRecord:
    """The chosen output with source citations and enforcement status."""

    response_id: str
    text: str = ""
    source_candidate_ids: list[str] = field(default_factory=list)
    source_fragment_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    scores: FinalResponseScores = field(default_factory=FinalResponseScores)
    enforcement_passed: bool = False
    enforcement_notes: str = ""
    token_count: int = 0
    created_at: str = field(default_factory=_now_iso)
    turn_id: str = ""
    salvage_path: str = "direct"
    retrieval_confidence: float = 0.0
    mode: str = "knowledge_only"
    # TF-097: the salvage rescue record (format
    # thoughtforge-rescue-record/1) as detached data,
    # carried into the TF-038 trace's final section;
    # None on finals salvage never composed.
    rescue_record: dict[str, Any] | None = None


# ── 13. WritebackRecord ───────────────────────────────────────────────────────

@dataclass
class WritebackRecord:
    """Instructions for updating persistent memory after a turn."""

    turn_id: str
    new_preferences: list[dict[str, Any]] = field(default_factory=list)
    new_facts: list[dict[str, Any]] = field(default_factory=list)
    new_episodes: list[dict[str, Any]] = field(default_factory=list)
    new_patterns: list[dict[str, Any]] = field(default_factory=list)
    thread_updates: dict[str, Any] = field(default_factory=dict)
    records_to_reinforce: list[str] = field(default_factory=list)
    records_to_archive: list[str] = field(default_factory=list)
    records_to_delete: list[str] = field(default_factory=list)


# ── 14. RuntimeTurnState ──────────────────────────────────────────────────────

@dataclass
class RuntimeTurnState:
    """Full transient state for one complete turn execution.

    TF-025: the typed turn-state object for the whole core loop.
    ``phase`` holds one of ``cognition.turn_state.PHASE_ORDER`` and
    only ever moves forward (see ``advance_phase`` there);
    ``repair_attempted`` records the quality-gate repair branch;
    ``salvage_ms`` is the measured compose (salvage + enforcement)
    time — exactly 0 when no candidates existed and no salvage ran.

    TF-027: ``interrupted`` / ``interrupt_reason`` /
    ``interrupted_phase`` record a cut (see
    ``cognition.interruption``): while a turn is cut,
    ``interrupted`` is True, the other two describe that cut, and
    ``final_response`` / ``completed_at`` stay empty.
    ``resume_count`` counts the times a cut turn was taken up
    again; the cut fields are cleared when a resume begins, so
    they always describe the *current* cut, never a stale one.

    TF-031: ``generation_failures`` carries the turn's
    ``cognition.error_honesty.GenerationFailure`` records — one
    per generation failure the core caught (kept untyped here to
    avoid a models → cognition import cycle). Empty on a clean
    turn and on a deliberate no-model turn: failures exist only
    where generation was attempted and did not produce.
    """

    turn_id: str
    input_sketch: InputSketch | None = None
    memory_bundle: MemoryActivationBundle | None = None
    scaffold: CognitionScaffold | None = None
    candidates: list[CandidateRecord] = field(default_factory=list)
    fragments: list[FragmentRecord] = field(default_factory=list)
    final_response: FinalResponseRecord | None = None
    writeback: WritebackRecord | None = None
    started_at: str = field(default_factory=_now_iso)
    completed_at: str = ""
    total_tokens_used: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    salvage_ms: int = 0
    phase: str = "created"
    repair_attempted: bool = False
    interrupted: bool = False
    interrupt_reason: str = ""
    interrupted_phase: str = ""
    resume_count: int = 0
    generation_failures: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True once the turn has reached the completed phase."""
        return self.phase == "completed" and bool(self.completed_at)

    @property
    def is_interrupted(self) -> bool:
        """TF-027: True while the turn stands cut, awaiting resume."""
        return self.interrupted and not self.is_complete
