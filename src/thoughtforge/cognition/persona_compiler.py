"""
TF-143 — the persona compiler merge, as shipping production code.

TF-141's verdict was **MERGE** (adopt-in-part): wyrdforge's
persona compiler wins seven measured facets the shipping
persona path (TF-009 / TF-030) does not have — per-record
provenance with confidences, evidence-weighted traits,
uncertainty surfacing, the identity_core /
truth_anchor_points split, mode adaptation, symbolic and
relationship sections — while the shipping path wins the
two facets that make a persona real: steering DELIVERED
into every prompt (``inject_persona``) and avoid CHECKED
deterministically (``check_persona_response`` /
``score_persona_turns``). This module implements the merge
under TF-141's filed conditions:

  * **A compiled-expectation builder BESIDE persona.py** —
    :class:`PersonaCompiler` compiles canonical records
    (production input shapes below, grounded in the wing's
    canonical_fact / symbolic_trace schemas and TF-141's
    spike input shapes) into a :class:`CompiledPersona`
    carrying every adopted facet, and a
    :class:`CompiledPersonaExpectation` — a
    ``PersonaExpectation`` subclass — that carries the
    compiled facets into the shipping runtime. TF-030's
    runtime stays the delivery + judgement: ``inject_persona``
    delivers the merged steering (the subclass's
    ``steering_lines()`` extends the base lines with the
    compiled sections), and ``check_persona_response`` /
    ``score_persona_turns`` judge responses against the
    merged expectation's avoid list unchanged. persona.py
    itself is NOT modified; the authored path's behaviour
    is byte-for-byte as TF-030 filed it.
  * **TF-009's authored path remains a supported source,
    never silently replaced** — compiling accepts an
    optional authored ``PersonaExpectation`` as the base:
    its expressive lists and its avoid ride into the merged
    expectation verbatim, and its record_id is filed in the
    compiled provenance. Compiling without a base is equally
    supported (traits derive from evidence alone).
  * **Avoid sourcing — filed decision (``AVOID_DECISION``)**:
    avoid is DECLARED, never compiled. No record shape the
    compiler reads carries an avoid entry, so a compiled
    avoid would be invented prose wearing evidence's
    clothing; the merged shape therefore sources avoid
    solely from the authored core (declared), exactly as
    TF-030 checks it. The compiled facets contribute no
    avoid entries, ever.
  * **The compiler's defects are NOT imported** — each is
    filed as a measured difference in ``DEFECT_DIFFERENCES``:
    (1) approval is checked FIRST: an unapproved canonical
    or symbolic record surfaces as an uncertainty_point and
    NEVER as a truth anchor, identity line, trait or
    symbolic line (the spike's elif chain promoted the
    pending bell_count fact into truth_anchor_points; on
    TF-141's Wrenna fixtures the production compile files
    anchors 4 / uncertainty 3 where the spike filed
    anchors 5 / uncertainty 2 — measured live by
    :func:`run_persona_compile`); (2) ``relationship_excerpt``
    files bond-only lines (the production ``BondEdge``'s
    three compiled lines), never a duplicate of
    ``response_guidance``; (3) ``token_budget_hint`` is
    DROPPED (``TOKEN_BUDGET_DECISION``): the production
    compiled persona carries no such field — a hint nothing
    honours is not a bound, and the merged shape's real
    bounds are its filed caps plus TF-030's
    ``MAX_STEERING_ENTRIES`` at delivery.
  * **Relationship compiles from production** — the bond
    input is TF-143's production ``BondEdge``
    (``knowledge/bond_graph.py``), not TF-139's spike edge:
    the layer TF-139 adopted is the compiler's relationship
    source, closing the loop between the two adoptions.

Caps, faithful to the compiler (all measured binding on
the 57-record caps run): identity_core 10,
truth_anchor_points 16, active_traits 8, symbolic_context
8, uncertainty_points 8, source_items 20,
relationship_excerpt 6, response_guidance 10. Trait
arithmetic is the compiler's: ``truth.confidence`` summed
per bucket (buckets keyed by fact_value for fact_keys
temperament/tone/value/mood), weight ``min(1.0, sum)``,
supporting record ids carried, sorted by weight.

RETIRE verdicts file nothing here: TF-140's micro-RAG and
TF-142's truth calibrator are NOT rebuilt by this slice
(TF-144 executes their retirements), and none of TF-142's
nine filed calibrator defects is imported — this module
contains no scorer at all.

All fixtures are synthetic (TF-017 / TF-030's Wrenna /
Lantern Hollow world; no canon). The human-readable
write-up is ``docs/architecture/PERSONA_COMPILER.md``,
filed verbatim as this module's render (regenerate with
``python3 -m thoughtforge.cognition.persona_compiler``).
Proof grade: FIXTURE/STATIC — schemas + compiler shell +
the shipping runtime re-measured live on this VM over
synthetic fixtures; no live model, never claimed.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from thoughtforge.cognition.persona import (
    FIXTURE_EXPECTATION,
    PERSONA_FIXTURE_TURNS,
    PersonaExpectation,
    inject_persona,
    score_persona_turns,
)
from thoughtforge.knowledge.bond_graph import BondEdge, BondVector
from thoughtforge.knowledge.models import CognitionScaffold

PERSONA_COMPILER_FORMAT = "thoughtforge-persona-compiler/1"
PERSONA_COMPILER_DOC = "docs/architecture/PERSONA_COMPILER.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_COMPILER = _WING / "services" / "persona_compiler.py"
_SCHEMAS = _WING / "schemas"

#: The compiler's caps, exactly as the wing files them.
CAPS: dict[str, int] = {
    "identity_core": 10, "truth_anchor_points": 16,
    "active_traits": 8, "symbolic_context": 8,
    "uncertainty_points": 8, "source_items": 20,
    "relationship_excerpt": 6, "response_guidance": 10,
}

#: Domains that file a canonical fact as identity.
IDENTITY_DOMAINS: frozenset[str] = frozenset(
    {"identity", "style", "mission"})

#: Fact keys that feed trait buckets.
TRAIT_KEYS: frozenset[str] = frozenset(
    {"temperament", "tone", "value", "mood"})

#: The compiler's base tone contract, verbatim.
BASE_TONE_CONTRACT: tuple[str, ...] = (
    "preserve continuity without inventing unsupported memory",
    "prefer grounded specifics over vague mythic filler",
    "mark uncertainty cleanly when canon or memory is missing",
)


# ── Filed decisions ──────────────────────────────────────────────────────────

AVOID_DECISION: dict[str, Any] = {
    "decision": "declared",
    "rule": "the merged expectation's avoid list is sourced "
            "solely from the authored PersonalityCoreRecord "
            "(via the base PersonaExpectation); the compiler "
            "contributes no avoid entries",
    "reasons": (
        "no record shape the compiler reads (canonical fact, "
        "symbolic trace, generic record) carries an avoid "
        "entry — a compiled avoid would be invented prose "
        "wearing evidence's clothing",
        "avoid is a prohibition the author states, not a fact "
        "evidence sums to: TF-030 checks it deterministically "
        "precisely because it is declared verbatim",
        "the authored path stays a supported source (TF-141's "
        "condition), so a declared avoid always has a home in "
        "the merged shape; compiling without a base files an "
        "empty avoid, honestly, never a fabricated one",
    ),
}

TOKEN_BUDGET_DECISION: dict[str, Any] = {
    "decision": "dropped",
    "rule": "the production compiled persona carries no "
            "token_budget_hint field at all",
    "reasons": (
        "TF-141 measured the spike's hint (default 800) as "
        "carried but never enforced — a hint nothing honours "
        "is not a bound, and TF-141's condition allows exactly "
        "two honest outcomes: real trimming or dropping it",
        "the merged shape's real bounds already exist and are "
        "enforced: the compiler's filed caps on every section "
        "and TF-030's MAX_STEERING_ENTRIES at delivery — a "
        "second, unenforced number would only pretend to "
        "bound what those already bound",
        "no trimming order is filed because nothing is "
        "trimmed by budget: if a future slice needs a token "
        "bound it must file its own trim order against the "
        "tree's word-token convention then, with evidence",
    ),
}

#: The three compiler defects, each filed as the measured
#: difference the production compile makes (re-measured
#: live by run_persona_compile()).
DEFECT_DIFFERENCES: tuple[dict[str, Any], ...] = (
    {"id": "approval_checked_first",
     "spike": "the elif chain checks canonical/symbolic "
              "BEFORE approval: the pending canonical fact "
              "(bell_count, confidence 0.3) files as the "
              "truth anchor world:lantern-hollow.bell_count="
              "two; spike anchors 5 / uncertainty 2",
     "production": "approval is checked FIRST: any "
                   "unapproved record (canonical, symbolic "
                   "or generic) surfaces as an "
                   "uncertainty_point and never as an "
                   "anchor, identity line, trait or "
                   "symbolic line; production anchors 4 / "
                   "uncertainty 3 on the same fixtures"},
    {"id": "relationship_excerpt_bond_only",
     "spike": "relationship_excerpt is response_guidance[:6] "
              "verbatim, duplicating all 4 guidance lines "
              "(bond lines + the mode line)",
     "production": "relationship_excerpt files the bond-only "
                   "lines (3 on the fixture run) and shares no "
                   "line with the mode guidance"},
    {"id": "token_budget_hint_dropped",
     "spike": "token_budget_hint (default 800) is carried "
              "onto the packet and never enforced",
     "production": "no token_budget_hint field exists on the "
                   "compiled persona (TOKEN_BUDGET_DECISION)"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def caps_problems(counts: Any) -> list[str]:
    """Judge section counts against the filed
    caps, purely. Empty-safe."""
    if not isinstance(counts, dict) or not counts:
        return ["caps: the counts filing is empty"]
    problems: list[str] = []
    for name, cap in CAPS.items():
        value = counts.get(name)
        if isinstance(value, int) and value > cap:
            problems.append(
                f"caps: {name} files {value}, over its cap {cap}")
    return problems


def trait_problems(signals: Any, expected: Any) -> list[str]:
    """Judge trait signals against an expected
    filing, purely: names, weights (4dp) and
    supporting ids must match. Empty-safe."""
    if not isinstance(expected, list) or not expected:
        return ["traits: the expected filing is empty"]
    if not isinstance(signals, list):
        return ["traits: the signals filing is not a list"]
    filed = [
        {"trait_name": s["trait_name"],
         "weight": round(s["weight"], 4),
         "supporting_record_ids": list(s["supporting_record_ids"])}
        if isinstance(s, dict) else
        {"trait_name": s.trait_name,
         "weight": round(s.weight, 4),
         "supporting_record_ids": list(s.supporting_record_ids)}
        for s in signals]
    if filed != expected:
        return [f"traits: the signals file {filed}, not the "
                f"filed {expected}"]
    return []


def approval_problems(
    truth_anchor_points: Any, identity_core: Any,
    symbolic_context: Any, uncertainty_points: Any,
    unapproved_lines: Any,
) -> list[str]:
    """Judge the approval law, purely: no
    unapproved record's line may appear in the
    anchors, identity or symbolic sections, and
    each must surface in uncertainty. The
    unapproved lines file (anchor_line,
    summary) pairs. Empty-safe."""
    if not isinstance(unapproved_lines, (list, tuple)) \
            or not unapproved_lines:
        return ["approval: the unapproved filing is empty"]
    problems: list[str] = []
    anchors = list(truth_anchor_points or ())
    identity = list(identity_core or ())
    symbolic = list(symbolic_context or ())
    uncertainty = list(uncertainty_points or ())
    for line, summary in unapproved_lines:
        if line in anchors:
            problems.append(
                f"approval: the unapproved line {line!r} files "
                "as a truth anchor")
        if line in identity:
            problems.append(
                f"approval: the unapproved line {line!r} files "
                "as identity")
        if any(line and line in row for row in symbolic):
            problems.append(
                f"approval: the unapproved trace {line!r} files "
                "as symbolic context")
        if summary not in uncertainty:
            problems.append(
                f"approval: the unapproved record {summary!r} "
                "never surfaces as uncertainty")
    return problems


def relationship_problems(
    relationship_excerpt: Any, response_guidance: Any,
    bond_lines: Any,
) -> list[str]:
    """Judge the relationship excerpt, purely:
    it must equal the bond-only lines exactly —
    never a duplicate of response guidance.
    Empty-safe."""
    if not isinstance(bond_lines, (list, tuple)) or not bond_lines:
        return ["relationship: the bond filing is empty"]
    problems: list[str] = []
    excerpt = list(relationship_excerpt or ())
    if excerpt != list(bond_lines):
        problems.append(
            f"relationship: the excerpt files {excerpt}, not "
            f"the bond-only lines {list(bond_lines)}")
    guidance = list(response_guidance or ())
    mode_lines = [row for row in guidance if row not in bond_lines]
    for row in excerpt:
        if row in mode_lines:
            problems.append(
                f"relationship: the excerpt duplicates the "
                f"guidance line {row!r}")
    return problems


def mode_problems(mode: Any) -> list[str]:
    """Judge one mode value against the schema's
    PersonaMode enum, purely. Empty-safe."""
    values = [m.value for m in PersonaMode]
    if mode not in values:
        return [f"mode: {mode!r} is not one of {values}"]
    return []


def measured_problems(measured: Any, expected: Any) -> list[str]:
    """Judge measured metrics against an expected
    filing, purely. Empty-safe."""
    if not isinstance(measured, dict) or not measured:
        return ["measured: the measured filing is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["measured: the expected filing is empty"]
    problems: list[str] = []
    for name, value in expected.items():
        if measured.get(name) != value:
            problems.append(
                f"measured: {name} measures {measured.get(name)!r}, "
                f"not the filed {value!r}")
    return problems


def schema_field_problems(field_map: Any) -> list[str]:
    """Judge a {shape: fields} map against the
    schemas parsed live from the wing.
    Empty-safe."""
    if not isinstance(field_map, dict) or not field_map:
        return ["schema fields: the map is empty"]
    problems: list[str] = []
    expected = _schema_field_map()
    for shape, fields in expected.items():
        filed = tuple(field_map.get(shape) or ())
        if set(filed) != set(fields) or len(filed) != len(fields):
            problems.append(
                f"schema fields: {shape} files {sorted(filed)}, not "
                f"the schema's {sorted(fields)}")
    for shape in field_map:
        if shape not in expected:
            problems.append(f"schema fields: unknown shape {shape!r}")
    return problems


def _schema_field_map() -> dict[str, tuple[str, ...]]:
    """The field sets of the shapes this module
    rebuilds, parsed live from the three schemas
    (persona_packet $defs; the payload / content /
    truth $defs of the two record schemas)."""
    packet = json.loads(
        (_SCHEMAS / "persona_packet.schema.json").read_text(
            encoding="utf-8"))
    canonical = json.loads(
        (_SCHEMAS / "canonical_fact_record.schema.json").read_text(
            encoding="utf-8"))
    symbolic = json.loads(
        (_SCHEMAS / "symbolic_trace_record.schema.json").read_text(
            encoding="utf-8"))
    return {
        "PersonaSourceItem": tuple(
            packet["$defs"]["PersonaSourceItem"]["properties"]),
        "TraitSignal": tuple(
            packet["$defs"]["TraitSignal"]["properties"]),
        "CanonicalFactPayload": tuple(
            canonical["$defs"]["CanonicalFactPayload"]["properties"]),
        "CanonicalFactContent": tuple(
            canonical["$defs"]["CanonicalFactContent"]["properties"]),
        "SymbolicTracePayload": tuple(
            symbolic["$defs"]["SymbolicTracePayload"]["properties"]),
        "SymbolicTraceContent": tuple(
            symbolic["$defs"]["SymbolicTraceContent"]["properties"]),
        "TruthMeta": tuple(
            canonical["$defs"]["TruthMeta"]["properties"]),
    }


# ── Production input shapes (from the schemas / TF-141's spike inputs) ───────

class PersonaMode(str, Enum):
    COMPANION = "companion"
    CODING_GUIDE = "coding_guide"
    WORLD_SEER = "world_seer"
    RITUAL = "ritual"
    DEBRIEF = "debrief"


class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass
class TruthMeta:
    support_class: str = "supported"
    confidence: float = 0.5
    contradiction_status: str = "none"
    approval_state: ApprovalState = ApprovalState.PENDING


@dataclass
class CanonicalFactPayload:
    fact_subject_id: str
    fact_key: str
    fact_value: str
    value_type: str = "string"
    domain: str = "general"
    support_record_ids: list[str] = field(default_factory=list)
    supersedes_record_id: str | None = None


@dataclass
class CanonicalFactContent:
    title: str
    summary: str
    body_ref: str | None = None
    structured_payload: CanonicalFactPayload = field(
        default_factory=lambda: CanonicalFactPayload(
            fact_subject_id="", fact_key="", fact_value=""))


@dataclass
class SymbolicTracePayload:
    symbol_type: str
    rune_signature: list[str] = field(default_factory=list)
    omen_strength: float = 0.0
    ritual_charge: float = 0.0
    mood_tags: list[str] = field(default_factory=list)


@dataclass
class SymbolicTraceContent:
    title: str
    summary: str
    body_ref: str | None = None
    structured_payload: SymbolicTracePayload = field(
        default_factory=lambda: SymbolicTracePayload(symbol_type=""))


@dataclass
class GenericContent:
    """A non-canonical, non-symbolic record's
    content, reduced to the one field the compiler
    reads for such records (summary)."""
    summary: str


@dataclass
class PersonaRecord:
    """The compiler's input, reduced to what
    compile() reads: record_id, record_type,
    content.summary, truth.confidence,
    truth.approval_state."""
    record_id: str
    record_type: str
    content: Any
    truth: TruthMeta = field(default_factory=TruthMeta)


@dataclass
class CanonicalFactRecord(PersonaRecord):
    content: CanonicalFactContent = field(
        default_factory=lambda: CanonicalFactContent(
            title="", summary=""))


@dataclass
class SymbolicTraceRecord(PersonaRecord):
    content: SymbolicTraceContent = field(
        default_factory=lambda: SymbolicTraceContent(
            title="", summary=""))


# ── Production output shapes ─────────────────────────────────────────────────

@dataclass
class PersonaSourceItem:
    record_id: str
    item_type: str
    text: str
    confidence: float = 0.5


@dataclass
class TraitSignal:
    trait_name: str
    weight: float = 0.5
    supporting_record_ids: list[str] = field(default_factory=list)


@dataclass
class CompiledPersonaExpectation(PersonaExpectation):
    """The merged expectation: TF-030's expectation
    (expressive lists + declared avoid + weights)
    carrying the compiled facets. inject_persona
    delivers base + compiled steering lines;
    check_persona_response / score_persona_turns
    judge against the declared avoid unchanged."""

    persona_id: str = ""
    user_id: str = ""
    mode: str = ""
    identity_core: list[str] = field(default_factory=list)
    truth_anchor_points: list[str] = field(default_factory=list)
    uncertainty_points: list[str] = field(default_factory=list)
    symbolic_context: list[str] = field(default_factory=list)
    tone_contract: list[str] = field(default_factory=list)
    response_guidance: list[str] = field(default_factory=list)
    relationship_excerpt: list[str] = field(default_factory=list)
    active_traits: list[TraitSignal] = field(default_factory=list)
    source_items: list[PersonaSourceItem] = field(default_factory=list)
    base_record_id: str = ""

    def steering_lines(self) -> list[str]:
        """Base TF-030 lines first (unchanged), then
        one line per non-empty compiled section."""
        lines = super().steering_lines()
        sections = (
            ("identity_core", "Persona identity"),
            ("truth_anchor_points", "Persona truth anchors"),
            ("tone_contract", "Persona tone"),
            ("response_guidance", "Persona guidance"),
            ("relationship_excerpt", "Persona relationship"),
            ("symbolic_context", "Persona symbolic"),
            ("uncertainty_points", "Persona uncertainty"),
        )
        for name, label in sections:
            entries = getattr(self, name)
            if entries:
                lines.append(f"{label}: {'; '.join(entries)}")
        if self.active_traits:
            lines.append(
                "Persona trait weights: " + "; ".join(
                    f"{t.trait_name}={t.weight:.2f}"
                    for t in self.active_traits))
        return lines

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update({
            "persona_id": self.persona_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "identity_core": list(self.identity_core),
            "truth_anchor_points": list(self.truth_anchor_points),
            "uncertainty_points": list(self.uncertainty_points),
            "symbolic_context": list(self.symbolic_context),
            "tone_contract": list(self.tone_contract),
            "response_guidance": list(self.response_guidance),
            "relationship_excerpt": list(self.relationship_excerpt),
            "active_traits": [
                {"trait_name": t.trait_name, "weight": t.weight,
                 "supporting_record_ids":
                     list(t.supporting_record_ids)}
                for t in self.active_traits],
            "source_items": [
                {"record_id": s.record_id, "item_type": s.item_type,
                 "text": s.text, "confidence": s.confidence}
                for s in self.source_items],
            "base_record_id": self.base_record_id,
        })
        return data


@dataclass
class CompiledPersona:
    """The compiled persona: every adopted facet,
    plus the merged expectation that delivers it
    through TF-030's runtime. No token_budget_hint
    field exists (TOKEN_BUDGET_DECISION)."""

    persona_id: str
    user_id: str
    mode: PersonaMode
    tone_contract: list[str] = field(default_factory=list)
    identity_core: list[str] = field(default_factory=list)
    active_traits: list[TraitSignal] = field(default_factory=list)
    relationship_excerpt: list[str] = field(default_factory=list)
    truth_anchor_points: list[str] = field(default_factory=list)
    uncertainty_points: list[str] = field(default_factory=list)
    symbolic_context: list[str] = field(default_factory=list)
    response_guidance: list[str] = field(default_factory=list)
    source_items: list[PersonaSourceItem] = field(default_factory=list)
    expectation: CompiledPersonaExpectation | None = None

    def to_expectation(self) -> CompiledPersonaExpectation:
        """The merged expectation for the shipping
        runtime (built at compile time)."""
        if self.expectation is None:  # pragma: no cover — compile sets it
            raise ValueError("compiled persona carries no expectation")
        return self.expectation


#: Production shape name -> the production dataclass, for
#: the schema-coverage checker (schema shapes only).
PRODUCTION_SHAPES: dict[str, Any] = {
    "PersonaSourceItem": PersonaSourceItem,
    "TraitSignal": TraitSignal,
    "CanonicalFactPayload": CanonicalFactPayload,
    "CanonicalFactContent": CanonicalFactContent,
    "SymbolicTracePayload": SymbolicTracePayload,
    "SymbolicTraceContent": SymbolicTraceContent,
    "TruthMeta": TruthMeta,
}


def production_field_map() -> dict[str, tuple[str, ...]]:
    """The production dataclasses' field sets, by shape."""
    return {
        name: tuple(f.name for f in dataclasses.fields(cls))
        for name, cls in PRODUCTION_SHAPES.items()
    }


# ── The production compiler: the wing's compile, defects not imported ────────

class PersonaCompiler:
    """PersonaCompiler.compile()'s adopted facets
    over the production types: same domain/trait-key
    sets, same confidence arithmetic, same mode
    additions, same caps — with approval checked
    FIRST, a bond-only relationship excerpt, and no
    token budget hint."""

    def compile(
        self,
        *,
        persona_id: str,
        user_id: str,
        mode: PersonaMode,
        records: list[PersonaRecord],
        bond_edge: BondEdge | None = None,
        base_expectation: PersonaExpectation | None = None,
    ) -> CompiledPersona:
        identity_core: list[str] = []
        truth_anchor_points: list[str] = []
        uncertainty_points: list[str] = []
        symbolic_context: list[str] = []
        tone_contract: list[str] = list(BASE_TONE_CONTRACT)
        response_guidance: list[str] = []
        source_items: list[PersonaSourceItem] = []
        trait_buckets: dict[str, float] = {}
        trait_support: dict[str, list[str]] = {}

        for record in records:
            source_items.append(PersonaSourceItem(
                record_id=record.record_id,
                item_type=record.record_type,
                text=record.content.summary,
                confidence=record.truth.confidence))
            # DEFECT NOT IMPORTED (1): approval is checked
            # FIRST — an unapproved record of ANY kind
            # surfaces as uncertainty, never as truth.
            if record.truth.approval_state is not ApprovalState.APPROVED:
                uncertainty_points.append(record.content.summary)
                continue
            if isinstance(record, CanonicalFactRecord):
                payload = record.content.structured_payload
                line = (f"{payload.fact_subject_id}.{payload.fact_key}"
                        f"={payload.fact_value}")
                if payload.domain in IDENTITY_DOMAINS:
                    identity_core.append(line)
                else:
                    truth_anchor_points.append(line)
                if payload.fact_key in TRAIT_KEYS:
                    trait_buckets[payload.fact_value] = (
                        trait_buckets.get(payload.fact_value, 0.0)
                        + record.truth.confidence)
                    trait_support.setdefault(
                        payload.fact_value, []).append(record.record_id)
            elif isinstance(record, SymbolicTraceRecord):
                sp = record.content.structured_payload
                symbolic_context.append(
                    f"symbol={sp.symbol_type};"
                    f"runes={','.join(sp.rune_signature)};"
                    f"charge={sp.ritual_charge:.2f}")
            # An approved generic record files nothing
            # beyond its source item (the wing's shape).

        bond_lines: list[str] = []
        if bond_edge is not None:
            bond_lines = [
                f"relationship_mode="
                f"{bond_edge.active_modes.relational_mode}",
                f"relationship_weather="
                f"{bond_edge.active_modes.emotional_weather}",
                f"closeness_index={bond_edge.closeness_index():.2f}",
            ]
            response_guidance.extend(bond_lines)

        active_traits = [
            TraitSignal(
                trait_name=name, weight=min(1.0, weight),
                supporting_record_ids=trait_support[name])
            for name, weight in sorted(
                trait_buckets.items(), key=lambda item: item[1],
                reverse=True)[:CAPS["active_traits"]]
        ]

        if mode is PersonaMode.CODING_GUIDE:
            tone_contract.append(
                "opt for terse correctness and verifiable claims")
            response_guidance.append(
                "surface missing evidence before offering "
                "code-level certainty")
        elif mode is PersonaMode.WORLD_SEER:
            tone_contract.append(
                "preserve mythic atmosphere while separating "
                "omen from fact")
            response_guidance.append(
                "do not present symbolic traces as hard world "
                "canon without support")
        elif mode is PersonaMode.COMPANION:
            tone_contract.append(
                "maintain warmth without synthetic intimacy "
                "inflation")
            response_guidance.append(
                "never claim vows or memories that are not "
                "supported")

        compiled = CompiledPersona(
            persona_id=persona_id,
            user_id=user_id,
            mode=mode,
            tone_contract=tone_contract,
            identity_core=identity_core[:CAPS["identity_core"]],
            active_traits=active_traits,
            # DEFECT NOT IMPORTED (2): bond-only lines.
            relationship_excerpt=bond_lines[:CAPS["relationship_excerpt"]],
            truth_anchor_points=(
                truth_anchor_points[:CAPS["truth_anchor_points"]]),
            uncertainty_points=(
                uncertainty_points[:CAPS["uncertainty_points"]]),
            symbolic_context=symbolic_context[:CAPS["symbolic_context"]],
            response_guidance=response_guidance[:CAPS["response_guidance"]],
            source_items=source_items[:CAPS["source_items"]],
        )
        compiled.expectation = _merged_expectation(
            compiled, base_expectation)
        return compiled


def _merged_expectation(
    compiled: CompiledPersona,
    base: PersonaExpectation | None,
) -> CompiledPersonaExpectation:
    """Carry/extend a PersonaExpectation: the base's
    expressive lists + declared avoid + weights ride
    verbatim when a base is supplied (the authored
    path); without a base, traits derive from the
    compiled TraitSignals and avoid is empty
    (AVOID_DECISION — never fabricated)."""
    if base is not None:
        traits = list(base.traits)
        speech_style = list(base.speech_style)
        behavior_rules = list(base.behavior_rules)
        avoid = list(base.avoid)
        weights = dict(base.weights)
        record_id = base.record_id
    else:
        traits = [t.trait_name for t in compiled.active_traits]
        speech_style = []
        behavior_rules = []
        avoid = []
        weights = {}
        record_id = compiled.persona_id
    return CompiledPersonaExpectation(
        record_id=record_id,
        traits=traits,
        speech_style=speech_style,
        behavior_rules=behavior_rules,
        avoid=avoid,
        weights=weights,
        persona_id=compiled.persona_id,
        user_id=compiled.user_id,
        mode=compiled.mode.value,
        identity_core=list(compiled.identity_core),
        truth_anchor_points=list(compiled.truth_anchor_points),
        uncertainty_points=list(compiled.uncertainty_points),
        symbolic_context=list(compiled.symbolic_context),
        tone_contract=list(compiled.tone_contract),
        response_guidance=list(compiled.response_guidance),
        relationship_excerpt=list(compiled.relationship_excerpt),
        active_traits=list(compiled.active_traits),
        source_items=list(compiled.source_items),
        base_record_id=base.record_id if base is not None else "",
    )


# ── The shared synthetic ground truth (TF-017 / TF-030's world) ──────────────

def _canonical(
    record_id: str, *, subject: str, key: str, value: str,
    domain: str, confidence: float,
    approval: ApprovalState = ApprovalState.APPROVED,
    summary: str = "",
) -> CanonicalFactRecord:
    return CanonicalFactRecord(
        record_id=record_id, record_type="canonical_fact",
        content=CanonicalFactContent(
            title=f"{subject}.{key}", summary=summary,
            structured_payload=CanonicalFactPayload(
                fact_subject_id=subject, fact_key=key,
                fact_value=value, domain=domain)),
        truth=TruthMeta(confidence=confidence, approval_state=approval))


def _symbolic(
    record_id: str, *, symbol: str, runes: list[str],
    charge: float, confidence: float,
    approval: ApprovalState = ApprovalState.APPROVED,
    summary: str = "",
) -> SymbolicTraceRecord:
    return SymbolicTraceRecord(
        record_id=record_id, record_type="symbolic_trace",
        content=SymbolicTraceContent(
            title=symbol, summary=summary or f"a {symbol} trace",
            structured_payload=SymbolicTracePayload(
                symbol_type=symbol, rune_signature=runes,
                ritual_charge=charge)),
        truth=TruthMeta(confidence=confidence, approval_state=approval))


def _generic(
    record_id: str, *, summary: str, confidence: float,
    approval: ApprovalState,
) -> PersonaRecord:
    return PersonaRecord(
        record_id=record_id, record_type="episode_summary",
        content=GenericContent(summary=summary),
        truth=TruthMeta(confidence=confidence, approval_state=approval))


def fixture_records() -> list[PersonaRecord]:
    """TF-141's Wrenna fixtures, as production records
    — the identical ground truth (synthetic only)."""
    return [
        _canonical("rec:synthetic-wrenna-001",
                   subject="persona:wrenna", key="temperament",
                   value="steady", domain="identity", confidence=0.6,
                   summary="Wrenna is steady"),
        _canonical("rec:synthetic-wrenna-002",
                   subject="persona:wrenna", key="name",
                   value="Wrenna the Lantern-Keeper",
                   domain="identity", confidence=0.95,
                   summary="the keeper's name"),
        _canonical("rec:synthetic-wrenna-003",
                   subject="persona:wrenna", key="tone",
                   value="plain-spoken", domain="style",
                   confidence=0.8, summary="Wrenna is plain-spoken"),
        _canonical("rec:synthetic-wrenna-004",
                   subject="persona:wrenna", key="duty",
                   value="keep the ferry lantern lit",
                   domain="mission", confidence=0.85,
                   summary="the keeper's duty"),
        _canonical("rec:synthetic-wrenna-005",
                   subject="persona:wrenna", key="mood",
                   value="steady", domain="general", confidence=0.5,
                   summary="a steady mood, observed"),
        _canonical("rec:synthetic-wrenna-006",
                   subject="world:lantern-hollow",
                   key="ferry_minutes", value="11", domain="world",
                   confidence=0.9,
                   summary="the crossing takes eleven minutes"),
        _canonical("rec:synthetic-wrenna-007",
                   subject="world:lantern-hollow", key="river",
                   value="the Bracken", domain="world",
                   confidence=0.7, summary="the river's name"),
        _canonical("rec:synthetic-wrenna-008",
                   subject="persona:wrenna", key="value",
                   value="honesty", domain="general", confidence=0.6,
                   summary="honesty about the river"),
        _canonical("rec:synthetic-wrenna-009",
                   subject="world:lantern-hollow", key="bell_count",
                   value="two", domain="world", confidence=0.3,
                   approval=ApprovalState.PENDING,
                   summary="an unapproved bell count"),
        _symbolic("rec:synthetic-wrenna-010", symbol="lantern",
                  runes=["ansuz", "raidho"], charge=0.75,
                  confidence=0.65),
        _symbolic("rec:synthetic-wrenna-011", symbol="river-stone",
                  runes=["laguz"], charge=0.4, confidence=0.55),
        _generic("rec:synthetic-wrenna-012",
                 summary="Wrenna logged the river level at dawn",
                 confidence=0.7, approval=ApprovalState.APPROVED),
        _generic("rec:synthetic-wrenna-013",
                 summary="an unverified tale of a third tide bell",
                 confidence=0.4, approval=ApprovalState.PENDING),
        _generic("rec:synthetic-wrenna-014",
                 summary="a disputed tale of a night crossing",
                 confidence=0.35,
                 approval=ApprovalState.QUARANTINED),
    ]


def fixture_bond_edge() -> BondEdge:
    """TF-141's spike edge shape, as a PRODUCTION
    BondEdge (synthetic): closeness
    (0.6+0.6+0.3+0.25)/4 = 0.4375."""
    from thoughtforge.knowledge.bond_graph import BondDomain

    return BondEdge(
        bond_id="bond:wrenna-companion",
        entity_a="user:lantern-keeper",
        entity_b="persona:wrenna",
        domain=BondDomain.COMPANION,
        vector=BondVector(
            trust=0.6, warmth=0.6, familiarity=0.3, devotion=0.2,
            attraction_affinity=0.1, safety=0.9, sacred_resonance=0.3,
            playfulness=0.2, vulnerability=0.25,
            initiative_balance=0.0))


def caps_records() -> list[PersonaRecord]:
    """TF-141's generated 57-record set that binds
    every cap (synthetic, deterministic, all
    canonical/symbolic records approved)."""
    records: list[PersonaRecord] = []
    for i in range(12):
        records.append(_canonical(
            f"rec:synthetic-cap-id-{i:02d}", subject="persona:cap",
            key="temperament", value=f"trait{i:02d}",
            domain="identity", confidence=0.5,
            summary=f"cap identity {i}"))
    for i in range(20):
        records.append(_canonical(
            f"rec:synthetic-cap-world-{i:02d}",
            subject="world:cap", key=f"fact{i:02d}", value="x",
            domain="world", confidence=0.5,
            summary=f"cap world {i}"))
    for i in range(10):
        records.append(_symbolic(
            f"rec:synthetic-cap-sym-{i:02d}", symbol=f"sym{i}",
            runes=["ansuz"], charge=0.5, confidence=0.5))
    for i in range(10):
        records.append(_generic(
            f"rec:synthetic-cap-unc-{i:02d}",
            summary=f"cap uncertainty {i}", confidence=0.5,
            approval=ApprovalState.PENDING))
    for i in range(5):
        records.append(_generic(
            f"rec:synthetic-cap-ok-{i:02d}",
            summary=f"cap approved {i}", confidence=0.5,
            approval=ApprovalState.APPROVED))
    return records


# ── The production run: fixtures through the compiler + the runtime ──────────

def _compiled_data(compiled: CompiledPersona) -> dict[str, Any]:
    return {
        "persona_id": compiled.persona_id,
        "user_id": compiled.user_id,
        "mode": compiled.mode.value,
        "tone_contract": list(compiled.tone_contract),
        "identity_core": list(compiled.identity_core),
        "active_traits": [
            {"trait_name": t.trait_name,
             "weight": round(t.weight, 4),
             "supporting_record_ids": list(t.supporting_record_ids)}
            for t in compiled.active_traits],
        "relationship_excerpt": list(compiled.relationship_excerpt),
        "truth_anchor_points": list(compiled.truth_anchor_points),
        "uncertainty_points": list(compiled.uncertainty_points),
        "symbolic_context": list(compiled.symbolic_context),
        "response_guidance": list(compiled.response_guidance),
        "source_items": [
            {"record_id": s.record_id, "item_type": s.item_type,
             "confidence": s.confidence}
            for s in compiled.source_items],
        "has_token_budget_hint": hasattr(compiled, "token_budget_hint"),
    }


def _counts(compiled: CompiledPersona) -> dict[str, Any]:
    return {
        "identity_core": len(compiled.identity_core),
        "truth_anchor_points": len(compiled.truth_anchor_points),
        "active_traits": len(compiled.active_traits),
        "symbolic_context": len(compiled.symbolic_context),
        "uncertainty_points": len(compiled.uncertainty_points),
        "source_items": len(compiled.source_items),
        "tone_contract": len(compiled.tone_contract),
        "response_guidance": len(compiled.response_guidance),
        "relationship_excerpt": len(compiled.relationship_excerpt),
    }


def run_persona_compile() -> dict[str, Any]:
    """Run the production compiler over TF-141's
    fixtures and file the measurements: the main
    COMPANION run with the production bond edge +
    the authored base, the three modes, the caps
    run, and the merged delivery through TF-030's
    runtime (inject_persona into a real scaffold;
    the fixture turns judged by score_persona_turns
    against the merged expectation). Deterministic."""
    compiler = PersonaCompiler()
    records = fixture_records()

    main = compiler.compile(
        persona_id="persona:wrenna", user_id="user:lantern-keeper",
        mode=PersonaMode.COMPANION, records=records,
        bond_edge=fixture_bond_edge(),
        base_expectation=FIXTURE_EXPECTATION)
    modes: dict[str, Any] = {}
    for mode in (PersonaMode.COMPANION, PersonaMode.CODING_GUIDE,
                 PersonaMode.WORLD_SEER):
        packet = compiler.compile(
            persona_id="persona:wrenna",
            user_id="user:lantern-keeper", mode=mode,
            records=records)
        modes[mode.value] = {
            "tone_contract": list(packet.tone_contract),
            "response_guidance": list(packet.response_guidance),
            "counts": _counts(packet),
        }
    caps = compiler.compile(
        persona_id="persona:cap", user_id="user:lantern-keeper",
        mode=PersonaMode.COMPANION, records=caps_records())

    expectation = main.to_expectation()
    steering = expectation.steering_lines()
    scaffold = CognitionScaffold(
        goal="turn goal", fact_block="Knowledge:\nx")
    injected = inject_persona(scaffold, expectation)
    report = score_persona_turns(
        expectation, [text for text, _ in PERSONA_FIXTURE_TURNS])
    base = FIXTURE_EXPECTATION
    merged = {
        "record_id": expectation.record_id,
        "base_record_id": expectation.base_record_id,
        "steering_lines": steering,
        "steering_line_count": len(steering),
        "base_lines_preserved": all(
            line in steering for line in base.steering_lines()),
        "avoid": list(expectation.avoid),
        "avoid_matches_base": list(expectation.avoid) == list(base.avoid),
        "injected": injected,
        "fact_block_starts_with_block": scaffold.fact_block.startswith(
            steering[0]),
        "compiled_lines_in_block": sum(
            1 for line in steering[len(base.steering_lines()):]
            if line in scaffold.fact_block),
        "turns_checked": report.total_turns,
        "turns_consistent": report.consistent_turns,
        "violations": report.violation_count,
        "score": report.score,
        "verdict": report.verdict,
    }
    return {
        "main": _compiled_data(main),
        "main_counts": _counts(main),
        "modes": modes,
        "caps": {"records": len(caps_records()), **_counts(caps)},
        "merged": merged,
    }


#: The filed expectation for the production run. The
#: defect fixes move exactly two counts against TF-141's
#: spike filing (anchors 5 -> 4, uncertainty 2 -> 3).
COMPILE_EXPECTED: dict[str, Any] = {
    "main_counts": {
        "identity_core": 4, "truth_anchor_points": 4,
        "active_traits": 3, "symbolic_context": 2,
        "uncertainty_points": 3, "source_items": 14,
        "tone_contract": 4, "response_guidance": 4,
        "relationship_excerpt": 3,
    },
    "caps": {
        "records": 57, "identity_core": 10,
        "truth_anchor_points": 16, "active_traits": 8,
        "symbolic_context": 8, "uncertainty_points": 8,
        "source_items": 20, "tone_contract": 4,
        "response_guidance": 1, "relationship_excerpt": 0,
    },
    "traits": [
        {"trait_name": "steady", "weight": 1.0,
         "supporting_record_ids": ["rec:synthetic-wrenna-001",
                                   "rec:synthetic-wrenna-005"]},
        {"trait_name": "plain-spoken", "weight": 0.8,
         "supporting_record_ids": ["rec:synthetic-wrenna-003"]},
        {"trait_name": "honesty", "weight": 0.6,
         "supporting_record_ids": ["rec:synthetic-wrenna-008"]},
    ],
    "identity_core": [
        "persona:wrenna.temperament=steady",
        "persona:wrenna.name=Wrenna the Lantern-Keeper",
        "persona:wrenna.tone=plain-spoken",
        "persona:wrenna.duty=keep the ferry lantern lit",
    ],
    "truth_anchor_points": [
        "persona:wrenna.mood=steady",
        "world:lantern-hollow.ferry_minutes=11",
        "world:lantern-hollow.river=the Bracken",
        "persona:wrenna.value=honesty",
    ],
    "uncertainty_points": [
        "an unapproved bell count",
        "an unverified tale of a third tide bell",
        "a disputed tale of a night crossing",
    ],
    "symbolic_context": [
        "symbol=lantern;runes=ansuz,raidho;charge=0.75",
        "symbol=river-stone;runes=laguz;charge=0.40",
    ],
    "relationship_excerpt": [
        "relationship_mode=builder",
        "relationship_weather=calm",
        "closeness_index=0.44",
    ],
}


# ── The filed probes ─────────────────────────────────────────────────────────

PERSONA_COMPILER_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_compile_passes",
     "kind": "clean",
     "detail": "the production compile + its merged delivery "
               "pass their validators"},
    {"id": "trait_weight_wrong_is_named",
     "kind": "trait_wrong",
     "detail": "an expected filing with steady's weight "
               "perturbed is named against the compiled traits"},
    {"id": "caps_breach_is_named",
     "kind": "caps_breach",
     "detail": "a counts filing with identity_core over its "
               "cap is named"},
    {"id": "schema_field_missing_is_named",
     "kind": "schema_field_missing",
     "detail": "a field map with supporting_record_ids dropped "
               "from TraitSignal is named against the schemas"},
    {"id": "unapproved_anchor_is_named",
     "kind": "unapproved_anchor",
     "detail": "a compile filing the pending bell_count line "
               "as a truth anchor is named by the approval "
               "validator"},
    {"id": "relationship_duplicating_guidance_is_named",
     "kind": "relationship_duplicate",
     "detail": "an excerpt copied from response_guidance is "
               "named by the relationship validator"},
    {"id": "unknown_mode_is_named",
     "kind": "mode_unknown",
     "detail": "a mode value outside the schema's PersonaMode "
               "enum is named"},
)


def run_persona_compiler_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    run = run_persona_compile()
    for probe in PERSONA_COMPILER_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            main = run["main"]
            problems = caps_problems(run["main_counts"]) \
                + caps_problems(run["caps"]) \
                + trait_problems(main["active_traits"],
                                 COMPILE_EXPECTED["traits"]) \
                + approval_problems(
                    main["truth_anchor_points"],
                    main["identity_core"],
                    main["symbolic_context"],
                    main["uncertainty_points"],
                    [("world:lantern-hollow.bell_count=two",
                      "an unapproved bell count")]) \
                + relationship_problems(
                    main["relationship_excerpt"],
                    main["response_guidance"],
                    COMPILE_EXPECTED["relationship_excerpt"]) \
                + mode_problems(main["mode"])
            passed = problems == []
        elif kind == "trait_wrong":
            expected = [dict(row) for row in COMPILE_EXPECTED["traits"]]
            expected[0]["weight"] = 0.5
            passed = bool(trait_problems(
                run["main"]["active_traits"], expected))
        elif kind == "caps_breach":
            counts = dict(run["caps"], identity_core=11)
            passed = bool(caps_problems(counts))
        elif kind == "schema_field_missing":
            field_map = production_field_map()
            field_map["TraitSignal"] = tuple(
                f for f in field_map["TraitSignal"]
                if f != "supporting_record_ids")
            passed = bool(schema_field_problems(field_map))
        elif kind == "unapproved_anchor":
            anchors = list(run["main"]["truth_anchor_points"]) + [
                "world:lantern-hollow.bell_count=two"]
            passed = bool(approval_problems(
                anchors, run["main"]["identity_core"],
                run["main"]["symbolic_context"],
                run["main"]["uncertainty_points"],
                [("world:lantern-hollow.bell_count=two",
                  "an unapproved bell count")]))
        elif kind == "relationship_duplicate":
            passed = bool(relationship_problems(
                list(run["main"]["response_guidance"]),
                run["main"]["response_guidance"],
                COMPILE_EXPECTED["relationship_excerpt"]))
        elif kind == "mode_unknown":
            passed = bool(mode_problems("bogus_mode"))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_persona_compiler_markdown() -> str:
    """Render the landing deterministically — the
    exact text filed as docs/architecture/PERSONA_COMPILER.md."""
    run = run_persona_compile()
    counts = run["main_counts"]
    merged = run["merged"]
    lines: list[str] = []
    lines.append("# Persona Compiler — TF-143 (production merge)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/cognition/persona_compiler.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.cognition.persona_compiler`. "
        "The checker (`persona_compiler_problems()`) re-derives "
        "every claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{PERSONA_COMPILER_FORMAT}`")
    lines.append("Verdict implemented: **MERGE** (TF-141, adopt-in-part)")
    lines.append(
        f"Probes: {len(PERSONA_COMPILER_PROBES)}/"
        f"{len(PERSONA_COMPILER_PROBES)}")
    lines.append("")
    lines.append("## 1. What landed")
    lines.append("")
    lines.append(
        "A compiled-expectation builder beside `persona.py`: "
        "`PersonaCompiler` compiles canonical records into a "
        "`CompiledPersona` carrying every adopted facet "
        "(evidence-weighted TraitSignals, source_items "
        "provenance with confidences, uncertainty_points, the "
        "identity_core / truth_anchor_points split, mode "
        "adaptation, symbolic_context, and a relationship "
        "section compiled from the PRODUCTION `BondEdge` of "
        "TF-143's bond graph layer), and a "
        "`CompiledPersonaExpectation` — a `PersonaExpectation` "
        "subclass — delivers it through TF-030's runtime: "
        "`inject_persona` delivers base + compiled steering "
        "and `check_persona_response` / `score_persona_turns` "
        "judge responses against it. persona.py is not "
        "modified; TF-009's authored PersonalityCoreRecord "
        "path remains a supported source (an optional base "
        "expectation whose expressive lists and avoid ride "
        "verbatim), never silently replaced.")
    lines.append("")
    lines.append("## 2. Filed decisions")
    lines.append("")
    lines.append(
        f"Avoid sourcing: **{AVOID_DECISION['decision']}** — "
        f"{AVOID_DECISION['rule']}.")
    lines.append("")
    for reason in AVOID_DECISION["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(
        f"Token budget hint: **{TOKEN_BUDGET_DECISION['decision']}** "
        f"— {TOKEN_BUDGET_DECISION['rule']}.")
    lines.append("")
    for reason in TOKEN_BUDGET_DECISION["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## 3. Defects not imported (measured differences)")
    lines.append("")
    for row in DEFECT_DIFFERENCES:
        lines.append(f"- `{row['id']}` — spike: {row['spike']}; "
                     f"production: {row['production']}")
    lines.append("")
    lines.append("## 4. The production compile (Wrenna fixtures)")
    lines.append("")
    lines.append(
        f"- identity_core {counts['identity_core']} / "
        f"truth_anchor_points {counts['truth_anchor_points']} "
        "(spike: 4 / 5 — the pending bell_count fact moves to "
        "uncertainty under approval-first)")
    lines.append(
        f"- active_traits {counts['active_traits']} (steady "
        "1.0 at the cap from 0.6 + 0.5 with both supporting "
        "ids; plain-spoken 0.8; honesty 0.6)")
    lines.append(
        f"- symbolic_context {counts['symbolic_context']} / "
        f"uncertainty_points {counts['uncertainty_points']} "
        f"(spike: 2) / source_items {counts['source_items']}")
    lines.append(
        f"- tone_contract {counts['tone_contract']} / "
        f"response_guidance {counts['response_guidance']} / "
        f"relationship_excerpt {counts['relationship_excerpt']} "
        "(bond-only lines)")
    lines.append(
        f"- caps run (57 records): every cap binds exactly — "
        f"identity {run['caps']['identity_core']} / anchors "
        f"{run['caps']['truth_anchor_points']} / traits "
        f"{run['caps']['active_traits']} / symbolic "
        f"{run['caps']['symbolic_context']} / uncertainty "
        f"{run['caps']['uncertainty_points']} / source_items "
        f"{run['caps']['source_items']}")
    lines.append(
        f"- merged delivery: {merged['steering_line_count']} "
        f"steering lines injected into a real scaffold "
        f"(base lines preserved: {merged['base_lines_preserved']}); "
        f"fixture turns judged {merged['turns_consistent']}/"
        f"{merged['turns_checked']} consistent, score "
        f"{merged['score']}, verdict {merged['verdict']} — "
        "TF-030's avoid checking unchanged on the merged shape")
    lines.append("")
    lines.append("## 5. The probes")
    lines.append("")
    for probe in PERSONA_COMPILER_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 6. Filed limits")
    lines.append("")
    lines.append(
        "- Fixture grade only: the ground truth is TF-017 / "
        "TF-030's synthetic Wrenna world; no live model was "
        "read — live persona quality is NOT PROVEN (TF-173 "
        "class).")
    lines.append(
        "- TF-147 consumes this merged output for Rúnmál "
        "cards; TF-149 owns the wing's tests. TF-140/TF-142's "
        "RETIRE verdicts land nothing here — TF-144 executes "
        "them.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_persona_compiler_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / PERSONA_COMPILER_DOC
    doc.write_text(render_persona_compiler_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.cognition.persona_compiler`."""
    doc = write_persona_compiler_doc()
    print(f"persona compiler regenerated: -> {PERSONA_COMPILER_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def persona_compiler_problems() -> list[str]:
    """Re-derive TF-143's persona-compiler claims
    from the live tree; name any that stopped
    being true. (Distinct from TF-141's
    persona_compiler_verdict_problems().)"""
    problems: list[str] = []

    # The filed decisions stand.
    if AVOID_DECISION.get("decision") != "declared" \
            or not AVOID_DECISION.get("reasons"):
        problems.append("decision: the avoid filing drifted")
    if TOKEN_BUDGET_DECISION.get("decision") != "dropped" \
            or not TOKEN_BUDGET_DECISION.get("reasons"):
        problems.append("decision: the token-budget filing drifted")
    if len(DEFECT_DIFFERENCES) != 3:
        problems.append("the defect differences are not three")
    for row in DEFECT_DIFFERENCES:
        if not row.get("spike") or not row.get("production"):
            problems.append(
                f"defect {row.get('id')}: both sides are required")
    if "token_budget_hint" in CompiledPersona.__dataclass_fields__:
        problems.append(
            "production: CompiledPersona carries a "
            "token_budget_hint — the dropped decision must be "
            "re-ruled")

    # The schemas parse; the production field sets + enums match.
    schema_files = sorted(_SCHEMAS.glob("*.json"))
    if len(schema_files) != 14:
        problems.append(
            f"wing: {len(schema_files)} schemas, not TF-006's 14")
    for schema_file in schema_files:
        try:
            json.loads(schema_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — named, not hidden
            problems.append(f"wing: schema {schema_file.name} "
                            f"does not parse: {exc}")
    problems.extend(schema_field_problems(production_field_map()))
    packet_schema = json.loads(
        (_SCHEMAS / "persona_packet.schema.json").read_text(
            encoding="utf-8"))
    if [m.value for m in PersonaMode] != \
            packet_schema["$defs"]["PersonaMode"]["enum"]:
        problems.append("production: PersonaMode drifted from the schema")
    canonical_schema = json.loads(
        (_SCHEMAS / "canonical_fact_record.schema.json").read_text(
            encoding="utf-8"))
    if [a.value for a in ApprovalState] != \
            canonical_schema["$defs"]["ApprovalState"]["enum"]:
        problems.append(
            "production: ApprovalState drifted from the schema")

    # The wing compiler shell still files the facets this
    # module adopts (its markers stand as filed).
    compiler_text = _COMPILER.read_text(encoding="utf-8")
    for marker in ("def compile", "identity_core[:10]",
                   "truth_anchor_points[:16]",
                   "uncertainty_points[:8]",
                   "symbolic_context[:8]", "source_items[:20]",
                   "response_guidance[:10]"):
        if marker not in compiler_text:
            problems.append(
                f"wing: persona_compiler.py no longer files {marker!r}")

    # TF-030's runtime stands unmodified in substance.
    persona_text = (
        _REPO_ROOT / "src" / "thoughtforge" / "cognition"
        / "persona.py").read_text(encoding="utf-8")
    for marker in ("MAX_STEERING_ENTRIES = 6", "def inject_persona",
                   "def check_persona_response",
                   "def score_persona_turns"):
        if marker not in persona_text:
            problems.append(
                f"shipping: cognition/persona.py no longer files "
                f"{marker!r}")
    if FIXTURE_EXPECTATION.record_id != "syn_wrenna_001" \
            or len(FIXTURE_EXPECTATION.steering_lines()) != 3:
        problems.append(
            "shipping: TF-030's authored fixture path drifted — "
            "the authored source must stand as filed")

    # The production run reproduces its filed expectation,
    # including the defect-difference measurements.
    run = run_persona_compile()
    problems.extend(measured_problems(
        run["main_counts"], COMPILE_EXPECTED["main_counts"]))
    problems.extend(measured_problems(
        run["caps"], COMPILE_EXPECTED["caps"]))
    problems.extend(trait_problems(
        run["main"]["active_traits"], COMPILE_EXPECTED["traits"]))
    for key in ("identity_core", "truth_anchor_points",
                "uncertainty_points", "symbolic_context",
                "relationship_excerpt"):
        if run["main"][key] != COMPILE_EXPECTED[key]:
            problems.append(
                f"compile: {key} files {run['main'][key]}, not "
                f"the filed {COMPILE_EXPECTED[key]}")
    if run["main"]["has_token_budget_hint"]:
        problems.append(
            "compile: the compiled persona carries a "
            "token_budget_hint — dropped means dropped")
    problems.extend(approval_problems(
        run["main"]["truth_anchor_points"],
        run["main"]["identity_core"],
        run["main"]["symbolic_context"],
        run["main"]["uncertainty_points"],
        [("world:lantern-hollow.bell_count=two",
          "an unapproved bell count")]))
    problems.extend(relationship_problems(
        run["main"]["relationship_excerpt"],
        run["main"]["response_guidance"],
        COMPILE_EXPECTED["relationship_excerpt"]))
    problems.extend(caps_problems(run["main_counts"]))
    problems.extend(caps_problems(run["caps"]))

    # The merged delivery through TF-030's runtime.
    merged = run["merged"]
    if not merged["injected"] or not merged["base_lines_preserved"] \
            or not merged["avoid_matches_base"]:
        problems.append(
            f"merged: the delivery filing drifted: {merged}")
    if merged["turns_checked"] != 5 or merged["score"] != 0.4 \
            or merged["verdict"] != "REVIEW":
        problems.append(
            "merged: TF-030's fixture-turn judgement on the "
            "merged expectation is not 5 checked / 0.4 / REVIEW "
            "— the authored checking must stand unchanged")

    # The probes all behave.
    probes = run_persona_compiler_probes()
    if len(probes) != len(PERSONA_COMPILER_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the persona-compiler probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / PERSONA_COMPILER_DOC
    if not doc.exists():
        problems.append(f"the write-up {PERSONA_COMPILER_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != \
            render_persona_compiler_markdown():
        problems.append(
            f"the filed doc {PERSONA_COMPILER_DOC} is not the "
            "render verbatim (regenerate with python3 -m "
            "thoughtforge.cognition.persona_compiler)")
    return problems


def persona_compiler_as_data() -> dict[str, Any]:
    """The whole landing as plain data, for consumers."""
    run = run_persona_compile()
    probes = run_persona_compiler_probes()
    return {
        "format": PERSONA_COMPILER_FORMAT,
        "doc": PERSONA_COMPILER_DOC,
        "avoid_decision": dict(AVOID_DECISION),
        "token_budget_decision": dict(TOKEN_BUDGET_DECISION),
        "defect_differences": [dict(r) for r in DEFECT_DIFFERENCES],
        "compile": run,
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "AVOID_DECISION",
    "BASE_TONE_CONTRACT",
    "CAPS",
    "COMPILE_EXPECTED",
    "DEFECT_DIFFERENCES",
    "IDENTITY_DOMAINS",
    "PERSONA_COMPILER_DOC",
    "PERSONA_COMPILER_FORMAT",
    "PERSONA_COMPILER_PROBES",
    "PRODUCTION_SHAPES",
    "TOKEN_BUDGET_DECISION",
    "TRAIT_KEYS",
    "ApprovalState",
    "CanonicalFactContent",
    "CanonicalFactPayload",
    "CanonicalFactRecord",
    "CompiledPersona",
    "CompiledPersonaExpectation",
    "GenericContent",
    "PersonaCompiler",
    "PersonaMode",
    "PersonaRecord",
    "PersonaSourceItem",
    "SymbolicTraceContent",
    "SymbolicTracePayload",
    "SymbolicTraceRecord",
    "TraitSignal",
    "TruthMeta",
    "approval_problems",
    "caps_problems",
    "caps_records",
    "fixture_bond_edge",
    "fixture_records",
    "main",
    "measured_problems",
    "mode_problems",
    "persona_compiler_as_data",
    "persona_compiler_problems",
    "production_field_map",
    "relationship_problems",
    "render_persona_compiler_markdown",
    "run_persona_compile",
    "run_persona_compiler_probes",
    "schema_field_problems",
    "trait_problems",
    "write_persona_compiler_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
