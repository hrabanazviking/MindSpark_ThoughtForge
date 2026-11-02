"""
TF-143 — the bond graph layer, as shipping production code.

TF-139's verdict was **ADOPT_AS_LAYER**: wyrdforge's bond graph
stands BESIDE the shipping knowledge store as a layer of its
own — it answers "how entities stand with each other" (edges
and their state) where the store answers "what is known"
(records and world knowledge). This module is that adoption,
implemented under TF-139's filed conditions:

  * **Production dataclasses rebuilt from the schemas** —
    ``BondEdge`` / ``Vow`` / ``Hurt`` plus ``BondVector``,
    ``BondScars``, ``BondChronology``, ``BondEvidence``,
    ``BondConstraints``, ``BondActiveModes`` and
    ``BondGovernance``, with the enums ``BondDomain`` /
    ``BondStatus`` / ``VowState`` exactly as
    ``bond_edge.schema.json`` / ``vow.schema.json`` /
    ``hurt.schema.json`` file them. These are production
    names in the shipping tree — NOT the spike's ``Spike*``
    names, NOT the research wing's missing
    ``wyrdforge.models`` package (that package is NOT
    resurrected by this slice; the wing stays as-is for
    TF-148/TF-149, as TF-143's filed reading states).
  * **A production service faithful to the wing shell** —
    :class:`BondGraph` re-implements ``BondGraphService``'s
    operations (``add_edge`` / ``add_vow`` / ``add_hurt`` /
    ``apply_event`` / ``excerpt``) with the shell's exact
    arithmetic: severity shifts {low 0.1, medium 0.2,
    high 0.35, mythic 0.5}, ``repair_debt = min(1, debt +
    shift)``, ``safety = max(0, safety - shift/2)``, status
    -> REPAIRING on a hurt, vector dims clamped 0..1,
    status -> ACTIVE at trust >= 0.55 and warmth >= 0.55,
    and the excerpt's filed line shapes.
  * **A layer, not a second store** — the graph keeps its
    edges in memory exactly as the shell does; bond evidence
    (``BondEvidence.supporting_record_ids`` /
    ``contradiction_record_ids``) REFERENCES shipping-store
    record ids, and ``apply_event`` appends the source
    record id. No fact is duplicated: the records themselves
    stay in the store, the layer holds only relational state.
  * **The weighting decision, filed with evidence** —
    BondGraphSpec.md files the derived indices as weighted
    means (closeness over trust/warmth/familiarity/
    vulnerability, sacred over devotion/sacred_resonance/
    inverse vow_strain, rupture over repair_debt/vow_strain/
    inverse safety) but NOWHERE files the weights: the spec
    text was re-read for this slice and carries no numeric
    weight for any index term (its only weight tables are
    the *event* weights, a different mechanism the service
    shell does not use for indices). TF-139's spike used
    equal weights as a filed assumption. The production
    decision (``WEIGHTING_DECISION`` below) is therefore
    EQUAL WEIGHTS: the only choice the spec's silence plus
    the spike's determinism supports — any unequal weighting
    would be an invented tuning with no evidence behind it,
    and equal weights reproduce TF-139's filed spike metrics
    exactly, keeping the verdict's measurement as the
    production baseline. The decision's LIMIT is filed with
    it: equal weights are a default under silence, not a
    measured optimum; a future slice with real evidence may
    re-rule them, and this module's checker pins the filed
    decision so any silent drift is named.

Parity, measured live: :func:`run_bond_parity` runs the
production service over TF-139's synthetic Emberwick
fixtures (TF-017 — invented entities only, never canon)
and reproduces the spike's filed metrics exactly: status
forming -> active -> repairing, repair_debt 0.35 /
safety 0.725 after a high hurt, closeness 0.4375 /
sacred 0.5333 / rupture 0.2083 under the filed equal
weights, and the 9-line excerpt verbatim.

RETIRE verdicts file nothing here: TF-140's micro-RAG and
TF-142's truth calibrator are NOT rebuilt by this slice —
TF-144 executes their retirements. This module imports
none of their defects (nor any wing code at all).

The human-readable write-up is
``docs/architecture/BOND_GRAPH.md``, filed verbatim as
this module's render (regenerate with
``python3 -m thoughtforge.knowledge.bond_graph``).
Proof grade: FIXTURE/STATIC — schemas + the wing service
shell + synthetic fixtures re-measured live on this VM;
no live model, never claimed. Whether a live system
honours a bond excerpt is NOT PROVEN (TF-173 class).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

BOND_GRAPH_FORMAT = "thoughtforge-bond-graph/1"
BOND_GRAPH_DOC = "docs/architecture/BOND_GRAPH.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_SERVICE = _WING / "services" / "bond_graph_service.py"
_SCHEMAS = _WING / "schemas"
_SPEC = _REPO_ROOT / "research_data" / "BondGraphSpec.md"


# ── The weighting decision (TF-139's condition, discharged) ──────────────────

WEIGHTING_DECISION: dict[str, Any] = {
    "decision": "equal_weights",
    "indices": {
        "closeness": ("trust", "warmth", "familiarity", "vulnerability"),
        "sacred": ("devotion", "sacred_resonance", "inverse_vow_strain"),
        "rupture": ("repair_debt", "vow_strain", "inverse_safety"),
    },
    "weights": {"closeness": 0.25, "sacred": 1 / 3, "rupture": 1 / 3},
    "evidence": (
        "BondGraphSpec.md files each index only as a weighted_mean "
        "over its terms — re-read for TF-143, the spec carries no "
        "numeric weight for any index term (its weight tables are "
        "event_weights for event ingestion, a different mechanism "
        "the service shell does not use for indices)",
        "the wing's BondGraphService computes no index itself — it "
        "calls edge.closeness_index() etc. on the missing models, "
        "so no implementation weighting exists to inherit",
        "TF-139's spike used equal weights as a filed assumption "
        "and filed deterministic metrics under them (closeness "
        "0.4375 / sacred 0.5333 / rupture 0.2083 on the Emberwick "
        "fixtures); equal weights reproduce those metrics exactly, "
        "keeping the verdict's measurement as the production "
        "baseline instead of silently re-basing it",
    ),
    "limits": (
        "equal weights are a default under spec silence, NOT a "
        "measured optimum — no evidence in the estate ranks one "
        "index term above another",
        "a future slice with real evidence (a measured corpus, a "
        "spec amendment) may re-rule the weights; until then any "
        "unequal weighting would be invented tuning",
    ),
}


def weighting_problems(decision: Any) -> list[str]:
    """Judge one weighting filing, purely: the
    equal-weights decision, its term sets, at
    least one evidence line and one limit.
    Empty-safe."""
    problems: list[str] = []
    if not isinstance(decision, dict) or not decision:
        return ["weighting: the filing is empty"]
    if decision.get("decision") != "equal_weights":
        problems.append(
            f"weighting: the decision is "
            f"{decision.get('decision')!r}, not 'equal_weights'")
    indices = decision.get("indices") or {}
    expected = {
        "closeness": ("trust", "warmth", "familiarity", "vulnerability"),
        "sacred": ("devotion", "sacred_resonance", "inverse_vow_strain"),
        "rupture": ("repair_debt", "vow_strain", "inverse_safety"),
    }
    for name, terms in expected.items():
        if tuple(indices.get(name) or ()) != terms:
            problems.append(
                f"weighting: {name} files terms "
                f"{indices.get(name)!r}, not the spec's {terms}")
    weights = decision.get("weights") or {}
    for name, terms in expected.items():
        if weights.get(name) != 1 / len(terms):
            problems.append(
                f"weighting: {name} weight is {weights.get(name)!r}, "
                f"not the equal weight {1 / len(terms)}")
    for key in ("evidence", "limits"):
        rows = decision.get(key) or ()
        if not rows or not all(isinstance(r, str) and r for r in rows):
            problems.append(f"weighting: the filing carries no {key}")
    return problems


# ── Pure validators (the probes run against these) ───────────────────────────

#: The service shell's hurt severity shifts, exactly as filed
#: in bond_graph_service.py (the checker re-reads them live).
SEVERITY_SHIFTS: dict[str, float] = {
    "low": 0.1, "medium": 0.2, "high": 0.35, "mythic": 0.5,
}


def severity_map_problems(mapping: Any) -> list[str]:
    """Judge one severity map against the filed
    service shifts, purely. Empty-safe."""
    if not isinstance(mapping, dict) or not mapping:
        return ["severity: the map is empty"]
    problems: list[str] = []
    for name, shift in SEVERITY_SHIFTS.items():
        if mapping.get(name) != shift:
            problems.append(
                f"severity: {name} shift is {mapping.get(name)!r}, "
                f"not the filed {shift}")
    for name in mapping:
        if name not in SEVERITY_SHIFTS:
            problems.append(f"severity: unknown severity {name!r}")
    return problems


def index_problems(
    closeness: float, sacred: float, rupture: float,
    expected: Any,
) -> list[str]:
    """Judge computed derived indices against an
    expected filing (2dp, as the excerpt files
    them), purely. Empty-safe."""
    if not isinstance(expected, dict) or not expected:
        return ["indices: the expected filing is empty"]
    problems: list[str] = []
    computed = {
        "closeness": round(closeness, 2),
        "sacred": round(sacred, 2),
        "rupture": round(rupture, 2),
    }
    for name, value in computed.items():
        if expected.get(name) != value:
            problems.append(
                f"indices: {name} computes {value}, not the filed "
                f"{expected.get(name)!r}")
    return problems


def schema_field_problems(field_map: Any) -> list[str]:
    """Judge a {shape: fields} map against the
    three schemas parsed live from the wing.
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
    """The field sets of the three schemas (top
    level + $defs sub-shapes), parsed live."""
    edge = json.loads(
        (_SCHEMAS / "bond_edge.schema.json").read_text(encoding="utf-8"))
    vow = json.loads(
        (_SCHEMAS / "vow.schema.json").read_text(encoding="utf-8"))
    hurt = json.loads(
        (_SCHEMAS / "hurt.schema.json").read_text(encoding="utf-8"))
    field_map: dict[str, tuple[str, ...]] = {
        "BondEdge": tuple(edge["properties"]),
        "Vow": tuple(vow["properties"]),
        "Hurt": tuple(hurt["properties"]),
    }
    for name, shape in edge["$defs"].items():
        if "properties" in shape:
            field_map[name] = tuple(shape["properties"])
    return field_map


def edge_problems(edge: Any) -> list[str]:
    """Judge one production edge's filed data,
    purely: ids present, vector dims in range
    (0..1; initiative_balance -1..1), scar
    values in range. Empty-safe — accepts a
    BondEdge or a plain dict of its fields."""
    if edge is None:
        return ["edge: the edge is empty"]
    data = dataclasses.asdict(edge) if dataclasses.is_dataclass(edge) \
        else edge
    if not isinstance(data, dict) or not data:
        return ["edge: the edge is empty"]
    problems: list[str] = []
    for key in ("bond_id", "entity_a", "entity_b"):
        if not data.get(key):
            problems.append(f"edge: {key} is missing")
    vector = data.get("vector") or {}
    for name, value in vector.items():
        if not isinstance(value, (int, float)):
            problems.append(f"edge: vector {name} is not a number")
            continue
        low = -1.0 if name == "initiative_balance" else 0.0
        if not low <= value <= 1.0:
            problems.append(
                f"edge: vector {name} is {value}, outside "
                f"{low}..1")
    scars = data.get("scars") or {}
    for name in ("repair_debt", "vow_strain"):
        value = scars.get(name)
        if isinstance(value, (int, float)) and not 0.0 <= value <= 1.0:
            problems.append(f"edge: scars {name} is {value}, outside 0..1")
    return problems


def hurt_problems(hurt: Any) -> list[str]:
    """Judge one production hurt's filed data,
    purely: ids present, severity one of the
    filed shifts. Empty-safe."""
    if hurt is None:
        return ["hurt: the hurt is empty"]
    data = dataclasses.asdict(hurt) if dataclasses.is_dataclass(hurt) \
        else hurt
    if not isinstance(data, dict) or not data:
        return ["hurt: the hurt is empty"]
    problems: list[str] = []
    for key in ("hurt_id", "bond_id", "source_event_id", "hurt_kind"):
        if not data.get(key):
            problems.append(f"hurt: {key} is missing")
    if data.get("severity") not in SEVERITY_SHIFTS:
        problems.append(
            f"hurt: severity {data.get('severity')!r} is not one of "
            f"{tuple(SEVERITY_SHIFTS)}")
    return problems


# ── Production dataclasses, rebuilt from the three schemas ───────────────────

class BondDomain(str, Enum):
    COMPANION = "companion"
    ROMANCE = "romance"
    FRIENDSHIP = "friendship"
    MENTOR = "mentor"
    BUILDER = "builder"
    RITUAL = "ritual"
    PARTY_MEMBER = "party_member"
    DEITY_DEVOTEE = "deity_devotee"


class BondStatus(str, Enum):
    FORMING = "forming"
    ACTIVE = "active"
    STRAINED = "strained"
    REPAIRING = "repairing"
    DORMANT = "dormant"
    BROKEN = "broken"
    TRANSFIGURED = "transfigured"


class VowState(str, Enum):
    PENDING = "pending"
    KEPT = "kept"
    STRAINED = "strained"
    BROKEN = "broken"
    RELEASED = "released"


@dataclass
class BondVector:
    trust: float = 0.5
    warmth: float = 0.5
    familiarity: float = 0.1
    devotion: float = 0.0
    attraction_affinity: float = 0.0
    safety: float = 0.8
    sacred_resonance: float = 0.0
    playfulness: float = 0.2
    vulnerability: float = 0.1
    initiative_balance: float = 0.0


@dataclass
class BondConstraints:
    exclusivity_mode: str = "none"
    intimacy_ceiling: str = "medium"
    boundary_profile_id: str = "default"


@dataclass
class BondScars:
    repair_debt: float = 0.0
    unresolved_hurts: list[str] = field(default_factory=list)
    vow_strain: float = 0.0


@dataclass
class BondActiveModes:
    relational_mode: str = "builder"
    emotional_weather: str = "calm"


@dataclass
class BondChronology:
    formed_at: Any = None
    last_major_shift_at: Any = None
    last_contact_at: Any = None


@dataclass
class BondEvidence:
    supporting_record_ids: list[str] = field(default_factory=list)
    contradiction_record_ids: list[str] = field(default_factory=list)


@dataclass
class BondGovernance:
    review_required_for_large_shift: bool = True
    synthetic_claim_guard: bool = True


@dataclass
class BondEdge:
    bond_id: str
    entity_a: str
    entity_b: str
    domain: BondDomain
    status: BondStatus = BondStatus.FORMING
    vector: BondVector = field(default_factory=BondVector)
    constraints: BondConstraints = field(default_factory=BondConstraints)
    scars: BondScars = field(default_factory=BondScars)
    active_modes: BondActiveModes = field(default_factory=BondActiveModes)
    chronology: BondChronology = field(default_factory=BondChronology)
    evidence: BondEvidence = field(default_factory=BondEvidence)
    governance: BondGovernance = field(default_factory=BondGovernance)

    def closeness_index(self) -> float:
        """Weighted mean per BondGraphSpec.md over
        (trust, warmth, familiarity, vulnerability)
        — EQUAL weights (WEIGHTING_DECISION)."""
        v = self.vector
        return (v.trust + v.warmth + v.familiarity + v.vulnerability) / 4

    def sacred_bond_index(self) -> float:
        """Weighted mean over (devotion,
        sacred_resonance, inverse vow_strain) —
        EQUAL weights (WEIGHTING_DECISION)."""
        v = self.vector
        return (v.devotion + v.sacred_resonance
                + (1.0 - self.scars.vow_strain)) / 3

    def rupture_index(self) -> float:
        """Weighted mean over (repair_debt,
        vow_strain, inverse safety) — EQUAL
        weights (WEIGHTING_DECISION)."""
        return (self.scars.repair_debt + self.scars.vow_strain
                + (1.0 - self.vector.safety)) / 3


@dataclass
class Vow:
    vow_id: str
    bond_id: str
    vow_text: str
    vow_kind: str
    created_from_record_id: str
    strength: str = "medium"
    state: VowState = VowState.PENDING
    witness_entities: list[str] = field(default_factory=list)
    kept_by_events: list[str] = field(default_factory=list)
    broken_by_events: list[str] = field(default_factory=list)


@dataclass
class Hurt:
    hurt_id: str
    bond_id: str
    source_event_id: str
    hurt_kind: str
    severity: str
    notes: str = ""


#: Production shape name -> the production dataclass, for
#: the schema-coverage checker.
PRODUCTION_SHAPES: dict[str, Any] = {
    "BondEdge": BondEdge,
    "BondVector": BondVector,
    "BondConstraints": BondConstraints,
    "BondScars": BondScars,
    "BondActiveModes": BondActiveModes,
    "BondChronology": BondChronology,
    "BondEvidence": BondEvidence,
    "BondGovernance": BondGovernance,
    "Vow": Vow,
    "Hurt": Hurt,
}


def production_field_map() -> dict[str, tuple[str, ...]]:
    """The production dataclasses' field sets, by shape."""
    return {
        name: tuple(f.name for f in dataclasses.fields(cls))
        for name, cls in PRODUCTION_SHAPES.items()
    }


# ── The production service: the wing shell's operations, faithfully ──────────

class BondGraph:
    """BondGraphService's operations over the
    production types: same stamps, same severity
    arithmetic, same clamps, same thresholds, same
    excerpt lines. The layer beside the shipping
    store — evidence references store record ids;
    no facts are duplicated here."""

    def __init__(self) -> None:
        self.edges: dict[str, BondEdge] = {}
        self.vows: dict[str, Vow] = {}
        self.hurts: dict[str, Hurt] = {}

    def add_edge(self, edge: BondEdge) -> None:
        if edge.chronology.formed_at is None:
            edge.chronology.formed_at = datetime.now(UTC)
        self.edges[edge.bond_id] = edge

    def add_vow(self, vow: Vow) -> None:
        self.vows[vow.vow_id] = vow

    def add_hurt(self, hurt: Hurt) -> None:
        self.hurts[hurt.hurt_id] = hurt
        edge = self.edges[hurt.bond_id]
        edge.scars.unresolved_hurts.append(hurt.hurt_id)
        severity_shift = SEVERITY_SHIFTS[hurt.severity]
        edge.scars.repair_debt = min(
            1.0, edge.scars.repair_debt + severity_shift)
        edge.vector.safety = max(
            0.0, edge.vector.safety - severity_shift / 2)
        edge.status = BondStatus.REPAIRING
        edge.chronology.last_major_shift_at = datetime.now(UTC)

    def apply_event(
        self, bond_id: str, *, warmth_delta: float = 0.0,
        trust_delta: float = 0.0, devotion_delta: float = 0.0,
        source_record_id: str | None = None,
    ) -> BondEdge:
        edge = self.edges[bond_id]
        edge.vector.warmth = min(1.0, max(0.0, edge.vector.warmth + warmth_delta))
        edge.vector.trust = min(1.0, max(0.0, edge.vector.trust + trust_delta))
        edge.vector.devotion = min(1.0, max(0.0, edge.vector.devotion + devotion_delta))
        edge.chronology.last_contact_at = datetime.now(UTC)
        if source_record_id:
            edge.evidence.supporting_record_ids.append(source_record_id)
        if edge.vector.trust >= 0.55 and edge.vector.warmth >= 0.55:
            edge.status = BondStatus.ACTIVE
        return edge

    def excerpt(self, bond_id: str) -> list[str]:
        edge = self.edges[bond_id]
        lines = [
            f"domain={edge.domain.value}",
            f"status={edge.status.value}",
            f"closeness_index={edge.closeness_index():.2f}",
            f"sacred_bond_index={edge.sacred_bond_index():.2f}",
            f"rupture_index={edge.rupture_index():.2f}",
            f"relational_mode={edge.active_modes.relational_mode}",
            f"emotional_weather={edge.active_modes.emotional_weather}",
        ]
        if edge.scars.unresolved_hurts:
            lines.append(
                f"unresolved_hurts={','.join(edge.scars.unresolved_hurts)}")
        related_vows = [v for v in self.vows.values() if v.bond_id == bond_id]
        if related_vows:
            lines.append("vows=" + " | ".join(
                f"{v.vow_kind}:{v.state.value}:{v.vow_text}"
                for v in related_vows))
        return lines


# ── Parity: TF-139's synthetic fixtures through the production service ───────

PARITY_BOND_ID = "bond:emberwick-keeper-companion"

#: The filed expectation for the parity run — TF-139's
#: spike filing, reproduced exactly (deterministic:
#: counts and indices only, rounded as filed).
PARITY_EXPECTED: dict[str, Any] = {
    "edges": 1, "vows": 1, "hurts": 1,
    "status_after_event": "active",
    "status_final": "repairing",
    "trust": 0.6, "warmth": 0.6,
    "repair_debt": 0.35, "safety": 0.725,
    "closeness": 0.4375, "sacred": 0.5333, "rupture": 0.2083,
    "evidence_records": 1, "unresolved_hurts": 1,
    "excerpt_lines": 9,
}

PARITY_EXPECTED_EXCERPT: tuple[str, ...] = (
    "domain=companion",
    "status=repairing",
    "closeness_index=0.44",
    "sacred_bond_index=0.53",
    "rupture_index=0.21",
    "relational_mode=builder",
    "emotional_weather=calm",
    "unresolved_hurts=hurt:emberwick-001",
    "vows=promise:pending:keep the lantern lit through winter",
)


def parity_edge() -> BondEdge:
    """TF-139's spike edge, as a production edge
    (synthetic Emberwick fixtures, TF-017)."""
    return BondEdge(
        bond_id=PARITY_BOND_ID,
        entity_a="user:emberwick-keeper",
        entity_b="persona:emberwick-lantern",
        domain=BondDomain.COMPANION,
        vector=BondVector(
            trust=0.4, warmth=0.4, familiarity=0.3, devotion=0.2,
            attraction_affinity=0.1, safety=0.9, sacred_resonance=0.3,
            playfulness=0.2, vulnerability=0.25, initiative_balance=0.0),
    )


def run_bond_parity() -> dict[str, Any]:
    """Exercise the full synthetic lifecycle through
    the PRODUCTION service: edge formed -> vow filed
    -> care event crosses the ACTIVE threshold -> a
    high hurt moves the edge to REPAIRING with the
    filed arithmetic -> the excerpt. Returns
    deterministic metrics + the excerpt lines."""
    graph = BondGraph()
    edge = parity_edge()
    graph.add_edge(edge)
    graph.add_vow(Vow(
        vow_id="vow:emberwick-001", bond_id=PARITY_BOND_ID,
        vow_text="keep the lantern lit through winter",
        vow_kind="promise",
        created_from_record_id="rec:synthetic-emberwick-000"))
    graph.apply_event(
        PARITY_BOND_ID, warmth_delta=0.2, trust_delta=0.2,
        devotion_delta=0.1,
        source_record_id="rec:synthetic-emberwick-001")
    status_after_event = edge.status.value
    graph.add_hurt(Hurt(
        hurt_id="hurt:emberwick-001", bond_id=PARITY_BOND_ID,
        source_event_id="evt:synthetic-emberwick-001",
        hurt_kind="false_memory", severity="high"))
    excerpt = graph.excerpt(PARITY_BOND_ID)
    metrics = {
        "edges": len(graph.edges), "vows": len(graph.vows),
        "hurts": len(graph.hurts),
        "status_after_event": status_after_event,
        "status_final": edge.status.value,
        "trust": round(edge.vector.trust, 4),
        "warmth": round(edge.vector.warmth, 4),
        "repair_debt": round(edge.scars.repair_debt, 4),
        "safety": round(edge.vector.safety, 4),
        "closeness": round(edge.closeness_index(), 4),
        "sacred": round(edge.sacred_bond_index(), 4),
        "rupture": round(edge.rupture_index(), 4),
        "evidence_records": len(edge.evidence.supporting_record_ids),
        "unresolved_hurts": len(edge.scars.unresolved_hurts),
        "excerpt_lines": len(excerpt),
    }
    return {"metrics": metrics, "excerpt": excerpt}


# ── The filed probes ─────────────────────────────────────────────────────────

BOND_GRAPH_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_production_and_weighting_pass",
     "kind": "clean",
     "detail": "the production run + weighting filing pass "
               "their validators"},
    {"id": "severity_shift_wrong_is_named",
     "kind": "severity_wrong",
     "detail": "a severity map with the high shift perturbed "
               "(0.35 -> 0.30) is named against the filed shifts"},
    {"id": "index_wrong_is_named",
     "kind": "index_wrong",
     "detail": "an expected filing with the parity closeness "
               "perturbed is named against the computed indices"},
    {"id": "schema_field_missing_is_named",
     "kind": "schema_field_missing",
     "detail": "a field map with vow_strain dropped from "
               "BondScars is named against the schemas"},
    {"id": "weighting_without_evidence_is_named",
     "kind": "weighting_no_evidence",
     "detail": "a weighting filing with its evidence emptied "
               "is named"},
    {"id": "edge_vector_out_of_range_is_named",
     "kind": "edge_out_of_range",
     "detail": "an edge whose trust files 1.4 is named by the "
               "edge validator"},
    {"id": "hurt_unknown_severity_is_named",
     "kind": "hurt_unknown_severity",
     "detail": "a hurt filing an unknown severity is named"},
)


def run_bond_graph_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in BOND_GRAPH_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            parity = run_bond_parity()
            problems = weighting_problems(WEIGHTING_DECISION) \
                + severity_map_problems(SEVERITY_SHIFTS) \
                + edge_problems(parity_edge())
            passed = problems == [] \
                and parity["metrics"] == PARITY_EXPECTED
        elif kind == "severity_wrong":
            mapping = dict(SEVERITY_SHIFTS, high=0.30)
            passed = bool(severity_map_problems(mapping))
        elif kind == "index_wrong":
            parity = run_bond_parity()
            m = parity["metrics"]
            expected = {
                "closeness": round(m["closeness"], 2) + 0.1,
                "sacred": round(m["sacred"], 2),
                "rupture": round(m["rupture"], 2),
            }
            passed = bool(index_problems(
                m["closeness"], m["sacred"], m["rupture"], expected))
        elif kind == "schema_field_missing":
            field_map = production_field_map()
            field_map["BondScars"] = tuple(
                f for f in field_map["BondScars"] if f != "vow_strain")
            passed = bool(schema_field_problems(field_map))
        elif kind == "weighting_no_evidence":
            data = dict(WEIGHTING_DECISION, evidence=())
            passed = bool(weighting_problems(data))
        elif kind == "edge_out_of_range":
            edge = parity_edge()
            edge.vector.trust = 1.4
            passed = bool(edge_problems(edge))
        elif kind == "hurt_unknown_severity":
            hurt = Hurt(
                hurt_id="hurt:probe", bond_id=PARITY_BOND_ID,
                source_event_id="evt:probe", hurt_kind="neglect",
                severity="catastrophic")
            passed = bool(hurt_problems(hurt))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_bond_graph_markdown() -> str:
    """Render the landing deterministically — the
    exact text filed as docs/architecture/BOND_GRAPH.md."""
    parity = run_bond_parity()
    metrics = parity["metrics"]
    lines: list[str] = []
    lines.append("# Bond Graph — TF-143 (production layer)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/knowledge/bond_graph.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.knowledge.bond_graph`. The "
        "checker (`bond_graph_problems()`) re-derives every "
        "claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{BOND_GRAPH_FORMAT}`")
    lines.append("Verdict implemented: **ADOPT_AS_LAYER** (TF-139)")
    lines.append(f"Probes: {len(BOND_GRAPH_PROBES)}/{len(BOND_GRAPH_PROBES)}")
    lines.append("")
    lines.append("## 1. What landed")
    lines.append("")
    lines.append(
        "Production dataclasses rebuilt from the bond_edge / "
        "vow / hurt schemas (BondEdge / Vow / Hurt + BondVector, "
        "BondScars, BondChronology, BondEvidence, BondConstraints, "
        "BondActiveModes, BondGovernance; enums BondDomain / "
        "BondStatus / VowState) and a production `BondGraph` "
        "service carrying the wing shell's operations faithfully "
        "(add_edge / add_vow / add_hurt / apply_event / excerpt). "
        "The layer stands BESIDE the shipping store: evidence "
        "references shipping-store record ids, no facts are "
        "duplicated, and it is not a second record store. The "
        "research wing's missing `wyrdforge.models` package is "
        "NOT resurrected — TF-143 lands production code in the "
        "shipping tree only; the wing stays as-is for "
        "TF-148/TF-149.")
    lines.append("")
    lines.append("## 2. The weighting decision (TF-139's condition)")
    lines.append("")
    lines.append(f"Decision: **{WEIGHTING_DECISION['decision']}**")
    lines.append("")
    lines.append("Evidence:")
    lines.append("")
    for row in WEIGHTING_DECISION["evidence"]:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("Limits:")
    lines.append("")
    for row in WEIGHTING_DECISION["limits"]:
        lines.append(f"- {row}")
    lines.append("")
    lines.append("## 3. Parity with TF-139's spike")
    lines.append("")
    lines.append(
        "The production service run over TF-139's synthetic "
        "Emberwick fixtures reproduces the spike's filed metrics "
        "exactly (equal weights — the filed decision — so no "
        "index moves):")
    lines.append("")
    lines.append(f"- status after event: {metrics['status_after_event']}")
    lines.append(f"- status final: {metrics['status_final']}")
    lines.append(f"- trust {metrics['trust']} / warmth {metrics['warmth']}")
    lines.append(
        f"- repair_debt {metrics['repair_debt']} / safety "
        f"{metrics['safety']} (high hurt: +0.35 debt, -0.175 safety)")
    lines.append(
        f"- closeness {metrics['closeness']} / sacred "
        f"{metrics['sacred']} / rupture {metrics['rupture']}")
    lines.append(f"- excerpt lines: {metrics['excerpt_lines']}")
    lines.append("")
    lines.append("The excerpt, verbatim:")
    lines.append("")
    lines.append("```")
    lines.extend(parity["excerpt"])
    lines.append("```")
    lines.append("")
    lines.append("## 4. The probes")
    lines.append("")
    for probe in BOND_GRAPH_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 5. Filed limits")
    lines.append("")
    lines.append(
        "- Fixture grade only: parity is proven on synthetic "
        "Emberwick fixtures (TF-017); no live model was read — "
        "whether a live system honours a bond excerpt is NOT "
        "PROVEN (TF-173 class).")
    lines.append(
        "- Equal weights are a filed default under spec silence, "
        "not a measured optimum (see §2).")
    lines.append(
        "- TF-146 owns the WYRD-consumable bond bridge; TF-149 "
        "owns the wing's tests. TF-140/TF-142's RETIRE verdicts "
        "land nothing here — TF-144 executes them.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_bond_graph_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / BOND_GRAPH_DOC
    doc.write_text(render_bond_graph_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.knowledge.bond_graph`."""
    doc = write_bond_graph_doc()
    print(f"bond graph regenerated: -> {BOND_GRAPH_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def bond_graph_problems() -> list[str]:
    """Re-derive TF-143's bond-graph claims from
    the live tree; name any that stopped being true."""
    problems: list[str] = []

    # The weighting decision, filed with evidence.
    problems.extend(weighting_problems(WEIGHTING_DECISION))
    spec_text = _SPEC.read_text(encoding="utf-8")
    for term in ("weighted_mean(trust, warmth, familiarity, vulnerability)",
                 "weighted_mean(devotion, sacred_resonance, "
                 "vow_integrity_inverse_strain)",
                 "weighted_mean(repair_debt, vow_strain, safety_inverse)"):
        if term not in spec_text:
            problems.append(
                f"spec: BondGraphSpec.md no longer files {term!r} — "
                "the weighting decision's premise must be re-ruled")

    # The production field sets match the schemas, live.
    problems.extend(schema_field_problems(production_field_map()))
    edge_schema = json.loads(
        (_SCHEMAS / "bond_edge.schema.json").read_text(encoding="utf-8"))
    if [e.value for e in BondDomain] != \
            edge_schema["$defs"]["BondDomain"]["enum"]:
        problems.append("production: BondDomain drifted from the schema")
    if [e.value for e in BondStatus] != \
            edge_schema["$defs"]["BondStatus"]["enum"]:
        problems.append("production: BondStatus drifted from the schema")
    vow_schema = json.loads(
        (_SCHEMAS / "vow.schema.json").read_text(encoding="utf-8"))
    if [e.value for e in VowState] != \
            vow_schema["$defs"]["VowState"]["enum"]:
        problems.append("production: VowState drifted from the schema")

    # The wing shell's severity map + operations still stand
    # as the production service's source.
    service_text = _SERVICE.read_text(encoding="utf-8")
    for name, shift in SEVERITY_SHIFTS.items():
        if f'"{name}": {shift}' not in service_text:
            problems.append(
                f"service: the severity shift for {name} is no "
                f"longer filed as {shift} in bond_graph_service.py")
    for op in ("def add_edge", "def add_vow", "def add_hurt",
               "def apply_event", "def excerpt"):
        if op not in service_text:
            problems.append(f"service: bond_graph_service.py lost {op}")
    problems.extend(severity_map_problems(SEVERITY_SHIFTS))

    # The layer discipline: the production module is not a
    # persisted record type and imports no wing code.
    from thoughtforge.knowledge.store_map import PERSISTED_RECORDS

    persisted = {row["name"] for row in PERSISTED_RECORDS}
    if any("bond" in name.lower() or "vow" in name.lower()
           or "hurt" in name.lower() for name in persisted):
        problems.append(
            "layer: a bond/vow/hurt record type entered the "
            "persisted census — the layer must stay beside the "
            "store, not inside it")
    module_text = Path(__file__).read_text(encoding="utf-8")
    wing_pkg = "wyrd" + "forge"
    if ("import " + wing_pkg) in module_text \
            or ("from " + wing_pkg) in module_text:
        problems.append(
            "layer: the production module imports wing code — "
            "production code must stand on its own")

    # The parity run reproduces TF-139's filed expectation.
    parity = run_bond_parity()
    if parity["metrics"] != PARITY_EXPECTED:
        problems.append(
            f"parity: the run files {parity['metrics']}, not the "
            f"filed {PARITY_EXPECTED}")
    if tuple(parity["excerpt"]) != PARITY_EXPECTED_EXCERPT:
        problems.append("parity: the excerpt drifted from its filing")
    from thoughtforge.knowledge.bond_graph_verdict import (
        SPIKE_EXPECTED,
        SPIKE_EXPECTED_EXCERPT,
    )

    if PARITY_EXPECTED != SPIKE_EXPECTED \
            or PARITY_EXPECTED_EXCERPT != SPIKE_EXPECTED_EXCERPT:
        problems.append(
            "parity: the production filing no longer matches "
            "TF-139's spike filing")

    # The probes all behave.
    probes = run_bond_graph_probes()
    if len(probes) != len(BOND_GRAPH_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the bond-graph probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / BOND_GRAPH_DOC
    if not doc.exists():
        problems.append(f"the write-up {BOND_GRAPH_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_bond_graph_markdown():
        problems.append(
            f"the filed doc {BOND_GRAPH_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.knowledge.bond_graph)")
    return problems


def bond_graph_as_data() -> dict[str, Any]:
    """The whole landing as plain data, for consumers."""
    parity = run_bond_parity()
    probes = run_bond_graph_probes()
    return {
        "format": BOND_GRAPH_FORMAT,
        "doc": BOND_GRAPH_DOC,
        "weighting_decision": {
            "decision": WEIGHTING_DECISION["decision"],
            "indices": {k: list(v) for k, v in
                        WEIGHTING_DECISION["indices"].items()},
            "weights": dict(WEIGHTING_DECISION["weights"]),
            "evidence": list(WEIGHTING_DECISION["evidence"]),
            "limits": list(WEIGHTING_DECISION["limits"]),
        },
        "parity": parity,
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BOND_GRAPH_DOC",
    "BOND_GRAPH_FORMAT",
    "BOND_GRAPH_PROBES",
    "PARITY_BOND_ID",
    "PARITY_EXPECTED",
    "PARITY_EXPECTED_EXCERPT",
    "PRODUCTION_SHAPES",
    "SEVERITY_SHIFTS",
    "WEIGHTING_DECISION",
    "BondActiveModes",
    "BondChronology",
    "BondConstraints",
    "BondDomain",
    "BondEdge",
    "BondEvidence",
    "BondGovernance",
    "BondGraph",
    "BondScars",
    "BondStatus",
    "BondVector",
    "Hurt",
    "Vow",
    "VowState",
    "bond_graph_as_data",
    "bond_graph_problems",
    "edge_problems",
    "hurt_problems",
    "index_problems",
    "main",
    "parity_edge",
    "production_field_map",
    "render_bond_graph_markdown",
    "run_bond_graph_probes",
    "run_bond_parity",
    "schema_field_problems",
    "severity_map_problems",
    "weighting_problems",
    "write_bond_graph_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
