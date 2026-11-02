"""
TF-139 — the bond graph verdict, stated as verifiable data.

The roadmap asks for a verdict on wyrdforge's bond graph
against the shipping knowledge store — *adopt (as a layer),
merge, or retire* — decided on TF-006's census plus a working
spike. The measured start, probed live this tick on this tree
before anything was built:

  * **The shipping store has no bond graph.** A grep for
    ``bond|relationship`` over
    ``src/thoughtforge/knowledge/*.py`` +
    ``src/thoughtforge/cognition/*.py`` returns nothing at all
    (this module excluded — it is the verdict, not the store).
    The shipping record types are PersonalityCore /
    UserPreference / UserFact / Episodic / ResponsePattern /
    ActiveThreadState etc. — records ABOUT one user or one
    persona, never an edge BETWEEN two entities. The only
    ``edges`` in the shipping store are ``conceptnet_edges``
    world-knowledge rows read by KnowledgeForge's
    ``sql_retrieve`` — concept-to-concept, not user<->persona.
  * **TF-006's census stands.** The bond graph is UNIQUE (no
    shipping equivalent); the wing cannot import —
    ``research_data/src/wyrdforge/models`` was never committed
    (verified absent live), while its services and 14 JSON
    schemas exist (verified: all 14 parse live) and the schemas
    fully specify the missing models' shapes.
    ``wyrdforge/__init__.py`` re-exports BondEdge / Vow / Hurt
    from that missing subpackage.
  * **The service shell is complete logic over missing types.**
    ``BondGraphService`` files add_edge (stamps formed_at),
    add_vow, add_hurt (severity shifts {low: 0.1, medium: 0.2,
    high: 0.35, mythic: 0.5}; repair_debt = min(1, debt+shift);
    safety = max(0, safety-shift/2); status -> REPAIRING;
    unresolved_hurts appended), apply_event (vector dims
    clamped 0..1, last_contact_at stamped, evidence record id
    appended, status -> ACTIVE when trust >= 0.55 and
    warmth >= 0.55) and excerpt (domain / status / closeness /
    sacred / rupture indices, modes, unresolved hurts, vows).
    BondGraphSpec.md files the derived indices as weighted
    means — closeness over (trust, warmth, familiarity,
    vulnerability), sacred over (devotion, sacred_resonance,
    inverse vow_strain), rupture over (repair_debt, vow_strain,
    inverse safety) — with the WEIGHTS UNSPECIFIED. The spike
    below uses equal weights and files that as a spike
    assumption, not a spec fact. One spec/schema drift is
    filed, not hidden: the spec's hurt shape carries
    repair_state / residual_sensitivity and severities
    low|medium|high, while hurt.schema.json carries notes
    (default "") and a mythic severity — the spike follows
    the SCHEMAS, as TF-006 ruled them the wing's most
    complete artefact.

What this module files:

  * ``COMPARISON_ROWS`` — the comparison as literal data:
    nine dimensions, each filing the bond graph's shape, the
    shipping store's shape, and a finding.
  * ``VERDICT`` — **ADOPT_AS_LAYER**, with filed reasons and
    filed conditions (see the constant: not a merge, not a
    retirement; adoption itself is TF-143's work, the WYRD
    bridge TF-146's, the wing's tests TF-149's — TF-139
    changes NO shipping behaviour and NO research-wing code).
  * **The working spike** — minimal spike dataclasses rebuilt
    from the three schemas (BondEdge / Vow / Hurt plus
    BondVector, BondScars, BondChronology, BondEvidence,
    BondConstraints, BondActiveModes, BondGovernance; enums
    BondDomain / BondStatus / VowState as the schemas file
    them), namespaced ``Spike*`` — explicitly NOT the missing
    ``wyrdforge.models`` package and NOT production adoption —
    and ``SpikeBondGraph``, the service shell's operations
    re-implemented faithfully over them.
    ``run_bond_spike()`` exercises a full synthetic lifecycle
    and returns deterministic citable lines/metrics (counts
    and indices only — timestamps exist as fields but no
    metric depends on a wall clock).
  * ``bond_graph_verdict_problems()`` — the checker:
    re-derives every claim above from the live tree (the
    shipping grep, the schemas, the missing models package,
    the service's severity map, the spike's field sets
    against the schemas, the filed doc as this module's
    render verbatim) and names any that stops being true.
  * ``BOND_PROBES`` — seven filed probes against the pure
    validators with perturbed copies (real filings never
    perturbed).

All spike fixtures are synthetic invented entities
(TF-017) — ``user:emberwick-keeper`` /
``persona:emberwick-lantern`` in TF-132's Emberwick style:
never Volmarr's canon, never the contents of ``data/``,
never a real persona name. (The wing's own demo_seed.py
names real-shaped ids; they are NOT copied here.)

Filed limits, stated plainly:

  * Spike grade only: the spike proves the bond graph's
    shapes and service logic can be rebuilt from the schemas
    and run deterministically on synthetic fixtures. It is
    NOT production adoption — TF-143 owns rebuilding
    ``wyrdforge.models`` from the 14 schemas and landing
    production code; TF-146 owns the WYRD-consumable bridge
    (the verdict adopted, so the bridge is live work);
    TF-149 owns the wing's tests.
  * The derived indices use EQUAL weights — the spec files
    the shapes but not the weights; TF-143 must file a
    weighting decision with evidence before production use.
  * No live model was read; whether a live system honours a
    bond excerpt is NOT PROVEN (TF-173 class).

The human-readable write-up is
``docs/architecture/BOND_GRAPH_VERDICT.md``, filed verbatim
as this module's render (regenerate with
``python3 -m thoughtforge.knowledge.bond_graph_verdict``).
Proof grade: FIXTURE/STATIC — schemas + service shell +
synthetic spike re-measured live on this VM; no live model,
never claimed.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

BOND_GRAPH_VERDICT_FORMAT = "thoughtforge-bond-graph-verdict/1"
BOND_GRAPH_VERDICT_DOC = "docs/architecture/BOND_GRAPH_VERDICT.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_SERVICE = _WING / "services" / "bond_graph_service.py"
_SCHEMAS = _WING / "schemas"

#: The three verdicts the roadmap allows.
VERDICTS: tuple[str, ...] = ("ADOPT_AS_LAYER", "MERGE", "RETIRE")

#: The findings a comparison row may file.
FINDINGS: tuple[str, ...] = ("unique", "layer-link", "different", "overlap")


# ── The comparison, as literal data ──────────────────────────────────────────

COMPARISON_ROWS: tuple[dict[str, Any], ...] = (
    {"dimension": "represented_subject",
     "bond_graph": "a directional EDGE between two entities "
                   "(entity_a + entity_b + domain): the relationship "
                   "itself is the stored object",
     "shipping_store": "RECORDS about one side — user facts, user "
                       "preferences, episodes, response patterns, the "
                       "personality core; no record type names a pair",
     "finding": "unique"},
    {"dimension": "vector_state",
     "bond_graph": "a ten-dimension bond vector per edge (trust, warmth, "
                   "familiarity, devotion, attraction_affinity, safety, "
                   "sacred_resonance, playfulness, vulnerability, "
                   "initiative_balance -1..1), clamped 0..1",
     "shipping_store": "per-record lifecycle scores (recency / half-life "
                       "in TF-041's ladder, pattern-learning scores) — "
                       "no per-relationship state vector exists",
     "finding": "unique"},
    {"dimension": "vows",
     "bond_graph": "first-class Vow objects (text, kind, strength, "
                   "pending|kept|strained|broken|released, witnesses, "
                   "kept/broken event links) anchored to a bond",
     "shipping_store": "no vow, promise or commitment object in any "
                       "record type — a promise can only be prose inside "
                       "a fact or episode summary",
     "finding": "unique"},
    {"dimension": "hurts_scars_repair",
     "bond_graph": "first-class Hurt objects plus edge scars: "
                   "repair_debt, vow_strain, unresolved_hurts; a hurt "
                   "moves the edge to REPAIRING by filed arithmetic",
     "shipping_store": "no hurt, scar or repair state — contradictions "
                       "are handled as knowledge conflicts "
                       "(knowledge/contradictions.py), never as "
                       "relational damage with a repair debt",
     "finding": "unique"},
    {"dimension": "evidence_links",
     "bond_graph": "BondEvidence.supporting_record_ids / "
                   "contradiction_record_ids name record ids — by "
                   "shape, the SHIPPING store's record ids: bond "
                   "claims must be backed by stored records, and "
                   "apply_event appends the source record id",
     "shipping_store": "the record store those ids live in: profile / "
                       "episodic / pattern records with stable "
                       "record_ids, plus KnowledgeForge citations",
     "finding": "layer-link"},
    {"dimension": "storage",
     "bond_graph": "the service shell keeps dicts in memory; "
                   "BondGraphSpec.md SUGGESTS a bond_edges SQL table "
                   "(pair+domain unique) — no storage is committed",
     "shipping_store": "MemoryStore JSONL/YAML files per record kind + "
                       "the SQLite knowledge DB, both committed and "
                       "lifecycle-managed (TF-041/TF-045)",
     "finding": "different"},
    {"dimension": "runtime_excerpt",
     "bond_graph": "excerpt(): seven filed lines (domain, status, the "
                   "three derived indices, relational mode, emotional "
                   "weather) + unresolved hurts + vows — sized for a "
                   "small model to honour, per the spec's small-model "
                   "tactics",
     "shipping_store": "retrieval returns records / knowledge chunks "
                       "for the turn; no compact per-relationship "
                       "excerpt exists for the prompt",
     "finding": "unique"},
    {"dimension": "update_law",
     "bond_graph": "event-derived bounded updates ONLY: clamped "
                   "integrator deltas from events, hurt severity "
                   "shifts by filed table, status transitions at filed "
                   "thresholds — no arbitrary mood swings (the spec's "
                   "update law)",
     "shipping_store": "ingestion-contract appends (TF-042) + lifecycle "
                       "ageing/pruning (TF-045): records change by being "
                       "written or aged, never by relational events",
     "finding": "different"},
    {"dimension": "governance",
     "bond_graph": "per-edge governance: review_required_for_large_shift "
                   "+ synthetic_claim_guard, plus the spec's overclaim "
                   "rules (no asserted shared memory without a "
                   "supporting record, no claimed vow without a vow "
                   "object, open hurts never silently skipped)",
     "shipping_store": "estate-wide governance: the TF-042 ingestion "
                       "contract, protected identity records, citation "
                       "discipline — real governance, but over RECORDS, "
                       "not over relational claims",
     "finding": "overlap"},
)


# ── The verdict ──────────────────────────────────────────────────────────────

VERDICT: dict[str, Any] = {
    "verdict": "ADOPT_AS_LAYER",
    "subject": "wyrdforge bond graph (bond_edge / vow / hurt schemas + "
               "BondGraphService) vs the shipping knowledge store",
    "decided_on": "TF-006's census (unique concept, wing cannot import, "
                  "schemas fully specify the missing models) + the "
                  "working spike filed in this module",
    "reasons": (
        "unique concept: five of the nine comparison dimensions are "
        "filed unique and none is filed redundant — the shipping store "
        "has no relationship edge, no bond vector, no vow, no hurt / "
        "repair state and no relational excerpt, so there is nothing "
        "to merge INTO and nothing whose loss retirement could justify",
        "not a merge: the shipping MemoryStore / KnowledgeForge answer "
        "'what is known' (records and world knowledge); the bond layer "
        "answers 'how entities stand with each other' (edges and their "
        "state) — folding edge state into record types, or records "
        "into edges, would corrupt both shapes",
        "not retire: the concept is complete as a design — spec, three "
        "schemas and a full service shell — and its absence is the "
        "estate's known gap for long-term relational continuity "
        "(TF-006: a RETIRE verdict must still harvest the concept; "
        "adopting it as a layer IS the harvest, with a home)",
        "the layer link already exists in the shapes: bond evidence "
        "names shipping-store record ids, so the layer can stand "
        "BESIDE the store referencing its records as evidence without "
        "duplicating a single fact",
        "the spike proves the shapes run: dataclasses rebuilt from "
        "the three schemas carry the service shell's operations "
        "faithfully (exact severity shifts, clamps, thresholds and "
        "excerpt lines) on synthetic fixtures, deterministically",
    ),
    "conditions": (
        "adopt AS A LAYER beside the shipping store, referencing "
        "store record ids as evidence — never duplicating facts, "
        "never a second record store",
        "adoption is NOT done in TF-139: TF-143 owns rebuilding "
        "wyrdforge.models from the 14 schemas and landing production "
        "code under the shipping tree's discipline; TF-146 owns the "
        "WYRD-consumable bond bridge (live work — the verdict is "
        "adopt); TF-149 owns the wing's tests",
        "TF-139 changes NO shipping behaviour and NO research-wing "
        "code: the spike lives in this verdict module only, "
        "namespaced Spike*, and is not the missing wyrdforge.models "
        "package",
        "the derived indices' weights are UNSPECIFIED in "
        "BondGraphSpec.md — the spike's equal weights are a spike "
        "assumption; TF-143 must file a weighting decision with "
        "evidence before any production use",
        "the schemas win over the spec where they drift (hurt "
        "carries notes + a mythic severity in the schema, not the "
        "spec's repair_state / residual_sensitivity) — TF-143 "
        "rebuilds from the schemas, per TF-006",
    ),
}


# ── Pure validators (the probes run against these) ───────────────────────────

def verdict_data_problems(data: Any) -> list[str]:
    """Judge one verdict filing, purely: a known
    verdict value, at least one reason, at least
    one condition. Empty-safe."""
    problems: list[str] = []
    if not isinstance(data, dict) or not data:
        return ["verdict: the filing is empty"]
    if data.get("verdict") not in VERDICTS:
        problems.append(
            f"verdict: {data.get('verdict')!r} is not one of {VERDICTS}")
    reasons = data.get("reasons") or ()
    if not reasons or not all(isinstance(r, str) and r for r in reasons):
        problems.append("verdict: the filing carries no reason")
    conditions = data.get("conditions") or ()
    if not conditions or not all(
            isinstance(c, str) and c for c in conditions):
        problems.append("verdict: the filing carries no condition")
    return problems


def comparison_row_problems(row: Any) -> list[str]:
    """Judge one comparison row, purely: both
    sides filed, a known finding. Empty-safe."""
    problems: list[str] = []
    if not isinstance(row, dict) or not row:
        return ["row: the row is empty"]
    name = row.get("dimension") or "<unnamed>"
    for side in ("dimension", "bond_graph", "shipping_store"):
        if not row.get(side):
            problems.append(f"row {name}: the {side} side is missing")
    if row.get("finding") not in FINDINGS:
        problems.append(
            f"row {name}: finding {row.get('finding')!r} is not one "
            f"of {FINDINGS}")
    return problems


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


# ── The spike: dataclasses rebuilt from the three schemas ────────────────────
#
# Namespaced Spike* — NOT wyrdforge.models, NOT production.
# Field sets match the schemas (required + defaults as filed);
# array fields take default_factory (the schemas file no
# default for them; a spike needs usable empties).

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
class SpikeBondVector:
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
class SpikeBondConstraints:
    exclusivity_mode: str = "none"
    intimacy_ceiling: str = "medium"
    boundary_profile_id: str = "default"


@dataclass
class SpikeBondScars:
    repair_debt: float = 0.0
    unresolved_hurts: list[str] = field(default_factory=list)
    vow_strain: float = 0.0


@dataclass
class SpikeBondActiveModes:
    relational_mode: str = "builder"
    emotional_weather: str = "calm"


@dataclass
class SpikeBondChronology:
    formed_at: Any = None
    last_major_shift_at: Any = None
    last_contact_at: Any = None


@dataclass
class SpikeBondEvidence:
    supporting_record_ids: list[str] = field(default_factory=list)
    contradiction_record_ids: list[str] = field(default_factory=list)


@dataclass
class SpikeBondGovernance:
    review_required_for_large_shift: bool = True
    synthetic_claim_guard: bool = True


@dataclass
class SpikeBondEdge:
    bond_id: str
    entity_a: str
    entity_b: str
    domain: BondDomain
    status: BondStatus = BondStatus.FORMING
    vector: SpikeBondVector = field(default_factory=SpikeBondVector)
    constraints: SpikeBondConstraints = field(
        default_factory=SpikeBondConstraints)
    scars: SpikeBondScars = field(default_factory=SpikeBondScars)
    active_modes: SpikeBondActiveModes = field(
        default_factory=SpikeBondActiveModes)
    chronology: SpikeBondChronology = field(
        default_factory=SpikeBondChronology)
    evidence: SpikeBondEvidence = field(default_factory=SpikeBondEvidence)
    governance: SpikeBondGovernance = field(
        default_factory=SpikeBondGovernance)

    def closeness_index(self) -> float:
        """Weighted mean per BondGraphSpec.md over
        (trust, warmth, familiarity, vulnerability)
        — EQUAL weights (spike assumption, filed)."""
        v = self.vector
        return (v.trust + v.warmth + v.familiarity + v.vulnerability) / 4

    def sacred_bond_index(self) -> float:
        """Weighted mean over (devotion,
        sacred_resonance, inverse vow_strain) —
        EQUAL weights (spike assumption, filed)."""
        v = self.vector
        return (v.devotion + v.sacred_resonance
                + (1.0 - self.scars.vow_strain)) / 3

    def rupture_index(self) -> float:
        """Weighted mean over (repair_debt,
        vow_strain, inverse safety) — EQUAL
        weights (spike assumption, filed)."""
        return (self.scars.repair_debt + self.scars.vow_strain
                + (1.0 - self.vector.safety)) / 3


@dataclass
class SpikeVow:
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
class SpikeHurt:
    hurt_id: str
    bond_id: str
    source_event_id: str
    hurt_kind: str
    severity: str
    notes: str = ""


#: Spike shape name -> the spike dataclass, for the
#: schema-coverage checker.
SPIKE_SHAPES: dict[str, Any] = {
    "BondEdge": SpikeBondEdge,
    "BondVector": SpikeBondVector,
    "BondConstraints": SpikeBondConstraints,
    "BondScars": SpikeBondScars,
    "BondActiveModes": SpikeBondActiveModes,
    "BondChronology": SpikeBondChronology,
    "BondEvidence": SpikeBondEvidence,
    "BondGovernance": SpikeBondGovernance,
    "Vow": SpikeVow,
    "Hurt": SpikeHurt,
}


def spike_field_map() -> dict[str, tuple[str, ...]]:
    """The spike dataclasses' field sets, by shape."""
    return {
        name: tuple(f.name for f in dataclasses.fields(cls))
        for name, cls in SPIKE_SHAPES.items()
    }


# ── The spike graph: the service shell, faithfully ───────────────────────────

class SpikeBondGraph:
    """BondGraphService's operations over the spike
    types: same stamps, same severity arithmetic,
    same clamps, same thresholds, same excerpt
    lines. Spike only — not production."""

    def __init__(self) -> None:
        self.edges: dict[str, SpikeBondEdge] = {}
        self.vows: dict[str, SpikeVow] = {}
        self.hurts: dict[str, SpikeHurt] = {}

    def add_edge(self, edge: SpikeBondEdge) -> None:
        if edge.chronology.formed_at is None:
            edge.chronology.formed_at = datetime.now(UTC)
        self.edges[edge.bond_id] = edge

    def add_vow(self, vow: SpikeVow) -> None:
        self.vows[vow.vow_id] = vow

    def add_hurt(self, hurt: SpikeHurt) -> None:
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
    ) -> SpikeBondEdge:
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


# ── The spike run: a full synthetic lifecycle ────────────────────────────────

SPIKE_BOND_ID = "bond:emberwick-keeper-companion"

#: The filed expectation for the spike run (deterministic:
#: counts and indices only, rounded as filed).
SPIKE_EXPECTED: dict[str, Any] = {
    "edges": 1, "vows": 1, "hurts": 1,
    "status_after_event": "active",
    "status_final": "repairing",
    "trust": 0.6, "warmth": 0.6,
    "repair_debt": 0.35, "safety": 0.725,
    "closeness": 0.4375, "sacred": 0.5333, "rupture": 0.2083,
    "evidence_records": 1, "unresolved_hurts": 1,
    "excerpt_lines": 9,
}

SPIKE_EXPECTED_EXCERPT: tuple[str, ...] = (
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


def run_bond_spike() -> dict[str, Any]:
    """Exercise the full synthetic lifecycle:
    edge formed -> vow filed -> care event crosses
    the ACTIVE threshold -> a high hurt moves the
    edge to REPAIRING with the filed arithmetic ->
    the excerpt. Returns deterministic metrics +
    the excerpt lines (no wall-clock values)."""
    graph = SpikeBondGraph()
    edge = SpikeBondEdge(
        bond_id=SPIKE_BOND_ID,
        entity_a="user:emberwick-keeper",
        entity_b="persona:emberwick-lantern",
        domain=BondDomain.COMPANION,
        vector=SpikeBondVector(
            trust=0.4, warmth=0.4, familiarity=0.3, devotion=0.2,
            attraction_affinity=0.1, safety=0.9, sacred_resonance=0.3,
            playfulness=0.2, vulnerability=0.25, initiative_balance=0.0),
    )
    graph.add_edge(edge)
    graph.add_vow(SpikeVow(
        vow_id="vow:emberwick-001", bond_id=SPIKE_BOND_ID,
        vow_text="keep the lantern lit through winter",
        vow_kind="promise",
        created_from_record_id="rec:synthetic-emberwick-000"))
    graph.apply_event(
        SPIKE_BOND_ID, warmth_delta=0.2, trust_delta=0.2,
        devotion_delta=0.1,
        source_record_id="rec:synthetic-emberwick-001")
    status_after_event = edge.status.value
    graph.add_hurt(SpikeHurt(
        hurt_id="hurt:emberwick-001", bond_id=SPIKE_BOND_ID,
        source_event_id="evt:synthetic-emberwick-001",
        hurt_kind="false_memory", severity="high"))
    excerpt = graph.excerpt(SPIKE_BOND_ID)
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

BOND_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_verdict_and_rows_pass",
     "kind": "clean",
     "detail": "the filed verdict + rows pass their validators"},
    {"id": "verdict_without_a_reason_is_named",
     "kind": "verdict_no_reason",
     "detail": "a verdict filing with its reasons emptied is named"},
    {"id": "comparison_row_missing_a_side_is_named",
     "kind": "row_missing_side",
     "detail": "a comparison row with its shipping side emptied "
               "is named"},
    {"id": "spike_hurt_shift_wrong_is_named",
     "kind": "severity_wrong",
     "detail": "a severity map with the high shift perturbed "
               "(0.35 -> 0.30) is named against the filed shifts"},
    {"id": "spike_index_wrong_is_named",
     "kind": "index_wrong",
     "detail": "an expected filing with the spike's closeness "
               "perturbed is named against the computed indices"},
    {"id": "schema_field_missing_from_spike_is_named",
     "kind": "schema_field_missing",
     "detail": "a field map with vow_strain dropped from BondScars "
               "is named against the schemas"},
    {"id": "unknown_verdict_value_is_named",
     "kind": "verdict_unknown",
     "detail": "a verdict value outside the roadmap's three is named"},
)


def run_bond_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in BOND_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems = verdict_data_problems(VERDICT) + [
                p for row in COMPARISON_ROWS
                for p in comparison_row_problems(row)]
            passed = problems == []
        elif kind == "verdict_no_reason":
            data = dict(VERDICT, reasons=())
            passed = bool(verdict_data_problems(data))
        elif kind == "row_missing_side":
            row = dict(COMPARISON_ROWS[0], shipping_store="")
            passed = bool(comparison_row_problems(row))
        elif kind == "severity_wrong":
            mapping = dict(SEVERITY_SHIFTS, high=0.30)
            passed = bool(severity_map_problems(mapping))
        elif kind == "index_wrong":
            spike = run_bond_spike()
            edge_metrics = spike["metrics"]
            expected = {
                "closeness": round(edge_metrics["closeness"], 2) + 0.1,
                "sacred": round(edge_metrics["sacred"], 2),
                "rupture": round(edge_metrics["rupture"], 2),
            }
            passed = bool(index_problems(
                edge_metrics["closeness"], edge_metrics["sacred"],
                edge_metrics["rupture"], expected))
        elif kind == "schema_field_missing":
            field_map = spike_field_map()
            field_map["BondScars"] = tuple(
                f for f in field_map["BondScars"] if f != "vow_strain")
            passed = bool(schema_field_problems(field_map))
        elif kind == "verdict_unknown":
            data = dict(VERDICT, verdict="SHRED")
            passed = bool(verdict_data_problems(data))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_bond_graph_verdict_markdown() -> str:
    """Render the verdict deterministically — the
    exact text filed as
    docs/architecture/BOND_GRAPH_VERDICT.md."""
    spike = run_bond_spike()
    metrics = spike["metrics"]
    lines: list[str] = []
    lines.append("# Bond Graph Verdict — TF-139")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/knowledge/bond_graph_verdict.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.knowledge.bond_graph_verdict`. "
        "The checker (`bond_graph_verdict_problems()`) re-derives "
        "every claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{BOND_GRAPH_VERDICT_FORMAT}`")
    lines.append(f"Verdict: **{VERDICT['verdict']}**")
    lines.append(f"Comparison rows: {len(COMPARISON_ROWS)}")
    lines.append(f"Probes: {len(BOND_PROBES)}/{len(BOND_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: a "
        "grep for `bond|relationship` over "
        "`src/thoughtforge/knowledge/*.py` + "
        "`src/thoughtforge/cognition/*.py` returns nothing (this "
        "verdict module excluded) — the shipping store has no bond "
        "or relationship graph, and its only `edges` are "
        "`conceptnet_edges` world-knowledge rows. "
        "`research_data/src/wyrdforge/models` does not exist (the "
        "wing cannot import, TF-006 / D-WING-1); the wing's 14 JSON "
        "schemas all parse, and its `BondGraphService` shell files "
        "complete logic over the missing types. TF-006's census "
        "files the bond graph UNIQUE — no shipping equivalent.")
    lines.append("")
    lines.append("## 2. The comparison")
    lines.append("")
    lines.append("| Dimension | Bond graph | Shipping store | Finding |")
    lines.append("|---|---|---|---|")
    for row in COMPARISON_ROWS:
        lines.append(
            f"| `{row['dimension']}` | {row['bond_graph']} | "
            f"{row['shipping_store']} | **{row['finding']}** |")
    lines.append("")
    lines.append("## 3. The verdict: ADOPT_AS_LAYER")
    lines.append("")
    lines.append("### Reasons")
    lines.append("")
    for reason in VERDICT["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("### Conditions")
    lines.append("")
    for condition in VERDICT["conditions"]:
        lines.append(f"- {condition}")
    lines.append("")
    lines.append("## 4. The working spike")
    lines.append("")
    lines.append(
        "Spike dataclasses rebuilt from the three schemas "
        "(BondEdge / Vow / Hurt + BondVector, BondScars, "
        "BondChronology, BondEvidence, BondConstraints, "
        "BondActiveModes, BondGovernance; enums BondDomain / "
        "BondStatus / VowState), namespaced `Spike*` in the "
        "verdict module — NOT the missing `wyrdforge.models` "
        "package and NOT production adoption. `SpikeBondGraph` "
        "re-implements the service shell faithfully: the exact "
        "severity shifts {low: 0.1, medium: 0.2, high: 0.35, "
        "mythic: 0.5}, repair_debt = min(1, debt + shift), "
        "safety = max(0, safety - shift/2), status -> REPAIRING "
        "on a hurt, vector clamps 0..1, status -> ACTIVE at "
        "trust >= 0.55 and warmth >= 0.55, and the service's "
        "excerpt line shapes. The derived indices use EQUAL "
        "weights — BondGraphSpec.md files the index shapes but "
        "not the weights; equal weights are a filed spike "
        "assumption. Fixtures are synthetic invented entities "
        "(TF-017): `user:emberwick-keeper` / "
        "`persona:emberwick-lantern`.")
    lines.append("")
    lines.append("Spike lifecycle (synthetic): edge formed -> "
                 "vow filed -> care event -> high hurt -> excerpt.")
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
    lines.append(
        f"- evidence records {metrics['evidence_records']} / "
        f"unresolved hurts {metrics['unresolved_hurts']}")
    lines.append("")
    lines.append("The excerpt, verbatim:")
    lines.append("")
    lines.append("```")
    lines.extend(spike["excerpt"])
    lines.append("```")
    lines.append("")
    lines.append("## 5. The probes")
    lines.append("")
    for probe in BOND_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-139 discharges the bond-graph share of D-WING-1 "
        "(verdict decided on TF-006's census + this working "
        "spike, narrowed in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 "
        "itself stays OPEN for the TF-140..TF-142 verdicts and "
        "the TF-143 rebuild / TF-149 tests it assigns.")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Spike grade only: production adoption is NOT done "
        "and NOT PROVEN here — TF-143 owns rebuilding "
        "`wyrdforge.models` from the 14 schemas and landing "
        "production code; TF-146 owns the WYRD-consumable "
        "bridge; TF-149 owns the wing's tests.")
    lines.append(
        "- The derived indices' weights are unspecified in "
        "BondGraphSpec.md; the spike's equal weights are an "
        "assumption TF-143 must replace with a filed decision.")
    lines.append(
        "- No live model was read; whether a live system "
        "honours a bond excerpt is NOT PROVEN (TF-173 class).")
    lines.append(
        "- Where the spec and the schemas drift (the hurt "
        "shape), the schemas win — TF-006 ruled them the "
        "wing's most complete artefact.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_bond_graph_verdict_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / BOND_GRAPH_VERDICT_DOC
    doc.write_text(render_bond_graph_verdict_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.knowledge.bond_graph_verdict`."""
    doc = write_bond_graph_verdict_doc()
    print(
        f"bond-graph verdict regenerated: {VERDICT['verdict']} -> "
        f"{BOND_GRAPH_VERDICT_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def bond_graph_verdict_problems() -> list[str]:
    """Re-derive TF-139's claims from the live
    tree; name any that stopped being true."""
    problems: list[str] = []

    # The verdict filing itself.
    problems.extend(verdict_data_problems(VERDICT))
    row_ids = [r["dimension"] for r in COMPARISON_ROWS]
    if len(row_ids) != len(set(row_ids)):
        problems.append("duplicate COMPARISON_ROWS dimensions")
    if len(COMPARISON_ROWS) != 9:
        problems.append(
            f"the comparison files {len(COMPARISON_ROWS)} rows, not 9")
    for row in COMPARISON_ROWS:
        problems.extend(comparison_row_problems(row))
    findings = [r["finding"] for r in COMPARISON_ROWS]
    if findings.count("unique") != 5 or findings.count("layer-link") != 1:
        problems.append(
            f"the findings are {findings.count('unique')} unique / "
            f"{findings.count('layer-link')} layer-link, not 5 / 1")

    # The shipping store still has no bond graph (this module
    # excluded — it is the verdict, not the store).
    knowledge_dir = _REPO_ROOT / "src" / "thoughtforge" / "knowledge"
    cognition_dir = _REPO_ROOT / "src" / "thoughtforge" / "cognition"
    for folder in (knowledge_dir, cognition_dir):
        for path in sorted(folder.glob("*.py")):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8").lower()
            if "bond" in text or "relationship" in text:
                problems.append(
                    f"shipping: {path.name} now names a bond / "
                    "relationship — the uniqueness premise must be "
                    "re-ruled")
    from thoughtforge.knowledge.store_map import PERSISTED_RECORDS

    persisted = {row["name"] for row in PERSISTED_RECORDS}
    if any("bond" in name.lower() or "vow" in name.lower()
           or "hurt" in name.lower() for name in persisted):
        problems.append(
            "shipping: a bond/vow/hurt record type entered the "
            "persisted census — the verdict must be re-ruled")

    # The wing as censused: models missing, schemas present and
    # parsing, the __init__ still importing the missing package.
    if (_WING / "models").exists():
        problems.append(
            "wing: wyrdforge.models now exists — TF-143's rebuild "
            "has landed and this verdict's premise must be "
            "re-ruled")
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
    init_text = (_WING / "__init__.py").read_text(encoding="utf-8")
    for name in ("BondEdge", "Vow", "Hurt"):
        if f"from .models.bond import" not in init_text \
                or name not in init_text:
            problems.append(f"wing: __init__ no longer imports {name}")

    # The service shell's severity map matches the spike's.
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

    # The spike's field sets match the schemas, live.
    problems.extend(schema_field_problems(spike_field_map()))

    # Enum shapes as the schemas file them.
    edge_schema = json.loads(
        (_SCHEMAS / "bond_edge.schema.json").read_text(encoding="utf-8"))
    if [e.value for e in BondDomain] != \
            edge_schema["$defs"]["BondDomain"]["enum"]:
        problems.append("spike: BondDomain drifted from the schema")
    if [e.value for e in BondStatus] != \
            edge_schema["$defs"]["BondStatus"]["enum"]:
        problems.append("spike: BondStatus drifted from the schema")
    vow_schema = json.loads(
        (_SCHEMAS / "vow.schema.json").read_text(encoding="utf-8"))
    if [e.value for e in VowState] != \
            vow_schema["$defs"]["VowState"]["enum"]:
        problems.append("spike: VowState drifted from the schema")

    # The spike run reproduces its filed expectation.
    spike = run_bond_spike()
    if spike["metrics"] != SPIKE_EXPECTED:
        problems.append(
            f"spike: the run files {spike['metrics']}, not the "
            f"filed {SPIKE_EXPECTED}")
    if tuple(spike["excerpt"]) != SPIKE_EXPECTED_EXCERPT:
        problems.append("spike: the excerpt drifted from its filing")

    # The probes all behave.
    probes = run_bond_probes()
    if len(probes) != len(BOND_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the bond probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / BOND_GRAPH_VERDICT_DOC
    if not doc.exists():
        problems.append(f"the write-up {BOND_GRAPH_VERDICT_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != \
            render_bond_graph_verdict_markdown():
        problems.append(
            f"the filed doc {BOND_GRAPH_VERDICT_DOC} is not the "
            "render verbatim (regenerate with python3 -m "
            "thoughtforge.knowledge.bond_graph_verdict)")
    return problems


def bond_graph_verdict_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    spike = run_bond_spike()
    probes = run_bond_probes()
    return {
        "format": BOND_GRAPH_VERDICT_FORMAT,
        "doc": BOND_GRAPH_VERDICT_DOC,
        "verdict": dict(VERDICT),
        "comparison_rows": [dict(r) for r in COMPARISON_ROWS],
        "spike": spike,
        "measured": {
            "comparison_rows": len(COMPARISON_ROWS),
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BOND_GRAPH_VERDICT_DOC",
    "BOND_GRAPH_VERDICT_FORMAT",
    "BOND_PROBES",
    "COMPARISON_ROWS",
    "FINDINGS",
    "SEVERITY_SHIFTS",
    "SPIKE_BOND_ID",
    "SPIKE_EXPECTED",
    "SPIKE_EXPECTED_EXCERPT",
    "SPIKE_SHAPES",
    "VERDICT",
    "VERDICTS",
    "BondDomain",
    "BondStatus",
    "SpikeBondActiveModes",
    "SpikeBondChronology",
    "SpikeBondConstraints",
    "SpikeBondEdge",
    "SpikeBondEvidence",
    "SpikeBondGovernance",
    "SpikeBondGraph",
    "SpikeBondScars",
    "SpikeBondVector",
    "SpikeHurt",
    "SpikeVow",
    "VowState",
    "bond_graph_verdict_as_data",
    "bond_graph_verdict_problems",
    "comparison_row_problems",
    "index_problems",
    "main",
    "render_bond_graph_verdict_markdown",
    "run_bond_probes",
    "run_bond_spike",
    "schema_field_problems",
    "severity_map_problems",
    "spike_field_map",
    "verdict_data_problems",
    "write_bond_graph_verdict_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
