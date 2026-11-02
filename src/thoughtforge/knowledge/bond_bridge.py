"""
TF-146 — the bond-graph bridge: bonds as
WYRD-consumable data.

TF-146's roadmap text: *The bond-graph bridge
(if adopted at TF-139): bonds as WYRD-consumable
data.* The antecedent is TRUE: TF-139's verdict
was **ADOPT_AS_LAYER** and TF-143 landed the
production bond graph
(``src/thoughtforge/knowledge/bond_graph.py``).
TF-145's boundary doc files this bridge as its
one not-built interface row (``bond_excerpt_
bridge``, status ``not-built-tf146``), with
``BondGraph.excerpt()``'s deterministic lines as
the source shape. This module builds exactly
that bridge.

The measured start, probed live on this tree
before anything was built:

  * **No bridge existed.** No file under
    ``src/thoughtforge/knowledge/`` named for a
    bridge, no ``docs/architecture/BOND_BRIDGE``
    doc, and the production ``BondGraph``'s only
    outward shape was ``excerpt()``'s human-
    readable lines — nothing a WYRD-side
    consumer could consume as DATA (no ids as
    fields, no vector, no scars, no vow/hurt
    records, no evidence links, no versioned
    format).
  * **The antecedent and the evidence stood
    clean live**: ``bond_graph_verdict_
    problems()`` [] with the verdict
    ADOPT_AS_LAYER, ``bond_graph_problems()``
    [] (parity excerpt 9 lines),
    ``wyrd_boundary_problems()`` [] with the
    bridge row filing ``not-built-tf146``, and
    TF-132's panel PASS (18 cases / 9 consistent
    / 8 contradicted / 1 honest-unanswerable).

What the bridge is (literal data below):

  * :func:`export_bond` — a documented,
    versioned, serialisable export of ONE bond,
    built FROM the production ``BondGraph``
    (TF-143): it reads the edge, its vows and
    its hurts through the graph's own stores and
    takes its excerpt from ``BondGraph.
    excerpt()`` itself. NO bond logic is
    duplicated here: no severity arithmetic, no
    index formula is re-derived for the export's
    VALUES — the indices are the edge's own
    methods' outputs. (The pure validator
    recomputes the indices from the exported
    vector/scars so a CONSUMER can verify an
    export without ThoughtForge code — that is
    a check on the data, not a second source of
    the values.)
  * ``BRIDGE_ROWS`` — the exported sections as
    literal data: each row files the section,
    its source in the production graph, and
    what it carries.
  * ``RESERVATIONS`` — what the export does NOT
    carry, filed as data and enforced by the
    validators: no world state (the canonical
    lived world is WYRD-side, TF-145), no canon
    data (Volmarr's, neither side's, TF-017),
    and no store FACTS — evidence carries the
    record IDS only, exactly the layer link
    TF-139/TF-143 filed; the records themselves
    stay in the shipping store.
  * The round-trip proof, on synthetic fixtures
    only (TF-017): a production ``BondGraph``
    built over TF-139/TF-143's Emberwick parity
    fixtures is exported, the export is JSON
    round-tripped byte-stable, and the exported
    shape is proven to carry everything this
    filing claims — ids, the full vector, the
    scars, the evidence record links, the
    derived indices and the excerpt lines,
    verbatim.

**TF-145's filed status, handled in the open**:
TF-145's boundary module still files the
bridge row ``not-built-tf146``, and this slice
LEAVES IT THERE. The estate's precedent is
uniform: TF-143 did not rewrite TF-139's
verdict module (which still files TF-143 as
the rebuild's owner after the rebuild landed),
and TF-144 did not rewrite TF-140/TF-142's
verdict modules — a filed slice's doc is the
record of what was true and measured when it
was filed, and the slice that changes the
estate files the change in ITS OWN module.
(TF-137's in-the-open updates were to
``benchmarks/harness.py``, a LIVING registry
of the current estate, not a filed slice doc —
a different object.) So the boundary doc
stands as TF-145 filed it; THIS module is the
record that the bridge now exists, TF-151/
TF-152 re-measure the estate with the verdicts
landed, and this module's checker pins
TF-145's row still naming TF-146 as the owner
— the antecedent the bridge stands on.

**Docket**: TF-146 has NO share of TF-016's
D-WING-1 — the entry's current text (checked
live for this slice) assigns the verdicts
(TF-139..TF-142), the adoptions (TF-143), the
retirements (TF-144), research hygiene
(TF-148) and the wing's tests (TF-149); the
bridge's ownership is filed in TF-139's
verdict conditions, TF-143's filing and
TF-145's boundary row, not in D-WING-1. Per
the TF-145 pattern's rule (a slice touches the
docket only for its own filed share), this
slice does NOT edit the docket and adds no
entry: TF-016 totals stand unchanged (137
entries — 33 OPEN / 1 OPEN-ENVIRONMENT / 100
FIXED / 3 CLOSED-BY-VERDICT), filed in the
open here and in the render.

The human-readable write-up is
``docs/architecture/BOND_BRIDGE.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.knowledge.
bond_bridge``).
Proof grade: FIXTURE/STATIC — the production
graph + synthetic fixtures re-measured live on
this VM; no live WYRD consumer was read — a
live system consuming the bridge is NOT PROVEN
(TF-173 class), never claimed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.bond_graph import (
    PARITY_BOND_ID,
    PARITY_EXPECTED,
    PARITY_EXPECTED_EXCERPT,
    BondGraph,
    Hurt,
    Vow,
    parity_edge,
)

BOND_BRIDGE_FORMAT = "thoughtforge-bond-bridge/1"
BOND_BRIDGE_DOC = "docs/architecture/BOND_BRIDGE.md"
SOURCE_FORMAT = "thoughtforge-bond-graph/1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BOUNDARY = _REPO_ROOT / "src" / "thoughtforge" / "wyrd_boundary.py"

#: Keys an export must never carry, at any depth:
#: the world state is WYRD-side and canon is
#: neither side's (TF-145's ownership rows).
FORBIDDEN_KEYS: tuple[str, ...] = (
    "world_state", "world", "canon", "canon_data",
    "facts", "store_records",
)

#: The vector dimensions the production
#: BondVector files, in declaration order.
VECTOR_DIMS: tuple[str, ...] = (
    "trust", "warmth", "familiarity", "devotion",
    "attraction_affinity", "safety", "sacred_resonance",
    "playfulness", "vulnerability", "initiative_balance",
)


# ── The exported sections, as literal data ───────────────────────────────────

BRIDGE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "identity",
     "section": "bond_id / entity_a / entity_b / domain / status",
     "source": "BondEdge's own fields (bond_id, entity_a, "
               "entity_b, domain, status)",
     "carries": "which bond this is, the two entities it "
                "stands between (ThoughtForge-side ids, filed "
                "as references — entity identity in the world "
                "stays WYRD-side, TF-145), its domain and its "
                "lifecycle status"},
    {"id": "vector",
     "section": "vector",
     "source": "BondVector, all ten dimensions",
     "carries": "the full relational vector exactly as the "
                "production edge holds it — no dimension "
                "dropped, none re-scaled"},
    {"id": "indices",
     "section": "indices (closeness / sacred / rupture)",
     "source": "BondEdge.closeness_index() / sacred_bond_"
               "index() / rupture_index() — the edge's own "
               "methods, under TF-143's filed equal weights",
     "carries": "the three derived indices a consumer reads "
                "without re-deriving them; the pure validator "
                "recomputes them from the exported vector + "
                "scars so the filing is checkable data-side"},
    {"id": "scars",
     "section": "scars (repair_debt / vow_strain / "
                "unresolved_hurts)",
     "source": "BondScars",
     "carries": "the repair state: how much repair debt the "
                "bond carries, its vow strain, and the ids of "
                "its unresolved hurts"},
    {"id": "vows",
     "section": "vows",
     "source": "the graph's Vow records for this bond, in "
               "the graph's filing order",
     "carries": "every vow on the bond as data — id, kind, "
                "state, text, strength, the store record it "
                "was created from, witnesses and its kept / "
                "broken event links"},
    {"id": "hurts",
     "section": "hurts",
     "source": "the graph's Hurt records for this bond, in "
               "the graph's filing order",
     "carries": "every hurt on the bond as data — id, kind, "
                "severity, notes, and the source event id it "
                "answers to"},
    {"id": "evidence",
     "section": "evidence (supporting_record_ids / "
                "contradiction_record_ids)",
     "source": "BondEvidence",
     "carries": "the layer link itself: the shipping-store "
                "record IDS the bond's state rests on — ids "
                "only, never the records (the facts stay in "
                "the store, TF-139's layer link)"},
    {"id": "excerpt",
     "section": "excerpt",
     "source": "BondGraph.excerpt(bond_id), called on the "
               "production graph — never re-rendered here",
     "carries": "TF-145's filed source shape, carried "
                "verbatim as data lines beside the structured "
                "sections, so a consumer that only reads "
                "lines and a consumer that reads fields see "
                "the same bond"},
    {"id": "context",
     "section": "constraints / active_modes / governance / "
                "chronology",
     "source": "BondConstraints / BondActiveModes / "
               "BondGovernance / BondChronology (timestamps "
               "as ISO strings or null)",
     "carries": "the bond's filed constraints, its active "
                "relational mode + emotional weather, its "
                "governance flags, and its chronology — the "
                "context a consumer needs to honour the bond, "
                "with no wall-clock value ever filed as a "
                "metric"},
)


# ── What the bridge does NOT carry, as literal data ──────────────────────────

RESERVATIONS: tuple[dict[str, Any], ...] = (
    {"id": "no_world_state",
     "reserved_to": "wyrd",
     "not_carried": "no canonical world state — no entity "
                    "attributes, locations or relations of "
                    "the lived world; the export's entity "
                    "ids are ThoughtForge-side references "
                    "only (TF-145: canonical_world_state + "
                    "entity_identity_in_world are WYRD's)"},
    {"id": "no_canon_data",
     "reserved_to": "neither",
     "not_carried": "no canon data — nothing from data/, no "
                    "persona material, no dataset content "
                    "(TF-145: canon_data is neither side's; "
                    "TF-017: fixtures synthetic only)"},
    {"id": "no_store_facts",
     "reserved_to": "thoughtforge-store",
     "not_carried": "no store facts — evidence carries the "
                    "record ids only; the records themselves "
                    "stay in the shipping store and are never "
                    "duplicated into the export (TF-139's "
                    "layer link, TF-143's layer discipline)"},
    {"id": "no_live_claim",
     "reserved_to": "neither",
     "not_carried": "no claim of live consumption — the "
                    "bridge files a consumable SHAPE; whether "
                    "a live WYRD-side system consumes it is "
                    "NOT PROVEN (TF-173 class), never claimed"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def _forbidden_problems(data: Any, path: str = "export") -> list[str]:
    problems: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in FORBIDDEN_KEYS:
                problems.append(
                    f"{path}: forbidden key {key!r} is carried — "
                    "the bridge reserves it (see RESERVATIONS)")
            problems.extend(_forbidden_problems(value, f"{path}.{key}"))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            problems.extend(_forbidden_problems(value, f"{path}[{i}]"))
    return problems


def _expected_indices(vector: dict, scars: dict) -> dict[str, float]:
    """The consumer-side check: TF-143's filed
    equal-weight means recomputed from exported
    data alone."""
    closeness = (vector["trust"] + vector["warmth"]
                 + vector["familiarity"]
                 + vector["vulnerability"]) / 4
    sacred = (vector["devotion"] + vector["sacred_resonance"]
              + (1.0 - scars["vow_strain"])) / 3
    rupture = (scars["repair_debt"] + scars["vow_strain"]
               + (1.0 - vector["safety"])) / 3
    return {"closeness": closeness, "sacred": sacred,
            "rupture": rupture}


def _expected_excerpt_head(export: dict) -> list[str]:
    idx = export["indices"]
    modes = export["active_modes"]
    return [
        f"domain={export['domain']}",
        f"status={export['status']}",
        f"closeness_index={idx['closeness']:.2f}",
        f"sacred_bond_index={idx['sacred']:.2f}",
        f"rupture_index={idx['rupture']:.2f}",
        f"relational_mode={modes['relational_mode']}",
        f"emotional_weather={modes['emotional_weather']}",
    ]


def export_problems(export: Any) -> list[str]:
    """Judge one bridge export, purely: the
    versioned format, the ids, the full vector in
    range, the indices recomputed from the
    exported vector + scars, the scars in range,
    the evidence id lists, every vow/hurt bound
    to this bond, the excerpt agreeing with the
    structured sections line for line, and no
    forbidden key anywhere. Empty-safe."""
    if not isinstance(export, dict) or not export:
        return ["export: the export is empty"]
    problems: list[str] = []
    if export.get("format") != BOND_BRIDGE_FORMAT:
        problems.append(
            f"export: format is {export.get('format')!r}, not "
            f"{BOND_BRIDGE_FORMAT!r}")
    if export.get("source_format") != SOURCE_FORMAT:
        problems.append(
            f"export: source_format is "
            f"{export.get('source_format')!r}, not {SOURCE_FORMAT!r}")
    for key in ("bond_id", "entity_a", "entity_b", "domain", "status"):
        if not export.get(key):
            problems.append(f"export: {key} is missing")
    vector = export.get("vector") or {}
    if tuple(vector) != VECTOR_DIMS:
        problems.append(
            f"export: vector files {tuple(vector)}, not the "
            f"production dims {VECTOR_DIMS}")
    for name, value in vector.items():
        if not isinstance(value, (int, float)):
            problems.append(f"export: vector {name} is not a number")
            continue
        low = -1.0 if name == "initiative_balance" else 0.0
        if not low <= value <= 1.0:
            problems.append(
                f"export: vector {name} is {value}, outside {low}..1")
    scars = export.get("scars") or {}
    for name in ("repair_debt", "vow_strain"):
        value = scars.get(name)
        if not isinstance(value, (int, float)) \
                or not 0.0 <= value <= 1.0:
            problems.append(
                f"export: scars {name} is {value!r}, outside 0..1")
    if not isinstance(scars.get("unresolved_hurts"), list) \
            or not all(isinstance(h, str)
                       for h in scars.get("unresolved_hurts") or []):
        problems.append("export: scars unresolved_hurts is not a "
                        "list of hurt ids")
    indices = export.get("indices") or {}
    if set(vector) == set(VECTOR_DIMS) \
            and isinstance(scars.get("repair_debt"), (int, float)) \
            and isinstance(scars.get("vow_strain"), (int, float)):
        expected = _expected_indices(vector, scars)
        for name, value in expected.items():
            filed = indices.get(name)
            if not isinstance(filed, (int, float)) \
                    or round(filed, 4) != round(value, 4):
                problems.append(
                    f"export: indices {name} files {filed!r}, not "
                    f"the recomputed {round(value, 4)}")
    evidence = export.get("evidence") or {}
    for key in ("supporting_record_ids", "contradiction_record_ids"):
        ids = evidence.get(key)
        if not isinstance(ids, list) \
                or not all(isinstance(i, str) and i for i in ids or []):
            problems.append(
                f"export: evidence {key} is not a list of record ids")
    bond_id = export.get("bond_id")
    for kind in ("vows", "hurts"):
        rows = export.get(kind)
        if not isinstance(rows, list):
            problems.append(f"export: {kind} is not a list")
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get(
                    "vow_id" if kind == "vows" else "hurt_id"):
                problems.append(f"export: a {kind} row files no id")
            elif row.get("bond_id") != bond_id:
                problems.append(
                    f"export: a {kind} row is bound to "
                    f"{row.get('bond_id')!r}, not this bond {bond_id!r}")
    excerpt = export.get("excerpt")
    if not isinstance(excerpt, list) or not excerpt \
            or not all(isinstance(line, str) for line in excerpt or []):
        problems.append("export: the excerpt is not a list of lines")
    elif export.get("domain") and export.get("status") \
            and isinstance(export.get("indices"), dict) \
            and isinstance(export.get("active_modes"), dict):
        head = _expected_excerpt_head(export)
        if list(excerpt[:7]) != head:
            problems.append(
                "export: the excerpt's head does not agree with "
                "the structured sections (domain / status / "
                "indices / modes)")
        unresolved = (export.get("scars") or {}).get(
            "unresolved_hurts") or []
        if unresolved:
            line = f"unresolved_hurts={','.join(unresolved)}"
            if line not in excerpt:
                problems.append(
                    "export: the excerpt carries no unresolved-"
                    "hurts line for its filed scars")
        vows = export.get("vows") or []
        if vows:
            line = "vows=" + " | ".join(
                f"{v['vow_kind']}:{v['state']}:{v['vow_text']}"
                for v in vows)
            if line not in excerpt:
                problems.append(
                    "export: the excerpt carries no vows line "
                    "for its filed vows")
    problems.extend(_forbidden_problems(export))
    return problems


def bridge_row_problems(row: Any) -> list[str]:
    """Judge one bridge-section filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["bridge row: the row is empty"]
    problems: list[str] = []
    for key in ("id", "section", "source", "carries"):
        if not row.get(key):
            problems.append(f"bridge row {row.get('id')}: no {key} filed")
    return problems


def reservation_problems(row: Any) -> list[str]:
    """Judge one reservation filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["reservation: the row is empty"]
    problems: list[str] = []
    for key in ("id", "reserved_to", "not_carried"):
        if not row.get(key):
            problems.append(
                f"reservation {row.get('id')}: no {key} filed")
    return problems


# ── The bridge itself: an export built FROM the production graph ────────────

def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def export_bond(graph: BondGraph, bond_id: str) -> dict[str, Any]:
    """Export one bond as WYRD-consumable data.

    Every value is read from the production graph:
    the edge's fields, the edge's own index
    methods, the graph's vow/hurt stores, and
    ``BondGraph.excerpt()`` itself for the excerpt
    lines. Nothing is recomputed here and no bond
    logic is duplicated. The result is plain data
    (dicts / lists / strings / numbers / null) —
    JSON-serialisable as it stands."""
    edge = graph.edges[bond_id]
    vows = [v for v in graph.vows.values() if v.bond_id == bond_id]
    hurts = [h for h in graph.hurts.values() if h.bond_id == bond_id]
    return {
        "format": BOND_BRIDGE_FORMAT,
        "source_format": SOURCE_FORMAT,
        "bond_id": edge.bond_id,
        "entity_a": edge.entity_a,
        "entity_b": edge.entity_b,
        "domain": edge.domain.value,
        "status": edge.status.value,
        "vector": {name: getattr(edge.vector, name)
                   for name in VECTOR_DIMS},
        "indices": {
            "closeness": round(edge.closeness_index(), 4),
            "sacred": round(edge.sacred_bond_index(), 4),
            "rupture": round(edge.rupture_index(), 4),
        },
        "scars": {
            "repair_debt": edge.scars.repair_debt,
            "vow_strain": edge.scars.vow_strain,
            "unresolved_hurts": list(edge.scars.unresolved_hurts),
        },
        "constraints": {
            "exclusivity_mode": edge.constraints.exclusivity_mode,
            "intimacy_ceiling": edge.constraints.intimacy_ceiling,
            "boundary_profile_id": edge.constraints.boundary_profile_id,
        },
        "active_modes": {
            "relational_mode": edge.active_modes.relational_mode,
            "emotional_weather": edge.active_modes.emotional_weather,
        },
        "governance": {
            "review_required_for_large_shift":
                edge.governance.review_required_for_large_shift,
            "synthetic_claim_guard": edge.governance.synthetic_claim_guard,
        },
        "chronology": {
            "formed_at": _iso(edge.chronology.formed_at),
            "last_major_shift_at": _iso(edge.chronology.last_major_shift_at),
            "last_contact_at": _iso(edge.chronology.last_contact_at),
        },
        "evidence": {
            "supporting_record_ids":
                list(edge.evidence.supporting_record_ids),
            "contradiction_record_ids":
                list(edge.evidence.contradiction_record_ids),
        },
        "vows": [{
            "vow_id": v.vow_id, "bond_id": v.bond_id,
            "vow_text": v.vow_text, "vow_kind": v.vow_kind,
            "created_from_record_id": v.created_from_record_id,
            "strength": v.strength, "state": v.state.value,
            "witness_entities": list(v.witness_entities),
            "kept_by_events": list(v.kept_by_events),
            "broken_by_events": list(v.broken_by_events),
        } for v in vows],
        "hurts": [{
            "hurt_id": h.hurt_id, "bond_id": h.bond_id,
            "source_event_id": h.source_event_id,
            "hurt_kind": h.hurt_kind, "severity": h.severity,
            "notes": h.notes,
        } for h in hurts],
        "excerpt": graph.excerpt(bond_id),
    }


def export_graph(graph: BondGraph) -> dict[str, Any]:
    """Export every bond in a graph, in the
    graph's filing order, under the same format."""
    return {
        "format": BOND_BRIDGE_FORMAT,
        "source_format": SOURCE_FORMAT,
        "bonds": [export_bond(graph, bond_id) for bond_id in graph.edges],
    }


# ── The round-trip proof on the parity fixtures ──────────────────────────────

#: The filed expectation for the fixture export —
#: deterministic values only (counts, ids, vector,
#: indices, excerpt); chronology values are
#: wall-clock stamps and are never filed as
#: metrics (the estate's rule).
BRIDGE_EXPECTED: dict[str, Any] = {
    "bond_id": PARITY_BOND_ID,
    "entity_a": "user:emberwick-keeper",
    "entity_b": "persona:emberwick-lantern",
    "domain": "companion",
    "status": "repairing",
    "vector_dims": 10,
    "trust": 0.6, "warmth": 0.6,
    "closeness": 0.4375, "sacred": 0.5333, "rupture": 0.2083,
    "repair_debt": 0.35, "safety": 0.725,
    "vows": 1, "hurts": 1,
    "evidence_supporting": 1, "evidence_contradiction": 0,
    "unresolved_hurts": 1,
    "excerpt_lines": 9,
}


def fixture_graph() -> BondGraph:
    """TF-143's parity lifecycle, rebuilt through
    the PRODUCTION graph exactly as
    ``run_bond_parity`` runs it (synthetic
    Emberwick fixtures, TF-017)."""
    graph = BondGraph()
    graph.add_edge(parity_edge())
    graph.add_vow(Vow(
        vow_id="vow:emberwick-001", bond_id=PARITY_BOND_ID,
        vow_text="keep the lantern lit through winter",
        vow_kind="promise",
        created_from_record_id="rec:synthetic-emberwick-000"))
    graph.apply_event(
        PARITY_BOND_ID, warmth_delta=0.2, trust_delta=0.2,
        devotion_delta=0.1,
        source_record_id="rec:synthetic-emberwick-001")
    graph.add_hurt(Hurt(
        hurt_id="hurt:emberwick-001", bond_id=PARITY_BOND_ID,
        source_event_id="evt:synthetic-emberwick-001",
        hurt_kind="false_memory", severity="high"))
    return graph


def run_bridge_fixture() -> dict[str, Any]:
    """Build the parity graph, export its bond,
    and file the deterministic metrics of the
    export (chronology values excluded — stamps,
    not metrics)."""
    graph = fixture_graph()
    export = export_bond(graph, PARITY_BOND_ID)
    metrics = {
        "bond_id": export["bond_id"],
        "entity_a": export["entity_a"],
        "entity_b": export["entity_b"],
        "domain": export["domain"],
        "status": export["status"],
        "vector_dims": len(export["vector"]),
        "trust": round(export["vector"]["trust"], 4),
        "warmth": round(export["vector"]["warmth"], 4),
        "closeness": export["indices"]["closeness"],
        "sacred": export["indices"]["sacred"],
        "rupture": export["indices"]["rupture"],
        "repair_debt": round(export["scars"]["repair_debt"], 4),
        "safety": round(export["vector"]["safety"], 4),
        "vows": len(export["vows"]),
        "hurts": len(export["hurts"]),
        "evidence_supporting": len(export["evidence"]["supporting_record_ids"]),
        "evidence_contradiction":
            len(export["evidence"]["contradiction_record_ids"]),
        "unresolved_hurts": len(export["scars"]["unresolved_hurts"]),
        "excerpt_lines": len(export["excerpt"]),
    }
    return {"metrics": metrics, "export": export}


# ── The filed probes ─────────────────────────────────────────────────────────

BOND_BRIDGE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_export_passes",
     "kind": "clean",
     "detail": "the fixture export + the filed bridge / "
               "reservation rows pass their validators, and "
               "the export JSON-round-trips unchanged"},
    {"id": "missing_bond_id_is_named",
     "kind": "missing_id",
     "detail": "an export with its bond_id emptied is named"},
    {"id": "index_mismatch_is_named",
     "kind": "index_wrong",
     "detail": "an export whose closeness is perturbed is "
               "named against the value recomputed from its "
               "own vector + scars"},
    {"id": "excerpt_drift_is_named",
     "kind": "excerpt_drift",
     "detail": "an export whose excerpt status line is "
               "perturbed is named against its structured "
               "sections"},
    {"id": "forbidden_world_state_is_named",
     "kind": "forbidden_key",
     "detail": "an export carrying a world_state key is "
               "named — the world state is WYRD-side, never "
               "the bridge's to file"},
    {"id": "vow_bound_to_another_bond_is_named",
     "kind": "vow_mismatch",
     "detail": "an export whose vow is bound to a different "
               "bond is named"},
    {"id": "vector_out_of_range_is_named",
     "kind": "vector_range",
     "detail": "an export whose trust files 1.4 is named"},
)


def run_bond_bridge_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    deep copies."""
    fixture = run_bridge_fixture()["export"]
    results: list[dict[str, Any]] = []
    for probe in BOND_BRIDGE_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems = export_problems(fixture)
            for row in BRIDGE_ROWS:
                problems += bridge_row_problems(row)
            for row in RESERVATIONS:
                problems += reservation_problems(row)
            round_tripped = json.loads(json.dumps(fixture))
            passed = problems == [] and round_tripped == fixture
        elif kind == "missing_id":
            data = copy.deepcopy(fixture)
            data["bond_id"] = ""
            passed = bool(export_problems(data))
        elif kind == "index_wrong":
            data = copy.deepcopy(fixture)
            data["indices"]["closeness"] = 0.99
            passed = bool(export_problems(data))
        elif kind == "excerpt_drift":
            data = copy.deepcopy(fixture)
            data["excerpt"][1] = "status=active"
            passed = bool(export_problems(data))
        elif kind == "forbidden_key":
            data = copy.deepcopy(fixture)
            data["world_state"] = {"mill": "turning"}
            passed = bool(export_problems(data))
        elif kind == "vow_mismatch":
            data = copy.deepcopy(fixture)
            data["vows"][0]["bond_id"] = "bond:elsewhere"
            passed = bool(export_problems(data))
        elif kind == "vector_range":
            data = copy.deepcopy(fixture)
            data["vector"]["trust"] = 1.4
            passed = bool(export_problems(data))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_bond_bridge_markdown() -> str:
    """Render the bridge filing deterministically —
    the exact text filed as
    docs/architecture/BOND_BRIDGE.md."""
    fixture = run_bridge_fixture()
    metrics = fixture["metrics"]
    export = fixture["export"]
    lines: list[str] = []
    lines.append("# The Bond-Graph Bridge — bonds as WYRD-consumable data (TF-146)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/knowledge/bond_bridge.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.knowledge.bond_bridge`. The "
        "checker (`bond_bridge_problems()`) re-derives every "
        "claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{BOND_BRIDGE_FORMAT}`")
    lines.append(f"Source format: `{SOURCE_FORMAT}` (TF-143)")
    lines.append("Antecedent: TF-139 **ADOPT_AS_LAYER** — the bridge is live work")
    lines.append(f"Exported sections: {len(BRIDGE_ROWS)}")
    lines.append(f"Reservations: {len(RESERVATIONS)}")
    lines.append(f"Probes: {len(BOND_BRIDGE_PROBES)}/{len(BOND_BRIDGE_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: "
        "**no bridge existed** — no bridge module under "
        "`src/thoughtforge/knowledge/`, no bridge doc, and the "
        "production `BondGraph`'s only outward shape was "
        "`excerpt()`'s human-readable lines: nothing a "
        "WYRD-side consumer could consume as data (no ids as "
        "fields, no vector, no scars, no vow/hurt records, no "
        "evidence links, no versioned format). The antecedent "
        "stood clean live: TF-139's verdict ADOPT_AS_LAYER "
        "(`bond_graph_verdict_problems()` []), TF-143's "
        "production graph clean (parity excerpt 9 lines), "
        "TF-145's boundary clean with the bridge row filing "
        "`not-built-tf146`, and TF-132's panel PASS (18 cases "
        "/ 9 consistent / 8 contradicted / 1 "
        "honest-unanswerable).")
    lines.append("")
    lines.append("## 2. The export shape")
    lines.append("")
    lines.append(
        "`export_bond(graph, bond_id)` builds the export FROM "
        "the production `BondGraph` (TF-143): every value is "
        "read from the graph — the edge's fields, the edge's "
        "own index methods, the graph's vow/hurt stores, and "
        "`BondGraph.excerpt()` itself for the excerpt lines. "
        "No bond logic is duplicated in the bridge. The export "
        "is plain data (dicts / lists / strings / numbers / "
        "null), JSON-serialisable as it stands, under the "
        "versioned format above.")
    lines.append("")
    lines.append("| Section | Source in the production graph | Carries |")
    lines.append("|---|---|---|")
    for row in BRIDGE_ROWS:
        lines.append(
            f"| {row['section']} | {row['source']} | {row['carries']} |")
    lines.append("")
    lines.append("## 3. Round-trip proof (synthetic fixtures, TF-017)")
    lines.append("")
    lines.append(
        "A production `BondGraph` built over TF-139/TF-143's "
        "Emberwick parity fixtures (edge → vow → care event → "
        "high hurt, exactly `run_bond_parity`'s lifecycle) is "
        "exported, and the export is proven to carry the "
        "filing's claims:")
    lines.append("")
    lines.append(f"- bond: {metrics['bond_id']} ({metrics['entity_a']} ↔ {metrics['entity_b']})")
    lines.append(f"- domain {metrics['domain']} / status {metrics['status']}")
    lines.append(
        f"- vector: {metrics['vector_dims']} dims (trust "
        f"{metrics['trust']} / warmth {metrics['warmth']} / "
        f"safety {metrics['safety']})")
    lines.append(
        f"- indices: closeness {metrics['closeness']} / sacred "
        f"{metrics['sacred']} / rupture {metrics['rupture']} — "
        "TF-143's parity values, unmoved")
    lines.append(
        f"- scars: repair_debt {metrics['repair_debt']} / "
        f"unresolved hurts {metrics['unresolved_hurts']}")
    lines.append(
        f"- vows {metrics['vows']} / hurts {metrics['hurts']} "
        "as full records bound to this bond")
    lines.append(
        f"- evidence links: {metrics['evidence_supporting']} "
        f"supporting / {metrics['evidence_contradiction']} "
        "contradiction record ids — ids only, never records")
    lines.append(f"- excerpt lines: {metrics['excerpt_lines']}, verbatim:")
    lines.append("")
    lines.append("```")
    lines.extend(export["excerpt"])
    lines.append("```")
    lines.append("")
    lines.append(
        "The export JSON-round-trips unchanged "
        "(`json.loads(json.dumps(export)) == export`, pinned "
        "by the clean probe and the tests), and a second "
        "build exports identically except its chronology "
        "stamps — timestamps are values a consumer may read, "
        "never metrics this filing pins.")
    lines.append("")
    lines.append("## 4. What the bridge does NOT carry")
    lines.append("")
    for row in RESERVATIONS:
        lines.append(f"- **{row['id']}** (reserved to {row['reserved_to']}): {row['not_carried']}")
    lines.append("")
    lines.append(
        "The reservation is enforced, not just filed: the "
        "pure validator names any export carrying a forbidden "
        "key (`world_state` / `world` / `canon` / `canon_data` "
        "/ `facts` / `store_records`) at any depth — probe "
        "`forbidden_world_state_is_named`.")
    lines.append("")
    lines.append("## 5. TF-145's filed status, handled in the open")
    lines.append("")
    lines.append(
        "TF-145's boundary doc files this bridge as its one "
        "not-built row (`bond_excerpt_bridge`, status "
        "`not-built-tf146`) — and **it stands as filed**. The "
        "estate's precedent is uniform: TF-143 did not rewrite "
        "TF-139's verdict module (which still files TF-143 as "
        "the rebuild's owner, after the rebuild landed) and "
        "TF-144 did not rewrite TF-140/TF-142's verdict "
        "modules; a filed slice's doc records what was true "
        "and measured when it was filed, and the slice that "
        "changes the estate files the change in its own "
        "module — this one. (TF-137's in-the-open updates were "
        "to `benchmarks/harness.py`, a living registry of the "
        "current estate, not a filed slice doc.) TF-151 / "
        "TF-152 re-measure the estate with the verdicts "
        "landed. This module's checker pins TF-145's row "
        "still naming TF-146 as the bridge's owner — the "
        "antecedent the bridge stands on — and the bridge was "
        "shaped to leave TF-145's live measurements true (no "
        "WYRD-named file, no live-client token, no consumer "
        "name under `src/`).")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-146 has **no share** of TF-016's D-WING-1: the "
        "entry's current text (checked live for this slice) "
        "assigns the verdicts (TF-139..TF-142), the adoptions "
        "(TF-143), the retirements (TF-144), research hygiene "
        "(TF-148) and the wing's tests (TF-149) — the "
        "bridge's ownership is filed in TF-139's verdict "
        "conditions, TF-143's filing and TF-145's boundary "
        "row, not in D-WING-1. Per the TF-145 pattern's rule "
        "(a slice touches the docket only for its own filed "
        "share), this slice does not edit the docket and adds "
        "no entry; TF-016 totals stand unchanged (137 entries "
        "— 33 OPEN / 1 OPEN-ENVIRONMENT / 100 FIXED / 3 "
        "CLOSED-BY-VERDICT), filed in the open here.")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in BOND_BRIDGE_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Fixture grade only: the round-trip proof runs on "
        "synthetic Emberwick fixtures (TF-017); no live WYRD "
        "consumer was read — a live system consuming the "
        "bridge is NOT PROVEN (TF-173 class), never claimed.")
    lines.append(
        "- The export is a shape, not a transport: no client, "
        "no feed, no protocol session is built here (TF-145 "
        "files transport as the consumer side's; Part IX "
        "owns bindings).")
    lines.append(
        "- The indices in an export are the production "
        "edge's own, under TF-143's filed equal-weights "
        "decision — the bridge files no weighting of its own.")
    lines.append(
        "- No canon data was touched; no README.md anywhere "
        "was touched.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_bond_bridge_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / BOND_BRIDGE_DOC
    doc.write_text(render_bond_bridge_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.knowledge.bond_bridge`."""
    doc = write_bond_bridge_doc()
    print(f"bond bridge regenerated: -> {BOND_BRIDGE_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def bond_bridge_problems() -> list[str]:
    """Re-derive TF-146's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    row_ids = [r["id"] for r in BRIDGE_ROWS]
    if len(row_ids) != len(set(row_ids)) or len(BRIDGE_ROWS) != 9:
        problems.append("the BRIDGE_ROWS filing is not nine unique rows")
    for row in BRIDGE_ROWS:
        problems.extend(bridge_row_problems(row))
    res_ids = [r["id"] for r in RESERVATIONS]
    if len(res_ids) != len(set(res_ids)) or len(RESERVATIONS) != 4:
        problems.append("the RESERVATIONS filing is not four unique rows")
    for row in RESERVATIONS:
        problems.extend(reservation_problems(row))

    # The antecedent: TF-139 adopted the bond graph
    # as a layer — without it this bridge is not
    # live work.
    from thoughtforge.knowledge.bond_graph_verdict import (
        VERDICT as BOND_VERDICT,
    )

    if BOND_VERDICT.get("verdict") != "ADOPT_AS_LAYER":
        problems.append(
            "antecedent: TF-139 no longer files ADOPT_AS_LAYER — "
            "the bridge's condition is gone")

    # TF-145's boundary row stands as filed (the
    # historical-gate handling, §5): it names
    # TF-146 as the bridge's owner.
    boundary_text = _BOUNDARY.read_text(encoding="utf-8")
    if "bond_excerpt_bridge" not in boundary_text \
            or "not-built-tf146" not in boundary_text:
        problems.append(
            "boundary: TF-145's filed bridge row no longer stands "
            "as filed — TF-146 leaves the boundary doc as its "
            "historical record (see §5)")

    # The fixture export carries the filing exactly:
    # TF-143's parity values unmoved, the excerpt
    # verbatim, the export valid and JSON-stable.
    fixture = run_bridge_fixture()
    export = fixture["export"]
    problems.extend(export_problems(export))
    if fixture["metrics"] != BRIDGE_EXPECTED:
        problems.append(
            f"fixture: the export files {fixture['metrics']}, not "
            f"the filed {BRIDGE_EXPECTED}")
    if tuple(export["excerpt"]) != PARITY_EXPECTED_EXCERPT:
        problems.append("fixture: the exported excerpt drifted "
                        "from TF-143's parity filing")
    if export["indices"] != {
            "closeness": PARITY_EXPECTED["closeness"],
            "sacred": PARITY_EXPECTED["sacred"],
            "rupture": PARITY_EXPECTED["rupture"]}:
        problems.append("fixture: the exported indices moved "
                        "from TF-143's parity metrics")
    if json.loads(json.dumps(export)) != export:
        problems.append("fixture: the export does not JSON-"
                        "round-trip unchanged")

    # Determinism: a second build exports identically
    # except the chronology stamps.
    second = run_bridge_fixture()["export"]
    first_rest = {k: v for k, v in export.items() if k != "chronology"}
    second_rest = {k: v for k, v in second.items() if k != "chronology"}
    if first_rest != second_rest:
        problems.append("fixture: a second build's export differs "
                        "beyond its chronology stamps")

    # The probes all behave.
    probes = run_bond_bridge_probes()
    if len(probes) != len(BOND_BRIDGE_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the bond-bridge probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / BOND_BRIDGE_DOC
    if not doc.exists():
        problems.append(f"the write-up {BOND_BRIDGE_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != render_bond_bridge_markdown():
        problems.append(
            f"the filed doc {BOND_BRIDGE_DOC} is not the render "
            "verbatim (regenerate with python3 -m "
            "thoughtforge.knowledge.bond_bridge)")
    return problems


def bond_bridge_as_data() -> dict[str, Any]:
    """The whole bridge filing as plain data, for
    consumers."""
    fixture = run_bridge_fixture()
    probes = run_bond_bridge_probes()
    return {
        "format": BOND_BRIDGE_FORMAT,
        "doc": BOND_BRIDGE_DOC,
        "source_format": SOURCE_FORMAT,
        "sections": [dict(r) for r in BRIDGE_ROWS],
        "reservations": [dict(r) for r in RESERVATIONS],
        "fixture_metrics": fixture["metrics"],
        "fixture_export": fixture["export"],
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "BOND_BRIDGE_DOC",
    "BOND_BRIDGE_FORMAT",
    "BOND_BRIDGE_PROBES",
    "BRIDGE_EXPECTED",
    "BRIDGE_ROWS",
    "FORBIDDEN_KEYS",
    "RESERVATIONS",
    "SOURCE_FORMAT",
    "VECTOR_DIMS",
    "bond_bridge_as_data",
    "bond_bridge_problems",
    "bridge_row_problems",
    "export_bond",
    "export_graph",
    "export_problems",
    "fixture_graph",
    "main",
    "render_bond_bridge_markdown",
    "reservation_problems",
    "run_bond_bridge_probes",
    "run_bridge_fixture",
    "write_bond_bridge_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
