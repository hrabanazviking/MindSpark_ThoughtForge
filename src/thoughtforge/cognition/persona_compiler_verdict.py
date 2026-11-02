"""
TF-141 — the persona compiler verdict, stated as verifiable data.

The roadmap asks for a verdict on wyrdforge's persona
compilation against the shipping config/persona path
(TF-009 / TF-030) — *adopt the compiler's shape where it
proves better*. The measured start, probed live this tick
on this tree before anything was built:

  * **The wing cannot import** (TF-006 / D-WING-1, still
    true live): ``research_data/src/wyrdforge/models`` does
    not exist, and ``persona_compiler.py`` imports
    BondEdge / CanonicalFactRecord / MemoryRecord /
    SymbolicTraceRecord / PersonaMode / PersonaPacket /
    PersonaSourceItem / TraitSignal from that missing
    package. Its schemas exist and parse: persona_packet
    (the packet shape), canonical_fact_record and
    symbolic_trace_record (the record shapes the compiler
    reads).
  * **The compiler** (``PersonaCompiler.compile``, 100
    lines) turns a list of memory records (+ an optional
    BondEdge) into a PersonaPacket: canonical facts become
    ``subject.key=value`` lines — identity_core when the
    fact's domain is identity/style/mission, else
    truth_anchor_points (caps 10 / 16); facts whose key is
    temperament/tone/value/mood also feed trait buckets
    weighted by the record's truth.confidence (summed,
    TraitSignal weight min(1.0, sum), cap 8, each trait
    carrying its supporting record ids); symbolic traces
    become symbolic_context lines (cap 8); records that
    are neither canonical nor symbolic AND not approved
    become uncertainty_points (cap 8) — the elif chain
    means an unapproved CANONICAL fact is never surfaced
    as uncertainty: it lands in truth_anchor_points as if
    it were truth (measured live by the spike below; a
    filed compiler defect, not a shape to adopt). Every
    record becomes a source_item with its confidence
    (cap 20). The tone_contract is 3 base lines + one
    mode-specific line for CODING_GUIDE / WORLD_SEER /
    COMPANION; a BondEdge adds three relationship lines
    (mode / weather / closeness) to response_guidance —
    and relationship_excerpt is filed as
    response_guidance[:6], i.e. it DUPLICATES the guidance
    (bond lines + mode line), it is not a bond-only
    excerpt. ``token_budget_hint`` (default 800) is
    carried onto the packet and never enforced: nothing
    trims by it.
  * **The shipping path** (TF-009's census + TF-030):
    a hand-authored PersonalityCoreRecord (configs /
    YAML) is loaded and snapshotted into a
    PersonaExpectation — traits / speech_style /
    behavior_rules / avoid carried verbatim, five tuning
    weights carried with NO prompt effect (TF-030 filed
    that deferral to this slice by name). inject_persona
    delivers the expressive lists to every prompt as
    steering lines capped at MAX_STEERING_ENTRIES=6;
    avoid steers separately through scaffold.avoid and is
    CHECKED deterministically against responses
    (check_persona_response / score_persona_turns). It
    has no provenance beyond the core's record_id, no
    trait weighting in effect, no uncertainty, no
    identity/truth-anchor split, no modes, no symbolic
    or relationship input — and the compiler has none of
    its delivery or checking. The two shapes are
    complementary, which is what the rows below measure.

Module home, filed in the open: TF-139/TF-140's verdicts
live in ``knowledge/`` because their subjects live there
(the store, the retrieval engine). This verdict's subject
is the cognition layer's persona path — TF-030's
``cognition/persona.py`` against the compiler that would
replace or extend it — so the verdict lives in
``cognition/`` beside it. One coupling follows, fixed in
the open exactly as TF-140 fixed its own: TF-139's
checker and tests grep the shipping knowledge + cognition
sources for ``bond`` / ``relationship`` vocabulary (the
shipping STORE has no bond graph) — this module
legitimately files the compiler's BondEdge /
relationship vocabulary, so TF-139's verdict-module
exclusion is extended to this module by name in both
places; the premise it guards is unchanged.

What this module files:

  * ``COMPARISON_ROWS`` — the comparison as literal
    data: twelve facets both paths claim, each filing
    the compiler's shape, the shipping path's shape,
    and a finding decided by the live probes in
    ``run_persona_spike()``.
  * ``VERDICT`` — **MERGE** (adopt-in-part), with filed
    reasons and filed conditions: the compiler's shape
    is adopted where it proves better (seven facets),
    the shipping path's delivery + checking are kept
    where they prove better (two facets); TF-143 owns
    implementing the merge (see the conditions — they
    name exactly what TF-143 must build, including the
    two compiler defects it must NOT import).
  * **The working spike** — spike dataclasses rebuilt
    FROM THE SCHEMAS inside this module, faithfully:
    the packet shapes in full (PersonaPacket /
    PersonaSourceItem / TraitSignal + the PersonaMode
    enum from persona_packet.schema.json), the payload
    / content / truth shapes the compiler reads
    (CanonicalFactPayload, CanonicalFactContent,
    SymbolicTracePayload, SymbolicTraceContent,
    TruthMeta + the ApprovalState enum from the two
    record schemas — the two schemas' TruthMeta /
    ApprovalState $defs are identical, checked live),
    namespaced ``Spike*`` — explicitly NOT the missing
    ``wyrdforge.models`` package and NOT production.
    The input records are spike records reduced to what
    the compiler actually reads (record_id, record_type,
    content.summary + content.structured_payload,
    truth.confidence + truth.approval_state) with their
    nested payload/content/truth field sets checked
    live against the schemas; the BondEdge input is
    TF-139's own spike edge, reused as-is. And
    ``SpikePersonaCompiler``, the compiler's compile()
    re-implemented faithfully (same branching, same
    caps, same arithmetic, same elif quirk).
  * ``run_persona_spike()`` — the SAME synthetic ground
    truth through both paths: TF-017's invented Wrenna /
    Lantern Hollow world (the TF-030 fixture persona,
    synthetic only — no canon anywhere) expressed once
    as compiler records (+ a TF-139 spike bond edge,
    three modes, and a generated caps run) and once as
    the shipping PersonaExpectation (steering delivered
    by inject_persona into a real scaffold, responses
    checked by score_persona_turns over TF-030's own
    fixture turns). Every filed metric is reproduced by
    this function and pinned in tests.
  * ``persona_compiler_verdict_problems()`` — the
    checker: re-derives every claim above from the live
    tree and names any that stops being true.
  * ``PERSONA_PROBES`` — seven filed probes against the
    pure validators with perturbed copies (real filings
    never perturbed).

Filed limits, stated plainly:

  * Spike grade only: the spike proves the compiler's
    filed arithmetic runs deterministically on synthetic
    fixtures and measures both paths over the same
    ground truth. It is NOT production, NOT the merge —
    TF-143 owns implementing the adopted facets;
    TF-147 consumes the merged output for Rúnmál cards
    (this verdict answers its "if adopted" with
    adopt-in-part); TF-149 owns the wing's tests.
  * Fixture grade only: the ground truth is TF-017 /
    TF-030's synthetic Wrenna world. No live model was
    read; whether a live model honours either shape is
    NOT PROVEN (TF-173 class), never claimed.
  * The compiler's two measured defects (unapproved
    canonical facts promoted to truth anchors by the
    elif chain; relationship_excerpt duplicating
    response_guidance) are filed as defects for TF-143
    to NOT import — the verdict adopts the compiler's
    facets, not its bugs.

The human-readable write-up is
``docs/architecture/PERSONA_COMPILER_VERDICT.md``,
filed verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.cognition.persona_compiler_verdict``).
Proof grade: FIXTURE/STATIC — schemas + compiler shell
+ shipping hooks re-measured live on this VM over
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
    MAX_STEERING_ENTRIES,
    PERSONA_FIXTURE_TURNS,
    PersonaExpectation,
    inject_persona,
    score_persona_turns,
)
from thoughtforge.knowledge.bond_graph_verdict import (
    BondDomain,
    SpikeBondEdge,
    SpikeBondVector,
)
from thoughtforge.knowledge.models import CognitionScaffold

PERSONA_COMPILER_VERDICT_FORMAT = "thoughtforge-persona-compiler-verdict/1"
PERSONA_COMPILER_VERDICT_DOC = "docs/architecture/PERSONA_COMPILER_VERDICT.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_COMPILER = _WING / "services" / "persona_compiler.py"
_SCHEMAS = _WING / "schemas"

#: The verdicts this slice may file (TF-139's vocabulary;
#: MERGE here is the roadmap's adopt-where-it-proves-better).
VERDICTS: tuple[str, ...] = ("ADOPT_AS_LAYER", "MERGE", "RETIRE")

#: The findings a comparison row may file.
FINDINGS: tuple[str, ...] = (
    "compiler-wins", "shipping-wins", "different", "overlap")


# ── The comparison, as literal data ──────────────────────────────────────────

COMPARISON_ROWS: tuple[dict[str, Any], ...] = (
    {"dimension": "expressive_coverage",
     "compiler": "a compiled packet of typed sections built from "
                 "records: identity_core lines, evidence-weighted "
                 "active_traits, a tone_contract, symbolic_context, "
                 "response_guidance — traits/style/behaviour reach "
                 "the packet only as facts about the persona "
                 "(identity/style/mission domains, trait keys), "
                 "never as verbatim authored lists",
     "shipping_path": "the authored core's three expressive lists "
                      "verbatim (traits / speech_style / "
                      "behavior_rules, 3 + 3 + 3 entries on the "
                      "fixture) delivered as three steering lines — "
                      "narrower (no symbolic, no relationship, no "
                      "uncertainty sections) but the author's exact "
                      "words",
     "finding": "different"},
    {"dimension": "provenance",
     "compiler": "every input record becomes a source_item "
                 "(record_id + item_type + summary text + the "
                 "record's truth.confidence, cap 20 — 14/14 carried "
                 "on the spike run), and every active trait carries "
                 "its supporting_record_ids",
     "shipping_path": "the expectation carries only the core's own "
                      "record_id — no per-fact provenance, no "
                      "confidences, nothing a scorer can trace a "
                      "steering line back to",
     "finding": "compiler-wins"},
    {"dimension": "trait_weighting",
     "compiler": "traits are DERIVED from evidence: records whose "
                 "fact_key is temperament/tone/value/mood add their "
                 "truth.confidence to the bucket named by fact_value; "
                 "the spike's 'steady' sums 0.6 + 0.5 = 1.1 and files "
                 "at the min(1.0, sum) cap with both supporting ids; "
                 "traits sort by weight, cap 8",
     "shipping_path": "traits are a static authored list, unweighted "
                      "in the prompt; the core's five tuning weights "
                      "are carried on the expectation with NO prompt "
                      "effect at all (TF-030 filed the deferral to "
                      "TF-141 by name) — the numbers steer nothing",
     "finding": "compiler-wins"},
    {"dimension": "uncertainty_surfacing",
     "compiler": "unapproved records surface as uncertainty_points "
                 "(cap 8): the spike's pending + quarantined generic "
                 "records file 2 points. Filed defect: the elif "
                 "chain checks canonical/symbolic BEFORE approval, "
                 "so the spike's PENDING canonical fact never "
                 "surfaces — it lands in truth_anchor_points as "
                 "truth (see identity_vs_truth_anchor_separation)",
     "shipping_path": "no uncertainty surface exists: the authored "
                      "core is taken as in force, and the store's "
                      "approval states never reach the expectation",
     "finding": "compiler-wins"},
    {"dimension": "identity_vs_truth_anchor_separation",
     "compiler": "canonical facts split by domain: identity/style/"
                 "mission -> identity_core (4 on the spike run), "
                 "everything else -> truth_anchor_points (5) — the "
                 "persona's self vs the world's facts are separate "
                 "sections. The split is only as honest as the "
                 "approval blind spot above: the pending fact is "
                 "one of the 5 anchors",
     "shipping_path": "no split: one authored core record IS the "
                      "identity, and world facts live in the "
                      "knowledge store on a different path entirely "
                      "— the persona output never files them side "
                      "by side, separated or otherwise",
     "finding": "compiler-wins"},
    {"dimension": "mode_adaptation",
     "compiler": "three modes adapt the packet: CODING_GUIDE, "
                 "WORLD_SEER and COMPANION each add one "
                 "tone_contract line + one response_guidance line "
                 "to the 3 base contract lines (4 / 1 measured per "
                 "mode, no bond); the schema files two further "
                 "modes (ritual, debrief) the compiler treats as "
                 "base-only",
     "shipping_path": "no modes: one expectation serves every "
                      "conversation and intent; adaptation is the "
                      "scaffold's per-turn business, never the "
                      "persona's",
     "finding": "compiler-wins"},
    {"dimension": "relationship_context",
     "compiler": "an optional BondEdge compiles to three guidance "
                 "lines (relationship_mode / relationship_weather / "
                 "closeness_index at 2dp — builder / calm / 0.44 on "
                 "the spike's TF-139 edge). Filed defect: "
                 "relationship_excerpt is response_guidance[:6] "
                 "verbatim — on the spike run it duplicates all 4 "
                 "guidance lines including the mode line; it is not "
                 "a bond-only excerpt",
     "shipping_path": "no relationship input exists: the expectation "
                      "is built from the personality core alone "
                      "(TF-139's bond layer, adopted as a layer, has "
                      "no consumer in the persona path)",
     "finding": "compiler-wins"},
    {"dimension": "symbolic_context",
     "compiler": "symbolic traces compile to formatted lines "
                 "(symbol=<type>;runes=<csv>;charge=<2dp>, cap 8 — "
                 "2 on the spike run) in their own packet section",
     "shipping_path": "no symbolic input or section: a symbolic "
                      "trace can only be prose inside a fact, "
                      "indistinguishable from any other steering",
     "finding": "compiler-wins"},
    {"dimension": "prompt_steering_delivered",
     "compiler": "the packet is inert data: no code in the shipping "
                 "tree consumes a PersonaPacket, injects it into a "
                 "scaffold, or renders it into a prompt — compiled "
                 "persona reaches no generation",
     "shipping_path": "inject_persona prepends the steering block to "
                      "the scaffold's fact_block and ThoughtForgeCore "
                      "runs it every turn (TF-030's wiring): on the "
                      "spike run the block lands verbatim above the "
                      "facts, first line as the duplicate guard",
     "finding": "shipping-wins"},
    {"dimension": "avoid_and_consistency_checking",
     "compiler": "no avoid concept and no response checker: the "
                 "tone_contract is prose a model may ignore, and "
                 "nothing in the wing scores a response against "
                 "the packet",
     "shipping_path": "avoid is checked deterministically: "
                      "check_persona_response matches the persona's "
                      "avoid entries (normalised, phrases / whole "
                      "words) and score_persona_turns aggregates — "
                      "on TF-030's fixture turns: 5 checked, 2 "
                      "consistent, 3 violations, score 0.4, verdict "
                      "REVIEW (never a vacuous pass)",
     "finding": "shipping-wins"},
    {"dimension": "bounds_caps_and_determinism",
     "compiler": "every section capped (identity 10, anchors 16, "
                 "traits 8, symbolic 8, uncertainty 8, source_items "
                 "20, relationship_excerpt 6, response_guidance 10 "
                 "— all measured binding on the caps run: 57 "
                 "records in, exactly the caps out) and the compile "
                 "is deterministic; but token_budget_hint (800) is "
                 "carried, never enforced — no trimming honours it",
     "shipping_path": "steering capped at MAX_STEERING_ENTRIES=6 per "
                      "list (a 9-trait expectation files exactly 6 "
                      "in its line), avoid uncapped in the "
                      "expectation but bounded by the scaffold's own "
                      "discipline, and the hooks are deterministic "
                      "pure functions — both paths bound and "
                      "reproduce; they bound different things",
     "finding": "different"},
    {"dimension": "source_of_truth_and_authoring",
     "compiler": "the persona is COMPILED from the memory record "
                 "store: change a canonical fact's confidence or "
                 "approval and the packet changes — persona as a "
                 "view over evidence, with the store as the source "
                 "of truth",
     "shipping_path": "the persona is DECLARED in one authored "
                      "config record (TF-009's census: the "
                      "personality core YAML), versioned by hand, "
                      "protected from auto-pruning — persona as a "
                      "document, with the author as the source of "
                      "truth",
     "finding": "different"},
)


# ── The verdict ──────────────────────────────────────────────────────────────

VERDICT: dict[str, Any] = {
    "verdict": "MERGE",
    "subject": "wyrdforge persona compilation (PersonaCompiler + "
               "persona_packet / canonical_fact / symbolic_trace "
               "schemas) vs the shipping config/persona path "
               "(TF-009 census + TF-030 PersonaExpectation)",
    "decided_on": "the same synthetic ground truth (TF-017 / "
                  "TF-030's Wrenna fixtures) run through both paths "
                  "by run_persona_spike() and filed in this module "
                  "+ TF-006's census + TF-030's filed deferral of "
                  "the weights question to this slice",
    "reasons": (
        "the compiler wins seven measured facets the shipping path "
        "does not have at all: per-record provenance with "
        "confidences (14/14 source items), evidence-weighted "
        "traits (steady 1.1 filed at the 1.0 cap with both "
        "supporting ids — the answer to TF-030's weights "
        "deferral), uncertainty surfacing (2 points), the "
        "identity/truth-anchor split (4 / 5), mode adaptation "
        "(three modes, 4-line contracts) and symbolic + "
        "relationship sections (2 symbolic lines, 3 bond "
        "guidance lines) — retiring or ignoring that shape "
        "throws away the estate's only compiled-persona design",
        "the shipping path wins the two facets that make a "
        "persona real: its steering is DELIVERED (inject_persona "
        "puts the block into every prompt; a PersonaPacket has "
        "no consumer anywhere in the shipping tree) and its "
        "avoid list is CHECKED deterministically "
        "(score_persona_turns over the fixture turns: 2/5 "
        "consistent, score 0.4) — the compiler's tone_contract "
        "is unchecked prose, so adopting the compiler INSTEAD "
        "of the shipping path would leave the persona "
        "uninjected and unjudged",
        "not adopt wholesale: the compiler imports two measured "
        "defects — its elif chain promotes unapproved canonical "
        "facts into truth_anchor_points (the spike's pending "
        "fact files as an anchor, never as uncertainty), "
        "relationship_excerpt duplicates response_guidance[:6] "
        "instead of filing bond-only lines, and its "
        "token_budget_hint is carried but never enforced; a "
        "merge adopts the facets and files the defects as "
        "TF-143 work, an adoption-as-is would import them",
        "not retire, not keep-shipping-alone: the shipping "
        "path's traits are static unweighted lists and its five "
        "tuning weights steer nothing (carried since TF-030 "
        "with the question expressly deferred here) — the "
        "compiler's TraitSignal arithmetic is the measured "
        "answer, and the packet's typed sections give TF-147's "
        "Rúnmál cards a compiled shape to export",
        "both paths are bounded and deterministic on the same "
        "fixtures (compiler caps bind exactly on the 57-record "
        "caps run; shipping steering caps at 6 and its checks "
        "are pure) — the merge keeps one persona path whose "
        "inputs may be compiled, whose delivery and checking "
        "are TF-030's",
    ),
    "conditions": (
        "MERGE the compiler's proven facets into the shipping "
        "persona path: TF-143 extends the shipping expectation "
        "(or lands a compiled-expectation builder beside it) "
        "carrying evidence-weighted TraitSignals with "
        "supporting record ids, source_items provenance with "
        "confidences, uncertainty_points, the identity_core / "
        "truth_anchor_points split, mode adaptation, and "
        "symbolic + relationship sections — compiled from "
        "canonical records where they exist, with TF-009's "
        "authored PersonalityCoreRecord config path remaining "
        "a supported source, never silently replaced",
        "TF-143 keeps TF-030's runtime as the merged path's "
        "delivery and judgement: inject_persona steering into "
        "the prompt and check_persona_response / "
        "score_persona_turns avoid checking are NOT replaced "
        "by a packet; how avoid is sourced in the merged shape "
        "(declared, compiled, or both) is TF-143's filed "
        "decision",
        "TF-143 must NOT import the compiler's defects: "
        "unapproved canonical / symbolic records surface as "
        "uncertainty, never silently as truth anchors (the "
        "verdict's recommendation, decided in the open there); "
        "relationship_excerpt files bond-only lines, not "
        "response_guidance[:6] duplicated; token_budget_hint "
        "is either enforced by real trimming or dropped — a "
        "hint nothing honours is not a bound",
        "TF-147 (persona compiler -> Rúnmál cards) consumes "
        "the MERGED compiled output: this verdict answers its "
        "'if adopted' with adopt-in-part, so TF-147 is live "
        "work on the merged shape, not on the wing's packet "
        "as-is",
        "TF-141 changes NO shipping behaviour and NO "
        "research-wing code: the spike lives in this verdict "
        "module only, namespaced Spike*, and is not the "
        "missing wyrdforge.models package; TF-149 owns the "
        "wing's tests",
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
    for side in ("dimension", "compiler", "shipping_path"):
        if not row.get(side):
            problems.append(f"row {name}: the {side} side is missing")
    if row.get("finding") not in FINDINGS:
        problems.append(
            f"row {name}: finding {row.get('finding')!r} is not one "
            f"of {FINDINGS}")
    return problems


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
    """The field sets of the shapes this spike
    rebuilds, parsed live from the three schemas
    (persona_packet top level + $defs; the payload /
    content / truth $defs of the two record schemas)."""
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
        "PersonaPacket": tuple(packet["properties"]),
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


# ── The spike: dataclasses rebuilt from the schemas ──────────────────────────
#
# Namespaced Spike* — NOT wyrdforge.models, NOT production.
# Packet shapes are rebuilt in full; the record inputs are
# reduced to what PersonaCompiler.compile actually reads,
# with their nested payload/content/truth field sets
# matching the schemas exactly (checked live).

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
class SpikePersonaSourceItem:
    record_id: str
    item_type: str
    text: str
    confidence: float = 0.5


@dataclass
class SpikeTraitSignal:
    trait_name: str
    weight: float = 0.5
    supporting_record_ids: list[str] = field(default_factory=list)


@dataclass
class SpikePersonaPacket:
    persona_id: str
    user_id: str
    mode: PersonaMode
    tone_contract: list[str] = field(default_factory=list)
    identity_core: list[str] = field(default_factory=list)
    active_traits: list[SpikeTraitSignal] = field(default_factory=list)
    relationship_excerpt: list[str] = field(default_factory=list)
    truth_anchor_points: list[str] = field(default_factory=list)
    uncertainty_points: list[str] = field(default_factory=list)
    symbolic_context: list[str] = field(default_factory=list)
    response_guidance: list[str] = field(default_factory=list)
    source_items: list[SpikePersonaSourceItem] = field(
        default_factory=list)
    token_budget_hint: int = 800


@dataclass
class SpikeTruthMeta:
    support_class: str = "supported"
    confidence: float = 0.5
    contradiction_status: str = "none"
    approval_state: ApprovalState = ApprovalState.PENDING


@dataclass
class SpikeCanonicalFactPayload:
    fact_subject_id: str
    fact_key: str
    fact_value: str
    value_type: str = "string"
    domain: str = "general"
    support_record_ids: list[str] = field(default_factory=list)
    supersedes_record_id: str | None = None


@dataclass
class SpikeCanonicalFactContent:
    title: str
    summary: str
    body_ref: str | None = None
    structured_payload: SpikeCanonicalFactPayload = field(
        default_factory=lambda: SpikeCanonicalFactPayload(
            fact_subject_id="", fact_key="", fact_value=""))


@dataclass
class SpikeSymbolicTracePayload:
    symbol_type: str
    rune_signature: list[str] = field(default_factory=list)
    omen_strength: float = 0.0
    ritual_charge: float = 0.0
    mood_tags: list[str] = field(default_factory=list)


@dataclass
class SpikeSymbolicTraceContent:
    title: str
    summary: str
    body_ref: str | None = None
    structured_payload: SpikeSymbolicTracePayload = field(
        default_factory=lambda: SpikeSymbolicTracePayload(
            symbol_type=""))


@dataclass
class SpikeGenericContent:
    """A non-canonical, non-symbolic record's content,
    reduced to the one field the compiler reads for
    such records (summary). Not a schema shape."""
    summary: str


@dataclass
class SpikeMemoryRecord:
    """The compiler's input, reduced to what compile()
    reads: record_id, record_type, content.summary,
    truth.confidence, truth.approval_state."""
    record_id: str
    record_type: str
    content: Any
    truth: SpikeTruthMeta = field(default_factory=SpikeTruthMeta)


@dataclass
class SpikeCanonicalFactRecord(SpikeMemoryRecord):
    content: SpikeCanonicalFactContent = field(
        default_factory=lambda: SpikeCanonicalFactContent(
            title="", summary=""))


@dataclass
class SpikeSymbolicTraceRecord(SpikeMemoryRecord):
    content: SpikeSymbolicTraceContent = field(
        default_factory=lambda: SpikeSymbolicTraceContent(
            title="", summary=""))


#: Spike shape name -> the spike dataclass, for the
#: schema-coverage checker (schema shapes only).
SPIKE_SHAPES: dict[str, Any] = {
    "PersonaPacket": SpikePersonaPacket,
    "PersonaSourceItem": SpikePersonaSourceItem,
    "TraitSignal": SpikeTraitSignal,
    "CanonicalFactPayload": SpikeCanonicalFactPayload,
    "CanonicalFactContent": SpikeCanonicalFactContent,
    "SymbolicTracePayload": SpikeSymbolicTracePayload,
    "SymbolicTraceContent": SpikeSymbolicTraceContent,
    "TruthMeta": SpikeTruthMeta,
}


def spike_field_map() -> dict[str, tuple[str, ...]]:
    """The spike dataclasses' field sets, by shape."""
    return {
        name: tuple(f.name for f in dataclasses.fields(cls))
        for name, cls in SPIKE_SHAPES.items()
    }


# ── The spike compiler: the wing's operations, faithfully ────────────────────

class SpikePersonaCompiler:
    """PersonaCompiler.compile() over the spike types:
    same branching (including the elif approval blind
    spot), same domain/trait-key sets, same confidence
    arithmetic, same mode additions, same caps, same
    relationship_excerpt duplication. Spike only."""

    def compile(
        self,
        *,
        persona_id: str,
        user_id: str,
        mode: PersonaMode,
        records: list[SpikeMemoryRecord],
        bond_edge: SpikeBondEdge | None = None,
        token_budget_hint: int = 800,
    ) -> SpikePersonaPacket:
        identity_core: list[str] = []
        truth_anchor_points: list[str] = []
        uncertainty_points: list[str] = []
        symbolic_context: list[str] = []
        tone_contract: list[str] = [
            "preserve continuity without inventing unsupported memory",
            "prefer grounded specifics over vague mythic filler",
            "mark uncertainty cleanly when canon or memory is missing",
        ]
        response_guidance: list[str] = []
        source_items: list[SpikePersonaSourceItem] = []
        trait_buckets: dict[str, float] = {}
        trait_support: dict[str, list[str]] = {}

        for record in records:
            source_items.append(SpikePersonaSourceItem(
                record_id=record.record_id,
                item_type=record.record_type,
                text=record.content.summary,
                confidence=record.truth.confidence))
            if isinstance(record, SpikeCanonicalFactRecord):
                payload = record.content.structured_payload
                line = (f"{payload.fact_subject_id}.{payload.fact_key}"
                        f"={payload.fact_value}")
                if payload.domain in {"identity", "style", "mission"}:
                    identity_core.append(line)
                else:
                    truth_anchor_points.append(line)
                if payload.fact_key in {
                        "temperament", "tone", "value", "mood"}:
                    trait_buckets[payload.fact_value] = (
                        trait_buckets.get(payload.fact_value, 0.0)
                        + record.truth.confidence)
                    trait_support.setdefault(
                        payload.fact_value, []).append(record.record_id)
            elif isinstance(record, SpikeSymbolicTraceRecord):
                sp = record.content.structured_payload
                symbolic_context.append(
                    f"symbol={sp.symbol_type};"
                    f"runes={','.join(sp.rune_signature)};"
                    f"charge={sp.ritual_charge:.2f}")
            elif record.truth.approval_state.value != "approved":
                uncertainty_points.append(record.content.summary)

        if bond_edge is not None:
            response_guidance.extend([
                f"relationship_mode="
                f"{bond_edge.active_modes.relational_mode}",
                f"relationship_weather="
                f"{bond_edge.active_modes.emotional_weather}",
                f"closeness_index={bond_edge.closeness_index():.2f}",
            ])

        active_traits = [
            SpikeTraitSignal(
                trait_name=name, weight=min(1.0, weight),
                supporting_record_ids=trait_support[name])
            for name, weight in sorted(
                trait_buckets.items(), key=lambda item: item[1],
                reverse=True)[:8]
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

        return SpikePersonaPacket(
            persona_id=persona_id,
            user_id=user_id,
            mode=mode,
            tone_contract=tone_contract,
            identity_core=identity_core[:10],
            active_traits=active_traits,
            relationship_excerpt=response_guidance[:6],
            truth_anchor_points=truth_anchor_points[:16],
            uncertainty_points=uncertainty_points[:8],
            symbolic_context=symbolic_context[:8],
            response_guidance=response_guidance[:10],
            source_items=source_items[:20],
            token_budget_hint=token_budget_hint,
        )


# ── The shared synthetic ground truth (TF-017 / TF-030's world) ──────────────

def _canonical(
    record_id: str, *, subject: str, key: str, value: str,
    domain: str, confidence: float,
    approval: ApprovalState = ApprovalState.APPROVED,
    summary: str = "",
) -> SpikeCanonicalFactRecord:
    return SpikeCanonicalFactRecord(
        record_id=record_id, record_type="canonical_fact",
        content=SpikeCanonicalFactContent(
            title=f"{subject}.{key}", summary=summary,
            structured_payload=SpikeCanonicalFactPayload(
                fact_subject_id=subject, fact_key=key,
                fact_value=value, domain=domain)),
        truth=SpikeTruthMeta(
            confidence=confidence, approval_state=approval))


def _symbolic(
    record_id: str, *, symbol: str, runes: list[str],
    charge: float, confidence: float,
) -> SpikeSymbolicTraceRecord:
    return SpikeSymbolicTraceRecord(
        record_id=record_id, record_type="symbolic_trace",
        content=SpikeSymbolicTraceContent(
            title=symbol, summary=f"a {symbol} trace",
            structured_payload=SpikeSymbolicTracePayload(
                symbol_type=symbol, rune_signature=runes,
                ritual_charge=charge)),
        truth=SpikeTruthMeta(
            confidence=confidence,
            approval_state=ApprovalState.APPROVED))


def _generic(
    record_id: str, *, summary: str, confidence: float,
    approval: ApprovalState,
) -> SpikeMemoryRecord:
    return SpikeMemoryRecord(
        record_id=record_id, record_type="episode_summary",
        content=SpikeGenericContent(summary=summary),
        truth=SpikeTruthMeta(
            confidence=confidence, approval_state=approval))


def fixture_records() -> list[SpikeMemoryRecord]:
    """Wrenna / Lantern Hollow as compiler records —
    the same ground truth TF-030's fixture core
    declares, expressed as evidence (synthetic only)."""
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


def fixture_bond_edge() -> SpikeBondEdge:
    """TF-139's spike edge shape, Wrenna-flavoured
    (synthetic): closeness (0.6+0.6+0.3+0.25)/4."""
    return SpikeBondEdge(
        bond_id="bond:wrenna-companion",
        entity_a="user:lantern-keeper",
        entity_b="persona:wrenna",
        domain=BondDomain.COMPANION,
        vector=SpikeBondVector(
            trust=0.6, warmth=0.6, familiarity=0.3, devotion=0.2,
            attraction_affinity=0.1, safety=0.9, sacred_resonance=0.3,
            playfulness=0.2, vulnerability=0.25,
            initiative_balance=0.0))


def caps_records() -> list[SpikeMemoryRecord]:
    """A generated 57-record set that binds every
    compiler cap (synthetic, deterministic)."""
    records: list[SpikeMemoryRecord] = []
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


def _packet_data(packet: SpikePersonaPacket) -> dict[str, Any]:
    """One compiled packet as plain filed data."""
    return {
        "persona_id": packet.persona_id,
        "user_id": packet.user_id,
        "mode": packet.mode.value,
        "tone_contract": list(packet.tone_contract),
        "identity_core": list(packet.identity_core),
        "active_traits": [
            {"trait_name": t.trait_name,
             "weight": round(t.weight, 4),
             "supporting_record_ids": list(t.supporting_record_ids)}
            for t in packet.active_traits],
        "relationship_excerpt": list(packet.relationship_excerpt),
        "truth_anchor_points": list(packet.truth_anchor_points),
        "uncertainty_points": list(packet.uncertainty_points),
        "symbolic_context": list(packet.symbolic_context),
        "response_guidance": list(packet.response_guidance),
        "source_items": [
            {"record_id": s.record_id, "item_type": s.item_type,
             "confidence": s.confidence}
            for s in packet.source_items],
        "token_budget_hint": packet.token_budget_hint,
    }


def _counts(packet: SpikePersonaPacket) -> dict[str, Any]:
    return {
        "identity_core": len(packet.identity_core),
        "truth_anchor_points": len(packet.truth_anchor_points),
        "active_traits": len(packet.active_traits),
        "symbolic_context": len(packet.symbolic_context),
        "uncertainty_points": len(packet.uncertainty_points),
        "source_items": len(packet.source_items),
        "tone_contract": len(packet.tone_contract),
        "response_guidance": len(packet.response_guidance),
        "relationship_excerpt": len(packet.relationship_excerpt),
    }


def run_persona_spike() -> dict[str, Any]:
    """Run the same synthetic ground truth through
    BOTH paths and file the measurements:
    the compiler spike (main COMPANION run with the
    bond edge, the three modes without it, and the
    57-record caps run) and the shipping path
    (TF-030's fixture expectation: steering lines,
    injection into a real scaffold, the fixture
    turns checked, the steering cap probed).
    Deterministic: a second run files identically."""
    compiler = SpikePersonaCompiler()
    records = fixture_records()

    main_packet = compiler.compile(
        persona_id="persona:wrenna", user_id="user:lantern-keeper",
        mode=PersonaMode.COMPANION, records=records,
        bond_edge=fixture_bond_edge())
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
    caps_packet = compiler.compile(
        persona_id="persona:cap", user_id="user:lantern-keeper",
        mode=PersonaMode.COMPANION, records=caps_records())

    expectation = FIXTURE_EXPECTATION
    steering = expectation.steering_lines()
    scaffold = CognitionScaffold(
        goal="turn goal", fact_block="Knowledge:\nx")
    injected = inject_persona(scaffold, expectation)
    report = score_persona_turns(
        expectation, [text for text, _ in PERSONA_FIXTURE_TURNS])
    cap_expectation = PersonaExpectation(
        record_id="cap",
        traits=[f"trait{i}" for i in range(9)])
    cap_line = cap_expectation.steering_lines()[0]
    steering_text = "\n".join(steering)
    shipping = {
        "record_id": expectation.record_id,
        "steering_lines": steering,
        "steering_entries": sum(
            len(getattr(expectation, name)[:MAX_STEERING_ENTRIES])
            for name in ("traits", "speech_style", "behavior_rules")),
        "avoid_entries": len(expectation.avoid),
        "avoid_in_steering": sum(
            1 for entry in expectation.avoid if entry in steering_text),
        "weights_carried": len(expectation.weights),
        "weights_in_steering": sum(
            1 for key in expectation.weights if key in steering_text),
        "injected": injected,
        "fact_block_starts_with_block": scaffold.fact_block.startswith(
            steering[0]),
        "turns_checked": report.total_turns,
        "turns_consistent": report.consistent_turns,
        "violations": report.violation_count,
        "score": report.score,
        "verdict": report.verdict,
        "steering_cap_entries": cap_line.count(";") + 1,
    }
    return {
        "compiler_main": _packet_data(main_packet),
        "compiler_main_counts": _counts(main_packet),
        "compiler_modes": modes,
        "compiler_caps": {
            "records": len(caps_records()), **_counts(caps_packet)},
        "shipping": shipping,
    }


#: The filed expectation for the spike run (deterministic:
#: the full main packet + per-mode filings + cap counts +
#: the shipping path's measurements).
SPIKE_EXPECTED: dict[str, Any] = {
    "compiler_main_counts": {
        "identity_core": 4, "truth_anchor_points": 5,
        "active_traits": 3, "symbolic_context": 2,
        "uncertainty_points": 2, "source_items": 14,
        "tone_contract": 4, "response_guidance": 4,
        "relationship_excerpt": 4,
    },
    "compiler_traits": [
        {"trait_name": "steady", "weight": 1.0,
         "supporting_record_ids": [
             "rec:synthetic-wrenna-001", "rec:synthetic-wrenna-005"]},
        {"trait_name": "plain-spoken", "weight": 0.8,
         "supporting_record_ids": ["rec:synthetic-wrenna-003"]},
        {"trait_name": "honesty", "weight": 0.6,
         "supporting_record_ids": ["rec:synthetic-wrenna-008"]},
    ],
    "compiler_identity_core": [
        "persona:wrenna.temperament=steady",
        "persona:wrenna.name=Wrenna the Lantern-Keeper",
        "persona:wrenna.tone=plain-spoken",
        "persona:wrenna.duty=keep the ferry lantern lit",
    ],
    "compiler_truth_anchors": [
        "persona:wrenna.mood=steady",
        "world:lantern-hollow.ferry_minutes=11",
        "world:lantern-hollow.river=the Bracken",
        "persona:wrenna.value=honesty",
        "world:lantern-hollow.bell_count=two",
    ],
    "compiler_symbolic": [
        "symbol=lantern;runes=ansuz,raidho;charge=0.75",
        "symbol=river-stone;runes=laguz;charge=0.40",
    ],
    "compiler_uncertainty": [
        "an unverified tale of a third tide bell",
        "a disputed tale of a night crossing",
    ],
    "compiler_guidance": [
        "relationship_mode=builder",
        "relationship_weather=calm",
        "closeness_index=0.44",
        "never claim vows or memories that are not supported",
    ],
    "compiler_caps": {
        "records": 57, "identity_core": 10,
        "truth_anchor_points": 16, "active_traits": 8,
        "symbolic_context": 8, "uncertainty_points": 8,
        "source_items": 20, "tone_contract": 4,
        "response_guidance": 1, "relationship_excerpt": 1,
    },
    "shipping": {
        "record_id": "syn_wrenna_001",
        "steering_lines": [
            "Persona traits: steady; observant; plain-spoken",
            "Persona speech style: short, concrete sentences; "
            "names the river level before giving advice; no filler",
            "Persona behavior: answer what was asked; cite the "
            "synthetic reference when using it; say plainly when "
            "the fixture store is silent",
        ],
        "steering_entries": 9, "avoid_entries": 3,
        "avoid_in_steering": 0, "weights_carried": 5,
        "weights_in_steering": 0, "injected": True,
        "fact_block_starts_with_block": True,
        "turns_checked": 5, "turns_consistent": 2,
        "violations": 3, "score": 0.4, "verdict": "REVIEW",
        "steering_cap_entries": 6,
    },
}


# ── The filed probes ─────────────────────────────────────────────────────────

PERSONA_PROBES: tuple[dict[str, Any], ...] = (
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
    {"id": "spike_trait_weight_wrong_is_named",
     "kind": "trait_wrong",
     "detail": "an expected filing with the steady trait's weight "
               "perturbed (1.0 -> 0.9) is named against the "
               "measured traits"},
    {"id": "caps_count_wrong_is_named",
     "kind": "caps_wrong",
     "detail": "an expected filing with the caps run's identity "
               "count perturbed (10 -> 9) is named against the "
               "measured caps"},
    {"id": "schema_field_missing_from_spike_is_named",
     "kind": "schema_field_missing",
     "detail": "a field map with token_budget_hint dropped from "
               "PersonaPacket is named against the schemas"},
    {"id": "unknown_verdict_value_is_named",
     "kind": "verdict_unknown",
     "detail": "a verdict value outside the roadmap's three is named"},
)


def run_persona_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in PERSONA_PROBES:
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
            row = dict(COMPARISON_ROWS[0], shipping_path="")
            passed = bool(comparison_row_problems(row))
        elif kind == "trait_wrong":
            spike = run_persona_spike()
            measured = {
                t["trait_name"]: t["weight"]
                for t in spike["compiler_main"]["active_traits"]}
            passed = bool(measured_problems(
                measured, {"steady": 0.9}))
        elif kind == "caps_wrong":
            spike = run_persona_spike()
            passed = bool(measured_problems(
                spike["compiler_caps"], {"identity_core": 9}))
        elif kind == "schema_field_missing":
            field_map = spike_field_map()
            field_map["PersonaPacket"] = tuple(
                f for f in field_map["PersonaPacket"]
                if f != "token_budget_hint")
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

def render_persona_compiler_verdict_markdown() -> str:
    """Render the verdict deterministically — the
    exact text filed as
    docs/architecture/PERSONA_COMPILER_VERDICT.md."""
    spike = run_persona_spike()
    main_counts = spike["compiler_main_counts"]
    caps = spike["compiler_caps"]
    ship = spike["shipping"]
    lines: list[str] = []
    lines.append("# Persona Compiler Verdict — TF-141")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/cognition/persona_compiler_verdict.py` "
        "— the filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.cognition.persona_compiler_verdict`. "
        "The checker (`persona_compiler_verdict_problems()`) "
        "re-derives every claim live from the tree and fails on "
        "any drift.")
    lines.append("")
    lines.append(f"Format: `{PERSONA_COMPILER_VERDICT_FORMAT}`")
    lines.append(f"Verdict: **{VERDICT['verdict']}**")
    lines.append(f"Comparison rows: {len(COMPARISON_ROWS)}")
    lines.append(f"Probes: {len(PERSONA_PROBES)}/{len(PERSONA_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: the "
        "wing cannot import — `research_data/src/wyrdforge/models` "
        "does not exist and `persona_compiler.py` imports its "
        "types from that missing package (TF-006 / D-WING-1). The "
        "compiler turns memory records (+ an optional BondEdge) "
        "into a PersonaPacket (identity/anchor split by domain, "
        "confidence-weighted traits, symbolic lines, uncertainty "
        "points, mode additions, bond guidance, source items with "
        "confidences). The shipping path (TF-009 + TF-030) loads "
        "an authored PersonalityCoreRecord into a "
        "PersonaExpectation whose steering is injected into every "
        "prompt and whose avoid list is checked deterministically "
        "— and whose five weights steer nothing, a deferral "
        "TF-030 filed to this slice by name. Both paths run here "
        "over the same synthetic ground truth (TF-017 / TF-030's "
        "Wrenna fixtures; no canon).")
    lines.append("")
    lines.append("## 2. The comparison")
    lines.append("")
    lines.append("| Dimension | Compiler | Shipping path | Finding |")
    lines.append("|---|---|---|---|")
    for row in COMPARISON_ROWS:
        lines.append(
            f"| `{row['dimension']}` | {row['compiler']} | "
            f"{row['shipping_path']} | **{row['finding']}** |")
    lines.append("")
    lines.append("## 3. The verdict: MERGE")
    lines.append("")
    lines.append("### Reasons")
    lines.append("")
    for reason in VERDICT["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("### Conditions — what TF-143 must implement")
    lines.append("")
    for condition in VERDICT["conditions"]:
        lines.append(f"- {condition}")
    lines.append("")
    lines.append("## 4. The working spike + the measurement")
    lines.append("")
    lines.append(
        "Spike dataclasses rebuilt from the schemas inside the "
        "verdict module (the packet shapes in full from "
        "persona_packet.schema.json; CanonicalFactPayload / "
        "CanonicalFactContent, SymbolicTracePayload / "
        "SymbolicTraceContent and TruthMeta from the two record "
        "schemas), namespaced `Spike*` — NOT the missing "
        "`wyrdforge.models` package and NOT production. The input "
        "records are reduced to what compile() actually reads; "
        "the BondEdge input is TF-139's spike edge, reused. "
        "`SpikePersonaCompiler` re-implements compile() "
        "faithfully, including its elif approval blind spot and "
        "its relationship_excerpt duplication, so both are "
        "measured, not asserted.")
    lines.append("")
    lines.append(
        f"Main run (COMPANION + bond edge, 14 records): "
        f"identity_core {main_counts['identity_core']} / "
        f"truth_anchor_points {main_counts['truth_anchor_points']} "
        f"/ active_traits {main_counts['active_traits']} / "
        f"symbolic_context {main_counts['symbolic_context']} / "
        f"uncertainty_points {main_counts['uncertainty_points']} / "
        f"source_items {main_counts['source_items']} / "
        f"tone_contract {main_counts['tone_contract']} / "
        f"response_guidance {main_counts['response_guidance']}.")
    lines.append("")
    lines.append("Active traits (evidence-weighted):")
    lines.append("")
    for trait in spike["compiler_main"]["active_traits"]:
        lines.append(
            f"- {trait['trait_name']}: weight {trait['weight']} "
            f"from {', '.join(trait['supporting_record_ids'])}")
    lines.append("")
    lines.append(
        "The pending canonical fact (bell_count, confidence 0.3, "
        "approval pending) files as the truth anchor "
        "`world:lantern-hollow.bell_count=two` and NOT as an "
        "uncertainty point — the compiler's elif blind spot, "
        "measured. relationship_excerpt duplicates all "
        f"{main_counts['relationship_excerpt']} response_guidance "
        "lines verbatim (bond lines + the mode line).")
    lines.append("")
    lines.append(
        f"Caps run (57 generated records): identity_core "
        f"{caps['identity_core']} / truth_anchor_points "
        f"{caps['truth_anchor_points']} / active_traits "
        f"{caps['active_traits']} / symbolic_context "
        f"{caps['symbolic_context']} / uncertainty_points "
        f"{caps['uncertainty_points']} / source_items "
        f"{caps['source_items']} — every cap binds exactly; "
        "token_budget_hint (800) is carried and never enforced.")
    lines.append("")
    lines.append(
        f"Shipping path (TF-030 fixture expectation): "
        f"{len(ship['steering_lines'])} steering lines / "
        f"{ship['steering_entries']} entries delivered by "
        f"inject_persona (injected={ship['injected']}); avoid "
        f"{ship['avoid_entries']} entries, 0 in the steering "
        f"block; weights carried {ship['weights_carried']}, in "
        f"steering {ship['weights_in_steering']}; fixture turns "
        f"checked {ship['turns_checked']}, consistent "
        f"{ship['turns_consistent']}, violations "
        f"{ship['violations']}, score {ship['score']}, verdict "
        f"{ship['verdict']}; a 9-trait expectation files "
        f"{ship['steering_cap_entries']} steering entries "
        f"(MAX_STEERING_ENTRIES={MAX_STEERING_ENTRIES}).")
    lines.append("")
    lines.append("## 5. The probes")
    lines.append("")
    for probe in PERSONA_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-141 discharges the persona-compiler share of "
        "D-WING-1 (verdict decided on the same synthetic ground "
        "truth through both paths, narrowed in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 "
        "itself stays OPEN for the TF-142 verdict and the "
        "TF-143 rebuild / TF-144 retirements / TF-149 tests it "
        "assigns. The merge this verdict files is implemented "
        "by TF-143, not by TF-141; TF-147 consumes the merged "
        "output.")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Spike grade only: production merging is NOT done "
        "and NOT PROVEN here — TF-143 owns implementing the "
        "adopted facets under the conditions in §3; TF-149 "
        "owns the wing's tests.")
    lines.append(
        "- The compiler's two measured defects (the elif "
        "approval blind spot promoting pending canonical "
        "facts to truth anchors; relationship_excerpt "
        "duplicating response_guidance) are filed for TF-143 "
        "to NOT import — the verdict adopts facets, not bugs.")
    lines.append(
        "- Fixture grade only: the ground truth is synthetic "
        "(Wrenna / Lantern Hollow); no live model was read "
        "and live persona quality is NOT PROVEN (TF-173 "
        "class).")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_persona_compiler_verdict_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / PERSONA_COMPILER_VERDICT_DOC
    doc.write_text(
        render_persona_compiler_verdict_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.cognition.persona_compiler_verdict`."""
    doc = write_persona_compiler_verdict_doc()
    print(
        f"persona-compiler verdict regenerated: {VERDICT['verdict']} "
        f"-> {PERSONA_COMPILER_VERDICT_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def persona_compiler_verdict_problems() -> list[str]:
    """Re-derive TF-141's claims from the live
    tree; name any that stopped being true."""
    problems: list[str] = []

    # The verdict filing itself.
    problems.extend(verdict_data_problems(VERDICT))
    row_ids = [r["dimension"] for r in COMPARISON_ROWS]
    if len(row_ids) != len(set(row_ids)):
        problems.append("duplicate COMPARISON_ROWS dimensions")
    if len(COMPARISON_ROWS) != 12:
        problems.append(
            f"the comparison files {len(COMPARISON_ROWS)} rows, not 12")
    for row in COMPARISON_ROWS:
        problems.extend(comparison_row_problems(row))
    findings = [r["finding"] for r in COMPARISON_ROWS]
    if findings.count("compiler-wins") != 7 \
            or findings.count("shipping-wins") != 2 \
            or findings.count("different") != 3:
        problems.append(
            f"the findings are {findings.count('compiler-wins')} "
            f"compiler-wins / {findings.count('shipping-wins')} "
            f"shipping-wins / {findings.count('different')} "
            "different, not 7 / 2 / 3")

    # The wing as censused: models missing, the compiler
    # importing them, its filed shapes as filed.
    if (_WING / "models").exists():
        problems.append(
            "wing: wyrdforge.models now exists — TF-143's rebuild "
            "has landed and this verdict's premise must be "
            "re-ruled")
    compiler_text = _COMPILER.read_text(encoding="utf-8")
    for marker in (
            "from wyrdforge.models.persona import",
            "from wyrdforge.models.memory import",
            "from wyrdforge.models.bond import",
            "def compile",
            '{"identity", "style", "mission"}',
            '{"temperament", "tone", "value", "mood"}',
            "min(1.0, weight)",
            "identity_core[:10]", "truth_anchor_points[:16]",
            "uncertainty_points[:8]", "symbolic_context[:8]",
            "source_items[:20]", "response_guidance[:10]",
            "response_guidance[:6]",
            "token_budget_hint: int = 800",
            "PersonaMode.CODING_GUIDE", "PersonaMode.WORLD_SEER",
            "PersonaMode.COMPANION", "closeness_index"):
        if marker not in compiler_text:
            problems.append(
                f"wing: persona_compiler.py no longer files "
                f"{marker!r}")
    wing_test = (
        _REPO_ROOT / "research_data" / "tests"
        / "test_memory_and_persona.py")
    wing_test_text = wing_test.read_text(encoding="utf-8")
    if "test_persona_compiler_builds_packet" not in wing_test_text \
            or "packet.identity_core" not in wing_test_text:
        problems.append(
            "wing: the wing's persona test intent is no longer "
            "filed in test_memory_and_persona.py")

    # The schemas parse; the spike's field sets + enums match.
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
    problems.extend(schema_field_problems(spike_field_map()))
    packet_schema = json.loads(
        (_SCHEMAS / "persona_packet.schema.json").read_text(
            encoding="utf-8"))
    if [m.value for m in PersonaMode] != \
            packet_schema["$defs"]["PersonaMode"]["enum"]:
        problems.append("spike: PersonaMode drifted from the schema")
    canonical_schema = json.loads(
        (_SCHEMAS / "canonical_fact_record.schema.json").read_text(
            encoding="utf-8"))
    symbolic_schema = json.loads(
        (_SCHEMAS / "symbolic_trace_record.schema.json").read_text(
            encoding="utf-8"))
    if [a.value for a in ApprovalState] != \
            canonical_schema["$defs"]["ApprovalState"]["enum"]:
        problems.append("spike: ApprovalState drifted from the schema")
    if canonical_schema["$defs"]["TruthMeta"] != \
            symbolic_schema["$defs"]["TruthMeta"]:
        problems.append(
            "schemas: the two record schemas' TruthMeta $defs "
            "differ — the spike's shared TruthMeta must be re-ruled")

    # The shipping path as filed by TF-009 / TF-030.
    persona_text = (
        _REPO_ROOT / "src" / "thoughtforge" / "cognition"
        / "persona.py").read_text(encoding="utf-8")
    for marker in ("MAX_STEERING_ENTRIES = 6", "def inject_persona",
                   "def check_persona_response",
                   "def score_persona_turns", "TF-141"):
        if marker not in persona_text:
            problems.append(
                f"shipping: cognition/persona.py no longer files "
                f"{marker!r}")
    if not (_REPO_ROOT / "docs" / "audit"
            / "TF-009_CONFIG_CENSUS.md").exists():
        problems.append("shipping: TF-009's config census is missing")
    if FIXTURE_EXPECTATION.record_id != "syn_wrenna_001":
        problems.append(
            "shipping: TF-030's fixture expectation is no longer "
            "the synthetic Wrenna core")

    # The spike run reproduces its filed expectation.
    spike = run_persona_spike()
    problems.extend(measured_problems(
        spike["compiler_main_counts"],
        SPIKE_EXPECTED["compiler_main_counts"]))
    problems.extend(measured_problems(
        spike["compiler_caps"], SPIKE_EXPECTED["compiler_caps"]))
    problems.extend(measured_problems(
        spike["shipping"], SPIKE_EXPECTED["shipping"]))
    if spike["compiler_main"]["active_traits"] != \
            SPIKE_EXPECTED["compiler_traits"]:
        problems.append("spike: the main traits drifted from filing")
    if spike["compiler_main"]["identity_core"] != \
            SPIKE_EXPECTED["compiler_identity_core"]:
        problems.append("spike: identity_core drifted from filing")
    if spike["compiler_main"]["truth_anchor_points"] != \
            SPIKE_EXPECTED["compiler_truth_anchors"]:
        problems.append("spike: truth anchors drifted from filing")
    if spike["compiler_main"]["symbolic_context"] != \
            SPIKE_EXPECTED["compiler_symbolic"]:
        problems.append("spike: symbolic lines drifted from filing")
    if spike["compiler_main"]["uncertainty_points"] != \
            SPIKE_EXPECTED["compiler_uncertainty"]:
        problems.append("spike: uncertainty drifted from filing")
    if spike["compiler_main"]["response_guidance"] != \
            SPIKE_EXPECTED["compiler_guidance"]:
        problems.append("spike: guidance drifted from filing")
    if spike["compiler_main"]["relationship_excerpt"] != \
            spike["compiler_main"]["response_guidance"]:
        problems.append(
            "spike: relationship_excerpt no longer duplicates "
            "response_guidance — the filed defect must be re-ruled")
    pending_anchor = "world:lantern-hollow.bell_count=two"
    if pending_anchor not in spike["compiler_main"][
            "truth_anchor_points"]:
        problems.append(
            "spike: the pending canonical fact no longer files "
            "as a truth anchor — the elif blind spot must be "
            "re-ruled")

    # The probes all behave.
    probes = run_persona_probes()
    if len(probes) != len(PERSONA_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the persona probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / PERSONA_COMPILER_VERDICT_DOC
    if not doc.exists():
        problems.append(
            f"the write-up {PERSONA_COMPILER_VERDICT_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != \
            render_persona_compiler_verdict_markdown():
        problems.append(
            f"the filed doc {PERSONA_COMPILER_VERDICT_DOC} is not "
            "the render verbatim (regenerate with python3 -m "
            "thoughtforge.cognition.persona_compiler_verdict)")
    return problems


def persona_compiler_verdict_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    spike = run_persona_spike()
    probes = run_persona_probes()
    return {
        "format": PERSONA_COMPILER_VERDICT_FORMAT,
        "doc": PERSONA_COMPILER_VERDICT_DOC,
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
    "COMPARISON_ROWS",
    "FINDINGS",
    "PERSONA_COMPILER_VERDICT_DOC",
    "PERSONA_COMPILER_VERDICT_FORMAT",
    "PERSONA_PROBES",
    "SPIKE_EXPECTED",
    "SPIKE_SHAPES",
    "VERDICT",
    "VERDICTS",
    "ApprovalState",
    "PersonaMode",
    "SpikeCanonicalFactContent",
    "SpikeCanonicalFactPayload",
    "SpikeCanonicalFactRecord",
    "SpikeGenericContent",
    "SpikeMemoryRecord",
    "SpikePersonaCompiler",
    "SpikePersonaPacket",
    "SpikePersonaSourceItem",
    "SpikeSymbolicTraceContent",
    "SpikeSymbolicTracePayload",
    "SpikeSymbolicTraceRecord",
    "SpikeTraitSignal",
    "SpikeTruthMeta",
    "caps_records",
    "comparison_row_problems",
    "fixture_bond_edge",
    "fixture_records",
    "main",
    "measured_problems",
    "persona_compiler_verdict_as_data",
    "persona_compiler_verdict_problems",
    "render_persona_compiler_verdict_markdown",
    "run_persona_probes",
    "run_persona_spike",
    "schema_field_problems",
    "spike_field_map",
    "verdict_data_problems",
    "write_persona_compiler_verdict_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
