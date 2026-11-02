"""
TF-140 — the micro-RAG verdict, stated as verifiable data.

The roadmap asks for a verdict on wyrdforge's micro-RAG
against Part III's shipping retrieval engine — *adopt,
merge, or retire* — decided BY MEASUREMENT on TF-043's
fixtures: the better mechanism wins. The measured start,
probed live this tick on this tree before anything was
built:

  * **The wing cannot import** (TF-006 / D-WING-1, still
    true live): ``research_data/src/wyrdforge/models`` does
    not exist, and ``micro_rag_pipeline.py`` imports
    MicroContextPacket / QueryMode / RankedCandidate /
    RetrievalItem / TruthPacket from that missing package.
  * **The pipeline is a reranker + assembler, not a
    retriever.** ``MicroRAGPipeline`` has exactly two
    operations: ``score()`` — a nine-term weighted sum over
    a SUBSTRING similarity (the fraction of query terms
    appearing anywhere in the item's text + lexical terms;
    no tf-idf, no bm25, recency filed as the constant 0.5)
    — and ``assemble()`` — score each pre-gathered family,
    take its top 5, admit items in MODE_TARGETS family
    order while the token budget (default 900) lasts, and
    file them into the packet's typed sections. It has NO
    corpus, NO index, NO chunking and NO candidate
    gathering of its own (MicroRAGPipelineSpec.md proposes
    seven index families; no code implements them), and NO
    score threshold: a family always fills its cap if it
    has the items, however weak the match.
  * **The shipping engine** (``KnowledgeForge``, measured
    by TF-043) searches its SQLite corpus itself — FTS5
    bm25 over entities + reference chunks, a ConceptNet
    edge pass, a GeoNames pass — fuses each record through
    ``knowledge/scoring.py``'s named weights, and applies
    the MIN_MEMORY_SCORE 0.45 threshold at the defaults.
  * **TF-006's census stands**: micro-RAG is OVERLAP in
    purpose, different shape (packet-based,
    small-model-first).

What this module files:

  * ``COMPARISON_ROWS`` — the comparison as literal data:
    nine dimensions, each filing the micro-RAG shape, the
    shipping engine's shape, and a finding.
  * ``VERDICT`` — **RETIRE** (as a retrieval mechanism),
    with filed reasons and filed conditions (see the
    constant: the shipping engine wins every aggregate
    the fixtures measure except MRR, which ties; the
    retirement itself is TF-144's work, its lessons —
    typed packet sections, token budgeting, contradiction
    injection — harvested there, per TF-006's rule that a
    retire verdict must still harvest).
  * **The working spike** — minimal spike dataclasses
    rebuilt from the two schemas (RetrievalItem from
    retrieval_item.schema.json; RankedCandidate /
    TruthPacket / MicroContextPacket + the QueryMode enum
    from micro_context_packet.schema.json), namespaced
    ``Spike*`` — explicitly NOT the missing
    ``wyrdforge.models`` package and NOT production —
    and ``SpikeMicroRAG``, the pipeline's ``score()`` +
    ``assemble()`` re-implemented faithfully (exact
    weights, exact MODE_TARGETS, per-family cap 5,
    budget 900, section mapping).
  * ``run_micro_spike()`` — BOTH mechanisms over TF-043's
    fixture corpus, verbatim: the shipping engine through
    TF-043's own ``evaluate_retrieval`` on a freshly built
    fixture DB (defaults + the open setting), the spike
    over the same corpus adapted to RetrievalItems under
    the filed adaptation below, scored by TF-043's own
    ``score_query``. Every filed metric is reproduced by
    this function and pinned in tests.
  * ``micro_rag_verdict_problems()`` — the checker:
    re-derives every claim above from the live tree (the
    wing's missing models, the pipeline's weights and
    mode targets parsed live, the schemas, the spike's
    field sets against the schemas, both mechanisms'
    measured aggregates against their filings, the filed
    doc as this module's render verbatim) and names any
    that stops being true.
  * ``MICRO_PROBES`` — seven filed probes against the
    pure validators with perturbed copies (real filings
    never perturbed).

The filed fixture adaptation (spike assumptions, stated
in the open — every value is a schema or wing default,
never tuned): each TF-043 corpus row becomes one
RetrievalItem whose ``item_id`` IS the bundle id TF-043's
relevant sets name (entity qid, chunk id as a string,
ConceptNet concept1, GeoNames name); text is the text the
shipping passes search (entity label + description,
chunk title + content, the ConceptNet triple as the
forge's content string, GeoNames name + its shipping
description string); ``confidence`` is the entity's
filed popularity for entities and the schema default
0.5 for every other row (the fixture files no other
confidence); ``lexical_terms`` is empty and ``facets``
empty (the wing's own test items file neither);
``token_cost`` is the schema default 80; ALL rows enter
as the ``canonical`` family — the fixture corpus is
canonical world facts, candidate gathering is the
pipeline's missing upstream step (the spec's gather
step has no code), and ``factual_lookup`` is the mode
TF-043's queries are. Filler rows are included verbatim
(60 entity + 30 chunk fillers; 106 items in total).

Filed limits, stated plainly:

  * Spike grade only: the spike proves the pipeline's
    filed arithmetic runs deterministically and loses on
    TF-043's fixtures. It is NOT production, NOT an
    adoption, and changes NO shipping behaviour.
  * The verdict is a verdict on the RETRIEVAL MECHANISM
    (scoring + assembly as a retriever). The packet's
    typed sections are a presentation shape, unmeasured
    by these fixtures; they are filed as a harvested
    lesson for TF-144's closing doc, not as a win.
  * Fixture grade only: TF-043's corpus is synthetic
    (Lantern Hollow, TF-017). No live model was read;
    live retrieval quality is NOT PROVEN (TF-173 class).
  * One coupling, fixed in the open by this slice:
    TF-139's checker and tests grep the shipping
    knowledge + cognition sources for ``bond`` — this
    verdict module legitimately files the wing's
    bond_fit / bond_excerpt vocabulary, so TF-139's
    verdict-module exclusion (its own module, by name)
    is extended to this module by name in both places;
    the premise it guards (the shipping STORE has no
    bond graph) is unchanged.

The human-readable write-up is
``docs/architecture/MICRO_RAG_VERDICT.md``, filed
verbatim as this module's render (regenerate with
``python3 -m thoughtforge.knowledge.micro_rag_verdict``).
Proof grade: FIXTURE — both mechanisms re-measured live
on this VM over TF-043's synthetic fixtures; no live
model, never claimed.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from thoughtforge.knowledge import retrieval_precision as _rp
from thoughtforge.knowledge.retrieval_precision import (
    BASELINE,
    FIXTURE_CONCEPTNET_EDGES,
    FIXTURE_ENTITIES,
    FIXTURE_GEONAMES,
    FIXTURE_QUERIES,
    FIXTURE_REFERENCE_CHUNKS,
    build_fixture_db,
    evaluate_retrieval,
    score_query,
)

MICRO_RAG_VERDICT_FORMAT = "thoughtforge-micro-rag-verdict/1"
MICRO_RAG_VERDICT_DOC = "docs/architecture/MICRO_RAG_VERDICT.md"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_PIPELINE = _WING / "services" / "micro_rag_pipeline.py"
_SCHEMAS = _WING / "schemas"

#: The three verdicts the roadmap allows (TF-139's vocabulary).
VERDICTS: tuple[str, ...] = ("ADOPT_AS_LAYER", "MERGE", "RETIRE")

#: The findings a comparison row may file.
FINDINGS: tuple[str, ...] = (
    "shipping-wins", "micro-wins", "different", "overlap")


# ── The comparison, as literal data ──────────────────────────────────────────

COMPARISON_ROWS: tuple[dict[str, Any], ...] = (
    {"dimension": "candidate_gathering_and_indexing",
     "micro_rag": "none in code: no corpus, no index, no chunking — "
                  "assemble() receives candidates_by_family "
                  "pre-gathered by an unnamed caller, items arriving "
                  "whole at whatever scale the caller filed them; "
                  "the spec's gather step and its seven index "
                  "families (atom / episode / canonical / bond / "
                  "symbolic / code / graph) are prose only",
     "shipping_engine": "the forge gathers itself over its own "
                        "corpus, which is the index: entities, "
                        "pre-chunked reference_chunks, conceptnet "
                        "edges and geonames places in FTS5-indexed "
                        "SQLite tables, searched by bm25 plus a "
                        "ConceptNet pass and a GeoNames pass",
     "finding": "different"},
    {"dimension": "ranking_signal",
     "micro_rag": "a nine-term weighted sum (filed in SCORE_WEIGHTS) "
                  "over a substring similarity — the fraction of "
                  "query terms appearing anywhere in text + lexical "
                  "terms; term frequency and rarity count for "
                  "nothing, and recency is the constant 0.5 for "
                  "every item",
     "shipping_engine": "bm25 term statistics from FTS5 fused per "
                        "source through scoring.py's named weights "
                        "(fts proxy, source recency proxies, "
                        "popularity / weight as importance), so rare "
                        "and repeated terms both move the score",
     "finding": "different"},
    {"dimension": "score_threshold_and_negative_queries",
     "micro_rag": "no score floor exists in the pipeline: assemble() "
                  "admits each family's top 5 however weak the match "
                  "— on TF-043's negative query (Q6, similarity 0 for "
                  "every item) it still returns the five "
                  "highest-confidence entities, scoring precision 0 / "
                  "recall 0 under TF-043's convention",
     "shipping_engine": "MIN_MEMORY_SCORE 0.45 at the defaults: the "
                        "negative query returns nothing and scores "
                        "precision 1 / recall 1 — correctly returning "
                        "nothing is a success (TF-043's convention)",
     "finding": "shipping-wins"},
    {"dimension": "measured_precision_and_f1",
     "micro_rag": "mean precision 0.3333 / mean F1 0.4749 on TF-043's "
                  "fixtures (run_micro_spike, filed below): the "
                  "per-family cap fills every packet with exactly "
                  "five items — 45 returned across the nine queries, "
                  "only 15 of them known answers",
     "shipping_engine": "mean precision 0.8889 / mean F1 0.9111 at "
                        "the defaults (TF-043's pinned baseline, "
                        "re-measured live by this module), returning "
                        "17 records across the nine queries, 14 "
                        "of them known answers; even the open "
                        "setting (min_score 0) files 0.8148 / "
                        "0.8815",
     "finding": "shipping-wins"},
    {"dimension": "measured_recall",
     "micro_rag": "mean recall 0.8889: every answered query's known "
                  "answers are found (8 perfect-recall queries), but "
                  "the negative query's junk return scores recall 0 "
                  "and drags the mean below the shipping engine's",
     "shipping_engine": "mean recall 0.9630 at the defaults: eight "
                        "queries at recall 1.0 and the geonames "
                        "query at 0.6667 (the filed threshold defect, "
                        "next row); the open setting files recall 1.0",
     "finding": "shipping-wins"},
    {"dimension": "place_records_geonames",
     "micro_rag": "the geonames query (Q9) files recall 1.0 / "
                  "precision 0.6: with no threshold, the GeoNames "
                  "record enters the packet on its substring "
                  "similarity like any other item",
     "shipping_engine": "Q9 files recall 0.6667 at the defaults: a "
                        "GeoNames record's composite is the fixed "
                        "0.285 (TF-043's filed finding), which can "
                        "never pass the 0.45 threshold — a defect of "
                        "the shipping threshold, already filed by "
                        "TF-043 and owned by the scoring line, not "
                        "evidence the micro mechanism ranks better",
     "finding": "micro-wins"},
    {"dimension": "packet_assembly",
     "micro_rag": "a typed MicroContextPacket: mode + goal + a "
                  "TruthPacket header (must_be_true / open_unknowns "
                  "/ forbidden_assumptions) + six typed sections "
                  "(canonical / recent / bond / symbolic / code / "
                  "contradiction), assembled in MODE_TARGETS family "
                  "order — a presentation shape the shipping bundle "
                  "does not have, unmeasured by these fixtures",
     "shipping_engine": "a flat MemoryActivationBundle: the ranked "
                        "activated records with their scores and "
                        "citations, no mode, no truth header, no "
                        "typed sections",
     "finding": "different"},
    {"dimension": "token_budgeting",
     "micro_rag": "an explicit packet budget (default 900) charged "
                  "per item's filed token_cost (schema default 80): "
                  "on the fixtures every packet uses exactly 400 "
                  "(5 x 80) — the budget never binds before the "
                  "per-family cap does, and nothing trims a weak "
                  "item",
     "shipping_engine": "no token budget: a global top_k 10 with "
                        "per-source sub-caps (reference top_k//2, "
                        "conceptnet top_k//3, geonames top_k//5 in "
                        "sql_retrieve) plus the score threshold "
                        "doing the trimming — 17 records returned "
                        "across the nine fixture queries",
     "finding": "different"},
    {"dimension": "citation_handling",
     "micro_rag": "each item carries item_id + a free-text source_ref "
                  "string (e.g. 'bond_graph' in the wing's test); no "
                  "citation is built, checked or attached to the "
                  "packet beyond those two fields",
     "shipping_engine": "each record carries its record_id and the "
                        "forge's citation payload (the ConceptNet "
                        "triple fields ride along, TF-048), under "
                        "the estate's citation discipline — both "
                        "sides carry provenance ids, only the "
                        "shipping side builds citations from them",
     "finding": "overlap"},
)


# ── The verdict ──────────────────────────────────────────────────────────────

VERDICT: dict[str, Any] = {
    "verdict": "RETIRE",
    "subject": "wyrdforge micro-RAG (MicroRAGPipeline + "
               "micro_context_packet / retrieval_item schemas) as a "
               "retrieval mechanism vs Part III's shipping engine "
               "(KnowledgeForge, TF-043)",
    "decided_on": "measurement on TF-043's fixtures, both mechanisms "
                  "run over the same corpus by run_micro_spike() and "
                  "filed in this module + TF-006's census (OVERLAP in "
                  "purpose, different shape)",
    "reasons": (
        "measured loss on every aggregate but one: on TF-043's "
        "fixtures the shipping engine at its defaults files "
        "precision 0.8889 / recall 0.9630 / F1 0.9111 against the "
        "micro-RAG spike's 0.3333 / 0.8889 / 0.4749, and mean "
        "reciprocal rank ties at 1.0 — the micro mechanism wins no "
        "aggregate, and even the shipping engine's open setting "
        "(threshold removed: 0.8148 / 1.0 / 0.8815) beats it on all "
        "three, so the loss is in its ranking and filling, not in "
        "the threshold alone",
        "the mechanism's defects are structural, not tuning: no "
        "score floor (the negative query returns five junk items "
        "and scores 0/0 where the shipping engine returns nothing "
        "and scores 1/1), substring similarity with no term "
        "statistics (frequency and rarity count for nothing), "
        "recency filed as a constant, and a per-family cap that "
        "always fills — 45 items returned across nine queries "
        "against the shipping engine's 17",
        "not adopt: TF-006 files micro-RAG OVERLAP in purpose — "
        "the estate already has a retriever, and the measured one "
        "is the shipping engine; adopting a second, worse "
        "retriever beside it buys a competing ranking law, not a "
        "layer (unlike TF-139's bond graph, there is no unique "
        "retrieval concept here to house)",
        "not a merge: folding the micro scoring sum into the "
        "shipping fusion would replace bm25 term statistics with "
        "substring matching and repeal the threshold — it would "
        "regress TF-043's pinned baseline by construction; the "
        "two ranking laws cannot be averaged into a better one "
        "on this evidence",
        "its one measured win is not its merit: Q9's recall "
        "(1.0 vs 0.6667) comes from having no threshold at all — "
        "the same absence that fails Q6 — while the shipping "
        "side's miss is TF-043's already-filed GeoNames "
        "threshold defect, owned by the shipping scoring line; "
        "fixing that defect does not need this mechanism",
    ),
    "conditions": (
        "RETIRE the micro-RAG retrieval mechanism: the shipping "
        "KnowledgeForge remains the estate's sole retriever; no "
        "micro-RAG scoring or assembly code is adopted or merged "
        "into the shipping tree",
        "retirement is NOT executed in TF-140: TF-144 owns "
        "executing each RETIRE verdict (code removed or "
        "archived-in-place, lessons written into the wing's "
        "closing doc); TF-143 owns adoptions — TF-140 files no "
        "adoption for it to land; TF-149 owns the wing's tests",
        "the harvest, filed for TF-144's closing doc per TF-006 "
        "(a retire verdict must still harvest): the typed packet "
        "sections, the explicit token budget and contradiction "
        "injection are presentation lessons recorded here — they "
        "are NOT adopted by this verdict, and any future packet "
        "presentation over shipping bundles needs its own "
        "roadmap word",
        "TF-140 changes NO shipping behaviour and NO "
        "research-wing code: the spike lives in this verdict "
        "module only, namespaced Spike*, and is not the missing "
        "wyrdforge.models package",
        "the fixture adaptation is filed in this module's "
        "docstring (schema defaults throughout, all rows as the "
        "canonical family, factual_lookup mode): the verdict "
        "stands on those filed terms, and a different candidate "
        "gathering — which the wing never implemented — is a "
        "different mechanism than the one measured",
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
    for side in ("dimension", "micro_rag", "shipping_engine"):
        if not row.get(side):
            problems.append(f"row {name}: the {side} side is missing")
    if row.get("finding") not in FINDINGS:
        problems.append(
            f"row {name}: finding {row.get('finding')!r} is not one "
            f"of {FINDINGS}")
    return problems


#: The pipeline's rerank weights, exactly as filed in
#: micro_rag_pipeline.py (the checker re-reads them live).
SCORE_WEIGHTS: dict[str, float] = {
    "similarity": 0.24, "task_relevance": 0.24,
    "support_quality": 0.14, "scope_match": 0.12, "recency": 0.08,
    "contradiction_penalty": -0.10, "token_cost_penalty": -0.04,
    "novelty_bonus": 0.04, "bond_fit": 0.08,
}


def score_weights_problems(mapping: Any) -> list[str]:
    """Judge one score-weight map against the
    filed pipeline weights, purely. Empty-safe."""
    if not isinstance(mapping, dict) or not mapping:
        return ["score weights: the map is empty"]
    problems: list[str] = []
    for name, weight in SCORE_WEIGHTS.items():
        if mapping.get(name) != weight:
            problems.append(
                f"score weights: {name} is {mapping.get(name)!r}, "
                f"not the filed {weight}")
    for name in mapping:
        if name not in SCORE_WEIGHTS:
            problems.append(f"score weights: unknown term {name!r}")
    return problems


def measured_problems(measured: Any, expected: Any) -> list[str]:
    """Judge measured aggregate metrics against
    an expected filing, purely. Empty-safe."""
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
    two schemas parsed live from the wing.
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
    """The field sets of the two schemas (top
    level + $defs sub-shapes), parsed live."""
    packet = json.loads(
        (_SCHEMAS / "micro_context_packet.schema.json").read_text(
            encoding="utf-8"))
    item = json.loads(
        (_SCHEMAS / "retrieval_item.schema.json").read_text(
            encoding="utf-8"))
    field_map: dict[str, tuple[str, ...]] = {
        "MicroContextPacket": tuple(packet["properties"]),
        "RetrievalItem": tuple(item["properties"]),
    }
    for name, shape in packet["$defs"].items():
        if "properties" in shape:
            field_map[name] = tuple(shape["properties"])
    return field_map


# ── The spike: dataclasses rebuilt from the two schemas ──────────────────────
#
# Namespaced Spike* — NOT wyrdforge.models, NOT production.
# Field sets match the schemas (required + defaults as filed);
# array / object fields take default_factory (a spike needs
# usable empties).

class QueryMode(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    COMPANION_CONTINUITY = "companion_continuity"
    WORLD_STATE = "world_state"
    SYMBOLIC_INTERPRETATION = "symbolic_interpretation"
    CODING_TASK = "coding_task"
    REPAIR_OR_BOUNDARY = "repair_or_boundary"
    CREATIVE_GENERATION = "creative_generation"


@dataclass
class SpikeRetrievalItem:
    item_id: str
    item_type: str
    text: str
    source_ref: str
    support_class: str = "supported"
    confidence: float = 0.5
    lexical_terms: list[str] = field(default_factory=list)
    facets: dict[str, list[str]] = field(default_factory=dict)
    token_cost: int = 80


@dataclass
class SpikeRankedCandidate(SpikeRetrievalItem):
    final_score: float = 0.0
    similarity: float = 0.0
    task_relevance: float = 0.0
    support_quality: float = 0.0
    scope_match: float = 0.0
    recency: float = 0.0
    contradiction_penalty: float = 0.0
    token_cost_penalty: float = 0.0
    novelty_bonus: float = 0.0
    bond_fit: float = 0.0


@dataclass
class SpikeTruthPacket:
    must_be_true: list[str] = field(default_factory=list)
    open_unknowns: list[str] = field(default_factory=list)
    forbidden_assumptions: list[str] = field(default_factory=list)


@dataclass
class SpikeMicroContextPacket:
    mode: QueryMode
    goal: str
    truth_packet: SpikeTruthPacket = field(
        default_factory=SpikeTruthPacket)
    canonical_facts: list[SpikeRankedCandidate] = field(
        default_factory=list)
    recent_events: list[SpikeRankedCandidate] = field(
        default_factory=list)
    bond_excerpt: list[SpikeRankedCandidate] = field(
        default_factory=list)
    symbolic_context: list[SpikeRankedCandidate] = field(
        default_factory=list)
    code_context: list[SpikeRankedCandidate] = field(
        default_factory=list)
    contradiction_items: list[SpikeRankedCandidate] = field(
        default_factory=list)
    packet_budget_used: int = 0


#: Spike shape name -> the spike dataclass, for the
#: schema-coverage checker.
SPIKE_SHAPES: dict[str, Any] = {
    "RetrievalItem": SpikeRetrievalItem,
    "RankedCandidate": SpikeRankedCandidate,
    "TruthPacket": SpikeTruthPacket,
    "MicroContextPacket": SpikeMicroContextPacket,
}


def spike_field_map() -> dict[str, tuple[str, ...]]:
    """The spike dataclasses' field sets, by shape."""
    return {
        name: tuple(f.name for f in dataclasses.fields(cls))
        for name, cls in SPIKE_SHAPES.items()
    }


# ── The spike pipeline: the wing's operations, faithfully ────────────────────

class SpikeMicroRAG:
    """MicroRAGPipeline's score() + assemble() over
    the spike types: same mode targets, same nine-term
    weights, same per-family cap, same budget law, same
    section mapping. Spike only — not production."""

    MODE_TARGETS: dict[QueryMode, tuple[str, ...]] = {
        QueryMode.FACTUAL_LOOKUP: ("canonical", "recent", "contradiction"),
        QueryMode.COMPANION_CONTINUITY: (
            "bond", "canonical", "recent", "symbolic"),
        QueryMode.WORLD_STATE: (
            "canonical", "recent", "symbolic", "contradiction"),
        QueryMode.SYMBOLIC_INTERPRETATION: (
            "symbolic", "canonical", "recent"),
        QueryMode.CODING_TASK: (
            "code", "canonical", "recent", "contradiction"),
        QueryMode.REPAIR_OR_BOUNDARY: (
            "bond", "canonical", "contradiction"),
        QueryMode.CREATIVE_GENERATION: (
            "canonical", "symbolic", "bond", "recent"),
    }

    def score(
        self, query: str, item: SpikeRetrievalItem, mode: QueryMode,
    ) -> SpikeRankedCandidate:
        q_terms = [term.lower() for term in query.split() if term.strip()]
        text = (item.text + " " + " ".join(item.lexical_terms)).lower()
        similarity = min(
            1.0,
            sum(1 for t in q_terms if t in text) / max(1, len(q_terms)))
        facet_values = {
            v.lower() for values in item.facets.values() for v in values}
        task_relevance = 1.0 if mode.value in facet_values or \
            item.item_type.startswith(mode.value.split("_")[0]) else 0.5
        support_quality = item.confidence
        scope_match = 1.0 if "global" in facet_values \
            or not facet_values else 0.7
        recency = 0.5
        contradiction_penalty = 0.6 \
            if item.item_type == "contradiction" else 0.0
        token_cost_penalty = min(1.0, item.token_cost / 400.0)
        lexical_counter = Counter(item.lexical_terms)
        novelty_bonus = 0.2 if lexical_counter and \
            lexical_counter.most_common(1)[0][1] == 1 else 0.0
        bond_fit = 0.8 if mode is QueryMode.COMPANION_CONTINUITY and \
            item.item_type in {"bond", "vow", "hurt"} else 0.0
        final_score = (
            similarity * SCORE_WEIGHTS["similarity"]
            + task_relevance * SCORE_WEIGHTS["task_relevance"]
            + support_quality * SCORE_WEIGHTS["support_quality"]
            + scope_match * SCORE_WEIGHTS["scope_match"]
            + recency * SCORE_WEIGHTS["recency"]
            + contradiction_penalty * SCORE_WEIGHTS["contradiction_penalty"]
            + token_cost_penalty * SCORE_WEIGHTS["token_cost_penalty"]
            + novelty_bonus * SCORE_WEIGHTS["novelty_bonus"]
            + bond_fit * SCORE_WEIGHTS["bond_fit"]
        )
        return SpikeRankedCandidate(
            item_id=item.item_id, item_type=item.item_type,
            text=item.text, source_ref=item.source_ref,
            support_class=item.support_class, confidence=item.confidence,
            lexical_terms=list(item.lexical_terms),
            facets=dict(item.facets), token_cost=item.token_cost,
            final_score=round(final_score, 4), similarity=similarity,
            task_relevance=task_relevance,
            support_quality=support_quality, scope_match=scope_match,
            recency=recency, contradiction_penalty=contradiction_penalty,
            token_cost_penalty=token_cost_penalty,
            novelty_bonus=novelty_bonus, bond_fit=bond_fit)

    def assemble(
        self,
        *,
        query: str,
        mode: QueryMode,
        candidates_by_family: dict[str, list[SpikeRetrievalItem]],
        truth_packet: SpikeTruthPacket | None = None,
        packet_budget: int = 900,
    ) -> SpikeMicroContextPacket:
        ranked_by_family: dict[str, list[SpikeRankedCandidate]] = {}
        for family, items in candidates_by_family.items():
            ranked = [self.score(query, item, mode) for item in items]
            ranked.sort(key=lambda item: item.final_score, reverse=True)
            ranked_by_family[family] = ranked

        packet = SpikeMicroContextPacket(
            mode=mode, goal=query,
            truth_packet=truth_packet or SpikeTruthPacket())
        budget = 0
        for family in self.MODE_TARGETS[mode]:
            for item in ranked_by_family.get(family, [])[:5]:
                if budget + item.token_cost > packet_budget:
                    continue
                budget += item.token_cost
                if family == "canonical":
                    packet.canonical_facts.append(item)
                elif family == "recent":
                    packet.recent_events.append(item)
                elif family == "bond":
                    packet.bond_excerpt.append(item)
                elif family == "symbolic":
                    packet.symbolic_context.append(item)
                elif family == "code":
                    packet.code_context.append(item)
                elif family == "contradiction":
                    packet.contradiction_items.append(item)
        packet.packet_budget_used = budget
        return packet


# ── TF-043's corpus as spike items (the filed adaptation) ────────────────────

def fixture_items() -> list[SpikeRetrievalItem]:
    """TF-043's fixture corpus, verbatim, as spike
    RetrievalItems under the adaptation filed in
    the module docstring: 106 items (68 entities
    incl. filler, 34 reference chunks incl. filler,
    2 ConceptNet edges, 2 GeoNames places)."""
    items: list[SpikeRetrievalItem] = []
    for qid, label, description, popularity in FIXTURE_ENTITIES:
        items.append(SpikeRetrievalItem(
            item_id=qid, item_type="entity",
            text=f"{label} {description}", source_ref="wikidata",
            confidence=popularity))
    for i in range(_rp.ENTITY_FILLER_COUNT):
        items.append(SpikeRetrievalItem(
            item_id=f"QF44{i:02d}", item_type="entity",
            text=f"filler {i} {_rp.ENTITY_FILLER_TEXT}",
            source_ref="wikidata", confidence=0.1))
    for chunk_id, title, content in FIXTURE_REFERENCE_CHUNKS:
        items.append(SpikeRetrievalItem(
            item_id=str(chunk_id), item_type="reference_chunk",
            text=f"{title} {content}", source_ref="reference"))
    for i in range(_rp.REFERENCE_FILLER_COUNT):
        items.append(SpikeRetrievalItem(
            item_id=str(4400 + i), item_type="reference_chunk",
            text=f"filler chunk {i} {_rp.REFERENCE_FILLER_TEXT}",
            source_ref="reference"))
    for concept1, relation, concept2, _weight in FIXTURE_CONCEPTNET_EDGES:
        items.append(SpikeRetrievalItem(
            item_id=concept1, item_type="concept_edge",
            text=f"{concept1} {relation} {concept2}",
            source_ref="conceptnet"))
    for _geonames_id, name, population in FIXTURE_GEONAMES:
        items.append(SpikeRetrievalItem(
            item_id=name, item_type="place",
            text=f"{name} P in SY, pop {population}",
            source_ref="geonames"))
    return items


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """TF-043's aggregate rules (score_query's
    conventions; the negative query excluded from
    the MRR mean), plus the total returned count."""
    answered = [q for q in per_query if q["kind"] != "negative"]
    return {
        "queries": len(per_query),
        "mean_precision": round(
            sum(q["precision"] for q in per_query) / len(per_query), 4),
        "mean_recall": round(
            sum(q["recall"] for q in per_query) / len(per_query), 4),
        "mean_f1": round(
            sum(q["f1"] for q in per_query) / len(per_query), 4),
        "mean_reciprocal_rank": round(
            sum(q["reciprocal_rank"] for q in answered) / len(answered), 4),
        "perfect_recall_queries": sum(
            1 for q in per_query if q["recall"] == 1.0),
        "total_returned": sum(len(q["returned"]) for q in per_query),
    }


def run_micro_spike() -> dict[str, Any]:
    """Run BOTH mechanisms over TF-043's fixtures:
    the spike (factual_lookup, all 106 items as the
    canonical family, budget 900) and the shipping
    engine through TF-043's evaluate_retrieval on a
    freshly built fixture DB, at the defaults and
    at the open setting. Deterministic: a second
    run files identically."""
    pipeline = SpikeMicroRAG()
    items = fixture_items()
    micro_per_query: list[dict[str, Any]] = []
    for row in FIXTURE_QUERIES:
        packet = pipeline.assemble(
            query=row["query"], mode=QueryMode.FACTUAL_LOOKUP,
            candidates_by_family={"canonical": items})
        returned = [c.item_id for c in packet.canonical_facts]
        scored = score_query(returned, row["relevant"])
        micro_per_query.append({
            "id": row["id"], "kind": row["kind"], "query": row["query"],
            "packet_budget_used": packet.packet_budget_used, **scored})
    result: dict[str, Any] = {
        "micro": {
            "per_query": micro_per_query,
            "aggregate": _aggregate(micro_per_query),
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        db_path = build_fixture_db(Path(tmp) / "tf140_fixture.db")
        for setting, kwargs in (
                ("shipping_default", {}),
                ("shipping_open", {"min_score": 0.0})):
            report = evaluate_retrieval(db_path, **kwargs)
            per_query = [{
                "id": q["id"], "kind": q["kind"], "query": q["query"],
                "returned": q["returned"], "hits": q["hits"],
                "precision": q["precision"], "recall": q["recall"],
                "f1": q["f1"], "reciprocal_rank": q["reciprocal_rank"],
            } for q in report["per_query"]]
            aggregate = {
                "queries": report["aggregate"]["queries"],
                "mean_precision": round(
                    report["aggregate"]["mean_precision"], 4),
                "mean_recall": round(
                    report["aggregate"]["mean_recall"], 4),
                "mean_f1": round(report["aggregate"]["mean_f1"], 4),
                "mean_reciprocal_rank": round(
                    report["aggregate"]["mean_reciprocal_rank"], 4),
                "perfect_recall_queries":
                    report["aggregate"]["perfect_recall_queries"],
                "total_returned": sum(
                    len(q["returned"]) for q in per_query),
            }
            result[setting] = {
                "per_query": per_query, "aggregate": aggregate}
    return result


#: The filed expectation for the spike run (deterministic:
#: aggregates + the micro mechanism's returned ids per query).
SPIKE_EXPECTED: dict[str, Any] = {
    "micro_aggregate": {
        "queries": 9, "mean_precision": 0.3333, "mean_recall": 0.8889,
        "mean_f1": 0.4749, "mean_reciprocal_rank": 1.0,
        "perfect_recall_queries": 8, "total_returned": 45,
    },
    "shipping_default_aggregate": {
        "queries": 9, "mean_precision": 0.8889, "mean_recall": 0.963,
        "mean_f1": 0.9111, "mean_reciprocal_rank": 1.0,
        "perfect_recall_queries": 8, "total_returned": 17,
    },
    "shipping_open_aggregate": {
        "queries": 9, "mean_precision": 0.8148, "mean_recall": 1.0,
        "mean_f1": 0.8815, "mean_reciprocal_rank": 1.0,
        "perfect_recall_queries": 9, "total_returned": 21,
    },
    "micro_returned": {
        "Q1": ["QF4301", "4301", "Lantern Hollow", "QF4311", "4311"],
        "Q2": ["QF4302", "4302", "mill", "QF4312", "Bracken Ford"],
        "Q3": ["QF4303", "4303", "QF4313", "QF4301", "QF4302"],
        "Q4": ["QF4304", "QF4301", "QF4302", "QF4303", "QF4305"],
        "Q5": ["QF4301", "QF4305", "4301", "ferry", "QF4302"],
        "Q6": ["QF4301", "QF4302", "QF4303", "QF4304", "QF4305"],
        "Q7": ["ferry", "QF4301", "QF4305", "QF4311", "4301"],
        "Q8": ["QF4311", "4311", "QF4301", "QF4304", "QF4313"],
        "Q9": ["QF4301", "4301", "Lantern Hollow", "QF4302", "QF4303"],
    },
    "micro_packet_budget_used": 400,
}


# ── The filed probes ─────────────────────────────────────────────────────────

MICRO_PROBES: tuple[dict[str, Any], ...] = (
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
    {"id": "spike_score_weight_wrong_is_named",
     "kind": "weight_wrong",
     "detail": "a weight map with the similarity weight perturbed "
               "(0.24 -> 0.20) is named against the filed weights"},
    {"id": "measured_metric_wrong_is_named",
     "kind": "measured_wrong",
     "detail": "an expected filing with the micro mean precision "
               "perturbed is named against the measured aggregate"},
    {"id": "schema_field_missing_from_spike_is_named",
     "kind": "schema_field_missing",
     "detail": "a field map with token_cost dropped from "
               "RetrievalItem is named against the schemas"},
    {"id": "unknown_verdict_value_is_named",
     "kind": "verdict_unknown",
     "detail": "a verdict value outside the roadmap's three is named"},
)


def run_micro_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in MICRO_PROBES:
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
            row = dict(COMPARISON_ROWS[0], shipping_engine="")
            passed = bool(comparison_row_problems(row))
        elif kind == "weight_wrong":
            mapping = dict(SCORE_WEIGHTS, similarity=0.20)
            passed = bool(score_weights_problems(mapping))
        elif kind == "measured_wrong":
            spike = run_micro_spike()
            expected = dict(
                SPIKE_EXPECTED["micro_aggregate"], mean_precision=0.5)
            passed = bool(measured_problems(
                spike["micro"]["aggregate"], expected))
        elif kind == "schema_field_missing":
            field_map = spike_field_map()
            field_map["RetrievalItem"] = tuple(
                f for f in field_map["RetrievalItem"]
                if f != "token_cost")
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

def render_micro_rag_verdict_markdown() -> str:
    """Render the verdict deterministically — the
    exact text filed as
    docs/architecture/MICRO_RAG_VERDICT.md."""
    spike = run_micro_spike()
    micro = spike["micro"]
    ship = spike["shipping_default"]
    ship_open = spike["shipping_open"]
    lines: list[str] = []
    lines.append("# Micro-RAG Verdict — TF-140")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/knowledge/micro_rag_verdict.py` — the "
        "filed doc IS the render: regenerate with "
        "`python3 -m thoughtforge.knowledge.micro_rag_verdict`. "
        "The checker (`micro_rag_verdict_problems()`) re-derives "
        "every claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{MICRO_RAG_VERDICT_FORMAT}`")
    lines.append(f"Verdict: **{VERDICT['verdict']}**")
    lines.append(f"Comparison rows: {len(COMPARISON_ROWS)}")
    lines.append(f"Probes: {len(MICRO_PROBES)}/{len(MICRO_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: the "
        "wing cannot import — `research_data/src/wyrdforge/models` "
        "does not exist and `micro_rag_pipeline.py` imports its "
        "types from that missing package (TF-006 / D-WING-1). The "
        "pipeline itself is a reranker + assembler, not a "
        "retriever: `score()` (a nine-term weighted sum over a "
        "substring similarity; recency is the constant 0.5) and "
        "`assemble()` (per-family top 5, token budget 900, typed "
        "packet sections, NO score threshold, NO corpus, NO "
        "index, NO chunking, NO candidate gathering in code). "
        "TF-006's census files micro-RAG OVERLAP in purpose vs "
        "the shipping engine, different shape. TF-043's fixtures "
        "are the measuring ground both mechanisms run over.")
    lines.append("")
    lines.append("## 2. The comparison")
    lines.append("")
    lines.append("| Dimension | Micro-RAG | Shipping engine | Finding |")
    lines.append("|---|---|---|---|")
    for row in COMPARISON_ROWS:
        lines.append(
            f"| `{row['dimension']}` | {row['micro_rag']} | "
            f"{row['shipping_engine']} | **{row['finding']}** |")
    lines.append("")
    lines.append("## 3. The verdict: RETIRE")
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
    lines.append("## 4. The working spike + the measurement")
    lines.append("")
    lines.append(
        "Spike dataclasses rebuilt from the two schemas "
        "(RetrievalItem; RankedCandidate / TruthPacket / "
        "MicroContextPacket + the QueryMode enum), namespaced "
        "`Spike*` in the verdict module — NOT the missing "
        "`wyrdforge.models` package and NOT production. "
        "`SpikeMicroRAG` re-implements the pipeline faithfully: "
        "the exact MODE_TARGETS per mode, the nine filed score "
        "weights (similarity 0.24, task_relevance 0.24, "
        "support_quality 0.14, scope_match 0.12, recency 0.08, "
        "contradiction_penalty -0.10, token_cost_penalty -0.04, "
        "novelty_bonus 0.04, bond_fit 0.08), per-family cap 5, "
        "budget 900, the section mapping. The fixture adaptation "
        "is filed in the module docstring: TF-043's corpus "
        "verbatim as 106 items (item_id = TF-043's bundle id; "
        "confidence = entity popularity, else the schema "
        "default 0.5; lexical_terms and facets empty, as the "
        "wing's own test items file them; token_cost the schema "
        "default 80), all as the canonical family, mode "
        "factual_lookup. The shipping engine runs through "
        "TF-043's own `evaluate_retrieval`; the spike's returns "
        "are scored by TF-043's own `score_query`.")
    lines.append("")
    lines.append("Aggregates (TF-043's nine queries):")
    lines.append("")
    for label, agg in (
            ("shipping, defaults (min_score 0.45)",
             ship["aggregate"]),
            ("shipping, open (min_score 0)", ship_open["aggregate"]),
            ("micro-RAG spike", micro["aggregate"])):
        lines.append(
            f"- {label}: precision {agg['mean_precision']} / "
            f"recall {agg['mean_recall']} / F1 {agg['mean_f1']} / "
            f"MRR {agg['mean_reciprocal_rank']} / perfect-recall "
            f"queries {agg['perfect_recall_queries']} / total "
            f"returned {agg['total_returned']}")
    lines.append("")
    lines.append("Per query (shipping defaults vs micro-RAG spike):")
    lines.append("")
    lines.append(
        "| Query | Shipping returned | P / R | Micro returned | P / R |")
    lines.append("|---|---|---|---|---|")
    ship_by_id = {q["id"]: q for q in ship["per_query"]}
    for q in micro["per_query"]:
        s = ship_by_id[q["id"]]
        lines.append(
            f"| `{q['id']}` {q['query']} | "
            f"{', '.join(s['returned']) or '(nothing)'} | "
            f"{round(s['precision'], 4)} / {round(s['recall'], 4)} | "
            f"{', '.join(q['returned'])} | "
            f"{round(q['precision'], 4)} / {round(q['recall'], 4)} |")
    lines.append("")
    lines.append(
        "The two decisive queries: Q6 (negative — 'volcano "
        "eruption crater') — the shipping engine returns nothing "
        "and scores 1/1; the spike returns the five "
        "highest-confidence entities at similarity 0 and scores "
        "0/0. Q9 (geonames — 'Lantern Hollow') — the spike files "
        "recall 1.0 against the shipping defaults' 0.6667: the "
        "filed GeoNames threshold defect (TF-043), the spike's "
        "one win, bought by the same missing threshold that "
        "fails Q6. Every micro packet used exactly 400 of its "
        "900 budget (5 items x the default token_cost 80).")
    lines.append("")
    lines.append("## 5. The probes")
    lines.append("")
    for probe in MICRO_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-140 discharges the micro-RAG share of D-WING-1 "
        "(verdict decided by measurement on TF-043's fixtures, "
        "narrowed in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). D-WING-1 "
        "itself stays OPEN for the TF-141..TF-142 verdicts and "
        "the TF-143 rebuild / TF-144 retirements / TF-149 tests "
        "it assigns. The retirement this verdict files is "
        "executed by TF-144, not by TF-140.")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Spike grade only: the spike is a faithful rebuild "
        "for measurement, NOT production and NOT an adoption — "
        "TF-144 owns executing the retirement; TF-149 owns the "
        "wing's tests.")
    lines.append(
        "- The verdict covers the retrieval mechanism. The "
        "packet's typed sections, token budget and "
        "contradiction injection are filed as harvested "
        "lessons for TF-144's closing doc — unmeasured by "
        "these fixtures and NOT adopted here.")
    lines.append(
        "- Fixture grade only: TF-043's corpus is synthetic; "
        "no live model was read and live retrieval quality "
        "is NOT PROVEN (TF-173 class).")
    lines.append(
        "- The candidate gathering the spec proposes was "
        "never implemented in the wing; a different gathering "
        "would be a different mechanism than the one measured.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_micro_rag_verdict_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / MICRO_RAG_VERDICT_DOC
    doc.write_text(render_micro_rag_verdict_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.knowledge.micro_rag_verdict`."""
    doc = write_micro_rag_verdict_doc()
    print(
        f"micro-RAG verdict regenerated: {VERDICT['verdict']} -> "
        f"{MICRO_RAG_VERDICT_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def _wing_mode_targets() -> dict[str, tuple[str, ...]]:
    """The pipeline's MODE_TARGETS, parsed live
    from micro_rag_pipeline.py by AST (the wing
    cannot be imported)."""
    tree = ast.parse(_PIPELINE.read_text(encoding="utf-8"))
    targets: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "MODE_TARGETS"
                   for t in node.targets):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Attribute) and isinstance(
                    value, ast.Tuple):
                targets[key.attr] = tuple(
                    elt.value for elt in value.elts
                    if isinstance(elt, ast.Constant))
    return targets


def micro_rag_verdict_problems() -> list[str]:
    """Re-derive TF-140's claims from the live
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
    if findings.count("shipping-wins") != 3 \
            or findings.count("micro-wins") != 1 \
            or findings.count("different") != 4 \
            or findings.count("overlap") != 1:
        problems.append(
            f"the findings are {findings.count('shipping-wins')} "
            f"shipping-wins / {findings.count('micro-wins')} "
            f"micro-wins / {findings.count('different')} different / "
            f"{findings.count('overlap')} overlap, not 3 / 1 / 4 / 1")

    # The wing as censused: models missing, the pipeline
    # importing them, its weights + mode targets as filed.
    if (_WING / "models").exists():
        problems.append(
            "wing: wyrdforge.models now exists — TF-143's rebuild "
            "has landed and this verdict's premise must be "
            "re-ruled")
    pipeline_text = _PIPELINE.read_text(encoding="utf-8")
    if "from wyrdforge.models.micro_rag import" not in pipeline_text:
        problems.append(
            "wing: micro_rag_pipeline.py no longer imports the "
            "missing models package")
    for op in ("def score", "def assemble", "MODE_TARGETS"):
        if op not in pipeline_text:
            problems.append(f"wing: micro_rag_pipeline.py lost {op}")
    for name, weight in SCORE_WEIGHTS.items():
        if f"{name} * {weight}" not in pipeline_text and \
                f"{name} * {weight:.2f}" not in pipeline_text:
            problems.append(
                f"wing: the score term {name} is no longer filed "
                f"as * {weight} in micro_rag_pipeline.py")
    problems.extend(score_weights_problems(SCORE_WEIGHTS))
    wing_targets = _wing_mode_targets()
    spike_targets = {
        mode.name: families
        for mode, families in SpikeMicroRAG.MODE_TARGETS.items()}
    if wing_targets != spike_targets:
        problems.append(
            f"spike: MODE_TARGETS {spike_targets} drifted from the "
            f"pipeline's {wing_targets}")

    # The schemas parse; the spike's field sets + enum match them.
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
        (_SCHEMAS / "micro_context_packet.schema.json").read_text(
            encoding="utf-8"))
    if [m.value for m in QueryMode] != \
            packet_schema["$defs"]["QueryMode"]["enum"]:
        problems.append("spike: QueryMode drifted from the schema")

    # The fixture adaptation covers TF-043's relevant ids.
    items = fixture_items()
    if len(items) != 106:
        problems.append(
            f"fixtures: the adaptation files {len(items)} items, "
            "not 106")
    item_ids = {item.item_id for item in items}
    for row in FIXTURE_QUERIES:
        missing = set(row["relevant"]) - item_ids
        if missing:
            problems.append(
                f"fixtures: query {row['id']} relevant ids "
                f"{sorted(missing)} have no spike item")

    # Both mechanisms reproduce their filed measurements, and
    # the shipping aggregates still match TF-043's baseline.
    spike = run_micro_spike()
    problems.extend(measured_problems(
        spike["micro"]["aggregate"], SPIKE_EXPECTED["micro_aggregate"]))
    problems.extend(measured_problems(
        spike["shipping_default"]["aggregate"],
        SPIKE_EXPECTED["shipping_default_aggregate"]))
    problems.extend(measured_problems(
        spike["shipping_open"]["aggregate"],
        SPIKE_EXPECTED["shipping_open_aggregate"]))
    micro_returned = {
        q["id"]: q["returned"] for q in spike["micro"]["per_query"]}
    if micro_returned != SPIKE_EXPECTED["micro_returned"]:
        problems.append(
            "spike: the micro returned ids drifted from their "
            "filing")
    budgets = {
        q["packet_budget_used"] for q in spike["micro"]["per_query"]}
    if budgets != {SPIKE_EXPECTED["micro_packet_budget_used"]}:
        problems.append(
            f"spike: the micro packets used budgets {budgets}, "
            "not the filed 400 each")
    for setting, key in (("default", "shipping_default"),
                         ("open", "shipping_open")):
        agg = spike[key]["aggregate"]
        for metric, pinned in BASELINE[setting].items():
            if agg[metric] != pinned:
                problems.append(
                    f"shipping: {setting}.{metric} measures "
                    f"{agg[metric]}, not TF-043's pinned {pinned}")

    # The probes all behave.
    probes = run_micro_probes()
    if len(probes) != len(MICRO_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the micro probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The doc files the render verbatim.
    doc = _REPO_ROOT / MICRO_RAG_VERDICT_DOC
    if not doc.exists():
        problems.append(f"the write-up {MICRO_RAG_VERDICT_DOC} is missing")
    elif doc.read_text(encoding="utf-8") != \
            render_micro_rag_verdict_markdown():
        problems.append(
            f"the filed doc {MICRO_RAG_VERDICT_DOC} is not the "
            "render verbatim (regenerate with python3 -m "
            "thoughtforge.knowledge.micro_rag_verdict)")
    return problems


def micro_rag_verdict_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    spike = run_micro_spike()
    probes = run_micro_probes()
    return {
        "format": MICRO_RAG_VERDICT_FORMAT,
        "doc": MICRO_RAG_VERDICT_DOC,
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
    "MICRO_PROBES",
    "MICRO_RAG_VERDICT_DOC",
    "MICRO_RAG_VERDICT_FORMAT",
    "SCORE_WEIGHTS",
    "SPIKE_EXPECTED",
    "SPIKE_SHAPES",
    "VERDICT",
    "VERDICTS",
    "QueryMode",
    "SpikeMicroContextPacket",
    "SpikeMicroRAG",
    "SpikeRankedCandidate",
    "SpikeRetrievalItem",
    "SpikeTruthPacket",
    "comparison_row_problems",
    "fixture_items",
    "measured_problems",
    "micro_rag_verdict_as_data",
    "micro_rag_verdict_problems",
    "render_micro_rag_verdict_markdown",
    "run_micro_probes",
    "run_micro_spike",
    "schema_field_problems",
    "score_weights_problems",
    "spike_field_map",
    "verdict_data_problems",
    "write_micro_rag_verdict_doc",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
