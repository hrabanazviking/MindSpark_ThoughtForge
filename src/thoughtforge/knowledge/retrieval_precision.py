"""
Retrieval precision fixtures, as data + measurement (TF-043).

Part III's second measurement slice: TF-041 mapped the store
and pinned its fusion arithmetic; this slice gives retrieval a
*known-answer* fixture and pins the baseline TF-064's gate
will judge against:

  * ``FIXTURE_ENTITIES`` / ``FIXTURE_REFERENCE_CHUNKS`` /
    ``FIXTURE_CONCEPTNET_EDGES`` / ``FIXTURE_GEONAMES`` — a
    synthetic corpus in TF-017's invented Lantern Hollow world
    (plus filler rows, so FTS bm25 lands in a realistic range —
    TF-041 measured that a one-row corpus scores near zero).
    Eight topical records, three hard distractors that share
    query terms without answering them, and filler.
  * ``FIXTURE_QUERIES`` — nine queries whose right answers are
    known by construction: entity+reference answers, a shared-
    term query with two right answers, a ConceptNet-isolating
    query, a GeoNames query, a hard-negative query where the
    distractor IS the target, and a negative query whose right
    answer is the empty bundle.
  * ``build_fixture_db`` — builds the corpus into a fresh
    SQLite DB through the tree's own schema module.
  * ``score_query`` / ``evaluate_retrieval`` — precision,
    recall, F1 and reciprocal rank per query and as means.
    Metric rules, stated once: precision counts returned
    records whose bundle id is in the known-answer set;
    an empty return against a non-empty answer set scores
    precision 0 and recall 0; an empty return against an
    empty answer set (the negative query) scores 1 and 1 —
    correctly returning nothing is a success, never a
    vacuous one, because the query is labelled negative by
    construction; reciprocal rank is 0 for a labelled-negative
    query and those queries are excluded from the MRR mean.
  * ``BASELINE`` — the aggregates measured on this VM at
    banking time, at the two settings that matter: the
    defaults the core actually runs (path sql, top_k 10,
    min_score = MIN_MEMORY_SCORE 0.45) and the open setting
    (min_score 0) that shows what the threshold is holding
    back. ``baseline_problems`` names any metric that falls
    below its pinned baseline (beyond tolerance) — TF-064's
    gate consumes it; improvement is never a problem.

Two findings the baseline carries honestly (measured, not
retuned — TF-044 owns the scoring questions):

  * **GeoNames is threshold-dead at the defaults.** A
    geonames record's fts_score is the fixed constant 0.5, so
    its composite is always exactly 0.285 — it can NEVER pass
    the 0.45 default threshold. The "Lantern Hollow" query's
    place answer is present at min_score 0 and absent at the
    defaults (recall 1.000 open vs 0.667 default).
  * **ConceptNet fires on any single shared term.** The
    ferry→river edge (weight 4.0, composite exactly 0.46)
    rides every query containing "ferry" or "river" whether
    it answers the query or not: three of the nine default
    queries return it as a non-answer, and each of those
    queries' precision is exactly 2/3 for that reason alone.

Proof grade: FIXTURE — synthetic corpus, SQL path only, on
this VM. The vector/hybrid paths are not exercised
(sentence-transformers absent, D-ENV-1; TF-053 owns).

The human-readable write-up is
``docs/architecture/RETRIEVAL_PRECISION.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thoughtforge.etl.schema import get_connection, initialize_schema
from thoughtforge.knowledge.forge import KnowledgeForge
from thoughtforge.knowledge.scoring import MIN_MEMORY_SCORE

# ── The synthetic corpus (TF-017's Lantern Hollow world) ─────────────────────
#
# (qid, label, description, popularity). Descriptions repeat
# their topic terms the way real reference text does; the
# distractors (QF431x) share terms with the targets without
# answering the target queries — and one query ("ferry house
# inn") makes a distractor the target, so the fixture cannot
# pass by merely ignoring the shared-term records.

FIXTURE_ENTITIES: tuple[tuple[str, str, str, float], ...] = (
    ("QF4301", "Lantern Hollow Ferry",
     "ferry ferry ferry river crossing lantern hollow ferry "
     "carries passengers cargo river crossing timetable ferry", 0.9),
    ("QF4302", "Bracken Mill",
     "mill grain flour bracken mill grinds grain flour "
     "waterwheel mill grain sacks", 0.85),
    ("QF4303", "Alder Observatory",
     "observatory telescope stars alder observatory astronomers "
     "stars telescope dome observatory", 0.8),
    ("QF4304", "Kestrel Lighthouse",
     "lighthouse beacon kestrel lighthouse beacon coast ships "
     "lighthouse keeper beacon", 0.8),
    ("QF4305", "Stonebridge Crossing",
     "bridge river crossing stonebridge bridge stone arch "
     "river crossing bridge", 0.7),
    # Distractors: shared terms, wrong answers (until targeted).
    ("QF4311", "Ferry House Inn",
     "ferry house inn lodging rooms ferry travellers rest "
     "ferry house meals", 0.6),
    ("QF4312", "Mill Cottage",
     "mill cottage lodging old mill stream cottage garden", 0.5),
    ("QF4313", "Star Inn",
     "stars inn sign painted stars lodging star inn rooms", 0.5),
)

ENTITY_FILLER_COUNT = 60
ENTITY_FILLER_TEXT = "unrelated filler text about mountains and deserts"

# (chunk_id, title, content). Explicit ids so the known-answer
# sets can name reference records exactly as the bundle does
# (the bundle's record_id for a chunk is str(chunk_id)).
FIXTURE_REFERENCE_CHUNKS: tuple[tuple[int, str, str], ...] = (
    (4301, "Lantern Hollow Ferry Timetable",
     "ferry timetable lantern hollow ferry departs dawn ferry "
     "timetable crossings daily ferry lantern hollow"),
    (4302, "Bracken Mill Grain Prices",
     "bracken mill grain flour prices mill grain sacks flour "
     "bracken mill grain"),
    (4303, "Alder Observatory Visiting",
     "observatory visiting alder telescope stars evening "
     "observatory alder observatory"),
    (4311, "Ferry House Inn Menu",
     "ferry house inn menu ferry travellers dinner ferry house"),
)

REFERENCE_FILLER_COUNT = 30
REFERENCE_FILLER_TEXT = "unrelated filler chunk about mountains and deserts"

# (concept1, relation, concept2, weight). The bundle id for a
# ConceptNet record is its concept1 (the forge's fallback).
FIXTURE_CONCEPTNET_EDGES: tuple[tuple[str, str, str, float], ...] = (
    ("ferry", "crosses", "river", 4.0),
    ("mill", "grinds", "grain", 2.0),
)

# (geonames_id, name, population). The bundle id for a
# GeoNames record is its name (the forge's fallback).
FIXTURE_GEONAMES: tuple[tuple[int, str, int], ...] = (
    (430001, "Lantern Hollow", 1200),
    (430002, "Bracken Ford", 300),
)

# ── The known-answer queries ─────────────────────────────────────────────────
#
# "relevant" names bundle record ids: entity qids, reference
# chunk ids as strings, a ConceptNet concept1, a GeoNames
# name — exactly the ids KnowledgeForge._build_bundle emits.

FIXTURE_QUERIES: tuple[dict[str, Any], ...] = (
    {"id": "Q1", "kind": "entity+reference",
     "query": "Lantern Hollow ferry",
     "relevant": ("QF4301", "4301")},
    {"id": "Q2", "kind": "entity+reference",
     "query": "Bracken Mill grain flour",
     "relevant": ("QF4302", "4302")},
    {"id": "Q3", "kind": "entity+reference",
     "query": "Alder Observatory telescope stars",
     "relevant": ("QF4303", "4303")},
    {"id": "Q4", "kind": "entity",
     "query": "Kestrel lighthouse beacon",
     "relevant": ("QF4304",)},
    {"id": "Q5", "kind": "shared-term",
     "query": "river crossing",
     "relevant": ("QF4301", "QF4305")},
    {"id": "Q6", "kind": "negative",
     "query": "volcano eruption crater",
     "relevant": ()},
    {"id": "Q7", "kind": "conceptnet",
     "query": "ferry crosses river",
     "relevant": ("ferry",)},
    {"id": "Q8", "kind": "distractor-as-target",
     "query": "ferry house inn",
     "relevant": ("QF4311", "4311")},
    {"id": "Q9", "kind": "geonames",
     "query": "Lantern Hollow",
     "relevant": ("QF4301", "4301", "Lantern Hollow")},
)

# ── Building the fixture DB ──────────────────────────────────────────────────


def build_fixture_db(path: Path) -> Path:
    """Build the fixture corpus into a fresh DB at ``path``."""
    initialize_schema(path)
    conn = get_connection(path)
    try:
        for qid, label, description, popularity in FIXTURE_ENTITIES:
            conn.execute(
                "INSERT INTO entities (qid, label_en, description_en,"
                " aliases_en, instance_of, popularity_score, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (qid, label, description, label, "place",
                 popularity, "wikidata"),
            )
        for i in range(ENTITY_FILLER_COUNT):
            conn.execute(
                "INSERT INTO entities (qid, label_en, description_en,"
                " aliases_en, instance_of, popularity_score, source)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"QF44{i:02d}", f"filler {i}", ENTITY_FILLER_TEXT,
                 "x", "place", 0.1, "wikidata"),
            )
        for chunk_id, title, content in FIXTURE_REFERENCE_CHUNKS:
            conn.execute(
                "INSERT INTO reference_chunks (chunk_id, source_file,"
                " chunk_index, title, content, tags, word_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (chunk_id, "synthetic.md", chunk_id, title, content,
                 "synthetic", len(content.split())),
            )
        for i in range(REFERENCE_FILLER_COUNT):
            conn.execute(
                "INSERT INTO reference_chunks (chunk_id, source_file,"
                " chunk_index, title, content, tags, word_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (4400 + i, "filler.md", 4400 + i, f"filler chunk {i}",
                 REFERENCE_FILLER_TEXT, "filler",
                 len(REFERENCE_FILLER_TEXT.split())),
            )
        for concept1, relation, concept2, weight in FIXTURE_CONCEPTNET_EDGES:
            conn.execute(
                "INSERT INTO conceptnet_edges"
                " (concept1, relation, concept2, weight)"
                " VALUES (?, ?, ?, ?)",
                (concept1, relation, concept2, weight),
            )
        for geonames_id, name, population in FIXTURE_GEONAMES:
            conn.execute(
                "INSERT INTO geonames_places (geonames_id, name,"
                " ascii_name, latitude, longitude, feature_class,"
                " feature_code, country_code, population)"
                " VALUES (?, ?, ?, 0, 0, 'P', 'PPL', 'SY', ?)",
                (geonames_id, name, name, population),
            )
        conn.commit()
        conn.execute("INSERT INTO entities_fts(entities_fts) VALUES('rebuild')")
        conn.execute("INSERT INTO reference_fts(reference_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()
    return path


# ── Metrics ──────────────────────────────────────────────────────────────────


def score_query(returned_ids: list[str], relevant: tuple[str, ...]) -> dict[str, Any]:
    """
    Precision / recall / F1 / reciprocal rank for one query,
    under the rules stated in the module docstring.
    """
    relevant_set = set(relevant)
    hits = sum(1 for record_id in returned_ids if record_id in relevant_set)
    if returned_ids:
        precision = hits / len(returned_ids)
    else:
        precision = 1.0 if not relevant_set else 0.0
    if relevant_set:
        recall = hits / len(relevant_set)
    else:
        recall = 1.0 if not returned_ids else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    reciprocal_rank = 0.0
    for rank, record_id in enumerate(returned_ids, start=1):
        if record_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "returned": list(returned_ids),
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate_retrieval(
    db_path: Path,
    *,
    top_k: int = 10,
    min_score: float = MIN_MEMORY_SCORE,
) -> dict[str, Any]:
    """
    Run every fixture query through ``KnowledgeForge`` on the
    SQL path (the path that can run on this machine) and
    return per-query scores plus the aggregate means.
    """
    forge = KnowledgeForge(db_path=db_path)
    per_query: list[dict[str, Any]] = []
    for row in FIXTURE_QUERIES:
        bundle = forge.retrieve(
            row["query"], path="sql", top_k=top_k, min_score=min_score
        )
        scored = score_query(
            [r.record_id for r in bundle.activated_records], row["relevant"]
        )
        per_query.append({"id": row["id"], "kind": row["kind"],
                          "query": row["query"], **scored})
    answered = [q for q in per_query if q["kind"] != "negative"]
    aggregate = {
        "queries": len(per_query),
        "mean_precision": sum(q["precision"] for q in per_query) / len(per_query),
        "mean_recall": sum(q["recall"] for q in per_query) / len(per_query),
        "mean_f1": sum(q["f1"] for q in per_query) / len(per_query),
        "mean_reciprocal_rank": (
            sum(q["reciprocal_rank"] for q in answered) / len(answered)
        ),
        "perfect_recall_queries": sum(
            1 for q in per_query if q["recall"] == 1.0
        ),
    }
    return {
        "top_k": top_k,
        "min_score": min_score,
        "per_query": per_query,
        "aggregate": aggregate,
    }


# ── The pinned baseline (measured on this VM at banking) ────────────────────
#
# Aggregates rounded to 4 decimals. bm25 is corpus-shaped, so
# the checker allows BASELINE_TOLERANCE of platform slack; a
# fall beyond it is a regression by name (TF-064's gate).

BASELINE_TOLERANCE = 0.01

BASELINE: dict[str, dict[str, float]] = {
    "default": {  # path sql, top_k 10, min_score 0.45 (the core's defaults)
        "mean_precision": 0.8889,
        "mean_recall": 0.9630,
        "mean_f1": 0.9111,
        "mean_reciprocal_rank": 1.0,
    },
    "open": {  # path sql, top_k 10, min_score 0.0
        "mean_precision": 0.8148,
        "mean_recall": 1.0,
        "mean_f1": 0.8815,
        "mean_reciprocal_rank": 1.0,
    },
}

# The two findings, as constants the tests pin live.
GEONAMES_DEFAULT_SCORE = 0.285   # a geonames record's fixed composite
CONCEPTNET_FERRY_EDGE_SCORE = 0.46  # the weight-4.0 ferry edge's composite


def baseline_problems(
    report: dict[str, Any], *, setting: str
) -> list[str]:
    """
    Name every aggregate in ``report`` that has fallen below
    the pinned ``BASELINE[setting]`` beyond tolerance.
    Improvement is never a problem.
    """
    if setting not in BASELINE:
        return [f"unknown baseline setting {setting!r}"]
    problems: list[str] = []
    for metric, pinned in BASELINE[setting].items():
        measured = report["aggregate"][metric]
        if measured < pinned - BASELINE_TOLERANCE:
            problems.append(
                f"{setting}.{metric} regressed: {measured:.4f} "
                f"< baseline {pinned} (tolerance {BASELINE_TOLERANCE})"
            )
    return problems


__all__ = [
    "BASELINE",
    "BASELINE_TOLERANCE",
    "CONCEPTNET_FERRY_EDGE_SCORE",
    "ENTITY_FILLER_COUNT",
    "FIXTURE_CONCEPTNET_EDGES",
    "FIXTURE_ENTITIES",
    "FIXTURE_GEONAMES",
    "FIXTURE_QUERIES",
    "FIXTURE_REFERENCE_CHUNKS",
    "GEONAMES_DEFAULT_SCORE",
    "REFERENCE_FILLER_COUNT",
    "build_fixture_db",
    "baseline_problems",
    "evaluate_retrieval",
    "score_query",
]
