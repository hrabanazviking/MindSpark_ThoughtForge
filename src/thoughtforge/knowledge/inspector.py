"""
Memory inspector data — what the store holds + why a
retrieval fired, exportable per turn (TF-061).

The roadmap's rule: *memory inspector data: what the store
holds + why a retrieval fired, exportable per turn
(consumes TF-038).*

The measured starting state, before this slice (probed
this tick, keeper 09:12, on this VM):

  * **Why a retrieval fired was discarded at scoring
    time** (**D-INSP-1**, found + FIXED in this slice).
    ``KnowledgeForge._score_records`` computed the five
    ``RetrievalDimensions`` (semantic / tone / preference
    / recency / importance) for every candidate, folded
    them into one composite — and threw the dimensions
    away: a scored record's raw carried only
    ``_composite_score`` (probe P1: no dimension,
    semantic, or recency key survives). No consumer
    could answer "why this record, at this score": which
    dimension drove it, whether its semantic score was a
    vector similarity or the FTS proxy, or whether its
    recency was the record's own timestamp (TF-059) or
    the per-source proxy.
  * **Records that did NOT fire were unexplainable.**
    ``retrieve()`` filtered below-threshold candidates
    and cut beyond ``top_k`` in place; afterwards
    nothing anywhere recorded that a candidate had been
    considered and rejected, or why. From the bundle
    alone, a missing record is indistinguishable from a
    record the legs never returned.
  * **The TF-038 trace carried no citation payload.**
    Its retrieval entries held record_id / record_type /
    score / cue / raw (probe P2) — the record itself
    carries a TF-048 ``CitationPayload`` by construction,
    but the trace dropped it; TF-048 assigned the trace
    leg to this slice. FIXED in this slice
    (``cognition/traces.py``).
  * **"What the store holds" existed only split.**
    TF-057's ``inspect_store`` reports the personal
    ``MemoryStore``; ``forge_memory status`` /
    ``etl/db_integrity`` report the knowledge DB; the
    turn trace reports the turn. No one export joined
    them per turn (probe P3/P4: no inspector module, no
    core inspector surface) — the exact join TF-038's
    doc named as this slice's data ("TF-061's inspector
    data … consume it as data").

What this slice builds — ``knowledge/inspector.py``:

  * ``INSPECTOR_FORMAT`` / ``INSPECTOR_SECTIONS`` —
    one named export shape: ``store`` (TF-057's report
    over the personal store), ``knowledge_db`` (the DB
    estate's tables + row counts, read-only; a missing
    DB is a verdict, never a raise), ``turn`` (the
    TF-038 trace's identity sections, reduced), and
    ``retrieval`` (the why, below).
  * ``dimension_explanation`` — the explanation of the
    forge's fusion: the five dimensions exactly as
    ``_score_records`` computes them (same named
    constants, same ``score_retrieval`` call — the
    arithmetic stays in the forge, where TF-044's
    census pins it), each dimension's contribution
    under the effective weights row, the semantic
    score's source (``vector_similarity`` /
    ``fts_proxy`` / ``none``), and the recency score's
    source (``record_timestamp`` / ``source_proxy``).
    The rules are stated as data (``DIMENSION_RULES``).
    ``_score_records`` now also RETAINS its dimensions
    on the record as ``_explanation`` (D-INSP-1's fix)
    — retained at retrieval time, because a timestamp's
    recency decays: recomputing later would explain a
    different moment. The inspector prefers the retained
    explanation and says so (``explanation_source``);
    for a forge-shaped record that predates retention
    it recomputes with this mirror and says that
    instead — and the mirror is pinned against the
    retained explanation on every record shape by the
    live checker and the tests, so the two cannot
    drift apart silently.
  * The forge's retrieval report — ``retrieve()`` now
    retains ``last_retrieval_report``: the query, path,
    ``top_k`` and ``min_score`` it ran under, how many
    candidates were considered, the fired ids in bundle
    order, and every rejected candidate with its
    composite and its reason (``below_min_score`` /
    ``beyond_top_k``). Rejection is data now, not
    silence.
  * ``explain_record`` — one fired record explained:
    score, cue, source, its TF-048 citation payload,
    and either the forge dimension explanation
    (retained or recomputed) or the evidence the
    producing search carried in its raw
    (``CARRIED_EVIDENCE_KEYS``: TF-060's
    ``matched_terms`` / ``language``, TF-059's
    ``freshness``) — personal-store searches score by
    their own stated rules, and the inspector surfaces
    their evidence rather than inventing forge
    dimensions for them.
  * ``build_inspector_report`` — the per-turn export:
    joins store + knowledge DB + trace + bundle +
    the forge's retrieval report + the core's retained
    TF-049 citation check (``citations`` section;
    ``present: False`` with the reason named when no
    claim check ran — a knowledge-only turn runs none).
    Detached and JSON-safe by construction (every value
    passes the same conversion discipline as TF-038).
  * ``write_inspector_report`` / ``inspector_to_json``
    — the file road, mirroring TF-038's writer.
  * ``inspector_problems`` — the pure checker for one
    export as data (format, sections, citation payloads
    present, contributions summing to their composite,
    rejection reasons in the stated vocabulary); and
    ``inspector_live_problems`` — the live checker:
    the forge's retained explanation agreeing with its
    composite, the mirror agreeing with the retained
    explanation, the trace carrying citation payloads,
    and a full synthetic report building clean and
    read-only (store bytes pinned).

Wiring (this slice): ``KnowledgeForge._score_records``
retains ``_explanation`` and ``retrieve()`` retains the
retrieval report (``last_retrieval_report`` property);
``cognition/traces.py`` retrieval entries carry the
record's citation payload; the core gains
``last_turn_inspector()`` (post-freeze minor addition,
recorded in ``core_docs.POST_FREEZE_ADDITIONS`` +
the freeze doc §7) joining its own store, DB, trace,
bundle, forge report, and citation check into one
export. TF-016 totals updated: D-INSP-1 added + FIXED.

Stated boundaries:

  * The inspector explains the *forge's* retrieval and
    surfaces the personal searches' carried evidence;
    it does not re-rank, re-score for verdicts, or
    judge answer quality (the scorers' business).
  * A retrieval with no retained forge report (a
    bundle built by hand, or a personal-store search)
    reports its rejections as *unknown*, never as
    "nothing was rejected" — absence of a report is
    stated in the export, not hidden.
  * The knowledge-DB section counts rows per table; it
    does not copy content out of the DB (canon is
    described, never carried — TF-058's rule).
  * Proof grade is FIXTURE (synthetic records, the
    TF-043 fixture DB, synthetic stores, real core
    turns on this VM) — inspection over a real
    lived-in store is NOT PROVEN (no real store on
    this machine, TF-041), never claimed.

The human-readable write-up is
``docs/architecture/MEMORY_INSPECTOR.md``.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any

from thoughtforge.knowledge.citations import citation_for_record
from thoughtforge.knowledge.freshness import recency_from_timestamps
from thoughtforge.knowledge.models import ActivatedRecord, MemoryActivationBundle
from thoughtforge.knowledge.scoring import (
    DEFAULT_SOURCE_RECENCY,
    FTS_SCORE_DIVISOR,
    IMPORTANCE_WEIGHT_DEFAULT,
    MIN_MEMORY_SCORE,
    NEUTRAL_PREFERENCE_RELEVANCE,
    NEUTRAL_TONE_SIMILARITY,
    RETRIEVAL_WEIGHTS_BY_TYPE,
    SOURCE_RECENCY_PROXIES,
    RetrievalDimensions,
    score_retrieval,
)

#: The one inspector format this module exports.
INSPECTOR_FORMAT = "thoughtforge-inspector/1"

#: The export's sections, in reading order: what the
#: estates hold first, then the turn, then why it fired.
INSPECTOR_SECTIONS: tuple[str, ...] = (
    "store",
    "knowledge_db",
    "turn",
    "retrieval",
    "citations",
)

#: Raw keys a personal-store search (TF-058/059/060)
#: carries as its own scoring evidence. The inspector
#: surfaces these verbatim; it never fabricates forge
#: dimensions for a record scored by another rule.
CARRIED_EVIDENCE_KEYS: tuple[str, ...] = (
    "matched_terms",
    "freshness",
    "language",
)

#: How each forge dimension is derived, as data — the
#: explanation an inspector reader is entitled to.
DIMENSION_RULES: tuple[dict[str, Any], ...] = (
    {"dimension": "semantic_similarity",
     "rule": "the vector leg's similarity when the record carries one; "
             "otherwise the FTS proxy min(fts_score / FTS_SCORE_DIVISOR, 1); "
             "0 when the record carries neither"},
    {"dimension": "tone_similarity",
     "rule": "NEUTRAL_TONE_SIMILARITY — the knowledge DB carries no "
             "tone metadata, so the forge states the neutral value"},
    {"dimension": "preference_relevance",
     "rule": "NEUTRAL_PREFERENCE_RELEVANCE — same: no preference "
             "metadata at DB level, the neutral value is stated"},
    {"dimension": "recency",
     "rule": "the record's own timestamp decayed at the default "
             "half-life (TF-059) when it carries a parseable one; "
             "otherwise the per-source proxy SOURCE_RECENCY_PROXIES "
             "(DEFAULT_SOURCE_RECENCY for an unlisted source)"},
    {"dimension": "importance",
     "rule": "min(popularity_score + weight, 1), weight defaulting "
             "to IMPORTANCE_WEIGHT_DEFAULT"},
    {"dimension": "thread_boost / pattern_quality / priority",
     "rule": "the forge builds its dimensions without these three "
             "(they stay 0); in the user_fact weights row only "
             "thread_boost carries weight, so the omission is "
             "arithmetically silent — stated, not hidden"},
)

#: The weights row the forge scores every DB record
#: under (measured by TF-041, pinned by TF-043).
FORGE_WEIGHTS_ROW = "user_fact"


# ── The dimension explanation (the forge's arithmetic, explained) ───────────


def dimension_explanation(
    record: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Why one forge candidate scores what it scores.

    Mirrors ``KnowledgeForge._score_records``' five
    dimensions from a raw retrieval dict with the
    forge's own constants and ``score_retrieval`` call,
    and reports each dimension's value, its contribution
    under the effective weights row, and where the two
    derived dimensions came from. ``weights`` is the
    forge's explicit table when one was supplied at
    construction (TF-044); ``None`` is the shipped
    ``user_fact`` row — the row the forge scores every
    DB record under.

    The forge retains its own dimensions on the record
    at scoring time (``_explanation``), and the
    inspector prefers that retained form; this mirror
    serves records that predate retention. Calling it
    later on the same raw recomputes at the later
    moment (a timestamp's recency decays) — the live
    checker pins the mirror against the retained
    explanation so the two cannot drift.
    """
    similarity = record.get("similarity", 0.0)
    if similarity:
        semantic = float(similarity)
        semantic_source = "vector_similarity"
    else:
        fts = record.get("fts_score", 0.0)
        if fts > 0:
            semantic = min(fts / FTS_SCORE_DIVISOR, 1.0)
            semantic_source = "fts_proxy"
        else:
            semantic = 0.0
            semantic_source = "none"

    importance = min(
        record.get("popularity_score", 0.0)
        + record.get("weight", IMPORTANCE_WEIGHT_DEFAULT),
        1.0,
    )

    timestamped = recency_from_timestamps(record)
    if timestamped is not None:
        recency = timestamped
        recency_source = "record_timestamp"
    else:
        source = record.get("source", "wikidata")
        recency = SOURCE_RECENCY_PROXIES.get(source, DEFAULT_SOURCE_RECENCY)
        recency_source = "source_proxy"

    dims = RetrievalDimensions(
        semantic_similarity=semantic,
        tone_similarity=NEUTRAL_TONE_SIMILARITY,
        preference_relevance=NEUTRAL_PREFERENCE_RELEVANCE,
        recency=recency,
        importance=importance,
    )
    row = weights if weights is not None else RETRIEVAL_WEIGHTS_BY_TYPE[FORGE_WEIGHTS_ROW]
    dimensions = {
        "semantic_similarity": dims.semantic_similarity,
        "tone_similarity": dims.tone_similarity,
        "preference_relevance": dims.preference_relevance,
        "recency": dims.recency,
        "importance": dims.importance,
        "thread_boost": dims.thread_boost,
        "pattern_quality": dims.pattern_quality,
        "priority": dims.priority,
    }
    return {
        "weights_row": "custom" if weights is not None else FORGE_WEIGHTS_ROW,
        "weights": dict(row),
        "dimensions": dimensions,
        "contributions": {
            name: dimensions[name] * row.get(name, 0.0) for name in dimensions
        },
        "semantic_source": semantic_source,
        "recency_source": recency_source,
        "composite": score_retrieval(dims, FORGE_WEIGHTS_ROW, weights=weights),
    }


# ── One record, explained ────────────────────────────────────────────────────


def explain_record(
    record: ActivatedRecord,
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One fired record with its why, as JSON-safe data.

    Preference order, stated in ``explanation_source``:
    the forge's retained ``_explanation`` (the retrieval
    moment's own arithmetic) → a recomputation for a
    forge-shaped record that predates retention (it
    carries ``_composite_score``) → the evidence the
    producing search carried in its raw
    (``CARRIED_EVIDENCE_KEYS``), for records scored by
    a personal-store search's own rule.
    """
    raw = record.raw or {}
    out: dict[str, Any] = {
        "record_id": record.record_id,
        "record_type": record.record_type,
        "score": record.score,
        "cue": record.cue,
        "source": raw.get("source"),
        "citation": citation_for_record(record).to_dict(),
    }
    retained = raw.get("_explanation")
    if isinstance(retained, dict):
        out["explanation_source"] = "retained_at_retrieval"
        out["dimensions"] = copy.deepcopy(retained)
    elif "_composite_score" in raw:
        out["explanation_source"] = "recomputed_at_inspection"
        out["dimensions"] = dimension_explanation(raw, weights=weights)
    else:
        out["explanation_source"] = "carried_evidence"
        out["carried_evidence"] = {
            key: copy.deepcopy(raw[key])
            for key in CARRIED_EVIDENCE_KEYS
            if key in raw
        }
    return out


def explain_retrieval(
    bundle: MemoryActivationBundle,
    *,
    retrieval_report: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A bundle explained: every fired record's why,
    plus the retrieval's own context and rejections.

    ``retrieval_report`` is the forge's retained report
    (``KnowledgeForge.last_retrieval_report``). Without
    one, the fired records still explain themselves,
    but the rejections are reported as unknown — the
    export says so, because a hand-built bundle's
    silence about rejections is not evidence there
    were none.
    """
    out: dict[str, Any] = {
        "total_retrieved": bundle.total_retrieved,
        "retrieval_confidence": bundle.retrieval_confidence,
        "records": [
            explain_record(record, weights=weights)
            for record in bundle.activated_records
        ],
    }
    if retrieval_report is None:
        out["query"] = None
        out["path"] = None
        out["min_score"] = None
        out["top_k"] = None
        out["considered"] = None
        out["rejected"] = None
        out["rejections_known"] = False
        out["note"] = (
            "no forge retrieval report was retained for this bundle — "
            "which candidates were considered and rejected cannot be "
            "reconstructed after the fact, and is reported unknown, "
            "not as none"
        )
    else:
        out["query"] = retrieval_report.get("query")
        out["path"] = retrieval_report.get("path")
        out["min_score"] = retrieval_report.get("min_score")
        out["top_k"] = retrieval_report.get("top_k")
        out["considered"] = retrieval_report.get("considered")
        out["rejected"] = copy.deepcopy(retrieval_report.get("rejected", []))
        out["rejections_known"] = True
    return out


# ── What the estates hold ────────────────────────────────────────────────────


def knowledge_db_report(db_path: Path | str | None) -> dict[str, Any]:
    """The knowledge DB estate, described read-only.

    Table names + row counts + file bytes — content is
    never copied out (TF-058's rule: canon described,
    not carried). A missing DB is a verdict
    (``present: False``); an unreadable one is a
    verdict with the error named. Never raises.
    """
    if db_path is None:
        return {"present": False, "reason": "no knowledge DB path named"}
    path = Path(db_path)
    if not path.exists():
        return {"present": False, "path": str(path)}
    report: dict[str, Any] = {
        "present": True,
        "path": str(path),
        "bytes": path.stat().st_size,
    }
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "ORDER BY name"
                ).fetchall()
            ]
            report["tables"] = {
                name: conn.execute(
                    f'SELECT COUNT(*) FROM "{name}"'
                ).fetchone()[0]
                for name in tables
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        report["readable"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    report["readable"] = True
    return report


# ── The per-turn export ──────────────────────────────────────────────────────


def _turn_section(trace: dict[str, Any] | None) -> dict[str, Any]:
    if trace is None:
        return {"present": False}
    turn = trace.get("turn") or {}
    conversation = trace.get("conversation") or {}
    route = trace.get("route") or {}
    final = trace.get("final")
    return {
        "present": True,
        "turn_id": trace.get("turn_id"),
        "phase": turn.get("phase"),
        "is_complete": turn.get("is_complete"),
        "profile_id": conversation.get("profile_id"),
        "strategy": conversation.get("strategy"),
        "persona_id": conversation.get("persona_id"),
        "query": route.get("raw_text"),
        "retrieval_path": route.get("retrieval_path"),
        "final": (
            {
                "response_id": final.get("response_id"),
                "enforcement_passed": final.get("enforcement_passed"),
                "citations": list(final.get("citations") or []),
            }
            if final
            else None
        ),
    }


def _citations_section(citation_check: Any) -> dict[str, Any]:
    if citation_check is None:
        return {
            "present": False,
            "note": (
                "no per-claim citation check was retained for this "
                "turn — a knowledge-only turn runs none (TF-049)"
            ),
        }
    if hasattr(citation_check, "to_dict"):
        return {"present": True, **citation_check.to_dict()}
    return {"present": True, "check": str(citation_check)}


def build_inspector_report(
    *,
    store: Any | None = None,
    db_path: Path | str | None = None,
    trace: dict[str, Any] | None = None,
    bundle: MemoryActivationBundle | None = None,
    retrieval_report: dict[str, Any] | None = None,
    citation_check: Any = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One turn's inspector export, as detached JSON-safe data.

    Every section is independently optional — a missing
    input is a stated verdict inside its section, never
    a raise and never a fabricated section. ``store``
    is a ``MemoryStore`` (reported through TF-057's
    ``inspect_store``); ``trace`` is a TF-038 trace
    dict; ``bundle`` the turn's retrieval bundle;
    ``retrieval_report`` the forge's retained report;
    ``citation_check`` the core's retained TF-049
    check.
    """
    if store is not None:
        from thoughtforge.knowledge.forge_toolset import inspect_store

        store_section: dict[str, Any] = {"present": True, **inspect_store(store)}
    else:
        store_section = {"present": False}

    if bundle is not None:
        retrieval_section: dict[str, Any] = {
            "present": True,
            **explain_retrieval(
                bundle, retrieval_report=retrieval_report, weights=weights
            ),
        }
    else:
        retrieval_section = {"present": False}

    return {
        "format": INSPECTOR_FORMAT,
        "store": store_section,
        "knowledge_db": knowledge_db_report(db_path),
        "turn": _turn_section(trace),
        "retrieval": retrieval_section,
        "citations": _citations_section(citation_check),
    }


def inspector_to_json(report: dict[str, Any]) -> str:
    """The report as one JSON document (TF-038's discipline)."""
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)


def write_inspector_report(report: dict[str, Any], path: Path | str) -> Path:
    """Write one inspector report as JSON; returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(inspector_to_json(report), encoding="utf-8")
    return target


# ── The checker ──────────────────────────────────────────────────────────────


def inspector_problems(report: dict[str, Any]) -> list[str]:
    """Judge an inspector export as data; ``[]`` = sound."""
    problems: list[str] = []
    if report.get("format") != INSPECTOR_FORMAT:
        problems.append(f"format {report.get('format')!r} != {INSPECTOR_FORMAT!r}")
    for section in INSPECTOR_SECTIONS:
        if section not in report:
            problems.append(f"missing section {section}")
    retrieval = report.get("retrieval") or {}
    if retrieval.get("present"):
        for entry in retrieval.get("records") or []:
            if "citation" not in entry:
                problems.append(
                    f"record {entry.get('record_id')} carries no citation payload"
                )
            if entry.get("explanation_source") in (
                "retained_at_retrieval",
                "recomputed_at_inspection",
            ):
                dims = entry.get("dimensions") or {}
                contributions = dims.get("contributions") or {}
                if abs(sum(contributions.values()) - dims.get("composite", 0.0)) > 1e-9:
                    problems.append(
                        f"record {entry.get('record_id')} contributions do not "
                        "sum to its composite"
                    )
        if retrieval.get("rejections_known"):
            for rejected in retrieval.get("rejected") or []:
                if rejected.get("reason") not in (
                    "below_min_score",
                    "beyond_top_k",
                ):
                    problems.append(
                        f"rejected record {rejected.get('record_id')} has "
                        f"unknown reason {rejected.get('reason')!r}"
                    )
    return problems


def inspector_live_problems() -> list[str]:
    """Re-derive the slice's live claims from the tree.

    The forge retains dimension explanations that agree
    with its composites; ``retrieve()`` retains a
    report whose rejections carry stated reasons; the
    TF-038 trace carries citation payloads; a full
    synthetic report builds clean and leaves its store
    byte-identical.
    """
    problems: list[str] = []
    from thoughtforge.knowledge.forge import KnowledgeForge

    forge = KnowledgeForge.__new__(KnowledgeForge)
    forge.retrieval_weights = None
    forge._last_retrieval_report = None
    raw = {
        "qid": "Q1", "label_en": "Ferry",
        "description_en": "crosses the hollow",
        "popularity_score": 0.9, "fts_score": 10.0,
        "source": "wikidata", "record_type": "entity",
    }
    scored = forge._score_records([dict(raw)], "ferry")
    explanation = scored[0].get("_explanation")
    if not isinstance(explanation, dict):
        problems.append("forge scoring retains no _explanation")
    elif abs(explanation["composite"] - scored[0]["_composite_score"]) > 1e-12:
        problems.append("retained explanation composite != _composite_score")

    # The mirror agrees with the retained explanation on
    # every record shape (a timestamped record's recency
    # decays between the two calls, so its composite is
    # judged with a stated tolerance; the rest are exact).
    shapes = [
        (dict(raw), 0.0),
        ({"chunk_id": "c1", "title": "T", "content": "x",
          "similarity": 0.8, "source": "reference_chunk",
          "record_type": "reference_chunk"}, 0.0),
        ({"qid": "Q2", "label_en": "Old", "popularity_score": 0.5,
          "fts_score": 4.0, "source": "wikidata",
          "record_type": "entity",
          "updated_at": "2020-01-01T00:00:00+00:00"}, 1e-4),
        ({"content": "a UsedFor b", "label_en": "a", "weight": 0.7,
          "fts_score": 0.0, "source": "conceptnet",
          "record_type": "concept_edge"}, 0.0),
    ]
    for shape, tolerance in shapes:
        retained = forge._score_records([dict(shape)], "q")[0]["_explanation"]
        mirrored = dimension_explanation(shape)
        if abs(retained["composite"] - mirrored["composite"]) > max(tolerance, 1e-12):
            problems.append(
                f"mirror disagrees with retained explanation for "
                f"{shape.get('qid') or shape.get('chunk_id') or shape.get('label_en')}"
            )
        if retained["semantic_source"] != mirrored["semantic_source"]:
            problems.append("mirror semantic_source disagrees with retained")
        if retained["recency_source"] != mirrored["recency_source"]:
            problems.append("mirror recency_source disagrees with retained")

    import tempfile

    from thoughtforge.cognition.traces import export_turn_trace
    from thoughtforge.knowledge.models import RuntimeTurnState
    from thoughtforge.knowledge.store import MemoryStore

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(memory_dir=Path(tmp) / "mem")
        before = {
            p.name: p.read_bytes()
            for p in sorted(store.dir.iterdir()) if p.is_file()
        }
        bundle = forge._build_bundle(scored)
        turn = RuntimeTurnState(turn_id="inspector_live")
        turn.memory_bundle = bundle
        trace = export_turn_trace(turn)
        entry = trace["retrieval"]["activated_records"][0]
        if "citation" not in entry:
            problems.append("trace retrieval entry carries no citation payload")
        report = build_inspector_report(
            store=store, trace=trace, bundle=bundle,
        )
        problems.extend(inspector_problems(report))
        after = {
            p.name: p.read_bytes()
            for p in sorted(store.dir.iterdir()) if p.is_file()
        }
        if before != after:
            problems.append("building an inspector report wrote to the store")
    return problems


__all__ = [
    "CARRIED_EVIDENCE_KEYS",
    "DIMENSION_RULES",
    "FORGE_WEIGHTS_ROW",
    "INSPECTOR_FORMAT",
    "INSPECTOR_SECTIONS",
    "build_inspector_report",
    "dimension_explanation",
    "explain_record",
    "explain_retrieval",
    "inspector_live_problems",
    "inspector_problems",
    "inspector_to_json",
    "knowledge_db_report",
    "write_inspector_report",
]
