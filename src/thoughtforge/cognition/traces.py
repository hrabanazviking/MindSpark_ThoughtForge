"""Logging & traces — a turn's full cognition trace, exportable (TF-038).

TF-025 retained the finished turn on the core
(``ThoughtForgeCore.last_turn_state``) and closed the state half
of CORE_LOOP §8-F3; this slice is the export half F3 assigned
here: *one turn, whole, as data a caller can keep*.

**The trace** (:func:`export_turn_trace`) is a JSON-safe dict in
one named format (``TRACE_FORMAT``) with named sections
(``TRACE_SECTIONS``, in pipeline order):

  * ``turn`` — id, phase, completion / interruption state,
    resume and repair flags, and the turn's timestamps;
  * ``conversation`` — the core-level context the turn ran under
    (selected profile / strategy / depth / candidate modes,
    determinism seed, persona id) when exported through the core;
    ``None`` for a bare turn-state export, because a turn state
    does not carry it;
  * ``route`` — the input sketch the router produced;
  * ``retrieval`` — the memory bundle: which records fired, with
    scores and cues, and (TF-061, TF-048's assignment landed)
    each record's citation payload; asking the *store* why a
    record fired is TF-061's inspector data
    (``knowledge/inspector.py``), which consumes this trace;
  * ``scaffold`` — goal / tone / focus / avoid / depth / modes /
    fact_block, exactly as built;
  * ``candidates`` — every candidate with its full scores,
    composite included;
  * ``fragments`` — every salvaged fragment, its source candidate,
    scores, and the keep verdict;
  * ``generation_failures`` — TF-031's failure records, so a
    degraded turn's trace says what failed instead of the log
    being the only witness;
  * ``final`` — the final response record whole: text, mode,
    salvage path, citations, enforcement verdict, scores with
    composite and quality tier;
  * ``writeback`` — the writeback record when one exists
    (usually ``None`` today; TF-047 owns the write-back leg);
  * ``timings`` — retrieval / generation / salvage milliseconds
    and the turn's token total.

**Trace ≠ fingerprint.** TF-036's ``turn_fingerprint`` answers
"would this run reproduce?" and therefore *excludes* timestamps
and timings by ruling and *refuses* an unfinished turn. A trace
answers "what happened in this turn?" — so it *includes*
timestamps and timings (they are part of what happened), and it
exports a turn at any phase: the trace of an interrupted turn
(TF-027) is exactly the record a resumer or a human inspector
needs. The two surfaces share no code and neither replaces the
other.

**Detached and JSON-safe.** The export is built field by field
into fresh containers — mutating an exported trace never touches
the live turn state — and every value passes through a JSON-safe
conversion, so ``json.dumps`` on a trace cannot fail on a store
record's ``raw`` payload or a writeback dict. The one conversion
rule: a value with no JSON shape becomes its ``str()`` — visible
in the trace, never silently dropped.

**A checker, not a hope.** :func:`trace_problems` judges an
exported trace as data: format, sections present, phase known,
a completed turn carrying its final, the final's turn id agreeing,
and every cross-reference resolving (the final's source candidate
/ fragment ids and each fragment's source candidate must exist in
the trace). A trace that fails it is not a trace of the turn it
claims to be.

**Logging's half.** The tree's logging surface is
``thoughtforge.utils.logging_setup.setup_logging`` — called by
both entry points (``forge_memory.py`` at import, and
``run_thoughtforge.py`` when its own debug setup has not run) and
by nobody else; the cognition modules log through the standard
``logging`` module under their own names and never configure.
Before this slice ``logging_setup`` had no direct test import
(TF-016's D-COV-3); this slice pins its behaviour directly
(one-shot configuration, level / console / file switches, the
rotating file it creates) and narrows D-COV-3 to its ``paths``
share, which stays TF-069's. The traces themselves do not go
*through* logging: a trace is a returned value, not a log line —
consumers (TF-061's inspector data, TF-084's joined inference
traces, Part IX's cognition API) take it as data.

Proof grade: the export is proven on real ``think()`` /
``think_stream()`` turns run on this VM — knowledge-only,
stub-backend (candidates + fragments + salvage path), stated
failure (TF-031 records riding the trace), and an interrupted
turn — plus hand-built turn states for the sections a live empty-
store turn cannot reach (a populated bundle, a writeback record).
No live model was traced; a live model's turn differs only in who
wrote the candidates, not in the state's shape.

The human-readable write-up is
``docs/architecture/LOGGING_AND_TRACES.md``.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from thoughtforge.cognition.turn_state import PHASE_ORDER
from thoughtforge.knowledge.models import RuntimeTurnState

#: The one trace format this module exports. A consumer that
#: meets another value is looking at another format (or a
#: corruption), never at a newer trace it should guess at —
#: TF-162 owns versioned refusal at the service boundary.
TRACE_FORMAT = "thoughtforge-turn-trace/1"

#: The trace's sections, in pipeline order. ``export_turn_trace``
#: emits exactly these keys (plus ``format`` and ``turn_id``), and
#: ``trace_problems`` requires every one of them present.
TRACE_SECTIONS: tuple[str, ...] = (
    "turn",
    "conversation",
    "route",
    "retrieval",
    "scaffold",
    "candidates",
    "fragments",
    "generation_failures",
    "final",
    "writeback",
    "timings",
)


# ── JSON safety ───────────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """``value`` as JSON-safe data, in fresh containers.

    Primitives pass through; dicts / lists / tuples / sets are
    rebuilt (dict keys stringified); dataclass instances become
    their field dict; anything else becomes ``str(value)`` —
    visible in the trace, never silently dropped, and never the
    live object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(v) for v in sorted(value, key=repr)]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return str(value)


# ── Section builders ──────────────────────────────────────────────────────────


def _scores_dict(scores: Any, *, composite: bool = False,
                 quality_tier: bool = False) -> dict[str, Any] | None:
    if scores is None:
        return None
    out = {
        name: _json_safe(getattr(scores, name))
        for name in scores.__dataclass_fields__
    }
    if composite:
        out["composite"] = scores.composite
    if quality_tier:
        out["quality_tier"] = scores.quality_tier
    return out


def _route_section(turn: RuntimeTurnState) -> dict[str, Any] | None:
    sketch = turn.input_sketch
    if sketch is None:
        return None
    return {
        "raw_text": sketch.raw_text,
        "intent": sketch.intent,
        "topic": sketch.topic,
        "tone_in": sketch.tone_in,
        "response_mode": sketch.response_mode,
        "memory_triggers": list(sketch.memory_triggers),
        "urgency": sketch.urgency,
        "retrieval_path": sketch.retrieval_path,
    }


def _retrieval_section(turn: RuntimeTurnState) -> dict[str, Any] | None:
    bundle = turn.memory_bundle
    if bundle is None:
        return None
    return {
        "personality_core_id": bundle.personality_core_id,
        "total_retrieved": bundle.total_retrieved,
        "retrieval_confidence": bundle.retrieval_confidence,
        "activated_records": [
            {
                "record_id": rec.record_id,
                "record_type": rec.record_type,
                "score": rec.score,
                "cue": rec.cue,
                # TF-061 (TF-048's assignment): the record's
                # citation payload rides the trace — before
                # this, an inspector reading the trace could
                # not say what marker a retrieved item was
                # citable by, though the record carried its
                # payload by construction.
                "citation": (
                    _json_safe(rec.citation.to_dict())
                    if getattr(rec, "citation", None) is not None
                    else None
                ),
                "raw": _json_safe(rec.raw),
            }
            for rec in bundle.activated_records
        ],
    }


def _scaffold_section(turn: RuntimeTurnState) -> dict[str, Any] | None:
    scaffold = turn.scaffold
    if scaffold is None:
        return None
    return {
        "goal": scaffold.goal,
        "tone": list(scaffold.tone),
        "focus": list(scaffold.focus),
        "avoid": list(scaffold.avoid),
        "depth": scaffold.depth,
        "candidate_modes": list(scaffold.candidate_modes),
        "fact_block": scaffold.fact_block,
    }


def _candidate_section(turn: RuntimeTurnState) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": cand.candidate_id,
            "mode": cand.mode,
            "text": cand.text,
            "token_estimate": cand.token_estimate,
            "scores": _scores_dict(cand.scores, composite=True),
        }
        for cand in turn.candidates
    ]


def _fragment_section(turn: RuntimeTurnState) -> list[dict[str, Any]]:
    return [
        {
            "fragment_id": frag.fragment_id,
            "source_candidate_id": frag.source_candidate_id,
            "text": frag.text,
            "position": frag.position,
            "keep": frag.keep,
            "scores": _scores_dict(frag.scores, composite=True),
        }
        for frag in turn.fragments
    ]


def _failure_section(turn: RuntimeTurnState) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for failure in turn.generation_failures:
        if isinstance(failure, dict):
            out.append(_json_safe(failure))
        else:
            out.append(
                {
                    "stage": getattr(failure, "stage", ""),
                    "mode": getattr(failure, "mode", ""),
                    "detail": getattr(failure, "detail", ""),
                }
            )
    return out


def _final_section(turn: RuntimeTurnState) -> dict[str, Any] | None:
    final = turn.final_response
    if final is None:
        return None
    return {
        "response_id": final.response_id,
        "turn_id": final.turn_id,
        "text": final.text,
        "mode": final.mode,
        "salvage_path": final.salvage_path,
        "source_candidate_ids": list(final.source_candidate_ids),
        "source_fragment_ids": list(final.source_fragment_ids),
        "citations": list(final.citations),
        "enforcement_passed": final.enforcement_passed,
        "enforcement_notes": final.enforcement_notes,
        "token_count": final.token_count,
        "retrieval_confidence": final.retrieval_confidence,
        "created_at": final.created_at,
        "scores": _scores_dict(
            final.scores, composite=True, quality_tier=True
        ),
    }


# ── The export ────────────────────────────────────────────────────────────────


def export_turn_trace(
    turn: RuntimeTurnState,
    conversation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One turn's full cognition trace, as detached JSON-safe data.

    ``turn`` is the retained ``RuntimeTurnState`` (TF-025) at any
    phase — finished, failed, or interrupted; an unfinished turn
    exports with ``final`` None and its phase stated, never with an
    invented completion. ``conversation`` is the core-level context
    (TF-038: profile / strategy / depth / modes / seed / persona)
    that a bare turn state does not carry; the core's
    ``last_turn_trace()`` supplies it, and a direct caller may pass
    their own or leave it None.
    """
    return {
        "format": TRACE_FORMAT,
        "turn_id": turn.turn_id,
        "turn": {
            "phase": turn.phase,
            "is_complete": turn.is_complete,
            "is_interrupted": turn.is_interrupted,
            "interrupted": turn.interrupted,
            "interrupt_reason": turn.interrupt_reason,
            "interrupted_phase": turn.interrupted_phase,
            "resume_count": turn.resume_count,
            "repair_attempted": turn.repair_attempted,
            "started_at": turn.started_at,
            "completed_at": turn.completed_at,
        },
        "conversation": _json_safe(conversation),
        "route": _route_section(turn),
        "retrieval": _retrieval_section(turn),
        "scaffold": _scaffold_section(turn),
        "candidates": _candidate_section(turn),
        "fragments": _fragment_section(turn),
        "generation_failures": _failure_section(turn),
        "final": _final_section(turn),
        "writeback": _json_safe(turn.writeback),
        "timings": {
            "retrieval_ms": turn.retrieval_ms,
            "generation_ms": turn.generation_ms,
            "salvage_ms": turn.salvage_ms,
            "total_tokens_used": turn.total_tokens_used,
        },
    }


def trace_to_json(trace: dict[str, Any]) -> str:
    """The trace as one JSON document (indented, newline-ended)."""
    return json.dumps(trace, indent=2, ensure_ascii=False) + "\n"


def write_turn_trace(trace: dict[str, Any], path: str | Path) -> Path:
    """Write ``trace`` to ``path`` as JSON; returns the path.

    Parent directories are created — an inspector writing into a
    fresh export directory should not have to pre-make it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(trace_to_json(trace), encoding="utf-8")
    return target


# ── The checker ───────────────────────────────────────────────────────────────


def trace_problems(trace: Any) -> list[str]:
    """Problems with an exported trace, as data ([] = sound).

    Judges the trace as a self-contained record: the format is
    this module's, every section is present, the phase is a known
    turn phase, a completed turn carries its final, the final's
    turn id agrees with the trace's, no candidate or fragment id
    is duplicated, every cross-reference resolves inside the
    trace, and no timing is negative. Content quality — whether
    the answer was *good* — is the scorers' business, not this
    checker's.
    """
    if not isinstance(trace, dict):
        return ["trace is not a mapping"]
    problems: list[str] = []
    if trace.get("format") != TRACE_FORMAT:
        problems.append(
            f"trace format is {trace.get('format')!r}, not {TRACE_FORMAT!r}"
        )
    for section in TRACE_SECTIONS:
        if section not in trace:
            problems.append(f"trace is missing section {section!r}")
    turn_id = trace.get("turn_id")
    if not turn_id:
        problems.append("trace has no turn_id")

    turn_section = trace.get("turn") or {}
    phase = turn_section.get("phase")
    if phase is not None and phase not in PHASE_ORDER:
        problems.append(f"turn phase {phase!r} is not a known turn phase")

    candidates = trace.get("candidates") or []
    fragments = trace.get("fragments") or []
    candidate_ids = [c.get("candidate_id") for c in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        problems.append("trace carries a duplicated candidate_id")
    fragment_ids = [f.get("fragment_id") for f in fragments]
    if len(set(fragment_ids)) != len(fragment_ids):
        problems.append("trace carries a duplicated fragment_id")
    for frag in fragments:
        source = frag.get("source_candidate_id")
        if source and source not in candidate_ids:
            problems.append(
                f"fragment {frag.get('fragment_id')!r} names source "
                f"candidate {source!r} absent from the trace"
            )

    final = trace.get("final")
    if turn_section.get("is_complete") and final is None:
        problems.append("turn is marked complete but carries no final")
    if final is not None:
        final_turn = final.get("turn_id")
        if final_turn and turn_id and final_turn != turn_id:
            problems.append(
                f"final turn_id {final_turn!r} disagrees with trace "
                f"turn_id {turn_id!r}"
            )
        for cid in final.get("source_candidate_ids") or []:
            if cid not in candidate_ids:
                problems.append(
                    f"final names source candidate {cid!r} absent "
                    "from the trace"
                )
        for fid in final.get("source_fragment_ids") or []:
            if fid not in fragment_ids:
                problems.append(
                    f"final names source fragment {fid!r} absent "
                    "from the trace"
                )

    timings = trace.get("timings") or {}
    for key, value in timings.items():
        if isinstance(value, (int, float)) and value < 0:
            problems.append(f"timing {key!r} is negative: {value}")
    return problems
