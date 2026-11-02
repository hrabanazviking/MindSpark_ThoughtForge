"""
Small-model honesty panel, as data (TF-126).

The roadmap's rule: *small-model honesty panel: the
same suites across model sizes — where the forge
helps most, published from results.* TF-122 built
the persona suite, TF-123 the memory suite, TF-124
the depth suite, TF-125 the presence suite — each
with a pinned harness baseline over its scripted
corpus, and each filing the same limit in its own
doc: a live model's behaviour is NOT PROVEN, and
running the suites *across model sizes* was
TF-126's road. No TF-016 entry ever numbered that
gap — TF-122..TF-125 each assigned it to TF-126 in
their filings without numbering it (the
D-BENCH-9 / D-BENCH-10 / D-BENCH-11 pattern). Filed
as D-BENCH-12 and fixed by this slice in the
measuring sense (see below).

The measured starting state, probed live this tick
on this tree before anything was built:

  * **Four suites, four pinned baselines, and
    nothing across sizes.** ``benchmarks/`` held
    exactly the four suite modules (persona,
    memory, depth, presence), each with its pinned
    harness baseline re-measured live this tick —
    persona overall 0.8453703703703704, memory
    overall 0.8055555555555557, depth OFF
    0.47222222222222227 / ON 0.8055555555555557,
    presence overall 0.7777777777777778 — and no
    module, panel or row anywhere in the estate
    had ever run any of them at more than one
    model size (grepped).
  * **No live model can run on this VM** — no GGUF
    in the tree and no ``llama_cpp`` (TF-078's
    filing, re-probed this tick). Live per-size
    behaviour therefore stays NOT PROVEN here, and
    this panel does not pretend otherwise: it
    measures *filed size-tier degradations* of the
    suites' own scripted corpora through the
    suites' REAL scorers / runners — TF-067's
    fixture-panel precedent (a published curve
    filed as literal data, recomputed live by its
    checker, over scripted fixtures through the
    real scorers). The tier a row names is a
    degradation profile anchored to a real
    catalogue model, never a claim that that model
    produced these answers.
  * **Every suite runner accepts an explicit
    corpus** — probed live: ``run_persona_suite(
    identity, voice, pressure)``,
    ``run_memory_suite(present, true, explain)``,
    ``run_depth_suite(chain, plan, check)``,
    ``run_presence_suite(coherence, steering,
    closure)``. The panel reuses those runners and
    the suites' corpora exactly as filed; it copies
    no corpus and no scorer.
  * **The tier anchors are real catalogue models.**
    ``ModelBrowser.CURATED_MODELS`` carries all
    four anchor filenames (probed live), and
    ``params_from_name`` returns 14.0 / 8.0 / 3.0
    for the first three anchors and None for the
    TinyLlama filename — its ``1.1b`` token is
    lower-case, which that reader does not take;
    the tiny tier's params are therefore filed
    from the filename's own ``1.1b`` token,
    checker-verified by regex, never invented.

What this module files, as verifiable data:

  * ``MODEL_TIERS`` — four size tiers, each
    anchored to a live catalogue filename, each
    filing its degradation as a list of transform
    ids and a coherence drift shift: the reference
    tier degrades nothing; smaller tiers drop
    citations, answer tersely, close generically
    and swap facts, in the filed order.
  * ``TRANSFORMS`` — the four degradation
    transforms as filed data over pure text
    functions (citation drop, terse first
    sentence, generic closer, fact swap).
  * The forge-repair roads, one per suite, using
    real machinery: memory's citation repair
    re-attaches the first filed record's real
    TF-048 marker; persona's generic strip removes
    sentences carrying a real generic phrase; the
    presence will repair replays each saga through
    TF-029's real ``GoalLedger`` and prepends the
    standing-goal line where the degraded answer
    dropped it; depth's ON road is the suite's own
    native scaffold road, degraded by the same
    tier. The coherence axis files NO repair —
    ON == degraded there, stated as a rule.
  * ``PANEL_ROWS`` — the published panel as
    literal data: 16 cells (4 tiers x 4 suites) of
    OFF (degraded, forge off) / ON (degraded +
    repaired, forge on) / delta, plus per-tier
    means and the ``FINDING`` (where the forge
    helps most), all recomputed live by the
    checker, TF-067's CURVE precedent. The numbers
    are filed *whatever they are*: the forge helps
    most at the MID tier overall, not the
    smallest; depth's delta goes negative at the
    small tier; several honest shapes like that
    are filed by name in the doc, never tuned
    away.

The panel verdict is filed plainly: PASS only
when the reference tier's OFF scores reproduce
the four pinned suite baselines EXACTLY (and the
depth reference ON reproduces its undegraded ON)
and all 16 cells are measured. An empty panel
scores 0.0 / REVIEW — no vacuous pass.

D-BENCH-12 is FIXED by this slice in the measuring
sense: the suites now run across filed model-size
tiers with a published, recomputed panel. What is
NOT fixed, and is stated in the filed limits:
live per-size behaviour of real models is NOT
PROVEN — no GGUF / llama_cpp on this VM (TF-078);
TF-173 owns the real-hardware road.

Filed limits, stated plainly:

  * The tiers are degradation *profiles*, not
    model runs: a tier says "answers degraded this
    way, as a smaller model degrades them in the
    filed shapes", scored through the real suite
    machinery over scripted synthetic corpora
    (TF-017; ``data/`` never opened). No row is a
    measurement of the anchor model itself.
  * The transforms are the filed degradation
    shapes only — citation loss, terseness, a
    generic closer, confident fact swaps, earlier
    coherence drift. A real small model degrades
    in more ways; those are not filed and not
    claimed.
  * Memory's citation repair applies to the two
    citation axes (present / true) only: TF-123's
    explain axis holds exactly when a non-memory
    answer carries NO citation, so appending one
    there would break the suite's own
    cite-or-explain discipline — explain cases run
    degraded-only on both roads, filed as a rule.
    The marker is appended into the answer's final
    sentence (before its terminal period), in the
    suite's own bracketed format, because TF-049
    distinguishes a citation in the claim sentence
    from one filed somewhere else (TF-123's p12).
  * Persona's ON at the reference tier is above
    its OFF even undegraded, honestly: the generic
    strip also removes the voice corpus's six
    deliberately generic turns' generic sentences.
    That is the filed repair acting on the filed
    corpus, stated here, not hidden.

The human-readable write-up is
``docs/architecture/HONESTY_PANEL.md``.
Proof grade: FIXTURE — synthetic scripted corpora
degraded by filed transforms and scored through
the real suite machinery on this VM; no live
model, never claimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

HONESTY_PANEL_FORMAT = "thoughtforge-honesty-panel/1"
HONESTY_PANEL_DOC = "docs/architecture/HONESTY_PANEL.md"

#: The suites the panel runs, in panel order.
PANEL_SUITES: tuple[str, ...] = (
    "persona", "memory", "depth", "presence")

#: The reference tier's pinned suite baselines,
#: reproduced EXACTLY by the reference OFF cells
#: (and, for depth, the reference ON cell) — the
#: harness's own pins, never re-measured into new
#: values here.
REFERENCE_BASELINES: dict[str, float] = {
    "persona_off": 0.8453703703703704,
    "memory_off": 0.8055555555555557,
    "depth_off": 0.47222222222222227,
    "depth_on": 0.8055555555555557,
    "presence_off": 0.7777777777777778,
}

# The real machinery, reused — never copied: the
# four suites and their corpora + runners, the
# persona scorer's generic-phrase vocabulary, the
# model catalogue + name reader the tiers anchor
# to, TF-048's citation payloads via the memory
# suite's own bindings, and TF-029's ledger.
from benchmarks import depth_suite as _depth  # noqa: E402
from benchmarks import memory_suite as _memory  # noqa: E402
from benchmarks import persona_suite as _persona  # noqa: E402
from benchmarks import presence_suite as _presence  # noqa: E402
from benchmarks.persona_consistency import (  # noqa: E402
    GENERIC_PHRASES,
)
from thoughtforge.cognition.will import GoalLedger  # noqa: E402
from thoughtforge.inference.model_browser import (  # noqa: E402
    ModelBrowser,
)
from thoughtforge.inference.profile_runner import (  # noqa: E402
    params_from_name,
)
from thoughtforge.knowledge.models import InputSketch  # noqa: E402

#: The generic closer's text + the GENERIC_PHRASES
#: member it carries verbatim (checker-verified
#: live against benchmarks.persona_consistency).
GENERIC_CLOSER_TEXT = " As an AI, I cannot provide more detail."
GENERIC_CLOSER_PHRASE = "as an ai"

#: The fact swap's filed pairs (TF-067's
#: confident-contradiction shape: a fluent answer
#: with the filed fact quietly wrong).
FACT_SWAPS: tuple[tuple[str, str], ...] = (
    ("dawn", "midnight"),
    ("eleven", "twelve"),
    ("twice", "once"),
)


# ── The degradation transforms ──────────────────────────────────────────────

def _citation_drop(text: str) -> str:
    """Strip bracketed citation markers + QID tokens."""
    text = re.sub(r"\[[^\]]{1,48}\]", "", text)
    return re.sub(r"\bQ\d{4,}\b", "", text)


def _terse_first_sentence(text: str) -> str:
    """Keep only the first sentence (whole if none)."""
    match = re.search(r"[.!?]", text)
    return text[: match.end()] if match else text


def _generic_closer(text: str) -> str:
    """Append the filed generic closer sentence."""
    return text + GENERIC_CLOSER_TEXT


def _fact_swap(text: str) -> str:
    """Swap the filed facts, word-boundary, case-kept."""
    for source, target in FACT_SWAPS:
        def _repl(match: re.Match[str],
                  target: str = target) -> str:
            word = match.group(0)
            if word.isupper():
                return target.upper()
            if word[:1].isupper():
                return target.capitalize()
            return target
        text = re.sub(rf"\b{source}\b", _repl, text,
                      flags=re.IGNORECASE)
    return text


TRANSFORM_FUNCS: dict[str, Any] = {
    "citation_drop": _citation_drop,
    "terse_first_sentence": _terse_first_sentence,
    "generic_closer": _generic_closer,
    "fact_swap": _fact_swap,
}

TRANSFORMS: tuple[dict[str, Any], ...] = (
    {"id": "citation_drop",
     "does": "strip bracketed citation markers "
             "(regex \\[[^\\]]{1,48}\\]) and standalone "
             "QID tokens (\\bQ\\d{4,}\\b) from the "
             "answer text"},
    {"id": "terse_first_sentence",
     "does": "keep only up to and including the "
             "first sentence-ending . ! or ? (the "
             "whole text when it has none)"},
    {"id": "generic_closer",
     "does": "append the filed generic closer "
             "sentence, which carries the "
             "GENERIC_PHRASES member 'as an ai' "
             "verbatim"},
    {"id": "fact_swap",
     "does": "swap the filed fact words "
             "dawn->midnight, eleven->twelve, "
             "twice->once (case-insensitive, "
             "word-boundary, case preserved)"},
)


def apply_transforms(text: str,
                     transform_ids: Any) -> str:
    """Apply a tier's filed transforms, in order."""
    for transform_id in transform_ids:
        text = TRANSFORM_FUNCS[transform_id](text)
    return text


# ── The model tiers ─────────────────────────────────────────────────────────
#
# Four tiers anchored LIVE to the catalogue: each
# anchor filename must exist in
# ModelBrowser.CURATED_MODELS (checker-verified).
# Degradation grows as size falls, in the filed
# order; the coherence drift shift moves a saga's
# drift onset earlier by the filed count.

MODEL_TIERS: tuple[dict[str, Any], ...] = (
    {"id": "reference_14b", "prefix": "reference",
     "anchor": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
     "params": 14.0, "params_basis": "params_from_name",
     "transforms": (), "drift_shift": 0},
    {"id": "mid_8b", "prefix": "mid",
     "anchor": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "params": 8.0, "params_basis": "params_from_name",
     "transforms": ("citation_drop",), "drift_shift": 0},
    {"id": "small_3b", "prefix": "small",
     "anchor": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "params": 3.0, "params_basis": "params_from_name",
     "transforms": ("citation_drop", "terse_first_sentence"),
     "drift_shift": 1},
    {"id": "tiny_1_1b", "prefix": "tiny",
     "anchor": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "params": 1.1, "params_basis": "filename_token",
     "transforms": ("citation_drop", "terse_first_sentence",
                    "generic_closer", "fact_swap"),
     "drift_shift": 2},
)

TIER_BY_ID: dict[str, dict[str, Any]] = {
    t["id"]: t for t in MODEL_TIERS}


# ── Degradation adapters (one per suite) ────────────────────────────────────
#
# Each adapter degrades a suite's OWN filed corpus
# by the tier's transforms, field by field as the
# task files: answer texts only — questions, filed
# expectations and scoring inputs are never touched.

def degraded_persona_corpora(
    tier: dict[str, Any],
) -> tuple[list, list, list]:
    """Persona corpora degraded: identity/pressure
    dicts' ``text``, voice strings."""
    ids = tier["transforms"]
    identity = [{**c, "text": apply_transforms(c["text"], ids)}
                for c in _persona.IDENTITY_TURNS]
    voice = [apply_transforms(v, ids) for v in _persona.VOICE_TURNS]
    pressure = [{**c, "text": apply_transforms(c["text"], ids)}
                for c in _persona.PRESSURE_TURNS]
    return identity, voice, pressure


def degraded_memory_corpora(
    tier: dict[str, Any],
) -> tuple[list, list, list]:
    """Memory corpora degraded: every case's ``text``."""
    ids = tier["transforms"]

    def _deg(cases: Any) -> list:
        return [{**c, "text": apply_transforms(c["text"], ids)}
                for c in cases]

    return (_deg(_memory.PRESENT_CASES), _deg(_memory.TRUE_CASES),
            _deg(_memory.EXPLAIN_CASES))


def degraded_depth_corpora(
    tier: dict[str, Any],
) -> tuple[list, list, list]:
    """Depth corpora degraded: ``off_answer``,
    ``working`` and ``on_answer`` of every case."""
    ids = tier["transforms"]

    def _deg(cases: Any) -> list:
        return [{**c,
                 "off_answer": apply_transforms(c["off_answer"], ids),
                 "working": apply_transforms(c["working"], ids),
                 "on_answer": apply_transforms(c["on_answer"], ids)}
                for c in cases]

    return (_deg(_depth.CHAIN_CASES), _deg(_depth.PLAN_CASES),
            _deg(_depth.CHECK_CASES))


def degraded_presence_corpora(
    tier: dict[str, Any],
) -> tuple[list, list, list]:
    """Presence corpora degraded: steering/closure
    turns' ``assistant`` texts; coherence sagas are
    shapes, not text — their tier degradation is the
    filed drift shift (the tier's ``drift_shift``
    moves ``drift_onset`` earlier, clamped at 2;
    onset None stays None; ``drift_turns`` and
    ``upto`` unchanged)."""
    ids = tier["transforms"]
    coherence: list = []
    for saga in _presence.COHERENCE_SAGAS:
        degraded = dict(saga)
        if saga["drift_onset"] is not None:
            degraded["drift_onset"] = max(
                2, saga["drift_onset"] - tier["drift_shift"])
        coherence.append(degraded)

    def _deg(sagas: Any) -> list:
        return [{**s, "turns": tuple(
            {**t, "assistant": apply_transforms(t["assistant"], ids)}
            for t in s["turns"])} for s in sagas]

    return (coherence, _deg(_presence.STEERING_SAGAS),
            _deg(_presence.CLOSURE_SAGAS))


# ── The forge-repair roads (one per suite) ──────────────────────────────────

def _strip_generic_sentences(text: str) -> str:
    """Remove any sentence carrying a GENERIC_PHRASES
    member (the persona repair; the scorer's own
    vocabulary, reused)."""
    if not text.strip():
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences
            if not any(p in s.lower() for p in GENERIC_PHRASES)]
    return " ".join(kept)


def repaired_persona_corpora(
    corpora: tuple[list, list, list],
) -> tuple[list, list, list]:
    """Persona ON: the generic strip over the
    degraded identity/pressure texts + voice strings."""
    identity, voice, pressure = corpora
    return (
        [{**c, "text": _strip_generic_sentences(c["text"])}
         for c in identity],
        [_strip_generic_sentences(v) for v in voice],
        [{**c, "text": _strip_generic_sentences(c["text"])}
         for c in pressure],
    )


def repair_memory_case(case: dict[str, Any]) -> dict[str, Any]:
    """Memory ON for one citation-axis case: when the
    (degraded) text carries no citation marker,
    re-attach the marker of the case's first filed
    record — the record's real TF-048 payload via
    ``citation_for_record`` — in the suite's own
    bracketed format, appended into the final
    sentence before its terminal period (TF-049's
    in-claim discipline; a marker after the period
    would file in its own sentence and score
    nowhere). A case already carrying a marker, or
    filing no record, is returned unchanged."""
    text = case["text"]
    if _memory.cited_markers_in(text) or not case.get("records"):
        return case
    marker = _memory.citation_for_record(
        _memory.RECORDS[case["records"][0]]).marker
    stripped = text.rstrip()
    if stripped.endswith("."):
        text = stripped[:-1] + f" [{marker}]."
    else:
        text = stripped + f" [{marker}]"
    return {**case, "text": text}


def repaired_memory_corpora(
    corpora: tuple[list, list, list],
) -> tuple[list, list, list]:
    """Memory ON: citation repair over the present +
    true axes. The explain axis runs degraded-only
    on both roads (filed rule: its holding condition
    is carrying NO citation, TF-123's cite-or-
    explain discipline)."""
    present, true_cases, explain = corpora
    return (
        [repair_memory_case(c) for c in present],
        [repair_memory_case(c) for c in true_cases],
        explain,
    )


def repair_will_sagas(sagas: list) -> list:
    """Presence ON for one will axis: replay the
    saga's filed user turns through a fresh real
    GoalLedger exactly as ``judge_will_saga`` does;
    on any turn where the ledger's will_line stands
    and the degraded answer lacks the standing
    statement, prepend ``Standing goal: <statement>. ``
    — TF-029's inject_will shape."""
    repaired: list = []
    for saga in sagas:
        ledger = GoalLedger()
        turns: list = []
        for turn in saga["turns"]:
            sketch = InputSketch(
                raw_text=turn["user"], intent=turn["intent"],
                topic=turn["topic"], tone_in="neutral",
                response_mode="conversational")
            ledger.observe_turn(sketch, _presence.RESPONSE_GOAL)
            answer = turn["assistant"]
            line = ledger.will_line(_presence.RESPONSE_GOAL)
            if line is not None and ledger.active is not None:
                if not _presence._contains(
                        _presence._norm(answer),
                        ledger.active.statement):
                    answer = (f"Standing goal: "
                              f"{ledger.active.statement}. " + answer)
            turns.append({**turn, "assistant": answer})
        repaired.append({**saga, "turns": tuple(turns)})
    return repaired


def repaired_presence_corpora(
    corpora: tuple[list, list, list],
) -> tuple[list, list, list]:
    """Presence ON: will repair over both will axes.
    The coherence axis has NO repair — ON is the
    degraded corpus itself, filed as a rule."""
    coherence, steering, closure = corpora
    return (coherence, repair_will_sagas(steering),
            repair_will_sagas(closure))


# ── The panel measurement ───────────────────────────────────────────────────

def measure_cell(tier: dict[str, Any],
                 suite_id: str) -> dict[str, Any]:
    """One panel cell, measured live through the
    suite's REAL runner over the tier-degraded
    corpus (OFF) and the degraded + repaired corpus
    (ON). Depth's ON is its native scaffold road."""
    if suite_id == "persona":
        degraded = degraded_persona_corpora(tier)
        off = _persona.run_persona_suite(*degraded).overall_score
        on = _persona.run_persona_suite(
            *repaired_persona_corpora(degraded)).overall_score
    elif suite_id == "memory":
        degraded_m = degraded_memory_corpora(tier)
        off = _memory.run_memory_suite(*degraded_m).overall_score
        on = _memory.run_memory_suite(
            *repaired_memory_corpora(degraded_m)).overall_score
    elif suite_id == "depth":
        result = _depth.run_depth_suite(
            *degraded_depth_corpora(tier))
        off, on = result.overall_off, result.overall_on
    elif suite_id == "presence":
        degraded_p = degraded_presence_corpora(tier)
        off = _presence.run_presence_suite(*degraded_p).overall_score
        on = _presence.run_presence_suite(
            *repaired_presence_corpora(degraded_p)).overall_score
    else:
        raise ValueError(f"unknown panel suite {suite_id!r}")
    return {"tier": tier["id"], "suite": suite_id,
            "off_score": off, "on_score": on, "delta": on - off}


def compute_panel_rows(
    tiers: Any = None, suites: Any = None,
) -> tuple[dict[str, Any], ...]:
    """The whole panel, measured live (tier-major)."""
    tiers = MODEL_TIERS if tiers is None else tiers
    suites = PANEL_SUITES if suites is None else suites
    return tuple(measure_cell(tier, suite_id)
                 for tier in tiers for suite_id in suites)


def tier_means_for(
    rows: Any, tiers: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Per-tier mean OFF / ON / delta across suites."""
    tiers = MODEL_TIERS if tiers is None else tiers
    means: list = []
    for tier in tiers:
        cells = [r for r in rows if r["tier"] == tier["id"]]
        if not cells:
            means.append({"tier": tier["id"], "cells": 0,
                          "mean_off": 0.0, "mean_on": 0.0,
                          "mean_delta": 0.0})
            continue
        means.append({
            "tier": tier["id"], "cells": len(cells),
            "mean_off": sum(r["off_score"] for r in cells) / len(cells),
            "mean_on": sum(r["on_score"] for r in cells) / len(cells),
            "mean_delta": sum(r["delta"] for r in cells) / len(cells),
        })
    return tuple(means)


def finding_for(rows: Any, tiers: Any = None) -> dict[str, Any]:
    """Where the forge helps most, computed from the
    rows: the tier of greatest mean delta (ties
    broken toward the SMALLER tier — later in the
    filed tier order, which runs large to small),
    and each suite's own helps-most tier by the
    same rule."""
    tiers = MODEL_TIERS if tiers is None else tiers
    order = {t["id"]: i for i, t in enumerate(tiers)}

    def _best(cells: Any) -> str | None:
        best: Any = None
        for cell in cells:
            if best is None or cell["delta"] > best["delta"] or (
                    cell["delta"] == best["delta"]
                    and order[cell["tier"]] > order[best["tier"]]):
                best = cell
        return best["tier"] if best is not None else None

    means = tier_means_for(rows, tiers)
    helps_most = _best(
        [{"tier": m["tier"], "delta": m["mean_delta"]} for m in means
         if m["cells"]]) if rows else None
    per_suite = {
        suite_id: _best([r for r in rows if r["suite"] == suite_id])
        for suite_id in PANEL_SUITES}
    return {"helps_most_tier": helps_most,
            "helps_most_mean_delta": (
                [m["mean_delta"] for m in means
                 if m["tier"] == helps_most][0]
                if helps_most is not None else 0.0),
            "per_suite": per_suite}


# ── The published panel (literal data, TF-067's CURVE precedent) ────────────
#
# Measured live on this VM at banking (2026-11-01)
# by running the suites' filed corpora, degraded by
# each tier's filed transforms, through the suites'
# real runners — twice, identical. The checker
# recomputes every row live and names any drift.
# Filed whatever the numbers are: see the doc for
# the honest shapes (a negative depth delta at the
# small tier; the mid tier helped most overall).

PANEL_ROWS: tuple[dict[str, Any], ...] = (
    {"tier": "reference_14b", "suite": "persona",
     "off_score": 0.8453703703703704,
     "on_score": 0.888888888888889,
     "delta": 0.04351851851851851},
    {"tier": "reference_14b", "suite": "memory",
     "off_score": 0.8055555555555557,
     "on_score": 0.8333333333333334,
     "delta": 0.02777777777777768},
    {"tier": "reference_14b", "suite": "depth",
     "off_score": 0.47222222222222227,
     "on_score": 0.8055555555555557,
     "delta": 0.3333333333333334},
    {"tier": "reference_14b", "suite": "presence",
     "off_score": 0.7777777777777778,
     "on_score": 0.8333333333333334,
     "delta": 0.05555555555555558},
    {"tier": "mid_8b", "suite": "persona",
     "off_score": 0.8453703703703704,
     "on_score": 0.888888888888889,
     "delta": 0.04351851851851851},
    {"tier": "mid_8b", "suite": "memory",
     "off_score": 0.2777777777777778,
     "on_score": 0.8055555555555557,
     "delta": 0.5277777777777779},
    {"tier": "mid_8b", "suite": "depth",
     "off_score": 0.47222222222222227,
     "on_score": 0.8055555555555557,
     "delta": 0.3333333333333334},
    {"tier": "mid_8b", "suite": "presence",
     "off_score": 0.7777777777777778,
     "on_score": 0.8333333333333334,
     "delta": 0.05555555555555558},
    {"tier": "small_3b", "suite": "persona",
     "off_score": 0.8361111111111111,
     "on_score": 0.8796296296296297,
     "delta": 0.04351851851851851},
    {"tier": "small_3b", "suite": "memory",
     "off_score": 0.2777777777777778,
     "on_score": 0.888888888888889,
     "delta": 0.6111111111111112},
    {"tier": "small_3b", "suite": "depth",
     "off_score": 0.47222222222222227,
     "on_score": 0.2777777777777778,
     "delta": -0.19444444444444448},
    {"tier": "small_3b", "suite": "presence",
     "off_score": 0.75,
     "on_score": 0.8055555555555555,
     "delta": 0.05555555555555547},
    {"tier": "tiny_1_1b", "suite": "persona",
     "off_score": 0.633888888888889,
     "on_score": 0.7870370370370371,
     "delta": 0.15314814814814814},
    {"tier": "tiny_1_1b", "suite": "memory",
     "off_score": 0.2777777777777778,
     "on_score": 0.3055555555555556,
     "delta": 0.02777777777777779},
    {"tier": "tiny_1_1b", "suite": "depth",
     "off_score": 0.0, "on_score": 0.0, "delta": 0.0},
    {"tier": "tiny_1_1b", "suite": "presence",
     "off_score": 0.7222222222222223,
     "on_score": 0.7777777777777778,
     "delta": 0.05555555555555547},
)

TIER_MEANS: tuple[dict[str, Any], ...] = (
    {"tier": "reference_14b", "cells": 4,
     "mean_off": 0.7252314814814815,
     "mean_on": 0.8402777777777779,
     "mean_delta": 0.1150462962962963},
    {"tier": "mid_8b", "cells": 4,
     "mean_off": 0.5932870370370371,
     "mean_on": 0.8333333333333335,
     "mean_delta": 0.24004629629629637},
    {"tier": "small_3b", "cells": 4,
     "mean_off": 0.5840277777777778,
     "mean_on": 0.712962962962963,
     "mean_delta": 0.12893518518518515},
    {"tier": "tiny_1_1b", "cells": 4,
     "mean_off": 0.40847222222222224,
     "mean_on": 0.4675925925925926,
     "mean_delta": 0.05912037037037035},
)

FINDING: dict[str, Any] = {
    "helps_most_tier": "mid_8b",
    "helps_most_mean_delta": 0.24004629629629637,
    "per_suite": {"persona": "tiny_1_1b", "memory": "small_3b",
                  "depth": "mid_8b", "presence": "mid_8b"},
}

PANEL_RULES: tuple[dict[str, Any], ...] = (
    {"id": "tiers_anchored_to_the_live_catalogue",
     "rule": "every tier anchors to a filename in "
             "ModelBrowser.CURATED_MODELS, checked live; "
             "params come from params_from_name where "
             "that reader takes the name, and for the "
             "tiny tier from the filename's own '1.1b' "
             "token (params_from_name returns None for "
             "it), regex-verified — never invented"},
    {"id": "degradation_is_filed_transforms_only",
     "rule": "a tier degrades a suite's filed corpus "
             "only by its filed transform ids, applied "
             "in the filed order to the filed answer "
             "fields; questions, expectations and "
             "scoring inputs are never touched"},
    {"id": "suites_scored_by_their_real_runners",
     "rule": "every cell is scored by the suite's own "
             "run_*_suite over an explicit corpus — the "
             "panel reuses the four suites' corpora, "
             "scorers and runners and copies none of "
             "them"},
    {"id": "coherence_degrades_by_drift_shift_only",
     "rule": "coherence sagas are shapes, not text: a "
             "tier's only coherence degradation is its "
             "filed drift_shift moving drift_onset "
             "earlier (clamped at 2; onset None stays "
             "None; drift_turns and upto unchanged)"},
    {"id": "forge_repair_roads_are_filed_per_suite",
     "rule": "ON is degraded + repaired, per suite: "
             "memory re-attaches the first filed "
             "record's real TF-048 marker into the "
             "final sentence (citation axes only — the "
             "explain axis carries no citation by its "
             "own holding rule and runs degraded-only); "
             "persona strips sentences carrying a real "
             "GENERIC_PHRASES member; presence prepends "
             "the standing-goal line a real GoalLedger "
             "replay says was dropped; depth's ON is "
             "the suite's native scaffold road over the "
             "same degraded cases"},
    {"id": "coherence_has_no_repair",
     "rule": "no forge repair exists for the "
             "coherence axis: presence ON == degraded "
             "on coherence, and the panel's presence "
             "delta is the will repair's alone, filed "
             "plainly"},
    {"id": "reference_tier_reproduces_the_pinned_baselines",
     "rule": "the reference tier applies no "
             "transforms, so its OFF cells must "
             "reproduce the four suites' pinned "
             "harness baselines EXACTLY (and depth's "
             "reference ON its undegraded ON) — the "
             "panel's PASS stands on that reproduction"},
    {"id": "panel_rows_are_literal_and_recomputed",
     "rule": "PANEL_ROWS / TIER_MEANS / FINDING are "
             "filed as literal data measured at "
             "banking and recomputed live by the "
             "checker — a row that moves is a finding, "
             "never silently re-filed (TF-067's CURVE "
             "precedent)"},
    {"id": "helps_most_ties_break_toward_the_smaller_tier",
     "rule": "helps-most is argmax delta; an exact "
             "tie is broken toward the smaller tier "
             "(later in the filed large-to-small tier "
             "order), for the tier means and per suite "
             "alike"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty panel scores 0.0 and files "
             "REVIEW; PASS requires all 16 cells "
             "measured and the reference reproduction "
             "exact — a negative cell delta is a "
             "finding filed by name, not a verdict "
             "failure"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "the panel scores the suites' scripted "
             "synthetic corpora (TF-017): canon is "
             "never a benchmark input and data/ was "
             "never opened"},
    {"id": "live_per_size_behaviour_is_not_proven",
     "rule": "no live model ran for any row — no "
             "GGUF in the tree, no llama_cpp on this "
             "VM (TF-078): a tier is a filed "
             "degradation profile anchored to a "
             "catalogue model, never a measurement of "
             "that model; TF-173 owns the real-"
             "hardware road"},
)


# ── The panel run ───────────────────────────────────────────────────────────

@dataclass
class HonestyPanelResult:
    """One full panel run: rows, tier means, finding."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    tier_means: list[dict[str, Any]] = field(default_factory=list)
    finding: dict[str, Any] = field(default_factory=dict)
    total_cells: int = 0
    overall_score: float = 0.0

    @property
    def reference_reproduces(self) -> bool:
        """The reference OFF cells (and depth ON)
        reproduce the pinned baselines exactly."""
        cells = {(r["tier"], r["suite"]): r for r in self.rows}
        try:
            return (
                cells[("reference_14b", "persona")]["off_score"]
                == REFERENCE_BASELINES["persona_off"]
                and cells[("reference_14b", "memory")]["off_score"]
                == REFERENCE_BASELINES["memory_off"]
                and cells[("reference_14b", "depth")]["off_score"]
                == REFERENCE_BASELINES["depth_off"]
                and cells[("reference_14b", "depth")]["on_score"]
                == REFERENCE_BASELINES["depth_on"]
                and cells[("reference_14b", "presence")]["off_score"]
                == REFERENCE_BASELINES["presence_off"])
        except KeyError:
            return False

    @property
    def verdict(self) -> str:
        """PASS only on a full 16-cell panel whose
        reference tier reproduces the baselines."""
        if self.total_cells == 0:
            return "REVIEW"
        if self.total_cells != 16:
            return "REVIEW"
        return "PASS" if self.reference_reproduces else "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_honesty_panel(
    tiers: Any = None, suites: Any = None,
) -> HonestyPanelResult:
    """Run the whole panel (filed tiers/suites default).

    Any explicit tier/suite selection may be given —
    including an empty one, which scores 0.0 and
    keeps the panel at REVIEW (no vacuous pass).
    """
    tiers = MODEL_TIERS if tiers is None else tiers
    suites = PANEL_SUITES if suites is None else suites
    rows = compute_panel_rows(tiers, suites)
    means = tier_means_for(rows, tiers)
    total = len(rows)
    overall = (
        sum(m["mean_delta"] for m in means) / len(means)
        if total and means else 0.0)
    return HonestyPanelResult(
        rows=[dict(r) for r in rows],
        tier_means=[dict(m) for m in means],
        finding=finding_for(rows, tiers),
        total_cells=total, overall_score=overall)


# ── The live checker ────────────────────────────────────────────────────────

def honesty_panel_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    # Tiers: shape, anchors live in the catalogue,
    # params honest (the reader, or the filename token).
    tier_ids = [t["id"] for t in MODEL_TIERS]
    if tier_ids != ["reference_14b", "mid_8b", "small_3b",
                    "tiny_1_1b"]:
        problems.append(f"MODEL_TIERS changed shape: {tier_ids}")
    catalogue = {m.filename for m in ModelBrowser.CURATED_MODELS}
    for tier in MODEL_TIERS:
        if tier["anchor"] not in catalogue:
            problems.append(
                f"tier {tier['id']} anchor {tier['anchor']} is not "
                f"in the live catalogue")
        if tier["params_basis"] == "params_from_name":
            got = params_from_name(tier["anchor"])
            if got != tier["params"]:
                problems.append(
                    f"tier {tier['id']} files params "
                    f"{tier['params']} but params_from_name reads "
                    f"{got}")
        elif tier["params_basis"] == "filename_token":
            match = re.search(r"(\d+(?:\.\d+)?)b", tier["anchor"],
                              flags=re.IGNORECASE)
            if match is None or float(match.group(1)) != tier["params"]:
                problems.append(
                    f"tier {tier['id']} params {tier['params']} do "
                    f"not come from its anchor's filename token")
            if params_from_name(tier["anchor"]) is not None:
                problems.append(
                    f"tier {tier['id']} files the filename-token "
                    f"basis but params_from_name now reads it")
        for transform_id in tier["transforms"]:
            if transform_id not in TRANSFORM_FUNCS:
                problems.append(
                    f"tier {tier['id']} names unknown transform "
                    f"{transform_id}")

    # Transforms + rules: ids filed, the closer honest.
    transform_ids = [t["id"] for t in TRANSFORMS]
    if transform_ids != list(TRANSFORM_FUNCS):
        problems.append(
            f"TRANSFORMS {transform_ids} != the filed functions "
            f"{list(TRANSFORM_FUNCS)}")
    if GENERIC_CLOSER_PHRASE not in GENERIC_PHRASES:
        problems.append(
            f"the closer phrase {GENERIC_CLOSER_PHRASE!r} is not "
            f"a GENERIC_PHRASES member")
    if GENERIC_CLOSER_PHRASE not in GENERIC_CLOSER_TEXT.lower():
        problems.append("the closer text does not carry its "
                        "filed phrase")
    rule_ids = [r["id"] for r in PANEL_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate PANEL_RULES ids")
    for needed in ("tiers_anchored_to_the_live_catalogue",
                   "degradation_is_filed_transforms_only",
                   "suites_scored_by_their_real_runners",
                   "coherence_degrades_by_drift_shift_only",
                   "forge_repair_roads_are_filed_per_suite",
                   "coherence_has_no_repair",
                   "reference_tier_reproduces_the_pinned_baselines",
                   "panel_rows_are_literal_and_recomputed",
                   "helps_most_ties_break_toward_the_smaller_tier",
                   "no_vacuous_pass_anywhere",
                   "fixtures_are_synthetic_only",
                   "live_per_size_behaviour_is_not_proven"):
        if needed not in rule_ids:
            problems.append(f"PANEL_RULES lost {needed}")

    # The reference reproduction, live.
    reference = compute_panel_rows(MODEL_TIERS[:1])
    ref_result = HonestyPanelResult(
        rows=[dict(r) for r in reference],
        tier_means=[], finding={}, total_cells=len(reference),
        overall_score=0.0)
    if not ref_result.reference_reproduces:
        problems.append(
            f"the reference tier does not reproduce the pinned "
            f"baselines: {reference}")

    # The published panel: literal rows recomputed
    # live, twice — filing, recomputation and
    # determinism in one pass.
    live_rows = compute_panel_rows()
    if [dict(r) for r in live_rows] != [dict(r) for r in PANEL_ROWS]:
        problems.append(
            "PANEL_ROWS drifted from the live recomputation")
    live_means = tier_means_for(live_rows)
    if [dict(m) for m in live_means] != [dict(m) for m in TIER_MEANS]:
        problems.append(
            "TIER_MEANS drifted from the live recomputation")
    if finding_for(live_rows) != FINDING:
        problems.append("FINDING drifted from the filed rows")
    second = compute_panel_rows()
    if second != live_rows:
        problems.append("the panel is not deterministic: two "
                        "live runs differ")

    # A fresh full run files PASS; an empty panel
    # files 0.0 / REVIEW.
    result = run_honesty_panel()
    if result.total_cells != 16 or result.verdict != "PASS" \
            or not result.passes:
        problems.append(
            f"the live panel run does not file 16 cells / PASS: "
            f"{result.total_cells} / {result.verdict}")
    empty = run_honesty_panel(tiers=())
    if empty.total_cells != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty panel does not file 0.0 / REVIEW")

    # Every earlier pinned baseline preserved exactly.
    from benchmarks.harness import baseline_problems

    earlier = baseline_problems()
    if earlier:
        problems.append(
            f"earlier harness baselines moved: {earlier}")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / HONESTY_PANEL_DOC
    if not doc.exists():
        problems.append(f"the write-up {HONESTY_PANEL_DOC} is missing")
    return problems


def honesty_panel_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_honesty_panel()
    return {
        "format": HONESTY_PANEL_FORMAT,
        "doc": HONESTY_PANEL_DOC,
        "suites": list(PANEL_SUITES),
        "tiers": [
            {**dict(t), "transforms": list(t["transforms"])}
            for t in MODEL_TIERS],
        "transforms": [dict(t) for t in TRANSFORMS],
        "reference_baselines": dict(REFERENCE_BASELINES),
        "rows": [dict(r) for r in PANEL_ROWS],
        "tier_means": [dict(m) for m in TIER_MEANS],
        "finding": dict(FINDING),
        "measured": {
            "total_cells": result.total_cells,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in PANEL_RULES],
    }


__all__ = [
    "FACT_SWAPS",
    "FINDING",
    "GENERIC_CLOSER_PHRASE",
    "GENERIC_CLOSER_TEXT",
    "HONESTY_PANEL_DOC",
    "HONESTY_PANEL_FORMAT",
    "MODEL_TIERS",
    "PANEL_ROWS",
    "PANEL_RULES",
    "PANEL_SUITES",
    "REFERENCE_BASELINES",
    "TIER_BY_ID",
    "TIER_MEANS",
    "TRANSFORMS",
    "TRANSFORM_FUNCS",
    "HonestyPanelResult",
    "apply_transforms",
    "compute_panel_rows",
    "degraded_depth_corpora",
    "degraded_memory_corpora",
    "degraded_persona_corpora",
    "degraded_presence_corpora",
    "finding_for",
    "honesty_panel_as_data",
    "honesty_panel_problems",
    "measure_cell",
    "repair_memory_case",
    "repair_will_sagas",
    "repaired_memory_corpora",
    "repaired_persona_corpora",
    "repaired_presence_corpora",
    "run_honesty_panel",
    "tier_means_for",
]
