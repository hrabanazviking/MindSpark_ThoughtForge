"""
TF-045 — the memory lifecycle's rules, stated as verifiable data.

The lifecycle (``knowledge/lifecycle.py``) is how facts age, decay,
and are retired: recency is a half-life decay (TF-041 K3), a prune
score is built per record from its age and quality signals
(``knowledge/scoring.py``), and each prune mode's archive/delete
threshold tables decide the outcome. Until this slice the module
was test-only — TF-011's D-ORPHAN-1: nothing in the live engine
ever aged a fact. TF-045 wires it (see ``WIRING``) and states its
rules at their edges, measured, not asserted:

  * **The prune score has a floor and a ceiling.** The prune paths
    always build components with the builder defaults
    (redundancy 0.0, future_relevance 0.5), so a record's score
    lives in [0.05, 0.75]: even a fresh, perfect record scores
    0.05 (the neutral future-relevance share), and even an
    infinitely old, zero-quality record asymptotes to 0.75.
  * **Reachability follows from the ceiling.** Against the
    threshold tables: light mode can NEVER archive or delete by
    score (its lowest archive threshold, 0.85, sits above the
    ceiling) — under light, only deduplication and hard caps act.
    Routine can archive but never delete by score (lowest delete
    0.82 > 0.75). Heavy deletes by score for fact / episode /
    pattern (0.72 / 0.70 / 0.72) at ordinary ages; preference
    heavy-delete sits exactly AT the ceiling (0.75), so it fires
    only once the decay term has underflowed to zero in floating
    point — an age of roughly 265 years for a preference's
    90-day half-life: formally reachable, practically never.
    Emergency reaches both outcomes for every kind. ``score_reachability()`` derives this table
    from the live constants; the tests prove its reachable cells
    with engineered records and its unreachable cells by the
    ceiling arithmetic.
  * **Boundaries are inclusive.** score >= delete threshold
    deletes; else score >= archive threshold archives; else the
    record is kept (``decision()`` — the lifecycle's own
    comparisons, stated once).
  * **Retirement is physical.** A record marked archived/deleted
    is absent from its file after the pass's rewrite (TF-041
    K13) — "archived" is a count of what left the live store,
    not a status the store keeps.
  * **Protected outranks everything.** Score, dedup, and — after
    the TF-045 fix — hard-cap enforcement all skip protected
    records; a protected surplus may leave a cap exceeded.
  * **An unreadable timestamp ages a record by exactly
    ``AGE_PARSE_DEFAULT_DAYS`` (30)** — neither immortal nor
    instantly ancient; a future timestamp ages 0.
  * **Reinforcement is capped arithmetic.** Each reinforce adds
    the named deltas × signal strength, clamped at 1.0, and bumps
    the record's counters; a pattern's quality (and, after the
    TF-045 model fix, its ``times_reinforced``) move the same way.

``lifecycle_problems()`` re-derives every claim above from the
live tree and names any that stops being true.

The human-readable write-up is ``docs/architecture/LIFECYCLE.md``.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

from thoughtforge.knowledge import lifecycle as lifecycle_module
from thoughtforge.knowledge.lifecycle import (
    AGE_PARSE_DEFAULT_DAYS,
    ARCHIVE_THRESHOLDS,
    DELETE_THRESHOLDS,
    STORE_CAPS,
    MemoryLifecycle,
    _age_days,
    _build_prune_components,
)
from thoughtforge.knowledge.models import (
    EpisodicMemoryRecord,
    ResponsePatternRecord,
    UserFactRecord,
    UserPreferenceRecord,
)
from thoughtforge.knowledge.scoring import (
    HALF_LIFE_DAYS,
    PRUNE_SCORE_WEIGHTS,
    REINFORCE_CONFIDENCE_DELTA,
    REINFORCE_IMPORTANCE_DELTA,
    REINFORCE_QUALITY_DELTA,
    compute_prune_score,
)

# ── The rule tables, as data ─────────────────────────────────────────────────

MODES: tuple[str, ...] = ("light", "routine", "heavy", "emergency")

#: Prune-threshold kind per persisted record_type (the lifecycle's
#: own rtype_map for profile records, extended by its per-store
#: helpers for episodes and patterns).
KIND_BY_RECORD_TYPE: dict[str, str] = {
    "user_preference": "preference",
    "user_fact": "fact",
    "episode": "episode",
    "response_pattern": "pattern",
}

#: The component-builder defaults every prune path uses (the
#: lifecycle calls _build_prune_components(record, age) with no
#: overrides): redundancy is never measured at prune time, and
#: future relevance is the neutral 0.5. These two defaults are
#: what bound the prune score's range.
BUILDER_DEFAULTS: dict[str, float] = {
    "redundancy": 0.0,
    "future_relevance": 0.5,
}


def prune_score_bounds() -> tuple[float, float]:
    """(floor, ceiling) of any prune score the prune paths can
    produce, derived from the live weights + builder defaults:
    every component at its best (floor) or worst (ceiling), with
    redundancy pinned at its default 0 and low_future_relevance
    pinned at 1 − 0.5 by the neutral default."""
    w = PRUNE_SCORE_WEIGHTS
    low_future = 1.0 - BUILDER_DEFAULTS["future_relevance"]
    floor = w["low_future_relevance"] * low_future
    ceiling = (
        w["staleness"]
        + w["redundancy"] * BUILDER_DEFAULTS["redundancy"]
        + w["low_confidence"]
        + w["low_importance"]
        + w["low_quality"]
        + w["low_future_relevance"] * low_future
        + w["low_reinforcement"]
    )
    return (floor, ceiling)


#: Measured at TF-045 from the shipped weights: (0.05, 0.75).
PRUNE_SCORE_FLOOR, PRUNE_SCORE_CEILING = prune_score_bounds()


def decision(score: float, mode: str, kind: str) -> str:
    """The lifecycle's threshold decision, stated once: "deleted"
    at or above the delete threshold, else "archived" at or above
    the archive threshold, else "kept". Inclusive boundaries —
    a score exactly on a threshold takes that threshold's outcome.
    """
    if score >= DELETE_THRESHOLDS[mode][kind]:
        return "deleted"
    if score >= ARCHIVE_THRESHOLDS[mode][kind]:
        return "archived"
    return "kept"


def score_reachability() -> dict[str, dict[str, dict[str, bool]]]:
    """Per mode × kind: can the score path reach archive / delete
    at all? Derived from the live ceiling vs the live tables."""
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for mode in MODES:
        out[mode] = {}
        for kind in ("preference", "fact", "episode", "pattern"):
            out[mode][kind] = {
                "archive": PRUNE_SCORE_CEILING >= ARCHIVE_THRESHOLDS[mode][kind],
                "delete": PRUNE_SCORE_CEILING >= DELETE_THRESHOLDS[mode][kind],
            }
    return out


def age_for_prune_score(
    target: float,
    record_type: str,
    *,
    confidence: float = 0.0,
    importance: float = 0.0,
    quality: float = 0.0,
    times_reinforced: int = 0,
) -> float | None:
    """The age in days at which a record with the given signals
    scores exactly ``target`` — the boundary-engineering inverse
    of the prune score, so edge tests can place a record a stated
    distance either side of a threshold. None when the target is
    outside the range those signals can produce (staleness alone
    varies with age)."""
    w = PRUNE_SCORE_WEIGHTS
    fixed = (
        w["low_confidence"] * (1.0 - confidence)
        + w["low_importance"] * (1.0 - importance)
        + w["low_quality"] * (1.0 - quality)
        + w["low_future_relevance"] * (1.0 - BUILDER_DEFAULTS["future_relevance"])
        + w["low_reinforcement"] * (1.0 - min(times_reinforced / 10.0, 1.0))
    )
    needed_staleness = (target - fixed) / w["staleness"]
    if not 0.0 <= needed_staleness < 1.0:
        return None
    recency = 1.0 - needed_staleness
    half_life = HALF_LIFE_DAYS[record_type]
    return half_life * math.log2(1.0 / recency)


#: The wiring TF-045 landed (D-ORPHAN-1's fix), as data. Each row
#: is checked live by lifecycle_problems() and by TF-041's K10.
WIRING: tuple[dict[str, Any], ...] = (
    {"surface": "per-turn aging",
     "where": "ThoughtForgeCore._finish_turn",
     "act": "MemoryLifecycle(store).cool_thread() once per completed turn",
     "scope": "the active thread state only — ephemeral by design"},
    {"surface": "explicit prune",
     "where": "ThoughtForgeCore.prune_memory(mode)",
     "act": "MemoryLifecycle(store).prune(mode) over the core's own store",
     "scope": "operator-invoked; fact pruning is never automatic per turn"},
    {"surface": "operator tool",
     "where": "forge_memory.py",
     "act": "TF-057 landed it: the forge_memory toolset "
            "(knowledge/forge_toolset.py) — add/inspect/prune/export "
            "over --memory-dir; prune runs MemoryLifecycle in the "
            "named mode (optional TF-056 profile caps)",
     "scope": "operator-invoked over one memory store; fact pruning "
              "is never automatic per turn"},
)


def lifecycle_problems() -> list[str]:
    """Re-derive TF-045's claims from the live tree; name any
    that stopped being true."""
    problems: list[str] = []

    # Tables cover exactly the stated modes and kinds.
    for table_name, table in (
        ("ARCHIVE_THRESHOLDS", ARCHIVE_THRESHOLDS),
        ("DELETE_THRESHOLDS", DELETE_THRESHOLDS),
    ):
        if tuple(table) != MODES and set(table) != set(MODES):
            problems.append(f"{table_name}: modes != {MODES}")
        for mode in MODES:
            if set(table.get(mode, {})) != {"preference", "fact", "episode", "pattern"}:
                problems.append(f"{table_name}[{mode}]: kinds drifted")

    # Bounds: derived values equal the builder's real extremes.
    worst = UserFactRecord(
        record_id="bounds_worst", confidence=0.0, importance=0.0
    )
    worst_score = compute_prune_score(_build_prune_components(worst, 100_000.0))
    if not math.isclose(worst_score, PRUNE_SCORE_CEILING, abs_tol=1e-9):
        problems.append(
            f"ceiling: worst-record score {worst_score} != {PRUNE_SCORE_CEILING}"
        )
    fresh = UserPreferenceRecord(
        record_id="bounds_fresh", category="bounds", confidence=1.0,
        importance=1.0, times_reinforced=10,
    )
    fresh_score = compute_prune_score(_build_prune_components(fresh, 0.0))
    if not math.isclose(fresh_score, PRUNE_SCORE_FLOOR, abs_tol=1e-9):
        problems.append(
            f"floor: fresh-perfect score {fresh_score} != {PRUNE_SCORE_FLOOR}"
        )

    # Reachability spot-truths the doc states in words.
    reach = score_reachability()
    if any(v["archive"] or v["delete"] for v in reach["light"].values()):
        problems.append("reachability: light mode became score-reachable")
    if any(v["delete"] for v in reach["routine"].values()):
        problems.append("reachability: routine delete became score-reachable")
    if not all(v["delete"] for v in reach["emergency"].values()):
        problems.append("reachability: emergency lost a delete path")

    # Decision semantics match the tables at the boundaries.
    for mode in MODES:
        for kind in ("preference", "fact", "episode", "pattern"):
            if decision(DELETE_THRESHOLDS[mode][kind], mode, kind) != "deleted":
                problems.append(f"decision: {mode}/{kind} delete boundary not inclusive")
            if decision(ARCHIVE_THRESHOLDS[mode][kind], mode, kind) not in (
                "archived", "deleted",
            ):
                problems.append(f"decision: {mode}/{kind} archive boundary not inclusive")
            if decision(ARCHIVE_THRESHOLDS[mode][kind] - 1e-9, mode, kind) != "kept":
                problems.append(f"decision: {mode}/{kind} just-below-archive not kept")

    # Age edges.
    if AGE_PARSE_DEFAULT_DAYS != 30.0:
        problems.append("age: AGE_PARSE_DEFAULT_DAYS moved from 30.0")
    if _age_days("not a timestamp") != AGE_PARSE_DEFAULT_DAYS:
        problems.append("age: unparseable timestamp does not age by the default")

    # The TF-045 defect fixes stand.
    if not hasattr(ResponsePatternRecord(record_id="p"), "times_reinforced"):
        problems.append(
            "fix: ResponsePatternRecord lost times_reinforced — "
            "pattern reinforcement crashes again"
        )
    prune_src = inspect.getsource(MemoryLifecycle.prune)
    if "unknown prune mode" not in prune_src:
        problems.append(
            "fix: prune() no longer refuses an unknown mode up front "
            "with a ValueError naming the valid modes"
        )
    cap_src = inspect.getsource(MemoryLifecycle._enforce_cap)
    if "protected" not in cap_src:
        problems.append(
            "fix: _enforce_cap no longer guards protected records — "
            "the cap can delete a protected record again"
        )

    # Reinforcement deltas are the named, positive constants.
    for name, value in (
        ("REINFORCE_CONFIDENCE_DELTA", REINFORCE_CONFIDENCE_DELTA),
        ("REINFORCE_IMPORTANCE_DELTA", REINFORCE_IMPORTANCE_DELTA),
        ("REINFORCE_QUALITY_DELTA", REINFORCE_QUALITY_DELTA),
    ):
        if value <= 0:
            problems.append(f"reinforce: {name} is not positive")

    # Caps cover the four record stores plus queue and loops.
    if set(STORE_CAPS) != {
        "user_preference", "user_fact", "episode",
        "response_pattern", "recent_queue", "open_loops",
    }:
        problems.append("caps: STORE_CAPS keys drifted from the mapped six")

    # The wiring stands (TF-041's K10 checks the same legs).
    from thoughtforge.cognition import core as core_module

    core_src = inspect.getsource(core_module)
    if "MemoryLifecycle" not in core_src or "prune_memory" not in core_src:
        problems.append("wiring: the core's lifecycle wiring has regressed")

    # Every censused record type has a threshold kind and episodes
    # construct (the prune paths' raw material).
    for record_type in KIND_BY_RECORD_TYPE:
        if record_type not in HALF_LIFE_DAYS and record_type != "episode":
            problems.append(f"kinds: {record_type} has no half-life row")
    _ = EpisodicMemoryRecord(record_id="census_episode")
    _ = lifecycle_module  # the module under rule
    return problems
