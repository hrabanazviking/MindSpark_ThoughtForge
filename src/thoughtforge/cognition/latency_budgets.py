"""
Latency budgets for the core loop (TF-033).

Before this slice the core loop had timings but no budgets:
``RuntimeTurnState`` carries measured ``retrieval_ms`` /
``generation_ms`` / ``salvage_ms`` (TF-025 made every one of them
real), and the perf tracker records ``think.total`` — but nothing
anywhere said how much time a turn is *allowed*, so a slow turn was
indistinguishable from a fast one in every test and report.

This module is the budget table, as data, with its derivation
stated plainly — no number here pretends to be a measurement it
isn't:

  * **retrieval** — the profile's OWN stated number:
    ``retrieval.retrieval_timeout_ms`` from the shipped profile
    JSON, read live through ``load_hardware_profile`` (never copied
    into a second table that could drift). This is also the first
    live consumer of any profile ``retrieval`` value — TF-012
    recorded that whole section as read by no code (D-PROF-1 keeps
    the rest of that section; this slice narrows only this key).
  * **generation / salvage** — campaign-set ceilings, derived from
    the desktop_cpu base by a stated hardware factor per profile
    (server_gpu ×0.5, desktop_gpu ×0.75, desktop_cpu ×1, pi_5 ×2,
    phone_low ×3, pi_zero ×5). The base was set from this tick's VM
    measurements (see docs/architecture/LATENCY_BUDGETS.md): on the
    VM profile every stage of a fixture turn measured 0 ms at the
    turn state's integer-ms granularity and wall-clock was ~1 ms
    warm / 51 ms first-turn, so the ceilings carry roughly an order
    of magnitude of headroom over fixture reality — deliberately,
    because this VM is shared and walls measured under contention
    are not baselines. The ceilings bind the CORE LOOP's behaviour
    with fixture-scale generation; they are not a claim about any
    real model's generation speed (no real model ran on this VM
    for this slice — real-model latency is TF-079/TF-175's road,
    and per-profile proof beyond the VM profile is NOT PROVEN).
  * **total** — never set independently: it is the sum of the
    three stage budgets, so the table cannot contradict itself.

Checking is pure: ``check_turn_latency`` reads a finished turn
state's three stage timings (plus a caller-measured wall-clock
total when one is supplied) and returns a ``LatencyVerdict`` —
per-stage verdicts, the list of overruns, and a within/outside
answer. A stage that did not run measures 0 (TF-025's ruling:
salvage 0 on the knowledge-only path means "did not run") and is
within budget by construction; the verdict records the measured
value either way, so "did not run" is visible, never hidden.

Enforcement lives in the tests (the roadmap's own words for this
slice): on the VM profile — ``desktop_cpu``, which the profile
selector auto-selects on this machine, re-verified live this
slice — real turns through the core must land inside every stage
budget and the wall-clock total, and a deliberately slow backend
must be caught by name. The core exposes the surface:
``core.latency_budget`` (the selected conversation profile's
budget, or the VM-profile default when none is selected — see
core.py) and ``core.last_turn_latency()``.

The human-readable write-up is
``docs/architecture/LATENCY_BUDGETS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The stages a budget covers, in turn order. "total" is checked
# only when the caller supplies a wall-clock measurement — the
# turn state carries no total field (the perf tracker holds it),
# and a stage-sum labelled as a wall total would be a lie.
STAGES: tuple[str, ...] = ("retrieval", "generation", "salvage")

# Campaign-set base ceilings for the VM profile (desktop_cpu),
# in milliseconds. Derivation in the module docstring + the
# architecture doc: fixture turns measure ~0 ms per stage and
# ~1 ms warm wall on this VM; these are ceilings with headroom,
# not targets and not real-model numbers.
BASE_GENERATION_MS = 1000
BASE_SALVAGE_MS = 500

# Stated hardware factor per profile for the campaign-set stages.
# Retrieval is NOT factored — it comes from each profile's own
# JSON. Every shipped profile must appear here; a profile added
# to the tree without a factor fails resolution loudly instead of
# silently inheriting desktop numbers.
PROFILE_FACTORS: dict[str, float] = {
    "server_gpu": 0.5,
    "desktop_gpu": 0.75,
    "desktop_cpu": 1.0,
    "pi_5": 2.0,
    "phone_low": 3.0,
    "pi_zero": 5.0,
}

# The profile this VM auto-selects (TF-012, re-verified live by
# TF-033 through ProfileSelector.auto_select) — the one profile
# whose budgets are measured + enforced in tests on this machine.
VM_PROFILE_ID = "desktop_cpu"


@dataclass(frozen=True)
class LatencyBudget:
    """One profile's core-loop latency ceilings, in milliseconds."""

    profile_id: str
    retrieval_ms: int
    generation_ms: int
    salvage_ms: int

    @property
    def total_ms(self) -> int:
        """The wall-clock ceiling: the stage budgets' sum, by
        construction — never an independent number."""
        return self.retrieval_ms + self.generation_ms + self.salvage_ms

    def stage_budget(self, stage: str) -> int:
        if stage == "retrieval":
            return self.retrieval_ms
        if stage == "generation":
            return self.generation_ms
        if stage == "salvage":
            return self.salvage_ms
        if stage == "total":
            return self.total_ms
        raise ValueError(f"unknown latency stage: {stage!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "salvage_ms": self.salvage_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class StageVerdict:
    """One stage's measured time against its budget."""

    stage: str
    measured_ms: float
    budget_ms: int

    @property
    def within(self) -> bool:
        return self.measured_ms <= self.budget_ms


@dataclass(frozen=True)
class LatencyVerdict:
    """A whole turn's latency verdict against one budget."""

    profile_id: str
    stages: tuple[StageVerdict, ...]

    @property
    def within_budget(self) -> bool:
        return all(stage.within for stage in self.stages)

    @property
    def overruns(self) -> tuple[StageVerdict, ...]:
        """The stages that broke their budget, in turn order."""
        return tuple(stage for stage in self.stages if not stage.within)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "within_budget": self.within_budget,
            "stages": [
                {
                    "stage": stage.stage,
                    "measured_ms": stage.measured_ms,
                    "budget_ms": stage.budget_ms,
                    "within": stage.within,
                }
                for stage in self.stages
            ],
        }


def budget_for_profile(profile_id: str) -> LatencyBudget:
    """Resolve one shipped profile's latency budget.

    The retrieval ceiling is read live from the profile JSON; the
    generation/salvage ceilings are the campaign base scaled by
    the profile's stated factor. An unknown profile — or a shipped
    profile with no factor recorded — raises ``ValueError`` naming
    the valid ids: a budget never silently defaults.
    """
    from thoughtforge.utils.config import (
        list_hardware_profiles,
        load_hardware_profile,
    )

    valid = list_hardware_profiles()
    if profile_id not in valid:
        raise ValueError(
            f"unknown hardware profile: {profile_id!r} "
            f"(valid: {', '.join(valid)})"
        )
    if profile_id not in PROFILE_FACTORS:
        raise ValueError(
            f"no latency factor recorded for profile: {profile_id!r}"
        )
    profile = load_hardware_profile(profile_id)
    retrieval_ms = int(profile["retrieval"]["retrieval_timeout_ms"])
    factor = PROFILE_FACTORS[profile_id]
    return LatencyBudget(
        profile_id=profile_id,
        retrieval_ms=retrieval_ms,
        generation_ms=int(round(BASE_GENERATION_MS * factor)),
        salvage_ms=int(round(BASE_SALVAGE_MS * factor)),
    )


def all_budgets() -> dict[str, LatencyBudget]:
    """Every shipped profile's budget, keyed by profile id."""
    from thoughtforge.utils.config import list_hardware_profiles

    return {
        profile_id: budget_for_profile(profile_id)
        for profile_id in list_hardware_profiles()
    }


def stage_measurements(turn: Any) -> dict[str, float]:
    """The turn state's measured stage timings, in turn order.

    Reads exactly the three fields TF-025 made real on
    ``RuntimeTurnState`` — nothing is estimated or filled in.
    """
    return {
        "retrieval": float(turn.retrieval_ms),
        "generation": float(turn.generation_ms),
        "salvage": float(turn.salvage_ms),
    }


def check_turn_latency(
    turn: Any,
    budget: LatencyBudget,
    total_ms: float | None = None,
) -> LatencyVerdict:
    """Judge one finished turn against ``budget``.

    ``total_ms`` is the caller's own wall-clock measurement around
    the turn; when supplied, a ``total`` stage verdict is appended
    (against the budget's summed total). When omitted, no total is
    judged — the turn state carries no honest total of its own.
    """
    measured = stage_measurements(turn)
    verdicts = [
        StageVerdict(
            stage=stage,
            measured_ms=measured[stage],
            budget_ms=budget.stage_budget(stage),
        )
        for stage in STAGES
    ]
    if total_ms is not None:
        verdicts.append(
            StageVerdict(
                stage="total",
                measured_ms=float(total_ms),
                budget_ms=budget.total_ms,
            )
        )
    return LatencyVerdict(
        profile_id=budget.profile_id, stages=tuple(verdicts)
    )
