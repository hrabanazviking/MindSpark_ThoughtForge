"""The small-model floor (TF-037).

Part II built the cognition core piece by piece and proved each
piece on comfortable budgets. This slice asks the floor question:
*does the whole core still hold together at the smallest end of
its own stated range — the ``phone_low`` profile, the weakest
hardware profile the tree ships (~2 GB RAM, a ≤1.1 B model) —
driven by the smallest fixture model in the estate?*

Three things are made data here, then proven together:

  * **The floor budget** — :class:`FloorBudget`, resolved LIVE
    from the shipped ``phone_low`` profile through the tree's own
    loaders (``load_hardware_profile`` +
    ``resolve_generation_params``, the selector's resolver), never
    copied into a second table that could drift: response cap
    180 word-tokens, per-turn token budget 180, context ceiling
    512, draft count 2, refinement cap 1 pass, memory-bundle cap
    5 facts — plus the profile's TF-033 latency budget
    (500 / 3,000 / 1,500 ms, total 5,000). "Simulated profile" is
    the honest phrase: the conversation *selects* phone_low
    through TF-032's config surface on this VM; no phone ran.
  * **The smallest fixture model** — :class:`FloorFixtureEngine`:
    TF-028's deterministic coherence fixture, wrapped so it
    honours a ``max_tokens`` cap the way a real small model must
    (word-truncated completion, cap recorded per call). It is the
    smallest model-shaped thing in the estate: no weights, no
    network, script-scale answers in TF-017's synthetic world.
    Where TF-028's engine deliberately *ignores* params so drift
    measurements stay pure, the floor engine *obeys* the cap —
    the floor question is precisely whether the core behaves
    when the model is this small and this constrained.
  * **The floor checker + runner** — :func:`check_floor_turn`
    judges one finished turn on five dimensions (response tokens,
    prompt context, drafts, refinement passes, latency) and names
    every overrun; :func:`run_floor_conversation` runs TF-028's
    full 20-turn coherence script through a phone_low-selected
    core and collects one observation per turn, so the floor
    verdict and the drift verdict (TF-028's scorer, reused) are
    measured on the same conversation.

DEFECT FOUND + FIXED in this slice (in core.py /
config_surface.py): the profile's ``max_refinement_passes`` was
resolved by the inference resolver and read by no cognition code
— salvage ran a flat 2 refine passes on every profile. Measured
pre-fix on the floor: 7 of 20 turns ran ``refine_pass_2`` under
phone_low, whose JSON promises at most 1. The TF-032 config
surface now resolves the profile's pass cap alongside its token
cap and ``_compose_final`` hands it to ``FragmentSalvage.forge``;
no profile selected → the standing 2 passes, byte-identical
(TF-028's 54-call pin stands untouched). This is D-PROF-4's floor
share and narrows D-PROF-3 again; TF-098 keeps the general
refinement-cost discipline (measured cost, all profiles).

Proof grade, stated plainly: FIXTURE model on a SIMULATED
profile, on this VM. The core holds at the floor — coherence
PASS, every turn inside every floor dimension — but no real
small model and no phone have been measured here. A live small
model at these budgets is TF-126's road; phone hardware is
TF-069/TF-128's. Retrieval stays at the core's flat top_k and
the bundle the empty store returns is empty, so the bundle-fact
cap is recorded in the budget as data but NOT exercised by this
run (populated-store retrieval is Part III's road, TF-041+).

The human-readable write-up is
``docs/architecture/SMALL_MODEL_FLOOR.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.cognition.coherence import (
    COHERENCE_SCRIPT,
    CoherenceFixtureEngine,
    ScriptTurn,
    score_conversation,
)
from thoughtforge.cognition.history_discipline import estimate_tokens
from thoughtforge.cognition.latency_budgets import (
    LatencyBudget,
    LatencyVerdict,
    budget_for_profile,
)

# The floor profile: the weakest hardware profile the tree ships.
FLOOR_PROFILE_ID = "phone_low"

# The dimensions a floor verdict judges, in turn order.
DIMENSIONS: tuple[str, ...] = (
    "response_tokens",
    "prompt_context",
    "drafts",
    "refinement",
    "latency",
)


# ── The floor budget ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FloorBudget:
    """phone_low's per-turn budget, resolved live from the profile.

    Every number is read from the shipped profile JSON through the
    tree's own resolver at call time — this dataclass is a view,
    not a copy. ``latency`` is TF-033's budget for the same profile.
    """

    profile_id: str
    max_response_tokens: int
    token_budget_per_turn: int
    max_context_tokens: int
    draft_count: int
    max_refinement_passes: int
    max_memory_bundle_facts: int
    latency: LatencyBudget

    @property
    def response_token_cap(self) -> int:
        """The binding response cap: the tighter of the profile's
        generation cap and its per-turn token budget (both 180 on
        phone_low — the cap cannot be looser than either claim)."""
        return min(self.max_response_tokens, self.token_budget_per_turn)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "max_response_tokens": self.max_response_tokens,
            "token_budget_per_turn": self.token_budget_per_turn,
            "max_context_tokens": self.max_context_tokens,
            "draft_count": self.draft_count,
            "max_refinement_passes": self.max_refinement_passes,
            "max_memory_bundle_facts": self.max_memory_bundle_facts,
            "latency": self.latency.to_dict(),
        }


def floor_budget(profile_id: str = FLOOR_PROFILE_ID) -> FloorBudget:
    """Resolve the floor budget for ``profile_id`` (default: the
    floor profile, phone_low) live from its shipped JSON.

    Raises ValueError for an unknown profile, through the loaders.
    """
    from thoughtforge.inference.profiles import resolve_generation_params
    from thoughtforge.utils.config import load_hardware_profile

    profile = load_hardware_profile(profile_id)
    resolved = resolve_generation_params(profile)
    memory = profile.get("memory") or {}
    return FloorBudget(
        profile_id=profile_id,
        max_response_tokens=resolved.max_response_tokens,
        token_budget_per_turn=resolved.token_budget_per_turn,
        max_context_tokens=resolved.max_context_tokens,
        draft_count=resolved.draft_count,
        max_refinement_passes=resolved.max_refinement_passes,
        max_memory_bundle_facts=int(memory.get("max_memory_bundle_facts", 0)),
        latency=budget_for_profile(profile_id),
    )


# ── The smallest fixture model ────────────────────────────────────────────────


class FloorFixtureEngine:
    """The smallest fixture model in the estate (see module docstring).

    Wraps TF-028's :class:`CoherenceFixtureEngine` — same script,
    same request-line routing, same deterministic answers — and
    adds the one discipline the floor demands: a ``max_tokens``
    cap on the generation params is honoured by word-truncation,
    as a real small model's sampler would honour it, and every
    call's cap is recorded in ``caps_seen``. With no cap set the
    wrapped engine's answer stands untouched.
    """

    def __init__(
        self,
        script: tuple[ScriptTurn, ...] = COHERENCE_SCRIPT,
        drift_from_turn: int | None = None,
    ) -> None:
        self._inner = CoherenceFixtureEngine(
            script=script, drift_from_turn=drift_from_turn
        )
        self.caps_seen: list[int | None] = []

    @property
    def calls(self) -> int:
        return self._inner.calls

    @property
    def prompts(self) -> list[str]:
        return self._inner.prompts

    def generate(self, prompt: str, params: Any = None) -> Any:
        cap = getattr(params, "max_tokens", None)
        self.caps_seen.append(cap)
        result = self._inner.generate(prompt, params)
        if cap is not None and result.tokens_generated > cap:
            words = result.text.split()[:cap]
            result.text = " ".join(words)
            result.tokens_generated = len(words)
        return result


# ── Observations, the checker, and the verdict ────────────────────────────────


def salvage_passes_used(salvage_path: str) -> int:
    """Refine passes a finished turn's salvage path records:
    ``refine_pass_N`` → N; every other path (best_draft,
    knowledge_only, repair, model_failed, empty) ran no salvage
    refine pass at all."""
    prefix = "refine_pass_"
    if salvage_path.startswith(prefix):
        suffix = salvage_path[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    return 0


@dataclass(frozen=True)
class FloorObservation:
    """One finished turn, measured against the floor budget.

    ``prompt_tokens_max`` is the largest generation prompt the
    turn produced (candidates + refine passes), in the tree's
    word-token convention — the context a small model must hold.
    """

    turn_id: str
    response_tokens: int
    prompt_tokens_max: int
    candidate_count: int
    refinement_passes: int
    latency: LatencyVerdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "response_tokens": self.response_tokens,
            "prompt_tokens_max": self.prompt_tokens_max,
            "candidate_count": self.candidate_count,
            "refinement_passes": self.refinement_passes,
            "latency": self.latency.to_dict(),
        }


@dataclass(frozen=True)
class DimensionVerdict:
    """One floor dimension's measured value against its cap."""

    dimension: str
    measured: float
    cap: float

    @property
    def within(self) -> bool:
        return self.measured <= self.cap


@dataclass(frozen=True)
class FloorVerdict:
    """One turn's whole floor verdict."""

    profile_id: str
    dimensions: tuple[DimensionVerdict, ...]

    @property
    def within_floor(self) -> bool:
        return all(d.within for d in self.dimensions)

    @property
    def overruns(self) -> tuple[DimensionVerdict, ...]:
        """The dimensions that broke the floor, in DIMENSIONS order."""
        return tuple(d for d in self.dimensions if not d.within)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "within_floor": self.within_floor,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "measured": d.measured,
                    "cap": d.cap,
                    "within": d.within,
                }
                for d in self.dimensions
            ],
        }


def check_floor_turn(
    observation: FloorObservation, budget: FloorBudget
) -> FloorVerdict:
    """Judge one observed turn against the floor budget — pure.

    Latency is judged as a single dimension reusing TF-033's
    per-stage verdict: its measured/cap pair reports the
    wall-clock total against the budget's summed total (the
    runner always supplies a wall total), and the dimension is
    within exactly when the whole latency verdict is — a single
    stage overrun breaks it even under a within-budget total.
    """
    latency = observation.latency
    total_stage = next(
        (s for s in latency.stages if s.stage == "total"), None
    )
    if total_stage is not None:
        latency_measured, latency_cap = (
            total_stage.measured_ms,
            float(total_stage.budget_ms),
        )
    else:  # pragma: no cover - the runner always supplies a total
        worst = max(latency.stages, key=lambda s: s.measured_ms - s.budget_ms)
        latency_measured, latency_cap = worst.measured_ms, float(worst.budget_ms)
    if not latency.within_budget:
        # A stage overrun is a floor overrun even when the summed
        # total stayed inside: report the dimension as broken by
        # pushing its measured value past the cap (the per-stage
        # detail lives in the latency verdict itself, carried on
        # the observation — nothing is hidden).
        latency_measured = max(latency_measured, latency_cap + 1)
    dimensions = (
        DimensionVerdict(
            "response_tokens",
            float(observation.response_tokens),
            float(budget.response_token_cap),
        ),
        DimensionVerdict(
            "prompt_context",
            float(observation.prompt_tokens_max),
            float(budget.max_context_tokens),
        ),
        DimensionVerdict(
            "drafts",
            float(observation.candidate_count),
            float(budget.draft_count),
        ),
        DimensionVerdict(
            "refinement",
            float(observation.refinement_passes),
            float(budget.max_refinement_passes),
        ),
        DimensionVerdict("latency", latency_measured, latency_cap),
    )
    return FloorVerdict(profile_id=budget.profile_id, dimensions=dimensions)


# ── The runner ────────────────────────────────────────────────────────────────


@dataclass
class FloorRun:
    """A finished floor conversation: per-turn observations and
    verdicts, the transcript, and TF-028's drift report over it."""

    observations: list[FloorObservation] = field(default_factory=list)
    verdicts: list[FloorVerdict] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    report: Any = None

    @property
    def within_floor(self) -> bool:
        return bool(self.verdicts) and all(v.within_floor for v in self.verdicts)


def run_floor_conversation(
    core: Any,
    engine: FloorFixtureEngine,
    budget: FloorBudget,
    script: tuple[ScriptTurn, ...] = COHERENCE_SCRIPT,
    history: Any = None,
) -> FloorRun:
    """Run the coherence script through a phone_low-selected core.

    The caller wires the core (``conversation.profile_id`` =
    phone_low, TF-032) and its generation source (``engine``);
    threading follows TF-028's runner exactly. Per turn, the
    observation is built from the retained turn state (candidate
    count, salvage path), the final's token count, the engine's
    prompts for that turn (context size), and the core's own
    TF-033 latency verdict against a measured wall-clock total.
    """
    from thoughtforge.cognition.chat_history import ChatHistory

    hist = history if history is not None else ChatHistory()
    run = FloorRun()
    for script_turn in script:
        prompts_before = len(engine.prompts)
        hist.add_user(script_turn.user)
        start = time.perf_counter()
        final = core.think(script_turn.user, history=hist)
        wall_ms = (time.perf_counter() - start) * 1000
        hist.add_assistant(final.text, turn_id=final.turn_id)

        turn = core.last_turn_state
        turn_prompts = engine.prompts[prompts_before:]
        observation = FloorObservation(
            turn_id=final.turn_id,
            response_tokens=final.token_count,
            prompt_tokens_max=max(
                (estimate_tokens(p) for p in turn_prompts), default=0
            ),
            candidate_count=len(turn.candidates) if turn else 0,
            refinement_passes=salvage_passes_used(final.salvage_path),
            latency=core.last_turn_latency(total_ms=wall_ms),
        )
        run.observations.append(observation)
        run.verdicts.append(check_floor_turn(observation, budget))
        run.transcript.append((script_turn.user, final.text))
    run.report = score_conversation(run.transcript, script)
    return run
