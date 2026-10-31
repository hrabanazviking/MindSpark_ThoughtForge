"""Turn state discipline — the typed turn-state object's phase law.

TF-025. The turn state itself is ``RuntimeTurnState``
(``thoughtforge.knowledge.models`` §14): one typed dataclass carrying
every artefact of a turn — sketch, bundle, scaffold, candidates,
fragments, final response, writeback slot, timestamps, token count,
and the retrieval / generation / salvage timings.

Before TF-025 the core treated that object as a scratch pad: it was
built inside ``think()``, its parts were *also* passed around as
loose positional argument bundles (``_compose_final`` took six,
``_repair`` five), its ``salvage_ms`` field was never assigned, and
the finished object was discarded — no caller could ever see it
(CORE_LOOP §8-F3).

The discipline pinned here:

  * **Phases are data.** ``PHASE_ORDER`` names the seven phases a
    turn passes through, in order. ``RuntimeTurnState.phase`` always
    holds one of them.
  * **Phases only move forward.** ``advance_phase`` is the one way
    the core moves a turn's phase: forward moves apply, re-asserting
    the current phase is a no-op, moving backwards or naming an
    unknown phase raises ``ValueError``. A turn's story can stall,
    never rewrite itself.
  * **The state outlives the call.** The core retains the finished
    turn on ``ThoughtForgeCore.last_turn_state`` (TF-038 owns trace
    *export*; TF-025 owns the state existing to be exported).
  * **Every timing field is real.** ``salvage_ms`` is measured
    around the compose step (salvage + enforcement) whenever
    candidates exist, and is exactly 0 on the knowledge-only path,
    where no salvage runs — 0 means "did not run", never "unmeasured".

Repair is not a phase: it is a branch inside compose → complete,
recorded on ``RuntimeTurnState.repair_attempted``.
"""

from __future__ import annotations

from typing import Literal

from thoughtforge.knowledge.models import RuntimeTurnState

TurnPhase = Literal[
    "created",
    "routed",
    "retrieved",
    "scaffolded",
    "generated",
    "composed",
    "completed",
]

PHASE_ORDER: tuple[str, ...] = (
    "created",
    "routed",
    "retrieved",
    "scaffolded",
    "generated",
    "composed",
    "completed",
)


def phase_index(phase: str) -> int:
    """Position of ``phase`` in PHASE_ORDER; ValueError if unknown."""
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        raise ValueError(f"unknown turn phase: {phase!r}") from None


def advance_phase(turn: RuntimeTurnState, phase: str) -> bool:
    """Move ``turn`` forward to ``phase``.

    Returns True when the phase changed, False when ``phase`` is the
    turn's current phase (a no-op re-assertion). Raises ValueError
    for an unknown phase or a backward move — phases never regress.
    """
    target = phase_index(phase)
    current = phase_index(turn.phase)
    if target < current:
        raise ValueError(
            f"turn phase cannot move backwards: {turn.phase!r} -> {phase!r}"
        )
    if target == current:
        return False
    turn.phase = phase
    return True
