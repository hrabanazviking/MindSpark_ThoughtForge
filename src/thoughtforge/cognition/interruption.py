"""Interruption & resume — a turn can be cut, and the state stays
coherent.

TF-027 (the Rúnmál voice lesson, applied to text: a speaking turn
that gets talked over does not corrupt the conversation — it stops
cleanly, the partial utterance vanishes whole, and the turn can be
taken up again from exactly where it was cut). Before this slice a
turn had only two fates: complete inside ``think()`` /
``think_stream()``, or vanish with the caller — no cut point, no
record of a cut, nothing resumable. TF-025 gave a turn a phased,
retained state; this slice makes interruption a first-class state
transition on that object.

The discipline pinned here:

  * **Interruption is cooperative.** A caller hands the turn an
    ``InterruptToken`` (thread-safe; it can be fired from another
    thread, or by a stream consumer between events). The core
    checks the token at named *cut points* only — after the
    scaffold is built (before generation), at each candidate and
    chunk boundary inside generation, and after generation
    (before compose). A stage is atomic between cut points: a
    token fired mid-stage is honoured at the next cut point, and
    a token fired during the final compose is not retroactive —
    a completed turn stays completed. Nothing is ever torn
    mid-flight.
  * **A cut keeps exactly what was true.** The turn's phase stays
    at the last completed stage (phases never move backward and
    never claim a stage that did not finish), ``final_response``
    stays None and ``completed_at`` stays empty — no fabricated
    completion. Completed candidates survive on the turn; a
    candidate in flight at the cut is discarded whole (TF-026's
    mid-stream rule, generalised). ``turn.interrupted`` is True,
    with the reason and the phase at the cut recorded.
  * **Being cut is not invisible.** A streamed turn ends with ONE
    terminal ``interrupted`` event instead of ``final``; a
    ``think()`` cut raises ``TurnInterrupted`` carrying the turn.
    A consumer who simply closes a stream mid-turn also cuts it:
    the state records ``consumer closed the stream``.
  * **Resume finishes the SAME turn.** ``resume_turn()`` continues
    the retained interrupted turn — same ``turn_id``, retained
    sketch / bundle / scaffold reused (no re-routing, no
    re-retrieval); generation runs only for the scaffold modes
    that produced no candidate, fragment scoring and compose run
    on the full candidate set, and the turn completes through the
    normal gates. ``resume_count`` records that a resume happened
    (the reason/phase fields describe only the *current* cut and
    are cleared when a resume begins). A turn cut before its
    scaffold existed (only possible when a stream consumer closes
    before generation starts) has nothing to resume *from*:
    resuming it raises ``ValueError`` and the caller starts a new
    turn — the cut state is still coherent, just not resumable.
  * **History never gains a cut turn.** The core appends nothing
    to chat history itself, but callers append from a turn's
    ``FinalResponseRecord`` — a cut turn has none, so no partial
    utterance can enter anyone's history. That is the point.
"""

from __future__ import annotations

import threading

from thoughtforge.knowledge.models import RuntimeTurnState


class TurnInterrupted(Exception):
    """A turn was cut at a cut point by its InterruptToken.

    Carries the retained ``turn`` (the same object as the core's
    ``last_turn_state``) and the ``reason`` the token was fired
    with. Raised out of ``think()`` and ``resume_turn()``; a
    streamed turn reports the same cut as a terminal
    ``interrupted`` event instead of raising.
    """

    def __init__(self, turn: RuntimeTurnState, reason: str = "") -> None:
        self.turn = turn
        self.reason = reason
        super().__init__(
            f"turn {turn.turn_id} interrupted at phase {turn.phase!r}"
            + (f": {reason}" if reason else "")
        )


class InterruptToken:
    """A cooperative cut signal for one (or several) turns.

    ``interrupt()`` is idempotent and safe to call from any
    thread; the first reason given is the one kept. The token is
    one-shot in spirit — a fired token cuts every turn that checks
    it at its next cut point — so a fresh token per turn is the
    normal use.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""

    def interrupt(self, reason: str = "") -> None:
        """Fire the token. Safe from any thread; first reason wins."""
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """The reason passed to the first ``interrupt()`` call."""
        return self._reason


def check_interrupt(turn: RuntimeTurnState, token: InterruptToken | None) -> None:
    """Raise TurnInterrupted if ``token`` has fired.

    The one check every cut point in the core uses. Called only at
    the cut points named in the module docstring — stages between
    cut points run atomically.
    """
    if token is not None and token.is_set:
        raise TurnInterrupted(turn, token.reason)


def mark_interrupted(turn: RuntimeTurnState, reason: str) -> None:
    """Record a cut on the turn: flag, reason, phase at the cut.

    Everything the turn already truthfully holds (phase, sketch,
    bundle, scaffold, completed candidates) is left untouched;
    ``final_response`` / ``completed_at`` are NOT touched — they
    stay empty, because the turn did not complete.
    """
    turn.interrupted = True
    turn.interrupt_reason = reason
    turn.interrupted_phase = turn.phase
