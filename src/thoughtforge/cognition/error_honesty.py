"""Error honesty — what a failed turn is allowed to say (TF-031).

Before TF-031 the core degraded *silently*. The failure points
were all logged and then swallowed:

* the model failing to load, a candidate generation raising, and
  the unified backend returning an error each produced an empty
  candidate list, and an empty candidate list produced the
  knowledge-only response — presented exactly like a deliberate
  no-model answer. When the store was also empty the user was
  told "The forge has no knowledge relevant to this query", which
  blamed the *knowledge* for what the *model* did;
* when one of the two candidate modes failed and the other
  survived, salvage composed a normal-looking answer from the
  survivor alone — the user saw a complete answer and had no way
  to know half the generation never happened;
* an engine returning empty text was appended as an *empty
  candidate* — a completion that contains nothing, scored and
  carried as if the model had answered.

This module is the honesty contract that replaces that:

* **Failures are data.** :class:`GenerationFailure` records one
  failure — stage, mode, and the exception's class + message
  (truncated at ``MAX_DETAIL_CHARS``) — and the core appends one
  to the turn state (``RuntimeTurnState.generation_failures``) at
  every swallow point: model load, per-mode candidate generation
  (plain and streamed), the unified backend (plain and streamed,
  including its ``GenerationResponse.error`` channel), an empty
  completed text (a non-answer is a failure, not a candidate),
  and the quality-gate repair pass. Nothing that is caught is
  left only in the log any more.
* **A model that was never configured is not a failure.**
  Knowledge-only is a legitimate, deliberate mode (TF-005 proved
  its empty-store answer honest in fact). Failures exist only
  where generation was *attempted* — an engine configured or a
  backend present — and did not produce. A core with no model
  path, no engine and no backend records nothing, and its
  knowledge-only answers are byte-identical to before TF-031.
* **Total failure is stated, never dressed up.** When generation
  was attempted and produced no candidate, the user sees a stated
  failure: :func:`failure_text` says the model failed, names the
  stage(s) and the error, and says plainly that no model answer
  was generated. When retrieved knowledge exists it may still be
  shown — but *labelled* as retrieved knowledge under the stated
  failure (:func:`degraded_knowledge_text`), never presented as
  the model's answer. The final record's mode is ``"failed"``
  when nothing at all can be shown, and its scores are the
  knowledge-only empty shape, not invented quality.
* **Partial failure is disclosed on the answer itself.** When at
  least one candidate survived, the composed answer stands — it
  is real model output — with :func:`partial_disclosure` appended:
  which mode(s) failed and that the answer was composed from the
  surviving candidate(s) only. Disclosure is appended exactly
  once, after the quality-gate repair decision, to whichever
  final text stands; the record's token count is recomputed.
* **Repair failure is recorded, not announced.** The repair pass
  is an optional improvement attempt over an already-composed
  answer; when it fails the user still holds exactly the real
  composed answer — nothing was fabricated and nothing was lost,
  so the failure rides the turn record (and TF-038's trace) but
  does not tax the answer's text. Every *generation* failure, by
  contrast, is always user-visible, because it changes what the
  answer is.
* **Interruption is not failure.** A cut turn (TF-027) raises
  ``TurnInterrupted`` and records nothing here — a cut is the
  caller's own act, already stated by the ``interrupted`` event /
  exception, and resume generates only the missing modes, whose
  own failures, if any, are recorded then.

Streamed turns (TF-026) already emitted ``candidate_error``
events per failed mode; TF-031 keeps those and makes the
terminal ``final`` event honest too, because a consumer that
only reads the final answer must not receive a silently
fabricated completion either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thoughtforge.knowledge.models import FinalResponseRecord

# ── The tables ─────────────────────────────────────────────────────────────────

#: Exception detail carried on a failure record (class + message).
#: Bounded like every other carried string in the cognition layer:
#: a failure explains itself, it does not dump a traceback into
#: the user's answer.
MAX_DETAIL_CHARS = 160

#: Failure stages, in pipeline order. ``model_load`` — the engine
#: could not be loaded; ``candidate`` — one local-engine candidate
#: mode (plain or streamed); ``backend`` — the unified backend
#: (plain or streamed, exception or its error channel);
#: ``repair`` — the quality-gate repair pass.
FAILURE_STAGES: tuple[str, ...] = (
    "model_load",
    "candidate",
    "backend",
    "repair",
)

#: Marker that opens every user-visible failure disclosure. Its
#: presence in a final text is the already-disclosed guard: a
#: disclosure is appended exactly once.
DISCLOSURE_MARKER = "Degraded turn (TF-031):"


def failure_detail(exc: BaseException) -> str:
    """One failure's detail string: ``ClassName: message``, capped."""
    detail = f"{type(exc).__name__}: {exc}".strip()
    if detail.endswith(":"):
        detail = type(exc).__name__
    if len(detail) > MAX_DETAIL_CHARS:
        detail = detail[: MAX_DETAIL_CHARS - 1] + "…"
    return detail


# ── The failure record ─────────────────────────────────────────────────────────


@dataclass
class GenerationFailure:
    """One generation failure, as data — what failed, where, how."""

    stage: str
    detail: str
    mode: str = ""

    def __post_init__(self) -> None:
        if self.stage not in FAILURE_STAGES:
            raise ValueError(f"unknown failure stage: {self.stage!r}")
        if len(self.detail) > MAX_DETAIL_CHARS:
            self.detail = self.detail[: MAX_DETAIL_CHARS - 1] + "…"

    @classmethod
    def from_exception(
        cls, stage: str, exc: BaseException, mode: str = ""
    ) -> "GenerationFailure":
        return cls(stage=stage, mode=mode, detail=failure_detail(exc))

    def describe(self) -> str:
        """One human-readable clause: stage, mode when any, detail."""
        where = self.stage if not self.mode else f"{self.stage} (mode {self.mode})"
        return f"{where}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "mode": self.mode, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationFailure":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def turn_failures(turn: Any) -> list[GenerationFailure]:
    """The failures recorded on a turn state (typed view of the list)."""
    return list(getattr(turn, "generation_failures", []) or [])


def generation_failures(failures: list[GenerationFailure]) -> list[GenerationFailure]:
    """The user-visible subset: every failure except repair's.

    Repair failure is recorded but not announced (module docstring):
    the composed answer it failed to improve is real and complete.
    """
    return [f for f in failures if f.stage != "repair"]


# ── The user-visible texts ─────────────────────────────────────────────────────


def _failure_clauses(failures: list[GenerationFailure]) -> str:
    return "; ".join(f.describe() for f in failures)


def failure_text(failures: list[GenerationFailure]) -> str:
    """The stated failure for a turn with nothing honest to show.

    Names the failure(s) and states plainly that no model answer
    was generated — the anti-fabrication sentence of this slice.
    Raises ValueError on an empty list: a stated failure with no
    failure behind it would itself be a fabrication.
    """
    visible = generation_failures(failures)
    if not visible:
        raise ValueError("no generation failure to state")
    return (
        f"{DISCLOSURE_MARKER} the model failed during this turn "
        f"({_failure_clauses(visible)}). No model answer was generated, "
        "and no retrieved knowledge is available to show instead — "
        "nothing here is presented as an answer."
    )


def degraded_knowledge_text(failures: list[GenerationFailure], body: str) -> str:
    """A stated failure heading the retrieved-knowledge fallback.

    The knowledge body is shown for what it is — retrieved records,
    not a model-written answer — under the failure that caused the
    fallback. Raises ValueError on an empty failure list.
    """
    visible = generation_failures(failures)
    if not visible:
        raise ValueError("no generation failure to state")
    return (
        f"{DISCLOSURE_MARKER} the model failed during this turn "
        f"({_failure_clauses(visible)}), so no model answer was "
        "generated. What follows is retrieved knowledge only, shown "
        f"as retrieved — not a model-written answer.\n\n{body}"
    )


def partial_disclosure(failures: list[GenerationFailure]) -> str:
    """The disclosure appended to an answer composed from survivors.

    Raises ValueError on an empty (or repair-only) list: an
    undisclosed-failure-free answer gets no disclosure.
    """
    visible = generation_failures(failures)
    if not visible:
        raise ValueError("no generation failure to disclose")
    return (
        f"{DISCLOSURE_MARKER} candidate generation failed "
        f"({_failure_clauses(visible)}). This answer was composed "
        "from the surviving candidate(s) only."
    )


def apply_partial_disclosure(
    final: FinalResponseRecord, failures: list[GenerationFailure]
) -> bool:
    """Append the partial-failure disclosure to a composed final.

    Returns True when appended. No-ops (False) when there is no
    user-visible failure or the disclosure is already present —
    exactly once, however many paths reach for it. The record's
    token count is recomputed over the disclosed text, and the
    failure is named in the record's enforcement notes.
    """
    visible = generation_failures(failures)
    if not visible:
        return False
    if DISCLOSURE_MARKER in final.text:
        return False
    final.text = f"{final.text}\n\n{partial_disclosure(visible)}"
    final.token_count = len(final.text.split())
    note = f"degraded turn: {_failure_clauses(visible)}"
    final.enforcement_notes = (
        f"{final.enforcement_notes}; {note}" if final.enforcement_notes else note
    )
    return True
