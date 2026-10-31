"""The will layer — goal/intent persistence across turns (TF-029).

The roadmap's promise for ThoughtForge is a small model with
"depth, presence, and **will**". Before TF-029 the forge had no
will at all, in a precise sense:

* the scaffold's goal is re-derived from *this turn's intent alone*
  every turn (``scaffold._INTENT_TO_GOAL``) — a user who spends a
  turn planning a project and the next turn asking a small factual
  question inside it watches the forge forget what the conversation
  is *for*;
* the one record shaped for cross-turn persistence,
  ``ActiveThreadStateRecord`` (topic, open loops, priority,
  aggressive expiry), is loaded by the core each turn and handed to
  the router — which reads only its first 3 tags for memory
  triggers. Nothing ever updates it, carries a goal in it, or lets
  it steer a prompt (CORE_LOOP, TF-019; spec §17's update rules
  were never implemented anywhere).

This module is the minimal will layer those hints specify:

* **A standing goal is data.** :class:`StandingGoal` — the user's
  goal *statement in their own terms*, the topic it rides on, the
  response goal in force when it was set, and its touch bookkeeping.
  At most one goal stands at a time: a conversation pursuing two
  stated goals in parallel is two conversations' worth of state,
  and the older goal is recorded ``superseded``, never silently
  dropped.
* **How a goal comes to stand** (deterministic, no model call —
  the router/scaffold discipline): an explicit declaration in any
  intent ("my goal is …", "the goal is …", "our goal is …",
  "I'm trying to …", "we're trying to …", "I want to …" — the
  captured tail, cut at the sentence end, capped at 120 chars), or
  a *goal-forming intent*: ``planning_request`` and
  ``technical_spec_request``, the two intents that are endeavours
  rather than one-shot asks (brainstorming/naming generate once;
  debugging reacts). A goal-forming turn on a new topic replaces
  the standing goal; on the same topic (token overlap, below) it
  *touches* it.
* **How a goal stays alive**: a later turn whose topic shares a
  content token with the standing goal's topic/statement touches
  it (``turns_since_touch`` resets; tokens get a crude singular
  fold — "lanterns" touches "lantern" — because exact-token
  matching proved too brittle for ordinary plurals). Untouched
  turns count up;
  past ``MAX_UNTOUCHED_TURNS`` (10 — the active-thread record's
  own ``expires_after_turns`` default) the goal goes ``stale``
  and stops steering. Will that nothing feeds is not will, it is
  a stale note — the lifecycle spec's aggressive thread decay,
  applied to goals.
* **How a goal ends**: explicit words close it — completion
  ("that's done", "we did it", "goal achieved/complete/completed",
  "done with …") marks it ``completed``; abandonment ("never
  mind", "forget it", "drop it/that/the goal", "give up on …")
  marks it ``abandoned``. Closing words are checked *before*
  goal-forming, so "never mind the plan" never starts a plan —
  and a closing turn forms no goal even when none stands.
  Closed goals are kept on the ledger's ``closed`` list with
  their final status — the record of what the conversation
  pursued, finished, dropped, or let lapse.
* **How a standing goal steers**: :func:`inject_will` prepends
  one line — ``Standing goal: <statement>`` plus the response
  goal when it differs from the current scaffold's — to the
  scaffold's ``fact_block``, the same injection channel the core
  already uses for chat history (TF-019/TF-024). The scaffold's
  own per-turn goal is never overwritten: the turn's intent
  still steers the turn; the standing goal steers the
  *conversation*. No injection happens on the turn that set the
  goal (that turn's scaffold already carries its response goal;
  persistence is only meaningful from the next turn on), and
  injection never duplicates its own line.
* **Persistence bridge**: the ledger serialises (``to_dict`` /
  ``from_dict``) and round-trips through the existing
  ``ActiveThreadStateRecord`` (statement ↔ ``summary``, topic ↔
  ``topic``, open loops carried through untouched — the will
  layer preserves loops, it does not invent them; loop lifecycle
  is TF-045's), so the store's existing thread-state file can
  carry a standing goal across core instances. The one honest
  loss in that bridge: the thread record has no response-goal
  field, so a goal restored from it steers by statement only.

Core wiring (TF-029, minimal): ``ThoughtForgeCore`` holds one
ledger per instance — a conversation's will — observes every
turn after its scaffold is built, and injects the standing goal
before generation. An interrupted turn still counts as observed:
the user stated their goal even if generation was cut (TF-027).
Resume does not re-observe (it finishes the same turn).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    CognitionScaffold,
    InputSketch,
)

# ── The tables ─────────────────────────────────────────────────────────────────

#: Intents that are endeavours, not one-shot asks: a turn routed to
#: one of these sets (or touches) the standing goal.
GOAL_FORMING_INTENTS: frozenset[str] = frozenset(
    {"planning_request", "technical_spec_request"}
)

#: Untouched turns a standing goal survives — mirrors
#: ActiveThreadStateRecord.expires_after_turns' default (10).
MAX_UNTOUCHED_TURNS = 10

#: A declared goal statement is capped at this many characters.
MAX_GOAL_CHARS = 120

_DECLARATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmy goal is (?:to )?(?P<g>.+)",
        r"\bthe goal is (?:to )?(?P<g>.+)",
        r"\bour goal is (?:to )?(?P<g>.+)",
        r"\bi(?:'m| am) trying to (?P<g>.+)",
        r"\bwe(?:'re| are) trying to (?P<g>.+)",
        r"\bi want to (?P<g>.+)",
    )
)

_COMPLETION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bthat(?:'s| is) done\b",
        r"\bwe did it\b",
        r"\bgoal (?:is )?(?:achieved|complete|completed)\b",
        r"\bdone with\b",
    )
)

_ABANDONMENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bnever mind\b",
        r"\bforget it\b",
        r"\bdrop (?:it|that|the goal)\b",
        r"\bgive up on\b",
    )
)

# The router's stopword discipline (router.py), mirrored small:
# topic overlap is judged on content words only.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "to", "of", "and", "or", "but", "in",
        "on", "at", "by", "for", "with", "about", "what", "who",
        "how", "when", "where", "why", "i", "you", "my", "your",
        "me", "it", "its", "this", "that", "can", "just", "please",
    }
)

GOAL_STATUSES: tuple[str, ...] = (
    "active", "completed", "abandoned", "stale", "superseded",
)


def _content_tokens(text: str) -> set[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens: set[str] = set()
    for w in clean.split():
        if w in _STOPWORDS or len(w) <= 2:
            continue
        tokens.add(w)
        if w.endswith("s") and len(w) > 3:
            tokens.add(w[:-1])  # crude singular fold: lanterns -> lantern
    return tokens


def extract_declaration(raw_text: str) -> str | None:
    """The explicitly declared goal in ``raw_text``, or None.

    The captured tail is cut at the first sentence end and capped
    at MAX_GOAL_CHARS; an empty capture is no declaration.
    """
    for pattern in _DECLARATION_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            text = re.split(r"[.!?]", match.group("g"), maxsplit=1)[0]
            text = " ".join(text.split()).strip(" ,;:")
            if text:
                return text[:MAX_GOAL_CHARS]
    return None


def _closes_with(patterns: tuple[re.Pattern[str], ...], raw_text: str) -> bool:
    return any(p.search(raw_text) for p in patterns)


# ── The standing goal ──────────────────────────────────────────────────────────


@dataclass
class StandingGoal:
    """One goal that persists across turns until it closes."""

    statement: str                     # the goal in the user's terms
    topic: str = ""
    source_intent: str = ""
    response_goal: str = ""            # scaffold goal in force when set
    set_at_turn: int = 0
    last_touched_turn: int = 0
    turns_since_touch: int = 0
    status: str = "active"
    open_loops: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "topic": self.topic,
            "source_intent": self.source_intent,
            "response_goal": self.response_goal,
            "set_at_turn": self.set_at_turn,
            "last_touched_turn": self.last_touched_turn,
            "turns_since_touch": self.turns_since_touch,
            "status": self.status,
            "open_loops": list(self.open_loops),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StandingGoal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── The ledger ─────────────────────────────────────────────────────────────────


class GoalLedger:
    """One conversation's will: the standing goal + its closed past."""

    def __init__(self) -> None:
        self.turn_count = 0
        self.active: StandingGoal | None = None
        self.closed: list[StandingGoal] = []

    # ── Observation ────────────────────────────────────────────────────────

    def observe_turn(self, sketch: InputSketch, response_goal: str) -> StandingGoal | None:
        """Fold one routed turn into the ledger; return the standing goal.

        Order is the law of this module: closing words first, then
        explicit declaration, then goal-forming intents, then the
        ordinary-turn touch/cool bookkeeping.
        """
        self.turn_count += 1
        turn = self.turn_count
        raw = sketch.raw_text or ""

        if _closes_with(_COMPLETION_PATTERNS, raw):
            if self.active is not None:
                self._close("completed")
            return self.active
        if _closes_with(_ABANDONMENT_PATTERNS, raw):
            if self.active is not None:
                self._close("abandoned")
            return self.active

        declared = extract_declaration(raw)
        if declared is not None:
            if self.active is not None and (
                self.active.statement.lower() == declared.lower()
            ):
                self._touch(turn)
            else:
                self._set(
                    StandingGoal(
                        statement=declared,
                        topic=sketch.topic,
                        source_intent=sketch.intent,
                        response_goal=response_goal,
                        set_at_turn=turn,
                        last_touched_turn=turn,
                    )
                )
            return self.active

        if sketch.intent in GOAL_FORMING_INTENTS:
            if self.active is not None and self._overlaps(sketch, self.active):
                self._touch(turn)
            else:
                self._set(
                    StandingGoal(
                        statement=sketch.topic or response_goal,
                        topic=sketch.topic,
                        source_intent=sketch.intent,
                        response_goal=response_goal,
                        set_at_turn=turn,
                        last_touched_turn=turn,
                    )
                )
            return self.active

        # Ordinary turn: touch on topic overlap, otherwise cool.
        if self.active is not None:
            if self._overlaps(sketch, self.active):
                self._touch(turn)
            else:
                self.active.turns_since_touch += 1
                if self.active.turns_since_touch > MAX_UNTOUCHED_TURNS:
                    self._close("stale")
        return self.active

    # ── Steering ───────────────────────────────────────────────────────────

    def will_line(self, current_response_goal: str = "") -> str | None:
        """The one-line steering text for the active goal, or None.

        None when no goal stands, and None on the turn that set
        the goal — that turn's scaffold already carries its
        response goal; persistence means something only later.
        """
        goal = self.active
        if goal is None or self.turn_count <= goal.set_at_turn:
            return None
        line = f"Standing goal: {goal.statement}"
        if goal.response_goal and goal.response_goal != current_response_goal:
            line += f" (pursue it as: {goal.response_goal})"
        return line

    # ── Internals ──────────────────────────────────────────────────────────

    def _set(self, goal: StandingGoal) -> None:
        if self.active is not None:
            self.active.status = "superseded"
            self.closed.append(self.active)
        self.active = goal

    def _touch(self, turn: int) -> None:
        assert self.active is not None
        self.active.last_touched_turn = turn
        self.active.turns_since_touch = 0

    def _close(self, status: str) -> None:
        assert self.active is not None
        self.active.status = status
        self.closed.append(self.active)
        self.active = None

    @staticmethod
    def _overlaps(sketch: InputSketch, goal: StandingGoal) -> bool:
        haystack = _content_tokens(f"{goal.topic} {goal.statement}")
        if not haystack:
            return False
        probe = _content_tokens(sketch.topic) | _content_tokens(sketch.raw_text)
        return bool(probe & haystack)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "active": self.active.to_dict() if self.active else None,
            "closed": [g.to_dict() for g in self.closed],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoalLedger":
        ledger = cls()
        ledger.turn_count = int(d.get("turn_count", 0))
        active = d.get("active")
        ledger.active = StandingGoal.from_dict(active) if active else None
        ledger.closed = [StandingGoal.from_dict(g) for g in d.get("closed", [])]
        return ledger

    # ── ActiveThreadStateRecord bridge ─────────────────────────────────────

    def to_thread_state(self, record_id: str) -> ActiveThreadStateRecord | None:
        """The standing goal as the store's thread-state record.

        None when no goal stands — there is no thread state to
        save. Open loops are carried from the goal untouched.
        """
        goal = self.active
        if goal is None:
            return None
        return ActiveThreadStateRecord(
            record_id=record_id,
            topic=goal.topic,
            summary=goal.statement,
            status="active",
            open_loops=list(goal.open_loops),
            expires_after_turns=max(
                0, MAX_UNTOUCHED_TURNS - goal.turns_since_touch
            ),
            turns_since_touch=goal.turns_since_touch,
        )

    @classmethod
    def from_thread_state(
        cls, record: ActiveThreadStateRecord | None
    ) -> "GoalLedger":
        """Seed a ledger from a saved thread-state record.

        Only an *active* record carrying a summary seeds a goal —
        the summary is the goal statement (this bridge's contract).
        The response goal is not recoverable from the record and
        is left empty: the restored goal steers by statement only.
        """
        ledger = cls()
        if record is None or record.status != "active" or not record.summary:
            return ledger
        ledger.active = StandingGoal(
            statement=record.summary,
            topic=record.topic,
            turns_since_touch=record.turns_since_touch,
            open_loops=list(record.open_loops),
        )
        return ledger


# ── Scaffold injection ─────────────────────────────────────────────────────────


def inject_will(scaffold: CognitionScaffold, ledger: GoalLedger) -> bool:
    """Prepend the standing-goal line to ``scaffold.fact_block``.

    Returns True when the line was added. Never overwrites the
    scaffold's own per-turn goal, never duplicates the line, and
    adds nothing when no goal stands (or on its setting turn).
    """
    line = ledger.will_line(scaffold.goal)
    if line is None:
        return False
    if line in scaffold.fact_block:
        return False
    scaffold.fact_block = (
        f"{line}\n\n{scaffold.fact_block}" if scaffold.fact_block else line
    )
    return True
