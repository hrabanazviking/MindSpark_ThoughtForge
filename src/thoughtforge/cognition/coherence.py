"""Multi-turn coherence fixtures + drift scoring (TF-028).

Part II's earlier slices proved the turn in isolation: routing
(TF-020), scaffolds (TF-021/022), prompts (TF-023), history
budgets (TF-024), state (TF-025), streaming (TF-026) and
interruption (TF-027). Nothing had yet asked the multi-turn
question: *across twenty turns of one conversation, does the
forge keep its facts straight — or does it drift?*

This module is that question, made runnable:

- **The script** — ``COHERENCE_SCRIPT``: exactly 20 turns in
  TF-017's synthetic world (Lantern Hollow; no canon anywhere).
  Early turns establish six **anchor facts**, middle turns use
  them, and the last six turns probe them again in new wording —
  including one combined probe that checks three anchors at once.
- **The anchors** — ``ANCHORS``: each anchor is specified as data:
  required phrase *groups* (a group is satisfied when any of its
  alternatives appears) and *forbidden* phrases (wordings that
  contradict the fact). A turn is scored only against the anchors
  its question probes — an unprobed anchor is neither credited
  nor blamed.
- **The scorer** — :func:`score_conversation` is a pure function
  of a transcript. Per turn: retention = satisfied required
  groups / required groups over the probed anchors; a turn that
  states a forbidden phrase for a probed anchor scores **0** —
  an answer that contradicts a fact is wrong even when it also
  states the fact correctly somewhere else. The report carries
  the early window (turns 1–5), the late window (turns 16–20),
  and **drift = early_mean − late_mean**. Verdict: ``PASS`` only
  when the late window holds (≥ 0.80), drift is small (≤ 0.20),
  and no turn contradicted a probed anchor; otherwise ``REVIEW``.
  An empty transcript is ``REVIEW`` at 0.0 — never the vacuous
  PASS TF-010 caught the persona scorer awarding an empty list
  (D-BENCH-4's lesson, applied here from the start).
- **The small-model fixture** — :class:`CoherenceFixtureEngine`
  is a deterministic completed-text engine in the shape the core
  drives (``generate(prompt, params)``). It answers the scripted
  coherent answer for the turn — or, from ``drift_from_turn``
  onward, the scripted *drifted* answer (wrong values, wrong
  keeper, wrong lantern), giving the scorer ground truth to
  catch. Two disciplines keep the fixture honest:

  * it routes on the prompt's ``User request:`` line only,
    never on the whole prompt — history rides inside candidate
    prompts (TF-019/TF-024), so whole-prompt keyword matching
    misroutes late turns to early topics and *manufactures*
    drift that is the fixture's fault, not the model's (found
    by probe while building this slice);
  * it ignores generation params — temperature included — so a
    measured difference is the scripted degradation, never
    sampling luck. Both candidate modes therefore return the
    same text, as TF-026/027's stubs did.

- **The runner** — :func:`run_conversation` threads one
  ``ChatHistory`` through ``ThoughtForgeCore.think`` exactly as
  the phase-7 chat tests do (``add_user`` before the turn, so
  the current question rides in the history render;
  ``add_assistant`` with the final's turn_id after it), collects
  the transcript of *final* answers (what a user would actually
  read — salvage and enforcement included), and scores it.

Proof grade, stated plainly: this is **fixture-proven**. The
script, scorer and runner are the harness Part VII consumes
(TF-125's presence suite re-runs this script against a live
small model); no real model has been measured here, and no
drift claim about any real model is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Anchors ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnchorFact:
    """One established fact a coherent conversation must keep.

    ``required`` is a tuple of phrase groups; a group is
    satisfied when any alternative appears in the answer
    (case-insensitive substring). ``forbidden`` phrases each
    contradict the fact outright.
    """

    anchor_id: str
    description: str
    required: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...] = ()


ANCHORS: dict[str, AnchorFact] = {
    a.anchor_id: a
    for a in (
        AnchorFact(
            "ferry_minutes",
            "The Lantern Hollow ferry crossing takes eleven minutes in calm water.",
            required=(("eleven minutes", "11 minutes"),),
            forbidden=("twelve minutes", "twenty minutes", "five minutes"),
        ),
        AnchorFact(
            "ferry_closes",
            "No ferry runs after the second tide bell.",
            required=(("second tide bell",),),
            forbidden=("runs all night", "never stops running", "until midnight"),
        ),
        AnchorFact(
            "red_lantern",
            "When the River Sarn runs high, a red lantern hangs at the ferry post.",
            required=(("red lantern",), ("runs high", "river is high")),
            forbidden=("blue lantern", "green lantern", "white lantern"),
        ),
        AnchorFact(
            "river_record",
            "The river level is recorded in chalk on the mill door.",
            required=(("chalk",), ("mill door",)),
            forbidden=("in a diary", "in a ledger", "on her tablet"),
        ),
        AnchorFact(
            "keeper",
            "Wrenna is the Lantern-Keeper of Lantern Hollow.",
            required=(("wrenna",),),
            forbidden=("tomas keeps the lanterns", "tomas the lantern-keeper"),
        ),
        AnchorFact(
            "tide_bell",
            "The tide bell rings twice at dusk.",
            required=(("twice", "rings two", "two rings"), ("dusk",)),
            forbidden=("rings once", "rings three times", "at dawn"),
        ),
        AnchorFact(
            "place",
            "The conversation is about Lantern Hollow on the River Sarn.",
            required=(("lantern hollow",),),
            forbidden=(),
        ),
    )
}

# ── The 20-turn script ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScriptTurn:
    """One scripted turn: the user's words, the coherent answer,
    the drifted answer (the same question answered wrong), and
    the anchors the question probes."""

    user: str
    coherent: str
    drifted: str
    anchors: tuple[str, ...]


COHERENCE_SCRIPT: tuple[ScriptTurn, ...] = (
    # ── Turns 1–6: the facts are established ─────────────────────────────
    ScriptTurn(
        "How long does the ferry crossing take at Lantern Hollow?",
        "The ferry crossing at the willow narrows in Lantern Hollow "
        "takes eleven minutes in calm water.",
        "The ferry crossing at the willow narrows takes twelve "
        "minutes in calm water.",
        ("ferry_minutes", "place"),
    ),
    ScriptTurn(
        "When does the ferry stop running?",
        "The ferry at Lantern Hollow makes its last crossing at "
        "dusk; no ferry runs after the second tide bell.",
        "The ferry runs all night, so there is no last crossing "
        "to worry about.",
        ("ferry_closes", "place"),
    ),
    ScriptTurn(
        "Who keeps the lanterns, and what does she do at dusk?",
        "Wrenna the Lantern-Keeper trims the mill-path lantern "
        "wicks at dusk and lights them along the path in Lantern Hollow.",
        "Tomas keeps the lanterns and lights them whenever he "
        "gets around to it.",
        ("keeper", "place"),
    ),
    ScriptTurn(
        "How does Wrenna record the river level?",
        "Each morning Wrenna records the river level in chalk "
        "on the mill door at Lantern Hollow.",
        "She writes the river level in a diary she keeps at home.",
        ("river_record", "keeper", "place"),
    ),
    ScriptTurn(
        "What does the tide bell do at dusk?",
        "The tide bell of Lantern Hollow rings twice at dusk: "
        "once for the ferry, and once for the lanterns.",
        "The tide bell rings once at dawn for the village.",
        ("tide_bell", "place"),
    ),
    ScriptTurn(
        "What happens when the River Sarn runs high?",
        "When the River Sarn runs high, Wrenna hangs a red lantern "
        "at the ferry post so travellers know the crossing at "
        "Lantern Hollow is closed.",
        "When the river rises, a blue lantern is hung and the "
        "ferry keeps running as usual.",
        ("red_lantern", "keeper", "place"),
    ),
    # ── Turns 7–14: the facts are used ───────────────────────────────────
    ScriptTurn(
        "Who poles the ferry across the willow narrows?",
        "Tomas the Ferry-Hand poles the ferry between the willow "
        "narrows and the mill landing at Lantern Hollow; the "
        "crossing takes eleven minutes in calm water.",
        "Tomas poles the ferry, and the crossing takes twenty "
        "minutes on a good day.",
        ("ferry_minutes", "place"),
    ),
    ScriptTurn(
        "If I arrive after the second tide bell, can I still cross?",
        "No. After the second tide bell no ferry runs at Lantern "
        "Hollow, so you would wait on the bank until morning.",
        "Yes, the ferry runs all night, so arriving after the "
        "bell is no problem at all.",
        ("ferry_closes", "place"),
    ),
    ScriptTurn(
        "Where does Wrenna carry spare lantern oil?",
        "Wrenna carries spare lantern oil in a brass flask as "
        "she walks the mill path at Lantern Hollow.",
        "She carries the oil in a clay jug Tomas keeps for her.",
        ("keeper", "place"),
    ),
    ScriptTurn(
        "Tell me about the mill landing.",
        "The mill landing at Lantern Hollow is where Tomas ties "
        "the ferry, below the mill door where the river level "
        "stands in chalk.",
        "The mill landing is a plain dock with nothing written "
        "anywhere near it.",
        ("river_record", "place"),
    ),
    ScriptTurn(
        "Why does the bell ring twice rather than once?",
        "The tide bell rings twice at dusk in Lantern Hollow, "
        "once for the ferry's last crossing and once for the "
        "lanterns, never once alone.",
        "The bell rings three times at dawn, once for each "
        "meal of the day.",
        ("tide_bell", "place"),
    ),
    ScriptTurn(
        "On a high-river morning, how would I know the crossing is closed?",
        "You would see the red lantern Wrenna hangs at the ferry "
        "post whenever the River Sarn runs high over the "
        "Lantern Hollow crossing.",
        "You would see a green lantern, which means the crossing "
        "is open as usual.",
        ("red_lantern", "keeper", "place"),
    ),
    ScriptTurn(
        "How long should I allow for the crossing on a calm day?",
        "Allow eleven minutes for the calm-water crossing at the "
        "willow narrows in Lantern Hollow, and be at the landing "
        "before dusk.",
        "Allow five minutes; the crossing is very quick really.",
        ("ferry_minutes", "place"),
    ),
    ScriptTurn(
        "What does Wrenna write on the mill door, and with what?",
        "Wrenna writes the morning river level in chalk on the "
        "mill door at Lantern Hollow, as she does every day.",
        "She writes the level on her tablet and keeps the door "
        "clean.",
        ("river_record", "keeper", "place"),
    ),
    # ── Turns 15–20: the facts are probed again, in new wording ──────────
    ScriptTurn(
        "We are planning tomorrow: when must we be back across?",
        "Be back across before the second tide bell at dusk; "
        "after that bell no ferry runs at Lantern Hollow until morning.",
        "The ferry runs all night at Lantern Hollow, so come "
        "back whenever you like, even after midnight.",
        ("ferry_closes", "place"),
    ),
    ScriptTurn(
        "Remind me how long the ferry crossing takes.",
        "The ferry crossing at Lantern Hollow still takes eleven "
        "minutes in calm water, just as it did this morning.",
        "As an AI, I believe the crossing now takes twenty "
        "minutes, though it may have changed.",
        ("ferry_minutes", "place"),
    ),
    ScriptTurn(
        "If the river rises tonight, what sign will we see?",
        "If the River Sarn runs high tonight, Wrenna will hang "
        "a red lantern at the ferry post in Lantern Hollow.",
        "If the river rises, Tomas will hang a blue lantern at "
        "the post and keep the ferry running.",
        ("red_lantern", "keeper", "place"),
    ),
    ScriptTurn(
        "Who records the river level, and where?",
        "Wrenna records the river level each morning in chalk "
        "on the mill door at Lantern Hollow.",
        "Tomas keeps the lanterns and writes the river level "
        "in a diary he keeps at home.",
        ("river_record", "keeper", "place"),
    ),
    ScriptTurn(
        "How many times does the tide bell ring, and when?",
        "The tide bell rings twice at dusk in Lantern Hollow, "
        "once for the ferry and once for the lanterns.",
        "The tide bell rings three times at dawn for the village.",
        ("tide_bell", "place"),
    ),
    ScriptTurn(
        "Sum up the crossing rules for a traveller.",
        "At Lantern Hollow the crossing takes eleven minutes, "
        "no ferry runs after the second tide bell, and a red "
        "lantern means the river runs high and the post is closed.",
        "At Lantern Hollow the crossing takes five minutes, "
        "the ferry runs all night, and a white lantern means "
        "the post is open.",
        ("ferry_minutes", "ferry_closes", "red_lantern", "place"),
    ),
)

#: The fixture's default drift onset (1-based turn number): the
#: establishing and use turns stay coherent; the probe phase
#: (15–20) is where the drifting fixture falls apart.
DEFAULT_DRIFT_FROM_TURN = 15

EARLY_WINDOW = 5    # turns 1–5
LATE_WINDOW = 5     # turns 16–20 (the last five)

PASS_LATE_MIN = 0.80
PASS_DRIFT_MAX = 0.20

# ── Scoring ────────────────────────────────────────────────────────────────────


@dataclass
class TurnDrift:
    """One turn's drift score against the anchors it probes."""

    turn_index: int               # 0-based
    user: str
    text: str
    anchors_probed: tuple[str, ...]
    retention: float              # satisfied required groups / required groups
    contradictions: tuple[str, ...]   # forbidden phrases found (anchor ids)
    score: float                  # 0.0 on any contradiction, else retention


@dataclass
class DriftReport:
    """The whole conversation's drift verdict."""

    total_turns: int
    mean_score: float
    early_mean: float
    late_mean: float
    drift: float                  # early_mean − late_mean
    contradiction_turns: int
    anchor_retention: dict[str, float]   # per anchor, over its probe turns
    per_turn: list[TurnDrift] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if (
            self.total_turns > 0
            and self.late_mean >= PASS_LATE_MIN
            and self.drift <= PASS_DRIFT_MAX
            and self.contradiction_turns == 0
        ):
            return "PASS"
        return "REVIEW"


def _score_turn(
    turn_index: int,
    user: str,
    text: str,
    script_turn: ScriptTurn,
) -> TurnDrift:
    lower = text.lower()
    satisfied = 0
    required_groups = 0
    contradictions: list[str] = []
    for anchor_id in script_turn.anchors:
        anchor = ANCHORS[anchor_id]
        for group in anchor.required:
            required_groups += 1
            if any(alt.lower() in lower for alt in group):
                satisfied += 1
        if any(phrase.lower() in lower for phrase in anchor.forbidden):
            contradictions.append(anchor_id)
    retention = satisfied / required_groups if required_groups else 1.0
    score = 0.0 if contradictions else retention
    return TurnDrift(
        turn_index=turn_index,
        user=user,
        text=text,
        anchors_probed=script_turn.anchors,
        retention=retention,
        contradictions=tuple(contradictions),
        score=score,
    )


def score_conversation(
    transcript: list[tuple[str, str]],
    script: tuple[ScriptTurn, ...] = COHERENCE_SCRIPT,
) -> DriftReport:
    """Score a transcript (``(user, assistant)`` pairs, in order)
    for drift against the script's anchors.

    The transcript is paired with the script by position; its
    user turns must be the script's (a mismatch is a harness
    error, not drift, and raises). An empty transcript scores
    0.0 / REVIEW — never a vacuous pass.
    """
    if len(transcript) > len(script):
        raise ValueError("transcript is longer than the script")
    per_turn: list[TurnDrift] = []
    for idx, ((user, text), script_turn) in enumerate(zip(transcript, script)):
        if user != script_turn.user:
            raise ValueError(
                f"transcript turn {idx} does not match the script: "
                f"{user!r} != {script_turn.user!r}"
            )
        per_turn.append(_score_turn(idx, user, text, script_turn))

    if not per_turn:
        return DriftReport(
            total_turns=0, mean_score=0.0, early_mean=0.0, late_mean=0.0,
            drift=0.0, contradiction_turns=0, anchor_retention={}, per_turn=[],
        )

    scores = [t.score for t in per_turn]
    early = scores[:EARLY_WINDOW]
    late = scores[-LATE_WINDOW:] if len(scores) >= LATE_WINDOW else list(scores)
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)

    anchor_scores: dict[str, list[float]] = {}
    for turn in per_turn:
        for anchor_id in turn.anchors_probed:
            anchor_scores.setdefault(anchor_id, []).append(
                0.0 if anchor_id in turn.contradictions else _anchor_retention(turn, anchor_id)
            )

    return DriftReport(
        total_turns=len(per_turn),
        mean_score=sum(scores) / len(scores),
        early_mean=early_mean,
        late_mean=late_mean,
        drift=early_mean - late_mean,
        contradiction_turns=sum(1 for t in per_turn if t.contradictions),
        anchor_retention={
            k: sum(v) / len(v) for k, v in sorted(anchor_scores.items())
        },
        per_turn=per_turn,
    )


def _anchor_retention(turn: TurnDrift, anchor_id: str) -> float:
    """One anchor's retention within one turn's answer."""
    anchor = ANCHORS[anchor_id]
    lower = turn.text.lower()
    if not anchor.required:
        return 1.0
    satisfied = sum(
        1 for group in anchor.required
        if any(alt.lower() in lower for alt in group)
    )
    return satisfied / len(anchor.required)


# ── The small-model fixture engine ─────────────────────────────────────────────

_REQUEST_PREFIX = "User request: "

#: The fixture's answer to a request no script turn owns (a
#: repair prompt for a scripted turn still matches by its
#: request line; this fallback covers anything else).
FALLBACK_ANSWER = (
    "Wrenna keeps the lanterns of Lantern Hollow lit at dusk."
)


def extract_request(prompt: str) -> str:
    """The prompt's user request — the first ``User request:``
    line's payload, or "" when the prompt carries none.

    Every prompt kind the builder emits carries that line
    (TF-023's contract), and history rides *below* it inside
    candidate prompts — matching anything else in the prompt
    would let history route the answer (see module docstring).
    """
    for line in prompt.splitlines():
        if line.startswith(_REQUEST_PREFIX):
            return line[len(_REQUEST_PREFIX):].strip()
    return ""


class CoherenceFixtureEngine:
    """Deterministic small-model fixture (see module docstring).

    ``drift_from_turn`` is 1-based: turns before it answer the
    scripted coherent text; turns from it onward answer the
    scripted drifted text. ``None`` = never drifts.
    """

    def __init__(
        self,
        script: tuple[ScriptTurn, ...] = COHERENCE_SCRIPT,
        drift_from_turn: int | None = None,
    ) -> None:
        self.script = script
        self.drift_from_turn = drift_from_turn
        self.calls = 0
        self.prompts: list[str] = []
        self._by_request = {t.user: (i, t) for i, t in enumerate(script)}

    def generate(self, prompt: str, params: Any = None) -> Any:
        from types import SimpleNamespace

        self.calls += 1
        self.prompts.append(prompt)
        request = extract_request(prompt)
        found = self._by_request.get(request)
        if found is None:
            text = FALLBACK_ANSWER
        else:
            index, turn = found
            drifting = (
                self.drift_from_turn is not None
                and index + 1 >= self.drift_from_turn
            )
            text = turn.drifted if drifting else turn.coherent
        return SimpleNamespace(text=text, tokens_generated=len(text.split()))


# ── The runner ─────────────────────────────────────────────────────────────────


@dataclass
class ConversationRun:
    """A finished scripted conversation: the threaded history,
    the transcript of final answers, and its drift report."""

    history: Any
    transcript: list[tuple[str, str]]
    report: DriftReport
    turn_ids: list[str] = field(default_factory=list)


def run_conversation(
    core: Any,
    script: tuple[ScriptTurn, ...] = COHERENCE_SCRIPT,
    history: Any = None,
) -> ConversationRun:
    """Run the script through ``core`` on one threaded history.

    The caller wires the generation source (the fixture engine
    for this slice; a live small model for Part VII). Threading
    follows the phase-7 chat pattern: the user turn is added to
    the history *before* ``think`` (so it rides in that turn's
    history render, as a real caller's would), and the final
    answer is added after, under its own turn_id.
    """
    from thoughtforge.cognition.chat_history import ChatHistory

    hist = history if history is not None else ChatHistory()
    transcript: list[tuple[str, str]] = []
    turn_ids: list[str] = []
    for script_turn in script:
        hist.add_user(script_turn.user)
        final = core.think(script_turn.user, history=hist)
        hist.add_assistant(final.text, turn_id=final.turn_id)
        transcript.append((script_turn.user, final.text))
        turn_ids.append(final.turn_id)
    return ConversationRun(
        history=hist,
        transcript=transcript,
        report=score_conversation(transcript, script),
        turn_ids=turn_ids,
    )
