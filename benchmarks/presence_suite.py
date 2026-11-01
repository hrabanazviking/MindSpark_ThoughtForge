"""
Presence suite, as data (TF-125).

The roadmap's rule: *presence suite: long-conversation
coherence + the will layer's follow-through (TF-029)
scored over scripted sagas.* TF-121's harness filed
the architecture, TF-122 built the persona suite,
TF-123 the memory suite, TF-124 the depth suite.
This slice is the presence consumer — and it closes
the harness's filed will gap: TF-121 registered the
rule ``the_will_axis_has_no_scorer_yet`` with TF-125
as its owner, and this suite is that scorer.

The measured starting state, probed live this tick
on this tree before anything was built:

  * **No presence / coherence / will scorer existed
    in ``benchmarks/``.** The estate held the persona
    scorer + suite, the memory suite, the depth suite
    and the profile runner; nothing scored whether a
    conversation keeps its facts over many turns, and
    nothing scored whether a standing goal actually
    steers later turns or stops steering when it
    closes. TF-029's steering line existed and the
    core injected it (``cognition/core.py`` holds one
    ``GoalLedger`` per instance and calls
    ``inject_will`` before generation — grepped live),
    but follow-through was asserted by design and
    measured by nothing.
  * **The coherence machinery is real — probed.**
    TF-028's ``score_conversation`` over its filed
    20-turn script (7 anchors) scores the coherent
    transcript mean 1.000 / PASS and the scripted
    drift-from-turn-15 transcript mean 0.700 /
    REVIEW (6 contradiction turns) — re-measured
    live this tick, agreeing with TF-028's filing.
    TF-063's 100-turn proof is a different leg: it
    carries no chat history and measures the store
    channel, not presence or will.
  * **The will machinery is real — probed.** TF-029's
    ``GoalLedger`` sets a goal from a declaration in
    any intent or from the two goal-forming intents,
    touches it on topic overlap, stales it after
    ``MAX_UNTOUCHED_TURNS`` (10) untouched turns, and
    closes it completed / abandoned on the filed
    closing words, recording superseded goals on the
    closed list (TF-029's own tests pin each leg).
    What no probe had ever done is run a scripted
    conversation's *answers* against the ledger and
    score whether they followed through.
  * **The gap was filed in the harness but never
    numbered in the docket** — the D-BENCH-9 /
    D-BENCH-10 pattern. TF-121's harness rule named
    the missing will scorer and TF-028's doc assigned
    its script's Part VII consumption to TF-125, but
    no TF-016 entry carried the gap. Filed as
    D-BENCH-11 and fixed by this slice in the
    measuring sense (see below).

What this module files, as verifiable data:

  * ``COHERENCE_SAGAS`` — 12 scripted sagas over
    TF-028's filed 20-turn script: each saga files a
    drift shape (a drift onset, a set of drifted
    turns, and/or a truncation) over the script's
    coherent / drifted answers, and its filed
    expectation. A saga holds exactly when TF-028's
    real ``score_conversation`` reports a whole-saga
    mean at or above ``HOLD_MIN_SCORE`` — TF-028's
    own ``PASS_LATE_MIN`` (0.80), reused as the
    suite's floor. TF-028's stricter conversation
    verdict (late window + zero contradictions) is
    carried per saga in the report data, and the two
    disagree on filed sagas by design (see the doc):
    this suite measures retention over the saga.
  * ``STEERING_SAGAS`` / ``CLOSURE_SAGAS`` — 24
    scripted sagas (12 per axis) of user turns with
    filed intents / topics and scripted assistant
    answers. The judge replays every user turn
    through a fresh real ``GoalLedger`` (TF-029):
    on any turn where the ledger's real
    ``will_line`` stands, the scripted answer must
    carry the standing goal's statement; on any
    turn, an answer carrying a *closed* goal's
    statement (completed, abandoned, stale or
    superseded) is a violation — a dead goal must
    never steer again. Each saga also files its
    expected end state (the active statement, or
    None, and the closed statuses in order), and a
    saga holds exactly when there are no violations
    and the ledger's end state matches the filing.
  * ``run_presence_suite()`` — per-axis scores (the
    fraction of filed sagas held) plus the overall
    (their mean), PASS / REVIEW by the filed
    threshold: PASS only on a non-empty run with
    every axis at or above ``AXIS_PASS_THRESHOLD``
    (0.75, the estate's shape). An empty axis or an
    empty suite scores 0.0 / REVIEW — no vacuous
    pass (the TF-122 / TF-123 / TF-124 precedent).

The honesty shapes are filed by name, as the task
names them: coherence breaks fail (drift from turn
15 — TF-028's canonical drifting run — and from
turn 16, and a truncated saga that drifts from
turn 12); a goal dropped mid-saga fails; a goal
contradicted by the answer fails; and the closure
axis files the still-steered failures — a goal
abandoned but still steered, a goal completed but
still steered, and a superseded goal still steered
in place of its successor all score held=False.

D-BENCH-11 is FIXED by this slice in the measuring
sense: presence and will follow-through now have a
mechanical scorer with a pinned harness baseline.
What is NOT fixed, and is stated in the filed
limits: the sagas' answers are scripted — a live
small model's presence over real long
conversations, and its answers' follow-through
against the ledger the core maintains for it, are
NOT PROVEN here (TF-126's road).

Filed limits, stated plainly:

  * The sagas are scripted synthetic fixtures
    (TF-017): TF-028's invented Lantern Hollow world
    for the coherence axis and invented ferry /
    harbour talk for the will axes, never canon,
    never a model's live output, and ``data/`` was
    never opened. The suite proves the *scoring*
    end to end through TF-028's and TF-029's real
    machinery.
  * The coherence hold rule is the whole-saga mean
    at TF-028's 0.80 floor. TF-028's own verdict is
    stricter (it additionally requires the late
    window at 0.80, drift at most 0.20, and zero
    contradictions); six filed sagas hold on the
    mean while TF-028's verdict would REVIEW them,
    and they are named in the doc — the suite files
    both numbers and claims only the mean leg.
  * The will axes judge the scripted answers
    against the ledger replayed from the filed user
    turns with filed intents / topics (the router
    is not re-run; TF-020 / TF-029 own routing).
    Steering is judged by the standing statement's
    presence in the answer — the deterministic,
    checkable shape of the core's own injection
    line — not by semantic fidelity to the goal.

The human-readable write-up is
``docs/architecture/PRESENCE_SUITE.md``.
Proof grade: FIXTURE — synthetic scripted sagas
through the real TF-028 / TF-029 machinery on this
VM; no live model, never claimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PRESENCE_SUITE_FORMAT = "thoughtforge-presence-suite/1"
PRESENCE_SUITE_DOC = "docs/architecture/PRESENCE_SUITE.md"

#: Per-axis pass threshold (the estate's filed 0.75,
#: TF-122 / TF-123 / TF-124's shape) and the filed
#: corpus scale: 12 sagas per axis, 36 in total.
AXIS_PASS_THRESHOLD = 0.75
CASES_PER_AXIS = 12

# The real machinery, reused — never copied: TF-028's
# script, anchors, scorer and fixture engine judge the
# coherence axis; TF-029's ledger, closing/declaration
# discipline and injection judge the will axes.
from thoughtforge.cognition.coherence import (  # noqa: E402
    ANCHORS,
    COHERENCE_SCRIPT,
    PASS_LATE_MIN,
    CoherenceFixtureEngine,
    score_conversation,
)
from thoughtforge.cognition.will import (  # noqa: E402
    MAX_UNTOUCHED_TURNS,
    GoalLedger,
    inject_will,
)
from thoughtforge.knowledge.models import (  # noqa: E402
    CognitionScaffold,
    InputSketch,
)

#: A coherence saga holds when TF-028's report mean is
#: at or above this floor — TF-028's own PASS_LATE_MIN
#: (0.80), reused, never re-invented.
HOLD_MIN_SCORE = PASS_LATE_MIN

#: The response goal every will saga's turns are
#: observed under (the scaffold goal in force). The
#: suite passes the same value when asking the ledger
#: for its steering line, so the line under judgement
#: is the bare ``Standing goal: <statement>`` shape
#: the core injects (TF-029).
RESPONSE_GOAL = "give an accurate, direct answer"


def _norm(text: str) -> str:
    """The suite's phrase normalisation (TF-122's shape)."""
    clean = re.sub(r"[^\w\s]", " ", str(text).lower())
    return " ".join(clean.split())


def _contains(text_norm: str, phrase: str) -> bool:
    return f" {_norm(phrase)} " in f" {text_norm} "


# ── The coherence axis ──────────────────────────────────────────────────────
#
# Twelve sagas over TF-028's filed 20-turn script.
# Each files a drift shape over the script's coherent
# / drifted answers: drift_onset (every turn from it
# answers drifted), drift_turns (only those turns
# answer drifted), upto (the saga stops after that
# many script turns). Filed shape, measured live at
# banking: 9 hold (the coherent run, the late onsets
# 17–20, the isolated middle / early slips, the
# truncated coherent run) and 3 fail by design (the
# canonical onset-15 break, onset 16, and a truncated
# saga drifting from turn 12).

COHERENCE_SAGAS: tuple[dict[str, Any], ...] = (
    {"id": "k01_coherent_full", "drift_onset": None,
     "drift_turns": (), "upto": None, "holds": True},
    {"id": "k02_drift_from_20", "drift_onset": 20,
     "drift_turns": (), "upto": None, "holds": True},
    {"id": "k03_drift_from_19", "drift_onset": 19,
     "drift_turns": (), "upto": None, "holds": True},
    {"id": "k04_drift_from_18", "drift_onset": 18,
     "drift_turns": (), "upto": None, "holds": True},
    {"id": "k05_drift_from_17", "drift_onset": 17,
     "drift_turns": (), "upto": None, "holds": True},
    {"id": "k06_drift_from_16_break", "drift_onset": 16,
     "drift_turns": (), "upto": None, "holds": False},
    {"id": "k07_drift_from_15_break", "drift_onset": 15,
     "drift_turns": (), "upto": None, "holds": False},
    {"id": "k08_middle_slip_turn_10", "drift_onset": None,
     "drift_turns": (10,), "upto": None, "holds": True},
    {"id": "k09_use_window_slips", "drift_onset": None,
     "drift_turns": (7, 10), "upto": None, "holds": True},
    {"id": "k10_early_slip_turn_2", "drift_onset": None,
     "drift_turns": (2,), "upto": None, "holds": True},
    {"id": "k11_truncated_coherent_14", "drift_onset": None,
     "drift_turns": (), "upto": 14, "holds": True},
    {"id": "k12_truncated_drift_from_12_break",
     "drift_onset": 12, "drift_turns": (), "upto": 14,
     "holds": False},
)


def transcript_for_saga(
    saga: dict[str, Any],
) -> list[tuple[str, str]]:
    """A saga's transcript from TF-028's real script:
    coherent answers everywhere except the saga's
    filed drift turns, which answer the script's
    drifted text."""
    script = COHERENCE_SCRIPT
    if saga["upto"] is not None:
        script = script[: saga["upto"]]
    onset = saga["drift_onset"]
    drifted_turns = set(saga["drift_turns"])
    transcript: list[tuple[str, str]] = []
    for index, turn in enumerate(script, start=1):
        drifting = (onset is not None and index >= onset) \
            or index in drifted_turns
        transcript.append(
            (turn.user, turn.drifted if drifting else turn.coherent))
    return transcript


def coherence_saga_report(saga: dict[str, Any]) -> dict[str, Any]:
    """One coherence saga judged by TF-028's real scorer.

    ``held`` is the suite's filed rule (whole-saga mean
    at HOLD_MIN_SCORE); ``verdict`` is TF-028's own
    stricter conversation verdict, carried as data.
    """
    report = score_conversation(transcript_for_saga(saga))
    return {
        "id": saga["id"], "turns": report.total_turns,
        "mean_score": report.mean_score,
        "early_mean": report.early_mean,
        "late_mean": report.late_mean, "drift": report.drift,
        "contradiction_turns": report.contradiction_turns,
        "verdict": report.verdict,
        "held": report.mean_score >= HOLD_MIN_SCORE,
    }


# ── The will axes ───────────────────────────────────────────────────────────
#
# Scripted sagas of user turns (filed intent + topic,
# the router's discipline) and scripted assistant
# answers, judged by replaying the user turns through
# TF-029's real GoalLedger. A steering answer carries
# the ledger's own line — "Standing goal: <statement>"
# — exactly the shape the core injects (TF-029).

def _turn(user: str, intent: str, topic: str,
          assistant: str) -> dict[str, Any]:
    return {"user": user, "intent": intent, "topic": topic,
            "assistant": assistant}


STEERING_SAGAS: tuple[dict[str, Any], ...] = (
    {"id": "w01_declared_goal_steers",
     "end_active": "restore the ferry timetable", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "A clear goal — we will get it posted again, "
               "step by step."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "The ferry leaves at dawn from the mill landing."),
         _turn("Tell me a joke about lanterns.",
               "light_conversation", "lanterns",
               "Standing goal: restore the ferry timetable. "
               "Here is a small one about a lantern that "
               "refused to retire."),
     )},
    {"id": "w02_wanted_goal_steers",
     "end_active": "trim the mill path lanterns before dusk",
     "end_closed": (), "holds": True,
     "turns": (
         _turn("I want to trim the mill path lanterns before dusk.",
               "light_conversation", "mill path lanterns",
               "Good — trimming them before dusk keeps the "
               "path safe."),
         _turn("Where does Wrenna keep the spare oil?",
               "factual_query", "lantern oil",
               "Standing goal: trim the mill path lanterns "
               "before dusk. She keeps the spare oil in a "
               "brass flask."),
         _turn("How long does the crossing take?",
               "factual_query", "ferry crossing",
               "Standing goal: trim the mill path lanterns "
               "before dusk. The crossing takes eleven "
               "minutes in calm water."),
     )},
    {"id": "w03_planning_goal_steers",
     "end_active": "lantern festival route", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("Help me plan the lantern festival route for "
               "the village.",
               "planning_request", "lantern festival route",
               "Here is the route: mill path first, then the "
               "landing, then the square."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: lantern festival route. It "
               "leaves at dawn, in good time for the route's "
               "first stop."),
         _turn("Who keeps the lanterns?",
               "factual_query", "lantern keeper",
               "Standing goal: lantern festival route. "
               "Wrenna keeps them, and she walks the route "
               "at dusk."),
     )},
    {"id": "w04_spec_goal_steers",
     "end_active": "ferry signal network", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("Design the ferry signal network for the landing.",
               "technical_spec_request", "ferry signal network",
               "The design: a bell post, two lantern relays, "
               "and a chalk board at the landing."),
         _turn("How does the tide bell work?",
               "factual_query", "tide bell",
               "Standing goal: ferry signal network. The bell "
               "rings twice at dusk, once for the ferry and "
               "once for the lanterns."),
         _turn("Tell me about the mill landing.",
               "light_conversation", "mill landing",
               "Standing goal: ferry signal network. The "
               "landing sits below the mill door, where the "
               "signal board hangs."),
     )},
    {"id": "w05_touch_after_cooling",
     "end_active": "restore the ferry timetable", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "A clear goal — noted."),
         _turn("What is the weather pattern over the northern sea?",
               "factual_query", "northern sea weather",
               "Standing goal: restore the ferry timetable. "
               "Calm winds and a high sky over the northern sea."),
         _turn("And the weather tomorrow?",
               "factual_query", "northern sea weather",
               "Standing goal: restore the ferry timetable. "
               "More of the same calm, by every sign."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn — the first sailing your timetable "
               "work must confirm."),
         _turn("Who poles the ferry?",
               "factual_query", "ferry hand",
               "Standing goal: restore the ferry timetable. "
               "Tomas poles it between the narrows and the "
               "landing."),
     )},
    {"id": "w06_redeclared_goal_touches",
     "end_active": "count the harbour lanterns", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("My goal is to count the harbour lanterns.",
               "light_conversation", "harbour lanterns",
               "Noted — a full count, post by post."),
         _turn("My goal is to count the harbour lanterns.",
               "light_conversation", "harbour lanterns",
               "Standing goal: count the harbour lanterns. "
               "Still the same count — the east posts first "
               "tonight."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: count the harbour lanterns. "
               "At dawn."),
     )},
    {"id": "w07_goal_survives_intent_changes",
     "end_active": "harbour market day", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("Help me plan the harbour market day.",
               "planning_request", "harbour market day",
               "The plan: stalls along the quay, music at "
               "noon, lanterns at dusk."),
         _turn("How many stalls fit on the quay?",
               "factual_query", "market stalls",
               "Standing goal: harbour market day. Twelve "
               "stalls fit along the quay with room to pass."),
         _turn("Give me ideas for stall banners.",
               "brainstorming", "stall banners",
               "Standing goal: harbour market day. Rope-and-"
               "canvas banners in harbour colours would suit "
               "the stalls."),
         _turn("Tell me a joke about markets.",
               "light_conversation", "markets",
               "Standing goal: harbour market day. The joke: "
               "the fish stall always gets the best spot — "
               "it has the best lines."),
     )},
    {"id": "w08_declaration_inside_factual_turn",
     "end_active": "chart the tide tables", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("What time does the ferry leave? My goal is to "
               "chart the tide tables.",
               "factual_query", "ferry departures",
               "It leaves at dawn — and charting the tide "
               "tables is a fine goal to keep beside that."),
         _turn("How does Wrenna record the river level?",
               "factual_query", "river level",
               "Standing goal: chart the tide tables. She "
               "records it in chalk on the mill door each "
               "morning."),
         _turn("Tell me about the mill.",
               "light_conversation", "mill",
               "Standing goal: chart the tide tables. The "
               "mill grinds grain above the landing, and its "
               "door carries the chalk marks."),
     )},
    {"id": "w09_we_are_trying_goal_steers",
     "end_active": "mend the harbour nets before the season",
     "end_closed": (), "holds": True,
     "turns": (
         _turn("We're trying to mend the harbour nets before "
               "the season.",
               "light_conversation", "harbour nets",
               "Good work to finish before the season turns."),
         _turn("Where do the boats land?",
               "factual_query", "boat landings",
               "Standing goal: mend the harbour nets before "
               "the season. They land below the mill, on the "
               "shingle."),
         _turn("What is the weather over the northern sea?",
               "factual_query", "northern sea weather",
               "Standing goal: mend the harbour nets before "
               "the season. Calm — good mending weather, "
               "in fact."),
     )},
    {"id": "w10_forming_turn_on_same_topic_touches",
     "end_active": "restore the ferry timetable", "end_closed": (),
     "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("Let us plan the ferry timetable work for this week.",
               "planning_request", "ferry timetable",
               "Standing goal: restore the ferry timetable. "
               "The week's work: confirm dawn, then the tide "
               "bell, then post it."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
     )},
    {"id": "w11_goal_dropped_fails",
     "end_active": "restore the ferry timetable", "end_closed": (),
     "holds": False,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted — we will get it posted again."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("Who keeps the lanterns?",
               "factual_query", "lantern keeper",
               "Wrenna keeps them and trims the wicks at dusk."),
     )},
    {"id": "w12_goal_contradicted_fails",
     "end_active": "restore the ferry timetable", "end_closed": (),
     "holds": False,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("What should we work on next?",
               "light_conversation", "next work",
               "Forget the timetable — let us design festival "
               "posters instead."),
     )},
)


def _stale_saga_turns() -> tuple[dict[str, Any], ...]:
    """The stale saga's turns, built to TF-029's filed
    count: a goal set on turn 1 survives exactly
    MAX_UNTOUCHED_TURNS untouched steering turns and
    goes stale on the next one."""
    turns: list[dict[str, Any]] = [
        _turn("My goal is to restore the ferry timetable.",
              "light_conversation", "ferry timetable",
              "A clear goal — noted.")]
    for n in range(1, MAX_UNTOUCHED_TURNS + 1):
        turns.append(_turn(
            f"What is the weather pattern over the northern "
            f"sea? (reading {n})",
            "factual_query", "northern sea weather",
            "Standing goal: restore the ferry timetable. "
            "Calm winds and a high sky over the northern sea."))
    turns.append(_turn(
        "What is the weather pattern over the northern sea? "
        "(the last reading)",
        "factual_query", "northern sea weather",
        "Calm winds over the northern sea again."))
    turns.append(_turn(
        "Tell me a joke about lanterns.",
        "light_conversation", "lanterns",
        "Here is one about a lantern that retired early."))
    return tuple(turns)


CLOSURE_SAGAS: tuple[dict[str, Any], ...] = (
    {"id": "c01_completion_closes",
     "end_active": None, "end_closed": ("completed",),
     "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("That's done, thank you.",
               "light_conversation", "thanks",
               "Finished and posted — a good piece of work."),
         _turn("Tell me a joke about lanterns.",
               "light_conversation", "lanterns",
               "Here is one about a lantern that retired early."),
     )},
    {"id": "c02_goal_achieved_closes",
     "end_active": None, "end_closed": ("completed",),
     "holds": True,
     "turns": (
         _turn("Our goal is to mend the harbour nets.",
               "light_conversation", "harbour nets",
               "Noted — net by net."),
         _turn("Where do the boats land?",
               "factual_query", "boat landings",
               "Standing goal: mend the harbour nets. Below "
               "the mill, on the shingle."),
         _turn("The goal is achieved — the last net is mended.",
               "light_conversation", "harbour nets",
               "All mended and hung to dry. Well done."),
     )},
    {"id": "c03_never_mind_abandons",
     "end_active": None, "end_closed": ("abandoned",),
     "holds": True,
     "turns": (
         _turn("My goal is to chart the tide tables.",
               "light_conversation", "tide tables",
               "Noted."),
         _turn("How does Wrenna record the river level?",
               "factual_query", "river level",
               "Standing goal: chart the tide tables. In "
               "chalk on the mill door."),
         _turn("Never mind the tide tables for now.",
               "light_conversation", "tide tables",
               "Set aside, then — no charts today."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "At dawn, from the mill landing."),
     )},
    {"id": "c04_forget_it_abandons",
     "end_active": None, "end_closed": ("abandoned",),
     "holds": True,
     "turns": (
         _turn("I want to rebuild the mill wheel.",
               "light_conversation", "mill wheel",
               "Noted — a big, honest job."),
         _turn("Who grinds the grain?",
               "factual_query", "mill grain",
               "Standing goal: rebuild the mill wheel. The "
               "miller and his son grind it."),
         _turn("Forget it — the wheel can wait a year.",
               "light_conversation", "mill wheel",
               "Waiting a year it is, then."),
     )},
    {"id": "c05_new_declaration_supersedes",
     "end_active": "mend the harbour nets",
     "end_closed": ("superseded",), "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("My goal is to mend the harbour nets.",
               "light_conversation", "harbour nets",
               "A new goal, then — the nets come first now."),
         _turn("Where do the boats land?",
               "factual_query", "boat landings",
               "Standing goal: mend the harbour nets. Below "
               "the mill, on the shingle."),
     )},
    {"id": "c06_new_plan_supersedes",
     "end_active": "lantern festival route",
     "end_closed": ("superseded",), "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("Help me plan the lantern festival route.",
               "planning_request", "lantern festival route",
               "The route: mill path, landing, then the "
               "square in lantern order."),
         _turn("Who keeps the lanterns?",
               "factual_query", "lantern keeper",
               "Standing goal: lantern festival route. "
               "Wrenna keeps them, and she walks the route "
               "at dusk."),
     )},
    {"id": "c07_untouched_goal_goes_stale",
     "end_active": None, "end_closed": ("stale",),
     "holds": True, "turns": _stale_saga_turns()},
    {"id": "c08_closing_words_with_no_goal",
     "end_active": None, "end_closed": (), "holds": True,
     "turns": (
         _turn("Never mind, it was nothing.",
               "light_conversation", "nothing",
               "As you like — nothing pursued, nothing lost."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "At dawn."),
     )},
    {"id": "c09_completion_then_new_goal",
     "end_active": "mend the harbour nets",
     "end_closed": ("completed",), "holds": True,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("We did it — the timetable is posted.",
               "light_conversation", "ferry timetable",
               "Posted and ringing true. Well done."),
         _turn("My goal is to mend the harbour nets.",
               "light_conversation", "harbour nets",
               "A new goal, then — the nets come first now."),
         _turn("Where do the boats land?",
               "factual_query", "boat landings",
               "Standing goal: mend the harbour nets. Below "
               "the mill."),
     )},
    {"id": "c10_abandoned_goal_still_steered_fails",
     "end_active": None, "end_closed": ("abandoned",),
     "holds": False,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("Never mind the timetable plan.",
               "light_conversation", "ferry timetable",
               "Set aside, then."),
         _turn("Who keeps the lanterns?",
               "factual_query", "lantern keeper",
               "Standing goal: restore the ferry timetable. "
               "Wrenna keeps them."),
     )},
    {"id": "c11_completed_goal_still_steered_fails",
     "end_active": None, "end_closed": ("completed",),
     "holds": False,
     "turns": (
         _turn("My goal is to chart the tide tables.",
               "light_conversation", "tide tables",
               "Noted."),
         _turn("How does Wrenna record the river level?",
               "factual_query", "river level",
               "Standing goal: chart the tide tables. In "
               "chalk on the mill door."),
         _turn("That's done, thank you.",
               "light_conversation", "thanks",
               "Finished — well done."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: chart the tide tables. At dawn."),
     )},
    {"id": "c12_superseded_goal_still_steered_fails",
     "end_active": "mend the harbour nets",
     "end_closed": ("superseded",), "holds": False,
     "turns": (
         _turn("My goal is to restore the ferry timetable.",
               "light_conversation", "ferry timetable",
               "Noted."),
         _turn("What time does the ferry leave?",
               "factual_query", "ferry departures",
               "Standing goal: restore the ferry timetable. "
               "At dawn."),
         _turn("My goal is to mend the harbour nets.",
               "light_conversation", "harbour nets",
               "A new goal, then."),
         _turn("Where do the boats land?",
               "factual_query", "boat landings",
               "Standing goal: restore the ferry timetable. "
               "Below the mill."),
     )},
)

ALL_SUITE_CASES: tuple[dict[str, Any], ...] = (
    COHERENCE_SAGAS + STEERING_SAGAS + CLOSURE_SAGAS
)

SUITE_AXES: tuple[dict[str, Any], ...] = (
    {"id": "long_conversation_coherence",
     "cases": len(COHERENCE_SAGAS),
     "scorer": "thoughtforge.cognition.coherence."
               "score_conversation (TF-028, reused) over "
               "saga transcripts built from TF-028's filed "
               "script answers",
     "computes": "fraction of the suite's scripted sagas "
                 "whose TF-028 whole-saga mean score is at "
                 "or above HOLD_MIN_SCORE (TF-028's "
                 "PASS_LATE_MIN, 0.80)"},
    {"id": "will_steering", "cases": len(STEERING_SAGAS),
     "scorer": "thoughtforge.cognition.will.GoalLedger "
               "replayed over the filed user turns "
               "(TF-029, reused); the ledger's will_line "
               "against the scripted answers",
     "computes": "fraction of the suite's scripted sagas "
                 "where every turn the ledger steers "
                 "carries the standing goal's statement in "
                 "the answer, no closed goal's statement "
                 "ever steers, and the ledger's end state "
                 "matches the filing"},
    {"id": "will_closure", "cases": len(CLOSURE_SAGAS),
     "scorer": "thoughtforge.cognition.will.GoalLedger "
               "replayed over the filed user turns "
               "(TF-029, reused), as the steering axis",
     "computes": "fraction of the suite's scripted sagas "
                 "where completion / abandonment words, "
                 "supersession, staleness, or a closing "
                 "with no goal standing end the steering "
                 "exactly as filed — and any answer still "
                 "steered by the closed goal fails the saga"},
)

SUITE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "three_axes_presence_and_will",
     "rule": "the suite scores exactly three axes — "
             "long-conversation coherence (TF-028), will "
             "steering and will closure (TF-029) — the "
             "presence claim and the will claim of "
             "Part VII, over scripted sagas"},
    {"id": "coherence_scored_by_tf028_real_scorer",
     "rule": "every coherence saga's transcript is "
             "built from TF-028's filed script (its "
             "coherent and drifted answers) and judged "
             "by TF-028's real score_conversation; the "
             "hold floor is TF-028's own PASS_LATE_MIN "
             "applied to the whole-saga mean, and "
             "TF-028's stricter verdict is carried per "
             "saga as data, never hidden"},
    {"id": "will_scored_by_tf029_real_ledger",
     "rule": "every will saga is judged by replaying "
             "its filed user turns through a fresh real "
             "GoalLedger: steering exists exactly where "
             "the ledger's will_line stands, and the "
             "injection leg (TF-029's inject_will over "
             "a replayed ledger) is run by the checker — "
             "this module files sagas and judges "
             "answers, not a second ledger"},
    {"id": "a_closed_goal_must_never_steer_again",
     "rule": "an answer carrying a closed goal's "
             "statement — completed, abandoned, stale "
             "or superseded — is a violation on any "
             "turn, including the closing turn itself; "
             "the still-steered failures (abandoned / "
             "completed / superseded but still steered) "
             "are filed by name and score held=False"},
    {"id": "no_vacuous_pass_anywhere",
     "rule": "an empty axis or an empty suite scores "
             "0.0 and files REVIEW; PASS requires a "
             "non-empty run with every axis at or above "
             "AXIS_PASS_THRESHOLD (the TF-122 / TF-123 / "
             "TF-124 precedent)"},
    {"id": "fixtures_are_synthetic_only",
     "rule": "every saga is invented for this suite "
             "(TF-017) in TF-028's synthetic Lantern "
             "Hollow world or in invented ferry / "
             "harbour talk: canon is never a benchmark "
             "input and data/ was never opened"},
    {"id": "failing_shapes_are_filed_not_hidden",
     "rule": "the corpora file the failing shapes by "
             "name (coherence breaks at filed onsets, a "
             "goal dropped, a goal contradicted, and "
             "the three still-steered closures) with "
             "holds=False — the suite scores failures, "
             "not just passes"},
)


# ── The will judge ──────────────────────────────────────────────────────────

def judge_will_saga(saga: dict[str, Any]) -> dict[str, Any]:
    """Judge one will saga by TF-029's real ledger.

    The filed user turns are replayed through a fresh
    ``GoalLedger``. A turn violates when the ledger's
    ``will_line`` stands and the scripted answer does
    not carry the standing goal's statement, or when
    the answer carries any closed goal's statement
    (a goal re-declared identically is the active
    goal again, never a closed one). The saga's end
    state must also match the filing.
    """
    ledger = GoalLedger()
    violations: list[str] = []
    for index, turn in enumerate(saga["turns"], start=1):
        sketch = InputSketch(
            raw_text=turn["user"], intent=turn["intent"],
            topic=turn["topic"], tone_in="neutral",
            response_mode="conversational")
        ledger.observe_turn(sketch, RESPONSE_GOAL)
        answer_norm = _norm(turn["assistant"])
        line = ledger.will_line(RESPONSE_GOAL)
        if line is not None and ledger.active is not None:
            if not _contains(answer_norm, ledger.active.statement):
                violations.append(
                    f"turn {index}: the standing goal "
                    f"{ledger.active.statement!r} steers but "
                    f"the answer does not carry it")
        for closed in ledger.closed:
            if ledger.active is not None \
                    and closed.statement == ledger.active.statement:
                continue
            if _contains(answer_norm, closed.statement):
                violations.append(
                    f"turn {index}: the {closed.status} goal "
                    f"{closed.statement!r} still steers")
    end_active = ledger.active.statement if ledger.active else None
    end_closed = [g.status for g in ledger.closed]
    if end_active != saga["end_active"]:
        violations.append(
            f"end state: active {end_active!r} != filed "
            f"{saga['end_active']!r}")
    if end_closed != list(saga["end_closed"]):
        violations.append(
            f"end state: closed {end_closed} != filed "
            f"{list(saga['end_closed'])}")
    return {"id": saga["id"], "held": not violations,
            "violations": violations,
            "end_active": end_active, "end_closed": end_closed}


# ── The axis scorers ────────────────────────────────────────────────────────

def score_coherence_sagas(
    sagas: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the coherence axis by TF-028's real scorer."""
    sagas = COHERENCE_SAGAS if sagas is None else sagas
    held = sum(int(coherence_saga_report(s)["held"]) for s in sagas)
    total = len(sagas)
    return {"axis": "long_conversation_coherence", "cases": total,
            "held": held, "score": (held / total) if total else 0.0}


def _score_will_axis(sagas, axis: str) -> dict[str, Any]:
    held = sum(int(judge_will_saga(s)["held"]) for s in sagas)
    total = len(sagas)
    return {"axis": axis, "cases": total, "held": held,
            "score": (held / total) if total else 0.0}


def score_steering_sagas(
    sagas: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the will-steering axis by TF-029's real ledger."""
    return _score_will_axis(
        STEERING_SAGAS if sagas is None else sagas, "will_steering")


def score_closure_sagas(
    sagas: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score the will-closure axis by TF-029's real ledger."""
    return _score_will_axis(
        CLOSURE_SAGAS if sagas is None else sagas, "will_closure")


# ── The suite run ───────────────────────────────────────────────────────────

@dataclass
class PresenceSuiteResult:
    """One full suite run: per-axis scores, overall, verdict."""

    coherence_score: float
    steering_score: float
    closure_score: float
    overall_score: float
    total_cases: int
    per_axis: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS only on a non-empty run with every axis passing."""
        if self.total_cases == 0:
            return "REVIEW"
        if min(self.coherence_score, self.steering_score,
               self.closure_score) >= AXIS_PASS_THRESHOLD:
            return "PASS"
        return "REVIEW"

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"


def run_presence_suite(
    coherence_sagas: Any = None,
    steering_sagas: Any = None,
    closure_sagas: Any = None,
) -> PresenceSuiteResult:
    """Run the whole suite (filed corpora by default).

    Any axis may be given an explicit corpus — including
    an empty one, which scores that axis 0.0 and keeps
    the suite at REVIEW (no vacuous pass anywhere).
    """
    coherence = score_coherence_sagas(coherence_sagas)
    steering = score_steering_sagas(steering_sagas)
    closure = score_closure_sagas(closure_sagas)
    per_axis = [coherence, steering, closure]
    total = sum(a["cases"] for a in per_axis)
    overall = (
        sum(a["score"] for a in per_axis) / len(per_axis)) if total else 0.0
    return PresenceSuiteResult(
        coherence_score=coherence["score"],
        steering_score=steering["score"],
        closure_score=closure["score"], overall_score=overall,
        total_cases=total, per_axis=per_axis)


# ── The live checker ────────────────────────────────────────────────────────

def presence_suite_problems() -> list[str]:
    """The filing against the real estate, judged live."""
    problems: list[str] = []

    axis_ids = [a["id"] for a in SUITE_AXES]
    if axis_ids != ["long_conversation_coherence", "will_steering",
                    "will_closure"]:
        problems.append(f"SUITE_AXES changed shape: {axis_ids}")
    rule_ids = [r["id"] for r in SUITE_RULES]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate SUITE_RULES ids")
    for needed in ("three_axes_presence_and_will",
                   "coherence_scored_by_tf028_real_scorer",
                   "will_scored_by_tf029_real_ledger",
                   "a_closed_goal_must_never_steer_again",
                   "no_vacuous_pass_anywhere",
                   "fixtures_are_synthetic_only",
                   "failing_shapes_are_filed_not_hidden"):
        if needed not in rule_ids:
            problems.append(f"SUITE_RULES lost {needed}")
    if HOLD_MIN_SCORE != PASS_LATE_MIN or HOLD_MIN_SCORE != 0.8:
        problems.append("HOLD_MIN_SCORE is not TF-028's PASS_LATE_MIN")

    # TF-028's estate, bound live: the filed script and
    # anchors the coherence axis is built from.
    if len(COHERENCE_SCRIPT) != 20 or len(ANCHORS) != 7:
        problems.append(
            f"TF-028's script/anchors drifted: "
            f"{len(COHERENCE_SCRIPT)} turns / {len(ANCHORS)} anchors")
    if MAX_UNTOUCHED_TURNS != 10:
        problems.append("TF-029's MAX_UNTOUCHED_TURNS drifted from 10")

    # Corpora: counts, balance, and filed expectations true.
    for name, sagas, held_want in (
            ("COHERENCE_SAGAS", COHERENCE_SAGAS, 9),
            ("STEERING_SAGAS", STEERING_SAGAS, 10),
            ("CLOSURE_SAGAS", CLOSURE_SAGAS, 9)):
        if len(sagas) != CASES_PER_AXIS:
            problems.append(f"{name} {len(sagas)} != {CASES_PER_AXIS}")
        if sum(s["holds"] for s in sagas) != held_want:
            problems.append(
                f"{name} files {sum(s['holds'] for s in sagas)} "
                f"holding (want {held_want})")
    saga_ids = [s["id"] for s in ALL_SUITE_CASES]
    if len(saga_ids) != len(set(saga_ids)):
        problems.append("duplicate saga ids across the corpora")
    for saga in COHERENCE_SAGAS:
        if coherence_saga_report(saga)["held"] != bool(saga["holds"]):
            problems.append(
                f"coherence expectation disagrees with TF-028's "
                f"scorer: {saga['id']}")
    for saga in STEERING_SAGAS + CLOSURE_SAGAS:
        judged = judge_will_saga(saga)
        if judged["held"] != bool(saga["holds"]):
            problems.append(
                f"will expectation disagrees with TF-029's "
                f"ledger: {saga['id']}: {judged['violations']}")

    # The coherence real-code leg: TF-028's fixture
    # engine reproduces every onset saga's transcript
    # (and the coherent prefix of a truncated one) from
    # the prompt's request line alone.
    for saga in COHERENCE_SAGAS:
        if saga["drift_turns"]:
            continue  # selective-drift shapes are scorer-only
        engine = CoherenceFixtureEngine(
            drift_from_turn=saga["drift_onset"])
        for user, answer in transcript_for_saga(saga):
            got = engine.generate(f"User request: {user}\n").text
            if got != answer:
                problems.append(
                    f"TF-028's fixture engine does not reproduce "
                    f"{saga['id']}")
                break

    # The filed coherence shapes, pinned: the coherent
    # saga passes TF-028's own verdict; the canonical
    # onset-15 break measures TF-028's filed 0.70.
    by_id = {s["id"]: s for s in ALL_SUITE_CASES}
    coherent = coherence_saga_report(by_id["k01_coherent_full"])
    if coherent["mean_score"] != 1.0 or coherent["verdict"] != "PASS":
        problems.append(f"the coherent saga no longer files 1.0 / PASS: "
                        f"{coherent}")
    broken = coherence_saga_report(by_id["k07_drift_from_15_break"])
    if abs(broken["mean_score"] - 0.7) > 1e-9 \
            or broken["verdict"] != "REVIEW" \
            or broken["contradiction_turns"] != 6:
        problems.append(f"the onset-15 break no longer measures "
                        f"TF-028's filed shape: {broken}")

    # The will real-code leg: TF-029's inject_will over a
    # ledger replayed from a filed steering saga prepends
    # the ledger's own line exactly once, and a replayed
    # ledger survives its dict round-trip.
    try:
        saga = by_id["w01_declared_goal_steers"]
        ledger = GoalLedger()
        for turn in saga["turns"][:2]:
            ledger.observe_turn(InputSketch(
                raw_text=turn["user"], intent=turn["intent"],
                topic=turn["topic"], tone_in="neutral",
                response_mode="conversational"), RESPONSE_GOAL)
        scaffold = CognitionScaffold(
            goal="turn goal", fact_block="Knowledge:\nx")
        if not inject_will(scaffold, ledger):
            problems.append("inject_will adds nothing over a "
                            "replayed steering saga")
        elif not scaffold.fact_block.startswith(
                "Standing goal: restore the ferry timetable"):
            problems.append("inject_will's line is not the filed "
                            "standing-goal line")
        if inject_will(scaffold, ledger):
            problems.append("inject_will duplicated its line")
        if GoalLedger.from_dict(ledger.to_dict()).to_dict() \
                != ledger.to_dict():
            problems.append("a replayed ledger does not survive "
                            "its dict round-trip")
    except Exception as exc:  # a leg that cannot run is a finding
        problems.append(f"the will injection leg failed: {exc!r}")

    # The D-BENCH-11 probes, pinned: the still-steered
    # closures and the dropped / contradicted answers
    # fail under the real ledger, by name.
    for saga_id in ("c10_abandoned_goal_still_steered_fails",
                    "c11_completed_goal_still_steered_fails",
                    "c12_superseded_goal_still_steered_fails",
                    "w11_goal_dropped_fails",
                    "w12_goal_contradicted_fails"):
        judged = judge_will_saga(by_id[saga_id])
        if judged["held"] or not judged["violations"]:
            problems.append(
                f"the filed failure saga {saga_id} no longer "
                f"fails under TF-029's ledger")

    # A fresh live run has the filed shape; empty is 0.0 / REVIEW.
    result = run_presence_suite()
    if result.total_cases != len(ALL_SUITE_CASES):
        problems.append(
            f"live run totals {result.total_cases} != corpus "
            f"{len(ALL_SUITE_CASES)}")
    if result.total_cases == 0 or result.overall_score <= 0.0:
        problems.append("the live suite run scores nothing")
    empty = run_presence_suite([], [], [])
    if empty.total_cases != 0 or empty.overall_score != 0.0 \
            or empty.verdict != "REVIEW" or empty.passes:
        problems.append("an empty suite does not file 0.0 / REVIEW")

    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / PRESENCE_SUITE_DOC
    if not doc.exists():
        problems.append(f"the write-up {PRESENCE_SUITE_DOC} is missing")
    return problems


def presence_suite_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    result = run_presence_suite()
    return {
        "format": PRESENCE_SUITE_FORMAT,
        "doc": PRESENCE_SUITE_DOC,
        "axis_pass_threshold": AXIS_PASS_THRESHOLD,
        "hold_min_score": HOLD_MIN_SCORE,
        "axes": [dict(a) for a in SUITE_AXES],
        "case_counts": {
            "long_conversation_coherence": len(COHERENCE_SAGAS),
            "will_steering": len(STEERING_SAGAS),
            "will_closure": len(CLOSURE_SAGAS),
            "total": result.total_cases,
        },
        "coherence_reports": [
            coherence_saga_report(s) for s in COHERENCE_SAGAS],
        "measured": {
            "coherence_score": result.coherence_score,
            "steering_score": result.steering_score,
            "closure_score": result.closure_score,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
        },
        "rules": [dict(r) for r in SUITE_RULES],
    }


__all__ = [
    "ALL_SUITE_CASES",
    "AXIS_PASS_THRESHOLD",
    "CASES_PER_AXIS",
    "CLOSURE_SAGAS",
    "COHERENCE_SAGAS",
    "HOLD_MIN_SCORE",
    "PRESENCE_SUITE_DOC",
    "PRESENCE_SUITE_FORMAT",
    "RESPONSE_GOAL",
    "STEERING_SAGAS",
    "SUITE_AXES",
    "SUITE_RULES",
    "PresenceSuiteResult",
    "coherence_saga_report",
    "judge_will_saga",
    "presence_suite_as_data",
    "presence_suite_problems",
    "run_presence_suite",
    "score_closure_sagas",
    "score_coherence_sagas",
    "score_steering_sagas",
    "transcript_for_saga",
]
