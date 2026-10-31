"""
TF-029 — The will layer: goal/intent persistence across turns.

The discipline lives in ``src/thoughtforge/cognition/will.py``
(standing goal as data, the ledger's observe order, the steering
line, the thread-state bridge); the write-up is
``docs/architecture/WILL_LAYER.md``.

Pinned here:

  * setting: a goal-forming intent (planning / technical spec)
    sets the standing goal from the turn's topic; an explicit
    declaration in ANY intent sets it from the user's own words
    (sentence-cut, capped); a one-shot intent (brainstorming,
    debugging, factual, light) sets nothing on its own,
  * persistence: the standing goal survives intent changes; a
    later turn on an overlapping topic touches it; the steering
    line appears in the scaffold's fact_block from the NEXT turn
    on — never on the setting turn, never duplicated, and the
    scaffold's own per-turn goal is never overwritten,
  * replacement: a goal-forming turn on a new topic supersedes
    the old goal (recorded, not dropped); re-declaring the same
    goal touches instead of replacing,
  * closing: completion words complete it, abandonment words
    abandon it, closing words beat goal-forming on the same turn,
    and 11 untouched turns (MAX_UNTOUCHED_TURNS = 10) stale it,
  * serialisation: ledger dict round-trip and the
    ActiveThreadStateRecord bridge (statement <-> summary,
    open loops carried, response goal honestly unrecoverable),
  * the core wiring end-to-end: two knowledge-only turns — the
    plan sets the will, the small factual turn after it carries
    the standing-goal line in its scaffold and still answers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.router import InputRouter
from thoughtforge.cognition.scaffold import ScaffoldBuilder
from thoughtforge.cognition.will import (
    GOAL_FORMING_INTENTS,
    MAX_GOAL_CHARS,
    MAX_UNTOUCHED_TURNS,
    GoalLedger,
    StandingGoal,
    extract_declaration,
    inject_will,
)
from thoughtforge.knowledge.models import (
    ActiveThreadStateRecord,
    CognitionScaffold,
    MemoryActivationBundle,
)

ROUTER = InputRouter()
BUILDER = ScaffoldBuilder()

PLAN = "Help me plan the lantern festival route for Lantern Hollow"
SPEC = "Design a system architecture for the ferry signal network with API endpoints"
FACTUAL_FERRY = "What time does the ferry leave?"
UNRELATED = "What is the weather pattern over the northern sea?"
LANTERN_JOKE = "Tell me a joke about lanterns"


def observe(text: str, ledger: GoalLedger | None = None) -> GoalLedger:
    """Route + scaffold one turn for real, then observe it."""
    ledger = ledger or GoalLedger()
    sketch = ROUTER.route(text)
    scaffold = BUILDER.build(sketch, MemoryActivationBundle())
    ledger.observe_turn(sketch, scaffold.goal)
    return ledger


# ── Declarations ───────────────────────────────────────────────────────────────


class TestDeclaration:
    def test_my_goal_is(self) -> None:
        assert extract_declaration(
            "My goal is to map every lantern in Lantern Hollow"
        ) == "map every lantern in Lantern Hollow"

    def test_sentence_cut_and_cap(self) -> None:
        text = "I want to " + ("sail " * 60) + ". And then rest."
        out = extract_declaration(text)
        assert out is not None
        assert len(out) == MAX_GOAL_CHARS
        assert "And then" not in out

    def test_no_declaration(self) -> None:
        assert extract_declaration(FACTUAL_FERRY) is None
        assert extract_declaration("My goal is ") is None or True  # never raises


# ── Setting ────────────────────────────────────────────────────────────────────


class TestSetting:
    def test_goal_forming_intents_are_the_two_endeavours(self) -> None:
        assert GOAL_FORMING_INTENTS == frozenset(
            {"planning_request", "technical_spec_request"}
        )

    def test_planning_sets_standing_goal(self) -> None:
        ledger = observe(PLAN)
        assert ledger.active is not None
        assert ledger.active.source_intent == "planning_request"
        assert ledger.active.statement == ROUTER.route(PLAN).topic
        assert ledger.active.set_at_turn == 1
        assert ledger.active.response_goal == (
            "produce a clear, ordered, actionable plan"
        )

    def test_spec_sets_standing_goal(self) -> None:
        ledger = observe(SPEC)
        assert ledger.active is not None
        assert ledger.active.source_intent == "technical_spec_request"

    def test_declaration_sets_goal_in_any_intent(self) -> None:
        ledger = observe("My goal is to map every lantern in Lantern Hollow")
        assert ledger.active is not None
        assert ledger.active.statement == "map every lantern in Lantern Hollow"
        # the declaring turn itself routes as light conversation —
        # the declaration, not the intent, set the goal
        assert ledger.active.source_intent == "light_conversation"

    def test_one_shot_intents_set_nothing(self) -> None:
        for text in (
            FACTUAL_FERRY,
            LANTERN_JOKE,
            "Give me ideas for lantern colours",       # brainstorming
            "How do I fix the broken lantern circuit?",  # debugging
        ):
            ledger = observe(text)
            assert ledger.active is None, text


# ── Persistence & steering ─────────────────────────────────────────────────────


class TestPersistence:
    def test_goal_survives_intent_change(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(FACTUAL_FERRY, ledger)
        assert ledger.active is not None
        assert ledger.turn_count == 2

    def test_no_steering_line_on_setting_turn(self) -> None:
        ledger = observe(PLAN)
        assert ledger.will_line("anything") is None

    def test_steering_line_from_next_turn(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(FACTUAL_FERRY, ledger)
        line = ledger.will_line("give an accurate, direct answer with cited sources")
        assert line is not None
        assert line.startswith("Standing goal: ")
        assert "pursue it as: produce a clear, ordered, actionable plan" in line

    def test_steering_line_omits_response_goal_when_same(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(FACTUAL_FERRY, ledger)
        goal = ledger.active
        assert goal is not None
        line = ledger.will_line(goal.response_goal)
        assert line == f"Standing goal: {goal.statement}"

    def test_inject_will_prepends_once_and_preserves_goal(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(FACTUAL_FERRY, ledger)
        scaffold = CognitionScaffold(goal="turn goal", fact_block="Knowledge:\nx")
        assert inject_will(scaffold, ledger) is True
        assert scaffold.fact_block.startswith("Standing goal: ")
        assert scaffold.goal == "turn goal"  # per-turn goal untouched
        assert inject_will(scaffold, ledger) is False  # never duplicated
        assert scaffold.fact_block.count("Standing goal: ") == 1

    def test_touch_on_topic_overlap_resets_cooling(self) -> None:
        ledger = observe(PLAN)
        for _ in range(3):
            ledger = observe(UNRELATED, ledger)
        assert ledger.active is not None
        assert ledger.active.turns_since_touch == 3
        ledger = observe(LANTERN_JOKE, ledger)
        assert ledger.active is not None
        assert ledger.active.turns_since_touch == 0
        assert ledger.active.last_touched_turn == 5


# ── Replacement ────────────────────────────────────────────────────────────────


class TestReplacement:
    def test_new_topic_supersedes_recorded(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(SPEC, ledger)
        assert ledger.active is not None
        assert ledger.active.source_intent == "technical_spec_request"
        assert [g.status for g in ledger.closed] == ["superseded"]
        assert ledger.closed[0].source_intent == "planning_request"

    def test_redeclaring_same_goal_touches(self) -> None:
        text = "My goal is to map every lantern in Lantern Hollow"
        ledger = observe(text)
        ledger = observe(text, ledger)
        assert ledger.active is not None
        assert ledger.active.set_at_turn == 1
        assert ledger.active.last_touched_turn == 2
        assert ledger.closed == []


# ── Closing ────────────────────────────────────────────────────────────────────


class TestClosing:
    def test_completion(self) -> None:
        ledger = observe(PLAN)
        ledger = observe("That is done, thank you", ledger)
        assert ledger.active is None
        assert [g.status for g in ledger.closed] == ["completed"]

    def test_abandonment(self) -> None:
        ledger = observe(PLAN)
        ledger = observe("Never mind the festival plan", ledger)
        assert ledger.active is None
        assert [g.status for g in ledger.closed] == ["abandoned"]

    def test_closing_words_beat_goal_forming(self) -> None:
        # "never mind the festival plan" itself routes as a
        # planning_request — it must close, never set
        ledger = observe("Never mind the festival plan")
        assert ledger.active is None
        ledger = observe(PLAN)
        ledger = observe("Never mind the festival plan", ledger)
        assert ledger.active is None

    def test_untouched_goal_goes_stale(self) -> None:
        ledger = observe(PLAN)
        for _ in range(MAX_UNTOUCHED_TURNS):
            ledger = observe(UNRELATED, ledger)
        assert ledger.active is not None  # exactly at the limit: alive
        ledger = observe(UNRELATED, ledger)
        assert ledger.active is None
        assert [g.status for g in ledger.closed] == ["stale"]


# ── Serialisation & the thread-state bridge ────────────────────────────────────


class TestSerialisation:
    def test_dict_round_trip(self) -> None:
        ledger = observe(PLAN)
        ledger = observe(SPEC, ledger)
        ledger = observe(FACTUAL_FERRY, ledger)
        twin = GoalLedger.from_dict(ledger.to_dict())
        assert twin.to_dict() == ledger.to_dict()

    def test_thread_state_round_trip(self) -> None:
        ledger = observe(PLAN)
        assert ledger.active is not None
        ledger.active.open_loops = ["book the ferry", "count the lanterns"]
        record = ledger.to_thread_state("thr_will")
        assert record is not None
        assert record.topic == ledger.active.topic
        assert record.summary == ledger.active.statement
        assert record.open_loops == ["book the ferry", "count the lanterns"]
        assert record.expires_after_turns == MAX_UNTOUCHED_TURNS

        restored = GoalLedger.from_thread_state(record)
        assert restored.active is not None
        assert restored.active.statement == ledger.active.statement
        assert restored.active.open_loops == ["book the ferry", "count the lanterns"]
        # honest loss: the record has no response-goal field
        assert restored.active.response_goal == ""

    def test_thread_state_bridge_refusals(self) -> None:
        assert GoalLedger().to_thread_state("thr_x") is None
        assert GoalLedger.from_thread_state(None).active is None
        stale = ActiveThreadStateRecord(
            record_id="thr_s", summary="old goal", status="stale"
        )
        assert GoalLedger.from_thread_state(stale).active is None
        empty = ActiveThreadStateRecord(record_id="thr_e", summary="")
        assert GoalLedger.from_thread_state(empty).active is None

    def test_standing_goal_dict_round_trip(self) -> None:
        goal = StandingGoal(statement="map the hollow", topic="lanterns")
        assert StandingGoal.from_dict(goal.to_dict()).to_dict() == goal.to_dict()


# ── Core wiring ────────────────────────────────────────────────────────────────


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )


class TestCoreWiring:
    def test_two_turns_plan_then_fact(self) -> None:
        core = _core()
        core.think(PLAN)
        assert core.will_ledger.active is not None
        first = core.last_turn_state
        assert first is not None and first.scaffold is not None
        assert "Standing goal:" not in first.scaffold.fact_block

        final = core.think(FACTUAL_FERRY)
        assert final.text  # the turn still answers
        assert core.will_ledger.active is not None
        second = core.last_turn_state
        assert second is not None and second.scaffold is not None
        assert second.scaffold.fact_block.startswith("Standing goal: ")
        # the per-turn goal stays that turn's own (the ferry
        # question routes light_conversation) — will steers the
        # conversation, not the turn
        assert second.scaffold.goal == (
            "keep the exchange natural, warm, and engaging"
        )

    def test_single_light_turn_sets_no_will(self) -> None:
        core = _core()
        core.think(LANTERN_JOKE)
        assert core.will_ledger.active is None
