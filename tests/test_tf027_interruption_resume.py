"""
TF-027 — Interruption & resume: a turn can be cut and the state
remains coherent (the Rúnmál voice lesson, applied to text).

The discipline lives in ``src/thoughtforge/cognition/interruption.py``
(InterruptToken, TurnInterrupted, check_interrupt, mark_interrupted);
the cut points and resume live in ``ThoughtForgeCore`` (think /
think_stream gain ``interrupt=``; ``resume_turn`` finishes a cut
turn); the cut record lives on ``RuntimeTurnState`` (TF-025's
object, new fields). The event contract gain is TF-026's
``interrupted`` terminal kind. Write-up:
``docs/architecture/INTERRUPTION_AND_RESUME.md``.

Pinned here:

  * the token: unset by default, first reason wins, idempotent,
    check_interrupt raises TurnInterrupted carrying turn + reason,
  * think() cut by a pre-fired token raises TurnInterrupted; the
    retained turn is coherent — phase "scaffolded" (the last
    finished stage), no final, no completed_at, candidates empty,
  * a cut mid-generation (non-stream) keeps the completed
    candidate and discards nothing it finished; resume generates
    ONLY the missing mode and completes the SAME turn_id,
  * a streamed cut mid-candidate ends the stream with exactly one
    terminal ``interrupted`` event, the partial candidate never
    lands on the turn, and resume still completes the turn,
  * a streamed cut between candidates keeps candidate 1; resume
    generates only candidate 2's mode,
  * a consumer closing the stream mid-turn cuts it coherently
    (reason recorded); closing before the scaffold leaves a
    coherent but non-resumable state (resume raises ValueError);
    closing after ``final`` rewrites nothing,
  * resume of nothing / of a completed turn raises ValueError;
    a resume can itself be cut and resumed again (resume_count),
  * a token fired after a turn completed does not rewrite it,
  * a caller-supplied ChatHistory gains nothing from a cut turn
    (the core never appends, and there is no final to append).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.chat_history import ChatHistory
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.interruption import (
    InterruptToken,
    TurnInterrupted,
    check_interrupt,
    mark_interrupted,
)
from thoughtforge.knowledge.models import RuntimeTurnState

TEXT = (
    "Yggdrasil is the world tree that joins the nine realms. "
    "Its roots reach into deep wells of wisdom and fate. "
    "Odin hung upon its branches for nine nights seeking runes."
)


def _core() -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )


class _FireOnGenerateEngine:
    """Completed-text engine; fires the token during generate()
    call number ``fire_on`` (that call still returns its text)."""

    def __init__(self, token: InterruptToken, fire_on: int = 1) -> None:
        self.token = token
        self.fire_on = fire_on
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt, params=None):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == self.fire_on:
            self.token.interrupt("user pressed stop")
        return SimpleNamespace(text=TEXT, tokens_generated=len(TEXT.split()))


class _StreamEngine:
    """Native streaming stub + a completed-text generate() (the
    road resume_turn takes)."""

    def __init__(self, text: str = TEXT, step: int = 17) -> None:
        self.text = text
        self.step = step
        self.generate_calls = 0

    def generate_stream(self, prompt, params=None):
        for i in range(0, len(self.text), self.step):
            yield self.text[i:i + self.step]

    def generate(self, prompt, params=None):
        self.generate_calls += 1
        return SimpleNamespace(text=self.text, tokens_generated=len(self.text.split()))


# ── The token ──────────────────────────────────────────────────────────────────


class TestInterruptToken:
    def test_defaults_unset(self) -> None:
        token = InterruptToken()
        assert token.is_set is False and token.reason == ""

    def test_first_reason_wins_and_idempotent(self) -> None:
        token = InterruptToken()
        token.interrupt("first")
        token.interrupt("second")
        assert token.is_set is True and token.reason == "first"

    def test_check_interrupt_noops(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        check_interrupt(turn, None)
        check_interrupt(turn, InterruptToken())

    def test_check_interrupt_raises_carrying_turn_and_reason(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        token = InterruptToken()
        token.interrupt("barge-in")
        with pytest.raises(TurnInterrupted) as excinfo:
            check_interrupt(turn, token)
        assert excinfo.value.turn is turn
        assert excinfo.value.reason == "barge-in"

    def test_mark_interrupted_records_cut_without_faking_completion(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        turn.phase = "generated"
        mark_interrupted(turn, "stop")
        assert turn.interrupted is True and turn.is_interrupted is True
        assert turn.interrupt_reason == "stop"
        assert turn.interrupted_phase == "generated"
        assert turn.final_response is None and turn.completed_at == ""

    def test_new_turn_state_cut_defaults(self) -> None:
        turn = RuntimeTurnState(turn_id="t1")
        assert turn.interrupted is False and turn.is_interrupted is False
        assert turn.interrupt_reason == "" and turn.interrupted_phase == ""
        assert turn.resume_count == 0


# ── think() cuts ───────────────────────────────────────────────────────────────


class TestThinkCut:
    def test_pre_fired_token_cuts_at_scaffold_checkpoint(self) -> None:
        core = _core()
        core._engine = _FireOnGenerateEngine(InterruptToken(), fire_on=99)
        token = InterruptToken()
        token.interrupt("changed my mind")
        with pytest.raises(TurnInterrupted) as excinfo:
            core.think("What is Yggdrasil?", interrupt=token)
        turn = core.last_turn_state
        assert turn is not None and excinfo.value.turn is turn
        assert excinfo.value.reason == "changed my mind"
        # Coherent: the last FINISHED stage, nothing fabricated.
        assert turn.phase == "scaffolded"
        assert turn.interrupted is True and turn.is_interrupted is True
        assert turn.interrupted_phase == "scaffolded"
        assert turn.interrupt_reason == "changed my mind"
        assert turn.final_response is None and turn.completed_at == ""
        assert turn.candidates == [] and turn.fragments == []
        assert turn.salvage_ms == 0           # compose never ran
        assert turn.input_sketch is not None and turn.scaffold is not None

    def test_mid_generation_cut_keeps_completed_candidate(self) -> None:
        core = _core()
        token = InterruptToken()
        engine = _FireOnGenerateEngine(token, fire_on=1)
        core._engine = engine
        with pytest.raises(TurnInterrupted):
            core.think("What is Yggdrasil?", interrupt=token)
        turn = core.last_turn_state
        assert turn is not None
        assert engine.calls == 1              # second mode never started
        assert len(turn.candidates) == 1      # the finished one survives
        assert turn.candidates[0].text == TEXT
        assert turn.phase == "scaffolded"     # generation never finished
        assert isinstance(turn.generation_ms, int)

    def test_token_fired_after_completion_rewrites_nothing(self) -> None:
        core = _core()
        core._engine = _FireOnGenerateEngine(InterruptToken(), fire_on=99)
        token = InterruptToken()
        result = core.think("What is Yggdrasil?", interrupt=token)
        token.interrupt("too late")
        turn = core.last_turn_state
        assert turn is not None and turn.final_response is result
        assert turn.is_complete is True and turn.interrupted is False

    def test_history_gains_nothing_from_a_cut_turn(self) -> None:
        core = _core()
        history = ChatHistory()
        history.add_user("What is Yggdrasil?")
        before = list(history.messages)
        token = InterruptToken()
        token.interrupt("stop")
        with pytest.raises(TurnInterrupted):
            core.think("What is Yggdrasil?", history=history, interrupt=token)
        assert history.messages == before     # no partial assistant entry


# ── Resume ─────────────────────────────────────────────────────────────────────


class TestResume:
    def test_resume_generates_only_the_missing_mode(self) -> None:
        core = _core()
        token = InterruptToken()
        engine = _FireOnGenerateEngine(token, fire_on=1)
        core._engine = engine
        with pytest.raises(TurnInterrupted):
            core.think("What is Yggdrasil?", interrupt=token)
        turn = core.last_turn_state
        assert turn is not None
        cut_turn_id = turn.turn_id
        first_candidate = turn.candidates[0]

        result = core.resume_turn()
        # generate() ran exactly once more — the missing mode only.
        assert engine.calls == 2
        assert turn.phase == "completed" and turn.is_complete is True
        assert turn.turn_id == cut_turn_id
        assert result.turn_id == cut_turn_id
        assert turn.final_response is result
        assert turn.resume_count == 1
        assert turn.interrupted is False and turn.is_interrupted is False
        assert turn.interrupt_reason == "" and turn.interrupted_phase == ""
        assert len(turn.candidates) == 2
        assert turn.candidates[0] is first_candidate   # kept, not regenerated
        modes = [c.mode for c in turn.candidates]
        assert turn.scaffold is not None
        assert modes == turn.scaffold.candidate_modes[:2]
        assert turn.fragments

    def test_resume_knowledge_only_turn(self) -> None:
        core = _core()
        token = InterruptToken()
        token.interrupt("stop")
        with pytest.raises(TurnInterrupted):
            core.think("What is Yggdrasil?", interrupt=token)
        result = core.resume_turn()
        turn = core.last_turn_state
        assert turn is not None and turn.is_complete is True
        assert result.mode == "knowledge_only"
        assert result.turn_id == turn.turn_id

    def test_resume_without_a_cut_turn_raises(self) -> None:
        core = _core()
        with pytest.raises(ValueError):
            core.resume_turn()                # no turn at all
        core.think("What is Yggdrasil?")
        with pytest.raises(ValueError):
            core.resume_turn()                # turn completed, not cut

    def test_resume_can_be_cut_and_resumed_again(self) -> None:
        core = _core()
        token = InterruptToken()
        token.interrupt("first stop")
        with pytest.raises(TurnInterrupted):
            core.think("What is Yggdrasil?", interrupt=token)
        token2 = InterruptToken()
        token2.interrupt("second stop")
        with pytest.raises(TurnInterrupted) as excinfo:
            core.resume_turn(interrupt=token2)
        turn = core.last_turn_state
        assert turn is not None and excinfo.value.turn is turn
        assert turn.interrupted is True
        assert turn.interrupt_reason == "second stop"
        assert turn.resume_count == 1
        result = core.resume_turn()
        assert turn.resume_count == 2
        assert turn.is_complete is True and turn.final_response is result


# ── Streamed cuts ──────────────────────────────────────────────────────────────


class TestStreamCut:
    def test_pre_fired_token_stream_is_started_then_interrupted(self) -> None:
        core = _core()
        core._engine = _StreamEngine()
        token = InterruptToken()
        token.interrupt("no, wait")
        events = list(core.think_stream("What is Yggdrasil?", interrupt=token))
        assert [e.kind for e in events] == ["turn_started", "interrupted"]
        assert events[1].text == "no, wait" and events[1].final is None
        turn = core.last_turn_state
        assert turn is not None and turn.interrupted is True
        assert turn.phase == "scaffolded" and turn.candidates == []

    def test_mid_candidate_cut_discards_partial_and_resumes(self) -> None:
        core = _core()
        engine = _StreamEngine()
        core._engine = engine
        token = InterruptToken()
        events = []
        chunks_seen = 0
        for ev in core.think_stream("What is Yggdrasil?", interrupt=token):
            events.append(ev)
            if ev.kind == "chunk":
                chunks_seen += 1
                if chunks_seen == 3:
                    token.interrupt("talked over")
        kinds = [e.kind for e in events]
        assert kinds[-1] == "interrupted" and kinds.count("final") == 0
        assert events[-1].text == "talked over"
        assert "candidate_end" not in kinds   # in-flight candidate never ended
        turn = core.last_turn_state
        assert turn is not None
        assert turn.candidates == []          # partial discarded whole
        assert turn.phase == "scaffolded" and turn.final_response is None

        result = core.resume_turn()
        assert engine.generate_calls == 2     # both modes, completed-text road
        assert turn.is_complete is True and turn.final_response is result
        assert len(turn.candidates) == 2
        assert all(c.text == TEXT for c in turn.candidates)

    def test_cut_between_candidates_keeps_first_and_resumes_second(self) -> None:
        core = _core()
        engine = _StreamEngine()
        core._engine = engine
        token = InterruptToken()
        for ev in core.think_stream("What is Yggdrasil?", interrupt=token):
            if ev.kind == "candidate_end":
                token.interrupt("enough")
                # keep consuming: the cut lands at the next cut point
        turn = core.last_turn_state
        assert turn is not None
        assert turn.interrupted is True
        assert len(turn.candidates) == 1 and turn.candidates[0].text == TEXT

        result = core.resume_turn()
        assert engine.generate_calls == 1     # only the missing mode
        assert len(turn.candidates) == 2
        assert turn.final_response is result and turn.is_complete is True

    def test_uninterrupted_stream_has_no_interrupted_event(self) -> None:
        core = _core()
        core._engine = _StreamEngine()
        events = list(core.think_stream("What is Yggdrasil?", interrupt=InterruptToken()))
        kinds = [e.kind for e in events]
        assert kinds[-1] == "final" and "interrupted" not in kinds
        turn = core.last_turn_state
        assert turn is not None and turn.interrupted is False


# ── Consumer closes the stream ─────────────────────────────────────────────────


class TestConsumerClose:
    def test_close_mid_stream_cuts_coherently_and_resumes(self) -> None:
        core = _core()
        core._engine = _StreamEngine()
        stream = core.think_stream("What is Yggdrasil?")
        first = next(stream)
        assert first.kind == "turn_started"
        second = next(stream)
        assert second.kind == "candidate_start"
        stream.close()
        turn = core.last_turn_state
        assert turn is not None
        assert turn.interrupted is True and turn.is_interrupted is True
        assert turn.interrupt_reason == "consumer closed the stream"
        assert turn.phase == "scaffolded"     # routing/retrieval/scaffold done
        assert turn.final_response is None and turn.candidates == []
        result = core.resume_turn()
        assert turn.is_complete is True and turn.final_response is result

    def test_close_right_after_start_is_coherent_but_not_resumable(self) -> None:
        core = _core()
        core._engine = _StreamEngine()
        stream = core.think_stream("What is Yggdrasil?")
        assert next(stream).kind == "turn_started"
        stream.close()
        turn = core.last_turn_state
        assert turn is not None
        assert turn.interrupted is True and turn.phase == "created"
        assert turn.input_sketch is None and turn.final_response is None
        with pytest.raises(ValueError):
            core.resume_turn()

    def test_close_after_final_rewrites_nothing(self) -> None:
        core = _core()
        core._engine = _StreamEngine()
        stream = core.think_stream("What is Yggdrasil?")
        kinds = [ev.kind for ev in stream]
        assert kinds[-1] == "final"
        stream.close()
        turn = core.last_turn_state
        assert turn is not None
        assert turn.is_complete is True and turn.interrupted is False
