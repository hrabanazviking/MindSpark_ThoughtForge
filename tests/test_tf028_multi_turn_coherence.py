"""
TF-028 — Multi-turn coherence fixtures: 20-turn conversations on
a small model fixture, scored for drift.

The discipline lives in ``src/thoughtforge/cognition/coherence.py``
(script + anchors as data, the pure drift scorer, the deterministic
fixture engine, the history-threading runner); the write-up is
``docs/architecture/MULTI_TURN_COHERENCE.md``.

Pinned here:

  * the script: exactly 20 turns, unique user turns, every turn
    probing ≥1 defined anchor, every anchor probed ≥3 times, all
    in TF-017's synthetic world (no canon names anywhere),
  * the scorer on pure transcripts: the coherent script scores a
    perfect PASS; a contradiction zeroes its turn even when the
    right fact is also stated; partial group satisfaction scores
    fractionally; unprobed anchors are neither credited nor
    blamed; an empty transcript is 0.0 / REVIEW, never a vacuous
    pass; user-turn mismatch against the script raises,
  * the windows: a hand-built drift-from-15 transcript scores
    early 1.0 / late 0.0 / drift 1.0 / REVIEW with per-anchor
    retention matching hand arithmetic,
  * the fixture engine: routing reads the ``User request:`` line
    only (history inside the prompt cannot misroute it), the
    drift boundary is exact, unknown requests get the grounded
    fallback, params are ignored and answers deterministic,
  * the runner end-to-end through ThoughtForgeCore: the coherent
    fixture holds 1.000 across all 20 turns (history 40 messages,
    0 dropped, 20 unique turn_ids, 54 generate calls = 40
    candidate + 14 salvage-refine), the drifting fixture is
    caught (mean 0.700, late 0.000, drift 1.000, 6 contradiction
    turns, REVIEW), history demonstrably rides into the next
    turn's prompt, and two coherent runs are byte-identical.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.coherence import (
    ANCHORS,
    COHERENCE_SCRIPT,
    FALLBACK_ANSWER,
    CoherenceFixtureEngine,
    extract_request,
    run_conversation,
    score_conversation,
)
from thoughtforge.cognition.core import ThoughtForgeCore

# The canon markers this file guards against, assembled from parts:
# TF-017's estate guard scans every test file for these literals in
# order to forbid them, so the literals themselves must never be
# written here (the guard file is the one allowed exception).
CANON_MARKERS = ("sig" + "rid", "ygg" + "drasil",
                 "knowledge_" + "reference", "pictures_" + "of")


def _coherent_transcript() -> list[tuple[str, str]]:
    return [(t.user, t.coherent) for t in COHERENCE_SCRIPT]


def _drift_transcript(from_turn: int = 15) -> list[tuple[str, str]]:
    return [
        (t.user, t.drifted if i + 1 >= from_turn else t.coherent)
        for i, t in enumerate(COHERENCE_SCRIPT)
    ]


def _core(engine: CoherenceFixtureEngine) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    core = ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    core._engine = engine
    return core


# ── The script ─────────────────────────────────────────────────────────────────


class TestScript:
    def test_exactly_twenty_turns_with_unique_users(self) -> None:
        assert len(COHERENCE_SCRIPT) == 20
        assert len({t.user for t in COHERENCE_SCRIPT}) == 20

    def test_every_turn_probes_defined_anchors_and_each_anchor_is_probed(self) -> None:
        probed: dict[str, int] = {}
        for turn in COHERENCE_SCRIPT:
            assert turn.anchors, turn.user
            for anchor_id in turn.anchors:
                assert anchor_id in ANCHORS
                probed[anchor_id] = probed.get(anchor_id, 0) + 1
        assert set(probed) == set(ANCHORS)
        assert min(probed.values()) >= 3

    def test_script_is_synthetic_world_only(self) -> None:
        blob = " ".join(
            [t.user + t.coherent + t.drifted for t in COHERENCE_SCRIPT]
            + [a.description for a in ANCHORS.values()]
        ).lower()
        for marker in CANON_MARKERS:
            assert marker not in blob
        assert "lantern hollow" in blob and "wrenna" in blob

    def test_drifted_answers_differ_from_coherent(self) -> None:
        for turn in COHERENCE_SCRIPT:
            assert turn.drifted != turn.coherent


# ── The scorer (pure transcripts) ──────────────────────────────────────────────


class TestScorer:
    def test_coherent_script_scores_perfect_pass(self) -> None:
        report = score_conversation(_coherent_transcript())
        assert report.total_turns == 20
        assert report.mean_score == 1.0
        assert report.early_mean == 1.0 and report.late_mean == 1.0
        assert report.drift == 0.0 and report.contradiction_turns == 0
        assert report.verdict == "PASS"
        assert report.anchor_retention == {k: 1.0 for k in ANCHORS}

    def test_empty_transcript_is_review_never_vacuous_pass(self) -> None:
        report = score_conversation([])
        assert report.total_turns == 0 and report.mean_score == 0.0
        assert report.verdict == "REVIEW"

    def test_contradiction_zeroes_turn_even_when_fact_also_stated(self) -> None:
        first = COHERENCE_SCRIPT[0]
        transcript = [(first.user, first.coherent + " It takes twelve minutes.")]
        report = score_conversation(transcript)
        turn = report.per_turn[0]
        assert turn.retention == 1.0                 # the fact IS stated
        assert turn.contradictions == ("ferry_minutes",)
        assert turn.score == 0.0                     # and the turn is still wrong
        assert report.contradiction_turns == 1

    def test_partial_group_satisfaction_scores_fractionally(self) -> None:
        turn4 = COHERENCE_SCRIPT[3]   # probes river_record + keeper + place
        transcript = _coherent_transcript()[:3] + [
            (turn4.user, "Wrenna writes in chalk at Lantern Hollow.")
        ]
        report = score_conversation(transcript)
        # groups: chalk ✓, mill door ✗, wrenna ✓, lantern hollow ✓ → 3/4
        assert report.per_turn[3].retention == 0.75
        assert report.per_turn[3].score == 0.75
        assert report.per_turn[3].contradictions == ()

    def test_unprobed_anchors_are_neither_credited_nor_blamed(self) -> None:
        turn9 = COHERENCE_SCRIPT[8]   # probes keeper + place only
        text = ("Wrenna carries spare oil at Lantern Hollow; "
                "the crossing takes twelve minutes.")
        transcript = _coherent_transcript()[:8] + [(turn9.user, text)]
        report = score_conversation(transcript)
        assert report.per_turn[8].score == 1.0
        assert report.per_turn[8].contradictions == ()

    def test_user_mismatch_and_overlength_raise(self) -> None:
        with pytest.raises(ValueError):
            score_conversation([("A different question entirely.", "Any answer.")])
        with pytest.raises(ValueError):
            score_conversation(_coherent_transcript() + [("Extra?", "Extra.")])

    def test_hand_built_drift_from_15_windows_and_anchors(self) -> None:
        report = score_conversation(_drift_transcript(15))
        assert report.early_mean == 1.0 and report.late_mean == 0.0
        assert report.drift == 1.0 and report.mean_score == 0.7
        assert report.contradiction_turns == 6
        assert report.verdict == "REVIEW"
        # ferry_minutes probed on turns 1, 7, 13 (kept) + 16, 20 (contradicted)
        assert report.anchor_retention["ferry_minutes"] == 0.6
        # red_lantern probed on turns 6, 12 (kept) + 17, 20 (contradicted)
        assert report.anchor_retention["red_lantern"] == 0.5


# ── The fixture engine ─────────────────────────────────────────────────────────


class TestFixtureEngine:
    def test_request_line_routing_beats_history_contamination(self) -> None:
        engine = CoherenceFixtureEngine()
        prompt = (
            "Write one direct, practical response.\n\n"
            f"User request: {COHERENCE_SCRIPT[0].user}\n\n"
            "Knowledge context:\nConversation history:\n"
            f"USER: {COHERENCE_SCRIPT[5].user}\n"
            f"ASSISTANT: {COHERENCE_SCRIPT[5].coherent}\n"
        )
        result = engine.generate(prompt)
        assert result.text == COHERENCE_SCRIPT[0].coherent

    def test_extract_request_first_line_and_absence(self) -> None:
        assert extract_request("x\nUser request: hello?\nUser request: later") == "hello?"
        assert extract_request("no request line here") == ""

    def test_drift_boundary_is_exact(self) -> None:
        engine = CoherenceFixtureEngine(drift_from_turn=15)
        t14, t15 = COHERENCE_SCRIPT[13], COHERENCE_SCRIPT[14]
        assert engine.generate(f"User request: {t14.user}").text == t14.coherent
        assert engine.generate(f"User request: {t15.user}").text == t15.drifted

    def test_unknown_request_gets_grounded_fallback(self) -> None:
        engine = CoherenceFixtureEngine()
        assert engine.generate("User request: Something unscripted?").text == FALLBACK_ANSWER
        assert engine.generate("a prompt with no request line").text == FALLBACK_ANSWER

    def test_params_ignored_and_deterministic(self) -> None:
        engine = CoherenceFixtureEngine()
        prompt = f"User request: {COHERENCE_SCRIPT[1].user}"
        first = engine.generate(prompt, params=object())
        second = engine.generate(prompt, params=None)
        assert first.text == second.text == COHERENCE_SCRIPT[1].coherent
        assert engine.calls == 2

    def test_refine_shaped_prompt_still_routes_by_request_line(self) -> None:
        engine = CoherenceFixtureEngine()
        turn = COHERENCE_SCRIPT[16]
        prompt = f"Using these useful fragments:\n- a fragment\n\nUser request: {turn.user}\n"
        assert engine.generate(prompt).text == turn.coherent


# ── The runner, end to end through the core ────────────────────────────────────


class TestRunner:
    def test_coherent_run_holds_across_twenty_turns(self) -> None:
        engine = CoherenceFixtureEngine()
        run = run_conversation(_core(engine))
        report = run.report
        assert report.total_turns == 20 and report.verdict == "PASS"
        assert report.mean_score == 1.0 and report.drift == 0.0
        assert report.contradiction_turns == 0
        assert [u for u, _ in run.transcript] == [t.user for t in COHERENCE_SCRIPT]

    def test_coherent_run_history_and_turn_bookkeeping(self) -> None:
        engine = CoherenceFixtureEngine()
        run = run_conversation(_core(engine))
        assert len(run.history.messages) == 40      # 20 turns, stored whole
        assert run.history.dropped_count == 0       # nothing silently lost
        assert len(set(run.turn_ids)) == 20
        # 40 candidate generations (2 modes × 20 turns) + 14 salvage
        # refine prompts, each routed by the same request line.
        assert engine.calls == 54

    def test_drifting_run_is_caught(self) -> None:
        engine = CoherenceFixtureEngine(drift_from_turn=15)
        run = run_conversation(_core(engine))
        report = run.report
        assert report.verdict == "REVIEW"
        assert report.mean_score == 0.7
        assert report.early_mean == 1.0 and report.late_mean == 0.0
        assert report.drift == 1.0 and report.contradiction_turns == 6
        assert report.anchor_retention["ferry_minutes"] == 0.6
        assert report.anchor_retention["red_lantern"] == 0.5
        assert all(t.score == 1.0 for t in report.per_turn[:14])
        assert all(t.score == 0.0 for t in report.per_turn[14:])

    def test_history_rides_into_the_next_turns_prompt(self) -> None:
        engine = CoherenceFixtureEngine()
        run_conversation(_core(engine))
        second_turn_prompts = [
            p for p in engine.prompts
            if f"User request: {COHERENCE_SCRIPT[1].user}" in p
            and p.startswith("Write one")
        ]
        assert second_turn_prompts
        prompt = second_turn_prompts[0]
        assert "Conversation history:" in prompt
        assert COHERENCE_SCRIPT[0].user in prompt
        assert COHERENCE_SCRIPT[0].coherent in prompt

    def test_two_coherent_runs_are_byte_identical(self) -> None:
        first = run_conversation(_core(CoherenceFixtureEngine()))
        second = run_conversation(_core(CoherenceFixtureEngine()))
        assert first.transcript == second.transcript
        assert first.report.mean_score == second.report.mean_score == 1.0
