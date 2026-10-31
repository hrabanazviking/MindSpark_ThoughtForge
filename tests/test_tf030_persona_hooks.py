"""
TF-030 — Persona consistency hooks: the cognition layer's half.

The discipline lives in ``src/thoughtforge/cognition/persona.py``
(the expectation as data, the steering block, the avoid check,
the aggregate report, the synthetic fixtures); the write-up is
``docs/architecture/PERSONA_CONSISTENCY_HOOKS.md``.

Pinned here:

  * the expectation is derived from the loaded record (the
    TF-017 synthetic Wrenna core), never hard-coded: every field
    carried, snapshot-copied, dict round-trip exact; no record →
    no expectation,
  * steering: the block carries traits / speech style / behavior
    rules (each list capped at MAX_STEERING_ENTRIES), and NOT the
    avoid list (already steered via scaffold.avoid) or the weights
    (scorer data, not prose — TF-141's question),
  * injection: prepends to the fact_block, never duplicates,
    no-ops on no persona / no expressive lines, and never touches
    the scaffold's per-turn fields,
  * the check: avoid entries matched literally after
    normalisation — multi-word as phrases, single-word as whole
    words; no persona in force is a ValueError, not a clean sheet,
  * the aggregate: empty batch = 0.0 / REVIEW (never the vacuous
    PASS of D-BENCH-4); PASS only on a non-empty fully consistent
    batch,
  * the fixtures: every PERSONA_FIXTURE_TURNS pair scores exactly
    its ground truth against FIXTURE_EXPECTATION, and the
    out-of-voice turns each trip exactly one avoid entry,
  * the core wiring end-to-end: a core whose store holds the
    synthetic persona exposes the expectation and carries the
    steering block in its scaffold from the FIRST turn, above the
    knowledge facts; a persona-free core is untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.persona import (
    FIXTURE_EXPECTATION,
    MAX_STEERING_ENTRIES,
    PERSONA_FIXTURE_TURNS,
    PersonaConsistencyReport,
    PersonaExpectation,
    check_persona_response,
    inject_persona,
    score_persona_turns,
)
from thoughtforge.knowledge.models import CognitionScaffold
from tests.fixtures import synthetic as syn


def _expectation() -> PersonaExpectation:
    exp = PersonaExpectation.from_record(syn.load_synthetic_personality())
    assert exp is not None
    return exp


# ── The expectation ────────────────────────────────────────────────────────────


class TestExpectation:
    def test_from_record_carries_every_field(self) -> None:
        record = syn.load_synthetic_personality()
        exp = _expectation()
        assert exp.record_id == record.record_id
        assert exp.traits == record.traits
        assert exp.speech_style == record.speech_style
        assert exp.behavior_rules == record.behavior_rules
        assert exp.avoid == record.avoid
        assert exp.weights == record.weights

    def test_from_no_record_is_none(self) -> None:
        assert PersonaExpectation.from_record(None) is None

    def test_snapshot_is_a_copy(self) -> None:
        record = syn.load_synthetic_personality()
        exp = PersonaExpectation.from_record(record)
        assert exp is not None
        record.traits.append("mutated after the fact")
        record.weights["directness"] = 0.0
        assert "mutated after the fact" not in exp.traits
        assert exp.weights["directness"] == 0.8

    def test_dict_round_trip(self) -> None:
        exp = _expectation()
        assert PersonaExpectation.from_dict(exp.to_dict()).to_dict() == exp.to_dict()


# ── Steering lines ─────────────────────────────────────────────────────────────


class TestSteeringLines:
    def test_lines_in_label_order_with_all_three_lists(self) -> None:
        lines = _expectation().steering_lines()
        assert lines == [
            "Persona traits: steady; observant; plain-spoken",
            "Persona speech style: short, concrete sentences; "
            "names the river level before giving advice; no filler",
            "Persona behavior: answer what was asked; "
            "cite the synthetic reference when using it; "
            "say plainly when the fixture store is silent",
        ]

    def test_avoid_and_weights_have_no_line(self) -> None:
        block = _expectation().steering_block()
        assert "vague handwaving" not in block  # an avoid entry
        assert "directness" not in block        # a weight name
        assert "0.8" not in block               # a weight value

    def test_each_list_capped_at_max_entries(self) -> None:
        exp = PersonaExpectation(
            record_id="cap_test",
            traits=[f"trait{i}" for i in range(MAX_STEERING_ENTRIES + 3)],
        )
        (line,) = exp.steering_lines()
        assert line.count(";") == MAX_STEERING_ENTRIES - 1
        assert f"trait{MAX_STEERING_ENTRIES}" not in line

    def test_empty_persona_has_no_lines(self) -> None:
        exp = PersonaExpectation(record_id="bare")
        assert exp.steering_lines() == []
        assert exp.steering_block() == ""


# ── Injection ──────────────────────────────────────────────────────────────────


class TestInjectPersona:
    def test_prepends_block_above_existing_facts(self) -> None:
        scaffold = CognitionScaffold(goal="turn goal", fact_block="Knowledge:\nx")
        assert inject_persona(scaffold, _expectation()) is True
        assert scaffold.fact_block.startswith("Persona traits: steady")
        assert scaffold.fact_block.endswith("Knowledge:\nx")

    def test_block_on_empty_fact_block_is_exactly_the_block(self) -> None:
        exp = _expectation()
        scaffold = CognitionScaffold(goal="turn goal")
        assert inject_persona(scaffold, exp) is True
        assert scaffold.fact_block == exp.steering_block()

    def test_never_duplicates(self) -> None:
        exp = _expectation()
        scaffold = CognitionScaffold(goal="turn goal", fact_block="facts")
        assert inject_persona(scaffold, exp) is True
        assert inject_persona(scaffold, exp) is False
        assert scaffold.fact_block.count("Persona traits:") == 1

    def test_no_persona_is_a_noop(self) -> None:
        scaffold = CognitionScaffold(goal="turn goal", fact_block="facts")
        assert inject_persona(scaffold, None) is False
        assert scaffold.fact_block == "facts"

    def test_persona_without_expressive_lines_is_a_noop(self) -> None:
        scaffold = CognitionScaffold(goal="turn goal", fact_block="facts")
        bare = PersonaExpectation(record_id="bare", avoid=["repetition"])
        assert inject_persona(scaffold, bare) is False
        assert scaffold.fact_block == "facts"

    def test_turn_fields_untouched(self) -> None:
        scaffold = CognitionScaffold(
            goal="turn goal", tone=["direct"], focus=["clarity"],
            avoid=["filler"], depth="medium", candidate_modes=["practical"],
        )
        inject_persona(scaffold, _expectation())
        assert scaffold.goal == "turn goal"
        assert scaffold.tone == ["direct"]
        assert scaffold.focus == ["clarity"]
        assert scaffold.avoid == ["filler"]
        assert scaffold.candidate_modes == ["practical"]


# ── The per-turn check ─────────────────────────────────────────────────────────


class TestCheck:
    def test_clean_response_is_consistent(self) -> None:
        check = check_persona_response(_expectation(), PERSONA_FIXTURE_TURNS[0][0])
        assert check.consistent
        assert check.violations == []

    def test_phrase_avoid_matches_despite_case_and_punctuation(self) -> None:
        check = check_persona_response(
            _expectation(), "Really just VAGUE  HANDWAVING, honestly."
        )
        assert check.violations == ["vague handwaving"]
        assert not check.consistent

    def test_single_word_avoid_needs_a_whole_word(self) -> None:
        exp = _expectation()
        hit = check_persona_response(exp, "Mind the repetition in the log.")
        assert hit.violations == ["repetition"]
        miss = check_persona_response(exp, "The repetitions are logged daily.")
        assert miss.violations == []

    def test_no_persona_in_force_raises(self) -> None:
        with pytest.raises(ValueError):
            check_persona_response(None, "anything")


# ── The aggregate report ───────────────────────────────────────────────────────


class TestScoreTurns:
    def test_empty_batch_is_review_at_zero_never_a_vacuous_pass(self) -> None:
        report = score_persona_turns(_expectation(), [])
        assert report.total_turns == 0
        assert report.score == 0.0
        assert report.verdict == "REVIEW"

    def test_all_consistent_batch_passes(self) -> None:
        good = [t for t, ok in PERSONA_FIXTURE_TURNS if ok]
        report = score_persona_turns(_expectation(), good)
        assert report.verdict == "PASS"
        assert report.score == 1.0
        assert report.violation_count == 0

    def test_mixed_batch_scores_the_consistent_fraction(self) -> None:
        texts = [t for t, _ in PERSONA_FIXTURE_TURNS]
        report = score_persona_turns(_expectation(), texts)
        assert report.total_turns == 5
        assert report.consistent_turns == 2
        assert report.score == pytest.approx(0.4)
        assert report.verdict == "REVIEW"

    def test_no_persona_in_force_raises(self) -> None:
        with pytest.raises(ValueError):
            score_persona_turns(None, ["anything"])

    def test_report_defaults_are_an_empty_review(self) -> None:
        report = PersonaConsistencyReport()
        assert report.score == 0.0
        assert report.verdict == "REVIEW"


# ── The fixtures ───────────────────────────────────────────────────────────────


class TestFixtures:
    def test_fixture_expectation_is_the_synthetic_persona(self) -> None:
        assert FIXTURE_EXPECTATION.to_dict() == _expectation().to_dict()

    def test_every_fixture_turn_scores_its_ground_truth(self) -> None:
        for text, expected in PERSONA_FIXTURE_TURNS:
            check = check_persona_response(FIXTURE_EXPECTATION, text)
            assert check.consistent is expected, text

    def test_each_out_of_voice_turn_trips_exactly_one_avoid(self) -> None:
        for text, expected in PERSONA_FIXTURE_TURNS:
            check = check_persona_response(FIXTURE_EXPECTATION, text)
            if expected:
                assert check.violations == []
            else:
                assert len(check.violations) == 1


# ── Core wiring ────────────────────────────────────────────────────────────────


def _core(with_persona: bool) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    if with_persona:
        syn.build_synthetic_store(Path(tmp) / "memory")
        return ThoughtForgeCore(
            memory_dir=Path(tmp) / "memory",
            db_path=Path(tmp) / "nonexistent.db",
            model_path=None,
        )
    # A genuinely persona-free core needs an explicit empty
    # cognition config: with the default config, the core's
    # fallback loads the shipped configs/personality_core.yaml
    # (configs/default.yaml's personality_core_path) even when the
    # memory dir is empty — measured by this slice (see the doc).
    return ThoughtForgeCore(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
        config={"cognition": {}},
    )


class TestCoreWiring:
    def test_core_with_persona_exposes_expectation(self) -> None:
        core = _core(with_persona=True)
        assert core.persona_expectation is not None
        assert core.persona_expectation.record_id == "syn_wrenna_001"

    def test_first_turn_scaffold_carries_the_persona_block(self) -> None:
        core = _core(with_persona=True)
        core.think("What time does the ferry leave?")
        turn = core.last_turn_state
        assert turn is not None and turn.scaffold is not None
        # persona steers from the first turn — identity, not persistence
        assert "Persona traits: steady; observant; plain-spoken" in (
            turn.scaffold.fact_block
        )
        assert "Persona speech style:" in turn.scaffold.fact_block
        assert "Persona behavior:" in turn.scaffold.fact_block
        # and the block appears exactly once
        assert turn.scaffold.fact_block.count("Persona traits:") == 1

    def test_persona_free_core_is_untouched(self) -> None:
        core = _core(with_persona=False)
        assert core.persona_expectation is None
        core.think("What time does the ferry leave?")
        turn = core.last_turn_state
        assert turn is not None and turn.scaffold is not None
        assert "Persona traits:" not in turn.scaffold.fact_block
