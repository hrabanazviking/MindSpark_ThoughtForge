"""
TF-096 — Persona-voice preservation under salvage:
refinement must not sand a character's voice flat
(fixtures from TF-030).

The estate lives in
``src/thoughtforge/refinement/persona_voice.py``
(``VOICE_RULES`` / ``VOICE_FIXTURES`` as data,
``persona_violations`` / ``voice_problems`` /
``check_voice``, the checkers
``persona_voice_problems`` /
``persona_voice_live_problems``, format
``thoughtforge-persona-voice/1``), with the guard
wired into ``FragmentSalvage`` (selection + the
refine adoption rule, active only when a persona
is supplied) and the core passing its TF-030
expectation in ``_compose_final``.
Write-up: ``docs/architecture/PERSONA_VOICE_PRESERVATION.md``.

Measured start: handed an in-voice cited draft
beside a longer cited rival tripping ``vague
handwaving``, score-only selection returned the
rival; and three stub-engine refine outputs built
on TF-030's out-of-voice fixture turns — each
tripping exactly one avoid entry, each padded
longer — were ALL adopted as refine_pass_1 over
the in-voice draft. Filed as D-SALV-3.

Pinned here:

  * the filed estate: both checkers clean, the
    doc present, proof grade stated, the guard
    wired into salvage.py (source anchors),
  * the judgement: TF-030's avoid subset only,
    relative to the source — a violation the
    source already carries is never counted as
    introduced by salvage; no persona in force
    raises, never passes silently,
  * refine adoption: the three flattened outputs
    rejected (candidate stands, attempt counted),
    the benign in-voice and same-violation legs
    adopted,
  * selection: an in-voice draft at hand outranks
    a flattened rival only when a persona is
    supplied; with none consistent, score order
    over all stands; a lone out-of-voice draft
    passes through verbatim,
  * the contrast legs: with no persona supplied,
    the flattened refine output is adopted and
    the longer rival selected — byte-identical
    to before this slice,
  * TF-088's guard, TF-030's checker and the
    scoring formula are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.cognition.persona import (
    FIXTURE_EXPECTATION,
    check_persona_response,
)
from thoughtforge.refinement.persona_voice import (
    VOICE_DOC,
    VOICE_FIXTURES,
    VOICE_FORMAT,
    VOICE_RULES,
    check_voice,
    persona_violations,
    persona_voice_as_data,
    persona_voice_live_problems,
    persona_voice_problems,
    run_voice_fixture,
    voice_problems,
)
from thoughtforge.refinement.salvage_map import run_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in VOICE_FIXTURES}

_IN_VOICE = _FIXTURE["flattened_handwaving_rejected"]["candidates"][0]
_FLAT = _FIXTURE["flattened_handwaving_rejected"]["engine_text"]


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert persona_voice_problems() == []
        assert persona_voice_live_problems() == []

    def test_format_doc_and_proof(self):
        data = persona_voice_as_data()
        assert data["format"] == VOICE_FORMAT
        assert data["voice_doc"] == VOICE_DOC
        assert (REPO_ROOT / VOICE_DOC).exists()
        assert data["proof"]["live_model_voice_drift"] == "not_proven"
        assert data["proof"]["judged_subset"] == \
            "persona_avoid_list_via_tf030"
        assert data["proof"]["flattening_without_a_violation"] == \
            "residual_not_judged"
        json.dumps(data)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in VOICE_RULES]
        assert len(ids) == len(set(ids)) == 9

    def test_fixture_counts(self):
        assert len(VOICE_FIXTURES) == 10
        kinds = [f["kind"] for f in VOICE_FIXTURES]
        assert kinds.count("refine") == 6
        assert kinds.count("selection") == 4


# ── The judgement ──────────────────────────────────────────────


class TestJudgement:
    def test_violations_are_tf030s_own(self):
        assert persona_violations(FIXTURE_EXPECTATION, _FLAT) == \
            check_persona_response(
                FIXTURE_EXPECTATION, _FLAT).violations == \
            ["vague handwaving"]
        assert persona_violations(FIXTURE_EXPECTATION, _IN_VOICE) == []

    def test_new_violations_are_relative_to_the_source(self):
        assert voice_problems(
            FIXTURE_EXPECTATION, _IN_VOICE, _FLAT) == \
            ["vague handwaving"]
        # Already in the source: not salvage's introduction.
        assert voice_problems(
            FIXTURE_EXPECTATION, _FLAT, _FLAT) == []

    def test_no_persona_raises_never_passes(self):
        with pytest.raises(ValueError):
            persona_violations(None, _IN_VOICE)
        with pytest.raises(ValueError):
            voice_problems(None, _IN_VOICE, _FLAT)
        with pytest.raises(ValueError):
            check_voice(None, _IN_VOICE, _FLAT)

    def test_check_voice_verdicts(self):
        benign = _FIXTURE["benign_in_voice_refine_adopted"]["engine_text"]
        assert check_voice(
            FIXTURE_EXPECTATION, _IN_VOICE, benign).verdict == "preserved"
        assert check_voice(
            FIXTURE_EXPECTATION, _IN_VOICE, _IN_VOICE).verdict == \
            "verbatim_pass_through"
        broken = check_voice(FIXTURE_EXPECTATION, _IN_VOICE, _FLAT)
        assert broken.verdict == "voice_broken"
        assert broken.new_violations == ("vague handwaving",)
        same = _FIXTURE["same_violation_refine_adopted"]
        already = check_voice(
            FIXTURE_EXPECTATION, same["candidates"][0],
            same["engine_text"])
        assert already.verdict == "already_out_of_voice"
        assert already.new_violations == ()


# ── Refine adoption ────────────────────────────────────────────


class TestRefineAdoption:
    @pytest.mark.parametrize("fixture_id", [
        "flattened_handwaving_rejected",
        "flattened_repetition_rejected",
        "flattened_crossings_rejected",
    ])
    def test_flattened_refine_outputs_are_rejected(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        result = run_voice_fixture(fixture_id)
        # The candidate stands; the attempt still counts.
        assert result.text == fixture["candidates"][0]
        assert result.salvage_path == "refine_pass_1"
        assert result.passes_used == 1

    @pytest.mark.parametrize("fixture_id", [
        "benign_in_voice_refine_adopted",
        "same_violation_refine_adopted",
    ])
    def test_clean_refine_outputs_still_adopt(self, fixture_id):
        result = run_voice_fixture(fixture_id)
        assert result.text == _FIXTURE[fixture_id]["engine_text"]
        assert result.salvage_path == "refine_pass_1"

    def test_flattened_adopted_without_a_persona(self):
        # The contrast leg: no persona supplied, the
        # guard is inert — pre-TF-096 adoption stands.
        result = run_voice_fixture("flattened_adopted_without_a_persona")
        assert result.text == _FLAT


# ── Selection ──────────────────────────────────────────────────


class TestSelection:
    def test_selection_prefers_the_in_voice_draft(self):
        result = run_voice_fixture("selection_prefers_the_in_voice_draft")
        assert result.text == _IN_VOICE
        assert result.salvage_path == "best_draft"

    def test_selection_without_a_persona_picks_the_longer(self):
        result = run_voice_fixture(
            "selection_without_a_persona_picks_the_longer")
        assert result.text == _FLAT

    def test_selection_all_out_of_voice_falls_back(self):
        result = run_voice_fixture("selection_all_out_of_voice_falls_back")
        assert result.text == _FLAT

    def test_out_of_voice_pass_through_verbatim(self):
        result = run_voice_fixture("out_of_voice_pass_through_verbatim")
        assert result.text == _FLAT
        assert result.salvage_path == "best_draft"


# ── What this slice does not touch ─────────────────────────────


class TestUntouched:
    def test_tf087s_estate_is_unchanged(self):
        # TF-087's fixtures run without a persona, as
        # filed — the guard never stirs.
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"
        assert result.text == " ".join(["The bell rings."] * 20)

    def test_tf030s_checker_is_unchanged(self):
        check = check_persona_response(FIXTURE_EXPECTATION, _FLAT)
        assert check.violations == ["vague handwaving"]
        assert not check.consistent
