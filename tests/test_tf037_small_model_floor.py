"""
TF-037 — The small-model floor: the core proven against the
smallest fixture model on phone_low budgets (simulated profile).

The discipline lives in
``src/thoughtforge/cognition/small_model_floor.py`` (the live
floor budget, the cap-honouring fixture engine, the pure floor
checker, the 20-turn floor runner); the write-up is
``docs/architecture/SMALL_MODEL_FLOOR.md``.

Pinned here:

  * the floor budget is resolved LIVE from the shipped phone_low
    profile (response 180, per-turn budget 180, context 512,
    drafts 2, refinement 1, bundle facts 5) with TF-033's phone_low
    latency budget embedded — never a copied table,
  * the defect this slice found and fixed: the profile's
    ``max_refinement_passes`` was read by no cognition code —
    salvage ran a flat 2 passes on every profile (pre-fix floor
    probe: 7 of 20 turns ran refine_pass_2 under phone_low). The
    TF-032 config surface now resolves it and the core's compose
    hands it to FragmentSalvage: phone_low → 1, pi_zero → 0,
    desktop_gpu → 3, no profile → the standing 2, byte-identical,
  * the smallest fixture model honours a max_tokens cap by
    word-truncation (a real small model must) and records caps,
  * the pure checker names overruns per dimension, in order —
    including a latency stage overrun hiding under a within
    total,
  * end to end: TF-028's 20-turn coherence script through a
    phone_low-selected core holds the floor on every turn
    (max response 30 tokens, max prompt 338, 2 candidates,
    ≤1 refine pass; 47 engine calls = 40 candidate + 7 refine)
    AND scores drift PASS 1.000 — the core holds together at
    the floor, not merely within its budgets,
  * an over-budget turn (a 400-word answer at the 180 cap) is
    caught by name — the floor is a gate, not a decoration.

Proof grade: FIXTURE model on a SIMULATED profile, on this VM.
No real small model and no phone were measured; live-model
floor proof is TF-126's, phone hardware TF-069/TF-128's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.chat_history import ChatHistory
from thoughtforge.cognition.coherence import COHERENCE_SCRIPT
from thoughtforge.cognition.config_surface import (
    DEFAULT_REFINE_PASSES,
    resolve_conversation_config,
)
from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.latency_budgets import (
    LatencyVerdict,
    StageVerdict,
    budget_for_profile,
)
from thoughtforge.cognition.small_model_floor import (
    DIMENSIONS,
    FLOOR_PROFILE_ID,
    FloorFixtureEngine,
    FloorObservation,
    check_floor_turn,
    floor_budget,
    run_floor_conversation,
    salvage_passes_used,
)
from thoughtforge.inference.profiles import resolve_generation_params
from thoughtforge.utils.config import load_hardware_profile


def _core(profile: str | None, engine=None) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    kwargs: dict = dict(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    if profile is not None:
        kwargs["config"] = {"conversation": {"profile_id": profile}}
    core = ThoughtForgeCore(**kwargs)
    if engine is not None:
        core._engine = engine
    return core


def _observation(**overrides) -> FloorObservation:
    budget = floor_budget()
    base = dict(
        turn_id="turn_test",
        response_tokens=25,
        prompt_tokens_max=300,
        candidate_count=2,
        refinement_passes=1,
        latency=LatencyVerdict(
            profile_id=budget.profile_id,
            stages=(
                StageVerdict("retrieval", 0.0, budget.latency.retrieval_ms),
                StageVerdict("generation", 10.0, budget.latency.generation_ms),
                StageVerdict("salvage", 5.0, budget.latency.salvage_ms),
                StageVerdict("total", 100.0, budget.latency.total_ms),
            ),
        ),
    )
    base.update(overrides)
    return FloorObservation(**base)


# ── The floor budget ──────────────────────────────────────────────────────────


class TestFloorBudget:
    def test_floor_profile_is_phone_low(self) -> None:
        assert FLOOR_PROFILE_ID == "phone_low"
        assert floor_budget().profile_id == "phone_low"

    def test_budget_is_resolved_live_from_the_profile(self) -> None:
        profile = load_hardware_profile("phone_low")
        resolved = resolve_generation_params(profile)
        budget = floor_budget()
        assert budget.max_response_tokens == resolved.max_response_tokens == 180
        assert budget.token_budget_per_turn == resolved.token_budget_per_turn == 180
        assert budget.max_context_tokens == resolved.max_context_tokens == 512
        assert budget.draft_count == resolved.draft_count == 2
        assert budget.max_refinement_passes == resolved.max_refinement_passes == 1
        assert budget.max_memory_bundle_facts == (
            profile["memory"]["max_memory_bundle_facts"]
        ) == 5

    def test_response_cap_is_the_tighter_of_the_two_token_claims(self) -> None:
        budget = floor_budget()
        assert budget.response_token_cap == min(
            budget.max_response_tokens, budget.token_budget_per_turn
        ) == 180

    def test_latency_budget_is_tf033s_phone_low_budget(self) -> None:
        assert floor_budget().latency == budget_for_profile("phone_low")
        assert floor_budget().latency.total_ms == 5000

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(Exception):
            floor_budget("bogus_profile")

    def test_budget_dict_shape(self) -> None:
        data = floor_budget().to_dict()
        assert data["profile_id"] == "phone_low"
        assert data["max_response_tokens"] == 180
        assert data["latency"]["total_ms"] == 5000


# ── The refinement-cap wiring (the slice's defect fix) ───────────────────────


class TestRefinementCapWiring:
    def test_config_surface_resolves_passes_per_profile(self) -> None:
        cases = {"phone_low": 1, "pi_zero": 0, "desktop_gpu": 3, "pi_5": 2}
        for profile_id, passes in cases.items():
            cfg = resolve_conversation_config(
                {"conversation": {"profile_id": profile_id}}
            )
            assert cfg.max_refinement_passes == passes, profile_id
            assert cfg.refinement_max_passes() == passes, profile_id

    def test_no_profile_keeps_the_standing_two_passes(self) -> None:
        cfg = resolve_conversation_config({})
        assert cfg.max_refinement_passes is None
        assert cfg.refinement_max_passes() == DEFAULT_REFINE_PASSES == 2

    def test_config_dict_round_trip_carries_the_pass_cap(self) -> None:
        cfg = resolve_conversation_config(
            {"conversation": {"profile_id": "phone_low"}}
        )
        assert cfg.to_dict()["max_refinement_passes"] == 1
        from thoughtforge.cognition.config_surface import ConversationConfig

        assert ConversationConfig.from_dict(cfg.to_dict()) == cfg

    def test_pi_zero_runs_no_refine_pass(self) -> None:
        # Turn 6 is a refining turn on the standing behaviour
        # (see the next test): under pi_zero's promised 0 passes
        # the same turn must come back best_draft, candidates only.
        engine = FloorFixtureEngine()
        core = _core("pi_zero", engine)
        turn = COHERENCE_SCRIPT[5]
        history = ChatHistory()
        history.add_user(turn.user)
        final = core.think(turn.user, history=history)
        assert final.salvage_path == "best_draft"
        assert engine.calls == 2

    def test_default_core_still_runs_two_refine_passes(self) -> None:
        # The standing behaviour, byte-identical when no profile
        # is selected (TF-028's 54-call pin rests on this too).
        engine = FloorFixtureEngine()
        core = _core(None, engine)
        turn = COHERENCE_SCRIPT[5]
        history = ChatHistory()
        history.add_user(turn.user)
        final = core.think(turn.user, history=history)
        assert final.salvage_path == "refine_pass_2"
        assert engine.calls == 4  # 2 candidates + 2 refine passes


# ── The smallest fixture model ────────────────────────────────────────────────


class TestFloorFixtureEngine:
    def test_cap_truncates_word_wise_and_is_recorded(self) -> None:
        engine = FloorFixtureEngine()
        turn = COHERENCE_SCRIPT[0]
        result = engine.generate(
            f"User request: {turn.user}",
            params=SimpleNamespace(max_tokens=5),
        )
        assert result.tokens_generated == 5
        assert result.text == " ".join(turn.coherent.split()[:5])
        assert engine.caps_seen == [5]

    def test_no_cap_leaves_the_answer_untouched(self) -> None:
        engine = FloorFixtureEngine()
        turn = COHERENCE_SCRIPT[0]
        result = engine.generate(f"User request: {turn.user}")
        assert result.text == turn.coherent
        assert engine.caps_seen == [None]

    def test_cap_above_the_answer_leaves_it_untouched(self) -> None:
        engine = FloorFixtureEngine()
        turn = COHERENCE_SCRIPT[0]
        result = engine.generate(
            f"User request: {turn.user}",
            params=SimpleNamespace(max_tokens=180),
        )
        assert result.text == turn.coherent


# ── The pure checker ──────────────────────────────────────────────────────────


class TestChecker:
    def test_salvage_passes_parsing(self) -> None:
        assert salvage_passes_used("refine_pass_1") == 1
        assert salvage_passes_used("refine_pass_2") == 2
        assert salvage_passes_used("best_draft") == 0
        assert salvage_passes_used("knowledge_only") == 0
        assert salvage_passes_used("repair") == 0
        assert salvage_passes_used("refine_pass_x") == 0

    def test_within_observation_passes_every_dimension_in_order(self) -> None:
        verdict = check_floor_turn(_observation(), floor_budget())
        assert verdict.within_floor
        assert verdict.overruns == ()
        assert [d.dimension for d in verdict.dimensions] == list(DIMENSIONS)

    def test_each_dimension_overrun_is_named(self) -> None:
        budget = floor_budget()
        cases = [
            ({"response_tokens": 181}, "response_tokens"),
            ({"prompt_tokens_max": 513}, "prompt_context"),
            ({"candidate_count": 3}, "drafts"),
            ({"refinement_passes": 2}, "refinement"),
        ]
        for overrides, dimension in cases:
            verdict = check_floor_turn(_observation(**overrides), budget)
            assert not verdict.within_floor
            assert [d.dimension for d in verdict.overruns] == [dimension]

    def test_exactly_at_cap_is_within(self) -> None:
        verdict = check_floor_turn(
            _observation(
                response_tokens=180, prompt_tokens_max=512,
                candidate_count=2, refinement_passes=1,
            ),
            floor_budget(),
        )
        assert verdict.within_floor

    def test_latency_stage_overrun_breaks_the_dimension_under_a_within_total(self) -> None:
        budget = floor_budget()
        latency = LatencyVerdict(
            profile_id=budget.profile_id,
            stages=(
                StageVerdict("retrieval", 0.0, 500),
                StageVerdict("generation", 4000.0, 3000),  # over
                StageVerdict("salvage", 0.0, 1500),
                StageVerdict("total", 4000.0, 5000),       # within
            ),
        )
        verdict = check_floor_turn(_observation(latency=latency), budget)
        assert not verdict.within_floor
        assert [d.dimension for d in verdict.overruns] == ["latency"]

    def test_verdict_dict_shape(self) -> None:
        data = check_floor_turn(_observation(), floor_budget()).to_dict()
        assert data["profile_id"] == "phone_low"
        assert data["within_floor"] is True
        assert data["dimensions"][0] == {
            "dimension": "response_tokens", "measured": 25.0,
            "cap": 180.0, "within": True,
        }


# ── End to end: the 20-turn floor run ─────────────────────────────────────────


class TestFloorRun:
    def test_twenty_turns_hold_the_floor_and_do_not_drift(self) -> None:
        engine = FloorFixtureEngine()
        core = _core("phone_low", engine)
        run = run_floor_conversation(core, engine, floor_budget())
        assert run.within_floor
        assert all(v.overruns == () for v in run.verdicts)
        assert run.report.verdict == "PASS"
        assert run.report.mean_score == 1.0 and run.report.drift == 0.0
        assert run.report.contradiction_turns == 0

    def test_floor_run_measured_maxima(self) -> None:
        engine = FloorFixtureEngine()
        core = _core("phone_low", engine)
        run = run_floor_conversation(core, engine, floor_budget())
        assert max(o.response_tokens for o in run.observations) == 30
        assert max(o.prompt_tokens_max for o in run.observations) == 338
        assert max(o.candidate_count for o in run.observations) == 2
        assert max(o.refinement_passes for o in run.observations) == 1
        # The pre-fix defect, measured gone: exactly the 7 turns
        # that used to run a forbidden second refine pass now run
        # exactly one; no turn runs two.
        assert sum(o.refinement_passes for o in run.observations) == 7
        # 40 candidate generations (2 modes × 20 turns) + 7 refine.
        assert engine.calls == 47

    def test_over_budget_answer_is_caught_by_name(self) -> None:
        class LongEngine(FloorFixtureEngine):
            def generate(self, prompt, params=None):
                self._inner.prompts.append(prompt)
                self.caps_seen.append(None)
                text = " ".join(["lantern"] * 400)
                return SimpleNamespace(text=text, tokens_generated=400)

        engine = LongEngine()
        core = _core("phone_low", engine)
        run = run_floor_conversation(
            core, engine, floor_budget(), script=COHERENCE_SCRIPT[:1]
        )
        assert not run.within_floor
        assert "response_tokens" in [
            d.dimension for d in run.verdicts[0].overruns
        ]
