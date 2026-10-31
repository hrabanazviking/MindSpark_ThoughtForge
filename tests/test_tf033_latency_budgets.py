"""
TF-033 — Latency budgets per profile for the core loop: measured
and enforced in tests on the VM profile.

The discipline lives in
``src/thoughtforge/cognition/latency_budgets.py``; the write-up is
``docs/architecture/LATENCY_BUDGETS.md``.

Pinned here:

  * the budget table's derivation: retrieval comes live from each
    profile JSON's own ``retrieval_timeout_ms`` (never a copied
    table), generation/salvage are the campaign base scaled by the
    stated per-profile factor, and total is the stage sum by
    construction,
  * every shipped profile resolves; an unknown profile raises
    ValueError naming the valid ids (never a silent default),
  * the checker is pure and honest: stage verdicts read the turn
    state's real TF-025 timings, overruns are named in turn order,
    a total is judged only when the caller supplies a wall-clock
    measurement,
  * enforcement on the VM profile (desktop_cpu — the profile the
    selector auto-selects on this machine, re-verified live):
    real knowledge-only and stub-backend turns through the core
    land inside every stage budget and the wall-clock total,
  * a deliberately slow backend is caught by name (generation +
    total overruns) — the budget is a gate, not a decoration,
  * core surface: ``latency_budget`` follows the TF-032 selected
    profile and defaults to the VM profile; ``last_turn_latency``
    is None before the first turn and a within-budget verdict
    after a real one.

Proof grade: FIXTURE generation on the VM profile only. No real
model ran for this slice; real-model latency belongs to
TF-079/TF-175, and other profiles' budgets are derived, NOT PROVEN.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from thoughtforge.cognition.core import ThoughtForgeCore
from thoughtforge.cognition.latency_budgets import (
    BASE_GENERATION_MS,
    BASE_SALVAGE_MS,
    PROFILE_FACTORS,
    STAGES,
    VM_PROFILE_ID,
    LatencyBudget,
    all_budgets,
    budget_for_profile,
    check_turn_latency,
    stage_measurements,
)
from thoughtforge.inference.profiles import ProfileSelector
from thoughtforge.inference.unified_backend import (
    GenerationResponse,
    UnifiedBackend,
)
from thoughtforge.utils.config import (
    list_hardware_profiles,
    load_hardware_profile,
)

GOOD_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water. Wrenna the lantern-keeper checks the river level each "
    "morning and chalk-marks it on the mill door."
)


def _turn(retrieval=0, generation=0, salvage=0):
    return SimpleNamespace(
        retrieval_ms=retrieval, generation_ms=generation, salvage_ms=salvage
    )


# ── The budget table and its derivation ──────────────────────────────────────


class TestBudgetTable:
    def test_stages_are_the_three_turn_stages(self) -> None:
        assert STAGES == ("retrieval", "generation", "salvage")

    def test_every_shipped_profile_resolves(self) -> None:
        budgets = all_budgets()
        assert set(budgets) == set(list_hardware_profiles())
        assert set(budgets) == set(PROFILE_FACTORS)

    def test_retrieval_budget_is_the_profiles_own_stated_timeout(self) -> None:
        for profile_id in list_hardware_profiles():
            profile = load_hardware_profile(profile_id)
            budget = budget_for_profile(profile_id)
            assert budget.retrieval_ms == (
                profile["retrieval"]["retrieval_timeout_ms"]
            ), profile_id

    def test_generation_and_salvage_scale_by_the_stated_factor(self) -> None:
        for profile_id, factor in PROFILE_FACTORS.items():
            budget = budget_for_profile(profile_id)
            assert budget.generation_ms == round(BASE_GENERATION_MS * factor)
            assert budget.salvage_ms == round(BASE_SALVAGE_MS * factor)

    def test_vm_profile_is_the_base_unscaled(self) -> None:
        budget = budget_for_profile(VM_PROFILE_ID)
        assert VM_PROFILE_ID == "desktop_cpu"
        assert budget.generation_ms == BASE_GENERATION_MS == 1000
        assert budget.salvage_ms == BASE_SALVAGE_MS == 500
        assert budget.retrieval_ms == 500

    def test_total_is_the_stage_sum_by_construction(self) -> None:
        for budget in all_budgets().values():
            assert budget.total_ms == (
                budget.retrieval_ms + budget.generation_ms + budget.salvage_ms
            )

    def test_unknown_profile_raises_naming_valid_ids(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            budget_for_profile("bogus_profile")
        for profile_id in list_hardware_profiles():
            assert profile_id in str(excinfo.value)

    def test_stage_budget_names_every_stage_and_rejects_unknown(self) -> None:
        budget = budget_for_profile(VM_PROFILE_ID)
        assert budget.stage_budget("retrieval") == budget.retrieval_ms
        assert budget.stage_budget("total") == budget.total_ms
        with pytest.raises(ValueError):
            budget.stage_budget("routing")

    def test_budget_dict_round_trip_shape(self) -> None:
        budget = budget_for_profile("pi_zero")
        assert budget.to_dict() == {
            "profile_id": "pi_zero",
            "retrieval_ms": 2000,
            "generation_ms": 5000,
            "salvage_ms": 2500,
            "total_ms": 9500,
        }


# ── The checker ──────────────────────────────────────────────────────────────


class TestChecker:
    def test_measurements_read_the_turns_real_fields(self) -> None:
        turn = _turn(retrieval=12, generation=34, salvage=5)
        assert stage_measurements(turn) == {
            "retrieval": 12.0,
            "generation": 34.0,
            "salvage": 5.0,
        }

    def test_within_budget_turn_passes_all_stages(self) -> None:
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_turn_latency(_turn(10, 20, 5), budget, total_ms=100.0)
        assert verdict.within_budget
        assert verdict.overruns == ()
        assert [s.stage for s in verdict.stages] == [
            "retrieval", "generation", "salvage", "total",
        ]

    def test_overruns_are_named_in_turn_order(self) -> None:
        budget = LatencyBudget(
            profile_id="test", retrieval_ms=10, generation_ms=10, salvage_ms=10
        )
        verdict = check_turn_latency(_turn(50, 1, 99), budget)
        assert not verdict.within_budget
        assert [s.stage for s in verdict.overruns] == ["retrieval", "salvage"]

    def test_exactly_at_budget_is_within(self) -> None:
        budget = LatencyBudget(
            profile_id="test", retrieval_ms=10, generation_ms=10, salvage_ms=10
        )
        verdict = check_turn_latency(_turn(10, 10, 10), budget, total_ms=30.0)
        assert verdict.within_budget

    def test_no_total_judged_without_a_wall_measurement(self) -> None:
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_turn_latency(_turn(1, 1, 1), budget)
        assert [s.stage for s in verdict.stages] == list(STAGES)

    def test_salvage_zero_means_did_not_run_and_is_within(self) -> None:
        # TF-025's ruling: knowledge-only turns carry salvage_ms 0
        # ("did not run") — the verdict records the 0, within budget.
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_turn_latency(_turn(0, 0, 0), budget)
        salvage = verdict.stages[2]
        assert salvage.measured_ms == 0.0 and salvage.within

    def test_verdict_dict_shape(self) -> None:
        budget = LatencyBudget(
            profile_id="test", retrieval_ms=10, generation_ms=10, salvage_ms=10
        )
        data = check_turn_latency(_turn(1, 2, 3), budget).to_dict()
        assert data["profile_id"] == "test"
        assert data["within_budget"] is True
        assert data["stages"][1] == {
            "stage": "generation", "measured_ms": 2.0,
            "budget_ms": 10, "within": True,
        }


# ── Core wiring + enforcement on the VM profile ─────────────────────────────


def _core(config: dict | None = None, backend=None) -> ThoughtForgeCore:
    tmp = tempfile.mkdtemp()
    kwargs: dict = dict(
        memory_dir=Path(tmp) / "memory",
        db_path=Path(tmp) / "nonexistent.db",
        model_path=None,
    )
    if config is not None:
        kwargs["config"] = config
    if backend is not None:
        kwargs["backend"] = backend
    return ThoughtForgeCore(**kwargs)


class _StubBackend(UnifiedBackend):
    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    def generate(self, request):
        if self._delay_s:
            time.sleep(self._delay_s)
        return GenerationResponse(
            text=GOOD_TEXT, tokens_generated=len(GOOD_TEXT.split())
        )

    def health_check(self) -> bool:
        return True

    def backend_name(self) -> str:
        return "stub-latency"


class TestCoreSurface:
    def test_vm_profile_is_what_this_machine_auto_selects(self) -> None:
        # The enforcement below binds to the VM profile; if this
        # machine ever auto-selects another, this pin fails loudly
        # instead of the suite enforcing the wrong profile's budget.
        assert ProfileSelector().auto_select().profile_id == VM_PROFILE_ID

    def test_latency_budget_defaults_to_vm_profile(self) -> None:
        assert _core().latency_budget == budget_for_profile(VM_PROFILE_ID)

    def test_latency_budget_follows_the_selected_profile(self) -> None:
        core = _core(config={"conversation": {"profile_id": "phone_low"}})
        assert core.latency_budget == budget_for_profile("phone_low")
        assert core.latency_budget.generation_ms == 3000

    def test_last_turn_latency_none_before_first_turn(self) -> None:
        assert _core().last_turn_latency() is None


class TestEnforcementOnVmProfile:
    def test_knowledge_only_turns_within_budget(self) -> None:
        core = _core()
        for _ in range(3):
            start = time.perf_counter()
            core.think("How long does the River Sarn ferry crossing take?")
            wall_ms = (time.perf_counter() - start) * 1000
            verdict = core.last_turn_latency(total_ms=wall_ms)
            assert verdict is not None and verdict.within_budget, (
                verdict.to_dict() if verdict else None
            )

    def test_stub_backend_turns_within_budget(self) -> None:
        core = _core(backend=_StubBackend())
        for _ in range(3):
            start = time.perf_counter()
            core.think("How long does the River Sarn ferry crossing take?")
            wall_ms = (time.perf_counter() - start) * 1000
            verdict = core.last_turn_latency(total_ms=wall_ms)
            assert verdict is not None and verdict.within_budget, (
                verdict.to_dict() if verdict else None
            )

    def test_slow_backend_is_caught_by_name(self) -> None:
        # Generation budget on the VM profile is 1000 ms and the
        # total is 2000 ms; a 2.5 s backend must trip both, by name.
        core = _core(backend=_StubBackend(delay_s=2.5))
        start = time.perf_counter()
        core.think("How long does the River Sarn ferry crossing take?")
        wall_ms = (time.perf_counter() - start) * 1000
        verdict = core.last_turn_latency(total_ms=wall_ms)
        assert verdict is not None and not verdict.within_budget
        assert [s.stage for s in verdict.overruns] == ["generation", "total"]
