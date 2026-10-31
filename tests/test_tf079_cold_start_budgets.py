"""
TF-079 — Cold-start budgets: time-to-first-token per
profile measured + capped in tests where the VM allows.

The budgets live in
``src/thoughtforge/inference/cold_start_budgets.py``
(``COLD_START_RULES`` / ``MEASURED_ROWS`` as data,
``budget_for_profile()`` / ``measure_cold_start()`` /
``check_cold_start()``, format
``thoughtforge-cold-start-budgets/1``).
Write-up: ``docs/architecture/COLD_START_BUDGETS.md``.

Measured start: no cold-start or TTFT measurement
existed anywhere; TF-033 budgets warm turn stages only
and defers real-model latency to this slice; TF-069
boots every profile but records no timing; the engine
load was unmeasured.

Pinned here:

  * the filed estate: both checkers clean, stages in
    boot order, rule ids unique,
  * the budget table's derivation: TF-033's
    PROFILE_FACTORS reused live (no second table),
    bases as filed, total the stage sum by
    construction, the VM profile unscaled,
  * the filed measured rows: every shipped profile,
    medians of 3 simulated boots, TTFT the stage sum,
    generation complete never before its first token,
    every filed median inside its budget,
  * the verdict: pure, at-budget boundary within,
    overruns named in boot order with the total last,
    a mismatched profile raising,
  * the deterministic cap proof: a scripted slow
    first token is caught BY NAME (first_token +
    total, profile carried), a scripted fast start
    passes, and an injectable FakeClock advanced by a
    scripted fixture delay measures exactly that
    delay as the first token,
  * streaming shape: the fixture's first chunk is
    distinct from its completed answer, and a
    simulated backend's single-chunk shape makes
    first chunk == complete by construction (the
    scripted measurement's stated default),
  * enforcement where the VM allows: a live simulated
    cold start on the VM profile (desktop_cpu, what
    this machine auto-selects) lands inside every
    stage budget and the TTFT total,
  * proof honesty: no GGUF in-tree and no llama_cpp
    on this VM, so a live model's TTFT stands
    not_proven in the export.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from thoughtforge.cognition.latency_budgets import PROFILE_FACTORS, VM_PROFILE_ID
from thoughtforge.inference.cold_start_budgets import (
    BASE_CORE_CONSTRUCTION_MS,
    BASE_ENGINE_LOAD_MS,
    BASE_FIRST_TOKEN_MS,
    BASE_PROFILE_LOAD_MS,
    COLD_START_FORMAT,
    COLD_START_RULES,
    FIXTURE_CHUNKS,
    MEASURED_ROWS,
    MEASURED_RUNS,
    STAGES,
    ColdStartBudget,
    FakeClock,
    FixtureStreamingBackend,
    all_budgets,
    budget_for_profile,
    check_cold_start,
    cold_start_as_data,
    cold_start_live_problems,
    cold_start_problems,
    measure_cold_start,
    scripted_measurement,
)
from thoughtforge.utils.config import list_hardware_profiles


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert cold_start_problems() == []
        assert cold_start_live_problems() == []

    def test_stages_in_boot_order(self):
        assert STAGES == ("profile_load", "core_construction",
                          "engine_load", "first_token")

    def test_rules_and_format(self):
        ids = [r["id"] for r in COLD_START_RULES]
        assert len(ids) == len(set(ids)) == 10
        assert cold_start_as_data()["format"] == COLD_START_FORMAT
        assert cold_start_as_data()["measured_runs"] == MEASURED_RUNS == 3


# ── The budget table and its derivation ────────────────────────


class TestBudgetTable:
    def test_every_shipped_profile_resolves(self):
        budgets = all_budgets()
        assert set(budgets) == set(list_hardware_profiles())
        assert set(budgets) == set(PROFILE_FACTORS)

    def test_budgets_reuse_tf033_factors_live(self):
        data = cold_start_as_data()
        assert data["factors"] == PROFILE_FACTORS
        for pid, factor in PROFILE_FACTORS.items():
            budget = budget_for_profile(pid)
            assert budget.profile_load_ms == round(
                BASE_PROFILE_LOAD_MS * factor)
            assert budget.core_construction_ms == round(
                BASE_CORE_CONSTRUCTION_MS * factor)
            assert budget.engine_load_ms == round(
                BASE_ENGINE_LOAD_MS * factor)
            assert budget.first_token_ms == round(
                BASE_FIRST_TOKEN_MS * factor)

    def test_vm_profile_is_the_base_unscaled(self):
        budget = budget_for_profile(VM_PROFILE_ID)
        assert VM_PROFILE_ID == "desktop_cpu"
        assert budget.to_dict() == {
            "profile_id": "desktop_cpu", "profile_load_ms": 50,
            "core_construction_ms": 500, "engine_load_ms": 1000,
            "first_token_ms": 1000, "total_ms": 2550}

    def test_total_is_the_stage_sum_by_construction(self):
        for budget in all_budgets().values():
            assert budget.total_ms == sum(
                budget.stage_budget(s) for s in STAGES)

    def test_scaled_totals(self):
        assert budget_for_profile("server_gpu").total_ms == 1275
        assert budget_for_profile("desktop_gpu").total_ms == 1913
        assert budget_for_profile("pi_5").total_ms == 5100
        assert budget_for_profile("phone_low").total_ms == 7650
        assert budget_for_profile("pi_zero").total_ms == 12750

    def test_unknown_profile_raises_naming_valid_ids(self):
        with pytest.raises(ValueError) as excinfo:
            budget_for_profile("bogus_profile")
        for pid in list_hardware_profiles():
            assert pid in str(excinfo.value)
        with pytest.raises(ValueError):
            measure_cold_start("bogus_profile")

    def test_stage_budget_rejects_unknown_stage(self):
        with pytest.raises(ValueError):
            budget_for_profile(VM_PROFILE_ID).stage_budget("routing")


# ── The filed measured rows ────────────────────────────────────


class TestMeasuredRows:
    def test_rows_cover_every_profile_with_all_keys(self):
        assert set(MEASURED_ROWS) == set(list_hardware_profiles())
        for row in MEASURED_ROWS.values():
            assert tuple(row) == (
                "profile_load_ms", "core_construction_ms",
                "engine_load_ms", "first_token_ms",
                "generation_complete_ms", "first_answer_ms", "ttft_ms")
            assert all(v > 0 for v in row.values())

    def test_filed_ttft_is_the_stage_sum(self):
        for row in MEASURED_ROWS.values():
            assert abs(row["ttft_ms"] - sum(
                row[f"{s}_ms"] for s in STAGES)) <= 0.01

    def test_filed_rows_inside_their_budgets(self):
        for pid, row in MEASURED_ROWS.items():
            budget = budget_for_profile(pid)
            for stage in STAGES:
                assert row[f"{stage}_ms"] <= budget.stage_budget(stage), (
                    pid, stage)
            assert row["ttft_ms"] <= budget.total_ms

    def test_sharp_filed_values(self):
        # The tick's record: engine load is the dominant
        # stage on every profile; the fixture first token
        # is sub-0.01 ms everywhere (no model computes).
        for row in MEASURED_ROWS.values():
            assert row["engine_load_ms"] > row["profile_load_ms"]
            assert row["first_token_ms"] < 0.01
            assert row["generation_complete_ms"] >= row["first_token_ms"]
        assert MEASURED_ROWS["desktop_cpu"]["ttft_ms"] == 181.86
        assert MEASURED_ROWS["pi_zero"]["engine_load_ms"] == 117.855


# ── The verdict ────────────────────────────────────────────────


class TestVerdict:
    def test_exactly_at_budget_is_within(self):
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_cold_start(scripted_measurement(VM_PROFILE_ID, {
            "profile_load": float(budget.profile_load_ms),
            "core_construction": float(budget.core_construction_ms),
            "engine_load": float(budget.engine_load_ms),
            "first_token": float(budget.first_token_ms)}), budget)
        assert verdict.within_budget

    def test_overruns_named_in_boot_order_total_last(self):
        budget = ColdStartBudget(
            profile_id="test", profile_load_ms=10,
            core_construction_ms=10, engine_load_ms=10, first_token_ms=10)
        verdict = check_cold_start(scripted_measurement("test", {
            "profile_load": 99.0, "core_construction": 1.0,
            "engine_load": 1.0, "first_token": 99.0}), budget)
        assert not verdict.within_budget
        assert [s.stage for s in verdict.overruns] == [
            "profile_load", "first_token", "total"]
        assert verdict.to_dict()["profile_id"] == "test"

    def test_mismatched_profile_raises(self):
        with pytest.raises(ValueError):
            check_cold_start(
                scripted_measurement("pi_5", {"first_token": 1.0}),
                budget_for_profile(VM_PROFILE_ID))

    def test_scripted_unknown_stage_raises(self):
        with pytest.raises(ValueError):
            scripted_measurement(VM_PROFILE_ID, {"routing": 1.0})


# ── The deterministic cap proof ────────────────────────────────


class TestDeterministicCapProof:
    def test_slow_first_token_caught_by_name(self):
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_cold_start(scripted_measurement(VM_PROFILE_ID, {
            "profile_load": 1.0, "core_construction": 10.0,
            "engine_load": 20.0, "first_token": 60_000.0}), budget)
        assert not verdict.within_budget
        assert verdict.profile_id == VM_PROFILE_ID == "desktop_cpu"
        assert [s.stage for s in verdict.overruns] == ["first_token", "total"]

    def test_fast_scripted_start_passes(self):
        budget = budget_for_profile(VM_PROFILE_ID)
        verdict = check_cold_start(scripted_measurement(VM_PROFILE_ID, {
            "profile_load": 1.0, "core_construction": 10.0,
            "engine_load": 20.0, "first_token": 30.0}), budget)
        assert verdict.within_budget and verdict.overruns == ()

    def test_slow_first_token_caught_on_a_scaled_profile(self):
        # The cap is per profile: pi_zero's first_token
        # budget is 5000 ms and its total is 12750 ms —
        # 6000 ms fails the stage alone (the total still
        # holds), and 20000 ms fails both, by name, with
        # no wall clock involved.
        stage_only = check_cold_start(
            scripted_measurement("pi_zero", {"first_token": 6000.0}),
            budget_for_profile("pi_zero"))
        assert [s.stage for s in stage_only.overruns] == ["first_token"]
        verdict = check_cold_start(
            scripted_measurement("pi_zero", {"first_token": 20_000.0}),
            budget_for_profile("pi_zero"))
        assert [s.stage for s in verdict.overruns] == ["first_token", "total"]

    def test_fake_clock_measures_exactly_the_scripted_delay(self):
        clock = FakeClock()
        backend = FixtureStreamingBackend(
            first_chunk_delay_ms=1234.0, clock=clock)
        measured = measure_cold_start(
            VM_PROFILE_ID, backend=backend, clock=clock)
        assert measured.first_token_ms == pytest.approx(1234.0)
        # Real work reads 0 against the fake clock — the
        # scripted delay is the only time that passes.
        assert measured.profile_load_ms == 0.0
        assert measured.engine_load_ms == 0.0
        verdict = check_cold_start(measured, budget_for_profile(VM_PROFILE_ID))
        assert not verdict.within_budget
        assert "first_token" in [s.stage for s in verdict.overruns]


# ── Streaming shape ────────────────────────────────────────────


class TestStreamingShape:
    def test_first_chunk_distinct_from_complete(self):
        measured = measure_cold_start(VM_PROFILE_ID)
        assert measured.first_chunk == FIXTURE_CHUNKS[0] == "The "
        assert measured.answer_text == "".join(FIXTURE_CHUNKS)
        assert measured.first_chunk != measured.answer_text
        assert measured.generation_complete_ms >= measured.first_token_ms

    def test_simulated_single_chunk_shape_is_equal_by_construction(self):
        # TF-072's simulated shape: one chunk that IS the
        # completed generation — the scripted measurement
        # files that default honestly (complete defaults
        # to the first-token delay, never below it).
        m = scripted_measurement(VM_PROFILE_ID, {"first_token": 42.0})
        assert m.generation_complete_ms == m.first_token_ms == 42.0
        assert m.first_answer_ms == 42.0


# ── Enforcement where the VM allows ────────────────────────────


class TestEnforcementOnVmProfile:
    def test_live_vm_cold_start_within_budget(self):
        measured = measure_cold_start(VM_PROFILE_ID)
        verdict = check_cold_start(measured, budget_for_profile(VM_PROFILE_ID))
        assert verdict.within_budget, verdict.to_dict()

    def test_live_measurement_shape(self):
        measured = measure_cold_start("pi_5")
        assert measured.profile_id == "pi_5"
        assert measured.ttft_ms == pytest.approx(
            measured.profile_load_ms + measured.core_construction_ms
            + measured.engine_load_ms + measured.first_token_ms)
        data = measured.to_dict()
        assert data["ttft_ms"] == pytest.approx(measured.ttft_ms)


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_no_live_model_grounds_exist_on_this_vm(self):
        root = Path(__file__).resolve().parents[1]
        assert list(root.rglob("*.gguf")) == []
        assert importlib.util.find_spec("llama_cpp") is None

    def test_export_states_not_proven(self):
        data = cold_start_as_data()
        assert data["live_proof"] == "not_proven"
        assert data["vm_profile_id"] == "desktop_cpu"
        assert set(data["measured_rows"]) == set(list_hardware_profiles())
        assert set(data["budgets"]) == set(list_hardware_profiles())
