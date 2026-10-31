"""
TF-080 — Memory-pressure behaviour: at the profile's
ceiling the engine sheds load by rule (context first),
never crashes mute.

The governor lives in
``src/thoughtforge/inference/memory_pressure.py``
(``PRESSURE_RULES`` / ``SHED_STEPS`` as data,
``apply_pressure()`` / ``profile_turn_load()`` /
``assess_pressure()`` / ``model_pressure()`` /
``store_pressure()``, format
``thoughtforge-memory-pressure/1``).
Write-up: ``docs/architecture/MEMORY_PRESSURE.md``.

Measured start: no memory-pressure behaviour existed
in inference or cognition (the tree's only pressure
response was the Wikidata ETL committing early at
85% host memory); the engine's token cap was a static
min-clip, pressure-blind; ``max_context_tokens`` was
used only as the load-time ``n_ctx`` and no context
ask was ever checked against it; a backend OOM became
TF-031's whole-turn degradation with no graduated
shedding attempted first.

Pinned here:

  * the filed estate: both checkers clean, rule ids
    unique, the step ladder in order with context
    first, the band thresholds as filed,
  * the bands: boundaries at 0.75 / 0.90 / 1.00,
    negative fractions refused,
  * the maximum turn loads: every field recomputed
    live from the profile JSONs, TF-056's budgets and
    the resolved generation params; the context gauge
    of a maximum load is exactly 1.0 per profile,
  * shedding by band: elevated sheds context only,
    high takes the first four steps, at_ceiling takes
    all six; steps that change nothing are not
    reported; the protected prompt is never touched,
  * never crashes mute: a prompt alone over the
    ceiling is refused BY NAME (profile, prompt size,
    ceiling in the reason) as data, not an exception;
    a supplied external fraction sheds by band with
    the after-gauge honestly None,
  * the gauges: explicit GB readings judged against
    the declared ram+vram ceilings; the model gauge
    keeps q2_k unprovable and proves q4_k_m; the
    store gauge reads TF-056's ratios live over a
    fixture store,
  * the real engine leg: a no-model TurboQuantEngine's
    resolved params match the maximum load's fields,
  * proof honesty: the export states live pressure
    not_proven, and the module never imports psutil
    (no host reading is dressed as a profile's).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.inference.memory_pressure import (
    BANDS,
    PRESSURE_FORMAT,
    PRESSURE_RULES,
    RESPONSE_FLOOR_TOKENS,
    SHED_STEPS,
    TurnLoad,
    apply_pressure,
    assess_pressure,
    band_for_fraction,
    ceiling_gb,
    context_gauge,
    max_context_tokens,
    model_pressure,
    pressure_as_data,
    pressure_live_problems,
    pressure_problems,
    profile_turn_load,
    store_pressure,
)
from thoughtforge.knowledge.memory_budgets import budget_for_profile
from thoughtforge.knowledge.models import UserFactRecord
from thoughtforge.knowledge.store import MemoryStore
from thoughtforge.utils.config import list_hardware_profiles, load_hardware_profile


def make_load(profile_id: str = "desktop_cpu", **overrides) -> TurnLoad:
    fields = dict(
        prompt_tokens=0, context_tokens=0, bundle_facts=15, sql_results=15,
        vector_top_k=8, draft_count=3, refinement_passes=2,
        max_response_tokens=400)
    fields.update(overrides)
    return TurnLoad(profile_id=profile_id, **fields)


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert pressure_problems() == []
        assert pressure_live_problems() == []

    def test_rules_steps_and_format(self):
        ids = [r["id"] for r in PRESSURE_RULES]
        assert len(ids) == len(set(ids)) == 10
        assert [s["id"] for s in SHED_STEPS] == [
            "context_halved", "retrieval_halved", "drafts_to_one",
            "refinement_off", "response_halved", "context_shed_entire"]
        assert SHED_STEPS[0]["component"] == "context"  # context first
        assert pressure_as_data()["format"] == PRESSURE_FORMAT

    def test_bands_as_filed(self):
        assert [b["id"] for b in BANDS] == [
            "normal", "elevated", "high", "at_ceiling"]
        assert [b["below"] for b in BANDS] == [0.75, 0.90, 1.00, None]


# ── The bands ──────────────────────────────────────────────────


class TestBands:
    @pytest.mark.parametrize("fraction,band", [
        (0.0, "normal"), (0.7499, "normal"), (0.75, "elevated"),
        (0.8999, "elevated"), (0.90, "high"), (0.9999, "high"),
        (1.0, "at_ceiling"), (2.5, "at_ceiling"),
    ])
    def test_band_boundaries(self, fraction, band):
        assert band_for_fraction(fraction) == band

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            band_for_fraction(-0.1)


# ── The maximum turn loads ─────────────────────────────────────


class TestProfileTurnLoad:
    def test_fields_recompute_live_for_every_profile(self):
        from thoughtforge.inference.profiles import resolve_generation_params

        for pid in list_hardware_profiles():
            profile = load_hardware_profile(pid)
            params = resolve_generation_params(profile)
            budget = budget_for_profile(pid)
            load = profile_turn_load(pid)
            assert load.context_tokens == params.max_context_tokens
            assert load.bundle_facts == budget.max_memory_bundle_facts
            assert load.sql_results == budget.sql_max_results
            assert load.vector_top_k == budget.vector_top_k
            assert load.draft_count == params.draft_count
            assert load.refinement_passes == \
                profile["generation"]["max_refinement_passes"]
            assert load.max_response_tokens == params.max_response_tokens
            assert load.prompt_tokens == 0

    def test_maximum_load_gauge_is_exactly_the_ceiling(self):
        for pid in list_hardware_profiles():
            assert context_gauge(pid, profile_turn_load(pid)) == 1.0
            assert max_context_tokens(pid) == \
                profile_turn_load(pid).context_tokens

    def test_sharp_values(self):
        assert profile_turn_load("pi_zero").to_dict() == {
            "profile_id": "pi_zero", "prompt_tokens": 0,
            "context_tokens": 256, "bundle_facts": 3, "sql_results": 3,
            "vector_top_k": 2, "draft_count": 1, "refinement_passes": 0,
            "max_response_tokens": 120, "total_context": 256}
        assert ceiling_gb("pi_zero") == 0.5
        assert ceiling_gb("server_gpu") == 88.0

    def test_unknown_profile_raises_naming_valid_ids(self):
        for fn in (profile_turn_load, ceiling_gb, max_context_tokens):
            with pytest.raises(ValueError) as excinfo:
                fn("bogus_profile")
            for pid in list_hardware_profiles():
                assert pid in str(excinfo.value)


# ── Shedding by band ───────────────────────────────────────────


class TestShedding:
    def test_normal_pressure_changes_nothing(self):
        load = make_load(prompt_tokens=10, context_tokens=100)
        report = apply_pressure("desktop_cpu", load)
        assert report.status == "no_pressure"
        assert report.steps_applied == ()
        assert report.load_after == load
        assert report.band_after == "normal"

    def test_elevated_sheds_context_only(self):
        load = make_load(prompt_tokens=100, context_tokens=3172)
        report = apply_pressure("desktop_cpu", load)
        assert report.band_before == "elevated"
        assert report.steps_applied == ("context_halved",)
        after = report.load_after
        assert after.context_tokens == 1586
        assert (after.bundle_facts, after.draft_count,
                after.refinement_passes, after.max_response_tokens) == \
            (15, 3, 2, 400)
        assert report.status == "shed"
        assert report.fraction_after == pytest.approx(1686 / 4096)

    def test_high_takes_the_first_four_steps(self):
        # gauge 3900/4096 = 0.952 -> high
        load = make_load(prompt_tokens=100, context_tokens=3800)
        report = apply_pressure("desktop_cpu", load)
        assert report.band_before == "high"
        assert report.steps_applied == (
            "context_halved", "retrieval_halved", "drafts_to_one",
            "refinement_off")
        after = report.load_after
        assert after.context_tokens == 1900
        assert (after.bundle_facts, after.sql_results,
                after.vector_top_k) == (7, 7, 4)
        assert after.draft_count == 1 and after.refinement_passes == 0
        assert after.max_response_tokens == 400  # step 5 not in high's prefix

    def test_at_ceiling_takes_all_six_and_context_goes_first(self):
        load = profile_turn_load("desktop_cpu")
        report = apply_pressure("desktop_cpu", load)
        assert report.band_before == "at_ceiling"
        assert report.steps_applied[0] == "context_halved"
        assert report.steps_applied[-1] == "context_shed_entire"
        assert len(report.steps_applied) == 6
        after = report.load_after
        assert after.context_tokens == 0
        assert after.max_response_tokens == 200
        assert report.status == "shed" and report.fraction_after == 0.0

    def test_noop_steps_are_not_reported(self):
        # pi_zero already runs one draft and zero refinement.
        load = TurnLoad(
            profile_id="pi_zero", prompt_tokens=20, context_tokens=492,
            bundle_facts=3, sql_results=3, vector_top_k=2, draft_count=1,
            refinement_passes=0, max_response_tokens=120)
        report = apply_pressure("pi_zero", load)
        assert report.steps_applied == (
            "context_halved", "retrieval_halved", "response_halved",
            "context_shed_entire")
        assert "drafts_to_one" not in report.steps_applied
        assert "refinement_off" not in report.steps_applied

    def test_response_floor_is_never_crossed(self):
        load = make_load(context_tokens=9000, max_response_tokens=40)
        report = apply_pressure("desktop_cpu", load)
        assert report.load_after.max_response_tokens == RESPONSE_FLOOR_TOKENS == 32

    def test_the_prompt_is_never_shed(self):
        for pid in ("pi_zero", "desktop_cpu", "server_gpu"):
            base = profile_turn_load(pid)
            load = TurnLoad(
                profile_id=pid, prompt_tokens=17,
                context_tokens=base.context_tokens,
                bundle_facts=base.bundle_facts, sql_results=base.sql_results,
                vector_top_k=base.vector_top_k, draft_count=base.draft_count,
                refinement_passes=base.refinement_passes,
                max_response_tokens=base.max_response_tokens)
            report = apply_pressure(pid, load)
            assert report.load_after.prompt_tokens == 17

    def test_mismatched_profile_raises(self):
        with pytest.raises(ValueError):
            apply_pressure("pi_5", make_load())
        with pytest.raises(ValueError):
            context_gauge("pi_5", make_load())


# ── Never crashes mute ─────────────────────────────────────────


class TestNeverCrashesMute:
    def test_prompt_alone_over_ceiling_is_refused_by_name(self):
        load = make_load(prompt_tokens=20_000, context_tokens=100)
        report = apply_pressure("desktop_cpu", load)
        assert report.status == "refused_named"
        assert report.load_after.context_tokens == 0  # everything sheddable shed
        for named in ("desktop_cpu", "20000", "4096"):
            assert named in report.reason
        data = report.to_dict()
        assert data["status"] == "refused_named" and data["reason"]

    def test_a_huge_context_ask_that_sheds_to_fit_is_not_refused(self):
        # 8192 asked on a 4096 profile: the at_ceiling ladder
        # sheds the sheddable context entire, the (zero) prompt
        # fits — shed, not refused.
        load = make_load(prompt_tokens=0, context_tokens=8192)
        report = apply_pressure("desktop_cpu", load)
        assert report.status == "shed"
        assert report.fraction_after == 0.0
        assert report.load_after.context_tokens == 0

    def test_supplied_fraction_sheds_by_band_with_after_gauge_none(self):
        load = make_load(prompt_tokens=10, context_tokens=100)
        report = apply_pressure("desktop_cpu", load, fraction=0.95)
        assert report.band_before == "high"
        assert report.gauge_derived is False
        assert report.fraction_after is None and report.band_after is None
        assert report.status == "shed"
        assert report.steps_applied == (
            "context_halved", "retrieval_halved", "drafts_to_one",
            "refinement_off")


# ── The gauges ─────────────────────────────────────────────────


class TestGauges:
    def test_explicit_gb_readings_against_declared_ceilings(self):
        verdict = assess_pressure("pi_zero", 0.45)
        assert verdict.ceiling_gb == 0.5
        assert verdict.fraction == pytest.approx(0.9)
        assert verdict.band == "high" and verdict.status == "measured"
        assert assess_pressure("desktop_cpu", 2.0).band == "normal"
        assert assess_pressure("server_gpu", 88.0).band == "at_ceiling"
        with pytest.raises(ValueError):
            assess_pressure("desktop_cpu", -1.0)

    def test_model_gauge_provable_and_unprovable(self):
        verdict = model_pressure("desktop_cpu", 7.0, "q4_k_m")
        assert verdict.status == "measured"
        # TF-069's accounting rounds to 4 decimals: 7 x 2/3 = 4.6667.
        assert verdict.used_gb == pytest.approx(7.0 * (2.0 / 3.0), rel=1e-3)
        assert verdict.band == "normal"
        unprovable = model_pressure("pi_zero", 0.5, "q2_k")
        assert unprovable.status == "unprovable"
        assert unprovable.used_gb is None
        assert unprovable.fraction is None and unprovable.band is None

    def test_store_gauge_reads_tf056_ratios_live(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)
        fresh = store_pressure(store, "phone_low")
        assert fresh["fraction"] == 0.0 and fresh["band"] == "normal"
        for i in range(70):
            store.append_profile_record(UserFactRecord(
                record_id=f"fct_tf080_{i:03d}", summary=f"pressure fact {i}",
                tags=[f"t{i}"], confidence=0.5))
        over = store_pressure(store, "phone_low")
        assert over["fraction"] == pytest.approx(70 / 64)
        assert over["band"] == "at_ceiling"
        assert over["ratios"]["user_fact"] == pytest.approx(70 / 64)


# ── The real engine leg ────────────────────────────────────────


class TestEngineLeg:
    def test_no_model_engine_params_match_the_maximum_load(self):
        from thoughtforge.inference.turboquant import TurboQuantEngine

        for pid in ("pi_zero", "desktop_cpu"):
            engine = TurboQuantEngine(profile_id=pid)
            try:
                engine.load()
                load = profile_turn_load(pid)
                assert engine._params is not None
                assert engine._params.max_context_tokens == load.context_tokens
                assert engine._params.draft_count == load.draft_count
                assert engine._params.max_response_tokens == \
                    load.max_response_tokens
            finally:
                engine.unload()


# ── Proof honesty ──────────────────────────────────────────────


class TestProofHonesty:
    def test_export_states_not_proven(self):
        data = pressure_as_data()
        assert data["live_proof"] == "not_proven"
        assert set(data["loads"]) == set(list_hardware_profiles())
        assert set(data["ceilings_gb"]) == set(list_hardware_profiles())

    def test_module_takes_no_host_reading(self):
        src = Path(__file__).resolve().parents[1].joinpath(
            "src", "thoughtforge", "inference",
            "memory_pressure.py").read_text(encoding="utf-8")
        assert "import psutil" not in src
        assert "psutil.virtual_memory" not in src
        assert "psutil.Process" not in src
