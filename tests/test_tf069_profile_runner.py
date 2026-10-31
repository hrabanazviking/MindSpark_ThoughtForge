"""TF-069 — Profile runner.

Pins: the runner format + live checker; the catalogue
calibration computed live (per-quant maximum GB-per-B
from parseable catalogue entries; q2_k uncalibrated,
fp16 definitional); name parameter parsing (a count is
read only where a name carries one); the fit verdicts
(fit / exceeds_ceiling / unprovable); the HARD ceiling
(a requested model that exceeds — or cannot be
accounted — is refused before any stack leg runs; a
fitting request boots); every shipped profile's
simulated boot (all stack legs pass on the profile's
declared hardware: selector self-selection, TF-056
budget caps live in the core's lifecycle, a real core
turn, a real no-model engine load recording TF-068's
no_model verdict); the filed ceiling rows (server_gpu's
fp16 cap exceeds its own ceiling; phone_low/pi_zero
unprovable at q2_k); the filed recommended-model
failures (D-PROF-5: pi_zero's TinyLlama-1.1B against a
0.5B cap, desktop_cpu's LLaMA-3-8B against a 7.0B cap);
deployment read live (pi_zero's knowledge_subset,
server_gpu's multi_gpu extras); and the overall
verdicts (desktop_gpu boots gap-free, the rest boot
with named ceiling gaps).
"""

from __future__ import annotations

import pytest

from thoughtforge.inference.profile_runner import (
    BOOT_LEGS,
    BOOT_VERDICTS,
    RAM_ROWS,
    RECOMMENDED_FAILURES,
    RUNNER_FORMAT,
    accounted_model_gb,
    catalogue_gb_per_b,
    check_model_fit,
    gb_per_b,
    params_from_name,
    profile_runner_problems,
    run_all_boots,
    run_profile_boot,
    runner_as_data,
)
from thoughtforge.utils.config import list_hardware_profiles


@pytest.fixture(scope="module")
def boots():
    return run_all_boots()


def test_runner_format_and_live_checker():
    assert RUNNER_FORMAT == "thoughtforge-profile-runner/1"
    assert profile_runner_problems() == []


def test_legs_are_data():
    assert BOOT_LEGS == (
        "profile", "params", "selector", "memory_budget", "core",
        "engine", "ram_ceiling", "recommended_models", "deployment",
    )


def test_calibration_from_the_catalogue_live():
    calib = catalogue_gb_per_b()
    assert abs(calib["q4_k_m"]["max_gb_per_b"] - 2.0 / 3.0) < 1e-9
    assert calib["q4_k_m"]["entries"] == 8
    assert abs(calib["q8_0"]["max_gb_per_b"] - 1.1) < 1e-9
    assert abs(calib["iq2_xs"]["max_gb_per_b"] - 19.5 / 70.0) < 1e-9
    assert abs(calib["q5_k_m"]["max_gb_per_b"] - 2.5 / 3.0) < 1e-9
    # The Phi-3 rows carry no count anywhere: counted, never used.
    assert calib["q4_k_m"]["unparseable"] == 1
    assert calib["q5_k_m"]["unparseable"] == 1


def test_gb_per_b_bases():
    assert gb_per_b("fp16") == (2.0, "definition")
    per_b, basis = gb_per_b("q4_k_m")
    assert basis == "catalogue_max" and per_b is not None
    # No q2_k figure exists in-tree: unprovable, never borrowed.
    assert gb_per_b("q2_k") == (None, "uncalibrated")
    assert accounted_model_gb(1.1, "q2_k") == (None, "uncalibrated")


def test_params_from_name():
    assert params_from_name("TinyLlama-1.1B-Chat-q2_k_subset") == 1.1
    assert params_from_name("LLaMA-3-8B-Instruct-q4_k_m") == 8.0
    assert params_from_name("Gemma-2B-it-q4_k_m") == 2.0
    assert params_from_name("Phi-3-mini-4k-instruct-q4_k_m") is None
    assert params_from_name("Granite-1B-Instruct") == 1.0


def test_fit_verdicts():
    fit = check_model_fit("desktop_cpu", 7.0, "q4_k_m")
    assert fit.status == "fit"
    assert abs(fit.accounted_gb - 4.6667) < 1e-3
    assert fit.ceiling_gb == 8.0 and fit.basis == "catalogue_max"

    over = check_model_fit("server_gpu", 70.0, "fp16")
    assert over.status == "exceeds_ceiling"
    assert over.accounted_gb == 140.0 and over.ceiling_gb == 88.0
    assert over.basis == "definition"

    unknown = check_model_fit("phone_low", 1.1, "q2_k")
    assert unknown.status == "unprovable" and unknown.accounted_gb is None


def test_hard_ceiling_refuses_before_any_stack_leg():
    refused = run_profile_boot("desktop_cpu", requested_model=(70.0, "q4_k_m"))
    assert refused.verdict == "refused"
    assert refused.legs["ram_ceiling"]["status"] == "fail"
    assert refused.legs["core"]["status"] == "not_run"
    assert refused.legs["engine"]["status"] == "not_run"

    unprovable = run_profile_boot("pi_5", requested_model=(1.1, "q2_k"))
    assert unprovable.verdict == "refused"

    fitting = run_profile_boot("desktop_cpu", requested_model=(7.0, "q4_k_m"))
    assert fitting.verdict != "refused"
    assert fitting.legs["ram_ceiling"]["requested"]["status"] == "fit"


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        run_profile_boot("no_such_profile")


def test_boots_cover_exactly_the_shipped_profiles(boots):
    assert set(boots) == set(list_hardware_profiles())


def test_every_stack_leg_passes_everywhere(boots):
    for pid, report in boots.items():
        for leg in ("profile", "params", "selector", "memory_budget",
                    "core", "engine", "deployment"):
            assert report.legs[leg]["status"] == "pass", (pid, leg, report.legs[leg])


def test_selector_self_selects_on_declared_hardware(boots):
    for pid, report in boots.items():
        assert report.legs["selector"]["selected"] == pid


def test_core_booted_on_each_profile(boots):
    expected_tokens = {
        "phone_low": 180, "pi_zero": 120, "pi_5": 250,
        "desktop_cpu": 400, "desktop_gpu": 600, "server_gpu": 2000,
    }
    for pid, want in expected_tokens.items():
        leg = boots[pid].legs["core"]
        assert leg["max_response_tokens"] == want
        assert leg["response_chars"] > 0
    assert boots["pi_zero"].legs["core"]["max_refinement_passes"] == 0


def test_engine_leg_records_policy_verdict_not_host_claim(boots):
    for pid, report in boots.items():
        leg = report.legs["engine"]
        assert leg["quant_verdict"] == "no_model"
        # The backend is the host's truth, recorded as such.
        assert leg["host_backend"] in {"cpu", "vulkan", "cuda", "rocm",
                                       "metal", "directml"}


def test_memory_budget_leg(boots):
    for report in boots.values():
        leg = report.legs["memory_budget"]
        assert leg["caps_from_profile"] is True
        assert leg["fresh_store_within_budget"] is True
    assert boots["phone_low"].legs["memory_budget"]["bundle_top_k"] == 5
    assert boots["pi_zero"].legs["memory_budget"]["bundle_top_k"] == 3


def test_filed_ram_rows_recompute(boots):
    for pid, want in RAM_ROWS.items():
        assert boots[pid].legs["ram_ceiling"]["cap"]["status"] == want
    server_cap = boots["server_gpu"].legs["ram_ceiling"]["cap"]
    assert server_cap["accounted_gb"] == 140.0
    assert server_cap["ceiling_gb"] == 88.0


def test_recommended_failures_are_exactly_d_prof_5(boots):
    measured = {
        (pid, row["name"])
        for pid, report in boots.items()
        for row in report.legs["recommended_models"]["rows"]
        if row["status"] == "exceeds_cap"
    }
    assert measured == set(RECOMMENDED_FAILURES)
    # The Phi-3 rows are unprovable, never rounded into a verdict.
    pi5_rows = {r["name"]: r["status"]
                for r in boots["pi_5"].legs["recommended_models"]["rows"]}
    assert pi5_rows["Phi-3-mini-4k-instruct-q4_k_m"] == "unprovable"


def test_deployment_read_live(boots):
    assert boots["pi_zero"].legs["deployment"]["extras"] == {
        "knowledge_subset": True, "knowledge_subset_max_gb": 0.5,
    }
    assert boots["server_gpu"].legs["deployment"]["extras"] == {
        "multi_gpu": True,
    }
    assert boots["phone_low"].legs["deployment"]["onnx_export"] is True
    assert boots["phone_low"].legs["deployment"]["runtime"] == [
        "termux", "mlc-llm",
    ]


def test_boot_verdicts(boots):
    for pid, want in BOOT_VERDICTS.items():
        assert boots[pid].verdict == want
    assert boots["desktop_gpu"].verdict == "boots"


def test_runner_as_data_shape():
    data = runner_as_data()
    assert data["format"] == RUNNER_FORMAT
    assert set(data["boots"]) == set(list_hardware_profiles())
    assert data["ram_rows"] == RAM_ROWS
    assert data["recommended_failures"] == sorted(RECOMMENDED_FAILURES)
    assert data["definitional_gb_per_b"] == {"fp16": 2.0}
