"""TF-068 — Quant selection policy.

Pins: the policy format + live checker; the curve classes
recomputed from TF-067's published curve (catastrophic /
steep_loss / working_band / parity); the policy table
against the shipped profiles read live (one row per
profile, shipped value exact, chosen == shipped — the ADR
ratifies or excepts, it does not silently rewrite); the
six rules as data; the floor discipline (a `keep` row is
at/above the floor; the below-floor rows are exactly the
two named constrained exceptions, each with a stated curve
cost and a review trigger); never_catastrophic (iq2_xs
chosen nowhere); neighbours_not_ranked (q5_k_m chosen
nowhere — the fixture corpus cannot rank it over q4_k_m);
and the D-QUANT-1 enforcement: the verdict function
(match / mismatch / unprovable-by-absence /
unprovable-by-generic-token / no_model, unknown profile
raises) and the engine wiring — a real engine load records
the verdict for match, mismatch, unprovable and no-model
loads alike, and a mismatch load still proceeds to the
file-existence truth (announced, not refused).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.inference.quant_policy import (
    CONSTRAINED_EXCEPTIONS,
    LEVEL_CLASSES,
    POLICY,
    POLICY_BY_PROFILE,
    POLICY_FORMAT,
    POLICY_RULES,
    check_quant_for_profile,
    curve_class,
    default_for_profile,
    policy_as_data,
    quant_policy_problems,
)
from thoughtforge.inference.quant_quality import CURVE
from thoughtforge.inference.turboquant import TurboQuantEngine
from thoughtforge.inference.turboquant_map import profile_quants


def test_policy_format_and_live_checker():
    assert POLICY_FORMAT == "thoughtforge-quant-selection-policy/1"
    assert quant_policy_problems() == []


def test_curve_classes_recomputed_from_the_curve():
    by_quant = {q: (m, l) for q, m, l, _p, _c in CURVE}
    assert LEVEL_CLASSES == {
        q: curve_class(m, l) for q, (m, l) in by_quant.items()
    }
    assert LEVEL_CLASSES["iq2_xs"] == "catastrophic"
    assert LEVEL_CLASSES["q2_k"] == "steep_loss"
    assert LEVEL_CLASSES["q4_k_m"] == LEVEL_CLASSES["q5_k_m"] == "working_band"
    assert LEVEL_CLASSES["q8_0"] == LEVEL_CLASSES["fp16"] == "parity"


def test_rules_are_data():
    assert {r["id"] for r in POLICY_RULES} == {
        "quality_floor", "never_catastrophic", "neighbours_not_ranked",
        "parity_needs_no_fp16", "constrained_exception",
        "announced_not_silent",
    }


def test_policy_table_matches_shipped_profiles_live():
    live = profile_quants()
    assert {r["profile"] for r in POLICY} == set(live)
    for row in POLICY:
        assert row["shipped"] == live[row["profile"]]
        assert row["chosen"] == row["shipped"]  # ratified, not rewritten


def test_floor_discipline_and_exceptions():
    for row in POLICY:
        cls = LEVEL_CLASSES[row["chosen"]]
        if row["decision"] == "keep":
            assert cls in {"working_band", "parity"}, row["profile"]
        else:
            assert row["decision"] == "keep_constrained_exception"
            assert cls == "steep_loss"
    assert CONSTRAINED_EXCEPTIONS == {"phone_low", "pi_zero"}
    curve = {q: l for q, _m, l, _p, _c in CURVE}
    for pid in CONSTRAINED_EXCEPTIONS:
        row = POLICY_BY_PROFILE[pid]
        assert curve[row["chosen"]] == 0.5625  # the stated fixture cost
        assert row["review_trigger"]


def test_never_catastrophic_and_neighbours_not_ranked():
    chosen = {r["chosen"] for r in POLICY}
    assert "iq2_xs" not in chosen
    assert "q5_k_m" not in chosen  # the corpus cannot rank it over q4_k_m


def test_policy_as_data_is_json_safe_and_complete():
    data = policy_as_data()
    assert data["format"] == POLICY_FORMAT
    assert data["constrained_exceptions"] == ["phone_low", "pi_zero"]
    rows = {r["profile"]: r for r in data["profiles"]}
    assert rows["desktop_gpu"]["curve_class"] == "parity"
    assert rows["server_gpu"]["curve_loss"] == 0.0
    assert rows["pi_5"]["curve_mean"] == 0.8125
    import json
    json.dumps(data)


def test_default_for_profile_and_unknown():
    assert default_for_profile("desktop_cpu") == "q4_k_m"
    with pytest.raises(ValueError, match="no quant policy row"):
        default_for_profile("no_such_profile")


def test_verdict_match_mismatch_unprovable_no_model():
    v = check_quant_for_profile("desktop_cpu", "Mistral-7B-Q4_K_M.gguf")
    assert (v.status, v.found, v.expected) == ("match", "q4_k_m", "q4_k_m")
    v = check_quant_for_profile("desktop_cpu", "model-Q8_0.gguf")
    assert v.status == "mismatch" and v.found == "q8_0"
    v = check_quant_for_profile("desktop_cpu", "model.gguf")
    assert v.status == "unprovable" and v.gap == "filename_proves_no_quant"
    v = check_quant_for_profile("pi_5", "Phi-3-mini-4k-instruct-q4.gguf")
    assert v.status == "unprovable"
    assert v.gap == "filename_under_specifies_variant"
    assert v.found == "q4"  # the class is recorded, never rounded up
    v = check_quant_for_profile("server_gpu", None)
    assert v.status == "no_model" and v.expected == "fp16"


def test_engine_records_verdict_on_no_model_load():
    engine = TurboQuantEngine(model_path=None, profile_id="desktop_cpu")
    engine.load()
    assert engine.quant_verdict is not None
    assert engine.quant_verdict.status == "no_model"
    assert engine.quant_verdict.expected == "q4_k_m"


def test_engine_records_match_and_mismatch(tmp_path):
    # The files do not exist, so load() must still reach the
    # FileNotFoundError truth — the verdict is recorded before
    # that, and a mismatch is announced, not refused.
    match = TurboQuantEngine(
        model_path=tmp_path / "model-Q4_K_M.gguf", profile_id="desktop_cpu")
    with pytest.raises(FileNotFoundError):
        match.load()
    assert match.quant_verdict is not None
    assert match.quant_verdict.status == "match"

    mismatched = TurboQuantEngine(
        model_path=tmp_path / "model-Q2_K.gguf", profile_id="desktop_cpu")
    with pytest.raises(FileNotFoundError):
        mismatched.load()
    assert mismatched.quant_verdict is not None
    assert mismatched.quant_verdict.status == "mismatch"
    assert mismatched.quant_verdict.found == "q2_k"


def test_engine_records_unprovable(tmp_path):
    engine = TurboQuantEngine(
        model_path=Path("/nonexistent/path/model.gguf"),
        profile_id="desktop_cpu")
    with pytest.raises(FileNotFoundError):
        engine.load()
    assert engine.quant_verdict is not None
    assert engine.quant_verdict.status == "unprovable"
