"""
TF-075 — Aesir + TurboQuant interplay: quantised models
served by Aesir, measured through ThoughtForge's fixtures.

The interplay lives in
``src/thoughtforge/inference/aesir_quant.py``
(``INTERPLAY_RULES`` / ``RESIDENT_MODEL_IDS`` /
``FakeAesirQuantServer`` / ``run_aesir_quant_panel``, format
``thoughtforge-aesir-quant-interplay/1``). Write-up:
``docs/architecture/AESIR_TURBOQUANT_INTERPLAY.md``.

Pinned here:

  * the filed estate: rule ids exact, both checkers clean,
    the data shape (residents / profiles / curve),
  * quant is identity, not a wire field: every resident id
    round-trips through TF-066's parser to its level; the
    real backend's payload (and every sweep payload) has
    no quant key — the level travels only in the model id,
  * D-AESIR-2's fix: TF-068's policy check judges served
    residents (match for the expected resident, mismatch
    for a wrong level, unprovable with its gap named for
    the tag-shaped catalog id) — at identity grade,
  * the per-profile residents: TF-068's chosen default,
    verdict match, curve class and decision carried
    verbatim for all six profiles; the served constrained
    exceptions stay exactly phone_low / pi_zero,
  * the sweep: TF-067's panel through the real
    AesirBackend over the fixture server reproduces
    TF-067's CURVE exactly (filed as CURVE_THROUGH_AESIR),
    deterministically, as a serial sweep — 48 calls,
    levels grouped contiguously in QUANT_LEVELS order,
  * usage: tokens_generated per level is the server's
    reported completion_tokens (the fixture server's
    contract estimate of each answer),
  * error honesty: a 404 (model-not-resident) level fails
    the whole panel run naming the resident — no false
    zero row, no partial curve,
  * a prompt the fixture serve does not know is a 422
    panel failure, not an invented answer.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from thoughtforge.inference.aesir_backend import AesirBackend
from thoughtforge.inference.aesir_quant import (
    AESIR_QUANT_FORMAT,
    CURVE_THROUGH_AESIR,
    INTERPLAY_RULES,
    RESIDENT_MODEL_IDS,
    FakeAesirQuantServer,
    aesir_quant_as_data,
    aesir_quant_live_problems,
    aesir_quant_problems,
    fixture_panel_through_aesir,
    generate_through_aesir,
    profile_residents,
    quant_of_resident,
    resident_verdict,
    run_aesir_quant_panel,
)
from thoughtforge.inference.backend_contract import estimate_tokens
from thoughtforge.inference.quant_policy import (
    LEVEL_CLASSES,
    POLICY_BY_PROFILE,
)
from thoughtforge.inference.quant_quality import (
    CURVE,
    FIXTURE_RESPONSES,
    QUANT_LEVELS,
    QUANT_PROMPTS,
)
from thoughtforge.inference.unified_backend import GenerationRequest

POST = "thoughtforge.inference.aesir_backend.requests.post"


# ── The filed estate ───────────────────────────────────────────


class TestFiledEstate:
    def test_format_and_checkers(self):
        assert AESIR_QUANT_FORMAT == \
            "thoughtforge-aesir-quant-interplay/1"
        assert aesir_quant_problems() == []
        assert aesir_quant_live_problems() == []

    def test_rule_ids_exact(self):
        assert [r["id"] for r in INTERPLAY_RULES] == [
            "quant_is_identity_not_a_wire_field",
            "one_resident_so_panel_is_serial",
            "same_panel_same_scorer_same_curve",
            "policy_check_at_identity_grade",
            "unprovable_resident_never_assumed",
            "serving_does_not_change_the_curve_class",
            "error_is_panel_failure_not_a_zero",
            "usage_counts_come_from_the_server",
            "live_proof_not_claimed",
        ]
        assert all(r["rule"] and r["measured_by"]
                   for r in INTERPLAY_RULES)

    def test_data_shape(self):
        data = aesir_quant_as_data()
        assert data["format"] == AESIR_QUANT_FORMAT
        assert data["residents"] == RESIDENT_MODEL_IDS
        assert [list(r) for r in CURVE_THROUGH_AESIR] == \
            data["curve_through_aesir"]
        assert len(data["profiles"]) == 6


# ── Quant as identity ──────────────────────────────────────────


class TestQuantAsIdentity:
    def test_residents_cover_exactly_the_panel_levels(self):
        assert set(RESIDENT_MODEL_IDS) == set(QUANT_LEVELS)
        assert len(RESIDENT_MODEL_IDS) == 6

    def test_each_resident_id_proves_its_level(self):
        for quant, resident in RESIDENT_MODEL_IDS.items():
            assert quant_of_resident(resident) == quant
            assert resident.endswith(".gguf")

    def test_payload_has_no_quant_field(self):
        payload = AesirBackend(
            model=RESIDENT_MODEL_IDS["q8_0"])._payload(
            GenerationRequest(prompt="x"), stream=False)
        assert not any("quant" in k.lower() for k in payload)
        assert payload["model"] == RESIDENT_MODEL_IDS["q8_0"]

    def test_tag_shaped_resident_proves_no_quant(self):
        # TF-074's own catalog example: an Ollama-style tag
        # carries no filename quant token at all.
        assert quant_of_resident("gemma4-e2b:latest") is None
        assert quant_of_resident("") is None
        assert quant_of_resident(None) is None


# ── The policy check, served (D-AESIR-2) ───────────────────────


class TestServedPolicyCheck:
    def test_expected_resident_matches(self):
        verdict = resident_verdict(
            "desktop_cpu", RESIDENT_MODEL_IDS["q4_k_m"])
        assert verdict.status == "match"
        assert verdict.expected == "q4_k_m"
        assert verdict.found == "q4_k_m"

    def test_wrong_level_resident_mismatches(self):
        verdict = resident_verdict(
            "desktop_cpu", RESIDENT_MODEL_IDS["q2_k"])
        assert verdict.status == "mismatch"
        assert verdict.expected == "q4_k_m"
        assert verdict.found == "q2_k"

    def test_unprovable_resident_named_not_assumed(self):
        verdict = resident_verdict("desktop_cpu", "gemma4-e2b:latest")
        assert verdict.status == "unprovable"
        assert verdict.found is None
        assert verdict.gap == "filename_proves_no_quant"

    def test_profile_residents_carry_tf068_verbatim(self):
        rows = {r["profile"]: r for r in profile_residents()}
        assert set(rows) == set(POLICY_BY_PROFILE)
        for profile_id, row in rows.items():
            policy = POLICY_BY_PROFILE[profile_id]
            assert row["quant"] == policy["chosen"]
            assert row["decision"] == policy["decision"]
            assert row["verdict"] == "match"
            assert row["curve_class"] == LEVEL_CLASSES[row["quant"]]
            assert row["resident"] == \
                RESIDENT_MODEL_IDS[policy["chosen"]]

    def test_served_constrained_exceptions_unchanged(self):
        exceptions = {r["profile"] for r in profile_residents()
                      if r["decision"] == "keep_constrained_exception"}
        assert exceptions == {"phone_low", "pi_zero"}
        for r in profile_residents():
            if r["profile"] in exceptions:
                assert r["curve_class"] == "steep_loss"


# ── The sweep through Aesir ────────────────────────────────────


class TestSweepThroughAesir:
    def test_curve_through_aesir_equals_tf067_curve(self):
        assert CURVE_THROUGH_AESIR == CURVE
        panel, _calls = fixture_panel_through_aesir()
        recomputed = [
            (r.quant, round(r.mean_score, 4),
             round(r.loss_vs_reference, 4), r.perfect,
             r.contradiction_prompts)
            for r in panel.rows
        ]
        assert recomputed == [
            (q, round(m, 4), round(l, 4), p, c)
            for q, m, l, p, c in CURVE_THROUGH_AESIR]

    def test_serial_sweep_shape(self):
        _panel, calls = fixture_panel_through_aesir()
        assert len(calls) == 48
        sequence = [quant_of_resident(c["json"]["model"])
                    for c in calls]
        assert sequence == [q for q in QUANT_LEVELS
                            for _ in QUANT_PROMPTS]
        for c in calls:
            assert c["url"].endswith("/v1/chat/completions")
            assert not any("quant" in k.lower() for k in c["json"])

    def test_sweep_is_deterministic(self):
        first, _ = fixture_panel_through_aesir()
        second, _ = fixture_panel_through_aesir()
        assert [(r.quant, r.mean_score) for r in first.rows] == \
            [(r.quant, r.mean_score) for r in second.rows]

    def test_run_with_explicit_backend_factory(self):
        server = FakeAesirQuantServer()
        with patch(POST, side_effect=server):
            panel = run_aesir_quant_panel(
                make_backend=lambda q: AesirBackend(
                    model=RESIDENT_MODEL_IDS[q]))
        assert panel.row("q8_0").mean_score == 1.0
        assert panel.row("iq2_xs").mean_score == 0.0

    def test_usage_counts_come_from_the_server(self):
        server = FakeAesirQuantServer()
        prompt = QUANT_PROMPTS[0]
        with patch(POST, side_effect=server):
            for quant in QUANT_LEVELS:
                resp = AesirBackend(
                    model=RESIDENT_MODEL_IDS[quant]).generate(
                    GenerationRequest(prompt=prompt.text))
                assert resp.tokens_generated == estimate_tokens(
                    FIXTURE_RESPONSES[(prompt.prompt_id, quant)])


# ── Error honesty ──────────────────────────────────────────────


class TestErrorHonesty:
    def test_404_level_fails_the_whole_run(self):
        with pytest.raises(RuntimeError) as excinfo:
            fixture_panel_through_aesir(
                fail_levels=frozenset({"q2_k"}))
        assert RESIDENT_MODEL_IDS["q2_k"] in str(excinfo.value)
        assert "not resident" in str(excinfo.value)

    def test_generate_through_aesir_raises_on_error(self):
        server = FakeAesirQuantServer(
            fail_levels=frozenset({"fp16"}))
        backend = AesirBackend(model=RESIDENT_MODEL_IDS["fp16"])
        with patch(POST, side_effect=server):
            with pytest.raises(RuntimeError, match="not resident"):
                generate_through_aesir(backend, QUANT_PROMPTS[0])

    def test_unknown_prompt_is_a_panel_failure(self):
        from thoughtforge.inference.quant_quality import QuantPrompt

        bogus = QuantPrompt(
            "bogus", "fact", "A question the serve never heard?",
            required=(("nothing",),), forbidden=(),
            reference="nothing")
        server = FakeAesirQuantServer()
        backend = AesirBackend(model=RESIDENT_MODEL_IDS["fp16"])
        with patch(POST, side_effect=server):
            with pytest.raises(RuntimeError, match="422"):
                generate_through_aesir(backend, bogus)
