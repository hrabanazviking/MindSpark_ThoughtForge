"""TF-067 — Quant quality fixtures.

Pins: the panel levels are exactly the levels the profiles +
catalogue span (re-derived live from TF-066's surfaces); the
same eight synthetic prompts at every level; every reference
answer scores a clean 1.0; the scorer's rule (retention over
required groups, a contradiction zeroes the prompt and is
named); the corpus is complete (prompts x levels, no missing
or extra cell) and a missing cell raises rather than invents;
the runner's discipline (identical prompt set per level, loss
against the panel's own fp16 row, a panel without the
reference level or with no prompts raises); the published
CURVE recomputed from the corpus exactly, including its one
filed non-monotone step (q4_k_m over q5_k_m on this corpus);
determinism (two runs identical); and the honesty boundary —
TF-066's loss claims stay unmeasured, because this curve
measures a fixture corpus, not any real model.
"""

from __future__ import annotations

import pytest

from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.quant_quality import (
    CURVE,
    FIXTURE_RESPONSES,
    NON_MONOTONE_STEPS,
    PANEL_FORMAT,
    PROMPTS_BY_ID,
    QUANT_LEVELS,
    QUANT_PROMPTS,
    REFERENCE_QUANT,
    fixture_generate,
    fixture_panel,
    quant_quality_problems,
    run_quant_panel,
    score_answer,
)
from thoughtforge.inference.turboquant_map import LOSS_CLAIMS, profile_quants


def test_panel_format_and_live_checker():
    assert PANEL_FORMAT == "thoughtforge-quant-quality-panel/1"
    assert quant_quality_problems() == []


def test_levels_are_exactly_the_spanned_levels():
    spanned = set(profile_quants().values()) | {
        m.quantization.lower() for m in ModelBrowser.CURATED_MODELS
    }
    assert set(QUANT_LEVELS) == spanned
    assert QUANT_LEVELS[0] == "iq2_xs"       # most compressed first
    assert QUANT_LEVELS[-1] == REFERENCE_QUANT == "fp16"


def test_same_prompts_every_level_and_corpus_complete():
    assert len(QUANT_PROMPTS) == 8
    assert len({p.prompt_id for p in QUANT_PROMPTS}) == 8
    expected = {(p.prompt_id, q) for p in QUANT_PROMPTS for q in QUANT_LEVELS}
    assert set(FIXTURE_RESPONSES) == expected
    assert {p.kind for p in QUANT_PROMPTS} == {"fact", "reasoning", "instruction"}


def test_reference_answers_score_clean():
    for prompt in QUANT_PROMPTS:
        scored = score_answer(prompt, prompt.reference)
        assert scored.score == 1.0, prompt.prompt_id
        assert scored.contradictions == ()


def test_scorer_rule_partial_and_contradiction():
    prompt = PROMPTS_BY_ID["red_lantern"]
    partial = score_answer(prompt, "A red lantern hangs at the ferry post.")
    assert partial.retention == 0.5 and partial.score == 0.5
    wrong = score_answer(
        prompt, "A red lantern hangs when the river is high; "
                "some call it a blue lantern.")
    assert wrong.score == 0.0 and "blue lantern" in wrong.contradictions


def test_missing_corpus_cell_raises():
    with pytest.raises(ValueError, match="no fixture response"):
        fixture_generate("q6_k", QUANT_PROMPTS[0])


def test_panel_requires_reference_and_prompts():
    with pytest.raises(ValueError, match="reference level"):
        run_quant_panel(fixture_generate, levels=("q4_k_m", "q8_0"))
    with pytest.raises(ValueError, match="at least one prompt"):
        run_quant_panel(fixture_generate, prompts=())


def test_runner_uses_any_generate_callable():
    def perfect(quant, prompt):
        return prompt.reference

    panel = run_quant_panel(perfect)
    assert all(r.mean_score == 1.0 and r.loss_vs_reference == 0.0
               for r in panel.rows)


def test_published_curve_matches_measurement():
    panel = fixture_panel()
    recomputed = {
        (r.quant, round(r.mean_score, 4), round(r.loss_vs_reference, 4),
         r.perfect, r.contradiction_prompts)
        for r in panel.rows
    }
    assert recomputed == set(CURVE)
    by_quant = {r.quant: r for r in panel.rows}
    assert by_quant["fp16"].mean_score == 1.0
    assert by_quant["q8_0"].loss_vs_reference == 0.0   # parity on this corpus
    assert by_quant["iq2_xs"].mean_score == 0.0
    assert by_quant["iq2_xs"].contradiction_prompts == 7
    assert by_quant["q2_k"].mean_score == 0.4375


def test_non_monotone_step_is_filed_not_smoothed():
    panel = fixture_panel()
    means = {r.quant: r.mean_score for r in panel.rows}
    assert NON_MONOTONE_STEPS == (("q4_k_m", "q5_k_m"),)
    assert means["q4_k_m"] > means["q5_k_m"]


def test_panel_is_deterministic():
    a = [(r.quant, r.mean_score, r.perfect) for r in fixture_panel().rows]
    b = [(r.quant, r.mean_score, r.perfect) for r in fixture_panel().rows]
    assert a == b


def test_loss_claims_stay_unmeasured():
    # The fixture curve does NOT convert TF-066's catalogue
    # claims into measurements of real models.
    assert all(not row["measured_in_tree"] for row in LOSS_CLAIMS)
