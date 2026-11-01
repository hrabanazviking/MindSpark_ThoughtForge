"""
TF-126 — Small-model honesty panel: the same
suites across model sizes — where the forge helps
most, published from results.

The proof lives as verifiable data in
``benchmarks/honesty_panel.py`` (``MODEL_TIERS``,
``TRANSFORMS``, ``PANEL_ROWS``, ``TIER_MEANS``,
``FINDING``, ``PANEL_RULES``, checked by
``honesty_panel_problems()``), registered in
``benchmarks/harness.py`` with the pinned baseline
``honesty_panel_synthetic_v1``; D-BENCH-12 (the
suites never run across model sizes; no per-size
row existed; the gap assigned to TF-126 by
TF-122..TF-125's filed limits without ever being
numbered) was added and fixed by this slice in
the measuring sense.

Pinned here:

  * the measured panel: 16 cells (4 tiers x 4
    suites) of OFF / ON / delta as literal data,
    the per-tier means, and the finding — the
    forge helps most at the MID tier overall, per
    suite persona at tiny / memory at small /
    depth and presence at mid, with depth's
    small-tier delta negative, filed by name,
  * the reference tier reproduces the four suites'
    pinned baselines EXACTLY (and depth's
    reference ON its undegraded ON),
  * the transforms do what they file (citation
    drop, terse first sentence, the generic closer
    carrying a real GENERIC_PHRASES member, the
    fact swaps), and the tier anchors live in the
    catalogue with honest params,
  * the repair legs: memory's marker re-attached
    via the real RECORDS payload, persona's
    generic sentence stripped, presence's will
    line re-prepended via a real ledger replay,
    coherence ON == degraded (no repair, filed),
  * the checker clean, the harness registration
    (14 / 7 / 7 / 54), the new panel baseline
    re-measured live twice — with every earlier
    pinned baseline preserved exactly,
  * TF-017 compliance: no canon access — risky
    markers assembled from parts.

All fixtures are synthetic (TF-017). Proof grade:
FIXTURE — no live model, never claimed; live
per-size behaviour is NOT PROVEN (TF-173).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import benchmarks.honesty_panel as panel
import benchmarks.harness as harness
from benchmarks.honesty_panel import (
    FINDING,
    GENERIC_CLOSER_PHRASE,
    GENERIC_CLOSER_TEXT,
    HONESTY_PANEL_DOC,
    HONESTY_PANEL_FORMAT,
    MODEL_TIERS,
    PANEL_ROWS,
    PANEL_RULES,
    PANEL_SUITES,
    REFERENCE_BASELINES,
    TIER_BY_ID,
    TIER_MEANS,
    TRANSFORMS,
    apply_transforms,
    compute_panel_rows,
    degraded_depth_corpora,
    degraded_memory_corpora,
    degraded_persona_corpora,
    degraded_presence_corpora,
    honesty_panel_as_data,
    honesty_panel_problems,
    repair_memory_case,
    repair_will_sagas,
    repaired_memory_corpora,
    repaired_persona_corpora,
    repaired_presence_corpora,
    run_honesty_panel,
)
from benchmarks.harness import (
    BASELINES,
    BENCHMARKS,
    FIXTURES,
    METRIC_DEFINITIONS,
    baseline_problems,
    measure_honesty_panel_baseline,
)
from benchmarks.depth_suite import CHAIN_CASES
from benchmarks.memory_suite import (
    PRESENT_CASES,
    RECORDS,
    citation_for_record,
)
from benchmarks.persona_consistency import (
    GENERIC_PHRASES,
    PersonaConsistencyScorer,
)
from benchmarks.presence_suite import (
    COHERENCE_SAGAS,
    STEERING_SAGAS,
    judge_will_saga,
    run_presence_suite,
)
from thoughtforge.inference.model_browser import ModelBrowser
from thoughtforge.inference.profile_runner import params_from_name

REPO_ROOT = Path(__file__).parent.parent


def _cell(tier_id, suite_id):
    return [r for r in PANEL_ROWS
            if r["tier"] == tier_id and r["suite"] == suite_id][0]


# ── The measured panel + the finding ────────────────────────────────────────


class TestMeasuredPanel:
    def test_panel_rows_are_the_filed_measurements(self) -> None:
        assert len(PANEL_ROWS) == 16
        assert [(r["tier"], r["suite"]) for r in PANEL_ROWS] == [
            (t["id"], s) for t in MODEL_TIERS for s in PANEL_SUITES]
        for row in PANEL_ROWS:
            assert row["delta"] == pytest.approx(
                row["on_score"] - row["off_score"])
        memory_mid = _cell("mid_8b", "memory")
        assert memory_mid["off_score"] == pytest.approx(
            0.2777777777777778)
        assert memory_mid["on_score"] == pytest.approx(
            0.8055555555555557)
        assert memory_mid["delta"] == pytest.approx(
            0.5277777777777779)
        memory_small = _cell("small_3b", "memory")
        assert memory_small["on_score"] == pytest.approx(
            0.888888888888889)

    def test_reference_tier_reproduces_the_baselines_exactly(
            self) -> None:
        assert _cell("reference_14b", "persona")["off_score"] == \
            REFERENCE_BASELINES["persona_off"] == 0.8453703703703704
        assert _cell("reference_14b", "memory")["off_score"] == \
            REFERENCE_BASELINES["memory_off"] == 0.8055555555555557
        depth = _cell("reference_14b", "depth")
        assert depth["off_score"] == \
            REFERENCE_BASELINES["depth_off"] == 0.47222222222222227
        assert depth["on_score"] == \
            REFERENCE_BASELINES["depth_on"] == 0.8055555555555557
        assert _cell("reference_14b", "presence")["off_score"] == \
            REFERENCE_BASELINES["presence_off"] == 0.7777777777777778
        live = compute_panel_rows(MODEL_TIERS[:1])
        assert [dict(r) for r in live] == [
            dict(r) for r in PANEL_ROWS[:4]]

    def test_tier_means_and_the_finding(self) -> None:
        means = {m["tier"]: m for m in TIER_MEANS}
        assert means["mid_8b"]["mean_delta"] == pytest.approx(
            0.24004629629629637)
        assert means["reference_14b"]["mean_delta"] == pytest.approx(
            0.1150462962962963)
        assert means["small_3b"]["mean_delta"] == pytest.approx(
            0.12893518518518515)
        assert means["tiny_1_1b"]["mean_delta"] == pytest.approx(
            0.05912037037037035)
        assert all(m["cells"] == 4 for m in TIER_MEANS)
        # The headline, filed from the numbers: the
        # forge helps most at the MID tier, not tiny.
        assert FINDING["helps_most_tier"] == "mid_8b"
        assert FINDING["helps_most_mean_delta"] == pytest.approx(
            means["mid_8b"]["mean_delta"])
        assert FINDING["per_suite"] == {
            "persona": "tiny_1_1b", "memory": "small_3b",
            "depth": "mid_8b", "presence": "mid_8b"}

    def test_honest_cells_are_filed_by_name(self) -> None:
        depth_small = _cell("small_3b", "depth")
        assert depth_small["delta"] == pytest.approx(
            -0.19444444444444448)
        assert depth_small["on_score"] < depth_small["off_score"]
        depth_tiny = _cell("tiny_1_1b", "depth")
        assert depth_tiny["off_score"] == 0.0
        assert depth_tiny["on_score"] == 0.0
        # Presence's delta is the will repair's alone:
        # the same +0.0556 shape at every tier.
        for tier_id in ("reference_14b", "mid_8b"):
            assert _cell(tier_id, "presence")["delta"] == \
                pytest.approx(0.05555555555555558)


# ── The transforms ──────────────────────────────────────────────────────────


class TestTransforms:
    def test_citation_drop_strips_brackets_and_qids(self) -> None:
        text = "The ferry departs dawn [ref:4301]. Q12345"
        out = apply_transforms(text, ("citation_drop",))
        assert "[ref:4301]" not in out and "Q12345" not in out
        assert "The ferry departs dawn" in out
        entity = apply_transforms(
            "It carries cargo [QF4301].", ("citation_drop",))
        assert "QF4301" not in entity

    def test_terse_first_sentence(self) -> None:
        assert apply_transforms(
            "First one. Second one! Third?",
            ("terse_first_sentence",)) == "First one."
        assert apply_transforms(
            "No terminator at all",
            ("terse_first_sentence",)) == "No terminator at all"

    def test_generic_closer_carries_a_real_generic_phrase(
            self) -> None:
        assert GENERIC_CLOSER_PHRASE in GENERIC_PHRASES
        out = apply_transforms("An answer.", ("generic_closer",))
        assert out == "An answer." + GENERIC_CLOSER_TEXT
        assert GENERIC_CLOSER_PHRASE in out.lower()

    def test_fact_swap_swaps_the_filed_pairs(self) -> None:
        out = apply_transforms(
            "At Dawn the ferry rings twice, eleven minutes.",
            ("fact_swap",))
        assert out == "At Midnight the ferry rings once, " \
                      "twelve minutes."
        assert apply_transforms("TWICE", ("fact_swap",)) == "ONCE"
        # Word-boundary: a longer word is not a swap.
        assert apply_transforms(
            "dawning light", ("fact_swap",)) == "dawning light"

    def test_transforms_apply_in_the_tiers_filed_order(self) -> None:
        tiny = TIER_BY_ID["tiny_1_1b"]
        out = apply_transforms(
            "The ferry departs dawn [ref:4301]. "
            "A second sentence follows.",
            tiny["transforms"])
        assert "[ref:4301]" not in out
        assert "second sentence" not in out
        assert "midnight" in out  # first sentence, swapped
        assert out.endswith(GENERIC_CLOSER_TEXT.strip())
        ids = [t["id"] for t in TRANSFORMS]
        assert ids == ["citation_drop", "terse_first_sentence",
                       "generic_closer", "fact_swap"]


# ── The tiers ───────────────────────────────────────────────────────────────


class TestTiers:
    def test_tier_shape_and_drift_shifts(self) -> None:
        assert [t["id"] for t in MODEL_TIERS] == [
            "reference_14b", "mid_8b", "small_3b", "tiny_1_1b"]
        assert [t["prefix"] for t in MODEL_TIERS] == [
            "reference", "mid", "small", "tiny"]
        assert [t["drift_shift"] for t in MODEL_TIERS] == [0, 0, 1, 2]
        assert TIER_BY_ID["reference_14b"]["transforms"] == ()

    def test_anchors_live_in_the_catalogue_params_honest(
            self) -> None:
        catalogue = {m.filename for m in ModelBrowser.CURATED_MODELS}
        for tier in MODEL_TIERS:
            assert tier["anchor"] in catalogue, tier["id"]
        for tier_id, want in (("reference_14b", 14.0),
                              ("mid_8b", 8.0), ("small_3b", 3.0)):
            tier = TIER_BY_ID[tier_id]
            assert params_from_name(tier["anchor"]) == want
            assert tier["params"] == want
        tiny = TIER_BY_ID["tiny_1_1b"]
        # The name reader does not take the lower-case
        # token — the params come from the filename
        # itself, regex-verified, never invented.
        assert params_from_name(tiny["anchor"]) is None
        match = re.search(r"(\d+(?:\.\d+)?)b", tiny["anchor"],
                          flags=re.IGNORECASE)
        assert match is not None
        assert float(match.group(1)) == tiny["params"] == 1.1


# ── The repair legs ─────────────────────────────────────────────────────────


class TestRepairs:
    def test_memory_marker_reattached_via_the_real_payload(
            self) -> None:
        degraded = degraded_memory_corpora(TIER_BY_ID["reference_14b"])
        missing = [c for c in degraded[0]
                   if c["id"] == "p11_missing_citation"][0]
        marker = citation_for_record(RECORDS["ferry_chunk"]).marker
        assert marker == "ref:4301"
        repaired = repair_memory_case(missing)
        assert repaired["text"] == \
            f"The lantern hollow ferry departs dawn [{marker}]."
        entity = repair_memory_case(
            {"id": "x", "records": ("ferry_entity",),
             "text": "The ferry carries passengers and cargo."})
        assert entity["text"].endswith(
            f"[{citation_for_record(RECORDS['ferry_entity']).marker}].")

    def test_memory_repair_leaves_cited_and_explain_cases(
            self) -> None:
        cited = dict(PRESENT_CASES[0])
        assert repair_memory_case(cited)["text"] == cited["text"]
        degraded = degraded_memory_corpora(TIER_BY_ID["tiny_1_1b"])
        repaired = repaired_memory_corpora(degraded)
        # The explain axis runs degraded-only on both
        # roads (its holding rule is NO citation).
        assert repaired[2] == degraded[2]
        assert repaired[0] != degraded[0]

    def test_persona_generic_sentence_stripped(self) -> None:
        degraded = degraded_persona_corpora(TIER_BY_ID["tiny_1_1b"])
        repaired = repaired_persona_corpora(degraded)
        scored = PersonaConsistencyScorer().score(repaired[1])
        assert scored.generic_phrase_hits == 0
        custom = repaired_persona_corpora((
            [{"text": "The ferry leaves at dawn."
                      " As an AI, I cannot provide more detail."}],
            ["A plain line stays."],
            [{"text": "Hold the line."}]))
        assert custom[0][0]["text"] == "The ferry leaves at dawn."
        assert custom[1] == ["A plain line stays."]

    def test_presence_will_line_reprepended_by_ledger_replay(
            self) -> None:
        dropped = [s for s in STEERING_SAGAS
                   if s["id"] == "w11_goal_dropped_fails"][0]
        assert judge_will_saga(dropped)["held"] is False
        repaired = repair_will_sagas([dropped])[0]
        assert judge_will_saga(repaired)["held"] is True
        assert repaired["turns"][2]["assistant"].startswith(
            "Standing goal: restore the ferry timetable. ")

    def test_coherence_has_no_repair_on_equals_degraded(
            self) -> None:
        degraded = degraded_presence_corpora(TIER_BY_ID["small_3b"])
        repaired = repaired_presence_corpora(degraded)
        assert repaired[0] == degraded[0]
        off = run_presence_suite(*degraded)
        on = run_presence_suite(*repaired)
        assert on.coherence_score == off.coherence_score
        assert on.steering_score > off.steering_score


# ── The degradation adapters ────────────────────────────────────────────────


class TestAdapters:
    def test_coherence_drift_shift_moves_onsets_only(self) -> None:
        small = degraded_presence_corpora(TIER_BY_ID["small_3b"])[0]
        by_id = {s["id"]: s for s in small}
        filed = {s["id"]: s for s in COHERENCE_SAGAS}
        assert by_id["k05_drift_from_17"]["drift_onset"] == 16
        assert by_id["k07_drift_from_15_break"]["drift_onset"] == 14
        assert by_id["k01_coherent_full"]["drift_onset"] is None
        assert by_id["k09_use_window_slips"]["drift_turns"] == \
            filed["k09_use_window_slips"]["drift_turns"]
        tiny = degraded_presence_corpora(TIER_BY_ID["tiny_1_1b"])[0]
        tiny_by_id = {s["id"]: s for s in tiny}
        assert tiny_by_id["k07_drift_from_15_break"]["drift_onset"] == 13

    def test_depth_working_is_degraded_with_the_answers(self) -> None:
        chain, _, _ = degraded_depth_corpora(TIER_BY_ID["small_3b"])
        first = chain[0]
        filed = [c for c in CHAIN_CASES
                 if c["id"] == first["id"]][0]
        # Terseness truncates the numbered working at
        # its first period — the filed scaffold loss.
        assert first["working"] == "1."
        assert first["working"] != filed["working"]
        assert first["on_answer"] == filed["on_answer"]

    def test_memory_tiny_texts_carry_the_swapped_fact(self) -> None:
        _, true_cases, _ = degraded_memory_corpora(
            TIER_BY_ID["tiny_1_1b"])
        first = [c for c in true_cases
                 if c["id"] == "t01_ferry_dawn_true"][0]
        assert "midnight" in first["text"]
        assert "dawn" not in first["text"]


# ── The panel run + its pinned baseline ─────────────────────────────────────


class TestPanelRun:
    def test_full_run_passes_with_16_cells(self) -> None:
        result = run_honesty_panel()
        assert result.total_cells == 16
        assert result.verdict == "PASS"
        assert result.passes is True
        assert result.reference_reproduces is True
        assert [dict(r) for r in result.rows] == \
            [dict(r) for r in PANEL_ROWS]
        assert result.finding == FINDING

    def test_empty_panel_scores_zero_and_reviews(self) -> None:
        result = run_honesty_panel(tiers=())
        assert result.total_cells == 0
        assert result.overall_score == 0.0
        assert result.verdict == "REVIEW"
        assert result.passes is False

    def test_a_panel_without_the_reference_tier_reviews(
            self) -> None:
        result = run_honesty_panel(tiers=MODEL_TIERS[1:])
        assert result.total_cells == 12
        assert result.reference_reproduces is False
        assert result.verdict == "REVIEW"

    def test_panel_is_deterministic(self) -> None:
        assert compute_panel_rows() == compute_panel_rows()

    def test_panel_baseline_remeasures_live_twice_identical(
            self) -> None:
        first = measure_honesty_panel_baseline()
        second = measure_honesty_panel_baseline()
        assert first == second
        assert first["total_cells"] == 16
        assert first["mid_delta"] == pytest.approx(0.24004629629629637)
        assert first["reference_off_score"] == pytest.approx(
            0.7252314814814815)
        assert baseline_problems("honesty_panel_synthetic_v1") == []

    def test_earlier_baselines_preserved_exactly(self) -> None:
        assert harness.measure_persona_baseline()[
            "consistency_score"] == pytest.approx(0.7166666666666667)
        assert harness.measure_persona_suite_baseline()[
            "overall_score"] == pytest.approx(0.8453703703703704)
        assert harness.measure_memory_suite_baseline()[
            "overall_score"] == pytest.approx(0.8055555555555557)
        depth = harness.measure_depth_suite_baseline()
        assert depth["overall_off_score"] == \
            pytest.approx(0.47222222222222227)
        assert depth["overall_on_score"] == \
            pytest.approx(0.8055555555555557)
        assert harness.measure_presence_suite_baseline()[
            "overall_score"] == pytest.approx(0.7777777777777778)
        assert baseline_problems() == []


# ── Harness registration ────────────────────────────────────────────────────


class TestHarnessRegistration:
    def test_panel_is_registered_with_fixture_and_baseline(
            self) -> None:
        entry = [b for b in BENCHMARKS if b["id"] == "honesty_panel"]
        assert len(entry) == 1
        assert entry[0]["kind"] == "panel"
        assert entry[0]["bind_module"] == "benchmarks.honesty_panel"
        assert entry[0]["bind_attribute"] == "run_honesty_panel"
        assert entry[0]["fixture"] == "honesty_panel_corpus"
        assert entry[0]["claims_axes"] == ("depth", "presence", "will")
        fixture = [f for f in FIXTURES
                   if f["id"] == "honesty_panel_corpus"]
        assert len(fixture) == 1 and fixture[0]["synthetic"] is True
        baseline = [b for b in BASELINES
                    if b["id"] == "honesty_panel_synthetic_v1"]
        assert len(baseline) == 1
        assert baseline[0]["metrics"]["total_cells"] == 16
        assert baseline[0]["targets"]["reference_off_score"] == \
            {"target": 0.7252314814814815, "met": True}
        assert not any("latency" in m for m in baseline[0]["metrics"])

    def test_new_metrics_are_defined_and_counts_moved(self) -> None:
        defined = {(m["benchmark"], m["id"])
                   for m in METRIC_DEFINITIONS}
        for prefix in ("reference", "mid", "small", "tiny"):
            for kind in ("off_score", "on_score", "delta"):
                assert ("honesty_panel", f"{prefix}_{kind}") in defined
        assert ("honesty_panel", "total_cells") in defined
        assert len(BENCHMARKS) == 19  # + quant_panel (TF-127) + profile_panel (TF-128) + regression_gate (TF-129) + spot_checks (TF-130) + data_hygiene (TF-131), in the open
        assert len(FIXTURES) == 12  # + quant_panel_corpus (TF-127) + profile_panel_corpus (TF-128) + regression_gate_corpus (TF-129) + spot_checks_corpus (TF-130) + data_hygiene_corpus (TF-131), in the open
        assert len(BASELINES) == 12  # + quant_panel_synthetic_v1 (TF-127) + profile_panel_synthetic_v1 (TF-128) + regression_gate_synthetic_v1 (TF-129) + spot_checks_synthetic_v1 (TF-130) + data_hygiene_synthetic_v1 (TF-131), in the open
        assert len(METRIC_DEFINITIONS) == 102  # + 16 quant_panel metrics (TF-127) + 13 profile_panel metrics (TF-128) + 6 regression_gate metrics (TF-129) + 6 spot_checks metrics (TF-130) + 7 data_hygiene metrics (TF-131), in the open
        assert harness.harness_problems() == []


# ── The filing as data + the checker ────────────────────────────────────────


class TestFilingAndChecker:
    def test_format_doc_and_rules(self) -> None:
        assert HONESTY_PANEL_FORMAT == "thoughtforge-honesty-panel/1"
        assert HONESTY_PANEL_DOC == "docs/architecture/HONESTY_PANEL.md"
        assert (REPO_ROOT / HONESTY_PANEL_DOC).exists()
        ids = [r["id"] for r in PANEL_RULES]
        assert len(ids) == len(set(ids)) == 12
        for needed in ("coherence_has_no_repair",
                       "reference_tier_reproduces_the_pinned_baselines",
                       "helps_most_ties_break_toward_the_smaller_tier",
                       "live_per_size_behaviour_is_not_proven"):
            assert needed in ids
        data = honesty_panel_as_data()
        assert data["format"] == HONESTY_PANEL_FORMAT
        assert len(data["rows"]) == 16
        assert data["finding"]["helps_most_tier"] == "mid_8b"
        assert data["measured"]["verdict"] == "PASS"

    def test_honesty_panel_checker_clean_on_the_real_estate(
            self) -> None:
        assert honesty_panel_problems() == []

    def test_a_broken_filing_is_named(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(panel, "PANEL_ROWS", ())
        problems = honesty_panel_problems()
        assert any("PANEL_ROWS" in p for p in problems)


# ── TF-017 canon guard ──────────────────────────────────────────────────────


class TestCanonGuard:
    def test_no_canon_markers_in_this_test_file(self) -> None:
        text = Path(__file__).read_text(encoding="utf-8")
        canon_name = "Sig" + "rid"
        assert canon_name not in text
        data_read = "REPO_ROOT / " + '"data"'
        assert data_read not in text
        assert not re.search(r'["\']data/(knowledge|knowledge\.db)',
                             text)

    def test_panel_world_is_the_suites_synthetic_corpora(self) -> None:
        data = honesty_panel_as_data()
        assert data["measured"]["total_cells"] == 16
        joined = " ".join(t["id"] for t in data["tiers"])
        assert "tiny" in joined and "reference" in joined
