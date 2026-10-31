"""
TF-089 — Salvage metrics: rescue rate + corruption
rate measured on the fixture corpus; policy
thresholds from measurements.

The estate lives in
``src/thoughtforge/refinement/salvage_metrics.py``
(``METRIC_RULES`` / ``BROKEN_OUTCOMES`` /
``ADVERSARIAL_OUTCOMES`` / ``FILED_MEASUREMENTS`` /
``THRESHOLDS`` as data, ``measure_metrics`` /
``run_broken``, the checkers
``salvage_metrics_problems`` /
``salvage_metrics_live_problems``, format
``thoughtforge-salvage-metrics/1``). Write-up:
``docs/architecture/SALVAGE_METRICS.md``.

Measured start: TF-087 mapped five absent rescues
and TF-088 guarded the refine gate, but no rate
had ever been put on either. Probed live this
tick: of seven rescue-eligible broken fixtures,
two are rescued and five return byte-identical;
all six marker fabrications are rejected, while
one of two adopted adversarial refine outputs is
TF-088's common-word residual.

Pinned here:

  * the filed estate: both checkers clean, the
    outcome tables covering each corpus fixture
    exactly once, proof grade stated,
  * the rescue numbers: 2/7 eligible, five
    unrescued verbatim named, the two non-eligible
    fixtures handled safely and out of the
    denominator,
  * the rambling share: rescued 1/2 by selection,
    lone compression 0/1 — measured, not fixed,
  * corruption in two tiers: marker adoption 0/6,
    adopted-output corruption 1/2 (the residual),
    invented facts in 2/9 outputs with attribution
    split (salvage 1, source draft 1),
  * the thresholds: absolutes at 0.0 / 1.0, the
    rest exactly the measured values.
"""

from __future__ import annotations

import pytest

from thoughtforge.refinement.salvage_metrics import (
    ADVERSARIAL_OUTCOMES,
    BROKEN_OUTCOMES,
    FILED_MEASUREMENTS,
    METRIC_RULES,
    METRICS_DOC,
    METRICS_FORMAT,
    RAMBLING_FIXTURES,
    THRESHOLDS,
    measure_metrics,
    metrics_as_data,
    public_measurements,
    run_broken,
    salvage_metrics_live_problems,
    salvage_metrics_problems,
)

_BROKEN = {r["fixture"]: r for r in BROKEN_OUTCOMES}
_ADV = {r["fixture"]: r for r in ADVERSARIAL_OUTCOMES}


@pytest.fixture(scope="module")
def measured():
    return measure_metrics()


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert salvage_metrics_problems() == []
        assert salvage_metrics_live_problems() == []

    def test_format_doc_and_proof(self):
        data = metrics_as_data()
        assert data["format"] == METRICS_FORMAT
        assert data["metrics_doc"] == METRICS_DOC
        assert data["proof"]["live_model_rates"] == "not_proven"
        assert data["proof"]["rambling_compression"] == \
            "absent_measured_zero"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in METRIC_RULES]
        assert len(ids) == len(set(ids)) == 7

    def test_live_measurements_equal_filed(self, measured):
        assert public_measurements(measured) == FILED_MEASUREMENTS


# ── Rescue ─────────────────────────────────────────────────────


class TestRescue:
    def test_corpus_shape(self, measured):
        assert measured["corpus_size"] == 18
        assert measured["broken_fixtures"] == 9
        assert measured["adversarial_fixtures"] == 9

    def test_rescue_rate_two_of_seven(self, measured):
        assert measured["rescue_eligible"] == 7
        assert measured["rescued"] == 2
        assert abs(measured["rescue_rate"] - 2 / 7) < 1e-12

    def test_the_two_rescues_named(self):
        rescued = {fid for fid, r in _BROKEN.items()
                   if r["outcome"].startswith("rescued")}
        assert rescued == {"rambling_beaten_by_cited_rival",
                           "refine_adopts_better_output"}

    def test_five_unrescued_verbatim_named(self, measured):
        unrescued = {fid for fid, r in _BROKEN.items()
                     if r["outcome"] == "unrescued_verbatim"}
        assert unrescued == {
            "truncated_mid_sentence", "rambling_alone_verbatim",
            "fragment_filter_keeps_only_scored_sentences",
            "broken_json_verbatim", "loop_verbatim"}
        assert measured["unrescued_verbatim"] == 5
        for fid in unrescued:
            fixture = measured["_broken_by_id"][fid]
            assert measured["_broken_texts"][fid] == \
                fixture["candidates"][0]

    def test_non_eligible_handled_safely(self, measured):
        assert _BROKEN["empty_candidates"]["eligible"] is False
        assert measured["_broken_paths"]["empty_candidates"] == \
            "empty"
        fid = "refine_engine_failure_falls_back"
        assert _BROKEN[fid]["eligible"] is False
        assert measured["_broken_texts"][fid] == \
            measured["_broken_by_id"][fid]["candidates"][0]

    def test_run_broken_live(self):
        result = run_broken("loop_verbatim")
        assert result.text.count("The bell rings.") == 20


# ── The rambling share ─────────────────────────────────────────


class TestRambling:
    def test_rambling_rate(self, measured):
        assert RAMBLING_FIXTURES == (
            "rambling_beaten_by_cited_rival",
            "rambling_alone_verbatim")
        assert measured["rambling_fixtures"] == 2
        assert measured["rambling_rescued"] == 1
        assert measured["rambling_rescue_rate"] == 0.5

    def test_lone_rambling_compression_is_zero(self, measured):
        # Measured, not fixed: no compression mechanism
        # exists, and this slice claims none.
        assert measured["lone_rambling_compression_rate"] == 0.0
        assert _BROKEN["rambling_alone_verbatim"]["outcome"] == \
            "unrescued_verbatim"


# ── Corruption ─────────────────────────────────────────────────


class TestCorruption:
    def test_marker_corruption_zero(self, measured):
        assert measured["fabrication_attempts"] == 6
        assert measured["fabrications_rejected"] == 6
        assert measured["fabrication_prevention_rate"] == 1.0
        assert measured["marker_adoption_rate"] == 0.0
        for fid, row in _ADV.items():
            if row["outcome"] == "fabrication_rejected":
                assert measured["_adv_texts"][fid] == \
                    measured["_adv_by_id"][fid]["candidate"]

    def test_adopted_corruption_is_the_residual(self, measured):
        assert measured["adversarial_refine_adoptions"] == 2
        assert measured["corrupted_adoptions"] == 1
        assert measured["adopted_corruption_rate"] == 0.5
        assert _ADV["common_word_elaboration_residual"][
            "outcome"] == "residual_elaboration_adopted"
        assert _ADV["benign_rephrase"]["outcome"] == \
            "clean_refine_adopted"

    def test_outputs_carrying_invented_facts_split(self, measured):
        # Two outputs carry invented facts: the residual
        # adoption (salvage's) and the source draft's own
        # inventions passed through (not salvage's).
        assert measured["outputs_carrying_invented_facts"] == 2
        assert measured[
            "salvage_attributable_corrupted_outputs"] == 1
        assert _ADV["source_invention_pass_through"][
            "outcome"] == "source_invention_pass_through"


# ── Thresholds ─────────────────────────────────────────────────


class TestThresholds:
    def test_absolutes(self):
        assert THRESHOLDS["marker_adoption_ceiling"] == 0.0
        assert THRESHOLDS["fabrication_prevention_floor"] == 1.0

    def test_thresholds_are_the_measurements(self):
        assert THRESHOLDS["rescue_rate_floor"] == \
            FILED_MEASUREMENTS["rescue_rate"]
        assert THRESHOLDS["unrescued_verbatim_ceiling"] == \
            FILED_MEASUREMENTS["unrescued_verbatim"] == 5
        assert THRESHOLDS["adopted_corruption_ceiling"] == \
            FILED_MEASUREMENTS["adopted_corruption_rate"] == 0.5
        assert THRESHOLDS["rambling_rescue_floor"] == \
            FILED_MEASUREMENTS["rambling_rescue_rate"] == 0.5

    def test_measured_rates_satisfy_thresholds(self, measured):
        assert measured["rescue_rate"] >= \
            THRESHOLDS["rescue_rate_floor"]
        assert measured["marker_adoption_rate"] <= \
            THRESHOLDS["marker_adoption_ceiling"]
        assert measured["adopted_corruption_rate"] <= \
            THRESHOLDS["adopted_corruption_ceiling"]
