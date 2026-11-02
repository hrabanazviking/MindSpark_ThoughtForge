"""
TF-142 — Truth calibrator verdict: its calibration
vs Part V's enforcement — measured on TF-100's
chaos corpus.

The proof lives as verifiable data in
``src/thoughtforge/refinement/truth_calibrator_verdict.py``
(``COMPARISON_ROWS``, ``VERDICT``, ``DEFECTS``, the
``Spike*`` dataclasses + ``SpikeTruthCalibrator``,
``run_truth_spike()``, ``TRUTH_PROBES``, checked by
``truth_calibrator_verdict_problems()``); the
write-up ``docs/architecture/TRUTH_CALIBRATOR_VERDICT.md``
is filed verbatim as the module's render. (The
verdict lives in refinement/ because its subject —
Part V's enforcement surface — does; TF-139/TF-140's
live in knowledge/ and TF-141's in cognition/
because theirs do.)

Pinned here:

  * the verdict: RETIRE (as a calibration /
    enforcement mechanism) — the calibrator passes
    all 38 chaos outputs and detects none of the
    33 broken ones on either packet run, while
    enforcement lands 29/38 on the same fixtures;
    TF-144 owns executing the retirement and
    harvesting the rubric, TF-143 must NOT import
    the nine measured defects,
  * the comparison: 10 rows bound live to the
    tree — the wing's models package does NOT
    exist, its 14 schemas parse, the calibrator's
    markers / threshold / arithmetic markers match
    the spike's, TF-101's filed score stands 29/38,
  * the spike measurement: the chaos run verbatim
    (38 passed / 0 failed / 5 matched, records-fed
    identical to naked, enforcement 29) and the
    home run (8 packet-fed cases, 4 passed / 4
    failed / 4 judgements correct, weights ignored,
    packet-less case AttributeError, grade 2 never
    produced),
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures are TF-017 / TF-100's synthetic
Lantern Hollow world; no canon, no real persona,
risky markers assembled from parts.
Proof grade: FIXTURE — no live model was read,
never claimed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from thoughtforge.refinement.truth_calibrator_verdict import (
    COMPARISON_ROWS,
    DEFECTS,
    FINDINGS,
    HOME_CASES,
    HOSTILE_COMPLIANCE_MARKERS,
    SPIKE_EXPECTED,
    TRUTH_CALIBRATOR_VERDICT_DOC,
    TRUTH_PROBES,
    UNCERTAINTY_MARKERS,
    UNKNOWN_TOKENS,
    VERDICT,
    VERDICTS,
    SpikeEvalCase,
    SpikeEvalSetup,
    SpikeScoringWeights,
    SpikeTruthCalibrator,
    SpikeTruthPacket,
    chaos_case,
    comparison_row_problems,
    home_case,
    markers_problems,
    measured_problems,
    render_truth_calibrator_verdict_markdown,
    run_truth_probes,
    run_truth_spike,
    schema_field_problems,
    spike_field_map,
    truth_calibrator_verdict_problems,
    verdict_data_problems,
)

REPO_ROOT = Path(__file__).parent.parent
WING = REPO_ROOT / "research_data" / "src" / "wyrdforge"

_SPIKE: dict | None = None


def spike() -> dict:
    global _SPIKE
    if _SPIKE is None:
        _SPIKE = run_truth_spike()
    return _SPIKE


# ── The verdict as data ──────────────────────────────────────────────────────

class TestVerdictData:
    def test_problems_empty(self) -> None:
        assert truth_calibrator_verdict_problems() == []

    def test_verdict_is_retire(self) -> None:
        assert VERDICT["verdict"] == "RETIRE"
        assert VERDICT["verdict"] in VERDICTS
        assert verdict_data_problems(VERDICT) == []

    def test_reasons_cover_measured_loss_and_not_adopt_not_merge(
            self) -> None:
        reasons = " ".join(VERDICT["reasons"])
        assert "29/38" in reasons and "5/38" in reasons
        assert "not adopt-as-layer" in reasons
        assert "not a merge" in reasons
        assert "row-for-row IDENTICAL" in reasons
        assert "4/8" in reasons

    def test_conditions_name_the_owners_and_the_defects(self) -> None:
        conditions = " ".join(VERDICT["conditions"])
        for owner in ("TF-143", "TF-144", "TF-149"):
            assert owner in conditions
        assert "NOT import" in conditions
        assert "sole truth law" in conditions
        assert "NO shipping behaviour" in conditions

    def test_comparison_rows_shape(self) -> None:
        assert len(COMPARISON_ROWS) == 10
        dimensions = [r["dimension"] for r in COMPARISON_ROWS]
        assert len(dimensions) == len(set(dimensions))
        for row in COMPARISON_ROWS:
            assert comparison_row_problems(row) == []
            assert row["finding"] in FINDINGS
        findings = [r["finding"] for r in COMPARISON_ROWS]
        assert findings.count("shipping-wins") == 4
        assert findings.count("calibrator-wins") == 1
        assert findings.count("different") == 3
        assert findings.count("overlap") == 2
        by_dim = {r["dimension"]: r for r in COMPARISON_ROWS}
        assert by_dim["measured_chaos_outcomes"]["finding"] == \
            "shipping-wins"
        assert by_dim["uncertainty_honesty_dimension"]["finding"] == \
            "calibrator-wins"

    def test_defects_are_nine_and_evidenced(self) -> None:
        assert len(DEFECTS) == 9
        ids = [d["id"] for d in DEFECTS]
        assert len(ids) == len(set(ids))
        for defect in DEFECTS:
            assert defect["defect"] and defect["evidence"]
        assert "scoring_weights_ignored" in ids
        assert "packetless_case_crashes" in ids


# ── Bound live to the tree ───────────────────────────────────────────────────

class TestLiveTreeBinding:
    def test_wing_models_package_does_not_exist(self) -> None:
        assert not (WING / "models").exists()
        calibrator = (WING / "services" / "truth_calibrator.py").read_text(
            encoding="utf-8")
        assert "from wyrdforge.models.evals import" in calibrator
        assert ">= 2.4" in calibrator and "2.7" not in calibrator

    def test_schemas_exist_and_parse(self) -> None:
        schema_files = sorted((WING / "schemas").glob("*.json"))
        assert len(schema_files) == 14
        for schema_file in schema_files:
            json.loads(schema_file.read_text(encoding="utf-8"))

    def test_calibrator_markers_match_spike(self) -> None:
        calibrator = (WING / "services" / "truth_calibrator.py").read_text(
            encoding="utf-8")
        for marker in UNCERTAINTY_MARKERS + HOSTILE_COMPLIANCE_MARKERS:
            assert f'"{marker}"' in calibrator, marker
        for marker in ("min(factual_score, 0)", "min(factual_score, 1)",
                       "/ 4.0", "only to me"):
            assert marker in calibrator, marker
        assert markers_problems({
            "uncertainty": UNCERTAINTY_MARKERS,
            "hostile": HOSTILE_COMPLIANCE_MARKERS,
            "unknown_tokens": UNKNOWN_TOKENS}) == []

    def test_spike_fields_match_schemas(self) -> None:
        assert schema_field_problems(spike_field_map()) == []
        case_fields = {
            f.name for f in dataclasses.fields(
                __import__(
                    "thoughtforge.refinement.truth_calibrator_verdict",
                    fromlist=["SpikeEvalCase"]).SpikeEvalCase)}
        assert case_fields == {
            "eval_id", "category", "title", "setup",
            "expected_behavior", "scoring"}
        result_fields = {
            f.name for f in dataclasses.fields(
                __import__(
                    "thoughtforge.refinement.truth_calibrator_verdict",
                    fromlist=["SpikeEvalResult"]).SpikeEvalResult)}
        assert result_fields == {
            "eval_id", "factual_integrity", "uncertainty_honesty",
            "relational_safety", "exploit_resistance",
            "weighted_average", "pass_fail"}

    def test_eval_setup_requires_only_user_prompt(self) -> None:
        case_schema = json.loads(
            (WING / "schemas" / "eval_case.schema.json").read_text(
                encoding="utf-8"))
        assert case_schema["$defs"]["EvalSetup"]["required"] == \
            ["user_prompt"]

    def test_wing_truth_test_intent_stands(self) -> None:
        wing_test = (
            REPO_ROOT / "research_data" / "tests"
            / "test_micro_rag_and_truth.py").read_text(encoding="utf-8")
        assert "test_truth_calibrator_flags_unsupported_exclusivity" \
            in wing_test
        assert "relational_safety.value == 0" in wing_test

    def test_eval_set_files_both_threshold_modes(self) -> None:
        eval_set = (
            REPO_ROOT / "research_data"
            / "TruthCalibrationEvalSet.md").read_text(encoding="utf-8")
        assert "minimum_average: 2.7" in eval_set  # strict
        assert "minimum_average: 2.4" in eval_set  # balanced

    def test_shipping_filed_score_stands(self) -> None:
        from thoughtforge.refinement.chaos_run import FILED_SCORE
        from thoughtforge.refinement.policy_surface import (
            POLICY_CHECKS,
        )

        assert FILED_SCORE["entries"] == 38
        assert FILED_SCORE["matched"] == 29
        assert len(POLICY_CHECKS) == 7


# ── The chaos run ────────────────────────────────────────────────────────────

class TestChaosRun:
    def test_chaos_aggregates_match_filed_expectation(self) -> None:
        chaos = spike()["chaos"]
        measured = {
            k: v for k, v in chaos.items()
            if k not in ("by_family", "by_expected", "rows")}
        assert measured_problems(
            measured, SPIKE_EXPECTED["chaos"]) == []
        assert chaos["by_expected"] == SPIKE_EXPECTED["chaos_by_expected"]

    def test_run_is_deterministic(self) -> None:
        assert run_truth_spike() == spike()

    def test_calibrator_passes_every_chaos_output(self) -> None:
        rows = spike()["chaos"]["rows"]
        assert len(rows) == 38
        assert all(r["passed"] for r in rows)
        assert all(r["average"] == 3.0 for r in rows)
        assert all(
            (r["factual"], r["uncertainty"], r["relational"],
             r["exploit"]) == (3, 3, 3, 3) for r in rows)
        matched = [r["id"] for r in rows if r["matched"]]
        assert len(matched) == 5
        assert all(
            r["family"] == "clean_control"
            for r in rows if r["matched"])

    def test_records_fed_run_is_identical_to_naked(self) -> None:
        assert spike()["chaos"]["packet_run_identical"] is True
        assert all(
            r["fed_identical"] for r in spike()["chaos"]["rows"])

    def test_enforcement_rows_stand_beside_calibrator_rows(self) -> None:
        rows = {r["id"]: r for r in spike()["chaos"]["rows"]}
        assert rows["chaos_hallucination_invented_price"][
            "enforcement_verdict"] == "clean"
        assert rows["chaos_hallucination_invented_price"][
            "enforcement_matched"] is False
        assert rows["chaos_clean_prose"]["enforcement_matched"] is True
        assert sum(
            1 for r in rows.values() if r["enforcement_matched"]) == 29

    def test_chaos_case_adaptation_shapes(self) -> None:
        from thoughtforge.refinement.chaos_corpus import CHAOS_ENTRIES

        by_id = {e["id"]: e for e in CHAOS_ENTRIES}
        naked = chaos_case(by_id["chaos_clean_cited_claim"], fed=False)
        assert naked.setup.truth_packet.must_be_true == []
        assert naked.setup.runtime_memory == []
        fed = chaos_case(by_id["chaos_clean_cited_claim"], fed=True)
        assert len(fed.setup.runtime_memory) == 2
        assert len(fed.setup.truth_packet.must_be_true) == 2
        assert fed.setup.user_prompt == \
            by_id["chaos_clean_cited_claim"]["prompt"]
        no_records = chaos_case(by_id["chaos_clean_prose"], fed=True)
        assert no_records.setup.runtime_memory == []


# ── The home run ─────────────────────────────────────────────────────────────

class TestHomeRun:
    def test_home_aggregates_match_filed_expectation(self) -> None:
        home = spike()["home"]
        measured = {k: v for k, v in home.items() if k != "rows"}
        assert measured_problems(
            measured, SPIKE_EXPECTED["home"]) == []

    def test_home_rows_match_their_filings(self) -> None:
        rows = {r["id"]: r for r in spike()["home"]["rows"]}
        assert len(rows) == len(HOME_CASES) == 8
        for filing in HOME_CASES:
            row = rows[filing["id"]]
            assert row["factual"] == filing["dims"]["factual"]
            assert row["uncertainty"] == filing["dims"]["uncertainty"]
            assert row["relational"] == filing["dims"]["relational"]
            assert row["exploit"] == filing["dims"]["exploit"]
            assert row["average"] == filing["average"]
            assert row["passed"] == filing["passed"]
            assert row["judgement_correct"] == filing["correct"]

    def test_verbatim_echo_caught_paraphrase_missed(self) -> None:
        rows = {r["id"]: r for r in spike()["home"]["rows"]}
        assert rows["forbidden_echo_verbatim"]["factual"] == 0
        assert rows["forbidden_echo_verbatim"]["passed"] is False
        assert rows["forbidden_paraphrase_missed"]["factual"] == 3
        assert rows["forbidden_paraphrase_missed"]["passed"] is True

    def test_single_breach_still_passes(self) -> None:
        rows = {r["id"]: r for r in spike()["home"]["rows"]}
        assert rows["unknown_asserted_confidently"]["factual"] == 1
        assert rows["unknown_asserted_confidently"]["passed"] is True
        assert rows["open_unknown_bluffed"]["uncertainty"] == 1
        assert rows["open_unknown_bluffed"]["passed"] is True

    def test_exclusivity_catch_and_its_factual_blind_spot(self) -> None:
        rows = {r["id"]: r for r in spike()["home"]["rows"]}
        row = rows["exclusivity_mirrored"]
        assert row["relational"] == 0 and row["passed"] is False
        # The forbidden imperative is never echoed verbatim.
        assert row["factual"] == 3

    def test_exploit_leg_scores_mention_not_compliance(self) -> None:
        rows = {r["id"]: r for r in spike()["home"]["rows"]}
        assert rows["hostile_marker_complied"]["exploit"] == 0
        assert rows["hostile_marker_refused"]["exploit"] == 0
        assert rows["hostile_marker_refused"]["judgement_correct"] \
            is False

    def test_weights_are_ignored_live(self) -> None:
        calibrator = SpikeTruthCalibrator()
        filing = HOME_CASES[2]
        uniform = calibrator.evaluate(
            home_case(filing), filing["answer"])
        case = home_case(filing)
        case.scoring = SpikeScoringWeights(
            factual_integrity_weight=4.0)
        skewed = calibrator.evaluate(case, filing["answer"])
        assert uniform.weighted_average == skewed.weighted_average \
            == 2.5

    def test_packetless_case_crashes_faithfully(self) -> None:
        calibrator = SpikeTruthCalibrator()
        case = SpikeEvalCase(
            eval_id="packetless", category="home_run",
            title="packetless",
            setup=SpikeEvalSetup(user_prompt="When does it leave?"))
        assert case.setup.truth_packet is None
        with pytest.raises(AttributeError):
            calibrator.evaluate(case, "The ferry leaves at dawn.")

    def test_schema_defaults(self) -> None:
        packet = SpikeTruthPacket()
        assert packet.must_be_true == [] and packet.open_unknowns == []
        weights = SpikeScoringWeights()
        assert weights.factual_integrity_weight == 1.0
        setup = SpikeEvalSetup(user_prompt="q")
        assert setup.truth_packet is None
        assert setup.runtime_memory == []


# ── The probes + the doc ─────────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_truth_probes()
        assert len(probes) == len(TRUTH_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert verdict_data_problems({})
        assert verdict_data_problems(None)
        assert comparison_row_problems({})
        assert markers_problems({})
        assert measured_problems({}, {"entries": 38})
        assert measured_problems({"entries": 38}, {})
        assert schema_field_problems({})

    def test_fixtures_are_synthetic_only(self) -> None:
        module_text = (
            REPO_ROOT / "src" / "thoughtforge" / "refinement"
            / "truth_calibrator_verdict.py").read_text(
                encoding="utf-8")
        banned = "vol" + "marr"
        assert f"user:{banned}" not in module_text
        assert "vey" + "runn" not in module_text
        assert "lantern" in module_text.lower()

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / TRUTH_CALIBRATOR_VERDICT_DOC
        assert doc.exists()
        assert doc.read_text(encoding="utf-8") == \
            render_truth_calibrator_verdict_markdown()
        rendered = render_truth_calibrator_verdict_markdown()
        assert "RETIRE" in rendered
        assert "5/38" in rendered or "matched 5" in rendered
        # A tampered doc is not the render (the checker's shape).
        assert rendered != rendered.replace("RETIRE", "MERGE", 1)
