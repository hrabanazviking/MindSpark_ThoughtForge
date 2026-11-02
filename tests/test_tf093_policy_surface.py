"""
TF-093 — Enforcement mapped + completed: the
enforcement module's checks unified with
TF-049's cite-or-explain into one policy
surface.

The estate lives in
``src/thoughtforge/refinement/policy_surface.py``
(``POLICY_RULES`` / ``POLICY_CHECKS`` /
``POLICY_FIXTURES`` as data, ``enforce_policy``,
the checkers ``policy_surface_problems`` /
``policy_surface_live_problems``, format
``thoughtforge-policy-surface/1``). Write-up:
``docs/architecture/ENFORCEMENT_POLICY_SURFACE.md``.

Measured start: enforcement was five separate
surfaces (EnforcementGate, TF-049 per-claim,
TF-090 format, TF-091 loop, TF-092 drift) with
no single map, order, or verdict composition,
and the per-claim rule and the legacy any-QID
floor stood unreconciled since TF-049.

Pinned here:

  * the filed estate: both checkers clean, rule
    and check ids unique, the map in execution
    order, proof grade stated, the doc present,
  * the reconciliation: with records, per-claim
    law governs and the floor is recorded
    subsumed — a no-claim answer is clean here
    while the real gate's with-records AND flags
    it; without records, the legacy floor applies
    exactly as before; EnforcementGate and
    CITATION_POLICY are themselves unchanged,
  * regeneration ruled not wired: the citation
    action stays flag, regeneration_required
    stays False, no model is called,
  * signals never guessed: an unsupplied-signal
    check is recorded skipped, never passed;
    a refusal stops the pipeline and later
    checks are skipped, never passed,
  * order: repairs and rescues run before
    judgement — citation judges the rescued
    text, and a rescued text judges clean on a
    second pass,
  * structured answers judged as structure:
    prose checks skipped as not applicable,
  * verdict precedence refused > flagged >
    rescued > clean, with every flag named,
  * FragmentSalvage untouched: TF-087's
    catalogue and TF-089's metrics stand.
"""

from __future__ import annotations

import json
from pathlib import Path

from thoughtforge.knowledge.models import ActivatedRecord
from thoughtforge.refinement.citation_enforcement import CITATION_POLICY
from thoughtforge.refinement.enforcement import EnforcementGate
from thoughtforge.refinement.policy_surface import (
    POLICY_CHECKS,
    POLICY_DOC,
    POLICY_FIXTURES,
    POLICY_FORMAT,
    POLICY_RULES,
    enforce_policy,
    policy_as_data,
    policy_surface_live_problems,
    policy_surface_problems,
)
from thoughtforge.refinement.salvage_map import run_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in POLICY_FIXTURES}
_ODIN_UNCITED = (
    "Odin is the chief god of Norse mythology, a god of wisdom and war.")
_WEATHER = "The weather over the fjord is calm and bright today."


def _records() -> list[ActivatedRecord]:
    return [
        ActivatedRecord(
            "Q123", "entity", 0.9, "Odin chief god of Norse mythology",
            raw={"qid": "Q123", "label_en": "Odin",
                 "description_en": "chief god of Norse mythology",
                 "source": "wikidata"}),
        ActivatedRecord(
            "chunk_7", "reference_chunk", 0.8,
            "lantern oil kept in the hollow stores",
            raw={"chunk_id": "chunk_7", "source_file": "hollow.md",
                 "title": "The Hollow",
                 "content": "lantern oil is kept in the hollow stores",
                 "source": "reference"}),
    ]


def _run(fixture_id: str):
    fixture = _FIXTURE[fixture_id]
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _records()
    return enforce_policy(fixture["text"], **signals)


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert policy_surface_problems() == []
        assert policy_surface_live_problems() == []

    def test_format_doc_and_proof(self):
        data = policy_as_data()
        assert data["format"] == POLICY_FORMAT
        assert data["policy_doc"] == POLICY_DOC
        assert (REPO_ROOT / POLICY_DOC).exists()
        assert data["proof"]["live_model_behaviour"] == "not_proven"
        assert data["proof"]["enforcement_gate"] == "unchanged"
        assert data["proof"]["fragment_salvage"] == \
            "unchanged_by_design"
        assert data["proof"]["regeneration"] == \
            "not_wired_by_ruling_tf093"
        json.dumps(data)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in POLICY_RULES]
        assert len(ids) == len(set(ids)) == 10

    def test_the_map_in_execution_order(self):
        assert [c["id"] for c in POLICY_CHECKS] == [
            "format", "truncation", "loop", "drift",
            "citation", "length", "genericness"]
        for entry in POLICY_CHECKS:
            assert entry["owner"] and entry["source"]
            assert entry["signal"] and entry["on_failure"]

    def test_fixture_counts(self):
        assert len(POLICY_FIXTURES) == 16
        verdicts = [f["verdict"] for f in POLICY_FIXTURES]
        assert verdicts.count("clean") == 3
        assert verdicts.count("rescued") == 4
        assert verdicts.count("flagged") == 5
        assert verdicts.count("refused") == 4

    def test_every_result_carries_the_full_map_in_order(self):
        result = _run("clean_prose_no_signals")
        assert [o.id for o in result.checks] == \
            [c["id"] for c in POLICY_CHECKS]

    def test_salvage_itself_is_unchanged(self):
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"


# ── The citation reconciliation ────────────────────────────────


class TestReconciliation:
    def test_no_claim_answer_clean_with_records(self):
        result = enforce_policy(
            _WEATHER, records=_records(), retrieved_qids={"Q123"})
        assert result.verdict == "clean"
        assert result.text == _WEATHER
        citation = result.check("citation")
        assert citation.status == "passed"
        assert citation.detail["mode"] == "per_claim"
        assert citation.detail["floor"] == "subsumed_by_per_claim"

    def test_the_real_gate_still_flags_that_answer(self):
        # The contrast that makes the reconciliation a
        # policy of this surface, not an edit to the gate:
        # the gate's own with-records AND keeps its floor.
        gate_result = EnforcementGate().check(
            _WEATHER, [], {"Q123"}, records=_records())
        assert gate_result.citation_check is False

    def test_legacy_floor_applies_without_records(self):
        result = _run("legacy_floor_flagged_without_records")
        assert result.verdict == "flagged"
        assert result.check("citation").detail["mode"] == "legacy_floor"
        assert result.flags == ("citation:no_retrieved_qids_cited",)

    def test_gate_no_records_behaviour_unchanged(self):
        gate = EnforcementGate()
        failed = gate.check("No citations in this response.", [], {"Q123"})
        assert failed.citation_check is False
        assert failed.claim_check is None
        passed = gate.check(
            "According to Q123, this is a true and stated fact.",
            ["Q123"], {"Q123"})
        assert passed.citation_check is True
        assert passed.claim_check is None

    def test_citation_policy_itself_unchanged(self):
        assert CITATION_POLICY["on_uncited_claim"] == "flag"
        assert CITATION_POLICY["on_foreign_marker"] == "flag"
        assert "not wired" in CITATION_POLICY["regeneration"]

    def test_uncited_claim_flagged_with_record(self):
        result = _run("citation_uncited_claim_flagged")
        assert result.verdict == "flagged"
        assert result.flags == ("citation:uncited_claim",)
        assert result.claim_check is not None
        assert result.claim_check.uncited_markers == ["Q123"]
        assert result.claim_check.regeneration_required is False

    def test_foreign_marker_flagged(self):
        result = enforce_policy(
            "Odin is the chief god of Norse mythology [Q999].",
            records=_records())
        assert result.verdict == "flagged"
        assert "citation:foreign_marker" in result.flags
        assert "citation:uncited_claim" in result.flags

    def test_cited_claim_clean(self):
        result = _run("citation_cited_claim_clean")
        assert result.verdict == "clean"
        assert result.check("citation").status == "passed"


# ── Signals, order, precedence ─────────────────────────────────


class TestSignalsAndOrder:
    def test_unsupplied_signals_are_skipped_never_passed(self):
        result = _run("clean_prose_no_signals")
        for check_id in ("format", "truncation", "drift", "citation"):
            outcome = result.check(check_id)
            assert outcome.status == "skipped"
            assert outcome.detail["reason"].startswith(
                "signal_not_supplied")
        assert result.verdict == "clean"

    def test_refusal_stops_the_pipeline(self):
        result = _run("format_refused_structured")
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == ("no_json_span",)
        for check_id in ("loop", "drift", "citation",
                         "length", "genericness"):
            assert result.check(check_id).status == "skipped"

    def test_citation_judges_the_rescued_text(self):
        # A loop of an uncited claim: the loop rescue
        # cuts first, then citation judges what remains
        # — and still flags it, once, on the kept copy.
        result = enforce_policy(
            " ".join([_ODIN_UNCITED] * 3), records=_records())
        assert result.check("loop").status == "rescued"
        assert result.check("citation").status == "flagged"
        assert result.verdict == "flagged"
        assert result.text == _ODIN_UNCITED + (
            "\n\n[Forge: limited verified citations available "
            "for this response.]")

    def test_rescued_text_judges_clean_on_second_pass(self):
        for fixture_id in ("loop_rescued_then_judged_clean",
                           "drift_rescued_then_judged_clean",
                           "truncation_repaired"):
            first = _run(fixture_id)
            assert first.verdict == "rescued"
            fixture = _FIXTURE[fixture_id]
            signals = dict(fixture["signals"])
            if signals.get("records") == "lantern":
                signals["records"] = _records()
            second = enforce_policy(first.text or "", **signals)
            assert second.verdict == "clean"
            assert second.text == first.text

    def test_structured_answers_judged_as_structure(self):
        result = _run("format_repaired_structured")
        assert result.verdict == "rescued"
        assert result.text == '{"ferry": "dawn"}'
        for check_id in ("loop", "drift", "citation",
                         "length", "genericness"):
            outcome = result.check(check_id)
            assert outcome.status == "skipped"
            assert outcome.detail["reason"] == \
                "skipped_not_applicable_structured"

    def test_drift_flagged_verbatim_under_the_surface(self):
        result = _run("drift_flagged_verbatim")
        assert result.verdict == "flagged"
        assert result.text == _FIXTURE["drift_flagged_verbatim"]["text"]
        assert result.flags == ("drift:whole_answer_off_language",)

    def test_empty_and_unsupported_language_refused_by_name(self):
        empty = _run("empty_refused")
        assert empty.verdict == "refused"
        assert empty.problems == ("empty_output",)
        unsupported = _run("drift_unsupported_language_refused")
        assert unsupported.verdict == "refused"
        assert unsupported.problems == ("unsupported_expected_language",)
        assert unsupported.check("citation").detail["reason"] == \
            "skipped_after_refusal"

    def test_gate_legs_flag_under_the_surface(self):
        short = _run("too_short_flagged")
        assert short.verdict == "flagged"
        assert short.check("length").status == "flagged"
        assert short.flags == ("length:too_short",)
        generic = _run("generic_phrase_flagged")
        assert generic.verdict == "flagged"
        assert generic.check("genericness").status == "flagged"
        assert generic.flags == ("genericness:hard_generic_phrase",)
