"""
TF-101 — Enforcement + chaos: the full policy
stack run over TF-100; outcomes scored
(rescued / refused / passed-clean).

The estate lives in
``src/thoughtforge/refinement/chaos_run.py``
(``SCORING_RULES`` / ``FILED_SCORE`` /
``FILED_RESULTS`` / ``NAMED_GAPS`` as data,
``run_chaos`` / ``run_chaos_entry`` /
``resolve_entry_signals``, the checkers
``chaos_run_problems`` /
``chaos_run_live_problems``, format
``thoughtforge-chaos-run/1``); the corpus is
loaded through TF-100's module from its JSON
file, never hardcoded here. Write-up:
``docs/architecture/CHAOS_RUN.md``.

Measured start: TF-100 filed the corpus as
demands and proved only the cheap legs gate
by gate, leaving draft-side hallucination,
generic filler and the invented reference
unjudged; TF-093 filed a live model's
behaviour under its surface NOT PROVEN with
TF-101 owning this score.

Pinned here (the measured truth, unretuned):

  * the filed estate: both checkers clean, the
    doc present, format and proof grades
    stated, rule ids unique,
  * the aggregate score exactly as measured:
    29/38 — rescued 12/12, refused 12/21,
    passed-clean 5/5; stack verdicts rescued
    12 / refused 4 / flagged 8 / clean 14,
  * per-family counts exactly (hallucination,
    persona-voice loss and genericness at
    0 matched; every other family complete),
  * every clean control clean byte-identical
    through the full stack — zero over-rescue,
  * zero fabrication rescues among the 7
    hallucination + fabricated_citation
    entries, with the companion truth pinned:
    the 4 hallucination entries pass clean,
  * the 9 mismatches exactly, each an
    expected-refused entry passed clean,
    each owned by exactly one named gap
    (TF-088 / TF-096 / TF-093),
  * the persona contrast: the stack passes
    all three persona entries clean while
    TF-096's diagnostic names a new violation
    in each — resolved, not consumed, and
    never scored,
  * the invented-reference caveat: matched by
    a blocking flag, whose mechanism is an
    incidental citation match, not reference
    verification,
  * tamper legs: a moved score, a flipped
    filed outcome and an uncovered mismatch
    are each named by the data checker.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from thoughtforge.cognition.persona import FIXTURE_EXPECTATION
from thoughtforge.refinement.chaos_corpus import CHAOS_ENTRIES
from thoughtforge.refinement.chaos_run import (
    CHAOS_RUN_DOC,
    CHAOS_RUN_FORMAT,
    ENTRY_NOTES,
    FILED_RESULTS,
    FILED_SCORE,
    NAMED_GAPS,
    SCORING_RULES,
    chaos_run_as_data,
    chaos_run_live_problems,
    chaos_run_problems,
    is_match,
    mapped_actual,
    resolve_entry_signals,
    run_chaos,
)
from thoughtforge.refinement.policy_surface import POLICY_CHECKS

REPO_ROOT = Path(__file__).resolve().parents[1]

_RUN = run_chaos()
_ROWS = {r["id"]: r for r in _RUN["results"]}
_ENTRY = {e["id"]: e for e in CHAOS_ENTRIES}
_MISMATCH_IDS = {
    "chaos_hallucination_invented_price",
    "chaos_hallucination_invented_destinations",
    "chaos_hallucination_negation_flip",
    "chaos_hallucination_common_word_elaboration",
    "chaos_voice_handwaving",
    "chaos_voice_repetition",
    "chaos_voice_invented_crossings",
    "chaos_generic_filler_non_answer",
    "chaos_generic_vague_restatement",
}


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert chaos_run_problems() == []
        assert chaos_run_live_problems() == []

    def test_format_doc_and_proof(self):
        data = chaos_run_as_data()
        assert data["format"] == CHAOS_RUN_FORMAT == \
            "thoughtforge-chaos-run/1"
        assert data["chaos_run_doc"] == CHAOS_RUN_DOC
        assert (REPO_ROOT / CHAOS_RUN_DOC).exists()
        assert data["proof"]["grade"] == "fixture"
        assert data["proof"]["live_model_score"] == "not_proven"
        assert data["proof"]["stack"] == \
            "tf093_enforce_policy_unchanged"
        assert data["proof"]["corpus"] == \
            "tf100_chaos_corpus_unchanged"
        assert data["proof"]["tf016_totals"] == "unchanged_90"
        json.dumps(data)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in SCORING_RULES]
        assert len(ids) == len(set(ids)) == 8

    def test_run_covers_the_corpus_in_order(self):
        assert [r["id"] for r in _RUN["results"]] == \
            [e["id"] for e in CHAOS_ENTRIES]
        assert len(_RUN["results"]) == 38
        assert [r["id"] for r in FILED_RESULTS] == \
            [e["id"] for e in CHAOS_ENTRIES]

    def test_every_row_carries_the_full_check_map(self):
        want = [c["id"] for c in POLICY_CHECKS]
        for row in _RUN["results"]:
            assert [c["id"] for c in row["checks"]] == want, row["id"]


# ── The score, exactly as measured ─────────────────────────────


class TestScore:
    def test_overall_score(self):
        assert _RUN["score"] == FILED_SCORE
        assert FILED_SCORE["entries"] == 38
        assert FILED_SCORE["matched"] == 29
        assert FILED_SCORE["mismatched"] == 9
        assert FILED_SCORE["score"] == pytest.approx(29 / 38)

    def test_per_family_counts(self):
        assert FILED_SCORE["by_family"] == {
            "truncation": {"entries": 3, "matched": 3},
            "loop_sentence": {"entries": 3, "matched": 3},
            "loop_cycle": {"entries": 2, "matched": 2},
            "loop_word": {"entries": 2, "matched": 2},
            "hallucination": {"entries": 4, "matched": 0},
            "fabricated_citation": {"entries": 3, "matched": 3},
            "format_collapse": {"entries": 4, "matched": 4},
            "language_drift": {"entries": 3, "matched": 3},
            "persona_voice_loss": {"entries": 3, "matched": 0},
            "genericness": {"entries": 2, "matched": 0},
            "refusal_shaped": {"entries": 2, "matched": 2},
            "empty_or_short": {"entries": 2, "matched": 2},
            "clean_control": {"entries": 5, "matched": 5}}

    def test_per_expected_outcome_counts(self):
        assert FILED_SCORE["by_expected_outcome"] == {
            "rescued": {"entries": 12, "matched": 12},
            "refused": {"entries": 21, "matched": 12},
            "passed-clean": {"entries": 5, "matched": 5}}

    def test_stack_verdict_counts(self):
        assert FILED_SCORE["stack_verdicts"] == {
            "clean": 14, "rescued": 12, "flagged": 8, "refused": 4}

    def test_every_demanded_rescue_delivered(self):
        rescued = [r for r in _RUN["results"]
                   if r["expected"] == "rescued"]
        assert len(rescued) == 12
        assert all(r["stack_verdict"] == "rescued" and r["matched"]
                   for r in rescued)


# ── Clean controls ─────────────────────────────────────────────


class TestCleanControls:
    def test_controls_clean_byte_identical(self):
        controls = [r for r in _RUN["results"]
                    if r["family"] == "clean_control"]
        assert len(controls) == 5
        for row in controls:
            assert row["stack_verdict"] == "clean", row["id"]
            assert row["matched"], row["id"]
            assert row["byte_identical"], row["id"]
            assert row["output_text"] == _ENTRY[row["id"]]["text"]

    def test_no_over_rescue_anywhere(self):
        for row in _RUN["results"]:
            if row["expected"] == "passed-clean":
                assert row["actual"] == "passed-clean", row["id"]
                assert row["flags"] == [], row["id"]


# ── Fabrication ────────────────────────────────────────────────


class TestFabrication:
    def _fabrication_rows(self):
        return [r for r in _RUN["results"] if r["family"] in
                ("hallucination", "fabricated_citation")]

    def test_zero_fabrication_rescues(self):
        rows = self._fabrication_rows()
        assert len(rows) == 7 == FILED_SCORE["fabrication_entries"]
        assert [r["id"] for r in rows
                if r["stack_verdict"] == "rescued"] == []
        assert FILED_SCORE["fabrication_rescues"] == 0

    def test_fabrication_passed_clean_is_the_hallucination_family(self):
        passed = [r["id"] for r in self._fabrication_rows()
                  if r["stack_verdict"] == "clean"]
        assert len(passed) == 4 == \
            FILED_SCORE["fabrication_passed_clean"]
        assert all(_ROWS[i]["family"] == "hallucination"
                   for i in passed)

    def test_fabricated_citations_all_flagged_and_matched(self):
        rows = [r for r in _RUN["results"]
                if r["family"] == "fabricated_citation"]
        assert len(rows) == 3
        for row in rows:
            assert row["stack_verdict"] == "flagged", row["id"]
            assert row["matched"], row["id"]
            assert row["flags"], row["id"]


# ── Mismatches and their named gaps ────────────────────────────


class TestMismatches:
    def test_mismatch_ids_exactly(self):
        assert {m["id"] for m in _RUN["mismatches"]} == _MISMATCH_IDS
        assert len(_RUN["mismatches"]) == 9

    def test_every_mismatch_is_refused_demanded_and_clean_passed(self):
        for mismatch in _RUN["mismatches"]:
            row = _ROWS[mismatch["id"]]
            assert row["expected"] == "refused", row["id"]
            assert row["stack_verdict"] == "clean", row["id"]
            assert row["actual"] == "passed-clean", row["id"]
            assert row["byte_identical"], row["id"]
            assert row["reason"], row["id"]
            assert row["gap"] and row["gap_owner"], row["id"]

    def test_gaps_partition_the_mismatches_with_owners(self):
        covered = [eid for gap in NAMED_GAPS
                   for eid in gap["entries"]]
        assert sorted(covered) == sorted(_MISMATCH_IDS)
        assert {g["id"]: g["owner"] for g in NAMED_GAPS} == {
            "draft_side_hallucination_not_judged_by_the_policy_surface":
                "TF-088",
            "persona_voice_not_a_policy_surface_check": "TF-096",
            "genericness_checks_hard_phrases_only": "TF-093"}
        by_family = {g["id"]: { _ROWS[e]["family"]
                                for e in g["entries"]}
                     for g in NAMED_GAPS}
        assert by_family[
            "draft_side_hallucination_not_judged_by_the_policy_surface"
        ] == {"hallucination"}
        assert by_family["persona_voice_not_a_policy_surface_check"] == \
            {"persona_voice_loss"}
        assert by_family["genericness_checks_hard_phrases_only"] == \
            {"genericness"}

    def test_persona_diagnostic_contrast(self):
        # The voice guard outside the stack names a
        # new violation in every persona entry the
        # stack passes clean — resolved, recorded,
        # and never scored.
        for row in _RUN["results"]:
            diagnostic = row["persona_voice_diagnostic"]
            if row["family"] == "persona_voice_loss":
                assert row["stack_verdict"] == "clean", row["id"]
                assert diagnostic is not None, row["id"]
                assert diagnostic["verdict"] == "voice_broken"
                assert diagnostic["new_violations"], row["id"]
                assert diagnostic["scored"] is False
            else:
                assert diagnostic is None, row["id"]

    def test_invented_reference_caveat(self):
        row = _ROWS["chaos_citation_invented_reference"]
        assert row["matched"] is True
        assert row["stack_verdict"] == "flagged"
        assert row["flags"] == ["citation:uncited_claim"]
        note = ENTRY_NOTES["chaos_citation_invented_reference"]
        assert row["note"] == note
        assert "Almanac" in note and "not reference verification" in note


# ── Scoring rules and signal resolution ────────────────────────


class TestScoringRules:
    def test_mapping_and_match_legs(self):
        assert mapped_actual("rescued", "cut", "full") == "rescued"
        assert mapped_actual("refused", None, "full") == "refused"
        assert mapped_actual("flagged", "text", "full") == "refused"
        assert mapped_actual("clean", "same", "same") == "passed-clean"
        assert mapped_actual("clean", "changed", "same") == \
            "clean-modified"
        assert is_match("rescued", "rescued", "cut", "full")
        assert not is_match("rescued", "clean", "full", "full")
        assert is_match("refused", "flagged", "text", "full")
        assert not is_match("refused", "clean", "full", "full")
        assert is_match("passed-clean", "clean", "same", "same")
        assert not is_match("passed-clean", "clean", "changed", "same")

    def test_records_resolution_is_tf093s_fixture_pair(self):
        kwargs, persona = resolve_entry_signals(
            _ENTRY["chaos_citation_foreign_qid"])
        assert persona is None
        assert kwargs["retrieved_qids"] == ["Q123"]
        records = kwargs["records"]
        assert [r.record_id for r in records] == ["Q123", "chunk_7"]

    def test_persona_resolution_is_tf030s_expectation(self):
        kwargs, persona = resolve_entry_signals(
            _ENTRY["chaos_voice_handwaving"])
        assert persona is FIXTURE_EXPECTATION
        # The surface takes no persona signal: the
        # resolved expectation is not a kwarg.
        assert "persona" not in kwargs
        assert "records" not in kwargs

    def test_structured_resolution(self):
        kwargs, _ = resolve_entry_signals(
            _ENTRY["chaos_format_trailing_comma"])
        assert kwargs["structured"] is True
        assert kwargs["required_keys"] == ("ferry", "crossing_minutes")
        assert kwargs["expected_format"] == "object"


# ── Tamper legs ────────────────────────────────────────────────


class TestTamper:
    def test_moved_score_named(self):
        score = copy.deepcopy(FILED_SCORE)
        score["matched"] = 30
        problems = chaos_run_problems(filed_score=score)
        assert any("matched" in p for p in problems)

    def test_flipped_filed_outcome_named(self):
        rows = copy.deepcopy(list(FILED_RESULTS))
        rows[0]["matched"] = False
        problems = chaos_run_problems(filed_results=tuple(rows))
        assert problems != []

    def test_uncovered_mismatch_named(self):
        gaps = copy.deepcopy(list(NAMED_GAPS))
        gaps[0]["entries"] = gaps[0]["entries"][:-1]
        problems = chaos_run_problems(
            gaps=tuple({**g, "entries": tuple(g["entries"])}
                       for g in gaps))
        assert any("exactly the filed mismatches" in p
                   for p in problems)

    def test_unknown_gap_owner_named(self):
        gaps = copy.deepcopy(list(NAMED_GAPS))
        gaps[0]["owner"] = "TF-999"
        problems = chaos_run_problems(
            gaps=tuple({**g, "entries": tuple(g["entries"])}
                       for g in gaps))
        assert any("unknown owner" in p for p in problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
