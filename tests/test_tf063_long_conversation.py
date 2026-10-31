"""TF-063 — the long-conversation proof.

Pins: the rules/counts as data; the script's shape
(100 turns, 10 rounds x 10 facts, unique user turns,
every fact probed exactly 10 times spread first-to-
last); the facts landing through the gate (adversarial
pass, summaries within the contract cap, citation
markers by TF-048's rule); the scorer on pure
transcripts (perfect PASS; a fabricated marker named
fabricated; a wrong record's marker untrue; a stated
text that is not the cited record's summary untrue; a
missing citation failing; a contradiction zeroing its
turn even with a true citation; partial retention
scoring fractionally when the citation is true; an
empty transcript REVIEW at 0.0; user mismatch raising);
the runner end-to-end (100/100 retrieval, PASS at
1.0 in both windows, drift 0, store bytes unchanged,
screened_out empty, two runs byte-identical); the
drift control (from turn 91 the last window falls to
0.0, drift 1.0, REVIEW, failures counted as untrue
citations); the empty-store fallback (no retrieval,
no citation, REVIEW — never a fabricated answer); and
the live checker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thoughtforge.knowledge.long_conversation import (
    FALLBACK_ANSWER,
    ROUND_FRAMES,
    SAGA_FACTS,
    SAGA_RULES,
    SAGA_SCRIPT,
    SAGA_TURNS,
    build_saga_store,
    long_conversation_problems,
    marker_map,
    run_saga,
    score_saga,
)
from thoughtforge.knowledge.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path):
    return build_saga_store(tmp_path)


@pytest.fixture()
def markers(store):
    return marker_map(store)


def _perfect_transcript(store) -> list[tuple[str, str]]:
    return run_saga(store).transcript


class TestDataShape:
    def test_rules_in_order(self):
        assert [row["name"] for row in SAGA_RULES] == [
            "exactly_one_hundred_turns",
            "memory_is_the_store_not_the_history",
            "answers_come_from_retrieval",
            "citation_must_resolve_and_be_true",
            "contradiction_or_false_citation_zeroes_the_turn",
            "saga_is_read_only",
            "synthetic_and_deterministic",
        ]

    def test_counts(self):
        assert SAGA_TURNS == 100
        assert len(SAGA_FACTS) == 10
        assert len(ROUND_FRAMES) == 10
        assert len(SAGA_SCRIPT) == 100

    def test_script_unique_users_and_rounds(self):
        assert len({t.user for t in SAGA_SCRIPT}) == 100
        assert [t.turn_number for t in SAGA_SCRIPT] == list(
            range(1, 101))
        assert {t.round_number for t in SAGA_SCRIPT} == set(
            range(1, 11))

    def test_every_fact_probed_ten_times_first_and_last(self):
        for fact in SAGA_FACTS:
            numbers = [t.turn_number for t in SAGA_SCRIPT
                       if t.fact_id == fact.fact_id]
            assert len(numbers) == 10
            assert min(numbers) <= 10
            assert max(numbers) >= 91

    def test_facts_are_synthetic_world_only(self):
        blob = " ".join(
            f.summary + f.question for f in SAGA_FACTS).lower()
        assert "lantern hollow" in blob
        for marker in ("sig" + "rid", "ygg" + "drasil",
                       "knowledge_" + "reference"):
            assert marker not in blob


class TestSeeding:
    def test_facts_land_through_the_gate(self, store):
        records = store.load_profile_records()
        assert len(records) == 10
        assert {r.record_id for r in records} == {
            f.record_id for f in SAGA_FACTS}
        assert all(r.source == "user_statement" for r in records)
        assert all(len(r.summary) <= 180 for r in records)

    def test_facts_pass_adversarial_screen(self, store):
        from thoughtforge.knowledge.adversarial import screen_record

        for record in store.load_profile_records():
            assert screen_record(record).passed, record.record_id

    def test_marker_map_by_tf048_rule(self, store, markers):
        assert set(markers) == {
            f"user_statement:{f.record_id}" for f in SAGA_FACTS}
        for fact in SAGA_FACTS:
            row = markers[f"user_statement:{fact.record_id}"]
            assert row == {"record_id": fact.record_id,
                           "summary": fact.summary}


class TestScorer:
    def test_perfect_transcript_passes(self, store, markers):
        report = score_saga(_perfect_transcript(store), markers)
        assert report.total_turns == 100
        assert report.verdict == "PASS"
        assert report.mean_score == 1.0
        assert report.first_mean == 1.0 and report.last_mean == 1.0
        assert report.drift == 0.0
        assert report.citation_failures == 0
        assert report.contradiction_turns == 0
        assert report.fact_retention == {
            f.fact_id: 1.0 for f in SAGA_FACTS}

    def test_empty_transcript_is_review_never_vacuous_pass(
            self, markers):
        report = score_saga([], markers)
        assert report.total_turns == 0 and report.mean_score == 0.0
        assert report.verdict == "REVIEW"

    def test_fabricated_citation_is_named_and_zeroes(
            self, store, markers):
        transcript = _perfect_transcript(store)
        user, text = transcript[0]
        transcript[0] = (user, text.replace(
            "[user_statement:fct_tf063_01]",
            "[user_statement:fct_tf063_99]"))
        report = score_saga(transcript, markers)
        turn = report.per_turn[0]
        assert turn.citation_status == "fabricated"
        assert turn.score == 0.0 and turn.retention == 1.0
        assert report.fabricated_citations == 1
        assert report.citation_failures == 1
        assert report.verdict == "REVIEW" or report.mean_score < 1.0

    def test_wrong_records_marker_is_untrue(self, store, markers):
        transcript = _perfect_transcript(store)
        user, text = transcript[0]
        transcript[0] = (user, text.replace(
            "[user_statement:fct_tf063_01]",
            "[user_statement:fct_tf063_02]"))
        report = score_saga(transcript, markers)
        assert report.per_turn[0].citation_status == "untrue"
        assert report.per_turn[0].score == 0.0
        assert report.untrue_citations == 1

    def test_stated_text_not_the_cited_summary_is_untrue(
            self, store, markers):
        transcript = _perfect_transcript(store)
        user, _ = transcript[0]
        transcript[0] = (
            user,
            "The ferry crossing takes eleven minutes in calm "
            "water. [user_statement:fct_tf063_01]")
        report = score_saga(transcript, markers)
        assert report.per_turn[0].citation_status == "untrue"
        assert report.per_turn[0].score == 0.0

    def test_missing_citation_fails_the_turn(self, store, markers):
        transcript = _perfect_transcript(store)
        user, text = transcript[0]
        transcript[0] = (user, text.split(" [")[0])
        report = score_saga(transcript, markers)
        assert report.per_turn[0].citation_status == "missing"
        assert report.per_turn[0].score == 0.0
        assert report.missing_citations == 1

    def test_contradiction_zeroes_even_with_true_citation(
            self, store, markers):
        transcript = _perfect_transcript(store)
        user, text = transcript[0]
        transcript[0] = (user, text + " It takes twelve minutes.")
        report = score_saga(transcript, markers)
        turn = report.per_turn[0]
        # the appended sentence also breaks stated-text equality,
        # so the citation reads untrue too — either way the turn
        # is zero and the contradiction is named.
        assert turn.contradictions == ("ferry_minutes",)
        assert turn.score == 0.0
        assert report.contradiction_turns == 1

    def test_user_mismatch_and_overlength_raise(self, store, markers):
        with pytest.raises(ValueError):
            score_saga([("A different question?", "Any.")], markers)
        transcript = _perfect_transcript(store)
        with pytest.raises(ValueError):
            score_saga(transcript + [("Extra?", "Extra.")], markers)


class TestRunner:
    def test_full_saga_passes_perfectly(self, store):
        run = run_saga(store)
        assert run.report.verdict == "PASS"
        assert run.report.mean_score == 1.0
        assert run.report.first_mean == 1.0
        assert run.report.last_mean == 1.0
        assert run.report.drift == 0.0
        assert run.retrieval_hits == 100
        assert run.screened_out_total == 0

    def test_saga_is_read_only_over_the_store(self, store, tmp_path):
        before = {p.name: p.read_bytes()
                  for p in sorted(tmp_path.iterdir()) if p.is_file()}
        run = run_saga(store)
        after = {p.name: p.read_bytes()
                 for p in sorted(tmp_path.iterdir()) if p.is_file()}
        assert run.store_unchanged is True
        assert after == before

    def test_two_runs_are_byte_identical(self, store):
        first = run_saga(store)
        second = run_saga(store)
        assert first.transcript == second.transcript
        assert first.report.mean_score == second.report.mean_score

    def test_drift_from_91_is_caught_in_the_last_window(self, store):
        run = run_saga(store, drift_from_turn=91)
        report = run.report
        assert report.verdict == "REVIEW"
        assert report.first_mean == 1.0 and report.last_mean == 0.0
        assert report.drift == 1.0
        assert report.mean_score == 0.9
        assert report.untrue_citations == 10
        assert all(t.score == 1.0 for t in report.per_turn[:90])
        assert all(t.score == 0.0 for t in report.per_turn[90:])

    def test_empty_store_answers_fallback_without_citation(
            self, tmp_path):
        empty = MemoryStore(memory_dir=tmp_path / "empty")
        run = run_saga(empty)
        assert run.retrieval_hits == 0
        assert all(text == FALLBACK_ANSWER
                   for _, text in run.transcript)
        assert run.report.missing_citations == 100
        assert run.report.fabricated_citations == 0
        assert run.report.verdict == "REVIEW"
        assert run.report.mean_score == 0.0


class TestChecker:
    def test_long_conversation_problems_empty(self):
        assert long_conversation_problems() == []
