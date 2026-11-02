"""
TF-099 — Streaming salvage: what can be rescued
mid-stream vs post-hoc, specified honestly per
strategy.

The estate lives in
``src/thoughtforge/refinement/streaming_salvage.py``
(``STREAM_RULES`` / ``STRATEGY_PLACEMENTS`` /
``STREAM_FIXTURES`` as data, ``StreamingSalvage``
/ ``run_streaming_salvage`` / ``salvage_stream``,
the checkers ``streaming_salvage_problems`` /
``streaming_salvage_live_problems``, format
``thoughtforge-streaming-salvage/1``). Write-up:
``docs/architecture/STREAMING_SALVAGE.md``.

Measured start: no refinement surface consumed a
stream (every gate takes complete text; TF-026's
chunks are delivered raw and salvage runs only on
the completed candidate); a loop is provable only
at TF-091's third repeat, so a second repeat must
be held to ever be cut; TF-090's format and
truncation judgements already "repair" partial
prefixes, proving them stream_end only; the
per-sentence judgements of TF-091 (period-1) and
TF-092 are the whole mid-stream set.

Pinned here:

  * the filed estate: both checkers clean, the
    doc present, proof grades stated, the
    mid-stream set exactly the two sentence-local
    deletion strategies, every strategy placed
    with an owner and a reason,
  * every fixture lands exactly as filed —
    chunked mid-word streams, loop cutting with
    the held repeat, emphasis released, drift
    dropped, the tail recorded never delivered,
  * equivalence: the loop legs equal TF-091's
    post-hoc gate and the minority-drift leg
    equals TF-092's rescue on the same texts,
  * pull-based honesty: nothing released before
    a sentence completes; the held repeat is not
    released by the chunk that carries it,
  * the error legs: an unsupported expected
    language raises by name; feed after finish
    and double finish raise,
  * mid-stream is not final: every result
    carries post_hoc_still_required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.refinement.drift_rescue import rescue_drift
from thoughtforge.refinement.loop_rescue import rescue_loops
from thoughtforge.refinement.streaming_salvage import (
    MID_STREAM_STRATEGIES,
    PLACEMENTS,
    STRATEGY_PLACEMENTS,
    STREAM_FIXTURES,
    STREAM_RULES,
    STREAM_SALVAGE_DOC,
    STREAM_SALVAGE_FORMAT,
    StreamingSalvage,
    run_stream_fixture,
    run_streaming_salvage,
    salvage_stream,
    streaming_salvage_as_data,
    streaming_salvage_live_problems,
    streaming_salvage_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in STREAM_FIXTURES}


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert streaming_salvage_problems() == []
        assert streaming_salvage_live_problems() == []

    def test_format_doc_and_proof(self):
        data = streaming_salvage_as_data()
        assert data["format"] == STREAM_SALVAGE_FORMAT
        assert data["stream_salvage_doc"] == STREAM_SALVAGE_DOC
        assert (REPO_ROOT / STREAM_SALVAGE_DOC).exists()
        assert data["proof"]["grade"] == "fixture"
        assert data["proof"]["live_model_stream"] == \
            "not_proven"
        assert data["proof"]["post_hoc_stack"] == \
            "still_required"
        assert data["proof"]["core_think_stream"] == "unchanged"
        json.dumps(data)

    def test_rule_and_fixture_counts(self):
        assert len({r["id"] for r in STREAM_RULES}) == \
            len(STREAM_RULES) == 9
        assert len(STREAM_FIXTURES) == 12
        assert len(STRATEGY_PLACEMENTS) == 17

    def test_mid_stream_set_is_exactly_two(self):
        assert set(MID_STREAM_STRATEGIES) == {
            "sentence_loop_cut_period_1",
            "drift_drop_per_sentence"}
        for placement in STRATEGY_PLACEMENTS:
            assert placement["placement"] in PLACEMENTS
            assert placement["owner"]
            assert placement["reason"]

    def test_stream_end_holds_format_and_truncation(self):
        by_strategy = {p["strategy"]: p["placement"]
                       for p in STRATEGY_PLACEMENTS}
        assert by_strategy["format_enforcement_json"] == \
            "stream_end"
        assert by_strategy["truncation_repair"] == "stream_end"
        assert by_strategy[
            "citation_and_policy_enforcement"] == "post_hoc"
        assert by_strategy[
            "loop_cut_cycles_and_word_loops"] == "post_hoc"
        assert by_strategy["drift_majority_flagging"] == \
            "post_hoc"


# ── The fixtures ───────────────────────────────────────────────


class TestFixtures:
    def test_every_fixture_lands_as_filed(self):
        for fixture in STREAM_FIXTURES:
            result, released = run_stream_fixture(fixture["id"])
            assert result.text == fixture["expected_text"], \
                fixture["id"]
            assert [c.kind for c in result.cuts] == \
                fixture["expected_cuts"], fixture["id"]
            assert result.verdict == \
                fixture["expected_verdict"], fixture["id"]
            assert " ".join(released) == result.text
            assert result.post_hoc_still_required is True

    def test_loop_equivalence_with_tf091(self):
        for fixture_id in ("sentence_loop_five",
                           "loop_then_recovery"):
            whole = "".join(_FIXTURE[fixture_id]["chunks"])
            result, _ = run_stream_fixture(fixture_id)
            assert rescue_loops(whole).text == result.text

    def test_drift_equivalence_with_tf092(self):
        fixture = _FIXTURE["drift_sentence_dropped"]
        whole = "".join(fixture["chunks"])
        result, _ = run_stream_fixture(fixture["id"])
        post_hoc = rescue_drift(whole, "en")
        assert post_hoc.verdict == "rescued"
        assert post_hoc.text == result.text

    def test_all_drift_empty_says_why(self):
        result, _ = run_stream_fixture(
            "drift_all_dropped_says_why")
        assert result.text == ""
        assert result.verdict == "empty"
        assert result.drift_sentences_dropped == 2
        assert [c.detected_language for c in result.cuts] == \
            ["es", "es"]

    def test_tail_only_is_named_not_dressed(self):
        result, _ = run_stream_fixture(
            "tail_only_no_complete_sentence")
        assert result.no_complete_sentence is True
        assert result.incomplete_tail == \
            "The lantern keeper checks the"
        assert result.text == ""

    def test_incomplete_tail_recorded_verbatim(self):
        result, _ = run_stream_fixture(
            "incomplete_tail_recorded")
        assert result.incomplete_tail == \
            "The lantern keeper checks the"
        assert result.no_complete_sentence is False


# ── Pull-based feeding ─────────────────────────────────────────


class TestPullBased:
    def test_nothing_released_before_a_sentence_completes(self):
        rescuer = StreamingSalvage()
        assert rescuer.feed("The bell ri") == []
        assert rescuer.feed("ngs.") == []
        # The terminator alone does not complete it —
        # following whitespace (more text) or
        # finish() does.
        assert rescuer.feed(" Wrenna") == ["The bell rings."]
        result = rescuer.finish()
        assert result.incomplete_tail == "Wrenna"

    def test_held_repeat_not_released_by_its_own_chunk(self):
        rescuer = StreamingSalvage()
        assert rescuer.feed(
            "The bell rings. The bell rings.") == \
            ["The bell rings."]
        # "Wrenna smiles." is not judgeable until it
        # completes (trailing whitespace here; at the
        # buffer's end it would wait for finish()).
        assert rescuer.feed(" Wrenna smiles. ") == [
            "The bell rings.", "Wrenna smiles."]
        result = rescuer.finish()
        assert result.verdict == "clean"
        assert result.sentences_delivered == 3

    def test_salvage_stream_yields_in_release_order(self):
        chunks = ["The bell rings. The bell rings. ",
                  "The bell rings. Then the harbour ",
                  "goes quiet."]
        assert list(salvage_stream(chunks)) == [
            "The bell rings.", "Then the harbour goes quiet."]

    def test_salvage_stream_releases_held_repeat_at_end(self):
        assert list(salvage_stream(
            ["The bell rings. The bell rings."])) == [
            "The bell rings.", "The bell rings."]

    def test_run_is_pull_based_over_a_generator(self):
        pulled = []

        def source():
            for chunk in ("The ferry leaves at dawn. ",
                          "The lantern keeper checks the ",
                          "river level each morning."):
                pulled.append(chunk)
                yield chunk

        result, released = run_streaming_salvage(source())
        assert len(pulled) == 3
        assert result.verdict == "clean"
        assert len(released) == 2


# ── The error legs ─────────────────────────────────────────────


class TestErrors:
    def test_unsupported_expected_language_raises_by_name(self):
        with pytest.raises(ValueError) as excinfo:
            StreamingSalvage(expected_language="fr")
        assert "fr" in str(excinfo.value)

    def test_feed_after_finish_and_double_finish_raise(self):
        rescuer = StreamingSalvage()
        rescuer.feed("The bell rings. ")
        rescuer.finish()
        with pytest.raises(ValueError):
            rescuer.feed("More.")
        with pytest.raises(ValueError):
            rescuer.finish()
