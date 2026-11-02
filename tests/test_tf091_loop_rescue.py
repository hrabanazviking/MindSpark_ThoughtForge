"""
TF-091 — Repetition & loop rescue: degenerate
loops detected and cut by rule.

The estate lives in
``src/thoughtforge/refinement/loop_rescue.py``
(``LOOP_RULES`` / ``LOOP_CONSTANTS`` /
``LOOP_FIXTURES`` as data, ``rescue_loops``, the
checkers ``loop_rescue_problems`` /
``loop_rescue_live_problems``, format
``thoughtforge-loop-rescue/1``). Write-up:
``docs/architecture/LOOP_RESCUE.md``.

Measured start: no loop cut existed anywhere —
TF-087's filed 20x loop fixture came back
byte-identical from FragmentSalvage (its dedup
touches the fragment list, never the output
text) and TF-089 files the loop among the 5
unrescued verbatim. This slice builds the
caller-invoked gate (the TF-090 shape):
salvage.py is untouched and TF-089's metrics
stand; TF-093 owns the unified surface.

Pinned here:

  * the filed estate: both checkers clean, rule
    ids unique, thresholds as filed, proof grade
    stated, salvage's verbatim behaviour
    unchanged,
  * clean prose passes byte-identical, and so
    does repetition below the thresholds (a
    doubled sentence, varied same-stem
    sentences),
  * sentence loops and cycles are cut to one
    period occurrence — TF-087's 20x fixture to
    exactly one sentence, a mid-answer loop with
    its surrounding prose preserved, a cycle's
    partial tail dropped with it,
  * detection is normalised (case / terminal
    punctuation variants still loop) while the
    kept occurrence is verbatim,
  * word loops: a punctuation-free phrase loop,
    a mid-sentence phrase loop, a single-word
    stutter, and a truncated tail word dropped
    never completed,
  * every cut is recorded (kind, kept unit,
    occurrences, dropped text verbatim), a
    rescued text judges clean on a second pass,
    and empty output is refused by name.
"""

from __future__ import annotations

from thoughtforge.refinement.loop_rescue import (
    LOOP_CONSTANTS,
    LOOP_DOC,
    LOOP_FIXTURES,
    LOOP_FORMAT,
    LOOP_RULES,
    loop_as_data,
    loop_rescue_live_problems,
    loop_rescue_problems,
    rescue_loops,
)
from thoughtforge.refinement.salvage_map import run_fixture

_FIXTURE = {f["id"]: f for f in LOOP_FIXTURES}


def _rescue(fixture_id: str):
    return rescue_loops(_FIXTURE[fixture_id]["text"])


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert loop_rescue_problems() == []
        assert loop_rescue_live_problems() == []

    def test_format_doc_and_proof(self):
        data = loop_as_data()
        assert data["format"] == LOOP_FORMAT
        assert data["loop_doc"] == LOOP_DOC
        assert data["proof"]["live_model_loops"] == "not_proven"
        assert data["proof"]["salvage_py"] == "unchanged_by_design"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in LOOP_RULES]
        assert len(ids) == len(set(ids)) == 8

    def test_thresholds_as_filed(self):
        assert LOOP_CONSTANTS["sentence_min_repeats"] == 3
        assert LOOP_CONSTANTS["sentence_max_period"] == 3
        assert LOOP_CONSTANTS["word_min_repeats"] == 4
        assert LOOP_CONSTANTS["word_max_period"] == 8

    def test_fixture_counts(self):
        assert len(LOOP_FIXTURES) == 16
        verdicts = [f["verdict"] for f in LOOP_FIXTURES]
        assert verdicts.count("clean") == 3
        assert verdicts.count("rescued") == 12
        assert verdicts.count("refused") == 1

    def test_salvage_itself_is_unchanged(self):
        # The gate is caller-invoked; FragmentSalvage
        # still returns the loop fixture verbatim
        # (TF-087's filings and TF-089's metrics stand).
        result = run_fixture("loop_verbatim")
        assert result.text == " ".join(["The bell rings."] * 20)
        assert result.salvage_path == "best_draft"


# ── Clean: untouched, byte-identical ───────────────────────────


class TestClean:
    def test_clean_prose_passes_byte_identical(self):
        result = _rescue("clean_prose")
        assert result.verdict == "clean"
        assert result.text == _FIXTURE["clean_prose"]["text"]
        assert result.cuts == ()

    def test_doubled_sentence_is_below_threshold(self):
        result = _rescue("clean_double_sentence_is_emphasis_room")
        assert result.verdict == "clean"
        assert result.text == "The bell rings. The bell rings."

    def test_varied_sentences_are_not_a_cycle(self):
        result = _rescue("clean_varied_bell_sentences")
        assert result.verdict == "clean"

    def test_non_consecutive_repetition_is_not_cut(self):
        text = ("The bell rings. The ferry leaves at dawn. "
                "The keeper lights the lantern. The bell rings.")
        result = rescue_loops(text)
        assert result.verdict == "clean"
        assert result.text == text


# ── Sentence loops & cycles ────────────────────────────────────


class TestSentenceLoops:
    def test_tf087_loop_fixture_cut_to_one(self):
        result = _rescue("tf087_loop_fixture")
        assert result.verdict == "rescued"
        assert result.text == "The bell rings."
        cut = result.cuts[0]
        assert cut.kind == "sentence_loop"
        assert cut.occurrences == 20
        assert cut.unit == "The bell rings."
        assert cut.dropped_text == \
            " ".join(["The bell rings."] * 19)

    def test_run_at_exact_threshold_is_cut(self):
        result = _rescue("sentence_run_at_threshold")
        assert result.text == "The tide is low."
        assert result.cuts[0].occurrences == 3

    def test_loop_in_the_middle_keeps_surrounding_prose(self):
        result = _rescue("loop_in_the_middle")
        assert result.text == (
            "The ferry leaves at dawn. The bell rings. "
            "The keeper lights the lantern.")

    def test_sentence_cycle_cut_to_one_cycle(self):
        result = _rescue("sentence_cycle")
        assert result.cuts[0].kind == "sentence_cycle"
        assert result.text == \
            "The ferry leaves at dawn. The bell rings once."

    def test_cycle_partial_tail_dropped_with_the_loop(self):
        result = _rescue("sentence_cycle_with_partial_tail")
        assert result.cuts[0].partial_units == 1
        assert result.text == \
            "The ferry leaves at dawn. The bell rings once."
        assert result.cuts[0].dropped_text.endswith(
            "The ferry leaves at dawn.")

    def test_normalised_variants_still_loop_kept_verbatim(self):
        result = _rescue("case_and_punctuation_variants_still_a_loop")
        assert result.text == "The bell rings."
        # The dropped copies are recorded verbatim,
        # in their own casing and punctuation:
        assert result.cuts[0].dropped_text == \
            "the bell rings! THE BELL RINGS."

    def test_two_loops_cut_in_order(self):
        result = _rescue("two_loops_in_one_draft")
        assert [c.occurrences for c in result.cuts] == [3, 4]
        assert result.text == (
            "The bell rings. The ferry leaves at dawn. "
            "The tide is low.")


# ── Word loops ─────────────────────────────────────────────────


class TestWordLoops:
    def test_phrase_loop_without_punctuation(self):
        result = _rescue("word_loop_no_punctuation")
        assert result.cuts[0].kind == "word_loop"
        assert result.text == "The bell rings"

    def test_phrase_loop_mid_sentence(self):
        result = _rescue("word_loop_mid_sentence")
        assert result.text == (
            "The keeper listened while the bell rings "
            "echoed over the harbour.")

    def test_truncated_tail_word_dropped_never_completed(self):
        result = _rescue("word_loop_with_truncated_tail_word")
        assert result.text == "the bell rings"
        assert "bel" not in (result.text or "").split()
        assert result.cuts[0].partial_units == 2
        assert result.cuts[0].dropped_text.endswith("the bel")

    def test_single_word_stutter(self):
        result = _rescue("single_word_stutter")
        assert result.text == "The bell rang and echoed."
        assert result.cuts[0].occurrences == 5

    def test_rambling_loop_shaped_subset_only(self):
        # TF-087's rambling fixture is one phrase
        # repeated verbatim: the loop cut collapses
        # it to its first occurrence — a fragment,
        # filed as such; non-repetitive rambling is
        # not compressed by this gate.
        result = _rescue("rambling_loop_shaped_subset")
        assert result.cuts[0].occurrences == 24
        assert result.text == "The river is nice and"


# ── Honesty: records, idempotence, refusal ─────────────────────


class TestHonesty:
    def test_every_rescue_judges_clean_on_a_second_pass(self):
        for fixture in LOOP_FIXTURES:
            if fixture["verdict"] != "rescued":
                continue
            first = rescue_loops(fixture["text"])
            second = rescue_loops(first.text or "")
            assert second.verdict == "clean", fixture["id"]
            assert second.text == first.text

    def test_a_cut_only_deletes(self):
        # The rescued text's words are the source's
        # own words, in order (a subsequence) — the
        # gate never introduces a word.
        for fixture in LOOP_FIXTURES:
            if fixture["verdict"] != "rescued":
                continue
            source_words = fixture["text"].split()
            kept_words = (fixture["expected_text"] or "").split()
            it = iter(source_words)
            assert all(any(word == kept for kept in it)
                       for word in kept_words), fixture["id"]

    def test_empty_output_refused_by_name(self):
        result = _rescue("refuse_empty")
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == ("empty_output",)

    def test_none_like_empty_is_refused(self):
        assert rescue_loops("").verdict == "refused"
