"""
TF-092 — Language drift rescue: answers
wandering off-language pulled back or flagged.

The estate lives in
``src/thoughtforge/refinement/drift_rescue.py``
(``DRIFT_RULES`` / ``DRIFT_FIXTURES`` as data,
``rescue_drift``, the checkers
``drift_rescue_problems`` /
``drift_rescue_live_problems``, format
``thoughtforge-drift-rescue/1``). Write-up:
``docs/architecture/DRIFT_RESCUE.md``.

Measured start: no drift check existed anywhere
in refinement — TF-087 catalogued salvage.py as
having no language detection at all, and a
drifted draft returns byte-identical from
FragmentSalvage. This slice builds the
caller-invoked gate (the TF-090/TF-091 shape)
on TF-060's real ``detect_language`` hints:
salvage.py is untouched and TF-089's metrics
stand; TF-093 owns the unified surface.

Pinned here:

  * the filed estate: both checkers clean, rule
    ids unique, supported languages exactly
    TF-060's known languages, proof grade
    stated, salvage's verbatim behaviour
    unchanged,
  * clean answers pass byte-identical in each
    supported language, and a neutral
    (unknown-hint) sentence is kept uncounted,
  * drift is pulled back by deletion only —
    trailing, leading, mid-answer, single
    sentence, both bilingual directions and a
    German leg — kept sentences verbatim, every
    segment recorded, nothing translated,
  * whole-answer and majority drift are
    flagged, never cut: the text returns
    byte-identical with the flag named,
  * empty output and an unsupported or unknown
    expected language are refused by name.
"""

from __future__ import annotations

from thoughtforge.knowledge.multilingual import KNOWN_LANGUAGES
from thoughtforge.refinement.drift_rescue import (
    DRIFT_DOC,
    DRIFT_FIXTURES,
    DRIFT_FORMAT,
    DRIFT_RULES,
    drift_as_data,
    drift_rescue_live_problems,
    drift_rescue_problems,
    rescue_drift,
)
from thoughtforge.refinement.salvage_map import BROKEN_FIXTURES, run_fixture

_FIXTURE = {f["id"]: f for f in DRIFT_FIXTURES}


def _rescue(fixture_id: str):
    fixture = _FIXTURE[fixture_id]
    return rescue_drift(fixture["text"], fixture["expected"])


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert drift_rescue_problems() == []
        assert drift_rescue_live_problems() == []

    def test_format_doc_and_proof(self):
        data = drift_as_data()
        assert data["format"] == DRIFT_FORMAT
        assert data["drift_doc"] == DRIFT_DOC
        assert data["proof"]["live_model_drift"] == "not_proven"
        assert data["proof"]["salvage_py"] == "unchanged_by_design"
        assert data["proof"]["detector"] == \
            "tf060_detect_language_hint"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in DRIFT_RULES]
        assert len(ids) == len(set(ids)) == 8

    def test_supported_languages_are_tf060s(self):
        data = drift_as_data()
        assert data["supported_expected_languages"] == \
            list(KNOWN_LANGUAGES) == ["en", "es", "de"]

    def test_fixture_counts(self):
        assert len(DRIFT_FIXTURES) == 16
        verdicts = [f["verdict"] for f in DRIFT_FIXTURES]
        assert verdicts.count("clean") == 4
        assert verdicts.count("rescued") == 6
        assert verdicts.count("flagged") == 3
        assert verdicts.count("refused") == 3

    def test_salvage_itself_is_unchanged(self):
        # The gate is caller-invoked; FragmentSalvage's
        # filed behaviour stands (TF-087/TF-089 filings
        # unmoved — no BROKEN_FIXTURES leg is a drift leg,
        # and the loop fixture still returns verbatim).
        assert all("drift" not in f["id"] for f in BROKEN_FIXTURES)
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"


# ── Clean: untouched, byte-identical ───────────────────────────


class TestClean:
    def test_clean_english_passes_byte_identical(self):
        result = _rescue("clean_english")
        assert result.verdict == "clean"
        assert result.text == _FIXTURE["clean_english"]["text"]
        assert result.segments == ()

    def test_clean_spanish_passes_byte_identical(self):
        result = _rescue("clean_spanish")
        assert result.verdict == "clean"
        assert result.text == _FIXTURE["clean_spanish"]["text"]

    def test_clean_german_passes_byte_identical(self):
        result = _rescue("clean_german")
        assert result.verdict == "clean"

    def test_neutral_sentence_is_kept_and_uncounted(self):
        result = _rescue("clean_neutral_sentence_kept")
        assert result.verdict == "clean"
        assert "Lantern Hollow." in (result.text or "")


# ── Rescued: pulled back by deletion only ──────────────────────


class TestRescued:
    def test_trailing_spanish_drift_cut(self):
        result = _rescue("trailing_spanish_drift")
        assert result.verdict == "rescued"
        assert result.text == (
            "The ferry leaves at dawn and the tide is low. "
            "The keeper lights the lantern when the ferry is late.")
        segment = result.segments[0]
        assert segment.language == "es"
        assert segment.cut is True
        assert segment.first_sentence == 2
        assert segment.sentences[0].startswith("El faro")

    def test_trailing_english_drift_cut_from_spanish(self):
        result = _rescue("trailing_english_drift_spanish_answer")
        assert result.verdict == "rescued"
        assert result.segments[0].language == "en"
        assert "ferry leaves" not in (result.text or "")

    def test_mid_answer_block_cut_surroundings_kept(self):
        result = _rescue("mid_answer_spanish_block")
        assert result.verdict == "rescued"
        assert result.segments[0].first_sentence == 1
        assert (result.text or "").endswith(
            "the water is calm.")
        assert "El faro" not in (result.text or "")

    def test_trailing_german_drift_cut(self):
        result = _rescue("trailing_german_drift")
        assert result.segments[0].language == "de"
        assert "Leuchtturm" not in (result.text or "")

    def test_single_sentence_drift_cut(self):
        result = _rescue("single_sentence_drift")
        assert result.verdict == "rescued"
        assert len(result.segments[0].sentences) == 1

    def test_leading_drift_cut_when_answer_recovers(self):
        result = _rescue("leading_drift_then_recovers")
        assert result.verdict == "rescued"
        assert result.segments[0].first_sentence == 0
        assert (result.text or "").startswith("The ferry leaves")

    def test_a_rescue_only_deletes_never_translates(self):
        # The rescued text's words are the source's own
        # words, in order (a subsequence) — the gate
        # introduces no word of its own in any language.
        for fixture in DRIFT_FIXTURES:
            if fixture["verdict"] != "rescued":
                continue
            source_words = fixture["text"].split()
            kept_words = (fixture["expected_text"] or "").split()
            it = iter(source_words)
            assert all(any(word == kept for kept in it)
                       for word in kept_words), fixture["id"]

    def test_every_rescue_judges_clean_on_a_second_pass(self):
        for fixture in DRIFT_FIXTURES:
            if fixture["verdict"] != "rescued":
                continue
            first = rescue_drift(fixture["text"], fixture["expected"])
            second = rescue_drift(first.text or "", fixture["expected"])
            assert second.verdict == "clean", fixture["id"]
            assert second.text == first.text


# ── Flagged: recorded, never cut ───────────────────────────────


class TestFlagged:
    def test_whole_answer_spanish_flagged_verbatim(self):
        fixture = _FIXTURE["whole_answer_spanish_flagged"]
        result = _rescue("whole_answer_spanish_flagged")
        assert result.verdict == "flagged"
        assert result.text == fixture["text"]
        assert result.flags == ("whole_answer_off_language",)
        assert result.segments[0].cut is False

    def test_whole_answer_english_flagged_for_spanish_ask(self):
        result = _rescue("whole_answer_english_flagged")
        assert result.verdict == "flagged"
        assert result.segments[0].language == "en"

    def test_majority_drift_flagged_not_cut_to_a_fragment(self):
        fixture = _FIXTURE["drift_majority_flagged"]
        result = _rescue("drift_majority_flagged")
        assert result.verdict == "flagged"
        assert result.flags == ("drift_is_majority",)
        # All three drifted sentences are still present:
        # nothing was silently removed.
        assert result.text == fixture["text"]
        assert len(result.segments[0].sentences) == 3


# ── Refused: named, nothing invented ───────────────────────────


class TestRefused:
    def test_empty_output_refused_by_name(self):
        result = _rescue("refuse_empty")
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == ("empty_output",)

    def test_unsupported_expected_language_refused(self):
        result = _rescue("refuse_unsupported_expected_language")
        assert result.verdict == "refused"
        assert result.problems == ("unsupported_expected_language",)

    def test_unknown_expected_language_refused(self):
        result = _rescue("refuse_unknown_expected_language")
        assert result.verdict == "refused"
        assert result.problems == ("unsupported_expected_language",)

    def test_expected_language_is_normalised(self):
        result = rescue_drift(
            "The ferry leaves at dawn and the tide is low.", " EN ")
        assert result.verdict == "clean"
        assert result.expected_language == "en"
