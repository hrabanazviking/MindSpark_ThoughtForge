"""
TF-090 — Format enforcement: structured outputs
(JSON-shaped asks) repaired or refused honestly,
per policy.

The estate lives in
``src/thoughtforge/refinement/format_enforcement.py``
(``FORMAT_RULES`` / ``FORMAT_FIXTURES`` /
``TRUNCATION_FIXTURES`` as data, ``enforce_format``
/ ``repair_truncation``, the checkers
``format_enforcement_problems`` /
``format_enforcement_live_problems``, format
``thoughtforge-format-enforcement/1``). Write-up:
``docs/architecture/FORMAT_ENFORCEMENT.md``.

Measured start: nothing in refinement could judge
a structured answer — TF-087's broken-JSON draft
came back byte-identical from FragmentSalvage, and
salvage has no signal that an ask was JSON-shaped.
This slice builds the caller-invoked gate (the
TF-049 shape): salvage.py is untouched and TF-089's
metrics stand; TF-093 owns the unified surface.

Pinned here:

  * the filed estate: both checkers clean, rule
    ids unique, proof grade stated, salvage's
    verbatim behaviour unchanged,
  * valid output passes byte-identical,
  * the honest repairs: fence / prose extraction,
    trailing commas, missing closers, Python
    literals — content preserved exactly,
  * truncation closure: the incomplete tail is
    dropped (recorded verbatim), containers are
    closed, and NO value is ever completed —
    TF-087's broken-JSON fixture now repairs to
    exactly its one complete pair,
  * honest refusals: prose when JSON was asked,
    a missing required key (never invented —
    including after a repair), the wrong top-level
    shape, a bare-word value, multiple spans,
  * prose truncation: intact endings pass, a
    mid-sentence ending is cut to the last complete
    sentence, and TF-087's truncated draft — which
    has no complete sentence — is refused rather
    than given an invented ending.
"""

from __future__ import annotations

from thoughtforge.refinement.format_enforcement import (
    FORMAT_DOC,
    FORMAT_FIXTURES,
    FORMAT_FORMAT,
    FORMAT_RULES,
    TRUNCATION_FIXTURES,
    enforce_format,
    format_as_data,
    format_enforcement_live_problems,
    format_enforcement_problems,
    repair_truncation,
)
from thoughtforge.refinement.salvage_map import run_fixture

_FORMAT = {f["id"]: f for f in FORMAT_FIXTURES}
_TRUNC = {f["id"]: f for f in TRUNCATION_FIXTURES}


def _enforce(fixture_id: str):
    fixture = _FORMAT[fixture_id]
    return enforce_format(
        fixture["text"],
        required_keys=tuple(fixture["required_keys"]),
        expected=fixture["expected"])


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert format_enforcement_problems() == []
        assert format_enforcement_live_problems() == []

    def test_format_doc_and_proof(self):
        data = format_as_data()
        assert data["format"] == FORMAT_FORMAT
        assert data["format_doc"] == FORMAT_DOC
        assert data["proof"]["live_model_structured_outputs"] == \
            "not_proven"
        assert data["proof"]["salvage_py"] == \
            "unchanged_by_design"

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in FORMAT_RULES]
        assert len(ids) == len(set(ids)) == 7

    def test_fixture_counts(self):
        assert len(FORMAT_FIXTURES) == 20
        assert len(TRUNCATION_FIXTURES) == 5

    def test_salvage_itself_is_unchanged(self):
        # The gate is caller-invoked; FragmentSalvage
        # still returns the broken fixtures verbatim
        # (TF-087's filings and TF-089's metrics stand).
        assert run_fixture("broken_json_verbatim").text == \
            '{"ferry": "dawn", "tide": '
        assert run_fixture("truncated_mid_sentence").text.endswith(
            "across the")


# ── Valid + extraction + surface repairs ───────────────────────


class TestValidAndExtraction:
    def test_valid_object_passes_byte_identical(self):
        result = _enforce("valid_object")
        assert result.verdict == "valid"
        assert result.text == _FORMAT["valid_object"]["text"]
        assert result.repairs == ()

    def test_fenced_json_extracted(self):
        result = _enforce("fenced_json")
        assert result.verdict == "repaired"
        assert result.repairs == ("extracted_from_fence",)
        assert result.parsed == {"ferry": "dawn"}

    def test_prose_wrapped_json_extracted(self):
        result = _enforce("prose_wrapped_json")
        assert result.repairs == ("extracted_from_prose",)
        assert result.parsed == {"ferry": "dawn"}

    def test_trailing_prose_trimmed_to_the_span(self):
        result = enforce_format(
            '[1, 2, {"k": "v"}] trailing words', expected="array")
        assert result.verdict == "repaired"
        assert result.repairs == ("extracted_from_prose",)
        assert result.parsed == [1, 2, {"k": "v"}]

    def test_trailing_comma_dropped(self):
        result = _enforce("trailing_comma")
        assert result.repairs == ("dropped_trailing_comma",)
        assert result.parsed == {"ferry": "dawn", "tide": "low"}

    def test_missing_closers_closed(self):
        result = _enforce("missing_closers")
        assert result.repairs == ("closed_open_containers",)
        assert result.parsed == {"ferry": "dawn", "tide": "low"}

    def test_python_literal_converted_content_exact(self):
        result = _enforce("python_literal")
        assert result.repairs == ("converted_python_literal",)
        assert result.parsed == {"ferry": "dawn", "bell": True,
                                 "tide": None}


# ── Truncation closure: drop the incomplete, invent nothing ────


class TestTruncationClosure:
    def test_tf087_broken_json_repairs_to_its_complete_pair(self):
        result = _enforce("tf087_broken_json_fixture")
        assert result.verdict == "repaired"
        assert result.parsed == {"ferry": "dawn"}
        # "tide" is dropped, never completed:
        assert "tide" not in result.parsed
        assert result.dropped_tail == ', "tide":'

    def test_truncated_mid_string_value_drops_the_pair(self):
        result = _enforce("truncated_mid_string_value")
        assert result.parsed == {"ferry": "dawn"}
        assert result.dropped_tail == ', "tide": "ris'

    def test_truncated_mid_key_drops_the_key(self):
        result = _enforce("truncated_mid_key")
        assert result.parsed == {"ferry": "dawn"}
        assert result.dropped_tail == ', "ti'

    def test_truncated_array_drops_the_cut_element(self):
        result = _enforce("truncated_array_mid_string")
        assert result.parsed == ["dawn"]
        assert "closed_open_containers" in result.repairs

    def test_unclosed_number_array_only_closes(self):
        result = _enforce("unclosed_array_of_numbers")
        assert result.parsed == [1, 2, 3]
        assert result.dropped_tail == ""

    def test_nested_truncation_keeps_the_complete_inner_object(self):
        result = _enforce("nested_truncation")
        assert result.parsed == {
            "route": {"from": "haven", "to": "isle"}}
        assert "bell" not in result.parsed


# ── Honest refusals ────────────────────────────────────────────


class TestRefusals:
    def test_prose_when_json_asked_is_refused(self):
        result = _enforce("refuse_prose_when_json_asked")
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == ("no_json_span",)

    def test_missing_required_key_is_never_invented(self):
        result = _enforce("refuse_missing_required_key")
        assert result.verdict == "refused"
        assert result.problems == ("missing_required_key: ferry",)
        assert result.parsed == {"tide": "low"}

    def test_missing_key_after_repair_still_refuses(self):
        # The truncation repair honestly yields {}
        # — and the schema gate then refuses, rather
        # than the gate inventing "ferry".
        result = _enforce("refuse_missing_key_after_repair")
        assert result.verdict == "refused"
        assert result.parsed == {}
        assert result.problems == ("missing_required_key: ferry",)

    def test_wrong_top_level_shape_refused(self):
        result = _enforce("refuse_wrong_top_level")
        assert result.verdict == "refused"
        assert result.problems == (
            "wrong_top_level_shape: expected object, got list",)

    def test_bare_word_value_is_unparseable_not_repaired(self):
        result = _enforce("refuse_bare_word_value")
        assert result.verdict == "refused"
        assert result.problems == ("unparseable_json",)

    def test_multiple_spans_are_ambiguous_and_refused(self):
        result = _enforce("refuse_multiple_spans")
        assert result.verdict == "refused"
        assert result.problems == ("multiple_json_spans",)

    def test_empty_output_refused(self):
        result = _enforce("refuse_empty")
        assert result.verdict == "refused"
        assert result.problems == ("empty_output",)

    def test_no_refusal_ever_carries_usable_text(self):
        for fixture in FORMAT_FIXTURES:
            if fixture["verdict"] == "refused":
                result = enforce_format(
                    fixture["text"],
                    required_keys=tuple(fixture["required_keys"]),
                    expected=fixture["expected"])
                assert result.text is None, fixture["id"]


# ── Prose truncation ───────────────────────────────────────────


class TestProseTruncation:
    def test_intact_sentence_passes(self):
        result = repair_truncation(_TRUNC["intact_sentence"]["text"])
        assert result.verdict == "intact"
        assert result.dropped_tail == ""

    def test_intact_question_inside_quotes(self):
        fixture = _TRUNC["intact_question_with_quote"]
        result = repair_truncation(fixture["text"])
        assert result.verdict == "intact"
        assert result.text == fixture["text"]

    def test_cut_to_last_complete_sentence(self):
        fixture = _TRUNC["cut_to_last_complete_sentence"]
        result = repair_truncation(fixture["text"])
        assert result.verdict == "repaired"
        assert result.text == "The ferry leaves at dawn."
        assert result.dropped_tail == \
            "The tide bell rings across the"

    def test_tf087_truncated_draft_refused_not_completed(self):
        # Its one sentence never completes, so there
        # is nothing honest to cut to — an ending
        # would have to be invented.
        fixture = _TRUNC["tf087_truncated_fixture_refused"]
        result = repair_truncation(fixture["text"])
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == ("no_complete_sentence",)

    def test_empty_prose_refused(self):
        result = repair_truncation("")
        assert result.verdict == "refused"
        assert result.problems == ("empty_output",)
