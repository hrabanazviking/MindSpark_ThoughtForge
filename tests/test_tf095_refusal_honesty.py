"""
TF-095 — Refusal honesty: when enforcement blocks
an answer, the user learns that and why, with the
salvage path offered.

The estate lives in
``src/thoughtforge/refinement/refusal_honesty.py``
(``REFUSAL_RULES`` / ``REASONS`` /
``REFUSAL_FIXTURES`` as data, ``refusal_notice`` /
``refusal_message``, the checkers
``refusal_honesty_problems`` /
``refusal_honesty_live_problems``, format
``thoughtforge-refusal-honesty/1``).
Write-up: ``docs/architecture/REFUSAL_HONESTY.md``.

Measured start: a refused verdict existed only as
data — ``text=None`` plus a tuple of machine
codes — and nothing in the tree rendered it for a
user; no message anywhere said an answer had been
blocked.

Pinned here:

  * the filed estate: both checkers clean, the
    catalogue covering every refusal code the
    TF-090–TF-094 fixture sets can produce, the
    doc present, proof grade stated,
  * that: the marker and the blocked statement
    open every message; a strict block is told
    as a strict block,
  * why: every problem verbatim, with its
    catalogued plain reason — a ': ' detail rides
    with its problem while the reason matches by
    head,
  * what next: each problem's salvage path is
    offered, deduplicated, never run,
  * the honesty edges: an uncatalogued problem is
    named and marked, never invented; a refusal
    with no problem named states the gap; blocked
    draft text never appears in its message,
  * only a block gets a message: a clean /
    rescued / flagged result raises,
  * TF-093's surface, TF-094's layer and
    FragmentSalvage are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.refinement.policy_config import enforce_configured
from thoughtforge.refinement.policy_surface import (
    PolicyResult,
    enforce_policy,
)
from thoughtforge.refinement.refusal_honesty import (
    NO_PROBLEM_REASON,
    REASONS,
    REFUSAL_DOC,
    REFUSAL_FIXTURES,
    REFUSAL_FORMAT,
    REFUSAL_MARKER,
    REFUSAL_RULES,
    UNCATALOGUED_PATH,
    UNCATALOGUED_REASON,
    refusal_honesty_as_data,
    refusal_honesty_live_problems,
    refusal_honesty_problems,
    refusal_message,
    refusal_notice,
)
from thoughtforge.refinement.salvage_map import run_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in REFUSAL_FIXTURES}


def _records():
    from thoughtforge.knowledge.models import ActivatedRecord

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
    if fixture["source"] == "synthetic":
        return PolicyResult("refused", problems=tuple(fixture["problems"]))
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _records()
    if fixture["source"] == "configured":
        return enforce_configured(
            fixture["text"], strictness=fixture["strictness"], **signals)
    return enforce_policy(fixture["text"], **signals)


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert refusal_honesty_problems() == []
        assert refusal_honesty_live_problems() == []

    def test_format_doc_and_proof(self):
        data = refusal_honesty_as_data()
        assert data["format"] == REFUSAL_FORMAT
        assert data["refusal_doc"] == REFUSAL_DOC
        assert (REPO_ROOT / REFUSAL_DOC).exists()
        assert data["proof"]["live_model_behaviour"] == "not_proven"
        assert data["proof"]["policy_surface"] == "unchanged"
        assert data["proof"]["salvage_path"] == \
            "offered_never_run_by_this_slice"
        json.dumps(data)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in REFUSAL_RULES]
        assert len(ids) == len(set(ids)) == 9

    def test_fixture_counts(self):
        assert len(REFUSAL_FIXTURES) == 10
        sources = [f["source"] for f in REFUSAL_FIXTURES]
        assert sources.count("policy") == 5
        assert sources.count("configured") == 3
        assert sources.count("synthetic") == 2

    def test_every_reason_carries_an_offered_path(self):
        for entry in REASONS.values():
            assert entry["reason"]
            assert entry["path"]


# ── That, why, and what next ───────────────────────────────────


class TestTheTelling:
    @pytest.mark.parametrize("fixture_id", list(_FIXTURE))
    def test_message_opens_with_the_block(self, fixture_id):
        notice = refusal_notice(_run(fixture_id))
        assert notice.message.startswith(REFUSAL_MARKER)
        assert "blocked" in notice.message
        assert "has not been shown" in notice.message

    @pytest.mark.parametrize("fixture_id", [
        "empty_refused",
        "format_refused_no_json_span",
        "missing_required_key_refused",
        "truncation_refused_no_complete_sentence",
        "unsupported_language_refused",
    ])
    def test_every_problem_verbatim_with_reason_and_path(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        notice = refusal_notice(_run(fixture_id))
        assert notice.problems == tuple(fixture["problems"])
        for problem in fixture["problems"]:
            key = problem.split(": ", 1)[0]
            assert problem in notice.message
            assert REASONS[key]["reason"] in notice.message
            assert REASONS[key]["path"] in notice.message

    def test_detail_rides_with_its_problem(self):
        notice = refusal_notice(_run("missing_required_key_refused"))
        assert "missing_required_key: ferry" in notice.message
        (explanation,) = notice.explanations
        assert explanation["key"] == "missing_required_key"
        assert explanation["catalogued"] is True

    @pytest.mark.parametrize("fixture_id", [
        "strict_blocked_citation",
        "strict_blocked_drift",
        "strict_blocked_two_flags",
    ])
    def test_a_strict_block_is_told_as_a_strict_block(self, fixture_id):
        notice = refusal_notice(_run(fixture_id))
        assert notice.strictness == "strict"
        assert notice.disposition == "blocked_flagged"
        assert notice.base_verdict == "flagged"
        assert "flagged" in notice.message
        assert "strict" in notice.message

    def test_two_flag_block_names_both_and_offers_both_paths(self):
        notice = refusal_notice(_run("strict_blocked_two_flags"))
        assert notice.problems == (
            "citation:uncited_claim", "length:too_short")
        for key in ("citation:uncited_claim", "length:too_short"):
            assert key in notice.message
            assert REASONS[key]["reason"] in notice.message
            assert REASONS[key]["path"] in notice.message

    def test_offered_paths_are_deduplicated(self):
        # Both drift codes share one offered path; a
        # (synthetic) block carrying both must offer
        # it once, in problem order.
        result = PolicyResult("refused", problems=(
            "drift:whole_answer_off_language", "drift:drift_is_majority"))
        notice = refusal_notice(result)
        assert notice.message.count(
            REASONS["drift:whole_answer_off_language"]["path"]) == 1


# ── The honesty edges ──────────────────────────────────────────


class TestHonestyEdges:
    def test_uncatalogued_problem_named_and_marked(self):
        notice = refusal_notice(_run("unknown_problem_uncatalogued"))
        assert "some_future_problem" in notice.message
        assert UNCATALOGUED_REASON in notice.message
        assert UNCATALOGUED_PATH in notice.message
        (explanation,) = notice.explanations
        assert explanation["catalogued"] is False

    def test_no_problem_named_states_the_gap(self):
        notice = refusal_notice(_run("refused_no_problem_named"))
        assert notice.problems == ()
        assert NO_PROBLEM_REASON in notice.message

    @pytest.mark.parametrize("fixture_id", [
        "format_refused_no_json_span",
        "truncation_refused_no_complete_sentence",
        "unsupported_language_refused",
        "strict_blocked_citation",
        "strict_blocked_two_flags",
    ])
    def test_blocked_text_never_in_its_message(self, fixture_id):
        fixture = _FIXTURE[fixture_id]
        message = refusal_message(_run(fixture_id))
        assert fixture["text"] not in message

    def test_refusal_message_matches_the_notice(self):
        result = _run("empty_refused")
        assert refusal_message(result) == refusal_notice(result).message


# ── Only a block gets a message ────────────────────────────────


class TestOnlyABlock:
    def test_clean_result_raises(self):
        result = enforce_policy(
            "The ferry leaves at dawn and the tide is low. "
            "The keeper lights the lantern when the ferry is late.")
        assert result.verdict == "clean"
        with pytest.raises(ValueError, match="no enforcement block"):
            refusal_notice(result)

    def test_flagged_result_raises(self):
        # A flagged answer under standard LEAVES — it
        # is not a block and is never dressed as one.
        result = enforce_policy("Too short.")
        assert result.verdict == "flagged"
        with pytest.raises(ValueError, match="no enforcement block"):
            refusal_message(result)

    def test_rescued_result_raises(self):
        result = enforce_policy('{"ferry": "dawn",}', structured=True)
        assert result.verdict == "rescued"
        with pytest.raises(ValueError, match="no enforcement block"):
            refusal_notice(result)


# ── What this slice does not touch ─────────────────────────────


class TestUntouched:
    def test_salvage_itself_is_unchanged(self):
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"

    def test_the_configured_layer_is_unchanged(self):
        # The strict block this slice tells is still
        # TF-094's own record — refused, no text, the
        # flags carried as problems.
        result = enforce_configured(
            "Odin is chief.", strictness="strict", records=_records())
        assert result.verdict == "refused"
        assert result.text is None
        assert result.problems == (
            "citation:uncited_claim", "length:too_short")
