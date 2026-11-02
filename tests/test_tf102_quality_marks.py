"""
TF-102 — User-visible quality marks: where the
product surfaces answers, a rescued/cited answer
can show its marks (data ready for consumers;
UI is the consumer's).

The estate lives in
``src/thoughtforge/refinement/quality_marks.py``
(``MARKS_RULES`` / ``MARK_KINDS`` /
``RESCUE_LABELS`` / ``MARKS_FIXTURES`` as data,
``quality_marks`` / ``marks_line``, the checkers
``quality_marks_problems`` /
``quality_marks_live_problems``, format
``thoughtforge-quality-marks/1``).
Write-up: ``docs/architecture/QUALITY_MARKS.md``.

Measured start: an answer that left the forge
carried its quality truth only as internal
data; the gate's in-text notes were the only
visible trace (and a genericness or drift flag
left none), citations rode the prose unsummarised,
and TF-097's record rode the trace, not a mark.

Pinned here:

  * the filed estate: both checkers clean, the
    doc present, proof grade stated,
  * cited needs two truths: the check passed
    AND a marker actually cited — the no-claim
    pass earns no cited mark, a flagged citation
    earns the flagged mark instead,
  * rescued names its mechanisms, and salvage's
    mark names the ADOPTED pass (TF-097's
    record), never the attempted path; a
    rejected attempt and a record-less result
    earn no salvage mark,
  * flagged / warning marks carry the flags
    verbatim with TF-095's catalogued reasons,
  * only a leaving answer gets marks: refused,
    strict-blocked, and empty records raise,
  * a plain answer's mark set is empty and its
    line is empty — no invented badge,
  * marks never alter a result's text, and
    TF-093 / TF-094 / FragmentSalvage are
    unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thoughtforge.refinement.policy_config import enforce_configured
from thoughtforge.refinement.policy_surface import enforce_policy
from thoughtforge.refinement.quality_marks import (
    MARK_KINDS,
    MARKS_DOC,
    MARKS_FIXTURES,
    MARKS_FORMAT,
    MARKS_RULES,
    RESCUE_LABELS,
    QualityMarks,
    marks_line,
    quality_marks,
    quality_marks_as_data,
    quality_marks_live_problems,
    quality_marks_problems,
)
from thoughtforge.refinement.refusal_honesty import REASONS
from thoughtforge.refinement.rescue_provenance import run_rescue_fixture
from thoughtforge.refinement.salvage import SalvageResult
from thoughtforge.refinement.salvage_map import run_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

_FIXTURE = {f["id"]: f for f in MARKS_FIXTURES}


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


_EN1 = "The ferry leaves at dawn and the tide is low."
_EN2 = "The keeper lights the lantern when the ferry is late."
_BELL = "The bell rings across the harbour at dawn."
_ODIN_UNCITED = ("Odin is the chief god of Norse mythology, "
                 "a god of wisdom and war.")
_ODIN_CITED = ("Odin [Q123] is the chief god of Norse mythology "
               "and wisdom.")


def _policy(fixture_id: str):
    fixture = _FIXTURE[fixture_id]
    signals = dict(fixture["signals"])
    if signals.get("records") == "lantern":
        signals["records"] = _records()
    if fixture["source"] == "configured":
        return enforce_configured(
            fixture["text"], strictness=fixture["strictness"],
            **signals)
    return enforce_policy(fixture["text"], **signals)


# ── The filed estate ───────────────────────────────────────────


class TestEstate:
    def test_checkers_clean(self):
        assert quality_marks_problems() == []
        assert quality_marks_live_problems() == []

    def test_format_doc_and_proof(self):
        data = quality_marks_as_data()
        assert data["format"] == MARKS_FORMAT
        assert data["marks_doc"] == MARKS_DOC
        assert (REPO_ROOT / MARKS_DOC).exists()
        assert data["proof"]["live_model_marks"] == "not_proven"
        assert data["proof"]["ui"] == "consumers_own_not_built_here"
        assert data["proof"]["policy_surface"] == "unchanged"
        json.dumps(data)

    def test_rule_ids_unique(self):
        ids = [r["id"] for r in MARKS_RULES]
        assert len(ids) == len(set(ids)) == 10

    def test_kinds_and_labels(self):
        assert MARK_KINDS == ("cited", "rescued", "flagged", "warning")
        for check_id in ("format", "truncation", "loop", "drift"):
            assert RESCUE_LABELS[check_id]
        assert RESCUE_LABELS["salvage_refine"]

    def test_fixture_counts(self):
        assert len(MARKS_FIXTURES) == 24
        sources = [f["source"] for f in MARKS_FIXTURES]
        assert sources.count("policy") == 15
        assert sources.count("configured") == 2
        assert sources.count("salvage") == 5
        assert sources.count("combined") == 1
        assert sources.count("synthetic_salvage") == 1


# ── Cited: two truths at once ──────────────────────────────────


class TestCited:
    @pytest.mark.parametrize("fixture_id,markers", [
        ("cited_entity_per_claim", ["Q123"]),
        ("cited_reference_chunk_per_claim", ["ref:chunk_7"]),
        ("legacy_floor_cited", ["Q123"]),
    ])
    def test_cited_marks_name_the_markers(self, fixture_id, markers):
        built = quality_marks(_policy(fixture_id))
        assert built.kinds() == ("cited",)
        detail = built.mark("cited").detail
        assert detail["markers"] == markers
        assert detail["mode"] == _FIXTURE[fixture_id]["cited_mode"]
        for marker in markers:
            assert marker in marks_line(built)

    def test_no_claim_pass_earns_no_cited_mark(self):
        # The weather answer passes its citation
        # check claiming no stored fact — a pass
        # is not a citation, and no mark is filed.
        result = _policy("no_claim_pass_no_cited_mark")
        assert result.check("citation").status == "passed"
        assert quality_marks(result).marks == ()

    def test_flagged_citation_is_flagged_never_cited(self):
        built = quality_marks(_policy("citation_flagged_not_cited"))
        assert built.kinds() == ("flagged",)
        with pytest.raises(KeyError):
            built.mark("cited")


# ── Rescued: mechanisms named, adopted not attempted ──────────


class TestRescued:
    @pytest.mark.parametrize("fixture_id,mechanism", [
        ("loop_rescued", "loop"),
        ("format_rescued", "format"),
        ("truncation_rescued", "truncation"),
        ("drift_rescued", "drift"),
    ])
    def test_policy_rescues_name_their_check(self, fixture_id, mechanism):
        built = quality_marks(_policy(fixture_id))
        assert built.kinds() == ("rescued",)
        (entry,) = built.mark("rescued").detail["mechanisms"]
        assert entry["mechanism"] == mechanism
        assert entry["label"] == RESCUE_LABELS[mechanism]
        assert RESCUE_LABELS[mechanism] in marks_line(built)

    def test_salvage_adopted_pass_is_marked(self):
        built = quality_marks(
            salvage_result=run_rescue_fixture("refine_adopted"))
        assert built.kinds() == ("rescued",)
        (entry,) = built.mark("rescued").detail["mechanisms"]
        assert entry["mechanism"] == "salvage_refine"
        assert entry["final_pass"] == 1
        assert "pass 1" in marks_line(built)

    def test_adopt_then_fabricate_names_pass_1_not_path_2(self):
        result = run_rescue_fixture("refine_adopt_then_fabricate")
        assert result.salvage_path == "refine_pass_2"
        built = quality_marks(salvage_result=result)
        (entry,) = built.mark("rescued").detail["mechanisms"]
        assert entry["final_pass"] == 1

    def test_rejected_attempt_earns_no_mark(self):
        result = run_rescue_fixture("refine_rejected_fabrication")
        assert result.salvage_path == "refine_pass_1"
        assert quality_marks(salvage_result=result).marks == ()

    def test_best_draft_earns_no_mark(self):
        built = quality_marks(
            salvage_result=run_rescue_fixture("best_draft_no_engine"))
        assert built.marks == ()
        assert built.salvage_final_source == "selected_draft"

    def test_recordless_salvage_earns_no_mark(self):
        # A path string with no TF-097 record
        # behind it would lie about adoption.
        result = SalvageResult(
            text="The ferry leaves at dawn.", citations=["Q123"],
            confidence=0.5, passes_used=1, best_draft_score=0.5,
            fragments_kept=1, salvage_path="refine_pass_1",
            rescue_record=None)
        assert quality_marks(salvage_result=result).marks == ()

    def test_combined_rescue_names_both_mechanisms(self):
        built = quality_marks(
            _policy("combined_policy_and_salvage_rescued"),
            salvage_result=run_rescue_fixture("refine_adopted"))
        assert built.kinds() == ("rescued",)
        names = [m["mechanism"]
                 for m in built.mark("rescued").detail["mechanisms"]]
        assert names == ["loop", "salvage_refine"]


# ── Flagged and warning: verbatim, with reasons ───────────────


class TestFlaggedAndWarning:
    @pytest.mark.parametrize("fixture_id,flag", [
        ("citation_flagged_not_cited", "citation:uncited_claim"),
        ("too_short_flagged", "length:too_short"),
        ("generic_flagged", "genericness:hard_generic_phrase"),
        ("drift_flagged", "drift:whole_answer_off_language"),
    ])
    def test_flags_verbatim_with_catalogued_reasons(
            self, fixture_id, flag):
        built = quality_marks(_policy(fixture_id))
        assert built.kinds() == ("flagged",)
        detail = built.mark("flagged").detail
        assert detail["flags"] == [flag]
        (reason,) = detail["reasons"]
        assert reason["flag"] == flag
        assert reason["catalogued"] is True
        assert reason["reason"] == REASONS[flag]["reason"]
        assert flag in marks_line(built)

    def test_rescued_and_flagged_carry_both(self):
        built = quality_marks(_policy("rescued_and_flagged"))
        assert built.kinds() == ("rescued", "flagged")

    def test_relaxed_flag_rides_as_warning_never_flagged(self):
        built = quality_marks(_policy("relaxed_citation_warning"))
        assert built.kinds() == ("warning",)
        assert built.strictness == "relaxed"
        assert built.disposition == "downgraded_to_warnings"
        detail = built.mark("warning").detail
        assert detail["flags"] == ["citation:uncited_claim"]
        assert detail["base_verdict"] == "flagged"


# ── Only a leaving answer gets marks ──────────────────────────


class TestOnlyALeavingAnswer:
    def test_refused_raises(self):
        with pytest.raises(ValueError, match="blocked answer"):
            quality_marks(_policy("refused_empty_raises"))

    def test_strict_block_raises(self):
        with pytest.raises(ValueError, match="blocked answer"):
            quality_marks(_policy("strict_blocked_raises"))

    def test_empty_salvage_raises(self):
        with pytest.raises(ValueError, match="empty salvage"):
            quality_marks(
                salvage_result=run_rescue_fixture("empty_candidates"))

    def test_no_record_at_all_raises(self):
        with pytest.raises(ValueError, match="no record to mark"):
            quality_marks()


# ── A plain answer, and the rendering ─────────────────────────


class TestPlainAndRendering:
    def test_plain_answer_has_no_marks(self):
        built = quality_marks(_policy("clean_plain_no_marks"))
        assert isinstance(built, QualityMarks)
        assert built.marks == ()
        assert built.has_marks is False
        assert marks_line(built) == ""
        assert built.to_dict()["has_marks"] is False

    def test_marked_line_shape(self):
        built = quality_marks(_policy("cited_entity_per_claim"))
        line = marks_line(built)
        assert line.startswith("Forge marks: ")
        assert line.endswith(".")
        assert "Q123" in line
        json.dumps(built.to_dict())

    def test_marks_never_alter_the_answer(self):
        result = _policy("citation_flagged_not_cited")
        before = result.text
        quality_marks(result)
        assert result.text == before


# ── What this slice does not touch ─────────────────────────────


class TestUntouched:
    def test_salvage_itself_is_unchanged(self):
        result = run_fixture("loop_verbatim")
        assert result.salvage_path == "best_draft"

    def test_the_policy_surface_is_unchanged(self):
        result = enforce_policy(" ".join([_BELL] * 4))
        assert result.verdict == "rescued"
        assert result.text == _BELL

    def test_the_configured_layer_is_unchanged(self):
        result = enforce_configured(
            _ODIN_UNCITED, strictness="relaxed", records=_records())
        assert result.verdict == "clean"
        assert result.warnings == ("citation:uncited_claim",)
