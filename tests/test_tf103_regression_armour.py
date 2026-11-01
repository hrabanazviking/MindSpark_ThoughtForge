"""TF-103 — regression armour: every historical
bad output named in the repo's docs becomes a
fixture before its fix is trusted.

Pins refinement/regression_armour.py: the
registry's data integrity (source docs on disk,
rerun checkers resolving, reference needles
present in their test files, gap families and
owners exactly TF-101's), the live re-proof
(every rerun checker green, every direct
fixture landing on its real fix surface, the
gap shape recomputed at exactly 9 mismatches
in the 3 filed families), the direct fixtures'
individual legs, and the tamper proofs (a
registry case with a missing doc, a dead
needle, a wrong gap owner, or an unknown kind
fails the data checker).
"""

from __future__ import annotations

import copy

import pytest

from thoughtforge.refinement import regression_armour as armour
from thoughtforge.refinement.regression_armour import (
    ARMOUR_CASES,
    ARMOUR_FIXTURES,
    ARMOUR_FORMAT,
    ARMOUR_KINDS,
    ARMOUR_RULES,
    COVERED_KINDS,
    GAP_FAMILIES,
    GAP_MISMATCH_TOTAL,
    armour_as_data,
    case_by_id,
    kind_counts,
    regression_armour_live_problems,
    regression_armour_problems,
)


# ── Data integrity ──────────────────────────────────────────────────────────


def test_data_checker_green():
    assert regression_armour_problems() == []


def test_live_checker_green():
    assert regression_armour_live_problems() == []


def test_format_and_doc_named():
    data = armour_as_data()
    assert data["format"] == ARMOUR_FORMAT
    assert data["armour_doc"] == \
        "docs/architecture/REGRESSION_ARMOUR.md"


def test_rule_ids_unique():
    ids = [r["id"] for r in ARMOUR_RULES]
    assert len(ids) == len(set(ids))


def test_case_ids_unique_and_lookup_detached():
    ids = [c["id"] for c in ARMOUR_CASES]
    assert len(ids) == len(set(ids))
    case = case_by_id(ids[0])
    case["name"] = "tampered"
    assert case_by_id(ids[0])["name"] != "tampered"


def test_kind_counts_cover_registry():
    counts = kind_counts()
    assert set(counts) == set(ARMOUR_KINDS)
    assert sum(counts.values()) == len(ARMOUR_CASES)
    assert counts == {"rerun": 12, "direct": 5, "reference": 3,
                      "gap_open": 3, "not_proven": 2}


def test_covered_is_rerun_direct_reference_only():
    data = armour_as_data()
    assert data["covered"] == 20
    assert data["total_cases"] == 25
    assert set(COVERED_KINDS) == {"rerun", "direct", "reference"}


def test_every_case_names_doc_slice_and_owner():
    for case in ARMOUR_CASES:
        assert case["source_doc"].endswith(".md")
        assert case["source_slice"].startswith("TF-")
        assert case["fix_owner"].startswith("TF-")


def test_gap_families_sum_to_filed_total():
    assert sum(g["entries"] for g in GAP_FAMILIES.values()) == \
        GAP_MISMATCH_TOTAL == 9
    assert {g["owner"] for g in GAP_FAMILIES.values()} == \
        {"TF-088", "TF-093", "TF-096"}


def test_not_proven_cases_carry_reasons_and_docket_ids():
    cases = [c for c in ARMOUR_CASES if c["kind"] == "not_proven"]
    assert {c["docket"] for c in cases} == {"D-AESIR-1", "D-OAI-4"}
    for case in cases:
        assert case["reason"].strip()


def test_no_canon_markers_anywhere():
    blob = repr(ARMOUR_CASES) + repr(ARMOUR_FIXTURES)
    for marker in armour.FORBIDDEN_MARKERS:
        assert marker not in blob


# ── Tamper proofs ───────────────────────────────────────────────────────────


def _tampered(**changes):
    cases = copy.deepcopy(list(ARMOUR_CASES))
    cases[0].update(changes)
    return tuple(cases)


def test_tamper_missing_source_doc_fails(monkeypatch):
    monkeypatch.setattr(
        armour, "ARMOUR_CASES",
        _tampered(source_doc="docs/architecture/NOPE.md"))
    assert any("not on disk" in p
               for p in regression_armour_problems())


def test_tamper_unknown_kind_fails(monkeypatch):
    monkeypatch.setattr(
        armour, "ARMOUR_CASES", _tampered(kind="vibes"))
    assert any("unknown kind" in p
               for p in regression_armour_problems())


def test_tamper_dead_needle_fails(monkeypatch):
    cases = copy.deepcopy(list(ARMOUR_CASES))
    for case in cases:
        if case["kind"] == "reference":
            case["needle"] = "a needle that is not in the file"
            break
    monkeypatch.setattr(armour, "ARMOUR_CASES", tuple(cases))
    assert any("not in" in p for p in regression_armour_problems())


def test_tamper_wrong_gap_owner_fails(monkeypatch):
    cases = copy.deepcopy(list(ARMOUR_CASES))
    for case in cases:
        if case["kind"] == "gap_open":
            case["fix_owner"] = "TF-999"
            break
    monkeypatch.setattr(armour, "ARMOUR_CASES", tuple(cases))
    assert any("gap owner" in p for p in regression_armour_problems())


def test_tamper_dead_checker_fails(monkeypatch):
    monkeypatch.setattr(
        armour, "ARMOUR_CASES",
        _tampered(checker="thoughtforge.refinement.nope:nope"))
    assert regression_armour_problems() != []


# ── Direct fixture legs ─────────────────────────────────────────────────────


def test_router_dead_pattern_legs():
    from thoughtforge.cognition.router import InputRouter

    fixture = next(f for f in ARMOUR_FIXTURES
                   if f["id"] == "router_dead_patterns")
    router = InputRouter()
    for leg in fixture["legs"]:
        sketch = router.route(leg["text"])
        assert getattr(sketch, leg["field"]) == leg["expected"]


def test_contradiction_leg_surfaces_not_overwrites():
    from thoughtforge.knowledge.contradictions import (
        assess_contradiction,
    )
    from thoughtforge.knowledge.models import UserFactRecord

    stored = UserFactRecord(
        record_id="s", summary="The ferry leaves at dawn",
        fact_value={"ferry_departure": "dawn"}, confidence=0.6,
        source="", tags=[])
    new = UserFactRecord(
        record_id="n", summary="The ferry leaves at midnight",
        fact_value={"ferry_departure": "midnight"},
        confidence=0.6, source="", tags=[])
    report = assess_contradiction(new, [stored])
    assert len(report.conflicts) == 1
    assert report.conflicts[0].rule == "fact_value_slot"
    assert report.conflicts[0].resolution == "preserve_both_flagged"


def test_oai_double_v1_leg():
    from thoughtforge.inference.openai_compatible import (
        normalize_base_url,
    )

    assert normalize_base_url("http://127.0.0.1:1234/v1") == \
        "http://127.0.0.1:1234"


def test_oai_outcome_legs():
    from thoughtforge.inference.openai_compatible import (
        chat_completion_outcome,
    )

    fixture = next(f for f in ARMOUR_FIXTURES
                   if f["id"] == "oai_outcome_legs")
    for leg in fixture["legs"]:
        outcome = chat_completion_outcome(copy.deepcopy(leg["payload"]))
        assert outcome["ok"] is leg["expected_ok"], leg["name"]


def test_loop_20x_leg():
    from thoughtforge.refinement.loop_rescue import rescue_loops

    result = rescue_loops(" ".join(["The bell rings."] * 20))
    assert result.verdict == "rescued"
    assert result.text.count("The bell rings.") == 1


# ── Gap shape ───────────────────────────────────────────────────────────────


def test_gap_shape_is_exactly_the_filed_nine():
    from thoughtforge.refinement.chaos_run import run_chaos

    run = run_chaos()
    assert len(run["mismatches"]) == GAP_MISMATCH_TOTAL
    families = {}
    for row in run["mismatches"]:
        families[row["family"]] = families.get(row["family"], 0) + 1
    assert families == {name: g["entries"]
                        for name, g in GAP_FAMILIES.items()}


def test_gap_open_cases_are_not_counted_covered():
    data = armour_as_data()
    assert data["proof"]["gap_open_cases"] == 3
    assert data["proof"]["not_proven_cases"] == 2
    assert data["proof"]["live_model_regressions"] == "not_proven"


def test_as_data_is_detached_and_complete():
    data = armour_as_data()
    data["cases"][0]["name"] = "tampered"
    assert armour_as_data()["cases"][0]["name"] != "tampered"
    assert len(data["cases"]) == len(ARMOUR_CASES)
    assert len(data["fixtures"]) == len(ARMOUR_FIXTURES)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
