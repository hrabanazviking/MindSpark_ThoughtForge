"""
TF-149 — The wing's tests: adopted code carries
shipping-grade tests; retired code's tests
archived with it.

The proof lives as verifiable data in
``src/thoughtforge/wing_tests.py``
(``WING_TEST_ROWS``, ``ADOPTED_TEST_ROWS``,
``GRADE_ROWS``, ``WING_TEST_PROBES``, checked by
``wing_tests_problems()``); the doc
``docs/architecture/WING_TESTS.md`` is filed
verbatim as the module's render.

Pinned here:

  * the wing's 4 test functions in 2 files, each
    with exactly one final disposition: 2
    archived with retired code (banner pinned
    live; TF-140's 29 / TF-142's 33 verdict tests
    named as evidence, not production tests), 1
    superseded by TF-143's shipping tests (named
    counterparts checked live), 1 fenced with its
    unadopted subject (no counterpart claimed),
  * TF-148 uncontradicted: no banner added or
    removed, no wing file moved, the models
    package still absent, the root testpaths
    still never collecting the wing,
  * the adopted estate: 4 production modules
    carrying 93 shipping tests (TF-143's file 41
    attributed 13 + 28 with the cross-cutting 6
    counted once under the persona row; TF-146
    26; TF-147 26), every count recomputed by
    AST, every module checker clean live,
  * the shipping-grade criteria: 6 rows,
  * D-WING-1 CLOSED-BY-VERDICT by TF-149, the
    last share its narrowings assign; TF-016
    totals 137 — 31 OPEN / 1 OPEN-ENVIRONMENT /
    101 FIXED / 4 CLOSED-BY-VERDICT; every prior
    narrowing's text preserved,
  * the probes: 7/7 against the pure validators,
  * the doc matches its render verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live model was
read, never claimed.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from thoughtforge.wing_tests import (
    ADOPTED_SHIPPING_TEST_TOTAL,
    ADOPTED_TEST_ROWS,
    ARCHIVE_BANNER_MARKER,
    FENCED_WING_TEST_FILE,
    GRADE_ROWS,
    RETIRED_WING_TEST_FILE,
    TF148_FILE_DISPOSITIONS,
    WING_TESTS_DOC,
    WING_TESTS_EXPECTED,
    WING_TESTS_FORMAT,
    WING_TEST_DISPOSITIONS,
    WING_TEST_PROBES,
    WING_TEST_ROWS,
    adopted_row_problems,
    doc_text_problems,
    grade_row_problems,
    render_wing_tests_markdown,
    run_wing_tests_probes,
    run_wing_tests_state,
    wing_function_set_problems,
    wing_test_row_problems,
    wing_tests_problems,
    wing_tests_state_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _ast_functions(rel: str) -> list[str]:
    tree = ast.parse(_read(rel))
    return [node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test")]


def _ast_classes(rel: str) -> dict[str, int]:
    tree = ast.parse(_read(rel))
    return {node.name: sum(
        1 for item in node.body
        if isinstance(item, ast.FunctionDef)
        and item.name.startswith("test"))
        for node in tree.body if isinstance(node, ast.ClassDef)}


# ── The filing ───────────────────────────────────────────────────────────────

class TestFiling:
    def test_problems_empty(self) -> None:
        assert wing_tests_problems() == []

    def test_format_is_versioned(self) -> None:
        assert WING_TESTS_FORMAT == "thoughtforge-wing-tests/1"

    def test_four_wing_rows_each_with_one_disposition(self) -> None:
        assert len(WING_TEST_ROWS) == 4
        for row in WING_TEST_ROWS:
            assert wing_test_row_problems(row) == []
            assert row["disposition"] in WING_TEST_DISPOSITIONS
        by_disp: dict[str, int] = {}
        for row in WING_TEST_ROWS:
            by_disp[row["disposition"]] = \
                by_disp.get(row["disposition"], 0) + 1
        assert by_disp == {
            "archived_with_retired_code": 2,
            "superseded_by_shipping_tests": 1,
            "fenced_with_unadopted_subject": 1}

    def test_wing_rows_are_the_four_measured_functions(self) -> None:
        filed = {(r["file"], r["test_name"]) for r in WING_TEST_ROWS}
        assert filed == {
            (RETIRED_WING_TEST_FILE,
             "test_micro_rag_assembles_packet"),
            (RETIRED_WING_TEST_FILE,
             "test_truth_calibrator_flags_unsupported_exclusivity"),
            (FENCED_WING_TEST_FILE,
             "test_persona_compiler_builds_packet"),
            (FENCED_WING_TEST_FILE,
             "test_memory_store_search_and_promote")}

    def test_retired_rows_name_verdict_evidence_not_production(
            self) -> None:
        retired = [r for r in WING_TEST_ROWS
                   if r["disposition"] == "archived_with_retired_code"]
        evidence = {r["evidence_file"]: r["evidence_test_count"]
                    for r in retired}
        assert evidence == {
            "tests/test_tf140_micro_rag_verdict.py": 29,
            "tests/test_tf142_truth_calibrator_verdict.py": 33}
        for row in retired:
            assert row["shipping_counterparts"] == ()
            assert "RETIRE" in row["subject_verdict"]
            assert "evidence, not" in row["evidence_note"]

    def test_persona_row_names_live_shipping_counterparts(self) -> None:
        row = next(r for r in WING_TEST_ROWS
                   if r["id"] == "persona_packet")
        assert row["disposition"] == "superseded_by_shipping_tests"
        assert row["shipping_counterparts"] == (
            "test_main_counts_move_exactly_as_filed",
            "test_inject_persona_delivers_base_and_compiled")
        live = set(_ast_functions("tests/test_tf143_adoptions.py"))
        for name in row["shipping_counterparts"]:
            assert name in live

    def test_memory_row_claims_no_counterpart(self) -> None:
        row = next(r for r in WING_TEST_ROWS
                   if r["id"] == "memory_store_search")
        assert row["disposition"] == "fenced_with_unadopted_subject"
        assert row["shipping_counterparts"] == ()
        assert "NO shipping counterpart is claimed" in \
            row["counterpart_reason"]
        assert "never adopted" in row["subject_verdict"]

    def test_adopted_rows_sum_to_93_and_reconcile(self) -> None:
        assert len(ADOPTED_TEST_ROWS) == 4
        for row in ADOPTED_TEST_ROWS:
            assert adopted_row_problems(row) == []
        assert sum(r["test_count"] for r in ADOPTED_TEST_ROWS) == \
            ADOPTED_SHIPPING_TEST_TOTAL == 93

    def test_tf143_attribution_counts_cross_cutting_once(self) -> None:
        rows = {r["id"]: r for r in ADOPTED_TEST_ROWS}
        assert rows["bond_graph"]["class_counts"] == {
            "TestBondGraphLanding": 13}
        assert rows["persona_compiler"]["class_counts"] == {
            "TestPersonaCompilerLanding": 17,
            "TestMergedDelivery": 5, "TestCrossCutting": 6}
        assert rows["persona_compiler"]["test_count"] == 28
        tf143_total = rows["bond_graph"]["test_count"] + \
            rows["persona_compiler"]["test_count"]
        assert tf143_total == rows["bond_graph"]["file_test_count"] \
            == 41

    def test_grade_rows_are_six(self) -> None:
        assert [r["id"] for r in GRADE_ROWS] == [
            "lives_under_tests_and_runs_in_the_full_gate",
            "module_checker_clean", "probes_pass",
            "doc_render_pinned_verbatim", "deterministic",
            "synthetic_fixtures_only"]
        for row in GRADE_ROWS:
            assert grade_row_problems(row) == []


# ── Live wing state ──────────────────────────────────────────────────────────

class TestLiveWing:
    def test_ast_function_sets_match_the_filing_per_file(self) -> None:
        for rel in (RETIRED_WING_TEST_FILE, FENCED_WING_TEST_FILE):
            filed = [r["test_name"] for r in WING_TEST_ROWS
                     if r["file"] == rel]
            assert wing_function_set_problems(
                filed, _ast_functions(rel)) == []
        total = len(_ast_functions(RETIRED_WING_TEST_FILE)) + \
            len(_ast_functions(FENCED_WING_TEST_FILE))
        assert total == 4

    def test_banner_only_on_the_retired_file(self) -> None:
        retired = _read(RETIRED_WING_TEST_FILE)
        assert retired.startswith(ARCHIVE_BANNER_MARKER)
        assert ARCHIVE_BANNER_MARKER not in _read(
            FENCED_WING_TEST_FILE)

    def test_tf148_filing_stands_uncontradicted(self) -> None:
        from thoughtforge.research_hygiene import FILE_ROWS

        tf148 = {r["path"]: r["disposition"] for r in FILE_ROWS}
        assert tf148 == {**tf148, **TF148_FILE_DISPOSITIONS} or all(
            tf148[path] == disp
            for path, disp in TF148_FILE_DISPOSITIONS.items())
        for row in WING_TEST_ROWS:
            assert row["tf148_disposition"] == \
                TF148_FILE_DISPOSITIONS[row["file"]]

    def test_models_package_still_absent_and_root_fence_stands(
            self) -> None:
        assert not (REPO_ROOT / "research_data/src/wyrdforge/"
                    "models").exists()
        data = tomllib.loads(_read("pyproject.toml"))
        assert data["tool"]["pytest"]["ini_options"][
            "testpaths"] == ["tests"]


# ── The adopted estate, live ─────────────────────────────────────────────────

class TestAdoptedLive:
    def test_class_counts_recompute_by_ast(self) -> None:
        for row in ADOPTED_TEST_ROWS:
            measured = _ast_classes(row["shipping_test_file"])
            for class_name, count in row["class_counts"].items():
                assert measured[class_name] == count
            assert len(_ast_functions(
                row["shipping_test_file"])) == \
                row["file_test_count"]
            assert (REPO_ROOT / row["production_path"]).exists()

    def test_no_class_is_counted_twice_across_rows(self) -> None:
        seen: set[tuple[str, str]] = set()
        for row in ADOPTED_TEST_ROWS:
            for class_name in row["class_counts"]:
                key = (row["shipping_test_file"], class_name)
                assert key not in seen
                seen.add(key)

    def test_verdict_evidence_counts_recompute_by_ast(self) -> None:
        assert len(_ast_functions(
            "tests/test_tf140_micro_rag_verdict.py")) == 29
        assert len(_ast_functions(
            "tests/test_tf142_truth_calibrator_verdict.py")) == 33

    def test_neighbouring_checkers_still_clean(self) -> None:
        from thoughtforge.cognition.persona_compiler import (
            persona_compiler_problems,
        )
        from thoughtforge.cognition.runmal_cards import (
            runmal_cards_problems,
        )
        from thoughtforge.knowledge.bond_bridge import (
            bond_bridge_problems,
        )
        from thoughtforge.knowledge.bond_graph import (
            bond_graph_problems,
        )
        from thoughtforge.research_hygiene import (
            research_hygiene_problems,
        )
        from thoughtforge.wing_retirements import (
            wing_retirements_problems,
        )

        assert research_hygiene_problems() == []
        assert wing_retirements_problems() == []
        assert bond_graph_problems() == []
        assert persona_compiler_problems() == []
        assert bond_bridge_problems() == []
        assert runmal_cards_problems() == []


# ── State, probes, validators ────────────────────────────────────────────────

class TestStateAndProbes:
    def test_state_matches_its_filing(self) -> None:
        state = run_wing_tests_state()
        assert state == WING_TESTS_EXPECTED
        assert wing_tests_state_problems(
            state, WING_TESTS_EXPECTED) == []

    def test_probes_all_pass(self) -> None:
        probes = run_wing_tests_probes()
        assert len(probes) == len(WING_TEST_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert wing_test_row_problems(None)
        assert adopted_row_problems(None)
        assert grade_row_problems(None)
        assert wing_function_set_problems(None, None)
        assert wing_tests_state_problems(None, None)
        assert doc_text_problems(None, None, "x")

    def test_each_probe_shape_is_named_by_the_validator(self) -> None:
        row = dict(WING_TEST_ROWS[0],
                   disposition="fenced_with_unadopted_subject")
        assert any("archived" in p
                   for p in wing_test_row_problems(row))
        row = dict(WING_TEST_ROWS[2], shipping_counterparts=())
        assert any("counterpart" in p
                   for p in wing_test_row_problems(row))
        row = dict(ADOPTED_TEST_ROWS[0], test_count=99)
        assert adopted_row_problems(row)
        row = dict(WING_TEST_ROWS[0], disposition="limbo")
        assert wing_test_row_problems(row)
        assert wing_function_set_problems(
            ["test_a"], ["test_a", "test_b"])
        assert wing_function_set_problems(
            ["test_a", "test_b"], ["test_a"])
        assert doc_text_problems(
            render_wing_tests_markdown() + "x",
            render_wing_tests_markdown(), "wing tests doc")


# ── Doc + docket ─────────────────────────────────────────────────────────────

class TestDocAndDocket:
    def test_doc_matches_render(self) -> None:
        assert _read(WING_TESTS_DOC) == render_wing_tests_markdown()

    def test_doc_files_the_dispositions_and_limits(self) -> None:
        doc = _read(WING_TESTS_DOC)
        assert "TF-149" in doc and "D-WING-1" in doc
        assert "archived_with_retired_code" in doc
        assert "superseded_by_shipping_tests" in doc
        assert "fenced_with_unadopted_subject" in doc
        assert "93" in doc
        assert "CLOSED-BY-VERDICT" in doc
        assert "NOT PROVEN" in doc
        assert "no README" in doc or "No README" in doc or \
            "README.md anywhere was touched" in doc

    def test_docket_closed_by_verdict_totals_moved(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries" in docket
        assert "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, " \
            "101 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT (D-CARD-2 by TF-086's Part " \
            "IV gate verdict; D-WING-1 by TF-149)" in normalized

    def test_docket_closure_preserves_every_prior_narrowing(
            self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        wing_line = next(
            line for line in docket.splitlines()
            if line.startswith("| D-WING-1 |"))
        for marker in ("narrowed by TF-139", "Narrowed by TF-140",
                       "Narrowed by TF-141", "Narrowed by TF-142",
                       "Narrowed by TF-143", "Narrowed by TF-144",
                       "Narrowed by TF-148",
                       "stays OPEN for the TF-149",
                       "research-hygiene share only"):
            assert marker in wing_line
        assert "**Closed by TF-149**" in wing_line
        assert "the last share this entry assigns" in wing_line
        assert "**CLOSED-BY-VERDICT**" in wing_line
        assert "TF-150/TF-152 own provenance/gate" in wing_line
        assert "93 shipping-grade tests" in wing_line

    def test_fixtures_are_synthetic_only(self) -> None:
        # This slice's own files file no canon identifiers:
        # the wing rows name mechanisms and test names only.
        for row in WING_TEST_ROWS:
            assert "volmarr" not in str(row).lower()
            assert "veyrunn" not in str(row).lower()
