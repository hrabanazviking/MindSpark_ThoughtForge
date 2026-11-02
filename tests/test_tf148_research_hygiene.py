"""
TF-148 — Research hygiene: research_data/ either
graduates, archives, or is clearly fenced as
research with its own test discipline — no
ambiguous middle state remains.

The proof lives as verifiable data in
``src/thoughtforge/research_hygiene.py``
(``AREA_ROWS``, ``FILE_ROWS``,
``TEST_DISCIPLINE_ROWS``, ``FENCE_RULES``,
``HYGIENE_PROBES``, checked by
``research_hygiene_problems()``); the hygiene doc
``docs/architecture/RESEARCH_HYGIENE.md`` and the
fence notice ``research_data/RESEARCH_FENCE.md``
are filed verbatim as the module's renders.

Pinned here:

  * the census: 12 baseline areas / 96 tracked
    files, plus the fence file = 97 once banked;
    the pure tracked-file validator accepts the
    fence file tracked or not yet tracked,
  * every code/test file filed with exactly one
    disposition: 2 graduated, 3 archived-retired,
    3 fenced research, 2 fenced standalone, 1
    fence import — the five previously
    undispositioned files now filed,
  * the sparse discipline: disk counts judged
    only for sparse-present areas; the 4
    sparse-excluded areas judged only through
    the tracked-file validator,
  * the fences live: both pyprojects parsed, no
    shipping import of the wing package, the
    models package still absent, the package
    init's import lines unaltered,
  * the test discipline: main suite, wing suite
    (collection error filed, not hidden),
    graduated tests, retired tests — TF-149's
    share explicitly not discharged,
  * the docket narrowed for the hygiene share
    only, totals unmoved,
  * the probes: 7/7 against the pure validators,
  * both docs match their renders verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live model was
read, never claimed.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from thoughtforge.research_hygiene import (
    AREA_DISPOSITIONS,
    AREA_ROWS,
    ARCHIVE_BANNER_MARKER,
    FENCE_RULES,
    FILE_DISPOSITIONS,
    FILE_ROWS,
    HYGIENE_EXPECTED,
    HYGIENE_PROBES,
    RESEARCH_FENCE_DOC,
    RESEARCH_HYGIENE_DOC,
    RESEARCH_HYGIENE_FORMAT,
    TEST_DISCIPLINE_ROWS,
    area_row_problems,
    classify_tracked_path,
    doc_text_problems,
    fence_rule_problems,
    file_row_problems,
    hygiene_state_problems,
    render_research_fence_markdown,
    render_research_hygiene_markdown,
    research_hygiene_problems,
    run_research_hygiene_probes,
    run_research_hygiene_state,
    discipline_row_problems,
    tracked_files_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _tracked() -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "research_data"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        check=True).stdout.splitlines()


# ── The filing ───────────────────────────────────────────────────────────────

class TestFiling:
    def test_problems_empty(self) -> None:
        assert research_hygiene_problems() == []

    def test_format_is_versioned(self) -> None:
        assert RESEARCH_HYGIENE_FORMAT == \
            "thoughtforge-research-hygiene/1"

    def test_twelve_baseline_areas_sum_to_96(self) -> None:
        baseline = [r for r in AREA_ROWS
                    if r["id"] != "research_fence_file"]
        assert len(baseline) == 12
        assert sum(r["tracked_count"] for r in baseline) == 96
        assert len(AREA_ROWS) == 13
        assert sum(r["tracked_count"] for r in AREA_ROWS) == 97
        for row in AREA_ROWS:
            assert area_row_problems(row) == []
            assert row["disposition"] in AREA_DISPOSITIONS

    def test_area_counts_are_the_measured_census(self) -> None:
        counts = {r["id"]: r["tracked_count"] for r in AREA_ROWS}
        assert counts == {
            "top_level_markdown": 58, "schemas": 14,
            "services": 5, "wyrd_runtime_packets": 5,
            "config_files": 4, "examples_files": 2,
            "security": 2, "wing_tests": 2, "runtime": 1,
            "scripts_files": 1, "package_init": 1,
            "wing_pyproject": 1, "research_fence_file": 1}

    def test_sparse_excluded_areas_are_the_four_filed(self) -> None:
        excluded = [r["id"] for r in AREA_ROWS
                    if not r["sparse_present"]
                    and r["id"] != "research_fence_file"]
        # The fence file is on disk; the excluded four are not.
        assert excluded == [
            "wyrd_runtime_packets", "config_files",
            "examples_files", "scripts_files"]
        assert sum(r["tracked_count"] for r in AREA_ROWS
                   if r["id"] in excluded) == 12
        for area_id in excluded:
            pass
        for name in ("config", "examples", "scripts",
                     "wyrd_runtime"):
            assert not (REPO_ROOT / "research_data" / name).exists()

    def test_every_code_file_files_exactly_one_disposition(self) -> None:
        assert len(FILE_ROWS) == 11
        for row in FILE_ROWS:
            assert file_row_problems(row) == []
            assert row["disposition"] in FILE_DISPOSITIONS
        by_disp: dict[str, int] = {}
        for row in FILE_ROWS:
            by_disp[row["disposition"]] = \
                by_disp.get(row["disposition"], 0) + 1
        assert by_disp == {
            "graduated_source": 2, "archived_retired": 3,
            "fenced_research": 3, "fenced_standalone": 2,
            "fence_import": 1}

    def test_file_rows_reconcile_with_split_areas(self) -> None:
        for row in AREA_ROWS:
            if row["disposition"] != "split_by_file":
                continue
            filed = [r for r in FILE_ROWS
                     if r["area"] == row["id"]]
            assert len(filed) == row["tracked_count"]

    def test_previously_undispositioned_files_are_now_filed(self) -> None:
        by_path = {r["path"]: r["disposition"] for r in FILE_ROWS}
        assert by_path[
            "research_data/src/wyrdforge/services/memory_store.py"
        ] == "fenced_research"
        assert by_path[
            "research_data/src/wyrdforge/runtime/demo_seed.py"
        ] == "fenced_research"
        assert by_path[
            "research_data/tests/test_memory_and_persona.py"
        ] == "fenced_research"
        assert by_path[
            "research_data/src/wyrdforge/security/permission_guard.py"
        ] == "fenced_standalone"
        assert by_path[
            "research_data/src/wyrdforge/security/"
            "prompt_injection_guard.py"] == "fenced_standalone"

    def test_test_discipline_rows_are_four(self) -> None:
        assert [r["id"] for r in TEST_DISCIPLINE_ROWS] == [
            "main_suite", "wing_suite", "graduated_tests",
            "retired_tests"]
        for row in TEST_DISCIPLINE_ROWS:
            assert discipline_row_problems(row) == []
        wing = TEST_DISCIPLINE_ROWS[1]
        assert "collection error" in wing["state"]
        assert "TF-149" in TEST_DISCIPLINE_ROWS[2]["state"]

    def test_fence_rules_are_six(self) -> None:
        assert len(FENCE_RULES) == 6
        for row in FENCE_RULES:
            assert fence_rule_problems(row) == []
        joined = " ".join(r["rule"] for r in FENCE_RULES)
        assert "new roadmap word" in joined
        assert "README" in joined


# ── The tracked-file validator (sparse discipline) ───────────────────────────

class TestTrackedFiles:
    def test_git_tracked_list_passes_in_either_banking_state(self) -> None:
        tracked = _tracked()
        # Before banking the fence file is untracked (96);
        # after banking it is tracked (97). Both validate.
        assert len(tracked) in (96, 97)
        assert tracked_files_problems(tracked) == []
        with_fence = tracked if RESEARCH_FENCE_DOC in tracked \
            else tracked + [RESEARCH_FENCE_DOC]
        assert tracked_files_problems(with_fence) == []

    def test_unclassified_tracked_file_is_named(self) -> None:
        tracked = _tracked() + ["research_data/mystery.txt"]
        problems = tracked_files_problems(tracked)
        assert any("unclassified" in p and "mystery" in p
                   for p in problems)

    def test_wrong_area_count_is_named(self) -> None:
        tracked = [p for p in _tracked()
                   if not p.endswith("vow.schema.json")]
        problems = tracked_files_problems(tracked)
        assert any("schemas" in p for p in problems)

    def test_classification_covers_the_fence_separately(self) -> None:
        assert classify_tracked_path(RESEARCH_FENCE_DOC) == \
            "research_fence_file"
        assert classify_tracked_path("research_data/README.md") == \
            "top_level_markdown"
        assert classify_tracked_path(
            "research_data/config/x.yaml") == "config_files"
        assert classify_tracked_path("elsewhere/x.py") is None


# ── Dispositions, live ───────────────────────────────────────────────────────

class TestLiveDispositions:
    def test_archived_files_open_with_the_tf144_banner(self) -> None:
        for row in FILE_ROWS:
            if row["disposition"] != "archived_retired":
                continue
            text = _read(row["path"])
            assert text.startswith(ARCHIVE_BANNER_MARKER)
            assert row["banner_marker"] == ARCHIVE_BANNER_MARKER

    def test_graduated_files_name_existing_counterparts(self) -> None:
        for row in FILE_ROWS:
            if row["disposition"] != "graduated_source":
                continue
            assert ARCHIVE_BANNER_MARKER not in _read(row["path"])
            assert (REPO_ROOT / row["production_counterpart"]).exists()
        by_path = {r["path"]: r["production_counterpart"]
                   for r in FILE_ROWS
                   if r["disposition"] == "graduated_source"}
        assert by_path == {
            "research_data/src/wyrdforge/services/"
            "bond_graph_service.py":
                "src/thoughtforge/knowledge/bond_graph.py",
            "research_data/src/wyrdforge/services/"
            "persona_compiler.py":
                "src/thoughtforge/cognition/persona_compiler.py"}

    def test_security_guards_import_nothing_from_the_models_package(
            self) -> None:
        for row in FILE_ROWS:
            if row["disposition"] != "fenced_standalone":
                continue
            text = _read(row["path"])
            assert "wyrdforge" not in text

    def test_fenced_files_import_the_missing_models_package(self) -> None:
        for row in FILE_ROWS:
            if row["disposition"] not in (
                    "fenced_research", "graduated_source",
                    "archived_retired"):
                continue
            assert "from wyrdforge.models" in _read(row["path"])
        assert not (
            REPO_ROOT / "research_data/src/wyrdforge/models").exists()

    def test_package_init_import_lines_stand_unaltered(self) -> None:
        text = _read("research_data/src/wyrdforge/__init__.py")
        for name in ("MemoryRecord", "ObservationRecord",
                     "CanonicalFactRecord", "BondEdge", "Vow",
                     "Hurt", "PersonaPacket",
                     "MicroContextPacket", "EvalCase", "EvalResult"):
            assert name in text
        assert text.count("from .models.") == 5


# ── The pyproject fences ─────────────────────────────────────────────────────

class TestPyprojectFences:
    def test_root_testpaths_never_collect_the_wing(self) -> None:
        data = tomllib.loads(_read("pyproject.toml"))
        assert data["tool"]["pytest"]["ini_options"][
            "testpaths"] == ["tests"]

    def test_wing_pyproject_is_its_own_package(self) -> None:
        data = tomllib.loads(_read("research_data/pyproject.toml"))
        assert data["project"]["name"] == "wyrdforge"
        opts = data["tool"]["pytest"]["ini_options"]
        assert opts["pythonpath"] == ["src"]
        assert opts["testpaths"] == ["tests"]

    def test_no_shipping_file_imports_the_wing_package(self) -> None:
        assert run_research_hygiene_state()[
            "shipping_wing_import_files"] == 0


# ── State, probes, validators ────────────────────────────────────────────────

class TestStateAndProbes:
    def test_state_matches_its_filing(self) -> None:
        state = run_research_hygiene_state()
        assert state == HYGIENE_EXPECTED
        assert hygiene_state_problems(
            state, HYGIENE_EXPECTED) == []

    def test_probes_all_pass(self) -> None:
        probes = run_research_hygiene_probes()
        assert len(probes) == len(HYGIENE_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert area_row_problems(None)
        assert file_row_problems(None)
        assert discipline_row_problems(None)
        assert fence_rule_problems(None)
        assert tracked_files_problems(None)
        assert tracked_files_problems([])
        assert hygiene_state_problems(None, None)
        assert doc_text_problems(None, None, "x")

    def test_each_probe_shape_is_named_by_the_validator(self) -> None:
        row = dict(FILE_ROWS[0], disposition="limbo")
        assert file_row_problems(row)
        row = dict(FILE_ROWS[0], production_counterpart="")
        assert any("counterpart" in p
                   for p in file_row_problems(row))
        row = dict(FILE_ROWS[2], banner_marker="")
        assert any("banner" in p for p in file_row_problems(row))
        row = dict(AREA_ROWS[0], disposition="limbo")
        assert area_row_problems(row)
        assert doc_text_problems(
            render_research_fence_markdown() + "x",
            render_research_fence_markdown(), "fence doc")


# ── Docs, fence file, docket, neighbours ─────────────────────────────────────

class TestDocsAndDocket:
    def test_both_docs_match_their_renders(self) -> None:
        assert _read(RESEARCH_HYGIENE_DOC) == \
            render_research_hygiene_markdown()
        assert _read(RESEARCH_FENCE_DOC) == \
            render_research_fence_markdown()

    def test_fence_file_names_the_whole_fence(self) -> None:
        fence = _read(RESEARCH_FENCE_DOC)
        assert "fenced research" in fence
        assert "TF-143" in fence and "TF-144" in fence
        assert "TF-149" in fence
        assert "new roadmap" in fence
        assert "cannot import" in fence

    def test_hygiene_doc_names_tf149s_remaining_share(self) -> None:
        doc = _read(RESEARCH_HYGIENE_DOC)
        assert "TF-148" in doc and "D-WING-1" in doc
        assert "TF-149" in doc
        assert "NOT PROVEN" in doc
        assert "no README" in doc or "No README" in doc

    def test_docket_narrowed_for_hygiene_only_totals_unmoved(
            self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        assert "Narrowed by TF-148" in docket
        assert "research-hygiene share only" in docket
        assert "stays OPEN for the TF-149" in docket
        assert "137 entries" in docket

    def test_neighbouring_checkers_still_clean(self) -> None:
        from thoughtforge.knowledge.bond_graph_verdict import (
            bond_graph_verdict_problems,
        )
        from thoughtforge.wing_retirements import (
            wing_retirements_problems,
        )

        assert wing_retirements_problems() == []
        assert bond_graph_verdict_problems() == []
