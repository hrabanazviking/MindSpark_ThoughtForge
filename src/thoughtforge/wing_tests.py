"""
TF-149 — the wyrdforge wing's tests, finally disposed.

TF-149's roadmap text: *The wing's tests: adopted
code carries shipping-grade tests; retired code's
tests archived with it.*

The measured start, probed live on this tree
before anything was built:

  * The wing carries exactly 2 test files with
    exactly 4 test functions (counted by AST):
    ``research_data/tests/test_micro_rag_and_truth.py``
    (``test_micro_rag_assembles_packet``, subject
    micro-RAG, RETIRE by TF-140;
    ``test_truth_calibrator_flags_unsupported_exclusivity``,
    subject truth calibrator, RETIRE by TF-142) and
    ``research_data/tests/test_memory_and_persona.py``
    (``test_persona_compiler_builds_packet``,
    subject persona compiler, MERGE by TF-141 and
    landed in TF-143;
    ``test_memory_store_search_and_promote``,
    subject the wing memory store + runtime demo
    seed — never adopted, fenced research per
    TF-148).
  * The first file was already archived-in-place
    by TF-144: it opens with the banner marker
    ``# ARCHIVED-IN-PLACE - RETIRED by TF-144
    (2026-11-02)``. The second file carries no
    banner. TF-148 files the first as
    ``archived_retired`` and the second as
    ``fenced_research`` — this slice does NOT
    contradict TF-148: it adds and removes no
    banner, moves and deletes no wing file, and
    does not resurrect ``wyrdforge.models`` (the
    wing package models are still absent by
    design, so the wing tests still cannot run).
  * The adopted production estate carries its
    shipping tests in the root ``tests/`` tree
    (counted by AST): TF-143's
    ``tests/test_tf143_adoptions.py`` carries 41
    tests in four classes — TestBondGraphLanding
    13, TestPersonaCompilerLanding 17,
    TestMergedDelivery 5, TestCrossCutting 6;
    TF-146's ``tests/test_tf146_bond_bridge.py``
    carries 26; TF-147's
    ``tests/test_tf147_runmal_cards.py`` carries
    26. The adopted shipping-test total is 93.
  * The root ``pyproject.toml`` files
    ``testpaths = ["tests"]`` — the main suite
    never collects the wing tests.
  * All six neighbouring checkers stood clean
    live: research_hygiene_problems(),
    wing_retirements_problems(),
    bond_graph_problems(),
    persona_compiler_problems(),
    bond_bridge_problems() and
    runmal_cards_problems().

**What this slice files** (literal data below):

  * ``WING_TEST_ROWS`` — one row per wing test
    FUNCTION (4), each with its subject, its
    subject's verdict/disposition, its final
    TF-149 disposition and its shipping
    counterpart or an explicit none-with-reason:
    the two retired subjects' tests are
    ``archived_with_retired_code`` (banner pinned
    live; their verdict proof lives in TF-140's
    29 and TF-142's 33 verdict tests, named as
    evidence, not as production tests); the
    persona-compiler wing test is
    ``superseded_by_shipping_tests`` (its intent
    — a compile builds a packet carrying identity
    + guidance — is carried in production shape
    by TF-143's shipping tests, named by exact
    test name and checked live; the wing original
    stays fenced in place, unbannered, exactly
    as TF-148 filed it); the memory-store wing
    test is ``fenced_with_unadopted_subject``
    (its subject was never adopted; NO shipping
    counterpart is claimed — the shipping memory
    estate is a different shape; the test stays
    in the wing suite under TF-148's fence).
  * ``ADOPTED_TEST_ROWS`` — one row per adopted
    production module (4: bond_graph,
    persona_compiler, bond_bridge, runmal_cards),
    each with its production path, adopting
    slice, shipping test file, AST-measured test
    count and grade evidence (the module's own
    ``*_problems()`` checker clean live). The
    TF-143 file's 41 tests are attributed without
    double-counting: bond_graph files its 13
    TestBondGraphLanding tests; persona_compiler
    files 28 = TestPersonaCompilerLanding 17 +
    TestMergedDelivery 5 + TestCrossCutting 6
    (the cross-cutting class judges both TF-143
    landings together and is counted exactly
    once, under this row, by the filed counting
    rule in the row itself). Row counts sum to
    the files' AST totals: 13 + 28 + 26 + 26 =
    93.
  * ``GRADE_ROWS`` — the shipping-grade criteria
    every adopted row is judged against: the
    tests live under ``tests/`` and run in the
    full gate; the module checker is clean; the
    module's probes pass; the module's doc
    render is pinned verbatim; the tests are
    deterministic; the fixtures are synthetic
    only. The checker verifies each live where
    it can (testpath fence, neighbour checkers —
    which themselves pin probes and docs).
  * 7 filed probes through the pure validators.

**Docket**: D-WING-1 in
``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``
assigns TF-149 the wing-tests share — the last
share every prior narrowing names. TF-149
closes the entry in the open as
**CLOSED-BY-VERDICT**: the wing was never
repaired into importability and will not be
under this roadmap — its every part received an
executed disposition. TF-016 totals move in the
open (137 entries — 31 OPEN / 1
OPEN-ENVIRONMENT / 101 FIXED / 4
CLOSED-BY-VERDICT; the closing verdict is a
verdict, not a repair, so the FIXED count does
not move). TF-150 / TF-152 own provenance and
the Part VIII gate, not this entry.

The human-readable doc is
``docs/architecture/WING_TESTS.md``, filed
verbatim as this module's render (regenerate
with ``python3 -m thoughtforge.wing_tests``).
Proof grade: FIXTURE/STATIC — filings + the live
tree re-measured on this VM; no live model,
never claimed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

WING_TESTS_FORMAT = "thoughtforge-wing-tests/1"
WING_TESTS_DOC = "docs/architecture/WING_TESTS.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WING_TEST_DIR = _REPO_ROOT / "research_data" / "tests"
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The retirement banner marker TF-144 filed at
#: the head of the retired wing test file.
ARCHIVE_BANNER_MARKER = "# ARCHIVED-IN-PLACE - RETIRED by TF-144 (2026-11-02)"

RETIRED_WING_TEST_FILE = "research_data/tests/test_micro_rag_and_truth.py"
FENCED_WING_TEST_FILE = "research_data/tests/test_memory_and_persona.py"

#: The final dispositions a wing test row may file.
WING_TEST_DISPOSITIONS: tuple[str, ...] = (
    "archived_with_retired_code",
    "superseded_by_shipping_tests",
    "fenced_with_unadopted_subject",
)

#: TF-148's filed dispositions for the two wing
#: test files — TF-149 must not contradict them.
TF148_FILE_DISPOSITIONS: dict[str, str] = {
    RETIRED_WING_TEST_FILE: "archived_retired",
    FENCED_WING_TEST_FILE: "fenced_research",
}


# ── The wing's tests, as literal data ────────────────────────────────────────

WING_TEST_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "micro_rag_packet",
     "file": RETIRED_WING_TEST_FILE,
     "test_name": "test_micro_rag_assembles_packet",
     "subject": "wyrdforge micro-RAG pipeline (assemble())",
     "subject_verdict": "RETIRE by TF-140 (as a retrieval mechanism)",
     "tf148_disposition": "archived_retired",
     "disposition": "archived_with_retired_code",
     "banner_marker": ARCHIVE_BANNER_MARKER,
     "shipping_counterparts": (),
     "counterpart_reason": "no shipping counterpart is claimed: the "
                           "subject mechanism was retired, not adopted; "
                           "its proof is the verdict evidence named here",
     "evidence_file": "tests/test_tf140_micro_rag_verdict.py",
     "evidence_test_count": 29,
     "evidence_note": "TF-140's 29 verdict tests are the measurement "
                      "proof for the RETIRE verdict — evidence, not "
                      "production tests for shipping code"},
    {"id": "truth_exclusivity",
     "file": RETIRED_WING_TEST_FILE,
     "test_name": "test_truth_calibrator_flags_unsupported_exclusivity",
     "subject": "wyrdforge truth calibrator (evaluate())",
     "subject_verdict": "RETIRE by TF-142 (as a calibration / "
                        "enforcement mechanism)",
     "tf148_disposition": "archived_retired",
     "disposition": "archived_with_retired_code",
     "banner_marker": ARCHIVE_BANNER_MARKER,
     "shipping_counterparts": (),
     "counterpart_reason": "no shipping counterpart is claimed: the "
                           "subject mechanism was retired, not adopted; "
                           "its proof is the verdict evidence named here",
     "evidence_file": "tests/test_tf142_truth_calibrator_verdict.py",
     "evidence_test_count": 33,
     "evidence_note": "TF-142's 33 verdict tests are the measurement "
                      "proof for the RETIRE verdict — evidence, not "
                      "production tests for shipping code"},
    {"id": "persona_packet",
     "file": FENCED_WING_TEST_FILE,
     "test_name": "test_persona_compiler_builds_packet",
     "subject": "wyrdforge persona compiler (compile())",
     "subject_verdict": "MERGE by TF-141 (adopt-in-part), landed in "
                        "TF-143 as production code",
     "tf148_disposition": "fenced_research",
     "disposition": "superseded_by_shipping_tests",
     "banner_marker": "",
     "shipping_counterparts": (
         "test_main_counts_move_exactly_as_filed",
         "test_inject_persona_delivers_base_and_compiled"),
     "counterpart_reason": "the wing test's intent — a compile builds "
                           "a packet carrying identity + guidance — is "
                           "carried in production shape by TF-143's "
                           "shipping tests named here (the compiled "
                           "counts pinned exactly, and inject_persona "
                           "delivering base + compiled facets); the "
                           "wing original stays fenced in place, "
                           "unbannered, exactly as TF-148 filed it",
     "evidence_file": "tests/test_tf143_adoptions.py",
     "evidence_test_count": 41,
     "evidence_note": "TF-143's adoption file (41 tests) carries the "
                      "production persona landing the wing test's "
                      "intent moved into"},
    {"id": "memory_store_search",
     "file": FENCED_WING_TEST_FILE,
     "test_name": "test_memory_store_search_and_promote",
     "subject": "wing memory store (InMemoryRecordStore) + runtime "
                "demo seed",
     "subject_verdict": "never adopted — no verdict slice adopted it; "
                        "fenced research per TF-148",
     "tf148_disposition": "fenced_research",
     "disposition": "fenced_with_unadopted_subject",
     "banner_marker": "",
     "shipping_counterparts": (),
     "counterpart_reason": "NO shipping counterpart is claimed: the "
                           "shipping memory estate is a different "
                           "shape and no equivalence is filed; the "
                           "test stays in the wing suite under "
                           "TF-148's fence with its unadopted subject",
     "evidence_file": "",
     "evidence_test_count": 0,
     "evidence_note": "no verdict evidence is claimed — the subject "
                      "was never measured for adoption"},
)


# ── The adopted estate and its shipping tests, as literal data ───────────────

ADOPTED_TEST_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "bond_graph",
     "production_path": "src/thoughtforge/knowledge/bond_graph.py",
     "adopting_slice": "TF-143",
     "shipping_test_file": "tests/test_tf143_adoptions.py",
     "file_test_count": 41,
     "test_count": 13,
     "class_counts": {"TestBondGraphLanding": 13},
     "counting_rule": "this row files exactly its own landing class "
                      "in TF-143's shared adoption file",
     "checker": "bond_graph_problems",
     "checker_module": "thoughtforge.knowledge.bond_graph"},
    {"id": "persona_compiler",
     "production_path": "src/thoughtforge/cognition/persona_compiler.py",
     "adopting_slice": "TF-143",
     "shipping_test_file": "tests/test_tf143_adoptions.py",
     "file_test_count": 41,
     "test_count": 28,
     "class_counts": {"TestPersonaCompilerLanding": 17,
                      "TestMergedDelivery": 5,
                      "TestCrossCutting": 6},
     "counting_rule": "17 landing + 5 merged-delivery + 6 "
                      "cross-cutting; the cross-cutting class judges "
                      "both TF-143 landings together and is counted "
                      "exactly once, under this row, so the two "
                      "TF-143 rows sum to the file's 41 with no "
                      "double-counting",
     "checker": "persona_compiler_problems",
     "checker_module": "thoughtforge.cognition.persona_compiler"},
    {"id": "bond_bridge",
     "production_path": "src/thoughtforge/knowledge/bond_bridge.py",
     "adopting_slice": "TF-146",
     "shipping_test_file": "tests/test_tf146_bond_bridge.py",
     "file_test_count": 26,
     "test_count": 26,
     "class_counts": {"TestBridgeFiled": 4, "TestExportShape": 6,
                      "TestRoundTrip": 3, "TestReservations": 3,
                      "TestBoundaryAndDocket": 5,
                      "TestProbesAndDoc": 5},
     "counting_rule": "the whole file judges this module only",
     "checker": "bond_bridge_problems",
     "checker_module": "thoughtforge.knowledge.bond_bridge"},
    {"id": "runmal_cards",
     "production_path": "src/thoughtforge/cognition/runmal_cards.py",
     "adopting_slice": "TF-147",
     "shipping_test_file": "tests/test_tf147_runmal_cards.py",
     "file_test_count": 26,
     "test_count": 26,
     "class_counts": {"TestCardsFiled": 4, "TestMapping": 7,
                      "TestFiledEmptyAndExtension": 4,
                      "TestRoundTrip": 2, "TestCouplingAndDocket": 4,
                      "TestProbesAndDoc": 5},
     "counting_rule": "the whole file judges this module only",
     "checker": "runmal_cards_problems",
     "checker_module": "thoughtforge.cognition.runmal_cards"},
)

ADOPTED_SHIPPING_TEST_TOTAL = 93


# ── The shipping-grade criteria, as literal data ─────────────────────────────

GRADE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "lives_under_tests_and_runs_in_the_full_gate",
     "criterion": "the shipping tests live under the root tests/ "
                  "tree, which the root pyproject's "
                  "testpaths = [\"tests\"] collects, so they run "
                  "in the campaign's full gate",
     "verified": "the checker re-reads the root testpaths live and "
                 "files every adopted test file under tests/"},
    {"id": "module_checker_clean",
     "criterion": "the adopted module's own *_problems() checker "
                  "is clean live",
     "verified": "the checker calls every adopted module's checker "
                 "(and both Part VIII neighbours) live"},
    {"id": "probes_pass",
     "criterion": "the adopted module's filed probes all pass",
     "verified": "each adopted module's checker pins its own "
                 "probes, so a clean checker carries this leg"},
    {"id": "doc_render_pinned_verbatim",
     "criterion": "the adopted module's architecture doc is filed "
                  "verbatim as its render",
     "verified": "each adopted module's checker pins its own doc "
                 "against its render, so a clean checker carries "
                 "this leg"},
    {"id": "deterministic",
     "criterion": "the shipping tests are deterministic — the "
                  "same tree measures the same counts and verdicts "
                  "on a re-run",
     "verified": "the counts in this filing are AST counts "
                 "recomputed live by the checker; the adopted "
                 "modules' own determinism tests stand in their "
                 "files"},
    {"id": "synthetic_fixtures_only",
     "criterion": "the shipping tests run on synthetic fixtures "
                  "only (TF-017) — no canon / character data",
     "verified": "filed as the estate's standing rule; the "
                 "adopted modules' own fixture tests pin it in "
                 "their files"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def wing_test_row_problems(row: Any) -> list[str]:
    """Judge one wing test filing, purely: file,
    test name, subject, subject verdict, TF-148
    disposition, a disposition in the filed
    vocabulary, and the counterpart discipline of
    that disposition — a retired subject's test
    must file archived, a superseded test must
    name shipping counterparts, a fenced test
    must claim none and say why. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["wing test: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("wing test: the row files no id")
    file = row.get("file")
    if not file or not str(file).startswith("research_data/tests/"):
        problems.append("wing test: the row files no wing test file")
    test_name = row.get("test_name")
    if not test_name or not str(test_name).startswith("test_"):
        problems.append(f"wing test {row.get('id')}: no test name filed")
    if not row.get("subject"):
        problems.append(f"wing test {row.get('id')}: no subject filed")
    if not row.get("subject_verdict"):
        problems.append(
            f"wing test {row.get('id')}: no subject verdict filed")
    if row.get("tf148_disposition") not in (
            "archived_retired", "fenced_research"):
        problems.append(
            f"wing test {row.get('id')}: TF-148 disposition is "
            f"{row.get('tf148_disposition')!r}, not a disposition "
            "TF-148 filed for a wing test file")
    if TF148_FILE_DISPOSITIONS.get(str(file)) is not None \
            and row.get("tf148_disposition") \
            != TF148_FILE_DISPOSITIONS[str(file)]:
        problems.append(
            f"wing test {row.get('id')}: TF-148 disposition "
            f"{row.get('tf148_disposition')!r} contradicts TF-148's "
            f"filing for {file}")
    disposition = row.get("disposition")
    if disposition not in WING_TEST_DISPOSITIONS:
        problems.append(
            f"wing test {row.get('id')}: disposition is "
            f"{disposition!r}, not one of {WING_TEST_DISPOSITIONS}")
    verdict = str(row.get("subject_verdict") or "")
    if "RETIRE" in verdict \
            and disposition != "archived_with_retired_code":
        problems.append(
            f"wing test {row.get('id')}: its subject was retired "
            "but the test is not filed archived_with_retired_code "
            "— retired code's tests are archived with it")
    if disposition == "archived_with_retired_code":
        if not row.get("banner_marker"):
            problems.append(
                f"wing test {row.get('id')}: archived row missing "
                "its banner marker")
        if not row.get("evidence_file") \
                or not row.get("evidence_test_count"):
            problems.append(
                f"wing test {row.get('id')}: archived row missing "
                "its verdict evidence (file + test count)")
    if disposition == "superseded_by_shipping_tests":
        if not row.get("shipping_counterparts"):
            problems.append(
                f"wing test {row.get('id')}: superseded row "
                "missing its shipping counterpart test names — "
                "a superseded claim must name what supersedes it")
    if disposition == "fenced_with_unadopted_subject":
        if row.get("shipping_counterparts"):
            problems.append(
                f"wing test {row.get('id')}: fenced row claims a "
                "shipping counterpart — no equivalence may be "
                "claimed for an unadopted subject")
        if not row.get("counterpart_reason"):
            problems.append(
                f"wing test {row.get('id')}: fenced row files no "
                "reason for its missing counterpart")
    return problems


def adopted_row_problems(row: Any) -> list[str]:
    """Judge one adopted-module filing, purely:
    production path, adopting slice, shipping
    test file, per-class counts that sum to the
    filed test count, a file total at least the
    row count, a counting rule and a checker.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["adopted: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("adopted: the row files no id")
    if not str(row.get("production_path") or "").startswith(
            "src/thoughtforge/"):
        problems.append(
            f"adopted {row.get('id')}: no shipping production "
            "path filed")
    if not row.get("adopting_slice"):
        problems.append(
            f"adopted {row.get('id')}: no adopting slice filed")
    if not str(row.get("shipping_test_file") or "").startswith(
            "tests/"):
        problems.append(
            f"adopted {row.get('id')}: shipping test file is not "
            "under tests/ — shipping-grade tests live in the "
            "gate's tree")
    class_counts = row.get("class_counts")
    if not isinstance(class_counts, dict) or not class_counts:
        problems.append(
            f"adopted {row.get('id')}: no per-class counts filed")
    else:
        summed = sum(class_counts.values())
        if summed != row.get("test_count"):
            problems.append(
                f"adopted {row.get('id')}: test count is "
                f"{row.get('test_count')!r} but its class counts "
                f"sum to {summed} — the filing must reconcile")
    if not isinstance(row.get("file_test_count"), int) \
            or not isinstance(row.get("test_count"), int) \
            or row.get("file_test_count", 0) < row.get("test_count", 0):
        problems.append(
            f"adopted {row.get('id')}: file test count "
            f"{row.get('file_test_count')!r} cannot carry the "
            f"row's {row.get('test_count')!r} tests")
    if not row.get("counting_rule"):
        problems.append(
            f"adopted {row.get('id')}: no counting rule filed")
    if not row.get("checker"):
        problems.append(
            f"adopted {row.get('id')}: no grade-evidence checker "
            "filed")
    return problems


def grade_row_problems(row: Any) -> list[str]:
    """Judge one shipping-grade criterion filing,
    purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["grade: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("grade: the row files no id")
    if not row.get("criterion"):
        problems.append(f"grade {row.get('id')}: no criterion filed")
    if not row.get("verified"):
        problems.append(
            f"grade {row.get('id')}: no verification filed")
    return problems


def wing_function_set_problems(filed: Any, measured: Any) -> list[str]:
    """Judge a filed wing test-function set against
    a measured one, purely: a function present in
    the tree but unfiled is named, and a filed
    function missing from the tree is named.
    Empty-safe."""
    if not isinstance(filed, (list, tuple, set)) or not filed:
        return ["wing functions: the filed set is empty"]
    if not isinstance(measured, (list, tuple, set)) or not measured:
        return ["wing functions: the measured set is empty"]
    problems: list[str] = []
    filed_set, measured_set = set(filed), set(measured)
    for name in sorted(measured_set - filed_set):
        problems.append(
            f"wing functions: {name} is present in the tree but "
            "unfiled — every wing test function files a disposition")
    for name in sorted(filed_set - measured_set):
        problems.append(
            f"wing functions: {name} is filed but missing from "
            "the tree")
    return problems


def wing_tests_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured wing-tests state against
    its expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["wing tests state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["wing tests state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"wing tests state: {key} measures "
                f"{state.get(key)!r}, not the filed {value!r}")
    return problems


def doc_text_problems(text: Any, render: Any, label: Any) -> list[str]:
    """Judge a filed doc against its render,
    purely: any drift is named. Empty-safe."""
    if not isinstance(render, str) or not render:
        return ["doc: the render is empty"]
    if not isinstance(text, str) or not text:
        return [f"doc: the {label} text is empty"]
    if text != render:
        return [f"doc: the {label} has drifted from its render"]
    return []


# ── Live measurement helpers ─────────────────────────────────────────────────

def _read_rel(rel: str) -> str | None:
    path = _REPO_ROOT / rel
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _ast_test_functions(rel: str) -> list[str]:
    """Every test function name in a file, by AST
    (top-level and inside classes)."""
    text = _read_rel(rel)
    if text is None:
        return []
    tree = ast.parse(text)
    return [node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")]


def _ast_class_test_counts(rel: str) -> dict[str, int]:
    """Per-class test-function counts in a file, by
    AST (classes at module level)."""
    text = _read_rel(rel)
    if text is None:
        return {}
    tree = ast.parse(text)
    counts: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            counts[node.name] = sum(
                1 for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name.startswith("test"))
    return counts


# ── The live wing-tests state ────────────────────────────────────────────────

WING_TESTS_EXPECTED: dict[str, Any] = {
    "wing_test_rows": 4,
    "wing_test_files": 2,
    "wing_test_functions": 4,
    "archived_with_retired_code": 2,
    "superseded_by_shipping_tests": 1,
    "fenced_with_unadopted_subject": 1,
    "bannered_wing_test_files": 1,
    "unbannered_wing_test_files": 1,
    "adopted_rows": 4,
    "adopted_shipping_tests": 93,
    "tf143_file_tests": 41,
    "tf146_file_tests": 26,
    "tf147_file_tests": 26,
    "verdict_evidence_tests": 62,
    "grade_rows": 6,
    "models_package_present": False,
    "main_suite_collects_wing_tests": False,
}


def run_wing_tests_state() -> dict[str, Any]:
    """Measure the wing tests' disposition on the
    live tree: wing files + AST function sets,
    banner placement, the adopted estate's AST
    counts and the verdict-evidence counts.
    Deterministic: counts and booleans only."""
    wing_files = sorted({row["file"] for row in WING_TEST_ROWS})
    texts = {rel: _read_rel(rel) or "" for rel in wing_files}
    adopted_files: dict[str, int] = {}
    for row in ADOPTED_TEST_ROWS:
        adopted_files[row["shipping_test_file"]] = len(
            _ast_test_functions(row["shipping_test_file"]))
    try:
        import tomllib

        root_data = tomllib.loads(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        testpaths = root_data["tool"]["pytest"]["ini_options"][
            "testpaths"]
    except Exception:
        testpaths = []
    return {
        "wing_test_rows": len(WING_TEST_ROWS),
        "wing_test_files": len(wing_files),
        "wing_test_functions": sum(
            len(_ast_test_functions(rel)) for rel in wing_files),
        "archived_with_retired_code": sum(
            1 for r in WING_TEST_ROWS
            if r["disposition"] == "archived_with_retired_code"),
        "superseded_by_shipping_tests": sum(
            1 for r in WING_TEST_ROWS
            if r["disposition"] == "superseded_by_shipping_tests"),
        "fenced_with_unadopted_subject": sum(
            1 for r in WING_TEST_ROWS
            if r["disposition"] == "fenced_with_unadopted_subject"),
        "bannered_wing_test_files": sum(
            1 for rel in wing_files
            if texts[rel].startswith(ARCHIVE_BANNER_MARKER)),
        "unbannered_wing_test_files": sum(
            1 for rel in wing_files
            if ARCHIVE_BANNER_MARKER not in texts[rel]),
        "adopted_rows": len(ADOPTED_TEST_ROWS),
        "adopted_shipping_tests": sum(
            r["test_count"] for r in ADOPTED_TEST_ROWS),
        "tf143_file_tests": adopted_files.get(
            "tests/test_tf143_adoptions.py", 0),
        "tf146_file_tests": adopted_files.get(
            "tests/test_tf146_bond_bridge.py", 0),
        "tf147_file_tests": adopted_files.get(
            "tests/test_tf147_runmal_cards.py", 0),
        "verdict_evidence_tests": sum(
            len(_ast_test_functions(r["evidence_file"]))
            for r in WING_TEST_ROWS
            if r["disposition"] == "archived_with_retired_code"),
        "grade_rows": len(GRADE_ROWS),
        "models_package_present": (_WING / "models").exists(),
        "main_suite_collects_wing_tests": testpaths != ["tests"],
    }


# ── The filed probes ─────────────────────────────────────────────────────────

WING_TEST_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_estate_passes",
     "kind": "clean",
     "detail": "the filed wing rows + adopted rows + grade rows "
               "pass their validators"},
    {"id": "retired_test_not_archived_is_named",
     "kind": "retired_not_archived",
     "detail": "a retired subject's test filed under a "
               "non-archived disposition is named"},
    {"id": "superseded_test_without_a_counterpart_is_named",
     "kind": "superseded_no_counterpart",
     "detail": "a superseded row with its shipping counterpart "
               "names emptied is named"},
    {"id": "adopted_row_test_count_wrong_is_named",
     "kind": "adopted_count_wrong",
     "detail": "an adopted row whose filed count disagrees with "
               "its class counts is named"},
    {"id": "unknown_disposition_is_named",
     "kind": "unknown_disposition",
     "detail": "a wing row filing a disposition outside the "
               "filed vocabulary is named"},
    {"id": "unfiled_wing_function_is_named",
     "kind": "unfiled_function",
     "detail": "a wing test function present in the tree but "
               "unfiled (or filed but missing) is named"},
    {"id": "doc_drift_is_named",
     "kind": "doc_drift",
     "detail": "a doc text that drifts from the filed render is "
               "named"},
)


def run_wing_tests_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in WING_TEST_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in WING_TEST_ROWS:
                problems += wing_test_row_problems(row)
            for row in ADOPTED_TEST_ROWS:
                problems += adopted_row_problems(row)
            for row in GRADE_ROWS:
                problems += grade_row_problems(row)
            passed = problems == []
        elif kind == "retired_not_archived":
            row = dict(WING_TEST_ROWS[0],
                       disposition="fenced_with_unadopted_subject")
            passed = bool(wing_test_row_problems(row))
        elif kind == "superseded_no_counterpart":
            row = dict(WING_TEST_ROWS[2], shipping_counterparts=())
            passed = bool(wing_test_row_problems(row))
        elif kind == "adopted_count_wrong":
            row = dict(ADOPTED_TEST_ROWS[0],
                       test_count=ADOPTED_TEST_ROWS[0]["test_count"] + 1)
            passed = bool(adopted_row_problems(row))
        elif kind == "unknown_disposition":
            row = dict(WING_TEST_ROWS[0], disposition="limbo")
            passed = bool(wing_test_row_problems(row))
        elif kind == "unfiled_function":
            filed = [r["test_name"] for r in WING_TEST_ROWS]
            measured = filed + ["test_unfiled_extra"]
            named_extra = bool(wing_function_set_problems(
                filed, measured))
            named_missing = bool(wing_function_set_problems(
                filed, filed[:-1]))
            passed = named_extra and named_missing
        elif kind == "doc_drift":
            drifted = render_wing_tests_markdown() + "drift\n"
            passed = bool(doc_text_problems(
                drifted, render_wing_tests_markdown(), "wing tests doc"))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the doc, verbatim) ──────────────────────────────────

def render_wing_tests_markdown() -> str:
    """Render the wing-tests doc deterministically
    — the exact text filed as
    docs/architecture/WING_TESTS.md."""
    state = run_wing_tests_state()
    lines: list[str] = []
    lines.append("# The Wing's Tests — Adopted Code Ships Tested, "
                 "Retired Code's Tests Archived (TF-149)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/wing_tests.py` — the filed doc IS "
        "the render: regenerate with "
        "`python3 -m thoughtforge.wing_tests`. The checker "
        "(`wing_tests_problems()`) re-derives every claim live "
        "from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{WING_TESTS_FORMAT}`")
    lines.append(
        f"Wing test functions filed: {state['wing_test_functions']} "
        f"in {state['wing_test_files']} files (archived with "
        f"retired code {state['archived_with_retired_code']} / "
        f"superseded by shipping tests "
        f"{state['superseded_by_shipping_tests']} / fenced with "
        f"unadopted subject {state['fenced_with_unadopted_subject']})")
    lines.append(
        f"Adopted shipping tests: {state['adopted_shipping_tests']} "
        f"across {state['adopted_rows']} adopted production modules")
    lines.append(f"Probes: {len(WING_TEST_PROBES)}/"
                 f"{len(WING_TEST_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: "
        "the wing carries exactly 2 test files with exactly 4 "
        "test functions (counted by AST). "
        "`test_micro_rag_and_truth.py` was already "
        "archived-in-place by TF-144 — it opens with the "
        "retirement banner; `test_memory_and_persona.py` "
        "carries no banner. TF-148 files the first "
        "archived_retired and the second fenced_research; "
        "this slice contradicts neither filing — it adds and "
        "removes no banner, moves and deletes no wing file, "
        "and does not resurrect `wyrdforge.models` (still "
        "absent by design, so the wing tests still cannot "
        "run). The adopted production estate carries 93 "
        "shipping tests in the root `tests/` tree (TF-143's "
        "file 41, TF-146's 26, TF-147's 26, counted by AST), "
        "and the root pyproject's testpaths = [\"tests\"] "
        "means the main suite never collects the wing tests. "
        "All six neighbouring checkers stood clean live.")
    lines.append("")
    lines.append("## 2. Every wing test function, filed")
    lines.append("")
    lines.append("| Wing test | Subject | Subject verdict | "
                 "TF-149 disposition | Counterpart / evidence |")
    lines.append("|---|---|---|---|---|")
    for row in WING_TEST_ROWS:
        if row["shipping_counterparts"]:
            counterpart = "shipping tests: " + ", ".join(
                f"`{name}`" for name in row["shipping_counterparts"])
        elif row["evidence_file"]:
            counterpart = (f"verdict evidence: "
                           f"`{row['evidence_file']}` "
                           f"({row['evidence_test_count']} tests)")
        else:
            counterpart = "none — no counterpart claimed"
        lines.append(
            f"| `{row['test_name']}` (`{row['file']}`) | "
            f"{row['subject']} | {row['subject_verdict']} | "
            f"{row['disposition']} | {counterpart} |")
    lines.append("")
    for row in WING_TEST_ROWS:
        lines.append(f"- `{row['test_name']}`: "
                     f"{row['counterpart_reason']}.")
    lines.append("")
    lines.append("## 3. The adopted estate's shipping tests")
    lines.append("")
    lines.append("| Production module | Adopting slice | "
                 "Shipping test file | Tests | Grade evidence |")
    lines.append("|---|---|---|---|---|")
    for row in ADOPTED_TEST_ROWS:
        classes = " + ".join(
            f"{name} {count}"
            for name, count in row["class_counts"].items())
        lines.append(
            f"| `{row['production_path']}` | "
            f"{row['adopting_slice']} | "
            f"`{row['shipping_test_file']}` | {row['test_count']} "
            f"({classes}) | `{row['checker']}()` clean live |")
    lines.append("")
    lines.append(
        "Counting rules, filed in the rows themselves: "
        + "; ".join(
            f"`{r['id']}` — {r['counting_rule']}"
            for r in ADOPTED_TEST_ROWS)
        + f". The rows sum to {state['adopted_shipping_tests']}, "
        "recomputed live by AST in the checker (the TF-143 "
        "file's 41 tests are attributed across its two rows "
        "with no class counted twice).")
    lines.append("")
    lines.append("## 4. Shipping-grade criteria")
    lines.append("")
    for row in GRADE_ROWS:
        lines.append(
            f"- `{row['id']}`: {row['criterion']} — Verified: "
            f"{row['verified']}.")
    lines.append("")
    lines.append("## 5. Docket")
    lines.append("")
    lines.append(
        "TF-149 closes D-WING-1 in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md` as "
        "**CLOSED-BY-VERDICT** — the wing-tests share was the "
        "last share every prior narrowing named. The verdict: "
        "the wing was never repaired into importability and "
        "will not be under this roadmap; its every part "
        "received an executed disposition (adopted concepts "
        "rebuilt as shipping production code carrying 93 "
        "shipping-grade tests; retired mechanisms and their "
        "shared test file archived-in-place by TF-144; the "
        "remaining wing test file fenced with its unadopted "
        "subject per TF-148, its persona test's intent "
        "superseded by TF-143's shipping tests). TF-016 "
        "totals move in the open: 137 entries — 31 OPEN / 1 "
        "OPEN-ENVIRONMENT / 101 FIXED / 4 CLOSED-BY-VERDICT. "
        "TF-150 / TF-152 own provenance and the Part VIII "
        "gate, not this entry.")
    lines.append("")
    lines.append("## 6. The probes")
    lines.append("")
    for probe in WING_TEST_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 7. Filed limits")
    lines.append("")
    lines.append(
        "- Fixture/static grade only: this slice files the "
        "wing tests' dispositions and re-measures the adopted "
        "estate's shipping tests by AST; the wing tests "
        "themselves still cannot run — the wing still cannot "
        "import by design, exactly as TF-148 filed it.")
    lines.append(
        "- No wing file was moved, deleted or bannered by "
        "this slice, and no README.md anywhere was touched; "
        "TF-150 owns the wing's provenance and TF-152 the "
        "Part VIII gate.")
    lines.append(
        "- The fenced memory-store test claims NO shipping "
        "counterpart: the shipping memory estate is a "
        "different shape, and no equivalence is filed.")
    lines.append(
        "- No live model was read; live behaviour is NOT "
        "PROVEN (TF-173 class), never claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_wing_tests_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / WING_TESTS_DOC
    doc.write_text(render_wing_tests_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.wing_tests`."""
    doc = write_wing_tests_doc()
    print(f"wing tests doc regenerated: -> {WING_TESTS_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def _docket_closure_problems() -> list[str]:
    """Judge D-WING-1's closure in the docket, live:
    the header totals moved as filed and the entry
    carries TF-149's closing paragraph with every
    prior narrowing's text preserved."""
    problems: list[str] = []
    text = _DOCKET.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)
    if "137 entries — 31 OPEN, 1 OPEN-ENVIRONMENT, 101 FIXED" \
            not in normalized:
        problems.append(
            "docket: the header totals do not file 137 entries "
            "— 31 OPEN / 1 OPEN-ENVIRONMENT / 101 FIXED")
    if "4 CLOSED-BY-VERDICT (D-CARD-2 by TF-086's Part IV gate " \
            "verdict; D-WING-1 by TF-149)" not in normalized:
        problems.append(
            "docket: the header does not file 4 CLOSED-BY-VERDICT "
            "with D-WING-1 named beside D-CARD-2")
    wing_line = next(
        (line for line in text.splitlines()
         if line.startswith("| D-WING-1 |")), "")
    if not wing_line:
        problems.append("docket: the D-WING-1 entry is missing")
        return problems
    for marker in ("narrowed by TF-139", "Narrowed by TF-148",
                   "stays OPEN for the TF-149"):
        if marker not in wing_line:
            problems.append(
                f"docket: D-WING-1 no longer carries the filed "
                f"text {marker!r} — prior narrowings stand "
                "byte-for-byte")
    if "**Closed by TF-149**" not in wing_line:
        problems.append(
            "docket: D-WING-1 does not file TF-149's closing")
    if "**CLOSED-BY-VERDICT**" not in wing_line:
        problems.append(
            "docket: D-WING-1 is not closed CLOSED-BY-VERDICT")
    return problems


def wing_tests_problems() -> list[str]:
    """Re-derive TF-149's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    row_ids = [r["id"] for r in WING_TEST_ROWS]
    if len(row_ids) != len(set(row_ids)) or len(WING_TEST_ROWS) != 4:
        problems.append("the WING_TEST_ROWS filing is not four "
                        "unique wing tests")
    for row in WING_TEST_ROWS:
        problems.extend(wing_test_row_problems(row))
    adopted_ids = [r["id"] for r in ADOPTED_TEST_ROWS]
    if len(adopted_ids) != len(set(adopted_ids)) \
            or len(ADOPTED_TEST_ROWS) != 4:
        problems.append("the ADOPTED_TEST_ROWS filing is not four "
                        "unique adopted modules")
    for row in ADOPTED_TEST_ROWS:
        problems.extend(adopted_row_problems(row))
    if sum(r["test_count"] for r in ADOPTED_TEST_ROWS) \
            != ADOPTED_SHIPPING_TEST_TOTAL:
        problems.append(
            "adopted: the rows do not sum to the filed shipping "
            f"total {ADOPTED_SHIPPING_TEST_TOTAL}")
    for row in GRADE_ROWS:
        problems.extend(grade_row_problems(row))

    # Wing files, live: AST sets match the filing
    # exactly per file; banners exactly as filed.
    for rel in sorted({r["file"] for r in WING_TEST_ROWS}):
        filed = [r["test_name"] for r in WING_TEST_ROWS
                 if r["file"] == rel]
        problems.extend(
            f"wing {rel}: {p}" for p in wing_function_set_problems(
                filed, _ast_test_functions(rel)))
        text = _read_rel(rel)
        if text is None:
            problems.append(f"wing: {rel} is missing")
            continue
        filed_banner = any(
            r["banner_marker"] for r in WING_TEST_ROWS
            if r["file"] == rel)
        if filed_banner and not text.startswith(ARCHIVE_BANNER_MARKER):
            problems.append(
                f"wing: {rel} no longer opens with its filed "
                "TF-144 banner — retired code's tests stay "
                "archived with it")
        if not filed_banner and ARCHIVE_BANNER_MARKER in text:
            problems.append(
                f"wing: {rel} carries a retirement banner but "
                "files none — TF-149 adds no banner")

    # TF-148's filing stands uncontradicted.
    from thoughtforge.research_hygiene import FILE_ROWS as TF148_FILES

    tf148 = {r["path"]: r["disposition"] for r in TF148_FILES}
    for rel, disposition in TF148_FILE_DISPOSITIONS.items():
        if tf148.get(rel) != disposition:
            problems.append(
                f"tf148: {rel} files {tf148.get(rel)!r} in "
                f"TF-148, not the {disposition!r} this slice "
                "pins")

    # Adopted estate, live: files exist, AST class
    # counts reconcile, no class counted twice,
    # file totals match.
    counted_classes: set[tuple[str, str]] = set()
    for row in ADOPTED_TEST_ROWS:
        if not (_REPO_ROOT / row["production_path"]).exists():
            problems.append(
                f"adopted: {row['production_path']} is missing")
        measured = _ast_class_test_counts(row["shipping_test_file"])
        for class_name, count in row["class_counts"].items():
            key = (row["shipping_test_file"], class_name)
            if key in counted_classes:
                problems.append(
                    f"adopted: {class_name} in "
                    f"{row['shipping_test_file']} is counted "
                    "twice across the adopted rows")
            counted_classes.add(key)
            if measured.get(class_name) != count:
                problems.append(
                    f"adopted {row['id']}: class {class_name} "
                    f"measures {measured.get(class_name)!r} "
                    f"tests, not the filed {count}")
        file_total = len(_ast_test_functions(
            row["shipping_test_file"]))
        if file_total != row["file_test_count"]:
            problems.append(
                f"adopted {row['id']}: "
                f"{row['shipping_test_file']} measures "
                f"{file_total} tests, not the filed file total "
                f"{row['file_test_count']}")

    # The superseded row's counterpart names exist
    # live in the shipping test file.
    tf143_names = set(_ast_test_functions(
        "tests/test_tf143_adoptions.py"))
    for row in WING_TEST_ROWS:
        for name in row["shipping_counterparts"]:
            if name not in tf143_names:
                problems.append(
                    f"counterpart: {name} is not a test in "
                    "tests/test_tf143_adoptions.py — a "
                    "superseded claim must name a live test")
        if row["disposition"] == "archived_with_retired_code":
            measured = len(_ast_test_functions(row["evidence_file"]))
            if measured != row["evidence_test_count"]:
                problems.append(
                    f"evidence: {row['evidence_file']} measures "
                    f"{measured} tests, not the filed "
                    f"{row['evidence_test_count']}")

    # The measured state matches its filing.
    problems.extend(wing_tests_state_problems(
        run_wing_tests_state(), WING_TESTS_EXPECTED))

    # Neighbouring checkers clean live (each adopted
    # module's grade evidence + the two Part VIII
    # neighbours).
    from thoughtforge.cognition.persona_compiler import (
        persona_compiler_problems,
    )
    from thoughtforge.cognition.runmal_cards import (
        runmal_cards_problems,
    )
    from thoughtforge.knowledge.bond_bridge import (
        bond_bridge_problems,
    )
    from thoughtforge.knowledge.bond_graph import bond_graph_problems
    from thoughtforge.research_hygiene import research_hygiene_problems
    from thoughtforge.wing_retirements import wing_retirements_problems

    for name, checker in (
            ("research_hygiene", research_hygiene_problems),
            ("wing_retirements", wing_retirements_problems),
            ("bond_graph", bond_graph_problems),
            ("persona_compiler", persona_compiler_problems),
            ("bond_bridge", bond_bridge_problems),
            ("runmal_cards", runmal_cards_problems)):
        neighbour = checker()
        if neighbour:
            problems.append(
                f"neighbour: {name}_problems() is not clean: "
                f"{neighbour[:3]}")

    # The docket carries this slice's closure.
    problems.extend(_docket_closure_problems())

    # The probes all behave.
    probes = run_wing_tests_probes()
    if len(probes) != len(WING_TEST_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the wing tests probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The filed doc matches its render verbatim.
    doc = _REPO_ROOT / WING_TESTS_DOC
    if not doc.exists():
        problems.append(f"the wing tests doc {WING_TESTS_DOC} is missing")
    else:
        problems.extend(
            f"doc: {p}" for p in doc_text_problems(
                doc.read_text(encoding="utf-8"),
                render_wing_tests_markdown(), "wing tests doc"))
    return problems


def wing_tests_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_wing_tests_probes()
    return {
        "format": WING_TESTS_FORMAT,
        "doc": WING_TESTS_DOC,
        "wing_tests": [dict(r) for r in WING_TEST_ROWS],
        "adopted": [dict(r) for r in ADOPTED_TEST_ROWS],
        "grade": [dict(r) for r in GRADE_ROWS],
        "state": run_wing_tests_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "ADOPTED_SHIPPING_TEST_TOTAL",
    "ADOPTED_TEST_ROWS",
    "ARCHIVE_BANNER_MARKER",
    "FENCED_WING_TEST_FILE",
    "GRADE_ROWS",
    "RETIRED_WING_TEST_FILE",
    "TF148_FILE_DISPOSITIONS",
    "WING_TESTS_DOC",
    "WING_TESTS_EXPECTED",
    "WING_TESTS_FORMAT",
    "WING_TEST_DISPOSITIONS",
    "WING_TEST_PROBES",
    "WING_TEST_ROWS",
    "adopted_row_problems",
    "doc_text_problems",
    "grade_row_problems",
    "main",
    "render_wing_tests_markdown",
    "run_wing_tests_probes",
    "run_wing_tests_state",
    "wing_function_set_problems",
    "wing_test_row_problems",
    "wing_tests_as_data",
    "wing_tests_problems",
    "wing_tests_state_problems",
    "write_wing_tests_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
