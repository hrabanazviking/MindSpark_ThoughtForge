"""
TF-148 — research hygiene for ``research_data/``.

TF-148's roadmap text: *Research hygiene:
``research_data/`` either graduates, archives, or
is clearly fenced as research with its own test
discipline — no ambiguous middle state remains.*

The measured start, probed live on this tree
before anything was built:

  * ``git ls-files research_data`` files 96 tracked
    files in twelve areas: top-level markdown 58,
    schemas 14, services 5, wyrd_runtime packets 5,
    config 4, examples 2, security 2, wing tests 2,
    runtime 1, scripts 1, the package ``__init__`` 1
    and the wing ``pyproject.toml`` 1. This clone is
    sparse: config, examples, scripts and
    wyrd_runtime are tracked but NOT on disk here
    (12 files) — TF-006's census used ``git
    ls-tree`` for the same reason, so this module
    counts disk areas for the sparse-present areas
    and judges the sparse-excluded areas only
    through a pure validator fed a tracked-file
    list.
  * The wing still cannot import:
    ``research_data/src/wyrdforge/models`` does not
    exist and the package ``__init__`` re-exports
    ten names from that missing package. TF-139's
    verdict checker reads that ``__init__`` live
    and requires those imports to remain — this
    slice does not remove or alter them; it files
    the broken-import state as the fence state.
  * Dispositions already executed before TF-148:
    the bond graph service and the persona
    compiler service are graduated sources
    (TF-143 landed production counterparts in the
    shipping tree; TF-144 pins that their wing
    files carry no retirement banner); the
    micro-RAG pipeline service, the truth
    calibrator service and their shared wing test
    file are archived-in-place retired by TF-144
    (a retirement banner at each head). Five code
    or test files had never been given any
    disposition at all — the wing memory store,
    the two security guards, the runtime demo seed
    and the memory/persona wing test file. That
    ambiguous middle is what TF-148 ends.
  * Fences already in force: the root
    ``pyproject.toml`` files
    ``testpaths = ["tests"]`` so the campaign
    suite never collects the wing's tests, and
    ``research_data/pyproject.toml`` is the wing's
    own package (wyrdforge 0.1.0, its own
    pythonpath and testpaths) — running pytest
    inside ``research_data/`` errors at collection
    (2 errors, 0 run, TF-006), a filed state this
    slice names rather than hides.

**What this slice files** (literal data below):

  * ``AREA_ROWS`` — every tracked area of
    ``research_data/``, each with its exact count,
    its disposition and its rule. Design and spec
    markdown and the wyrd_runtime packets are
    fenced reference; the schemas are fenced
    evidence (verdict evidence, kept unmarked per
    TF-144); config and examples are fenced
    examples; the scripts file is fenced tooling
    (a research-side generator, not the shipping
    build); the wing pyproject is the fence
    marker (its own package, its own test
    discipline); the code and test areas are
    split by file under ``FILE_ROWS``. A thirteenth
    row files this slice's own new fence notice,
    ``research_data/RESEARCH_FENCE.md`` (1 file),
    so the tracked census is 96 baseline files
    plus the fence file — 97 once banked.
  * ``FILE_ROWS`` — every code or test file under
    ``research_data/src`` and
    ``research_data/tests`` (11 files), each with
    exactly one disposition: graduated_source
    (2, naming their production counterparts),
    archived_retired (3, their TF-144 banner
    marker pinned live), fenced_research (3 —
    they import the missing models package,
    cannot run and are not shipping),
    fenced_standalone (2 — the security guards,
    the only wing code importable standalone
    because they import nothing from the missing
    models package; still research, not adopted),
    and fence_import (1 — the package
    ``__init__``, filed as the broken-import
    fence state).
  * ``TEST_DISCIPLINE_ROWS`` — the test
    discipline, filed in four rows: the main
    suite, the wing suite, graduated tests and
    retired tests. TF-149 owns the wing tests'
    final disposition; this slice explicitly does
    NOT discharge it.
  * ``FENCE_RULES`` — the fence in force: no
    shipping file imports the wing package (the
    checker judges real import statements, parsed
    — filings that mention the wing in prose,
    docstrings or string data are not imports), no
    research file is shipping code, revival of
    any fenced or archived file needs a new
    roadmap word, schemas stay as evidence,
    README files are never touched, and the
    missing models package is not resurrected
    here (TF-149 / a new roadmap word own any
    change to that state).

**Docket**: D-WING-1 in
``docs/audit/TF-016_BUG_AND_GAP_DOCKET.md``
assigns TF-148 the research-hygiene share and
TF-149 the wing's tests. TF-148 narrows the
entry in the open for the hygiene share ONLY;
the entry stays OPEN for TF-149 and TF-016
totals do not move (137 entries — 33 OPEN / 1
OPEN-ENVIRONMENT / 100 FIXED / 3
CLOSED-BY-VERDICT). No new docket entry is
added: this slice's probes found no new defect.

The human-readable hygiene doc is
``docs/architecture/RESEARCH_HYGIENE.md`` and
the fence notice a reader landing in
``research_data/`` sees is
``research_data/RESEARCH_FENCE.md``; both are
filed verbatim as this module's renders
(regenerate both with
``python3 -m thoughtforge.research_hygiene``).
Proof grade: FIXTURE/STATIC — filings + the live
tree re-measured on this VM; no live model,
never claimed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

RESEARCH_HYGIENE_FORMAT = "thoughtforge-research-hygiene/1"
RESEARCH_HYGIENE_DOC = "docs/architecture/RESEARCH_HYGIENE.md"
RESEARCH_FENCE_DOC = "research_data/RESEARCH_FENCE.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH = _REPO_ROOT / "research_data"
_WING = _RESEARCH / "src" / "wyrdforge"
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The retirement banner marker TF-144 filed at
#: the head of every archived file.
ARCHIVE_BANNER_MARKER = "# ARCHIVED-IN-PLACE - RETIRED by TF-144 (2026-11-02)"

#: The dispositions an AREA row may file.
AREA_DISPOSITIONS: tuple[str, ...] = (
    "fenced_reference",
    "fenced_evidence",
    "fenced_examples",
    "fenced_tooling",
    "fence_marker",
    "split_by_file",
)

#: The dispositions a FILE row may file — every
#: code/test file gets exactly one.
FILE_DISPOSITIONS: tuple[str, ...] = (
    "graduated_source",
    "archived_retired",
    "fenced_research",
    "fenced_standalone",
    "fence_import",
)

BASELINE_TRACKED_FILES = 96
FENCE_TRACKED_FILES = 1


# ── Areas, as literal data ───────────────────────────────────────────────────

AREA_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "top_level_markdown",
     "area": "research_data/*.md (top-level design and spec markdown)",
     "pattern": "research_data/*.md",
     "tracked_count": 58,
     "disposition": "fenced_reference",
     "rule": "design and spec provenance, read as reference; "
             "not shipping documentation and not code",
     "sparse_present": True},
    {"id": "schemas",
     "area": "research_data/src/wyrdforge/schemas/*.json",
     "pattern": "research_data/src/wyrdforge/schemas/*",
     "tracked_count": 14,
     "disposition": "fenced_evidence",
     "rule": "verdict evidence, kept unmarked per TF-144; the "
             "shapes execute nothing and the verdict spikes are "
             "checked against them live",
     "sparse_present": True},
    {"id": "services",
     "area": "research_data/src/wyrdforge/services/*.py",
     "pattern": "research_data/src/wyrdforge/services/*",
     "tracked_count": 5,
     "disposition": "split_by_file",
     "rule": "split by FILE_ROWS: each service files its own "
             "graduated, archived or fenced disposition",
     "sparse_present": True},
    {"id": "wyrd_runtime_packets",
     "area": "research_data/wyrd_runtime/*.md",
     "pattern": "research_data/wyrd_runtime/*",
     "tracked_count": 5,
     "disposition": "fenced_reference",
     "rule": "runtime packet provenance, read as reference; "
             "tracked but not on disk in this sparse clone, so "
             "judged only through the tracked-file validator",
     "sparse_present": False},
    {"id": "config_files",
     "area": "research_data/config/*.yaml",
     "pattern": "research_data/config/*",
     "tracked_count": 4,
     "disposition": "fenced_examples",
     "rule": "example configuration for the research wing only; "
             "tracked but not on disk in this sparse clone, so "
             "judged only through the tracked-file validator",
     "sparse_present": False},
    {"id": "examples_files",
     "area": "research_data/examples/*.yaml",
     "pattern": "research_data/examples/*",
     "tracked_count": 2,
     "disposition": "fenced_examples",
     "rule": "example data for the research wing only; tracked "
             "but not on disk in this sparse clone, so judged "
             "only through the tracked-file validator",
     "sparse_present": False},
    {"id": "security",
     "area": "research_data/src/wyrdforge/security/*.py",
     "pattern": "research_data/src/wyrdforge/security/*",
     "tracked_count": 2,
     "disposition": "split_by_file",
     "rule": "split by FILE_ROWS: both guards file "
             "fenced_standalone",
     "sparse_present": True},
    {"id": "wing_tests",
     "area": "research_data/tests/*.py",
     "pattern": "research_data/tests/*",
     "tracked_count": 2,
     "disposition": "split_by_file",
     "rule": "split by FILE_ROWS and judged under the wing's "
             "own test discipline; TF-149 owns their final "
             "disposition",
     "sparse_present": True},
    {"id": "runtime",
     "area": "research_data/src/wyrdforge/runtime/*.py",
     "pattern": "research_data/src/wyrdforge/runtime/*",
     "tracked_count": 1,
     "disposition": "split_by_file",
     "rule": "split by FILE_ROWS: the demo seed files "
             "fenced_research",
     "sparse_present": True},
    {"id": "scripts_files",
     "area": "research_data/scripts/*.py",
     "pattern": "research_data/scripts/*",
     "tracked_count": 1,
     "disposition": "fenced_tooling",
     "rule": "generate_json_schemas.py is a research-side "
             "generator, not the shipping build; tracked but "
             "not on disk in this sparse clone, so judged only "
             "through the tracked-file validator",
     "sparse_present": False},
    {"id": "package_init",
     "area": "research_data/src/wyrdforge/__init__.py",
     "pattern": "research_data/src/wyrdforge/__init__.py",
     "tracked_count": 1,
     "disposition": "split_by_file",
     "rule": "split by FILE_ROWS: the package init files "
             "fence_import, the broken-import fence state",
     "sparse_present": True},
    {"id": "wing_pyproject",
     "area": "research_data/pyproject.toml",
     "pattern": "research_data/pyproject.toml",
     "tracked_count": 1,
     "disposition": "fence_marker",
     "rule": "the fence marker itself: the wing is its own "
             "package (wyrdforge 0.1.0) with its own pythonpath "
             "and testpaths — its own test discipline",
     "sparse_present": True},
    {"id": "research_fence_file",
     "area": "research_data/RESEARCH_FENCE.md",
     "pattern": "research_data/RESEARCH_FENCE.md",
     "tracked_count": 1,
     "disposition": "fence_marker",
     "rule": "this slice's fence notice, filed verbatim as "
             "render_research_fence_markdown(); the tracked "
             "census is the 96 baseline files plus this file",
     "sparse_present": True},
)


def classify_tracked_path(path: Any) -> str | None:
    """Classify one tracked ``research_data`` path
    into its filed area, purely. Returns None for
    anything unclassified — the validator names it."""
    if not isinstance(path, str) or not path:
        return None
    if path == RESEARCH_FENCE_DOC:
        return "research_fence_file"
    if not path.startswith("research_data/"):
        return None
    rel = path[len("research_data/"):]
    if "/" not in rel:
        if rel.endswith(".md"):
            return "top_level_markdown"
        if rel == "pyproject.toml":
            return "wing_pyproject"
        return None
    if rel.startswith("wyrd_runtime/") and rel.count("/") == 1 \
            and rel.endswith(".md"):
        return "wyrd_runtime_packets"
    if rel.startswith("config/") and rel.count("/") == 1:
        return "config_files"
    if rel.startswith("examples/") and rel.count("/") == 1:
        return "examples_files"
    if rel.startswith("scripts/") and rel.count("/") == 1:
        return "scripts_files"
    if rel.startswith("tests/") and rel.count("/") == 1:
        return "wing_tests"
    if rel == "src/wyrdforge/__init__.py":
        return "package_init"
    if rel.startswith("src/wyrdforge/schemas/") \
            and rel.count("/") == 3:
        return "schemas"
    if rel.startswith("src/wyrdforge/services/") \
            and rel.count("/") == 3:
        return "services"
    if rel.startswith("src/wyrdforge/security/") \
            and rel.count("/") == 3:
        return "security"
    if rel.startswith("src/wyrdforge/runtime/") \
            and rel.count("/") == 3:
        return "runtime"
    return None


# ── Code/test files, as literal data ─────────────────────────────────────────

FILE_ROWS: tuple[dict[str, Any], ...] = (
    {"path": "research_data/src/wyrdforge/services/bond_graph_service.py",
     "area": "services",
     "disposition": "graduated_source",
     "rule": "graduated by TF-143: the production counterpart "
             "in the shipping tree is the code that ships; this "
             "wing shell stays as the adopted source and carries "
             "no retirement banner (TF-144)",
     "production_counterpart": "src/thoughtforge/knowledge/bond_graph.py",
     "banner_marker": "",
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.bond import",
                 "class BondGraphService")},
    {"path": "research_data/src/wyrdforge/services/persona_compiler.py",
     "area": "services",
     "disposition": "graduated_source",
     "rule": "graduated by TF-143: the production counterpart "
             "in the shipping tree is the code that ships; this "
             "wing shell stays as the merged source and carries "
             "no retirement banner (TF-144)",
     "production_counterpart": "src/thoughtforge/cognition/persona_compiler.py",
     "banner_marker": "",
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.persona import",
                 "class PersonaCompiler")},
    {"path": "research_data/src/wyrdforge/services/micro_rag_pipeline.py",
     "area": "services",
     "disposition": "archived_retired",
     "rule": "retired by TF-140's verdict and archived-in-place "
             "by TF-144 under its filed banner; verdict evidence, "
             "not to be imported or revived without a new "
             "roadmap word",
     "production_counterpart": "",
     "banner_marker": ARCHIVE_BANNER_MARKER,
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.micro_rag import",
                 "def score", "def assemble")},
    {"path": "research_data/src/wyrdforge/services/truth_calibrator.py",
     "area": "services",
     "disposition": "archived_retired",
     "rule": "retired by TF-142's verdict and archived-in-place "
             "by TF-144 under its filed banner; verdict evidence, "
             "not to be imported or revived without a new "
             "roadmap word",
     "production_counterpart": "",
     "banner_marker": ARCHIVE_BANNER_MARKER,
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.evals import",
                 "def evaluate")},
    {"path": "research_data/src/wyrdforge/services/memory_store.py",
     "area": "services",
     "disposition": "fenced_research",
     "rule": "fenced research: never adopted and never retired; "
             "it imports the missing models package, cannot "
             "run, and is not shipping code",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.memory import",
                 "class InMemoryRecordStore")},
    {"path": "research_data/src/wyrdforge/security/permission_guard.py",
     "area": "security",
     "disposition": "fenced_standalone",
     "rule": "fenced research, standalone leg: it imports "
             "nothing from the missing models package and is "
             "the wing code that can be imported on its own — "
             "still research, not adopted into the shipping tree",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "no models-package import",
     "markers": ("class PermissionGuard",)},
    {"path": "research_data/src/wyrdforge/security/prompt_injection_guard.py",
     "area": "security",
     "disposition": "fenced_standalone",
     "rule": "fenced research, standalone leg: it imports "
             "nothing from the missing models package and is "
             "the wing code that can be imported on its own — "
             "still research, not adopted into the shipping tree",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "no models-package import",
     "markers": ("def detect_prompt_injection",)},
    {"path": "research_data/src/wyrdforge/runtime/demo_seed.py",
     "area": "runtime",
     "disposition": "fenced_research",
     "rule": "fenced research: a demo seed for the wing's own "
             "tests; it imports the missing models package, "
             "cannot run, and is not shipping code",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.memory import",
                 "def build_seed_fact")},
    {"path": "research_data/src/wyrdforge/__init__.py",
     "area": "package_init",
     "disposition": "fence_import",
     "rule": "the filed broken-import state: the init "
             "re-exports ten names from the missing models "
             "package, TF-139's checker requires those import "
             "lines to remain, and TF-148 does not alter them — "
             "the fence files the breakage, it does not hide it",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "imports the missing models package "
                     "through relative model imports",
     "markers": ("from .models.memory import",
                 "from .models.bond import",
                 "from .models.persona import",
                 "from .models.micro_rag import",
                 "from .models.evals import",
                 "MemoryRecord", "BondEdge", "PersonaPacket",
                 "MicroContextPacket", "EvalResult")},
    {"path": "research_data/tests/test_memory_and_persona.py",
     "area": "wing_tests",
     "disposition": "fenced_research",
     "rule": "fenced research under the wing's own test "
             "discipline: it imports the missing models "
             "package and errors at collection there; TF-149 "
             "owns the wing tests' final disposition, which "
             "this slice does not discharge",
     "production_counterpart": "",
     "banner_marker": "",
     "import_state": "imports the missing models package",
     "markers": ("from wyrdforge.models.bond import",
                 "def test_memory_store_search_and_promote",
                 "def test_persona_compiler_builds_packet")},
    {"path": "research_data/tests/test_micro_rag_and_truth.py",
     "area": "wing_tests",
     "disposition": "archived_retired",
     "rule": "archived-in-place with the retired code by "
             "TF-144 under its filed banner; verdict evidence; "
             "TF-149 owns the wing tests' final disposition",
     "production_counterpart": "",
     "banner_marker": ARCHIVE_BANNER_MARKER,
     "import_state": "imports the missing models package",
     "markers": ("def test_micro_rag_assembles_packet",
                 "def test_truth_calibrator_flags_unsupported_exclusivity")},
)


# ── The test discipline, as literal data ─────────────────────────────────────

TEST_DISCIPLINE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "main_suite",
     "scope": "the campaign suite at the repo root",
     "rule": "the root pyproject files testpaths = [\"tests\"] "
             "only, so the main suite never collects the wing's "
             "tests under research_data/tests",
     "state": "in force, checked live against both pyprojects"},
    {"id": "wing_suite",
     "scope": "the wing's own suite inside research_data/",
     "rule": "the wing's own pyproject files its own package, "
             "pythonpath = [\"src\"] and testpaths = [\"tests\"]; "
             "its 2 test files carry 4 test functions",
     "state": "collection error, filed not hidden: the missing "
              "models package makes the wing suite error at "
              "collection (2 errors, 0 run, TF-006)"},
    {"id": "graduated_tests",
     "scope": "tests for graduated code",
     "rule": "adopted code is judged by shipping tests under "
             "the root tests/ tree (TF-143's landings carry "
             "theirs), never by the wing's own tests",
     "state": "in force; TF-149 owns the wing-test final "
              "disposition, explicitly not discharged here"},
    {"id": "retired_tests",
     "scope": "tests for retired code",
     "rule": "retired code's tests are archived-in-place with "
             "it under the TF-144 banner, standing as verdict "
             "evidence",
     "state": "executed by TF-144 for the shared micro/truth "
              "wing test file"},
)


# ── The fence rules, as literal data ─────────────────────────────────────────

FENCE_RULES: tuple[dict[str, Any], ...] = (
    {"id": "no_shipping_imports_the_wing",
     "rule": "no shipping file under src/thoughtforge imports "
             "the wing package — judged on real import "
             "statements, parsed; prose, docstrings and string "
             "data mentioning the wing are not imports"},
    {"id": "no_research_file_is_shipping",
     "rule": "no file under research_data/ is shipping code: "
             "graduated ideas ship only as their production "
             "counterparts in the shipping tree"},
    {"id": "revival_needs_a_new_roadmap_word",
     "rule": "reviving, importing or building on any fenced "
             "or archived file needs a new roadmap word"},
    {"id": "schemas_stay_evidence",
     "rule": "the 14 schemas stay unmarked as verdict "
             "evidence; they specify shapes and execute nothing"},
    {"id": "readmes_never_touched",
     "rule": "no README.md anywhere is touched by the fence — "
             "the fence notice is its own new file"},
    {"id": "models_package_not_resurrected",
     "rule": "the missing models package is not resurrected "
             "by this slice; if it ever appears, this filing "
             "must be updated and the checker names that"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def area_row_problems(row: Any) -> list[str]:
    """Judge one area filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["area: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("area: the row files no id")
    if not row.get("area"):
        problems.append(f"area {row.get('id')}: no area filed")
    if not row.get("pattern"):
        problems.append(f"area {row.get('id')}: no pattern filed")
    if not isinstance(row.get("tracked_count"), int) \
            or row.get("tracked_count", 0) <= 0:
        problems.append(
            f"area {row.get('id')}: tracked count is "
            f"{row.get('tracked_count')!r}, not a positive count")
    if row.get("disposition") not in AREA_DISPOSITIONS:
        problems.append(
            f"area {row.get('id')}: disposition is "
            f"{row.get('disposition')!r}, not one of "
            f"{AREA_DISPOSITIONS}")
    if not row.get("rule"):
        problems.append(f"area {row.get('id')}: no rule filed")
    if not isinstance(row.get("sparse_present"), bool):
        problems.append(
            f"area {row.get('id')}: sparse presence is not filed")
    return problems


def file_row_problems(row: Any) -> list[str]:
    """Judge one code/test file filing, purely: a
    path, an area, a disposition in the filed
    vocabulary, a rule, an import state and
    markers; a graduated row must name its
    production counterpart and an archived row
    must name its banner marker. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["file: the row is empty"]
    problems: list[str] = []
    path = row.get("path")
    if not path or not str(path).startswith("research_data/"):
        problems.append("file: the row files no research_data path")
    if not row.get("area"):
        problems.append(f"file {path}: no area filed")
    if row.get("disposition") not in FILE_DISPOSITIONS:
        problems.append(
            f"file {path}: disposition is "
            f"{row.get('disposition')!r}, not one of "
            f"{FILE_DISPOSITIONS} — no ambiguous middle state "
            "may remain")
    if not row.get("rule"):
        problems.append(f"file {path}: no rule filed")
    if not row.get("import_state"):
        problems.append(f"file {path}: no import state filed")
    if not row.get("markers"):
        problems.append(
            f"file {path}: no markers filed — the filing "
            "cannot be verified live")
    if row.get("disposition") == "graduated_source" \
            and not row.get("production_counterpart"):
        problems.append(
            f"file {path}: graduated row missing its "
            "production counterpart")
    if row.get("disposition") == "archived_retired" \
            and not row.get("banner_marker"):
        problems.append(
            f"file {path}: archived row missing its banner")
    if row.get("disposition") == "fenced_standalone" \
            and row.get("import_state") != "no models-package import":
        problems.append(
            f"file {path}: a standalone filing must file that "
            "it imports nothing from the models package")
    return problems


def discipline_row_problems(row: Any) -> list[str]:
    """Judge one test-discipline filing, purely.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["test discipline: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("test discipline: the row files no id")
    for key in ("scope", "rule", "state"):
        if not row.get(key):
            problems.append(
                f"test discipline {row.get('id')}: no {key} filed")
    return problems


def fence_rule_problems(row: Any) -> list[str]:
    """Judge one fence-rule filing, purely. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["fence rule: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("fence rule: the row files no id")
    if not row.get("rule"):
        problems.append(
            f"fence rule {row.get('id')}: no rule filed")
    return problems


def tracked_files_problems(
        tracked: Any,
        area_rows: Any = None,
        file_rows: Any = None) -> list[str]:
    """Judge a tracked-file list for research_data
    against the filings, purely: every path
    classifies into a filed area, every area's
    count matches its filing (the fence file may
    be present or not yet tracked — both banking
    states pass), no duplicates, and every filed
    code/test file is tracked. Empty-safe."""
    if not isinstance(tracked, (list, tuple)) or not tracked:
        return ["tracked files: the list is empty"]
    areas = AREA_ROWS if area_rows is None else area_rows
    files = FILE_ROWS if file_rows is None else file_rows
    problems: list[str] = []
    for row in areas or ():
        problems.extend(area_row_problems(row))
    for row in files or ():
        problems.extend(file_row_problems(row))
    paths = list(tracked)
    if len(paths) != len(set(paths)):
        problems.append("tracked files: a path is filed twice")
    counts: dict[str, int] = {}
    for path in paths:
        area_id = classify_tracked_path(path)
        if area_id is None:
            problems.append(
                f"tracked files: {path} is unclassified — no "
                "ambiguous middle state may remain")
        else:
            counts[area_id] = counts.get(area_id, 0) + 1
    for row in areas or ():
        measured = counts.get(row["id"], 0)
        if row["id"] == "research_fence_file":
            if measured not in (0, row["tracked_count"]):
                problems.append(
                    f"tracked files: area research_fence_file "
                    f"counts {measured}, not 0 (not yet banked) "
                    f"or the filed {row['tracked_count']}")
            continue
        if measured != row["tracked_count"]:
            problems.append(
                f"tracked files: area {row['id']} counts "
                f"{measured}, not the filed "
                f"{row['tracked_count']}")
    tracked_set = set(paths)
    for row in files or ():
        if row.get("path") not in tracked_set:
            problems.append(
                f"tracked files: the filed file "
                f"{row.get('path')} is not tracked")
    return problems


def hygiene_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured hygiene state against its
    expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["hygiene state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["hygiene state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"hygiene state: {key} measures "
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


# ── The live hygiene state ───────────────────────────────────────────────────

HYGIENE_EXPECTED: dict[str, Any] = {
    "area_rows": 13,
    "baseline_areas": 12,
    "baseline_tracked_files": 96,
    "tracked_files_with_fence": 97,
    "file_rows": 11,
    "graduated_files": 2,
    "archived_files": 3,
    "fenced_research_files": 3,
    "fenced_standalone_files": 2,
    "fence_import_files": 1,
    "test_discipline_rows": 4,
    "fence_rules": 6,
    "disk_top_level_markdown": 58,
    "disk_schemas": 14,
    "disk_services": 5,
    "disk_security": 2,
    "disk_wing_tests": 2,
    "disk_runtime_files": 1,
    "disk_sparse_excluded_dirs_present": 0,
    "archived_bannered": 3,
    "graduated_bannered": 0,
    "production_counterparts_present": 2,
    "security_models_imports": 0,
    "models_import_files": 8,
    "init_model_import_lines": 5,
    "wing_test_files": 2,
    "wing_test_functions": 4,
    "models_package_present": False,
    "fence_file_present": True,
    "shipping_wing_import_files": 0,
}


def _read_rel(rel: str) -> str | None:
    path = _REPO_ROOT / rel
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _shipping_wing_import_files() -> int:
    """Count shipping files carrying a real import
    of the wing package, by parsing — prose,
    docstrings and string data are not imports."""
    count = 0
    src_root = _REPO_ROOT / "src" / "thoughtforge"
    for path in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            count += 1
            continue
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "wyrdforge"
                       or alias.name.startswith("wyrdforge.")
                       for alias in node.names):
                    found = True
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "wyrdforge" \
                        or module.startswith("wyrdforge."):
                    found = True
        if found:
            count += 1
    return count


def run_research_hygiene_state() -> dict[str, Any]:
    """Measure the fence on the live tree: disk
    counts for the sparse-present areas only, the
    filed dispositions re-read from the wing files
    themselves, and the shipping import scan.
    Deterministic: counts and booleans only."""
    baseline = [r for r in AREA_ROWS
                if r["id"] != "research_fence_file"]
    texts = {row["path"]: _read_rel(row["path"]) for row in FILE_ROWS}
    archived = [r for r in FILE_ROWS
                if r["disposition"] == "archived_retired"]
    graduated = [r for r in FILE_ROWS
                 if r["disposition"] == "graduated_source"]
    models_import = 0
    for row in FILE_ROWS:
        if row["disposition"] in (
                "fenced_research", "archived_retired",
                "graduated_source"):
            text = texts.get(row["path"]) or ""
            if "from wyrdforge.models" in text:
                models_import += 1
    init_text = texts.get(
        "research_data/src/wyrdforge/__init__.py") or ""
    wing_tests_text = "".join(
        texts.get(row["path"]) or "" for row in FILE_ROWS
        if row["area"] == "wing_tests")
    sparse_dirs = ("config", "examples", "scripts", "wyrd_runtime")
    return {
        "area_rows": len(AREA_ROWS),
        "baseline_areas": len(baseline),
        "baseline_tracked_files": sum(
            r["tracked_count"] for r in baseline),
        "tracked_files_with_fence": sum(
            r["tracked_count"] for r in AREA_ROWS),
        "file_rows": len(FILE_ROWS),
        "graduated_files": len(graduated),
        "archived_files": len(archived),
        "fenced_research_files": sum(
            1 for r in FILE_ROWS
            if r["disposition"] == "fenced_research"),
        "fenced_standalone_files": sum(
            1 for r in FILE_ROWS
            if r["disposition"] == "fenced_standalone"),
        "fence_import_files": sum(
            1 for r in FILE_ROWS
            if r["disposition"] == "fence_import"),
        "test_discipline_rows": len(TEST_DISCIPLINE_ROWS),
        "fence_rules": len(FENCE_RULES),
        "disk_top_level_markdown": len([
            p for p in _RESEARCH.glob("*.md")
            if p.name != "RESEARCH_FENCE.md"]),
        "disk_schemas": len(list((_WING / "schemas").glob("*.json"))),
        "disk_services": len(list((_WING / "services").glob("*.py"))),
        "disk_security": len(list((_WING / "security").glob("*.py"))),
        "disk_wing_tests": len(list(
            (_RESEARCH / "tests").glob("*.py"))),
        "disk_runtime_files": len(list(
            (_WING / "runtime").glob("*.py"))),
        "disk_sparse_excluded_dirs_present": sum(
            1 for name in sparse_dirs
            if (_RESEARCH / name).exists()),
        "archived_bannered": sum(
            1 for r in archived
            if (texts.get(r["path"]) or "").startswith(
                r["banner_marker"])),
        "graduated_bannered": sum(
            1 for r in graduated
            if ARCHIVE_BANNER_MARKER in (texts.get(r["path"]) or "")),
        "production_counterparts_present": sum(
            1 for r in graduated
            if (_REPO_ROOT / r["production_counterpart"]).exists()),
        "security_models_imports": sum(
            1 for r in FILE_ROWS
            if r["disposition"] == "fenced_standalone"
            and "wyrdforge.models" in (texts.get(r["path"]) or "")),
        "models_import_files": models_import,
        "init_model_import_lines": sum(
            1 for line in init_text.splitlines()
            if line.startswith("from .models.")),
        "wing_test_files": sum(
            1 for r in FILE_ROWS if r["area"] == "wing_tests"),
        "wing_test_functions": wing_tests_text.count("def test_"),
        "models_package_present": (_WING / "models").exists(),
        "fence_file_present": (_REPO_ROOT / RESEARCH_FENCE_DOC).exists(),
        "shipping_wing_import_files": _shipping_wing_import_files(),
    }


def _synthetic_tracked_files() -> list[str]:
    """Build a complete tracked-file list from the
    filings alone (placeholder names under each
    area's pattern, the filed code/test files in
    their places) — the probes' clean ground."""
    paths: list[str] = []
    filed_code = {row["path"] for row in FILE_ROWS}
    for row in AREA_ROWS:
        area_id = row["id"]
        if area_id == "top_level_markdown":
            paths += [f"research_data/spec_{i:02d}.md"
                      for i in range(row["tracked_count"])]
        elif area_id == "schemas":
            paths += [f"research_data/src/wyrdforge/schemas/s{i}.json"
                      for i in range(row["tracked_count"])]
        elif area_id in ("services", "security", "runtime",
                         "package_init", "wing_tests"):
            paths += [r["path"] for r in FILE_ROWS
                      if r["area"] == area_id]
        elif area_id == "wyrd_runtime_packets":
            paths += [f"research_data/wyrd_runtime/p{i}.md"
                      for i in range(row["tracked_count"])]
        elif area_id == "config_files":
            paths += [f"research_data/config/c{i}.yaml"
                      for i in range(row["tracked_count"])]
        elif area_id == "examples_files":
            paths += [f"research_data/examples/e{i}.yaml"
                      for i in range(row["tracked_count"])]
        elif area_id == "scripts_files":
            paths += ["research_data/scripts/generate_json_schemas.py"]
        elif area_id == "wing_pyproject":
            paths += ["research_data/pyproject.toml"]
        elif area_id == "research_fence_file":
            paths += [RESEARCH_FENCE_DOC]
    # The filed code set must be exactly the split areas' files.
    assert filed_code <= set(paths)
    return paths


# ── The filed probes ─────────────────────────────────────────────────────────

HYGIENE_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_estate_passes",
     "kind": "clean",
     "detail": "the filed areas + files + discipline + a "
               "complete tracked list pass their validators"},
    {"id": "unclassified_tracked_file_is_named",
     "kind": "unclassified",
     "detail": "a tracked list carrying an extra unclassified "
               "path is named"},
    {"id": "wrong_area_count_is_named",
     "kind": "wrong_count",
     "detail": "a tracked list missing one schema file files "
               "the wrong schemas count and is named"},
    {"id": "unknown_disposition_is_named",
     "kind": "unknown_disposition",
     "detail": "a file row filing a disposition outside the "
               "filed vocabulary is named"},
    {"id": "graduated_row_missing_counterpart_is_named",
     "kind": "graduated_no_counterpart",
     "detail": "a graduated file row with its production "
               "counterpart emptied is named"},
    {"id": "archived_row_missing_banner_is_named",
     "kind": "archived_no_banner",
     "detail": "an archived file row with its banner marker "
               "emptied is named"},
    {"id": "fence_doc_drift_is_named",
     "kind": "fence_drift",
     "detail": "a fence text that drifts from the filed "
               "render is named"},
)


def run_research_hygiene_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in HYGIENE_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in AREA_ROWS:
                problems += area_row_problems(row)
            for row in FILE_ROWS:
                problems += file_row_problems(row)
            for row in TEST_DISCIPLINE_ROWS:
                problems += discipline_row_problems(row)
            problems += tracked_files_problems(
                _synthetic_tracked_files())
            passed = problems == []
        elif kind == "unclassified":
            tracked = _synthetic_tracked_files() + [
                "research_data/mystery_thing.txt"]
            passed = any("unclassified" in p for p in
                         tracked_files_problems(tracked))
        elif kind == "wrong_count":
            tracked = [p for p in _synthetic_tracked_files()
                       if not p.endswith("schemas/s0.json")]
            passed = any("schemas" in p for p in
                         tracked_files_problems(tracked))
        elif kind == "unknown_disposition":
            row = dict(FILE_ROWS[0], disposition="limbo")
            passed = bool(file_row_problems(row))
        elif kind == "graduated_no_counterpart":
            row = dict(FILE_ROWS[0], production_counterpart="")
            passed = bool(file_row_problems(row))
        elif kind == "archived_no_banner":
            row = dict(FILE_ROWS[2], banner_marker="")
            passed = bool(file_row_problems(row))
        elif kind == "fence_drift":
            drifted = render_research_fence_markdown() + "drift\n"
            passed = bool(doc_text_problems(
                drifted, render_research_fence_markdown(),
                "fence doc"))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable renders (both docs, verbatim) ───────────────────────────────

def render_research_fence_markdown() -> str:
    """Render the fence notice deterministically —
    the exact text filed as
    research_data/RESEARCH_FENCE.md."""
    lines: list[str] = []
    lines.append("# RESEARCH FENCE — this tree is fenced research (TF-148)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/research_hygiene.py` — regenerate "
        "with `python3 -m thoughtforge.research_hygiene`.")
    lines.append("")
    lines.append(
        "Everything under `research_data/` is **fenced "
        "research**: provenance, design, spec and verdict "
        "evidence for ThoughtForge. No file in this tree is "
        "shipping code, and no shipping file imports this "
        "tree's package.")
    lines.append("")
    lines.append("## What graduated")
    lines.append("")
    lines.append(
        "- The bond graph service graduated in TF-143 — the "
        "shipping code is "
        "`src/thoughtforge/knowledge/bond_graph.py`.")
    lines.append(
        "- The persona compiler service graduated in TF-143 "
        "— the shipping code is "
        "`src/thoughtforge/cognition/persona_compiler.py`.")
    lines.append("")
    lines.append("## What is archived-retired")
    lines.append("")
    lines.append(
        "- The micro-RAG pipeline service, the truth "
        "calibrator service and their shared test file are "
        "archived-in-place retired by TF-144, each under its "
        "filed retirement banner (TF-140 / TF-142 verdicts). "
        "They stand as verdict evidence.")
    lines.append("")
    lines.append("## What remains fenced research or evidence")
    lines.append("")
    lines.append(
        "- The design and spec markdown and the wyrd_runtime "
        "packets: fenced reference.")
    lines.append(
        "- The 14 schemas: fenced evidence, kept unmarked.")
    lines.append(
        "- Config and examples: fenced examples. The scripts "
        "generator: fenced tooling.")
    lines.append(
        "- The memory store, the runtime demo seed and the "
        "memory/persona test file: fenced research — they "
        "import the missing models package, cannot run, and "
        "are not shipping.")
    lines.append(
        "- The two security guards: fenced standalone "
        "research — importable on their own, still not "
        "adopted.")
    lines.append(
        "- The package init: the filed broken-import state — "
        "it re-exports ten names from the missing models "
        "package. The wing cannot import, and that state is "
        "filed here, not hidden.")
    lines.append("")
    lines.append("## Its own test discipline")
    lines.append("")
    lines.append(
        "This tree is its own package (see its "
        "`pyproject.toml`) with its own testpaths. The "
        "campaign suite at the repo root never collects its "
        "tests. Running its own suite errors at collection "
        "while the models package is missing — a filed "
        "state (TF-006). TF-149 owns the wing tests' final "
        "disposition.")
    lines.append("")
    lines.append("## Revival")
    lines.append("")
    lines.append(
        "Reviving, importing or building on any fenced or "
        "archived file in this tree needs a new roadmap "
        "word. The full filing is "
        "`docs/architecture/RESEARCH_HYGIENE.md`.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_research_hygiene_markdown() -> str:
    """Render the hygiene doc deterministically —
    the exact text filed as
    docs/architecture/RESEARCH_HYGIENE.md."""
    state = run_research_hygiene_state()
    lines: list[str] = []
    lines.append("# Research Hygiene — research_data/ Fenced (TF-148)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/research_hygiene.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.research_hygiene` (the "
        "command also regenerates "
        "`research_data/RESEARCH_FENCE.md`). The checker "
        "(`research_hygiene_problems()`) re-derives every "
        "claim live from the tree and fails on any drift.")
    lines.append("")
    lines.append(f"Format: `{RESEARCH_HYGIENE_FORMAT}`")
    lines.append(
        f"Tracked census: {state['baseline_tracked_files']} "
        f"baseline files in {state['baseline_areas']} areas, "
        f"plus the fence file = "
        f"{state['tracked_files_with_fence']} tracked files "
        "once banked")
    lines.append(
        f"Code/test files filed by disposition: "
        f"{state['file_rows']} (graduated "
        f"{state['graduated_files']} / archived "
        f"{state['archived_files']} / fenced research "
        f"{state['fenced_research_files']} / fenced "
        f"standalone {state['fenced_standalone_files']} / "
        f"fence import {state['fence_import_files']})")
    lines.append(f"Probes: {len(HYGIENE_PROBES)}/"
                 f"{len(HYGIENE_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: "
        "`git ls-files research_data` files 96 tracked files "
        "in twelve areas. This clone is sparse — config, "
        "examples, scripts and wyrd_runtime are tracked but "
        "not on disk here (12 files), so this doc counts disk "
        "areas for the sparse-present areas and judges the "
        "sparse-excluded areas only through a pure validator "
        "fed a tracked-file list. The wing still cannot "
        "import: the models package does not exist and the "
        "package init re-exports ten names from it (that "
        "init's import lines stand unaltered — a neighbouring "
        "verdict checker reads them live). Two services had "
        "graduated in TF-143, two services and their shared "
        "test file were archived-retired in TF-144, and five "
        "code or test files had never been given any "
        "disposition — the ambiguous middle this slice ends.")
    lines.append("")
    lines.append("## 2. Every area, filed")
    lines.append("")
    lines.append("| Area | Files | Disposition | Rule |")
    lines.append("|---|---|---|---|")
    for row in AREA_ROWS:
        lines.append(
            f"| {row['area']} | {row['tracked_count']} | "
            f"{row['disposition']} | {row['rule']} |")
    lines.append("")
    lines.append("## 3. Every code and test file, filed")
    lines.append("")
    lines.append("| File | Disposition | Rule |")
    lines.append("|---|---|---|")
    for row in FILE_ROWS:
        lines.append(
            f"| `{row['path']}` | {row['disposition']} | "
            f"{row['rule']} |")
    lines.append("")
    lines.append(
        "Graduated counterparts, checked live: "
        + "; ".join(
            f"`{r['path']}` -> `{r['production_counterpart']}`"
            for r in FILE_ROWS
            if r["disposition"] == "graduated_source")
        + ". Archived files carry their TF-144 banner at "
        "their head, checked live; graduated files carry "
        "none, checked live.")
    lines.append("")
    lines.append("## 4. Its own test discipline")
    lines.append("")
    for row in TEST_DISCIPLINE_ROWS:
        lines.append(
            f"- `{row['id']}` ({row['scope']}): {row['rule']} "
            f"— State: {row['state']}.")
    lines.append("")
    lines.append("## 5. The fence rules")
    lines.append("")
    for row in FENCE_RULES:
        lines.append(f"- `{row['id']}`: {row['rule']}")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-148 discharges the research-hygiene share of "
        "D-WING-1 (narrowed in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). "
        "D-WING-1 itself stays OPEN for TF-149's wing-tests "
        "share — TF-148 is not the last Part VIII slice on "
        "the entry. No new docket entry is added and TF-016 "
        "totals do not move (137 entries — 33 OPEN / 1 "
        "OPEN-ENVIRONMENT / 100 FIXED / 3 "
        "CLOSED-BY-VERDICT).")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in HYGIENE_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Fence/static grade only: this slice files "
        "dispositions and fences the tree; it adopts no "
        "research code, retires nothing new, and resurrects "
        "no package — the wing still cannot import, exactly "
        "as filed.")
    lines.append(
        "- TF-149 owns the wing tests' final disposition "
        "and TF-150 the wing's provenance; neither share is "
        "discharged here.")
    lines.append(
        "- The sparse-excluded areas are judged through "
        "the tracked-file list only, never assumed to "
        "exist on disk in this clone.")
    lines.append(
        "- No README.md anywhere was touched; no canon "
        "data was touched; no live model was read — live "
        "behaviour is NOT PROVEN (TF-173 class), never "
        "claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_research_hygiene_docs() -> tuple[Path, Path]:
    """Write both renders to their filed homes."""
    hygiene = _REPO_ROOT / RESEARCH_HYGIENE_DOC
    hygiene.write_text(
        render_research_hygiene_markdown(), encoding="utf-8")
    fence = _REPO_ROOT / RESEARCH_FENCE_DOC
    fence.write_text(
        render_research_fence_markdown(), encoding="utf-8")
    return hygiene, fence


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.research_hygiene` —
    regenerates both filed docs."""
    hygiene, fence = write_research_hygiene_docs()
    print(f"research hygiene doc regenerated: -> "
          f"{RESEARCH_HYGIENE_DOC} ({hygiene})")
    print(f"research fence regenerated: -> "
          f"{RESEARCH_FENCE_DOC} ({fence})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def research_hygiene_problems() -> list[str]:
    """Re-derive TF-148's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    area_ids = [r["id"] for r in AREA_ROWS]
    if len(area_ids) != len(set(area_ids)) or len(AREA_ROWS) != 13:
        problems.append("the AREA_ROWS filing is not thirteen "
                        "unique areas")
    for row in AREA_ROWS:
        problems.extend(area_row_problems(row))
    file_paths = [r["path"] for r in FILE_ROWS]
    if len(file_paths) != len(set(file_paths)) or len(FILE_ROWS) != 11:
        problems.append("the FILE_ROWS filing is not eleven "
                        "unique files")
    for row in FILE_ROWS:
        problems.extend(file_row_problems(row))
    for row in TEST_DISCIPLINE_ROWS:
        problems.extend(discipline_row_problems(row))
    for row in FENCE_RULES:
        problems.extend(fence_rule_problems(row))

    # File rows reconcile with the split areas' counts.
    for row in AREA_ROWS:
        if row["disposition"] != "split_by_file":
            continue
        filed = sum(1 for r in FILE_ROWS if r["area"] == row["id"])
        if filed != row["tracked_count"]:
            problems.append(
                f"areas: split area {row['id']} files "
                f"{row['tracked_count']} tracked files but "
                f"{filed} file rows")

    # The tracked census, when git can file it: the
    # pure validator accepts the fence file tracked
    # or not yet tracked (both banking states).
    try:
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "research_data"],
            cwd=_REPO_ROOT, capture_output=True, text=True,
            timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            tracked = result.stdout.splitlines()
            problems.extend(
                f"tracked: {p}"
                for p in tracked_files_problems(tracked))
    except Exception as exc:  # pragma: no cover — git is present here
        problems.append(f"tracked: git ls-files failed: {exc}")

    # File dispositions, live.
    for row in FILE_ROWS:
        text = _read_rel(row["path"])
        if text is None:
            problems.append(f"files: {row['path']} is missing")
            continue
        for marker in row["markers"]:
            if marker not in text:
                problems.append(
                    f"files: {row['path']} no longer retains "
                    f"the filed marker {marker!r}")
        if row["disposition"] == "archived_retired":
            if not text.startswith(row["banner_marker"]):
                problems.append(
                    f"files: {row['path']} no longer opens "
                    "with its filed TF-144 banner")
        if row["disposition"] == "graduated_source":
            if ARCHIVE_BANNER_MARKER in text:
                problems.append(
                    f"files: {row['path']} is graduated but "
                    "carries a retirement banner")
            counterpart = _REPO_ROOT / row["production_counterpart"]
            if not counterpart.exists():
                problems.append(
                    f"files: {row['path']} names production "
                    f"counterpart {row['production_counterpart']} "
                    "which does not exist")
        if row["disposition"] == "fenced_standalone":
            if "wyrdforge.models" in text or "wyrdforge" in text:
                problems.append(
                    f"files: {row['path']} is filed standalone "
                    "but now names the wing package")

    # The models package stays absent — if that ever
    # changes, this filing must be updated.
    if (_WING / "models").exists():
        problems.append(
            "wing: the models package now exists — this "
            "hygiene filing must be updated")

    # Both pyproject fences, parsed.
    try:
        import tomllib

        root_data = tomllib.loads(
            (_REPO_ROOT / "pyproject.toml").read_text(
                encoding="utf-8"))
        root_testpaths = root_data["tool"]["pytest"][
            "ini_options"]["testpaths"]
        if root_testpaths != ["tests"]:
            problems.append(
                f"pyproject: root testpaths are "
                f"{root_testpaths}, not ['tests']")
        wing_data = tomllib.loads(
            (_RESEARCH / "pyproject.toml").read_text(
                encoding="utf-8"))
        if wing_data["project"]["name"] != "wyrdforge":
            problems.append("pyproject: the wing package is "
                            "not named wyrdforge")
        wing_opts = wing_data["tool"]["pytest"]["ini_options"]
        if wing_opts.get("pythonpath") != ["src"] \
                or wing_opts.get("testpaths") != ["tests"]:
            problems.append(
                "pyproject: the wing's own pythonpath / "
                "testpaths fence has drifted")
    except Exception as exc:
        problems.append(f"pyproject: a fence cannot be parsed: {exc}")

    # The measured state matches its filing.
    problems.extend(hygiene_state_problems(
        run_research_hygiene_state(), HYGIENE_EXPECTED))

    # The neighbouring retirement checker still stands.
    from thoughtforge.wing_retirements import (
        wing_retirements_problems,
    )

    neighbour = wing_retirements_problems()
    if neighbour:
        problems.append(
            f"neighbour: wing_retirements_problems() is not "
            f"clean: {neighbour[:3]}")

    # The docket carries this slice's narrowing.
    docket_text = _DOCKET.read_text(encoding="utf-8")
    if "Narrowed by TF-148" not in docket_text:
        problems.append(
            "docket: D-WING-1 does not file TF-148's narrowing")

    # The probes all behave.
    probes = run_research_hygiene_probes()
    if len(probes) != len(HYGIENE_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the hygiene probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # Both filed docs match their renders verbatim.
    for rel, render, label in (
            (RESEARCH_HYGIENE_DOC, render_research_hygiene_markdown(),
             "hygiene doc"),
            (RESEARCH_FENCE_DOC, render_research_fence_markdown(),
             "fence doc")):
        doc = _REPO_ROOT / rel
        if not doc.exists():
            problems.append(f"the {label} {rel} is missing")
        else:
            problems.extend(
                f"{label}: {p}" for p in doc_text_problems(
                    doc.read_text(encoding="utf-8"), render, label))
    return problems


def research_hygiene_as_data() -> dict[str, Any]:
    """The whole filing as plain data, for consumers."""
    probes = run_research_hygiene_probes()
    return {
        "format": RESEARCH_HYGIENE_FORMAT,
        "doc": RESEARCH_HYGIENE_DOC,
        "fence_doc": RESEARCH_FENCE_DOC,
        "areas": [dict(r) for r in AREA_ROWS],
        "files": [dict(r) for r in FILE_ROWS],
        "test_discipline": [dict(r) for r in TEST_DISCIPLINE_ROWS],
        "fence_rules": [dict(r) for r in FENCE_RULES],
        "state": run_research_hygiene_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "AREA_DISPOSITIONS",
    "AREA_ROWS",
    "ARCHIVE_BANNER_MARKER",
    "BASELINE_TRACKED_FILES",
    "FENCE_RULES",
    "FENCE_TRACKED_FILES",
    "FILE_DISPOSITIONS",
    "FILE_ROWS",
    "HYGIENE_EXPECTED",
    "HYGIENE_PROBES",
    "RESEARCH_FENCE_DOC",
    "RESEARCH_HYGIENE_DOC",
    "RESEARCH_HYGIENE_FORMAT",
    "TEST_DISCIPLINE_ROWS",
    "area_row_problems",
    "classify_tracked_path",
    "doc_text_problems",
    "fence_rule_problems",
    "file_row_problems",
    "hygiene_state_problems",
    "main",
    "render_research_fence_markdown",
    "render_research_hygiene_markdown",
    "research_hygiene_as_data",
    "research_hygiene_problems",
    "run_research_hygiene_probes",
    "run_research_hygiene_state",
    "discipline_row_problems",
    "tracked_files_problems",
    "write_research_hygiene_docs",
]

if __name__ == "__main__":
    raise SystemExit(main())
