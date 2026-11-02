"""
TF-144 — the wyrdforge wing's retirements, executed.

TF-140 retired wyrdforge's micro-RAG (as a retrieval
mechanism, by measurement on TF-043's fixtures) and
TF-142 retired its truth calibrator (as a calibration /
enforcement mechanism, by measurement on TF-100's chaos
corpus). Both verdicts filed the same execution terms:
the retirement itself is TF-144's work — each RETIRE
verdict's code removed or archived-in-place, lessons
written into the wing's closing doc — TF-143 owns
adoptions (it filed none for these two), TF-149 owns
the wing's tests. TF-139's bond graph (ADOPT_AS_LAYER)
and TF-141's persona compiler (MERGE) were adopted in
TF-143 and are NOT retired here; this module's checker
pins that their wing files carry no retirement banner.

The measured start, probed live on this tree before
anything was built:

  * The wing still cannot import (TF-006 / D-WING-1):
    ``research_data/src/wyrdforge/models`` does not
    exist; the two retired service shells import their
    types from that missing package.
  * No retirement had been executed: neither retired
    service file nor the wing's shared micro/truth test
    file carried any archived / retired marking, and
    the three files stood at exactly TF-006's censused
    sizes (micro_rag_pipeline.py 4,198 B,
    truth_calibrator.py 3,157 B,
    test_micro_rag_and_truth.py 2,173 B).
  * The shipping tree holds no retired mechanism: no
    file under ``src/thoughtforge/`` outside the two
    verdict modules names MicroRAGPipeline or
    TruthCalibrator at all — the mechanisms were never
    adopted, so there is no shipping code to remove;
    the retired code is the wing's, and only the wing's.
  * All four neighbouring checkers stood clean live:
    micro_rag_verdict_problems(),
    truth_calibrator_verdict_problems(),
    bond_graph_problems(), persona_compiler_problems().

**Disposition: archived-in-place, for both mechanisms**
— the verdicts' disjunction ("removed or
archived-in-place") resolved by the estate's own filed
evidence, recorded in ``DISPOSITION_EVIDENCE`` below:

  * TF-140's and TF-142's checkers RE-READ the retired
    wing files live (the pipeline's weights / mode
    targets parsed by AST, the calibrator's markers /
    threshold read as text, the wing test's relational
    catch read as text). Deleting the files would break
    the verdicts' own proof — the estate's standing
    rule is that a filed checker fails on drift, and a
    retirement that falsifies its own verdict is no
    execution at all.
  * TF-142 files the disposition for its test in so
    many words: the wing's truth test "is archived with
    the calibrator, not silently dropped"; TF-149
    likewise files "retired code's tests archived with
    it". The shared wing test file
    (test_micro_rag_and_truth.py carries BOTH retired
    mechanisms' tests) is therefore archived-in-place
    with them.
  * The schemas the retired mechanisms were specified
    by (micro_context_packet / retrieval_item /
    eval_case / eval_result) are verdict EVIDENCE —
    TF-140/TF-142's spikes are checked against them
    live — and TF-006 files the 14 schemas as the
    wing's most complete artefact. They stay, unmarked:
    they specify shapes, they execute nothing.
  * TF-150 owns the wing's provenance; archived code
    keeps its history readable in place, deleted code
    does not.

Archiving-in-place here means, exactly: a retirement
banner filed as the constant ``BANNERS`` below is
prepended VERBATIM to each retired file, the original
file body stands byte-for-byte below it (every marker
the verdict checkers read is preserved — re-proven
live by this module's checker and by TF-140/TF-142's
own checkers staying clean), and the banner states the
verdict, the successor mechanism, and that the file is
not to be imported, revived, or built on without a new
roadmap word. Nothing is deleted. No shipping
behaviour changes. The verdict modules' Spike*
rebuilds stay exactly as filed — they are the
verdicts' measurement evidence, not the wing's code.

**The harvest** (``LESSON_ROWS``): per TF-006's rule
that a retire verdict must still harvest, the lessons
the two verdicts filed are written into the wing's
closing doc — which is this module's render,
``docs/architecture/WING_CLOSING.md`` (TF-150 completes
the wing's provenance there at its own slice; TF-148
owns the research_data fencing). Every lesson is filed
harvested-NOT-adopted: no packet presentation, rubric,
or eval scoring is adopted by this slice, and any
future use needs its own roadmap word.

The human-readable closing doc is
``docs/architecture/WING_CLOSING.md``, filed verbatim
as this module's render (regenerate with
``python3 -m thoughtforge.wing_retirements``).
Proof grade: FIXTURE/STATIC — banners + verdict
filings + the live tree re-measured on this VM; no
live model, never claimed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

WING_RETIREMENTS_FORMAT = "thoughtforge-wing-retirements/1"
WING_CLOSING_DOC = "docs/architecture/WING_CLOSING.md"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WING = _REPO_ROOT / "research_data" / "src" / "wyrdforge"
_SERVICES = _WING / "services"
_SCHEMAS = _WING / "schemas"
_WING_TESTS = _REPO_ROOT / "research_data" / "tests"
_DOCKET = _REPO_ROOT / "docs" / "audit" / "TF-016_BUG_AND_GAP_DOCKET.md"

#: The dispositions TF-144 may file. Removal is a real
#: word in the verdicts' disjunction; it is not the
#: disposition this estate's evidence supports (see
#: DISPOSITION_EVIDENCE), and the validator names any
#: other value.
DISPOSITIONS: tuple[str, ...] = ("archived-in-place",)

#: The retirement banner marker every archived file
#: opens with.
BANNER_MARKER = "# ARCHIVED-IN-PLACE - RETIRED by TF-144 (2026-11-02)"


def _banner(mechanism_lines: list[str]) -> str:
    lines = [BANNER_MARKER, "#"]
    lines.extend(f"# {line}" if line else "#"
                 for line in mechanism_lines)
    lines.append("# " + "-" * 66)
    return "\n".join(lines) + "\n"


_MICRO_BANNER = _banner([
    "Mechanism: wyrdforge micro-RAG retrieval",
    "(MicroRAGPipeline, services/micro_rag_pipeline.py).",
    "Verdict: TF-140 - RETIRE as a retrieval mechanism,",
    "decided by measurement on TF-043's fixtures",
    "(shipping defaults precision 0.8889 / recall 0.9630",
    "/ F1 0.9111 vs the micro-RAG spike's 0.3333 /",
    "0.8889 / 0.4749).",
    "Successor: the shipping KnowledgeForge (Part III)",
    "is the estate's sole retriever. No micro-RAG",
    "scoring or assembly code is adopted or merged.",
    "This file is archived in place as verdict evidence:",
    "TF-140's checker re-reads it live, and its body",
    "below stands byte-for-byte as filed. Do not import,",
    "revive, or build on it without a new roadmap word.",
    "Lessons harvested in docs/architecture/",
    "WING_CLOSING.md (TF-144).",
])

_TRUTH_BANNER = _banner([
    "Mechanism: wyrdforge truth calibrator",
    "(TruthCalibrator, services/truth_calibrator.py).",
    "Verdict: TF-142 - RETIRE as a calibration /",
    "enforcement mechanism, decided by measurement on",
    "TF-100's chaos corpus (Part V enforcement 29/38 vs",
    "the calibrator spike's 5/38; the spike passes all",
    "38 outputs and detects 0 of 33 broken ones).",
    "Successor: Part V's shipping enforcement (the",
    "TF-093 surface) is the estate's sole runtime",
    "truth law. No calibrator scoring is adopted or",
    "merged into the shipping tree.",
    "This file is archived in place as verdict evidence:",
    "TF-142's checker re-reads it live, and its body",
    "below stands byte-for-byte as filed. Do not import,",
    "revive, or build on it without a new roadmap word.",
    "Lessons harvested in docs/architecture/",
    "WING_CLOSING.md (TF-144).",
])

_TEST_BANNER = _banner([
    "The wing's tests for the two RETIRED mechanisms:",
    "test_micro_rag_assembles_packet (TF-140's subject)",
    "and",
    "test_truth_calibrator_flags_unsupported_exclusivity",
    "(TF-142's subject - the relational catch,",
    "archived with the calibrator per TF-142's",
    "filed condition, not silently dropped).",
    "Archived in place with the retired code (TF-149:",
    "retired code's tests archived with it): the file",
    "stands byte-for-byte below as verdict evidence -",
    "TF-142's checker re-reads the truth test's intent",
    "here live. These tests still cannot run: the",
    "wing's models package was never committed",
    "(TF-006 / D-WING-1). Do not revive without a new",
    "roadmap word.",
])

#: Archived file (repo-relative) -> the banner filed
#: verbatim at its head.
BANNERS: dict[str, str] = {
    "research_data/src/wyrdforge/services/micro_rag_pipeline.py":
        _MICRO_BANNER,
    "research_data/src/wyrdforge/services/truth_calibrator.py":
        _TRUTH_BANNER,
    "research_data/tests/test_micro_rag_and_truth.py":
        _TEST_BANNER,
}

#: Why archived-in-place, and not removal — the
#: estate's own filed evidence, in the open.
DISPOSITION_EVIDENCE: tuple[str, ...] = (
    "TF-140's and TF-142's checkers re-read the retired wing "
    "files live (weights / mode targets by AST, markers / "
    "threshold / the wing test's relational catch as text): "
    "deleting the files would falsify the verdicts' own "
    "proof, so removal is not an execution this estate can "
    "file honestly",
    "TF-142 files the test disposition in so many words: the "
    "wing's truth test 'is archived with the calibrator, not "
    "silently dropped'; TF-149 files the same rule for all "
    "retired code's tests",
    "the retired mechanisms' schemas are verdict evidence "
    "(the TF-140/TF-142 spikes are checked against them live) "
    "and TF-006 files the 14 schemas as the wing's most "
    "complete artefact — they stay unmarked, as shapes that "
    "execute nothing",
    "TF-150 owns the wing's provenance: archived code keeps "
    "its history readable in place; deleted code does not",
)


# ── The retirements, as literal data ─────────────────────────────────────────

RETIREMENT_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "micro_rag",
     "mechanism": "wyrdforge micro-RAG retrieval "
                  "(MicroRAGPipeline: score() + assemble())",
     "verdict_slice": "TF-140",
     "verdict": "RETIRE",
     "disposition": "archived-in-place",
     "wing_files": (
         "research_data/src/wyrdforge/services/micro_rag_pipeline.py",
         "research_data/tests/test_micro_rag_and_truth.py"),
     "schemas_kept": ("micro_context_packet", "retrieval_item"),
     "successor": "the shipping KnowledgeForge (Part III) — "
                  "the estate's sole retriever; no micro-RAG "
                  "scoring or assembly code adopted or merged",
     "measured": "shipping defaults precision 0.8889 / recall "
                 "0.9630 / F1 0.9111 vs the micro-RAG spike's "
                 "0.3333 / 0.8889 / 0.4749 on TF-043's fixtures "
                 "(MRR tied 1.0); the spike returns 45 items "
                 "across nine queries against the shipping "
                 "engine's 17, and five junk items on the "
                 "negative query where the shipping engine "
                 "returns nothing",
     "code_markers": (
         "from wyrdforge.models.micro_rag import",
         "def score", "def assemble", "MODE_TARGETS")},
    {"id": "truth_calibrator",
     "mechanism": "wyrdforge truth calibrator "
                  "(TruthCalibrator.evaluate())",
     "verdict_slice": "TF-142",
     "verdict": "RETIRE",
     "disposition": "archived-in-place",
     "wing_files": (
         "research_data/src/wyrdforge/services/truth_calibrator.py",
         "research_data/tests/test_micro_rag_and_truth.py"),
     "schemas_kept": ("eval_case", "eval_result"),
     "successor": "Part V's shipping enforcement (TF-093's "
                  "surface, TF-049's per-claim law, TF-094's "
                  "dispositions, TF-095's telling) — the "
                  "estate's sole runtime truth law; no "
                  "calibrator scoring adopted or merged",
     "measured": "Part V enforcement 29/38 as demanded vs the "
                 "calibrator spike's 5/38 on TF-100's chaos "
                 "corpus (clean controls only; rescued 0/12, "
                 "refused-side 0/21, broken outputs detected "
                 "0/33, mean average 3.0 — it passes all 38); "
                 "the records-fed packet run is row-for-row "
                 "identical to the naked run",
     "code_markers": (
         "from wyrdforge.models.evals import",
         "class TruthCalibrator", "def evaluate")},
)

#: Wing files the ADOPT verdicts own — TF-144 must
#: never banner them (pinned by the checker).
ADOPTED_WING_FILES: tuple[str, ...] = (
    "research_data/src/wyrdforge/services/bond_graph_service.py",
    "research_data/src/wyrdforge/services/persona_compiler.py",
    "research_data/src/wyrdforge/services/memory_store.py",
)

#: The shared wing test file's filed test names, one
#: per retired mechanism.
WING_TEST_MARKERS: tuple[str, ...] = (
    "def test_micro_rag_assembles_packet",
    "def test_truth_calibrator_flags_unsupported_exclusivity",
    "relational_safety.value == 0",
)


# ── The harvest: lessons for the closing doc ─────────────────────────────────

LESSON_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "typed_packet_sections",
     "source_verdict": "TF-140",
     "lesson": "the typed packet sections: a context packet "
               "filed as mode + goal + a TruthPacket header "
               "(must_be_true / open_unknowns / "
               "forbidden_assumptions) + six typed sections "
               "(canonical / recent / bond / symbolic / code "
               "/ contradiction), assembled in per-mode family "
               "order — a presentation shape over retrieved "
               "records that the shipping flat bundle does not "
               "have",
     "status": "harvested-not-adopted",
     "future": "any future packet presentation over shipping "
               "bundles needs its own roadmap word"},
    {"id": "explicit_token_budget",
     "source_verdict": "TF-140",
     "lesson": "the explicit token budget: a packet carries a "
               "stated budget (default 900) charged per item's "
               "filed token_cost — budgeting as a filed, "
               "checkable number rather than an implicit "
               "top-k (on TF-043's fixtures the budget never "
               "bound before the per-family cap did: every "
               "packet used exactly 400)",
     "status": "harvested-not-adopted",
     "future": "any future packet presentation over shipping "
               "bundles needs its own roadmap word"},
    {"id": "contradiction_injection",
     "source_verdict": "TF-140",
     "lesson": "contradiction injection: contradiction items "
               "are a first-class packet family, admitted by "
               "mode (factual_lookup / world_state / "
               "coding_task / repair_or_boundary) with a "
               "scoring penalty — contradictions presented to "
               "the model as filed items, not hidden from it",
     "status": "harvested-not-adopted",
     "future": "any future packet presentation over shipping "
               "bundles needs its own roadmap word"},
    {"id": "four_dimension_rubric",
     "source_verdict": "TF-142",
     "lesson": "the four-dimension rubric as SCORED eval "
               "dimensions — factual_integrity / "
               "uncertainty_honesty / relational_safety / "
               "exploit_resistance, each 0..3 with a rationale "
               "— an offline eval-scorecard design; "
               "uncertainty_honesty in particular names a leg "
               "Part V's runtime surface does not judge",
     "status": "harvested-not-adopted",
     "future": "any future eval scoring (a Part VII-style "
               "panel is the shape the estate already knows) "
               "needs its own roadmap word and must be built "
               "defect-free, not by reviving this scorer"},
    {"id": "truth_packet_shape",
     "source_verdict": "TF-142",
     "lesson": "the TruthPacket shape: must_be_true / "
               "open_unknowns / forbidden_assumptions filed "
               "per case, before the answer exists — the "
               "expectations an eval grades against are data "
               "authored in advance, not impressions after "
               "the fact",
     "status": "harvested-not-adopted",
     "future": "any future eval scoring needs its own "
               "roadmap word and must be built defect-free, "
               "not by reviving this scorer"},
    {"id": "eval_set_categories",
     "source_verdict": "TF-142",
     "lesson": "the eval-set categories + modes as a design: "
               "a filed eval set with named categories and "
               "strict (higher threshold + zero-tolerance) vs "
               "balanced modes — filed here as a DESIGN "
               "lesson only; the retired code implemented "
               "only the balanced threshold, which is one of "
               "its nine measured defects, not part of the "
               "lesson",
     "status": "harvested-not-adopted",
     "future": "any future eval scoring needs its own "
               "roadmap word and must be built defect-free, "
               "not by reviving this scorer"},
)


# ── Pure validators (the probes run against these) ───────────────────────────

def retirement_row_problems(row: Any) -> list[str]:
    """Judge one retirement filing, purely: verdict
    RETIRE, a filed disposition, a successor, wing
    files, kept schemas, a measured ground and code
    markers. Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["retirement: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("retirement: the row files no id")
    if not row.get("mechanism"):
        problems.append(
            f"retirement {row.get('id')}: no mechanism filed")
    if row.get("verdict") != "RETIRE":
        problems.append(
            f"retirement {row.get('id')}: verdict is "
            f"{row.get('verdict')!r}, not 'RETIRE'")
    if row.get("verdict_slice") not in ("TF-140", "TF-142"):
        problems.append(
            f"retirement {row.get('id')}: verdict slice is "
            f"{row.get('verdict_slice')!r}, not a RETIRE "
            "verdict slice (TF-140 / TF-142)")
    if row.get("disposition") not in DISPOSITIONS:
        problems.append(
            f"retirement {row.get('id')}: disposition is "
            f"{row.get('disposition')!r}, not one of "
            f"{DISPOSITIONS}")
    if not row.get("successor"):
        problems.append(
            f"retirement {row.get('id')}: no successor filed "
            "— a retirement must name what stands instead")
    if not row.get("wing_files"):
        problems.append(
            f"retirement {row.get('id')}: no wing files filed")
    if not row.get("schemas_kept"):
        problems.append(
            f"retirement {row.get('id')}: no kept schemas "
            "filed — the verdict evidence must be named")
    if not row.get("measured"):
        problems.append(
            f"retirement {row.get('id')}: no measured ground "
            "filed")
    if not row.get("code_markers"):
        problems.append(
            f"retirement {row.get('id')}: no code markers "
            "filed — the archived body cannot be verified")
    return problems


def lesson_row_problems(row: Any) -> list[str]:
    """Judge one harvested-lesson filing, purely:
    an id, a source RETIRE verdict, the lesson, the
    harvested-not-adopted status and a future path.
    Empty-safe."""
    if not isinstance(row, dict) or not row:
        return ["lesson: the row is empty"]
    problems: list[str] = []
    if not row.get("id"):
        problems.append("lesson: the row files no id")
    if row.get("source_verdict") not in ("TF-140", "TF-142"):
        problems.append(
            f"lesson {row.get('id')}: source verdict is "
            f"{row.get('source_verdict')!r}, not a RETIRE "
            "verdict (TF-140 / TF-142)")
    if not row.get("lesson"):
        problems.append(
            f"lesson {row.get('id')}: no lesson filed")
    if row.get("status") != "harvested-not-adopted":
        problems.append(
            f"lesson {row.get('id')}: status is "
            f"{row.get('status')!r} — TF-144 harvests, it "
            "does not adopt")
    if not row.get("future"):
        problems.append(
            f"lesson {row.get('id')}: no future path filed")
    return problems


def banner_problems(banner: Any, file_text: Any) -> list[str]:
    """Judge one archived file's banner state, purely:
    the banner opens with the filed marker and the
    file text carries it verbatim at its head.
    Empty-safe."""
    problems: list[str] = []
    if not isinstance(banner, str) or not banner:
        return ["banner: the banner is empty"]
    if not banner.startswith(BANNER_MARKER):
        problems.append(
            "banner: the banner does not open with the "
            "filed marker")
    if not isinstance(file_text, str) or not file_text:
        return problems + ["banner: the file text is empty"]
    if not file_text.startswith(banner):
        problems.append(
            "banner: the file does not carry the filed "
            "banner verbatim at its head")
    return problems


def disposition_problems(archived_paths: Any) -> list[str]:
    """Judge the archived set as a whole, purely:
    exactly the filed files, no duplicates, and no
    adopted mechanism's file among them. Empty-safe."""
    if not isinstance(archived_paths, (list, tuple)) \
            or not archived_paths:
        return ["disposition: the archived set is empty"]
    problems: list[str] = []
    paths = list(archived_paths)
    if len(paths) != len(set(paths)):
        problems.append(
            "disposition: the archived set files a path twice")
    filed = set(BANNERS)
    for path in paths:
        if path not in filed:
            problems.append(
                f"disposition: {path} is archived but not "
                "filed — TF-144 archives exactly the retired "
                "mechanisms' files")
        if path in ADOPTED_WING_FILES:
            problems.append(
                f"disposition: {path} belongs to an ADOPT "
                "verdict — it must not be retired")
    for path in filed:
        if path not in paths:
            problems.append(
                f"disposition: the filed archived file {path} "
                "is missing from the set")
    return problems


def archive_state_problems(state: Any, expected: Any) -> list[str]:
    """Judge a measured archive state against its
    expected filing, purely. Empty-safe."""
    if not isinstance(state, dict) or not state:
        return ["archive state: the state is empty"]
    if not isinstance(expected, dict) or not expected:
        return ["archive state: the expected filing is empty"]
    problems: list[str] = []
    for key, value in expected.items():
        if state.get(key) != value:
            problems.append(
                f"archive state: {key} measures "
                f"{state.get(key)!r}, not the filed {value!r}")
    return problems


# ── The live archive state ───────────────────────────────────────────────────

#: The filed expectation for the executed retirement —
#: deterministic counts and booleans only.
ARCHIVE_EXPECTED: dict[str, Any] = {
    "retirements": 2,
    "archived_files": 3,
    "files_present": 3,
    "files_bannered": 3,
    "code_markers_retained": True,
    "wing_test_markers_retained": True,
    "adopted_files_present": 3,
    "adopted_files_bannered": 0,
    "schemas_present": 14,
    "models_package_present": False,
    "shipping_stray_mechanisms": 0,
    "lessons": 6,
    "lessons_from_tf140": 3,
    "lessons_from_tf142": 3,
}

_SHIPPING_EXCLUDED = {
    "micro_rag_verdict.py", "truth_calibrator_verdict.py",
    "wing_retirements.py",
}
_SHIPPING_TOKENS = ("MicroRAGPipeline", "TruthCalibrator")


def _read_rel(rel: str) -> str | None:
    path = _REPO_ROOT / rel
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def run_archive_state() -> dict[str, Any]:
    """Measure the executed retirement on the live
    tree: banners in place verbatim, archived bodies
    retaining every filed code marker, the adopted
    wing files unbannered, the schemas whole, the
    shipping tree free of the retired mechanisms.
    Deterministic: counts and booleans only."""
    texts = {rel: _read_rel(rel) for rel in BANNERS}
    present = sum(1 for t in texts.values() if t is not None)
    bannered = sum(
        1 for rel, text in texts.items()
        if text is not None and text.startswith(BANNERS[rel]))
    markers_ok = True
    for row in RETIREMENT_ROWS:
        for rel in row["wing_files"]:
            if rel not in BANNERS:
                continue
            text = texts.get(rel) or ""
            if rel.endswith("test_micro_rag_and_truth.py"):
                if not all(m in text for m in WING_TEST_MARKERS):
                    markers_ok = False
            elif not all(m in text for m in row["code_markers"]):
                markers_ok = False
    test_text = texts.get(
        "research_data/tests/test_micro_rag_and_truth.py") or ""
    adopted_texts = {rel: _read_rel(rel) for rel in ADOPTED_WING_FILES}
    stray = 0
    src_root = _REPO_ROOT / "src" / "thoughtforge"
    for path in sorted(src_root.rglob("*.py")):
        if path.name in _SHIPPING_EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in _SHIPPING_TOKENS):
            stray += 1
    return {
        "retirements": len(RETIREMENT_ROWS),
        "archived_files": len(BANNERS),
        "files_present": present,
        "files_bannered": bannered,
        "code_markers_retained": markers_ok,
        "wing_test_markers_retained": all(
            m in test_text for m in WING_TEST_MARKERS),
        "adopted_files_present": sum(
            1 for t in adopted_texts.values() if t is not None),
        "adopted_files_bannered": sum(
            1 for t in adopted_texts.values()
            if t is not None and BANNER_MARKER in t),
        "schemas_present": len(list(_SCHEMAS.glob("*.json"))),
        "models_package_present": (_WING / "models").exists(),
        "shipping_stray_mechanisms": stray,
        "lessons": len(LESSON_ROWS),
        "lessons_from_tf140": sum(
            1 for r in LESSON_ROWS
            if r["source_verdict"] == "TF-140"),
        "lessons_from_tf142": sum(
            1 for r in LESSON_ROWS
            if r["source_verdict"] == "TF-142"),
    }


# ── The filed probes ─────────────────────────────────────────────────────────

RETIREMENT_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "clean_retirement_filing_passes",
     "kind": "clean",
     "detail": "the filed retirements + lessons + archived "
               "set pass their validators"},
    {"id": "retirement_without_a_successor_is_named",
     "kind": "row_no_successor",
     "detail": "a retirement row with its successor emptied "
               "is named"},
    {"id": "lesson_without_a_source_is_named",
     "kind": "lesson_no_source",
     "detail": "a lesson row with its source verdict "
               "emptied is named"},
    {"id": "banner_missing_from_file_is_named",
     "kind": "banner_missing",
     "detail": "a file text with the banner stripped is "
               "named against the filed banner"},
    {"id": "adopted_file_in_archived_set_is_named",
     "kind": "adopted_in_set",
     "detail": "an archived set carrying the bond graph "
               "service (an ADOPT verdict's file) is named"},
    {"id": "unknown_disposition_is_named",
     "kind": "unknown_disposition",
     "detail": "a retirement row filing a disposition "
               "outside the filed vocabulary is named"},
    {"id": "duplicate_archived_path_is_named",
     "kind": "duplicate_path",
     "detail": "an archived set filing one path twice is "
               "named"},
)


def run_retirement_probes() -> list[dict[str, Any]]:
    """Run the filed probes against the pure
    validators; each returns its verdict. Real
    filings are never perturbed — probes work on
    copies."""
    results: list[dict[str, Any]] = []
    for probe in RETIREMENT_PROBES:
        kind = probe["kind"]
        if kind == "clean":
            problems: list[str] = []
            for row in RETIREMENT_ROWS:
                problems += retirement_row_problems(row)
            for row in LESSON_ROWS:
                problems += lesson_row_problems(row)
            problems += disposition_problems(tuple(BANNERS))
            passed = problems == []
        elif kind == "row_no_successor":
            row = dict(RETIREMENT_ROWS[0], successor="")
            passed = bool(retirement_row_problems(row))
        elif kind == "lesson_no_source":
            row = dict(LESSON_ROWS[0], source_verdict="")
            passed = bool(lesson_row_problems(row))
        elif kind == "banner_missing":
            banner = BANNERS[
                "research_data/src/wyrdforge/services/"
                "micro_rag_pipeline.py"]
            stripped = "from __future__ import annotations\n"
            passed = bool(banner_problems(banner, stripped))
        elif kind == "adopted_in_set":
            paths = tuple(BANNERS) + (ADOPTED_WING_FILES[0],)
            passed = bool(disposition_problems(paths))
        elif kind == "unknown_disposition":
            row = dict(RETIREMENT_ROWS[0], disposition="deleted")
            passed = bool(retirement_row_problems(row))
        elif kind == "duplicate_path":
            paths = tuple(BANNERS) + (tuple(BANNERS)[0],)
            passed = bool(disposition_problems(paths))
        else:  # pragma: no cover — the checker names unknown kinds
            passed = False
        results.append({
            "id": probe["id"], "passed": passed,
            "detail": probe["detail"]})
    return results


# ── The readable render (the closing doc, verbatim) ──────────────────────────

def render_wing_closing_markdown() -> str:
    """Render the wing's closing doc deterministically
    — the exact text filed as
    docs/architecture/WING_CLOSING.md."""
    state = run_archive_state()
    lines: list[str] = []
    lines.append("# The Wyrdforge Wing — Closing Doc (TF-144 retirements)")
    lines.append("")
    lines.append(
        "GENERATED from the filed data in "
        "`src/thoughtforge/wing_retirements.py` — the filed "
        "doc IS the render: regenerate with "
        "`python3 -m thoughtforge.wing_retirements`. The "
        "checker (`wing_retirements_problems()`) re-derives "
        "every claim live from the tree and fails on any "
        "drift.")
    lines.append("")
    lines.append(f"Format: `{WING_RETIREMENTS_FORMAT}`")
    lines.append("Retirements executed: 2 (TF-140 micro-RAG, "
                 "TF-142 truth calibrator)")
    lines.append("Disposition: **archived-in-place** (both)")
    lines.append(f"Probes: {len(RETIREMENT_PROBES)}/"
                 f"{len(RETIREMENT_PROBES)}")
    lines.append("")
    lines.append("## 1. Measured start")
    lines.append("")
    lines.append(
        "Probed live on this tree before anything was built: "
        "the wing still cannot import — "
        "`research_data/src/wyrdforge/models` does not exist "
        "and both retired service shells import their types "
        "from that missing package (TF-006 / D-WING-1). No "
        "retirement had been executed: neither retired "
        "service file nor the wing's shared micro/truth test "
        "file carried any archived / retired marking, and "
        "the three files stood at exactly TF-006's censused "
        "sizes (micro_rag_pipeline.py 4,198 B / "
        "truth_calibrator.py 3,157 B / "
        "test_micro_rag_and_truth.py 2,173 B). The shipping "
        "tree holds no retired mechanism — no file under "
        "`src/thoughtforge/` outside the two verdict modules "
        "names MicroRAGPipeline or TruthCalibrator — so the "
        "retired code is the wing's, and only the wing's. "
        "The four neighbouring checkers (both verdicts, both "
        "TF-143 production landings) stood clean live.")
    lines.append("")
    lines.append("## 2. The retirements, executed")
    lines.append("")
    lines.append(
        "| Mechanism | Verdict | Disposition | Successor |")
    lines.append("|---|---|---|---|")
    for row in RETIREMENT_ROWS:
        lines.append(
            f"| {row['mechanism']} | {row['verdict_slice']} "
            f"{row['verdict']} | {row['disposition']} | "
            f"{row['successor']} |")
    lines.append("")
    for row in RETIREMENT_ROWS:
        lines.append(f"### {row['id']} ({row['verdict_slice']})")
        lines.append("")
        lines.append(f"Measured ground: {row['measured']}")
        lines.append("")
        lines.append("Archived files:")
        lines.append("")
        for rel in row["wing_files"]:
            lines.append(f"- `{rel}`")
        lines.append("")
        lines.append(
            "Schemas kept unmarked as verdict evidence: "
            + " / ".join(
                f"`{name}.schema.json`" for name in row["schemas_kept"]))
        lines.append("")
    lines.append(
        "Each archived file opens with the filed retirement "
        "banner verbatim (verdict, successor, "
        "do-not-revive terms); its original body stands "
        "byte-for-byte below the banner — every code marker "
        "the verdict checkers read is retained, re-proven "
        f"live: {state['files_bannered']}/{state['archived_files']} "
        "files bannered, code markers retained "
        f"{state['code_markers_retained']}, wing test markers "
        f"retained {state['wing_test_markers_retained']}.")
    lines.append("")
    lines.append("## 3. Why archived-in-place, not removed")
    lines.append("")
    for evidence in DISPOSITION_EVIDENCE:
        lines.append(f"- {evidence}")
    lines.append("")
    lines.append("## 4. Lessons harvested (the wing's closing lessons)")
    lines.append("")
    lines.append(
        "Per TF-006's rule that a retire verdict must still "
        "harvest. Every lesson is filed "
        "**harvested-not-adopted**: TF-144 adopts nothing.")
    lines.append("")
    for row in LESSON_ROWS:
        lines.append(
            f"- `{row['id']}` ({row['source_verdict']}): "
            f"{row['lesson']} — Future: {row['future']}.")
    lines.append("")
    lines.append("## 5. What was NOT retired, and NOT touched")
    lines.append("")
    lines.append(
        "- TF-139's bond graph (ADOPT_AS_LAYER) and TF-141's "
        "persona compiler (MERGE) were adopted in TF-143 and "
        "are NOT retired: their wing files "
        "(`bond_graph_service.py`, `persona_compiler.py`, "
        "and the shared `memory_store.py`) carry no "
        "retirement banner — pinned live by this doc's "
        f"checker (adopted files bannered: "
        f"{state['adopted_files_bannered']}).")
    lines.append(
        "- All 14 wing schemas stand and parse; the specs "
        "and design docs stand as provenance (TF-150); the "
        "verdict modules' Spike* rebuilds stand as the "
        "verdicts' measurement evidence — none of them is "
        "the wing's retired code.")
    lines.append(
        "- No shipping behaviour changes; no README.md "
        "anywhere was touched; the wing's models package "
        "was not resurrected.")
    lines.append("")
    lines.append("## 6. Docket")
    lines.append("")
    lines.append(
        "TF-144 discharges the retirements share of D-WING-1 "
        "(narrowed in the open in "
        "`docs/audit/TF-016_BUG_AND_GAP_DOCKET.md`). "
        "D-WING-1 itself stays OPEN for the TF-148 research "
        "hygiene and TF-149 wing-tests work it assigns — "
        "TF-144 is not the last Part VIII slice on the "
        "entry, and the TF-143 pattern holds: no new docket "
        "entry, totals do not move.")
    lines.append("")
    lines.append("## 7. The probes")
    lines.append("")
    for probe in RETIREMENT_PROBES:
        lines.append(f"- `{probe['id']}`: {probe['detail']}")
    lines.append("")
    lines.append("## 8. Filed limits")
    lines.append("")
    lines.append(
        "- Archive grade only: archiving-in-place fences the "
        "retired code with its verdict and successor filed "
        "at its head; it does not make the wing importable "
        "(nothing here resurrects `wyrdforge.models`) and "
        "the archived tests still cannot run (TF-149 owns "
        "the wing's test disposition).")
    lines.append(
        "- The lessons are harvested, NOT adopted: any "
        "future packet presentation or eval scoring needs "
        "its own roadmap word, defect-free (TF-142's nine "
        "defects stay filed against the calibrator).")
    lines.append(
        "- TF-148 owns the `research_data/` fencing and "
        "TF-150 the wing's full provenance in this closing "
        "doc's line; this doc closes the RETIREMENTS, not "
        "the wing's whole Part VIII account (TF-145.."
        "TF-152 stand).")
    lines.append(
        "- No live model was read; live behaviour is NOT "
        "PROVEN (TF-173 class), never claimed.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_wing_closing_doc() -> Path:
    """Write the render to its filed home."""
    doc = _REPO_ROOT / WING_CLOSING_DOC
    doc.write_text(render_wing_closing_markdown(), encoding="utf-8")
    return doc


def main(argv: Any = None) -> int:
    """The regeneration command's entry point:
    `python3 -m thoughtforge.wing_retirements`."""
    doc = write_wing_closing_doc()
    print(f"wing closing doc regenerated: -> {WING_CLOSING_DOC} ({doc})")
    return 0


# ── The live checker ─────────────────────────────────────────────────────────

def wing_retirements_problems() -> list[str]:
    """Re-derive TF-144's claims from the live tree;
    name any that stopped being true."""
    problems: list[str] = []

    # The filings themselves.
    row_ids = [r["id"] for r in RETIREMENT_ROWS]
    if len(row_ids) != len(set(row_ids)) or len(RETIREMENT_ROWS) != 2:
        problems.append("the RETIREMENT_ROWS filing is not two "
                        "unique retirements")
    for row in RETIREMENT_ROWS:
        problems.extend(retirement_row_problems(row))
    lesson_ids = [r["id"] for r in LESSON_ROWS]
    if len(lesson_ids) != len(set(lesson_ids)) or len(LESSON_ROWS) != 6:
        problems.append("the LESSON_ROWS filing is not six unique "
                        "lessons")
    for row in LESSON_ROWS:
        problems.extend(lesson_row_problems(row))
    problems.extend(disposition_problems(tuple(BANNERS)))

    # The verdicts still file RETIRE (cheap imports — the
    # verdicts' own checkers carry the deep proof).
    from thoughtforge.knowledge.micro_rag_verdict import (
        VERDICT as MICRO_VERDICT,
    )
    from thoughtforge.refinement.truth_calibrator_verdict import (
        VERDICT as TRUTH_VERDICT,
    )

    if MICRO_VERDICT.get("verdict") != "RETIRE":
        problems.append("verdicts: TF-140 no longer files RETIRE")
    if TRUTH_VERDICT.get("verdict") != "RETIRE":
        problems.append("verdicts: TF-142 no longer files RETIRE")

    # The banners stand verbatim; the bodies retain their
    # markers; the truth file still files no strict
    # threshold (TF-142's premise, preserved by the banner).
    for rel, banner in BANNERS.items():
        text = _read_rel(rel)
        if text is None:
            problems.append(f"archive: {rel} is missing")
            continue
        problems.extend(
            f"archive {rel}: {p}"
            for p in banner_problems(banner, text))
    for row in RETIREMENT_ROWS:
        for rel in row["wing_files"]:
            if rel.endswith("test_micro_rag_and_truth.py"):
                continue
            text = _read_rel(rel) or ""
            for marker in row["code_markers"]:
                if marker not in text:
                    problems.append(
                        f"archive: {rel} no longer retains the "
                        f"filed marker {marker!r} — the archived "
                        "body must stand as filed")
    truth_text = _read_rel(
        "research_data/src/wyrdforge/services/"
        "truth_calibrator.py") or ""
    if "2.7" in truth_text:
        problems.append(
            "archive: truth_calibrator.py now files a 2.7 "
            "threshold — the banner must not disturb TF-142's "
            "premise")

    # The measured state matches its filing (adopted files
    # unbannered, schemas whole, models still missing, no
    # stray retired mechanism in the shipping tree).
    problems.extend(archive_state_problems(
        run_archive_state(), ARCHIVE_EXPECTED))

    # The docket carries this slice's narrowing.
    docket_text = _DOCKET.read_text(encoding="utf-8")
    if "Narrowed by TF-144" not in docket_text:
        problems.append(
            "docket: D-WING-1 does not file TF-144's narrowing")

    # The probes all behave.
    probes = run_retirement_probes()
    if len(probes) != len(RETIREMENT_PROBES) \
            or not all(p["passed"] for p in probes):
        problems.append(
            f"the retirement probes do not all pass: "
            f"{[p['id'] for p in probes if not p['passed']]}")

    # The closing doc files the render verbatim.
    # TF-150 extension, in the open: TF-150 completes
    # the wing's provenance in this doc as its
    # section 9, appended below this render — so the
    # completed doc is exactly this render plus
    # TF-150's filed appendix, and both shapes'
    # prefix (this render, byte-for-byte) is pinned.
    doc = _REPO_ROOT / WING_CLOSING_DOC
    if not doc.exists():
        problems.append(f"the closing doc {WING_CLOSING_DOC} is missing")
    else:
        text = doc.read_text(encoding="utf-8")
        render = render_wing_closing_markdown()
        completed = None
        try:
            from thoughtforge.wing_provenance import (
                render_wing_closing_appendix_markdown,
            )

            completed = render + render_wing_closing_appendix_markdown()
        except Exception:
            completed = None
        if text != render and text != completed:
            problems.append(
                f"the filed doc {WING_CLOSING_DOC} is not the render "
                "verbatim, nor the render completed by TF-150's "
                "provenance appendix (regenerate with python3 -m "
                "thoughtforge.wing_retirements, or complete it with "
                "python3 -m thoughtforge.wing_provenance)")
    return problems


def wing_retirements_as_data() -> dict[str, Any]:
    """The whole execution as plain data, for consumers."""
    probes = run_retirement_probes()
    return {
        "format": WING_RETIREMENTS_FORMAT,
        "doc": WING_CLOSING_DOC,
        "retirements": [dict(r) for r in RETIREMENT_ROWS],
        "lessons": [dict(r) for r in LESSON_ROWS],
        "archived_files": sorted(BANNERS),
        "state": run_archive_state(),
        "measured": {
            "probes_passed": sum(1 for p in probes if p["passed"]),
            "probes_total": len(probes),
        },
        "probes": probes,
    }


__all__ = [
    "ADOPTED_WING_FILES",
    "ARCHIVE_EXPECTED",
    "BANNERS",
    "BANNER_MARKER",
    "DISPOSITIONS",
    "DISPOSITION_EVIDENCE",
    "LESSON_ROWS",
    "RETIREMENT_PROBES",
    "RETIREMENT_ROWS",
    "WING_CLOSING_DOC",
    "WING_RETIREMENTS_FORMAT",
    "WING_TEST_MARKERS",
    "archive_state_problems",
    "banner_problems",
    "disposition_problems",
    "lesson_row_problems",
    "main",
    "render_wing_closing_markdown",
    "retirement_row_problems",
    "run_archive_state",
    "run_retirement_probes",
    "wing_retirements_as_data",
    "wing_retirements_problems",
    "write_wing_closing_doc",
]

if __name__ == "__main__":
    raise SystemExit(main())
