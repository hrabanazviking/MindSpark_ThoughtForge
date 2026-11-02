"""
TF-144 — Retirements executed: each RETIRE verdict's
code removed or archived-in-place, lessons written
into the wing's closing doc.

The proof lives as verifiable data in
``src/thoughtforge/wing_retirements.py``
(``RETIREMENT_ROWS``, ``LESSON_ROWS``, ``BANNERS``,
``DISPOSITION_EVIDENCE``, ``RETIREMENT_PROBES``,
checked by ``wing_retirements_problems()``); the
wing's closing doc ``docs/architecture/WING_CLOSING.md``
is filed verbatim as the module's render.

Pinned here:

  * the execution: both RETIRE verdicts (TF-140
    micro-RAG, TF-142 truth calibrator) executed as
    archived-in-place — the filed banner verbatim at
    the head of the two wing service files + the
    shared wing test file, each original body standing
    byte-for-byte below it (insertions only vs HEAD),
  * the disposition's evidence: removal would falsify
    TF-140/TF-142's live checkers, TF-142/TF-149 file
    retired tests archived with their code, the
    schemas stay as verdict evidence, TF-150 owns
    provenance,
  * the harvest: six lessons (three per verdict) in
    the closing doc, every one harvested-not-adopted,
  * the fence: TF-139/TF-141's adopted wing files
    carry no banner, all 14 schemas stand, the models
    package stays missing, no shipping file outside
    the verdict modules names a retired mechanism,
    both verdict checkers still clean live,
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live model was
read, never claimed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from thoughtforge.wing_retirements import (
    ADOPTED_WING_FILES,
    ARCHIVE_EXPECTED,
    BANNERS,
    BANNER_MARKER,
    DISPOSITIONS,
    DISPOSITION_EVIDENCE,
    LESSON_ROWS,
    RETIREMENT_PROBES,
    RETIREMENT_ROWS,
    WING_CLOSING_DOC,
    WING_TEST_MARKERS,
    archive_state_problems,
    banner_problems,
    disposition_problems,
    lesson_row_problems,
    render_wing_closing_markdown,
    retirement_row_problems,
    run_archive_state,
    run_retirement_probes,
    wing_retirements_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_FILE = "research_data/src/wyrdforge/services/micro_rag_pipeline.py"
TRUTH_FILE = "research_data/src/wyrdforge/services/truth_calibrator.py"
TEST_FILE = "research_data/tests/test_micro_rag_and_truth.py"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ── The execution ────────────────────────────────────────────────────────────

class TestRetirementsExecuted:
    def test_problems_empty(self) -> None:
        assert wing_retirements_problems() == []

    def test_two_retirements_both_retire_both_archived(self) -> None:
        assert [r["id"] for r in RETIREMENT_ROWS] == [
            "micro_rag", "truth_calibrator"]
        assert [r["verdict_slice"] for r in RETIREMENT_ROWS] == [
            "TF-140", "TF-142"]
        for row in RETIREMENT_ROWS:
            assert row["verdict"] == "RETIRE"
            assert row["disposition"] == "archived-in-place"
            assert retirement_row_problems(row) == []

    def test_successors_are_the_sole_mechanisms(self) -> None:
        micro, truth = RETIREMENT_ROWS
        assert "KnowledgeForge" in micro["successor"]
        assert "sole retriever" in micro["successor"]
        assert "TF-093" in truth["successor"]
        assert "sole runtime truth law" in truth["successor"]

    def test_archived_set_is_exactly_three_files(self) -> None:
        assert set(BANNERS) == {MICRO_FILE, TRUTH_FILE, TEST_FILE}
        assert disposition_problems(tuple(BANNERS)) == []

    def test_each_archived_file_opens_with_its_banner(self) -> None:
        for rel, banner in BANNERS.items():
            text = _read(rel)
            assert text.startswith(banner)
            assert banner.startswith(BANNER_MARKER)
            assert banner_problems(banner, text) == []

    def test_bodies_stand_byte_for_byte_below_the_banners(self) -> None:
        # git is the witness, in both tree states: before
        # banking HEAD holds the unbannered original (the
        # body below the banner must equal it exactly);
        # after banking HEAD holds the bannered file (the
        # body must equal HEAD's body). Either way the
        # original body is never edited.
        for rel, banner in BANNERS.items():
            head = subprocess.run(
                ["git", "show", f"HEAD:{rel}"],
                cwd=REPO_ROOT, capture_output=True, text=True,
                check=True).stdout
            body = _read(rel)[len(banner):]
            expected = head[len(banner):] \
                if head.startswith(banner) else head
            assert body == expected

    def test_archived_bodies_retain_their_code_markers(self) -> None:
        for row in RETIREMENT_ROWS:
            service = next(
                rel for rel in row["wing_files"]
                if rel.startswith("research_data/src/"))
            text = _read(service)
            for marker in row["code_markers"]:
                assert marker in text

    def test_wing_test_markers_retained(self) -> None:
        text = _read(TEST_FILE)
        for marker in WING_TEST_MARKERS:
            assert marker in text

    def test_truth_banner_disturbs_no_tf142_premise(self) -> None:
        # TF-142's checker: no strict threshold in the file.
        assert "2.7" not in _read(TRUTH_FILE)

    def test_archive_state_matches_its_filing(self) -> None:
        state = run_archive_state()
        assert state == ARCHIVE_EXPECTED
        assert archive_state_problems(state, ARCHIVE_EXPECTED) == []


# ── The disposition ──────────────────────────────────────────────────────────

class TestDisposition:
    def test_dispositions_vocabulary_is_archive_only(self) -> None:
        assert DISPOSITIONS == ("archived-in-place",)

    def test_disposition_evidence_is_four_filed_lines(self) -> None:
        assert len(DISPOSITION_EVIDENCE) == 4
        joined = " ".join(DISPOSITION_EVIDENCE)
        assert "re-read the retired wing files live" in joined
        assert "archived with the calibrator" in joined
        assert "TF-150" in joined

    def test_schemas_kept_are_named_per_mechanism(self) -> None:
        micro, truth = RETIREMENT_ROWS
        assert micro["schemas_kept"] == (
            "micro_context_packet", "retrieval_item")
        assert truth["schemas_kept"] == ("eval_case", "eval_result")
        schemas = REPO_ROOT / "research_data/src/wyrdforge/schemas"
        assert len(list(schemas.glob("*.json"))) == 14


# ── The harvest ──────────────────────────────────────────────────────────────

class TestHarvest:
    def test_six_lessons_three_per_verdict(self) -> None:
        assert len(LESSON_ROWS) == 6
        assert [r["id"] for r in LESSON_ROWS] == [
            "typed_packet_sections", "explicit_token_budget",
            "contradiction_injection", "four_dimension_rubric",
            "truth_packet_shape", "eval_set_categories"]
        for row in LESSON_ROWS:
            assert lesson_row_problems(row) == []

    def test_every_lesson_is_harvested_not_adopted(self) -> None:
        for row in LESSON_ROWS:
            assert row["status"] == "harvested-not-adopted"
            assert "roadmap word" in row["future"]

    def test_lessons_are_in_the_closing_doc(self) -> None:
        doc = _read(WING_CLOSING_DOC)
        for row in LESSON_ROWS:
            assert row["id"] in doc and row["lesson"][:40] in doc


# ── The fence: what was NOT retired ──────────────────────────────────────────

class TestFence:
    def test_adopted_wing_files_carry_no_banner(self) -> None:
        for rel in ADOPTED_WING_FILES:
            text = _read(rel)
            assert BANNER_MARKER not in text
            assert not text.startswith("# ARCHIVED")

    def test_models_package_still_missing(self) -> None:
        assert not (
            REPO_ROOT / "research_data/src/wyrdforge/models").exists()

    def test_no_stray_retired_mechanism_in_shipping(self) -> None:
        assert run_archive_state()["shipping_stray_mechanisms"] == 0

    def test_both_verdict_checkers_still_clean(self) -> None:
        from thoughtforge.knowledge.micro_rag_verdict import (
            micro_rag_verdict_problems,
        )
        from thoughtforge.refinement.truth_calibrator_verdict import (
            truth_calibrator_verdict_problems,
        )

        assert micro_rag_verdict_problems() == []
        assert truth_calibrator_verdict_problems() == []

    def test_adopted_production_checkers_still_clean(self) -> None:
        from thoughtforge.cognition.persona_compiler import (
            persona_compiler_problems,
        )
        from thoughtforge.knowledge.bond_graph import (
            bond_graph_problems,
        )

        assert bond_graph_problems() == []
        assert persona_compiler_problems() == []


# ── Probes + validators + doc ────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_retirement_probes()
        assert len(probes) == len(RETIREMENT_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert retirement_row_problems(None)
        assert lesson_row_problems(None)
        assert banner_problems(None, None)
        assert disposition_problems(None)
        assert archive_state_problems(None, None)

    def test_doc_matches_render(self) -> None:
        # TF-150 extension, in the open: the closing doc
        # is TF-144's render, completed by TF-150's
        # provenance appendix as its section 9 — the
        # render stands byte-for-byte as the prefix.
        from thoughtforge.wing_provenance import (
            render_wing_closing_appendix_markdown,
        )

        doc = REPO_ROOT / WING_CLOSING_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_wing_closing_markdown() + \
            render_wing_closing_appendix_markdown()

    def test_closing_doc_names_both_verdicts_and_lessons(self) -> None:
        doc = _read(WING_CLOSING_DOC)
        assert "TF-140" in doc and "TF-142" in doc
        assert "archived-in-place" in doc
        assert "harvested-not-adopted" in doc
        assert "D-WING-1" in doc
