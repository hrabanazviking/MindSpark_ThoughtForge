"""
TF-039 — Core docs: the architecture doc (TF-019) reconciled
with shipped behaviour; examples runnable.

The reconciliation lives as verifiable data in
``src/thoughtforge/cognition/core_docs.py``
(``RECONCILED_FINDINGS`` + ``findings_problems()``,
``POST_FREEZE_ADDITIONS`` + ``additions_problems()``,
``EXAMPLES`` + ``run_example()``), the reconciled docstring in
``core.py``, the corrected record comments in
``knowledge/models.py``, and the write-up
``docs/architecture/CORE_DOCS_RECONCILED.md`` (with
``CORE_LOOP.md`` carrying TF-039 reconciliation notes).

Pinned here:

  * every CORE_LOOP §8 finding F1–F8 has exactly one reconciled
    status (at TF-039: 5 closed, 2 ruled, 1 still open — F2,
    owned by TF-041, recorded open rather than hidden; TF-041
    has since closed F2 with evidence and the distribution pin
    below was updated by that slice with the reason recorded),
    and the live checker finds no stale evidence in either
    direction,
  * F4's fix is real: the core module docstring states the
    reconciled 7-step loop and the old 9-step tail is gone;
    the InputSketch comments name the router's true intent /
    tone / response-mode taxonomies (TF-020's observation),
  * the post-freeze minor additions (TF-035–TF-038) are all
    recorded and all exist live exactly as named, while the
    freeze itself stands untouched at 1.0.0,
  * the examples are runnable: every one executes against the
    real core and returns its measured result (knowledge-only
    honesty, a stub-backend turn, a streamed turn, a trace
    export, a seeded replay reproducing TF-036's exact first
    turn id), and every example source is printed verbatim in
    the TF-039 doc, so doc and code cannot drift apart.

Proof grade: FIXTURE generation sources on this VM (stub
backend, knowledge-only cores), as with TF-019–TF-038.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from thoughtforge.cognition import core as core_module
from thoughtforge.cognition.api_surface import API_VERSION, verify_api_surface
from thoughtforge.cognition.core_docs import (
    EXAMPLES,
    POST_FREEZE_ADDITIONS,
    RECONCILED_FINDINGS,
    additions_problems,
    example_names,
    findings_problems,
    run_example,
)
from thoughtforge.cognition.scaffold_registry import scaffold_registry
from thoughtforge.cognition.traces import TRACE_SECTIONS
from thoughtforge.knowledge.models import CognitionScaffold, InputSketch
from thoughtforge.refinement import salvage as salvage_module

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_LOOP_DOC = REPO_ROOT / "docs" / "architecture" / "CORE_LOOP.md"
RECONCILED_DOC = (
    REPO_ROOT / "docs" / "architecture" / "CORE_DOCS_RECONCILED.md"
)
FREEZE_DOC = REPO_ROOT / "docs" / "architecture" / "CORE_API_FREEZE_V1.md"
MODELS_SRC = REPO_ROOT / "src" / "thoughtforge" / "knowledge" / "models.py"

FERRY_TEXT = (
    "The River Sarn ferry crossing takes eleven minutes in calm "
    "water between the willow narrows and the mill landing, and "
    "the lantern-keeper checks the river level each morning."
)


# ── The findings record ──────────────────────────────────────────────────────


class TestFindingsRecord:
    def test_covers_f1_to_f8_exactly(self) -> None:
        assert [row["finding"] for row in RECONCILED_FINDINGS] == [
            f"F{i}" for i in range(1, 9)
        ]

    def test_status_distribution_is_honest(self) -> None:
        by_status: dict[str, list[str]] = {}
        for row in RECONCILED_FINDINGS:
            by_status.setdefault(row["status"], []).append(row["finding"])
        # TF-041 superseded this pin with reason: at TF-039 the
        # distribution was 5 closed / 2 ruled / 1 open (F2);
        # TF-041 landed the F2 fix and flipped F2's record to
        # closed with evidence, exactly as TF-039's checker
        # instructed ("if TF-041 fixed the gate, flip F2's
        # status"). No finding stands open after TF-041.
        assert sorted(by_status["closed"]) == ["F1", "F2", "F3", "F4", "F7", "F8"]
        assert sorted(by_status["ruled"]) == ["F5", "F6"]
        assert by_status.get("open", []) == []
        owner = {r["finding"]: r["owner"] for r in RECONCILED_FINDINGS}
        assert owner["F2"] == "TF-041"

    def test_live_evidence_matches_the_record(self) -> None:
        assert findings_problems() == []

    def test_f7_prompt_text_stays_removed(self) -> None:
        fields = {f.name for f in dataclasses.fields(CognitionScaffold)}
        assert "prompt_text" not in fields

    def test_f6_both_thresholds_stand_at_ruled_values(self) -> None:
        assert core_module._MIN_FRAGMENT_SCORE == 0.25
        assert salvage_module._MIN_FRAGMENT_SCORE == 0.54

    def test_f8_no_intent_repeats_a_candidate_mode(self) -> None:
        for entry in scaffold_registry().values():
            modes = list(entry.candidate_modes)
            assert len(modes) == len(set(modes))
        brainstorming = scaffold_registry()["brainstorming"]
        assert list(brainstorming.candidate_modes) == [
            "brainstorm_distinct",
            "practical",
        ]


# ── F4: the docstring and record comments tell the truth ─────────────────────


class TestDocstringReconciled:
    def test_core_docstring_states_the_reconciled_loop(self) -> None:
        doc = core_module.__doc__ or ""
        for step in (
            "ROUTE",
            "RETRIEVE",
            "SCAFFOLD",
            "GENERATE",
            "FRAGMENT",
            "COMPOSE",
            "QUALITY GATE",
        ):
            assert step in doc
        # The old 9-step story (prompt-building and enforcement as
        # top-level steps, fragment scoring omitted) is gone.
        assert "FinalResponseRecord assembly" not in doc
        assert "TurboQuantEngine → draft generation" not in doc

    def test_core_docstring_names_the_three_drivers(self) -> None:
        doc = core_module.__doc__ or ""
        for driver in ("think()", "think_stream()", "resume_turn()"):
            assert driver in doc

    def test_input_sketch_comments_name_the_true_taxonomies(self) -> None:
        src = MODELS_SRC.read_text(encoding="utf-8")
        sketch = src[src.index("class InputSketch"):]
        sketch = sketch[: sketch.index("class MemoryActivationBundle")]
        for intent in (
            "technical_debugging",
            "technical_spec_request",
            "editing_refinement",
            "brainstorming",
            "naming_or_branding",
            "planning_request",
            "emotional_support",
            "factual_query",
            "light_conversation",
        ):
            assert intent in sketch
        for mode in (
            "structured_technical",
            "concise_editorial",
            "creative_brainstorm",
            "calm_supportive",
            "factual_direct",
            "light_natural_chat",
        ):
            assert mode in sketch
        # The drifted examples TF-020 flagged are gone.
        assert '"question", "command"' not in sketch
        assert "technical | supportive | creative" not in sketch

    def test_input_sketch_defaults_unchanged(self) -> None:
        # Comments were reconciled; construction behaviour was not
        # (the freeze's record contract stands).
        sketch = InputSketch(raw_text="hello")
        assert sketch.intent == ""
        assert sketch.response_mode == "conversational"
        assert sketch.retrieval_path == "hybrid"


# ── The post-freeze additions record ─────────────────────────────────────────


class TestPostFreezeAdditions:
    def test_all_additions_exist_live_as_recorded(self) -> None:
        assert additions_problems() == []

    def test_freeze_itself_untouched(self) -> None:
        assert API_VERSION == "1.0.0"
        assert verify_api_surface() == []

    def test_additions_named_with_their_slices(self) -> None:
        by_name = {row["name"]: row for row in POST_FREEZE_ADDITIONS}
        assert by_name["determinism_mode"]["added_by"] == "TF-036"
        assert by_name["last_turn_trace"]["added_by"] == "TF-038"
        modules = {
            row["name"]: row["added_by"]
            for row in POST_FREEZE_ADDITIONS
            if row["kind"] == "module"
        }
        assert modules == {
            "thoughtforge.cognition.determinism": "TF-036",
            "thoughtforge.cognition.concurrency": "TF-035",
            "thoughtforge.cognition.small_model_floor": "TF-037",
            "thoughtforge.cognition.traces": "TF-038",
        }

    def test_freeze_doc_carries_the_record(self) -> None:
        text = FREEZE_DOC.read_text(encoding="utf-8")
        assert "TF-039" in text
        for row in POST_FREEZE_ADDITIONS:
            assert row["name"].rpartition(".")[2] in text


# ── The examples are runnable ────────────────────────────────────────────────


class TestRunnableExamples:
    def test_example_names(self) -> None:
        assert example_names() == (
            "knowledge_only_turn",
            "stub_backend_turn",
            "streaming_turn",
            "trace_export",
            "seeded_replay",
        )

    def test_examples_are_synthetic_and_temp_only(self) -> None:
        canon_name = "Sig" + "rid"  # TF-017 estate guard: assembled
        for source in EXAMPLES.values():
            assert "tempfile" in source
            assert canon_name not in source
            assert "pictures" + "_of_" not in source

    def test_knowledge_only_example(self) -> None:
        result = run_example("knowledge_only_turn")
        assert result == {
            "mode": "knowledge_only",
            "text": "The forge has no knowledge relevant to this query.",
            "tokens": 9,
        }

    def test_stub_backend_example(self) -> None:
        result = run_example("stub_backend_turn")
        assert result["mode"] == "full"
        assert result["candidates"] == 1
        assert result["salvage_path"] == "best_draft"
        assert result["text"] == FERRY_TEXT

    def test_streaming_example(self) -> None:
        result = run_example("streaming_turn")
        kinds = result["kinds"]
        assert kinds[0] == "turn_started"
        assert kinds[-1] == "final"
        assert kinds.count("final") == 1
        assert "candidate_start" in kinds
        assert "candidate_end" in kinds
        assert result["final_mode"] == "full"

    def test_trace_example(self) -> None:
        result = run_example("trace_export")
        assert result["format"] == "thoughtforge-turn-trace/1"
        assert result["problems"] == []
        assert result["final_mode"] == "knowledge_only"
        for section in TRACE_SECTIONS:
            assert section in result["sections"]

    def test_seeded_replay_example(self) -> None:
        result = run_example("seeded_replay")
        assert result["same_turn_id"] is True
        # Cross-pin: TF-036 measured this exact first seeded id.
        assert result["turn_id"] == "turn_3881d6d8"

    def test_unknown_example_raises(self) -> None:
        with pytest.raises(KeyError):
            run_example("no_such_example")

    def test_doc_prints_every_example_verbatim(self) -> None:
        text = RECONCILED_DOC.read_text(encoding="utf-8")
        for name, source in EXAMPLES.items():
            assert name in text
            assert source in text


# ── The architecture doc carries the reconciliation ──────────────────────────


class TestCoreLoopDocReconciled:
    def test_core_loop_doc_marks_every_finding(self) -> None:
        text = CORE_LOOP_DOC.read_text(encoding="utf-8")
        assert "TF-039 reconciliation" in text
        for row in RECONCILED_FINDINGS:
            assert row["finding"] in text
        # The one open finding is named as open in the doc too.
        assert "F2" in text and "TF-041" in text
