"""
TF-151 — Panel re-run: TF-132 + persona suites
re-measured with the verdicts landed; deltas
published.

The proof lives as verifiable data in
``src/thoughtforge/panel_rerun.py``
(``RERUN_ROWS``, ``VERDICT_ROWS``, ``DELTA_ROWS``,
``RERUN_PROBES``, checked by
``panel_rerun_problems()``); the doc
``docs/architecture/PANEL_RERUN.md`` is filed
verbatim as the module's render.

Pinned here:

  * the three re-run panels (TF-132 WYRD panel,
    TF-122 persona suite, and the
    persona-consistency scorer baseline under
    TF-122's voice axis) re-measure live exactly
    at their pinned harness values — 19 metrics,
    19 unmoved, 0 moved,
  * both panel verdicts PASS before and after,
  * the seven Part VIII verdicts in force
    (TF-139 ADOPT_AS_LAYER, TF-140 RETIRE,
    TF-141 MERGE, TF-142 RETIRE, TF-143
    adoptions, TF-146 bridge, TF-147 cards) are
    consumed by NO re-run panel — proven live by
    source inspection (no adopted-code token in
    any re-run measurer's source),
  * no harness baseline added or re-pinned and
    no docket touch: TF-016 totals unchanged
    137 — 32 OPEN / 1 OPEN-ENVIRONMENT /
    100 FIXED / 4 CLOSED-BY-VERDICT,
  * the probes: 7/7 against the pure validators,
  * the rerun doc matches its render verbatim.

All fixtures synthetic (TF-017); no canon —
only synthetic fixture scores are filed.
Proof grade: FIXTURE/STATIC — no live model
was read, never claimed.
"""

from __future__ import annotations

import re
from pathlib import Path

from thoughtforge.panel_rerun import (
    ADOPTED_CODE_TOKENS,
    DELTA_OUTCOMES,
    DELTA_ROWS,
    PANEL_RERUN_DOC,
    PANEL_RERUN_FORMAT,
    RERUN_EXPECTED,
    RERUN_PROBES,
    RERUN_ROWS,
    VERDICT_ROWS,
    delta_row_problems,
    doc_text_problems,
    panel_rerun_problems,
    render_panel_rerun_markdown,
    rerun_row_problems,
    rerun_state_problems,
    run_rerun_probes,
    run_rerun_state,
    verdict_row_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class TestFiling:
    def test_problems_empty(self) -> None:
        assert panel_rerun_problems() == []

    def test_format_is_versioned(self) -> None:
        assert PANEL_RERUN_FORMAT == "thoughtforge-panel-rerun/1"

    def test_rerun_rows_are_three_unique(self) -> None:
        assert len(RERUN_ROWS) == 3
        assert {r["id"] for r in RERUN_ROWS} == {
            "wyrd_panel", "persona_suite", "persona_consistency"}
        for row in RERUN_ROWS:
            assert rerun_row_problems(row) == []
            assert row["consumes_adopted_code"] is False
            assert row["verdict_before"] == "PASS"
            assert row["verdict_after"] == "PASS"

    def test_verdict_rows_are_seven_unconsumed(self) -> None:
        assert len(VERDICT_ROWS) == 7
        assert [r["slice"] for r in VERDICT_ROWS] == [
            "TF-139", "TF-140", "TF-141", "TF-142",
            "TF-143", "TF-146", "TF-147"]
        for row in VERDICT_ROWS:
            assert verdict_row_problems(row) == []
            assert row["consumed_by_rerun"] is False
        by_slice = {r["slice"]: r["verdict"] for r in VERDICT_ROWS}
        assert by_slice["TF-139"] == "ADOPT_AS_LAYER"
        assert by_slice["TF-140"] == "RETIRE"
        assert by_slice["TF-141"] == "MERGE"
        assert by_slice["TF-142"] == "RETIRE"

    def test_delta_rows_are_nineteen_all_unmoved(self) -> None:
        assert len(DELTA_ROWS) == 19
        for row in DELTA_ROWS:
            assert delta_row_problems(row) == []
            assert row["outcome"] == "unmoved"
            assert row["delta"] == 0
            assert row["pinned"] == row["remeasured"]
            assert row["outcome"] in DELTA_OUTCOMES
        by_panel: dict[str, int] = {}
        for row in DELTA_ROWS:
            by_panel[row["panel"]] = by_panel.get(row["panel"], 0) + 1
        assert by_panel == {
            "wyrd_panel": 7, "persona_suite": 5,
            "persona_consistency": 7}


class TestLiveRerun:
    def test_wyrd_panel_remeasures_at_its_pin(self) -> None:
        from benchmarks.harness import measure_wyrd_panel_baseline
        from benchmarks.wyrd_panel import run_wyrd_panel

        live = measure_wyrd_panel_baseline()
        for row in [r for r in DELTA_ROWS if r["panel"] == "wyrd_panel"]:
            assert live[row["metric"]] == row["pinned"] == row["remeasured"]
        result = run_wyrd_panel()
        assert result.total_cases == 18
        assert result.overall_score == 0.5555555555555556
        assert result.verdict == "PASS"

    def test_persona_suite_remeasures_at_its_pin(self) -> None:
        from benchmarks.harness import measure_persona_suite_baseline
        from benchmarks.persona_suite import run_persona_suite

        live = measure_persona_suite_baseline()
        for row in [r for r in DELTA_ROWS if r["panel"] == "persona_suite"]:
            assert live[row["metric"]] == row["pinned"] == row["remeasured"]
        result = run_persona_suite()
        assert result.total_turns == 108
        assert result.overall_score == 0.8453703703703704
        assert result.verdict == "PASS"

    def test_persona_consistency_remeasures_at_its_pin(self) -> None:
        from benchmarks.harness import measure_persona_baseline

        live = measure_persona_baseline()
        for row in [r for r in DELTA_ROWS
                    if r["panel"] == "persona_consistency"]:
            assert live[row["metric"]] == row["pinned"] == row["remeasured"]
        assert live["consistency_score"] == 0.7166666666666667

    def test_no_rerun_source_consumes_adopted_code(self) -> None:
        for row in RERUN_ROWS:
            text = _read(row["source_file"])
            for token in ADOPTED_CODE_TOKENS:
                assert token not in text, (row["id"], token)

    def test_verdict_modules_still_file_their_verdicts(self) -> None:
        from thoughtforge.cognition.persona_compiler_verdict import (
            VERDICT as persona_verdict,
        )
        from thoughtforge.knowledge.bond_graph_verdict import (
            VERDICT as bond_verdict,
        )
        from thoughtforge.knowledge.micro_rag_verdict import (
            VERDICT as micro_verdict,
        )
        from thoughtforge.refinement.truth_calibrator_verdict import (
            VERDICT as truth_verdict,
        )

        assert bond_verdict["verdict"] == "ADOPT_AS_LAYER"
        assert micro_verdict["verdict"] == "RETIRE"
        assert persona_verdict["verdict"] == "MERGE"
        assert truth_verdict["verdict"] == "RETIRE"

    def test_neighbouring_checkers_still_clean(self) -> None:
        from benchmarks.harness import baseline_problems
        from benchmarks.persona_suite import persona_suite_problems
        from benchmarks.wyrd_panel import wyrd_panel_problems
        from thoughtforge.cognition.persona_compiler import (
            persona_compiler_problems,
        )
        from thoughtforge.knowledge.bond_graph import bond_graph_problems

        assert wyrd_panel_problems() == []
        assert persona_suite_problems() == []
        assert bond_graph_problems() == []
        assert persona_compiler_problems() == []
        assert baseline_problems() == []


class TestStateAndProbes:
    def test_state_matches_its_filing(self) -> None:
        state = run_rerun_state()
        assert state == RERUN_EXPECTED
        assert rerun_state_problems(state, RERUN_EXPECTED) == []

    def test_probes_all_pass(self) -> None:
        probes = run_rerun_probes()
        assert len(probes) == len(RERUN_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert rerun_row_problems(None)
        assert verdict_row_problems(None)
        assert delta_row_problems(None)
        assert rerun_state_problems(None, None)
        assert doc_text_problems(None, None, "x")

    def test_each_probe_shape_is_named_by_the_validator(self) -> None:
        row = dict(DELTA_ROWS[0], remeasured=19, delta=1,
                   outcome="unmoved")
        assert delta_row_problems(row)
        row = dict(DELTA_ROWS[0], delta=99)
        assert delta_row_problems(row)
        row = dict(RERUN_ROWS[0], consumes_adopted_code=True)
        assert rerun_row_problems(row)
        row = dict(VERDICT_ROWS[0], consumed_by_rerun=True)
        assert verdict_row_problems(row)
        assert doc_text_problems(
            render_panel_rerun_markdown() + "x",
            render_panel_rerun_markdown(), "panel rerun doc")


class TestDocsAndDocket:
    def test_rerun_doc_matches_render(self) -> None:
        assert _read(PANEL_RERUN_DOC) == render_panel_rerun_markdown()

    def test_doc_files_the_delta_headline(self) -> None:
        doc = _read(PANEL_RERUN_DOC)
        assert "19 unmoved, 0" in doc
        assert "NOT PROVEN" in doc
        assert "TF-132" in doc and "TF-122" in doc
        assert "0.5555555555555556" in doc
        assert "0.8453703703703704" in doc

    def test_no_new_harness_baseline(self) -> None:
        # TF-138 precedent: a re-measurement slice adds
        # no baseline — the pins compared against stand.
        from benchmarks.harness import BASELINES

        ids = [b["id"] for b in BASELINES]
        assert len(ids) == 18
        assert "panel_rerun_synthetic_v1" not in ids

    def test_docket_untouched_totals_unchanged(self) -> None:
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        normalized = re.sub(r"\s+", " ", docket)
        assert "137 entries — 32 OPEN, 1 OPEN-ENVIRONMENT, " \
            "100 FIXED" in normalized
        assert "4 CLOSED-BY-VERDICT" in normalized

    def test_fixtures_are_synthetic_only(self) -> None:
        for row in DELTA_ROWS + VERDICT_ROWS + RERUN_ROWS:
            assert "veyrunn" not in str(row).lower()
