"""
TF-145 — WYRD boundary doc: ThoughtForge vs the
WYRD Protocol — what each owns (cognition vs
world-model), the interface between them, written
for the Second Life/Hermes consumers.

The proof lives as verifiable data in
``src/thoughtforge/wyrd_boundary.py``
(``OWNERSHIP_ROWS``, ``INTERFACE_ROWS``,
``CONSUMER_ROWS``, ``BOUNDARY_PROBES``, checked by
``wyrd_boundary_problems()``); the boundary doc
``docs/architecture/WYRD_BOUNDARY.md`` is filed
verbatim as the module's render.

Pinned here:

  * the measured start: no WYRD interface code in
    the shipping tree (0 interface files, 0
    live-client tokens, 0 Heimdall mentions under
    src/) — the doc files the boundary and the
    interface shapes, never a live integration,
  * the ownership split: 12 rows — 6 ThoughtForge
    (cognition turn, knowledge store, bond layer,
    persona compilation, truth enforcement, the
    consistency measurement) / 3 WYRD (canonical
    world state, its change log, entity identity
    in the world) / 3 neither (embodiment, canon
    data, live behaviour proof),
  * the interface: 5 filed shapes — 2 in
    production (bond evidence link, cognition
    turn), 2 fixture grade (TF-132's replay +
    judgement over a synthetic world), 1 not
    built (the bond excerpt bridge, TF-146's),
  * the consumers: Second Life / Heimdall and
    Hermes, both named-consumer-not-integrated,
  * the evidence neighbours re-measured live:
    TF-132's panel PASS (18 / 9 / 8 / 1), TF-143's
    bond parity excerpt at 9 lines, the TF-139 /
    TF-143 / TF-144 checkers clean,
  * the docket: TF-145 files NO D-WING-1 share in
    the doc — the docket file itself is untouched
    by this slice (its TF-144 narrowing stands as
    the last one),
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live model, no
live WYRD, never claimed.
"""

from __future__ import annotations

from pathlib import Path

from thoughtforge.wyrd_boundary import (
    BOUNDARY_EXPECTED,
    BOUNDARY_PROBES,
    CONSUMER_ROWS,
    INTERFACE_ROWS,
    INTERFACE_STATUSES,
    OWNERS,
    OWNERSHIP_ROWS,
    WYRD_BOUNDARY_DOC,
    boundary_state_problems,
    consumer_row_problems,
    interface_row_problems,
    ownership_row_problems,
    render_wyrd_boundary_markdown,
    run_boundary_probes,
    run_boundary_state,
    wyrd_boundary_problems,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ── The boundary, filed ──────────────────────────────────────────────────────

class TestBoundaryFiled:
    def test_problems_empty(self) -> None:
        assert wyrd_boundary_problems() == []

    def test_twelve_ownership_rows_in_filed_order(self) -> None:
        assert [r["id"] for r in OWNERSHIP_ROWS] == [
            "cognition_turn", "knowledge_store", "bond_layer",
            "persona_compilation", "truth_enforcement",
            "consistency_measurement", "canonical_world_state",
            "world_change_log", "entity_identity_in_world",
            "embodiment_and_transport", "canon_data",
            "live_behaviour_proof"]
        for row in OWNERSHIP_ROWS:
            assert ownership_row_problems(row) == []

    def test_ownership_split_is_six_three_three(self) -> None:
        assert OWNERS == ("thoughtforge", "wyrd", "neither")
        owners = [r["owner"] for r in OWNERSHIP_ROWS]
        assert owners.count("thoughtforge") == 6
        assert owners.count("wyrd") == 3
        assert owners.count("neither") == 3

    def test_wyrd_rows_cite_no_shipping_code(self) -> None:
        # No WYRD code stands in this repo, so a WYRD
        # row's evidence is the roadmap + filed
        # shapes — never a src/ code path presented
        # as WYRD's own.
        for row in OWNERSHIP_ROWS:
            if row["owner"] == "wyrd":
                assert "benchmarks/wyrd_panel.py" in row["evidence"] \
                    or "ROADMAP" in row["evidence"] \
                    or "evidence-link" in row["evidence"]

    def test_thoughtforge_rows_cite_real_estate(self) -> None:
        for rel in ("src/thoughtforge/cognition/core.py",
                    "src/thoughtforge/knowledge/bond_graph.py",
                    "src/thoughtforge/cognition/persona_compiler.py",
                    "src/thoughtforge/wing_retirements.py",
                    "benchmarks/wyrd_panel.py"):
            assert (REPO_ROOT / rel).exists(), rel

    def test_live_proof_is_owned_by_neither(self) -> None:
        row = [r for r in OWNERSHIP_ROWS
               if r["id"] == "live_behaviour_proof"][0]
        assert row["owner"] == "neither"
        assert "NOT PROVEN" in row["meaning"]


# ── The interface ────────────────────────────────────────────────────────────

class TestInterface:
    def test_five_interface_rows_in_filed_order(self) -> None:
        assert [r["id"] for r in INTERFACE_ROWS] == [
            "world_state_replay", "consistency_judgement",
            "bond_evidence_link", "bond_excerpt_bridge",
            "cognition_turn"]
        for row in INTERFACE_ROWS:
            assert interface_row_problems(row) == []

    def test_status_vocabulary_carries_no_live_status(self) -> None:
        assert INTERFACE_STATUSES == (
            "fixture-stand-in-only", "exists-as-panel-fixture",
            "exists-in-production", "not-built-tf146")
        assert not any("live" in s for s in INTERFACE_STATUSES)

    def test_status_split_is_two_two_one(self) -> None:
        statuses = [r["status"] for r in INTERFACE_ROWS]
        assert statuses.count("exists-in-production") == 2
        assert statuses.count("fixture-stand-in-only") == 1
        assert statuses.count("exists-as-panel-fixture") == 1
        assert statuses.count("not-built-tf146") == 1

    def test_the_bridge_is_tf146s_and_not_built(self) -> None:
        row = [r for r in INTERFACE_ROWS
               if r["id"] == "bond_excerpt_bridge"][0]
        assert row["status"] == "not-built-tf146"
        assert "TF-146" in row["shape"]
        assert "excerpt" in row["shape"]

    def test_production_rows_name_real_surfaces(self) -> None:
        link = [r for r in INTERFACE_ROWS
                if r["id"] == "bond_evidence_link"][0]
        turn = [r for r in INTERFACE_ROWS
                if r["id"] == "cognition_turn"][0]
        assert "supporting_record_ids" in link["shape"]
        assert "think()" in turn["shape"]


# ── The consumers ────────────────────────────────────────────────────────────

class TestConsumers:
    def test_two_consumers_both_not_integrated(self) -> None:
        assert [r["id"] for r in CONSUMER_ROWS] == [
            "second_life_heimdall", "hermes"]
        for row in CONSUMER_ROWS:
            assert consumer_row_problems(row) == []
            assert row["status"] == "named-consumer-not-integrated"

    def test_consumer_evidence_names_their_roadmap_homes(self) -> None:
        sl, hermes = CONSUMER_ROWS
        assert "TF-145" in sl["evidence"]
        assert "TF-158" in hermes["evidence"]


# ── The measured state ───────────────────────────────────────────────────────

class TestMeasuredState:
    def test_boundary_state_matches_its_filing(self) -> None:
        state = run_boundary_state()
        assert state == BOUNDARY_EXPECTED
        assert boundary_state_problems(state, BOUNDARY_EXPECTED) == []

    def test_no_live_wyrd_interface_in_the_tree(self) -> None:
        state = run_boundary_state()
        assert state["live_wyrd_interface_files"] == 0
        assert state["live_client_tokens"] == 0
        assert state["heimdall_mentions_in_src"] == 0

    def test_panel_evidence_remeasured(self) -> None:
        state = run_boundary_state()
        assert (state["wyrd_panel_cases"],
                state["wyrd_panel_consistent"],
                state["wyrd_panel_contradicted"],
                state["wyrd_panel_honest"]) == (18, 9, 8, 1)
        assert state["wyrd_panel_verdict"] == "PASS"
        assert state["bond_excerpt_lines"] == 9

    def test_evidence_neighbour_checkers_still_clean(self) -> None:
        from benchmarks.wyrd_panel import wyrd_panel_problems
        from thoughtforge.knowledge.bond_graph import (
            bond_graph_problems,
        )
        from thoughtforge.knowledge.bond_graph_verdict import (
            bond_graph_verdict_problems,
        )
        from thoughtforge.wing_retirements import (
            wing_retirements_problems,
        )

        assert wyrd_panel_problems() == []
        assert bond_graph_problems() == []
        assert bond_graph_verdict_problems() == []
        assert wing_retirements_problems() == []


# ── Probes + validators + doc ────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_boundary_probes()
        assert len(probes) == len(BOUNDARY_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert ownership_row_problems(None)
        assert interface_row_problems(None)
        assert consumer_row_problems(None)
        assert boundary_state_problems(None, None)

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / WYRD_BOUNDARY_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_wyrd_boundary_markdown()

    def test_doc_files_the_honest_limits(self) -> None:
        doc = _read(WYRD_BOUNDARY_DOC)
        assert "no WYRD interface code exists" in doc
        assert "NOT PROVEN" in doc
        assert "TF-146" in doc
        assert "Second Life / Heimdall" in doc
        assert "Hermes" in doc

    def test_doc_files_no_docket_share(self) -> None:
        doc = _read(WYRD_BOUNDARY_DOC)
        assert "no share" in doc
        assert "totals stand unchanged" in doc

    def test_docket_file_untouched_by_this_slice(self) -> None:
        # The last narrowing filed in the docket is
        # TF-144's; TF-145 adds none (its no-share
        # ruling is filed in the boundary doc).
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        assert "Narrowed by TF-144" in docket
        assert "Narrowed by TF-145" not in docket
