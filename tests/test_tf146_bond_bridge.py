"""
TF-146 — The bond-graph bridge (if adopted at
TF-139): bonds as WYRD-consumable data.

The antecedent is TRUE: TF-139's verdict was
ADOPT_AS_LAYER and TF-143 landed the production
bond graph, so the bridge is live work. The
proof lives as verifiable data in
``src/thoughtforge/knowledge/bond_bridge.py``
(``BRIDGE_ROWS``, ``RESERVATIONS``,
``BRIDGE_EXPECTED``, ``BOND_BRIDGE_PROBES``,
checked by ``bond_bridge_problems()``); the
write-up ``docs/architecture/BOND_BRIDGE.md``
is filed verbatim as the module's render.

Pinned here:

  * the antecedent + the source: TF-139 files
    ADOPT_AS_LAYER, TF-145's boundary row files
    the bridge not-built-tf146 naming TF-146,
    and the export is built FROM the production
    BondGraph — its values are the parity
    filing's, unmoved,
  * the export shape: versioned format, ids,
    the full ten-dimension vector, the scars,
    full vow/hurt records bound to the bond,
    the evidence record links (ids only), the
    derived indices and the excerpt lines,
    verbatim,
  * the round trip: the export JSON-round-trips
    unchanged, a second build differs only in
    its chronology stamps, and the pure
    validator recomputes the indices from the
    exported vector + scars,
  * the reservations enforced: no world state,
    no canon data, no store facts in any
    export — a forbidden key at any depth is
    named,
  * TF-145 left as filed (the estate's
    historical-gate precedent: TF-143 left
    TF-139's verdict, TF-144 left TF-140/142's)
    and its live measurements still true with
    the bridge landed (checker clean),
  * the docket: TF-146 files NO D-WING-1 share
    — the entry's text assigns no bridge share,
    so the docket file is untouched by this
    slice,
  * the probes: 7/7 against the pure validators,
  * the doc matches the render verbatim.

All fixtures synthetic (TF-017); no canon.
Proof grade: FIXTURE/STATIC — no live WYRD
consumer, never claimed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from thoughtforge.knowledge.bond_bridge import (
    BOND_BRIDGE_DOC,
    BOND_BRIDGE_FORMAT,
    BOND_BRIDGE_PROBES,
    BRIDGE_EXPECTED,
    BRIDGE_ROWS,
    FORBIDDEN_KEYS,
    RESERVATIONS,
    SOURCE_FORMAT,
    VECTOR_DIMS,
    bond_bridge_problems,
    bridge_row_problems,
    export_bond,
    export_graph,
    export_problems,
    fixture_graph,
    render_bond_bridge_markdown,
    reservation_problems,
    run_bond_bridge_probes,
    run_bridge_fixture,
)
from thoughtforge.knowledge.bond_graph import (
    PARITY_BOND_ID,
    PARITY_EXPECTED_EXCERPT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _export() -> dict:
    return run_bridge_fixture()["export"]


# ── The antecedent + the bridge, filed ───────────────────────────────────────

class TestBridgeFiled:
    def test_problems_empty(self) -> None:
        assert bond_bridge_problems() == []

    def test_antecedent_is_adopt_as_layer(self) -> None:
        from thoughtforge.knowledge.bond_graph_verdict import VERDICT

        assert VERDICT.get("verdict") == "ADOPT_AS_LAYER"

    def test_nine_sections_four_reservations(self) -> None:
        assert [r["id"] for r in BRIDGE_ROWS] == [
            "identity", "vector", "indices", "scars", "vows",
            "hurts", "evidence", "excerpt", "context"]
        for row in BRIDGE_ROWS:
            assert bridge_row_problems(row) == []
        assert [r["id"] for r in RESERVATIONS] == [
            "no_world_state", "no_canon_data", "no_store_facts",
            "no_live_claim"]
        for row in RESERVATIONS:
            assert reservation_problems(row) == []

    def test_format_is_versioned_and_names_its_source(self) -> None:
        export = _export()
        assert export["format"] == BOND_BRIDGE_FORMAT == \
            "thoughtforge-bond-bridge/1"
        assert export["source_format"] == SOURCE_FORMAT == \
            "thoughtforge-bond-graph/1"


# ── The export carries the filing ────────────────────────────────────────────

class TestExportShape:
    def test_fixture_metrics_match_the_filing(self) -> None:
        fixture = run_bridge_fixture()
        assert fixture["metrics"] == BRIDGE_EXPECTED
        assert export_problems(fixture["export"]) == []

    def test_ids_vector_and_indices_are_the_parity_values(self) -> None:
        export = _export()
        assert export["bond_id"] == PARITY_BOND_ID
        assert export["entity_a"] == "user:emberwick-keeper"
        assert export["entity_b"] == "persona:emberwick-lantern"
        assert export["domain"] == "companion"
        assert export["status"] == "repairing"
        assert tuple(export["vector"]) == VECTOR_DIMS
        assert len(export["vector"]) == 10
        assert export["indices"] == {
            "closeness": 0.4375, "sacred": 0.5333, "rupture": 0.2083}

    def test_scars_vows_hurts_and_evidence_links(self) -> None:
        export = _export()
        assert export["scars"]["repair_debt"] == 0.35
        assert export["scars"]["unresolved_hurts"] == [
            "hurt:emberwick-001"]
        vow = export["vows"][0]
        assert vow["vow_id"] == "vow:emberwick-001"
        assert vow["bond_id"] == PARITY_BOND_ID
        assert vow["created_from_record_id"] == \
            "rec:synthetic-emberwick-000"
        assert vow["state"] == "pending"
        hurt = export["hurts"][0]
        assert hurt["hurt_id"] == "hurt:emberwick-001"
        assert hurt["severity"] == "high"
        assert hurt["source_event_id"] == "evt:synthetic-emberwick-001"
        # The layer link: record IDS only.
        assert export["evidence"] == {
            "supporting_record_ids": ["rec:synthetic-emberwick-001"],
            "contradiction_record_ids": []}

    def test_excerpt_is_the_production_excerpt_verbatim(self) -> None:
        export = _export()
        graph = fixture_graph()
        assert export["excerpt"] == graph.excerpt(PARITY_BOND_ID)
        assert tuple(export["excerpt"]) == PARITY_EXPECTED_EXCERPT
        assert len(export["excerpt"]) == 9

    def test_export_comes_from_the_live_graph_not_a_copy(self) -> None:
        # Mutating the production graph moves the
        # export — the bridge reads the graph, it
        # does not snapshot a filing.
        graph = fixture_graph()
        graph.apply_event(
            PARITY_BOND_ID, warmth_delta=-0.2,
            source_record_id="rec:synthetic-emberwick-002")
        export = export_bond(graph, PARITY_BOND_ID)
        assert export["evidence"]["supporting_record_ids"] == [
            "rec:synthetic-emberwick-001",
            "rec:synthetic-emberwick-002"]
        assert export_problems(export) == []

    def test_export_graph_exports_every_bond(self) -> None:
        bundle = export_graph(fixture_graph())
        assert bundle["format"] == BOND_BRIDGE_FORMAT
        assert [b["bond_id"] for b in bundle["bonds"]] == [
            PARITY_BOND_ID]
        assert export_problems(bundle["bonds"][0]) == []


# ── The round trip ───────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_export_json_round_trips_unchanged(self) -> None:
        export = _export()
        assert json.loads(json.dumps(export)) == export

    def test_second_build_differs_only_in_chronology(self) -> None:
        first, second = _export(), _export()
        assert {k: v for k, v in first.items()
                if k != "chronology"} == \
            {k: v for k, v in second.items() if k != "chronology"}
        for stamp in second["chronology"].values():
            assert stamp is None or isinstance(stamp, str)

    def test_validator_recomputes_indices_from_exported_data(self) -> None:
        export = _export()
        export["indices"]["rupture"] = 0.5
        assert any("rupture" in p for p in export_problems(export))


# ── The reservations, enforced ───────────────────────────────────────────────

class TestReservations:
    def test_no_forbidden_key_in_the_fixture_export(self) -> None:
        def keys(data):
            if isinstance(data, dict):
                for k, v in data.items():
                    yield k
                    yield from keys(v)
            elif isinstance(data, list):
                for v in data:
                    yield from keys(v)

        assert not (set(keys(_export())) & set(FORBIDDEN_KEYS))

    def test_forbidden_key_at_any_depth_is_named(self) -> None:
        export = _export()
        export["vows"][0]["canon"] = "invented"
        assert any("canon" in p for p in export_problems(export))

    def test_evidence_carries_ids_never_records(self) -> None:
        export = _export()
        for ids in export["evidence"].values():
            assert all(isinstance(i, str) for i in ids)
        assert "store_records" not in json.dumps(export)


# ── TF-145 + the docket, handled in the open ─────────────────────────────────

class TestBoundaryAndDocket:
    def test_tf145_row_stands_as_filed(self) -> None:
        from thoughtforge.wyrd_boundary import INTERFACE_ROWS

        row = [r for r in INTERFACE_ROWS
               if r["id"] == "bond_excerpt_bridge"][0]
        assert row["status"] == "not-built-tf146"
        assert "TF-146" in row["shape"]

    def test_tf145_checker_still_clean_with_the_bridge_landed(self) -> None:
        # The bridge was shaped to leave TF-145's
        # live measurements true: no WYRD-named
        # file, no live-client token, no consumer
        # name under src/.
        from thoughtforge.wyrd_boundary import wyrd_boundary_problems

        assert wyrd_boundary_problems() == []

    def test_bridge_module_carries_no_consumer_name(self) -> None:
        text = _read("src/thoughtforge/knowledge/bond_bridge.py")
        assert "Heimdall" not in text and "heimdall" not in text
        assert "WyrdClient" not in text and "wyrd_client" not in text

    def test_docket_file_untouched_by_this_slice(self) -> None:
        # D-WING-1's text assigns no bridge share
        # (verdicts / TF-143 / TF-144 / TF-148 /
        # TF-149 only), so TF-146 adds no narrowing
        # — the last narrowing stands TF-144's.
        docket = _read("docs/audit/TF-016_BUG_AND_GAP_DOCKET.md")
        assert "Narrowed by TF-144" in docket
        assert "Narrowed by TF-146" not in docket
        assert "Narrowed by TF-145" not in docket

    def test_neighbour_checkers_still_clean(self) -> None:
        from benchmarks.wyrd_panel import wyrd_panel_problems
        from thoughtforge.knowledge.bond_graph import bond_graph_problems
        from thoughtforge.knowledge.bond_graph_verdict import (
            bond_graph_verdict_problems,
        )

        assert wyrd_panel_problems() == []
        assert bond_graph_problems() == []
        assert bond_graph_verdict_problems() == []


# ── Probes + validators + doc ────────────────────────────────────────────────

class TestProbesAndDoc:
    def test_probes_all_pass(self) -> None:
        probes = run_bond_bridge_probes()
        assert len(probes) == len(BOND_BRIDGE_PROBES) == 7
        assert all(p["passed"] for p in probes)

    def test_validators_are_empty_safe(self) -> None:
        assert export_problems(None)
        assert export_problems({})
        assert bridge_row_problems(None)
        assert reservation_problems(None)

    def test_each_probe_shape_is_named_by_the_validator(self) -> None:
        export = _export()
        broken = copy.deepcopy(export)
        broken["bond_id"] = ""
        assert export_problems(broken)
        broken = copy.deepcopy(export)
        broken["excerpt"][1] = "status=active"
        assert export_problems(broken)
        broken = copy.deepcopy(export)
        broken["vector"]["trust"] = 1.4
        assert export_problems(broken)

    def test_doc_matches_render(self) -> None:
        doc = REPO_ROOT / BOND_BRIDGE_DOC
        assert doc.read_text(encoding="utf-8") == \
            render_bond_bridge_markdown()

    def test_doc_files_the_honest_limits(self) -> None:
        doc = _read(BOND_BRIDGE_DOC)
        assert "ADOPT_AS_LAYER" in doc
        assert "NOT PROVEN" in doc
        assert "stands as filed" in doc
        assert "no share" in doc
        assert "TF-017" in doc
